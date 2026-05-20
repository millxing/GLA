#!/usr/bin/env python3
"""admin/cli.py

Standalone admin CLI for the NBA_Data repository.

Responsibilities:
- Ensure the NBA_Data repo exists locally (clone if missing)
- Update/download season CSVs in nested NBA_Data family folders in *game-level* schema
  (one row per game with *_home and *_road columns)
- Generate/update contribution JSONs and interpretation artifacts
- Show git status and optionally commit+push changes from NBA_Data repo

Critical behavior:
- Enforces the exact 50-column schema used by NBA_Data game logs.
- Enforces dtypes to match historical files (ints stay ints; pct columns are floats;
  neutral_site is bool; game_type is snake_case strings).
- Preserves any existing rows (and their game_type values like nba_cup_group)
  by ONLY appending brand-new game_id values when updating.

Usage examples (from backend directory):
  python admin/cli.py update-data --season 2025-26
  python admin/cli.py download-data --start 2020-21 --end 2024-25
  python admin/cli.py commit-and-push --message "Update data"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import httpx
import json
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

import pandas as pd

# Data pulls
from nba_api.stats.endpoints import leaguegamelog, boxscoresummaryv3, boxscoreadvancedv3

# Import calculation functions for interpretation generation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.calculations import compute_four_factors, compute_game_ratings
from services.llm import generate_interpretation_sync, LLM_MODELS
from config import (
    DEFAULT_NBA_DATA_REPO_DIR,
    PBP_WINPROB_BASE_ROOT,
    PBP_WINPROB_MODELS_ROOT,
    build_data_filename,
    get_canonical_data_file_path,
    get_current_season,
    iter_data_family_files,
    resolve_data_file_path,
)

PARQUET_BRIDGE_PYTHON = sys.executable or "python3"


NBA_DATA_REPO_URL = "https://github.com/millxing/NBA_Data"

# Default: use the user's canonical NBA_Data folder.
# This keeps GLA_Admin fully separate from any NBA/ NBA_alpha folders.
DEFAULT_REPO_DIR = DEFAULT_NBA_DATA_REPO_DIR
DEFAULT_SHUF_DATASETS_DIR = Path(__file__).resolve().parents[2] / "shuf_datasets"
PBP_ROOT_DIRNAME = "PBPdata"
PBP_MANIFEST_FILENAME = "manifest.csv"
PBP_MANIFEST_COLUMNS = [
    "source",
    "season",
    "season_type",
    "file_path",
    "row_count",
    "game_count",
    "sha256",
    "updated_at",
]
PBP_FETCH_TIMEOUT_SECONDS = 12.0
PBP_FETCH_RETRIES = 2
PBP_FETCH_MAX_WORKERS = 6
PBP_FETCH_BACKOFF_BASE_SECONDS = 0.5
PBPV3_CANONICAL_COLUMNS = [
    "actionNumber",
    "clock",
    "period",
    "teamId",
    "teamTricode",
    "personId",
    "playerName",
    "playerNameI",
    "xLegacy",
    "yLegacy",
    "shotDistance",
    "shotResult",
    "isFieldGoal",
    "scoreHome",
    "scoreAway",
    "pointsTotal",
    "location",
    "description",
    "actionType",
    "subType",
    "videoAvailable",
    "shotValue",
    "actionId",
    "gameId",
]


# ---- Canonical NBA_Data game-log schema + dtypes (matches *_correct.csv) ----
EXPECTED_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "game_type",
    "neutral_site",
    "team_id_home",
    "team_abbreviation_home",
    "team_name_home",
    "team_id_road",
    "team_abbreviation_road",
    "team_name_road",
    "pts_home",
    "pts_road",
    "wl_home",
    "fgm_home",
    "fga_home",
    "fg_pct_home",
    "fg3m_home",
    "fg3a_home",
    "fg3_pct_home",
    "ftm_home",
    "fta_home",
    "ft_pct_home",
    "oreb_home",
    "dreb_home",
    "reb_home",
    "ast_home",
    "stl_home",
    "blk_home",
    "tov_home",
    "pf_home",
    "plus_minus_home",
    "fgm_road",
    "fga_road",
    "fg_pct_road",
    "fg3m_road",
    "fg3a_road",
    "fg3_pct_road",
    "ftm_road",
    "fta_road",
    "ft_pct_road",
    "oreb_road",
    "dreb_road",
    "reb_road",
    "ast_road",
    "stl_road",
    "blk_road",
    "tov_road",
    "pf_road",
    "plus_minus_road",
]

INT_COLS = [
    "team_id_home",
    "team_id_road",
    "pts_home",
    "pts_road",
    "fgm_home",
    "fga_home",
    "fg3m_home",
    "fg3a_home",
    "ftm_home",
    "fta_home",
    "oreb_home",
    "dreb_home",
    "reb_home",
    "ast_home",
    "stl_home",
    "blk_home",
    "tov_home",
    "pf_home",
    "plus_minus_home",
    "fgm_road",
    "fga_road",
    "fg3m_road",
    "fg3a_road",
    "ftm_road",
    "fta_road",
    "oreb_road",
    "dreb_road",
    "reb_road",
    "ast_road",
    "stl_road",
    "blk_road",
    "tov_road",
    "pf_road",
    "plus_minus_road",
]

# Columns that should be forced to string/object dtype in season CSVs
ID_STR_COLS = ["game_id"]

FLOAT_COLS = [
    "fg_pct_home",
    "fg3_pct_home",
    "ft_pct_home",
    "fg_pct_road",
    "fg3_pct_road",
    "ft_pct_road",
]

OBJ_COLS = [
    "game_date",
    "season",
    "game_type",
    "team_abbreviation_home",
    "team_name_home",
    "team_abbreviation_road",
    "team_name_road",
    "wl_home",
]


# Mapping from NBA_Data stat prefixes -> nba_api team log columns
STAT_MAP = {
    "pts": "PTS",
    "fgm": "FGM",
    "fga": "FGA",
    "fg_pct": "FG_PCT",
    "fg3m": "FG3M",
    "fg3a": "FG3A",
    "fg3_pct": "FG3_PCT",
    "ftm": "FTM",
    "fta": "FTA",
    "ft_pct": "FT_PCT",
    "oreb": "OREB",
    "dreb": "DREB",
    "reb": "REB",
    "ast": "AST",
    "stl": "STL",
    "blk": "BLK",
    "tov": "TOV",
    "pf": "PF",
    "plus_minus": "PLUS_MINUS",
}


# ---- NBA Cup knockout dates file ----
# Expected CSV format: date,game_type (e.g., "2024-12-14,nba_cup_semi")
NBA_CUP_DATES_FILE = Path(__file__).parent.parent.parent / "NBACup_knockout_dates.csv"

# Cancelled/invalid games to exclude (game was scheduled but never played)
# IND @ BOS on 2013-04-16 was cancelled due to Boston Marathon bombing, never rescheduled
CANCELLED_GAME_IDS = {
    "0021201214",  # 2012-13 IND @ BOS cancelled 4/16/2013
}

# Fallback data for games missing from boxscoreadvancedv3 endpoint
# When the API fails to return data for these games, use this hardcoded data instead
ADVANCED_FALLBACK_DATA = {
    # 2003-04 WAS @ NOP 2/18/2004 - not in boxscoreadvancedv3
    "0020300778": {
        "game_id": "0020300778",
        "game_date": "2004-02-18",
        "season": "2003-04",
        "team_id_home": 1610612740,
        "team_abbreviation_home": "NOH",
        "minutes_home": 240,
        "possessions_home": 97.0,
        "team_id_road": 1610612764,
        "team_abbreviation_road": "WAS",
        "minutes_road": 240,
        "possessions_road": 97.0,
    },
}


def _load_nba_cup_dates() -> Dict[str, str]:
    """Load NBA Cup knockout dates from CSV file.

    Returns a dict mapping date strings (YYYY-MM-DD) to game_type
    (e.g., "nba_cup_semi" or "nba_cup_final").
    Returns empty dict if file doesn't exist.
    """
    if not NBA_CUP_DATES_FILE.exists():
        return {}

    try:
        df = pd.read_csv(NBA_CUP_DATES_FILE, dtype=str)
        if "date" not in df.columns or "game_type" not in df.columns:
            print(f"[warning] NBA Cup dates file missing required columns (date, game_type)")
            return {}

        # Normalize date format to YYYY-MM-DD
        date_map = {}
        for _, row in df.iterrows():
            date_str = str(row["date"]).strip()
            game_type = str(row["game_type"]).strip()
            # Try to parse and normalize the date
            try:
                parsed = pd.to_datetime(date_str)
                date_map[parsed.strftime("%Y-%m-%d")] = game_type
            except Exception:
                print(f"[warning] Could not parse date: {date_str}")
        return date_map
    except Exception as e:
        print(f"[warning] Could not load NBA Cup dates file: {e}")
        return {}


def _apply_nba_cup_overrides(df: pd.DataFrame, cup_dates: Dict[str, str]) -> pd.DataFrame:
    """Override game_type for games that fall on NBA Cup knockout dates.

    Only applies to games currently marked as 'regular_season'.
    """
    if not cup_dates or df.empty:
        return df

    df = df.copy()
    for idx, row in df.iterrows():
        if row.get("game_type") == "regular_season":
            game_date = str(row.get("game_date", "")).strip()
            if game_date in cup_dates:
                df.at[idx, "game_type"] = cup_dates[game_date]

    return df


# ---- LineScore schema (BoxScoreSummaryV3) ----
# Uses pts_ot_total (calculated from score - Q1-Q4) instead of individual OT periods
LINESCORE_COLUMNS = [
    "game_id", "game_date", "season",
    "team_id_home", "team_abbreviation_home", "team_name_home",
    "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home",
    "pts_ot_total_home", "pts_home",
    "team_id_road", "team_abbreviation_road", "team_name_road",
    "pts_qtr1_road", "pts_qtr2_road", "pts_qtr3_road", "pts_qtr4_road",
    "pts_ot_total_road", "pts_road",
]

LINESCORE_INT_COLS = [
    "team_id_home", "team_id_road",
    "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home",
    "pts_ot_total_home", "pts_home",
    "pts_qtr1_road", "pts_qtr2_road", "pts_qtr3_road", "pts_qtr4_road",
    "pts_ot_total_road", "pts_road",
]

# ---- Advanced stats schema (BoxScoreAdvancedV3) ----
# Includes minutes (for determining OT periods) and possessions
ADVANCED_COLUMNS = [
    "game_id", "game_date", "season",
    "team_id_home", "team_abbreviation_home", "minutes_home", "possessions_home",
    "team_id_road", "team_abbreviation_road", "minutes_road", "possessions_road",
]

ADVANCED_INT_COLS = ["team_id_home", "team_id_road", "minutes_home", "minutes_road"]

ADVANCED_FLOAT_COLS = ["possessions_home", "possessions_road"]


# ------------------------- timeout wrapper -------------------------

T = TypeVar("T")

# Hard timeout for individual API calls (seconds)
# If an API call hangs longer than this, it will be abandoned
API_HARD_TIMEOUT = 90

# Maximum games to fetch before auto-restart (to avoid rate limiting issues)
BATCH_RESTART_SIZE = 300


def _call_with_timeout(func: Callable[[], T], timeout: int = API_HARD_TIMEOUT) -> Optional[T]:
    """Execute a function with a hard timeout using a thread pool.

    If the function doesn't complete within the timeout, returns None.
    This is more reliable than HTTP timeouts for detecting hung connections.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            print(f"[TIMEOUT after {timeout}s]", end=" ", flush=True)
            return None
        except Exception as e:
            # Let the caller handle other exceptions
            raise e


# ------------------------- git + repo helpers -------------------------

def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=check,
    )


def ensure_data_repo(repo_dir: Path) -> Path:
    repo_dir = repo_dir.resolve()

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[repo] Cloning {NBA_DATA_REPO_URL} -> {repo_dir}")
        subprocess.run(["git", "clone", NBA_DATA_REPO_URL, str(repo_dir)], check=True)
    else:
        if not (repo_dir / ".git").exists():
            raise RuntimeError(f"Expected {repo_dir} to be a git repository, but .git was not found.")

    return repo_dir


# ------------------------- schema utilities -------------------------

def _season_to_filename(season: str) -> str:
    return build_data_filename("team_game_logs", season)


def _linescore_filename(season: str) -> str:
    return build_data_filename("linescores", season)


def _advanced_filename(season: str) -> str:
    return build_data_filename("box_score_advanced", season)


def _canonical_repo_data_path(repo_dir: Path, filename: str) -> Path:
    return get_canonical_data_file_path(filename, repo_dir=repo_dir)


def _resolve_repo_data_path(repo_dir: Path, filename: str) -> Path:
    return resolve_data_file_path(filename, repo_dir=repo_dir)


def _snake_case(x: object) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    # Common normalization for game_type consistency
    if s == "regularseason":
        return "regular_season"
    if s == "playoff":
        return "playoffs"
    if s == "playin":
        return "play_in"
    return s


def _normalize_game_id(game_id: Any) -> str:
    """Normalize game_id to a 10-digit string for reliable cross-file joins."""
    if pd.isna(game_id):
        return ""

    gid = str(game_id).strip()
    if gid.endswith(".0"):
        gid = gid[:-2]

    digits = "".join(ch for ch in gid if ch.isdigit())
    if not digits:
        return gid
    return digits.zfill(10)


def _load_existing_season_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    # Force game_id to load as string (matches canonical NBA_Data behavior)
    df = pd.read_csv(path, dtype={"game_id": "string"})

    # Normalize either schema's date column to datetime
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    return df


def _normalize_game_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """Force exact NBA_Data game-log schema + dtypes.

    - Drops extra columns
    - Adds missing expected columns
    - Forces ID columns to int64 (no .0)
    - Forces neutral_site to bool
    - Forces pct columns to float64
    - Forces game_date to YYYY-MM-DD string
    """

    d = df.copy()

    # Add missing columns and drop extras
    for c in EXPECTED_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[EXPECTED_COLUMNS]

    # game_type normalization (keep as string)
    d["game_type"] = d["game_type"].map(lambda v: _snake_case(v) if pd.notna(v) else v)

    # Date normalization to YYYY-MM-DD string
    gd = pd.to_datetime(d["game_date"], errors="coerce")
    d["game_date"] = gd.dt.date.astype("string")
    d.loc[gd.isna(), "game_date"] = pd.NA

    # neutral_site -> bool
    def _to_bool(v: object) -> bool:
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        s = str(v).strip().lower()
        return s in ("true", "1", "t", "yes", "y")

    d["neutral_site"] = d["neutral_site"].map(_to_bool).astype(bool)

    # --- ID normalization ---
    # Keep game_id as a string column. Clean up values like "22500002.0" -> "22500002".
    # IMPORTANT: CSVs do not preserve dtypes, so we enforce string on load and here.
    d["game_id"] = d["game_id"].astype("string")
    d["game_id"] = d["game_id"].str.strip()
    d["game_id"] = d["game_id"].str.replace(r"\.0$", "", regex=True)
    d.loc[d["game_id"].isin(["", "<NA>", "nan", "NaN", "None"]), "game_id"] = pd.NA
    # Force canonical 10-char game ids with leading zeros (e.g., 0022500002)
    d["game_id"] = d["game_id"].map(lambda v: v.zfill(10) if isinstance(v, str) and v.isdigit() else v)

    # Numeric coercions
    for c in INT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in FLOAT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Drop invalid rows
    d = d.dropna(subset=["game_id"])

    # De-dupe by game_id (game-level identity)
    d = d.drop_duplicates(subset=["game_id"], keep="first")

    # Remove known cancelled games
    cancelled_mask = d["game_id"].isin(CANCELLED_GAME_IDS)
    if cancelled_mask.any():
        print(f"[data] Filtering out {cancelled_mask.sum()} cancelled game(s)")
        d = d[~cancelled_mask].copy()

    # Sort by game_date
    if "game_date" in d.columns:
        d = d.sort_values("game_date")

    # Cast ints to true int64 (matching historical files)
    # (safe now because we dropped NaN game_id rows)
    for c in INT_COLS:
        if d[c].isna().any():
            # Should not happen for required columns, but keep as pandas nullable int
            d[c] = d[c].astype("Int64")
        else:
            d[c] = d[c].astype("int64")

    # Ensure game_id is a string dtype
    d["game_id"] = d["game_id"].astype("string")

    # pct columns float64
    for c in FLOAT_COLS:
        d[c] = d[c].astype("float64")

    # object columns as strings (or keep NaN)
    for c in OBJ_COLS:
        if c in d.columns:
            # game_date already string; others keep as object
            pass

    return d


# ------------------------- nba_api pulls + conversion -------------------------

def _fetch_season_team_game_logs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Fetch team game logs for a given season using nba_api.

    Args:
        season: NBA season string (e.g., "2024-25")
        season_type: One of "Regular Season", "Playoffs", "PlayIn", "Pre Season", "All Star"

    Returns one row per team per game (each NBA game appears twice).
    """
    resp = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
    )
    df = resp.get_data_frames()[0]

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    return df


def _teamlogs_to_gamelogs(team_df: pd.DataFrame, season: str, game_type: str = "regular_season") -> pd.DataFrame:
    """Convert nba_api team-level logs -> NBA_Data game-level rows.

    Args:
        team_df: DataFrame from LeagueGameLog
        season: NBA season string (e.g., "2024-25")
        game_type: Game type label (e.g., "regular_season", "playoffs", "play_in")

    Output uses exact EXPECTED_COLUMNS layout (later enforced by normalizer).
    """

    df = team_df.copy()

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    # Determine home/away from MATCHUP
    # Normal games: home has "vs.", away has "@"
    # Neutral site games (NBA Cup final): BOTH teams have "@"
    matchup = df.get("MATCHUP", pd.Series("", index=df.index)).astype(str)
    df["_IS_HOME"] = matchup.str.contains("vs.", na=False)
    df["_IS_AWAY"] = matchup.str.contains("@", na=False)

    out_rows: list[dict] = []

    # Ensure IDs are strings for grouping stability
    df["GAME_ID"] = df["GAME_ID"].astype(str)

    for gid, g in df.groupby("GAME_ID"):
        home = g[g["_IS_HOME"]]
        away = g[g["_IS_AWAY"]]

        is_neutral_site = False

        if len(home) == 1 and len(away) == 1:
            # Normal game with clear home/away
            h = home.iloc[0]
            a = away.iloc[0]
        elif len(home) == 0 and len(away) == 2:
            # Neutral site game - both teams marked as "@"
            # Use matchup to determine designated home: "TEAM @ OPPONENT" -> OPPONENT is home
            # Pick the team whose abbreviation appears AFTER "@" in the other team's matchup
            is_neutral_site = True
            row1, row2 = away.iloc[0], away.iloc[1]
            m1 = str(row1.get("MATCHUP", ""))
            # In "NYK @ SAS", SAS is designated as home
            if " @ " in m1:
                designated_home_abbr = m1.split(" @ ")[1].strip()
                if row1.get("TEAM_ABBREVIATION") == designated_home_abbr:
                    h, a = row1, row2
                else:
                    h, a = row2, row1
            else:
                # Fallback: alphabetically first is home
                if row1.get("TEAM_ABBREVIATION", "") < row2.get("TEAM_ABBREVIATION", ""):
                    h, a = row1, row2
                else:
                    h, a = row2, row1
        else:
            # Skip weird games rather than creating malformed rows
            continue

        # NBA_Data uses snake_case game_type labels
        row: dict = {
            "game_id": str(gid),
            "game_date": (pd.to_datetime(h.get("GAME_DATE"), errors="coerce").date().isoformat()
                          if pd.notna(h.get("GAME_DATE")) else pd.NA),
            "season": season,
            "game_type": game_type,
            "neutral_site": is_neutral_site,
            "team_id_home": int(h.get("TEAM_ID")) if pd.notna(h.get("TEAM_ID")) else pd.NA,
            "team_abbreviation_home": h.get("TEAM_ABBREVIATION"),
            "team_name_home": h.get("TEAM_NAME"),
            "team_id_road": int(a.get("TEAM_ID")) if pd.notna(a.get("TEAM_ID")) else pd.NA,
            "team_abbreviation_road": a.get("TEAM_ABBREVIATION"),
            "team_name_road": a.get("TEAM_NAME"),
            "pts_home": h.get("PTS"),
            "pts_road": a.get("PTS"),
            "wl_home": h.get("WL"),
        }

        for stat_prefix, src_col in STAT_MAP.items():
            row[f"{stat_prefix}_home"] = h.get(src_col)
            row[f"{stat_prefix}_road"] = a.get(src_col)

        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    return out


# ------------------------- boxscore fetching (linescore + advanced) -------------------------

def _fetch_linescore(game_id: str, game_date: str, season: str, home_team_id: int) -> Optional[dict]:
    """Fetch linescore data for a single game from BoxScoreSummaryV3.

    Returns a game-level row dict with home/road quarter scoring and OT total, or None on error.
    OT total is calculated as: score - (Q1 + Q2 + Q3 + Q4)
    Uses hard timeout wrapper to auto-skip stuck requests.
    """
    def _do_fetch() -> Optional[dict]:
        resp = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id, timeout=60)
        ls_df = resp.line_score.get_data_frame()

        if ls_df.empty or len(ls_df) < 2:
            return None

        # Identify home vs road using home_team_id from gamelog
        home_row = ls_df[ls_df["teamId"] == home_team_id]
        road_row = ls_df[ls_df["teamId"] != home_team_id]

        if home_row.empty or road_row.empty:
            return None

        h = home_row.iloc[0]
        r = road_row.iloc[0]

        # V3 API uses: period1Score-period4Score, score, teamId, teamTricode, teamName
        # Calculate OT total as: score - (Q1 + Q2 + Q3 + Q4)
        h_q1 = h.get("period1Score", 0) or 0
        h_q2 = h.get("period2Score", 0) or 0
        h_q3 = h.get("period3Score", 0) or 0
        h_q4 = h.get("period4Score", 0) or 0
        h_total = h.get("score", 0) or 0
        h_ot_total = h_total - (h_q1 + h_q2 + h_q3 + h_q4)

        r_q1 = r.get("period1Score", 0) or 0
        r_q2 = r.get("period2Score", 0) or 0
        r_q3 = r.get("period3Score", 0) or 0
        r_q4 = r.get("period4Score", 0) or 0
        r_total = r.get("score", 0) or 0
        r_ot_total = r_total - (r_q1 + r_q2 + r_q3 + r_q4)

        return {
            "game_id": game_id,
            "game_date": game_date,
            "season": season,
            "team_id_home": int(h.get("teamId", 0)),
            "team_abbreviation_home": h.get("teamTricode", ""),
            "team_name_home": h.get("teamName", ""),
            "pts_qtr1_home": h_q1,
            "pts_qtr2_home": h_q2,
            "pts_qtr3_home": h_q3,
            "pts_qtr4_home": h_q4,
            "pts_ot_total_home": h_ot_total,
            "pts_home": h_total,
            "team_id_road": int(r.get("teamId", 0)),
            "team_abbreviation_road": r.get("teamTricode", ""),
            "team_name_road": r.get("teamName", ""),
            "pts_qtr1_road": r_q1,
            "pts_qtr2_road": r_q2,
            "pts_qtr3_road": r_q3,
            "pts_qtr4_road": r_q4,
            "pts_ot_total_road": r_ot_total,
            "pts_road": r_total,
        }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = _call_with_timeout(_do_fetch)
            if result is not None:
                return result
            # Timeout or empty result - retry
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"[err:{e}]", end=" ", flush=True)
            return None


def _fetch_advanced_stats(game_id: str, game_date: str, season: str, home_team_id: int) -> Optional[dict]:
    """Fetch possessions for a single game from BoxScoreAdvancedV3.

    Returns a game-level row dict with home/road possessions, or None on error.
    Uses hard timeout wrapper to auto-skip stuck requests.
    """
    def _do_fetch() -> Optional[dict]:
        resp = boxscoreadvancedv3.BoxScoreAdvancedV3(
            game_id=game_id,
            start_period=0,
            end_period=0,
            start_range=0,
            end_range=28800,
            range_type=0,
            timeout=60,
        )
        team_df = resp.team_stats.get_data_frame()

        if team_df.empty or len(team_df) < 2:
            return None

        # Identify home vs road using home_team_id from gamelog
        home_row = team_df[team_df["teamId"] == home_team_id]
        road_row = team_df[team_df["teamId"] != home_team_id]

        if home_row.empty or road_row.empty:
            return None

        h = home_row.iloc[0]
        r = road_row.iloc[0]

        # Parse minutes from "290:00:00" or "290:00" format to integer
        def parse_minutes(mins_str: str) -> int:
            if not mins_str:
                return 0
            # Format is either "290:00:00" or "290:00" - take first part
            parts = str(mins_str).split(":")
            try:
                return int(parts[0])
            except (ValueError, IndexError):
                return 0

        return {
            "game_id": game_id,
            "game_date": game_date,
            "season": season,
            "team_id_home": int(h.get("teamId", 0)),
            "team_abbreviation_home": h.get("teamTricode", ""),
            "minutes_home": parse_minutes(h.get("minutes", "")),
            "possessions_home": h.get("possessions", 0.0),
            "team_id_road": int(r.get("teamId", 0)),
            "team_abbreviation_road": r.get("teamTricode", ""),
            "minutes_road": parse_minutes(r.get("minutes", "")),
            "possessions_road": r.get("possessions", 0.0),
        }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = _call_with_timeout(_do_fetch)
            if result is not None:
                return result
            # Timeout or empty result - retry
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"[err:{e}]", end=" ", flush=True)
            return None


def _advanced_row_missing_core_values(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in ("minutes_home", "minutes_road", "possessions_home", "possessions_road"):
        value = pd.to_numeric(row.get(field), errors="coerce")
        if pd.isna(value) or float(value) <= 0.0:
            missing.append(field)
    return missing


def _assert_valid_advanced_row(row: dict[str, Any], *, context: str) -> None:
    missing = _advanced_row_missing_core_values(row)
    if missing:
        game_id = _normalize_game_id(row.get("game_id"))
        details = ", ".join(f"{field}={row.get(field)!r}" for field in missing)
        raise ValueError(
            f"{context}: advanced box score missing core values for game {game_id or '<unknown>'} ({details})"
        )


def _assert_no_invalid_advanced_rows(df: pd.DataFrame, *, context: str) -> None:
    if df is None or df.empty:
        return

    invalid_mask = pd.Series(False, index=df.index)
    for field in ("minutes_home", "minutes_road", "possessions_home", "possessions_road"):
        values = pd.to_numeric(df[field], errors="coerce")
        invalid_mask = invalid_mask | values.isna() | (values <= 0.0)

    invalid_rows = df.loc[invalid_mask, [
        "game_id",
        "game_date",
        "team_abbreviation_home",
        "team_abbreviation_road",
        "minutes_home",
        "possessions_home",
        "minutes_road",
        "possessions_road",
    ]]
    if invalid_rows.empty:
        return

    sample_lines = []
    for _, row in invalid_rows.head(5).iterrows():
        sample_lines.append(
            f"{row['game_id']} {row['team_abbreviation_road']}@{row['team_abbreviation_home']} "
            f"mh={row['minutes_home']} ph={row['possessions_home']} "
            f"mr={row['minutes_road']} pr={row['possessions_road']}"
        )
    remainder = len(invalid_rows) - len(sample_lines)
    sample = "; ".join(sample_lines)
    if remainder > 0:
        sample = f"{sample}; ... and {remainder} more"
    raise ValueError(f"{context}: advanced box score file contains invalid rows: {sample}")


def _normalize_linescore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize linescore DataFrame to canonical schema and dtypes."""
    d = df.copy()

    # Add missing columns and drop extras
    for c in LINESCORE_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[LINESCORE_COLUMNS]

    # Normalize game_id
    d["game_id"] = d["game_id"].astype("string")
    d["game_id"] = d["game_id"].str.strip()
    d["game_id"] = d["game_id"].str.replace(r"\.0$", "", regex=True)
    d["game_id"] = d["game_id"].map(lambda v: v.zfill(10) if isinstance(v, str) and v.isdigit() else v)

    # Normalize game_date to YYYY-MM-DD format
    if "game_date" in d.columns:
        d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
        d["game_date"] = d["game_date"].dt.strftime("%Y-%m-%d")

    # Integer columns
    for c in LINESCORE_INT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("int64")

    # Drop rows without valid game_id
    d = d.dropna(subset=["game_id"])
    d = d.drop_duplicates(subset=["game_id"], keep="first")

    # Sort by game_date
    if "game_date" in d.columns:
        d = d.sort_values("game_date")

    return d


def _normalize_advanced_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize advanced stats DataFrame to canonical schema and dtypes."""
    d = df.copy()

    # Add missing columns and drop extras
    for c in ADVANCED_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[ADVANCED_COLUMNS]

    # Normalize game_id
    d["game_id"] = d["game_id"].astype("string")
    d["game_id"] = d["game_id"].str.strip()
    d["game_id"] = d["game_id"].str.replace(r"\.0$", "", regex=True)
    d["game_id"] = d["game_id"].map(lambda v: v.zfill(10) if isinstance(v, str) and v.isdigit() else v)

    # Normalize game_date to YYYY-MM-DD format
    if "game_date" in d.columns:
        d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
        d["game_date"] = d["game_date"].dt.strftime("%Y-%m-%d")

    # Integer columns
    for c in ADVANCED_INT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("int64")

    # Float columns
    for c in ADVANCED_FLOAT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")

    # Drop rows without valid game_id
    d = d.dropna(subset=["game_id"])
    d = d.drop_duplicates(subset=["game_id"], keep="first")

    # Sort by game_date
    if "game_date" in d.columns:
        d = d.sort_values("game_date")

    return d


def _fetch_boxscore_data(
    game_ids: list[str],
    game_info: Dict[str, tuple],
    season: str,
    linescore_path: Path,
    advanced_path: Path,
    existing_ls: Optional[pd.DataFrame],
    existing_adv: Optional[pd.DataFrame],
) -> tuple[int, int]:
    """Fetch linescore and advanced stats for multiple games with incremental saves.

    Args:
        game_ids: List of game IDs to fetch
        game_info: Dict mapping game_id -> (game_date, home_team_id)
        season: Season string (e.g. "2024-25")
        linescore_path: Path to save linescore CSV
        advanced_path: Path to save advanced CSV
        existing_ls: Existing linescore DataFrame (or None)
        existing_adv: Existing advanced DataFrame (or None)

    Returns:
        Tuple of (linescore_count, advanced_count) - number of new rows added
    """
    linescore_rows: list[dict] = []
    advanced_rows: list[dict] = []
    ls_total_added = 0
    adv_total_added = 0

    total = len(game_ids)
    save_interval = 50  # Save every 50 games

    for i, gid in enumerate(game_ids, 1):
        game_date, home_team_id = game_info.get(gid, ("", 0))

        # Show which game we're fetching
        print(f"  [{i}/{total}] Fetching {gid}...", end=" ", flush=True)

        # Fetch linescore
        ls_row = _fetch_linescore(gid, game_date, season, home_team_id)
        if ls_row:
            linescore_rows.append(ls_row)
            print("LS:OK", end=" ", flush=True)
        else:
            print("LS:FAIL", end=" ", flush=True)
        time.sleep(1.0)  # Delay to avoid rate limiting

        # Fetch advanced stats (with fallback for known missing games)
        adv_row = _fetch_advanced_stats(gid, game_date, season, home_team_id)
        if adv_row:
            _assert_valid_advanced_row(adv_row, context="Fetched advanced row")
            advanced_rows.append(adv_row)
            print("ADV:OK")
        elif gid in ADVANCED_FALLBACK_DATA:
            fallback_row = ADVANCED_FALLBACK_DATA[gid].copy()
            _assert_valid_advanced_row(fallback_row, context="Fallback advanced row")
            advanced_rows.append(fallback_row)
            print("ADV:FALLBACK")
        else:
            print("ADV:FAIL")
        time.sleep(1.0)  # Delay to avoid rate limiting

        # Incremental save
        if i % save_interval == 0 or i == total:
            print(f"  [data] Saving progress ({i}/{total})...")

            # Save linescores
            if linescore_rows:
                new_ls = pd.DataFrame(linescore_rows)
                if existing_ls is not None and not existing_ls.empty:
                    combined_ls = pd.concat([existing_ls, new_ls], ignore_index=True)
                else:
                    combined_ls = new_ls
                combined_ls = _normalize_linescore_df(combined_ls)
                combined_ls.to_csv(linescore_path, index=False)
                existing_ls = combined_ls  # Update for next iteration
                ls_total_added += len(linescore_rows)
                linescore_rows = []  # Reset buffer

            # Save advanced
            if advanced_rows:
                new_adv = pd.DataFrame(advanced_rows)
                if existing_adv is not None and not existing_adv.empty:
                    combined_adv = pd.concat([existing_adv, new_adv], ignore_index=True)
                else:
                    combined_adv = new_adv
                combined_adv = _normalize_advanced_df(combined_adv)
                _assert_no_invalid_advanced_rows(combined_adv, context="Saving advanced box score data")
                combined_adv.to_csv(advanced_path, index=False)
                existing_adv = combined_adv  # Update for next iteration
                adv_total_added += len(advanced_rows)
                advanced_rows = []  # Reset buffer

    return ls_total_added, adv_total_added


# ------------------------- Raw PBP helpers -------------------------

def _season_start_year_from_str(season: str) -> int:
    m = re.match(r"^(\d{4})-(\d{2})$", str(season).strip())
    if not m:
        raise ValueError(f"Invalid season format: {season}. Expected YYYY-YY (e.g., 2025-26)")
    return int(m.group(1))


def _season_label_from_start_year(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _pbp_data_subdir(repo_dir: Path, source: str, season_type: str, create: bool = True) -> Path:
    path = repo_dir / PBP_ROOT_DIRNAME / source / season_type
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _historical_pbp_basename(source: str, start_year: int, season_type: str) -> str:
    if season_type == "playoffs":
        return f"{source}_po_{start_year}"
    return f"{source}_{start_year}"


def _find_historical_pbp_file(source_dir: Path, source: str, start_year: int, season_type: str) -> Optional[Path]:
    base = _historical_pbp_basename(source, start_year, season_type)
    csv_path = source_dir / f"{base}.csv"
    archive_path = source_dir / f"{base}.tar.xz"

    # Prefer plain CSV when present to avoid extra extraction work.
    if csv_path.exists():
        return csv_path
    if archive_path.exists():
        return archive_path
    return None


def _copy_or_extract_historical_csv(src_path: Path, dest_csv_path: Path) -> None:
    if src_path.suffix.lower() == ".csv":
        shutil.copy2(src_path, dest_csv_path)
        return

    if src_path.name.lower().endswith(".tar.xz"):
        with tarfile.open(src_path, mode="r:xz") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".csv")]
            if not members:
                raise RuntimeError(f"No CSV found inside archive: {src_path}")
            member = members[0]
            extracted = tf.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to read archive member {member.name} from {src_path}")
            with dest_csv_path.open("wb") as out_f:
                shutil.copyfileobj(extracted, out_f)
        return

    raise ValueError(f"Unsupported source file format: {src_path}")


def _pbp_output_filename(source: str, start_year: int, season_type: str) -> str:
    ext = ".parquet" if source == "nbastatsv3" else ".csv"
    if season_type == "playoffs":
        return f"{source}_po_{start_year}{ext}"
    return f"{source}_{start_year}{ext}"


def _backup_existing_pbp_file(pbp_path: Path, dry_run: bool = False) -> Optional[Path]:
    if not pbp_path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = pbp_path.with_name(f"{pbp_path.stem}_pre_nba_api_update_{timestamp}{pbp_path.suffix}")
    if dry_run:
        print(f"[pbp] DRY RUN - would backup {pbp_path.name} -> {backup_path.name}")
        return backup_path
    shutil.move(str(pbp_path), str(backup_path))
    print(f"[pbp] Backup created: {backup_path}")
    return backup_path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _guess_game_id_column(columns: list[str]) -> Optional[str]:
    for candidate in ("gameId", "GAME_ID", "game_id"):
        if candidate in columns:
            return candidate
    return None


def _read_pbp_table(pbp_path: Path) -> pd.DataFrame:
    if pbp_path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(pbp_path)
        except Exception:
            with tempfile.NamedTemporaryFile(prefix="pbp_parquet_bridge_", suffix=".csv", delete=False) as tmp_f:
                tmp_csv = Path(tmp_f.name)
            try:
                script = (
                    "import pandas as pd, sys; "
                    "d = pd.read_parquet(sys.argv[1]); "
                    "d.to_csv(sys.argv[2], index=False)"
                )
                proc = subprocess.run(
                    [PARQUET_BRIDGE_PYTHON, "-c", script, str(pbp_path), str(tmp_csv)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Could not read parquet file {pbp_path}. "
                        "Install pyarrow/fastparquet in this environment, "
                        f"or ensure {PARQUET_BRIDGE_PYTHON} has pyarrow. Error: {proc.stderr.strip() or proc.stdout.strip()}"
                    )
                return pd.read_csv(tmp_csv, low_memory=False)
            finally:
                try:
                    tmp_csv.unlink(missing_ok=True)
                except Exception:
                    pass
    return pd.read_csv(pbp_path, low_memory=False)


def _write_pbp_table(df: pd.DataFrame, pbp_path: Path) -> None:
    if pbp_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(pbp_path, index=False, compression=None)
        except Exception:
            with tempfile.NamedTemporaryFile(prefix="pbp_parquet_bridge_", suffix=".csv", delete=False) as tmp_f:
                tmp_csv = Path(tmp_f.name)
            try:
                df.to_csv(tmp_csv, index=False)
                script = (
                    "import pandas as pd, sys; "
                    "d = pd.read_csv(sys.argv[1], low_memory=False); "
                    "d.to_parquet(sys.argv[2], engine='pyarrow', compression=None, index=False)"
                )
                proc = subprocess.run(
                    [PARQUET_BRIDGE_PYTHON, "-c", script, str(tmp_csv), str(pbp_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Could not write parquet file {pbp_path}. "
                        "Install pyarrow/fastparquet in this environment, "
                        f"or ensure {PARQUET_BRIDGE_PYTHON} has pyarrow. Error: {proc.stderr.strip() or proc.stdout.strip()}"
                    )
            finally:
                try:
                    tmp_csv.unlink(missing_ok=True)
                except Exception:
                    pass
        return
    df.to_csv(pbp_path, index=False)


def _pbp_row_and_game_counts(pbp_path: Path) -> tuple[int, int]:
    if pbp_path.suffix.lower() == ".parquet":
        df = _read_pbp_table(pbp_path)
        if df.empty:
            return 0, 0
        game_id_col = _guess_game_id_column(list(df.columns))
        if not game_id_col:
            return int(len(df)), 0
        normalized = df[game_id_col].map(_normalize_game_id)
        game_count = int(sum(1 for gid in set(normalized.tolist()) if gid))
        return int(len(df)), game_count

    csv_path = pbp_path
    row_count = 0
    game_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        game_id_col = _guess_game_id_column(list(fieldnames))

        for row in reader:
            row_count += 1
            if game_id_col:
                gid = _normalize_game_id(row.get(game_id_col))
                if gid:
                    game_ids.add(gid)

    return row_count, len(game_ids)


def _load_pbp_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []

    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: row.get(k, "") for k in PBP_MANIFEST_COLUMNS})
    return rows


def _write_pbp_manifest_rows(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PBP_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PBP_MANIFEST_COLUMNS})


def _upsert_pbp_manifest_row(manifest_path: Path, record: dict[str, str]) -> None:
    rows = _load_pbp_manifest_rows(manifest_path)
    key = (
        record.get("source", ""),
        record.get("season", ""),
        record.get("season_type", ""),
        record.get("file_path", ""),
    )

    updated = False
    for i, row in enumerate(rows):
        row_key = (
            row.get("source", ""),
            row.get("season", ""),
            row.get("season_type", ""),
            row.get("file_path", ""),
        )
        if row_key == key:
            rows[i] = {k: record.get(k, "") for k in PBP_MANIFEST_COLUMNS}
            updated = True
            break

    if not updated:
        rows.append({k: record.get(k, "") for k in PBP_MANIFEST_COLUMNS})

    rows.sort(key=lambda r: (r.get("source", ""), r.get("season", ""), r.get("season_type", ""), r.get("file_path", "")))
    _write_pbp_manifest_rows(manifest_path, rows)


def _update_pbp_manifest_for_csv(
    repo_dir: Path,
    source: str,
    season: str,
    season_type: str,
    csv_path: Path,
) -> None:
    row_count, game_count = _pbp_row_and_game_counts(csv_path)
    try:
        rel_path = csv_path.relative_to(repo_dir).as_posix()
    except Exception:
        rel_path = str(csv_path)

    manifest_path = repo_dir / PBP_ROOT_DIRNAME / PBP_MANIFEST_FILENAME
    _upsert_pbp_manifest_row(
        manifest_path,
        {
            "source": source,
            "season": season,
            "season_type": season_type,
            "file_path": rel_path,
            "row_count": str(row_count),
            "game_count": str(game_count),
            "sha256": _sha256_file(csv_path),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _existing_game_ids_from_pbp_csv(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    if csv_path.suffix.lower() == ".parquet":
        try:
            d = _read_pbp_table(csv_path)
        except Exception:
            return set()
        if d.empty:
            return set()
        game_col = _guess_game_id_column(list(d.columns))
        if not game_col:
            return set()
        normalized = d[game_col].map(_normalize_game_id)
        return set(gid for gid in normalized.tolist() if gid)

    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return set()

    game_col = _guess_game_id_column(list(header.columns))
    if not game_col:
        return set()

    s = pd.read_csv(csv_path, usecols=[game_col], dtype={game_col: "string"})[game_col]
    normalized = s.map(_normalize_game_id)
    return set(gid for gid in normalized.tolist() if gid)


def _dedupe_pbp_actions_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    d = df.copy()
    game_col = _guess_game_id_column(list(d.columns))
    # Prefer the strongest event identity key first. nba_api v3 can repeat
    # actionNumber while keeping actionId unique, so deduping on actionNumber
    # can drop valid events for recent games.
    event_identity_cols: list[str] = []
    if "actionId" in d.columns:
        event_identity_cols = ["actionId"]
    elif "orderNumber" in d.columns:
        event_identity_cols = ["orderNumber"]
    else:
        for c in ("actionNumber", "EVENTNUM", "evt"):
            if c in d.columns:
                event_identity_cols = [c]
                break

    dedupe_cols: list[str] = []
    if game_col:
        dedupe_cols.append(game_col)
    dedupe_cols.extend(event_identity_cols)

    if len(dedupe_cols) >= 2:
        d = d.drop_duplicates(subset=dedupe_cols, keep="last")
    elif dedupe_cols:
        d = d.drop_duplicates(subset=dedupe_cols, keep="last")
    else:
        d = d.drop_duplicates(keep="last")

    sort_cols = [c for c in (game_col, "period", "PERIOD", "orderNumber", "actionNumber", "EVENTNUM", "evt") if c and c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols, kind="stable")

    return d.reset_index(drop=True)


def _build_game_metadata_from_team_logs(team_logs: pd.DataFrame, season_type_label: str) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    if team_logs.empty:
        return meta

    for _, row in team_logs.iterrows():
        gid = _normalize_game_id(row.get("GAME_ID"))
        if not gid:
            continue

        game_date = ""
        raw_date = row.get("GAME_DATE")
        if pd.notna(raw_date):
            dt = pd.to_datetime(raw_date, errors="coerce")
            if pd.notna(dt):
                game_date = dt.strftime("%Y-%m-%d")

        matchup = str(row.get("MATCHUP", "") or "")
        existing = meta.get(gid)
        if existing is None:
            meta[gid] = {
                "GAME_DATE": game_date,
                "MATCHUP": matchup,
                "SEASON_TYPE": season_type_label,
            }
            continue

        # Prefer a home-team formatted matchup (`vs.`) if available.
        if "vs." in matchup and "vs." not in existing.get("MATCHUP", ""):
            existing["MATCHUP"] = matchup
        if not existing.get("GAME_DATE") and game_date:
            existing["GAME_DATE"] = game_date

    return meta


def _build_game_metadata_from_local_gamelog(
    repo_dir: Path,
    season: str,
    season_type: str,
) -> Dict[str, Dict[str, str]]:
    """Build game metadata from local team_game_logs family data.

    This fallback is used when league game-list API calls time out.
    """
    csv_path = _resolve_repo_data_path(repo_dir, _season_to_filename(season))
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(
            csv_path,
            dtype={
                "game_id": "string",
                "game_date": "string",
                "game_type": "string",
                "team_abbreviation_home": "string",
                "team_abbreviation_road": "string",
            },
            usecols=[
                "game_id",
                "game_date",
                "game_type",
                "team_abbreviation_home",
                "team_abbreviation_road",
            ],
        )
    except Exception:
        return {}

    if df.empty:
        return {}

    d = df.copy()
    d["game_id"] = d["game_id"].map(_normalize_game_id)
    d["game_type"] = d["game_type"].map(_snake_case)
    d = d[d["game_id"] != ""].copy()

    if season_type == "regular":
        d = d[~d["game_type"].isin(["playoffs", "play_in"])]
        season_type_label = "REGULAR"
    else:
        d = d[d["game_type"].isin(["playoffs", "play_in"])]
        season_type_label = "PLAYOFFS"

    out: Dict[str, Dict[str, str]] = {}
    for _, row in d.iterrows():
        gid = str(row.get("game_id") or "")
        if not gid:
            continue

        game_date = ""
        raw_date = row.get("game_date")
        if pd.notna(raw_date):
            dt = pd.to_datetime(raw_date, errors="coerce")
            if pd.notna(dt):
                game_date = dt.strftime("%Y-%m-%d")

        home = str(row.get("team_abbreviation_home") or "")
        road = str(row.get("team_abbreviation_road") or "")
        matchup = f"{road} @ {home}".strip()

        out[gid] = {
            "GAME_DATE": game_date,
            "MATCHUP": matchup,
            "SEASON_TYPE": season_type_label,
        }

    return out


def _normalize_api_pbpv3_df(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce canonical column structure compatible with backfilled nbastatsv3 files."""
    d = df.copy()

    for col in PBPV3_CANONICAL_COLUMNS:
        if col not in d.columns:
            d[col] = pd.NA

    d = d[PBPV3_CANONICAL_COLUMNS]

    # Normalize IDs and numeric defaults to match nbastatsv3 conventions.
    d["gameId"] = d["gameId"].map(_normalize_game_id)
    d = d[d["gameId"] != ""].copy()

    int_default_zero_cols = [
        "actionNumber",
        "period",
        "teamId",
        "personId",
        "xLegacy",
        "yLegacy",
        "shotDistance",
        "isFieldGoal",
        "pointsTotal",
        "videoAvailable",
        "shotValue",
    ]
    for col in int_default_zero_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype("int64")

    # Keep score columns nullable numeric (nbastatsv3 has many null score rows).
    d["scoreHome"] = pd.to_numeric(d["scoreHome"], errors="coerce")
    d["scoreAway"] = pd.to_numeric(d["scoreAway"], errors="coerce")

    # actionId should be non-null in nbastatsv3; if missing, fall back to actionNumber.
    action_id_numeric = pd.to_numeric(d["actionId"], errors="coerce")
    d["actionId"] = action_id_numeric.fillna(d["actionNumber"]).astype("int64")

    sort_cols = [c for c in ("gameId", "actionNumber") if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols, kind="stable")

    return d.reset_index(drop=True)


def _short_error(exc: Exception, max_len: int = 220) -> str:
    msg = str(exc).replace("\n", " ").strip()
    return msg[:max_len] + ("..." if len(msg) > max_len else "")


def _fetch_pbp_from_cdnnba(game_id: str, request_timeout: float) -> pd.DataFrame:
    """Fetch liveData play-by-play JSON directly from CDN for one game."""
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=request_timeout, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        payload = resp.json()

    actions = payload.get("game", {}).get("actions", [])
    if not isinstance(actions, list) or not actions:
        raise RuntimeError("cdn.nba.com returned no actions")

    df = pd.DataFrame(actions)
    if df.empty:
        raise RuntimeError("cdn.nba.com produced empty actions table")
    if "gameId" not in df.columns:
        df["gameId"] = game_id

    return _normalize_api_pbpv3_df(df)


def _fetch_single_game_pbp(
    game_id: str,
    playbyplayv3_module: Any,
    request_timeout: float,
    retries: int,
) -> tuple[str, Optional[pd.DataFrame], str, str]:
    """Fetch one game's PBP with fast retries and source fallback.

    Returns:
      (game_id, dataframe_or_none, source_name, error_message)
    """
    attempt_count = max(1, int(retries))
    errors: list[str] = []

    for attempt in range(1, attempt_count + 1):
        try:
            resp = playbyplayv3_module.PlayByPlayV3(game_id=game_id, timeout=request_timeout)
            dfs = resp.get_data_frames()
            if dfs and dfs[0] is not None and not dfs[0].empty:
                df = dfs[0]
                if "gameId" not in df.columns:
                    df["gameId"] = game_id
                return game_id, _normalize_api_pbpv3_df(df), "nba_api", ""
            errors.append(f"nba_api attempt {attempt}: empty response")
        except Exception as e:
            errors.append(f"nba_api attempt {attempt}: {_short_error(e)}")

        try:
            cdn_df = _fetch_pbp_from_cdnnba(game_id, request_timeout=request_timeout)
            if cdn_df is not None and not cdn_df.empty:
                return game_id, cdn_df, "cdnnba", ""
            errors.append(f"cdnnba attempt {attempt}: empty response")
        except Exception as e:
            errors.append(f"cdnnba attempt {attempt}: {_short_error(e)}")

        if attempt < attempt_count:
            # Jittered backoff to reduce synchronized retry storms.
            time.sleep(PBP_FETCH_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0.0, 0.35))

    return game_id, None, "", " | ".join(errors[-4:])


# ------------------------- Raw PBP commands -------------------------

def backfill_pbp_raw(
    start_season: str,
    end_season: str,
    repo_dir: Path,
    source_dir: Path,
    include_cdnnba: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    start = time.time()
    try:
        repo_dir = ensure_data_repo(repo_dir)
        source_dir = source_dir.resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Historical source directory not found: {source_dir}")

        start_year = _season_start_year_from_str(start_season)
        end_year = _season_start_year_from_str(end_season)
        if end_year < start_year:
            raise ValueError("--end must be >= --start")

        copied = 0
        skipped_existing = 0
        missing_source = 0

        print(f"[pbp] Backfill from {start_season} to {end_season}")
        print(f"[pbp] Source dir: {source_dir}")
        print(f"[pbp] Target dir: {repo_dir / PBP_ROOT_DIRNAME}")

        for year in range(start_year, end_year + 1):
            season = _season_label_from_start_year(year)
            for season_type in ("regular", "playoffs"):
                for source in ("nbastatsv3", "cdnnba"):
                    if source == "cdnnba" and (not include_cdnnba or year < 2020):
                        continue

                    src_file = _find_historical_pbp_file(source_dir, source, year, season_type)
                    if src_file is None:
                        print(f"[pbp] Missing source file: {source} {season} {season_type}")
                        missing_source += 1
                        continue

                    dest_dir = _pbp_data_subdir(repo_dir, source, season_type, create=not dry_run)
                    basename = _historical_pbp_basename(source, year, season_type)
                    dest_file = dest_dir / f"{basename}.csv"

                    if dest_file.exists() and not overwrite:
                        print(f"[pbp] Skip existing: {dest_file}")
                        skipped_existing += 1
                        if not dry_run:
                            _update_pbp_manifest_for_csv(repo_dir, source, season, season_type, dest_file)
                        continue

                    print(f"[pbp] {'Would write' if dry_run else 'Writing'} {dest_file.name} from {src_file.name}")
                    if dry_run:
                        copied += 1
                        continue

                    _copy_or_extract_historical_csv(src_file, dest_file)
                    _update_pbp_manifest_for_csv(repo_dir, source, season, season_type, dest_file)
                    copied += 1

        elapsed = time.time() - start
        print("\n[pbp] Historical backfill complete")
        print(f"  copied_or_updated: {copied}")
        print(f"  skipped_existing: {skipped_existing}")
        print(f"  missing_source_files: {missing_source}")
        print(f"  dry_run: {dry_run}")
        print(f"  time: {elapsed:.1f}s")
        return 0

    except Exception as e:
        print(f"[error] backfill-pbp-raw failed: {e}")
        return 1


def _prune_pbp_manifest_missing_files(repo_dir: Path) -> int:
    manifest_path = repo_dir / PBP_ROOT_DIRNAME / PBP_MANIFEST_FILENAME
    rows = _load_pbp_manifest_rows(manifest_path)
    if not rows:
        return 0

    kept: list[dict[str, str]] = []
    for row in rows:
        rel = (row.get("file_path") or "").strip()
        if not rel:
            continue
        candidate = Path(rel)
        path = candidate if candidate.is_absolute() else (repo_dir / candidate)
        if path.exists():
            kept.append(row)

    removed = len(rows) - len(kept)
    if removed > 0:
        _write_pbp_manifest_rows(manifest_path, kept)
    return removed


def migrate_nbastatsv3_csv_to_parquet(
    repo_dir: Path,
    archive_rel_dir: str = "PBPdata/nbastatsv3_csv_archive",
    dry_run: bool = False,
) -> int:
    repo_dir = ensure_data_repo(repo_dir)
    archive_root = Path(archive_rel_dir)
    if not archive_root.is_absolute():
        archive_root = repo_dir / archive_root

    converted = 0
    archived = 0

    try:
        for phase in ("regular", "playoffs"):
            src_dir = repo_dir / PBP_ROOT_DIRNAME / "nbastatsv3" / phase
            if not src_dir.exists():
                continue

            for csv_path in sorted(src_dir.glob("*.csv")):
                archive_path = archive_root / phase / csv_path.name
                m = re.match(r"^nbastatsv3(?:_po)?_(\d{4})\.csv$", csv_path.name)
                parquet_path = csv_path.with_suffix(".parquet") if m else None

                if dry_run:
                    if parquet_path is not None:
                        print(f"[pbp] DRY RUN - convert {csv_path.name} -> {parquet_path.name}")
                        converted += 1
                    print(f"[pbp] DRY RUN - archive {csv_path.name} -> {archive_path}")
                    archived += 1
                    continue

                if parquet_path is not None:
                    script = (
                        "import pandas as pd, sys; "
                        "d = pd.read_csv(sys.argv[1], low_memory=False); "
                        "d.to_parquet(sys.argv[2], engine='pyarrow', compression=None, index=False)"
                    )
                    proc = subprocess.run(
                        ["python3", "-c", script, str(csv_path), str(parquet_path)],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"Failed to convert {csv_path} to parquet: "
                            f"{proc.stderr.strip() or proc.stdout.strip()}"
                        )
                    start_year = int(m.group(1))
                    season = _season_label_from_start_year(start_year)
                    _update_pbp_manifest_for_csv(
                        repo_dir=repo_dir,
                        source="nbastatsv3",
                        season=season,
                        season_type=phase,
                        csv_path=parquet_path,
                    )
                    converted += 1

                archive_path.parent.mkdir(parents=True, exist_ok=True)
                if archive_path.exists():
                    archive_path.unlink()
                shutil.move(str(csv_path), str(archive_path))
                archived += 1

        removed_manifest = 0
        if not dry_run:
            removed_manifest = _prune_pbp_manifest_missing_files(repo_dir)

        print("\n[pbp] nbastatsv3 CSV -> Parquet migration complete")
        print(f"  converted_to_parquet: {converted}")
        print(f"  archived_csv_files: {archived}")
        print(f"  manifest_rows_removed: {removed_manifest}")
        print(f"  dry_run: {dry_run}")
        return 0
    except Exception as e:
        print(f"[error] migrate nbastatsv3 csv->parquet failed: {e}")
        return 1


def _update_pbp_for_season_type(
    season: str,
    repo_dir: Path,
    season_type: str,
    output_source: str = "nbastatsv3",
    backup_existing: bool = True,
    overwrite_existing: bool = False,
    max_games: Optional[int] = None,
    request_timeout: float = PBP_FETCH_TIMEOUT_SECONDS,
    retries: int = PBP_FETCH_RETRIES,
    max_workers: int = PBP_FETCH_MAX_WORKERS,
    dry_run: bool = False,
) -> int:
    if season_type not in {"regular", "playoffs"}:
        raise ValueError(f"Unsupported season_type: {season_type}")
    if output_source not in {"nbastatsv3", "api_pbpv3"}:
        raise ValueError(f"Unsupported output_source: {output_source}")

    try:
        from nba_api.stats.endpoints import playbyplayv3
    except Exception as e:
        print(f"[error] Could not import playbyplayv3 endpoint: {e}")
        return 1

    start_year = _season_start_year_from_str(season)
    filename = _pbp_output_filename(output_source, start_year, season_type)
    out_dir = _pbp_data_subdir(repo_dir, output_source, season_type, create=not dry_run)
    out_path = out_dir / filename
    backup_done = False

    def ensure_backup_before_write() -> None:
        nonlocal backup_done
        if backup_done:
            return
        if not backup_existing:
            return
        if output_source != "nbastatsv3":
            return
        if not out_path.exists():
            return
        _backup_existing_pbp_file(out_path, dry_run=dry_run)
        backup_done = True

    api_groups = (
        [("Regular Season", "REGULAR"), ("IST", "IST")]
        if season_type == "regular"
        else [("Playoffs", "PLAYOFFS"), ("PlayIn", "PLAY_IN")]
    )

    # Prefer local game logs to avoid long season-list API timeouts.
    game_meta: Dict[str, Dict[str, str]] = _build_game_metadata_from_local_gamelog(
        repo_dir=repo_dir,
        season=season,
        season_type=season_type,
    )
    if game_meta:
        print(f"[pbp] Using local game logs for {season} ({season_type}); games={len(game_meta)}")
    else:
        for api_season_type, label in api_groups:
            try:
                logs = _fetch_season_team_game_logs(season, season_type=api_season_type)
            except Exception as e:
                print(f"[pbp] Warning: failed to fetch {api_season_type} game list for {season}: {e}")
                continue
            if logs.empty:
                continue
            partial = _build_game_metadata_from_team_logs(logs, label)
            for gid, meta in partial.items():
                if gid not in game_meta:
                    game_meta[gid] = meta

    if not game_meta:
        print(f"[pbp] No games found for {season} ({season_type})")
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    candidate_game_ids = sorted(
        gid for gid, meta in game_meta.items()
        if meta.get("GAME_DATE") and meta.get("GAME_DATE") != today
    )
    existing_game_ids = _existing_game_ids_from_pbp_csv(out_path)
    if overwrite_existing:
        missing_game_ids = list(candidate_game_ids)
    else:
        missing_game_ids = [gid for gid in candidate_game_ids if gid not in existing_game_ids]

    if max_games is not None and max_games > 0:
        missing_game_ids = missing_game_ids[:max_games]

    print(f"[pbp] Season {season} ({season_type})")
    print(f"[pbp] Output: {out_path}")
    print(f"[pbp] Candidate games: {len(candidate_game_ids)}")
    print(f"[pbp] Already present: {len(existing_game_ids)}")
    print(f"[pbp] Missing to fetch: {len(missing_game_ids)}")
    if overwrite_existing:
        print("[pbp] Overwrite mode: replacing existing rows for selected game_ids")

    if not missing_game_ids:
        if out_path.exists():
            # Still rewrite/normalize existing output so field defaults stay consistent.
            try:
                existing_df = _read_pbp_table(out_path)
            except Exception:
                existing_df = pd.DataFrame()
            if not existing_df.empty:
                existing_df = _normalize_api_pbpv3_df(existing_df)
                existing_df = _dedupe_pbp_actions_df(existing_df)
                existing_df = _normalize_api_pbpv3_df(existing_df)
                ensure_backup_before_write()
                _write_pbp_table(existing_df, out_path)
            _update_pbp_manifest_for_csv(repo_dir, output_source, season, season_type, out_path)
        return 0

    if dry_run:
        print("[pbp] DRY RUN - no API calls performed")
        return 0

    fetched_frames: list[pd.DataFrame] = []
    ok = 0
    failed = 0
    workers = max(1, int(max_workers))
    timeout_s = float(request_timeout)
    retry_count = max(1, int(retries))
    total = len(missing_game_ids)

    print(f"[pbp] Fetch config: workers={workers}, timeout={timeout_s:.1f}s, retries={retry_count}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _fetch_single_game_pbp,
                gid,
                playbyplayv3,
                timeout_s,
                retry_count,
            ): gid
            for gid in missing_game_ids
        }

        for idx, future in enumerate(as_completed(future_map), 1):
            gid = future_map[future]
            try:
                _, pbp_df, source_name, err_msg = future.result()
            except Exception as e:
                pbp_df = None
                source_name = ""
                err_msg = _short_error(e)

            if pbp_df is not None and not pbp_df.empty:
                fetched_frames.append(pbp_df)
                ok += 1
                print(f"[pbp]   [{idx}/{total}] {gid} OK ({source_name}, rows={len(pbp_df)})")
            else:
                failed += 1
                print(f"[pbp]   [{idx}/{total}] {gid} FAIL ({err_msg})")

    if not fetched_frames:
        print(f"[pbp] No new rows fetched for {season} ({season_type})")
        return 1 if failed else 0

    new_df = pd.concat(fetched_frames, ignore_index=True)

    if out_path.exists():
        try:
            existing_df = _read_pbp_table(out_path)
        except Exception:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    if not existing_df.empty:
        existing_df = _normalize_api_pbpv3_df(existing_df)
        if overwrite_existing:
            game_col = _guess_game_id_column(list(existing_df.columns))
            if game_col:
                refresh_set = set(missing_game_ids)
                existing_df = existing_df[
                    ~existing_df[game_col].map(_normalize_game_id).isin(refresh_set)
                ].copy()

    existing_rows = len(existing_df)
    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    combined = _normalize_api_pbpv3_df(combined)
    combined = _dedupe_pbp_actions_df(combined)
    combined = _normalize_api_pbpv3_df(combined)
    ensure_backup_before_write()
    _write_pbp_table(combined, out_path)

    _update_pbp_manifest_for_csv(repo_dir, output_source, season, season_type, out_path)

    print(f"[pbp] Wrote {out_path.name}: rows {existing_rows} -> {len(combined)}")
    print(f"[pbp] Fetch summary: success={ok}, failed={failed}")
    return 1 if failed else 0


def update_pbp_raw(
    season: str,
    repo_dir: Path,
    season_phase: str = "both",
    target_source: str = "nbastatsv3",
    backup_existing: bool = True,
    overwrite_existing: bool = False,
    max_games: Optional[int] = None,
    request_timeout: float = PBP_FETCH_TIMEOUT_SECONDS,
    retries: int = PBP_FETCH_RETRIES,
    max_workers: int = PBP_FETCH_MAX_WORKERS,
    migrate_nbastatsv3_to_parquet: bool = False,
    csv_archive_dir: str = "PBPdata/nbastatsv3_csv_archive",
    dry_run: bool = False,
) -> int:
    try:
        repo_dir = ensure_data_repo(repo_dir)
        if migrate_nbastatsv3_to_parquet:
            return migrate_nbastatsv3_csv_to_parquet(
                repo_dir=repo_dir,
                archive_rel_dir=csv_archive_dir,
                dry_run=dry_run,
            )
        _season_start_year_from_str(season)  # validate format
        if target_source not in {"nbastatsv3", "api_pbpv3"}:
            raise ValueError(f"Unsupported --target-source: {target_source}")
        current_season = get_current_season()
        if target_source == "nbastatsv3" and season != current_season and not overwrite_existing:
            raise ValueError(
                f"nbastatsv3 updates are limited to current season ({current_season}). "
                "Use --target-source api_pbpv3 for non-current seasons, "
                "or pass --overwrite-existing for a full historical refresh."
            )

        phases = ["regular", "playoffs"] if season_phase == "both" else [season_phase]
        exit_code = 0
        for phase in phases:
            rc = _update_pbp_for_season_type(
                season=season,
                repo_dir=repo_dir,
                season_type=phase,
                output_source=target_source,
                backup_existing=backup_existing,
                overwrite_existing=overwrite_existing,
                max_games=max_games,
                request_timeout=request_timeout,
                retries=retries,
                max_workers=max_workers,
                dry_run=dry_run,
            )
            if rc != 0:
                exit_code = rc
        return exit_code

    except Exception as e:
        print(f"[error] update-pbp-raw failed: {e}")
        return 1


def fetch_pbp_game(
    season: str,
    game_id: str,
    repo_dir: Path,
    season_phase: str = "regular",
    target_source: str = "nbastatsv3",
    backup_existing: bool = True,
    request_timeout: float = PBP_FETCH_TIMEOUT_SECONDS,
    retries: int = PBP_FETCH_RETRIES,
    overwrite_game: bool = False,
    dry_run: bool = False,
) -> int:
    """Fetch raw PBP for one game_id and merge into a season source file."""
    try:
        from nba_api.stats.endpoints import playbyplayv3
    except Exception as e:
        print(f"[error] Could not import playbyplayv3 endpoint: {e}")
        return 1

    try:
        repo_dir = ensure_data_repo(repo_dir)
        start_year = _season_start_year_from_str(season)  # validates YYYY-YY
        if season_phase not in {"regular", "playoffs"}:
            raise ValueError(f"Unsupported --phase: {season_phase}")
        if target_source not in {"nbastatsv3", "api_pbpv3"}:
            raise ValueError(f"Unsupported --target-source: {target_source}")

        gid = _normalize_game_id(game_id)
        if len(gid) != 10 or not gid.isdigit():
            raise ValueError(f"Invalid --game-id: {game_id}. Expected a 10-digit game id like 0021201216")

        out_dir = _pbp_data_subdir(repo_dir, target_source, season_phase, create=not dry_run)
        out_name = _pbp_output_filename(target_source, start_year, season_phase)
        out_path = out_dir / out_name

        existing_df = pd.DataFrame()
        existing_ids: set[str] = set()
        if out_path.exists():
            try:
                existing_df = _read_pbp_table(out_path)
            except Exception:
                existing_df = pd.DataFrame()
            existing_ids = _existing_game_ids_from_pbp_csv(out_path)

        already_present = gid in existing_ids
        print(f"[pbp] Single-game fetch: season={season}, phase={season_phase}, source={target_source}, game_id={gid}")
        print(f"[pbp] Output: {out_path}")
        print(f"[pbp] Already present: {already_present}")

        if already_present and not overwrite_game:
            print("[pbp] Skip (game already present). Use --overwrite-game to replace existing rows for this game.")
            return 0

        if dry_run:
            print("[pbp] DRY RUN - no API calls performed")
            return 0

        _, pbp_df, source_name, err_msg = _fetch_single_game_pbp(
            gid,
            playbyplayv3,
            float(request_timeout),
            max(1, int(retries)),
        )
        if pbp_df is None or pbp_df.empty:
            print(f"[pbp] FAIL {gid} ({err_msg})")
            return 1

        if not existing_df.empty:
            existing_df = _normalize_api_pbpv3_df(existing_df)
            if overwrite_game:
                game_col = _guess_game_id_column(list(existing_df.columns))
                if game_col:
                    mask = existing_df[game_col].map(_normalize_game_id) != gid
                    existing_df = existing_df[mask].copy()

        combined = pd.concat([existing_df, pbp_df], ignore_index=True, sort=False)
        combined = _normalize_api_pbpv3_df(combined)
        combined = _dedupe_pbp_actions_df(combined)
        combined = _normalize_api_pbpv3_df(combined)

        if backup_existing and target_source == "nbastatsv3" and out_path.exists():
            _backup_existing_pbp_file(out_path, dry_run=False)

        _write_pbp_table(combined, out_path)
        _update_pbp_manifest_for_csv(repo_dir, target_source, season, season_phase, out_path)

        print(f"[pbp] OK {gid} ({source_name}, rows={len(pbp_df)})")
        print(f"[pbp] Wrote {out_path.name} with total rows={len(combined)} games={len(_existing_game_ids_from_pbp_csv(out_path))}")
        return 0

    except Exception as e:
        print(f"[error] fetch-pbp-game failed: {e}")
        return 1


# ------------------------- CLI commands -------------------------


def _discover_state_seasons(repo_dir: Path, input_root: Optional[str], phase: str) -> list[str]:
    root = Path(input_root) if input_root else (repo_dir / "PBPdata" / "game_states")
    phase_dir = root / phase
    seasons: set[str] = set()

    if phase_dir.exists() and phase_dir.is_dir():
        for child in phase_dir.iterdir():
            if child.is_dir() and re.match(r"^\d{4}-\d{2}$", child.name):
                seasons.add(child.name)
            elif child.is_file():
                m = re.match(r"^_states_(\d{4}-\d{2})_[a-z]+\.parquet$", child.name)
                if m:
                    seasons.add(m.group(1))

    if not seasons:
        for p in iter_data_family_files(repo_dir, "team_game_logs", "team_game_logs_????-??.csv"):
            season = p.stem.replace("team_game_logs_", "", 1)
            if re.match(r"^\d{4}-\d{2}$", season):
                seasons.add(season)

    return sorted(seasons, key=_season_start_year_from_str)


def pack_pbp_game_states_cmd(
    season: str,
    repo_dir: Path,
    phase: str = "regular",
    input_root: Optional[str] = None,
    output_root: Optional[str] = None,
    compression: str = "zstd",
    overwrite: bool = False,
    delete_json: bool = False,
) -> int:
    from admin.pbp_game_states import pack_pbp_game_states

    phase_norm = str(phase or "regular").strip().lower()
    if phase_norm not in {"regular", "playoffs", "both"}:
        raise ValueError("Invalid phase. Expected regular, playoffs, or both.")
    phases = ["regular", "playoffs"] if phase_norm == "both" else [phase_norm]

    season_norm = str(season or "").strip().lower()
    if season_norm == "all":
        season_set: set[str] = set()
        for p in phases:
            season_set.update(_discover_state_seasons(repo_dir=repo_dir, input_root=input_root, phase=p))
        seasons = sorted(season_set, key=_season_start_year_from_str)
    else:
        seasons = [season]

    if not seasons:
        print("[pbp-pack] No seasons found to pack.")
        return 1

    total_jobs = len(seasons) * len(phases)
    failures = 0
    completed = 0

    for p in phases:
        for s in seasons:
            completed += 1
            print(f"[pbp-pack] [{completed}/{total_jobs}] season={s} phase={p}")
            rc = pack_pbp_game_states(
                season=s,
                repo_dir=repo_dir,
                phase=p,
                input_root=input_root,
                output_root=output_root,
                compression=compression,
                overwrite=overwrite,
                delete_json=delete_json,
            )
            if rc != 0:
                failures += 1

    if failures:
        print(f"[pbp-pack] Finished with failures: {failures}/{total_jobs}")
        return 1

    print(f"[pbp-pack] Done: {total_jobs} season/phase pack jobs")
    return 0


def check_pbp_completeness_cmd(
    season: str,
    repo_dir: Path,
    phase: str = "regular",
    pbp_source: str = "nbastatsv3",
    list_limit: int = 20,
) -> int:
    from admin.pbp_game_states import check_pbp_completeness

    summary = check_pbp_completeness(
        season=season,
        repo_dir=repo_dir,
        phase=phase,
        source=pbp_source,
    )
    print(
        f"[pbp-check] Season={summary['season']} phase={summary['phase']} "
        f"source={summary['source']}"
    )
    print(f"[pbp-check] Path={summary['path']}")
    print(f"[pbp-check] Expected games={summary['expected_game_count']}")
    print(f"[pbp-check] PBP games={summary['pbp_game_count']}")
    print(f"[pbp-check] Missing expected games={summary['missing_game_count']}")
    print(f"[pbp-check] Extra PBP games={summary['extra_game_count']}")

    if not summary["file_exists"]:
        print("[pbp-check] Status=MISSING_FILE")
        return 1

    print(f"[pbp-check] Status={'COMPLETE' if summary['complete'] else 'INCOMPLETE'}")

    limit = max(0, int(list_limit))
    if summary["missing_game_ids"]:
        shown = summary["missing_game_ids"][:limit] if limit else []
        if shown:
            print(f"[pbp-check] Missing game IDs (first {len(shown)}): {', '.join(shown)}")
        remaining = len(summary["missing_game_ids"]) - len(shown)
        if remaining > 0:
            print(f"[pbp-check] Missing game IDs not shown: {remaining}")

    if summary["extra_game_ids"]:
        shown = summary["extra_game_ids"][:limit] if limit else []
        if shown:
            print(f"[pbp-check] Extra game IDs (first {len(shown)}): {', '.join(shown)}")
        remaining = len(summary["extra_game_ids"]) - len(shown)
        if remaining > 0:
            print(f"[pbp-check] Extra game IDs not shown: {remaining}")

    return 0 if summary["complete"] else 1


def build_pbp_timeline_metrics_cmd(
    season: str,
    repo_dir: Path,
    phase: str = "regular",
    input_root: Optional[str] = None,
    output_root: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    from admin.pbp_game_states import build_timeline_metrics

    phase_norm = str(phase or "regular").strip().lower()
    if phase_norm not in {"regular", "playoffs", "both"}:
        raise ValueError("Invalid phase. Expected regular, playoffs, or both.")
    phases = ["regular", "playoffs"] if phase_norm == "both" else [phase_norm]

    season_norm = str(season or "").strip().lower()
    if season_norm == "all":
        season_set: set[str] = set()
        for p in phases:
            season_set.update(_discover_state_seasons(repo_dir=repo_dir, input_root=input_root, phase=p))
        seasons = sorted(season_set, key=_season_start_year_from_str)
    else:
        seasons = [season]

    if not seasons:
        print("[pbp-metrics] No seasons found to process.")
        return 1

    total_jobs = len(seasons) * len(phases)
    failures = 0
    completed = 0

    for p in phases:
        for s in seasons:
            completed += 1
            print(f"[pbp-metrics] [{completed}/{total_jobs}] season={s} phase={p}")
            rc = build_timeline_metrics(
                season=s,
                repo_dir=repo_dir,
                phase=p,
                input_root=input_root,
                output_root=output_root,
                overwrite=overwrite,
            )
            if rc != 0:
                failures += 1

    if failures:
        print(f"[pbp-metrics] Finished with failures: {failures}/{total_jobs}")
        return 1

    print(f"[pbp-metrics] Done: {total_jobs} season/phase metric jobs")
    return 0


def update_data(season: str, repo_dir: Path, force_refresh: bool = False) -> int:
    start = time.time()
    try:
        repo_dir = ensure_data_repo(repo_dir)
        csv_path = _canonical_repo_data_path(repo_dir, _season_to_filename(season))
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        existing_raw = _load_existing_season_csv(csv_path)
        existing: Optional[pd.DataFrame]
        if existing_raw is None:
            existing = None
        else:
            # If previously polluted, normalizer will drop extras and enforce schema
            existing = _normalize_game_level_df(existing_raw)

        # Load NBA Cup knockout dates for game_type overrides
        cup_dates = _load_nba_cup_dates()
        if cup_dates:
            print(f"[data] Loaded {len(cup_dates)} NBA Cup knockout date(s)")

        # Fetch all season types and combine
        # IST = In-Season Tournament (NBA Cup) - these count as regular_season
        # except for the final which gets overridden via cup_dates
        season_types = [
            ("Regular Season", "regular_season"),
            ("IST", "regular_season"),  # NBA Cup games - semifinals/finals overridden by cup_dates
            ("Playoffs", "playoffs"),
            ("PlayIn", "play_in"),
        ]

        all_gamelogs: list[pd.DataFrame] = []
        for api_type, game_type_label in season_types:
            print(f"[data] Fetching {season} {api_type} from NBA API...")
            try:
                team_logs = _fetch_season_team_game_logs(season, season_type=api_type)
                if not team_logs.empty:
                    gamelogs = _teamlogs_to_gamelogs(team_logs, season=season, game_type=game_type_label)
                    if not gamelogs.empty:
                        print(f"[data]   Found {len(gamelogs)} {api_type} games")
                        all_gamelogs.append(gamelogs)
                    else:
                        print(f"[data]   No {api_type} games found")
                else:
                    print(f"[data]   No {api_type} games found")
            except Exception as e:
                print(f"[data]   Error fetching {api_type}: {e}")

        if not all_gamelogs:
            print("[data] No games found for any season type")
            return 1

        print("[data] Combining all game types...")
        fresh_raw = pd.concat(all_gamelogs, ignore_index=True)

        # Apply NBA Cup knockout date overrides (only affects regular_season games)
        if cup_dates:
            fresh_raw = _apply_nba_cup_overrides(fresh_raw, cup_dates)
            # Count how many were overridden
            cup_games = fresh_raw[fresh_raw["game_type"].isin(["nba_cup_semi", "nba_cup_final"])]
            if not cup_games.empty:
                print(f"[data] Tagged {len(cup_games)} game(s) as NBA Cup knockout")

        fresh = _normalize_game_level_df(fresh_raw)

        # Skip today's games to avoid incomplete data from in-progress games
        today_str = datetime.now().strftime("%Y-%m-%d")
        games_today = fresh["game_date"] == today_str
        if games_today.any():
            skipped_count = games_today.sum()
            print(f"[data] Skipping {skipped_count} game(s) from today ({today_str}) - may be in progress")
            fresh = fresh[~games_today].copy()

        if existing is None or existing.empty:
            merged = fresh
            before = 0
            added = len(fresh)
        else:
            before = len(existing)
            existing_ids = set(existing["game_id"].astype(str).tolist())
            fresh_ids = set(fresh["game_id"].astype(str).tolist())

            # Find brand-new games
            fresh_new = fresh[~fresh["game_id"].astype(str).isin(existing_ids)].copy()
            added = len(fresh_new)

            if force_refresh:
                # Update game_type for existing games that appear in fresh data
                # This allows re-categorizing games (e.g., adding IST games, fixing game_types)
                games_to_update = existing_ids & fresh_ids
                if games_to_update:
                    # Create a mapping of game_id -> new game_type from fresh data
                    fresh_game_types = fresh.set_index(fresh["game_id"].astype(str))["game_type"].to_dict()

                    # Update existing rows
                    updated_count = 0
                    for idx, row in existing.iterrows():
                        gid = str(row["game_id"])
                        if gid in games_to_update:
                            old_type = existing.at[idx, "game_type"]
                            new_type = fresh_game_types.get(gid, old_type)
                            if old_type != new_type:
                                existing.at[idx, "game_type"] = new_type
                                updated_count += 1

                    if updated_count > 0:
                        print(f"[data] Updated game_type for {updated_count} existing game(s)")

                # Also add any games from fresh that don't exist yet
                merged = pd.concat([existing, fresh_new], ignore_index=True)
            else:
                # Standard behavior: ONLY append brand-new games; preserve existing rows
                merged = pd.concat([existing, fresh_new], ignore_index=True)

            merged = _normalize_game_level_df(merged)

        merged.to_csv(csv_path, index=False)

        # ---- Fetch linescore and advanced stats for new games ----
        linescore_path = _canonical_repo_data_path(repo_dir, _linescore_filename(season))
        advanced_path = _canonical_repo_data_path(repo_dir, _advanced_filename(season))
        linescore_path.parent.mkdir(parents=True, exist_ok=True)
        advanced_path.parent.mkdir(parents=True, exist_ok=True)

        ls_added = 0
        adv_added = 0
        batch_num = 0

        # Loop with auto-restart after BATCH_RESTART_SIZE games to avoid rate limiting
        while True:
            batch_num += 1

            # Load existing boxscore data (re-load each iteration to pick up saved progress)
            existing_ls: Optional[pd.DataFrame] = None
            existing_adv: Optional[pd.DataFrame] = None
            if linescore_path.exists():
                existing_ls = pd.read_csv(linescore_path, dtype={"game_id": "string"})
            if advanced_path.exists():
                existing_adv = pd.read_csv(advanced_path, dtype={"game_id": "string"})
                existing_adv = _normalize_advanced_df(existing_adv)
                _assert_no_invalid_advanced_rows(
                    existing_adv,
                    context="Existing advanced box score data",
                )

            # Determine which games need boxscore data (excluding today's games)
            today_str = datetime.now().strftime("%Y-%m-%d")
            all_game_ids = set(
                merged[merged["game_date"] != today_str]["game_id"].astype(str).tolist()
            )
            existing_ls_ids = set(existing_ls["game_id"].astype(str).tolist()) if existing_ls is not None else set()
            existing_adv_ids = set(existing_adv["game_id"].astype(str).tolist()) if existing_adv is not None else set()

            # Games that need fetching (not in BOTH linescore and advanced)
            already_fetched = existing_ls_ids & existing_adv_ids
            new_game_ids = all_game_ids - already_fetched

            if not new_game_ids:
                break  # All games fetched

            # Limit to BATCH_RESTART_SIZE games per batch
            batch_game_ids = list(new_game_ids)[:BATCH_RESTART_SIZE]

            # Build game_info from merged gamelog
            game_info: Dict[str, tuple] = {}
            for _, row in merged.iterrows():
                gid = str(row["game_id"])
                if gid in batch_game_ids:
                    game_info[gid] = (row["game_date"], int(row["team_id_home"]))

            print(f"\n[data] Batch {batch_num}: Fetching {len(batch_game_ids)} games ({len(new_game_ids)} remaining)...")
            batch_ls, batch_adv = _fetch_boxscore_data(
                batch_game_ids,
                game_info,
                season,
                linescore_path,
                advanced_path,
                existing_ls,
                existing_adv,
            )
            ls_added += batch_ls
            adv_added += batch_adv

            # If we fetched a full batch and more games remain, do a true process restart
            if len(batch_game_ids) == BATCH_RESTART_SIZE and (all_game_ids - already_fetched - set(batch_game_ids)):
                print(f"\n[data] Auto-restart: spawning new process after 10s pause...")
                time.sleep(10)
                # Spawn a new Python process to continue
                # Note: --repo-dir is a global arg that must come BEFORE the subcommand
                import sys
                result = subprocess.run(
                    [sys.executable, __file__, "--repo-dir", str(repo_dir), "update-data", "--season", season],
                    cwd=str(Path(__file__).parent),
                )
                # The subprocess handles the rest; we're done
                return result.returncode

        # Count final rows in boxscore files
        ls_total = len(pd.read_csv(linescore_path)) if linescore_path.exists() else 0
        adv_total = 0
        if advanced_path.exists():
            final_adv = _normalize_advanced_df(pd.read_csv(advanced_path, dtype={"game_id": "string"}))
            _assert_no_invalid_advanced_rows(final_adv, context="Final advanced box score data")
            adv_total = len(final_adv)

        elapsed = time.time() - start
        latest_date = None
        if "game_date" in merged.columns and merged["game_date"].notna().any():
            latest_date = str(pd.to_datetime(merged["game_date"], errors="coerce").max().date())

        print("\n[data] Update complete")
        print(f"  gamelog: {csv_path.name} ({len(merged)} rows, +{added} new)")
        print(f"  linescore: {linescore_path.name} ({ls_total} rows, +{ls_added} new)")
        print(f"  advanced: {advanced_path.name} ({adv_total} rows, +{adv_added} new)")
        if latest_date:
            print(f"  latest: {latest_date}")
        print(f"  time: {elapsed:.1f}s")
        return 0

    except Exception as e:
        print(f"[error] update-data failed: {e}")
        return 1


def download_data(start_season: str, end_season: str, repo_dir: Path) -> int:
    start = time.time()
    try:
        repo_dir = ensure_data_repo(repo_dir)

        s0 = int(start_season.split("-")[0])
        s1 = int(end_season.split("-")[0])
        if s1 < s0:
            raise ValueError("--end must be >= --start")

        failed: list[str] = []
        total = 0

        for yr in range(s0, s1 + 1):
            season = f"{yr}-{str(yr + 1)[-2:]}"
            total += 1

            try:
                # If a file already exists, preserve it and only append new game_ids
                print(f"[data] Downloading/updating {season}")
                update_data(season, repo_dir)
                time.sleep(0.6)
            except Exception as e:
                print(f"  [warn] Failed {season}: {e}")
                failed.append(season)

        elapsed = time.time() - start
        print("\n[data] Bulk download complete")
        print(f"  total: {total}")
        print(f"  failed: {failed if failed else 'None'}")
        print(f"  time: {elapsed:.1f}s")

        return 1 if failed else 0

    except Exception as e:
        print(f"[error] download-data failed: {e}")
        return 1


def git_status(repo_dir: Path) -> int:
    try:
        repo_dir = ensure_data_repo(repo_dir)
        res = _run_git(["status", "--short"], cwd=repo_dir, check=True)
        out = res.stdout.strip()
        print(out if out else "(clean)")
        return 0
    except subprocess.CalledProcessError as e:
        print("[error] git-status failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return 1
    except Exception as e:
        print(f"[error] git-status failed: {e}")
        return 1


def commit_and_push(message: str, repo_dir: Path, dry_run: bool = False) -> int:
    try:
        repo_dir = ensure_data_repo(repo_dir)

        # Limit status to entire repo (season CSVs, contributions, interpretations, etc.)
        res = _run_git(["status", "--short", "."], cwd=repo_dir, check=True)
        status_out = res.stdout.strip()

        print("[git] Changes in NBA_Data repo:")
        print(status_out if status_out else "(no changes)")

        if not status_out:
            return 0

        if dry_run:
            print("[git] DRY RUN: not committing/pushing")
            return 0

        print("[git] Adding all changes")
        _run_git(["add", "."], cwd=repo_dir, check=True)

        print(f"[git] Committing: {message}")
        _run_git(["commit", "-m", message], cwd=repo_dir, check=True)

        print("[git] Pushing")
        _run_git(["push"], cwd=repo_dir, check=True)

        print("[git] Done")
        return 0

    except subprocess.CalledProcessError as e:
        print("[error] commit-and-push failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return 1
    except Exception as e:
        print(f"[error] commit-and-push failed: {e}")
        return 1


# ----------------------- Interpretation Generation -----------------------

# Quintile thresholds from 2018-19 to 2024-25 (7 seasons of game-level data)
QUINTILE_THRESHOLDS_2018_25 = {
    "off_rating": {"p20": 102.7, "p40": 109.4, "p60": 115.1, "p80": 122.0},
    "def_rating": {"p20": 102.7, "p40": 109.4, "p60": 115.1, "p80": 122.0},
    "net_rating": {"p20": -12.1, "p40": -4.5, "p60": 4.5, "p80": 12.1},
    "efg": {"p20": 48.1, "p40": 51.9, "p60": 55.2, "p80": 59.3},
    "ball_handling": {"p20": 84.4, "p40": 86.5, "p60": 88.3, "p80": 90.2},
    "oreb": {"p20": 17.0, "p40": 21.2, "p60": 25.0, "p80": 29.4},
    "ft_rate": {"p20": 13.6, "p40": 17.5, "p60": 21.2, "p80": 25.8},
}


def _classify_quintile(value: float, thresholds: dict, higher_is_better: bool = True) -> str:
    """Classify a value into quintile label based on thresholds."""
    p20, p40, p60, p80 = thresholds["p20"], thresholds["p40"], thresholds["p60"], thresholds["p80"]

    if higher_is_better:
        if value <= p20:
            return "POOR"
        elif value <= p40:
            return "SUBPAR"
        elif value <= p60:
            return "AVERAGE"
        elif value <= p80:
            return "GOOD"
        else:
            return "EXCELLENT"
    else:
        # For metrics where lower is better (like defensive rating)
        if value >= p80:
            return "POOR"
        elif value >= p60:
            return "SUBPAR"
        elif value >= p40:
            return "AVERAGE"
        elif value >= p20:
            return "GOOD"
        else:
            return "EXCELLENT"


def generate_interpretations(
    season: str,
    repo_dir: Path,
    current_season: bool = False,
    incremental: bool = False,
    dry_run: bool = False,
    limit: int = None,
    max_new: int | None = None,
) -> int:
    """
    Generate LLM interpretations for all games in a season.

    Args:
        season: Season string (e.g., "2024-25")
        repo_dir: Path to NBA_Data repo
        current_season: If True, use current-season model (gpt-5.4); else use historical model (gpt-4o-mini)
        incremental: If True, only generate for games not already in output file
        dry_run: If True, show what would be generated without calling LLM
        limit: If set, only process this many games (for testing)
        max_new: If set, fail when the number of new games exceeds this value
    """
    try:
        repo_dir = ensure_data_repo(repo_dir)
        interpretations_dir = repo_dir / "interpretations"
        interpretations_dir.mkdir(parents=True, exist_ok=True)

        output_file = interpretations_dir / f"gamesummaries_{season}_2018-25.json"

        # Select model based on current_season flag
        model = LLM_MODELS["current"] if current_season else LLM_MODELS["historical"]
        print(f"[interp] Season: {season}")
        print(f"[interp] Model: {model}")
        print(f"[interp] Incremental: {incremental}")
        print(f"[interp] Output: {output_file}")

        # Load existing interpretations if incremental
        existing_data = {"season": season, "prompt_version": "v3_quintiles", "interpretations": {}}
        if incremental and output_file.exists():
            with open(output_file, "r") as f:
                existing_data = json.load(f)
            existing_interpretations = existing_data.get("interpretations", {})
            existing_data["interpretations"] = {
                _normalize_game_id(gid): payload
                for gid, payload in existing_interpretations.items()
                if _normalize_game_id(gid)
            }
            print(f"[interp] Loaded {len(existing_data.get('interpretations', {}))} existing interpretations")

        # Load season data
        csv_path = _resolve_repo_data_path(repo_dir, _season_to_filename(season))
        if not csv_path.exists():
            print(f"[error] Season CSV not found: {csv_path}")
            return 1

        df = pd.read_csv(csv_path, dtype={"game_id": "string"})
        df["game_id"] = df["game_id"].map(_normalize_game_id)
        df = df[df["game_id"] != ""].copy()
        print(f"[interp] Loaded {len(df)} games from {csv_path.name}")

        # Merge actual possessions from advanced stats
        adv_path = _resolve_repo_data_path(repo_dir, _advanced_filename(season))
        if adv_path.exists():
            adv_df = pd.read_csv(adv_path, dtype={"game_id": "string"})
            adv_df["game_id"] = adv_df["game_id"].map(_normalize_game_id)
            adv_df = adv_df[adv_df["game_id"] != ""].copy()
            df = df.merge(
                adv_df[["game_id", "possessions_home", "possessions_road", "minutes_home", "minutes_road"]],
                on="game_id",
                how="left",
            )

        # Load per-game contributions for this season (source of model-aligned contributions)
        contributions_path = repo_dir / "contributions" / f"contributions_{season}.json"
        if not contributions_path.exists():
            print(f"[error] Contribution file not found: {contributions_path}")
            return 1

        with open(contributions_path, "r") as f:
            contribution_payload = json.load(f)

        contribution_games = contribution_payload.get("games", [])
        contribution_by_game_id = {
            _normalize_game_id(g.get("game_id")): g
            for g in contribution_games
            if _normalize_game_id(g.get("game_id"))
        }
        decomposition_model_id = "json_contributions"
        print(f"[interp] Using decomposition model: {decomposition_model_id}")

        # Get list of games to process
        game_ids = [gid for gid in df["game_id"].astype(str).unique().tolist() if gid]
        existing_ids = set(existing_data.get("interpretations", {}).keys())

        if incremental:
            games_to_process = [gid for gid in game_ids if gid not in existing_ids]
        else:
            games_to_process = game_ids

        total_to_process = len(games_to_process)
        if max_new is not None and total_to_process > max_new:
            print(
                f"[error] Refusing to generate {total_to_process} interpretations "
                f"(max allowed: {max_new}). Run manually without --max-new for backfills."
            )
            return 2

        if limit:
            games_to_process = games_to_process[:limit]

        print(f"[interp] Games to process: {len(games_to_process)}")

        if dry_run:
            print("[interp] DRY RUN - would process these games:")
            for gid in games_to_process[:10]:
                print(f"  - {gid}")
            if len(games_to_process) > 10:
                print(f"  ... and {len(games_to_process) - 10} more")
            return 0

        # Process games
        success_count = 0
        fail_count = 0
        interpretations = existing_data.get("interpretations", {})

        for i, game_id in enumerate(games_to_process):
            game_id_str = _normalize_game_id(game_id)
            print(f"[interp] [{i+1}/{len(games_to_process)}] Processing game {game_id_str}...")

            try:
                # Get game row from dataframe
                game_row = df[df["game_id"] == game_id_str].iloc[0]

                contribution_entry = contribution_by_game_id.get(game_id_str)
                if contribution_entry is None:
                    print("    [warn] Missing contribution entry for game")
                    fail_count += 1
                    continue

                # Build game data in flat format with quintile classifications
                game_data = _build_game_data_with_quintiles(game_row, contribution_entry)

                # Generate interpretation (eight_factors only)
                interp_text = generate_interpretation_sync(
                    game_data,
                    "eight_factors",
                    model=model
                )

                if interp_text:
                    interpretations[game_id_str] = {
                        "generated_at": datetime.now().isoformat(),
                        "model": model,
                        "eight_factors": interp_text
                    }
                    success_count += 1
                else:
                    print(f"    [warn] Failed to generate interpretation")
                    fail_count += 1

                # Save progress every 50 games
                if (i + 1) % 50 == 0:
                    wrote = _save_interpretations(output_file, season, interpretations, decomposition_model_id)
                    if wrote:
                        print(f"[interp] Progress saved ({success_count} successful, {fail_count} failed)")
                    else:
                        print(f"[interp] Progress unchanged ({success_count} successful, {fail_count} failed)")

                # Rate limiting - small delay between calls
                time.sleep(0.5)

            except Exception as e:
                print(f"    [error] {e}")
                fail_count += 1

        # Final save
        wrote_final = _save_interpretations(output_file, season, interpretations, decomposition_model_id)

        print(f"[interp] Done! {success_count} successful, {fail_count} failed")
        if wrote_final:
            print(f"[interp] Output updated: {output_file}")
        else:
            print(f"[interp] Output unchanged: {output_file}")
        if fail_count > 0:
            # Surface partial-generation failures to callers (e.g. morning report script).
            print(
                f"[error] Interpretation generation completed with failures "
                f"(successful={success_count}, failed={fail_count})"
            )
            return 2
        return 0

    except Exception as e:
        print(f"[error] generate-interpretations failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def _build_game_data_with_quintiles(game_row: pd.Series, contribution_entry: Dict) -> Dict:
    """Build game data in flat format with quintile classifications for the new prompt."""
    home_team_row = {
        "fgm": game_row.get("fgm_home", 0),
        "fga": game_row.get("fga_home", 0),
        "fg3m": game_row.get("fg3m_home", 0),
        "ftm": game_row.get("ftm_home", 0),
        "fta": game_row.get("fta_home", 0),
        "oreb": game_row.get("oreb_home", 0),
        "dreb": game_row.get("dreb_home", 0),
        "tov": game_row.get("tov_home", 0),
        "pts": game_row.get("pts_home", 0),
    }

    road_team_row = {
        "fgm": game_row.get("fgm_road", 0),
        "fga": game_row.get("fga_road", 0),
        "fg3m": game_row.get("fg3m_road", 0),
        "ftm": game_row.get("ftm_road", 0),
        "fta": game_row.get("fta_road", 0),
        "oreb": game_row.get("oreb_road", 0),
        "dreb": game_row.get("dreb_road", 0),
        "tov": game_row.get("tov_road", 0),
        "pts": game_row.get("pts_road", 0),
    }

    actual_poss_home = game_row.get("possessions_home")
    actual_poss_road = game_row.get("possessions_road")
    actual_mins_home = game_row.get("minutes_home")
    actual_mins_road = game_row.get("minutes_road")

    home_factors = compute_four_factors(home_team_row, road_team_row, possessions=actual_poss_home)
    road_factors = compute_four_factors(road_team_row, home_team_row, possessions=actual_poss_road)

    home_ratings = compute_game_ratings(
        home_team_row, road_team_row,
        actual_possessions=actual_poss_home, opp_actual_possessions=actual_poss_road,
        actual_minutes=actual_mins_home,
    )
    road_ratings = compute_game_ratings(
        road_team_row, home_team_row,
        actual_possessions=actual_poss_road, opp_actual_possessions=actual_poss_home,
        actual_minutes=actual_mins_road,
    )

    # Use stored per-game contributions from the season contribution JSON
    factor_keys = ["shooting", "ball_handling", "orebounding", "free_throws"]
    home_factor_rows = contribution_entry.get("factors", {}).get("home", [])
    road_factor_rows = contribution_entry.get("factors", {}).get("road", [])
    contributions: Dict[str, float] = {}

    for i, factor_key in enumerate(factor_keys):
        home_contrib = home_factor_rows[i].get("contribution", 0) if i < len(home_factor_rows) else 0
        road_contrib = road_factor_rows[i].get("contribution", 0) if i < len(road_factor_rows) else 0
        contributions[f"home_{factor_key}"] = round(float(home_contrib), 2)
        contributions[f"road_{factor_key}"] = round(float(road_contrib), 2)

    # Build flat output with quintile classifications
    home_team = game_row.get("team_abbreviation_home", "")
    road_team = game_row.get("team_abbreviation_road", "")
    home_pts = int(game_row.get("pts_home", 0))
    road_pts = int(game_row.get("pts_road", 0))

    thresholds = QUINTILE_THRESHOLDS_2018_25

    return {
        "game_id": str(game_row.get("game_id", "")),
        "game_date": str(game_row.get("game_date", "")),
        "matchup": f"{road_team}@{home_team}",
        "score": f"{road_pts}-{home_pts}",
        "home_team": home_team,
        "road_team": road_team,
        "home_pts": home_pts,
        "road_pts": road_pts,
        "winner": home_team if home_pts > road_pts else road_team,
        "margin": abs(home_pts - road_pts),
        "model": contribution_entry.get("model", {}).get("model_id", "json_contributions"),

        # Home team ratings
        "home_off_rating": round(home_ratings["offensive_rating"], 1),
        "home_off_rating_class": _classify_quintile(home_ratings["offensive_rating"], thresholds["off_rating"]),
        "home_def_rating": round(home_ratings["defensive_rating"], 1),
        "home_def_rating_class": _classify_quintile(home_ratings["defensive_rating"], thresholds["def_rating"], higher_is_better=False),
        "home_net_rating": round(home_ratings["net_rating"], 1),
        "home_net_rating_class": _classify_quintile(home_ratings["net_rating"], thresholds["net_rating"]),

        # Road team ratings
        "road_off_rating": round(road_ratings["offensive_rating"], 1),
        "road_off_rating_class": _classify_quintile(road_ratings["offensive_rating"], thresholds["off_rating"]),
        "road_def_rating": round(road_ratings["defensive_rating"], 1),
        "road_def_rating_class": _classify_quintile(road_ratings["defensive_rating"], thresholds["def_rating"], higher_is_better=False),
        "road_net_rating": round(road_ratings["net_rating"], 1),
        "road_net_rating_class": _classify_quintile(road_ratings["net_rating"], thresholds["net_rating"]),

        # Home team factors
        "home_efg": round(home_factors["efg"], 1),
        "home_efg_class": _classify_quintile(home_factors["efg"], thresholds["efg"]),
        "home_efg_contrib": contributions.get("home_shooting", 0),

        "home_ball_handling": round(home_factors["ball_handling"], 1),
        "home_ball_handling_class": _classify_quintile(home_factors["ball_handling"], thresholds["ball_handling"]),
        "home_ball_handling_contrib": contributions.get("home_ball_handling", 0),

        "home_oreb": round(home_factors["oreb"], 1),
        "home_oreb_class": _classify_quintile(home_factors["oreb"], thresholds["oreb"]),
        "home_oreb_contrib": contributions.get("home_orebounding", 0),

        "home_ft_rate": round(home_factors["ft_rate"], 1),
        "home_ft_rate_class": _classify_quintile(home_factors["ft_rate"], thresholds["ft_rate"]),
        "home_ft_rate_contrib": contributions.get("home_free_throws", 0),

        # Road team factors
        "road_efg": round(road_factors["efg"], 1),
        "road_efg_class": _classify_quintile(road_factors["efg"], thresholds["efg"]),
        "road_efg_contrib": contributions.get("road_shooting", 0),

        "road_ball_handling": round(road_factors["ball_handling"], 1),
        "road_ball_handling_class": _classify_quintile(road_factors["ball_handling"], thresholds["ball_handling"]),
        "road_ball_handling_contrib": contributions.get("road_ball_handling", 0),

        "road_oreb": round(road_factors["oreb"], 1),
        "road_oreb_class": _classify_quintile(road_factors["oreb"], thresholds["oreb"]),
        "road_oreb_contrib": contributions.get("road_orebounding", 0),

        "road_ft_rate": round(road_factors["ft_rate"], 1),
        "road_ft_rate_class": _classify_quintile(road_factors["ft_rate"], thresholds["ft_rate"]),
        "road_ft_rate_contrib": contributions.get("road_free_throws", 0),
    }


def _save_interpretations(
    output_file: Path, season: str, interpretations: Dict, decomposition_model_id: str = None
) -> bool:
    """Save interpretations to JSON file only when content changes.

    Returns:
        bool: True when the file was written, False when content was unchanged.
    """
    data = {
        "season": season,
        "prompt_version": "v3_quintiles",
        "interpretations": interpretations,
    }
    # Store which decomposition model was used so API can check for match
    if decomposition_model_id:
        data["decomposition_model_id"] = decomposition_model_id
    serialized = json.dumps(data, indent=2)

    if output_file.exists():
        existing = output_file.read_text(encoding="utf-8")
        if existing == serialized or existing.rstrip("\n") == serialized:
            return False

    output_file.write_text(serialized, encoding="utf-8")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Admin CLI for NBA_Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from backend directory):
  python admin/cli.py update-data --season 2025-26
  python admin/cli.py download-data --start 2020-21 --end 2024-25
  python admin/cli.py backfill-pbp-raw --start 2000-01 --end 2025-26
  python admin/cli.py update-pbp-raw --season 2025-26 --phase both
  python admin/cli.py commit-and-push --message "Update data"
  python admin/cli.py git-status
""",
    )

    parser.add_argument(
        "--repo-dir",
        type=str,
        default=str(DEFAULT_REPO_DIR),
        help=(
            "Local path to the NBA_Data repo. If it doesn't exist, it will be cloned. "
            f"Default: {DEFAULT_REPO_DIR}"
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_update = sub.add_parser(
        "update-data",
        help="Update a single season's game log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python admin/cli.py update-data --season 2025-26

This fetches the latest game logs from the NBA API for the specified season
and updates the team_game_logs, linescores, and box_score_advanced
family folders in the NBA_Data repo.
""",
    )
    p_update.add_argument("--season", required=True, help="Season like 2025-26")
    p_update.add_argument("--force-refresh", action="store_true",
                          help="Update game_type for existing games (use after adding IST or fixing categorization)")

    p_dl = sub.add_parser(
        "download-data",
        help="Download/update a range of seasons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py download-data --start 2020-21 --end 2024-25
  python admin/cli.py download-data --start 2015-16 --end 2015-16  # single season

Downloads or updates game logs for all seasons in the specified range.
Each season creates/updates three CSV files under the NBA_Data family folders.
""",
    )
    p_dl.add_argument("--start", required=True, help="Start season like 2019-20")
    p_dl.add_argument("--end", required=True, help="End season like 2024-25")

    p_interp = sub.add_parser(
        "generate-interpretations",
        help="Generate LLM interpretations for games in a season",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py generate-interpretations --season 2023-24
  python admin/cli.py generate-interpretations --season 2024-25 --current
  python admin/cli.py generate-interpretations --season 2024-25 --current --incremental
  python admin/cli.py generate-interpretations --season 2024-25 --dry-run

Generates LLM interpretations for all games in a season and saves them to
NBA_Data/interpretations/{season}.json.

Use --current for the current season to use gpt-5.4.
Use --incremental to only generate for games not already in the output file.
Use --limit N to test with a small number of games.
Use --max-new N to fail when more than N new games would be generated.
""",
    )
    p_interp.add_argument("--season", required=True, help="Season like 2024-25")
    p_interp.add_argument("--current", action="store_true",
                          help="Use current-season model (gpt-5.4)")
    p_interp.add_argument("--incremental", action="store_true",
                          help="Only generate for games not already in output file")
    p_interp.add_argument("--dry-run", action="store_true",
                          help="Show what would be generated without calling LLM")
    p_interp.add_argument("--limit", type=int, default=None,
                          help="Limit number of games to process (for testing)")
    p_interp.add_argument("--max-new", type=int, default=None,
                          help="Fail when more than N new games would be generated")

    p_backfill_pbp = sub.add_parser(
        "backfill-pbp-raw",
        help="Backfill raw historical PBP into NBA_Data/PBPdata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py backfill-pbp-raw --start 2000-01 --end 2025-26
  python admin/cli.py backfill-pbp-raw --start 2020-21 --end 2025-26 --skip-cdnnba
  python admin/cli.py backfill-pbp-raw --start 2024-25 --end 2025-26 --dry-run

Reads existing historical files from shuf_datasets and writes raw CSVs to:
  NBA_Data/PBPdata/<source>/<regular|playoffs>/*.csv

No parsing or feature engineering is performed.
""",
    )
    p_backfill_pbp.add_argument("--start", required=True, help="Start season like 2000-01")
    p_backfill_pbp.add_argument("--end", required=True, help="End season like 2025-26")
    p_backfill_pbp.add_argument(
        "--source-dir",
        default=str(DEFAULT_SHUF_DATASETS_DIR),
        help=f"Directory containing historical PBP files (default: {DEFAULT_SHUF_DATASETS_DIR})",
    )
    p_backfill_pbp.add_argument("--skip-cdnnba", action="store_true", help="Do not include cdnnba historical files")
    p_backfill_pbp.add_argument("--overwrite", action="store_true", help="Overwrite existing destination CSV files")
    p_backfill_pbp.add_argument("--dry-run", action="store_true", help="Show planned actions only")

    p_update_pbp = sub.add_parser(
        "update-pbp-raw",
        help="Incrementally fetch missing raw PBP from nba_api into NBA_Data/PBPdata source files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py update-pbp-raw --season 2025-26
  python admin/cli.py update-pbp-raw --season 2025-26 --phase regular
  python admin/cli.py update-pbp-raw --season 2025-26 --phase both --max-games 20
  python admin/cli.py update-pbp-raw --season 2025-26 --target-source api_pbpv3
  python admin/cli.py update-pbp-raw --season 2025-26 --phase regular --max-games 20 --workers 8 --request-timeout 10 --retries 2

Default write target:
  NBA_Data/PBPdata/nbastatsv3/regular/nbastatsv3_YYYY.parquet
  NBA_Data/PBPdata/nbastatsv3/playoffs/nbastatsv3_po_YYYY.parquet
with a timestamped backup of existing files before overwrite.

Optional legacy target:
  NBA_Data/PBPdata/api_pbpv3/regular/api_pbpv3_YYYY.csv
  NBA_Data/PBPdata/api_pbpv3/playoffs/api_pbpv3_po_YYYY.csv

By default, only missing game IDs are fetched.
Use --overwrite-existing to refetch and replace all eligible games for the season/phase.
""",
    )
    p_update_pbp.add_argument("--season", required=True, help="Season like 2025-26")
    p_update_pbp.add_argument(
        "--migrate-nbastatsv3-to-parquet",
        action="store_true",
        help="Convert PBPdata/nbastatsv3/*.csv to .parquet and archive old CSVs",
    )
    p_update_pbp.add_argument(
        "--csv-archive-dir",
        default="PBPdata/nbastatsv3_csv_archive",
        help="Archive directory (relative to --repo-dir unless absolute) for old nbastatsv3 CSV files",
    )
    p_update_pbp.add_argument(
        "--target-source",
        choices=["nbastatsv3", "api_pbpv3"],
        default="nbastatsv3",
        help="Destination source folder to update (default: nbastatsv3)",
    )
    p_update_pbp.add_argument(
        "--phase",
        choices=["regular", "playoffs", "both"],
        default="both",
        help="Which phase to update (default: both)",
    )
    p_update_pbp.add_argument("--max-games", type=int, default=None, help="Optional cap on missing games to fetch")
    p_update_pbp.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Refetch and replace existing rows for all eligible game IDs (historical refresh mode)",
    )
    p_update_pbp.add_argument(
        "--workers",
        type=int,
        default=PBP_FETCH_MAX_WORKERS,
        help=f"Parallel fetch workers (default: {PBP_FETCH_MAX_WORKERS})",
    )
    p_update_pbp.add_argument(
        "--request-timeout",
        type=float,
        default=PBP_FETCH_TIMEOUT_SECONDS,
        help=f"Per-request timeout seconds (default: {PBP_FETCH_TIMEOUT_SECONDS})",
    )
    p_update_pbp.add_argument(
        "--retries",
        type=int,
        default=PBP_FETCH_RETRIES,
        help=f"Attempts per game across nba_api + cdnnba fallback (default: {PBP_FETCH_RETRIES})",
    )
    p_update_pbp.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup file creation when overwriting nbastatsv3 outputs",
    )
    p_update_pbp.add_argument("--dry-run", action="store_true", help="Show what would be fetched without API calls")

    p_fetch_pbp_game = sub.add_parser(
        "fetch-pbp-game",
        help="Fetch raw PBP for exactly one game_id and merge into a season source file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py fetch-pbp-game --season 2012-13 --game-id 0021201216 --phase regular
  python admin/cli.py fetch-pbp-game --season 2012-13 --game-id 0021201216 --phase regular --overwrite-game
  python admin/cli.py fetch-pbp-game --season 2025-26 --game-id 0022500001 --target-source api_pbpv3

This command is intended for targeted repairs of missing games.
Season format is always YYYY-YY.
""",
    )
    p_fetch_pbp_game.add_argument("--season", required=True, help="Season like 2012-13")
    p_fetch_pbp_game.add_argument("--game-id", required=True, help="10-digit game_id like 0021201216")
    p_fetch_pbp_game.add_argument(
        "--phase",
        choices=["regular", "playoffs"],
        default="regular",
        help="Which phase file to update (default: regular)",
    )
    p_fetch_pbp_game.add_argument(
        "--target-source",
        choices=["nbastatsv3", "api_pbpv3"],
        default="nbastatsv3",
        help="Destination source folder to update (default: nbastatsv3)",
    )
    p_fetch_pbp_game.add_argument(
        "--request-timeout",
        type=float,
        default=PBP_FETCH_TIMEOUT_SECONDS,
        help=f"Per-request timeout seconds (default: {PBP_FETCH_TIMEOUT_SECONDS})",
    )
    p_fetch_pbp_game.add_argument(
        "--retries",
        type=int,
        default=PBP_FETCH_RETRIES,
        help=f"Attempts across nba_api + cdnnba fallback (default: {PBP_FETCH_RETRIES})",
    )
    p_fetch_pbp_game.add_argument(
        "--overwrite-game",
        action="store_true",
        help="Replace existing rows for this game_id instead of skipping when already present",
    )
    p_fetch_pbp_game.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup file creation when writing nbastatsv3 outputs",
    )
    p_fetch_pbp_game.add_argument("--dry-run", action="store_true", help="Show what would be fetched without API calls")

    p_build_states = sub.add_parser(
        "build-pbp-game-states",
        help="Build per-event cumulative game-log states from raw PBP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py build-pbp-game-states --season 2023-24 --phase regular
  python admin/cli.py build-pbp-game-states --season 2023-24 --phase regular --max-games 10
  python admin/cli.py build-pbp-game-states --season 2023-24 --game-id 0022300001 --overwrite

Writes one JSON per game, with a full cumulative game-log state after each PBP row.
Final event state is validated against team_game_logs season totals.
""",
    )
    p_build_states.add_argument("--season", required=True, help="Season like 2023-24")
    p_build_states.add_argument(
        "--phase",
        choices=["regular", "playoffs"],
        default="regular",
        help="Which phase to process (default: regular)",
    )
    p_build_states.add_argument(
        "--pbp-source",
        choices=["auto", "nbastatsv3", "api_pbpv3"],
        default="auto",
        help=(
            "PBP source to parse. 'auto' prefers nbastatsv3 first, verifies "
            "coverage, and only falls back to api_pbpv3 when nbastatsv3 is incomplete."
        ),
    )
    p_build_states.add_argument("--output-root", default=None, help="Optional root directory for output JSON files")
    p_build_states.add_argument("--max-games", type=int, default=None, help="Optional max number of games to process")
    p_build_states.add_argument("--game-id", default=None, help="Optional specific game_id to process")
    p_build_states.add_argument("--overwrite", action="store_true", help="Overwrite existing per-game output files")

    p_check_pbp = sub.add_parser(
        "check-pbp-completeness",
        help="Check whether a stored raw PBP source covers every expected game in team_game_logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py check-pbp-completeness --season 2025-26 --phase regular
  python admin/cli.py check-pbp-completeness --season 2025-26 --phase playoffs --pbp-source nbastatsv3
  python admin/cli.py check-pbp-completeness --season 2025-26 --phase regular --pbp-source api_pbpv3
""",
    )
    p_check_pbp.add_argument("--season", required=True, help="Season like 2025-26")
    p_check_pbp.add_argument(
        "--phase",
        choices=["regular", "playoffs"],
        default="regular",
        help="Which phase to inspect (default: regular)",
    )
    p_check_pbp.add_argument(
        "--pbp-source",
        choices=["nbastatsv3", "api_pbpv3"],
        default="nbastatsv3",
        help="Stored raw PBP source to check (default: nbastatsv3)",
    )
    p_check_pbp.add_argument(
        "--list-limit",
        type=int,
        default=20,
        help="Maximum number of missing/extra game IDs to print (default: 20)",
    )

    p_player_shots = sub.add_parser(
        "build-player-shots",
        help="Build one-row-per-shot player shooting datasets from raw PBP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py build-player-shots --season 2025-26 --phase regular
  python admin/cli.py build-player-shots --season 2025-26 --phase both --overwrite
  python admin/cli.py build-player-shots --season all --phase both

Output:
  NBA_Data/player_shots/player_shots_YYYY-YY.parquet
""",
    )
    p_player_shots.add_argument("--season", required=True, help="Season like 2025-26, or all")
    p_player_shots.add_argument(
        "--phase",
        choices=["regular", "playoffs", "both"],
        default="both",
        help="Which phase to process (default: both)",
    )
    p_player_shots.add_argument(
        "--pbp-source",
        choices=["auto", "nbastatsv3", "api_pbpv3"],
        default="auto",
        help="PBP source to parse (default: auto)",
    )
    p_player_shots.add_argument("--output-root", default=None, help="Optional root directory for output parquet files")
    p_player_shots.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    p_pack_states = sub.add_parser(
        "pack-pbp-game-states",
        help="Pack per-game game-state JSON files into one parquet per season/phase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py pack-pbp-game-states --season 2023-24 --phase regular
  python admin/cli.py pack-pbp-game-states --season all --phase both --compression zstd --overwrite
  python admin/cli.py pack-pbp-game-states --season 2023-24 --phase playoffs --delete-json

Input defaults to:
  <repo-dir>/PBPdata/game_states/<phase>/<season>/*.json
Output defaults to:
  <repo-dir>/PBPdata/game_states/<phase>/<season>/_states_<season>_<phase>.parquet
""",
    )
    p_pack_states.add_argument("--season", required=True, help="Season like 2023-24, or 'all'")
    p_pack_states.add_argument(
        "--phase",
        choices=["regular", "playoffs", "both"],
        default="regular",
        help="Which phase to pack (default: regular)",
    )
    p_pack_states.add_argument(
        "--input-root",
        default=None,
        help=(
            "Optional game-state root. Accepted shapes: "
            "<root>/<phase>/<season> or <root>/<season> or <root>."
        ),
    )
    p_pack_states.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional game-state output root. Default writes next to input season directories."
        ),
    )
    p_pack_states.add_argument(
        "--compression",
        choices=["zstd", "snappy", "gzip", "none"],
        default="zstd",
        help="Parquet compression codec (default: zstd)",
    )
    p_pack_states.add_argument("--overwrite", action="store_true", help="Overwrite existing packed parquet file")
    p_pack_states.add_argument(
        "--delete-json",
        action="store_true",
        help="Delete per-game JSON files after successful pack",
    )

    p_timeline_metrics = sub.add_parser(
        "build-pbp-timeline-metrics",
        help="Build per-game excitement/comeback metrics from game-state timelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py build-pbp-timeline-metrics --season 2025-26 --phase regular
  python admin/cli.py build-pbp-timeline-metrics --season all --phase both --overwrite
  python admin/cli.py build-pbp-timeline-metrics --season 2024-25 --input-root /path/to/game_states

Reads from:
  <repo-dir>/PBPdata/game_states/<phase>/<season>
and writes:
  <repo-dir>/PBPdata/game_states/<phase>/<season>/_timeline_metrics_<season>_<phase>.json
""",
    )
    p_timeline_metrics.add_argument("--season", required=True, help="Season like 2025-26, or 'all'")
    p_timeline_metrics.add_argument(
        "--phase",
        choices=["regular", "playoffs", "both"],
        default="regular",
        help="Which phase to process (default: regular)",
    )
    p_timeline_metrics.add_argument(
        "--input-root",
        default=None,
        help=(
            "Optional game-state root. Accepted shapes: "
            "<root>/<phase>/<season> or <root>/<season> or <root>."
        ),
    )
    p_timeline_metrics.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional output root. Files are written to: "
            "<output-root>/<phase>/<season>/_timeline_metrics_<season>_<phase>.json"
        ),
    )
    p_timeline_metrics.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metrics JSON if present",
    )

    p_winprob_base = sub.add_parser(
        "build-pbp-winprob-base",
        help="Build stacked win-probability baseline CSV from packed game states or JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py build-pbp-winprob-base --season 2023-24
  python admin/cli.py build-pbp-winprob-base --season 2023-24 --phase regular --overwrite
  python admin/cli.py build-pbp-winprob-base --season 2023-24 --input-root /path/to/game_states
  python admin/cli.py build-pbp-winprob-base --season 2023-24 --output-root /path/to/winprob_base

By default this reads from:
  <repo-dir>/PBPdata/game_states/<phase>/<season>
preferring packed parquet:
  _states_<season>_<phase>.parquet
and falling back to per-game JSON files when parquet is absent.
and writes to:
  <repo-dir>/PBPdata/winprob_base/<phase>/stacked_<season>_winprob_base.csv
""",
    )
    p_winprob_base.add_argument("--season", required=True, help="Season like 2023-24")
    p_winprob_base.add_argument(
        "--phase",
        choices=["regular", "playoffs"],
        default="regular",
        help="Which phase to process (default: regular)",
    )
    p_winprob_base.add_argument(
        "--input-root",
        default=None,
        help=(
            "Optional game-state root. Accepted shapes: "
            "<root>/<phase>/<season> or <root>/<season> or <root>."
        ),
    )
    p_winprob_base.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional root directory for output CSV. "
            "CSV path becomes <output-root>/<phase>/stacked_<season>_winprob_base.csv"
        ),
    )
    p_winprob_base.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSV if present",
    )

    default_winprob_input_root = str(PBP_WINPROB_BASE_ROOT)
    default_wpm_output_root = str(PBP_WINPROB_MODELS_ROOT)

    p_wpm = sub.add_parser(
        "build-pbp-winprob-models",
        help="Train season-specific win probability models and export JSON artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python admin/cli.py build-pbp-winprob-models --season 2025-26
  python admin/cli.py build-pbp-winprob-models --lookback-seasons 3 --overwrite
  python admin/cli.py build-pbp-winprob-models --input-root {default_winprob_input_root}

Training rule:
  - 2000-01: in-sample training on 2000-01
  - Later seasons: train on up to 3 prior seasons (no in-season rows from target season)
""",
    )
    p_wpm.add_argument("--season", default=None, help="Optional single target season like 2025-26")
    p_wpm.add_argument(
        "--phase",
        choices=["regular", "playoffs"],
        default="regular",
        help="Phase namespace for input data path (default: regular). Artifacts are written flat under --output-root.",
    )
    p_wpm.add_argument(
        "--lookback-seasons",
        type=int,
        default=3,
        help="Number of prior seasons for training window (default: 3)",
    )
    p_wpm.add_argument(
        "--input-root",
        default=default_winprob_input_root,
        help="Root directory containing stacked winprob CSVs",
    )
    p_wpm.add_argument(
        "--output-root",
        default=default_wpm_output_root,
        help="Root directory for WPM JSON artifacts",
    )
    p_wpm.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing model artifacts if present",
    )

    p_wpm_predict = sub.add_parser(
        "predict-pbp-winprob",
        help="Predict home win probability from a season WPM artifact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python admin/cli.py predict-pbp-winprob --season 2025-26 --quarter 4 --seconds-left 120 --differential 3 --possession-numeric 1
  python admin/cli.py predict-pbp-winprob --season 2025-26 --quarter 2 --seconds-left 300 --differential -5 --home-team BOS --road-team NYK --possession BOS
  python admin/cli.py predict-pbp-winprob --season 2025-26 --output-root {default_wpm_output_root}
""",
    )
    p_wpm_predict.add_argument("--season", required=True, help="Season like 2025-26")
    p_wpm_predict.add_argument(
        "--phase",
        choices=["regular", "playoffs"],
        default="regular",
        help="Legacy artifact subfolder fallback (default: regular). New artifacts are read from flat --output-root paths.",
    )
    p_wpm_predict.add_argument(
        "--output-root",
        default=default_wpm_output_root,
        help="Root directory containing WPM JSON artifacts",
    )
    p_wpm_predict.add_argument("--quarter", type=int, required=True, help="Quarter number (1.., OT > 4)")
    p_wpm_predict.add_argument("--seconds-left", type=float, required=True, help="Seconds left in current period")
    p_wpm_predict.add_argument("--differential", type=float, required=True, help="Home score minus road score at current moment")
    p_wpm_predict.add_argument(
        "--possession-numeric",
        type=int,
        default=None,
        choices=[-1, 0, 1],
        help="Optional possession encoding (home=1, road=-1, unknown=0). Overrides string possession mapping.",
    )
    p_wpm_predict.add_argument("--home-team", default=None, help="Optional home team tricode (used with --possession)")
    p_wpm_predict.add_argument("--road-team", default=None, help="Optional road team tricode (used with --possession)")
    p_wpm_predict.add_argument("--possession", default=None, help="Optional team tricode with possession")

    sub.add_parser(
        "git-status",
        help="Show git status (short) in NBA_Data repo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python admin/cli.py git-status

Shows uncommitted changes in the NBA_Data repository.
""",
    )

    p_cap = sub.add_parser(
        "commit-and-push",
        help="Commit+push changes to NBA_Data repo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py commit-and-push --message "Update 2025-26 data"
  python admin/cli.py commit-and-push --message "Regenerate contributions" --dry-run

Commits all changes in the NBA_Data repo and pushes to GitHub.
Use --dry-run to preview changes without committing.
""",
    )
    p_cap.add_argument("--message", required=True, help="Commit message")
    p_cap.add_argument("--dry-run", action="store_true", help="Print what would happen without committing")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_dir = Path(args.repo_dir)

    if args.command == "update-data":
        return update_data(args.season, repo_dir, force_refresh=args.force_refresh)

    if args.command == "download-data":
        return download_data(args.start, args.end, repo_dir)

    if args.command == "generate-interpretations":
        return generate_interpretations(
            args.season,
            repo_dir,
            current_season=args.current,
            incremental=args.incremental,
            dry_run=args.dry_run,
            limit=args.limit,
            max_new=args.max_new,
        )

    if args.command == "backfill-pbp-raw":
        return backfill_pbp_raw(
            start_season=args.start,
            end_season=args.end,
            repo_dir=repo_dir,
            source_dir=Path(args.source_dir),
            include_cdnnba=not args.skip_cdnnba,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    if args.command == "update-pbp-raw":
        return update_pbp_raw(
            season=args.season,
            repo_dir=repo_dir,
            season_phase=args.phase,
            target_source=args.target_source,
            backup_existing=not args.no_backup,
            overwrite_existing=args.overwrite_existing,
            max_games=args.max_games,
            request_timeout=args.request_timeout,
            retries=args.retries,
            max_workers=args.workers,
            migrate_nbastatsv3_to_parquet=args.migrate_nbastatsv3_to_parquet,
            csv_archive_dir=args.csv_archive_dir,
            dry_run=args.dry_run,
        )

    if args.command == "fetch-pbp-game":
        return fetch_pbp_game(
            season=args.season,
            game_id=args.game_id,
            repo_dir=repo_dir,
            season_phase=args.phase,
            target_source=args.target_source,
            backup_existing=not args.no_backup,
            request_timeout=args.request_timeout,
            retries=args.retries,
            overwrite_game=args.overwrite_game,
            dry_run=args.dry_run,
        )

    if args.command == "build-pbp-game-states":
        from admin.pbp_game_states import build_pbp_game_states

        return build_pbp_game_states(
            season=args.season,
            repo_dir=repo_dir,
            phase=args.phase,
            source=args.pbp_source,
            output_root=args.output_root,
            max_games=args.max_games,
            game_id=args.game_id,
            overwrite=args.overwrite,
        )

    if args.command == "check-pbp-completeness":
        return check_pbp_completeness_cmd(
            season=args.season,
            repo_dir=repo_dir,
            phase=args.phase,
            pbp_source=args.pbp_source,
            list_limit=args.list_limit,
        )

    if args.command == "build-player-shots":
        from admin.player_shots import build_player_shots

        return build_player_shots(
            season=args.season,
            repo_dir=repo_dir,
            phase=args.phase,
            pbp_source=args.pbp_source,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )

    if args.command == "pack-pbp-game-states":
        return pack_pbp_game_states_cmd(
            season=args.season,
            repo_dir=repo_dir,
            phase=args.phase,
            input_root=args.input_root,
            output_root=args.output_root,
            compression=args.compression,
            overwrite=args.overwrite,
            delete_json=args.delete_json,
        )

    if args.command == "build-pbp-timeline-metrics":
        return build_pbp_timeline_metrics_cmd(
            season=args.season,
            repo_dir=repo_dir,
            phase=args.phase,
            input_root=args.input_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )

    if args.command == "build-pbp-winprob-base":
        from admin.pbp_game_states import build_winprob_base

        return build_winprob_base(
            season=args.season,
            repo_dir=repo_dir,
            phase=args.phase,
            input_root=args.input_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )

    if args.command == "build-pbp-winprob-models":
        from admin.winprob_models import build_winprob_models

        return build_winprob_models(
            input_root=args.input_root,
            output_root=args.output_root,
            phase=args.phase,
            season=args.season,
            lookback_seasons=args.lookback_seasons,
            overwrite=args.overwrite,
        )

    if args.command == "predict-pbp-winprob":
        from admin.winprob_models import predict_winprob

        return predict_winprob(
            season=args.season,
            output_root=args.output_root,
            phase=args.phase,
            quarter=args.quarter,
            seconds_left=args.seconds_left,
            differential=args.differential,
            possession_numeric=args.possession_numeric,
            possession=args.possession,
            home_team=args.home_team,
            road_team=args.road_team,
        )

    if args.command == "git-status":
        return git_status(repo_dir)

    if args.command == "commit-and-push":
        return commit_and_push(args.message, repo_dir, dry_run=args.dry_run)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
