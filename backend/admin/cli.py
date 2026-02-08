#!/usr/bin/env python3
"""admin/cli.py

Standalone admin CLI for the NBA_Data repository.

Responsibilities:
- Ensure the NBA_Data repo exists locally (clone if missing)
- Update/download season CSVs in NBA_Data repo root in *game-level* schema
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
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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
from config import get_current_season


NBA_DATA_REPO_URL = "https://github.com/millxing/NBA_Data"

# Default: use the user's canonical NBA_Data folder.
# This keeps GLA_Admin fully separate from any NBA/ NBA_alpha folders.
DEFAULT_REPO_DIR = Path("/Users/robschoen/Dropbox/CC/NBA_Data").resolve()


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
    return f"team_game_logs_{season}.csv"


def _linescore_filename(season: str) -> str:
    return f"linescores_{season}.csv"


def _advanced_filename(season: str) -> str:
    return f"box_score_advanced_{season}.csv"


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
            advanced_rows.append(adv_row)
            print("ADV:OK")
        elif gid in ADVANCED_FALLBACK_DATA:
            advanced_rows.append(ADVANCED_FALLBACK_DATA[gid].copy())
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
                combined_adv.to_csv(advanced_path, index=False)
                existing_adv = combined_adv  # Update for next iteration
                adv_total_added += len(advanced_rows)
                advanced_rows = []  # Reset buffer

    return ls_total_added, adv_total_added


# ------------------------- CLI commands -------------------------

def update_data(season: str, repo_dir: Path, force_refresh: bool = False) -> int:
    start = time.time()
    try:
        repo_dir = ensure_data_repo(repo_dir)
        csv_path = repo_dir / _season_to_filename(season)

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
        linescore_path = repo_dir / _linescore_filename(season)
        advanced_path = repo_dir / _advanced_filename(season)

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
        adv_total = len(pd.read_csv(advanced_path)) if advanced_path.exists() else 0

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
        current_season: If True, use better model (Sonnet 4); else use cheaper model (GPT-4o-mini)
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
        csv_path = repo_dir / _season_to_filename(season)
        if not csv_path.exists():
            print(f"[error] Season CSV not found: {csv_path}")
            return 1

        df = pd.read_csv(csv_path, dtype={"game_id": "string"})
        df["game_id"] = df["game_id"].map(_normalize_game_id)
        df = df[df["game_id"] != ""].copy()
        print(f"[interp] Loaded {len(df)} games from {csv_path.name}")

        # Merge actual possessions from advanced stats
        adv_path = repo_dir / _advanced_filename(season)
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
                    _save_interpretations(output_file, season, interpretations, decomposition_model_id)
                    print(f"[interp] Progress saved ({success_count} successful, {fail_count} failed)")

                # Rate limiting - small delay between calls
                time.sleep(0.5)

            except Exception as e:
                print(f"    [error] {e}")
                fail_count += 1

        # Final save
        _save_interpretations(output_file, season, interpretations, decomposition_model_id)

        print(f"[interp] Done! {success_count} successful, {fail_count} failed")
        print(f"[interp] Output: {output_file}")
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
) -> None:
    """Save interpretations to JSON file."""
    data = {
        "season": season,
        "prompt_version": "v3_quintiles",
        "interpretations": interpretations,
    }
    # Store which decomposition model was used so API can check for match
    if decomposition_model_id:
        data["decomposition_model_id"] = decomposition_model_id
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Admin CLI for NBA_Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from backend directory):
  python admin/cli.py update-data --season 2025-26
  python admin/cli.py download-data --start 2020-21 --end 2024-25
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
and updates team_game_logs_YYYY-YY.csv, linescores_YYYY-YY.csv, and
box_score_advanced_YYYY-YY.csv in the NBA_Data repo.
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
Each season creates/updates three CSV files in the NBA_Data repo.
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

Use --current for the current season to use a better model (Sonnet 4).
Use --incremental to only generate for games not already in the output file.
Use --limit N to test with a small number of games.
Use --max-new N to fail when more than N new games would be generated.
""",
    )
    p_interp.add_argument("--season", required=True, help="Season like 2024-25")
    p_interp.add_argument("--current", action="store_true",
                          help="Use better model (Sonnet 4) for current season")
    p_interp.add_argument("--incremental", action="store_true",
                          help="Only generate for games not already in output file")
    p_interp.add_argument("--dry-run", action="store_true",
                          help="Show what would be generated without calling LLM")
    p_interp.add_argument("--limit", type=int, default=None,
                          help="Limit number of games to process (for testing)")
    p_interp.add_argument("--max-new", type=int, default=None,
                          help="Fail when more than N new games would be generated")

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

    if args.command == "git-status":
        return git_status(repo_dir)

    if args.command == "commit-and-push":
        return commit_and_push(args.message, repo_dir, dry_run=args.dry_run)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
