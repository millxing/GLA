#!/usr/bin/env python3
"""admin/cli.py

Standalone admin CLI for the NBA_Data repository.

Responsibilities:
- Ensure the NBA_Data repo exists locally (clone if missing)
- Update/download season CSVs in NBA_Data repo root in *game-level* schema
  (one row per game with *_home and *_road columns)
- Train simple Four-Factors / Eight-Factors style linear models and save JSONs
  into NBA_Data/models/
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
  python admin/cli.py train-models --seasons 2024-25,2025-26 --output latest.json
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
from typing import Callable, Dict, Optional, TypeVar

import pandas as pd

# Data pulls
from nba_api.stats.endpoints import leaguegamelog, boxscoresummaryv3, boxscoreadvancedv3

# Modeling
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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

    (repo_dir / "models").mkdir(parents=True, exist_ok=True)
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

        # Fetch advanced stats
        adv_row = _fetch_advanced_stats(gid, game_date, season, home_team_id)
        if adv_row:
            advanced_rows.append(adv_row)
            print("ADV:OK")
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


# ------------------------- modeling helpers -------------------------

def _gamelogs_to_teamrows(game_df: pd.DataFrame, adv_df: pd.DataFrame = None) -> pd.DataFrame:
    """Expand game-level rows into team-level rows for modeling.

    If adv_df is provided, merge possession data from box_score_advanced.
    """
    df = _normalize_game_level_df(game_df)

    # Merge advanced stats (possessions) if available
    if adv_df is not None and not adv_df.empty:
        adv_df = adv_df.copy()
        # Normalize game_id to 10-char with leading zeros (same as _normalize_game_level_df)
        adv_df["game_id"] = adv_df["game_id"].astype(str).map(
            lambda v: v.zfill(10) if v.isdigit() else v
        )
        df = df.merge(
            adv_df[["game_id", "possessions_home", "possessions_road"]],
            on="game_id",
            how="left"
        )

    # Home team rows
    home_data = {
        "GAME_ID": df["game_id"].astype(str),
        "TEAM_ID": df["team_id_home"],
        "TEAM_ABBREVIATION": df["team_abbreviation_home"],
        "TEAM_NAME": df["team_name_home"],
        "GAME_DATE": pd.to_datetime(df["game_date"], errors="coerce"),
        "MATCHUP": df["team_abbreviation_home"].astype(str) + " vs. " + df["team_abbreviation_road"].astype(str),
        "PLUS_MINUS": df["plus_minus_home"],
        "PTS": df["pts_home"],
        "FGM": df["fgm_home"],
        "FGA": df["fga_home"],
        "FG3M": df["fg3m_home"],
        "FG3A": df["fg3a_home"],
        "FTM": df["ftm_home"],
        "FTA": df["fta_home"],
        "OREB": df["oreb_home"],
        "DREB": df["dreb_home"],
        "REB": df["reb_home"],
        "AST": df["ast_home"],
        "TOV": df["tov_home"],
        "STL": df["stl_home"],
        "BLK": df["blk_home"],
        "PF": df["pf_home"],
    }
    # Add possession columns if available
    if "possessions_home" in df.columns:
        home_data["POSS"] = df["possessions_home"]
        home_data["OPP_POSS"] = df["possessions_road"]
    home = pd.DataFrame(home_data)

    # Road team rows
    road_data = {
        "GAME_ID": df["game_id"].astype(str),
        "TEAM_ID": df["team_id_road"],
        "TEAM_ABBREVIATION": df["team_abbreviation_road"],
        "TEAM_NAME": df["team_name_road"],
        "GAME_DATE": pd.to_datetime(df["game_date"], errors="coerce"),
        "MATCHUP": df["team_abbreviation_road"].astype(str) + " @ " + df["team_abbreviation_home"].astype(str),
        "PLUS_MINUS": df["plus_minus_road"],
        "PTS": df["pts_road"],
        "FGM": df["fgm_road"],
        "FGA": df["fga_road"],
        "FG3M": df["fg3m_road"],
        "FG3A": df["fg3a_road"],
        "FTM": df["ftm_road"],
        "FTA": df["fta_road"],
        "OREB": df["oreb_road"],
        "DREB": df["dreb_road"],
        "REB": df["reb_road"],
        "AST": df["ast_road"],
        "TOV": df["tov_road"],
        "STL": df["stl_road"],
        "BLK": df["blk_road"],
        "PF": df["pf_road"],
    }
    # Add possession columns if available
    if "possessions_road" in df.columns:
        road_data["POSS"] = df["possessions_road"]
        road_data["OPP_POSS"] = df["possessions_home"]
    road = pd.DataFrame(road_data)

    out = pd.concat([home, road], ignore_index=True)

    # Numeric conversions
    for c in [
        "PTS","FGM","FGA","FG3M","FG3A","FTM","FTA",
        "OREB","DREB","REB","AST","TOV","STL","BLK","PF","PLUS_MINUS"
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _estimate_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate possessions from box score stats when actual data is unavailable.

    Uses the standard formula: POSS ≈ FGA - OREB + TOV + 0.44 * FTA

    This allows training models on seasons without advanced stats data.
    Only fills in missing POSS values; preserves actual possession data where available.
    """
    out = df.copy()

    # Check if we have the columns needed for estimation
    required_cols = {"FGA", "OREB", "TOV", "FTA"}
    if not required_cols.issubset(out.columns):
        return out  # Can't estimate without basic box score data

    # Ensure numeric
    for col in required_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Estimate possessions: FGA - OREB + TOV + 0.44 * FTA
    estimated_poss = out["FGA"] - out["OREB"] + out["TOV"] + 0.44 * out["FTA"]

    # If POSS column doesn't exist, create it with estimates
    if "POSS" not in out.columns:
        out["POSS"] = estimated_poss
        print(f"[estimate-poss] Created POSS column with {estimated_poss.notna().sum()} estimated values")
    else:
        # Fill only missing values with estimates
        missing_count = out["POSS"].isna().sum()
        if missing_count > 0:
            out["POSS"] = out["POSS"].fillna(estimated_poss)
            print(f"[estimate-poss] Filled {missing_count} missing POSS values with estimates")

    # Also estimate OPP_POSS if OPP_ columns exist
    opp_required = {"OPP_FGA", "OPP_OREB", "OPP_TOV", "OPP_FTA"}
    if opp_required.issubset(out.columns):
        for col in opp_required:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        estimated_opp_poss = out["OPP_FGA"] - out["OPP_OREB"] + out["OPP_TOV"] + 0.44 * out["OPP_FTA"]

        if "OPP_POSS" not in out.columns:
            out["OPP_POSS"] = estimated_opp_poss
        else:
            missing_opp = out["OPP_POSS"].isna().sum()
            if missing_opp > 0:
                out["OPP_POSS"] = out["OPP_POSS"].fillna(estimated_opp_poss)
                print(f"[estimate-poss] Filled {missing_opp} missing OPP_POSS values with estimates")

    return out


def _compute_four_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Four Factors using common boxscore columns.

    Requires (or expects):
      FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, TOV

    Outputs:
      EFG_PCT, TOV_PCT, OREB_PCT, FT_RATE
    """
    out = df.copy()

    # Defensive conversion
    for col in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "TOV"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # eFG% = (FGM + 0.5*3PM) / FGA
    if {"FGM", "FG3M", "FGA"}.issubset(out.columns):
        out["EFG_PCT"] = (out["FGM"] + 0.5 * out["FG3M"]) / out["FGA"].replace(0, pd.NA)

    # TOV% = TOV / (FGA + 0.44*FTA + TOV)
    if {"TOV", "FGA", "FTA"}.issubset(out.columns):
        denom = out["FGA"] + 0.44 * out["FTA"] + out["TOV"]
        out["TOV_PCT"] = out["TOV"] / denom.replace(0, pd.NA)

    # OREB% requires opponent DREB, which is not directly present.
    # We'll compute a proxy later when we attach opponent rows; for now keep team OREB.
    # FT Rate = FTM / FGA (or FTA/FGA; using FTM/FGA is common in 4 factors contexts)
    if {"FTM", "FGA"}.issubset(out.columns):
        out["FT_RATE"] = out["FTM"] / out["FGA"].replace(0, pd.NA)

    return out


def _compute_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute offensive, defensive, and net ratings.

    Requires: PTS, POSS, OPP_PTS, OPP_POSS columns.
    Outputs: OFF_RATING, DEF_RATING, NET_RATING
    """
    out = df.copy()

    # Check required columns
    if not {"PTS", "POSS", "OPP_POSS"}.issubset(out.columns):
        return out  # Cannot compute without possession data

    # Ensure numeric
    for col in ["PTS", "POSS", "OPP_POSS"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Offensive Rating = (PTS / POSS) * 100
    out["OFF_RATING"] = (out["PTS"] / out["POSS"].replace(0, pd.NA)) * 100

    # Defensive Rating requires opponent points - will be computed after opponent merge
    # For now, just compute OFF_RATING

    return out


def _attach_opponent_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent boxscore columns by self-joining on GAME_ID.

    Assumes the data has 2 rows per GAME_ID.
    Adds columns prefixed with OPP_.
    """
    if "GAME_ID" not in df.columns or "TEAM_ID" not in df.columns:
        return df

    cols_to_copy = [
        "TEAM_ID",
        "PTS",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "TOV",
        "STL",
        "BLK",
        "PF",
        "PLUS_MINUS",
    ]
    cols_to_copy = [c for c in cols_to_copy if c in df.columns]

    opp = df[["GAME_ID", *cols_to_copy]].copy()
    opp = opp.rename(columns={c: f"OPP_{c}" for c in cols_to_copy if c != "TEAM_ID"})
    opp = opp.rename(columns={"TEAM_ID": "OPP_TEAM_ID"})

    # Merge all possible opponent rows; we'll filter out self-matchups.
    merged = df.merge(opp, on="GAME_ID", how="left")

    # Remove self matches (where OPP_TEAM_ID == TEAM_ID)
    if "OPP_TEAM_ID" in merged.columns:
        merged = merged[merged["OPP_TEAM_ID"] != merged["TEAM_ID"]]

    # In case of weird duplicates (shouldn't happen), keep first.
    merged = merged.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])

    return merged


def _compute_factor_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute differentials for factors between team and opponent.

    Adds:
      EFG_DIFF, TOV_DIFF, FT_DIFF, OREB_DIFF

    Also computes OREB% once opponent DREB is available.
    """
    out = df.copy()

    # Ensure opponent rows attached
    if "OPP_DREB" in out.columns and "OREB" in out.columns:
        # OREB% = OREB / (OREB + Opp DREB)
        denom = out["OREB"] + out["OPP_DREB"]
        out["OREB_PCT"] = out["OREB"] / denom.replace(0, pd.NA)

    # Opponent four factors
    if {"OPP_FGM", "OPP_FG3M", "OPP_FGA"}.issubset(out.columns):
        out["OPP_EFG_PCT"] = (out["OPP_FGM"] + 0.5 * out["OPP_FG3M"]) / out["OPP_FGA"].replace(0, pd.NA)

    if {"OPP_TOV", "OPP_FGA", "OPP_FTA"}.issubset(out.columns):
        denom = out["OPP_FGA"] + 0.44 * out["OPP_FTA"] + out["OPP_TOV"]
        out["OPP_TOV_PCT"] = out["OPP_TOV"] / denom.replace(0, pd.NA)

    if {"OPP_FTM", "OPP_FGA"}.issubset(out.columns):
        out["OPP_FT_RATE"] = out["OPP_FTM"] / out["OPP_FGA"].replace(0, pd.NA)

    # Differentials
    if {"EFG_PCT", "OPP_EFG_PCT"}.issubset(out.columns):
        out["EFG_DIFF"] = out["EFG_PCT"] - out["OPP_EFG_PCT"]

    if {"TOV_PCT", "OPP_TOV_PCT"}.issubset(out.columns):
        # Lower TOV% is better, so differential is (opp - team)
        out["TOV_DIFF"] = out["OPP_TOV_PCT"] - out["TOV_PCT"]

    if {"FT_RATE", "OPP_FT_RATE"}.issubset(out.columns):
        out["FT_DIFF"] = out["FT_RATE"] - out["OPP_FT_RATE"]

    if {"OREB_PCT", "OPP_DREB"}.issubset(out.columns) and "OPP_OREB" in out.columns:
        # Opponent OREB% proxy (requires our DREB)
        denom = out["OPP_OREB"] + out.get("DREB", pd.Series(pd.NA, index=out.index))
        opp_oreb_pct = out["OPP_OREB"] / denom.replace(0, pd.NA)
        out["OREB_DIFF"] = out["OREB_PCT"] - opp_oreb_pct

    # Compute net rating and differential (requires possessions)
    if {"PTS", "POSS", "OPP_PTS", "OPP_POSS"}.issubset(out.columns):
        # Offensive Rating = (PTS / POSS) * 100
        out["OFF_RATING"] = (out["PTS"] / out["POSS"].replace(0, pd.NA)) * 100
        # Defensive Rating = (OPP_PTS / OPP_POSS) * 100
        out["DEF_RATING"] = (out["OPP_PTS"] / out["OPP_POSS"].replace(0, pd.NA)) * 100
        # Net Rating = OFF_RATING - DEF_RATING
        out["NET_RATING"] = out["OFF_RATING"] - out["DEF_RATING"]

        # Note: OPP_NET_RATING would just be -NET_RATING (mirror image), so
        # NET_RATING_DIFF = NET_RATING - OPP_NET_RATING = 2 * NET_RATING.
        # We use NET_RATING directly as the target to avoid double-counting.

    return out


def _train_linear_model(df: pd.DataFrame, feature_cols: list[str], target_col: str = "PLUS_MINUS") -> dict:
    """Train a simple linear regression model and return serializable artifacts."""
    work = df.dropna(subset=feature_cols + [target_col]).copy()

    if work.empty:
        raise ValueError("No training rows after dropping NaNs. Check your feature/target columns.")

    X = work[feature_cols].values
    y = work[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    coef_map = {col: float(coef) for col, coef in zip(feature_cols, model.coef_)}

    return {
        "model_family": "linear_regression",
        "target": target_col,
        "features": feature_cols,
        "intercept": float(model.intercept_),
        "coefficients": coef_map,
        "r_squared": float(r2),
        "training_games": int(len(work)),
    }


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


def train_models(seasons_csv: str, output_name: str, repo_dir: Path) -> int:
    start = time.time()
    try:
        repo_dir = ensure_data_repo(repo_dir)
        models_dir = repo_dir / "models"

        seasons = [s.strip() for s in seasons_csv.split(",") if s.strip()]
        if not seasons:
            raise ValueError("No seasons provided. Example: --seasons 2024-25,2025-26")

        frames: list[pd.DataFrame] = []
        adv_frames: list[pd.DataFrame] = []
        missing: list[str] = []
        missing_adv: list[str] = []

        for season in seasons:
            fp = repo_dir / _season_to_filename(season)
            if not fp.exists():
                missing.append(season)
                continue
            df = _load_existing_season_csv(fp)
            if df is not None:
                frames.append(_normalize_game_level_df(df))

            # Load advanced stats for possessions
            adv_fp = repo_dir / _advanced_filename(season)
            if adv_fp.exists():
                adv_df = pd.read_csv(adv_fp)
                adv_frames.append(adv_df)
            else:
                missing_adv.append(season)

        if missing:
            raise FileNotFoundError(
                f"Missing season CSVs for: {', '.join(missing)}. Run update-data or download-data first."
            )

        if missing_adv:
            print(f"[info] Missing advanced stats for: {', '.join(missing_adv)}. Possessions will be estimated from box score data.")

        game_df = pd.concat(frames, ignore_index=True)
        adv_df = pd.concat(adv_frames, ignore_index=True) if adv_frames else None

        print("[train] Expanding game logs into team-level rows (with possession data)")
        team_df = _gamelogs_to_teamrows(game_df, adv_df)

        print("[train] Computing four factors")
        team_df = _compute_four_factors(team_df)
        team_df = _attach_opponent_rows(team_df)
        team_df = _estimate_possessions(team_df)  # Fill missing possession data with estimates
        team_df = _compute_factor_differentials(team_df)

        # Train four factors model on NET_RATING_DIFF (not PLUS_MINUS)
        # The Four Factors are rate statistics, so they should predict a rate outcome
        feature_cols = ["EFG_DIFF", "TOV_DIFF", "OREB_DIFF", "FT_DIFF"]
        feature_cols = [c for c in feature_cols if c in team_df.columns]

        # Check if NET_RATING is available
        if "NET_RATING" not in team_df.columns:
            raise ValueError("NET_RATING not computed. Ensure possession data is available.")

        # Drop rows without NET_RATING
        before_count = len(team_df)
        team_df = team_df.dropna(subset=["NET_RATING"])
        after_count = len(team_df)
        if before_count > after_count:
            print(f"[train] Dropped {before_count - after_count} rows without rating data")

        print(f"[train] Training Four Factors model on: {', '.join(feature_cols)}")
        print(f"[train] Target: NET_RATING (per-100-possession rating)")

        # Save training data CSV for manual verification
        # Create game-level export with HOME_ and ROAD_ columns
        target_col = "NET_RATING"

        # Compute Ball Handling = 1 - TOV_PCT (higher is better, like other offensive stats)
        team_df["BH"] = 1.0 - team_df["TOV_PCT"]

        # Separate home and away rows based on MATCHUP
        team_df["IS_HOME"] = team_df["MATCHUP"].str.contains("vs.", na=False)
        home_rows = team_df[team_df["IS_HOME"]].copy()
        road_rows = team_df[~team_df["IS_HOME"]].copy()

        # Build home export with clean column names
        home_export = pd.DataFrame({
            "Game_ID": home_rows["GAME_ID"].astype(str).str.zfill(10),
            "Game_Date": home_rows["GAME_DATE"],
            "Home_Team": home_rows["TEAM_ABBREVIATION"],
            "Home_OFF_Rating": home_rows["OFF_RATING"].round(6),
            "Home_EFG": home_rows["EFG_PCT"].round(6),
            "Home_BH": home_rows["BH"].round(6),
            "Home_OREB": home_rows["OREB_PCT"].round(6),
            "Home_FT_Rate": home_rows["FT_RATE"].round(6),
        })

        # Build road export
        road_export = pd.DataFrame({
            "Game_ID": road_rows["GAME_ID"].astype(str).str.zfill(10),
            "Road_Team": road_rows["TEAM_ABBREVIATION"],
            "Road_OFF_Rating": road_rows["OFF_RATING"].round(6),
            "Road_EFG": road_rows["EFG_PCT"].round(6),
            "Road_BH": road_rows["BH"].round(6),
            "Road_OREB": road_rows["OREB_PCT"].round(6),
            "Road_FT_Rate": road_rows["FT_RATE"].round(6),
        })

        # Merge home and road on Game_ID
        export_df = home_export.merge(road_export, on="Game_ID", how="inner")

        # Drop rows with missing OFF_Rating
        export_df = export_df.dropna(subset=["Home_OFF_Rating", "Road_OFF_Rating"])

        # Reorder columns: context first, then home stats, then road stats
        export_df = export_df[[
            "Game_ID", "Game_Date", "Home_Team", "Road_Team",
            "Home_OFF_Rating", "Home_EFG", "Home_BH", "Home_OREB", "Home_FT_Rate",
            "Road_OFF_Rating", "Road_EFG", "Road_BH", "Road_OREB", "Road_FT_Rate",
        ]]

        # Derive CSV filename from output name (e.g., "2021-2025.json" -> "2021-2025_modeldata.csv")
        csv_name = output_name.replace(".json", "_modeldata.csv")
        csv_path = models_dir / csv_name
        export_df.to_csv(csv_path, index=False)
        print(f"[train] Saved training data: {csv_path.name} ({len(export_df)} games)")

        raw_model = _train_linear_model(team_df, feature_cols, target_col=target_col)

        # Map coefficient names to what the backend expects
        coef_name_map = {
            "EFG_DIFF": "shooting",
            "TOV_DIFF": "ball_handling",
            "OREB_DIFF": "orebounding",
            "FT_DIFF": "free_throws",
        }
        mapped_coefficients = {
            coef_name_map.get(k, k): v
            for k, v in raw_model["coefficients"].items()
        }

        # Compute league averages for eight-factor decomposition
        # These are the mean values across all games in the training data
        league_averages = {}
        if "EFG_PCT" in team_df.columns:
            league_averages["efg"] = float(team_df["EFG_PCT"].mean())
        if "TOV_PCT" in team_df.columns:
            # ball_handling = 1 - TOV_PCT, so we store ball_handling average
            league_averages["ball_handling"] = float(1.0 - team_df["TOV_PCT"].mean())
        if "OREB_PCT" in team_df.columns:
            league_averages["oreb_pct"] = float(team_df["OREB_PCT"].mean())
        if "FT_RATE" in team_df.columns:
            league_averages["ft_rate"] = float(team_df["FT_RATE"].mean())

        four_factors_output = {
            "coefficients": mapped_coefficients,
            "intercept": raw_model["intercept"],
            "league_averages": league_averages,
            "r_squared": raw_model["r_squared"],
            "training_games": raw_model["training_games"],
        }

        models_output: Dict[str, dict] = {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "training_seasons": seasons,
            "four_factors": four_factors_output,
        }

        output_path = models_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(models_output, f, indent=2)

        elapsed = time.time() - start
        print("\n[train] Saved model")
        print(f"  file: {output_path}")
        print(f"  r_squared: {raw_model['r_squared']:.4f}")
        print(f"  training_games: {raw_model['training_games']}")
        print(f"  time: {elapsed:.1f}s")
        return 0

    except Exception as e:
        print(f"[error] train-models failed: {e}")
        return 1


def train_season_models(seasons_csv: str, output_name: str, repo_dir: Path) -> int:
    """Train a season-level Eight Factors model.

    Unlike train_models which uses individual games as rows, this function:
    1. Aggregates game-level data to team-season level (one row per team per season)
    2. Uses 8 factors as features (team's own 4 factors + opponent's 4 factors)
    3. Predicts the team's season net rating from those 8 factors

    This model is used for the Contribution Analysis page to decompose
    a team's net rating into contributions from each factor.
    """
    start = time.time()
    try:
        repo_dir = ensure_data_repo(repo_dir)
        models_dir = repo_dir / "models"

        seasons = [s.strip() for s in seasons_csv.split(",") if s.strip()]
        if not seasons:
            raise ValueError("No seasons provided. Example: --seasons 2000-01,2001-02,...,2024-25")

        frames: list[pd.DataFrame] = []
        adv_frames: list[pd.DataFrame] = []
        missing: list[str] = []
        missing_adv: list[str] = []

        for season in seasons:
            fp = repo_dir / _season_to_filename(season)
            if not fp.exists():
                missing.append(season)
                continue
            df = _load_existing_season_csv(fp)
            if df is not None:
                norm_df = _normalize_game_level_df(df)
                norm_df["season"] = season  # Ensure season is set
                frames.append(norm_df)

            # Load advanced stats for possessions
            adv_fp = repo_dir / _advanced_filename(season)
            if adv_fp.exists():
                adv_df = pd.read_csv(adv_fp)
                adv_frames.append(adv_df)
            else:
                missing_adv.append(season)

        if missing:
            raise FileNotFoundError(
                f"Missing season CSVs for: {', '.join(missing)}. Run update-data or download-data first."
            )

        if missing_adv:
            print(f"[info] Missing advanced stats for: {', '.join(missing_adv)}. Possessions will be estimated from box score data.")

        game_df = pd.concat(frames, ignore_index=True)
        adv_df = pd.concat(adv_frames, ignore_index=True) if adv_frames else None

        print("[train-season] Expanding game logs into team-level rows (with possession data)")
        team_df = _gamelogs_to_teamrows(game_df, adv_df)

        # Add season column to team_df by parsing from GAME_ID or GAME_DATE
        # The season info was in the original game_df, so we need to merge it back
        # Actually, let's add season to the team rows based on the game dates
        team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"], errors="coerce")

        # Derive season from game date: Oct-Dec = year-year+1, Jan-Sep = year-1-year
        def get_season_from_date(dt):
            if pd.isna(dt):
                return None
            year = dt.year
            month = dt.month
            if month >= 10:  # Oct-Dec
                return f"{year}-{str(year + 1)[-2:]}"
            else:  # Jan-Sep
                return f"{year - 1}-{str(year)[-2:]}"

        team_df["SEASON"] = team_df["GAME_DATE"].apply(get_season_from_date)

        print("[train-season] Computing four factors")
        team_df = _compute_four_factors(team_df)
        team_df = _attach_opponent_rows(team_df)
        team_df = _estimate_possessions(team_df)  # Fill missing possession data with estimates
        team_df = _compute_factor_differentials(team_df)

        # Compute Ball Handling = 1 - TOV_PCT (higher is better)
        team_df["BH"] = 1.0 - team_df["TOV_PCT"]
        team_df["OPP_BH"] = 1.0 - team_df["OPP_TOV_PCT"]

        # Drop rows without NET_RATING
        before_count = len(team_df)
        team_df = team_df.dropna(subset=["NET_RATING", "SEASON"])
        after_count = len(team_df)
        if before_count > after_count:
            print(f"[train-season] Dropped {before_count - after_count} rows without rating/season data")

        # Aggregate to team-season level
        print("[train-season] Aggregating to team-season level")
        agg_cols = {
            "NET_RATING": "mean",
            "OFF_RATING": "mean",
            "DEF_RATING": "mean",
            "EFG_PCT": "mean",
            "BH": "mean",
            "OREB_PCT": "mean",
            "FT_RATE": "mean",
            "OPP_EFG_PCT": "mean",
            "OPP_BH": "mean",
            "GAME_ID": "count",  # Count games for reference
        }
        # Add OPP_OREB_PCT if available
        if "OPP_OREB" in team_df.columns and "DREB" in team_df.columns:
            denom = team_df["OPP_OREB"] + team_df["DREB"]
            team_df["OPP_OREB_PCT"] = team_df["OPP_OREB"] / denom.replace(0, pd.NA)
            agg_cols["OPP_OREB_PCT"] = "mean"

        # Add OPP_FT_RATE if available
        if "OPP_FT_RATE" in team_df.columns:
            agg_cols["OPP_FT_RATE"] = "mean"

        season_df = team_df.groupby(["SEASON", "TEAM_ABBREVIATION"]).agg(agg_cols).reset_index()
        season_df = season_df.rename(columns={"GAME_ID": "GAMES_PLAYED"})

        print(f"[train-season] Created {len(season_df)} team-season rows from {len(seasons)} seasons")

        # Define feature columns for the eight-factor model
        feature_cols = [
            "EFG_PCT", "BH", "OREB_PCT", "FT_RATE",
            "OPP_EFG_PCT", "OPP_BH", "OPP_OREB_PCT", "OPP_FT_RATE"
        ]
        # Keep only columns that exist
        feature_cols = [c for c in feature_cols if c in season_df.columns]

        print(f"[train-season] Training Eight Factors model on: {', '.join(feature_cols)}")
        print(f"[train-season] Target: NET_RATING (per-100-possession rating)")

        # Save training data CSV
        export_df = season_df[["SEASON", "TEAM_ABBREVIATION", "GAMES_PLAYED", "NET_RATING"] + feature_cols].copy()
        export_df = export_df.rename(columns={
            "SEASON": "Season",
            "TEAM_ABBREVIATION": "Team",
            "GAMES_PLAYED": "Games",
            "NET_RATING": "Net_Rating",
            "EFG_PCT": "EFG",
            "BH": "BH",
            "OREB_PCT": "OREB",
            "FT_RATE": "FT_Rate",
            "OPP_EFG_PCT": "Opp_EFG",
            "OPP_BH": "Opp_BH",
            "OPP_OREB_PCT": "Opp_OREB",
            "OPP_FT_RATE": "Opp_FT_Rate",
        })
        export_df = export_df.round(6)

        # Derive CSV filename from output name
        csv_name = output_name.replace(".json", "_modeldata.csv")
        csv_path = models_dir / csv_name
        export_df.to_csv(csv_path, index=False)
        print(f"[train-season] Saved training data: {csv_path.name} ({len(export_df)} team-seasons)")

        # Train the model
        work = season_df.dropna(subset=feature_cols + ["NET_RATING"]).copy()

        if work.empty:
            raise ValueError("No training rows after dropping NaNs. Check your feature/target columns.")

        X = work[feature_cols].values
        y = work["NET_RATING"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # Map coefficient names
        coef_name_map = {
            "EFG_PCT": "shooting",
            "BH": "ball_handling",
            "OREB_PCT": "orebounding",
            "FT_RATE": "free_throws",
            "OPP_EFG_PCT": "opp_shooting",
            "OPP_BH": "opp_ball_handling",
            "OPP_OREB_PCT": "opp_orebounding",
            "OPP_FT_RATE": "opp_free_throws",
        }
        mapped_coefficients = {
            coef_name_map.get(k, k): float(coef)
            for k, coef in zip(feature_cols, model.coef_)
        }

        # Compute league averages
        league_averages = {}
        for col in feature_cols:
            if col in season_df.columns:
                avg_key = coef_name_map.get(col, col)
                league_averages[avg_key] = float(season_df[col].mean())

        # Build the output
        eight_factors_output = {
            "coefficients": mapped_coefficients,
            "intercept": float(model.intercept_),
            "league_averages": league_averages,
            "r_squared": float(r2),
            "training_team_seasons": int(len(work)),
        }

        models_output: Dict[str, dict] = {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "training_seasons": seasons,
            "model_type": "season_level",
            "eight_factors": eight_factors_output,
        }

        output_path = models_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(models_output, f, indent=2)

        elapsed = time.time() - start
        print("\n[train-season] Saved model")
        print(f"  file: {output_path}")
        print(f"  r_squared: {r2:.4f}")
        print(f"  training_team_seasons: {len(work)}")
        print(f"  time: {elapsed:.1f}s")
        return 0

    except Exception as e:
        print(f"[error] train-season-models failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def list_models(repo_dir: Path) -> int:
    try:
        repo_dir = ensure_data_repo(repo_dir)
        models_dir = repo_dir / "models"

        json_files = sorted(models_dir.glob("*.json"))
        if not json_files:
            print(f"[models] No model files found in {models_dir}")
            return 0

        print(f"[models] Found {len(json_files)} file(s) in {models_dir}\n")

        for fp in json_files:
            print(f"- {fp.name}")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as e:
                print(f"  (failed to read JSON: {e})")
                continue

            trained_at = payload.get("trained_at")
            seasons = payload.get("training_seasons")
            if trained_at:
                print(f"  trained_at: {trained_at}")
            if seasons:
                print(f"  seasons: {', '.join(seasons)}")

            # Show four_factors model info
            if "four_factors" in payload:
                model = payload["four_factors"]
                r2 = model.get("r_squared") if isinstance(model, dict) else None
                games = model.get("training_games") if isinstance(model, dict) else None
                print(f"  four_factors:")
                if r2 is not None:
                    print(f"    r_squared: {r2:.4f}")
                if games is not None:
                    print(f"    training_games: {games}")

            print()

        return 0

    except Exception as e:
        print(f"[error] list-models failed: {e}")
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

        # Limit status to entire repo (season CSVs + models)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Admin CLI for NBA_Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from backend directory):
  python admin/cli.py update-data --season 2025-26
  python admin/cli.py download-data --start 2020-21 --end 2024-25
  python admin/cli.py train-models --seasons 2024-25,2025-26 --output latest.json
  python admin/cli.py commit-and-push --message "Update data"
  python admin/cli.py git-status
  python admin/cli.py list-models
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

    p_train = sub.add_parser(
        "train-models",
        help="Train Four Factors model and save JSON into NBA_Data/models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py train-models --seasons 2024-25,2025-26 --output latest.json
  python admin/cli.py train-models --seasons 2023-24 --output 2023-24.json

Trains a Four Factors linear regression model using game log data from the
specified seasons. The model JSON is saved to NBA_Data/models/, and a
companion CSV with training data (features + target) is saved alongside
for manual verification.
""",
    )
    p_train.add_argument(
        "--seasons",
        required=True,
        help="Comma-separated seasons, e.g. 2024-25,2025-26",
    )
    p_train.add_argument(
        "--output",
        required=True,
        help="Output JSON filename (saved under NBA_Data/models/), e.g. latest.json",
    )

    p_train_season = sub.add_parser(
        "train-season-models",
        help="Train season-level Eight Factors model for Contribution Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin/cli.py train-season-models --seasons 2000-01,2001-02,...,2024-25 --output season_2000-2025.json

Trains a season-level Eight Factors linear regression model. Unlike train-models
which uses individual games as rows, this aggregates to team-season level and
uses 8 factors (team's own 4 + opponent's 4) to predict season net rating.

This model is used for the Contribution Analysis page to decompose a team's
net rating into contributions from each of the eight factors.
""",
    )
    p_train_season.add_argument(
        "--seasons",
        required=True,
        help="Comma-separated seasons, e.g. 2000-01,2001-02,...,2024-25",
    )
    p_train_season.add_argument(
        "--output",
        required=True,
        help="Output JSON filename (saved under NBA_Data/models/), e.g. season_2000-2025.json",
    )

    sub.add_parser(
        "list-models",
        help="List saved model JSON files in NBA_Data/models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python admin/cli.py list-models

Lists all model JSON files in NBA_Data/models/ with their training
metadata (date, seasons used, R-squared scores).
""",
    )

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
  python admin/cli.py commit-and-push --message "Add new models" --dry-run

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

    if args.command == "train-models":
        return train_models(args.seasons, args.output, repo_dir)

    if args.command == "train-season-models":
        return train_season_models(args.seasons, args.output, repo_dir)

    if args.command == "list-models":
        return list_models(repo_dir)

    if args.command == "git-status":
        return git_status(repo_dir)

    if args.command == "commit-and-push":
        return commit_and_push(args.message, repo_dir, dry_run=args.dry_run)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())