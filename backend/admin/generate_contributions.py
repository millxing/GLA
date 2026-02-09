#!/usr/bin/env python3
"""
Generate per-season contribution JSON files for LLM game analysis.

For each season, produces an out-of-sample eight-factor decomposition of every game.
The model is trained on up to 7 prior seasons (as many as available) + current-season
games before each game date, ensuring contributions are out-of-sample.

Output: NBA_Data/contributions/contributions_{season}.json

Usage:
    cd backend
    python admin/generate_contributions.py                         # all seasons
    python admin/generate_contributions.py --season 2024-25        # single season
    python admin/generate_contributions.py --repo-dir /path/to/NBA_Data
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Import helpers from cli.py (same directory)
from cli import (
    DEFAULT_REPO_DIR,
    ensure_data_repo,
    _season_to_filename,
    _linescore_filename,
    _advanced_filename,
    _load_existing_season_csv,
    _normalize_game_level_df,
)

# Import season helpers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_available_seasons, SEASON_START_YEAR


# ---- Game type display names ----
GAME_TYPE_DISPLAY = {
    "regular_season": "Regular Season",
    "playoffs": "Playoffs",
    "play_in": "Play-In",
    "nba_cup_semi": "NBA Cup Semifinal",
    "nba_cup_final": "NBA Cup Final",
    "nba_cup_group": "Regular Season",  # Group stage counted as regular season
}

# ---- Factor metadata ----
# Maps model coefficient keys to DataFrame column names and display names
FACTOR_INFO = [
    ("shooting", "EFG_PCT", "Shooting (eFG%)"),
    ("ball_handling", "BH", "Ball Handling"),        # BH = 1 - TOV_PCT, computed below
    ("orebounding", "OREB_PCT", "Off Rebounding"),
    ("free_throws", "FT_RATE", "Free Throw Rate"),
]

# Model coefficient keys → differential column names (for training)
FEATURE_COLS = ["EFG_DIFF", "TOV_DIFF", "OREB_DIFF", "FT_DIFF"]
COEF_NAME_MAP = {
    "EFG_DIFF": "shooting",
    "TOV_DIFF": "ball_handling",
    "OREB_DIFF": "orebounding",
    "FT_DIFF": "free_throws",
}

PERCENTILE_LEVELS = [
    ("p10", 0.10),
    ("p25", 0.25),
    ("p40", 0.40),
    ("p60", 0.60),
    ("p75", 0.75),
    ("p90", 0.90),
]

FACTOR_PERCENTILE_COLS = [
    ("efg", "EFG_PCT"),
    ("ball_handling", "BH"),
    ("oreb", "OREB_PCT"),
    ("ft_rate", "FT_RATE"),
]

PRIOR_SEASONS_FOR_TRAINING = 7


def _gamelogs_to_teamrows(game_df: pd.DataFrame, adv_df: pd.DataFrame = None) -> pd.DataFrame:
    """Expand game-level rows into team-level rows for modeling.

    If adv_df is provided, merge possession data from box_score_advanced.
    """
    df = _normalize_game_level_df(game_df)

    # Merge advanced stats (possessions) if available.
    if adv_df is not None and not adv_df.empty:
        adv_df = adv_df.copy()
        adv_df["game_id"] = adv_df["game_id"].astype(str).map(
            lambda v: v.zfill(10) if v.isdigit() else v
        )
        df = df.merge(
            adv_df[["game_id", "possessions_home", "possessions_road"]],
            on="game_id",
            how="left",
        )

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
    if "possessions_home" in df.columns:
        home_data["POSS"] = df["possessions_home"]
        home_data["OPP_POSS"] = df["possessions_road"]
    home = pd.DataFrame(home_data)

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
    if "possessions_road" in df.columns:
        road_data["POSS"] = df["possessions_road"]
        road_data["OPP_POSS"] = df["possessions_home"]
    road = pd.DataFrame(road_data)

    out = pd.concat([home, road], ignore_index=True)

    for c in [
        "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PLUS_MINUS",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _estimate_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate possession columns loaded from box score advanced data."""
    out = df.copy()

    if "POSS" not in out.columns:
        print("[possessions] WARNING: No POSS column found - actual possessions not merged")
    else:
        missing = out["POSS"].isna().sum()
        if missing > 0:
            print(f"[possessions] WARNING: {missing} rows missing actual POSS data")

    if "OPP_POSS" not in out.columns:
        print("[possessions] WARNING: No OPP_POSS column found - actual possessions not merged")
    else:
        missing_opp = out["OPP_POSS"].isna().sum()
        if missing_opp > 0:
            print(f"[possessions] WARNING: {missing_opp} rows missing actual OPP_POSS data")

    return out


def _compute_four_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Four Factors using team box score stats."""
    out = df.copy()

    for col in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "TOV"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"FGM", "FG3M", "FGA"}.issubset(out.columns):
        out["EFG_PCT"] = (out["FGM"] + 0.5 * out["FG3M"]) / out["FGA"].replace(0, pd.NA)

    if "POSS" in out.columns and "TOV" in out.columns:
        out["TOV_PCT"] = out["TOV"] / out["POSS"].replace(0, pd.NA)

    if {"FTM", "FGA"}.issubset(out.columns):
        out["FT_RATE"] = out["FTM"] / out["FGA"].replace(0, pd.NA)

    return out


def _attach_opponent_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent rows by GAME_ID, retaining one opponent row per team row."""
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

    merged = df.merge(opp, on="GAME_ID", how="left")

    if "OPP_TEAM_ID" in merged.columns:
        merged = merged[merged["OPP_TEAM_ID"] != merged["TEAM_ID"]]

    merged = merged.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    return merged


def _compute_factor_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-vs-opponent differential columns for model training."""
    out = df.copy()

    if "OPP_DREB" in out.columns and "OREB" in out.columns:
        denom = out["OREB"] + out["OPP_DREB"]
        out["OREB_PCT"] = out["OREB"] / denom.replace(0, pd.NA)

    if {"OPP_FGM", "OPP_FG3M", "OPP_FGA"}.issubset(out.columns):
        out["OPP_EFG_PCT"] = (out["OPP_FGM"] + 0.5 * out["OPP_FG3M"]) / out["OPP_FGA"].replace(0, pd.NA)

    if {"OPP_TOV", "OPP_POSS"}.issubset(out.columns):
        out["OPP_TOV_PCT"] = out["OPP_TOV"] / out["OPP_POSS"].replace(0, pd.NA)

    if {"OPP_FTM", "OPP_FGA"}.issubset(out.columns):
        out["OPP_FT_RATE"] = out["OPP_FTM"] / out["OPP_FGA"].replace(0, pd.NA)

    if {"EFG_PCT", "OPP_EFG_PCT"}.issubset(out.columns):
        out["EFG_DIFF"] = out["EFG_PCT"] - out["OPP_EFG_PCT"]

    if {"TOV_PCT", "OPP_TOV_PCT"}.issubset(out.columns):
        out["TOV_DIFF"] = out["OPP_TOV_PCT"] - out["TOV_PCT"]

    if {"FT_RATE", "OPP_FT_RATE"}.issubset(out.columns):
        out["FT_DIFF"] = out["FT_RATE"] - out["OPP_FT_RATE"]

    if {"OREB_PCT", "OPP_DREB"}.issubset(out.columns) and "OPP_OREB" in out.columns:
        denom = out["OPP_OREB"] + out.get("DREB", pd.Series(pd.NA, index=out.index))
        opp_oreb_pct = out["OPP_OREB"] / denom.replace(0, pd.NA)
        out["OREB_DIFF"] = out["OREB_PCT"] - opp_oreb_pct

    if {"PTS", "POSS", "OPP_PTS", "OPP_POSS"}.issubset(out.columns):
        out["OFF_RATING"] = (out["PTS"] / out["POSS"].replace(0, pd.NA)) * 100
        out["DEF_RATING"] = (out["OPP_PTS"] / out["OPP_POSS"].replace(0, pd.NA)) * 100
        out["NET_RATING"] = out["OFF_RATING"] - out["DEF_RATING"]

    return out


def _train_linear_model(df: pd.DataFrame, feature_cols: list[str], target_col: str = "PLUS_MINUS") -> dict:
    """Train a linear regression model and return serializable metrics/artifacts."""
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


def get_prior_seasons(season: str, max_prior: int = PRIOR_SEASONS_FOR_TRAINING) -> List[str]:
    """Return up to `max_prior` seasons before `season`, oldest -> newest."""
    start_year = int(season.split("-")[0])
    prior: List[str] = []
    for offset in range(max_prior, 0, -1):
        prior_start = start_year - offset
        if prior_start < SEASON_START_YEAR:
            continue
        prior_end = prior_start + 1
        prior.append(f"{prior_start}-{str(prior_end)[-2:]}")
    return prior


def load_season_data(season: str, repo_dir: Path) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load game logs, advanced stats, and linescores for a season.

    Returns (game_df, adv_df, linescore_df) — any may be None if file missing.
    """
    game_path = repo_dir / _season_to_filename(season)
    adv_path = repo_dir / _advanced_filename(season)
    ls_path = repo_dir / _linescore_filename(season)

    game_df = None
    if game_path.exists():
        raw = _load_existing_season_csv(game_path)
        if raw is not None:
            game_df = _normalize_game_level_df(raw)

    adv_df = None
    if adv_path.exists():
        adv_df = pd.read_csv(adv_path)

    ls_df = None
    if ls_path.exists():
        ls_df = pd.read_csv(ls_path, dtype={"game_id": "string"})
        # Normalize game_id
        ls_df["game_id"] = ls_df["game_id"].astype(str).str.strip()
        ls_df["game_id"] = ls_df["game_id"].str.replace(r"\.0$", "", regex=True)
        ls_df["game_id"] = ls_df["game_id"].map(
            lambda v: v.zfill(10) if isinstance(v, str) and v.isdigit() else v
        )

    return game_df, adv_df, ls_df


def precompute_team_rows(game_dfs: list[pd.DataFrame], adv_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Run the full cli.py pipeline on combined game data.

    Returns a DataFrame with 2 rows per game (home + road), containing:
    EFG_PCT, TOV_PCT, OREB_PCT, FT_RATE, OFF_RATING, DEF_RATING, NET_RATING,
    POSS, OPP_POSS, EFG_DIFF, TOV_DIFF, OREB_DIFF, FT_DIFF, etc.
    All factor values are on 0-1 proportion scale.
    """
    game_df = pd.concat(game_dfs, ignore_index=True)
    adv_df = pd.concat(adv_dfs, ignore_index=True) if adv_dfs else None

    team_df = _gamelogs_to_teamrows(game_df, adv_df)
    team_df = _compute_four_factors(team_df)
    team_df = _attach_opponent_rows(team_df)
    team_df = _estimate_possessions(team_df)
    team_df = _compute_factor_differentials(team_df)

    # Add Ball Handling = 1 - TOV_PCT (higher is better, 0-1 scale)
    # TOV_PCT is now computed as TOV/POSS in _compute_four_factors (using actual possessions)
    if "TOV_PCT" in team_df.columns:
        team_df["BH"] = 1.0 - team_df["TOV_PCT"]

    # ---- Merge actual minutes for pace computation ----
    # _gamelogs_to_teamrows only merges possessions, not minutes
    if adv_df is not None and not adv_df.empty:
        adv_mins = adv_df[["game_id", "minutes_home", "minutes_road"]].copy()
        adv_mins["game_id"] = adv_mins["game_id"].astype(str).map(
            lambda v: v.zfill(10) if v.isdigit() else v
        )
        # For home rows: ACTUAL_MINUTES = minutes_home
        # For road rows: ACTUAL_MINUTES = minutes_road
        # We'll merge both and pick the right one based on IS_HOME
        team_df = team_df.merge(adv_mins, left_on="GAME_ID", right_on="game_id", how="left")
        if "game_id" in team_df.columns and "GAME_ID" in team_df.columns:
            team_df = team_df.drop(columns=["game_id"])

    # Add IS_HOME flag based on MATCHUP
    team_df["IS_HOME"] = team_df["MATCHUP"].str.contains("vs.", na=False)

    # Compute actual pace where minutes are available
    # Pace = avg_possessions * (48 / actual_game_minutes)
    if {"POSS", "OPP_POSS"}.issubset(team_df.columns):
        avg_poss = (team_df["POSS"] + team_df["OPP_POSS"]) / 2

        # Get actual minutes for this team's side
        if "minutes_home" in team_df.columns:
            actual_mins = team_df["minutes_home"].where(team_df["IS_HOME"], team_df.get("minutes_road"))
            actual_mins = pd.to_numeric(actual_mins, errors="coerce")
            game_mins = actual_mins / 5  # Convert per-player minutes to game minutes
            team_df["PACE"] = avg_poss * (48 / game_mins)
        else:
            team_df["PACE"] = pd.NA

    return team_df


def compute_league_averages(training_df: pd.DataFrame) -> Dict[str, float]:
    """Compute league averages from training data.

    Factor averages are on 0-1 scale. Rating averages are per-100-poss.
    """
    avgs = {}

    if "EFG_PCT" in training_df.columns:
        avgs["EFG_PCT"] = float(training_df["EFG_PCT"].mean())
    if "BH" in training_df.columns:
        avgs["BH"] = float(training_df["BH"].mean())
    elif "TOV_PCT" in training_df.columns:
        avgs["BH"] = float(1.0 - training_df["TOV_PCT"].mean())
    if "OREB_PCT" in training_df.columns:
        avgs["OREB_PCT"] = float(training_df["OREB_PCT"].mean())
    if "FT_RATE" in training_df.columns:
        avgs["FT_RATE"] = float(training_df["FT_RATE"].mean())
    if "OFF_RATING" in training_df.columns:
        avgs["OFF_RATING"] = float(training_df["OFF_RATING"].mean())
    if "PACE" in training_df.columns:
        pace_vals = training_df["PACE"].dropna()
        if len(pace_vals) > 0:
            avgs["PACE"] = float(pace_vals.mean())

    return avgs


def compute_factor_percentiles(training_df: pd.DataFrame) -> Dict[str, float]:
    """Compute requested factor percentiles from training data.

    Returned values are on 0-1 scale and converted for output at write time.
    """
    percentiles: Dict[str, float] = {}

    for out_key, col in FACTOR_PERCENTILE_COLS:
        if col not in training_df.columns:
            continue
        vals = pd.to_numeric(training_df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        for pct_label, pct_val in PERCENTILE_LEVELS:
            percentiles[f"{out_key}_{pct_label}"] = float(vals.quantile(pct_val))

    return percentiles


def decompose_game(
    game_id: str,
    home_row: pd.Series,
    road_row: pd.Series,
    model_coefficients: Dict[str, float],
    league_avgs: Dict[str, float],
    game_info: Dict[str, Any],
    linescore_row: Optional[pd.Series],
    advanced_row: Optional[pd.Series],
) -> Dict[str, Any]:
    """Build a single game JSON record from precomputed values.

    home_row and road_row come from the precomputed team-rows DataFrame
    (0-1 scale for factors, per-100-poss for ratings).
    """
    # ---- Factors ----
    factors = {"road": [], "home": []}

    for coef_key, col_name, display_name in FACTOR_INFO:
        coef = model_coefficients.get(coef_key, 0.0)
        home_val = float(home_row.get(col_name, 0) or 0)
        road_val = float(road_row.get(col_name, 0) or 0)
        avg_val = league_avgs.get(col_name, 0.0)

        # Contributions (0-1 scale, no division by 100 needed)
        home_contrib = coef * (home_val - avg_val)
        road_contrib = -(coef * (road_val - avg_val))  # negated for display

        # Display values (convert 0-1 → 0-100 percentage)
        home_display = round(home_val * 100, 1)
        road_display = round(road_val * 100, 1)
        avg_display = avg_val * 100
        home_vs_avg = round(home_display - avg_display, 1)
        road_vs_avg = round(road_display - avg_display, 1)

        factors["road"].append({
            "factor": display_name,
            "contribution": round(road_contrib, 1),
            "value": road_display,
            "vs_avg": road_vs_avg,
        })
        factors["home"].append({
            "factor": display_name,
            "contribution": round(home_contrib, 1),
            "value": home_display,
            "vs_avg": home_vs_avg,
        })

    # ---- Ratings ----
    home_off = float(home_row.get("OFF_RATING", 0) or 0)
    home_def = float(home_row.get("DEF_RATING", 0) or 0)
    home_net = float(home_row.get("NET_RATING", 0) or 0)
    road_off = float(road_row.get("OFF_RATING", 0) or 0)
    road_def = float(road_row.get("DEF_RATING", 0) or 0)
    road_net = float(road_row.get("NET_RATING", 0) or 0)

    # Pace from precomputed column (actual_poss * 48/game_minutes, or NA if no minutes)
    home_pace = home_row.get("PACE")
    pace = round(float(home_pace), 1) if pd.notna(home_pace) else None

    avg_off = league_avgs.get("OFF_RATING", 0)
    avg_pace = league_avgs.get("PACE")

    pace_vs_avg = round(pace - avg_pace, 1) if pace is not None and avg_pace is not None else None

    ratings = {
        "road": [
            {"metric": "Offensive Rating", "value": round(road_off, 1), "vs_avg": round(road_off - avg_off, 1)},
            {"metric": "Defensive Rating", "value": round(road_def, 1), "vs_avg": round(road_def - avg_off, 1)},
            {"metric": "Net Rating", "value": round(road_net, 1), "vs_avg": round(road_net, 1)},  # league avg net = 0
            {"metric": "Pace", "value": pace, "vs_avg": pace_vs_avg},
        ],
        "home": [
            {"metric": "Offensive Rating", "value": round(home_off, 1), "vs_avg": round(home_off - avg_off, 1)},
            {"metric": "Defensive Rating", "value": round(home_def, 1), "vs_avg": round(home_def - avg_off, 1)},
            {"metric": "Net Rating", "value": round(home_net, 1), "vs_avg": round(home_net, 1)},
            {"metric": "Pace", "value": pace, "vs_avg": pace_vs_avg},
        ],
    }

    # ---- Linescore ----
    def _format_numeric(value: Any, decimals: int = 1) -> Optional[float | int]:
        num = pd.to_numeric(value, errors="coerce")
        if pd.isna(num):
            return None
        val = float(num)
        if abs(val - round(val)) < 1e-9:
            return int(round(val))
        return round(val, decimals)

    linescore = None
    if linescore_row is not None:
        home_minutes = None
        road_minutes = None
        home_possessions = None
        road_possessions = None

        if advanced_row is not None:
            home_minutes = _format_numeric(advanced_row.get("minutes_home"))
            road_minutes = _format_numeric(advanced_row.get("minutes_road"))
            home_possessions = _format_numeric(advanced_row.get("possessions_home"))
            road_possessions = _format_numeric(advanced_row.get("possessions_road"))

        # Fallbacks in case advanced rows are missing for a game.
        if home_minutes is None:
            home_minutes = _format_numeric(home_row.get("minutes_home"))
        if road_minutes is None:
            road_minutes = _format_numeric(road_row.get("minutes_road"))
        if home_possessions is None:
            home_possessions = _format_numeric(home_row.get("POSS"))
        if road_possessions is None:
            road_possessions = _format_numeric(road_row.get("POSS"))

        linescore = {
            "road": {
                "Q1": int(linescore_row.get("pts_qtr1_road", 0) or 0),
                "Q2": int(linescore_row.get("pts_qtr2_road", 0) or 0),
                "Q3": int(linescore_row.get("pts_qtr3_road", 0) or 0),
                "Q4": int(linescore_row.get("pts_qtr4_road", 0) or 0),
                "OT": int(linescore_row.get("pts_ot_total_road", 0) or 0),
                "Total": int(linescore_row.get("pts_road", 0) or 0),
                "Minutes": road_minutes,
                "Possessions": road_possessions,
            },
            "home": {
                "Q1": int(linescore_row.get("pts_qtr1_home", 0) or 0),
                "Q2": int(linescore_row.get("pts_qtr2_home", 0) or 0),
                "Q3": int(linescore_row.get("pts_qtr3_home", 0) or 0),
                "Q4": int(linescore_row.get("pts_qtr4_home", 0) or 0),
                "OT": int(linescore_row.get("pts_ot_total_home", 0) or 0),
                "Total": int(linescore_row.get("pts_home", 0) or 0),
                "Minutes": home_minutes,
                "Possessions": home_possessions,
            },
        }

    # ---- Assemble ----
    return {
        "game_id": game_id,
        "game_info": game_info,
        "linescore": linescore,
        "factors": factors,
        "ratings": ratings,
    }


def process_season(season: str, repo_dir: Path) -> List[Dict[str, Any]]:
    """Process a single season: rolling model training + game decomposition.

    Returns list of game records sorted by game date.
    """
    prior_seasons = get_prior_seasons(season)
    is_first_season = int(season.split("-")[0]) == SEASON_START_YEAR

    print(f"\n{'='*60}")
    if is_first_season:
        print(f"Processing {season} (in-sample exception: full season training)")
    elif prior_seasons:
        print(f"Processing {season} (prior seasons: {', '.join(prior_seasons)})")
    else:
        print(f"Processing {season} (no prior seasons available)")
    print(f"{'='*60}")

    # ---- Load data ----
    game_dfs = []
    adv_dfs = []

    for prior_season in prior_seasons:
        prior_game, prior_adv, _ = load_season_data(prior_season, repo_dir)
        if prior_game is not None:
            prior_game = prior_game.copy()
            prior_game["_season_tag"] = prior_season
            game_dfs.append(prior_game)
        if prior_adv is not None:
            adv_dfs.append(prior_adv)

    curr_game, curr_adv, curr_ls = load_season_data(season, repo_dir)
    if curr_game is None:
        print(f"  [skip] No game data for {season}")
        return []

    curr_game = curr_game.copy()
    curr_game["_season_tag"] = season
    game_dfs.append(curr_game)
    if curr_adv is not None:
        adv_dfs.append(curr_adv)

    if not game_dfs:
        print(f"  [skip] No data loaded for {season}")
        return []

    # ---- Build linescore lookup ----
    ls_lookup: Dict[str, pd.Series] = {}
    if curr_ls is not None:
        for _, row in curr_ls.iterrows():
            gid = str(row.get("game_id", "")).strip()
            if gid:
                ls_lookup[gid] = row

    adv_lookup: Dict[str, pd.Series] = {}
    if curr_adv is not None:
        adv_norm = curr_adv.copy()
        if "game_id" in adv_norm.columns:
            adv_norm["game_id"] = adv_norm["game_id"].astype(str).str.strip()
            adv_norm["game_id"] = adv_norm["game_id"].str.replace(r"\.0$", "", regex=True)
            adv_norm["game_id"] = adv_norm["game_id"].map(
                lambda v: v.zfill(10) if isinstance(v, str) and v.isdigit() else v
            )
            for _, row in adv_norm.iterrows():
                gid = str(row.get("game_id", "")).strip()
                if gid:
                    adv_lookup[gid] = row

    # ---- Build game info lookup from raw game-level CSV ----
    game_info_lookup: Dict[str, Dict[str, Any]] = {}
    for _, row in curr_game.iterrows():
        gid = str(row.get("game_id", "")).strip()
        if not gid:
            continue
        game_type_raw = str(row.get("game_type", "regular_season") or "regular_season")
        game_info_lookup[gid] = {
            "road": str(row.get("team_abbreviation_road", "") or ""),
            "home": str(row.get("team_abbreviation_home", "") or ""),
            "game_date": str(row.get("game_date", "") or "")[:10],  # YYYY-MM-DD
            "game_type": GAME_TYPE_DISPLAY.get(game_type_raw, game_type_raw),
        }

    # ---- Precompute team rows ----
    print(f"  Precomputing team rows...")
    combined_game = pd.concat(game_dfs, ignore_index=True)
    # Preserve _season_tag through the pipeline
    season_tags = combined_game[["game_id", "_season_tag"]].drop_duplicates()

    team_df = precompute_team_rows(game_dfs, adv_dfs)

    # Drop rows without NET_RATING (can't train or decompose without it)
    team_df = team_df.dropna(subset=["NET_RATING"])

    # Merge season tag back
    team_df["GAME_ID"] = team_df["GAME_ID"].astype(str)
    season_tags["game_id"] = season_tags["game_id"].astype(str)
    team_df = team_df.merge(
        season_tags.rename(columns={"game_id": "GAME_ID"}),
        on="GAME_ID",
        how="left",
    )

    # Separate prior and current season rows
    prior_rows = team_df[team_df["_season_tag"] != season] if prior_seasons else pd.DataFrame()
    current_rows = team_df[team_df["_season_tag"] == season]

    print(f"  Prior season rows: {len(prior_rows)}")
    print(f"  Current season rows: {len(current_rows)}")

    # ---- Get unique game dates in current season ----
    current_rows = current_rows.copy()
    current_rows["GAME_DATE"] = pd.to_datetime(current_rows["GAME_DATE"], errors="coerce")
    game_dates = sorted(current_rows["GAME_DATE"].dropna().unique())

    print(f"  Game dates to process: {len(game_dates)}")

    # ---- Process each game date ----
    all_records: List[Dict[str, Any]] = []
    for i, game_date in enumerate(game_dates):
        date_str = pd.Timestamp(game_date).strftime("%Y-%m-%d")

        # 2000-01 exception: use full current season in-sample.
        if is_first_season:
            training_df = current_rows.copy()
        else:
            # Out-of-sample: up to 7 prior seasons + current season before this game date.
            current_before = current_rows[current_rows["GAME_DATE"] < game_date]
            training_df = pd.concat([prior_rows, current_before], ignore_index=True)

        # Need enough data to train
        training_clean = training_df.dropna(subset=FEATURE_COLS + ["NET_RATING"])
        if len(training_clean) < 30:
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(game_dates)}] {date_str}: skipping (only {len(training_clean)} training rows)")
            continue

        try:
            raw_model = _train_linear_model(training_clean, FEATURE_COLS, target_col="NET_RATING")
        except Exception as e:
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(game_dates)}] {date_str}: model training failed: {e}")
            continue

        training_dates = pd.to_datetime(training_clean.get("GAME_DATE"), errors="coerce")
        training_start = training_dates.min()
        training_end = training_dates.max()
        training_start_date = (
            pd.Timestamp(training_start).strftime("%Y-%m-%d")
            if pd.notna(training_start)
            else None
        )
        training_end_date = (
            pd.Timestamp(training_end).strftime("%Y-%m-%d")
            if pd.notna(training_end)
            else None
        )

        # Map coefficient names
        model = {
            "coefficients": {
                COEF_NAME_MAP.get(k, k): v
                for k, v in raw_model["coefficients"].items()
            },
            "intercept": raw_model["intercept"],
            "r_squared": raw_model["r_squared"],
            "training_games": raw_model["training_games"],
            "training_start_date": training_start_date,
            "training_end_date": training_end_date,
        }
        league_avgs = compute_league_averages(training_clean)
        factor_percentiles = compute_factor_percentiles(training_clean)

        # Get games on this date
        date_games = current_rows[current_rows["GAME_DATE"] == game_date]
        home_games = date_games[date_games["IS_HOME"] == True]

        for _, home_row in home_games.iterrows():
            game_id = str(home_row["GAME_ID"])

            # Find matching road row
            road_matches = date_games[
                (date_games["GAME_ID"] == game_id) & (date_games["IS_HOME"] == False)
            ]
            if road_matches.empty:
                continue
            road_row = road_matches.iloc[0]

            # Get game info and linescore
            game_info = game_info_lookup.get(game_id, {
                "road": str(road_row.get("TEAM_ABBREVIATION", "")),
                "home": str(home_row.get("TEAM_ABBREVIATION", "")),
                "game_date": date_str,
                "game_type": "Regular Season",
            })
            ls_row = ls_lookup.get(game_id)
            adv_row = adv_lookup.get(game_id)

            # Build league_averages for output (convert to display scale)
            model_output = {
                "training_games": model["training_games"],
                "training_start_date": model.get("training_start_date"),
                "training_end_date": model.get("training_end_date"),
                "r_squared": round(model["r_squared"], 3),
                "intercept": round(model["intercept"], 2),
                "coefficients": {k: round(v, 1) for k, v in model["coefficients"].items()},
                "league_averages": {
                    "efg": round(league_avgs.get("EFG_PCT", 0) * 100, 1),
                    "ball_handling": round(league_avgs.get("BH", 0) * 100, 1),
                    "oreb": round(league_avgs.get("OREB_PCT", 0) * 100, 1),
                    "ft_rate": round(league_avgs.get("FT_RATE", 0) * 100, 1),
                    "off_rating": round(league_avgs.get("OFF_RATING", 0), 1),
                    "pace": round(league_avgs["PACE"], 1) if league_avgs.get("PACE") is not None else None,
                },
            }
            for key, value in (factor_percentiles or {}).items():
                model_output["league_averages"][key] = round(value * 100, 1)

            record = decompose_game(
                game_id=game_id,
                home_row=home_row,
                road_row=road_row,
                model_coefficients=model["coefficients"],
                league_avgs=league_avgs,
                game_info=game_info,
                linescore_row=ls_row,
                advanced_row=adv_row,
            )
            record["model"] = model_output
            all_records.append(record)

        if (i + 1) % 20 == 0 or i == len(game_dates) - 1:
            print(f"  [{i+1}/{len(game_dates)}] {date_str}: {len(all_records)} games processed so far")

    print(f"  Total games: {len(all_records)}")
    return all_records


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-season contribution JSON files for LLM game analysis"
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Specific season to process (e.g., 2024-25). Default: all seasons.",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help=f"Path to NBA_Data repository (default: {DEFAULT_REPO_DIR})",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir) if args.repo_dir else DEFAULT_REPO_DIR
    repo_dir = ensure_data_repo(repo_dir)

    # Create contributions directory
    contributions_dir = repo_dir / "contributions"
    contributions_dir.mkdir(parents=True, exist_ok=True)

    # Determine seasons to process
    if args.season:
        seasons = [args.season]
    else:
        seasons = get_available_seasons()

    print(f"Seasons to process: {len(seasons)}")
    print(f"Output directory: {contributions_dir}")

    total_start = time.time()
    total_games = 0

    for season in seasons:
        season_start = time.time()

        records = process_season(season, repo_dir)

        if not records:
            continue

        # Sort by game date
        records.sort(key=lambda r: r.get("game_info", {}).get("game_date", ""))

        output = {
            "season": season,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "games": records,
        }

        output_path = contributions_dir / f"contributions_{season}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        elapsed = time.time() - season_start
        total_games += len(records)
        print(f"  Saved: {output_path.name} ({len(records)} games, {elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"\nDone! {total_games} total games across {len(seasons)} seasons in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
