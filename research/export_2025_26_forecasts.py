#!/usr/bin/env python3
"""
Export out-of-sample 2025-26 game forecasts for six training-window models.

Output files:
  research/outputs/research_test_model_1.csv
  ...
  research/outputs/research_test_model_6.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
ADMIN_DIR = BACKEND_DIR / "admin"
sys.path.insert(0, str(ADMIN_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from cli import DEFAULT_REPO_DIR, _train_linear_model  # type: ignore  # noqa: E402
from config import get_available_seasons  # type: ignore  # noqa: E402
from generate_contributions import FEATURE_COLS, load_season_data, precompute_team_rows  # type: ignore  # noqa: E402


TARGET_COL = "NET_RATING"
TARGET_SEASON = "2025-26"
MIN_TRAINING_ROWS = 30


@dataclass(frozen=True)
class ModelSpec:
    model_num: int
    model_id: str
    mode: str
    value: int


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(1, "benchmark_prior_1", "fixed_prior_seasons", 1),
    ModelSpec(2, "prior_2", "fixed_prior_seasons", 2),
    ModelSpec(3, "prior_3", "fixed_prior_seasons", 3),
    ModelSpec(4, "prior_4", "fixed_prior_seasons", 4),
    ModelSpec(5, "rolling_1000_games", "rolling_total_games", 1000),
    ModelSpec(6, "rolling_2000_games", "rolling_total_games", 2000),
    ModelSpec(7, "prior_5", "fixed_prior_seasons", 5),
    ModelSpec(8, "prior_6", "fixed_prior_seasons", 6),
    ModelSpec(9, "prior_7", "fixed_prior_seasons", 7),
    ModelSpec(10, "prior_8", "fixed_prior_seasons", 8),
    ModelSpec(11, "prior_9", "fixed_prior_seasons", 9),
    ModelSpec(12, "prior_10", "fixed_prior_seasons", 10),
]


def parse_season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def load_team_rows_for_season(season: str, repo_dir: Path) -> pd.DataFrame:
    game_df, adv_df, _ = load_season_data(season, repo_dir)
    if game_df is None:
        return pd.DataFrame()

    game_df = game_df.copy()
    game_df["_season_tag"] = season
    game_dfs = [game_df]
    adv_dfs = [adv_df] if adv_df is not None else []
    team_df = precompute_team_rows(game_dfs, adv_dfs)
    if team_df.empty:
        return team_df

    team_df = team_df.copy()
    team_df["GAME_ID"] = team_df["GAME_ID"].astype(str)
    team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"], errors="coerce")
    team_df["IS_HOME"] = team_df["MATCHUP"].str.contains("vs.", na=False)
    team_df = team_df.dropna(subset=["GAME_ID", "GAME_DATE", TARGET_COL])
    return team_df


def previous_seasons(all_seasons: List[str], season: str, n: int) -> List[str]:
    idx = all_seasons.index(season)
    start = max(0, idx - n)
    return all_seasons[start:idx]


def latest_prior_games_rows(prior_rows: pd.DataFrame, n_games: int) -> pd.DataFrame:
    if n_games <= 0 or prior_rows.empty:
        return prior_rows.iloc[0:0].copy()

    games = (
        prior_rows[["GAME_ID", "GAME_DATE"]]
        .drop_duplicates(subset=["GAME_ID"])
        .sort_values(["GAME_DATE", "GAME_ID"], ascending=[False, False])
    )
    keep_ids = set(games.head(n_games)["GAME_ID"].tolist())
    return prior_rows[prior_rows["GAME_ID"].isin(keep_ids)].copy()


def build_training_rows(
    spec: ModelSpec,
    all_seasons: List[str],
    team_rows_by_season: Dict[str, pd.DataFrame],
    current_before: pd.DataFrame,
) -> pd.DataFrame:
    if spec.mode == "fixed_prior_seasons":
        prior_list = previous_seasons(all_seasons, TARGET_SEASON, spec.value)
        prior_parts = [team_rows_by_season[s] for s in prior_list if s in team_rows_by_season]
        prior_rows = pd.concat(prior_parts, ignore_index=True) if prior_parts else current_before.iloc[0:0].copy()
        return pd.concat([prior_rows, current_before], ignore_index=True)

    if spec.mode == "rolling_total_games":
        idx = all_seasons.index(TARGET_SEASON)
        prior_list = all_seasons[:idx]
        prior_parts = [team_rows_by_season[s] for s in prior_list if s in team_rows_by_season]
        prior_rows = pd.concat(prior_parts, ignore_index=True) if prior_parts else current_before.iloc[0:0].copy()

        current_games = int(current_before["GAME_ID"].nunique()) if not current_before.empty else 0
        need_prior_games = max(0, spec.value - current_games)
        prior_subset = latest_prior_games_rows(prior_rows, need_prior_games)
        return pd.concat([prior_subset, current_before], ignore_index=True)

    raise ValueError(f"Unsupported mode: {spec.mode}")


def model_forecast(frame: pd.DataFrame, model: dict) -> np.ndarray:
    coeff_vec = np.array([model["coefficients"][c] for c in FEATURE_COLS], dtype=float)
    x = frame[FEATURE_COLS].to_numpy(dtype=float)
    return model["intercept"] + x @ coeff_vec


def build_game_info_lookup(repo_dir: Path) -> dict:
    game_df, _, _ = load_season_data(TARGET_SEASON, repo_dir)
    if game_df is None:
        return {}

    game_df = game_df.copy()
    out: dict = {}
    for _, row in game_df.iterrows():
        gid = str(row.get("game_id", "")).strip()
        if not gid:
            continue
        out[gid] = {
            "Game Date": str(row.get("game_date", ""))[:10],
            "Road": str(row.get("team_abbreviation_road", "")),
            "Home": str(row.get("team_abbreviation_home", "")),
        }
    return out


def export_model_file(
    spec: ModelSpec,
    all_seasons: List[str],
    team_rows_by_season: Dict[str, pd.DataFrame],
    game_info_lookup: dict,
    output_dir: Path,
) -> Path:
    season_rows = team_rows_by_season[TARGET_SEASON].copy()
    season_rows = season_rows.sort_values(["GAME_DATE", "GAME_ID"])
    game_dates = sorted(season_rows["GAME_DATE"].dropna().unique())

    output_rows: List[dict] = []

    for game_date in game_dates:
        current_before = season_rows[season_rows["GAME_DATE"] < game_date]
        training_df = build_training_rows(spec, all_seasons, team_rows_by_season, current_before)
        training_clean = training_df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

        date_games = season_rows[(season_rows["GAME_DATE"] == game_date) & (season_rows["IS_HOME"] == True)].copy()
        date_games = date_games.dropna(subset=FEATURE_COLS + [TARGET_COL])
        if date_games.empty:
            continue

        if len(training_clean) < MIN_TRAINING_ROWS:
            preds = np.full(len(date_games), np.nan)
            c1 = c2 = c3 = c4 = np.nan
            first_date = ""
            last_date = ""
        else:
            model = _train_linear_model(training_clean, FEATURE_COLS, target_col=TARGET_COL)
            preds = model_forecast(date_games, model)
            c1 = float(model["coefficients"]["EFG_DIFF"])
            c2 = float(model["coefficients"]["TOV_DIFF"])
            c3 = float(model["coefficients"]["OREB_DIFF"])
            c4 = float(model["coefficients"]["FT_DIFF"])
            first_date = pd.Timestamp(training_clean["GAME_DATE"].min()).strftime("%Y-%m-%d")
            last_date = pd.Timestamp(training_clean["GAME_DATE"].max()).strftime("%Y-%m-%d")

        for idx, (_, row) in enumerate(date_games.iterrows()):
            gid = str(row["GAME_ID"])
            info = game_info_lookup.get(gid, {})
            game_date_str = info.get("Game Date", pd.Timestamp(row["GAME_DATE"]).strftime("%Y-%m-%d"))
            road = info.get("Road", "")
            home = info.get("Home", "")

            output_rows.append(
                {
                    "Game ID": gid,
                    "Game Date": game_date_str,
                    "Road": road,
                    "Home": home,
                    "Net Rating": float(row[TARGET_COL]),
                    "Forecast": float(preds[idx]) if pd.notna(preds[idx]) else np.nan,
                    "Date of First Game in Model": first_date,
                    "Date of Last Game in Model": last_date,
                    "Coefficient 1": c1,
                    "Coefficient 2": c2,
                    "Coefficient 3": c3,
                    "Coefficient 4": c4,
                }
            )

    out_df = pd.DataFrame(output_rows).sort_values(["Game Date", "Game ID"])
    out_path = output_dir / f"research_test_model_{spec.model_num}.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    repo_dir = Path(DEFAULT_REPO_DIR)
    output_dir = ROOT_DIR / "research" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_seasons = get_available_seasons()
    all_seasons = [s for s in all_seasons if parse_season_start_year(s) <= parse_season_start_year(TARGET_SEASON)]

    if TARGET_SEASON not in all_seasons:
        raise ValueError(f"{TARGET_SEASON} not found in available seasons.")

    print("Loading team rows cache...")
    team_rows_by_season: Dict[str, pd.DataFrame] = {}
    for season in all_seasons:
        team_rows_by_season[season] = load_team_rows_for_season(season, repo_dir)

    if team_rows_by_season[TARGET_SEASON].empty:
        raise ValueError(f"No team rows for {TARGET_SEASON}.")

    game_info_lookup = build_game_info_lookup(repo_dir)

    for spec in MODEL_SPECS:
        out_path = export_model_file(spec, all_seasons, team_rows_by_season, game_info_lookup, output_dir)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
