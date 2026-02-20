#!/usr/bin/env python3
"""
Out-of-sample net-rating model comparison for contribution-model training windows.

This script does not modify existing backend/frontend logic. It reuses the same
data preparation and linear model helper used by contribution generation, then
evaluates alternative training-window definitions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import sys

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
ADMIN_DIR = BACKEND_DIR / "admin"
sys.path.insert(0, str(ADMIN_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from cli import DEFAULT_REPO_DIR, _train_linear_model  # type: ignore  # noqa: E402
from config import get_available_seasons, SEASON_START_YEAR  # type: ignore  # noqa: E402
from generate_contributions import FEATURE_COLS, load_season_data, precompute_team_rows  # type: ignore  # noqa: E402


TARGET_COL = "NET_RATING"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    description: str
    mode: str
    value: int


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        model_id="benchmark_prior_1",
        description="Current season before date + 1 prior season",
        mode="fixed_prior_seasons",
        value=1,
    ),
    ModelSpec(
        model_id="prior_2",
        description="Current season before date + 2 prior seasons",
        mode="fixed_prior_seasons",
        value=2,
    ),
    ModelSpec(
        model_id="prior_3",
        description="Current season before date + 3 prior seasons",
        mode="fixed_prior_seasons",
        value=3,
    ),
    ModelSpec(
        model_id="prior_4",
        description="Current season before date + 4 prior seasons",
        mode="fixed_prior_seasons",
        value=4,
    ),
    ModelSpec(
        model_id="rolling_1000_games",
        description="Current season before date + enough prior games for 1000 total games",
        mode="rolling_total_games",
        value=1000,
    ),
    ModelSpec(
        model_id="rolling_2000_games",
        description="Current season before date + enough prior games for 2000 total games",
        mode="rolling_total_games",
        value=2000,
    ),
]


def parse_season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def previous_seasons(all_seasons: List[str], season: str, n: int) -> List[str]:
    idx = all_seasons.index(season)
    start = max(0, idx - n)
    return all_seasons[start:idx]


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
    team_df["_season_tag"] = season
    team_df = team_df.dropna(subset=["GAME_DATE", "GAME_ID", TARGET_COL])
    return team_df


def build_team_rows_cache(seasons: Iterable[str], repo_dir: Path) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    for season in seasons:
        cache[season] = load_team_rows_for_season(season, repo_dir)
    return cache


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
    season: str,
    current_before: pd.DataFrame,
) -> pd.DataFrame:
    if spec.mode == "fixed_prior_seasons":
        prior_list = previous_seasons(all_seasons, season, spec.value)
        prior_parts = [team_rows_by_season[s] for s in prior_list if s in team_rows_by_season]
        prior_rows = pd.concat(prior_parts, ignore_index=True) if prior_parts else current_before.iloc[0:0].copy()
        return pd.concat([prior_rows, current_before], ignore_index=True)

    if spec.mode == "rolling_total_games":
        idx = all_seasons.index(season)
        prior_list = all_seasons[:idx]
        prior_parts = [team_rows_by_season[s] for s in prior_list if s in team_rows_by_season]
        prior_rows = pd.concat(prior_parts, ignore_index=True) if prior_parts else current_before.iloc[0:0].copy()

        current_games = int(current_before["GAME_ID"].nunique()) if not current_before.empty else 0
        need_prior_games = max(0, spec.value - current_games)
        prior_subset = latest_prior_games_rows(prior_rows, need_prior_games)
        return pd.concat([prior_subset, current_before], ignore_index=True)

    raise ValueError(f"Unsupported model mode: {spec.mode}")


def predict_from_model(coefficients: Dict[str, float], intercept: float, frame: pd.DataFrame) -> np.ndarray:
    coeff_vec = np.array([coefficients[c] for c in FEATURE_COLS], dtype=float)
    x = frame[FEATURE_COLS].to_numpy(dtype=float)
    return intercept + x @ coeff_vec


def generate_predictions(
    history_seasons: List[str],
    evaluation_seasons: List[str],
    team_rows_by_season: Dict[str, pd.DataFrame],
    specs: List[ModelSpec],
    min_training_rows: int = 30,
) -> pd.DataFrame:
    rows: List[dict] = []

    for spec in specs:
        print(f"\n=== Evaluating {spec.model_id}: {spec.description} ===")

        for season in evaluation_seasons:
            season_rows = team_rows_by_season.get(season, pd.DataFrame())
            if season_rows.empty:
                continue

            season_rows = season_rows.sort_values(["GAME_DATE", "GAME_ID"]).copy()
            game_dates = sorted(season_rows["GAME_DATE"].dropna().unique())
            if not game_dates:
                continue

            season_pred_count = 0
            for game_date in game_dates:
                current_before = season_rows[season_rows["GAME_DATE"] < game_date]
                training_df = build_training_rows(
                    spec=spec,
                    all_seasons=history_seasons,
                    team_rows_by_season=team_rows_by_season,
                    season=season,
                    current_before=current_before,
                )

                training_clean = training_df.dropna(subset=FEATURE_COLS + [TARGET_COL])
                if len(training_clean) < min_training_rows:
                    continue

                try:
                    model = _train_linear_model(training_clean, FEATURE_COLS, target_col=TARGET_COL)
                except Exception:
                    continue

                date_rows = season_rows[(season_rows["GAME_DATE"] == game_date) & (season_rows["IS_HOME"] == True)].copy()
                date_rows = date_rows.dropna(subset=FEATURE_COLS + [TARGET_COL])
                if date_rows.empty:
                    continue

                preds = predict_from_model(
                    coefficients=model["coefficients"],
                    intercept=model["intercept"],
                    frame=date_rows,
                )

                for idx, (_, game_row) in enumerate(date_rows.iterrows()):
                    actual = float(game_row[TARGET_COL])
                    pred = float(preds[idx])
                    rows.append(
                        {
                            "model_id": spec.model_id,
                            "model_description": spec.description,
                            "season": season,
                            "game_id": str(game_row["GAME_ID"]),
                            "game_date": pd.Timestamp(game_row["GAME_DATE"]),
                            "actual_net_rating": actual,
                            "predicted_net_rating": pred,
                            "error": pred - actual,
                            "abs_error": abs(pred - actual),
                            "squared_error": (pred - actual) ** 2,
                        }
                    )
                    season_pred_count += 1

            print(f"  {season}: {season_pred_count} game predictions")

    if not rows:
        return pd.DataFrame(
            columns=[
                "model_id",
                "model_description",
                "season",
                "game_id",
                "game_date",
                "actual_net_rating",
                "predicted_net_rating",
                "error",
                "abs_error",
                "squared_error",
            ]
        )

    return pd.DataFrame(rows)


def summarize_game_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[dict] = []
    for model_id, grp in predictions.groupby("model_id"):
        corr = grp["predicted_net_rating"].corr(grp["actual_net_rating"])
        summary_rows.append(
            {
                "model_id": model_id,
                "model_description": grp["model_description"].iloc[0],
                "games": int(len(grp)),
                "corr": float(corr) if pd.notna(corr) else np.nan,
                "mae": float(grp["abs_error"].mean()),
                "rmse": float(np.sqrt(grp["squared_error"].mean())),
                "bias": float(grp["error"].mean()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("mae", ascending=True)


def summarize_period_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_level = (
        predictions.groupby(["model_id", "model_description", "season"], as_index=False)
        .agg(
            period_games=("game_id", "nunique"),
            actual_period_net=("actual_net_rating", "mean"),
            predicted_period_net=("predicted_net_rating", "mean"),
        )
    )
    period_level["period_error"] = period_level["predicted_period_net"] - period_level["actual_period_net"]
    period_level["period_abs_error"] = period_level["period_error"].abs()
    period_level["period_squared_error"] = period_level["period_error"] ** 2

    summary_rows: List[dict] = []
    for model_id, grp in period_level.groupby("model_id"):
        corr = grp["predicted_period_net"].corr(grp["actual_period_net"])
        summary_rows.append(
            {
                "model_id": model_id,
                "model_description": grp["model_description"].iloc[0],
                "periods": int(len(grp)),
                "period_corr": float(corr) if pd.notna(corr) else np.nan,
                "period_mae": float(grp["period_abs_error"].mean()),
                "period_rmse": float(np.sqrt(grp["period_squared_error"].mean())),
                "period_bias": float(grp["period_error"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("period_mae", ascending=True)
    return period_level, summary


def season_range_filter(
    seasons: List[str],
    start_season: str | None,
    end_season: str | None,
    include_first_season: bool,
) -> List[str]:
    out = seasons
    if start_season:
        out = [s for s in out if parse_season_start_year(s) >= parse_season_start_year(start_season)]
    if end_season:
        out = [s for s in out if parse_season_start_year(s) <= parse_season_start_year(end_season)]
    if not include_first_season:
        out = [s for s in out if parse_season_start_year(s) > SEASON_START_YEAR]
    return out


def history_seasons_for_evaluation(all_seasons: List[str], evaluation_seasons: List[str]) -> List[str]:
    if not evaluation_seasons:
        return []
    max_eval_year = max(parse_season_start_year(s) for s in evaluation_seasons)
    return [s for s in all_seasons if parse_season_start_year(s) <= max_eval_year]


def run_comparison(
    repo_dir: Path,
    output_dir: Path,
    start_season: str | None = None,
    end_season: str | None = None,
    include_first_season: bool = False,
    min_training_rows: int = 30,
) -> None:
    all_seasons = get_available_seasons()
    evaluation_seasons = season_range_filter(all_seasons, start_season, end_season, include_first_season)
    if not evaluation_seasons:
        raise ValueError("No seasons selected after applying filters.")
    history_seasons = history_seasons_for_evaluation(all_seasons, evaluation_seasons)

    print(f"Repo dir: {repo_dir}")
    print(
        f"Evaluation seasons: {evaluation_seasons[0]} to {evaluation_seasons[-1]} "
        f"({len(evaluation_seasons)})"
    )
    print(
        f"History seasons loaded for training windows: {history_seasons[0]} to {history_seasons[-1]} "
        f"({len(history_seasons)})"
    )
    print(f"Min training rows: {min_training_rows}")

    print("\nLoading and precomputing team rows...")
    team_rows_by_season = build_team_rows_cache(history_seasons, repo_dir)

    predictions = generate_predictions(
        history_seasons=history_seasons,
        evaluation_seasons=evaluation_seasons,
        team_rows_by_season=team_rows_by_season,
        specs=MODEL_SPECS,
        min_training_rows=min_training_rows,
    )
    if predictions.empty:
        raise ValueError("No predictions were generated. Check data availability and filters.")

    game_summary = summarize_game_metrics(predictions)
    period_level, period_summary = summarize_period_metrics(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "game_predictions.csv", index=False)
    game_summary.to_csv(output_dir / "game_metrics_summary.csv", index=False)
    period_level.to_csv(output_dir / "season_period_metrics.csv", index=False)
    period_summary.to_csv(output_dir / "season_period_summary.csv", index=False)

    print("\n=== Game-level summary (sorted by MAE) ===")
    print(game_summary.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
    print("\n=== Season-period summary (sorted by period MAE) ===")
    print(period_summary.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
    print(f"\nSaved outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare out-of-sample net-rating model windows.")
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=str(DEFAULT_REPO_DIR),
        help="Path to NBA_Data repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "research" / "outputs" / "net_rating_model_comparison"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--start-season",
        type=str,
        default=None,
        help="Optional start season (e.g., 2010-11).",
    )
    parser.add_argument(
        "--end-season",
        type=str,
        default=None,
        help="Optional end season (e.g., 2025-26).",
    )
    parser.add_argument(
        "--include-2000-01",
        action="store_true",
        help="Include 2000-01. Default excludes it because it's effectively in-sample in legacy workflow.",
    )
    parser.add_argument(
        "--min-training-rows",
        type=int,
        default=30,
        help="Minimum number of team-rows required to fit a model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_comparison(
        repo_dir=Path(args.repo_dir),
        output_dir=Path(args.output_dir),
        start_season=args.start_season,
        end_season=args.end_season,
        include_first_season=args.include_2000_01,
        min_training_rows=args.min_training_rows,
    )


if __name__ == "__main__":
    main()
