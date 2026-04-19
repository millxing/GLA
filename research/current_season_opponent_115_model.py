#!/usr/bin/env python3
"""
Explain current-season opponent-threshold split ratings with simple in-sample models.

The analysis matches League Summary's season definition:
  - include regular-season games
  - include nba_cup_semi
  - exclude nba_cup_final, playoffs, play_in

For the selected season, the script:
  1. builds team base ratings and split ratings
  2. computes a naive "base rating + league-average split delta" expectation
  3. fits weighted linear models for split ORtg and DRtg
  4. writes residual tables and a short markdown report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from config import (  # type: ignore  # noqa: E402
    DEFAULT_NBA_DATA_REPO_DIR,
    build_data_filename,
    resolve_data_file_path,
)


EXCLUDED_GAME_TYPES = {"nba_cup_final", "playoffs", "play_in"}
DEFAULT_SEASON = "2025-26"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "research" / "outputs" / "opponent_115_current_season_model"
THRESHOLD = 115.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model current-season opponent-115 split ratings")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season in YYYY-YY format")
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_NBA_DATA_REPO_DIR),
        help="Path to the NBA_Data repository",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated outputs",
    )
    return parser.parse_args()


def _normalize_game_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    if text == "playoff":
        return "playoffs"
    if text == "playin":
        return "play_in"
    if not text:
        return "regular_season"
    return text


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_num = pd.to_numeric(values, errors="coerce")
    weights_num = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = values_num.notna() & weights_num.gt(0)
    if not mask.any():
        return float("nan")
    return float(np.average(values_num[mask], weights=weights_num[mask]))


def _weighted_mae(actual: pd.Series, predicted: pd.Series, weights: pd.Series) -> float:
    err = (pd.to_numeric(actual, errors="coerce") - pd.to_numeric(predicted, errors="coerce")).abs()
    return _weighted_mean(err, weights)


def _weighted_rmse(actual: pd.Series, predicted: pd.Series, weights: pd.Series) -> float:
    err2 = (pd.to_numeric(actual, errors="coerce") - pd.to_numeric(predicted, errors="coerce")) ** 2
    mean_err2 = _weighted_mean(err2, weights)
    return float(math.sqrt(mean_err2)) if not math.isnan(mean_err2) else float("nan")


def _weighted_r2(actual: pd.Series, predicted: pd.Series, weights: pd.Series) -> float:
    y = pd.to_numeric(actual, errors="coerce")
    yhat = pd.to_numeric(predicted, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = y.notna() & yhat.notna() & w.gt(0)
    if not mask.any():
        return float("nan")
    y = y[mask]
    yhat = yhat[mask]
    w = w[mask]
    ybar = float(np.average(y, weights=w))
    sse = float(np.sum(w * (y - yhat) ** 2))
    sst = float(np.sum(w * (y - ybar) ** 2))
    if sst <= 0:
        return float("nan")
    return 1.0 - (sse / sst)


def _format_float(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _simple_table(df: pd.DataFrame) -> str:
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    lines = []
    lines.append(" | ".join(val.ljust(widths[idx]) for idx, val in enumerate(rows[0])))
    lines.append("-|-".join("-" * widths[idx] for idx in range(len(widths))))
    for row in rows[1:]:
        lines.append(" | ".join(val.ljust(widths[idx]) for idx, val in enumerate(row)))
    return "\n".join(lines)


def load_team_games(season: str, repo_dir: Path) -> pd.DataFrame:
    logs = pd.read_csv(resolve_data_file_path(build_data_filename("team_game_logs", season), repo_dir=repo_dir))
    adv = pd.read_csv(resolve_data_file_path(build_data_filename("box_score_advanced", season), repo_dir=repo_dir))

    logs = logs.copy()
    logs["game_type"] = logs.get("game_type", "regular_season").map(_normalize_game_type)
    logs = logs[~logs["game_type"].isin(EXCLUDED_GAME_TYPES)].copy()

    merged = logs.merge(
        adv[["game_id", "possessions_home", "possessions_road"]],
        on="game_id",
        how="inner",
    )

    home = pd.DataFrame(
        {
            "game_id": merged["game_id"],
            "game_date": pd.to_datetime(merged["game_date"], errors="coerce"),
            "game_type": merged["game_type"],
            "team": merged["team_abbreviation_home"],
            "opponent": merged["team_abbreviation_road"],
            "points_for": merged["pts_home"],
            "points_against": merged["pts_road"],
            "off_poss": merged["possessions_home"],
            "def_poss": merged["possessions_road"],
        }
    )
    road = pd.DataFrame(
        {
            "game_id": merged["game_id"],
            "game_date": pd.to_datetime(merged["game_date"], errors="coerce"),
            "game_type": merged["game_type"],
            "team": merged["team_abbreviation_road"],
            "opponent": merged["team_abbreviation_home"],
            "points_for": merged["pts_road"],
            "points_against": merged["pts_home"],
            "off_poss": merged["possessions_road"],
            "def_poss": merged["possessions_home"],
        }
    )
    return pd.concat([home, road], ignore_index=True)


def build_base_table(team_games: pd.DataFrame) -> pd.DataFrame:
    base = team_games.groupby("team", as_index=False).agg(
        games=("game_id", "count"),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
        off_poss=("off_poss", "sum"),
        def_poss=("def_poss", "sum"),
    )
    base["base_off_rating"] = base["points_for"] / base["off_poss"] * 100.0
    base["base_def_rating"] = base["points_against"] / base["def_poss"] * 100.0
    base["base_net_rating"] = base["base_off_rating"] - base["base_def_rating"]
    return base


def build_split_table(team_games: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    opp_lookup = base.set_index("team")[["base_off_rating", "base_def_rating"]].rename(
        columns={
            "base_off_rating": "opp_season_off_rating",
            "base_def_rating": "opp_season_def_rating",
        }
    )
    work = team_games.join(opp_lookup, on="opponent", how="left")
    work["opp_season_net_rating"] = work["opp_season_off_rating"] - work["opp_season_def_rating"]

    split_specs = [
        ("opp_off_rating", "opp_season_off_rating"),
        ("opp_def_rating", "opp_season_def_rating"),
    ]
    pieces: list[pd.DataFrame] = []
    for split_type, column in split_specs:
        for bucket_label, mask, bucket_is_gt in [
            ("opp_gt_115", work[column] > THRESHOLD, 1),
            ("opp_lte_115", work[column] <= THRESHOLD, 0),
        ]:
            subset = work.loc[mask].copy()
            grouped = subset.groupby("team", as_index=False).agg(
                split_games=("game_id", "count"),
                points_for=("points_for", "sum"),
                points_against=("points_against", "sum"),
                off_poss=("off_poss", "sum"),
                def_poss=("def_poss", "sum"),
                opp_mean_off_rating=("opp_season_off_rating", "mean"),
                opp_mean_def_rating=("opp_season_def_rating", "mean"),
                opp_mean_net_rating=("opp_season_net_rating", "mean"),
            )
            grouped["split_type"] = split_type
            grouped["bucket"] = bucket_label
            grouped["bucket_is_gt_115"] = bucket_is_gt
            grouped["split_off_rating"] = grouped["points_for"] / grouped["off_poss"] * 100.0
            grouped["split_def_rating"] = grouped["points_against"] / grouped["def_poss"] * 100.0
            grouped["split_net_rating"] = grouped["split_off_rating"] - grouped["split_def_rating"]
            pieces.append(grouped)

    split_df = pd.concat(pieces, ignore_index=True)
    split_df = split_df.merge(
        base[["team", "games", "base_off_rating", "base_def_rating", "base_net_rating"]],
        on="team",
        how="left",
    )
    split_df["split_game_share"] = split_df["split_games"] / split_df["games"].replace(0, pd.NA)
    return split_df.sort_values(["split_type", "bucket", "team"]).reset_index(drop=True)


def fit_models(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "base_off_rating",
        "base_def_rating",
        "bucket_is_gt_115",
        "split_games",
        "split_game_share",
        "opp_mean_off_rating",
        "opp_mean_def_rating",
    ]

    metric_rows: list[dict[str, Any]] = []
    predicted_frames: list[pd.DataFrame] = []

    for split_type in sorted(split_df["split_type"].unique()):
        frame = split_df[split_df["split_type"] == split_type].copy()
        weights = frame["split_games"].astype(float)

        for target in ["split_off_rating", "split_def_rating"]:
            base_col = "base_off_rating" if target == "split_off_rating" else "base_def_rating"

            delta_col = f"{target}_minus_base"
            frame[delta_col] = frame[target] - frame[base_col]
            bucket_deltas = (
                frame.groupby("bucket")
                .apply(lambda g: _weighted_mean(g[delta_col], g["split_games"]), include_groups=False)
                .to_dict()
            )
            frame[f"naive_pred_{target}"] = frame.apply(
                lambda row: row[base_col] + float(bucket_deltas.get(str(row["bucket"]), 0.0)),
                axis=1,
            )

            model = LinearRegression()
            model.fit(frame[feature_cols], frame[target], sample_weight=weights)
            frame[f"model_pred_{target}"] = model.predict(frame[feature_cols])
            frame[f"naive_resid_{target}"] = frame[target] - frame[f"naive_pred_{target}"]
            frame[f"model_resid_{target}"] = frame[target] - frame[f"model_pred_{target}"]

            metric_rows.append(
                {
                    "split_type": split_type,
                    "target": target,
                    "rows": int(len(frame)),
                    "weighted_games": float(weights.sum()),
                    "naive_weighted_mae": _weighted_mae(frame[target], frame[f"naive_pred_{target}"], weights),
                    "model_weighted_mae": _weighted_mae(frame[target], frame[f"model_pred_{target}"], weights),
                    "naive_weighted_rmse": _weighted_rmse(frame[target], frame[f"naive_pred_{target}"], weights),
                    "model_weighted_rmse": _weighted_rmse(frame[target], frame[f"model_pred_{target}"], weights),
                    "naive_weighted_r2": _weighted_r2(frame[target], frame[f"naive_pred_{target}"], weights),
                    "model_weighted_r2": _weighted_r2(frame[target], frame[f"model_pred_{target}"], weights),
                    "intercept": float(model.intercept_),
                    **{f"coef_{col}": float(coef) for col, coef in zip(feature_cols, model.coef_)},
                }
            )

        frame["naive_pred_split_net_rating"] = (
            frame["naive_pred_split_off_rating"] - frame["naive_pred_split_def_rating"]
        )
        frame["model_pred_split_net_rating"] = (
            frame["model_pred_split_off_rating"] - frame["model_pred_split_def_rating"]
        )
        frame["naive_resid_split_net_rating"] = frame["split_net_rating"] - frame["naive_pred_split_net_rating"]
        frame["model_resid_split_net_rating"] = frame["split_net_rating"] - frame["model_pred_split_net_rating"]
        predicted_frames.append(frame)

    predictions = pd.concat(predicted_frames, ignore_index=True).sort_values(
        ["split_type", "bucket", "team"]
    )
    metrics = pd.DataFrame(metric_rows).sort_values(["split_type", "target"]).reset_index(drop=True)
    return predictions, metrics


def build_report(
    season: str,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> str:
    metrics_view = metrics[
        [
            "split_type",
            "target",
            "naive_weighted_mae",
            "model_weighted_mae",
            "naive_weighted_r2",
            "model_weighted_r2",
        ]
    ].copy()
    for col in ["naive_weighted_mae", "model_weighted_mae", "naive_weighted_r2", "model_weighted_r2"]:
        metrics_view[col] = metrics_view[col].map(lambda x: _format_float(x, 3))

    key_rows = []
    for split_type, split_label in [
        ("opp_off_rating", "Split by opponent season ORtg"),
        ("opp_def_rating", "Split by opponent season DRtg"),
    ]:
        for bucket, bucket_label in [("opp_gt_115", "> 115"), ("opp_lte_115", "<= 115")]:
            subset = predictions[
                (predictions["split_type"] == split_type) & (predictions["bucket"] == bucket)
            ].copy()
            top_pos = subset.nlargest(5, "model_resid_split_net_rating")[
                ["team", "split_net_rating", "model_pred_split_net_rating", "model_resid_split_net_rating"]
            ].copy()
            top_neg = subset.nsmallest(5, "model_resid_split_net_rating")[
                ["team", "split_net_rating", "model_pred_split_net_rating", "model_resid_split_net_rating"]
            ].copy()
            for frame in (top_pos, top_neg):
                for col in ["split_net_rating", "model_pred_split_net_rating", "model_resid_split_net_rating"]:
                    frame[col] = frame[col].map(lambda x: _format_float(x, 2))
            key_rows.append((split_label, bucket_label, top_pos, top_neg))

    report_lines = [
        f"# {season} Opponent-115 Split Model",
        "",
        "This is an in-sample explanatory model for the current season only.",
        "Season definition matches League Summary: include `regular_season` and `nba_cup_semi`, exclude `nba_cup_final`.",
        "",
        "Method:",
        "- Build each team's base season ORtg/DRtg/Net and split ratings in the two opponent buckets.",
        "- Naive baseline: `base rating + league-average split delta` within the same split bucket.",
        "- Model: weighted linear regression using base ORtg, base DRtg, bucket indicator, split games/share, and mean opponent season ORtg/DRtg inside the bucket.",
        "- Sample weights are split games, so 40-game splits count more than 32-game splits.",
        "",
        "## Fit Summary",
        "",
        "```text",
        _simple_table(metrics_view),
        "```",
        "",
    ]

    for split_label, bucket_label, top_pos, top_neg in key_rows:
        report_lines.extend(
            [
                f"## {split_label}, bucket {bucket_label}",
                "",
                "Most positive net residuals:",
                "",
                "```text",
                _simple_table(top_pos.rename(columns={
                    "split_net_rating": "actual_net",
                    "model_pred_split_net_rating": "expected_net",
                    "model_resid_split_net_rating": "residual_net",
                })),
                "```",
                "",
                "Most negative net residuals:",
                "",
                "```text",
                _simple_table(top_neg.rename(columns={
                    "split_net_rating": "actual_net",
                    "model_pred_split_net_rating": "expected_net",
                    "model_resid_split_net_rating": "residual_net",
                })),
                "```",
                "",
            ]
        )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines))
    return "\n".join(report_lines)


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    team_games = load_team_games(season=args.season, repo_dir=repo_dir)
    base = build_base_table(team_games)
    split_df = build_split_table(team_games, base)
    predictions, metrics = fit_models(split_df)

    split_df.to_csv(output_dir / "team_split_inputs.csv", index=False)
    predictions.to_csv(output_dir / "team_split_predictions.csv", index=False)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)

    summary = {
        "season": args.season,
        "repo_dir": str(repo_dir),
        "rows": int(len(predictions)),
        "teams": int(predictions["team"].nunique()),
        "split_types": sorted(predictions["split_type"].unique().tolist()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    build_report(args.season, predictions=predictions, metrics=metrics, output_dir=output_dir)

    print(output_dir)


if __name__ == "__main__":
    main()
