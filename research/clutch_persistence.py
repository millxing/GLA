#!/usr/bin/env python3
"""
Measure whether team clutch-time net rating is persistent or mostly random.

This script reuses the persisted clutch scope already produced in NBA_Data:
  - team_game_logs_clutch_<season>.csv
  - box_score_advanced_clutch_<season>.csv
alongside the full-season game logs to build team-season, split-half, and
year-to-year persistence datasets.

Outputs are written to research/outputs/clutch_persistence/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from config import DEFAULT_NBA_DATA_REPO_DIR, get_available_seasons  # type: ignore  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT_DIR / "research" / "outputs" / "clutch_persistence"
DEFAULT_MIN_HALF_GAMES = 10
DEFAULT_MIN_HALF_POSS = 100
DEFAULT_MIN_YOY_POSS = 150
DEFAULT_THRESHOLDS = (75, 100, 125, 150)
DEFAULT_PERMUTATIONS = 5000
DEFAULT_SEED = 42
EXCLUDED_GAME_TYPES = {"playoffs", "play_in", "nba_cup_final"}
METRIC_SPECS = {
    "raw_clutch_net": {
        "label": "Raw clutch net rating",
        "column": "clutch_net_rating",
        "pair_column": "clutch_net_rating",
    },
    "clutch_minus_non_clutch": {
        "label": "Clutch minus non-clutch net rating",
        "column": "clutch_minus_non_clutch_net",
        "pair_column": "clutch_minus_non_clutch_net",
    },
}


@dataclass(frozen=True)
class AnalysisConfig:
    repo_dir: Path
    output_dir: Path
    include_playoffs: bool
    min_half_games: int
    min_half_poss: int
    min_yoy_poss: int
    permutations: int
    seed: int
    thresholds: tuple[int, ...]


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(description="Analyze clutch net-rating persistence")
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_NBA_DATA_REPO_DIR),
        help="Path to the NBA_Data repository",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated research outputs",
    )
    parser.add_argument(
        "--include-playoffs",
        action="store_true",
        help="Include playoff and play-in games in the analysis",
    )
    parser.add_argument(
        "--min-half-games",
        type=int,
        default=DEFAULT_MIN_HALF_GAMES,
        help="Minimum clutch games required in each split half",
    )
    parser.add_argument(
        "--min-half-poss",
        type=int,
        default=DEFAULT_MIN_HALF_POSS,
        help="Minimum clutch possessions required in each split half",
    )
    parser.add_argument(
        "--min-yoy-poss",
        type=int,
        default=DEFAULT_MIN_YOY_POSS,
        help="Minimum clutch possessions required in each adjacent season",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="Number of permutation draws for null benchmarks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for permutation benchmarks",
    )
    args = parser.parse_args()

    return AnalysisConfig(
        repo_dir=Path(args.repo_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        include_playoffs=bool(args.include_playoffs),
        min_half_games=max(1, int(args.min_half_games)),
        min_half_poss=max(1, int(args.min_half_poss)),
        min_yoy_poss=max(1, int(args.min_yoy_poss)),
        permutations=max(100, int(args.permutations)),
        seed=int(args.seed),
        thresholds=tuple(sorted(set(int(x) for x in DEFAULT_THRESHOLDS))),
    )


def _season_start_year(season: str) -> int:
    return int(str(season).split("-")[0])


def _normalize_game_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    if text == "playoff":
        return "playoffs"
    if text == "playin":
        return "play_in"
    if not text:
        return "regular_season"
    return text


def _normalize_game_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else text


def _franchise_key(value: Any, team: Any) -> str:
    if pd.notna(value):
        try:
            return str(int(float(value)))
        except Exception:
            text = str(value).strip()
            if text:
                return text
    return str(team or "").strip().upper()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _load_scope_frames(season: str, repo_dir: Path, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_name = f"team_game_logs_{season}.csv" if scope == "all" else f"team_game_logs_{scope}_{season}.csv"
    adv_name = f"box_score_advanced_{season}.csv" if scope == "all" else f"box_score_advanced_{scope}_{season}.csv"
    return _read_csv(repo_dir / game_name), _read_csv(repo_dir / adv_name)


def _prepare_merged_games(
    game_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    season: str,
    include_playoffs: bool,
) -> pd.DataFrame:
    games = game_df.copy()
    adv = adv_df.copy()

    games["game_id"] = games["game_id"].map(_normalize_game_id)
    adv["game_id"] = adv["game_id"].map(_normalize_game_id)
    games["season"] = season
    adv["season"] = season
    games["game_date"] = pd.to_datetime(games["game_date"], format="%Y-%m-%d", errors="coerce")
    games["game_type"] = games.get("game_type", "regular_season").map(_normalize_game_type)

    merged = games.merge(
        adv[
            [
                "game_id",
                "season",
                "team_id_home",
                "team_abbreviation_home",
                "minutes_home",
                "possessions_home",
                "team_id_road",
                "team_abbreviation_road",
                "minutes_road",
                "possessions_road",
            ]
        ],
        on=["game_id", "season"],
        how="inner",
        suffixes=("", "_adv"),
    )

    if not include_playoffs:
        merged = merged[~merged["game_type"].isin(EXCLUDED_GAME_TYPES)].copy()

    if merged.empty:
        return merged

    merged["team_abbreviation_home"] = (
        merged["team_abbreviation_home_adv"].combine_first(merged["team_abbreviation_home"])
    )
    merged["team_abbreviation_road"] = (
        merged["team_abbreviation_road_adv"].combine_first(merged["team_abbreviation_road"])
    )
    return merged


def _build_team_game_rows(merged: pd.DataFrame, season: str, scope: str) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "scope",
                "franchise_id",
                "team",
                "opponent",
                "game_id",
                "game_date",
                "game_type",
                "home_away",
                "pts",
                "opp_pts",
                "possessions",
                "opp_possessions",
                "minutes",
                "opp_minutes",
            ]
        )

    home = pd.DataFrame(
        {
            "season": season,
            "scope": scope,
            "franchise_id": merged.apply(
                lambda row: _franchise_key(row["team_id_home"], row["team_abbreviation_home"]), axis=1
            ),
            "team": merged["team_abbreviation_home"].astype(str).str.strip(),
            "opponent": merged["team_abbreviation_road"].astype(str).str.strip(),
            "game_id": merged["game_id"],
            "game_date": merged["game_date"],
            "game_type": merged["game_type"],
            "home_away": "home",
            "pts": pd.to_numeric(merged["pts_home"], errors="coerce").fillna(0.0),
            "opp_pts": pd.to_numeric(merged["pts_road"], errors="coerce").fillna(0.0),
            "possessions": pd.to_numeric(merged["possessions_home"], errors="coerce").fillna(0.0),
            "opp_possessions": pd.to_numeric(merged["possessions_road"], errors="coerce").fillna(0.0),
            "minutes": pd.to_numeric(merged["minutes_home"], errors="coerce").fillna(0.0),
            "opp_minutes": pd.to_numeric(merged["minutes_road"], errors="coerce").fillna(0.0),
        }
    )
    road = pd.DataFrame(
        {
            "season": season,
            "scope": scope,
            "franchise_id": merged.apply(
                lambda row: _franchise_key(row["team_id_road"], row["team_abbreviation_road"]), axis=1
            ),
            "team": merged["team_abbreviation_road"].astype(str).str.strip(),
            "opponent": merged["team_abbreviation_home"].astype(str).str.strip(),
            "game_id": merged["game_id"],
            "game_date": merged["game_date"],
            "game_type": merged["game_type"],
            "home_away": "road",
            "pts": pd.to_numeric(merged["pts_road"], errors="coerce").fillna(0.0),
            "opp_pts": pd.to_numeric(merged["pts_home"], errors="coerce").fillna(0.0),
            "possessions": pd.to_numeric(merged["possessions_road"], errors="coerce").fillna(0.0),
            "opp_possessions": pd.to_numeric(merged["possessions_home"], errors="coerce").fillna(0.0),
            "minutes": pd.to_numeric(merged["minutes_road"], errors="coerce").fillna(0.0),
            "opp_minutes": pd.to_numeric(merged["minutes_home"], errors="coerce").fillna(0.0),
        }
    )
    out = pd.concat([home, road], ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out.sort_values(["season", "team", "game_date", "game_id"]).reset_index(drop=True)


def _load_rows_for_season(season: str, config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_games, all_adv = _load_scope_frames(season=season, repo_dir=config.repo_dir, scope="all")
    clutch_games, clutch_adv = _load_scope_frames(season=season, repo_dir=config.repo_dir, scope="clutch")
    all_merged = _prepare_merged_games(all_games, all_adv, season=season, include_playoffs=config.include_playoffs)
    clutch_merged = _prepare_merged_games(
        clutch_games, clutch_adv, season=season, include_playoffs=config.include_playoffs
    )
    return (
        _build_team_game_rows(all_merged, season=season, scope="all"),
        _build_team_game_rows(clutch_merged, season=season, scope="clutch"),
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def _round(value: Any, digits: int = 3) -> float:
    return round(float(value), digits)


def _describe(series: Sequence[float]) -> dict[str, float]:
    values = np.asarray([float(x) for x in series if pd.notna(x)], dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": _round(np.mean(values)),
        "std": _round(np.std(values, ddof=1) if values.size > 1 else 0.0),
        "min": _round(np.min(values)),
        "p25": _round(np.percentile(values, 25)),
        "median": _round(np.percentile(values, 50)),
        "p75": _round(np.percentile(values, 75)),
        "max": _round(np.max(values)),
    }


def _summarize_rows(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "games": 0,
            "minutes": 0.0,
            "possessions": 0.0,
            "opp_possessions": 0.0,
            "pts": 0.0,
            "opp_pts": 0.0,
            "off_rating": 0.0,
            "def_rating": 0.0,
            "net_rating": 0.0,
        }

    pts = float(pd.to_numeric(rows["pts"], errors="coerce").fillna(0.0).sum())
    opp_pts = float(pd.to_numeric(rows["opp_pts"], errors="coerce").fillna(0.0).sum())
    poss = float(pd.to_numeric(rows["possessions"], errors="coerce").fillna(0.0).sum())
    opp_poss = float(pd.to_numeric(rows["opp_possessions"], errors="coerce").fillna(0.0).sum())
    minutes = float(pd.to_numeric(rows["minutes"], errors="coerce").fillna(0.0).sum())
    off_rating = _safe_rate(pts, poss)
    def_rating = _safe_rate(opp_pts, opp_poss)
    return {
        "games": int(rows["game_id"].nunique()),
        "minutes": minutes,
        "possessions": poss,
        "opp_possessions": opp_poss,
        "pts": pts,
        "opp_pts": opp_pts,
        "off_rating": off_rating,
        "def_rating": def_rating,
        "net_rating": off_rating - def_rating,
    }


def _combine_segment_metrics(all_rows: pd.DataFrame, clutch_rows: pd.DataFrame) -> dict[str, Any]:
    all_stats = _summarize_rows(all_rows)
    clutch_stats = _summarize_rows(clutch_rows)

    non_clutch_pts = max(0.0, all_stats["pts"] - clutch_stats["pts"])
    non_clutch_opp_pts = max(0.0, all_stats["opp_pts"] - clutch_stats["opp_pts"])
    non_clutch_poss = max(0.0, all_stats["possessions"] - clutch_stats["possessions"])
    non_clutch_opp_poss = max(0.0, all_stats["opp_possessions"] - clutch_stats["opp_possessions"])
    non_clutch_minutes = max(0.0, all_stats["minutes"] - clutch_stats["minutes"])

    non_clutch_off = _safe_rate(non_clutch_pts, non_clutch_poss)
    non_clutch_def = _safe_rate(non_clutch_opp_pts, non_clutch_opp_poss)
    non_clutch_net = non_clutch_off - non_clutch_def

    return {
        "all_games": all_stats["games"],
        "all_minutes": _round(all_stats["minutes"]),
        "all_possessions": _round(all_stats["possessions"]),
        "all_opp_possessions": _round(all_stats["opp_possessions"]),
        "all_pts": _round(all_stats["pts"]),
        "all_opp_pts": _round(all_stats["opp_pts"]),
        "all_off_rating": _round(all_stats["off_rating"]),
        "all_def_rating": _round(all_stats["def_rating"]),
        "all_net_rating": _round(all_stats["net_rating"]),
        "clutch_games": clutch_stats["games"],
        "clutch_minutes": _round(clutch_stats["minutes"]),
        "clutch_possessions": _round(clutch_stats["possessions"]),
        "clutch_opp_possessions": _round(clutch_stats["opp_possessions"]),
        "clutch_pts": _round(clutch_stats["pts"]),
        "clutch_opp_pts": _round(clutch_stats["opp_pts"]),
        "clutch_off_rating": _round(clutch_stats["off_rating"]),
        "clutch_def_rating": _round(clutch_stats["def_rating"]),
        "clutch_net_rating": _round(clutch_stats["net_rating"]),
        "non_clutch_games": all_stats["games"],
        "non_clutch_minutes": _round(non_clutch_minutes),
        "non_clutch_possessions": _round(non_clutch_poss),
        "non_clutch_opp_possessions": _round(non_clutch_opp_poss),
        "non_clutch_pts": _round(non_clutch_pts),
        "non_clutch_opp_pts": _round(non_clutch_opp_pts),
        "non_clutch_off_rating": _round(non_clutch_off),
        "non_clutch_def_rating": _round(non_clutch_def),
        "non_clutch_net_rating": _round(non_clutch_net),
        "clutch_minus_non_clutch_net": _round(clutch_stats["net_rating"] - non_clutch_net),
        "clutch_minus_all_net": _round(clutch_stats["net_rating"] - all_stats["net_rating"]),
    }


def build_team_season_metrics(all_rows: pd.DataFrame, clutch_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_groups = {(season, fid): grp.copy() for (season, fid), grp in all_rows.groupby(["season", "franchise_id"])}
    clutch_groups = {
        (season, fid): grp.copy() for (season, fid), grp in clutch_rows.groupby(["season", "franchise_id"])
    }

    for season, franchise_id in sorted(all_groups.keys()):
        team_all = all_groups[(season, franchise_id)]
        team_clutch = clutch_groups.get((season, franchise_id), team_all.iloc[0:0].copy())
        metrics = _combine_segment_metrics(team_all, team_clutch)
        team_name = str(team_all["team"].mode(dropna=True).iat[0]) if not team_all["team"].mode(dropna=True).empty else ""
        rows.append(
            {
                "season": season,
                "season_start_year": _season_start_year(season),
                "franchise_id": franchise_id,
                "team": team_name,
                **metrics,
            }
        )

    out = pd.DataFrame(rows).sort_values(["season_start_year", "team"]).reset_index(drop=True)
    out["clutch_time_pct"] = (
        pd.to_numeric(out["clutch_minutes"], errors="coerce")
        / pd.to_numeric(out["all_minutes"], errors="coerce").replace(0, np.nan)
        * 100.0
    ).fillna(0.0).round(3)
    out["clutch_poss_pct"] = (
        pd.to_numeric(out["clutch_possessions"], errors="coerce")
        / pd.to_numeric(out["all_possessions"], errors="coerce").replace(0, np.nan)
        * 100.0
    ).fillna(0.0).round(3)
    return out


def build_split_half_table(all_rows: pd.DataFrame, clutch_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_groups = {(season, fid): grp.copy() for (season, fid), grp in all_rows.groupby(["season", "franchise_id"])}

    for (season, franchise_id), clutch_team in clutch_rows.groupby(["season", "franchise_id"]):
        clutch_team = clutch_team.sort_values(["game_date", "game_id"]).copy()
        all_team = all_groups.get((season, franchise_id))
        if all_team is None or clutch_team.empty:
            continue

        team_name = str(clutch_team["team"].mode(dropna=True).iat[0]) if not clutch_team["team"].mode(dropna=True).empty else ""
        clutch_games = (
            clutch_team[["game_id", "game_date"]]
            .drop_duplicates(subset=["game_id"])
            .sort_values(["game_date", "game_id"])
        )
        game_ids = clutch_games["game_id"].tolist()
        split_idx = len(game_ids) // 2
        first_ids = set(game_ids[:split_idx])
        second_ids = set(game_ids[split_idx:])

        if not first_ids or not second_ids:
            continue

        first_all = all_team[all_team["game_id"].isin(first_ids)].copy()
        first_clutch = clutch_team[clutch_team["game_id"].isin(first_ids)].copy()
        second_all = all_team[all_team["game_id"].isin(second_ids)].copy()
        second_clutch = clutch_team[clutch_team["game_id"].isin(second_ids)].copy()

        first_metrics = _combine_segment_metrics(first_all, first_clutch)
        second_metrics = _combine_segment_metrics(second_all, second_clutch)

        row: dict[str, Any] = {
            "season": season,
            "season_start_year": _season_start_year(season),
            "franchise_id": franchise_id,
            "team": team_name,
            "clutch_games_total": int(clutch_team["game_id"].nunique()),
        }
        for prefix, metrics in (("first", first_metrics), ("second", second_metrics)):
            for key, value in metrics.items():
                row[f"{key}_{prefix}"] = value
        row["eligible_default"] = bool(
            row["clutch_games_first"] >= DEFAULT_MIN_HALF_GAMES
            and row["clutch_games_second"] >= DEFAULT_MIN_HALF_GAMES
            and row["clutch_possessions_first"] >= DEFAULT_MIN_HALF_POSS
            and row["clutch_possessions_second"] >= DEFAULT_MIN_HALF_POSS
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["season_start_year", "team"]).reset_index(drop=True)


def build_year_to_year_pairs(team_season_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seasons = sorted(team_season_df["season"].unique(), key=_season_start_year)

    for idx in range(len(seasons) - 1):
        season_t = seasons[idx]
        season_t1 = seasons[idx + 1]
        current_df = team_season_df[team_season_df["season"] == season_t].copy()
        next_df = team_season_df[team_season_df["season"] == season_t1].copy()
        merged = current_df.merge(
            next_df,
            on="franchise_id",
            how="inner",
            suffixes=("_t", "_t1"),
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "season_t": season_t,
                    "season_t1": season_t1,
                    "season_pair": f"{season_t}->{season_t1}",
                    "franchise_id": row["franchise_id"],
                    "team_t": row["team_t"],
                    "team_t1": row["team_t1"],
                    "clutch_possessions_t": row["clutch_possessions_t"],
                    "clutch_possessions_t1": row["clutch_possessions_t1"],
                    "clutch_games_t": row["clutch_games_t"],
                    "clutch_games_t1": row["clutch_games_t1"],
                    "clutch_net_rating_t": row["clutch_net_rating_t"],
                    "clutch_net_rating_t1": row["clutch_net_rating_t1"],
                    "non_clutch_net_rating_t": row["non_clutch_net_rating_t"],
                    "non_clutch_net_rating_t1": row["non_clutch_net_rating_t1"],
                    "all_net_rating_t": row["all_net_rating_t"],
                    "all_net_rating_t1": row["all_net_rating_t1"],
                    "clutch_minus_non_clutch_net_t": row["clutch_minus_non_clutch_net_t"],
                    "clutch_minus_non_clutch_net_t1": row["clutch_minus_non_clutch_net_t1"],
                    "clutch_minus_all_net_t": row["clutch_minus_all_net_t"],
                    "clutch_minus_all_net_t1": row["clutch_minus_all_net_t1"],
                }
            )

    out = pd.DataFrame(rows).sort_values(["season_t", "team_t"]).reset_index(drop=True)
    if out.empty:
        return out
    out["eligible_default"] = (
        (pd.to_numeric(out["clutch_possessions_t"], errors="coerce") >= DEFAULT_MIN_YOY_POSS)
        & (pd.to_numeric(out["clutch_possessions_t1"], errors="coerce") >= DEFAULT_MIN_YOY_POSS)
    )
    return out


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_vals = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float)
    if x_vals.size < 2 or y_vals.size < 2:
        return None
    if np.allclose(x_vals, x_vals[0]) or np.allclose(y_vals, y_vals[0]):
        return None
    return float(np.corrcoef(x_vals, y_vals)[0, 1])


def _spearman_brown(r: Optional[float]) -> Optional[float]:
    if r is None:
        return None
    denom = 1.0 + float(r)
    if abs(denom) < 1e-12:
        return None
    return float((2.0 * float(r)) / denom)


def _weighted_regression(y: np.ndarray, x_cols: list[np.ndarray], weights: np.ndarray) -> dict[str, Any]:
    if y.size == 0:
        return {"coefficients": [], "intercept": None, "r2": None}
    design = np.column_stack([np.ones(y.size)] + x_cols)
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    xtwx = design.T @ (w[:, None] * design)
    xtwy = design.T @ (w * y)
    beta = np.linalg.pinv(xtwx) @ xtwy
    fitted = design @ beta
    y_bar = float(np.average(y, weights=w))
    ss_res = float(np.sum(w * (y - fitted) ** 2))
    ss_tot = float(np.sum(w * (y - y_bar) ** 2))
    r2 = None if ss_tot <= 0 else float(1.0 - (ss_res / ss_tot))
    return {
        "intercept": float(beta[0]),
        "coefficients": [float(v) for v in beta[1:]],
        "r2": r2,
    }


def _within_sample_for_threshold(split_df: pd.DataFrame, threshold: int, min_half_games: int) -> pd.DataFrame:
    if split_df.empty:
        return split_df.copy()
    mask = (
        (pd.to_numeric(split_df["clutch_games_first"], errors="coerce") >= min_half_games)
        & (pd.to_numeric(split_df["clutch_games_second"], errors="coerce") >= min_half_games)
        & (pd.to_numeric(split_df["clutch_possessions_first"], errors="coerce") >= threshold)
        & (pd.to_numeric(split_df["clutch_possessions_second"], errors="coerce") >= threshold)
    )
    return split_df[mask].copy()


def _yoy_sample_for_threshold(yoy_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if yoy_df.empty:
        return yoy_df.copy()
    mask = (
        (pd.to_numeric(yoy_df["clutch_possessions_t"], errors="coerce") >= threshold)
        & (pd.to_numeric(yoy_df["clutch_possessions_t1"], errors="coerce") >= threshold)
    )
    return yoy_df[mask].copy()


def analyze_within_season(split_df: pd.DataFrame, threshold: int, min_half_games: int) -> list[dict[str, Any]]:
    sample = _within_sample_for_threshold(split_df, threshold=threshold, min_half_games=min_half_games)
    rows: list[dict[str, Any]] = []

    for metric_key, spec in METRIC_SPECS.items():
        metric = spec["pair_column"]
        x_col = f"{metric}_first"
        y_col = f"{metric}_second"
        control_col = "non_clutch_net_rating_first"

        if sample.empty:
            corr = None
            slope = None
            control_beta = None
            control_non_clutch = None
            control_r2 = None
            n_obs = 0
        else:
            x = pd.to_numeric(sample[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sample[y_col], errors="coerce").to_numpy(dtype=float)
            control = pd.to_numeric(sample[control_col], errors="coerce").to_numpy(dtype=float)
            weights = np.minimum(
                pd.to_numeric(sample["clutch_possessions_first"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(sample["clutch_possessions_second"], errors="coerce").to_numpy(dtype=float),
            )
            corr = _safe_corr(x, y)
            wls_simple = _weighted_regression(y=y, x_cols=[x], weights=weights)
            wls_control = _weighted_regression(y=y, x_cols=[x, control], weights=weights)
            slope = wls_simple["coefficients"][0] if wls_simple["coefficients"] else None
            control_beta = wls_control["coefficients"][0] if wls_control["coefficients"] else None
            control_non_clutch = wls_control["coefficients"][1] if len(wls_control["coefficients"]) > 1 else None
            control_r2 = wls_control["r2"]
            n_obs = int(len(sample))

        rows.append(
            {
                "analysis": "within_season",
                "metric_key": metric_key,
                "metric_label": spec["label"],
                "threshold": int(threshold),
                "n_obs": n_obs,
                "pearson_r": None if corr is None else _round(corr, 6),
                "spearman_brown": None if _spearman_brown(corr) is None else _round(_spearman_brown(corr), 6),
                "wls_slope": None if slope is None else _round(slope, 6),
                "control_metric_beta": None if control_beta is None else _round(control_beta, 6),
                "control_non_clutch_beta": None if control_non_clutch is None else _round(control_non_clutch, 6),
                "control_r2": None if control_r2 is None else _round(control_r2, 6),
            }
        )

    return rows


def analyze_year_to_year(yoy_df: pd.DataFrame, threshold: int) -> list[dict[str, Any]]:
    sample = _yoy_sample_for_threshold(yoy_df, threshold=threshold)
    rows: list[dict[str, Any]] = []

    for metric_key, spec in METRIC_SPECS.items():
        metric = spec["pair_column"]
        x_col = f"{metric}_t"
        y_col = f"{metric}_t1"
        control_col = "non_clutch_net_rating_t"

        if sample.empty:
            corr = None
            slope = None
            control_beta = None
            control_non_clutch = None
            control_r2 = None
            n_obs = 0
        else:
            x = pd.to_numeric(sample[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sample[y_col], errors="coerce").to_numpy(dtype=float)
            control = pd.to_numeric(sample[control_col], errors="coerce").to_numpy(dtype=float)
            weights = np.minimum(
                pd.to_numeric(sample["clutch_possessions_t"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(sample["clutch_possessions_t1"], errors="coerce").to_numpy(dtype=float),
            )
            corr = _safe_corr(x, y)
            wls_simple = _weighted_regression(y=y, x_cols=[x], weights=weights)
            wls_control = _weighted_regression(y=y, x_cols=[x, control], weights=weights)
            slope = wls_simple["coefficients"][0] if wls_simple["coefficients"] else None
            control_beta = wls_control["coefficients"][0] if wls_control["coefficients"] else None
            control_non_clutch = wls_control["coefficients"][1] if len(wls_control["coefficients"]) > 1 else None
            control_r2 = wls_control["r2"]
            n_obs = int(len(sample))

        rows.append(
            {
                "analysis": "year_to_year",
                "metric_key": metric_key,
                "metric_label": spec["label"],
                "threshold": int(threshold),
                "n_obs": n_obs,
                "pearson_r": None if corr is None else _round(corr, 6),
                "wls_slope": None if slope is None else _round(slope, 6),
                "control_metric_beta": None if control_beta is None else _round(control_beta, 6),
                "control_non_clutch_beta": None if control_non_clutch is None else _round(control_non_clutch, 6),
                "control_r2": None if control_r2 is None else _round(control_r2, 6),
            }
        )

    return rows


def _permutation_summary(observed: Optional[float], null_values: np.ndarray) -> dict[str, Any]:
    if observed is None or null_values.size == 0:
        return {
            "observed": None,
            "null_mean": None,
            "null_ci_low": None,
            "null_ci_high": None,
            "empirical_p_value": None,
        }
    abs_obs = abs(float(observed))
    abs_null = np.abs(null_values)
    return {
        "observed": _round(observed, 6),
        "null_mean": _round(float(np.mean(null_values)), 6),
        "null_ci_low": _round(float(np.percentile(null_values, 2.5)), 6),
        "null_ci_high": _round(float(np.percentile(null_values, 97.5)), 6),
        "empirical_p_value": _round(float((np.sum(abs_null >= abs_obs) + 1) / (null_values.size + 1)), 6),
    }


def permutation_null_within(
    split_df: pd.DataFrame,
    metric_key: str,
    min_half_games: int,
    min_half_poss: int,
    permutations: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    sample = _within_sample_for_threshold(
        split_df,
        threshold=min_half_poss,
        min_half_games=min_half_games,
    ).reset_index(drop=True)
    metric = METRIC_SPECS[metric_key]["pair_column"]
    if sample.empty:
        return _permutation_summary(None, np.array([])), np.array([])

    x = pd.to_numeric(sample[f"{metric}_first"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sample[f"{metric}_second"], errors="coerce").to_numpy(dtype=float)
    observed = _safe_corr(x, y)

    rng = np.random.default_rng(seed)
    null_values = np.empty(permutations, dtype=float)
    group_indices = [grp.index.to_numpy(dtype=int) for _, grp in sample.groupby("season")]
    x_perm = x.copy()

    for idx in range(permutations):
        x_perm[:] = x
        for group in group_indices:
            if group.size > 1:
                x_perm[group] = x_perm[group][rng.permutation(group.size)]
        null_values[idx] = _safe_corr(x_perm, y) or 0.0

    return _permutation_summary(observed, null_values), null_values


def permutation_null_yoy(
    yoy_df: pd.DataFrame,
    metric_key: str,
    min_yoy_poss: int,
    permutations: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    sample = _yoy_sample_for_threshold(yoy_df, threshold=min_yoy_poss).reset_index(drop=True)
    metric = METRIC_SPECS[metric_key]["pair_column"]
    if sample.empty:
        return _permutation_summary(None, np.array([])), np.array([])

    x = pd.to_numeric(sample[f"{metric}_t"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sample[f"{metric}_t1"], errors="coerce").to_numpy(dtype=float)
    observed = _safe_corr(x, y)

    rng = np.random.default_rng(seed)
    null_values = np.empty(permutations, dtype=float)
    group_indices = [grp.index.to_numpy(dtype=int) for _, grp in sample.groupby("season_pair")]
    x_perm = x.copy()

    for idx in range(permutations):
        x_perm[:] = x
        for group in group_indices:
            if group.size > 1:
                x_perm[group] = x_perm[group][rng.permutation(group.size)]
        null_values[idx] = _safe_corr(x_perm, y) or 0.0

    return _permutation_summary(observed, null_values), null_values


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _histogram_bins(values: np.ndarray, bins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    edges = np.linspace(vmin, vmax, num=bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    return edges, counts.astype(float)


def write_histogram_svg(
    values: Sequence[float],
    path: Path,
    title: str,
    x_label: str,
    observed: Optional[float] = None,
    width: int = 900,
    height: int = 520,
) -> None:
    vals = np.asarray([float(v) for v in values if pd.notna(v)], dtype=float)
    edges, counts = _histogram_bins(vals)
    left, right, top, bottom = 80, 40, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_count = max(float(np.max(counts)) if counts.size else 0.0, 1.0)
    x_min = float(edges[0])
    x_max = float(edges[-1])

    def x_pos(value: float) -> float:
        return left + ((value - x_min) / max(x_max - x_min, 1e-9)) * plot_w

    def y_pos(value: float) -> float:
        return top + plot_h - (value / max_count) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{left}" y="34" font-size="24" font-family="Helvetica, Arial, sans-serif" fill="#16202a">{_svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#4b5563" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#4b5563" stroke-width="2"/>',
    ]

    for idx, count in enumerate(counts):
        x0 = x_pos(float(edges[idx]))
        x1 = x_pos(float(edges[idx + 1]))
        y = y_pos(float(count))
        bar_w = max(1.0, x1 - x0 - 2.0)
        bar_h = top + plot_h - y
        parts.append(
            f'<rect x="{x0 + 1:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#2f6f9f" opacity="0.82"/>'
        )

    for frac, label in zip((0.0, 0.5, 1.0), (f"{x_min:.0f}", f"{(x_min + x_max) / 2:.0f}", f"{x_max:.0f}")):
        x_tick = left + frac * plot_w
        parts.append(f'<line x1="{x_tick:.2f}" y1="{top + plot_h}" x2="{x_tick:.2f}" y2="{top + plot_h + 8}" stroke="#4b5563"/>')
        parts.append(
            f'<text x="{x_tick:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">{_svg_escape(label)}</text>'
        )

    for frac in (0.0, 0.5, 1.0):
        y_tick_val = frac * max_count
        y_tick = y_pos(y_tick_val)
        parts.append(f'<line x1="{left - 8}" y1="{y_tick:.2f}" x2="{left}" y2="{y_tick:.2f}" stroke="#4b5563"/>')
        parts.append(
            f'<text x="{left - 14}" y="{y_tick + 5:.2f}" text-anchor="end" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">{int(round(y_tick_val))}</text>'
        )

    if observed is not None:
        obs_x = x_pos(float(observed))
        parts.append(
            f'<line x1="{obs_x:.2f}" y1="{top}" x2="{obs_x:.2f}" y2="{top + plot_h}" stroke="#c0392b" stroke-width="3" stroke-dasharray="8 6"/>'
        )
        parts.append(
            f'<text x="{obs_x + 8:.2f}" y="{top + 18}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#c0392b">Observed = {observed:.3f}</text>'
        )

    parts.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#16202a">{_svg_escape(x_label)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_scatter_svg(
    x: Sequence[float],
    y: Sequence[float],
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 900,
    height: int = 520,
) -> None:
    x_vals = np.asarray([float(v) for v in x], dtype=float)
    y_vals = np.asarray([float(v) for v in y], dtype=float)
    left, right, top, bottom = 90, 40, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    if x_vals.size == 0 or y_vals.size == 0:
        x_vals = np.array([0.0, 1.0])
        y_vals = np.array([0.0, 1.0])

    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    if math.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def x_pos(value: float) -> float:
        return left + ((value - x_min) / max(x_max - x_min, 1e-9)) * plot_w

    def y_pos(value: float) -> float:
        return top + plot_h - ((value - y_min) / max(y_max - y_min, 1e-9)) * plot_h

    corr = _safe_corr(x_vals, y_vals)
    fit = np.polyfit(x_vals, y_vals, 1) if x_vals.size >= 2 and not np.allclose(x_vals, x_vals[0]) else None

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{left}" y="34" font-size="24" font-family="Helvetica, Arial, sans-serif" fill="#16202a">{_svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#4b5563" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#4b5563" stroke-width="2"/>',
    ]

    for xv, yv in zip(x_vals, y_vals):
        parts.append(
            f'<circle cx="{x_pos(float(xv)):.2f}" cy="{y_pos(float(yv)):.2f}" r="5" fill="#2f6f9f" opacity="0.72"/>'
        )

    if fit is not None:
        x0, x1 = x_min, x_max
        y0, y1 = fit[0] * x0 + fit[1], fit[0] * x1 + fit[1]
        parts.append(
            f'<line x1="{x_pos(x0):.2f}" y1="{y_pos(float(y0)):.2f}" x2="{x_pos(x1):.2f}" y2="{y_pos(float(y1)):.2f}" stroke="#c0392b" stroke-width="3"/>'
        )

    for frac, label in zip((0.0, 0.5, 1.0), (x_min, (x_min + x_max) / 2, x_max)):
        x_tick = left + frac * plot_w
        parts.append(f'<line x1="{x_tick:.2f}" y1="{top + plot_h}" x2="{x_tick:.2f}" y2="{top + plot_h + 8}" stroke="#4b5563"/>')
        parts.append(
            f'<text x="{x_tick:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">{label:.1f}</text>'
        )

    for frac, label in zip((0.0, 0.5, 1.0), (y_min, (y_min + y_max) / 2, y_max)):
        y_tick = top + plot_h - frac * plot_h
        parts.append(f'<line x1="{left - 8}" y1="{y_tick:.2f}" x2="{left}" y2="{y_tick:.2f}" stroke="#4b5563"/>')
        parts.append(
            f'<text x="{left - 14}" y="{y_tick + 5:.2f}" text-anchor="end" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">{label:.1f}</text>'
        )

    parts.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#16202a">{_svg_escape(x_label)}</text>'
    )
    parts.append(
        f'<text transform="translate(28 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#16202a">{_svg_escape(y_label)}</text>'
    )
    if corr is not None:
        parts.append(
            f'<text x="{left + plot_w - 8}" y="{top + 20}" text-anchor="end" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">r = {corr:.3f} | n = {len(x_vals)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_multi_null_histogram_svg(
    panels: Sequence[dict[str, Any]],
    path: Path,
    width: int = 1000,
    height: int = 760,
) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        '<text x="40" y="34" font-size="24" font-family="Helvetica, Arial, sans-serif" fill="#16202a">Permutation null benchmarks</text>',
    ]

    panel_w = (width - 100) / 2
    panel_h = (height - 120) / 2
    for idx, panel in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x0 = 40 + col * (panel_w + 20)
        y0 = 60 + row * (panel_h + 20)
        values = np.asarray(panel["null_values"], dtype=float)
        edges, counts = _histogram_bins(values)
        max_count = max(float(np.max(counts)) if counts.size else 0.0, 1.0)
        inner_left, inner_right, inner_top, inner_bottom = 55, 20, 36, 42
        plot_w = panel_w - inner_left - inner_right
        plot_h = panel_h - inner_top - inner_bottom
        x_min = float(edges[0])
        x_max = float(edges[-1])

        def x_pos(value: float) -> float:
            return x0 + inner_left + ((value - x_min) / max(x_max - x_min, 1e-9)) * plot_w

        def y_pos(value: float) -> float:
            return y0 + inner_top + plot_h - (value / max_count) * plot_h

        parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" fill="#fffdf8" stroke="#d6d3d1"/>')
        parts.append(
            f'<text x="{x0 + inner_left:.2f}" y="{y0 + 22:.2f}" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#16202a">{_svg_escape(panel["title"])}</text>'
        )
        parts.append(
            f'<line x1="{x0 + inner_left:.2f}" y1="{y0 + inner_top + plot_h:.2f}" x2="{x0 + inner_left + plot_w:.2f}" y2="{y0 + inner_top + plot_h:.2f}" stroke="#4b5563" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{x0 + inner_left:.2f}" y1="{y0 + inner_top:.2f}" x2="{x0 + inner_left:.2f}" y2="{y0 + inner_top + plot_h:.2f}" stroke="#4b5563" stroke-width="2"/>'
        )

        for bin_idx, count in enumerate(counts):
            bx0 = x_pos(float(edges[bin_idx]))
            bx1 = x_pos(float(edges[bin_idx + 1]))
            by = y_pos(float(count))
            parts.append(
                f'<rect x="{bx0 + 1:.2f}" y="{by:.2f}" width="{max(1.0, bx1 - bx0 - 2):.2f}" height="{(y0 + inner_top + plot_h - by):.2f}" fill="#2f6f9f" opacity="0.82"/>'
            )

        observed = panel.get("observed")
        if observed is not None:
            ox = x_pos(float(observed))
            parts.append(
                f'<line x1="{ox:.2f}" y1="{y0 + inner_top:.2f}" x2="{ox:.2f}" y2="{y0 + inner_top + plot_h:.2f}" stroke="#c0392b" stroke-width="3" stroke-dasharray="8 6"/>'
            )
            parts.append(
                f'<text x="{ox + 6:.2f}" y="{y0 + inner_top + 14:.2f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#c0392b">Obs {float(observed):.3f}</text>'
            )

        parts.append(
            f'<text x="{x0 + inner_left + plot_w / 2:.2f}" y="{y0 + panel_h - 10:.2f}" text-anchor="middle" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#374151">Pearson r under null</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(
    config: AnalysisConfig,
    team_season_df: pd.DataFrame,
    split_df: pd.DataFrame,
    yoy_df: pd.DataFrame,
    summary_metrics: dict[str, Any],
) -> str:
    within_default = {
        row["metric_key"]: row
        for row in summary_metrics["within_season"]["default_threshold_results"]
    }
    yoy_default = {
        row["metric_key"]: row
        for row in summary_metrics["year_to_year"]["default_threshold_results"]
    }
    perm = summary_metrics["permutation_nulls"]
    desc = summary_metrics["descriptive"]
    integrity = summary_metrics["data_integrity"]

    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"

    lines = [
        "# Clutch Net Rating Persistence Study",
        "",
        "## Setup",
        "",
        f"- Seasons analyzed: {summary_metrics['dataset']['first_season']} to {summary_metrics['dataset']['last_season']}",
        f"- Team-seasons: {len(team_season_df)}",
        f"- Split-half samples: {len(split_df)}",
        f"- Year-to-year franchise pairs: {len(yoy_df)}",
        f"- Included playoffs: {'yes' if config.include_playoffs else 'no'}",
        f"- Default within-season filter: >= {config.min_half_games} clutch games and >= {config.min_half_poss} clutch possessions in each half",
        f"- Default year-to-year filter: >= {config.min_yoy_poss} clutch possessions in each season",
        "",
        "## Descriptive Sample Notes",
        "",
        f"- Median team-season clutch games: {fmt(desc['clutch_games']['median'])}",
        f"- Median team-season clutch possessions: {fmt(desc['clutch_possessions']['median'])}",
        f"- Median clutch time share: {fmt(desc['clutch_time_pct']['median'])}%",
        f"- Median raw clutch net-rating spread by season (std dev across teams): {fmt(desc['seasonal_raw_spread']['median'])}",
        f"- Median clutch-minus-non-clutch spread by season (std dev across teams): {fmt(desc['seasonal_residual_spread']['median'])}",
        "",
        "## Headline Persistence Results",
        "",
        "| Analysis | Metric | n | Pearson r | WLS slope | Control beta | Control non-clutch beta | Spearman-Brown |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Within-season | Raw clutch net | {within_default['raw_clutch_net']['n_obs']} | {fmt(within_default['raw_clutch_net']['pearson_r'])} | {fmt(within_default['raw_clutch_net']['wls_slope'])} | {fmt(within_default['raw_clutch_net']['control_metric_beta'])} | {fmt(within_default['raw_clutch_net']['control_non_clutch_beta'])} | {fmt(within_default['raw_clutch_net']['spearman_brown'])} |",
        f"| Within-season | Clutch minus non-clutch | {within_default['clutch_minus_non_clutch']['n_obs']} | {fmt(within_default['clutch_minus_non_clutch']['pearson_r'])} | {fmt(within_default['clutch_minus_non_clutch']['wls_slope'])} | {fmt(within_default['clutch_minus_non_clutch']['control_metric_beta'])} | {fmt(within_default['clutch_minus_non_clutch']['control_non_clutch_beta'])} | {fmt(within_default['clutch_minus_non_clutch']['spearman_brown'])} |",
        f"| Year-to-year | Raw clutch net | {yoy_default['raw_clutch_net']['n_obs']} | {fmt(yoy_default['raw_clutch_net']['pearson_r'])} | {fmt(yoy_default['raw_clutch_net']['wls_slope'])} | {fmt(yoy_default['raw_clutch_net']['control_metric_beta'])} | {fmt(yoy_default['raw_clutch_net']['control_non_clutch_beta'])} | n/a |",
        f"| Year-to-year | Clutch minus non-clutch | {yoy_default['clutch_minus_non_clutch']['n_obs']} | {fmt(yoy_default['clutch_minus_non_clutch']['pearson_r'])} | {fmt(yoy_default['clutch_minus_non_clutch']['wls_slope'])} | {fmt(yoy_default['clutch_minus_non_clutch']['control_metric_beta'])} | {fmt(yoy_default['clutch_minus_non_clutch']['control_non_clutch_beta'])} | n/a |",
        "",
        "## Permutation Null Benchmarks",
        "",
        "| Analysis | Metric | Observed r | Null mean | Null 95% interval | Empirical p-value |",
        "| --- | --- | ---: | ---: | --- | ---: |",
        f"| Within-season | Raw clutch net | {fmt(perm['within_season']['raw_clutch_net']['observed'])} | {fmt(perm['within_season']['raw_clutch_net']['null_mean'])} | [{fmt(perm['within_season']['raw_clutch_net']['null_ci_low'])}, {fmt(perm['within_season']['raw_clutch_net']['null_ci_high'])}] | {fmt(perm['within_season']['raw_clutch_net']['empirical_p_value'])} |",
        f"| Within-season | Clutch minus non-clutch | {fmt(perm['within_season']['clutch_minus_non_clutch']['observed'])} | {fmt(perm['within_season']['clutch_minus_non_clutch']['null_mean'])} | [{fmt(perm['within_season']['clutch_minus_non_clutch']['null_ci_low'])}, {fmt(perm['within_season']['clutch_minus_non_clutch']['null_ci_high'])}] | {fmt(perm['within_season']['clutch_minus_non_clutch']['empirical_p_value'])} |",
        f"| Year-to-year | Raw clutch net | {fmt(perm['year_to_year']['raw_clutch_net']['observed'])} | {fmt(perm['year_to_year']['raw_clutch_net']['null_mean'])} | [{fmt(perm['year_to_year']['raw_clutch_net']['null_ci_low'])}, {fmt(perm['year_to_year']['raw_clutch_net']['null_ci_high'])}] | {fmt(perm['year_to_year']['raw_clutch_net']['empirical_p_value'])} |",
        f"| Year-to-year | Clutch minus non-clutch | {fmt(perm['year_to_year']['clutch_minus_non_clutch']['observed'])} | {fmt(perm['year_to_year']['clutch_minus_non_clutch']['null_mean'])} | [{fmt(perm['year_to_year']['clutch_minus_non_clutch']['null_ci_low'])}, {fmt(perm['year_to_year']['clutch_minus_non_clutch']['null_ci_high'])}] | {fmt(perm['year_to_year']['clutch_minus_non_clutch']['empirical_p_value'])} |",
        "",
        "## Data Integrity Checks",
        "",
        f"- Seasons loaded successfully: {integrity['seasons_loaded']} / {integrity['seasons_expected']}",
        f"- Max clutch-minus-all possession overage: {fmt(integrity['max_possession_overage'])}",
        f"- Max clutch-minus-all minute overage: {fmt(integrity['max_minute_overage'])}",
        f"- Max complement reconstruction error (possessions): {fmt(integrity['max_complement_poss_error'])}",
        f"- Max complement reconstruction error (minutes): {fmt(integrity['max_complement_minute_error'])}",
        "",
        "## Sensitivity Summary",
        "",
        "Headline correlations across the threshold grid are recorded in `summary_metrics.json` under `within_season.threshold_grid_results` and `year_to_year.threshold_grid_results` for thresholds 75, 100, 125, and 150 possessions.",
        "",
        "## Plain-English Conclusion",
        "",
        f"Raw clutch performance shows {'some' if (within_default['raw_clutch_net']['pearson_r'] or 0) > 0.15 else 'little'} within-season persistence (r = {fmt(within_default['raw_clutch_net']['pearson_r'])}) and {'some' if (yoy_default['raw_clutch_net']['pearson_r'] or 0) > 0.15 else 'little'} year-to-year carryover (r = {fmt(yoy_default['raw_clutch_net']['pearson_r'])}).",
        "",
        f"After controlling for ordinary team strength with clutch-minus-non-clutch net rating, the persistence is {'still present' if (within_default['clutch_minus_non_clutch']['pearson_r'] or 0) > 0.10 else 'much weaker'} within a season (r = {fmt(within_default['clutch_minus_non_clutch']['pearson_r'])}) and {'still present' if (yoy_default['clutch_minus_non_clutch']['pearson_r'] or 0) > 0.10 else 'close to zero'} from one season to the next (r = {fmt(yoy_default['clutch_minus_non_clutch']['pearson_r'])}).",
        "",
        f"Relative to the no-persistence permutation null, the observed residual clutch signal is {'materially above random noise' if (perm['within_season']['clutch_minus_non_clutch']['empirical_p_value'] or 1.0) < 0.05 or (perm['year_to_year']['clutch_minus_non_clutch']['empirical_p_value'] or 1.0) < 0.05 else 'not much different from what random ordering would produce'}, so the better reading is that raw clutch results contain some real signal but a large share of apparent clutch dominance is explained by overall team quality and sampling noise.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows_by_season: list[pd.DataFrame] = []
    clutch_rows_by_season: list[pd.DataFrame] = []
    seasons_loaded = 0

    seasons = get_available_seasons()
    for season in seasons:
        all_rows, clutch_rows = _load_rows_for_season(season=season, config=config)
        all_rows_by_season.append(all_rows)
        clutch_rows_by_season.append(clutch_rows)
        seasons_loaded += 1

    all_rows = pd.concat(all_rows_by_season, ignore_index=True)
    clutch_rows = pd.concat(clutch_rows_by_season, ignore_index=True)
    team_season_df = build_team_season_metrics(all_rows=all_rows, clutch_rows=clutch_rows)
    split_df = build_split_half_table(all_rows=all_rows, clutch_rows=clutch_rows)
    yoy_df = build_year_to_year_pairs(team_season_df=team_season_df)

    team_season_path = config.output_dir / "team_season_clutch_metrics.csv"
    split_path = config.output_dir / "within_season_split_half.csv"
    yoy_path = config.output_dir / "year_to_year_pairs.csv"
    team_season_df.to_csv(team_season_path, index=False)
    split_df.to_csv(split_path, index=False)
    yoy_df.to_csv(yoy_path, index=False)

    within_results: list[dict[str, Any]] = []
    yoy_results: list[dict[str, Any]] = []
    for threshold in config.thresholds:
        within_results.extend(analyze_within_season(split_df, threshold=threshold, min_half_games=config.min_half_games))
        yoy_results.extend(analyze_year_to_year(yoy_df, threshold=threshold))

    within_default_results = analyze_within_season(
        split_df,
        threshold=config.min_half_poss,
        min_half_games=config.min_half_games,
    )
    yoy_default_results = analyze_year_to_year(yoy_df, threshold=config.min_yoy_poss)

    perm_within_raw, null_within_raw = permutation_null_within(
        split_df=split_df,
        metric_key="raw_clutch_net",
        min_half_games=config.min_half_games,
        min_half_poss=config.min_half_poss,
        permutations=config.permutations,
        seed=config.seed,
    )
    perm_within_resid, null_within_resid = permutation_null_within(
        split_df=split_df,
        metric_key="clutch_minus_non_clutch",
        min_half_games=config.min_half_games,
        min_half_poss=config.min_half_poss,
        permutations=config.permutations,
        seed=config.seed + 1,
    )
    perm_yoy_raw, null_yoy_raw = permutation_null_yoy(
        yoy_df=yoy_df,
        metric_key="raw_clutch_net",
        min_yoy_poss=config.min_yoy_poss,
        permutations=config.permutations,
        seed=config.seed + 2,
    )
    perm_yoy_resid, null_yoy_resid = permutation_null_yoy(
        yoy_df=yoy_df,
        metric_key="clutch_minus_non_clutch",
        min_yoy_poss=config.min_yoy_poss,
        permutations=config.permutations,
        seed=config.seed + 3,
    )

    data_integrity = {
        "seasons_expected": len(seasons),
        "seasons_loaded": seasons_loaded,
        "max_possession_overage": _round(
            max(
                0.0,
                float(
                    (pd.to_numeric(team_season_df["clutch_possessions"], errors="coerce")
                     - pd.to_numeric(team_season_df["all_possessions"], errors="coerce")).max()
                ),
            )
        ),
        "max_minute_overage": _round(
            max(
                0.0,
                float(
                    (pd.to_numeric(team_season_df["clutch_minutes"], errors="coerce")
                     - pd.to_numeric(team_season_df["all_minutes"], errors="coerce")).max()
                ),
            )
        ),
        "max_complement_poss_error": _round(
            float(
                np.max(
                    np.abs(
                        pd.to_numeric(team_season_df["all_possessions"], errors="coerce")
                        - (
                            pd.to_numeric(team_season_df["clutch_possessions"], errors="coerce")
                            + pd.to_numeric(team_season_df["non_clutch_possessions"], errors="coerce")
                        )
                    )
                )
            )
        ),
        "max_complement_minute_error": _round(
            float(
                np.max(
                    np.abs(
                        pd.to_numeric(team_season_df["all_minutes"], errors="coerce")
                        - (
                            pd.to_numeric(team_season_df["clutch_minutes"], errors="coerce")
                            + pd.to_numeric(team_season_df["non_clutch_minutes"], errors="coerce")
                        )
                    )
                )
            )
        ),
    }

    seasonal_spread = (
        team_season_df.groupby("season", as_index=False)
        .agg(
            raw_std=("clutch_net_rating", "std"),
            residual_std=("clutch_minus_non_clutch_net", "std"),
        )
        .fillna(0.0)
    )

    summary_metrics: dict[str, Any] = {
        "config": {
            "repo_dir": str(config.repo_dir),
            "output_dir": str(config.output_dir),
            "include_playoffs": config.include_playoffs,
            "min_half_games": config.min_half_games,
            "min_half_poss": config.min_half_poss,
            "min_yoy_poss": config.min_yoy_poss,
            "permutations": config.permutations,
            "seed": config.seed,
            "thresholds": list(config.thresholds),
        },
        "dataset": {
            "first_season": str(team_season_df["season"].min()) if not team_season_df.empty else None,
            "last_season": str(team_season_df["season"].max()) if not team_season_df.empty else None,
            "team_seasons": int(len(team_season_df)),
            "split_half_rows": int(len(split_df)),
            "year_to_year_pairs": int(len(yoy_df)),
        },
        "data_integrity": data_integrity,
        "descriptive": {
            "clutch_games": _describe(team_season_df["clutch_games"].tolist()),
            "clutch_minutes": _describe(team_season_df["clutch_minutes"].tolist()),
            "clutch_possessions": _describe(team_season_df["clutch_possessions"].tolist()),
            "clutch_time_pct": _describe(team_season_df["clutch_time_pct"].tolist()),
            "seasonal_raw_spread": _describe(seasonal_spread["raw_std"].tolist()),
            "seasonal_residual_spread": _describe(seasonal_spread["residual_std"].tolist()),
        },
        "within_season": {
            "default_threshold": config.min_half_poss,
            "default_threshold_results": within_default_results,
            "threshold_grid_results": within_results,
        },
        "year_to_year": {
            "default_threshold": config.min_yoy_poss,
            "default_threshold_results": yoy_default_results,
            "threshold_grid_results": yoy_results,
        },
        "permutation_nulls": {
            "within_season": {
                "raw_clutch_net": perm_within_raw,
                "clutch_minus_non_clutch": perm_within_resid,
            },
            "year_to_year": {
                "raw_clutch_net": perm_yoy_raw,
                "clutch_minus_non_clutch": perm_yoy_resid,
            },
        },
    }

    summary_path = config.output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(_to_serializable(summary_metrics), indent=2), encoding="utf-8")

    write_histogram_svg(
        values=team_season_df["clutch_possessions"].tolist(),
        path=config.output_dir / "clutch_sample_distribution.svg",
        title="Team-season clutch possession distribution",
        x_label="Clutch possessions per team-season",
    )

    split_default = _within_sample_for_threshold(split_df, threshold=config.min_half_poss, min_half_games=config.min_half_games)
    yoy_default = _yoy_sample_for_threshold(yoy_df, threshold=config.min_yoy_poss)

    write_scatter_svg(
        x=split_default["clutch_net_rating_first"].tolist(),
        y=split_default["clutch_net_rating_second"].tolist(),
        path=config.output_dir / "raw_split_half_scatter.svg",
        title="Within-season split-half persistence: raw clutch net rating",
        x_label="First-half clutch net rating",
        y_label="Second-half clutch net rating",
    )
    write_scatter_svg(
        x=split_default["clutch_minus_non_clutch_net_first"].tolist(),
        y=split_default["clutch_minus_non_clutch_net_second"].tolist(),
        path=config.output_dir / "residual_split_half_scatter.svg",
        title="Within-season split-half persistence: clutch minus non-clutch",
        x_label="First-half clutch minus non-clutch net rating",
        y_label="Second-half clutch minus non-clutch net rating",
    )
    write_scatter_svg(
        x=yoy_default["clutch_net_rating_t"].tolist(),
        y=yoy_default["clutch_net_rating_t1"].tolist(),
        path=config.output_dir / "raw_year_to_year_scatter.svg",
        title="Year-to-year persistence: raw clutch net rating",
        x_label="Season t clutch net rating",
        y_label="Season t+1 clutch net rating",
    )
    write_scatter_svg(
        x=yoy_default["clutch_minus_non_clutch_net_t"].tolist(),
        y=yoy_default["clutch_minus_non_clutch_net_t1"].tolist(),
        path=config.output_dir / "residual_year_to_year_scatter.svg",
        title="Year-to-year persistence: clutch minus non-clutch",
        x_label="Season t clutch minus non-clutch net rating",
        y_label="Season t+1 clutch minus non-clutch net rating",
    )
    write_multi_null_histogram_svg(
        panels=[
            {
                "title": "Within-season raw clutch net",
                "null_values": null_within_raw,
                "observed": perm_within_raw["observed"],
            },
            {
                "title": "Within-season clutch minus non-clutch",
                "null_values": null_within_resid,
                "observed": perm_within_resid["observed"],
            },
            {
                "title": "Year-to-year raw clutch net",
                "null_values": null_yoy_raw,
                "observed": perm_yoy_raw["observed"],
            },
            {
                "title": "Year-to-year clutch minus non-clutch",
                "null_values": null_yoy_resid,
                "observed": perm_yoy_resid["observed"],
            },
        ],
        path=config.output_dir / "permutation_null_histograms.svg",
    )

    report_text = build_report(
        config=config,
        team_season_df=team_season_df,
        split_df=split_df,
        yoy_df=yoy_df,
        summary_metrics=summary_metrics,
    )
    report_path = config.output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "team_season_path": team_season_path,
        "split_path": split_path,
        "yoy_path": yoy_path,
        "summary_path": summary_path,
        "report_path": report_path,
    }


def main() -> None:
    config = parse_args()
    outputs = run_analysis(config)
    print("Wrote clutch persistence outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
