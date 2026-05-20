from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import NBA_DATA_REPO_DIR, get_available_seasons


PLAYER_SHOTS_ROOT = Path(NBA_DATA_REPO_DIR) / "player_shots"
PLAYER_SHOTS_COLUMNS = [
    "season",
    "pbp_phase",
    "game_id",
    "game_date",
    "game_type",
    "team_id",
    "team",
    "opponent_id",
    "opponent",
    "home_road",
    "player_id",
    "player_name",
    "shot_type",
    "result",
    "action_number",
    "action_id",
    "period",
    "clock",
    "description",
]
PLAYER_SHOTS_GAME_TYPES = {"regular_season", "playoffs", "play_in", "nba_cup_semi", "nba_cup_final"}
PLAYER_SHOT_PLAYER_COLUMNS = ["player_id", "player_name", "team", "game_type", "shot_type", "result"]
PLAYER_SHOT_STREAKINESS_COLUMNS = [
    "season",
    "game_type",
    "team",
    "player_id",
    "player_name",
    "shot_type",
    "result",
    "game_date",
    "game_id",
    "action_number",
    "action_id",
]
PLAYER_SHOT_TYPES = {"fta", "2ptfga", "3ptfga"}
PLAYER_SHOT_CLASSIFICATIONS = {"consistent", "streaky", "volatile", "alternating", "ordinary"}


def _season_sort_key(season: str) -> int:
    try:
        return int(str(season).split("-")[0])
    except Exception:
        return 0


def _iter_seasons(start_season: Optional[str], end_season: Optional[str]) -> list[str]:
    seasons = sorted(get_available_seasons(), key=_season_sort_key)
    if start_season:
        seasons = [season for season in seasons if _season_sort_key(season) >= _season_sort_key(start_season)]
    if end_season:
        seasons = [season for season in seasons if _season_sort_key(season) <= _season_sort_key(end_season)]
    return seasons


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PLAYER_SHOTS_COLUMNS)


def list_player_shot_players(season: str, *, game_type: Optional[str] = None) -> list[dict[str, Any]]:
    path = PLAYER_SHOTS_ROOT / f"player_shots_{season}.parquet"
    if not path.exists():
        return []

    filters = []
    if game_type:
        filters.append(("game_type", "=", str(game_type).strip().lower()))

    try:
        frame = pd.read_parquet(path, columns=PLAYER_SHOT_PLAYER_COLUMNS, filters=filters or None)
    except TypeError:
        frame = pd.read_parquet(path, columns=PLAYER_SHOT_PLAYER_COLUMNS)

    if frame.empty:
        return []

    if game_type:
        frame = frame[frame["game_type"].astype(str).str.lower().eq(str(game_type).strip().lower())]

    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce").fillna(0).astype(int)
    frame = frame[frame["player_id"] > 0].copy()
    if frame.empty:
        return []

    summary = (
        frame.assign(
            is_3pa=frame["shot_type"].astype(str).eq("3ptfga"),
            is_2pa=frame["shot_type"].astype(str).eq("2ptfga"),
            is_fta=frame["shot_type"].astype(str).eq("fta"),
            is_make=frame["result"].astype(str).eq("make"),
        )
        .groupby(["player_id", "player_name"], dropna=False)
        .agg(
            attempts=("shot_type", "size"),
            makes=("is_make", "sum"),
            three_pa=("is_3pa", "sum"),
            two_pa=("is_2pa", "sum"),
            fta=("is_fta", "sum"),
            teams=("team", lambda values: sorted({str(v) for v in values if str(v)})),
        )
        .reset_index()
    )
    summary["player_name"] = summary["player_name"].astype(str).replace({"": "Unknown"})
    summary = summary.sort_values(["player_name", "attempts"], ascending=[True, False], kind="stable")

    rows: list[dict[str, Any]] = []
    for row in summary.to_dict(orient="records"):
        rows.append(
            {
                "player_id": int(row["player_id"]),
                "player_name": str(row["player_name"]),
                "teams": list(row["teams"]),
                "attempts": int(row["attempts"]),
                "makes": int(row["makes"]),
                "three_pa": int(row["three_pa"]),
                "two_pa": int(row["two_pa"]),
                "fta": int(row["fta"]),
            }
        )
    return rows


def load_player_shots_frame(
    *,
    player_id: Optional[int] = None,
    player_name: Optional[str] = None,
    start_season: Optional[str] = None,
    end_season: Optional[str] = None,
    game_type: Optional[str] = None,
    shot_type: Optional[str] = None,
    result: Optional[str] = None,
    team: Optional[str] = None,
    opponent: Optional[str] = None,
    limit: int = 5000,
    offset: int = 0,
) -> pd.DataFrame:
    if player_id is None and not player_name:
        raise ValueError("player_id or player_name is required")

    frames: list[pd.DataFrame] = []
    seasons = _iter_seasons(start_season, end_season)
    columns = PLAYER_SHOTS_COLUMNS

    for season in seasons:
        path = PLAYER_SHOTS_ROOT / f"player_shots_{season}.parquet"
        if not path.exists():
            continue

        filters = []
        if player_id is not None:
            filters.append(("player_id", "=", int(player_id)))
        if game_type:
            filters.append(("game_type", "=", str(game_type).strip().lower()))

        try:
            frame = pd.read_parquet(path, columns=columns, filters=filters or None)
        except TypeError:
            frame = pd.read_parquet(path, columns=columns)
        if frame.empty:
            continue

        if player_id is not None:
            frame = frame[pd.to_numeric(frame["player_id"], errors="coerce").fillna(0).astype(int).eq(int(player_id))]
        if player_name:
            needle = str(player_name).strip().lower()
            frame = frame[frame["player_name"].astype(str).str.lower().str.contains(needle, regex=False, na=False)]
        if game_type:
            frame = frame[frame["game_type"].astype(str).str.lower().eq(str(game_type).strip().lower())]
        if shot_type:
            frame = frame[frame["shot_type"].astype(str).str.lower().eq(str(shot_type).strip().lower())]
        if result:
            frame = frame[frame["result"].astype(str).str.lower().eq(str(result).strip().lower())]
        if team:
            frame = frame[frame["team"].astype(str).str.upper().eq(str(team).strip().upper())]
        if opponent:
            frame = frame[frame["opponent"].astype(str).str.upper().eq(str(opponent).strip().upper())]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return _empty_frame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["game_date", "game_id", "action_number", "action_id"], kind="stable").reset_index(drop=True)
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 5000), 50000))
    return combined.iloc[offset : offset + limit].copy()


def build_player_shots_payload(
    *,
    player_id: Optional[int] = None,
    player_name: Optional[str] = None,
    start_season: Optional[str] = None,
    end_season: Optional[str] = None,
    game_type: Optional[str] = None,
    shot_type: Optional[str] = None,
    result: Optional[str] = None,
    team: Optional[str] = None,
    opponent: Optional[str] = None,
    limit: int = 5000,
    offset: int = 0,
) -> dict[str, Any]:
    frame = load_player_shots_frame(
        player_id=player_id,
        player_name=player_name,
        start_season=start_season,
        end_season=end_season,
        game_type=game_type,
        shot_type=shot_type,
        result=result,
        team=team,
        opponent=opponent,
        limit=limit,
        offset=offset,
    )
    rows = frame.where(pd.notnull(frame), None).to_dict(orient="records")
    return {
        "player_id": int(player_id) if player_id is not None else None,
        "player_name": player_name,
        "start_season": start_season,
        "end_season": end_season,
        "game_type": game_type,
        "shot_type": shot_type,
        "result": result,
        "team": team,
        "opponent": opponent,
        "limit": max(1, min(int(limit or 5000), 50000)),
        "offset": max(0, int(offset or 0)),
        "row_count": len(rows),
        "rows": rows,
    }


def _percentile_at_or_below(observed: float, simulated: np.ndarray) -> float:
    if simulated.size == 0 or not np.isfinite(observed):
        return 50.0
    below = float(np.count_nonzero(simulated < observed))
    tied = float(np.count_nonzero(simulated == observed))
    return max(0.0, min(100.0, 100.0 * (below + 0.5 * tied) / float(simulated.size)))


def _percentile_at_or_above(observed: float, simulated: np.ndarray) -> float:
    if simulated.size == 0 or not np.isfinite(observed):
        return 50.0
    above = float(np.count_nonzero(simulated > observed))
    tied = float(np.count_nonzero(simulated == observed))
    return max(0.0, min(100.0, 100.0 * (above + 0.5 * tied) / float(simulated.size)))


def _runs_count(sequence: np.ndarray) -> int:
    if sequence.size == 0:
        return 0
    return int(1 + np.count_nonzero(sequence[1:] != sequence[:-1]))


def _transition_effect(sequence: np.ndarray) -> float:
    if sequence.size < 2:
        return 0.0
    prev = sequence[:-1]
    current = sequence[1:]
    after_make = current[prev == 1]
    after_miss = current[prev == 0]
    make_after_make = float(after_make.mean()) if after_make.size else 0.0
    make_after_miss = float(after_miss.mean()) if after_miss.size else 0.0
    return make_after_make - make_after_miss


def _longest_run(sequence: np.ndarray, value: int) -> int:
    longest = 0
    current = 0
    for item in sequence:
        if int(item) == value:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def _window_variance(sequence: np.ndarray, window_size: int) -> Optional[float]:
    if window_size <= 1 or sequence.size < 2 * window_size:
        return None
    window_count = int(sequence.size // window_size)
    if window_count < 2:
        return None
    trimmed = sequence[: window_count * window_size]
    rates = trimmed.reshape(window_count, window_size).mean(axis=1)
    return float(np.var(rates))


def _default_window_size(shot_type: str) -> int:
    return 10 if shot_type == "fta" else 25


def _stable_seed(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _average_defined(*values: Optional[float]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _classify_shooter(
    *,
    streakiness_score: Optional[float],
    consistency_score: Optional[float],
    window_variance_percentile: Optional[float],
    runs_cluster_percentile: float,
    runs_alternation_percentile: float,
    transition_percentile: float,
) -> str:
    if runs_alternation_percentile >= 90.0 or transition_percentile <= 10.0:
        return "Alternating"
    if streakiness_score is not None and streakiness_score >= 90.0:
        return "Streaky"
    if consistency_score is not None and consistency_score >= 90.0 and (streakiness_score is None or streakiness_score < 75.0):
        return "Consistent"
    if (
        window_variance_percentile is not None
        and window_variance_percentile >= 90.0
        and runs_cluster_percentile < 75.0
        and transition_percentile < 75.0
    ):
        return "Volatile"
    return "Ordinary"


def analyze_shot_streakiness_sequence(
    sequence: list[int] | np.ndarray,
    *,
    season: str,
    player_id: int,
    shot_type: str,
    game_type: Optional[str] = None,
    simulations: int = 1000,
    window_size: Optional[int] = None,
) -> dict[str, Any]:
    makes = np.asarray(sequence, dtype=np.int8)
    attempts = int(makes.size)
    made = int(makes.sum())
    shot_type_norm = str(shot_type).strip().lower()
    resolved_window_size = int(window_size or _default_window_size(shot_type_norm))
    simulations = max(100, min(int(simulations or 1000), 5000))

    observed_runs = _runs_count(makes)
    observed_transition = _transition_effect(makes)
    observed_window_variance = _window_variance(makes, resolved_window_size)
    observed_longest_make = _longest_run(makes, 1)
    observed_longest_miss = _longest_run(makes, 0)

    # With zero makes or zero misses, every make-preserving shuffle is identical
    # for the metrics that matter here, so neutral percentiles avoid fake signals.
    if attempts < 2 or made == 0 or made == attempts:
        window_variance_percentile = 50.0 if observed_window_variance is not None else None
        consistency_score = 100.0 - window_variance_percentile if window_variance_percentile is not None else None
        streakiness_score = _average_defined(50.0, 50.0, window_variance_percentile)
        classification = _classify_shooter(
            streakiness_score=streakiness_score,
            consistency_score=consistency_score,
            window_variance_percentile=window_variance_percentile,
            runs_cluster_percentile=50.0,
            runs_alternation_percentile=50.0,
            transition_percentile=50.0,
        )
        return {
            "attempts": attempts,
            "makes": made,
            "make_pct": float(made / attempts) if attempts else 0.0,
            "classification": classification,
            "streakiness_score": streakiness_score,
            "consistency_score": consistency_score,
            "runs": observed_runs,
            "runs_cluster_percentile": 50.0,
            "runs_alternation_percentile": 50.0,
            "transition_effect": observed_transition,
            "transition_percentile": 50.0,
            "window_size": resolved_window_size if observed_window_variance is not None else None,
            "window_variance": observed_window_variance,
            "window_variance_percentile": window_variance_percentile,
            "longest_make_run": observed_longest_make,
            "longest_make_run_percentile": 50.0,
            "longest_miss_run": observed_longest_miss,
            "longest_miss_run_percentile": 50.0,
        }

    rng = np.random.default_rng(_stable_seed(season, player_id, shot_type_norm, game_type or "", attempts, made, simulations))
    sim_runs = np.empty(simulations, dtype=np.float64)
    sim_transitions = np.empty(simulations, dtype=np.float64)
    sim_longest_makes = np.empty(simulations, dtype=np.float64)
    sim_longest_misses = np.empty(simulations, dtype=np.float64)
    sim_window_variances = np.empty(simulations, dtype=np.float64) if observed_window_variance is not None else None

    for idx in range(simulations):
        shuffled = rng.permutation(makes)
        sim_runs[idx] = _runs_count(shuffled)
        sim_transitions[idx] = _transition_effect(shuffled)
        sim_longest_makes[idx] = _longest_run(shuffled, 1)
        sim_longest_misses[idx] = _longest_run(shuffled, 0)
        if sim_window_variances is not None:
            value = _window_variance(shuffled, resolved_window_size)
            sim_window_variances[idx] = 0.0 if value is None else value

    runs_cluster_percentile = _percentile_at_or_above(float(observed_runs), sim_runs)
    runs_alternation_percentile = _percentile_at_or_below(float(observed_runs), sim_runs)
    transition_percentile = _percentile_at_or_below(float(observed_transition), sim_transitions)
    longest_make_percentile = _percentile_at_or_below(float(observed_longest_make), sim_longest_makes)
    longest_miss_percentile = _percentile_at_or_below(float(observed_longest_miss), sim_longest_misses)

    window_variance_percentile = None
    if sim_window_variances is not None and observed_window_variance is not None:
        window_variance_percentile = _percentile_at_or_below(float(observed_window_variance), sim_window_variances)

    consistency_score = 100.0 - window_variance_percentile if window_variance_percentile is not None else None
    streakiness_score = _average_defined(runs_cluster_percentile, transition_percentile, window_variance_percentile)
    classification = _classify_shooter(
        streakiness_score=streakiness_score,
        consistency_score=consistency_score,
        window_variance_percentile=window_variance_percentile,
        runs_cluster_percentile=runs_cluster_percentile,
        runs_alternation_percentile=runs_alternation_percentile,
        transition_percentile=transition_percentile,
    )

    return {
        "attempts": attempts,
        "makes": made,
        "make_pct": float(made / attempts) if attempts else 0.0,
        "classification": classification,
        "streakiness_score": streakiness_score,
        "consistency_score": consistency_score,
        "runs": observed_runs,
        "runs_cluster_percentile": runs_cluster_percentile,
        "runs_alternation_percentile": runs_alternation_percentile,
        "transition_effect": observed_transition,
        "transition_percentile": transition_percentile,
        "window_size": resolved_window_size if observed_window_variance is not None else None,
        "window_variance": observed_window_variance,
        "window_variance_percentile": window_variance_percentile,
        "longest_make_run": observed_longest_make,
        "longest_make_run_percentile": longest_make_percentile,
        "longest_miss_run": observed_longest_miss,
        "longest_miss_run_percentile": longest_miss_percentile,
    }


@lru_cache(maxsize=64)
def _build_player_shot_streakiness_payload_cached(
    *,
    season: str,
    game_type: Optional[str] = None,
    shot_type: Optional[str] = None,
    min_attempts: int = 100,
    simulations: int = 1000,
    classification: Optional[str] = None,
    source_mtime_ns: int = 0,
) -> dict[str, Any]:
    path = PLAYER_SHOTS_ROOT / f"player_shots_{season}.parquet"
    game_type_norm = str(game_type or "regular_season").strip().lower()
    shot_type_norm = str(shot_type).strip().lower() if shot_type else None
    classification_norm = str(classification).strip().lower() if classification else None
    min_attempts = max(1, int(min_attempts or 100))
    simulations = max(100, min(int(simulations or 1000), 5000))

    if not path.exists():
        return {
            "season": season,
            "game_type": game_type_norm,
            "shot_type": shot_type_norm,
            "min_attempts": min_attempts,
            "simulations": simulations,
            "classification": classification_norm,
            "row_count": 0,
            "rows": [],
        }

    filters = [("game_type", "=", game_type_norm)]
    if shot_type_norm:
        filters.append(("shot_type", "=", shot_type_norm))

    try:
        frame = pd.read_parquet(path, columns=PLAYER_SHOT_STREAKINESS_COLUMNS, filters=filters)
    except TypeError:
        frame = pd.read_parquet(path, columns=PLAYER_SHOT_STREAKINESS_COLUMNS)

    if frame.empty:
        rows: list[dict[str, Any]] = []
    else:
        frame = frame[frame["game_type"].astype(str).str.lower().eq(game_type_norm)].copy()
        if shot_type_norm:
            frame = frame[frame["shot_type"].astype(str).str.lower().eq(shot_type_norm)]
        frame = frame[frame["shot_type"].astype(str).str.lower().isin(PLAYER_SHOT_TYPES)]
        frame = frame[frame["result"].astype(str).str.lower().isin({"make", "miss"})]
        frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce").fillna(0).astype(int)
        frame = frame[frame["player_id"] > 0]
        frame = frame.sort_values(["player_id", "shot_type", "game_date", "game_id", "action_number", "action_id"], kind="stable")

        rows = []
        grouped = frame.groupby(["player_id", "player_name", "shot_type"], dropna=False, sort=False)
        for (player_id, player_name, group_shot_type), group in grouped:
            if len(group) < min_attempts:
                continue
            results = group["result"].astype(str).str.lower().map({"make": 1, "miss": 0}).to_numpy(dtype=np.int8)
            metrics = analyze_shot_streakiness_sequence(
                results,
                season=season,
                player_id=int(player_id),
                shot_type=str(group_shot_type),
                game_type=game_type_norm,
                simulations=simulations,
            )
            teams = sorted({str(value) for value in group["team"].dropna().astype(str) if str(value)})
            row = {
                "season": season,
                "game_type": game_type_norm,
                "player_id": int(player_id),
                "player_name": str(player_name) if str(player_name) else "Unknown",
                "teams": teams,
                "shot_type": str(group_shot_type),
                **metrics,
            }
            if classification_norm and row["classification"].lower() != classification_norm:
                continue
            rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row.get("streakiness_score") or -1),
            -int(row.get("attempts") or 0),
            str(row.get("player_name") or ""),
            str(row.get("shot_type") or ""),
        )
    )
    return {
        "season": season,
        "game_type": game_type_norm,
        "shot_type": shot_type_norm,
        "min_attempts": min_attempts,
        "simulations": simulations,
        "classification": classification_norm,
        "row_count": len(rows),
        "rows": rows,
    }


def build_player_shot_streakiness_payload(
    *,
    season: str,
    game_type: Optional[str] = None,
    shot_type: Optional[str] = None,
    min_attempts: int = 100,
    simulations: int = 1000,
    classification: Optional[str] = None,
) -> dict[str, Any]:
    path = PLAYER_SHOTS_ROOT / f"player_shots_{season}.parquet"
    source_mtime_ns = path.stat().st_mtime_ns if path.exists() else 0
    return _build_player_shot_streakiness_payload_cached(
        season=season,
        game_type=game_type,
        shot_type=shot_type,
        min_attempts=min_attempts,
        simulations=simulations,
        classification=classification,
        source_mtime_ns=source_mtime_ns,
    )


build_player_shot_streakiness_payload.cache_clear = _build_player_shot_streakiness_payload_cached.cache_clear  # type: ignore[attr-defined]
