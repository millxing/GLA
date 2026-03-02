#!/usr/bin/env python3
"""
Build persisted situational datasets from packed PBP game-state artifacts.

Outputs per season:
  - team_game_logs_garbage_filtered_<season>.csv
  - team_game_logs_clutch_<season>.csv
  - box_score_advanced_garbage_filtered_<season>.csv
  - box_score_advanced_clutch_<season>.csv

Definitions (post-event state):
  - Garbage state (stateful latch):
      Enter when period >= 3 and abs(diff) > 5 and
      (home_wp >= garbage_wp_on or home_wp <= (1 - garbage_wp_on)),
      except no new garbage entry in the final game minute.
      Stay latched until home_wp moves back inside
      ((1 - garbage_wp_off), garbage_wp_off).
  - Clutch event:
      period >= 4 and seconds_left < 300 and abs(diff) <= 5

These conditions are mutually exclusive by construction because garbage requires
abs(diff) > 5 while clutch requires abs(diff) <= 5.

Scope-state mode:
  - pre (default): attribute each event to the game state before it occurred.
  - post: attribute each event to the game state after it occurred.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import pyarrow.parquet as pq

from cli import (
    ADVANCED_COLUMNS,
    DEFAULT_REPO_DIR,
    EXPECTED_COLUMNS,
    _advanced_filename,
    _load_existing_season_csv,
    _normalize_advanced_df,
    _normalize_game_level_df,
    _season_to_filename,
    ensure_data_repo,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_available_seasons


CLOCK_RE = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")
DATA_SCOPES = ("garbage_filtered", "clutch")
DEFAULT_SCOPE_STATE_MODE = "pre"
VALID_SCOPE_STATE_MODES = {"pre", "post"}
DEFAULT_GARBAGE_WP_ON = 0.95
DEFAULT_GARBAGE_WP_OFF = 0.90
TRACKED_STATS = (
    "pts",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "plus_minus",
)


def _normalize_game_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    return digits.zfill(10)


def _to_optional_int(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, out))


def _clock_to_seconds_left(clock_text: Any, period: Optional[int]) -> Optional[int]:
    if period is None or period <= 0:
        return None
    text = str(clock_text or "").strip()
    match = CLOCK_RE.match(text)
    if not match:
        return None
    mins = float(match.group(1) or 0.0)
    secs = float(match.group(2) or 0.0)
    total = int(mins * 60.0 + secs)
    max_seconds = 300 if period > 4 else 720
    if total < 0:
        return 0
    if total > max_seconds:
        return max_seconds
    return total


def _clock_to_seconds_left_float(clock_text: Any, period: Optional[int]) -> Optional[float]:
    if period is None or period <= 0:
        return None
    text = str(clock_text or "").strip()
    match = CLOCK_RE.match(text)
    if not match:
        return None
    mins = float(match.group(1) or 0.0)
    secs = float(match.group(2) or 0.0)
    total = mins * 60.0 + secs
    max_seconds = 300.0 if period > 4 else 720.0
    if total < 0:
        return 0.0
    if total > max_seconds:
        return max_seconds
    return float(total)


def _period_elapsed_seconds(period: Optional[int], clock_text: Any) -> Optional[float]:
    if period is None or period <= 0:
        return None
    seconds_left = _clock_to_seconds_left_float(clock_text, period)
    if seconds_left is None:
        return None
    if period <= 4:
        period_len = 720.0
        elapsed_before = float((period - 1) * 720)
    else:
        period_len = 300.0
        elapsed_before = 2880.0 + float((period - 5) * 300)
    elapsed = elapsed_before + max(0.0, period_len - seconds_left)
    return max(0.0, elapsed)


def _period_total_seconds(max_period: int) -> float:
    if max_period <= 4:
        return float(max_period * 720)
    return 2880.0 + float((max_period - 4) * 300)


def _scope_game_logs_filename(season: str, scope: str) -> str:
    return f"team_game_logs_{scope}_{season}.csv"


def _scope_advanced_filename(season: str, scope: str) -> str:
    return f"box_score_advanced_{scope}_{season}.csv"


def _states_parquet_path(repo_dir: Path, season: str, phase: str) -> Path:
    return repo_dir / "PBPdata" / "game_states" / phase / season / f"_states_{season}_{phase}.parquet"


def _parse_payload(raw: Any) -> Optional[dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except Exception:
            return None
        if isinstance(decoded, dict):
            return decoded
    return None


def _parse_home_wp_series(raw: Any) -> Optional[list[Optional[float]]]:
    values: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            values = json.loads(text)
        except Exception:
            return None
    if not isinstance(values, list):
        return None

    normalized: list[Optional[float]] = []
    for value in values:
        normalized.append(_to_optional_float(value))
    return normalized


def _attach_home_wp_series(payload: dict[str, Any], series: Optional[list[Optional[float]]]) -> None:
    if not series:
        return
    events = payload.get("events")
    if not isinstance(events, list):
        return
    max_idx = min(len(events), len(series))
    for idx in range(max_idx):
        if series[idx] is None:
            continue
        event = events[idx]
        if not isinstance(event, dict):
            continue
        if event.get("home_win_prob") is None:
            event["home_win_prob"] = float(series[idx])


def _iter_season_payloads(repo_dir: Path, season: str) -> Iterable[tuple[str, dict[str, Any]]]:
    for phase in ("regular", "playoffs"):
        parquet_path = _states_parquet_path(repo_dir=repo_dir, season=season, phase=phase)
        if not parquet_path.exists():
            continue

        try:
            table = pq.read_table(
                parquet_path,
                columns=["game_id", "payload_json", "home_win_prob_by_event_json"],
                use_threads=False,
            )
            has_series = True
        except Exception:
            table = pq.read_table(
                parquet_path,
                columns=["game_id", "payload_json"],
                use_threads=False,
            )
            has_series = False

        col_game = table.column("game_id")
        col_payload = table.column("payload_json")
        col_series = table.column("home_win_prob_by_event_json") if has_series else None

        for idx in range(table.num_rows):
            gid = _normalize_game_id(col_game[idx].as_py())
            payload = _parse_payload(col_payload[idx].as_py())
            if not gid or payload is None:
                continue
            if col_series is not None:
                series = _parse_home_wp_series(col_series[idx].as_py())
                _attach_home_wp_series(payload, series)
            yield gid, payload


def _empty_totals() -> Dict[str, float]:
    return {k: 0.0 for k in TRACKED_STATS}


def _init_scope_state() -> Dict[str, Any]:
    return {
        "included_events": 0,
        "home": _empty_totals(),
        "road": _empty_totals(),
        "poss_home": 0.0,
        "poss_road": 0.0,
        "seconds": 0.0,
    }


def _coerce_side(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"home", "road"}:
        return text
    return None


def _validate_garbage_wp_threshold(value: float, name: str) -> float:
    threshold = float(value)
    if not (0.5 < threshold < 1.0):
        raise ValueError(f"{name} must be between 0.5 and 1.0 (exclusive); got {value!r}")
    return threshold


def _validate_garbage_wp_pair(garbage_wp_on: float, garbage_wp_off: float) -> None:
    if garbage_wp_off > garbage_wp_on:
        raise ValueError(
            "garbage_wp_off must be <= garbage_wp_on so exit is not stricter than entry "
            f"(got on={garbage_wp_on!r}, off={garbage_wp_off!r})"
        )


def _event_context(event: dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int], Optional[float], Optional[int]]:
    period = _to_optional_int(event.get("period"))
    if period is None or period <= 0:
        return None, None, None, None, None

    state = event.get("game_log_state")
    if not isinstance(state, dict):
        state = {}
    pts_home = _to_optional_int(state.get("pts_home"))
    pts_road = _to_optional_int(state.get("pts_road"))
    if pts_home is None:
        pts_home = _to_optional_int(event.get("score_home"))
    if pts_road is None:
        pts_road = _to_optional_int(event.get("score_away"))

    seconds_left = _clock_to_seconds_left(event.get("clock"), period)
    home_wp = _to_optional_float(event.get("home_win_prob"))
    return period, pts_home, pts_road, home_wp, seconds_left


def _classify_clutch_event(event: dict[str, Any]) -> bool:
    period, pts_home, pts_road, _home_wp, seconds_left = _event_context(event)
    if period is None:
        return False

    is_clutch = bool(
        pts_home is not None
        and pts_road is not None
        and abs(pts_home - pts_road) <= 5
        and period >= 4
        and seconds_left is not None
        and seconds_left < 300
    )
    return is_clutch


def _should_enter_garbage(event: dict[str, Any], garbage_wp_on: float) -> bool:
    period, pts_home, pts_road, home_wp, _seconds_left = _event_context(event)
    if period is None:
        return False
    abs_diff: Optional[int] = None
    if pts_home is not None and pts_road is not None:
        abs_diff = abs(pts_home - pts_road)

    return bool(
        abs_diff is not None
        and abs_diff > 5
        and period >= 3
        and home_wp is not None
        and (home_wp >= garbage_wp_on or home_wp <= (1.0 - garbage_wp_on))
    )


def _should_exit_garbage(event: dict[str, Any], garbage_wp_off: float) -> bool:
    _period, _pts_home, _pts_road, home_wp, _seconds_left = _event_context(event)
    if home_wp is None:
        return False
    return bool((1.0 - garbage_wp_off) < home_wp < garbage_wp_off)


def _is_final_game_minute(
    *,
    period: Optional[int],
    clock_text: Any,
    max_period: int,
) -> bool:
    if period is None or period <= 0:
        return False
    if period != max_period:
        return False
    seconds_left = _clock_to_seconds_left(clock_text, period)
    return bool(seconds_left is not None and seconds_left < 60)


def _accumulate_changed_stats(scope_state: Dict[str, Any], event: dict[str, Any]) -> None:
    changed = event.get("changed_stats")
    if not isinstance(changed, dict):
        return

    for side in ("home", "road"):
        side_changes = changed.get(side)
        if not isinstance(side_changes, dict):
            continue
        totals = scope_state[side]
        for stat_key in TRACKED_STATS:
            delta = side_changes.get(stat_key)
            if delta is None:
                continue
            try:
                totals[stat_key] += float(delta)
            except Exception:
                continue


def _accumulate_possessions(scope_state: Dict[str, Any], event: dict[str, Any]) -> None:
    if not bool(event.get("possession_changed")):
        return

    before_side = _coerce_side(event.get("possession_before_side"))
    after_side = _coerce_side(event.get("possession_after_side"))
    if before_side not in {"home", "road"}:
        return
    if after_side == before_side:
        return

    if before_side == "home":
        scope_state["poss_home"] += 1.0
    else:
        scope_state["poss_road"] += 1.0


def _pct(made: int, attempts: int) -> float:
    if attempts <= 0:
        return 0.0
    return round(float(made) / float(attempts), 3)


def _resolve_wl_home(home_pts: int, road_pts: int, base_row: Dict[str, Any]) -> str:
    if home_pts > road_pts:
        return "W"
    if home_pts < road_pts:
        return "L"
    fallback = str(base_row.get("wl_home") or "").strip().upper()
    if fallback in {"W", "L"}:
        return fallback
    return "L"


def _default_minutes_from_payload(payload: dict[str, Any]) -> int:
    events = payload.get("events")
    max_period = 4
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            period = _to_optional_int(event.get("period"))
            if period is not None and period > max_period:
                max_period = period
    game_minutes = 48 + max(0, max_period - 4) * 5
    return int(game_minutes * 5)


def _build_rows_for_game(
    payload: dict[str, Any],
    base_row: Dict[str, Any],
    base_adv_row: Optional[Dict[str, Any]],
    scope_state_mode: str = DEFAULT_SCOPE_STATE_MODE,
    garbage_wp_on: float = DEFAULT_GARBAGE_WP_ON,
    garbage_wp_off: float = DEFAULT_GARBAGE_WP_OFF,
) -> Dict[str, tuple[Dict[str, Any], Dict[str, Any]]]:
    if scope_state_mode not in VALID_SCOPE_STATE_MODES:
        raise ValueError(
            f"Invalid scope_state_mode={scope_state_mode!r}. "
            f"Expected one of {sorted(VALID_SCOPE_STATE_MODES)}"
        )
    garbage_wp_on = _validate_garbage_wp_threshold(garbage_wp_on, "garbage_wp_on")
    garbage_wp_off = _validate_garbage_wp_threshold(garbage_wp_off, "garbage_wp_off")
    _validate_garbage_wp_pair(garbage_wp_on=garbage_wp_on, garbage_wp_off=garbage_wp_off)

    scope_states = {scope: _init_scope_state() for scope in DATA_SCOPES}

    events = payload.get("events")
    if not isinstance(events, list):
        events = []

    filtered_events: list[dict[str, Any]] = []
    max_period = 4
    for event in events:
        if not isinstance(event, dict):
            continue
        if bool(event.get("event_quarantined")):
            continue
        filtered_events.append(event)
        period = _to_optional_int(event.get("period"))
        if period is not None and period > max_period:
            max_period = period

    annotated_events: list[dict[str, Any]] = []
    prev_is_garbage = False
    prev_is_clutch = False
    for event in filtered_events:
        period = _to_optional_int(event.get("period"))
        elapsed = _period_elapsed_seconds(period=period, clock_text=event.get("clock"))

        if prev_is_garbage:
            is_garbage = not _should_exit_garbage(event, garbage_wp_off=garbage_wp_off)
        else:
            is_garbage = (
                not _is_final_game_minute(
                    period=period,
                    clock_text=event.get("clock"),
                    max_period=max_period,
                )
                and _should_enter_garbage(event, garbage_wp_on=garbage_wp_on)
            )
        is_clutch = _classify_clutch_event(event)
        if scope_state_mode == "pre":
            include_garbage_filtered = not prev_is_garbage
            include_clutch = prev_is_clutch
        else:
            include_garbage_filtered = not is_garbage
            include_clutch = is_clutch
        annotated_events.append({
            "event": event,
            "elapsed": elapsed,
            "include_by_scope": {
                "garbage_filtered": include_garbage_filtered,
                "clutch": include_clutch,
            },
        })
        prev_is_garbage = is_garbage
        prev_is_clutch = is_clutch

    total_game_seconds = _period_total_seconds(max_period)

    for idx, item in enumerate(annotated_events):
        event = item["event"]
        include_by_scope = item["include_by_scope"]
        elapsed_curr = item.get("elapsed")

        elapsed_next: Optional[float] = None
        for nxt in annotated_events[idx + 1:]:
            candidate = nxt.get("elapsed")
            if candidate is not None:
                elapsed_next = float(candidate)
                break
        if elapsed_next is None and elapsed_curr is not None:
            elapsed_next = total_game_seconds

        segment_seconds = 0.0
        if elapsed_curr is not None and elapsed_next is not None:
            segment_seconds = max(0.0, float(elapsed_next) - float(elapsed_curr))

        for scope, include in include_by_scope.items():
            if not include:
                continue
            scope_state = scope_states[scope]
            scope_state["included_events"] += 1
            _accumulate_changed_stats(scope_state, event)
            _accumulate_possessions(scope_state, event)
            if segment_seconds > 0.0:
                scope_state["seconds"] += segment_seconds

    out: Dict[str, tuple[Dict[str, Any], Dict[str, Any]]] = {}

    for scope in DATA_SCOPES:
        state = scope_states[scope]
        if state["included_events"] <= 0:
            continue

        home = state["home"]
        road = state["road"]

        signal_total = 0.0
        for stat in ("pts", "fga", "fta", "tov", "oreb", "dreb"):
            signal_total += abs(float(home.get(stat, 0.0))) + abs(float(road.get(stat, 0.0)))
        if signal_total <= 0.0 and (state["poss_home"] + state["poss_road"]) <= 0.0:
            continue

        home_pts = int(round(home.get("pts", 0.0)))
        road_pts = int(round(road.get("pts", 0.0)))
        home_pm = home_pts - road_pts
        road_pm = -home_pm

        game_row = dict(base_row)
        game_row["pts_home"] = home_pts
        game_row["pts_road"] = road_pts
        game_row["wl_home"] = _resolve_wl_home(home_pts=home_pts, road_pts=road_pts, base_row=base_row)

        for stat in (
            "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb",
            "dreb", "reb", "ast", "stl", "blk", "tov", "pf",
        ):
            game_row[f"{stat}_home"] = int(round(home.get(stat, 0.0)))
            game_row[f"{stat}_road"] = int(round(road.get(stat, 0.0)))

        # Keep rebounds internally consistent if event deltas did not provide REB directly.
        if int(game_row["reb_home"]) == 0 and (int(game_row["oreb_home"]) or int(game_row["dreb_home"])):
            game_row["reb_home"] = int(game_row["oreb_home"]) + int(game_row["dreb_home"])
        if int(game_row["reb_road"]) == 0 and (int(game_row["oreb_road"]) or int(game_row["dreb_road"])):
            game_row["reb_road"] = int(game_row["oreb_road"]) + int(game_row["dreb_road"])

        game_row["plus_minus_home"] = home_pm
        game_row["plus_minus_road"] = road_pm
        game_row["fg_pct_home"] = _pct(int(game_row["fgm_home"]), int(game_row["fga_home"]))
        game_row["fg3_pct_home"] = _pct(int(game_row["fg3m_home"]), int(game_row["fg3a_home"]))
        game_row["ft_pct_home"] = _pct(int(game_row["ftm_home"]), int(game_row["fta_home"]))
        game_row["fg_pct_road"] = _pct(int(game_row["fgm_road"]), int(game_row["fga_road"]))
        game_row["fg3_pct_road"] = _pct(int(game_row["fg3m_road"]), int(game_row["fg3a_road"]))
        game_row["ft_pct_road"] = _pct(int(game_row["ftm_road"]), int(game_row["fta_road"]))

        scoped_player_minutes = int(round(float(state.get("seconds", 0.0)) / 12.0))
        if scoped_player_minutes < 0:
            scoped_player_minutes = 0

        # Fallback to possession ratio if clock-based elapsed time is unavailable.
        if scoped_player_minutes <= 0 and base_adv_row:
            base_minutes_home = _to_optional_int(base_adv_row.get("minutes_home"))
            base_minutes_road = _to_optional_int(base_adv_row.get("minutes_road"))
            base_poss_home = pd.to_numeric(base_adv_row.get("possessions_home"), errors="coerce")
            base_poss_road = pd.to_numeric(base_adv_row.get("possessions_road"), errors="coerce")

            if (
                base_minutes_home is not None
                and pd.notna(base_poss_home)
                and float(base_poss_home) > 0.0
                and float(state["poss_home"]) > 0.0
            ):
                scoped_player_minutes = int(
                    round(float(base_minutes_home) * float(state["poss_home"]) / float(base_poss_home))
                )
            elif (
                base_minutes_road is not None
                and pd.notna(base_poss_road)
                and float(base_poss_road) > 0.0
                and float(state["poss_road"]) > 0.0
            ):
                scoped_player_minutes = int(
                    round(float(base_minutes_road) * float(state["poss_road"]) / float(base_poss_road))
                )

        # Never fall back to full-game 240/265 minutes for scoped rows. When
        # we have any scoped signal but unresolved elapsed time, use a 1-minute
        # floor to avoid pace distortion.
        if scoped_player_minutes <= 0 and (
            signal_total > 0.0 or state["poss_home"] > 0.0 or state["poss_road"] > 0.0
        ):
            scoped_player_minutes = 1

        adv_row = {
            "game_id": _normalize_game_id(base_row.get("game_id")),
            "game_date": str(base_row.get("game_date") or ""),
            "season": str(base_row.get("season") or ""),
            "team_id_home": _to_optional_int(base_row.get("team_id_home")) or 0,
            "team_abbreviation_home": str(base_row.get("team_abbreviation_home") or ""),
            "minutes_home": scoped_player_minutes,
            "possessions_home": round(float(state["poss_home"]), 1),
            "team_id_road": _to_optional_int(base_row.get("team_id_road")) or 0,
            "team_abbreviation_road": str(base_row.get("team_abbreviation_road") or ""),
            "minutes_road": scoped_player_minutes,
            "possessions_road": round(float(state["poss_road"]), 1),
        }

        if base_adv_row:
            adv_row["team_id_home"] = _to_optional_int(base_adv_row.get("team_id_home")) or adv_row["team_id_home"]
            adv_row["team_abbreviation_home"] = str(base_adv_row.get("team_abbreviation_home") or adv_row["team_abbreviation_home"])
            adv_row["team_id_road"] = _to_optional_int(base_adv_row.get("team_id_road")) or adv_row["team_id_road"]
            adv_row["team_abbreviation_road"] = str(base_adv_row.get("team_abbreviation_road") or adv_row["team_abbreviation_road"])

        out[scope] = (game_row, adv_row)

    return out


def _merge_incremental(existing: pd.DataFrame, new_rows: pd.DataFrame, key: str) -> pd.DataFrame:
    if existing.empty:
        return new_rows
    if new_rows.empty:
        return existing

    existing = existing.copy()
    new_rows = new_rows.copy()
    existing[key] = existing[key].map(_normalize_game_id)
    new_rows[key] = new_rows[key].map(_normalize_game_id)
    to_add = new_rows[~new_rows[key].isin(set(existing[key]))]
    if to_add.empty:
        return existing
    return pd.concat([existing, to_add], ignore_index=True)


def build_situational_files_for_season(
    season: str,
    repo_dir: Path,
    incremental: bool = True,
    scope_state_mode: str = DEFAULT_SCOPE_STATE_MODE,
    garbage_wp_on: float = DEFAULT_GARBAGE_WP_ON,
    garbage_wp_off: float = DEFAULT_GARBAGE_WP_OFF,
) -> int:
    if scope_state_mode not in VALID_SCOPE_STATE_MODES:
        raise ValueError(
            f"Invalid scope_state_mode={scope_state_mode!r}. "
            f"Expected one of {sorted(VALID_SCOPE_STATE_MODES)}"
        )
    garbage_wp_on = _validate_garbage_wp_threshold(garbage_wp_on, "garbage_wp_on")
    garbage_wp_off = _validate_garbage_wp_threshold(garbage_wp_off, "garbage_wp_off")
    _validate_garbage_wp_pair(garbage_wp_on=garbage_wp_on, garbage_wp_off=garbage_wp_off)

    game_logs_path = repo_dir / _season_to_filename(season)
    if not game_logs_path.exists():
        print(f"[situational] Skip {season}: missing {game_logs_path.name}")
        return 0

    base_game_raw = _load_existing_season_csv(game_logs_path)
    if base_game_raw is None:
        print(f"[situational] Skip {season}: failed to load {game_logs_path.name}")
        return 0
    base_game_df = _normalize_game_level_df(base_game_raw)
    base_game_lookup = {
        _normalize_game_id(row.get("game_id")): row.to_dict()
        for _, row in base_game_df.iterrows()
    }

    base_adv_path = repo_dir / _advanced_filename(season)
    base_adv_lookup: Dict[str, Dict[str, Any]] = {}
    if base_adv_path.exists():
        try:
            base_adv_df = _normalize_advanced_df(pd.read_csv(base_adv_path, dtype={"game_id": "string"}))
            base_adv_lookup = {
                _normalize_game_id(row.get("game_id")): row.to_dict()
                for _, row in base_adv_df.iterrows()
            }
        except Exception as exc:
            print(f"[situational] Warning: failed to load {base_adv_path.name}: {exc}")

    game_rows_by_scope: Dict[str, list[Dict[str, Any]]] = {scope: [] for scope in DATA_SCOPES}
    adv_rows_by_scope: Dict[str, list[Dict[str, Any]]] = {scope: [] for scope in DATA_SCOPES}

    seen_game_ids: set[str] = set()
    payload_count = 0
    for gid, payload in _iter_season_payloads(repo_dir=repo_dir, season=season):
        if gid in seen_game_ids:
            continue
        seen_game_ids.add(gid)
        payload_count += 1

        base_row = base_game_lookup.get(gid)
        if base_row is None:
            continue
        scope_rows = _build_rows_for_game(
            payload=payload,
            base_row=base_row,
            base_adv_row=base_adv_lookup.get(gid),
            scope_state_mode=scope_state_mode,
            garbage_wp_on=garbage_wp_on,
            garbage_wp_off=garbage_wp_off,
        )
        for scope, (game_row, adv_row) in scope_rows.items():
            game_rows_by_scope[scope].append(game_row)
            adv_rows_by_scope[scope].append(adv_row)

    if payload_count == 0:
        print(f"[situational] Skip {season}: no packed game-state parquet rows found")
        return 0

    for scope in DATA_SCOPES:
        output_game_path = repo_dir / _scope_game_logs_filename(season, scope)
        output_adv_path = repo_dir / _scope_advanced_filename(season, scope)

        new_game_df = pd.DataFrame(game_rows_by_scope[scope], columns=EXPECTED_COLUMNS)
        new_adv_df = pd.DataFrame(adv_rows_by_scope[scope], columns=ADVANCED_COLUMNS)

        if incremental and output_game_path.exists():
            try:
                existing_game_df = _normalize_game_level_df(pd.read_csv(output_game_path, dtype={"game_id": "string"}))
            except Exception:
                existing_game_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
            merged_game = _merge_incremental(existing_game_df, new_game_df, key="game_id")
        else:
            merged_game = new_game_df

        if incremental and output_adv_path.exists():
            try:
                existing_adv_df = _normalize_advanced_df(pd.read_csv(output_adv_path, dtype={"game_id": "string"}))
            except Exception:
                existing_adv_df = pd.DataFrame(columns=ADVANCED_COLUMNS)
            merged_adv = _merge_incremental(existing_adv_df, new_adv_df, key="game_id")
        else:
            merged_adv = new_adv_df

        if merged_game.empty:
            merged_game = pd.DataFrame(columns=EXPECTED_COLUMNS)
        else:
            merged_game = _normalize_game_level_df(merged_game)

        if merged_adv.empty:
            merged_adv = pd.DataFrame(columns=ADVANCED_COLUMNS)
        else:
            merged_adv = _normalize_advanced_df(merged_adv)

        merged_game.to_csv(output_game_path, index=False)
        merged_adv.to_csv(output_adv_path, index=False)

        print(
            f"[situational] {season} {scope}: "
            f"team_rows={len(merged_game)} adv_rows={len(merged_adv)} "
            f"mode={scope_state_mode} garbage_wp_on={garbage_wp_on:.3f} "
            f"garbage_wp_off={garbage_wp_off:.3f} "
            f"({output_game_path.name}, {output_adv_path.name})"
        )

    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build situational game-log/advanced CSVs from packed game-state artifacts."
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Specific season (e.g., 2025-26). Default: all available seasons.",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help=f"Path to NBA_Data repository (default: {DEFAULT_REPO_DIR})",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Append only missing game_id rows to existing situational files.",
    )
    parser.add_argument(
        "--scope-state-mode",
        type=str,
        default=DEFAULT_SCOPE_STATE_MODE,
        choices=sorted(VALID_SCOPE_STATE_MODES),
        help=(
            "Event state attribution mode for situational scopes. "
            "'pre' attributes events to the prior state, 'post' to the resulting state."
        ),
    )
    parser.add_argument(
        "--garbage-wp-on",
        type=float,
        default=DEFAULT_GARBAGE_WP_ON,
        help=(
            "Garbage-state entry threshold for home WP (symmetric by side): "
            "enter when home_wp >= threshold or <= (1-threshold)."
        ),
    )
    parser.add_argument(
        "--garbage-wp-off",
        type=float,
        default=DEFAULT_GARBAGE_WP_OFF,
        help=(
            "Garbage-state exit threshold for home WP (symmetric by side): "
            "exit when (1-threshold) < home_wp < threshold."
        ),
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir) if args.repo_dir else DEFAULT_REPO_DIR
    repo_dir = ensure_data_repo(repo_dir)

    seasons = [args.season] if args.season else get_available_seasons()
    processed = 0
    for season in seasons:
        processed += build_situational_files_for_season(
            season=season,
            repo_dir=repo_dir,
            incremental=args.incremental,
            scope_state_mode=args.scope_state_mode,
            garbage_wp_on=args.garbage_wp_on,
            garbage_wp_off=args.garbage_wp_off,
        )

    print(f"[situational] Completed. seasons_processed={processed}/{len(seasons)}")


if __name__ == "__main__":
    main()
