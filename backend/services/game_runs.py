import math
from typing import Any, Optional


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return float(min(1.0, max(0.0, result)))
    except (TypeError, ValueError):
        return None


def _normalize_possession_side(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"home", "road"}:
        return text
    return None


def _start_possession_snapshot(
    raw_events: list[dict[str, Any]],
    event_idx: int,
    side: str,
    use_previous_anchor: bool = False,
) -> dict[str, Any]:
    anchor_idx = event_idx
    if use_previous_anchor and event_idx > 0:
        prev_wp = _to_float_or_none(raw_events[event_idx - 1].get("home_win_prob"))
        if prev_wp is not None:
            anchor_idx = event_idx - 1

    anchor = raw_events[anchor_idx]
    state = anchor.get("game_log_state")
    if not isinstance(state, dict):
        state = {}

    return {
        "side": side,
        "start_event_pos": anchor_idx,
        "start_event_index": _to_int_or_none(anchor.get("event_index")),
        "start_period": _to_int_or_none(anchor.get("period")),
        "start_clock": str(anchor.get("clock") or "") or None,
        "start_description": str(anchor.get("description") or ""),
        "start_home_win_prob": _to_float_or_none(anchor.get("home_win_prob")),
        "start_home_score": _to_int_or_none(state.get("pts_home")),
        "start_road_score": _to_int_or_none(state.get("pts_road")),
    }


def extract_timeline_possessions(
    raw_events: list[dict[str, Any]],
    home_team: str,
    road_team: str,
) -> list[dict[str, Any]]:
    if not raw_events:
        return []

    possessions: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None

    for idx, event in enumerate(raw_events):
        before_side = _normalize_possession_side(event.get("possession_before_side"))
        after_side = _normalize_possession_side(event.get("possession_after_side"))
        changed = bool(event.get("possession_changed"))

        if current is None:
            inferred_start = before_side or after_side
            if inferred_start:
                current = _start_possession_snapshot(
                    raw_events=raw_events,
                    event_idx=idx,
                    side=inferred_start,
                    use_previous_anchor=(before_side is None and after_side is not None),
                )
        elif before_side and before_side != current["side"]:
            current = _start_possession_snapshot(
                raw_events=raw_events,
                event_idx=idx,
                side=before_side,
            )

        if current is None:
            continue

        current_side = current["side"]
        if changed and before_side == current_side and after_side != current_side:
            state = event.get("game_log_state")
            if not isinstance(state, dict):
                state = {}

            end_home_score = _to_int_or_none(state.get("pts_home"))
            end_road_score = _to_int_or_none(state.get("pts_road"))
            start_home_score = current.get("start_home_score")
            start_road_score = current.get("start_road_score")
            possessions.append(
                {
                    **current,
                    "team": home_team if current_side == "home" else road_team,
                    "end_event_pos": idx,
                    "end_event_index": _to_int_or_none(event.get("event_index")),
                    "end_period": _to_int_or_none(event.get("period")),
                    "end_clock": str(event.get("clock") or "") or None,
                    "end_description": str(event.get("description") or ""),
                    "end_home_win_prob": _to_float_or_none(event.get("home_win_prob")),
                    "end_home_score": end_home_score,
                    "end_road_score": end_road_score,
                    "home_points_scored": (
                        end_home_score - start_home_score
                        if end_home_score is not None and start_home_score is not None
                        else None
                    ),
                    "road_points_scored": (
                        end_road_score - start_road_score
                        if end_road_score is not None and start_road_score is not None
                        else None
                    ),
                }
            )
            current = (
                _start_possession_snapshot(raw_events=raw_events, event_idx=idx, side=after_side)
                if after_side in {"home", "road"}
                else None
            )

    if current is not None:
        final_event = raw_events[-1]
        state = final_event.get("game_log_state")
        if not isinstance(state, dict):
            state = {}

        end_home_score = _to_int_or_none(state.get("pts_home"))
        end_road_score = _to_int_or_none(state.get("pts_road"))
        start_home_score = current.get("start_home_score")
        start_road_score = current.get("start_road_score")
        possessions.append(
            {
                **current,
                "team": home_team if current["side"] == "home" else road_team,
                "end_event_pos": len(raw_events) - 1,
                "end_event_index": _to_int_or_none(final_event.get("event_index")),
                "end_period": _to_int_or_none(final_event.get("period")),
                "end_clock": str(final_event.get("clock") or "") or None,
                "end_description": str(final_event.get("description") or ""),
                "end_home_win_prob": _to_float_or_none(final_event.get("home_win_prob")),
                "end_home_score": end_home_score,
                "end_road_score": end_road_score,
                "home_points_scored": (
                    end_home_score - start_home_score
                    if end_home_score is not None and start_home_score is not None
                    else None
                ),
                "road_points_scored": (
                    end_road_score - start_road_score
                    if end_road_score is not None and start_road_score is not None
                    else None
                ),
            }
        )

    return possessions


def rank_non_overlapping_runs(
    possessions: list[dict[str, Any]],
    home_team: str,
    road_team: str,
    max_possessions: Optional[int] = None,
    run_alpha: float = 0.6,
    min_possessions: int = 1,
    min_margin: int = 0,
    numerator: str = "dwp",
    limit: int = 4,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if limit <= 0:
        return candidates
    if min_possessions <= 0:
        return candidates
    if min_margin < 0:
        return candidates
    if max_possessions is not None and max_possessions <= 0:
        return candidates
    if numerator not in {"dwp", "dscore"}:
        return candidates

    for start_idx in range(len(possessions)):
        start_wp = _to_float_or_none(possessions[start_idx].get("start_home_win_prob"))
        if start_wp is None:
            continue

        end_bound = len(possessions) if max_possessions is None else min(len(possessions), start_idx + max_possessions)
        for end_idx in range(start_idx, end_bound):
            end_wp = _to_float_or_none(possessions[end_idx].get("end_home_win_prob"))
            if end_wp is None:
                continue

            possession_count = end_idx - start_idx + 1
            if possession_count < min_possessions:
                continue
            delta_home_wp = float(end_wp - start_wp)

            start_poss = possessions[start_idx]
            end_poss = possessions[end_idx]
            home_score_start = _to_int_or_none(start_poss.get("start_home_score"))
            road_score_start = _to_int_or_none(start_poss.get("start_road_score"))
            home_score_end = _to_int_or_none(end_poss.get("end_home_score"))
            road_score_end = _to_int_or_none(end_poss.get("end_road_score"))
            home_points_scored = (
                home_score_end - home_score_start
                if home_score_end is not None and home_score_start is not None
                else None
            )
            road_points_scored = (
                road_score_end - road_score_start
                if road_score_end is not None and road_score_start is not None
                else None
            )
            score_margin_delta = (
                (home_points_scored - road_points_scored)
                if home_points_scored is not None and road_points_scored is not None
                else None
            )
            if score_margin_delta is None or abs(score_margin_delta) < min_margin:
                continue
            numerator_value = float(delta_home_wp if numerator == "dwp" else score_margin_delta)
            run_score = numerator_value / math.pow(possession_count + 1, run_alpha)
            abs_run_score = abs(run_score)
            if abs_run_score <= 0.0:
                continue

            candidates.append(
                {
                    "run_side": "home" if numerator_value > 0 else "road" if numerator_value < 0 else None,
                    "run_team": home_team if numerator_value > 0 else road_team if numerator_value < 0 else None,
                    "start_possession_index": start_idx,
                    "end_possession_index": end_idx,
                    "possession_count": possession_count,
                    "start_event_index": _to_int_or_none(start_poss.get("start_event_index")),
                    "end_event_index": _to_int_or_none(end_poss.get("end_event_index")),
                    "start_period": _to_int_or_none(start_poss.get("start_period")),
                    "start_clock": start_poss.get("start_clock"),
                    "end_period": _to_int_or_none(end_poss.get("end_period")),
                    "end_clock": end_poss.get("end_clock"),
                    "start_description": str(start_poss.get("start_description") or ""),
                    "end_description": str(end_poss.get("end_description") or ""),
                    "home_win_prob_start": start_wp,
                    "home_win_prob_end": end_wp,
                    "delta_home_win_prob": delta_home_wp,
                    "run_score_numerator_value": numerator_value,
                    "run_score": run_score,
                    "abs_run_score": abs_run_score,
                    "home_score_start": home_score_start,
                    "road_score_start": road_score_start,
                    "home_score_end": home_score_end,
                    "road_score_end": road_score_end,
                    "home_points_scored": home_points_scored,
                    "road_points_scored": road_points_scored,
                    "score_margin_delta": score_margin_delta,
                }
            )

    candidates.sort(
        key=lambda row: (
            -float(row["abs_run_score"]),
            -abs(float(row["run_score_numerator_value"])),
            int(row["possession_count"]),
            int(row["start_possession_index"]),
            int(row["end_possession_index"]),
        )
    )

    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        overlaps = any(
            not (
                candidate["end_possession_index"] < existing["start_possession_index"]
                or candidate["start_possession_index"] > existing["end_possession_index"]
            )
            for existing in selected
        )
        if overlaps:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break

    for rank, run in enumerate(selected, start=1):
        run["rank"] = rank

    return selected
