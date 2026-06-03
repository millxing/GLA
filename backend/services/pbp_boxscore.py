from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import unicodedata
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow.parquet as pq

from admin.pbp_game_states import (  # type: ignore
    _build_pbp_path,
    _counts_toward_pf,
    _load_pbp_df,
    _normalize_game_id,
    _normalize_text,
    _safe_str,
)
from config import (
    PBP_GAME_STATES_ROOT,
    PBP_GITHUB_RAW_BASE_URL,
    PBP_REMOTE_CACHE_DIR,
    build_box_score_traditional_filename,
    build_data_file_url,
    build_data_filename,
    resolve_data_file_path,
)

ROTATION_TIMEOUT_SECONDS = 20.0
ROTATION_RETRIES = 2
REMOTE_CACHE_TTL_SECONDS = max(
    60,
    int(os.getenv("PBP_REMOTE_CACHE_TTL_SECONDS", "300") or "300"),
)
REMOTE_FETCH_TIMEOUT_SECONDS = max(
    10,
    int(os.getenv("PBP_REMOTE_FETCH_TIMEOUT_SECONDS", "60") or "60"),
)
logger = logging.getLogger(__name__)

SUPPORTED_STATS = [
    "pts",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "reb",
    "ast",
    "tov",
    "stl",
    "blk",
    "oreb",
    "dreb",
    "pf",
    "plus_minus",
]

ASSIST_RE = re.compile(r"\(([^()]*)\s+\d+\s+AST\)", re.IGNORECASE)
OFF_DEF_REB_RE = re.compile(r"\(Off:(\d+)\s+Def:(\d+)\)", re.IGNORECASE)
SUB_RE = re.compile(r"^SUB:\s*(.*?)\s+FOR\s+(.*?)\s*$", re.IGNORECASE)
CLOCK_RE = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")
HOME_WIN_PROB_BY_EVENT_JSON_COLUMN = "home_win_prob_by_event_json"
VALID_BOX_SCORE_SEGMENTS = {
    "all",
    "q1",
    "q2",
    "q3",
    "q4",
    "ot",
    "h1",
    "h2",
    "garbage_filtered",
    "garbage_time",
    "clutch",
}
DEFAULT_GARBAGE_WP_ON = 0.95
DEFAULT_GARBAGE_WP_OFF = 0.90


@lru_cache(maxsize=16)
def _read_data_csv(filename: str, dtype_key: tuple[tuple[str, str], ...] = ()) -> pd.DataFrame:
    dtype = dict(dtype_key) if dtype_key else None
    local_path = resolve_data_file_path(filename)
    if local_path.exists():
        return pd.read_csv(local_path, dtype=dtype)
    return pd.read_csv(build_data_file_url(filename), dtype=dtype)


def _load_data_csv(filename: str, dtype: Optional[dict[str, str]] = None) -> pd.DataFrame:
    dtype_key = tuple(sorted((dtype or {}).items()))
    return _read_data_csv(filename, dtype_key).copy()


def clear_pbp_boxscore_cache() -> None:
    _read_data_csv.cache_clear()
    try:
        if Path(PBP_REMOTE_CACHE_DIR).exists():
            shutil.rmtree(PBP_REMOTE_CACHE_DIR)
    except Exception:
        logger.warning("Failed to clear remote PBP cache: %s", PBP_REMOTE_CACHE_DIR)


def _download_remote_pbpdata_file(relative_path: str) -> Optional[Path]:
    rel = relative_path.lstrip("/")
    cache_path = (Path(PBP_REMOTE_CACHE_DIR) / rel).resolve()
    cache_exists = cache_path.exists()
    if cache_exists:
        try:
            age_seconds = max(0, int(time.time() - cache_path.stat().st_mtime))
            if age_seconds < REMOTE_CACHE_TTL_SECONDS:
                return cache_path
        except Exception:
            return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    remote_url = f"{PBP_GITHUB_RAW_BASE_URL.rstrip('/')}/{rel}"
    request = Request(remote_url, headers={"User-Agent": "GLA-pbp-boxscore-fallback"})
    try:
        with urlopen(request, timeout=REMOTE_FETCH_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        logger.warning("PBP box score remote fallback failed: relative_path=%s url=%s", rel, remote_url)
        return cache_path if cache_exists else None

    if not payload:
        logger.warning("PBP box score remote fallback returned empty payload: relative_path=%s url=%s", rel, remote_url)
        return cache_path if cache_exists else None

    cache_path.write_bytes(payload)
    return cache_path


def _pbpdata_relative_path(repo_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(Path(repo_dir) / "PBPdata").as_posix()
    except ValueError:
        pass

    rel = path
    try:
        rel = path.relative_to(repo_dir)
    except ValueError:
        pass
    parts = rel.parts
    if parts and parts[0] == "PBPdata":
        return Path(*parts[1:]).as_posix()
    return rel.as_posix()


def _resolve_pbp_input_path(repo_dir: Path, season: str, phase: str) -> tuple[Path, str]:
    pbp_path, pbp_source = _build_pbp_path(repo_dir, season, phase, source="auto")
    if pbp_path.exists():
        return pbp_path, pbp_source

    for source in ("nbastatsv3", "api_pbpv3"):
        candidate_path, candidate_source = _build_pbp_path(repo_dir, season, phase, source=source)
        if candidate_path.exists():
            return candidate_path, candidate_source
        remote_path = _download_remote_pbpdata_file(_pbpdata_relative_path(repo_dir, candidate_path))
        if remote_path and remote_path.exists():
            return remote_path, f"{candidate_source}_remote"

    return pbp_path, pbp_source


def _clean_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _safe_str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace(".", " ")
    text = re.sub(r"[^A-Za-z0-9' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _to_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _clock_to_seconds_remaining(clock: Any) -> Optional[float]:
    if pd.isna(clock):
        return None
    match = CLOCK_RE.match(str(clock).strip())
    if not match:
        return None
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2)) if match.group(2) else 0.0
    return minutes * 60.0 + seconds


def _period_length_seconds(period: int) -> float:
    return 300.0 if period > 4 else 720.0


def _absolute_elapsed_seconds(period: int, clock_remaining: float) -> float:
    if period <= 0:
        return 0.0
    if period <= 4:
        return ((period - 1) * 720.0) + max(0.0, 720.0 - float(clock_remaining))
    return (4 * 720.0) + ((period - 5) * 300.0) + max(0.0, 300.0 - float(clock_remaining))


def _sort_pbp_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    d = df.copy()
    period = pd.to_numeric(d.get("period"), errors="coerce").fillna(0)
    clock_remaining = d.get("clock", pd.Series(index=d.index, dtype=object)).map(_clock_to_seconds_remaining)
    d["_sort_period"] = period
    d["_sort_elapsed"] = [
        _absolute_elapsed_seconds(int(period_value), clock_value if clock_value is not None else _period_length_seconds(int(period_value)))
        for period_value, clock_value in zip(period, clock_remaining)
    ]
    d["_sort_action_id"] = pd.to_numeric(d.get("actionId"), errors="coerce")
    d["_sort_action_number"] = pd.to_numeric(d.get("actionNumber"), errors="coerce")
    d = d.sort_values(
        ["_sort_period", "_sort_elapsed", "_sort_action_id", "_sort_action_number"],
        kind="stable",
        na_position="last",
    )
    return d.drop(columns=["_sort_period", "_sort_elapsed", "_sort_action_id", "_sort_action_number"])


def _format_minutes(seconds: float) -> str:
    total_seconds = int(round(max(0.0, float(seconds))))
    return f"{total_seconds // 60}:{total_seconds % 60:02d}"


def _empty_stat_line() -> dict[str, Any]:
    return {
        "minutes": "0:00",
        "seconds": 0.0,
        "is_starter": False,
        "pts": 0,
        "fgm": 0,
        "fga": 0,
        "fg3m": 0,
        "fg3a": 0,
        "ftm": 0,
        "fta": 0,
        "reb": 0,
        "ast": 0,
        "tov": 0,
        "stl": 0,
        "blk": 0,
        "oreb": 0,
        "dreb": 0,
        "pf": 0,
        "plus_minus": 0,
    }


def _pbp_phase_from_game_type(game_type: str) -> str:
    game_type_norm = _normalize_text(game_type)
    if game_type_norm in {"playoffs", "play_in"}:
        return "playoffs"
    return "regular"


def _load_game_metadata(season: str, game_id: str) -> dict[str, Any]:
    logs_df = _load_data_csv(build_data_filename("team_game_logs", season), dtype={"game_id": "string"})
    logs_df["game_id_norm"] = logs_df["game_id"].map(_normalize_game_id)
    matches = logs_df[logs_df["game_id_norm"] == _normalize_game_id(game_id)].copy()
    if matches.empty:
        raise ValueError(f"Game {game_id} not found in team_game_logs for season {season}")

    row = matches.iloc[0]
    return {
        "season": season,
        "game_id": _normalize_game_id(game_id),
        "game_date": _safe_str(row.get("game_date")),
        "game_type": _safe_str(row.get("game_type")),
        "phase": _pbp_phase_from_game_type(_safe_str(row.get("game_type"))),
        "home_team_id": _to_int(row.get("team_id_home")),
        "home_team": _safe_str(row.get("team_abbreviation_home")),
        "road_team_id": _to_int(row.get("team_id_road")),
        "road_team": _safe_str(row.get("team_abbreviation_road")),
    }


def normalize_boxscore_segment(value: Optional[str]) -> str:
    text = str(value or "all").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "game": "all",
        "all": "all",
        "q1": "q1",
        "q2": "q2",
        "q3": "q3",
        "q4": "q4",
        "ot": "ot",
        "overtime": "ot",
        "h1": "h1",
        "first_half": "h1",
        "h2": "h2",
        "second_half": "h2",
        "no_garbage": "garbage_filtered",
        "non_garbage": "garbage_filtered",
        "garbage_filtered": "garbage_filtered",
        "garbage": "garbage_time",
        "garbage_time": "garbage_time",
        "clutch": "clutch",
    }
    normalized = aliases.get(text, text)
    if normalized not in VALID_BOX_SCORE_SEGMENTS:
        raise ValueError(
            f"Invalid box score segment: {value!r}. "
            f"Expected one of: {', '.join(sorted(VALID_BOX_SCORE_SEGMENTS))}"
        )
    return normalized


def _load_game_state_payload(meta: dict[str, Any]) -> Optional[dict[str, Any]]:
    parquet_path = (
        Path(PBP_GAME_STATES_ROOT)
        / meta["phase"]
        / meta["season"]
        / f"_states_{meta['season']}_{meta['phase']}.parquet"
    )
    if not parquet_path.exists():
        remote_path = _download_remote_pbpdata_file(
            f"game_states/{meta['phase']}/{meta['season']}/_states_{meta['season']}_{meta['phase']}.parquet"
        )
        if not remote_path or not remote_path.exists():
            return None
        parquet_path = remote_path

    try:
        table = pq.read_table(
            parquet_path,
            columns=["payload_json", HOME_WIN_PROB_BY_EVENT_JSON_COLUMN],
            filters=[("game_id", "==", meta["game_id"])],
            use_threads=False,
        )
    except Exception:
        return None

    if table.num_rows <= 0 or "payload_json" not in table.column_names:
        return None

    raw_payload = table.column("payload_json")[0].as_py()
    if not isinstance(raw_payload, str) or not raw_payload.strip():
        return None

    try:
        payload = json.loads(raw_payload)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    events = payload.get("events")
    if not isinstance(events, list):
        return payload

    if HOME_WIN_PROB_BY_EVENT_JSON_COLUMN in table.column_names:
        raw_series = table.column(HOME_WIN_PROB_BY_EVENT_JSON_COLUMN)[0].as_py()
        if isinstance(raw_series, str) and raw_series.strip():
            try:
                series = json.loads(raw_series)
            except Exception:
                series = None
            if isinstance(series, list):
                for idx, prob in enumerate(series[: len(events)]):
                    try:
                        if prob is None:
                            continue
                        if isinstance(events[idx], dict):
                            events[idx]["home_win_prob"] = float(min(1.0, max(0.0, float(prob))))
                    except Exception:
                        continue

    return payload


def _event_state_context(event: dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int], Optional[float], Optional[int]]:
    period = _to_int(event.get("period"), default=0) or None
    state = event.get("game_log_state")
    if not isinstance(state, dict):
        state = {}
    pts_home = _to_int(state.get("pts_home"), default=0) if state.get("pts_home") is not None else None
    pts_road = _to_int(state.get("pts_road"), default=0) if state.get("pts_road") is not None else None
    if pts_home is None and event.get("score_home") is not None:
        pts_home = _to_int(event.get("score_home"), default=0)
    if pts_road is None and event.get("score_away") is not None:
        pts_road = _to_int(event.get("score_away"), default=0)
    home_wp = None
    try:
        if event.get("home_win_prob") is not None:
            home_wp = float(event.get("home_win_prob"))
    except Exception:
        home_wp = None
    seconds_left = _clock_to_seconds_remaining(event.get("clock"))
    return period, pts_home, pts_road, home_wp, int(seconds_left) if seconds_left is not None else None


def _segment_scope_flags(period: Optional[int]) -> dict[str, bool]:
    if period is None or period <= 0:
        return {
            "q1": False,
            "q2": False,
            "q3": False,
            "q4": False,
            "ot": False,
            "h1": False,
            "h2": False,
        }
    return {
        "q1": period == 1,
        "q2": period == 2,
        "q3": period == 3,
        "q4": period == 4,
        "ot": period > 4,
        "h1": period in {1, 2},
        "h2": period >= 3,
    }


def _is_clutch_event(event: dict[str, Any]) -> bool:
    period, pts_home, pts_road, _home_wp, seconds_left = _event_state_context(event)
    return bool(
        period is not None
        and pts_home is not None
        and pts_road is not None
        and period >= 4
        and seconds_left is not None
        and seconds_left < 300
        and abs(pts_home - pts_road) <= 5
    )


def _should_enter_garbage(event: dict[str, Any], garbage_wp_on: float = DEFAULT_GARBAGE_WP_ON) -> bool:
    period, pts_home, pts_road, home_wp, _seconds_left = _event_state_context(event)
    return bool(
        period is not None
        and pts_home is not None
        and pts_road is not None
        and period >= 3
        and abs(pts_home - pts_road) > 5
        and home_wp is not None
        and (home_wp >= garbage_wp_on or home_wp <= (1.0 - garbage_wp_on))
    )


def _should_exit_garbage(event: dict[str, Any], garbage_wp_off: float = DEFAULT_GARBAGE_WP_OFF) -> bool:
    _period, _pts_home, _pts_road, home_wp, _seconds_left = _event_state_context(event)
    return bool(home_wp is not None and (1.0 - garbage_wp_off) < home_wp < garbage_wp_off)


def _is_final_game_minute(event: dict[str, Any], max_period: int) -> bool:
    period, _pts_home, _pts_road, _home_wp, seconds_left = _event_state_context(event)
    return bool(period is not None and period == max_period and seconds_left is not None and seconds_left < 60)


def _build_segment_include_map(meta: dict[str, Any]) -> Optional[dict[tuple[int, int], dict[str, bool]]]:
    payload = _load_game_state_payload(meta)
    if not isinstance(payload, dict):
        return None
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return None

    events = [event for event in raw_events if isinstance(event, dict) and not bool(event.get("event_quarantined"))]
    if not events:
        return None

    max_period = 4
    for event in events:
        period = _to_int(event.get("period"), default=0)
        if period > max_period:
            max_period = period

    include_map: dict[tuple[int, int], dict[str, bool]] = {}
    prev_is_garbage = False
    prev_is_clutch = False
    for event in events:
        period = _to_int(event.get("period"), default=0) or None
        is_garbage = (
            not prev_is_garbage
            and not _is_final_game_minute(event, max_period)
            and _should_enter_garbage(event)
        )
        if prev_is_garbage:
            is_garbage = not _should_exit_garbage(event)
        is_clutch = _is_clutch_event(event)

        flags = {
            "all": True,
            "garbage_filtered": not prev_is_garbage,
            "garbage_time": prev_is_garbage,
            "clutch": prev_is_clutch,
            **_segment_scope_flags(period),
        }
        key = (_to_int(event.get("action_number")), _to_int(event.get("action_id")))
        include_map[key] = flags
        prev_is_garbage = is_garbage
        prev_is_clutch = is_clutch

    return include_map


def _raw_row_segment_match(row: pd.Series, segment: str, include_map: Optional[dict[tuple[int, int], dict[str, bool]]]) -> bool:
    if segment == "all":
        return True

    key = (_to_int(row.get("actionNumber")), _to_int(row.get("actionId")))
    if include_map and key in include_map:
        return bool(include_map[key].get(segment, False))

    period = _to_int(row.get("period"), default=0)
    if segment in {"q1", "q2", "q3", "q4", "ot", "h1", "h2"}:
        return _segment_scope_flags(period).get(segment, False)
    if segment == "clutch":
        score_home = pd.to_numeric(row.get("scoreHome"), errors="coerce")
        score_road = pd.to_numeric(row.get("scoreAway"), errors="coerce")
        seconds_left = _clock_to_seconds_remaining(row.get("clock"))
        return bool(
            pd.notna(score_home)
            and pd.notna(score_road)
            and period >= 4
            and seconds_left is not None
            and seconds_left < 300
            and abs(int(score_home) - int(score_road)) <= 5
        )
    return False


def _token_player_id(token: str) -> Optional[int]:
    parts = token.split(":")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _name_aliases(value: Any) -> set[str]:
    cleaned = _clean_name(value)
    if not cleaned:
        return set()

    aliases = {cleaned}
    parts = [part for part in cleaned.split() if part]
    if len(parts) >= 2:
        aliases.add(parts[-1])
        if len(parts[0]) == 1:
            aliases.add(f"{parts[0]} {parts[-1]}")
    return {alias for alias in aliases if alias}


def _build_name_registry(
    pbp_df: pd.DataFrame,
    team_ids: list[int],
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, set[str]]]]:
    display_by_token: dict[int, dict[str, str]] = {team_id: {} for team_id in team_ids}
    tokens_by_name: dict[int, dict[str, set[str]]] = {team_id: defaultdict(set) for team_id in team_ids}

    for _, row in pbp_df.iterrows():
        team_id = _to_int(row.get("teamId"))
        if team_id not in team_ids:
            continue
        player_id = _to_int(row.get("personId"))
        player_name = _safe_str(row.get("playerName"))
        player_name_i = _safe_str(row.get("playerNameI"))
        if player_id <= 0 and not player_name and not player_name_i:
            continue
        if player_id > 0:
            token = f"{team_id}:{player_id}"
        else:
            token = f"{team_id}:name:{_clean_name(player_name_i or player_name)}"
        display = player_name_i or player_name
        if display:
            display_by_token[team_id].setdefault(token, display)
        for candidate in (player_name, player_name_i, display):
            for alias in _name_aliases(candidate):
                tokens_by_name[team_id][alias].add(token)

    return display_by_token, tokens_by_name


def _resolve_token(
    team_id: int,
    raw_name: str,
    display_by_token: dict[int, dict[str, str]],
    tokens_by_name: dict[int, dict[str, set[str]]],
    fallback_prefix: str,
) -> Optional[str]:
    cleaned = _clean_name(raw_name)
    if not cleaned:
        return None
    matches = sorted(tokens_by_name[team_id].get(cleaned, set()))
    player_matches = [token for token in matches if _token_player_id(token)]
    if len(player_matches) == 1:
        token = player_matches[0]
        if raw_name:
            display_by_token[team_id].setdefault(token, raw_name)
        return token
    if len(matches) == 1:
        token = matches[0]
        if raw_name:
            display_by_token[team_id].setdefault(token, raw_name)
        return token
    token = f"{team_id}:{fallback_prefix}:{cleaned}"
    if raw_name:
        display_by_token[team_id].setdefault(token, raw_name)
    tokens_by_name[team_id][cleaned].add(token)
    return token


def _infer_period_start_lineup(
    period_df: pd.DataFrame,
    team_id: int,
    display_by_token: dict[int, dict[str, str]],
    tokens_by_name: dict[int, dict[str, set[str]]],
) -> set[str]:
    constraints: list[tuple[str, str, Optional[str]]] = []
    players: set[str] = set()

    for _, row in period_df.iterrows():
        row_team_id = _to_int(row.get("teamId"))
        action_type = _normalize_text(row.get("actionType"))
        if row_team_id != team_id:
            continue

        description = _safe_str(row.get("description")).strip()
        if action_type == "substitution":
            match = SUB_RE.match(description)
            if not match:
                continue
            player_out_id = _to_int(row.get("personId"))
            player_out = (
                f"{team_id}:{player_out_id}"
                if player_out_id > 0
                else _resolve_token(
                    team_id,
                    match.group(2),
                    display_by_token,
                    tokens_by_name,
                    fallback_prefix="subout",
                )
            )
            player_in = _resolve_token(
                team_id,
                match.group(1),
                display_by_token,
                tokens_by_name,
                fallback_prefix="subin",
            )
            if player_in and player_out:
                constraints.append(("sub", player_out, player_in))
                players.add(player_out)
                players.add(player_in)
            continue

        player_id = _to_int(row.get("personId"))
        player_name = _safe_str(row.get("playerNameI")) or _safe_str(row.get("playerName"))
        if player_id > 0:
            token = f"{team_id}:{player_id}"
        else:
            token = _resolve_token(
                team_id,
                player_name,
                display_by_token,
                tokens_by_name,
                fallback_prefix="event",
            )
        if token and action_type not in {"timeout", "period", "game"}:
            constraints.append(("on", token, None))
            players.add(token)

    player_pool = sorted(players)
    if len(player_pool) < 5:
        raise ValueError(f"Unable to infer starting lineup for team_id={team_id}, period={period_df['period'].iloc[0]}")

    valid_lineups: list[set[str]] = []
    for combo in combinations(player_pool, 5):
        lineup = set(combo)
        is_valid = True
        for kind, first, second in constraints:
            if kind == "on":
                if first not in lineup:
                    is_valid = False
                    break
            else:
                if first not in lineup or second is None or second in lineup:
                    is_valid = False
                    break
                lineup.remove(first)
                lineup.add(second)
        if is_valid:
            valid_lineups.append(set(combo))

    if not valid_lineups:
        raise ValueError(f"No valid lineup solution for team_id={team_id}, period={period_df['period'].iloc[0]}")

    seen_before_first_sub: set[str] = set()
    for kind, first, _ in constraints:
        if kind == "sub":
            break
        seen_before_first_sub.add(first)

    valid_lineups.sort(key=lambda lineup: (-len(lineup & seen_before_first_sub), sorted(lineup)))
    return valid_lineups[0]


def _resolve_assist_tokens(
    team_id: int,
    description: str,
    display_by_token: dict[int, dict[str, str]],
    tokens_by_name: dict[int, dict[str, set[str]]],
) -> list[str]:
    resolved: list[str] = []
    for raw_alias in ASSIST_RE.findall(description or ""):
        token = _resolve_token(
            team_id,
            raw_alias,
            display_by_token,
            tokens_by_name,
            fallback_prefix="assist",
        )
        if token:
            resolved.append(token)
    return resolved


def _fetch_game_rotation(game_id: str) -> dict[int, list[dict[str, Any]]]:
    from nba_api.stats.endpoints import gamerotation

    last_error: Optional[Exception] = None
    for _ in range(ROTATION_RETRIES):
        try:
            response = gamerotation.GameRotation(game_id=game_id, timeout=ROTATION_TIMEOUT_SECONDS)
            away_df = response.away_team.get_data_frame()
            home_df = response.home_team.get_data_frame()
            frames = [away_df, home_df]
            rows_by_team: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for frame in frames:
                if frame is None or frame.empty:
                    continue
                for _, row in frame.iterrows():
                    team_id = _to_int(row.get("TEAM_ID"))
                    person_id = _to_int(row.get("PERSON_ID"))
                    if team_id <= 0 or person_id <= 0:
                        continue
                    rows_by_team[team_id].append(
                        {
                            "team_id": team_id,
                            "player_id": person_id,
                            "player_name": f"{_safe_str(row.get('PLAYER_FIRST'))} {_safe_str(row.get('PLAYER_LAST'))}".strip(),
                            "start_seconds": max(0.0, pd.to_numeric(row.get("IN_TIME_REAL"), errors="coerce") / 10.0),
                            "end_seconds": max(0.0, pd.to_numeric(row.get("OUT_TIME_REAL"), errors="coerce") / 10.0),
                            "seconds": max(
                                0.0,
                                (pd.to_numeric(row.get("OUT_TIME_REAL"), errors="coerce") - pd.to_numeric(row.get("IN_TIME_REAL"), errors="coerce")) / 10.0,
                            ),
                            "plus_minus": _to_int(row.get("PT_DIFF")),
                        }
                    )
            if rows_by_team:
                return rows_by_team
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"GameRotation returned no data for game {game_id}")


def _build_empty_starter_info(team_ids: list[int]) -> dict[int, dict[str, set[Any]]]:
    return {
        team_id: {
            "tokens": set(),
            "player_ids": set(),
            "names": set(),
        }
        for team_id in team_ids
    }


def _add_starter_identity(
    starter_info: dict[int, dict[str, set[Any]]],
    team_id: int,
    *,
    token: Optional[str] = None,
    player_id: Optional[int] = None,
    player_name: Optional[str] = None,
) -> None:
    info = starter_info.setdefault(team_id, {"tokens": set(), "player_ids": set(), "names": set()})
    if token:
        info["tokens"].add(token)
        token_player_id = _token_player_id(token)
        if token_player_id:
            info["player_ids"].add(token_player_id)
    if player_id:
        info["player_ids"].add(int(player_id))
    cleaned_name = _clean_name(player_name)
    if cleaned_name:
        info["names"].add(cleaned_name)


def _identify_game_starters(
    *,
    game_df: pd.DataFrame,
    team_ids: list[int],
    display_by_token: dict[int, dict[str, str]],
    tokens_by_name: dict[int, dict[str, set[str]]],
    rotation_rows_by_team: dict[int, list[dict[str, Any]]],
) -> dict[int, dict[str, set[Any]]]:
    starter_info = _build_empty_starter_info(team_ids)

    for team_id in team_ids:
        for rotation_row in rotation_rows_by_team.get(team_id, []):
            if float(rotation_row.get("start_seconds", 0.0)) > 0.5:
                continue
            token = f"{team_id}:{rotation_row['player_id']}"
            display_name = display_by_token[team_id].get(token, rotation_row.get("player_name"))
            _add_starter_identity(
                starter_info,
                team_id,
                token=token,
                player_id=_to_int(rotation_row.get("player_id"), default=0) or None,
                player_name=display_name,
            )

    period_one_df = _sort_pbp_events(game_df[game_df["period"] == 1])
    if period_one_df.empty:
        return starter_info

    for team_id in team_ids:
        if len(starter_info[team_id]["tokens"]) >= 5 or len(starter_info[team_id]["player_ids"]) >= 5:
            continue
        try:
            lineup = _infer_period_start_lineup(
                period_df=period_one_df,
                team_id=team_id,
                display_by_token=display_by_token,
                tokens_by_name=tokens_by_name,
            )
        except ValueError:
            continue
        for token in lineup:
            _add_starter_identity(
                starter_info,
                team_id,
                token=token,
                player_name=display_by_token[team_id].get(token, token),
            )

    return starter_info


def _apply_official_starter_info(
    *,
    season: str,
    game_id: str,
    team_ids: list[int],
    starter_info: dict[int, dict[str, set[Any]]],
) -> dict[int, dict[str, set[Any]]]:
    try:
        players_df = _load_data_csv(
            build_box_score_traditional_filename("players", season),
            dtype={"game_id": "string"},
        )
    except Exception:
        return starter_info

    if "game_id" not in players_df.columns:
        return starter_info

    players_df["game_id_norm"] = players_df["game_id"].map(_normalize_game_id)
    game_players = players_df[players_df["game_id_norm"] == _normalize_game_id(game_id)].copy()
    if game_players.empty:
        return starter_info

    for _, row in game_players.iterrows():
        team_id = _to_int(row.get("team_id"))
        if team_id not in team_ids:
            continue
        if not _safe_str(row.get("position")).strip():
            continue

        player_id = _to_int(row.get("person_id"), default=0) or None
        first_name = _safe_str(row.get("first_name"))
        family_name = _safe_str(row.get("family_name"))
        player_name = _safe_str(row.get("name_i")) or f"{first_name} {family_name}".strip()
        _add_starter_identity(
            starter_info,
            team_id,
            token=f"{team_id}:{player_id}" if player_id else None,
            player_id=player_id,
            player_name=player_name,
        )

    return starter_info


def _is_starter_row(
    *,
    team_id: int,
    player_id: Optional[int],
    player_name: str,
    token: Optional[str],
    starter_info: Optional[dict[int, dict[str, set[Any]]]],
) -> bool:
    if not starter_info:
        return False
    info = starter_info.get(team_id)
    if not info:
        return False
    if token and token in info["tokens"]:
        return True
    if player_id and player_id in info["player_ids"]:
        return True
    return _clean_name(player_name) in info["names"]


def _ensure_player_entry(
    stats_by_team: dict[int, dict[str, dict[str, Any]]],
    meta: dict[str, Any],
    team_id: int,
    token: str,
    player_name: str,
) -> dict[str, Any]:
    team_stats = stats_by_team[team_id]
    team_stats.setdefault(
        token,
        {
            "player_id": _token_player_id(token),
            "player_name": player_name,
            "team_id": team_id,
            "team_abbreviation": meta["home_team"] if team_id == meta["home_team_id"] else meta["road_team"],
            **_empty_stat_line(),
        },
    )
    return team_stats[token]


def _apply_rotation_segment_minutes_and_plus_minus(
    *,
    game_df: pd.DataFrame,
    meta: dict[str, Any],
    team_ids: list[int],
    segment: str,
    include_map: Optional[dict[tuple[int, int], dict[str, bool]]],
    rotation_rows_by_team: dict[int, list[dict[str, Any]]],
    stats_by_team: dict[int, dict[str, dict[str, Any]]],
    display_by_token: dict[int, dict[str, str]],
) -> None:
    score_home = 0
    score_road = 0

    for period in sorted(int(value) for value in game_df["period"].dropna().unique() if int(value) > 0):
        period_df = _sort_pbp_events(game_df[game_df["period"] == period])
        prev_clock = _period_length_seconds(period)

        for _, row in period_df.iterrows():
            current_clock = _clock_to_seconds_remaining(row.get("clock"))
            current_clock = prev_clock if current_clock is None else current_clock
            include_segment = _raw_row_segment_match(row, segment, include_map)

            start_elapsed = _absolute_elapsed_seconds(period, prev_clock)
            end_elapsed = _absolute_elapsed_seconds(period, current_clock)
            if include_segment and end_elapsed >= start_elapsed:
                for team_id in team_ids:
                    for rotation_row in rotation_rows_by_team.get(team_id, []):
                        overlap = max(
                            0.0,
                            min(end_elapsed, float(rotation_row["end_seconds"])) - max(start_elapsed, float(rotation_row["start_seconds"])),
                        )
                        if overlap <= 0.0:
                            continue
                        token = f"{team_id}:{rotation_row['player_id']}"
                        display_name = display_by_token[team_id].get(token, rotation_row["player_name"])
                        display_by_token[team_id].setdefault(token, display_name)
                        stat_line = _ensure_player_entry(
                            stats_by_team=stats_by_team,
                            meta=meta,
                            team_id=team_id,
                            token=token,
                            player_name=display_name,
                        )
                        stat_line["seconds"] += overlap

            new_score_home = pd.to_numeric(row.get("scoreHome"), errors="coerce")
            new_score_road = pd.to_numeric(row.get("scoreAway"), errors="coerce")
            if include_segment and pd.notna(new_score_home) and pd.notna(new_score_road):
                home_delta = int(new_score_home) - score_home
                road_delta = int(new_score_road) - score_road
                if home_delta or road_delta:
                    plus_minus_delta = home_delta - road_delta
                    event_elapsed = _absolute_elapsed_seconds(period, current_clock)
                    for lineup_team_id in team_ids:
                        sign = 1 if lineup_team_id == meta["home_team_id"] else -1
                        for rotation_row in rotation_rows_by_team.get(lineup_team_id, []):
                            if not (float(rotation_row["start_seconds"]) <= event_elapsed < float(rotation_row["end_seconds"])):
                                continue
                            token = f"{lineup_team_id}:{rotation_row['player_id']}"
                            display_name = display_by_token[lineup_team_id].get(token, rotation_row["player_name"])
                            display_by_token[lineup_team_id].setdefault(token, display_name)
                            stat_line = _ensure_player_entry(
                                stats_by_team=stats_by_team,
                                meta=meta,
                                team_id=lineup_team_id,
                                token=token,
                                player_name=display_name,
                            )
                            stat_line["plus_minus"] += plus_minus_delta * sign
                score_home = int(new_score_home)
                score_road = int(new_score_road)
            elif pd.notna(new_score_home) and pd.notna(new_score_road):
                score_home = int(new_score_home)
                score_road = int(new_score_road)

            prev_clock = current_clock


def _load_traditional_boxscore_fallback(
    season: str,
    game_id: str,
    meta: dict[str, Any],
    starter_info: Optional[dict[int, dict[str, set[Any]]]] = None,
) -> dict[str, Any]:
    players_df = _load_data_csv(
        build_box_score_traditional_filename("players", season),
        dtype={"game_id": "string"},
    )
    players_df["game_id_norm"] = players_df["game_id"].map(_normalize_game_id)
    game_players = players_df[players_df["game_id_norm"] == _normalize_game_id(game_id)].copy()
    if game_players.empty:
        raise ValueError(f"Game {game_id} not found in traditional box score data for season {season}")

    def _serialize_side(team_id: int) -> list[dict[str, Any]]:
        team_rows = game_players[game_players["team_id"].map(_to_int) == team_id].copy()
        if team_rows.empty:
            return []

        team_rows["_minutes_sort"] = team_rows["minutes"].map(
            lambda value: sum(
                part * multiplier for part, multiplier in zip(
                    [_to_int(piece) for piece in str(value).split(":")[:2]],
                    [60, 1],
                )
            ) if isinstance(value, str) and ":" in value else None
        )
        team_rows = team_rows.sort_values(
            by=["_minutes_sort", "family_name", "first_name"],
            ascending=[False, True, True],
            kind="stable",
            na_position="last",
        )

        rows: list[dict[str, Any]] = []
        for _, row in team_rows.iterrows():
            first_name = _safe_str(row.get("first_name"))
            family_name = _safe_str(row.get("family_name"))
            full_name = f"{first_name} {family_name}".strip() or _safe_str(row.get("name_i")) or "Unknown"
            player_id = _to_int(row.get("person_id"), default=0) or None
            csv_starter_flag = any(
                bool(_safe_str(row.get(column)).strip())
                for column in ("position", "start_position", "startPosition", "starting_position")
            )
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": full_name,
                    "is_starter": _is_starter_row(
                        team_id=team_id,
                        player_id=player_id,
                        player_name=full_name,
                        token=f"{team_id}:{player_id}" if player_id else None,
                        starter_info=starter_info,
                    ) or csv_starter_flag,
                    "team_id": team_id,
                    "team_abbreviation": _safe_str(row.get("team_tricode")),
                    "minutes": _safe_str(row.get("minutes")) or "0:00",
                    "pts": _to_int(row.get("points")),
                    "fgm": _to_int(row.get("field_goals_made")),
                    "fga": _to_int(row.get("field_goals_attempted")),
                    "fg3m": _to_int(row.get("three_pointers_made")),
                    "fg3a": _to_int(row.get("three_pointers_attempted")),
                    "ftm": _to_int(row.get("free_throws_made")),
                    "fta": _to_int(row.get("free_throws_attempted")),
                    "reb": _to_int(row.get("rebounds_total")),
                    "ast": _to_int(row.get("assists")),
                    "tov": _to_int(row.get("turnovers")),
                    "stl": _to_int(row.get("steals")),
                    "blk": _to_int(row.get("blocks")),
                    "oreb": _to_int(row.get("rebounds_offensive")),
                    "dreb": _to_int(row.get("rebounds_defensive")),
                    "pf": _to_int(row.get("fouls_personal")),
                    "plus_minus": _to_int(row.get("plus_minus_points")),
                }
            )
        return rows

    return {
        "season": meta["season"],
        "phase": meta["phase"],
        "game_id": meta["game_id"],
        "game_date": meta["game_date"] or None,
        "game_type": meta["game_type"] or None,
        "home_team": meta["home_team"],
        "road_team": meta["road_team"],
        "source": "box_score_traditional_v3_fallback",
        "minutes_plus_minus_source": "box_score_traditional_v3",
        "home_players": _serialize_side(meta["home_team_id"]),
        "road_players": _serialize_side(meta["road_team_id"]),
    }


def compute_pbp_traditional_boxscore(
    season: str,
    game_id: str,
    segment: str = "all",
) -> dict[str, Any]:
    segment = normalize_boxscore_segment(segment)
    meta = _load_game_metadata(season=season, game_id=game_id)
    if segment == "all":
        return _load_traditional_boxscore_fallback(
            season=season,
            game_id=game_id,
            meta=meta,
            starter_info=None,
        )

    team_ids = [meta["home_team_id"], meta["road_team_id"]]

    data_repo_dir = Path(resolve_data_file_path(build_data_filename("team_game_logs", season)).parents[1])
    pbp_path, pbp_source = _resolve_pbp_input_path(data_repo_dir, season, meta["phase"])
    try:
        pbp_df = _load_pbp_df(pbp_path)
    except Exception:
        if segment == "all":
            return _load_traditional_boxscore_fallback(
                season=season,
                game_id=game_id,
                meta=meta,
                starter_info=None,
            )
        raise
    game_id_norm = _normalize_game_id(game_id)
    game_df = pbp_df[pbp_df["game_id_norm"] == game_id_norm].copy()
    if game_df.empty:
        if segment == "all":
            return _load_traditional_boxscore_fallback(
                season=season,
                game_id=game_id,
                meta=meta,
                starter_info=None,
            )
        raise ValueError(f"No PBP rows found for game {game_id_norm}")

    game_df = _sort_pbp_events(game_df).reset_index(drop=True)
    display_by_token, tokens_by_name = _build_name_registry(game_df, team_ids=team_ids)

    for _, row in game_df.iterrows():
        if _normalize_text(row.get("actionType")) != "substitution":
            continue
        team_id = _to_int(row.get("teamId"))
        if team_id not in team_ids:
            continue
        match = SUB_RE.match(_safe_str(row.get("description")).strip())
        if not match:
            continue
        _resolve_token(team_id, match.group(1), display_by_token, tokens_by_name, fallback_prefix="subin")
        _resolve_token(team_id, match.group(2), display_by_token, tokens_by_name, fallback_prefix="subout")

    stats_by_team: dict[int, dict[str, dict[str, Any]]] = {team_id: {} for team_id in team_ids}
    rebound_prev: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: {"off": 0, "def": 0})
    offensive_foul_turnover_actions: dict[str, list[int]] = defaultdict(list)
    include_map = _build_segment_include_map(meta) if segment != "all" else None
    minutes_plus_minus_source = "pbp_inferred" if segment == "all" else f"pbp_segmented:{segment}"

    rotation_rows_by_team: dict[int, list[dict[str, Any]]] = {}

    if segment == "all":
        try:
            rotation_rows_by_team = _fetch_game_rotation(game_id_norm)
            minutes_plus_minus_source = "gamerotation"
            for team_id, rotation_rows in rotation_rows_by_team.items():
                if team_id not in team_ids:
                    continue
                for rotation_row in rotation_rows:
                    token = f"{team_id}:{rotation_row['player_id']}"
                    display_name = display_by_token[team_id].get(token, rotation_row["player_name"])
                    display_by_token[team_id].setdefault(token, display_name)
                    stat_line = _ensure_player_entry(
                        stats_by_team=stats_by_team,
                        meta=meta,
                        team_id=team_id,
                        token=token,
                        player_name=display_name,
                    )
                    stat_line["seconds"] += float(rotation_row["seconds"])
                    stat_line["plus_minus"] += int(rotation_row["plus_minus"])
        except Exception:
            rotation_rows_by_team = {}
    else:
        try:
            rotation_rows_by_team = _fetch_game_rotation(game_id_norm)
            _apply_rotation_segment_minutes_and_plus_minus(
                game_df=game_df,
                meta=meta,
                team_ids=team_ids,
                segment=segment,
                include_map=include_map,
                rotation_rows_by_team=rotation_rows_by_team,
                stats_by_team=stats_by_team,
                display_by_token=display_by_token,
            )
            minutes_plus_minus_source = f"gamerotation_segmented:{segment}"
        except Exception:
            rotation_rows_by_team = {}

    starter_info = _identify_game_starters(
        game_df=game_df,
        team_ids=team_ids,
        display_by_token=display_by_token,
        tokens_by_name=tokens_by_name,
        rotation_rows_by_team=rotation_rows_by_team,
    )
    starter_info = _apply_official_starter_info(
        season=season,
        game_id=game_id,
        team_ids=team_ids,
        starter_info=starter_info,
    )

    for _, row in game_df.iterrows():
        team_id = _to_int(row.get("teamId"))
        if team_id not in team_ids:
            continue
        player_id = _to_int(row.get("personId"))
        if player_id <= 0:
            continue
        if _normalize_text(row.get("actionType")) == "turnover" and "offensive foul" in _normalize_text(row.get("subType")):
            offensive_foul_turnover_actions[f"{team_id}:{player_id}"].append(_to_int(row.get("actionNumber")))
    if minutes_plus_minus_source.startswith("pbp"):
        try:
            score_home = 0
            score_road = 0
            periods = sorted(int(value) for value in game_df["period"].dropna().unique() if int(value) > 0)

            for period in periods:
                period_df = _sort_pbp_events(game_df[game_df["period"] == period])
                current_lineups = {
                    team_id: _infer_period_start_lineup(
                        period_df=period_df,
                        team_id=team_id,
                        display_by_token=display_by_token,
                        tokens_by_name=tokens_by_name,
                    )
                    for team_id in team_ids
                }

                for team_id, lineup in current_lineups.items():
                    for token in lineup:
                        _ensure_player_entry(
                            stats_by_team=stats_by_team,
                            meta=meta,
                            team_id=team_id,
                            token=token,
                            player_name=display_by_token[team_id].get(token, token),
                        )

                prev_clock = 720.0 if period <= 4 else 300.0

                for _, row in period_df.iterrows():
                    clock_seconds = _clock_to_seconds_remaining(row.get("clock"))
                    current_clock = prev_clock if clock_seconds is None else clock_seconds
                    elapsed = max(0.0, prev_clock - current_clock)
                    include_segment = _raw_row_segment_match(row, segment, include_map)
                    if include_segment and elapsed > 0.0:
                        for team_id, lineup in current_lineups.items():
                            for token in lineup:
                                stats_by_team[team_id][token]["seconds"] += elapsed
                    prev_clock = current_clock

                    team_id = _to_int(row.get("teamId"))
                    action_type = _normalize_text(row.get("actionType"))
                    description = _safe_str(row.get("description"))
                    player_id = _to_int(row.get("personId"))
                    token = f"{team_id}:{player_id}" if team_id in team_ids and player_id > 0 else None

                    new_score_home = pd.to_numeric(row.get("scoreHome"), errors="coerce")
                    new_score_road = pd.to_numeric(row.get("scoreAway"), errors="coerce")
                    if pd.notna(new_score_home) and pd.notna(new_score_road):
                        home_delta = int(new_score_home) - score_home
                        road_delta = int(new_score_road) - score_road
                        if (home_delta or road_delta) and include_segment:
                            plus_minus_delta = home_delta - road_delta
                            for token_in_lineup in current_lineups[meta["home_team_id"]]:
                                stats_by_team[meta["home_team_id"]][token_in_lineup]["plus_minus"] += plus_minus_delta
                            for token_in_lineup in current_lineups[meta["road_team_id"]]:
                                stats_by_team[meta["road_team_id"]][token_in_lineup]["plus_minus"] -= plus_minus_delta
                            score_home = int(new_score_home)
                            score_road = int(new_score_road)

                    if action_type == "substitution" and team_id in team_ids:
                        match = SUB_RE.match(description.strip())
                        if match:
                            player_out = token or _resolve_token(
                                team_id,
                                match.group(2),
                                display_by_token,
                                tokens_by_name,
                                fallback_prefix="subout",
                            )
                            player_in = _resolve_token(
                                team_id,
                                match.group(1),
                                display_by_token,
                                tokens_by_name,
                                fallback_prefix="subin",
                            )
                            if player_out in current_lineups[team_id]:
                                current_lineups[team_id].remove(player_out)
                            if player_in:
                                current_lineups[team_id].add(player_in)
                                _ensure_player_entry(
                                    stats_by_team=stats_by_team,
                                    meta=meta,
                                    team_id=team_id,
                                    token=player_in,
                                    player_name=display_by_token[team_id].get(player_in, player_in),
                                )

                remaining = max(0.0, prev_clock)
                period_end_row = period_df.iloc[-1] if not period_df.empty else None
                include_remaining = bool(period_end_row is not None and _raw_row_segment_match(period_end_row, segment, include_map))
                if include_remaining and remaining > 0.0:
                    for team_id, lineup in current_lineups.items():
                        for token in lineup:
                            stats_by_team[team_id][token]["seconds"] += remaining
        except ValueError:
            if segment == "all":
                return _load_traditional_boxscore_fallback(
                    season=season,
                    game_id=game_id,
                    meta=meta,
                    starter_info=starter_info,
                )
            minutes_plus_minus_source = f"pbp_stats_only:{segment}"

    for _, row in game_df.iterrows():
        team_id = _to_int(row.get("teamId"))
        action_type = _normalize_text(row.get("actionType"))
        sub_type = _normalize_text(row.get("subType"))
        description = _safe_str(row.get("description"))
        description_lower = description.lower()
        description_upper = description.upper()

        player_id = _to_int(row.get("personId"))
        token = f"{team_id}:{player_id}" if team_id in team_ids and player_id > 0 else None
        rebound_off_delta: Optional[int] = None
        rebound_def_delta: Optional[int] = None
        if token and team_id in team_ids and action_type == "rebound":
            reb_match = OFF_DEF_REB_RE.search(description)
            if reb_match:
                off_total = int(reb_match.group(1))
                def_total = int(reb_match.group(2))
                prev = rebound_prev[(team_id, token)]
                rebound_off_delta = max(0, off_total - prev["off"])
                rebound_def_delta = max(0, def_total - prev["def"])
                rebound_prev[(team_id, token)] = {
                    "off": max(prev["off"], off_total),
                    "def": max(prev["def"], def_total),
                }

        if not _raw_row_segment_match(row, segment, include_map):
            continue
        if token and team_id in team_ids:
            stat_line = _ensure_player_entry(
                stats_by_team=stats_by_team,
                meta=meta,
                team_id=team_id,
                token=token,
                player_name=display_by_token[team_id].get(token, token),
            )
            shot_result = _normalize_text(row.get("shotResult"))
            shot_value = _to_int(row.get("shotValue"))
            action_number = _to_int(row.get("actionNumber"))

            if action_type in {"2pt", "3pt", "made shot", "missed shot", "heave"}:
                is_three = action_type == "3pt" or shot_value == 3 or "3pt" in description_lower
                stat_line["fga"] += 1
                if is_three:
                    stat_line["fg3a"] += 1
                made_shot = action_type == "made shot" or shot_result == "made"
                if made_shot:
                    stat_line["fgm"] += 1
                    stat_line["pts"] += 3 if is_three else 2
                    if is_three:
                        stat_line["fg3m"] += 1
                    for assist_token in _resolve_assist_tokens(team_id, description, display_by_token, tokens_by_name):
                        assist_line = _ensure_player_entry(
                            stats_by_team=stats_by_team,
                            meta=meta,
                            team_id=team_id,
                            token=assist_token,
                            player_name=display_by_token[team_id].get(assist_token, assist_token),
                        )
                        assist_line["ast"] += 1

            elif action_type in {"freethrow", "free throw"}:
                stat_line["fta"] += 1
                made_ft = shot_result == "made" or ("free throw" in description_lower and "miss" not in description_lower)
                if made_ft:
                    stat_line["ftm"] += 1
                    stat_line["pts"] += 1

            elif action_type == "rebound":
                if sub_type == "offensive":
                    stat_line["oreb"] += 1
                    stat_line["reb"] += 1
                elif sub_type == "defensive":
                    stat_line["dreb"] += 1
                    stat_line["reb"] += 1
                else:
                    if rebound_off_delta is not None or rebound_def_delta is not None:
                        if rebound_off_delta:
                            stat_line["oreb"] += rebound_off_delta
                            stat_line["reb"] += rebound_off_delta
                        if rebound_def_delta:
                            stat_line["dreb"] += rebound_def_delta
                            stat_line["reb"] += rebound_def_delta
                    elif "offensive rebound" in description_lower:
                        stat_line["oreb"] += 1
                        stat_line["reb"] += 1
                    elif "defensive rebound" in description_lower:
                        stat_line["dreb"] += 1
                        stat_line["reb"] += 1

            elif action_type == "turnover":
                stat_line["tov"] += 1

            elif action_type == "foul" and _counts_toward_pf(sub_type, description_lower):
                stat_line["pf"] += 1
                offensive_foul_like = "offensive" in sub_type or "charge" in sub_type or "off.foul" in description_lower
                if offensive_foul_like:
                    nearby_turnover = any(
                        abs(action_number - candidate) <= 3
                        for candidate in offensive_foul_turnover_actions.get(token, [])
                    )
                    if not nearby_turnover:
                        stat_line["tov"] += 1

            elif action_type == "steal" or (not action_type and "STEAL" in description_upper):
                stat_line["stl"] += 1

            elif action_type == "block" or (not action_type and "BLOCK" in description_upper):
                stat_line["blk"] += 1

    def _serialize_team(team_id: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for token, stat_line in stats_by_team[team_id].items():
            if stat_line["seconds"] <= 0 and all(stat_line[key] == 0 for key in SUPPORTED_STATS):
                continue
            row = {
                key: stat_line[key]
                for key in ["player_id", "player_name", "team_id", "team_abbreviation", *SUPPORTED_STATS]
            }
            row["is_starter"] = _is_starter_row(
                team_id=team_id,
                player_id=stat_line["player_id"],
                player_name=stat_line["player_name"],
                token=token,
                starter_info=starter_info,
            )
            row["minutes"] = _format_minutes(stat_line["seconds"])
            row["_seconds_sort"] = float(stat_line["seconds"])
            rows.append(row)
        rows.sort(key=lambda item: (-float(item["_seconds_sort"]), item["player_name"]))
        for row in rows:
            row.pop("_seconds_sort", None)
        return rows

    return {
        "season": meta["season"],
        "phase": meta["phase"],
        "game_id": meta["game_id"],
        "game_date": meta["game_date"] or None,
        "game_type": meta["game_type"] or None,
        "home_team": meta["home_team"],
        "road_team": meta["road_team"],
        "source": pbp_source,
        "minutes_plus_minus_source": minutes_plus_minus_source,
        "home_players": _serialize_team(meta["home_team_id"]),
        "road_players": _serialize_team(meta["road_team_id"]),
    }
