#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.admin.pbp_game_states import _build_pbp_path, _counts_toward_pf, _load_pbp_df  # type: ignore
from backend.config import build_data_filename, resolve_data_file_path  # type: ignore


SUPPORTED_STATS = [
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
    "pts",
]
UNSUPPORTED_COMPARE_STATS = ["minutes", "plus_minus"]
BOXSCORE_REQUEST_TIMEOUT = 30.0
PBP_REQUEST_TIMEOUT = 20.0
REQUEST_RETRIES = 3
REQUEST_PAUSE_SECONDS = 0.35
OFF_DEF_REB_RE = re.compile(r"\(Off:(\d+)\s+Def:(\d+)\)", re.IGNORECASE)
ASSIST_RE = re.compile(r"\(([^()]*)\s+\d+\s+AST\)", re.IGNORECASE)


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


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _to_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _clean_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace(".", " ")
    text = re.sub(r"[^A-Za-z0-9' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _name_i_from_full_name(full_name: str) -> str:
    parts = [part for part in re.split(r"\s+", full_name.strip()) if part]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def _season_start_year(season: str) -> int:
    match = re.match(r"^(\d{4})-\d{2}$", season.strip())
    if not match:
        raise ValueError(f"Invalid season format: {season!r}. Expected YYYY-YY.")
    return int(match.group(1))


def _month_label(month: int) -> str:
    return datetime(2000, month, 1).strftime("%B").lower()


def _output_dir(base_dir: Path, season: str, month: int, phase: str) -> Path:
    return base_dir / "player_boxscore_pbp_compare" / season / phase / _month_label(month)


def _load_month_game_logs(repo_dir: Path, season: str, month: int, phase: str) -> list[dict[str, Any]]:
    logs_path = resolve_data_file_path(
        build_data_filename("team_game_logs", season),
        repo_dir=repo_dir,
    )
    if not logs_path.exists():
        raise FileNotFoundError(f"Missing team logs: {logs_path}")

    df = pd.read_csv(logs_path, dtype={"game_id": "string"})
    if df.empty:
        return []

    d = df.copy()
    d["game_id"] = d["game_id"].map(_normalize_game_id)
    d["game_date_dt"] = pd.to_datetime(d["game_date"], errors="coerce")
    d["game_type_norm"] = d["game_type"].map(_normalize_text)
    d = d[d["game_date_dt"].dt.month == month].copy()
    if phase == "regular":
        d = d[d["game_type_norm"] == "regular_season"].copy()
    elif phase == "playoffs":
        d = d[d["game_type_norm"].isin(["playoffs", "play_in"])].copy()
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    d = d.sort_values(["game_date_dt", "game_id"], kind="stable")

    games: list[dict[str, Any]] = []
    for _, row in d.iterrows():
        games.append(
            {
                "game_id": _safe_str(row.get("game_id")),
                "game_date": row["game_date_dt"].strftime("%Y-%m-%d") if pd.notna(row["game_date_dt"]) else "",
                "game_type": _safe_str(row.get("game_type")),
                "home_team_id": _to_int(row.get("team_id_home")),
                "home_team": _safe_str(row.get("team_abbreviation_home")),
                "home_team_name": _safe_str(row.get("team_name_home")),
                "road_team_id": _to_int(row.get("team_id_road")),
                "road_team": _safe_str(row.get("team_abbreviation_road")),
                "road_team_name": _safe_str(row.get("team_name_road")),
            }
        )
    return games


def _normalize_pbp_df(df: pd.DataFrame, game_id: str = "") -> pd.DataFrame:
    d = df.copy()
    required_cols = [
        "actionNumber",
        "clock",
        "period",
        "teamId",
        "teamTricode",
        "personId",
        "playerName",
        "playerNameI",
        "shotResult",
        "isFieldGoal",
        "description",
        "actionType",
        "subType",
        "shotValue",
        "actionId",
        "gameId",
    ]
    for col in required_cols:
        if col not in d.columns:
            d[col] = pd.NA

    if game_id and "gameId" in d.columns:
        d["gameId"] = d["gameId"].fillna(game_id)

    d["gameId"] = d["gameId"].map(_normalize_game_id)
    d = d[d["gameId"] != ""].copy()

    int_cols = ["actionNumber", "period", "teamId", "personId", "isFieldGoal", "shotValue"]
    for col in int_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype("int64")

    action_id_numeric = pd.to_numeric(d["actionId"], errors="coerce")
    d["actionId"] = action_id_numeric.fillna(d["actionNumber"]).astype("int64")

    return d.sort_values(["gameId", "actionNumber", "actionId"], kind="stable").reset_index(drop=True)


def _fetch_cdnnba_pbp(game_id: str, timeout: float) -> pd.DataFrame:
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    actions = payload.get("game", {}).get("actions", [])
    df = pd.DataFrame(actions)
    if df.empty:
        raise RuntimeError(f"cdnnba returned no actions for game_id={game_id}")
    if "gameId" not in df.columns:
        df["gameId"] = game_id
    return _normalize_pbp_df(df, game_id=game_id)


def _fetch_live_pbp(game_id: str, timeout: float, retries: int) -> tuple[pd.DataFrame, str]:
    from nba_api.stats.endpoints import playbyplayv3

    errors: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            resp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=timeout)
            frames = resp.get_data_frames()
            if frames and frames[0] is not None and not frames[0].empty:
                return _normalize_pbp_df(frames[0], game_id=game_id), "nba_api"
            errors.append(f"nba_api attempt {attempt}: empty")
        except Exception as exc:
            errors.append(f"nba_api attempt {attempt}: {exc}")

        try:
            return _fetch_cdnnba_pbp(game_id, timeout=timeout), "cdnnba"
        except Exception as exc:
            errors.append(f"cdnnba attempt {attempt}: {exc}")

        if attempt < retries:
            time.sleep(0.4 * attempt)

    raise RuntimeError(" | ".join(errors[-4:]))


def _empty_stat_line() -> dict[str, int]:
    return {stat: 0 for stat in SUPPORTED_STATS}


def _normalize_v2_players(player_df: pd.DataFrame) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    for _, row in player_df.iterrows():
        full_name = _safe_str(row.get("PLAYER_NAME"))
        players.append(
            {
                "game_id": _normalize_game_id(row.get("GAME_ID")),
                "team_id": _to_int(row.get("TEAM_ID")),
                "team_abbreviation": _safe_str(row.get("TEAM_ABBREVIATION")),
                "player_id": _to_int(row.get("PLAYER_ID")),
                "player_name": full_name,
                "name_i": _name_i_from_full_name(full_name),
                "first_name": full_name.split(" ", 1)[0] if full_name else "",
                "family_name": full_name.split(" ", 1)[1] if " " in full_name else "",
                "comment": _safe_str(row.get("COMMENT")),
                "minutes": _safe_str(row.get("MIN")),
                "plus_minus": _to_int(row.get("PLUS_MINUS")),
                "fgm": _to_int(row.get("FGM")),
                "fga": _to_int(row.get("FGA")),
                "fg3m": _to_int(row.get("FG3M")),
                "fg3a": _to_int(row.get("FG3A")),
                "ftm": _to_int(row.get("FTM")),
                "fta": _to_int(row.get("FTA")),
                "oreb": _to_int(row.get("OREB")),
                "dreb": _to_int(row.get("DREB")),
                "reb": _to_int(row.get("REB")),
                "ast": _to_int(row.get("AST")),
                "stl": _to_int(row.get("STL")),
                "blk": _to_int(row.get("BLK")),
                "tov": _to_int(row.get("TO")),
                "pf": _to_int(row.get("PF")),
                "pts": _to_int(row.get("PTS")),
            }
        )
    return players


def _normalize_v3_players(player_df: pd.DataFrame) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    for _, row in player_df.iterrows():
        first_name = _safe_str(row.get("firstName"))
        family_name = _safe_str(row.get("familyName"))
        full_name = f"{first_name} {family_name}".strip()
        players.append(
            {
                "game_id": _normalize_game_id(row.get("gameId")),
                "team_id": _to_int(row.get("teamId")),
                "team_abbreviation": _safe_str(row.get("teamTricode")),
                "player_id": _to_int(row.get("personId")),
                "player_name": full_name,
                "name_i": _safe_str(row.get("nameI")) or _name_i_from_full_name(full_name),
                "first_name": first_name,
                "family_name": family_name,
                "comment": _safe_str(row.get("comment")),
                "minutes": _safe_str(row.get("minutes")),
                "plus_minus": _to_int(row.get("plusMinusPoints")),
                "fgm": _to_int(row.get("fieldGoalsMade")),
                "fga": _to_int(row.get("fieldGoalsAttempted")),
                "fg3m": _to_int(row.get("threePointersMade")),
                "fg3a": _to_int(row.get("threePointersAttempted")),
                "ftm": _to_int(row.get("freeThrowsMade")),
                "fta": _to_int(row.get("freeThrowsAttempted")),
                "oreb": _to_int(row.get("reboundsOffensive")),
                "dreb": _to_int(row.get("reboundsDefensive")),
                "reb": _to_int(row.get("reboundsTotal")),
                "ast": _to_int(row.get("assists")),
                "stl": _to_int(row.get("steals")),
                "blk": _to_int(row.get("blocks")),
                "tov": _to_int(row.get("turnovers")),
                "pf": _to_int(row.get("foulsPersonal")),
                "pts": _to_int(row.get("points")),
            }
        )
    return players


def _fetch_official_boxscore(
    game_id: str,
    timeout: float,
    retries: int,
    try_v2: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from nba_api.stats.endpoints import boxscoretraditionalv2, boxscoretraditionalv3

    v2_error = ""
    v2_player_rows = 0

    if try_v2:
        for attempt in range(1, retries + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    v2_resp = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=timeout)
                v2_frames = v2_resp.get_data_frames()
                v2_player_df = v2_frames[0] if v2_frames else pd.DataFrame()
                v2_player_rows = int(len(v2_player_df))
                if not v2_player_df.empty:
                    return _normalize_v2_players(v2_player_df), {
                        "endpoint_requested": "BoxScoreTraditionalV2",
                        "endpoint_used": "BoxScoreTraditionalV2",
                        "v2_player_rows": v2_player_rows,
                        "v2_status": "ok",
                        "v2_error": "",
                    }
                v2_error = "empty player rows"
                break
            except Exception as exc:
                v2_error = str(exc)
                if attempt < retries:
                    time.sleep(0.4 * attempt)
    else:
        v2_error = "skipped after earlier empty 2025-26 response"

    for attempt in range(1, retries + 1):
        try:
            v3_resp = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=timeout)
            v3_frames = v3_resp.get_data_frames()
            v3_player_df = v3_frames[0] if v3_frames else pd.DataFrame()
            if v3_player_df.empty:
                raise RuntimeError("empty player rows")
            return _normalize_v3_players(v3_player_df), {
                "endpoint_requested": "BoxScoreTraditionalV2",
                "endpoint_used": "BoxScoreTraditionalV3",
                "v2_player_rows": v2_player_rows,
                "v2_status": "empty" if v2_player_rows == 0 else "error",
                "v2_error": v2_error,
            }
        except Exception as exc:
            if attempt >= retries:
                raise RuntimeError(f"Could not fetch official box score for {game_id}: {exc}") from exc
            time.sleep(0.4 * attempt)

    raise RuntimeError(f"Could not fetch official box score for {game_id}")


def _direct_aliases_for_player(player: dict[str, Any]) -> set[str]:
    first_name = _safe_str(player.get("first_name"))
    family_name = _safe_str(player.get("family_name"))
    aliases = {
        _clean_name(player.get("player_name", "")),
        _clean_name(player.get("name_i", "")),
        _clean_name(f"{first_name[:1]}. {family_name}"),
        _clean_name(family_name),
    }
    if len(first_name) >= 3 and family_name:
        aliases.add(_clean_name(f"{first_name[:3]} {family_name}"))
    return {alias for alias in aliases if alias}


def _build_player_index(
    official_players: list[dict[str, Any]],
    pbp_df: pd.DataFrame,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, set[int]]], dict[int, dict[str, list[dict[str, Any]]]]]:
    players_by_id = {int(player["player_id"]): player for player in official_players if int(player["player_id"]) > 0}
    team_alias_map: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    alias_catalog: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for player in official_players:
        team_id = int(player["team_id"])
        player_id = int(player["player_id"])
        if team_id <= 0 or player_id <= 0:
            continue
        for alias in _direct_aliases_for_player(player):
            team_alias_map[team_id][alias].add(player_id)

    if not pbp_df.empty:
        for _, row in pbp_df.iterrows():
            team_id = _to_int(row.get("teamId"))
            player_id = _to_int(row.get("personId"))
            if team_id <= 0 or player_id <= 0 or player_id not in players_by_id:
                continue
            for raw_alias in [
                _safe_str(row.get("playerName")),
                _safe_str(row.get("playerNameI")),
                _safe_str(row.get("description")).split(" ", 1)[0],
            ]:
                alias = _clean_name(raw_alias)
                if alias:
                    team_alias_map[team_id][alias].add(player_id)

    for team_id, alias_map in team_alias_map.items():
        for alias, player_ids in alias_map.items():
            alias_catalog[team_id][alias] = [players_by_id[pid] for pid in sorted(player_ids) if pid in players_by_id]

    return players_by_id, team_alias_map, alias_catalog


def _resolve_assist_player_ids(
    description: str,
    team_id: int,
    team_alias_map: dict[int, dict[str, set[int]]],
) -> tuple[list[int], list[str]]:
    resolved: list[int] = []
    unresolved_aliases: list[str] = []
    for raw_alias in ASSIST_RE.findall(description or ""):
        alias = _clean_name(raw_alias)
        if not alias:
            continue
        player_ids = sorted(team_alias_map.get(team_id, {}).get(alias, set()))
        if len(player_ids) == 1:
            resolved.append(player_ids[0])
        else:
            unresolved_aliases.append(alias)
    return resolved, unresolved_aliases


def _build_pbp_player_stats(
    pbp_df: pd.DataFrame,
    official_players: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    players_by_id, team_alias_map, alias_catalog = _build_player_index(official_players, pbp_df)
    stats_by_player: dict[int, dict[str, Any]] = {}
    for player in official_players:
        player_id = int(player["player_id"])
        if player_id <= 0:
            continue
        stats_by_player[player_id] = {
            "player_id": player_id,
            "player_name": player["player_name"],
            "team_id": int(player["team_id"]),
            "team_abbreviation": player["team_abbreviation"],
            **_empty_stat_line(),
        }

    assist_issues: list[dict[str, Any]] = []
    rebound_prev: dict[tuple[int, int], dict[str, int]] = {}
    offensive_foul_turnover_actions: dict[int, list[int]] = defaultdict(list)

    events = pbp_df.sort_values(["actionNumber", "actionId"], kind="stable")
    for _, row in events.iterrows():
        action_type = _normalize_text(row.get("actionType"))
        sub_type = _normalize_text(row.get("subType"))
        player_id = _to_int(row.get("personId"))
        action_number = _to_int(row.get("actionNumber"))
        if player_id > 0 and action_type == "turnover" and "offensive foul" in sub_type:
            offensive_foul_turnover_actions[player_id].append(action_number)

    for _, row in events.iterrows():
        action_type = _normalize_text(row.get("actionType"))
        sub_type = _normalize_text(row.get("subType"))
        description = _safe_str(row.get("description"))
        description_lower = description.lower()
        description_upper = description.upper()
        player_id = _to_int(row.get("personId"))
        team_id = _to_int(row.get("teamId"))
        shot_result = _normalize_text(row.get("shotResult"))
        shot_value = _to_int(row.get("shotValue"))
        action_number = _to_int(row.get("actionNumber"))

        stat_line = stats_by_player.get(player_id)

        if player_id > 0 and stat_line is None and team_id > 0:
            stat_line = {
                "player_id": player_id,
                "player_name": _safe_str(row.get("playerNameI")) or _safe_str(row.get("playerName")),
                "team_id": team_id,
                "team_abbreviation": _safe_str(row.get("teamTricode")),
                **_empty_stat_line(),
            }
            stats_by_player[player_id] = stat_line

        if action_type in {"2pt", "3pt", "made shot", "missed shot", "heave"} and stat_line is not None:
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
                assist_ids, unresolved_aliases = _resolve_assist_player_ids(description, team_id, team_alias_map)
                for assist_player_id in assist_ids:
                    assist_line = stats_by_player.get(assist_player_id)
                    if assist_line is not None:
                        assist_line["ast"] += 1
                for alias in unresolved_aliases:
                    assist_issues.append(
                        {
                            "game_id": _normalize_game_id(row.get("gameId")),
                            "team_id": team_id,
                            "team_abbreviation": _safe_str(row.get("teamTricode")),
                            "alias": alias,
                            "description": description,
                            "candidate_players": [
                                {
                                    "player_id": int(candidate["player_id"]),
                                    "player_name": candidate["player_name"],
                                }
                                for candidate in alias_catalog.get(team_id, {}).get(alias, [])
                            ],
                            "issue": "ambiguous" if len(alias_catalog.get(team_id, {}).get(alias, [])) > 1 else "unresolved",
                        }
                    )

        elif action_type in {"freethrow", "free throw"} and stat_line is not None:
            stat_line["fta"] += 1
            made_ft = shot_result == "made" or ("free throw" in description_lower and "miss" not in description_lower)
            if made_ft:
                stat_line["ftm"] += 1
                stat_line["pts"] += 1

        elif action_type == "rebound" and stat_line is not None:
            if player_id <= 0:
                continue
            if sub_type == "offensive":
                stat_line["oreb"] += 1
                stat_line["reb"] += 1
            elif sub_type == "defensive":
                stat_line["dreb"] += 1
                stat_line["reb"] += 1
            else:
                reb_match = OFF_DEF_REB_RE.search(description)
                if reb_match:
                    off_total = int(reb_match.group(1))
                    def_total = int(reb_match.group(2))
                    key = (team_id, player_id)
                    prev = rebound_prev.get(key, {"off": 0, "def": 0})
                    off_delta = max(0, off_total - prev["off"])
                    def_delta = max(0, def_total - prev["def"])
                    if off_delta:
                        stat_line["oreb"] += off_delta
                        stat_line["reb"] += off_delta
                    if def_delta:
                        stat_line["dreb"] += def_delta
                        stat_line["reb"] += def_delta
                    rebound_prev[key] = {"off": max(prev["off"], off_total), "def": max(prev["def"], def_total)}
                elif "offensive rebound" in description_lower:
                    stat_line["oreb"] += 1
                    stat_line["reb"] += 1
                elif "defensive rebound" in description_lower:
                    stat_line["dreb"] += 1
                    stat_line["reb"] += 1

        elif action_type == "turnover" and stat_line is not None:
            stat_line["tov"] += 1

        elif action_type == "foul" and stat_line is not None and _counts_toward_pf(sub_type, description_lower):
            stat_line["pf"] += 1
            offensive_foul_like = (
                "offensive" in sub_type
                or "charge" in sub_type
                or "off.foul" in description_lower
            )
            if offensive_foul_like:
                nearby_turnover = any(
                    abs(action_number - candidate) <= 3
                    for candidate in offensive_foul_turnover_actions.get(player_id, [])
                )
                if not nearby_turnover:
                    stat_line["tov"] += 1

        elif stat_line is not None and (action_type == "steal" or (not action_type and "STEAL" in description_upper)):
            stat_line["stl"] += 1

        elif stat_line is not None and (action_type == "block" or (not action_type and "BLOCK" in description_upper)):
            stat_line["blk"] += 1

    extra_players = [
        player
        for player_id, player in sorted(stats_by_player.items())
        if player_id not in players_by_id and any(int(player[stat]) != 0 for stat in SUPPORTED_STATS)
    ]
    return stats_by_player, assist_issues, extra_players


def _compare_game(
    official_players: list[dict[str, Any]],
    pbp_stats_by_player: dict[int, dict[str, Any]],
    assist_issues: list[dict[str, Any]],
    extra_players: list[dict[str, Any]],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    stat_match_counter = Counter()
    stat_total_counter = Counter()

    for player in sorted(official_players, key=lambda item: (item["team_abbreviation"], item["player_name"], item["player_id"])):
        player_id = int(player["player_id"])
        pbp_stats = pbp_stats_by_player.get(player_id, {
            "player_id": player_id,
            "player_name": player["player_name"],
            "team_id": int(player["team_id"]),
            "team_abbreviation": player["team_abbreviation"],
            **_empty_stat_line(),
        })
        diffs = {
            stat: int(pbp_stats[stat]) - int(player[stat])
            for stat in SUPPORTED_STATS
        }
        for stat in SUPPORTED_STATS:
            stat_total_counter[stat] += 1
            if diffs[stat] == 0:
                stat_match_counter[stat] += 1
        comparisons.append(
            {
                "player_id": player_id,
                "player_name": player["player_name"],
                "team_id": int(player["team_id"]),
                "team_abbreviation": player["team_abbreviation"],
                "official": {stat: int(player[stat]) for stat in SUPPORTED_STATS},
                "pbp": {stat: int(pbp_stats[stat]) for stat in SUPPORTED_STATS},
                "diff": diffs,
                "match": all(delta == 0 for delta in diffs.values()),
                "minutes": player["minutes"],
                "plus_minus_official": int(player["plus_minus"]),
                "comment": player["comment"],
            }
        )

    ambiguity_counts = Counter()
    unresolved_counts = Counter()
    for issue in assist_issues:
        key = (issue["team_abbreviation"], issue["alias"], issue["issue"])
        if issue["issue"] == "ambiguous":
            ambiguity_counts[key] += 1
        else:
            unresolved_counts[key] += 1

    return {
        "players": comparisons,
        "players_total": len(comparisons),
        "players_matching_all_supported_stats": sum(1 for item in comparisons if item["match"]),
        "players_with_any_mismatch": sum(1 for item in comparisons if not item["match"]),
        "stat_match_counts": {stat: stat_match_counter[stat] for stat in SUPPORTED_STATS},
        "stat_total_counts": {stat: stat_total_counter[stat] for stat in SUPPORTED_STATS},
        "assist_alias_issues": assist_issues,
        "assist_alias_issues_summary": {
            "ambiguous": [
                {
                    "team_abbreviation": team_abbreviation,
                    "alias": alias,
                    "occurrences": count,
                }
                for (team_abbreviation, alias, _), count in sorted(ambiguity_counts.items())
            ],
            "unresolved": [
                {
                    "team_abbreviation": team_abbreviation,
                    "alias": alias,
                    "occurrences": count,
                }
                for (team_abbreviation, alias, _), count in sorted(unresolved_counts.items())
            ],
        },
        "extra_pbp_players": extra_players,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_compare_outputs(
    season: str,
    month: int,
    phase: str,
    data_repo_dir: Path,
    output_base_dir: Path,
    max_games: Optional[int],
    include_live_pbp_fallback: bool,
    pbp_source_mode: str,
) -> dict[str, Path]:
    games = _load_month_game_logs(data_repo_dir, season=season, month=month, phase=phase)
    if max_games is not None and max_games > 0:
        games = games[:max_games]
    if not games:
        raise RuntimeError(f"No games found for season={season} month={month} phase={phase}")

    pbp_path, stored_pbp_source = _build_pbp_path(data_repo_dir, season, phase, source="nbastatsv3")
    local_pbp_df = _load_pbp_df(pbp_path) if pbp_path.exists() else pd.DataFrame()
    local_pbp_df = _normalize_pbp_df(local_pbp_df) if not local_pbp_df.empty else pd.DataFrame(columns=["gameId"])
    local_grouped = local_pbp_df.groupby("gameId", sort=False) if not local_pbp_df.empty else None

    official_games: list[dict[str, Any]] = []
    compare_games: list[dict[str, Any]] = []

    endpoint_usage = Counter()
    live_pbp_source_usage = Counter()
    summary_stat_match = Counter()
    summary_stat_total = Counter()
    games_with_ambiguity = 0
    games_missing_pbp = 0
    live_pbp_fetches = 0
    season_v2_empty = False

    for idx, game in enumerate(games, 1):
        game_id = game["game_id"]
        official_players, source_meta = _fetch_official_boxscore(
            game_id,
            timeout=BOXSCORE_REQUEST_TIMEOUT,
            retries=REQUEST_RETRIES,
            try_v2=not season_v2_empty,
        )
        if source_meta.get("v2_status") == "empty":
            season_v2_empty = True
        endpoint_usage[source_meta["endpoint_used"]] += 1
        time.sleep(REQUEST_PAUSE_SECONDS)

        local_pbp_hit = local_grouped is not None and game_id in local_grouped.indices
        pbp_source = "missing"
        pbp_df = pd.DataFrame(columns=local_pbp_df.columns)

        if pbp_source_mode == "live_first":
            if include_live_pbp_fallback:
                try:
                    pbp_df, live_source = _fetch_live_pbp(game_id, timeout=PBP_REQUEST_TIMEOUT, retries=REQUEST_RETRIES)
                    pbp_source = f"live_{live_source}"
                    live_pbp_source_usage[pbp_source] += 1
                    live_pbp_fetches += 1
                    time.sleep(REQUEST_PAUSE_SECONDS)
                except Exception:
                    if local_pbp_hit:
                        pbp_df = local_grouped.get_group(game_id).copy()
                        pbp_source = f"stored_{stored_pbp_source}_fallback"
                    else:
                        raise
            elif local_pbp_hit:
                pbp_df = local_grouped.get_group(game_id).copy()
                pbp_source = f"stored_{stored_pbp_source}"
        else:
            if local_pbp_hit:
                pbp_df = local_grouped.get_group(game_id).copy()
                pbp_source = f"stored_{stored_pbp_source}"
            elif include_live_pbp_fallback:
                pbp_df, live_source = _fetch_live_pbp(game_id, timeout=PBP_REQUEST_TIMEOUT, retries=REQUEST_RETRIES)
                pbp_source = f"live_{live_source}"
                live_pbp_source_usage[pbp_source] += 1
                live_pbp_fetches += 1
                time.sleep(REQUEST_PAUSE_SECONDS)

        if pbp_df.empty:
            games_missing_pbp += 1

        pbp_stats_by_player, assist_issues, extra_players = _build_pbp_player_stats(pbp_df, official_players)
        comparison = _compare_game(
            official_players=official_players,
            pbp_stats_by_player=pbp_stats_by_player,
            assist_issues=assist_issues,
            extra_players=extra_players,
        )

        if comparison["assist_alias_issues"]:
            games_with_ambiguity += 1

        for stat in SUPPORTED_STATS:
            summary_stat_match[stat] += comparison["stat_match_counts"][stat]
            summary_stat_total[stat] += comparison["stat_total_counts"][stat]

        official_games.append(
            {
                **game,
                "official_source": source_meta,
                "players": official_players,
            }
        )
        compare_games.append(
            {
                **game,
                "official_source": source_meta,
                "pbp_source": pbp_source,
                "local_pbp_hit": local_pbp_hit,
                "players": comparison["players"],
                "players_total": comparison["players_total"],
                "players_matching_all_supported_stats": comparison["players_matching_all_supported_stats"],
                "players_with_any_mismatch": comparison["players_with_any_mismatch"],
                "assist_alias_issues": comparison["assist_alias_issues"],
                "assist_alias_issues_summary": comparison["assist_alias_issues_summary"],
                "extra_pbp_players": comparison["extra_pbp_players"],
            }
        )

        print(
            f"[{idx}/{len(games)}] {game_id} {game['road_team']} @ {game['home_team']} "
            f"official={source_meta['endpoint_used']} pbp={pbp_source} "
            f"player_matches={comparison['players_matching_all_supported_stats']}/{comparison['players_total']}"
        )

    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "season": season,
        "phase": phase,
        "month": month,
        "month_name": _month_label(month),
        "built_at_utc": built_at,
        "games_total": len(games),
        "supported_stats": SUPPORTED_STATS,
        "unsupported_compare_stats": UNSUPPORTED_COMPARE_STATS,
        "stored_pbp_path": str(pbp_path),
        "stored_pbp_source": stored_pbp_source,
        "official_endpoint_usage": dict(endpoint_usage),
        "live_pbp_fetches": live_pbp_fetches,
        "live_pbp_source_usage": dict(live_pbp_source_usage),
        "games_missing_pbp": games_missing_pbp,
        "games_with_assist_alias_issues": games_with_ambiguity,
        "stat_match_counts": {stat: summary_stat_match[stat] for stat in SUPPORTED_STATS},
        "stat_total_counts": {stat: summary_stat_total[stat] for stat in SUPPORTED_STATS},
        "stat_match_rates": {
            stat: round(summary_stat_match[stat] / summary_stat_total[stat], 4) if summary_stat_total[stat] else 0.0
            for stat in SUPPORTED_STATS
        },
    }

    output_dir = _output_dir(output_base_dir, season=season, month=month, phase=phase)
    official_path = output_dir / f"official_boxscores_{season}_{phase}_{_month_label(month)}.json"
    compare_path = output_dir / f"pbp_compare_{season}_{phase}_{_month_label(month)}.json"
    summary_path = output_dir / f"summary_{season}_{phase}_{_month_label(month)}.json"

    _write_json(
        official_path,
        {
            "metadata": {
                "season": season,
                "phase": phase,
                "month": month,
                "month_name": _month_label(month),
                "built_at_utc": built_at,
                "endpoint_requested": "BoxScoreTraditionalV2",
                "note": (
                    "BoxScoreTraditionalV2 is empty for 2025-26 games, so the script falls back to "
                    "BoxScoreTraditionalV3 and normalizes it into a traditional-stat shape."
                ),
            },
            "games": official_games,
        },
    )
    _write_json(
        compare_path,
        {
            "metadata": {
                "season": season,
                "phase": phase,
                "month": month,
                "month_name": _month_label(month),
                "built_at_utc": built_at,
                "supported_stats": SUPPORTED_STATS,
                "unsupported_compare_stats": UNSUPPORTED_COMPARE_STATS,
                "stored_pbp_path": str(pbp_path),
                "stored_pbp_source": stored_pbp_source,
            },
            "games": compare_games,
        },
    )
    _write_json(summary_path, summary)

    return {
        "official": official_path,
        "compare": compare_path,
        "summary": summary_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch official player box scores for one month of one season and compare "
            "them against a player-ID-based PBP reconstruction."
        )
    )
    parser.add_argument("--season", default="2025-26", help="Season like 2025-26")
    parser.add_argument("--month", type=int, default=2, help="Calendar month number, e.g. 2 for February")
    parser.add_argument("--phase", choices=["regular", "playoffs"], default="regular")
    parser.add_argument(
        "--data-repo-dir",
        default=str((REPO_ROOT.parent / "NBA_Data").resolve()),
        help="Path to NBA_Data repo",
    )
    parser.add_argument(
        "--output-base-dir",
        default=str((REPO_ROOT / "reports").resolve()),
        help="Base output directory for generated JSON files",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Optional cap for test runs")
    parser.add_argument(
        "--pbp-source",
        choices=["stored_first", "live_first"],
        default="stored_first",
        help="Whether to prefer stored nbastatsv3 PBP or fetch live PlayByPlayV3 data first.",
    )
    parser.add_argument(
        "--no-live-pbp-fallback",
        action="store_true",
        help="Do not fetch missing stored PBP games live",
    )
    args = parser.parse_args()

    outputs = build_compare_outputs(
        season=args.season,
        month=args.month,
        phase=args.phase,
        data_repo_dir=Path(args.data_repo_dir).resolve(),
        output_base_dir=Path(args.output_base_dir).resolve(),
        max_games=args.max_games,
        include_live_pbp_fallback=not args.no_live_pbp_fallback,
        pbp_source_mode=args.pbp_source,
    )

    print("\nWrote outputs:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
