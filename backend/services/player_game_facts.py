from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from admin.pbp_game_states import _normalize_game_id, _safe_str  # type: ignore
from config import (
    build_box_score_traditional_filename,
    build_data_filename,
    resolve_data_file_path,
)


COUNT_STAT_COLUMN_MAP = {
    "points": "pts",
    "field_goals_made": "fgm",
    "field_goals_attempted": "fga",
    "three_pointers_made": "fg3m",
    "three_pointers_attempted": "fg3a",
    "free_throws_made": "ftm",
    "free_throws_attempted": "fta",
    "rebounds_offensive": "oreb",
    "rebounds_defensive": "dreb",
    "rebounds_total": "reb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "tov",
    "fouls_personal": "pf",
    "plus_minus_points": "plus_minus",
}


def _to_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _to_optional_float(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_minutes_to_seconds(value: Any) -> int:
    text = _safe_str(value)
    if not text or ":" not in text:
        return 0
    pieces = text.split(":")
    if len(pieces) < 2:
        return 0
    minutes = _to_int(pieces[0], default=0)
    seconds = _to_int(pieces[1], default=0)
    return max(0, minutes * 60 + seconds)


def _normalize_boxscore_game_id_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_id"] = out["game_id"].astype("string")
    out["game_id"] = out["game_id"].map(_normalize_game_id)
    out = out[out["game_id"] != ""].copy()
    return out


def _load_player_boxscore_df(season: str) -> pd.DataFrame:
    players_path = resolve_data_file_path(build_box_score_traditional_filename("players", season))
    players_df = pd.read_csv(players_path, dtype={"game_id": "string"})
    players_df = _normalize_boxscore_game_id_column(players_df)
    players_df["player_id"] = pd.to_numeric(players_df["person_id"], errors="coerce").fillna(0).astype(int)
    players_df = players_df[players_df["player_id"] > 0].copy()
    players_df = players_df.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
    return players_df


def _load_team_logs_df(season: str) -> pd.DataFrame:
    logs_path = resolve_data_file_path(build_data_filename("team_game_logs", season))
    logs_df = pd.read_csv(logs_path, dtype={"game_id": "string"})
    logs_df = _normalize_boxscore_game_id_column(logs_df)
    return logs_df


def _load_advanced_boxscore_df(season: str) -> pd.DataFrame:
    advanced_path = resolve_data_file_path(build_data_filename("box_score_advanced", season))
    advanced_df = pd.read_csv(advanced_path, dtype={"game_id": "string"})
    advanced_df = _normalize_boxscore_game_id_column(advanced_df)
    advanced_df = advanced_df.drop_duplicates(subset=["game_id"], keep="last")
    return advanced_df


def load_player_game_facts_frame(
    season: str,
    *,
    game_id: Optional[str] = None,
    player_id: Optional[int] = None,
    team_id: Optional[int] = None,
    include_dnp: bool = False,
) -> pd.DataFrame:
    players_df = _load_player_boxscore_df(season)
    logs_df = _load_team_logs_df(season)
    advanced_df = _load_advanced_boxscore_df(season)

    if game_id:
        game_id_norm = _normalize_game_id(game_id)
        players_df = players_df[players_df["game_id"] == game_id_norm].copy()
    if player_id is not None:
        players_df = players_df[players_df["player_id"] == int(player_id)].copy()
    if team_id is not None:
        players_df = players_df[pd.to_numeric(players_df["team_id"], errors="coerce").fillna(0).astype(int) == int(team_id)].copy()

    if players_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id", "season", "game_date", "game_type", "player_id", "player_name",
                "team_id", "team_abbreviation", "opponent_team_id", "opponent_abbreviation",
                "home_or_road", "is_starter", "position", "status_comment", "minutes",
                "seconds_played", *COUNT_STAT_COLUMN_MAP.values(), "possessions_team",
                "possessions_opp", "source_boxscore", "source_possessions",
            ]
        )

    logs_keep = logs_df[
        [
            "game_id", "game_date", "game_type",
            "team_id_home", "team_abbreviation_home",
            "team_id_road", "team_abbreviation_road",
        ]
    ].copy()
    advanced_keep = advanced_df[
        [
            "game_id",
            "possessions_home", "minutes_home",
            "possessions_road", "minutes_road",
        ]
    ].copy()

    merged = players_df.merge(logs_keep, on="game_id", how="left", suffixes=("", "_log"))
    merged = merged.merge(advanced_keep, on="game_id", how="left", suffixes=("", "_adv"))

    merged["team_id"] = pd.to_numeric(merged["team_id"], errors="coerce").fillna(0).astype(int)
    merged["team_id_home"] = pd.to_numeric(merged["team_id_home"], errors="coerce").fillna(0).astype(int)
    merged["team_id_road"] = pd.to_numeric(merged["team_id_road"], errors="coerce").fillna(0).astype(int)
    merged["home_or_road"] = merged["home_or_road"].astype(str).str.strip().str.lower()
    merged["is_home"] = merged["home_or_road"].eq("home")

    merged["opponent_team_id"] = merged.apply(
        lambda row: int(row["team_id_road"]) if bool(row["is_home"]) else int(row["team_id_home"]),
        axis=1,
    )
    merged["opponent_abbreviation"] = merged.apply(
        lambda row: _safe_str(row.get("team_abbreviation_road")) if bool(row["is_home"]) else _safe_str(row.get("team_abbreviation_home")),
        axis=1,
    )
    merged["team_abbreviation"] = merged["team_tricode"].map(_safe_str)
    merged["player_name"] = (
        merged["first_name"].map(_safe_str)
        + " "
        + merged["family_name"].map(_safe_str)
    ).str.strip()
    merged.loc[merged["player_name"] == "", "player_name"] = merged["name_i"].map(_safe_str)
    merged["position"] = merged["position"].map(_safe_str)
    merged["position"] = merged["position"].replace("", None)
    merged["status_comment"] = merged["comment"].map(_safe_str)
    merged["status_comment"] = merged["status_comment"].replace("", None)
    merged["minutes"] = merged["minutes"].map(_safe_str)
    merged["minutes"] = merged["minutes"].replace("", "0:00")
    merged["seconds_played"] = merged["minutes"].map(_parse_minutes_to_seconds)
    merged["is_starter"] = None

    for source_column, target_column in COUNT_STAT_COLUMN_MAP.items():
        merged[target_column] = pd.to_numeric(merged[source_column], errors="coerce").fillna(0).astype(int)

    merged["possessions_team"] = merged.apply(
        lambda row: _to_optional_float(row.get("possessions_home")) if bool(row["is_home"]) else _to_optional_float(row.get("possessions_road")),
        axis=1,
    )
    merged["possessions_opp"] = merged.apply(
        lambda row: _to_optional_float(row.get("possessions_road")) if bool(row["is_home"]) else _to_optional_float(row.get("possessions_home")),
        axis=1,
    )
    merged["source_boxscore"] = "box_score_traditional_v3_players"
    merged["source_possessions"] = "box_score_advanced"

    if not include_dnp:
        merged = merged[merged["seconds_played"] > 0].copy()

    ordered_columns = [
        "game_id",
        "season",
        "game_date",
        "game_type",
        "player_id",
        "player_name",
        "team_id",
        "team_abbreviation",
        "opponent_team_id",
        "opponent_abbreviation",
        "home_or_road",
        "is_starter",
        "position",
        "status_comment",
        "minutes",
        "seconds_played",
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
        "possessions_team",
        "possessions_opp",
        "source_boxscore",
        "source_possessions",
    ]
    merged = merged[ordered_columns].copy()
    merged = merged.sort_values(
        ["game_date", "game_id", "team_abbreviation", "seconds_played", "player_name"],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return merged


def build_player_game_facts_payload(
    season: str,
    *,
    game_id: Optional[str] = None,
    player_id: Optional[int] = None,
    team_id: Optional[int] = None,
    include_dnp: bool = False,
) -> dict[str, Any]:
    frame = load_player_game_facts_frame(
        season,
        game_id=game_id,
        player_id=player_id,
        team_id=team_id,
        include_dnp=include_dnp,
    )

    rows = frame.where(pd.notnull(frame), None).to_dict(orient="records")
    return {
        "season": season,
        "game_id": _normalize_game_id(game_id) if game_id else None,
        "player_id": int(player_id) if player_id is not None else None,
        "team_id": int(team_id) if team_id is not None else None,
        "include_dnp": include_dnp,
        "row_count": len(rows),
        "rows": rows,
    }
