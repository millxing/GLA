from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from admin.pbp_game_states import _build_pbp_path, _load_pbp_df, _normalize_game_id, _safe_str  # type: ignore
from config import build_data_filename, get_available_seasons, resolve_data_file_path


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


def _player_shots_output_path(repo_dir: Path, season: str, output_root: Optional[str] = None) -> Path:
    root = Path(output_root).expanduser().resolve() if output_root else Path(repo_dir) / "player_shots"
    return root / f"player_shots_{season}.parquet"


def _load_team_logs(repo_dir: Path, season: str) -> pd.DataFrame:
    logs_path = resolve_data_file_path(build_data_filename("team_game_logs", season), repo_dir=repo_dir)
    logs = pd.read_csv(logs_path, dtype={"game_id": "string"})
    logs["game_id"] = logs["game_id"].map(_normalize_game_id)
    parsed_dates = pd.to_datetime(logs["game_date"], errors="coerce", format="mixed")
    logs["game_date"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna(logs["game_date"].map(_safe_str))
    return logs[
        [
            "game_id",
            "game_date",
            "game_type",
            "team_id_home",
            "team_abbreviation_home",
            "team_id_road",
            "team_abbreviation_road",
        ]
    ].drop_duplicates(subset=["game_id"], keep="last")


def _description_result(description: pd.Series) -> pd.Series:
    desc = description.astype("string").fillna("").str.strip()
    result = pd.Series(pd.NA, index=description.index, dtype="string")
    result = result.mask(desc.str.upper().str.startswith("MISS"), "miss")
    result = result.mask(result.isna() & desc.str.contains(r"\(\d+\s+PTS\)", case=False, regex=True, na=False), "make")
    return result


def extract_player_shots_frame(pbp_df: pd.DataFrame, logs_df: pd.DataFrame, *, season: str, phase: str) -> pd.DataFrame:
    if pbp_df.empty:
        return pd.DataFrame(columns=PLAYER_SHOTS_COLUMNS)

    d = pbp_df.copy()
    for col in [
        "actionNumber",
        "actionId",
        "period",
        "clock",
        "teamId",
        "teamTricode",
        "personId",
        "playerName",
        "shotResult",
        "isFieldGoal",
        "shotValue",
        "description",
        "actionType",
        "subType",
        "gameId",
    ]:
        if col not in d.columns:
            d[col] = pd.NA

    action = d["actionType"].astype("string").fillna("").str.strip().str.lower()
    subtype = d["subType"].astype("string").fillna("").str.strip().str.lower()
    desc = d["description"].astype("string").fillna("").str.strip().str.lower()
    shot_result = d["shotResult"].astype("string").fillna("").str.strip().str.lower()
    is_field_goal = pd.to_numeric(d["isFieldGoal"], errors="coerce").fillna(0).astype(int).eq(1)
    shot_value = pd.to_numeric(d["shotValue"], errors="coerce")

    player_id = pd.to_numeric(d["personId"], errors="coerce").fillna(0).astype(int)
    is_free_throw = action.isin({"freethrow", "free throw"}) | subtype.str.contains(r"\bfree throw\b", regex=True, na=False)
    is_field_goal_attempt = is_field_goal | action.isin({"2pt", "3pt", "made shot", "missed shot"})
    is_three = action.eq("3pt") | shot_value.eq(3) | desc.str.contains(r"\b3\s*pt\b|\b3-pointer\b|\b3 point\b", regex=True, na=False)
    is_shot = player_id.gt(0) & (is_free_throw | is_field_goal_attempt)

    shots = d.loc[is_shot].copy()
    if shots.empty:
        return pd.DataFrame(columns=PLAYER_SHOTS_COLUMNS)

    shot_type = pd.Series("2ptfga", index=shots.index, dtype="string")
    shot_type = shot_type.mask(is_free_throw.loc[shots.index], "fta")
    shot_type = shot_type.mask(is_field_goal_attempt.loc[shots.index] & is_three.loc[shots.index], "3ptfga")

    result = pd.Series(pd.NA, index=shots.index, dtype="string")
    result = result.mask(shot_result.loc[shots.index].eq("made"), "make")
    result = result.mask(shot_result.loc[shots.index].eq("missed"), "miss")
    result = result.mask(result.isna() & action.loc[shots.index].eq("made shot"), "make")
    result = result.mask(result.isna() & action.loc[shots.index].eq("missed shot"), "miss")
    result = result.fillna(_description_result(shots["description"]))

    out = pd.DataFrame(
        {
            "season": season,
            "pbp_phase": phase,
            "game_id": shots["gameId"].map(_normalize_game_id),
            "team_id": pd.to_numeric(shots["teamId"], errors="coerce").fillna(0).astype(int),
            "team": shots["teamTricode"].map(_safe_str),
            "player_id": pd.to_numeric(shots["personId"], errors="coerce").fillna(0).astype(int),
            "player_name": shots["playerName"].map(_safe_str),
            "shot_type": shot_type,
            "result": result,
            "action_number": pd.to_numeric(shots["actionNumber"], errors="coerce").fillna(0).astype(int),
            "action_id": pd.to_numeric(shots["actionId"], errors="coerce").fillna(0).astype(int),
            "period": pd.to_numeric(shots["period"], errors="coerce").fillna(0).astype(int),
            "clock": shots["clock"].map(_safe_str),
            "description": shots["description"].map(_safe_str),
        }
    )
    out = out[out["result"].isin(["make", "miss"])].copy()

    merged = out.merge(logs_df, on="game_id", how="left")
    merged["team_id_home"] = pd.to_numeric(merged["team_id_home"], errors="coerce").fillna(0).astype(int)
    merged["team_id_road"] = pd.to_numeric(merged["team_id_road"], errors="coerce").fillna(0).astype(int)
    merged["is_home"] = merged["team_id"].eq(merged["team_id_home"])
    merged["home_road"] = merged["is_home"].map({True: "home", False: "road"})
    merged["opponent_id"] = merged.apply(
        lambda row: int(row["team_id_road"]) if bool(row["is_home"]) else int(row["team_id_home"]),
        axis=1,
    )
    merged["opponent"] = merged.apply(
        lambda row: _safe_str(row.get("team_abbreviation_road")) if bool(row["is_home"]) else _safe_str(row.get("team_abbreviation_home")),
        axis=1,
    )

    merged = merged[PLAYER_SHOTS_COLUMNS].copy()
    merged = merged.sort_values(["game_date", "game_id", "action_number", "action_id"], kind="stable").reset_index(drop=True)
    return merged


def build_player_shots(
    *,
    season: str,
    repo_dir: Path,
    phase: str = "regular",
    pbp_source: str = "auto",
    output_root: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    if season == "all":
        for season_name in get_available_seasons():
            build_player_shots(
                season=season_name,
                repo_dir=repo_dir,
                phase=phase,
                pbp_source=pbp_source,
                output_root=output_root,
                overwrite=overwrite,
            )
        return 0

    out_path = _player_shots_output_path(repo_dir, season, output_root=output_root)
    if out_path.exists() and not overwrite:
        print(f"[player-shots] Skip existing: {out_path}")
        return 0

    logs_df = _load_team_logs(repo_dir, season)
    frames: list[pd.DataFrame] = []
    phases = ["regular", "playoffs"] if phase == "both" else [phase]

    for phase_name in phases:
        pbp_path, resolved_source = _build_pbp_path(repo_dir, season, phase_name, source=pbp_source)
        if not pbp_path.exists():
            print(f"[player-shots] Missing PBP source for {season} {phase_name}: {pbp_path}")
            continue

        print(f"[player-shots] Reading {pbp_path} ({resolved_source})")
        pbp_df = _load_pbp_df(pbp_path)
        shots = extract_player_shots_frame(pbp_df, logs_df, season=season, phase=phase_name)
        if not shots.empty:
            frames.append(shots)

    combined = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else pd.DataFrame(columns=PLAYER_SHOTS_COLUMNS)
    )
    combined = combined.sort_values(["game_date", "game_id", "action_number", "action_id"], kind="stable").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False, compression="zstd")
    print(f"[player-shots] Wrote {out_path} rows={len(combined)} phases={','.join(phases)}")

    return 0
