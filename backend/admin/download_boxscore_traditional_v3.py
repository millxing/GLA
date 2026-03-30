#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config import (  # type: ignore
    DEFAULT_NBA_DATA_REPO_DIR,
    build_box_score_traditional_filename,
    build_data_filename,
    get_canonical_data_file_path,
    resolve_data_file_path,
)


DEFAULT_REPO_DIR = DEFAULT_NBA_DATA_REPO_DIR
API_TIMEOUT_SECONDS = 30.0
REQUEST_RETRIES = 3
REQUEST_PAUSE_SECONDS = 0.75
SAVE_EVERY_GAMES = 25
TIMEOUT_COOLDOWN_THRESHOLD = 3
TIMEOUT_COOLDOWN_SECONDS = 90.0
BATCH_REST_EVERY_GAMES = 100
BATCH_REST_SECONDS = 30.0
MAX_TIMEOUT_COOLDOWNS_PER_RUN = 2
AUTO_RESUME_SLEEP_SECONDS = 300.0
MAX_AUTO_RESUME_PASSES = 100
BETWEEN_SEASONS_SLEEP_SECONDS = 240.0
AUTO_RESUME_OFFSET_STEP = 50


PLAYER_RENAME_MAP = {
    "gameId": "game_id",
    "teamId": "team_id",
    "teamCity": "team_city",
    "teamName": "team_name",
    "teamTricode": "team_tricode",
    "teamSlug": "team_slug",
    "personId": "person_id",
    "firstName": "first_name",
    "familyName": "family_name",
    "nameI": "name_i",
    "playerSlug": "player_slug",
    "jerseyNum": "jersey_num",
    "fieldGoalsMade": "field_goals_made",
    "fieldGoalsAttempted": "field_goals_attempted",
    "fieldGoalsPercentage": "field_goals_percentage",
    "threePointersMade": "three_pointers_made",
    "threePointersAttempted": "three_pointers_attempted",
    "threePointersPercentage": "three_pointers_percentage",
    "freeThrowsMade": "free_throws_made",
    "freeThrowsAttempted": "free_throws_attempted",
    "freeThrowsPercentage": "free_throws_percentage",
    "reboundsOffensive": "rebounds_offensive",
    "reboundsDefensive": "rebounds_defensive",
    "reboundsTotal": "rebounds_total",
    "foulsPersonal": "fouls_personal",
    "plusMinusPoints": "plus_minus_points",
}

TEAM_RENAME_MAP = {
    "gameId": "game_id",
    "teamId": "team_id",
    "teamCity": "team_city",
    "teamName": "team_name",
    "teamTricode": "team_tricode",
    "teamSlug": "team_slug",
    "fieldGoalsMade": "field_goals_made",
    "fieldGoalsAttempted": "field_goals_attempted",
    "fieldGoalsPercentage": "field_goals_percentage",
    "threePointersMade": "three_pointers_made",
    "threePointersAttempted": "three_pointers_attempted",
    "threePointersPercentage": "three_pointers_percentage",
    "freeThrowsMade": "free_throws_made",
    "freeThrowsAttempted": "free_throws_attempted",
    "freeThrowsPercentage": "free_throws_percentage",
    "reboundsOffensive": "rebounds_offensive",
    "reboundsDefensive": "rebounds_defensive",
    "reboundsTotal": "rebounds_total",
    "foulsPersonal": "fouls_personal",
    "plusMinusPoints": "plus_minus_points",
}

STARTER_BENCH_RENAME_MAP = {
    "gameId": "game_id",
    "teamId": "team_id",
    "teamCity": "team_city",
    "teamName": "team_name",
    "teamTricode": "team_tricode",
    "teamSlug": "team_slug",
    "fieldGoalsMade": "field_goals_made",
    "fieldGoalsAttempted": "field_goals_attempted",
    "fieldGoalsPercentage": "field_goals_percentage",
    "threePointersMade": "three_pointers_made",
    "threePointersAttempted": "three_pointers_attempted",
    "threePointersPercentage": "three_pointers_percentage",
    "freeThrowsMade": "free_throws_made",
    "freeThrowsAttempted": "free_throws_attempted",
    "freeThrowsPercentage": "free_throws_percentage",
    "reboundsOffensive": "rebounds_offensive",
    "reboundsDefensive": "rebounds_defensive",
    "reboundsTotal": "rebounds_total",
    "foulsPersonal": "fouls_personal",
    "startersBench": "starters_bench",
}

GAME_META_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "game_type",
    "neutral_site",
    "team_id_home",
    "team_abbreviation_home",
    "team_name_home",
    "team_id_road",
    "team_abbreviation_road",
    "team_name_road",
]

PLAYER_DEDUPE_KEYS = ["game_id", "team_id", "person_id"]
TEAM_DEDUPE_KEYS = ["game_id", "team_id"]
STARTER_BENCH_DEDUPE_KEYS = ["game_id", "team_id", "starters_bench"]


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


def _season_start_year(season: str) -> int:
    match = re.match(r"^(\d{4})-(\d{2})$", season.strip())
    if not match:
        raise ValueError(f"Invalid season format: {season!r}. Expected YYYY-YY.")
    return int(match.group(1))


def _season_label_from_start_year(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _season_range(start_season: str, end_season: str) -> list[str]:
    start_year = _season_start_year(start_season)
    end_year = _season_start_year(end_season)
    if end_year < start_year:
        raise ValueError("--end must be the same season or later than --start.")
    return [_season_label_from_start_year(year) for year in range(start_year, end_year + 1)]


def _players_path(repo_dir: Path, season: str) -> Path:
    return get_canonical_data_file_path(
        build_box_score_traditional_filename("players", season),
        repo_dir=repo_dir,
    )


def _teams_path(repo_dir: Path, season: str) -> Path:
    return get_canonical_data_file_path(
        build_box_score_traditional_filename("teams", season),
        repo_dir=repo_dir,
    )


def _starter_bench_path(repo_dir: Path, season: str) -> Path:
    return get_canonical_data_file_path(
        build_box_score_traditional_filename("starter_bench", season),
        repo_dir=repo_dir,
    )


def _load_game_metadata(repo_dir: Path, season: str, include_today: bool) -> pd.DataFrame:
    path = resolve_data_file_path(
        build_data_filename("team_game_logs", season),
        repo_dir=repo_dir,
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing team game logs for {season}: {path}")

    df = pd.read_csv(path, dtype={"game_id": "string"})
    if df.empty:
        raise RuntimeError(f"team_game_logs_{season}.csv is empty.")

    d = df.copy()
    d["game_id"] = d["game_id"].map(_normalize_game_id)
    d["game_date"] = pd.to_datetime(
        d["game_date"].astype("string").str.strip(),
        format="%Y-%m-%d",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    d = d[d["game_id"] != ""].copy()

    if not include_today:
        today_str = datetime.now().strftime("%Y-%m-%d")
        d = d[d["game_date"] != today_str].copy()

    d = d[GAME_META_COLUMNS].drop_duplicates(subset=["game_id"], keep="last")
    d = d.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)
    return d


def _read_existing_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, dtype={"game_id": "string"}, usecols=["game_id"])
    except Exception:
        return set()
    return {gid for gid in df["game_id"].map(_normalize_game_id).tolist() if gid}


def _complete_existing_game_ids(players_path: Path, teams_path: Path, starter_bench_path: Path) -> set[str]:
    if not (players_path.exists() and teams_path.exists() and starter_bench_path.exists()):
        return set()
    return (
        _read_existing_game_ids(players_path)
        & _read_existing_game_ids(teams_path)
        & _read_existing_game_ids(starter_bench_path)
    )


def _remaining_game_ids_for_season(
    season: str,
    repo_dir: Path,
    include_today: bool,
    force_refresh: bool,
    max_games: Optional[int] = None,
    start_offset: int = 0,
) -> list[str]:
    game_meta = _load_game_metadata(repo_dir, season, include_today=include_today)
    candidate_game_ids = game_meta["game_id"].tolist()
    if force_refresh:
        remaining_game_ids = candidate_game_ids
    else:
        remaining_game_ids = [
            game_id
            for game_id in candidate_game_ids
            if game_id not in _complete_existing_game_ids(
                _players_path(repo_dir, season),
                _teams_path(repo_dir, season),
                _starter_bench_path(repo_dir, season),
            )
        ]
    if max_games is not None:
        remaining_game_ids = remaining_game_ids[:max_games]
    if remaining_game_ids and start_offset:
        offset = start_offset % len(remaining_game_ids)
        if offset:
            remaining_game_ids = remaining_game_ids[offset:] + remaining_game_ids[:offset]
    return remaining_game_ids


def _merge_game_metadata(df: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["game_id"] = d["game_id"].map(_normalize_game_id)
    merged = d.merge(game_meta, on="game_id", how="left")
    if "team_id" in merged.columns:
        team_id_numeric = pd.to_numeric(merged["team_id"], errors="coerce")
        home_team_numeric = pd.to_numeric(merged["team_id_home"], errors="coerce")
        road_team_numeric = pd.to_numeric(merged["team_id_road"], errors="coerce")
        merged["home_or_road"] = pd.Series(pd.NA, index=merged.index, dtype="string")
        merged.loc[team_id_numeric == home_team_numeric, "home_or_road"] = "home"
        merged.loc[team_id_numeric == road_team_numeric, "home_or_road"] = "road"
    return merged


def _normalize_numeric_columns(df: pd.DataFrame, integer_columns: list[str], float_columns: list[str]) -> pd.DataFrame:
    d = df.copy()
    for column in integer_columns:
        if column in d.columns:
            d[column] = pd.to_numeric(d[column], errors="coerce").astype("Int64")
    for column in float_columns:
        if column in d.columns:
            d[column] = pd.to_numeric(d[column], errors="coerce").astype("float64")
    return d


def _normalize_minutes_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["game_id"] = d["game_id"].astype("string")
    id_columns = [
        "team_id",
        "person_id",
        "team_id_home",
        "team_id_road",
    ]
    for column in id_columns:
        if column in d.columns:
            d[column] = pd.to_numeric(d[column], errors="coerce").astype("Int64")
    if "neutral_site" in d.columns:
        d["neutral_site"] = d["neutral_site"].astype("boolean")
    for column in ("minutes", "comment", "position", "jersey_num", "name_i", "first_name", "family_name", "player_slug", "team_slug", "team_city", "team_name", "team_tricode", "starters_bench", "game_type", "home_or_road"):
        if column in d.columns:
            d[column] = d[column].astype("string")
    return d


def _sort_columns(df: pd.DataFrame, leading_columns: list[str]) -> pd.DataFrame:
    remaining = [column for column in df.columns if column not in leading_columns]
    return df[leading_columns + remaining]


def _normalize_player_rows(player_df: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    d = player_df.rename(columns=PLAYER_RENAME_MAP).copy()
    d = _merge_game_metadata(d, game_meta)
    d = _normalize_numeric_columns(
        d,
        integer_columns=[
            "team_id",
            "person_id",
            "field_goals_made",
            "field_goals_attempted",
            "three_pointers_made",
            "three_pointers_attempted",
            "free_throws_made",
            "free_throws_attempted",
            "rebounds_offensive",
            "rebounds_defensive",
            "rebounds_total",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "fouls_personal",
            "points",
            "plus_minus_points",
        ],
        float_columns=[
            "field_goals_percentage",
            "three_pointers_percentage",
            "free_throws_percentage",
        ],
    )
    d = _normalize_minutes_and_ids(d)
    d = d.drop_duplicates(subset=PLAYER_DEDUPE_KEYS, keep="last")
    d = d.sort_values(["game_date", "game_id", "home_or_road", "team_id", "person_id"], kind="stable")
    return _sort_columns(
        d,
        [
            "game_id",
            "game_date",
            "season",
            "game_type",
            "home_or_road",
            "neutral_site",
            "team_id",
            "team_tricode",
            "team_name",
            "person_id",
            "first_name",
            "family_name",
            "name_i",
            "position",
            "minutes",
            "points",
            "plus_minus_points",
        ],
    )


def _normalize_team_rows(team_df: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    d = team_df.rename(columns=TEAM_RENAME_MAP).copy()
    d = _merge_game_metadata(d, game_meta)
    d = _normalize_numeric_columns(
        d,
        integer_columns=[
            "team_id",
            "field_goals_made",
            "field_goals_attempted",
            "three_pointers_made",
            "three_pointers_attempted",
            "free_throws_made",
            "free_throws_attempted",
            "rebounds_offensive",
            "rebounds_defensive",
            "rebounds_total",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "fouls_personal",
            "points",
            "plus_minus_points",
        ],
        float_columns=[
            "field_goals_percentage",
            "three_pointers_percentage",
            "free_throws_percentage",
        ],
    )
    d = _normalize_minutes_and_ids(d)
    d = d.drop_duplicates(subset=TEAM_DEDUPE_KEYS, keep="last")
    d = d.sort_values(["game_date", "game_id", "home_or_road", "team_id"], kind="stable")
    return _sort_columns(
        d,
        [
            "game_id",
            "game_date",
            "season",
            "game_type",
            "home_or_road",
            "neutral_site",
            "team_id",
            "team_tricode",
            "team_name",
            "minutes",
            "points",
            "plus_minus_points",
        ],
    )


def _normalize_starter_bench_rows(starter_bench_df: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    d = starter_bench_df.rename(columns=STARTER_BENCH_RENAME_MAP).copy()
    d = _merge_game_metadata(d, game_meta)
    d = _normalize_numeric_columns(
        d,
        integer_columns=[
            "team_id",
            "field_goals_made",
            "field_goals_attempted",
            "three_pointers_made",
            "three_pointers_attempted",
            "free_throws_made",
            "free_throws_attempted",
            "rebounds_offensive",
            "rebounds_defensive",
            "rebounds_total",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "fouls_personal",
            "points",
        ],
        float_columns=[
            "field_goals_percentage",
            "three_pointers_percentage",
            "free_throws_percentage",
        ],
    )
    d = _normalize_minutes_and_ids(d)
    d = d.drop_duplicates(subset=STARTER_BENCH_DEDUPE_KEYS, keep="last")
    d = d.sort_values(["game_date", "game_id", "home_or_road", "team_id", "starters_bench"], kind="stable")
    return _sort_columns(
        d,
        [
            "game_id",
            "game_date",
            "season",
            "game_type",
            "home_or_road",
            "neutral_site",
            "team_id",
            "team_tricode",
            "team_name",
            "starters_bench",
            "minutes",
            "points",
        ],
    )


def _load_existing_output(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"game_id": "string"})


def _is_timeout_like_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    return "timed out" in text or "read timeout" in text or "too many requests" in text or "429" in text


def _retry_sleep_seconds(attempt: int, error_text: str, pause_seconds: float) -> float:
    if _is_timeout_like_error(error_text):
        return max(pause_seconds * attempt, min(20.0, 5.0 * attempt))
    return max(0.5, pause_seconds * attempt)


def _write_combined_output(
    path: Path,
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    dedupe_keys: list[str],
    sort_columns: list[str],
) -> pd.DataFrame:
    if existing_df.empty:
        combined = new_df.copy()
    elif new_df.empty:
        combined = existing_df.copy()
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
                category=FutureWarning,
            )
            combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)

    if combined.empty:
        return combined

    combined["game_id"] = combined["game_id"].map(_normalize_game_id)
    combined = combined.drop_duplicates(subset=dedupe_keys, keep="last")
    combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def _fetch_boxscoretraditional_v3(
    game_id: str,
    timeout_seconds: float,
    retries: int,
    retry_pause_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=timeout_seconds)
            player_df = response.player_stats.get_data_frame()
            starter_bench_df = response.team_starter_bench_stats.get_data_frame()
            team_df = response.team_stats.get_data_frame()
            if player_df.empty or starter_bench_df.empty or team_df.empty:
                raise RuntimeError(
                    f"Incomplete response for {game_id}: "
                    f"player_rows={len(player_df)} starter_bench_rows={len(starter_bench_df)} team_rows={len(team_df)}"
                )
            return player_df, team_df, starter_bench_df
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(_retry_sleep_seconds(attempt, str(exc), retry_pause_seconds))
    raise RuntimeError(f"BoxScoreTraditionalV3 failed for {game_id}: {last_error}") from last_error


def download_season(
    season: str,
    repo_dir: Path,
    include_today: bool,
    force_refresh: bool,
    max_games: Optional[int],
    timeout_seconds: float,
    retries: int,
    pause_seconds: float,
    save_every: int,
    rest_every_games: int,
    rest_seconds: float,
    max_timeout_cooldowns: int,
    start_offset: int,
    dry_run: bool,
) -> int:
    start_time = time.time()
    game_meta = _load_game_metadata(repo_dir, season, include_today=include_today)

    players_path = _players_path(repo_dir, season)
    teams_path = _teams_path(repo_dir, season)
    starter_bench_path = _starter_bench_path(repo_dir, season)

    candidate_game_ids = game_meta["game_id"].tolist()
    remaining_game_ids = _remaining_game_ids_for_season(
        season=season,
        repo_dir=repo_dir,
        include_today=include_today,
        force_refresh=force_refresh,
        max_games=max_games,
        start_offset=start_offset,
    )

    print(f"[traditional_v3] {season}: {len(candidate_game_ids)} candidate games, {len(remaining_game_ids)} to fetch")
    if remaining_game_ids and start_offset:
        print(f"[traditional_v3] {season}: starting at rotated offset {start_offset % len(remaining_game_ids)} within the remaining-game queue")
    print(f"[traditional_v3] outputs:")
    print(f"  {players_path.name}")
    print(f"  {teams_path.name}")
    print(f"  {starter_bench_path.name}")

    if dry_run:
        return 0

    if not remaining_game_ids:
        print(f"[traditional_v3] {season}: nothing to do")
        return 0

    existing_players = _load_existing_output(players_path)
    existing_teams = _load_existing_output(teams_path)
    existing_starter_bench = _load_existing_output(starter_bench_path)

    player_buffer: list[pd.DataFrame] = []
    team_buffer: list[pd.DataFrame] = []
    starter_bench_buffer: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    fetched_games = 0
    consecutive_timeout_failures = 0
    timeout_cooldowns_used = 0
    aborted_for_throttling = False

    for index, game_id in enumerate(remaining_game_ids, start=1):
        print(f"  [{index}/{len(remaining_game_ids)}] {game_id}", end=" ", flush=True)
        try:
            raw_player_df, raw_team_df, raw_starter_bench_df = _fetch_boxscoretraditional_v3(
                game_id=game_id,
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_pause_seconds=pause_seconds,
            )
            player_buffer.append(_normalize_player_rows(raw_player_df, game_meta))
            team_buffer.append(_normalize_team_rows(raw_team_df, game_meta))
            starter_bench_buffer.append(_normalize_starter_bench_rows(raw_starter_bench_df, game_meta))
            fetched_games += 1
            consecutive_timeout_failures = 0
            print(
                f"OK players={len(raw_player_df)} teams={len(raw_team_df)} starter_bench={len(raw_starter_bench_df)}"
            )
            if (
                rest_every_games > 0
                and fetched_games % rest_every_games == 0
                and index < len(remaining_game_ids)
            ):
                print(f"  [rest] fetched {fetched_games} game(s); sleeping {rest_seconds:.0f}s to stay under throttling")
                time.sleep(rest_seconds)
        except Exception as exc:
            error_text = str(exc)
            failures.append((game_id, error_text))
            print(f"FAIL {error_text}")
            if _is_timeout_like_error(error_text):
                consecutive_timeout_failures += 1
                if consecutive_timeout_failures >= TIMEOUT_COOLDOWN_THRESHOLD:
                    timeout_cooldowns_used += 1
                    print(
                        f"  [cooldown] {consecutive_timeout_failures} consecutive timeout-like failures; "
                        f"sleeping {TIMEOUT_COOLDOWN_SECONDS:.0f}s before continuing"
                    )
                    time.sleep(TIMEOUT_COOLDOWN_SECONDS)
                    consecutive_timeout_failures = 0
                    if timeout_cooldowns_used >= max_timeout_cooldowns:
                        print(
                            f"  [abort] throttling persisted after {timeout_cooldowns_used} cooldowns; "
                            "stopping early so a rerun can resume from saved progress"
                        )
                        aborted_for_throttling = True
                        break
            else:
                consecutive_timeout_failures = 0

        should_flush = index % save_every == 0 or index == len(remaining_game_ids)
        if should_flush and player_buffer:
            print(f"  [save] writing progress through {index}/{len(remaining_game_ids)}")
            existing_players = _write_combined_output(
                players_path,
                existing_players,
                pd.concat(player_buffer, ignore_index=True, sort=False),
                dedupe_keys=PLAYER_DEDUPE_KEYS,
                sort_columns=["game_date", "game_id", "home_or_road", "team_id", "person_id"],
            )
            existing_teams = _write_combined_output(
                teams_path,
                existing_teams,
                pd.concat(team_buffer, ignore_index=True, sort=False),
                dedupe_keys=TEAM_DEDUPE_KEYS,
                sort_columns=["game_date", "game_id", "home_or_road", "team_id"],
            )
            existing_starter_bench = _write_combined_output(
                starter_bench_path,
                existing_starter_bench,
                pd.concat(starter_bench_buffer, ignore_index=True, sort=False),
                dedupe_keys=STARTER_BENCH_DEDUPE_KEYS,
                sort_columns=["game_date", "game_id", "home_or_road", "team_id", "starters_bench"],
            )
            player_buffer = []
            team_buffer = []
            starter_bench_buffer = []

        if index < len(remaining_game_ids):
            time.sleep(pause_seconds)

    if player_buffer:
        print("  [save] writing final buffered rows")
        existing_players = _write_combined_output(
            players_path,
            existing_players,
            pd.concat(player_buffer, ignore_index=True, sort=False),
            dedupe_keys=PLAYER_DEDUPE_KEYS,
            sort_columns=["game_date", "game_id", "home_or_road", "team_id", "person_id"],
        )
        existing_teams = _write_combined_output(
            teams_path,
            existing_teams,
            pd.concat(team_buffer, ignore_index=True, sort=False),
            dedupe_keys=TEAM_DEDUPE_KEYS,
            sort_columns=["game_date", "game_id", "home_or_road", "team_id"],
        )
        existing_starter_bench = _write_combined_output(
            starter_bench_path,
            existing_starter_bench,
            pd.concat(starter_bench_buffer, ignore_index=True, sort=False),
            dedupe_keys=STARTER_BENCH_DEDUPE_KEYS,
            sort_columns=["game_date", "game_id", "home_or_road", "team_id", "starters_bench"],
        )

    elapsed = time.time() - start_time
    print(f"[traditional_v3] {season} complete")
    print(f"  fetched games: {fetched_games}")
    print(f"  failed games: {len(failures)}")
    print(f"  players rows: {len(existing_players)}")
    print(f"  teams rows: {len(existing_teams)}")
    print(f"  starter/bench rows: {len(existing_starter_bench)}")
    print(f"  elapsed: {elapsed:.1f}s")
    if aborted_for_throttling:
        print("  status: stopped early due to persistent throttling; rerun the same command to resume")
        return 2
    if failures:
        print("  failures:")
        for game_id, error_text in failures[:20]:
            print(f"    {game_id}: {error_text}")
        if len(failures) > 20:
            print(f"    ... {len(failures) - 20} more")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download season-level BoxScoreTraditionalV3 data into NBA_Data CSVs."
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=str(DEFAULT_REPO_DIR),
        help=f"NBA_Data repo root (default: {DEFAULT_REPO_DIR})",
    )
    parser.add_argument("--season", help="Single season like 2025-26")
    parser.add_argument("--start", help="Start season like 2000-01")
    parser.add_argument("--end", help="End season like 2025-26")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refetch games even when all three output CSVs already contain that game_id.",
    )
    parser.add_argument(
        "--include-today",
        action="store_true",
        help="Include games dated today. Default skips today to avoid in-progress box scores.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on games fetched per season.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=API_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds (default: {API_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=REQUEST_RETRIES,
        help=f"Attempts per game before failing (default: {REQUEST_RETRIES})",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=REQUEST_PAUSE_SECONDS,
        help=f"Pause between game requests (default: {REQUEST_PAUSE_SECONDS})",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY_GAMES,
        help=f"Save progress every N fetched games (default: {SAVE_EVERY_GAMES})",
    )
    parser.add_argument(
        "--rest-every-games",
        type=int,
        default=BATCH_REST_EVERY_GAMES,
        help=f"Sleep briefly after every N successful games (default: {BATCH_REST_EVERY_GAMES}, 0 disables)",
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=BATCH_REST_SECONDS,
        help=f"Cooldown applied at each rest checkpoint (default: {BATCH_REST_SECONDS})",
    )
    parser.add_argument(
        "--max-timeout-cooldowns",
        type=int,
        default=MAX_TIMEOUT_COOLDOWNS_PER_RUN,
        help=(
            "Stop the run early after this many timeout cooldown cycles so a rerun can resume later "
            f"(default: {MAX_TIMEOUT_COOLDOWNS_PER_RUN})"
        ),
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Keep retrying throttled/incomplete seasons automatically until no games remain, then continue to the next season.",
    )
    parser.add_argument(
        "--resume-sleep-seconds",
        type=float,
        default=AUTO_RESUME_SLEEP_SECONDS,
        help=f"Sleep between auto-resume passes for the same season (default: {AUTO_RESUME_SLEEP_SECONDS})",
    )
    parser.add_argument(
        "--max-auto-resume-passes",
        type=int,
        default=MAX_AUTO_RESUME_PASSES,
        help=f"Maximum auto-resume passes per season (default: {MAX_AUTO_RESUME_PASSES})",
    )
    parser.add_argument(
        "--between-seasons-sleep-seconds",
        type=float,
        default=BETWEEN_SEASONS_SLEEP_SECONDS,
        help=(
            "Sleep after finishing one season before starting the next season in a multi-season run "
            f"(default: {BETWEEN_SEASONS_SLEEP_SECONDS})"
        ),
    )
    parser.add_argument(
        "--resume-offset-step",
        type=int,
        default=AUTO_RESUME_OFFSET_STEP,
        help=(
            "When auto-resume is enabled, rotate the remaining-game queue by this many positions on each pass "
            f"(default: {AUTO_RESUME_OFFSET_STEP})"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned work without hitting the API.")
    return parser


def _seasons_from_args(args: argparse.Namespace) -> list[str]:
    has_single = bool(args.season)
    has_range = bool(args.start or args.end)
    if has_single and has_range:
        raise ValueError("Use either --season or --start/--end, not both.")
    if has_single:
        return [args.season]
    if args.start and args.end:
        return _season_range(args.start, args.end)
    raise ValueError("Provide --season or both --start and --end.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_dir = Path(args.repo_dir).resolve()

    try:
        seasons = _seasons_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.auto_resume and args.dry_run:
        parser.error("--auto-resume cannot be combined with --dry-run.")

    overall_status = 0
    for season_index, season in enumerate(seasons):
        if not args.auto_resume:
            status = download_season(
                season=season,
                repo_dir=repo_dir,
                include_today=args.include_today,
                force_refresh=args.force_refresh,
                max_games=args.max_games,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
                pause_seconds=args.pause_seconds,
                save_every=max(1, args.save_every),
                rest_every_games=max(0, args.rest_every_games),
                rest_seconds=max(0.0, args.rest_seconds),
                max_timeout_cooldowns=max(1, args.max_timeout_cooldowns),
                start_offset=0,
                dry_run=args.dry_run,
            )
            if status != 0:
                overall_status = status
            continue

        passes = 0
        previous_remaining: Optional[int] = None
        season_done = False

        while passes < max(1, args.max_auto_resume_passes):
            start_offset = max(0, passes - 1) * max(0, args.resume_offset_step)
            remaining_before = len(
                _remaining_game_ids_for_season(
                    season=season,
                    repo_dir=repo_dir,
                    include_today=args.include_today,
                    force_refresh=args.force_refresh,
                    max_games=args.max_games,
                    start_offset=start_offset,
                )
            )
            if remaining_before == 0:
                print(f"[traditional_v3] {season}: already complete")
                season_done = True
                break

            passes += 1
            print(
                f"[traditional_v3] {season}: auto-resume pass {passes} "
                f"starting with {remaining_before} remaining game(s) and offset {start_offset}"
            )
            status = download_season(
                season=season,
                repo_dir=repo_dir,
                include_today=args.include_today,
                force_refresh=args.force_refresh,
                max_games=args.max_games,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
                pause_seconds=args.pause_seconds,
                save_every=max(1, args.save_every),
                rest_every_games=max(0, args.rest_every_games),
                rest_seconds=max(0.0, args.rest_seconds),
                max_timeout_cooldowns=max(1, args.max_timeout_cooldowns),
                start_offset=start_offset,
                dry_run=args.dry_run,
            )
            remaining_after = len(
                _remaining_game_ids_for_season(
                    season=season,
                    repo_dir=repo_dir,
                    include_today=args.include_today,
                    force_refresh=args.force_refresh,
                    max_games=args.max_games,
                    start_offset=start_offset,
                )
            )
            print(
                f"[traditional_v3] {season}: auto-resume pass {passes} finished "
                f"with {remaining_after} remaining game(s)"
            )

            if remaining_after == 0:
                season_done = True
                break

            if previous_remaining is not None and remaining_after >= previous_remaining and status != 2:
                print(
                    f"[traditional_v3] {season}: no progress on the latest pass and status={status}; "
                    "stopping auto-resume for this season"
                )
                overall_status = max(overall_status, 1)
                break

            previous_remaining = remaining_after
            sleep_seconds = max(0.0, args.resume_sleep_seconds)
            if sleep_seconds > 0 and passes < max(1, args.max_auto_resume_passes):
                print(
                    f"[traditional_v3] {season}: sleeping {sleep_seconds:.0f}s before the next auto-resume pass"
                )
                time.sleep(sleep_seconds)

        if not season_done:
            overall_status = max(overall_status, 1)
            continue

        is_last_season = season_index == len(seasons) - 1
        if args.auto_resume and not is_last_season:
            between_sleep = max(0.0, args.between_seasons_sleep_seconds)
            if between_sleep > 0:
                next_season = seasons[season_index + 1]
                print(
                    f"[traditional_v3] {season}: complete; sleeping {between_sleep:.0f}s before starting {next_season}"
                )
                time.sleep(between_sleep)

    return overall_status


if __name__ == "__main__":
    raise SystemExit(main())
