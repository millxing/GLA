from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from typing import Optional, Dict, Any
import hmac
import math
import subprocess
import json
import re
import sys
import logging
import os
import time
from copy import deepcopy
import pandas as pd
from importlib.metadata import version as pkg_version
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import pyarrow.parquet as pq
from config import (
    ADMIN_SECRET_KEY,
    PBP_ENABLE_LEGACY_GLA_FALLBACK,
    PBP_GAME_STATES_ROOT,
    PBP_GITHUB_RAW_BASE_URL,
    PBP_LEGACY_GLA_ROOT,
    PBP_REMOTE_CACHE_DIR,
    get_available_seasons,
)
from services.cache import clear_cache
from services.data_loader import (
    get_normalized_season_data,
    get_normalized_data_with_possessions,
    get_games_list,
    get_teams_list,
    load_contributions,
    normalize_data_scope,
)
from services.calculations import (
    compute_league_aggregates,
    compute_league_summary_averages,
    compute_scope_time_metrics,
    compute_trend_series,
    compute_league_average,
    compute_contribution_analysis,
    compute_league_top_contributors,
)
from services.llm import (
    _build_interpretation_prompt,
    generate_interpretation,
    get_runtime_interpretation_model,
    is_llm_configured,
)
from services.game_runs import extract_timeline_possessions, rank_non_overlapping_runs
from services.pbp_boxscore import (
    clear_pbp_boxscore_cache,
    compute_pbp_traditional_boxscore,
    normalize_boxscore_segment,
)
from services.player_game_facts import build_player_game_facts_payload
from services.player_shots import (
    PLAYER_SHOT_CLASSIFICATIONS,
    PLAYER_SHOTS_GAME_TYPES,
    PLAYER_SHOT_TYPES,
    build_player_shot_streakiness_payload,
    build_player_shots_payload,
    list_player_shot_players,
)
from services.data_loader import get_game_interpretation
from schemas.models import (
    SeasonResponse,
    GamesResponse,
    GameItem,
    TeamsResponse,
    DecompositionResponse,
    LeagueSummaryResponse,
    TeamStats,
    TrendsResponse,
    TrendPoint,
    LinescoreData,
    QuarterScores,
    ContributionAnalysisResponse,
    TopContributor,
    ContributionTrendPoint,
    LeagueContributorItem,
    LeagueTopContributorsResponse,
    InterpretationRequest,
    InterpretationResponse,
    GameTimelineResponse,
    GameTimelineEvent,
    GameTimelineState,
    GameRunsResponse,
    GameRun,
    PBPTraditionalBoxScoreResponse,
    PlayerGameFactsResponse,
    PlayerShotsResponse,
    PlayerShotPlayersResponse,
    PlayerShotStreakinessResponse,
)

_SEASON_RE = re.compile(r"^\d{4}-\d{2}$")
_GAME_ID_RE = re.compile(r"^\d{1,10}$")
_TEAM_RE = re.compile(r"^[A-Z]{3}$")
_VALID_SEASONS = set(get_available_seasons())


def validate_season(season: str) -> str:
    """Validate season format (YYYY-YY) and existence. Returns the season or raises 400."""
    if not _SEASON_RE.match(season):
        raise HTTPException(status_code=400, detail="Invalid season format. Expected YYYY-YY (e.g. 2024-25)")
    if season not in _VALID_SEASONS:
        raise HTTPException(status_code=400, detail=f"Season {season} is not available")
    return season


def validate_game_id(game_id: str) -> str:
    """Validate game ID is numeric (1-10 digits). Returns the game_id or raises 400."""
    if not _GAME_ID_RE.match(game_id):
        raise HTTPException(status_code=400, detail="Invalid game ID. Must be numeric (up to 10 digits)")
    return game_id


def validate_team(team: str) -> str:
    """Validate team abbreviation is 3 uppercase letters. Returns the team or raises 400."""
    if not _TEAM_RE.match(team):
        raise HTTPException(status_code=400, detail="Invalid team abbreviation. Expected 3 uppercase letters (e.g. BOS)")
    return team


STAT_ALIASES = {
    "ORTG": "off_rating",
    "DRTG": "def_rating",
    "NET_RTG": "net_rating",
    "EFG": "efg_pct",
    "EFG%": "efg_pct",
    "FG2P": "fg2p",
    "FG3P": "fg3p",
    "PACE": "pace",
}

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)
WINPROB_APP_PATH = (Path(__file__).resolve().parents[1] / "winprob_wizard_app.html").resolve()
WINPROB_HYPOTHETICAL_APP_PATH = (Path(__file__).resolve().parents[1] / "winprob_hypothetical_app.html").resolve()
PLAYER_SHOT_SEQUENCES_APP_PATH = (Path(__file__).resolve().parents[1] / "player_shot_sequences_app.html").resolve()
STATES_PARQUET_FILENAME_TEMPLATE = "_states_{season}_{phase}.parquet"
TIMELINE_METRICS_FILENAME_TEMPLATE = "_timeline_metrics_{season}_{phase}.json"
HOME_WIN_PROB_BY_EVENT_JSON_COLUMN = "home_win_prob_by_event_json"
LEGACY_PBP_GAME_STATES_ROOT = (PBP_LEGACY_GLA_ROOT / "game_states").resolve()
REMOTE_CACHE_TTL_SECONDS = max(
    60,
    int(os.getenv("PBP_REMOTE_CACHE_TTL_SECONDS", "300") or "300"),
)
REMOTE_FETCH_TIMEOUT_SECONDS = max(
    10,
    int(os.getenv("PBP_REMOTE_FETCH_TIMEOUT_SECONDS", "60") or "60"),
)

LEAGUE_SUMMARY_SCOPE_ALIASES = {
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
    "h2_incl_ot": "h2",
    "h2_including_ot": "h2",
    "second_half_incl_ot": "h2",
    "second_half_including_ot": "h2",
    "non_garbage_time": "garbage_filtered",
    "non_garbage": "garbage_filtered",
    "garbage_filtered": "garbage_filtered",
    "garbage_time": "garbage_time",
    "garbage": "garbage_time",
    "clutch": "clutch",
    "clutch_time": "clutch",
    "non_clutch_time": "non_clutch_time",
    "non_clutch": "non_clutch_time",
}
LEAGUE_SUMMARY_VALID_SCOPES = {
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
    "non_clutch_time",
}
LEAGUE_SUMMARY_PERSISTED_SCOPES = {
    "q1",
    "q2",
    "q3",
    "q4",
    "ot",
    "h1",
    "h2",
    "garbage_filtered",
    "clutch",
}
INTERPRETATION_ENABLED_SEASON = "2025-26"
INTERPRETATION_ENABLED_SCOPE = "all"

PBP_GAME_STATES_ROOTS = [PBP_GAME_STATES_ROOT]
if PBP_ENABLE_LEGACY_GLA_FALLBACK and LEGACY_PBP_GAME_STATES_ROOT != PBP_GAME_STATES_ROOT:
    PBP_GAME_STATES_ROOTS.append(LEGACY_PBP_GAME_STATES_ROOT)


def _normalize_league_summary_scope(value: Optional[str]) -> str:
    text = str(value or "all").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = LEAGUE_SUMMARY_SCOPE_ALIASES.get(text, text)
    if normalized not in LEAGUE_SUMMARY_VALID_SCOPES:
        raise ValueError(
            f"Invalid league summary data_scope: {value!r}. "
            f"Expected one of: {', '.join(sorted(LEAGUE_SUMMARY_VALID_SCOPES))}"
        )
    return normalized


def _normalize_scope_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_id"] = (
        out["game_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )
    out["team"] = out["team"].astype(str).str.strip()
    out["home_away"] = out["home_away"].astype(str).str.strip().str.lower()
    return out


def _should_clip_scope_delta(column: str) -> bool:
    if "plus_minus" in column:
        return False
    base_count_stats = {"fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts"}
    if column in base_count_stats:
        return True
    if column.startswith("opp_") and column[4:] in base_count_stats:
        return True
    if column in {"actual_poss", "opp_actual_poss", "actual_minutes", "opp_actual_minutes"}:
        return True
    return False


def _derive_scope_complement_df(all_df: pd.DataFrame, included_df: pd.DataFrame) -> pd.DataFrame:
    """Return segment dataframe computed as all_df - included_df at team/game row level."""
    if all_df is None or all_df.empty:
        return pd.DataFrame(columns=(all_df.columns if all_df is not None else []))

    base = _normalize_scope_key_columns(all_df)
    if included_df is None or included_df.empty:
        return base

    inc = _normalize_scope_key_columns(included_df)
    key_cols = ["game_id", "team", "home_away"]
    numeric_cols = [
        c for c in base.columns
        if c in inc.columns and c not in key_cols and pd.api.types.is_numeric_dtype(base[c])
    ]
    if not numeric_cols:
        return base

    inc_numeric = inc[key_cols + numeric_cols].copy().rename(columns={c: f"__inc_{c}" for c in numeric_cols})
    merged = base.merge(inc_numeric, on=key_cols, how="left")

    for col in numeric_cols:
        left = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        right = pd.to_numeric(merged[f"__inc_{col}"], errors="coerce").fillna(0.0)
        delta = left - right
        if _should_clip_scope_delta(col):
            delta = delta.clip(lower=0.0)
        merged[col] = delta

    merged = merged.drop(columns=[f"__inc_{c}" for c in numeric_cols], errors="ignore")
    return merged[all_df.columns]


async def _resolve_league_summary_scope_df(season: str, scope: str) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Resolve selected League Summary scope into a normalized dataframe and the all-data baseline."""
    all_df = await get_normalized_data_with_possessions(season, data_scope="all")
    if all_df is None:
        return None, None

    if scope == "all":
        return all_df, all_df

    if scope in LEAGUE_SUMMARY_PERSISTED_SCOPES:
        scoped = await get_normalized_data_with_possessions(season, data_scope=scope)
        return scoped, all_df

    if scope == "garbage_time":
        non_garbage_df = await get_normalized_data_with_possessions(season, data_scope="garbage_filtered")
        if non_garbage_df is None:
            return None, all_df
        return _derive_scope_complement_df(all_df, non_garbage_df), all_df

    if scope == "non_clutch_time":
        clutch_df = await get_normalized_data_with_possessions(season, data_scope="clutch")
        if clutch_df is None:
            return None, all_df
        return _derive_scope_complement_df(all_df, clutch_df), all_df

    return None, all_df


def _download_remote_pbpdata_file(relative_path: str) -> Optional[Path]:
    rel = relative_path.lstrip("/")
    cache_path = (PBP_REMOTE_CACHE_DIR / rel).resolve()
    cache_exists = cache_path.exists()
    if cache_exists:
        try:
            age_seconds = max(0, int(time.time() - cache_path.stat().st_mtime))
            if age_seconds < REMOTE_CACHE_TTL_SECONDS:
                return cache_path
        except Exception:
            # If mtime checks fail, conservatively use cache to avoid hard failure.
            return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    remote_url = f"{PBP_GITHUB_RAW_BASE_URL}/{rel}"
    req = Request(remote_url, headers={"User-Agent": "GLA-timeline-fallback"})
    try:
        with urlopen(req, timeout=REMOTE_FETCH_TIMEOUT_SECONDS) as resp:
            payload = resp.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        logger.warning(
            "Timeline remote fallback failed: relative_path=%s url=%s",
            rel,
            remote_url,
        )
        # If refresh failed but a stale cache exists, prefer stale over missing.
        return cache_path if cache_exists else None

    if not payload:
        logger.warning(
            "Timeline remote fallback returned empty payload: relative_path=%s url=%s",
            rel,
            remote_url,
        )
        return cache_path if cache_exists else None
    cache_path.write_bytes(payload)
    if cache_exists:
        logger.warning(
            "Timeline remote cache refreshed: relative_path=%s url=%s cache_path=%s ttl_seconds=%s",
            rel,
            remote_url,
            cache_path,
            REMOTE_CACHE_TTL_SECONDS,
        )
    else:
        logger.warning(
            "Timeline remote fallback used: relative_path=%s url=%s cache_path=%s",
            rel,
            remote_url,
            cache_path,
        )
    return cache_path


def _resolve_remote_timeline_parquet(season: str, phase: str) -> Optional[Path]:
    relative = f"game_states/{phase}/{season}/{STATES_PARQUET_FILENAME_TEMPLATE.format(season=season, phase=phase)}"
    return _download_remote_pbpdata_file(relative)


def _resolve_remote_timeline_metrics(season: str, phase: str) -> Optional[Path]:
    relative = f"game_states/{phase}/{season}/{TIMELINE_METRICS_FILENAME_TEMPLATE.format(season=season, phase=phase)}"
    return _download_remote_pbpdata_file(relative)


def _timeline_season_dirs(phase: str, season: str) -> list[Path]:
    dirs: list[Path] = []
    for root in PBP_GAME_STATES_ROOTS:
        season_dir = root / phase / season
        if season_dir.exists():
            dirs.append(season_dir)
    return dirs


def _normalize_game_id_for_timeline(game_id: str) -> str:
    text = str(game_id or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else ""


def _normalize_game_type_for_timeline(game_type: Optional[str]) -> str:
    value = str(game_type or "").strip().lower().replace(" ", "_")
    if value == "playoff":
        return "playoffs"
    if value == "playin":
        return "play_in"
    return value


def _timeline_phase_candidates(game_type: Optional[str]) -> list[str]:
    normalized = _normalize_game_type_for_timeline(game_type)
    if normalized in {"playoffs", "play_in"}:
        return ["playoffs", "regular"]
    return ["regular", "playoffs"]


def _sanitize_team_code(team: Optional[str]) -> str:
    return "".join(ch for ch in str(team or "").upper() if ch.isalnum())


def _build_timeline_parquet_path(season_dir: Path, season: str, phase: str) -> Path:
    return season_dir / STATES_PARQUET_FILENAME_TEMPLATE.format(season=season, phase=phase)


def _build_timeline_metrics_path(season_dir: Path, season: str, phase: str) -> Path:
    return season_dir / TIMELINE_METRICS_FILENAME_TEMPLATE.format(season=season, phase=phase)


def _find_timeline_json_file(
    season: str,
    game_id: str,
    game_type: Optional[str] = None,
    home_team: Optional[str] = None,
    road_team: Optional[str] = None,
) -> tuple[Optional[Path], Optional[str]]:
    home_code = _sanitize_team_code(home_team)
    road_code = _sanitize_team_code(road_team)

    for phase in _timeline_phase_candidates(game_type):
        for season_dir in _timeline_season_dirs(phase, season):
            if home_code and road_code:
                exact = season_dir / f"{season}_{home_code}_{road_code}_{game_id}.json"
                if exact.exists():
                    return exact, phase

            matches = sorted(season_dir.glob(f"{season}_*_*_{game_id}.json"))
            if matches:
                return matches[0], phase

    return None, None


def _find_timeline_parquet_file(season: str, phase: str) -> Optional[Path]:
    for season_dir in _timeline_season_dirs(phase, season):
        parquet_path = _build_timeline_parquet_path(season_dir, season=season, phase=phase)
        if parquet_path.exists():
            return parquet_path

    return _resolve_remote_timeline_parquet(season=season, phase=phase)


def _find_timeline_metrics_file(season: str, phase: str) -> Optional[Path]:
    for season_dir in _timeline_season_dirs(phase, season):
        metrics_path = _build_timeline_metrics_path(season_dir, season=season, phase=phase)
        if metrics_path.exists():
            return metrics_path

    return _resolve_remote_timeline_metrics(season=season, phase=phase)


def _safe_numeric(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    except Exception:
        return None


def _percentile_rank(values: list[float], value: float) -> Optional[float]:
    if not values:
        return None
    tol = 1e-12
    lower = sum(1 for v in values if v < (value - tol))
    equal = sum(1 for v in values if abs(v - value) <= tol)
    percentile = ((lower + (0.5 * equal)) / len(values)) * 100.0
    return max(0.0, min(100.0, float(percentile)))


def _load_timeline_metrics_for_game(season: str, phase: str, game_id: str) -> dict[str, Optional[float]]:
    result: dict[str, Optional[float]] = {
        "excitement_factor": None,
        "comeback_factor": None,
        "excitement_percentile": None,
        "comeback_percentile": None,
    }
    metrics_path = _find_timeline_metrics_file(season=season, phase=phase)
    if metrics_path is None:
        return result

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return result
    if not isinstance(payload, dict):
        return result

    games = payload.get("games")
    if not isinstance(games, list):
        return result

    excitement_values: list[float] = []
    comeback_values: list[float] = []
    target_row: Optional[dict[str, Any]] = None

    for row in games:
        if not isinstance(row, dict):
            continue
        excitement = _safe_numeric(row.get("excitement_factor"))
        comeback = _safe_numeric(row.get("comeback_factor"))
        if excitement is not None:
            excitement_values.append(excitement)
        if comeback is not None:
            comeback_values.append(comeback)

        if _normalize_game_id_for_timeline(str(row.get("game_id") or "")) == game_id and target_row is None:
            target_row = row

    if target_row is None:
        return result

    target_excitement = _safe_numeric(target_row.get("excitement_factor"))
    target_comeback = _safe_numeric(target_row.get("comeback_factor"))
    result["excitement_factor"] = target_excitement
    result["comeback_factor"] = target_comeback
    if target_excitement is not None:
        result["excitement_percentile"] = _percentile_rank(excitement_values, target_excitement)
    if target_comeback is not None:
        result["comeback_percentile"] = _percentile_rank(comeback_values, target_comeback)
    return result


def _timeline_payload_from_parquet(
    parquet_path: Path,
    game_id: str,
    home_team: Optional[str] = None,
    road_team: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    home_code = _sanitize_team_code(home_team)
    road_code = _sanitize_team_code(road_team)
    filters: list[tuple[str, str, str]] = [("game_id", "==", game_id)]
    if home_code:
        filters.append(("home_team", "==", home_code))
    if road_code:
        filters.append(("road_team", "==", road_code))

    def _parse_payload(raw: Any) -> Optional[dict[str, Any]]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except Exception:
                return None
            if isinstance(parsed, dict):
                return parsed
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
            try:
                if value is None:
                    normalized.append(None)
                else:
                    normalized.append(float(min(1.0, max(0.0, float(value)))))
            except Exception:
                normalized.append(None)
        return normalized

    def _apply_home_wp_series(payload: dict[str, Any], series: Optional[list[Optional[float]]]) -> dict[str, Any]:
        if not series:
            return payload
        events = payload.get("events")
        if not isinstance(events, list):
            return payload

        max_idx = min(len(events), len(series))
        for idx in range(max_idx):
            prob = series[idx]
            if prob is None:
                continue
            event = events[idx]
            if isinstance(event, dict):
                event["home_win_prob"] = float(prob)
        return payload

    # Primary path: push predicate down to parquet so only the matching game row is read.
    try:
        table = pq.read_table(
            parquet_path,
            columns=["payload_json", HOME_WIN_PROB_BY_EVENT_JSON_COLUMN],
            filters=filters,
            use_threads=False,
        )
    except Exception:
        table = pq.read_table(
            parquet_path,
            columns=["payload_json"],
            filters=filters,
            use_threads=False,
        )
    if table.num_rows > 0 and "payload_json" in table.column_names:
        payload = _parse_payload(table.column("payload_json")[0].as_py())
        if payload is not None:
            series = _parse_home_wp_series(
                table.column(HOME_WIN_PROB_BY_EVENT_JSON_COLUMN)[0].as_py()
                if HOME_WIN_PROB_BY_EVENT_JSON_COLUMN in table.column_names
                else None
            )
            return _apply_home_wp_series(payload, series)

    # Fallback: if strict home/road filter missed, retry by game_id only.
    if home_code or road_code:
        try:
            fallback = pq.read_table(
                parquet_path,
                columns=["home_team", "road_team", "payload_json", HOME_WIN_PROB_BY_EVENT_JSON_COLUMN],
                filters=[("game_id", "==", game_id)],
                use_threads=False,
            )
        except Exception:
            fallback = pq.read_table(
                parquet_path,
                columns=["home_team", "road_team", "payload_json"],
                filters=[("game_id", "==", game_id)],
                use_threads=False,
            )
        if fallback.num_rows > 0:
            cols = {name: fallback.column(name) for name in fallback.column_names}
            for i in range(fallback.num_rows):
                row_home = _sanitize_team_code(cols.get("home_team")[i].as_py() if "home_team" in cols else "")
                row_road = _sanitize_team_code(cols.get("road_team")[i].as_py() if "road_team" in cols else "")
                if home_code and row_home != home_code:
                    continue
                if road_code and row_road != road_code:
                    continue
                payload = _parse_payload(cols.get("payload_json")[i].as_py() if "payload_json" in cols else None)
                if payload is not None:
                    series = _parse_home_wp_series(
                        cols.get(HOME_WIN_PROB_BY_EVENT_JSON_COLUMN)[i].as_py()
                        if HOME_WIN_PROB_BY_EVENT_JSON_COLUMN in cols
                        else None
                    )
                    return _apply_home_wp_series(payload, series)
            payload = _parse_payload(cols.get("payload_json")[0].as_py() if "payload_json" in cols else None)
            if payload is not None:
                series = _parse_home_wp_series(
                    cols.get(HOME_WIN_PROB_BY_EVENT_JSON_COLUMN)[0].as_py()
                    if HOME_WIN_PROB_BY_EVENT_JSON_COLUMN in cols
                    else None
                )
                return _apply_home_wp_series(payload, series)

    return None


def _teams_from_timeline_filename(path: Path) -> tuple[str, str]:
    # Expected shape: season_HOME_ROAD_gameid.json
    parts = path.stem.split("_")
    if len(parts) >= 4:
        return parts[1], parts[2]
    return "", ""


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


def _clock_to_seconds_left(clock_text: Optional[str], period: Optional[int]) -> Optional[int]:
    if not period or period <= 0:
        return None
    match = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", str(clock_text or "").strip())
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


def _timeline_possession_numeric(event: dict[str, Any], home_team: str, road_team: str) -> int:
    home_code = _sanitize_team_code(home_team)
    road_code = _sanitize_team_code(road_team)
    poss_team = _sanitize_team_code(event.get("possession_team_tricode"))
    if poss_team == home_code:
        return 1
    if poss_team == road_code:
        return -1

    poss_side = str(event.get("possession_after_side") or "").strip().lower()
    if poss_side == "home":
        return 1
    if poss_side == "road":
        return -1
    return 0


def _resolve_timeline_payload_for_game(
    season: str,
    game_id: str,
    game_type: Optional[str] = None,
    home_team: Optional[str] = None,
    road_team: Optional[str] = None,
) -> tuple[dict[str, Any], str, Optional[Path]]:
    game_id_norm = _normalize_game_id_for_timeline(game_id)
    if not game_id_norm:
        raise HTTPException(status_code=400, detail="Invalid game_id")

    phase: Optional[str] = None
    timeline_path: Optional[Path] = None
    payload: Optional[dict[str, Any]] = None

    parquet_errors: list[str] = []
    phase_candidates = _timeline_phase_candidates(game_type)
    for candidate_phase in phase_candidates:
        parquet_path = _find_timeline_parquet_file(season=season, phase=candidate_phase)
        if parquet_path is None:
            continue

        try:
            candidate_payload = _timeline_payload_from_parquet(
                parquet_path=parquet_path,
                game_id=game_id_norm,
                home_team=home_team,
                road_team=road_team,
            )
        except Exception as exc:
            parquet_errors.append(f"{candidate_phase}: {exc}")
            continue

        if candidate_payload is not None:
            payload = candidate_payload
            phase = candidate_phase
            if candidate_phase != phase_candidates[0]:
                logger.warning(
                    "Timeline phase fallback used: season=%s game_id=%s requested_game_type=%s phase=%s",
                    season,
                    game_id_norm,
                    game_type,
                    candidate_phase,
                )
            break

    if payload is None:
        timeline_path, phase = _find_timeline_json_file(
            season=season,
            game_id=game_id_norm,
            game_type=game_type,
            home_team=home_team,
            road_team=road_team,
        )
        if timeline_path is None or phase is None:
            if parquet_errors:
                logger.error(
                    "Timeline parquet failed for all phase candidates: season=%s game_id=%s requested_game_type=%s errors=%s",
                    season,
                    game_id_norm,
                    game_type,
                    " | ".join(parquet_errors),
                )
                raise HTTPException(
                    status_code=500,
                    detail="Failed to read timeline data",
                )
            raise HTTPException(
                status_code=404,
                detail=f"Timeline not found for season={season}, game_id={game_id_norm}",
            )
        logger.warning(
            "Timeline JSON fallback used: season=%s game_id=%s requested_game_type=%s file=%s",
            season,
            game_id_norm,
            game_type,
            timeline_path,
        )

        try:
            loaded = json.loads(timeline_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.exception("Failed to read timeline JSON: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to read timeline data")
        payload = loaded if isinstance(loaded, dict) else None

    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid timeline payload")

    return payload, str(phase or ""), timeline_path


def _populate_missing_home_win_prob(
    raw_events: list[dict[str, Any]],
    season: str,
    phase: str,
    home_team: str,
    road_team: str,
) -> None:
    wp_states: list[dict[str, Any]] = []
    wp_event_positions: list[int] = []

    for idx, event in enumerate(raw_events):
        period = _to_int_or_none(event.get("period"))
        clock = str(event.get("clock") or "")
        state = event.get("game_log_state")
        if not isinstance(state, dict):
            state = {}

        pts_home = _to_int_or_none(state.get("pts_home"))
        pts_road = _to_int_or_none(state.get("pts_road"))
        seconds_left = _clock_to_seconds_left(clock, period)
        if (
            _to_float_or_none(event.get("home_win_prob")) is None
            and period
            and seconds_left is not None
            and pts_home is not None
            and pts_road is not None
        ):
            wp_states.append(
                {
                    "quarter": period,
                    "seconds_left": seconds_left,
                    "differential": pts_home - pts_road,
                    "possession_numeric": _timeline_possession_numeric(
                        event=event,
                        home_team=home_team,
                        road_team=road_team,
                    ),
                }
            )
            wp_event_positions.append(idx)

    if not wp_states:
        return

    from admin.winprob_models import DEFAULT_OUTPUT_ROOT, predict_home_winprob_batch

    try:
        wp_probs = predict_home_winprob_batch(
            season=season,
            output_root=str(DEFAULT_OUTPUT_ROOT),
            phase=phase,
            states=wp_states,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to compute timeline win probability: {exc} "
                f"(runtime scikit-learn={_get_pkg_version('scikit-learn')}, "
                f"numpy={_get_pkg_version('numpy')}, pandas={_get_pkg_version('pandas')})"
            ),
        )

    for event_idx, p_home in zip(wp_event_positions, wp_probs):
        if p_home is not None:
            raw_events[event_idx]["home_win_prob"] = float(p_home)


def _parse_maxposs(value: Optional[str]) -> Optional[int]:
    text = str(value or "").strip().lower()
    if not text or text in {"inf", "infinity", "none", "all", "max"}:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid maxposs. Use a positive integer or 'inf'.",
        )
    if parsed <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid maxposs. Use a positive integer or 'inf'.",
        )
    return parsed


def _parse_run_numerator(value: Optional[str]) -> str:
    text = str(value or "dwp").strip().lower()
    if text in {"dwp", "dscore"}:
        return text
    raise HTTPException(
        status_code=400,
        detail="Invalid numerator. Use 'dwp' or 'dscore'.",
    )


def _get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:7]  # Short hash
    except Exception:
        pass
    return "unknown"


def _get_pkg_version(name: str) -> Optional[str]:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


# Cache at startup
GIT_COMMIT = _get_git_commit()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion for optional request payload values."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


LLM_QUINTILE_THRESHOLDS_2018_25 = {
    "off_rating": {"p20": 102.7, "p40": 109.4, "p60": 115.1, "p80": 122.0},
    "def_rating": {"p20": 102.7, "p40": 109.4, "p60": 115.1, "p80": 122.0},
    "net_rating": {"p20": -12.1, "p40": -4.5, "p60": 4.5, "p80": 12.1},
    "efg": {"p20": 48.1, "p40": 51.9, "p60": 55.2, "p80": 59.3},
    "ball_handling": {"p20": 84.4, "p40": 86.5, "p60": 88.3, "p80": 90.2},
    "oreb": {"p20": 17.0, "p40": 21.2, "p60": 25.0, "p80": 29.4},
    "ft_rate": {"p20": 13.6, "p40": 17.5, "p60": 21.2, "p80": 25.8},
}


def _safe_optional_float(value: Any) -> Optional[float]:
    """Best-effort float conversion that preserves missing/invalid as None."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_quintile_thresholds(request: InterpretationRequest) -> Dict[str, Dict[str, float]]:
    """Use request-provided quintile thresholds when complete; otherwise fallback to 2018-25 defaults."""
    provided = request.factor_ranges or {}
    thresholds: Dict[str, Dict[str, float]] = {}

    for metric, defaults in LLM_QUINTILE_THRESHOLDS_2018_25.items():
        candidate = provided.get(metric, {})
        if isinstance(candidate, dict) and all(k in candidate for k in ("p20", "p40", "p60", "p80")):
            thresholds[metric] = {
                "p20": _safe_float(candidate.get("p20"), defaults["p20"]),
                "p40": _safe_float(candidate.get("p40"), defaults["p40"]),
                "p60": _safe_float(candidate.get("p60"), defaults["p60"]),
                "p80": _safe_float(candidate.get("p80"), defaults["p80"]),
            }
        else:
            thresholds[metric] = defaults

    return thresholds


def _classify_quintile(
    value: Optional[float],
    thresholds: Dict[str, float],
    higher_is_better: bool = True,
) -> str:
    """Classify a value into POOR/SUBPAR/AVERAGE/GOOD/EXCELLENT quintiles."""
    if value is None:
        return "AVERAGE"

    p20 = thresholds["p20"]
    p40 = thresholds["p40"]
    p60 = thresholds["p60"]
    p80 = thresholds["p80"]

    if higher_is_better:
        if value <= p20:
            return "POOR"
        if value <= p40:
            return "SUBPAR"
        if value <= p60:
            return "AVERAGE"
        if value <= p80:
            return "GOOD"
        return "EXCELLENT"

    # For metrics where lower is better (e.g., defensive rating)
    if value >= p80:
        return "POOR"
    if value >= p60:
        return "SUBPAR"
    if value >= p40:
        return "AVERAGE"
    if value >= p20:
        return "GOOD"
    return "EXCELLENT"


def _build_llm_decomposition_data(request: InterpretationRequest) -> Dict[str, Any]:
    """
    Normalize interpretation request payload into the flat schema expected by llm.py.

    The prompt builder reads flat keys like `home_efg_contrib`; without this mapping,
    nested API payloads default to zeros and produce generic fallback output.
    """
    home_factors = request.home_factors or {}
    road_factors = request.road_factors or {}
    home_ratings = request.home_ratings or {}
    road_ratings = request.road_ratings or {}
    contributions = request.contributions or {}
    thresholds = _resolve_quintile_thresholds(request)

    home_off_rating = _safe_optional_float(home_ratings.get("offensive_rating"))
    home_def_rating = _safe_optional_float(home_ratings.get("defensive_rating"))
    home_net_rating = _safe_optional_float(home_ratings.get("net_rating"))
    road_off_rating = _safe_optional_float(road_ratings.get("offensive_rating"))
    road_def_rating = _safe_optional_float(road_ratings.get("defensive_rating"))
    road_net_rating = _safe_optional_float(road_ratings.get("net_rating"))

    home_efg = _safe_optional_float(home_factors.get("efg"))
    home_ball_handling = _safe_optional_float(home_factors.get("ball_handling"))
    home_oreb = _safe_optional_float(home_factors.get("oreb"))
    home_ft_rate = _safe_optional_float(home_factors.get("ft_rate"))
    road_efg = _safe_optional_float(road_factors.get("efg"))
    road_ball_handling = _safe_optional_float(road_factors.get("ball_handling"))
    road_oreb = _safe_optional_float(road_factors.get("oreb"))
    road_ft_rate = _safe_optional_float(road_factors.get("ft_rate"))

    home_off_rating_class = _classify_quintile(home_off_rating, thresholds["off_rating"])
    home_def_rating_class = _classify_quintile(
        home_def_rating, thresholds["def_rating"], higher_is_better=False
    )
    home_net_rating_class = _classify_quintile(home_net_rating, thresholds["net_rating"])
    road_off_rating_class = _classify_quintile(road_off_rating, thresholds["off_rating"])
    road_def_rating_class = _classify_quintile(
        road_def_rating, thresholds["def_rating"], higher_is_better=False
    )
    road_net_rating_class = _classify_quintile(road_net_rating, thresholds["net_rating"])

    home_efg_class = _classify_quintile(home_efg, thresholds["efg"])
    home_ball_handling_class = _classify_quintile(
        home_ball_handling, thresholds["ball_handling"]
    )
    home_oreb_class = _classify_quintile(home_oreb, thresholds["oreb"])
    home_ft_rate_class = _classify_quintile(home_ft_rate, thresholds["ft_rate"])
    road_efg_class = _classify_quintile(road_efg, thresholds["efg"])
    road_ball_handling_class = _classify_quintile(
        road_ball_handling, thresholds["ball_handling"]
    )
    road_oreb_class = _classify_quintile(road_oreb, thresholds["oreb"])
    road_ft_rate_class = _classify_quintile(road_ft_rate, thresholds["ft_rate"])

    if request.factor_type == "eight_factors":
        home_efg_contrib = _safe_float(contributions.get("home_shooting"))
        home_ball_handling_contrib = _safe_float(contributions.get("home_ball_handling"))
        home_oreb_contrib = _safe_float(contributions.get("home_orebounding"))
        home_ft_rate_contrib = _safe_float(contributions.get("home_free_throws"))

        road_efg_contrib = _safe_float(contributions.get("road_shooting"))
        road_ball_handling_contrib = _safe_float(contributions.get("road_ball_handling"))
        road_oreb_contrib = _safe_float(contributions.get("road_orebounding"))
        road_ft_rate_contrib = _safe_float(contributions.get("road_free_throws"))
    else:
        # Four-factors mode only has aggregate contributions per factor.
        # Keep a best-effort mapping so external API users still get non-zero context.
        home_efg_contrib = _safe_float(contributions.get("shooting"))
        home_ball_handling_contrib = _safe_float(contributions.get("ball_handling"))
        home_oreb_contrib = _safe_float(contributions.get("orebounding"))
        home_ft_rate_contrib = _safe_float(contributions.get("free_throws"))

        road_efg_contrib = 0.0
        road_ball_handling_contrib = 0.0
        road_oreb_contrib = 0.0
        road_ft_rate_contrib = 0.0

    return {
        "game_id": request.game_id,
        "game_date": request.game_date,
        "matchup": f"{request.road_team}@{request.home_team}",
        "score": f"{request.road_pts}-{request.home_pts}",
        "home_team": request.home_team,
        "road_team": request.road_team,
        "home_pts": request.home_pts,
        "road_pts": request.road_pts,
        "model": request.model_id or "2018-2025",
        "predicted_rating_diff": request.predicted_rating_diff,
        "actual_rating_diff": request.actual_rating_diff,
        "home_off_rating": _safe_float(home_off_rating),
        "home_off_rating_class": home_off_rating_class,
        "home_def_rating": _safe_float(home_def_rating),
        "home_def_rating_class": home_def_rating_class,
        "home_net_rating": _safe_float(home_net_rating),
        "home_net_rating_class": home_net_rating_class,
        "road_off_rating": _safe_float(road_off_rating),
        "road_off_rating_class": road_off_rating_class,
        "road_def_rating": _safe_float(road_def_rating),
        "road_def_rating_class": road_def_rating_class,
        "road_net_rating": _safe_float(road_net_rating),
        "road_net_rating_class": road_net_rating_class,
        "home_efg": _safe_float(home_efg),
        "home_efg_class": home_efg_class,
        "home_ball_handling": _safe_float(home_ball_handling),
        "home_ball_handling_class": home_ball_handling_class,
        "home_oreb": _safe_float(home_oreb),
        "home_oreb_class": home_oreb_class,
        "home_ft_rate": _safe_float(home_ft_rate),
        "home_ft_rate_class": home_ft_rate_class,
        "road_efg": _safe_float(road_efg),
        "road_efg_class": road_efg_class,
        "road_ball_handling": _safe_float(road_ball_handling),
        "road_ball_handling_class": road_ball_handling_class,
        "road_oreb": _safe_float(road_oreb),
        "road_oreb_class": road_oreb_class,
        "road_ft_rate": _safe_float(road_ft_rate),
        "road_ft_rate_class": road_ft_rate_class,
        "home_efg_contrib": home_efg_contrib,
        "home_ball_handling_contrib": home_ball_handling_contrib,
        "home_oreb_contrib": home_oreb_contrib,
        "home_ft_rate_contrib": home_ft_rate_contrib,
        "road_efg_contrib": road_efg_contrib,
        "road_ball_handling_contrib": road_ball_handling_contrib,
        "road_oreb_contrib": road_oreb_contrib,
        "road_ft_rate_contrib": road_ft_rate_contrib,
        # Keep originals for any downstream logic that still reads nested keys.
        "contributions": contributions,
        "home_factors": home_factors,
        "road_factors": road_factors,
        "league_averages": request.league_averages,
        "factor_ranges": request.factor_ranges,
    }


@router.get("/version")
async def get_version():
    """Return the current git commit hash for deployment verification."""
    return {
        "commit": GIT_COMMIT,
        "python_version": sys.version.split()[0],
        "packages": {
            "scikit-learn": _get_pkg_version("scikit-learn"),
            "numpy": _get_pkg_version("numpy"),
            "pandas": _get_pkg_version("pandas"),
        },
    }


@router.get("/winprob/model-seasons")
async def get_winprob_model_seasons(
    phase: str = Query("regular", description="Legacy artifact subfolder fallback"),
):
    from admin.winprob_models import DEFAULT_OUTPUT_ROOT, list_wpm_seasons

    seasons = list_wpm_seasons(output_root=str(DEFAULT_OUTPUT_ROOT), phase=phase)
    return {"seasons": seasons}


@router.get("/winprob/forecast")
async def get_winprob_forecast(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: str = Query(..., description="Game ID from stacked winprob data"),
    game_seconds_left: float = Query(..., ge=0.0, le=720.0, description="Seconds left in game (0-720)"),
    phase: str = Query("regular", description="Phase namespace for base data; also used for legacy artifact fallback"),
):
    validate_season(season)
    validate_game_id(game_id)
    from admin.winprob_models import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT, forecast_from_game_seconds_left

    try:
        result = forecast_from_game_seconds_left(
            season=season,
            phase=phase,
            output_root=str(DEFAULT_OUTPUT_ROOT),
            input_root=str(DEFAULT_INPUT_ROOT),
            game_id=game_id,
            game_seconds_left=game_seconds_left,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to generate forecast: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate forecast")

    return result


@router.get("/winprob/app", response_class=HTMLResponse)
async def get_winprob_app():
    if not WINPROB_APP_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Winprob app file not found: {WINPROB_APP_PATH}")
    return FileResponse(str(WINPROB_APP_PATH), media_type="text/html")


@router.get("/winprob/hypothetical-forecast")
async def get_winprob_hypothetical_forecast(
    season: str = Query(..., description="Season in format YYYY-YY"),
    quarter: int = Query(..., ge=1, description="Quarter number (1-4 regulation, 5+ OT)"),
    seconds_left: float = Query(..., ge=0.0, le=720.0, description="Seconds left in current period (0-720)"),
    differential: float = Query(..., description="Home minus road score differential"),
    possession_numeric: int = Query(..., ge=-1, le=1, description="home=1, road=-1, unknown=0"),
    phase: str = Query("regular", description="Legacy artifact subfolder fallback"),
):
    from admin.winprob_models import DEFAULT_OUTPUT_ROOT, forecast_hypothetical

    try:
        result = forecast_hypothetical(
            season=season,
            phase=phase,
            output_root=str(DEFAULT_OUTPUT_ROOT),
            quarter=quarter,
            seconds_left=seconds_left,
            differential=differential,
            possession_numeric=possession_numeric,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to generate hypothetical forecast: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate hypothetical forecast")

    return result


@router.get("/winprob/hypothetical-app", response_class=HTMLResponse)
async def get_winprob_hypothetical_app():
    if not WINPROB_HYPOTHETICAL_APP_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Hypothetical app file not found: {WINPROB_HYPOTHETICAL_APP_PATH}")
    return FileResponse(str(WINPROB_HYPOTHETICAL_APP_PATH), media_type="text/html")


@router.get("/player-shot-sequences/app", response_class=HTMLResponse)
async def get_player_shot_sequences_app():
    if not PLAYER_SHOT_SEQUENCES_APP_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Player shot sequences app file not found: {PLAYER_SHOT_SEQUENCES_APP_PATH}")
    return FileResponse(str(PLAYER_SHOT_SEQUENCES_APP_PATH), media_type="text/html")


STAT_LABELS = {
    "pts": "Points",
    "fg_pct": "FG%",
    "fg3_pct": "3P%",
    "ft_pct": "FT%",
    "efg_pct": "eFG%",
    "oreb": "Offensive Rebounds",
    "dreb": "Defensive Rebounds",
    "reb": "Total Rebounds",
    "tov": "Turnovers",
    "tov_pct": "TOV%",
    "off_rating": "ORtg",
    "def_rating": "DRtg",
    "net_rating": "Net Rating",
    "ball_handling": "BH",
    "oreb_pct": "OREB%",
    "ft_rate": "FT Rate",
    "opp_efg_pct": "Opp eFG%",
    "opp_ball_handling": "Opp BH",
    "opp_oreb_pct": "Opp OREB%",
    "opp_ft_rate": "Opp FT Rate",
    "opp_ft_pct": "Opp FT%",
    "fg2_pct": "FG2%",
    "fg3a_rate": "FG3A Rate",
    "opp_fg2_pct": "Opp FG2%",
    "opp_fg3_pct": "Opp FG3%",
    "opp_fg3a_rate": "Opp FG3A Rate",
    "pace": "Pace",
}

@router.get("/seasons", response_model=SeasonResponse)
async def get_seasons():
    seasons = get_available_seasons()
    seasons.reverse()
    return SeasonResponse(seasons=seasons)

@router.get("/games", response_model=GamesResponse)
async def get_games(
    season: str = Query(..., description="Season in format YYYY-YY"),
    data_scope: str = Query("all", description="Data scope: all, garbage_filtered, clutch"),
):
    validate_season(season)
    try:
        scope = normalize_data_scope(data_scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    games = await get_games_list(season, data_scope=scope)
    if not games:
        return GamesResponse(games=[])
    game_items = [GameItem(**g) for g in games]
    return GamesResponse(games=game_items)

@router.get("/teams", response_model=TeamsResponse)
async def get_teams(
    season: str = Query(..., description="Season in format YYYY-YY"),
    data_scope: str = Query("all", description="Data scope: all, garbage_filtered, clutch"),
):
    validate_season(season)
    try:
        scope = normalize_data_scope(data_scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    teams = await get_teams_list(season, data_scope=scope)
    return TeamsResponse(teams=teams)


@router.get("/game-timeline", response_model=GameTimelineResponse)
async def get_game_timeline(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: str = Query(..., description="Game ID"),
    game_type: Optional[str] = Query(None, description="Game type (regular_season/playoffs/play_in)"),
    home_team: Optional[str] = Query(None, description="Home team abbreviation"),
    road_team: Optional[str] = Query(None, description="Road team abbreviation"),
):
    validate_season(season)
    validate_game_id(game_id)
    if home_team:
        validate_team(home_team)
    if road_team:
        validate_team(road_team)
    game_id_norm = _normalize_game_id_for_timeline(game_id)
    payload, phase, timeline_path = _resolve_timeline_payload_for_game(
        season=season,
        game_id=game_id,
        game_type=game_type,
        home_team=home_team,
        road_team=road_team,
    )

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raw_events = []

    file_home, file_road = _teams_from_timeline_filename(timeline_path) if timeline_path else ("", "")
    resolved_season = str(payload.get("season") or season)
    resolved_phase = str(payload.get("phase") or phase or "regular")
    resolved_home = str(payload.get("home_team") or home_team or file_home)
    resolved_road = str(payload.get("road_team") or road_team or file_road)
    _populate_missing_home_win_prob(
        raw_events=raw_events,
        season=resolved_season,
        phase=resolved_phase,
        home_team=resolved_home,
        road_team=resolved_road,
    )

    events: list[GameTimelineEvent] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        state = event.get("game_log_state")
        if not isinstance(state, dict):
            state = {}
        period = _to_int_or_none(event.get("period"))
        clock = str(event.get("clock") or "")
        pts_home = _to_int_or_none(state.get("pts_home"))
        pts_road = _to_int_or_none(state.get("pts_road"))
        events.append(
            GameTimelineEvent(
                event_index=_to_int_or_none(event.get("event_index")),
                period=period,
                clock=clock,
                description=str(event.get("description") or ""),
                possession_after_side=str(event.get("possession_after_side") or "") or None,
                possession_team_tricode=str(event.get("possession_team_tricode") or "") or None,
                home_win_prob=_to_float_or_none(event.get("home_win_prob")),
                game_log_state=GameTimelineState(
                    pts_home=pts_home,
                    pts_road=pts_road,
                ),
            )
        )

    validation = payload.get("validation")
    validation_match = validation.get("match") if isinstance(validation, dict) else None

    resolved_game_id = _normalize_game_id_for_timeline(str(payload.get("game_id") or game_id_norm))
    metrics = _load_timeline_metrics_for_game(
        season=resolved_season,
        phase=resolved_phase,
        game_id=resolved_game_id,
    )

    return GameTimelineResponse(
        season=resolved_season,
        phase=resolved_phase,
        game_id=resolved_game_id,
        game_date=str(payload.get("game_date") or "") or None,
        game_type=_normalize_game_type_for_timeline(payload.get("game_type") or game_type) or None,
        home_team=resolved_home,
        road_team=resolved_road,
        excitement_factor=metrics["excitement_factor"],
        comeback_factor=metrics["comeback_factor"],
        excitement_percentile=metrics["excitement_percentile"],
        comeback_percentile=metrics["comeback_percentile"],
        events=events,
        validation_match=validation_match,
    )


@router.get("/game-runs", response_model=GameRunsResponse)
async def get_game_runs(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: str = Query(..., description="Game ID"),
    game_type: Optional[str] = Query(None, description="Game type (regular_season/playoffs/play_in)"),
    home_team: Optional[str] = Query(None, description="Home team abbreviation"),
    road_team: Optional[str] = Query(None, description="Road team abbreviation"),
    maxposs: Optional[str] = Query("inf", description="Maximum run length in possessions, or 'inf' for no cap"),
    minposs: int = Query(1, ge=1, description="Minimum run length in possessions"),
    minmargin: int = Query(0, ge=0, description="Minimum absolute score-margin swing for a run"),
    numerator: str = Query("dwp", description="Run-score numerator: dwp or dscore"),
    run_alpha: float = Query(0.6, ge=0.0, description="Run score exponent applied to possessions + 1"),
    limit: int = Query(4, ge=1, le=10, description="Number of non-overlapping runs to return"),
):
    validate_season(season)
    validate_game_id(game_id)
    if home_team:
        validate_team(home_team)
    if road_team:
        validate_team(road_team)
    resolved_maxposs = _parse_maxposs(maxposs)
    resolved_numerator = _parse_run_numerator(numerator)

    payload, phase, timeline_path = _resolve_timeline_payload_for_game(
        season=season,
        game_id=game_id,
        game_type=game_type,
        home_team=home_team,
        road_team=road_team,
    )

    raw_events = [event for event in payload.get("events", []) if isinstance(event, dict)]
    file_home, file_road = _teams_from_timeline_filename(timeline_path) if timeline_path else ("", "")
    resolved_season = str(payload.get("season") or season)
    resolved_phase = str(payload.get("phase") or phase or "regular")
    resolved_home = str(payload.get("home_team") or home_team or file_home)
    resolved_road = str(payload.get("road_team") or road_team or file_road)
    resolved_game_id = _normalize_game_id_for_timeline(str(payload.get("game_id") or game_id))

    _populate_missing_home_win_prob(
        raw_events=raw_events,
        season=resolved_season,
        phase=resolved_phase,
        home_team=resolved_home,
        road_team=resolved_road,
    )

    possessions = extract_timeline_possessions(
        raw_events=raw_events,
        home_team=resolved_home,
        road_team=resolved_road,
    )
    ranked_runs = rank_non_overlapping_runs(
        possessions=possessions,
        home_team=resolved_home,
        road_team=resolved_road,
        max_possessions=resolved_maxposs,
        run_alpha=run_alpha,
        min_possessions=minposs,
        min_margin=minmargin,
        numerator=resolved_numerator,
        limit=limit,
    )

    return GameRunsResponse(
        season=resolved_season,
        phase=resolved_phase,
        game_id=resolved_game_id,
        game_date=str(payload.get("game_date") or "") or None,
        game_type=_normalize_game_type_for_timeline(payload.get("game_type") or game_type) or None,
        home_team=resolved_home,
        road_team=resolved_road,
        max_possessions=resolved_maxposs,
        min_possessions=minposs,
        min_margin=minmargin,
        run_alpha=run_alpha,
        numerator=resolved_numerator,
        runs=[GameRun(**run) for run in ranked_runs],
    )


@router.get("/pbp-boxscore-traditional", response_model=PBPTraditionalBoxScoreResponse)
async def get_pbp_boxscore_traditional(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: str = Query(..., description="Game ID"),
    segment: str = Query("game", description="Box score segment: game, q1, q2, q3, q4, ot, h1, h2, no_garbage, garbage, clutch"),
):
    validate_season(season)
    validate_game_id(game_id)
    try:
        segment_norm = normalize_boxscore_segment(segment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        payload = compute_pbp_traditional_boxscore(season=season, game_id=game_id, segment=segment_norm)
        return PBPTraditionalBoxScoreResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to compute PBP traditional box score: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to compute PBP traditional box score") from exc


@router.get("/player-game-facts", response_model=PlayerGameFactsResponse)
async def get_player_game_facts(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: Optional[str] = Query(None, description="Optional 10-digit game ID filter"),
    player_id: Optional[int] = Query(None, description="Optional NBA player ID filter"),
    team_id: Optional[int] = Query(None, description="Optional NBA team ID filter"),
    include_dnp: bool = Query(False, description="Include 0-minute DNP/DND/NWT rows"),
):
    validate_season(season)
    if game_id is not None:
        validate_game_id(game_id)
    if player_id is not None and int(player_id) <= 0:
        raise HTTPException(status_code=400, detail="player_id must be a positive integer")
    if team_id is not None and int(team_id) <= 0:
        raise HTTPException(status_code=400, detail="team_id must be a positive integer")

    try:
        payload = build_player_game_facts_payload(
            season=season,
            game_id=game_id,
            player_id=player_id,
            team_id=team_id,
            include_dnp=include_dnp,
        )
        return PlayerGameFactsResponse(**payload)
    except Exception as exc:
        logger.exception("Failed to build player game facts: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build player game facts") from exc


@router.get("/player-shot-players", response_model=PlayerShotPlayersResponse)
async def get_player_shot_players(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_type: Optional[str] = Query(None, description="Optional game type: regular_season, playoffs, play_in, nba_cup_semi, nba_cup_final"),
):
    validate_season(season)
    game_type_norm = str(game_type).strip().lower() if game_type is not None else None
    if game_type_norm is not None and game_type_norm not in PLAYER_SHOTS_GAME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="game_type must be one of regular_season, playoffs, play_in, nba_cup_semi, nba_cup_final",
        )

    try:
        players = list_player_shot_players(season, game_type=game_type_norm)
        return PlayerShotPlayersResponse(
            season=season,
            game_type=game_type_norm,
            player_count=len(players),
            players=players,
        )
    except Exception as exc:
        logger.exception("Failed to list player shot players: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list player shot players") from exc


@router.get("/player-shots", response_model=PlayerShotsResponse)
async def get_player_shots(
    player_id: Optional[int] = Query(None, description="Optional NBA player ID filter"),
    player_name: Optional[str] = Query(None, description="Optional case-insensitive player name search"),
    start_season: Optional[str] = Query(None, description="Optional first season in format YYYY-YY"),
    end_season: Optional[str] = Query(None, description="Optional final season in format YYYY-YY"),
    game_type: Optional[str] = Query(None, description="Optional game type: regular_season, playoffs, play_in, nba_cup_semi, nba_cup_final"),
    shot_type: Optional[str] = Query(None, description="Optional shot type: fta, 2ptfga, or 3ptfga"),
    result: Optional[str] = Query(None, description="Optional result: make or miss"),
    team: Optional[str] = Query(None, description="Optional team abbreviation"),
    opponent: Optional[str] = Query(None, description="Optional opponent abbreviation"),
    limit: int = Query(5000, ge=1, le=50000, description="Maximum rows returned"),
    offset: int = Query(0, ge=0, description="Rows to skip after sorting by career sequence"),
):
    if player_id is None and not player_name:
        raise HTTPException(status_code=400, detail="player_id or player_name is required")
    if player_id is not None and int(player_id) <= 0:
        raise HTTPException(status_code=400, detail="player_id must be a positive integer")
    if start_season is not None:
        validate_season(start_season)
    if end_season is not None:
        validate_season(end_season)

    game_type_norm = str(game_type).strip().lower() if game_type is not None else None
    if game_type_norm is not None and game_type_norm not in PLAYER_SHOTS_GAME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="game_type must be one of regular_season, playoffs, play_in, nba_cup_semi, nba_cup_final",
        )
    if shot_type is not None and str(shot_type).strip().lower() not in {"fta", "2ptfga", "3ptfga"}:
        raise HTTPException(status_code=400, detail="shot_type must be one of fta, 2ptfga, 3ptfga")
    if result is not None and str(result).strip().lower() not in {"make", "miss"}:
        raise HTTPException(status_code=400, detail="result must be make or miss")
    if team is not None:
        validate_team(team.upper())
    if opponent is not None:
        validate_team(opponent.upper())

    try:
        return PlayerShotsResponse(
            **build_player_shots_payload(
                player_id=player_id,
                player_name=player_name,
                start_season=start_season,
                end_season=end_season,
                game_type=game_type_norm,
                shot_type=str(shot_type).strip().lower() if shot_type else None,
                result=str(result).strip().lower() if result else None,
                team=team.upper() if team else None,
                opponent=opponent.upper() if opponent else None,
                limit=limit,
                offset=offset,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to build player shots: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build player shots") from exc


@router.get("/player-shot-streakiness", response_model=PlayerShotStreakinessResponse)
async def get_player_shot_streakiness(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_type: Optional[str] = Query(None, description="Optional game type; defaults to regular_season"),
    shot_type: Optional[str] = Query(None, description="Optional shot type: fta, 2ptfga, or 3ptfga"),
    min_attempts: int = Query(100, ge=1, le=2000, description="Minimum attempts for leaderboard inclusion"),
    simulations: int = Query(1000, ge=100, le=5000, description="Number of make-preserving random shuffles"),
    classification: Optional[str] = Query(None, description="Optional classification filter"),
):
    validate_season(season)
    game_type_norm = str(game_type or "regular_season").strip().lower()
    if game_type_norm not in PLAYER_SHOTS_GAME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="game_type must be one of regular_season, playoffs, play_in, nba_cup_semi, nba_cup_final",
        )
    shot_type_norm = str(shot_type).strip().lower() if shot_type else None
    if shot_type_norm is not None and shot_type_norm not in PLAYER_SHOT_TYPES:
        raise HTTPException(status_code=400, detail="shot_type must be one of fta, 2ptfga, 3ptfga")
    classification_norm = str(classification).strip().lower() if classification else None
    if classification_norm is not None and classification_norm not in PLAYER_SHOT_CLASSIFICATIONS:
        raise HTTPException(
            status_code=400,
            detail="classification must be one of consistent, streaky, volatile, alternating, ordinary",
        )

    try:
        return PlayerShotStreakinessResponse(
            **build_player_shot_streakiness_payload(
                season=season,
                game_type=game_type_norm,
                shot_type=shot_type_norm,
                min_attempts=min_attempts,
                simulations=simulations,
                classification=classification_norm,
            )
        )
    except Exception as exc:
        logger.exception("Failed to build player shot streakiness: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build player shot streakiness") from exc


@router.get("/decomposition", response_model=DecompositionResponse)
async def get_decomposition(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: str = Query(..., description="Game ID"),
    model_id: Optional[str] = Query(None, description="Deprecated, ignored"),
    factor_type: str = Query("eight_factors", description="Factor type: eight_factors (default)"),
    data_scope: str = Query("all", description="Data scope: all, garbage_filtered, clutch"),
):
    validate_season(season)
    validate_game_id(game_id)
    try:
        scope = normalize_data_scope(data_scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Load pre-calculated contributions for the season
    contrib_data = await load_contributions(season, data_scope=scope)
    if contrib_data is None:
        raise HTTPException(status_code=404, detail="Contributions not found for season")

    # Find the game in the contributions JSON
    # Normalize game_id: contributions use 10-digit zero-padded IDs (e.g. "0022400407")
    # while the games list may return shorter IDs (e.g. "42400407")
    game_id_padded = str(game_id).zfill(10)
    game_entry = None
    for g in contrib_data.get("games", []):
        if str(g["game_id"]).zfill(10) == game_id_padded:
            game_entry = g
            break

    if game_entry is None:
        raise HTTPException(status_code=404, detail="Game not found in contributions")

    game_info = game_entry["game_info"]
    model_info = game_entry["model"]
    ls = game_entry.get("linescore", {})

    # Map factor arrays to dicts
    # Factor order in JSON: Shooting, Ball Handling, Off Rebounding, Free Throw Rate
    FACTOR_KEYS = ["shooting", "ball_handling", "orebounding", "free_throws"]
    FACTOR_INTERNAL = ["efg", "ball_handling", "oreb", "ft_rate"]

    home_factors_list = game_entry["factors"]["home"]
    road_factors_list = game_entry["factors"]["road"]

    home_factors = {FACTOR_INTERNAL[i]: home_factors_list[i]["value"] for i in range(4)}
    road_factors = {FACTOR_INTERNAL[i]: road_factors_list[i]["value"] for i in range(4)}

    # Build contributions dict based on factor_type
    if factor_type == "eight_factors":
        contributions = {}
        for i, key in enumerate(FACTOR_KEYS):
            contributions[f"home_{key}"] = round(home_factors_list[i]["contribution"], 2)
            contributions[f"road_{key}"] = round(road_factors_list[i]["contribution"], 2)
    else:
        # Four factors: sum home + road contributions per factor
        contributions = {}
        for i, key in enumerate(FACTOR_KEYS):
            contributions[key] = round(
                home_factors_list[i]["contribution"] + road_factors_list[i]["contribution"], 2
            )

    # Map ratings arrays to dicts
    # Rating order in JSON: Offensive Rating, Defensive Rating, Net Rating, Pace
    RATING_KEYS = ["offensive_rating", "defensive_rating", "net_rating", "pace"]
    home_ratings_list = game_entry["ratings"]["home"]
    road_ratings_list = game_entry["ratings"]["road"]

    home_ratings = {RATING_KEYS[i]: home_ratings_list[i]["value"] for i in range(4)}
    road_ratings = {RATING_KEYS[i]: road_ratings_list[i]["value"] for i in range(4)}

    # Attach scoped minutes/possessions when available so the UI can show
    # segment duration and compute pace context for each data scope.
    if ls:
        home_ls = ls.get("home", {})
        road_ls = ls.get("road", {})

        home_minutes = _safe_numeric(home_ls.get("Minutes"))
        road_minutes = _safe_numeric(road_ls.get("Minutes"))
        home_possessions = _safe_numeric(home_ls.get("Possessions"))
        road_possessions = _safe_numeric(road_ls.get("Possessions"))

        if home_minutes is not None:
            home_ratings["minutes"] = float(home_minutes)
        if road_minutes is not None:
            road_ratings["minutes"] = float(road_minutes)
        if home_possessions is not None:
            home_ratings["possessions"] = float(home_possessions)
        if road_possessions is not None:
            road_ratings["possessions"] = float(road_possessions)

    # League averages from model metadata
    league_avgs = dict(model_info.get("league_averages", {}))
    # Derive def_rating = off_rating (league average ORtg equals DRtg)
    if "off_rating" in league_avgs and "def_rating" not in league_avgs:
        league_avgs["def_rating"] = league_avgs["off_rating"]

    # Linescore (JSON uses uppercase keys Q1..Q4, OT, Total; Pydantic expects lowercase)
    linescore_response = None
    is_overtime = False
    overtime_count = 0

    if ls:
        home_ls = ls.get("home", {})
        road_ls = ls.get("road", {})
        linescore_response = LinescoreData(
            home=QuarterScores(
                q1=home_ls.get("Q1", 0), q2=home_ls.get("Q2", 0),
                q3=home_ls.get("Q3", 0), q4=home_ls.get("Q4", 0),
                ot=home_ls.get("OT", 0),
            ),
            road=QuarterScores(
                q1=road_ls.get("Q1", 0), q2=road_ls.get("Q2", 0),
                q3=road_ls.get("Q3", 0), q4=road_ls.get("Q4", 0),
                ot=road_ls.get("OT", 0),
            ),
        )
        is_overtime = home_ls.get("OT", 0) > 0 or road_ls.get("OT", 0) > 0

    # Compute scalars
    home_pts = ls.get("home", {}).get("Total", 0) if ls else 0
    road_pts = ls.get("road", {}).get("Total", 0) if ls else 0
    actual_margin = home_pts - road_pts
    actual_rating_diff = round(home_ratings.get("net_rating", 0), 2)

    intercept = model_info.get("intercept", 0)
    predicted_rating_diff = round(
        intercept + sum(contributions.values()), 2
    )

    # Normalize game_type from display format to snake_case
    game_type_raw = game_info.get("game_type", "")
    game_type_map = {
        "Regular Season": "regular_season",
        "Playoffs": "playoffs",
        "NBA Cup (Group)": "nba_cup_group",
        "NBA Cup (Knockout)": "nba_cup_knockout",
    }
    game_type = game_type_map.get(game_type_raw, game_type_raw.lower().replace(" ", "_") if game_type_raw else None)

    response = DecompositionResponse(
        game_id=game_id,
        game_date=game_info.get("game_date", ""),
        home_team=game_info.get("home", ""),
        road_team=game_info.get("road", ""),
        home_pts=home_pts,
        road_pts=road_pts,
        actual_margin=actual_margin,
        actual_rating_diff=actual_rating_diff,
        predicted_rating_diff=predicted_rating_diff,
        factor_type=factor_type,
        home_factors=home_factors,
        road_factors=road_factors,
        contributions=contributions,
        intercept=intercept,
        home_ratings=home_ratings,
        road_ratings=road_ratings,
        league_averages=league_avgs,
        linescore=linescore_response,
        is_overtime=is_overtime,
        overtime_count=overtime_count,
        game_type=game_type,
    )

    return response


@router.get("/contributions/single-game")
async def get_single_game_contribution_json(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: str = Query(..., description="Game ID"),
    data_scope: str = Query("all", description="Data scope: all, garbage_filtered, clutch"),
):
    """Return a contribution JSON payload with exactly one game entry.

    The response mirrors the normal `contributions_*.json` structure, but with
    the `games` array filtered to the requested game.
    """
    try:
        scope = normalize_data_scope(data_scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    contrib_data = await load_contributions(season, data_scope=scope)
    if contrib_data is None:
        raise HTTPException(status_code=404, detail="Contributions not found for season")

    game_id_norm = _normalize_game_id_for_timeline(game_id)
    if not game_id_norm:
        raise HTTPException(status_code=400, detail="Invalid game_id")

    matching_game = None
    for row in contrib_data.get("games", []):
        if not isinstance(row, dict):
            continue
        row_game_id = _normalize_game_id_for_timeline(str(row.get("game_id") or ""))
        if row_game_id == game_id_norm:
            matching_game = row
            break

    if matching_game is None:
        raise HTTPException(status_code=404, detail="Game not found in contributions")

    payload = deepcopy(contrib_data) if isinstance(contrib_data, dict) else {}
    payload["season"] = season
    payload["data_scope"] = scope
    payload["games"] = [matching_game]
    return payload


@router.get("/interpretation/prompt")
async def get_interpretation_prompt(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: str = Query(..., description="Game ID"),
    factor_type: str = Query("eight_factors", description="Factor type: eight_factors (default)"),
    data_scope: str = Query("all", description="Data scope: all, garbage_filtered, clutch"),
):
    """Return the fully rendered prompt text that would be sent to the LLM."""
    try:
        scope = normalize_data_scope(data_scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    decomposition = await get_decomposition(
        season=season,
        game_id=game_id,
        model_id=None,
        factor_type=factor_type,
        data_scope=scope,
    )

    interp_request = InterpretationRequest(
        game_id=decomposition.game_id,
        game_date=decomposition.game_date,
        season=season,
        home_team=decomposition.home_team,
        road_team=decomposition.road_team,
        home_pts=decomposition.home_pts,
        road_pts=decomposition.road_pts,
        contributions=decomposition.contributions,
        predicted_rating_diff=decomposition.predicted_rating_diff,
        actual_rating_diff=decomposition.actual_rating_diff,
        factor_type=factor_type,
        model_id=None,
        home_factors=decomposition.home_factors,
        road_factors=decomposition.road_factors,
        home_ratings=decomposition.home_ratings,
        road_ratings=decomposition.road_ratings,
        league_averages=decomposition.league_averages,
        factor_ranges=None,
    )

    decomposition_data = _build_llm_decomposition_data(interp_request)
    prompt = _build_interpretation_prompt(decomposition_data, factor_type=factor_type)
    return {
        "season": season,
        "game_id": str(decomposition.game_id).zfill(10),
        "factor_type": factor_type,
        "data_scope": scope,
        "prompt": prompt,
    }


@router.post("/interpretation", response_model=InterpretationResponse)
async def get_interpretation(request: InterpretationRequest):
    """Get AI interpretation of factor contributions for a game.

    First checks for pre-generated interpretation, then falls back to real-time generation.
    """
    if request.factor_type != "eight_factors":
        raise HTTPException(status_code=404, detail="Interpretations are only available for eight_factors")

    if not request.season:
        raise HTTPException(
            status_code=400,
            detail=f"Season is required. Interpretations are only available for {INTERPRETATION_ENABLED_SEASON}.",
        )

    if request.season != INTERPRETATION_ENABLED_SEASON:
        raise HTTPException(
            status_code=404,
            detail=f"Interpretations are only available for {INTERPRETATION_ENABLED_SEASON}.",
        )

    try:
        request_scope = normalize_data_scope(request.data_scope or "all")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if request_scope != INTERPRETATION_ENABLED_SCOPE:
        raise HTTPException(
            status_code=404,
            detail="Interpretations are only available for the All Game scope.",
        )

    # Try to get pre-generated interpretation first.
    # Pass model_id so we only use pre-generated if it matches the decomposition model.
    if request.game_id:
        pre_generated = await get_game_interpretation(
            season=request.season,
            game_id=request.game_id,
            factor_type=request.factor_type,
            model_id=request.model_id,
        )
        if pre_generated:
            return InterpretationResponse(
                interpretation=pre_generated["text"],
                model=pre_generated["model"],
            )

    # Fall back to real-time generation
    if not is_llm_configured():
        raise HTTPException(
            status_code=503,
            detail="Interpretation service requires OPENAI_API_KEY for gpt-5-mini runtime generation.",
        )

    # Build normalized, flat payload expected by llm.py prompt builder
    decomposition_data = _build_llm_decomposition_data(request)

    runtime_model = get_runtime_interpretation_model()

    interpretation = await generate_interpretation(
        decomposition_data=decomposition_data,
        factor_type=request.factor_type,
        model_id=request.model_id,
    )

    if interpretation is None:
        raise HTTPException(status_code=503, detail="Failed to generate interpretation")

    return InterpretationResponse(interpretation=interpretation, model=runtime_model or "unknown")


@router.get("/league-summary", response_model=LeagueSummaryResponse)
async def get_league_summary(
    season: str = Query(..., description="Season in format YYYY-YY"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    exclude_playoffs: bool = Query(True, description="Exclude playoff, play-in, and NBA Cup final games"),
    last_n_games: Optional[int] = Query(None, description="Use each team's last N games"),
    data_scope: str = Query(
        "all",
        description=(
            "Data scope: all, q1, q2, q3, q4, ot, h1, h2, "
            "garbage_filtered (non-garbage), garbage_time, clutch, non_clutch_time"
        ),
    ),
):
    validate_season(season)
    try:
        scope = _normalize_league_summary_scope(data_scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    df, all_df = await _resolve_league_summary_scope_df(season=season, scope=scope)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    # Get date bounds for the season
    bounds_df = all_df if all_df is not None else df
    first_game_date = bounds_df["game_date"].min().strftime("%Y-%m-%d") if len(bounds_df) > 0 else None
    last_game_date = bounds_df["game_date"].max().strftime("%Y-%m-%d") if len(bounds_df) > 0 else None

    team_stats_df = compute_league_aggregates(
        df=df,
        start_date=start_date,
        end_date=end_date,
        exclude_playoffs=exclude_playoffs,
        last_n_games=last_n_games,
    )
    scope_metrics_df = None
    if scope != "all" and all_df is not None:
        scope_metrics_df = compute_scope_time_metrics(
            all_df=all_df,
            scoped_df=df,
            data_scope=scope,
            start_date=start_date,
            end_date=end_date,
            exclude_playoffs=exclude_playoffs,
            last_n_games=last_n_games,
        )
        if not scope_metrics_df.empty:
            team_stats_df = team_stats_df.merge(scope_metrics_df, on="team", how="left")

    teams = []
    for _, row in team_stats_df.iterrows():
        scope_games_num = _safe_numeric(row.get("scope_games"))
        scope_time_pct_num = _safe_numeric(row.get("scope_time_pct"))
        teams.append(TeamStats(
            team=row["team"],
            games=int(row["games"]),
            wins=int(row["wins"]),
            losses=int(row["losses"]),
            win_pct=float(row["win_pct"]),
            ppg=float(row["ppg"]),
            opp_ppg=float(row["opp_ppg"]),
            fg_pct=float(row["fg_pct"]),
            fg3_pct=float(row["fg3_pct"]),
            ft_pct=float(row["ft_pct"]),
            fg2_pct=float(row["fg2_pct"]),
            fg3a_rate=float(row["fg3a_rate"]),
            efg_pct=float(row["efg_pct"]),
            oreb_pct=float(row["oreb_pct"]),
            dreb_pct=float(row["dreb_pct"]),
            tov_pct=float(row["tov_pct"]),
            ball_handling=float(row["ball_handling"]),
            ft_rate=float(row["ft_rate"]),
            off_rating=float(row["off_rating"]),
            def_rating=float(row["def_rating"]),
            net_rating=float(row["net_rating"]),
            opp_efg_pct=float(row["opp_efg_pct"]),
            opp_ft_pct=float(row["opp_ft_pct"]),
            opp_fg2_pct=float(row["opp_fg2_pct"]),
            opp_fg3_pct=float(row["opp_fg3_pct"]),
            opp_fg3a_rate=float(row["opp_fg3a_rate"]),
            opp_tov_pct=float(row["opp_tov_pct"]),
            opp_ball_handling=float(row["opp_ball_handling"]),
            opp_oreb_pct=float(row["opp_oreb_pct"]),
            opp_ft_rate=float(row["opp_ft_rate"]),
            pace=float(row["pace"]),
            sos=float(row["sos"]),
            off_sos=float(row["off_sos"]),
            def_sos=float(row["def_sos"]),
            adj_net_rating=float(row["adj_net_rating"]),
            adj_off_rating=float(row["adj_off_rating"]),
            adj_def_rating=float(row["adj_def_rating"]),
            scope_games=int(round(scope_games_num)) if scope_games_num is not None else None,
            scope_time_pct=float(scope_time_pct_num) if scope_time_pct_num is not None else None,
        ))

    league_averages = compute_league_summary_averages(
        df=df,
        team_stats_df=team_stats_df,
        start_date=start_date,
        end_date=end_date,
        exclude_playoffs=exclude_playoffs,
        last_n_games=last_n_games,
        scope_metrics_df=scope_metrics_df if scope != "all" else None,
    )

    return LeagueSummaryResponse(
        teams=teams,
        league_averages=league_averages,
        first_game_date=first_game_date,
        last_game_date=last_game_date,
    )

@router.get("/trends", response_model=TrendsResponse)
async def get_trends(
    season: str = Query(..., description="Season in format YYYY-YY"),
    team: str = Query(..., description="Team abbreviation"),
    stat: str = Query(..., description="Statistic to plot"),
    exclude_non_regular: bool = Query(False, description="Exclude playoffs, play-in, and NBA Cup final from trends"),
    data_scope: str = Query("all", description="Data scope: all, garbage_filtered, clutch"),
):
    validate_season(season)
    validate_team(team)
    try:
        scope = normalize_data_scope(data_scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    df = await get_normalized_data_with_possessions(season, data_scope=scope)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    if team not in df["team"].unique():
        raise HTTPException(status_code=404, detail="Team not found")

    stat_key = stat.upper()

    if stat_key in STAT_ALIASES:
        stat_internal = STAT_ALIASES[stat_key]
    else:
        stat_internal = stat.lower()

    if stat_internal not in STAT_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid stat: {stat}")

    trend_df = compute_trend_series(df, team, stat_internal, exclude_non_regular=exclude_non_regular)

    data = []
    for _, row in trend_df.iterrows():
        data.append(TrendPoint(
            game_id=str(row["game_id"]),
            game_date=row["game_date"],
            opponent=row["opponent"],
            home_away=row["home_away"],
            value=float(row["value"]) if row["value"] is not None else 0,
            ma_5=float(row["ma_5"]) if row["ma_5"] is not None else 0,
            ma_10=float(row["ma_10"]) if row["ma_10"] is not None else 0,
            wl=row["wl"] if row["wl"] else "",
        ))

    season_average = round(trend_df["value"].mean(), 1) if len(trend_df) > 0 else 0
    league_average = compute_league_average(df, stat_internal)

    return TrendsResponse(
        team=team,
        stat=stat,
        stat_label=STAT_LABELS.get(stat_internal, stat),
        data=data,
        season_average=season_average,
        league_average=league_average,
    )

@router.get("/contribution-analysis", response_model=ContributionAnalysisResponse)
async def get_contribution_analysis(
    season: str = Query(..., description="Season in format YYYY-YY"),
    team: str = Query(..., description="Team abbreviation"),
    model_id: Optional[str] = Query(None, description="Deprecated, ignored"),
    date_range_type: str = Query("season", description="Type: season, last_n, or custom"),
    last_n_games: Optional[int] = Query(None, description="Number of games for last_n type"),
    start_date: Optional[str] = Query(None, description="Start date for custom type (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for custom type (YYYY-MM-DD)"),
    exclude_playoffs: bool = Query(False, description="Exclude playoff and play-in games"),
    data_scope: str = Query("all", description="Data scope: all, garbage_filtered, clutch"),
):
    """Analyze a team's net rating decomposition over a period using contribution JSON."""
    validate_season(season)
    validate_team(team)
    del model_id  # Explicitly ignored for backwards compatibility.

    # Load season data
    try:
        scope = normalize_data_scope(data_scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    df = await get_normalized_data_with_possessions(season, data_scope=scope)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    # Check if team exists
    if team not in df["team"].unique():
        raise HTTPException(status_code=404, detail="Team not found in this season")

    # Load pre-calculated per-game contributions
    contrib_data = await load_contributions(season, data_scope=scope)
    if contrib_data is None:
        raise HTTPException(status_code=404, detail="Contributions not found for season")

    # Filter to team's games
    team_df = df[df["team"] == team].copy()
    team_df = team_df.sort_values("game_date")

    # Exclude nba_cup_final
    team_df = team_df[team_df["game_type"] != "nba_cup_final"]

    if team_df.empty:
        raise HTTPException(status_code=404, detail="No games found for team")

    # Apply date range filter
    filter_start_date = None
    filter_end_date = None
    filter_last_n_games = None
    date_range_label = "Season-to-Date"

    if date_range_type == "last_n" and last_n_games:
        filter_last_n_games = last_n_games
        date_range_label = f"Last {last_n_games} Games"
    elif date_range_type == "custom" and start_date and end_date:
        filter_start_date = start_date
        filter_end_date = end_date
        date_range_label = f"{start_date} to {end_date}"
    # else: season-to-date (no filter needed, use all games)

    if exclude_playoffs and date_range_label == "Season-to-Date":
        date_range_label = "Season-to-Date, No Playoffs"

    try:
        result = compute_contribution_analysis(
            team_df=team_df,
            league_df=df,
            contributions_data=contrib_data,
            start_date=filter_start_date,
            end_date=filter_end_date,
            exclude_playoffs=exclude_playoffs,
            last_n_games=filter_last_n_games,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build top contributors response
    top_contributors = [
        TopContributor(
            factor=tc["factor"],
            factor_label=tc["factor_label"],
            value=tc["value"],
            league_avg=tc["league_avg"],
            contribution=tc["contribution"],
            trend_data=[
                ContributionTrendPoint(**point) for point in tc["trend_data"]
            ]
        )
        for tc in result["top_contributors"]
    ]

    return ContributionAnalysisResponse(
        team=team,
        season=season,
        date_range_label=date_range_label,
        start_date=result["start_date"],
        end_date=result["end_date"],
        games_analyzed=result["games_analyzed"],
        wins=result["wins"],
        losses=result["losses"],
        win_pct=result["win_pct"],
        net_rating=result["net_rating"],
        predicted_net_rating=result["predicted_net_rating"],
        contributions=result["contributions"],
        factor_values=result["factor_values"],
        league_averages=result["league_averages"],
        top_contributors=top_contributors,
        intercept=result["intercept"],
    )


@router.get("/league-top-contributors", response_model=LeagueTopContributorsResponse)
async def get_league_top_contributors(
    season: str = Query(..., description="Season in format YYYY-YY"),
    model_id: Optional[str] = Query(None, description="Deprecated, ignored"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    exclude_playoffs: bool = Query(False, description="Exclude playoff and play-in games"),
    last_n_games: Optional[int] = Query(None, description="Use each team's last N games"),
    data_scope: str = Query(
        "all",
        description=(
            "Data scope: all, q1, q2, q3, q4, ot, h1, h2, "
            "garbage_filtered (non-garbage), garbage_time, clutch, non_clutch_time"
        ),
    ),
):
    """Get top positive and negative contributors to net rating across all teams."""
    response_model_id = model_id or "json_contributions"

    # Load season data
    try:
        scope = _normalize_league_summary_scope(data_scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Derived League Summary scopes do not yet have persisted contribution JSON artifacts.
    if scope in {"garbage_time", "non_clutch_time"}:
        return LeagueTopContributorsResponse(
            season=season,
            start_date="",
            end_date="",
            model_id=response_model_id,
            top_positive=[],
            top_negative=[],
            league_averages={},
            coefficients={},
        )

    df = await get_normalized_data_with_possessions(season, data_scope=scope)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    # Load pre-calculated per-game contributions
    contrib_data = await load_contributions(season, data_scope=scope)
    if contrib_data is None:
        raise HTTPException(status_code=404, detail="Contributions not found for season")

    try:
        result = compute_league_top_contributors(
            league_df=df,
            contributions_data=contrib_data,
            start_date=start_date,
            end_date=end_date,
            exclude_playoffs=exclude_playoffs,
            last_n_games=last_n_games,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build response items
    top_positive = [
        LeagueContributorItem(
            team=c["team"],
            factor=c["factor"],
            factor_label=c["factor_label"],
            value=c["value"],
            contribution=c["contribution"],
        )
        for c in result["top_positive"]
    ]

    top_negative = [
        LeagueContributorItem(
            team=c["team"],
            factor=c["factor"],
            factor_label=c["factor_label"],
            value=c["value"],
            contribution=c["contribution"],
        )
        for c in result["top_negative"]
    ]

    return LeagueTopContributorsResponse(
        season=season,
        start_date=result["start_date"],
        end_date=result["end_date"],
        model_id=response_model_id,
        top_positive=top_positive,
        top_negative=top_negative,
        league_averages=result["league_averages"],
        coefficients=result["coefficients"],
    )


@router.post("/admin/clear-cache")
async def admin_clear_cache(
    key: str = Query(..., description="Admin secret key"),
):
    """Clear the in-memory cache. Requires ADMIN_SECRET_KEY."""
    if not ADMIN_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Admin endpoint not configured")
    if not hmac.compare_digest(key, ADMIN_SECRET_KEY):
        raise HTTPException(status_code=403, detail="Invalid key")

    clear_cache()
    clear_pbp_boxscore_cache()
    build_player_shot_streakiness_payload.cache_clear()
    return {"status": "ok", "message": "Cache cleared"}
