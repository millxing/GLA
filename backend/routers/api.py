from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from typing import Optional, Dict, Any
import subprocess
import json
import re
from pathlib import Path
import pyarrow.parquet as pq
from config import get_available_seasons, ADMIN_SECRET_KEY
from services.cache import clear_cache
from services.data_loader import (
    get_normalized_season_data,
    get_normalized_data_with_possessions,
    get_games_list,
    get_teams_list,
    load_contributions,
)
from services.calculations import (
    compute_league_aggregates,
    compute_trend_series,
    compute_league_average,
    compute_contribution_analysis,
    compute_league_top_contributors,
)
from services.llm import generate_interpretation, is_llm_configured
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
)

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
WINPROB_APP_PATH = (Path(__file__).resolve().parents[1] / "winprob_wizard_app.html").resolve()
WINPROB_HYPOTHETICAL_APP_PATH = (Path(__file__).resolve().parents[1] / "winprob_hypothetical_app.html").resolve()
PBP_GAME_STATES_ROOT = (Path(__file__).resolve().parents[2] / "data" / "pbp" / "processed" / "game_states").resolve()
STATES_PARQUET_FILENAME_TEMPLATE = "_states_{season}_{phase}.parquet"


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
        season_dir = PBP_GAME_STATES_ROOT / phase / season
        if not season_dir.exists():
            continue

        if home_code and road_code:
            exact = season_dir / f"{season}_{home_code}_{road_code}_{game_id}.json"
            if exact.exists():
                return exact, phase

        matches = sorted(season_dir.glob(f"{season}_*_*_{game_id}.json"))
        if matches:
            return matches[0], phase

    return None, None


def _find_timeline_parquet_file(
    season: str,
    game_type: Optional[str] = None,
) -> tuple[Optional[Path], Optional[str]]:
    for phase in _timeline_phase_candidates(game_type):
        season_dir = PBP_GAME_STATES_ROOT / phase / season
        parquet_path = _build_timeline_parquet_path(season_dir, season=season, phase=phase)
        if parquet_path.exists():
            return parquet_path, phase
    return None, None


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

    # Primary path: push predicate down to parquet so only the matching game row is read.
    table = pq.read_table(
        parquet_path,
        columns=["payload_json"],
        filters=filters,
        use_threads=False,
    )
    if table.num_rows > 0 and "payload_json" in table.column_names:
        payload = _parse_payload(table.column("payload_json")[0].as_py())
        if payload is not None:
            return payload

    # Fallback: if strict home/road filter missed, retry by game_id only.
    if home_code or road_code:
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
                    return payload
            payload = _parse_payload(cols.get("payload_json")[0].as_py() if "payload_json" in cols else None)
            if payload is not None:
                return payload

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
        "home_off_rating": _safe_float(home_ratings.get("offensive_rating")),
        "home_def_rating": _safe_float(home_ratings.get("defensive_rating")),
        "home_net_rating": _safe_float(home_ratings.get("net_rating")),
        "road_off_rating": _safe_float(road_ratings.get("offensive_rating")),
        "road_def_rating": _safe_float(road_ratings.get("defensive_rating")),
        "road_net_rating": _safe_float(road_ratings.get("net_rating")),
        "home_efg": _safe_float(home_factors.get("efg")),
        "home_ball_handling": _safe_float(home_factors.get("ball_handling")),
        "home_oreb": _safe_float(home_factors.get("oreb")),
        "home_ft_rate": _safe_float(home_factors.get("ft_rate")),
        "road_efg": _safe_float(road_factors.get("efg")),
        "road_ball_handling": _safe_float(road_factors.get("ball_handling")),
        "road_oreb": _safe_float(road_factors.get("oreb")),
        "road_ft_rate": _safe_float(road_factors.get("ft_rate")),
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
    return {"commit": GIT_COMMIT}


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
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {exc}")

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
        raise HTTPException(status_code=500, detail=f"Failed to generate hypothetical forecast: {exc}")

    return result


@router.get("/winprob/hypothetical-app", response_class=HTMLResponse)
async def get_winprob_hypothetical_app():
    if not WINPROB_HYPOTHETICAL_APP_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Hypothetical app file not found: {WINPROB_HYPOTHETICAL_APP_PATH}")
    return FileResponse(str(WINPROB_HYPOTHETICAL_APP_PATH), media_type="text/html")


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
async def get_games(season: str = Query(..., description="Season in format YYYY-YY")):
    games = await get_games_list(season)
    if not games:
        return GamesResponse(games=[])
    game_items = [GameItem(**g) for g in games]
    return GamesResponse(games=game_items)

@router.get("/teams", response_model=TeamsResponse)
async def get_teams(season: str = Query(..., description="Season in format YYYY-YY")):
    teams = await get_teams_list(season)
    return TeamsResponse(teams=teams)


@router.get("/game-timeline", response_model=GameTimelineResponse)
async def get_game_timeline(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: str = Query(..., description="Game ID"),
    game_type: Optional[str] = Query(None, description="Game type (regular_season/playoffs/play_in)"),
    home_team: Optional[str] = Query(None, description="Home team abbreviation"),
    road_team: Optional[str] = Query(None, description="Road team abbreviation"),
):
    game_id_norm = _normalize_game_id_for_timeline(game_id)
    if not game_id_norm:
        raise HTTPException(status_code=400, detail="Invalid game_id")

    phase: Optional[str] = None
    timeline_path: Optional[Path] = None
    payload: Optional[dict[str, Any]] = None

    parquet_path, parquet_phase = _find_timeline_parquet_file(
        season=season,
        game_type=game_type,
    )
    if parquet_path is not None and parquet_phase is not None:
        phase = parquet_phase
        try:
            payload = _timeline_payload_from_parquet(
                parquet_path=parquet_path,
                game_id=game_id_norm,
                home_team=home_team,
                road_team=road_team,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read timeline parquet: {exc}")

    if payload is None:
        timeline_path, phase = _find_timeline_json_file(
            season=season,
            game_id=game_id_norm,
            game_type=game_type,
            home_team=home_team,
            road_team=road_team,
        )
        if timeline_path is None or phase is None:
            raise HTTPException(
                status_code=404,
                detail=f"Timeline not found for season={season}, game_id={game_id_norm}",
            )

        try:
            loaded = json.loads(timeline_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read timeline JSON: {exc}")
        payload = loaded if isinstance(loaded, dict) else None

    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid timeline payload")

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raw_events = []

    file_home, file_road = _teams_from_timeline_filename(timeline_path) if timeline_path else ("", "")
    resolved_season = str(payload.get("season") or season)
    resolved_phase = str(payload.get("phase") or phase or "regular")
    resolved_home = str(payload.get("home_team") or home_team or file_home)
    resolved_road = str(payload.get("road_team") or road_team or file_road)

    events: list[GameTimelineEvent] = []
    wp_states: list[dict[str, Any]] = []
    wp_event_positions: list[int] = []
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
        possession_numeric = _timeline_possession_numeric(
            event=event,
            home_team=resolved_home,
            road_team=resolved_road,
        )

        events.append(
            GameTimelineEvent(
                event_index=_to_int_or_none(event.get("event_index")),
                period=period,
                clock=clock,
                description=str(event.get("description") or ""),
                possession_after_side=str(event.get("possession_after_side") or "") or None,
                possession_team_tricode=str(event.get("possession_team_tricode") or "") or None,
                game_log_state=GameTimelineState(
                    pts_home=pts_home,
                    pts_road=pts_road,
                ),
            )
        )

        seconds_left = _clock_to_seconds_left(clock, period)
        if period and seconds_left is not None and pts_home is not None and pts_road is not None:
            wp_states.append(
                {
                    "quarter": period,
                    "seconds_left": seconds_left,
                    "differential": pts_home - pts_road,
                    "possession_numeric": possession_numeric,
                }
            )
            wp_event_positions.append(len(events) - 1)

    if wp_states:
        from admin.winprob_models import DEFAULT_OUTPUT_ROOT, predict_home_winprob_batch

        try:
            wp_probs = predict_home_winprob_batch(
                season=resolved_season,
                output_root=str(DEFAULT_OUTPUT_ROOT),
                phase=resolved_phase,
                states=wp_states,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to compute timeline win probability: {exc}")

        for event_idx, p_home in zip(wp_event_positions, wp_probs):
            if p_home is not None:
                events[event_idx].home_win_prob = float(p_home)

    validation = payload.get("validation")
    validation_match = validation.get("match") if isinstance(validation, dict) else None

    return GameTimelineResponse(
        season=resolved_season,
        phase=resolved_phase,
        game_id=_normalize_game_id_for_timeline(str(payload.get("game_id") or game_id_norm)),
        game_date=str(payload.get("game_date") or "") or None,
        game_type=_normalize_game_type_for_timeline(payload.get("game_type") or game_type) or None,
        home_team=resolved_home,
        road_team=resolved_road,
        events=events,
        validation_match=validation_match,
    )


@router.get("/decomposition", response_model=DecompositionResponse)
async def get_decomposition(
    season: str = Query(..., description="Season in format YYYY-YY"),
    game_id: str = Query(..., description="Game ID"),
    model_id: Optional[str] = Query(None, description="Deprecated, ignored"),
    factor_type: str = Query("eight_factors", description="Factor type: eight_factors (default)"),
):
    # Load pre-calculated contributions for the season
    contrib_data = await load_contributions(season)
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


@router.post("/interpretation", response_model=InterpretationResponse)
async def get_interpretation(request: InterpretationRequest):
    """Get AI interpretation of factor contributions for a game.

    First checks for pre-generated interpretation, then falls back to real-time generation.
    """
    # Try to get pre-generated interpretation first
    # Pass model_id so we only use pre-generated if it matches the decomposition model
    if hasattr(request, 'season') and request.season and request.game_id:
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
        raise HTTPException(status_code=503, detail="Interpretation service not configured")

    # Build normalized, flat payload expected by llm.py prompt builder
    decomposition_data = _build_llm_decomposition_data(request)

    interpretation = await generate_interpretation(
        decomposition_data=decomposition_data,
        factor_type=request.factor_type,
        model_id=request.model_id,
    )

    if interpretation is None:
        raise HTTPException(status_code=503, detail="Failed to generate interpretation")

    # Real-time uses fallback model (gpt-4o-mini or claude-3-5-haiku)
    return InterpretationResponse(interpretation=interpretation, model="gpt-4o-mini")


@router.get("/league-summary", response_model=LeagueSummaryResponse)
async def get_league_summary(
    season: str = Query(..., description="Season in format YYYY-YY"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    exclude_playoffs: bool = Query(True, description="Exclude playoff, play-in, and NBA Cup final games"),
    last_n_games: Optional[int] = Query(None, description="Use each team's last N games"),
):
    df = await get_normalized_data_with_possessions(season)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    # Get date bounds for the season
    first_game_date = df["game_date"].min().strftime("%Y-%m-%d") if len(df) > 0 else None
    last_game_date = df["game_date"].max().strftime("%Y-%m-%d") if len(df) > 0 else None

    team_stats_df = compute_league_aggregates(
        df=df,
        start_date=start_date,
        end_date=end_date,
        exclude_playoffs=exclude_playoffs,
        last_n_games=last_n_games,
    )

    teams = []
    for _, row in team_stats_df.iterrows():
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
        ))

    numeric_cols = [
        "win_pct", "ppg", "opp_ppg", "fg_pct", "fg3_pct", "ft_pct",
        "efg_pct", "oreb_pct", "dreb_pct", "tov_pct", "ball_handling",
        "ft_rate", "off_rating", "def_rating", "net_rating",
        "opp_efg_pct", "opp_tov_pct", "opp_ft_rate", "pace",
        "sos", "off_sos", "def_sos", "adj_net_rating", "adj_off_rating", "adj_def_rating",
    ]
    league_averages = {}
    for col in numeric_cols:
        if col in team_stats_df.columns:
            league_averages[col] = round(team_stats_df[col].mean(), 1)

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
):
    df = await get_normalized_data_with_possessions(season)
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
):
    """Analyze a team's net rating decomposition over a period using contribution JSON."""
    del model_id  # Explicitly ignored for backwards compatibility.

    # Load season data
    df = await get_normalized_data_with_possessions(season)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    # Check if team exists
    if team not in df["team"].unique():
        raise HTTPException(status_code=404, detail="Team not found in this season")

    # Load pre-calculated per-game contributions
    contrib_data = await load_contributions(season)
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
):
    """Get top positive and negative contributors to net rating across all teams."""
    response_model_id = model_id or "json_contributions"

    # Load season data
    df = await get_normalized_data_with_possessions(season)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    # Load pre-calculated per-game contributions
    contrib_data = await load_contributions(season)
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
    if key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")

    clear_cache()
    return {"status": "ok", "message": "Cache cleared"}
