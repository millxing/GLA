from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import subprocess
from config import get_available_seasons, AVAILABLE_MODELS, ADMIN_SECRET_KEY
from services.cache import clear_cache
from services.data_loader import (
    get_normalized_season_data,
    get_normalized_data_with_possessions,
    get_games_list,
    get_teams_list,
    load_contributions,
    discover_season_models,
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
    ModelsResponse,
    ModelItem,
    DecompositionResponse,
    LeagueSummaryResponse,
    TeamStats,
    TrendsResponse,
    TrendPoint,
    LinescoreData,
    QuarterScores,
    SeasonModelItem,
    SeasonModelsResponse,
    ContributionAnalysisResponse,
    TopContributor,
    ContributionTrendPoint,
    LeagueContributorItem,
    LeagueTopContributorsResponse,
    InterpretationRequest,
    InterpretationResponse,
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

@router.get("/models", response_model=ModelsResponse)
async def get_models():
    model_items = [ModelItem(id=m["id"], name=m["name"]) for m in AVAILABLE_MODELS]
    return ModelsResponse(models=model_items)

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


@router.get("/season-models", response_model=SeasonModelsResponse)
async def get_season_models():
    """Get available season-level models for contribution analysis."""
    models = await discover_season_models()
    model_items = [SeasonModelItem(id=m["id"], name=m["name"]) for m in models]
    return SeasonModelsResponse(models=model_items)


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
