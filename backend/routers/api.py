from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import subprocess
from config import get_available_seasons, AVAILABLE_MODELS, ADMIN_SECRET_KEY
from services.cache import clear_cache
from services.data_loader import (
    get_normalized_season_data,
    get_normalized_data_with_possessions,
    get_games_list,
    get_teams_list,
    get_game_data,
    load_model,
    discover_season_models,
)
from services.calculations import (
    compute_four_factors,
    compute_game_ratings,
    compute_factor_differentials,
    compute_decomposition,
    compute_league_aggregates,
    compute_trend_series,
    compute_league_average,
    compute_contribution_analysis,
    compute_league_top_contributors,
)
from services.llm import generate_interpretation, is_llm_configured
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
    model_id: str = Query(..., description="Model ID"),
    factor_type: str = Query("four_factors", description="Factor type: four_factors or eight_factors"),
):
    game_data = await get_game_data(season, game_id)
    if game_data is None:
        raise HTTPException(status_code=404, detail="Game not found")

    model_config = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    if model_config is None:
        raise HTTPException(status_code=404, detail="Model not found")

    model = await load_model(model_config["file"])
    if model is None:
        raise HTTPException(status_code=500, detail="Failed to load model")

    # Load season data to compute league average ratings
    season_df = await get_normalized_data_with_possessions(season)
    league_avg_off_rating = None
    league_avg_def_rating = None
    league_avg_pace = None
    if season_df is not None and len(season_df) > 0:
        # Use compute_league_aggregates to get team stats with computed ratings
        team_stats = compute_league_aggregates(season_df, None, None, exclude_playoffs=False)
        if len(team_stats) > 0:
            league_avg_off_rating = round(team_stats["off_rating"].mean(), 1)
            league_avg_def_rating = round(team_stats["def_rating"].mean(), 1)
            if "pace" in team_stats.columns:
                league_avg_pace = round(team_stats["pace"].mean(), 1)

    home_row = game_data["home"]
    road_row = game_data["road"]

    home_team_row = {
        "fgm": home_row.get("fgm", 0),
        "fga": home_row.get("fga", 0),
        "fg3m": home_row.get("fg3m", 0),
        "fg3a": home_row.get("fg3a", 0),
        "ftm": home_row.get("ftm", 0),
        "fta": home_row.get("fta", 0),
        "oreb": home_row.get("oreb", 0),
        "dreb": home_row.get("dreb", 0),
        "tov": home_row.get("tov", 0),
        "pts": home_row.get("pts", 0),
    }

    road_team_row = {
        "fgm": road_row.get("fgm", 0),
        "fga": road_row.get("fga", 0),
        "fg3m": road_row.get("fg3m", 0),
        "fg3a": road_row.get("fg3a", 0),
        "ftm": road_row.get("ftm", 0),
        "fta": road_row.get("fta", 0),
        "oreb": road_row.get("oreb", 0),
        "dreb": road_row.get("dreb", 0),
        "tov": road_row.get("tov", 0),
        "pts": road_row.get("pts", 0),
    }

    home_factors = compute_four_factors(home_team_row, road_team_row)
    road_factors = compute_four_factors(road_team_row, home_team_row)

    # Extract actual possessions and minutes from game_data
    actual_poss_home = game_data.get("actual_possessions_home")
    actual_poss_road = game_data.get("actual_possessions_road")
    actual_mins_home = game_data.get("actual_minutes_home")
    actual_mins_road = game_data.get("actual_minutes_road")

    home_ratings = compute_game_ratings(
        home_team_row, road_team_row,
        actual_possessions=actual_poss_home,
        opp_actual_possessions=actual_poss_road,
        actual_minutes=actual_mins_home,
    )
    road_ratings = compute_game_ratings(
        road_team_row, home_team_row,
        actual_possessions=actual_poss_road,
        opp_actual_possessions=actual_poss_home,
        actual_minutes=actual_mins_road,
    )

    differentials = compute_factor_differentials(home_factors, road_factors)

    try:
        decomposition = compute_decomposition(
            differentials=differentials,
            model=model,
            factor_type=factor_type,
            home_factors=home_factors,
            away_factors=road_factors,
            league_averages=None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    home_pts = int(home_row.get("pts", 0) or 0)
    road_pts = int(road_row.get("pts", 0) or 0)
    actual_margin = home_pts - road_pts

    # Process linescore data
    linescore_data = game_data.get("linescore")
    linescore_response = None
    is_overtime = False
    overtime_count = 0

    if linescore_data:
        linescore_response = LinescoreData(
            home=QuarterScores(**linescore_data["home"]),
            road=QuarterScores(**linescore_data["road"]),
        )
        home_ot = linescore_data["home"]["ot"]
        road_ot = linescore_data["road"]["ot"]
        is_overtime = home_ot > 0 or road_ot > 0

        # Calculate OT periods from minutes if available
        if actual_mins_home and actual_mins_home > 0:
            game_minutes = actual_mins_home / 5
            if game_minutes > 48:
                overtime_count = int((game_minutes - 48) / 5)

    # Actual rating differential is simply the home team's net rating
    # (home_off - home_def), which already captures the full game outcome.
    # The road team's net rating is the mirror image, so subtracting it would double-count.
    actual_rating_diff = home_ratings.get("net_rating", 0)

    response = DecompositionResponse(
        game_id=game_id,
        game_date=game_data["game_date"],
        home_team=game_data["home_team"],
        road_team=game_data["road_team"],
        home_pts=home_pts,
        road_pts=road_pts,
        actual_margin=actual_margin,
        actual_rating_diff=round(actual_rating_diff, 2),
        predicted_rating_diff=decomposition["predicted_rating_diff"],
        factor_type=factor_type,
        home_factors=home_factors,
        road_factors=road_factors,
        contributions=decomposition["contributions"],
        intercept=decomposition["intercept"],
        home_ratings=home_ratings,
        road_ratings=road_ratings,
        linescore=linescore_response,
        is_overtime=is_overtime,
        overtime_count=overtime_count,
        game_type=game_data["home"].get("game_type"),
    )

    if factor_type == "eight_factors":
        response.factor_values = decomposition.get("factor_values")

    # Include league averages for both factor types, adding ratings averages
    league_avgs = decomposition.get("league_averages", {}) or {}
    # Normalize oreb_pct to oreb for consistency with home_factors/road_factors
    if "oreb_pct" in league_avgs and "oreb" not in league_avgs:
        league_avgs["oreb"] = league_avgs.pop("oreb_pct")
    if league_avg_off_rating is not None:
        league_avgs["off_rating"] = league_avg_off_rating
    if league_avg_def_rating is not None:
        league_avgs["def_rating"] = league_avg_def_rating
    if league_avg_pace is not None:
        league_avgs["pace"] = league_avg_pace
    response.league_averages = league_avgs

    # Include factor ranges for AI summary context
    factor_ranges = decomposition.get("factor_ranges", {}) or {}
    # Normalize oreb_pct to oreb for consistency
    if "oreb_pct" in factor_ranges and "oreb" not in factor_ranges:
        factor_ranges["oreb"] = factor_ranges.pop("oreb_pct")
    if factor_ranges:
        response.factor_ranges = factor_ranges

    return response


@router.post("/interpretation", response_model=InterpretationResponse)
async def get_interpretation(request: InterpretationRequest):
    """Generate AI interpretation of factor contributions for a game."""
    if not is_llm_configured():
        raise HTTPException(status_code=503, detail="Interpretation service not configured")

    # Build data dict for LLM service
    decomposition_data = {
        "game_id": request.game_id,
        "game_date": request.game_date,
        "home_team": request.home_team,
        "road_team": request.road_team,
        "home_pts": request.home_pts,
        "road_pts": request.road_pts,
        "contributions": request.contributions,
        "predicted_rating_diff": request.predicted_rating_diff,
        "actual_rating_diff": request.actual_rating_diff,
        "home_factors": request.home_factors,
        "road_factors": request.road_factors,
        "league_averages": request.league_averages,
        "factor_ranges": request.factor_ranges,
    }

    interpretation = await generate_interpretation(
        decomposition_data=decomposition_data,
        factor_type=request.factor_type,
        model_id=request.model_id,
    )

    if interpretation is None:
        raise HTTPException(status_code=503, detail="Failed to generate interpretation")

    return InterpretationResponse(interpretation=interpretation)


@router.get("/league-summary", response_model=LeagueSummaryResponse)
async def get_league_summary(
    season: str = Query(..., description="Season in format YYYY-YY"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    exclude_playoffs: bool = Query(True, description="Exclude playoff, play-in, and NBA Cup final games"),
):
    df = await get_normalized_data_with_possessions(season)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    # Get date bounds for the season
    first_game_date = df["game_date"].min().strftime("%Y-%m-%d") if len(df) > 0 else None
    last_game_date = df["game_date"].max().strftime("%Y-%m-%d") if len(df) > 0 else None

    team_stats_df = compute_league_aggregates(df, start_date, end_date, exclude_playoffs)

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
        ))

    numeric_cols = [
        "win_pct", "ppg", "opp_ppg", "fg_pct", "fg3_pct", "ft_pct",
        "efg_pct", "oreb_pct", "dreb_pct", "tov_pct", "ball_handling",
        "ft_rate", "off_rating", "def_rating", "net_rating",
        "opp_efg_pct", "opp_tov_pct", "opp_ft_rate", "pace"
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
    model_id: str = Query(..., description="Season model ID"),
    date_range_type: str = Query("season", description="Type: season, last_n, or custom"),
    last_n_games: Optional[int] = Query(None, description="Number of games for last_n type"),
    start_date: Optional[str] = Query(None, description="Start date for custom type (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for custom type (YYYY-MM-DD)"),
    exclude_playoffs: bool = Query(False, description="Exclude playoff and play-in games"),
):
    """Analyze a team's net rating decomposition over a period using eight factors."""
    # Load season data
    df = await get_normalized_data_with_possessions(season)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    # Check if team exists
    if team not in df["team"].unique():
        raise HTTPException(status_code=404, detail="Team not found in this season")

    # Load the season model
    available_models = await discover_season_models()
    model_config = next((m for m in available_models if m["id"] == model_id), None)
    if model_config is None:
        raise HTTPException(status_code=404, detail="Season model not found")

    model = await load_model(model_config["file"])
    if model is None:
        raise HTTPException(status_code=500, detail="Failed to load model")

    # Verify it's a season-level model
    if model.get("model_type") != "season_level":
        raise HTTPException(status_code=400, detail="Model is not a season-level model")

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
    date_range_label = "Season-to-Date"

    if date_range_type == "last_n" and last_n_games:
        # Get last N games
        team_df = team_df.tail(last_n_games)
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
            model=model,
            start_date=filter_start_date,
            end_date=filter_end_date,
            exclude_playoffs=exclude_playoffs,
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
    model_id: str = Query(..., description="Season model ID"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    exclude_playoffs: bool = Query(False, description="Exclude playoff and play-in games"),
):
    """Get top positive and negative contributors to net rating across all teams."""
    # Load season data
    df = await get_normalized_data_with_possessions(season)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    # Load the season model
    available_models = await discover_season_models()
    model_config = next((m for m in available_models if m["id"] == model_id), None)
    if model_config is None:
        raise HTTPException(status_code=404, detail="Season model not found")

    model = await load_model(model_config["file"])
    if model is None:
        raise HTTPException(status_code=500, detail="Failed to load model")

    # Verify it's a season-level model
    if model.get("model_type") != "season_level":
        raise HTTPException(status_code=400, detail="Model is not a season-level model")

    try:
        result = compute_league_top_contributors(
            league_df=df,
            model=model,
            start_date=start_date,
            end_date=end_date,
            exclude_playoffs=exclude_playoffs,
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
        model_id=model_id,
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
