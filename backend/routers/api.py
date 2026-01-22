from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from config import get_available_seasons, AVAILABLE_MODELS
from services.data_loader import (
    get_normalized_season_data,
    get_games_list,
    get_teams_list,
    get_game_data,
    load_model,
)
from services.calculations import (
    compute_four_factors,
    compute_game_ratings,
    compute_factor_differentials,
    compute_decomposition,
    compute_league_aggregates,
    compute_trend_series,
)
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
)

router = APIRouter(prefix="/api")

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
    "off_rating": "Offensive Rating",
    "def_rating": "Defensive Rating",
    "net_rating": "Net Rating",
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

    home_ratings = compute_game_ratings(home_team_row, road_team_row)
    road_ratings = compute_game_ratings(road_team_row, home_team_row)

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

    response = DecompositionResponse(
        game_id=game_id,
        game_date=game_data["game_date"],
        home_team=game_data["home_team"],
        road_team=game_data["road_team"],
        home_pts=home_pts,
        road_pts=road_pts,
        actual_margin=actual_margin,
        predicted_margin=decomposition["predicted_margin"],
        factor_type=factor_type,
        home_factors=home_factors,
        road_factors=road_factors,
        contributions=decomposition["contributions"],
        intercept=decomposition["intercept"],
        home_ratings=home_ratings,
        road_ratings=road_ratings,
    )

    if factor_type == "eight_factors":
        response.factor_values = decomposition.get("factor_values")
        response.league_averages = decomposition.get("league_averages")

    return response

@router.get("/league-summary", response_model=LeagueSummaryResponse)
async def get_league_summary(
    season: str = Query(..., description="Season in format YYYY-YY"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    df = await get_normalized_season_data(season)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    team_stats_df = compute_league_aggregates(df, start_date, end_date)

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
            opp_ft_rate=float(row["opp_ft_rate"]),
        ))

    numeric_cols = [
        "win_pct", "ppg", "opp_ppg", "fg_pct", "fg3_pct", "ft_pct",
        "efg_pct", "oreb_pct", "dreb_pct", "tov_pct", "ball_handling",
        "ft_rate", "off_rating", "def_rating", "net_rating",
        "opp_efg_pct", "opp_tov_pct", "opp_ft_rate"
    ]
    league_averages = {}
    for col in numeric_cols:
        if col in team_stats_df.columns:
            league_averages[col] = round(team_stats_df[col].mean(), 1)

    return LeagueSummaryResponse(teams=teams, league_averages=league_averages)

@router.get("/trends", response_model=TrendsResponse)
async def get_trends(
    season: str = Query(..., description="Season in format YYYY-YY"),
    team: str = Query(..., description="Team abbreviation"),
    stat: str = Query(..., description="Statistic to plot"),
):
    df = await get_normalized_season_data(season)
    if df is None:
        raise HTTPException(status_code=404, detail="Season data not found")

    if team not in df["team"].unique():
        raise HTTPException(status_code=404, detail="Team not found")

    if stat not in STAT_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid stat: {stat}")

    trend_df = compute_trend_series(df, team, stat)

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

    return TrendsResponse(
        team=team,
        stat=stat,
        stat_label=STAT_LABELS.get(stat, stat),
        data=data,
        season_average=season_average,
    )
