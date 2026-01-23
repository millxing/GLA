from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SeasonResponse(BaseModel):
    seasons: List[str]

class GameItem(BaseModel):
    game_id: str
    date: str
    home_team: str
    road_team: str
    home_pts: int
    road_pts: int
    label: str

class GamesResponse(BaseModel):
    games: List[GameItem]

class TeamsResponse(BaseModel):
    teams: List[str]

class ModelItem(BaseModel):
    id: str
    name: str

class ModelsResponse(BaseModel):
    models: List[ModelItem]

class FactorComparison(BaseModel):
    factor: str
    home_value: float
    road_value: float
    differential: float

class DecompositionResponse(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    road_team: str
    home_pts: int
    road_pts: int
    actual_margin: int
    predicted_margin: float
    factor_type: str
    home_factors: Dict[str, float]
    road_factors: Dict[str, float]
    contributions: Dict[str, float]
    intercept: float
    home_ratings: Dict[str, float]
    road_ratings: Dict[str, float]
    factor_values: Optional[Dict[str, float]] = None
    league_averages: Optional[Dict[str, float]] = None

class TeamStats(BaseModel):
    team: str
    games: int
    wins: int
    losses: int
    win_pct: float
    ppg: float
    opp_ppg: float
    fg_pct: float
    fg3_pct: float
    ft_pct: float
    efg_pct: float
    oreb_pct: float
    dreb_pct: float
    tov_pct: float
    ball_handling: float
    ft_rate: float
    off_rating: float
    def_rating: float
    net_rating: float
    opp_efg_pct: float
    opp_tov_pct: float
    opp_ball_handling: float
    opp_oreb_pct: float
    opp_ft_rate: float

class LeagueSummaryResponse(BaseModel):
    teams: List[TeamStats]
    league_averages: Dict[str, float]
    first_game_date: Optional[str] = None
    last_game_date: Optional[str] = None

class TrendPoint(BaseModel):
    game_id: str
    game_date: str
    opponent: str
    home_away: str
    value: float
    ma_5: float
    ma_10: float
    wl: str

class TrendsResponse(BaseModel):
    team: str
    stat: str
    stat_label: str
    data: List[TrendPoint]
    season_average: float
    league_average: float
