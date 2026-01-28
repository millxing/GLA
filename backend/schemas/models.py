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

class QuarterScores(BaseModel):
    q1: int
    q2: int
    q3: int
    q4: int
    ot: int = 0

class LinescoreData(BaseModel):
    home: QuarterScores
    road: QuarterScores

class DecompositionResponse(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    road_team: str
    home_pts: int
    road_pts: int
    actual_margin: int  # Raw point differential (kept for display)
    actual_rating_diff: float  # Actual net rating differential (home - road)
    predicted_rating_diff: float  # Model's predicted rating differential
    factor_type: str
    home_factors: Dict[str, float]
    road_factors: Dict[str, float]
    contributions: Dict[str, float]
    intercept: float
    home_ratings: Dict[str, float]
    road_ratings: Dict[str, float]
    factor_values: Optional[Dict[str, float]] = None
    league_averages: Optional[Dict[str, float]] = None
    linescore: Optional[LinescoreData] = None
    is_overtime: bool = False
    overtime_count: int = 0
    game_type: Optional[str] = None

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
    pace: float

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


# Contribution Analysis schemas
class SeasonModelItem(BaseModel):
    id: str
    name: str


class SeasonModelsResponse(BaseModel):
    models: List[SeasonModelItem]


class ContributionTrendPoint(BaseModel):
    """Simplified trend point for contribution analysis mini-charts."""
    game_id: str
    game_date: str
    opponent: str
    home_away: str
    value: float
    ma_5: float
    wl: str


class TopContributor(BaseModel):
    """A top contributing factor with its trend data."""
    factor: str
    factor_label: str
    value: float
    league_avg: float
    contribution: float
    trend_data: List[ContributionTrendPoint]


class ContributionAnalysisResponse(BaseModel):
    """Response for contribution analysis endpoint."""
    team: str
    season: str
    date_range_label: str
    start_date: str
    end_date: str
    games_analyzed: int
    net_rating: float
    predicted_net_rating: float
    contributions: Dict[str, float]
    factor_values: Dict[str, float]
    league_averages: Dict[str, float]
    top_contributors: List[TopContributor]
    intercept: float


class LeagueContributorItem(BaseModel):
    """A single contributor (team + factor) for league-wide top contributors."""
    team: str
    factor: str
    factor_label: str
    value: float
    contribution: float


class LeagueTopContributorsResponse(BaseModel):
    """Response for league-wide top contributors endpoint."""
    season: str
    start_date: str
    end_date: str
    model_id: str
    top_positive: List[LeagueContributorItem]
    top_negative: List[LeagueContributorItem]
    league_averages: Dict[str, float]
    coefficients: Dict[str, float]  # Model coefficients for debugging
