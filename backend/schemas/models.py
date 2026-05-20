from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SeasonResponse(BaseModel):
    seasons: List[str]

class GameItem(BaseModel):
    game_id: str
    date: str
    home_team: str
    road_team: str
    game_type: str
    home_pts: int
    road_pts: int
    is_overtime: bool = False
    label: str

class GamesResponse(BaseModel):
    games: List[GameItem]

class TeamsResponse(BaseModel):
    teams: List[str]

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

class FactorRange(BaseModel):
    """Interquartile range for a factor (Q1 and Q3 values)."""
    q1: float
    q3: float

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
    factor_ranges: Optional[Dict[str, FactorRange]] = None
    linescore: Optional[LinescoreData] = None
    is_overtime: bool = False
    overtime_count: int = 0
    game_type: Optional[str] = None


class GameTimelineState(BaseModel):
    pts_home: Optional[int] = None
    pts_road: Optional[int] = None


class GameTimelineEvent(BaseModel):
    event_index: Optional[int] = None
    period: Optional[int] = None
    clock: Optional[str] = None
    description: str = ""
    possession_after_side: Optional[str] = None
    possession_team_tricode: Optional[str] = None
    home_win_prob: Optional[float] = None
    game_log_state: GameTimelineState


class GameTimelineResponse(BaseModel):
    season: str
    phase: str
    game_id: str
    game_date: Optional[str] = None
    game_type: Optional[str] = None
    home_team: str
    road_team: str
    excitement_factor: Optional[float] = None
    comeback_factor: Optional[float] = None
    excitement_percentile: Optional[float] = None
    comeback_percentile: Optional[float] = None
    events: List[GameTimelineEvent]
    validation_match: Optional[bool] = None


class GameRun(BaseModel):
    rank: int
    run_side: Optional[str] = None
    run_team: Optional[str] = None
    start_possession_index: int
    end_possession_index: int
    possession_count: int
    start_event_index: Optional[int] = None
    end_event_index: Optional[int] = None
    start_period: Optional[int] = None
    start_clock: Optional[str] = None
    end_period: Optional[int] = None
    end_clock: Optional[str] = None
    start_description: str = ""
    end_description: str = ""
    home_win_prob_start: Optional[float] = None
    home_win_prob_end: Optional[float] = None
    delta_home_win_prob: float
    run_score_numerator_value: float
    run_score: float
    abs_run_score: float
    home_score_start: Optional[int] = None
    road_score_start: Optional[int] = None
    home_score_end: Optional[int] = None
    road_score_end: Optional[int] = None
    home_points_scored: Optional[int] = None
    road_points_scored: Optional[int] = None
    score_margin_delta: Optional[int] = None


class GameRunsResponse(BaseModel):
    season: str
    phase: str
    game_id: str
    game_date: Optional[str] = None
    game_type: Optional[str] = None
    home_team: str
    road_team: str
    max_possessions: Optional[int] = None
    min_possessions: int
    min_margin: int
    run_alpha: float
    numerator: str
    runs: List[GameRun]


class PBPTraditionalBoxScorePlayer(BaseModel):
    player_id: Optional[int] = None
    player_name: str
    is_starter: bool = False
    team_id: int
    team_abbreviation: str
    minutes: str
    pts: int
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    reb: int
    ast: int
    tov: int
    stl: int
    blk: int
    oreb: int
    dreb: int
    pf: int
    plus_minus: int


class PBPTraditionalBoxScoreResponse(BaseModel):
    season: str
    phase: str
    game_id: str
    game_date: Optional[str] = None
    game_type: Optional[str] = None
    home_team: str
    road_team: str
    source: str
    minutes_plus_minus_source: Optional[str] = None
    home_players: List[PBPTraditionalBoxScorePlayer]
    road_players: List[PBPTraditionalBoxScorePlayer]


class PlayerGameFact(BaseModel):
    game_id: str
    season: str
    game_date: Optional[str] = None
    game_type: Optional[str] = None
    player_id: int
    player_name: str
    team_id: int
    team_abbreviation: str
    opponent_team_id: Optional[int] = None
    opponent_abbreviation: Optional[str] = None
    home_or_road: str
    is_starter: Optional[bool] = None
    position: Optional[str] = None
    status_comment: Optional[str] = None
    minutes: str
    seconds_played: int
    pts: int
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    oreb: int
    dreb: int
    reb: int
    ast: int
    stl: int
    blk: int
    tov: int
    pf: int
    plus_minus: int
    possessions_team: Optional[float] = None
    possessions_opp: Optional[float] = None
    source_boxscore: str
    source_possessions: Optional[str] = None


class PlayerGameFactsResponse(BaseModel):
    season: str
    game_id: Optional[str] = None
    player_id: Optional[int] = None
    team_id: Optional[int] = None
    include_dnp: bool = False
    row_count: int
    rows: List[PlayerGameFact]


class PlayerShot(BaseModel):
    season: str
    pbp_phase: str
    game_id: str
    game_date: Optional[str] = None
    game_type: Optional[str] = None
    team_id: int
    team: str
    opponent_id: Optional[int] = None
    opponent: Optional[str] = None
    home_road: str
    player_id: int
    player_name: str
    shot_type: str
    result: str
    action_number: int
    action_id: int
    period: int
    clock: str
    description: str


class PlayerShotsResponse(BaseModel):
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    start_season: Optional[str] = None
    end_season: Optional[str] = None
    game_type: Optional[str] = None
    shot_type: Optional[str] = None
    result: Optional[str] = None
    team: Optional[str] = None
    opponent: Optional[str] = None
    limit: int
    offset: int
    row_count: int
    rows: List[PlayerShot]


class PlayerShotPlayer(BaseModel):
    player_id: int
    player_name: str
    teams: List[str]
    attempts: int
    makes: int
    three_pa: int
    two_pa: int
    fta: int


class PlayerShotPlayersResponse(BaseModel):
    season: str
    game_type: Optional[str] = None
    player_count: int
    players: List[PlayerShotPlayer]


class PlayerShotStreakinessRow(BaseModel):
    season: str
    game_type: str
    player_id: int
    player_name: str
    teams: List[str]
    shot_type: str
    attempts: int
    makes: int
    make_pct: float
    classification: str
    streakiness_score: Optional[float] = None
    consistency_score: Optional[float] = None
    runs: int
    runs_cluster_percentile: float
    runs_alternation_percentile: float
    transition_effect: float
    transition_percentile: float
    window_size: Optional[int] = None
    window_variance: Optional[float] = None
    window_variance_percentile: Optional[float] = None
    longest_make_run: int
    longest_make_run_percentile: float
    longest_miss_run: int
    longest_miss_run_percentile: float


class PlayerShotStreakinessResponse(BaseModel):
    season: str
    game_type: str
    shot_type: Optional[str] = None
    min_attempts: int
    simulations: int
    classification: Optional[str] = None
    row_count: int
    rows: List[PlayerShotStreakinessRow]


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
    fg2_pct: float
    fg3a_rate: float
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
    opp_ft_pct: float
    opp_fg2_pct: float
    opp_fg3_pct: float
    opp_fg3a_rate: float
    opp_tov_pct: float
    opp_ball_handling: float
    opp_oreb_pct: float
    opp_ft_rate: float
    pace: float
    sos: float
    off_sos: float
    def_sos: float
    adj_net_rating: float
    adj_off_rating: float
    adj_def_rating: float
    scope_games: Optional[int] = None
    scope_time_pct: Optional[float] = None

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
    wins: int
    losses: int
    win_pct: float
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


class InterpretationRequest(BaseModel):
    """Request body for chart interpretation."""
    game_id: str
    game_date: str
    season: Optional[str] = None  # Season for pre-generated interpretation lookup
    data_scope: Optional[str] = "all"
    home_team: str
    road_team: str
    home_pts: int
    road_pts: int
    contributions: Dict[str, float]
    predicted_rating_diff: float
    actual_rating_diff: float
    factor_type: str
    model_id: Optional[str] = None
    home_factors: Optional[Dict[str, float]] = None
    road_factors: Optional[Dict[str, float]] = None
    home_ratings: Optional[Dict[str, float]] = None
    road_ratings: Optional[Dict[str, float]] = None
    league_averages: Optional[Dict[str, float]] = None
    factor_ranges: Optional[Dict[str, Dict[str, float]]] = None


class InterpretationResponse(BaseModel):
    """Response for chart interpretation."""
    interpretation: str
    model: Optional[str] = None
