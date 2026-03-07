import { useState, useEffect, useMemo } from 'react'
import { getSeasons, getLeagueSummary, getLeagueTopContributors } from '../api'
import { usePersistedState } from '../hooks/usePersistedState'
import { ScatterChart, Scatter, XAxis, YAxis, ReferenceLine, ResponsiveContainer, LabelList, Tooltip } from 'recharts'
import { useNavigate, useSearchParams } from 'react-router-dom'
import './LeagueSummary.css'

const VIEW_FOUR_FACTORS = 'four_factors'
const VIEW_SHOOTING = 'shooting'
const VIEW_SOS_ADJUSTMENTS = 'sos_adjustments'

// BH = Ball Handling = 100 - TOV% (higher is better)
// isOpp indicates opponent stats (for gradient reversal)
const FOUR_FACTOR_COLUMNS = [
  { key: 'team', label: 'Team', labelLine2: '', sortable: true },
  { key: 'games', label: 'GP', labelLine2: '', sortable: true },
  { key: 'wins', label: 'W', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'losses', label: 'L', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'win_pct', label: 'PCT', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'off_rating', label: 'ORtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'def_rating', label: 'DRtg', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'net_rating', label: 'NRtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'efg_pct', label: 'eFG%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'ball_handling', label: 'BH', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'oreb_pct', label: 'OREB%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'ft_rate', label: 'FT Rate', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'opp_efg_pct', label: 'eFG%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_ball_handling', label: 'BH', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_oreb_pct', label: 'OREB%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_ft_rate', label: 'FT Rate', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'pace', label: 'Pace', labelLine2: '', sortable: true, higherBetter: null },
]

const FOUR_FACTOR_SCOPED_COLUMNS = [
  { key: 'team', label: 'Team', labelLine2: '', sortable: true },
  { key: 'scope_games', label: 'G', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'scope_time_pct', label: '% of Time', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'off_rating', label: 'ORtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'def_rating', label: 'DRtg', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'net_rating', label: 'NRtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'efg_pct', label: 'eFG%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'ball_handling', label: 'BH', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'oreb_pct', label: 'OREB%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'ft_rate', label: 'FT Rate', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'opp_efg_pct', label: 'eFG%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_ball_handling', label: 'BH', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_oreb_pct', label: 'OREB%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_ft_rate', label: 'FT Rate', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'pace', label: 'Pace', labelLine2: '', sortable: true, higherBetter: null },
]

const SOS_COLUMNS = [
  { key: 'team', label: 'Team', labelLine2: '', sortable: true },
  { key: 'games', label: 'GP', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'wins', label: 'W', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'losses', label: 'L', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'win_pct', label: 'PCT', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'net_rating', label: 'NRtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'off_rating', label: 'ORtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'def_rating', label: 'DRtg', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'sos', label: 'SOS', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'off_sos', label: 'Off SOS', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'def_sos', label: 'Def SOS', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'adj_net_rating', label: 'Adj NRtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'adj_off_rating', label: 'Adj ORtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'adj_def_rating', label: 'Adj DRtg', labelLine2: '', sortable: true, higherBetter: false },
]

const SOS_SCOPED_COLUMNS = [
  { key: 'team', label: 'Team', labelLine2: '', sortable: true },
  { key: 'scope_games', label: 'G', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'scope_time_pct', label: '% of Time', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'net_rating', label: 'NRtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'off_rating', label: 'ORtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'def_rating', label: 'DRtg', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'sos', label: 'SOS', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'off_sos', label: 'Off SOS', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'def_sos', label: 'Def SOS', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'adj_net_rating', label: 'Adj NRtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'adj_off_rating', label: 'Adj ORtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'adj_def_rating', label: 'Adj DRtg', labelLine2: '', sortable: true, higherBetter: false },
]

const SHOOTING_COLUMNS = [
  { key: 'team', label: 'Team', labelLine2: '', sortable: true },
  { key: 'games', label: 'GP', labelLine2: '', sortable: true },
  { key: 'wins', label: 'W', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'losses', label: 'L', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'win_pct', label: 'PCT', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'off_rating', label: 'ORtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'def_rating', label: 'DRtg', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'net_rating', label: 'NRtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'ft_pct', label: 'FT%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'fg2_pct', label: '2pt%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'fg3_pct', label: '3pt%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'fg3a_rate', label: '3ptAr', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'opp_ft_pct', label: 'FT%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_fg2_pct', label: '2pt%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_fg3_pct', label: '3pt%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_fg3a_rate', label: '3ptAr', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
]

const SHOOTING_SCOPED_COLUMNS = [
  { key: 'team', label: 'Team', labelLine2: '', sortable: true },
  { key: 'scope_games', label: 'G', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'scope_time_pct', label: '% of Time', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'off_rating', label: 'ORtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'def_rating', label: 'DRtg', labelLine2: '', sortable: true, higherBetter: false },
  { key: 'net_rating', label: 'NRtg', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'ft_pct', label: 'FT%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'fg2_pct', label: '2pt%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'fg3_pct', label: '3pt%', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'fg3a_rate', label: '3ptAr', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'opp_ft_pct', label: 'FT%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_fg2_pct', label: '2pt%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_fg3_pct', label: '3pt%', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
  { key: 'opp_fg3a_rate', label: '3ptAr', labelLine2: 'Opp', sortable: true, higherBetter: false, isOpp: true },
]

const SECTION_START_COLUMNS = {
  [VIEW_FOUR_FACTORS]: new Set(['off_rating', 'efg_pct', 'opp_efg_pct', 'pace']),
  [VIEW_SHOOTING]: new Set(['off_rating', 'ft_pct', 'opp_ft_pct']),
  [VIEW_SOS_ADJUSTMENTS]: new Set(['net_rating', 'sos', 'adj_net_rating']),
}

const ALL_LEAGUE_SUMMARY_COLUMNS = [
  ...FOUR_FACTOR_COLUMNS,
  ...FOUR_FACTOR_SCOPED_COLUMNS,
  ...SOS_COLUMNS,
  ...SOS_SCOPED_COLUMNS,
  ...SHOOTING_COLUMNS,
  ...SHOOTING_SCOPED_COLUMNS,
]

const COLUMN_CONFIG_BY_KEY = new Map(
  ALL_LEAGUE_SUMMARY_COLUMNS.map((column) => [column.key, column])
)

const ALL_SORTABLE_COLUMN_KEYS = new Set(
  ALL_LEAGUE_SUMMARY_COLUMNS
    .filter((column) => column.sortable)
    .map((column) => column.key)
)

const TRENDS_DIRECT_STAT_COLUMNS = new Set([
  'net_rating',
  'off_rating',
  'def_rating',
  'ft_pct',
  'fg2_pct',
  'fg3_pct',
  'fg3a_rate',
  'efg_pct',
  'ball_handling',
  'oreb_pct',
  'ft_rate',
  'opp_efg_pct',
  'opp_ft_pct',
  'opp_fg2_pct',
  'opp_fg3_pct',
  'opp_fg3a_rate',
  'opp_ball_handling',
  'opp_oreb_pct',
  'opp_ft_rate',
  'pace',
])

const TRENDS_FALLBACK_BY_COLUMN = {
  team: 'net_rating',
  games: 'net_rating',
  wins: 'net_rating',
  losses: 'net_rating',
  win_pct: 'net_rating',
  scope_games: 'net_rating',
  scope_time_pct: 'net_rating',
  sos: 'net_rating',
  off_sos: 'off_rating',
  def_sos: 'def_rating',
  adj_net_rating: 'net_rating',
  adj_off_rating: 'off_rating',
  adj_def_rating: 'def_rating',
}

const TRENDS_STAT_LABELS = {
  net_rating: 'Net Rating',
  off_rating: 'Offensive Rating',
  def_rating: 'Defensive Rating',
  efg_pct: 'Effective FG%',
  ball_handling: 'Ball Handling',
  oreb_pct: 'Offensive Rebound %',
  ft_rate: 'Free Throw Rate',
  ft_pct: 'Free Throw Percentage',
  fg2_pct: '2-Point FG%',
  fg3_pct: '3-Point FG%',
  fg3a_rate: '3-Point Attempt Rate',
  opp_efg_pct: 'Opp Effective FG%',
  opp_ft_pct: 'Opp Free Throw Percentage',
  opp_fg2_pct: 'Opp 2-Point FG%',
  opp_fg3_pct: 'Opp 3-Point FG%',
  opp_fg3a_rate: 'Opp 3-Point Attempt Rate',
  opp_ball_handling: 'Opp Ball Handling',
  opp_oreb_pct: 'Opp Offensive Rebound %',
  opp_ft_rate: 'Opp Free Throw Rate',
  pace: 'Pace',
}

const resolveTrendStatForColumn = (columnKey) => {
  if (TRENDS_DIRECT_STAT_COLUMNS.has(columnKey)) return columnKey
  return TRENDS_FALLBACK_BY_COLUMN[columnKey] || 'net_rating'
}

const GLOSSARY_ITEMS = [
  { term: 'Rank', definition: 'League rank based on the currently selected sort column.' },
  { term: 'Team', definition: 'Team abbreviation.' },
  { term: 'GP', definition: 'Games Played' },
  { term: 'G', definition: 'Games containing the selected situational segment (garbage time or clutch time).' },
  { term: '% of Time', definition: 'Share of total game time spent in the selected situational segment.' },
  { term: 'W', definition: 'Wins' },
  { term: 'L', definition: 'Losses' },
  { term: 'ORtg', definition: 'Offensive Rating - Points scored per 100 possessions' },
  { term: 'DRtg', definition: 'Defensive Rating - Points allowed per 100 possessions (lower is better)' },
  { term: 'NRtg', definition: 'Net Rating - Offensive Rating minus Defensive Rating' },
  { term: 'eFG%', definition: 'Effective Field Goal Percentage - Adjusts FG% for the added value of 3-pointers. Formula: (FGM + 0.5 × 3PM) / FGA × 100' },
  { term: 'BH', definition: 'Ball Handling - Measures ability to take care of the ball. Calculated as 100 - TOV%. Higher is better.' },
  { term: 'OREB%', definition: 'Offensive Rebounding Percentage - Percentage of available offensive rebounds grabbed' },
  { term: 'FT Rate', definition: 'Free Throw Rate - Free throws made per field goal attempt (FTM / FGA × 100)' },
  { term: 'FT%', definition: 'Free Throw Percentage - Free throws made divided by free throw attempts (FTM / FTA × 100).' },
  { term: '2pt%', definition: '2-point field goal percentage: 2PT FGM / 2PT FGA × 100.' },
  { term: '3pt%', definition: '3-point field goal percentage: 3PT FGM / 3PT FGA × 100.' },
  { term: '3ptAr', definition: '3-point attempt rate: 3PA / FGA × 100.' },
  { term: 'Opp', definition: 'Opponent statistics - For defensive stats, lower opponent values are better for your team' },
  { term: 'Pace', definition: 'Average possessions per game. Higher pace indicates a faster-paced playing style.' },
  { term: 'PCT', definition: 'Winning Percentage (shown as .xxx)' },
  { term: 'SOS', definition: 'Strength of Schedule - Average net rating of opponents played.' },
  { term: 'Off SOS', definition: 'Average opponent offensive rating minus league-average offensive rating.' },
  { term: 'Def SOS', definition: 'Average opponent defensive rating minus league-average defensive rating.' },
  { term: 'Adj NRtg', definition: 'Adjusted Net Rating = NRtg + SOS.' },
  { term: 'Adj ORtg', definition: 'Adjusted Offensive Rating = ORtg - Def SOS.' },
  { term: 'Adj DRtg', definition: 'Adjusted Defensive Rating = DRtg - Off SOS.' },
  { term: 'Contribution', definition: 'How much a factor contributed to a team\'s net rating relative to league average. Calculated as: (Team Value - League Avg) × Model Coefficient. Positive contributions help the team; negative contributions hurt.' },
]

const GLOSSARY_DEFINITIONS = Object.fromEntries(
  GLOSSARY_ITEMS.map(({ term, definition }) => [term, definition])
)

const HEADER_TOOLTIP_TERMS = {
  rank: 'Rank',
  team: 'Team',
  games: 'GP',
  scope_games: 'G',
  scope_time_pct: '% of Time',
  wins: 'W',
  losses: 'L',
  win_pct: 'PCT',
  off_rating: 'ORtg',
  def_rating: 'DRtg',
  net_rating: 'NRtg',
  efg_pct: 'eFG%',
  ball_handling: 'BH',
  oreb_pct: 'OREB%',
  ft_rate: 'FT Rate',
  pace: 'Pace',
  ft_pct: 'FT%',
  fg2_pct: '2pt%',
  fg3_pct: '3pt%',
  fg3a_rate: '3ptAr',
  opp_efg_pct: 'eFG%',
  opp_ball_handling: 'BH',
  opp_oreb_pct: 'OREB%',
  opp_ft_rate: 'FT Rate',
  opp_ft_pct: 'FT%',
  opp_fg2_pct: '2pt%',
  opp_fg3_pct: '3pt%',
  opp_fg3a_rate: '3ptAr',
  sos: 'SOS',
  off_sos: 'Off SOS',
  def_sos: 'Def SOS',
  adj_net_rating: 'Adj NRtg',
  adj_off_rating: 'Adj ORtg',
  adj_def_rating: 'Adj DRtg',
}

function getHeaderTooltip(columnKey, isOpponentColumn = false) {
  const glossaryTerm = HEADER_TOOLTIP_TERMS[columnKey]
  if (!glossaryTerm) return ''

  const definition = GLOSSARY_DEFINITIONS[glossaryTerm]
  if (!definition) return ''

  if (!isOpponentColumn) return definition

  const opponentDefinition = GLOSSARY_DEFINITIONS.Opp
  return `Opponent ${glossaryTerm} - ${definition} ${opponentDefinition}`
}

const DATE_RANGE_OPTIONS = [
  { value: 'season', label: 'Season' },
  { value: 'season_regular', label: 'Regular Season' },
  { value: 'this_month', label: 'This Month' },
  { value: 'last_2_months', label: 'Last 2 months' },
  { value: 'last_3_months', label: 'Last 3 months' },
  { value: 'last_10_games', label: 'Last 10 games' },
  { value: 'last_15_games', label: 'Last 15 games' },
  { value: 'last_20_games', label: 'Last 20 games' },
  { value: 'custom', label: 'Custom Dates' },
]
const DATA_SCOPE_OPTIONS = [
  { value: 'all', label: 'All Data' },
  { value: 'garbage_filtered', label: 'Non-Garbage Time' },
  { value: 'garbage_time', label: 'Garbage Time' },
  { value: 'clutch', label: 'Clutch Time' },
  { value: 'non_clutch_time', label: 'Non-Clutch Time' },
]
const DERIVED_SCOPE_VALUES = new Set(['garbage_time', 'non_clutch_time'])
const VALID_DATE_RANGE_VALUES = new Set(DATE_RANGE_OPTIONS.map((option) => option.value))
const VALID_DATA_SCOPE_VALUES = new Set(DATA_SCOPE_OPTIONS.map((option) => option.value))
const VALID_TABLE_VIEW_VALUES = new Set([VIEW_FOUR_FACTORS, VIEW_SHOOTING, VIEW_SOS_ADJUSTMENTS])

const SORT_COLUMN_ALIASES = {
  ortg: 'off_rating',
  offensive_rating: 'off_rating',
  drtg: 'def_rating',
  defensive_rating: 'def_rating',
  nrtg: 'net_rating',
  net_rating: 'net_rating',
  sos_adjusted: 'adj_net_rating',
}

const DATE_RANGE_ALIASES = {
  regular_season: 'season_regular',
  season_to_date: 'season',
  last10games: 'last_10_games',
  last15games: 'last_15_games',
  last20games: 'last_20_games',
}

const DATA_SCOPE_ALIASES = {
  non_garbage: 'garbage_filtered',
  non_garbage_time: 'garbage_filtered',
  garbage_filtered: 'garbage_filtered',
  garbage: 'garbage_time',
  garbage_time: 'garbage_time',
  clutch_time: 'clutch',
  non_clutch: 'non_clutch_time',
  non_clutch_time: 'non_clutch_time',
}

const TABLE_VIEW_ALIASES = {
  four_factor: VIEW_FOUR_FACTORS,
  four_factors: VIEW_FOUR_FACTORS,
  shooting: VIEW_SHOOTING,
  sos: VIEW_SOS_ADJUSTMENTS,
  sos_adjustment: VIEW_SOS_ADJUSTMENTS,
  sos_adjustments: VIEW_SOS_ADJUSTMENTS,
  strength_of_schedule: VIEW_SOS_ADJUSTMENTS,
}

const SORT_DIRECTION_ALIASES = {
  ascending: 'asc',
  descending: 'desc',
}

const EFFICIENCY_CHART_MARGIN = { top: 20, right: 56, bottom: 54, left: 56 }
const TEAM_NAME_BY_ABBR = {
  ATL: 'Atlanta Hawks',
  BOS: 'Boston Celtics',
  BKN: 'Brooklyn Nets',
  BRK: 'Brooklyn Nets',
  CHH: 'Charlotte Hornets',
  CHA: 'Charlotte Hornets',
  CHO: 'Charlotte Hornets',
  CHI: 'Chicago Bulls',
  CLE: 'Cleveland Cavaliers',
  DAL: 'Dallas Mavericks',
  DEN: 'Denver Nuggets',
  DET: 'Detroit Pistons',
  GSW: 'Golden State Warriors',
  HOU: 'Houston Rockets',
  IND: 'Indiana Pacers',
  LAC: 'LA Clippers',
  LAL: 'Los Angeles Lakers',
  MEM: 'Memphis Grizzlies',
  MIA: 'Miami Heat',
  MIL: 'Milwaukee Bucks',
  MIN: 'Minnesota Timberwolves',
  NJN: 'New Jersey Nets',
  NOH: 'New Orleans Hornets',
  NOK: 'New Orleans/Oklahoma City Hornets',
  NOP: 'New Orleans Pelicans',
  NYK: 'New York Knicks',
  OKC: 'Oklahoma City Thunder',
  ORL: 'Orlando Magic',
  PHI: 'Philadelphia 76ers',
  PHX: 'Phoenix Suns',
  POR: 'Portland Trail Blazers',
  SAC: 'Sacramento Kings',
  SAS: 'San Antonio Spurs',
  SEA: 'Seattle SuperSonics',
  TOR: 'Toronto Raptors',
  UTA: 'Utah Jazz',
  VAN: 'Vancouver Grizzlies',
  WAS: 'Washington Wizards',
  WSB: 'Washington Bullets',
}

const TEAM_CITY_BY_ABBR = {
  ATL: 'Atlanta',
  BOS: 'Boston',
  BKN: 'Brooklyn',
  BRK: 'Brooklyn',
  CHH: 'Charlotte',
  CHA: 'Charlotte',
  CHO: 'Charlotte',
  CHI: 'Chicago',
  CLE: 'Cleveland',
  DAL: 'Dallas',
  DEN: 'Denver',
  DET: 'Detroit',
  GSW: 'Golden State',
  HOU: 'Houston',
  IND: 'Indiana',
  LAC: 'LA Clippers',
  LAL: 'LA Lakers',
  MEM: 'Memphis',
  MIA: 'Miami',
  MIL: 'Milwaukee',
  MIN: 'Minnesota',
  NJN: 'New Jersey',
  NOH: 'New Orleans',
  NOK: 'New Orleans/Oklahoma City',
  NOP: 'New Orleans',
  NYK: 'New York',
  OKC: 'Oklahoma City',
  ORL: 'Orlando',
  PHI: 'Philadelphia',
  PHX: 'Phoenix',
  POR: 'Portland',
  SAC: 'Sacramento',
  SAS: 'San Antonio',
  SEA: 'Seattle',
  TOR: 'Toronto',
  UTA: 'Utah',
  VAN: 'Vancouver',
  WAS: 'Washington',
  WSB: 'Washington',
}

function renderXAxisTitle(props) {
  const { viewBox } = props || {}
  if (!viewBox) return null
  const x = viewBox.x
  const y = viewBox.y + viewBox.height + 28
  return (
    <text
      x={x}
      y={y}
      textAnchor="middle"
      dominantBaseline="middle"
      fill="var(--color-text-secondary)"
      fontSize="15"
    >
      Offensive Rating
    </text>
  )
}

function renderYAxisTitle(props) {
  const { viewBox } = props || {}
  if (!viewBox) return null
  const x = viewBox.x - 28
  const y = viewBox.y
  return (
    <text
      x={x}
      y={y}
      textAnchor="middle"
      dominantBaseline="middle"
      fill="var(--color-text-secondary)"
      fontSize="15"
      transform={`rotate(-90 ${x} ${y})`}
    >
      Defensive Rating
    </text>
  )
}

function EfficiencyTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null
  const point = payload[0]?.payload
  if (!point) return null

  return (
    <div className="efficiency-tooltip">
      <div className="efficiency-tooltip-team">{point.team_name}</div>
      <div className="efficiency-tooltip-row">
        <span>Offensive Rating</span>
        <span>{point.off_rating.toFixed(1)}</span>
      </div>
      <div className="efficiency-tooltip-row">
        <span>Defensive Rating</span>
        <span>{point.def_rating.toFixed(1)}</span>
      </div>
      <div className="efficiency-tooltip-row">
        <span>Net Rating</span>
        <span>{point.net_rating.toFixed(1)}</span>
      </div>
    </div>
  )
}

function normalizeQueryToken(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/%/g, ' pct ')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

function readSearchParam(searchParams, ...keys) {
  for (const key of keys) {
    const value = searchParams.get(key)
    if (value) return value
  }
  return ''
}

function normalizeDateRangeParam(value) {
  const token = normalizeQueryToken(value)
  if (!token) return ''
  if (VALID_DATE_RANGE_VALUES.has(token)) return token
  return DATE_RANGE_ALIASES[token] || ''
}

function normalizeDataScopeParam(value) {
  const token = normalizeQueryToken(value)
  if (!token) return ''
  if (VALID_DATA_SCOPE_VALUES.has(token)) return token
  return DATA_SCOPE_ALIASES[token] || ''
}

function normalizeTableViewParam(value) {
  const token = normalizeQueryToken(value)
  if (!token) return ''
  if (VALID_TABLE_VIEW_VALUES.has(token)) return token
  return TABLE_VIEW_ALIASES[token] || ''
}

function normalizeSortDirectionParam(value) {
  const token = normalizeQueryToken(value)
  if (!token) return ''
  if (token === 'asc' || token === 'desc') return token
  return SORT_DIRECTION_ALIASES[token] || ''
}

function normalizeSortColumnParam(value) {
  const token = normalizeQueryToken(value)
  if (!token) return ''
  if (ALL_SORTABLE_COLUMN_KEYS.has(token)) return token
  const aliasedKey = SORT_COLUMN_ALIASES[token]
  return aliasedKey && ALL_SORTABLE_COLUMN_KEYS.has(aliasedKey) ? aliasedKey : ''
}

function getDefaultSortForView(view) {
  return view === VIEW_SOS_ADJUSTMENTS
    ? { column: 'adj_net_rating', direction: 'desc' }
    : { column: 'net_rating', direction: 'desc' }
}

function getDefaultSortDirectionForColumn(columnKey) {
  if (columnKey === 'team') return 'asc'
  const column = COLUMN_CONFIG_BY_KEY.get(columnKey)
  if (column?.higherBetter === false) return 'asc'
  return 'desc'
}

function normalizeDateInputValue(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value || '') ? value : ''
}

function LeagueSummary() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [seasons, setSeasons] = useState([])
  const [selectedSeason, setSelectedSeason] = usePersistedState('leaguesummary_season', '')
  const [selectedDataScope, setSelectedDataScope] = usePersistedState('leaguesummary_datascope', 'all')
  const [dateRangePreset, setDateRangePreset] = usePersistedState('leaguesummary_daterange', 'season')
  const [customStartDate, setCustomStartDate] = useState('')
  const [customEndDate, setCustomEndDate] = useState('')
  const [seasonBounds, setSeasonBounds] = useState({ first: '', last: '' })
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sortColumn, setSortColumn] = useState('net_rating')
  const [sortDirection, setSortDirection] = useState('desc')
  const [tableView, setTableView] = usePersistedState('leaguesummary_view', VIEW_FOUR_FACTORS)
  const [glossaryExpanded, setGlossaryExpanded] = useState(false)
  const [topContributors, setTopContributors] = useState(null)
  const [contributorsLoading, setContributorsLoading] = useState(false)
  const [topContributorsNote, setTopContributorsNote] = useState('')
  const [hoverTooltip, setHoverTooltip] = useState(null)
  const isCustomDateRangeVisible = dateRangePreset === 'custom'
  const urlSelections = useMemo(() => {
    const season = readSearchParam(searchParams, 'season').trim()
    return {
      season,
      dataScope: normalizeDataScopeParam(readSearchParam(searchParams, 'scope', 'data_scope', 'dataScope')),
      dateRange: normalizeDateRangeParam(readSearchParam(searchParams, 'range', 'date_range', 'dateRange')),
      tableView: normalizeTableViewParam(readSearchParam(searchParams, 'view', 'table_view', 'tableView')),
      sortColumn: normalizeSortColumnParam(readSearchParam(searchParams, 'sort', 'sort_by', 'sortBy')),
      sortDirection: normalizeSortDirectionParam(readSearchParam(searchParams, 'direction', 'sort_direction', 'sortDirection')),
      customStartDate: normalizeDateInputValue(readSearchParam(searchParams, 'start', 'start_date', 'startDate')),
      customEndDate: normalizeDateInputValue(readSearchParam(searchParams, 'end', 'end_date', 'endDate')),
    }
  }, [searchParams])

  useEffect(() => {
    async function loadInitialData() {
      try {
        const seasonsRes = await getSeasons()
        setSeasons(seasonsRes.seasons)
        // Keep persisted season if valid, otherwise default to first
        setSelectedSeason(prev => {
          if (urlSelections.season && seasonsRes.seasons.includes(urlSelections.season)) {
            return urlSelections.season
          }
          if (prev && seasonsRes.seasons.includes(prev)) return prev
          return seasonsRes.seasons.length > 0 ? seasonsRes.seasons[0] : ''
        })
      } catch (err) {
        setError(err.message)
      }
    }
    loadInitialData()
  }, [setSelectedSeason, urlSelections.season])

  useEffect(() => {
    if (urlSelections.season && seasons.includes(urlSelections.season) && selectedSeason !== urlSelections.season) {
      setSelectedSeason(urlSelections.season)
    }
  }, [seasons, selectedSeason, setSelectedSeason, urlSelections.season])

  useEffect(() => {
    if (urlSelections.dataScope && urlSelections.dataScope !== selectedDataScope) {
      setSelectedDataScope(urlSelections.dataScope)
    }
  }, [selectedDataScope, setSelectedDataScope, urlSelections.dataScope])

  useEffect(() => {
    if (urlSelections.dateRange && urlSelections.dateRange !== dateRangePreset) {
      setDateRangePreset(urlSelections.dateRange)
    }
  }, [dateRangePreset, setDateRangePreset, urlSelections.dateRange])

  useEffect(() => {
    if (urlSelections.tableView && urlSelections.tableView !== tableView) {
      setTableView(urlSelections.tableView)
    }
  }, [setTableView, tableView, urlSelections.tableView])

  useEffect(() => {
    if (urlSelections.dateRange !== 'custom') return
    if (urlSelections.customStartDate && urlSelections.customStartDate !== customStartDate) {
      setCustomStartDate(urlSelections.customStartDate)
    }
    if (urlSelections.customEndDate && urlSelections.customEndDate !== customEndDate) {
      setCustomEndDate(urlSelections.customEndDate)
    }
  }, [customEndDate, customStartDate, urlSelections.customEndDate, urlSelections.customStartDate, urlSelections.dateRange])

  // Reset date preset when season changes
  useEffect(() => {
    const nextDateRangePreset = urlSelections.dateRange || 'season'
    const shouldUseCustomDates = nextDateRangePreset === 'custom'

    setDateRangePreset(nextDateRangePreset)
    setCustomStartDate(shouldUseCustomDates ? urlSelections.customStartDate : '')
    setCustomEndDate(shouldUseCustomDates ? urlSelections.customEndDate : '')
    setSeasonBounds({ first: '', last: '' })
    setData(null)
    setTopContributors(null)
    setTopContributorsNote('')
  }, [
    selectedSeason,
    selectedDataScope,
    urlSelections.dateRange,
    urlSelections.customStartDate,
    urlSelections.customEndDate,
    setDateRangePreset,
  ])

  // Keep persisted date preset valid when option list changes
  useEffect(() => {
    if (!DATE_RANGE_OPTIONS.some(opt => opt.value === dateRangePreset)) {
      setDateRangePreset('season')
    }
  }, [dateRangePreset, setDateRangePreset])

  useEffect(() => {
    if (!hoverTooltip) return undefined

    const hideTooltip = () => setHoverTooltip(null)
    window.addEventListener('scroll', hideTooltip, true)
    window.addEventListener('resize', hideTooltip)

    return () => {
      window.removeEventListener('scroll', hideTooltip, true)
      window.removeEventListener('resize', hideTooltip)
    }
  }, [hoverTooltip])

  const formatUtcDate = (dateObj) => {
    const y = dateObj.getUTCFullYear()
    const m = String(dateObj.getUTCMonth() + 1).padStart(2, '0')
    const d = String(dateObj.getUTCDate()).padStart(2, '0')
    return `${y}-${m}-${d}`
  }

  // Calculate API range params from preset
  const { startDate, endDate, lastNGames, excludePlayoffs } = useMemo(() => {
    if (!seasonBounds.first || !seasonBounds.last) {
      return { startDate: '', endDate: '', lastNGames: null, excludePlayoffs: false }
    }

    const seasonStart = seasonBounds.first
    const seasonEnd = seasonBounds.last

    const getMonthStart = (anchorDateStr, monthsBack) => {
      const [year, month] = anchorDateStr.split('-').map(Number)
      const monthStart = new Date(Date.UTC(year, month - 1 - monthsBack, 1))
      return formatUtcDate(monthStart)
    }

    switch (dateRangePreset) {
      case 'season':
        return { startDate: seasonStart, endDate: seasonEnd, lastNGames: null, excludePlayoffs: false }

      case 'season_regular':
        return { startDate: seasonStart, endDate: seasonEnd, lastNGames: null, excludePlayoffs: true }

      case 'this_month': {
        const monthStart = getMonthStart(seasonEnd, 0)
        if (monthStart < seasonStart) {
          return { startDate: seasonStart, endDate: seasonEnd, lastNGames: null, excludePlayoffs: false }
        }
        return { startDate: monthStart, endDate: seasonEnd, lastNGames: null, excludePlayoffs: false }
      }

      case 'last_2_months': {
        const start = getMonthStart(seasonEnd, 1)
        if (start < seasonStart) {
          return { startDate: seasonStart, endDate: seasonEnd, lastNGames: null, excludePlayoffs: false }
        }
        return { startDate: start, endDate: seasonEnd, lastNGames: null, excludePlayoffs: false }
      }

      case 'last_3_months': {
        const start = getMonthStart(seasonEnd, 2)
        if (start < seasonStart) {
          return { startDate: seasonStart, endDate: seasonEnd, lastNGames: null, excludePlayoffs: false }
        }
        return { startDate: start, endDate: seasonEnd, lastNGames: null, excludePlayoffs: false }
      }

      case 'last_10_games':
        return { startDate: null, endDate: null, lastNGames: 10, excludePlayoffs: false }

      case 'last_15_games':
        return { startDate: null, endDate: null, lastNGames: 15, excludePlayoffs: false }

      case 'last_20_games':
        return { startDate: null, endDate: null, lastNGames: 20, excludePlayoffs: false }

      case 'custom': {
        const safeStart = customStartDate || seasonStart
        const safeEnd = customEndDate || seasonEnd
        const clampedStart = safeStart < seasonStart ? seasonStart : safeStart
        const clampedEnd = safeEnd > seasonEnd ? seasonEnd : safeEnd
        if (clampedStart > clampedEnd) {
          return { startDate: seasonStart, endDate: seasonEnd, lastNGames: null, excludePlayoffs: false }
        }
        return { startDate: clampedStart, endDate: clampedEnd, lastNGames: null, excludePlayoffs: false }
      }

      default:
        return { startDate: seasonStart, endDate: seasonEnd, lastNGames: null, excludePlayoffs: false }
    }
  }, [dateRangePreset, seasonBounds, customStartDate, customEndDate])

  // First, load season bounds (without date filtering)
  useEffect(() => {
    let isCurrent = true
    async function loadSeasonBounds() {
      if (!selectedSeason) return
      setLoading(true)
      setError(null)
      try {
        // Load without date filters to get season bounds
        const res = await getLeagueSummary(selectedSeason, null, null, false, null, selectedDataScope)
        if (isCurrent && res.first_game_date && res.last_game_date) {
          setSeasonBounds({ first: res.first_game_date, last: res.last_game_date })
          setCustomStartDate(res.first_game_date)
          setCustomEndDate(res.last_game_date)
          setError(null)
        }
      } catch (err) {
        if (isCurrent) setError(err.message)
      } finally {
        if (isCurrent) setLoading(false)
      }
    }
    loadSeasonBounds()
    return () => { isCurrent = false }
  }, [selectedSeason, selectedDataScope])

  // Load data with calculated date range
  useEffect(() => {
    if (!selectedSeason) return
    if (!lastNGames && (!startDate || !endDate)) return
    let isCurrent = true
    async function loadData() {
      setLoading(true)
      setError(null)
      try {
        const res = await getLeagueSummary(
          selectedSeason,
          startDate,
          endDate,
          excludePlayoffs,
          lastNGames,
          selectedDataScope
        )
        if (isCurrent) {
          setData(res)
          setError(null)
        }
      } catch (err) {
        if (isCurrent) setError(err.message)
      } finally {
        if (isCurrent) setLoading(false)
      }
    }
    loadData()
    return () => { isCurrent = false }
  }, [selectedSeason, startDate, endDate, lastNGames, excludePlayoffs, selectedDataScope])

  // Load top contributors when date range changes
  useEffect(() => {
    if (!selectedSeason) return
    if (!lastNGames && (!startDate || !endDate)) return
    if (DERIVED_SCOPE_VALUES.has(selectedDataScope)) {
      setTopContributors(null)
      setTopContributorsNote('Top contributors are currently unavailable for Garbage Time and Non-Clutch Time views.')
      setContributorsLoading(false)
      return
    }
    let isCurrent = true
    async function loadTopContributors() {
      setContributorsLoading(true)
      setTopContributorsNote('')
      try {
        const res = await getLeagueTopContributors(
          selectedSeason,
          startDate,
          endDate,
          excludePlayoffs,
          lastNGames,
          selectedDataScope
        )
        if (isCurrent) {
          setTopContributors(res)
        }
      } catch (err) {
        // Silently fail for contributors - don't show error to user
        if (isCurrent) {
          setTopContributors(null)
          setTopContributorsNote('')
        }
      } finally {
        if (isCurrent) setContributorsLoading(false)
      }
    }
    loadTopContributors()
    return () => { isCurrent = false }
  }, [selectedSeason, startDate, endDate, lastNGames, excludePlayoffs, selectedDataScope])

  const isScopedSummary = selectedDataScope !== 'all'

  const activeColumns = useMemo(() => {
    if (tableView === VIEW_SOS_ADJUSTMENTS) {
      return isScopedSummary ? SOS_SCOPED_COLUMNS : SOS_COLUMNS
    }
    if (tableView === VIEW_SHOOTING) {
      return isScopedSummary ? SHOOTING_SCOPED_COLUMNS : SHOOTING_COLUMNS
    }
    return isScopedSummary ? FOUR_FACTOR_SCOPED_COLUMNS : FOUR_FACTOR_COLUMNS
  }, [tableView, isScopedSummary])

  useEffect(() => {
    if (!urlSelections.sortColumn) return
    if (!activeColumns.some((column) => column.key === urlSelections.sortColumn)) return
    if (urlSelections.sortColumn !== sortColumn) {
      setSortColumn(urlSelections.sortColumn)
    }
    const nextDirection = urlSelections.sortDirection || getDefaultSortDirectionForColumn(urlSelections.sortColumn)
    if (nextDirection !== sortDirection) {
      setSortDirection(nextDirection)
    }
  }, [activeColumns, sortColumn, sortDirection, urlSelections.sortColumn, urlSelections.sortDirection])

  useEffect(() => {
    const defaultSortByView = getDefaultSortForView(tableView)

    if (!activeColumns.some(col => col.key === sortColumn)) {
      setSortColumn(defaultSortByView.column)
      setSortDirection(defaultSortByView.direction)
    }
  }, [tableView, activeColumns, sortColumn])

  const sortedTeams = useMemo(() => {
    if (!data?.teams) return []
    if (!activeColumns.some(col => col.key === sortColumn)) return [...data.teams]
    const sorted = [...data.teams].sort((a, b) => {
      const aVal = a[sortColumn]
      const bVal = b[sortColumn]
      if (typeof aVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
    })
    return sorted
  }, [data, sortColumn, sortDirection, activeColumns])

  // Compute min/max for sorted column only
  const sortedColumnStats = useMemo(() => {
    if (!data?.teams || data.teams.length === 0) return null
    const col = activeColumns.find(c => c.key === sortColumn)
    if (!col || col.key === 'team' || col.higherBetter === null) return null
    const values = data.teams.map(t => t[sortColumn]).filter(v => typeof v === 'number')
    if (values.length === 0) return null
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      higherBetter: col.higherBetter,
      isOpp: col.isOpp,
    }
  }, [data, sortColumn, activeColumns])

  const getCellColor = (column, value) => {
    // Only color the sorted column
    if (column !== sortColumn || !sortedColumnStats) return null
    const { min, max, higherBetter, isOpp } = sortedColumnStats
    if (min === max) return null

    const normalized = (value - min) / (max - min)

    // For "higher is better" stats: high values = green, low values = red
    // For "lower is better" stats (like DRtg, or Opp stats): reverse - low values = green
    // isOpp stats: higher opponent values are BAD for your team, so reverse gradient
    const shouldReverse = higherBetter === false || isOpp

    if (shouldReverse) {
      // Low = green (good), High = red (bad)
      const green = `rgba(34, 197, 94, ${0.15 + (1 - normalized) * 0.35})`
      const red = `rgba(239, 68, 68, ${0.15 + normalized * 0.35})`
      return normalized < 0.5 ? green : red
    } else {
      // High = green (good), Low = red (bad)
      const green = `rgba(34, 197, 94, ${0.15 + normalized * 0.35})`
      const red = `rgba(239, 68, 68, ${0.15 + (1 - normalized) * 0.35})`
      return normalized > 0.5 ? green : red
    }
  }

  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(column)
      const col = activeColumns.find((c) => c.key === column)
      setSortDirection(col?.higherBetter === false ? 'asc' : 'desc')
    }
  }

  const isSectionStart = (columnKey) => {
    return SECTION_START_COLUMNS[tableView]?.has(columnKey) || false
  }

  const handleOpenTrendFromCell = (teamAbbr, columnKey) => {
    if (!teamAbbr || !selectedSeason) return
    const stat = resolveTrendStatForColumn(columnKey)
    const trendsScope = selectedDataScope === 'garbage_filtered' ? 'garbage_filtered' : 'all'
    navigate('/trends', {
      state: {
        season: selectedSeason,
        team: teamAbbr,
        stat,
        dataScope: trendsScope,
      },
    })
  }

  const handleCellKeyDown = (event, teamAbbr, columnKey) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      handleOpenTrendFromCell(teamAbbr, columnKey)
    }
  }

  const showHoverTooltip = (event, tooltipContent) => {
    if (!tooltipContent) return

    const rect = event.currentTarget.getBoundingClientRect()
    const showAbove = rect.top > 96
    const centerX = rect.left + (rect.width / 2)
    const x = Math.min(Math.max(centerX, 24), window.innerWidth - 24)
    const y = showAbove ? rect.top - 10 : rect.bottom + 10

    setHoverTooltip({
      ...tooltipContent,
      x,
      y,
      placement: showAbove ? 'top' : 'bottom',
    })
  }

  const hideHoverTooltip = () => {
    setHoverTooltip(null)
  }

  const isHeaderCenteredValueColumn = (columnKey) => {
    return ['games', 'wins', 'losses', 'win_pct', 'scope_games', 'scope_time_pct'].includes(columnKey)
  }

  const formatValue = (value, column) => {
    if (value === null || value === undefined) return '-'
    if (column === 'team') return value
    if (typeof value === 'number') {
      if (column === 'scope_time_pct') {
        return `${value.toFixed(1)}%`
      }
      if (column === 'win_pct') {
        const pct = value / 100
        return pct >= 1 ? '1.000' : pct.toFixed(3).replace(/^0/, '')
      }
      // For GP, W, L: show integer if whole number, otherwise 1 decimal
      if (['games', 'wins', 'losses', 'scope_games'].includes(column)) {
        return Number.isInteger(value) ? value : value.toFixed(1)
      }
      return value.toFixed(1)
    }
    return value
  }

  const formatReadableDate = (dateString) => {
    if (!dateString) return ''
    return new Date(`${dateString}T00:00:00`).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
  }

  const contributorPeriodHeadingLabel = useMemo(() => {
    if (!topContributors?.start_date || !topContributors?.end_date) return ''

    const startLabel = formatReadableDate(topContributors.start_date)
    const endLabel = formatReadableDate(topContributors.end_date)

    if (lastNGames) {
      return `Last ${lastNGames} games`
    }

    if (dateRangePreset === 'season_regular') {
      return 'Regular season'
    }

    if (dateRangePreset === 'season') {
      return 'Season to date'
    }

    if (startLabel === endLabel) return startLabel
    return `${startLabel} - ${endLabel}`
  }, [topContributors, lastNGames, dateRangePreset])

  const contributorPeriodSentenceLabel = useMemo(() => {
    if (!topContributors?.start_date || !topContributors?.end_date) return 'this period'

    const startLabel = formatReadableDate(topContributors.start_date)
    const endLabel = formatReadableDate(topContributors.end_date)

    if (lastNGames) {
      return `the last ${lastNGames} games`
    }

    if (dateRangePreset === 'season_regular') {
      return 'the regular season to date'
    }

    if (dateRangePreset === 'season') {
      return 'the season to date'
    }

    if (startLabel === endLabel) {
      return `games on ${startLabel}`
    }

    return `games from ${startLabel} to ${endLabel}`
  }, [topContributors, lastNGames, dateRangePreset])

  const getContributorTooltip = (item) => {
    if (!item) return null

    const teamLabel = TEAM_CITY_BY_ABBR[item.team] || TEAM_NAME_BY_ABBR[item.team] || item.team
    const contributionValue = Math.abs(item.contribution).toFixed(2)
    const verb = item.contribution >= 0 ? 'added' : 'subtracted'
    const preposition = item.contribution >= 0 ? 'to' : 'from'
    const sentence = `${teamLabel}'s ${item.factor_label} ${verb} ${contributionValue} points ${preposition} their net rating in ${contributorPeriodSentenceLabel}.`

    return {
      lines: [sentence],
    }
  }

  // Compute league averages for the new columns
  const leagueAverages = useMemo(() => {
    if (!data?.teams || data.teams.length === 0) return null
    const avg = {}
    activeColumns.forEach(col => {
      if (col.key === 'team') {
        avg[col.key] = 'Average'
      } else {
        const values = data.teams.map(t => t[col.key]).filter(v => typeof v === 'number')
        if (values.length > 0) {
          avg[col.key] = values.reduce((a, b) => a + b, 0) / values.length
        }
      }
    })
    return avg
  }, [data, activeColumns])

  // Export to XLSX
  const handleExport = () => {
    if (!sortedTeams.length) return

    // Build CSV content (XLSX-compatible)
    const headers = ['Rank', ...activeColumns.map(col => col.labelLine2 ? `${col.labelLine2} ${col.label}` : col.label)]
    const rows = sortedTeams.map((team, index) => {
      return [
        index + 1,
        ...activeColumns.map(col => formatValue(team[col.key], col.key))
      ]
    })

    // Add league averages row
    if (leagueAverages) {
      rows.push([
        '',
        ...activeColumns.map(col => col.key === 'team' ? 'Average' : formatValue(leagueAverages[col.key], col.key))
      ])
    }

    // Create CSV string
    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.map(cell => {
        // Escape cells containing commas
        const cellStr = String(cell)
        return cellStr.includes(',') ? `"${cellStr}"` : cellStr
      }).join(','))
    ].join('\n')

    // Create and download file
    const blob = new Blob([csvContent], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const viewSuffix =
      tableView === VIEW_SOS_ADJUSTMENTS
        ? 'sos_adjusted'
        : tableView === VIEW_SHOOTING
          ? 'shooting'
          : 'four_factors'
    const rangeSuffix = lastNGames ? `last_${lastNGames}_games` : `${startDate}_${endDate}`
    a.download = `league_summary_${viewSuffix}_${selectedSeason}_${rangeSuffix}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const efficiencyChartData = useMemo(() => {
    if (!sortedTeams.length) return []
    return sortedTeams.map((team) => ({
      team: team.team,
      team_name: TEAM_NAME_BY_ABBR[team.team] || team.team,
      team_city: TEAM_CITY_BY_ABBR[team.team] || TEAM_NAME_BY_ABBR[team.team] || team.team,
      off_rating: team.off_rating,
      def_rating: team.def_rating,
      net_rating: team.net_rating,
    }))
  }, [sortedTeams])

  const efficiencyDomains = useMemo(() => {
    if (!efficiencyChartData.length) {
      return {
        x: [100, 125],
        y: [120, 100],
      }
    }

    const offValues = efficiencyChartData.map((d) => d.off_rating)
    const defValues = efficiencyChartData.map((d) => d.def_rating)
    const offAvgFallback = offValues.reduce((sum, v) => sum + v, 0) / offValues.length
    const defAvgFallback = defValues.reduce((sum, v) => sum + v, 0) / defValues.length

    const avgOff = typeof data?.league_averages?.off_rating === 'number'
      ? data.league_averages.off_rating
      : offAvgFallback
    const avgDef = typeof data?.league_averages?.def_rating === 'number'
      ? data.league_averages.def_rating
      : defAvgFallback

    const xMinRaw = Math.min(...offValues)
    const xMaxRaw = Math.max(...offValues)
    const yMinRaw = Math.min(...defValues)
    const yMaxRaw = Math.max(...defValues)

    const xRadius = Math.max(Math.abs(avgOff - xMinRaw), Math.abs(xMaxRaw - avgOff))
    const yRadius = Math.max(Math.abs(avgDef - yMinRaw), Math.abs(yMaxRaw - avgDef))
    const xPad = Math.max(0.8, xRadius * 0.12)
    const yPad = Math.max(0.8, yRadius * 0.12)

    return {
      x: [avgOff - xRadius - xPad, avgOff + xRadius + xPad],
      y: [avgDef - yRadius - yPad, avgDef + yRadius + yPad],
    }
  }, [efficiencyChartData, data])

  const efficiencyLeagueAverages = useMemo(() => {
    const fallbackOff = efficiencyChartData.length
      ? efficiencyChartData.reduce((sum, d) => sum + d.off_rating, 0) / efficiencyChartData.length
      : 112.5
    const fallbackDef = efficiencyChartData.length
      ? efficiencyChartData.reduce((sum, d) => sum + d.def_rating, 0) / efficiencyChartData.length
      : 112.5

    return {
      off: typeof data?.league_averages?.off_rating === 'number' ? data.league_averages.off_rating : fallbackOff,
      def: typeof data?.league_averages?.def_rating === 'number' ? data.league_averages.def_rating : fallbackDef,
    }
  }, [data, efficiencyChartData])

  const efficiencySubtitle = useMemo(() => {
    const nowLabel = new Date().toLocaleDateString('en-US', {
      month: 'long',
      day: 'numeric',
      year: 'numeric',
    })

    let rangeLabel = 'Season-to-Date'
    if (lastNGames) {
      rangeLabel = `Last ${lastNGames} games`
    } else if (startDate && seasonBounds.first && startDate !== seasonBounds.first) {
      const startLabel = new Date(`${startDate}T00:00:00`).toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      })
      rangeLabel = `Since ${startLabel}`
    }

    return `${nowLabel} | ${rangeLabel}`
  }, [lastNGames, startDate, seasonBounds.first])

  const quadrantTeams = useMemo(() => {
    const offCut = efficiencyLeagueAverages.off
    const defCut = efficiencyLeagueAverages.def
    const byNetDesc = (a, b) => b.net_rating - a.net_rating

    return {
      badOffGoodDef: efficiencyChartData
        .filter((team) => team.off_rating < offCut && team.def_rating <= defCut)
        .sort(byNetDesc),
      goodOffGoodDef: efficiencyChartData
        .filter((team) => team.off_rating >= offCut && team.def_rating <= defCut)
        .sort(byNetDesc),
      badOffBadDef: efficiencyChartData
        .filter((team) => team.off_rating < offCut && team.def_rating > defCut)
        .sort(byNetDesc),
      goodOffBadDef: efficiencyChartData
        .filter((team) => team.off_rating >= offCut && team.def_rating > defCut)
        .sort(byNetDesc),
    }
  }, [efficiencyChartData, efficiencyLeagueAverages])

  const formatSignedNetRating = (value) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}`

  const renderQuadrantNote = (title, teams, keyPrefix) => (
    <>
      <h3 className="efficiency-note-title">{title}</h3>
      <div className="efficiency-note-columns">
        <span>Team</span>
        <span>NRtg</span>
      </div>
      {teams.length > 0 ? (
        <ul className="efficiency-note-list">
          {teams.map((team) => (
            <li key={`${keyPrefix}-${team.team}`} className="efficiency-note-list-item">
              <span className="efficiency-note-team">{team.team_city}</span>
              <span className="efficiency-note-net">{formatSignedNetRating(team.net_rating)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <div className="efficiency-note-empty">No teams</div>
      )}
    </>
  )

  return (
    <div className="league-summary container">
      <div className="page-header">
        <div className="page-title-row">
          <h1 className="page-title">League Summary</h1>
          <p className="page-description">
            Compare team statistics across the league. Click column headers to sort.
          </p>
        </div>
      </div>

      <div className="controls card">
        <div className="form-row controls-top-row">
          <div className="form-group season-control">
            <label className="form-label">Season</label>
            <select
              className="form-select"
              value={selectedSeason}
              onChange={(e) => setSelectedSeason(e.target.value)}
            >
              <option value="">Select season...</option>
              {seasons.map((season) => (
                <option key={season} value={season}>{season}</option>
              ))}
            </select>
          </div>

          <div className="form-group data-scope-control">
            <label className="form-label">Data Scope</label>
            <select
              className="form-select"
              value={selectedDataScope}
              onChange={(e) => setSelectedDataScope(e.target.value)}
            >
              {DATA_SCOPE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group date-range-control">
            <label className="form-label">Date Range</label>
            <select
              className="form-select"
              value={dateRangePreset}
              onChange={(e) => setDateRangePreset(e.target.value)}
            >
              {DATE_RANGE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>

        {isCustomDateRangeVisible && (
          <div className="form-row custom-dates-row">
            <div className="form-group custom-date-control">
              <label className="form-label">Start Date</label>
              <input
                type="date"
                className="form-input"
                value={customStartDate}
                min={seasonBounds.first || undefined}
                max={seasonBounds.last || undefined}
                onChange={(e) => setCustomStartDate(e.target.value)}
              />
            </div>

            <div className="form-group custom-date-control">
              <label className="form-label">End Date</label>
              <input
                type="date"
                className="form-input"
                value={customEndDate}
                min={seasonBounds.first || undefined}
                max={seasonBounds.last || undefined}
                onChange={(e) => setCustomEndDate(e.target.value)}
              />
            </div>
          </div>
        )}

        <div className="form-row table-view-row">
          <div className="form-group table-view-control inline-toggle-control">
            <label className="form-label">Table View:</label>
            <div className="view-toggle-group">
              <button
                type="button"
                className={`view-toggle-btn ${tableView === VIEW_FOUR_FACTORS ? 'active' : ''}`}
                onClick={() => setTableView(VIEW_FOUR_FACTORS)}
              >
                Four Factors
              </button>
              <button
                type="button"
                className={`view-toggle-btn ${tableView === VIEW_SHOOTING ? 'active' : ''}`}
                onClick={() => setTableView(VIEW_SHOOTING)}
              >
                Shooting
              </button>
              <button
                type="button"
                className={`view-toggle-btn ${tableView === VIEW_SOS_ADJUSTMENTS ? 'active' : ''}`}
                onClick={() => setTableView(VIEW_SOS_ADJUSTMENTS)}
              >
                {isCustomDateRangeVisible ? 'SOS' : 'Strength of Schedule'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading league data...
        </div>
      )}

      {data && !loading && (
        <div className="card">
          <div className="table-container">
            <table className="summary-table">
              <thead>
                <tr>
                  <th
                    className="rank-col"
                    onMouseEnter={(event) => showHoverTooltip(event, { lines: [getHeaderTooltip('rank')] })}
                    onMouseLeave={hideHoverTooltip}
                  >
                    <div className="header-content">
                      <span className="header-line1">Rank</span>
                    </div>
                  </th>
                  {activeColumns.map((col) => (
                    <th
                      key={col.key}
                      className={`stat-header ${col.sortable ? 'sortable' : ''} ${sortColumn === col.key ? 'sorted' : ''} ${isSectionStart(col.key) ? 'section-divider' : ''} ${col.key === 'scope_time_pct' ? 'time-pct-header' : ''}`}
                      onClick={() => col.sortable && handleSort(col.key)}
                      onMouseEnter={(event) => showHoverTooltip(event, { lines: [getHeaderTooltip(col.key, col.labelLine2 === 'Opp')] })}
                      onMouseLeave={hideHoverTooltip}
                    >
                      <div className="header-content">
                        {col.labelLine2 && <span className="header-line2">{col.labelLine2}</span>}
                        <span className="header-line1">
                          {col.label}
                          {sortColumn === col.key && (
                            <span className="sort-indicator">
                              {sortDirection === 'asc' ? ' ↑' : ' ↓'}
                            </span>
                          )}
                        </span>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedTeams.map((team, index) => (
                  <tr key={team.team}>
                    <td
                      className="rank-col trend-link-cell"
                      role="button"
                      tabIndex={0}
                      title={`Click to open Statistical Trends: ${team.team} Net Rating`}
                      onClick={() => handleOpenTrendFromCell(team.team, 'net_rating')}
                      onKeyDown={(event) => handleCellKeyDown(event, team.team, 'net_rating')}
                    >
                      <span className="rank-value">{index + 1}</span>
                    </td>
                    {activeColumns.map((col) => {
                      const trendStat = resolveTrendStatForColumn(col.key)
                      const trendLabel = TRENDS_STAT_LABELS[trendStat] || 'Net Rating'
                      const fromLabel = col.labelLine2 ? `${col.labelLine2} ${col.label}` : col.label
                      const title = TRENDS_DIRECT_STAT_COLUMNS.has(col.key)
                        ? `Click to open Statistical Trends: ${team.team} ${fromLabel}`
                        : `Click to open Statistical Trends: ${team.team} ${trendLabel} (from ${fromLabel})`

                      return (
                        <td
                          key={col.key}
                          style={col.key !== 'team' ? { backgroundColor: getCellColor(col.key, team[col.key]) } : {}}
                          className={`${col.key === 'team' ? 'team-cell' : 'stat-cell'} trend-link-cell ${isSectionStart(col.key) ? 'section-divider' : ''} ${col.key === 'scope_time_pct' ? 'time-pct-col' : ''}`}
                          role="button"
                          tabIndex={0}
                          title={title}
                          onClick={() => handleOpenTrendFromCell(team.team, col.key)}
                          onKeyDown={(event) => handleCellKeyDown(event, team.team, col.key)}
                        >
                          {col.key === 'team' ? (
                            formatValue(team[col.key], col.key)
                          ) : (
                            <span className={`ls-stat-value ${isHeaderCenteredValueColumn(col.key) ? 'ls-stat-value--center' : ''}`}>
                              {formatValue(team[col.key], col.key)}
                            </span>
                          )}
                        </td>
                      )
                    })}
                  </tr>
                ))}
                {/* League Averages Row */}
                {leagueAverages && (
                  <tr className="league-avg-row">
                    <td className="rank-col"></td>
                    {activeColumns.map((col) => (
                      <td
                        key={col.key}
                        className={`${col.key === 'team' ? 'team-cell' : 'stat-cell'} ${isSectionStart(col.key) ? 'section-divider' : ''} ${col.key === 'scope_time_pct' ? 'time-pct-col' : ''}`}
                      >
                        {col.key === 'team' ? (
                          formatValue(leagueAverages[col.key], col.key)
                        ) : (
                          <span className={`ls-stat-value ${isHeaderCenteredValueColumn(col.key) ? 'ls-stat-value--center' : ''}`}>
                            {formatValue(leagueAverages[col.key], col.key)}
                          </span>
                        )}
                      </td>
                    ))}
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="export-section">
            <button className="btn btn-primary" onClick={handleExport}>
              Export to Excel
            </button>
          </div>

          {!!efficiencyChartData.length && (
            <div className="efficiency-landscape card">
              <h2 className="efficiency-title">The Efficiency Landscape (@KirkGoldsberry)</h2>
              <div className="efficiency-subtitle">{efficiencySubtitle}</div>
              <div className="efficiency-chart-shell">
                <div className="efficiency-chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={EFFICIENCY_CHART_MARGIN}>
                      <XAxis
                        type="number"
                        dataKey="off_rating"
                        domain={efficiencyDomains.x}
                        ticks={[]}
                        tick={false}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        type="number"
                        dataKey="def_rating"
                        domain={efficiencyDomains.y}
                        reversed
                        ticks={[]}
                        tick={false}
                        tickLine={false}
                        axisLine={false}
                      />
                      <ReferenceLine
                        x={efficiencyLeagueAverages.off}
                        stroke="var(--color-text-muted)"
                        strokeWidth={1}
                        label={renderXAxisTitle}
                      />
                      <ReferenceLine
                        y={efficiencyLeagueAverages.def}
                        stroke="var(--color-text-muted)"
                        strokeWidth={1}
                        label={renderYAxisTitle}
                      />
                      <Tooltip
                        cursor={false}
                        content={<EfficiencyTooltip />}
                        wrapperStyle={{ outline: 'none' }}
                      />
                      <Scatter data={efficiencyChartData} fill="var(--color-primary)">
                        <LabelList dataKey="team" position="top" offset={8} fontSize={11} fill="var(--color-text)" />
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                  <div className="efficiency-quadrant-notes">
                    <div className="efficiency-note efficiency-note--left-top efficiency-note--data">
                      {renderQuadrantNote('Bad Offense & Good Defense', quadrantTeams.badOffGoodDef, 'bad-off-good-def')}
                    </div>
                    <div className="efficiency-note efficiency-note--right-top efficiency-note--data">
                      {renderQuadrantNote('Good Offense & Good Defense', quadrantTeams.goodOffGoodDef, 'good-off-good-def')}
                    </div>
                    <div className="efficiency-note efficiency-note--left-bottom efficiency-note--data">
                      {renderQuadrantNote('Bad Offense & Bad Defense', quadrantTeams.badOffBadDef, 'bad-off-bad-def')}
                    </div>
                    <div className="efficiency-note efficiency-note--right-bottom efficiency-note--data">
                      {renderQuadrantNote('Good Offense & Bad Defense', quadrantTeams.goodOffBadDef, 'good-off-bad-def')}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {topContributors && !contributorsLoading && (
            <div className="top-contributors-section">
              <h2 className="section-title">
                Top Contributors to Net Rating
                {contributorPeriodHeadingLabel ? <span className="section-title-context"> ({contributorPeriodHeadingLabel})</span> : null}
              </h2>
              <div className="contributors-grid">
                <div className="contributors-column">
                  <h3 className="column-title positive">Top Positive Contributors</h3>
                  <table className="contributors-table">
                    <thead>
                      <tr>
                        <th>Team</th>
                        <th>Factor</th>
                        <th className="value-header">Value</th>
                        <th className="contribution-header">Contribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topContributors.top_positive.map((item, index) => (
                        <tr
                          key={`pos-${index}`}
                          className="contributors-data-row"
                          onMouseEnter={(event) => showHoverTooltip(event, getContributorTooltip(item))}
                          onMouseLeave={hideHoverTooltip}
                        >
                          <td className="team-cell">{item.team}</td>
                          <td>{item.factor_label}</td>
                          <td className="value-cell">{item.value.toFixed(1)}</td>
                          <td className="contribution-cell positive">
                            +{item.contribution.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="contributors-column">
                  <h3 className="column-title negative">Top Negative Contributors</h3>
                  <table className="contributors-table">
                    <thead>
                      <tr>
                        <th>Team</th>
                        <th>Factor</th>
                        <th className="value-header">Value</th>
                        <th className="contribution-header">Contribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topContributors.top_negative.map((item, index) => (
                        <tr
                          key={`neg-${index}`}
                          className="contributors-data-row"
                          onMouseEnter={(event) => showHoverTooltip(event, getContributorTooltip(item))}
                          onMouseLeave={hideHoverTooltip}
                        >
                          <td className="team-cell">{item.team}</td>
                          <td>{item.factor_label}</td>
                          <td className="value-cell">{item.value.toFixed(1)}</td>
                          <td className="contribution-cell negative">
                            {item.contribution.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {topContributorsNote && !contributorsLoading && (
            <div className="contributors-loading">
              {topContributorsNote}
            </div>
          )}

          {contributorsLoading && (
            <div className="contributors-loading">
              <div className="loading-spinner"></div>
              Loading top contributors...
            </div>
          )}
        </div>
      )}

      {hoverTooltip && (
        <div
          className={`header-tooltip-popover header-tooltip-popover--${hoverTooltip.placement}`}
          style={{ left: `${hoverTooltip.x}px`, top: `${hoverTooltip.y}px` }}
          role="tooltip"
          aria-hidden="true"
        >
          {hoverTooltip.title ? <div className="header-tooltip-title">{hoverTooltip.title}</div> : null}
          {(hoverTooltip.lines || []).map((line, index) => (
            <div key={`${line}-${index}`} className="header-tooltip-line">{line}</div>
          ))}
        </div>
      )}

      <div className="glossary-section card">
          <button
            className="glossary-toggle"
            onClick={() => setGlossaryExpanded(!glossaryExpanded)}
            aria-expanded={glossaryExpanded}
          >
            <span>Glossary</span>
            <span className="toggle-icon">{glossaryExpanded ? '−' : '+'}</span>
          </button>

          {glossaryExpanded && (
            <div className="glossary-content">
              <dl className="glossary-list">
                {GLOSSARY_ITEMS.map(item => (
                  <div key={item.term} className="glossary-item">
                    <dt>{item.term}</dt>
                    <dd>{item.definition}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </div>
    </div>
  )
}

export default LeagueSummary
