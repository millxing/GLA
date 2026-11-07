import { useState, useEffect, useMemo, useRef } from 'react'
import { getSeasons, getLeagueSummary, getLeagueTopContributors } from '../api'
import { usePersistedState } from '../hooks/usePersistedState'
import { ScatterChart, Scatter, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts'
import { useNavigate, useSearchParams } from 'react-router-dom'
import SegmentedControl from '../components/SegmentedControl'
import useMediaQuery from '../hooks/useMediaQuery'
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

const RECORD_COLUMN_KEYS = new Set(['games', 'wins', 'losses', 'win_pct'])
const SAMPLE_COLUMN_KEYS = new Set(['scope_games', 'scope_time_pct'])
const RATING_COLUMN_KEYS = new Set(['off_rating', 'def_rating', 'net_rating'])
const OFFENSE_COLUMN_KEYS = new Set(['efg_pct', 'ball_handling', 'oreb_pct', 'ft_rate', 'ft_pct', 'fg2_pct', 'fg3_pct', 'fg3a_rate'])
const SCHEDULE_COLUMN_KEYS = new Set(['sos', 'off_sos', 'def_sos'])

// Column groups drive the grouped header row and the group divider borders.
function getColumnGroup(column) {
  const { key } = column
  if (key === 'team') return { id: 'team', label: '' }
  if (RECORD_COLUMN_KEYS.has(key)) return { id: 'record', label: 'Record' }
  if (SAMPLE_COLUMN_KEYS.has(key)) return { id: 'sample', label: '' }
  if (RATING_COLUMN_KEYS.has(key)) return { id: 'ratings', label: 'Ratings' }
  if (column.isOpp) return { id: 'defense', label: 'Defense (Opp)' }
  if (OFFENSE_COLUMN_KEYS.has(key)) return { id: 'offense', label: 'Offense' }
  if (SCHEDULE_COLUMN_KEYS.has(key)) return { id: 'schedule', label: 'Schedule' }
  if (key.startsWith('adj_')) return { id: 'adjusted', label: 'Adjusted' }
  return { id: key, label: '' }
}

// On narrow screens the Ratings group moves next to Team so NRtg is visible without scrolling.
function orderColumnsForDisplay(columns, compact) {
  if (!compact) return columns
  const ratings = columns.filter((column) => RATING_COLUMN_KEYS.has(column.key))
  const rest = columns.filter((column) => column.key !== 'team' && !RATING_COLUMN_KEYS.has(column.key))
  const team = columns.filter((column) => column.key === 'team')
  return [...team, ...ratings, ...rest]
}

function buildColumnGroups(columns) {
  const groups = []
  columns.forEach((column) => {
    const group = getColumnGroup(column)
    const last = groups[groups.length - 1]
    if (last && last.id === group.id) {
      last.span += 1
    } else {
      groups.push({ ...group, span: 1, startKey: column.key })
    }
  })
  return groups
}

// Color scale direction per column: 1 = higher is better, -1 = lower is better, 0 = no tint.
// Volume/context columns (GP, sample size, pace, 3PA rate, schedule strength) are descriptive,
// not good/bad, so they stay neutral. Opponent stats are lower-is-better (Opp BH high = opponents
// protected the ball).
const NEUTRAL_TINT_COLUMN_KEYS = new Set([
  'games', 'scope_games', 'scope_time_pct', 'pace', 'fg3a_rate', 'opp_fg3a_rate', 'sos', 'off_sos', 'def_sos',
])

function getTintDirection(column) {
  if (column.key === 'team' || NEUTRAL_TINT_COLUMN_KEYS.has(column.key)) return 0
  if (column.higherBetter === true) return 1
  if (column.higherBetter === false) return -1
  return 0
}

const TINT_Z_FLOOR = 0.5
const TINT_Z_CAP = 2

function parseIsoDate(value) {
  return new Date(`${value}T00:00:00`)
}

const formatMonthDay = (date) => date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
const formatFullDate = (date) => date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })

// "Nov 1–2, 2026", "Oct 20 – Nov 6, 2026", "Dec 20, 2025 – Jan 6, 2026"
function formatDateRange(start, end) {
  if (!start || !end) return ''
  const startDate = parseIsoDate(start)
  const endDate = parseIsoDate(end)
  if (start === end) return formatFullDate(endDate)
  const sameYear = startDate.getFullYear() === endDate.getFullYear()
  if (sameYear && startDate.getMonth() === endDate.getMonth()) {
    return `${formatMonthDay(startDate)}–${endDate.getDate()}, ${endDate.getFullYear()}`
  }
  if (sameYear) return `${formatMonthDay(startDate)} – ${formatFullDate(endDate)}`
  return `${formatFullDate(startDate)} – ${formatFullDate(endDate)}`
}

const TABLE_VIEW_TITLES = {
  [VIEW_FOUR_FACTORS]: 'Four Factors',
  [VIEW_SHOOTING]: 'Shooting',
  [VIEW_SOS_ADJUSTMENTS]: 'Strength of Schedule',
}

const TABLE_COLUMN_WIDTHS = {
  regular: { rank: 44, team: 64, stat: 64, timePct: 84 },
  compact: { rank: 36, team: 54, stat: 58, timePct: 76 },
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

const TABLE_VIEWPORT_BOTTOM_GUTTER = 24
const MIN_TABLE_VIEWPORT_HEIGHT = 320

const GLOSSARY_ITEMS = [
  { term: 'Rank', definition: 'Rank by the sorted column (by NRtg when sorted by Team).' },
  { term: 'Team', definition: 'Team abbreviation.' },
  { term: 'GP', definition: 'Games Played' },
  { term: 'G', definition: 'Games containing the selected data-scope segment.' },
  { term: '% of Time', definition: 'Share of total game time spent in the selected data-scope segment.' },
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
  { value: 'season_regular', label: 'Regular season only' },
  { value: 'this_month', label: 'This month' },
  { value: 'last_2_months', label: 'Last 2 months' },
  { value: 'last_3_months', label: 'Last 3 months' },
  { value: 'last_10_games', label: 'Last 10 games' },
  { value: 'last_15_games', label: 'Last 15 games' },
  { value: 'last_20_games', label: 'Last 20 games' },
  { value: 'custom', label: 'Custom range…' },
]
const DATA_SCOPE_OPTIONS = [
  { value: 'all', label: 'All data' },
  { value: 'q1', label: 'Q1' },
  { value: 'q2', label: 'Q2' },
  { value: 'q3', label: 'Q3' },
  { value: 'q4', label: 'Q4' },
  { value: 'ot', label: 'OT' },
  { value: 'h1', label: 'H1' },
  { value: 'h2', label: 'H2 (incl. OT)' },
  { value: 'garbage_filtered', label: 'Excl. garbage time' },
  { value: 'garbage_time', label: 'Garbage time only' },
  { value: 'clutch', label: 'Clutch time' },
  { value: 'non_clutch_time', label: 'Non-clutch time' },
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
  q1: 'q1',
  q2: 'q2',
  q3: 'q3',
  q4: 'q4',
  ot: 'ot',
  overtime: 'ot',
  h1: 'h1',
  first_half: 'h1',
  h2: 'h2',
  second_half: 'h2',
  h2_incl_ot: 'h2',
  h2_including_ot: 'h2',
  second_half_incl_ot: 'h2',
  second_half_including_ot: 'h2',
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

const EFFICIENCY_CHART_LAYOUT = {
  regular: { margin: { top: 12, right: 16, bottom: 4, left: 4 }, yAxisWidth: 56, xAxisHeight: 44 },
  compact: { margin: { top: 8, right: 8, bottom: 6, left: 0 }, yAxisWidth: 46, xAxisHeight: 40 },
}
const POINT_RADIUS = 4.5
const LABEL_FONT_SIZE = 11
const LABEL_CHAR_WIDTH = 7.4
const LABEL_HEIGHT = 11
// Candidate label offsets relative to the point centre (text baseline), tried in order.
const LABEL_CANDIDATES = [
  { dx: 0, dy: -8, anchor: 'middle' },
  { dx: 0, dy: 16, anchor: 'middle' },
  { dx: 7, dy: 4, anchor: 'start' },
  { dx: -7, dy: 4, anchor: 'end' },
  { dx: 6, dy: -6, anchor: 'start' },
  { dx: -6, dy: -6, anchor: 'end' },
  { dx: 6, dy: 14, anchor: 'start' },
  { dx: -6, dy: 14, anchor: 'end' },
  { dx: 0, dy: -16, anchor: 'middle' },
  { dx: 0, dy: 24, anchor: 'middle' },
  { dx: 12, dy: 4, anchor: 'start' },
  { dx: -12, dy: 4, anchor: 'end' },
]
// Breathing room around label boxes so near-touching labels count as collisions.
const LABEL_PADDING = 2
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

function getNiceTicks([min, max], targetCount) {
  const span = max - min
  if (!(span > 0)) return []
  const rough = span / targetCount
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((candidate) => candidate >= rough) || 10 * magnitude
  const ticks = []
  for (let value = Math.ceil(min / step) * step; value <= max + 1e-9; value += step) {
    ticks.push(Number(value.toFixed(2)))
  }
  return ticks
}

const formatTick = (value) => (Number.isInteger(value) ? String(value) : value.toFixed(1))

function getBoxOverlap(a, b) {
  const width = Math.min(a.right, b.right) - Math.max(a.left, b.left)
  const height = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top)
  return width > 0 && height > 0 ? width * height : 0
}

function getLabelBox(x, y, text, anchor) {
  const width = text.length * LABEL_CHAR_WIDTH
  const left = anchor === 'middle' ? x - width / 2 : anchor === 'end' ? x - width : x
  return {
    left: left - LABEL_PADDING,
    right: left + width + LABEL_PADDING,
    top: y - LABEL_HEIGHT + 2 - LABEL_PADDING,
    bottom: y + 2 + LABEL_PADDING,
  }
}

/**
 * Greedy label placement: each label takes the first candidate offset that does not
 * collide with points, already-placed labels, or the reference-line captions.
 * Returns Map(team -> { dx, dy, anchor }); teams missing from the map are unlabeled.
 */
function placePointLabels({ points, labeledTeams, size, layout, domains, averages }) {
  const placements = new Map()
  if (!points.length) return placements
  const { margin, yAxisWidth, xAxisHeight } = layout
  const plotLeft = margin.left + yAxisWidth
  const plotTop = margin.top
  const plotWidth = size.width - plotLeft - margin.right
  const plotHeight = size.height - margin.top - margin.bottom - xAxisHeight

  if (!(plotWidth > 0 && plotHeight > 0)) {
    points.forEach((point) => {
      if (labeledTeams.has(point.team)) placements.set(point.team, LABEL_CANDIDATES[0])
    })
    return placements
  }

  const [xMin, xMax] = domains.x
  const [yMin, yMax] = domains.y
  const toPixel = (point) => ({
    x: plotLeft + ((point.off_rating - xMin) / (xMax - xMin)) * plotWidth,
    // Y axis is reversed: the lowest (best) defensive rating sits at the top.
    y: plotTop + ((point.def_rating - yMin) / (yMax - yMin)) * plotHeight,
  })
  const pixels = new Map(points.map((point) => [point.team, toPixel(point)]))
  const avgX = plotLeft + ((averages.off - xMin) / (xMax - xMin)) * plotWidth
  const avgY = plotTop + ((averages.def - yMin) / (yMax - yMin)) * plotHeight

  const obstacles = [
    // Reference-line captions (see renderAverageLabel)
    { left: avgX + 2, right: avgX + 86, top: plotTop, bottom: plotTop + 16 },
    { left: plotLeft + plotWidth - 86, right: plotLeft + plotWidth, top: avgY - 16, bottom: avgY },
    ...points.map((point) => {
      const { x, y } = pixels.get(point.team)
      return { team: point.team, left: x - POINT_RADIUS, right: x + POINT_RADIUS, top: y - POINT_RADIUS, bottom: y + POINT_RADIUS }
    }),
  ]
  const bounds = { left: plotLeft - 4, right: plotLeft + plotWidth + 4, top: plotTop - 4, bottom: plotTop + plotHeight + 4 }

  const order = points
    .filter((point) => labeledTeams.has(point.team))
    .sort((a, b) => {
      const distance = (point) => {
        const { x, y } = pixels.get(point.team)
        return Math.hypot((x - avgX) / plotWidth, (y - avgY) / plotHeight)
      }
      return distance(b) - distance(a)
    })

  order.forEach((point) => {
    const { x, y } = pixels.get(point.team)
    let best = null
    LABEL_CANDIDATES.forEach((candidate, index) => {
      const box = getLabelBox(x + candidate.dx, y + candidate.dy, point.team, candidate.anchor)
      // A label may touch its own point (the padding would otherwise reject the "above" slot).
      let score = obstacles.reduce(
        (sum, obstacle) => (obstacle.team === point.team ? sum : sum + getBoxOverlap(box, obstacle)),
        0
      )
      if (box.left < bounds.left || box.right > bounds.right || box.top < bounds.top || box.bottom > bounds.bottom) {
        score += 1000
      }
      score += index * 0.01
      if (!best || score < best.score) best = { score, candidate, box }
    })
    placements.set(point.team, best.candidate)
    obstacles.push(best.box)
  })

  return placements
}

// Callback-ref based so it attaches whenever the (conditionally rendered) node mounts.
function useElementSize() {
  const [node, setNode] = useState(null)
  const [size, setSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    if (!node || typeof ResizeObserver === 'undefined') return undefined
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize((prev) => (prev.width === width && prev.height === height ? prev : { width, height }))
    })
    observer.observe(node)
    return () => observer.disconnect()
  }, [node])

  return [setNode, size]
}

function renderAverageLabel(text, orientation) {
  return function AverageLabel({ viewBox }) {
    if (!viewBox) return null
    const isVertical = orientation === 'vertical'
    return (
      <text
        x={isVertical ? viewBox.x + 5 : viewBox.x + viewBox.width - 4}
        y={isVertical ? viewBox.y + 11 : viewBox.y - 5}
        textAnchor={isVertical ? 'start' : 'end'}
        fill="var(--color-text-muted)"
        fontSize="11"
      >
        {text}
      </text>
    )
  }
}

function DownloadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M8 2v8m0 0L4.75 6.75M8 10l3.25-3.25M2.75 11.5v1.25c0 .41.34.75.75.75h9c.41 0 .75-.34.75-.75V11.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
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
  const [searchParams, setSearchParams] = useSearchParams()
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
  const [colorScaleOn, setColorScaleOn] = usePersistedState('leaguesummary_colorscale', false)
  const [topContributors, setTopContributors] = useState(null)
  const [contributorsLoading, setContributorsLoading] = useState(false)
  const [topContributorsNote, setTopContributorsNote] = useState('')
  const [hoverTooltip, setHoverTooltip] = useState(null)
  const [tableMaxHeight, setTableMaxHeight] = useState(null)
  const [loadedDataKey, setLoadedDataKey] = useState('')
  const isCompact = useMediaQuery('(max-width: 768px)')
  const isNarrow = useMediaQuery('(max-width: 600px)')
  const [chartRef, chartSize] = useElementSize()
  const hasUserInteractedRef = useRef(false)
  const tableContainerRef = useRef(null)
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
    if (hasUserInteractedRef.current) return
    if (urlSelections.season && seasons.includes(urlSelections.season) && selectedSeason !== urlSelections.season) {
      setSelectedSeason(urlSelections.season)
    }
  }, [seasons, selectedSeason, setSelectedSeason, urlSelections.season])

  useEffect(() => {
    if (hasUserInteractedRef.current) return
    if (urlSelections.dataScope && urlSelections.dataScope !== selectedDataScope) {
      setSelectedDataScope(urlSelections.dataScope)
    }
  }, [selectedDataScope, setSelectedDataScope, urlSelections.dataScope])

  useEffect(() => {
    if (hasUserInteractedRef.current) return
    if (urlSelections.dateRange && urlSelections.dateRange !== dateRangePreset) {
      setDateRangePreset(urlSelections.dateRange)
    }
  }, [dateRangePreset, setDateRangePreset, urlSelections.dateRange])

  useEffect(() => {
    if (hasUserInteractedRef.current) return
    if (urlSelections.tableView && urlSelections.tableView !== tableView) {
      setTableView(urlSelections.tableView)
    }
  }, [setTableView, tableView, urlSelections.tableView])

  useEffect(() => {
    if (hasUserInteractedRef.current) return
    if (urlSelections.dateRange !== 'custom') return
    if (urlSelections.customStartDate && urlSelections.customStartDate !== customStartDate) {
      setCustomStartDate(urlSelections.customStartDate)
    }
    if (urlSelections.customEndDate && urlSelections.customEndDate !== customEndDate) {
      setCustomEndDate(urlSelections.customEndDate)
    }
  }, [customEndDate, customStartDate, urlSelections.customEndDate, urlSelections.customStartDate, urlSelections.dateRange])

  const buildSearchParamsFromState = (overrides = {}) => {
    const nextState = {
      season: selectedSeason,
      dataScope: selectedDataScope,
      dateRange: dateRangePreset,
      tableView,
      sortColumn,
      sortDirection,
      customStartDate,
      customEndDate,
      ...overrides,
    }

    const nextParams = new URLSearchParams()
    if (nextState.season) nextParams.set('season', nextState.season)
    if (nextState.dataScope) nextParams.set('scope', nextState.dataScope)
    if (nextState.dateRange) nextParams.set('range', nextState.dateRange)
    if (nextState.dateRange === 'custom') {
      if (nextState.customStartDate) nextParams.set('start', nextState.customStartDate)
      if (nextState.customEndDate) nextParams.set('end', nextState.customEndDate)
    }
    if (nextState.tableView) nextParams.set('view', nextState.tableView)
    if (nextState.sortColumn) nextParams.set('sort', nextState.sortColumn)
    if (nextState.sortDirection) nextParams.set('direction', nextState.sortDirection)
    return nextParams
  }

  const updateUrlFromUserChange = (overrides = {}) => {
    hasUserInteractedRef.current = true
    const nextParams = buildSearchParamsFromState(overrides)
    if (nextParams.toString() !== searchParams.toString()) {
      setSearchParams(nextParams, { replace: true })
    }
  }

  const handleSeasonChange = (value) => {
    updateUrlFromUserChange({ season: value })
    setSelectedSeason(value)
  }

  const handleDataScopeChange = (value) => {
    updateUrlFromUserChange({ dataScope: value })
    setSelectedDataScope(value)
  }

  const handleDateRangeChange = (value) => {
    updateUrlFromUserChange({ dateRange: value })
    setDateRangePreset(value)
  }

  const handleCustomStartDateChange = (value) => {
    updateUrlFromUserChange({ dateRange: 'custom', customStartDate: value })
    setCustomStartDate(value)
  }

  const handleCustomEndDateChange = (value) => {
    updateUrlFromUserChange({ dateRange: 'custom', customEndDate: value })
    setCustomEndDate(value)
  }

  const handleTableViewChange = (value) => {
    updateUrlFromUserChange({ tableView: value })
    setTableView(value)
  }

  // Reset date preset when season changes
  useEffect(() => {
    const nextDateRangePreset = hasUserInteractedRef.current
      ? dateRangePreset
      : (urlSelections.dateRange || 'season')
    const shouldUseCustomDates = nextDateRangePreset === 'custom'

    setDateRangePreset(nextDateRangePreset)
    setCustomStartDate(shouldUseCustomDates ? customStartDate : '')
    setCustomEndDate(shouldUseCustomDates ? customEndDate : '')
    setSeasonBounds({ first: '', last: '' })
    setTopContributorsNote('')
  }, [
    selectedSeason,
    selectedDataScope,
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

  useEffect(() => {
    if (!data) {
      setTableMaxHeight(null)
      return undefined
    }

    let frameId = 0

    const updateTableMaxHeight = () => {
      frameId = 0
      const tableNode = tableContainerRef.current
      if (!tableNode) return

      const { top } = tableNode.getBoundingClientRect()
      const availableHeight = Math.floor(window.innerHeight - top - TABLE_VIEWPORT_BOTTOM_GUTTER)
      if (availableHeight <= 0) return

      const nextHeight = Math.max(MIN_TABLE_VIEWPORT_HEIGHT, availableHeight)
      setTableMaxHeight((prevHeight) => (prevHeight === nextHeight ? prevHeight : nextHeight))
    }

    const scheduleTableResize = () => {
      if (frameId) cancelAnimationFrame(frameId)
      frameId = requestAnimationFrame(updateTableMaxHeight)
    }

    scheduleTableResize()
    window.addEventListener('resize', scheduleTableResize)
    window.addEventListener('orientationchange', scheduleTableResize)

    return () => {
      if (frameId) cancelAnimationFrame(frameId)
      window.removeEventListener('resize', scheduleTableResize)
      window.removeEventListener('orientationchange', scheduleTableResize)
    }
  }, [data, isCustomDateRangeVisible, tableView])

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
          if (dateRangePreset !== 'custom') {
            setCustomStartDate(res.first_game_date)
            setCustomEndDate(res.last_game_date)
          } else {
            if (!customStartDate) setCustomStartDate(res.first_game_date)
            if (!customEndDate) setCustomEndDate(res.last_game_date)
          }
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
  }, [selectedSeason, selectedDataScope, dateRangePreset, customStartDate, customEndDate])

  useEffect(() => {
    if (!hasUserInteractedRef.current) return
    const nextParams = buildSearchParamsFromState()
    if (nextParams.toString() !== searchParams.toString()) {
      setSearchParams(nextParams, { replace: true })
    }
  }, [
    customEndDate,
    customStartDate,
    dateRangePreset,
    searchParams,
    selectedDataScope,
    selectedSeason,
    setSearchParams,
    sortColumn,
    sortDirection,
    tableView,
  ])

  // Identifies the data the current filters ask for; while it differs from the loaded
  // data's key the previous table stays on screen, dimmed.
  const requestKey = [selectedSeason, selectedDataScope, startDate, endDate, lastNGames, excludePlayoffs].join('|')
  const isRefreshing = Boolean(data) && !error && (loading || loadedDataKey !== requestKey)

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
          setLoadedDataKey(requestKey)
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
      setTopContributorsNote('Top contributors are currently unavailable for the garbage-time-only and non-clutch-time views.')
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
          setTopContributorsNote(
            selectedDataScope === 'all'
              ? ''
              : 'Top contributors are currently unavailable for this data scope.'
          )
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
    if (hasUserInteractedRef.current) return
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

  // Mean / SD per tinted column, for the optional color scale.
  const columnTintStats = useMemo(() => {
    const stats = new Map()
    if (!data?.teams?.length) return stats
    activeColumns.forEach((column) => {
      const direction = getTintDirection(column)
      if (!direction) return
      const values = data.teams.map((team) => team[column.key]).filter((value) => typeof value === 'number')
      if (values.length < 2) return
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length
      const sd = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length)
      if (sd > 0) stats.set(column.key, { mean, sd, direction })
    })
    return stats
  }, [data, activeColumns])

  // Restrained diverging tint by z-score, mixed toward the tint tokens and capped.
  const getCellTint = (columnKey, value) => {
    if (!colorScaleOn || typeof value !== 'number') return undefined
    const stats = columnTintStats.get(columnKey)
    if (!stats) return undefined
    const z = ((value - stats.mean) / stats.sd) * stats.direction
    const magnitude = Math.min(Math.abs(z), TINT_Z_CAP)
    if (magnitude < TINT_Z_FLOOR) return undefined
    const token = z > 0 ? '--color-positive-bg' : '--color-negative-bg'
    const strength = (magnitude - TINT_Z_FLOOR) / (TINT_Z_CAP - TINT_Z_FLOOR)
    return `color-mix(in srgb, var(${token}) ${Math.round(40 + strength * 60)}%, transparent)`
  }

  // Rank follows the sorted column; an alphabetical (Team) sort has no meaningful order, so
  // rank by NRtg instead.
  const netRatingRankByTeam = useMemo(() => {
    const ranks = new Map()
    if (!data?.teams) return ranks
    ;[...data.teams]
      .sort((a, b) => b.net_rating - a.net_rating)
      .forEach((team, index) => ranks.set(team.team, index + 1))
    return ranks
  }, [data])

  const getRank = (team, index) => (
    sortColumn === 'team' ? netRatingRankByTeam.get(team.team) ?? index + 1 : index + 1
  )

  const handleSort = (column) => {
    if (sortColumn === column) {
      const nextDirection = sortDirection === 'asc' ? 'desc' : 'asc'
      updateUrlFromUserChange({ sortColumn: column, sortDirection: nextDirection })
      setSortDirection(nextDirection)
    } else {
      const nextDirection = getDefaultSortDirectionForColumn(column)
      updateUrlFromUserChange({ sortColumn: column, sortDirection: nextDirection })
      setSortColumn(column)
      setSortDirection(nextDirection)
    }
  }

  const displayColumns = useMemo(
    () => orderColumnsForDisplay(activeColumns, isCompact),
    [activeColumns, isCompact]
  )

  const columnGroups = useMemo(() => buildColumnGroups(displayColumns), [displayColumns])

  // Group dividers start at every group after the first stat group (Team has its own sticky edge).
  const groupStartKeys = useMemo(
    () => new Set(columnGroups.slice(2).map((group) => group.startKey)),
    [columnGroups]
  )

  const columnWidths = isCompact ? TABLE_COLUMN_WIDTHS.compact : TABLE_COLUMN_WIDTHS.regular
  const getColumnWidth = (columnKey) => {
    if (columnKey === 'team') return columnWidths.team
    if (columnKey === 'scope_time_pct') return columnWidths.timePct
    return columnWidths.stat
  }
  const tableMinWidth = displayColumns.reduce(
    (sum, column) => sum + getColumnWidth(column.key),
    columnWidths.rank
  )

  const handleOpenTrendFromCell = (teamAbbr, columnKey) => {
    if (!teamAbbr || !selectedSeason) return
    const stat = resolveTrendStatForColumn(columnKey)
    const trendsScope = selectedDataScope === 'garbage_filtered' || selectedDataScope === 'clutch'
      ? selectedDataScope
      : 'all'
    navigate('/trends', {
      state: {
        season: selectedSeason,
        team: teamAbbr,
        stat,
        dataScope: trendsScope,
      },
    })
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
    const avg = { team: 'Average' }
    activeColumns.forEach(col => {
      if (col.key === 'team') return

      const backendValue = data?.league_averages?.[col.key]
      if (typeof backendValue === 'number') {
        avg[col.key] = backendValue
        return
      }

      const values = data.teams.map(t => t[col.key]).filter(v => typeof v === 'number')
      if (values.length > 0) {
        avg[col.key] = values.reduce((a, b) => a + b, 0) / values.length
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

  // Describes the range actually applied, ending at the last game in range rather than today.
  const effectiveRangeLabel = useMemo(() => {
    const throughLabel = seasonBounds.last ? `through ${formatFullDate(parseIsoDate(seasonBounds.last))}` : ''
    const withThrough = (prefix) => [prefix, throughLabel].filter(Boolean).join(' · ')
    if (lastNGames) return withThrough(`Last ${lastNGames} games`)
    if (dateRangePreset === 'season') return withThrough('Season to date')
    if (dateRangePreset === 'season_regular') return withThrough('Regular season')
    return formatDateRange(startDate, endDate)
  }, [lastNGames, dateRangePreset, startDate, endDate, seasonBounds.last])

  const dataScopeLabel = DATA_SCOPE_OPTIONS.find((option) => option.value === selectedDataScope)?.label || ''

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

  const efficiencyLayout = isCompact ? EFFICIENCY_CHART_LAYOUT.compact : EFFICIENCY_CHART_LAYOUT.regular

  const efficiencyTicks = useMemo(() => ({
    x: getNiceTicks(efficiencyDomains.x, isNarrow ? 4 : 8),
    y: getNiceTicks(efficiencyDomains.y, isNarrow ? 4 : 8),
  }), [efficiencyDomains, isNarrow])

  // Below 600px only the extremes are labeled; every point still has a tap/hover tooltip.
  const labeledTeams = useMemo(() => {
    if (!isNarrow) return new Set(efficiencyChartData.map((point) => point.team))
    const extremes = new Set()
    const pick = (key, direction) => {
      const sorted = [...efficiencyChartData].sort((a, b) => (direction === 'max' ? b[key] - a[key] : a[key] - b[key]))
      if (sorted[0]) extremes.add(sorted[0].team)
    }
    pick('net_rating', 'max')
    pick('net_rating', 'min')
    pick('off_rating', 'max')
    pick('off_rating', 'min')
    pick('def_rating', 'max')
    pick('def_rating', 'min')
    return extremes
  }, [efficiencyChartData, isNarrow])

  const labelPlacements = useMemo(() => placePointLabels({
    points: efficiencyChartData,
    labeledTeams,
    size: chartSize,
    layout: efficiencyLayout,
    domains: efficiencyDomains,
    averages: efficiencyLeagueAverages,
  }), [efficiencyChartData, labeledTeams, chartSize, efficiencyLayout, efficiencyDomains, efficiencyLeagueAverages])

  const renderEfficiencyPoint = ({ cx, cy, payload }) => {
    if (typeof cx !== 'number' || typeof cy !== 'number') return null
    const placement = labelPlacements.get(payload.team)
    return (
      <g className="ls-point">
        <circle cx={cx} cy={cy} r={POINT_RADIUS} fill="var(--color-link)" fillOpacity={0.9} />
        {placement && (
          <text
            x={cx + placement.dx}
            y={cy + placement.dy}
            textAnchor={placement.anchor}
            fontSize={LABEL_FONT_SIZE}
            fontWeight={500}
            fill="var(--color-text-secondary)"
            stroke="var(--color-background)"
            strokeWidth={3}
            strokeLinejoin="round"
            paintOrder="stroke"
            pointerEvents="none"
          >
            {payload.team}
          </text>
        )}
      </g>
    )
  }

  const formatSignedNetRating = (value) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}`

  const renderQuadrantNote = (title, teams, keyPrefix) => (
    <div className={`ls-quadrant ls-quadrant--${keyPrefix}`}>
      <h3 className="ls-quadrant-title">{title}</h3>
      <div className="ls-quadrant-head" aria-hidden="true">
        <span>Team</span>
        <span>NRtg</span>
      </div>
      {teams.length > 0 ? (
        <ul className="ls-quadrant-list">
          {teams.map((team) => (
            <li key={`${keyPrefix}-${team.team}`} className="ls-quadrant-item">
              <span className="ls-quadrant-team">{team.team_city}</span>
              <span className={`ls-quadrant-net ${team.net_rating >= 0 ? 'text-positive' : 'text-negative'}`}>
                {formatSignedNetRating(team.net_rating)}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="ls-quadrant-empty">No teams</p>
      )}
    </div>
  )

  const renderContributorsTable = (items, tone) => (
    <table className="ls-contrib-table">
      <thead>
        <tr>
          <th scope="col">Team</th>
          <th scope="col">Factor</th>
          <th scope="col" className="text-right">Value</th>
          <th scope="col" className="text-right">Contribution</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item, index) => (
          <tr
            key={`${tone}-${index}`}
            onMouseEnter={(event) => showHoverTooltip(event, getContributorTooltip(item))}
            onMouseLeave={hideHoverTooltip}
          >
            <td className="ls-contrib-team">{item.team}</td>
            <td>{item.factor_label}</td>
            <td className="text-right">{item.value.toFixed(1)}</td>
            <td className={`text-right ls-contrib-value ${tone === 'positive' ? 'text-positive' : 'text-negative'}`}>
              {tone === 'positive' ? '+' : ''}{item.contribution.toFixed(2)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )

  const getSortLabel = (direction) => (direction === 'asc' ? 'ascending' : 'descending')

  const viewOptions = [
    { value: VIEW_FOUR_FACTORS, label: 'Four Factors' },
    { value: VIEW_SHOOTING, label: 'Shooting' },
    {
      value: VIEW_SOS_ADJUSTMENTS,
      label: isCustomDateRangeVisible || isNarrow ? 'SOS' : 'Strength of Schedule',
      title: 'Strength of Schedule',
    },
  ]

  return (
    <div className="ls-page">
      <div className="page-header">
        <div className="page-title-row">
          <h1 className="page-title">League Summary</h1>
          <p className="page-description">
            Compare team statistics across the league. Click column headers to sort.
          </p>
        </div>
      </div>

      <div className="card ls-controls">
        <div className="ls-filters">
          <div className="ls-field ls-field--season">
            <label className="form-label" htmlFor="ls-season">Season</label>
            <select
              id="ls-season"
              className="form-select"
              value={selectedSeason}
              onChange={(e) => handleSeasonChange(e.target.value)}
            >
              {/* Placeholder only while seasons load; once loaded a season is always selected */}
              {(!seasons.length || !selectedSeason) && (
                <option value="" disabled hidden>{seasons.length ? 'Select season...' : 'Loading…'}</option>
              )}
              {seasons.map((season) => (
                <option key={season} value={season}>{season}</option>
              ))}
            </select>
          </div>

          <div className="ls-field">
            <label className="form-label" htmlFor="ls-scope">Data Scope</label>
            <select
              id="ls-scope"
              className="form-select"
              value={selectedDataScope}
              onChange={(e) => handleDataScopeChange(e.target.value)}
            >
              {DATA_SCOPE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="ls-field ls-field--range">
            <label className="form-label" htmlFor="ls-range">Date Range</label>
            <select
              id="ls-range"
              className="form-select"
              value={dateRangePreset}
              onChange={(e) => handleDateRangeChange(e.target.value)}
            >
              {DATE_RANGE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          {isCustomDateRangeVisible && (
            <>
              <div className="ls-field ls-field--date">
                <label className="form-label" htmlFor="ls-start-date">Start Date</label>
                <input
                  id="ls-start-date"
                  type="date"
                  className="form-input"
                  value={customStartDate}
                  min={seasonBounds.first || undefined}
                  max={seasonBounds.last || undefined}
                  onChange={(e) => handleCustomStartDateChange(e.target.value)}
                />
              </div>

              <div className="ls-field ls-field--date">
                <label className="form-label" htmlFor="ls-end-date">End Date</label>
                <input
                  id="ls-end-date"
                  type="date"
                  className="form-input"
                  value={customEndDate}
                  min={seasonBounds.first || undefined}
                  max={seasonBounds.last || undefined}
                  onChange={(e) => handleCustomEndDateChange(e.target.value)}
                />
              </div>
            </>
          )}

          <div className="ls-field ls-field--view">
            <span className="form-label" aria-hidden="true">Table View</span>
            <SegmentedControl
              label="Table view"
              options={viewOptions}
              value={tableView}
              onChange={handleTableViewChange}
              block={isCompact}
            />
          </div>

          <div className="ls-refresh-slot" role="status" aria-live="polite">
            {isRefreshing && (
              <span className="refresh-indicator">
                <span className="loading-spinner" aria-hidden="true"></span>
                Updating…
              </span>
            )}
          </div>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {!data && loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading league data...
        </div>
      )}

      {data && (
        <div className={`ls-results${isRefreshing ? ' is-refreshing' : ''}`} aria-busy={isRefreshing}>
          <section className="card ls-table-card">
            <div className="ls-table-head">
              <div className="ls-card-head">
                <h2 className="card-title">{TABLE_VIEW_TITLES[tableView]}</h2>
                <span className="ls-card-meta">
                  {[dataScopeLabel, effectiveRangeLabel].filter(Boolean).join(' · ')}
                </span>
              </div>
              <div className="ls-table-actions">
                <button
                  type="button"
                  className="btn btn-secondary ls-scale-toggle"
                  aria-pressed={colorScaleOn}
                  title="Tint each stat column from worse (red) to better (green) than the league"
                  onClick={() => setColorScaleOn((on) => !on)}
                >
                  <span className="ls-scale-swatch" aria-hidden="true"></span>
                  Color scale
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleExport}
                  disabled={!sortedTeams.length}
                >
                  <DownloadIcon />
                  Export CSV
                </button>
              </div>
            </div>

            <div
              ref={tableContainerRef}
              className="table-scroll ls-table-scroll"
              role="region"
              aria-label="League summary table"
              tabIndex={0}
              style={tableMaxHeight ? { '--ls-table-max-height': `${tableMaxHeight}px` } : undefined}
            >
              <table
                className="ls-table"
                style={{ minWidth: `${tableMinWidth}px`, '--ls-rank-width': `${columnWidths.rank}px` }}
              >
                <colgroup>
                  <col style={{ width: `${columnWidths.rank}px` }} />
                  {displayColumns.map((col) => (
                    <col key={col.key} style={{ width: `${getColumnWidth(col.key)}px` }} />
                  ))}
                </colgroup>
                <thead>
                  <tr className="ls-group-row">
                    {columnGroups.map((group, index) => (
                      <th
                        key={`${group.id}-${group.startKey}`}
                        scope="colgroup"
                        colSpan={index === 0 ? group.span + 1 : group.span}
                        className={[
                          index === 0 ? 'ls-sticky ls-sticky--corner' : '',
                          groupStartKeys.has(group.startKey) ? 'ls-group-start' : '',
                        ].filter(Boolean).join(' ') || undefined}
                      >
                        {group.label}
                      </th>
                    ))}
                  </tr>
                  <tr className="ls-col-row">
                    <th
                      scope="col"
                      className="ls-rank ls-sticky"
                      onMouseEnter={(event) => showHoverTooltip(event, { lines: [getHeaderTooltip('rank')] })}
                      onMouseLeave={hideHoverTooltip}
                    >
                      Rank
                    </th>
                    {displayColumns.map((col) => {
                      const isSorted = sortColumn === col.key
                      const isOpponent = col.labelLine2 === 'Opp'
                      const tooltip = { lines: [getHeaderTooltip(col.key, isOpponent)] }
                      const classes = [
                        col.key === 'team' ? 'ls-team ls-sticky' : 'ls-num',
                        isSorted ? 'is-sorted' : '',
                        groupStartKeys.has(col.key) ? 'ls-group-start' : '',
                      ].filter(Boolean).join(' ')

                      return (
                        <th
                          key={col.key}
                          scope="col"
                          className={classes}
                          aria-sort={isSorted ? getSortLabel(sortDirection) : undefined}
                          onMouseEnter={(event) => showHoverTooltip(event, tooltip)}
                          onMouseLeave={hideHoverTooltip}
                        >
                          <button
                            type="button"
                            className="th-sort"
                            onClick={() => handleSort(col.key)}
                            onFocus={(event) => showHoverTooltip(event, tooltip)}
                            onBlur={hideHoverTooltip}
                          >
                            {isOpponent && <span className="sr-only">Opponent </span>}
                            {col.label}
                            <span className="ls-sort-icon" aria-hidden="true">
                              {isSorted ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                            </span>
                          </button>
                        </th>
                      )
                    })}
                  </tr>
                </thead>
                <tbody>
                  {sortedTeams.map((team, index) => (
                    <tr key={team.team}>
                      <td
                        className="ls-rank ls-sticky ls-cell-link"
                        title={`Click to open Statistical Trends: ${team.team} Net Rating`}
                        onClick={() => handleOpenTrendFromCell(team.team, 'net_rating')}
                      >
                        {getRank(team, index)}
                      </td>
                      {displayColumns.map((col) => {
                        if (col.key === 'team') {
                          return (
                            <th key={col.key} scope="row" className="ls-team ls-sticky">
                              <button
                                type="button"
                                className="ls-team-link"
                                title={`Open Statistical Trends: ${team.team} Net Rating`}
                                onClick={() => handleOpenTrendFromCell(team.team, col.key)}
                              >
                                {team.team}
                              </button>
                            </th>
                          )
                        }

                        const trendStat = resolveTrendStatForColumn(col.key)
                        const trendLabel = TRENDS_STAT_LABELS[trendStat] || 'Net Rating'
                        const fromLabel = col.labelLine2 ? `${col.labelLine2} ${col.label}` : col.label
                        const title = TRENDS_DIRECT_STAT_COLUMNS.has(col.key)
                          ? `Click to open Statistical Trends: ${team.team} ${fromLabel}`
                          : `Click to open Statistical Trends: ${team.team} ${trendLabel} (from ${fromLabel})`

                        return (
                          <td
                            key={col.key}
                            className={`ls-num ls-cell-link${groupStartKeys.has(col.key) ? ' ls-group-start' : ''}`}
                            style={{ backgroundColor: getCellTint(col.key, team[col.key]) }}
                            title={title}
                            onClick={() => handleOpenTrendFromCell(team.team, col.key)}
                          >
                            {formatValue(team[col.key], col.key)}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
                {leagueAverages && (
                  <tfoot>
                    <tr className="ls-avg-row">
                      <td className="ls-rank ls-sticky"></td>
                      {displayColumns.map((col) => (
                        col.key === 'team' ? (
                          <th key={col.key} scope="row" className="ls-team ls-sticky" title="League average">Avg</th>
                        ) : (
                          <td
                            key={col.key}
                            className={`ls-num${groupStartKeys.has(col.key) ? ' ls-group-start' : ''}`}
                          >
                            {formatValue(leagueAverages[col.key], col.key)}
                          </td>
                        )
                      ))}
                    </tr>
                  </tfoot>
                )}
              </table>
            </div>
          </section>

          {!!efficiencyChartData.length && (
            <section className="card ls-chart-card">
              <div className="ls-card-head">
                <h2 className="card-title">Efficiency Landscape</h2>
                <span className="ls-card-meta">{effectiveRangeLabel}</span>
              </div>
              <div className="ls-landscape">
                <div className="ls-landscape-chart" ref={chartRef}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={efficiencyLayout.margin}>
                      <XAxis
                        type="number"
                        dataKey="off_rating"
                        domain={efficiencyDomains.x}
                        ticks={efficiencyTicks.x}
                        tickFormatter={formatTick}
                        tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
                        tickLine={false}
                        axisLine={{ stroke: 'var(--chart-grid)' }}
                        height={efficiencyLayout.xAxisHeight}
                        label={{
                          value: 'Offensive Rating',
                          position: 'insideBottom',
                          offset: 0,
                          fill: 'var(--color-text-secondary)',
                          fontSize: 12,
                        }}
                      />
                      <YAxis
                        type="number"
                        dataKey="def_rating"
                        domain={efficiencyDomains.y}
                        reversed
                        ticks={efficiencyTicks.y}
                        tickFormatter={formatTick}
                        tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
                        tickLine={false}
                        axisLine={{ stroke: 'var(--chart-grid)' }}
                        width={efficiencyLayout.yAxisWidth}
                        label={{
                          value: 'Defensive Rating',
                          angle: -90,
                          position: 'insideLeft',
                          offset: 4,
                          fill: 'var(--color-text-secondary)',
                          fontSize: 12,
                          style: { textAnchor: 'middle' },
                        }}
                      />
                      <ReferenceLine
                        x={efficiencyLeagueAverages.off}
                        stroke="var(--chart-axis)"
                        strokeDasharray="4 4"
                        strokeWidth={1}
                        label={renderAverageLabel(`Lg avg ${efficiencyLeagueAverages.off.toFixed(1)}`, 'vertical')}
                      />
                      <ReferenceLine
                        y={efficiencyLeagueAverages.def}
                        stroke="var(--chart-axis)"
                        strokeDasharray="4 4"
                        strokeWidth={1}
                        label={renderAverageLabel(`Lg avg ${efficiencyLeagueAverages.def.toFixed(1)}`, 'horizontal')}
                      />
                      <Tooltip
                        cursor={false}
                        content={<EfficiencyTooltip />}
                        wrapperStyle={{ outline: 'none' }}
                      />
                      <Scatter data={efficiencyChartData} shape={renderEfficiencyPoint} isAnimationActive={false} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                {renderQuadrantNote('Bad Offense & Good Defense', quadrantTeams.badOffGoodDef, 'tl')}
                {renderQuadrantNote('Good Offense & Good Defense', quadrantTeams.goodOffGoodDef, 'tr')}
                {renderQuadrantNote('Bad Offense & Bad Defense', quadrantTeams.badOffBadDef, 'bl')}
                {renderQuadrantNote('Good Offense & Bad Defense', quadrantTeams.goodOffBadDef, 'br')}
              </div>
              <p className="ls-chart-credit">Chart concept: Kirk Goldsberry</p>
            </section>
          )}

          {(topContributors || contributorsLoading || topContributorsNote) && (
            <section className="card ls-contrib-card">
              <div className="ls-card-head">
                <h2 className="card-title">Top Contributors to Net Rating</h2>
                {topContributors && contributorPeriodHeadingLabel ? (
                  <span className="ls-card-meta">{contributorPeriodHeadingLabel}</span>
                ) : null}
              </div>

              {topContributors && (
                <div className={`ls-contrib-grid${contributorsLoading && !isRefreshing ? ' is-refreshing' : ''}`}>
                  <div className="ls-contrib-column">
                    <h3 className="ls-contrib-title text-positive">Top Positive Contributors</h3>
                    {renderContributorsTable(topContributors.top_positive, 'positive')}
                  </div>
                  <div className="ls-contrib-column">
                    <h3 className="ls-contrib-title text-negative">Top Negative Contributors</h3>
                    {renderContributorsTable(topContributors.top_negative, 'negative')}
                  </div>
                </div>
              )}

              {!topContributors && contributorsLoading && (
                <div className="loading">
                  <div className="loading-spinner"></div>
                  Loading top contributors...
                </div>
              )}

              {topContributorsNote && !contributorsLoading && (
                <p className="ls-contrib-note">{topContributorsNote}</p>
              )}
            </section>
          )}
        </div>
      )}

      {hoverTooltip && (
        <div
          className={`ls-tooltip ls-tooltip--${hoverTooltip.placement}`}
          style={{ left: `${hoverTooltip.x}px`, top: `${hoverTooltip.y}px` }}
          role="tooltip"
          aria-hidden="true"
        >
          {hoverTooltip.title ? <div className="ls-tooltip-title">{hoverTooltip.title}</div> : null}
          {(hoverTooltip.lines || []).map((line, index) => (
            <div key={`${line}-${index}`}>{line}</div>
          ))}
        </div>
      )}

      <section className="card ls-glossary">
        <button
          type="button"
          className="ls-glossary-toggle"
          onClick={() => setGlossaryExpanded(!glossaryExpanded)}
          aria-expanded={glossaryExpanded}
          aria-controls="ls-glossary-list"
        >
          <span>Glossary</span>
          <span className="ls-glossary-icon" aria-hidden="true">{glossaryExpanded ? '−' : '+'}</span>
        </button>

        {glossaryExpanded && (
          <dl id="ls-glossary-list" className="ls-glossary-list">
            {GLOSSARY_ITEMS.map((item) => (
              <div key={item.term} className="ls-glossary-item">
                <dt>{item.term}</dt>
                <dd>{item.definition}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>
    </div>
  )
}

export default LeagueSummary
