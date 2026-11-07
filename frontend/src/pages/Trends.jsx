import { useState, useEffect, useMemo, useRef } from 'react'
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import {
  ComposedChart,
  Bar,
  Cell,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { getSeasons, getTeams, getTrends } from '../api'
import { usePersistedState } from '../hooks/usePersistedState'
import { getTeamName } from '../constants/teams'
import useMediaQuery from '../hooks/useMediaQuery'
import { niceTicks, formatShortDate, TOOLTIP_WRAPPER_STYLE } from '../utils/chart'
import './Trends.css'

const STAT_OPTIONS = [
  { value: 'net_rating', label: 'Net Rating' },
  { value: 'off_rating', label: 'Offensive Rating' },
  { value: 'def_rating', label: 'Defensive Rating' },
  { value: 'efg_pct', label: 'Effective FG%' },
  { value: 'ball_handling', label: 'Ball-Handling' },
  { value: 'oreb_pct', label: 'Offensive Rebound %' },
  { value: 'ft_rate', label: 'Free Throw Rate' },
  { value: 'ft_pct', label: 'Free Throw %' },
  { value: 'opp_efg_pct', label: 'Opp Effective FG%' },
  { value: 'opp_ball_handling', label: 'Opp Ball-Handling' },
  { value: 'opp_oreb_pct', label: 'Opp Offensive Rebound %' },
  { value: 'opp_ft_rate', label: 'Opp Free Throw Rate' },
  { value: 'opp_ft_pct', label: 'Opp Free Throw %' },
  { value: 'fg2_pct', label: '2-Point FG%' },
  { value: 'fg3_pct', label: '3-Point FG%' },
  { value: 'fg3a_rate', label: '3-Point Attempt Rate' },
  { value: 'opp_fg2_pct', label: 'Opp 2-Point FG%' },
  { value: 'opp_fg3_pct', label: 'Opp 3-Point FG%' },
  { value: 'opp_fg3a_rate', label: 'Opp 3-Point Attempt Rate' },
  { value: 'pace', label: 'Pace' },
]

// Stats where lower values are better for the team
const LOWER_IS_BETTER_STATS = new Set([
  'def_rating',
  'opp_efg_pct',
  'opp_ball_handling',
  'opp_oreb_pct',
  'opp_ft_rate',
  'opp_ft_pct',
  'opp_fg2_pct',
  'opp_fg3_pct',
  'opp_fg3a_rate',
])

const GLOSSARY_ITEMS = [
  { term: 'Net Rating', definition: 'Offensive Rating minus Defensive Rating. Measures overall point differential per 100 possessions.' },
  { term: 'ORtg', definition: 'Offensive Rating - Points scored per 100 possessions.' },
  { term: 'DRtg', definition: 'Defensive Rating - Points allowed per 100 possessions. Lower is better.' },
  { term: 'eFG%', definition: 'Effective Field Goal Percentage - Adjusts FG% for the added value of 3-pointers. Formula: (FGM + 0.5 × 3PM) / FGA × 100' },
  { term: 'BH', definition: 'Ball Handling - Measures ability to take care of the ball. Calculated as 100 - TOV%. Higher is better.' },
  { term: 'OREB%', definition: 'Offensive Rebounding Percentage - Percentage of available offensive rebounds grabbed.' },
  { term: 'FT Rate', definition: 'Free Throw Rate - Free throws made per field goal attempt (FTM / FGA × 100).' },
  { term: 'FT%', definition: 'Free Throw Percentage - Free throws made divided by free throw attempts (FTM / FTA × 100).' },
  { term: 'FG2%', definition: '2-Point Field Goal Percentage - 2PT FGM / 2PT FGA × 100.' },
  { term: 'FG3%', definition: '3-Point Field Goal Percentage - 3PT FGM / 3PT FGA × 100.' },
  { term: 'FG3A Rate', definition: '3-Point Attempt Rate - Percentage of field goal attempts that are 3-pointers (3PA / FGA × 100).' },
  { term: 'Opp', definition: 'Opponent statistics - What your opponents shoot/do against you.' },
  { term: 'Pace', definition: 'Average possessions per game for both teams. Higher pace indicates a faster-paced game.' },
]

// Bar fill encodes W/L at full strength (≥3:1 on white); legend swatches use the same fills.
const WIN_BAR_FILL = 'var(--chart-positive)'
const LOSS_BAR_FILL = 'var(--chart-negative)'
const TEAM_AVERAGE_STROKE = 'var(--color-accent)'
const MOBILE_QUERY = '(max-width: 768px)'
const DATA_SCOPE_OPTIONS = [
  { value: 'all', label: 'All data' },
  { value: 'garbage_filtered', label: 'Excl. garbage time' },
]
const MOVING_AVERAGE_OPTIONS = [3, 4, 5, 10, 15, 20, 25, 30]
const VALID_STAT_OPTIONS = new Set(STAT_OPTIONS.map((option) => option.value))
const VALID_DATA_SCOPE_OPTIONS = new Set(DATA_SCOPE_OPTIONS.map((option) => option.value))
const VALID_MOVING_AVERAGE_OPTIONS = new Set(MOVING_AVERAGE_OPTIONS)
const normalizeGameId = (value) => {
  const text = String(value ?? '').trim()
  if (!text) return ''
  return /^\d+$/.test(text) ? text.padStart(10, '0') : text
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

function normalizeDataScopeParam(value) {
  const token = normalizeQueryToken(value)
  if (!token) return ''
  if (VALID_DATA_SCOPE_OPTIONS.has(token)) return token
  if (token === 'garbage' || token === 'non_garbage' || token === 'non_garbage_time') return 'garbage_filtered'
  return ''
}

function normalizeStatParam(value) {
  const token = normalizeQueryToken(value)
  if (!token) return ''
  if (VALID_STAT_OPTIONS.has(token)) return token
  const aliases = {
    ortg: 'off_rating',
    offensive_rating: 'off_rating',
    drtg: 'def_rating',
    defensive_rating: 'def_rating',
    nrtg: 'net_rating',
    net_rating: 'net_rating',
  }
  return aliases[token] || ''
}

function normalizeMovingAverageParam(value) {
  const parsed = Number.parseInt(String(value ?? '').trim(), 10)
  return VALID_MOVING_AVERAGE_OPTIONS.has(parsed) ? parsed : null
}

// Evenly spaced "nice" ticks (steps of 1, 2, 2.5, 5 × 10^k) covering [min, max]
// "2026-10-20" -> "10/20"
// Two-line horizontal tick: date on top, opponent below. Value is "10/20 @DET".
function GameTick({ x, y, payload }) {
  const [date, opponent] = String(payload?.value ?? '').split(' ')
  return (
    <g transform={`translate(${x},${y})`}>
      <text textAnchor="middle" fill="var(--chart-axis)" fontSize={11}>
        <tspan x={0} dy={12}>{date}</tspan>
        <tspan x={0} dy={13}>{opponent}</tspan>
      </text>
    </g>
  )
}

// Table date as a real link to the game's Game Analysis view (keyboard/screen-reader drill-down)
function GameDateLink({ target, game }) {
  const dates = (
    <>
      <span className="st-date-long">{game.game_date}</span>
      <span className="st-date-short">{String(game.game_date).slice(5).replace('-', '/')}</span>
    </>
  )
  if (!target) return dates
  const opponent = `${game.home_away === 'home' ? 'vs' : 'at'} ${game.opponent}`
  return (
    <Link
      className="st-game-link"
      to={target.to}
      state={target.state}
      aria-label={`Open Game Analysis: ${game.game_date} ${opponent}`}
    >
      {dates}
    </Link>
  )
}

function Trends() {
  const location = useLocation()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const [seasons, setSeasons] = useState([])
  const [teams, setTeams] = useState([])
  const [selectedSeason, setSelectedSeason] = usePersistedState('trends_season', '')
  const [selectedDataScope, setSelectedDataScope] = usePersistedState('trends_datascope', 'all')
  const [selectedTeam, setSelectedTeam] = usePersistedState('trends_team', '', { persist: true })
  const [selectedStat, setSelectedStat] = usePersistedState('trends_stat', 'net_rating')
  const [selectedMovingAverage, setSelectedMovingAverage] = usePersistedState('trends_moving_average', 5)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [error, setError] = useState(null)
  const [glossaryExpanded, setGlossaryExpanded] = useState(false)
  const hasUserInteractedRef = useRef(false)
  const isMobile = useMediaQuery(MOBILE_QUERY)
  const urlSelections = useMemo(() => ({
    season: readSearchParam(searchParams, 'season').trim(),
    team: readSearchParam(searchParams, 'team').trim().toUpperCase(),
    stat: normalizeStatParam(readSearchParam(searchParams, 'stat', 'metric')),
    dataScope: normalizeDataScopeParam(readSearchParam(searchParams, 'scope', 'data_scope', 'dataScope')),
    movingAverage: normalizeMovingAverageParam(readSearchParam(searchParams, 'ma', 'moving_average', 'movingAverage')),
  }), [searchParams])

  const buildSearchParamsFromState = (overrides = {}) => {
    const nextState = {
      season: selectedSeason,
      team: selectedTeam,
      stat: selectedStat,
      dataScope: selectedDataScope,
      movingAverage: selectedMovingAverage,
      ...overrides,
    }

    const nextParams = new URLSearchParams()
    if (nextState.season) nextParams.set('season', nextState.season)
    if (nextState.team) nextParams.set('team', nextState.team)
    if (nextState.stat) nextParams.set('stat', nextState.stat)
    if (nextState.dataScope) nextParams.set('scope', nextState.dataScope)
    if (nextState.movingAverage) nextParams.set('ma', String(nextState.movingAverage))
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

  const handleTeamChange = (value) => {
    updateUrlFromUserChange({ team: value })
    setSelectedTeam(value)
  }

  const handleDataScopeChange = (value) => {
    updateUrlFromUserChange({ dataScope: value })
    setSelectedDataScope(value)
  }

  const handleStatChange = (value) => {
    updateUrlFromUserChange({ stat: value })
    setSelectedStat(value)
  }

  const handleMovingAverageChange = (value) => {
    const nextValue = Number.parseInt(value, 10)
    if (!VALID_MOVING_AVERAGE_OPTIONS.has(nextValue)) return
    updateUrlFromUserChange({ movingAverage: nextValue })
    setSelectedMovingAverage(nextValue)
  }

  useEffect(() => {
    const navState = location.state
    if (!navState || typeof navState !== 'object') return

    if (typeof navState.season === 'string' && navState.season) {
      setSelectedSeason(navState.season)
    }
    if (typeof navState.dataScope === 'string' && VALID_DATA_SCOPE_OPTIONS.has(navState.dataScope)) {
      setSelectedDataScope(navState.dataScope)
    }
    if (typeof navState.stat === 'string' && VALID_STAT_OPTIONS.has(navState.stat)) {
      setSelectedStat(navState.stat)
    }
    if (typeof navState.team === 'string' && navState.team) {
      setSelectedTeam(navState.team)
    }
  }, [location.key, location.state, setSelectedSeason, setSelectedDataScope, setSelectedStat, setSelectedTeam])

  useEffect(() => {
    if (urlSelections.dataScope && urlSelections.dataScope !== selectedDataScope) {
      setSelectedDataScope(urlSelections.dataScope)
    }
  }, [selectedDataScope, setSelectedDataScope, urlSelections.dataScope])

  useEffect(() => {
    if (urlSelections.stat && urlSelections.stat !== selectedStat) {
      setSelectedStat(urlSelections.stat)
    }
  }, [selectedStat, setSelectedStat, urlSelections.stat])

  useEffect(() => {
    if (urlSelections.movingAverage && urlSelections.movingAverage !== selectedMovingAverage) {
      setSelectedMovingAverage(urlSelections.movingAverage)
    }
  }, [selectedMovingAverage, setSelectedMovingAverage, urlSelections.movingAverage])

  useEffect(() => {
    if (!DATA_SCOPE_OPTIONS.some((option) => option.value === selectedDataScope)) {
      setSelectedDataScope('all')
    }
  }, [selectedDataScope, setSelectedDataScope])

  useEffect(() => {
    if (!VALID_MOVING_AVERAGE_OPTIONS.has(selectedMovingAverage)) {
      setSelectedMovingAverage(5)
    }
  }, [selectedMovingAverage, setSelectedMovingAverage])

  useEffect(() => {
    let isCurrent = true
    async function loadSeasons() {
      try {
        const res = await getSeasons()
        if (!isCurrent) return
        setSeasons(res.seasons)
        // Keep persisted season if valid, otherwise default to first
        setSelectedSeason(prev => {
          if (prev && res.seasons.includes(prev)) return prev
          return res.seasons.length > 0 ? res.seasons[0] : ''
        })
      } catch (err) {
        if (isCurrent) setError(err.message)
      }
    }
    loadSeasons()
    return () => { isCurrent = false }
  }, [setSelectedSeason])

  useEffect(() => {
    if (urlSelections.season && seasons.includes(urlSelections.season) && selectedSeason !== urlSelections.season) {
      setSelectedSeason(urlSelections.season)
    }
  }, [seasons, selectedSeason, setSelectedSeason, urlSelections.season])

  useEffect(() => {
    let isCurrent = true
    async function loadTeams() {
      if (!selectedSeason) return
      // Keep the previous chart on screen (dimmed) while the new season/scope loads.
      setInitializing(true)
      try {
        const res = await getTeams(selectedSeason, selectedDataScope)
        if (!isCurrent) return
        setTeams(res.teams)
        if (res.teams.length === 0) setData(null)
        // Keep persisted team if valid, otherwise default to BOS or first team
        setSelectedTeam(prev => {
          if (prev && res.teams.includes(prev)) return prev
          if (res.teams.includes('BOS')) return 'BOS'
          if (res.teams.length > 0) return res.teams[0]
          return ''
        })
      } catch (err) {
        if (isCurrent) {
          setError(err.message)
          setData(null)
        }
      } finally {
        if (isCurrent) setInitializing(false)
      }
    }
    loadTeams()
    return () => { isCurrent = false }
  }, [selectedSeason, selectedDataScope, setSelectedTeam])

  useEffect(() => {
    if (urlSelections.team && teams.includes(urlSelections.team) && selectedTeam !== urlSelections.team) {
      setSelectedTeam(urlSelections.team)
    }
  }, [teams, selectedTeam, setSelectedTeam, urlSelections.team])

  useEffect(() => {
    if (!hasUserInteractedRef.current) return
    const nextParams = buildSearchParamsFromState()
    if (nextParams.toString() !== searchParams.toString()) {
      setSearchParams(nextParams, { replace: true })
    }
  }, [
    searchParams,
    selectedDataScope,
    selectedMovingAverage,
    selectedSeason,
    selectedStat,
    selectedTeam,
    setSearchParams,
  ])

  useEffect(() => {
    let isCurrent = true
    async function loadTrends() {
      if (!selectedSeason || !selectedTeam || !selectedStat) return
      setLoading(true)
      if (isCurrent) setError(null)
      try {
        const res = await getTrends(selectedSeason, selectedTeam, selectedStat, selectedDataScope)
        if (isCurrent) setData(res)
      } catch (err) {
        if (isCurrent) setError(err.message)
      } finally {
        if (isCurrent) setLoading(false)
      }
    }
    loadTrends()
    return () => { isCurrent = false }
  }, [selectedSeason, selectedTeam, selectedStat, selectedDataScope])

  // Prepare chart data with x-axis labels
  const chartData = useMemo(() => {
    if (!data?.data) return []
    let rollingSum = 0
    const leagueAverage = Number(data.league_average)
    const baseline = Number.isFinite(leagueAverage) ? leagueAverage : 0

    return data.data.map((game, index, series) => {
      const value = Number(game.value ?? 0)
      rollingSum += value

      if (index >= selectedMovingAverage) {
        rollingSum -= Number(series[index - selectedMovingAverage].value ?? 0)
      }

      const windowSize = Math.min(index + 1, selectedMovingAverage)
      const movingAverage = Number((rollingSum / windowSize).toFixed(1))

      return {
        ...game,
        game_id: normalizeGameId(game.game_id),
        xLabel: `${formatShortDate(game.game_date)} ${game.home_away === 'road' ? '@' : ''}${game.opponent}`,
        moving_average: movingAverage,
        // Bars span league avg → game value, so length encodes the deviation, never a cropped axis minimum
        bar_range: [baseline, Number(game.value)],
      }
    })
  }, [data, selectedMovingAverage])

  // Evenly spaced x labels: every Nth game, N chosen so labels never crowd
  const xTickInterval = useMemo(() => {
    const maxLabels = isMobile ? 4 : 12
    return Math.max(0, Math.ceil(chartData.length / maxLabels) - 1)
  }, [chartData.length, isMobile])

  // Y-axis: nice ticks covering every bar plus both reference lines
  const yAxisConfig = useMemo(() => {
    if (!data?.data || data.data.length === 0) return { ticks: [], domain: [0, 100] }
    const values = [
      ...data.data.map((d) => Number(d.value ?? 0)),
      Number(data.league_average),
      Number(data.season_average),
    ].filter(Number.isFinite)
    const ticks = niceTicks(Math.min(...values), Math.max(...values), isMobile ? 5 : 8)
    return { ticks, domain: [ticks[0], ticks[ticks.length - 1]] }
  }, [data, isMobile])

  // Table data sorted by most recent first
  const tableData = useMemo(() => {
    if (!chartData.length) return []
    return [...chartData].reverse()
  }, [chartData])

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null
    const entry = payload[0]?.payload
    if (!entry) return null

    return (
      <div className="st-tooltip">
        <p className="tooltip-date">{entry.game_date}</p>
        <p className="tooltip-opponent">
          {entry.home_away === 'home' ? 'vs' : '@'} {entry.opponent}
          {' · '}
          <span className={entry.wl === 'W' ? 'text-positive' : 'text-negative'}>
            {entry.wl === 'W' ? 'Win' : 'Loss'}
          </span>
        </p>
        <dl className="tooltip-stats">
          <dt>Value</dt>
          <dd>{entry.value?.toFixed(1)}</dd>
          <dt>{selectedMovingAverage}-game avg</dt>
          <dd>{entry.moving_average?.toFixed(1)}</dd>
        </dl>
        <p className="tooltip-hint">Click the bar or the table date to open Game Analysis</p>
      </div>
    )
  }

  const vsLeague = data ? (data.season_average - data.league_average).toFixed(1) : 0
  const vsLeagueFormatted = data ? (vsLeague >= 0 ? `+${vsLeague}` : vsLeague) : '-'

  // Determine if team is performing better than league average
  // For lower-is-better stats, being below league avg is good
  const isLowerBetter = LOWER_IS_BETTER_STATS.has(data?.stat || selectedStat)
  const isBetterThanLeague = data
    ? isLowerBetter
      ? data.season_average < data.league_average
      : data.season_average > data.league_average
    : false

  // One Game Analysis destination for a game, shared by the chart bars and the table date links
  const buildGameTarget = (gameEntry) => {
    const gameId = normalizeGameId(gameEntry?.game_id)
    if (!selectedSeason || !gameId) return null
    const nextParams = new URLSearchParams({
      season: selectedSeason,
      game: gameId,
      scope: 'all',
      factors: 'eight_factors',
      view: 'factor',
    })
    return {
      to: { pathname: '/four-factor', search: `?${nextParams.toString()}` },
      state: {
        season: selectedSeason,
        gameId,
        gameDate: gameEntry?.game_date || '',
        team: selectedTeam,
        opponent: gameEntry?.opponent || '',
        homeAway: gameEntry?.home_away || '',
        source: 'trends',
      },
    }
  }

  const handleBarClick = (gameEntry) => {
    const target = buildGameTarget(gameEntry)
    if (target) navigate(target.to, { state: target.state })
  }

  const isBusy = loading || initializing
  const statLabel = STAT_OPTIONS.find((s) => s.value === data?.stat)?.label || data?.stat_label || ''
  const chartHeight = isMobile ? 300 : 420

  return (
    <div className="st-page">
      <div className="page-header">
        <div className="page-title-row">
          <h1 className="page-title">Statistical Trends</h1>
          <p className="page-description">
            Plot the time series of team statistics over the season
          </p>
        </div>
      </div>

      <div className="st-controls card">
        <div className="form-row">
          <div className="form-group">
            <label className="form-label" htmlFor="st-season">Season</label>
            <select
              id="st-season"
              className="form-select"
              value={selectedSeason}
              onChange={(e) => handleSeasonChange(e.target.value)}
            >
              {seasons.length === 0 && <option value="">Loading seasons…</option>}
              {seasons.map((season) => (
                <option key={season} value={season}>{season}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="st-team">Team</label>
            <select
              id="st-team"
              className="form-select"
              value={selectedTeam}
              onChange={(e) => handleTeamChange(e.target.value)}
              disabled={!selectedSeason || teams.length === 0}
            >
              {teams.length === 0 && <option value="">Loading teams…</option>}
              {teams.map((team) => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="st-scope">Data Scope</label>
            <select
              id="st-scope"
              className="form-select"
              value={selectedDataScope}
              onChange={(e) => handleDataScopeChange(e.target.value)}
            >
              {DATA_SCOPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="st-stat">Statistic</label>
            <select
              id="st-stat"
              className="form-select"
              value={selectedStat}
              onChange={(e) => handleStatChange(e.target.value)}
            >
              {STAT_OPTIONS.map((stat) => (
                <option key={stat.value} value={stat.value}>{stat.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="st-ma">Moving Average</label>
            <select
              id="st-ma"
              className="form-select"
              value={selectedMovingAverage}
              onChange={(e) => handleMovingAverageChange(e.target.value)}
            >
              {MOVING_AVERAGE_OPTIONS.map((windowSize) => (
                <option key={windowSize} value={windowSize}>{windowSize} games</option>
              ))}
            </select>
          </div>
        </div>
        <div className="st-status" role="status" aria-live="polite">
          {isBusy && data && (
            <span className="refresh-indicator">
              <span className="loading-spinner" aria-hidden="true"></span>
              Updating…
            </span>
          )}
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {isBusy && !data && (
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading trend data...
        </div>
      )}

      {data && (
        <div className={`st-results${isBusy ? ' is-refreshing' : ''}`} aria-busy={isBusy}>
          <div className="st-summary card">
            <div className="st-stat">
              <span className="st-stat-label">Games played</span>
              <span className="st-stat-value">{data.data.length}</span>
            </div>
            <div className="st-stat">
              <span className="st-stat-label">Team average</span>
              <span className="st-stat-value">{data.season_average.toFixed(1)}</span>
            </div>
            <div className="st-stat">
              <span className="st-stat-label">League average</span>
              <span className="st-stat-value">{data.league_average.toFixed(1)}</span>
            </div>
            <div className="st-stat">
              <span className="st-stat-label">vs League</span>
              <span className={`st-stat-value ${isBetterThanLeague ? 'text-positive' : 'text-negative'}`}>
                {vsLeagueFormatted}
              </span>
            </div>
          </div>

          <div className="st-chart-card card">
            <h2 className="card-title">{getTeamName(data.team)} {statLabel}</h2>
            <ul className="st-legend" aria-label="Chart legend">
              <li className="st-legend-item">
                <span className="st-swatch st-swatch--win" aria-hidden="true"></span> Win
              </li>
              <li className="st-legend-item">
                <span className="st-swatch st-swatch--loss" aria-hidden="true"></span> Loss
              </li>
              <li className="st-legend-item">
                <span className="st-legend-line st-legend-line--moving" aria-hidden="true"></span> {selectedMovingAverage}-game avg
              </li>
              <li className="st-legend-item">
                <span className="st-legend-line st-legend-line--team" aria-hidden="true"></span> Team avg
              </li>
              <li className="st-legend-item">
                <span className="st-legend-line st-legend-line--league" aria-hidden="true"></span> League avg (bar baseline)
              </li>
            </ul>
            <div className="st-chart" style={{ height: chartHeight }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={chartData}
                  barCategoryGap="20%"
                  margin={isMobile ? { top: 8, right: 16, left: 0, bottom: 0 } : { top: 8, right: 28, left: 0, bottom: 0 }}
                >
                  <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
                  <XAxis
                    dataKey="xLabel"
                    tick={<GameTick />}
                    tickLine={false}
                    axisLine={{ stroke: 'var(--color-border-strong)' }}
                    height={34}
                    interval={xTickInterval}
                    fontSize={11}
                  />
                  <YAxis
                    tick={{ fill: 'var(--chart-axis)', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={isMobile ? 32 : 44}
                    domain={yAxisConfig.domain}
                    ticks={yAxisConfig.ticks}
                    allowDataOverflow
                  />
                  <Tooltip
                    content={<CustomTooltip />}
                    cursor={{ fill: 'rgba(20, 22, 26, 0.04)' }}
                    allowEscapeViewBox={{ x: false, y: false }}
                    wrapperStyle={TOOLTIP_WRAPPER_STYLE}
                  />
                  <ReferenceLine
                    y={data.league_average}
                    stroke="var(--chart-reference)"
                    strokeWidth={1.5}
                  />
                  <ReferenceLine
                    y={data.season_average}
                    stroke={TEAM_AVERAGE_STROKE}
                    strokeWidth={1.5}
                    strokeDasharray="6 3"
                  />
                  <Bar
                    dataKey="bar_range"
                    name="Value"
                    maxBarSize={isMobile ? 24 : 40}
                    cursor="pointer"
                    isAnimationActive={false}
                  >
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`game-bar-${entry.game_id ?? index}`}
                        fill={entry.wl === 'W' ? WIN_BAR_FILL : LOSS_BAR_FILL}
                        onClick={() => handleBarClick(entry)}
                      />
                    ))}
                  </Bar>
                  <Line
                    type="linear"
                    dataKey="moving_average"
                    name={`${selectedMovingAverage}-Game Avg`}
                    stroke="var(--chart-average)"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 3 }}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="st-games card">
            <h2 className="card-title">Game-by-Game Data</h2>
            <div className="st-games-scroll table-scroll" tabIndex={0} aria-label="Game-by-game data table">
              <table>
                <thead>
                  <tr>
                    <th className="st-col-index">#</th>
                    <th>Date</th>
                    <th>Opponent</th>
                    <th>Result</th>
                    <th className="text-right">Value</th>
                    <th className="text-right">{selectedMovingAverage}-Game Avg</th>
                  </tr>
                </thead>
                <tbody>
                  {tableData.map((game, index) => (
                    <tr key={game.game_id}>
                      <td className="st-col-index">{tableData.length - index}</td>
                      <td className="st-nowrap">
                        <GameDateLink target={buildGameTarget(game)} game={game} />
                      </td>
                      <td className="st-nowrap">
                        {game.home_away === 'home' ? 'vs' : '@'} {game.opponent}
                      </td>
                      <td className={game.wl === 'W' ? 'text-positive' : 'text-negative'}>
                        {game.wl}
                      </td>
                      <td className="text-right">{game.value.toFixed(1)}</td>
                      <td className="text-right">{game.moving_average.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="st-glossary card">
            <button
              type="button"
              className="st-glossary-toggle"
              onClick={() => setGlossaryExpanded(!glossaryExpanded)}
              aria-expanded={glossaryExpanded}
              aria-controls="st-glossary-content"
            >
              <span>Glossary</span>
              <span className="st-toggle-icon" aria-hidden="true">{glossaryExpanded ? '−' : '+'}</span>
            </button>

            {glossaryExpanded && (
              <div className="st-glossary-content" id="st-glossary-content">
                <dl className="st-glossary-list">
                  {GLOSSARY_ITEMS.map(item => (
                    <div key={item.term} className="st-glossary-item">
                      <dt>{item.term}</dt>
                      <dd>{item.definition}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default Trends
