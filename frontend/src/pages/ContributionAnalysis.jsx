import { useState, useEffect, useMemo, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import { usePersistedState } from '../hooks/usePersistedState'
import {
  BarChart,
  Bar,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  LabelList,
} from 'recharts'
import { getSeasons, getTeams, getLeagueSummary, getContributionAnalysis } from '../api'
import { getTeamName } from '../constants/teams'
import useMediaQuery from '../hooks/useMediaQuery'
import { niceTicks, formatShortDate, TOOLTIP_WRAPPER_STYLE } from '../utils/chart'
import './ContributionAnalysis.css'

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
  { value: 'garbage_filtered', label: 'Excl. garbage time' },
]
const VALID_DATE_RANGE_VALUES = new Set(DATE_RANGE_OPTIONS.map((option) => option.value))
const VALID_DATA_SCOPE_VALUES = new Set(DATA_SCOPE_OPTIONS.map((option) => option.value))

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
}

const GLOSSARY_ITEMS = [
  { term: 'Net Rating', definition: 'Offensive Rating minus Defensive Rating. Measures overall point differential per 100 possessions.' },
  { term: 'Contribution', definition: 'How much each factor contributed to the team\'s net rating relative to league average. Positive = above average performance in that area.' },
  { term: 'eFG%', definition: 'Effective Field Goal Percentage - Adjusts FG% for the added value of 3-pointers. Formula: (FGM + 0.5 x 3PM) / FGA x 100' },
  { term: 'BH (Ball Handling)', definition: 'Measures ability to take care of the ball. Calculated as 100 - TOV%. Higher is better.' },
  { term: 'OREB%', definition: 'Offensive Rebounding Percentage - Percentage of available offensive rebounds grabbed.' },
  { term: 'FT Rate', definition: 'Free Throw Rate - Free throws made per field goal attempt (FTM / FGA x 100).' },
  { term: 'Opp Factors', definition: 'Opponent statistics - What your opponents do against you. Better defense = lower opponent values.' },
]

const FACTOR_ORDER = [
  'shooting',
  'ball_handling',
  'orebounding',
  'free_throws',
  'opp_shooting',
  'opp_ball_handling',
  'opp_orebounding',
  'opp_free_throws',
]

const FACTOR_LABELS = {
  shooting: 'Shooting',
  ball_handling: 'Ball Handling',
  orebounding: 'Off Rebounding',
  free_throws: 'Free Throws',
  opp_shooting: 'Opp Shooting',
  opp_ball_handling: 'Opp Ball Handling',
  opp_orebounding: 'Opp Off Rebounding',
  opp_free_throws: 'Opp Free Throws',
}

const DECOMP_GAP_ROW = '__gap__'
const DECOMP_TOTAL_ROW = 'Net rating (actual)'
const DECOMP_ERROR_FILL = 'var(--chart-neutral)'
const DECOMP_TOOLTIP_MAX_WIDTH = 260
const DECOMP_MARGIN = { top: 4, right: 12, left: 0, bottom: 4 }
const DECOMP_LABEL_GAP = 6
const TEAM_AVERAGE_STROKE = 'var(--color-accent)'

// Opponent factors: a lower opponent value helps the team (same direction as the model's contributions)
const LOWER_IS_BETTER_FACTORS = new Set([
  'opp_shooting',
  'opp_ball_handling',
  'opp_orebounding',
  'opp_free_throws',
])
const MOBILE_QUERY = '(max-width: 768px)'

function formatSigned(value, digits) {
  const n = Number(value) || 0
  return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}`
}

// Baseball-style win percentage: ".750", undefeated "1.000"
function formatWinPct(value) {
  const n = Number(value)
  if (!Number.isFinite(n)) return ''
  const text = n.toFixed(3)
  return text.startsWith('0') ? text.slice(1) : text
}

// Approximate rendered width of a 12px tabular value label such as "+22.50"
function estimateLabelWidth(text) {
  return String(text).length * 7 + DECOMP_LABEL_GAP + 2
}

/*
 * Domain for a horizontal bar chart with end-of-bar labels: grows [lo, hi] until every
 * label fits inside the plot (right of positive bars, left of negative bars).
 * plotWidth is the measured pixel width of the plotting area.
 */
function fitLabelDomain(values, plotWidth) {
  let lo = Math.min(0, ...values)
  let hi = Math.max(0, ...values)
  if (lo === hi) hi = lo + 1
  if (!(plotWidth > 0)) {
    const pad = (hi - lo) * 0.3
    return [lo < 0 ? lo - pad : lo, hi > 0 ? hi + pad : hi]
  }
  for (let i = 0; i < 8; i += 1) {
    const unitsPerPx = (hi - lo) / plotWidth
    let nextLo = lo
    let nextHi = hi
    values.forEach((v) => {
      const need = estimateLabelWidth(formatSigned(v, 2)) * unitsPerPx
      if (v >= 0) nextHi = Math.max(nextHi, v + need)
      else nextLo = Math.min(nextLo, v - need)
    })
    if (nextLo === lo && nextHi === hi) break
    lo = nextLo
    hi = nextHi
  }
  return [lo, hi]
}

// Tracks an element's content width (0 until measured)
function useElementWidth(ref, active) {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return undefined
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref, active])
  return width
}

function normalizeQueryToken(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
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

function normalizeDateInputValue(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value || '') ? value : ''
}

function ContributionAnalysis() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [seasons, setSeasons] = useState([])
  const [teams, setTeams] = useState([])
  const [selectedSeason, setSelectedSeason] = usePersistedState('contribution_season', '')
  const [selectedDataScope, setSelectedDataScope] = usePersistedState('contribution_datascope', 'all')
  const [selectedTeam, setSelectedTeam] = usePersistedState('contribution_team', 'BOS', { persist: true })
  const [dateRangeType, setDateRangeType] = usePersistedState('contribution_daterange', 'season')
  const [customStartDate, setCustomStartDate] = useState('')
  const [customEndDate, setCustomEndDate] = useState('')
  const [customDatesInitialized, setCustomDatesInitialized] = useState(false)
  const [seasonBounds, setSeasonBounds] = useState({ first: '', last: '' })
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [error, setError] = useState(null)
  const [glossaryExpanded, setGlossaryExpanded] = useState(false)
  const hasUserInteractedRef = useRef(false)
  const isMobile = useMediaQuery(MOBILE_QUERY)
  const decompChartRef = useRef(null)
  const urlSelections = useMemo(() => {
    const season = readSearchParam(searchParams, 'season').trim()
    const team = readSearchParam(searchParams, 'team').trim().toUpperCase()
    return {
      season,
      team,
      dataScope: normalizeDataScopeParam(readSearchParam(searchParams, 'scope', 'data_scope', 'dataScope')),
      dateRange: normalizeDateRangeParam(readSearchParam(searchParams, 'range', 'date_range', 'dateRange')),
      customStartDate: normalizeDateInputValue(readSearchParam(searchParams, 'start', 'start_date', 'startDate')),
      customEndDate: normalizeDateInputValue(readSearchParam(searchParams, 'end', 'end_date', 'endDate')),
    }
  }, [searchParams])

  const buildSearchParamsFromState = (overrides = {}) => {
    const nextState = {
      season: selectedSeason,
      team: selectedTeam,
      dataScope: selectedDataScope,
      dateRange: dateRangeType,
      customStartDate,
      customEndDate,
      ...overrides,
    }

    const nextParams = new URLSearchParams()
    if (nextState.season) nextParams.set('season', nextState.season)
    if (nextState.team) nextParams.set('team', nextState.team)
    if (nextState.dataScope) nextParams.set('scope', nextState.dataScope)
    if (nextState.dateRange) nextParams.set('range', nextState.dateRange)
    if (nextState.dateRange === 'custom') {
      if (nextState.customStartDate) nextParams.set('start', nextState.customStartDate)
      if (nextState.customEndDate) nextParams.set('end', nextState.customEndDate)
    }
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

  const handleDateRangeChange = (value) => {
    updateUrlFromUserChange({ dateRange: value })
    setDateRangeType(value)
  }

  const handleCustomStartDateChange = (value) => {
    updateUrlFromUserChange({ dateRange: 'custom', customStartDate: value })
    setCustomStartDate(value)
  }

  const handleCustomEndDateChange = (value) => {
    updateUrlFromUserChange({ dateRange: 'custom', customEndDate: value })
    setCustomEndDate(value)
  }

  // Load seasons on mount
  useEffect(() => {
    async function loadInitialData() {
      try {
        const seasonsRes = await getSeasons()
        setSeasons(seasonsRes.seasons)

        // Default to the newest season that has games; a just-started season has none yet,
        // so fall back to the previous one.
        let defaultSeason = seasonsRes.seasons[0] || ''
        for (const season of seasonsRes.seasons.slice(0, 2)) {
          try {
            const teamsRes = await getTeams(season)
            if (teamsRes.teams?.length) {
              defaultSeason = season
              break
            }
          } catch {
            // Try the next season.
          }
        }

        setSelectedSeason((prevSeason) => {
          if (urlSelections.season && seasonsRes.seasons.includes(urlSelections.season)) {
            return urlSelections.season
          }
          if (prevSeason && seasonsRes.seasons.includes(prevSeason)) return prevSeason
          return defaultSeason
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

  // Keep persisted date preset valid when option list changes
  useEffect(() => {
    if (!DATE_RANGE_OPTIONS.some(opt => opt.value === dateRangeType)) {
      setDateRangeType('season')
    }
  }, [dateRangeType, setDateRangeType])

  useEffect(() => {
    if (!DATA_SCOPE_OPTIONS.some((opt) => opt.value === selectedDataScope)) {
      setSelectedDataScope('all')
    }
  }, [selectedDataScope, setSelectedDataScope])

  useEffect(() => {
    if (urlSelections.dataScope && urlSelections.dataScope !== selectedDataScope) {
      setSelectedDataScope(urlSelections.dataScope)
    }
  }, [selectedDataScope, setSelectedDataScope, urlSelections.dataScope])

  useEffect(() => {
    if (urlSelections.dateRange && urlSelections.dateRange !== dateRangeType) {
      setDateRangeType(urlSelections.dateRange)
    }
  }, [dateRangeType, setDateRangeType, urlSelections.dateRange])

  useEffect(() => {
    if (urlSelections.dateRange !== 'custom') return
    if (urlSelections.customStartDate && urlSelections.customStartDate !== customStartDate) {
      setCustomStartDate(urlSelections.customStartDate)
    }
    if (urlSelections.customEndDate && urlSelections.customEndDate !== customEndDate) {
      setCustomEndDate(urlSelections.customEndDate)
    }
  }, [customEndDate, customStartDate, urlSelections.customEndDate, urlSelections.customStartDate, urlSelections.dateRange])

  // Load teams and season bounds when season changes
  useEffect(() => {
    let isCurrent = true
    async function loadTeamsAndBounds() {
      if (!selectedSeason) return
      // Keep the previous analysis on screen (dimmed via .is-refreshing) while the
      // new season context loads; it is replaced when the new analysis arrives.
      setInitializing(true)
      try {
        const [teamsRes, summaryRes] = await Promise.all([
          getTeams(selectedSeason, selectedDataScope),
          getLeagueSummary(selectedSeason, null, null, false, null, selectedDataScope),
        ])
        if (!isCurrent) return

        setTeams(teamsRes.teams)
        if (teamsRes.teams.length === 0) setData(null)
        // Keep persisted team if it exists in this season, otherwise default to BOS or first team
        setSelectedTeam(prevTeam => {
          if (urlSelections.team && teamsRes.teams.includes(urlSelections.team)) {
            return urlSelections.team
          }
          if (teamsRes.teams.includes(prevTeam)) return prevTeam
          if (teamsRes.teams.includes('BOS')) return 'BOS'
          if (teamsRes.teams.length > 0) return teamsRes.teams[0]
          return ''
        })

        // Extract season date bounds from league summary (actual data, not scheduled games)
        if (summaryRes.first_game_date && summaryRes.last_game_date) {
          setSeasonBounds({ first: summaryRes.first_game_date, last: summaryRes.last_game_date })
        }
      } catch (err) {
        if (isCurrent) {
          setError(err.message)
          setData(null)
        }
      } finally {
        if (isCurrent) setInitializing(false)
      }
    }
    loadTeamsAndBounds()
    return () => { isCurrent = false }
  }, [selectedSeason, selectedDataScope, setSelectedTeam, urlSelections.team])

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
    customEndDate,
    customStartDate,
    dateRangeType,
    searchParams,
    selectedDataScope,
    selectedSeason,
    selectedTeam,
    setSearchParams,
  ])

  // Populate custom date fields when switching to custom and bounds are available
  useEffect(() => {
    if (dateRangeType === 'custom' && seasonBounds.first && seasonBounds.last && !customDatesInitialized) {
      setCustomStartDate(seasonBounds.first)
      setCustomEndDate(seasonBounds.last)
      setCustomDatesInitialized(true)
    }
    // Reset initialization flag when switching away from custom
    if (dateRangeType !== 'custom') {
      setCustomDatesInitialized(false)
    }
  }, [dateRangeType, seasonBounds.first, seasonBounds.last, customDatesInitialized])

  const formatUtcDate = (dateObj) => {
    const y = dateObj.getUTCFullYear()
    const m = String(dateObj.getUTCMonth() + 1).padStart(2, '0')
    const d = String(dateObj.getUTCDate()).padStart(2, '0')
    return `${y}-${m}-${d}`
  }

  // Calculate API parameters from preset (aligned with LeagueSummary)
  const { apiRangeType, apiStartDate, apiEndDate, apiLastNGames, apiExcludePlayoffs } = useMemo(() => {
    const hasBounds = Boolean(seasonBounds.first && seasonBounds.last)

    if (!hasBounds) {
      if (dateRangeType === 'last_10_games') {
        return { apiRangeType: 'last_n', apiStartDate: null, apiEndDate: null, apiLastNGames: 10, apiExcludePlayoffs: false }
      }
      if (dateRangeType === 'last_15_games') {
        return { apiRangeType: 'last_n', apiStartDate: null, apiEndDate: null, apiLastNGames: 15, apiExcludePlayoffs: false }
      }
      if (dateRangeType === 'last_20_games') {
        return { apiRangeType: 'last_n', apiStartDate: null, apiEndDate: null, apiLastNGames: 20, apiExcludePlayoffs: false }
      }
      if (dateRangeType === 'custom') {
        return { apiRangeType: 'custom', apiStartDate: '', apiEndDate: '', apiLastNGames: null, apiExcludePlayoffs: false }
      }
      return {
        apiRangeType: 'season',
        apiStartDate: null,
        apiEndDate: null,
        apiLastNGames: null,
        apiExcludePlayoffs: dateRangeType === 'season_regular',
      }
    }

    const seasonStart = seasonBounds.first
    const seasonEnd = seasonBounds.last

    const getMonthStart = (anchorDateStr, monthsBack) => {
      const [year, month] = anchorDateStr.split('-').map(Number)
      const monthStart = new Date(Date.UTC(year, month - 1 - monthsBack, 1))
      return formatUtcDate(monthStart)
    }

    switch (dateRangeType) {
      case 'season':
        return {
          apiRangeType: 'season',
          apiStartDate: null,
          apiEndDate: null,
          apiLastNGames: null,
          apiExcludePlayoffs: false,
        }

      case 'season_regular':
        return {
          apiRangeType: 'season',
          apiStartDate: null,
          apiEndDate: null,
          apiLastNGames: null,
          apiExcludePlayoffs: true,
        }

      case 'this_month': {
        const monthStart = getMonthStart(seasonEnd, 0)
        return {
          apiRangeType: 'custom',
          apiStartDate: monthStart < seasonStart ? seasonStart : monthStart,
          apiEndDate: seasonEnd,
          apiLastNGames: null,
          apiExcludePlayoffs: false,
        }
      }

      case 'last_2_months': {
        const start = getMonthStart(seasonEnd, 1)
        return {
          apiRangeType: 'custom',
          apiStartDate: start < seasonStart ? seasonStart : start,
          apiEndDate: seasonEnd,
          apiLastNGames: null,
          apiExcludePlayoffs: false,
        }
      }

      case 'last_3_months': {
        const start = getMonthStart(seasonEnd, 2)
        return {
          apiRangeType: 'custom',
          apiStartDate: start < seasonStart ? seasonStart : start,
          apiEndDate: seasonEnd,
          apiLastNGames: null,
          apiExcludePlayoffs: false,
        }
      }

      case 'last_10_games':
        return {
          apiRangeType: 'last_n',
          apiStartDate: null,
          apiEndDate: null,
          apiLastNGames: 10,
          apiExcludePlayoffs: false,
        }

      case 'last_15_games':
        return {
          apiRangeType: 'last_n',
          apiStartDate: null,
          apiEndDate: null,
          apiLastNGames: 15,
          apiExcludePlayoffs: false,
        }

      case 'last_20_games':
        return {
          apiRangeType: 'last_n',
          apiStartDate: null,
          apiEndDate: null,
          apiLastNGames: 20,
          apiExcludePlayoffs: false,
        }

      case 'custom': {
        const safeStart = customStartDate || seasonStart
        const safeEnd = customEndDate || seasonEnd
        const clampedStart = safeStart < seasonStart ? seasonStart : safeStart
        const clampedEnd = safeEnd > seasonEnd ? seasonEnd : safeEnd
        if (clampedStart > clampedEnd) {
          return {
            apiRangeType: 'custom',
            apiStartDate: seasonStart,
            apiEndDate: seasonEnd,
            apiLastNGames: null,
            apiExcludePlayoffs: false,
          }
        }
        return {
          apiRangeType: 'custom',
          apiStartDate: clampedStart,
          apiEndDate: clampedEnd,
          apiLastNGames: null,
          apiExcludePlayoffs: false,
        }
      }

      default:
        return {
          apiRangeType: 'custom',
          apiStartDate: seasonStart,
          apiEndDate: seasonEnd,
          apiLastNGames: null,
          apiExcludePlayoffs: false,
        }
    }
  }, [dateRangeType, customStartDate, customEndDate, seasonBounds.first, seasonBounds.last])

  // Load contribution analysis when parameters change
  useEffect(() => {
    let isCurrent = true
    async function loadAnalysis() {
      if (!selectedSeason || !selectedTeam) return

      const hasValidRange =
        apiRangeType === 'season' ||
        (apiRangeType === 'last_n' && Boolean(apiLastNGames)) ||
        (apiRangeType === 'custom' && Boolean(apiStartDate) && Boolean(apiEndDate))

      if (!hasValidRange) return

      setLoading(true)
      if (isCurrent) setError(null)

      try {
        const res = await getContributionAnalysis(
          selectedSeason,
          selectedTeam,
          apiRangeType,
          apiLastNGames,
          apiStartDate,
          apiEndDate,
          apiExcludePlayoffs,
          selectedDataScope
        )
        if (isCurrent) setData(res)
      } catch (err) {
        if (isCurrent) setError(err.message)
      } finally {
        if (isCurrent) setLoading(false)
      }
    }
    loadAnalysis()
    return () => { isCurrent = false }
  }, [selectedSeason, selectedDataScope, selectedTeam, dateRangeType, customStartDate, customEndDate, apiRangeType, apiStartDate, apiEndDate, apiLastNGames, apiExcludePlayoffs])

  // Decomposition rows: the 8 factor contributions (most positive first) + Error, which
  // together add up to the actual net rating; then a divider row and the actual total.
  const mainChartData = useMemo(() => {
    if (!data) return []

    const contributionBars = FACTOR_ORDER.map((factor) => {
      const contribution = data.contributions[factor] || 0
      return {
        name: FACTOR_LABELS[factor] || factor,
        factor,
        value: contribution,
        actualValue: data.factor_values?.[factor],
        leagueAvg: data.league_averages?.[factor],
        fill: contribution >= 0 ? 'var(--chart-positive)' : 'var(--chart-negative)',
      }
    }).sort((a, b) => b.value - a.value)

    return [
      ...contributionBars,
      {
        name: 'Error',
        value: data.net_rating - data.predicted_net_rating,
        fill: DECOMP_ERROR_FILL,
        isError: true,
      },
      { name: DECOMP_GAP_ROW, value: null, isGap: true },
      { name: DECOMP_TOTAL_ROW, value: data.net_rating, fill: 'var(--color-ink)', isNetRating: true },
    ]
  }, [data])

  const decompWidth = useElementWidth(decompChartRef, Boolean(data))
  const decompYAxisWidth = isMobile ? 132 : 148
  // The plot (right of the factor labels) is too narrow on phones for Recharts' own
  // placement, so pin the tooltip's left edge such that it always ends inside the chart.
  const decompTooltipMaxWidth = Math.min(DECOMP_TOOLTIP_MAX_WIDTH, Math.max(decompWidth - 8, 160))
  const decompTooltipPosition = decompWidth > 0 && decompWidth < 560
    ? { x: Math.max(0, decompWidth - decompTooltipMaxWidth) }
    : undefined

  // X-axis: domain sized from the measured plot width so no end-of-bar label clips
  const mainChartXAxis = useMemo(() => {
    const values = mainChartData.filter((d) => !d.isGap).map((d) => d.value)
    if (values.length === 0) return { domain: [-5, 5], ticks: [-5, 0, 5] }
    const plotWidth = decompWidth - decompYAxisWidth - DECOMP_MARGIN.left - DECOMP_MARGIN.right
    const [lo, hi] = fitLabelDomain(values, plotWidth)
    const ticks = niceTicks(lo, hi, isMobile ? 4 : 6).filter((t) => t >= lo - 1e-9 && t <= hi + 1e-9)
    return { domain: [lo, hi], ticks }
  }, [mainChartData, decompWidth, decompYAxisWidth, isMobile])

  const isBusy = loading || initializing
  const decompRowHeight = isMobile ? 30 : 32
  const decompHeight = mainChartData.length * decompRowHeight + 36
  const predictionError = data ? data.net_rating - data.predicted_net_rating : 0

  return (
    <div className="ca-page">
      <div className="page-header">
        <div className="page-title-row">
          <h1 className="page-title">Contribution Analysis</h1>
          <p className="page-description">
            Analyze how each of the eight factors contributed to a team's net rating over a period
          </p>
        </div>
      </div>

      <div className="ca-controls card">
        <div className="form-row">
          <div className="form-group">
            <label className="form-label" htmlFor="ca-season">Season</label>
            <select
              id="ca-season"
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
            <label className="form-label" htmlFor="ca-team">Team</label>
            <select
              id="ca-team"
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
            <label className="form-label" htmlFor="ca-scope">Data Scope</label>
            <select
              id="ca-scope"
              className="form-select"
              value={selectedDataScope}
              onChange={(e) => handleDataScopeChange(e.target.value)}
            >
              {DATA_SCOPE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="ca-range">Date Range</label>
            <select
              id="ca-range"
              className="form-select"
              value={dateRangeType}
              onChange={(e) => handleDateRangeChange(e.target.value)}
            >
              {DATE_RANGE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>

        {dateRangeType === 'custom' && (
          <div className="form-row ca-custom-dates">
            <div className="form-group">
              <label className="form-label" htmlFor="ca-start-date">Start Date</label>
              <input
                id="ca-start-date"
                type="date"
                className="form-input"
                value={customStartDate}
                onChange={(e) => handleCustomStartDateChange(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="ca-end-date">End Date</label>
              <input
                id="ca-end-date"
                type="date"
                className="form-input"
                value={customEndDate}
                onChange={(e) => handleCustomEndDateChange(e.target.value)}
              />
            </div>
          </div>
        )}

        <div className="ca-status" role="status" aria-live="polite">
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
          Loading analysis...
        </div>
      )}

      {data && (
        <div className={`ca-results${isBusy ? ' is-refreshing' : ''}`} aria-busy={isBusy}>
          <div className="ca-summary card">
            <div className="ca-summary-team">
              <div className="ca-team-name">{getTeamName(data.team)}</div>
              <div className="ca-team-record">
                <span>{data.wins}-{data.losses}</span>
                <span className="ca-team-pct">{formatWinPct(data.win_pct)}</span>
              </div>
            </div>
            <dl className="ca-summary-range">
              <div>
                <dt>Range</dt>
                <dd>{data.date_range_label}</dd>
              </div>
              <div>
                <dt>Games played</dt>
                <dd>
                  {data.games_analyzed} {data.games_analyzed === 1 ? 'game' : 'games'}:{' '}
                  <span className="ca-nowrap">{data.start_date}</span> – <span className="ca-nowrap">{data.end_date}</span>
                </dd>
              </div>
            </dl>
            <div className="ca-summary-stats">
              <div className="ca-stat">
                <span className="ca-stat-label">Actual net rating</span>
                <span className={`ca-stat-value ${data.net_rating >= 0 ? 'text-positive' : 'text-negative'}`}>
                  {formatSigned(data.net_rating, 1)}
                </span>
              </div>
              <div className="ca-stat">
                <span className="ca-stat-label">Predicted</span>
                <span className="ca-stat-value">{formatSigned(data.predicted_net_rating, 1)}</span>
              </div>
              <div className="ca-stat">
                <span className="ca-stat-label">Error</span>
                <span className="ca-stat-value">{formatSigned(predictionError, 1)}</span>
              </div>
            </div>
          </div>

          <div className="ca-decomp card">
            <div className="ca-chart-header">
              <h2 className="card-title">Net Rating Decomposition</h2>
              <ul className="ca-legend" aria-label="Chart legend">
                <li className="ca-legend-item">
                  <span className="ca-swatch ca-swatch--positive" aria-hidden="true"></span>
                  Helps {data.team}
                </li>
                <li className="ca-legend-item">
                  <span className="ca-swatch ca-swatch--negative" aria-hidden="true"></span>
                  Hurts {data.team}
                </li>
                <li className="ca-legend-item">
                  <span className="ca-swatch ca-swatch--error" aria-hidden="true"></span>
                  Error (actual − predicted)
                </li>
                <li className="ca-legend-item">
                  <span className="ca-swatch ca-swatch--total" aria-hidden="true"></span>
                  Net rating (actual)
                </li>
              </ul>
            </div>
            <div className="ca-decomp-chart" ref={decompChartRef} style={{ height: decompHeight }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={mainChartData}
                  layout="vertical"
                  margin={DECOMP_MARGIN}
                  barCategoryGap="22%"
                >
                  <CartesianGrid horizontal={false} stroke="var(--chart-grid)" />
                  <XAxis
                    type="number"
                    domain={mainChartXAxis.domain}
                    ticks={mainChartXAxis.ticks}
                    tick={{ fill: 'var(--chart-axis)', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    height={24}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={decompYAxisWidth}
                    tick={<DecompCategoryTick />}
                    tickLine={false}
                    axisLine={false}
                    interval={0}
                  />
                  <Tooltip
                    cursor={{ fill: 'rgba(20, 22, 26, 0.04)' }}
                    allowEscapeViewBox={{ x: false, y: false }}
                    wrapperStyle={{ ...TOOLTIP_WRAPPER_STYLE, maxWidth: decompTooltipMaxWidth }}
                    position={decompTooltipPosition}
                    content={({ active, payload }) => {
                      if (!active || !payload || !payload.length) return null
                      const d = payload[0].payload
                      if (d.isGap) return null
                      if (d.isError) {
                        return (
                          <div className="ca-tooltip">
                            <div className="ca-tooltip-title">Error</div>
                            <div className="ca-tooltip-sub">Actual − predicted (not explained by the factors)</div>
                            <div>
                              {formatSigned(data.net_rating, 2)} − {formatSigned(data.predicted_net_rating, 2)} = {formatSigned(d.value, 2)}
                            </div>
                          </div>
                        )
                      }
                      if (d.isNetRating) {
                        return (
                          <div className="ca-tooltip">
                            <div className="ca-tooltip-title">{DECOMP_TOTAL_ROW}</div>
                            <div className="ca-tooltip-sub">
                              Factors {formatSigned(data.predicted_net_rating, 2)} + error {formatSigned(data.net_rating - data.predicted_net_rating, 2)}
                            </div>
                            <div>{formatSigned(d.value, 2)}</div>
                          </div>
                        )
                      }
                      return (
                        <div className="ca-tooltip">
                          <div className="ca-tooltip-title">{d.name}</div>
                          {d.actualValue != null && (
                            <div className="ca-tooltip-sub">
                              {d.actualValue.toFixed(1)} (Lg Avg: {d.leagueAvg?.toFixed(1)})
                            </div>
                          )}
                          <div>
                            Contribution: {formatSigned(d.value, 2)}{' '}
                            <span className={d.value >= 0 ? 'text-positive' : 'text-negative'}>
                              ({d.value >= 0 ? 'helps' : 'hurts'} {data.team})
                            </span>
                          </div>
                        </div>
                      )
                    }}
                  />
                  <ReferenceLine x={0} stroke="var(--chart-axis)" />
                  <ReferenceLine y={DECOMP_GAP_ROW} stroke="var(--color-border-strong)" />
                  <Bar dataKey="value" isAnimationActive={false}>
                    {mainChartData.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="value" content={DecompValueLabel} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <section className="ca-top" aria-labelledby="ca-top-title">
            <div className="ca-top-header">
              <h2 className="section-title" id="ca-top-title">Top Contributing Factors</h2>
              <ul className="ca-legend" aria-label="Mini chart legend">
                <li className="ca-legend-item">
                  <span className="ca-swatch ca-swatch--positive" aria-hidden="true"></span>
                  Game helps {data.team}
                </li>
                <li className="ca-legend-item">
                  <span className="ca-swatch ca-swatch--negative" aria-hidden="true"></span>
                  Game hurts {data.team}
                </li>
                <li className="ca-legend-item">
                  <span className="ca-legend-line ca-legend-line--moving" aria-hidden="true"></span>
                  5-game avg
                </li>
                <li className="ca-legend-item">
                  <span className="ca-legend-line ca-legend-line--team" aria-hidden="true"></span>
                  Team avg
                </li>
                <li className="ca-legend-item">
                  <span className="ca-legend-line ca-legend-line--league" aria-hidden="true"></span>
                  League avg (bar baseline)
                </li>
              </ul>
            </div>
            <div className="ca-top-grid">
              {data.top_contributors.map((contributor, index) => (
                <MiniContributorChart
                  key={contributor.factor}
                  contributor={contributor}
                  index={index + 1}
                  team={data.team}
                />
              ))}
            </div>
          </section>

          <div className="ca-glossary card">
            <button
              type="button"
              className="ca-glossary-toggle"
              onClick={() => setGlossaryExpanded(!glossaryExpanded)}
              aria-expanded={glossaryExpanded}
              aria-controls="ca-glossary-content"
            >
              <span>Glossary</span>
              <span className="ca-toggle-icon" aria-hidden="true">{glossaryExpanded ? '−' : '+'}</span>
            </button>

            {glossaryExpanded && (
              <div className="ca-glossary-content" id="ca-glossary-content">
                <dl className="ca-glossary-list">
                  {GLOSSARY_ITEMS.map(item => (
                    <div key={item.term} className="ca-glossary-item">
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

function DecompCategoryTick({ x, y, payload }) {
  if (payload?.value === DECOMP_GAP_ROW) return null
  const isTotal = payload?.value === DECOMP_TOTAL_ROW
  return (
    <text
      x={x - 8}
      y={y}
      dy={4}
      textAnchor="end"
      fontSize={12}
      fontWeight={isTotal ? 600 : 400}
      fill="var(--color-text)"
    >
      {payload?.value}
    </text>
  )
}

// Value printed just past the end of each bar (left of negative bars, right of positive ones)
function DecompValueLabel({ x, y, width, height, value }) {
  if (value == null || !Number.isFinite(value)) return null
  const left = Math.min(x, x + width)
  const right = Math.max(x, x + width)
  const isNegative = value < 0
  return (
    <text
      x={isNegative ? left - 6 : right + 6}
      y={y + height / 2}
      dy={4}
      textAnchor={isNegative ? 'end' : 'start'}
      fontSize={12}
      fill="var(--color-text-secondary)"
      style={{ fontVariantNumeric: 'tabular-nums' }}
    >
      {formatSigned(value, 2)}
    </text>
  )
}

// First/last date ticks, anchored inward so neither is clipped at the plot edges
function EdgeDateTick({ x, y, payload, index }) {
  return (
    <text
      x={x}
      y={y}
      dy={12}
      textAnchor={index === 0 ? 'start' : 'end'}
      fontSize={11}
      fill="var(--chart-axis)"
    >
      {formatShortDate(payload?.value)}
    </text>
  )
}

function MiniContributorChart({ contributor, index, team }) {
  const leagueAvg = contributor.league_avg
  // +1 when a higher value helps the team, −1 for opponent factors (lower opp value helps)
  const direction = LOWER_IS_BETTER_FACTORS.has(contributor.factor) ? -1 : 1

  // Each game is a deviation bar from the league average: [baseline, value]
  const chartData = contributor.trend_data.map((point) => {
    const value = point.value == null ? null : Number(point.value)
    const deviation = value == null ? null : value - leagueAvg
    return {
      ...point,
      deviationRange: value == null ? null : [leagueAvg, value],
      deviation,
      helps: deviation != null && deviation * direction >= 0,
      tooltipLabel: `${point.game_date}: ${point.home_away === 'road' ? '@' : 'vs '}${point.opponent}`,
    }
  })

  const yAxisConfig = useMemo(() => {
    const values = [
      ...chartData.map((d) => d.value),
      ...chartData.map((d) => d.ma_5),
      leagueAvg,
      contributor.value,
    ].filter((v) => v != null && Number.isFinite(Number(v))).map(Number)
    if (values.length === 0) return { domain: [0, 100], ticks: [] }
    const ticks = niceTicks(Math.min(...values), Math.max(...values), 4, { integer: true })
    return { domain: [ticks[0], ticks[ticks.length - 1]], ticks }
  }, [chartData, leagueAvg, contributor.value])

  const xTicks = chartData.length > 1
    ? [chartData[0].game_date, chartData[chartData.length - 1].game_date]
    : chartData.map((d) => d.game_date)

  const isPositive = contributor.contribution >= 0

  // Tooltip: date, opponent, the game's value and its deviation from the league average
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length > 0) {
      const point = payload[0].payload
      return (
        <div className="ca-tooltip ca-tooltip--mini">
          <div>{point.tooltipLabel}</div>
          {point.value != null && (
            <div className="ca-tooltip-sub">
              {Number(point.value).toFixed(1)} vs Lg Avg {leagueAvg.toFixed(1)}{' '}
              <span className={point.helps ? 'text-positive' : 'text-negative'}>
                ({formatSigned(point.deviation, 1)}, {point.helps ? 'helps' : 'hurts'} {team})
              </span>
            </div>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <div className="ca-mini card">
      <div className="ca-mini-header">
        <span className="ca-mini-rank">#{index}</span>
        <div className="ca-mini-info">
          <h3 className="ca-mini-title">{contributor.factor_label}</h3>
          <div className="ca-mini-stats">
            <span>Value: {contributor.value.toFixed(1)}</span>
            <span>League Avg: {leagueAvg.toFixed(1)}</span>
            <span className={`ca-mini-contribution ${isPositive ? 'text-positive' : 'text-negative'}`}>
              Contribution: {isPositive ? '+' : ''}{contributor.contribution.toFixed(2)}
            </span>
          </div>
        </div>
      </div>
      <div className="ca-mini-chart">
        <ResponsiveContainer width="100%" height={170}>
          <ComposedChart
            data={chartData}
            margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
          >
            <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
            <XAxis
              dataKey="game_date"
              ticks={xTicks}
              interval={0}
              tick={<EdgeDateTick />}
              tickLine={false}
              axisLine={{ stroke: 'var(--color-border-strong)' }}
              height={20}
              padding={{ left: 8, right: 8 }}
            />
            <YAxis
              domain={yAxisConfig.domain}
              ticks={yAxisConfig.ticks}
              tick={{ fill: 'var(--chart-axis)', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={32}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: 'rgba(20, 22, 26, 0.04)' }}
              allowEscapeViewBox={{ x: false, y: false }}
              wrapperStyle={TOOLTIP_WRAPPER_STYLE}
            />
            <Bar dataKey="deviationRange" maxBarSize={10} isAnimationActive={false}>
              {chartData.map((point) => (
                <Cell
                  key={point.game_id ?? point.game_date}
                  fill={point.helps ? 'var(--chart-positive)' : 'var(--chart-negative)'}
                />
              ))}
            </Bar>
            <ReferenceLine
              y={leagueAvg}
              stroke="var(--chart-reference)"
              strokeWidth={1.5}
            />
            <ReferenceLine
              y={contributor.value}
              stroke={TEAM_AVERAGE_STROKE}
              strokeWidth={1.5}
              strokeDasharray="6 3"
            />
            <Line
              type="linear"
              dataKey="ma_5"
              stroke="var(--chart-average)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default ContributionAnalysis
