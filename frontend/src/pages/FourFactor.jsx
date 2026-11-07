import { useState, useEffect, useMemo, useRef } from 'react'
import { useLocation, useSearchParams } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'
import {
  getSeasons,
  getGames,
  getDecomposition,
  getInterpretation,
  getGameTimeline,
  getPbpTraditionalBoxScore,
  getInterpretationPrompt,
} from '../api'
import GameTimeline from '../components/GameTimeline'
import SegmentedControl from '../components/SegmentedControl'
import useMediaQuery from '../hooks/useMediaQuery'
import './FourFactor.css'

// BBRef uses different abbreviations for some teams
const BBREF_TEAM_MAP = {
  PHX: 'PHO',
  BKN: 'BRK',
  CHA: 'CHO',
}

const toBBRefTeam = (team) => BBREF_TEAM_MAP[team] || team

// Team abbreviation to city name mapping
const TEAM_CITIES = {
  ATL: 'Atlanta', BOS: 'Boston', BKN: 'Brooklyn', CHA: 'Charlotte',
  CHI: 'Chicago', CLE: 'Cleveland', DAL: 'Dallas', DEN: 'Denver',
  DET: 'Detroit', GSW: 'Golden State', HOU: 'Houston', IND: 'Indiana',
  LAC: 'LA Clippers', LAL: 'LA Lakers', MEM: 'Memphis', MIA: 'Miami',
  MIL: 'Milwaukee', MIN: 'Minnesota', NOP: 'New Orleans', NYK: 'New York',
  OKC: 'Oklahoma City', ORL: 'Orlando', PHI: 'Philadelphia', PHX: 'Phoenix',
  POR: 'Portland', SAC: 'Sacramento', SAS: 'San Antonio', TOR: 'Toronto',
  UTA: 'Utah', WAS: 'Washington',
  // Historical/alternate abbreviations
  NJN: 'New Jersey', SEA: 'Seattle', VAN: 'Vancouver', CHO: 'Charlotte',
  NOH: 'New Orleans', NOK: 'New Orleans/Oklahoma City',
}

const toCityName = (abbr) => TEAM_CITIES[abbr] || abbr

const TEAM_FULL_NAMES = {
  ATL: 'Atlanta Hawks', BOS: 'Boston Celtics', BKN: 'Brooklyn Nets', CHI: 'Chicago Bulls',
  CLE: 'Cleveland Cavaliers', DAL: 'Dallas Mavericks', DEN: 'Denver Nuggets', DET: 'Detroit Pistons',
  GSW: 'Golden State Warriors', HOU: 'Houston Rockets', IND: 'Indiana Pacers', LAC: 'LA Clippers',
  LAL: 'Los Angeles Lakers', MEM: 'Memphis Grizzlies', MIA: 'Miami Heat', MIL: 'Milwaukee Bucks',
  MIN: 'Minnesota Timberwolves', NOP: 'New Orleans Pelicans', NYK: 'New York Knicks',
  OKC: 'Oklahoma City Thunder', ORL: 'Orlando Magic', PHI: 'Philadelphia 76ers', PHX: 'Phoenix Suns',
  POR: 'Portland Trail Blazers', SAC: 'Sacramento Kings', SAS: 'San Antonio Spurs', TOR: 'Toronto Raptors',
  UTA: 'Utah Jazz', WAS: 'Washington Wizards',
  NJN: 'New Jersey Nets', SEA: 'Seattle SuperSonics', VAN: 'Vancouver Grizzlies', CHO: 'Charlotte Hornets',
  NOH: 'New Orleans Hornets', NOK: 'New Orleans/Oklahoma City Hornets',
}

// CHA was the Bobcats from 2004-05 through 2013-14 and the Hornets otherwise.
const toFullTeamName = (abbr, season) => {
  if (abbr === 'CHA') {
    const startYear = Number(String(season || '').slice(0, 4))
    return startYear >= 2004 && startYear <= 2013 ? 'Charlotte Bobcats' : 'Charlotte Hornets'
  }
  return TEAM_FULL_NAMES[abbr] || toCityName(abbr)
}

const INTERPRETATION_UI_ENABLED = true
const INTERPRETATION_ENABLED_SEASONS = new Set(['2025-26', '2026-27'])
const DATA_SCOPE_OPTIONS = [
  { value: 'all', label: 'Full game' },
  { value: 'garbage_filtered', label: 'Excl. garbage time' },
  { value: 'clutch', label: 'Clutch time' },
]
const VALID_DATA_SCOPE_OPTIONS = new Set(DATA_SCOPE_OPTIONS.map((option) => option.value))
const VALID_ANALYSIS_VIEWS = new Set(['boxscore', 'factor', 'timeline'])
const HIDDEN_SAVE_OPTION_MS = 8000
const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]
const ANALYSIS_VIEW_OPTIONS = [
  { value: 'boxscore', label: 'Box Score' },
  {
    value: 'factor',
    label: (
      <>
        <span className="ga-label-long">Factor Analysis</span>
        <span className="ga-label-short">Factors</span>
      </>
    ),
  },
  { value: 'timeline', label: 'Timeline' },
]
const SCOPE_SHORT_LABELS = {
  all: 'Full game',
  garbage_filtered: 'Excl. garbage',
  clutch: 'Clutch',
}
// Below this many minutes a scope's rates and contributions are mostly noise.
const LOW_SAMPLE_MINUTES = 8
const CONTRIBUTION_DIVIDER_KEY = '__divider'
const FACTOR_STAT_LABELS = {
  shooting: 'Shooting',
  ball_handling: 'Ball Handling',
  orebounding: 'Off Rebounds',
  free_throws: 'FT Rate',
}
const BOX_SCORE_COLUMNS = [
  { key: 'minutes', label: 'MIN', className: 'boxscore-minutes' },
  { key: 'pts', label: 'PTS' },
  { key: 'fgmFga', label: 'FG' },
  { key: 'fg3mFg3a', label: '3PT' },
  { key: 'ftmFta', label: 'FT' },
  { key: 'oreb', label: 'OREB' },
  { key: 'dreb', label: 'DREB' },
  { key: 'reb', label: 'REB' },
  { key: 'ast', label: 'AST' },
  { key: 'stl', label: 'STL' },
  { key: 'blk', label: 'BLK' },
  { key: 'tov', label: 'TOV' },
  { key: 'pf', label: 'PF' },
  { key: 'plusMinus', label: '+/-', className: 'boxscore-plus-minus' },
]
const BOX_SCORE_SEGMENTS = [
  { label: 'Full game', value: 'game' },
  { label: 'Q1', value: 'q1' },
  { label: 'Q2', value: 'q2' },
  { label: 'Q3', value: 'q3' },
  { label: 'Q4', value: 'q4' },
  { label: 'OT', value: 'ot' },
  { label: 'H1', value: 'h1' },
  { label: 'H2', value: 'h2' },
  { label: 'Excl. garbage', value: 'no_garbage' },
  { label: 'Garbage only', value: 'garbage' },
  { label: 'Clutch', value: 'clutch' },
]

const getMonthKeyFromDate = (dateStr) => {
  if (typeof dateStr !== 'string') return ''
  const [year, month] = dateStr.split('-')
  if (!year || !month) return ''
  return `${year}-${month}`
}

const formatMonthLabel = (monthKey) => {
  if (typeof monthKey !== 'string') return monthKey
  const [year, month] = monthKey.split('-')
  const monthIndex = Number(month) - 1
  if (!year || !Number.isInteger(monthIndex) || monthIndex < 0 || monthIndex > 11) {
    return monthKey
  }
  return `${MONTH_NAMES[monthIndex]} ${year}`
}

const formatGameOptionLabel = (game) => {
  if (!game) return ''
  const date = game.date || ''
  const roadTeam = game.road_team || ''
  const homeTeam = game.home_team || ''
  const roadPts = Number.isFinite(game.road_pts) ? game.road_pts : '?'
  const homePts = Number.isFinite(game.home_pts) ? game.home_pts : '?'
  const gameTypeTagMap = {
    play_in: 'Play-In',
    nba_cup_final: 'NBA Cup Final',
    playoffs: 'Playoff',
  }
  const gameTypeTag = gameTypeTagMap[game.game_type]
  const overtimeSuffix = game.is_overtime ? ' (OT)' : ''
  const gameTypeSuffix = gameTypeTag ? ` - ${gameTypeTag}` : ''
  const incompleteSuffix = game.data_incomplete ? ' (data incomplete)' : ''
  return `${date}: ${roadTeam} ${roadPts} @ ${homeTeam} ${homePts}${overtimeSuffix}${gameTypeSuffix}${incompleteSuffix}`
}

const normalizeGameId = (value) => {
  const text = String(value ?? '').trim()
  if (!text) return ''
  return /^\d+$/.test(text) ? text.padStart(10, '0') : text
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

function normalizeDataScopeParam(value) {
  const token = normalizeQueryToken(value)
  if (!token) return ''
  if (VALID_DATA_SCOPE_OPTIONS.has(token)) return token
  if (token === 'garbage' || token === 'non_garbage' || token === 'non_garbage_time') return 'garbage_filtered'
  if (token === 'clutch_time') return 'clutch'
  return ''
}

function normalizeAnalysisViewParam(value) {
  const token = normalizeQueryToken(value)
  if (!token) return ''
  if (VALID_ANALYSIS_VIEWS.has(token)) return token
  if (token === 'traditional' || token === 'traditional_box' || token === 'traditional_box_score') return 'boxscore'
  if (token === 'factors') return 'factor'
  return ''
}

function parseMinutesToSeconds(value) {
  if (typeof value !== 'string') return 0
  const [minutesText, secondsText = '0'] = value.split(':')
  const minutes = Number(minutesText)
  const seconds = Number(secondsText)
  if (!Number.isFinite(minutes) || !Number.isFinite(seconds)) return 0
  return (minutes * 60) + seconds
}

function formatSecondsAsMinutes(value) {
  if (!Number.isFinite(value) || value <= 0) return '-'
  const minutes = Math.floor(value / 60)
  const seconds = Math.round(value % 60)
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

function formatPlusMinus(value) {
  if (!Number.isFinite(value)) return '-'
  return value > 0 ? `+${value}` : String(value)
}

function formatBoxScoreCell(player, key) {
  switch (key) {
    case 'minutes':
      return player.minutes || '-'
    case 'fgmFga':
      return `${player.fgm}-${player.fga}`
    case 'fg3mFg3a':
      return `${player.fg3m}-${player.fg3a}`
    case 'ftmFta':
      return `${player.ftm}-${player.fta}`
    case 'plusMinus':
      return formatPlusMinus(player.plus_minus)
    default:
      return player[key] ?? '-'
  }
}

function buildBoxScoreTotals(players) {
  return players.reduce((totals, player) => ({
    minutes: totals.minutes + parseMinutesToSeconds(player.minutes),
    pts: totals.pts + (player.pts || 0),
    fgm: totals.fgm + (player.fgm || 0),
    fga: totals.fga + (player.fga || 0),
    fg3m: totals.fg3m + (player.fg3m || 0),
    fg3a: totals.fg3a + (player.fg3a || 0),
    ftm: totals.ftm + (player.ftm || 0),
    fta: totals.fta + (player.fta || 0),
    oreb: totals.oreb + (player.oreb || 0),
    dreb: totals.dreb + (player.dreb || 0),
    reb: totals.reb + (player.reb || 0),
    ast: totals.ast + (player.ast || 0),
    stl: totals.stl + (player.stl || 0),
    blk: totals.blk + (player.blk || 0),
    tov: totals.tov + (player.tov || 0),
    pf: totals.pf + (player.pf || 0),
    plus_minus: totals.plus_minus + (player.plus_minus || 0),
  }), {
    minutes: 0,
    pts: 0,
    fgm: 0,
    fga: 0,
    fg3m: 0,
    fg3a: 0,
    ftm: 0,
    fta: 0,
    oreb: 0,
    dreb: 0,
    reb: 0,
    ast: 0,
    stl: 0,
    blk: 0,
    tov: 0,
    pf: 0,
    plus_minus: 0,
  })
}

// "Giannis Antetokounmpo" -> "G. Antetokounmpo" for the narrow sticky player column
function abbreviatePlayerName(name) {
  const parts = String(name || '').trim().split(/\s+/)
  if (parts.length < 2) return name
  return `${parts[0].charAt(0)}. ${parts.slice(1).join(' ')}`
}

function formatMinutesValue(minutes) {
  const rounded = Math.round(minutes * 10) / 10
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1)
}

function FourFactor() {
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()
  const [seasons, setSeasons] = useState([])
  const [games, setGames] = useState([])
  const [selectedSeason, setSelectedSeason] = useState('')
  const [selectedMonth, setSelectedMonth] = useState('')
  const [selectedDataScope, setSelectedDataScope] = useState('all')
  const [selectedGame, setSelectedGame] = useState('')
  const [pendingNavigationSelection, setPendingNavigationSelection] = useState(null)
  const factorType = 'eight_factors'
  const [analysisView, setAnalysisView] = useState('boxscore')
  const [decomposition, setDecomposition] = useState(null)
  const [boxScore, setBoxScore] = useState(null)
  const [boxScoreLoading, setBoxScoreLoading] = useState(false)
  const [boxScoreError, setBoxScoreError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [error, setError] = useState(null)
  const [glossaryExpanded, setGlossaryExpanded] = useState(false)
  const [interpretation, setInterpretation] = useState(null)
  const [interpretationLoading, setInterpretationLoading] = useState(false)
  const [timelineData, setTimelineData] = useState(null)
  const [timelineLoading, setTimelineLoading] = useState(false)
  const [timelineError, setTimelineError] = useState(null)
  const [boxScoreSegment, setBoxScoreSegment] = useState('game')
  const [scopeNotice, setScopeNotice] = useState('')
  const [scopeMinutesByScope, setScopeMinutesByScope] = useState({})
  const [showSingleGameSaveOption, setShowSingleGameSaveOption] = useState(false)
  const [singleGameSaveMessage, setSingleGameSaveMessage] = useState('')
  const [singleGameSaveError, setSingleGameSaveError] = useState(false)
  const [singleGameSaving, setSingleGameSaving] = useState(false)
  const isCompactChart = useMediaQuery('(max-width: 640px)')
  const [clutchGames, setClutchGames] = useState(null)
  const singleGameSaveTimerRef = useRef(null)
  const hasUserInteractedRef = useRef(false)
  const lastMonthLabelRef = useRef('')
  const lastGameLabelRef = useRef('')
  const urlSelections = useMemo(() => ({
    season: readSearchParam(searchParams, 'season').trim(),
    month: readSearchParam(searchParams, 'month').trim(),
    gameId: normalizeGameId(readSearchParam(searchParams, 'game', 'game_id', 'gameId')),
    dataScope: normalizeDataScopeParam(readSearchParam(searchParams, 'scope', 'data_scope', 'dataScope')),
    analysisView: normalizeAnalysisViewParam(readSearchParam(searchParams, 'view', 'analysis_view', 'analysisView')),
  }), [searchParams])

  const buildSearchParamsFromState = (overrides = {}) => {
    const nextState = {
      season: selectedSeason,
      month: selectedMonth,
      gameId: selectedGame,
      dataScope: selectedDataScope,
      analysisView,
      ...overrides,
    }

    const nextParams = new URLSearchParams()
    if (nextState.season) nextParams.set('season', nextState.season)
    if (nextState.month) nextParams.set('month', nextState.month)
    if (nextState.gameId) nextParams.set('game', normalizeGameId(nextState.gameId))
    if (nextState.dataScope) nextParams.set('scope', nextState.dataScope)
    if (nextState.analysisView) nextParams.set('view', nextState.analysisView)
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
    updateUrlFromUserChange({ season: value, month: '', gameId: '' })
    if (value && value !== selectedSeason) setInitializing(true)
    // Drop the old season's games so the month/game auto-select can't pair an old game
    // with the new season (that fired 404 requests) while the new list loads.
    setGames([])
    setSelectedSeason(value)
    setSelectedMonth('')
    setSelectedGame('')
  }

  const handleMonthChange = (value) => {
    updateUrlFromUserChange({ month: value, gameId: '' })
    setSelectedMonth(value)
    setSelectedGame('')
  }

  const handleGameChange = (value) => {
    updateUrlFromUserChange({ gameId: value })
    setSelectedGame(value)
  }

  const handleAnalysisViewChange = (value) => {
    updateUrlFromUserChange({ analysisView: value })
    setAnalysisView(value)
  }

  const handleDataScopeChange = (value) => {
    updateUrlFromUserChange({ dataScope: value })
    setSelectedDataScope(value)
  }
  const interpretationEnabled =
    INTERPRETATION_UI_ENABLED &&
    INTERPRETATION_ENABLED_SEASONS.has(selectedSeason) &&
    selectedDataScope === 'all' &&
    factorType === 'eight_factors'

  useEffect(() => {
    let isCurrent = true
    async function loadInitialData() {
      try {
        const seasonsRes = await getSeasons()
        if (!isCurrent) return
        setSeasons(seasonsRes.seasons)
        // Default to first available season.
        setSelectedSeason(prev => {
          if (urlSelections.season && seasonsRes.seasons.includes(urlSelections.season)) {
            return urlSelections.season
          }
          if (prev && seasonsRes.seasons.includes(prev)) return prev
          return seasonsRes.seasons.length > 0 ? seasonsRes.seasons[0] : ''
        })
      } catch (err) {
        if (isCurrent) setError(err.message)
      }
    }
    loadInitialData()
    return () => { isCurrent = false }
  }, [urlSelections.season])

  useEffect(() => {
    if (urlSelections.season && seasons.includes(urlSelections.season) && selectedSeason !== urlSelections.season) {
      setSelectedSeason(urlSelections.season)
    }
  }, [seasons, selectedSeason, urlSelections.season])

  useEffect(() => {
    if (urlSelections.dataScope && selectedDataScope !== urlSelections.dataScope) {
      setSelectedDataScope(urlSelections.dataScope)
    }
  }, [selectedDataScope, urlSelections.dataScope])

  useEffect(() => {
    if (urlSelections.analysisView && analysisView !== urlSelections.analysisView) {
      setAnalysisView(urlSelections.analysisView)
    }
  }, [analysisView, urlSelections.analysisView])

  useEffect(() => {
    return () => {
      if (singleGameSaveTimerRef.current) {
        clearTimeout(singleGameSaveTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    const navState = location.state
    if (!navState || typeof navState !== 'object') return

    const navGameIdRaw = navState.gameId
    const navGameId = normalizeGameId(navGameIdRaw)
    if (!navGameId) return

    const navSeason = typeof navState.season === 'string' ? navState.season : ''
    if (navSeason) {
      setSelectedSeason(navSeason)
    }
    const nextParams = new URLSearchParams()
    if (navSeason) nextParams.set('season', navSeason)
    nextParams.set('game', navGameId)
    nextParams.set('scope', 'all')
    nextParams.set('view', 'boxscore')
    setSearchParams(nextParams, { replace: true })
    setPendingNavigationSelection({
      season: navSeason,
      gameId: navGameId,
      gameDate: typeof navState.gameDate === 'string' ? navState.gameDate : '',
      team: typeof navState.team === 'string' ? navState.team : '',
      opponent: typeof navState.opponent === 'string' ? navState.opponent : '',
      homeAway: typeof navState.homeAway === 'string' ? navState.homeAway : '',
    })
    setSelectedDataScope('all')
    setAnalysisView('boxscore')
  }, [location.key, location.state, setSearchParams])

  useEffect(() => {
    let isCurrent = true
    async function loadGames() {
      if (!selectedSeason) return
      setInitializing(true)
      if (isCurrent) {
        setSelectedMonth('')
        setSelectedGame('')
      }
      try {
        const gamesRes = await getGames(selectedSeason, 'all')
        if (!isCurrent) return
        const normalizedGames = (gamesRes.games || []).map((game) => ({
          ...game,
          game_id: normalizeGameId(game.game_id),
        }))
        setGames(normalizedGames)
      } catch (err) {
        if (isCurrent) setError(err.message)
      } finally {
        if (isCurrent) setInitializing(false)
      }
    }
    loadGames()
    return () => { isCurrent = false }
  }, [selectedSeason])

  const monthOptions = useMemo(() => {
    const seen = new Set()
    const options = []

    games.forEach((game) => {
      const monthKey = getMonthKeyFromDate(game.date)
      if (!monthKey || seen.has(monthKey)) return
      seen.add(monthKey)
      options.push({ value: monthKey, label: formatMonthLabel(monthKey) })
    })

    return options
  }, [games])

  const filteredGames = useMemo(() => {
    if (!selectedMonth) return []
    return games.filter((game) => getMonthKeyFromDate(game.date) === selectedMonth)
  }, [games, selectedMonth])

  const selectedGameInfo = useMemo(() => {
    const gameId = normalizeGameId(selectedGame)
    if (!gameId) return null
    return games.find((game) => normalizeGameId(game.game_id) === gameId) || null
  }, [games, selectedGame])
  const selectedGameIncomplete = Boolean(selectedGameInfo?.data_incomplete)

  useEffect(() => {
    if (!pendingNavigationSelection || games.length === 0) return
    const seasonMismatch =
      pendingNavigationSelection.season &&
      pendingNavigationSelection.season !== selectedSeason &&
      seasons.includes(pendingNavigationSelection.season)
    if (seasonMismatch) return

    let targetGame = games.find((game) => normalizeGameId(game.game_id) === pendingNavigationSelection.gameId)
    if (!targetGame && pendingNavigationSelection.gameDate && pendingNavigationSelection.team && pendingNavigationSelection.opponent) {
      const targetDate = pendingNavigationSelection.gameDate.slice(0, 10)
      if (pendingNavigationSelection.homeAway === 'home') {
        targetGame = games.find((game) =>
          game.date === targetDate &&
          game.home_team === pendingNavigationSelection.team &&
          game.road_team === pendingNavigationSelection.opponent
        )
      } else if (pendingNavigationSelection.homeAway === 'road') {
        targetGame = games.find((game) =>
          game.date === targetDate &&
          game.road_team === pendingNavigationSelection.team &&
          game.home_team === pendingNavigationSelection.opponent
        )
      }
    }
    if (!targetGame) {
      setPendingNavigationSelection(null)
      return
    }

    const monthKey = getMonthKeyFromDate(targetGame.date)
    if (monthKey) {
      setSelectedMonth(monthKey)
    }
    setSelectedGame(targetGame.game_id)
    setPendingNavigationSelection(null)
  }, [games, pendingNavigationSelection, seasons, selectedSeason])

  useEffect(() => {
    if (monthOptions.length === 0) {
      setSelectedMonth('')
      return
    }
    setSelectedMonth((prev) => {
      if (urlSelections.month && monthOptions.some((option) => option.value === urlSelections.month)) {
        return urlSelections.month
      }
      if (prev && monthOptions.some((option) => option.value === prev)) return prev
      return monthOptions[0].value
    })
  }, [monthOptions, urlSelections.month])

  useEffect(() => {
    if (urlSelections.month && monthOptions.some((option) => option.value === urlSelections.month) && selectedMonth !== urlSelections.month) {
      setSelectedMonth(urlSelections.month)
    }
  }, [monthOptions, selectedMonth, urlSelections.month])

  useEffect(() => {
    if (!selectedMonth) {
      if (pendingNavigationSelection) return
      setSelectedGame('')
      return
    }
    setSelectedGame((prev) => {
      if (urlSelections.gameId) {
        const urlGame = filteredGames.find((game) => normalizeGameId(game.game_id) === urlSelections.gameId)
        if (urlGame) return urlGame.game_id
      }
      if (prev) {
        const matchedGame = filteredGames.find((game) => normalizeGameId(game.game_id) === normalizeGameId(prev))
        if (matchedGame) return matchedGame.game_id
      }
      return filteredGames.length > 0 ? filteredGames[0].game_id : ''
    })
  }, [selectedMonth, filteredGames, pendingNavigationSelection, urlSelections.gameId])

  useEffect(() => {
    if (!urlSelections.gameId || games.length === 0) return
    const targetGame = games.find((game) => normalizeGameId(game.game_id) === urlSelections.gameId)
    if (!targetGame) return

    const monthKey = getMonthKeyFromDate(targetGame.date)
    if (monthKey && selectedMonth !== monthKey) {
      setSelectedMonth(monthKey)
    }
    if (selectedGame !== targetGame.game_id) {
      setSelectedGame(targetGame.game_id)
    }
  }, [games, selectedGame, selectedMonth, urlSelections.gameId])

  useEffect(() => {
    if (!hasUserInteractedRef.current) return
    const nextParams = buildSearchParamsFromState()
    if (nextParams.toString() !== searchParams.toString()) {
      setSearchParams(nextParams, { replace: true })
    }
  }, [
    analysisView,
    searchParams,
    selectedDataScope,
    selectedGame,
    selectedMonth,
    selectedSeason,
    setSearchParams,
  ])

  // Games that have clutch-time contributions, one request per season. Lets us skip the
  // clutch decomposition request (which 404s) for games with no clutch minutes.
  useEffect(() => {
    let isCurrent = true
    if (!selectedSeason) return undefined
    getGames(selectedSeason, 'clutch')
      .then((res) => {
        if (!isCurrent) return
        const ids = new Set((res.games || []).map((game) => normalizeGameId(game.game_id)))
        setClutchGames({ season: selectedSeason, ids })
      })
      .catch(() => {
        // Unknown: fall back to requesting the clutch scope directly
        if (isCurrent) setClutchGames({ season: selectedSeason, ids: null })
      })
    return () => { isCurrent = false }
  }, [selectedSeason])

  // 'pending' until this season's clutch list arrives; 'yes' when the clutch request is worth making
  const gameClutchStatus = !selectedGame || clutchGames?.season !== selectedSeason
    ? 'pending'
    : (clutchGames.ids === null || clutchGames.ids.has(normalizeGameId(selectedGame)) ? 'yes' : 'no')
  const decompositionClutchStatus = selectedDataScope === 'clutch' ? gameClutchStatus : 'n/a'

  useEffect(() => {
    let isCurrent = true
    async function loadDecomposition() {
      // Previous results stay on screen (dimmed) until the new ones arrive.
      if (!selectedSeason || !selectedGame) {
        if (isCurrent) setLoading(false)
        return
      }
      setLoading(true)
      if (isCurrent) {
        setError(null)
        setScopeNotice('')
      }
      if (decompositionClutchStatus === 'pending') return
      if (decompositionClutchStatus === 'no') {
        setDecomposition(null)
        setScopeNotice('No clutch time data for this game.')
        setLoading(false)
        return
      }
      try {
        const data = await getDecomposition(selectedSeason, selectedGame, factorType, selectedDataScope)
        if (isCurrent) setDecomposition(data)
      } catch (err) {
        if (!isCurrent) return
        setDecomposition(null)
        const isScopedMissing =
          selectedDataScope !== 'all' &&
          (err.message === 'Game not found in contributions' || err.message === 'Contributions not found for season')
        if (isScopedMissing) {
          const scopeLabel = DATA_SCOPE_OPTIONS.find((opt) => opt.value === selectedDataScope)?.label || 'selected scope'
          setScopeNotice(`No ${scopeLabel.toLowerCase()} data for this game.`)
          setError(null)
        } else {
          setError(err.message)
        }
      } finally {
        if (isCurrent) setLoading(false)
      }
    }
    loadDecomposition()
    return () => { isCurrent = false }
  }, [selectedSeason, selectedGame, factorType, selectedDataScope, decompositionClutchStatus])

  useEffect(() => {
    let isCurrent = true

    async function loadBoxScore() {
      if (!selectedSeason || !selectedGame) {
        if (isCurrent) setBoxScoreLoading(false)
        return
      }

      setBoxScoreLoading(true)
      if (isCurrent) setBoxScoreError(null)

      try {
        const data = await getPbpTraditionalBoxScore(selectedSeason, selectedGame, boxScoreSegment)
        if (isCurrent) setBoxScore(data)
      } catch (err) {
        if (isCurrent) {
          setBoxScore(null)
          setBoxScoreError(err.message)
        }
      } finally {
        if (isCurrent) setBoxScoreLoading(false)
      }
    }

    loadBoxScore()
    return () => { isCurrent = false }
  }, [boxScoreSegment, selectedSeason, selectedGame])

  useEffect(() => {
    let isCurrent = true

    async function loadScopeMinutes() {
      if (!selectedSeason || !selectedGame) {
        if (isCurrent) setScopeMinutesByScope({})
        return
      }

      if (isCurrent) setScopeMinutesByScope({})
      if (gameClutchStatus === 'pending') return

      const parseScopeMinutes = (data) => {
        const roadMinutes = Number(data?.road_ratings?.minutes)
        if (Number.isFinite(roadMinutes) && roadMinutes >= 0) {
          return roadMinutes / 5
        }
        const poss = Number(data?.road_ratings?.possessions)
        const pace = Number(data?.road_ratings?.pace)
        if (Number.isFinite(poss) && poss >= 0 && Number.isFinite(pace) && pace > 0) {
          return (poss * 48) / pace
        }
        return null
      }

      const entries = await Promise.all(
        DATA_SCOPE_OPTIONS.map(async (option) => {
          if (option.value === 'clutch' && gameClutchStatus === 'no') return [option.value, 0]
          try {
            const data = await getDecomposition(selectedSeason, selectedGame, 'eight_factors', option.value)
            return [option.value, parseScopeMinutes(data)]
          } catch (err) {
            const msg = err?.message || ''
            const notFound =
              msg === 'Game not found in contributions' || msg === 'Contributions not found for season'
            return [option.value, notFound ? 0 : null]
          }
        })
      )

      if (isCurrent) {
        setScopeMinutesByScope(Object.fromEntries(entries))
      }
    }

    loadScopeMinutes()
    return () => { isCurrent = false }
  }, [selectedSeason, selectedGame, gameClutchStatus])

  useEffect(() => {
    let isCurrent = true
    async function loadInterpretation() {
      if (!decomposition || !interpretationEnabled) {
        if (isCurrent) {
          setInterpretation(null)
          setInterpretationLoading(false)
        }
        return
      }
      // A previous game's decomposition is still on screen while the new one loads
      if (normalizeGameId(decomposition.game_id) !== normalizeGameId(selectedGame)) return

      setInterpretationLoading(true)
      try {
        const data = await getInterpretation(
          decomposition,
          factorType,
          selectedSeason,
          selectedDataScope
        )
        if (isCurrent) setInterpretation(data.interpretation)
      } catch (err) {
        // Silently fail - interpretation is optional enhancement
        if (isCurrent) setInterpretation(null)
      } finally {
        if (isCurrent) setInterpretationLoading(false)
      }
    }
    loadInterpretation()
    return () => { isCurrent = false }
  }, [decomposition, factorType, interpretationEnabled, selectedDataScope, selectedGame, selectedSeason])

  useEffect(() => {
    setTimelineError(null)
    setTimelineLoading(false)
  }, [selectedSeason, selectedGame])

  useEffect(() => {
    if (!selectedGame) return
    const nextView = urlSelections.analysisView || 'boxscore'
    if (analysisView !== nextView) {
      setAnalysisView(nextView)
    }
  }, [selectedGame, urlSelections.analysisView])

  useEffect(() => {
    if (!selectedGame) return
    const nextScope = urlSelections.dataScope || 'all'
    if (selectedDataScope !== nextScope) {
      setSelectedDataScope(nextScope)
    }
  }, [selectedGame, urlSelections.dataScope])

  useEffect(() => {
    let isCurrent = true

    async function loadTimeline() {
      if (analysisView !== 'timeline' || !selectedSeason || !selectedGame || !decomposition) return
      // Wait for the selected game's decomposition; a stale one would request the wrong teams.
      if (normalizeGameId(decomposition.game_id) !== normalizeGameId(selectedGame)) return
      if (timelineData && normalizeGameId(timelineData.game_id) === normalizeGameId(selectedGame)) return

      setTimelineLoading(true)
      setTimelineError(null)
      try {
        const data = await getGameTimeline(selectedSeason, selectedGame, {
          gameType: decomposition.game_type,
          homeTeam: decomposition.home_team,
          roadTeam: decomposition.road_team,
        })
        if (isCurrent) setTimelineData(data)
      } catch (err) {
        if (isCurrent) {
          setTimelineData(null)
          setTimelineError(err.message)
        }
      } finally {
        if (isCurrent) setTimelineLoading(false)
      }
    }

    loadTimeline()
    return () => { isCurrent = false }
  }, [analysisView, decomposition, selectedSeason, selectedGame])

  const contributionChartData = useMemo(() => {
    if (!decomposition) return []
    const contributions = decomposition.contributions || {}

    // Above/below league average for one side's factor (eight-factor keys only)
    const getQuality = (side, factorKey) => {
      if (!decomposition.league_averages) return ''
      const statKey = { shooting: 'efg', ball_handling: 'ball_handling', orebounding: 'oreb', free_throws: 'ft_rate' }[factorKey]
      const factorValue = decomposition[`${side}_factors`]?.[statKey]
      const leagueAvg = decomposition.league_averages[statKey]
      if (factorValue === undefined || leagueAvg === undefined) return ''
      return factorValue >= leagueAvg ? 'Good' : 'Bad'
    }

    const getFactorValue = (factor) => {
      const valueMap = {
        'shooting': `${decomposition.road_factors.efg.toFixed(1)} vs ${decomposition.home_factors.efg.toFixed(1)}`,
        'ball_handling': `${decomposition.road_factors.ball_handling.toFixed(1)} vs ${decomposition.home_factors.ball_handling.toFixed(1)}`,
        'orebounding': `${decomposition.road_factors.oreb.toFixed(1)} vs ${decomposition.home_factors.oreb.toFixed(1)}`,
        'free_throws': `${decomposition.road_factors.ft_rate.toFixed(1)} vs ${decomposition.home_factors.ft_rate.toFixed(1)}`,
        'home_shooting': decomposition.home_factors.efg.toFixed(1),
        'road_shooting': decomposition.road_factors.efg.toFixed(1),
        'home_ball_handling': decomposition.home_factors.ball_handling.toFixed(1),
        'road_ball_handling': decomposition.road_factors.ball_handling.toFixed(1),
        'home_orebounding': decomposition.home_factors.oreb.toFixed(1),
        'road_orebounding': decomposition.road_factors.oreb.toFixed(1),
        'home_free_throws': decomposition.home_factors.ft_rate.toFixed(1),
        'road_free_throws': decomposition.road_factors.ft_rate.toFixed(1),
      }
      return valueMap[factor] || null
    }

    // Axis label at every width: "GSW Shooting ▲" (▲/▼ = above/below league avg, keyed in the legend).
    // The tooltip spells it out: "Good GSW Shooting".
    const buildLabel = (factor) => {
      const match = factor.match(/^(home|road)_(.+)$/)
      const side = match ? match[1] : null
      const statKey = match ? match[2] : factor
      const stat = FACTOR_STAT_LABELS[statKey]
      if (!stat) return { axis: factor, full: factor }
      if (!side) return { axis: stat, full: stat }
      const team = side === 'home' ? decomposition.home_team : decomposition.road_team
      const quality = getQuality(side, statKey)
      const arrow = quality === 'Good' ? ' ▲' : quality === 'Bad' ? ' ▼' : ''
      return {
        axis: `${team} ${stat}${arrow}`,
        full: `${quality ? `${quality} ` : ''}${team} ${stat}`,
      }
    }

    const roadWon = decomposition.road_pts > decomposition.home_pts

    // Keep winning-side contributions at the top:
    // - home win: most positive (helps home) first
    // - road win: most negative (helps road) first
    const factorBars = Object.entries(contributions)
      .map(([factor, value]) => {
        const label = buildLabel(factor)
        return {
          factor,
          axisLabel: label.axis,
          fullLabel: label.full,
          value,
          actualValue: getFactorValue(factor),
          // Contribution direction = the team it helps, drawn in that team's color
          fill: value >= 0 ? 'var(--color-home)' : 'var(--color-road)',
          fillOpacity: 1,
        }
      })
      .sort((a, b) => roadWon ? a.value - b.value : b.value - a.value)

    // Empty spacer row separates the model's unexplained error from the factors
    factorBars.push({
      factor: CONTRIBUTION_DIVIDER_KEY,
      isDivider: true,
      value: null,
    })

    // Error = actual rating diff - predicted rating diff (always last, not sorted)
    const error = decomposition.actual_rating_diff - decomposition.predicted_rating_diff
    factorBars.push({
      factor: 'Error',
      axisLabel: 'Error',
      fullLabel: 'Error (not explained by the factors)',
      value: error,
      isError: true,
      // Solid neutral (≥3:1 on white), gray so it never reads as either team
      fill: 'var(--chart-neutral)',
      fillOpacity: 1,
    })

    return factorBars
  }, [decomposition])

  const contributionByFactor = useMemo(
    () => new Map(contributionChartData.map((entry) => [entry.factor, entry])),
    [contributionChartData]
  )

  // X-axis ticks for contribution chart: every 1 point, or a coarser round step
  // when the range is wide (tiny samples can produce ±100+ contributions)
  const contributionXAxisConfig = useMemo(() => {
    const values = contributionChartData
      .filter((d) => Number.isFinite(d.value))
      .map((d) => d.value)
    if (values.length === 0) return { domain: [-5, 5], ticks: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] }

    const maxTicks = isCompactChart ? 8 : 18
    const rawMin = Math.floor(Math.min(...values)) - 1
    const rawMax = Math.ceil(Math.max(...values)) + 1
    const step = [1, 2, 5, 10, 20, 25, 50, 100, 200].find((s) => (rawMax - rawMin) / s <= maxTicks) || 500
    const tickMin = Math.floor(rawMin / step) * step
    const tickMax = Math.ceil(rawMax / step) * step

    const ticks = []
    for (let t = tickMin; t <= tickMax; t += step) ticks.push(t)

    return { domain: [tickMin, tickMax], ticks }
  }, [contributionChartData, isCompactChart])

  const formatFactorValue = (value) => {
    if (value === null || value === undefined) return '-'
    return typeof value === 'number' ? value.toFixed(1) : value
  }

  // Determine background color class based on comparison to league average
  // Green (positive) if above average, red (negative) if below
  const getFactorCellClass = (value, factorKey) => {
    if (!decomposition?.league_averages || value === null || value === undefined) return ''

    // Map internal factor keys to league_averages keys
    // Note: backend normalizes oreb_pct to oreb in the response (api.py:301-303)
    const keyMap = {
      efg: 'efg',
      ball_handling: 'ball_handling',
      oreb: 'oreb',
      ft_rate: 'ft_rate',
    }

    const avgKey = keyMap[factorKey]
    const avg = decomposition.league_averages[avgKey]
    if (avg === undefined) return ''

    return value >= avg ? 'bg-positive' : 'bg-negative'
  }

  // Determine color class for ratings based on league average
  // For ORtg: higher is better (above avg = green)
  // For DRtg: lower is better (below avg = green)
  const getRatingCellClass = (value, ratingType) => {
    if (!decomposition?.league_averages || value === null || value === undefined) return ''

    const avgKey = ratingType === 'off' ? 'off_rating' : 'def_rating'
    const avg = decomposition.league_averages[avgKey]
    if (avg === undefined) return ''

    if (ratingType === 'off') {
      // Higher ORtg is better
      return value >= avg ? 'bg-positive' : 'bg-negative'
    } else {
      // Lower DRtg is better
      return value <= avg ? 'bg-positive' : 'bg-negative'
    }
  }

  // Format game type from snake_case to readable format
  const formatGameType = (type) => {
    if (!type) return ''
    const typeMap = {
      'regular_season': 'Regular Season',
      'playoffs': 'Playoffs',
      'nba_cup_group': 'NBA Cup (Group)',
      'nba_cup_knockout': 'NBA Cup (Knockout)',
    }
    return typeMap[type] || type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }

  const getScopeMinutes = (scopeValue) => {
    const minutes = scopeMinutesByScope[scopeValue]
    return Number.isFinite(minutes) ? minutes : null
  }

  const formatScopeMinutesLabel = (scopeValue) => {
    const minutes = getScopeMinutes(scopeValue)
    return minutes === null ? '-- min' : `${formatMinutesValue(minutes)} min`
  }

  const dataScopeOptions = DATA_SCOPE_OPTIONS.map((option) => {
    const minutes = getScopeMinutes(option.value)
    const empty = option.value !== 'all' && minutes === 0
    return {
      value: option.value,
      disabled: empty,
      title: empty
        ? (option.value === 'clutch' ? 'No clutch minutes in this game' : `No ${option.label.toLowerCase()} minutes in this game`)
        : undefined,
      label: (
        <>
          <span className="ga-scope-name ga-label-long">{option.label}</span>
          <span className="ga-scope-name ga-label-short">{SCOPE_SHORT_LABELS[option.value]}</span>
          <span className="ga-scope-minutes">{formatScopeMinutesLabel(option.value)}</span>
        </>
      ),
    }
  })

  const selectedScopeMinutes = getScopeMinutes(selectedDataScope)
  const isLowSample =
    selectedDataScope !== 'all' &&
    selectedScopeMinutes !== null &&
    selectedScopeMinutes > 0 &&
    selectedScopeMinutes < LOW_SAMPLE_MINUTES

  // Disable box-score periods we know have no data for this game
  const gameHasOvertime = Boolean(selectedGameInfo?.is_overtime ?? decomposition?.is_overtime)
  const allGameMinutes = getScopeMinutes('all')
  const nonGarbageMinutes = getScopeMinutes('garbage_filtered')
  const boxScoreSegmentOptions = BOX_SCORE_SEGMENTS.map((segment) => {
    let title
    if (segment.value === 'ot' && selectedGameInfo && !gameHasOvertime) title = 'No overtime in this game'
    if (segment.value === 'clutch' && getScopeMinutes('clutch') === 0) title = 'No clutch minutes in this game'
    if (
      segment.value === 'garbage' &&
      allGameMinutes !== null &&
      nonGarbageMinutes !== null &&
      Math.abs(allGameMinutes - nonGarbageMinutes) < 0.05
    ) {
      title = 'No garbage time in this game'
    }
    return { ...segment, disabled: Boolean(title), title }
  })
  const boxScoreSegmentDisabled = boxScoreSegmentOptions.some(
    (segment) => segment.value === boxScoreSegment && segment.disabled
  )

  useEffect(() => {
    if (boxScoreSegmentDisabled) setBoxScoreSegment('game')
  }, [boxScoreSegmentDisabled])

  // Mobile period chips: one scrollable row with edge fades; keep the active chip in view
  const periodScrollRef = useRef(null)
  const [periodFade, setPeriodFade] = useState({ left: false, right: false })
  const updatePeriodFade = () => {
    const el = periodScrollRef.current
    if (!el) return
    const left = el.scrollLeft > 2
    const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 2
    setPeriodFade((prev) => (prev.left === left && prev.right === right ? prev : { left, right }))
  }

  useEffect(() => {
    const el = periodScrollRef.current
    if (!el) return undefined
    const active = el.querySelector('[aria-pressed="true"]')
    if (active) {
      const start = active.offsetLeft - 24
      const end = active.offsetLeft + active.offsetWidth + 24 - el.clientWidth
      if (el.scrollLeft > start) el.scrollLeft = Math.max(0, start)
      else if (el.scrollLeft < end) el.scrollLeft = end
    }
    updatePeriodFade()
    window.addEventListener('resize', updatePeriodFade)
    return () => window.removeEventListener('resize', updatePeriodFade)
  }, [boxScoreSegment, analysisView, selectedGame, boxScore])

  const roadBoxScoreTotals = useMemo(
    () => buildBoxScoreTotals(boxScore?.road_players || []),
    [boxScore]
  )

  const homeBoxScoreTotals = useMemo(
    () => buildBoxScoreTotals(boxScore?.home_players || []),
    [boxScore]
  )

  const renderBoxScoreRow = (player) => (
    <tr key={`${player.team_abbreviation}-${player.player_id ?? player.player_name}`}>
      <th scope="row" className="ga-player-col" title={player.player_name}>
        <span className="ga-name-full">{player.player_name}</span>
        <span className="ga-name-short" aria-hidden="true">{abbreviatePlayerName(player.player_name)}</span>
      </th>
      {BOX_SCORE_COLUMNS.map((column) => (
        <td key={column.key} className={column.className || ''}>
          {formatBoxScoreCell(player, column.key)}
        </td>
      ))}
    </tr>
  )

  const renderBoxScoreTable = (players, totals, oppTotals, side, teamAbbr) => {
    const teamName = toFullTeamName(teamAbbr, boxScore?.season || selectedSeason)
    // Starters first, then a "Bench" divider row (stable order within each group)
    const starters = players.filter((player) => player.is_starter)
    const bench = players.filter((player) => !player.is_starter)
    const showBenchDivider = starters.length > 0 && bench.length > 0
    return (
      <section className="ga-boxscore-team card" key={side} aria-labelledby={`ga-boxscore-${side}`}>
        <h2 className="ga-boxscore-title" id={`ga-boxscore-${side}`}>
          <span className={`ga-team-marker ga-team-marker--${side}`} aria-hidden="true" />
          {teamName}
          <span className="ga-boxscore-side">{side === 'road' ? 'Road' : 'Home'}</span>
        </h2>
        <div className="table-scroll" role="region" aria-label={`${teamName} box score`} tabIndex={0}>
          <table className="ga-boxscore-table">
            <thead>
              <tr>
                <th scope="col" className="ga-player-col">Player</th>
                {BOX_SCORE_COLUMNS.map((column) => (
                  <th key={column.key} scope="col" className={column.className || ''}>
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(showBenchDivider ? starters : players).map(renderBoxScoreRow)}
              {showBenchDivider && (
                <tr className="ga-bench-row">
                  <th scope="rowgroup" className="ga-player-col">Bench</th>
                  <td colSpan={BOX_SCORE_COLUMNS.length} />
                </tr>
              )}
              {showBenchDivider && bench.map(renderBoxScoreRow)}
            </tbody>
            <tfoot>
              <tr className="ga-totals-row">
                <th scope="row" className="ga-player-col">Totals</th>
                <td className="boxscore-minutes"></td>
                <td>{totals.pts}</td>
                <td>{`${totals.fgm}-${totals.fga}`}</td>
                <td>{`${totals.fg3m}-${totals.fg3a}`}</td>
                <td>{`${totals.ftm}-${totals.fta}`}</td>
                <td>{totals.oreb}</td>
                <td>{totals.dreb}</td>
                <td>{totals.reb}</td>
                <td>{totals.ast}</td>
                <td>{totals.stl}</td>
                <td>{totals.blk}</td>
                <td>{totals.tov}</td>
                <td>{totals.pf}</td>
                <td className="boxscore-plus-minus" title="Team point margin for this period">
                  {formatPlusMinus(totals.pts - oppTotals.pts)}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </section>
    )
  }
  const scheduleSingleGameSaveOptionHide = () => {
    if (singleGameSaveTimerRef.current) {
      clearTimeout(singleGameSaveTimerRef.current)
    }
    singleGameSaveTimerRef.current = setTimeout(() => {
      setShowSingleGameSaveOption(false)
      setSingleGameSaveMessage('')
      setSingleGameSaveError(false)
    }, HIDDEN_SAVE_OPTION_MS)
  }

  const handlePageTitleContextMenu = (event) => {
    event.preventDefault()
    setShowSingleGameSaveOption(true)
    setSingleGameSaveMessage('')
    setSingleGameSaveError(false)
    scheduleSingleGameSaveOptionHide()
  }

  const handleSaveSingleGameJson = async () => {
    if (!selectedSeason || !selectedGame) {
      setSingleGameSaveMessage('Select a season and game first.')
      setSingleGameSaveError(true)
      scheduleSingleGameSaveOptionHide()
      return
    }

    setSingleGameSaving(true)
    setSingleGameSaveMessage('')
    setSingleGameSaveError(false)

    try {
      const payload = await getInterpretationPrompt(
        selectedSeason,
        selectedGame,
        factorType,
        selectedDataScope
      )

      const normalizedGameId = normalizeGameId(selectedGame)
      const scope = payload?.data_scope || selectedDataScope || 'all'
      const promptFactorType = payload?.factor_type || factorType
      const filename = `llm_prompt_${selectedSeason}_${scope}_${promptFactorType}_${normalizedGameId}.txt`
      const promptText = payload?.prompt || ''
      const blob = new Blob([promptText], { type: 'text/plain;charset=utf-8' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)

      setSingleGameSaveMessage('LLM prompt downloaded.')
      setSingleGameSaveError(false)
    } catch (err) {
      setSingleGameSaveMessage(err.message || 'Failed to download LLM prompt.')
      setSingleGameSaveError(true)
    } finally {
      setSingleGameSaving(false)
      scheduleSingleGameSaveOptionHide()
    }
  }

  const decompositionIsStale = Boolean(
    decomposition && normalizeGameId(decomposition.game_id) !== normalizeGameId(selectedGame)
  )
  const boxScoreIsStale = Boolean(
    boxScore && normalizeGameId(boxScore.game_id) !== normalizeGameId(selectedGame)
  )
  const timelineIsStale = Boolean(
    timelineData && normalizeGameId(timelineData.game_id) !== normalizeGameId(selectedGame)
  )
  const hasContent = Boolean(decomposition || boxScore || timelineData || (selectedGameIncomplete && selectedGameInfo))
  const isBusy =
    initializing ||
    loading ||
    (analysisView === 'boxscore' && boxScoreLoading) ||
    (analysisView === 'timeline' && timelineLoading)
  // Game/season change: dim the whole result area (scoreboard included) but keep it on screen
  const pageRefreshing = hasContent && (initializing || !selectedGame || decompositionIsStale)
  // View-level change (scope, period, timeline fetch): dim only that view's content
  const viewRefreshing = !pageRefreshing && (
    analysisView === 'factor' ? loading
      : analysisView === 'boxscore' ? (boxScoreLoading || boxScoreIsStale)
        : (timelineLoading || timelineIsStale)
  )
  const showResults = Boolean(selectedGame || hasContent)

  // While a new season/month resolves, the selects keep showing the previous choice
  // instead of flashing "Select month…/Select game…" over the still-visible old game.
  const currentMonthLabel = monthOptions.find((option) => option.value === selectedMonth)?.label
  const currentGameLabel = selectedGameInfo ? formatGameOptionLabel(selectedGameInfo) : ''
  if (currentMonthLabel) lastMonthLabelRef.current = currentMonthLabel
  if (currentGameLabel) lastGameLabelRef.current = currentGameLabel
  const selectionPending = initializing || (hasContent && (!selectedMonth || !selectedGame))
  const monthPlaceholder = selectionPending && lastMonthLabelRef.current ? lastMonthLabelRef.current : 'Select month...'
  const gamePlaceholder = selectionPending && lastGameLabelRef.current ? lastGameLabelRef.current : 'Select game...'

  const renderScoreboard = ({ roadTeam, homeTeam, roadPts, homePts }, details) => {
    const roadWon = Number(roadPts) > Number(homePts)
    const homeWon = Number(homePts) > Number(roadPts)
    return (
      <section className="ga-scoreboard card" aria-label="Final score">
        <div className="ga-matchup">
          <div className="ga-team ga-team--road">
            <span className="ga-team-name">{toCityName(roadTeam)}</span>
            <span className={`ga-team-score ${roadWon ? 'is-winner' : ''}`}>{roadPts}</span>
          </div>
          <span className="ga-at" aria-label="at">@</span>
          <div className="ga-team ga-team--home">
            <span className="ga-team-name">{toCityName(homeTeam)}</span>
            <span className={`ga-team-score ${homeWon ? 'is-winner' : ''}`}>{homePts}</span>
          </div>
        </div>
        {details}
      </section>
    )
  }

  const renderLinescoreRow = (side, team, line, total, won) => (
    <tr className={won ? 'is-winner' : ''}>
      <th scope="row" className="ga-linescore-team">
        <span className={`ga-team-marker ga-team-marker--${side}`} aria-hidden="true" />
        {team}
      </th>
      <td>{line.q1}</td>
      <td>{line.q2}</td>
      <td>{line.q3}</td>
      <td>{line.q4}</td>
      {decomposition.is_overtime && <td>{line.ot}</td>}
      <td className="ga-linescore-total">{total}</td>
    </tr>
  )

  const renderGameDetails = () => (
    <>
      {decomposition.linescore && (
        <div className="ga-linescore">
          <table>
            <thead>
              <tr>
                <th scope="col"><span className="sr-only">Team</span></th>
                <th scope="col">Q1</th>
                <th scope="col">Q2</th>
                <th scope="col">Q3</th>
                <th scope="col">Q4</th>
                {decomposition.is_overtime && (
                  <th scope="col">
                    {decomposition.overtime_count > 1 ? `${decomposition.overtime_count}OT` : 'OT'}
                  </th>
                )}
                <th scope="col">T</th>
              </tr>
            </thead>
            <tbody>
              {renderLinescoreRow(
                'road',
                decomposition.road_team,
                decomposition.linescore.road,
                decomposition.road_pts,
                decomposition.road_pts > decomposition.home_pts
              )}
              {renderLinescoreRow(
                'home',
                decomposition.home_team,
                decomposition.linescore.home,
                decomposition.home_pts,
                decomposition.home_pts > decomposition.road_pts
              )}
            </tbody>
          </table>
        </div>
      )}
      <div className="ga-game-info">
        <div className="ga-game-meta">
          <span className="ga-game-date">{decomposition.game_date}</span>
          {decomposition.game_type && (
            <span className="ga-game-type">{formatGameType(decomposition.game_type)}</span>
          )}
        </div>
        <div className="ga-external-links">
          <a
            href={`https://www.basketball-reference.com/boxscores/${decomposition.game_date.replace(/-/g, '')}0${toBBRefTeam(decomposition.home_team)}.html`}
            target="_blank"
            rel="noopener noreferrer"
          >
            BBRef Box Score
          </a>
          <a
            href={`https://www.nba.com/game/${decomposition.road_team.toLowerCase()}-vs-${decomposition.home_team.toLowerCase()}-${String(decomposition.game_id).padStart(10, '0')}/box-score`}
            target="_blank"
            rel="noopener noreferrer"
          >
            NBA.com Box Score
          </a>
        </div>
      </div>
    </>
  )

  const renderContributionTick = ({ x, y, payload }) => {
    const entry = contributionByFactor.get(payload.value)
    if (!entry || entry.isDivider) return <g />
    const fill = entry.isError ? 'var(--color-text-muted)' : 'var(--color-text-secondary)'
    return (
      <text x={x - 4} y={y} dy="0.35em" textAnchor="end" fill={fill} fontSize={isCompactChart ? 12 : 13}>
        <title>{entry.fullLabel}</title>
        {entry.axisLabel}
      </text>
    )
  }

  const renderContributionTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null
    const data = payload[0].payload
    if (!data || data.isDivider || !Number.isFinite(data.value)) return null
    const helpsHome = data.value >= 0
    return (
      <div className="ga-chart-tooltip">
        <div className="ga-chart-tooltip-title">{data.fullLabel}</div>
        {data.actualValue && (
          <div className="ga-chart-tooltip-meta">Value: {data.actualValue}</div>
        )}
        <div>
          Contribution: <span className="mono">{data.value > 0 ? '+' : ''}{data.value.toFixed(1)}</span>
          {!data.isError && (
            <span className={helpsHome ? 'ga-text-home' : 'ga-text-road'}>
              {' '}· helps {helpsHome ? decomposition.home_team : decomposition.road_team}
            </span>
          )}
        </div>
      </div>
    )
  }

  const contributionYAxisWidth = isCompactChart ? 136 : 156
  const contributionChartRight = isCompactChart ? 12 : 24

  return (
    <div className="ga-page">
      <div className="ga-topbar">
        <div className="ga-title-group">
          <h1 className="page-title" onContextMenu={handlePageTitleContextMenu}>
            Game Analysis
          </h1>
          <span className="refresh-indicator ga-refresh" role="status" aria-live="polite">
            {isBusy && hasContent && (
              <>
                <span className="loading-spinner" aria-hidden="true" />
                Loading…
              </>
            )}
          </span>
          {showSingleGameSaveOption && (
            <div className="ga-hidden-save" role="status" aria-live="polite">
              <button
                type="button"
                className="btn btn-secondary ga-hidden-save-button"
                onClick={handleSaveSingleGameJson}
                disabled={!selectedSeason || !selectedGame || singleGameSaving}
                title={!selectedGame ? 'Select a game first' : 'Download the exact prompt sent to the LLM for this game'}
              >
                {singleGameSaving ? 'Saving…' : 'Save LLM Prompt'}
              </button>
              {singleGameSaveMessage && (
                <span className={`ga-hidden-save-message ${singleGameSaveError ? 'is-error' : ''}`}>
                  {singleGameSaveMessage}
                </span>
              )}
            </div>
          )}
        </div>

        <SegmentedControl
          className="ga-view-seg"
          label="Analysis view"
          options={ANALYSIS_VIEW_OPTIONS}
          value={analysisView}
          onChange={handleAnalysisViewChange}
        />

        <div className="ga-selects">
          <div className="ga-field ga-field--season">
            <label className="form-label" htmlFor="ga-season">Season</label>
            <select
              id="ga-season"
              className="form-select"
              value={selectedSeason}
              onChange={(e) => handleSeasonChange(e.target.value)}
            >
              {/* Placeholder only until seasons load; a loaded season can't be blanked */}
              {(seasons.length === 0 || !selectedSeason) && (
                <option value="" disabled={seasons.length > 0}>Select season...</option>
              )}
              {seasons.map((season) => (
                <option key={season} value={season}>{season}</option>
              ))}
            </select>
          </div>

          <div className="ga-field ga-field--month">
            <label className="form-label" htmlFor="ga-month">Month</label>
            <select
              id="ga-month"
              className="form-select"
              value={selectedMonth}
              onChange={(e) => handleMonthChange(e.target.value)}
              disabled={!selectedSeason || monthOptions.length === 0}
            >
              {(monthOptions.length === 0 || !selectedMonth) && (
                <option value="" disabled={monthOptions.length > 0}>{monthPlaceholder}</option>
              )}
              {monthOptions.map((month) => (
                <option key={month.value} value={month.value}>{month.label}</option>
              ))}
            </select>
          </div>

          <div className="ga-field ga-field--game">
            <label className="form-label" htmlFor="ga-game">Game</label>
            <select
              id="ga-game"
              className="form-select"
              value={selectedGame}
              onChange={(e) => handleGameChange(e.target.value)}
              disabled={!selectedSeason || !selectedMonth || filteredGames.length === 0}
            >
              {(filteredGames.length === 0 || !selectedGame) && (
                <option value="" disabled={filteredGames.length > 0}>{gamePlaceholder}</option>
              )}
              {filteredGames.map((game) => (
                <option key={game.game_id} value={game.game_id}>{formatGameOptionLabel(game)}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className={`ga-body ${pageRefreshing ? 'is-refreshing' : ''}`} aria-busy={isBusy}>
        {decomposition
          ? renderScoreboard(
            {
              roadTeam: decomposition.road_team,
              homeTeam: decomposition.home_team,
              roadPts: decomposition.road_pts,
              homePts: decomposition.home_pts,
            },
            renderGameDetails()
          )
          : selectedGameIncomplete && selectedGameInfo && !loading && renderScoreboard({
            roadTeam: selectedGameInfo.road_team,
            homeTeam: selectedGameInfo.home_team,
            roadPts: selectedGameInfo.road_pts,
            homePts: selectedGameInfo.home_pts,
          })}

        <div className="ga-results">
          {selectedGameIncomplete && !loading && (
            <div className="ga-notice ga-notice--warning">
              <strong>Data incomplete.</strong> NBA.com hasn&apos;t published the advanced box score
              for this game yet, so possession-based stats and the factor analysis aren&apos;t available.
              They&apos;ll appear automatically after the next nightly data update.
            </div>
          )}

          {error && !(selectedGameIncomplete && error === 'Game not found in contributions') && (
            <div className="error">{error}</div>
          )}

          {!hasContent && (loading || initializing) && (
            <div className="loading">
              <div className="loading-spinner"></div>
              Loading analysis...
            </div>
          )}

          {showResults && analysisView === 'boxscore' && (
            <div className="ga-boxscore">
              <div className="ga-view-toolbar">
                <div
                  ref={periodScrollRef}
                  className={[
                    'ga-period-scroll',
                    periodFade.left ? 'has-more-left' : '',
                    periodFade.right ? 'has-more-right' : '',
                  ].filter(Boolean).join(' ')}
                  onScroll={updatePeriodFade}
                >
                  <SegmentedControl
                    size="sm"
                    className="ga-period-seg"
                    label="Box score period"
                    options={boxScoreSegmentOptions}
                    value={boxScoreSegment}
                    onChange={setBoxScoreSegment}
                  />
                </div>
              </div>

              {boxScoreError && !boxScoreLoading && (
                <div className="error">{boxScoreError}</div>
              )}

              {!boxScore && boxScoreLoading && (
                <div className="loading">
                  <div className="loading-spinner"></div>
                  Loading box score...
                </div>
              )}

              {boxScore && !boxScoreError && (
                <div className={`ga-boxscore-grid ${viewRefreshing ? 'is-refreshing' : ''}`}>
                  {renderBoxScoreTable(boxScore.road_players, roadBoxScoreTotals, homeBoxScoreTotals, 'road', boxScore.road_team)}
                  {renderBoxScoreTable(boxScore.home_players, homeBoxScoreTotals, roadBoxScoreTotals, 'home', boxScore.home_team)}
                </div>
              )}
            </div>
          )}

          {showResults && analysisView === 'factor' && (
            <div className="ga-factor">
              <div className="ga-view-toolbar">
                <SegmentedControl
                  className="ga-scope-seg"
                  label="Factor analysis data scope"
                  options={dataScopeOptions}
                  value={selectedDataScope}
                  onChange={handleDataScopeChange}
                />
              </div>

              {isLowSample && decomposition && (
                <div className="ga-notice ga-notice--warning" role="note">
                  <strong>Small sample.</strong> This game has only {formatMinutesValue(selectedScopeMinutes)} min
                  of {selectedDataScope === 'clutch' ? 'clutch time' : 'data in this scope'}, so the rates,
                  ratings and contributions below are mostly noise.
                </div>
              )}

              {scopeNotice && !selectedGameIncomplete && (
                <div className="ga-notice">{scopeNotice}</div>
              )}

              {!decomposition && loading && (
                <div className="loading">
                  <div className="loading-spinner"></div>
                  Loading analysis...
                </div>
              )}

              {decomposition && (
                <div
                  className={[
                    'ga-factor-content',
                    isLowSample ? 'is-low-sample' : '',
                    viewRefreshing ? 'is-refreshing' : '',
                  ].filter(Boolean).join(' ')}
                >
                  <div className="ga-analysis-grid">
                    <section className="card ga-stat-card" aria-labelledby="ga-factor-comparison">
                      <h2 className="card-title" id="ga-factor-comparison">Factor Comparison</h2>
                      <p className="ga-table-legend">
                        <span className="ga-legend-item"><span className="ga-swatch ga-swatch--better" aria-hidden="true" />Better than league avg</span>
                        <span className="ga-legend-item"><span className="ga-swatch ga-swatch--worse" aria-hidden="true" />Worse than league avg</span>
                      </p>
                      <div className="table-scroll">
                        <table className="ga-stat-table">
                          <thead>
                            <tr>
                              <th scope="col">Factor</th>
                              <th scope="col" className="text-center">
                                <span className="ga-team-marker ga-team-marker--road" aria-hidden="true" />{decomposition.road_team}
                              </th>
                              <th scope="col" className="text-center">
                                <span className="ga-team-marker ga-team-marker--home" aria-hidden="true" />{decomposition.home_team}
                              </th>
                              <th scope="col" className="text-center">Lg Avg</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td>Shooting (eFG%)</td>
                              <td className={`text-center ${getFactorCellClass(decomposition.road_factors.efg, 'efg')}`}>{formatFactorValue(decomposition.road_factors.efg)}</td>
                              <td className={`text-center ${getFactorCellClass(decomposition.home_factors.efg, 'efg')}`}>{formatFactorValue(decomposition.home_factors.efg)}</td>
                              <td className="text-center ga-league-avg">{formatFactorValue(decomposition.league_averages?.efg)}</td>
                            </tr>
                            <tr>
                              <td>Ball Handling</td>
                              <td className={`text-center ${getFactorCellClass(decomposition.road_factors.ball_handling, 'ball_handling')}`}>{formatFactorValue(decomposition.road_factors.ball_handling)}</td>
                              <td className={`text-center ${getFactorCellClass(decomposition.home_factors.ball_handling, 'ball_handling')}`}>{formatFactorValue(decomposition.home_factors.ball_handling)}</td>
                              <td className="text-center ga-league-avg">{formatFactorValue(decomposition.league_averages?.ball_handling)}</td>
                            </tr>
                            <tr>
                              <td>Off Rebounding</td>
                              <td className={`text-center ${getFactorCellClass(decomposition.road_factors.oreb, 'oreb')}`}>{formatFactorValue(decomposition.road_factors.oreb)}</td>
                              <td className={`text-center ${getFactorCellClass(decomposition.home_factors.oreb, 'oreb')}`}>{formatFactorValue(decomposition.home_factors.oreb)}</td>
                              <td className="text-center ga-league-avg">{formatFactorValue(decomposition.league_averages?.oreb)}</td>
                            </tr>
                            <tr>
                              <td>FT Rate</td>
                              <td className={`text-center ${getFactorCellClass(decomposition.road_factors.ft_rate, 'ft_rate')}`}>{formatFactorValue(decomposition.road_factors.ft_rate)}</td>
                              <td className={`text-center ${getFactorCellClass(decomposition.home_factors.ft_rate, 'ft_rate')}`}>{formatFactorValue(decomposition.home_factors.ft_rate)}</td>
                              <td className="text-center ga-league-avg">{formatFactorValue(decomposition.league_averages?.ft_rate)}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </section>

                    <section className="card ga-stat-card" aria-labelledby="ga-game-ratings">
                      <h2 className="card-title" id="ga-game-ratings">Game Ratings</h2>
                      <p className="ga-table-legend">
                        <span className="ga-legend-item"><span className="ga-swatch ga-swatch--better" aria-hidden="true" />Better than league avg</span>
                        <span className="ga-legend-item"><span className="ga-swatch ga-swatch--worse" aria-hidden="true" />Worse than league avg</span>
                      </p>
                      <div className="table-scroll">
                        <table className="ga-stat-table">
                          <thead>
                            <tr>
                              <th scope="col">Metric</th>
                              <th scope="col" className="text-center">
                                <span className="ga-team-marker ga-team-marker--road" aria-hidden="true" />{decomposition.road_team}
                              </th>
                              <th scope="col" className="text-center">
                                <span className="ga-team-marker ga-team-marker--home" aria-hidden="true" />{decomposition.home_team}
                              </th>
                              <th scope="col" className="text-center">Lg Avg</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td>Offensive Rating</td>
                              <td className={`text-center ${getRatingCellClass(decomposition.road_ratings.offensive_rating, 'off')}`}>{formatFactorValue(decomposition.road_ratings.offensive_rating)}</td>
                              <td className={`text-center ${getRatingCellClass(decomposition.home_ratings.offensive_rating, 'off')}`}>{formatFactorValue(decomposition.home_ratings.offensive_rating)}</td>
                              <td className="text-center ga-league-avg">{formatFactorValue(decomposition.league_averages?.off_rating)}</td>
                            </tr>
                            <tr>
                              <td>Defensive Rating</td>
                              <td className={`text-center ${getRatingCellClass(decomposition.road_ratings.defensive_rating, 'def')}`}>{formatFactorValue(decomposition.road_ratings.defensive_rating)}</td>
                              <td className={`text-center ${getRatingCellClass(decomposition.home_ratings.defensive_rating, 'def')}`}>{formatFactorValue(decomposition.home_ratings.defensive_rating)}</td>
                              <td className="text-center ga-league-avg">{formatFactorValue(decomposition.league_averages?.def_rating)}</td>
                            </tr>
                            <tr>
                              <td>Net Rating</td>
                              <td className={`text-center ${decomposition.road_ratings.net_rating >= 0 ? 'text-positive' : 'text-negative'}`}>
                                {decomposition.road_ratings.net_rating > 0 ? '+' : ''}{formatFactorValue(decomposition.road_ratings.net_rating)}
                              </td>
                              <td className={`text-center ${decomposition.home_ratings.net_rating >= 0 ? 'text-positive' : 'text-negative'}`}>
                                {decomposition.home_ratings.net_rating > 0 ? '+' : ''}{formatFactorValue(decomposition.home_ratings.net_rating)}
                              </td>
                              <td className="text-center ga-league-avg">0.0</td>
                            </tr>
                            <tr>
                              <td>Pace</td>
                              <td className="text-center">{formatFactorValue(decomposition.road_ratings.pace ?? decomposition.road_ratings.possessions)}</td>
                              <td className="text-center">{formatFactorValue(decomposition.home_ratings.pace ?? decomposition.home_ratings.possessions)}</td>
                              <td className="text-center ga-league-avg">{formatFactorValue(decomposition.league_averages?.pace)}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </section>
                  </div>

                  <section className="card ga-contributions" aria-labelledby="ga-contributions-title">
                    <div className="ga-chart-header">
                      <h2 className="card-title" id="ga-contributions-title">Factor Contributions to Rating Differential</h2>
                      <div className="ga-chart-legend">
                        <span className="ga-legend-item">
                          <span className="ga-swatch ga-swatch--road" aria-hidden="true" />
                          Helps {decomposition.road_team}
                        </span>
                        <span className="ga-legend-item">
                          <span className="ga-swatch ga-swatch--home" aria-hidden="true" />
                          Helps {decomposition.home_team}
                        </span>
                        <span className="ga-legend-item">
                          <span className="ga-swatch ga-swatch--error" aria-hidden="true" />
                          Error
                        </span>
                        <span className="ga-legend-item">▲▼ above / below league avg</span>
                      </div>
                    </div>

                    <div className="ga-chart">
                      <ResponsiveContainer width="100%" height={isCompactChart ? 420 : 440}>
                        <BarChart
                          data={contributionChartData}
                          layout="vertical"
                          margin={{ top: 4, right: contributionChartRight, left: 0, bottom: 0 }}
                          barCategoryGap="22%"
                        >
                          <CartesianGrid horizontal={false} stroke="var(--chart-grid)" />
                          <XAxis
                            type="number"
                            domain={contributionXAxisConfig.domain}
                            ticks={contributionXAxisConfig.ticks}
                            tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }}
                            axisLine={{ stroke: 'var(--chart-grid)' }}
                            tickLine={false}
                          />
                          <YAxis
                            type="category"
                            dataKey="factor"
                            tick={renderContributionTick}
                            tickLine={false}
                            axisLine={false}
                            width={contributionYAxisWidth}
                            interval={0}
                          />
                          <Tooltip
                            cursor={{ fill: 'var(--color-button-hover-muted)' }}
                            content={renderContributionTooltip}
                            isAnimationActive={false}
                          />
                          <ReferenceLine x={0} stroke="var(--chart-reference)" />
                          <ReferenceLine y={CONTRIBUTION_DIVIDER_KEY} stroke="var(--color-border)" />
                          <Bar dataKey="value" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                            {contributionChartData.map((entry) => (
                              <Cell key={entry.factor} fill={entry.fill} fillOpacity={entry.fillOpacity} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <div
                        className="ga-axis-captions"
                        style={{ paddingLeft: contributionYAxisWidth, paddingRight: contributionChartRight }}
                      >
                        <span className="ga-text-road">← Helps {decomposition.road_team}</span>
                        <span className="ga-text-home">Helps {decomposition.home_team} →</span>
                      </div>
                    </div>

                    {interpretationEnabled && (
                      <div className="ga-interpretation">
                        <h3 className="ga-interpretation-title">AI-generated contribution analysis</h3>
                        {interpretationLoading && (
                          <div className="refresh-indicator">
                            <span className="loading-spinner" aria-hidden="true" />
                            Generating analysis...
                          </div>
                        )}
                        {interpretation && !interpretationLoading && (
                          <div className="ga-interpretation-text">
                            {interpretation.includes('-') ? (
                              <ul>
                                {interpretation
                                  .split(/\n|(?=- )/)
                                  .map(line => line.trim())
                                  .filter(line => line.startsWith('-') || line.startsWith('•'))
                                  .map((line, i) => (
                                    <li key={i}>{line.replace(/^[-•]\s*/, '')}</li>
                                  ))}
                              </ul>
                            ) : (
                              <p>{interpretation}</p>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </section>

                  <section className="ga-glossary card">
                    <button
                      type="button"
                      className="ga-glossary-toggle"
                      onClick={() => setGlossaryExpanded(!glossaryExpanded)}
                      aria-expanded={glossaryExpanded}
                    >
                      <span>Glossary</span>
                      <span className="ga-glossary-icon" aria-hidden="true">{glossaryExpanded ? '−' : '+'}</span>
                    </button>

                    {glossaryExpanded && (
                      <div className="ga-glossary-content">
                        <div className="ga-glossary-grid">
                          <div className="ga-glossary-group">
                            <h3>Four Factors</h3>
                            <dl>
                              <dt>Shooting (eFG%)</dt>
                              <dd>Effective field goal percentage, which adjusts for the added value of three-pointers. Formula: (FGM + 0.5 × 3PM) / FGA × 100</dd>

                              <dt>Ball Handling</dt>
                              <dd>A team's ability to take care of the ball, calculated as 100 − TOV%. Higher is better. TOV% = TOV / Possessions × 100</dd>

                              <dt>Off Rebounding (OREB%)</dt>
                              <dd>The percentage of available offensive rebounds a team grabs. Formula: OREB / (OREB + OPP_DREB) × 100</dd>

                              <dt>FT Rate (Free Throw Rate)</dt>
                              <dd>Measures how often a team gets to the free throw line relative to field goal attempts. Formula: FTM / FGA × 100</dd>
                            </dl>
                          </div>

                          <div className="ga-glossary-group">
                            <h3>Ratings</h3>
                            <dl>
                              <dt>Offensive Rating (ORtg)</dt>
                              <dd>Points scored per 100 possessions. Formula: (Points / Possessions) × 100</dd>

                              <dt>Defensive Rating (DRtg)</dt>
                              <dd>Points allowed per 100 possessions. Formula: (Opponent Points / Opponent Possessions) × 100. Lower is better.</dd>

                              <dt>Net Rating (NRtg)</dt>
                              <dd>The difference between Offensive Rating and Defensive Rating. Formula: ORtg − DRtg</dd>

                              <dt>Pace</dt>
                              <dd>Number of possessions per 48 minutes. Formula: Avg Possessions × (48 / Actual Game Minutes). For standard games (48 min), Pace equals average possessions.</dd>
                            </dl>
                          </div>

                          <div className="ga-glossary-group ga-glossary-group--full">
                            <h3>How the Model Works</h3>
                            <p>
                              This tool uses a linear regression model trained on historical NBA games to predict the <strong>net rating differential</strong> (per 100 possessions) based on the Four Factors. Since the Four Factors are rate statistics (efficiency measures), the model predicts a rate outcome rather than raw point margin.
                            </p>
                            <ol>
                              <li><strong>Factor Differentials:</strong> For each of the Four Factors, we calculate the difference between the home team and road team performance.</li>
                              <li><strong>Weighted Contributions:</strong> Each differential is multiplied by a coefficient that represents its importance in predicting efficiency outcomes. These coefficients are learned from historical data.</li>
                              <li><strong>Predicted Rating Differential:</strong> The sum of all weighted contributions gives us the predicted net rating differential (points per 100 possessions).</li>
                            </ol>
                            <p>
                              <strong>Four Factors Mode:</strong> Uses differentials (Home − Road) for each factor.
                            </p>
                            <p>
                              <strong>Eight Factors Mode:</strong> Analyzes home and road team factors separately, centering each around league averages for more nuanced analysis.
                            </p>
                            <p className="ga-model-formula">
                              Formula: <code>Predicted Rating Diff = Intercept + Σ(Coefficient × Factor Differential)</code>
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </section>
                </div>
              )}
            </div>
          )}

          {showResults && analysisView === 'timeline' && (
            <div className="ga-timeline">
              {!timelineData && (timelineLoading || (loading && !timelineError)) && (
                <div className="loading">
                  <div className="loading-spinner"></div>
                  Loading game timeline...
                </div>
              )}
              {timelineError && !timelineLoading && (
                <div className="error">{timelineError}</div>
              )}
              {timelineData && !timelineError && (
                <div className={viewRefreshing ? 'is-refreshing' : ''}>
                  <GameTimeline timeline={timelineData} />
                </div>
              )}
            </div>
          )}

          {!selectedGame && !hasContent && !loading && !initializing && (
            <div className="ga-empty-state card">
              Select a game to load the available Game Analysis views.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default FourFactor

