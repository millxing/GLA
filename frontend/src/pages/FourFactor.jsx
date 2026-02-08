import { useState, useEffect, useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'
import { getSeasons, getGames, getDecomposition, getInterpretation } from '../api'
import { usePersistedState } from '../hooks/usePersistedState'
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

// Temporary kill-switch for LLM summaries without removing implementation code.
const AI_SUMMARIES_ENABLED = false
const AI_SUMMARY_MAINTENANCE_MESSAGE = 'AI game summaries closed for renovations'

function FourFactor() {
  const [seasons, setSeasons] = useState([])
  const [games, setGames] = useState([])
  const [selectedSeason, setSelectedSeason] = usePersistedState('fourfactor_season', '')
  const [selectedGame, setSelectedGame] = useState('')
  const [factorType, setFactorType] = usePersistedState('fourfactor_factortype', 'eight_factors')
  const [decomposition, setDecomposition] = useState(null)
  const [loading, setLoading] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [error, setError] = useState(null)
  const [glossaryExpanded, setGlossaryExpanded] = useState(false)
  const [interpretation, setInterpretation] = useState(null)
  const [interpretationLoading, setInterpretationLoading] = useState(false)

  useEffect(() => {
    let isCurrent = true
    async function loadInitialData() {
      try {
        const seasonsRes = await getSeasons()
        if (!isCurrent) return
        setSeasons(seasonsRes.seasons)
        // Keep persisted season if valid, otherwise default to first
        setSelectedSeason(prev => {
          if (prev && seasonsRes.seasons.includes(prev)) return prev
          return seasonsRes.seasons.length > 0 ? seasonsRes.seasons[0] : ''
        })
      } catch (err) {
        if (isCurrent) setError(err.message)
      }
    }
    loadInitialData()
    return () => { isCurrent = false }
  }, [])

  useEffect(() => {
    let isCurrent = true
    async function loadGames() {
      if (!selectedSeason) return
      setInitializing(true)
      if (isCurrent) {
        setDecomposition(null)
        setSelectedGame('')
      }
      try {
        const gamesRes = await getGames(selectedSeason)
        if (!isCurrent) return
        setGames(gamesRes.games)
        // Default to most recent game (first in list, sorted by date descending)
        if (gamesRes.games.length > 0) {
          setSelectedGame(gamesRes.games[0].game_id)
        } else {
          setSelectedGame('')
        }
      } catch (err) {
        if (isCurrent) setError(err.message)
      } finally {
        if (isCurrent) setInitializing(false)
      }
    }
    loadGames()
    return () => { isCurrent = false }
  }, [selectedSeason])

  useEffect(() => {
    let isCurrent = true
    async function loadDecomposition() {
      if (!selectedSeason || !selectedGame) return
      setLoading(true)
      if (isCurrent) setError(null)
      try {
        const data = await getDecomposition(selectedSeason, selectedGame, factorType)
        if (isCurrent) setDecomposition(data)
      } catch (err) {
        if (isCurrent) setError(err.message)
      } finally {
        if (isCurrent) setLoading(false)
      }
    }
    loadDecomposition()
    return () => { isCurrent = false }
  }, [selectedSeason, selectedGame, factorType])

  useEffect(() => {
    let isCurrent = true
    async function loadInterpretation() {
      if (AI_SUMMARIES_ENABLED === false) {
        if (isCurrent) {
          setInterpretation(AI_SUMMARY_MAINTENANCE_MESSAGE)
          setInterpretationLoading(false)
        }
        return
      }

      if (!decomposition || factorType !== 'eight_factors') {
        if (isCurrent) {
          setInterpretation(null)
          setInterpretationLoading(false)
        }
        return
      }

      setInterpretationLoading(true)
      try {
        const data = await getInterpretation(decomposition, factorType, selectedSeason)
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
  }, [decomposition, factorType])

  const getContributionChartData = () => {
    if (!decomposition) return []
    const contributions = decomposition.contributions

    // Helper to determine Good/Bad prefix for Eight Factors mode
    const getQualityPrefix = (factorKey) => {
      if (!decomposition.league_averages) return ''

      // Map factor keys to the corresponding factors object key and league avg key
      const factorMapping = {
        'home_shooting': { factors: 'home_factors', key: 'efg', avgKey: 'efg' },
        'road_shooting': { factors: 'road_factors', key: 'efg', avgKey: 'efg' },
        'home_ball_handling': { factors: 'home_factors', key: 'ball_handling', avgKey: 'ball_handling' },
        'road_ball_handling': { factors: 'road_factors', key: 'ball_handling', avgKey: 'ball_handling' },
        'home_orebounding': { factors: 'home_factors', key: 'oreb', avgKey: 'oreb' },
        'road_orebounding': { factors: 'road_factors', key: 'oreb', avgKey: 'oreb' },
        'home_free_throws': { factors: 'home_factors', key: 'ft_rate', avgKey: 'ft_rate' },
        'road_free_throws': { factors: 'road_factors', key: 'ft_rate', avgKey: 'ft_rate' },
      }

      const mapping = factorMapping[factorKey]
      if (!mapping) return ''

      const factorValue = decomposition[mapping.factors]?.[mapping.key]
      const leagueAvg = decomposition.league_averages[mapping.avgKey]

      if (factorValue === undefined || leagueAvg === undefined) return ''

      return factorValue >= leagueAvg ? 'Good ' : 'Bad '
    }

    // Map factor names to display labels with team abbreviations
    // For Eight Factors, we use a two-line format: "Good/Bad TEAM" on line 1, stat name on line 2
    const factorLabels = {
      'shooting': { line1: 'Shooting', line2: null },
      'ball_handling': { line1: 'Ball Handling', line2: null },
      'orebounding': { line1: 'Off Rebounds', line2: null },
      'free_throws': { line1: 'FT Rate', line2: null },
      // Eight factors mode - split into two lines for better readability
      'home_shooting': { line1: `${getQualityPrefix('home_shooting')}${decomposition.home_team}`, line2: 'Shooting' },
      'road_shooting': { line1: `${getQualityPrefix('road_shooting')}${decomposition.road_team}`, line2: 'Shooting' },
      'home_ball_handling': { line1: `${getQualityPrefix('home_ball_handling')}${decomposition.home_team}`, line2: 'Ball Handling' },
      'road_ball_handling': { line1: `${getQualityPrefix('road_ball_handling')}${decomposition.road_team}`, line2: 'Ball Handling' },
      'home_orebounding': { line1: `${getQualityPrefix('home_orebounding')}${decomposition.home_team}`, line2: 'Off Rebounds' },
      'road_orebounding': { line1: `${getQualityPrefix('road_orebounding')}${decomposition.road_team}`, line2: 'Off Rebounds' },
      'home_free_throws': { line1: `${getQualityPrefix('home_free_throws')}${decomposition.home_team}`, line2: 'FT Rate' },
      'road_free_throws': { line1: `${getQualityPrefix('road_free_throws')}${decomposition.road_team}`, line2: 'FT Rate' },
    }

    // Map contribution keys to actual factor values
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

    // Build factor bars sorted by value descending (most positive at top)
    const factorBars = Object.entries(contributions)
      .map(([factor, value]) => {
        const label = factorLabels[factor] || { line1: factor, line2: null }
        return {
          factor: label.line2 ? `${label.line1}\n${label.line2}` : label.line1,
          labelLine1: label.line1,
          labelLine2: label.line2,
          factorKey: factor,
          value,
          actualValue: getFactorValue(factor),
          fill: value >= 0 ? 'var(--color-positive)' : 'var(--color-negative)',
        }
      })
      .sort((a, b) => b.value - a.value)

    // Add error bar at the end (always at bottom, not sorted)
    // Error = actual rating diff - predicted rating diff
    const error = decomposition.actual_rating_diff - decomposition.predicted_rating_diff
    factorBars.push({
      factor: 'Error',
      labelLine1: 'Error',
      labelLine2: null,
      value: error,
      fill: 'var(--color-neutral)',
    })

    return factorBars
  }

  // Calculate X-axis ticks for contribution chart (every 1 point)
  const contributionXAxisConfig = useMemo(() => {
    const chartData = getContributionChartData()
    if (chartData.length === 0) return { domain: [-5, 5], ticks: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] }

    const values = chartData.map(d => d.value)
    const minVal = Math.min(...values)
    const maxVal = Math.max(...values)

    // Round to nearest integer, with some padding
    const tickMin = Math.floor(minVal) - 1
    const tickMax = Math.ceil(maxVal) + 1

    const ticks = []
    for (let t = tickMin; t <= tickMax; t++) {
      ticks.push(t)
    }

    return { domain: [tickMin, tickMax], ticks }
  }, [decomposition, factorType])

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

  return (
    <div className="four-factor container">
      <h1 className="page-title">Game Analysis</h1>
      <p className="page-description">
        Analyze how each of Dean Oliver's Four Factors contributed to the game outcome.
      </p>

      <div className="controls card">
        <div className="form-row">
          <div className="form-group">
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

          <div className="form-group">
            <label className="form-label">Game</label>
            <select
              className="form-select"
              value={selectedGame}
              onChange={(e) => setSelectedGame(e.target.value)}
              disabled={!selectedSeason || games.length === 0}
            >
              <option value="">Select game...</option>
              {games.map((game) => (
                <option key={game.game_id} value={game.game_id}>{game.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Factor Type</label>
            <select
              className="form-select"
              value={factorType}
              onChange={(e) => setFactorType(e.target.value)}
            >
              <option value="four_factors">Four Factors</option>
              <option value="eight_factors">Eight Factors</option>
            </select>
          </div>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {(loading || initializing) && (
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading analysis...
        </div>
      )}

      {decomposition && !loading && !initializing && (
        <div className="results">
          <div className="game-header card">
            <div className="game-header-content">
              <div className="matchup-left">
                <div className="team road-team">
                  <span className="team-abbr">{toCityName(decomposition.road_team)}</span>
                  <span className="team-score">{decomposition.road_pts}</span>
                </div>
                <div className="at-symbol">@</div>
                <div className="team home-team">
                  <span className="team-abbr">{toCityName(decomposition.home_team)}</span>
                  <span className="team-score">{decomposition.home_pts}</span>
                </div>
              </div>
              {decomposition.linescore && (
                <div className="linescore-center">
                  <table className="linescore-table">
                    <thead>
                      <tr>
                        <th></th>
                        <th>Q1</th>
                        <th>Q2</th>
                        <th>Q3</th>
                        <th>Q4</th>
                        {decomposition.is_overtime && <th>OT</th>}
                        <th>Total</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="team-cell">{decomposition.road_team}</td>
                        <td>{decomposition.linescore.road.q1}</td>
                        <td>{decomposition.linescore.road.q2}</td>
                        <td>{decomposition.linescore.road.q3}</td>
                        <td>{decomposition.linescore.road.q4}</td>
                        {decomposition.is_overtime && <td>{decomposition.linescore.road.ot}</td>}
                        <td className="total-cell">{decomposition.road_pts}</td>
                      </tr>
                      <tr>
                        <td className="team-cell">{decomposition.home_team}</td>
                        <td>{decomposition.linescore.home.q1}</td>
                        <td>{decomposition.linescore.home.q2}</td>
                        <td>{decomposition.linescore.home.q3}</td>
                        <td>{decomposition.linescore.home.q4}</td>
                        {decomposition.is_overtime && <td>{decomposition.linescore.home.ot}</td>}
                        <td className="total-cell">{decomposition.home_pts}</td>
                      </tr>
                    </tbody>
                  </table>
                  {decomposition.is_overtime && decomposition.overtime_count > 0 && (
                    <span className="ot-indicator">
                      {decomposition.overtime_count === 1 ? 'OT' : `${decomposition.overtime_count}OT`}
                    </span>
                  )}
                </div>
              )}
              <div className="game-info-right">
                <div className="game-info-details">
                  <div className="game-date">{decomposition.game_date}</div>
                  {decomposition.game_type && (
                    <div className="game-type">{formatGameType(decomposition.game_type)}</div>
                  )}
                </div>
                <div className="external-links">
                  <a
                    href={`https://www.basketball-reference.com/boxscores/${decomposition.game_date.replace(/-/g, '')}0${toBBRefTeam(decomposition.home_team)}.html`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link"
                  >
                    BBRef Box Score
                  </a>
                  <a
                    href={`https://www.nba.com/game/${decomposition.road_team.toLowerCase()}-vs-${decomposition.home_team.toLowerCase()}-${String(decomposition.game_id).padStart(10, '0')}/box-score`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="external-link"
                  >
                    NBA.com Box Score
                  </a>
                </div>
              </div>
            </div>
          </div>

          <div className="analysis-grid">
            <div className="factors-table card">
              <h2 className="card-title">Factor Comparison</h2>
              <p className="table-legend">
                <span className="legend-item"><span className="legend-swatch bg-positive"></span> Better than league avg</span>
                <span className="legend-item"><span className="legend-swatch bg-negative"></span> Worse than league avg</span>
              </p>
              <table>
                <thead>
                  <tr>
                    <th>Factor</th>
                    <th className="text-center">{decomposition.road_team}</th>
                    <th className="text-center">{decomposition.home_team}</th>
                    <th className="text-center">Lg Avg</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Shooting (eFG%)</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.road_factors.efg, 'efg')}`}>{formatFactorValue(decomposition.road_factors.efg)}</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.home_factors.efg, 'efg')}`}>{formatFactorValue(decomposition.home_factors.efg)}</td>
                    <td className="text-center text-secondary">{formatFactorValue(decomposition.league_averages?.efg)}</td>
                  </tr>
                  <tr>
                    <td>Ball Handling</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.road_factors.ball_handling, 'ball_handling')}`}>{formatFactorValue(decomposition.road_factors.ball_handling)}</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.home_factors.ball_handling, 'ball_handling')}`}>{formatFactorValue(decomposition.home_factors.ball_handling)}</td>
                    <td className="text-center text-secondary">{formatFactorValue(decomposition.league_averages?.ball_handling)}</td>
                  </tr>
                  <tr>
                    <td>Off Rebounding</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.road_factors.oreb, 'oreb')}`}>{formatFactorValue(decomposition.road_factors.oreb)}</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.home_factors.oreb, 'oreb')}`}>{formatFactorValue(decomposition.home_factors.oreb)}</td>
                    <td className="text-center text-secondary">{formatFactorValue(decomposition.league_averages?.oreb)}</td>
                  </tr>
                  <tr>
                    <td>FT Rate</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.road_factors.ft_rate, 'ft_rate')}`}>{formatFactorValue(decomposition.road_factors.ft_rate)}</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.home_factors.ft_rate, 'ft_rate')}`}>{formatFactorValue(decomposition.home_factors.ft_rate)}</td>
                    <td className="text-center text-secondary">{formatFactorValue(decomposition.league_averages?.ft_rate)}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="ratings-table card">
              <h2 className="card-title">Game Ratings</h2>
              <p className="table-legend">
                <span className="legend-item"><span className="legend-swatch bg-positive"></span> Better than league avg</span>
                <span className="legend-item"><span className="legend-swatch bg-negative"></span> Worse than league avg</span>
              </p>
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th className="text-center">{decomposition.road_team}</th>
                    <th className="text-center">{decomposition.home_team}</th>
                    <th className="text-center">Lg Avg</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Offensive Rating</td>
                    <td className={`text-center ${getRatingCellClass(decomposition.road_ratings.offensive_rating, 'off')}`}>{formatFactorValue(decomposition.road_ratings.offensive_rating)}</td>
                    <td className={`text-center ${getRatingCellClass(decomposition.home_ratings.offensive_rating, 'off')}`}>{formatFactorValue(decomposition.home_ratings.offensive_rating)}</td>
                    <td className="text-center text-secondary">{formatFactorValue(decomposition.league_averages?.off_rating)}</td>
                  </tr>
                  <tr>
                    <td>Defensive Rating</td>
                    <td className={`text-center ${getRatingCellClass(decomposition.road_ratings.defensive_rating, 'def')}`}>{formatFactorValue(decomposition.road_ratings.defensive_rating)}</td>
                    <td className={`text-center ${getRatingCellClass(decomposition.home_ratings.defensive_rating, 'def')}`}>{formatFactorValue(decomposition.home_ratings.defensive_rating)}</td>
                    <td className="text-center text-secondary">{formatFactorValue(decomposition.league_averages?.def_rating)}</td>
                  </tr>
                  <tr>
                    <td>Net Rating</td>
                    <td className={`text-center ${decomposition.road_ratings.net_rating >= 0 ? 'text-positive' : 'text-negative'}`}>
                      {decomposition.road_ratings.net_rating > 0 ? '+' : ''}{formatFactorValue(decomposition.road_ratings.net_rating)}
                    </td>
                    <td className={`text-center ${decomposition.home_ratings.net_rating >= 0 ? 'text-positive' : 'text-negative'}`}>
                      {decomposition.home_ratings.net_rating > 0 ? '+' : ''}{formatFactorValue(decomposition.home_ratings.net_rating)}
                    </td>
                    <td className="text-center text-secondary">0.0</td>
                  </tr>
                  <tr>
                    <td>Pace</td>
                    <td className="text-center">{formatFactorValue(decomposition.road_ratings.pace ?? decomposition.road_ratings.possessions)}</td>
                    <td className="text-center">{formatFactorValue(decomposition.home_ratings.pace ?? decomposition.home_ratings.possessions)}</td>
                    <td className="text-center text-secondary">{formatFactorValue(decomposition.league_averages?.pace)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="contributions-chart card">
            <div className="chart-header">
              <h2 className="card-title">Factor Contributions to Rating Differential</h2>
              <div className="chart-legend">
                <span className="legend-item">
                  <span className="legend-swatch bg-positive"></span>
                  Helps {decomposition.home_team}
                </span>
                <span className="legend-item">
                  <span className="legend-swatch bg-negative"></span>
                  Helps {decomposition.road_team}
                </span>
              </div>
            </div>

            <div className="chart-container">
              <ResponsiveContainer width="100%" height={550}>
                <BarChart
                  data={getContributionChartData()}
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    type="number"
                    domain={contributionXAxisConfig.domain}
                    ticks={contributionXAxisConfig.ticks}
                    tick={{ fill: 'var(--color-text-secondary)' }}
                  />
                  <YAxis
                    type="category"
                    dataKey="factor"
                    tick={(props) => {
                      const { x, y, payload } = props
                      // Find the data entry to get line1/line2
                      const chartData = getContributionChartData()
                      const entry = chartData.find(d => d.factor === payload.value)
                      const line1 = entry?.labelLine1 || payload.value
                      const line2 = entry?.labelLine2

                      if (line2) {
                        return (
                          <text x={x} y={y} textAnchor="end" fill="var(--color-text-secondary)" fontSize={14}>
                            <tspan x={x} dy="-0.35em">{line1}</tspan>
                            <tspan x={x} dy="1.3em">{line2}</tspan>
                          </text>
                        )
                      }
                      return (
                        <text x={x} y={y} textAnchor="end" fill="var(--color-text-secondary)" fontSize={14} dy="0.35em">
                          {line1}
                        </text>
                      )
                    }}
                    width={120}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--color-background)',
                      border: '1px solid var(--color-border)',
                      borderRadius: 'var(--border-radius-sm)',
                    }}
                    content={({ active, payload }) => {
                      if (!active || !payload || !payload.length) return null
                      const data = payload[0].payload
                      return (
                        <div style={{
                          backgroundColor: 'var(--color-background)',
                          border: '1px solid var(--color-border)',
                          borderRadius: 'var(--border-radius-sm)',
                          padding: '8px 12px',
                        }}>
                          <div style={{ fontWeight: 600, marginBottom: '4px' }}>
                          {data.labelLine2 ? `${data.labelLine1} ${data.labelLine2}` : data.labelLine1}
                        </div>
                          {data.actualValue && (
                            <div style={{ color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
                              Value: {data.actualValue}
                            </div>
                          )}
                          <div>Contribution: {data.value.toFixed(1)}</div>
                        </div>
                      )
                    }}
                  />
                  <ReferenceLine x={0} stroke="var(--color-neutral)" />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {getContributionChartData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} stroke="#000000" strokeWidth={1} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-axis-labels">
              <span className="axis-label-left">← Helps {decomposition.road_team}</span>
              <span className="axis-label-right">Helps {decomposition.home_team} →</span>
            </div>

            {/* AI Interpretation */}
            <div className="interpretation-box">
              {interpretationLoading && (
                <div className="interpretation-loading">
                  <span className="interpretation-spinner"></span>
                  Generating analysis...
                </div>
              )}
              {interpretation && !interpretationLoading && (
                <div className="interpretation-content">
                  <div className="interpretation-text">
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
                </div>
              )}
            </div>
          </div>

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
                <div className="glossary-grid">
                  <div className="glossary-section-group">
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

                  <div className="glossary-section-group">
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

                  <div className="glossary-section-group full-width">
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
                    <p className="model-formula">
                      Formula: <code>Predicted Rating Diff = Intercept + Σ(Coefficient × Factor Differential)</code>
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default FourFactor
