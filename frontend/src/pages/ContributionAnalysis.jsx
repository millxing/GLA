import { useState, useEffect, useMemo } from 'react'
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
} from 'recharts'
import { getSeasons, getTeams, getLeagueSummary, getSeasonModels, getContributionAnalysis } from '../api'
import { getTeamName } from '../constants/teams'
import './ContributionAnalysis.css'

const DATE_RANGE_OPTIONS = [
  { value: 'season', label: 'Season-to-Date' },
  { value: 'season_no_playoffs', label: 'Season-to-Date, No Playoffs' },
  { value: 'month', label: 'Month-to-Date' },
  { value: 'last_10', label: 'Last 10 Games' },
  { value: 'last_20', label: 'Last 20 Games' },
  { value: 'last_30', label: 'Last 30 Games' },
  { value: 'custom', label: 'Custom Date Range' },
]

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

// Factors where lower values are better (opponent stats)
const LOWER_IS_BETTER_FACTORS = new Set([
  'opp_shooting',
  'opp_ball_handling',
  'opp_orebounding',
  'opp_free_throws',
])

function ContributionAnalysis() {
  const [seasons, setSeasons] = useState([])
  const [teams, setTeams] = useState([])
  const [models, setModels] = useState([])
  const [selectedSeason, setSelectedSeason] = usePersistedState('contribution_season', '2025-26')
  const [selectedTeam, setSelectedTeam] = usePersistedState('contribution_team', 'BOS')
  const [selectedModel, setSelectedModel] = usePersistedState('contribution_model', 'season_2018-2025')
  const [dateRangeType, setDateRangeType] = usePersistedState('contribution_daterange', 'season')
  const [customStartDate, setCustomStartDate] = useState('')
  const [customEndDate, setCustomEndDate] = useState('')
  const [customDatesInitialized, setCustomDatesInitialized] = useState(false)
  const [seasonBounds, setSeasonBounds] = useState({ first: '', last: '' })
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [glossaryExpanded, setGlossaryExpanded] = useState(false)

  // Load seasons and models on mount
  useEffect(() => {
    async function loadInitialData() {
      try {
        const [seasonsRes, modelsRes] = await Promise.all([
          getSeasons(),
          getSeasonModels(),
        ])
        setSeasons(seasonsRes.seasons)
        setModels(modelsRes.models)

        // Default to most recent season if 2025-26 not available
        if (seasonsRes.seasons.length > 0 && !seasonsRes.seasons.includes('2025-26')) {
          setSelectedSeason(seasonsRes.seasons[0])
        }

        // Default to first model if season_2018-2025 not available
        if (modelsRes.models.length > 0) {
          const defaultModel = modelsRes.models.find(m => m.id === 'season_2018-2025')
          if (!defaultModel) {
            setSelectedModel(modelsRes.models[0].id)
          }
        }
      } catch (err) {
        setError(err.message)
      }
    }
    loadInitialData()
  }, [])

  // Load teams and season bounds when season changes
  useEffect(() => {
    async function loadTeamsAndBounds() {
      if (!selectedSeason) return
      try {
        const [teamsRes, summaryRes] = await Promise.all([
          getTeams(selectedSeason),
          getLeagueSummary(selectedSeason, null, null, false),
        ])
        setTeams(teamsRes.teams)
        // Keep persisted team if it exists in this season, otherwise default to BOS or first team
        setSelectedTeam(prevTeam => {
          if (teamsRes.teams.includes(prevTeam)) return prevTeam
          if (teamsRes.teams.includes('BOS')) return 'BOS'
          if (teamsRes.teams.length > 0) return teamsRes.teams[0]
          return ''
        })

        // Extract season date bounds from league summary (actual data, not scheduled games)
        if (summaryRes.first_game_date && summaryRes.last_game_date) {
          setSeasonBounds({ first: summaryRes.first_game_date, last: summaryRes.last_game_date })
        }

        setData(null)
      } catch (err) {
        setError(err.message)
      }
    }
    loadTeamsAndBounds()
  }, [selectedSeason])

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

  // Calculate API parameters based on date range preset (like LeagueSummary)
  const { apiRangeType, apiStartDate, apiEndDate, apiLastNGames, apiExcludePlayoffs } = useMemo(() => {
    // Default: season-to-date
    let rangeType = 'season'
    let startDate = null
    let endDate = null
    let lastNGames = null
    let excludePlayoffs = false

    if (dateRangeType.startsWith('last_')) {
      rangeType = 'last_n'
      lastNGames = parseInt(dateRangeType.split('_')[1])
    } else if (dateRangeType === 'custom') {
      if (customStartDate && customEndDate) {
        rangeType = 'custom'
        startDate = customStartDate
        endDate = customEndDate
      }
    } else if (dateRangeType === 'season_no_playoffs') {
      rangeType = 'season'
      excludePlayoffs = true
    } else if (dateRangeType === 'month') {
      if (seasonBounds.last) {
        // Calculate first day of the month containing the last game
        // Parse the date string directly to avoid timezone issues
        const [year, month] = seasonBounds.last.split('-')
        const monthStartStr = `${year}-${month}-01`
        // Use season start if month start is before season start
        startDate = monthStartStr < seasonBounds.first ? seasonBounds.first : monthStartStr
        endDate = seasonBounds.last
        rangeType = 'custom'
      }
    }

    return {
      apiRangeType: rangeType,
      apiStartDate: startDate,
      apiEndDate: endDate,
      apiLastNGames: lastNGames,
      apiExcludePlayoffs: excludePlayoffs,
    }
  }, [dateRangeType, customStartDate, customEndDate, seasonBounds.first, seasonBounds.last])

  // Load contribution analysis when parameters change
  useEffect(() => {
    async function loadAnalysis() {
      if (!selectedSeason || !selectedTeam || !selectedModel) return

      // Don't load if custom dates are selected but not filled in
      if (dateRangeType === 'custom' && (!customStartDate || !customEndDate)) return

      // Don't load month-to-date if we don't have the calculated dates yet
      if (dateRangeType === 'month' && !apiEndDate) return

      setLoading(true)
      setError(null)

      try {
        const res = await getContributionAnalysis(
          selectedSeason,
          selectedTeam,
          selectedModel,
          apiRangeType,
          apiLastNGames,
          apiStartDate,
          apiEndDate,
          apiExcludePlayoffs
        )
        setData(res)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadAnalysis()
  }, [selectedSeason, selectedTeam, selectedModel, dateRangeType, customStartDate, customEndDate, apiRangeType, apiStartDate, apiEndDate, apiLastNGames, apiExcludePlayoffs])

  // Prepare main chart data
  const mainChartData = useMemo(() => {
    if (!data) return []

    // Net Rating bar (blue)
    const chartData = [
      {
        name: 'Net Rating',
        value: data.net_rating,
        fill: 'var(--color-primary)',
      },
    ]

    // Factor contributions sorted from most positive to most negative
    const contributionBars = FACTOR_ORDER.map((factor) => {
      const contribution = data.contributions[factor] || 0
      return {
        name: FACTOR_LABELS[factor] || factor,
        value: contribution,
        fill: contribution >= 0 ? 'var(--color-positive)' : 'var(--color-negative)',
      }
    }).sort((a, b) => b.value - a.value)

    return [...chartData, ...contributionBars]
  }, [data])

  // Y-axis config for main chart
  const mainChartYAxis = useMemo(() => {
    if (mainChartData.length === 0) return { domain: [-5, 5], ticks: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] }

    const values = mainChartData.map(d => d.value)
    const minVal = Math.min(...values)
    const maxVal = Math.max(...values)

    // Minimal padding - just round to nearest integer
    const tickMin = Math.floor(minVal)
    const tickMax = Math.ceil(maxVal)

    const ticks = []
    for (let t = tickMin; t <= tickMax; t += 1) {
      ticks.push(t)
    }

    return { domain: [tickMin, tickMax], ticks }
  }, [mainChartData])

  return (
    <div className="contribution-analysis container">
      <h1 className="page-title">Contribution Analysis</h1>
      <p className="page-description">
        Analyze how each of the eight factors contributed to a team's net rating over a period
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
            <label className="form-label">Team</label>
            <select
              className="form-select"
              value={selectedTeam}
              onChange={(e) => setSelectedTeam(e.target.value)}
              disabled={!selectedSeason || teams.length === 0}
            >
              <option value="">Select team...</option>
              {teams.map((team) => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Date Range</label>
            <select
              className="form-select"
              value={dateRangeType}
              onChange={(e) => setDateRangeType(e.target.value)}
            >
              {DATE_RANGE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Model</label>
            <select
              className="form-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="">Select model...</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>{model.name}</option>
              ))}
            </select>
          </div>
        </div>

        {dateRangeType === 'custom' && (
          <div className="form-row custom-dates">
            <div className="form-group">
              <label className="form-label">Start Date</label>
              <input
                type="date"
                className="form-input"
                value={customStartDate}
                onChange={(e) => setCustomStartDate(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">End Date</label>
              <input
                type="date"
                className="form-input"
                value={customEndDate}
                onChange={(e) => setCustomEndDate(e.target.value)}
              />
            </div>
          </div>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading analysis...
        </div>
      )}

      {data && !loading && (
        <div className="results">
          <div className="summary-header card">
            <div className="summary-dates-info">
              <div className="date-range-label">{data.date_range_label}</div>
              <div className="date-range-dates">{data.start_date} to {data.end_date}</div>
              <div className="games-count">{data.games_analyzed} games</div>
            </div>
            <div className="summary-team">
              <div className="team-name">{getTeamName(data.team)}</div>
              <div className="team-record">{data.wins}-{data.losses} .{(data.win_pct * 1000).toFixed(0).padStart(3, '0')}</div>
            </div>
            <div className="summary-ratings">
              <div className="rating-row">
                <span className="rating-label">Actual Net Rating:</span>
                <span className={`rating-value ${data.net_rating >= 0 ? 'text-positive' : 'text-negative'}`}>
                  {data.net_rating >= 0 ? '+' : ''}{data.net_rating.toFixed(1)}
                </span>
              </div>
              <div className="rating-row">
                <span className="rating-label">Predicted Net Rating:</span>
                <span className="rating-value">
                  {data.predicted_net_rating >= 0 ? '+' : ''}{data.predicted_net_rating.toFixed(1)}
                </span>
              </div>
              <div className="rating-row">
                <span className="rating-label">Error =</span>
                <span className="rating-value">
                  {(data.net_rating - data.predicted_net_rating) >= 0 ? '+' : ''}{(data.net_rating - data.predicted_net_rating).toFixed(1)}
                </span>
              </div>
            </div>
          </div>

          <div className="main-chart card">
            <h2 className="card-title">Net Rating Decomposition</h2>
            <p className="chart-subtitle">
              How each factor contributed to the team's net rating relative to league average
            </p>
            <div className="chart-container chart-container--main">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={mainChartData}
                  margin={{ top: 5, right: 10, left: 10, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: 'var(--color-text)', fontSize: 14 }}
                    textAnchor="middle"
                    height={40}
                    interval={0}
                  />
                  <YAxis
                    domain={mainChartYAxis.domain}
                    ticks={mainChartYAxis.ticks}
                    tick={{ fill: 'var(--color-text-secondary)' }}
                    width={40}
                  />
                  <Tooltip
                    formatter={(value) => [value.toFixed(2), 'Value']}
                    labelStyle={{ color: '#1f2937' }}
                    contentStyle={{
                      backgroundColor: '#ffffff',
                      border: '1px solid var(--color-border)',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                    }}
                  />
                  <ReferenceLine y={0} stroke="var(--color-text)" strokeWidth={2} />
                  <Bar dataKey="value" barSize={45}>
                    {mainChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} stroke="#000" strokeWidth={1} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="top-contributors-section">
            <h2 className="section-title">Top Contributing Factors</h2>
            <div className="top-contributors grid grid-2">
              {data.top_contributors.map((contributor, index) => (
                <MiniContributorChart
                  key={contributor.factor}
                  contributor={contributor}
                  index={index + 1}
                />
              ))}
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
      )}
    </div>
  )
}

function MiniContributorChart({ contributor, index }) {
  const chartData = contributor.trend_data.map(point => ({
    ...point,
    xLabel: `${point.home_away === 'road' ? '@' : 'vs '}${point.opponent}`,
    tooltipLabel: `${point.game_date}: ${point.home_away === 'road' ? '@' : 'vs '}${point.opponent}`,
  }))

  const yAxisConfig = useMemo(() => {
    if (chartData.length === 0) return { domain: [0, 100], ticks: [] }

    const values = chartData.map(d => d.value).filter(v => v != null)
    if (values.length === 0) return { domain: [0, 100], ticks: [] }

    const minVal = Math.min(...values)
    const maxVal = Math.max(...values)

    // Calculate evenly spaced ticks
    const padding = 2
    const tickMin = Math.floor(minVal - padding)
    const tickMax = Math.ceil(maxVal + padding)
    const range = tickMax - tickMin

    // Aim for 4-6 ticks with nice intervals
    let tickInterval
    if (range <= 10) tickInterval = 2
    else if (range <= 20) tickInterval = 4
    else if (range <= 30) tickInterval = 5
    else tickInterval = 10

    const ticks = []
    const startTick = Math.floor(tickMin / tickInterval) * tickInterval
    for (let t = startTick; t <= tickMax; t += tickInterval) {
      ticks.push(t)
    }

    return { domain: [ticks[0], ticks[ticks.length - 1]], ticks }
  }, [chartData])

  const isPositive = contributor.contribution >= 0

  // Custom tooltip that only shows date and opponent
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length > 0) {
      const data = payload[0].payload
      return (
        <div style={{
          backgroundColor: '#ffffff',
          border: '1px solid #e5e7eb',
          borderRadius: '4px',
          padding: '6px 10px',
          fontSize: '12px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        }}>
          {data.tooltipLabel}
        </div>
      )
    }
    return null
  }

  return (
    <div className="contributor-card card">
      <div className="contributor-header">
        <div className="contributor-rank">#{index}</div>
        <div className="contributor-info">
          <h3 className="contributor-title">{contributor.factor_label}</h3>
          <div className="contributor-stats">
            <span className="stat">Value: {contributor.value.toFixed(1)}</span>
            <span className="stat">League Avg: {contributor.league_avg.toFixed(1)}</span>
            <span className={`stat contribution ${isPositive ? 'positive' : 'negative'}`}>
              Contribution: {isPositive ? '+' : ''}{contributor.contribution.toFixed(2)}
            </span>
          </div>
        </div>
      </div>
      <div className="mini-chart-container">
        <ResponsiveContainer width="100%" height={180}>
          <ComposedChart
            data={chartData}
            margin={{ top: 10, right: 10, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="xLabel" hide />
            <YAxis
              domain={yAxisConfig.domain}
              ticks={yAxisConfig.ticks}
              tick={{ fill: 'var(--color-text-secondary)', fontSize: 10 }}
              width={35}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              y={contributor.league_avg}
              stroke="#000000"
              strokeWidth={1.5}
            />
            <ReferenceLine
              y={contributor.value}
              stroke={(() => {
                const isLowerBetter = LOWER_IS_BETTER_FACTORS.has(contributor.factor)
                const isBetter = isLowerBetter
                  ? contributor.value < contributor.league_avg
                  : contributor.value >= contributor.league_avg
                return isBetter ? 'var(--color-positive)' : 'var(--color-negative)'
              })()}
              strokeWidth={1.5}
            />
            <Bar
              dataKey="value"
              barSize={8}
              fill="#d1d5db"
              stroke="#6b7280"
              strokeWidth={0.5}
            />
            <Line
              type="monotone"
              dataKey="ma_5"
              stroke="#2563eb"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default ContributionAnalysis
