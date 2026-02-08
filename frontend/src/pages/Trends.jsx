import { useState, useEffect, useMemo } from 'react'
import {
  ComposedChart,
  Bar,
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
import './Trends.css'

const STAT_OPTIONS = [
  { value: 'net_rating', label: 'Net Rating' },
  { value: 'off_rating', label: 'Offensive Rating' },
  { value: 'def_rating', label: 'Defensive Rating' },
  { value: 'efg_pct', label: 'Effective FG%' },
  { value: 'ball_handling', label: 'Ball-Handling' },
  { value: 'oreb_pct', label: 'Offensive Rebound %' },
  { value: 'ft_rate', label: 'Free Throw Rate' },
  { value: 'opp_efg_pct', label: 'Opp Effective FG%' },
  { value: 'opp_ball_handling', label: 'Opp Ball-Handling' },
  { value: 'opp_oreb_pct', label: 'Opp Offensive Rebound %' },
  { value: 'opp_ft_rate', label: 'Opp Free Throw Rate' },
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
  { term: 'FG2%', definition: '2-Point Field Goal Percentage - 2PT FGM / 2PT FGA × 100.' },
  { term: 'FG3%', definition: '3-Point Field Goal Percentage - 3PT FGM / 3PT FGA × 100.' },
  { term: 'FG3A Rate', definition: '3-Point Attempt Rate - Percentage of field goal attempts that are 3-pointers (3PA / FGA × 100).' },
  { term: 'Opp', definition: 'Opponent statistics - What your opponents shoot/do against you.' },
  { term: 'Pace', definition: 'Average possessions per game for both teams. Higher pace indicates a faster-paced game.' },
]

function Trends() {
  const [seasons, setSeasons] = useState([])
  const [teams, setTeams] = useState([])
  const [selectedSeason, setSelectedSeason] = usePersistedState('trends_season', '')
  const [selectedTeam, setSelectedTeam] = usePersistedState('trends_team', '')
  const [selectedStat, setSelectedStat] = usePersistedState('trends_stat', 'net_rating')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [error, setError] = useState(null)
  const [glossaryExpanded, setGlossaryExpanded] = useState(false)

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
  }, [])

  useEffect(() => {
    let isCurrent = true
    async function loadTeams() {
      if (!selectedSeason) return
      setInitializing(true)
      if (isCurrent) setData(null)
      try {
        const res = await getTeams(selectedSeason)
        if (!isCurrent) return
        setTeams(res.teams)
        // Keep persisted team if valid, otherwise default to BOS or first team
        setSelectedTeam(prev => {
          if (prev && res.teams.includes(prev)) return prev
          if (res.teams.includes('BOS')) return 'BOS'
          if (res.teams.length > 0) return res.teams[0]
          return ''
        })
      } catch (err) {
        if (isCurrent) setError(err.message)
      } finally {
        if (isCurrent) setInitializing(false)
      }
    }
    loadTeams()
    return () => { isCurrent = false }
  }, [selectedSeason])

  useEffect(() => {
    let isCurrent = true
    async function loadTrends() {
      if (!selectedSeason || !selectedTeam || !selectedStat) return
      setLoading(true)
      if (isCurrent) setError(null)
      try {
        const res = await getTrends(selectedSeason, selectedTeam, selectedStat)
        if (isCurrent) setData(res)
      } catch (err) {
        if (isCurrent) setError(err.message)
      } finally {
        if (isCurrent) setLoading(false)
      }
    }
    loadTrends()
    return () => { isCurrent = false }
  }, [selectedSeason, selectedTeam, selectedStat])

  // Prepare chart data with x-axis labels
  const chartData = useMemo(() => {
    if (!data?.data) return []
    return data.data.map(game => ({
      ...game,
      xLabel: `${game.home_away === 'road' ? '@' : ''}${game.opponent} on ${game.game_date}`,
    }))
  }, [data])

  // Calculate Y-axis ticks at multiples of 2
  const yAxisConfig = useMemo(() => {
    if (!data?.data || data.data.length === 0) return { ticks: [], domain: [0, 100] }

    const values = data.data.map(d => d.value)
    const minVal = Math.min(...values)
    const maxVal = Math.max(...values)

    // Round down to nearest 2 for min, round up for max, with small padding
    const tickMin = Math.floor((minVal - 1) / 2) * 2
    const tickMax = Math.ceil((maxVal + 1) / 2) * 2

    // Generate ticks at every 2
    const ticks = []
    for (let t = tickMin; t <= tickMax; t += 2) {
      ticks.push(t)
    }

    return { ticks, domain: [tickMin, tickMax] }
  }, [data])

  // Table data sorted by most recent first
  const tableData = useMemo(() => {
    if (!data?.data) return []
    return [...data.data].reverse()
  }, [data])

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null
    const entry = payload[0]?.payload
    if (!entry) return null

    return (
      <div className="custom-tooltip">
        <p className="tooltip-date">{entry.game_date}</p>
        <p className="tooltip-opponent">
          {entry.home_away === 'home' ? 'vs' : '@'} {entry.opponent}
        </p>
        <p className={`tooltip-result ${entry.wl === 'W' ? 'win' : 'loss'}`}>
          {entry.wl === 'W' ? 'Win' : 'Loss'}
        </p>
        <div className="tooltip-stats">
          <p><strong>Value:</strong> {entry.value?.toFixed(1)}</p>
          <p><strong>5-Game Avg:</strong> {entry.ma_5?.toFixed(1)}</p>
        </div>
      </div>
    )
  }

  const vsLeague = data ? (data.season_average - data.league_average).toFixed(1) : 0
  const vsLeagueFormatted = data ? (vsLeague >= 0 ? `+${vsLeague}` : vsLeague) : '-'

  // Determine if team is performing better than league average
  // For lower-is-better stats, being below league avg is good
  const isLowerBetter = LOWER_IS_BETTER_STATS.has(selectedStat)
  const isBetterThanLeague = data
    ? isLowerBetter
      ? data.season_average < data.league_average
      : data.season_average > data.league_average
    : false

  return (
    <div className="trends container">
      <h1 className="page-title">Statistical Trends</h1>
      <p className="page-description">
        Plot the time series of team statistics over the season
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
            <label className="form-label">Statistic</label>
            <select
              className="form-select"
              value={selectedStat}
              onChange={(e) => setSelectedStat(e.target.value)}
            >
              {STAT_OPTIONS.map((stat) => (
                <option key={stat.value} value={stat.value}>{stat.label}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {(loading || initializing) && (
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading trend data...
        </div>
      )}

      {data && !loading && !initializing && (
        <div className="results">
          <div className="chart-card card">
            <h2 className="card-title">{getTeamName(data.team)} {STAT_OPTIONS.find(s => s.value === selectedStat)?.label || data.stat_label}</h2>
            <div className="chart-legend">
              <span className="legend-item">
                <span
                  className="legend-line"
                  style={{ backgroundColor: isBetterThanLeague ? 'var(--color-positive)' : 'var(--color-negative)' }}
                ></span> Team Average
              </span>
              <span className="legend-item">
                <span className="legend-line league-avg"></span> League Average
              </span>
              <span className="legend-item">
                <span className="legend-line ma5"></span> 5-Game Avg
              </span>
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={550}>
                <ComposedChart
                  data={chartData}
                  margin={{ top: 20, right: 30, left: 50, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    dataKey="xLabel"
                    tick={{ fill: 'var(--color-text-secondary)', fontSize: 9 }}
                    angle={-90}
                    textAnchor="end"
                    height={130}
                    interval={0}
                  />
                  <YAxis
                    tick={{ fill: 'var(--color-text-secondary)' }}
                    domain={yAxisConfig.domain}
                    ticks={yAxisConfig.ticks}
                    label={{
                      value: STAT_OPTIONS.find(s => s.value === selectedStat)?.label || data.stat_label,
                      angle: -90,
                      position: 'insideLeft',
                      style: { fill: 'var(--color-text)', fontWeight: 600 }
                    }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  {/* League average line - black */}
                  <ReferenceLine
                    y={data.league_average}
                    stroke="#000000"
                    strokeWidth={2}
                  />
                  {/* Team average line - green if better than league, red if worse */}
                  <ReferenceLine
                    y={data.season_average}
                    stroke={isBetterThanLeague ? 'var(--color-positive)' : 'var(--color-negative)'}
                    strokeWidth={2}
                  />
                  <Bar
                    dataKey="value"
                    name="Value"
                    barSize={15}
                    fill="#d1d5db"
                    stroke="#6b7280"
                    strokeWidth={1}
                  />
                  <Line
                    type="monotone"
                    dataKey="ma_5"
                    name="5-Game Avg"
                    stroke="#2563eb"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="summary-stats card">
            <h3 className="summary-title">Summary</h3>
            <div className="summary-grid">
              <div className="stat-item">
                <span className="stat-label">Games Played</span>
                <span className="stat-value">{data.data.length}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Team Average</span>
                <span className="stat-value">{data.season_average.toFixed(1)}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">League Average</span>
                <span className="stat-value">{data.league_average.toFixed(1)}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">vs League</span>
                <span className={`stat-value ${isBetterThanLeague ? 'text-positive' : 'text-negative'}`}>
                  {vsLeagueFormatted}
                </span>
              </div>
            </div>
          </div>

          <div className="games-table card">
            <h2 className="card-title">Game-by-Game Data</h2>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Date</th>
                    <th>Opponent</th>
                    <th>Result</th>
                    <th className="text-right">Value</th>
                    <th className="text-right">5-Game Avg</th>
                  </tr>
                </thead>
                <tbody>
                  {tableData.map((game, index) => (
                    <tr key={game.game_id}>
                      <td>{tableData.length - index}</td>
                      <td>{game.game_date}</td>
                      <td>
                        {game.home_away === 'home' ? 'vs' : '@'} {game.opponent}
                      </td>
                      <td className={game.wl === 'W' ? 'text-positive' : 'text-negative'}>
                        {game.wl}
                      </td>
                      <td className="text-right">{game.value.toFixed(1)}</td>
                      <td className="text-right">{game.ma_5.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
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

export default Trends
