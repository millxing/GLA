import { useState, useEffect } from 'react'
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { getSeasons, getTeams, getTrends } from '../api'
import './Trends.css'

const STAT_OPTIONS = [
  { value: 'pts', label: 'Points' },
  { value: 'off_rating', label: 'Offensive Rating' },
  { value: 'def_rating', label: 'Defensive Rating' },
  { value: 'net_rating', label: 'Net Rating' },
  { value: 'efg_pct', label: 'eFG%' },
  { value: 'fg_pct', label: 'FG%' },
  { value: 'fg3_pct', label: '3P%' },
  { value: 'ft_pct', label: 'FT%' },
  { value: 'tov_pct', label: 'TOV%' },
  { value: 'oreb', label: 'Offensive Rebounds' },
  { value: 'dreb', label: 'Defensive Rebounds' },
  { value: 'reb', label: 'Total Rebounds' },
  { value: 'tov', label: 'Turnovers' },
]

function Trends() {
  const [seasons, setSeasons] = useState([])
  const [teams, setTeams] = useState([])
  const [selectedSeason, setSelectedSeason] = useState('')
  const [selectedTeam, setSelectedTeam] = useState('')
  const [selectedStat, setSelectedStat] = useState('pts')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function loadSeasons() {
      try {
        const res = await getSeasons()
        setSeasons(res.seasons)
        if (res.seasons.length > 0) {
          setSelectedSeason(res.seasons[0])
        }
      } catch (err) {
        setError(err.message)
      }
    }
    loadSeasons()
  }, [])

  useEffect(() => {
    async function loadTeams() {
      if (!selectedSeason) return
      try {
        const res = await getTeams(selectedSeason)
        setTeams(res.teams)
        setSelectedTeam('')
        setData(null)
      } catch (err) {
        setError(err.message)
      }
    }
    loadTeams()
  }, [selectedSeason])

  useEffect(() => {
    async function loadTrends() {
      if (!selectedSeason || !selectedTeam || !selectedStat) return
      setLoading(true)
      setError(null)
      try {
        const res = await getTrends(selectedSeason, selectedTeam, selectedStat)
        setData(res)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadTrends()
  }, [selectedSeason, selectedTeam, selectedStat])

  const getBarColor = (entry) => {
    if (entry.wl === 'W') return 'var(--color-positive)'
    if (entry.wl === 'L') return 'var(--color-negative)'
    return 'var(--color-neutral)'
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return ''
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }

  const CustomTooltip = ({ active, payload, label }) => {
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
          <p><strong>10-Game Avg:</strong> {entry.ma_10?.toFixed(1)}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="trends container">
      <h1 className="page-title">Statistical Trends</h1>
      <p className="page-description">
        Track team performance over the season with moving averages.
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

      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading trend data...
        </div>
      )}

      {data && !loading && (
        <div className="results">
          <div className="summary-stats card">
            <div className="stat-item">
              <span className="stat-label">Team</span>
              <span className="stat-value">{data.team}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Statistic</span>
              <span className="stat-value">{data.stat_label}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Season Average</span>
              <span className="stat-value">{data.season_average.toFixed(1)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Games Played</span>
              <span className="stat-value">{data.data.length}</span>
            </div>
          </div>

          <div className="chart-card card">
            <h2 className="card-title">{data.stat_label} Trend</h2>
            <div className="chart-legend">
              <span className="legend-item">
                <span className="legend-color win"></span> Win
              </span>
              <span className="legend-item">
                <span className="legend-color loss"></span> Loss
              </span>
              <span className="legend-item">
                <span className="legend-line ma5"></span> 5-Game Avg
              </span>
              <span className="legend-item">
                <span className="legend-line ma10"></span> 10-Game Avg
              </span>
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart
                  data={data.data}
                  margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    dataKey="game_date"
                    tickFormatter={formatDate}
                    tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    tick={{ fill: 'var(--color-text-secondary)' }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="Value" barSize={20}>
                    {data.data.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getBarColor(entry)} />
                    ))}
                  </Bar>
                  <Line
                    type="monotone"
                    dataKey="ma_5"
                    name="5-Game Avg"
                    stroke="var(--color-highlight)"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="ma_10"
                    name="10-Game Avg"
                    stroke="var(--color-road)"
                    strokeWidth={2}
                    dot={false}
                    strokeDasharray="5 5"
                  />
                </ComposedChart>
              </ResponsiveContainer>
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
                    <th className="text-right">10-Game Avg</th>
                  </tr>
                </thead>
                <tbody>
                  {data.data.map((game, index) => (
                    <tr key={game.game_id}>
                      <td>{index + 1}</td>
                      <td>{game.game_date}</td>
                      <td>
                        {game.home_away === 'home' ? 'vs' : '@'} {game.opponent}
                      </td>
                      <td className={game.wl === 'W' ? 'text-positive' : 'text-negative'}>
                        {game.wl}
                      </td>
                      <td className="text-right">{game.value.toFixed(1)}</td>
                      <td className="text-right">{game.ma_5.toFixed(1)}</td>
                      <td className="text-right">{game.ma_10.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Trends
