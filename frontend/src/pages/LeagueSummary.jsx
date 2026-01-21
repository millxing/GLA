import { useState, useEffect, useMemo } from 'react'
import { getSeasons, getLeagueSummary } from '../api'
import './LeagueSummary.css'

const COLUMNS = [
  { key: 'team', label: 'Team', sortable: true },
  { key: 'games', label: 'GP', sortable: true },
  { key: 'wins', label: 'W', sortable: true },
  { key: 'losses', label: 'L', sortable: true },
  { key: 'win_pct', label: 'WIN%', sortable: true, higherBetter: true },
  { key: 'ppg', label: 'PPG', sortable: true, higherBetter: true },
  { key: 'opp_ppg', label: 'OPP PPG', sortable: true, higherBetter: false },
  { key: 'off_rating', label: 'ORtg', sortable: true, higherBetter: true },
  { key: 'def_rating', label: 'DRtg', sortable: true, higherBetter: false },
  { key: 'net_rating', label: 'NRtg', sortable: true, higherBetter: true },
  { key: 'efg_pct', label: 'eFG%', sortable: true, higherBetter: true },
  { key: 'opp_efg_pct', label: 'OPP eFG%', sortable: true, higherBetter: false },
  { key: 'tov_pct', label: 'TOV%', sortable: true, higherBetter: false },
  { key: 'opp_tov_pct', label: 'OPP TOV%', sortable: true, higherBetter: true },
  { key: 'oreb_pct', label: 'OREB%', sortable: true, higherBetter: true },
  { key: 'dreb_pct', label: 'DREB%', sortable: true, higherBetter: true },
  { key: 'ft_rate', label: 'FT Rate', sortable: true, higherBetter: true },
  { key: 'opp_ft_rate', label: 'OPP FT Rate', sortable: true, higherBetter: false },
  { key: 'fg_pct', label: 'FG%', sortable: true, higherBetter: true },
  { key: 'fg3_pct', label: '3P%', sortable: true, higherBetter: true },
  { key: 'ft_pct', label: 'FT%', sortable: true, higherBetter: true },
]

function LeagueSummary() {
  const [seasons, setSeasons] = useState([])
  const [selectedSeason, setSelectedSeason] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sortColumn, setSortColumn] = useState('net_rating')
  const [sortDirection, setSortDirection] = useState('desc')

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
    async function loadData() {
      if (!selectedSeason) return
      setLoading(true)
      setError(null)
      try {
        const res = await getLeagueSummary(selectedSeason, startDate || null, endDate || null)
        setData(res)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [selectedSeason, startDate, endDate])

  const sortedTeams = useMemo(() => {
    if (!data?.teams) return []
    const sorted = [...data.teams].sort((a, b) => {
      const aVal = a[sortColumn]
      const bVal = b[sortColumn]
      if (typeof aVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
    })
    return sorted
  }, [data, sortColumn, sortDirection])

  const columnStats = useMemo(() => {
    if (!data?.teams || data.teams.length === 0) return {}
    const stats = {}
    COLUMNS.forEach((col) => {
      if (col.key === 'team') return
      const values = data.teams.map((t) => t[col.key]).filter((v) => typeof v === 'number')
      if (values.length > 0) {
        stats[col.key] = {
          min: Math.min(...values),
          max: Math.max(...values),
        }
      }
    })
    return stats
  }, [data])

  const getCellColor = (column, value) => {
    const col = COLUMNS.find((c) => c.key === column)
    if (!col || col.key === 'team' || !columnStats[column]) return null

    const { min, max } = columnStats[column]
    if (min === max) return null

    const normalized = (value - min) / (max - min)
    const intensity = Math.round(normalized * 100)

    if (col.higherBetter === true) {
      return `rgba(34, 197, 94, ${0.1 + normalized * 0.3})`
    } else if (col.higherBetter === false) {
      return `rgba(239, 68, 68, ${0.1 + (1 - normalized) * 0.3})`
    }
    return null
  }

  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(column)
      const col = COLUMNS.find((c) => c.key === column)
      setSortDirection(col?.higherBetter === false ? 'asc' : 'desc')
    }
  }

  const formatValue = (value, column) => {
    if (value === null || value === undefined) return '-'
    if (column === 'team') return value
    if (typeof value === 'number') {
      if (['games', 'wins', 'losses'].includes(column)) return value
      return value.toFixed(1)
    }
    return value
  }

  return (
    <div className="league-summary container">
      <h1 className="page-title">League Summary</h1>
      <p className="page-description">
        Compare team statistics across the league. Click column headers to sort.
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
            <label className="form-label">Start Date</label>
            <input
              type="date"
              className="form-input"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">End Date</label>
            <input
              type="date"
              className="form-input"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Sort</label>
            <div className="sort-controls">
              <select
                className="form-select"
                value={sortColumn}
                onChange={(e) => handleSort(e.target.value)}
              >
                {COLUMNS.filter((c) => c.sortable).map((col) => (
                  <option key={col.key} value={col.key}>{col.label}</option>
                ))}
              </select>
              <button
                className="btn btn-secondary sort-btn"
                onClick={() => setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')}
              >
                {sortDirection === 'asc' ? '↑ Asc' : '↓ Desc'}
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
                  <th className="rank-col">#</th>
                  {COLUMNS.map((col) => (
                    <th
                      key={col.key}
                      className={`${col.sortable ? 'sortable' : ''} ${sortColumn === col.key ? 'sorted' : ''}`}
                      onClick={() => col.sortable && handleSort(col.key)}
                    >
                      {col.label}
                      {sortColumn === col.key && (
                        <span className="sort-indicator">
                          {sortDirection === 'asc' ? ' ↑' : ' ↓'}
                        </span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedTeams.map((team, index) => (
                  <tr key={team.team}>
                    <td className="rank-col">{index + 1}</td>
                    {COLUMNS.map((col) => (
                      <td
                        key={col.key}
                        style={col.key !== 'team' ? { backgroundColor: getCellColor(col.key, team[col.key]) } : {}}
                        className={col.key === 'team' ? 'team-cell' : 'stat-cell'}
                      >
                        {formatValue(team[col.key], col.key)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {data.league_averages && (
            <div className="league-averages">
              <h3>League Averages</h3>
              <div className="averages-grid">
                {Object.entries(data.league_averages).slice(0, 10).map(([key, value]) => {
                  const col = COLUMNS.find((c) => c.key === key)
                  return (
                    <div key={key} className="average-item">
                      <span className="average-label">{col?.label || key}</span>
                      <span className="average-value">{typeof value === 'number' ? value.toFixed(1) : value}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default LeagueSummary
