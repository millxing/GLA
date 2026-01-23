import { useState, useEffect, useMemo } from 'react'
import { getSeasons, getLeagueSummary } from '../api'
import './LeagueSummary.css'

// Column definitions with new structure
// BH = Ball Handling = 100 - TOV% (higher is better)
// isOpp indicates opponent stats (for gradient reversal)
const COLUMNS = [
  { key: 'team', label: 'Team', labelLine2: '', sortable: true },
  { key: 'games', label: 'GP', labelLine2: '', sortable: true },
  { key: 'wins', label: 'W', labelLine2: '', sortable: true, higherBetter: true },
  { key: 'losses', label: 'L', labelLine2: '', sortable: true, higherBetter: false },
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
]

const GLOSSARY_ITEMS = [
  { term: 'GP', definition: 'Games Played' },
  { term: 'W', definition: 'Wins' },
  { term: 'L', definition: 'Losses' },
  { term: 'ORtg', definition: 'Offensive Rating - Points scored per 100 possessions' },
  { term: 'DRtg', definition: 'Defensive Rating - Points allowed per 100 possessions (lower is better)' },
  { term: 'NRtg', definition: 'Net Rating - Offensive Rating minus Defensive Rating' },
  { term: 'eFG%', definition: 'Effective Field Goal Percentage - Adjusts FG% for the added value of 3-pointers. Formula: (FGM + 0.5 × 3PM) / FGA × 100' },
  { term: 'BH', definition: 'Ball Handling - Measures ability to take care of the ball. Calculated as 100 - TOV%. Higher is better.' },
  { term: 'OREB%', definition: 'Offensive Rebounding Percentage - Percentage of available offensive rebounds grabbed' },
  { term: 'FT Rate', definition: 'Free Throw Rate - Free throws made per field goal attempt (FTM / FGA × 100)' },
  { term: 'Opp', definition: 'Opponent statistics - For defensive stats, lower opponent values are better for your team' },
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
  const [glossaryExpanded, setGlossaryExpanded] = useState(false)

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

  // Reset dates when season changes
  useEffect(() => {
    setStartDate('')
    setEndDate('')
    setDatesInitialized(false)
  }, [selectedSeason])

  // Track if we've set initial dates for this season
  const [datesInitialized, setDatesInitialized] = useState(false)

  useEffect(() => {
    async function loadData() {
      if (!selectedSeason) return
      setLoading(true)
      setError(null)
      try {
        const res = await getLeagueSummary(selectedSeason, startDate || null, endDate || null)
        setData(res)
        // Set default dates on first load for a season
        if (!datesInitialized && res.first_game_date && res.last_game_date) {
          setStartDate(res.first_game_date)
          setEndDate(res.last_game_date)
          setDatesInitialized(true)
        }
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [selectedSeason, startDate, endDate, datesInitialized])

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

  // Compute min/max for sorted column only
  const sortedColumnStats = useMemo(() => {
    if (!data?.teams || data.teams.length === 0) return null
    const col = COLUMNS.find(c => c.key === sortColumn)
    if (!col || col.key === 'team') return null
    const values = data.teams.map(t => t[sortColumn]).filter(v => typeof v === 'number')
    if (values.length === 0) return null
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      higherBetter: col.higherBetter,
      isOpp: col.isOpp,
    }
  }, [data, sortColumn])

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

  // Compute league averages for the new columns
  const leagueAverages = useMemo(() => {
    if (!data?.teams || data.teams.length === 0) return null
    const avg = {}
    COLUMNS.forEach(col => {
      if (col.key === 'team') {
        avg[col.key] = 'League Avg'
      } else {
        const values = data.teams.map(t => t[col.key]).filter(v => typeof v === 'number')
        if (values.length > 0) {
          avg[col.key] = values.reduce((a, b) => a + b, 0) / values.length
        }
      }
    })
    return avg
  }, [data])

  // Export to XLSX
  const handleExport = () => {
    if (!sortedTeams.length) return

    // Build CSV content (XLSX-compatible)
    const headers = ['Rank', ...COLUMNS.map(col => col.labelLine2 ? `${col.labelLine2} ${col.label}` : col.label)]
    const rows = sortedTeams.map((team, index) => {
      return [
        index + 1,
        ...COLUMNS.map(col => formatValue(team[col.key], col.key))
      ]
    })

    // Add league averages row
    if (leagueAverages) {
      rows.push([
        '',
        ...COLUMNS.map(col => col.key === 'team' ? 'League Avg' : formatValue(leagueAverages[col.key], col.key))
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
    a.download = `league_summary_${selectedSeason}_${startDate}_${endDate}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
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
                  <th className="rank-col">Rank</th>
                  {COLUMNS.map((col) => (
                    <th
                      key={col.key}
                      className={`stat-header ${col.sortable ? 'sortable' : ''} ${sortColumn === col.key ? 'sorted' : ''}`}
                      onClick={() => col.sortable && handleSort(col.key)}
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
                {/* League Averages Row */}
                {leagueAverages && (
                  <tr className="league-avg-row">
                    <td className="rank-col"></td>
                    {COLUMNS.map((col) => (
                      <td
                        key={col.key}
                        className={col.key === 'team' ? 'team-cell' : 'stat-cell'}
                      >
                        {formatValue(leagueAverages[col.key], col.key)}
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

          <div className="glossary-section">
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

export default LeagueSummary
