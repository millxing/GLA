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
  { key: 'pace', label: 'Pace', labelLine2: '', sortable: true, higherBetter: null },
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
  { term: 'Pace', definition: 'Average possessions per game. Higher pace indicates a faster-paced playing style.' },
]

const DATE_RANGE_OPTIONS = [
  { value: 'season', label: 'Season-to-Date' },
  { value: 'season_no_playoffs', label: 'Season-to-Date, No Playoffs' },
  { value: 'month', label: 'Month-to-Date' },
  { value: 'last30', label: 'Last 30 Days' },
  { value: 'last60', label: 'Last 60 Days' },
  { value: 'last90', label: 'Last 90 Days' },
  { value: 'custom', label: 'Custom Date Range' },
]

function LeagueSummary() {
  const [seasons, setSeasons] = useState([])
  const [selectedSeason, setSelectedSeason] = useState('')
  const [dateRangePreset, setDateRangePreset] = useState('season')
  const [customStartDate, setCustomStartDate] = useState('')
  const [customEndDate, setCustomEndDate] = useState('')
  const [seasonBounds, setSeasonBounds] = useState({ first: '', last: '' })
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

  // Reset date preset when season changes
  useEffect(() => {
    setDateRangePreset('season')
    setCustomStartDate('')
    setCustomEndDate('')
    setSeasonBounds({ first: '', last: '' })
  }, [selectedSeason])

  // Calculate actual start/end dates based on preset
  const { startDate, endDate } = useMemo(() => {
    if (!seasonBounds.first || !seasonBounds.last) {
      return { startDate: '', endDate: '' }
    }

    const firstDate = new Date(seasonBounds.first)
    const lastDate = new Date(seasonBounds.last)
    const seasonDays = Math.floor((lastDate - firstDate) / (1000 * 60 * 60 * 24))

    switch (dateRangePreset) {
      case 'season':
        return { startDate: seasonBounds.first, endDate: seasonBounds.last }

      case 'season_no_playoffs':
        return { startDate: seasonBounds.first, endDate: seasonBounds.last }

      case 'month': {
        // First day of the month containing the last game
        const monthStart = new Date(lastDate.getFullYear(), lastDate.getMonth(), 1)
        const monthStartStr = monthStart.toISOString().split('T')[0]
        // Use season start if month start is before season start
        return {
          startDate: monthStartStr < seasonBounds.first ? seasonBounds.first : monthStartStr,
          endDate: seasonBounds.last
        }
      }

      case 'last30': {
        if (seasonDays < 30) {
          return { startDate: seasonBounds.first, endDate: seasonBounds.last }
        }
        const start = new Date(lastDate)
        start.setDate(start.getDate() - 29)
        return { startDate: start.toISOString().split('T')[0], endDate: seasonBounds.last }
      }

      case 'last60': {
        if (seasonDays < 60) {
          return { startDate: seasonBounds.first, endDate: seasonBounds.last }
        }
        const start = new Date(lastDate)
        start.setDate(start.getDate() - 59)
        return { startDate: start.toISOString().split('T')[0], endDate: seasonBounds.last }
      }

      case 'last90': {
        if (seasonDays < 90) {
          return { startDate: seasonBounds.first, endDate: seasonBounds.last }
        }
        const start = new Date(lastDate)
        start.setDate(start.getDate() - 89)
        return { startDate: start.toISOString().split('T')[0], endDate: seasonBounds.last }
      }

      case 'custom':
        return {
          startDate: customStartDate || seasonBounds.first,
          endDate: customEndDate || seasonBounds.last
        }

      default:
        return { startDate: seasonBounds.first, endDate: seasonBounds.last }
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
        const res = await getLeagueSummary(selectedSeason, null, null)
        if (isCurrent && res.first_game_date && res.last_game_date) {
          setSeasonBounds({ first: res.first_game_date, last: res.last_game_date })
          setCustomStartDate(res.first_game_date)
          setCustomEndDate(res.last_game_date)
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
  }, [selectedSeason])

  // Load data with calculated date range
  useEffect(() => {
    if (!selectedSeason || !startDate || !endDate) return
    let isCurrent = true
    async function loadData() {
      setLoading(true)
      setError(null)
      try {
        const excludePlayoffs = dateRangePreset === 'season_no_playoffs'
        const res = await getLeagueSummary(selectedSeason, startDate, endDate, excludePlayoffs)
        if (isCurrent) {
          setData(res)
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
  }, [selectedSeason, startDate, endDate, dateRangePreset])

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
      // For GP, W, L: show integer if whole number, otherwise 1 decimal
      if (['games', 'wins', 'losses'].includes(column)) {
        return Number.isInteger(value) ? value : value.toFixed(1)
      }
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
        avg[col.key] = 'Average'
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
        ...COLUMNS.map(col => col.key === 'team' ? 'Average' : formatValue(leagueAverages[col.key], col.key))
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
            <label className="form-label">Date Range</label>
            <select
              className="form-select"
              value={dateRangePreset}
              onChange={(e) => setDateRangePreset(e.target.value)}
            >
              {DATE_RANGE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          {dateRangePreset === 'custom' && (
            <>
              <div className="form-group">
                <label className="form-label">Start Date</label>
                <input
                  type="date"
                  className="form-input"
                  value={customStartDate}
                  min={seasonBounds.first}
                  max={seasonBounds.last}
                  onChange={(e) => setCustomStartDate(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">End Date</label>
                <input
                  type="date"
                  className="form-input"
                  value={customEndDate}
                  min={seasonBounds.first}
                  max={seasonBounds.last}
                  onChange={(e) => setCustomEndDate(e.target.value)}
                />
              </div>
            </>
          )}
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
                  <th className="rank-col">
                    <div className="header-content">
                      <span className="header-line1">Rank</span>
                    </div>
                  </th>
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
