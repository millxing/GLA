import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'
import { getSeasons, getGames, getModels, getDecomposition } from '../api'
import './FourFactor.css'

function FourFactor() {
  const [seasons, setSeasons] = useState([])
  const [games, setGames] = useState([])
  const [models, setModels] = useState([])
  const [selectedSeason, setSelectedSeason] = useState('')
  const [selectedGame, setSelectedGame] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [factorType, setFactorType] = useState('eight_factors')
  const [decomposition, setDecomposition] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [glossaryExpanded, setGlossaryExpanded] = useState(false)

  useEffect(() => {
    async function loadInitialData() {
      try {
        const [seasonsRes, modelsRes] = await Promise.all([getSeasons(), getModels()])
        setSeasons(seasonsRes.seasons)
        setModels(modelsRes.models)
        if (seasonsRes.seasons.length > 0) {
          setSelectedSeason(seasonsRes.seasons[0])
        }
        if (modelsRes.models.length > 0) {
          setSelectedModel(modelsRes.models[0].id)
        }
      } catch (err) {
        setError(err.message)
      }
    }
    loadInitialData()
  }, [])

  useEffect(() => {
    async function loadGames() {
      if (!selectedSeason) return
      try {
        const gamesRes = await getGames(selectedSeason)
        setGames(gamesRes.games)
        // Default to most recent game (first in list, sorted by date descending)
        if (gamesRes.games.length > 0) {
          setSelectedGame(gamesRes.games[0].game_id)
        } else {
          setSelectedGame('')
        }
        setDecomposition(null)
      } catch (err) {
        setError(err.message)
      }
    }
    loadGames()
  }, [selectedSeason])

  useEffect(() => {
    async function loadDecomposition() {
      if (!selectedSeason || !selectedGame || !selectedModel) return
      setLoading(true)
      setError(null)
      try {
        const data = await getDecomposition(selectedSeason, selectedGame, selectedModel, factorType)
        setDecomposition(data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadDecomposition()
  }, [selectedSeason, selectedGame, selectedModel, factorType])

  const getContributionChartData = () => {
    if (!decomposition) return []
    const contributions = decomposition.contributions

    // Map factor names to display labels with team abbreviations
    const factorLabels = {
      'shooting': `eFG%`,
      'ball_handling': `Ball Handling`,
      'orebounding': `OREB%`,
      'free_throws': `FT Rate`,
      // Eight factors mode
      'home_shooting': `${decomposition.home_team} eFG%`,
      'road_shooting': `${decomposition.road_team} eFG%`,
      'home_ball_handling': `${decomposition.home_team} Ball Handling`,
      'road_ball_handling': `${decomposition.road_team} Ball Handling`,
      'home_orebounding': `${decomposition.home_team} OREB%`,
      'road_orebounding': `${decomposition.road_team} OREB%`,
      'home_free_throws': `${decomposition.home_team} FT Rate`,
      'road_free_throws': `${decomposition.road_team} FT Rate`,
    }

    // Build factor bars sorted by value descending (most positive at top)
    const factorBars = Object.entries(contributions)
      .map(([factor, value]) => ({
        factor: factorLabels[factor] || factor,
        value,
        fill: value >= 0 ? 'var(--color-positive)' : 'var(--color-negative)',
      }))
      .sort((a, b) => b.value - a.value)

    // Add error bar at the end (always at bottom, not sorted)
    const error = decomposition.actual_margin - decomposition.predicted_margin
    factorBars.push({
      factor: 'Error',
      value: error,
      fill: 'var(--color-neutral)',
    })

    return factorBars
  }

  const formatFactorValue = (value) => {
    if (value === null || value === undefined) return '-'
    return typeof value === 'number' ? value.toFixed(1) : value
  }

  // Determine background color class based on comparison to league average
  // Green (positive) if above average, red (negative) if below
  const getFactorCellClass = (value, factorKey) => {
    if (!decomposition?.league_averages || value === null || value === undefined) return ''

    // Map internal factor keys to league_averages keys
    const keyMap = {
      efg: 'efg',
      ball_handling: 'ball_handling',
      oreb: 'oreb_pct',
      ft_rate: 'ft_rate',
    }

    const avgKey = keyMap[factorKey]
    const avg = decomposition.league_averages[avgKey]
    if (avg === undefined) return ''

    return value >= avg ? 'bg-positive' : 'bg-negative'
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

  <div style={{ marginBottom: "1rem" }}>
    <label>
      Season:&nbsp;
      <select
        value={selectedSeason}
        onChange={(e) => setSelectedSeason(e.target.value)}
      >
        {seasons.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
    </label>

    &nbsp;&nbsp;

    <label>
      Game:&nbsp;
      <select
        value={selectedGame}
        onChange={(e) => setSelectedGame(e.target.value)}
        disabled={!games.length}
      >
        <option value="">-- select game --</option>
        {games.map((g) => (
          <option key={g.game_id} value={g.game_id}>
            {g.label}
          </option>
        ))}
      </select>
    </label>

    &nbsp;&nbsp;

    <label>
      Model:&nbsp;
      <select
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name}
          </option>
        ))}
      </select>
    </label>
  </div>

  return (
    <div className="four-factor container">
      <h1 className="page-title">Factor Game Decomposition</h1>
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
            <label className="form-label">Model</label>
            <select
              className="form-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((model) => (
                <option key={model.id} value={model.id}>{model.name}</option>
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

      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          Loading analysis...
        </div>
      )}

      {decomposition && !loading && (
        <div className="results">
          <div className="game-header card">
            <div className="game-header-content">
              <div className="matchup-left">
                <div className="team road-team">
                  <span className="team-abbr">{decomposition.road_team}</span>
                  <span className="team-score">{decomposition.road_pts}</span>
                </div>
                <div className="at-symbol">@</div>
                <div className="team home-team">
                  <span className="team-abbr">{decomposition.home_team}</span>
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
                <div className="game-date">{decomposition.game_date}</div>
                {decomposition.game_type && (
                  <div className="game-type">{formatGameType(decomposition.game_type)}</div>
                )}
              </div>
            </div>
          </div>

          <div className="analysis-grid">
            <div className="factors-table card">
              <h2 className="card-title">Factor Comparison</h2>
              <table>
                <thead>
                  <tr>
                    <th>Factor</th>
                    <th className="text-center">{decomposition.home_team} (Home)</th>
                    <th className="text-center">{decomposition.road_team} (Road)</th>
                    <th className="text-center">Differential</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>eFG%</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.home_factors.efg, 'efg')}`}>{formatFactorValue(decomposition.home_factors.efg)}</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.road_factors.efg, 'efg')}`}>{formatFactorValue(decomposition.road_factors.efg)}</td>
                    <td className={`text-center ${(decomposition.home_factors.efg - decomposition.road_factors.efg) >= 0 ? 'text-positive' : 'text-negative'}`}>
                      {formatFactorValue(decomposition.home_factors.efg - decomposition.road_factors.efg)}
                    </td>
                  </tr>
                  <tr>
                    <td>Ball Handling</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.home_factors.ball_handling, 'ball_handling')}`}>{formatFactorValue(decomposition.home_factors.ball_handling)}</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.road_factors.ball_handling, 'ball_handling')}`}>{formatFactorValue(decomposition.road_factors.ball_handling)}</td>
                    <td className={`text-center ${(decomposition.home_factors.ball_handling - decomposition.road_factors.ball_handling) >= 0 ? 'text-positive' : 'text-negative'}`}>
                      {formatFactorValue(decomposition.home_factors.ball_handling - decomposition.road_factors.ball_handling)}
                    </td>
                  </tr>
                  <tr>
                    <td>OREB%</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.home_factors.oreb, 'oreb')}`}>{formatFactorValue(decomposition.home_factors.oreb)}</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.road_factors.oreb, 'oreb')}`}>{formatFactorValue(decomposition.road_factors.oreb)}</td>
                    <td className={`text-center ${(decomposition.home_factors.oreb - decomposition.road_factors.oreb) >= 0 ? 'text-positive' : 'text-negative'}`}>
                      {formatFactorValue(decomposition.home_factors.oreb - decomposition.road_factors.oreb)}
                    </td>
                  </tr>
                  <tr>
                    <td>FT Rate</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.home_factors.ft_rate, 'ft_rate')}`}>{formatFactorValue(decomposition.home_factors.ft_rate)}</td>
                    <td className={`text-center ${getFactorCellClass(decomposition.road_factors.ft_rate, 'ft_rate')}`}>{formatFactorValue(decomposition.road_factors.ft_rate)}</td>
                    <td className={`text-center ${(decomposition.home_factors.ft_rate - decomposition.road_factors.ft_rate) >= 0 ? 'text-positive' : 'text-negative'}`}>
                      {formatFactorValue(decomposition.home_factors.ft_rate - decomposition.road_factors.ft_rate)}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="ratings-table card">
              <h2 className="card-title">Game Ratings</h2>
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th className="text-center">{decomposition.home_team}</th>
                    <th className="text-center">{decomposition.road_team}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Offensive Rating</td>
                    <td className="text-center">{formatFactorValue(decomposition.home_ratings.offensive_rating)}</td>
                    <td className="text-center">{formatFactorValue(decomposition.road_ratings.offensive_rating)}</td>
                  </tr>
                  <tr>
                    <td>Defensive Rating</td>
                    <td className="text-center">{formatFactorValue(decomposition.home_ratings.defensive_rating)}</td>
                    <td className="text-center">{formatFactorValue(decomposition.road_ratings.defensive_rating)}</td>
                  </tr>
                  <tr>
                    <td>Net Rating</td>
                    <td className={`text-center ${decomposition.home_ratings.net_rating >= 0 ? 'text-positive' : 'text-negative'}`}>
                      {decomposition.home_ratings.net_rating > 0 ? '+' : ''}{formatFactorValue(decomposition.home_ratings.net_rating)}
                    </td>
                    <td className={`text-center ${decomposition.road_ratings.net_rating >= 0 ? 'text-positive' : 'text-negative'}`}>
                      {decomposition.road_ratings.net_rating > 0 ? '+' : ''}{formatFactorValue(decomposition.road_ratings.net_rating)}
                    </td>
                  </tr>
                  <tr>
                    <td>Pace</td>
                    <td className="text-center">{formatFactorValue(decomposition.home_ratings.pace ?? decomposition.home_ratings.possessions)}</td>
                    <td className="text-center">{formatFactorValue(decomposition.home_ratings.pace ?? decomposition.home_ratings.possessions)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="contributions-chart card">
            <h2 className="card-title">Factor Contributions to Predicted Margin</h2>
            <p className="chart-subtitle">
              Positive values favor the home team ({decomposition.home_team}), negative values favor the road team ({decomposition.road_team})
            </p>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={550}>
                <BarChart
                  data={getContributionChartData()}
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 160, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    type="number"
                    tick={{ fill: 'var(--color-text-secondary)' }}
                    label={{
                      value: 'Contribution',
                      position: 'bottom',
                      offset: 10,
                      style: { fill: 'var(--color-text)', fontWeight: 600 }
                    }}
                  />
                  <YAxis
                    type="category"
                    dataKey="factor"
                    tick={{ fill: 'var(--color-text-secondary)' }}
                    width={150}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--color-background)',
                      border: '1px solid var(--color-border)',
                      borderRadius: 'var(--border-radius-sm)',
                    }}
                    formatter={(value) => [value.toFixed(1), 'Contribution']}
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
                      <dt>eFG% (Effective Field Goal %)</dt>
                      <dd>Adjusts field goal percentage to account for the added value of three-pointers. Formula: (FGM + 0.5 × 3PM) / FGA × 100</dd>

                      <dt>Ball Handling</dt>
                      <dd>A team's ability to take care of the ball, calculated as 100 − TOV%. Higher is better. TOV% = TOV / (FGA + 0.44×FTA + TOV) × 100</dd>

                      <dt>OREB% (Offensive Rebounding %)</dt>
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
                      This tool uses a linear regression model trained on historical NBA games to predict the final margin based on the Four Factors. The model calculates:
                    </p>
                    <ol>
                      <li><strong>Factor Differentials:</strong> For each of the Four Factors, we calculate the difference between the home team and road team performance.</li>
                      <li><strong>Weighted Contributions:</strong> Each differential is multiplied by a coefficient that represents its importance in predicting game outcomes. These coefficients are learned from historical data.</li>
                      <li><strong>Predicted Margin:</strong> The sum of all weighted contributions gives us the predicted point margin.</li>
                    </ol>
                    <p>
                      <strong>Four Factors Mode:</strong> Uses differentials (Home − Road) for each factor.
                    </p>
                    <p>
                      <strong>Eight Factors Mode:</strong> Analyzes home and road team factors separately, centering each around league averages for more nuanced analysis.
                    </p>
                    <p className="model-formula">
                      Formula: <code>Predicted Margin = Intercept + Σ(Coefficient × Factor Differential)</code>
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
