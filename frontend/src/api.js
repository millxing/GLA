const API_BASE_URL = import.meta.env.VITE_API_URL || ''

async function fetchApi(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(error.detail || 'Request failed')
  }

  return response.json()
}

export async function getSeasons() {
  return fetchApi('/api/seasons')
}

export async function getGames(season) {
  return fetchApi(`/api/games?season=${encodeURIComponent(season)}`)
}

export async function getTeams(season) {
  return fetchApi(`/api/teams?season=${encodeURIComponent(season)}`)
}

export async function getDecomposition(season, gameId, factorType) {
  const params = new URLSearchParams({
    season,
    game_id: gameId,
    factor_type: factorType,
  })
  return fetchApi(`/api/decomposition?${params}`)
}

export async function getLeagueSummary(
  season,
  startDate,
  endDate,
  excludePlayoffs = false,
  lastNGames = null
) {
  const params = new URLSearchParams({ season })
  if (startDate) params.append('start_date', startDate)
  if (endDate) params.append('end_date', endDate)
  if (lastNGames) params.append('last_n_games', lastNGames)
  params.append('exclude_playoffs', excludePlayoffs ? 'true' : 'false')
  return fetchApi(`/api/league-summary?${params}`)
}

export async function getTrends(season, team, stat) {
  const params = new URLSearchParams({ season, team, stat })
  return fetchApi(`/api/trends?${params}`)
}

export async function getContributionAnalysis(
  season,
  team,
  dateRangeType = 'season',
  lastNGames = null,
  startDate = null,
  endDate = null,
  excludePlayoffs = false
) {
  const params = new URLSearchParams({
    season,
    team,
    date_range_type: dateRangeType,
  })
  if (lastNGames) params.append('last_n_games', lastNGames)
  if (startDate) params.append('start_date', startDate)
  if (endDate) params.append('end_date', endDate)
  params.append('exclude_playoffs', excludePlayoffs ? 'true' : 'false')
  return fetchApi(`/api/contribution-analysis?${params}`)
}

export async function getLeagueTopContributors(
  season,
  startDate = null,
  endDate = null,
  excludePlayoffs = false,
  lastNGames = null
) {
  const params = new URLSearchParams({ season })
  if (startDate) params.append('start_date', startDate)
  if (endDate) params.append('end_date', endDate)
  if (lastNGames) params.append('last_n_games', lastNGames)
  params.append('exclude_playoffs', excludePlayoffs ? 'true' : 'false')
  return fetchApi(`/api/league-top-contributors?${params}`)
}

export async function getInterpretation(decomposition, factorType, season = null) {
  return fetchApi('/api/interpretation', {
    method: 'POST',
    body: JSON.stringify({
      game_id: decomposition.game_id,
      game_date: decomposition.game_date,
      season: season || decomposition.season,
      home_team: decomposition.home_team,
      road_team: decomposition.road_team,
      home_pts: decomposition.home_pts,
      road_pts: decomposition.road_pts,
      contributions: decomposition.contributions,
      predicted_rating_diff: decomposition.predicted_rating_diff,
      actual_rating_diff: decomposition.actual_rating_diff,
      factor_type: factorType,
      home_factors: decomposition.home_factors,
      road_factors: decomposition.road_factors,
      home_ratings: decomposition.home_ratings,
      road_ratings: decomposition.road_ratings,
      league_averages: decomposition.league_averages,
      factor_ranges: decomposition.factor_ranges,
    }),
  })
}
