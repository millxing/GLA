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

export async function getModels() {
  return fetchApi('/api/models')
}

export async function getDecomposition(season, gameId, modelId, factorType) {
  const params = new URLSearchParams({
    season,
    game_id: gameId,
    model_id: modelId,
    factor_type: factorType,
  })
  return fetchApi(`/api/decomposition?${params}`)
}

export async function getLeagueSummary(season, startDate, endDate, excludePlayoffs = false) {
  const params = new URLSearchParams({ season })
  if (startDate) params.append('start_date', startDate)
  if (endDate) params.append('end_date', endDate)
  if (excludePlayoffs) params.append('exclude_playoffs', 'true')
  return fetchApi(`/api/league-summary?${params}`)
}

export async function getTrends(season, team, stat) {
  const params = new URLSearchParams({ season, team, stat })
  return fetchApi(`/api/trends?${params}`)
}
