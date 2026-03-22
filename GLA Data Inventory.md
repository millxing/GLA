# GLA Codebase Data Inventory

## Context
Inventory of all raw data accessed, data calculated, data stored, and API endpoints in the NBA Game Log Analytics codebase.

---

## 1. Raw Data Sources (fetched from GitHub `millxing/NBA_Data`)

### Per-Season CSVs

| File | Key Columns | Purpose |
|------|-------------|---------|
| `team_game_logs_{SEASON}.csv` | game_id, game_date, season, game_type, team_abbreviation_home/road, fgm/fga/fg3m/fg3a/ftm/fta/oreb/dreb/tov/pts (home & road), wl_home/road, plus_minus | Core box score stats, 1 row per game |
| `box_score_advanced_{SEASON}.csv` | game_id, possessions_home/road, minutes_home/road | Actual possession counts (never estimated) |
| `linescores_{SEASON}.csv` | game_id, pts_qtr1-4_home/road, pts_ot_total_home/road | Quarter-by-quarter scoring |

### Scoped Variants (situational)
Same schema as above, filtered by game situation:
- `team_game_logs_{SCOPE}_{SEASON}.csv` + `box_score_advanced_{SCOPE}_{SEASON}.csv`
- Scopes: `garbage_filtered`, `clutch`, `q1`-`q4`, `h1`, `h2`, `ot`
- Built by `build_situational_gamelogs.py` from PBP game-state data

### Pre-Generated JSONs

| File | Content |
|------|---------|
| `contributions/contributions_{SEASON}.json` (+ scoped variants) | Per-game eight-factor decomposition: factor values, contributions, ratings, linescore, model metadata |
| `interpretations/gamesummaries_{SEASON}_2018-25.json` | LLM-generated game summaries keyed by game_id, for both four_factors and eight_factors |

### Play-by-Play Data
- `PBPdata/PBP_nbastatsv3_{SEASON}_{phase}.csv` — raw event-level PBP
- `PBPdata/game_states/{phase}/{SEASON}/_states_{SEASON}_{phase}.parquet` — packed game-state snapshots with win probability
- `PBPdata/game_states/{phase}/{SEASON}/_timeline_metrics_{SEASON}_{phase}.json` — excitement_factor, comeback_factor per game
- `PBPdata/game_states/{phase}/{SEASON}/_index_{SEASON}_{phase}.json` — game index for PBP viewer

---

## 2. Data Calculated at Runtime (backend/services/calculations.py)

### Four Factors (per team per game)
- **eFG%** = (FGM + 0.5 × FG3M) / FGA × 100
- **TOV%** = TOV / actual_possessions × 100
- **Ball Handling** = 100 − TOV%
- **OREB%** = OREB / (OREB + Opp_DREB) × 100
- **FT Rate** = FTM / FGA × 100

### Ratings (per team per game)
- **ORtg** = PTS / possessions × 100
- **DRtg** = Opp_PTS / Opp_possessions × 100
- **Net Rating** = ORtg − DRtg
- **Pace** = avg(home_poss, road_poss) × 48 / actual_minutes

### Decomposition (from pre-generated contribution JSONs)
- **Eight Factors**: home_contribution = coef × (home_value − league_avg) / 100; road negated
- **Four Factors**: contribution = coef × differential / 100
- predicted_rating_diff = intercept + Σ contributions

### League Aggregates (computed on demand for league-summary endpoint)
Per-team over filtered date range:
- W/L, Win%, PPG, Opp PPG
- All four factors + opponent versions
- ORtg, DRtg, Net Rating, Pace
- **SOS** = game-weighted avg opponent net rating
- **Adj ORtg** = ORtg − Def_SOS; **Adj DRtg** = DRtg − Off_SOS; **Adj Net** = Net + SOS

### Trend Series (computed on demand)
- Per-game stat value + 5-game and 10-game moving averages for 27 stats

### Contribution Analysis (computed on demand)
- Averages per-game contributions from JSON over a date range for one team
- Identifies top 4 factors by |contribution|, builds trend mini-charts

### League Top Contributors (computed on demand)
- Ranks all team-factor combinations; returns top/bottom 10

### Game Runs (from PBP timeline)
- Extracts possession sequences, scores `delta_home_wp / (poss_count+1)^alpha`
- Returns top N non-overlapping runs

---

## 3. Data Stored / Cached

| Layer | Mechanism | TTL / Scope |
|-------|-----------|-------------|
| In-memory cache | `cachetools.TTLCache` (50 entries, 30 min TTL) | CSV fetches, normalized DataFrames, JSON fetches |
| NBA_Data repo (GitHub) | Git-versioned CSVs + JSONs | Updated nightly by `update_and_push.sh` |
| Local NBA_Data clone | Same files on disk | Used when `NBA_DATA_REPO_DIR` is set |
| Reports | `reports/updateYYYYMMDD.txt` | Nightly pipeline run logs |

---

## 4. API Endpoints

### Season & Game Selection
| Endpoint | Method | Params | Returns |
|----------|--------|--------|---------|
| `/api/seasons` | GET | — | List of seasons (2025-26 → 2000-01) |
| `/api/games` | GET | season, data_scope | Game list: game_id, date, teams, points, game_type, is_overtime |
| `/api/teams` | GET | season, data_scope | List of team abbreviations |

### Game Analysis (GA page)
| Endpoint | Method | Params | Returns |
|----------|--------|--------|---------|
| `/api/decomposition` | GET | season, game_id, factor_type, data_scope | Full decomposition: factors, contributions, ratings, linescore, league avgs, factor ranges |
| `/api/contributions/single-game` | GET | season, game_id, data_scope | Raw contribution JSON entry for one game |
| `/api/game-timeline` | GET | season, game_id, game_type?, home_team?, road_team? | Event-by-event timeline with win probability, excitement/comeback factors |
| `/api/game-runs` | GET | season, game_id, maxposs, minposs, minmargin, run_alpha, limit | Top scoring runs with WP deltas |
| `/api/interpretation` | POST | InterpretationRequest body | LLM-generated game summary (pre-generated or real-time) |
| `/api/interpretation/prompt` | GET | season, game_id, factor_type, data_scope | Raw LLM prompt for debugging |

### League Summary (LS page)
| Endpoint | Method | Params | Returns |
|----------|--------|--------|---------|
| `/api/league-summary` | GET | season, data_scope, start_date?, end_date?, exclude_playoffs?, last_n_games? | 30-team stats table + league averages |
| `/api/league-top-contributors` | GET | season, data_scope, start_date?, end_date?, exclude_playoffs?, last_n_games? | Top 10 positive/negative team-factor contributors |

### Statistical Trends (ST page)
| Endpoint | Method | Params | Returns |
|----------|--------|--------|---------|
| `/api/trends` | GET | season, team, stat, exclude_non_regular?, data_scope | Per-game time series with 5/10-game moving averages |

### Contribution Analysis (CA page)
| Endpoint | Method | Params | Returns |
|----------|--------|--------|---------|
| `/api/contribution-analysis` | GET | season, team, date_range_type, last_n_games?, start_date?, end_date?, exclude_playoffs?, data_scope | Averaged contributions, top 4 factors with trend charts |

### Win Probability
| Endpoint | Method | Params | Returns |
|----------|--------|--------|---------|
| `/api/winprob/model-seasons` | GET | — | Seasons with trained WP models |
| `/api/winprob/forecast` | GET | season, game_id, game_seconds_left, phase | Home win probability |
| `/api/winprob/hypothetical-forecast` | GET | season, quarter, seconds_left, differential, possession_numeric, phase | Hypothetical home WP |
| `/api/winprob/app` | GET | — | Interactive WP HTML app |
| `/api/winprob/hypothetical-app` | GET | — | Hypothetical WP HTML app |

### Admin
| Endpoint | Method | Params | Returns |
|----------|--------|--------|---------|
| `/api/admin/clear-cache` | POST | key (secret) | Clears in-memory cache |
| `/api/version` | GET | — | Git hash, Python/package versions |

---

## 5. Admin Data Pipeline (offline generation)

### Nightly flow (`update_and_push.sh`)
1. `cli.py update-data` — fetch new games from nba_api → season CSVs
2. `cli.py update-pbp-raw` — fetch raw PBP events → PBP CSVs
3. `cli.py build-pbp-game-states` → packed Parquet with WP
4. `cli.py build-pbp-timeline-metrics` → excitement/comeback JSONs
5. `build_situational_gamelogs.py` → scoped CSVs (clutch, garbage_filtered, quarters, halves)
6. Commit & push to NBA_Data; wait for GitHub propagation
7. `generate_contributions.py` → contribution JSONs (all, garbage_filtered, clutch scopes)
8. Commit & push contributions
9. `cli.py generate-interpretations` → LLM game summaries (optional)
10. Commit & push interpretations
11. Clear Render API cache

---

## 6. Frontend Page → Endpoint Mapping

| Page | Endpoints Used |
|------|---------------|
| Home | None (static) |
| Game Analysis (GA) | seasons, games, decomposition, interpretation, game-timeline, interpretation/prompt |
| League Summary (LS) | seasons, league-summary, league-top-contributors |
| Statistical Trends (ST) | seasons, teams, trends |
| Contribution Analysis (CA) | seasons, teams, league-summary, contribution-analysis |
| Blog | None (static markdown) |
