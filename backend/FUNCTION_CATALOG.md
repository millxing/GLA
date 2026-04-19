# Function Catalog

This file catalogs the specialized analytics functions and endpoints in the backend that have custom purpose-specific logic, non-obvious parameters, or interpretation caveats.

Scope rules:
- Document specialized analytics surfaces, not every helper.
- Prefer documenting stable semantics and defaults rather than implementation detail.
- Update this file whenever a specialized endpoint or scoring/filter parameter changes.

## Run Analysis

### `GET /api/game-runs`
- Source: `backend/routers/api.py`
- Purpose: return the top `limit` non-overlapping contiguous possession windows in a game, ranked by win-probability swing after a length penalty.
- Inputs:
  - `season`: season string like `2025-26`
  - `game_id`: numeric game id
  - `game_type`: optional phase hint such as `regular_season`, `playoffs`, `play_in`
  - `home_team`, `road_team`: optional team disambiguators
  - `numerator`: `dwp` for home win-probability swing or `dscore` for score-margin swing
  - `maxposs`: maximum possession-window length; positive integer or `"inf"` for no cap
  - `minposs`: minimum possession-window length; default `1`
  - `minmargin`: minimum absolute score-margin swing across the run; default `0`
  - `run_alpha`: exponent in the length penalty; default `0.6`
  - `limit`: number of non-overlapping runs to return; default `4`
- Scoring:
  - If `numerator=dwp`: `run_score = delta_home_win_prob / (possession_count + 1) ^ run_alpha`
  - If `numerator=dscore`: `run_score = score_margin_delta / (possession_count + 1) ^ run_alpha`
  - Ranking uses `abs(run_score)`, then absolute numerator value, then shorter run, then earlier start.
- Semantics:
  - `possession_count` means total consecutive possessions in the game flow, counting both teams combined.
  - Non-overlap is enforced on possession indices, not event indices.
  - `run_side` / `run_team` indicate which team benefited in win-probability terms.
  - `minmargin` filters on absolute score swing, using `home_points_scored - road_points_scored`.
- Output:
  - game metadata
  - normalized run parameters actually used
  - list of runs with possession span, event span, descriptions, score swing, WP swing, and rank
- Common use:
  - identify “best runs” in a game without manually scanning PBP or timeline charts

### `extract_timeline_possessions(raw_events, home_team, road_team)`
- Source: `backend/services/game_runs.py`
- Purpose: collapse timeline events into possession segments using stored possession-state metadata.
- Inputs:
  - `raw_events`: timeline payload `events` array
  - `home_team`, `road_team`: abbreviations used for labeling possession owner
- Semantics:
  - Uses `possession_before_side`, `possession_after_side`, and `possession_changed`.
  - Uses the previous event as the possession anchor when the first usable possession is only visible in `after_side`.
  - Final open possession is closed at the last event.
- Output:
  - ordered possession segments with start/end event info, start/end score, and start/end home WP

### `rank_non_overlapping_runs(possessions, home_team, road_team, ...)`
- Source: `backend/services/game_runs.py`
- Purpose: enumerate candidate run windows over possessions, score them, then greedily select non-overlapping windows.
- Inputs:
  - `possessions`: output of `extract_timeline_possessions`
  - `home_team`, `road_team`: labels for assigning the benefiting team
  - `max_possessions`: `None` means unbounded
  - `numerator`: `dwp` or `dscore`
  - `run_alpha`: exponent in the length penalty
  - `min_possessions`: minimum window size
  - `min_margin`: minimum absolute score swing
  - `limit`: maximum number of runs to return
- Caveat:
  - If `max_possessions=None` and `run_alpha` is small, very long “game-control” stretches can outrank shorter intuitive runs.

## Timeline / Win Probability

### `GET /api/game-timeline`
- Source: `backend/routers/api.py`
- Purpose: return a game timeline with score state, possession owner, and per-event home win probability.
- Inputs:
  - `season`, `game_id`
  - optional `game_type`, `home_team`, `road_team`
- Behavior:
  - loads packed game-state payloads from `NBA_Data/PBPdata/game_states`
  - fills missing event WP from stored win-prob models when needed
  - returns cached timeline metrics such as excitement and comeback factor
- Output:
  - game metadata
  - `events[]` with clock, description, possession, score state, and `home_win_prob`
  - `excitement_factor`, `comeback_factor`, and percentiles when available

### `GET /api/winprob/forecast`
- Source: `backend/routers/api.py`
- Purpose: forecast win probability for an actual game state from stored win-prob base/model artifacts.
- Inputs:
  - `season`
  - `game_id`
  - `game_seconds_left`: seconds remaining in the full game
  - `phase`: artifact namespace, default `regular`

### `GET /api/winprob/hypothetical-forecast`
- Source: `backend/routers/api.py`
- Purpose: forecast win probability for a hypothetical game state not necessarily tied to a stored game.
- Inputs:
  - `season`
  - `quarter`
  - `seconds_left`: seconds remaining in current period
  - `differential`: home minus road score
  - `possession_numeric`: `1` home, `-1` road, `0` unknown
  - `phase`: artifact namespace, default `regular`

## Game-Level Factor Analysis

### `GET /api/decomposition`
- Source: `backend/routers/api.py`
- Purpose: return single-game factor decomposition from pre-generated contributions JSON.
- Inputs:
  - `season`
  - `game_id`
  - `factor_type`: typically `eight_factors`
  - `data_scope`: `all`, `garbage_filtered`, or `clutch`
- Output:
  - score, linescore, factor values, contributions, ratings, league averages
- Caveat:
  - This endpoint reads stored contribution artifacts; it does not retrain on request.

### `GET /api/contributions/single-game`
- Source: `backend/routers/api.py`
- Purpose: return a contributions payload narrowed to exactly one game while preserving the broader contribution JSON shape.
- Inputs:
  - `season`
  - `game_id`
  - `data_scope`

### `GET /api/interpretation/prompt`
- Source: `backend/routers/api.py`
- Purpose: render the exact LLM prompt that would be used for a game interpretation.
- Inputs:
  - `season`
  - `game_id`
  - `factor_type`
  - `data_scope`
- Use case:
  - inspect prompt content without making an LLM call

### `POST /api/interpretation`
- Source: `backend/routers/api.py`
- Purpose: return AI-generated game interpretation, preferring pre-generated text when available.
- Input body:
  - `InterpretationRequest`
- Constraints:
  - currently only `eight_factors`
  - currently only enabled for `2025-26`
  - currently only enabled for `data_scope=all`

## Team / League Analysis

### `GET /api/league-summary`
- Source: `backend/routers/api.py`
- Purpose: return team-level season or date-range aggregates for sortable league views.
- Inputs:
  - `season`
  - `start_date`, `end_date`
  - `exclude_playoffs`
  - `last_n_games`
  - `data_scope`: `all`, `garbage_filtered`, `garbage_time`, `clutch`, `non_clutch_time`
- Output:
  - `teams[]` with advanced and adjusted metrics
  - league averages and season date bounds

### `GET /api/trends`
- Source: `backend/routers/api.py`
- Purpose: return game-by-game time series for one team/stat pair.
- Inputs:
  - `season`
  - `team`
  - `stat`
  - `exclude_non_regular`
  - `data_scope`
- Output:
  - per-game values plus `ma_5` and `ma_10`

### `GET /api/contribution-analysis`
- Source: `backend/routers/api.py`
- Purpose: aggregate a team’s stored game-level contributions over a season, custom date range, or rolling window.
- Inputs:
  - `season`
  - `team`
  - `date_range_type`: `season`, `last_n`, `custom`
  - `last_n_games`
  - `start_date`, `end_date`
  - `exclude_playoffs`
  - `data_scope`
- Output:
  - aggregated contribution profile, record, predicted net rating, top contributors, mini-trend data

### `GET /api/league-top-contributors`
- Source: `backend/routers/api.py`
- Purpose: rank the strongest positive and negative contribution factors across the league over a selected window.
- Inputs:
  - `season`
  - `start_date`, `end_date`
  - `exclude_playoffs`
  - `last_n_games`
  - `data_scope`
- Caveat:
  - derived League Summary scopes such as `garbage_time` and `non_clutch_time` currently do not have persisted contribution JSON, so this can return empty lists for those scopes.

## Suggested Maintenance Pattern

When adding a new specialized function or endpoint:
- add one short entry here
- document defaults and semantic meaning of each non-obvious parameter
- call out any interpretation traps, especially when a parameter changes ranking behavior in unintuitive ways
