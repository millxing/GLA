# PBP.md

## Purpose
This document defines the canonical Play-By-Play (PBP) data layout and operational workflow for GLA and NBA_Data.

## Canonical Locations
- GLA code: `/Users/robschoen/Dropbox/CC/GLA`
- NBA data repo: `/Users/robschoen/Dropbox/CC/NBA_Data`
- Canonical PBP root (raw + processed): `/Users/robschoen/Dropbox/CC/NBA_Data/PBPdata`
- Scheduler script: `/Users/robschoen/Dropbox/CC/GLA/backend/admin/scripts/update_and_push.sh`

## Runtime Standard (Critical)
- Canonical admin/runtime interpreter:
  - `/opt/miniconda3/envs/gla_admin/bin/python`
- Required versions:
  - `nba_api==1.11.4`
  - `scikit-learn==1.8.0`
- Scheduler preflight validates python path, nba_api, scikit-learn, and parquet engine before running updates.
- `.venv` and `.venv311` are no longer canonical for scheduled/admin PBP workflows.
- Win-probability batch prediction wrappers filter known non-fatal sklearn parallel `UserWarning` messages; this does not change WP outputs.

## PBPdata Layout
Under `/Users/robschoen/Dropbox/CC/NBA_Data/PBPdata`:

- Raw:
  - `nbastatsv3/regular/nbastatsv3_YYYY.parquet`
  - `nbastatsv3/playoffs/nbastatsv3_po_YYYY.parquet`
  - `api_pbpv3/regular/api_pbpv3_YYYY.csv`
  - `api_pbpv3/playoffs/api_pbpv3_po_YYYY.csv`
  - `cdnnba/regular/cdnnba_YYYY.csv`
  - `cdnnba/playoffs/cdnnba_po_YYYY.csv`
  - `manifest.csv`
- Processed:
  - `game_states/<phase>/<season>/_states_<season>_<phase>.parquet`
    - Includes `payload_json` and `home_win_prob_by_event_json`.
    - `payload_json` event objects include cached `home_win_prob` values.
  - `game_states/<phase>/<season>/_timeline_metrics_<season>_<phase>.json`
  - `winprob_models/wpm_<season>.json`
  - `winprob_base/<phase>/stacked_<season>_winprob_base.csv`
- One-off backfill staging:
  - `raw_backfill/`
  - `raw_backfill_playin/`

## Git Tracking Policy
- Track in git:
  - `PBPdata/game_states/**`
  - `PBPdata/winprob_models/**`
- Do not track in git (rebuildable intermediates):
  - `PBPdata/winprob_base/**`
- Keep existing ignores:
  - `PBPdata/cdnnba/**`
  - `PBPdata/nbastatsv3_csv_archive/**`

## Staged Cutover Note
- Temporary staged behavior is enabled via `PBP_ENABLE_LEGACY_GLA_FALLBACK=1`:
  - Read order is `NBA_Data/PBPdata` first, then legacy `GLA/data/pbp/processed`, then GitHub raw fallback for deploy environments.
- After the first successful post-cutover scheduler verification window, disable legacy fallback and remove remaining legacy references.

## Update Commands
Run from `/Users/robschoen/Dropbox/CC/GLA`.

Historical backfill:
```bash
/opt/miniconda3/envs/gla_admin/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  backfill-pbp-raw \
  --start 2000-01 \
  --end 2025-26 \
  --source-dir /path/to/shuf_datasets
```

Incremental raw update:
```bash
/opt/miniconda3/envs/gla_admin/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  update-pbp-raw \
  --season 2025-26 \
  --phase regular \
  --workers 8 \
  --request-timeout 8 \
  --retries 1
```

Build timeline game states (writes to NBA_Data/PBPdata):
```bash
/opt/miniconda3/envs/gla_admin/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  build-pbp-game-states \
  --season 2025-26 \
  --phase regular \
  --output-root /Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/game_states
```

Pack game states to parquet and remove per-game JSON:
```bash
/opt/miniconda3/envs/gla_admin/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  pack-pbp-game-states \
  --season 2025-26 \
  --phase regular \
  --input-root /Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/game_states \
  --compression zstd \
  --overwrite \
  --delete-json
```

Build timeline excitement/comeback metrics:
```bash
/opt/miniconda3/envs/gla_admin/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  build-pbp-timeline-metrics \
  --season 2025-26 \
  --phase regular \
  --input-root /Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/game_states \
  --output-root /Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/game_states \
  --overwrite
```

Build winprob base:
```bash
/opt/miniconda3/envs/gla_admin/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  build-pbp-winprob-base \
  --season 2025-26 \
  --phase regular \
  --input-root /Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/game_states \
  --output-root /Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/winprob_base
```

Build/rebuild winprob model artifacts:
```bash
/opt/miniconda3/envs/gla_admin/bin/python backend/admin/cli.py \
  build-pbp-winprob-models \
  --phase regular \
  --output-root /Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/winprob_models \
  --overwrite
```

## Scheduler Behavior
- Daily launch target is 6:00 AM local via LaunchAgent.
- Scheduler updates raw PBP, rebuilds regular/playoff timeline states, packs parquet, builds per-game timeline metrics, updates index files, and commits through NBA_Data `commit-and-push`.
- Timeline artifacts now write to:
  - `/Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/game_states`
- Daily scheduler guardrails:
  - Packed game-state parquet must include `home_win_prob_by_event_json`.
  - Timeline metrics JSON must be written for each processed phase:
    - `/Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/game_states/<phase>/<season>/_timeline_metrics_<season>_<phase>.json`

## Timeline Metric Definitions
- Event WP:
  - Home win probability is computed for each event and cached both in:
    - Per-event payload: `events[*].home_win_prob`
    - Packed parquet column: `home_win_prob_by_event_json`
- Excitement factor:
  - Interval unit is change of possession (not fixed clock intervals).
  - For each possession change, compute `abs(current_home_wp - previous_home_wp)`.
  - Metric is `100 * average(abs_delta_over_possession_changes)`.
  - If no valid possession changes exist, value is `0.0`.
- Comeback factor:
  - If home team wins: `1 - min(home_win_prob)`.
  - If road team wins: `max(home_win_prob)` (home's peak WP before losing).
  - If winner/WP is unavailable (or tie): `0.0`.

## Situational Filters (Garbage-Time Filtered + Clutch)
- Scope intent:
  - In addition to full-game (`all`) stats/contributions, maintain two persisted alternates:
    - `garbage_filtered`: excludes garbage-time events.
    - `clutch`: includes only clutch events.
- Event classification source of truth:
  - Classification thresholds use timeline payload state (`events[*]`) only.
  - Threshold inputs are:
    - Win probability: `events[*].home_win_prob`.
    - Score differential: `events[*].game_log_state.pts_home - events[*].game_log_state.pts_road`.
    - Period/clock: `events[*].period`, `events[*].clock`.
  - Attribution mode is switchable in the builder:
    - `pre` (default): attribute each event to the prior event's classified state.
    - `post`: attribute each event to its own resulting classified state.
- Garbage-time state definition (stateful latch):
  - Enter garbage when all are true:
    - `period >= 3` (second half and OT),
    - `abs(point_differential) > 5`,
    - `home_win_prob >= garbage_wp_on` or `home_win_prob <= (1 - garbage_wp_on)`,
    - not in the final minute of the game (`period == final_period` and `seconds_left_in_period < 60`).
  - Once active, garbage stays latched until:
    - `(1 - garbage_wp_off) < home_win_prob < garbage_wp_off`.
  - Defaults are `garbage_wp_on = 0.95` and `garbage_wp_off = 0.90`.
- Clutch event definition:
  - `abs(point_differential) <= 5`, and
  - `period >= 4` (4th quarter or OT), and
  - `seconds_left_in_period < 300`.
- Boundary behavior:
  - Garbage entry uses inclusive edge checks (`>= garbage_wp_on`, `<= 1-garbage_wp_on`).
  - Garbage cannot newly enter during the final game minute; this gate affects entry only.
  - Garbage exit uses a strict interior check (`(1-garbage_wp_off) < wp < garbage_wp_off`).
  - Clutch uses `<= 5` differential and strict `< 300` seconds.
  - Because garbage requires `abs(diff) > 5` and clutch requires `abs(diff) <= 5`, they are mutually exclusive by construction.
- Persistence requirement:
  - Filtered stats/contributions must be stored as durable artifacts in `NBA_Data` and reused by API/UI.
  - Recompute scope should be incremental (new game IDs only), not full historical reruns, during daily updates.
  - If garbage latch thresholds/logic change, run a non-incremental rebuild for situational CSVs and scoped contributions.
- Scoped advanced minutes + pace:
  - In `box_score_advanced_<scope>_<season>.csv`, `minutes_home` and `minutes_road` represent in-scope elapsed time (team player-minutes), not full-game 240/265 minutes.
  - Minutes are derived from event timeline elapsed windows using the selected builder attribution mode (`--scope-state-mode pre|post`).
  - If timeline elapsed cannot be resolved, builder falls back to possession-ratio scaling from full-game advanced minutes.
  - This ensures pace in scoped contributions uses scoped possessions over scoped minutes.
- Persisted artifact names (season-scoped):
  - `team_game_logs_garbage_filtered_<season>.csv`
  - `team_game_logs_clutch_<season>.csv`
  - `box_score_advanced_garbage_filtered_<season>.csv`
  - `box_score_advanced_clutch_<season>.csv`
  - `contributions/contributions_garbage_filtered_<season>.json`
  - `contributions/contributions_clutch_<season>.json`
- Builder commands:
  - Incremental daily update:
    - `/opt/miniconda3/envs/gla_admin/bin/python backend/admin/build_situational_gamelogs.py --season <season> --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data --incremental --scope-state-mode pre --garbage-wp-on 0.95 --garbage-wp-off 0.90`
  - Rule-change historical rebuild (non-incremental):
    - `/opt/miniconda3/envs/gla_admin/bin/python backend/admin/build_situational_gamelogs.py --season <season> --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data --scope-state-mode pre --garbage-wp-on 0.95 --garbage-wp-off 0.90`
- API/UX contract (data scope):
  - Use data-scope values `all`, `garbage_filtered`, and `clutch` for backend endpoint selection and frontend module toggles.

## Guardrails
- Keep schemas aligned with canonical `api_pbpv3` columns and `manifest.csv` updates.
- Do not add ad hoc derived analytics columns into raw source files.
- Validate coverage against `team_game_logs_YYYY-YY.csv` after major updates.
