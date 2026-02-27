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
  - `game_states/<phase>/<season>/...`
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
- Scheduler updates raw PBP, rebuilds regular/playoff timeline states, packs parquet, updates index files, and commits through NBA_Data `commit-and-push`.
- Timeline artifacts now write to:
  - `/Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/game_states`

## Guardrails
- Keep schemas aligned with canonical `api_pbpv3` columns and `manifest.csv` updates.
- Do not add ad hoc derived analytics columns into raw source files.
- Validate coverage against `team_game_logs_YYYY-YY.csv` after major updates.
