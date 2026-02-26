# PBP.md

## Purpose
This file documents the raw Play-By-Play (PBP) data layer used by GLA so future threads can work from a consistent baseline.

Scope of this layer:
- Raw ingestion and storage only.
- No garbage-time logic.
- No win-probability modeling.
- No derived analytics tables.

## Canonical Locations
- GLA code (CLI logic): `/Users/robschoen/Dropbox/CC/GLA/backend/admin/cli.py`
- NBA data repo (raw/output storage): `/Users/robschoen/Dropbox/CC/NBA_Data`
- Historical source files: `/Users/robschoen/Dropbox/CC/GLA/shuf_datasets`
- Raw PBP target root: `/Users/robschoen/Dropbox/CC/NBA_Data/PBPdata`

## PBPdata Directory Layout
Under `/Users/robschoen/Dropbox/CC/NBA_Data/PBPdata`:

- `nbastatsv3/regular/nbastatsv3_YYYY.parquet`
- `nbastatsv3/playoffs/nbastatsv3_po_YYYY.parquet`
- `nbastatsv3_csv_archive/` (git-ignored archive for pre-migration CSVs)
- `cdnnba/regular/cdnnba_YYYY.csv`
- `cdnnba/playoffs/cdnnba_po_YYYY.csv`
- `api_pbpv3/regular/api_pbpv3_YYYY.csv`
- `api_pbpv3/playoffs/api_pbpv3_po_YYYY.csv`
- `manifest.csv`

## Processed PBP Working Directory (GLA)
For local visualizations/notebooks in this repo, use:

- `/Users/robschoen/Dropbox/CC/GLA/data/pbp/processed/game_states/<phase>/<season>/`
- `/Users/robschoen/Dropbox/CC/GLA/data/pbp/processed/winprob_base/<phase>/stacked_<season>_winprob_base.csv`
- `/Users/robschoen/Dropbox/CC/GLA/data/pbp/processed/game_logs/team_game_logs_<season>.csv`

This is the canonical replacement for legacy `GLA/temp/temp_*pbp*` folders.

## Manifest
File: `/Users/robschoen/Dropbox/CC/NBA_Data/PBPdata/manifest.csv`

Columns:
- `source`
- `season`
- `season_type`
- `file_path`
- `row_count`
- `game_count`
- `sha256`
- `updated_at`

`manifest.csv` is updated by the CLI after each write/normalize pass.

## Canonical Schema
All normalized `api_pbpv3` files are enforced to this 24-column schema (same names/order as `nbastatsv3`):

1. `actionNumber`
2. `clock`
3. `period`
4. `teamId`
5. `teamTricode`
6. `personId`
7. `playerName`
8. `playerNameI`
9. `xLegacy`
10. `yLegacy`
11. `shotDistance`
12. `shotResult`
13. `isFieldGoal`
14. `scoreHome`
15. `scoreAway`
16. `pointsTotal`
17. `location`
18. `description`
19. `actionType`
20. `subType`
21. `videoAvailable`
22. `shotValue`
23. `actionId`
24. `gameId`

Normalization behavior for `api_pbpv3`:
- Sorted by `gameId`, then `actionNumber`.
- Numeric defaults use `0` for key integer-like fields.
- `actionId` falls back to `actionNumber` if missing.
- `scoreHome` and `scoreAway` remain nullable numeric (to match historical behavior).

## Current 2025-26 Regular Season Status
As of 2026-02-22:
- Team game logs (`team_game_logs_2025-26.csv`) games: `845`
- `api_pbpv3_2025.csv` games: `845`
- Missing games vs team logs: `0`
- `api_pbpv3_2025.csv` rows: `485671`

Important note:
- Historical `nbastatsv3` snapshots in `shuf_datasets` have partial 2025-26 coverage (`482` games), so row/game totals differ from fully updated current-season outputs.

## Update Commands
Run from `/Users/robschoen/Dropbox/CC/GLA`.

Historical backfill:
```bash
.venv/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  backfill-pbp-raw \
  --start 2000-01 \
  --end 2025-26 \
  --source-dir /Users/robschoen/Dropbox/CC/GLA/shuf_datasets
```

Incremental updates (recommended fast settings):
```bash
.venv/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  update-pbp-raw \
  --season 2025-26 \
  --phase regular \
  --workers 8 \
  --request-timeout 8 \
  --retries 1
```

Dry-run check:
```bash
.venv/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  update-pbp-raw \
  --season 2025-26 \
  --phase regular \
  --dry-run
```

## Processed Build Commands (For Timeline + Notebook)
Run from `/Users/robschoen/Dropbox/CC/GLA`.

1) Build per-game state JSON files into the GLA processed directory:
```bash
.venv/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  build-pbp-game-states \
  --season 2023-24 \
  --phase regular \
  --output-root /Users/robschoen/Dropbox/CC/GLA/data/pbp/processed/game_states
```

2) Build stacked win-probability baseline CSV from those JSON files:
```bash
.venv/bin/python backend/admin/cli.py \
  --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data \
  build-pbp-winprob-base \
  --season 2023-24 \
  --phase regular \
  --input-root /Users/robschoen/Dropbox/CC/GLA/data/pbp/processed/game_states \
  --output-root /Users/robschoen/Dropbox/CC/GLA/data/pbp/processed/winprob_base
```

3) Copy team logs for timeline dropdown date labels:
```bash
cp /Users/robschoen/Dropbox/CC/NBA_Data/team_game_logs_2023-24.csv \
   /Users/robschoen/Dropbox/CC/GLA/data/pbp/processed/game_logs/team_game_logs_2023-24.csv
```

## Fast Fetch Algorithm (Implemented)
`update-pbp-raw` now:
- Uses local team game logs first for game discovery (avoids slow season-list API calls).
- Fetches games in parallel with a configurable worker pool.
- Uses short configurable request timeout (default 12s).
- Uses configurable retries.
- Falls back from `nba_api` `PlayByPlayV3` to `cdn.nba.com` liveData endpoint per game.

## Expected Differences Between Sources
Even with consistent schema, row counts per game can differ between:
- `nbastatsv3` historical snapshots.
- `api_pbpv3` (mixed `nba_api` + CDN fallback).

Known pattern:
- Many CDN-fetched games have `videoAvailable=0` and `shotValue=0` for all rows.
- This is source-population behavior, not schema drift.

## Guardrails For Future Threads
- Keep this layer raw-only. Do not add derived analytics columns in `PBPdata`.
- Do not change frontend/backend runtime endpoints just to update PBP raw storage.
- If defaults/typing drift appears, re-run `update-pbp-raw` with no missing games; it rewrites/normalizes the existing file.
- Validate coverage against `team_game_logs_YYYY-YY.csv` after major updates.
