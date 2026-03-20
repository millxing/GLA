# AGENTS.md

## Project Overview
NBA Game Log Analytics (GLA) is a full-stack app with a Python API backend and a Vite/React frontend. It ingests NBA game data, serves analytics endpoints, and renders dashboards/visualizations in the web UI.
All data is accessed from my github repo, https://github.com/millxing/NBA_Data. The seasons covered are 2000-2001 to 2025-26 (current season as of 2/7/2026).
Data in the repo: team_game_logs_XXXX-XX.csv, linescores_XXXX-XX.csv, box_score_advanced_XXXX-XX.csv (only possessions and minutes), LLM game interpretations, and contribution jsons.
There are currerntly four modules in the app.
Game Analysis (GA): analyze idividual games using four-factor contribution models
League Summary (LS): sortable table showing team performance in various statisical categories. Also identifies largest contributors to net rating. 
Statistical Trends (ST): graphic view of time series for team statistical categories. 
Contribution Analysis (CA): Decomposes a team's net rating into the eight factors (four each for team and opponent).

Notes/assumptions:
- `README.md` is empty; details below are inferred from `DEPLOYMENT.md`, top-level structure, and filenames.
- Backend is assumed to be FastAPI because deployment uses `uvicorn main:app`.

## High-Level Architecture
- **Backend (Python)**: API service, data loading, caching, calculations, and schemas. Likely FastAPI with routers/services modules.
- **Frontend (React + Vite)**: SPA consuming `/api` endpoints; Vite dev server proxies API.
- **Data/Admin tooling**: `backend/admin` appears to contain CLI/scripts for data updates and interpretation generation.
- **Deployment**: `render.yaml` suggests Render deployment for both backend and frontend.

## Key Directories and Entry Points
- `backend/main.py`: Backend entry point (served with `uvicorn main:app`).
- `backend/routers/`: API routes (see `backend/routers/api.py` per `DEPLOYMENT.md`).
- `backend/services/`: Core logic (cache, calculations, data loading).
- `backend/schemas/`: Data models/schemas.
- `backend/admin/`: Admin CLI and scripts for data updates.
- `frontend/src/main.jsx`: Frontend entry point.
- `frontend/src/App.jsx`: App root component.
- `frontend/src/pages/`: Page-level views (Home, FourFactor, LeagueSummary, Trends).
- `frontend/src/components/`: Shared UI components.
- `render.yaml`: Render blueprint for deployment.

## How To Run (from `DEPLOYMENT.md`)
Backend:
1. `cd backend`
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `cp .env.example .env` (optional)
5. `python main.py` (API on `http://localhost:8000`)

Frontend:
1. `cd frontend`
2. `npm install`
3. `npm run dev` (UI on `http://localhost:5173`)

Build/preview:
- `npm run build`
- `npm run preview`

## Update Troubleshooting
- If `backend/admin/cli.py update-data` starts failing (timeouts/errors despite valid commands), check and update `nba_api` first. Current known-good version: `1.11.4`.
- Keep `NBA_Data/PBPdata/` git-ignored. Raw PBP CSV files can exceed GitHub size limits and can break the daily `commit-and-push` step if staged.
- If the scheduled overnight update is missed (for example after a reboot), manually run the full scheduler pipeline from the GLA repo root:
  - `cd /Users/robschoen/Dropbox/CC/GLA`
  - `SEASON=2025-26 bash backend/admin/scripts/update_and_push.sh`
- The canonical admin interpreter for that workflow is `/opt/miniconda3/envs/gla_admin/bin/python`. The script checks `nba_api==1.11.4`, `scikit-learn==1.8.0`, and writes a report to `reports/updateYYYYMMDD.txt`.
- After each successful `NBA_Data` push, the scheduler now waits for the updated files to become visible on GitHub raw and then clears the Render API cache automatically. This avoids the old 30-minute stale-data window after updates.
- Automatic cache clearing uses `POST https://extrapass-api.onrender.com/api/admin/clear-cache` and requires a matching secret via `CACHE_CLEAR_KEY` or `ADMIN_SECRET_KEY` in the updater environment.
- Useful variants:
  - Preview only: `DRY_RUN=1 SEASON=2025-26 bash backend/admin/scripts/update_and_push.sh`
  - Skip LLM interpretations: `ENABLE_INTERPRETATIONS=0 SEASON=2025-26 bash backend/admin/scripts/update_and_push.sh`
- If only the narrow season CSV refresh is needed instead of the full nightly workflow:
  - `cd /Users/robschoen/Dropbox/CC/GLA`
  - `/opt/miniconda3/envs/gla_admin/bin/python -m backend.admin.cli --repo-dir /Users/robschoen/Dropbox/CC/NBA_Data update-data --season 2025-26`

Notes/assumptions:
- No explicit test or lint commands are documented in the repo root or `DEPLOYMENT.md`.

## Coding Conventions / Patterns (Inferred)
- Backend is structured by **routers/services/schemas**, implying a layered FastAPI-style architecture.
- Data/cache logic sits in `backend/services`, with models in `backend/schemas`.
- Frontend is a Vite React SPA with `pages/` and `components/` separation and CSS modules per page/component.

## Contribution/Model Process (Current)
- Per-game contribution JSONs are generated by `backend/admin/generate_contributions.py` and written to `NBA_Data/contributions/contributions_XXXX-XX.json`.
- Default training window for contributions:
  - `2001-02` onward: current-season games before game date + up to 7 prior seasons.
  - `2000-01` only: in-sample exception using full-season training (ensures no missing game contribution entries).
- Possession policy for turnover/ball-handling logic: use actual possessions from `box_score_advanced_XXXX-XX.csv`; do not use the 0.44 possession approximation.
- Game Analysis decomposition endpoint (`/api/decomposition`) reads pre-generated contribution JSONs rather than retraining models on request.
- Regeneration scope guidance:
  - If a change only affects one season (or a single-season exception), regenerate only that season's contribution JSON.
  - If a change affects model windows or shared factor math, regenerate all seasons' contribution JSONs.

Notes/assumptions:
- Naming and structure are inferred from directory layout and `DEPLOYMENT.md`, not from deep file inspection.
