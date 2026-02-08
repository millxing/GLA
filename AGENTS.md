# AGENTS.md

## Project Overview
NBA Game Log Analytics (GLA) is a full-stack app with a Python API backend and a Vite/React frontend. It ingests NBA game data, serves analytics endpoints, and renders dashboards/visualizations in the web UI.
All data is accessed from my github repo, https://github.com/millxing/NBA_Data. The seasons covered are 2000-2001 to 2025-26 (current season as of 2/7/2026).
Data in the repo: team_game_logs_XXXX-XX.csv, linescores_XXXX-XX.csv, box_score_advanced_XXXX-XX.csv (only poseessions and minutes),contribution models, LLM game interpretations, and contribution jsons. 
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

Notes/assumptions:
- No explicit test or lint commands are documented in the repo root or `DEPLOYMENT.md`.

## Coding Conventions / Patterns (Inferred)
- Backend is structured by **routers/services/schemas**, implying a layered FastAPI-style architecture.
- Data/cache logic sits in `backend/services`, with models in `backend/schemas`.
- Frontend is a Vite React SPA with `pages/` and `components/` separation and CSS modules per page/component.

Notes/assumptions:
- Naming and structure are inferred from directory layout and `DEPLOYMENT.md`, not from deep file inspection.
