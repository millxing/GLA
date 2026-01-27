# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Game Log Analytics (GLA) - A web application for analyzing NBA game statistics using "Four Factors" and "Eight Factors" decomposition models. The app shows how much each statistical factor (shooting, ball handling, rebounding, free throws) contributed to a game's outcome.

## Development Commands

### Backend (FastAPI)
```bash
cd backend
python -m venv venv && source venv/bin/activate  # setup only
pip install -r requirements.txt                   # setup only
python main.py                                    # starts server at http://localhost:8000
```
API docs available at http://localhost:8000/docs

### Frontend (React/Vite)
```bash
cd frontend
npm install       # setup only
npm run dev       # starts dev server at http://localhost:5173
npm run build     # production build
npm run preview   # preview production build
```

### Admin CLI (Data Updates & Model Training)
Located at `backend/admin/cli.py`. Manages the external NBA_Data repository.
```bash
cd backend
python admin/cli.py update-data --season 2024-25
python admin/cli.py train-models --seasons 2023-24,2024-25 --output 2023-2025.json
python admin/cli.py list-models
python admin/cli.py git-status
```

## Architecture

### Backend Structure
- **main.py** - FastAPI app entry point, CORS configuration
- **routers/api.py** - All API endpoints under `/api` prefix
- **services/data_loader.py** - Fetches CSV data from GitHub (millxing/NBA_Data), caches responses
- **services/calculations.py** - Four/Eight Factors computations, league aggregates, trend series
- **services/cache.py** - TTL-based in-memory cache
- **schemas/models.py** - Pydantic models for API responses
- **config.py** - Data source URLs, available models, season helpers

### Frontend Structure
- **src/App.jsx** - React Router setup with 4 routes
- **src/api.js** - API client functions (getSeasons, getGames, getDecomposition, etc.)
- **src/pages/** - Page components: Home, FourFactor, LeagueSummary, Trends
- **src/components/Layout.jsx** - Shared navigation layout

### Data Flow
1. Backend fetches raw CSV data from GitHub (team_game_logs, box_score_advanced, linescores)
2. Data is normalized and cached in memory (1 hour TTL)
3. Calculations are performed on-demand (Four Factors, ratings, decomposition)
4. Frontend calls API endpoints and renders data with Recharts

### Key Concepts
- **Four Factors**: eFG%, TOV%, OREB%, FT Rate - metrics that explain game outcomes
- **Eight Factors**: Same as Four Factors but split into home and road team contributions centered on league averages
- **Decomposition**: Uses linear regression coefficients from trained models to attribute net rating differential to each factor

## Environment Variables

Backend (.env):
- `ALLOWED_ORIGINS` - CORS origins (default: localhost:5173,localhost:3000)
- `PORT` - Server port (default: 8000)

Frontend (.env):
- `VITE_API_URL` - Backend API URL (empty uses relative URLs with Vite proxy)
