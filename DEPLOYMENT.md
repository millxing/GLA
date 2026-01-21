# NBA Game Log Analytics - Deployment Guide

## Repository Structure

```
GLA/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── routers/
│   │   ├── __init__.py
│   │   └── api.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── calculations.py
│   │   └── data_loader.py
│   └── schemas/
│       ├── __init__.py
│       └── models.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── .env.example
│   ├── .env.production
│   ├── public/
│   │   └── basketball.svg
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       ├── components/
│       │   ├── Layout.jsx
│       │   └── Layout.css
│       ├── pages/
│       │   ├── Home.jsx
│       │   ├── Home.css
│       │   ├── FourFactor.jsx
│       │   ├── FourFactor.css
│       │   ├── LeagueSummary.jsx
│       │   ├── LeagueSummary.css
│       │   ├── Trends.jsx
│       │   └── Trends.css
│       └── styles/
│           ├── index.css
│           └── variables.css
├── render.yaml
├── .gitignore
└── DEPLOYMENT.md
```

## Local Development

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file (optional, defaults work for local dev):
```bash
cp .env.example .env
```

5. Start the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`.
API documentation is at `http://localhost:8000/docs`.

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`.
The Vite dev server proxies `/api` requests to the backend.

## Deployment to Render

### Option 1: Using render.yaml (Blueprint)

1. Push your code to a GitHub repository.

2. Go to [Render Dashboard](https://dashboard.render.com/).

3. Click "New" → "Blueprint".

4. Connect your GitHub repository.

5. Render will detect the `render.yaml` file and create both services.

6. Update the environment variables:
   - For `nba-analytics-api`: Update `ALLOWED_ORIGINS` with your actual frontend URL.
   - For `nba-analytics-frontend`: Update `VITE_API_URL` with your actual API URL.

### Option 2: Manual Deployment

#### Deploy Backend (FastAPI)

1. Go to Render Dashboard → "New" → "Web Service".

2. Connect your GitHub repository.

3. Configure the service:
   - **Name**: `nba-analytics-api`
   - **Runtime**: Python
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`

4. Add environment variables:
   - `ALLOWED_ORIGINS`: Your frontend URL (e.g., `https://your-frontend.onrender.com`)

5. Click "Create Web Service".

#### Deploy Frontend (React)

1. Go to Render Dashboard → "New" → "Static Site".

2. Connect your GitHub repository.

3. Configure the service:
   - **Name**: `nba-analytics-frontend`
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/dist`

4. Add environment variables:
   - `VITE_API_URL`: Your backend URL (e.g., `https://your-api.onrender.com`)

5. Add a rewrite rule:
   - Source: `/*`
   - Destination: `/index.html`

6. Click "Create Static Site".

## Verification Steps

1. **Check Backend Health**:
   ```bash
   curl https://your-api.onrender.com/health
   ```
   Expected response: `{"status":"healthy"}`

2. **Check API Endpoints**:
   ```bash
   curl https://your-api.onrender.com/api/seasons
   ```
   Expected response: List of available seasons.

3. **Check Frontend**:
   - Open your frontend URL in a browser.
   - Navigate through all pages.
   - Verify data loads correctly.

## Common Issues and Debugging

### Backend Issues

1. **Cold Start Delays**:
   - Free tier services sleep after 15 minutes of inactivity.
   - First request after sleep may take 30-60 seconds.
   - This is normal for free tier.

2. **CORS Errors**:
   - Verify `ALLOWED_ORIGINS` environment variable includes your frontend URL.
   - Check browser console for specific CORS error messages.

3. **Memory Errors**:
   - Free tier has 512MB memory limit.
   - Cache is designed to stay within limits.
   - If issues persist, reduce `CACHE_MAX_SIZE` in `config.py`.

4. **Data Loading Failures**:
   - Check GitHub rate limits (60 requests/hour for unauthenticated).
   - Caching reduces API calls significantly.
   - Wait and retry if rate limited.

### Frontend Issues

1. **API Connection Errors**:
   - Verify `VITE_API_URL` is set correctly.
   - Check that backend is running and accessible.
   - Ensure CORS is configured correctly on backend.

2. **Routing Issues (404 on refresh)**:
   - Ensure rewrite rule is configured: `/* → /index.html`.
   - This is required for client-side routing with React Router.

3. **Build Failures**:
   - Check Node.js version compatibility.
   - Run `npm install` and `npm run build` locally to debug.

### Debugging Commands

Check backend logs:
```bash
# In Render dashboard, go to your service → Logs
```

Test API locally:
```bash
curl http://localhost:8000/api/seasons
curl "http://localhost:8000/api/games?season=2024-25"
curl "http://localhost:8000/api/decomposition?season=2024-25&game_id=YOUR_GAME_ID&model_id=2021-2025&factor_type=four_factors"
```

Test frontend build locally:
```bash
cd frontend
npm run build
npm run preview
```

## Environment Variables Reference

### Backend

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost:5173,http://localhost:3000` |

### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | Empty (uses relative URLs) |

## Performance Notes

- Backend uses in-memory TTL cache (1 hour) to reduce GitHub API calls.
- Cache rebuilds on service restart/redeploy.
- First load after cold start fetches data from GitHub.
- Subsequent requests use cached data.
- Frontend uses standard React patterns (no external state management).
- Charts render client-side using Recharts.
