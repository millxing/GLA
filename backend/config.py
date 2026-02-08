import os
from datetime import datetime

GITHUB_USER = "millxing"
GITHUB_BRANCH = "main"
DATA_REPO = "NBA_Data"
MODEL_REPO = "NBA_Data"

# GitHub token for API requests (increases rate limit from 60 to 5000 requests/hour)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Admin secret key for protected endpoints (e.g., cache clearing)
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY")

DATA_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{DATA_REPO}/{GITHUB_BRANCH}"
MODEL_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{MODEL_REPO}/{GITHUB_BRANCH}"

CACHE_TTL_SECONDS = 1800  # 30 minutes
CACHE_MAX_SIZE = 50

SEASON_START_YEAR = 2000

def get_current_season() -> str:
    now = datetime.now()
    year = now.year
    month = now.month
    if month >= 10:
        start_year = year
    else:
        start_year = year - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"

def get_available_seasons() -> list:
    current = get_current_season()
    current_start = int(current.split("-")[0])
    seasons = []
    for start_year in range(SEASON_START_YEAR, current_start + 1):
        end_year = start_year + 1
        season_str = f"{start_year}-{str(end_year)[-2:]}"
        seasons.append(season_str)
    return seasons

AVAILABLE_MODELS = [
    {"id": "2023-2025", "name": "2023-2025 Model", "file": "models/2023-2025.json"},
    {"id": "2021-2025", "name": "2021-2025 Model", "file": "models/2021-2025.json"},
    {"id": "2020-2025", "name": "2020-2025 Model", "file": "models/2020-2025.json"},
    {"id": "2019-2025", "name": "2019-2025 Model", "file": "models/2019-2025.json"},
    {"id": "2018-2025", "name": "2018-2025 Model", "file": "models/2018-2025.json"},
    {"id": "2017-2025", "name": "2017-2025 Model", "file": "models/2017-2025.json"},
]

# Fallback list of known season-level models (used if GitHub API discovery fails)
# These are discovered dynamically when possible, see services/data_loader.py:discover_season_models()
KNOWN_SEASON_MODELS = [
    {"id": "season_2018-2025", "name": "2018-2025 Model", "file": "models/season_2018-2025.json"},
]

# LLM configuration for interpretation generation
LLM_MODELS = {
    "historical": "gpt-4o-mini",           # Cheap model for historical seasons
    "current": "claude-sonnet-4-20250514", # Better model for current season
    "fallback": "gpt-4o-mini",             # Fallback for real-time generation
}

# URL for pre-generated interpretations
INTERPRETATIONS_BASE_URL = f"{DATA_BASE_URL}/interpretations"
