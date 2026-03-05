import os
from datetime import datetime
from pathlib import Path

GITHUB_USER = "millxing"
GITHUB_BRANCH = "main"
DATA_REPO = "NBA_Data"

# GitHub token for API requests (increases rate limit from 60 to 5000 requests/hour)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Admin secret key for protected endpoints (e.g., cache clearing)
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY")

DATA_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{DATA_REPO}/{GITHUB_BRANCH}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NBA_DATA_REPO_DIR = Path("/Users/robschoen/Dropbox/CC/NBA_Data").resolve()
NBA_DATA_REPO_DIR = Path(
    os.getenv("NBA_DATA_REPO_DIR", str(DEFAULT_NBA_DATA_REPO_DIR))
).expanduser().resolve()
PBP_ROOT_DIR = Path(
    os.getenv("PBP_ROOT_DIR", str(NBA_DATA_REPO_DIR / "PBPdata"))
).expanduser().resolve()
PBP_GAME_STATES_ROOT = Path(
    os.getenv("PBP_GAME_STATES_ROOT", str(PBP_ROOT_DIR / "game_states"))
).expanduser().resolve()
PBP_WINPROB_BASE_ROOT = Path(
    os.getenv("PBP_WINPROB_BASE_ROOT", str(PBP_ROOT_DIR / "winprob_base"))
).expanduser().resolve()
PBP_WINPROB_MODELS_ROOT = Path(
    os.getenv("PBP_WINPROB_MODELS_ROOT", str(PBP_ROOT_DIR / "winprob_models"))
).expanduser().resolve()
PBP_GITHUB_RAW_BASE_URL = os.getenv(
    "PBP_GITHUB_RAW_BASE_URL", f"{DATA_BASE_URL}/PBPdata"
).rstrip("/")
PBP_ENABLE_LEGACY_GLA_FALLBACK = os.getenv(
    "PBP_ENABLE_LEGACY_GLA_FALLBACK", "1"
).strip().lower() in {"1", "true", "yes", "on"}
PBP_LEGACY_GLA_ROOT = Path(
    os.getenv("PBP_LEGACY_GLA_ROOT", str(PROJECT_ROOT / "data" / "pbp" / "processed"))
).expanduser().resolve()
PBP_REMOTE_CACHE_DIR = Path(
    os.getenv("PBP_REMOTE_CACHE_DIR", "/tmp/gla_pbp_cache")
).expanduser().resolve()

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

# LLM configuration for interpretation generation
LLM_MODELS = {
    "historical": "gpt-4o-mini",           # Legacy/historical generation
    "current": "gpt-5.4",                  # Current-season interpretation model (new games)
    "fallback": "gpt-5-mini",              # Real-time fallback model
}

# URL for pre-generated interpretations
INTERPRETATIONS_BASE_URL = f"{DATA_BASE_URL}/interpretations"
