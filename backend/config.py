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

DATA_FAMILY_DIRECTORIES = {
    "team_game_logs": "team_game_logs",
    "linescores": "linescores",
    "box_score_advanced": "box_score_advanced",
    "box_score_traditional": "box_score_traditional",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Default: ../NBA_Data relative to project root (sibling directory).
# Override with NBA_DATA_REPO_DIR env var for other machines/deployments.
DEFAULT_NBA_DATA_REPO_DIR = (PROJECT_ROOT / ".." / "NBA_Data").resolve()
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

DEFAULT_CACHE_MAX_SIZE = "12" if os.getenv("RENDER") else "50"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "1800") or "1800")
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", DEFAULT_CACHE_MAX_SIZE) or DEFAULT_CACHE_MAX_SIZE)

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


def detect_data_family(filename: str | Path) -> str:
    basename = Path(filename).name
    for family in DATA_FAMILY_DIRECTORIES:
        if basename.startswith(f"{family}_"):
            return family
    raise ValueError(f"Unsupported NBA_Data family for filename: {basename}")


def build_data_filename(family: str, season: str, data_scope: str = "all") -> str:
    if family not in DATA_FAMILY_DIRECTORIES:
        raise ValueError(f"Unsupported NBA_Data family: {family}")

    scope = str(data_scope or "all").strip()
    if family in {"team_game_logs", "box_score_advanced"}:
        if scope == "all":
            return f"{family}_{season}.csv"
        return f"{family}_{scope}_{season}.csv"

    if scope != "all":
        raise ValueError(f"{family} does not support data_scope={data_scope!r}")
    return f"{family}_{season}.csv"


def build_box_score_traditional_filename(kind: str, season: str) -> str:
    kind_norm = str(kind or "").strip().lower()
    valid_kinds = {"players", "teams", "starter_bench"}
    if kind_norm not in valid_kinds:
        raise ValueError(f"Unsupported box_score_traditional kind: {kind}")
    return f"box_score_traditional_v3_{kind_norm}_{season}.csv"


def get_legacy_data_relative_path(filename: str | Path) -> Path:
    return Path(Path(filename).name)


def get_canonical_data_relative_path(filename: str | Path) -> Path:
    basename = Path(filename).name
    family = detect_data_family(basename)
    return Path(DATA_FAMILY_DIRECTORIES[family]) / basename


def get_canonical_data_file_path(
    filename: str | Path,
    repo_dir: Path = NBA_DATA_REPO_DIR,
) -> Path:
    return Path(repo_dir) / get_canonical_data_relative_path(filename)


def get_legacy_data_file_path(
    filename: str | Path,
    repo_dir: Path = NBA_DATA_REPO_DIR,
) -> Path:
    return Path(repo_dir) / get_legacy_data_relative_path(filename)


def get_data_file_candidates(
    filename: str | Path,
    repo_dir: Path = NBA_DATA_REPO_DIR,
) -> list[Path]:
    canonical = get_canonical_data_file_path(filename, repo_dir=repo_dir)
    legacy = get_legacy_data_file_path(filename, repo_dir=repo_dir)
    if canonical == legacy:
        return [canonical]
    return [canonical, legacy]


def resolve_data_file_path(
    filename: str | Path,
    repo_dir: Path = NBA_DATA_REPO_DIR,
) -> Path:
    for candidate in get_data_file_candidates(filename, repo_dir=repo_dir):
        if candidate.exists():
            return candidate
    return get_canonical_data_file_path(filename, repo_dir=repo_dir)


def build_data_file_url(filename: str | Path, base_url: str | None = None) -> str:
    rel_path = get_canonical_data_relative_path(filename).as_posix()
    base = (base_url or DATA_BASE_URL).rstrip("/")
    return f"{base}/{rel_path}"


def iter_data_family_files(
    repo_dir: Path,
    family: str,
    pattern: str = "*.csv",
) -> list[Path]:
    if family not in DATA_FAMILY_DIRECTORIES:
        raise ValueError(f"Unsupported NBA_Data family: {family}")

    seen: set[str] = set()
    matches: list[Path] = []
    canonical_dir = Path(repo_dir) / DATA_FAMILY_DIRECTORIES[family]
    if canonical_dir.exists():
        for path in sorted(canonical_dir.glob(pattern)):
            seen.add(path.name)
            matches.append(path)

    for path in sorted(Path(repo_dir).glob(pattern)):
        if path.name in seen:
            continue
        try:
            if detect_data_family(path.name) != family:
                continue
        except ValueError:
            continue
        matches.append(path)

    return matches

# LLM configuration for interpretation generation
LLM_MODELS = {
    "historical": "gpt-4o-mini",           # Legacy/historical generation
    "current": "gpt-5.4",                  # Current-season interpretation model (new games)
    "fallback": "gpt-5-mini",              # Real-time fallback model
}

# URL for pre-generated interpretations
INTERPRETATIONS_BASE_URL = f"{DATA_BASE_URL}/interpretations"
