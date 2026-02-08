import pandas as pd
import httpx
import json
from typing import Optional, List, Dict
from config import DATA_BASE_URL, INTERPRETATIONS_BASE_URL, get_available_seasons
from services.cache import get_cache_key, get_cached, set_cached

STAT_COLUMNS = [
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "tov", "pts", "plus_minus"
]

async def fetch_csv(url: str) -> Optional[pd.DataFrame]:
    cache_key = get_cache_key("fetch_csv", url)
    cached = get_cached(cache_key)
    if cached is not None:
        return cached
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            set_cached(cache_key, df)
            return df
    except Exception:
        return None

async def fetch_json(url: str) -> Optional[dict]:
    cache_key = get_cache_key("fetch_json", url)
    cached = get_cached(cache_key)
    if cached is not None:
        return cached
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            set_cached(cache_key, data)
            return data
    except Exception:
        return None

async def load_season_data(season: str) -> Optional[pd.DataFrame]:
    url = f"{DATA_BASE_URL}/team_game_logs_{season}.csv"
    return await fetch_csv(url)


async def load_contributions(season: str) -> Optional[dict]:
    """Load pre-calculated contribution JSON for a season from GitHub."""
    url = f"{DATA_BASE_URL}/contributions/contributions_{season}.json"
    return await fetch_json(url)


async def load_advanced_stats(season: str) -> Optional[pd.DataFrame]:
    """Load box score advanced stats for a season (actual possessions, minutes)."""
    url = f"{DATA_BASE_URL}/box_score_advanced_{season}.csv"
    return await fetch_csv(url)

async def load_linescores(season: str) -> Optional[pd.DataFrame]:
    """Load linescore data for a season (quarter-by-quarter scoring)."""
    url = f"{DATA_BASE_URL}/linescores_{season}.csv"
    return await fetch_csv(url)

def _normalize_game_type(game_type: str) -> str:
    """Normalize game_type values to consistent format.

    Older seasons use 'playoff' and 'playin', newer seasons use 'playoffs' and 'play_in'.
    Standardize to the newer format for consistent filtering.
    """
    if not game_type:
        return "regular_season"
    gt = str(game_type).strip().lower()
    # Normalize variations
    if gt == "playoff":
        return "playoffs"
    if gt == "playin":
        return "play_in"
    return gt


def _normalize_game_id(game_id) -> str:
    """Normalize game_id to a 10-digit string for reliable cross-file joins."""
    if pd.isna(game_id):
        return ""

    gid = str(game_id).strip()
    # CSV numeric parsing can coerce IDs to float-looking strings (e.g., "22400061.0")
    if gid.endswith(".0"):
        gid = gid[:-2]

    digits = "".join(ch for ch in gid if ch.isdigit())
    if not digits:
        return gid
    return digits.zfill(10)


def normalize_game_logs(df: pd.DataFrame, season: str) -> pd.DataFrame:
    cache_key = get_cache_key("normalize_game_logs", season)
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    rows = []

    for _, row in df.iterrows():
        game_id = _normalize_game_id(row.get("game_id"))
        game_date = row.get("game_date")
        game_type = _normalize_game_type(row.get("game_type", "regular_season"))

        home_team = row.get("team_abbreviation_home")
        road_team = row.get("team_abbreviation_road")

        home_row = {
            "game_id": game_id,
            "game_date": game_date,
            "season": season,
            "game_type": game_type,
            "team": home_team,
            "opponent": road_team,
            "home_away": "home",
            "wl": row.get("wl_home"),
        }

        road_row = {
            "game_id": game_id,
            "game_date": game_date,
            "season": season,
            "game_type": game_type,
            "team": road_team,
            "opponent": home_team,
            "home_away": "road",
            "wl": row.get("wl_road"),
        }

        for stat in STAT_COLUMNS:
            home_col = f"{stat}_home"
            road_col = f"{stat}_road"

            home_row[stat] = row.get(home_col, 0)
            home_row[f"opp_{stat}"] = row.get(road_col, 0)

            road_row[stat] = row.get(road_col, 0)
            road_row[f"opp_{stat}"] = row.get(home_col, 0)

        rows.append(home_row)
        rows.append(road_row)

    normalized_df = pd.DataFrame(rows)

    if "game_date" in normalized_df.columns:
        normalized_df["game_date"] = pd.to_datetime(normalized_df["game_date"])

    set_cached(cache_key, normalized_df)
    return normalized_df

async def get_normalized_season_data(season: str) -> Optional[pd.DataFrame]:
    cache_key = get_cache_key("get_normalized_season_data", season)
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    raw_df = await load_season_data(season)
    if raw_df is None:
        return None

    normalized = normalize_game_logs(raw_df, season)
    set_cached(cache_key, normalized)
    return normalized

async def get_normalized_data_with_possessions(season: str) -> Optional[pd.DataFrame]:
    """Get normalized season data with actual possessions merged from advanced stats."""
    cache_key = get_cache_key("get_normalized_data_with_possessions", season)
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    df = await get_normalized_season_data(season)
    if df is None:
        return None

    adv_df = await load_advanced_stats(season)
    if adv_df is None:
        # Fall back to data without actual possessions
        set_cached(cache_key, df)
        return df

    df = df.copy()
    adv_df = adv_df.copy()

    # Ensure consistent game_id types
    df["game_id"] = df["game_id"].apply(_normalize_game_id)
    adv_df["game_id"] = adv_df["game_id"].apply(_normalize_game_id)

    # Create a lookup dict for possessions by game_id
    poss_lookup = {}
    for _, row in adv_df.iterrows():
        gid = row["game_id"]
        poss_home = row.get("possessions_home")
        poss_road = row.get("possessions_road")
        if pd.notna(poss_home):
            poss_home = float(poss_home)
        else:
            poss_home = None
        if pd.notna(poss_road):
            poss_road = float(poss_road)
        else:
            poss_road = None
        poss_lookup[gid] = {"home": poss_home, "road": poss_road}

    # Add actual_poss and opp_actual_poss columns
    actual_poss = []
    opp_actual_poss = []

    for _, row in df.iterrows():
        gid = _normalize_game_id(row["game_id"])
        home_away = row["home_away"]
        poss_data = poss_lookup.get(gid, {})

        if home_away == "home":
            actual_poss.append(poss_data.get("home"))
            opp_actual_poss.append(poss_data.get("road"))
        else:
            actual_poss.append(poss_data.get("road"))
            opp_actual_poss.append(poss_data.get("home"))

    df["actual_poss"] = actual_poss
    df["opp_actual_poss"] = opp_actual_poss

    set_cached(cache_key, df)
    return df

async def get_games_list(season: str) -> list:
    df = await get_normalized_season_data(season)
    if df is None:
        return []

    games_df = df[df["home_away"] == "home"].copy()
    # Filter out rows with missing team data
    games_df = games_df.dropna(subset=["team", "opponent"])
    games_df = games_df.sort_values("game_date", ascending=False)

    games = []
    for _, row in games_df.iterrows():
        date_str = row["game_date"].strftime("%Y-%m-%d") if pd.notna(row["game_date"]) else ""
        games.append({
            "game_id": str(row["game_id"]).zfill(10),
            "date": date_str,
            "home_team": str(row["team"]),
            "road_team": str(row["opponent"]),
            "home_pts": int(row["pts"]) if pd.notna(row["pts"]) else 0,
            "road_pts": int(row["opp_pts"]) if pd.notna(row["opp_pts"]) else 0,
            "label": f"{date_str}: {row['opponent']} @ {row['team']}"
        })

    return games

async def get_teams_list(season: str) -> list:
    df = await get_normalized_season_data(season)
    if df is None:
        return []

    teams = sorted(df["team"].dropna().unique().tolist())
    return teams

async def get_game_data(season: str, game_id: str) -> Optional[dict]:
    df = await get_normalized_season_data(season)
    if df is None:
        return None

    # ensure consistent types
    df = df.copy()
    df["game_id"] = df["game_id"].apply(_normalize_game_id)
    game_id = _normalize_game_id(game_id)

    game_df = df[df["game_id"] == game_id]
    if len(game_df) != 2:
        return None

    home_row = game_df[game_df["home_away"] == "home"].iloc[0].to_dict()
    road_row = game_df[game_df["home_away"] == "road"].iloc[0].to_dict()

    # Load advanced stats for actual possessions and minutes
    actual_poss_home = None
    actual_poss_road = None
    actual_minutes_home = None
    actual_minutes_road = None

    adv_df = await load_advanced_stats(season)
    if adv_df is not None:
        adv_df = adv_df.copy()
        adv_df["game_id"] = adv_df["game_id"].apply(_normalize_game_id)
        adv_game = adv_df[adv_df["game_id"] == game_id]
        if len(adv_game) == 1:
            adv_row = adv_game.iloc[0]
            actual_poss_home = adv_row.get("possessions_home")
            actual_poss_road = adv_row.get("possessions_road")
            actual_minutes_home = adv_row.get("minutes_home")
            actual_minutes_road = adv_row.get("minutes_road")
            # Convert to float/int if not null
            if pd.notna(actual_poss_home):
                actual_poss_home = float(actual_poss_home)
            else:
                actual_poss_home = None
            if pd.notna(actual_poss_road):
                actual_poss_road = float(actual_poss_road)
            else:
                actual_poss_road = None
            if pd.notna(actual_minutes_home):
                actual_minutes_home = int(actual_minutes_home)
            else:
                actual_minutes_home = None
            if pd.notna(actual_minutes_road):
                actual_minutes_road = int(actual_minutes_road)
            else:
                actual_minutes_road = None

    # Load linescore data
    linescore = None
    ls_df = await load_linescores(season)
    if ls_df is not None:
        ls_df = ls_df.copy()
        ls_df["game_id"] = ls_df["game_id"].apply(_normalize_game_id)
        ls_game = ls_df[ls_df["game_id"] == game_id]
        if len(ls_game) == 1:
            ls_row = ls_game.iloc[0]
            linescore = {
                "home": {
                    "q1": int(ls_row.get("pts_qtr1_home", 0) or 0),
                    "q2": int(ls_row.get("pts_qtr2_home", 0) or 0),
                    "q3": int(ls_row.get("pts_qtr3_home", 0) or 0),
                    "q4": int(ls_row.get("pts_qtr4_home", 0) or 0),
                    "ot": int(ls_row.get("pts_ot_total_home", 0) or 0),
                },
                "road": {
                    "q1": int(ls_row.get("pts_qtr1_road", 0) or 0),
                    "q2": int(ls_row.get("pts_qtr2_road", 0) or 0),
                    "q3": int(ls_row.get("pts_qtr3_road", 0) or 0),
                    "q4": int(ls_row.get("pts_qtr4_road", 0) or 0),
                    "ot": int(ls_row.get("pts_ot_total_road", 0) or 0),
                },
            }

    return {
        "game_id": game_id,
        "game_date": home_row["game_date"].strftime("%Y-%m-%d") if pd.notna(home_row["game_date"]) else "",
        "home_team": home_row["team"],
        "road_team": road_row["team"],
        "home": home_row,
        "road": road_row,
        "actual_possessions_home": actual_poss_home,
        "actual_possessions_road": actual_poss_road,
        "actual_minutes_home": actual_minutes_home,
        "actual_minutes_road": actual_minutes_road,
        "linescore": linescore,
    }


async def get_interpretations(season: str) -> Optional[Dict]:
    """Fetch pre-generated interpretations for a season from GitHub.

    Returns dict with structure:
    {
        "season": "2024-25",
        "prompt_version": "v2_bullets",
        "interpretations": {
            "0022400123": {
                "generated_at": "2025-01-15T...",
                "model": "claude-sonnet-4-...",
                "eight_factors": "- Bullet 1...\n- Bullet 2...",
                "four_factors": "- ..."
            },
            ...
        }
    }
    """
    url = f"{INTERPRETATIONS_BASE_URL}/gamesummaries_{season}_2018-25.json"
    return await fetch_json(url)


async def get_game_interpretation(
    season: str, game_id: str, factor_type: str, model_id: str = None
) -> Optional[Dict]:
    """Get pre-generated interpretation for a specific game.

    Args:
        season: Season string (e.g., "2024-25")
        game_id: Game ID string
        factor_type: "four_factors" or "eight_factors"
        model_id: Decomposition model ID (e.g., "2018-2025"). If provided,
                  only returns pre-generated interpretation if it was generated
                  using the same model. This ensures interpretation matches
                  the contributions shown on the page.

    Returns:
        Dict with 'text' and 'model' keys if found, None otherwise
    """
    interp_data = await get_interpretations(season)
    if interp_data is None:
        return None

    # Check if the requested model matches what was used for pre-generation
    if model_id:
        stored_model_id = interp_data.get("decomposition_model_id")
        if stored_model_id and stored_model_id != model_id:
            # Model mismatch - don't use pre-generated interpretation
            return None

    interpretations = interp_data.get("interpretations", {})
    game_interp = interpretations.get(str(game_id)) or interpretations.get(_normalize_game_id(game_id))
    if game_interp is None:
        return None

    text = game_interp.get(factor_type)
    if text is None:
        return None

    return {
        "text": text,
        "model": game_interp.get("model", "unknown"),
    }
