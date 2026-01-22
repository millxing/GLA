import pandas as pd
import httpx
import json
from typing import Optional
from config import DATA_BASE_URL, MODEL_BASE_URL, get_available_seasons
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

async def load_model(model_file: str) -> Optional[dict]:
    url = f"{MODEL_BASE_URL}/{model_file}"
    return await fetch_json(url)

def normalize_game_logs(df: pd.DataFrame, season: str) -> pd.DataFrame:
    cache_key = get_cache_key("normalize_game_logs", season)
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    rows = []

    for _, row in df.iterrows():
        game_id = row.get("game_id")
        game_date = row.get("game_date")
        game_type = row.get("game_type", "Regular Season")

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

async def get_games_list(season: str) -> list:
    df = await get_normalized_season_data(season)
    if df is None:
        return []

    games_df = df[df["home_away"] == "home"].copy()
    games_df = games_df.sort_values("game_date", ascending=False)

    games = []
    for _, row in games_df.iterrows():
        date_str = row["game_date"].strftime("%Y-%m-%d") if pd.notna(row["game_date"]) else ""
        games.append({
            "game_id": str(row["game_id"]),
            "date": date_str,
            "home_team": row["team"],
            "road_team": row["opponent"],
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

    game_df = df[df["game_id"] == game_id]
    if len(game_df) != 2:
        return None

    home_row = game_df[game_df["home_away"] == "home"].iloc[0].to_dict()
    road_row = game_df[game_df["home_away"] == "road"].iloc[0].to_dict()

    return {
        "game_id": game_id,
        "game_date": home_row["game_date"].strftime("%Y-%m-%d") if pd.notna(home_row["game_date"]) else "",
        "home_team": home_row["team"],
        "road_team": road_row["team"],
        "home": home_row,
        "road": road_row
    }
