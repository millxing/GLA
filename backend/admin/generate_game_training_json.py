#!/usr/bin/env python3
"""
Generate JSON training data for LLM game analysis.

This script creates a structured JSON file containing game data with:
- Team ratings (ORtg, DRtg, Net Rating)
- 8 factors (eFG%, BH%, OREB%, FT Rate for both teams)
- Quintile classifications based on 2018-25 training period
- Contributions from the 2018-2025 game model
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AVAILABLE_MODELS, get_current_season, get_available_seasons
from services.data_loader import (
    load_model,
    get_normalized_data_with_possessions,
    get_game_data,
    get_games_list,
)
from services.calculations import (
    compute_four_factors,
    compute_game_ratings,
    compute_decomposition,
    compute_factor_differentials,
)


# Quintile labels
QUINTILE_LABELS = {
    1: "POOR",       # 0-20%
    2: "SUBPAR",     # 20-40%
    3: "AVERAGE",    # 40-60%
    4: "GOOD",       # 60-80%
    5: "EXCELLENT",  # 80-100%
}

# For net rating, we need to handle negative values differently
# (more negative = worse, more positive = better)
NET_RATING_QUINTILE_LABELS = {
    1: "POOR",       # bottom 20%
    2: "SUBPAR",     # 20-40%
    3: "AVERAGE",    # 40-60%
    4: "GOOD",       # 60-80%
    5: "EXCELLENT",  # top 20%
}


async def compute_quintile_thresholds(seasons: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute quintile thresholds for each metric from historical game data.

    Args:
        seasons: List of seasons to include (e.g., ["2018-19", "2019-20", ...])

    Returns:
        Dict mapping metric names to their quintile boundaries (p20, p40, p60, p80)
    """
    print(f"Computing quintile thresholds from {len(seasons)} seasons...")

    # Collect all game-level values for each metric
    all_values = {
        "off_rating": [],
        "def_rating": [],
        "net_rating": [],
        "efg": [],
        "ball_handling": [],
        "oreb": [],
        "ft_rate": [],
    }

    for season in seasons:
        print(f"  Loading {season}...")
        df = await get_normalized_data_with_possessions(season)
        if df is None:
            print(f"  Warning: Could not load {season}")
            continue

        # Filter to exclude non-standard games
        df = df[~df["game_type"].isin(["nba_cup_final"])]

        # For each game row, compute the metrics
        for _, row in df.iterrows():
            team_row = {
                "fgm": row.get("fgm", 0) or 0,
                "fga": row.get("fga", 0) or 0,
                "fg3m": row.get("fg3m", 0) or 0,
                "fg3a": row.get("fg3a", 0) or 0,
                "ftm": row.get("ftm", 0) or 0,
                "fta": row.get("fta", 0) or 0,
                "oreb": row.get("oreb", 0) or 0,
                "dreb": row.get("dreb", 0) or 0,
                "tov": row.get("tov", 0) or 0,
                "pts": row.get("pts", 0) or 0,
            }

            opp_row = {
                "fgm": row.get("opp_fgm", 0) or 0,
                "fga": row.get("opp_fga", 0) or 0,
                "fg3m": row.get("opp_fg3m", 0) or 0,
                "fg3a": row.get("opp_fg3a", 0) or 0,
                "ftm": row.get("opp_ftm", 0) or 0,
                "fta": row.get("opp_fta", 0) or 0,
                "oreb": row.get("opp_oreb", 0) or 0,
                "dreb": row.get("opp_dreb", 0) or 0,
                "tov": row.get("opp_tov", 0) or 0,
                "pts": row.get("opp_pts", 0) or 0,
            }

            # Compute four factors and ratings using actual possessions
            actual_poss = row.get("actual_poss")
            opp_actual_poss = row.get("opp_actual_poss")
            factors = compute_four_factors(team_row, opp_row, possessions=actual_poss)

            ratings = compute_game_ratings(
                team_row, opp_row,
                actual_possessions=actual_poss,
                opp_actual_possessions=opp_actual_poss,
            )

            # Collect values
            all_values["off_rating"].append(ratings["offensive_rating"])
            all_values["def_rating"].append(ratings["defensive_rating"])
            all_values["net_rating"].append(ratings["net_rating"])
            all_values["efg"].append(factors["efg"])
            all_values["ball_handling"].append(factors["ball_handling"])
            all_values["oreb"].append(factors["oreb"])
            all_values["ft_rate"].append(factors["ft_rate"])

    # Compute quintile boundaries for each metric
    thresholds = {}
    for metric, values in all_values.items():
        if not values:
            continue
        arr = np.array(values)
        thresholds[metric] = {
            "p20": float(np.percentile(arr, 20)),
            "p40": float(np.percentile(arr, 40)),
            "p60": float(np.percentile(arr, 60)),
            "p80": float(np.percentile(arr, 80)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(values),
        }
        print(f"  {metric}: p20={thresholds[metric]['p20']:.1f}, p40={thresholds[metric]['p40']:.1f}, "
              f"p60={thresholds[metric]['p60']:.1f}, p80={thresholds[metric]['p80']:.1f}")

    return thresholds


def classify_value(value: float, thresholds: Dict[str, float], higher_is_better: bool = True) -> str:
    """
    Classify a value into a quintile label based on thresholds.

    Args:
        value: The metric value to classify
        thresholds: Dict with p20, p40, p60, p80 boundaries
        higher_is_better: If True, higher values get better labels (EXCELLENT)
                         If False, lower values get better labels (for defensive metrics)

    Returns:
        Quintile label (POOR, SUBPAR, AVERAGE, GOOD, EXCELLENT)
    """
    p20, p40, p60, p80 = thresholds["p20"], thresholds["p40"], thresholds["p60"], thresholds["p80"]

    if higher_is_better:
        if value <= p20:
            return "POOR"
        elif value <= p40:
            return "SUBPAR"
        elif value <= p60:
            return "AVERAGE"
        elif value <= p80:
            return "GOOD"
        else:
            return "EXCELLENT"
    else:
        # For metrics where lower is better (like defensive rating)
        if value >= p80:
            return "POOR"
        elif value >= p60:
            return "SUBPAR"
        elif value >= p40:
            return "AVERAGE"
        elif value >= p20:
            return "GOOD"
        else:
            return "EXCELLENT"


async def process_game(
    season: str,
    game_id: str,
    model: Dict,
    thresholds: Dict[str, Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    """
    Process a single game and return structured data.
    """
    game_data = await get_game_data(season, game_id)
    if game_data is None:
        return None

    home_row = game_data["home"]
    road_row = game_data["road"]

    # Build team stat dicts
    home_team_row = {
        "fgm": home_row.get("fgm", 0) or 0,
        "fga": home_row.get("fga", 0) or 0,
        "fg3m": home_row.get("fg3m", 0) or 0,
        "fg3a": home_row.get("fg3a", 0) or 0,
        "ftm": home_row.get("ftm", 0) or 0,
        "fta": home_row.get("fta", 0) or 0,
        "oreb": home_row.get("oreb", 0) or 0,
        "dreb": home_row.get("dreb", 0) or 0,
        "tov": home_row.get("tov", 0) or 0,
        "pts": home_row.get("pts", 0) or 0,
    }

    road_team_row = {
        "fgm": road_row.get("fgm", 0) or 0,
        "fga": road_row.get("fga", 0) or 0,
        "fg3m": road_row.get("fg3m", 0) or 0,
        "fg3a": road_row.get("fg3a", 0) or 0,
        "ftm": road_row.get("ftm", 0) or 0,
        "fta": road_row.get("fta", 0) or 0,
        "oreb": road_row.get("oreb", 0) or 0,
        "dreb": road_row.get("dreb", 0) or 0,
        "tov": road_row.get("tov", 0) or 0,
        "pts": road_row.get("pts", 0) or 0,
    }

    # Compute four factors and ratings using actual possessions
    actual_poss_home = game_data.get("actual_possessions_home")
    actual_poss_road = game_data.get("actual_possessions_road")
    actual_mins_home = game_data.get("actual_minutes_home")

    home_factors = compute_four_factors(home_team_row, road_team_row, possessions=actual_poss_home)
    road_factors = compute_four_factors(road_team_row, home_team_row, possessions=actual_poss_road)

    home_ratings = compute_game_ratings(
        home_team_row, road_team_row,
        actual_possessions=actual_poss_home,
        opp_actual_possessions=actual_poss_road,
        actual_minutes=actual_mins_home,
    )
    road_ratings = compute_game_ratings(
        road_team_row, home_team_row,
        actual_possessions=actual_poss_road,
        opp_actual_possessions=actual_poss_home,
    )

    # Compute decomposition using eight_factors mode
    differentials = compute_factor_differentials(home_factors, road_factors)
    decomposition = compute_decomposition(
        differentials=differentials,
        model=model,
        factor_type="eight_factors",
        home_factors=home_factors,
        away_factors=road_factors,
    )

    contributions = decomposition.get("contributions", {})

    # Build the output structure
    home_team = game_data["home_team"]
    road_team = game_data["road_team"]
    home_pts = int(home_row.get("pts", 0) or 0)
    road_pts = int(road_row.get("pts", 0) or 0)

    # Calculate net rating from home team's perspective
    # Home team's net rating is (home_off - home_def), which equals margin from home's view
    home_net_rating = home_ratings["net_rating"]
    road_net_rating = road_ratings["net_rating"]

    result = {
        "game_id": game_id,
        "game_date": game_data["game_date"],
        "matchup": f"{road_team}@{home_team}",
        "score": f"{road_pts}-{home_pts}",
        "home_team": home_team,
        "road_team": road_team,
        "home_pts": home_pts,
        "road_pts": road_pts,
        "winner": home_team if home_pts > road_pts else road_team,
        "margin": abs(home_pts - road_pts),
        "model": "2018-2025",

        # Home team ratings
        "home_off_rating": round(home_ratings["offensive_rating"], 1),
        "home_off_rating_class": classify_value(home_ratings["offensive_rating"], thresholds["off_rating"]),
        "home_def_rating": round(home_ratings["defensive_rating"], 1),
        "home_def_rating_class": classify_value(home_ratings["defensive_rating"], thresholds["def_rating"], higher_is_better=False),
        "home_net_rating": round(home_net_rating, 1),
        "home_net_rating_class": classify_value(home_net_rating, thresholds["net_rating"]),

        # Road team ratings
        "road_off_rating": round(road_ratings["offensive_rating"], 1),
        "road_off_rating_class": classify_value(road_ratings["offensive_rating"], thresholds["off_rating"]),
        "road_def_rating": round(road_ratings["defensive_rating"], 1),
        "road_def_rating_class": classify_value(road_ratings["defensive_rating"], thresholds["def_rating"], higher_is_better=False),
        "road_net_rating": round(road_net_rating, 1),
        "road_net_rating_class": classify_value(road_net_rating, thresholds["net_rating"]),

        # Home team factors
        "home_efg": round(home_factors["efg"], 1),
        "home_efg_class": classify_value(home_factors["efg"], thresholds["efg"]),
        "home_efg_contrib": contributions.get("home_shooting", 0),

        "home_ball_handling": round(home_factors["ball_handling"], 1),
        "home_ball_handling_class": classify_value(home_factors["ball_handling"], thresholds["ball_handling"]),
        "home_ball_handling_contrib": contributions.get("home_ball_handling", 0),

        "home_oreb": round(home_factors["oreb"], 1),
        "home_oreb_class": classify_value(home_factors["oreb"], thresholds["oreb"]),
        "home_oreb_contrib": contributions.get("home_orebounding", 0),

        "home_ft_rate": round(home_factors["ft_rate"], 1),
        "home_ft_rate_class": classify_value(home_factors["ft_rate"], thresholds["ft_rate"]),
        "home_ft_rate_contrib": contributions.get("home_free_throws", 0),

        # Road team factors
        "road_efg": round(road_factors["efg"], 1),
        "road_efg_class": classify_value(road_factors["efg"], thresholds["efg"]),
        "road_efg_contrib": contributions.get("road_shooting", 0),

        "road_ball_handling": round(road_factors["ball_handling"], 1),
        "road_ball_handling_class": classify_value(road_factors["ball_handling"], thresholds["ball_handling"]),
        "road_ball_handling_contrib": contributions.get("road_ball_handling", 0),

        "road_oreb": round(road_factors["oreb"], 1),
        "road_oreb_class": classify_value(road_factors["oreb"], thresholds["oreb"]),
        "road_oreb_contrib": contributions.get("road_orebounding", 0),

        "road_ft_rate": round(road_factors["ft_rate"], 1),
        "road_ft_rate_class": classify_value(road_factors["ft_rate"], thresholds["ft_rate"]),
        "road_ft_rate_contrib": contributions.get("road_free_throws", 0),

        # Model intercept
        "intercept": decomposition.get("intercept", 0),
        "predicted_margin": decomposition.get("predicted_rating_diff", 0),
    }

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate JSON training data for game analysis LLM")
    parser.add_argument("--num-games", type=int, default=50, help="Number of recent games to process")
    parser.add_argument("--output", type=str, default="game_training_data.json", help="Output JSON file")
    parser.add_argument("--season", type=str, default=None, help="Season to process (default: current)")
    args = parser.parse_args()

    # Determine season
    season = args.season or get_current_season()
    print(f"Processing season: {season}")

    # Load the 2018-2025 game model
    model_config = next((m for m in AVAILABLE_MODELS if m["id"] == "2018-2025"), None)
    if model_config is None:
        print("Error: 2018-2025 model not found")
        return

    print(f"Loading model: {model_config['id']}")
    model = await load_model(model_config["file"])
    if model is None:
        print("Error: Failed to load model")
        return

    # Compute quintile thresholds from training seasons (2018-19 to 2024-25)
    training_seasons = [
        "2018-19", "2019-20", "2020-21", "2021-22",
        "2022-23", "2023-24", "2024-25"
    ]
    thresholds = await compute_quintile_thresholds(training_seasons)

    # Get games list for the season
    print(f"\nFetching games for {season}...")
    games = await get_games_list(season)
    if not games:
        print("Error: No games found")
        return

    # Take the most recent N games
    recent_games = games[:args.num_games]
    print(f"Processing {len(recent_games)} most recent games...")

    # Process each game
    results = []
    for i, game in enumerate(recent_games):
        game_id = game["game_id"]
        print(f"  [{i+1}/{len(recent_games)}] Processing {game['label']}...")

        game_result = await process_game(season, game_id, model, thresholds)
        if game_result:
            results.append(game_result)

    # Build output structure
    output = {
        "generated_at": datetime.now().isoformat(),
        "season": season,
        "model_id": "2018-2025",
        "training_seasons": training_seasons,
        "quintile_thresholds": thresholds,
        "games": results,
    }

    # Write to file
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nGenerated {len(results)} game records")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
