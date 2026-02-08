#!/usr/bin/env python3
"""
Generate a JSON training set of LLM prompts for the last N games.

Usage:
    cd backend
    python admin/generate_prompt_training_set.py --season 2025-26 --count 50

Output:
    Creates prompt_training_set.json with prompts for both factor types.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.calculations import compute_four_factors, compute_game_ratings
from services.llm import _build_interpretation_prompt


def _compute_season_league_averages(df: pd.DataFrame) -> dict:
    """Compute league averages for a season."""
    efg_vals = []
    bh_vals = []
    oreb_vals = []
    ft_vals = []

    for _, row in df.iterrows():
        home_stats = {
            "fgm": row.get("fgm_home", 0),
            "fga": row.get("fga_home", 0),
            "fg3m": row.get("fg3m_home", 0),
            "ftm": row.get("ftm_home", 0),
            "oreb": row.get("oreb_home", 0),
            "tov": row.get("tov_home", 0),
        }
        road_stats = {"dreb": row.get("dreb_road", 0)}

        if home_stats["fga"] > 0:
            efg = (home_stats["fgm"] + 0.5 * home_stats["fg3m"]) / home_stats["fga"] * 100
            efg_vals.append(efg)
            ft_rate = home_stats["ftm"] / home_stats["fga"] * 100
            ft_vals.append(ft_rate)

        poss_approx = home_stats["fga"] + 0.44 * row.get("fta_home", 0) + home_stats["tov"] - home_stats["oreb"]
        if poss_approx > 0:
            tov_pct = home_stats["tov"] / poss_approx * 100
            bh_vals.append(100 - tov_pct)

        total_oreb_chances = home_stats["oreb"] + road_stats["dreb"]
        if total_oreb_chances > 0:
            oreb_pct = home_stats["oreb"] / total_oreb_chances * 100
            oreb_vals.append(oreb_pct)

        # Road team
        road_team_stats = {
            "fgm": row.get("fgm_road", 0),
            "fga": row.get("fga_road", 0),
            "fg3m": row.get("fg3m_road", 0),
            "ftm": row.get("ftm_road", 0),
            "oreb": row.get("oreb_road", 0),
            "tov": row.get("tov_road", 0),
        }
        home_dreb = {"dreb": row.get("dreb_home", 0)}

        if road_team_stats["fga"] > 0:
            efg = (road_team_stats["fgm"] + 0.5 * road_team_stats["fg3m"]) / road_team_stats["fga"] * 100
            efg_vals.append(efg)
            ft_rate = road_team_stats["ftm"] / road_team_stats["fga"] * 100
            ft_vals.append(ft_rate)

        poss_approx = road_team_stats["fga"] + 0.44 * row.get("fta_road", 0) + road_team_stats["tov"] - road_team_stats["oreb"]
        if poss_approx > 0:
            tov_pct = road_team_stats["tov"] / poss_approx * 100
            bh_vals.append(100 - tov_pct)

        total_oreb_chances = road_team_stats["oreb"] + home_dreb["dreb"]
        if total_oreb_chances > 0:
            oreb_pct = road_team_stats["oreb"] / total_oreb_chances * 100
            oreb_vals.append(oreb_pct)

    return {
        "efg": sum(efg_vals) / len(efg_vals) if efg_vals else 52.0,
        "ball_handling": sum(bh_vals) / len(bh_vals) if bh_vals else 86.0,
        "oreb": sum(oreb_vals) / len(oreb_vals) if oreb_vals else 25.0,
        "ft_rate": sum(ft_vals) / len(ft_vals) if ft_vals else 20.0,
    }


def _compute_factor_ranges(df: pd.DataFrame) -> dict:
    """Compute Q1/Q3 ranges for factors."""
    return {
        "efg": {"q1": 48, "q3": 58},
        "ball_handling": {"q1": 83, "q3": 90},
        "oreb": {"q1": 18, "q3": 32},
        "ft_rate": {"q1": 14, "q3": 26},
    }


def _build_decomposition_data(game_row: pd.Series, model_data: dict, league_avgs: dict, factor_ranges: dict) -> dict:
    """Build decomposition data structure from a game row."""
    home_team_row = {
        "fgm": game_row.get("fgm_home", 0),
        "fga": game_row.get("fga_home", 0),
        "fg3m": game_row.get("fg3m_home", 0),
        "ftm": game_row.get("ftm_home", 0),
        "fta": game_row.get("fta_home", 0),
        "oreb": game_row.get("oreb_home", 0),
        "dreb": game_row.get("dreb_home", 0),
        "tov": game_row.get("tov_home", 0),
        "pts": game_row.get("pts_home", 0),
    }

    road_team_row = {
        "fgm": game_row.get("fgm_road", 0),
        "fga": game_row.get("fga_road", 0),
        "fg3m": game_row.get("fg3m_road", 0),
        "ftm": game_row.get("ftm_road", 0),
        "fta": game_row.get("fta_road", 0),
        "oreb": game_row.get("oreb_road", 0),
        "dreb": game_row.get("dreb_road", 0),
        "tov": game_row.get("tov_road", 0),
        "pts": game_row.get("pts_road", 0),
    }

    actual_poss_home = game_row.get("possessions_home")
    actual_poss_road = game_row.get("possessions_road")
    actual_mins_home = game_row.get("minutes_home")
    actual_mins_road = game_row.get("minutes_road")

    home_factors = compute_four_factors(home_team_row, road_team_row, possessions=actual_poss_home)
    road_factors = compute_four_factors(road_team_row, home_team_row, possessions=actual_poss_road)

    home_ratings = compute_game_ratings(
        home_team_row, road_team_row,
        actual_possessions=actual_poss_home, opp_actual_possessions=actual_poss_road,
        actual_minutes=actual_mins_home,
    )
    road_ratings = compute_game_ratings(
        road_team_row, home_team_row,
        actual_possessions=actual_poss_road, opp_actual_possessions=actual_poss_home,
        actual_minutes=actual_mins_road,
    )

    return {
        "game_id": str(game_row.get("game_id", "")),
        "game_date": str(game_row.get("game_date", "")),
        "home_team": game_row.get("team_abbreviation_home", ""),
        "road_team": game_row.get("team_abbreviation_road", ""),
        "home_pts": int(game_row.get("pts_home", 0)),
        "road_pts": int(game_row.get("pts_road", 0)),
        "home_factors": home_factors,
        "road_factors": road_factors,
        "home_ratings": home_ratings,
        "road_ratings": road_ratings,
        "league_averages": league_avgs,
        "factor_ranges": factor_ranges,
    }


def _adjust_decomp_for_factor_type(decomp_data: dict, factor_type: str, model_data: dict) -> dict:
    """Add contributions based on factor type.

    Mirrors the logic in services/calculations.py:compute_decomposition().
    Key: factor values are in percentage form (54.0), but model coefficients
    expect decimal form (0.54), so we divide by 100 before multiplying.
    """
    result = decomp_data.copy()

    home_factors = decomp_data["home_factors"]
    road_factors = decomp_data["road_factors"]
    league_avgs = decomp_data["league_averages"]

    # Game-level models only have four_factors - use those coefficients for both modes
    model = model_data.get("four_factors", {})
    coefs = model.get("coefficients", {})
    intercept = model.get("intercept", 0)

    if factor_type == "eight_factors":
        # Eight factors mode: break into home/road contributions centered on league average
        contributions = {}
        total_home = 0.0
        total_road = 0.0

        for factor_key, coef_key in [
            ("efg", "shooting"),
            ("ball_handling", "ball_handling"),
            ("oreb", "orebounding"),
            ("ft_rate", "free_throws"),
        ]:
            home_val = home_factors.get(factor_key, 0)
            road_val = road_factors.get(factor_key, 0)
            avg = league_avgs.get(factor_key, 0)
            coef = coefs.get(coef_key, 0)

            # Divide by 100 to convert from percentage to decimal scale
            home_centered = (home_val - avg) / 100.0
            road_centered = (road_val - avg) / 100.0

            home_contrib = coef * home_centered
            road_contrib = coef * road_centered

            contributions[f"home_{coef_key}"] = round(home_contrib, 2)
            # Negate road contribution for display: positive = helps home team
            contributions[f"road_{coef_key}"] = round(-road_contrib, 2)

            total_home += home_contrib
            total_road += road_contrib

        result["contributions"] = contributions
        result["predicted_rating_diff"] = round(intercept + total_home - total_road, 2)

    else:
        # Four factors mode: use differentials
        contributions = {}
        total_contribution = intercept

        for factor_key, coef_key in [
            ("efg", "shooting"),
            ("ball_handling", "ball_handling"),
            ("oreb", "orebounding"),
            ("ft_rate", "free_throws"),
        ]:
            home_val = home_factors.get(factor_key, 0)
            road_val = road_factors.get(factor_key, 0)
            diff = home_val - road_val
            coef = coefs.get(coef_key, 0)

            # Divide by 100 to convert from percentage to decimal scale
            contribution = coef * (diff / 100.0)
            contributions[coef_key] = round(contribution, 2)
            total_contribution += contribution

        result["contributions"] = contributions
        result["predicted_rating_diff"] = round(total_contribution, 2)

    # Calculate actual rating diff
    home_ratings = decomp_data.get("home_ratings", {})
    road_ratings = decomp_data.get("road_ratings", {})
    result["actual_rating_diff"] = round(
        home_ratings.get("net_rating", 0) - road_ratings.get("net_rating", 0), 2
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate prompt training set for LLM interpretation testing")
    parser.add_argument("--season", required=True, help="Season (e.g., 2025-26)")
    parser.add_argument("--count", type=int, default=50, help="Number of most recent games to include")
    parser.add_argument("--repo-dir", default="/Users/robschoen/Dropbox/CC/NBA_Data", help="Path to NBA_Data repo")
    parser.add_argument("--output", default="prompt_training_set.json", help="Output file name")
    parser.add_argument("--factor-type", choices=["eight_factors", "four_factors", "both"], default="eight_factors",
                        help="Factor type for prompts")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)

    # Load game data
    csv_path = repo_dir / f"team_game_logs_{args.season}.csv"
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} games from {csv_path.name}")

    # Merge actual possessions from advanced stats
    adv_path = repo_dir / f"box_score_advanced_{args.season}.csv"
    if adv_path.exists():
        adv_df = pd.read_csv(adv_path)
        adv_df["game_id"] = adv_df["game_id"].astype(str).map(
            lambda v: v.zfill(10) if v.isdigit() else v
        )
        df["game_id"] = df["game_id"].astype(str).map(
            lambda v: v.zfill(10) if v.isdigit() else v
        )
        df = df.merge(
            adv_df[["game_id", "possessions_home", "possessions_road", "minutes_home", "minutes_road"]],
            on="game_id",
            how="left",
        )

    # Sort by date descending and get last N games
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date", ascending=False)

    # Get unique games (each game appears once in the merged CSV)
    games = df.head(args.count)
    print(f"Processing {len(games)} most recent games")

    # Load model
    model_file = repo_dir / "models" / "2018-2025.json"
    if not model_file.exists():
        model_file = repo_dir / "models" / "2023-2025.json"
    if not model_file.exists():
        print(f"Error: No model file found")
        return 1

    with open(model_file, "r") as f:
        model_data = json.load(f)
    print(f"Using model: {model_file.name}")

    # Compute league averages
    full_df = pd.read_csv(csv_path)
    league_avgs = _compute_season_league_averages(full_df)
    factor_ranges = _compute_factor_ranges(full_df)

    # Generate prompts
    training_set = {
        "description": f"Training set of {args.count} prompts from {args.season} season",
        "generated_at": datetime.now().isoformat(),
        "season": args.season,
        "factor_type": args.factor_type,
        "prompts": []
    }

    factor_types = ["eight_factors", "four_factors"] if args.factor_type == "both" else [args.factor_type]

    for _, game_row in games.iterrows():
        game_id = str(game_row["game_id"])
        game_date = str(game_row["game_date"].date())
        home_team = game_row["team_abbreviation_home"]
        road_team = game_row["team_abbreviation_road"]
        home_pts = int(game_row["pts_home"])
        road_pts = int(game_row["pts_road"])

        # Build decomposition
        decomp_data = _build_decomposition_data(game_row, model_data, league_avgs, factor_ranges)

        for factor_type in factor_types:
            decomp_for_type = _adjust_decomp_for_factor_type(decomp_data, factor_type, model_data)
            prompt = _build_interpretation_prompt(decomp_for_type, factor_type)

            training_set["prompts"].append({
                "game_id": game_id,
                "game_date": game_date,
                "matchup": f"{road_team} {road_pts} @ {home_team} {home_pts}",
                "factor_type": factor_type,
                "prompt": prompt,
                "expected_output": None,  # User fills this in
            })

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(training_set, f, indent=2)

    print(f"\nGenerated {len(training_set['prompts'])} prompts")
    print(f"Output: {output_path.absolute()}")
    print("\nYou can now use these prompts with the API Playground or your preferred testing tool.")


if __name__ == "__main__":
    sys.exit(main() or 0)
