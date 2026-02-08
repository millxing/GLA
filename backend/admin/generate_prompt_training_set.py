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


def _normalize_game_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else text


def _build_decomposition_data(game_row: pd.Series, contribution_entry: dict) -> dict:
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

    factor_keys = ["shooting", "ball_handling", "orebounding", "free_throws"]
    home_factor_rows = contribution_entry.get("factors", {}).get("home", [])
    road_factor_rows = contribution_entry.get("factors", {}).get("road", [])
    contributions = {}
    for i, factor_key in enumerate(factor_keys):
        home_contrib = home_factor_rows[i].get("contribution", 0) if i < len(home_factor_rows) else 0
        road_contrib = road_factor_rows[i].get("contribution", 0) if i < len(road_factor_rows) else 0
        contributions[f"home_{factor_key}"] = round(float(home_contrib), 2)
        contributions[f"road_{factor_key}"] = round(float(road_contrib), 2)

    intercept = float(contribution_entry.get("model", {}).get("intercept", 0) or 0)
    actual_rating_diff = round(home_ratings.get("net_rating", 0) - road_ratings.get("net_rating", 0), 2)

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
        "contributions": contributions,
        "intercept": intercept,
        "actual_rating_diff": actual_rating_diff,
        "model": contribution_entry.get("model", {}).get("model_id", "json_contributions"),
    }


def _adjust_decomp_for_factor_type(decomp_data: dict, factor_type: str) -> dict:
    """Add contributions based on factor type using pre-generated game contributions."""
    result = decomp_data.copy()
    contributions = decomp_data.get("contributions", {})
    intercept = float(decomp_data.get("intercept", 0) or 0)

    if factor_type == "eight_factors":
        result["contributions"] = contributions
        result["predicted_rating_diff"] = round(intercept + sum(contributions.values()), 2)
    else:
        four_contributions = {
            "shooting": round(contributions.get("home_shooting", 0) + contributions.get("road_shooting", 0), 2),
            "ball_handling": round(
                contributions.get("home_ball_handling", 0) + contributions.get("road_ball_handling", 0), 2
            ),
            "orebounding": round(contributions.get("home_orebounding", 0) + contributions.get("road_orebounding", 0), 2),
            "free_throws": round(contributions.get("home_free_throws", 0) + contributions.get("road_free_throws", 0), 2),
        }
        result["contributions"] = four_contributions
        result["predicted_rating_diff"] = round(intercept + sum(four_contributions.values()), 2)

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

    # Load per-game contribution JSON for this season
    contributions_path = repo_dir / "contributions" / f"contributions_{args.season}.json"
    if not contributions_path.exists():
        print(f"Error: Contribution file not found: {contributions_path}")
        return 1

    with open(contributions_path, "r") as f:
        contribution_payload = json.load(f)
    contributions_by_game = {
        _normalize_game_id(g.get("game_id")): g
        for g in contribution_payload.get("games", [])
        if _normalize_game_id(g.get("game_id"))
    }

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

        game_id_norm = _normalize_game_id(game_row.get("game_id"))
        contribution_entry = contributions_by_game.get(game_id_norm)
        if contribution_entry is None:
            print(f"Warning: Missing contribution entry for game {game_id_norm}, skipping")
            continue

        # Build decomposition
        decomp_data = _build_decomposition_data(game_row, contribution_entry)

        for factor_type in factor_types:
            decomp_for_type = _adjust_decomp_for_factor_type(decomp_data, factor_type)
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
