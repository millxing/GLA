import pandas as pd
from typing import Dict, Any, Optional

FACTOR_NAME_MAP = {
    "efg": "eFG%",
    "tov": "TOV%",
    "ball_handling": "Ball Handling",
    "oreb": "OREB%",
    "ft_rate": "FT Rate",
}

EIGHT_FACTOR_NAMES = {
    "home_efg": "Home eFG%",
    "home_tov": "Home TOV%",
    "home_oreb": "Home OREB%",
    "home_ft_rate": "Home FT Rate",
    "away_efg": "Away eFG%",
    "away_tov": "Away TOV%",
    "away_oreb": "Away OREB%",
    "away_ft_rate": "Away FT Rate",
}

# Model coefficient keys -> internal factor keys
MODEL_FACTOR_MAP = {
    "shooting": "efg",
    "ball_handling": "ball_handling",
    "orebounding": "oreb",
    "free_throws": "ft_rate",
}

def compute_four_factors(team_row: Dict, opponent_row: Dict) -> Dict[str, float]:
    fgm = team_row.get("fgm", 0) or 0
    fga = team_row.get("fga", 0) or 0
    fg3m = team_row.get("fg3m", 0) or 0
    ftm = team_row.get("ftm", 0) or 0
    fta = team_row.get("fta", 0) or 0
    oreb = team_row.get("oreb", 0) or 0
    tov = team_row.get("tov", 0) or 0

    opp_dreb = opponent_row.get("dreb", 0) or 0

    efg = (fgm + 0.5 * fg3m) / fga * 100 if fga > 0 else 0
    tov_pct = tov / (fga + 0.32 * fta + tov) * 100 if (fga + 0.32 * fta + tov) > 0 else 0
    ball_handling = 100 - tov_pct
    oreb_pct = oreb / (oreb + opp_dreb) * 100 if (oreb + opp_dreb) > 0 else 0
    ft_rate = ftm / fga * 100 if fga > 0 else 0

    return {
        "efg": round(efg, 2),
        "tov": round(tov_pct, 2),
        "ball_handling": round(ball_handling, 2),
        "oreb": round(oreb_pct, 2),
        "ft_rate": round(ft_rate, 2),
    }

def compute_possessions(team_row: Dict) -> float:
    fga = team_row.get("fga", 0) or 0
    fta = team_row.get("fta", 0) or 0
    oreb = team_row.get("oreb", 0) or 0
    tov = team_row.get("tov", 0) or 0

    possessions = fga + 0.32 * fta - oreb + tov
    return round(possessions, 2)

def compute_game_ratings(
    team_row: Dict,
    opponent_row: Dict,
    actual_possessions: Optional[float] = None,
    opp_actual_possessions: Optional[float] = None,
    actual_minutes: Optional[int] = None,
) -> Dict[str, float]:
    pts = team_row.get("pts", 0) or 0
    opp_pts = opponent_row.get("pts", 0) or 0

    # Use actual possessions if available, otherwise estimate
    if actual_possessions is not None and actual_possessions > 0:
        team_poss = actual_possessions
    else:
        team_poss = compute_possessions(team_row)

    if opp_actual_possessions is not None and opp_actual_possessions > 0:
        opp_poss = opp_actual_possessions
    else:
        opp_poss = compute_possessions(opponent_row)

    off_rating = pts / team_poss * 100 if team_poss > 0 else 0
    def_rating = opp_pts / opp_poss * 100 if opp_poss > 0 else 0

    # Calculate Pace if actual minutes available
    # Pace = avg possessions * (48 / actual game minutes)
    pace = None
    if actual_minutes is not None and actual_minutes > 0:
        actual_game_minutes = actual_minutes / 5  # Convert team minutes to game minutes
        avg_poss = (team_poss + opp_poss) / 2
        pace = avg_poss * (48 / actual_game_minutes)
        pace = round(pace, 1)

    return {
        "offensive_rating": round(off_rating, 2),
        "defensive_rating": round(def_rating, 2),
        "net_rating": round(off_rating - def_rating, 2),
        "possessions": round(team_poss, 1),
        "pace": pace,
    }

def compute_factor_differentials(home_factors: Dict, away_factors: Dict) -> Dict[str, float]:
    differentials = {}
    for key in home_factors:
        home_val = home_factors.get(key, 0)
        away_val = away_factors.get(key, 0)
        differentials[key] = round(home_val - away_val, 2)
    return differentials

def compute_decomposition(
    differentials: Dict[str, float],
    model: Dict,
    factor_type: str,
    home_factors: Optional[Dict] = None,
    away_factors: Optional[Dict] = None,
    league_averages: Optional[Dict] = None
) -> Dict[str, Any]:
    if factor_type == "four_factors":
        model_data = model.get("four_factors", {})
        coefficients = model_data.get("coefficients", {})
        intercept = model_data.get("intercept", 0)
        model_league_avgs = model_data.get("league_averages", {}) or {}

        contributions = {}
        total_contribution = intercept

        # Iterate over model keys to internal keys
        for model_key, internal_key in MODEL_FACTOR_MAP.items():
            if model_key not in coefficients:
                raise ValueError(f"Missing model factor: {model_key}")

            coef = coefficients[model_key]
            diff_pct = differentials.get(internal_key, 0)
            diff = diff_pct / 100.0
            contribution = coef * diff

            contributions[model_key] = round(contribution, 2)
            total_contribution += contribution

        # Convert league averages to percentage scale
        league_avgs_out: Dict[str, float] = {}
        for k, v in model_league_avgs.items():
            vv = float(v or 0)
            if 0 <= vv <= 1.5:
                vv *= 100.0
            league_avgs_out[k] = round(vv, 2)

        return {
            "factor_type": "four_factors",
            "contributions": contributions,
            "intercept": round(intercept, 2),
            "predicted_rating_diff": round(total_contribution, 2),
            "differentials": {k: v for k, v in differentials.items() if k in MODEL_FACTOR_MAP.values()},
            "league_averages": league_avgs_out,
        }

    elif factor_type == "eight_factors":
        # Symmetrical-coefficient eight-factors mode:
        # Break prediction into HOME and ROAD contributions by centering on league averages.
        # Uses the same coefficients as four_factors, just applies them differently.
        #
        # home_contribution = coef * (home_value - league_avg)
        # road_contribution = coef * (road_value - league_avg)
        # predicted = intercept + sum(home_contribution) - sum(road_contribution)

        # Use four_factors coefficients for both modes
        model_data = model.get("four_factors", {})
        coefficients = model_data.get("coefficients", {})
        intercept = model_data.get("intercept", 0)
        model_league_avgs = model_data.get("league_averages", {}) or {}

        if home_factors is None or away_factors is None:
            raise ValueError("Eight factors requires home_factors and away_factors")

        model_to_internal = MODEL_FACTOR_MAP

        model_to_league_key = {
            "shooting": "efg",
            "ball_handling": "ball_handling",
            "orebounding": "oreb_pct",
            "free_throws": "ft_rate",
        }

        contributions: Dict[str, float] = {}
        factor_values: Dict[str, float] = {}
        total_home = 0.0
        total_road = 0.0

        for model_factor, internal_key in model_to_internal.items():
            if model_factor not in coefficients:
                raise ValueError(f"Missing model factor: {model_factor}")

            coef = float(coefficients[model_factor])

            home_val = float(home_factors.get(internal_key, 0) or 0)
            road_val = float(away_factors.get(internal_key, 0) or 0)

            league_key = model_to_league_key.get(model_factor)
            league_avg = float(model_league_avgs.get(league_key, 0) or 0)

            # Convert league averages from proportion scale (0-1) to percent scale (0-100)
            # to match compute_four_factors output.
            if 0 <= league_avg <= 1.5:
                league_avg *= 100.0

            home_centered = (home_val - league_avg) / 100.0
            road_centered = (road_val - league_avg) / 100.0

            home_contrib = coef * home_centered
            road_contrib = coef * road_centered

            contributions[f"home_{model_factor}"] = round(home_contrib, 2)
            # Negate road contribution for display: positive = helps home team (green),
            # negative = hurts home team (red). Since road contrib is subtracted in prediction,
            # a positive road_contrib hurts home margin, so we negate for intuitive display.
            contributions[f"road_{model_factor}"] = round(-road_contrib, 2)

            factor_values[f"home_{model_factor}"] = round(home_val, 2)
            factor_values[f"road_{model_factor}"] = round(road_val, 2)

            total_home += home_contrib
            total_road += road_contrib

        predicted = float(intercept) + total_home - total_road

        league_avgs_out: Dict[str, float] = {}
        for k, v in model_league_avgs.items():
            vv = float(v or 0)
            if 0 <= vv <= 1.5:
                vv *= 100.0
            league_avgs_out[k] = round(vv, 2)

        return {
            "factor_type": "eight_factors",
            "contributions": contributions,
            "intercept": round(float(intercept), 2),
            "predicted_rating_diff": round(predicted, 2),
            "factor_values": factor_values,
            "league_averages": league_avgs_out,
        }

    else:
        raise ValueError(f"Unknown factor type: {factor_type}")

def compute_league_aggregates(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str], exclude_playoffs: bool = False) -> pd.DataFrame:
    filtered_df = df.copy()

    if start_date:
        filtered_df = filtered_df[filtered_df["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df["game_date"] <= pd.to_datetime(end_date)]
    # NBA Cup final never counts in league stats (always excluded)
    filtered_df = filtered_df[filtered_df["game_type"] != "nba_cup_final"]

    if exclude_playoffs:
        # Exclude playoffs and play-in games
        # Note: nba_cup_semi IS included as it counts toward regular season stats
        filtered_df = filtered_df[~filtered_df["game_type"].isin(["playoffs", "play_in"])]

    agg_cols = {
        "fgm": "sum",
        "fga": "sum",
        "fg3m": "sum",
        "fg3a": "sum",
        "ftm": "sum",
        "fta": "sum",
        "oreb": "sum",
        "dreb": "sum",
        "tov": "sum",
        "pts": "sum",
        "opp_fgm": "sum",
        "opp_fga": "sum",
        "opp_fg3m": "sum",
        "opp_fg3a": "sum",
        "opp_ftm": "sum",
        "opp_fta": "sum",
        "opp_oreb": "sum",
        "opp_dreb": "sum",
        "opp_tov": "sum",
        "opp_pts": "sum",
        "game_id": "count",
    }

    # Add actual possessions to aggregation if available
    if "actual_poss" in filtered_df.columns:
        agg_cols["actual_poss"] = "sum"
    if "opp_actual_poss" in filtered_df.columns:
        agg_cols["opp_actual_poss"] = "sum"

    team_stats = filtered_df.groupby("team").agg(agg_cols).reset_index()
    team_stats = team_stats.rename(columns={"game_id": "games"})

    team_stats["fg_pct"] = (team_stats["fgm"] / team_stats["fga"] * 100).round(1)
    team_stats["fg3_pct"] = (team_stats["fg3m"] / team_stats["fg3a"] * 100).round(1)
    team_stats["ft_pct"] = (team_stats["ftm"] / team_stats["fta"] * 100).round(1)
    team_stats["efg_pct"] = ((team_stats["fgm"] + 0.5 * team_stats["fg3m"]) / team_stats["fga"] * 100).round(1)

    team_stats["reb"] = team_stats["oreb"] + team_stats["dreb"]
    team_stats["opp_reb"] = team_stats["opp_oreb"] + team_stats["opp_dreb"]

    team_stats["oreb_pct"] = (team_stats["oreb"] / (team_stats["oreb"] + team_stats["opp_dreb"]) * 100).round(1)
    team_stats["dreb_pct"] = (team_stats["dreb"] / (team_stats["dreb"] + team_stats["opp_oreb"]) * 100).round(1)

    tov_denom = team_stats["fga"] + 0.32 * team_stats["fta"] + team_stats["tov"]
    team_stats["tov_pct"] = (team_stats["tov"] / tov_denom * 100).round(1)
    team_stats["ball_handling"] = (100 - team_stats["tov_pct"]).round(1)

    team_stats["ft_rate"] = (team_stats["ftm"] / team_stats["fga"] * 100).round(1)

    # Calculate estimated possessions
    estimated_poss = team_stats["fga"] + 0.32 * team_stats["fta"] - team_stats["oreb"] + team_stats["tov"]
    estimated_opp_poss = team_stats["opp_fga"] + 0.32 * team_stats["opp_fta"] - team_stats["opp_oreb"] + team_stats["opp_tov"]

    # Use actual possessions if available, otherwise use estimated
    if "actual_poss" in team_stats.columns:
        team_stats["possessions"] = team_stats["actual_poss"].fillna(estimated_poss).round(1)
    else:
        team_stats["possessions"] = estimated_poss.round(1)

    if "opp_actual_poss" in team_stats.columns:
        team_stats["opp_possessions"] = team_stats["opp_actual_poss"].fillna(estimated_opp_poss).round(1)
    else:
        team_stats["opp_possessions"] = estimated_opp_poss.round(1)

    team_stats["off_rating"] = (team_stats["pts"] / team_stats["possessions"] * 100).round(1)
    team_stats["def_rating"] = (team_stats["opp_pts"] / team_stats["opp_possessions"] * 100).round(1)
    team_stats["net_rating"] = (team_stats["off_rating"] - team_stats["def_rating"]).round(1)

    team_stats["ppg"] = (team_stats["pts"] / team_stats["games"]).round(1)
    team_stats["opp_ppg"] = (team_stats["opp_pts"] / team_stats["games"]).round(1)

    team_stats["opp_efg_pct"] = ((team_stats["opp_fgm"] + 0.5 * team_stats["opp_fg3m"]) / team_stats["opp_fga"] * 100).round(1)
    opp_tov_denom = team_stats["opp_fga"] + 0.32 * team_stats["opp_fta"] + team_stats["opp_tov"]
    team_stats["opp_tov_pct"] = (team_stats["opp_tov"] / opp_tov_denom * 100).round(1)
    team_stats["opp_ball_handling"] = (100 - team_stats["opp_tov_pct"]).round(1)
    team_stats["opp_ft_rate"] = (team_stats["opp_ftm"] / team_stats["opp_fga"] * 100).round(1)
    # Opponent OREB% = 100 - your DREB% (since DREB% + Opp OREB% = 100 for the same rebound pool)
    team_stats["opp_oreb_pct"] = (100 - team_stats["dreb_pct"]).round(1)

    # Determine wins/losses from points (wl column may be missing for road games)
    filtered_df = filtered_df.copy()
    filtered_df["is_win"] = filtered_df["pts"] > filtered_df["opp_pts"]
    wins = filtered_df[filtered_df["is_win"]].groupby("team").size().reset_index(name="wins")
    team_stats = team_stats.merge(wins, on="team", how="left")
    team_stats["wins"] = team_stats["wins"].fillna(0).astype(int)
    team_stats["losses"] = team_stats["games"] - team_stats["wins"]
    team_stats["win_pct"] = (team_stats["wins"] / team_stats["games"] * 100).where(team_stats["games"] > 0, 0).round(1)

    # Pace = average possessions per game (both teams combined / 2)
    team_stats["pace"] = ((team_stats["possessions"] + team_stats["opp_possessions"]) / 2 / team_stats["games"]).round(1)

    return team_stats

def _compute_stat_value(df: pd.DataFrame, stat: str) -> pd.Series:
    """Compute a stat value for each row in the dataframe."""
    if stat == "pts":
        return df["pts"]
    elif stat == "fg_pct":
        return (df["fgm"] / df["fga"] * 100).round(1)
    elif stat == "fg3_pct":
        return (df["fg3m"] / df["fg3a"] * 100).round(1)
    elif stat == "ft_pct":
        return (df["ftm"] / df["fta"] * 100).round(1)
    elif stat == "efg_pct":
        return ((df["fgm"] + 0.5 * df["fg3m"]) / df["fga"] * 100).round(1)
    elif stat == "oreb":
        return df["oreb"]
    elif stat == "dreb":
        return df["dreb"]
    elif stat == "reb":
        return df["oreb"] + df["dreb"]
    elif stat == "tov":
        return df["tov"]
    elif stat == "tov_pct":
        denom = df["fga"] + 0.32 * df["fta"] + df["tov"]
        return (df["tov"] / denom * 100).round(1)
    elif stat == "off_rating":
        # Use actual possessions if available, otherwise fall back to estimated
        estimated_poss = df["fga"] + 0.32 * df["fta"] - df["oreb"] + df["tov"]
        if "actual_poss" in df.columns:
            poss = df["actual_poss"].fillna(estimated_poss)
        else:
            poss = estimated_poss
        return (df["pts"] / poss * 100).round(1)
    elif stat == "def_rating":
        # Use opponent actual possessions if available
        estimated_opp_poss = df["opp_fga"] + 0.32 * df["opp_fta"] - df["opp_oreb"] + df["opp_tov"]
        if "opp_actual_poss" in df.columns:
            opp_poss = df["opp_actual_poss"].fillna(estimated_opp_poss)
        else:
            opp_poss = estimated_opp_poss
        return (df["opp_pts"] / opp_poss * 100).round(1)
    elif stat == "net_rating":
        estimated_poss = df["fga"] + 0.32 * df["fta"] - df["oreb"] + df["tov"]
        estimated_opp_poss = df["opp_fga"] + 0.32 * df["opp_fta"] - df["opp_oreb"] + df["opp_tov"]
        if "actual_poss" in df.columns:
            poss = df["actual_poss"].fillna(estimated_poss)
        else:
            poss = estimated_poss
        if "opp_actual_poss" in df.columns:
            opp_poss = df["opp_actual_poss"].fillna(estimated_opp_poss)
        else:
            opp_poss = estimated_opp_poss
        off_rtg = df["pts"] / poss * 100
        def_rtg = df["opp_pts"] / opp_poss * 100
        return (off_rtg - def_rtg).round(1)
    elif stat == "ball_handling":
        denom = df["fga"] + 0.32 * df["fta"] + df["tov"]
        tov_pct = df["tov"] / denom * 100
        return (100 - tov_pct).round(1)
    elif stat == "oreb_pct":
        return (df["oreb"] / (df["oreb"] + df["opp_dreb"]) * 100).round(1)
    elif stat == "ft_rate":
        return (df["ftm"] / df["fga"] * 100).round(1)
    elif stat == "opp_efg_pct":
        return ((df["opp_fgm"] + 0.5 * df["opp_fg3m"]) / df["opp_fga"] * 100).round(1)
    elif stat == "opp_ball_handling":
        opp_denom = df["opp_fga"] + 0.32 * df["opp_fta"] + df["opp_tov"]
        opp_tov_pct = df["opp_tov"] / opp_denom * 100
        return (100 - opp_tov_pct).round(1)
    elif stat == "opp_oreb_pct":
        return (df["opp_oreb"] / (df["opp_oreb"] + df["dreb"]) * 100).round(1)
    elif stat == "opp_ft_rate":
        return (df["opp_ftm"] / df["opp_fga"] * 100).round(1)
    elif stat == "fg2_pct":
        fg2m = df["fgm"] - df["fg3m"]
        fg2a = df["fga"] - df["fg3a"]
        return (fg2m / fg2a * 100).round(1)
    elif stat == "fg3a_rate":
        return (df["fg3a"] / df["fga"] * 100).round(1)
    elif stat == "opp_fg2_pct":
        opp_fg2m = df["opp_fgm"] - df["opp_fg3m"]
        opp_fg2a = df["opp_fga"] - df["opp_fg3a"]
        return (opp_fg2m / opp_fg2a * 100).round(1)
    elif stat == "opp_fg3_pct":
        return (df["opp_fg3m"] / df["opp_fg3a"] * 100).round(1)
    elif stat == "opp_fg3a_rate":
        return (df["opp_fg3a"] / df["opp_fga"] * 100).round(1)
    elif stat == "pace":
        estimated_poss = df["fga"] + 0.32 * df["fta"] - df["oreb"] + df["tov"]
        estimated_opp_poss = df["opp_fga"] + 0.32 * df["opp_fta"] - df["opp_oreb"] + df["opp_tov"]
        if "actual_poss" in df.columns:
            poss = df["actual_poss"].fillna(estimated_poss)
        else:
            poss = estimated_poss
        if "opp_actual_poss" in df.columns:
            opp_poss = df["opp_actual_poss"].fillna(estimated_opp_poss)
        else:
            opp_poss = estimated_opp_poss
        return ((poss + opp_poss) / 2).round(1)
    else:
        return pd.Series([0] * len(df), index=df.index)


def compute_league_average(df: pd.DataFrame, stat: str) -> float:
    """Compute the league-wide average for a stat across all games."""
    df_copy = df.copy()
    values = _compute_stat_value(df_copy, stat)
    return round(values.mean(), 1) if len(values) > 0 else 0.0


def compute_trend_series(df: pd.DataFrame, team: str, stat: str, exclude_non_regular: bool = True) -> pd.DataFrame:
    team_df = df[df["team"] == team].copy()

    # NBA Cup final never counts in stats (always excluded)
    team_df = team_df[team_df["game_type"] != "nba_cup_final"]

    # Filter out non-regular-season games if requested
    if exclude_non_regular:
        # Exclude playoffs and play-in games
        # Note: nba_cup_semi IS included as it counts toward regular season stats
        team_df = team_df[~team_df["game_type"].isin(["playoffs", "play_in"])]

    team_df = team_df.sort_values("game_date")

    team_df["value"] = _compute_stat_value(team_df, stat)

    # Determine win/loss from points if wl is missing
    team_df["wl"] = team_df.apply(
        lambda row: row["wl"] if pd.notna(row["wl"]) and row["wl"] in ["W", "L"]
        else ("W" if row["pts"] > row["opp_pts"] else "L"),
        axis=1
    )

    team_df["ma_5"] = team_df["value"].rolling(window=5, min_periods=1).mean().round(1)
    team_df["ma_10"] = team_df["value"].rolling(window=10, min_periods=1).mean().round(1)

    result = team_df[["game_id", "game_date", "opponent", "home_away", "value", "ma_5", "ma_10", "wl"]].copy()
    result["game_date"] = result["game_date"].dt.strftime("%Y-%m-%d")

    return result


# Factor labels for Contribution Analysis
CONTRIBUTION_FACTOR_LABELS = {
    "shooting": "Shooting (eFG%)",
    "ball_handling": "Ball Handling",
    "orebounding": "Offensive Rebounding",
    "free_throws": "Free Throws",
    "opp_shooting": "Opp Shooting (eFG%)",
    "opp_ball_handling": "Opp Ball Handling",
    "opp_orebounding": "Opp Off Rebounding",
    "opp_free_throws": "Opp Free Throws",
}

# Map contribution factor keys to stat keys for trend computation
CONTRIBUTION_FACTOR_TO_STAT = {
    "shooting": "efg_pct",
    "ball_handling": "ball_handling",
    "orebounding": "oreb_pct",
    "free_throws": "ft_rate",
    "opp_shooting": "opp_efg_pct",
    "opp_ball_handling": "opp_ball_handling",
    "opp_orebounding": "opp_oreb_pct",
    "opp_free_throws": "opp_ft_rate",
}


def compute_contribution_analysis(
    team_df: pd.DataFrame,
    league_df: pd.DataFrame,
    model: Dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exclude_playoffs: bool = False,
) -> Dict[str, Any]:
    """Compute contribution analysis for a team over a period.

    Uses a season-level eight-factor model to decompose the team's net rating
    into contributions from each factor.

    Args:
        team_df: DataFrame with team's games (already filtered to team)
        league_df: DataFrame with all league games (for computing league averages)
        model: Season-level model with eight_factors coefficients
        start_date: Optional start date filter (inclusive)
        end_date: Optional end date filter (inclusive)
        exclude_playoffs: If True, exclude playoff and play-in games

    Returns:
        Dict with contributions, factor_values, top_contributors, etc.
    """
    # Apply date filters to team games only
    filtered_team = team_df.copy()

    # Exclude nba_cup_final from all calculations
    filtered_team = filtered_team[filtered_team["game_type"] != "nba_cup_final"]
    full_season_league = league_df[league_df["game_type"] != "nba_cup_final"].copy()

    # Exclude playoffs/play-in if requested
    if exclude_playoffs:
        filtered_team = filtered_team[~filtered_team["game_type"].isin(["playoffs", "play_in"])]
        full_season_league = full_season_league[~full_season_league["game_type"].isin(["playoffs", "play_in"])]

    if start_date:
        filtered_team = filtered_team[filtered_team["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_team = filtered_team[filtered_team["game_date"] <= pd.to_datetime(end_date)]

    if filtered_team.empty:
        raise ValueError("No games found for the specified team and date range")

    games_analyzed = len(filtered_team)

    # Compute team's aggregated stats over the period
    # Using the same logic as compute_league_aggregates but for a single team
    team_stats = _compute_team_period_stats(filtered_team)

    # Compute league averages for the FULL SEASON (not filtered by date range)
    # This provides a stable baseline that doesn't change with the analysis period
    season_league_avgs = _compute_league_period_averages(full_season_league)

    # Get model data (coefficients only - league averages come from season data)
    # Season-level models store coefficients under eight_factors
    model_data = model.get("eight_factors", {})
    coefficients = model_data.get("coefficients", {})
    intercept = model_data.get("intercept", 0)

    # Map factor keys to stats
    factor_stats = {
        "shooting": ("efg_pct", team_stats.get("efg_pct", 0)),
        "ball_handling": ("ball_handling", team_stats.get("ball_handling", 0)),
        "orebounding": ("oreb_pct", team_stats.get("oreb_pct", 0)),
        "free_throws": ("ft_rate", team_stats.get("ft_rate", 0)),
        "opp_shooting": ("opp_efg_pct", team_stats.get("opp_efg_pct", 0)),
        "opp_ball_handling": ("opp_ball_handling", team_stats.get("opp_ball_handling", 0)),
        "opp_orebounding": ("opp_oreb_pct", team_stats.get("opp_oreb_pct", 0)),
        "opp_free_throws": ("opp_ft_rate", team_stats.get("opp_ft_rate", 0)),
    }

    contributions = {}
    factor_values = {}
    total_contribution = 0.0  # Sum of contributions only (not including intercept)

    for factor_key, (stat_key, team_value) in factor_stats.items():
        coef = float(coefficients.get(factor_key, 0))

        # Get league average from season data (already in percentage scale)
        league_avg = float(season_league_avgs.get(factor_key, 0))

        # Center on league average and compute contribution
        # Team value is already in percentage scale (0-100)
        centered = (team_value - league_avg) / 100.0
        contribution = coef * centered
        # Model coefficients already have correct signs:
        # - Positive coefs for team factors (higher = better)
        # - Negative coefs for opponent factors (higher opp stats = worse for you)

        contributions[factor_key] = round(contribution, 2)
        factor_values[factor_key] = round(team_value, 2)
        total_contribution += contribution

    # Build league_averages output from season averages
    league_avgs_out = {k: round(v, 2) for k, v in season_league_avgs.items()}

    # Compute actual net rating for the period
    actual_net_rating = team_stats.get("net_rating", 0)

    # Find top 4 contributors by absolute value
    sorted_factors = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_4 = sorted_factors[:4]

    # Build top contributors with trend data
    top_contributors = []
    for factor_key, contribution_val in top_4:
        stat_key = CONTRIBUTION_FACTOR_TO_STAT.get(factor_key, "net_rating")
        trend_data = _compute_factor_trend(filtered_team, stat_key)

        top_contributors.append({
            "factor": factor_key,
            "factor_label": CONTRIBUTION_FACTOR_LABELS.get(factor_key, factor_key),
            "value": factor_values.get(factor_key, 0),
            "league_avg": league_avgs_out.get(factor_key, 0),
            "contribution": contribution_val,
            "trend_data": trend_data,
        })

    # Determine date range for output
    first_date = filtered_team["game_date"].min()
    last_date = filtered_team["game_date"].max()

    return {
        "games_analyzed": games_analyzed,
        "net_rating": round(actual_net_rating, 1),
        "predicted_net_rating": round(total_contribution, 1),
        "contributions": contributions,
        "factor_values": factor_values,
        "league_averages": league_avgs_out,
        "top_contributors": top_contributors,
        "intercept": round(float(intercept), 2),
        "start_date": first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else "",
        "end_date": last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "",
    }


def _compute_team_period_stats(team_df: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregated stats for a team over a period."""
    if team_df.empty:
        return {}

    # Sum up counting stats
    totals = {
        "fgm": team_df["fgm"].sum(),
        "fga": team_df["fga"].sum(),
        "fg3m": team_df["fg3m"].sum(),
        "fg3a": team_df["fg3a"].sum(),
        "ftm": team_df["ftm"].sum(),
        "fta": team_df["fta"].sum(),
        "oreb": team_df["oreb"].sum(),
        "dreb": team_df["dreb"].sum(),
        "tov": team_df["tov"].sum(),
        "pts": team_df["pts"].sum(),
        "opp_fgm": team_df["opp_fgm"].sum(),
        "opp_fga": team_df["opp_fga"].sum(),
        "opp_fg3m": team_df["opp_fg3m"].sum(),
        "opp_ftm": team_df["opp_ftm"].sum(),
        "opp_fta": team_df["opp_fta"].sum(),
        "opp_oreb": team_df["opp_oreb"].sum(),
        "opp_dreb": team_df["opp_dreb"].sum(),
        "opp_tov": team_df["opp_tov"].sum(),
        "opp_pts": team_df["opp_pts"].sum(),
    }

    # Compute rate stats
    stats = {}

    # eFG%
    if totals["fga"] > 0:
        stats["efg_pct"] = (totals["fgm"] + 0.5 * totals["fg3m"]) / totals["fga"] * 100
    else:
        stats["efg_pct"] = 0

    # Ball Handling (100 - TOV%)
    tov_denom = totals["fga"] + 0.32 * totals["fta"] + totals["tov"]
    if tov_denom > 0:
        tov_pct = totals["tov"] / tov_denom * 100
        stats["ball_handling"] = 100 - tov_pct
    else:
        stats["ball_handling"] = 100

    # OREB%
    oreb_denom = totals["oreb"] + totals["opp_dreb"]
    if oreb_denom > 0:
        stats["oreb_pct"] = totals["oreb"] / oreb_denom * 100
    else:
        stats["oreb_pct"] = 0

    # FT Rate
    if totals["fga"] > 0:
        stats["ft_rate"] = totals["ftm"] / totals["fga"] * 100
    else:
        stats["ft_rate"] = 0

    # Opponent stats
    if totals["opp_fga"] > 0:
        stats["opp_efg_pct"] = (totals["opp_fgm"] + 0.5 * totals["opp_fg3m"]) / totals["opp_fga"] * 100
    else:
        stats["opp_efg_pct"] = 0

    opp_tov_denom = totals["opp_fga"] + 0.32 * totals["opp_fta"] + totals["opp_tov"]
    if opp_tov_denom > 0:
        opp_tov_pct = totals["opp_tov"] / opp_tov_denom * 100
        stats["opp_ball_handling"] = 100 - opp_tov_pct
    else:
        stats["opp_ball_handling"] = 100

    opp_oreb_denom = totals["opp_oreb"] + totals["dreb"]
    if opp_oreb_denom > 0:
        stats["opp_oreb_pct"] = totals["opp_oreb"] / opp_oreb_denom * 100
    else:
        stats["opp_oreb_pct"] = 0

    if totals["opp_fga"] > 0:
        stats["opp_ft_rate"] = totals["opp_ftm"] / totals["opp_fga"] * 100
    else:
        stats["opp_ft_rate"] = 0

    # Ratings (use actual possessions if available)
    if "actual_poss" in team_df.columns:
        poss = team_df["actual_poss"].sum()
    else:
        poss = totals["fga"] + 0.32 * totals["fta"] - totals["oreb"] + totals["tov"]

    if "opp_actual_poss" in team_df.columns:
        opp_poss = team_df["opp_actual_poss"].sum()
    else:
        opp_poss = totals["opp_fga"] + 0.32 * totals["opp_fta"] - totals["opp_oreb"] + totals["opp_tov"]

    if poss > 0:
        stats["off_rating"] = round(totals["pts"] / poss * 100, 1)
    else:
        stats["off_rating"] = 0

    if opp_poss > 0:
        stats["def_rating"] = round(totals["opp_pts"] / opp_poss * 100, 1)
    else:
        stats["def_rating"] = 0

    # Calculate net_rating from rounded values to match league_aggregates
    stats["net_rating"] = round(stats["off_rating"] - stats["def_rating"], 1)

    return {k: round(v, 1) for k, v in stats.items()}


def _compute_league_period_averages(league_df: pd.DataFrame) -> Dict[str, float]:
    """Compute league-wide averages for the period (all 8 factors)."""
    if league_df.empty:
        return {}

    # Use the same aggregation as compute_league_aggregates
    totals = {
        "fgm": league_df["fgm"].sum(),
        "fga": league_df["fga"].sum(),
        "fg3m": league_df["fg3m"].sum(),
        "ftm": league_df["ftm"].sum(),
        "fta": league_df["fta"].sum(),
        "oreb": league_df["oreb"].sum(),
        "dreb": league_df["dreb"].sum(),
        "tov": league_df["tov"].sum(),
        "opp_fgm": league_df["opp_fgm"].sum(),
        "opp_fga": league_df["opp_fga"].sum(),
        "opp_fg3m": league_df["opp_fg3m"].sum(),
        "opp_ftm": league_df["opp_ftm"].sum(),
        "opp_fta": league_df["opp_fta"].sum(),
        "opp_oreb": league_df["opp_oreb"].sum(),
        "opp_dreb": league_df["opp_dreb"].sum(),
        "opp_tov": league_df["opp_tov"].sum(),
    }

    avgs = {}

    # Team factors
    if totals["fga"] > 0:
        avgs["shooting"] = (totals["fgm"] + 0.5 * totals["fg3m"]) / totals["fga"] * 100
    else:
        avgs["shooting"] = 0

    tov_denom = totals["fga"] + 0.32 * totals["fta"] + totals["tov"]
    if tov_denom > 0:
        tov_pct = totals["tov"] / tov_denom * 100
        avgs["ball_handling"] = 100 - tov_pct
    else:
        avgs["ball_handling"] = 100

    oreb_denom = totals["oreb"] + totals["opp_dreb"]
    if oreb_denom > 0:
        avgs["orebounding"] = totals["oreb"] / oreb_denom * 100
    else:
        avgs["orebounding"] = 0

    if totals["fga"] > 0:
        avgs["free_throws"] = totals["ftm"] / totals["fga"] * 100
    else:
        avgs["free_throws"] = 0

    # Opponent factors (for eight-factor model)
    if totals["opp_fga"] > 0:
        avgs["opp_shooting"] = (totals["opp_fgm"] + 0.5 * totals["opp_fg3m"]) / totals["opp_fga"] * 100
    else:
        avgs["opp_shooting"] = 0

    opp_tov_denom = totals["opp_fga"] + 0.32 * totals["opp_fta"] + totals["opp_tov"]
    if opp_tov_denom > 0:
        opp_tov_pct = totals["opp_tov"] / opp_tov_denom * 100
        avgs["opp_ball_handling"] = 100 - opp_tov_pct
    else:
        avgs["opp_ball_handling"] = 100

    # Opponent OREB% (their offensive rebounds / their offensive rebound opportunities)
    opp_oreb_denom = totals["opp_oreb"] + totals["dreb"]
    if opp_oreb_denom > 0:
        avgs["opp_orebounding"] = totals["opp_oreb"] / opp_oreb_denom * 100
    else:
        avgs["opp_orebounding"] = 0

    if totals["opp_fga"] > 0:
        avgs["opp_free_throws"] = totals["opp_ftm"] / totals["opp_fga"] * 100
    else:
        avgs["opp_free_throws"] = 0

    return {k: round(v, 2) for k, v in avgs.items()}


def _compute_factor_trend(team_df: pd.DataFrame, stat: str) -> list:
    """Compute trend data for a specific factor/stat."""
    sorted_df = team_df.sort_values("game_date").copy()

    sorted_df["value"] = _compute_stat_value(sorted_df, stat)
    sorted_df["ma_5"] = sorted_df["value"].rolling(window=5, min_periods=1).mean().round(1)

    # Determine win/loss
    sorted_df["wl"] = sorted_df.apply(
        lambda row: row["wl"] if pd.notna(row.get("wl")) and row.get("wl") in ["W", "L"]
        else ("W" if row["pts"] > row["opp_pts"] else "L"),
        axis=1
    )

    result = []
    for _, row in sorted_df.iterrows():
        result.append({
            "game_id": str(row["game_id"]),
            "game_date": row["game_date"].strftime("%Y-%m-%d") if pd.notna(row["game_date"]) else "",
            "opponent": str(row.get("opponent", "")),
            "home_away": str(row.get("home_away", "")),
            "value": round(float(row["value"]) if pd.notna(row["value"]) else 0, 1),
            "ma_5": round(float(row["ma_5"]) if pd.notna(row["ma_5"]) else 0, 1),
            "wl": str(row["wl"]),
        })

    return result


def compute_league_top_contributors(
    league_df: pd.DataFrame,
    model: Dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exclude_playoffs: bool = False,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Compute top positive and negative contributors across all teams.

    Each contributor is a team + factor combination, showing how that team's
    performance in that factor contributed to their net rating.

    Args:
        league_df: DataFrame with all league games
        model: Season-level model with eight_factors coefficients
        start_date: Optional start date filter (inclusive)
        end_date: Optional end date filter (inclusive)
        exclude_playoffs: If True, exclude playoff and play-in games
        top_n: Number of top contributors to return for each category

    Returns:
        Dict with top_positive, top_negative lists, and league_averages
    """
    # Filter data
    filtered_df = league_df.copy()

    # Exclude nba_cup_final from all calculations
    filtered_df = filtered_df[filtered_df["game_type"] != "nba_cup_final"]

    if exclude_playoffs:
        filtered_df = filtered_df[~filtered_df["game_type"].isin(["playoffs", "play_in"])]

    if start_date:
        filtered_df = filtered_df[filtered_df["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df["game_date"] <= pd.to_datetime(end_date)]

    if filtered_df.empty:
        return {
            "top_positive": [],
            "top_negative": [],
            "league_averages": {},
            "start_date": start_date or "",
            "end_date": end_date or "",
        }

    # Compute league averages for the period
    league_avgs = _compute_league_period_averages(filtered_df)

    # Get model coefficients (season-level models use eight_factors)
    model_data = model.get("eight_factors", {})
    coefficients = model_data.get("coefficients", {})

    # Get unique teams
    teams = filtered_df["team"].unique()

    # Collect all team-factor contributions
    all_contributions = []

    for team in teams:
        team_df = filtered_df[filtered_df["team"] == team]
        if team_df.empty:
            continue

        # Compute team's aggregated stats
        team_stats = _compute_team_period_stats(team_df)

        # Map factor keys to stats
        factor_stats = {
            "shooting": ("efg_pct", team_stats.get("efg_pct", 0)),
            "ball_handling": ("ball_handling", team_stats.get("ball_handling", 0)),
            "orebounding": ("oreb_pct", team_stats.get("oreb_pct", 0)),
            "free_throws": ("ft_rate", team_stats.get("ft_rate", 0)),
            "opp_shooting": ("opp_efg_pct", team_stats.get("opp_efg_pct", 0)),
            "opp_ball_handling": ("opp_ball_handling", team_stats.get("opp_ball_handling", 0)),
            "opp_orebounding": ("opp_oreb_pct", team_stats.get("opp_oreb_pct", 0)),
            "opp_free_throws": ("opp_ft_rate", team_stats.get("opp_ft_rate", 0)),
        }

        for factor_key, (stat_key, team_value) in factor_stats.items():
            coef = float(coefficients.get(factor_key, 0))
            league_avg = float(league_avgs.get(factor_key, 0))

            # Center on league average and compute contribution
            centered = (team_value - league_avg) / 100.0
            contribution = coef * centered

            all_contributions.append({
                "team": team,
                "factor": factor_key,
                "factor_label": CONTRIBUTION_FACTOR_LABELS.get(factor_key, factor_key),
                "value": round(team_value, 1),
                "contribution": round(contribution, 2),
            })

    # Sort by contribution
    sorted_positive = sorted(
        [c for c in all_contributions if c["contribution"] > 0],
        key=lambda x: x["contribution"],
        reverse=True
    )[:top_n]

    sorted_negative = sorted(
        [c for c in all_contributions if c["contribution"] < 0],
        key=lambda x: x["contribution"]
    )[:top_n]

    # Determine actual date range from data
    first_date = filtered_df["game_date"].min()
    last_date = filtered_df["game_date"].max()

    return {
        "top_positive": sorted_positive,
        "top_negative": sorted_negative,
        "league_averages": {k: round(v, 2) for k, v in league_avgs.items()},
        "coefficients": {k: round(float(v), 4) for k, v in coefficients.items()},
        "start_date": first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else "",
        "end_date": last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "",
    }
