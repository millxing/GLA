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
            "predicted_margin": round(total_contribution, 2),
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
            "predicted_margin": round(predicted, 2),
            "factor_values": factor_values,
            "league_averages": league_avgs_out,
        }

    else:
        raise ValueError(f"Unknown factor type: {factor_type}")

def compute_league_aggregates(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    filtered_df = df.copy()

    if start_date:
        filtered_df = filtered_df[filtered_df["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df["game_date"] <= pd.to_datetime(end_date)]

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


def compute_trend_series(df: pd.DataFrame, team: str, stat: str) -> pd.DataFrame:
    team_df = df[df["team"] == team].copy()
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
