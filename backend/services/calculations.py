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

def compute_four_factors(team_row: Dict, opponent_row: Dict, possessions: Optional[float] = None) -> Dict[str, float]:
    fgm = team_row.get("fgm", 0) or 0
    fga = team_row.get("fga", 0) or 0
    fg3m = team_row.get("fg3m", 0) or 0
    ftm = team_row.get("ftm", 0) or 0
    oreb = team_row.get("oreb", 0) or 0
    tov = team_row.get("tov", 0) or 0

    opp_dreb = opponent_row.get("dreb", 0) or 0

    efg = (fgm + 0.5 * fg3m) / fga * 100 if fga > 0 else 0
    tov_pct = tov / possessions * 100 if possessions and possessions > 0 else 0
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

def compute_game_ratings(
    team_row: Dict,
    opponent_row: Dict,
    actual_possessions: Optional[float] = None,
    opp_actual_possessions: Optional[float] = None,
    actual_minutes: Optional[int] = None,
) -> Dict[str, float]:
    pts = team_row.get("pts", 0) or 0
    opp_pts = opponent_row.get("pts", 0) or 0

    team_poss = actual_possessions if actual_possessions and actual_possessions > 0 else 0
    opp_poss = opp_actual_possessions if opp_actual_possessions and opp_actual_possessions > 0 else 0

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


def _estimated_team_possessions(df: pd.DataFrame) -> pd.Series:
    return df["fga"] + 0.32 * df["fta"] - df["oreb"] + df["tov"]


def _estimated_opp_possessions(df: pd.DataFrame) -> pd.Series:
    return df["opp_fga"] + 0.32 * df["opp_fta"] - df["opp_oreb"] + df["opp_tov"]


def _resolve_team_possessions(df: pd.DataFrame) -> pd.Series:
    estimated = _estimated_team_possessions(df)
    if "actual_poss" in df.columns:
        actual = pd.to_numeric(df["actual_poss"], errors="coerce")
        return actual.where(actual > 0, estimated)
    return estimated


def _resolve_opp_possessions(df: pd.DataFrame) -> pd.Series:
    estimated = _estimated_opp_possessions(df)
    if "opp_actual_poss" in df.columns:
        actual = pd.to_numeric(df["opp_actual_poss"], errors="coerce")
        return actual.where(actual > 0, estimated)
    return estimated


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
        model_factor_ranges = model_data.get("factor_ranges", {}) or {}

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

        # Convert factor ranges to percentage scale
        factor_ranges_out: Dict[str, Dict[str, float]] = {}
        for k, v in model_factor_ranges.items():
            if isinstance(v, dict):
                q1 = float(v.get("q1", 0) or 0)
                q3 = float(v.get("q3", 0) or 0)
                # Convert from proportion to percentage if needed
                if 0 <= q1 <= 1.5:
                    q1 *= 100.0
                if 0 <= q3 <= 1.5:
                    q3 *= 100.0
                factor_ranges_out[k] = {"q1": round(q1, 2), "q3": round(q3, 2)}

        return {
            "factor_type": "four_factors",
            "contributions": contributions,
            "intercept": round(intercept, 2),
            "predicted_rating_diff": round(total_contribution, 2),
            "differentials": {k: v for k, v in differentials.items() if k in MODEL_FACTOR_MAP.values()},
            "league_averages": league_avgs_out,
            "factor_ranges": factor_ranges_out,
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
        model_factor_ranges = model_data.get("factor_ranges", {}) or {}

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

        # Convert factor ranges to percentage scale
        factor_ranges_out: Dict[str, Dict[str, float]] = {}
        for k, v in model_factor_ranges.items():
            if isinstance(v, dict):
                q1 = float(v.get("q1", 0) or 0)
                q3 = float(v.get("q3", 0) or 0)
                # Convert from proportion to percentage if needed
                if 0 <= q1 <= 1.5:
                    q1 *= 100.0
                if 0 <= q3 <= 1.5:
                    q3 *= 100.0
                factor_ranges_out[k] = {"q1": round(q1, 2), "q3": round(q3, 2)}

        return {
            "factor_type": "eight_factors",
            "contributions": contributions,
            "intercept": round(float(intercept), 2),
            "predicted_rating_diff": round(predicted, 2),
            "factor_values": factor_values,
            "league_averages": league_avgs_out,
            "factor_ranges": factor_ranges_out,
        }

    else:
        raise ValueError(f"Unknown factor type: {factor_type}")

def compute_league_aggregates(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
    exclude_playoffs: bool = False,
    last_n_games: Optional[int] = None,
) -> pd.DataFrame:
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

    # Last N games is applied per team (different teams can span different date windows)
    if last_n_games is not None and last_n_games > 0:
        filtered_df = (
            filtered_df.sort_values("game_date")
            .groupby("team", group_keys=False)
            .tail(last_n_games)
        )

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

    team_stats["possessions"] = _resolve_team_possessions(team_stats).round(1)
    team_stats["opp_possessions"] = _resolve_opp_possessions(team_stats).round(1)

    team_stats["tov_pct"] = (
        team_stats["tov"] / team_stats["possessions"].replace(0, pd.NA) * 100
    ).fillna(0).round(1)
    team_stats["ball_handling"] = (100 - team_stats["tov_pct"]).round(1)

    team_stats["ft_rate"] = (team_stats["ftm"] / team_stats["fga"] * 100).round(1)

    team_stats["off_rating"] = (
        team_stats["pts"] / team_stats["possessions"].replace(0, pd.NA) * 100
    ).fillna(0).round(1)
    team_stats["def_rating"] = (
        team_stats["opp_pts"] / team_stats["opp_possessions"].replace(0, pd.NA) * 100
    ).fillna(0).round(1)
    team_stats["net_rating"] = (team_stats["off_rating"] - team_stats["def_rating"]).round(1)

    team_stats["ppg"] = (team_stats["pts"] / team_stats["games"]).round(1)
    team_stats["opp_ppg"] = (team_stats["opp_pts"] / team_stats["games"]).round(1)

    team_stats["opp_efg_pct"] = ((team_stats["opp_fgm"] + 0.5 * team_stats["opp_fg3m"]) / team_stats["opp_fga"] * 100).round(1)
    team_stats["opp_tov_pct"] = (
        team_stats["opp_tov"] / team_stats["opp_possessions"].replace(0, pd.NA) * 100
    ).fillna(0).round(1)
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
    team_stats["pace"] = (
        (team_stats["possessions"] + team_stats["opp_possessions"]) / 2 / team_stats["games"].replace(0, pd.NA)
    ).fillna(0).round(1)

    # Strength of Schedule (SOS): average opponent ratings over games played.
    # This is game-weighted (teams faced more often have greater influence).
    if not filtered_df.empty:
        opponent_ratings = team_stats.set_index("team")[["off_rating", "def_rating", "net_rating"]]
        schedule_df = filtered_df[["team", "opponent"]].merge(
            opponent_ratings,
            left_on="opponent",
            right_index=True,
            how="left",
        )

        sos_by_team = schedule_df.groupby("team").agg(
            sos=("net_rating", "mean"),
            opp_avg_off_rating=("off_rating", "mean"),
            opp_avg_def_rating=("def_rating", "mean"),
        ).reset_index()

        team_stats = team_stats.merge(sos_by_team, on="team", how="left")
    else:
        team_stats["sos"] = 0.0
        team_stats["opp_avg_off_rating"] = 0.0
        team_stats["opp_avg_def_rating"] = 0.0

    league_avg_off_rating = float(team_stats["off_rating"].mean()) if len(team_stats) > 0 else 0.0
    league_avg_def_rating = float(team_stats["def_rating"].mean()) if len(team_stats) > 0 else 0.0

    team_stats["sos"] = team_stats["sos"].fillna(0).round(1)
    team_stats["off_sos"] = (team_stats["opp_avg_off_rating"].fillna(0) - league_avg_off_rating).round(1)
    team_stats["def_sos"] = (team_stats["opp_avg_def_rating"].fillna(0) - league_avg_def_rating).round(1)

    # Apply schedule adjustments in the opposite phase:
    # - Opponents with strong offense (high Off SOS) should make your defensive rating look better,
    #   so subtract Off SOS from DRtg.
    # - Opponents with weak/strong defense (Def SOS) should adjust ORtg in the opposite direction,
    #   so subtract Def SOS from ORtg.
    team_stats["adj_net_rating"] = (team_stats["net_rating"] + team_stats["sos"]).round(1)
    team_stats["adj_off_rating"] = (team_stats["off_rating"] - team_stats["def_sos"]).round(1)
    team_stats["adj_def_rating"] = (team_stats["def_rating"] - team_stats["off_sos"]).round(1)

    team_stats = team_stats.drop(columns=["opp_avg_off_rating", "opp_avg_def_rating"], errors="ignore")

    return team_stats

def _compute_stat_value(df: pd.DataFrame, stat: str) -> pd.Series:
    """Compute a stat value for each row in the dataframe."""
    poss = _resolve_team_possessions(df)
    opp_poss = _resolve_opp_possessions(df)

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
        return (df["tov"] / poss.replace(0, pd.NA) * 100).fillna(0).round(1)
    elif stat == "off_rating":
        return (df["pts"] / poss.replace(0, pd.NA) * 100).fillna(0).round(1)
    elif stat == "def_rating":
        return (df["opp_pts"] / opp_poss.replace(0, pd.NA) * 100).fillna(0).round(1)
    elif stat == "net_rating":
        off_rtg = df["pts"] / poss.replace(0, pd.NA) * 100
        def_rtg = df["opp_pts"] / opp_poss.replace(0, pd.NA) * 100
        return (off_rtg - def_rtg).fillna(0).round(1)
    elif stat == "ball_handling":
        tov_pct = df["tov"] / poss.replace(0, pd.NA) * 100
        return (100 - tov_pct).fillna(100).round(1)
    elif stat == "oreb_pct":
        return (df["oreb"] / (df["oreb"] + df["opp_dreb"]) * 100).round(1)
    elif stat == "ft_rate":
        return (df["ftm"] / df["fga"] * 100).round(1)
    elif stat == "opp_efg_pct":
        return ((df["opp_fgm"] + 0.5 * df["opp_fg3m"]) / df["opp_fga"] * 100).round(1)
    elif stat == "opp_ball_handling":
        opp_tov_pct = df["opp_tov"] / opp_poss.replace(0, pd.NA) * 100
        return (100 - opp_tov_pct).fillna(100).round(1)
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
        return ((poss + opp_poss) / 2).fillna(0).round(1)
    else:
        return pd.Series([0] * len(df), index=df.index)


def compute_league_average(df: pd.DataFrame, stat: str) -> float:
    """Compute the league-wide average for a stat across all games."""
    df_copy = df.copy()
    values = _compute_stat_value(df_copy, stat)
    return round(values.mean(), 1) if len(values) > 0 else 0.0


def compute_trend_series(df: pd.DataFrame, team: str, stat: str, exclude_non_regular: bool = False) -> pd.DataFrame:
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


def _normalize_contrib_game_id(game_id: Any) -> str:
    """Normalize game IDs to 10-digit strings for joins with contribution JSON."""
    if pd.isna(game_id):
        return ""
    gid = str(game_id).strip()
    if gid.endswith(".0"):
        gid = gid[:-2]
    digits = "".join(ch for ch in gid if ch.isdigit())
    if not digits:
        return gid
    return digits.zfill(10)


def _compute_factor_trend_from_snapshots(factor_snapshots: list, factor_key: str) -> list:
    """Build mini-chart trend series from per-game JSON factor values."""
    ordered = sorted(factor_snapshots, key=lambda s: s["game_date"])
    running_values = []
    trend_data = []

    for snap in ordered:
        value = float(snap["factor_values"].get(factor_key, 0) or 0)
        running_values.append(value)
        ma_5 = sum(running_values[-5:]) / min(5, len(running_values))

        trend_data.append({
            "game_id": snap["game_id"],
            "game_date": snap["game_date"].strftime("%Y-%m-%d") if pd.notna(snap["game_date"]) else "",
            "opponent": snap["opponent"],
            "home_away": snap["home_away"],
            "value": round(value, 1),
            "ma_5": round(ma_5, 1),
            "wl": snap["wl"],
        })

    return trend_data


def compute_contribution_analysis(
    team_df: pd.DataFrame,
    league_df: pd.DataFrame,
    contributions_data: Dict[str, Any],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exclude_playoffs: bool = False,
    last_n_games: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute contribution analysis from per-game contribution JSON values.

    Contributions in the source JSON are stored from the home-team perspective.
    This function converts each game to the selected team's perspective, then
    averages each of the 8 factors over the selected period.
    """
    del league_df  # Kept in signature for compatibility with existing call sites.

    filtered_team = team_df.copy()

    # NBA Cup final is always excluded.
    filtered_team = filtered_team[filtered_team["game_type"] != "nba_cup_final"]

    if exclude_playoffs:
        filtered_team = filtered_team[~filtered_team["game_type"].isin(["playoffs", "play_in"])]

    if start_date:
        filtered_team = filtered_team[filtered_team["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_team = filtered_team[filtered_team["game_date"] <= pd.to_datetime(end_date)]

    filtered_team = filtered_team.sort_values("game_date")

    if last_n_games is not None and last_n_games > 0:
        filtered_team = filtered_team.tail(last_n_games)

    if filtered_team.empty:
        raise ValueError("No games found for the specified team and date range")

    contrib_games = contributions_data.get("games", []) if isinstance(contributions_data, dict) else []
    contrib_by_game_id = {
        _normalize_contrib_game_id(g.get("game_id")): g
        for g in contrib_games
        if isinstance(g, dict)
    }

    base_factors = ["shooting", "ball_handling", "orebounding", "free_throws"]
    all_factors = [
        "shooting",
        "ball_handling",
        "orebounding",
        "free_throws",
        "opp_shooting",
        "opp_ball_handling",
        "opp_orebounding",
        "opp_free_throws",
    ]
    league_avg_key_map = {
        "shooting": "efg",
        "ball_handling": "ball_handling",
        "orebounding": "oreb",
        "free_throws": "ft_rate",
    }

    factor_snapshots = []
    usable_game_ids = set()

    for _, row in filtered_team.iterrows():
        gid = _normalize_contrib_game_id(row.get("game_id"))
        game_entry = contrib_by_game_id.get(gid)
        if not game_entry:
            continue

        factors = game_entry.get("factors", {})
        home_factors = factors.get("home", []) if isinstance(factors, dict) else []
        road_factors = factors.get("road", []) if isinstance(factors, dict) else []
        if len(home_factors) < 4 or len(road_factors) < 4:
            continue

        home_away = str(row.get("home_away", "home") or "home")
        perspective_sign = 1.0 if home_away == "home" else -1.0
        team_factors = home_factors if home_away == "home" else road_factors
        opp_factors = road_factors if home_away == "home" else home_factors

        game_contributions: Dict[str, float] = {}
        game_factor_values: Dict[str, float] = {}
        game_league_averages: Dict[str, float] = {}

        model_info = game_entry.get("model", {})
        model_league_avgs = model_info.get("league_averages", {}) if isinstance(model_info, dict) else {}

        for i, factor in enumerate(base_factors):
            team_item = team_factors[i] if i < len(team_factors) else {}
            opp_item = opp_factors[i] if i < len(opp_factors) else {}

            game_contributions[factor] = perspective_sign * float(team_item.get("contribution", 0) or 0)
            game_contributions[f"opp_{factor}"] = perspective_sign * float(opp_item.get("contribution", 0) or 0)

            game_factor_values[factor] = float(team_item.get("value", 0) or 0)
            game_factor_values[f"opp_{factor}"] = float(opp_item.get("value", 0) or 0)

            league_avg_val = float(model_league_avgs.get(league_avg_key_map[factor], 0) or 0)
            game_league_averages[factor] = league_avg_val
            game_league_averages[f"opp_{factor}"] = league_avg_val

        wl_value = row.get("wl")
        if pd.isna(wl_value) or wl_value not in ["W", "L"]:
            wl_value = "W" if float(row.get("pts", 0) or 0) > float(row.get("opp_pts", 0) or 0) else "L"

        factor_snapshots.append({
            "game_id": gid,
            "game_date": pd.to_datetime(row.get("game_date"), errors="coerce"),
            "opponent": str(row.get("opponent", "")),
            "home_away": home_away,
            "wl": str(wl_value),
            "contributions": game_contributions,
            "factor_values": game_factor_values,
            "league_averages": game_league_averages,
        })
        usable_game_ids.add(gid)

    if not factor_snapshots:
        raise ValueError("No contribution records found for the specified team and date range")

    games_mask = filtered_team["game_id"].map(_normalize_contrib_game_id).isin(usable_game_ids)
    filtered_team_with_contrib = filtered_team[games_mask].copy()
    filtered_team_with_contrib = filtered_team_with_contrib.sort_values("game_date")

    games_analyzed = len(filtered_team_with_contrib)
    wins = int((filtered_team_with_contrib["pts"] > filtered_team_with_contrib["opp_pts"]).sum())
    losses = games_analyzed - wins
    win_pct = round(wins / games_analyzed, 3) if games_analyzed > 0 else 0.0

    team_stats = _compute_team_period_stats(filtered_team_with_contrib)
    actual_net_rating = team_stats.get("net_rating", 0.0)

    contributions: Dict[str, float] = {}
    factor_values: Dict[str, float] = {}
    league_avgs_out: Dict[str, float] = {}

    for factor in all_factors:
        contribution_vals = [snap["contributions"].get(factor, 0) for snap in factor_snapshots]
        value_vals = [snap["factor_values"].get(factor, 0) for snap in factor_snapshots]
        avg_vals = [snap["league_averages"].get(factor, 0) for snap in factor_snapshots]

        contributions[factor] = round(sum(contribution_vals) / len(contribution_vals), 2)
        factor_values[factor] = round(sum(value_vals) / len(value_vals), 2)
        league_avgs_out[factor] = round(sum(avg_vals) / len(avg_vals), 2)

    predicted_net_rating = round(sum(contributions.values()), 1)

    sorted_factors = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_4 = sorted_factors[:4]

    top_contributors = []
    for factor_key, contribution_val in top_4:
        trend_data = _compute_factor_trend_from_snapshots(factor_snapshots, factor_key)
        top_contributors.append({
            "factor": factor_key,
            "factor_label": CONTRIBUTION_FACTOR_LABELS.get(factor_key, factor_key),
            "value": factor_values.get(factor_key, 0),
            "league_avg": league_avgs_out.get(factor_key, 0),
            "contribution": contribution_val,
            "trend_data": trend_data,
        })

    first_date = filtered_team_with_contrib["game_date"].min()
    last_date = filtered_team_with_contrib["game_date"].max()

    return {
        "games_analyzed": games_analyzed,
        "wins": wins,
        "losses": losses,
        "win_pct": win_pct,
        "net_rating": round(actual_net_rating, 1),
        "predicted_net_rating": predicted_net_rating,
        "contributions": contributions,
        "factor_values": factor_values,
        "league_averages": league_avgs_out,
        "top_contributors": top_contributors,
        "intercept": 0.0,
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

    # Ball Handling (100 - TOV%), using actual possessions
    poss = (
        pd.to_numeric(team_df["actual_poss"], errors="coerce").where(
            pd.to_numeric(team_df["actual_poss"], errors="coerce") > 0
        ).sum()
        if "actual_poss" in team_df.columns else 0
    )
    if poss <= 0:
        poss = totals["fga"] + 0.32 * totals["fta"] - totals["oreb"] + totals["tov"]
    if poss > 0:
        tov_pct = totals["tov"] / poss * 100
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

    opp_poss = (
        pd.to_numeric(team_df["opp_actual_poss"], errors="coerce").where(
            pd.to_numeric(team_df["opp_actual_poss"], errors="coerce") > 0
        ).sum()
        if "opp_actual_poss" in team_df.columns else 0
    )
    if opp_poss <= 0:
        opp_poss = totals["opp_fga"] + 0.32 * totals["opp_fta"] - totals["opp_oreb"] + totals["opp_tov"]
    if opp_poss > 0:
        opp_tov_pct = totals["opp_tov"] / opp_poss * 100
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

    # Ratings using actual possessions
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

    poss = (
        pd.to_numeric(league_df["actual_poss"], errors="coerce").where(
            pd.to_numeric(league_df["actual_poss"], errors="coerce") > 0
        ).sum()
        if "actual_poss" in league_df.columns else 0
    )
    opp_poss = (
        pd.to_numeric(league_df["opp_actual_poss"], errors="coerce").where(
            pd.to_numeric(league_df["opp_actual_poss"], errors="coerce") > 0
        ).sum()
        if "opp_actual_poss" in league_df.columns else 0
    )
    if poss <= 0:
        poss = totals["fga"] + 0.32 * totals["fta"] - totals["oreb"] + totals["tov"]
    if opp_poss <= 0:
        opp_poss = totals["opp_fga"] + 0.32 * totals["opp_fta"] - totals["opp_oreb"] + totals["opp_tov"]

    if poss > 0:
        tov_pct = totals["tov"] / poss * 100
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

    if opp_poss > 0:
        opp_tov_pct = totals["opp_tov"] / opp_poss * 100
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
    contributions_data: Dict[str, Any],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exclude_playoffs: bool = False,
    last_n_games: Optional[int] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Compute league-wide top contributors from per-game contribution JSON."""
    filtered_df = league_df.copy()

    # NBA Cup final is always excluded.
    filtered_df = filtered_df[filtered_df["game_type"] != "nba_cup_final"]

    if exclude_playoffs:
        filtered_df = filtered_df[~filtered_df["game_type"].isin(["playoffs", "play_in"])]

    if start_date:
        filtered_df = filtered_df[filtered_df["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df["game_date"] <= pd.to_datetime(end_date)]

    # Last N games is applied per team (different teams can span different date windows)
    if last_n_games is not None and last_n_games > 0:
        filtered_df = (
            filtered_df.sort_values("game_date")
            .groupby("team", group_keys=False)
            .tail(last_n_games)
        )

    if filtered_df.empty:
        return {
            "top_positive": [],
            "top_negative": [],
            "league_averages": {},
            "coefficients": {},
            "start_date": start_date or "",
            "end_date": end_date or "",
        }

    contrib_games = contributions_data.get("games", []) if isinstance(contributions_data, dict) else []
    contrib_by_game_id = {
        _normalize_contrib_game_id(g.get("game_id")): g
        for g in contrib_games
        if isinstance(g, dict)
    }

    base_factors = ["shooting", "ball_handling", "orebounding", "free_throws"]
    all_factors = [
        "shooting",
        "ball_handling",
        "orebounding",
        "free_throws",
        "opp_shooting",
        "opp_ball_handling",
        "opp_orebounding",
        "opp_free_throws",
    ]
    league_avg_key_map = {
        "shooting": "efg",
        "ball_handling": "ball_handling",
        "orebounding": "oreb",
        "free_throws": "ft_rate",
    }

    snapshots = []
    usable_game_ids = set()

    for _, row in filtered_df.iterrows():
        gid = _normalize_contrib_game_id(row.get("game_id"))
        game_entry = contrib_by_game_id.get(gid)
        if not game_entry:
            continue

        factors = game_entry.get("factors", {})
        home_factors = factors.get("home", []) if isinstance(factors, dict) else []
        road_factors = factors.get("road", []) if isinstance(factors, dict) else []
        if len(home_factors) < 4 or len(road_factors) < 4:
            continue

        home_away = str(row.get("home_away", "home") or "home")
        perspective_sign = 1.0 if home_away == "home" else -1.0
        team_factors = home_factors if home_away == "home" else road_factors
        opp_factors = road_factors if home_away == "home" else home_factors

        model_info = game_entry.get("model", {})
        model_league_avgs = model_info.get("league_averages", {}) if isinstance(model_info, dict) else {}
        model_coefficients = model_info.get("coefficients", {}) if isinstance(model_info, dict) else {}

        game_contributions: Dict[str, float] = {}
        game_factor_values: Dict[str, float] = {}
        game_league_averages: Dict[str, float] = {}
        game_coefficients: Dict[str, float] = {}

        for i, factor in enumerate(base_factors):
            team_item = team_factors[i] if i < len(team_factors) else {}
            opp_item = opp_factors[i] if i < len(opp_factors) else {}

            game_contributions[factor] = perspective_sign * float(team_item.get("contribution", 0) or 0)
            game_contributions[f"opp_{factor}"] = perspective_sign * float(opp_item.get("contribution", 0) or 0)

            game_factor_values[factor] = float(team_item.get("value", 0) or 0)
            game_factor_values[f"opp_{factor}"] = float(opp_item.get("value", 0) or 0)

            league_avg_val = float(model_league_avgs.get(league_avg_key_map[factor], 0) or 0)
            game_league_averages[factor] = league_avg_val
            game_league_averages[f"opp_{factor}"] = league_avg_val

            coef_val = float(model_coefficients.get(factor, 0) or 0)
            game_coefficients[factor] = coef_val
            game_coefficients[f"opp_{factor}"] = coef_val

        snapshots.append({
            "team": str(row.get("team", "")),
            "game_id": gid,
            "game_date": pd.to_datetime(row.get("game_date"), errors="coerce"),
            "contributions": game_contributions,
            "factor_values": game_factor_values,
            "league_averages": game_league_averages,
            "coefficients": game_coefficients,
        })
        usable_game_ids.add(gid)

    if not snapshots:
        return {
            "top_positive": [],
            "top_negative": [],
            "league_averages": {},
            "coefficients": {},
            "start_date": "",
            "end_date": "",
        }

    filtered_with_contrib = filtered_df[
        filtered_df["game_id"].map(_normalize_contrib_game_id).isin(usable_game_ids)
    ].copy()

    teams = sorted({snap["team"] for snap in snapshots if snap["team"]})
    all_contributions = []

    for team in teams:
        team_snaps = [snap for snap in snapshots if snap["team"] == team]
        if not team_snaps:
            continue

        for factor in all_factors:
            contribution_vals = [snap["contributions"].get(factor, 0) for snap in team_snaps]
            value_vals = [snap["factor_values"].get(factor, 0) for snap in team_snaps]

            avg_contribution = sum(contribution_vals) / len(contribution_vals)
            avg_value = sum(value_vals) / len(value_vals)

            all_contributions.append({
                "team": team,
                "factor": factor,
                "factor_label": CONTRIBUTION_FACTOR_LABELS.get(factor, factor),
                "value": round(avg_value, 1),
                "contribution": round(avg_contribution, 2),
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

    league_avgs_out: Dict[str, float] = {}
    coefficients_out: Dict[str, float] = {}
    for factor in all_factors:
        avg_vals = [snap["league_averages"].get(factor, 0) for snap in snapshots]
        coef_vals = [snap["coefficients"].get(factor, 0) for snap in snapshots]
        league_avgs_out[factor] = round(sum(avg_vals) / len(avg_vals), 2)
        coefficients_out[factor] = round(sum(coef_vals) / len(coef_vals), 4)

    first_date = filtered_with_contrib["game_date"].min()
    last_date = filtered_with_contrib["game_date"].max()

    return {
        "top_positive": sorted_positive,
        "top_negative": sorted_negative,
        "league_averages": league_avgs_out,
        "coefficients": coefficients_out,
        "start_date": first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else "",
        "end_date": last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "",
    }
