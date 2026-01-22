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
    tov_pct = tov / (fga + 0.44 * fta + tov) * 100 if (fga + 0.44 * fta + tov) > 0 else 0
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

    possessions = fga + 0.44 * fta - oreb + tov
    return round(possessions, 2)

def compute_game_ratings(team_row: Dict, opponent_row: Dict) -> Dict[str, float]:
    pts = team_row.get("pts", 0) or 0
    opp_pts = opponent_row.get("pts", 0) or 0

    team_poss = compute_possessions(team_row)
    opp_poss = compute_possessions(opponent_row)

    off_rating = pts / team_poss * 100 if team_poss > 0 else 0
    def_rating = opp_pts / opp_poss * 100 if opp_poss > 0 else 0

    return {
        "offensive_rating": round(off_rating, 2),
        "defensive_rating": round(def_rating, 2),
        "net_rating": round(off_rating - def_rating, 2),
        "possessions": team_poss,
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

        return {
            "factor_type": "four_factors",
            "contributions": contributions,
            "intercept": round(intercept, 2),
            "predicted_margin": round(total_contribution, 2),
            "differentials": {k: v for k, v in differentials.items() if k in MODEL_FACTOR_MAP.values()},
        }

    elif factor_type == "eight_factors":
        # Symmetrical-coefficient eight-factors mode:
        # Break prediction into HOME and ROAD contributions by centering on league averages.
        #
        # home_contribution = coef * (home_value - league_avg)
        # road_contribution = coef * (road_value - league_avg)
        # predicted = intercept + sum(home_contribution) - sum(road_contribution)

        model_data = model.get("eight_factors", {})
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
            contributions[f"road_{model_factor}"] = round(road_contrib, 2)

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

    tov_denom = team_stats["fga"] + 0.44 * team_stats["fta"] + team_stats["tov"]
    team_stats["tov_pct"] = (team_stats["tov"] / tov_denom * 100).round(1)
    team_stats["ball_handling"] = (100 - team_stats["tov_pct"]).round(1)

    team_stats["ft_rate"] = (team_stats["ftm"] / team_stats["fga"] * 100).round(1)

    team_stats["possessions"] = (team_stats["fga"] + 0.44 * team_stats["fta"] - team_stats["oreb"] + team_stats["tov"]).round(1)
    team_stats["opp_possessions"] = (team_stats["opp_fga"] + 0.44 * team_stats["opp_fta"] - team_stats["opp_oreb"] + team_stats["opp_tov"]).round(1)

    team_stats["off_rating"] = (team_stats["pts"] / team_stats["possessions"] * 100).round(1)
    team_stats["def_rating"] = (team_stats["opp_pts"] / team_stats["opp_possessions"] * 100).round(1)
    team_stats["net_rating"] = (team_stats["off_rating"] - team_stats["def_rating"]).round(1)

    team_stats["ppg"] = (team_stats["pts"] / team_stats["games"]).round(1)
    team_stats["opp_ppg"] = (team_stats["opp_pts"] / team_stats["games"]).round(1)

    team_stats["opp_efg_pct"] = ((team_stats["opp_fgm"] + 0.5 * team_stats["opp_fg3m"]) / team_stats["opp_fga"] * 100).round(1)
    opp_tov_denom = team_stats["opp_fga"] + 0.44 * team_stats["opp_fta"] + team_stats["opp_tov"]
    team_stats["opp_tov_pct"] = (team_stats["opp_tov"] / opp_tov_denom * 100).round(1)
    team_stats["opp_ft_rate"] = (team_stats["opp_ftm"] / team_stats["opp_fga"] * 100).round(1)

    wins = filtered_df[filtered_df["wl"] == "W"].groupby("team").size().reset_index(name="wins")
    team_stats = team_stats.merge(wins, on="team", how="left")
    team_stats["wins"] = team_stats["wins"].fillna(0).astype(int)
    team_stats["losses"] = team_stats["games"] - team_stats["wins"]
    team_stats["win_pct"] = (team_stats["wins"] / team_stats["games"] * 100).round(1)

    return team_stats

def compute_trend_series(df: pd.DataFrame, team: str, stat: str) -> pd.DataFrame:
    team_df = df[df["team"] == team].copy()
    team_df = team_df.sort_values("game_date")

    stat_map = {
        "pts": "pts",
        "fg_pct": None,
        "fg3_pct": None,
        "ft_pct": None,
        "efg_pct": None,
        "oreb": "oreb",
        "dreb": "dreb",
        "reb": None,
        "tov": "tov",
        "tov_pct": None,
        "off_rating": None,
        "def_rating": None,
        "net_rating": None,
    }

    if stat in stat_map and stat_map[stat] is not None:
        team_df["value"] = team_df[stat_map[stat]]
    elif stat == "fg_pct":
        team_df["value"] = (team_df["fgm"] / team_df["fga"] * 100).round(1)
    elif stat == "fg3_pct":
        team_df["value"] = (team_df["fg3m"] / team_df["fg3a"] * 100).round(1)
    elif stat == "ft_pct":
        team_df["value"] = (team_df["ftm"] / team_df["fta"] * 100).round(1)
    elif stat == "efg_pct":
        team_df["value"] = ((team_df["fgm"] + 0.5 * team_df["fg3m"]) / team_df["fga"] * 100).round(1)
    elif stat == "reb":
        team_df["value"] = team_df["oreb"] + team_df["dreb"]
    elif stat == "tov_pct":
        denom = team_df["fga"] + 0.44 * team_df["fta"] + team_df["tov"]
        team_df["value"] = (team_df["tov"] / denom * 100).round(1)
    elif stat == "off_rating":
        poss = team_df["fga"] + 0.44 * team_df["fta"] - team_df["oreb"] + team_df["tov"]
        team_df["value"] = (team_df["pts"] / poss * 100).round(1)
    elif stat == "def_rating":
        opp_poss = team_df["opp_fga"] + 0.44 * team_df["opp_fta"] - team_df["opp_oreb"] + team_df["opp_tov"]
        team_df["value"] = (team_df["opp_pts"] / opp_poss * 100).round(1)
    elif stat == "net_rating":
        poss = team_df["fga"] + 0.44 * team_df["fta"] - team_df["oreb"] + team_df["tov"]
        opp_poss = team_df["opp_fga"] + 0.44 * team_df["opp_fta"] - team_df["opp_oreb"] + team_df["opp_tov"]
        off_rtg = team_df["pts"] / poss * 100
        def_rtg = team_df["opp_pts"] / opp_poss * 100
        team_df["value"] = (off_rtg - def_rtg).round(1)
    else:
        team_df["value"] = 0

    team_df["ma_5"] = team_df["value"].rolling(window=5, min_periods=1).mean().round(1)
    team_df["ma_10"] = team_df["value"].rolling(window=10, min_periods=1).mean().round(1)

    result = team_df[["game_id", "game_date", "opponent", "home_away", "value", "ma_5", "ma_10", "wl"]].copy()
    result["game_date"] = result["game_date"].dt.strftime("%Y-%m-%d")

    return result
