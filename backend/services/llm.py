"""
LLM service for generating chart interpretations.
Supports both Anthropic (Claude) and OpenAI (GPT) providers.
"""

import os
import httpx
from typing import Optional, Dict, Any

from services.cache import get_cache_key, get_cached, set_cached

# Model configuration
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
OPENAI_MODEL = "gpt-4o-mini"

# Timeout for LLM API calls
LLM_TIMEOUT = 15.0


def _get_llm_config():
    """Get LLM configuration from environment (read fresh each time)."""
    config = {
        "provider": os.getenv("LLM_PROVIDER", "anthropic"),
        "anthropic_key": os.getenv("ANTHROPIC_API_KEY"),
        "openai_key": os.getenv("OPENAI_API_KEY"),
    }
    print(f"[LLM DEBUG] provider={config['provider']}, has_anthropic={bool(config['anthropic_key'])}, has_openai={bool(config['openai_key'])}")
    return config


async def generate_interpretation(
    decomposition_data: Dict[str, Any],
    factor_type: str,
    model_id: str,
) -> Optional[str]:
    """
    Generate a plain-English interpretation of the factor contribution chart.
    Returns cached response if available.
    """
    game_id = decomposition_data.get("game_id", "unknown")

    # Build cache key
    cache_key = get_cache_key("interpretation", game_id, factor_type, model_id)

    # Check cache
    cached = get_cached(cache_key)
    if cached:
        return cached

    # Build prompt
    prompt = _build_interpretation_prompt(decomposition_data, factor_type)

    # Get config fresh from environment
    config = _get_llm_config()

    # Call LLM based on provider
    interpretation = None
    if config["provider"] == "openai" and config["openai_key"]:
        interpretation = await _call_openai(prompt, config["openai_key"])
    elif config["anthropic_key"]:
        interpretation = await _call_anthropic(prompt, config["anthropic_key"])
    elif config["openai_key"]:
        interpretation = await _call_openai(prompt, config["openai_key"])

    # Cache result if successful
    if interpretation:
        set_cached(cache_key, interpretation)

    return interpretation


# Team abbreviation to city name mapping
TEAM_CITIES = {
    "ATL": "Atlanta", "BOS": "Boston", "BKN": "Brooklyn", "CHA": "Charlotte",
    "CHI": "Chicago", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver",
    "DET": "Detroit", "GSW": "Golden State", "HOU": "Houston", "IND": "Indiana",
    "LAC": "LA Clippers", "LAL": "LA Lakers", "MEM": "Memphis", "MIA": "Miami",
    "MIL": "Milwaukee", "MIN": "Minnesota", "NOP": "New Orleans", "NYK": "New York",
    "OKC": "Oklahoma City", "ORL": "Orlando", "PHI": "Philadelphia", "PHX": "Phoenix",
    "POR": "Portland", "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto",
    "UTA": "Utah", "WAS": "Washington",
    # Historical/alternate abbreviations
    "NJN": "New Jersey", "SEA": "Seattle", "VAN": "Vancouver", "CHO": "Charlotte",
    "NOH": "New Orleans", "NOK": "New Orleans/Oklahoma City",
}


def _get_city_name(abbr: str) -> str:
    """Convert team abbreviation to city name."""
    return TEAM_CITIES.get(abbr, abbr)


def _build_interpretation_prompt(data: Dict[str, Any], factor_type: str) -> str:
    """Build the prompt for the LLM."""
    home_abbr = data.get("home_team", "Home")
    road_abbr = data.get("road_team", "Road")
    home_city = _get_city_name(home_abbr)
    road_city = _get_city_name(road_abbr)
    home_pts = data.get("home_pts", 0)
    road_pts = data.get("road_pts", 0)
    game_date = data.get("game_date", "")

    # Get factor values and league averages
    home_factors = data.get("home_factors", {})
    road_factors = data.get("road_factors", {})
    league_avgs = data.get("league_averages", {})

    # Get league average values
    lg_efg = league_avgs.get("efg", 52.0)
    lg_bh = league_avgs.get("ball_handling", 86.0)
    lg_oreb = league_avgs.get("oreb", 25.0)
    lg_ft = league_avgs.get("ft_rate", 20.0)

    # Format factor values with comparison to league average
    def format_factors_with_comparison(factors: dict, team: str) -> str:
        if not factors:
            return ""
        efg = factors.get("efg", 0)
        bh = factors.get("ball_handling", 0)
        oreb = factors.get("oreb", 0)
        ft = factors.get("ft_rate", 0)

        def compare(val, avg, higher_is_better=True):
            diff = val - avg
            if higher_is_better:
                return "GOOD" if diff > 0 else "BAD"
            else:
                return "BAD" if diff > 0 else "GOOD"

        lines = [
            f"  {team}:",
            f"    - eFG%: {efg:.1f}% (league avg: {lg_efg:.1f}%) → {compare(efg, lg_efg)}",
            f"    - Ball Handling: {bh:.1f}% (league avg: {lg_bh:.1f}%) → {compare(bh, lg_bh)}",
            f"    - OREB%: {oreb:.1f}% (league avg: {lg_oreb:.1f}%) → {compare(oreb, lg_oreb)}",
            f"    - FT Rate: {ft:.1f}% (league avg: {lg_ft:.1f}%) → {compare(ft, lg_ft)}",
        ]
        return "\n".join(lines)

    home_factors_text = format_factors_with_comparison(home_factors, home_city)
    road_factors_text = format_factors_with_comparison(road_factors, road_city)

    # Get contributions and sort by absolute value
    contributions = data.get("contributions", {})
    sorted_contributions = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Map contribution keys to readable names
    factor_display = {
        "home_shooting": f"{home_city} Shooting",
        "road_shooting": f"{road_city} Shooting",
        "home_ball_handling": f"{home_city} Ball Handling",
        "road_ball_handling": f"{road_city} Ball Handling",
        "home_orebounding": f"{home_city} Off Rebounding",
        "road_orebounding": f"{road_city} Off Rebounding",
        "home_free_throws": f"{home_city} FT Rate",
        "road_free_throws": f"{road_city} FT Rate",
        "shooting": "Shooting Differential",
        "ball_handling": "Ball Handling Differential",
        "orebounding": "Off Rebounding Differential",
        "free_throws": "FT Rate Differential",
    }

    contributions_text = "\n".join(
        f"  - {factor_display.get(factor, factor)}: {value:+.1f} (helped {home_city if value > 0 else road_city})"
        for factor, value in sorted_contributions
    )

    # Determine winner
    if home_pts > road_pts:
        winner_city = home_city
        loser_city = road_city
        margin = home_pts - road_pts
    else:
        winner_city = road_city
        loser_city = home_city
        margin = road_pts - home_pts

    prompt = f"""Analyze this NBA game's factor contributions.

{road_city} {road_pts} @ {home_city} {home_pts} ({game_date})
Winner: {winner_city} by {margin} points

FACTOR VALUES (with league average comparison):
{home_factors_text}
{road_factors_text}

CONTRIBUTIONS TO OUTCOME (sorted by impact):
{contributions_text}

Write a brief paragraph (3-6 sentences) explaining what decided this game.

CRITICAL RULES:
1. Focus on the biggest contributors first - these decided the game
2. A factor marked "GOOD" means above league average (positive for that team)
3. A factor marked "BAD" means below league average (negative for that team)
4. Describe BAD factors negatively: "struggled", "poor shooting", "sloppy with the ball"
5. Describe GOOD factors positively: "shot well", "took care of the ball", "dominated the glass"
6. Reference actual percentages (e.g., "shot 55.6% eFG" or "grabbed 32% of offensive rebounds")
7. Only mention "above/below league average" when it adds clarity, not every time
8. Do NOT quote the +/- contribution numbers
9. Use city names ({home_city}, {road_city}), never abbreviations
10. Write naturally for casual basketball fans"""

    return prompt


async def _call_anthropic(prompt: str, api_key: str) -> Optional[str]:
    """Call Claude API."""
    if not api_key:
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 400,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=LLM_TIMEOUT,
            )

            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            return None
    except Exception:
        return None


async def _call_openai(prompt: str, api_key: str) -> Optional[str]:
    """Call OpenAI API."""
    if not api_key:
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "max_tokens": 400,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=LLM_TIMEOUT,
            )

            print(f"[LLM DEBUG] OpenAI status: {response.status_code}")
            if response.status_code != 200:
                print(f"[LLM DEBUG] OpenAI error: {response.text}")

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            return None
    except Exception as e:
        print(f"[LLM DEBUG] OpenAI exception: {e}")
        return None


def is_llm_configured() -> bool:
    """Check if at least one LLM provider is configured."""
    config = _get_llm_config()
    return bool(config["anthropic_key"] or config["openai_key"])
