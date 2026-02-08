"""
LLM service for generating chart interpretations.
Supports both Anthropic (Claude) and OpenAI (GPT) providers.
"""

import json
import os
from typing import Optional, Dict, Any

import httpx

from services.cache import get_cache_key, get_cached, set_cached

# Model configuration - defaults for real-time generation
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
OPENAI_MODEL = "gpt-4o-mini"

# Model tiers for batch generation
LLM_MODELS = {
    "historical": "gpt-4o-mini",           # Cheap model for historical seasons
    "current": "claude-sonnet-4-20250514", # Better model for current season
    "fallback": "gpt-4o-mini",             # Fallback for real-time generation
}

# Timeout for LLM API calls
LLM_TIMEOUT = 30.0  # Increased for larger prompts with examples
LLM_TIMEOUT_BATCH = 60.0  # Even longer for batch operations

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


def _build_interpretation_prompt(data: Dict[str, Any], factor_type: str = "eight_factors") -> str:
    """Build the prompt for the LLM using quintile-based classifications.

    Expects data with pre-computed _class fields (POOR, SUBPAR, AVERAGE, GOOD, EXCELLENT)
    and _contrib fields for each factor.
    """
    home_abbr = data.get("home_team", "Home")
    road_abbr = data.get("road_team", "Road")
    home_city = _get_city_name(home_abbr)
    road_city = _get_city_name(road_abbr)
    home_pts = data.get("home_pts", 0)
    road_pts = data.get("road_pts", 0)
    game_date = data.get("game_date", "")

    # Get score string
    score = data.get("score", f"{road_pts}-{home_pts}")
    matchup = data.get("matchup", f"{road_abbr}@{home_abbr}")

    # Determine winner
    if home_pts > road_pts:
        winner = home_abbr
        margin = home_pts - road_pts
    else:
        winner = road_abbr
        margin = road_pts - home_pts

    # Get ratings with classifications
    home_off_rating = data.get("home_off_rating", 0)
    home_off_rating_class = data.get("home_off_rating_class", "AVERAGE")
    home_def_rating = data.get("home_def_rating", 0)
    home_def_rating_class = data.get("home_def_rating_class", "AVERAGE")
    home_net_rating = data.get("home_net_rating", 0)
    home_net_rating_class = data.get("home_net_rating_class", "AVERAGE")

    road_off_rating = data.get("road_off_rating", 0)
    road_off_rating_class = data.get("road_off_rating_class", "AVERAGE")
    road_def_rating = data.get("road_def_rating", 0)
    road_def_rating_class = data.get("road_def_rating_class", "AVERAGE")
    road_net_rating = data.get("road_net_rating", 0)
    road_net_rating_class = data.get("road_net_rating_class", "AVERAGE")

    # Get factor values and classifications
    home_efg = data.get("home_efg", 0)
    home_efg_class = data.get("home_efg_class", "AVERAGE")
    home_efg_contrib = data.get("home_efg_contrib", 0)

    home_ball_handling = data.get("home_ball_handling", 0)
    home_ball_handling_class = data.get("home_ball_handling_class", "AVERAGE")
    home_ball_handling_contrib = data.get("home_ball_handling_contrib", 0)

    home_oreb = data.get("home_oreb", 0)
    home_oreb_class = data.get("home_oreb_class", "AVERAGE")
    home_oreb_contrib = data.get("home_oreb_contrib", 0)

    home_ft_rate = data.get("home_ft_rate", 0)
    home_ft_rate_class = data.get("home_ft_rate_class", "AVERAGE")
    home_ft_rate_contrib = data.get("home_ft_rate_contrib", 0)

    road_efg = data.get("road_efg", 0)
    road_efg_class = data.get("road_efg_class", "AVERAGE")
    road_efg_contrib = data.get("road_efg_contrib", 0)

    road_ball_handling = data.get("road_ball_handling", 0)
    road_ball_handling_class = data.get("road_ball_handling_class", "AVERAGE")
    road_ball_handling_contrib = data.get("road_ball_handling_contrib", 0)

    road_oreb = data.get("road_oreb", 0)
    road_oreb_class = data.get("road_oreb_class", "AVERAGE")
    road_oreb_contrib = data.get("road_oreb_contrib", 0)

    road_ft_rate = data.get("road_ft_rate", 0)
    road_ft_rate_class = data.get("road_ft_rate_class", "AVERAGE")
    road_ft_rate_contrib = data.get("road_ft_rate_contrib", 0)

    model = data.get("model", "2018-2025")

    # Build the data JSON section for the prompt
    game_data = f'''      "game_id": "{data.get("game_id", "")}",
      "game_date": "{game_date}",
      "matchup": "{matchup}",
      "score": "{score}",
      "home_team": "{home_abbr}",
      "road_team": "{road_abbr}",
      "home_pts": {home_pts},
      "road_pts": {road_pts},
      "winner": "{winner}",
      "margin": {margin},
      "model": "{model}",
      "home_off_rating": {home_off_rating},
      "home_off_rating_class": "{home_off_rating_class}",
      "home_def_rating": {home_def_rating},
      "home_def_rating_class": "{home_def_rating_class}",
      "home_net_rating": {home_net_rating},
      "home_net_rating_class": "{home_net_rating_class}",
      "road_off_rating": {road_off_rating},
      "road_off_rating_class": "{road_off_rating_class}",
      "road_def_rating": {road_def_rating},
      "road_def_rating_class": "{road_def_rating_class}",
      "road_net_rating": {road_net_rating},
      "road_net_rating_class": "{road_net_rating_class}",
      "home_efg": {home_efg},
      "home_efg_class": "{home_efg_class}",
      "home_efg_contrib": {home_efg_contrib},
      "home_ball_handling": {home_ball_handling},
      "home_ball_handling_class": "{home_ball_handling_class}",
      "home_ball_handling_contrib": {home_ball_handling_contrib},
      "home_oreb": {home_oreb},
      "home_oreb_class": "{home_oreb_class}",
      "home_oreb_contrib": {home_oreb_contrib},
      "home_ft_rate": {home_ft_rate},
      "home_ft_rate_class": "{home_ft_rate_class}",
      "home_ft_rate_contrib": {home_ft_rate_contrib},
      "road_efg": {road_efg},
      "road_efg_class": "{road_efg_class}",
      "road_efg_contrib": {road_efg_contrib},
      "road_ball_handling": {road_ball_handling},
      "road_ball_handling_class": "{road_ball_handling_class}",
      "road_ball_handling_contrib": {road_ball_handling_contrib},
      "road_oreb": {road_oreb},
      "road_oreb_class": "{road_oreb_class}",
      "road_oreb_contrib": {road_oreb_contrib},
      "road_ft_rate": {road_ft_rate},
      "road_ft_rate_class": "{road_ft_rate_class}",
      "road_ft_rate_contrib": {road_ft_rate_contrib}'''

    prompt = f'''You are an NBA media analyst producing brief, advanced, team-level bullet summaries from 4-factor analytics and modeled factor contributions.

OUTPUT FORMAT (HARD RULES)
- Output MUST be bullet points only.
- Each bullet MUST start with "- " (dash + space).
- No headings, no intro sentence, no tables, no numbering, no extra text before/after bullets.
- Never show the contribution values.

TERMS & INPUTS
- You will receive: final score, each team’s rating (pts/100) and rating classification by quintile:
  POOR (0–20%), SUBPAR (20–40%), AVERAGE (40–60%), GOOD (60–80%), EXCELLENT (80–100%).
- You will also receive four factors for each team (8 total): eFG%, TOV% (ball handling), OREB%, and free throw rate (FTM/FGA),
  each with a value and a quintile classification.
- You will also receive a modeled game contribution for each of the 8 factors. Contributions can be positive or negative.

NAMING RULES
- Refer to teams by CITY name only (never abbreviations).
- Exception: if the two Los Angeles teams play each other, use "Clippers" and "Lakers".

ROUNDING & NUMBERS
- When you cite a factor value that is a percentage, round to the nearest integer percent (e.g., 52%).
- If a stat is provided as a decimal rate that is naturally discussed as a percent, convert to percent then round (e.g., 0.247 -> 25%).
- For FTM/FGA, you may present it either as a rounded decimal (e.g., 0.24) OR as a percent-equivalent (24%); be consistent within a bullet.
- Never invent stats. Only use values present in the input.

WHAT TO INCLUDE
- Consider all 8 factors, but ONLY write bullets for factors with abs(contribution) >= 1.0 rating point.
- If none meet the threshold, output exactly one bullet explaining the game was not driven by any single 4-factor edge at the 1-point level (still bullet-only).

BULLET ORDERING (MOST IMPORTANT)
1) Build an internal list of candidate factors where abs(contribution) >= 1.0.
2) Sort candidates by abs(contribution) descending.
3) Tie-breakers (in order): higher abs(contribution), then HOME team factor before ROAD team factor (if home/road is available), then factor name alphabetical.
4) Write bullets strictly in that sorted order. Do NOT reorder for narrative flow.

CONSISTENCY RULES (MOST IMPORTANT)
Before writing the final bullets, do an internal verification pass:
- Each bullet’s wording must match the factor’s classification (POOR/SUBPAR/AVERAGE/GOOD/EXCELLENT).
- Each bullet must be directionally consistent with the contribution sign (helped vs hurt) and the game outcome (winner/loser context).
- Do NOT mix up which team owned which value/classification.
- Do NOT attribute an offensive stat to the opponent unless you explicitly frame it as defense (e.g., “X forced turnovers” vs “Y sloppy ball-handling”).

STYLE RULES
- Write for advanced NBA fans: crisp, analytical, natural language.
- Vary perspective to avoid monotony:
  • eFG%: “hot shooting” vs “smothered looks / strong contests”
  • TOV%: “clean ball security” vs “turnover pressure / forced mistakes”
  • OREB%: “owned the glass” vs “failed to finish possessions defensively”
  • FTM/FGA: “lived at the line” vs “fouled too much / couldn’t draw contact”
- Vary descriptive language: do not repeat the same adjective for GOOD/EXCELLENT/POOR every time.
- Reference the specific stat often, but not always (roughly 60–80% of bullets include a number).

PROCESS (DO THIS SILENTLY; DO NOT PRINT THESE STEPS)
- Parse teams, score, winner, and all factor values/classifications/contributions.
- Create the sorted candidate list using the ordering rules.
- Draft one bullet per candidate, in order, then run the consistency verification.
- Output only the final bullets.

Here is the data for the matchup:
{game_data}'''

    return prompt


async def _call_anthropic(prompt: str, api_key: str, model: str = None, timeout: float = None) -> Optional[str]:
    """Call Claude API."""
    if not api_key:
        return None

    use_model = model or ANTHROPIC_MODEL
    use_timeout = timeout or LLM_TIMEOUT

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
                    "model": use_model,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=use_timeout,
            )

            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                print(f"[LLM DEBUG] Anthropic error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"[LLM DEBUG] Anthropic exception: {e}")
        return None


async def _call_openai(prompt: str, api_key: str, model: str = None, timeout: float = None) -> Optional[str]:
    """Call OpenAI API."""
    if not api_key:
        return None

    use_model = model or OPENAI_MODEL
    use_timeout = timeout or LLM_TIMEOUT

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": use_model,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=use_timeout,
            )

            if response.status_code != 200:
                print(f"[LLM DEBUG] OpenAI error {response.status_code}: {response.text}")

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


# ----------------------- Synchronous versions for CLI batch use -----------------------

def _call_anthropic_sync(prompt: str, api_key: str, model: str = None, timeout: float = None) -> Optional[str]:
    """Synchronous version of Claude API call for CLI use."""
    if not api_key:
        return None

    use_model = model or ANTHROPIC_MODEL
    use_timeout = timeout or LLM_TIMEOUT_BATCH

    try:
        with httpx.Client() as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": use_model,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=use_timeout,
            )

            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                print(f"[LLM] Anthropic error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"[LLM] Anthropic exception: {e}")
        return None


def _call_openai_sync(prompt: str, api_key: str, model: str = None, timeout: float = None) -> Optional[str]:
    """Synchronous version of OpenAI API call for CLI use."""
    if not api_key:
        return None

    use_model = model or OPENAI_MODEL
    use_timeout = timeout or LLM_TIMEOUT_BATCH

    try:
        with httpx.Client() as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": use_model,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=use_timeout,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"[LLM] OpenAI error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"[LLM] OpenAI exception: {e}")
        return None


def generate_interpretation_sync(
    decomposition_data: Dict[str, Any],
    factor_type: str = "eight_factors",
    model: str = None,
) -> Optional[str]:
    """
    Synchronous version for batch generation (CLI use).

    Args:
        decomposition_data: Game data with pre-computed quintile classifications
        factor_type: 'eight_factors' (only supported type)
        model: Specific model to use (e.g., 'gpt-4o-mini', 'claude-sonnet-4-20250514')

    Returns:
        Generated interpretation text or None
    """
    # Build prompt
    prompt = _build_interpretation_prompt(decomposition_data, factor_type)

    # Get config
    config = _get_llm_config()

    # Determine which API to call based on model
    if model:
        if model.startswith("claude") or model.startswith("anthropic"):
            if config["anthropic_key"]:
                return _call_anthropic_sync(prompt, config["anthropic_key"], model=model)
        elif model.startswith("gpt") or model.startswith("o1"):
            if config["openai_key"]:
                return _call_openai_sync(prompt, config["openai_key"], model=model)

    # Fallback to default provider
    if config["anthropic_key"]:
        return _call_anthropic_sync(prompt, config["anthropic_key"], model=model)
    elif config["openai_key"]:
        return _call_openai_sync(prompt, config["openai_key"], model=model)

    return None
