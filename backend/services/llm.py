"""
LLM service for generating chart interpretations.
Supports both Anthropic (Claude) and OpenAI (GPT) providers.
"""

import json
import os
from typing import Optional, Dict, Any, Tuple

import httpx

from services.cache import get_cache_key, get_cached, set_cached

# Model configuration - defaults for real-time generation
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
OPENAI_MODEL = "gpt-5-mini"

# Model tiers for batch generation
LLM_MODELS = {
    "historical": "gpt-4o-mini",           # Legacy/historical generation
    "current": "gpt-5-mini",               # Current-season model (quality/cost target)
    "fallback": "gpt-5-mini",              # Real-time fallback model
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
    return config


def _select_runtime_provider_and_model(config: Dict[str, Optional[str]]) -> Tuple[Optional[str], Optional[str]]:
    """Select provider/model pair for real-time generation.

    Interpretations are currently OpenAI-only to keep output/model behavior
    consistent for the 2025-26 rollout.
    """
    if config["openai_key"]:
        return "openai", OPENAI_MODEL
    return None, None


def get_runtime_interpretation_model() -> Optional[str]:
    """Get the model name that real-time interpretation would use."""
    config = _get_llm_config()
    _, model = _select_runtime_provider_and_model(config)
    return model


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

    # Call LLM based on provider/model selection
    provider, model = _select_runtime_provider_and_model(config)
    interpretation = None
    if provider == "openai":
        interpretation = await _call_openai(prompt, config["openai_key"], model=model)
    elif provider == "anthropic":
        interpretation = await _call_anthropic(prompt, config["anthropic_key"], model=model)

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

    prompt = f'''You are an NBA media analyst. You produce ONLY bullet-point summaries from 4-factor analytics. You NEVER output reasoning, analysis steps, or explanations — ONLY the final bullets.

ABSOLUTE OUTPUT RULE: Your entire response must be bullet points starting with "- ". Nothing else. No preamble, no reasoning, no headers, no sign-off. If your response contains anything other than bullets starting with "- ", you have failed.

FORMAT RULES
- Each bullet starts with EXACTLY "- " (dash + space). Never use "*", "•", or numbers.
- Never show contribution values, sources, model names, or prompt references.

FOUR-FACTOR DEFINITIONS — WHAT EACH STAT MEASURES (AND DOES NOT MEASURE)
Each factor describes ONLY the named team's own performance. A team's factor NEVER implies anything about the opponent's performance in that same category. The two teams' stats for each factor are independent of each other.

  eFG% — How efficiently the team shot the ball. Low eFG% means THAT TEAM shot poorly. It does NOT mean the opponent shot well. High eFG% means THAT TEAM shot efficiently. It does NOT mean the opponent defended poorly.

  OREB% — How often the team grabbed its own offensive misses to create second-chance opportunities FOR ITSELF. Low OREB% means THAT TEAM failed to generate second-chance opportunities for itself. It does NOT mean the opponent got second-chance points. It does NOT describe defensive rebounding. It says NOTHING about the opponent's rebounding. High OREB% means THAT TEAM crashed the offensive glass and created extra chances for itself.

  Ball Handling (inverse of TOV%) — How well the team protected the ball and avoided turnovers. High ball handling means THAT TEAM avoided turnovers and kept its possessions. It does NOT mean possessions were "flipped" to anyone. Poor ball handling means THAT TEAM turned the ball over and gifted the opponent extra possessions via turnovers.

  FT Rate (FTA/FGA) — How often the team got to the free-throw line relative to its field-goal attempts. Low FT rate means THAT TEAM couldn't draw fouls or get to the stripe. It does NOT mean the opponent got free-throw opportunities. High FT rate means THAT TEAM drew fouls and lived at the line. It says NOTHING about the opponent's free-throw opportunities.

COMMON ERRORS TO AVOID
- NEVER say a team's low OREB% "allowed the opponent second-chance opportunities" or that the opponent "converted" second chances because of it. Those are independent stats.
- NEVER say a team's low OREB% means they "couldn't finish possessions defensively" or "couldn't clean the defensive glass." OREB% is an OFFENSIVE stat — it measures a team's ability to rebound its own misses on offense.
- NEVER say a team's low FT rate "ceded free-throw opportunities" to the opponent or that the opponent benefited at the line because of it.
- NEVER describe good ball handling as "flipping possessions the other way." Good ball handling simply means the team did NOT turn the ball over.
- NEVER call a factor a "decisive edge" or similar language for a team that LOST the game. For a losing team's positive factor, use language like "bright spot in defeat," "kept them competitive," or "wasn't enough to overcome."

CONTRIBUTION SIGN CONVENTION
All contributions use the HOME TEAM's perspective:
  POSITIVE contribution → helped the HOME team, hurt the ROAD team.
  NEGATIVE contribution → helped the ROAD team, hurt the HOME team.
This applies to ALL 8 factors (both home_* and road_*).

SIGN CONVENTION REFERENCE TABLE
  home_factor_contrib > 0 → home team's factor helped home team
  home_factor_contrib < 0 → home team's factor helped road team
  road_factor_contrib > 0 → road team's factor helped home team (road team's factor was bad for them)
  road_factor_contrib < 0 → road team's factor helped road team (road team's factor was good for them)

Pay special attention to road factors with POSITIVE contributions. A positive road_ball_handling_contrib means the road team's ball handling HELPED THE HOME TEAM — i.e., the road team turned it over and gave the home team extra possessions. Do NOT write that this helped the road team.

INTERPRETING EACH FACTOR CORRECTLY BY SIGN

  eFG (positive contrib = helped home):
    The named team shot well (if home) or shot poorly (if road with positive contrib, meaning it helped home).
    The named team shot poorly (if home with negative contrib) or shot well (if road with negative contrib, meaning it helped road).

  Ball Handling (positive contrib = helped home):
    If home_ball_handling_contrib > 0: The home team protected the ball well, denying the road team extra possessions.
    If home_ball_handling_contrib < 0: The home team turned it over, gifting the road team extra possessions.
    If road_ball_handling_contrib > 0: The road team turned it over, gifting the home team extra possessions.
    If road_ball_handling_contrib < 0: The road team protected the ball well, denying the home team extra possessions.

  OREB (positive contrib = helped home):
    If home_oreb_contrib > 0: The home team crashed the offensive glass and created second-chance opportunities for itself.
    If home_oreb_contrib < 0: The home team failed to rebound its own misses, missing out on second-chance opportunities for itself.
    If road_oreb_contrib > 0: The road team failed to rebound its own misses, missing out on second-chance opportunities for itself. This helped the home team.
    If road_oreb_contrib < 0: The road team crashed the offensive glass and created second-chance opportunities for itself. This helped the road team.

  FT Rate (positive contrib = helped home):
    If home_ft_rate_contrib > 0: The home team drew fouls and got to the free-throw line frequently.
    If home_ft_rate_contrib < 0: The home team couldn't draw fouls or get to the stripe.
    If road_ft_rate_contrib > 0: The road team couldn't draw fouls or get to the stripe. This helped the home team.
    If road_ft_rate_contrib < 0: The road team drew fouls and lived at the free-throw line. This helped the road team.

EXAMPLES OF CORRECT INTERPRETATION
Given: home=GSW, road=LAC, winner=LAC
  home_efg_contrib = -8.8 → negative → helped LAC → "Golden State's frigid 48% eFG was the game's defining weakness."
  road_oreb_contrib = -7.7 → negative → helped LAC → "Los Angeles crashed the glass at 38% OREB, generating second-chance opportunities for itself."
  home_oreb_contrib = +2.9 → positive → helped GSW → "Golden State's 29% OREB was a rare bright spot in defeat."

Given: home=WAS, road=HOU, winner=HOU
  road_ball_handling_contrib = +7.6 → positive → helped WAS (not HOU!) → "Houston's sloppy 80% ball handling gifted Washington extra possessions via turnovers."
  road_oreb_contrib = -15.2 → negative → helped HOU → "Houston dominated the offensive glass at 51% OREB, creating a wave of second-chance opportunities for itself."

EXAMPLES OF INCORRECT INTERPRETATION (DO NOT DO THESE)
  ✗ "Toronto's anemic 12% offensive rebound rate surrendered second-chance chances that New York repeatedly converted."
    WHY WRONG: Low OREB% means Toronto couldn't get its own offensive rebounds. It says NOTHING about New York converting second chances.
    ✓ CORRECT: "Toronto's anemic 12% OREB left them with almost no second-chance opportunities."

  ✗ "New York's 21% offensive-rebound mark oddly favored Toronto by allowing more defensive reset opportunities."
    WHY WRONG: 21% OREB is subpar — it simply means New York couldn't generate second-chance opportunities for itself. There is nothing odd about a subpar stat helping the opponent. Do not use the word "oddly."
    ✓ CORRECT: "New York's shaky 21% OREB limited their second-chance opportunities."

  ✗ "Cleveland's clean handles (89% ball-handling) flipped extra possessions the other way and were a major factor in the home win."
    WHY WRONG: Good ball handling means Cleveland did NOT turn it over — it doesn't "flip" anything. It simply means they protected the ball.
    ✓ CORRECT: "Cleveland's clean 89% ball handling kept possessions secure and denied Detroit extra opportunities."

  ✗ "Milwaukee's 20% OREB couldn't finish possessions defensively, allowing Boston second-chance opportunities."
    WHY WRONG: OREB% is an offensive stat. It means Milwaukee couldn't get its own offensive rebounds. It says nothing about defensive possessions or Boston's second chances.
    ✓ CORRECT: "Milwaukee's 20% OREB meant they generated almost no second-chance opportunities for themselves."

  ✗ "Washington couldn't clean the defensive glass (15% OREB), allowing Houston to extend possessions repeatedly."
    WHY WRONG: OREB% measures offensive rebounding, not defensive. Low OREB% means Washington couldn't rebound its own misses. It says nothing about Houston's possessions.
    ✓ CORRECT: "Washington's dismal 15% OREB left them with almost no second-chance looks."

  ✗ "Washington's inability to get to the line (12% FT rate) hurt them, ceding free-throw opportunities to Houston."
    WHY WRONG: Washington's FT rate only describes Washington's ability to draw fouls. It says nothing about Houston's free-throw opportunities.
    ✓ CORRECT: "Washington's anemic 12% FT rate meant they rarely got to the stripe."

  ✗ "Detroit's elite 57% eFG sliced through Cleveland's defense and was a decisive edge for the visitors."
    WHY WRONG: Detroit lost the game. A factor cannot be a "decisive edge" for a team that lost.
    ✓ CORRECT: "Detroit's elite 57% eFG kept the visitors competitive, but it wasn't enough to overcome Cleveland."

NAMING RULES
- Use CITY names only (never abbreviations). Exception: if both LA teams play, use "Clippers" and "Lakers".

ROUNDING
- Round percentages to the nearest integer (e.g., 52.3 → 52%).
- NEVER invent stats. NEVER say "season-worst", "career-high", "worst performance", or any comparison to other games. You have no season data.

WHICH FACTORS TO INCLUDE
- ONLY factors where abs(contribution) >= 1.0.
- If none qualify, output: "- This game was not driven by any single four-factor edge at the 1-point level."

BULLET ORDER
1) Filter to abs(contribution) >= 1.0.
2) Sort by abs(contribution) descending.
3) Tie-break: home before road, then alphabetical by factor name.
4) Write bullets in exactly this order.

CLASSIFICATION LANGUAGE
- Classifications (POOR/SUBPAR/AVERAGE/GOOD/EXCELLENT) are for your reasoning only. NEVER output these words.
- Translate: POOR → "dismal/frigid/anemic"; SUBPAR → "shaky/inconsistent"; AVERAGE → "solid/steady"; GOOD → "efficient/strong"; EXCELLENT → "dominant/elite".

STYLE
- Write for advanced NBA fans. Crisp, analytical, natural.
- Vary perspective: shooting can be "hot shooting" or "smothered looks"; turnovers can be "clean handles" or "forced mistakes"; OREB can be "crashed the boards" or "couldn't generate second-chance looks"; FT rate can be "lived at the line" or "couldn't get to the stripe".
- Include the stat value in ~60–80% of bullets. Do not repeat adjectives.

BEFORE OUTPUTTING, SILENTLY VERIFY EACH BULLET
1. Does the bullet credit the correct team per the contribution sign?
2. Is the stat value attributed to the team that actually owns it?
3. Does the language match the classification tier?
4. Does the bullet make sense given who won?
5. Are all facts from the input data (nothing invented)?
6. Does the bullet describe ONLY the named team's own performance, without implying anything about the opponent's performance in the same category?
7. If the bullet mentions a losing team's positive factor, does it avoid language like "decisive edge" or "dominant factor"?
8. If the bullet is about OREB%, does it describe offensive rebounding (not defensive)?
9. If the bullet is about ball handling, does it describe turnover avoidance (not "flipping possessions")?
10. If the bullet is about FT rate, does it describe ONLY that team's ability to draw fouls (not the opponent's FT opportunities)?
If any check fails, silently fix the bullet. Do NOT output your verification — output ONLY the corrected bullets.

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
    is_gpt5_family = use_model.startswith("gpt-5")
    token_key = "max_completion_tokens" if is_gpt5_family or use_model.startswith("o") else "max_tokens"
    token_budget = 1500 if is_gpt5_family else 500

    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": use_model,
                token_key: token_budget,
                "messages": [{"role": "user", "content": prompt}]
            }
            if is_gpt5_family:
                payload["reasoning_effort"] = "low"
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
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
    """Check if the interpretation runtime provider is configured."""
    config = _get_llm_config()
    return bool(config["openai_key"])


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
    is_gpt5_family = use_model.startswith("gpt-5")
    token_key = "max_completion_tokens" if is_gpt5_family or use_model.startswith("o") else "max_tokens"
    token_budget = 1500 if is_gpt5_family else 500

    try:
        with httpx.Client() as client:
            payload = {
                "model": use_model,
                token_key: token_budget,
                "messages": [{"role": "user", "content": prompt}]
            }
            if is_gpt5_family:
                payload["reasoning_effort"] = "low"
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
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
            return None
        elif model.startswith("gpt") or model.startswith("o1"):
            if config["openai_key"]:
                return _call_openai_sync(prompt, config["openai_key"], model=model)
            return None

    # Fallback to default provider
    if config["anthropic_key"]:
        return _call_anthropic_sync(prompt, config["anthropic_key"], model=model)
    elif config["openai_key"]:
        return _call_openai_sync(prompt, config["openai_key"], model=model)

    return None
