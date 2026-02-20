"""
Test script for NBA Stats API daily update endpoints.
Tests LeagueGameLog, BoxScoreSummaryV3, and BoxScoreAdvancedV3 for 2026-02-19.
"""

import requests
import json
import sys
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_DATE = "2026-02-19"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}
BASE_URL = "https://stats.nba.com/stats"
TIMEOUT = 30

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = {}

# ── Helpers ──────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_response(name: str, resp: requests.Response) -> dict:
    """Validate a raw response and return a result dict."""
    result = {
        "endpoint": name,
        "status_code": resp.status_code,
        "ok": False,
        "row_count": None,
        "error": None,
    }

    if resp.status_code != 200:
        result["error"] = f"HTTP {resp.status_code}"
        return result

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        result["error"] = f"JSON decode error: {e}"
        return result

    # NBA Stats API wraps data in resultSets or resultSet
    result_sets = data.get("resultSets") or data.get("resultSet") or []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]

    if not result_sets:
        result["error"] = "No resultSets found in response"
        return result

    total_rows = sum(len(rs.get("rowSet", [])) for rs in result_sets)
    result["row_count"] = total_rows
    result["ok"] = True
    return result


def print_result(result: dict):
    status = PASS if result["ok"] else FAIL
    print(f"\n  {status}  {result['endpoint']}")
    print(f"        HTTP Status : {result['status_code']}")
    if result["ok"]:
        print(f"        Rows returned: {result['row_count']}")
    else:
        print(f"        Error        : {result['error']}")


# ── 1. LeagueGameLog ─────────────────────────────────────────────────────────

print_section("1 — LeagueGameLog")

params = {
    "Counter": 0,
    "DateFrom": TARGET_DATE,
    "DateTo": TARGET_DATE,
    "Direction": "DESC",
    "LeagueID": "00",
    "PlayerOrTeam": "T",   # Team-level logs; change to "P" for player-level
    "Season": "2025-26",
    "SeasonType": "Regular Season",
    "Sorter": "DATE",
}

try:
    resp = requests.get(f"{BASE_URL}/leaguegamelog", headers=HEADERS, params=params, timeout=TIMEOUT)
    result = check_response("LeagueGameLog", resp)
except requests.RequestException as e:
    result = {"endpoint": "LeagueGameLog", "status_code": None, "ok": False, "row_count": None, "error": str(e)}

print_result(result)
results["LeagueGameLog"] = result


# ── Fetch game IDs for box score tests ──────────────────────────────────────

game_ids = []
if result["ok"]:
    data = resp.json()
    rs = data["resultSets"][0]
    headers_list = rs["headers"]
    rows = rs["rowSet"]
    try:
        gid_idx = headers_list.index("GAME_ID")
        game_ids = list({row[gid_idx] for row in rows})
        print(f"\n  Found {len(game_ids)} game(s) on {TARGET_DATE}: {game_ids}")
    except ValueError:
        print("  WARNING: Could not locate GAME_ID column — skipping box score tests.")
else:
    print(f"\n  WARNING: LeagueGameLog failed — will attempt box score tests with a placeholder game ID.")
    game_ids = []  # Box score tests will be skipped or use a dummy id


# ── 2. BoxScoreSummaryV3 ─────────────────────────────────────────────────────

print_section("2 — BoxScoreSummaryV3")

bs_summary_results = []

if not game_ids:
    print("  ⚠️  No game IDs available — skipping BoxScoreSummaryV3 test.")
    results["BoxScoreSummaryV3"] = {"endpoint": "BoxScoreSummaryV3", "ok": None, "error": "No game IDs"}
else:
    for game_id in game_ids:
        params = {"GameID": game_id, "LeagueID": "00"}
        try:
            resp = requests.get(
                f"{BASE_URL}/boxscoresummaryv3",
                headers=HEADERS,
                params=params,
                timeout=TIMEOUT,
            )
            result = check_response(f"BoxScoreSummaryV3 [{game_id}]", resp)
        except requests.RequestException as e:
            result = {
                "endpoint": f"BoxScoreSummaryV3 [{game_id}]",
                "status_code": None,
                "ok": False,
                "row_count": None,
                "error": str(e),
            }
        print_result(result)
        bs_summary_results.append(result)

    results["BoxScoreSummaryV3"] = bs_summary_results


# ── 3. BoxScoreAdvancedV3 ────────────────────────────────────────────────────

print_section("3 — BoxScoreAdvancedV3")

bs_advanced_results = []

if not game_ids:
    print("  ⚠️  No game IDs available — skipping BoxScoreAdvancedV3 test.")
    results["BoxScoreAdvancedV3"] = {"endpoint": "BoxScoreAdvancedV3", "ok": None, "error": "No game IDs"}
else:
    for game_id in game_ids:
        params = {
            "GameID": game_id,
            "LeagueID": "00",
            "StartPeriod": 0,
            "EndPeriod": 0,
            "StartRange": 0,
            "EndRange": 0,
            "RangeType": 0,
        }
        try:
            resp = requests.get(
                f"{BASE_URL}/boxscoreadvancedv3",
                headers=HEADERS,
                params=params,
                timeout=TIMEOUT,
            )
            result = check_response(f"BoxScoreAdvancedV3 [{game_id}]", resp)
        except requests.RequestException as e:
            result = {
                "endpoint": f"BoxScoreAdvancedV3 [{game_id}]",
                "status_code": None,
                "ok": False,
                "row_count": None,
                "error": str(e),
            }
        print_result(result)
        bs_advanced_results.append(result)

    results["BoxScoreAdvancedV3"] = bs_advanced_results


# ── Summary ──────────────────────────────────────────────────────────────────

print_section("SUMMARY")

def flatten(v):
    if isinstance(v, list):
        return v
    return [v]

all_results = []
for v in results.values():
    all_results.extend(flatten(v))

passed = [r for r in all_results if r.get("ok") is True]
failed = [r for r in all_results if r.get("ok") is False]
skipped = [r for r in all_results if r.get("ok") is None]

print(f"\n  Date tested : {TARGET_DATE}")
print(f"  Passed      : {len(passed)}")
print(f"  Failed      : {len(failed)}")
print(f"  Skipped     : {len(skipped)}")

if failed:
    print("\n  Failed endpoints:")
    for r in failed:
        print(f"    • {r['endpoint']} — {r.get('error', 'unknown error')}")

print()
sys.exit(1 if failed else 0)
