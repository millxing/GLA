from __future__ import annotations

import csv
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

TRACKED_STATS = [
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "pts",
    "plus_minus",
]

VALIDATION_STATS = [
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "pts",
    "plus_minus",
]

VALIDATION_SOFT_TOLERANCE_BY_STAT = {
    # Official game logs can include late stat corrections that do not always
    # map 1:1 to event-level PBP rows (especially for defensive counting stats).
    "fga": 1,
    "fg3a": 1,
    "oreb": 1,
    "dreb": 1,
    "reb": 1,
    "ast": 1,
    "stl": 1,
    "blk": 1,
    "tov": 1,
    "pf": 1,
}

WINPROB_BASE_COLUMNS = [
    "gameid",
    "home",
    "road",
    "quarter",
    "seconds_left",
    "home_score",
    "road_score",
    "differential",
    "possession",
    "final_score_diff",
]

SHOT_ACTIONS = {"2pt", "3pt", "made shot", "missed shot"}
FREE_THROW_ACTIONS = {"freethrow", "free throw"}
OFFENSIVE_FOUL_WINDOW = 3
OFF_DEF_REB_RE = re.compile(r"\(Off:(\d+)\s+Def:(\d+)\)", re.IGNORECASE)
AST_RE = re.compile(r"\bAST\)")
TEAM_FOUL_RE = re.compile(r"\.T(\d+)\)|T#(\d+)", re.IGNORECASE)
CLOCK_RE = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")
FREE_THROW_OF_RE = re.compile(r"free throw\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
FT_MISS_RE = re.compile(r"\bmiss(?:ed|es)?\b", re.IGNORECASE)
TURNOVER_COUNTER_RE = re.compile(r"T#(\d+)|\.T(\d+)\)", re.IGNORECASE)
BLOCK_COUNTER_RE = re.compile(r"^(.*?)\s+BLOCK\s*\((\d+)\s+BLK\)", re.IGNORECASE)


def _normalize_game_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return s
    return digits.zfill(10)


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", value.upper()).strip()


def _to_optional_int(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _clock_to_seconds_remaining(clock: Any) -> Optional[float]:
    if pd.isna(clock):
        return None
    s = str(clock).strip()
    if not s:
        return None
    m = CLOCK_RE.match(s)
    if not m:
        return None
    minutes = float(m.group(1)) if m.group(1) else 0.0
    seconds = float(m.group(2)) if m.group(2) else 0.0
    return minutes * 60.0 + seconds


def _safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _other_side(side: Optional[str]) -> Optional[str]:
    if side == "home":
        return "road"
    if side == "road":
        return "home"
    return None


def _is_missing_player(value: Any) -> bool:
    if pd.isna(value):
        return True
    s = str(value).strip()
    if not s:
        return True
    return s.lower() in {"nan", "none", "team"}


def _season_start_year(season: str) -> int:
    m = re.match(r"^(\d{4})-\d{2}$", season.strip())
    if not m:
        raise ValueError(f"Invalid season format: {season!r}. Expected YYYY-YY.")
    return int(m.group(1))


def _pct(made: int, attempts: int) -> float:
    if attempts <= 0:
        return 0.0
    return round(made / attempts, 3)


def _normalize_phase(phase: str) -> str:
    p = phase.strip().lower()
    if p not in {"regular", "playoffs"}:
        raise ValueError(f"Invalid phase: {phase!r}. Expected 'regular' or 'playoffs'.")
    return p


def _build_pbp_path(repo_dir: Path, season: str, phase: str, source: str = "auto") -> tuple[Path, str]:
    year = _season_start_year(season)
    if phase == "regular":
        api_name = f"api_pbpv3_{year}.csv"
        nbs_base = f"nbastatsv3_{year}"
    else:
        api_name = f"api_pbpv3_po_{year}.csv"
        nbs_base = f"nbastatsv3_po_{year}"

    api_path = repo_dir / "PBPdata" / "api_pbpv3" / phase / api_name
    nbs_dir = repo_dir / "PBPdata" / "nbastatsv3" / phase
    nbs_paths = [nbs_dir / f"{nbs_base}.parquet", nbs_dir / f"{nbs_base}.csv"]

    source_norm = source.strip().lower()
    if source_norm not in {"auto", "api_pbpv3", "nbastatsv3"}:
        raise ValueError(f"Invalid source: {source!r}. Expected auto|api_pbpv3|nbastatsv3.")

    if source_norm == "api_pbpv3":
        return api_path, "api_pbpv3"
    if source_norm == "nbastatsv3":
        for nbs_path in nbs_paths:
            if nbs_path.exists():
                return nbs_path, "nbastatsv3"
        return nbs_paths[0], "nbastatsv3"

    # Prefer nbastatsv3 for all seasons, with api_pbpv3 as fallback.
    # This keeps a single primary source path when current-season rows are
    # incrementally refreshed from nba_api into nbastatsv3.
    candidates = [(p, "nbastatsv3") for p in nbs_paths] + [(api_path, "api_pbpv3")]

    for path, source in candidates:
        if path.exists():
            return path, source
    return candidates[0]


def _build_output_dir(repo_dir: Path, season: str, phase: str, output_root: Optional[str]) -> Path:
    if output_root:
        root = Path(output_root)
    else:
        root = repo_dir / "PBPdata" / "game_states"
    return root / phase / season


def _resolve_states_input_dir(repo_dir: Path, season: str, phase: str, input_root: Optional[str]) -> Path:
    if input_root:
        root = Path(input_root)
    else:
        root = repo_dir / "PBPdata" / "game_states"

    candidates = [root / phase / season, root / season, root]
    summary_filename = f"_summary_{season}_{phase}.json"
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if (candidate / summary_filename).exists():
            return candidate
        if any(candidate.glob(f"{season}_*_*.json")):
            return candidate
    return candidates[0]


def _build_winprob_output_path(repo_dir: Path, season: str, phase: str, output_root: Optional[str]) -> Path:
    if output_root:
        root = Path(output_root)
    else:
        root = repo_dir / "PBPdata" / "winprob_base"
    return root / phase / f"stacked_{season}_winprob_base.csv"


def _normalize_game_type(value: Any) -> str:
    v = _normalize_text(value)
    if v == "playoff":
        return "playoffs"
    if v == "playin":
        return "play_in"
    return v


def _load_gamelogs(repo_dir: Path, season: str, phase: str) -> pd.DataFrame:
    csv_path = repo_dir / f"team_game_logs_{season}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing game log file: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"game_id": "string"}, low_memory=False)
    if df.empty:
        return df

    d = df.copy()
    d["game_id_norm"] = d["game_id"].map(_normalize_game_id)
    d["game_type_norm"] = d["game_type"].map(_normalize_game_type)
    d["game_date_sort"] = pd.to_datetime(d["game_date"], errors="coerce")

    if phase == "regular":
        d = d[~d["game_type_norm"].isin(["playoffs", "play_in"])].copy()
        # All-Star is not part of the regular-season game-log/PBP workflow.
        d = d[d["game_type_norm"] != "all_star"].copy()
        d = d[~d["game_id_norm"].str.startswith("006", na=False)].copy()
    else:
        d = d[d["game_type_norm"].isin(["playoffs", "play_in"])].copy()

    d = d[d["game_id_norm"] != ""].copy()
    return d.sort_values(["game_date_sort", "game_id_norm"], kind="stable").reset_index(drop=True)


def _load_pbp_df(pbp_path: Path) -> pd.DataFrame:
    if not pbp_path.exists():
        raise FileNotFoundError(f"Missing PBP file: {pbp_path}")

    if pbp_path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(pbp_path)
        except Exception:
            with tempfile.NamedTemporaryFile(prefix="pbp_parquet_bridge_", suffix=".csv", delete=False) as tmp_f:
                tmp_csv = Path(tmp_f.name)
            try:
                script = (
                    "import pandas as pd, sys; "
                    "d = pd.read_parquet(sys.argv[1]); "
                    "d.to_csv(sys.argv[2], index=False)"
                )
                proc = subprocess.run(
                    ["python3", "-c", script, str(pbp_path), str(tmp_csv)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        "Parquet read failed and bridge via python3 was unsuccessful: "
                        f"{proc.stderr.strip() or proc.stdout.strip()}"
                    )
                df = pd.read_csv(tmp_csv, low_memory=False)
            finally:
                try:
                    tmp_csv.unlink(missing_ok=True)
                except Exception:
                    pass
    else:
        df = pd.read_csv(pbp_path, low_memory=False)
    if df.empty:
        return df

    d = df.copy()
    d["game_id_norm"] = d["gameId"].map(_normalize_game_id)
    d["action_type_norm"] = d["actionType"].map(_normalize_text)
    d["sub_type_norm"] = d["subType"].map(_normalize_text)

    for col in ("actionNumber", "actionId", "teamId", "personId", "isFieldGoal", "shotValue", "scoreHome", "scoreAway"):
        if col not in d.columns:
            d[col] = pd.NA
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d[d["game_id_norm"] != ""].copy()
    return d


def _initial_state() -> dict[str, dict[str, int]]:
    return {
        "home": {k: 0 for k in TRACKED_STATS},
        "road": {k: 0 for k in TRACKED_STATS},
    }


def _build_aliases(game_row: pd.Series) -> dict[str, list[str]]:
    home_abbr = _safe_str(game_row.get("team_abbreviation_home"))
    road_abbr = _safe_str(game_row.get("team_abbreviation_road"))
    home_name = _safe_str(game_row.get("team_name_home"))
    road_name = _safe_str(game_row.get("team_name_road"))

    def _name_tokens(name: str) -> set[str]:
        tokens = set()
        tokens.add(_normalize_token(name))
        parts = [p for p in name.split() if p]
        if parts:
            tokens.add(_normalize_token(parts[-1]))
        if len(parts) >= 2:
            tokens.add(_normalize_token(f"{parts[-2]} {parts[-1]}"))
        return tokens

    home_tokens = {_normalize_token(home_abbr)} | _name_tokens(home_name)
    road_tokens = {_normalize_token(road_abbr)} | _name_tokens(road_name)

    home_clean = sorted([t for t in home_tokens if t], key=len, reverse=True)
    road_clean = sorted([t for t in road_tokens if t], key=len, reverse=True)
    return {"home": home_clean, "road": road_clean}


def _resolve_side(event_row: pd.Series, game_row: pd.Series, aliases: dict[str, list[str]]) -> Optional[str]:
    home_abbr = _safe_str(game_row.get("team_abbreviation_home")).upper()
    road_abbr = _safe_str(game_row.get("team_abbreviation_road")).upper()
    home_id = _to_optional_int(game_row.get("team_id_home"))
    road_id = _to_optional_int(game_row.get("team_id_road"))

    tri = _safe_str(event_row.get("teamTricode")).upper().strip()
    if tri == home_abbr:
        return "home"
    if tri == road_abbr:
        return "road"

    tid = _to_optional_int(event_row.get("teamId"))
    if tid is not None and home_id is not None and tid == home_id:
        return "home"
    if tid is not None and road_id is not None and tid == road_id:
        return "road"

    desc = _normalize_token(_safe_str(event_row.get("description")))
    if desc:
        for token in aliases["home"]:
            if re.match(rf"^{re.escape(token)}\b", desc):
                return "home"
        for token in aliases["road"]:
            if re.match(rf"^{re.escape(token)}\b", desc):
                return "road"
    return None


def _add_delta(state: dict[str, dict[str, int]], side: Optional[str], stat: str, delta: int, changed: dict[str, dict[str, int]]) -> None:
    if side not in {"home", "road"}:
        return
    if delta == 0:
        return
    state[side][stat] += delta
    changed[side][stat] = changed[side].get(stat, 0) + delta


def _offensive_foul_turnover(action_type: str, sub_type: str, desc_lower: str) -> bool:
    if action_type != "turnover":
        return False
    return ("offensive foul" in sub_type) or ("offensive foul" in desc_lower)


def _offensive_foul_event(action_type: str, sub_type: str, desc_lower: str) -> bool:
    if action_type != "foul":
        return False
    return ("offensive" in sub_type) or ("offensive foul" in desc_lower) or ("charge offensive" in desc_lower)


def _extract_team_foul_number(desc: str) -> Optional[int]:
    m = TEAM_FOUL_RE.search(desc)
    if not m:
        return None
    for group in m.groups():
        if group is not None:
            return int(group)
    return None


def _counts_toward_pf(sub_type: str, desc_lower: str) -> bool:
    st = sub_type.lower()
    if any(token in st for token in ("technical", "defense 3 second", "flopping")):
        return False

    include_tokens = (
        "shooting",
        "personal",
        "offensive",
        "offensive charge",
        "loose ball",
        "personal take",
        "transition take",
        "away from play",
        "flagrant",
    )
    if any(token in st for token in include_tokens):
        return True

    # Legacy fallbacks where subtype can be blank.
    if any(token in desc_lower for token in ("p.foul", "s.foul", "l.b.foul", "offensive foul")):
        return True

    return False


def _is_fg_event(action_type: str, is_field_goal: Optional[int]) -> bool:
    if action_type in SHOT_ACTIONS:
        return True
    return is_field_goal == 1


def _is_three_point(action_type: str, shot_value: Optional[int], desc_lower: str) -> bool:
    if action_type == "3pt":
        return True
    if shot_value == 3:
        return True
    return "3pt" in desc_lower


def _is_made_shot(action_type: str, shot_result: str) -> bool:
    if action_type == "made shot":
        return True
    return shot_result == "made"


def _is_made_ft(shot_result: str, desc_upper: str) -> bool:
    if shot_result == "made":
        return True
    if shot_result == "missed":
        return False
    if "FREE THROW" in desc_upper and not FT_MISS_RE.search(desc_upper):
        return True
    return False


def _free_throw_trip_numbers(desc_lower: str) -> tuple[Optional[int], Optional[int]]:
    m = FREE_THROW_OF_RE.search(desc_lower)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _is_final_free_throw(desc_lower: str) -> bool:
    attempt, total = _free_throw_trip_numbers(desc_lower)
    if attempt is None or total is None:
        return False
    return attempt == total


def _free_throw_retains_possession(sub_type: str, desc_lower: str) -> bool:
    text = f"{sub_type} {desc_lower}"
    return any(token in text for token in ("technical", "flagrant", "clear path", "away from play"))


def _rebound_possession_kind(
    side: Optional[str],
    sub_type: str,
    desc_lower: str,
    changed: dict[str, dict[str, int]],
) -> Optional[str]:
    if side not in {"home", "road"}:
        return None
    side_changed = changed[side]
    if side_changed.get("dreb", 0) > 0:
        return "defensive_rebound"
    if side_changed.get("oreb", 0) > 0:
        return "offensive_rebound"
    if sub_type == "defensive" or "defensive rebound" in desc_lower:
        return "defensive_rebound"
    if sub_type == "offensive" or "offensive rebound" in desc_lower:
        return "offensive_rebound"
    return None


def _accept_score_update(
    score_known: bool,
    old_home: int,
    old_road: int,
    new_home: int,
    new_road: int,
    action_type: str,
    sub_type: str,
    desc_lower: str,
    out_of_sequence: bool,
) -> bool:
    if not score_known:
        return not (new_home == 0 and new_road == 0)

    old_total = old_home + old_road
    new_total = new_home + new_road

    if new_home == 0 and new_road == 0 and old_total > 0:
        if action_type in {"period", "game"}:
            return False
    if new_total == 0 and old_total > 0:
        return False

    if new_total < old_total:
        if out_of_sequence:
            return False
        # Most downward score updates in raw feeds are stale replay snapshots.
        if action_type in {"instant replay", "period", "game"}:
            # Keep explicit overturn events only.
            if "overturn" not in sub_type and "overturn" not in desc_lower:
                return False
        # For non-replay rows, only allow very small downward corrections.
        if (old_total - new_total) > 4:
            return False

    return True


def _should_swap_score_columns(events_sorted: pd.DataFrame, game_row: pd.Series) -> bool:
    """Detect games where feed scoreHome/scoreAway are reversed.

    Some games (rare) appear with score columns mirrored relative to the game
    log home/road team assignment. We detect only strong, near-certain cases.
    """
    exp_home = _to_optional_int(game_row.get("pts_home"))
    exp_road = _to_optional_int(game_row.get("pts_road"))
    if exp_home is None or exp_road is None:
        return False

    score_home_series = pd.to_numeric(events_sorted.get("scoreHome"), errors="coerce")
    score_away_series = pd.to_numeric(events_sorted.get("scoreAway"), errors="coerce")
    if score_home_series.empty or score_away_series.empty:
        return False

    valid_mask = score_home_series.notna() & score_away_series.notna()
    if not bool(valid_mask.any()):
        return False

    pairs = list(
        zip(
            score_home_series[valid_mask].astype("int64").tolist(),
            score_away_series[valid_mask].astype("int64").tolist(),
        )
    )
    # Use recent score snapshots; late-game scores are most reliable for orientation.
    tail_pairs = pairs[-25:]

    normal_best = min(abs(h - exp_home) + abs(a - exp_road) for h, a in tail_pairs)
    swapped_best = min(abs(a - exp_home) + abs(h - exp_road) for h, a in tail_pairs)

    # Require a strong improvement and a close swapped fit to avoid false positives.
    return swapped_best <= 2 and (normal_best - swapped_best) >= 10


def _live_wl(pts_home: int, pts_road: int) -> str:
    if pts_home > pts_road:
        return "W"
    if pts_home < pts_road:
        return "L"
    return "T"


def _build_game_log_state(game_row: pd.Series, state: dict[str, dict[str, int]], season: str) -> dict[str, Any]:
    h = state["home"]
    r = state["road"]
    return {
        "game_id": _normalize_game_id(game_row.get("game_id")),
        "game_date": _safe_str(game_row.get("game_date")),
        "season": season,
        "game_type": _safe_str(game_row.get("game_type")),
        "neutral_site": bool(game_row.get("neutral_site")),
        "team_id_home": _to_optional_int(game_row.get("team_id_home")) or 0,
        "team_abbreviation_home": _safe_str(game_row.get("team_abbreviation_home")),
        "team_name_home": _safe_str(game_row.get("team_name_home")),
        "team_id_road": _to_optional_int(game_row.get("team_id_road")) or 0,
        "team_abbreviation_road": _safe_str(game_row.get("team_abbreviation_road")),
        "team_name_road": _safe_str(game_row.get("team_name_road")),
        "pts_home": h["pts"],
        "pts_road": r["pts"],
        "wl_home": _live_wl(h["pts"], r["pts"]),
        "fgm_home": h["fgm"],
        "fga_home": h["fga"],
        "fg_pct_home": _pct(h["fgm"], h["fga"]),
        "fg3m_home": h["fg3m"],
        "fg3a_home": h["fg3a"],
        "fg3_pct_home": _pct(h["fg3m"], h["fg3a"]),
        "ftm_home": h["ftm"],
        "fta_home": h["fta"],
        "ft_pct_home": _pct(h["ftm"], h["fta"]),
        "oreb_home": h["oreb"],
        "dreb_home": h["dreb"],
        "reb_home": h["reb"],
        "ast_home": h["ast"],
        "stl_home": h["stl"],
        "blk_home": h["blk"],
        "tov_home": h["tov"],
        "pf_home": h["pf"],
        "plus_minus_home": h["plus_minus"],
        "fgm_road": r["fgm"],
        "fga_road": r["fga"],
        "fg_pct_road": _pct(r["fgm"], r["fga"]),
        "fg3m_road": r["fg3m"],
        "fg3a_road": r["fg3a"],
        "fg3_pct_road": _pct(r["fg3m"], r["fg3a"]),
        "ftm_road": r["ftm"],
        "fta_road": r["fta"],
        "ft_pct_road": _pct(r["ftm"], r["fta"]),
        "oreb_road": r["oreb"],
        "dreb_road": r["dreb"],
        "reb_road": r["reb"],
        "ast_road": r["ast"],
        "stl_road": r["stl"],
        "blk_road": r["blk"],
        "tov_road": r["tov"],
        "pf_road": r["pf"],
        "plus_minus_road": r["plus_minus"],
    }


def _normalize_expected_stat_for_validation(
    game_row: pd.Series,
    stat: str,
    expected_home: int,
    expected_road: int,
    actual_home: int,
    actual_road: int,
) -> tuple[int, int]:
    # Legacy 2000-01 playoff logs contain a known doubled-TOV anomaly for a
    # small subset of games. Normalize expected totals only for validation.
    if stat != "tov":
        return expected_home, expected_road
    if _safe_str(game_row.get("season")) != "2000-01":
        return expected_home, expected_road
    if _normalize_game_type(game_row.get("game_type")) != "playoffs":
        return expected_home, expected_road
    if expected_home <= 0 and expected_road <= 0:
        return expected_home, expected_road
    if expected_home == (2 * actual_home) and expected_road == (2 * actual_road):
        return expected_home // 2, expected_road // 2
    return expected_home, expected_road


def _turnover_counter_totals_from_events(event_rows: list[dict[str, Any]]) -> dict[str, Optional[int]]:
    max_counter: dict[str, Optional[int]] = {"home": None, "road": None}
    for event in event_rows:
        side = event.get("resolved_side")
        if side not in {"home", "road"}:
            continue
        action_type = _normalize_text(event.get("action_type"))
        if action_type != "turnover":
            continue
        desc = _safe_str(event.get("description"))
        m = TURNOVER_COUNTER_RE.search(desc)
        if not m:
            continue
        n = int(m.group(1) or m.group(2))
        prev = max_counter[side]
        max_counter[side] = n if prev is None else max(prev, n)
    return max_counter


def _block_counter_totals_from_events(event_rows: list[dict[str, Any]]) -> dict[str, int]:
    per_side_player_max: dict[str, dict[str, int]] = {"home": {}, "road": {}}
    for event in event_rows:
        side = event.get("resolved_side")
        if side not in {"home", "road"}:
            continue
        desc = _safe_str(event.get("description"))
        m = BLOCK_COUNTER_RE.match(desc)
        if not m:
            continue
        player = _normalize_token(m.group(1))
        if not player:
            continue
        n = int(m.group(2))
        prev = per_side_player_max[side].get(player, 0)
        if n > prev:
            per_side_player_max[side][player] = n
    return {
        "home": int(sum(per_side_player_max["home"].values())),
        "road": int(sum(per_side_player_max["road"].values())),
    }


def _source_consistent_differences(
    strict_differences: dict[str, dict[str, int]],
    final_state: dict[str, dict[str, int]],
    event_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    source_consistent: dict[str, dict[str, int]] = {}
    reasons: dict[str, str] = {}
    if not strict_differences:
        return source_consistent, reasons

    if "tov" in strict_differences:
        counter_tov = _turnover_counter_totals_from_events(event_rows)
        if (
            counter_tov["home"] is not None
            and counter_tov["road"] is not None
            and int(final_state["home"]["tov"]) == int(counter_tov["home"])
            and int(final_state["road"]["tov"]) == int(counter_tov["road"])
        ):
            source_consistent["tov"] = strict_differences["tov"]
            reasons["tov"] = "turnover_counter_t_tag"

    if "blk" in strict_differences:
        counter_blk = _block_counter_totals_from_events(event_rows)
        if (
            int(final_state["home"]["blk"]) == int(counter_blk["home"])
            and int(final_state["road"]["blk"]) == int(counter_blk["road"])
        ):
            source_consistent["blk"] = strict_differences["blk"]
            reasons["blk"] = "block_counter_blk_tag"

    return source_consistent, reasons


def _build_validation(
    game_row: pd.Series,
    final_state: dict[str, dict[str, int]],
    event_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    strict_differences: dict[str, dict[str, int]] = {}
    tolerated_differences: dict[str, dict[str, int]] = {}
    for stat in VALIDATION_STATS:
        got_home = int(final_state["home"][stat])
        got_road = int(final_state["road"][stat])
        expected_home_raw = int(game_row[f"{stat}_home"])
        expected_road_raw = int(game_row[f"{stat}_road"])
        expected_home, expected_road = _normalize_expected_stat_for_validation(
            game_row=game_row,
            stat=stat,
            expected_home=expected_home_raw,
            expected_road=expected_road_raw,
            actual_home=got_home,
            actual_road=got_road,
        )
        if expected_home != got_home or expected_road != got_road:
            diff = {
                "expected_home": expected_home,
                "expected_road": expected_road,
                "actual_home": got_home,
                "actual_road": got_road,
                "delta_home": got_home - expected_home,
                "delta_road": got_road - expected_road,
            }
            tol = VALIDATION_SOFT_TOLERANCE_BY_STAT.get(stat)
            if tol is not None and abs(diff["delta_home"]) <= tol and abs(diff["delta_road"]) <= tol:
                tolerated_differences[stat] = diff
            else:
                strict_differences[stat] = diff

    source_consistent_diffs, source_consistency_reasons = _source_consistent_differences(
        strict_differences=strict_differences,
        final_state=final_state,
        event_rows=event_rows,
    )
    parser_differences = {
        stat: diff
        for stat, diff in strict_differences.items()
        if stat not in source_consistent_diffs
    }

    return {
        "strict_match": not (strict_differences or tolerated_differences),
        "match": not strict_differences,
        "differences": strict_differences,
        "tolerated_differences": tolerated_differences,
        "source_consistent_differences": source_consistent_diffs,
        "source_consistency_reasons": source_consistency_reasons,
        "parser_differences": parser_differences,
        "source_consistent_mismatch": bool(source_consistent_diffs) and not parser_differences,
    }


def _event_payload(
    row: pd.Series,
    event_index: int,
    side: Optional[str],
    changed: dict[str, dict[str, int]],
    state_line: dict[str, Any],
    possession_before: Optional[str],
    possession_after: Optional[str],
    possession_changed: bool,
    possession_reason: Optional[str],
    possession_confidence: Optional[str],
    score_home_aligned: Optional[int] = None,
    score_away_aligned: Optional[int] = None,
    score_columns_swapped: bool = False,
) -> dict[str, Any]:
    action_number = _to_optional_int(row.get("actionNumber"))
    action_id = _to_optional_int(row.get("actionId"))
    period = _to_optional_int(row.get("period"))
    team_id = _to_optional_int(row.get("teamId"))
    person_id = _to_optional_int(row.get("personId"))
    is_field_goal = _to_optional_int(row.get("isFieldGoal"))
    score_home_raw = _to_optional_int(row.get("scoreHome"))
    score_away_raw = _to_optional_int(row.get("scoreAway"))
    score_home = score_home_aligned if score_home_aligned is not None else score_home_raw
    score_away = score_away_aligned if score_away_aligned is not None else score_away_raw
    points_total = _to_optional_int(row.get("pointsTotal"))
    shot_value = _to_optional_int(row.get("shotValue"))
    possession_team_id: Optional[int] = None
    possession_team_tricode: str = ""
    if possession_after == "home":
        possession_team_id = int(state_line.get("team_id_home") or 0)
        possession_team_tricode = _safe_str(state_line.get("team_abbreviation_home"))
    elif possession_after == "road":
        possession_team_id = int(state_line.get("team_id_road") or 0)
        possession_team_tricode = _safe_str(state_line.get("team_abbreviation_road"))

    return {
        "event_index": event_index,
        "action_number": action_number,
        "action_id": action_id,
        "period": period,
        "clock": _safe_str(row.get("clock")),
        "team_id": team_id,
        "team_tricode": _safe_str(row.get("teamTricode")),
        "person_id": person_id,
        "player_name": _safe_str(row.get("playerName")),
        "description": _safe_str(row.get("description")),
        "action_type": _safe_str(row.get("actionType")),
        "sub_type": _safe_str(row.get("subType")),
        "shot_result": _safe_str(row.get("shotResult")),
        "is_field_goal": is_field_goal,
        "shot_value": shot_value,
        "score_home": score_home,
        "score_away": score_away,
        "score_home_raw": score_home_raw,
        "score_away_raw": score_away_raw,
        "score_columns_swapped": score_columns_swapped,
        "points_total": points_total,
        "resolved_side": side,
        "possession_before_side": possession_before,
        "possession_after_side": possession_after,
        "possession_changed": possession_changed,
        "possession_change_reason": possession_reason,
        "possession_confidence": possession_confidence,
        "possession_team_id": possession_team_id,
        "possession_team_tricode": possession_team_tricode,
        "changed_stats": {
            "home": changed["home"],
            "road": changed["road"],
        },
        "game_log_state": state_line,
    }


def _build_game_payload(
    season: str,
    phase: str,
    game_row: pd.Series,
    events_df: pd.DataFrame,
    pbp_file: Path,
    pbp_source: str,
) -> dict[str, Any]:
    events_sorted = events_df.sort_values(["actionNumber", "actionId"], kind="stable").reset_index(drop=True)
    period_series = pd.to_numeric(events_sorted.get("period"), errors="coerce")
    max_period_in_game = int(period_series.max()) if not period_series.empty and pd.notna(period_series.max()) else 0
    score_columns_swapped = _should_swap_score_columns(events_sorted, game_row)
    aliases = _build_aliases(game_row)
    state = _initial_state()

    rebound_prev: dict[tuple[str, str], dict[str, int]] = {}
    offensive_foul_nums: dict[str, list[int]] = {"home": [], "road": []}
    offensive_foul_team_nums: dict[str, set[int]] = {"home": set(), "road": set()}
    excess_timeout_side_by_key: dict[tuple[int, str], str] = {}

    # Pre-index offensive foul rows for turnover->PF fallback.
    for _, row in events_sorted.iterrows():
        side = _resolve_side(row, game_row, aliases)
        if side is None:
            continue
        action_type = _normalize_text(row.get("action_type_norm", row.get("actionType")))
        sub_type = _normalize_text(row.get("sub_type_norm", row.get("subType")))
        desc_lower = _safe_str(row.get("description")).lower()
        if _offensive_foul_event(action_type, sub_type, desc_lower):
            action_number = _to_optional_int(row.get("actionNumber"))
            if action_number is not None:
                offensive_foul_nums[side].append(action_number)
            tf_num = _extract_team_foul_number(desc_lower)
            if tf_num is not None:
                offensive_foul_team_nums[side].add(tf_num)

        if action_type == "foul" and "excess timeout technical" in sub_type:
            period = _to_optional_int(row.get("period")) or 0
            clock = _safe_str(row.get("clock"))
            excess_timeout_side_by_key[(period, clock)] = side

    score_known = False
    score_home = 0
    score_away = 0
    possession_side: Optional[str] = None
    period_min_clock: dict[int, float] = {}
    max_period_seen = 0

    event_rows: list[dict[str, Any]] = []

    for idx, row in events_sorted.iterrows():
        action_type = _normalize_text(row.get("action_type_norm", row.get("actionType")))
        sub_type = _normalize_text(row.get("sub_type_norm", row.get("subType")))
        desc = _safe_str(row.get("description"))
        desc_lower = desc.lower()
        desc_upper = desc.upper()
        shot_result = _normalize_text(row.get("shotResult"))
        side = _resolve_side(row, game_row, aliases)
        changed: dict[str, dict[str, int]] = {"home": {}, "road": {}}

        action_number = _to_optional_int(row.get("actionNumber")) or (idx + 1)
        is_field_goal = _to_optional_int(row.get("isFieldGoal"))
        shot_value = _to_optional_int(row.get("shotValue"))
        period = _to_optional_int(row.get("period")) or 0
        clock = _safe_str(row.get("clock"))

        # Legacy excess-timeout turnovers can omit team fields; infer via paired technical.
        if side is None and action_type == "turnover" and "excess timeout turnover" in sub_type:
            side = excess_timeout_side_by_key.get((period, clock))

        fg_event = _is_fg_event(action_type, is_field_goal) and side is not None
        fg_made = fg_event and _is_made_shot(action_type, shot_result)
        ft_event = action_type in FREE_THROW_ACTIONS and side is not None
        ft_made = ft_event and _is_made_ft(shot_result, desc_upper)
        ft_final = ft_event and _is_final_free_throw(desc_lower)
        ft_retains = ft_event and _free_throw_retains_possession(sub_type, desc_lower)
        turnover_event = action_type == "turnover" and side is not None

        clock_sec = _clock_to_seconds_remaining(clock)
        out_of_sequence = False
        if period > 0 and clock_sec is not None:
            min_clock = period_min_clock.get(period)
            if min_clock is not None and clock_sec > (min_clock + 1e-6):
                out_of_sequence = True
            if period < max_period_seen:
                out_of_sequence = True
            if min_clock is None:
                period_min_clock[period] = clock_sec
            else:
                period_min_clock[period] = min(min_clock, clock_sec)
            max_period_seen = max(max_period_seen, period)

        # Field goals
        if fg_event:
            _add_delta(state, side, "fga", 1, changed)
            is_three = _is_three_point(action_type, shot_value, desc_lower)
            if is_three:
                _add_delta(state, side, "fg3a", 1, changed)

            if fg_made:
                _add_delta(state, side, "fgm", 1, changed)
                if is_three:
                    _add_delta(state, side, "fg3m", 1, changed)
                if AST_RE.search(desc):
                    _add_delta(state, side, "ast", 1, changed)
                if not score_known:
                    _add_delta(state, side, "pts", 3 if is_three else 2, changed)

        # Free throws
        if ft_event:
            _add_delta(state, side, "fta", 1, changed)
            if ft_made:
                _add_delta(state, side, "ftm", 1, changed)
                if not score_known:
                    _add_delta(state, side, "pts", 1, changed)

        # Rebounds (exclude team/dead-ball style rebounds)
        if action_type == "rebound" and side is not None:
            player_name = row.get("playerName")
            if not _is_missing_player(player_name):
                if sub_type == "offensive":
                    _add_delta(state, side, "oreb", 1, changed)
                    _add_delta(state, side, "reb", 1, changed)
                elif sub_type == "defensive":
                    _add_delta(state, side, "dreb", 1, changed)
                    _add_delta(state, side, "reb", 1, changed)
                else:
                    m = OFF_DEF_REB_RE.search(desc)
                    if m:
                        off_total = int(m.group(1))
                        def_total = int(m.group(2))
                        person_id = _to_optional_int(row.get("personId"))
                        player_key = str(person_id) if person_id and person_id > 0 else _safe_str(player_name).strip()
                        reb_key = (side, player_key)
                        prev = rebound_prev.get(reb_key, {"off": 0, "def": 0})
                        off_delta = max(0, off_total - prev["off"])
                        def_delta = max(0, def_total - prev["def"])
                        if off_delta:
                            _add_delta(state, side, "oreb", off_delta, changed)
                            _add_delta(state, side, "reb", off_delta, changed)
                        if def_delta:
                            _add_delta(state, side, "dreb", def_delta, changed)
                            _add_delta(state, side, "reb", def_delta, changed)
                        rebound_prev[reb_key] = {
                            "off": max(prev["off"], off_total),
                            "def": max(prev["def"], def_total),
                        }
                    elif "offensive rebound" in desc_lower:
                        _add_delta(state, side, "oreb", 1, changed)
                        _add_delta(state, side, "reb", 1, changed)
                    elif "defensive rebound" in desc_lower:
                        _add_delta(state, side, "dreb", 1, changed)
                        _add_delta(state, side, "reb", 1, changed)

        # Turnovers
        if turnover_event:
            _add_delta(state, side, "tov", 1, changed)
            if pbp_source == "api_pbpv3" and _offensive_foul_turnover(action_type, sub_type, desc_lower):
                nearby = any(abs(action_number - n) <= OFFENSIVE_FOUL_WINDOW for n in offensive_foul_nums[side])
                tf_num = _extract_team_foul_number(desc_lower)
                if tf_num is not None and tf_num in offensive_foul_team_nums[side]:
                    nearby = True
                if not nearby:
                    _add_delta(state, side, "pf", 1, changed)

        # Fouls
        if action_type == "foul" and side is not None and _counts_toward_pf(sub_type, desc_lower):
            if "double personal" in sub_type or "double personal" in desc_lower:
                _add_delta(state, "home", "pf", 1, changed)
                _add_delta(state, "road", "pf", 1, changed)
            else:
                _add_delta(state, side, "pf", 1, changed)

        # Steals/blocks (legacy blank-action fallback included)
        if side is not None:
            if action_type == "steal" or (not action_type and "STEAL" in desc_upper):
                _add_delta(state, side, "stl", 1, changed)
            if action_type == "block" or (not action_type and "BLOCK" in desc_upper):
                _add_delta(state, side, "blk", 1, changed)

        # Scoreboard alignment for points
        new_score_home = _to_optional_int(row.get("scoreHome"))
        new_score_away = _to_optional_int(row.get("scoreAway"))
        if score_columns_swapped:
            new_score_home, new_score_away = new_score_away, new_score_home
        if new_score_home is not None and new_score_away is not None:
            old_pts_home = state["home"]["pts"]
            old_pts_road = state["road"]["pts"]
            if _accept_score_update(
                score_known,
                score_home,
                score_away,
                new_score_home,
                new_score_away,
                action_type,
                sub_type,
                desc_lower,
                out_of_sequence,
            ):
                score_known = True
                score_home = new_score_home
                score_away = new_score_away

                home_delta = score_home - old_pts_home
                road_delta = score_away - old_pts_road
                if home_delta:
                    _add_delta(state, "home", "pts", home_delta, changed)
                if road_delta:
                    _add_delta(state, "road", "pts", road_delta, changed)

        # Keep plus/minus as derived from score.
        pm_home_new = state["home"]["pts"] - state["road"]["pts"]
        pm_home_delta = pm_home_new - state["home"]["plus_minus"]
        pm_road_new = -pm_home_new
        pm_road_delta = pm_road_new - state["road"]["plus_minus"]
        if pm_home_delta:
            _add_delta(state, "home", "plus_minus", pm_home_delta, changed)
        if pm_road_delta:
            _add_delta(state, "road", "plus_minus", pm_road_delta, changed)

        possession_before = possession_side
        possession_after = possession_side
        possession_reason: Optional[str] = None
        possession_confidence: Optional[str] = None
        rebound_kind: Optional[str] = None

        if action_type == "rebound":
            rebound_kind = _rebound_possession_kind(side, sub_type, desc_lower, changed)

        period_start = action_type == "period" and "start of" in desc_lower
        period_end = action_type == "period" and "end of" in desc_lower

        if period_start:
            if period == 1:
                # Opening tip has not occurred yet.
                possession_after = None
                possession_reason = "period_start_opening_tip_pending"
                possession_confidence = "high"
            elif period > 4:
                # Every OT starts with a jump ball.
                possession_after = None
                possession_reason = "period_start_overtime_tip_pending"
                possession_confidence = "high"
            else:
                # Q2-Q4 retain possession from prior period end.
                possession_after = possession_side
                possession_reason = "period_start_carry"
                possession_confidence = "high" if possession_after is not None else "unknown"
        elif period_end:
            next_period = period + 1 if period > 0 else 0
            overtime_next = next_period > 4 and next_period <= max_period_in_game
            if overtime_next:
                possession_after = None
                possession_reason = "period_end_overtime_tip_pending"
                possession_confidence = "high"
            elif period < 4:
                # End Q1-Q3 carries into next regulation quarter.
                possession_after = possession_side
                possession_reason = "period_end_carry"
                possession_confidence = "high" if possession_after is not None else "unknown"
            else:
                # Final regulation/OT end: no next possession state.
                possession_after = None
                possession_reason = "period_end_terminal"
                possession_confidence = "high"
        elif turnover_event:
            possession_after = _other_side(side)
            possession_reason = "turnover"
            possession_confidence = "high"
        elif rebound_kind == "defensive_rebound" and side is not None:
            possession_after = side
            possession_reason = "defensive_rebound"
            possession_confidence = "high"
        elif rebound_kind == "offensive_rebound" and side is not None:
            possession_after = side
            possession_reason = "offensive_rebound"
            possession_confidence = "high"
        elif fg_made:
            possession_after = _other_side(side)
            possession_reason = "made_field_goal"
            possession_confidence = "high"
        elif fg_event and side is not None:
            possession_after = side
            possession_reason = "field_goal_attempt"
            possession_confidence = "medium"
        elif ft_event and side is not None:
            if ft_made and ft_final and not ft_retains:
                possession_after = _other_side(side)
                possession_reason = "made_final_free_throw"
                possession_confidence = "medium"
            else:
                possession_after = side
                possession_reason = "free_throw_sequence"
                possession_confidence = "medium"
        elif action_type == "violation" and side is not None and ("jump ball" in sub_type or "jump ball" in desc_lower):
            possession_after = _other_side(side)
            possession_reason = "jump_ball_violation"
            possession_confidence = "medium"

        if possession_reason is None:
            possession_reason = "carry_forward"
            possession_confidence = "high" if possession_after is not None else "unknown"

        possession_changed = possession_after != possession_before
        possession_side = possession_after

        state_line = _build_game_log_state(game_row, state, season)
        event_rows.append(
            _event_payload(
                row,
                idx + 1,
                side,
                changed,
                state_line,
                possession_before=possession_before,
                possession_after=possession_after,
                possession_changed=possession_changed,
                possession_reason=possession_reason,
                possession_confidence=possession_confidence,
                score_home_aligned=new_score_home,
                score_away_aligned=new_score_away,
                score_columns_swapped=score_columns_swapped,
            )
        )

    validation = _build_validation(game_row, state, event_rows)

    expected_totals = {}
    for stat in VALIDATION_STATS:
        expected_totals[f"{stat}_home"] = int(game_row[f"{stat}_home"])
        expected_totals[f"{stat}_road"] = int(game_row[f"{stat}_road"])

    return {
        "season": season,
        "phase": phase,
        "game_id": _normalize_game_id(game_row.get("game_id")),
        "game_date": _safe_str(game_row.get("game_date")),
        "game_type": _safe_str(game_row.get("game_type")),
        "home_team": _safe_str(game_row.get("team_abbreviation_home")),
        "road_team": _safe_str(game_row.get("team_abbreviation_road")),
        "home_team_id": _to_optional_int(game_row.get("team_id_home")) or 0,
        "road_team_id": _to_optional_int(game_row.get("team_id_road")) or 0,
        "score_columns_swapped": score_columns_swapped,
        "pbp_source_file": str(pbp_file),
        "events": event_rows,
        "final_state": _build_game_log_state(game_row, state, season),
        "expected_totals": expected_totals,
        "validation": validation,
    }


def build_pbp_game_states(
    season: str,
    repo_dir: Path,
    phase: str = "regular",
    source: str = "auto",
    output_root: Optional[str] = None,
    max_games: Optional[int] = None,
    game_id: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    phase_norm = _normalize_phase(phase)
    pbp_path, pbp_source = _build_pbp_path(repo_dir, season, phase_norm, source=source)
    output_dir = _build_output_dir(repo_dir, season, phase_norm, output_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    game_logs = _load_gamelogs(repo_dir, season, phase_norm)
    if game_logs.empty:
        print(f"[pbp-state] No games found in game logs for {season} ({phase_norm})")
        return 1

    if game_id:
        gid_filter = _normalize_game_id(game_id)
        game_logs = game_logs[game_logs["game_id_norm"] == gid_filter].copy()
        if game_logs.empty:
            print(f"[pbp-state] Game not found in game logs: {game_id}")
            return 1

    pbp_df = _load_pbp_df(pbp_path)
    if pbp_df.empty:
        print(f"[pbp-state] No rows found in {pbp_path}")
        return 1

    pbp_grouped = pbp_df.groupby("game_id_norm", sort=False)
    pbp_game_ids = set(pbp_grouped.indices.keys())

    print(f"[pbp-state] Season={season} phase={phase_norm}")
    print(f"[pbp-state] PBP source={pbp_path} ({pbp_source})")
    print(f"[pbp-state] Output dir={output_dir}")
    print(f"[pbp-state] Candidate games={len(game_logs)}")

    processed = 0
    matched = 0
    mismatched = 0
    source_consistent_mismatched = 0
    parser_mismatched = 0
    mixed_mismatched = 0
    missing_pbp = 0
    pending_pbp = 0
    skipped_existing = 0
    mismatch_records: list[dict[str, Any]] = []
    missing_pbp_game_ids: list[str] = []
    pending_pbp_game_ids: list[str] = []
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    for _, game_row in game_logs.iterrows():
        if max_games is not None and processed >= max_games:
            break

        gid = game_row["game_id_norm"]
        home = _safe_str(game_row.get("team_abbreviation_home"))
        road = _safe_str(game_row.get("team_abbreviation_road"))
        out_path = output_dir / f"{season}_{home}_{road}_{gid}.json"

        if out_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        if gid not in pbp_game_ids:
            game_date_sort = game_row.get("game_date_sort")
            game_date_norm: Optional[pd.Timestamp] = None
            if pd.notna(game_date_sort):
                ts = pd.to_datetime(game_date_sort, errors="coerce")
                if pd.notna(ts):
                    game_date_norm = ts.tz_localize(None).normalize() if getattr(ts, "tzinfo", None) is not None else ts.normalize()

            # Current-day (or future) games can legitimately be absent from raw PBP.
            if game_date_norm is not None and game_date_norm >= today:
                pending_pbp += 1
                pending_pbp_game_ids.append(gid)
            else:
                missing_pbp += 1
                missing_pbp_game_ids.append(gid)
            continue

        events_df = pbp_grouped.get_group(gid)
        payload = _build_game_payload(
            season=season,
            phase=phase_norm,
            game_row=game_row,
            events_df=events_df,
            pbp_file=pbp_path,
            pbp_source=pbp_source,
        )

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)

        processed += 1
        if payload["validation"]["match"]:
            matched += 1
        else:
            mismatched += 1
            source_diffs = payload["validation"].get("source_consistent_differences", {})
            parser_diffs = payload["validation"].get("parser_differences", {})
            if source_diffs and not parser_diffs:
                source_consistent_mismatched += 1
            elif source_diffs and parser_diffs:
                mixed_mismatched += 1
            if parser_diffs:
                parser_mismatched += 1
            mismatch_records.append(
                {
                    "game_id": gid,
                    "home_team": home,
                    "road_team": road,
                    "differences": payload["validation"]["differences"],
                    "parser_differences": parser_diffs,
                    "source_consistent_differences": source_diffs,
                    "source_consistency_reasons": payload["validation"].get("source_consistency_reasons", {}),
                    "source_consistent_mismatch": bool(payload["validation"].get("source_consistent_mismatch")),
                    "output_file": str(out_path),
                }
            )

        if processed % 50 == 0:
            print(
                f"[pbp-state] Processed={processed} "
                f"matched={matched} mismatched={mismatched} "
                f"missing_pbp={missing_pbp} pending_pbp={pending_pbp}"
            )

    summary = {
        "season": season,
        "phase": phase_norm,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pbp_source_file": str(pbp_path),
        "pbp_source": pbp_source,
        "output_dir": str(output_dir),
        "candidate_games": int(len(game_logs)),
        "processed_games": processed,
        "skipped_existing_games": skipped_existing,
        "matched_games": matched,
        "mismatched_games": mismatched,
        "source_consistent_mismatched_games": source_consistent_mismatched,
        "parser_mismatched_games": parser_mismatched,
        "mixed_mismatched_games": mixed_mismatched,
        "missing_pbp_games": missing_pbp,
        "missing_pbp_game_ids": missing_pbp_game_ids,
        "pending_pbp_games": pending_pbp,
        "pending_pbp_game_ids": pending_pbp_game_ids,
        "mismatches": mismatch_records,
    }

    summary_path = output_dir / f"_summary_{season}_{phase_norm}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(
        f"[pbp-state] Done. processed={processed} matched={matched} "
        f"mismatched={mismatched} source_consistent_mismatched={source_consistent_mismatched} "
        f"parser_mismatched={parser_mismatched} mixed_mismatched={mixed_mismatched} missing_pbp={missing_pbp} "
        f"pending_pbp={pending_pbp} skipped={skipped_existing}"
    )
    print(f"[pbp-state] Summary: {summary_path}")
    return 0 if mismatched == 0 else 2


def build_winprob_base(
    season: str,
    repo_dir: Path,
    phase: str = "regular",
    input_root: Optional[str] = None,
    output_root: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    phase_norm = _normalize_phase(phase)
    input_dir = _resolve_states_input_dir(repo_dir, season, phase_norm, input_root=input_root)
    if not input_dir.exists():
        print(f"[pbp-winprob] Missing game-state input directory: {input_dir}")
        return 1

    output_path = _build_winprob_output_path(repo_dir, season, phase_norm, output_root=output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"[pbp-winprob] Output already exists (use --overwrite): {output_path}")
        return 0

    json_files = sorted(
        p
        for p in input_dir.glob(f"{season}_*_*.json")
        if not p.name.startswith("_summary_")
    )
    if not json_files:
        print(f"[pbp-winprob] No game-state JSON files found in: {input_dir}")
        return 1

    games_processed = 0
    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WINPROB_BASE_COLUMNS)
        writer.writeheader()

        for game_file in json_files:
            with game_file.open("r", encoding="utf-8") as src:
                payload = json.load(src)

            final_state = payload.get("final_state") or {}
            home_team = _safe_str(payload.get("home_team") or final_state.get("team_abbreviation_home"))
            road_team = _safe_str(payload.get("road_team") or final_state.get("team_abbreviation_road"))
            final_score_diff = _to_optional_int(final_state.get("plus_minus_home"))
            if final_score_diff is None:
                final_home = _to_optional_int(final_state.get("pts_home")) or 0
                final_road = _to_optional_int(final_state.get("pts_road")) or 0
                final_score_diff = final_home - final_road

            game_id_norm = _normalize_game_id(payload.get("game_id"))
            try:
                gameid_value: int | str = int(game_id_norm)
            except Exception:
                gameid_value = game_id_norm

            events = payload.get("events") or []
            for event in events:
                period = _to_optional_int(event.get("period"))
                if period is None:
                    continue
                seconds_float = _clock_to_seconds_remaining(event.get("clock"))
                if seconds_float is None:
                    continue
                seconds_left = max(0, int(round(seconds_float)))

                state = event.get("game_log_state") or {}
                home_score = _to_optional_int(event.get("score_home"))
                road_score = _to_optional_int(event.get("score_away"))
                if home_score is None:
                    home_score = _to_optional_int(state.get("pts_home"))
                if road_score is None:
                    road_score = _to_optional_int(state.get("pts_road"))
                if home_score is None:
                    home_score = 0
                if road_score is None:
                    road_score = 0

                possession = _safe_str(event.get("possession_team_tricode"))
                if not possession:
                    possession_after = _safe_str(event.get("possession_after_side"))
                    if possession_after == "home":
                        possession = home_team
                    elif possession_after == "road":
                        possession = road_team

                writer.writerow(
                    {
                        "gameid": gameid_value,
                        "home": home_team,
                        "road": road_team,
                        "quarter": period,
                        "seconds_left": seconds_left,
                        "home_score": home_score,
                        "road_score": road_score,
                        "differential": home_score - road_score,
                        "possession": possession,
                        "final_score_diff": final_score_diff,
                    }
                )
                rows_written += 1

            games_processed += 1
            if games_processed % 100 == 0:
                print(f"[pbp-winprob] Processed games={games_processed} rows={rows_written}")

    print(
        f"[pbp-winprob] Wrote {rows_written} rows from {games_processed} games "
        f"to {output_path}"
    )
    return 0
