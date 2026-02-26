#!/usr/bin/env python3
"""Build a per-season game index for pbp_analyze.html.

This index lets the frontend list games without relying on directory listing.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, UTC
from pathlib import Path


def build_index(states_root: Path, season: str, phase: str) -> int:
    season_dir = states_root / phase / season
    if not season_dir.exists() or not season_dir.is_dir():
        print(f"[pbp-index] Missing season directory: {season_dir}")
        return 1

    rx = re.compile(rf"^{re.escape(season)}_([A-Z0-9]+)_([A-Z0-9]+)_(\d{{10}})\.json$")
    games: list[dict[str, str]] = []

    for game_path in sorted(season_dir.glob(f"{season}_*_*.json")):
        if game_path.name.startswith("_"):
            continue

        m = rx.match(game_path.name)
        if not m:
            continue

        game_date = season
        try:
            payload = json.loads(game_path.read_text(encoding="utf-8"))
            parsed_date = str(payload.get("game_date") or "").strip()
            if parsed_date:
                game_date = parsed_date
        except Exception:
            # Keep fallback season label when file cannot be parsed.
            pass

        games.append(
            {
                "game_date": game_date,
                "team_abbreviation_home": m.group(1),
                "team_abbreviation_road": m.group(2),
                "game_id_norm": m.group(3),
                "file_name": game_path.name,
            }
        )

    # Sort newest first to match dropdown behavior.
    games.sort(key=lambda g: (str(g.get("game_date") or ""), str(g.get("game_id_norm") or "")), reverse=True)

    out_path = season_dir / f"_index_{season}_{phase}.json"
    out_payload = {
        "season": season,
        "phase": phase,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "count": len(games),
        "games": games,
    }
    out_path.write_text(json.dumps(out_payload, ensure_ascii=True), encoding="utf-8")
    print(f"[pbp-index] Wrote {out_path} (games={len(games)})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pbp_analyze index for one season/phase")
    parser.add_argument("--states-root", required=True, help="Root path like <GLA>/data/pbp/processed/game_states")
    parser.add_argument("--season", required=True, help="Season like 2025-26")
    parser.add_argument("--phase", required=True, choices=["regular", "playoffs"], help="Season phase")
    args = parser.parse_args()

    return build_index(Path(args.states_root), args.season, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
