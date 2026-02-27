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

import pandas as pd


def build_index(states_root: Path, season: str, phase: str) -> int:
    season_dir = states_root / phase / season
    if not season_dir.exists() or not season_dir.is_dir():
        print(f"[pbp-index] Missing season directory: {season_dir}")
        return 1

    games: list[dict[str, str]] = []
    parquet_path = season_dir / f"_states_{season}_{phase}.parquet"

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            print(f"[pbp-index] Failed to read packed states {parquet_path}: {exc}")
            return 1

        if not df.empty:
            for _, row in df.iterrows():
                game_id = str(row.get("game_id") or "").strip()
                if game_id.endswith(".0"):
                    game_id = game_id[:-2]
                digits = "".join(ch for ch in game_id if ch.isdigit())
                game_id_norm = digits.zfill(10) if digits else ""

                home = str(row.get("home_team") or "").strip().upper()
                road = str(row.get("road_team") or "").strip().upper()
                game_date = str(row.get("game_date") or season).strip() or season

                if (not home or not road) and isinstance(row.get("payload_json"), str):
                    try:
                        payload = json.loads(row.get("payload_json"))
                    except Exception:
                        payload = {}
                    if isinstance(payload, dict):
                        home = home or str(payload.get("home_team") or "").strip().upper()
                        road = road or str(payload.get("road_team") or "").strip().upper()
                        game_date = str(payload.get("game_date") or game_date).strip() or season

                if not (game_id_norm and home and road):
                    continue
                games.append(
                    {
                        "game_date": game_date,
                        "team_abbreviation_home": home,
                        "team_abbreviation_road": road,
                        "game_id_norm": game_id_norm,
                        "file_name": f"{season}_{home}_{road}_{game_id_norm}.json",
                    }
                )
    else:
        rx = re.compile(rf"^{re.escape(season)}_([A-Z0-9]+)_([A-Z0-9]+)_(\d{{10}})\.json$")

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
    parser.add_argument("--states-root", required=True, help="Root path like <NBA_Data>/PBPdata/game_states")
    parser.add_argument("--season", required=True, help="Season like 2025-26")
    parser.add_argument("--phase", required=True, choices=["regular", "playoffs"], help="Season phase")
    args = parser.parse_args()

    return build_index(Path(args.states_root), args.season, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
