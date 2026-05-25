import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services import pbp_boxscore


class PBPBoxScoreFallbackTest(unittest.TestCase):
    def setUp(self):
        pbp_boxscore._read_data_csv.cache_clear()

    def tearDown(self):
        pbp_boxscore._read_data_csv.cache_clear()

    def test_clear_pbp_boxscore_cache_flushes_cached_csv_reads(self):
        sample = pd.DataFrame({"game_id": ["0042500311"]})

        with (
            mock.patch.object(pbp_boxscore, "resolve_data_file_path", return_value=Path("/missing/team_game_logs.csv")),
            mock.patch.object(pbp_boxscore, "build_data_file_url", return_value="https://example.test/team_game_logs.csv"),
            mock.patch.object(pbp_boxscore.pd, "read_csv", return_value=sample) as read_csv,
        ):
            pbp_boxscore._load_data_csv("team_game_logs_2025-26.csv", dtype={"game_id": "string"})
            pbp_boxscore._load_data_csv("team_game_logs_2025-26.csv", dtype={"game_id": "string"})
            self.assertEqual(read_csv.call_count, 1)

            pbp_boxscore.clear_pbp_boxscore_cache()
            pbp_boxscore._load_data_csv("team_game_logs_2025-26.csv", dtype={"game_id": "string"})
            self.assertEqual(read_csv.call_count, 2)

    def test_full_game_uses_traditional_fallback_when_pbp_is_unavailable(self):
        meta = {
            "season": "2025-26",
            "game_id": "0042500301",
            "game_date": "2026-05-19",
            "game_type": "playoffs",
            "phase": "playoffs",
            "home_team_id": 1610612752,
            "home_team": "NYK",
            "road_team_id": 1610612739,
            "road_team": "CLE",
        }
        fallback_payload = {"source": "box_score_traditional_v3_fallback"}

        with (
            mock.patch.object(pbp_boxscore, "_load_game_metadata", return_value=meta),
            mock.patch.object(pbp_boxscore, "_build_pbp_path", return_value=(Path("/missing/pbp.parquet"), "nbastatsv3")),
            mock.patch.object(pbp_boxscore, "_load_pbp_df", side_effect=FileNotFoundError("missing PBP")),
            mock.patch.object(pbp_boxscore, "_load_traditional_boxscore_fallback", return_value=fallback_payload),
        ):
            payload = pbp_boxscore.compute_pbp_traditional_boxscore("2025-26", "0042500301", "game")

        self.assertEqual(payload, fallback_payload)

    def test_segmented_box_score_still_requires_pbp(self):
        meta = {
            "season": "2025-26",
            "game_id": "0042500301",
            "game_date": "2026-05-19",
            "game_type": "playoffs",
            "phase": "playoffs",
            "home_team_id": 1610612752,
            "home_team": "NYK",
            "road_team_id": 1610612739,
            "road_team": "CLE",
        }

        with (
            mock.patch.object(pbp_boxscore, "_load_game_metadata", return_value=meta),
            mock.patch.object(pbp_boxscore, "_build_pbp_path", return_value=(Path("/missing/pbp.parquet"), "nbastatsv3")),
            mock.patch.object(pbp_boxscore, "_load_pbp_df", side_effect=FileNotFoundError("missing PBP")),
        ):
            with self.assertRaises(FileNotFoundError):
                pbp_boxscore.compute_pbp_traditional_boxscore("2025-26", "0042500301", "q1")

    def test_segmented_box_score_returns_stats_when_lineup_inference_fails(self):
        meta = {
            "season": "2025-26",
            "game_id": "0042500301",
            "game_date": "2026-05-19",
            "game_type": "playoffs",
            "phase": "playoffs",
            "home_team_id": 1610612752,
            "home_team": "NYK",
            "road_team_id": 1610612739,
            "road_team": "CLE",
        }
        pbp_df = pd.DataFrame(
            [
                {
                    "game_id_norm": "0042500301",
                    "period": 1,
                    "actionNumber": 1,
                    "actionId": 1,
                    "clock": "PT11M30.00S",
                    "teamId": 1610612752,
                    "personId": 123,
                    "playerName": "Example Player",
                    "playerNameI": "E. Player",
                    "actionType": "Made Shot",
                    "subType": "Jump Shot",
                    "shotResult": "Made",
                    "shotValue": 2,
                    "description": "Example Player 12' Jump Shot",
                    "scoreHome": 2,
                    "scoreAway": 0,
                }
            ]
        )

        with (
            mock.patch.object(pbp_boxscore, "_load_game_metadata", return_value=meta),
            mock.patch.object(pbp_boxscore, "_build_pbp_path", return_value=(Path("/fake/pbp.parquet"), "nbastatsv3")),
            mock.patch.object(pbp_boxscore, "_load_pbp_df", return_value=pbp_df),
            mock.patch.object(pbp_boxscore, "_build_segment_include_map", return_value=None),
            mock.patch.object(pbp_boxscore, "_fetch_game_rotation", side_effect=RuntimeError("rotation unavailable")),
            mock.patch.object(pbp_boxscore, "_infer_period_start_lineup", side_effect=ValueError("lineup unavailable")),
        ):
            payload = pbp_boxscore.compute_pbp_traditional_boxscore("2025-26", "0042500301", "q1")

        self.assertEqual(payload["minutes_plus_minus_source"], "pbp_stats_only:q1")
        self.assertEqual(payload["home_players"][0]["player_name"], "E. Player")
        self.assertEqual(payload["home_players"][0]["pts"], 2)


if __name__ == "__main__":
    unittest.main()
