import sys
import tempfile
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

    def test_resolve_pbp_input_path_uses_remote_pbp_when_local_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "NBA_Data"
            remote_file = Path(tmp) / "remote.parquet"
            remote_file.write_bytes(b"parquet")
            seen = []

            def fake_download(relative_path):
                seen.append(relative_path)
                if relative_path == "nbastatsv3/playoffs/nbastatsv3_po_2025.parquet":
                    return remote_file
                return None

            with mock.patch.object(pbp_boxscore, "_download_remote_pbpdata_file", side_effect=fake_download):
                path, source = pbp_boxscore._resolve_pbp_input_path(repo_dir, "2025-26", "playoffs")

        self.assertEqual(path, remote_file)
        self.assertEqual(source, "nbastatsv3_remote")
        self.assertIn("nbastatsv3/playoffs/nbastatsv3_po_2025.parquet", seen)

    def test_official_position_marks_starter_when_rotation_is_unavailable(self):
        starter_info = pbp_boxscore._build_empty_starter_info([1610612760])
        players_df = pd.DataFrame(
            [
                {
                    "game_id": "0042500316",
                    "team_id": 1610612760,
                    "person_id": 1628983,
                    "name_i": "S. Gilgeous-Alexander",
                    "first_name": "Shai",
                    "family_name": "Gilgeous-Alexander",
                    "position": "G",
                },
                {
                    "game_id": "0042500316",
                    "team_id": 1610612760,
                    "person_id": 1627936,
                    "name_i": "A. Caruso",
                    "first_name": "Alex",
                    "family_name": "Caruso",
                    "position": "",
                },
            ]
        )

        with mock.patch.object(pbp_boxscore, "_load_data_csv", return_value=players_df):
            starter_info = pbp_boxscore._apply_official_starter_info(
                season="2025-26",
                game_id="0042500316",
                team_ids=[1610612760],
                starter_info=starter_info,
            )

        self.assertTrue(
            pbp_boxscore._is_starter_row(
                team_id=1610612760,
                player_id=1628983,
                player_name="S. Gilgeous-Alexander",
                token="1610612760:1628983",
                starter_info=starter_info,
            )
        )
        self.assertFalse(
            pbp_boxscore._is_starter_row(
                team_id=1610612760,
                player_id=1627936,
                player_name="A. Caruso",
                token="1610612760:1627936",
                starter_info=starter_info,
            )
        )

    def test_full_game_uses_traditional_fallback_without_loading_pbp(self):
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
            mock.patch.object(pbp_boxscore, "_download_remote_pbpdata_file", return_value=None),
            mock.patch.object(pbp_boxscore, "_load_pbp_df") as load_pbp_df,
            mock.patch.object(pbp_boxscore, "_load_traditional_boxscore_fallback", return_value=fallback_payload),
        ):
            payload = pbp_boxscore.compute_pbp_traditional_boxscore("2025-26", "0042500301", "game")

        self.assertEqual(payload, fallback_payload)
        load_pbp_df.assert_not_called()

    def test_segmented_box_score_uses_game_state_payload_without_loading_season_pbp(self):
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
        payload = {
            "events": [
                {
                    "action_number": 1,
                    "action_id": 1,
                    "period": 1,
                    "clock": "PT11M30.00S",
                    "team_id": 1610612752,
                    "team_tricode": "NYK",
                    "person_id": 123,
                    "player_name": "Example Player",
                    "description": "Example Player 12' Jump Shot",
                    "action_type": "2pt",
                    "sub_type": "Jump Shot",
                    "shot_result": "Made",
                    "shot_value": 2,
                    "score_home": 2,
                    "score_away": 0,
                }
            ]
        }

        with (
            mock.patch.object(pbp_boxscore, "_load_game_metadata", return_value=meta),
            mock.patch.object(pbp_boxscore, "_load_game_state_payload", return_value=payload),
            mock.patch.object(pbp_boxscore, "_load_pbp_df") as load_pbp_df,
            mock.patch.object(pbp_boxscore, "_build_segment_include_map", return_value=None),
            mock.patch.object(pbp_boxscore, "_fetch_game_rotation", side_effect=RuntimeError("rotation unavailable")),
            mock.patch.object(pbp_boxscore, "_infer_period_start_lineup", side_effect=ValueError("lineup unavailable")),
            mock.patch.object(pbp_boxscore, "_apply_official_starter_info", side_effect=lambda **kwargs: kwargs["starter_info"]),
        ):
            payload = pbp_boxscore.compute_pbp_traditional_boxscore("2025-26", "0042500301", "q1")

        load_pbp_df.assert_not_called()
        self.assertEqual(payload["source"], "game_states")
        self.assertEqual(payload["minutes_plus_minus_source"], "pbp_stats_only:q1")
        self.assertEqual(payload["home_players"][0]["player_name"], "Example Player")
        self.assertEqual(payload["home_players"][0]["pts"], 2)

    def test_segmented_box_score_falls_back_to_season_pbp_when_game_states_are_unavailable(self):
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
            mock.patch.object(pbp_boxscore, "_load_game_state_payload", return_value=None),
            mock.patch.object(pbp_boxscore, "_build_pbp_path", return_value=(Path("/missing/pbp.parquet"), "nbastatsv3")),
            mock.patch.object(pbp_boxscore, "_download_remote_pbpdata_file", return_value=None),
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
            mock.patch.object(pbp_boxscore, "_load_game_state_payload", return_value=None),
            mock.patch.object(pbp_boxscore, "_build_pbp_path", return_value=(Path("/fake/pbp.parquet"), "nbastatsv3")),
            mock.patch.object(pbp_boxscore, "_download_remote_pbpdata_file", return_value=None),
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
