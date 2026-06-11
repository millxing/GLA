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

    def test_singular_playoff_game_type_maps_to_playoff_phase(self):
        self.assertEqual(pbp_boxscore._pbp_phase_from_game_type("playoff"), "playoffs")

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

    def test_segmented_plus_minus_ignores_prior_out_of_segment_scoring(self):
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
        home_lineup = {f"{meta['home_team_id']}:{player_id}" for player_id in range(101, 106)}
        road_lineup = {f"{meta['road_team_id']}:{player_id}" for player_id in range(201, 206)}
        pbp_df = pd.DataFrame(
            [
                {
                    "game_id_norm": "0042500301",
                    "period": 1,
                    "actionNumber": 1,
                    "actionId": 1,
                    "clock": "PT11M30.00S",
                    "teamId": meta["home_team_id"],
                    "personId": 101,
                    "playerName": "Home One",
                    "playerNameI": "H. One",
                    "actionType": "Made Shot",
                    "subType": "Jump Shot",
                    "shotResult": "Made",
                    "shotValue": 2,
                    "description": "Home One 12' Jump Shot",
                    "scoreHome": 2,
                    "scoreAway": 0,
                },
                {
                    "game_id_norm": "0042500301",
                    "period": 4,
                    "actionNumber": 2,
                    "actionId": 2,
                    "clock": "PT11M00.00S",
                    "teamId": meta["home_team_id"],
                    "personId": 101,
                    "playerName": "Home One",
                    "playerNameI": "H. One",
                    "actionType": "Made Shot",
                    "subType": "Jump Shot",
                    "shotResult": "Made",
                    "shotValue": 3,
                    "description": "Home One 25' 3PT Jump Shot",
                    "scoreHome": 5,
                    "scoreAway": 0,
                },
            ]
        )

        def fake_infer_period_start_lineup(*, team_id, **_kwargs):
            return set(home_lineup if team_id == meta["home_team_id"] else road_lineup)

        with (
            mock.patch.object(pbp_boxscore, "_load_game_metadata", return_value=meta),
            mock.patch.object(pbp_boxscore, "_load_game_state_payload", return_value=None),
            mock.patch.object(pbp_boxscore, "_build_pbp_path", return_value=(Path("/fake/pbp.parquet"), "nbastatsv3")),
            mock.patch.object(pbp_boxscore, "_download_remote_pbpdata_file", return_value=None),
            mock.patch.object(pbp_boxscore, "_load_pbp_df", return_value=pbp_df),
            mock.patch.object(pbp_boxscore, "_fetch_game_rotation", side_effect=RuntimeError("rotation unavailable")),
            mock.patch.object(pbp_boxscore, "_infer_period_start_lineup", side_effect=fake_infer_period_start_lineup),
            mock.patch.object(pbp_boxscore, "_apply_official_starter_info", side_effect=lambda **kwargs: kwargs["starter_info"]),
        ):
            payload = pbp_boxscore.compute_pbp_traditional_boxscore("2025-26", "0042500301", "q4")

        home_players = {player["player_id"]: player for player in payload["home_players"]}
        road_players = {player["player_id"]: player for player in payload["road_players"]}

        self.assertEqual(payload["minutes_plus_minus_source"], "pbp_segmented:q4")
        self.assertEqual(home_players[101]["pts"], 3)
        self.assertEqual(home_players[101]["plus_minus"], 3)
        self.assertEqual(road_players[201]["plus_minus"], -3)

    def test_q4_rotation_segment_uses_stint_plus_minus(self):
        meta = {
            "season": "2025-26",
            "game_id": "0042500404",
            "game_date": "2026-06-10",
            "game_type": "playoffs",
            "phase": "playoffs",
            "home_team_id": 1610612752,
            "home_team": "NYK",
            "road_team_id": 1610612759,
            "road_team": "SAS",
        }
        pbp_df = pd.DataFrame(
            [
                {
                    "game_id_norm": "0042500404",
                    "period": 4,
                    "actionNumber": 1,
                    "actionId": 1,
                    "clock": "PT12M00.00S",
                    "teamId": 0,
                    "personId": 0,
                    "playerName": "",
                    "playerNameI": "",
                    "actionType": "period",
                    "subType": "start",
                    "shotResult": "",
                    "shotValue": 0,
                    "description": "Start of 4th Period",
                    "scoreHome": 75,
                    "scoreAway": 90,
                },
                {
                    "game_id_norm": "0042500404",
                    "period": 4,
                    "actionNumber": 2,
                    "actionId": 2,
                    "clock": "PT11M00.00S",
                    "teamId": meta["road_team_id"],
                    "personId": 1641705,
                    "playerName": "Victor Wembanyama",
                    "playerNameI": "V. Wembanyama",
                    "actionType": "Made Shot",
                    "subType": "Driving Layup",
                    "shotResult": "Made",
                    "shotValue": 2,
                    "description": "Wembanyama 1' Driving Layup",
                    "scoreHome": 75,
                    "scoreAway": 92,
                },
            ]
        )
        rotation_rows = {
            meta["road_team_id"]: [
                {
                    "team_id": meta["road_team_id"],
                    "player_id": 1641705,
                    "player_name": "Victor Wembanyama",
                    "start_seconds": 2160.0,
                    "end_seconds": 2880.0,
                    "seconds": 720.0,
                    "plus_minus": -16,
                }
            ],
            meta["home_team_id"]: [
                {
                    "team_id": meta["home_team_id"],
                    "player_id": 1628973,
                    "player_name": "Jalen Brunson",
                    "start_seconds": 2170.0,
                    "end_seconds": 2880.0,
                    "seconds": 710.0,
                    "plus_minus": 18,
                }
            ],
        }

        with (
            mock.patch.object(pbp_boxscore, "_load_game_metadata", return_value=meta),
            mock.patch.object(pbp_boxscore, "_load_game_state_payload", return_value=None),
            mock.patch.object(pbp_boxscore, "_build_pbp_path", return_value=(Path("/fake/pbp.parquet"), "nbastatsv3")),
            mock.patch.object(pbp_boxscore, "_download_remote_pbpdata_file", return_value=None),
            mock.patch.object(pbp_boxscore, "_load_pbp_df", return_value=pbp_df),
            mock.patch.object(pbp_boxscore, "_fetch_game_rotation", return_value=rotation_rows),
            mock.patch.object(pbp_boxscore, "_apply_official_starter_info", side_effect=lambda **kwargs: kwargs["starter_info"]),
        ):
            payload = pbp_boxscore.compute_pbp_traditional_boxscore("2025-26", "0042500404", "q4")

        road_players = {player["player_id"]: player for player in payload["road_players"]}
        home_players = {player["player_id"]: player for player in payload["home_players"]}

        self.assertEqual(payload["minutes_plus_minus_source"], "gamerotation_segmented:q4")
        self.assertEqual(road_players[1641705]["minutes"], "12:00")
        self.assertEqual(road_players[1641705]["pts"], 2)
        self.assertEqual(road_players[1641705]["plus_minus"], -16)
        self.assertEqual(home_players[1628973]["minutes"], "11:50")
        self.assertEqual(home_players[1628973]["plus_minus"], 18)

    def test_rotation_same_clock_sub_between_free_throws_uses_pre_sub_lineup(self):
        meta = {
            "season": "2025-26",
            "game_id": "0042500404",
            "game_date": "2026-06-10",
            "game_type": "playoffs",
            "phase": "playoffs",
            "home_team_id": 1610612752,
            "home_team": "NYK",
            "road_team_id": 1610612759,
            "road_team": "SAS",
        }
        pbp_df = pd.DataFrame(
            [
                {
                    "game_id_norm": "0042500404",
                    "period": 4,
                    "actionNumber": 1,
                    "actionId": 1,
                    "clock": "PT12M00.00S",
                    "teamId": 0,
                    "personId": 0,
                    "playerName": "",
                    "playerNameI": "",
                    "actionType": "period",
                    "subType": "start",
                    "shotResult": "",
                    "shotValue": 0,
                    "description": "Start of 4th Period",
                    "scoreHome": 0,
                    "scoreAway": 0,
                },
                {
                    "game_id_norm": "0042500404",
                    "period": 4,
                    "actionNumber": 2,
                    "actionId": 2,
                    "clock": "PT06M24.00S",
                    "teamId": meta["road_team_id"],
                    "personId": 1642264,
                    "playerName": "Stephon Castle",
                    "playerNameI": "S. Castle",
                    "actionType": "Free Throw",
                    "subType": "Free Throw 1 of 2",
                    "shotResult": "Made",
                    "shotValue": 0,
                    "description": "Castle Free Throw 1 of 2",
                    "scoreHome": 0,
                    "scoreAway": 1,
                },
                {
                    "game_id_norm": "0042500404",
                    "period": 4,
                    "actionNumber": 3,
                    "actionId": 3,
                    "clock": "PT06M24.00S",
                    "teamId": meta["road_team_id"],
                    "personId": 1642844,
                    "playerName": "Dylan Harper",
                    "playerNameI": "D. Harper",
                    "actionType": "Substitution",
                    "subType": "",
                    "shotResult": "",
                    "shotValue": 0,
                    "description": "SUB: Fox FOR Harper",
                    "scoreHome": None,
                    "scoreAway": None,
                },
                {
                    "game_id_norm": "0042500404",
                    "period": 4,
                    "actionNumber": 4,
                    "actionId": 4,
                    "clock": "PT06M24.00S",
                    "teamId": meta["road_team_id"],
                    "personId": 1642264,
                    "playerName": "Stephon Castle",
                    "playerNameI": "S. Castle",
                    "actionType": "Free Throw",
                    "subType": "Free Throw 2 of 2",
                    "shotResult": "Made",
                    "shotValue": 0,
                    "description": "Castle Free Throw 2 of 2",
                    "scoreHome": 0,
                    "scoreAway": 2,
                },
                {
                    "game_id_norm": "0042500404",
                    "period": 4,
                    "actionNumber": 5,
                    "actionId": 5,
                    "clock": "PT00M00.00S",
                    "teamId": 0,
                    "personId": 0,
                    "playerName": "",
                    "playerNameI": "",
                    "actionType": "period",
                    "subType": "end",
                    "shotResult": "",
                    "shotValue": 0,
                    "description": "End of 4th Period",
                    "scoreHome": 0,
                    "scoreAway": 2,
                },
            ]
        )
        rotation_rows = {
            meta["road_team_id"]: [
                {
                    "team_id": meta["road_team_id"],
                    "player_id": 1642844,
                    "player_name": "Dylan Harper",
                    "start_seconds": 2000.0,
                    "end_seconds": 2496.0,
                    "seconds": 496.0,
                    "plus_minus": 0,
                },
                {
                    "team_id": meta["road_team_id"],
                    "player_id": 1628368,
                    "player_name": "De'Aaron Fox",
                    "start_seconds": 2496.0,
                    "end_seconds": 2880.0,
                    "seconds": 384.0,
                    "plus_minus": 0,
                },
                {
                    "team_id": meta["road_team_id"],
                    "player_id": 1642264,
                    "player_name": "Stephon Castle",
                    "start_seconds": 2160.0,
                    "end_seconds": 2880.0,
                    "seconds": 720.0,
                    "plus_minus": 0,
                },
            ],
            meta["home_team_id"]: [
                {
                    "team_id": meta["home_team_id"],
                    "player_id": 1628973,
                    "player_name": "Jalen Brunson",
                    "start_seconds": 2160.0,
                    "end_seconds": 2880.0,
                    "seconds": 720.0,
                    "plus_minus": 0,
                }
            ],
        }

        with (
            mock.patch.object(pbp_boxscore, "_load_game_metadata", return_value=meta),
            mock.patch.object(pbp_boxscore, "_load_game_state_payload", return_value=None),
            mock.patch.object(pbp_boxscore, "_build_pbp_path", return_value=(Path("/fake/pbp.parquet"), "nbastatsv3")),
            mock.patch.object(pbp_boxscore, "_download_remote_pbpdata_file", return_value=None),
            mock.patch.object(pbp_boxscore, "_load_pbp_df", return_value=pbp_df),
            mock.patch.object(pbp_boxscore, "_fetch_game_rotation", return_value=rotation_rows),
            mock.patch.object(pbp_boxscore, "_apply_official_starter_info", side_effect=lambda **kwargs: kwargs["starter_info"]),
        ):
            payload = pbp_boxscore.compute_pbp_traditional_boxscore("2025-26", "0042500404", "q4")

        road_players = {player["player_id"]: player for player in payload["road_players"]}
        home_players = {player["player_id"]: player for player in payload["home_players"]}

        self.assertEqual(road_players[1642844]["plus_minus"], 2)
        self.assertEqual(road_players[1628368]["plus_minus"], 0)
        self.assertEqual(road_players[1642264]["plus_minus"], 2)
        self.assertEqual(home_players[1628973]["plus_minus"], -2)


if __name__ == "__main__":
    unittest.main()
