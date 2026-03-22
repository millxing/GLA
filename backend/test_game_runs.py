import math
import unittest

from services.game_runs import extract_timeline_possessions, rank_non_overlapping_runs


class GameRunsAlgorithmTest(unittest.TestCase):
    def test_extract_timeline_possessions_uses_previous_event_as_opening_anchor(self):
        events = [
            {
                "event_index": 1,
                "period": 1,
                "clock": "PT12M00.00S",
                "description": "Jump ball",
                "home_win_prob": 0.50,
                "game_log_state": {"pts_home": 0, "pts_road": 0},
            },
            {
                "event_index": 2,
                "period": 1,
                "clock": "PT11M42.00S",
                "description": "Road miss",
                "home_win_prob": 0.47,
                "possession_before_side": None,
                "possession_after_side": "road",
                "possession_changed": True,
                "game_log_state": {"pts_home": 0, "pts_road": 0},
            },
            {
                "event_index": 3,
                "period": 1,
                "clock": "PT11M38.00S",
                "description": "Home rebound",
                "home_win_prob": 0.51,
                "possession_before_side": "road",
                "possession_after_side": "home",
                "possession_changed": True,
                "game_log_state": {"pts_home": 0, "pts_road": 0},
            },
            {
                "event_index": 4,
                "period": 1,
                "clock": "PT11M20.00S",
                "description": "Home basket",
                "home_win_prob": 0.61,
                "possession_before_side": "home",
                "possession_after_side": "road",
                "possession_changed": True,
                "game_log_state": {"pts_home": 2, "pts_road": 0},
            },
        ]

        possessions = extract_timeline_possessions(events, home_team="OKC", road_team="HOU")

        self.assertEqual(len(possessions), 3)
        self.assertEqual(possessions[0]["team"], "HOU")
        self.assertEqual(possessions[0]["start_event_index"], 1)
        self.assertEqual(possessions[0]["end_event_index"], 3)
        self.assertAlmostEqual(possessions[0]["start_home_win_prob"], 0.50)
        self.assertAlmostEqual(possessions[0]["end_home_win_prob"], 0.51)
        self.assertEqual(possessions[1]["team"], "OKC")
        self.assertEqual(possessions[1]["home_points_scored"], 2)
        self.assertEqual(possessions[1]["road_points_scored"], 0)
        self.assertEqual(possessions[2]["team"], "HOU")
        self.assertAlmostEqual(possessions[2]["start_home_win_prob"], 0.61)
        self.assertAlmostEqual(possessions[2]["end_home_win_prob"], 0.61)

    def test_rank_non_overlapping_runs_is_greedy_by_absolute_run_score(self):
        run_alpha = 0.3
        possessions = [
            {
                "side": "home",
                "team": "OKC",
                "start_event_index": 1,
                "end_event_index": 2,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT12M00.00S",
                "end_clock": "PT11M30.00S",
                "start_description": "start 0",
                "end_description": "end 0",
                "start_home_win_prob": 0.50,
                "end_home_win_prob": 0.60,
                "start_home_score": 0,
                "start_road_score": 0,
                "end_home_score": 2,
                "end_road_score": 0,
            },
            {
                "side": "home",
                "team": "OKC",
                "start_event_index": 2,
                "end_event_index": 3,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT11M30.00S",
                "end_clock": "PT11M00.00S",
                "start_description": "start 1",
                "end_description": "end 1",
                "start_home_win_prob": 0.60,
                "end_home_win_prob": 0.82,
                "start_home_score": 2,
                "start_road_score": 0,
                "end_home_score": 5,
                "end_road_score": 0,
            },
            {
                "side": "home",
                "team": "OKC",
                "start_event_index": 3,
                "end_event_index": 4,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT11M00.00S",
                "end_clock": "PT10M30.00S",
                "start_description": "start 2",
                "end_description": "end 2",
                "start_home_win_prob": 0.82,
                "end_home_win_prob": 0.84,
                "start_home_score": 5,
                "start_road_score": 0,
                "end_home_score": 5,
                "end_road_score": 0,
            },
            {
                "side": "road",
                "team": "HOU",
                "start_event_index": 4,
                "end_event_index": 5,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT10M30.00S",
                "end_clock": "PT10M00.00S",
                "start_description": "start 3",
                "end_description": "end 3",
                "start_home_win_prob": 0.84,
                "end_home_win_prob": 0.61,
                "start_home_score": 5,
                "start_road_score": 0,
                "end_home_score": 5,
                "end_road_score": 4,
            },
            {
                "side": "road",
                "team": "HOU",
                "start_event_index": 5,
                "end_event_index": 6,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT10M00.00S",
                "end_clock": "PT09M30.00S",
                "start_description": "start 4",
                "end_description": "end 4",
                "start_home_win_prob": 0.61,
                "end_home_win_prob": 0.63,
                "start_home_score": 5,
                "start_road_score": 4,
                "end_home_score": 5,
                "end_road_score": 4,
            },
        ]

        runs = rank_non_overlapping_runs(
            possessions,
            home_team="OKC",
            road_team="HOU",
            max_possessions=2,
            run_alpha=run_alpha,
            limit=3,
        )

        self.assertEqual(len(runs), 3)

        first = runs[0]
        second = runs[1]
        third = runs[2]

        self.assertEqual((first["start_possession_index"], first["end_possession_index"]), (0, 1))
        self.assertEqual(first["run_team"], "OKC")
        self.assertAlmostEqual(first["run_score"], (0.82 - 0.50) / math.pow(3, run_alpha))

        self.assertEqual((second["start_possession_index"], second["end_possession_index"]), (3, 3))
        self.assertEqual(second["run_team"], "HOU")
        self.assertLess(second["run_score"], 0.0)

        self.assertLess(first["end_possession_index"], second["start_possession_index"])
        self.assertEqual((third["start_possession_index"], third["end_possession_index"]), (2, 2))
        self.assertGreater(third["run_score"], 0.0)

    def test_rank_non_overlapping_runs_supports_unbounded_maxposs(self):
        runs = rank_non_overlapping_runs(
            [
                {
                    "side": "home",
                    "team": "OKC",
                    "start_event_index": 1,
                    "end_event_index": 2,
                    "start_period": 1,
                    "end_period": 1,
                    "start_clock": "PT12M00.00S",
                    "end_clock": "PT11M30.00S",
                    "start_description": "start 0",
                    "end_description": "end 0",
                    "start_home_win_prob": 0.50,
                    "end_home_win_prob": 0.60,
                    "start_home_score": 0,
                    "start_road_score": 0,
                    "end_home_score": 2,
                    "end_road_score": 0,
                },
                {
                    "side": "home",
                    "team": "OKC",
                    "start_event_index": 2,
                    "end_event_index": 3,
                    "start_period": 1,
                    "end_period": 1,
                    "start_clock": "PT11M30.00S",
                    "end_clock": "PT11M00.00S",
                    "start_description": "start 1",
                    "end_description": "end 1",
                    "start_home_win_prob": 0.60,
                    "end_home_win_prob": 0.70,
                    "start_home_score": 2,
                    "start_road_score": 0,
                    "end_home_score": 4,
                    "end_road_score": 0,
                },
                {
                    "side": "home",
                    "team": "OKC",
                    "start_event_index": 3,
                    "end_event_index": 4,
                    "start_period": 1,
                    "end_period": 1,
                    "start_clock": "PT11M00.00S",
                    "end_clock": "PT10M30.00S",
                    "start_description": "start 2",
                    "end_description": "end 2",
                    "start_home_win_prob": 0.70,
                    "end_home_win_prob": 0.90,
                    "start_home_score": 4,
                    "start_road_score": 0,
                    "end_home_score": 7,
                    "end_road_score": 0,
                },
            ],
            home_team="OKC",
            road_team="HOU",
            max_possessions=None,
            run_alpha=0.3,
            limit=1,
        )

        self.assertEqual((runs[0]["start_possession_index"], runs[0]["end_possession_index"]), (0, 2))

    def test_rank_non_overlapping_runs_respects_minposs(self):
        possessions = [
            {
                "side": "home",
                "team": "OKC",
                "start_event_index": 1,
                "end_event_index": 2,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT12M00.00S",
                "end_clock": "PT11M30.00S",
                "start_description": "start 0",
                "end_description": "end 0",
                "start_home_win_prob": 0.50,
                "end_home_win_prob": 0.65,
                "start_home_score": 0,
                "start_road_score": 0,
                "end_home_score": 3,
                "end_road_score": 0,
            },
            {
                "side": "road",
                "team": "HOU",
                "start_event_index": 2,
                "end_event_index": 3,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT11M30.00S",
                "end_clock": "PT11M00.00S",
                "start_description": "start 1",
                "end_description": "end 1",
                "start_home_win_prob": 0.65,
                "end_home_win_prob": 0.62,
                "start_home_score": 3,
                "start_road_score": 0,
                "end_home_score": 3,
                "end_road_score": 2,
            },
            {
                "side": "home",
                "team": "OKC",
                "start_event_index": 3,
                "end_event_index": 4,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT11M00.00S",
                "end_clock": "PT10M30.00S",
                "start_description": "start 2",
                "end_description": "end 2",
                "start_home_win_prob": 0.62,
                "end_home_win_prob": 0.72,
                "start_home_score": 3,
                "start_road_score": 2,
                "end_home_score": 5,
                "end_road_score": 2,
            },
        ]

        runs = rank_non_overlapping_runs(
            possessions,
            home_team="OKC",
            road_team="HOU",
            max_possessions=None,
            min_possessions=2,
            run_alpha=0.5,
            limit=5,
        )

        self.assertTrue(all(run["possession_count"] >= 2 for run in runs))

    def test_rank_non_overlapping_runs_respects_minmargin(self):
        possessions = [
            {
                "side": "home",
                "team": "OKC",
                "start_event_index": 1,
                "end_event_index": 2,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT12M00.00S",
                "end_clock": "PT11M30.00S",
                "start_description": "start 0",
                "end_description": "end 0",
                "start_home_win_prob": 0.50,
                "end_home_win_prob": 0.60,
                "start_home_score": 0,
                "start_road_score": 0,
                "end_home_score": 4,
                "end_road_score": 0,
            },
            {
                "side": "home",
                "team": "OKC",
                "start_event_index": 2,
                "end_event_index": 3,
                "start_period": 1,
                "end_period": 1,
                "start_clock": "PT11M30.00S",
                "end_clock": "PT11M00.00S",
                "start_description": "start 1",
                "end_description": "end 1",
                "start_home_win_prob": 0.60,
                "end_home_win_prob": 0.80,
                "start_home_score": 4,
                "start_road_score": 0,
                "end_home_score": 8,
                "end_road_score": 0,
            },
        ]

        runs = rank_non_overlapping_runs(
            possessions,
            home_team="OKC",
            road_team="HOU",
            max_possessions=None,
            min_possessions=1,
            min_margin=8,
            run_alpha=0.5,
            limit=5,
        )

        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0]["start_possession_index"], runs[0]["end_possession_index"]), (0, 1))
        self.assertEqual(runs[0]["score_margin_delta"], 8)


if __name__ == "__main__":
    unittest.main()
