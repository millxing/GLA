import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routers import api
from services import player_shots
from services.player_shots import analyze_shot_streakiness_sequence


def _valid_test_season() -> str:
    return sorted(api._VALID_SEASONS)[-1]


class ShotStreakinessMetricTest(unittest.TestCase):
    def test_all_makes_are_neutral_not_streaky(self):
        metrics = analyze_shot_streakiness_sequence(
            [1] * 100,
            season="2025-26",
            player_id=1,
            shot_type="fta",
            simulations=200,
        )

        self.assertEqual(metrics["classification"], "Ordinary")
        self.assertEqual(metrics["runs_cluster_percentile"], 50.0)
        self.assertEqual(metrics["transition_percentile"], 50.0)

    def test_perfect_alternation_is_alternating(self):
        metrics = analyze_shot_streakiness_sequence(
            [1, 0] * 60,
            season="2025-26",
            player_id=2,
            shot_type="3ptfga",
            simulations=300,
        )

        self.assertEqual(metrics["classification"], "Alternating")
        self.assertGreaterEqual(metrics["runs_alternation_percentile"], 90.0)

    def test_clustered_sequence_is_streaky(self):
        metrics = analyze_shot_streakiness_sequence(
            [1] * 60 + [0] * 60,
            season="2025-26",
            player_id=3,
            shot_type="3ptfga",
            simulations=300,
        )

        self.assertEqual(metrics["classification"], "Streaky")
        self.assertGreaterEqual(metrics["streakiness_score"], 90.0)

    def test_even_window_distribution_scores_as_consistent(self):
        rng = np.random.default_rng(7)
        sequence = []
        for _ in range(8):
            window = np.array([1] * 10 + [0] * 15, dtype=np.int8)
            rng.shuffle(window)
            sequence.extend(window.tolist())

        metrics = analyze_shot_streakiness_sequence(
            sequence,
            season="2025-26",
            player_id=4,
            shot_type="3ptfga",
            simulations=300,
        )

        self.assertGreaterEqual(metrics["consistency_score"], 90.0)

    def test_seeded_random_sequence_is_ordinary(self):
        rng = np.random.default_rng(11)
        sequence = rng.binomial(1, 0.38, size=160).tolist()

        metrics = analyze_shot_streakiness_sequence(
            sequence,
            season="2025-26",
            player_id=5,
            shot_type="3ptfga",
            simulations=300,
        )

        self.assertEqual(metrics["classification"], "Ordinary")


class ShotStreakinessApiTest(unittest.TestCase):
    def setUp(self):
        player_shots.build_player_shot_streakiness_payload.cache_clear()
        self.app = FastAPI()
        self.app.include_router(api.router)
        self.client = TestClient(self.app)
        self.season = _valid_test_season()

    def tearDown(self):
        player_shots.build_player_shot_streakiness_payload.cache_clear()

    def _write_player_shots(self, root: Path):
        rows = []
        for idx in range(120):
            rows.append(
                {
                    "season": self.season,
                    "game_type": "regular_season",
                    "team": "BOS",
                    "player_id": 1,
                    "player_name": "High Volume",
                    "shot_type": "3ptfga",
                    "result": "make" if idx % 3 == 0 else "miss",
                    "game_date": "2026-01-01",
                    "game_id": f"001{idx // 20:04d}",
                    "action_number": idx,
                    "action_id": idx,
                }
            )
        for idx in range(80):
            rows.append(
                {
                    "season": self.season,
                    "game_type": "regular_season",
                    "team": "NYK",
                    "player_id": 2,
                    "player_name": "Low Volume",
                    "shot_type": "3ptfga",
                    "result": "make" if idx % 2 == 0 else "miss",
                    "game_date": "2026-01-01",
                    "game_id": f"002{idx // 20:04d}",
                    "action_number": idx,
                    "action_id": idx,
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_parquet(root / f"player_shots_{self.season}.parquet", index=False)

    def test_rejects_invalid_filters(self):
        response = self.client.get(f"/api/player-shot-streakiness?season={self.season}&shot_type=bad")
        self.assertEqual(response.status_code, 400)

        response = self.client.get(f"/api/player-shot-streakiness?season={self.season}&game_type=bad")
        self.assertEqual(response.status_code, 400)

    def test_missing_parquet_returns_empty_response(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(player_shots, "PLAYER_SHOTS_ROOT", Path(tmpdir)):
                response = self.client.get(
                    f"/api/player-shot-streakiness?season={self.season}&shot_type=3ptfga&simulations=100"
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["row_count"], 0)
        self.assertEqual(payload["rows"], [])

    def test_honors_min_attempts_and_is_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_player_shots(root)
            with mock.patch.object(player_shots, "PLAYER_SHOTS_ROOT", root):
                url = f"/api/player-shot-streakiness?season={self.season}&shot_type=3ptfga&min_attempts=100&simulations=100"
                first = self.client.get(url)
                second = self.client.get(url)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())
        payload = first.json()
        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["rows"][0]["player_name"], "High Volume")


if __name__ == "__main__":
    unittest.main()
