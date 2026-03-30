import tempfile
import unittest
import sys
from pathlib import Path
from unittest import mock

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from admin import cli as admin_cli
from services import data_loader


class ConfigDataPathTest(unittest.TestCase):
    def test_build_canonical_and_legacy_paths(self):
        filename = config.build_data_filename("team_game_logs", "2025-26")
        self.assertEqual(filename, "team_game_logs_2025-26.csv")
        self.assertEqual(
            config.get_canonical_data_relative_path(filename),
            Path("team_game_logs") / filename,
        )
        self.assertEqual(config.get_legacy_data_relative_path(filename), Path(filename))

    def test_build_scoped_and_traditional_filenames(self):
        self.assertEqual(
            config.build_data_filename("box_score_advanced", "2025-26", "clutch"),
            "box_score_advanced_clutch_2025-26.csv",
        )
        self.assertEqual(
            config.build_box_score_traditional_filename("players", "2025-26"),
            "box_score_traditional_v3_players_2025-26.csv",
        )

    def test_resolve_data_file_path_prefers_nested_then_legacy(self):
        filename = config.build_data_filename("team_game_logs", "2025-26")
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir)
            legacy_path = config.get_legacy_data_file_path(filename, repo_dir=repo_dir)
            legacy_path.write_text("legacy", encoding="utf-8")

            self.assertEqual(
                config.resolve_data_file_path(filename, repo_dir=repo_dir),
                legacy_path,
            )

            canonical_path = config.get_canonical_data_file_path(filename, repo_dir=repo_dir)
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            canonical_path.write_text("nested", encoding="utf-8")

            self.assertEqual(
                config.resolve_data_file_path(filename, repo_dir=repo_dir),
                canonical_path,
            )

    def test_build_data_file_url_points_to_nested_path(self):
        filename = config.build_data_filename("linescores", "2025-26")
        self.assertTrue(
            config.build_data_file_url(filename).endswith(
                "linescores/linescores_2025-26.csv"
            )
        )


class DataLoaderPathResolutionTest(unittest.TestCase):
    def test_resolve_local_path_from_url_prefers_nested_file(self):
        filename = config.build_data_filename("team_game_logs", "2025-26")
        url = config.build_data_file_url(filename)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir)
            nested = config.get_canonical_data_file_path(filename, repo_dir=repo_dir)
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_text("nested", encoding="utf-8")

            with mock.patch.object(data_loader, "NBA_DATA_REPO_DIR", repo_dir):
                resolved = data_loader._resolve_local_path_from_url(url)

            self.assertEqual(resolved, nested.resolve())

    def test_resolve_local_path_from_url_falls_back_to_legacy_file(self):
        filename = config.build_data_filename("box_score_advanced", "2025-26")
        url = config.build_data_file_url(filename)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir)
            legacy = config.get_legacy_data_file_path(filename, repo_dir=repo_dir)
            legacy.write_text("legacy", encoding="utf-8")

            with mock.patch.object(data_loader, "NBA_DATA_REPO_DIR", repo_dir):
                resolved = data_loader._resolve_local_path_from_url(url)

            self.assertEqual(resolved, legacy.resolve())


class AdminCliDataPathTest(unittest.TestCase):
    def test_cli_canonical_data_path_targets_family_directory(self):
        filename = admin_cli._season_to_filename("2025-26")
        repo_dir = Path("/tmp/nba-data")
        self.assertEqual(
            admin_cli._canonical_repo_data_path(repo_dir, filename),
            repo_dir / "team_game_logs" / filename,
        )


if __name__ == "__main__":
    unittest.main()
