import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admin import pbp_game_states as states
from services import pbp_boxscore


class PBPEventOrderingTest(unittest.TestCase):
    def test_late_action_numbers_do_not_move_third_quarter_events_to_fourth(self):
        fixture = Path(__file__).parent / 'fixtures' / 'pbp_chi_nyk_0022600177.csv'
        raw = states._load_pbp_df(fixture)
        ordered, _ = states._prepare_events_for_processing(raw)
        displaced = ordered[ordered.actionId.isin([411, 412, 413])]
        self.assertEqual(displaced.period.tolist(), [3, 3, 3])

        meta = {
            'season': '2026-27', 'game_id': '0022600177',
            'game_date': '2026-11-04', 'game_type': 'regular_season',
            'phase': 'regular', 'home_team_id': 1610612752,
            'road_team_id': 1610612741, 'home_team': 'NYK', 'road_team': 'CHI',
        }
        with (
            mock.patch.object(pbp_boxscore, '_load_game_metadata', return_value=meta),
            mock.patch.object(pbp_boxscore, '_load_game_pbp_df', return_value=(ordered, 'fixture')),
            mock.patch.object(pbp_boxscore, '_fetch_game_rotation', side_effect=RuntimeError('unavailable')),
            mock.patch.object(pbp_boxscore, '_build_segment_include_map', return_value=None),
            mock.patch.object(pbp_boxscore, '_apply_official_starter_info', side_effect=lambda **kw: kw['starter_info']),
        ):
            result = pbp_boxscore.compute_pbp_traditional_boxscore('2026-27', '0022600177', 'q4')
        self.assertEqual(result['minutes_plus_minus_source'], 'pbp_segmented:q4')
        for side, points, differential in [('home_players', 22, -20), ('road_players', 26, 20)]:
            rows = result[side]
            self.assertTrue(all(row['minutes'] != '0:00' for row in rows))
            self.assertEqual(sum(row['pts'] for row in rows), points)
            self.assertEqual(sum(row['plus_minus'] for row in rows), differential)
        # Official GameRotation Q4 stint totals, including substitutions
        # between free throws at the same clock time.
        expected = {
            'Knecht': -4, 'Wiseman': -7, 'Alvarado': -6, 'Dadiet': -6,
            'Diawara': 0, 'Clarkson': -2, 'Kolek': 2, 'Drummond': 3,
            'Miller': 4, 'Williams': 4, 'Sellers': 6, 'Hawkins': 5,
            'Swain': 5, 'Buzelis': -1, 'Giddey': -1, 'Essengue': -2,
        }
        self.assertEqual({r['player_name'].split('. ')[-1]: r['plus_minus'] for r in
                          result['home_players'] + result['road_players']}, expected)

    def test_unsupported_period_backtrack_is_still_repaired(self):
        raw = pd.DataFrame([
            {'actionNumber': 1, 'actionId': 1, 'period': 4, 'clock': 'PT01M00S', 'actionType': 'Made Shot'},
            {'actionNumber': 2, 'actionId': 2, 'period': 3, 'clock': 'PT00M50S', 'actionType': 'Rebound'},
            {'actionNumber': 3, 'actionId': 3, 'period': 4, 'clock': 'PT00M40S', 'actionType': 'Made Shot'},
        ])
        ordered, anomalies = states._prepare_events_for_processing(raw)
        self.assertEqual(ordered.period.tolist(), [4, 4, 4])
        self.assertEqual(anomalies['period_backtracks_in_feed'], 1)


if __name__ == '__main__':
    unittest.main()
