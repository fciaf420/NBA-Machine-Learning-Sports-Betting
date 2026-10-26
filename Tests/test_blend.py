"""No pandas/numpy imports: test the exact core used by the frame adapter."""
import unittest
from src.Utils.blend import blend_records, prior_snapshot_date, dataset_table
from src.Utils.snapshots import select_snapshot_date


class BlendTests(unittest.TestCase):
    def setUp(self):
        self.current = [dict(TEAM_ID=1, TEAM_NAME="A", GP=0, W=0, L=0, W_PCT=.2, PTS=100, PTS_RANK=20),
                        dict(TEAM_ID=2, TEAM_NAME="B", GP=4, W=1, L=3, W_PCT=.25, PTS=90, PTS_RANK=25)]
        self.prior = [dict(TEAM_ID=2, PTS=80, W_PCT=.4, PTS_RANK=30),
                      dict(TEAM_ID=1, PTS=120, W_PCT=.8, PTS_RANK=1)]

    def test_zero_identity(self):
        out = blend_records(self.current, self.prior, 0)
        self.assertEqual(out, self.current)
        self.assertIsNot(out[0], self.current[0])

    def test_infinite_games(self):
        self.current[0]["GP"] = 10**12
        self.assertAlmostEqual(blend_records(self.current, self.prior, 10)[0]["PTS"], 100, places=8)

    def test_zero_games_prior_regression(self):
        self.assertEqual(blend_records(self.current, self.prior, 10)[0]["PTS"], 115)

    def test_id_match_and_shuffle(self):
        a = blend_records(self.current, self.prior, 10, 1)
        b = blend_records(self.current, list(reversed(self.prior)), 10, 1)
        self.assertEqual(a, b)
        self.assertEqual(a[0]["PTS"], 120)
        self.assertAlmostEqual(a[1]["PTS"], (4 * 90 + 10 * 80) / 14)

    def test_raw_counts_and_ranks(self):
        out = blend_records(self.current, self.prior, 10)
        for name in ("TEAM_ID", "TEAM_NAME", "GP", "W", "L"):
            self.assertEqual(out[1][name], self.current[1][name])
        self.assertNotEqual(out[1]["W_PCT"], self.current[1]["W_PCT"])
        self.assertNotEqual(out[1]["PTS_RANK"], self.current[1]["PTS_RANK"])

    def test_missing_team_league_mean(self):
        self.current[0]["TEAM_ID"] = 99
        self.assertEqual(blend_records(self.current, self.prior, 10)[0]["PTS"], 100)

    def test_numeric_id_types(self):
        self.current[0]["TEAM_ID"] = 1.0
        self.prior[1]["TEAM_ID"] = "1"
        self.assertEqual(blend_records(self.current, self.prior, 10, 1)[0]["PTS"], 120)

    def test_market_baseline_and_metrics(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "evaluate_blend", Path(__file__).resolve().parents[1] / "src/Train-Models/Evaluate_Blend.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertAlmostEqual(module.market_probability(-150, 150), .6)
        self.assertAlmostEqual(module.market_probability(-110, -110), .5)
        self.assertIsNone(module.market_probability(None, 100))
        self.assertIsNone(module.market_probability(0, 100))
        loss, brier, accuracy = module.metrics([0, 1], [.25, .75])
        self.assertAlmostEqual(loss, -__import__("math").log(.75))
        self.assertEqual((brier, accuracy), (.0625, 1.0))

    def test_missing_prior(self):
        self.assertEqual(blend_records(self.current, [], 10), self.current)

    def test_prior_snapshot_only_previous_season(self):
        config = {"get-data": {"2025-26": {"start_date": "2025-10-21", "end_date": "2026-6-21"},
                               "2026-27": {"start_date": "2026-10-20", "end_date": "2027-6-20"}}}
        dates = ["2025-06-21", "2025-10-20", "2026-06-20", "2026-06-21", "2026-06-22", "2026-10-25"]
        self.assertEqual(prior_snapshot_date(dates, "2026-27", config), "2026-06-21")
        self.assertIsNone(prior_snapshot_date(["2026-10-25"], "2026-27", config))
        self.assertIsNone(prior_snapshot_date(dates, "2012-13", config))
        self.assertEqual(select_snapshot_date(dates, "2026-10-25", "2026-10-20"), None)

    def test_settings_and_table(self):
        self.assertEqual(dataset_table(0), "dataset_2012-27")
        self.assertEqual(dataset_table(10.0), "dataset_2012-27_k10")
        for k, r in [(-1, .75), (1, 2), (float("nan"), .75)]:
            with self.assertRaises(ValueError):
                blend_records(self.current, self.prior, k, r)


if __name__ == "__main__":
    unittest.main()
