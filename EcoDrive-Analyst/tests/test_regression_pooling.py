import unittest
from unittest.mock import patch

from src.vde_core.regression import load_regression_dataset
from src.vde_app.components.pwt_fuel_energy import _regression_sample_quality


class RegressionPoolingTests(unittest.TestCase):
    @patch("src.vde_core.regression.fetchall")
    def test_active_vde_id_does_not_limit_regression_candidate_dataset(self, mock_fetchall):
        mock_fetchall.return_value = [
            {"vde_id": 5038, "electrification": "ICE", "fuel_l_per_100km": 6.4, "energy_Wh_per_km": None, "engine_max_power_kw": 100.0, "category": "B", "make": "AUDI", "vde_net_mj_per_km": 1.7},
            {"vde_id": 6100, "electrification": "ICE", "fuel_l_per_100km": 6.7, "energy_Wh_per_km": None, "engine_max_power_kw": 104.0, "category": "B", "make": "VW", "vde_net_mj_per_km": 1.8},
        ]

        df = load_regression_dataset({"vde_id": 5038, "legislation": "EPA"})

        sql = mock_fetchall.call_args[0][0]
        self.assertNotIn("f.vde_id = ?", sql)
        self.assertEqual(sorted(df["vde_id"].tolist()), [5038, 6100])

    @patch("src.vde_core.regression.fetchall")
    def test_view_all_with_peer_filters_still_queries_broad_training_pool(self, mock_fetchall):
        mock_fetchall.return_value = []

        load_regression_dataset(
            {
                "legislation": "EPA",
                "electrification": "ICE",
                "category": "B",
                "make": "AUDI",
                "power_kw_range": (75.0, 120.0),
            }
        )

        sql = mock_fetchall.call_args[0][0]
        self.assertIn("v.legislation = ?", sql)
        self.assertIn("f.electrification = ?", sql)
        self.assertIn("v.category = ?", sql)
        self.assertIn("v.make = ?", sql)
        self.assertIn("f.engine_max_power_kw BETWEEN ? AND ?", sql)
        self.assertNotIn("f.vde_id = ?", sql)

    def test_regression_sample_quality_enforces_minimum_sample(self):
        self.assertFalse(_regression_sample_quality(4)["can_fit"])
        self.assertEqual(_regression_sample_quality(4)["label"], "Insufficient sample")
        self.assertTrue(_regression_sample_quality(5)["can_fit"])
        self.assertEqual(_regression_sample_quality(10)["label"], "Low confidence / small sample")

    @patch("src.vde_core.regression.fetchall")
    def test_regression_can_filter_by_multiple_fuelcons_ids(self, mock_fetchall):
        mock_fetchall.return_value = []

        load_regression_dataset({"fuelcons_ids": [11, 14, 21], "legislation": "EPA"})

        sql = mock_fetchall.call_args[0][0]
        params = mock_fetchall.call_args[0][1]
        self.assertIn("f.id IN (?,?,?)", sql)
        self.assertIn("v.legislation = ?", sql)
        self.assertEqual(params[-3:], (11, 14, 21))


if __name__ == "__main__":
    unittest.main()
