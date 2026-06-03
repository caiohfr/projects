import unittest

import pandas as pd

from src.vde_core.services import (
    autoresolve_test_mass,
    compute_wltp_test_mass,
    epa_city_hwy_from_phase,
    inertia_class_from_mass,
    wltp_phases_from_phase,
)


class CoreServicesTests(unittest.TestCase):
    def test_compute_wltp_test_mass_uses_expected_formula(self):
        tm = compute_wltp_test_mass(1500.0, options_kg=50.0, tpmlm_kg=2000.0, category=1)
        expected = (1500.0 + 50.0) + 25.0 + 0.15 * (2000.0 - 1500.0 - 25.0 - 50.0)
        self.assertAlmostEqual(tm, expected)

    def test_inertia_class_from_mass_returns_expected_step(self):
        self.assertEqual(inertia_class_from_mass(1550.0), 1701.0)
        self.assertEqual(inertia_class_from_mass(340.0), 454.0)
        self.assertIsNone(inertia_class_from_mass(None))

    def test_autoresolve_test_mass_for_epa_sets_inertia_class(self):
        row = {"legislation": "EPA", "mass_kg": 1550.0}
        resolved = autoresolve_test_mass(row)
        self.assertEqual(resolved["inertia_class"], 1701.0)

    def test_epa_city_hwy_from_phase_returns_combined_value(self):
        df = pd.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "v": [8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 12.0, 12.0, 12.0],
                "phase": ["bag1", "bag1", "bag1", "bag2", "bag2", "bag2", "hwfet", "hwfet", "hwfet"],
            }
        )

        result = epa_city_hwy_from_phase(df, 120.0, 0.02, 0.011, 1550.0)

        self.assertGreater(result["urb_MJ_km"], 0.0)
        self.assertGreater(result["hw_MJ_km"], 0.0)
        self.assertAlmostEqual(
            result["net_comb_MJ_km"],
            0.55 * result["urb_MJ_km"] + 0.45 * result["hw_MJ_km"],
        )

    def test_wltp_phases_from_phase_returns_phase_outputs(self):
        df = pd.DataFrame(
            {
                "t": [
                    0.0, 1.0, 2.0,
                    0.0, 1.0, 2.0,
                    0.0, 1.0, 2.0,
                    0.0, 1.0, 2.0,
                ],
                "v": [
                    6.0, 6.0, 6.0,
                    8.0, 8.0, 8.0,
                    10.0, 10.0, 10.0,
                    12.0, 12.0, 12.0,
                ],
                "phase": [
                    "low", "low", "low",
                    "mid", "mid", "mid",
                    "high", "high", "high",
                    "xhigh", "xhigh", "xhigh",
                ],
            }
        )

        result = wltp_phases_from_phase(df, 120.0, 0.02, 0.011, 1550.0)

        self.assertGreater(result["vde_low_mj_per_km"], 0.0)
        self.assertGreater(result["vde_mid_mj_per_km"], 0.0)
        self.assertGreater(result["vde_high_mj_per_km"], 0.0)
        self.assertGreater(result["vde_extra_high_mj_per_km"], 0.0)
        self.assertGreater(result["vde_net_mj_per_km"], 0.0)


if __name__ == "__main__":
    unittest.main()
