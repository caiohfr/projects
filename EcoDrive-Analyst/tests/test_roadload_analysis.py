from __future__ import annotations

import unittest

import pandas as pd

from src.vde_core.roadload_analysis import (
    build_cycle_power_analysis,
    build_roadload_curve,
    canonical_cycle_segments,
    roadload_force_N,
)


class TestRoadloadAnalysis(unittest.TestCase):
    def test_roadload_force_formula_supports_scalar_and_sequence(self):
        self.assertAlmostEqual(roadload_force_N(120.0, 0.020, 0.0080, 100.0), 202.0, places=9)
        self.assertEqual(
            [round(value, 2) for value in roadload_force_N(120.0, 0.020, 0.0080, [0.0, 50.0, 100.0, 120.0])],
            [120.0, 141.0, 202.0, 237.6],
        )

    def test_build_roadload_curve_returns_expected_checkpoint_forces(self):
        curve = build_roadload_curve({"A": 110.0, "B": 0.015, "C": 0.0072}, speed_max_kph=120, step_kph=10)

        self.assertEqual(curve["speed_kph"][0], 0)
        self.assertEqual(curve["speed_kph"][-1], 120)
        checkpoint_map = dict(zip(curve["speed_kph"], curve["force_N"]))
        self.assertAlmostEqual(checkpoint_map[0], 110.0, places=9)
        self.assertAlmostEqual(checkpoint_map[50], 128.75, places=9)
        self.assertAlmostEqual(checkpoint_map[100], 183.5, places=9)
        self.assertAlmostEqual(checkpoint_map[120], 215.48, places=9)

    def test_canonical_cycle_segments_excludes_combined_physical_selection(self):
        frame = pd.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0],
                "v": [0.0, 1.0, 2.0, 3.0],
                "phase": ["Bag 1", "Bag 1", "HWFET", "HWFET"],
            }
        )

        segments = canonical_cycle_segments(frame)

        self.assertEqual(list(segments), ["FTP-75", "HWFET"])
        self.assertNotIn("Combined", segments)

    def test_cycle_power_analysis_uses_total_and_net_abc_with_inertial_power(self):
        frame = pd.DataFrame({"t": [0.0, 1.0, 2.0], "v": [0.0, 10.0, 10.0]})
        analysis = build_cycle_power_analysis(
            frame,
            [
                {
                    "id": "baseline",
                    "label": "Baseline",
                    "mass_kg": 1600.0,
                    "total": {"A": 100.0, "B": 0.0, "C": 0.0},
                    "net": {"A": 80.0, "B": 0.0, "C": 0.0},
                }
            ],
        )

        self.assertEqual(len(analysis["series"]), 2)
        total = next(item for item in analysis["series"] if item["boundary"] == "TOTAL")
        net = next(item for item in analysis["series"] if item["boundary"] == "NET")
        self.assertGreater(total["demanded_power_kw"][1], net["demanded_power_kw"][1])
        self.assertIsNotNone(total["inertial_power_kw"])
        self.assertFalse(analysis["decomposition_available"])


if __name__ == "__main__":
    unittest.main()
