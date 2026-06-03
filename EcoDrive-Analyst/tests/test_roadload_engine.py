import math
import unittest

from src.vde_core.roadload import (
    build_request_from_manual_inputs,
    cdA_to_C,
    run_roadload_scenario,
)


class RoadloadEngineTests(unittest.TestCase):
    def test_run_without_deltas_returns_baseline(self):
        req = build_request_from_manual_inputs(
            A=120,
            B=0.02,
            C=0.011,
            mass_kg=1550,
            legislation="EPA",
            category="MIDSIZE",
        )

        equiv = run_roadload_scenario(req)

        self.assertAlmostEqual(equiv.A, 120.0)
        self.assertAlmostEqual(equiv.B, 0.02)
        self.assertAlmostEqual(equiv.C, 0.011)
        self.assertAlmostEqual(equiv.mass_kg, 1550.0)
        self.assertEqual(len(equiv.component_table), 1)
        self.assertEqual(equiv.component_table[0]["name"], "roadload_total")

    def test_delta_mass_changes_only_mass(self):
        req = build_request_from_manual_inputs(
            A=120,
            B=0.02,
            C=0.011,
            mass_kg=1550,
            legislation="EPA",
            category="MIDSIZE",
            delta_mass_kg=80,
        )

        equiv = run_roadload_scenario(req)

        self.assertAlmostEqual(equiv.A, 120.0)
        self.assertAlmostEqual(equiv.B, 0.02)
        self.assertAlmostEqual(equiv.C, 0.011)
        self.assertAlmostEqual(equiv.mass_kg, 1630.0)

    def test_delta_abc_accumulates_across_components(self):
        req = build_request_from_manual_inputs(
            A=120,
            B=0.02,
            C=0.011,
            mass_kg=1550,
            tire_delta_A=1.5,
            tire_delta_B=0.003,
            brake_delta_A=0.5,
            brake_delta_B=-0.002,
            brake_delta_C=0.001,
            parasitic_delta_C=0.002,
        )

        equiv = run_roadload_scenario(req)

        self.assertAlmostEqual(equiv.A, 122.0)
        self.assertAlmostEqual(equiv.B, 0.021)
        self.assertAlmostEqual(equiv.C, 0.014)

    def test_delta_cda_maps_into_c(self):
        req = build_request_from_manual_inputs(
            A=120,
            B=0.02,
            C=0.011,
            mass_kg=1550,
            delta_cda_m2=-0.02,
        )

        equiv = run_roadload_scenario(req)
        expected_c = 0.011 + cdA_to_C(-0.02)

        self.assertTrue(math.isclose(equiv.C, expected_c, rel_tol=1e-9, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
