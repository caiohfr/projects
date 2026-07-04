import unittest

import pandas as pd

from src.vde_app.plots import compute_roadload_curve, roadload_curve_comparison_chart
from src.vde_app.units import (
    UNIT_SYSTEM_METRIC,
    UNIT_SYSTEM_US,
    format_quantity,
    to_canonical,
    to_display,
    unit_label,
)
from src.vde_core.vde_workflow_service import build_vde_setup_preview_from_ctx


class VdeAppUnitsTests(unittest.TestCase):
    def test_compute_roadload_curve_metric_display(self):
        curve_df = compute_roadload_curve(100.0, 0.5, 0.04, unit_system=UNIT_SYSTEM_METRIC)

        self.assertEqual(curve_df["speed_unit"].iloc[0], "km/h")
        self.assertEqual(curve_df["force_unit"].iloc[0], "N")
        self.assertEqual(curve_df["speed_display"].iloc[0], 0.0)
        self.assertEqual(curve_df["speed_display"].iloc[-1], 160.0)
        self.assertAlmostEqual(curve_df["force_display"].iloc[0], 100.0, places=9)
        self.assertAlmostEqual(curve_df["force_display"].iloc[-1], 1204.0, places=9)
        self.assertAlmostEqual(curve_df["power_kW"].iloc[-1], 53.5111111111, places=6)

    def test_compute_roadload_curve_us_display(self):
        curve_df = compute_roadload_curve(100.0, 0.5, 0.04, unit_system=UNIT_SYSTEM_US)

        self.assertEqual(curve_df["speed_unit"].iloc[0], "mph")
        self.assertEqual(curve_df["force_unit"].iloc[0], "lbf")
        self.assertEqual(curve_df["speed_display"].iloc[0], 0.0)
        self.assertEqual(curve_df["speed_display"].iloc[-1], 100.0)
        self.assertAlmostEqual(curve_df["speed_kph"].iloc[-1], 160.9344, places=6)
        self.assertAlmostEqual(curve_df["force_N"].iloc[-1], 1216.4624441344001, places=6)
        self.assertAlmostEqual(curve_df["force_display"].iloc[-1], 273.4716363863451, places=6)

    def test_compute_roadload_curve_accepts_us_and_si_aliases(self):
        us_curve_df = compute_roadload_curve(100.0, 0.5, 0.04, unit_system="US")
        si_curve_df = compute_roadload_curve(100.0, 0.5, 0.04, unit_system="SI")

        self.assertEqual(us_curve_df["speed_unit"].iloc[0], "mph")
        self.assertEqual(si_curve_df["speed_unit"].iloc[0], "km/h")

    def test_roadload_curve_comparison_chart_renders_total_and_net(self):
        fig = roadload_curve_comparison_chart(
            [
                {"label": "ABC_TOTAL", "A_N": 100.0, "B_N_per_kph": 0.5, "C_N_per_kph2": 0.04},
                {"label": "ABC_NET", "A_N": 90.0, "B_N_per_kph": 0.45, "C_N_per_kph2": 0.038},
            ],
            unit_system=UNIT_SYSTEM_US,
        )

        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)
        self.assertEqual({trace.name for trace in fig.data}, {"ABC_TOTAL", "ABC_NET"})
        self.assertEqual(fig.layout.xaxis.title.text, "Vehicle Speed [mph]")
        self.assertEqual(fig.layout.yaxis.title.text, "Road Load Force [lbf]")

    def test_roundtrip_conversions(self):
        cases = [
            ("mass", 1550.0),
            ("force", 120.0),
            ("force_per_speed", 0.021),
            ("force_per_speed_squared", 0.0115),
            ("cda", 0.66),
            ("pressure", 32.0),
            ("speed", 96.0),
            ("energy_per_distance", 1.85),
            ("rrc", 9.5),
        ]
        for quantity, canonical in cases:
            with self.subTest(quantity=quantity):
                display_value = to_display(canonical, quantity, UNIT_SYSTEM_US)
                roundtrip = to_canonical(display_value, quantity, UNIT_SYSTEM_US)
                self.assertAlmostEqual(roundtrip, canonical, places=9)

    def test_rrc_keeps_same_numeric_value_between_metric_and_us(self):
        canonical = 9.5
        self.assertAlmostEqual(to_display(canonical, "rrc", UNIT_SYSTEM_METRIC), canonical, places=9)
        self.assertAlmostEqual(to_display(canonical, "rrc", UNIT_SYSTEM_US), canonical, places=9)
        self.assertEqual(unit_label("rrc", UNIT_SYSTEM_METRIC), "N/kN")
        self.assertEqual(unit_label("rrc", UNIT_SYSTEM_US), "lbf/klbf")

    def test_format_quantity_uses_selected_display_unit(self):
        self.assertEqual(format_quantity(1550.0, "mass", UNIT_SYSTEM_METRIC, format_str="%.1f"), "1550.0 kg")
        self.assertEqual(format_quantity(1550.0, "mass", UNIT_SYSTEM_US, format_str="%.1f"), "3417.2 lb")

    def _base_ctx(self):
        return {
            "legislation": "EPA",
            "category": "MIDSIZE",
            "make": "FORD",
            "model": "TEST",
            "year": 2025,
            "mode": "New line (manual / test)",
            "abc_total_source_ui": "From test coastdown",
            "A": 120.0,
            "B": 0.02,
            "C": 0.011,
            "mass_kg": 1550.0,
            "test_mass_kg": 1686.0,
            "tire_load_mass_basis": "TEST_MASS",
            "weight_dist_fr_pct": 50.0,
            "cda_m2": 0.66,
            "front_pressure_psi": 32.0,
            "rear_pressure_psi": 32.0,
            "rrc_N_per_kN": 9.5,
            "delta_rr_N": 0.0,
            "delta_brake_N": 0.0,
            "delta_parasitics_N": 0.0,
            "delta_aero_cdA": 0.0,
            "transmission_losses_source": "Manual",
            "trans_A_coef_N": 10.0,
            "trans_B_coef_Npkph": 0.005,
            "trans_C_coef_Npkph2": 0.001,
            "cycle_df": pd.DataFrame(
                {
                    "t": [0.0, 1.0, 2.0, 3.0],
                    "v": [0.0, 8.0, 10.0, 10.0],
                }
            ),
        }

    def test_metric_and_us_inputs_resolve_to_same_canonical_preview(self):
        base_ctx = self._base_ctx()
        us_ctx = dict(base_ctx)
        quantity_by_field = {
            "A": "force",
            "B": "force_per_speed",
            "C": "force_per_speed_squared",
            "mass_kg": "mass",
            "test_mass_kg": "mass",
            "cda_m2": "cda",
            "front_pressure_psi": "pressure",
            "rear_pressure_psi": "pressure",
            "rrc_N_per_kN": "rrc",
            "trans_A_coef_N": "force",
            "trans_B_coef_Npkph": "force_per_speed",
            "trans_C_coef_Npkph2": "force_per_speed_squared",
        }
        for field, quantity in quantity_by_field.items():
            display_value = to_display(base_ctx[field], quantity, UNIT_SYSTEM_US)
            us_ctx[field] = to_canonical(display_value, quantity, UNIT_SYSTEM_US)

        metric_preview = build_vde_setup_preview_from_ctx(base_ctx)
        us_preview = build_vde_setup_preview_from_ctx(us_ctx)

        self.assertTrue(metric_preview["ok"])
        self.assertTrue(us_preview["ok"])
        self.assertAlmostEqual(us_ctx["mass_kg"], base_ctx["mass_kg"], places=9)
        self.assertAlmostEqual(metric_preview["abc_total"]["A"], us_preview["abc_total"]["A"], places=9)
        self.assertAlmostEqual(metric_preview["abc_net"]["A"], us_preview["abc_net"]["A"], places=9)
        self.assertAlmostEqual(metric_preview["vde_total"]["mj_per_km"], us_preview["vde_total"]["mj_per_km"], places=9)
        self.assertAlmostEqual(metric_preview["vde_net"]["mj_per_km"], us_preview["vde_net"]["mj_per_km"], places=9)


if __name__ == "__main__":
    unittest.main()
