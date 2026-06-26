import unittest

import pandas as pd

from src.vde_core.roadload.models import EquivalentABC
from src.vde_core.vde_setup_service import (
    build_test_mass_hint,
    build_delta_mode_ctx_updates,
    build_compute_vde_from_ctx,
    build_edit_core_update,
    resolve_tire_calculation_mass,
    build_vde_phase_update,
    build_vde_insert_row,
    resolve_test_mass_kg,
)


class VdeSetupServiceTests(unittest.TestCase):
    def test_resolve_test_mass_kg_defaults_epa_to_curb_plus_136(self):
        self.assertEqual(
            resolve_test_mass_kg({"legislation": "EPA", "mass_kg": 1550.0}),
            1686.0,
        )

    def test_resolve_test_mass_kg_uses_wltp_formula_when_inputs_are_available(self):
        self.assertEqual(
            resolve_test_mass_kg(
                {
                    "legislation": "WLTP",
                    "mass_kg": 1500.0,
                    "mro_kg": 1575.0,
                    "tpmlm_kg": 2000.0,
                    "options_kg": 50.0,
                    "wltp_category": 1,
                }
            ),
            1702.5,
        )

    def test_resolve_test_mass_kg_wltp_falls_back_to_curb_weight_without_tpmlm(self):
        self.assertEqual(
            resolve_test_mass_kg({"legislation": "WLTP", "mass_kg": 1500.0}),
            1500.0,
        )

    def test_resolve_test_mass_kg_prefers_manual_value(self):
        self.assertEqual(
            resolve_test_mass_kg({"legislation": "EPA", "mass_kg": 1550.0, "test_mass_kg": 1705.0}),
            1705.0,
        )

    def test_resolve_test_mass_kg_rejects_manual_value_below_curb_weight(self):
        with self.assertRaisesRegex(ValueError, "Test mass cannot be lower than curb weight"):
            resolve_test_mass_kg({"legislation": "EPA", "mass_kg": 1550.0, "test_mass_kg": 1500.0})

    def test_build_test_mass_hint_returns_epa_default_message(self):
        self.assertEqual(
            build_test_mass_hint({"legislation": "EPA"}),
            "default Curb +300 pounds / 136 kg",
        )

    def test_resolve_tire_calculation_mass_for_epa_twc_uses_current_curb_weight(self):
        resolved = resolve_tire_calculation_mass(
            {
                "legislation": "EPA",
                "mass_kg": 1550.0,
                "inertia_class": 1644.0,
                "tire_load_mass_basis": "TWC",
            }
        )

        self.assertEqual(resolved["basis"], "TWC")
        self.assertEqual(resolved["mass_kg"], 1701.0)
        self.assertEqual(resolved["source"], "inertia_class_from_mass")

    def test_build_delta_mode_ctx_updates_prefers_baseline_values(self):
        base = {
            "A": 120.0,
            "B": 0.02,
            "C": 0.011,
            "mass_kg": 1550.0,
            "crr1_frac_at_120kph": 0.12,
            "rrc_N_per_kN": 9.7,
            "tire_size": "225/50R17",
        }

        updates = build_delta_mode_ctx_updates(base)

        self.assertEqual(updates["A"], 120.0)
        self.assertEqual(updates["B"], 0.02)
        self.assertEqual(updates["C"], 0.011)
        self.assertEqual(updates["mass_kg"], 1550.0)
        self.assertEqual(updates["crr1_frac_at_120kph"], 0.12)
        self.assertEqual(updates["rrc_N_per_kN"], 9.7)
        self.assertEqual(updates["tire_size"], "225/50R17")

    def test_build_compute_vde_from_ctx_returns_positive_preview_and_deltas(self):
        ctx = {
            "legislation": "EPA",
            "category": "MIDSIZE",
            "A": 120.0,
            "B": 0.02,
            "C": 0.011,
            "mass_kg": 1550.0,
            "delta_rr_N": 1.0,
            "delta_brake_N": 2.0,
            "delta_parasitics_N": 3.0,
            "delta_aero_cdA": -0.02,
            "delta_mass_kg": 80.0,
            "cycle_df": pd.DataFrame(
                {
                    "t": [0.0, 1.0, 2.0, 3.0],
                    "v": [0.0, 8.0, 10.0, 10.0],
                }
            ),
        }

        result = build_compute_vde_from_ctx(ctx)

        self.assertTrue(result["ok"])
        self.assertGreater(result["total_mj_km"], 0.0)
        self.assertAlmostEqual(result["equiv"].mass_kg, 1766.0)
        self.assertEqual(result["deltas"]["delta_rr_N"], 1.0)
        self.assertEqual(result["deltas"]["delta_brake_N"], 2.0)
        self.assertEqual(result["deltas"]["delta_parasitics_N"], 3.0)
        self.assertEqual(result["deltas"]["delta_mass_kg"], 80.0)
        self.assertIn("delta_aero_Npkph2", result["deltas"])

    def test_build_vde_insert_row_merges_context_and_filters_empty_values(self):
        equiv = EquivalentABC(
            A=121.0,
            B=0.021,
            C=0.012,
            mass_kg=1630.0,
            component_table=[{"name": "roadload_total", "A": 121.0, "B": 0.021, "C": 0.012}],
        )
        ctx = {
            "mass_kg": 1550.0,
            "engine_type": "ICE",
            "transmission_type": "AT",
            "tire_size": "225/50R17",
            "notes": "",
            "vde_id_parent": 77,
            "baseline_dict": {
                "A": 120.0,
                "B": 0.02,
                "C": 0.011,
                "mass_kg": 1550.0,
            },
        }

        row = build_vde_insert_row(
            ctx,
            leg="EPA",
            cat="MIDSIZE",
            make="FORD",
            model="TEST",
            year=2025,
            notes="snapshot",
            cycle_name="FTP75_HWFET",
            cycle_source="standard",
            equiv=equiv,
            total_mj_km=1.234,
            by_phase={"city": 1.1, "hwy": 1.4},
            deltas={
                "delta_rr_N": 1.0,
                "delta_brake_N": 2.0,
                "delta_parasitics_N": 3.0,
                "delta_aero_Npkph2": -0.001,
                "delta_mass_kg": 80.0,
            },
            decomp={"rr_alpha_N": 9.0, "parasitic_A_coef_N": 4.0},
        )

        self.assertEqual(row["legislation"], "EPA")
        self.assertEqual(row["coast_A_N"], 121.0)
        self.assertEqual(row["mass_kg"], 1550.0)
        self.assertEqual(row["test_mass_kg"], 1686.0)
        self.assertEqual(row["vde_urb_mj_per_km"], 1.1)
        self.assertEqual(row["vde_hw_mj_per_km"], 1.4)
        self.assertEqual(row["vde_id_parent"], 77)
        self.assertEqual(row["baseline_A_N"], 120.0)
        self.assertEqual(row["engine_type"], "ICE")
        self.assertEqual(row["transmission_type"], "AT")
        self.assertEqual(row["rr_alpha_N"], 9.0)
        self.assertEqual(row["parasitic_A_coef_N"], 4.0)
        self.assertNotIn("driveline_eff", row)

    def test_build_edit_core_update_includes_test_mass_when_present(self):
        payload = build_edit_core_update(
            A=120.0,
            B=0.02,
            C=0.011,
            mass_kg=1550.0,
            test_mass_kg=1686.0,
            make="FORD",
            model="TEST",
            year=2025,
            notes="snapshot",
        )

        self.assertEqual(payload["mass_kg"], 1550.0)
        self.assertEqual(payload["test_mass_kg"], 1686.0)
        self.assertEqual(payload["make"], "FORD")

    def test_build_vde_phase_update_for_epa_maps_expected_keys(self):
        df_cycle = pd.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "v": [8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 12.0, 12.0, 12.0],
                "phase": ["bag1", "bag1", "bag1", "bag2", "bag2", "bag2", "hwfet", "hwfet", "hwfet"],
            }
        )

        upd = build_vde_phase_update(df_cycle, "EPA", A=120.0, B=0.02, C=0.011, mass_kg=1550.0)

        self.assertIn("vde_urb_mj", upd)
        self.assertIn("vde_hw_mj", upd)
        self.assertIn("vde_net_mj_per_km", upd)
        self.assertGreater(upd["vde_net_mj_per_km"], 0.0)


if __name__ == "__main__":
    unittest.main()
