import unittest
from unittest.mock import patch

import pandas as pd

from src.vde_core.vde_workflow_service import (
    build_vde_pre_save_review,
    build_vde_setup_preview,
    build_vde_setup_preview_from_ctx,
    build_vde_workflow_payload_from_ctx,
    save_vde_setup_result,
    summarize_component_build_up_from_ctx,
)


class VdeWorkflowServiceTests(unittest.TestCase):
    def _base_payload(self):
        return {
            "legislation": "EPA",
            "category": "MIDSIZE",
            "make": "FORD",
            "model": "TEST",
            "year": 2025,
            "mass_kg": 1550.0,
            "test_mass_kg": 1686.0,
            "mass_basis": "TEST_MASS",
            "initial_abc_total_source": "manual",
            "initial_abc_total": {"A": 120.0, "B": 0.02, "C": 0.011},
            "cycle_df": pd.DataFrame(
                {
                    "t": [0.0, 1.0, 2.0, 3.0],
                    "v": [0.0, 8.0, 10.0, 10.0],
                }
            ),
        }

    @patch("src.vde_core.vde_workflow_service.insert_vde_row")
    @patch("src.vde_core.vde_workflow_service.update_vde_by_id")
    @patch("src.vde_core.vde_workflow_service.delete_vde_by_id")
    def test_build_vde_setup_preview_does_not_persist(
        self,
        mock_delete,
        mock_update,
        mock_insert,
    ):
        preview = build_vde_setup_preview(self._base_payload())

        self.assertTrue(preview["ok"])
        mock_insert.assert_not_called()
        mock_update.assert_not_called()
        mock_delete.assert_not_called()

    def test_build_vde_setup_preview_uses_total_abc_as_vde_total_source(self):
        preview = build_vde_setup_preview(self._base_payload())

        self.assertEqual(preview["abc_total"]["A"], 120.0)
        self.assertIsNotNone(preview["vde_total"])
        self.assertGreater(preview["vde_total"]["mj_per_km"], 0.0)
        self.assertIsNone(preview["abc_net"])
        self.assertIsNone(preview["vde_net"])
        self.assertIn("vde_net_unavailable_transmission_losses_missing", preview["warnings"])

    def test_build_vde_setup_preview_subtracts_transmission_losses_for_net(self):
        payload = self._base_payload()
        payload["transmission_losses"] = {
            "source": "manual",
            "A_TRANS": 10.0,
            "B_TRANS": 0.005,
            "C_TRANS": 0.001,
        }

        preview = build_vde_setup_preview(payload)

        self.assertEqual(preview["transmission_losses"]["status"], "available")
        self.assertAlmostEqual(preview["abc_net"]["A"], 110.0)
        self.assertAlmostEqual(preview["abc_net"]["B"], 0.015)
        self.assertAlmostEqual(preview["abc_net"]["C"], 0.01)
        self.assertIsNotNone(preview["vde_net"])
        self.assertGreater(preview["vde_total"]["mj_per_km"], preview["vde_net"]["mj_per_km"])

    def test_build_vde_setup_preview_accepts_nested_resolved_transmission_shape(self):
        payload = self._base_payload()
        payload["transmission_losses"] = {
            "source": "INHERITED",
            "status": "available",
            "abc": {
                "A": 10.0,
                "B": 0.005,
                "C": 0.001,
            },
        }

        preview = build_vde_setup_preview(payload)

        self.assertEqual(preview["transmission_losses"]["status"], "available")
        self.assertAlmostEqual(preview["abc_net"]["A"], 110.0)
        self.assertAlmostEqual(preview["abc_net"]["B"], 0.015)
        self.assertAlmostEqual(preview["abc_net"]["C"], 0.01)

    def test_build_vde_setup_preview_warns_when_weight_distribution_missing(self):
        payload = self._base_payload()
        preview = build_vde_setup_preview(payload)

        self.assertEqual(preview["mass_setup"]["weight_dist_fr_pct"], 50.0)
        self.assertIn("weight_distribution_missing_default_50pct", preview["warnings"])

    def test_build_vde_setup_preview_warns_when_twc_missing_for_non_epa(self):
        payload = self._base_payload()
        payload["legislation"] = "WLTP"
        payload["mass_basis"] = "TWC"
        payload["test_mass_kg"] = 1600.0

        preview = build_vde_setup_preview(payload)

        self.assertIsNone(preview["mass_setup"]["resolved_mass_used_kg"])
        self.assertIn("twc_selected_but_inertia_class_missing", preview["warnings"])

    def test_build_vde_setup_preview_keeps_epa_vde_mass_on_twc_even_when_test_mass_differs(self):
        payload = self._base_payload()
        payload["inertia_class"] = 1750.0
        payload["test_mass_kg"] = 2400.0

        preview = build_vde_setup_preview(payload)

        self.assertEqual(preview["mass_setup"]["vde_mass_basis"], "TWC")
        self.assertEqual(preview["mass_setup"]["resolved_mass_used_kg"], 1750.0)
        self.assertEqual(preview["mass_setup"]["test_mass_kg"], 2400.0)

    def test_build_vde_setup_preview_reuses_baseline_total_source(self):
        payload = self._base_payload()
        payload["initial_abc_total_source"] = "baseline"
        payload["baseline_row"] = {
            "id": 10,
            "coast_A_N": 130.0,
            "coast_B_N_per_kph": 0.03,
            "coast_C_N_per_kph2": 0.012,
            "vde_net_mj_per_km": 1.9,
        }

        preview = build_vde_setup_preview(payload)

        self.assertEqual(preview["line_source"]["baseline_id"], 10)
        self.assertEqual(preview["abc_total"]["A"], 130.0)
        self.assertIn("legacy_vde_net_used_as_total_candidate", preview["warnings"])

    def test_build_vde_setup_preview_adds_total_components_on_top_of_baseline_source(self):
        payload = self._base_payload()
        payload["initial_abc_total_source"] = "baseline"
        payload["baseline_row"] = {
            "coast_A_N": 130.0,
            "coast_B_N_per_kph": 0.03,
            "coast_C_N_per_kph2": 0.012,
        }
        payload["components"] = {
            "rr_delta": {"role": "TOTAL_COMPONENT", "A": 5.0, "B": 0.001, "C": 0.0},
            "aero_delta": {"role": "TOTAL_COMPONENT", "A": 0.0, "B": 0.0, "C": 0.002},
        }

        preview = build_vde_setup_preview(payload)

        self.assertEqual(preview["initial_abc_total_base"]["A"], 130.0)
        self.assertAlmostEqual(preview["component_abc_total"]["A"], 5.0)
        self.assertAlmostEqual(preview["abc_total"]["A"], 135.0)
        self.assertAlmostEqual(preview["abc_total"]["B"], 0.031)
        self.assertAlmostEqual(preview["abc_total"]["C"], 0.014)

    def test_build_vde_setup_preview_reuses_component_build_up_source(self):
        payload = self._base_payload()
        payload["initial_abc_total_source"] = "component_build_up"
        payload["components"] = {
            "tires": {"role": "TOTAL_COMPONENT", "A": 50.0, "B": 0.01, "C": 0.003},
            "aero": {"role": "TOTAL_COMPONENT", "A": 0.0, "B": 0.0, "C": 0.006},
            "transmission": {"role": "NET_SUBTRACTION", "A": 8.0, "B": 0.0, "C": 0.0},
        }

        preview = build_vde_setup_preview(payload)

        self.assertAlmostEqual(preview["abc_total"]["A"], 50.0)
        self.assertAlmostEqual(preview["abc_total"]["B"], 0.01)
        self.assertAlmostEqual(preview["abc_total"]["C"], 0.009)

    def test_build_vde_workflow_payload_from_ctx_maps_delta_inputs(self):
        payload = build_vde_workflow_payload_from_ctx(
            {
                "vde_id_parent": 41,
                "from_delta": "Deltas",
                "mode": "From baseline (editable)",
                "A": 120.0,
                "B": 0.02,
                "C": 0.011,
                "mass_kg": 1550.0,
                "test_mass_kg": 1686.0,
                "legislation": "EPA",
                "category": "MIDSIZE",
                "delta_rr_N": 6.0,
                "crr1_frac_at_120kph": 0.12,
                "delta_brake_N": 2.0,
                "delta_parasitics_N": 3.0,
                "delta_aero_cdA": 0.01,
                "delta_mass_kg": 45.0,
                "trans_A_coef_N": 9.0,
                "trans_B_Npkph": 0.004,
            }
        )

        self.assertEqual(payload["initial_abc_total_source"], "BASELINE")
        self.assertEqual(payload["line_source_mode"], "BASELINE")
        self.assertAlmostEqual(payload["mass_kg"], 1595.0)
        self.assertIn("rolling_resistance_delta", payload["components"])
        self.assertIn("brakes_delta", payload["components"])
        self.assertIn("parasitics_delta", payload["components"])
        self.assertIn("aero_delta", payload["components"])
        self.assertEqual(payload["transmission_losses"]["source"], "MANUAL")
        self.assertAlmostEqual(payload["transmission_losses"]["B_TRANS"], 0.004)

    def test_build_vde_workflow_payload_from_ctx_uses_component_build_up_for_manual_editor(self):
        payload = build_vde_workflow_payload_from_ctx(
            {
                "mode": "Define all parameters (no baseline)",
                "tire_component_source": "Manual RR",
                "rr_alpha_N": 90.0,
                "rr_beta_Npkph": 0.4,
                "aero_C_coef_Npkph2": 0.011,
                "parasitic_A_coef_N": 3.0,
                "brake_A_coef_N": 2.0,
            }
        )

        self.assertEqual(payload["initial_abc_total_source"], "COMPONENT_BUILD_UP")
        self.assertIn("tires_manual_rr", payload["components"])
        self.assertIn("aerodynamics", payload["components"])
        self.assertIn("parasitics", payload["components"])
        self.assertIn("brakes", payload["components"])

    def test_build_vde_setup_preview_from_ctx_builds_preview(self):
        preview = build_vde_setup_preview_from_ctx(
            {
                "mode": "From baseline (editable)",
                "from_delta": "Deltas",
                "A": 120.0,
                "B": 0.02,
                "C": 0.011,
                "mass_kg": 1550.0,
                "test_mass_kg": 1686.0,
                "legislation": "EPA",
                "category": "MIDSIZE",
                "cycle_df": pd.DataFrame(
                    {
                        "t": [0.0, 1.0, 2.0, 3.0],
                        "v": [0.0, 8.0, 10.0, 10.0],
                    }
                ),
                "delta_rr_N": 5.0,
                "crr1_frac_at_120kph": 0.12,
            }
        )

        self.assertTrue(preview["ok"])
        self.assertAlmostEqual(preview["abc_total"]["A"], 125.0)
        self.assertAlmostEqual(preview["abc_total"]["B"], 0.025)

    def test_build_vde_setup_preview_from_ctx_uses_component_build_up_totals(self):
        preview = build_vde_setup_preview_from_ctx(
            {
                "mode": "Define all parameters (no baseline)",
                "legislation": "EPA",
                "category": "MIDSIZE",
                "mass_kg": 1550.0,
                "test_mass_kg": 1686.0,
                "rr_alpha_N": 95.0,
                "rr_beta_Npkph": 0.45,
                "tire_component_source": "Manual RR",
                "aero_C_coef_Npkph2": 0.012,
                "parasitic_A_coef_N": 4.0,
                "brake_A_coef_N": 1.0,
                "cycle_df": pd.DataFrame(
                    {
                        "t": [0.0, 1.0, 2.0, 3.0],
                        "v": [0.0, 8.0, 10.0, 10.0],
                    }
                ),
            }
        )

        self.assertTrue(preview["ok"])
        self.assertEqual(preview["initial_abc_total_source"], "COMPONENT_BUILD_UP")
        self.assertAlmostEqual(preview["abc_total"]["A"], 100.0)
        self.assertAlmostEqual(preview["abc_total"]["B"], 0.45)
        self.assertAlmostEqual(preview["abc_total"]["C"], 0.012)

    def test_summarize_component_build_up_from_ctx_returns_component_total(self):
        summary = summarize_component_build_up_from_ctx(
            {
                "mode": "Define all parameters (no baseline)",
                "tire_component_source": "Manual RR",
                "rr_alpha_N": 90.0,
                "rr_beta_Npkph": 0.4,
                "aero_C_coef_Npkph2": 0.01,
                "parasitic_A_coef_N": 3.0,
                "brake_A_coef_N": 2.0,
            }
        )

        self.assertTrue(summary["enabled"])
        self.assertAlmostEqual(summary["abc_total"]["A"], 95.0)
        self.assertAlmostEqual(summary["abc_total"]["B"], 0.4)
        self.assertAlmostEqual(summary["abc_total"]["C"], 0.01)

    def test_build_vde_setup_preview_from_ctx_prepares_rich_save_payload(self):
        preview = build_vde_setup_preview_from_ctx(
            {
                "A": 120.0,
                "B": 0.02,
                "C": 0.011,
                "mass_kg": 1550.0,
                "test_mass_kg": 1686.0,
                "legislation": "EPA",
                "category": "MIDSIZE",
                "make": "FORD",
                "model": "TEST",
                "year": 2025,
                "engine_type": "ICE",
                "transmission_type": "AT",
                "delta_mass_kg": 45.0,
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
        )

        row = preview["save_payload"]["insert_row"]
        self.assertEqual(row["engine_type"], "ICE")
        self.assertEqual(row["transmission_type"], "AT")
        self.assertEqual(row["delta_mass_kg"], 45.0)
        self.assertEqual(row["coast_A_N"], 120.0)
        self.assertAlmostEqual(row["trans_A_coef_N"], 10.0)
        self.assertIn("vde_total_mj_per_km", row)
        self.assertIn("vde_net_mj_per_km", row)
        self.assertGreater(row["vde_total_mj_per_km"], row["vde_net_mj_per_km"])

    def test_build_vde_setup_preview_respects_manual_test_mass_rule(self):
        payload = self._base_payload()
        payload["test_mass_kg"] = 1500.0

        with self.assertRaisesRegex(ValueError, "Test mass cannot be lower than curb weight"):
            build_vde_setup_preview(payload)

    def test_build_vde_setup_preview_aligns_epa_inertia_mass_before_validation(self):
        payload = self._base_payload()
        payload["mass_kg"] = 2416.0
        payload["test_mass_kg"] = 2268.0
        payload["test_mass_basis"] = "EPA_INERTIA_CLASS"
        payload["inertia_class"] = 2268.0
        payload["mass_intention"] = "EPA_PLUS_1_TWC"

        preview = build_vde_setup_preview(payload)

        self.assertTrue(preview["ok"])
        self.assertEqual(preview["mass_setup"]["mass_kg"], 2189.0)
        self.assertEqual(preview["mass_setup"]["test_mass_kg"], 2268.0)

    @patch("src.vde_core.vde_workflow_service.insert_vde_row")
    def test_save_vde_setup_result_inserts_new_row(self, mock_insert):
        mock_insert.return_value = 321
        preview = build_vde_setup_preview(self._base_payload())

        result = save_vde_setup_result(preview, "insert_new")

        self.assertEqual(result["action"], "insert_new")
        self.assertEqual(result["vde_id"], 321)
        self.assertEqual(result["row"]["coast_A_N"], 120.0)
        self.assertIn("vde_total_mj_per_km", result["row"])
        mock_insert.assert_called_once()

    @patch("src.vde_core.vde_workflow_service.insert_vde_row")
    def test_save_vde_setup_result_with_ctx_preserves_rich_snapshot_fields(self, mock_insert):
        mock_insert.return_value = 654
        ctx = {
            "A": 120.0,
            "B": 0.02,
            "C": 0.011,
            "mass_kg": 1550.0,
            "test_mass_kg": 1686.0,
            "legislation": "EPA",
            "category": "MIDSIZE",
            "make": "FORD",
            "model": "TEST",
            "year": 2025,
            "engine_type": "ICE",
            "transmission_type": "AT",
            "delta_rr_N": 2.0,
            "trans_A_coef_N": 8.0,
            "trans_B_coef_Npkph": 0.004,
            "cycle_df": pd.DataFrame(
                {
                    "t": [0.0, 1.0, 2.0, 3.0],
                    "v": [0.0, 8.0, 10.0, 10.0],
                }
            ),
        }
        preview = build_vde_setup_preview_from_ctx(ctx)

        result = save_vde_setup_result(preview, "insert_new", ctx=ctx)

        self.assertEqual(result["vde_id"], 654)
        self.assertEqual(result["row"]["engine_type"], "ICE")
        self.assertEqual(result["row"]["coast_A_N"], 122.0)
        self.assertAlmostEqual(result["row"]["trans_A_coef_N"], 8.0)
        self.assertIn("vde_total_mj_per_km", result["row"])
        self.assertIn("vde_net_mj_per_km", result["row"])
        mock_insert.assert_called_once()

    @patch("src.vde_core.vde_workflow_service.update_vde_by_id")
    def test_save_vde_setup_result_updates_existing_row(self, mock_update):
        preview = build_vde_setup_preview({**self._base_payload(), "target_vde_id": 77})

        result = save_vde_setup_result(preview, "update_existing")

        self.assertEqual(result["action"], "update_existing")
        self.assertEqual(result["vde_id"], 77)
        mock_update.assert_called_once()

    @patch("src.vde_core.vde_workflow_service.delete_vde_by_id")
    def test_save_vde_setup_result_deletes_only_on_explicit_mode(self, mock_delete):
        mock_delete.return_value = 1
        preview = build_vde_setup_preview({**self._base_payload(), "target_vde_id": 88})

        result = save_vde_setup_result(preview, "delete_existing")

        self.assertEqual(result["action"], "delete_existing")
        self.assertEqual(result["deleted_rows"], 1)
        mock_delete.assert_called_once_with(88)

    def test_save_vde_setup_result_rejects_deactivate_without_schema_support(self):
        preview = build_vde_setup_preview(self._base_payload())

        with self.assertRaisesRegex(ValueError, "deactivate_existing is not supported"):
            save_vde_setup_result(preview, "deactivate_existing")

    def test_build_vde_pre_save_review_for_baseline_with_tire_delta(self):
        ctx = {
            "mode": "From baseline (editable)",
            "from_delta": "Deltas",
            "abc_total_source_ui": "Baseline ABC",
            "baseline_dict": {
                "A": 120.0,
                "B": 0.02,
                "C": 0.011,
                "mass_kg": 1550.0,
                "test_mass_kg": 1686.0,
                "legislation": "EPA",
                "category": "MIDSIZE",
            },
            "mass_kg": 1550.0,
            "test_mass_kg": 1686.0,
            "legislation": "EPA",
            "category": "MIDSIZE",
            "delta_rr_N": 5.0,
            "crr1_frac_at_120kph": 0.12,
            "component_mode_tires": "Apply delta",
            "tire_scenario_application": "Manual Delta RR",
            "cycle_df": pd.DataFrame(
                {
                    "t": [0.0, 1.0, 2.0, 3.0],
                    "v": [0.0, 8.0, 10.0, 10.0],
                }
            ),
        }

        preview = build_vde_setup_preview_from_ctx(ctx)
        review = build_vde_pre_save_review(ctx, preview, preview["save_payload"])

        self.assertEqual(review["reference_snapshot"]["kind"], "Baseline Snapshot")
        self.assertEqual(review["change_summary"]["roadload_basis"]["state"], "Inherited")
        self.assertEqual(review["change_summary"]["tires"]["state"], "Applied")
        self.assertIs(review["staged_save_payload"], preview["save_payload"])
        abc_total_row = next(row for row in review["baseline_vs_working_rows"] if row["Field"] == "ABC_TOTAL")
        self.assertEqual(abc_total_row["Reference"], "120.000 / 0.020000 / 0.01100000")
        self.assertEqual(abc_total_row["Working scenario"], "125.000 / 0.025000 / 0.01100000")

    def test_build_vde_pre_save_review_for_manual_coastdown(self):
        ctx = {
            "mode": "New line (manual / test)",
            "abc_total_source_ui": "From test coastdown",
            "A": 130.0,
            "B": 0.03,
            "C": 0.012,
            "mass_kg": 1600.0,
            "test_mass_kg": 1736.0,
            "legislation": "EPA",
            "category": "MIDSIZE",
            "cycle_name": "FTP",
            "cycle_df": pd.DataFrame(
                {
                    "t": [0.0, 1.0, 2.0, 3.0],
                    "v": [0.0, 8.0, 10.0, 10.0],
                }
            ),
        }

        preview = build_vde_setup_preview_from_ctx(ctx)
        review = build_vde_pre_save_review(ctx, preview, preview["save_payload"])

        self.assertEqual(review["reference_snapshot"]["kind"], "Measured Coastdown Reference")
        self.assertEqual(review["change_summary"]["roadload_basis"]["state"], "Applied")
        self.assertEqual(review["change_summary"]["cycle"]["state"], "Applied")
        self.assertEqual(review["change_summary"]["transmission"]["state"], "Missing")

    def test_build_vde_pre_save_review_for_component_build_up(self):
        ctx = {
            "mode": "Define all parameters (no baseline)",
            "abc_total_source_ui": "Component Build-up",
            "legislation": "EPA",
            "category": "MIDSIZE",
            "mass_kg": 1550.0,
            "test_mass_kg": 1686.0,
            "rr_alpha_N": 95.0,
            "rr_beta_Npkph": 0.45,
            "tire_component_source": "Manual RR",
            "aero_C_coef_Npkph2": 0.012,
            "parasitic_A_coef_N": 4.0,
            "brake_A_coef_N": 1.0,
            "cycle_name": "FTP",
            "cycle_df": pd.DataFrame(
                {
                    "t": [0.0, 1.0, 2.0, 3.0],
                    "v": [0.0, 8.0, 10.0, 10.0],
                }
            ),
        }

        preview = build_vde_setup_preview_from_ctx(ctx)
        review = build_vde_pre_save_review(ctx, preview, preview["save_payload"])

        self.assertEqual(review["reference_snapshot"]["kind"], "Component Build-up Reference")
        self.assertEqual(review["change_summary"]["roadload_basis"]["state"], "Derived")
        self.assertEqual(review["change_summary"]["tires"]["state"], "Applied")
        self.assertEqual(review["change_summary"]["aero"]["state"], "Applied")
        self.assertEqual(review["change_summary"]["brakes"]["state"], "Applied")
        self.assertEqual(review["change_summary"]["parasitics"]["state"], "Applied")

    def test_build_vde_pre_save_review_marks_transmission_missing(self):
        ctx = {
            "mode": "New line (manual / test)",
            "abc_total_source_ui": "From test coastdown",
            "A": 120.0,
            "B": 0.02,
            "C": 0.011,
            "mass_kg": 1550.0,
            "test_mass_kg": 1686.0,
            "legislation": "EPA",
            "category": "MIDSIZE",
            "cycle_df": pd.DataFrame(
                {
                    "t": [0.0, 1.0, 2.0, 3.0],
                    "v": [0.0, 8.0, 10.0, 10.0],
                }
            ),
        }

        preview = build_vde_setup_preview_from_ctx(ctx)
        review = build_vde_pre_save_review(ctx, preview, preview["save_payload"])

        self.assertEqual(review["change_summary"]["transmission"]["state"], "Missing")
        abc_net_row = next(row for row in review["baseline_vs_working_rows"] if row["Field"] == "ABC_NET")
        self.assertEqual(abc_net_row["Working scenario"], "-")
        self.assertEqual(abc_net_row["Change"], "Unavailable")

    def test_build_vde_pre_save_review_for_scenario_without_baseline(self):
        ctx = {
            "mode": "New line (manual / test)",
            "abc_total_source_ui": "From test coastdown",
            "A": 128.0,
            "B": 0.025,
            "C": 0.0105,
            "mass_kg": 1520.0,
            "test_mass_kg": 1656.0,
            "legislation": "EPA",
            "category": "MIDSIZE",
            "cycle_name": "FTP",
            "cycle_df": pd.DataFrame(
                {
                    "t": [0.0, 1.0, 2.0, 3.0],
                    "v": [0.0, 8.0, 10.0, 10.0],
                }
            ),
        }

        preview = build_vde_setup_preview_from_ctx(ctx)
        review = build_vde_pre_save_review(ctx, preview, preview["save_payload"])

        self.assertEqual(review["working_scenario_summary"]["line_source"]["mode"], "NEW")
        self.assertIsNone(review["reference_snapshot"]["baseline_id"])
        self.assertIsNone(review["staged_save_payload"].get("target_vde_id"))
        roadload_row = next(row for row in review["baseline_vs_working_rows"] if row["Field"] == "Roadload basis")
        self.assertEqual(roadload_row["Reference"], "Measured Coastdown Reference")


if __name__ == "__main__":
    unittest.main()
