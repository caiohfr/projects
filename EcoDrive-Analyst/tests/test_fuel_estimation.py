import unittest
from unittest.mock import patch

from src.vde_core.fuel_estimation import (
    FuelEstimateRequest,
    build_fuel_scenario_save_payload,
    run_fuel_estimation,
    save_fuel_estimate_result,
)


class FuelEstimationTests(unittest.TestCase):
    @patch("src.vde_core.fuel_estimation.insert_fuelcons_row")
    @patch("src.vde_core.fuel_estimation.update_fuelcons_by_id")
    @patch("src.vde_core.fuel_estimation.delete_fuelcons_by_id")
    def test_run_fuel_estimation_does_not_persist(
        self,
        mock_delete,
        mock_update,
        mock_insert,
    ):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=1,
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={"electrification": "ICE", "vde_total_mj_per_km": 1.8},
                powertrain_features={"eta_pt_est": 0.3, "fuel_type": "Gasoline"},
            )
        )

        self.assertEqual(result.method, "physics_simple")
        mock_insert.assert_not_called()
        mock_update.assert_not_called()
        mock_delete.assert_not_called()

    def test_run_fuel_estimation_warns_when_vde_net_is_unavailable(self):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=1,
                energy_basis="VDE_NET",
                method="physics_simple",
                vehicle_features={"electrification": "ICE"},
                powertrain_features={"eta_pt_est": 0.3, "fuel_type": "Gasoline"},
            )
        )

        self.assertIn("vde_net_selected_but_unavailable", result.warnings)
        self.assertIn("energy_basis_value_missing", result.warnings)
        self.assertIsNone(result.fuel_l_100km)

    def test_run_fuel_estimation_physics_simple_ice(self):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=1,
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={"electrification": "ICE", "vde_total_mj_per_km": 1.8},
                powertrain_features={"eta_pt_est": 0.3, "fuel_type": "Gasoline", "LHV_MJ_per_L": 32.0},
            )
        )

        self.assertAlmostEqual(result.fuel_l_100km, 18.75, places=4)
        self.assertAlmostEqual(result.gco2_km, 433.125, places=4)
        self.assertIsNone(result.energy_Wh_km)

    def test_run_fuel_estimation_physics_simple_bev(self):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=2,
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={"electrification": "BEV", "vde_total_mj_per_km": 1.5},
                powertrain_features={"bev_eff_drive": 0.9, "grid_gco2_per_kwh": 400.0},
            )
        )

        self.assertAlmostEqual(result.energy_Wh_km, 462.962962963, places=6)
        self.assertAlmostEqual(result.gco2_km, 185.1851851852, places=6)
        self.assertIsNone(result.fuel_l_100km)

    def test_run_fuel_estimation_physics_simple_phev_combines_both_paths(self):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=6,
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={"electrification": "PHEV", "vde_total_mj_per_km": 1.8},
                powertrain_features={
                    "fuel_type": "Gasoline",
                    "eta_pt_est": 0.3,
                    "LHV_MJ_per_L": 32.0,
                    "bev_eff_drive": 0.9,
                    "utility_factor": 0.4,
                    "grid_gco2_per_kwh": 400.0,
                },
            )
        )

        self.assertAlmostEqual(result.fuel_l_100km, 11.25, places=4)
        self.assertAlmostEqual(result.energy_Wh_km, 222.2222222224, places=6)
        self.assertAlmostEqual(result.gco2_km, 259.875, places=4)

    def test_run_fuel_estimation_manual_imported_passes_through_values(self):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=3,
                energy_basis="MANUAL_VALUE",
                method="manual_imported",
                vehicle_features={"electrification": "ICE"},
                manual_inputs={
                    "source": "lab_sheet",
                    "vde_mj_per_km": 1.7,
                    "fuel_l_100km": 7.2,
                    "gco2_km": 155.0,
                },
            )
        )

        self.assertEqual(result.confidence, "provided")
        self.assertAlmostEqual(result.fuel_l_100km, 7.2)
        self.assertAlmostEqual(result.gco2_km, 155.0)
        self.assertEqual(result.assumptions["source"], "lab_sheet")

    def test_run_fuel_estimation_regression_existing_accepts_runner(self):
        def regression_runner(request_dict, vde_mj_per_km):
            self.assertEqual(request_dict["method"], "regression_existing")
            self.assertAlmostEqual(vde_mj_per_km, 1.4)
            return {
                "fuel_l_100km": 6.4,
                "gco2_km": 148.0,
                "fuel_l_per_100km_urb": 7.1,
                "fuel_l_per_100km_hw": 5.8,
            }

        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=4,
                energy_basis="VDE_TOTAL",
                method="regression_existing",
                vehicle_features={"electrification": "ICE", "vde_total_mj_per_km": 1.4},
                model_options={"regression_runner": regression_runner},
            )
        )

        self.assertAlmostEqual(result.fuel_l_100km, 6.4)
        self.assertAlmostEqual(result.gco2_km, 148.0)
        self.assertEqual(result.confidence, "medium")
        self.assertAlmostEqual(result.phase_outputs["fuel_ftp75_l_per_100km"], 7.1)
        self.assertAlmostEqual(result.phase_outputs["fuel_hwfet_l_per_100km"], 5.8)

    @patch("src.vde_core.ml_prediction.find_ml_artifact_paths", return_value=[])
    def test_run_fuel_estimation_ml_prediction_without_artifact_is_honest(self, _mock_find_artifacts):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=10,
                energy_basis="VDE_TOTAL",
                method="ml_prediction",
                vehicle_features={
                    "electrification": "ICE",
                    "category": "SUV",
                    "make": "TEST",
                    "year": 2026,
                    "vde_total_mj_per_km": 1.8,
                    "vde_net_mj_per_km": 1.6,
                },
            )
        )

        self.assertEqual(result.method, "ml_prediction")
        self.assertIsNone(result.fuel_l_100km)
        self.assertIn("ml_notebook_exists_but_no_exported_inference_artifact_found", result.warnings)
        self.assertEqual(result.assumptions["integration_status"], "export_pending")
        self.assertIn("vde_net_mj_per_km", result.assumptions["features_used"])

    def test_run_fuel_estimation_ml_prediction_accepts_injected_predictor(self):
        def ml_predictor(request_dict, feature_row, metadata):
            self.assertEqual(request_dict["method"], "ml_prediction")
            self.assertEqual(feature_row["electrification"], "ICE")
            self.assertEqual(feature_row["make"], "AUDI")
            self.assertIn("notebook", metadata)
            return {
                "fuel_l_100km": 6.2,
                "gco2_km": 144.0,
                "fuel_l_per_100km_urb": 7.0,
                "fuel_l_per_100km_hw": 5.5,
                "confidence": "medium",
                "model_name": "StubML",
                "coverage_status": "in_domain",
                "feature_contributions": {
                    "vde_net_mj_per_km": 0.31,
                    "coast_A_N": 0.10,
                    "engine_size_l": -0.06,
                    "gear_count": 0.02,
                },
                "warnings": ["peer_guidance_pending"],
            }

        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=11,
                energy_basis="VDE_NET",
                method="ml_prediction",
                vehicle_features={
                    "electrification": "ICE",
                    "category": "MIDSIZE",
                    "make": "AUDI",
                    "year": 2025,
                    "vde_total_mj_per_km": 1.9,
                    "vde_net_mj_per_km": 1.7,
                },
                model_options={"ml_predictor": ml_predictor},
            )
        )

        self.assertAlmostEqual(result.fuel_l_100km, 6.2)
        self.assertAlmostEqual(result.gco2_km, 144.0)
        self.assertEqual(result.assumptions["model_name"], "StubML")
        self.assertEqual(result.assumptions["coverage_status"], "in_domain")
        self.assertIn("peer_guidance_pending", result.warnings)
        self.assertAlmostEqual(result.phase_outputs["fuel_ftp75_l_per_100km"], 7.0)
        self.assertAlmostEqual(result.phase_outputs["fuel_hwfet_l_per_100km"], 5.5)
        self.assertTrue(result.assumptions["shap_available"])
        self.assertEqual(result.assumptions["ml_explanation"]["status"], "available")

    def test_run_fuel_estimation_ml_prediction_reports_out_of_domain_when_target_is_far(self):
        def ml_predictor(request_dict, feature_row, metadata):
            del request_dict, feature_row, metadata
            return {
                "fuel_l_100km": 9.1,
                "gco2_km": 210.0,
                "confidence": "low",
            }

        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=12,
                energy_basis="VDE_TOTAL",
                method="ml_prediction",
                vehicle_features={
                    "electrification": "ICE",
                    "category": "UNKNOWN_SEGMENT",
                    "make": "RAREBRAND",
                    "year": 2099,
                    "engine_size_l": 9.9,
                    "vde_total_mj_per_km": 9.9,
                    "vde_net_mj_per_km": 9.1,
                },
                model_options={"ml_predictor": ml_predictor},
            )
        )

        self.assertIn(result.assumptions["coverage_status"], {"partial_domain", "out_of_domain", "metadata_unavailable"})
        self.assertIn("coverage_details", result.assumptions)

    def test_build_fuel_scenario_save_payload_is_common_across_methods(self):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=7,
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={
                    "electrification": "ICE",
                    "vde_total_mj_per_km": 1.8,
                    "phase_outputs": {
                        "vde_urb_mj_per_km": 2.0,
                        "vde_hw_mj_per_km": 1.5,
                    },
                    "source_vde_revision": "2026-06-23T10:00:00",
                    "source_vde_created_at": "2026-06-20T10:00:00",
                    "source_vde_updated_at": "2026-06-23T10:00:00",
                },
                powertrain_features={"eta_pt_est": 0.3, "fuel_type": "Gasoline", "LHV_MJ_per_L": 32.0},
            )
        )

        staged = build_fuel_scenario_save_payload(result, extra_payload={"gear_count": 6})

        self.assertEqual(staged.data_origin, "physics")
        self.assertEqual(staged.payload["vde_id"], 7)
        self.assertEqual(staged.payload["gear_count"], 6)
        self.assertEqual(staged.payload["energy_basis"], "VDE_TOTAL")
        self.assertEqual(staged.payload["engine_method"], "physics_simple")
        self.assertEqual(staged.payload["engine_version"], "fuel_estimation_v1")
        self.assertEqual(staged.payload["source_vde_revision"], "2026-06-23T10:00:00")
        self.assertIn("\"eta_pt_est\": 0.3", staged.payload["assumptions_json"])
        self.assertIn("\"engine_method\": \"physics_simple\"", staged.payload["provenance_json"])
        self.assertIn("fuel_ftp75_l_per_100km", staged.payload)
        self.assertIn("fuel_hwfet_l_per_100km", staged.payload)

    def test_build_fuel_scenario_save_payload_scales_utility_factor_to_percent_field(self):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=70,
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={"electrification": "PHEV", "vde_total_mj_per_km": 1.8},
                powertrain_features={
                    "fuel_type": "Gasoline",
                    "eta_pt_est": 0.3,
                    "LHV_MJ_per_L": 32.0,
                    "bev_eff_drive": 0.9,
                    "utility_factor": 0.4,
                    "grid_gco2_per_kwh": 400.0,
                },
            )
        )

        staged = build_fuel_scenario_save_payload(result)

        self.assertAlmostEqual(staged.payload["utility_factor_pct"], 40.0)

    @patch("src.vde_core.fuel_estimation.insert_fuelcons_row")
    def test_save_fuel_estimate_result_persists_insert_payload(self, mock_insert):
        mock_insert.return_value = 99
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=5,
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={"electrification": "BEV", "vde_total_mj_per_km": 1.5},
                powertrain_features={"bev_eff_drive": 0.9},
            )
        )

        saved = save_fuel_estimate_result(result, "insert_new")

        self.assertEqual(saved["action"], "insert_new")
        self.assertEqual(saved["row_id"], 99)
        self.assertEqual(saved["payload"]["vde_id"], 5)
        self.assertEqual(saved["payload"]["electrification"], "BEV")
        mock_insert.assert_called_once()

    @patch("src.vde_core.fuel_estimation.insert_fuelcons_row")
    def test_save_fuel_estimate_result_accepts_extra_payload(self, mock_insert):
        mock_insert.return_value = 101
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=8,
                energy_basis="MANUAL_VALUE",
                method="manual_imported",
                vehicle_features={"electrification": "ICE"},
                manual_inputs={"source": "lab", "fuel_l_100km": 6.8},
            )
        )

        saved = save_fuel_estimate_result(result, "insert_new", extra_payload={"gear_count": 7})

        self.assertEqual(saved["row_id"], 101)
        self.assertEqual(saved["payload"]["gear_count"], 7)

    @patch("src.vde_core.fuel_estimation.update_fuelcons_by_id")
    def test_save_fuel_estimate_result_supports_update_existing(self, mock_update):
        result = run_fuel_estimation(
            FuelEstimateRequest(
                vde_id=9,
                energy_basis="VDE_TOTAL",
                method="physics_simple",
                vehicle_features={"electrification": "ICE", "vde_total_mj_per_km": 1.8},
                powertrain_features={"eta_pt_est": 0.3, "fuel_type": "Gasoline"},
            )
        )

        saved = save_fuel_estimate_result(result, "update_existing", row_id=77, extra_payload={"gear_count": 8})

        self.assertEqual(saved["action"], "update_existing")
        self.assertEqual(saved["row_id"], 77)
        self.assertEqual(saved["payload"]["gear_count"], 8)
        mock_update.assert_called_once()


if __name__ == "__main__":
    unittest.main()
