import unittest
from unittest.mock import patch

from src.vde_core.pwt_fuel_energy_service import (
    apply_bev_placeholders,
    build_fuel_estimate_request_from_vde,
    build_bev_placeholder_payload,
    compare_saved_scenario_revision,
    compute_vde_total_from_ctx,
    default_electrification_from_vde,
    fetch_distinct_transmission_models,
    fetch_vde_rows_by_ids,
    preview_fuel_estimate_from_vde,
    resolve_vde_energy_values,
    resolve_vde_source_revision,
    save_fuelcons_payload,
    summarize_saved_scenario_revision_states,
)
from src.vde_core.roadload.decomposition import (
    component_delta_vs_baseline,
    decompose_equivalent_abc,
)
from src.vde_core.roadload.models import EquivalentABC, ResolvedBaseline


class PwtFuelEnergyServiceTests(unittest.TestCase):
    def test_build_bev_placeholder_payload_contains_expected_fields(self):
        payload = build_bev_placeholder_payload()
        self.assertEqual(payload["engine_size_l"], 0.001)
        self.assertEqual(payload["transmission_type"], "SS")
        self.assertNotIn("engine_model", {k: v for k, v in payload.items() if v is None})

    def test_compute_vde_total_from_ctx_uses_resolved_row_values(self):
        result = compute_vde_total_from_ctx({"vde_total_mj_per_km": 2.0, "vde_net_mj_per_km": 1.84}, {"eta_trans": 0.92})
        self.assertAlmostEqual(result["vde_net_mj_per_km"], 1.84)
        self.assertAlmostEqual(result["vde_total_mj_per_km"], 2.0)
        self.assertEqual(result["warnings"], [])

    def test_resolve_vde_energy_values_never_promotes_net_only_row_without_legacy_origin(self):
        result = resolve_vde_energy_values({"vde_net_mj_per_km": 1.84})

        self.assertIsNone(result["vde_total_mj_per_km"])
        self.assertIsNone(result["vde_net_mj_per_km"])
        self.assertIn("vde_total_missing", result["warnings"])
        self.assertNotIn("legacy_vde_net_used_as_total_candidate", result["warnings"])

    def test_resolve_vde_energy_values_reports_legacy_row_as_total_unavailable(self):
        result = resolve_vde_energy_values(
            {"vde_net_mj_per_km": 1.84, "record_origin": "LEGACY"}
        )

        self.assertIsNone(result["vde_total_mj_per_km"])
        self.assertIsNone(result["vde_net_mj_per_km"])
        self.assertIn("vde_total_missing", result["warnings"])

    def test_resolve_vde_energy_values_reports_canonical_total_and_net(self):
        result = resolve_vde_energy_values(
            {"vde_total_mj_per_km": 2.0, "vde_net_mj_per_km": 1.84, "record_origin": "VDE_SETUP"}
        )

        self.assertAlmostEqual(result["vde_total_mj_per_km"], 2.0)
        self.assertAlmostEqual(result["vde_net_mj_per_km"], 1.84)
        self.assertEqual(result["warnings"], [])

    def test_build_fuel_estimate_request_from_vde_maps_row_into_contract(self):
        request = build_fuel_estimate_request_from_vde(
            {
                "id": 21,
                "legislation": "EPA",
                "category": "MIDSIZE",
                "make": "FORD",
                "model": "TEST",
                "year": 2025,
                "engine_type": "HEV",
                "cycle_name": "FTP75_HWFET",
                "mass_kg": 1735.0,
                "test_mass_kg": 1814.0,
                "inertia_class": 1814.0,
                "engine_size_l": 2.0,
                "transmission_type": "AT",
                "drive_type": "FWD",
                "gear_count": 6,
                "final_drive_ratio": 3.91,
                "coast_A_N": 120.0,
                "coast_B_N_per_kph": 0.03,
                "coast_C_N_per_kph2": 0.0002,
                "vde_net_mj_per_km": 1.7,
                "created_at": "2026-06-20T08:00:00",
                "updated_at": "2026-06-23T09:30:00",
            },
            energy_basis="VDE_TOTAL",
            method="physics_simple",
            powertrain_features={"eta_pt_est": 0.34},
        )

        self.assertEqual(request.vde_id, 21)
        self.assertEqual(request.energy_basis, "VDE_TOTAL")
        self.assertEqual(request.method, "physics_simple")
        self.assertEqual(request.vehicle_features["electrification"], "HEV")
        self.assertIsNone(request.vehicle_features["vde_total_mj_per_km"])
        self.assertAlmostEqual(request.vehicle_features["mass_kg"], 1735.0)
        self.assertAlmostEqual(request.vehicle_features["test_mass_kg"], 1814.0)
        self.assertAlmostEqual(request.vehicle_features["inertia_class"], 1814.0)
        self.assertAlmostEqual(request.vehicle_features["engine_size_l"], 2.0)
        self.assertEqual(request.vehicle_features["transmission_type"], "AT")
        self.assertAlmostEqual(request.vehicle_features["coast_A_N"], 120.0)
        self.assertEqual(request.vehicle_features["source_vde_revision"], "2026-06-23T09:30:00")
        self.assertIn(
            "vde_total_missing",
            request.vehicle_features["compatibility_warnings"],
        )
        self.assertNotIn(
            "legacy_vde_net_used_as_total_candidate",
            request.vehicle_features["compatibility_warnings"],
        )

    def test_resolve_vde_source_revision_prefers_updated_at(self):
        revision = resolve_vde_source_revision(
            {"created_at": "2026-06-20T08:00:00", "updated_at": "2026-06-23T09:30:00"}
        )
        self.assertEqual(revision, "2026-06-23T09:30:00")

    def test_compare_saved_scenario_revision_detects_changed_source(self):
        state = compare_saved_scenario_revision(
            "2026-06-21T00:00:00",
            {"created_at": "2026-06-20T08:00:00", "updated_at": "2026-06-23T09:30:00"},
        )
        self.assertEqual(state["status"], "changed")
        self.assertIn("Refresh / Recalculate required", state["message"])

    def test_compare_saved_scenario_revision_detects_current_source(self):
        state = compare_saved_scenario_revision(
            "2026-06-23T09:30:00",
            {"created_at": "2026-06-20T08:00:00", "updated_at": "2026-06-23T09:30:00"},
        )
        self.assertEqual(state["status"], "current")

    def test_summarize_saved_scenario_revision_states_counts_refresh_required(self):
        summary = summarize_saved_scenario_revision_states(
            [
                {"source_vde_revision": "2026-06-23T09:30:00"},
                {"source_vde_revision": "2026-06-21T00:00:00"},
                {"source_vde_revision": None},
            ],
            {"created_at": "2026-06-20T08:00:00", "updated_at": "2026-06-23T09:30:00"},
        )

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["current"], 1)
        self.assertEqual(summary["changed"], 1)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["refresh_required"], 2)

    def test_preview_fuel_estimate_from_vde_uses_new_estimation_contract(self):
        result = preview_fuel_estimate_from_vde(
            {
                "id": 22,
                "engine_type": "BEV",
                "vde_total_mj_per_km": 1.5,
            },
            electrification="BEV",
            energy_basis="VDE_TOTAL",
            method="physics_simple",
            powertrain_features={"bev_eff_drive": 0.9},
        )

        self.assertEqual(result.method, "physics_simple")
        self.assertEqual(result.energy_basis_used, "VDE_TOTAL")
        self.assertAlmostEqual(result.energy_Wh_km, 462.962962963, places=6)

    def test_preview_fuel_estimate_from_vde_respects_selected_vde_net_basis(self):
        result = preview_fuel_estimate_from_vde(
            {
                "id": 23,
                "engine_type": "ICE",
                "vde_total_mj_per_km": 2.0,
                "vde_net_mj_per_km": 1.5,
            },
            electrification="ICE",
            energy_basis="VDE_NET",
            method="physics_simple",
            powertrain_features={"eta_pt_est": 0.3, "fuel_type": "Gasoline", "LHV_MJ_per_L": 32.0},
        )

        self.assertEqual(result.energy_basis_used, "VDE_NET")
        self.assertAlmostEqual(result.fuel_l_100km, 15.625, places=4)

    @patch("src.vde_core.pwt_fuel_energy_service.fetch_vde_engine_type")
    def test_default_electrification_from_vde_maps_engine_type(self, mock_fetch_vde_engine_type):
        mock_fetch_vde_engine_type.return_value = "BEV"
        self.assertEqual(default_electrification_from_vde(10), "BEV")

        mock_fetch_vde_engine_type.return_value = "HEV"
        self.assertEqual(default_electrification_from_vde(11), "HEV")

        mock_fetch_vde_engine_type.return_value = "ICE"
        self.assertEqual(default_electrification_from_vde(12), "ICE")

    @patch("src.vde_core.pwt_fuel_energy_service.fetch_vde_distinct_transmission_models")
    def test_fetch_distinct_transmission_models_flattens_rows(self, mock_fetch_vde_distinct_transmission_models):
        mock_fetch_vde_distinct_transmission_models.return_value = ["AT6", "CVT"]
        self.assertEqual(fetch_distinct_transmission_models(), ["AT6", "CVT"])

    def test_apply_bev_placeholders_returns_draft_only_payload(self):
        payload = apply_bev_placeholders(42)
        self.assertEqual(payload["transmission_type"], "SS")
        self.assertEqual(payload["engine_size_l"], 0.001)

    @patch("src.vde_core.pwt_fuel_energy_service.insert_fuelcons_row")
    def test_save_fuelcons_payload_returns_inserted_id(self, mock_insert):
        mock_insert.return_value = 123
        result = save_fuelcons_payload({"vde_id": 1})
        self.assertEqual(result, 123)

    @patch("src.vde_core.pwt_fuel_energy_service.fetch_vde_by_ids")
    def test_fetch_vde_rows_by_ids_returns_dataframe(self, mock_fetch_vde_by_ids):
        mock_fetch_vde_by_ids.return_value = [{"id": 1, "make": "FORD"}, {"id": 2, "make": "VW"}]
        df = fetch_vde_rows_by_ids([1, 2, 2, None])
        self.assertEqual(df["id"].tolist(), [1, 2])


class DecompositionTests(unittest.TestCase):
    def test_decompose_equivalent_abc_normalizes_payload(self):
        equiv = EquivalentABC(
            A=121.0,
            B=0.021,
            C=0.012,
            mass_kg=1630.0,
            component_table=[{"name": "roadload_total"}],
            warnings=["check"],
        )

        payload = decompose_equivalent_abc(equiv)

        self.assertEqual(payload["A"], 121.0)
        self.assertEqual(payload["mass_kg"], 1630.0)
        self.assertEqual(payload["components"], [{"name": "roadload_total"}])
        self.assertEqual(payload["warnings"], ["check"])

    def test_component_delta_vs_baseline_returns_expected_deltas(self):
        baseline = ResolvedBaseline(A=120.0, B=0.02, C=0.011, mass_kg=1550.0)
        equiv = EquivalentABC(A=121.0, B=0.021, C=0.012, mass_kg=1630.0)

        delta = component_delta_vs_baseline(baseline, equiv)

        self.assertAlmostEqual(delta["delta_A"], 1.0)
        self.assertAlmostEqual(delta["delta_B"], 0.001)
        self.assertAlmostEqual(delta["delta_C"], 0.001)
        self.assertAlmostEqual(delta["delta_mass_kg"], 80.0)


if __name__ == "__main__":
    unittest.main()
