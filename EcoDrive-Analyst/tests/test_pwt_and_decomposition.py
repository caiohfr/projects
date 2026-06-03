import unittest
from unittest.mock import patch

from src.vde_core.pwt_fuel_energy_service import (
    apply_bev_placeholders,
    build_bev_placeholder_payload,
    compute_vde_total_from_ctx,
    default_electrification_from_vde,
    fetch_distinct_transmission_models,
    fetch_vde_rows_by_ids,
    save_fuelcons_payload,
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

    def test_compute_vde_total_from_ctx_uses_eta_trans_when_available(self):
        result = compute_vde_total_from_ctx({"vde_net_mj_per_km": 1.84}, {"eta_trans": 0.92})
        self.assertAlmostEqual(result["vde_net_mj_per_km"], 1.84)
        self.assertAlmostEqual(result["vde_total_mj_per_km"], 2.0)

    @patch("src.vde_core.pwt_fuel_energy_service.fetchone")
    def test_default_electrification_from_vde_maps_engine_type(self, mock_fetchone):
        mock_fetchone.return_value = {"engine_type": "BEV"}
        self.assertEqual(default_electrification_from_vde(10), "BEV")

        mock_fetchone.return_value = {"engine_type": "HEV"}
        self.assertEqual(default_electrification_from_vde(11), "HEV")

        mock_fetchone.return_value = {"engine_type": "ICE"}
        self.assertEqual(default_electrification_from_vde(12), "ICE")

    @patch("src.vde_core.pwt_fuel_energy_service.fetchall")
    def test_fetch_distinct_transmission_models_flattens_rows(self, mock_fetchall):
        mock_fetchall.return_value = [
            {"transmission_model": "AT6"},
            {"transmission_model": "CVT"},
        ]
        self.assertEqual(fetch_distinct_transmission_models(), ["AT6", "CVT"])

    @patch("src.vde_core.pwt_fuel_energy_service.update_vde")
    def test_apply_bev_placeholders_updates_snapshot(self, mock_update_vde):
        payload = apply_bev_placeholders(42)
        mock_update_vde.assert_called_once_with(42, payload)
        self.assertEqual(payload["transmission_type"], "SS")

    @patch("src.vde_core.pwt_fuel_energy_service.insert_fuelcons")
    def test_save_fuelcons_payload_returns_inserted_id(self, mock_insert):
        mock_insert.return_value = 123
        result = save_fuelcons_payload({"vde_id": 1})
        self.assertEqual(result, 123)

    @patch("src.vde_core.pwt_fuel_energy_service.fetchall")
    def test_fetch_vde_rows_by_ids_returns_dataframe(self, mock_fetchall):
        mock_fetchall.return_value = [{"id": 1, "make": "FORD"}, {"id": 2, "make": "VW"}]
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
