"""Sprint 10D: tests for the new derive_reference_pse()/
list_benchmark_fuelcons_candidates() functions in pwt_fuel_energy_service.py
-- a Streamlit-free extraction of the existing Powertrain Scenario
donor-PSE computation (pwt_fuel_energy._derive_reference_pse), used both for
"Current PSE" and "Benchmark PSE" (Sec 8/9: the same computation, pointed at
different rows -- no second formula for either).
"""

import gc
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.pwt_fuel_energy_service import (
    derive_reference_pse,
    list_benchmark_fuelcons_candidates,
    resolve_reference_fuel_type,
)
from src.vde_core.qa_mock_data import seed_qa_database, seed_qa_fuelcons_mock_rows


class DeriveReferencePseNoDbTests(unittest.TestCase):
    def test_missing_vde_id_is_unavailable(self):
        self.assertEqual(
            derive_reference_pse({"vde_id": None}),
            {"value": None, "status": "unavailable", "basis": None},
        )

    def test_blank_vde_id_is_unavailable(self):
        self.assertEqual(
            derive_reference_pse({"vde_id": ""})["status"],
            "unavailable",
        )

    def test_resolve_reference_fuel_type_reads_assumptions_json(self):
        row = {"assumptions_json": '{"fuel_type": "Diesel"}'}
        self.assertEqual(resolve_reference_fuel_type(row), "Diesel")

    def test_resolve_reference_fuel_type_falls_back_to_provenance_scenario_feature_values(self):
        row = {"provenance_json": '{"scenario_feature_values": {"fuel_type": "E10"}}'}
        self.assertEqual(resolve_reference_fuel_type(row), "E10")

    def test_resolve_reference_fuel_type_none_when_absent(self):
        self.assertIsNone(resolve_reference_fuel_type({}))


class DeriveReferencePseWithQaDatabaseTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "pse_reference_qa.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(self.db_path)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_derive_reference_pse_from_fuel_consumption(self):
        result = derive_reference_pse({"vde_id": 900001, "fuel_l_per_100km": 6.5, "energy_basis": "VDE_TOTAL"})
        self.assertEqual(result["status"], "available")
        self.assertGreater(result["value"], 0.0)

    def test_derive_reference_pse_from_bev_energy(self):
        result = derive_reference_pse({"vde_id": 900001, "energy_Wh_per_km": 150.0, "energy_basis": "VDE_TOTAL"})
        self.assertEqual(result["status"], "available")
        self.assertGreater(result["value"], 0.0)

    def test_derive_reference_pse_prefers_fuel_over_bev_energy_when_both_present(self):
        fuel_only = derive_reference_pse({"vde_id": 900001, "fuel_l_per_100km": 6.5})
        both = derive_reference_pse(
            {"vde_id": 900001, "fuel_l_per_100km": 6.5, "energy_Wh_per_km": 999999.0}
        )
        self.assertEqual(fuel_only["value"], both["value"])

    def test_derive_reference_pse_missing_observed_result_when_neither_present(self):
        result = derive_reference_pse({"vde_id": 900001})
        self.assertEqual(result["status"], "missing_observed_result")
        self.assertIsNone(result["value"])

    def test_derive_reference_pse_unknown_vde_id_is_unavailable(self):
        result = derive_reference_pse({"vde_id": 99999999, "fuel_l_per_100km": 6.5})
        self.assertIn(result["status"], ("unavailable", "missing_demand"))
        self.assertIsNone(result["value"])

    def test_derive_reference_pse_net_basis_selection(self):
        total_result = derive_reference_pse(
            {"vde_id": 900001, "fuel_l_per_100km": 6.5, "energy_basis": "VDE_TOTAL"}
        )
        net_result = derive_reference_pse(
            {"vde_id": 900001, "fuel_l_per_100km": 6.5, "energy_basis": "VDE_NET"}
        )
        self.assertEqual(total_result["basis"], "VDE_TOTAL")
        self.assertEqual(net_result["basis"], "VDE_NET")

    def test_list_benchmark_fuelcons_candidates_excludes_active_vde(self):
        candidates = list_benchmark_fuelcons_candidates(900001)
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.get("vde_id") != 900001 for candidate in candidates))

    def test_list_benchmark_fuelcons_candidates_includes_other_vde_rows(self):
        candidates = list_benchmark_fuelcons_candidates(900001)
        candidate_vde_ids = {candidate.get("vde_id") for candidate in candidates}
        self.assertTrue(candidate_vde_ids)


if __name__ == "__main__":
    unittest.main()
