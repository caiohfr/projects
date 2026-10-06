from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
import sprint_12f14b_vde_semantic_freeze as sprint  # noqa: E402


class Sprint12F14BVdeSemanticFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(sprint.SUMMARY.read_text(encoding="utf-8"))

    def test_01_parent_lineage_contract_is_nullable_self_fk(self) -> None:
        lineage = self.summary["schema_evidence"]["lineage"]
        self.assertTrue(lineage["parent_column_nullable"])
        self.assertTrue(lineage["parent_self_fk_valid"])
        self.assertEqual(lineage["invalid_parent_rows"], 0)
        self.assertTrue(self.summary["semantic_checks"]["vde_id_parent_sufficient"])

    def test_02_same_configuration_can_have_derived_vde_family(self) -> None:
        lineage = self.summary["schema_evidence"]["lineage"]
        self.assertGreater(lineage["same_configuration_multi_vde_families"], 0)
        self.assertFalse(lineage["vehicle_configuration_unique_constraint_blocks_family"])
        self.assertTrue(lineage["same_configuration_derived_vde_allowed"])

    def test_03_mass_and_snapshot_fields_are_preserved(self) -> None:
        missing = self.summary["schema_evidence"]["missing_fields"]
        self.assertEqual(missing["mass"], [])
        self.assertEqual(missing["roadload_component_snapshot"], [])
        self.assertEqual(missing["wide_results"], [])
        self.assertTrue(self.summary["semantic_checks"]["performance_custom_vde_representable"])

    def test_04_vde_population_and_candidate_are_unchanged(self) -> None:
        self.assertEqual(self.summary["counts_before"]["vde"], 11626)
        self.assertEqual(self.summary["counts_before"]["vde"], self.summary["counts_after"]["vde"])
        self.assertFalse(self.summary["vde_rows_changed"])
        self.assertFalse(self.summary["schema_changed"])
        self.assertEqual(self.summary["input_hash_before"], self.summary["input_hash_after"])
        self.assertTrue(self.summary["candidate_matches_12f14a"])

    def test_05_run_and_fuelcons_populations_are_unchanged(self) -> None:
        before = self.summary["counts_before"]
        after = self.summary["counts_after"]
        self.assertEqual(before["run"], 29250)
        self.assertEqual(before["fuelcons"], 10822)
        self.assertEqual(before["fuelcons_run_adoption"], 18323)
        self.assertEqual(before, after)
        self.assertFalse(self.summary["run_rows_changed"])
        self.assertFalse(self.summary["fuelcons_rows_changed"])

    def test_06_no_speculative_architecture_was_introduced(self) -> None:
        forbidden = self.summary["schema_evidence"]["forbidden"]
        self.assertEqual(forbidden["tables_present"], [])
        self.assertEqual(forbidden["vde_columns_present"], [])
        self.assertTrue(self.summary["current_schema_sufficient"])
        self.assertFalse(self.summary["semantic_checks"]["current_real_data_gap_found"])

    def test_07_compatibility_reads_and_surfaces_are_preserved(self) -> None:
        missing = self.summary["schema_evidence"]["missing_fields"]
        self.assertEqual(missing["vde_compatibility_read"], [])
        self.assertEqual(missing["fuelcons_compatibility_read"], [])
        self.assertTrue(self.summary["compatibility_surface_unchanged"])
        self.assertEqual(self.summary["application_contract_status"], "PARTIAL")
        self.assertEqual(self.summary["localized_integration_changes_expected"], 1)
        self.assertEqual(self.summary["page_changes_expected"], 0)

    def test_08_runtime_databases_are_unchanged(self) -> None:
        self.assertEqual(self.summary["runtime_hashes_before"], self.summary["runtime_hashes_after"])
        self.assertFalse(self.summary["runtime_changed"])
        self.assertTrue(self.summary["documentation_frozen"])
        self.assertEqual(self.summary["status"], sprint.STATUS)
        self.assertEqual(self.summary["focused_assertions"], {"passed": 8, "total": 8})


if __name__ == "__main__":
    unittest.main()
