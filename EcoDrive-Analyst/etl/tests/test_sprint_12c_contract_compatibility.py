from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12c_contract_compatibility as contract  # noqa: E402


class Sprint12CCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_hash_before = hashlib.sha256(contract.DB_PATH.read_bytes()).hexdigest()
        cls.payload = contract.main()
        cls.db_hash_after = hashlib.sha256(contract.DB_PATH.read_bytes()).hexdigest()
        cls.out = contract.OUT

    def test_runtime_database_is_byte_identical_after_materialization(self) -> None:
        self.assertEqual(self.db_hash_before, self.db_hash_after)

    def test_exact_contract_covers_all_nine_entities(self) -> None:
        with (self.out / "canonical_field_contract_v1.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(set(contract.ENTITIES), {row["entity"] for row in rows})
        self.assertEqual(self.payload["field_contract_count"], len(rows))
        self.assertEqual(len(rows), 333)
        required = {
            ("PROGRAM", "program_id"),
            ("VEHICLE_CONFIGURATION", "vehicle_configuration_id"),
            ("COMPONENT_DB", "component_id"),
            ("TIRE_DB", "tire_id"),
            ("COMPONENT_INSTANCE", "component_instance_id"),
            ("COMPONENT_RESOLUTION", "component_resolution_id"),
            ("VDE", "vehicle_configuration_id"),
            ("RUN", "run_type"),
            ("RUN", "evidence_kind"),
            ("FUELCONS", "comparison_basis"),
            ("FUELCONS", "adopted_run_ids_json"),
        }
        self.assertTrue(required.issubset({(row["entity"], row["field_name"]) for row in rows}))

    def test_all_legacy_vde_and_fuelcons_values_are_exact(self) -> None:
        summary = self.payload["compatibility_summary"]
        self.assertEqual(summary["vde_rows"], 5003)
        self.assertEqual(summary["fuelcons_rows"], 5004)
        self.assertEqual(summary["vde_value_mismatches"], 0)
        self.assertEqual(summary["fuelcons_value_mismatches"], 0)

    def test_relationship_and_null_checks_are_exact(self) -> None:
        by_category = {row["category"]: row for row in self.payload["compatibility_checks"]}
        for category in (
            "VDE-FuelCons linkage",
            "FuelCons multiplicity per VDE",
            "VDE parent lineage",
            "NULL behavior",
            "mass",
            "roadload ABC",
            "TOTAL/NET",
            "Urban/Highway/Combined results",
            "labels/filters",
            "RUN adoption lineage",
        ):
            self.assertEqual(by_category[category]["status"], "EXACT_EQUIVALENCE", category)
        self.assertEqual(self.payload["compatibility_summary"]["orphan_fuelcons"], 0)
        self.assertEqual(self.payload["compatibility_summary"]["vdes_with_multiple_fuelcons"], 2)

    def test_no_cdr_blocker_and_recommendation_is_ready_for_ddl(self) -> None:
        self.assertFalse([row for row in self.payload["compatibility_checks"] if row["status"] == "CDR_BLOCKER"])
        self.assertEqual(self.payload["recommendation"], "READY_FOR_DDL")
        self.assertIn("production migration remains unauthorized", self.payload["recommendation_scope"])
        self.assertEqual(self.payload["contract_validation_violations"], 0)
        self.assertEqual(self.payload["compatibility_summary"]["checks"], 18)

    def test_public_materialization_scope_defers_unsupported_epa_vde(self) -> None:
        by_source = {row["source"]: row for row in self.payload["public_materialization_scope"]}
        epa = by_source["EPA Test Car MY2026"]
        self.assertEqual(epa["entities_materialized"], "PROGRAM; VEHICLE_CONFIGURATION")
        self.assertIn("VDE", epa["entities_deferred"])
        self.assertEqual(self.payload["materialization_counts"]["VDE"]["public"], 249)

    def test_staging_files_match_materialization_counts(self) -> None:
        for entity in contract.ENTITIES:
            path = contract.STAGING / f"{entity.lower()}.jsonl"
            with path.open(encoding="utf-8") as handle:
                actual = sum(1 for _ in handle)
            self.assertEqual(actual, self.payload["materialization_counts"][entity]["total"], entity)

    def test_mismatch_report_explicitly_records_no_difference(self) -> None:
        with (self.out / "mismatch_report.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "EXACT_EQUIVALENCE")


if __name__ == "__main__":
    unittest.main()
