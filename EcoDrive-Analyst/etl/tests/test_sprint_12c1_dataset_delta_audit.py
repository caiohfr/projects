from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12c1_dataset_delta_audit as audit  # noqa: E402


class Sprint12C1DatasetDeltaAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hash_before = hashlib.sha256(audit.DB_PATH.read_bytes()).hexdigest()
        cls.payload = audit.main()
        cls.hash_after = hashlib.sha256(audit.DB_PATH.read_bytes()).hexdigest()

    def test_database_is_byte_identical_and_ddl_is_paused(self) -> None:
        self.assertEqual(self.hash_before, self.hash_after)
        self.assertTrue(self.payload["database_byte_identical"])
        self.assertEqual(self.payload["decision"], "PAUSE_DDL — SOURCE_POPULATION_DECISION_REQUIRED")

    def test_legacy_core_and_post_import_records_are_separated(self) -> None:
        populations = {row["population"]: row for row in self.payload["population_summary"]}
        self.assertEqual(populations["Current legacy core in SQLite"]["vde_or_reporting_rows"], 4_999)
        self.assertEqual(populations["Current post-import VDE additions"]["vde_or_reporting_rows"], 4)
        self.assertEqual(populations["Current post-import FuelCons additions"]["vde_or_reporting_rows"], 5)
        self.assertEqual(len(self.payload["legacy_duplicate_groups"]), 46)

    def test_legacy_component_abc_is_classified_as_estimated(self) -> None:
        coverage = {(row["population"], row["concept"]): row for row in self.payload["engineering_coverage"]}
        trans = coverage[("Legacy core (4,999 VDE)", "Transmission loss ABC")]
        brake = coverage[("Legacy core (4,999 VDE)", "Brake ABC")]
        self.assertEqual(trans["available_rows"], 2_958)
        self.assertEqual(trans["provenance_quality"], "ESTIMATED")
        self.assertEqual(brake["provenance_quality"], "ESTIMATED")
        self.assertTrue(self.payload["legacy_notebook_evidence"]["all_expected_evidence_present"])

    def test_epa_2026_gain_is_measured_at_multiple_grains(self) -> None:
        populations = {row["population"]: row for row in self.payload["population_summary"]}
        epa = populations["EPA Test Car MY2026"]
        self.assertEqual(epa["source_rows"], 3_901)
        self.assertEqual(epa["test_candidates"], 3_557)
        self.assertEqual(epa["vde_or_reporting_rows"], 1_380)
        self.assertEqual(epa["vehicle_or_mmy_groups"], 784)
        self.assertEqual(self.payload["strict_epa_2026_configuration_candidates"], 1_291)

    def test_usability_keeps_provenance_distinction(self) -> None:
        levels = {(row["population"], row["level"]): row["rows"] for row in self.payload["engineering_usability"]}
        self.assertEqual(levels[("Legacy core (4,999 VDE)", "L3")], 2_958)
        self.assertEqual(levels[("EPA Test Car MY2026", "L2")], 3_901)
        self.assertEqual(levels[("JRC technical dataset", "L3")], 249)
        self.assertEqual(levels[("EEA 2025 provisional", "L0")], 10_833_597)


if __name__ == "__main__":
    unittest.main()
