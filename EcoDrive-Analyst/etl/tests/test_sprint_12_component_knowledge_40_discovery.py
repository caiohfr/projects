from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from etl.scripts.sprint_12_component_knowledge_40_discovery import (
    DEFAULT_CANDIDATE,
    ROOT,
    build_discovery_artifacts,
    sha256_file,
)


class Sprint12ComponentKnowledge40DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="component_40_discovery_")
        cls.output = Path(cls.temporary.name) / "artifacts"
        cls.candidate_hash_before = sha256_file(DEFAULT_CANDIDATE)
        cls.report = build_discovery_artifacts(ROOT, DEFAULT_CANDIDATE, cls.output)
        cls.candidate_hash_after = sha256_file(DEFAULT_CANDIDATE)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @staticmethod
    def _rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_source_inventory_is_reproducible(self) -> None:
        second_output = Path(self.temporary.name) / "second"
        second = build_discovery_artifacts(ROOT, DEFAULT_CANDIDATE, second_output)
        self.assertEqual(self.report, second)
        for name in (
            "COMPONENT_SOURCE_INVENTORY.csv",
            "COMPONENT_KNOWLEDGE_COVERAGE.csv",
            "COMPONENT_SEED_AUDIT.csv",
            "COMPONENT_REFERENCE_POPULATIONS.csv",
            "COMPONENT_OUTLIER_REVIEW.csv",
        ):
            self.assertEqual((self.output / name).read_bytes(), (second_output / name).read_bytes())

    def test_no_raw_component_loss_source_is_misclassified_as_available(self) -> None:
        rows = self._rows(self.output / "COMPONENT_SOURCE_INVENTORY.csv")
        self.assertFalse(self.report.component_source_data_available)
        self.assertFalse(
            any(
                row["raw_numeric_component_source"] == "YES"
                and row["usable_for_component_resolution"] == "YES"
                for row in rows
            )
        )
        mock_rows = [row for row in rows if row["source_type"] == "CSV_SYNTHETIC_FIXTURE"]
        self.assertEqual(len(mock_rows), 4)
        self.assertTrue(all(row["usable_for_component_resolution"] == "NO" for row in mock_rows))
        prior = next(row for row in rows if row["source_type"] == "CSV_ENGINEERING_PRIOR")
        self.assertEqual(prior["provenance_quality"], "ESTIMATED_DEFAULT_PRIOR_SPLIT")
        self.assertEqual(prior["usable_for_component_resolution"], "NO")

    def test_coverage_freezes_ten_dimensions_without_implementing_rag(self) -> None:
        rows = self._rows(self.output / "COMPONENT_KNOWLEDGE_COVERAGE.csv")
        self.assertEqual([int(row["dimension_number"]) for row in rows], list(range(1, 11)))
        self.assertEqual([row["status"] for row in rows[:4]], [
            "PARTIAL_NOW",
            "PARTIAL_NOW",
            "NOT_SUPPORTED_BY_CURRENT_SOURCE",
            "PARTIAL_NOW",
        ])
        self.assertTrue(all(row["status"] == "DEFERRED_RAG" for row in rows[4:]))
        self.assertEqual(self.report.implemented_dimensions, 0)
        self.assertEqual(self.report.partial_dimensions, 3)

    def test_candidate_is_read_only_and_integrity_remains_clean(self) -> None:
        self.assertEqual(self.candidate_hash_before, self.candidate_hash_after)
        self.assertEqual(self.report.candidate_sha256_before, self.report.candidate_sha256_after)
        self.assertEqual(self.report.quick_check, "ok")
        self.assertEqual(self.report.foreign_key_issues, 0)
        self.assertEqual(self.report.component_db_count, 0)
        self.assertEqual(self.report.component_instance_count, 843)
        self.assertEqual(self.report.component_resolution_count, 0)
        self.assertEqual(self.report.vde_component_resolution_count, 0)
        self.assertEqual(self.report.tire_db_count, 1)

    def test_blocked_population_artifacts_are_explicitly_empty(self) -> None:
        reference_rows = self._rows(self.output / "COMPONENT_REFERENCE_POPULATIONS.csv")
        outlier_rows = self._rows(self.output / "COMPONENT_OUTLIER_REVIEW.csv")
        audit_rows = self._rows(self.output / "COMPONENT_SEED_AUDIT.csv")
        self.assertEqual(reference_rows, [])
        self.assertEqual(outlier_rows, [])
        self.assertEqual(len(audit_rows), 5)
        self.assertTrue(all(row["package_action"] == "NO_WRITE" for row in audit_rows))
        resolution = next(row for row in audit_rows if row["entity"] == "component_resolution")
        self.assertEqual(resolution["status"], "BLOCKED_NO_RAW_COMPONENT_LOSS_SOURCE")


if __name__ == "__main__":
    unittest.main()
