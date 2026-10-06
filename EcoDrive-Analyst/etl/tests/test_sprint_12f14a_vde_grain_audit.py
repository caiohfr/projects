from __future__ import annotations

import csv
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
import sprint_12f14a_vde_grain_audit as sprint  # noqa: E402


class Sprint12F14AVDEGrainAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(sprint.SUMMARY.read_text(encoding="utf-8"))

    def csv_rows(self, name: str) -> list[dict[str, str]]:
        with (sprint.OUT / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_01_input_database_is_unchanged(self) -> None:
        self.assertEqual(self.summary["input_hash_before"], self.summary["input_hash_after"])
        self.assertFalse(self.summary["input_candidate_changed"])

    def test_02_runtime_database_hashes_are_unchanged(self) -> None:
        self.assertEqual(self.summary["runtime_hashes_before"], self.summary["runtime_hashes_after"])
        self.assertFalse(self.summary["runtime_db_changed"])

    def test_03_every_current_vde_maps_to_exactly_one_group(self) -> None:
        rows = self.csv_rows("vde_group_membership.csv")
        counts = Counter(row["vde_id"] for row in rows)
        self.assertEqual(len(rows), self.summary["counts"]["current_vde_rows"])
        self.assertTrue(all(count == 1 for count in counts.values()))
        self.assertTrue(self.summary["grouping"]["every_vde_mapped_once"])

    def test_04_candidate_grouping_is_deterministic(self) -> None:
        grouping = self.summary["grouping"]
        self.assertTrue(grouping["deterministic"])
        self.assertEqual(grouping["group_signature"], grouping["repeat_group_signature"])
        self.assertEqual(grouping["membership_signature"], grouping["repeat_membership_signature"])

    def test_05_grouping_tolerances_are_explicit(self) -> None:
        grouping = self.summary["grouping"]
        self.assertEqual(grouping["float_decimal_places"], 9)
        self.assertEqual(grouping["physical_absolute_tolerance"], 1e-9)
        self.assertEqual(grouping["result_absolute_tolerance"], 1e-10)
        self.assertEqual(grouping["result_relative_tolerance"], 1e-9)

    def test_06_grouping_key_does_not_use_vde_id(self) -> None:
        fields = self.summary["grouping"]["physical_signature_fields"]
        self.assertNotIn("vde_id", fields)
        self.assertNotIn("id", fields)

    def test_07_grouping_key_excludes_source_test_identity(self) -> None:
        fields = self.summary["grouping"]["physical_signature_fields"]
        for forbidden in ("cycle_name", "cycle_source", "source_name", "source_record_id", "run_id"):
            self.assertNotIn(forbidden, fields)
        self.assertTrue(self.summary["grouping"]["keys_exclude_vde_and_source_test_identity"])

    def test_08_reported_physical_matches_share_selected_state(self) -> None:
        memberships = self.csv_rows("vde_group_membership.csv")
        by_group: dict[str, list[dict[str, str]]] = {}
        for row in memberships:
            by_group.setdefault(row["candidate_group_id"], []).append(row)
        for members in by_group.values():
            signatures = {
                (row["vehicle_configuration_id"], row["legislation"], row["effective_calc_mass_kg"], row["coast_A_N"], row["coast_B_N_per_kph"], row["coast_C_N_per_kph2"])
                for row in members
            }
            self.assertEqual(len(signatures), 1)

    def test_09_result_mismatches_are_surfaced_not_hidden(self) -> None:
        families = self.csv_rows("vde_source_grain_family_audit.csv")
        mismatches = [row for row in families if row["result_parity_status"] in {"RESULT_MISMATCH", "PARTIAL_NULL"}]
        self.assertEqual(len(mismatches), self.summary["counts"]["family_result_mismatches"])
        self.assertTrue(all(row["review_required"] == "True" for row in mismatches))

    def test_10_run_fuelcons_relationships_are_inspected_not_mutated(self) -> None:
        self.assertEqual(self.summary["database_rows_mutated"], 0)
        self.assertEqual(self.summary["database_rows_inspected"]["vde"], self.summary["counts"]["current_vde_rows"])
        self.assertEqual(self.summary["status"], sprint.STATUS)
        self.assertEqual(self.summary["focused_assertions"], {"passed": 10, "total": 10})


if __name__ == "__main__":
    unittest.main()
