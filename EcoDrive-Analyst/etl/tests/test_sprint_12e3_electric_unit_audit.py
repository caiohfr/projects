from __future__ import annotations

import csv
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12e3_electric_unit_audit as sprint  # noqa: E402


class Sprint12E3ElectricUnitAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(sprint.SUMMARY.read_text(encoding="utf-8"))

    def rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_01_mi_per_kwh_to_wh_per_km(self) -> None:
        self.assertTrue(math.isclose(sprint.mi_per_kwh_to_wh_per_km(4), 155.34279805933347))

    def test_02_km_per_kwh_to_wh_per_km(self) -> None:
        self.assertEqual(sprint.km_per_kwh_to_wh_per_km(5), 200)

    def test_03_kwh_per_100mi_to_wh_per_km(self) -> None:
        self.assertTrue(math.isclose(sprint.kwh_per_100mi_to_wh_per_km(30), 186.41135767120018))

    def test_04_wh_per_mile_to_wh_per_km(self) -> None:
        self.assertTrue(math.isclose(sprint.wh_per_mile_to_wh_per_km(300), 186.41135767120018))

    def test_05_mpge_requires_authoritative_basis(self) -> None:
        with self.assertRaises(sprint.UnsupportedUnitError):
            sprint.mpge_to_wh_per_km(100)
        self.assertTrue(math.isclose(sprint.mpge_to_wh_per_km(100, 33705), 209.4331603435934))

    def test_06_ambiguous_unit_does_not_auto_convert(self) -> None:
        self.assertIsNone(sprint.convert_to_wh_per_km(30, "MPG"))
        self.assertIsNone(sprint.convert_to_wh_per_km(30, "unknown/mixed"))

    def test_07_missing_value_stays_null(self) -> None:
        self.assertIsNone(sprint.mi_per_kwh_to_wh_per_km(None))
        self.assertIsNone(sprint.convert_to_wh_per_km(float("nan"), "Wh/km"))

    def test_08_zero_is_not_treated_as_missing(self) -> None:
        with self.assertRaises(ValueError):
            sprint.kwh_per_100mi_to_wh_per_km(0)

    def test_09_stale_override_does_not_silently_apply(self) -> None:
        source = {"source_system": "EPA", "source_file_version": "v2", "source_record_id": "1", "raw_field": "RND_ADJ_FE", "raw_value": "30"}
        override = {**source, "source_file_version": "v1", "expected_raw_value": "30", "corrected_unit": "kWh/100mi", "validation_status": "ACTIVE_USER_VALIDATED"}
        with self.assertRaises(sprint.StaleOverrideError):
            sprint.apply_override(source, override)

    def test_10_override_matches_source_specific_key_and_version(self) -> None:
        source = {"source_system": "EPA", "source_file_version": "v1", "source_record_id": "1", "raw_field": "RND_ADJ_FE", "raw_value": "30"}
        override = {**source, "expected_raw_value": "30", "corrected_unit": "kWh/100mi", "corrected_value": "", "validation_status": "ACTIVE_USER_VALIDATED"}
        result = sprint.apply_override(source, override)
        self.assertTrue(math.isclose(result["canonical_wh_per_km"], sprint.kwh_per_100mi_to_wh_per_km(30)))

    def test_11_manual_correction_preserves_provenance(self) -> None:
        source = {"source_system": "EPA", "source_file_version": "v1", "source_record_id": "1", "raw_field": "RND_ADJ_FE", "raw_value": "30"}
        override = {**source, "expected_raw_value": "30", "corrected_unit": "kWh/100mi", "corrected_value": "", "validation_status": "ACTIVE_USER_VALIDATED"}
        self.assertEqual(sprint.apply_override(source, override)["provenance_class"], "USER_VALIDATED_CORRECTION")

    def test_12_refreshed_source_can_retire_old_override(self) -> None:
        source = {"source_system": "EPA", "source_file_version": "v2", "source_record_id": "1", "raw_field": "RND_ADJ_FE", "raw_value": "30"}
        retired = {**source, "corrected_unit": "kWh/100mi", "validation_status": "RETIRED_SOURCE_RECORD_ABSENT"}
        self.assertEqual(sprint.apply_override(source, retired), source)
        registry = self.rows(sprint.OVERRIDES)
        self.assertEqual(sum(row["validation_status"] == "RETIRED_SOURCE_RECORD_ABSENT" for row in registry), 6)

    def test_13_runtime_databases_remain_unchanged(self) -> None:
        self.assertFalse(self.summary["runtime_db_changed"])
        self.assertEqual(self.summary["runtime_fingerprints_before"], self.summary["runtime_fingerprints_after"])

    def test_14_all_12e2_electric_cases_are_dispositioned(self) -> None:
        previous = [row for row in self.rows(sprint.OUT_12E2 / "unresolved_materialization_cases.csv") if row["reason_code"] == "UNSUPPORTED_ENERGY_OR_EQUIVALENT_FUEL_UNIT"]
        dispositions = self.rows(sprint.OUT / "resolved_12e2_electric_cases.csv")
        self.assertEqual(len(dispositions), len(previous))
        self.assertTrue(all(row.get("sprint_12e3a_disposition") or row.get("sprint_12e3_disposition") for row in dispositions))

    def test_15_canonical_output_is_deterministic(self) -> None:
        self.assertTrue(self.summary["previous_signature_available"])
        self.assertTrue(self.summary["rebuild_deterministic"])
        self.assertEqual(self.summary["status"], "ELECTRIC_UNITS_READY — PROCEED_TO_12F_NOTEBOOKS")


if __name__ == "__main__":
    unittest.main()
