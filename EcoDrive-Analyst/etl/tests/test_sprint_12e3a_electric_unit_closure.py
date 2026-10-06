from __future__ import annotations

import csv
import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
import sprint_12e3_electric_unit_audit as base  # noqa: E402
import sprint_12e3a_electric_unit_closure as sprint  # noqa: E402


class Sprint12E3AElectricUnitClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(base.SUMMARY.read_text(encoding="utf-8"))

    def rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_01_example_29_9_kwh_per_100mi(self) -> None:
        value = base.kwh_per_100mi_to_wh_per_km(29.9)
        self.assertTrue(math.isclose(value, 185.78998647896285))
        self.assertTrue(math.isclose(value / 10, 18.578998647896285))

    def test_02_resolved_527_records_use_kwh_per_100mi(self) -> None:
        rows = self.rows(base.OUT / "historical_manual_corrections.csv")
        resolved = [row for row in rows if row["status"] == "RESOLVED"]
        self.assertEqual(len(resolved), 527)
        self.assertTrue(all(row["interpreted_unit"] == "kWh/100mi" for row in resolved))
        self.assertTrue(all(row["correction_type"] == "DETERMINISTIC_UNIT_INTERPRETATION" for row in resolved))

    def test_03_raw_values_are_unchanged(self) -> None:
        overrides = self.rows(base.OVERRIDES)
        active = [row for row in overrides if row["validation_status"] == "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION"]
        self.assertEqual(len(active), 527)
        self.assertTrue(all(row["corrected_value"] == "" for row in active))

    def test_04_two_crv_fcev_zeros_are_not_converted(self) -> None:
        rows = self.rows(base.OUT / "historical_manual_corrections.csv")
        zeros = [row for row in rows if row["source_record_id"] in {"EPA-LEGACY-ROW-1836", "EPA-LEGACY-ROW-1838"}]
        self.assertEqual(len(zeros), 2)
        self.assertTrue(all(row["canonical_wh_per_km"] == "" for row in zeros))
        self.assertTrue(all(row["status"] == "NOT_APPLICABLE_OR_UNRESOLVED_FOR_ELECTRIC_RULE" for row in zeros))

    def test_05_zero_never_becomes_infinity(self) -> None:
        with self.assertRaises(ValueError):
            base.kwh_per_100mi_to_wh_per_km(0)
        sanity = self.rows(base.OUT / "electric_energy_sanity_dataset.csv")
        self.assertFalse(any(str(value).casefold() in {"inf", "infinity"} for row in sanity for value in row.values()))

    def test_06_zero_never_becomes_canonical_zero_energy(self) -> None:
        sanity = self.rows(base.OUT / "electric_energy_sanity_dataset.csv")
        excluded = [row for row in sanity if row["correction_status"] == "EXCLUDED_FCEV_ZERO_UNRESOLVED"]
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(row["raw_value"] == "0.0" and row["canonical_wh_per_km"] == "" for row in excluded))

    def test_07_six_absent_rows_remain_retired(self) -> None:
        rows = self.rows(base.OVERRIDES)
        self.assertEqual(sum(row["validation_status"] == "RETIRED_SOURCE_RECORD_ABSENT" for row in rows), 6)

    def test_08_active_overrides_are_exactly_guarded(self) -> None:
        override = next(row for row in self.rows(base.OVERRIDES) if row["validation_status"] == "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION")
        source = {key: override[key] for key in ("source_system", "source_file_version", "source_record_id", "raw_field")}
        source["raw_value"] = override["expected_raw_value"]
        result = base.apply_override(source, override)
        self.assertEqual(result["provenance_class"], "DETERMINISTIC_UNIT_INTERPRETATION")

    def test_09_stale_override_mismatch_fails(self) -> None:
        override = next(row for row in self.rows(base.OVERRIDES) if row["validation_status"] == "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION")
        source = {key: override[key] for key in ("source_system", "source_file_version", "source_record_id", "raw_field")}
        source["raw_value"] = "STALE"
        with self.assertRaises(base.StaleOverrideError):
            base.apply_override(source, override)

    def test_10_all_1257_electricity_cases_are_dispositioned(self) -> None:
        rows = self.rows(base.OUT / "resolved_12e2_electric_cases.csv")
        electricity = [row for row in rows if row["fuel_type"] == "Electricity"]
        counts = Counter(row["sprint_12e3a_disposition"] for row in electricity)
        self.assertEqual(len(electricity), 1257)
        self.assertEqual(counts, {"NOW_RESOLVED_KWH_PER_100MI": 276, "NOT_SAME_SOURCE_SEMANTICS": 915, "CONFLICTING_SOURCE_CONTEXT": 66})

    def test_11_all_32_hydrogen_cases_stay_outside_rule(self) -> None:
        rows = self.rows(base.OUT / "resolved_12e2_electric_cases.csv")
        hydrogen = [row for row in rows if row["fuel_type"] == "Hydrogen 5"]
        self.assertEqual(len(hydrogen), 32)
        self.assertTrue(all(row["sprint_12e3a_disposition"] == "HYDROGEN_OUTSIDE_ELECTRIC_RULE" for row in hydrogen))

    def test_12_unit_resolution_does_not_bypass_materialization(self) -> None:
        rows = self.rows(base.OUT / "resolved_12e2_electric_cases.csv")
        resolved = [row for row in rows if row["sprint_12e3a_disposition"] == "NOW_RESOLVED_KWH_PER_100MI"]
        self.assertEqual(len(resolved), 276)
        self.assertTrue(all(row["materialized_additional_fuelcons"] == "False" for row in resolved))
        self.assertEqual(self.summary["additional_canonical_fuelcons"], 0)

    def test_13_runtime_databases_are_unchanged(self) -> None:
        self.assertFalse(self.summary["runtime_db_changed"])
        self.assertEqual(self.summary["runtime_fingerprints_before"], self.summary["runtime_fingerprints_after"])

    def test_14_rebuild_is_deterministic_and_ready(self) -> None:
        self.assertTrue(self.summary["previous_signature_available"])
        self.assertTrue(self.summary["rebuild_deterministic"])
        self.assertEqual(self.summary["status"], "ELECTRIC_UNITS_READY — PROCEED_TO_12F_NOTEBOOKS")
        self.assertEqual(self.summary["user_decisions_required"], 0)


if __name__ == "__main__":
    unittest.main()
