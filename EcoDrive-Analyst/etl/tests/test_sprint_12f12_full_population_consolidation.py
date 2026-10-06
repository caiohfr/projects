from __future__ import annotations

import csv
import json
import sqlite3
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
import sprint_12f12_full_population_consolidation as sprint  # noqa: E402


class Sprint12F12FullPopulationConsolidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(sprint.SUMMARY.read_text(encoding="utf-8"))
        cls.db = Path(cls.summary["output_database"])

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db.resolve().as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        return connection

    def csv_rows(self, name: str) -> list[dict[str, str]]:
        with (sprint.OUT / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_01_rebuild_is_deterministic(self) -> None:
        self.assertTrue(self.summary["rebuild_deterministic"])
        self.assertEqual(self.summary["content_signature"], self.summary["repeat_content_signature"])
        self.assertEqual(self.summary["status"], sprint.STATUS)

    def test_02_runtime_and_demo_are_unchanged(self) -> None:
        self.assertEqual(self.summary["runtime_and_demo_fingerprints_before"], self.summary["runtime_and_demo_fingerprints_after"])
        self.assertFalse(self.summary["runtime_db_changed"])
        self.assertFalse(self.summary["demo_db_changed"])
        with self.assertRaises(ValueError):
            sprint.guard_output_path(sprint.RUNTIME_DBS[0])

    def test_03_foreign_keys_are_clean(self) -> None:
        with self.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_04_sqlite_quick_check_is_ok(self) -> None:
        with self.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")

    def test_05_primary_keys_are_unique(self) -> None:
        checks = self.csv_rows("integrity_checks.csv")
        pk = [row for row in checks if row["check"].startswith("PK_UNIQUENESS::")]
        self.assertEqual(len(pk), len(self.summary["candidate_counts"]))
        self.assertTrue(all(row["passed"] == "True" for row in pk))

    def test_06_adoptions_are_same_vde_and_lineage_is_complete(self) -> None:
        self.assertEqual(self.summary["same_vde_adoption_violations"], 0)
        self.assertEqual(self.summary["fuelcons_missing_run_lineage"], 0)

    def test_07_every_table_export_matches_database(self) -> None:
        self.assertTrue(self.summary["export_counts_match"])
        manifest = {row["table"]: int(row["rows"]) for row in self.csv_rows("table_export_manifest.csv")}
        self.assertEqual(manifest, self.summary["candidate_counts"])

    def test_08_electric_rules_and_case_dispositions_are_applied(self) -> None:
        records = self.csv_rows("electric_unit_record_population.csv")
        active = [row for row in records if row["validation_status"] == "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION"]
        self.assertEqual(len(active), 527)
        self.assertTrue(all(row["interpreted_unit"] == "kWh/100mi" and row["canonical_wh_per_km"] for row in active))
        self.assertEqual(Counter(row["candidate_representation"] for row in active), {"CANONICAL_RUN_EVIDENCE": 521, "SOURCE_ONLY_QUARANTINED_IDENTITY_ANOMALY": 6})
        dispositions = Counter(row["sprint_12e3a_disposition"] for row in self.csv_rows("electric_case_dispositions.csv"))
        self.assertEqual(dispositions["NOW_RESOLVED_KWH_PER_100MI"], 276)
        self.assertEqual(dispositions["NOT_SAME_SOURCE_SEMANTICS"], 915)
        self.assertEqual(dispositions["CONFLICTING_SOURCE_CONTEXT"], 66)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT result_details_json FROM run "
                "WHERE json_type(result_details_json, '$.sprint_12e3a_electric_unit_closure')='object'"
            ).fetchall()
        annotations = [json.loads(row[0])[sprint.ANNOTATION_KEY] for row in rows]
        units = [item for annotation in annotations for item in annotation["unit_interpretations"]]
        self.assertEqual(len(annotations), 2689)
        self.assertEqual(sum(item["canonical_wh_per_km"] is not None for item in units), 521)
        self.assertTrue(all(item["fuelcons_materialized"] is False for item in units))

    def test_09_hydrogen_fcev_and_retired_exclusions_hold(self) -> None:
        records = self.csv_rows("electric_unit_record_population.csv")
        zeros = [row for row in records if row["validation_status"] == "EXCLUDED_FCEV_ZERO_UNRESOLVED"]
        self.assertEqual(len(zeros), 2)
        self.assertTrue(all(row["raw_value"] == "0.0" and row["canonical_wh_per_km"] == "" for row in zeros))
        self.assertEqual(sum(row["validation_status"] == "RETIRED_SOURCE_RECORD_ABSENT" for row in records), 6)
        dispositions = self.csv_rows("electric_case_dispositions.csv")
        hydrogen = [row for row in dispositions if row["fuel_type"] == "Hydrogen 5"]
        self.assertEqual(len(hydrogen), 32)
        self.assertTrue(all(row["sprint_12e3a_disposition"] == "HYDROGEN_OUTSIDE_ELECTRIC_RULE" for row in hydrogen))

    def test_10_unit_resolution_does_not_force_fuelcons(self) -> None:
        self.assertEqual(self.summary["electric"]["new_electric_fuelcons"], 0)
        self.assertEqual(self.summary["source_counts"]["fuelcons"], self.summary["candidate_counts"]["fuelcons"])
        with self.connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND electrification='BEV'").fetchone()[0]
        self.assertEqual(count, 0)

    def test_11_legacy_baseline_is_not_duplicated(self) -> None:
        self.assertEqual(self.summary["legacy_positive_fuelcons_retained"], 5)
        self.assertEqual(self.summary["legacy_epa_fuelcons_duplicated"], 0)
        self.assertEqual(self.summary["source_schema_signature"], self.summary["candidate_schema_signature"])


if __name__ == "__main__":
    unittest.main()
