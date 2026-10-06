from __future__ import annotations

import csv
import json
import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12e_migration_rehearsal as rehearsal  # noqa: E402


class Sprint12EMigrationRehearsalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(rehearsal.PREVIOUS_SUMMARY.read_text(encoding="utf-8"))
        cls.db = Path(cls.summary["output_database"])

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db.resolve().as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        return con

    def test_runtime_database_is_opened_read_only(self) -> None:
        with rehearsal.open_readonly(rehearsal.RUNTIME_DB) as con:
            self.assertEqual(con.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                con.execute("CREATE TABLE forbidden_write(id INTEGER)")

    def test_protected_runtime_paths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rehearsal.guard_output_path(rehearsal.RUNTIME_DB)
        with self.assertRaises(ValueError):
            rehearsal.guard_output_path(rehearsal.QA_DB)
        with self.assertRaises(ValueError):
            rehearsal.guard_output_path(rehearsal.ROOT / "etl" / "reports" / "not_a_database.md")

    def test_empty_schema_and_full_rehearsal_database_built(self) -> None:
        with sqlite3.connect(":memory:") as con:
            con.executescript(rehearsal.SCHEMA_SQL.read_text(encoding="utf-8"))
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(set(rehearsal.PHYSICAL_TABLES), tables & set(rehearsal.PHYSICAL_TABLES))
        self.assertTrue(self.db.exists())
        self.assertEqual(self.summary["status"], "MIGRATION_REHEARSAL_REVIEW_REQUIRED")

    def test_migration_completed_on_all_available_source_boundaries(self) -> None:
        self.assertEqual(self.summary["epa"]["source_rows"], 30_194)
        self.assertEqual(self.summary["jrc_loaded_rows"], 249)
        self.assertEqual(self.summary["eea_audited_source_rows"], 10_833_597)
        self.assertEqual(self.summary["eea_runtime_rows_loaded"], 0)

    def test_clean_rebuild_is_deterministic(self) -> None:
        self.assertTrue(self.summary["previous_signature_available"])
        self.assertTrue(self.summary["rebuild_deterministic"])

    def test_no_orphan_program_configuration_vde_run_or_fuelcons(self) -> None:
        with (rehearsal.OUT / "relationship_checks.csv").open(encoding="utf-8", newline="") as handle:
            checks = list(csv.DictReader(handle))
        orphan_checks = [row for row in checks if row["check"].startswith("orphan_") or row["check"] == "foreign_key_check"]
        self.assertTrue(orphan_checks)
        self.assertTrue(all(row["status"] == "PASS" for row in orphan_checks))

    def test_duplicate_legacy_vde_states_are_preserved(self) -> None:
        with (rehearsal.OUT / "relationship_checks.csv").open(encoding="utf-8", newline="") as handle:
            row = next(row for row in csv.DictReader(handle) if row["check"] == "duplicate_legacy_vde_states_preserved")
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["actual"], "46")

    def test_one_vde_may_have_multiple_fuelcons(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM (SELECT vde_id FROM fuelcons GROUP BY vde_id HAVING COUNT(*)>1)").fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_one_vde_may_have_multiple_runs(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM (SELECT vde_id FROM run GROUP BY vde_id HAVING COUNT(*)>1)").fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_fuelcons_multi_run_same_vde_invariant_is_enforced(self) -> None:
        source = sqlite3.connect(self.db)
        memory = sqlite3.connect(":memory:")
        try:
            source.backup(memory)
            memory.execute("PRAGMA foreign_keys=ON")
            fuel = memory.execute("SELECT id,vde_id FROM fuelcons ORDER BY id LIMIT 1").fetchone()
            other_run = memory.execute("SELECT run_id,vde_id FROM run WHERE vde_id<>? LIMIT 1", (fuel[1],)).fetchone()
            with self.assertRaises(sqlite3.IntegrityError):
                memory.execute("INSERT INTO fuelcons_run_adoption(fuelcons_id,run_id,vde_id,adoption_role) VALUES(?,?,?,?)", (fuel[0], other_run[0], fuel[1], "SUPPORTING"))
        finally:
            memory.close()
            source.close()

    def test_vde_can_exist_without_component_resolution(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM vde v LEFT JOIN vde_component_resolution x ON x.vde_id=v.id WHERE x.vde_id IS NULL").fetchone()[0]
        self.assertGreater(count, 0)

    def test_historical_master_update_does_not_rewrite_snapshot(self) -> None:
        with (rehearsal.OUT / "relationship_checks.csv").open(encoding="utf-8", newline="") as handle:
            row = next(row for row in csv.DictReader(handle) if row["check"] == "master_update_does_not_rewrite_snapshot")
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["actual"], "135.0")

    def test_null_stays_null_for_every_legacy_field(self) -> None:
        with (rehearsal.OUT / "compatibility_checks.csv").open(encoding="utf-8", newline="") as handle:
            checks = list(csv.DictReader(handle))
        self.assertEqual(len(checks), 180)
        self.assertTrue(all(row["legacy_null_count"] == row["canonical_null_count"] for row in checks))

    def test_compatibility_vde_legacy_column_and_value_equivalence(self) -> None:
        with (rehearsal.OUT / "compatibility_checks.csv").open(encoding="utf-8", newline="") as handle:
            checks = [row for row in csv.DictReader(handle) if row["entity"] == "VDE"]
        self.assertEqual(len(checks), 101)
        self.assertTrue(all(row["classification"] == "EXACT_EQUIVALENCE" for row in checks))

    def test_compatibility_fuelcons_reports_known_json_constraint_conflict(self) -> None:
        with (rehearsal.OUT / "compatibility_checks.csv").open(encoding="utf-8", newline="") as handle:
            checks = [row for row in csv.DictReader(handle) if row["entity"] == "FUELCONS"]
        self.assertEqual(len(checks), 79)
        failures = [row for row in checks if row["classification"] != "EXACT_EQUIVALENCE"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["field"], "assumptions_json")
        self.assertEqual(failures[0]["mismatch_count"], "1")

    def test_current_runtime_database_fingerprint_is_unchanged(self) -> None:
        self.assertFalse(self.summary["runtime_db_changed"])
        self.assertEqual(self.summary["runtime_before_sha256"], self.summary["runtime_after_sha256"])

    def test_eea_follows_approved_analytical_storage_boundary(self) -> None:
        self.assertEqual(self.summary["eea_runtime_rows_loaded"], 0)
        with self.connect() as con:
            names = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertFalse(any("eea" in name.lower() for name in names))


if __name__ == "__main__":
    unittest.main()
