from __future__ import annotations

import csv
import json
import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12e1_clean_rebuild_migration as clean  # noqa: E402


class Sprint12E1CleanRebuildMigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(clean.SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.db = Path(cls.summary["output_database"])

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db.resolve().as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        return con

    def csv_rows(self, name: str) -> list[dict[str, str]]:
        with (clean.OUT / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_runtime_is_read_only_and_output_guarded(self) -> None:
        with clean.e12.open_readonly(clean.RUNTIME_DB) as con:
            self.assertEqual(con.execute("PRAGMA query_only").fetchone()[0], 1)
        with self.assertRaises(ValueError):
            clean.guard_output_path(clean.RUNTIME_DB)

    def test_clean_database_builds_with_nine_domain_entities(self) -> None:
        domain_tables = set(clean.e12.PHYSICAL_TABLES[:9])
        with self.connect() as con:
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(domain_tables, tables & domain_tables)
        self.assertEqual(self.summary["status"], "CLEAN_REBUILD_READY — PROCEED_TO_INTEGRATION")

    def test_full_current_source_population_was_processed(self) -> None:
        self.assertEqual(self.summary["epa"]["source_rows"], 30_194)
        self.assertEqual(self.summary["epa"]["loaded_rows"], 30_117)
        self.assertEqual(self.summary["jrc_program_count"], 249)

    def test_clean_rebuild_is_deterministic(self) -> None:
        self.assertTrue(self.summary["previous_signature_available"])
        self.assertTrue(self.summary["rebuild_deterministic"])

    def test_program_population_has_no_parallel_legacy_tree(self) -> None:
        self.assertLess(self.summary["canonical_counts"]["program"], 4_000)
        self.assertFalse(self.summary["duplicate_legacy_program_tree_retained"])
        with self.connect() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM program WHERE source_scope='LEGACY_ECODRIVE'").fetchone()[0], 0)

    def test_safe_program_consolidation_only(self) -> None:
        rows = self.csv_rows("program_consolidation_results.csv")
        self.assertEqual(len(rows), 5_715)
        self.assertFalse(any(row["decision"] == "PROBABLE_CONSOLIDATE" for row in rows))
        self.assertEqual(len({row["canonical_program_id"] for row in rows}), 2_868)

    def test_all_legacy_vde_and_fuelcons_have_disposition(self) -> None:
        rows = self.csv_rows("legacy_record_disposition.csv")
        self.assertEqual(len(rows), 10_007)
        self.assertFalse(any(not row["disposition"] for row in rows))

    def test_only_four_legacy_scenario_vdes_survive(self) -> None:
        with self.connect() as con:
            ids = {row[0] for row in con.execute("SELECT id FROM vde WHERE id>0")}
        self.assertEqual(ids, clean.DERIVED_VDE_IDS)

    def test_only_five_irreproducible_legacy_fuelcons_survive(self) -> None:
        with self.connect() as con:
            ids = {row[0] for row in con.execute("SELECT id FROM fuelcons WHERE id>0")}
        self.assertEqual(ids, clean.IRREPRODUCIBLE_FUELCONS_IDS)

    def test_irreproducible_states_have_explicit_classification(self) -> None:
        rows = self.csv_rows("legacy_irreproducible_state.csv")
        self.assertEqual(len(rows), 10)
        allowed = {"DERIVED_SCENARIO", "USER_CREATED", "MANUAL_CORRECTION", "ML_PREDICTION", "OTHER_IRREPRODUCIBLE_STATE"}
        self.assertTrue({row["classification"] for row in rows} <= allowed)
        self.assertNotIn("LEGACY", {row["classification"] for row in rows})

    def test_scenario_parent_matches_are_safe_or_explicitly_unresolved(self) -> None:
        rows = self.csv_rows("configuration_match_results.csv")
        self.assertEqual(len(rows), 4)
        unresolved = [row for row in rows if not row["canonical_parent_vde_id"]]
        self.assertEqual([row["scenario_vde_id"] for row in unresolved], ["5033"])
        self.assertEqual(unresolved[0]["match_classification"], "PROGRAM_MATCH_ONLY")

    def test_no_orphans_and_same_vde_adoption_invariant(self) -> None:
        checks = self.csv_rows("relationship_checks.csv")
        relevant = [row for row in checks if row["check"].startswith("orphan_") or row["check"] in {"foreign_key_check", "invalid_adoption_same_vde"}]
        self.assertTrue(relevant)
        self.assertTrue(all(row["status"] == "PASS" for row in relevant))

    def test_source_run_grain_allows_multiple_runs_per_vde(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM (SELECT vde_id FROM run GROUP BY vde_id HAVING COUNT(*)>1)").fetchone()[0]
        self.assertGreater(count, 0)

    def test_no_epa_fuelcons_is_invented_without_pacification(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM fuelcons WHERE source_name='EPA_TESTCAR_2014_PRESENT'").fetchone()[0]
        self.assertEqual(count, 0)
        row = next(row for row in self.csv_rows("fuelcons_materialization_results.csv") if row["fuelcons_id"] == "EPA_ALL")
        self.assertEqual(row["materialization"], "RUN_ONLY_NO_APPROVED_PACIFICATION")

    def test_nan_to_null_is_approved_and_original_is_preserved(self) -> None:
        self.assertEqual(self.summary["nan_to_null_classification"], "APPROVED_CONTRACT_CORRECTION")
        with self.connect() as con:
            canonical = con.execute("SELECT assumptions_json FROM fuelcons WHERE id=5018").fetchone()[0]
            provenance = con.execute("SELECT provenance_json FROM run WHERE source_record_id='5018' AND source_name='LEGACY_IRREPRODUCIBLE_STATE'").fetchone()[0]
        self.assertTrue(json.loads(canonical))
        self.assertNotIn("NaN", canonical)
        self.assertIn("NaN", json.loads(provenance)["original_assumptions_json"])

    def test_null_and_zero_remain_distinct_and_snapshots_are_immutable(self) -> None:
        checks = {row["check"]: row for row in self.csv_rows("relationship_checks.csv")}
        self.assertEqual(checks["master_update_does_not_rewrite_snapshot"]["status"], "PASS")
        with self.connect() as con:
            zero = con.execute("SELECT COUNT(*) FROM tire_db WHERE rr_n_per_kn=0").fetchone()[0]
            nulls = con.execute("SELECT COUNT(*) FROM vde WHERE vde_net_mj_per_km IS NULL").fetchone()[0]
        self.assertGreater(zero, 0)
        self.assertGreater(nulls, 0)

    def test_eea_stays_outside_operational_sqlite(self) -> None:
        self.assertEqual(self.summary["eea_runtime_rows_loaded"], 0)
        with self.connect() as con:
            names = {row[0].lower() for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertFalse(any("eea" in name for name in names))

    def test_performance_has_no_blockers(self) -> None:
        rows = self.csv_rows("performance_results.csv")
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))

    def test_runtime_fingerprints_remain_byte_identical(self) -> None:
        rows = self.csv_rows("runtime_db_fingerprints.csv")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["byte_identical"] == "True" for row in rows))
        self.assertFalse(self.summary["runtime_db_changed"])


if __name__ == "__main__":
    unittest.main()
