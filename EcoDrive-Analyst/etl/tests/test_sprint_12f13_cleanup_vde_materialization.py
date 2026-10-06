from __future__ import annotations

import csv
import inspect
import json
import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))
import sprint_12f13_cleanup_vde_materialization as sprint  # noqa: E402
from src.vde_core.vehicle_demand.adapters import build_vehicle_demand_request, resolve_vehicle_demand_cycle  # noqa: E402
from src.vde_core.vehicle_demand.engine import calculate_vehicle_demand  # noqa: E402


class Sprint12F13CleanupVDEMaterializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(sprint.SUMMARY.read_text(encoding="utf-8"))
        cls.db = Path(cls.summary["output_database"])

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def csv_rows(self, name: str) -> list[dict[str, str]]:
        with (sprint.OUT / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_01_exact_four_known_test_vdes_are_absent(self) -> None:
        with self.connect() as connection:
            found = connection.execute("SELECT id FROM vde WHERE id IN (5031,5033,5034,5038)").fetchall()
        self.assertEqual(found, [])
        self.assertEqual(self.summary["removed_counts"]["vde"], 4)

    def test_02_real_epa_jrc_population_not_removed_by_labels(self) -> None:
        self.assertTrue(self.summary["real_source_population_preserved"])
        self.assertEqual(self.summary["preservation_signatures_before"]["real_vde_identity"], self.summary["preservation_signatures_after"]["real_vde_identity"])

    def test_03_dependent_fuelcons_adoptions_and_runs_removed_safely(self) -> None:
        self.assertEqual(self.summary["removed_counts"]["fuelcons"], 4)
        self.assertEqual(self.summary["removed_counts"]["fuelcons_run_adoption"], 4)
        self.assertEqual(self.summary["removed_counts"]["run"], 8)
        with self.connect() as connection:
            for table in ("fuelcons", "fuelcons_run_adoption", "run"):
                self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM {table} WHERE vde_id IN (5031,5033,5034,5038)").fetchone()[0], 0)

    def test_04_orphan_artificial_parent_cleanup_is_exact(self) -> None:
        audit = self.csv_rows("test_artifact_cleanup.csv")
        configs = [row["entity_id"] for row in audit if row["entity_table"] == "vehicle_configuration" and row["action"] == "DELETE"]
        programs = [row["entity_id"] for row in audit if row["entity_table"] == "program" and row["action"] == "PRESERVE"]
        self.assertEqual(len(configs), 4)
        with self.connect() as connection:
            self.assertEqual(sum(connection.execute("SELECT COUNT(*) FROM vehicle_configuration WHERE vehicle_configuration_id=?", (item,)).fetchone()[0] for item in configs), 0)
            self.assertEqual(sum(connection.execute("SELECT COUNT(*) FROM program WHERE program_id=?", (item,)).fetchone()[0] for item in set(programs)), len(set(programs)))

    def test_05_reusable_master_records_are_not_over_deleted(self) -> None:
        for table in ("component_db", "tire_db"):
            self.assertEqual(self.summary["preservation_signatures_before"][table], self.summary["preservation_signatures_after"][table])
        self.assertEqual(self.summary["ml_fuelcons"]["action"], "PRESERVED")

    def test_06_canonical_vehicle_demand_engine_is_used(self) -> None:
        self.assertIn("calculate_vehicle_demand", inspect.getsource(sprint.calculated_values))
        with self.connect() as connection:
            row = dict(connection.execute("SELECT * FROM vde ORDER BY id LIMIT 1").fetchone())
        with sprint.repository_working_directory():
            request = build_vehicle_demand_request(row)
            expected = calculate_vehicle_demand(request, resolve_vehicle_demand_cycle(row)).total_summary.vde_mj_per_km
        self.assertTrue(sprint.values_match(row["vde_total_mj_per_km"], expected))
        provenance = json.loads(row["source_payload_json"])[sprint.MATERIALIZATION_KEY]
        self.assertEqual(provenance["engine"], self.summary["engine"])

    def test_07_total_materialization_is_deterministic(self) -> None:
        self.assertEqual(self.summary["materialization"]["total_newly_materialized"], self.summary["candidate_counts"]["vde"])
        self.assertEqual(self.summary["sanity"]["non_null"], self.summary["candidate_counts"]["vde"])
        self.assertEqual(self.summary["sanity"]["zero_or_negative"], 0)
        flags = self.csv_rows("vde_materialization_sanity_flags.csv")
        self.assertEqual(len(flags), self.summary["sanity"]["review_flags"])
        self.assertTrue(all(row["action"] == "FLAG_ONLY_NO_AUTO_DELETE" for row in flags))

    def test_08_existing_values_are_parity_checked_before_overwrite(self) -> None:
        self.assertTrue(sprint.values_match(1.0, 1.0 + sprint.ABS_TOLERANCE / 2))
        self.assertFalse(sprint.values_match(1.0, 1.1))
        self.assertEqual(self.summary["materialization"]["total_mismatches"], 0)

    def test_09_unsupported_or_missing_inputs_remain_null_with_reason(self) -> None:
        with self.connect() as connection:
            base = dict(connection.execute("SELECT * FROM vde ORDER BY id LIMIT 1").fetchone())
        base["id"] = 999999999
        base["coast_A_N"] = None
        result, values = sprint.materialize_row(base, {}, {})
        self.assertEqual(result["total_status"], "MISSING_REQUIRED_INPUT")
        self.assertTrue(result["reason"])
        self.assertIsNone(values)

    def test_10_net_is_not_fabricated_without_loss_resolution(self) -> None:
        self.assertEqual(self.summary["materialization"]["net_newly_materialized"], 0)
        self.assertEqual(self.summary["materialization"]["net_legitimately_unavailable"], self.summary["candidate_counts"]["vde"])
        with self.connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM vde WHERE vde_net_mj_per_km IS NOT NULL").fetchone()[0], 0)

    def test_11_full_export_counts_equal_database_counts(self) -> None:
        self.assertTrue(self.summary["integrity"]["export_counts_match"])
        manifest = {row["table"]: int(row["rows"]) for row in self.csv_rows("table_export_manifest.csv")}
        self.assertEqual(manifest, self.summary["candidate_counts"])

    def test_12_zero_foreign_key_violations(self) -> None:
        with self.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")

    def test_13_same_vde_adoption_invariant_and_lineage(self) -> None:
        self.assertEqual(self.summary["integrity"]["adoption_relationship_failures"], 0)
        self.assertEqual(self.summary["integrity"]["fuelcons_missing_lineage"], 0)

    def test_14_runtime_hashes_are_unchanged(self) -> None:
        self.assertEqual(self.summary["runtime_and_demo_fingerprints_before"], self.summary["runtime_and_demo_fingerprints_after"])
        self.assertFalse(self.summary["runtime_db_changed"])
        with self.assertRaises(ValueError):
            sprint.guard_output_path(sprint.RUNTIME_DBS[0])

    def test_15_repeated_build_has_same_relationship_result_signature(self) -> None:
        self.assertTrue(self.summary["rebuild_deterministic"])
        self.assertEqual(self.summary["relationship_result_signature"], self.summary["repeat_relationship_result_signature"])
        self.assertEqual(self.summary["status"], sprint.STATUS_READY)


if __name__ == "__main__":
    unittest.main()
