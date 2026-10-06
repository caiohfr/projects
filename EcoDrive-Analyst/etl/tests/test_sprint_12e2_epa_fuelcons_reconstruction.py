from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12e2_epa_fuelcons_reconstruction as sprint  # noqa: E402


class Sprint12E2EPAFuelConsReconstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(sprint.SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.db = Path(cls.summary["output_database"])

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db.resolve().as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        return con

    def csv_rows(self, name: str) -> list[dict[str, str]]:
        with (sprint.OUT / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_source_row_does_not_automatically_imply_one_run(self) -> None:
        self.assertLess(self.summary["epa_runs_after"], self.summary["epa_runs_before"])

    def test_source_row_lineage_is_complete_after_grouping(self) -> None:
        rows = self.csv_rows("run_source_row_lineage.csv")
        self.assertEqual(len(rows), self.summary["epa_source_rows_loaded"])
        self.assertEqual(len({int(row["source_excel_row"]) for row in rows}), len(rows))

    def test_multiple_source_rows_can_map_to_one_run(self) -> None:
        rows = self.csv_rows("run_grouping_results.csv")
        self.assertTrue(any(int(row["source_row_count"]) > 1 for row in rows))

    def test_condition_conflicts_remain_distinct_runs(self) -> None:
        rows = self.csv_rows("run_grouping_results.csv")
        by_test: dict[str, set[str]] = defaultdict(set)
        conflict_keys: set[str] = set()
        for row in rows:
            by_test[row["broad_test_key"]].add(row["canonical_run_id"])
            if row["condition_or_result_conflicts"]:
                conflict_keys.add(row["broad_test_key"])
        self.assertTrue(conflict_keys)
        self.assertTrue(all(len(by_test[key]) > 1 for key in conflict_keys))

    def test_one_fuelcons_can_adopt_multiple_runs(self) -> None:
        counts = Counter(row["fuelcons_id"] for row in self.csv_rows("fuelcons_run_adoption.csv"))
        self.assertTrue(any(value >= 2 for value in counts.values()))

    def test_fuelcons_creation_is_deterministic(self) -> None:
        self.assertTrue(self.summary["previous_signature_available"])
        self.assertTrue(self.summary["rebuild_deterministic"])

    def test_repeated_conflicting_runs_are_not_selected_arbitrarily(self) -> None:
        unresolved = self.csv_rows("unresolved_materialization_cases.csv")
        conflicts = [row for row in unresolved if "MULTIPLE_CONFLICTING_ELIGIBLE_RUNS" in row["reason_code"]]
        self.assertTrue(conflicts)
        created = {(row["canonical_vde_id"], row["comparison_basis"], row["fuel_type"], row["cycle"]) for row in self.csv_rows("fuelcons_materialization_results.csv")}
        self.assertFalse(any((row["canonical_vde_id"], row["comparison_basis"], row["fuel_type"], row["cycle"]) in created for row in conflicts))

    def test_epa_55_45_combination_uses_correct_units(self) -> None:
        with self.connect() as con:
            row = con.execute("SELECT fuel_ftp75_l_per_100km city, fuel_hwfet_l_per_100km hwy, fuel_l_per_100km combined FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND comparison_basis='EPA_LABEL_2_CYCLE' LIMIT 1").fetchone()
        self.assertIsNotNone(row)
        self.assertTrue(math.isclose(row["combined"], sprint.epa_combined_cons_l100(row["city"], row["hwy"]), rel_tol=1e-12))

    def test_unavailable_energy_and_range_remain_null(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND (energy_Wh_per_km IS NOT NULL OR label_range_km IS NOT NULL)").fetchone()[0]
        self.assertEqual(count, 0)

    def test_ice_applicability(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND electrification='ICE' AND fuel_l_per_100km IS NOT NULL").fetchone()[0]
        self.assertGreater(count, 0)

    def test_hev_applicability(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND electrification='HEV' AND fuel_l_per_100km IS NOT NULL").fetchone()[0]
        self.assertGreater(count, 0)

    def test_phev_applicability(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND electrification='PHEV' AND fuel_l_per_100km IS NOT NULL").fetchone()[0]
        self.assertGreater(count, 0)

    def test_bev_unsupported_unit_is_not_fabricated(self) -> None:
        with self.connect() as con:
            count = con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND electrification='BEV'").fetchone()[0]
        self.assertEqual(count, 0)
        matrix = {row["electrification"]: row for row in self.csv_rows("electrification_applicability_matrix.csv")}
        self.assertEqual(matrix["BEV"]["electric_energy"], "DEFERRED_UNIT_AMBIGUITY")

    def test_specialized_cycles_do_not_fill_urban_or_highway(self) -> None:
        with self.connect() as con:
            bad = con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND comparison_basis='EPA_TEST_PROCEDURE' AND (fuel_ftp75_l_per_100km IS NOT NULL OR fuel_hwfet_l_per_100km IS NOT NULL OR fuel_l_per_100km IS NOT NULL)").fetchone()[0]
            supported = con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND comparison_basis='EPA_TEST_PROCEDURE' AND (fuel_us06_l_per_100km IS NOT NULL OR fuel_sc03_l_per_100km IS NOT NULL)").fetchone()[0]
        self.assertEqual(bad, 0)
        self.assertGreater(supported, 0)

    def test_lineage_runs_belong_to_the_same_vde(self) -> None:
        with self.connect() as con:
            mismatches = con.execute("SELECT COUNT(*) FROM fuelcons_run_adoption a JOIN fuelcons f ON f.id=a.fuelcons_id JOIN run r ON r.run_id=a.run_id WHERE a.vde_id<>f.vde_id OR a.vde_id<>r.vde_id").fetchone()[0]
        self.assertEqual(mismatches, 0)

    def test_legacy_baseline_fuelcons_was_not_copied(self) -> None:
        with self.connect() as con:
            legacy_epa = con.execute("SELECT COUNT(*) FROM fuelcons WHERE record_origin='LEGACY' AND label_program='EPA'").fetchone()[0]
            retained_positive = con.execute("SELECT COUNT(*) FROM fuelcons WHERE id>0").fetchone()[0]
        self.assertEqual(legacy_epa, 0)
        self.assertEqual(retained_positive, 5)

    def test_runtime_databases_remain_byte_identical(self) -> None:
        rows = self.csv_rows("runtime_db_fingerprints.csv")
        self.assertTrue(all(row["byte_identical"] == "True" for row in rows))
        self.assertFalse(self.summary["runtime_db_changed"])

    def test_output_is_guarded_and_rebuild_is_deterministic(self) -> None:
        with self.assertRaises(ValueError):
            sprint.guard_output_path(sprint.RUNTIME_DB)
        self.assertEqual(self.summary["status"], "EPA_FUELCONS_READY — PROCEED_TO_APPLICATION_INTEGRATION")
        self.assertTrue(self.summary["rebuild_deterministic"])


if __name__ == "__main__":
    unittest.main()
