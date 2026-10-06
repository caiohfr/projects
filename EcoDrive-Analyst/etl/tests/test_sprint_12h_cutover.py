from __future__ import annotations

import gc
import hashlib
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import (
    build_comparison_dataset,
    list_comparison_scenarios_detailed,
)
from src.vde_core.cycles import use_standard_cycle
from src.vde_core.fuel_estimation import FuelEstimateRequest, save_fuel_estimate_result
from src.vde_core.quick_scenario import QuickScenario, VehicleQuickOverrides, resolve_quick_vehicle_scenario
from src.vde_core.repositories import (
    delete_fuelcons_by_id,
    delete_vde_by_id,
    fetch_fuelcons_by_vde_id,
    fetch_vde_by_id,
)
from src.vde_core.vde_workflow_service import (
    build_vde_pre_save_review,
    build_vde_setup_preview_from_ctx,
    save_vde_setup_result,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE = (
    ROOT
    / "etl"
    / "data"
    / "staging"
    / "sprint_12f13_vde_materialized"
    / "eco_drive_canonical_vde_materialized_candidate.db"
)
RUNTIME = ROOT / "data" / "db" / "eco_drive.db"
QA_RUNTIME = ROOT / "data" / "db" / "eco_drive_qa.db"
LEGACY_BACKUP = ROOT / "data" / "backups" / "eco_drive_pre_sprint12_20260915.db"
CANONICAL_SHA256 = "243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB"
LEGACY_SHA256 = "CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262"
QA_SHA256 = CANONICAL_SHA256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class Sprint12HCutoverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_path = db_module.current_db_path()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.disposable = Path(cls.temp_dir.name) / "canonical_runtime_disposable.db"
        shutil.copy2(CANONICAL_SOURCE, cls.disposable)
        db_module.configure_db_path(cls.disposable)
        with sqlite3.connect(cls.disposable) as connection:
            connection.row_factory = sqlite3.Row
            cls.epa = dict(connection.execute(
                "SELECT v.* , f.id AS fuelcons_id FROM vde v JOIN fuelcons f ON f.vde_id=v.id "
                "WHERE v.legislation='EPA' AND v.mass_kg IS NOT NULL "
                "AND v.test_mass_kg IS NOT NULL AND v.coast_A_N IS NOT NULL "
                "AND v.coast_B_N_per_kph IS NOT NULL AND v.coast_C_N_per_kph2 IS NOT NULL "
                "AND v.vde_total_mj_per_km IS NOT NULL ORDER BY v.id LIMIT 1"
            ).fetchone())
            cls.wltp = dict(connection.execute(
                "SELECT v.id AS vde_id,f.id AS fuelcons_id FROM vde v JOIN fuelcons f ON f.vde_id=v.id "
                "WHERE v.legislation='WLTP' AND v.vde_low_mj_per_km IS NOT NULL ORDER BY v.id LIMIT 1"
            ).fetchone())
            cls.multi_vde = dict(connection.execute(
                "SELECT vehicle_configuration_id,MIN(id) AS first_id,MAX(id) AS second_id,COUNT(*) AS n "
                "FROM vde GROUP BY vehicle_configuration_id HAVING COUNT(*)>1 ORDER BY vehicle_configuration_id LIMIT 1"
            ).fetchone())
            cls.multi_fuelcons = dict(connection.execute(
                "SELECT vde_id,MIN(id) AS first_id,MAX(id) AS second_id,COUNT(*) AS n "
                "FROM fuelcons GROUP BY vde_id HAVING COUNT(*)>1 ORDER BY vde_id LIMIT 1"
            ).fetchone())
            cls.no_fuelcons = int(connection.execute(
                "SELECT v.id FROM vde v LEFT JOIN fuelcons f ON f.vde_id=v.id "
                "WHERE f.id IS NULL ORDER BY v.id LIMIT 1"
            ).fetchone()[0])

    @classmethod
    def tearDownClass(cls) -> None:
        db_module.configure_db_path(cls.original_path)
        gc.collect()
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        db_module.configure_db_path(self.disposable)

    def test_01_default_runtime_is_canonical_without_override(self) -> None:
        db_module.configure_db_path(db_module.DEFAULT_DB_PATH)
        self.assertEqual(db_module.current_db_path(), db_module.DEFAULT_DB_PATH)
        self.assertEqual(db_module.VDE_WRITE_TABLE, "vde")
        self.assertEqual(db_module.FUELCONS_WRITE_TABLE, "fuelcons")
        self.assertFalse(db_module.LEGACY_FIXTURE_MODE)
        self.assertEqual(_sha256(RUNTIME), CANONICAL_SHA256)
        rows = list_comparison_scenarios_detailed({"make": self.epa["make"]})
        self.assertIn(self.epa["fuelcons_id"], {row["fuelcons_id"] for row in rows})
        self.assertEqual(fetch_vde_by_id(self.epa["id"])["id"], self.epa["id"])
        self.assertIn(
            self.epa["fuelcons_id"],
            {row["id"] for row in fetch_fuelcons_by_vde_id(self.epa["id"])},
        )

    def test_02_final_runtime_population_and_views(self) -> None:
        with sqlite3.connect(RUNTIME) as connection:
            counts = {
                table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                for table in ("program", "vehicle_configuration", "vde", "run", "fuelcons", "fuelcons_run_adoption")
            }
            self.assertEqual(counts, {
                "program": 3117,
                "vehicle_configuration": 10211,
                "vde": 11626,
                "run": 29250,
                "fuelcons": 10822,
                "fuelcons_run_adoption": 18323,
            })
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0], 11626)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM fuelcons_db").fetchone()[0], 10822)

    def test_03_default_startup_does_not_run_legacy_bootstrap(self) -> None:
        db_module.configure_db_path(db_module.DEFAULT_DB_PATH)
        before = _sha256(RUNTIME)
        db_module.ensure_db()
        self.assertEqual(_sha256(RUNTIME), before)
        self.assertFalse(db_module.LEGACY_FIXTURE_MODE)

    def test_04_real_vde_setup_preview_save_and_read_back(self) -> None:
        baseline = dict(self.epa)
        cycle = use_standard_cycle("EPA")
        self.assertIsNotNone(cycle)
        ctx = {
            **baseline,
            "mode": "From baseline (editable)",
            "from_delta": "Deltas",
            "abc_total_source_ui": "Baseline ABC_TOTAL",
            "baseline_id": baseline["id"],
            "vde_id_parent": baseline["id"],
            "A": baseline["coast_A_N"],
            "B": baseline["coast_B_N_per_kph"],
            "C": baseline["coast_C_N_per_kph2"],
            "delta_rr_N": 1.25,
            "crr1_frac_at_120kph": 0.0,
            "cycle_df": cycle,
            "notes": "Sprint 12H disposable full VDE Setup save",
        }
        preview = build_vde_setup_preview_from_ctx(ctx)
        review = build_vde_pre_save_review(ctx, preview, preview["save_payload"])
        self.assertTrue(preview["ok"])
        self.assertGreater(len(review["staged_save_payload"]["insert_row"]), 20)
        self.assertAlmostEqual(preview["abc_total"]["A"], baseline["coast_A_N"] + 1.25)
        self.assertIsNotNone(preview["vde_total"])
        result = save_vde_setup_result(preview, "insert_new", ctx=ctx)
        new_id = result["vde_id"]
        try:
            with sqlite3.connect(self.disposable) as connection:
                connection.row_factory = sqlite3.Row
                physical = dict(connection.execute("SELECT * FROM vde WHERE id=?", (new_id,)).fetchone())
                projected = dict(connection.execute("SELECT * FROM vde_db WHERE id=?", (new_id,)).fetchone())
            self.assertEqual(physical["vehicle_configuration_id"], baseline["vehicle_configuration_id"])
            self.assertEqual(physical["vde_id_parent"], baseline["id"])
            self.assertAlmostEqual(physical["mass_kg"], result["row"]["mass_kg"])
            self.assertAlmostEqual(physical["coast_A_N"], preview["abc_total"]["A"])
            self.assertAlmostEqual(physical["coast_B_N_per_kph"], preview["abc_total"]["B"])
            self.assertAlmostEqual(physical["coast_C_N_per_kph2"], preview["abc_total"]["C"])
            self.assertAlmostEqual(physical["vde_total_mj_per_km"], preview["vde_total"]["mj_per_km"])
            if preview["vde_net"] is not None:
                self.assertAlmostEqual(physical["vde_net_mj_per_km"], preview["vde_net"]["mj_per_km"])
            self.assertEqual(projected["id"], new_id)
            self.assertEqual(fetch_vde_by_id(new_id)["id"], new_id)
            with sqlite3.connect(self.disposable) as connection:
                connection.execute("PRAGMA foreign_keys=ON")
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        finally:
            delete_vde_by_id(new_id)

    def test_05_fuelcons_write_physical_read_compatibility_view(self) -> None:
        request = FuelEstimateRequest(
            vde_id=self.epa["id"],
            energy_basis="MANUAL_VALUE",
            method="manual_imported",
            vehicle_features={"electrification": "ICE"},
            powertrain_features={"fuel_type": "Gasoline", "eta_pt_est": 0.30},
            manual_inputs={"source": "Sprint 12H disposable", "fuel_l_100km": 7.35},
        )
        saved = save_fuel_estimate_result(request, "insert_new")
        try:
            with sqlite3.connect(self.disposable) as connection:
                physical = connection.execute(
                    "SELECT vde_id,energy_basis,fuel_l_per_100km FROM fuelcons WHERE id=?",
                    (saved["row_id"],),
                ).fetchone()
            projected = next(
                row for row in fetch_fuelcons_by_vde_id(self.epa["id"])
                if row["id"] == saved["row_id"]
            )
            self.assertEqual(physical[0], self.epa["id"])
            self.assertEqual(physical[1], "SOURCE_DECLARED")
            self.assertAlmostEqual(physical[2], 7.35)
            self.assertAlmostEqual(projected["fuel_l_per_100km"], 7.35)
        finally:
            delete_fuelcons_by_id(saved["row_id"])

    def test_06_core_application_flows_use_canonical_copy(self) -> None:
        rows = list_comparison_scenarios_detailed({"make": self.epa["make"]})
        self.assertIn(self.epa["fuelcons_id"], {row["fuelcons_id"] for row in rows})
        dataset = build_comparison_dataset(
            {"kind": "FUELCONS_SCENARIO", "fuelcons_id": self.epa["fuelcons_id"]},
            [{"kind": "VDE_ONLY", "vde_id": self.wltp["vde_id"]}],
        )
        self.assertEqual(dataset.comparisons[0].vde_id, self.wltp["vde_id"])
        scenario = QuickScenario(
            source_identity=f"vde:{self.epa['id']}", slot=1, vehicle_overrides=VehicleQuickOverrides()
        )
        self.assertTrue(resolve_quick_vehicle_scenario(
            scenario, source_vde_row=fetch_vde_by_id(self.epa["id"])
        ).is_ready)

    def test_07_representative_relationship_cases_exist(self) -> None:
        self.assertGreater(self.multi_vde["n"], 1)
        self.assertNotEqual(self.multi_vde["first_id"], self.multi_vde["second_id"])
        self.assertGreater(self.multi_fuelcons["n"], 1)
        self.assertEqual(fetch_fuelcons_by_vde_id(self.no_fuelcons), [])

    def test_08_final_integrity_and_adoption_invariant(self) -> None:
        with sqlite3.connect(RUNTIME) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")
            failures = connection.execute(
                "SELECT COUNT(*) FROM fuelcons_run_adoption a "
                "JOIN fuelcons f ON f.id=a.fuelcons_id JOIN run r ON r.run_id=a.run_id "
                "WHERE a.vde_id<>f.vde_id OR a.vde_id<>r.vde_id"
            ).fetchone()[0]
            self.assertEqual(failures, 0)
            primary_keys = {
                "program": "program_id",
                "vehicle_configuration": "vehicle_configuration_id",
                "vde": "id",
                "run": "run_id",
                "fuelcons": "id",
            }
            for table, primary_key in primary_keys.items():
                total, distinct_ids = connection.execute(
                    f'SELECT COUNT(*),COUNT(DISTINCT "{primary_key}") FROM "{table}"'
                ).fetchone()
                self.assertEqual(total, distinct_ids)
            duplicate_adoptions = connection.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT fuelcons_id,run_id,result_dimension,COUNT(*) AS n "
                "FROM fuelcons_run_adoption GROUP BY fuelcons_id,run_id,result_dimension HAVING n>1)"
            ).fetchone()[0]
            self.assertEqual(duplicate_adoptions, 0)

    def test_09_canonical_runtime_blocks_legacy_truncate(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            db_module.truncate_db(RUNTIME)
        self.assertEqual(_sha256(RUNTIME), CANONICAL_SHA256)

    def test_10_legacy_backup_and_rollback_rehearsal(self) -> None:
        self.assertTrue(LEGACY_BACKUP.exists())
        self.assertEqual(_sha256(LEGACY_BACKUP), LEGACY_SHA256)
        rollback_copy = Path(self.temp_dir.name) / "rollback_rehearsal.db"
        shutil.copy2(LEGACY_BACKUP, rollback_copy)
        with db_module.using_db_path(rollback_copy):
            db_module.ensure_db()
            self.assertEqual(db_module.VDE_WRITE_TABLE, "vde_db")
            with sqlite3.connect(rollback_copy) as connection:
                self.assertGreater(connection.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0], 0)

    def test_11_canonical_source_is_immutable(self) -> None:
        self.assertEqual(_sha256(CANONICAL_SOURCE), CANONICAL_SHA256)

    def test_12_runtime_and_qa_safety_hashes(self) -> None:
        self.assertEqual(_sha256(RUNTIME), CANONICAL_SHA256)
        self.assertEqual(_sha256(QA_RUNTIME), QA_SHA256)
        self.assertEqual(_sha256(LEGACY_BACKUP), LEGACY_SHA256)


if __name__ == "__main__":
    unittest.main()
