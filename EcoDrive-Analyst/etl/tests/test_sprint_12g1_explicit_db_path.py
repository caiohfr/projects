from __future__ import annotations

import gc
import hashlib
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vde_app.components import pwt_system_scenario
from src.vde_app.powertrain_system_scenario_viewmodels import current_draft
from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import (
    build_comparison_dataset,
    list_comparison_scenarios_detailed,
)
from src.vde_core.fuel_estimation import FuelEstimateRequest, save_fuel_estimate_result
from src.vde_core.quick_scenario import (
    QuickScenario,
    VehicleQuickOverrides,
    resolve_quick_vehicle_scenario,
)
from src.vde_core.repositories import (
    delete_fuelcons_by_id,
    delete_vde_by_id,
    fetch_fuelcons_by_vde_id,
    fetch_vde_by_id,
    insert_vde_row,
)
from src.vde_core.repositories import fuelcons_repository
from src.vde_core.system_scenario import ArchitectureClass
from src.vde_core.vde_setup_service import fetch_vde_edit_rows
from src.vde_core.vde_workflow_service import save_vde_setup_result


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE = (
    ROOT
    / "etl"
    / "data"
    / "staging"
    / "sprint_12f13_vde_materialized"
    / "eco_drive_canonical_vde_materialized_candidate.db"
)
RUNTIME_DBS = (
    ROOT / "data" / "db" / "eco_drive.db",
    ROOT / "data" / "db" / "eco_drive_qa.db",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class Sprint12G1ExplicitDatabasePathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_path = db_module.current_db_path()
        cls.source_hash = _sha256(CANONICAL_SOURCE)
        cls.runtime_hashes = {str(path): _sha256(path) for path in RUNTIME_DBS}
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.integration_db = Path(cls.temp_dir.name) / "canonical_explicit.db"
        shutil.copy2(CANONICAL_SOURCE, cls.integration_db)
        db_module.configure_db_path(cls.integration_db)
        with sqlite3.connect(cls.integration_db) as connection:
            connection.row_factory = sqlite3.Row
            cls.epa = dict(connection.execute(
                "SELECT v.id AS vde_id,f.id AS fuelcons_id,v.make,v.vehicle_configuration_id "
                "FROM vde v JOIN fuelcons f ON f.vde_id=v.id "
                "WHERE v.legislation='EPA' AND v.vde_total_mj_per_km IS NOT NULL "
                "ORDER BY v.id LIMIT 1"
            ).fetchone())
            cls.wltp = dict(connection.execute(
                "SELECT v.id AS vde_id,f.id AS fuelcons_id FROM vde v "
                "JOIN fuelcons f ON f.vde_id=v.id "
                "WHERE v.legislation='WLTP' AND v.vde_low_mj_per_km IS NOT NULL "
                "ORDER BY v.id LIMIT 1"
            ).fetchone())

    @classmethod
    def tearDownClass(cls) -> None:
        db_module.configure_db_path(cls.original_path)
        gc.collect()
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        db_module.configure_db_path(self.integration_db)

    def _clone_payload(self, parent_id: int, marker: str) -> dict:
        parent = fetch_vde_by_id(parent_id)
        fields = (
            "legislation", "category", "make", "model", "year", "mass_kg",
            "test_mass_kg", "test_mass_low_kg", "test_mass_high_kg",
            "test_mass_basis", "inertia_class", "cycle_name", "cycle_source",
            "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2",
            "vde_total_mj_per_km", "vde_net_mj_per_km", "record_origin",
        )
        row = {field: parent.get(field) for field in fields if parent.get(field) is not None}
        row.update({"vde_id_parent": int(parent_id), "notes": marker})
        return row

    def test_01_explicit_canonical_db_path_can_be_supplied(self) -> None:
        self.assertEqual(db_module.current_db_path(), self.integration_db)
        self.assertEqual(db_module.VDE_WRITE_TABLE, "vde")
        self.assertEqual(db_module.FUELCONS_WRITE_TABLE, "fuelcons")
        self.assertFalse(db_module.LEGACY_FIXTURE_MODE)
        db_module.configure_db_path(str(self.integration_db))
        self.assertEqual(db_module.VDE_WRITE_TABLE, "vde")
        self.assertFalse(db_module.LEGACY_FIXTURE_MODE)

    def test_02_application_reads_vde_db_from_selected_file(self) -> None:
        row = fetch_vde_by_id(self.epa["vde_id"])
        self.assertEqual(row["id"], self.epa["vde_id"])
        with sqlite3.connect(self.integration_db) as connection:
            kind = connection.execute(
                "SELECT type FROM sqlite_master WHERE name='vde_db'"
            ).fetchone()[0]
        self.assertEqual(kind, "view")

    def test_03_application_reads_fuelcons_db_from_selected_file(self) -> None:
        rows = fetch_fuelcons_by_vde_id(self.epa["vde_id"])
        self.assertIn(self.epa["fuelcons_id"], {row["id"] for row in rows})
        with sqlite3.connect(self.integration_db) as connection:
            kind = connection.execute(
                "SELECT type FROM sqlite_master WHERE name='fuelcons_db'"
            ).fetchone()[0]
        self.assertEqual(kind, "view")

    def test_04_canonical_vde_write_targets_physical_vde(self) -> None:
        new_id = insert_vde_row(self._clone_payload(self.epa["vde_id"], "12G.1 direct VDE"))
        try:
            with sqlite3.connect(self.integration_db) as connection:
                physical = connection.execute("SELECT id FROM vde WHERE id=?", (new_id,)).fetchone()
                projected = connection.execute("SELECT id FROM vde_db WHERE id=?", (new_id,)).fetchone()
            self.assertEqual(physical, (new_id,))
            self.assertEqual(projected, (new_id,))
        finally:
            delete_vde_by_id(new_id)

    def test_05_canonical_fuelcons_write_targets_physical_fuelcons(self) -> None:
        request = FuelEstimateRequest(
            vde_id=self.epa["vde_id"],
            energy_basis="MANUAL_VALUE",
            method="manual_imported",
            vehicle_features={"electrification": "ICE"},
            powertrain_features={"fuel_type": "Gasoline", "eta_pt_est": 0.30},
            manual_inputs={"source": "12G.1 disposable", "fuel_l_100km": 7.4},
        )
        result = save_fuel_estimate_result(request, "insert_new")
        try:
            with sqlite3.connect(self.integration_db) as connection:
                physical = connection.execute(
                    "SELECT energy_basis FROM fuelcons WHERE id=?", (result["row_id"],)
                ).fetchone()
            self.assertEqual(physical, ("SOURCE_DECLARED",))
        finally:
            delete_fuelcons_by_id(result["row_id"])

    def test_06_no_schema_autodetection_helper_is_required(self) -> None:
        source = (ROOT / "src" / "vde_core" / "db.py").read_text(encoding="utf-8")
        self.assertNotIn("_is_canonical_connection", source)
        self.assertNotIn("_sqlite_object_type", source)
        self.assertNotIn("is_canonical_database", source)

    def test_07_no_automatic_legacy_to_canonical_router_remains(self) -> None:
        source = (ROOT / "src" / "vde_core" / "db.py").read_text(encoding="utf-8")
        self.assertNotIn("canonical_write_target", source)
        self.assertNotIn("_CANONICAL_WRITE_TARGETS", source)
        self.assertEqual((db_module.VDE_WRITE_TABLE, db_module.FUELCONS_WRITE_TABLE), ("vde", "fuelcons"))
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            db_module.truncate_db(self.integration_db)

    def test_08_vde_setup_write_read_back_works_on_disposable_copy(self) -> None:
        preview = {
            "ok": True,
            "save_payload": {"insert_row": self._clone_payload(self.epa["vde_id"], "12G.1 setup")},
            "phase_update_row": {},
        }
        result = save_vde_setup_result(preview, "insert_new")
        new_id = result["vde_id"]
        try:
            row = fetch_vde_by_id(new_id)
            self.assertEqual(row["vde_id_parent"], self.epa["vde_id"])
            with sqlite3.connect(self.integration_db) as connection:
                owner = connection.execute(
                    "SELECT vehicle_configuration_id FROM vde WHERE id=?", (new_id,)
                ).fetchone()[0]
            self.assertEqual(owner, self.epa["vehicle_configuration_id"])
        finally:
            delete_vde_by_id(new_id)

    def test_09_quick_scenario_uses_explicit_canonical_baseline(self) -> None:
        source = fetch_vde_by_id(self.epa["vde_id"])
        scenario = QuickScenario(
            source_identity=f"vde:{self.epa['vde_id']}",
            slot=1,
            vehicle_overrides=VehicleQuickOverrides(),
        )
        result = resolve_quick_vehicle_scenario(scenario, source_vde_row=source)
        self.assertTrue(result.is_ready, result.issues)

    def test_10_comparison_uses_explicit_canonical_database(self) -> None:
        dataset = build_comparison_dataset(
            {"kind": "FUELCONS_SCENARIO", "fuelcons_id": self.epa["fuelcons_id"]},
            [{"kind": "VDE_ONLY", "vde_id": self.wltp["vde_id"]}],
        )
        self.assertEqual(dataset.reference.vde_id, self.epa["vde_id"])
        self.assertEqual(dataset.comparisons[0].vde_id, self.wltp["vde_id"])

    def test_11_browse_uses_explicit_canonical_database(self) -> None:
        rows = list_comparison_scenarios_detailed({"make": self.epa["make"]})
        self.assertTrue(rows)
        self.assertIn(self.epa["fuelcons_id"], {row["fuelcons_id"] for row in rows})

    def test_12_powertrain_resolves_from_explicit_canonical_database(self) -> None:
        drafts = (current_draft(
            self.epa["vde_id"], ArchitectureClass.ICE, fuelcons_id=self.epa["fuelcons_id"]
        ),)
        sources, _ = pwt_system_scenario._load_sources(
            self.epa["vde_id"], self.epa["fuelcons_id"], drafts=drafts
        )
        self.assertEqual(sources[self.epa["vde_id"]].fuelcons_row["id"], self.epa["fuelcons_id"])

    def test_13_materialized_fuelcons_read_does_not_traverse_run(self) -> None:
        observed: list[str] = []
        original = fuelcons_repository.fetchall

        def recording(sql, params=()):
            observed.append(str(sql))
            return original(sql, params)

        with patch.object(fuelcons_repository, "fetchall", side_effect=recording):
            rows = fuelcons_repository.fetch_fuelcons_by_vde_id(self.epa["vde_id"])
        self.assertTrue(rows)
        self.assertTrue(all("fuelcons_run_adoption" not in sql.lower() for sql in observed))
        self.assertTrue(all(" run " not in f" {sql.lower()} " for sql in observed))

    def test_14_legacy_default_database_behavior_is_unchanged(self) -> None:
        legacy = Path(self.temp_dir.name) / "legacy_default.db"
        with db_module.using_db_path(legacy):
            db_module.ensure_db()
            row_id = db_module.insert_vde({
                "legislation": "EPA", "category": "Passenger Car", "make": "12G1",
                "model": "legacy", "mass_kg": 1500.0,
            })
            self.assertEqual(db_module.VDE_WRITE_TABLE, "vde_db")
            self.assertEqual(fetch_vde_by_id(row_id)["make"], "12G1")
        self.assertEqual(db_module.current_db_path(), self.integration_db)
        self.assertEqual(db_module.VDE_WRITE_TABLE, "vde")

    def test_15_canonical_source_candidate_hash_is_unchanged(self) -> None:
        self.assertEqual(_sha256(CANONICAL_SOURCE), self.source_hash)

    def test_16_runtime_database_hashes_are_unchanged(self) -> None:
        self.assertEqual({str(path): _sha256(path) for path in RUNTIME_DBS}, self.runtime_hashes)

    def test_17_foreign_key_check_passes_after_disposable_writes(self) -> None:
        with sqlite3.connect(self.integration_db) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_18_sqlite_quick_check_is_ok(self) -> None:
        with sqlite3.connect(self.integration_db) as connection:
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")


if __name__ == "__main__":
    unittest.main()
