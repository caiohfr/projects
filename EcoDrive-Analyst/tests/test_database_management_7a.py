from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.data_change_log_repository import append_change_log, fetch_change_log
from src.vde_core.database_management_contract import (
    ChangeAction,
    ChangeCommand,
    EntityType,
    LOCAL_ADMIN_ACTOR,
)
from src.vde_core.database_management_policy import FieldAccess, field_access_for
from src.vde_core.database_management_service import local_actor_context, preview_change
from src.vde_core.qa_mock_data import build_tire_seed_rows, build_vde_seed_rows


class DatabaseManagement7ATests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "legacy.db"

    def tearDown(self):
        # Existing DB helpers use sqlite connections as context managers; on
        # Windows their handles are finalized by GC rather than closed by the
        # context manager itself.
        gc.collect()
        self._temp_dir.cleanup()

    def test_actor_context_uses_local_admin_contract(self):
        self.assertEqual(local_actor_context(), LOCAL_ADMIN_ACTOR)
        self.assertEqual(LOCAL_ADMIN_ACTOR.actor_id, "local_admin")
        self.assertEqual(LOCAL_ADMIN_ACTOR.actor_role, "admin")

    def test_migration_preserves_legacy_rows_and_is_idempotent(self):
        self._create_legacy_database()

        with db_module.using_db_path(self.db_path):
            db_module.ensure_db()
            db_module.ensure_migrations()
            db_module.ensure_migrations()

        with sqlite3.connect(self.db_path) as con:
            vde = con.execute(
                "SELECT mass_kg, coast_A_N, record_origin, record_status FROM vde_db WHERE id=1"
            ).fetchone()
            fuel = con.execute(
                "SELECT vde_id, record_origin, record_status, review_status FROM fuelcons_db WHERE id=1"
            ).fetchone()
            tire = con.execute(
                "SELECT rr_n_per_kn, is_active, record_origin FROM tire_roadload_db WHERE id=1"
            ).fetchone()
            tire_columns = {row[1] for row in con.execute("PRAGMA table_info(tire_roadload_db)")}
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        self.assertEqual(vde, (1500.0, 118.0, "LEGACY", "ACTIVE"))
        self.assertEqual(fuel, (1, "LEGACY", "ACTIVE", "CURRENT"))
        self.assertEqual(tire, (8.0, 1, "LEGACY"))
        self.assertNotIn("record_status", tire_columns)
        self.assertIn("data_change_log", tables)

    def test_migration_adds_expected_management_columns(self):
        self._create_legacy_database()
        with db_module.using_db_path(self.db_path):
            db_module.ensure_db()

        with sqlite3.connect(self.db_path) as con:
            vde_columns = {row[1] for row in con.execute("PRAGMA table_info(vde_db)")}
            fuel_columns = {row[1] for row in con.execute("PRAGMA table_info(fuelcons_db)")}
            tire_columns = {row[1] for row in con.execute("PRAGMA table_info(tire_roadload_db)")}
            log_columns = {row[1] for row in con.execute("PRAGMA table_info(data_change_log)")}

        self.assertTrue({"record_origin", "record_status", "source_name", "source_record_id"} <= vde_columns)
        self.assertTrue(
            {"updated_at", "record_origin", "record_status", "source_name", "source_record_id", "review_status"}
            <= fuel_columns
        )
        self.assertTrue({"record_origin", "source_name", "source_record_id"} <= tire_columns)
        self.assertTrue(
            {
                "operation_id",
                "actor_id",
                "actor_role",
                "entity_type",
                "record_id",
                "action",
                "reason",
                "before_json",
                "after_json",
                "impact_json",
            }
            <= log_columns
        )

    def test_qa_database_paths_remain_isolated(self):
        first = Path(self._temp_dir.name) / "qa_a.db"
        second = Path(self._temp_dir.name) / "qa_b.db"
        original = db_module.current_db_path()

        with db_module.using_db_path(first):
            db_module.ensure_db()
            with db_module._con() as con:
                con.execute(
                    "INSERT INTO vde_db (legislation, category, make, model, mass_kg) VALUES ('EPA','QA','A','ONE',1500)"
                )
        with db_module.using_db_path(second):
            db_module.ensure_db()
            with db_module._con() as con:
                count = con.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0]

        self.assertEqual(count, 0)
        self.assertEqual(db_module.current_db_path(), original)

    def test_qa_seed_rows_declare_non_legacy_origins(self):
        vde_rows = build_vde_seed_rows()
        tire_rows = build_tire_seed_rows()

        self.assertTrue(vde_rows)
        self.assertTrue(tire_rows)
        self.assertEqual({row["record_origin"] for row in vde_rows}, {"IMPORTED_REFERENCE"})
        self.assertEqual({row["record_origin"] for row in tire_rows}, {"QA"})
        self.assertTrue(all(row["source_name"] == "qa_mock_seed" for row in [*vde_rows, *tire_rows]))

    def test_vde_setup_policy_blocks_direct_physics_and_derived_results(self):
        self.assertEqual(field_access_for(EntityType.VDE, "VDE_SETUP", "make"), FieldAccess.EDITABLE)
        self.assertEqual(field_access_for(EntityType.VDE, "VDE_SETUP", "mass_kg"), FieldAccess.DERIVED)
        self.assertEqual(field_access_for(EntityType.VDE, "VDE_SETUP", "vde_total_mj_per_km"), FieldAccess.DERIVED)

        preview = preview_change(
            ChangeCommand(
                entity_type="VDE",
                action="UPDATE",
                record_id=7,
                record_origin="VDE_SETUP",
                reason="Correct model label",
                current_record={"id": 7, "make": "QA", "mass_kg": 1500.0},
                payload={"make": "AUDI", "mass_kg": 1600.0},
            )
        )

        self.assertFalse(preview.can_commit)
        self.assertEqual(preview.normalized_payload, {"make": "AUDI"})
        self.assertIn("field_derived", {issue.code for issue in preview.validation_issues})

    def test_imported_vde_physical_correction_requires_reason_and_warns(self):
        missing_reason = preview_change(
            ChangeCommand(
                entity_type="VDE",
                action=ChangeAction.UPDATE,
                record_id=8,
                record_origin="IMPORTED_REFERENCE",
                current_record={"id": 8, "mass_kg": 1500.0},
                payload={"mass_kg": 0.0},
            )
        )
        confirmed = preview_change(
            ChangeCommand(
                entity_type="VDE",
                action=ChangeAction.UPDATE,
                record_id=8,
                record_origin="IMPORTED_REFERENCE",
                reason="Correct source record",
                current_record={"id": 8, "mass_kg": 1500.0},
                payload={"mass_kg": 0.0},
            )
        )

        self.assertFalse(missing_reason.can_commit)
        self.assertIn("reason_required", {issue.code for issue in missing_reason.validation_issues})
        self.assertTrue(confirmed.can_commit)
        self.assertEqual(confirmed.normalized_payload["mass_kg"], 0.0)
        self.assertEqual(confirmed.field_diff[0].after, 0.0)
        self.assertIn("advanced_correction", {issue.code for issue in confirmed.validation_issues})

    def test_estimated_fuel_outputs_are_read_only_but_measured_values_are_correctable(self):
        estimated = preview_change(
            ChangeCommand(
                entity_type="FUEL_CONSUMPTION",
                action="UPDATE",
                record_id=10,
                record_origin="ESTIMATED",
                reason="Attempt direct output edit",
                current_record={"id": 10, "fuel_l_per_100km": 8.0},
                payload={"fuel_l_per_100km": 7.5},
            )
        )
        measured = preview_change(
            ChangeCommand(
                entity_type="FUEL_CONSUMPTION",
                action="UPDATE",
                record_id=11,
                record_origin="MEASURED",
                reason="Correct laboratory transcription",
                current_record={"id": 11, "fuel_l_per_100km": 8.0},
                payload={"fuel_l_per_100km": 7.5},
            )
        )

        self.assertFalse(estimated.can_commit)
        self.assertIn("field_derived", {issue.code for issue in estimated.validation_issues})
        self.assertTrue(measured.can_commit)
        self.assertIn("advanced_correction", {issue.code for issue in measured.validation_issues})

    def test_change_log_records_actor_and_json_receipt(self):
        self._create_legacy_database()
        with db_module.using_db_path(self.db_path):
            db_module.ensure_db()
            preview = preview_change(
                ChangeCommand(
                    entity_type="VDE",
                    action="UPDATE",
                    record_id=1,
                    record_origin="LEGACY",
                    reason="Correct make",
                    current_record={"id": 1, "make": "QA"},
                    payload={"make": "AUDI"},
                )
            )
            log_id = append_change_log(
                preview,
                LOCAL_ADMIN_ACTOR,
                reason="Correct make",
                before={"make": "QA"},
                after={"make": "AUDI"},
                impact={"dependencies": 0},
            )
            receipt = fetch_change_log(preview.operation_id)

        self.assertGreater(log_id, 0)
        self.assertEqual(receipt["actor_id"], "local_admin")
        self.assertEqual(receipt["actor_role"], "admin")
        self.assertEqual(receipt["before_json"], {"make": "QA"})
        self.assertEqual(receipt["after_json"], {"make": "AUDI"})
        self.assertEqual(receipt["impact_json"], {"dependencies": 0})

    def test_unknown_and_immutable_fields_are_rejected(self):
        preview = preview_change(
            ChangeCommand(
                entity_type="TIRE",
                action="UPDATE",
                record_id=3,
                record_origin="QA",
                reason="Bad payload",
                current_record={"id": 3},
                payload={"id": 4, "mystery_column": 1},
            )
        )
        codes = {issue.code for issue in preview.validation_issues}
        self.assertFalse(preview.can_commit)
        self.assertEqual(preview.normalized_payload, {})
        self.assertIn("field_immutable", codes)
        self.assertIn("field_unknown", codes)

    def test_create_requires_explicit_origin_and_tire_uses_is_active_status(self):
        missing_origin = preview_change(
            ChangeCommand(
                entity_type="TIRE",
                action="CREATE",
                payload={"tire_test_code": "NEW-TIRE"},
            )
        )
        valid = preview_change(
            ChangeCommand(
                entity_type="TIRE",
                action="CREATE",
                record_origin="IMPORTED",
                payload={"tire_test_code": "NEW-TIRE"},
            )
        )

        self.assertFalse(missing_origin.can_commit)
        self.assertIn("record_origin_required", {issue.code for issue in missing_origin.validation_issues})
        self.assertTrue(valid.can_commit)
        self.assertEqual(valid.normalized_payload["record_origin"], "IMPORTED")
        self.assertEqual(valid.normalized_payload["is_active"], 1)
        self.assertNotIn("record_status", valid.normalized_payload)

    def _create_legacy_database(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executescript(
                """
                PRAGMA foreign_keys=ON;
                CREATE TABLE vde_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT,
                    legislation TEXT NOT NULL,
                    category TEXT NOT NULL,
                    make TEXT NOT NULL,
                    model TEXT NOT NULL,
                    year INTEGER,
                    mass_kg REAL NOT NULL,
                    coast_A_N REAL
                );
                CREATE TABLE fuelcons_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    vde_id INTEGER NOT NULL REFERENCES vde_db(id) ON DELETE CASCADE,
                    electrification TEXT NOT NULL
                );
                CREATE TABLE tire_roadload_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT,
                    tire_test_code TEXT NOT NULL UNIQUE,
                    manufacturer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    size_code TEXT,
                    is_active INTEGER DEFAULT 1,
                    standard_family TEXT NOT NULL,
                    rr_n_per_kn REAL NOT NULL
                );
                INSERT INTO vde_db (
                    id, legislation, category, make, model, year, mass_kg, coast_A_N
                ) VALUES (1, 'EPA', 'QA', 'QA', 'LEGACY', 2026, 1500, 118);
                INSERT INTO fuelcons_db (id, vde_id, electrification) VALUES (1, 1, 'ICE');
                INSERT INTO tire_roadload_db (
                    id, tire_test_code, manufacturer, model, size_code, is_active, standard_family, rr_n_per_kn
                ) VALUES (1, 'LEGACY-TIRE', 'QA', 'TIRE', '235/55R19', 1, 'ISO', 8);
                """
            )


if __name__ == "__main__":
    unittest.main()
