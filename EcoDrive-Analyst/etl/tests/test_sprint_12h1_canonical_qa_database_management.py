from __future__ import annotations

import hashlib
import os
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.vde_core import db as db_module
from src.vde_core.database_management_contract import ChangeCommand, EntityType, normalize_record_origin
from src.vde_core.database_management_policy import FieldAccess, field_access_for, is_record_origin_protected
from src.vde_core.database_management_service import browse_records, preview_change


ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "data" / "db" / "eco_drive.db"
QA = ROOT / "data" / "db" / "eco_drive_qa.db"
STAGING = ROOT / "data" / "db" / "staging" / "eco_drive_canonical_candidate.db"
LEGACY_PROD = ROOT / "data" / "db" / "archive" / "eco_drive_legacy_pre_sprint12.db"
LEGACY_QA = ROOT / "data" / "db" / "archive" / "eco_drive_qa_legacy_pre_sprint12.db"
PAGE = ROOT / "pages" / "Database_Management.py"
RUNTIME_CANONICAL_SHA256 = "243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB"
STAGING_ID_NORMALIZED_SHA256 = "1E3B251F39B21A835EF9F0B4F6291A495636A8C425CEF7051754D1D750840ACF"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


class Sprint12H1CanonicalQaDatabaseManagementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_path = db_module.current_db_path()
        cls.hashes_before = {path: _sha256(path) for path in (PROD, QA, STAGING)}
        db_module.configure_db_path(QA)

    @classmethod
    def tearDownClass(cls) -> None:
        db_module.configure_db_path(cls.original_path)
        for path, digest in cls.hashes_before.items():
            if _sha256(path) != digest:
                raise AssertionError(f"Canonical DB changed during focused tests: {path}")

    def setUp(self) -> None:
        db_module.configure_db_path(QA)

    def test_01_staging_is_id_normalized_while_qa_prod_remain_unpromoted(self) -> None:
        expected = {
            "program": 3117,
            "vehicle_configuration": 10211,
            "vde": 11626,
            "run": 29250,
            "fuelcons": 10822,
            "fuelcons_run_adoption": 18323,
            "vde_component_resolution": 0,
        }
        self.assertEqual(_sha256(STAGING), STAGING_ID_NORMALIZED_SHA256)
        for path in (QA, PROD):
            self.assertEqual(_sha256(path), RUNTIME_CANONICAL_SHA256)
        for path in (STAGING, QA, PROD):
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
                actual = {
                    table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                    for table in expected
                }
                self.assertEqual(actual, expected)

    def test_02_legacy_databases_are_archived_and_not_selected(self) -> None:
        self.assertTrue(LEGACY_PROD.is_file())
        self.assertTrue(LEGACY_QA.is_file())
        self.assertNotEqual(_sha256(LEGACY_PROD), RUNTIME_CANONICAL_SHA256)
        self.assertNotEqual(_sha256(LEGACY_QA), RUNTIME_CANONICAL_SHA256)
        self.assertEqual(db_module.current_db_path(), QA)
        self.assertEqual((db_module.VDE_WRITE_TABLE, db_module.FUELCONS_WRITE_TABLE), ("vde", "fuelcons"))
        self.assertFalse(db_module.LEGACY_FIXTURE_MODE)

    def test_03_configure_db_path_selects_instance_not_architecture(self) -> None:
        for path in (STAGING, QA, PROD):
            db_module.configure_db_path(path)
            self.assertEqual(db_module.current_db_path(), path)
            self.assertEqual((db_module.VDE_WRITE_TABLE, db_module.FUELCONS_WRITE_TABLE), ("vde", "fuelcons"))
            self.assertFalse(db_module.LEGACY_FIXTURE_MODE)

    def test_04_environment_variable_explicitly_selects_canonical_qa(self) -> None:
        with patch.dict(os.environ, {db_module.DB_PATH_ENV_VAR: str(QA)}):
            self.assertEqual(db_module._db_path_from_env().resolve(), QA.resolve())

    def test_05_all_actual_vde_origins_are_accepted_and_protected(self) -> None:
        expected = {"SOURCE_REFRESHED": 9907, "NEW_SOURCE": 1719}
        with sqlite3.connect(QA) as connection:
            actual = dict(connection.execute("SELECT record_origin,COUNT(*) FROM vde GROUP BY record_origin"))
        self.assertEqual(actual, expected)
        for origin in actual:
            self.assertEqual(normalize_record_origin(EntityType.VDE, origin), origin)
            self.assertTrue(is_record_origin_protected(EntityType.VDE, origin))
            self.assertIs(field_access_for(EntityType.VDE, origin, "make"), FieldAccess.IMMUTABLE)
            self.assertIs(field_access_for(EntityType.VDE, origin, "mass_kg"), FieldAccess.IMMUTABLE)

    def test_06_all_actual_fuelcons_origins_are_accepted_and_protected(self) -> None:
        expected = {"EPA_RECONSTRUCTED": 10572, "NEW_SOURCE": 249, "ML_PREDICTION": 1}
        with sqlite3.connect(QA) as connection:
            actual = dict(connection.execute("SELECT record_origin,COUNT(*) FROM fuelcons GROUP BY record_origin"))
        self.assertEqual(actual, expected)
        for origin in actual:
            self.assertEqual(normalize_record_origin(EntityType.FUEL_CONSUMPTION, origin), origin)
            self.assertTrue(is_record_origin_protected(EntityType.FUEL_CONSUMPTION, origin))
            self.assertIs(field_access_for(EntityType.FUEL_CONSUMPTION, origin, "method_note"), FieldAccess.IMMUTABLE)
            self.assertIs(field_access_for(EntityType.FUEL_CONSUMPTION, origin, "gco2_per_km"), FieldAccess.IMMUTABLE)

    def test_07_actual_tire_origin_is_accepted_and_protected(self) -> None:
        rows = browse_records(EntityType.TIRE, include_archived=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], rows[0]["tire_id"])
        self.assertEqual(rows[0]["record_origin"], "OTHER_IRREPRODUCIBLE_STATE")
        self.assertTrue(is_record_origin_protected(EntityType.TIRE, rows[0]["record_origin"]))
        self.assertIs(
            field_access_for(EntityType.TIRE, rows[0]["record_origin"], "rr_n_per_kn"),
            FieldAccess.IMMUTABLE,
        )

    def test_08_existing_user_created_origin_policies_are_preserved(self) -> None:
        self.assertIs(field_access_for(EntityType.VDE, "MANUAL", "make"), FieldAccess.EDITABLE)
        self.assertIs(field_access_for(EntityType.VDE, "MANUAL", "mass_kg"), FieldAccess.ADVANCED_CORRECTION)
        self.assertIs(field_access_for(EntityType.VDE, "VDE_SETUP", "make"), FieldAccess.EDITABLE)
        self.assertIs(field_access_for(EntityType.VDE, "VDE_SETUP", "mass_kg"), FieldAccess.DERIVED)
        self.assertIs(field_access_for(EntityType.FUEL_CONSUMPTION, "ESTIMATED", "method_note"), FieldAccess.EDITABLE)
        self.assertIs(field_access_for(EntityType.FUEL_CONSUMPTION, "ESTIMATED", "gco2_per_km"), FieldAccess.DERIVED)
        self.assertIs(field_access_for(EntityType.COMPONENT, "MANUAL", "component_name"), FieldAccess.EDITABLE)

    def test_09_source_managed_update_is_rejected_by_preview(self) -> None:
        current = browse_records(EntityType.VDE, limit=1)[0]
        preview = preview_change(
            ChangeCommand(
                entity_type=EntityType.VDE,
                action="UPDATE",
                record_id=current["id"],
                record_origin=current["record_origin"],
                current_record=current,
                payload={"make": "NOT ALLOWED"},
                reason="proof",
            )
        )
        self.assertFalse(preview.can_commit)
        self.assertIn("source_managed_read_only", {issue.code for issue in preview.validation_issues})

    def test_10_canonical_vde_fuel_tire_component_browse_does_not_crash(self) -> None:
        self.assertEqual(len(browse_records(EntityType.VDE)), 250)
        self.assertEqual(len(browse_records(EntityType.FUEL_CONSUMPTION)), 250)
        self.assertEqual(browse_records(EntityType.TIRE), [])
        self.assertEqual(browse_records(EntityType.COMPONENT, component_domain="transmission"), [])

    def test_11_database_management_apptest_renders_canonical_qa(self) -> None:
        app = AppTest.from_file(str(PAGE))
        app.session_state["ctx"] = {"db_path": str(QA)}
        app.run(timeout=120)
        self.assertEqual(len(app.exception), 0, [str(item.value) for item in app.exception])
        self.assertEqual(len(app.tabs), 4)
        self.assertGreaterEqual(len(app.dataframe), 4)
        self.assertTrue(any("Source-managed canonical record" in str(item.value) for item in app.info))

    def test_12_staging_ids_are_positive_without_promoting_qa(self) -> None:
        with sqlite3.connect(STAGING) as connection:
            self.assertEqual(connection.execute("SELECT MIN(id),MAX(id),COUNT(*) FROM vde").fetchone(), (1, 11626, 11626))
            self.assertEqual(connection.execute("SELECT MIN(id),MAX(id),COUNT(*) FROM fuelcons").fetchone(), (1, 10822, 10822))
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM vde WHERE id<0").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM fuelcons WHERE id<0").fetchone()[0], 0)
        with sqlite3.connect(QA) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM vde WHERE id>=0").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM fuelcons WHERE id<0").fetchone()[0], 10821)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM fuelcons WHERE id=5018").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
