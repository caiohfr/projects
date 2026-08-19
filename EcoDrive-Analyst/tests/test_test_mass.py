import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.test_mass import (
    compute_wltp_test_masses,
    get_wltp_light_duty_scope_warning,
    normalize_wltp_category,
    resolve_test_mass_kg,
)


class TestMassTests(unittest.TestCase):
    def test_wltp_m1_known_case(self):
        result = compute_wltp_test_masses(
            mass_kg=2666,
            payload_kg=373,
            options_kg=22,
            wltp_category="M1",
        )

        self.assertEqual(result.laden_mass_kg, 3039)
        self.assertAlmostEqual(result.test_mass_low_kg, 2806.95)
        self.assertAlmostEqual(result.test_mass_high_kg, 2825.65)
        self.assertAlmostEqual(result.available_load_low_kg, 273.0)
        self.assertAlmostEqual(result.available_load_high_kg, 251.0)

    def test_category_normalization_accepts_supported_values(self):
        self.assertEqual(normalize_wltp_category("M1"), "M1")
        self.assertEqual(normalize_wltp_category("m1"), "M1")
        self.assertEqual(normalize_wltp_category(" M1 "), "M1")
        self.assertEqual(normalize_wltp_category("M2"), "M2")
        self.assertEqual(normalize_wltp_category("n1"), "N1")
        self.assertEqual(normalize_wltp_category("N2"), "N2")
        self.assertEqual(normalize_wltp_category(1), "M1")
        self.assertEqual(normalize_wltp_category("1"), "M1")
        self.assertIsNone(normalize_wltp_category(2))
        self.assertIsNone(normalize_wltp_category("2"))

    def test_wltp_m2_uses_passenger_vehicle_factor(self):
        result = compute_wltp_test_masses(
            mass_kg=2000,
            payload_kg=500,
            options_kg=50,
            wltp_category="M2",
        )

        self.assertEqual(result.wltp_category, "M2")
        self.assertAlmostEqual(result.load_factor, 0.15)
        self.assertAlmostEqual(result.test_mass_low_kg, 2160.0)
        self.assertAlmostEqual(result.test_mass_high_kg, 2202.5)

    def test_wltp_n2_uses_goods_vehicle_factor(self):
        result = compute_wltp_test_masses(
            mass_kg=2000,
            payload_kg=500,
            options_kg=50,
            wltp_category="N2",
        )

        self.assertEqual(result.wltp_category, "N2")
        self.assertAlmostEqual(result.load_factor, 0.28)
        self.assertAlmostEqual(result.test_mass_low_kg, 2212.0)
        self.assertAlmostEqual(result.test_mass_high_kg, 2248.0)

    def test_scope_warning_is_none_within_standard_light_duty_range(self):
        warning = get_wltp_light_duty_scope_warning(
            category="M1",
            reference_mass_kg=2610,
        )

        self.assertIsNone(warning)

    def test_scope_warning_appears_for_extension_range(self):
        warning = get_wltp_light_duty_scope_warning(
            category="M2",
            reference_mass_kg=2700,
        )

        self.assertIn("above 2610 kg and up to 2840 kg", warning)

    def test_scope_warning_strengthens_above_upper_threshold(self):
        warning = get_wltp_light_duty_scope_warning(
            category="N2",
            reference_mass_kg=2900,
        )

        self.assertIn("above 2840 kg", warning)

    def test_scope_warning_allows_unknown_category_message(self):
        warning = get_wltp_light_duty_scope_warning(
            category="X9",
            reference_mass_kg=2500,
        )

        self.assertEqual(warning, "Unknown WLTP category.")

    def test_compute_wltp_test_masses_surfaces_scope_warning_without_blocking(self):
        result = compute_wltp_test_masses(
            mass_kg=2750,
            payload_kg=200,
            options_kg=10,
            wltp_category="N2",
        )

        self.assertIsNotNone(result.test_mass_low_kg)
        self.assertIsNotNone(result.test_mass_high_kg)
        self.assertEqual(result.reference_mass_kg, 2850.0)
        self.assertIn("above 2840 kg", result.light_duty_scope_warning)

    def test_wltp_missing_payload_returns_warning_and_none(self):
        result = compute_wltp_test_masses(
            mass_kg=2666,
            payload_kg=None,
            options_kg=22,
            wltp_category="M1",
        )

        self.assertIsNone(result.test_mass_low_kg)
        self.assertIsNone(result.test_mass_high_kg)
        self.assertTrue(result.warnings)

    def test_wltp_missing_options_makes_tmh_equal_tml(self):
        result = compute_wltp_test_masses(
            mass_kg=2666,
            payload_kg=373,
            options_kg=None,
            wltp_category="M1",
        )

        self.assertAlmostEqual(result.test_mass_low_kg, result.test_mass_high_kg)

    def test_resolve_wltp_tmh(self):
        resolved, basis, warnings = resolve_test_mass_kg(
            basis="WLTP_TMH",
            mass_kg=2666,
            test_mass_high_kg=2825.65,
        )

        self.assertAlmostEqual(resolved, 2825.65)
        self.assertEqual(basis, "WLTP_TMH")
        self.assertFalse(warnings)

    def test_resolve_curb_plus_driver(self):
        resolved, basis, warnings = resolve_test_mass_kg(
            basis="CURB_PLUS_DRIVER",
            mass_kg=2666,
            options_kg=22,
        )

        self.assertAlmostEqual(resolved, 2763.0)
        self.assertEqual(basis, "CURB_PLUS_DRIVER")
        self.assertFalse(warnings)

    def test_resolve_gvwr(self):
        resolved, basis, warnings = resolve_test_mass_kg(
            basis="GVWR",
            mass_kg=2666,
            gvwr_kg=3100,
        )

        self.assertAlmostEqual(resolved, 3100.0)
        self.assertEqual(basis, "GVWR")
        self.assertFalse(warnings)

    def test_resolve_gcwr_trailer_warns_when_trailer_mass_missing(self):
        resolved, basis, warnings = resolve_test_mass_kg(
            basis="GCWR_TRAILER",
            mass_kg=2666,
            gcwr_kg=5200,
        )

        self.assertAlmostEqual(resolved, 5200.0)
        self.assertEqual(basis, "GCWR_TRAILER")
        self.assertIn("trailer_mass_kg is unavailable", warnings[0])

    def test_db_migration_is_idempotent_and_adds_new_columns(self):
        original_db_path = db_module.DB_PATH
        temp_db_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpfile:
                temp_db_path = Path(tmpfile.name)
            db_module.DB_PATH = temp_db_path
            db_module.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            db_module.ensure_db()
            db_module.ensure_migrations()
            db_module.ensure_migrations()

            con = sqlite3.connect(db_module.DB_PATH)
            try:
                cols = {
                    row[1]
                    for row in con.execute("PRAGMA table_info(vde_db);").fetchall()
                }
            finally:
                con.close()

            self.assertIn("test_mass_low_kg", cols)
            self.assertIn("test_mass_high_kg", cols)
            self.assertIn("test_mass_basis", cols)
            self.assertIn("gvwr_kg", cols)
            self.assertIn("gcwr_kg", cols)
            self.assertIn("trailer_mass_kg", cols)
            self.assertIn("trailer_code", cols)
            self.assertIn("trailer_roadload_source", cols)
            self.assertIn("trailer_A_coef_N", cols)
            self.assertIn("trailer_B_coef_Npkph", cols)
            self.assertIn("trailer_C_coef_Npkph2", cols)
            self.assertIn("mass_rule_status", cols)
            self.assertIn("mass_rule_notes", cols)
        finally:
            db_module.DB_PATH = original_db_path
            if temp_db_path and temp_db_path.exists():
                try:
                    os.unlink(temp_db_path)
                except PermissionError:
                    pass

    def test_migration_does_not_silently_overwrite_test_mass(self):
        original_db_path = db_module.DB_PATH
        temp_db_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpfile:
                temp_db_path = Path(tmpfile.name)
            db_module.DB_PATH = temp_db_path
            db_module.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            db_module.ensure_db()

            con = sqlite3.connect(db_module.DB_PATH)
            try:
                con.execute(
                    """
                    INSERT INTO vde_db (
                        legislation, category, make, model, year, mass_kg
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("WLTP", "M1", "TEST", "ROW", 2026, 2666.0),
                )
                con.commit()
            finally:
                con.close()

            db_module.ensure_migrations()

            con = sqlite3.connect(db_module.DB_PATH)
            try:
                row = con.execute(
                    "SELECT test_mass_kg, test_mass_basis FROM vde_db LIMIT 1"
                ).fetchone()
            finally:
                con.close()

            self.assertIsNone(row[0])
            self.assertIsNone(row[1])
        finally:
            db_module.DB_PATH = original_db_path
            if temp_db_path and temp_db_path.exists():
                try:
                    os.unlink(temp_db_path)
                except PermissionError:
                    pass


if __name__ == "__main__":
    unittest.main()
