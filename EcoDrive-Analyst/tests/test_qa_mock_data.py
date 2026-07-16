from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
import shutil

from src.vde_app.components.vde_request_lookup import component_lookup_rows, vde_lookup_rows
from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import (
    DEFAULT_QA_DB_PATH,
    GOLDEN_QA_SCENARIO,
    QA_COMPONENT_FIXTURES,
    QA_DATA_DIR,
    build_seed_manifest,
    build_tire_seed_rows,
    build_vde_seed_rows,
    inertia_row_for_mass,
    is_safe_qa_db_path,
    seed_qa_database,
    seeded_database_digest,
)
from src.vde_core.repositories import fetch_vde_all_rows, fetch_vde_by_id
from src.vde_core.tire_roadload_service import get_tire_by_code
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_compact_state import apply_v22_baseline, create_v22_state


class QaMockDataTests(unittest.TestCase):
    def _temp_db_path(self) -> Path:
        QA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="qa_seed_", dir=str(QA_DATA_DIR)))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir / "qa_seed.db"

    def test_manifest_has_unique_stable_ids(self):
        manifest = build_seed_manifest()
        baseline_ids = [item["qa_id"] for item in manifest["baselines"]]
        tire_ids = [item["qa_id"] for item in manifest["tires"]]
        tire_codes = [row["tire_test_code"] for row in build_tire_seed_rows()]

        self.assertEqual(len(baseline_ids), len(set(baseline_ids)))
        self.assertEqual(len(tire_ids), len(set(tire_ids)))
        self.assertEqual(len(tire_codes), len(set(tire_codes)))

    def test_vde_seed_rows_follow_canonical_inertia_table(self):
        for row in build_vde_seed_rows():
            with self.subTest(vde_id=row["id"]):
                canonical = inertia_row_for_mass(row["mass_kg"])
                if row["id"] == 900007:
                    self.assertIsNotNone(canonical)
                    self.assertNotEqual(float(canonical["inertia_class_kg"]), float(row["inertia_class"]))
                    continue
                self.assertIsNotNone(canonical)
                self.assertEqual(float(canonical["inertia_class_kg"]), float(row["inertia_class"]))
                self.assertEqual(float(row["test_mass_kg"]), float(row["inertia_class"]))

    def test_golden_scenario_and_component_fixtures_exist(self):
        manifest = build_seed_manifest()
        baseline_ids = {item["vde_id"] for item in manifest["baselines"]}
        tire_ids = {item["tire_id"] for item in manifest["tires"]}

        self.assertIn(GOLDEN_QA_SCENARIO["baseline_id"], baseline_ids)
        self.assertIn(GOLDEN_QA_SCENARIO["tire_id"], tire_ids)
        self.assertIn("transmission", QA_COMPONENT_FIXTURES)
        self.assertIn("TRANS-MOCK-001", {item["component_id"] for item in QA_COMPONENT_FIXTURES["transmission"]})

    def test_incomplete_fixtures_are_intentionally_incomplete(self):
        tires = {row["tire_test_code"]: row for row in build_tire_seed_rows()}
        baseline_rows = {row["id"]: row for row in build_vde_seed_rows()}

        self.assertIsNone(tires["TIRE-QA-004"].get("test_pressure_value"))
        self.assertIsNone(tires["TIRE-QA-008"].get("sae_reference_load_n"))
        self.assertEqual(tires["TIRE-QA-009"].get("test_mileage_km"), 0.0)
        self.assertIsNone(baseline_rows[900006].get("trans_A_coef_N"))
        self.assertIsNone(baseline_rows[900006].get("vde_net_mj_per_km"))

    def test_seed_generates_real_schema_and_repository_reads(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            rows = fetch_vde_all_rows()
            tire = get_tire_by_code("TIRE-QA-010")
            nominal = fetch_vde_by_id(900001)

        self.assertEqual(len(rows), 7)
        self.assertEqual(tire["tire_test_code"], "TIRE-QA-010")
        self.assertEqual(nominal["model"], "Nominal EPA baseline")

    def test_seed_is_deterministic_across_overwrite(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)
        first_digest = seeded_database_digest(db_path)

        seed_qa_database(db_path, overwrite=True)
        second_digest = seeded_database_digest(db_path)

        self.assertEqual(first_digest, second_digest)

    def test_overwrite_guard_rejects_non_qa_paths(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
            outside_path = Path(handle.name)
        self.addCleanup(lambda: outside_path.unlink(missing_ok=True))
        outside_path.write_bytes(b"placeholder")

        self.assertFalse(is_safe_qa_db_path(outside_path))
        with self.assertRaises(ValueError):
            seed_qa_database(outside_path, overwrite=True)

    def test_lookup_adapters_can_read_seeded_db(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        component_lookup_rows.clear()
        vde_lookup_rows.clear()
        with db_module.using_db_path(db_path):
            baseline_rows = vde_lookup_rows("mass", "VDE-QA-001")
            tire_rows = component_lookup_rows("tire", "TIRE-QA-010")

        self.assertEqual(len(baseline_rows), 1)
        self.assertEqual(str(baseline_rows[0]["VDE ID"]), "900001")
        self.assertEqual(len(tire_rows), 1)
        self.assertEqual(str(tire_rows[0]["lookup_id"]), "920010")

    def test_preview_bundle_smoke_runs_on_seeded_db(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            baseline_row = fetch_vde_by_id(900001)
            state = apply_v22_baseline(create_v22_state(), baseline_row)
            bundle = build_v22_preview_bundle(
                state,
                baseline_context=compact_baseline_context(state),
            )

        self.assertTrue(bundle["fingerprint"])
        self.assertIn("resolution_result", bundle)
        self.assertIn("validation_summary", bundle)

    def test_seeded_db_contains_expected_tables_and_counts(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with sqlite3.connect(str(db_path)) as con:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            vde_count = con.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0]
            tire_count = con.execute("SELECT COUNT(*) FROM tire_roadload_db").fetchone()[0]

        self.assertIn("vde_db", tables)
        self.assertIn("tire_roadload_db", tables)
        self.assertEqual(vde_count, 7)
        self.assertEqual(tire_count, 10)

    def test_default_demo_path_stays_under_qa_directory(self):
        self.assertTrue(is_safe_qa_db_path(DEFAULT_QA_DB_PATH))


if __name__ == "__main__":
    unittest.main()
