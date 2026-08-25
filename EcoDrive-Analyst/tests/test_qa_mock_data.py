from __future__ import annotations

import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
import shutil

from src.vde_app.components.vde_request_lookup import component_lookup_rows, vde_lookup_rows
from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import (
    COMPLETE_SAE_QA_TIRE_SPECS,
    DEFAULT_QA_DB_PATH,
    GOLDEN_QA_SCENARIO,
    QA_COMPONENT_FIXTURES,
    QA_DATA_DIR,
    build_fuelcons_seed_rows,
    build_seed_manifest,
    build_tire_seed_rows,
    build_vde_seed_rows,
    inertia_row_for_mass,
    is_safe_qa_db_path,
    seed_qa_database,
    seed_qa_fuelcons_mock_rows,
    seeded_database_digest,
)
from src.vde_core.roadload import KPA_PER_PSI, calculate_applied_rrc_by_axle
from src.vde_core.repositories import fetch_vde_all_rows, fetch_vde_by_id
from src.vde_core.tire_roadload_service import get_tire_by_code, preview_tire_roadload_from_row
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_compact_state import apply_v22_baseline, create_v22_state


class QaMockDataTests(unittest.TestCase):
    @staticmethod
    def _simple_reference_rrc(row: dict, *, speed_kph: float = 80.0) -> float:
        scale = float(row["sae_reference_pressure_kpa"]) ** float(row["sae_alpha"])
        scale *= float(row["sae_reference_load_n"]) ** float(row["sae_beta"])
        frr_n = scale * (
            float(row["sae_a"])
            + (float(row["sae_b"]) * speed_kph)
            + (float(row["sae_c"]) * speed_kph * speed_kph)
        )
        return frr_n * 1000.0 / float(row["sae_reference_load_n"])

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

    def test_complete_sae_seed_records_exist(self):
        tires = {row["tire_test_code"]: row for row in build_tire_seed_rows()}
        manifest_tires = {item["qa_id"] for item in build_seed_manifest()["tires"]}

        for spec in COMPLETE_SAE_QA_TIRE_SPECS:
            with self.subTest(tire_code=spec["qa_id"]):
                row = tires[spec["qa_id"]]
                self.assertEqual(row["id"], spec["row_id"])
                self.assertEqual(row["standard_family"], "SAE")
                self.assertEqual(row["calculation_mode"], "SAE_J2452")
                self.assertIn("Synthetic QA data", row["notes"])
                self.assertIn(spec["qa_id"], manifest_tires)

    def test_complete_sae_seed_reference_equation_matches_stored_rrc(self):
        tires = {row["tire_test_code"]: row for row in build_tire_seed_rows()}

        for spec in COMPLETE_SAE_QA_TIRE_SPECS:
            with self.subTest(tire_code=spec["qa_id"]):
                row = tires[spec["qa_id"]]
                calculated = self._simple_reference_rrc(row, speed_kph=80.0)
                self.assertAlmostEqual(calculated, float(spec["rr_n_per_kn"]), places=3)
                self.assertAlmostEqual(float(row["rr_n_per_kn"]), float(spec["rr_n_per_kn"]), places=9)

    def test_complete_sae_seed_preserves_pressure_and_load_unit_conversions(self):
        tires = {row["tire_test_code"]: row for row in build_tire_seed_rows()}

        for spec in COMPLETE_SAE_QA_TIRE_SPECS:
            with self.subTest(tire_code=spec["qa_id"]):
                row = tires[spec["qa_id"]]
                self.assertAlmostEqual(float(row["sae_reference_pressure_kpa"]), float(spec["pressure_psi"]) * KPA_PER_PSI, places=3)
                self.assertAlmostEqual(float(row["sae_reference_load_n"]), float(spec["load_kg"]) * 9.80665, places=3)
                self.assertEqual(float(row["test_speed_value"]), 80.0)

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
        self.assertEqual(QA_COMPONENT_FIXTURES["transmission"][0]["net_bridge_eligible"], "TRUE")
        self.assertEqual(QA_COMPONENT_FIXTURES["brake"][0]["component_type"], "BRAKE_STANDARD")
        self.assertEqual(QA_COMPONENT_FIXTURES["parasitic"][0]["component_type"], "OTHER_RESIDUAL_COMPONENT_LOSSES")

    def test_incomplete_fixtures_are_intentionally_incomplete(self):
        tires = {row["tire_test_code"]: row for row in build_tire_seed_rows()}
        baseline_rows = {row["id"]: row for row in build_vde_seed_rows()}

        for tire_code in ("TIRE-QA-001", "TIRE-QA-002", "TIRE-QA-003", "TIRE-QA-005", "TIRE-QA-006", "TIRE-QA-009"):
            self.assertIsNotNone(tires[tire_code].get("test_pressure_value"))
            self.assertIsNotNone(tires[tire_code].get("iso_test_pressure_kpa"))
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

        self.assertEqual(len(rows), 8)
        self.assertEqual(tire["tire_test_code"], "TIRE-QA-010")
        self.assertEqual(nominal["model"], "Nominal EPA baseline")

    def test_nominal_seeded_baseline_references_qa_base_without_explicit_tire_abc(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            nominal = fetch_vde_by_id(900001)

        self.assertEqual(int(nominal["front_tire_id"]), 920101)
        self.assertEqual(int(nominal["rear_tire_id"]), 920101)
        self.assertIsNone(nominal["tire_A_final"])
        self.assertIsNone(nominal["tire_B_final"])
        self.assertIsNone(nominal["tire_C_final"])

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

    def test_tire_lookup_browse_returns_seeded_rows_without_search(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        component_lookup_rows.clear()
        with db_module.using_db_path(db_path):
            tire_rows = component_lookup_rows("tire", "", limit=25)

        codes = {str(row["Tire code"]) for row in tire_rows}
        self.assertGreaterEqual(len(tire_rows), 8)
        self.assertLessEqual(len(tire_rows), 25)
        self.assertIn("QA-BASE", codes)
        self.assertIn("QA-ECO", codes)
        self.assertIn("QA-HIGH-RRC", codes)
        self.assertIn("QA-LOAD", codes)
        self.assertIn("QA-NEUTRAL", codes)
        self.assertIn("QA-SAME-RRC-DIFF-SAE", codes)
        self.assertIn("QA-LOW-PRESSURE", codes)
        self.assertIn("QA-HIGH-PRESSURE", codes)
        self.assertIn("QA-INCOMPLETE", codes)

        qa_base = next(row for row in tire_rows if str(row["Tire code"]) == "QA-BASE")
        self.assertIn("alpha", qa_base)
        self.assertIn("beta", qa_base)
        self.assertIn("a", qa_base)
        self.assertIn("b", qa_base)
        self.assertIn("c", qa_base)

    def test_tire_lookup_search_qa_returns_all_complete_sae_records(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        component_lookup_rows.clear()
        with db_module.using_db_path(db_path):
            tire_rows = component_lookup_rows("tire", "QA", limit=25)

        codes = {str(row["Tire code"]) for row in tire_rows}
        expected = {str(spec["qa_id"]) for spec in COMPLETE_SAE_QA_TIRE_SPECS}
        self.assertTrue(expected.issubset(codes))
        self.assertIn("QA-INCOMPLETE", codes)

    def test_tire_lookup_search_filters_synthetic_qa_records(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        component_lookup_rows.clear()
        with db_module.using_db_path(db_path):
            tire_rows = component_lookup_rows("tire", "QA-ECO", limit=25)

        self.assertEqual(len(tire_rows), 1)
        self.assertEqual(str(tire_rows[0]["lookup_id"]), "920102")
        self.assertEqual(tire_rows[0]["Tire code"], "QA-ECO")

    def test_incomplete_tire_lookup_fixture_is_browseable(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        component_lookup_rows.clear()
        with db_module.using_db_path(db_path):
            tire_rows = component_lookup_rows("tire", "QA-INCOMPLETE", limit=25)

        self.assertEqual(len(tire_rows), 1)
        self.assertEqual(tire_rows[0]["Tire code"], "QA-INCOMPLETE")
        self.assertIsNone(tire_rows[0]["Reference pressure"])
        self.assertEqual(float(tire_rows[0]["Test load"]), 610.0)

    def test_seeded_lookup_fetch_preserves_full_sae_coefficients(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            tire = get_tire_by_code("QA-LOAD")

        self.assertEqual(tire["tire_test_code"], "QA-LOAD")
        self.assertAlmostEqual(float(tire["rr_n_per_kn"]), 8.8, places=9)
        self.assertAlmostEqual(float(tire["sae_alpha"]), -0.28, places=9)
        self.assertAlmostEqual(float(tire["sae_beta"]), 1.05, places=9)
        self.assertAlmostEqual(float(tire["sae_a"]), 0.0231280363, places=10)
        self.assertAlmostEqual(float(tire["sae_b"]), 0.00002200, places=10)
        self.assertAlmostEqual(float(tire["sae_c"]), 0.0000000600, places=10)

    def test_complete_sae_seed_neutral_lookup_is_engineering_neutral(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            qa_base = get_tire_by_code("QA-BASE")
            qa_neutral = get_tire_by_code("QA-NEUTRAL")
            base_result = calculate_applied_rrc_by_axle(
                front_tire=qa_base,
                rear_tire=qa_base,
                inputs={"mass_kg": 1600.0, "front_weight_distribution_pct": 55.0, "front_pressure_kpa": 38.0 * KPA_PER_PSI, "rear_pressure_kpa": 38.0 * KPA_PER_PSI},
            )
            neutral_result = calculate_applied_rrc_by_axle(
                front_tire=qa_neutral,
                rear_tire=qa_neutral,
                inputs={"mass_kg": 1600.0, "front_weight_distribution_pct": 55.0, "front_pressure_kpa": 38.0 * KPA_PER_PSI, "rear_pressure_kpa": 38.0 * KPA_PER_PSI},
            )

        self.assertAlmostEqual(base_result["vehicle_rrc_n_per_kn"], neutral_result["vehicle_rrc_n_per_kn"], places=9)
        self.assertAlmostEqual(base_result["vehicle_force_n"], neutral_result["vehicle_force_n"], places=9)

    def test_same_reference_rrc_with_different_sae_coefficients_changes_curve(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)
        reference_inputs = {
            "mass_kg": 2440.0,
            "front_weight_distribution_pct": 50.0,
            "front_pressure_kpa": 38.0 * KPA_PER_PSI,
            "rear_pressure_kpa": 38.0 * KPA_PER_PSI,
        }

        with db_module.using_db_path(db_path):
            qa_base = get_tire_by_code("QA-BASE")
            qa_same_rrc = get_tire_by_code("QA-SAME-RRC-DIFF-SAE")
            base_result = calculate_applied_rrc_by_axle(
                front_tire=qa_base,
                rear_tire=qa_base,
                inputs=reference_inputs,
            )
            same_rrc_result = calculate_applied_rrc_by_axle(
                front_tire=qa_same_rrc,
                rear_tire=qa_same_rrc,
                inputs=reference_inputs,
            )
            base_preview = preview_tire_roadload_from_row(
                {"legislation": "EPA", "mass_kg": 2440.0, "test_mass_kg": 2440.0, "weight_dist_fr_pct": 50.0, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
                {"front_tire_id": 920101, "rear_tire_id": 920101, "tire_load_mass_basis": "TEST_MASS"},
            )
            same_rrc_preview = preview_tire_roadload_from_row(
                {"legislation": "EPA", "mass_kg": 2440.0, "test_mass_kg": 2440.0, "weight_dist_fr_pct": 50.0, "front_pressure_psi": 38.0, "rear_pressure_psi": 38.0},
                {"front_tire_id": 920109, "rear_tire_id": 920109, "tire_load_mass_basis": "TEST_MASS"},
            )

        self.assertAlmostEqual(base_result["vehicle_rrc_n_per_kn"], 8.0, places=6)
        self.assertAlmostEqual(same_rrc_result["vehicle_rrc_n_per_kn"], 8.0, places=6)
        self.assertAlmostEqual(base_result["vehicle_force_n"], same_rrc_result["vehicle_force_n"], places=5)

        base_curve = base_preview["calculation"]["total_final_abc"]
        same_rrc_curve = same_rrc_preview["calculation"]["total_final_abc"]
        for speed in (0.0, 50.0, 120.0):
            base_force = base_curve["A"] + (base_curve["B"] * speed) + (base_curve["C"] * speed * speed)
            same_force = same_rrc_curve["A"] + (same_rrc_curve["B"] * speed) + (same_rrc_curve["C"] * speed * speed)
            self.assertNotAlmostEqual(base_force, same_force, places=6)
        base_force_80 = base_curve["A"] + (base_curve["B"] * 80.0) + (base_curve["C"] * 80.0 * 80.0)
        same_force_80 = same_rrc_curve["A"] + (same_rrc_curve["B"] * 80.0) + (same_rrc_curve["C"] * 80.0 * 80.0)
        self.assertAlmostEqual(base_force_80, same_force_80, places=6)

    def test_complete_sae_seed_rrc_direction_matches_expectation(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            qa_eco = get_tire_by_code("QA-ECO")
            qa_base = get_tire_by_code("QA-BASE")
            qa_high = get_tire_by_code("QA-HIGH-RRC")
            eco_result = calculate_applied_rrc_by_axle(
                front_tire=qa_eco,
                rear_tire=qa_eco,
                inputs={"mass_kg": 1600.0, "front_weight_distribution_pct": 55.0, "front_pressure_kpa": 35.0 * KPA_PER_PSI, "rear_pressure_kpa": 35.0 * KPA_PER_PSI},
            )
            base_result = calculate_applied_rrc_by_axle(
                front_tire=qa_base,
                rear_tire=qa_base,
                inputs={"mass_kg": 1600.0, "front_weight_distribution_pct": 55.0, "front_pressure_kpa": 38.0 * KPA_PER_PSI, "rear_pressure_kpa": 38.0 * KPA_PER_PSI},
            )
            high_result = calculate_applied_rrc_by_axle(
                front_tire=qa_high,
                rear_tire=qa_high,
                inputs={"mass_kg": 1600.0, "front_weight_distribution_pct": 55.0, "front_pressure_kpa": 32.0 * KPA_PER_PSI, "rear_pressure_kpa": 32.0 * KPA_PER_PSI},
            )

        self.assertLess(eco_result["vehicle_rrc_n_per_kn"], base_result["vehicle_rrc_n_per_kn"])
        self.assertGreater(high_result["vehicle_rrc_n_per_kn"], base_result["vehicle_rrc_n_per_kn"])

    def test_complete_sae_seed_load_sensitivity_flows_through_preview_path(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            light = preview_tire_roadload_from_row(
                {"legislation": "EPA", "mass_kg": 1500.0, "test_mass_kg": 1500.0, "weight_dist_fr_pct": 55.0, "front_pressure_psi": 30.0, "rear_pressure_psi": 30.0},
                {"front_tire_id": 920104, "rear_tire_id": 920104, "tire_load_mass_basis": "TEST_MASS"},
            )
            heavy = preview_tire_roadload_from_row(
                {"legislation": "EPA", "mass_kg": 1900.0, "test_mass_kg": 1900.0, "weight_dist_fr_pct": 55.0, "front_pressure_psi": 30.0, "rear_pressure_psi": 30.0},
                {"front_tire_id": 920104, "rear_tire_id": 920104, "tire_load_mass_basis": "TEST_MASS"},
            )

        self.assertGreater(heavy["calculation"]["applied_rr_n_per_kn"], light["calculation"]["applied_rr_n_per_kn"])
        self.assertGreater(heavy["calculation"]["applied_rrc"]["vehicle_force_n"], light["calculation"]["applied_rrc"]["vehicle_force_n"])
        self.assertGreater(heavy["calculation"]["total_final_abc"]["A"], light["calculation"]["total_final_abc"]["A"])

    def test_complete_sae_seed_pressure_sensitivity_matches_negative_alpha_direction(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            tire = get_tire_by_code("QA-BASE")
            low = calculate_applied_rrc_by_axle(
                front_tire=tire,
                rear_tire=tire,
                inputs={"mass_kg": 1600.0, "front_weight_distribution_pct": 55.0, "front_pressure_kpa": 30.0 * KPA_PER_PSI, "rear_pressure_kpa": 30.0 * KPA_PER_PSI},
            )
            ref = calculate_applied_rrc_by_axle(
                front_tire=tire,
                rear_tire=tire,
                inputs={"mass_kg": 1600.0, "front_weight_distribution_pct": 55.0, "front_pressure_kpa": 38.0 * KPA_PER_PSI, "rear_pressure_kpa": 38.0 * KPA_PER_PSI},
            )
            high = calculate_applied_rrc_by_axle(
                front_tire=tire,
                rear_tire=tire,
                inputs={"mass_kg": 1600.0, "front_weight_distribution_pct": 55.0, "front_pressure_kpa": 45.0 * KPA_PER_PSI, "rear_pressure_kpa": 45.0 * KPA_PER_PSI},
            )

        self.assertGreater(low["vehicle_rrc_n_per_kn"], ref["vehicle_rrc_n_per_kn"])
        self.assertGreater(ref["vehicle_rrc_n_per_kn"], high["vehicle_rrc_n_per_kn"])

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
        self.assertEqual(vde_count, 8)
        self.assertEqual(tire_count, 19)

    def test_default_demo_path_stays_under_qa_directory(self):
        self.assertTrue(is_safe_qa_db_path(DEFAULT_QA_DB_PATH))

    def test_wltp_baseline_exists_alongside_the_epa_baselines(self):
        baseline_rows = {row["id"]: row for row in build_vde_seed_rows()}
        self.assertEqual(baseline_rows[900008]["legislation"], "WLTP")
        self.assertEqual(baseline_rows[900008]["cycle_name"], "WLTC")
        epa_ids = [row_id for row_id, row in baseline_rows.items() if row["legislation"] == "EPA"]
        self.assertGreaterEqual(len(epa_ids), 1)


class QaFuelconsMockRowsTests(unittest.TestCase):
    """Comparison Browse Compact UX + QA Data package: opt-in FuelCons mock
    rows for exercising the Fuel Economy / NET / Complete Engineering Data
    filters and presets with real data instead of empty catalogs.
    """

    def _temp_db_path(self) -> Path:
        QA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="qa_fuelcons_", dir=str(QA_DATA_DIR)))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir / "qa_fuelcons.db"

    def test_seeding_a_qa_database_leaves_fuelcons_db_empty_by_default(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)
        with sqlite3.connect(str(db_path)) as con:
            count = con.execute("SELECT COUNT(*) FROM fuelcons_db").fetchone()[0]
        self.assertEqual(count, 0)

    def test_opt_in_helper_inserts_every_mock_row_linked_to_a_real_vde(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(db_path)

        vde_ids = {row["id"] for row in build_vde_seed_rows()}
        rows = build_fuelcons_seed_rows()
        self.assertGreaterEqual(len(rows), 6)
        with sqlite3.connect(str(db_path)) as con:
            con.row_factory = sqlite3.Row
            stored = {r["id"]: dict(r) for r in con.execute("SELECT * FROM fuelcons_db").fetchall()}
        self.assertEqual(set(stored.keys()), {row["id"] for row in rows})
        for row in rows:
            self.assertIn(row["vde_id"], vde_ids)

    def test_coverage_includes_fuel_economy_present_and_missing_cases(self):
        rows = build_fuelcons_seed_rows()
        with_fe = [r for r in rows if r.get("fuel_l_per_100km") is not None]
        without_fe = [r for r in rows if r.get("fuel_l_per_100km") is None]
        self.assertGreaterEqual(len(with_fe), 1)
        self.assertGreaterEqual(len(without_fe), 1)

    def test_coverage_includes_a_net_available_and_a_net_unavailable_case(self):
        baseline_rows = {row["id"]: row for row in build_vde_seed_rows()}
        rows = build_fuelcons_seed_rows()
        net_available = [r for r in rows if baseline_rows[r["vde_id"]].get("trans_A_coef_N") is not None]
        net_unavailable = [r for r in rows if baseline_rows[r["vde_id"]].get("trans_A_coef_N") is None]
        self.assertGreaterEqual(len(net_available), 1)
        self.assertGreaterEqual(len(net_unavailable), 1)

    def test_coverage_includes_both_epa_and_wltp_scenarios(self):
        baseline_rows = {row["id"]: row for row in build_vde_seed_rows()}
        rows = build_fuelcons_seed_rows()
        legislations = {baseline_rows[r["vde_id"]]["legislation"] for r in rows}
        self.assertIn("EPA", legislations)
        self.assertIn("WLTP", legislations)

    def test_opt_in_helper_is_idempotent(self):
        db_path = self._temp_db_path()
        seed_qa_database(db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(db_path)
        seed_qa_fuelcons_mock_rows(db_path)
        with sqlite3.connect(str(db_path)) as con:
            count = con.execute("SELECT COUNT(*) FROM fuelcons_db").fetchone()[0]
        self.assertEqual(count, len(build_fuelcons_seed_rows()))


if __name__ == "__main__":
    unittest.main()
