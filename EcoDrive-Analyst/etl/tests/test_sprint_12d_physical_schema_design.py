from __future__ import annotations

import hashlib
import csv
import json
import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12d_physical_schema_design as design  # noqa: E402


def new_database() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(design.SCHEMA_SQL.read_text(encoding="utf-8"))
    con.executescript(design.COMPAT_SQL.read_text(encoding="utf-8"))
    return con


def insert_minimal_graph(con: sqlite3.Connection) -> None:
    con.execute(
        "INSERT INTO program (program_id, commercial_make, commercial_model, identity_status, source_identity_json, source_scope) "
        "VALUES ('P1', 'QA MAKE', 'QA MODEL', 'PROVISIONAL_SOURCE_SCOPED', '{}', 'SYNTHETIC_QA')"
    )
    con.execute(
        "INSERT INTO vehicle_configuration "
        "(vehicle_configuration_id, program_id, identity_status, source_identity_json, source_scope) "
        "VALUES ('VC1', 'P1', 'SOURCE_SCOPED', '{}', 'SYNTHETIC_QA')"
    )
    con.execute(
        "INSERT INTO component_db (component_id, component_domain, provenance_json) "
        "VALUES ('C1', 'ENGINE', '{\"origin\":\"SYNTHETIC_QA\"}')"
    )
    con.execute(
        "INSERT INTO tire_db (tire_id, tire_test_code, manufacturer, model, standard_family, rr_n_per_kn) "
        "VALUES (1, 'T-QA-1', 'QA TIRE', 'MODEL', 'CUSTOM', 7.5)"
    )
    con.execute(
        "INSERT INTO component_instance "
        "(component_instance_id, vehicle_configuration_id, component_domain, component_id, provenance_json) "
        "VALUES ('CI1', 'VC1', 'ENGINE', 'C1', '{\"origin\":\"SYNTHETIC_QA\"}')"
    )
    con.execute(
        "INSERT INTO component_resolution "
        "(component_resolution_id, vehicle_configuration_id, boundary, method, provenance_json) "
        "VALUES ('CR1', 'VC1', 'TRANSMISSION', 'QA_METHOD', '{\"origin\":\"SYNTHETIC_QA\"}')"
    )
    con.execute(
        "INSERT INTO vde "
        "(id, vehicle_configuration_id, legislation, category, make, model, mass_kg, source_semantic_status, vde_net_mj_per_km) "
        "VALUES (1, 'VC1', 'EPA', 'QA', 'QA MAKE', 'QA MODEL', 1500.0, 'DIRECT', 0.5)"
    )
    con.execute(
        "INSERT INTO run (run_id, vde_id, run_type, evidence_kind, provenance_json) "
        "VALUES ('R1', 1, 'TEST', 'SOURCE_RECORD', '{\"origin\":\"SYNTHETIC_QA\"}')"
    )
    con.execute(
        "INSERT INTO fuelcons (id, vde_id, electrification, fuel_l_per_100km) "
        "VALUES (1, 1, 'ICE', 6.5)"
    )
    con.commit()


class Sprint12DPhysicalSchemaDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime_hash_before = hashlib.sha256(design.DB_PATH.read_bytes()).hexdigest()

    def test_schema_creates_in_memory_with_nine_domain_entities(self) -> None:
        with new_database() as con:
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(set(design.DOMAIN_TABLES), tables & set(design.DOMAIN_TABLES))
            self.assertEqual(set(design.HELPER_TABLES), tables & set(design.HELPER_TABLES))
            self.assertEqual(len(tables & set(design.DOMAIN_TABLES)), 9)

    def test_foreign_keys_are_enabled_and_valid(self) -> None:
        with new_database() as con:
            self.assertEqual(con.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(con.execute("PRAGMA foreign_key_check").fetchall(), [])
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute(
                    "INSERT INTO vehicle_configuration "
                    "(vehicle_configuration_id, program_id, identity_status, source_identity_json, source_scope) "
                    "VALUES ('ORPHAN', 'MISSING', 'UNRESOLVED', '{}', 'SYNTHETIC_QA')"
                )

    def test_constraints_reject_invalid_rows(self) -> None:
        with new_database() as con:
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute(
                    "INSERT INTO program (program_id, commercial_make, commercial_model, identity_status, source_identity_json, source_scope) "
                    "VALUES ('P-BAD', 'QA', 'QA', 'GUESSED', '{}', 'SYNTHETIC_QA')"
                )
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute(
                    "INSERT INTO program (program_id, commercial_make, commercial_model, identity_status, source_identity_json, source_scope) "
                    "VALUES ('P-JSON', 'QA', 'QA', 'UNRESOLVED', 'not-json', 'SYNTHETIC_QA')"
                )
            con.execute(
                "INSERT INTO program (program_id, commercial_make, commercial_model, identity_status, source_identity_json, source_scope) "
                "VALUES ('P-VALID', 'QA', 'QA', 'UNRESOLVED', '{}', 'SYNTHETIC_QA')"
            )
            con.execute(
                "INSERT INTO vehicle_configuration "
                "(vehicle_configuration_id, program_id, identity_status, source_identity_json, source_scope) "
                "VALUES ('VC-VALID', 'P-VALID', 'UNRESOLVED', '{}', 'SYNTHETIC_QA')"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute(
                    "INSERT INTO vde (id, vehicle_configuration_id, legislation, category, make, model, mass_kg, source_semantic_status) "
                    "VALUES (99, 'VC-VALID', 'EPA', 'QA', 'QA', 'QA', -1, 'DIRECT')"
                )

    def test_valid_minimal_rows_cover_all_nine_entities(self) -> None:
        with new_database() as con:
            insert_minimal_graph(con)
            for table in design.DOMAIN_TABLES:
                self.assertEqual(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0], 1, table)
            self.assertEqual(con.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_required_cardinalities(self) -> None:
        with new_database() as con:
            insert_minimal_graph(con)
            con.execute(
                "INSERT INTO vehicle_configuration "
                "(vehicle_configuration_id, program_id, identity_status, source_identity_json, source_scope) "
                "VALUES ('VC2', 'P1', 'SOURCE_SCOPED', '{}', 'SYNTHETIC_QA')"
            )
            con.execute(
                "INSERT INTO vde (id, vehicle_configuration_id, legislation, category, make, model, mass_kg, source_semantic_status) "
                "VALUES (2, 'VC1', 'EPA', 'QA', 'QA MAKE', 'QA MODEL', 1600, 'DIRECT')"
            )
            con.execute(
                "INSERT INTO run (run_id, vde_id, run_type, evidence_kind, provenance_json) "
                "VALUES ('R2', 1, 'CALCULATION', 'ENGINEERING', '{}')"
            )
            con.execute("INSERT INTO fuelcons (id, vde_id, electrification) VALUES (2, 1, 'ICE')")
            self.assertEqual(con.execute("SELECT COUNT(*) FROM vehicle_configuration WHERE program_id='P1'").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM vde WHERE vehicle_configuration_id='VC1'").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM run WHERE vde_id=1").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM fuelcons WHERE vde_id=1").fetchone()[0], 2)

    def test_fuelcons_supports_multiple_run_lineage(self) -> None:
        with new_database() as con:
            insert_minimal_graph(con)
            con.execute(
                "INSERT INTO run (run_id, vde_id, run_type, evidence_kind, provenance_json) "
                "VALUES ('R2', 1, 'CALCULATION', 'ENGINEERING', '{}')"
            )
            con.execute(
                "INSERT INTO fuelcons_run_adoption (fuelcons_id, run_id, vde_id, adoption_role, ordinal) "
                "VALUES (1, 'R1', 1, 'PRIMARY', 0), (1, 'R2', 1, 'SUPPORTING', 1)"
            )
            self.assertEqual(con.execute("SELECT COUNT(*) FROM fuelcons_run_adoption WHERE fuelcons_id=1").fetchone()[0], 2)
            adopted = json.loads(con.execute("SELECT adopted_run_ids_json FROM fuelcons_lineage_v1 WHERE id=1").fetchone()[0])
            self.assertEqual(adopted, ["R1", "R2"])
            con.execute(
                "INSERT INTO vehicle_configuration "
                "(vehicle_configuration_id, program_id, identity_status, source_identity_json, source_scope) "
                "VALUES ('VC2', 'P1', 'SOURCE_SCOPED', '{}', 'SYNTHETIC_QA')"
            )
            con.execute(
                "INSERT INTO vde (id, vehicle_configuration_id, legislation, category, make, model, mass_kg, source_semantic_status) "
                "VALUES (2, 'VC2', 'EPA', 'QA', 'QA', 'OTHER', 1400, 'DIRECT')"
            )
            con.execute("INSERT INTO run (run_id, vde_id, run_type, evidence_kind, provenance_json) VALUES ('R-OTHER', 2, 'TEST', 'SOURCE_RECORD', '{}')")
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute(
                    "INSERT INTO fuelcons_run_adoption (fuelcons_id, run_id, vde_id, adoption_role) "
                    "VALUES (1, 'R-OTHER', 1, 'SUPPORTING')"
                )

    def test_vde_can_exist_without_component_resolution(self) -> None:
        with new_database() as con:
            insert_minimal_graph(con)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM vde WHERE id=1").fetchone()[0], 1)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM vde_component_resolution WHERE vde_id=1").fetchone()[0], 0)

    def test_jrc_unresolved_identity_is_representable(self) -> None:
        with new_database() as con:
            con.execute(
                "INSERT INTO program (program_id, commercial_make, commercial_model, identity_status, identity_confidence, source_identity_json, source_scope, source_name) "
                "VALUES ('P-JRC', 'OEM anon 1', 'Model anon 1', 'UNRESOLVED', 'LOW', '{\"row\":1}', 'JRC_2021', 'JRC')"
            )
            con.execute(
                "INSERT INTO vehicle_configuration "
                "(vehicle_configuration_id, program_id, identity_status, identity_confidence, source_identity_json, source_scope, source_name) "
                "VALUES ('VC-JRC', 'P-JRC', 'UNRESOLVED', 'LOW', '{\"row\":1}', 'JRC_2021', 'JRC')"
            )
            self.assertEqual(con.execute("SELECT identity_status FROM program WHERE program_id='P-JRC'").fetchone()[0], "UNRESOLVED")

    def test_null_and_zero_remain_distinct(self) -> None:
        with new_database() as con:
            insert_minimal_graph(con)
            con.execute("UPDATE fuelcons SET energy_Wh_per_km=0, gco2_per_km=NULL WHERE id=1")
            row = con.execute("SELECT energy_Wh_per_km, gco2_per_km FROM fuelcons_db WHERE id=1").fetchone()
            self.assertEqual(row["energy_Wh_per_km"], 0)
            self.assertIsNone(row["gco2_per_km"])

    def _runtime_columns(self, table: str) -> list[str]:
        con = sqlite3.connect(design.DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            con.execute("PRAGMA query_only=ON")
            return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]
        finally:
            con.close()

    def test_compatibility_vde_columns_match_runtime(self) -> None:
        with new_database() as con:
            actual = [row[1] for row in con.execute("PRAGMA table_info(vde_db)")]
        self.assertEqual(actual, self._runtime_columns("vde_db"))
        self.assertEqual(len(actual), 101)

    def test_compatibility_fuelcons_columns_match_runtime(self) -> None:
        with new_database() as con:
            actual = [row[1] for row in con.execute("PRAGMA table_info(fuelcons_db)")]
        self.assertEqual(actual, self._runtime_columns("fuelcons_db"))
        self.assertEqual(len(actual), 79)

    def test_compatibility_projection_values(self) -> None:
        with new_database() as con:
            insert_minimal_graph(con)
            vde = con.execute("SELECT id, make, mass_kg, vde_net_mj_per_km FROM vde_db WHERE id=1").fetchone()
            fuel = con.execute("SELECT id, vde_id, electrification, fuel_l_per_100km FROM fuelcons_db WHERE id=1").fetchone()
            self.assertEqual(tuple(vde), (1, "QA MAKE", 1500.0, 0.5))
            self.assertEqual(tuple(fuel), (1, 1, "ICE", 6.5))

    def test_master_change_preserves_snapshot_and_explicit_rebuild_adopts_new_value(self) -> None:
        with new_database() as con:
            insert_minimal_graph(con)
            con.execute("UPDATE component_db SET rated_power_kw=150.0 WHERE component_id='C1'")
            con.execute("UPDATE fuelcons SET engine_max_power_kw=135.0 WHERE id=1")

            con.execute("UPDATE component_db SET rated_power_kw=180.0 WHERE component_id='C1'")
            historical = con.execute(
                "SELECT engine_max_power_kw FROM fuelcons_db WHERE id=1"
            ).fetchone()[0]
            self.assertEqual(historical, 135.0)

            resolved_power = con.execute(
                "SELECT rated_power_kw FROM component_db WHERE component_id='C1'"
            ).fetchone()[0]
            con.execute(
                "INSERT INTO fuelcons (id, vde_id, electrification, engine_max_power_kw, provenance_json) "
                "VALUES (2, 1, 'ICE', ?, '{\"origin\":\"SYNTHETIC_QA_EXPLICIT_REBUILD\"}')",
                (resolved_power,),
            )
            rebuilt = con.execute(
                "SELECT engine_max_power_kw FROM fuelcons_db WHERE id=2"
            ).fetchone()[0]
            self.assertEqual(rebuilt, 180.0)

    def test_ownership_map_classifies_representative_snapshots(self) -> None:
        with (design.OUT / "legacy_to_canonical_column_map.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = {
                (row["legacy_table"], row["legacy_field"]): row
                for row in csv.DictReader(handle)
            }

        expected = {
            ("fuelcons_db", "engine_max_power_kw"): "RESOLVED_SNAPSHOT",
            ("fuelcons_db", "engine_max_torque_nm"): "RESOLVED_SNAPSHOT",
            ("fuelcons_db", "gear_count"): "RESOLVED_SNAPSHOT",
            ("fuelcons_db", "final_drive_ratio"): "RESOLVED_SNAPSHOT",
            ("fuelcons_db", "battery_capacity_kwh"): "RESOLVED_SNAPSHOT",
            ("fuelcons_db", "electrification"): "RESOLVED_SNAPSHOT",
            ("fuelcons_db", "fuel_l_per_100km"): "RESULT_STATE",
            ("vde_db", "make"): "COMPATIBILITY_SNAPSHOT",
            ("vde_db", "cda_m2"): "RESOLVED_SNAPSHOT",
            ("vde_db", "vde_net_mj_per_km"): "RESULT_STATE",
        }
        allowed = {
            "AUTHORITATIVE_MASTER", "RESOLVED_SNAPSHOT",
            "COMPATIBILITY_SNAPSHOT", "RESULT_STATE",
        }
        self.assertEqual(len(rows), 180)
        self.assertTrue({row["ownership_semantic"] for row in rows.values()} <= allowed)
        for key, classification in expected.items():
            self.assertEqual(rows[key]["ownership_semantic"], classification, key)
        self.assertEqual(
            rows[("fuelcons_db", "engine_max_power_kw")]["intentional_duplication"],
            "YES",
        )

    def test_program_consolidation_preserves_children(self) -> None:
        with new_database() as con:
            insert_minimal_graph(con)
            con.execute(
                "INSERT INTO program (program_id, commercial_make, commercial_model, identity_status, source_identity_json, source_scope) "
                "VALUES ('P-CANON', 'QA MAKE', 'QA MODEL', 'CONFIRMED', '{}', 'CANONICAL')"
            )
            before = con.execute("SELECT COUNT(*) FROM vehicle_configuration").fetchone()[0]
            con.execute("UPDATE program SET superseded_by_program_id='P-CANON', record_status='SUPERSEDED' WHERE program_id='P1'")
            after = con.execute("SELECT COUNT(*) FROM vehicle_configuration").fetchone()[0]
            self.assertEqual(before, after)
            self.assertEqual(con.execute("SELECT program_id FROM vehicle_configuration WHERE vehicle_configuration_id='VC1'").fetchone()[0], "P1")

    def test_component_instance_reference_mechanics(self) -> None:
        with new_database() as con:
            insert_minimal_graph(con)
            con.execute(
                "INSERT INTO component_instance "
                "(component_instance_id, vehicle_configuration_id, component_domain, provenance_json) "
                "VALUES ('CI-UNRESOLVED', 'VC1', 'TIRE', '{}')"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute(
                    "INSERT INTO component_instance "
                    "(component_instance_id, vehicle_configuration_id, component_domain, component_id, tire_id, provenance_json) "
                    "VALUES ('CI-BOTH', 'VC1', 'TIRE', 'C1', 1, '{}')"
                )
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute(
                    "INSERT INTO component_instance "
                    "(component_instance_id, vehicle_configuration_id, component_domain, component_id, provenance_json) "
                    "VALUES ('CI-WRONG', 'VC1', 'TIRE', 'C1', '{}')"
                )

    def test_query_plan_uses_vde_configuration_index(self) -> None:
        with new_database() as con:
            plan = " ".join(row[3] for row in con.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM vde WHERE vehicle_configuration_id=?", ("VC1",)
            ))
            self.assertIn("idx_vde_configuration", plan)

    def test_outputs_are_scoped_to_etl(self) -> None:
        etl_root = (ROOT / "etl").resolve()
        for path in design.OUTPUT_PATHS:
            self.assertTrue(path.resolve().is_relative_to(etl_root), path)

    def test_runtime_database_is_byte_identical(self) -> None:
        current = hashlib.sha256(design.DB_PATH.read_bytes()).hexdigest()
        self.assertEqual(self.runtime_hash_before, current)
        self.assertEqual(current.upper(), "CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262")


if __name__ == "__main__":
    unittest.main()
