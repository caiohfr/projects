from __future__ import annotations

import math
import json
from pathlib import Path
import sqlite3
import unittest

from etl.scripts.sprint_12_closure_phase2 import (
    EPA_COLD,
    EPA_CUSTOM,
    EPA_NORMAL,
    HP_TO_KW,
    assign_temporal_parents,
    classify_epa_roadload_condition,
    cylinders_rotors,
    exact_signature,
    group_signature,
    horsepower_to_kw,
)
from etl.scripts.sprint_12e_migration_rehearsal import EPA_VDE_CARRYOVER_FIELDS
from etl.scripts.sprint_12e2_epa_fuelcons_reconstruction import (
    fuelcons_run_evidence_signature,
    run_carryover_signature,
)
from src.vde_core.vde_request_save import _proposal_row_payload


class Sprint12ClosurePhase2Tests(unittest.TestCase):
    def test_epa_normal_roadload_condition_ignores_run_schedule(self):
        result = classify_epa_roadload_condition({
            "Test Category": "Cold CO",
            "Test Procedure Description": "SC03",
        })
        self.assertEqual(result["cycle_name"], EPA_NORMAL)
        self.assertEqual(result["cycle_source"], "EPA_TESTCAR")
        self.assertIsNone(result["roadload_temperature_c"])
        self.assertFalse(result["provenance"]["run_schedule_fields_used"])

    def test_epa_cold_roadload_condition_requires_explicit_evidence(self):
        result = classify_epa_roadload_condition({
            "Roadload Condition": "Cold roadload",
            "Roadload Temperature (C)": -7,
        })
        self.assertEqual(result["cycle_name"], EPA_COLD)
        self.assertEqual(result["roadload_temperature_c"], -7.0)
        self.assertEqual(result["provenance"]["scalar_value_basis"], "SOURCE_DIRECT")

    def test_epa_custom_roadload_condition_preserves_direct_scalars(self):
        result = classify_epa_roadload_condition({
            "Road Load Condition Description": "High-altitude custom condition",
            "Road Load Temperature (C)": 35,
            "Road Load Ambient Pressure (kPa)": 82.5,
        })
        self.assertEqual(result["cycle_name"], EPA_CUSTOM)
        self.assertEqual(result["roadload_temperature_c"], 35.0)
        self.assertEqual(result["roadload_ambient_pressure_kpa"], 82.5)

    def test_vde_signature_changes_with_test_number_or_procedure(self):
        base = {field: None for field in EPA_VDE_CARRYOVER_FIELDS}
        base.update({"Test Number": "T-1", "Test Procedure Cd": 31})
        changed_number = dict(base, **{"Test Number": "T-2"})
        changed_procedure = dict(base, **{"Test Procedure Cd": 3})
        signature = group_signature([base], EPA_VDE_CARRYOVER_FIELDS)
        self.assertNotEqual(signature, group_signature([changed_number], EPA_VDE_CARRYOVER_FIELDS))
        self.assertNotEqual(signature, group_signature([changed_procedure], EPA_VDE_CARRYOVER_FIELDS))

    def test_approved_vehicle_configuration_scalar_conversions(self):
        self.assertTrue(math.isclose(horsepower_to_kw(300), 300 * HP_TO_KW, rel_tol=0, abs_tol=1e-12))
        self.assertEqual(cylinders_rotors(6.0), 6)
        self.assertIsNone(horsepower_to_kw(None))
        self.assertIsNone(cylinders_rotors(None))
        with self.assertRaises(ValueError):
            cylinders_rotors(5.5)

    def test_temporal_parent_is_immediate_previous_available_year(self):
        signature = exact_signature({"vehicle": "same", "abc": [1, 2, 3]})
        result = assign_temporal_parents([
            {"id": -1, "year": 2022, "signature": signature},
            {"id": -2, "year": 2024, "signature": signature},
            {"id": -3, "year": 2025, "signature": signature},
        ])
        self.assertEqual(result[-1]["status"], "ROOT")
        self.assertEqual(result[-2]["parent_id"], -1)
        self.assertEqual(result[-2]["parent_year"], 2022)
        self.assertEqual(result[-3]["parent_id"], -2)

    def test_changed_signature_never_links(self):
        result = assign_temporal_parents([
            {"id": 1, "year": 2022, "signature": exact_signature({"set_a": 13.03})},
            {"id": 2, "year": 2023, "signature": exact_signature({"set_a": 13.39})},
        ])
        self.assertIsNone(result[1]["parent_id"])
        self.assertIsNone(result[2]["parent_id"])

    def test_no_false_carryover_when_test_evidence_changes(self):
        first = exact_signature({"target_abc": [1, 2, 3], "procedure": "FTP", "set_abc": [4, 5, 6]})
        changed = exact_signature({"target_abc": [1, 2, 3], "procedure": "HWY", "set_abc": [4, 5, 6]})
        result = assign_temporal_parents([
            {"id": "2022", "year": 2022, "signature": first},
            {"id": "2023", "year": 2023, "signature": changed},
        ])
        self.assertIsNone(result["2023"]["parent_id"])

    def test_engineering_scenario_lineage_remains_distinct(self):
        payload = _proposal_row_payload(
            {"resolved_columns": {"baseline": {
                "selected_baseline_vde_id": 42, "legislation": "EPA", "category": "Car",
                "make": "TEST", "model": "Scenario", "year": 2026,
            }}},
            {"resolved_snapshot": {"mass_kg": 1000}, "abc_total": {"A": 1, "B": 2, "C": 3}},
            {}, final_name="Scenario", note_text="",
        )
        provenance = json.loads(payload["provenance_json"])
        self.assertEqual(payload["vde_id_parent"], 42)
        self.assertEqual(provenance["lineage_relation"], "ENGINEERING_SCENARIO")

    def test_ambiguous_same_year_signature_is_not_linked(self):
        signature = exact_signature({"same": True})
        result = assign_temporal_parents([
            {"id": 1, "year": 2022, "signature": signature},
            {"id": 2, "year": 2023, "signature": signature},
            {"id": 3, "year": 2023, "signature": signature},
        ])
        self.assertEqual(result[2]["status"], "AMBIGUOUS")
        self.assertEqual(result[3]["status"], "AMBIGUOUS")
        self.assertIsNone(result[2]["parent_id"])
        self.assertIsNone(result[3]["parent_id"])

    def test_run_signature_allows_same_test_number_and_execution_evidence(self):
        record = {
            "Represented Test Veh Make": "FORD",
            "Represented Test Veh Model": "Edge AWD",
            "Test Number": "KFMX10071543",
            "ADFE Test Number": "ADFE-1",
            "Actual Tested Testgroup": "KFMXT02.73HF",
            "Test Vehicle ID": "VEH-1",
            "Test Veh Configuration #": 0,
            "Test Procedure Cd": 21,
            "Set Coef A (lbf)": 10.0,
            "Set Coef B (lbf/mph)": 0.2,
            "Set Coef C (lbf/mph**2)": 0.01,
        }
        signature = run_carryover_signature(record, {"RND_ADJ_FE": 25.0})
        result = assign_temporal_parents([
            {"id": "MY2021", "year": 2021, "signature": signature},
            {"id": "MY2022", "year": 2022, "signature": signature},
        ])
        self.assertEqual(result["MY2022"]["parent_id"], "MY2021")

    def test_run_signature_blocks_different_test_number(self):
        base = {
            "Represented Test Veh Make": "FORD",
            "Represented Test Veh Model": "Edge AWD",
            "Test Number": "KFMX10067860",
            "Actual Tested Testgroup": "KFMXT02.73HF",
            "Set Coef A (lbf)": 10.0,
            "Set Coef B (lbf/mph)": 0.2,
            "Set Coef C (lbf/mph**2)": 0.01,
        }
        later = dict(base, **{"Test Number": "KFMX10071543"})
        result = assign_temporal_parents([
            {"id": "MY2021", "year": 2021, "signature": run_carryover_signature(base, {})},
            {"id": "MY2022", "year": 2022, "signature": run_carryover_signature(later, {})},
        ])
        self.assertIsNone(result["MY2022"]["parent_id"])

    def test_run_signature_blocks_different_adfe_test_number(self):
        base = {"Test Number": "T-1", "ADFE Test Number": "A-1"}
        later = {"Test Number": "T-1", "ADFE Test Number": "A-2"}
        self.assertNotEqual(
            run_carryover_signature(base, {}),
            run_carryover_signature(later, {}),
        )

    def test_fuelcons_run_evidence_signature_remains_unchanged_by_test_number(self):
        base = {"Test Number": "KFMX10067860", "ADFE Test Number": "ADFE-1"}
        later = {"Test Number": "KFMX10071543", "ADFE Test Number": "ADFE-2"}
        self.assertEqual(
            fuelcons_run_evidence_signature(base, {}),
            fuelcons_run_evidence_signature(later, {}),
        )

    def test_run_signature_keeps_cadillac_same_test_number_set_abc_variants_distinct(self):
        records = []
        expected = {}
        for variant, set_a in (("A", 12.1), ("B", 12.2), ("C", 12.3)):
            signature = run_carryover_signature({
                "Test Number": "NGMX91004749",
                "Actual Tested Testgroup": "NGMXT02.7119",
                "Set Coef A (lbf)": set_a,
                "Set Coef B (lbf/mph)": 0.1,
                "Set Coef C (lbf/mph**2)": 0.02,
            }, {})
            for year in (2022, 2023, 2024):
                identifier = f"{variant}-{year}"
                records.append({"id": identifier, "year": year, "signature": signature})
                expected[identifier] = None if year == 2022 else f"{variant}-{year - 1}"
        result = assign_temporal_parents(records)
        self.assertEqual({identifier: row["parent_id"] for identifier, row in result.items()}, expected)


class Sprint12ClosurePhase2CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[2] / "data" / "db" / "staging" / "eco_drive_canonical_candidate.db"
        if not cls.path.exists():
            raise unittest.SkipTest("canonical candidate has not been generated")
        cls.connection = sqlite3.connect(cls.path)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "connection"):
            cls.connection.close()

    def test_epa_vde_uses_roadload_family_while_runs_keep_schedules(self):
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM vde WHERE legislation='EPA' "
                "AND (cycle_name!='EPA NORMAL' OR cycle_source!='EPA_TESTCAR')"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM vde WHERE legislation='EPA' "
                "AND (roadload_temperature_c IS NOT NULL OR roadload_ambient_pressure_kpa IS NOT NULL)"
            ).fetchone()[0],
            0,
        )
        categories = {
            row[0] for row in self.connection.execute(
                "SELECT DISTINCT json_extract(result_details_json,'$.canonical_result.Test Category') "
                "FROM run WHERE vde_id=(SELECT id FROM vde WHERE source_record_id='PREVIEW-VDE-6068BFB3E6A607D9')"
            )
        }
        self.assertTrue({"FTP", "HWY", "SC03", "US06"} <= categories)

    def test_run_identity_review_reason_is_always_present(self):
        count, missing = self.connection.execute(
            "SELECT COUNT(*),SUM(CASE WHEN trim(COALESCE(json_extract(provenance_json,'$.run_identity_review_reason'),''))='' "
            "THEN 1 ELSE 0 END) FROM run WHERE review_status='RUN_IDENTITY_REVIEW'"
        ).fetchone()
        self.assertGreater(count, 0)
        self.assertEqual(missing, 0)

    def test_linked_runs_have_identical_test_number_evidence(self):
        mismatches = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM run child
            JOIN run parent
              ON parent.run_id=json_extract(child.provenance_json,'$.carryover_from_run_id')
            WHERE json_extract(child.provenance_json,'$.lineage_relation')='EPA_MODEL_YEAR_CARRYOVER'
              AND (
                json_extract(child.conditions_json,'$.test_number')
                  IS NOT json_extract(parent.conditions_json,'$.test_number')
                OR json_extract(child.conditions_json,'$.adfe_test_number')
                  IS NOT json_extract(parent.conditions_json,'$.adfe_test_number')
              )
            """
        ).fetchone()[0]
        self.assertEqual(mismatches, 0)

    def test_ford_edge_changed_test_number_run_is_not_linked(self):
        count = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM run child
            JOIN run parent
              ON parent.run_id=json_extract(child.provenance_json,'$.carryover_from_run_id')
            WHERE json_extract(child.provenance_json,'$.lineage_relation')='EPA_MODEL_YEAR_CARRYOVER'
              AND json_extract(child.conditions_json,'$.test_number')='KFMX10071543'
              AND json_extract(parent.conditions_json,'$.test_number')='KFMX10067860'
            """
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_linked_vdes_have_identical_test_number_and_procedure_evidence(self):
        mismatches = self.connection.execute(
            """
            SELECT COUNT(*) FROM vde child JOIN vde parent ON parent.id=child.vde_id_parent
            WHERE json_extract(child.provenance_json,'$.lineage_relation')='EPA_MODEL_YEAR_CARRYOVER'
              AND (
                (SELECT json_group_array(value) FROM (SELECT DISTINCT
                   json_extract(conditions_json,'$.test_number') value FROM run
                   WHERE vde_id=child.id ORDER BY value))
                !=
                (SELECT json_group_array(value) FROM (SELECT DISTINCT
                   json_extract(conditions_json,'$.test_number') value FROM run
                   WHERE vde_id=parent.id ORDER BY value))
                OR
                (SELECT json_group_array(value) FROM (SELECT DISTINCT
                   COALESCE(procedure_description,procedure_code) value FROM run
                   WHERE vde_id=child.id ORDER BY value))
                !=
                (SELECT json_group_array(value) FROM (SELECT DISTINCT
                   COALESCE(procedure_description,procedure_code) value FROM run
                   WHERE vde_id=parent.id ORDER BY value))
              )
            """
        ).fetchone()[0]
        self.assertEqual(mismatches, 0)

    def test_bmw_330i_annual_chain_is_preserved(self):
        rows = self.connection.execute(
            "SELECT id,year,vde_id_parent FROM vde WHERE source_record_id IN "
            "('PREVIEW-VDE-294263A2B23C4FC5','PREVIEW-VDE-ED8B31106ED070DF','PREVIEW-VDE-D38AB97C7F865288') "
            "ORDER BY year"
        ).fetchall()
        self.assertEqual([row[1] for row in rows], [2020, 2021, 2022])
        self.assertTrue(all(row[0] > 0 for row in rows))
        self.assertIsNone(rows[0][2])
        self.assertEqual(rows[1][2], rows[0][0])
        self.assertEqual(rows[2][2], rows[1][0])

    def test_carryover_relation_provenance_and_human_note(self):
        rows = self.connection.execute(
            "SELECT notes,json_extract(provenance_json,'$.lineage_relation') FROM vde "
            "WHERE source_record_id IN ('PREVIEW-VDE-ED8B31106ED070DF','PREVIEW-VDE-D38AB97C7F865288') ORDER BY year"
        ).fetchall()
        self.assertTrue(all(row[1] == "EPA_MODEL_YEAR_CARRYOVER" for row in rows))
        self.assertTrue(all("EPA model-year carryover from" in row[0] for row in rows))

    def test_fuelcons_annual_rows_are_preserved_without_reference_repurposing(self):
        rows = self.connection.execute(
            "SELECT vde_id,reference_fuelcons_id FROM fuelcons "
            "WHERE vde_id IN (SELECT id FROM vde WHERE source_record_id IN "
            "('PREVIEW-VDE-294263A2B23C4FC5','PREVIEW-VDE-ED8B31106ED070DF','PREVIEW-VDE-D38AB97C7F865288')) "
            "ORDER BY vde_id"
        ).fetchall()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row[1] is None for row in rows))

    def test_compatibility_views_remain_usable(self):
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM vde_db").fetchone()[0], 11626)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM fuelcons_db").fetchone()[0], 10822)

    def test_cadillac_set_abc_variants_are_distinct_reviewed_chains(self):
        rows = self.connection.execute(
            """
            SELECT run_id,v.year,
                   json_extract(r.conditions_json,'$.set_abc_native'),
                   json_extract(r.provenance_json,'$.carryover_from_run_id'),
                   r.review_status
            FROM run r JOIN vde v ON v.id=r.vde_id
            WHERE json_extract(r.conditions_json,'$.test_number')='NGMX91004749'
            ORDER BY 3,v.year
            """
        ).fetchall()
        self.assertEqual(len(rows), 15)
        self.assertEqual(len({row[2] for row in rows}), 3)
        self.assertTrue(all(row[4] == "RUN_IDENTITY_REVIEW" for row in rows))
        for variant in {row[2] for row in rows}:
            chain = [row for row in rows if row[2] == variant]
            self.assertEqual([row[1] for row in chain], [2022, 2023, 2024, 2025, 2026])
            self.assertIsNone(chain[0][3])
            self.assertEqual([row[3] for row in chain[1:]], [row[0] for row in chain[:-1]])

    def test_audi_r8_distinct_roadloads_remain_distinct(self):
        rows = self.connection.execute(
            "SELECT coast_A_N,coast_B_N_per_kph,coast_C_N_per_kph2 FROM vde "
            "WHERE source_record_id IN "
            "('PREVIEW-VDE-FD18F8B48AC283BC','PREVIEW-VDE-D29CDB3AF3EB52AB','PREVIEW-VDE-606462735D0D828F')"
        ).fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(set(rows)), 3)

    def test_scalars_and_non_invention_boundaries(self):
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM vehicle_configuration WHERE engine_rated_power_kw IS NOT NULL").fetchone()[0], 9962)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM vehicle_configuration WHERE engine_cylinders_rotors IS NOT NULL").fetchone()[0], 8641)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM component_db").fetchone()[0], 0)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM component_instance").fetchone()[0], 843)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM component_resolution").fetchone()[0], 0)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM tire_db").fetchone()[0], 1)
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND reference_fuelcons_id IS NOT NULL"
            ).fetchone()[0],
            0,
        )


if __name__ == "__main__":
    unittest.main()
