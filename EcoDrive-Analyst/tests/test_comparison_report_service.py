from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import (
    ComparisonRole,
    LineageStatus,
    RevisionStatus,
    SourceKind,
    TransmissionStatus,
    build_comparison_dataset,
    build_scenario_comparison_item,
    build_vde_comparison_item,
    resolve_cycle_vde_results,
    resolve_roadload_boundaries,
    resolve_temporary_transmission_from_component,
    resolve_transmission_boundary,
    resolve_vde_aggregate,
)
from src.vde_core.qa_mock_data import build_vde_seed_rows, seed_qa_database


def _qa_row(vde_id: int) -> dict:
    return dict({r["id"]: r for r in build_vde_seed_rows()}[vde_id])


class TransmissionAndRoadloadTests(unittest.TestCase):
    def test_stored_transmission_available_when_all_three_present(self):
        row = _qa_row(900001)
        resolution = resolve_transmission_boundary(row)
        self.assertEqual(resolution.status, TransmissionStatus.AVAILABLE)
        self.assertEqual(resolution.source, "STORED")
        self.assertEqual((resolution.abc.A, resolution.abc.B, resolution.abc.C), (8.5, 0.004, 0.0008))
        self.assertEqual(resolution.warnings, ())

    def test_missing_transmission_is_missing_not_zero(self):
        row = _qa_row(900006)
        self.assertIsNone(row["trans_A_coef_N"])
        resolution = resolve_transmission_boundary(row)
        self.assertEqual(resolution.status, TransmissionStatus.MISSING)
        self.assertIsNone(resolution.abc)
        self.assertIn("transmission_missing", resolution.warnings)

    def test_zero_transmission_coefficients_are_available_not_missing(self):
        row = _qa_row(900006)
        row["trans_A_coef_N"] = 0.0
        row["trans_B_coef_Npkph"] = 0.0
        row["trans_C_coef_Npkph2"] = 0.0
        resolution = resolve_transmission_boundary(row)
        self.assertEqual(resolution.status, TransmissionStatus.AVAILABLE)

    def test_temporary_transmission_used_only_when_stored_missing(self):
        row = _qa_row(900006)
        temp = {"source": "MANUAL", "A": 9.0, "B": 0.003, "C": 0.0006}
        resolution = resolve_transmission_boundary(row, temp)
        self.assertEqual(resolution.status, TransmissionStatus.TEMPORARY)
        self.assertEqual(resolution.source, "MANUAL")
        self.assertIn("temporary_transmission_used", resolution.warnings)

    def test_temporary_transmission_ignored_when_stored_available(self):
        row = _qa_row(900001)
        temp = {"source": "MANUAL", "A": 99.0, "B": 9.0, "C": 9.0}
        resolution = resolve_transmission_boundary(row, temp)
        self.assertEqual(resolution.status, TransmissionStatus.AVAILABLE)
        self.assertEqual(resolution.abc.A, 8.5)

    def test_roadload_total_only_row_has_no_fake_net(self):
        row = _qa_row(900006)
        boundaries = resolve_roadload_boundaries(row)
        self.assertTrue(boundaries["total"].available)
        self.assertFalse(boundaries["net"].available)

    def test_roadload_net_is_total_minus_transmission(self):
        row = _qa_row(900001)
        boundaries = resolve_roadload_boundaries(row)
        self.assertTrue(boundaries["net"].available)
        self.assertAlmostEqual(boundaries["net"].A, 118.0 - 8.5)
        self.assertAlmostEqual(boundaries["net"].B, 0.0200 - 0.0040)
        self.assertAlmostEqual(boundaries["net"].C, 0.0090 - 0.0008)

    def test_roadload_never_derives_total_from_net(self):
        row = _qa_row(900006)
        row["coast_A_N"] = None
        row["coast_B_N_per_kph"] = None
        row["coast_C_N_per_kph2"] = None
        row["vde_net_mj_per_km"] = 0.9
        boundaries = resolve_roadload_boundaries(row)
        self.assertFalse(boundaries["total"].available)
        self.assertFalse(boundaries["net"].available)


class TemporaryTransmissionFromComponentTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "qa_component.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_component_lookup_returns_canonical_abc_without_db_mutation(self):
        before = db_module.fetchall("SELECT * FROM component_db WHERE component_code='TRANS-MOCK-001'")
        temp = resolve_temporary_transmission_from_component("TRANS-MOCK-001")
        after = db_module.fetchall("SELECT * FROM component_db WHERE component_code='TRANS-MOCK-001'")

        self.assertIsNotNone(temp)
        self.assertEqual((temp["A"], temp["B"], temp["C"]), (12.0, 0.0050, 0.0009))
        self.assertEqual(temp["source"], "COMPONENT:TRANS-MOCK-001")
        self.assertEqual(before, after)

    def test_unknown_component_returns_none(self):
        self.assertIsNone(resolve_temporary_transmission_from_component("NOT-A-REAL-COMPONENT"))

    def test_temporary_transmission_via_component_yields_net_result(self):
        row = _qa_row(900006)
        temp = resolve_temporary_transmission_from_component("TRANS-MOCK-001")
        results = resolve_cycle_vde_results(row, temp)
        self.assertIsNotNone(results["net"].aggregate)
        self.assertIn("temporary_transmission_used", results["net"].warnings)


class CycleVdeResultsTests(unittest.TestCase):
    def test_epa_total_and_net_both_available_and_net_lower(self):
        row = _qa_row(900001)
        results = resolve_cycle_vde_results(row)
        self.assertIsNotNone(results["total"].aggregate)
        self.assertIsNotNone(results["net"].aggregate)
        self.assertLess(results["net"].aggregate, results["total"].aggregate)
        self.assertEqual(results["total"].boundary, "TOTAL")
        self.assertEqual(results["net"].boundary, "NET")

    def test_missing_transmission_leaves_net_unavailable_no_fake_value(self):
        row = _qa_row(900006)
        results = resolve_cycle_vde_results(row)
        self.assertIsNotNone(results["total"].aggregate)
        self.assertIsNone(results["net"].aggregate)
        self.assertIn("transmission_missing", results["net"].warnings)

    def test_unsupported_legislation_is_unavailable_not_invented(self):
        row = _qa_row(900001)
        row["legislation"] = "CN"
        results = resolve_cycle_vde_results(row)
        self.assertIsNone(results["total"].aggregate)
        self.assertIn("cycle_trace_unavailable", results["total"].warnings)

    def test_legacy_phase_columns_are_not_trusted_recomputation_proves_source(self):
        row = _qa_row(900001)
        # Poison the historical phase columns with an obviously-wrong value;
        # the new path must not read them.
        row["vde_urb_mj_per_km"] = -999.0
        row["vde_hw_mj_per_km"] = -999.0
        results = resolve_cycle_vde_results(row)
        self.assertGreater(results["total"].aggregate, 0)
        self.assertEqual(results["total"].source, "on_demand_calculation")

    def test_wltp_phase_results_available_from_synthetic_row(self):
        row = _qa_row(900001)
        row["legislation"] = "WLTP"
        results = resolve_cycle_vde_results(row)
        self.assertIsNotNone(results["total"].aggregate)
        self.assertTrue(set(results["total"].by_phase) & {"low", "mid", "high", "xhigh"})


class VdeAggregateTests(unittest.TestCase):
    def test_stored_total_and_net_available(self):
        row = _qa_row(900001)
        aggregate = resolve_vde_aggregate(row)
        self.assertEqual(aggregate["total"], 1.24)
        self.assertEqual(aggregate["net"], 1.18)
        self.assertEqual(aggregate["net_source"], "stored")

    def test_missing_stored_net_falls_back_to_on_demand_when_cycle_results_given(self):
        row = _qa_row(900001)
        row["vde_net_mj_per_km"] = None
        cycle_results = resolve_cycle_vde_results(row)
        aggregate = resolve_vde_aggregate(row, cycle_results=cycle_results)
        self.assertEqual(aggregate["total"], 1.24)
        self.assertIsNotNone(aggregate["net"])
        self.assertEqual(aggregate["net_source"], "on_demand")

    def test_missing_stored_net_without_cycle_results_stays_unavailable(self):
        row = _qa_row(900001)
        row["vde_net_mj_per_km"] = None
        aggregate = resolve_vde_aggregate(row)
        self.assertIsNone(aggregate["net"])
        self.assertIsNone(aggregate["net_source"])


class ItemBuilderTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "item_builder.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        self._insert_fuelcons_rows()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _insert_fuelcons_rows(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO fuelcons_db (
                    id, vde_id, electrification, fuel_type, record_origin,
                    engine_method, engine_version, source_vde_revision,
                    fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km
                ) VALUES (1, 900001, 'ICE', 'Gasoline', 'ESTIMATED', 'physics_simple', 'v1',
                          '2026-07-16T00:00:00Z', 6.5, 15.4, NULL, 150.0)
                """
            )
            con.execute(
                """
                INSERT INTO fuelcons_db (
                    id, vde_id, electrification, fuel_type, record_origin,
                    engine_method, engine_version, source_vde_revision,
                    fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km
                ) VALUES (2, 900001, 'ICE', 'Gasoline', 'HOMOLOGATED', 'label', 'v1',
                          '2020-01-01T00:00:00Z', 6.9, 14.5, NULL, 160.0)
                """
            )
            con.execute(
                """
                INSERT INTO fuelcons_db (
                    id, vde_id, electrification, fuel_type, record_origin,
                    engine_method, engine_version, source_vde_revision,
                    fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km
                ) VALUES (3, 900002, 'ICE', 'Gasoline', 'SCENARIO' , 'physics_simple', 'v1', NULL,
                          7.1, 14.1, NULL, 165.0)
                """
            )
            con.commit()

    def test_scenario_item_resolves_linked_vde(self):
        item = build_scenario_comparison_item(1)
        self.assertEqual(item.source_kind, SourceKind.FUELCONS_SCENARIO)
        self.assertEqual(item.vde_id, 900001)
        self.assertEqual(item.fuelcons_id, 1)
        self.assertEqual(item.fuel_energy["fuel_l_per_100km"], 6.5)
        self.assertEqual(item.vde["aggregate"]["total"], 1.24)

    def test_vde_only_item_is_valid_without_fuelcons(self):
        item = build_vde_comparison_item(900006)
        self.assertEqual(item.source_kind, SourceKind.VDE_ONLY)
        self.assertIsNone(item.fuelcons_id)
        self.assertIsNone(item.fuel_energy)
        self.assertNotIn("FUEL_CONSUMPTION", item.availability)

    def test_multiple_scenarios_sharing_one_vde_stay_distinct(self):
        item1 = build_scenario_comparison_item(1)
        item2 = build_scenario_comparison_item(2)
        self.assertEqual(item1.vde_id, item2.vde_id)
        self.assertNotEqual(item1.fuelcons_id, item2.fuelcons_id)
        self.assertEqual(item1.provenance.record_origin, "ESTIMATED")
        self.assertEqual(item2.provenance.record_origin, "HOMOLOGATED")

    def test_reference_role_does_not_mutate_source_row(self):
        before = db_module.fetchone("SELECT * FROM vde_db WHERE id=900001")
        build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE)
        after = db_module.fetchone("SELECT * FROM vde_db WHERE id=900001")
        self.assertEqual(before, after)

    def test_any_valid_item_can_be_reference(self):
        scenario_ref = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        vde_ref = build_vde_comparison_item(900006, role=ComparisonRole.REFERENCE)
        self.assertEqual(scenario_ref.role, ComparisonRole.REFERENCE)
        self.assertEqual(vde_ref.role, ComparisonRole.REFERENCE)

    def test_dataset_batches_vde_fetch_for_vde_only_specs(self):
        dataset = build_comparison_dataset(
            {"kind": "VDE_ONLY", "vde_id": 900001},
            [
                {"kind": "VDE_ONLY", "vde_id": 900002},
                {"kind": "VDE_ONLY", "vde_id": 900003},
                {"kind": "FUELCONS_SCENARIO", "fuelcons_id": 1},
            ],
        )
        self.assertEqual(dataset.reference.vde_id, 900001)
        self.assertEqual({c.source_kind for c in dataset.comparisons}, {SourceKind.VDE_ONLY, SourceKind.FUELCONS_SCENARIO})
        self.assertEqual(len(dataset.comparisons), 3)

    def test_mixed_provenance_dataset_builds_without_rejecting_any_item(self):
        dataset = build_comparison_dataset(
            {"kind": "FUELCONS_SCENARIO", "fuelcons_id": 2},
            [
                {"kind": "FUELCONS_SCENARIO", "fuelcons_id": 1},
                {"kind": "FUELCONS_SCENARIO", "fuelcons_id": 3},
                {"kind": "VDE_ONLY", "vde_id": 900006},
            ],
        )
        origins = {dataset.reference.provenance.record_origin} | {
            c.provenance.record_origin for c in dataset.comparisons
        }
        self.assertEqual(origins, {"HOMOLOGATED", "ESTIMATED", "SCENARIO", "IMPORTED_REFERENCE"})


class StaleSourceTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "stale.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _insert(self, row_id: int, source_vde_revision):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, record_origin, source_vde_revision) "
                "VALUES (?, 900001, 'ICE', 'ESTIMATED', ?)",
                (row_id, source_vde_revision),
            )
            con.commit()

    def test_matching_revision_is_current(self):
        current = db_module.fetchone("SELECT updated_at, created_at FROM vde_db WHERE id=900001")
        revision = current.get("updated_at") or current.get("created_at")
        self._insert(10, revision)
        item = build_scenario_comparison_item(10)
        self.assertEqual(item.provenance.revision_status, RevisionStatus.CURRENT)

    def test_different_revision_is_stale_and_still_builds(self):
        self._insert(11, "2000-01-01T00:00:00Z")
        item = build_scenario_comparison_item(11)
        self.assertEqual(item.provenance.revision_status, RevisionStatus.STALE)
        self.assertIn("fuelcons_source_stale", item.warnings)

    def test_missing_source_revision_is_missing_status(self):
        self._insert(12, None)
        item = build_scenario_comparison_item(12)
        self.assertEqual(item.provenance.revision_status, RevisionStatus.MISSING)


class LineageTests(unittest.TestCase):
    def test_explicit_lineage_when_parent_populated(self):
        row = _qa_row(900001)
        row["vde_id_parent"] = 900000
        item = build_vde_comparison_item(900001, vde_row=row)
        self.assertEqual(item.lineage["lineage_status"], LineageStatus.EXPLICIT.value)
        self.assertEqual(item.lineage["parent_id"], 900000)

    def test_no_lineage_when_parent_absent(self):
        row = _qa_row(900001)
        row["vde_id_parent"] = None
        item = build_vde_comparison_item(900001, vde_row=row)
        self.assertEqual(item.lineage["lineage_status"], LineageStatus.NONE.value)
        self.assertIsNone(item.lineage["parent_id"])


class DeterminismTests(unittest.TestCase):
    def test_building_same_item_twice_is_equal(self):
        row = _qa_row(900001)
        item1 = build_vde_comparison_item(900001, vde_row=row)
        item2 = build_vde_comparison_item(900001, vde_row=row)
        self.assertEqual(item1, item2)


class ComparisonReportPageSmokeTests(unittest.TestCase):
    """comparison_report_service.py grew from 3 unused helpers to the full
    Package 8A API in one change. The old Comparison Report page never
    imports this module, but this guards against a future accidental
    circular import or runtime side effect as later packages build on it.
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "comparison_page_smoke.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_existing_comparison_report_page_still_opens_without_exception(self):
        from streamlit.testing.v1 import AppTest

        page_path = Path(__file__).resolve().parents[1] / "pages" / "Comparison_Report.py"
        app = AppTest.from_file(str(page_path))
        app.session_state["ctx"] = {"db_path": str(self.db_path)}
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
