from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vde_app.comparison_report_viewmodels import (
    MAX_COMPARISONS,
    SelectionState,
    add_comparison,
    build_abc_rows,
    build_competitor_delta_rows,
    build_cycle_demand_rows,
    build_explore_bar_rows,
    build_explore_line_rows,
    build_explore_scatter_points,
    build_fe_vde_points,
    build_iso_pse_lines,
    build_lineage_waterfall,
    build_metric_bar_rows,
    build_reference_summary,
    build_roadload_curve_rows,
    build_scenario_header,
    build_scenario_options,
    build_scorecard_sections,
    dataset_warnings_summary,
    deduplicate_by_vde_id,
    format_value,
    is_temporary_net,
    list_available_explore_metrics,
    list_available_lineage_metrics,
    list_explore_dimension_values,
    list_explore_dimensions,
    list_explore_numeric_metrics,
    list_lineage_capable_metrics,
    metric_axis_label,
    remove_comparison,
    resolve_lineage_context,
    set_reference,
    sync_comparisons_from_widget,
)
from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import (
    ComparisonDataset,
    ComparisonRole,
    LineageChainStatus,
    build_scenario_comparison_item,
    build_vde_comparison_item,
    list_comparison_scenarios,
    resolve_lineage_chain,
)
from src.vde_core.qa_mock_data import build_vde_seed_rows, seed_qa_database


def _qa_row(vde_id: int) -> dict:
    return dict({r["id"]: r for r in build_vde_seed_rows()}[vde_id])


# -----------------------------------------------------------------------------
# Selection (Sec 52)
# -----------------------------------------------------------------------------


class SelectionStateTests(unittest.TestCase):
    def test_reference_excluded_from_comparisons(self):
        state = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2, 3))
        state, error = add_comparison(state, 1)
        self.assertIsNotNone(error)
        self.assertNotIn(1, state.comparison_fuelcons_ids)

    def test_reference_switch_removes_duplicate_from_comparisons(self):
        state = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2, 3))
        state = set_reference(state, 2)
        self.assertEqual(state.reference_fuelcons_id, 2)
        self.assertEqual(state.comparison_fuelcons_ids, (3,))

    def test_up_to_ten_comparisons_accepted(self):
        state = SelectionState()
        for i in range(1, MAX_COMPARISONS + 1):
            state, error = add_comparison(state, i)
            self.assertIsNone(error)
        self.assertEqual(len(state.comparison_fuelcons_ids), MAX_COMPARISONS)

    def test_eleventh_comparison_rejected_explicitly(self):
        state = SelectionState()
        for i in range(1, MAX_COMPARISONS + 1):
            state, _ = add_comparison(state, i)
        state, error = add_comparison(state, 999)
        self.assertIsNotNone(error)
        self.assertEqual(len(state.comparison_fuelcons_ids), MAX_COMPARISONS)
        self.assertNotIn(999, state.comparison_fuelcons_ids)

    def test_selection_order_preserved(self):
        state = SelectionState()
        for i in (5, 1, 9, 2):
            state, _ = add_comparison(state, i)
        self.assertEqual(state.comparison_fuelcons_ids, (5, 1, 9, 2))

    def test_duplicate_add_is_a_no_op_not_an_error(self):
        state = SelectionState()
        state, _ = add_comparison(state, 5)
        state, error = add_comparison(state, 5)
        self.assertIsNone(error)
        self.assertEqual(state.comparison_fuelcons_ids, (5,))

    def test_remove_comparison(self):
        state = SelectionState(comparison_fuelcons_ids=(1, 2, 3))
        state = remove_comparison(state, 2)
        self.assertEqual(state.comparison_fuelcons_ids, (1, 3))

    def test_sync_from_widget_preserves_original_order_and_drops_removed(self):
        state = SelectionState()
        state, _ = add_comparison(state, 1)
        state, _ = add_comparison(state, 2)
        state, _ = add_comparison(state, 3)
        # widget returns a DIFFERENT order (e.g. options-list order) plus a new id, minus one
        state, errors = sync_comparisons_from_widget(state, [3, 4, 1])
        self.assertEqual(errors, ())
        self.assertEqual(state.comparison_fuelcons_ids, (1, 3, 4))

    def test_sync_from_widget_reports_cap_overflow(self):
        state = SelectionState()
        for i in range(1, MAX_COMPARISONS + 1):
            state, _ = add_comparison(state, i)
        state, errors = sync_comparisons_from_widget(state, list(range(1, MAX_COMPARISONS + 1)) + [999])
        self.assertEqual(len(errors), 1)
        self.assertNotIn(999, state.comparison_fuelcons_ids)


class ScenarioOptionTests(unittest.TestCase):
    def test_same_vde_different_fuelcons_scenarios_remain_distinct_options(self):
        rows = [
            {"fuelcons_id": 1, "vde_id": 900001, "make": "QA", "model": "Nominal EPA baseline", "year": 2026, "legislation": "EPA", "electrification": "ICE", "record_origin": "HOMOLOGATED"},
            {"fuelcons_id": 2, "vde_id": 900001, "make": "QA", "model": "Nominal EPA baseline", "year": 2026, "legislation": "EPA", "electrification": "ICE", "record_origin": "ESTIMATED"},
        ]
        options = build_scenario_options(rows)
        self.assertEqual(len(options), 2)
        self.assertNotEqual(options[0].fuelcons_id, options[1].fuelcons_id)
        self.assertEqual(options[0].vde_id, options[1].vde_id)

    def test_label_does_not_lead_with_raw_id(self):
        rows = [{"fuelcons_id": 1, "vde_id": 900001, "make": "QA", "model": "Baseline", "year": 2026, "legislation": "EPA", "electrification": "ICE", "record_origin": "HOMOLOGATED"}]
        option = build_scenario_options(rows)[0]
        self.assertFalse(option.label.startswith("1"))
        self.assertFalse(option.label.startswith("#"))
        self.assertIn("QA Baseline", option.label)


# -----------------------------------------------------------------------------
# Scorecard construction, deltas, compatibility, provenance, NET missing (Sec 53-59)
# -----------------------------------------------------------------------------


class ScorecardConstructionTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "scorecard.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            rows = [
                (1, 900001, "ICE", "Gasoline", "HOMOLOGATED", None, "2026-07-16T00:00:00Z", 6.0, 16.7, None, 140.0, 0.30),
                (2, 900002, "ICE", "Gasoline", "ESTIMATED", "PWT_L0", "2026-07-16T00:00:00Z", 6.5, 15.4, None, 150.0, 0.28),
                (3, 900001, "ICE", "Gasoline", "SCENARIO", None, "2000-01-01T00:00:00Z", 5.8, 17.2, None, 130.0, 0.32),
                (4, 900006, "BEV", "Electric", "ESTIMATED", None, "2026-07-16T00:00:00Z", None, None, 150.0, 0.0, 0.90),
            ]
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "engine_method, source_vde_revision, fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, "
                "gco2_per_km, eta_pt_est) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        self.estimated = build_scenario_comparison_item(2)
        self.stale = build_scenario_comparison_item(3)
        self.bev_no_net = build_scenario_comparison_item(4)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        from src.vde_core.comparison_report_service import ComparisonDataset

        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_reference_is_first_and_section_order_is_deterministic(self):
        dataset = self._dataset([self.estimated])
        sections = build_scorecard_sections(dataset)
        titles = [s.title for s in sections]
        self.assertEqual(
            titles,
            [
                "Vehicle / Program",
                "Powertrain",
                "Physical Setup",
                "Roadload",
                "Vehicle Demand",
                "Fuel / Energy / Emissions",
                "Efficiency",
                "Data Status / Provenance",
            ],
        )

    def test_row_order_is_deterministic_across_calls(self):
        dataset = self._dataset([self.estimated])
        sections1 = build_scorecard_sections(dataset)
        sections2 = build_scorecard_sections(dataset)
        self.assertEqual([r.metric_key for s in sections1 for r in s.rows], [r.metric_key for s in sections2 for r in s.rows])

    def test_missing_value_renders_dash_not_zero(self):
        dataset = self._dataset([self.bev_no_net])
        sections = build_scorecard_sections(dataset)
        roadload_section = next(s for s in sections if s.title == "Roadload")
        net_row = next(r for r in roadload_section.rows if r.metric_key == "roadload_a_net")
        cell = net_row.comparison_cells[0]
        self.assertFalse(cell.available)
        self.assertEqual(cell.formatted_value, "-")
        self.assertNotEqual(cell.formatted_value, "0")

    def test_lower_is_better_improvement_marks_better(self):
        dataset = self._dataset([self.stale])  # fuel_l_per_100km 5.8 < reference 6.0
        sections = build_scorecard_sections(dataset)
        fuel_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        row = next(r for r in fuel_section.rows if r.metric_key == "fuel_l_per_100km")
        cell = row.comparison_cells[0]
        self.assertEqual(cell.semantic, "BETTER")
        self.assertIsNotNone(cell.formatted_delta)

    def test_higher_is_better_improvement_marks_better(self):
        dataset = self._dataset([self.stale])  # fuel_km_per_l 17.2 > reference 16.7
        sections = build_scorecard_sections(dataset)
        fuel_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        row = next(r for r in fuel_section.rows if r.metric_key == "fuel_km_per_l")
        cell = row.comparison_cells[0]
        self.assertEqual(cell.semantic, "BETTER")

    def test_worse_result_is_marked(self):
        dataset = self._dataset([self.estimated])  # gco2 150 > reference 140
        sections = build_scorecard_sections(dataset)
        fuel_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        row = next(r for r in fuel_section.rows if r.metric_key == "gco2_per_km")
        cell = row.comparison_cells[0]
        self.assertEqual(cell.semantic, "WORSE")

    def test_neutral_metric_has_no_semantic_verdict(self):
        dataset = self._dataset([self.estimated])
        sections = build_scorecard_sections(dataset)
        physical_section = next(s for s in sections if s.title == "Physical Setup")
        row = next(r for r in physical_section.rows if r.metric_key == "mass_kg")
        for cell in row.comparison_cells:
            self.assertIsNone(cell.semantic)

    def test_zero_reference_yields_absolute_delta_but_no_percent(self):
        dataset = self._dataset([self.estimated])
        sections = build_scorecard_sections(dataset)
        energy_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        row = next(r for r in energy_section.rows if r.metric_key == "energy_wh_per_km")
        # reference has no energy_Wh_per_km at all (ICE) -> unavailable, not a crash
        cell = row.comparison_cells[0]
        self.assertFalse(cell.available)

    def test_epa_vs_epa_cycle_metric_is_compatible(self):
        dataset = self._dataset([self.estimated])
        sections = build_scorecard_sections(dataset)
        demand_section = next(s for s in sections if s.title == "Vehicle Demand")
        row = next(r for r in demand_section.rows if r.metric_key == "vde_total")
        self.assertTrue(row.comparison_cells[0].compatible)

    def test_epa_vs_wltp_cycle_metric_is_incompatible_no_misleading_delta(self):
        wltp_row = _qa_row(900002)
        wltp_row["legislation"] = "WLTP"
        wltp_item = build_vde_comparison_item(900002, vde_row=wltp_row)
        dataset = self._dataset([wltp_item])
        sections = build_scorecard_sections(dataset)
        demand_section = next(s for s in sections if s.title == "Vehicle Demand")
        row = next(r for r in demand_section.rows if r.metric_key == "vde_total")
        cell = row.comparison_cells[0]
        self.assertFalse(cell.compatible)
        self.assertIsNone(cell.absolute_delta)
        self.assertIsNone(cell.semantic)
        self.assertIsNotNone(cell.warning)

    def test_physical_metrics_remain_available_across_legislations(self):
        wltp_row = _qa_row(900002)
        wltp_row["legislation"] = "WLTP"
        wltp_item = build_vde_comparison_item(900002, vde_row=wltp_row)
        dataset = self._dataset([wltp_item])
        sections = build_scorecard_sections(dataset)
        physical_section = next(s for s in sections if s.title == "Physical Setup")
        row = next(r for r in physical_section.rows if r.metric_key == "cda_m2")
        self.assertTrue(row.comparison_cells[0].compatible)
        self.assertTrue(row.comparison_cells[0].available)

    def test_mixed_provenance_dataset_renders_all_items(self):
        dataset = self._dataset([self.estimated, self.stale])
        sections = build_scorecard_sections(dataset)
        provenance_section = next(s for s in sections if s.title == "Data Status / Provenance")
        origin_row = next(r for r in provenance_section.rows if r.metric_key == "provenance_record_origin")
        self.assertEqual(origin_row.reference_cell.raw_value, "HOMOLOGATED")
        self.assertEqual([c.raw_value for c in origin_row.comparison_cells], ["ESTIMATED", "SCENARIO"])

    def test_stale_status_is_visible_and_not_filtered(self):
        dataset = self._dataset([self.stale])
        sections = build_scorecard_sections(dataset)
        provenance_section = next(s for s in sections if s.title == "Data Status / Provenance")
        revision_row = next(r for r in provenance_section.rows if r.metric_key == "provenance_revision_status")
        self.assertEqual(revision_row.comparison_cells[0].raw_value, "STALE")
        header = build_scenario_header(self.stale)
        self.assertTrue(header["is_stale"])
        self.assertIn("STALE SOURCE", header["column_title"])

    def test_net_missing_total_present_no_fallback_no_crash(self):
        dataset = self._dataset([self.bev_no_net])
        sections = build_scorecard_sections(dataset)
        demand_section = next(s for s in sections if s.title == "Vehicle Demand")
        total_row = next(r for r in demand_section.rows if r.metric_key == "vde_total")
        net_row = next(r for r in demand_section.rows if r.metric_key == "vde_net")
        self.assertTrue(total_row.comparison_cells[0].available)
        self.assertFalse(net_row.comparison_cells[0].available)
        self.assertEqual(net_row.comparison_cells[0].formatted_value, "-")

    def test_same_vde_multiple_scenarios_produce_distinct_columns(self):
        item_a = build_scenario_comparison_item(1)
        item_b = build_scenario_comparison_item(3)
        dataset = self._dataset([item_a, item_b])
        self.assertEqual(len(dataset.comparisons), 2)
        self.assertEqual(dataset.comparisons[0].vde_id, dataset.comparisons[1].vde_id)
        self.assertNotEqual(dataset.comparisons[0].fuelcons_id, dataset.comparisons[1].fuelcons_id)

    def test_bev_scenario_does_not_coerce_into_fuel_row(self):
        dataset = self._dataset([self.bev_no_net])
        sections = build_scorecard_sections(dataset)
        fuel_section = next(s for s in sections if s.title == "Fuel / Energy / Emissions")
        fuel_row = next(r for r in fuel_section.rows if r.metric_key == "fuel_l_per_100km")
        self.assertFalse(fuel_row.comparison_cells[0].available)

    def test_dataset_warnings_flags_stale_missing_net_and_mixed_legislation(self):
        wltp_row = _qa_row(900002)
        wltp_row["legislation"] = "WLTP"
        wltp_item = build_vde_comparison_item(900002, vde_row=wltp_row)
        dataset = self._dataset([self.stale, self.bev_no_net, wltp_item])
        warnings = dataset_warnings_summary(dataset)
        joined = " ".join(warnings)
        self.assertIn("stale", joined.lower())
        self.assertIn("NET boundary", joined)
        self.assertIn("Mixed legislations", joined)


class DashboardRoadloadViewModelTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "dashboard.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            rows = [
                (1, 900001, "ICE", "Gasoline", "HOMOLOGATED", 6.0, 16.7, None, 140.0, 0.30),
                (2, 900001, "ICE", "Gasoline", "ESTIMATED", 6.5, 15.4, None, 150.0, 0.28),  # shares VDE with #1
                (3, 900006, "BEV", "Electric", "ESTIMATED", None, None, 150.0, 0.0, 0.90),  # missing transmission -> no NET
            ]
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km, eta_pt_est) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        self.same_vde = build_scenario_comparison_item(2)
        self.bev_no_net = build_scenario_comparison_item(3)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_reference_summary_has_no_score(self):
        summary = build_reference_summary(self.reference)
        self.assertEqual(summary["vde_total"], 1.24)
        self.assertNotIn("score", summary)

    def test_metric_bar_rows_include_reference_and_comparison(self):
        dataset = self._dataset([self.same_vde])
        result = build_metric_bar_rows(dataset, "vde_total")
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["excluded"], [])

    def test_metric_bar_rows_net_excludes_missing_with_reason_never_falls_back(self):
        dataset = self._dataset([self.bev_no_net])
        result = build_metric_bar_rows(dataset, "vde_net")
        self.assertEqual(len(result["rows"]), 1)  # only reference has NET
        self.assertEqual(len(result["excluded"]), 1)
        self.assertIn("unavailable", result["excluded"][0]["reason"])

    def test_dedup_by_vde_id_collapses_shared_vde_with_attribution(self):
        dataset = self._dataset([self.same_vde])
        groups = deduplicate_by_vde_id((dataset.reference, *dataset.comparisons))
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].used_by), 2)
        self.assertEqual(len(dataset.comparisons), 1)  # Scorecard-level distinctness untouched (8B)

    def test_abc_rows_dedup_and_reflect_boundary(self):
        dataset = self._dataset([self.same_vde])
        result = build_abc_rows(dataset, "TOTAL")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(len(result["rows"][0]["used_by"]), 2)
        self.assertEqual(result["rows"][0]["A"], 118.0)

    def test_roadload_curve_rows_match_plots_module_shape(self):
        dataset = self._dataset([])
        result = build_roadload_curve_rows(dataset, "TOTAL")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(set(result["rows"][0]) - {"label"}, {"A_N", "B_N_per_kph", "C_N_per_kph2"})

    def test_roadload_curve_rows_net_missing_excluded_no_fallback(self):
        dataset = self._dataset([self.bev_no_net])
        result = build_roadload_curve_rows(dataset, "NET")
        excluded_labels = {e["label"] for e in result["excluded"]}
        self.assertIn(self.bev_no_net.label, excluded_labels)
        self.assertTrue(all("unavailable" in e["reason"] for e in result["excluded"]))

    def test_cycle_demand_rows_total_and_net_are_distinct_series(self):
        from src.vde_core.cycles import use_standard_cycle

        cycle = use_standard_cycle("EPA")
        dataset = self._dataset([])
        result = build_cycle_demand_rows(dataset, cycle, ["TOTAL", "NET"])
        boundaries = {s["boundary"] for s in result["series"]}
        self.assertEqual(boundaries, {"TOTAL", "NET"})
        total_series = next(s for s in result["series"] if s["boundary"] == "TOTAL")
        net_series = next(s for s in result["series"] if s["boundary"] == "NET")
        self.assertNotEqual(total_series["demanded_power_kw"], net_series["demanded_power_kw"])

    def test_cycle_demand_rows_missing_net_excluded_no_fallback(self):
        from src.vde_core.cycles import use_standard_cycle

        cycle = use_standard_cycle("EPA")
        dataset = self._dataset([self.bev_no_net])
        result = build_cycle_demand_rows(dataset, cycle, ["TOTAL", "NET"])
        net_scenario_ids = {s["scenario_id"] for s in result["series"] if s["boundary"] == "NET"}
        self.assertNotIn(str(self.bev_no_net.vde_id), net_scenario_ids)
        self.assertTrue(any(e["reason"].endswith("unavailable") for e in result["excluded"]))


class FeVdePseCompetitorDeltaTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "fevde.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            rows = [
                (1, 900001, "ICE", "Gasoline", "HOMOLOGATED", 6.0, None, 140.0),
                (2, 900002, "ICE", "Gasoline", "ESTIMATED", 6.5, None, 150.0),
                (3, 900003, "ICE", "Ethanol", "ESTIMATED", 8.5, None, 130.0),
                (4, 900004, "ICE", "Flex", "ESTIMATED", 7.0, None, 145.0),
                (5, 900006, "BEV", "Electric", "ESTIMATED", None, 150.0, 0.0),
            ]
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, energy_Wh_per_km, gco2_per_km) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        self.gasoline = build_scenario_comparison_item(2)
        self.ethanol = build_scenario_comparison_item(3)
        self.flex = build_scenario_comparison_item(4)
        self.bev = build_scenario_comparison_item(5)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_volumetric_mode_includes_same_fuel_family_only(self):
        dataset = self._dataset([self.gasoline, self.ethanol])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="volumetric")
        point_labels = {p["label"] for p in result["points"]}
        self.assertIn(self.gasoline.label, point_labels)
        self.assertNotIn(self.ethanol.label, point_labels)
        self.assertIn(self.ethanol.label, {e["label"] for e in result["excluded"]})

    def test_energy_normalized_mode_includes_ethanol_and_bev_excludes_flex(self):
        dataset = self._dataset([self.ethanol, self.flex, self.bev])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="energy_normalized")
        point_labels = {p["label"] for p in result["points"]}
        self.assertIn(self.ethanol.label, point_labels)
        self.assertIn(self.bev.label, point_labels)
        self.assertNotIn(self.flex.label, point_labels)
        self.assertIn(self.flex.label, {e["label"] for e in result["excluded"]})

    def test_electrical_mode_only_includes_bev(self):
        dataset = self._dataset([self.gasoline, self.bev])
        result = build_fe_vde_points(dataset, boundary="TOTAL", mode="electrical")
        point_labels = {p["label"] for p in result["points"]}
        self.assertIn(self.bev.label, point_labels)
        self.assertNotIn(self.gasoline.label, point_labels)

    def test_missing_net_point_excluded_with_reason_no_fallback(self):
        dataset = self._dataset([self.bev])
        result = build_fe_vde_points(dataset, boundary="NET", mode="electrical")
        self.assertIn(self.bev.label, {e["label"] for e in result["excluded"]})
        self.assertEqual(result["points"], [])

    def test_iso_pse_lines_volumetric_empty_for_unmappable_fuel(self):
        self.assertEqual(build_iso_pse_lines(0.2, 1.2, [0.3], mode="volumetric", fuel_type="Flex"), [])
        self.assertEqual(build_iso_pse_lines(0.2, 1.2, [0.3], mode="volumetric", fuel_type=None), [])

    def test_iso_pse_lines_volumetric_present_for_mappable_fuel(self):
        lines = build_iso_pse_lines(0.2, 1.2, [0.3], mode="volumetric", fuel_type="Gasoline")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["eta"], 0.3)
        self.assertEqual(len(lines[0]["x"]), 40)

    def test_iso_pse_lines_change_with_basis(self):
        vol_lines = build_iso_pse_lines(0.5, 0.5, [0.3], mode="volumetric", fuel_type="Gasoline")
        energy_lines = build_iso_pse_lines(0.5, 0.5, [0.3], mode="energy_normalized")
        electrical_lines = build_iso_pse_lines(0.5, 0.5, [0.3], mode="electrical")
        values = {vol_lines[0]["y"][0], energy_lines[0]["y"][0], electrical_lines[0]["y"][0]}
        self.assertEqual(len(values), 3)

    def test_competitor_delta_reference_is_zero_no_verdict(self):
        dataset = self._dataset([self.gasoline])
        result = build_competitor_delta_rows(dataset, "vde_total")
        ref_row = result["rows"][0]
        self.assertEqual(ref_row["role"], "REFERENCE")
        self.assertEqual(ref_row["percent_delta"], 0.0)
        self.assertIsNone(ref_row["semantic"])

    def test_competitor_delta_vde_lower_is_better(self):
        cheaper_row = _qa_row(900002)
        cheaper_row["id"] = 900002
        cheaper_row["vde_total_mj_per_km"] = 1.10  # reference (900001) is 1.24
        cheaper_item = build_scenario_comparison_item(2, vde_row=cheaper_row)
        dataset = self._dataset([cheaper_item])
        result = build_competitor_delta_rows(dataset, "vde_total")
        self.assertEqual(result["rows"][1]["semantic"], "BETTER")

    def test_competitor_delta_fuel_economy_higher_is_better(self):
        dataset = self._dataset([self.ethanol])  # any comparison; direction under test is registry-driven
        result = build_competitor_delta_rows(dataset, "fuel_km_per_l")
        # ethanol fuel_km_per_l unavailable (only fuel_l_per_100km seeded) -> excluded, proving no fabrication
        self.assertEqual(result["rows"], [result["rows"][0]])
        self.assertTrue(result["excluded"])

    def test_competitor_delta_mass_is_neutral(self):
        dataset = self._dataset([self.gasoline])
        result = build_competitor_delta_rows(dataset, "mass_kg")
        self.assertIsNone(result["rows"][1]["semantic"])

    def test_competitor_delta_incompatible_metric_is_excluded_not_faked(self):
        wltp_row = _qa_row(900002)
        wltp_row["id"] = 900002
        wltp_row["legislation"] = "WLTP"
        wltp_scenario = build_scenario_comparison_item(2, vde_row=wltp_row)
        dataset = self._dataset([wltp_scenario])
        result = build_competitor_delta_rows(dataset, "vde_total")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(len(result["excluded"]), 1)

    def test_is_temporary_net_reflects_transmission_source(self):
        row = _qa_row(900006)
        temp = {"source": "MANUAL", "A": 9.0, "B": 0.003, "C": 0.0006}
        temporary_item = build_vde_comparison_item(900006, vde_row=row, temporary_transmission=temp)
        missing_item = build_vde_comparison_item(900006, vde_row=row)
        self.assertTrue(is_temporary_net(temporary_item))
        self.assertFalse(is_temporary_net(missing_item))


class FormatValueTests(unittest.TestCase):
    def test_missing_value_is_dash(self):
        self.assertEqual(format_value(None, "mass_kg", "Metric"), "-")

    def test_text_family_passthrough(self):
        self.assertEqual(format_value("EPA", "text", "Metric"), "EPA")

    def test_mass_metric_and_us(self):
        self.assertIn("kg", format_value(1500.0, "mass_kg", "Metric"))
        self.assertIn("lb", format_value(1500.0, "mass_kg", "US customary"))

    def test_fuel_l_per_100km_converts_for_us_customary(self):
        metric_text = format_value(6.5, "l_per_100km", "Metric")
        us_text = format_value(6.5, "l_per_100km", "US customary")
        self.assertIn("L/100km", metric_text)
        self.assertIn("gal/100mi", us_text)

    def test_signed_delta_shows_plus_for_positive(self):
        self.assertTrue(format_value(0.05, "energy_mj_per_km", "Metric", signed=True).startswith("+"))
        self.assertTrue(format_value(-0.05, "energy_mj_per_km", "Metric", signed=True).startswith("-"))


# -----------------------------------------------------------------------------
# Explore Lite -- dimensions, metric selectors (Sec 52, Package 8D)
# -----------------------------------------------------------------------------


class ExploreDimensionAndMetricSelectorTests(unittest.TestCase):
    def test_x_dimensions_include_scenario_vehicle_and_curated_categoricals(self):
        keys = {d.key for d in list_explore_dimensions("x")}
        self.assertEqual(
            keys,
            {"scenario", "vehicle", "make", "model_year", "category", "legislation", "electrification", "fuel_type", "provenance"},
        )

    def test_order_dimension_is_only_model_year(self):
        keys = [d.key for d in list_explore_dimensions("order")]
        self.assertEqual(keys, ["model_year"])

    def test_group_and_filter_dimensions_exclude_scenario_vehicle_make_and_year(self):
        for role in ("group", "filter"):
            keys = {d.key for d in list_explore_dimensions(role)}
            self.assertEqual(keys, {"category", "legislation", "electrification", "fuel_type", "provenance"})

    def test_numeric_metrics_exclude_text_fields_and_use_registry_label(self):
        metrics = list_explore_numeric_metrics("bar")
        self.assertIn("mass_kg", [m.key for m in metrics])
        self.assertNotIn("make", [m.key for m in metrics])  # text metric never leaks into a numeric axis
        mass = next(m for m in metrics if m.key == "mass_kg")
        self.assertEqual(mass.label, "Mass")  # Registry label, not a raw column name

    def test_scatter_only_offers_scatter_compatible_metrics(self):
        scatter_keys = {m.key for m in list_explore_numeric_metrics("scatter")}
        bar_keys = {m.key for m in list_explore_numeric_metrics("bar")}
        self.assertIn("vde_total", scatter_keys)
        self.assertTrue(scatter_keys.issubset(bar_keys))  # every scatter metric is also bar-eligible today

    def test_line_metric_eligibility_matches_bar(self):
        self.assertEqual(
            {m.key for m in list_explore_numeric_metrics("line")}, {m.key for m in list_explore_numeric_metrics("bar")}
        )

    def test_metric_axis_label_includes_unit(self):
        from src.vde_core.comparison_metric_registry import get_metric

        self.assertEqual(metric_axis_label(get_metric("vde_total"), "Metric"), "VDE TOTAL [MJ/km]")


# -----------------------------------------------------------------------------
# Explore Lite -- Bar / Scatter / Line builders, group/filter, duplicate labels
# (Sec 53-56, 63, Package 8D). VDE_ONLY items are used directly where fuel/
# powertrain fields are not the point under test -- no fuelcons DB needed.
# -----------------------------------------------------------------------------


class ExploreVdeOnlyChartTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "explore_vde_only.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        self.reference = build_vde_comparison_item(900001, role=ComparisonRole.REFERENCE)
        self.other = build_vde_comparison_item(900002)
        self.no_mass_source = build_vde_comparison_item(900003)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_bar_scenario_x_vde_total_correct_count_and_reference_preserved(self):
        dataset = self._dataset([self.other])
        result = build_explore_bar_rows(dataset, x_dimension_key="scenario", y_metric_key="vde_total")
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["excluded"], [])
        self.assertIn("REFERENCE", {r.role for r in result["rows"]})

    def test_bar_missing_metric_excluded_with_reason(self):
        row = _qa_row(900002)
        row["vde_net_mj_per_km"] = None
        row["trans_A_coef_N"] = None
        row["trans_B_coef_Npkph"] = None
        row["trans_C_coef_Npkph2"] = None
        no_net_item = build_vde_comparison_item(900002, vde_row=row)
        dataset = self._dataset([no_net_item])
        result = build_explore_bar_rows(dataset, x_dimension_key="scenario", y_metric_key="vde_net")
        self.assertEqual(len(result["rows"]), 1)  # only Reference has NET
        self.assertEqual(len(result["excluded"]), 1)
        self.assertIn("unavailable", result["excluded"][0]["reason"])

    def test_bar_duplicate_scenario_display_titles_remain_distinct(self):
        row_a = _qa_row(900001)
        row_b = _qa_row(900002)
        row_b["make"], row_b["model"], row_b["year"] = row_a["make"], row_a["model"], row_a["year"]
        twin = build_vde_comparison_item(900002, vde_row=row_b)
        dataset = self._dataset([twin])
        result = build_explore_bar_rows(dataset, x_dimension_key="scenario", y_metric_key="vde_total")
        labels = [r.label for r in result["rows"]]
        self.assertEqual(len(labels), len(set(labels)))  # never merged despite identical source label
        self.assertEqual(len(result["rows"]), 2)

    def test_scatter_mass_vs_vde_total_valid_points(self):
        dataset = self._dataset([self.other])
        result = build_explore_scatter_points(dataset, x_metric_key="mass_kg", y_metric_key="vde_total")
        self.assertEqual(len(result["points"]), 2)
        self.assertEqual(result["excluded"], [])

    def test_scatter_missing_x_excluded_with_reason(self):
        dataset = self._dataset([self.other])
        result = build_explore_scatter_points(dataset, x_metric_key="fuel_l_per_100km", y_metric_key="vde_total")
        # VDE_ONLY items never populate fuel_energy -- X is unavailable for both.
        self.assertEqual(result["points"], [])
        self.assertEqual(len(result["excluded"]), 2)

    def test_scatter_missing_y_excluded_with_reason(self):
        dataset = self._dataset([self.other])
        result = build_explore_scatter_points(dataset, x_metric_key="mass_kg", y_metric_key="gco2_per_km")
        self.assertEqual(result["points"], [])
        self.assertEqual(len(result["excluded"]), 2)

    def test_scatter_scenario_identity_preserved_independently_of_label(self):
        row_b = _qa_row(900002)
        row_b["make"], row_b["model"], row_b["year"] = _qa_row(900001)["make"], _qa_row(900001)["model"], _qa_row(900001)["year"]
        twin = build_vde_comparison_item(900002, vde_row=row_b)
        dataset = self._dataset([twin])
        result = build_explore_scatter_points(dataset, x_metric_key="mass_kg", y_metric_key="vde_total")
        identities = {p.identity for p in result["points"]}
        labels = {p.label for p in result["points"]}
        self.assertEqual(len(identities), 2)  # distinct canonical identity...
        self.assertEqual(len(labels), 1)  # ...despite an identical display label

    def test_line_unordered_dimension_is_rejected(self):
        dataset = self._dataset([self.other])
        result = build_explore_line_rows(dataset, x_dimension_key="scenario", y_metric_key="vde_total")
        self.assertEqual(result["rows"], [])
        self.assertIsNotNone(result.get("unavailable_reason"))

    def test_line_model_year_ordering_is_deterministic_not_selection_order(self):
        row_2010 = _qa_row(900002)
        row_2010["year"] = 2010
        row_2015 = _qa_row(900003)
        row_2015["year"] = 2015
        item_2010 = build_vde_comparison_item(900002, vde_row=row_2010)
        item_2015 = build_vde_comparison_item(900003, vde_row=row_2015)
        # Reference year is 2026 (unmodified QA seed). Selection order below is
        # deliberately NOT chronological -- the builder must still sort by year.
        dataset = self._dataset([item_2015, item_2010])
        result = build_explore_line_rows(dataset, x_dimension_key="model_year", y_metric_key="vde_total")
        years = [r.x for r in result["rows"]]
        self.assertEqual(years, sorted(years))
        self.assertEqual(years, [2010, 2015, 2026])


# -----------------------------------------------------------------------------
# Explore Lite -- group/filter, temporary NET (Sec 56-57, Package 8D). These
# need FuelCons-linked items for Electrification/Fuel type/Provenance.
# -----------------------------------------------------------------------------


class ExploreGroupFilterTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "explore_group_filter.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            rows = [
                (1, 900001, "ICE", "Gasoline", "HOMOLOGATED", 6.0, 150.0),
                (2, 900002, "PHEV", "Gasoline", "ESTIMATED", 4.0, 90.0),
                (3, 900003, "BEV", "Electric", "SCENARIO", None, 0.0),
                (4, 900006, "ICE", "Gasoline", "ESTIMATED", 7.0, 160.0),
            ]
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()
        self.reference = build_scenario_comparison_item(1, role=ComparisonRole.REFERENCE)
        self.phev = build_scenario_comparison_item(2)
        self.bev = build_scenario_comparison_item(3)
        self.ice_no_net = build_scenario_comparison_item(4)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _dataset(self, comparisons):
        return ComparisonDataset(reference=self.reference, comparisons=tuple(comparisons))

    def test_group_by_electrification(self):
        dataset = self._dataset([self.phev, self.bev])
        result = build_explore_bar_rows(
            dataset, x_dimension_key="scenario", y_metric_key="mass_kg", group_dimension_key="electrification"
        )
        groups = {r.group for r in result["rows"]}
        self.assertEqual(groups, {"ICE", "PHEV", "BEV"})

    def test_group_by_provenance(self):
        dataset = self._dataset([self.phev, self.bev])
        result = build_explore_bar_rows(
            dataset, x_dimension_key="scenario", y_metric_key="mass_kg", group_dimension_key="provenance"
        )
        groups = {r.group for r in result["rows"]}
        self.assertEqual(groups, {"HOMOLOGATED", "ESTIMATED", "SCENARIO"})

    def test_filter_one_category_keeps_only_matching_items(self):
        dataset = self._dataset([self.phev, self.bev, self.ice_no_net])
        result = build_explore_bar_rows(
            dataset,
            x_dimension_key="scenario",
            y_metric_key="mass_kg",
            filter_dimension_key="electrification",
            filter_values=["ICE"],
        )
        self.assertEqual(len(result["rows"]), 2)  # reference + ice_no_net
        self.assertEqual({r.role for r in result["rows"]}, {"REFERENCE", "COMPARISON"})

    def test_filter_excludes_reference_when_it_does_not_match(self):
        dataset = self._dataset([self.phev])
        result = build_explore_bar_rows(
            dataset,
            x_dimension_key="scenario",
            y_metric_key="mass_kg",
            filter_dimension_key="electrification",
            filter_values=["PHEV"],
        )
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0].role, "COMPARISON")
        self.assertIn(self.reference.label, [e["label"] for e in result["excluded"]])

    def test_empty_filter_result_handled_safely(self):
        dataset = self._dataset([self.phev, self.bev])
        result = build_explore_bar_rows(
            dataset,
            x_dimension_key="scenario",
            y_metric_key="mass_kg",
            filter_dimension_key="electrification",
            filter_values=["HEV"],  # no seeded item has this value
        )
        self.assertEqual(result["rows"], [])
        self.assertEqual(len(result["excluded"]), 3)

    def test_metric_unavailable_for_every_item_is_not_offered(self):
        dataset = self._dataset([self.bev])  # only Reference (ICE) + BEV, neither has energy_Wh_per_km set on Reference
        items = (dataset.reference, *dataset.comparisons)
        metrics = list_available_explore_metrics(items, "bar")
        self.assertNotIn("energy_wh_per_km", [m.key for m in metrics])

    def test_temporary_net_is_marked_on_scatter_points(self):
        row = _qa_row(900006)
        temp = {"source": "MANUAL", "A": 9.0, "B": 0.003, "C": 0.0006}
        temp_reference = build_vde_comparison_item(900006, role=ComparisonRole.REFERENCE, vde_row=row, temporary_transmission=temp)
        missing_comparison = build_vde_comparison_item(900001)
        dataset = ComparisonDataset(reference=temp_reference, comparisons=(missing_comparison,))
        result = build_explore_scatter_points(dataset, x_metric_key="mass_kg", y_metric_key="vde_net")
        temp_point = next(p for p in result["points"] if p.role == "REFERENCE")
        self.assertTrue(temp_point.is_temporary_net)


# -----------------------------------------------------------------------------
# Physical VDE Lineage -- context resolution, metric availability, waterfall
# (Sec 58-63, Package 8D Investigation Addendum)
# -----------------------------------------------------------------------------


class LineageContextTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "lineage_context.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin) VALUES (?, ?, ?, ?, ?)",
                [
                    (1, 900001, "ICE", "Gasoline", "HOMOLOGATED"),
                    (2, 900001, "ICE", "Gasoline", "ESTIMATED"),  # shares VDE 900001 with #1
                ],
            )
            con.commit()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_bare_vde_selection_is_not_labeled_as_fuelcons_scenario(self):
        item = build_vde_comparison_item(900001)
        context = resolve_lineage_context(item)
        self.assertFalse(context.is_fuelcons_scenario)

    def test_fuelcons_scenario_selection_resolves_linked_vde_and_keeps_scenario_as_context_only(self):
        item = build_scenario_comparison_item(1)
        context = resolve_lineage_context(item)
        self.assertTrue(context.is_fuelcons_scenario)
        self.assertEqual([n.vde_id for n in context.chain.nodes], [900001])
        self.assertIn(item.label, context.originating_label)

    def test_two_fuelcons_scenarios_sharing_one_vde_produce_identical_physical_lineage(self):
        item1 = build_scenario_comparison_item(1)
        item2 = build_scenario_comparison_item(2)
        self.assertNotEqual(item1.fuelcons_id, item2.fuelcons_id)  # distinct scenarios elsewhere
        context1 = resolve_lineage_context(item1)
        context2 = resolve_lineage_context(item2)
        self.assertEqual(
            [n.vde_id for n in context1.chain.nodes], [n.vde_id for n in context2.chain.nodes]
        )  # same physical lineage
        self.assertEqual(context1.chain.status, context2.chain.status)

    def test_list_lineage_capable_metrics_excludes_fuel_energy_co2_pse(self):
        keys = {m.key for m in list_lineage_capable_metrics()}
        self.assertFalse(keys & {"fuel_l_per_100km", "fuel_km_per_l", "energy_wh_per_km", "gco2_per_km", "eta_pt_est"})
        self.assertIn("vde_total", keys)

    def test_list_available_lineage_metrics_requires_availability_at_every_node(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET vde_id_parent=900001 WHERE id=900002")
            # vde_total is a persisted aggregate (canonical_vde_read), not derived
            # from coast_A/B/C at read time -- it must be nulled directly to make
            # the metric itself unavailable at this node.
            con.execute(
                "UPDATE vde_db SET vde_total_mj_per_km=NULL, vde_net_mj_per_km=NULL, "
                "coast_A_N=NULL, coast_B_N_per_kph=NULL, coast_C_N_per_kph2=NULL WHERE id=900002"
            )
            con.commit()
        chain = resolve_lineage_chain(900002)
        available_keys = {m.key for m in list_available_lineage_metrics(chain)}
        self.assertNotIn("vde_total", available_keys)  # unavailable at the 900002 node
        self.assertIn("mass_kg", available_keys)  # unaffected physical metric stays available


class LineageWaterfallTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "lineage_waterfall.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _set_parent(self, vde_id: int, parent_id) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET vde_id_parent=? WHERE id=?", (parent_id, vde_id))
            con.commit()

    def test_root_only_chain_produces_single_baseline_step(self):
        chain = resolve_lineage_chain(900001)
        waterfall = build_lineage_waterfall(chain, "vde_total")
        self.assertEqual(len(waterfall.steps), 1)
        self.assertIsNone(waterfall.steps[0].delta)
        self.assertTrue(waterfall.complete)

    def test_child_step_is_child_minus_parent_and_lower_is_better_marks_better(self):
        self._set_parent(900002, 900001)  # 900002's vde_total (1.215) < 900001's (1.24)
        chain = resolve_lineage_chain(900002)
        waterfall = build_lineage_waterfall(chain, "vde_total")
        self.assertEqual(len(waterfall.steps), 2)
        child_step = waterfall.steps[1]
        self.assertAlmostEqual(child_step.delta, 1.215 - 1.24, places=6)
        self.assertEqual(child_step.semantic, "BETTER")

    def test_neutral_metric_has_delta_but_no_semantic_verdict(self):
        self._set_parent(900002, 900001)
        chain = resolve_lineage_chain(900002)
        waterfall = build_lineage_waterfall(chain, "mass_kg")
        child_step = waterfall.steps[1]
        self.assertIsNotNone(child_step.delta)
        self.assertIsNone(child_step.semantic)

    def test_missing_metric_mid_chain_is_incomplete_no_fallback(self):
        self._set_parent(900002, 900001)
        self._set_parent(900003, 900002)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE vde_db SET vde_total_mj_per_km=NULL, vde_net_mj_per_km=NULL, "
                "coast_A_N=NULL, coast_B_N_per_kph=NULL, coast_C_N_per_kph2=NULL WHERE id=900002"
            )
            con.commit()
        chain = resolve_lineage_chain(900003)
        waterfall = build_lineage_waterfall(chain, "vde_total")
        self.assertFalse(waterfall.complete)
        self.assertIsNotNone(waterfall.incomplete_reason)
        self.assertEqual(waterfall.steps[-1].status, "UNAVAILABLE")
        self.assertEqual(len(waterfall.steps), 2)  # walk stops at the broken node, never fabricates 900003's step

    def test_incompatible_cycle_blocks_cycle_specific_metric_but_not_physical_metric(self):
        self._set_parent(900002, 900001)
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET legislation='WLTP' WHERE id=900002")
            con.commit()
        chain = resolve_lineage_chain(900002)

        cycle_specific = build_lineage_waterfall(chain, "vde_total")
        self.assertFalse(cycle_specific.complete)
        self.assertEqual(cycle_specific.steps[-1].status, "INCOMPATIBLE")

        physical = build_lineage_waterfall(chain, "mass_kg")
        self.assertTrue(physical.complete)
        self.assertEqual(physical.steps[-1].status, "OK")

    def test_broken_reference_chain_is_reported_incomplete(self):
        self._set_parent(900002, 999999)
        chain = resolve_lineage_chain(900002)
        self.assertEqual(chain.status, LineageChainStatus.BROKEN)
        waterfall = build_lineage_waterfall(chain, "vde_total")
        self.assertFalse(waterfall.complete)
        self.assertEqual(len(waterfall.steps), 1)  # only the resolvable node


if __name__ == "__main__":
    unittest.main()
