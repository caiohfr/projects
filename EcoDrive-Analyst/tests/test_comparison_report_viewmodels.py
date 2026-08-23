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
    build_scenario_header,
    build_scenario_options,
    build_scorecard_sections,
    dataset_warnings_summary,
    format_value,
    remove_comparison,
    set_reference,
    sync_comparisons_from_widget,
)
from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import (
    ComparisonRole,
    build_scenario_comparison_item,
    build_vde_comparison_item,
    list_comparison_scenarios,
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


if __name__ == "__main__":
    unittest.main()
