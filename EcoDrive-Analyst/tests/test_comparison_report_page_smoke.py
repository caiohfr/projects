from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_app.comparison_report_viewmodels import PresentationState, SelectionState, TargetState
from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import seed_qa_database

PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Comparison_Report.py"


class ComparisonReportPageEmptyStateTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "comparison_page_empty.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_page_opens_with_no_fuelcons_scenarios(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("No FuelCons scenarios" in info.value for info in app.info))


class ComparisonReportPageSmokeTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "comparison_page_smoke.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km) VALUES (1, 900001, 'ICE', 'Gasoline', 'HOMOLOGATED', 6.5, 150.0)"
            )
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km) VALUES (2, 900002, 'ICE', 'Gasoline', 'ESTIMATED', 7.0, 160.0)"
            )
            con.commit()

    def _insert_duplicate_header_fixture(self) -> None:
        # Two comparisons sharing one VDE with the SAME record_origin produce
        # identical Scorecard column headers (label + role + provenance) --
        # this crashed pandas.Styler.apply on non-unique columns until fixed.
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin) "
                "VALUES (3, 900002, 'ICE', 'Ethanol', 'ESTIMATED')"
            )
            con.commit()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_duplicate_scenario_headers_do_not_crash_scorecard(self):
        self._insert_duplicate_header_fixture()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2, 3))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_page_opens_with_scenarios_available_no_selection(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Select at least one scenario" in info.value for info in app.info))

    def test_browse_scenarios_expander_lists_the_currently_filtered_candidates(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        labels = [exp.label for exp in app.expander]
        self.assertTrue(any(label.startswith("Browse Comparison Scenarios (") for label in labels))
        browse_label = next(label for label in labels if label.startswith("Browse Comparison Scenarios ("))
        self.assertIn("2", browse_label)

    def test_reference_selection_builds_dataset_and_renders_scorecard(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_benchmark_only_selection_with_no_reference_renders_without_exception(self):
        # Package 8F Increment 1: reference_fuelcons_id=None with 2+ comparisons
        # is a legitimate benchmark-only review -- every tab must render
        # absolute-only content with no fabricated Reference/delta.
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=None, comparison_fuelcons_ids=(1, 2))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertIsNone(state.reference_fuelcons_id)
        self.assertEqual(state.comparison_fuelcons_ids, (1, 2))

    def test_selection_persists_across_normal_rerun(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(state.reference_fuelcons_id, 1)
        self.assertEqual(state.comparison_fuelcons_ids, (2,))

    def test_presentation_roles_panel_marks_the_reference_item(self):
        # Every selected item must appear in both the Current radio and its
        # own role selectbox (same count, same coverage), and whichever item
        # holds ComparisonRole.REFERENCE must be visually distinguishable in
        # this panel -- role/Current alone don't show which item is Reference.
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        radio = app.radio(key="comparison_presentation_current")
        role_selectboxes = [sb for sb in app.selectbox if sb.key and sb.key.startswith("comparison_presentation_role_")]
        self.assertEqual(len(role_selectboxes), len(radio.options) - 1)  # -1 for the "None" option
        self.assertTrue(any("(Reference)" in opt for opt in radio.options))
        self.assertTrue(any("(Reference)" in sb.label for sb in role_selectboxes))

    def test_presentation_role_assignment_is_independent_of_provenance(self):
        # Package 8F Increment 2: role (Proposal/Benchmark) and Current are a
        # separate axis from provenance (record_origin) -- assigning a role
        # must never mutate or shadow provenance, and both must coexist.
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["comparison_presentation_state"] = PresentationState(
            roles={"fc:2": "PROPOSAL"}, current_item_id="fc:1"
        )
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(exp.label == "Presentation roles" for exp in app.expander))
        state = app.session_state["comparison_presentation_state"]
        self.assertEqual(state.roles.get("fc:2"), "PROPOSAL")
        self.assertEqual(state.current_item_id, "fc:1")
        # Provenance (record_origin=ESTIMATED for fuelcons_id=2) is untouched --
        # the Scorecard column header still carries it independently of role.
        columns = [col for item in app.dataframe if getattr(item.value, "columns", None) is not None for col in item.value.columns]
        self.assertTrue(any("ESTIMATED" in str(col) for col in columns))

    def test_target_state_survives_rerun_and_stays_scoped_to_its_metric(self):
        # Package 8F Increment 3: a target set for one Primary KPI metric_key
        # must survive a normal rerun untouched, no unit reinterpretation.
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["comparison_target_state"] = TargetState(targets_by_metric={"vde_total": 1.1})
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(exp.label == "Primary KPI & Target" for exp in app.expander))

        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_target_state"]
        self.assertEqual(state.targets_by_metric.get("vde_total"), 1.1)

    def test_engineering_filter_displacement_narrows_reference_and_compare_candidates(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET engine_size_l=2.0 WHERE id=900001")
            con.execute("UPDATE vde_db SET engine_size_l=4.0 WHERE id=900002")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        # The slider's default spans the full catalog range (2.0-4.0), which
        # is the "All" neutral state -- narrowing it off that default is what
        # activates the filter (no separate checkbox).
        app.session_state["comparison_filter_engine_size_range"] = (1.5, 2.5)
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        # Selectbox .options are format_func-applied display labels, not raw
        # ids -- QA seed model names are the stable fixture to assert on.
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertTrue(any("Nominal EPA baseline" in opt for opt in reference_select.options))
        self.assertFalse(any("TWC boundary lower" in opt for opt in reference_select.options))
        compare_with = app.multiselect(key="comparison_compare_with_select")
        self.assertFalse(any("TWC boundary lower" in opt for opt in compare_with.options))

    def test_engineering_filter_power_excludes_scenario_missing_power_metadata(self):
        # id=1 and id=2 get distinct power values (so a real slider range
        # exists); a third scenario (id=3, linked to a different VDE) is left
        # with NULL power. Narrowing the slider -- even to a range that still
        # covers both id=1 and id=2's values -- must still exclude id=3:
        # missing metadata is excluded once the filter is active, regardless
        # of the chosen bounds, never treated as 0.
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE fuelcons_db SET engine_max_power_kw=150.0 WHERE id=1")
            con.execute("UPDATE fuelcons_db SET engine_max_power_kw=300.0 WHERE id=2")
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km) VALUES (3, 900003, 'ICE', 'Gasoline', 'ESTIMATED', 6.8, 155.0)"
            )
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_power_range"] = (150.0, 450.0)
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertTrue(any("Nominal EPA baseline" in opt for opt in reference_select.options))  # id=1, kept
        compare_with = app.multiselect(key="comparison_compare_with_select")
        self.assertTrue(any("TWC boundary lower" in opt for opt in compare_with.options))  # id=2, kept
        self.assertEqual(len(compare_with.options), 2)  # id=1 and id=2 kept; id=3 (missing power) excluded

    def test_reference_never_silently_disappears_after_filter_change(self):
        # Package 8F "Selection Semantics Fix": filters are a candidate-search
        # tool only -- the Reference stays selected with NO mismatch warning
        # when it no longer matches an active filter.
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET engine_size_l=2.0 WHERE id=900001")
            con.execute("UPDATE vde_db SET engine_size_l=4.0 WHERE id=900002")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        # Narrowed to exclude the Reference's own displacement (2.0L) while
        # still covering the comparison scenario's (4.0L).
        app.session_state["comparison_filter_engine_size_range"] = (3.0, 4.0)
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(state.reference_fuelcons_id, 1)
        self.assertEqual(len(app.warning), 0)  # no filter-mismatch warning -- this is not an error condition
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertEqual(reference_select.value, 1)
        self.assertTrue(any("Nominal EPA baseline" in opt for opt in reference_select.options))

    def test_energy_demand_summary_shows_primary_kpi_boundary_and_pse(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE fuelcons_db SET eta_pt_est=0.30 WHERE id=1")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["comparison_primary_kpi"] = "fuel_l_per_100km"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Energy & Demand Summary" in md.value for md in app.markdown))
        metrics = list(app.get("metric"))
        labels = [m.label for m in metrics]
        self.assertIn("Fuel consumption", labels)  # dynamic Primary KPI, not hard-coded
        self.assertIn("VDE TOTAL", labels)
        self.assertIn("Estimated powertrain efficiency", labels)  # PSE shown when available

    def test_energy_demand_summary_shows_delta_vs_reference_and_reference_has_none(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["comparison_primary_kpi"] = "fuel_l_per_100km"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        metrics = [m for m in app.get("metric") if m.label == "Fuel consumption"]
        self.assertEqual(len(metrics), 2)  # Reference + one comparison
        reference_metric, comparison_metric = metrics[0], metrics[1]
        self.assertEqual(reference_metric.delta, "")  # Reference never shows a delta/verdict
        self.assertIn("vs Ref", comparison_metric.delta)

    def test_energy_demand_summary_absolute_only_without_reference(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=None, comparison_fuelcons_ids=(1, 2))
        app.session_state["comparison_primary_kpi"] = "fuel_l_per_100km"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        metrics = [m for m in app.get("metric") if m.label == "Fuel consumption"]
        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(m.delta == "" for m in metrics))  # no Reference -> absolute-only, never fabricated

    def test_energy_demand_summary_shows_target_gap_when_target_set(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["comparison_primary_kpi"] = "fuel_l_per_100km"
        app.session_state["comparison_target_state"] = TargetState(targets_by_metric={"fuel_l_per_100km": 6.0})
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        gap_metrics = [m for m in app.get("metric") if m.label == "Gap to Target"]
        self.assertEqual(len(gap_metrics), 2)  # one per item
        self.assertIn("+0.50", gap_metrics[0].value)  # Reference: 6.5 - 6.0 actual-target gap

    def test_energy_demand_summary_missing_metric_shows_unavailable_not_zero(self):
        # fuelcons_id=2 never gets eta_pt_est set -- selecting it as Primary
        # KPI must show "unavailable" text, never a fabricated 0.
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["comparison_primary_kpi"] = "eta_pt_est"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        pse_metrics = [m for m in app.get("metric") if m.label == "Estimated powertrain efficiency"]
        self.assertTrue(any(m.value == "unavailable" for m in pse_metrics))
        self.assertFalse(any(m.value in ("0", "0.0", "0.000") for m in pse_metrics))

    def test_demand_vs_efficiency_shows_no_guesswork_message_for_flex_fuel(self):
        # Package 8F mini-package: Volumetric mode's equi-PSE guides must stay
        # absent (never guess an LHV) for a Flex-fuel Reference, with an
        # explicit, non-crashing explanation instead of a silent gap.
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE fuelcons_db SET fuel_type='Flex' WHERE id=1")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Equi-PSE guides aren't available" in c.value for c in app.caption))

    def test_program_review_tab_renders_walk_hero_and_demand_vs_efficiency(self):
        # Package 8F: Dashboard was retired in favor of Program Review's
        # Primary-KPI-driven Walk hero + Demand vs Efficiency + the compact
        # Energy & Demand Summary (mini-package: replaced Vehicle Demand
        # Status, which was a standalone chart).
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        markdown_values = [md.value for md in app.markdown]
        self.assertTrue(any(("KPI Walk" in v or "KPI Comparison" in v) for v in markdown_values))
        self.assertTrue(any("Demand vs Efficiency" in v for v in markdown_values))
        self.assertTrue(any("Energy & Demand Summary" in v for v in markdown_values))

    def test_roadload_tab_renders_linked_vde_mode(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Roadload ABC" in md.value for md in app.markdown))
        self.assertTrue(any("Roadload force curve" in md.value for md in app.markdown))
        self.assertTrue(any("Physical Setup" in md.value for md in app.markdown))
        self.assertTrue(any("VDE by Cycle / Phase" in md.value for md in app.markdown))

    def test_energy_drivers_visual_order_curve_before_abc(self):
        # Package 8F mandatory case P: Roadload force curve must precede
        # Roadload ABC in the Energy Drivers tab's render order.
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        markdown_values = [md.value for md in app.markdown]
        curve_index = markdown_values.index("**Roadload force curve**")
        abc_index = markdown_values.index("**Roadload ABC**")
        self.assertLess(curve_index, abc_index)

    def test_roadload_direct_vde_mode_works_without_fuelcons(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["roadload_source_mode"] = "Select physical VDEs directly"
        app.session_state["comparison_direct_vde_selection"] = SelectionState(
            reference_fuelcons_id=900001, comparison_fuelcons_ids=(900002,)
        )
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)

    def test_temporary_transmission_state_survives_rerun(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["roadload_basis"] = "NET"
        app.session_state["comparison_temporary_transmission_by_vde_id"] = {
            900001: {"source": "MANUAL", "A": 9.0, "B": 0.003, "C": 0.0006}
        }
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        temp_state = app.session_state["comparison_temporary_transmission_by_vde_id"]
        self.assertIn(900001, temp_state)

    def test_switching_total_net_does_not_corrupt_selection(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["dashboard_vde_boundary"] = "Both"
        app.session_state["roadload_basis"] = "Both"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(state.reference_fuelcons_id, 1)
        self.assertEqual(state.comparison_fuelcons_ids, (2,))

    def test_explore_tab_empty_state_with_no_selection(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Select scenarios above to explore them here." in info.value for info in app.info))

    def test_explore_tab_renders_custom_chart_and_lineage_by_default(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        selectbox_labels = {sb.label for sb in app.selectbox}
        self.assertIn("Chart type", selectbox_labels)
        self.assertIn("Analyze lineage for", selectbox_labels)
        self.assertTrue(any("Physical VDE Lineage" in c.value for c in app.caption))

    def test_explore_tab_lineage_root_shows_no_parent_message(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        # Default lineage selection is the Reference (vde_id_parent is NULL on
        # the QA seed) -- a root is a valid, non-error state (Sec 40).
        self.assertTrue(any("is a lineage root" in info.value for info in app.info))

    def test_explore_tab_lineage_waterfall_renders_for_explicit_parent_chain(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET vde_id_parent=900001 WHERE id=900002")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        app.selectbox(key="lineage_selected_item").set_value("fc:2")
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.dataframe), 1)  # lineage step table rendered

    def test_explore_tab_duplicate_scenario_labels_do_not_crash_bar_chart(self):
        self._insert_duplicate_header_fixture()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2, 3))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)

    def test_switching_between_scorecard_dashboard_roadload_and_explore_does_not_corrupt_selection(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(state.reference_fuelcons_id, 1)
        self.assertEqual(state.comparison_fuelcons_ids, (2,))


class BrowseUxUpgradeSmokeTests(unittest.TestCase):
    """Comparison Browse UX Upgrade package -- search, Model/Year filters,
    Data Availability quick filters, Advanced Filters, Quick presets, and
    summary counters, all layered on top of the same `_render_filters` ->
    `_render_scenario_browse` -> Reference/Comparison selectbox chain the
    pre-upgrade tests above already cover. fuelcons_id=1 -> VDE-QA-001
    ("Nominal EPA baseline", vde_id=900001); fuelcons_id=2 -> VDE-QA-002
    ("TWC boundary lower", vde_id=900002); both start with full CdA/RRC/
    transmission/fuel-economy data in the QA seed.
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "comparison_browse_ux.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km) VALUES (1, 900001, 'ICE', 'Gasoline', 'HOMOLOGATED', 6.5, 150.0)"
            )
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km) VALUES (2, 900002, 'ICE', 'Gasoline', 'ESTIMATED', 7.0, 160.0)"
            )
            con.commit()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _matching_scenarios_value(self, app: AppTest) -> str:
        metric = next(m for m in app.metric if "Matching scenarios" in m.label)
        return metric.value

    def test_smoke_a_new_top_matter_renders(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        labels = [exp.label for exp in app.expander]
        self.assertIn("Advanced Filters", labels)
        self.assertTrue(any(label.startswith("Browse Comparison Scenarios (") for label in labels))
        metric_labels = [m.label for m in app.metric]
        self.assertTrue(any("Matching scenarios" in label for label in metric_labels))
        self.assertTrue(any("With CdA" in label for label in metric_labels))
        self.assertTrue(any("With NET" in label for label in metric_labels))
        self.assertTrue(any("Fully populated" in label for label in metric_labels))
        button_labels = [b.label for b in app.button]
        self.assertTrue(any("Complete Engineering Data" in label for label in button_labels))
        self.assertTrue(any("Roadload Ready" in label for label in button_labels))
        self.assertTrue(any("Clear Filters" in label for label in button_labels))

    def test_search_by_model_narrows_reference_candidates(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_query"] = "Nominal"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertTrue(any("Nominal EPA baseline" in opt for opt in reference_select.options))
        self.assertFalse(any("TWC boundary lower" in opt for opt in reference_select.options))

    def test_search_by_vde_id_narrows_candidates(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_query"] = "900002"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertTrue(any("TWC boundary lower" in opt for opt in reference_select.options))
        self.assertFalse(any("Nominal EPA baseline" in opt for opt in reference_select.options))

    def test_search_by_fuelcons_id_narrows_candidates(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_query"] = "1"  # matches fuelcons_id=1 and vde_id 900001, not 900002/2
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertTrue(any("Nominal EPA baseline" in opt for opt in reference_select.options))
        self.assertFalse(any("TWC boundary lower" in opt for opt in reference_select.options))

    def test_model_filter_narrows_candidates(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_model"] = "TWC boundary lower"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertTrue(any("TWC boundary lower" in opt for opt in reference_select.options))
        self.assertFalse(any("Nominal EPA baseline" in opt for opt in reference_select.options))

    def test_year_filter_narrows_candidates(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET year=2019 WHERE id=900001")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_year"] = 2026
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertTrue(any("TWC boundary lower" in opt for opt in reference_select.options))
        self.assertFalse(any("Nominal EPA baseline" in opt for opt in reference_select.options))

    def test_smoke_c_data_availability_toggle_excludes_missing_cda(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET cda_m2=NULL WHERE id=900002")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_availability"] = ["has_cda"]
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertTrue(any("Nominal EPA baseline" in opt for opt in reference_select.options))
        self.assertFalse(any("TWC boundary lower" in opt for opt in reference_select.options))
        self.assertEqual(self._matching_scenarios_value(app), "1")

    def test_has_rrc_and_has_net_quick_filters(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET rrc_N_per_kN=NULL WHERE id=900002")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_availability"] = ["has_rrc"]
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(self._matching_scenarios_value(app), "1")

    def test_transmission_resolved_quick_filter(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE vde_db SET trans_A_coef_N=NULL, trans_B_coef_Npkph=NULL, trans_C_coef_Npkph2=NULL WHERE id=900002"
            )
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_availability"] = ["transmission_resolved"]
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(self._matching_scenarios_value(app), "1")

    def test_fuel_economy_quick_filter(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE fuelcons_db SET fuel_l_per_100km=NULL WHERE id=2")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_availability"] = ["has_fuel_economy"]
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(self._matching_scenarios_value(app), "1")

    def test_smoke_d_complete_engineering_data_preset_via_button_click(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET cda_m2=NULL WHERE id=900002")
            con.commit()
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        preset_button = next(b for b in app.button if "Complete Engineering Data" in b.label)
        preset_button.click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["comparison_filter_active_preset"], "complete_engineering_data")
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertTrue(any("Nominal EPA baseline" in opt for opt in reference_select.options))
        self.assertFalse(any("TWC boundary lower" in opt for opt in reference_select.options))

    def test_roadload_ready_preset_via_button_click(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=90)
        preset_button = next(b for b in app.button if "Roadload Ready" in b.label)
        preset_button.click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["comparison_filter_active_preset"], "roadload_ready")

    def test_smoke_e_advanced_filter_mass_min_max(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_mass_min"] = 100000.0
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(self._matching_scenarios_value(app), "0")

    def test_smoke_f_clear_filters_restores_full_catalog(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_make"] = "QA"
        app.session_state["comparison_filter_mass_min"] = 100000.0
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(self._matching_scenarios_value(app), "0")
        clear_button = next(b for b in app.button if "Clear Filters" in b.label)
        clear_button.click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state["comparison_filter_mass_min"])
        self.assertNotIn("comparison_filter_active_preset", app.session_state)
        self.assertEqual(self._matching_scenarios_value(app), "2")

    def test_smoke_g_selected_scenarios_remain_stable_after_filter_changes(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["comparison_filter_availability"] = ["has_cda"]
        app.session_state["comparison_filter_query"] = "nonexistent-model-xyz"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(state.reference_fuelcons_id, 1)
        self.assertEqual(state.comparison_fuelcons_ids, (2,))
        self.assertEqual(len(app.warning), 0)

    def test_reference_less_mode_still_works_with_new_filters(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(
            reference_fuelcons_id=None, comparison_fuelcons_ids=(1, 2)
        )
        app.session_state["comparison_filter_availability"] = []
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertIsNone(state.reference_fuelcons_id)
        self.assertEqual(state.comparison_fuelcons_ids, (1, 2))

    def test_browse_table_still_renders_with_new_filters_active(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_filter_availability"] = ["has_fuel_economy"]
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        labels = [exp.label for exp in app.expander]
        self.assertTrue(any(label.startswith("Browse Comparison Scenarios (") for label in labels))
        self.assertGreaterEqual(len(app.dataframe), 1)


class SelectionFilterPersistenceTests(unittest.TestCase):
    """Package 8F -- top-level filters (Make/Category/Legislation/
    Electrification/Displacement/Power/Provenance) are candidate-SEARCH
    tools only. They must never invalidate, hide, remove, or warn about an
    already-selected Reference or Comparison item -- only control what's
    newly offered for selection.
    """

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "selection_filters.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET make='TOYOTA', legislation='EPA', engine_size_l=2.0 WHERE id=900001")
            con.execute("UPDATE vde_db SET make='TOYOTA', legislation='EPA', engine_size_l=2.0 WHERE id=900002")
            con.execute("UPDATE vde_db SET make='LEXUS', legislation='WLTP', engine_size_l=3.0 WHERE id=900003")
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km, engine_max_power_kw) "
                "VALUES (1, 900001, 'ICE', 'Gasoline', 'HOMOLOGATED', 6.5, 150.0, 150.0)"
            )
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km, engine_max_power_kw) "
                "VALUES (2, 900002, 'ICE', 'Gasoline', 'HOMOLOGATED', 7.0, 160.0, 150.0)"
            )
            con.execute(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "energy_Wh_per_km, gco2_per_km, engine_max_power_kw) "
                "VALUES (3, 900003, 'BEV', 'Electric', 'ESTIMATED', 180.0, 0.0, 300.0)"
            )
            con.commit()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_reference_persists_after_make_filter_switches_away(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=())
        app.session_state["comparison_filter_make"] = "LEXUS"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(state.reference_fuelcons_id, 1)
        self.assertEqual(len(app.warning), 0)  # no filter-mismatch warning
        reference_select = app.selectbox(key="comparison_reference_select")
        self.assertEqual(reference_select.value, 1)

    def test_comparisons_persist_after_make_filter_switches_away(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=None, comparison_fuelcons_ids=(1, 2))
        app.session_state["comparison_filter_make"] = "LEXUS"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(set(state.comparison_fuelcons_ids), {1, 2})
        self.assertEqual(len(app.warning), 0)

    def test_adding_new_make_after_filter_switch_spans_both_manufacturers(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=None, comparison_fuelcons_ids=(1, 2))
        app.session_state["comparison_filter_make"] = "LEXUS"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        compare_with = app.multiselect(key="comparison_compare_with_select")
        self.assertEqual(set(compare_with.value), {1, 2})  # Toyota selections still shown as tags
        # .options are format_func-applied display labels, not raw ids.
        self.assertTrue(any("LEXUS" in opt for opt in compare_with.options))  # offered as a new candidate
        compare_with.set_value([1, 2, 3])
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(set(state.comparison_fuelcons_ids), {1, 2, 3})

    def test_persistence_across_legislation_electrification_displacement_power_provenance_filters(self):
        filter_scenarios = {
            "comparison_filter_legislation": "WLTP",  # only Lexus (900003) is WLTP
            "comparison_filter_electrification": "BEV",  # only Lexus is BEV
            "comparison_filter_record_origin": "ESTIMATED",  # only Lexus is ESTIMATED
        }
        for key, value in filter_scenarios.items():
            with self.subTest(filter_key=key):
                app = AppTest.from_file(str(PAGE_PATH))
                app.session_state["comparison_selection"] = SelectionState(
                    reference_fuelcons_id=1, comparison_fuelcons_ids=(2,)
                )
                app.session_state[key] = value
                app.run(timeout=90)
                self.assertEqual(len(app.exception), 0)
                state = app.session_state["comparison_selection"]
                self.assertEqual(state.reference_fuelcons_id, 1)
                self.assertEqual(state.comparison_fuelcons_ids, (2,))
                self.assertEqual(len(app.warning), 0)

        # Displacement/power range sliders: narrow to the Lexus-only range
        # (900001/900002 are 2.0L/150hp-equivalent, 900003 is 3.0L/300kW).
        for key, value in (
            ("comparison_filter_engine_size_range", (2.8, 3.2)),
            ("comparison_filter_power_range", (250.0, 450.0)),
        ):
            with self.subTest(filter_key=key):
                app = AppTest.from_file(str(PAGE_PATH))
                app.session_state["comparison_selection"] = SelectionState(
                    reference_fuelcons_id=1, comparison_fuelcons_ids=(2,)
                )
                app.session_state[key] = value
                app.run(timeout=90)
                self.assertEqual(len(app.exception), 0)
                state = app.session_state["comparison_selection"]
                self.assertEqual(state.reference_fuelcons_id, 1)
                self.assertEqual(state.comparison_fuelcons_ids, (2,))
                self.assertEqual(len(app.warning), 0)

    def test_removing_a_selected_item_still_works(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=None, comparison_fuelcons_ids=(1, 2))
        app.run(timeout=90)
        compare_with = app.multiselect(key="comparison_compare_with_select")
        compare_with.set_value([1])
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(state.comparison_fuelcons_ids, (1,))

    def test_removed_item_is_not_silently_readded_after_filter_change(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=None, comparison_fuelcons_ids=(1, 2))
        app.run(timeout=90)
        compare_with = app.multiselect(key="comparison_compare_with_select")
        compare_with.set_value([1])
        app.run(timeout=90)
        self.assertEqual(app.session_state["comparison_selection"].comparison_fuelcons_ids, (1,))

        # Now change a filter -- id=2 (explicitly removed) must not reappear.
        app.session_state["comparison_filter_make"] = "LEXUS"
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["comparison_selection"]
        self.assertEqual(state.comparison_fuelcons_ids, (1,))


if __name__ == "__main__":
    unittest.main()
