"""Sprint 10E: AppTest-based smoke coverage for the Quick Scenarios tab on
the live Comparison Report page -- this codebase's established substitute
for a manual browser check (Sprint 9 precedent, mirrored by
test_comparison_report_page_smoke.py).
"""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_app.comparison_report_viewmodels import SelectionState
from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import seed_qa_database, seed_qa_fuelcons_mock_rows
from src.vde_core.quick_scenario import PseProvenance, ScalarChangeMode, build_quick_comparison_item

PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Comparison_Report.py"


class QuickScenarioTabSmokeTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "quick_scenario_ui_qa.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(self.db_path)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _app_with_selection(self) -> AppTest:
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(
            reference_fuelcons_id=900102, comparison_fuelcons_ids=(900104,)
        )
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_tab_renders_with_no_slots_yet(self):
        app = self._app_with_selection()
        self.assertTrue(any("No Quick Scenarios yet" in info.value for info in app.info))

    def test_add_slot_renders_an_editor(self):
        app = self._app_with_selection()
        app.button(key="comparison_quick_add_slot").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertIsNotNone(app.text_input(key="comparison_quick_label_fc:900102_1"))

    def test_calculate_a_neutral_change_produces_a_ready_slot_and_inserts_into_scorecard(self):
        app = self._app_with_selection()
        app.button(key="comparison_quick_add_slot").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        app.checkbox(key="comparison_quick_aero_fc:900102_1_enabled").set_value(True).run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        app.selectbox(key="comparison_quick_aero_fc:900102_1_mode").select(ScalarChangeMode.PERCENT).run(
            timeout=90
        )
        self.assertEqual(len(app.exception), 0)

        app.button(key="comparison_quick_calculate").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        results = app.session_state["comparison_quick_results"]["fc:900102"][1]
        vehicle_resolution, _efficiency_resolution = results
        self.assertTrue(vehicle_resolution.is_ready)
        # The Quick item's label is rendered into at least one existing
        # table (e.g. Technical Scorecard) -- confirms it flowed into the
        # SAME dataset every other tab already consumes, not a parallel
        # display of its own.
        label_fragment = "Quick #1 of fc:900102"
        self.assertTrue(
            any(label_fragment in str(df.value) for df in app.dataframe),
            "Expected the Quick item's label to appear in at least one rendered table.",
        )

    def test_reset_clears_all_quick_scenario_state(self):
        app = self._app_with_selection()
        app.button(key="comparison_quick_add_slot").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        app.button(key="comparison_quick_reset").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        # Rendering the tab again after Reset lazily recreates an empty
        # per-source entry (a harmless setdefault side effect) -- what
        # matters is that no slots survive for the active source.
        self.assertEqual(app.session_state["comparison_quick_scenarios"].get("fc:900102", {}), {})
        self.assertEqual(app.session_state["comparison_quick_results"].get("fc:900102", {}), {})

    def test_accepting_a_benchmark_reference_sets_final_pse_and_provenance(self):
        # Closure audit finding: the "Accept as Final PSE" button set a
        # side-channel session key that the Final PSE number_input's
        # value= argument could never actually apply (Streamlit ignores
        # value= once a keyed widget already has stored state) -- this
        # end-to-end flow is exactly what would have caught that bug.
        app = self._app_with_selection()
        app.button(key="comparison_quick_add_slot").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        app.selectbox(key="comparison_quick_pse_fc:900102_1_benchmark_select").select("fc:900103").run(
            timeout=90
        )
        self.assertEqual(len(app.exception), 0)

        app.button(key="comparison_quick_calculate").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        _vehicle_resolution, efficiency_resolution = app.session_state["comparison_quick_results"][
            "fc:900102"
        ][1]
        self.assertTrue(efficiency_resolution.benchmark_pse.is_available)
        benchmark_value = efficiency_resolution.benchmark_pse.value_percent

        accept_button = app.button(key="comparison_quick_pse_fc:900102_1_accept_benchmark")
        accept_button.click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        updated_scenario = app.session_state["comparison_quick_scenarios"]["fc:900102"][1]
        self.assertAlmostEqual(updated_scenario.final_pse_percent, benchmark_value, places=6)
        self.assertEqual(updated_scenario.pse_provenance, PseProvenance.BENCHMARK_ACCEPTED)

        # The visible Final PSE widget reflects the accepted value too, not
        # just the underlying QuickScenario object.
        final_pse_widget = app.number_input(key="comparison_quick_pse_fc:900102_1_final_value")
        self.assertAlmostEqual(final_pse_widget.value, benchmark_value, places=6)

    def test_mass_and_tire_change_together_resolve_and_calculate(self):
        # Smoke-matrix Case B (Mass + Tire): fc:900102/vde_id=900001 is an
        # EPA-legislation source (per the QA fixture). A curb-mass Delta
        # plus a Tire Improvement together must resolve through both
        # domains and calculate to a ready slot.
        app = self._app_with_selection()
        app.button(key="comparison_quick_add_slot").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        app.radio(key="comparison_quick_mass_fc:900102_1_mode").set_value(
            "Target curb-to-TWC / WLTP mass line"
        ).run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        app.selectbox(key="comparison_quick_mass_fc:900102_1_scalar_mode").select(
            ScalarChangeMode.DELTA
        ).run(timeout=90)
        app.number_input(key="comparison_quick_mass_fc:900102_1_scalar_value").set_value(-50.0).run(
            timeout=90
        )
        self.assertEqual(len(app.exception), 0)

        app.number_input(key="comparison_quick_tire_fc:900102_1_improvement_pct").set_value(5.0).run(
            timeout=90
        )
        self.assertEqual(len(app.exception), 0)

        app.button(key="comparison_quick_calculate").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        vehicle_resolution, _efficiency_resolution = app.session_state["comparison_quick_results"][
            "fc:900102"
        ][1]
        self.assertTrue(vehicle_resolution.is_ready)
        self.assertLess(vehicle_resolution.resolved_curb_mass_kg, 1500.0)  # QA source curb mass

    def test_three_sibling_quick_scenarios_get_stable_distinct_identities(self):
        # Smoke-matrix Case F: up to 3 siblings from one source, each with
        # its own stable identity, and the 4th slot is refused.
        app = self._app_with_selection()
        for _ in range(3):
            app.button(key="comparison_quick_add_slot").click().run(timeout=90)
            self.assertEqual(len(app.exception), 0)

        self.assertTrue(app.button(key="comparison_quick_add_slot").disabled)
        for slot in (1, 2, 3):
            self.assertIsNotNone(app.text_input(key=f"comparison_quick_label_fc:900102_{slot}"))

        app.button(key="comparison_quick_calculate").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        results = app.session_state["comparison_quick_results"]["fc:900102"]
        self.assertEqual(set(results.keys()), {1, 2, 3})
        scenarios = app.session_state["comparison_quick_scenarios"]["fc:900102"]
        identities = {scenario.identity for scenario in scenarios.values()}
        self.assertEqual(len(identities), 3)
        for scenario in scenarios.values():
            # No Quick -> Quick lineage: every sibling's source is still the
            # real Comparison item, never another Quick Scenario.
            self.assertFalse(scenario.source_identity.startswith("qs:"))

    def test_editing_after_calculate_marks_stale_then_recalculate_keeps_identity(self):
        # Smoke-matrix Case G: calculate, edit an input, confirm the visible
        # "Needs recalculation" state, then recalculate and confirm the same
        # temporary identity is reused (no duplicate item for the slot).
        app = self._app_with_selection()
        app.button(key="comparison_quick_add_slot").click().run(timeout=90)
        app.checkbox(key="comparison_quick_aero_fc:900102_1_enabled").set_value(True).run(timeout=90)
        app.selectbox(key="comparison_quick_aero_fc:900102_1_mode").select(ScalarChangeMode.PERCENT).run(
            timeout=90
        )
        app.button(key="comparison_quick_calculate").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Slot 1 -- Ready" in c.value for c in app.caption))

        scenario_before, vehicle_before = (
            app.session_state["comparison_quick_scenarios"]["fc:900102"][1],
            app.session_state["comparison_quick_results"]["fc:900102"][1][0],
        )
        item_before = build_quick_comparison_item(scenario_before, vehicle_before, None)

        app.number_input(key="comparison_quick_aero_fc:900102_1_value").set_value(3.0).run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Needs recalculation" in c.value for c in app.caption))

        app.button(key="comparison_quick_calculate").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Slot 1 -- Ready" in c.value for c in app.caption))

        scenario_after, vehicle_after = (
            app.session_state["comparison_quick_scenarios"]["fc:900102"][1],
            app.session_state["comparison_quick_results"]["fc:900102"][1][0],
        )
        item_after = build_quick_comparison_item(scenario_after, vehicle_after, None)
        self.assertTrue(vehicle_after.is_ready)
        self.assertNotEqual(vehicle_before.resolved_cda_m2, vehicle_after.resolved_cda_m2)
        # Same slot -> same synthetic identity, even though the physical
        # result genuinely changed -- recalculation replaces, never
        # duplicates.
        self.assertEqual(item_before.vde_id, item_after.vde_id)

    def test_switching_active_source_preserves_other_sources_quick_scenarios(self):
        app = self._app_with_selection()
        app.button(key="comparison_quick_add_slot").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        app.checkbox(key="comparison_quick_aero_fc:900102_1_enabled").set_value(True).run(timeout=90)
        app.selectbox(key="comparison_quick_aero_fc:900102_1_mode").select(ScalarChangeMode.PERCENT).run(
            timeout=90
        )
        app.button(key="comparison_quick_calculate").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertIn(1, app.session_state["comparison_quick_scenarios"]["fc:900102"])

        app.selectbox(key="comparison_quick_source_select").select("fc:900104").run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        # The other source's already-calculated slot is untouched -- Quick
        # Scenarios are additive across sources, not replaced when the
        # active editing source changes.
        self.assertIn(1, app.session_state["comparison_quick_scenarios"]["fc:900102"])


if __name__ == "__main__":
    unittest.main()
