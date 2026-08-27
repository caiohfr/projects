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
from src.vde_core.quick_scenario import ScalarChangeMode

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
