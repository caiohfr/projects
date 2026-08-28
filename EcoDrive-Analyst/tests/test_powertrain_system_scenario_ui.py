"""Sprint 11D AppTest coverage for the live Powertrain page."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import seed_qa_database, seed_qa_fuelcons_mock_rows
from src.vde_core.system_scenario import ArchitectureClass, DomainKind, SolverReadiness
from src.vde_app.components.pwt_fuel_energy import _vde_snapshot_options


PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Powertrain_Scenario.py"


class PowertrainSystemScenarioAppTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "powertrain_system_scenario.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(self.db_path)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _app(self) -> AppTest:
        app = AppTest.from_file(str(PAGE_PATH))
        labels, _ = _vde_snapshot_options()
        app.session_state["pwt_active_vde_source"] = next(
            label for label in labels if label.startswith("#900008 ")
        )
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_current_only_renders_compact_matrix_and_calculates(self):
        app = self._app()
        self.assertTrue(any("Multi-domain System Scenarios" in item.value for item in app.subheader))
        self.assertTrue(any("Vehicle Demand" in str(frame.value) for frame in app.dataframe))
        app.button(key="pwt_ss:calculate").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        calculation = app.session_state["pwt_ss_calculations"]["SYS-CURRENT"]
        self.assertIs(
            calculation.readiness,
            SolverReadiness.READY,
            (
                app.session_state["pwt_ss_drafts"][0].vde_id,
                calculation.result.resolved_scenario.issues,
                calculation.result.resolved_scenario.fuel_estimate_request.powertrain_features,
            )
            if calculation.result
            else calculation.programming_error,
        )

    def test_current_plus_three_proposals_and_fourth_is_prevented(self):
        app = self._app()
        for _ in range(3):
            app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
            self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["pwt_ss_drafts"]), 4)
        self.assertTrue(app.button(key="pwt_ss:add_proposal").disabled)

    def test_visible_label_changes_without_changing_stable_identity(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        before = app.session_state["pwt_ss_drafts"][1].identity
        app.text_input(key="pwt_ss:SYS-P1:label").set_value("Efficiency concept").run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        after = app.session_state["pwt_ss_drafts"][1].identity
        self.assertEqual(before, after)
        self.assertEqual(app.session_state["pwt_ss_drafts"][1].label, "Efficiency concept")

    def test_proposal_selects_a_different_vehicle_demand(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        app.selectbox(key="pwt_ss:editor:scenario").select("SYS-P1").run(timeout=90)
        app.selectbox(key="pwt_ss:editor:domain").set_value(DomainKind.VEHICLE_DEMAND).run(timeout=90)
        widget = app.selectbox(key="pwt_ss:SYS-P1:VEHICLE_DEMAND:vde_id")
        current = app.session_state["pwt_ss_drafts"][1].vde_id
        alternative = 900001 if current != 900001 else 900002
        widget.set_value(alternative).run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        drafts = app.session_state["pwt_ss_drafts"]
        self.assertNotEqual(drafts[0].vde_id, drafts[1].vde_id)

    def test_bev_architecture_updates_engine_to_na_and_keeps_partial_results(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        app.selectbox(key="pwt_ss:editor:scenario").select("SYS-P1").run(timeout=90)
        app.selectbox(key="pwt_ss:editor:domain").set_value(DomainKind.ARCHITECTURE).run(timeout=90)
        app.selectbox(key="pwt_ss:SYS-P1:ARCHITECTURE:architecture").set_value(ArchitectureClass.BEV).run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        proposal = app.session_state["pwt_ss_drafts"][1]
        self.assertEqual(proposal.selection_for(DomainKind.ENGINE_FUEL_CONVERTER), "NOT_APPLICABLE")
        app.button(key="pwt_ss:calculate").click().run(timeout=90)
        calculations = app.session_state["pwt_ss_calculations"]
        self.assertIs(calculations["SYS-CURRENT"].readiness, SolverReadiness.READY)
        self.assertIs(calculations["SYS-P1"].readiness, SolverReadiness.NOT_READY)

    def test_edit_after_calculation_marks_only_that_identity_stale(self):
        app = self._app()
        app.button(key="pwt_ss:calculate").click().run(timeout=90)
        app.selectbox(key="pwt_ss:editor:domain").set_value(DomainKind.VEHICLE_DEMAND).run(timeout=90)
        widget = app.selectbox(key="pwt_ss:SYS-CURRENT:VEHICLE_DEMAND:vde_id")
        current = app.session_state["pwt_ss_drafts"][0].vde_id
        alternative = 900001 if current != 900001 else 900002
        widget.set_value(alternative).run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Needs recalculation" in item.value for item in app.warning))
        app.button(key="pwt_ss:calculate").click().run(timeout=90)
        self.assertEqual(app.session_state["pwt_ss_calculations"]["SYS-CURRENT"].scenario_id, "SYS-CURRENT")

    def test_legacy_evidence_capability_is_opt_in_not_second_default_workflow(self):
        app = self._app()
        self.assertEqual(len([button for button in app.button if button.label == "Confirm baseline"]), 0)
        app.checkbox(key="pwt_ss_load_evidence_tools").set_value(True).run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len([button for button in app.button if button.label == "Confirm baseline"]), 1)


if __name__ == "__main__":
    unittest.main()
