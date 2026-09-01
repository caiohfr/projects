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
from src.vde_app.components.pwt_system_scenario import (
    _assumption_availability,
    _driver_label,
    _driver_narrative,
    _metric_delta,
)


PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Powertrain_Scenario.py"


def _fuelcons_id(label: str) -> int:
    return int(label.split(" ", 1)[0].removeprefix("FuelCons-"))


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

    def _app(self, *, vde_prefix: str = "#900008 ") -> AppTest:
        app = AppTest.from_file(str(PAGE_PATH))
        labels, _ = _vde_snapshot_options()
        app.session_state["pwt_active_vde_source"] = next(
            label for label in labels if label.startswith(vde_prefix)
        )
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        selector = app.selectbox(key="pwt_ss:current_baseline")
        selected = next(
            option for option in selector.options if f"VDE-{vde_prefix[1:7]}" in option
        )
        if selector.value != selected:
            selector.set_value(_fuelcons_id(selected)).run(timeout=90)
        return app

    def test_primary_workspace_is_canonical_without_legacy_renderers(self):
        app = self._app()
        self.assertTrue(any(item.label == "Source Baseline" for item in app.selectbox))
        self.assertTrue(any(item.value == "Source Baseline" for item in app.subheader))
        self.assertTrue(any("Effective Current" in item.value for item in app.markdown))
        self.assertFalse(
            any("Current System Baseline" in item.value for item in app.subheader)
        )
        self.assertFalse(
            any("Current System Baseline" in item.value for item in app.markdown)
        )
        self.assertTrue(any(item.label == "FuelCons" for item in app.metric))
        availability_surface = "\n".join(
            [item.value for item in app.caption]
            + [item.value for item in app.markdown]
        )
        self.assertIn("Electric-path efficiency", availability_surface)
        self.assertIn("Utility factor", availability_surface)
        self.assertGreaterEqual(availability_surface.count("Not applicable"), 2)
        self.assertTrue(any(item.label == "Fuel" and item.value == "—" for item in app.metric))
        self.assertTrue(any("Multi-domain System Scenarios" in item.value for item in app.subheader))
        self.assertTrue(any("Vehicle Demand" in str(frame.value) for frame in app.dataframe))
        self.assertFalse(any("legacy" in item.label.lower() for item in app.expander))
        self.assertFalse(any(item.label == "Active VDE snapshot" for item in app.selectbox))
        self.assertFalse(any(item.label == "Baseline powertrain source" for item in app.selectbox))
        self.assertFalse(any("Load source pairing" in item.label for item in app.checkbox))

    def test_fuelcons_baseline_options_exclude_unlinked_vde_snapshots(self):
        app = self._app()
        selector = app.selectbox(key="pwt_ss:current_baseline")
        self.assertFalse(any("VDE-900007" in option for option in selector.options))
        self.assertTrue(any("Vehicle Demand" in str(frame.value) for frame in app.dataframe))
        app.button(key="pwt_ss:calculate").click().run(timeout=90)
        calculation = app.session_state["pwt_ss_calculations"]["SYS-CURRENT"]
        self.assertIs(calculation.readiness, SolverReadiness.READY)

    def test_canonical_result_keeps_technical_trace_without_legacy_footer(self):
        app = self._app()
        self.assertFalse(any(item.label == "Technical audit and diagnostics" for item in app.expander))
        app.button(key="pwt_ss:calculate").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(item.label == "PSE" and item.value.endswith("%") for item in app.metric)
        )
        self.assertTrue(any(item.label == "Technical baseline details" for item in app.expander))

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

    def test_calculate_action_remains_available_after_domain_workspace(self):
        app = self._app()
        self.assertTrue(any(item.key == "pwt_ss:calculate_after_editor" for item in app.button))
        app.button(key="pwt_ss:calculate_after_editor").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertIn("SYS-CURRENT", app.session_state["pwt_ss_calculations"])

    def test_upper_and_lower_calculate_use_the_same_canonical_path(self):
        upper = self._app()
        upper.button(key="pwt_ss:calculate").click().run(timeout=90)
        upper_outputs = dict(
            upper.session_state["pwt_ss_calculations"]["SYS-CURRENT"]
            .result.effective_outputs
        )

        lower = self._app()
        lower.button(key="pwt_ss:calculate_after_editor").click().run(timeout=90)
        lower_outputs = dict(
            lower.session_state["pwt_ss_calculations"]["SYS-CURRENT"]
            .result.effective_outputs
        )
        self.assertEqual(upper_outputs, lower_outputs)

    def test_current_plus_three_proposals_and_fourth_is_prevented(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        self.assertTrue(any("INHERIT" in str(frame.value) for frame in app.dataframe))
        for _ in range(2):
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

    def test_demand_driven_story_is_ordered_and_uses_compact_neutral_deltas(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        app.selectbox(key="pwt_ss:editor:scenario").select("SYS-P1").run(timeout=90)
        app.selectbox(key="pwt_ss:editor:domain").set_value(
            DomainKind.VEHICLE_DEMAND
        ).run(timeout=90)
        widget = app.selectbox(key="pwt_ss:SYS-P1:VEHICLE_DEMAND:vde_id")
        current = app.session_state["pwt_ss_drafts"][1].vde_id
        alternative = 900001 if current != 900001 else 900002
        widget.set_value(alternative).run(timeout=90)
        app.button(key="pwt_ss:calculate_after_editor").click().run(timeout=90)

        self.assertEqual(len(app.exception), 0)
        story = "\n".join(
            [item.value for item in app.markdown]
            + [item.value for item in app.caption]
        )
        self.assertIn("RESULT DRIVER", story)
        self.assertIn("DEMAND-DRIVEN", story)
        self.assertLess(story.index("1 · Vehicle Demand"), story.index("2 · Powertrain / PSE"))
        self.assertLess(story.index("2 · Powertrain / PSE"), story.index("3 · Final result"))
        self.assertIn("No represented powertrain-efficiency improvement contributed.", story)
        pse_metrics = [item for item in app.metric if item.label == "PSE"]
        self.assertTrue(any(item.value.endswith("%") for item in pse_metrics))
        self.assertTrue(any(item.delta.endswith(" pp") for item in pse_metrics if item.delta))

    def test_inherited_proposal_stays_compact_until_a_deviation_is_created(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        app.selectbox(key="pwt_ss:editor:scenario").select("SYS-P1").run(timeout=90)
        app.selectbox(key="pwt_ss:editor:domain").set_value(
            DomainKind.ENGINE_FUEL_CONVERTER
        ).run(timeout=90)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("INHERIT" in item.value for item in app.caption))
        self.assertFalse(any(item.label == "Current correction" for item in app.expander))
        app.button(key="pwt_ss:SYS-P1:ENGINE_FUEL_CONVERTER:create").click().run(
            timeout=90
        )
        self.assertNotEqual(
            app.session_state["pwt_ss_drafts"][1].selection_for(
                DomainKind.ENGINE_FUEL_CONVERTER
            ),
            "CURRENT",
        )

    def test_powertrain_driven_story_follows_an_adopted_l0_assumption(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        app.selectbox(key="pwt_ss:editor:scenario").select("SYS-P1").run(timeout=90)
        app.selectbox(key="pwt_ss:editor:domain").set_value(
            DomainKind.ENGINE_FUEL_CONVERTER
        ).run(timeout=90)
        app.button(key="pwt_ss:SYS-P1:ENGINE_FUEL_CONVERTER:create").click().run(
            timeout=90
        )
        app.number_input(key="pwt_ss:proposal:ENG-P01:assumption_value").set_value(
            0.4
        ).run(timeout=90)
        app.checkbox(key="pwt_ss:proposal:ENG-P01:adopt").set_value(True).run(
            timeout=90
        )
        app.button(key="pwt_ss:calculate_after_editor").click().run(timeout=90)

        story = "\n".join(
            [item.value for item in app.markdown]
            + [item.value for item in app.caption]
        )
        self.assertEqual(len(app.exception), 0)
        self.assertIn("RESULT DRIVER", story)
        self.assertIn("POWERTRAIN-DRIVEN", story)
        self.assertIn("Vehicle Demand remained unchanged", story)
        self.assertIn("Adopted L0 impacts", story)

    def test_inherited_proposal_has_a_concise_no_change_story(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        app.button(key="pwt_ss:calculate_after_editor").click().run(timeout=90)
        story = "\n".join(
            [item.value for item in app.markdown]
            + [item.value for item in app.caption]
        )
        self.assertEqual(len(app.exception), 0)
        self.assertIn("NO QUANTITATIVE CHANGE", story)
        self.assertNotIn("Adopted L0 impacts", story)

    def test_configuration_only_story_says_not_represented(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        app.selectbox(key="pwt_ss:editor:scenario").select("SYS-P1").run(timeout=90)
        app.selectbox(key="pwt_ss:editor:domain").set_value(
            DomainKind.TRANSMISSION_DRIVELINE
        ).run(timeout=90)
        app.button(
            key="pwt_ss:SYS-P1:TRANSMISSION_DRIVELINE:create"
        ).click().run(timeout=90)
        final_drive = app.number_input(
            key="pwt_ss:proposal:TRANS-P01:final_drive_ratio"
        )
        final_drive.set_value(float(final_drive.value) + 0.1).run(timeout=90)
        app.button(key="pwt_ss:calculate_after_editor").click().run(timeout=90)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(
                "CONFIGURATION ONLY" in item.value
                and "NOT REPRESENTED" in item.value
                for item in app.info
            )
        )
        self.assertTrue(
            any("Not represented" in str(frame.value) for frame in app.dataframe)
        )

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

    def test_current_baseline_change_materializes_current_source_before_engine_edit(self):
        app = self._app()
        selector = app.selectbox(key="pwt_ss:current_baseline")
        selected = next(option for option in selector.options if "VDE-900001" in option)
        selector.set_value(_fuelcons_id(selected)).run(timeout=90)
        self.assertTrue(any("resets domain proposals" in item.value for item in app.warning))
        app.button(key="pwt_ss:confirm_baseline_change").click().run(timeout=90)
        self.assertEqual(app.session_state["pwt_ss_drafts"][0].vde_id, 900001)
        self.assertIsNotNone(app.session_state["pwt_ss_drafts"][0].fuelcons_id)

        app.selectbox(key="pwt_ss:editor:domain").set_value(
            DomainKind.ENGINE_FUEL_CONVERTER
        ).run(timeout=90)
        self.assertEqual(len(app.exception), 0)

        app.button(key="pwt_ss:calculate").click().run(timeout=90)
        calculation = app.session_state["pwt_ss_calculations"]["SYS-CURRENT"]
        self.assertEqual(calculation.result.selected_vehicle_demand_identity, "vde:900001")
        self.assertIs(calculation.readiness, SolverReadiness.READY)

    def test_current_correction_is_reachable_and_separate_from_proposals(self):
        app = self._app()
        app.selectbox(key="pwt_ss:editor:domain").set_value(
            DomainKind.ENGINE_FUEL_CONVERTER
        ).run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(item.label == "Current correction" for item in app.expander))
        self.assertFalse(
            any(item.label == "Evidence source" for item in app.selectbox),
            "manual entry must not offer decorative ML/Benchmark/Regression provenance",
        )

    def test_baseline_change_resets_proposals_and_results_but_keeps_identities(self):
        app = self._app()
        app.button(key="pwt_ss:add_proposal").click().run(timeout=90)
        app.button(key="pwt_ss:calculate").click().run(timeout=90)
        identities_before = [draft.identity for draft in app.session_state["pwt_ss_drafts"]]

        selector = app.selectbox(key="pwt_ss:current_baseline")
        selected = next(option for option in selector.options if "VDE-900001" in option)
        selector.set_value(_fuelcons_id(selected)).run(timeout=90)
        app.button(key="pwt_ss:confirm_baseline_change").click().run(timeout=90)

        drafts = app.session_state["pwt_ss_drafts"]
        self.assertEqual([draft.identity for draft in drafts], identities_before)
        self.assertTrue(all(draft.vde_id == 900001 for draft in drafts))
        self.assertTrue(
            all(
                selection == "CURRENT"
                for draft in drafts
                for selection in draft.selections.values()
            )
        )
        self.assertEqual(app.session_state["pwt_ss_calculations"], {})

    def test_canonical_page_has_no_legacy_baseline_action(self):
        app = self._app()
        self.assertEqual(len([button for button in app.button if button.label == "Confirm baseline"]), 0)
        self.assertFalse(any(item.key == "pwt_ss_load_evidence_tools" for item in app.checkbox))


class PowertrainSystemScenarioPresentationTests(unittest.TestCase):
    def test_assumption_availability_distinguishes_not_applicable_and_not_provided(self):
        self.assertEqual(
            _assumption_availability(None, applicable=False, digits=2),
            "Not applicable",
        )
        self.assertEqual(
            _assumption_availability(None, applicable=True, digits=2),
            "Not provided",
        )

    def test_pse_delta_uses_percentage_points(self):
        self.assertEqual(
            _metric_delta(0.0091, digits=2, scale=100.0, suffix=" pp"),
            "+0.91 pp",
        )

    def test_all_result_story_labels_and_interpretations_are_explicit(self):
        expectations = {
            "DEMAND-DRIVEN": "Demand-driven",
            "POWERTRAIN-DRIVEN": "Powertrain-driven",
            "MIXED DEMAND + POWERTRAIN": "Mixed demand + powertrain",
            "NO QUANTITATIVE CHANGE": "No quantitative change",
        }
        for driver, label in expectations.items():
            with self.subTest(driver=driver):
                self.assertEqual(_driver_label(driver), label)
                self.assertTrue(_driver_narrative(driver))


if __name__ == "__main__":
    unittest.main()
