"""Sprint 11F AppTest coverage for selective legacy-tool routing."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import seed_qa_database, seed_qa_fuelcons_mock_rows
from src.vde_app.components import legacy_engineering_tools
from src.vde_app.components import pwt_fuel_energy


PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Legacy_Engineering_Tools.py"


class _RadioStub:
    def __init__(self, selections: dict[str, str]):
        self.selections = selections

    def radio(self, label, *_args, **_kwargs):
        return self.selections[label]

    def title(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass


class LegacyEngineeringToolsAppTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "legacy_engineering_tools.db"
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
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_legacy_destination_starts_unloaded_and_labeled(self):
        app = self._app()
        self.assertTrue(any("Legacy & Engineering Tools" in item.value for item in app.title))
        self.assertTrue(any("Legacy / compatibility workspace" in item.value for item in app.warning))
        self.assertIsNotNone(app.radio(key="legacy_engineering_area"))
        self.assertFalse(any(item.label == "Active VDE snapshot" for item in app.selectbox))
        self.assertFalse(any(item.label == "Legacy comparison workflow" for item in app.radio))

    def test_powertrain_legacy_is_reachable_only_after_selection(self):
        app = self._app()
        app.radio(key="legacy_engineering_area").set_value("Powertrain Legacy").run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(item.label == "Active VDE snapshot" for item in app.selectbox))
        self.assertFalse(any(item.label == "Legacy comparison workflow" for item in app.radio))

    def test_comparison_legacy_does_not_render_powertrain_legacy(self):
        app = self._app()
        app.radio(key="legacy_engineering_area").set_value("Comparison Legacy").run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(item.label == "Legacy comparison workflow" for item in app.radio))
        self.assertFalse(any(item.label == "Active VDE snapshot" for item in app.selectbox))


class LegacyRenderingDispatchTests(unittest.TestCase):
    def test_legacy_area_dispatches_only_the_selected_renderer(self):
        streamlit = _RadioStub({"Legacy area": "Comparison Legacy"})
        with (
            patch.object(legacy_engineering_tools, "st", streamlit),
            patch.object(legacy_engineering_tools, "_legacy_notice"),
            patch.object(legacy_engineering_tools, "_render_powertrain_legacy") as powertrain,
            patch.object(legacy_engineering_tools, "_render_comparison_legacy") as comparison,
            patch.object(legacy_engineering_tools, "_render_estimate_management") as estimates,
        ):
            legacy_engineering_tools.render_legacy_engineering_tools()

        comparison.assert_called_once_with()
        powertrain.assert_not_called()
        estimates.assert_not_called()

    def test_legacy_comparison_dispatches_only_the_selected_subworkflow(self):
        streamlit = _RadioStub({"Legacy comparison workflow": "Saved Estimates"})
        with (
            patch.object(pwt_fuel_energy, "st", streamlit),
            patch.object(pwt_fuel_energy, "_render_comparison_report_overview") as overview,
            patch.object(pwt_fuel_energy, "render_scorecard_panel") as scorecard,
            patch.object(pwt_fuel_energy, "render_analysis_lab_panel") as analysis,
            patch.object(pwt_fuel_energy, "render_benchmark_regulatory_panel") as benchmark,
            patch.object(pwt_fuel_energy, "render_saved_scenarios_panel") as saved,
        ):
            pwt_fuel_energy.render_legacy_comparison_workspace(1, {"id": 1})

        saved.assert_called_once_with(1)
        overview.assert_not_called()
        scorecard.assert_not_called()
        analysis.assert_not_called()
        benchmark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
