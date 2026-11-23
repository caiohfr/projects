from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import seed_qa_database, seed_qa_fuelcons_mock_rows


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class HomePage15AAppTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "home_15a_qa.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        seed_qa_fuelcons_mock_rows(self.db_path)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _app(self) -> AppTest:
        app = AppTest.from_file(str(APP_PATH))
        app.session_state["ctx"] = {"db_path": str(self.db_path)}
        return app

    def test_first_access_has_real_destinations_and_empty_recent_state(self):
        app = self._app().run(timeout=90)

        self.assertEqual(len(app.exception), 0)
        self.assertIsNotNone(app.button(key="home_new_proposal"))
        labels = [link.label for link in app.get("page_link")]
        self.assertIn("VDE Setup", labels)
        self.assertIn("Comparison Report", labels)
        self.assertIn("Methodology & Models", labels)
        self.assertTrue(any("No recoverable recent analysis" in item.value for item in app.info))
        self.assertTrue(app.button(key="home_open_lab_disabled").disabled)

    def test_shared_navigation_registers_all_eight_destinations(self):
        app = self._app().run(timeout=90)

        self.assertEqual(len(app.exception), 0)
        registered_page_paths = (
            "pages/VDE_Setup.py",
            "pages/Roadload_Load_Case_Lab.py",
            "pages/Powertrain_Scenario.py",
            "pages/Comparison_Report.py",
            "pages/Database_Management.py",
            "pages/Methodology_Models.py",
            "pages/Legacy_Engineering_Tools.py",
        )
        for page_path in registered_page_paths:
            with self.subTest(page_path=page_path):
                app.switch_page(page_path)

    def test_active_state_distinguishes_baseline_proposal_and_saved_fuelcons(self):
        app = self._app()
        app.session_state["vde_setup_v22"] = {
            "baseline": {"selected_vde_id": 900001},
            "proposals": [
                {
                    "proposal_id": "requested_1",
                    "name": "QA mass study",
                    "domains": {"mass": {"proposal_type": "MASS_ABSOLUTE"}},
                    "inputs": {"mass": {"mass_kg": 1800}},
                }
            ],
            "preview": {"status": "success"},
        }
        app.session_state["pwt_ss:current_baseline"] = 900102
        app.run(timeout=90)

        self.assertEqual(len(app.exception), 0)
        self.assertIsNotNone(app.button(key="home_continue"))
        self.assertFalse(app.button(key="home_open_lab").disabled)
        markdown = "\n".join(item.value for item in app.markdown)
        self.assertIn("Persisted VDE #900001", markdown)
        self.assertIn("1 temporary proposal", markdown)
        self.assertIn("Persisted FuelCons #900102", markdown)
        self.assertIn("not persisted", markdown)

    def test_stale_database_references_warn_and_do_not_enable_lab(self):
        app = self._app()
        app.session_state["vde_setup_v22"] = {"baseline": {"selected_vde_id": 999999}}
        app.session_state["pwt_ss:current_baseline"] = 999998
        app.run(timeout=90)

        self.assertEqual(len(app.exception), 0)
        warnings = "\n".join(item.value for item in app.warning)
        self.assertIn("VDE #999999 is no longer available", warnings)
        self.assertIn("FuelCons #999998 is no longer available", warnings)
        self.assertTrue(app.button(key="home_open_lab_disabled").disabled)
        self.assertIsNotNone(app.button(key="home_new_proposal"))


if __name__ == "__main__":
    unittest.main()
