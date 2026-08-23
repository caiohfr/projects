from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_app.comparison_report_viewmodels import SelectionState
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

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def test_page_opens_with_scenarios_available_no_selection(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Select a reference scenario" in info.value for info in app.info))

    def test_reference_selection_builds_dataset_and_renders_scorecard(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.dataframe), 1)

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


if __name__ == "__main__":
    unittest.main()
