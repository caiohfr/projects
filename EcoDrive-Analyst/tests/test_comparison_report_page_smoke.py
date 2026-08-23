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

    def test_dashboard_tab_renders_reference_summary(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Reference summary" in md.value for md in app.markdown))
        self.assertTrue(any("FE × VDE" in md.value for md in app.markdown))

    def test_roadload_tab_renders_linked_vde_mode(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Roadload ABC" in md.value for md in app.markdown))
        self.assertTrue(any("Roadload force curve" in md.value for md in app.markdown))

    def test_roadload_direct_vde_mode_works_without_fuelcons(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["roadload_source_mode"] = "Select VDEs directly"
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

    def test_explore_placeholder_unchanged(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Explore Lite is planned for a future package." in info.value for info in app.info))


if __name__ == "__main__":
    unittest.main()
