from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_core import db as db_module
from src.vde_core.qa_mock_data import seed_qa_database


PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Database_Management.py"


class DatabaseManagementUi7CTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temporary_directory.name) / "database_management_ui_7c.db"
        seed_qa_database(self.db_path, overwrite=False)
        self._original_path = db_module.current_db_path()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temporary_directory.cleanup()

    def test_page_renders_all_management_tabs_against_runtime_db(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["ctx"] = {"db_path": str(self.db_path)}
        app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Database Management" in heading.value for heading in app.title))
        self.assertGreaterEqual(len(app.tabs), 4)
        self.assertGreaterEqual(len(app.dataframe), 1)


if __name__ == "__main__":
    unittest.main()
