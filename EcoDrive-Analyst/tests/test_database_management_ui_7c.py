from __future__ import annotations

import gc
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_core import db as db_module
PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Database_Management.py"
CANONICAL_QA_PATH = Path(__file__).resolve().parents[1] / "data" / "db" / "eco_drive_qa.db"


class DatabaseManagementUi7CTests(unittest.TestCase):
    def setUp(self):
        self.db_path = CANONICAL_QA_PATH
        self._original_path = db_module.current_db_path()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()

    def test_page_renders_all_management_tabs_against_runtime_db(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["ctx"] = {"db_path": str(self.db_path)}
        app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Database Management" in heading.value for heading in app.title))
        self.assertGreaterEqual(len(app.tabs), 4)
        self.assertGreaterEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.get("file_uploader")), 4)


if __name__ == "__main__":
    unittest.main()
