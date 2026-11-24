from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE_PATH = ROOT / "pages" / "Database_Management.py"


class DatabaseManagementVisual15DTests(unittest.TestCase):
    def test_page_preserves_four_tabs_and_explicit_review_boundary(self):
        with patch("src.vde_app.components.database_management.browse_records", return_value=[]):
            app = AppTest.from_file(str(PAGE_PATH))
            app.run(timeout=120)
        self.assertEqual(len(app.exception), 0, [str(item.value) for item in app.exception])
        self.assertEqual([tab.label for tab in app.tabs], ["VDE", "Fuel Consumption", "Tires", "Components"])
        surface = "\n".join(item.value for item in app.markdown)
        self.assertIn("Data governance", surface)
        self.assertIn("Controlled data operations", surface)
        self.assertIn("0 pending", surface)
        self.assertIn("Review &amp; Commit", surface)
        self.assertTrue(any("No changes staged" in item.value for item in app.caption))
        self.assertEqual(len(app.get("file_uploader")), 4)


if __name__ == "__main__":
    unittest.main()
