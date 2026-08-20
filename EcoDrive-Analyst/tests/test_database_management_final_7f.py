from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DatabaseManagementFinalCutoverTests(unittest.TestCase):
    def test_database_management_is_the_active_catalog_page(self) -> None:
        page = PROJECT_ROOT / "pages" / "Database_Management.py"
        self.assertTrue(page.is_file())
        self.assertIn("render_database_management", page.read_text(encoding="utf-8"))

    def test_legacy_tire_editor_is_archived_not_navigable(self) -> None:
        self.assertFalse((PROJECT_ROOT / "pages" / "Tire_Database.py").exists())
        archive = PROJECT_ROOT / "docs" / "archive" / "pages" / "Tire_Database_legacy.py"
        self.assertTrue(archive.is_file())

    def test_public_docs_name_the_canonical_catalog_workflow(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        checkpoint = (PROJECT_ROOT / "docs" / "sprints" / "SPRINT_7_DATABASE_MANAGEMENT.md").read_text(encoding="utf-8")
        self.assertIn("official controlled catalog-administration", readme)
        self.assertIn("canonical catalog-administration", checkpoint)


if __name__ == "__main__":
    unittest.main()
