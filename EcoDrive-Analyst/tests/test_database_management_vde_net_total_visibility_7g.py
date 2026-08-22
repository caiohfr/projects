from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_app.components.database_management import _DETAIL_FIELDS
from src.vde_core import db as db_module
from src.vde_core.database_management_contract import EntityType, LOCAL_ADMIN_ACTOR
from src.vde_core.database_management_policy import FieldAccess, field_access_for
from src.vde_core.qa_mock_data import seed_qa_database
from src.vde_core.vde_net_total_normalization import (
    apply_vde_net_total_normalization,
    preview_vde_net_total_normalization,
)


PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Database_Management.py"


class DatabaseManagementVdeNetTotalVisibilityTests(unittest.TestCase):
    def test_vde_detail_fields_include_read_only_total_net_and_review_status(self):
        fields = _DETAIL_FIELDS[EntityType.VDE]
        for field in ("vde_total_mj_per_km", "vde_net_mj_per_km", "review_status"):
            with self.subTest(field=field):
                self.assertIn(field, fields)
                self.assertNotEqual(
                    field_access_for(EntityType.VDE, "LEGACY", field), FieldAccess.EDITABLE
                )
                self.assertNotEqual(
                    field_access_for(EntityType.VDE, "VDE_SETUP", field), FieldAccess.EDITABLE
                )

    def test_page_renders_normalized_vde_record_without_exception(self):
        temp_dir = tempfile.TemporaryDirectory()
        original_path = db_module.current_db_path()
        self.addCleanup(temp_dir.cleanup)
        self.addCleanup(gc.collect)
        self.addCleanup(db_module.configure_db_path, original_path)
        db_path = Path(temp_dir.name) / "database_management_vde_visibility.db"
        seed_qa_database(db_path, overwrite=False)

        with db_module.using_db_path(db_path):
            preview = preview_vde_net_total_normalization()
            apply_vde_net_total_normalization(
                preview, LOCAL_ADMIN_ACTOR, reason="test_visibility"
            )

        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["ctx"] = {"db_path": str(db_path)}
        app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
