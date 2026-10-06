from __future__ import annotations

import gc
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_app.comparison_report_viewmodels import SelectionState
from src.vde_core import db as db_module
from src.vde_core.quick_scenario.contracts import ScalarChangeMode, TireTransformMode


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE = (
    ROOT
    / "etl"
    / "data"
    / "staging"
    / "sprint_12f13_vde_materialized"
    / "eco_drive_canonical_vde_materialized_candidate.db"
)
VDE_PAGE = ROOT / "pages" / "VDE_Setup.py"
COMPARISON_PAGE = ROOT / "pages" / "Comparison_Report.py"
POWERTRAIN_PAGE = ROOT / "pages" / "Powertrain_Scenario.py"


class Sprint12GCanonicalApplicationAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_path = db_module.current_db_path()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.integration_db = Path(cls.temp_dir.name) / "canonical_apptest.db"
        shutil.copy2(CANONICAL_SOURCE, cls.integration_db)
        db_module.configure_db_path(cls.integration_db)
        with sqlite3.connect(cls.integration_db) as connection:
            cls.epa = connection.execute(
                "SELECT v.id,f.id,v.make,v.model FROM vde v JOIN fuelcons f ON f.vde_id=v.id "
                "WHERE v.legislation='EPA' ORDER BY f.id LIMIT 1"
            ).fetchone()
            cls.wltp = connection.execute(
                "SELECT v.id,f.id,v.make,v.model FROM vde v JOIN fuelcons f ON f.vde_id=v.id "
                "WHERE v.legislation='WLTP' ORDER BY f.id LIMIT 1"
            ).fetchone()

    @classmethod
    def tearDownClass(cls) -> None:
        db_module.configure_db_path(cls.original_path)
        gc.collect()
        cls.temp_dir.cleanup()

    def _comparison_app(self, *, with_selection: bool) -> AppTest:
        db_module.configure_db_path(self.integration_db)
        app = AppTest.from_file(str(COMPARISON_PAGE))
        app.session_state["ctx"] = {"db_path": str(self.integration_db)}
        if with_selection:
            app.session_state["comparison_selection"] = SelectionState(
                reference_fuelcons_id=int(self.epa[1]),
                comparison_fuelcons_ids=(int(self.wltp[1]),),
            )
        app.run(timeout=120)
        self.assertEqual(len(app.exception), 0, [str(item.value) for item in app.exception])
        return app

    def test_apptest_01_browse_loads_canonical_catalog(self) -> None:
        app = self._comparison_app(with_selection=False)
        surface = "\n".join(str(frame.value) for frame in app.dataframe)
        self.assertIn("Fuelcons ID", surface)
        self.assertTrue(any("Matching scenarios" in metric.label for metric in app.metric))

    def test_apptest_02_vde_setup_opens_canonical_rows(self) -> None:
        db_module.configure_db_path(self.integration_db)
        app = AppTest.from_file(str(VDE_PAGE))
        app.session_state["ctx"] = {"db_path": str(self.integration_db)}
        app.run(timeout=120)
        self.assertEqual(len(app.exception), 0, [str(item.value) for item in app.exception])
        text = "\n".join([item.value for item in app.caption] + [item.value for item in app.markdown])
        self.assertIn("Baseline", text)

    def test_apptest_03_comparison_renders_selected_epa_and_wltp(self) -> None:
        app = self._comparison_app(with_selection=True)
        rendered = "\n".join(str(frame.value) for frame in app.dataframe)
        self.assertIn(str(self.epa[3]), rendered)
        self.assertIn(str(self.wltp[3]), rendered)
        selection = app.session_state["comparison_selection"]
        self.assertEqual(selection.reference_fuelcons_id, int(self.epa[1]))
        self.assertIn(int(self.wltp[1]), selection.comparison_fuelcons_ids)

    def test_apptest_04_quick_scenario_calculates_from_canonical_source(self) -> None:
        app = self._comparison_app(with_selection=True)
        app.button(key="comparison_quick_add_slot").click().run(timeout=120)
        self.assertEqual(len(app.exception), 0, [str(item.value) for item in app.exception])
        mass_prefix = f"comparison_quick_mass_fc:{self.epa[1]}_1"
        app.radio(key=f"{mass_prefix}_mode").set_value(
            "Target curb-to-TWC / WLTP mass line"
        ).run(timeout=120)
        app.selectbox(key=f"{mass_prefix}_scalar_mode").select(
            ScalarChangeMode.DELTA
        ).run(timeout=120)
        app.number_input(key=f"{mass_prefix}_scalar_value").set_value(-25.0).run(timeout=120)
        tire_prefix = f"comparison_quick_tire_fc:{self.epa[1]}_1"
        app.selectbox(key=f"{tire_prefix}_transform_mode").select(
            TireTransformMode.NONE
        ).run(timeout=120)
        app.button(key="comparison_quick_calculate").click().run(timeout=120)
        self.assertEqual(len(app.exception), 0, [str(item.value) for item in app.exception])
        resolution, _ = app.session_state["comparison_quick_results"][f"fc:{self.epa[1]}"][1]
        self.assertTrue(resolution.is_ready, resolution.issues)

    def test_apptest_05_powertrain_scenario_loads_canonical_baseline(self) -> None:
        db_module.configure_db_path(self.integration_db)
        app = AppTest.from_file(str(POWERTRAIN_PAGE))
        app.session_state["ctx"] = {"db_path": str(self.integration_db)}
        app.run(timeout=120)
        self.assertEqual(len(app.exception), 0, [str(item.value) for item in app.exception])
        self.assertTrue(any(item.label == "Source Baseline" for item in app.selectbox))


if __name__ == "__main__":
    unittest.main()
