from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.vde_core import db as db_module
from src.vde_core.db import fetchone
from src.vde_core.qa_mock_data import seed_qa_database
from src.vde_app.components.roadload_load_case_lab import (
    build_drive_cycle_profile_figure,
    build_force_curve_figure,
    build_wheel_power_curve_figure,
)


PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Roadload_Load_Case_Lab.py"


class RoadloadLoadCaseLabAppTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "load_case_lab.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _app(self, *, selected_vde_id=None, unit_system=None):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["ctx"] = {"db_path": str(self.db_path)}
        if selected_vde_id is not None:
            app.session_state["roadload_load_case_vde_id"] = selected_vde_id
        if unit_system is not None:
            app.session_state["unit_system"] = unit_system
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_page_exposes_three_modes_and_no_save_action(self):
        app = self._app()
        mode = app.selectbox(key="load_case_analysis_mode")
        self.assertEqual(mode.options, ["Drive Cycle", "Fixed Speed", "Speed Sweep"])
        self.assertTrue(app.checkbox(key="load_case_use_estimated_rrc_prior").value)
        self.assertFalse(any("save" in button.label.lower() for button in app.button))
        self.assertTrue(any("no Save action" in info.value for info in app.info))

    def test_vde_setup_handoff_preselects_baseline(self):
        app = self._app(selected_vde_id=900004)
        selector = app.selectbox(key="load_case_baseline_selector")
        self.assertIn("VDE-900004", selector.value)

    def test_us_unit_system_uses_psi_for_tire_pressure_delta(self):
        app = self._app(unit_system="US customary")
        app.checkbox(key="load_case_tire_pressure_delta__enabled").check().run(timeout=90)
        pressure = app.number_input(key="load_case_tire_pressure_delta__psi")
        self.assertIn("[psi]", pressure.label)

    def test_drive_cycle_calculation_renders_comparison_without_db_write(self):
        before = fetchone("SELECT COUNT(*) AS n FROM vde_db;")["n"]
        app = self._app()
        app.button(key="calculate_roadload_load_case").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        after = fetchone("SELECT COUNT(*) AS n FROM vde_db;")["n"]
        self.assertEqual(before, after)
        self.assertTrue(any("Baseline VDE [MJ/km]" in str(frame.value) for frame in app.dataframe))
        metric_labels = {metric.label for metric in app.metric}
        self.assertTrue(
            {
                "Baseline calculation mass",
                "Baseline basis",
                "Source",
                "Scenario calculation mass",
                "Scenario basis",
                "System mass",
                "Rolling correction",
            }
            <= metric_labels
        )
        result = app.session_state["roadload_load_case_result"]
        cycle_figure = build_drive_cycle_profile_figure(result)
        self.assertEqual(cycle_figure.layout.title.text, "Drive Cycle Dynamic Demand")
        self.assertEqual([cycle_figure.data[0].name, cycle_figure.data[2].name], ["Baseline", "Scenario"])
        self.assertEqual(result.baseline_mass_source_field, "inertia_class")
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Source"), "inertia_class")
        self.assertTrue(any(expander.label == "Roadload Reference Curve" for expander in app.expander))

    def test_estimated_rrc_diagnostic_uses_linked_fuelcons_context(self):
        db_module.update_row(
            "vde_db",
            900004,
            {
                "category": "Midsize Cars",
                "engine_type": None,
                "transmission_type": "AUTOMATIC",
                "rrc_N_per_kN": None,
                "front_tire_id": None,
                "rear_tire_id": None,
            },
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO fuelcons_db (vde_id, electrification, gear_count) VALUES (?, ?, ?)",
                (900004, "BEV", 1),
            )
        before = fetchone("SELECT COUNT(*) AS n FROM fuelcons_db;")["n"]
        app = self._app(selected_vde_id=900004)
        app.button(key="calculate_roadload_load_case").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(fetchone("SELECT COUNT(*) AS n FROM fuelcons_db;")["n"], before)

        result = app.session_state["roadload_load_case_result"]
        diagnostic = result.rolling_state.diagnostic
        self.assertEqual(result.rolling_rrc_source, "RULE_ESTIMATED")
        self.assertEqual(diagnostic.resolved_electrification, "BEV")
        self.assertEqual(diagnostic.gears, 1)
        self.assertEqual(diagnostic.resolved_transmission_key, "SS")
        self.assertEqual(diagnostic.matched_rule_key, "Midsize Cars|BEV|SS")
        rendered_frames = "\n".join(str(frame.value) for frame in app.dataframe)
        for label in (
            "Raw category",
            "Lookup category",
            "Raw electrification",
            "Resolved electrification",
            "Raw transmission",
            "Gears",
            "Resolved transmission",
            "Attempted key",
            "Matched key",
            "Reason",
        ):
            self.assertIn(label, rendered_frames)

    def test_fixed_speed_and_sweep_render_steady_state_outputs(self):
        app = self._app()
        app.selectbox(key="load_case_analysis_mode").set_value("Fixed Speed").run(timeout=90)
        app.button(key="calculate_roadload_load_case").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Wheel energy intensity [MJ/km]" in str(frame.value) for frame in app.dataframe))
        fixed_result = app.session_state["roadload_load_case_result"]
        fixed_figure = build_force_curve_figure(fixed_result)
        self.assertEqual([trace.name for trace in fixed_figure.data[:2]], ["Baseline", "Scenario"])
        marker = next(trace for trace in fixed_figure.data if str(trace.name).startswith("Selected operating point"))
        self.assertAlmostEqual(marker.y[0], fixed_result.steady_state_points[0].total.wheel_force_N, places=9)

        app.selectbox(key="load_case_analysis_mode").set_value("Speed Sweep").run(timeout=90)
        app.button(key="calculate_roadload_load_case").click().run(timeout=90)
        self.assertEqual(len(app.exception), 0)
        sweep_result = app.session_state["roadload_load_case_result"]
        sweep_figure = build_force_curve_figure(sweep_result)
        power_figure = build_wheel_power_curve_figure(sweep_result)
        self.assertEqual([trace.name for trace in sweep_figure.data[:2]], ["Baseline", "Scenario"])
        self.assertEqual([trace.name for trace in power_figure.data[:2]], ["Baseline", "Scenario"])
        self.assertEqual(
            [point.speed_kph for point in sweep_result.force_curve_points],
            [point.speed_kph for point in sweep_result.steady_state_points],
        )


if __name__ == "__main__":
    unittest.main()
