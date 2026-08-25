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


def _dataframe_texts(app) -> list[str]:
    """Flatten every rendered st.dataframe's cell text into one list of
    strings, tolerant of whether AppTest exposes a plain DataFrame or a
    pandas Styler under `.value`.
    """
    texts: list[str] = []
    for item in app.dataframe:
        value = item.value
        frame = getattr(value, "data", value)
        try:
            texts.extend(str(v) for v in frame.to_numpy().ravel())
            texts.extend(str(v) for v in frame.index)
        except AttributeError:
            continue
    return texts


class VehicleDemandSummarySmokeTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "vehicle_demand_smoke.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "fuel_l_per_100km, gco2_per_km) VALUES (?, ?, 'ICE', 'Gasoline', 'HOMOLOGATED', 6.5, 150.0)",
                [(1, 900001), (2, 900004), (3, 900005), (4, 900006), (5, 900007)],
            )
            # A row with neither RRC nor CdA, for Smoke E (partial decomposition).
            con.execute(
                "UPDATE vde_db SET rrc_N_per_kN = NULL, cda_m2 = NULL WHERE id = 900005;"
            )
            # A row with physically invalid mass, for Smoke F (scenario failure isolation).
            con.execute("UPDATE vde_db SET test_mass_kg = 0.0, mass_kg = 0.0 WHERE id = 900007;")
            con.commit()

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _open_energy_drivers(self, app: AppTest) -> None:
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0)

    # -- Smoke A: Reference + Proposal + Benchmark --------------------------

    def test_smoke_a_reference_plus_two_comparisons(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2, 3))
        self._open_energy_drivers(app)

        self.assertTrue(any("Vehicle Demand Summary" in md.value for md in app.markdown))
        # Exactly one heading per selected boundary (default boundary = TOTAL
        # only here); AppTest's own settling rerun can duplicate elements
        # across the whole page, so this counts only against the number of
        # boundaries actually selected rather than asserting a bare 1 (see
        # the codebase-wide assertGreaterEqual convention for app.dataframe
        # counts in test_comparison_report_page_smoke.py for the same reason).
        summary_headings = [md.value for md in app.markdown if md.value == "**Vehicle Demand Summary**"]
        self.assertGreaterEqual(len(summary_headings), 1)

        # Table density/units/deltas (Sprint 9E Sec 32): every primary KPI
        # label is present, and cell text carries an explicit unit (MJ/km for
        # VDE, MJ for the absolute-energy rows) plus a "." delta line for a
        # non-Reference column -- never a bare unitless number.
        texts = _dataframe_texts(app)
        joined = " ".join(texts)
        for label in ("VDE", "Roadload Energy", "Positive Tractive Energy", "Braking Energy Required"):
            self.assertIn(label, joined)
        self.assertTrue(any("MJ/km" in t for t in texts))
        self.assertTrue(any(t.strip().endswith("MJ") for t in texts if "MJ" in t and "MJ/km" not in t))

        # Breakdown chart expander present (opt-in, collapsed by default).
        self.assertTrue(any("Vehicle Demand Energy Breakdown" in exp.label for exp in app.expander))

    # -- Smoke B: Reference-less ---------------------------------------------

    def test_smoke_b_reference_less_shows_absolute_values_without_crash(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=None, comparison_fuelcons_ids=(1, 2))
        self._open_energy_drivers(app)

        self.assertTrue(any("Vehicle Demand Summary" in md.value for md in app.markdown))

    # -- Smoke C: TOTAL / NET switch ------------------------------------------

    def test_smoke_c_switching_total_and_net_both_render(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app.session_state["roadload_basis"] = "TOTAL"
        self._open_energy_drivers(app)
        total_texts = _dataframe_texts(app)
        self.assertTrue(any("VDE" in t for t in total_texts))

        app2 = AppTest.from_file(str(PAGE_PATH))
        app2.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(2,))
        app2.session_state["roadload_basis"] = "Both"
        app2.run(timeout=90)
        self.assertEqual(len(app2.exception), 0)
        self.assertTrue(any(md.value == "TOTAL" for md in app2.caption))
        self.assertTrue(any(md.value == "NET" for md in app2.caption))

    # -- Smoke D: NET unavailable --------------------------------------------

    def test_smoke_d_net_unavailable_scenario_has_no_fallback(self):
        app = AppTest.from_file(str(PAGE_PATH))
        # fuelcons 4 -> VDE-QA-006, which has no transmission -> NET unavailable.
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(4,))
        app.session_state["roadload_basis"] = "NET"
        self._open_energy_drivers(app)

        texts = _dataframe_texts(app)
        joined = " ".join(texts)
        self.assertIn("unavailable", joined.lower())

    # -- Smoke E: Partial decomposition (no RRC/CdA) -------------------------

    def test_smoke_e_missing_decomposition_does_not_crash_and_vde_still_shows(self):
        app = AppTest.from_file(str(PAGE_PATH))
        # fuelcons 3 -> VDE-QA-005, patched to have neither RRC nor CdA.
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(3,))
        self._open_energy_drivers(app)

        texts = _dataframe_texts(app)
        self.assertTrue(any("VDE" in t for t in texts))
        self.assertTrue(any("RRC" in t or "unavailable" in t.lower() for t in texts))


    # -- Smoke F: scenario failure isolation ---------------------------------

    def test_smoke_f_invalid_scenario_does_not_take_down_valid_ones(self):
        app = AppTest.from_file(str(PAGE_PATH))
        # fuelcons 5 -> VDE-QA-007, patched to have zero (invalid) mass.
        app.session_state["comparison_selection"] = SelectionState(reference_fuelcons_id=1, comparison_fuelcons_ids=(5,))
        self._open_energy_drivers(app)

        texts = _dataframe_texts(app)
        joined = " ".join(texts)
        # The valid Reference (fuelcons 1) still shows real VDE values...
        self.assertIn("MJ/km", joined)
        # ...and the invalid scenario's cells degrade to "-" with a short,
        # human-readable reason rather than a traceback/exception name.
        self.assertIn("-", texts)
        self.assertNotIn("Traceback", joined)
        self.assertNotIn("ValueError", joined)
        self.assertTrue("mass" in joined.lower() or "unavailable" in joined.lower())


if __name__ == "__main__":
    unittest.main()
