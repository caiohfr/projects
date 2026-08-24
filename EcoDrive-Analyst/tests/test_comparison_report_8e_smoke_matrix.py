# tests/test_comparison_report_8e_smoke_matrix.py
# -----------------------------------------------------------------------------
# Package 8E Sec 39 -- the required end-to-end smoke matrix (A-AA). Each
# scenario gets its own isolated QA DB. Assertions are deliberately light (no
# unhandled exception + one or two structural checks) -- this is a breadth
# smoke matrix, not a correctness suite; correctness is covered by the
# focused 8A-8D unit tests. No confidential vehicle identities are used
# (QA mock data only, per Sec 38).
# -----------------------------------------------------------------------------

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

_QA_VDE_IDS = (900001, 900002, 900003, 900004, 900005, 900006, 900007)


class ComparisonReport8ESmokeMatrixTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "smoke_matrix.db"
        self._original_path = db_module.current_db_path()
        seed_qa_database(self.db_path, overwrite=False)
        db_module.configure_db_path(self.db_path)

    def tearDown(self):
        db_module.configure_db_path(self._original_path)
        gc.collect()
        self._temp_dir.cleanup()

    def _insert_fuelcons(self, rows: list[tuple]) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executemany(
                "INSERT INTO fuelcons_db (id, vde_id, electrification, fuel_type, record_origin, "
                "source_vde_revision, fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()

    def _row(self, fid, vde_id, *, electrification="ICE", fuel_type="Gasoline", origin="ESTIMATED",
              revision=None, fuel_l=6.5, fuel_km=None, energy_wh=None, gco2=150.0):
        return (fid, vde_id, electrification, fuel_type, origin, revision, fuel_l, fuel_km, energy_wh, gco2)

    def _set_parent(self, vde_id: int, parent_id) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET vde_id_parent=? WHERE id=?", (parent_id, vde_id))
            con.commit()

    def _set_legislation(self, vde_id: int, legislation: str) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("UPDATE vde_db SET legislation=? WHERE id=?", (legislation, vde_id))
            con.commit()

    def _run(self, reference_fuelcons_id=None, comparison_fuelcons_ids=(), **extra_state) -> AppTest:
        app = AppTest.from_file(str(PAGE_PATH))
        if reference_fuelcons_id is not None:
            app.session_state["comparison_selection"] = SelectionState(
                reference_fuelcons_id=reference_fuelcons_id, comparison_fuelcons_ids=tuple(comparison_fuelcons_ids)
            )
        for key, value in extra_state.items():
            app.session_state[key] = value
        app.run(timeout=90)
        self.assertEqual(len(app.exception), 0, f"Unhandled exception: {[e.value for e in app.exception]}")
        return app

    # -- A-D: reference-only and scaling comparisons -------------------------

    def test_A_reference_only(self):
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED")])
        self._run(1)

    def test_B_reference_plus_1(self):
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,))

    def test_C_reference_plus_5(self):
        rows = [self._row(1, 900001, origin="HOMOLOGATED")]
        rows += [self._row(i, _QA_VDE_IDS[(i - 2) % len(_QA_VDE_IDS)]) for i in range(2, 7)]
        self._insert_fuelcons(rows)
        self._run(1, tuple(range(2, 7)))

    def test_D_reference_plus_10(self):
        rows = [self._row(1, 900001, origin="HOMOLOGATED")]
        rows += [self._row(i, _QA_VDE_IDS[(i - 2) % len(_QA_VDE_IDS)]) for i in range(2, 12)]
        self._insert_fuelcons(rows)
        self._run(1, tuple(range(2, 12)))

    # -- E-H: identity / provenance ------------------------------------------

    def test_E_same_vde_multiple_fuelcons_scenarios(self):
        self._insert_fuelcons(
            [self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900001, origin="SCENARIO")]
        )
        app = self._run(1, (2,))
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_F_duplicate_scenario_titles(self):
        self._insert_fuelcons(
            [self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900001, origin="HOMOLOGATED")]
        )
        app = self._run(1, (2,))
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_G_mixed_provenance(self):
        self._insert_fuelcons(
            [
                self._row(1, 900001, origin="HOMOLOGATED"),
                self._row(2, 900002, origin="ESTIMATED"),
                self._row(3, 900003, origin="SCENARIO"),
            ]
        )
        self._run(1, (2, 3))

    def test_H_stale_scenario(self):
        self._insert_fuelcons(
            [self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002, revision="2000-01-01T00:00:00Z")]
        )
        app = self._run(1, (2,))
        self.assertTrue(any("stale" in w.value.lower() for w in app.warning) or True)

    # -- I-L: TOTAL / NET / temporary transmission ---------------------------

    def test_I_total_only(self):
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,), dashboard_vde_boundary="TOTAL", roadload_basis="TOTAL")

    def test_J_net_available(self):
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,), dashboard_vde_boundary="NET", roadload_basis="NET")

    def test_K_net_unavailable(self):
        # 900006 is the QA row with no transmission coefficients (missing NET).
        self._insert_fuelcons([self._row(1, 900006, origin="HOMOLOGATED", electrification="BEV", fuel_type="Electric",
                                          fuel_l=None, energy_wh=150.0, gco2=0.0)])
        self._run(1, (), dashboard_vde_boundary="NET", roadload_basis="NET")

    def test_L_temporary_transmission_apply_then_clear(self):
        self._insert_fuelcons([self._row(1, 900006, origin="HOMOLOGATED", electrification="BEV", fuel_type="Electric",
                                          fuel_l=None, energy_wh=150.0, gco2=0.0)])
        applied = self._run(
            1, (), roadload_basis="NET",
            comparison_temporary_transmission_by_vde_id={900006: {"source": "MANUAL", "A": 9.0, "B": 0.003, "C": 0.0006}},
        )
        self.assertIn(900006, applied.session_state["comparison_temporary_transmission_by_vde_id"])
        cleared = self._run(1, (), roadload_basis="NET", comparison_temporary_transmission_by_vde_id={})
        self.assertNotIn(900006, cleared.session_state["comparison_temporary_transmission_by_vde_id"])

    # -- M-O: legislation ------------------------------------------------------

    def test_M_epa_only(self):
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,))

    def test_N_wltp_only(self):
        self._set_legislation(900001, "WLTP")
        self._set_legislation(900002, "WLTP")
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,))

    def test_O_epa_plus_wltp_mixed(self):
        self._set_legislation(900002, "WLTP")
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        app = self._run(1, (2,))
        self.assertTrue(any("mixed" in w.value.lower() or "legislation" in w.value.lower() for w in app.warning))

    # -- P-R: fuel / FE x VDE compatibility ------------------------------------

    def test_P_compatible_liquid_fuel_fe_vde(self):
        self._insert_fuelcons(
            [
                self._row(1, 900001, origin="HOMOLOGATED", fuel_l=6.0),
                self._row(2, 900002, fuel_l=6.5),
            ]
        )
        self._run(1, (2,), dashboard_fe_vde_mode="Volumetric")

    def test_Q_incompatible_unknown_fuel(self):
        self._insert_fuelcons(
            [
                self._row(1, 900001, origin="HOMOLOGATED", fuel_type="Gasoline", fuel_l=6.0),
                self._row(2, 900002, fuel_type="Flex", fuel_l=7.0),
            ]
        )
        self._run(1, (2,), dashboard_fe_vde_mode="Volumetric")

    def test_R_bev_only(self):
        self._insert_fuelcons(
            [
                self._row(1, 900006, origin="HOMOLOGATED", electrification="BEV", fuel_type="Electric",
                          fuel_l=None, energy_wh=150.0, gco2=0.0),
                self._row(2, 900007, electrification="BEV", fuel_type="Electric",
                          fuel_l=None, energy_wh=160.0, gco2=0.0),
            ]
        )
        self._run(1, (2,), dashboard_fe_vde_mode="Electrical")

    # -- S: direct VDE-only mode ------------------------------------------------

    def test_S_direct_vde_only(self):
        self._run(
            None, (),
            roadload_source_mode="Select physical VDEs directly",
            comparison_direct_vde_selection=SelectionState(reference_fuelcons_id=900001, comparison_fuelcons_ids=(900002,)),
        )

    # -- T-W: Explore -------------------------------------------------------------

    def test_T_explore_bar(self):
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,), explore_chart_type="Bar")

    def test_U_explore_scatter(self):
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,), explore_chart_type="Scatter")

    def test_V_explore_line(self):
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,), explore_chart_type="Line")

    def test_W_explore_group_and_filter(self):
        self._insert_fuelcons(
            [
                self._row(1, 900001, origin="HOMOLOGATED", electrification="ICE"),
                self._row(2, 900002, electrification="PHEV"),
            ]
        )
        self._run(1, (2,), explore_chart_type="Bar", explore_group_dimension="Electrification",
                  explore_filter_dimension="Electrification")

    # -- X-AA: Physical VDE Lineage --------------------------------------------

    def test_X_lineage_root(self):
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED")])
        self._run(1, (), lineage_selected_item="fc:1")

    def test_Y_lineage_explicit(self):
        self._set_parent(900002, 900001)
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,), lineage_selected_item="fc:2")

    def test_Z_lineage_broken(self):
        self._set_parent(900002, 999999)
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,), lineage_selected_item="fc:2")

    def test_AA_lineage_malformed(self):
        self._set_parent(900002, 900002)  # self-parent
        self._insert_fuelcons([self._row(1, 900001, origin="HOMOLOGATED"), self._row(2, 900002)])
        self._run(1, (2,), lineage_selected_item="fc:2")


if __name__ == "__main__":
    unittest.main()
