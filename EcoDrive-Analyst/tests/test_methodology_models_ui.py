from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.vde_core.vde_calc import compute_vde_net, compute_vde_series
from src.vde_core.vehicle_demand import (
    AmbientState,
    RoadloadBasis,
    RoadloadCoefficients,
    VehicleDemandRequest,
    build_vehicle_demand_profile,
    summarize_vehicle_demand,
)


PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "Methodology_Models.py"


class MethodologyModelsUiTests(unittest.TestCase):
    def _app(self) -> AppTest:
        app = AppTest.from_file(str(PAGE_PATH))
        app.run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_page_exposes_seven_topic_internal_index(self):
        app = self._app()
        self.assertTrue(any(item.value == "Methodology & Models" for item in app.title))
        selector = app.radio(key="methodology_topic")
        self.assertEqual(
            selector.options,
            [
                "Vehicle Demanded Energy",
                "Roadload & Coastdown",
                "Powertrain L0",
                "Tires & Components",
                "Contracts / TOTAL vs NET",
                "Experimental Methods",
                "References & Provenance",
            ],
        )

    def test_total_net_topic_uses_current_contract_and_removes_obsolete_direction(self):
        app = self._app()
        app.radio(key="methodology_topic").set_value("Contracts / TOTAL vs NET").run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        surface = "\n".join(str(item.value) for item in [*app.markdown, *app.table])
        self.assertIn("authoritative coastdown", surface.lower())
        self.assertIn("Never substituted with TOTAL", surface)
        self.assertNotIn("VDE_NET represents the ideal tractive energy", surface)
        self.assertNotIn("Add Transmission Losses?", surface)

    def test_vehicle_demand_formula_matches_positive_power_integration(self):
        app = self._app()
        formulae = "\n".join(str(item.value) for item in app.latex)
        surface = "\n".join(str(item.value) for item in app.markdown)
        self.assertIn(r"\max(0", formulae)
        self.assertIn(r"\int_{0}^{T}", formulae)
        self.assertIn("Negative tractive power", surface)
        self.assertIn("no regenerative-energy credit", surface)

        cycle = pd.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "v": [0.0, 10.0, 10.0, 0.0]})
        t = cycle["t"].to_numpy(float)
        v = cycle["v"].to_numpy(float)
        series = compute_vde_series(t, v, 100.0, 0.0, 0.0, 1_500.0)
        self.assertTrue(np.any(series["P_W"] < 0.0))
        expected_energy_j = np.trapezoid(np.clip(series["P_W"], 0.0, None), t)
        expected_distance_km = np.trapezoid(v, t) / 1_000.0
        result = compute_vde_net(cycle, 100.0, 0.0, 0.0, 1_500.0)
        self.assertAlmostEqual(result["MJ_km"], expected_energy_j / 1e6 / expected_distance_km)
        request = VehicleDemandRequest(
            source_kind="TEST",
            vde_id=None,
            fuelcons_id=None,
            label="Methodology parity",
            cycle_name=None,
            cycle_source="synthetic regression",
            cycle_version="1",
            test_mass_kg=1_500.0,
            roadload_total=RoadloadCoefficients(100.0, 0.0, 0.0),
            roadload_net=None,
            rrc_n_per_kn=None,
            cda_m2=None,
            ambient=AmbientState(),
        )
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        summary = summarize_vehicle_demand(profile, request)
        self.assertGreater(summary.braking_energy_required_MJ, 0.0)
        self.assertAlmostEqual(summary.vde_mj_per_km, result["MJ_km"])

    def test_experimental_topic_does_not_publish_future_capabilities(self):
        app = self._app()
        app.radio(key="methodology_topic").set_value("Experimental Methods").run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        surface = "\n".join(str(item.value) for item in [*app.markdown, *app.table])
        self.assertIn("Planned", surface)
        self.assertIn("Unavailable as a published workflow", surface)


if __name__ == "__main__":
    unittest.main()
