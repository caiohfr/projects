from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.vde_app.components.vde_request_compact_viewmodels import build_engineering_comparison_payload
from src.vde_core.phase_aggregation import epa_city_hwy_from_phase, wltp_phases_from_phase
from src.vde_core.test_mass import inertia_class_from_mass
from src.vde_core.vde_calc import compute_vde_net
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_compact_state import (
    apply_v22_baseline,
    apply_v22_domain_inputs,
    apply_v22_proposal_matrix,
    create_v22_state,
)
from src.vde_core.vde_workflow_service import build_vde_setup_preview


def _constant_speed_cycle(speed_mps: float = 10.0, duration_s: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [float(value) for value in range(duration_s + 1)],
            "v": [float(speed_mps)] * (duration_s + 1),
        }
    )


def _linear_accel_cycle(final_speed_mps: float = 10.0, duration_s: int = 10) -> pd.DataFrame:
    step_count = duration_s + 1
    return pd.DataFrame(
        {
            "t": [float(value) for value in range(step_count)],
            "v": [float(final_speed_mps) * value / duration_s for value in range(step_count)],
        }
    )


def _epa_phase_cycle() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "v": [8.0, 9.0, 10.0, 8.0, 9.0, 10.0, 8.0, 9.0, 10.0],
            "phase": ["bag1", "bag1", "bag1", "bag2", "bag2", "bag2", "hwfet", "hwfet", "hwfet"],
        }
    )


def _preview_baseline_row() -> dict:
    return {
        "id": 4998,
        "make": "AUDI",
        "model": "Q6",
        "year": 2027,
        "legislation": "EPA",
        "cycle_name": "FTP75",
        "notes": "Compact baseline row",
        "mass_kg": 1600.0,
        "test_mass_kg": 1736.0,
        "inertia_class": 1750.0,
        "weight_dist_fr_pct": 55.0,
        "cda_m2": 0.62,
        "A": 120.0,
        "B": 0.02,
        "C": 0.01,
        "rrc_N_per_kN": 8.4,
        "front_pressure_psi": 35.0,
        "rear_pressure_psi": 35.0,
        "trans_A_coef_N": 10.0,
        "trans_B_coef_Npkph": 0.005,
        "trans_C_coef_Npkph2": 0.001,
        "brake_A_coef_N": 4.0,
        "brake_B_Npkph": 0.001,
        "brake_C_coef_Npkph2": 0.0001,
    }


class VdePhysicsQaVde1BTests(unittest.TestCase):
    def _workflow_payload(self, **overrides) -> dict:
        payload = {
            "legislation": "EPA",
            "category": "MIDSIZE",
            "make": "FORD",
            "model": "TEST",
            "year": 2025,
            "mass_kg": 1550.0,
            "test_mass_kg": 1686.0,
            "mass_basis": "TEST_MASS",
            "initial_abc_total_source": "manual",
            "initial_abc_total": {"A": 120.0, "B": 0.02, "C": 0.011},
            "cycle_df": _linear_accel_cycle(),
        }
        payload.update(overrides)
        return payload

    def test_combined_roadload_matches_constant_speed_analytical_result(self):
        result = compute_vde_net(_constant_speed_cycle(), 100.0, 2.0, 0.01, 0.0)

        expected_force_n = 100.0 + 2.0 * 36.0 + 0.01 * (36.0**2)
        expected_mj_per_km = expected_force_n * 0.001

        self.assertAlmostEqual(result["km"], 1.0, places=9)
        self.assertAlmostEqual(result["MJ_total"], expected_mj_per_km, places=9)
        self.assertAlmostEqual(result["MJ_km"], expected_mj_per_km, places=9)

    def test_isolated_a_b_and_c_terms_use_kph_inside_polynomial(self):
        cycle = _constant_speed_cycle()

        a_only = compute_vde_net(cycle, 100.0, 0.0, 0.0, 0.0)
        b_only = compute_vde_net(cycle, 0.0, 2.0, 0.0, 0.0)
        c_only = compute_vde_net(cycle, 0.0, 0.0, 0.01, 0.0)

        self.assertAlmostEqual(a_only["MJ_km"], 0.100, places=9)
        self.assertAlmostEqual(b_only["MJ_km"], 0.072, places=9)
        self.assertAlmostEqual(c_only["MJ_km"], 0.01296, places=9)

    def test_positive_acceleration_matches_analytical_kinetic_energy(self):
        result = compute_vde_net(_linear_accel_cycle(final_speed_mps=10.0, duration_s=10), 0.0, 0.0, 0.0, 1000.0)

        expected_mj_total = 0.5 * 1000.0 * (10.0**2) / 1e6
        expected_km = 50.0 / 1000.0
        expected_mj_per_km = expected_mj_total / expected_km

        self.assertAlmostEqual(result["MJ_total"], expected_mj_total, places=9)
        self.assertAlmostEqual(result["km"], expected_km, places=9)
        self.assertAlmostEqual(result["MJ_km"], expected_mj_per_km, places=9)

    def test_distance_integration_handles_constant_and_varying_speed(self):
        constant = compute_vde_net(_constant_speed_cycle(), 0.0, 0.0, 0.0, 0.0)
        varying = compute_vde_net(
            pd.DataFrame({"t": [0.0, 50.0, 100.0], "v": [0.0, 10.0, 20.0]}),
            0.0,
            0.0,
            0.0,
            0.0,
        )

        self.assertAlmostEqual(constant["km"], 1.0, places=9)
        self.assertAlmostEqual(varying["km"], 1.0, places=9)

    def test_invalid_cycles_fail_safely(self):
        cases = [
            pd.DataFrame({"t": [0.0, 1.0, 2.0], "v": [0.0, 0.0, 0.0]}),
            pd.DataFrame({"t": [0.0, 1.0, 1.0, 2.0], "v": [0.0, 1.0, 2.0, 3.0]}),
            pd.DataFrame({"t": [0.0, 2.0, 1.0], "v": [0.0, 1.0, 2.0]}),
            pd.DataFrame({"t": [0.0, 1.0, 2.0], "v": [0.0, float("nan"), 2.0]}),
        ]

        for cycle in cases:
            with self.subTest(cycle=cycle.to_dict(orient="list")):
                with self.assertRaises(ValueError):
                    compute_vde_net(cycle, 0.0, 0.0, 0.0, 0.0)

    def test_epa_aggregation_uses_city_hwy_formula(self):
        groups = {
            "bag1": pd.DataFrame({"t": [0.0, 1.0], "v": [10.0, 10.0]}),
            "bag2": pd.DataFrame({"t": [0.0, 1.0], "v": [20.0, 20.0]}),
            "hwfet": pd.DataFrame({"t": [0.0, 1.0], "v": [30.0, 30.0]}),
        }

        def fake_compute(group, *_args, **_kwargs):
            marker = float(group["v"].iloc[0])
            if marker == 10.0:
                return {"MJ_total": 0.4, "km": 0.2, "MJ_km": 2.0}
            if marker == 20.0:
                return {"MJ_total": 0.6, "km": 0.3, "MJ_km": 2.0}
            return {"MJ_total": 0.5, "km": 1.0, "MJ_km": 0.5}

        with patch("src.vde_core.phase_aggregation.compute_vde_net", side_effect=fake_compute):
            result = epa_city_hwy_from_phase(groups, 0.0, 0.0, 0.0, 1644.0, mass_is_resolved_twc=True)

        self.assertAlmostEqual(result["urb_MJ_km"], 2.0)
        self.assertAlmostEqual(result["hw_MJ_km"], 0.5)
        self.assertAlmostEqual(result["net_comb_MJ_km"], 1.325)

    def test_wltp_aggregation_is_distance_weighted_not_simple_average(self):
        groups = {
            "low": pd.DataFrame({"t": [0.0, 1.0], "v": [10.0, 10.0]}),
            "mid": pd.DataFrame({"t": [0.0, 1.0], "v": [20.0, 20.0]}),
            "high": pd.DataFrame({"t": [0.0, 1.0], "v": [30.0, 30.0]}),
            "xhigh": pd.DataFrame({"t": [0.0, 1.0], "v": [40.0, 40.0]}),
        }

        def fake_compute(group, *_args, **_kwargs):
            marker = float(group["v"].iloc[0])
            if marker == 10.0:
                return {"MJ_total": 1.0, "km": 1.0, "MJ_km": 1.0}
            if marker == 20.0:
                return {"MJ_total": 1.0, "km": 2.0, "MJ_km": 0.5}
            if marker == 30.0:
                return {"MJ_total": 4.0, "km": 1.0, "MJ_km": 4.0}
            return {"MJ_total": 4.0, "km": 6.0, "MJ_km": 4.0 / 6.0}

        with patch("src.vde_core.phase_aggregation.compute_vde_net", side_effect=fake_compute):
            result = wltp_phases_from_phase(groups, 0.0, 0.0, 0.0, 1500.0)

        self.assertAlmostEqual(result["vde_low_mj_per_km"], 1.0)
        self.assertAlmostEqual(result["vde_mid_mj_per_km"], 0.5)
        self.assertAlmostEqual(result["vde_high_mj_per_km"], 4.0)
        self.assertAlmostEqual(result["vde_extra_high_mj_per_km"], 4.0 / 6.0)
        self.assertAlmostEqual(result["vde_net_mj_per_km"], 1.0)

    def test_workflow_epa_calculation_consumes_resolved_twc(self):
        payload = self._workflow_payload(inertia_class=1750.0, test_mass_kg=2400.0)

        with patch(
            "src.vde_core.vde_workflow_service.compute_vde_preview_from_inputs",
            return_value={"ok": True, "error": None, "total_mj_km": 1.23, "by_phase": {}},
        ) as mocked_preview:
            preview = build_vde_setup_preview(payload)

        self.assertTrue(preview["ok"])
        self.assertEqual(mocked_preview.call_count, 1)
        self.assertEqual(mocked_preview.call_args.kwargs["mass_kg"], 1750.0)

    def test_workflow_epa_does_not_resolve_twc_twice_in_phase_aggregation(self):
        payload = self._workflow_payload(
            mass_kg=1500.0,
            inertia_class=1644.0,
            cycle_df=_epa_phase_cycle(),
        )
        captured_masses = []

        def fake_compute(_group, *_args, **kwargs):
            captured_masses.append(kwargs.get("mass_kg") if "mass_kg" in kwargs else _args[-1])
            return {"MJ_total": 0.1, "km": 0.1, "MJ_km": 1.0}

        with patch("src.vde_core.phase_aggregation.inertia_class_from_mass") as mocked_twc, patch(
            "src.vde_core.phase_aggregation.compute_vde_net",
            side_effect=fake_compute,
        ):
            preview = build_vde_setup_preview(payload)

        self.assertTrue(preview["ok"])
        mocked_twc.assert_not_called()
        self.assertGreaterEqual(len(captured_masses), 3)
        self.assertTrue(all(value == 1644.0 for value in captured_masses))

    def test_same_epa_twc_produces_same_inertial_vde(self):
        cycle = _linear_accel_cycle(final_speed_mps=10.0, duration_s=10)
        payload_a = self._workflow_payload(
            mass_kg=1424.0,
            test_mass_kg=1560.0,
            initial_abc_total={"A": 0.0, "B": 0.0, "C": 0.0},
            cycle_df=cycle,
        )
        payload_b = self._workflow_payload(
            mass_kg=1480.0,
            test_mass_kg=1616.0,
            initial_abc_total={"A": 0.0, "B": 0.0, "C": 0.0},
            cycle_df=cycle,
        )

        preview_a = build_vde_setup_preview(payload_a)
        preview_b = build_vde_setup_preview(payload_b)

        self.assertEqual(preview_a["mass_setup"]["resolved_mass_used_kg"], inertia_class_from_mass(1424.0))
        self.assertEqual(preview_b["mass_setup"]["resolved_mass_used_kg"], inertia_class_from_mass(1480.0))
        self.assertEqual(preview_a["mass_setup"]["resolved_mass_used_kg"], preview_b["mass_setup"]["resolved_mass_used_kg"])
        self.assertAlmostEqual(preview_a["vde_total"]["mj_per_km"], preview_b["vde_total"]["mj_per_km"], places=9)

    def test_workflow_wltp_uses_physical_test_mass_directly(self):
        payload = self._workflow_payload(
            legislation="WLTP",
            wltp_category="M1",
            mass_kg=1500.0,
            payload_kg=300.0,
            test_mass_kg=1800.0,
        )

        with patch(
            "src.vde_core.vde_workflow_service.compute_vde_preview_from_inputs",
            return_value={"ok": True, "error": None, "total_mj_km": 1.23, "by_phase": {}},
        ) as mocked_preview:
            preview = build_vde_setup_preview(payload)

        self.assertTrue(preview["ok"])
        self.assertEqual(mocked_preview.call_args.kwargs["mass_kg"], 1800.0)

    def test_wltp_pure_inertial_energy_scales_linearly_with_mass(self):
        base_payload = self._workflow_payload(
            legislation="WLTP",
            wltp_category="M1",
            mass_kg=1000.0,
            payload_kg=200.0,
            initial_abc_total={"A": 0.0, "B": 0.0, "C": 0.0},
            cycle_df=_linear_accel_cycle(final_speed_mps=10.0, duration_s=10),
        )
        preview_light = build_vde_setup_preview({**base_payload, "test_mass_kg": 1000.0})
        preview_heavy = build_vde_setup_preview({**base_payload, "test_mass_kg": 2000.0})

        self.assertAlmostEqual(
            preview_heavy["vde_total"]["mj_per_km"] / preview_light["vde_total"]["mj_per_km"],
            2.0,
            places=9,
        )

    def test_total_minus_net_matches_transmission_force_delta(self):
        cycle = _constant_speed_cycle()
        payload = self._workflow_payload(
            legislation="WLTP",
            wltp_category="M1",
            mass_kg=1500.0,
            payload_kg=300.0,
            test_mass_kg=1800.0,
            initial_abc_total={"A": 100.0, "B": 2.0, "C": 0.01},
            transmission_losses={"source": "manual", "A_TRANS": 10.0, "B_TRANS": 0.5, "C_TRANS": 0.001},
            cycle_df=cycle,
        )

        preview = build_vde_setup_preview(payload)
        transmission_force_n = 10.0 + 0.5 * 36.0 + 0.001 * (36.0**2)

        self.assertAlmostEqual(
            preview["vde_total"]["mj_per_km"] - preview["vde_net"]["mj_per_km"],
            transmission_force_n * 0.001,
            places=9,
        )

    def test_baseline_measured_abc_remains_authoritative_when_components_do_not_close(self):
        payload = self._workflow_payload(
            legislation="WLTP",
            wltp_category="M1",
            mass_kg=1500.0,
            payload_kg=300.0,
            test_mass_kg=1800.0,
            initial_abc_total_source="baseline",
            baseline_row={
                "coast_A_N": 100.0,
                "coast_B_N_per_kph": 2.0,
                "coast_C_N_per_kph2": 0.01,
            },
            components={
                "tire_delta": {"role": "TOTAL_COMPONENT", "A": 5.0, "B": 0.0, "C": 0.0},
                "unrelated_component_shape": {"role": "TOTAL_COMPONENT", "A": 0.0, "B": 0.0, "C": 0.0},
            },
        )

        preview = build_vde_setup_preview(payload)

        self.assertAlmostEqual(preview["initial_abc_total_base"]["A"], 100.0)
        self.assertAlmostEqual(preview["component_abc_total"]["A"], 5.0)
        self.assertAlmostEqual(preview["abc_total"]["A"], 105.0)

    def test_total_and_net_reuse_same_mass_basis(self):
        payload = self._workflow_payload(
            legislation="WLTP",
            wltp_category="M1",
            mass_kg=1500.0,
            payload_kg=300.0,
            test_mass_kg=1800.0,
            transmission_losses={"source": "manual", "A_TRANS": 10.0, "B_TRANS": 0.5, "C_TRANS": 0.001},
        )

        with patch(
            "src.vde_core.vde_workflow_service.compute_vde_preview_from_inputs",
            return_value={"ok": True, "error": None, "total_mj_km": 1.23, "by_phase": {}},
        ) as mocked_preview:
            preview = build_vde_setup_preview(payload)

        self.assertTrue(preview["ok"])
        self.assertEqual(mocked_preview.call_count, 2)
        masses = [call.kwargs["mass_kg"] for call in mocked_preview.call_args_list]
        self.assertEqual(masses, [1800.0, 1800.0])

    def test_preview_engineering_comparison_matches_core_preview_outputs(self):
        state = apply_v22_baseline(create_v22_state(), _preview_baseline_row())
        state = apply_v22_proposal_matrix(
            state,
            [
                {"proposal_id": "requested_1", "walk_from": "baseline", "mass": "Custom test mass", "aero": "Delta CdA"},
                {"proposal_id": "requested_2", "walk_from": "requested_1", "mass": "Inherit", "aero": "Inherit"},
            ],
        )
        state = apply_v22_domain_inputs(state, "mass", {"requested_1": {"test_mass_kg": 1810.0}})
        state = apply_v22_domain_inputs(state, "aero", {"requested_1": {"delta_CdA": -0.01}})
        bundle = build_v22_preview_bundle(state, baseline_context=compact_baseline_context(state))
        state["preview"] = {"status": "fresh", "fingerprint": bundle["fingerprint"], "result": bundle}

        comparison = build_engineering_comparison_payload(state, "Metric")
        proposal = bundle["resolution_result"]["proposal_results"][0]
        rows = {
            row["field_key"]: row
            for group in comparison["groups"]
            for row in group["rows"]
        }

        self.assertAlmostEqual(rows["abc_total_A"]["raw_values"]["requested_1"], proposal["abc_total"]["A"])
        self.assertAlmostEqual(rows["abc_total_B"]["raw_values"]["requested_1"], proposal["abc_total"]["B"])
        self.assertAlmostEqual(rows["abc_total_C"]["raw_values"]["requested_1"], proposal["abc_total"]["C"])
        self.assertAlmostEqual(rows["abc_net_A"]["raw_values"]["requested_1"], proposal["abc_net"]["A"])
        self.assertAlmostEqual(rows["abc_net_B"]["raw_values"]["requested_1"], proposal["abc_net"]["B"])
        self.assertAlmostEqual(rows["abc_net_C"]["raw_values"]["requested_1"], proposal["abc_net"]["C"])
        self.assertAlmostEqual(rows["vde_total_mj_per_km"]["raw_values"]["requested_1"], proposal["vde_results"]["total"]["mj_per_km"])
        self.assertAlmostEqual(rows["vde_net_mj_per_km"]["raw_values"]["requested_1"], proposal["vde_results"]["net"]["mj_per_km"])
