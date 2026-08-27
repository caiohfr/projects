"""Resolver-level cross-path parity: Quick resolved Mass/Aero state vs. the
canonical VDE Setup resolver functions, compared BEFORE either downstream
VDE engine runs.

This file is the parity boundary called out explicitly in
docs/sprints/SPRINT_10B_QUICK_MASS_AERO_RESOLUTION.md: it never calls
resolve_cycle_vde_results() (legacy, EPA-phase-weighted) or
calculate_vehicle_demand() (frozen Sprint 9, whole-trace). Those two
engines are known -- via Sprint 9's own reconciliation suite -- to
disagree with each other for EPA rows unless deliberately re-combined by
EPA policy phase weights; comparing across them is not a Quick Scenario
concern (see test_quick_scenario_vehicle_demand_integration.py's module
docstring for that separate, downstream-engine-level comparison, which
compares Quick against each engine independently, never one engine
against the other).

Every comparison here answers one question only: "does the Quick resolver
produce the exact same resolved physical state (curb mass, VDE calculation
mass, mass basis, CdA, roadload ABC TOTAL/NET) as calling the same
canonical VDE Setup resolver function directly, for an equivalent
engineering intention?" The expected value is always computed by calling
resolve_mass_proposal()/cdA_to_C()/resolve_roadload_boundaries()
independently -- never a hand-guessed number.
"""

import unittest

from src.vde_core.comparison_report_service import resolve_roadload_boundaries
from src.vde_core.quick_scenario import (
    MassQuickChange,
    QuickScenario,
    ScalarChange,
    ScalarChangeMode,
    VehicleQuickOverrides,
    resolve_quick_vehicle_scenario,
)
from src.vde_core.roadload import cdA_to_C
from src.vde_core.test_mass import inertia_step_for_mass
from src.vde_core.vde_mass_proposal_resolver import resolve_mass_proposal


def _epa_row(**overrides) -> dict:
    row = {
        "id": 1,
        "legislation": "EPA",
        "mass_kg": 1500.0,
        "test_mass_kg": 1644.0,
        "inertia_class": 1644.0,
        "coast_A_N": 118.0,
        "coast_B_N_per_kph": 0.0200,
        "coast_C_N_per_kph2": 0.0090,
        "trans_A_coef_N": 8.5,
        "trans_B_coef_Npkph": 0.0040,
        "trans_C_coef_Npkph2": 0.0008,
        "rrc_N_per_kN": 8.0,
        "cda_m2": 0.620,
    }
    row.update(overrides)
    return row


def _wltp_row(**overrides) -> dict:
    row = {
        "id": 990001,
        "legislation": "WLTP",
        "category": "QA_WLTP",
        "wltp_category": "M1",
        "mass_kg": 1600.0,
        "test_mass_kg": 1780.0,
        "payload_kg": 180.0,
        "options_kg": 0.0,
        "coast_A_N": 115.0,
        "coast_B_N_per_kph": 0.019,
        "coast_C_N_per_kph2": 0.0088,
        "trans_A_coef_N": 8.2,
        "trans_B_coef_Npkph": 0.0039,
        "trans_C_coef_Npkph2": 0.0008,
        "rrc_N_per_kN": 8.0,
        "cda_m2": 0.60,
    }
    row.update(overrides)
    return row


def _scenario(overrides: VehicleQuickOverrides, *, source_identity="vde:1") -> QuickScenario:
    return QuickScenario(source_identity=source_identity, slot=1, vehicle_overrides=overrides)


def _assert_mass_state_matches(test_case, quick_result, expected_outcome) -> None:
    expected = expected_outcome["resolved_snapshot"]
    test_case.assertEqual(quick_result.resolved_curb_mass_kg, expected.get("curb_mass_kg"))
    test_case.assertEqual(
        quick_result.resolved_vde_calculation_mass_kg, expected.get("vde_calculation_mass_kg")
    )
    test_case.assertEqual(quick_result.resolved_vde_mass_basis, expected.get("vde_mass_basis"))


class EpaCurbChangeInsideSameTwcParityTests(unittest.TestCase):
    def test_matches_canonical_epa_curb_to_twc_resolver(self):
        row = _epa_row()
        source_bracket = inertia_step_for_mass(row["mass_kg"])
        target_curb = row["mass_kg"] + 1.0
        target_bracket = inertia_step_for_mass(target_curb)
        self.assertEqual(
            source_bracket["inertia_class_kg"],
            target_bracket["inertia_class_kg"],
            "Fixture assumption broken: +1kg crossed a TWC bracket.",
        )

        expected = resolve_mass_proposal(dict(row), "EPA_CURB_TO_TWC", {"mass_kg": target_curb})
        self.assertEqual(expected["status"], "OK")

        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.ABSOLUTE, target_curb))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        _assert_mass_state_matches(self, result, expected)
        # This is the "inside one bracket" case: TWC must stay pinned.
        self.assertEqual(result.resolved_vde_calculation_mass_kg, row["test_mass_kg"])


class EpaCurbChangeCrossingTwcParityTests(unittest.TestCase):
    def test_matches_canonical_epa_curb_to_twc_resolver(self):
        row = _epa_row()
        source_bracket = inertia_step_for_mass(row["mass_kg"])
        target_curb = float(source_bracket["upper_bound_inclusive"]) + 1.0
        target_bracket = inertia_step_for_mass(target_curb)
        self.assertNotEqual(source_bracket["inertia_class_kg"], target_bracket["inertia_class_kg"])

        expected = resolve_mass_proposal(dict(row), "EPA_CURB_TO_TWC", {"mass_kg": target_curb})
        self.assertEqual(expected["status"], "OK")

        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.ABSOLUTE, target_curb))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        _assert_mass_state_matches(self, result, expected)
        self.assertNotEqual(result.resolved_vde_calculation_mass_kg, row["test_mass_kg"])


class EpaExplicitTwcShiftParityTests(unittest.TestCase):
    """"Explicit TWC shift/target where supported": TWC Shift
    (MASS_TWC_SHIFT, step-based) and Target TWC (EPA_CURB_TO_TWC, covered
    above) are the two supported explicit-regulatory-mass modes (Sec 4.1 /
    Decision 3). This class covers the Shift mode; Target TWC parity is
    covered by the two classes above.
    """

    def test_twc_shift_up_matches_canonical_mass_twc_shift_resolver(self):
        row = _epa_row()
        expected = resolve_mass_proposal(
            dict(row), "MASS_TWC_SHIFT", {"shift_steps": 1.0, "target_side": "Up"}
        )
        self.assertEqual(expected["status"], "OK")

        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(twc_shift_steps=1.0, twc_shift_side="Up")
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        _assert_mass_state_matches(self, result, expected)
        self.assertGreater(result.resolved_vde_calculation_mass_kg, row["test_mass_kg"])

    def test_twc_shift_down_matches_canonical_mass_twc_shift_resolver(self):
        row = _epa_row()
        expected = resolve_mass_proposal(
            dict(row), "MASS_TWC_SHIFT", {"shift_steps": 1.0, "target_side": "Down"}
        )
        self.assertEqual(expected["status"], "OK")

        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(twc_shift_steps=1.0, twc_shift_side="Down")
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        _assert_mass_state_matches(self, result, expected)
        self.assertLess(result.resolved_vde_calculation_mass_kg, row["test_mass_kg"])


class WltpMassLineParityTests(unittest.TestCase):
    def test_matches_canonical_wltp_mass_line_resolver(self):
        row = _wltp_row()
        target_curb = row["mass_kg"] - 20.0
        expected = resolve_mass_proposal(dict(row), "WLTP_MASS_LINE", {"mass_kg": target_curb})
        self.assertEqual(expected["status"], "OK")

        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        _assert_mass_state_matches(self, result, expected)


class CdaAbsoluteParityTests(unittest.TestCase):
    def test_matches_canonical_cda_to_c_composition(self):
        row = _epa_row()
        target_cda = 0.60
        reference_cda = row["cda_m2"]
        delta_cda = target_cda - reference_cda
        expected_row = dict(row)
        expected_row["cda_m2"] = target_cda
        expected_row["coast_C_N_per_kph2"] = row["coast_C_N_per_kph2"] + cdA_to_C(delta_cda)
        expected_boundaries = resolve_roadload_boundaries(expected_row)

        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.ABSOLUTE, target_cda))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)

        self.assertEqual(result.resolved_cda_m2, target_cda)
        self.assertAlmostEqual(result.abc_total.A_N, expected_boundaries["total"].A, places=9)
        self.assertAlmostEqual(result.abc_total.B_N_per_kph, expected_boundaries["total"].B, places=9)
        self.assertAlmostEqual(result.abc_total.C_N_per_kph2, expected_boundaries["total"].C, places=9)
        self.assertAlmostEqual(result.abc_net.C_N_per_kph2, expected_boundaries["net"].C, places=9)


class CdaDeltaParityTests(unittest.TestCase):
    def test_matches_canonical_cda_to_c_composition(self):
        row = _epa_row()
        delta_cda = -0.05
        expected_row = dict(row)
        expected_row["cda_m2"] = row["cda_m2"] + delta_cda
        expected_row["coast_C_N_per_kph2"] = row["coast_C_N_per_kph2"] + cdA_to_C(delta_cda)
        expected_boundaries = resolve_roadload_boundaries(expected_row)

        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.DELTA, delta_cda))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)

        self.assertAlmostEqual(result.resolved_cda_m2, expected_row["cda_m2"], places=9)
        self.assertAlmostEqual(result.abc_total.C_N_per_kph2, expected_boundaries["total"].C, places=9)
        self.assertAlmostEqual(result.abc_net.C_N_per_kph2, expected_boundaries["net"].C, places=9)


class CdaPercentParityTests(unittest.TestCase):
    def test_matches_canonical_cda_to_c_composition(self):
        row = _epa_row()
        percent = -5.0
        target_cda = row["cda_m2"] * (1.0 + percent / 100.0)
        delta_cda = target_cda - row["cda_m2"]
        expected_row = dict(row)
        expected_row["cda_m2"] = target_cda
        expected_row["coast_C_N_per_kph2"] = row["coast_C_N_per_kph2"] + cdA_to_C(delta_cda)
        expected_boundaries = resolve_roadload_boundaries(expected_row)

        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.PERCENT, percent))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)

        self.assertAlmostEqual(result.resolved_cda_m2, target_cda, places=9)
        self.assertAlmostEqual(result.abc_total.C_N_per_kph2, expected_boundaries["total"].C, places=9)
        self.assertAlmostEqual(result.abc_net.C_N_per_kph2, expected_boundaries["net"].C, places=9)


class CombinedMassAndCdaParityTests(unittest.TestCase):
    def test_combined_change_matches_both_canonical_resolvers_independently(self):
        row = _epa_row()
        target_curb = row["mass_kg"] - 20.0
        delta_cda = -0.02

        expected_mass = resolve_mass_proposal(dict(row), "EPA_CURB_TO_TWC", {"mass_kg": target_curb})
        self.assertEqual(expected_mass["status"], "OK")

        expected_row = dict(row)
        expected_row["mass_kg"] = expected_mass["resolved_snapshot"]["curb_mass_kg"]
        expected_row["test_mass_kg"] = expected_mass["resolved_snapshot"]["vde_calculation_mass_kg"]
        expected_row["cda_m2"] = row["cda_m2"] + delta_cda
        expected_row["coast_C_N_per_kph2"] = row["coast_C_N_per_kph2"] + cdA_to_C(delta_cda)
        expected_boundaries = resolve_roadload_boundaries(expected_row)

        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0)),
            cda_change=ScalarChange(ScalarChangeMode.DELTA, delta_cda),
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)

        _assert_mass_state_matches(self, result, expected_mass)
        self.assertAlmostEqual(result.resolved_cda_m2, expected_row["cda_m2"], places=9)
        self.assertAlmostEqual(result.abc_total.C_N_per_kph2, expected_boundaries["total"].C, places=9)
        self.assertAlmostEqual(result.abc_net.C_N_per_kph2, expected_boundaries["net"].C, places=9)
        self.assertAlmostEqual(result.abc_total.A_N, expected_boundaries["total"].A, places=9)


if __name__ == "__main__":
    unittest.main()
