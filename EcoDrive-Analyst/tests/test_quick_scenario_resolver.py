import copy
import unittest
from unittest import mock

from src.vde_core.quick_scenario import (
    DomainReadiness,
    MassQuickChange,
    QuickScenario,
    ReferencePressureProvenance,
    ScalarChange,
    ScalarChangeMode,
    TireQuickChange,
    TireSource,
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


def _scenario(overrides: VehicleQuickOverrides, *, source_identity="vde:1", slot=1) -> QuickScenario:
    return QuickScenario(source_identity=source_identity, slot=slot, vehicle_overrides=overrides)


class NeutralScenarioTests(unittest.TestCase):
    def test_no_overrides_is_ready_and_reproduces_source(self):
        row = _epa_row()
        result = resolve_quick_vehicle_scenario(_scenario(VehicleQuickOverrides()), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.readiness.mass, DomainReadiness.NOT_REQUESTED)
        self.assertEqual(result.readiness.aero, DomainReadiness.NOT_REQUESTED)
        self.assertEqual(result.readiness.tire, DomainReadiness.NOT_REQUESTED)
        self.assertEqual(result.resolved_curb_mass_kg, row["mass_kg"])
        self.assertEqual(result.resolved_vde_calculation_mass_kg, row["test_mass_kg"])
        self.assertEqual(result.resolved_cda_m2, row["cda_m2"])
        self.assertIsNotNone(result.vehicle_demand_result)


class CurbMassChangeTests(unittest.TestCase):
    def test_absolute_curb_change(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.ABSOLUTE, 1450.0))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_curb_mass_kg, 1450.0)

    def test_delta_curb_change(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_curb_mass_kg, 1480.0)

    def test_percent_curb_change(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.PERCENT, -2.0))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertAlmostEqual(result.resolved_curb_mass_kg, 1470.0)

    def test_delta_zero_is_neutral(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, 0.0))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_curb_mass_kg, row["mass_kg"])

    def test_percent_zero_is_neutral(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.PERCENT, 0.0))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_curb_mass_kg, row["mass_kg"])

    def test_missing_source_curb_for_delta_is_missing(self):
        row = _epa_row(mass_kg=None)
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertFalse(result.is_ready)
        self.assertEqual(result.readiness.mass, DomainReadiness.MISSING)
        self.assertIsNone(result.vehicle_demand_result)
        self.assertTrue(any("curb mass" in issue for issue in result.issues))

    def test_missing_source_curb_for_percent_is_missing(self):
        row = _epa_row(mass_kg=None)
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.PERCENT, -2.0))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertFalse(result.is_ready)
        self.assertEqual(result.readiness.mass, DomainReadiness.MISSING)


class EpaTwcBoundaryTests(unittest.TestCase):
    """Bracket membership is always derived at test time via the canonical
    resolver's own inertia_step_for_mass -- never a hand-guessed cutoff
    (Sec 22: "use the same canonical resolver independently to generate the
    expected reference result").
    """

    def test_curb_change_staying_inside_one_twc_bracket_leaves_twc_unchanged(self):
        row = _epa_row()
        source_bracket = inertia_step_for_mass(row["mass_kg"])
        # A 1 kg nudge should, for essentially every bracket, stay inside
        # the same TWC step.
        target_curb = row["mass_kg"] + 1.0
        target_bracket = inertia_step_for_mass(target_curb)
        self.assertEqual(
            source_bracket["inertia_class_kg"],
            target_bracket["inertia_class_kg"],
            "Test fixture assumption broken: +1kg crossed a TWC bracket.",
        )

        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.ABSOLUTE, target_curb))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_vde_calculation_mass_kg, target_bracket["inertia_class_kg"])
        self.assertEqual(result.resolved_vde_calculation_mass_kg, row["test_mass_kg"])

    def test_curb_change_crossing_a_twc_bracket_changes_twc(self):
        row = _epa_row()
        source_bracket = inertia_step_for_mass(row["mass_kg"])
        target_curb = float(source_bracket["upper_bound_inclusive"]) + 1.0
        target_bracket = inertia_step_for_mass(target_curb)
        self.assertNotEqual(source_bracket["inertia_class_kg"], target_bracket["inertia_class_kg"])

        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.ABSOLUTE, target_curb))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_vde_calculation_mass_kg, target_bracket["inertia_class_kg"])
        self.assertNotEqual(result.resolved_vde_calculation_mass_kg, row["test_mass_kg"])


class EpaTwcShiftTests(unittest.TestCase):
    def test_twc_shift_up(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(twc_shift_steps=1.0, twc_shift_side="Up")
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertGreater(result.resolved_vde_calculation_mass_kg, row["test_mass_kg"])

    def test_twc_shift_down(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(twc_shift_steps=1.0, twc_shift_side="Down")
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertLess(result.resolved_vde_calculation_mass_kg, row["test_mass_kg"])

    def test_twc_shift_rejected_for_non_epa_legislation(self):
        row = _wltp_row()
        overrides = VehicleQuickOverrides(mass_change=MassQuickChange(twc_shift_steps=1.0))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertFalse(result.is_ready)
        self.assertEqual(result.readiness.mass, DomainReadiness.MISSING)
        self.assertTrue(any("EPA-only" in issue for issue in result.issues))


class WltpMassLineTests(unittest.TestCase):
    def test_wltp_curb_change_resolves_via_wltp_mass_line(self):
        row = _wltp_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0))
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_curb_mass_kg, 1580.0)
        self.assertNotEqual(result.resolved_vde_calculation_mass_kg, result.resolved_curb_mass_kg)
        self.assertIn("WLTP", str(result.resolved_vde_mass_basis))


class AeroCdaChangeTests(unittest.TestCase):
    def test_absolute_cda_change(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.ABSOLUTE, 0.60))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_cda_m2, 0.60)

    def test_delta_cda_change(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.02))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertAlmostEqual(result.resolved_cda_m2, 0.60)

    def test_percent_cda_change(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.PERCENT, -5.0))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertAlmostEqual(result.resolved_cda_m2, 0.589)

    def test_delta_zero_is_neutral(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.DELTA, 0.0))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_cda_m2, row["cda_m2"])
        self.assertEqual(result.abc_total.C_N_per_kph2, row["coast_C_N_per_kph2"])

    def test_percent_zero_is_neutral(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.PERCENT, 0.0))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_cda_m2, row["cda_m2"])

    def test_missing_source_cda_for_delta_is_missing(self):
        row = _epa_row(cda_m2=None)
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.02))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertFalse(result.is_ready)
        self.assertEqual(result.readiness.aero, DomainReadiness.MISSING)

    def test_missing_source_cda_for_percent_is_missing(self):
        row = _epa_row(cda_m2=None)
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.PERCENT, -5.0))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertFalse(result.is_ready)
        self.assertEqual(result.readiness.aero, DomainReadiness.MISSING)

    def test_absolute_cda_missing_source_and_no_reference_is_missing(self):
        row = _epa_row(cda_m2=None)
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.ABSOLUTE, 0.60))
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertFalse(result.is_ready)
        self.assertEqual(result.readiness.aero, DomainReadiness.MISSING)
        self.assertTrue(any("reference CdA" in issue for issue in result.issues))

    def test_absolute_cda_missing_source_with_user_reference_is_ready(self):
        row = _epa_row(cda_m2=None)
        overrides = VehicleQuickOverrides(
            cda_change=ScalarChange(ScalarChangeMode.ABSOLUTE, 0.60),
            aero_reference_cda_m2=0.62,
            aero_reference_cda_provenance=ReferencePressureProvenance.USER_PROVIDED,
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.resolved_cda_m2, 0.60)
        self.assertTrue(any("Manual reference CdA" in issue for issue in result.issues))
        # delta = 0.60 - 0.62 = -0.02, same as the DELTA case above. Uses
        # the real canonical cdA_to_C() to compute the expected value --
        # never a hand-written second implementation of the formula.
        self.assertAlmostEqual(
            result.abc_total.C_N_per_kph2,
            _epa_row()["coast_C_N_per_kph2"] + cdA_to_C(-0.02),
        )


class MassAndAeroCombinedTests(unittest.TestCase):
    def test_mass_and_aero_combined(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0)),
            cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.02),
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.readiness.mass, DomainReadiness.READY)
        self.assertEqual(result.readiness.aero, DomainReadiness.READY)
        self.assertEqual(result.resolved_curb_mass_kg, 1480.0)
        self.assertAlmostEqual(result.resolved_cda_m2, 0.60)

    def test_mass_ready_aero_missing_leaves_whole_scenario_unresolved(self):
        row = _epa_row(cda_m2=None)
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0)),
            cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.02),
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertFalse(result.is_ready)
        self.assertEqual(result.readiness.mass, DomainReadiness.READY)
        self.assertEqual(result.readiness.aero, DomainReadiness.MISSING)
        # Sec 18: no silent partial calc -- mass being ready must not, on
        # its own, produce a VehicleDemandResult while aero is missing.
        self.assertIsNone(result.vehicle_demand_result)
        self.assertIsNone(result.vde_total_mj_per_km)


class TireNeutralResolutionTests(unittest.TestCase):
    def test_requesting_current_tire_none_is_ready(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            tire_change=TireQuickChange(source=TireSource.CURRENT)
        )
        result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        self.assertEqual(result.readiness.tire, DomainReadiness.READY)
        self.assertEqual(result.resolved_rrc_n_per_kn, row["rrc_N_per_kN"])


class ReuseProofTests(unittest.TestCase):
    """Sec 22: prove reuse by spying on the real canonical function -- the
    spy `wraps` the genuine imported implementation, so these tests still
    exercise real physics, not a stand-in.
    """

    def test_mass_resolution_calls_canonical_resolve_mass_proposal(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0))
        )
        with mock.patch(
            "src.vde_core.quick_scenario.resolver.resolve_mass_proposal",
            wraps=resolve_mass_proposal,
        ) as spy:
            result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        spy.assert_called_once()
        called_snapshot, called_proposal_type, called_inputs = spy.call_args[0]
        self.assertEqual(called_proposal_type, "EPA_CURB_TO_TWC")
        self.assertEqual(called_inputs["mass_kg"], 1480.0)

    def test_aero_resolution_calls_canonical_cda_to_c(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.02))
        with mock.patch(
            "src.vde_core.quick_scenario.resolver.cdA_to_C", wraps=cdA_to_C
        ) as spy:
            result = resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertTrue(result.is_ready)
        spy.assert_called_once()
        (called_delta_cda,), _kwargs = spy.call_args
        self.assertAlmostEqual(called_delta_cda, -0.02)


class ImmutabilityAndDeterminismTests(unittest.TestCase):
    def test_source_row_is_never_mutated(self):
        row = _epa_row()
        original = copy.deepcopy(row)
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0)),
            cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.02),
        )
        resolve_quick_vehicle_scenario(_scenario(overrides), source_vde_row=row)
        self.assertEqual(row, original)

    def test_repeated_resolution_is_deterministic(self):
        row = _epa_row()
        overrides = VehicleQuickOverrides(
            mass_change=MassQuickChange(curb_change=ScalarChange(ScalarChangeMode.DELTA, -20.0)),
            cda_change=ScalarChange(ScalarChangeMode.DELTA, -0.02),
        )
        scenario = _scenario(overrides)
        first = resolve_quick_vehicle_scenario(scenario, source_vde_row=row)
        second = resolve_quick_vehicle_scenario(scenario, source_vde_row=row)
        self.assertEqual(first, second)


class QuickScenarioIdentitySharedVdeTests(unittest.TestCase):
    def test_two_quick_scenarios_from_distinct_fuelcons_sources_stay_distinct(self):
        first = _scenario(VehicleQuickOverrides(), source_identity="fc:1")
        second = _scenario(VehicleQuickOverrides(), source_identity="fc:2")
        self.assertNotEqual(first.identity, second.identity)

        row = _epa_row()
        result_1 = resolve_quick_vehicle_scenario(first, source_vde_row=row)
        result_2 = resolve_quick_vehicle_scenario(second, source_vde_row=row)
        self.assertNotEqual(result_1.quick_scenario_identity, result_2.quick_scenario_identity)
        # Same underlying physical source -> same resolved physics.
        self.assertEqual(result_1.resolved_curb_mass_kg, result_2.resolved_curb_mass_kg)


if __name__ == "__main__":
    unittest.main()
