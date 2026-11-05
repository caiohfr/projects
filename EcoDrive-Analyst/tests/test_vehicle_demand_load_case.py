import unittest
from unittest.mock import patch

import pandas as pd

from src.vde_core.roadload.tire_model import G_MPS2
from src.vde_core.vehicle_demand import EnergyMode, RoadloadBasis
from src.vde_core.vehicle_demand.adapters import build_vehicle_demand_request
from src.vde_core.vehicle_demand.engine import calculate_vehicle_demand
from src.vde_core.vehicle_demand.load_case import (
    FixedSpeedSpec,
    LoadCaseAnalysisMode,
    LoadCaseEnvironment,
    LoadCaseMass,
    LoadCaseMassMode,
    LoadCaseRoute,
    LoadCaseTrailer,
    RoadloadLoadCaseRequest,
    SpeedSweepSpec,
    calculate_roadload_load_case,
    grade_force_N,
    speed_sweep_values,
)


def _row(**overrides):
    row = {
        "id": 13001,
        "record_origin": "SYNTHETIC_QA",
        "legislation": "WLTP",
        "category": "QA",
        "make": "QA",
        "model": "LOAD-CASE",
        "year": 2026,
        "mass_kg": 1500.0,
        "test_mass_kg": 1600.0,
        "coast_A_N": 100.0,
        "coast_B_N_per_kph": 1.0,
        "coast_C_N_per_kph2": 0.01,
        "trans_A_coef_N": 10.0,
        "trans_B_coef_Npkph": 0.1,
        "trans_C_coef_Npkph2": 0.001,
        "rrc_N_per_kN": 8.0,
        "cda_m2": 0.65,
    }
    row.update(overrides)
    return row


def _cycle():
    return pd.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "v": [0.0, 10.0, 10.0, 0.0]})


def _request(**overrides):
    values = {
        "baseline_vde_id": 13001,
        "analysis_mode": LoadCaseAnalysisMode.DRIVE_CYCLE,
    }
    values.update(overrides)
    return RoadloadLoadCaseRequest(**values)


class GradePhysicsTests(unittest.TestCase):
    def test_zero_grade_is_exact_noop(self):
        self.assertEqual(grade_force_N(2000.0, 0.0), 0.0)

    def test_grade_sign(self):
        self.assertGreater(grade_force_N(2000.0, 6.0), 0.0)
        self.assertLess(grade_force_N(2000.0, -6.0), 0.0)
        self.assertAlmostEqual(grade_force_N(2000.0, 6.0), -grade_force_N(2000.0, -6.0), places=9)


class DriveCycleLoadCaseTests(unittest.TestCase):
    def test_no_overrides_has_canonical_parity(self):
        result = calculate_roadload_load_case(_row(), _request(), cycle_frame=_cycle())
        self.assertEqual(result.scenario_request.roadload_total, result.baseline_request.roadload_total)
        self.assertAlmostEqual(
            result.baseline_result.total_summary.vde_mj_per_km,
            result.scenario_result.total_summary.vde_mj_per_km,
            places=12,
        )

    def test_heavier_cycle_mass_increases_inertial_demand(self):
        result = calculate_roadload_load_case(
            _row(rrc_N_per_kN=None),
            _request(
                mass=LoadCaseMass(
                    mode=LoadCaseMassMode.CUSTOM,
                    custom_vehicle_mass_kg=2100.0,
                    use_estimated_rrc_prior=False,
                )
            ),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.scenario_request.roadload_total, result.baseline_request.roadload_total)
        self.assertGreater(
            result.scenario_result.total_summary.positive_inertial_work_MJ,
            result.baseline_result.total_summary.positive_inertial_work_MJ,
        )

    def test_mass_only_cycle_keeps_static_roadload_but_changes_dynamic_profile(self):
        result = calculate_roadload_load_case(
            _row(rrc_N_per_kN=None),
            _request(
                mass=LoadCaseMass(
                    mode=LoadCaseMassMode.CUSTOM,
                    custom_vehicle_mass_kg=2500.0,
                    use_estimated_rrc_prior=False,
                )
            ),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.baseline_request.roadload_total, result.scenario_request.roadload_total)
        self.assertTrue(all(point.baseline_force_N == point.scenario_force_N for point in result.force_curve_points))
        self.assertIsNotNone(result.baseline_total_profile)
        self.assertIsNotNone(result.scenario_total_profile)
        acceleration_indices = [
            index for index, value in enumerate(result.baseline_total_profile.accel_mps2) if value > 0
        ]
        self.assertTrue(acceleration_indices)
        self.assertTrue(
            any(
                result.baseline_total_profile.tractive_force_N[index]
                != result.scenario_total_profile.tractive_force_N[index]
                for index in acceleration_indices
            )
        )
        self.assertNotEqual(
            result.baseline_result.total_summary.vde_mj_per_km,
            result.scenario_result.total_summary.vde_mj_per_km,
        )

    def test_load_case_baseline_has_canonical_vehicle_demand_parity(self):
        row = _row(
            legislation="EPA",
            test_mass_kg=1777.0,
            test_mass_basis="EPA_INERTIA_CLASS",
            inertia_class=2041.0,
        )
        canonical_request = build_vehicle_demand_request(row)
        canonical_result = calculate_vehicle_demand(canonical_request, _cycle())
        load_case = calculate_roadload_load_case(row, _request(), cycle_frame=_cycle())
        self.assertEqual(load_case.baseline_request, canonical_request)
        self.assertEqual(load_case.baseline_result, canonical_result)

    def test_total_net_external_overlay_preserves_gap(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(
                trailer=LoadCaseTrailer(True, 500.0, 20.0, 0.2, 0.002),
                route=LoadCaseRoute(grade_pct=4.0),
            ),
            cycle_frame=_cycle(),
        )
        before = result.baseline_request.roadload_total.A_N - result.baseline_request.roadload_net.A_N
        after = result.scenario_request.roadload_total.A_N - result.scenario_request.roadload_net.A_N
        self.assertAlmostEqual(before, after, places=12)

    def test_calculation_performs_no_db_writes(self):
        with patch("src.vde_core.db.insert_vde") as insert, patch("src.vde_core.db.update_vde") as update:
            calculate_roadload_load_case(_row(), _request(), cycle_frame=_cycle())
        insert.assert_not_called()
        update.assert_not_called()

    def test_repeated_request_is_deterministic(self):
        request = _request(route=LoadCaseRoute(grade_pct=2.5))
        first = calculate_roadload_load_case(_row(), request, cycle_frame=_cycle())
        second = calculate_roadload_load_case(_row(), request, cycle_frame=_cycle())
        self.assertEqual(first, second)

    def test_request_keyword_order_does_not_change_result(self):
        first_request = RoadloadLoadCaseRequest(
            baseline_vde_id=13001,
            mass=LoadCaseMass(mode=LoadCaseMassMode.CUSTOM, custom_vehicle_mass_kg=1900.0),
            route=LoadCaseRoute(grade_pct=2.0),
        )
        second_request = RoadloadLoadCaseRequest(
            route=LoadCaseRoute(grade_pct=2.0),
            mass=LoadCaseMass(custom_vehicle_mass_kg=1900.0, mode=LoadCaseMassMode.CUSTOM),
            baseline_vde_id=13001,
        )
        first = calculate_roadload_load_case(_row(), first_request, cycle_frame=_cycle())
        second = calculate_roadload_load_case(_row(), second_request, cycle_frame=_cycle())
        self.assertEqual(first.scenario_result, second.scenario_result)


class MassAndTrailerTests(unittest.TestCase):
    def test_baseline_and_gvwr_scenario_are_distinct_physical_states(self):
        result = calculate_roadload_load_case(
            _row(
                legislation="EPA",
                test_mass_kg=1777.0,
                test_mass_basis="EPA_INERTIA_CLASS",
                inertia_class=2041.0,
            ),
            _request(
                analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED,
                fixed_speed=FixedSpeedSpec(90.0),
                mass=LoadCaseMass(mode=LoadCaseMassMode.GVWR, gvwr_kg=2900.0),
            ),
        )
        self.assertEqual(result.baseline_effective_mass_kg, 2041.0)
        self.assertEqual(result.baseline_calculation_mass_kg, 2041.0)
        self.assertEqual(result.baseline_mass_basis, "EPA_INERTIA_CLASS")
        self.assertEqual(result.baseline_mass_source_field, "inertia_class")
        self.assertEqual(result.baseline_request.test_mass_kg, 2041.0)
        self.assertEqual(result.scenario_vehicle_mass_kg, 2900.0)
        self.assertEqual(result.scenario_calculation_mass_kg, 2900.0)
        self.assertEqual(result.scenario_mass_basis, "GVWR")
        self.assertEqual(result.scenario_system_mass_kg, 2900.0)

    def test_physical_test_mass_uses_persisted_test_mass(self):
        result = calculate_roadload_load_case(
            _row(
                legislation="EPA",
                test_mass_kg=1777.0,
                test_mass_basis="PHYSICAL_TEST_MASS",
                inertia_class=2041.0,
            ),
            _request(),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.baseline_effective_mass_kg, 1777.0)
        self.assertEqual(result.baseline_mass_basis, "PHYSICAL_TEST_MASS")
        self.assertEqual(result.baseline_mass_source_field, "test_mass_kg")

    def test_stored_test_mass_without_special_basis_reports_test_mass_source(self):
        result = calculate_roadload_load_case(
            _row(test_mass_kg=1701.0, test_mass_basis="CERTIFIED_TEST_WEIGHT", inertia_class=1814.0),
            _request(),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.baseline_calculation_mass_kg, 1701.0)
        self.assertEqual(result.baseline_mass_basis, "CERTIFIED_TEST_WEIGHT")
        self.assertEqual(result.baseline_mass_source_field, "test_mass_kg")

    def test_curb_mass_is_explicit_final_fallback_with_source(self):
        result = calculate_roadload_load_case(
            _row(test_mass_kg=None, test_mass_basis=None, inertia_class=None, mass_kg=1531.0),
            _request(),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.baseline_calculation_mass_kg, 1531.0)
        self.assertEqual(result.baseline_mass_basis, "CURB_FALLBACK")
        self.assertEqual(result.baseline_mass_source_field, "mass_kg")

    def test_payload_delta_and_target_payload_are_resolved(self):
        delta = calculate_roadload_load_case(
            _row(),
            _request(mass=LoadCaseMass(mode=LoadCaseMassMode.PAYLOAD_DELTA, payload_delta_kg=200.0)),
            cycle_frame=_cycle(),
        )
        target = calculate_roadload_load_case(
            _row(),
            _request(mass=LoadCaseMass(mode=LoadCaseMassMode.TARGET_PAYLOAD, target_payload_kg=300.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(delta.mass_state.resolved_vehicle_mass_kg, 1800.0)
        self.assertEqual(target.mass_state.resolved_vehicle_mass_kg, 1800.0)

    def test_gvwr_uses_existing_mass_contract(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(mass=LoadCaseMass(mode=LoadCaseMassMode.GVWR, gvwr_kg=2300.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.mass_state.resolved_vehicle_mass_kg, 2300.0)
        self.assertEqual(result.mass_state.system_mass_kg, 2300.0)

    def test_gcwr_resolves_vehicle_mass_and_does_not_add_trailer_twice(self):
        result = calculate_roadload_load_case(
            _row(gvwr_kg=2400.0),
            _request(
                mass=LoadCaseMass(mode=LoadCaseMassMode.GCWR, gcwr_kg=3200.0),
                trailer=LoadCaseTrailer(True, 800.0, 20.0, 0.2, 0.002),
            ),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.mass_state.resolved_vehicle_mass_kg, 2400.0)
        self.assertEqual(result.mass_state.vehicle_mass_at_gcwr_kg, 2400.0)
        self.assertEqual(result.mass_state.system_mass_kg, 3200.0)

    def test_trailer_mass_greater_than_gcwr_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "lower than GCWR"):
            calculate_roadload_load_case(
                _row(),
                _request(
                    mass=LoadCaseMass(mode=LoadCaseMassMode.GCWR, gcwr_kg=1000.0),
                    trailer=LoadCaseTrailer(True, 1000.0, 1.0, 0.0, 0.0),
                ),
                cycle_frame=_cycle(),
            )

    def test_complete_trailer_abc_is_added_once(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(trailer=LoadCaseTrailer(True, 500.0, 20.0, 0.2, 0.002)),
            cycle_frame=_cycle(),
        )
        self.assertAlmostEqual(result.scenario_request.roadload_total.A_N, 120.0)
        self.assertAlmostEqual(result.scenario_request.roadload_total.B_N_per_kph, 1.2)
        self.assertAlmostEqual(result.scenario_request.roadload_total.C_N_per_kph2, 0.012)
        self.assertIsNone(result.scenario_request.rrc_n_per_kn)
        self.assertEqual(result.rolling_rrc_source, "VDE_SPECIFIC")
        self.assertEqual(result.rolling_delta_abc.A_N, 0.0)

    def test_missing_trailer_abc_is_unavailable_not_invented(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(trailer=LoadCaseTrailer(True, 500.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.trailer_state.status, "UNAVAILABLE")
        self.assertEqual(result.scenario_request.roadload_total.A_N, result.baseline_request.roadload_total.A_N)
        self.assertTrue(any("Trailer roadload unavailable" in warning for warning in result.warnings))


class RollingPriorTests(unittest.TestCase):
    @staticmethod
    def _prior_row(**overrides):
        values = {
            "category": "Midsize Cars",
            "electrification": "BEV",
            "transmission_type": "AUTOMATIC",
            "gears": 1,
            "rrc_N_per_kN": None,
        }
        values.update(overrides)
        return _row(**values)

    @staticmethod
    def _custom_mass(mass_kg, *, use_prior=True):
        return LoadCaseMass(
            mode=LoadCaseMassMode.CUSTOM,
            custom_vehicle_mass_kg=mass_kg,
            use_estimated_rrc_prior=use_prior,
        )

    def test_specific_vde_rrc_wins_over_exact_prior(self):
        result = calculate_roadload_load_case(
            self._prior_row(rrc_N_per_kN=7.0),
            _request(mass=self._custom_mass(2000.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.rolling_rrc_source, "VDE_SPECIFIC")
        self.assertEqual(result.rolling_rrc_value, 7.0)
        self.assertEqual(result.rolling_rrc_rule_key, "rrc_N_per_kN")

    @patch("src.vde_core.vehicle_demand.load_case.preview_tire_roadload_from_row")
    def test_applicable_tire_db_wins_over_exact_prior(self, preview):
        preview.side_effect = [
            {
                "calculation": {"total_final_abc": {"A": 100.0, "B": 1.0, "C": 0.01}},
                "save_payload": {"rrc_N_per_kN": 6.9},
            },
            {
                "calculation": {"total_final_abc": {"A": 125.0, "B": 1.1, "C": 0.011}},
                "save_payload": {"rrc_N_per_kN": 6.9},
            },
        ]
        result = calculate_roadload_load_case(
            self._prior_row(front_tire_id=101, rear_tire_id=102),
            _request(mass=self._custom_mass(2000.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.rolling_rrc_source, "TIRE_DB")
        self.assertEqual(result.rolling_rrc_value, 6.9)
        self.assertEqual(result.rolling_delta_abc.A_N, 25.0)
        self.assertEqual(preview.call_count, 2)

    def test_exact_prior_is_applied_with_traceable_provenance(self):
        row = self._prior_row()
        result = calculate_roadload_load_case(
            row,
            _request(mass=self._custom_mass(2000.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.rolling_rrc_source, "RULE_ESTIMATED")
        self.assertEqual(result.rolling_rrc_value, 8.73)
        self.assertEqual(result.rolling_rrc_rule_key, "Midsize Cars|BEV|SS")
        self.assertEqual(result.rolling_correction_status, "APPLIED")
        self.assertGreater(result.rolling_delta_abc.A_N, 0.0)
        self.assertGreater(
            result.rolling_state.scenario_rolling_abc.A_N,
            result.rolling_state.baseline_rolling_abc.A_N,
        )
        self.assertTrue(any(point.rolling_force_N != 0.0 for point in result.force_curve_points))
        self.assertTrue(
            any(point.scenario_force_N != point.baseline_force_N for point in result.force_curve_points)
        )
        self.assertEqual(result.baseline_request.roadload_total.A_N, row["coast_A_N"])
        self.assertEqual(result.baseline_request.roadload_total.B_N_per_kph, row["coast_B_N_per_kph"])
        self.assertEqual(result.baseline_request.roadload_total.C_N_per_kph2, row["coast_C_N_per_kph2"])
        self.assertEqual(result.rolling_state.diagnostic.raw_transmission, "AUTOMATIC")
        self.assertEqual(result.rolling_state.diagnostic.gears, 1)
        self.assertEqual(result.rolling_state.diagnostic.resolved_transmission_key, "SS")
        self.assertTrue(any("persisted VDE is not modified" in warning for warning in result.warnings))

    def test_no_exact_match_is_unavailable_and_preserves_abc(self):
        row = self._prior_row(transmission_type="UNKNOWN", gears=8)
        result = calculate_roadload_load_case(
            row,
            _request(mass=self._custom_mass(2000.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.rolling_correction_status, "UNAVAILABLE")
        self.assertIsNone(result.rolling_delta_abc)
        self.assertEqual(result.scenario_request.roadload_total, result.baseline_request.roadload_total)
        self.assertTrue(any("Static authoritative roadload remains unchanged" in warning for warning in result.warnings))

    def test_generic_category_does_not_select_numeric_prior(self):
        result = calculate_roadload_load_case(
            self._prior_row(category="Cars"),
            _request(mass=self._custom_mass(2000.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.rolling_correction_status, "UNAVAILABLE")
        self.assertIsNone(result.rolling_rrc_value)

    def test_mass_decrease_produces_negative_rolling_delta(self):
        result = calculate_roadload_load_case(
            self._prior_row(),
            _request(mass=self._custom_mass(1200.0)),
            cycle_frame=_cycle(),
        )
        self.assertLess(result.rolling_delta_abc.A_N, 0.0)

    def test_no_mass_change_produces_zero_rolling_delta(self):
        result = calculate_roadload_load_case(self._prior_row(), _request(), cycle_frame=_cycle())
        self.assertEqual(result.rolling_rrc_source, "RULE_ESTIMATED")
        self.assertEqual(result.rolling_delta_abc.A_N, 0.0)
        self.assertEqual(result.rolling_delta_abc.B_N_per_kph, 0.0)
        self.assertEqual(result.rolling_delta_abc.C_N_per_kph2, 0.0)

    def test_disabling_prior_preserves_authoritative_abc(self):
        result = calculate_roadload_load_case(
            self._prior_row(),
            _request(mass=self._custom_mass(2000.0, use_prior=False)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.rolling_correction_status, "DISABLED")
        self.assertIsNone(result.rolling_delta_abc)
        self.assertEqual(result.scenario_request.roadload_total, result.baseline_request.roadload_total)

    def test_gcwr_rolling_uses_vehicle_mass_and_excludes_trailer(self):
        result = calculate_roadload_load_case(
            self._prior_row(),
            _request(
                mass=LoadCaseMass(mode=LoadCaseMassMode.GCWR, gcwr_kg=3200.0),
                trailer=LoadCaseTrailer(True, 800.0, 20.0, 0.2, 0.002),
            ),
            cycle_frame=_cycle(),
        )
        expected_vehicle_delta = 8.73 * G_MPS2 * (2400.0 - 1600.0) / 1000.0
        self.assertEqual(result.scenario_vehicle_mass_kg, 2400.0)
        self.assertEqual(result.scenario_system_mass_kg, 3200.0)
        self.assertAlmostEqual(result.rolling_delta_abc.A_N, expected_vehicle_delta, places=12)
        self.assertAlmostEqual(
            result.scenario_request.roadload_total.A_N,
            result.baseline_request.roadload_total.A_N + expected_vehicle_delta + 20.0,
            places=12,
        )

    def test_baseline_total_abc_is_immutable_and_grade_stays_outside_scenario_abc(self):
        row = self._prior_row()
        baseline_abc = (row["coast_A_N"], row["coast_B_N_per_kph"], row["coast_C_N_per_kph2"])
        result = calculate_roadload_load_case(
            row,
            _request(mass=self._custom_mass(2000.0), route=LoadCaseRoute(grade_pct=5.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(
            (
                result.baseline_request.roadload_total.A_N,
                result.baseline_request.roadload_total.B_N_per_kph,
                result.baseline_request.roadload_total.C_N_per_kph2,
            ),
            baseline_abc,
        )
        self.assertAlmostEqual(
            result.scenario_engine_request.roadload_total.A_N - result.scenario_request.roadload_total.A_N,
            result.grade_force_N,
            places=12,
        )


class AmbientAndTireTests(unittest.TestCase):
    def test_ambient_delta_applies_only_with_baseline_density_and_cda(self):
        row = _row(roadload_temperature_c=20.0, roadload_ambient_pressure_kpa=101.325)
        result = calculate_roadload_load_case(
            row,
            _request(environment=LoadCaseEnvironment(temperature_C=0.0, pressure_kPa=101.325)),
            cycle_frame=_cycle(),
        )
        self.assertGreater(result.condition_delta_C_N_per_kph2, 0.0)
        self.assertEqual(result.provenance["ambient_aero_delta"], "CALCULATED")

    def test_missing_baseline_ambient_makes_correction_unavailable(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(environment=LoadCaseEnvironment(temperature_C=0.0, pressure_kPa=101.325)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.condition_delta_C_N_per_kph2, 0.0)
        self.assertEqual(result.provenance["ambient_aero_delta"], "UNAVAILABLE")

    def test_explicit_scenario_density_wins(self):
        row = _row(roadload_temperature_c=20.0, roadload_ambient_pressure_kpa=101.325)
        result = calculate_roadload_load_case(
            row,
            _request(environment=LoadCaseEnvironment(temperature_C=35.0, pressure_kPa=90.0, air_density_kg_m3=1.4)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.scenario_request.ambient.air_density_kg_m3, 1.4)
        self.assertEqual(result.scenario_request.ambient.density_basis.value, "ASSUMED")
        self.assertEqual(result.provenance["scenario_air_density"], "ASSUMED")

    def test_tire_pressure_without_complete_owner_state_is_unavailable(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(environment=LoadCaseEnvironment(tire_pressure_delta_kPa=10.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.provenance["tire_pressure_delta"], "UNAVAILABLE")
        self.assertTrue(any("Tire-pressure correction unavailable" in warning for warning in result.warnings))

    @patch("src.vde_core.vehicle_demand.load_case.preview_tire_roadload_from_row")
    def test_supported_tire_owner_supplies_explicit_abc_delta(self, preview):
        preview.side_effect = [
            {"calculation": {"total_final_abc": {"A": 100.0, "B": 1.0, "C": 0.01}}},
            {"calculation": {"total_final_abc": {"A": 95.0, "B": 0.9, "C": 0.009}}},
        ]
        result = calculate_roadload_load_case(
            _row(front_pressure_psi=35.0, rear_pressure_psi=35.0),
            _request(environment=LoadCaseEnvironment(tire_pressure_delta_kPa=10.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.provenance["tire_pressure_delta"], "APPLIED")
        self.assertAlmostEqual(result.tire_delta_roadload.A_N, -5.0)
        self.assertAlmostEqual(result.scenario_request.roadload_total.A_N, 95.0)
        self.assertEqual(preview.call_count, 2)

    @patch("src.vde_core.vehicle_demand.load_case.preview_tire_roadload_from_row")
    def test_applied_zero_tire_delta_is_explicit_not_silent(self, preview):
        unchanged = {"calculation": {"total_final_abc": {"A": 100.0, "B": 1.0, "C": 0.01}}}
        preview.side_effect = [unchanged, unchanged]
        result = calculate_roadload_load_case(
            _row(front_pressure_psi=35.0, rear_pressure_psi=35.0),
            _request(environment=LoadCaseEnvironment(tire_pressure_delta_kPa=10.0)),
            cycle_frame=_cycle(),
        )
        self.assertEqual(result.provenance["tire_pressure_delta"], "APPLIED")
        self.assertEqual(result.tire_delta_roadload.A_N, 0.0)
        self.assertTrue(any("APPLIED" in note and "delta is zero" in note for note in result.explanatory_notes))


class FixedSpeedTests(unittest.TestCase):
    def test_zero_override_force_curve_matches_baseline(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED, fixed_speed=FixedSpeedSpec(90.0)),
        )
        self.assertTrue(result.force_curve_points)
        for point in result.force_curve_points:
            self.assertAlmostEqual(point.scenario_force_N, point.baseline_force_N, places=10)

    def test_grade_is_constant_force_offset_across_context_curve(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(
                analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED,
                fixed_speed=FixedSpeedSpec(90.0),
                route=LoadCaseRoute(grade_pct=6.0),
            ),
        )
        self.assertGreater(result.grade_force_N, 0.0)
        self.assertEqual({point.grade_force_N for point in result.force_curve_points}, {result.grade_force_N})
        for point in result.force_curve_points:
            self.assertAlmostEqual(
                point.scenario_force_N - point.baseline_force_N,
                result.grade_force_N,
                places=9,
            )

    def test_trailer_and_grade_curve_contributions_close_additively(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(
                analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED,
                fixed_speed=FixedSpeedSpec(93.0),
                trailer=LoadCaseTrailer(True, 500.0, 20.0, 0.2, 0.002),
                route=LoadCaseRoute(grade_pct=4.0),
            ),
        )
        for point in result.force_curve_points:
            expected_trailer = 20.0 + 0.2 * point.speed_kph + 0.002 * point.speed_kph**2
            self.assertAlmostEqual(point.trailer_force_N, expected_trailer, places=9)
            self.assertAlmostEqual(
                point.scenario_force_N,
                point.baseline_force_N
                + point.trailer_force_N
                + point.grade_force_N
                + point.condition_force_N,
                places=9,
            )

    def test_selected_curve_marker_matches_reported_wheel_force(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(
                analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED,
                fixed_speed=FixedSpeedSpec(93.0),
                route=LoadCaseRoute(grade_pct=3.0),
            ),
        )
        selected = [point for point in result.force_curve_points if point.selected_operating_point]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].speed_kph, 93.0)
        self.assertAlmostEqual(
            selected[0].scenario_force_N,
            result.steady_state_points[0].total.wheel_force_N,
            places=9,
        )

    def test_90_kph_matches_analytical_force_power_and_intensity(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED, fixed_speed=FixedSpeedSpec(90.0)),
        )
        point = result.steady_state_points[0].total
        expected_force = 100.0 + 1.0 * 90.0 + 0.01 * 90.0**2
        self.assertAlmostEqual(point.wheel_force_N, expected_force, places=9)
        self.assertAlmostEqual(point.wheel_power_kW, expected_force * (90.0 / 3.6) / 1000.0, places=9)
        self.assertAlmostEqual(point.wheel_energy_intensity_MJ_per_km, expected_force / 1000.0, places=9)

    def test_mass_alone_does_not_change_flat_steady_demand(self):
        baseline = calculate_roadload_load_case(
            _row(rrc_N_per_kN=None),
            _request(
                analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED,
                fixed_speed=FixedSpeedSpec(90.0),
                mass=LoadCaseMass(use_estimated_rrc_prior=False),
            ),
        )
        heavy = calculate_roadload_load_case(
            _row(rrc_N_per_kN=None),
            _request(
                analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED,
                fixed_speed=FixedSpeedSpec(90.0),
                mass=LoadCaseMass(
                    mode=LoadCaseMassMode.CUSTOM,
                    custom_vehicle_mass_kg=2500.0,
                    use_estimated_rrc_prior=False,
                ),
            ),
        )
        self.assertEqual(
            baseline.steady_state_points[0].total.wheel_force_N,
            heavy.steady_state_points[0].total.wheel_force_N,
        )
        self.assertTrue(all(point.baseline_force_N == point.scenario_force_N for point in heavy.force_curve_points))
        self.assertIn(
            "Mass changed, but steady-state force is unchanged at 0% grade because no active force contribution "
            "currently depends on the mass override.",
            heavy.explanatory_notes,
        )

    def test_grade_and_trailer_are_included_once(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(
                analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED,
                fixed_speed=FixedSpeedSpec(90.0),
                trailer=LoadCaseTrailer(True, 500.0, 20.0, 0.2, 0.002),
                route=LoadCaseRoute(grade_pct=6.0),
            ),
        )
        point = result.steady_state_points[0].total
        self.assertAlmostEqual(
            point.wheel_force_N,
            point.baseline_roadload_force_N + point.trailer_force_N + point.grade_force_N + point.condition_force_N,
            places=8,
        )

    def test_negative_grade_is_not_clamped(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(
                analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED,
                fixed_speed=FixedSpeedSpec(20.0),
                route=LoadCaseRoute(grade_pct=-20.0),
            ),
        )
        point = result.steady_state_points[0].total
        self.assertLess(point.wheel_force_N, 0.0)
        self.assertLess(point.wheel_power_kW, 0.0)
        self.assertEqual(point.energy_mode, EnergyMode.BRAKING)

    def test_zero_speed_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            calculate_roadload_load_case(
                _row(), _request(analysis_mode=LoadCaseAnalysisMode.FIXED_SPEED, fixed_speed=FixedSpeedSpec(0.0))
            )


class SpeedSweepTests(unittest.TestCase):
    def test_exact_deterministic_points(self):
        spec = SpeedSweepSpec(40.0, 120.0, 20.0)
        self.assertEqual(speed_sweep_values(spec), (40.0, 60.0, 80.0, 100.0, 120.0))
        result = calculate_roadload_load_case(
            _row(), _request(analysis_mode=LoadCaseAnalysisMode.SPEED_SWEEP, speed_sweep=spec)
        )
        self.assertEqual(tuple(point.speed_kph for point in result.steady_state_points), speed_sweep_values(spec))

    def test_invalid_step_and_range(self):
        with self.assertRaises(ValueError):
            speed_sweep_values(SpeedSweepSpec(40.0, 120.0, 0.0))
        with self.assertRaises(ValueError):
            speed_sweep_values(SpeedSweepSpec(120.0, 40.0, 20.0))
        with self.assertRaisesRegex(ValueError, "limit"):
            speed_sweep_values(SpeedSweepSpec(1.0, 600.0, 1.0))

    def test_same_load_case_only_speed_changes(self):
        result = calculate_roadload_load_case(
            _row(),
            _request(
                analysis_mode=LoadCaseAnalysisMode.SPEED_SWEEP,
                speed_sweep=SpeedSweepSpec(40.0, 80.0, 20.0),
                route=LoadCaseRoute(grade_pct=3.0),
            ),
        )
        self.assertEqual({point.total.grade_force_N for point in result.steady_state_points}, {result.grade_force_N})
        self.assertTrue(all(point.total.roadload_basis is RoadloadBasis.TOTAL for point in result.steady_state_points))


if __name__ == "__main__":
    unittest.main()
