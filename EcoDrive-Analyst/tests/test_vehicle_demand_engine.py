import json
import unittest

import pandas as pd

from src.vde_core.roadload.tire_model import G_MPS2
from src.vde_core.vde_calc import compute_vde_net
from src.vde_core.vehicle_demand import (
    AmbientState,
    EnergyMode,
    Provenance,
    RoadloadBasis,
    RoadloadCoefficients,
    VehicleDemandRequest,
    build_vehicle_demand_profile,
    calculate_vehicle_demand,
    summarize_vehicle_demand,
    to_serializable,
    vehicle_demand_result_from_dict,
)
from src.vde_core.vehicle_demand.physics import POWER_EPSILON_W, classify_energy_mode


def _request(
    *,
    total_abc=(120.0, 2.0, 0.02),
    net_abc=None,
    mass_kg=1500.0,
    rrc_n_per_kn=None,
    cda_m2=None,
    ambient=None,
) -> VehicleDemandRequest:
    net = None
    if net_abc is not None:
        net = RoadloadCoefficients(A_N=net_abc[0], B_N_per_kph=net_abc[1], C_N_per_kph2=net_abc[2])
    return VehicleDemandRequest(
        source_kind="VDE_ONLY",
        vde_id=1,
        fuelcons_id=None,
        label="QA Vehicle",
        cycle_name="SYNTHETIC",
        cycle_source="SYNTHETIC",
        cycle_version=None,
        test_mass_kg=mass_kg,
        roadload_total=RoadloadCoefficients(A_N=total_abc[0], B_N_per_kph=total_abc[1], C_N_per_kph2=total_abc[2]),
        roadload_net=net,
        rrc_n_per_kn=rrc_n_per_kn,
        cda_m2=cda_m2,
        ambient=ambient if ambient is not None else AmbientState(),
        provenance={},
        model_version="QA",
    )


def _constant_speed_cycle(speed_mps=20.0, duration_s=50, dt=1.0) -> pd.DataFrame:
    n = int(duration_s / dt) + 1
    t = [i * dt for i in range(n)]
    v = [speed_mps] * n
    return pd.DataFrame({"t": t, "v": v})


def _linear_cycle(v0: float, accel_mps2: float, duration_s: float, dt: float = 1.0) -> pd.DataFrame:
    n = int(duration_s / dt) + 1
    t = [i * dt for i in range(n)]
    v = [v0 + accel_mps2 * ti for ti in t]
    assert all(value >= 0 for value in v), "test fixture must stay non-negative to avoid IDLE ambiguity"
    return pd.DataFrame({"t": t, "v": v})


def _trapezoidal_cycle() -> pd.DataFrame:
    """Accel 0->20 m/s over 20s, cruise 30s, decel 20->0 over 20s, idle 10s."""
    t: list[float] = []
    v: list[float] = []
    cur_t = 0.0
    for i in range(21):
        t.append(cur_t)
        v.append(20.0 * i / 20.0)
        cur_t += 1.0
    for _ in range(30):
        t.append(cur_t)
        v.append(20.0)
        cur_t += 1.0
    for i in range(1, 21):
        t.append(cur_t)
        v.append(20.0 * (1 - i / 20.0))
        cur_t += 1.0
    for _ in range(10):
        t.append(cur_t)
        v.append(0.0)
        cur_t += 1.0
    return pd.DataFrame({"t": t, "v": v})


class ConstantSpeedTests(unittest.TestCase):
    """QA-1."""

    def test_constant_speed_has_zero_inertial_force_and_pure_traction(self):
        request = _request(total_abc=(100.0, 1.0, 0.01), mass_kg=1500.0)
        cycle = _constant_speed_cycle(speed_mps=20.0)

        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        summary = summarize_vehicle_demand(profile, request)

        for value in profile.inertial_force_N:
            self.assertAlmostEqual(value, 0.0, places=6)
        for tractive, roadload in zip(profile.tractive_force_N, profile.authoritative_roadload_force_N):
            self.assertAlmostEqual(tractive, roadload, places=6)
        self.assertTrue(all(mode is EnergyMode.TRACTION for mode in profile.energy_mode))
        self.assertGreater(summary.positive_tractive_energy_MJ, 0.0)
        self.assertAlmostEqual(summary.braking_energy_required_MJ, 0.0, places=9)
        self.assertAlmostEqual(summary.positive_inertial_work_MJ, 0.0, places=9)


class NaturalCoastTests(unittest.TestCase):
    """QA-2. Built as an exact analytical solution of m*a = -F_road (F_road
    constant, B=C=0), so v(t) is linear and np.gradient recovers the exact
    acceleration -- tractive_power is exactly ~0, not approximately so.
    """

    def test_natural_coast_has_near_zero_tractive_power(self):
        mass_kg = 1500.0
        F_road = 50.0
        a0 = -F_road / mass_kg
        request = _request(total_abc=(F_road, 0.0, 0.0), mass_kg=mass_kg)
        cycle = _linear_cycle(v0=20.0, accel_mps2=a0, duration_s=100.0)

        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

        for power in profile.tractive_power_W:
            self.assertAlmostEqual(power, 0.0, places=6)
        self.assertTrue(all(mode is EnergyMode.COASTING for mode in profile.energy_mode))


class HardDecelerationTests(unittest.TestCase):
    """QA-3."""

    def test_hard_deceleration_is_braking_with_positive_required_energy(self):
        mass_kg = 1500.0
        request = _request(total_abc=(50.0, 0.0, 0.0), mass_kg=mass_kg)
        cycle = _linear_cycle(v0=20.0, accel_mps2=-2.0, duration_s=5.0)

        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        summary = summarize_vehicle_demand(profile, request)

        self.assertTrue(all(mode is EnergyMode.BRAKING for mode in profile.energy_mode))
        self.assertGreater(summary.braking_energy_required_MJ, 0.0)


class GentleDecelerationTests(unittest.TestCase):
    """QA-4 -- deceleration must not be automatically classified as braking."""

    def test_gentle_deceleration_stays_traction(self):
        mass_kg = 1500.0
        F_road = 50.0
        request = _request(total_abc=(F_road, 0.0, 0.0), mass_kg=mass_kg)
        # Gentler than the natural-coast deceleration (-F_road/mass_kg == -0.0333...).
        cycle = _linear_cycle(v0=20.0, accel_mps2=-0.01, duration_s=10.0)

        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

        for accel in profile.accel_mps2:
            self.assertLess(accel, 0.0)
        self.assertTrue(all(mode is EnergyMode.TRACTION for mode in profile.energy_mode))


class AeroDirectionalityTests(unittest.TestCase):
    """QA-5, QA-6, QA-7."""

    def test_higher_cda_yields_higher_known_aero_energy(self):
        cycle = _constant_speed_cycle(speed_mps=25.0)
        low = _request(cda_m2=0.3, ambient=AmbientState(air_density_kg_m3=1.2, density_basis=Provenance.SOURCE))
        high = _request(cda_m2=0.6, ambient=AmbientState(air_density_kg_m3=1.2, density_basis=Provenance.SOURCE))

        low_summary = summarize_vehicle_demand(build_vehicle_demand_profile(low, cycle, RoadloadBasis.TOTAL), low)
        high_summary = summarize_vehicle_demand(build_vehicle_demand_profile(high, cycle, RoadloadBasis.TOTAL), high)

        self.assertGreater(high_summary.known_aero_energy_MJ, low_summary.known_aero_energy_MJ)

    def test_lower_temperature_yields_higher_known_aero_via_higher_density(self):
        cycle = _constant_speed_cycle(speed_mps=25.0)
        cold = _request(cda_m2=0.5, ambient=AmbientState(temperature_C=0.0, pressure_kPa=101.325))
        warm = _request(cda_m2=0.5, ambient=AmbientState(temperature_C=35.0, pressure_kPa=101.325))

        cold_summary = summarize_vehicle_demand(build_vehicle_demand_profile(cold, cycle, RoadloadBasis.TOTAL), cold)
        warm_summary = summarize_vehicle_demand(build_vehicle_demand_profile(warm, cycle, RoadloadBasis.TOTAL), warm)

        self.assertGreater(cold_summary.known_aero_energy_MJ, warm_summary.known_aero_energy_MJ)
        self.assertAlmostEqual(cold_summary.roadload_energy_MJ, warm_summary.roadload_energy_MJ, places=9)

    def test_higher_pressure_yields_higher_known_aero_via_higher_density(self):
        cycle = _constant_speed_cycle(speed_mps=25.0)
        low_p = _request(cda_m2=0.5, ambient=AmbientState(temperature_C=20.0, pressure_kPa=90.0))
        high_p = _request(cda_m2=0.5, ambient=AmbientState(temperature_C=20.0, pressure_kPa=101.325))

        low_summary = summarize_vehicle_demand(build_vehicle_demand_profile(low_p, cycle, RoadloadBasis.TOTAL), low_p)
        high_summary = summarize_vehicle_demand(build_vehicle_demand_profile(high_p, cycle, RoadloadBasis.TOTAL), high_p)

        self.assertGreater(high_summary.known_aero_energy_MJ, low_summary.known_aero_energy_MJ)


class RollingDirectionalityTests(unittest.TestCase):
    """QA-8."""

    def test_higher_rrc_yields_higher_known_rolling_energy(self):
        cycle = _constant_speed_cycle(speed_mps=20.0)
        low = _request(rrc_n_per_kn=6.0, mass_kg=1500.0)
        high = _request(rrc_n_per_kn=10.0, mass_kg=1500.0)

        low_summary = summarize_vehicle_demand(build_vehicle_demand_profile(low, cycle, RoadloadBasis.TOTAL), low)
        high_summary = summarize_vehicle_demand(build_vehicle_demand_profile(high, cycle, RoadloadBasis.TOTAL), high)

        self.assertGreater(high_summary.known_rolling_energy_MJ, low_summary.known_rolling_energy_MJ)

        expected_low_force = 6.0 * 1500.0 * G_MPS2 / 1000.0
        profile = build_vehicle_demand_profile(low, cycle, RoadloadBasis.TOTAL)
        for force in profile.known_rolling_force_N:
            self.assertAlmostEqual(force, expected_low_force, places=6)


class MissingContributionTests(unittest.TestCase):
    """QA-9, QA-10."""

    def test_missing_rrc_leaves_rolling_unavailable_but_vde_still_computed(self):
        request = _request(rrc_n_per_kn=None)
        cycle = _constant_speed_cycle()

        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        summary = summarize_vehicle_demand(profile, request)

        self.assertIsNone(profile.known_rolling_force_N)
        self.assertIsNone(summary.known_rolling_energy_MJ)
        self.assertIsNotNone(summary.roadload_energy_MJ)
        self.assertIsNotNone(summary.vde_mj_per_km)
        self.assertIn("rolling", summary.provenance)
        self.assertEqual(summary.provenance["rolling"], "UNAVAILABLE")

    def test_missing_cda_leaves_aero_unavailable_without_inferring_from_c(self):
        request = _request(total_abc=(100.0, 1.0, 0.05), cda_m2=None)
        cycle = _constant_speed_cycle()

        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        summary = summarize_vehicle_demand(profile, request)

        self.assertIsNone(profile.known_aero_force_N)
        self.assertIsNone(summary.known_aero_energy_MJ)
        self.assertEqual(summary.provenance["aero"], "UNAVAILABLE")


class RoadloadClosureTests(unittest.TestCase):
    """QA-11."""

    def test_known_contributions_plus_residual_close_authoritative_roadload(self):
        request = _request(
            total_abc=(120.0, 2.0, 0.02),
            rrc_n_per_kn=8.0,
            cda_m2=0.6,
            ambient=AmbientState(air_density_kg_m3=1.2, density_basis=Provenance.SOURCE),
        )
        cycle = _trapezoidal_cycle()

        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)

        for rolling, aero, residual, authoritative in zip(
            profile.known_rolling_force_N,
            profile.known_aero_force_N,
            profile.residual_roadload_force_N,
            profile.authoritative_roadload_force_N,
        ):
            self.assertAlmostEqual(rolling + aero + residual, authoritative, places=6)


class NegativeResidualTests(unittest.TestCase):
    """QA-12."""

    def test_over_attributed_known_contributions_preserve_negative_residual(self):
        request = _request(
            total_abc=(1.0, 0.0, 0.0),  # tiny authoritative roadload
            rrc_n_per_kn=8.0,  # ~117 N alone, already >> 1 N
            cda_m2=0.6,
            mass_kg=1500.0,
            ambient=AmbientState(air_density_kg_m3=1.2, density_basis=Provenance.SOURCE),
        )
        cycle = _constant_speed_cycle(speed_mps=20.0)

        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        summary = summarize_vehicle_demand(profile, request)

        self.assertTrue(all(force < 0 for force in profile.residual_roadload_force_N))
        self.assertLess(summary.residual_roadload_energy_MJ, 0.0)
        self.assertTrue(any("negative" in warning.lower() for warning in summary.warnings))


class BrakingZeroTests(unittest.TestCase):
    """QA-13."""

    def test_no_negative_tractive_power_yields_exactly_zero_braking_energy(self):
        request = _request(total_abc=(100.0, 1.0, 0.01))
        cycle = _constant_speed_cycle(speed_mps=15.0)

        summary = summarize_vehicle_demand(build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL), request)

        self.assertEqual(summary.braking_energy_required_MJ, 0.0)


class TotalNetIndependenceTests(unittest.TestCase):
    """QA-14, QA-15."""

    def test_total_and_net_profiles_and_summaries_differ(self):
        request = _request(total_abc=(120.0, 2.0, 0.02), net_abc=(100.0, 1.5, 0.015))
        cycle = _trapezoidal_cycle()

        total_profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        net_profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.NET)
        total_summary = summarize_vehicle_demand(total_profile, request)
        net_summary = summarize_vehicle_demand(net_profile, request)

        self.assertNotEqual(total_profile.authoritative_roadload_force_N, net_profile.authoritative_roadload_force_N)
        self.assertNotEqual(total_summary.vde_mj_per_km, net_summary.vde_mj_per_km)

    def test_missing_net_never_falls_back_to_total(self):
        request = _request(total_abc=(120.0, 2.0, 0.02), net_abc=None)
        cycle = _trapezoidal_cycle()

        net_profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.NET)
        self.assertIsNone(net_profile)

        result = calculate_vehicle_demand(request, cycle)
        self.assertIsNotNone(result.total_summary)
        self.assertIsNone(result.net_summary)


class SerializationAfterPhysicsTests(unittest.TestCase):
    """QA-16."""

    def test_real_vehicle_demand_result_round_trips_through_json(self):
        request = _request(
            total_abc=(120.0, 2.0, 0.02),
            net_abc=(100.0, 1.5, 0.015),
            rrc_n_per_kn=8.0,
            cda_m2=0.6,
            ambient=AmbientState(air_density_kg_m3=1.2, density_basis=Provenance.SOURCE),
        )
        cycle = _trapezoidal_cycle()

        result = calculate_vehicle_demand(request, cycle)

        serialized = to_serializable(result)
        json_text = json.dumps(serialized)
        restored = vehicle_demand_result_from_dict(json.loads(json_text))

        self.assertEqual(restored, result)


class CanonicalVdeReconciliationTests(unittest.TestCase):
    """Sprint 9B Sec 53 -- the most important gate: VehicleDemandSummary.vde_mj_per_km
    must reconcile with the project's existing canonical VDE calculation
    (compute_vde_net), not a second implementation of it.
    """

    def _assert_reconciles(self, total_abc, mass_kg, cycle):
        request = _request(total_abc=total_abc, mass_kg=mass_kg)
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.TOTAL)
        summary = summarize_vehicle_demand(profile, request)

        canonical = compute_vde_net(cycle, *total_abc, mass_kg)

        self.assertAlmostEqual(summary.vde_mj_per_km, canonical["MJ_km"], places=9)
        self.assertAlmostEqual(summary.positive_tractive_energy_MJ, canonical["MJ_total"], places=9)
        self.assertAlmostEqual(summary.distance_km, canonical["km"], places=9)

    def test_reconciles_on_constant_speed_cycle(self):
        self._assert_reconciles((100.0, 2.0, 0.01), 1500.0, _constant_speed_cycle())

    def test_reconciles_on_linear_accel_cycle(self):
        self._assert_reconciles((120.0, 2.0, 0.02), 1750.0, _linear_cycle(v0=0.0, accel_mps2=1.0, duration_s=10.0))

    def test_reconciles_on_trapezoidal_multi_phase_cycle(self):
        self._assert_reconciles((120.0, 2.0, 0.02), 1750.0, _trapezoidal_cycle())

    def test_reconciles_for_net_basis_too(self):
        cycle = _trapezoidal_cycle()
        net_abc = (100.0, 1.5, 0.015)
        request = _request(total_abc=(120.0, 2.0, 0.02), net_abc=net_abc, mass_kg=1750.0)
        profile = build_vehicle_demand_profile(request, cycle, RoadloadBasis.NET)
        summary = summarize_vehicle_demand(profile, request)

        canonical = compute_vde_net(cycle, *net_abc, 1750.0)

        self.assertAlmostEqual(summary.vde_mj_per_km, canonical["MJ_km"], places=9)


class EnergyModeEpsilonTests(unittest.TestCase):
    def test_near_zero_speed_is_idle_regardless_of_power_sign(self):
        modes = classify_energy_mode([0.0, 0.02], [1000.0, -1000.0])
        self.assertEqual(modes, (EnergyMode.IDLE, EnergyMode.IDLE))

    def test_power_within_epsilon_band_is_coasting(self):
        modes = classify_energy_mode([10.0], [POWER_EPSILON_W - 1.0])
        self.assertEqual(modes, (EnergyMode.COASTING,))

    def test_power_just_above_epsilon_is_traction(self):
        modes = classify_energy_mode([10.0], [POWER_EPSILON_W + 1.0])
        self.assertEqual(modes, (EnergyMode.TRACTION,))

    def test_power_just_below_negative_epsilon_is_braking(self):
        modes = classify_energy_mode([10.0], [-POWER_EPSILON_W - 1.0])
        self.assertEqual(modes, (EnergyMode.BRAKING,))


if __name__ == "__main__":
    unittest.main()
