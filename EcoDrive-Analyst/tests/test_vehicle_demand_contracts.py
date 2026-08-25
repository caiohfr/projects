import json
import math
import unittest

from src.vde_core.vehicle_demand import (
    AmbientState,
    EnergyMode,
    Provenance,
    RoadloadBasis,
    RoadloadCoefficients,
    VehicleDemandProfile,
    VehicleDemandRequest,
    VehicleDemandResult,
    VehicleDemandSummary,
    ambient_state_from_dict,
    to_serializable,
    vehicle_demand_profile_from_dict,
    vehicle_demand_request_from_dict,
    vehicle_demand_result_from_dict,
    vehicle_demand_summary_from_dict,
)


def _sample_request() -> VehicleDemandRequest:
    return VehicleDemandRequest(
        source_kind="VDE_ONLY",
        vde_id=101,
        fuelcons_id=None,
        label="2026 Sample Vehicle",
        cycle_name="FTP75_HWFET",
        cycle_source="STANDARD",
        cycle_version="1.0",
        test_mass_kg=1750.0,
        roadload_total=RoadloadCoefficients(A_N=120.0, B_N_per_kph=0.5, C_N_per_kph2=0.03),
        roadload_net=RoadloadCoefficients(A_N=100.0, B_N_per_kph=0.4, C_N_per_kph2=0.025),
        rrc_n_per_kn=8.5,
        cda_m2=0.65,
        ambient=AmbientState(
            temperature_C=20.0,
            pressure_kPa=101.325,
            air_density_kg_m3=1.2,
            temperature_basis=Provenance.REGULATORY_REFERENCE,
            pressure_basis=Provenance.REGULATORY_REFERENCE,
            density_basis=Provenance.CALCULATED,
        ),
        provenance={"roadload_total": "SOURCE", "roadload_net": "RESOLVED"},
        model_version="VDE_SETUP_V22",
    )


def _sample_profile(basis: RoadloadBasis) -> VehicleDemandProfile:
    n = 5
    return VehicleDemandProfile(
        roadload_basis=basis,
        time_s=tuple(float(i) for i in range(n)),
        speed_mps=(0.0, 2.0, 4.0, 4.0, 0.0),
        accel_mps2=(2.0, 2.0, 0.0, -4.0, -4.0),
        authoritative_roadload_force_N=(120.0, 122.0, 128.0, 128.0, 120.0),
        inertial_force_N=(3500.0, 3500.0, 0.0, -7000.0, -7000.0),
        tractive_force_N=(3620.0, 3622.0, 128.0, -6872.0, -6880.0),
        authoritative_roadload_power_W=(0.0, 244.0, 512.0, 512.0, 0.0),
        inertial_power_W=(0.0, 7000.0, 0.0, -28000.0, 0.0),
        tractive_power_W=(0.0, 7244.0, 512.0, -27488.0, 0.0),
        energy_mode=(EnergyMode.IDLE, EnergyMode.TRACTION, EnergyMode.TRACTION, EnergyMode.BRAKING, EnergyMode.IDLE),
    )


def _sample_summary(basis: RoadloadBasis) -> VehicleDemandSummary:
    return VehicleDemandSummary(
        roadload_basis=basis,
        distance_km=17.8,
        roadload_energy_MJ=3.1,
        positive_inertial_work_MJ=1.2,
        positive_tractive_energy_MJ=4.3,
        braking_energy_required_MJ=0.9,
        vde_mj_per_km=0.55 if basis is RoadloadBasis.TOTAL else 0.50,
        availability=frozenset({"vde", "roadload_energy"}),
        warnings=(),
        provenance={"vde": "CALCULATED"},
        cycle_name="FTP75_HWFET",
        cycle_source="STANDARD",
        model_version="VDE_SETUP_V22",
    )


class RoadloadBasisAndEnergyModeContractTests(unittest.TestCase):
    def test_roadload_basis_values_are_stable(self):
        self.assertEqual(RoadloadBasis.TOTAL.value, "TOTAL")
        self.assertEqual(RoadloadBasis.NET.value, "NET")
        self.assertEqual({b.value for b in RoadloadBasis}, {"TOTAL", "NET"})

    def test_energy_mode_values_are_stable_and_closed(self):
        self.assertEqual(
            {m.value for m in EnergyMode},
            {"IDLE", "TRACTION", "COASTING", "BRAKING"},
        )

    def test_roadload_basis_is_json_string_valued(self):
        self.assertEqual(json.dumps(RoadloadBasis.TOTAL.value), '"TOTAL"')


class VehicleDemandProfileShapeValidationTests(unittest.TestCase):
    def test_matching_lengths_are_accepted(self):
        profile = _sample_profile(RoadloadBasis.TOTAL)
        self.assertEqual(len(profile.time_s), 5)

    def test_mismatched_required_series_length_is_rejected(self):
        with self.assertRaises(ValueError):
            VehicleDemandProfile(
                roadload_basis=RoadloadBasis.TOTAL,
                time_s=tuple(float(i) for i in range(100)),
                speed_mps=tuple(float(i) for i in range(100)),
                accel_mps2=tuple(float(i) for i in range(99)),  # short by one
                authoritative_roadload_force_N=tuple(float(i) for i in range(100)),
                inertial_force_N=tuple(float(i) for i in range(100)),
                tractive_force_N=tuple(float(i) for i in range(100)),
                authoritative_roadload_power_W=tuple(float(i) for i in range(100)),
                inertial_power_W=tuple(float(i) for i in range(100)),
                tractive_power_W=tuple(float(i) for i in range(100)),
                energy_mode=tuple(EnergyMode.IDLE for _ in range(100)),
            )

    def test_mismatched_optional_series_length_is_rejected(self):
        base = _sample_profile(RoadloadBasis.TOTAL)
        with self.assertRaises(ValueError):
            VehicleDemandProfile(
                roadload_basis=base.roadload_basis,
                time_s=base.time_s,
                speed_mps=base.speed_mps,
                accel_mps2=base.accel_mps2,
                authoritative_roadload_force_N=base.authoritative_roadload_force_N,
                inertial_force_N=base.inertial_force_N,
                tractive_force_N=base.tractive_force_N,
                authoritative_roadload_power_W=base.authoritative_roadload_power_W,
                inertial_power_W=base.inertial_power_W,
                tractive_power_W=base.tractive_power_W,
                energy_mode=base.energy_mode,
                known_rolling_force_N=(1.0, 2.0, 3.0),  # short
            )

    def test_absent_optional_series_is_allowed(self):
        profile = _sample_profile(RoadloadBasis.NET)
        self.assertIsNone(profile.known_rolling_force_N)
        self.assertIsNone(profile.known_aero_force_N)
        self.assertIsNone(profile.residual_roadload_force_N)


class TotalNetDistinctnessTests(unittest.TestCase):
    def test_total_and_net_summaries_are_distinct_values(self):
        total = _sample_summary(RoadloadBasis.TOTAL)
        net = _sample_summary(RoadloadBasis.NET)
        self.assertEqual(total.roadload_basis, RoadloadBasis.TOTAL)
        self.assertEqual(net.roadload_basis, RoadloadBasis.NET)
        self.assertNotEqual(total.vde_mj_per_km, net.vde_mj_per_km)

    def test_result_requires_total_summary_to_be_total_basis(self):
        net_summary = _sample_summary(RoadloadBasis.NET)
        with self.assertRaises(ValueError):
            VehicleDemandResult(total_summary=net_summary, net_summary=None)

    def test_result_requires_net_summary_to_be_net_basis_when_present(self):
        total_summary = _sample_summary(RoadloadBasis.TOTAL)
        wrong_basis_summary = _sample_summary(RoadloadBasis.TOTAL)
        with self.assertRaises(ValueError):
            VehicleDemandResult(total_summary=total_summary, net_summary=wrong_basis_summary)

    def test_result_net_summary_is_optional_with_no_fallback_to_total(self):
        total_summary = _sample_summary(RoadloadBasis.TOTAL)
        result = VehicleDemandResult(total_summary=total_summary, net_summary=None)
        self.assertIsNone(result.net_summary)
        self.assertIsNotNone(result.total_summary)

    def test_request_net_roadload_is_optional_and_independent_of_total(self):
        request = _sample_request()
        object.__setattr__(request, "roadload_net", None)
        self.assertIsNotNone(request.roadload_total)
        self.assertIsNone(request.roadload_net)


class ZeroVsMissingTests(unittest.TestCase):
    def test_zero_vde_is_not_none(self):
        summary = VehicleDemandSummary(roadload_basis=RoadloadBasis.TOTAL, vde_mj_per_km=0.0)
        self.assertEqual(summary.vde_mj_per_km, 0.0)
        self.assertIsNotNone(summary.vde_mj_per_km)

    def test_missing_vde_is_none_not_zero(self):
        summary = VehicleDemandSummary(roadload_basis=RoadloadBasis.TOTAL, vde_mj_per_km=None)
        self.assertIsNone(summary.vde_mj_per_km)
        self.assertNotEqual(summary.vde_mj_per_km, 0.0)

    def test_zero_roadload_coefficient_survives_serialization(self):
        coeffs = RoadloadCoefficients(A_N=0.0, B_N_per_kph=0.0, C_N_per_kph2=0.03)
        serialized = to_serializable(coeffs)
        self.assertEqual(serialized["A_N"], 0.0)
        self.assertIsNotNone(serialized["A_N"])

    def test_nan_is_serialized_as_none_not_zero(self):
        summary = VehicleDemandSummary(roadload_basis=RoadloadBasis.TOTAL, vde_mj_per_km=float("nan"))
        serialized = to_serializable(summary)
        self.assertIsNone(serialized["vde_mj_per_km"])


class SerializationRoundTripTests(unittest.TestCase):
    def test_ambient_state_round_trip(self):
        ambient = AmbientState(
            temperature_C=25.0,
            pressure_kPa=100.0,
            air_density_kg_m3=None,
            temperature_basis=Provenance.SOURCE,
            pressure_basis=Provenance.ASSUMED,
            density_basis=None,
        )
        serialized = to_serializable(ambient)
        json.dumps(serialized)  # must be plain-JSON-safe
        restored = ambient_state_from_dict(serialized)
        self.assertEqual(restored, ambient)

    def test_vehicle_demand_request_round_trip(self):
        request = _sample_request()
        serialized = to_serializable(request)
        json.dumps(serialized)
        restored = vehicle_demand_request_from_dict(serialized)
        self.assertEqual(restored, request)

    def test_vehicle_demand_request_with_no_net_roadload_round_trips(self):
        request = _sample_request()
        object.__setattr__(request, "roadload_net", None)
        serialized = to_serializable(request)
        json.dumps(serialized)
        restored = vehicle_demand_request_from_dict(serialized)
        self.assertIsNone(restored.roadload_net)
        self.assertEqual(restored, request)

    def test_vehicle_demand_profile_round_trip(self):
        profile = _sample_profile(RoadloadBasis.TOTAL)
        serialized = to_serializable(profile)
        json.dumps(serialized)
        restored = vehicle_demand_profile_from_dict(serialized)
        self.assertEqual(restored, profile)
        self.assertEqual(restored.energy_mode, profile.energy_mode)

    def test_vehicle_demand_summary_round_trip(self):
        summary = _sample_summary(RoadloadBasis.NET)
        serialized = to_serializable(summary)
        json.dumps(serialized)
        restored = vehicle_demand_summary_from_dict(serialized)
        self.assertEqual(restored, summary)

    def test_vehicle_demand_result_round_trip_with_net(self):
        result = VehicleDemandResult(
            total_summary=_sample_summary(RoadloadBasis.TOTAL),
            net_summary=_sample_summary(RoadloadBasis.NET),
            metadata={"note": "sample"},
        )
        serialized = to_serializable(result)
        json.dumps(serialized)
        restored = vehicle_demand_result_from_dict(serialized)
        self.assertEqual(restored, result)

    def test_vehicle_demand_result_round_trip_without_net(self):
        result = VehicleDemandResult(total_summary=_sample_summary(RoadloadBasis.TOTAL), net_summary=None)
        serialized = to_serializable(result)
        self.assertIsNone(serialized["net_summary"])
        restored = vehicle_demand_result_from_dict(serialized)
        self.assertIsNone(restored.net_summary)


if __name__ == "__main__":
    unittest.main()
