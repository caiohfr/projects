# src/vde_core/vehicle_demand/engine.py
# -----------------------------------------------------------------------------
# Vehicle Demand physics engine (Sprint 9B).
#
# Turns a VehicleDemandRequest + cycle trace into a VehicleDemandProfile and
# VehicleDemandSummary. This module does NOT reimplement VDE physics: the
# road-load/inertial/tractive-force math and cycle validation are delegated
# entirely to src/vde_core/vde_calc.py (extract_cycle_arrays,
# compute_vde_series) -- the exact functions compute_vde_net itself uses --
# so a VehicleDemandSummary.vde_mj_per_km computed here is, by construction,
# the same calculation as the project's canonical VDE, not a second one.
#
# The only physics genuinely new in this module is: air density from
# temperature/pressure, known rolling/aero force (physics.py), residual
# roadload, and EnergyMode classification.
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd

from src.vde_core.vde_calc import compute_vde_series, extract_cycle_arrays

from .contracts import (
    RoadloadBasis,
    RoadloadCoefficients,
    VehicleDemandProfile,
    VehicleDemandRequest,
    VehicleDemandResult,
    VehicleDemandSummary,
)
from .physics import classify_energy_mode, known_aero_force_N, known_rolling_force_N, resolve_air_density

VEHICLE_DEMAND_ENGINE_VERSION = "0.1"


def _select_roadload(request: VehicleDemandRequest, roadload_basis: RoadloadBasis) -> RoadloadCoefficients | None:
    if roadload_basis is RoadloadBasis.TOTAL:
        return request.roadload_total
    return request.roadload_net


def build_vehicle_demand_profile(
    request: VehicleDemandRequest,
    cycle_frame: pd.DataFrame,
    roadload_basis: RoadloadBasis,
) -> VehicleDemandProfile | None:
    """Build the time-resolved physical profile for ONE RoadloadBasis.

    Returns None when the requested basis (or the effective test mass every
    Profile needs for its inertial term) is not available on the request --
    never a fabricated substitute (e.g. NET from TOTAL). A structurally
    invalid cycle_frame still raises ValueError, via
    vde_calc.extract_cycle_arrays -- the same validation compute_vde_net has
    always applied.
    """
    coefficients = _select_roadload(request, roadload_basis)
    if coefficients is None or request.test_mass_kg is None:
        return None

    t, v = extract_cycle_arrays(cycle_frame)
    mass_kg = float(request.test_mass_kg)

    series = compute_vde_series(
        t,
        v,
        coefficients.A_N,
        coefficients.B_N_per_kph,
        coefficients.C_N_per_kph2,
        mass_kg,
    )
    F_road = series["F_road_N"]
    F_inertia = series["F_inertia_N"]
    F_tractive = series["F_tractive_N"]
    tractive_power = series["P_W"]
    roadload_power = F_road * v
    inertial_power = F_inertia * v

    rho, _density_basis, _density_warnings = resolve_air_density(request.ambient)
    aero_force = known_aero_force_N(request.cda_m2, rho, v)

    rolling_force_scalar = known_rolling_force_N(request.rrc_n_per_kn, mass_kg)
    rolling_force = np.full_like(v, rolling_force_scalar) if rolling_force_scalar is not None else None

    residual_force = F_road.copy()
    if rolling_force is not None:
        residual_force = residual_force - rolling_force
    if aero_force is not None:
        residual_force = residual_force - aero_force

    energy_mode = classify_energy_mode(v, tractive_power)

    return VehicleDemandProfile(
        roadload_basis=roadload_basis,
        time_s=tuple(t.tolist()),
        speed_mps=tuple(v.tolist()),
        accel_mps2=tuple(series["a_mps2"].tolist()),
        authoritative_roadload_force_N=tuple(F_road.tolist()),
        inertial_force_N=tuple(F_inertia.tolist()),
        tractive_force_N=tuple(F_tractive.tolist()),
        authoritative_roadload_power_W=tuple(roadload_power.tolist()),
        inertial_power_W=tuple(inertial_power.tolist()),
        tractive_power_W=tuple(tractive_power.tolist()),
        energy_mode=energy_mode,
        known_rolling_force_N=tuple(rolling_force.tolist()) if rolling_force is not None else None,
        known_aero_force_N=tuple(aero_force.tolist()) if aero_force is not None else None,
        residual_roadload_force_N=tuple(residual_force.tolist()),
    )


def summarize_vehicle_demand(profile: VehicleDemandProfile, request: VehicleDemandRequest) -> VehicleDemandSummary:
    """Cycle-level aggregates for a Profile already built for ONE RoadloadBasis.

    Integration mirrors compute_vde_net exactly (np.trapezoid, positive-power
    clipping only for the *_positive_* fields, MJ_km = MJ_total / max(km,
    1e-9)) so vde_mj_per_km reconciles with the canonical whole-cycle VDE.
    """
    t = np.asarray(profile.time_s, dtype=float)
    v = np.asarray(profile.speed_mps, dtype=float)

    s_m = np.trapezoid(v, t)
    if s_m <= 0:
        raise ValueError("cycle distance must be positive")
    distance_km = s_m / 1000.0

    roadload_power = np.asarray(profile.authoritative_roadload_power_W, dtype=float)
    inertial_power = np.asarray(profile.inertial_power_W, dtype=float)
    tractive_power = np.asarray(profile.tractive_power_W, dtype=float)
    residual_power = np.asarray(profile.residual_roadload_force_N, dtype=float) * v

    roadload_energy_MJ = np.trapezoid(roadload_power, t) / 1e6
    residual_roadload_energy_MJ = np.trapezoid(residual_power, t) / 1e6

    known_rolling_energy_MJ = None
    if profile.known_rolling_force_N is not None:
        rolling_power = np.asarray(profile.known_rolling_force_N, dtype=float) * v
        known_rolling_energy_MJ = np.trapezoid(rolling_power, t) / 1e6

    known_aero_energy_MJ = None
    if profile.known_aero_force_N is not None:
        aero_power = np.asarray(profile.known_aero_force_N, dtype=float) * v
        known_aero_energy_MJ = np.trapezoid(aero_power, t) / 1e6

    positive_inertial_work_MJ = np.trapezoid(np.clip(inertial_power, 0.0, None), t) / 1e6
    positive_tractive_energy_MJ = np.trapezoid(np.clip(tractive_power, 0.0, None), t) / 1e6
    braking_energy_required_MJ = -np.trapezoid(np.minimum(tractive_power, 0.0), t) / 1e6

    vde_mj_per_km = positive_tractive_energy_MJ / max(distance_km, 1e-9)

    _rho, density_basis, density_warnings = resolve_air_density(request.ambient)

    warnings: list[str] = list(density_warnings)
    provenance: dict[str, str] = {}

    if profile.known_rolling_force_N is not None:
        provenance["rolling"] = "CALCULATED"
    else:
        provenance["rolling"] = "UNAVAILABLE"
        warnings.append("Known rolling contribution unavailable: rrc_n_per_kn or test_mass_kg missing.")

    if profile.known_aero_force_N is not None:
        provenance["aero"] = "CALCULATED"
    else:
        provenance["aero"] = "UNAVAILABLE"
        if request.cda_m2 is None:
            warnings.append("Known aero contribution unavailable: cda_m2 missing.")
        elif _rho is None:
            warnings.append("Known aero contribution unavailable: air density could not be resolved.")

    provenance["residual"] = "CALCULATED"
    if any(force < 0 for force in profile.residual_roadload_force_N):
        warnings.append(
            "Residual roadload is negative at one or more timesteps: known rolling + known aero "
            "exceeds the authoritative roadload for this basis (possible physical inconsistency "
            "or boundary mismatch). Preserved as-is, not clipped."
        )

    provenance["air_density"] = density_basis.value if density_basis is not None else "UNAVAILABLE"
    if request.ambient.temperature_basis is not None:
        provenance["temperature"] = request.ambient.temperature_basis.value
    if request.ambient.pressure_basis is not None:
        provenance["pressure"] = request.ambient.pressure_basis.value

    availability = {"roadload_energy", "positive_inertial_work", "positive_tractive_energy", "braking_energy_required", "vde", "distance"}
    if known_rolling_energy_MJ is not None:
        availability.add("known_rolling_energy")
    if known_aero_energy_MJ is not None:
        availability.add("known_aero_energy")

    return VehicleDemandSummary(
        roadload_basis=profile.roadload_basis,
        distance_km=distance_km,
        roadload_energy_MJ=roadload_energy_MJ,
        known_rolling_energy_MJ=known_rolling_energy_MJ,
        known_aero_energy_MJ=known_aero_energy_MJ,
        residual_roadload_energy_MJ=residual_roadload_energy_MJ,
        positive_inertial_work_MJ=positive_inertial_work_MJ,
        positive_tractive_energy_MJ=positive_tractive_energy_MJ,
        braking_energy_required_MJ=braking_energy_required_MJ,
        vde_mj_per_km=vde_mj_per_km,
        availability=frozenset(availability),
        warnings=tuple(warnings),
        provenance=provenance,
        cycle_name=request.cycle_name,
        cycle_source=request.cycle_source,
        model_version=request.model_version,
    )


def calculate_vehicle_demand(request: VehicleDemandRequest, cycle_frame: pd.DataFrame) -> VehicleDemandResult:
    """Single-call TOTAL(+NET) service. TOTAL is required by the frozen 9A
    contract (VehicleDemandRequest.roadload_total is non-optional), so a
    missing TOTAL profile here means test_mass_kg was unavailable and is a
    hard error, not a silent omission. NET stays optional with no fallback:
    a request with no roadload_net simply produces net_summary=None.
    """
    total_profile = build_vehicle_demand_profile(request, cycle_frame, RoadloadBasis.TOTAL)
    if total_profile is None:
        raise ValueError(
            "Cannot compute TOTAL Vehicle Demand: request.test_mass_kg is required and was not provided."
        )
    total_summary = summarize_vehicle_demand(total_profile, request)

    net_summary = None
    if request.roadload_net is not None:
        net_profile = build_vehicle_demand_profile(request, cycle_frame, RoadloadBasis.NET)
        if net_profile is not None:
            net_summary = summarize_vehicle_demand(net_profile, request)

    return VehicleDemandResult(
        total_summary=total_summary,
        net_summary=net_summary,
        metadata={"engine_version": VEHICLE_DEMAND_ENGINE_VERSION},
    )


__all__ = [
    "VEHICLE_DEMAND_ENGINE_VERSION",
    "build_vehicle_demand_profile",
    "summarize_vehicle_demand",
    "calculate_vehicle_demand",
]
