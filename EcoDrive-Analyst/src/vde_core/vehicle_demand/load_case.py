"""Temporary Roadload & Load Case Lab orchestration.

This module is deliberately repository- and Streamlit-free.  It adapts one
already-fetched persisted VDE row into an in-memory Vehicle Demand request,
adds explicitly resolved external overlays, and delegates every cycle and
steady-state calculation to the canonical Vehicle Demand engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping

import pandas as pd

from src.vde_core.roadload.tire_model import G_MPS2, KPA_PER_PSI
from src.vde_core.rolling_rrc_prior import (
    RollingRrcPriorResolution,
    calculate_rolling_rrc_mass_delta,
    resolve_rolling_rrc_prior,
)
from src.vde_core.tire_roadload_service import preview_tire_roadload_from_row
from src.vde_core.vde_calculation_mass import resolve_vde_calculation_mass
from src.vde_core.vde_mass_proposal_resolver import resolve_mass_proposal

from .adapters import build_vehicle_demand_request, resolve_vehicle_demand_cycle
from .contracts import (
    AmbientState,
    EnergyMode,
    Provenance,
    RoadloadBasis,
    RoadloadCoefficients,
    VehicleDemandRequest,
    VehicleDemandProfile,
    VehicleDemandResult,
)
from .engine import build_vehicle_demand_profile, calculate_vehicle_demand
from .physics import resolve_air_density


LOAD_CASE_CONTRACT_VERSION = "0.5"
MAX_SWEEP_POINTS = 500


class _TextEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class LoadCaseAnalysisMode(_TextEnum):
    DRIVE_CYCLE = "DRIVE_CYCLE"
    FIXED_SPEED = "FIXED_SPEED"
    SPEED_SWEEP = "SPEED_SWEEP"


class LoadCaseMassMode(_TextEnum):
    INHERIT = "INHERIT"
    PAYLOAD_DELTA = "PAYLOAD_DELTA"
    TARGET_PAYLOAD = "TARGET_PAYLOAD"
    CUSTOM = "CUSTOM"
    GVWR = "GVWR"
    GCWR = "GCWR"


@dataclass(frozen=True)
class LoadCaseEnvironment:
    temperature_C: float | None = None
    pressure_kPa: float | None = None
    air_density_kg_m3: float | None = None
    tire_pressure_delta_kPa: float | None = None


@dataclass(frozen=True)
class LoadCaseMass:
    mode: LoadCaseMassMode = LoadCaseMassMode.INHERIT
    payload_delta_kg: float | None = None
    target_payload_kg: float | None = None
    custom_vehicle_mass_kg: float | None = None
    gvwr_kg: float | None = None
    gcwr_kg: float | None = None
    use_estimated_rrc_prior: bool = True


@dataclass(frozen=True)
class LoadCaseTrailer:
    enabled: bool = False
    mass_kg: float | None = None
    A_N: float | None = None
    B_N_per_kph: float | None = None
    C_N_per_kph2: float | None = None


@dataclass(frozen=True)
class LoadCaseRoute:
    grade_pct: float = 0.0


@dataclass(frozen=True)
class FixedSpeedSpec:
    speed_kph: float


@dataclass(frozen=True)
class SpeedSweepSpec:
    start_kph: float
    end_kph: float
    increment_kph: float
    max_points: int = MAX_SWEEP_POINTS


@dataclass(frozen=True)
class RoadloadLoadCaseRequest:
    baseline_vde_id: int | None
    analysis_mode: LoadCaseAnalysisMode = LoadCaseAnalysisMode.DRIVE_CYCLE
    environment: LoadCaseEnvironment = field(default_factory=LoadCaseEnvironment)
    mass: LoadCaseMass = field(default_factory=LoadCaseMass)
    trailer: LoadCaseTrailer = field(default_factory=LoadCaseTrailer)
    route: LoadCaseRoute = field(default_factory=LoadCaseRoute)
    fixed_speed: FixedSpeedSpec | None = None
    speed_sweep: SpeedSweepSpec | None = None
    contract_version: str = LOAD_CASE_CONTRACT_VERSION


@dataclass(frozen=True)
class ResolvedMassState:
    resolved_vehicle_mass_kg: float
    system_mass_kg: float
    mass_basis: str
    vehicle_mass_at_gcwr_kg: float | None = None
    trailer_mass_kg: float = 0.0


@dataclass(frozen=True)
class ResolvedTrailerState:
    enabled: bool
    mass_kg: float
    roadload: RoadloadCoefficients | None
    status: str


@dataclass(frozen=True)
class ResolvedRollingCorrection:
    rrc_value: float | None
    source: str
    rule_key: str | None
    status: str
    delta_abc: RoadloadCoefficients | None
    match: str | None = None
    confidence: str | None = None
    diagnostic: RollingRrcPriorResolution | None = None
    baseline_rolling_abc: RoadloadCoefficients | None = None
    scenario_rolling_abc: RoadloadCoefficients | None = None


@dataclass(frozen=True)
class LoadCaseComparison:
    roadload_basis: RoadloadBasis
    baseline_mj_per_km: float
    scenario_mj_per_km: float
    delta_mj_per_km: float
    delta_pct: float | None


@dataclass(frozen=True)
class SteadyStateBoundaryDemand:
    roadload_basis: RoadloadBasis
    baseline_roadload_force_N: float
    trailer_force_N: float
    rolling_force_N: float
    grade_force_N: float
    condition_force_N: float
    wheel_force_N: float
    wheel_power_kW: float
    wheel_energy_intensity_MJ_per_km: float
    energy_mode: EnergyMode


@dataclass(frozen=True)
class SteadyStateDemandPoint:
    speed_kph: float
    total: SteadyStateBoundaryDemand
    net: SteadyStateBoundaryDemand | None = None


@dataclass(frozen=True)
class ForceCurvePoint:
    speed_kph: float
    baseline_force_N: float
    trailer_force_N: float
    rolling_force_N: float
    grade_force_N: float
    condition_force_N: float
    scenario_force_N: float
    selected_operating_point: bool = False


@dataclass(frozen=True)
class RoadloadLoadCaseResult:
    request: RoadloadLoadCaseRequest
    baseline_request: VehicleDemandRequest
    scenario_request: VehicleDemandRequest
    scenario_engine_request: VehicleDemandRequest
    mass_state: ResolvedMassState
    trailer_state: ResolvedTrailerState
    rolling_state: ResolvedRollingCorrection
    baseline_result: VehicleDemandResult | None
    scenario_result: VehicleDemandResult | None
    baseline_total_profile: VehicleDemandProfile | None
    scenario_total_profile: VehicleDemandProfile | None
    comparisons: tuple[LoadCaseComparison, ...]
    steady_state_points: tuple[SteadyStateDemandPoint, ...]
    force_curve_points: tuple[ForceCurvePoint, ...]
    baseline_effective_mass_kg: float
    baseline_calculation_mass_kg: float
    baseline_mass_basis: str
    baseline_mass_source_field: str
    scenario_vehicle_mass_kg: float
    scenario_calculation_mass_kg: float
    scenario_mass_basis: str
    scenario_system_mass_kg: float
    rolling_rrc_value: float | None
    rolling_rrc_source: str
    rolling_rrc_rule_key: str | None
    rolling_correction_status: str
    rolling_delta_abc: RoadloadCoefficients | None
    grade_force_N: float
    condition_delta_C_N_per_kph2: float
    tire_delta_roadload: RoadloadCoefficients | None
    baseline_air_density_kg_m3: float | None
    scenario_air_density_kg_m3: float | None
    provenance: Mapping[str, str]
    explanatory_notes: tuple[str, ...]
    warnings: tuple[str, ...]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero, got {result!r}.")
    return result


def _row_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return _finite(value, key)
    return None


def baseline_ambient_from_vde_row(vde_row: Mapping[str, Any]) -> AmbientState:
    """Read canonical ambient fields when the runtime row actually has them."""
    temperature = _row_float(vde_row, "roadload_temperature_c")
    pressure = _row_float(vde_row, "roadload_ambient_pressure_kpa")
    density = _row_float(vde_row, "roadload_air_density_kg_m3", "air_density_kg_m3")
    return AmbientState(
        temperature_C=temperature,
        pressure_kPa=pressure,
        air_density_kg_m3=density,
        temperature_basis=Provenance.SOURCE if temperature is not None else None,
        pressure_basis=Provenance.SOURCE if pressure is not None else None,
        density_basis=Provenance.SOURCE if density is not None else None,
    )


def resolve_scenario_ambient(baseline: AmbientState, override: LoadCaseEnvironment) -> AmbientState:
    explicit_density = override.air_density_kg_m3
    if explicit_density is not None:
        return AmbientState(
            temperature_C=override.temperature_C if override.temperature_C is not None else baseline.temperature_C,
            pressure_kPa=override.pressure_kPa if override.pressure_kPa is not None else baseline.pressure_kPa,
            air_density_kg_m3=_positive(explicit_density, "environment.air_density_kg_m3"),
            temperature_basis=Provenance.ASSUMED if override.temperature_C is not None else baseline.temperature_basis,
            pressure_basis=Provenance.ASSUMED if override.pressure_kPa is not None else baseline.pressure_basis,
            density_basis=Provenance.ASSUMED,
        )
    return AmbientState(
        temperature_C=(
            _finite(override.temperature_C, "environment.temperature_C")
            if override.temperature_C is not None
            else baseline.temperature_C
        ),
        pressure_kPa=(
            _positive(override.pressure_kPa, "environment.pressure_kPa")
            if override.pressure_kPa is not None
            else baseline.pressure_kPa
        ),
        temperature_basis=Provenance.ASSUMED if override.temperature_C is not None else baseline.temperature_basis,
        pressure_basis=Provenance.ASSUMED if override.pressure_kPa is not None else baseline.pressure_basis,
    )


def grade_force_N(system_mass_kg: float, grade_pct: float) -> float:
    mass = _positive(system_mass_kg, "system_mass_kg")
    grade = _finite(grade_pct, "grade_pct")
    if grade == 0.0:
        return 0.0
    theta = math.atan(grade / 100.0)
    return mass * G_MPS2 * math.sin(theta)


def resolve_baseline_mass(vde_row: Mapping[str, Any]) -> tuple[float, str, str]:
    """Expose the shared persisted-VDE calculation mass and provenance."""
    resolved = resolve_vde_calculation_mass(vde_row)
    if resolved.mass_kg is None:
        raise ValueError("Baseline VDE has no resolvable vehicle/test mass.")
    return (
        _positive(resolved.mass_kg, "baseline_calculation_mass_kg"),
        resolved.basis or "UNRESOLVED",
        resolved.source_field or "UNAVAILABLE",
    )


def resolve_mass_state(
    vde_row: Mapping[str, Any],
    baseline: VehicleDemandRequest,
    spec: LoadCaseMass,
    trailer: LoadCaseTrailer,
) -> ResolvedMassState:
    baseline_vehicle = _positive(baseline.test_mass_kg, "baseline.test_mass_kg")
    trailer_mass = 0.0
    if trailer.enabled:
        if trailer.mass_kg is None:
            raise ValueError("trailer.mass_kg is required when trailer is enabled.")
        trailer_mass = _positive(trailer.mass_kg, "trailer.mass_kg")

    source = dict(vde_row)
    mode = spec.mode
    proposal = None
    if mode is LoadCaseMassMode.INHERIT:
        vehicle_mass = baseline_vehicle
        mass_basis = "INHERITED_BASELINE"
    elif mode is LoadCaseMassMode.PAYLOAD_DELTA:
        if spec.payload_delta_kg is None:
            raise ValueError("mass.payload_delta_kg is required for PAYLOAD_DELTA.")
        target = baseline_vehicle + _finite(spec.payload_delta_kg, "mass.payload_delta_kg")
        proposal = resolve_mass_proposal(source, "CUSTOM_MASS", {"test_mass_kg": target, "test_mass_basis": "PAYLOAD_DELTA"})
        vehicle_mass = target
        mass_basis = "PAYLOAD_DELTA"
    elif mode is LoadCaseMassMode.TARGET_PAYLOAD:
        if spec.target_payload_kg is None:
            raise ValueError("mass.target_payload_kg is required for TARGET_PAYLOAD.")
        payload = _finite(spec.target_payload_kg, "mass.target_payload_kg")
        proposal = resolve_mass_proposal(source, "GVWR", {"payload_kg": payload})
        vehicle_mass = proposal["resolved_snapshot"].get("vde_calculation_mass_kg")
        mass_basis = "TARGET_PAYLOAD"
    elif mode is LoadCaseMassMode.CUSTOM:
        if spec.custom_vehicle_mass_kg is None:
            raise ValueError("mass.custom_vehicle_mass_kg is required for CUSTOM.")
        target = _positive(spec.custom_vehicle_mass_kg, "mass.custom_vehicle_mass_kg")
        proposal = resolve_mass_proposal(source, "CUSTOM_MASS", {"test_mass_kg": target, "test_mass_basis": "CUSTOM"})
        vehicle_mass = target
        mass_basis = "CUSTOM"
    elif mode is LoadCaseMassMode.GVWR:
        if spec.gvwr_kg is None:
            raise ValueError("mass.gvwr_kg is required for GVWR.")
        proposal = resolve_mass_proposal(source, "GVWR", {"gvwr_kg": _positive(spec.gvwr_kg, "mass.gvwr_kg")})
        vehicle_mass = proposal["resolved_snapshot"].get("vde_calculation_mass_kg")
        mass_basis = "GVWR"
    elif mode is LoadCaseMassMode.GCWR:
        gcwr = spec.gcwr_kg
        if gcwr is None:
            raise ValueError("mass.gcwr_kg is required for GCWR.")
        if not trailer.enabled:
            raise ValueError("GCWR mass mode requires an enabled trailer.")
        proposal = resolve_mass_proposal(
            source,
            "GCWR",
            {
                "gcwr_kg": _positive(gcwr, "mass.gcwr_kg"),
                "trailer_mass_kg": trailer_mass,
                "trailer_A": trailer.A_N,
                "trailer_B": trailer.B_N_per_kph,
                "trailer_C": trailer.C_N_per_kph2,
            },
        )
        snapshot = proposal["resolved_snapshot"]
        vehicle_mass = snapshot.get("vehicle_mass_at_gcwr")
        issues = [item for item in proposal.get("issues", ()) if item.get("severity") == "invalid"]
        if issues:
            raise ValueError(issues[0].get("message") or "Invalid GCWR load case.")
        if vehicle_mass is None:
            raise ValueError("GCWR load case could not resolve vehicle mass.")
        return ResolvedMassState(
            resolved_vehicle_mass_kg=_positive(vehicle_mass, "vehicle_mass_at_gcwr"),
            system_mass_kg=_positive(gcwr, "mass.gcwr_kg"),
            mass_basis="COMBINED_GCWR",
            vehicle_mass_at_gcwr_kg=float(vehicle_mass),
            trailer_mass_kg=trailer_mass,
        )
    else:  # pragma: no cover - Enum keeps this defensive branch unreachable.
        raise ValueError(f"Unsupported mass mode: {mode!r}.")

    if proposal is not None:
        invalid = [item for item in proposal.get("issues", ()) if item.get("severity") in {"invalid", "missing"}]
        if invalid:
            raise ValueError(invalid[0].get("message") or "Mass state could not be resolved.")
    vehicle = _positive(vehicle_mass, "resolved_vehicle_mass_kg")
    return ResolvedMassState(
        resolved_vehicle_mass_kg=vehicle,
        system_mass_kg=vehicle + trailer_mass,
        mass_basis=mass_basis,
        trailer_mass_kg=trailer_mass,
    )


def resolve_trailer_state(spec: LoadCaseTrailer) -> ResolvedTrailerState:
    if not spec.enabled:
        return ResolvedTrailerState(False, 0.0, None, "NOT_USED")
    if spec.mass_kg is None:
        raise ValueError("trailer.mass_kg is required when trailer is enabled.")
    mass = _positive(spec.mass_kg, "trailer.mass_kg")
    values = (spec.A_N, spec.B_N_per_kph, spec.C_N_per_kph2)
    if all(value is None for value in values):
        return ResolvedTrailerState(True, mass, None, "UNAVAILABLE")
    if any(value is None for value in values):
        return ResolvedTrailerState(True, mass, None, "REVIEW")
    coefficients = RoadloadCoefficients(
        A_N=_finite(spec.A_N, "trailer.A_N"),
        B_N_per_kph=_finite(spec.B_N_per_kph, "trailer.B_N_per_kph"),
        C_N_per_kph2=_finite(spec.C_N_per_kph2, "trailer.C_N_per_kph2"),
    )
    return ResolvedTrailerState(True, mass, coefficients, "RESOLVED")


def _constant_rrc_mass_states(
    rrc_n_per_kn: float,
    baseline_vehicle_mass_kg: float,
    scenario_vehicle_mass_kg: float,
    crr1_frac_at_120kph: float = 0.0,
) -> tuple[RoadloadCoefficients, RoadloadCoefficients, RoadloadCoefficients]:
    delta = calculate_rolling_rrc_mass_delta(
        rrc_n_per_kn,
        baseline_vehicle_mass_kg,
        scenario_vehicle_mass_kg,
        crr1_frac_at_120kph,
    )
    baseline = RoadloadCoefficients(
        A_N=delta.baseline_A_N,
        B_N_per_kph=delta.baseline_B_N_per_kph,
        C_N_per_kph2=0.0,
    )
    scenario = RoadloadCoefficients(
        A_N=delta.scenario_A_N,
        B_N_per_kph=delta.scenario_B_N_per_kph,
        C_N_per_kph2=0.0,
    )
    delta_abc = RoadloadCoefficients(A_N=delta.A_N, B_N_per_kph=delta.B_N_per_kph, C_N_per_kph2=0.0)
    return baseline, scenario, delta_abc


def _tire_owner_mass_delta(
    vde_row: Mapping[str, Any],
    baseline_vehicle_mass_kg: float,
    scenario_vehicle_mass_kg: float,
) -> ResolvedRollingCorrection | None:
    front_id = vde_row.get("front_tire_id")
    rear_id = vde_row.get("rear_tire_id")
    if front_id in (None, "") or rear_id in (None, ""):
        return None
    common = {"tire_load_mass_basis": "TEST_MASS"}
    try:
        baseline = preview_tire_roadload_from_row(
            dict(vde_row),
            {**common, "test_mass_kg": baseline_vehicle_mass_kg},
        )
        scenario = preview_tire_roadload_from_row(
            dict(vde_row),
            {**common, "test_mass_kg": scenario_vehicle_mass_kg},
        )
    except (KeyError, TypeError, ValueError):
        return None
    baseline_abc = (baseline.get("calculation") or {}).get("total_final_abc") or {}
    scenario_abc = (scenario.get("calculation") or {}).get("total_final_abc") or {}
    if any(abc.get(key) is None for abc in (baseline_abc, scenario_abc) for key in ("A", "B", "C")):
        return None
    baseline_coefficients = RoadloadCoefficients(
        A_N=_finite(baseline_abc["A"], "baseline_tire.A"),
        B_N_per_kph=_finite(baseline_abc["B"], "baseline_tire.B"),
        C_N_per_kph2=_finite(baseline_abc["C"], "baseline_tire.C"),
    )
    scenario_coefficients = RoadloadCoefficients(
        A_N=_finite(scenario_abc["A"], "scenario_tire.A"),
        B_N_per_kph=_finite(scenario_abc["B"], "scenario_tire.B"),
        C_N_per_kph2=_finite(scenario_abc["C"], "scenario_tire.C"),
    )
    delta = RoadloadCoefficients(
        A_N=scenario_coefficients.A_N - baseline_coefficients.A_N,
        B_N_per_kph=scenario_coefficients.B_N_per_kph - baseline_coefficients.B_N_per_kph,
        C_N_per_kph2=scenario_coefficients.C_N_per_kph2 - baseline_coefficients.C_N_per_kph2,
    )
    rrc_value = (scenario.get("save_payload") or {}).get("rrc_N_per_kN")
    return ResolvedRollingCorrection(
        rrc_value=None if rrc_value is None else _finite(rrc_value, "scenario_tire.rrc_N_per_kN"),
        source="TIRE_DB",
        rule_key=f"front_tire_id={front_id}|rear_tire_id={rear_id}",
        status="APPLIED",
        delta_abc=delta,
        match="Applicable Tire DB resolution",
        confidence="Tire-specific calculation",
        baseline_rolling_abc=baseline_coefficients,
        scenario_rolling_abc=scenario_coefficients,
    )


def resolve_load_dependent_rolling_delta(
    vde_row: Mapping[str, Any],
    baseline_vehicle_mass_kg: float,
    scenario_vehicle_mass_kg: float,
    *,
    use_estimated_prior: bool,
    linked_prior_context: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
) -> ResolvedRollingCorrection:
    """Resolve one temporary load-dependent rolling delta by source priority."""
    specific_rrc = _row_float(vde_row, "rrc_N_per_kN")
    if specific_rrc is not None:
        baseline_rolling, scenario_rolling, delta = _constant_rrc_mass_states(
            specific_rrc,
            baseline_vehicle_mass_kg,
            scenario_vehicle_mass_kg,
            _row_float(vde_row, "crr1_frac_at_120kph") or 0.0,
        )
        return ResolvedRollingCorrection(
            rrc_value=specific_rrc,
            source="VDE_SPECIFIC",
            rule_key="rrc_N_per_kN",
            status="APPLIED",
            delta_abc=delta,
            match="Persisted VDE RRC",
            confidence="Vehicle-specific",
            baseline_rolling_abc=baseline_rolling,
            scenario_rolling_abc=scenario_rolling,
        )

    tire_delta = _tire_owner_mass_delta(vde_row, baseline_vehicle_mass_kg, scenario_vehicle_mass_kg)
    if tire_delta is not None:
        return tire_delta

    prior = resolve_rolling_rrc_prior(vde_row, linked_context=linked_prior_context)
    if not use_estimated_prior:
        return ResolvedRollingCorrection(None, "NONE", None, "DISABLED", None, diagnostic=prior)
    if prior.status != "APPLIED" or prior.rrc_n_per_kn is None:
        return ResolvedRollingCorrection(None, "NONE", None, "UNAVAILABLE", None, diagnostic=prior)
    baseline_rolling, scenario_rolling, delta = _constant_rrc_mass_states(
        prior.rrc_n_per_kn,
        baseline_vehicle_mass_kg,
        scenario_vehicle_mass_kg,
        prior.crr1_frac_at_120kph or 0.0,
    )
    return ResolvedRollingCorrection(
        rrc_value=prior.rrc_n_per_kn,
        source=prior.source,
        rule_key=prior.matched_rule_key,
        status="APPLIED",
        delta_abc=delta,
        match=(
            f"{prior.lookup_category} / {prior.resolved_electrification} / "
            f"{prior.resolved_transmission_key}"
        ),
        confidence="Estimated prior",
        diagnostic=prior,
        baseline_rolling_abc=baseline_rolling,
        scenario_rolling_abc=scenario_rolling,
    )


def _condition_delta_C(
    baseline_request: VehicleDemandRequest,
    scenario_ambient: AmbientState,
    environment: LoadCaseEnvironment,
) -> tuple[float, tuple[str, ...], str]:
    has_environment_override = any(
        value is not None
        for value in (environment.temperature_C, environment.pressure_kPa, environment.air_density_kg_m3)
    )
    if not has_environment_override:
        return 0.0, (), "NOT_REQUESTED"
    baseline_rho, _baseline_basis, baseline_warnings = resolve_air_density(baseline_request.ambient)
    scenario_rho, _scenario_basis, scenario_warnings = resolve_air_density(scenario_ambient)
    warnings = list(baseline_warnings) + list(scenario_warnings)
    if baseline_request.cda_m2 is None:
        warnings.append("Ambient aero correction unavailable: baseline CdA is missing.")
        return 0.0, tuple(warnings), Provenance.UNAVAILABLE.value
    if baseline_rho is None:
        warnings.append("Ambient aero correction unavailable: baseline air density is unresolved.")
        return 0.0, tuple(warnings), Provenance.UNAVAILABLE.value
    if scenario_rho is None:
        warnings.append("Ambient aero correction unavailable: scenario air density is unresolved.")
        return 0.0, tuple(warnings), Provenance.UNAVAILABLE.value
    cda = _finite(baseline_request.cda_m2, "baseline.cda_m2")
    delta_c = 0.5 * cda * (scenario_rho - baseline_rho) / (3.6**2)
    return delta_c, tuple(warnings), Provenance.CALCULATED.value


def resolve_tire_pressure_delta(
    vde_row: Mapping[str, Any],
    pressure_delta_kPa: float | None,
    resolved_vehicle_mass_kg: float,
) -> tuple[RoadloadCoefficients | None, tuple[str, ...], str]:
    """Resolve a pressure delta through the existing Tire Roadload owner.

    Missing tire identity, saved pressure, or model support is an explicit
    unavailable result.  No generic pressure multiplier is ever applied.
    """
    if pressure_delta_kPa is None:
        return None, (), "NOT_REQUESTED"
    delta = _finite(pressure_delta_kPa, "environment.tire_pressure_delta_kPa")
    front_psi = _row_float(vde_row, "front_pressure_psi")
    rear_psi = _row_float(vde_row, "rear_pressure_psi")
    if front_psi is None or rear_psi is None:
        return None, ("Tire-pressure correction unavailable: saved front/rear pressures are missing.",), Provenance.UNAVAILABLE.value
    front_kpa = front_psi * KPA_PER_PSI + delta
    rear_kpa = rear_psi * KPA_PER_PSI + delta
    if front_kpa <= 0 or rear_kpa <= 0:
        raise ValueError("Tire-pressure delta produces a non-positive front or rear pressure.")
    try:
        common_mass = {
            "test_mass_kg": resolved_vehicle_mass_kg,
            "tire_load_mass_basis": "TEST_MASS",
        }
        baseline = preview_tire_roadload_from_row(dict(vde_row), common_mass)
        scenario = preview_tire_roadload_from_row(
            dict(vde_row),
            {
                **common_mass,
                "front_pressure_kpa": front_kpa,
                "rear_pressure_kpa": rear_kpa,
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, (f"Tire-pressure correction unavailable: {exc}",), Provenance.UNAVAILABLE.value
    baseline_abc = (baseline.get("calculation") or {}).get("total_final_abc") or {}
    scenario_abc = (scenario.get("calculation") or {}).get("total_final_abc") or {}
    if any(abc.get(key) is None for abc in (baseline_abc, scenario_abc) for key in ("A", "B", "C")):
        return None, ("Tire-pressure correction unavailable: Tire Roadload owner returned incomplete ABC.",), Provenance.UNAVAILABLE.value
    delta_abc = RoadloadCoefficients(
        A_N=_finite(scenario_abc["A"], "scenario_tire.A") - _finite(baseline_abc["A"], "baseline_tire.A"),
        B_N_per_kph=_finite(scenario_abc["B"], "scenario_tire.B") - _finite(baseline_abc["B"], "baseline_tire.B"),
        C_N_per_kph2=_finite(scenario_abc["C"], "scenario_tire.C") - _finite(baseline_abc["C"], "baseline_tire.C"),
    )
    return delta_abc, (), "APPLIED"


def _add_coefficients(
    baseline: RoadloadCoefficients | None,
    rolling_delta: RoadloadCoefficients | None,
    trailer: RoadloadCoefficients | None,
    grade_N: float,
    condition_delta_C: float,
    tire_delta: RoadloadCoefficients | None,
) -> RoadloadCoefficients | None:
    if baseline is None:
        return None
    return RoadloadCoefficients(
        A_N=float(baseline.A_N) + (float(rolling_delta.A_N) if rolling_delta else 0.0) + (float(trailer.A_N) if trailer else 0.0) + grade_N + (float(tire_delta.A_N) if tire_delta else 0.0),
        B_N_per_kph=float(baseline.B_N_per_kph) + (float(rolling_delta.B_N_per_kph) if rolling_delta else 0.0) + (float(trailer.B_N_per_kph) if trailer else 0.0) + (float(tire_delta.B_N_per_kph) if tire_delta else 0.0),
        C_N_per_kph2=float(baseline.C_N_per_kph2) + (float(rolling_delta.C_N_per_kph2) if rolling_delta else 0.0) + (float(trailer.C_N_per_kph2) if trailer else 0.0) + condition_delta_C + (float(tire_delta.C_N_per_kph2) if tire_delta else 0.0),
    )


def _comparison(
    basis: RoadloadBasis,
    baseline_value: float | None,
    scenario_value: float | None,
) -> LoadCaseComparison | None:
    if baseline_value is None or scenario_value is None:
        return None
    delta = scenario_value - baseline_value
    pct = None if baseline_value == 0 else delta / baseline_value * 100.0
    return LoadCaseComparison(basis, baseline_value, scenario_value, delta, pct)


def build_constant_speed_trace(speed_kph: float) -> pd.DataFrame:
    speed = _positive(speed_kph, "speed_kph") / 3.6
    return pd.DataFrame({"t": [0.0, 1.0], "v": [speed, speed]})


def speed_sweep_values(spec: SpeedSweepSpec) -> tuple[float, ...]:
    start = _positive(spec.start_kph, "speed_sweep.start_kph")
    end = _positive(spec.end_kph, "speed_sweep.end_kph")
    step = _positive(spec.increment_kph, "speed_sweep.increment_kph")
    if end < start:
        raise ValueError("speed_sweep.end_kph must be greater than or equal to start_kph.")
    max_points = int(spec.max_points)
    if max_points <= 0 or max_points > MAX_SWEEP_POINTS:
        raise ValueError(f"speed_sweep.max_points must be between 1 and {MAX_SWEEP_POINTS}.")
    count = int(math.floor((end - start) / step + 1e-12)) + 1
    if count > max_points:
        raise ValueError(f"Speed sweep contains {count} points; limit is {max_points}.")
    values = [start + index * step for index in range(count)]
    if values[-1] < end and math.isclose(values[-1] + step, end, rel_tol=0.0, abs_tol=1e-9):
        values.append(end)
    return tuple(values)


def _force_at(coefficients: RoadloadCoefficients, speed_kph: float) -> float:
    return (
        float(coefficients.A_N)
        + float(coefficients.B_N_per_kph) * speed_kph
        + float(coefficients.C_N_per_kph2) * speed_kph**2
    )


def _steady_boundary(
    baseline_request: VehicleDemandRequest,
    scenario_request: VehicleDemandRequest,
    basis: RoadloadBasis,
    speed_kph: float,
    rolling_delta: RoadloadCoefficients | None,
    trailer: ResolvedTrailerState,
    grade_N: float,
    condition_delta_C: float,
    tire_delta: RoadloadCoefficients | None,
) -> SteadyStateBoundaryDemand | None:
    baseline_coeffs = baseline_request.roadload_total if basis is RoadloadBasis.TOTAL else baseline_request.roadload_net
    scenario_coeffs = scenario_request.roadload_total if basis is RoadloadBasis.TOTAL else scenario_request.roadload_net
    if baseline_coeffs is None or scenario_coeffs is None:
        return None
    trace = build_constant_speed_trace(speed_kph)
    profile = build_vehicle_demand_profile(scenario_request, trace, basis)
    if profile is None:
        return None
    wheel_force = float(profile.tractive_force_N[-1])
    speed_mps = speed_kph / 3.6
    rolling_force = _force_at(rolling_delta, speed_kph) if rolling_delta else 0.0
    trailer_force = _force_at(trailer.roadload, speed_kph) if trailer.roadload else 0.0
    condition_force = condition_delta_C * speed_kph**2 + (_force_at(tire_delta, speed_kph) if tire_delta else 0.0)
    return SteadyStateBoundaryDemand(
        roadload_basis=basis,
        baseline_roadload_force_N=_force_at(baseline_coeffs, speed_kph),
        trailer_force_N=trailer_force,
        rolling_force_N=rolling_force,
        grade_force_N=grade_N,
        condition_force_N=condition_force,
        wheel_force_N=wheel_force,
        wheel_power_kW=wheel_force * speed_mps / 1000.0,
        wheel_energy_intensity_MJ_per_km=wheel_force / 1000.0,
        energy_mode=profile.energy_mode[-1],
    )


def _steady_point(
    baseline_request: VehicleDemandRequest,
    scenario_request: VehicleDemandRequest,
    speed_kph: float,
    rolling_delta: RoadloadCoefficients | None,
    trailer: ResolvedTrailerState,
    grade_N: float,
    condition_delta_C: float,
    tire_delta: RoadloadCoefficients | None,
) -> SteadyStateDemandPoint:
    total = _steady_boundary(
        baseline_request, scenario_request, RoadloadBasis.TOTAL, speed_kph, rolling_delta, trailer, grade_N, condition_delta_C, tire_delta
    )
    if total is None:  # TOTAL is mandatory in VehicleDemandRequest.
        raise ValueError("TOTAL steady-state demand could not be calculated.")
    net = _steady_boundary(
        baseline_request, scenario_request, RoadloadBasis.NET, speed_kph, rolling_delta, trailer, grade_N, condition_delta_C, tire_delta
    )
    return SteadyStateDemandPoint(speed_kph=speed_kph, total=total, net=net)


def _contextual_curve_speeds(request: RoadloadLoadCaseRequest) -> tuple[float, ...]:
    if request.analysis_mode is LoadCaseAnalysisMode.SPEED_SWEEP:
        if request.speed_sweep is None:
            raise ValueError("speed_sweep is required for SPEED_SWEEP mode.")
        return speed_sweep_values(request.speed_sweep)
    if request.analysis_mode is LoadCaseAnalysisMode.FIXED_SPEED:
        if request.fixed_speed is None:
            raise ValueError("fixed_speed is required for FIXED_SPEED mode.")
        selected = _positive(request.fixed_speed.speed_kph, "fixed_speed.speed_kph")
        upper = max(130.0, math.ceil(selected * 1.25 / 10.0) * 10.0)
        speeds = [float(value) for value in range(0, int(upper) + 1, 5)]
        if not any(math.isclose(value, selected, rel_tol=0.0, abs_tol=1e-9) for value in speeds):
            speeds.append(selected)
            speeds.sort()
        return tuple(speeds)
    return tuple(float(value) for value in range(0, 131, 5))


def _build_force_curve(
    request: RoadloadLoadCaseRequest,
    baseline_request: VehicleDemandRequest,
    scenario_request: VehicleDemandRequest,
    rolling_delta: RoadloadCoefficients | None,
    trailer: ResolvedTrailerState,
    grade_N: float,
    condition_delta_C: float,
    tire_delta: RoadloadCoefficients | None,
) -> tuple[ForceCurvePoint, ...]:
    selected_speed = request.fixed_speed.speed_kph if request.fixed_speed is not None else None
    points = []
    for speed in _contextual_curve_speeds(request):
        baseline_force = _force_at(baseline_request.roadload_total, speed)
        rolling_force = _force_at(rolling_delta, speed) if rolling_delta else 0.0
        trailer_force = _force_at(trailer.roadload, speed) if trailer.roadload else 0.0
        condition_force = condition_delta_C * speed**2 + (_force_at(tire_delta, speed) if tire_delta else 0.0)
        scenario_force = baseline_force + rolling_force + trailer_force + grade_N + condition_force
        effective_force = _force_at(scenario_request.roadload_total, speed)
        if not math.isclose(scenario_force, effective_force, rel_tol=1e-12, abs_tol=1e-9):
            raise RuntimeError("Temporary force-curve contributions do not close against the effective TOTAL boundary.")
        points.append(
            ForceCurvePoint(
                speed_kph=speed,
                baseline_force_N=baseline_force,
                trailer_force_N=trailer_force,
                rolling_force_N=rolling_force,
                grade_force_N=grade_N,
                condition_force_N=condition_force,
                scenario_force_N=scenario_force,
                selected_operating_point=(
                    selected_speed is not None
                    and math.isclose(speed, selected_speed, rel_tol=0.0, abs_tol=1e-9)
                ),
            )
        )
    return tuple(points)


def calculate_roadload_load_case(
    vde_row: Mapping[str, Any],
    request: RoadloadLoadCaseRequest,
    *,
    cycle_frame: pd.DataFrame | None = None,
    rolling_prior_context: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
) -> RoadloadLoadCaseResult:
    """Resolve and calculate one temporary load case without persistence."""
    row_id = vde_row.get("id")
    if request.baseline_vde_id is not None and row_id is not None and int(request.baseline_vde_id) != int(row_id):
        raise ValueError("request.baseline_vde_id does not match the supplied VDE row.")

    baseline_ambient = baseline_ambient_from_vde_row(vde_row)
    baseline_effective_mass, baseline_mass_basis, baseline_mass_source_field = resolve_baseline_mass(vde_row)
    baseline_request = build_vehicle_demand_request(vde_row, ambient=baseline_ambient)
    scenario_ambient = resolve_scenario_ambient(baseline_ambient, request.environment)
    mass_state = resolve_mass_state(vde_row, baseline_request, request.mass, request.trailer)
    trailer_state = resolve_trailer_state(request.trailer)
    rolling_state = resolve_load_dependent_rolling_delta(
        vde_row,
        baseline_effective_mass,
        mass_state.resolved_vehicle_mass_kg,
        use_estimated_prior=request.mass.use_estimated_rrc_prior,
        linked_prior_context=rolling_prior_context,
    )
    grade_N = grade_force_N(mass_state.system_mass_kg, request.route.grade_pct)
    condition_delta_C, ambient_warnings, ambient_status = _condition_delta_C(
        baseline_request, scenario_ambient, request.environment
    )
    baseline_density, _baseline_density_basis, _baseline_density_warnings = resolve_air_density(baseline_ambient)
    scenario_density, _scenario_density_basis, _scenario_density_warnings = resolve_air_density(scenario_ambient)

    tire_delta, tire_warnings, tire_status = resolve_tire_pressure_delta(
        vde_row,
        request.environment.tire_pressure_delta_kPa,
        mass_state.resolved_vehicle_mass_kg,
    )
    warnings = list(ambient_warnings) + list(tire_warnings)
    explanatory_notes: list[str] = []
    mass_changed_for_rolling = not math.isclose(
        baseline_effective_mass,
        mass_state.resolved_vehicle_mass_kg,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    if rolling_state.source == "RULE_ESTIMATED":
        warnings.append(
            "Vehicle-specific RRC was unavailable. A category/electrification/transmission prior is being used "
            "for temporary sensitivity analysis. The persisted VDE is not modified."
        )
    elif rolling_state.status == "UNAVAILABLE" and mass_changed_for_rolling:
        warnings.append(
            "Load-dependent rolling correction unavailable. No vehicle-specific RRC, applicable Tire DB value, "
            "or sufficiently specific RULE_ESTIMATED prior was found. Static authoritative roadload remains unchanged."
        )
    if trailer_state.enabled and trailer_state.roadload is None:
        warnings.append(
            "Trailer roadload unavailable: complete trailer A/B/C is required; no generic trailer force was invented."
        )
    scenario_request = replace(
        baseline_request,
        test_mass_kg=mass_state.system_mass_kg,
        roadload_total=_add_coefficients(
            baseline_request.roadload_total, rolling_state.delta_abc, trailer_state.roadload, 0.0, condition_delta_C, tire_delta
        ),
        roadload_net=_add_coefficients(
            baseline_request.roadload_net, rolling_state.delta_abc, trailer_state.roadload, 0.0, condition_delta_C, tire_delta
        ),
        ambient=scenario_ambient,
        rrc_n_per_kn=None if trailer_state.enabled else baseline_request.rrc_n_per_kn,
        provenance={
            **dict(baseline_request.provenance),
            "mass": Provenance.RESOLVED.value,
            "rolling_correction": rolling_state.status,
            "rolling_rrc_source": rolling_state.source,
            "trailer": Provenance.CALCULATED.value if trailer_state.roadload else Provenance.UNAVAILABLE.value,
            "grade": Provenance.CALCULATED.value,
            "ambient_aero_delta": ambient_status,
            "tire_pressure_delta": tire_status,
            "baseline_air_density": (
                _baseline_density_basis.value if _baseline_density_basis is not None else Provenance.UNAVAILABLE.value
            ),
            "scenario_air_density": (
                _scenario_density_basis.value if _scenario_density_basis is not None else Provenance.UNAVAILABLE.value
            ),
        },
    )
    # Grade is an external force, not a mutation of the temporary scenario
    # ABC contract. The canonical engine still receives an equivalent
    # effective boundary for profile integration.
    scenario_engine_request = replace(
        scenario_request,
        roadload_total=_add_coefficients(scenario_request.roadload_total, None, None, grade_N, 0.0, None),
        roadload_net=_add_coefficients(scenario_request.roadload_net, None, None, grade_N, 0.0, None),
    )

    baseline_result = None
    scenario_result = None
    baseline_total_profile = None
    scenario_total_profile = None
    comparisons: list[LoadCaseComparison] = []
    points: tuple[SteadyStateDemandPoint, ...] = ()
    force_curve = _build_force_curve(
        request,
        baseline_request,
        scenario_engine_request,
        rolling_state.delta_abc,
        trailer_state,
        grade_N,
        condition_delta_C,
        tire_delta,
    )

    mass_changed = not math.isclose(
        baseline_effective_mass,
        mass_state.resolved_vehicle_mass_kg,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    force_curve_unchanged = all(
        math.isclose(point.baseline_force_N, point.scenario_force_N, rel_tol=0.0, abs_tol=1e-9)
        for point in force_curve
    )
    if (
        request.analysis_mode in {LoadCaseAnalysisMode.FIXED_SPEED, LoadCaseAnalysisMode.SPEED_SWEEP}
        and mass_changed
        and request.route.grade_pct == 0.0
        and not trailer_state.enabled
        and force_curve_unchanged
    ):
        explanatory_notes.append(
            "Mass changed, but steady-state force is unchanged at 0% grade because no active force "
            "contribution currently depends on the mass override."
        )
    if tire_status == "APPLIED" and tire_delta is not None and all(
        math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12)
        for value in (tire_delta.A_N, tire_delta.B_N_per_kph, tire_delta.C_N_per_kph2)
    ):
        explanatory_notes.append(
            "Tire-pressure override was APPLIED by the Tire Roadload owner; the resolved ABC force delta is zero."
        )

    if request.analysis_mode is LoadCaseAnalysisMode.DRIVE_CYCLE:
        active_cycle = cycle_frame if cycle_frame is not None else resolve_vehicle_demand_cycle(vde_row)
        if active_cycle is None:
            raise ValueError("No canonical drive cycle is available for the selected VDE.")
        baseline_result = calculate_vehicle_demand(baseline_request, active_cycle)
        scenario_result = calculate_vehicle_demand(scenario_engine_request, active_cycle)
        baseline_total_profile = build_vehicle_demand_profile(
            baseline_request, active_cycle, RoadloadBasis.TOTAL
        )
        scenario_total_profile = build_vehicle_demand_profile(
            scenario_engine_request, active_cycle, RoadloadBasis.TOTAL
        )
        total_comparison = _comparison(
            RoadloadBasis.TOTAL,
            baseline_result.total_summary.vde_mj_per_km,
            scenario_result.total_summary.vde_mj_per_km,
        )
        if total_comparison:
            comparisons.append(total_comparison)
        if baseline_result.net_summary is not None and scenario_result.net_summary is not None:
            net_comparison = _comparison(
                RoadloadBasis.NET,
                baseline_result.net_summary.vde_mj_per_km,
                scenario_result.net_summary.vde_mj_per_km,
            )
            if net_comparison:
                comparisons.append(net_comparison)
    elif request.analysis_mode is LoadCaseAnalysisMode.FIXED_SPEED:
        if request.fixed_speed is None:
            raise ValueError("fixed_speed is required for FIXED_SPEED mode.")
        speed = _positive(request.fixed_speed.speed_kph, "fixed_speed.speed_kph")
        points = (_steady_point(baseline_request, scenario_engine_request, speed, rolling_state.delta_abc, trailer_state, grade_N, condition_delta_C, tire_delta),)
    elif request.analysis_mode is LoadCaseAnalysisMode.SPEED_SWEEP:
        if request.speed_sweep is None:
            raise ValueError("speed_sweep is required for SPEED_SWEEP mode.")
        points = tuple(
            _steady_point(baseline_request, scenario_engine_request, speed, rolling_state.delta_abc, trailer_state, grade_N, condition_delta_C, tire_delta)
            for speed in speed_sweep_values(request.speed_sweep)
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported analysis mode: {request.analysis_mode!r}.")

    return RoadloadLoadCaseResult(
        request=request,
        baseline_request=baseline_request,
        scenario_request=scenario_request,
        scenario_engine_request=scenario_engine_request,
        mass_state=mass_state,
        trailer_state=trailer_state,
        rolling_state=rolling_state,
        baseline_result=baseline_result,
        scenario_result=scenario_result,
        baseline_total_profile=baseline_total_profile,
        scenario_total_profile=scenario_total_profile,
        comparisons=tuple(comparisons),
        steady_state_points=points,
        force_curve_points=force_curve,
        baseline_effective_mass_kg=baseline_effective_mass,
        baseline_calculation_mass_kg=baseline_effective_mass,
        baseline_mass_basis=baseline_mass_basis,
        baseline_mass_source_field=baseline_mass_source_field,
        scenario_vehicle_mass_kg=mass_state.resolved_vehicle_mass_kg,
        scenario_calculation_mass_kg=_positive(scenario_request.test_mass_kg, "scenario_calculation_mass_kg"),
        scenario_mass_basis=("GCWR" if request.mass.mode is LoadCaseMassMode.GCWR else mass_state.mass_basis),
        scenario_system_mass_kg=mass_state.system_mass_kg,
        rolling_rrc_value=rolling_state.rrc_value,
        rolling_rrc_source=rolling_state.source,
        rolling_rrc_rule_key=rolling_state.rule_key,
        rolling_correction_status=rolling_state.status,
        rolling_delta_abc=rolling_state.delta_abc,
        grade_force_N=grade_N,
        condition_delta_C_N_per_kph2=condition_delta_C,
        tire_delta_roadload=tire_delta,
        baseline_air_density_kg_m3=baseline_density,
        scenario_air_density_kg_m3=scenario_density,
        provenance={
            "baseline": Provenance.SOURCE.value,
            "baseline_mass": baseline_mass_basis,
            "baseline_mass_source_field": baseline_mass_source_field,
            "scenario_mass": ("GCWR" if request.mass.mode is LoadCaseMassMode.GCWR else mass_state.mass_basis),
            "rolling_correction": rolling_state.status,
            "rolling_rrc_source": rolling_state.source,
            "trailer": trailer_state.status,
            "grade": Provenance.CALCULATED.value,
            "ambient_aero_delta": ambient_status,
            "tire_pressure_delta": tire_status,
            "baseline_air_density": (
                _baseline_density_basis.value if _baseline_density_basis is not None else Provenance.UNAVAILABLE.value
            ),
            "scenario_air_density": (
                _scenario_density_basis.value if _scenario_density_basis is not None else Provenance.UNAVAILABLE.value
            ),
        },
        explanatory_notes=tuple(explanatory_notes),
        warnings=tuple(warnings),
    )


__all__ = [
    "LOAD_CASE_CONTRACT_VERSION",
    "MAX_SWEEP_POINTS",
    "LoadCaseAnalysisMode",
    "LoadCaseMassMode",
    "LoadCaseEnvironment",
    "LoadCaseMass",
    "LoadCaseTrailer",
    "LoadCaseRoute",
    "FixedSpeedSpec",
    "SpeedSweepSpec",
    "RoadloadLoadCaseRequest",
    "ResolvedMassState",
    "ResolvedTrailerState",
    "ResolvedRollingCorrection",
    "LoadCaseComparison",
    "SteadyStateBoundaryDemand",
    "SteadyStateDemandPoint",
    "ForceCurvePoint",
    "RoadloadLoadCaseResult",
    "baseline_ambient_from_vde_row",
    "resolve_baseline_mass",
    "resolve_scenario_ambient",
    "grade_force_N",
    "resolve_mass_state",
    "resolve_trailer_state",
    "resolve_load_dependent_rolling_delta",
    "resolve_tire_pressure_delta",
    "build_constant_speed_trace",
    "speed_sweep_values",
    "calculate_roadload_load_case",
]
