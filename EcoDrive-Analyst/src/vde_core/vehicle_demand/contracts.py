# src/vde_core/vehicle_demand/contracts.py
# -----------------------------------------------------------------------------
# Sprint 9A - canonical, Streamlit-free contracts for the future Vehicle
# Demand Model layer.
#
# This module defines ONLY data shape. It does not compute roadload, VDE,
# acceleration, or cycle distance -- those remain the responsibility of the
# existing physics in src/vde_core/roadload/ and
# vde_setup_service.compute_vde_preview_from_inputs. A future Vehicle Demand
# Engine (Sprint 9B) will consume VehicleDemandRequest and produce
# VehicleDemandProfile/VehicleDemandSummary using that existing physics; it is
# not implemented here.
#
# Core principle (see docs/sprints/PACKAGE_9A_VEHICLE_DEMAND_CONTRACTS.md):
# TOTAL and NET are authoritative values already resolved elsewhere in the
# codebase (vde_net_total_contract.canonical_vde_read). This module never
# reconstructs them from components, and never lets one silently stand in for
# the other -- absence stays absence, zero stays zero.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


VEHICLE_DEMAND_CONTRACT_VERSION = "0.1"


class _TextEnum(str, Enum):
    """Same str-Enum shape used throughout vde_core (see
    vde_net_total_contract._TextEnum, comparison_report_service._TextEnum,
    fuel_energy.LhvBasis) -- JSON-serializable as a plain string, stable
    across the module boundary.
    """

    def __str__(self) -> str:
        return self.value


class Provenance(_TextEnum):
    """Shared provenance vocabulary for the Vehicle Demand contracts.

    Reused across AmbientState's per-field basis, VehicleDemandRequest's
    provenance map, and VehicleDemandSummary's provenance map, so a value
    such as "temperature = REGULATORY_REFERENCE" or "rho = CALCULATED" means
    the same thing everywhere it appears. This is intentionally a single
    small enum rather than a second project-wide provenance framework --
    see Sprint 9A package Sec 12.
    """

    SOURCE = "SOURCE"
    RESOLVED = "RESOLVED"
    REGULATORY_REFERENCE = "REGULATORY_REFERENCE"
    CANONICAL_ASSUMPTION = "CANONICAL_ASSUMPTION"
    CALCULATED = "CALCULATED"
    ASSUMED = "ASSUMED"
    UNAVAILABLE = "UNAVAILABLE"


class RoadloadBasis(_TextEnum):
    """TOTAL and NET are distinct, explicit values with no implicit fallback
    between them. A caller that has only TOTAL must not receive a NET value,
    and vice versa -- see vde_net_total_contract.py, the pre-existing
    authority for this rule at the vde_db row level.
    """

    TOTAL = "TOTAL"
    NET = "NET"


class EnergyMode(_TextEnum):
    """Four fundamental wheel-side demand states (Sprint 9A Sec 4.3).

    IDLE     -- vehicle approximately stopped.
    TRACTION -- positive tractive demand at the wheels.
    COASTING -- approximately zero tractive demand.
    BRAKING  -- negative tractive demand; wheel-side braking required.

    Deceleration is not necessarily BRAKING (e.g. coasting deceleration under
    roadload alone). Kinematic phases such as LAUNCH/ACCELERATION/CRUISE/
    DECELERATION are a separate, future classification and are deliberately
    not represented by this enum.
    """

    IDLE = "IDLE"
    TRACTION = "TRACTION"
    COASTING = "COASTING"
    BRAKING = "BRAKING"


@dataclass(frozen=True)
class AmbientState:
    """Ambient conditions available to future physical corrections.

    Units: temperature_C in degrees Celsius, pressure_kPa in kilopascals,
    air_density_kg_m3 in kg/m^3. All fields are independently optional --
    Sprint 9A does not implement the rho = p / (R_air * T) correction engine,
    it only defines a shape that can hold the inputs and result of that
    future calculation. A None field means "not resolved yet", never 0.
    """

    temperature_C: float | None = None
    pressure_kPa: float | None = None
    air_density_kg_m3: float | None = None

    temperature_basis: Provenance | None = None
    pressure_basis: Provenance | None = None
    density_basis: Provenance | None = None


@dataclass(frozen=True)
class RoadloadCoefficients:
    """A/B/C road-load coefficients for one boundary (TOTAL or NET).

    Units match the project-wide convention already used by vde_db and
    roadload/models.py: N, N/kph, N/kph^2. This is a deliberately small,
    local struct -- RoadLoadComponent/EquivalentABC/RoadloadBoundary each
    already define an equivalent A/B/C triplet for their own layer, and
    Vehicle Demand contracts should not import cross-domain from the
    Comparison layer (src/vde_core/comparison_report_service.py) merely to
    reuse a 3-field struct.
    """

    A_N: float | None
    B_N_per_kph: float | None
    C_N_per_kph2: float | None


@dataclass(frozen=True)
class VehicleDemandRequest:
    """Input contract for the future Vehicle Demand Engine (Sprint 9B).

    Composes references to already-resolved upstream objects (scenario
    identity, cycle identity, authoritative ABC) rather than duplicating the
    full vde_db row -- see Sprint 9A Sec 5. roadload_total is the only
    required roadload boundary; roadload_net is optional and, when absent,
    must never be filled in from roadload_total.
    """

    # Resolved vehicle/scenario source identity. source_kind mirrors the
    # existing SourceKind vocabulary ("VDE_ONLY" / "FUELCONS_SCENARIO") in
    # comparison_report_service.py without importing that enum directly.
    source_kind: str
    vde_id: int | None
    fuelcons_id: int | None
    label: str

    # Cycle identity -- the canonical CSV name resolvable via
    # src/vde_core/cycles.py, never an embedded full trace.
    cycle_name: str | None
    cycle_source: str | None
    cycle_version: str | None

    # Effective test mass used for inertial force/power.
    test_mass_kg: float | None

    # Authoritative roadload -- always the already-resolved values, never
    # rebuilt from components.
    roadload_total: RoadloadCoefficients
    roadload_net: RoadloadCoefficients | None

    # Resolved physical references (optional, informational only; not used
    # to reconstruct roadload_total/roadload_net).
    rrc_n_per_kn: float | None
    cda_m2: float | None

    ambient: AmbientState

    # field name -> Provenance value, e.g. {"roadload_total": "SOURCE"}.
    provenance: Mapping[str, str] = field(default_factory=dict)

    # Which upstream physics/model version produced the authoritative
    # values above (e.g. "VDE_SETUP_V22"), not this contract's own version.
    model_version: str | None = None

    contract_version: str = VEHICLE_DEMAND_CONTRACT_VERSION


@dataclass(frozen=True)
class VehicleDemandProfile:
    """Time-resolved physical result for ONE RoadloadBasis (Sprint 9A Sec 6).

    One VehicleDemandProfile per RoadloadBasis (never one object holding
    both TOTAL and NET series) -- see Sprint 9A Sec 6. Sprint 9A defines
    this shape only; no physics populates it yet.

    Units: time_s in seconds, speed_mps in m/s, accel_mps2 in m/s^2, forces
    in N, powers in W.

    Sign convention: tractive_power_W > 0 is propulsion/traction demand,
    < 0 is braking demand. known_*/residual_* component forces are optional
    because component attribution is not always available (see
    roadload_analysis.build_cycle_power_analysis's existing
    decomposition_available=False precedent) -- their absence is distinct
    from a resolved value of 0. residual_roadload_force_N is signed and may
    legitimately go negative (Sprint 9B Sec 15-16): it is
    authoritative_roadload_force_N minus whichever known contributions were
    actually calculated, never clipped or redistributed, so it is the one
    force field that is not itself a "how much resistance" magnitude.
    """

    roadload_basis: RoadloadBasis

    time_s: tuple[float, ...]
    speed_mps: tuple[float, ...]
    accel_mps2: tuple[float, ...]

    authoritative_roadload_force_N: tuple[float, ...]
    inertial_force_N: tuple[float, ...]
    tractive_force_N: tuple[float, ...]

    authoritative_roadload_power_W: tuple[float, ...]
    inertial_power_W: tuple[float, ...]
    tractive_power_W: tuple[float, ...]

    energy_mode: tuple[EnergyMode, ...]

    known_rolling_force_N: tuple[float | None, ...] | None = None
    known_aero_force_N: tuple[float | None, ...] | None = None
    residual_roadload_force_N: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        n = len(self.time_s)
        required = {
            "speed_mps": self.speed_mps,
            "accel_mps2": self.accel_mps2,
            "authoritative_roadload_force_N": self.authoritative_roadload_force_N,
            "inertial_force_N": self.inertial_force_N,
            "tractive_force_N": self.tractive_force_N,
            "authoritative_roadload_power_W": self.authoritative_roadload_power_W,
            "inertial_power_W": self.inertial_power_W,
            "tractive_power_W": self.tractive_power_W,
            "energy_mode": self.energy_mode,
        }
        for name, series in required.items():
            if len(series) != n:
                raise ValueError(
                    f"VehicleDemandProfile field '{name}' has length {len(series)}, "
                    f"expected {n} (time_s length)."
                )
        optional = {
            "known_rolling_force_N": self.known_rolling_force_N,
            "known_aero_force_N": self.known_aero_force_N,
            "residual_roadload_force_N": self.residual_roadload_force_N,
        }
        for name, series in optional.items():
            if series is not None and len(series) != n:
                raise ValueError(
                    f"VehicleDemandProfile field '{name}' has length {len(series)}, "
                    f"expected {n} (time_s length)."
                )


@dataclass(frozen=True)
class VehicleDemandSummary:
    """Cycle-level aggregates for ONE RoadloadBasis (Sprint 9A Sec 8).

    Energy fields use the project's dominant VDE energy convention, MJ (see
    fuel_energy.py / vde_setup_service.compute_vde_preview_from_inputs,
    which store cycle results in MJ/km); vde_mj_per_km is the rate form of
    the same convention. Aggregates that represent a quantity of energy are
    non-negative magnitudes (Sprint 9A Sec 7) -- direction is carried by the
    field's own name (e.g. braking_energy_required_MJ), never by a negative
    sign on an aggregate. residual_roadload_energy_MJ is the one exception
    to this non-negative-magnitude convention (Sprint 9B Sec 27): it is the
    integral of authoritative-minus-known roadload power and may legitimately
    be negative when known contributions exceed the authoritative roadload
    (see VehicleDemandProfile.residual_roadload_force_N) -- clipping it to
    zero would hide a real basis mismatch rather than report it.

    A None field means unavailable/not computed; it is never coerced to 0,
    and a 0 value is never coerced to None.
    """

    roadload_basis: RoadloadBasis

    distance_km: float | None = None

    roadload_energy_MJ: float | None = None
    known_rolling_energy_MJ: float | None = None
    known_aero_energy_MJ: float | None = None
    residual_roadload_energy_MJ: float | None = None

    positive_inertial_work_MJ: float | None = None

    positive_tractive_energy_MJ: float | None = None
    braking_energy_required_MJ: float | None = None

    vde_mj_per_km: float | None = None

    availability: frozenset[str] = frozenset()
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)

    cycle_name: str | None = None
    cycle_source: str | None = None
    model_version: str | None = None


@dataclass(frozen=True)
class VehicleDemandResult:
    """Simple TOTAL/NET container -- only justified because callers routinely
    need both boundaries side by side (mirrors the TOTAL/NET pairing already
    used throughout comparison_report_service.py and roadload_analysis.py).
    Does not embed VehicleDemandProfile: profiles are computed on demand and
    are never persisted (Sprint 9A Sec 9), so bundling one into a result
    object would invite accidental persistence/serialization of a large
    time series that this contract does not own the lifecycle of.
    """

    total_summary: VehicleDemandSummary
    net_summary: VehicleDemandSummary | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_summary.roadload_basis is not RoadloadBasis.TOTAL:
            raise ValueError("VehicleDemandResult.total_summary must have roadload_basis TOTAL.")
        if self.net_summary is not None and self.net_summary.roadload_basis is not RoadloadBasis.NET:
            raise ValueError("VehicleDemandResult.net_summary must have roadload_basis NET.")


__all__ = [
    "VEHICLE_DEMAND_CONTRACT_VERSION",
    "Provenance",
    "RoadloadBasis",
    "EnergyMode",
    "AmbientState",
    "RoadloadCoefficients",
    "VehicleDemandRequest",
    "VehicleDemandProfile",
    "VehicleDemandSummary",
    "VehicleDemandResult",
]
