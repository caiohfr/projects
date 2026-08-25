# src/vde_core/vehicle_demand/physics.py
# -----------------------------------------------------------------------------
# Small, pure physics helpers for the Vehicle Demand engine (Sprint 9B).
#
# This module intentionally holds only the NEW physics 9B introduces (air
# density, known rolling/aero force, EnergyMode classification). The
# authoritative road-load/inertial/tractive math is NOT re-derived here --
# it lives in src/vde_core/vde_calc.py (compute_vde_series/compute_vde_net)
# and is reused by engine.py, never reimplemented in this package.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math

import numpy as np

from src.vde_core.roadload.tire_model import G_MPS2

from .contracts import AmbientState, EnergyMode, Provenance

ABSOLUTE_ZERO_C = -273.15

# Dry-air specific gas constant, J/(kg*K) -- ISO 2533 standard atmosphere
# value. Repo-wide search found no existing constant for this (Sprint 9A/9B
# investigation); this is the only new physical constant this package adds.
R_AIR_J_PER_KG_K = 287.058

# EnergyMode classification thresholds (Sprint 9B Sec 21). Both are small,
# fixed, local constants -- not a project-wide configuration system.
#
# SPEED_EPSILON_MPS: below this, the vehicle is treated as "approximately
# stopped" (IDLE) regardless of instantaneous tractive power. 0.05 m/s
# (0.18 km/h) is well under any real drive-cycle's slowest moving segment
# and only absorbs trace/discretization noise at a nominal stop.
SPEED_EPSILON_MPS = 0.05

# POWER_EPSILON_W: band around zero tractive power treated as COASTING
# rather than TRACTION/BRAKING. 5 W is small relative to any physically
# meaningful roadload/inertial power for a passenger vehicle at a nonzero
# speed (tens of W to tens of kW), so it only absorbs floating-point/
# np.gradient discretization noise at true zero-crossings, never a real
# driving intent.
POWER_EPSILON_W = 5.0


def _require_finite(value: float, field_name: str) -> float:
    """Fail fast on NaN/inf rather than let it silently propagate to a
    result that only reveals the problem once it reaches the JSON boundary
    as a None (Sprint 9C Sec 24). Missing (None) stays a soft "unavailable"
    everywhere in this module -- this only guards values that are PRESENT
    but not physically representable.
    """
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}.")
    return value


def resolve_air_density(ambient: AmbientState) -> tuple[float | None, Provenance | None, tuple[str, ...]]:
    """Resolve air density from AmbientState alone -- never invents a value.

    Hierarchy (Sprint 9B Sec 9): explicit density > calculated from
    temperature+pressure > unavailable. No regulatory-reference/canonical-
    assumption default is synthesized here: a repo-wide audit before this
    package found no existing standard-atmosphere constant anywhere, and
    Sprint 9B Sec 10 explicitly allows Aero Known to stay unavailable rather
    than fabricate one ("Aero Known = unavailable is acceptable, do not
    block Vehicle Demand for it").

    A value that is PRESENT but physically impossible (temperature at/below
    absolute zero, non-positive pressure, non-positive density) raises
    ValueError (Sprint 9C Sec 22) rather than silently producing a
    nonsensical or negative density -- this is distinct from a value that is
    simply absent, which stays a soft "unavailable" as above.
    """
    if ambient.air_density_kg_m3 is not None:
        density = _require_finite(float(ambient.air_density_kg_m3), "AmbientState.air_density_kg_m3")
        if density <= 0:
            raise ValueError(f"AmbientState.air_density_kg_m3 must be positive, got {density!r}.")
        basis = ambient.density_basis or Provenance.SOURCE
        return density, basis, ()

    if ambient.temperature_C is not None and ambient.pressure_kPa is not None:
        temperature_C = _require_finite(float(ambient.temperature_C), "AmbientState.temperature_C")
        pressure_kPa = _require_finite(float(ambient.pressure_kPa), "AmbientState.pressure_kPa")
        if temperature_C <= ABSOLUTE_ZERO_C:
            raise ValueError(f"AmbientState.temperature_C must be above absolute zero, got {temperature_C!r}.")
        if pressure_kPa <= 0:
            raise ValueError(f"AmbientState.pressure_kPa must be positive, got {pressure_kPa!r}.")
        temperature_kelvin = temperature_C + 273.15
        pressure_pa = pressure_kPa * 1000.0
        rho = pressure_pa / (R_AIR_J_PER_KG_K * temperature_kelvin)
        return rho, Provenance.CALCULATED, ()

    if ambient.temperature_C is not None or ambient.pressure_kPa is not None:
        return None, None, ("Air density requires both temperature_C and pressure_kPa; only one was provided.",)

    return None, None, ()


def known_rolling_force_N(rrc_n_per_kn: float | None, mass_kg: float | None) -> float | None:
    """Vehicle-level rolling force from RRC and effective mass.

    F = rrc_n_per_kn * load_kN, mirroring the ISO MVP tire model already
    canonical in roadload.tire_model.calculate_iso_tire_abc_for_single_tire
    (A = rr_n_per_kn * load_kN, B = C = 0, i.e. speed-independent). That
    function operates per tire on a front/rear axle load split; this vehicle
    -level form is the algebraic sum of that formula across all tires when
    one RRC applies vehicle-wide (axle loads always sum to the full vehicle
    weight), which is the only form VehicleDemandRequest.rrc_n_per_kn (a
    single scalar, no axle split) can represent.

    RRC = 0 is a valid known-zero rolling force. RRC < 0 raises ValueError
    (Sprint 9C Sec 20) rather than silently producing a negative rolling
    force -- the project has no defined upper bound for RRC, so only the
    physically-impossible negative case is rejected here.
    """
    if rrc_n_per_kn is None or mass_kg is None:
        return None
    rrc = _require_finite(float(rrc_n_per_kn), "rrc_n_per_kn")
    if rrc < 0:
        raise ValueError(f"rrc_n_per_kn must be non-negative, got {rrc!r}.")
    load_kN = _require_finite(float(mass_kg), "mass_kg") * G_MPS2 / 1000.0
    return rrc * load_kN


def known_aero_force_N(cda_m2: float | None, air_density_kg_m3: float | None, speed_mps: np.ndarray) -> np.ndarray | None:
    """F_aero(t) = 0.5 * rho * CdA * v(t)^2. CdA = 0 is a valid known zero;
    a missing CdA or unresolved rho makes the whole series unavailable.
    CdA < 0 raises ValueError (Sprint 9C Sec 21) rather than silently
    producing a negative aero force.
    """
    if cda_m2 is None or air_density_kg_m3 is None:
        return None
    cda = _require_finite(float(cda_m2), "cda_m2")
    if cda < 0:
        raise ValueError(f"cda_m2 must be non-negative, got {cda!r}.")
    rho = _require_finite(float(air_density_kg_m3), "air_density_kg_m3")
    return 0.5 * rho * cda * np.square(speed_mps)


def classify_energy_mode(speed_mps: np.ndarray, tractive_power_W: np.ndarray) -> tuple[EnergyMode, ...]:
    """Classify each timestep from tractive demand alone (Sprint 9B Sec 20-22).

    Deceleration is NOT classified as BRAKING by itself -- only the sign of
    tractive_power_W (which already reflects both road-load and inertial
    force) decides TRACTION vs COASTING vs BRAKING. IDLE is decided purely
    by speed, independent of the power sign.
    """
    modes = []
    for speed, power in zip(speed_mps, tractive_power_W):
        if abs(float(speed)) <= SPEED_EPSILON_MPS:
            modes.append(EnergyMode.IDLE)
        elif power > POWER_EPSILON_W:
            modes.append(EnergyMode.TRACTION)
        elif power < -POWER_EPSILON_W:
            modes.append(EnergyMode.BRAKING)
        else:
            modes.append(EnergyMode.COASTING)
    return tuple(modes)


__all__ = [
    "R_AIR_J_PER_KG_K",
    "ABSOLUTE_ZERO_C",
    "SPEED_EPSILON_MPS",
    "POWER_EPSILON_W",
    "resolve_air_density",
    "known_rolling_force_N",
    "known_aero_force_N",
    "classify_energy_mode",
]
