# src/vde_core/vehicle_demand/serialization.py
# -----------------------------------------------------------------------------
# JSON/API/MCP-ready boundary for the Vehicle Demand contracts (Sprint 9A
# Sec 10). This module only converts between the typed domain objects in
# contracts.py and plain, json.dumps-safe Python structures (dict/list/str/
# float/bool/None). It does not call any physics and has no Streamlit or
# database dependency.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping

import numpy as np

from .contracts import (
    AmbientState,
    EnergyMode,
    Provenance,
    RoadloadBasis,
    RoadloadCoefficients,
    VehicleDemandProfile,
    VehicleDemandRequest,
    VehicleDemandResult,
    VehicleDemandSummary,
)


def _clean_scalar(value: Any) -> Any:
    """Normalize one leaf value for the JSON boundary.

    NaN (Python float or numpy floating) becomes None -- distinct from a
    resolved 0 -- because "unavailable" must never be confused with "zero"
    (Sprint 9A Sec 8/13). numpy integer/floating/bool scalars are converted
    to native Python types since json.dumps cannot serialize numpy scalars.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        return None if math.isnan(as_float) else as_float
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def to_serializable(obj: Any) -> Any:
    """Recursively convert a Vehicle Demand contract object (or any nested
    dataclass/Enum/Mapping/sequence built from these contracts) into a plain,
    JSON-serializable structure.
    """
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {str(key): to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list, frozenset, set)):
        return [to_serializable(item) for item in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {name: to_serializable(getattr(obj, name)) for name in obj.__dataclass_fields__}
    return _clean_scalar(obj)


def _get(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return data.get(key, default)


def _enum_or_none(enum_cls, value: Any):
    return None if value is None else enum_cls(value)


def _tuple_or_none(values: Any):
    return None if values is None else tuple(values)


def ambient_state_from_dict(data: Mapping[str, Any] | None) -> AmbientState:
    data = data or {}
    return AmbientState(
        temperature_C=_get(data, "temperature_C"),
        pressure_kPa=_get(data, "pressure_kPa"),
        air_density_kg_m3=_get(data, "air_density_kg_m3"),
        temperature_basis=_enum_or_none(Provenance, _get(data, "temperature_basis")),
        pressure_basis=_enum_or_none(Provenance, _get(data, "pressure_basis")),
        density_basis=_enum_or_none(Provenance, _get(data, "density_basis")),
    )


def roadload_coefficients_from_dict(data: Mapping[str, Any] | None) -> RoadloadCoefficients | None:
    if data is None:
        return None
    return RoadloadCoefficients(
        A_N=_get(data, "A_N"),
        B_N_per_kph=_get(data, "B_N_per_kph"),
        C_N_per_kph2=_get(data, "C_N_per_kph2"),
    )


def vehicle_demand_request_from_dict(data: Mapping[str, Any]) -> VehicleDemandRequest:
    return VehicleDemandRequest(
        source_kind=data["source_kind"],
        vde_id=_get(data, "vde_id"),
        fuelcons_id=_get(data, "fuelcons_id"),
        label=data["label"],
        cycle_name=_get(data, "cycle_name"),
        cycle_source=_get(data, "cycle_source"),
        cycle_version=_get(data, "cycle_version"),
        test_mass_kg=_get(data, "test_mass_kg"),
        roadload_total=roadload_coefficients_from_dict(data["roadload_total"]),
        roadload_net=roadload_coefficients_from_dict(_get(data, "roadload_net")),
        rrc_n_per_kn=_get(data, "rrc_n_per_kn"),
        cda_m2=_get(data, "cda_m2"),
        ambient=ambient_state_from_dict(_get(data, "ambient")),
        provenance=dict(_get(data, "provenance") or {}),
        model_version=_get(data, "model_version"),
        contract_version=_get(data, "contract_version") or "0.1",
    )


def vehicle_demand_profile_from_dict(data: Mapping[str, Any]) -> VehicleDemandProfile:
    return VehicleDemandProfile(
        roadload_basis=RoadloadBasis(data["roadload_basis"]),
        time_s=tuple(data["time_s"]),
        speed_mps=tuple(data["speed_mps"]),
        accel_mps2=tuple(data["accel_mps2"]),
        authoritative_roadload_force_N=tuple(data["authoritative_roadload_force_N"]),
        inertial_force_N=tuple(data["inertial_force_N"]),
        tractive_force_N=tuple(data["tractive_force_N"]),
        authoritative_roadload_power_W=tuple(data["authoritative_roadload_power_W"]),
        inertial_power_W=tuple(data["inertial_power_W"]),
        tractive_power_W=tuple(data["tractive_power_W"]),
        energy_mode=tuple(EnergyMode(value) for value in data["energy_mode"]),
        known_rolling_force_N=_tuple_or_none(_get(data, "known_rolling_force_N")),
        known_aero_force_N=_tuple_or_none(_get(data, "known_aero_force_N")),
        residual_roadload_force_N=_tuple_or_none(_get(data, "residual_roadload_force_N")),
    )


def vehicle_demand_summary_from_dict(data: Mapping[str, Any]) -> VehicleDemandSummary:
    return VehicleDemandSummary(
        roadload_basis=RoadloadBasis(data["roadload_basis"]),
        distance_km=_get(data, "distance_km"),
        roadload_energy_MJ=_get(data, "roadload_energy_MJ"),
        known_rolling_energy_MJ=_get(data, "known_rolling_energy_MJ"),
        known_aero_energy_MJ=_get(data, "known_aero_energy_MJ"),
        residual_roadload_energy_MJ=_get(data, "residual_roadload_energy_MJ"),
        positive_inertial_work_MJ=_get(data, "positive_inertial_work_MJ"),
        positive_tractive_energy_MJ=_get(data, "positive_tractive_energy_MJ"),
        braking_energy_required_MJ=_get(data, "braking_energy_required_MJ"),
        vde_mj_per_km=_get(data, "vde_mj_per_km"),
        availability=frozenset(_get(data, "availability") or ()),
        warnings=tuple(_get(data, "warnings") or ()),
        provenance=dict(_get(data, "provenance") or {}),
        cycle_name=_get(data, "cycle_name"),
        cycle_source=_get(data, "cycle_source"),
        model_version=_get(data, "model_version"),
    )


def vehicle_demand_result_from_dict(data: Mapping[str, Any]) -> VehicleDemandResult:
    net_summary_data = _get(data, "net_summary")
    return VehicleDemandResult(
        total_summary=vehicle_demand_summary_from_dict(data["total_summary"]),
        net_summary=vehicle_demand_summary_from_dict(net_summary_data) if net_summary_data else None,
        metadata=dict(_get(data, "metadata") or {}),
    )


__all__ = [
    "to_serializable",
    "ambient_state_from_dict",
    "roadload_coefficients_from_dict",
    "vehicle_demand_request_from_dict",
    "vehicle_demand_profile_from_dict",
    "vehicle_demand_summary_from_dict",
    "vehicle_demand_result_from_dict",
]
