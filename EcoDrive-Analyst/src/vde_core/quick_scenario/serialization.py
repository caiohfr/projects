# src/vde_core/quick_scenario/serialization.py
# -----------------------------------------------------------------------------
# JSON/API-boundary conversion for the Quick Scenario contracts (Sprint 10A).
# This module only converts between the typed domain objects in contracts.py
# and plain, json.dumps-safe Python structures (dict/list/str/float/bool/
# None). It does not call any physics and has no Streamlit or database
# dependency, mirroring src/vde_core/vehicle_demand/serialization.py.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping

from .contracts import (
    QUICK_SCENARIO_CONTRACT_VERSION,
    DomainReadiness,
    EfficiencyQuickInputs,
    MassQuickChange,
    PseProvenance,
    QuickScenario,
    QuickVehicleReadiness,
    ReferencePressureProvenance,
    ScalarChange,
    ScalarChangeMode,
    TechDeltaAssumption,
    TirePressureDelta,
    TireQuickChange,
    TireSource,
    TireTransformMode,
    VehicleQuickOverrides,
)


def _clean_scalar(value: Any) -> Any:
    """Normalize one leaf value for the JSON boundary. NaN becomes None --
    distinct from a resolved 0 -- because "unavailable" must never be
    confused with "zero" (mirrors vehicle_demand.serialization._clean_scalar).
    """

    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def to_serializable(obj: Any) -> Any:
    """Recursively convert a Quick Scenario contract object (or any nested
    dataclass/Enum/Mapping/sequence built from these contracts) into a
    plain, JSON-serializable structure.
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


def scalar_change_from_dict(data: Mapping[str, Any] | None) -> ScalarChange | None:
    if data is None:
        return None
    return ScalarChange(mode=ScalarChangeMode(data["mode"]), value=float(data["value"]))


def mass_quick_change_from_dict(data: Mapping[str, Any] | None) -> MassQuickChange | None:
    if data is None:
        return None
    return MassQuickChange(
        curb_change=scalar_change_from_dict(_get(data, "curb_change")),
        twc_shift_steps=_get(data, "twc_shift_steps"),
        twc_shift_side=_get(data, "twc_shift_side"),
        twc_curb_position=_get(data, "twc_curb_position"),
        wltp_line_type=_get(data, "wltp_line_type"),
    )


def tire_pressure_delta_from_dict(data: Mapping[str, Any] | None) -> TirePressureDelta | None:
    if data is None:
        return None
    return TirePressureDelta(
        front_delta_psi=float(data["front_delta_psi"]),
        rear_delta_psi=_get(data, "rear_delta_psi"),
        reference_pressure_psi=_get(data, "reference_pressure_psi"),
        reference_pressure_provenance=_enum_or_none(
            ReferencePressureProvenance, _get(data, "reference_pressure_provenance")
        ),
    )


def tire_quick_change_from_dict(data: Mapping[str, Any] | None) -> TireQuickChange | None:
    if data is None:
        return None
    return TireQuickChange(
        source=TireSource(data["source"]),
        transform_mode=TireTransformMode(_get(data, "transform_mode") or TireTransformMode.NONE.value),
        tire_db_id=_get(data, "tire_db_id"),
        target_rrc_n_per_kn=_get(data, "target_rrc_n_per_kn"),
        rrc_delta_n_per_kn=_get(data, "rrc_delta_n_per_kn"),
        improvement_pct=_get(data, "improvement_pct"),
        pressure_delta=tire_pressure_delta_from_dict(_get(data, "pressure_delta")),
    )


def vehicle_quick_overrides_from_dict(data: Mapping[str, Any] | None) -> VehicleQuickOverrides:
    data = data or {}
    return VehicleQuickOverrides(
        mass_change=mass_quick_change_from_dict(_get(data, "mass_change")),
        cda_change=scalar_change_from_dict(_get(data, "cda_change")),
        aero_reference_cda_m2=_get(data, "aero_reference_cda_m2"),
        aero_reference_cda_provenance=_enum_or_none(
            ReferencePressureProvenance, _get(data, "aero_reference_cda_provenance")
        ),
        tire_change=tire_quick_change_from_dict(_get(data, "tire_change")),
    )


def quick_vehicle_readiness_from_dict(data: Mapping[str, Any] | None) -> QuickVehicleReadiness:
    data = data or {}
    return QuickVehicleReadiness(
        mass=DomainReadiness(_get(data, "mass") or DomainReadiness.NOT_REQUESTED.value),
        aero=DomainReadiness(_get(data, "aero") or DomainReadiness.NOT_REQUESTED.value),
        tire=DomainReadiness(_get(data, "tire") or DomainReadiness.NOT_REQUESTED.value),
    )


def tech_delta_assumption_from_dict(data: Mapping[str, Any]) -> TechDeltaAssumption:
    return TechDeltaAssumption(
        name=data["name"],
        effect_basis=data["effect_basis"],
        effect_value=float(data["effect_value"]),
        affected_subsystem=_get(data, "affected_subsystem", "whole powertrain"),
        source_type=_get(data, "source_type", "manual"),
        maturity_level=_get(data, "maturity_level", "engineering_assumption"),
        confidence=_get(data, "confidence", "unknown"),
        notes=_get(data, "notes", ""),
        enabled=bool(_get(data, "enabled", True)),
    )


def efficiency_quick_inputs_from_dict(data: Mapping[str, Any] | None) -> EfficiencyQuickInputs:
    data = data or {}
    return EfficiencyQuickInputs(
        benchmark_source_identity=_get(data, "benchmark_source_identity"),
        request_ml_recommendation=bool(_get(data, "request_ml_recommendation", False)),
        technology_deltas=tuple(
            tech_delta_assumption_from_dict(item) for item in _get(data, "technology_deltas") or ()
        ),
    )


def quick_scenario_from_dict(data: Mapping[str, Any]) -> QuickScenario:
    return QuickScenario(
        source_identity=data["source_identity"],
        slot=int(data["slot"]),
        label=_get(data, "label"),
        vehicle_overrides=vehicle_quick_overrides_from_dict(_get(data, "vehicle_overrides")),
        vehicle_readiness=quick_vehicle_readiness_from_dict(_get(data, "vehicle_readiness")),
        efficiency_inputs=efficiency_quick_inputs_from_dict(_get(data, "efficiency_inputs")),
        final_pse_percent=_get(data, "final_pse_percent"),
        pse_provenance=_enum_or_none(PseProvenance, _get(data, "pse_provenance")),
        issues=tuple(_get(data, "issues") or ()),
        contract_version=_get(data, "contract_version") or QUICK_SCENARIO_CONTRACT_VERSION,
    )


__all__ = [
    "to_serializable",
    "scalar_change_from_dict",
    "mass_quick_change_from_dict",
    "tire_pressure_delta_from_dict",
    "tire_quick_change_from_dict",
    "vehicle_quick_overrides_from_dict",
    "quick_vehicle_readiness_from_dict",
    "tech_delta_assumption_from_dict",
    "efficiency_quick_inputs_from_dict",
    "quick_scenario_from_dict",
]
