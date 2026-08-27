# src/vde_core/system_scenario/serialization.py
# -----------------------------------------------------------------------------
# JSON/API-boundary conversion for the System Scenario contracts (Sprint
# 11A). Reuses the existing, fully-generic `to_serializable()` from
# src/vde_core/vehicle_demand/serialization.py (canonical ownership -- this
# module does not redefine it) since System Scenario contracts nest frozen
# Sprint 9 VehicleDemandResult objects and that implementation already
# handles numpy scalars safely. No physics, no Streamlit, no DB dependency.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Mapping

from src.vde_core.vehicle_demand.serialization import to_serializable

from .contracts import (
    ArchitectureClass,
    ArchitectureConfiguration,
    AuxThermalConfiguration,
    ControlsConfiguration,
    DomainCorrection,
    DomainKind,
    DomainProposal,
    DomainProposalIdentity,
    DomainSourceState,
    EffectiveDomainState,
    ElectricDriveConfiguration,
    EnergyStorageConfiguration,
    EngineConfiguration,
    FidelityLevel,
    FidelityManifest,
    ProvenanceKind,
    SystemScenarioIdentity,
    SystemScenarioRole,
    TransmissionConfiguration,
    VehicleDemandConfiguration,
)


def _get(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return data.get(key, default)


def _enum_or_none(enum_cls, value: Any):
    return None if value is None else enum_cls(value)


def domain_kind_from_dict(value: Any) -> DomainKind:
    return DomainKind(value)


def vehicle_demand_configuration_from_dict(data: Mapping[str, Any] | None) -> VehicleDemandConfiguration:
    """`vehicle_demand_result` is intentionally NOT round-tripped here --
    reconstructing a frozen Sprint 9 `VehicleDemandResult` from a plain dict
    belongs to `vehicle_demand.serialization`'s own (not-yet-existing)
    from-dict helper, not to this module (Sprint 11A does not add one, since
    nothing in 11A requires deserializing a full VehicleDemandResult back
    from JSON -- only forward serialization via `to_serializable` is
    exercised). Round-tripping the reference/identity field is supported.
    """

    data = data or {}
    return VehicleDemandConfiguration(
        source_identity=_get(data, "source_identity"),
        vehicle_demand_result=None,
    )


def architecture_configuration_from_dict(data: Mapping[str, Any] | None) -> ArchitectureConfiguration:
    data = data or {}
    return ArchitectureConfiguration(
        architecture_class=_enum_or_none(ArchitectureClass, _get(data, "architecture_class")),
        topology_notes=_get(data, "topology_notes"),
    )


def engine_configuration_from_dict(data: Mapping[str, Any] | None) -> EngineConfiguration:
    data = data or {}
    return EngineConfiguration(
        fuel_type=_get(data, "fuel_type"),
        engine_family_id=_get(data, "engine_family_id"),
        displacement_l=_get(data, "displacement_l"),
        rated_power_kw=_get(data, "rated_power_kw"),
        rated_torque_nm=_get(data, "rated_torque_nm"),
        technology_descriptors=tuple(_get(data, "technology_descriptors") or ()),
    )


def transmission_configuration_from_dict(data: Mapping[str, Any] | None) -> TransmissionConfiguration:
    data = data or {}
    return TransmissionConfiguration(
        transmission_type=_get(data, "transmission_type"),
        transmission_model_id=_get(data, "transmission_model_id"),
        gear_count=_get(data, "gear_count"),
        final_drive_ratio=_get(data, "final_drive_ratio"),
    )


def electric_drive_configuration_from_dict(data: Mapping[str, Any] | None) -> ElectricDriveConfiguration:
    data = data or {}
    return ElectricDriveConfiguration(
        motor_role=_get(data, "motor_role"),
        motor_count=_get(data, "motor_count"),
        motor_position=_get(data, "motor_position"),
        rated_power_kw=_get(data, "rated_power_kw"),
        peak_power_kw=_get(data, "peak_power_kw"),
        rated_torque_nm=_get(data, "rated_torque_nm"),
        peak_torque_nm=_get(data, "peak_torque_nm"),
        nominal_voltage_v=_get(data, "nominal_voltage_v"),
        motor_identifier=_get(data, "motor_identifier"),
        inverter_identifier=_get(data, "inverter_identifier"),
    )


def energy_storage_configuration_from_dict(data: Mapping[str, Any] | None) -> EnergyStorageConfiguration:
    data = data or {}
    return EnergyStorageConfiguration(
        gross_capacity_kwh=_get(data, "gross_capacity_kwh"),
        usable_capacity_kwh=_get(data, "usable_capacity_kwh"),
        nominal_voltage_v=_get(data, "nominal_voltage_v"),
        charge_power_limit_kw=_get(data, "charge_power_limit_kw"),
        discharge_power_limit_kw=_get(data, "discharge_power_limit_kw"),
        regen_power_limit_kw=_get(data, "regen_power_limit_kw"),
        soc_window_low_pct=_get(data, "soc_window_low_pct"),
        soc_window_high_pct=_get(data, "soc_window_high_pct"),
    )


def controls_configuration_from_dict(data: Mapping[str, Any] | None) -> ControlsConfiguration:
    data = data or {}
    return ControlsConfiguration(
        hybrid_operating_strategy=_get(data, "hybrid_operating_strategy"),
        utility_factor_pct=_get(data, "utility_factor_pct"),
        regen_metadata=_get(data, "regen_metadata"),
        start_stop_enabled=_get(data, "start_stop_enabled"),
        calibration_notes=_get(data, "calibration_notes"),
    )


def aux_thermal_configuration_from_dict(data: Mapping[str, Any] | None) -> AuxThermalConfiguration:
    data = data or {}
    return AuxThermalConfiguration(notes=_get(data, "notes"))


_CONFIGURATION_FROM_DICT_BY_DOMAIN = {
    DomainKind.VEHICLE_DEMAND: vehicle_demand_configuration_from_dict,
    DomainKind.ARCHITECTURE: architecture_configuration_from_dict,
    DomainKind.ENGINE_FUEL_CONVERTER: engine_configuration_from_dict,
    DomainKind.TRANSMISSION_DRIVELINE: transmission_configuration_from_dict,
    DomainKind.ELECTRIC_DRIVE: electric_drive_configuration_from_dict,
    DomainKind.ENERGY_STORAGE: energy_storage_configuration_from_dict,
    DomainKind.ENERGY_MANAGEMENT_CONTROLS: controls_configuration_from_dict,
    DomainKind.AUX_THERMAL: aux_thermal_configuration_from_dict,
}


def domain_configuration_from_dict(domain: DomainKind, data: Mapping[str, Any] | None):
    """Dispatches on the already-known `domain` (never on the dict's own
    shape) to the matching per-domain from-dict helper -- consistent with
    `configuration_type_for`'s fixed, non-dynamic domain->type mapping in
    contracts.py."""

    return _CONFIGURATION_FROM_DICT_BY_DOMAIN[domain](data)


def domain_source_state_from_dict(data: Mapping[str, Any]) -> DomainSourceState:
    domain = domain_kind_from_dict(data["domain"])
    return DomainSourceState(
        domain=domain,
        configuration=domain_configuration_from_dict(domain, _get(data, "configuration")),
        provenance=ProvenanceKind(_get(data, "provenance") or ProvenanceKind.SOURCE_OBSERVED.value),
        notes=_get(data, "notes", ""),
    )


def domain_correction_from_dict(data: Mapping[str, Any]) -> DomainCorrection:
    domain = domain_kind_from_dict(data["domain"])
    return DomainCorrection(
        domain=domain,
        configuration=domain_configuration_from_dict(domain, _get(data, "configuration")),
        reason=_get(data, "reason", ""),
        provenance=ProvenanceKind(_get(data, "provenance") or ProvenanceKind.CORRECTED.value),
    )


def effective_domain_state_from_dict(data: Mapping[str, Any]) -> EffectiveDomainState:
    domain = domain_kind_from_dict(data["domain"])
    correction_data = _get(data, "correction")
    return EffectiveDomainState(
        domain=domain,
        configuration=domain_configuration_from_dict(domain, _get(data, "configuration")),
        source=domain_source_state_from_dict(data["source"]),
        correction=domain_correction_from_dict(correction_data) if correction_data else None,
        provenance=ProvenanceKind(_get(data, "provenance") or ProvenanceKind.SOURCE_OBSERVED.value),
    )


def domain_proposal_identity_from_dict(data: Mapping[str, Any]) -> DomainProposalIdentity:
    return DomainProposalIdentity(
        domain=domain_kind_from_dict(data["domain"]),
        proposal_id=data["proposal_id"],
    )


def domain_proposal_from_dict(data: Mapping[str, Any]) -> DomainProposal:
    domain = domain_kind_from_dict(data["domain"])
    return DomainProposal(
        identity=domain_proposal_identity_from_dict(data["identity"]),
        domain=domain,
        configuration=domain_configuration_from_dict(domain, _get(data, "configuration")),
        based_on=effective_domain_state_from_dict(data["based_on"]),
        label=_get(data, "label"),
        l0_effective_assumption=dict(_get(data, "l0_effective_assumption") or {}),
        technology_delta_ids=tuple(_get(data, "technology_delta_ids") or ()),
        notes=_get(data, "notes", ""),
    )


def fidelity_manifest_from_dict(data: Mapping[str, Any] | None) -> FidelityManifest:
    data = data or {}
    per_domain_raw = _get(data, "per_domain") or {}
    return FidelityManifest(
        per_domain={DomainKind(key): FidelityLevel(value) for key, value in per_domain_raw.items()}
    )


def system_scenario_identity_from_dict(data: Mapping[str, Any]) -> SystemScenarioIdentity:
    return SystemScenarioIdentity(
        scenario_id=data["scenario_id"],
        role=SystemScenarioRole(data["role"]),
        proposal_index=_get(data, "proposal_index"),
    )


__all__ = [
    "to_serializable",
    "domain_kind_from_dict",
    "vehicle_demand_configuration_from_dict",
    "architecture_configuration_from_dict",
    "engine_configuration_from_dict",
    "transmission_configuration_from_dict",
    "electric_drive_configuration_from_dict",
    "energy_storage_configuration_from_dict",
    "controls_configuration_from_dict",
    "aux_thermal_configuration_from_dict",
    "domain_configuration_from_dict",
    "domain_source_state_from_dict",
    "domain_correction_from_dict",
    "effective_domain_state_from_dict",
    "domain_proposal_identity_from_dict",
    "domain_proposal_from_dict",
    "fidelity_manifest_from_dict",
    "system_scenario_identity_from_dict",
]
