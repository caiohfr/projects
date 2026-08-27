# src/vde_core/system_scenario/__init__.py
# -----------------------------------------------------------------------------
# Sprint 11A/11B - canonical, Streamlit-free contracts + legacy adapters +
# domain resolution service for the multi-domain System Scenario
# architecture.
#
#     legacy vde_db/fuelcons_db row
#         -> adapter (legacy_adapter.py)
#         -> canonical DomainSourceState (contracts.py)
#         -> [11B] resolve_effective_domain_state() -> EffectiveDomainState
#         -> [11B] domain_resolution.resolve_domain_proposal() -> DomainProposal
#         -> [11C] SystemScenarioDefinition -> Resolver -> ResolvedSystemScenario
#         -> [11C] EnergyBalanceL0Adapter -> existing FuelEstimateRequest/run_fuel_estimation()
#         -> [11C] SystemScenarioResult
#
# `contracts.py` defines data shape only and stays Streamlit- and DB-free
# (it does import the existing canonical `TechDeltaAssumption` from
# `quick_scenario.contracts` -- reuse, not a second Technology Delta
# schema). `legacy_adapter.py` is the one place this package touches a raw
# row shape, and it never exposes that shape through the canonical
# contracts themselves (INV-11-012). `domain_resolution.py` is the
# Streamlit-independent service layer that turns Effective Current +
# requested changes into a Domain Proposal. No file in this package
# imports Streamlit, calls `fuel_estimation.run_fuel_estimation`, or
# stacks/combines Technology Deltas -- see
# docs/sprints/SPRINT_11A_SYSTEM_SCENARIO_CONTRACTS.md and
# docs/sprints/SPRINT_11B_DOMAIN_STATES_AND_PROPOSALS.md for the full
# audit and design record this package was built against.
# -----------------------------------------------------------------------------

from .contracts import (
    ALL_DOMAIN_KINDS,
    MAX_DOMAIN_PROPOSALS_PER_DOMAIN,
    MAX_SYSTEM_SCENARIO_PROPOSALS,
    SYSTEM_SCENARIO_CONTRACT_VERSION,
    ArchitectureClass,
    ArchitectureConfiguration,
    AuxThermalConfiguration,
    ControlsConfiguration,
    DomainApplicability,
    DomainConfiguration,
    DomainCorrection,
    DomainKind,
    DomainProposal,
    DomainProposalIdentity,
    DomainSelection,
    DomainSourceState,
    EffectiveDomainState,
    ElectricDriveConfiguration,
    EnergyStorageConfiguration,
    EngineConfiguration,
    FidelityLevel,
    FidelityManifest,
    ProvenanceKind,
    ResolvedSystemScenario,
    SystemScenarioDefinition,
    SystemScenarioIdentity,
    SystemScenarioResult,
    SystemScenarioRole,
    TransmissionConfiguration,
    VehicleDemandConfiguration,
    configuration_type_for,
    domain_applicability_for,
    domain_typically_applicable,
    resolve_effective_domain_state,
    resolve_system_scenario_shell,
)
from .domain_resolution import changed_fields, resolve_domain_proposal
from .legacy_adapter import (
    architecture_domain_state_from_legacy_vde_row,
    aux_thermal_domain_state_from_legacy_row,
    controls_domain_state_from_legacy_row,
    electric_drive_domain_state_sparse,
    energy_storage_domain_state_from_legacy_row,
    engine_domain_state_from_legacy_row,
    transmission_domain_state_from_legacy_row,
    vehicle_demand_domain_state_from_legacy_vde_row,
    vehicle_demand_domain_state_from_result,
)
from .serialization import to_serializable

__all__ = [
    "SYSTEM_SCENARIO_CONTRACT_VERSION",
    "MAX_SYSTEM_SCENARIO_PROPOSALS",
    "MAX_DOMAIN_PROPOSALS_PER_DOMAIN",
    "DomainKind",
    "ALL_DOMAIN_KINDS",
    "ArchitectureClass",
    "DomainApplicability",
    "domain_applicability_for",
    "FidelityLevel",
    "ProvenanceKind",
    "VehicleDemandConfiguration",
    "ArchitectureConfiguration",
    "EngineConfiguration",
    "TransmissionConfiguration",
    "ElectricDriveConfiguration",
    "EnergyStorageConfiguration",
    "ControlsConfiguration",
    "AuxThermalConfiguration",
    "DomainConfiguration",
    "configuration_type_for",
    "domain_typically_applicable",
    "DomainSourceState",
    "DomainCorrection",
    "EffectiveDomainState",
    "resolve_effective_domain_state",
    "DomainProposalIdentity",
    "DomainProposal",
    "DomainSelection",
    "SystemScenarioRole",
    "SystemScenarioIdentity",
    "SystemScenarioDefinition",
    "FidelityManifest",
    "ResolvedSystemScenario",
    "SystemScenarioResult",
    "resolve_system_scenario_shell",
    "resolve_domain_proposal",
    "changed_fields",
    "vehicle_demand_domain_state_from_result",
    "vehicle_demand_domain_state_from_legacy_vde_row",
    "engine_domain_state_from_legacy_row",
    "transmission_domain_state_from_legacy_row",
    "architecture_domain_state_from_legacy_vde_row",
    "electric_drive_domain_state_sparse",
    "energy_storage_domain_state_from_legacy_row",
    "controls_domain_state_from_legacy_row",
    "aux_thermal_domain_state_from_legacy_row",
    "to_serializable",
]
