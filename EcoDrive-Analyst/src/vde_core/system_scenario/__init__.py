# src/vde_core/system_scenario/__init__.py
# -----------------------------------------------------------------------------
# Sprint 11A - canonical, Streamlit-free contracts + minimal legacy adapter
# for the multi-domain System Scenario architecture.
#
#     legacy vde_db/fuelcons_db row
#         -> adapter (legacy_adapter.py)
#         -> canonical Domain State(s) (contracts.py)
#         -> [11B] Corrections / Effective Current / Domain Proposals
#         -> [11C] SystemScenarioDefinition -> Resolver -> ResolvedSystemScenario
#         -> [11C] EnergyBalanceL0Adapter -> existing FuelEstimateRequest/run_fuel_estimation()
#         -> [11C] SystemScenarioResult
#
# `contracts.py` defines data shape only and stays Streamlit- and
# DB-free. `legacy_adapter.py` is the one place this package touches a raw
# row shape, and it never exposes that shape through the canonical
# contracts themselves (INV-11-012). No file in this package imports
# Streamlit, calls `fuel_estimation.run_fuel_estimation`, or composes
# Technology Deltas -- see docs/sprints/SPRINT_11A_SYSTEM_SCENARIO_CONTRACTS.md
# for the full audit and design record this package was built against.
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
    domain_typically_applicable,
    resolve_effective_domain_state,
    resolve_system_scenario_shell,
)
from .legacy_adapter import (
    architecture_domain_state_from_legacy_vde_row,
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
    "vehicle_demand_domain_state_from_result",
    "vehicle_demand_domain_state_from_legacy_vde_row",
    "engine_domain_state_from_legacy_row",
    "transmission_domain_state_from_legacy_row",
    "architecture_domain_state_from_legacy_vde_row",
    "to_serializable",
]
