"""
Canonical, Streamlit-free contracts for Interactive Quick Scenarios
(Sprint 10A).

    Existing resolved Comparison scenario (source_identity)
        -> QuickScenario (Vehicle overrides + Final PSE assumption)
        -> [later package] canonical Mass/Tire/Aero resolvers + PSE/fuel path
        -> Temporary Comparison Result (never persisted)

This package defines data shape only. It does not resolve Mass/Tire/Aero,
does not call the canonical VDE resolvers, and does not import
src/vde_core/vehicle_demand/ (frozen Sprint 9 core) or any Streamlit/
Comparison UI module. See
docs/sprints/SPRINT_10A_QUICK_SCENARIO_CONTRACT_AUDIT.md for the reuse audit
these contracts were designed against.
"""

from .contracts import (
    MAX_QUICK_SCENARIOS_PER_SOURCE,
    QUICK_SCENARIO_CONTRACT_VERSION,
    QUICK_SCENARIO_IDENTITY_PREFIX,
    DomainReadiness,
    PseProvenance,
    QuickScenario,
    QuickVehicleReadiness,
    ReferencePressureProvenance,
    ScalarChange,
    ScalarChangeMode,
    TirePressureDelta,
    TireQuickChange,
    TireSource,
    TireTransformMode,
    VehicleQuickOverrides,
    build_quick_scenario_identity,
)
from .serialization import (
    quick_scenario_from_dict,
    quick_vehicle_readiness_from_dict,
    scalar_change_from_dict,
    tire_pressure_delta_from_dict,
    tire_quick_change_from_dict,
    to_serializable,
    vehicle_quick_overrides_from_dict,
)

__all__ = [
    "MAX_QUICK_SCENARIOS_PER_SOURCE",
    "QUICK_SCENARIO_CONTRACT_VERSION",
    "QUICK_SCENARIO_IDENTITY_PREFIX",
    "DomainReadiness",
    "PseProvenance",
    "QuickScenario",
    "QuickVehicleReadiness",
    "ReferencePressureProvenance",
    "ScalarChange",
    "ScalarChangeMode",
    "TirePressureDelta",
    "TireQuickChange",
    "TireSource",
    "TireTransformMode",
    "VehicleQuickOverrides",
    "build_quick_scenario_identity",
    "quick_scenario_from_dict",
    "quick_vehicle_readiness_from_dict",
    "scalar_change_from_dict",
    "tire_pressure_delta_from_dict",
    "tire_quick_change_from_dict",
    "to_serializable",
    "vehicle_quick_overrides_from_dict",
]
