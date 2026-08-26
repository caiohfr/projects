"""
Canonical, Streamlit-free contracts and resolver for Interactive Quick
Scenarios (Sprint 10A contracts + Sprint 10B Mass/Aero resolution).

    Existing resolved Comparison scenario (source_identity)
        -> QuickScenario (Vehicle overrides + Final PSE assumption)
        -> resolve_quick_vehicle_scenario() (Mass + Aero; Tire deferred)
        -> QuickVehicleResolution (resolved state + VehicleDemandRequest/Result)
        -> Temporary Comparison Result (never persisted)

`contracts.py`/`serialization.py` (10A) define data shape only and stay
Streamlit- and vehicle_demand-free. `resolution.py`/`resolver.py` (10B) are
where this package first depends on the frozen Sprint 9 Vehicle Demand Core
(src/vde_core/vehicle_demand/) -- expected, since consuming that frozen core
is the entire point of resolving a QuickScenario. No file in this package
imports Streamlit or any Comparison UI module. See
docs/sprints/SPRINT_10A_QUICK_SCENARIO_CONTRACT_AUDIT.md and
docs/sprints/SPRINT_10B_QUICK_MASS_AERO_RESOLUTION.md for the reuse audits
this package was designed against.
"""

from .contracts import (
    MAX_QUICK_SCENARIOS_PER_SOURCE,
    QUICK_SCENARIO_CONTRACT_VERSION,
    QUICK_SCENARIO_IDENTITY_PREFIX,
    DomainReadiness,
    MassQuickChange,
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
from .resolution import QuickVehicleResolution
from .resolver import resolve_quick_vehicle_scenario
from .serialization import (
    mass_quick_change_from_dict,
    quick_scenario_from_dict,
    quick_vehicle_readiness_from_dict,
    scalar_change_from_dict,
    tire_pressure_delta_from_dict,
    tire_quick_change_from_dict,
    to_serializable,
    vehicle_quick_overrides_from_dict,
)

__all__ = [
    "QuickVehicleResolution",
    "resolve_quick_vehicle_scenario",
    "MAX_QUICK_SCENARIOS_PER_SOURCE",
    "QUICK_SCENARIO_CONTRACT_VERSION",
    "QUICK_SCENARIO_IDENTITY_PREFIX",
    "DomainReadiness",
    "MassQuickChange",
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
    "mass_quick_change_from_dict",
    "quick_scenario_from_dict",
    "quick_vehicle_readiness_from_dict",
    "scalar_change_from_dict",
    "tire_pressure_delta_from_dict",
    "tire_quick_change_from_dict",
    "to_serializable",
    "vehicle_quick_overrides_from_dict",
]
