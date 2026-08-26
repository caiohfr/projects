"""
Canonical, Streamlit-free contracts and resolver for Interactive Quick
Scenarios (Sprint 10A contracts + Sprint 10B/10C Vehicle resolution).

    Existing resolved Comparison scenario (source_identity)
        -> QuickScenario (Vehicle overrides + Final PSE assumption)
        -> resolve_quick_vehicle_scenario() (Mass -> Tire -> Aero)
        -> QuickVehicleResolution (resolved state + VehicleDemandRequest/Result)
        -> Temporary Comparison Result (never persisted)

`contracts.py`/`serialization.py` (10A) define data shape only and stay
Streamlit- and vehicle_demand-free. `resolution.py`/`resolver.py` (10B/10C) are
where this package first depends on the frozen Sprint 9 Vehicle Demand Core
(src/vde_core/vehicle_demand/) -- expected, since consuming that frozen core
is the entire point of resolving a QuickScenario. No file in this package
imports Streamlit or any Comparison UI module. See
docs/sprints/SPRINT_10A_QUICK_SCENARIO_CONTRACT_AUDIT.md and
docs/sprints/SPRINT_10B_QUICK_MASS_AERO_RESOLUTION.md plus
docs/sprints/SPRINT_10C_QUICK_TIRE_RESOLUTION.md for the reuse audits
this package was designed against.
"""

from .contracts import (
    MAX_QUICK_SCENARIOS_PER_SOURCE,
    MAX_TECH_DELTAS_PER_SCENARIO,
    QUICK_SCENARIO_CONTRACT_VERSION,
    QUICK_SCENARIO_IDENTITY_PREFIX,
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
    build_quick_scenario_identity,
)
from .efficiency_resolution import (
    MlPseRecommendation,
    PseReference,
    QuickEfficiencyResolution,
    TechDeltaSuggestion,
)
from .efficiency_resolver import resolve_quick_efficiency_scenario
from .resolution import QuickVehicleResolution
from .resolver import resolve_quick_vehicle_scenario
from .tech_delta_catalog import (
    DEFAULT_QUICK_TECH_DELTA_CATALOG_PATH,
    load_quick_tech_delta_catalog,
)
from .serialization import (
    efficiency_quick_inputs_from_dict,
    mass_quick_change_from_dict,
    quick_scenario_from_dict,
    quick_vehicle_readiness_from_dict,
    scalar_change_from_dict,
    tech_delta_assumption_from_dict,
    tire_pressure_delta_from_dict,
    tire_quick_change_from_dict,
    to_serializable,
    vehicle_quick_overrides_from_dict,
)

__all__ = [
    "QuickVehicleResolution",
    "resolve_quick_vehicle_scenario",
    "QuickEfficiencyResolution",
    "PseReference",
    "MlPseRecommendation",
    "TechDeltaSuggestion",
    "resolve_quick_efficiency_scenario",
    "MAX_QUICK_SCENARIOS_PER_SOURCE",
    "MAX_TECH_DELTAS_PER_SCENARIO",
    "QUICK_SCENARIO_CONTRACT_VERSION",
    "QUICK_SCENARIO_IDENTITY_PREFIX",
    "DomainReadiness",
    "EfficiencyQuickInputs",
    "MassQuickChange",
    "PseProvenance",
    "QuickScenario",
    "QuickVehicleReadiness",
    "ReferencePressureProvenance",
    "ScalarChange",
    "ScalarChangeMode",
    "TechDeltaAssumption",
    "TirePressureDelta",
    "TireQuickChange",
    "TireSource",
    "TireTransformMode",
    "VehicleQuickOverrides",
    "DEFAULT_QUICK_TECH_DELTA_CATALOG_PATH",
    "load_quick_tech_delta_catalog",
    "build_quick_scenario_identity",
    "efficiency_quick_inputs_from_dict",
    "mass_quick_change_from_dict",
    "quick_scenario_from_dict",
    "quick_vehicle_readiness_from_dict",
    "scalar_change_from_dict",
    "tech_delta_assumption_from_dict",
    "tire_pressure_delta_from_dict",
    "tire_quick_change_from_dict",
    "to_serializable",
    "vehicle_quick_overrides_from_dict",
]
