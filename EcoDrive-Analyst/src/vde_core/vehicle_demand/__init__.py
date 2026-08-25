"""
Canonical Vehicle Demand contracts and physics engine (Sprint 9A + 9B).

    Resolved VDE Scenario + Cycle + AmbientState
        -> VehicleDemandRequest
        -> calculate_vehicle_demand() / build_vehicle_demand_profile() / summarize_vehicle_demand()
        -> VehicleDemandProfile / VehicleDemandSummary / VehicleDemandResult

The engine (engine.py) reuses the project's existing canonical VDE math
(src/vde_core/vde_calc.py) rather than reimplementing it -- see engine.py's
module docstring. TOTAL/NET values carried by these contracts are always the
pre-existing authoritative values (see src/vde_core/vde_net_total_contract.py);
this package never rebuilds them from components.
"""

from .contracts import (
    VEHICLE_DEMAND_CONTRACT_VERSION,
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
from .engine import (
    VEHICLE_DEMAND_ENGINE_VERSION,
    build_vehicle_demand_profile,
    calculate_vehicle_demand,
    summarize_vehicle_demand,
)
from .serialization import (
    ambient_state_from_dict,
    roadload_coefficients_from_dict,
    to_serializable,
    vehicle_demand_profile_from_dict,
    vehicle_demand_request_from_dict,
    vehicle_demand_result_from_dict,
    vehicle_demand_summary_from_dict,
)

__all__ = [
    "VEHICLE_DEMAND_CONTRACT_VERSION",
    "VEHICLE_DEMAND_ENGINE_VERSION",
    "AmbientState",
    "EnergyMode",
    "Provenance",
    "RoadloadBasis",
    "RoadloadCoefficients",
    "VehicleDemandProfile",
    "VehicleDemandRequest",
    "VehicleDemandResult",
    "VehicleDemandSummary",
    "build_vehicle_demand_profile",
    "calculate_vehicle_demand",
    "summarize_vehicle_demand",
    "to_serializable",
    "ambient_state_from_dict",
    "roadload_coefficients_from_dict",
    "vehicle_demand_request_from_dict",
    "vehicle_demand_profile_from_dict",
    "vehicle_demand_summary_from_dict",
    "vehicle_demand_result_from_dict",
]
