"""
Canonical Vehicle Demand contracts (Sprint 9A).

This package defines the typed, Streamlit-free boundary between resolved
VDE/roadload scenarios and the future Vehicle Demand Engine (Sprint 9B+):

    Resolved VDE Scenario + Cycle + AmbientState
        -> VehicleDemandRequest
        -> [future Vehicle Demand Engine]
        -> VehicleDemandProfile / VehicleDemandSummary / VehicleDemandResult

No physics is implemented here. TOTAL/NET values carried by these contracts
are always the pre-existing authoritative values (see
src/vde_core/vde_net_total_contract.py); this package never rebuilds them
from components.
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
    "AmbientState",
    "EnergyMode",
    "Provenance",
    "RoadloadBasis",
    "RoadloadCoefficients",
    "VehicleDemandProfile",
    "VehicleDemandRequest",
    "VehicleDemandResult",
    "VehicleDemandSummary",
    "to_serializable",
    "ambient_state_from_dict",
    "roadload_coefficients_from_dict",
    "vehicle_demand_request_from_dict",
    "vehicle_demand_profile_from_dict",
    "vehicle_demand_summary_from_dict",
    "vehicle_demand_result_from_dict",
]
