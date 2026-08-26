# src/vde_core/quick_scenario/resolution.py
# -----------------------------------------------------------------------------
# Sprint 10B - the Quick Vehicle Scenario *output* contract.
#
# Unlike contracts.py (Sprint 10A, deliberately Streamlit- and
# vehicle_demand-free), this module depends on the frozen Sprint 9 Vehicle
# Demand Core (src/vde_core/vehicle_demand/) because that dependency is the
# entire point of resolving a QuickScenario: a resolved Mass/Aero state is
# only useful once expressed as the same VehicleDemandRequest/
# VehicleDemandResult every other consumer of the frozen core uses. This
# module still defines data shape only -- resolver.py owns the actual
# physics call chain.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

from src.vde_core.vehicle_demand import (
    RoadloadCoefficients,
    VehicleDemandRequest,
    VehicleDemandResult,
)

from .contracts import QuickVehicleReadiness


@dataclass(frozen=True)
class QuickVehicleResolution:
    """The resolved physical outcome of one QuickScenario's Vehicle Quick
    layer (Mass + Aero, Sprint 10B; Tire deferred to a later package).

    `vehicle_demand_request`/`vehicle_demand_result` and every resolved
    physical field stay None whenever `not readiness.all_ready` -- a
    scenario with an unresolved requested domain never produces a silently
    partial calculation (Sec 18). `abc_total`/`abc_net`/
    `vde_total_mj_per_km`/`vde_net_mj_per_km` come from the same
    already-canonical `resolve_roadload_boundaries`/`resolve_cycle_vde_results`
    path Comparison itself uses, independent of (but expected to reconcile
    with) `vehicle_demand_result`'s own TOTAL/NET summaries -- exposing both
    is deliberate cross-path auditability, not redundancy for its own sake.
    """

    quick_scenario_identity: str
    readiness: QuickVehicleReadiness
    issues: tuple[str, ...] = ()

    resolved_curb_mass_kg: float | None = None
    resolved_vde_calculation_mass_kg: float | None = None
    resolved_vde_mass_basis: str | None = None

    resolved_cda_m2: float | None = None

    abc_total: RoadloadCoefficients | None = None
    abc_net: RoadloadCoefficients | None = None
    vde_total_mj_per_km: float | None = None
    vde_net_mj_per_km: float | None = None

    vehicle_demand_request: VehicleDemandRequest | None = None
    vehicle_demand_result: VehicleDemandResult | None = None

    @property
    def is_ready(self) -> bool:
        return self.readiness.all_ready


__all__ = ["QuickVehicleResolution"]
