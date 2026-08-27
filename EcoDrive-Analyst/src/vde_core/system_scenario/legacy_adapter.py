# src/vde_core/system_scenario/legacy_adapter.py
# -----------------------------------------------------------------------------
# Sprint 11A - the minimum legacy-boundary adapter required to prove that
# current vde_db/fuelcons_db data can populate canonical System Scenario
# Domain States without the canonical contracts themselves depending on raw
# row layout (spec Sec 30/Sec "Legacy Adapter Boundary", INV-11-012).
#
#     legacy vde_db / fuelcons_db row
#         -> this module
#         -> canonical DomainSourceState (contracts.py)
#
# This is a PROOF of the seam, not a migration of the whole Powertrain page
# (Sec 8: "Do not perform broad migration of all current page behavior in
# 11A"). It covers a representative subset of domains for which the current
# schema already has real columns (confirmed by the Sprint 11A audit):
# Vehicle Demand, Architecture (from `electrification`), Engine, and
# Transmission. Electric Drive / Energy Storage / Energy Management-Controls
# / Aux-Thermal have no confirmed legacy columns today (motor/battery-
# specific fields are not present in vde_db/fuelcons_db) -- their adapters
# are deferred rather than populated from guessed field names ("missing
# future fields remain missing", spec Sec 4). No DB fields are added and no
# schema is migrated here.
#
# The Vehicle Demand adapter reuses the EXACT existing frozen Sprint 9 call
# chain Quick Scenario's own resolver already uses
# (build_vehicle_demand_request / resolve_vehicle_demand_cycle /
# calculate_vehicle_demand, src/vde_core/quick_scenario/resolver.py:53-57) --
# never a new roadload/VDE computation (INV-11-006).
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Mapping

from src.vde_core.vehicle_demand import VehicleDemandResult, calculate_vehicle_demand
from src.vde_core.vehicle_demand.adapters import (
    build_vehicle_demand_request,
    resolve_vehicle_demand_cycle,
)

from .contracts import (
    ArchitectureClass,
    ArchitectureConfiguration,
    DomainKind,
    DomainSourceState,
    EngineConfiguration,
    ProvenanceKind,
    TransmissionConfiguration,
    VehicleDemandConfiguration,
)

# Sec 6/REQ-11-005: legacy `electrification` values already used throughout
# fuel_estimation.py/pwt_fuel_energy_service.py (confirmed by the Sprint 11A
# audit) mapped onto the 5-value Architecture classification. "HEV"/"MHEV"
# are not distinct legacy electrification values in the audited columns
# today -- both collapse from the same source label where the legacy schema
# does not distinguish them; this is a classification mapping, not a new
# assumption about the vehicle's actual hardware.
_ARCHITECTURE_CLASS_BY_LEGACY_ELECTRIFICATION: Mapping[str, ArchitectureClass] = {
    "ICE": ArchitectureClass.ICE,
    "MHEV": ArchitectureClass.MHEV,
    "HEV": ArchitectureClass.HEV,
    "PHEV": ArchitectureClass.PHEV,
    "BEV": ArchitectureClass.BEV,
}


def vehicle_demand_domain_state_from_result(
    result: VehicleDemandResult, *, source_identity: str | None = None
) -> DomainSourceState:
    """Wrap an ALREADY-RESOLVED frozen Sprint 9 `VehicleDemandResult` into a
    canonical Vehicle Demand `DomainSourceState`. Never re-derives roadload/
    VDE physics (INV-11-006) -- this is a pure wrap, for callers (Quick
    Scenario, Comparison, a future System Scenario resolver) that already
    hold a computed result.
    """

    return DomainSourceState(
        domain=DomainKind.VEHICLE_DEMAND,
        configuration=VehicleDemandConfiguration(
            source_identity=source_identity, vehicle_demand_result=result
        ),
        provenance=ProvenanceKind.CALCULATED,
    )


def vehicle_demand_domain_state_from_legacy_vde_row(
    vde_row: Mapping[str, Any], *, source_identity: str | None = None
) -> DomainSourceState:
    """The fuller legacy-boundary proof for Vehicle Demand: given a raw
    `vde_db`-shaped row, reuse the EXISTING frozen Sprint 9 Vehicle Demand
    Core call chain (the same 3 functions Quick Scenario's own resolver
    already calls, `quick_scenario/resolver.py:53-57`) to produce a real
    `VehicleDemandResult`, then wrap it exactly as
    `vehicle_demand_domain_state_from_result` does. Returns a
    DomainSourceState with `vehicle_demand_result=None` (not an error) when
    this row's legislation has no standard cycle trace available --
    consistent with how the frozen core itself reports that condition
    elsewhere (never silently guesses a result).
    """

    request = build_vehicle_demand_request(dict(vde_row))
    cycle_frame = resolve_vehicle_demand_cycle(dict(vde_row))
    result = calculate_vehicle_demand(request, cycle_frame) if cycle_frame is not None else None
    return DomainSourceState(
        domain=DomainKind.VEHICLE_DEMAND,
        configuration=VehicleDemandConfiguration(
            source_identity=source_identity, vehicle_demand_result=result
        ),
        provenance=ProvenanceKind.CALCULATED,
    )


def architecture_domain_state_from_legacy_vde_row(
    vde_row: Mapping[str, Any], fuelcons_row: Mapping[str, Any] | None = None
) -> DomainSourceState:
    """Sec 6/Case L: classification only, from the legacy `electrification`
    column already read throughout the audited Powertrain path
    (`fuel_estimation.py:185`, `_POWERTRAIN_FUELCONS_FIELDS` in
    `comparison_report_service.py`). Confirmed directly against the live QA
    fixture during this sprint: `electrification` is a `fuelcons_db` column,
    not a `vde_db` one -- `vde_row` alone never carries it, so this adapter
    accepts an optional `fuelcons_row` and reads it from there (matching the
    Engine/Transmission adapters' own fuelcons-preferred pattern below). An
    unrecognized/missing value maps to `architecture_class=None` -- never a
    guessed default -- with the raw legacy text preserved in
    `topology_notes` for audit.
    """

    fuelcons_row = fuelcons_row or {}
    raw = fuelcons_row.get("electrification")
    text = str(raw).strip().upper() if raw not in (None, "") else None
    architecture_class = _ARCHITECTURE_CLASS_BY_LEGACY_ELECTRIFICATION.get(text) if text else None
    return DomainSourceState(
        domain=DomainKind.ARCHITECTURE,
        configuration=ArchitectureConfiguration(
            architecture_class=architecture_class,
            topology_notes=f"legacy electrification={raw!r}" if raw is not None else None,
        ),
        provenance=ProvenanceKind.SOURCE_OBSERVED if architecture_class is not None else ProvenanceKind.NOT_AVAILABLE,
    )


def engine_domain_state_from_legacy_row(
    vde_row: Mapping[str, Any], fuelcons_row: Mapping[str, Any] | None = None
) -> DomainSourceState:
    """Populates `EngineConfiguration` from the legacy columns the Sprint
    11A audit confirmed are actually read today: `engine_size_l`
    (`vde_row`, consumed by `build_fuel_estimate_request_from_vde`) and
    `fuel_type`/`engine_max_power_kw` (`fuelcons_row`, when available).
    Fields with no confirmed legacy source (engine family id, rated torque,
    technology descriptors) stay `None`/empty -- never guessed.
    """

    fuelcons_row = fuelcons_row or {}
    return DomainSourceState(
        domain=DomainKind.ENGINE_FUEL_CONVERTER,
        configuration=EngineConfiguration(
            fuel_type=fuelcons_row.get("fuel_type"),
            displacement_l=vde_row.get("engine_size_l"),
            rated_power_kw=fuelcons_row.get("engine_max_power_kw"),
        ),
        provenance=ProvenanceKind.SOURCE_OBSERVED,
    )


def transmission_domain_state_from_legacy_row(
    vde_row: Mapping[str, Any], fuelcons_row: Mapping[str, Any] | None = None
) -> DomainSourceState:
    """Populates `TransmissionConfiguration` from the legacy columns
    confirmed by the Sprint 11A audit: `transmission_type` (`vde_row`),
    `gear_count`/`final_drive_ratio` (present on both `vde_row` and
    `fuelcons_row` in the audited code -- `fuelcons_row` preferred when both
    are supplied, since it is the more recently-edited scenario-level
    value; falls back to `vde_row`)."""

    fuelcons_row = fuelcons_row or {}
    return DomainSourceState(
        domain=DomainKind.TRANSMISSION_DRIVELINE,
        configuration=TransmissionConfiguration(
            transmission_type=vde_row.get("transmission_type"),
            gear_count=fuelcons_row.get("gear_count") if fuelcons_row.get("gear_count") is not None else vde_row.get("gear_count"),
            final_drive_ratio=(
                fuelcons_row.get("final_drive_ratio")
                if fuelcons_row.get("final_drive_ratio") is not None
                else vde_row.get("final_drive_ratio")
            ),
        ),
        provenance=ProvenanceKind.SOURCE_OBSERVED,
    )


__all__ = [
    "vehicle_demand_domain_state_from_result",
    "vehicle_demand_domain_state_from_legacy_vde_row",
    "architecture_domain_state_from_legacy_vde_row",
    "engine_domain_state_from_legacy_row",
    "transmission_domain_state_from_legacy_row",
]
