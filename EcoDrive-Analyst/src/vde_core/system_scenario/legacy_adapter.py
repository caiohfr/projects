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
# Sprint 11A proved the seam for 4 domains (Vehicle Demand, Architecture,
# Engine, Transmission). Sprint 11B expanded it after a direct PRAGMA
# table_info() query against the live schema (not assumption/memory)
# confirmed real columns exist for 3 more: Energy Storage
# (`battery_capacity_kwh`/`battery_usable_kwh`/`bms_discharge_limit_kw`/
# `bms_regen_limit_kw`), Energy Management/Controls (`utility_factor_pct`
# -- confirmed persisted, correcting an 11A assumption that it was only a
# runtime request parameter), and Aux/Thermal (`ambient_temp_c`/`ac_on` --
# correcting 11A's claim that this domain had no confirmed columns; two
# fields were added to `AuxThermalConfiguration` in 11B specifically
# because this real data was found). Electric Drive remains the one domain
# with genuinely NO configuration-level legacy column (only `bev_eff_drive`,
# an L0 efficiency ASSUMPTION, not motor configuration) --
# `electric_drive_domain_state_sparse()` represents it explicitly as valid,
# all-missing state rather than leaving it unrepresented. This is still a
# PROOF of the seam plus a fuller legacy-boundary pass, not a migration of
# the whole Powertrain page. No DB fields are added and no schema is
# migrated here.
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
    AuxThermalConfiguration,
    ControlsConfiguration,
    DomainKind,
    DomainSourceState,
    ElectricDriveConfiguration,
    EnergyStorageConfiguration,
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
    11A audit confirmed are actually read today: `engine_size_l`/
    `engine_model`/`engine_type`/`engine_aspiration` (`vde_row`) and
    `fuel_type`/`engine_max_power_kw`/`engine_max_torque_nm` (`fuelcons_row`,
    when available). Expanded in Sprint 11B after a direct PRAGMA
    table_info() query against the live schema confirmed
    `fuelcons_db.engine_max_torque_nm` and `vde_db.engine_model`/
    `engine_type`/`engine_aspiration` are real, populated columns --
    `engine_family_id` maps to `engine_model` (the closest existing
    identifier-like column; there is no separate "family" concept in the
    schema), and `technology_descriptors` collects `engine_type`/
    `engine_aspiration` as free-text labels, never a controlled vocabulary
    invented for this sprint.
    """

    fuelcons_row = fuelcons_row or {}
    descriptors = tuple(
        str(value) for value in (vde_row.get("engine_type"), vde_row.get("engine_aspiration")) if value
    )
    return DomainSourceState(
        domain=DomainKind.ENGINE_FUEL_CONVERTER,
        configuration=EngineConfiguration(
            fuel_type=fuelcons_row.get("fuel_type"),
            engine_family_id=vde_row.get("engine_model"),
            displacement_l=vde_row.get("engine_size_l"),
            rated_power_kw=fuelcons_row.get("engine_max_power_kw"),
            rated_torque_nm=fuelcons_row.get("engine_max_torque_nm"),
            technology_descriptors=descriptors,
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
    value; falls back to `vde_row`). `transmission_model_id` (added in
    Sprint 11B) maps to `vde_db.transmission_model`, confirmed present by a
    direct schema query -- missed in the 11A adapter, added here."""

    fuelcons_row = fuelcons_row or {}
    return DomainSourceState(
        domain=DomainKind.TRANSMISSION_DRIVELINE,
        configuration=TransmissionConfiguration(
            transmission_type=vde_row.get("transmission_type"),
            transmission_model_id=vde_row.get("transmission_model"),
            gear_count=fuelcons_row.get("gear_count") if fuelcons_row.get("gear_count") is not None else vde_row.get("gear_count"),
            final_drive_ratio=(
                fuelcons_row.get("final_drive_ratio")
                if fuelcons_row.get("final_drive_ratio") is not None
                else vde_row.get("final_drive_ratio")
            ),
        ),
        provenance=ProvenanceKind.SOURCE_OBSERVED,
    )


def electric_drive_domain_state_sparse() -> DomainSourceState:
    """Sprint 11B Sec 12: a direct schema query (PRAGMA table_info against
    the live QA-seeded database) confirmed neither `vde_db` nor
    `fuelcons_db` has ANY motor-role/count/position/rated-power/rated-
    torque/voltage/identifier column today -- `bev_eff_drive` is the only
    EV-related column, and it is an L0 efficiency ASSUMPTION (Sec 12: "L0
    model assumptions"), not motor CONFIGURATION, so it is deliberately not
    placed into `ElectricDriveConfiguration` here (Sec 6: never put an
    assumption into physical configuration merely because a legacy row
    contains it). Electric Drive therefore has no real legacy adapter
    today -- this function exists so the domain is still explicitly
    representable as a valid, all-missing `DomainSourceState` (Sec 15/24:
    sparse domain data is valid; missing stays missing) rather than simply
    absent from a caller's domain map.
    """

    return DomainSourceState(
        domain=DomainKind.ELECTRIC_DRIVE,
        configuration=ElectricDriveConfiguration(),
        provenance=ProvenanceKind.NOT_AVAILABLE,
        notes="No motor/inverter configuration columns exist in the current schema.",
    )


def energy_storage_domain_state_from_legacy_row(fuelcons_row: Mapping[str, Any] | None) -> DomainSourceState:
    """Sprint 11B: populates `EnergyStorageConfiguration` from
    `fuelcons_db.battery_capacity_kwh`/`battery_usable_kwh`/
    `bms_discharge_limit_kw`/`bms_regen_limit_kw` -- confirmed real,
    populated columns by a direct schema query. There is no separate
    `charge_power_limit_kw` or nominal-voltage/SOC-window column in the
    current schema, so those fields stay `None` -- never guessed.
    `bms_note`, when present, is carried as this DomainSourceState's own
    top-level `notes` (no new configuration field needed for a single
    free-text note)."""

    fuelcons_row = fuelcons_row or {}
    return DomainSourceState(
        domain=DomainKind.ENERGY_STORAGE,
        configuration=EnergyStorageConfiguration(
            gross_capacity_kwh=fuelcons_row.get("battery_capacity_kwh"),
            usable_capacity_kwh=fuelcons_row.get("battery_usable_kwh"),
            discharge_power_limit_kw=fuelcons_row.get("bms_discharge_limit_kw"),
            regen_power_limit_kw=fuelcons_row.get("bms_regen_limit_kw"),
        ),
        provenance=ProvenanceKind.SOURCE_OBSERVED,
        notes=str(fuelcons_row.get("bms_note") or ""),
    )


def controls_domain_state_from_legacy_row(fuelcons_row: Mapping[str, Any] | None) -> DomainSourceState:
    """Sprint 11B: populates `ControlsConfiguration.utility_factor_pct`
    from `fuelcons_db.utility_factor_pct` -- confirmed a real, stored
    column by a direct schema query (corrects an 11A assumption that
    utility factor was only ever a runtime request parameter; it is also
    persisted). `hybrid_operating_strategy`/`regen_metadata`/
    `start_stop_enabled`/`calibration_notes` have no confirmed legacy
    column and stay `None` -- never guessed."""

    fuelcons_row = fuelcons_row or {}
    return DomainSourceState(
        domain=DomainKind.ENERGY_MANAGEMENT_CONTROLS,
        configuration=ControlsConfiguration(utility_factor_pct=fuelcons_row.get("utility_factor_pct")),
        provenance=ProvenanceKind.SOURCE_OBSERVED if fuelcons_row.get("utility_factor_pct") is not None else ProvenanceKind.NOT_AVAILABLE,
    )


def aux_thermal_domain_state_from_legacy_row(fuelcons_row: Mapping[str, Any] | None) -> DomainSourceState:
    """Sprint 11B: populates `AuxThermalConfiguration.ambient_temp_c`/
    `ac_on` from `fuelcons_db.ambient_temp_c`/`fuelcons_db.ac_on` --
    confirmed real, populated columns by a direct schema query (this
    corrects the 11A closure doc's claim that Aux/Thermal had no confirmed
    legacy columns; it does, just none were checked for at the time). Both
    fields were added to `AuxThermalConfiguration` in Sprint 11B
    specifically because this real data was found -- not invented to make
    the domain look complete."""

    fuelcons_row = fuelcons_row or {}
    ac_on_raw = fuelcons_row.get("ac_on")
    return DomainSourceState(
        domain=DomainKind.AUX_THERMAL,
        configuration=AuxThermalConfiguration(
            ambient_temp_c=fuelcons_row.get("ambient_temp_c"),
            ac_on=bool(ac_on_raw) if ac_on_raw is not None else None,
        ),
        provenance=ProvenanceKind.SOURCE_OBSERVED,
    )


__all__ = [
    "vehicle_demand_domain_state_from_result",
    "vehicle_demand_domain_state_from_legacy_vde_row",
    "architecture_domain_state_from_legacy_vde_row",
    "engine_domain_state_from_legacy_row",
    "transmission_domain_state_from_legacy_row",
    "electric_drive_domain_state_sparse",
    "energy_storage_domain_state_from_legacy_row",
    "controls_domain_state_from_legacy_row",
    "aux_thermal_domain_state_from_legacy_row",
]
