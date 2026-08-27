# src/vde_core/quick_scenario/comparison_adapter.py
# -----------------------------------------------------------------------------
# Sprint 10E - turns a resolved QuickScenario into an ordinary ComparisonItem
# that the existing Comparison Report page (Program Review / Energy Drivers /
# Technical Scorecard / Explore) can render with zero code changes of its
# own. No physics lives here: every numeric value already comes from
# QuickVehicleResolution/QuickEfficiencyResolution (Sprint 10B-10D), which
# themselves only ever call the existing canonical resolvers. This module's
# only job is data marshalling (row/dict shaping) plus per-slot
# orchestration -- it hands the shaped row to the EXISTING
# build_vde_comparison_item/build_scenario_comparison_item builders
# (comparison_report_service.py) rather than constructing a ComparisonItem
# by hand, so Quick items go through the identical builder every real
# Comparison item goes through (no second Comparison engine).
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any, Mapping, Sequence

from src.vde_core.comparison_report_service import (
    ComparisonDataset,
    ComparisonItem,
    ComparisonRole,
    build_scenario_comparison_item,
    build_vde_comparison_item,
)

from .contracts import QuickScenario
from .efficiency_resolution import QuickEfficiencyResolution
from .resolution import QuickVehicleResolution
from .resolver import _parse_source_identity, fetch_quick_source_rows, resolve_quick_vehicle_scenario

try:  # Sprint 10D efficiency resolver -- optional import keeps this module
    # usable for Vehicle-only Quick Scenarios without a hard dependency on
    # fuel_estimation's heavier import chain.
    from .efficiency_resolver import resolve_quick_efficiency_scenario
except ImportError:  # pragma: no cover - defensive only, never expected
    resolve_quick_efficiency_scenario = None  # type: ignore[assignment]


class QuickSlotCalculationState(str, Enum):
    """Sec (surviving summary): a simple, non-physics UI-state enum tracking
    whether a Quick Scenario slot's last calculation is still valid against
    its current (possibly since-edited) inputs.
    """

    NOT_CALCULATED = "NOT_CALCULATED"
    READY = "READY"
    NEEDS_RECALCULATION = "NEEDS_RECALCULATION"
    MISSING_OR_INVALID = "MISSING_OR_INVALID"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


_QUICK_RECORD_ORIGIN = "QUICK_SCENARIO"


_SOURCE_KIND_SENTINEL_CODE = {"fc": 1, "vde": 2}


def quick_slot_sentinel_id(source_identity: str, slot: int) -> int:
    """A deterministic negative id, unique per (source_identity, slot), that
    never collides with a real (positive) vde_db/fuelcons_db primary key.
    Stable across reruns so the same Quick slot always maps to the same
    sentinel identity within one session.

    Keyed off the full `source_identity` (kind + record id) rather than the
    resolved vde_id: two distinct fuelcons_db scenarios sharing one vde_id
    (the established "fc:900102"/"fc:900104" -> vde_id=900001 QA fixture
    pattern used throughout Sprints 10A-10D) each get their OWN up-to-3
    Quick slots (MAX_QUICK_SCENARIOS_PER_SOURCE is scoped per source, not
    per vde_id) -- keying by vde_id alone would collide their same-numbered
    slots into one sentinel id, silently conflating two distinct Quick
    Scenarios. Found by the Sprint 10E closure traceability audit.
    """

    kind, record_id = _parse_source_identity(source_identity)
    kind_code = _SOURCE_KIND_SENTINEL_CODE[kind]
    return -(kind_code * 10**15 + abs(int(record_id)) * 10 + int(slot))


def fetch_quick_source_rows_once(
    source_identity: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Thin re-export of resolver.fetch_quick_source_rows for callers that
    only import this adapter module.
    """

    return fetch_quick_source_rows(source_identity)


def resolve_quick_slot(
    quick_scenario: QuickScenario,
    *,
    source_vde_row: Mapping[str, Any],
    source_fuelcons_row: Mapping[str, Any] | None,
) -> tuple[QuickVehicleResolution, QuickEfficiencyResolution | None]:
    """Resolve one Quick Scenario slot's Vehicle layer, then (only if
    Vehicle succeeded) its Efficiency layer. Never raises for a
    MISSING/INVALID domain -- both canonical resolvers already return
    structured results, so a bad slot cannot interrupt its siblings; the
    caller resolving several slots for one source simply calls this once
    per slot in a plain loop.
    """

    vehicle_resolution = resolve_quick_vehicle_scenario(quick_scenario, source_vde_row=source_vde_row)
    if not vehicle_resolution.is_ready or resolve_quick_efficiency_scenario is None:
        return vehicle_resolution, None

    efficiency_resolution = resolve_quick_efficiency_scenario(
        quick_scenario, vehicle_resolution, source_fuelcons_row=source_fuelcons_row
    )
    return vehicle_resolution, efficiency_resolution


_FUEL_ENERGY_RENAME = {
    "fuel_l_100km": "fuel_l_per_100km",
    "energy_Wh_km": "energy_Wh_per_km",
    "gco2_km": "gco2_per_km",
}


def _quick_source_label(quick_scenario: QuickScenario) -> str:
    base = f"Quick #{quick_scenario.slot} of {quick_scenario.source_identity}"
    return f"{base} ({quick_scenario.label})" if quick_scenario.label else base


def _stamped_vde_row(
    vehicle_resolution: QuickVehicleResolution, sentinel_id: int
) -> dict[str, Any]:
    row = dict(vehicle_resolution.resolved_vde_row or {})
    row["id"] = sentinel_id
    row["record_origin"] = _QUICK_RECORD_ORIGIN
    return row


def _synthetic_fuelcons_row(
    quick_scenario: QuickScenario,
    efficiency_resolution: QuickEfficiencyResolution,
    sentinel_id: int,
    linked_vde_sentinel_id: int,
    source_fuelcons_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = efficiency_resolution.fuel_estimate_result
    assert result is not None  # only called when efficiency_resolution.is_ready

    row: dict[str, Any] = {
        "id": sentinel_id,
        "vde_id": linked_vde_sentinel_id,
        "record_origin": _QUICK_RECORD_ORIGIN,
        "engine_method": result.method,
        "engine_version": None,
        # No saved-revision concept applies to a temporary Quick item --
        # explicit None so _resolve_revision reports MISSING, never a
        # spurious STALE comparison against the real source's VDE.
        "source_vde_revision": None,
    }
    for result_field, row_field in _FUEL_ENERGY_RENAME.items():
        row[row_field] = getattr(result, result_field)

    source = dict(source_fuelcons_row or {})
    row["electrification"] = source.get("electrification")
    row["fuel_type"] = source.get("fuel_type") or result.assumptions.get("fuel_type")
    for passthrough in ("gear_count", "final_drive_ratio", "engine_max_power_kw", "battery_capacity_kwh"):
        row[passthrough] = source.get(passthrough)
    row["eta_pt_est"] = result.assumptions.get("eta_pt_est")
    return row


def build_quick_comparison_item(
    quick_scenario: QuickScenario,
    vehicle_resolution: QuickVehicleResolution,
    efficiency_resolution: QuickEfficiencyResolution | None,
    *,
    source_fuelcons_row: Mapping[str, Any] | None = None,
    role: ComparisonRole = ComparisonRole.COMPARISON,
) -> ComparisonItem | None:
    """Convert one resolved Quick Scenario slot into a ComparisonItem, or
    `None` when its Vehicle layer never resolved (Sec: a Vehicle-unresolved
    Quick Scenario is never inserted into the Comparison dataset at all --
    "no silent partial calc" is already enforced one layer down by
    QuickVehicleResolution itself; this function just declines to insert
    what that layer already marked unresolved).
    """

    if not vehicle_resolution.is_ready or vehicle_resolution.resolved_vde_row is None:
        return None

    sentinel_vde_id = quick_slot_sentinel_id(quick_scenario.source_identity, quick_scenario.slot)
    stamped_row = _stamped_vde_row(vehicle_resolution, sentinel_vde_id)

    label = _quick_source_label(quick_scenario)

    if efficiency_resolution is not None and efficiency_resolution.is_ready:
        # Same sentinel integer as the vde_row's id, stored in a different
        # ComparisonItem field (fuelcons_id vs vde_id) -- downstream
        # consumers (e.g. deduplicate_by_vde_id) group by vde_id only, so
        # there is no cross-field collision to avoid here.
        sentinel_fc_id = sentinel_vde_id
        synthetic_fuelcons_row = _synthetic_fuelcons_row(
            quick_scenario,
            efficiency_resolution,
            sentinel_fc_id,
            sentinel_vde_id,
            source_fuelcons_row,
        )
        item = build_scenario_comparison_item(
            sentinel_fc_id,
            role=role,
            fuelcons_row=synthetic_fuelcons_row,
            vde_row=stamped_row,
        )
    else:
        item = build_vde_comparison_item(sentinel_vde_id, role=role, vde_row=stamped_row)

    return _with_label(item, label)


def _with_label(item: ComparisonItem, label: str) -> ComparisonItem:
    return replace(item, label=label)


def merge_quick_items_into_dataset(
    dataset: ComparisonDataset, quick_items: Sequence[ComparisonItem]
) -> ComparisonDataset:
    """Append resolved Quick items to `dataset.comparisons`. `dataset.reference`
    and the existing `comparisons` tuple are never touched or reordered --
    Quick items are additive only (no Save/Promote in this sprint, so a
    Quick item can never become the Reference).
    """

    if not quick_items:
        return dataset

    all_items = (dataset.reference,) if dataset.reference is not None else ()
    all_items = all_items + dataset.comparisons + tuple(quick_items)
    warnings = tuple(dict.fromkeys(w for item in all_items for w in item.warnings))
    return replace(
        dataset,
        comparisons=dataset.comparisons + tuple(quick_items),
        warnings=warnings,
    )


def derive_quick_slot_calculation_state(
    current_scenario: QuickScenario,
    last_calculated_scenario: QuickScenario | None,
    vehicle_resolution: QuickVehicleResolution | None,
    efficiency_resolution: QuickEfficiencyResolution | None,
) -> QuickSlotCalculationState:
    """Pure, Streamlit-free UI-state derivation (no numeric/physics
    decisions): NOT_CALCULATED before the first calculation;
    NEEDS_RECALCULATION once the live inputs (a frozen dataclass, so `!=`
    is structural equality) diverge from the snapshot as of the last
    successful calculation; MISSING_OR_INVALID when Vehicle didn't resolve,
    or Efficiency was requested but isn't ready; READY otherwise.
    """

    if vehicle_resolution is None or last_calculated_scenario is None:
        return QuickSlotCalculationState.NOT_CALCULATED
    if current_scenario != last_calculated_scenario:
        return QuickSlotCalculationState.NEEDS_RECALCULATION
    if not vehicle_resolution.is_ready:
        return QuickSlotCalculationState.MISSING_OR_INVALID

    efficiency_requested = (
        not current_scenario.efficiency_inputs.is_empty or current_scenario.final_pse_percent is not None
    )
    if efficiency_requested:
        if efficiency_resolution is None or not efficiency_resolution.is_ready:
            return QuickSlotCalculationState.MISSING_OR_INVALID
    return QuickSlotCalculationState.READY


__all__ = [
    "QuickSlotCalculationState",
    "quick_slot_sentinel_id",
    "fetch_quick_source_rows_once",
    "resolve_quick_slot",
    "build_quick_comparison_item",
    "merge_quick_items_into_dataset",
    "derive_quick_slot_calculation_state",
]
