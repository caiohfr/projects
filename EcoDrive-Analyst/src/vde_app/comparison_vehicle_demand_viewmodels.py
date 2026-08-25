# src/vde_app/comparison_vehicle_demand_viewmodels.py
# -----------------------------------------------------------------------------
# Sprint 9D - pure view-model layer bridging Comparison's canonical
# ComparisonItem/ComparisonDataset into the frozen Vehicle Demand Core
# (Sprint 9A-9C). No Streamlit import; no physics is implemented here --
# every quantity comes directly from a VehicleDemandResult produced by
# calculate_vehicle_demand().
#
# Dependency direction (Sprint 9D Sec 5): this module imports FROM
# src.vde_core.comparison_report_service (only the ComparisonItem/
# ComparisonDataset types) and FROM src.vde_core.vehicle_demand (contracts +
# engine + the Sprint 9C cycle-resolution adapter). It is never imported BY
# either of those modules -- Comparison depends on Vehicle Demand here,
# never the reverse.
#
# A ComparisonItem already carries pre-resolved TOTAL/NET roadload
# (item.roadload, built by comparison_report_service.resolve_roadload_
# boundaries) and mass/RRC/CdA/legislation (item.vehicle). This module maps
# that already-resolved shape directly into a VehicleDemandRequest, rather
# than routing through vehicle_demand.adapters.build_vehicle_demand_request
# (which expects a raw vde_db row with different column names and would
# require re-fetching the row a second time). The only piece reused from
# vehicle_demand.adapters is resolve_vehicle_demand_cycle, which only needs
# a "legislation" key -- item.vehicle already provides that directly.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from src.vde_core.comparison_report_service import ComparisonDataset, ComparisonItem
from src.vde_core.cycles import default_cycle_for_legislation
from src.vde_core.vehicle_demand import (
    AmbientState,
    Provenance,
    RoadloadBasis,
    RoadloadCoefficients,
    VehicleDemandRequest,
    VehicleDemandResult,
    VehicleDemandSummary,
    calculate_vehicle_demand,
)
from src.vde_core.vehicle_demand.adapters import resolve_vehicle_demand_cycle

from .comparison_report_viewmodels import ScorecardCell, ScorecardRow, ScorecardSection, _format_delta, dataset_items, format_value

VEHICLE_DEMAND_SECTION_TITLE = "Vehicle Demand Summary"

_UNAVAILABLE_REASON_TEXT = {
    "rolling": "RRC unavailable",
    "aero": "CdA/air density unavailable",
}


@dataclass(frozen=True)
class _VehicleDemandKpi:
    field_name: str
    label: str
    unit_family: str
    unavailable_reason: str | None = None


# Primary KPIs first (Sprint 9D Sec 13), explanatory drivers after -- one
# flat, ordered table rather than two separately-weighted visual tiers.
_VEHICLE_DEMAND_KPIS: tuple[_VehicleDemandKpi, ...] = (
    _VehicleDemandKpi("vde_mj_per_km", "VDE", "energy_mj_per_km"),
    _VehicleDemandKpi("roadload_energy_MJ", "Roadload Energy", "energy_mj"),
    _VehicleDemandKpi("positive_tractive_energy_MJ", "Positive Tractive Energy", "energy_mj"),
    _VehicleDemandKpi("braking_energy_required_MJ", "Braking Energy Required", "energy_mj"),
    _VehicleDemandKpi("known_rolling_energy_MJ", "Known Rolling Energy", "energy_mj", "rolling"),
    _VehicleDemandKpi("known_aero_energy_MJ", "Known Aero Energy", "energy_mj", "aero"),
    _VehicleDemandKpi("residual_roadload_energy_MJ", "Residual / Unattributed Roadload", "energy_mj"),
    _VehicleDemandKpi("positive_inertial_work_MJ", "Positive Inertial Work", "energy_mj"),
)


@dataclass(frozen=True)
class VehicleDemandOutcome:
    """Wraps one ComparisonItem's Vehicle Demand computation -- either a real
    VehicleDemandResult, or a short, human-readable reason it could not be
    computed (Sprint 9D Sec 41-42: a scenario-level failure never raises out
    of this module, and is never shown as a traceback/internal field code).
    """

    result: VehicleDemandResult | None
    unavailable_reason: str | None


def _vehicle_demand_request_from_comparison_item(item: ComparisonItem) -> VehicleDemandRequest | None:
    """None only when the item has no TOTAL roadload -- the one case a
    VehicleDemandRequest genuinely cannot represent (roadload_total is
    required by the frozen Sprint 9A contract).
    """
    total_boundary = item.roadload["total"]
    if not total_boundary.available:
        return None
    net_boundary = item.roadload["net"]

    roadload_total = RoadloadCoefficients(A_N=total_boundary.A, B_N_per_kph=total_boundary.B, C_N_per_kph2=total_boundary.C)
    roadload_net = (
        RoadloadCoefficients(A_N=net_boundary.A, B_N_per_kph=net_boundary.B, C_N_per_kph2=net_boundary.C)
        if net_boundary.available
        else None
    )

    rrc = item.vehicle.get("rrc_N_per_kN")
    cda = item.vehicle.get("cda_m2")
    cycle_name = default_cycle_for_legislation(item.vehicle.get("legislation"))

    provenance = {
        "roadload_total": Provenance.SOURCE.value,
        "roadload_net": Provenance.CALCULATED.value if roadload_net is not None else Provenance.UNAVAILABLE.value,
        "rrc": Provenance.SOURCE.value if rrc is not None else Provenance.UNAVAILABLE.value,
        "cda": Provenance.SOURCE.value if cda is not None else Provenance.UNAVAILABLE.value,
    }

    return VehicleDemandRequest(
        source_kind=item.source_kind.value,
        vde_id=item.vde_id,
        fuelcons_id=item.fuelcons_id,
        label=item.label,
        cycle_name=cycle_name,
        cycle_source="STANDARD" if cycle_name else None,
        cycle_version=None,
        test_mass_kg=item.vehicle.get("test_mass_kg"),
        roadload_total=roadload_total,
        roadload_net=roadload_net,
        rrc_n_per_kn=float(rrc) if rrc is not None else None,
        cda_m2=float(cda) if cda is not None else None,
        ambient=AmbientState(),
        provenance=provenance,
        model_version=None,
    )


def get_vehicle_demand_result(item: ComparisonItem) -> VehicleDemandOutcome:
    """Compute-on-demand, never persisted (VehicleDemandProfile is a runtime
    object, Sprint 9B). Any ValueError the frozen engine raises for invalid
    mass/RRC/CdA/ambient or a malformed cycle is caught here and turned into
    a short reason -- one scenario's invalid data never raises out of this
    function or takes the rest of Comparison down with it.
    """
    request = _vehicle_demand_request_from_comparison_item(item)
    if request is None:
        return VehicleDemandOutcome(result=None, unavailable_reason="Roadload TOTAL is unavailable for this scenario.")
    if request.test_mass_kg is None:
        return VehicleDemandOutcome(result=None, unavailable_reason="Effective test mass is unavailable for this scenario.")

    cycle_frame = resolve_vehicle_demand_cycle(item.vehicle)
    if cycle_frame is None:
        return VehicleDemandOutcome(
            result=None, unavailable_reason="No standard cycle trace is available for this scenario's legislation."
        )

    try:
        result = calculate_vehicle_demand(request, cycle_frame)
    except ValueError as exc:
        return VehicleDemandOutcome(result=None, unavailable_reason=f"Vehicle Demand unavailable: {exc}")
    return VehicleDemandOutcome(result=result, unavailable_reason=None)


def resolve_vehicle_demand_outcomes(dataset: ComparisonDataset) -> dict[int, VehicleDemandOutcome]:
    """One VehicleDemandOutcome per item, keyed on Python object identity --
    a per-call memo (Sprint 9D Sec 10-11), not a persistent cache. Callers
    that need more than one presentation view of the same dataset in a
    single render (the KPI table and the breakdown chart both do) should
    compute this once and pass it to both, rather than each independently
    triggering its own calculate_vehicle_demand() pass per item.
    """
    return {id(item): get_vehicle_demand_result(item) for item in dataset_items(dataset)}


def _summary_for_basis(outcome: VehicleDemandOutcome, basis: RoadloadBasis) -> VehicleDemandSummary | None:
    if outcome.result is None:
        return None
    return outcome.result.total_summary if basis is RoadloadBasis.TOTAL else outcome.result.net_summary


def _kpi_cell(
    kpi: _VehicleDemandKpi, summary: VehicleDemandSummary | None, outcome: VehicleDemandOutcome, basis: RoadloadBasis, *, unit_system: str
) -> ScorecardCell:
    if summary is None:
        # outcome.unavailable_reason is set when the whole computation failed
        # (bad mass/roadload/cycle); when it is None here, the computation
        # succeeded but this specific basis genuinely has no roadload (e.g.
        # NET with no resolved transmission) -- never TOTAL shown in its
        # place (Sprint 9D Sec 8, 46: no TOTAL<->NET fallback).
        reason = outcome.unavailable_reason or f"{basis.value} roadload is unavailable for this scenario."
        return ScorecardCell(
            raw_value=None,
            formatted_value="-",
            absolute_delta=None,
            formatted_delta=None,
            percent_delta=None,
            semantic=None,
            compatible=True,
            available=False,
            basis_mismatch=False,
            warning=reason,
        )

    value = getattr(summary, kpi.field_name)
    warning = None
    if value is None and kpi.unavailable_reason is not None:
        warning = _UNAVAILABLE_REASON_TEXT[kpi.unavailable_reason]

    formatted = format_value(value, kpi.unit_family, unit_system)
    # Residual is the one signed KPI (Sprint 9B Sec 27): a negative value is
    # preserved as-is, never abs()'d or clipped, with a discreet "(Review)"
    # marker embedded directly in the value text so it always shows even
    # when a Reference delta is also present for this cell (Sprint 9D Sec 20).
    if kpi.field_name == "residual_roadload_energy_MJ" and value is not None and value < 0:
        formatted = f"{formatted} (Review)"
        warning = "Known contributions exceed authoritative roadload for part of the cycle; residual is preserved."

    return ScorecardCell(
        raw_value=value,
        formatted_value=formatted,
        absolute_delta=None,
        formatted_delta=None,
        percent_delta=None,
        semantic=None,
        compatible=True,
        available=value is not None,
        basis_mismatch=False,
        warning=warning,
    )


def _with_delta(
    cell: ScorecardCell,
    reference_cell: ScorecardCell,
    reference_item: ComparisonItem,
    item: ComparisonItem,
    kpi: _VehicleDemandKpi,
    unit_system: str,
) -> ScorecardCell:
    """Mirrors comparison_report_service.compare_metric's exact delta
    formula and its SAME_LEGISLATION_CYCLE compatibility check (Sprint 9D
    Sec 30: no new delta logic) -- Vehicle Demand KPIs are cycle-dependent
    quantities, same as the Registry's own vde_total/vde_net/roadload_*
    metrics.
    """
    if not cell.available or not reference_cell.available:
        return cell
    same_legislation = reference_item.vehicle.get("legislation") == item.vehicle.get("legislation")
    if not same_legislation:
        return replace(
            cell,
            compatible=False,
            basis_mismatch=True,
            warning=cell.warning or "Different cycle / incompatible basis",
        )
    ref_value = float(reference_cell.raw_value)
    cmp_value = float(cell.raw_value)
    absolute_delta = cmp_value - ref_value
    percent_delta = ((cmp_value / ref_value) - 1.0) * 100.0 if ref_value != 0 else None
    formatted_delta = _format_delta(absolute_delta, percent_delta, kpi.unit_family, unit_system)
    return replace(cell, absolute_delta=absolute_delta, percent_delta=percent_delta, formatted_delta=formatted_delta)


def build_vehicle_demand_comparison_rows(
    dataset: ComparisonDataset,
    basis: RoadloadBasis,
    unit_system: str,
    *,
    outcomes: Mapping[int, VehicleDemandOutcome] | None = None,
) -> ScorecardSection:
    """The canonical builder: VehicleDemandResult -> directly renderable
    ScorecardSection (Sprint 9D Sec 43). Contains presentation mapping,
    availability, and delta-formatting inputs only -- never physics.

    `outcomes` (from resolve_vehicle_demand_outcomes) lets a caller that also
    needs build_vehicle_demand_breakdown_rows for the same dataset share one
    computation instead of triggering calculate_vehicle_demand() twice per
    item (Sprint 9D Sec 10-11); when omitted, one is computed locally so this
    function stays independently usable/testable.
    """
    items = dataset_items(dataset)
    outcomes = outcomes if outcomes is not None else resolve_vehicle_demand_outcomes(dataset)
    reference = dataset.reference

    rows: list[ScorecardRow] = []
    for kpi in _VEHICLE_DEMAND_KPIS:
        if reference is not None:
            reference_outcome = outcomes[id(reference)]
            reference_cell = _kpi_cell(kpi, _summary_for_basis(reference_outcome, basis), reference_outcome, basis, unit_system=unit_system)
            comparison_cells = []
            for item in items:
                if item is reference:
                    continue
                outcome = outcomes[id(item)]
                cell = _kpi_cell(kpi, _summary_for_basis(outcome, basis), outcome, basis, unit_system=unit_system)
                cell = _with_delta(cell, reference_cell, reference, item, kpi, unit_system)
                comparison_cells.append(cell)
        else:
            # Reference-less (Sprint 9D Sec 7, 30): absolute values only,
            # never a fabricated delta/baseline.
            cells = []
            for item in items:
                outcome = outcomes[id(item)]
                cells.append(_kpi_cell(kpi, _summary_for_basis(outcome, basis), outcome, basis, unit_system=unit_system))
            reference_cell, comparison_cells = cells[0], cells[1:]

        rows.append(
            ScorecardRow(
                metric_key=f"vehicle_demand_{kpi.field_name}",
                label=kpi.label,
                reference_cell=reference_cell,
                comparison_cells=tuple(comparison_cells),
            )
        )

    return ScorecardSection(title=VEHICLE_DEMAND_SECTION_TITLE, rows=tuple(rows))


def build_vehicle_demand_breakdown_rows(
    dataset: ComparisonDataset,
    basis: RoadloadBasis,
    *,
    outcomes: Mapping[int, VehicleDemandOutcome] | None = None,
) -> dict[str, Any]:
    """Rows for the one optional Vehicle Demand chart (Sprint 9D Sec 16-18):
    Known Rolling / Known Aero / Residual, which always sum exactly to
    Authoritative Roadload Energy for that item and basis -- an explicit
    Vehicle Demand Core identity, never a forced-to-100% decomposition.
    known_rolling_MJ/known_aero_MJ are omitted (None) rather than shown as
    zero when genuinely unavailable. See build_vehicle_demand_comparison_
    rows for why `outcomes` can be shared across both builders.
    """
    items = dataset_items(dataset)
    outcomes = outcomes if outcomes is not None else resolve_vehicle_demand_outcomes(dataset)
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for item in items:
        outcome = outcomes[id(item)]
        summary = _summary_for_basis(outcome, basis)
        if summary is None:
            excluded.append({"label": item.label or "Unknown vehicle", "reason": outcome.unavailable_reason or "Vehicle Demand unavailable."})
            continue
        rows.append(
            {
                "label": item.label or "Unknown vehicle",
                "known_rolling_MJ": summary.known_rolling_energy_MJ,
                "known_aero_MJ": summary.known_aero_energy_MJ,
                "residual_MJ": summary.residual_roadload_energy_MJ,
            }
        )
    return {"rows": rows, "excluded": excluded}


__all__ = [
    "VEHICLE_DEMAND_SECTION_TITLE",
    "VehicleDemandOutcome",
    "get_vehicle_demand_result",
    "resolve_vehicle_demand_outcomes",
    "build_vehicle_demand_comparison_rows",
    "build_vehicle_demand_breakdown_rows",
]
