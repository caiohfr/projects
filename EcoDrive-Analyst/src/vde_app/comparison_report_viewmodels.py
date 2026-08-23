# src/vde_app/comparison_report_viewmodels.py
# -----------------------------------------------------------------------------
# Package 8B - pure view-model layer for the Comparison Scorecard. No Streamlit
# import. Never re-implements physics or compatibility rules -- those stay in
# src/vde_core/comparison_report_service.py and comparison_metric_registry.py;
# this module only selects/formats/groups what those already computed.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.vde_app.units import format_quantity
from src.vde_core.comparison_metric_registry import get_metric, list_metrics
from src.vde_core.fuel_energy import LHV_MJ_PER_L, MJ_TO_Wh
from src.vde_core.comparison_report_service import (
    ComparisonDataset,
    ComparisonItem,
    ComparisonRole,
    RevisionStatus,
    compare_metric,
)

MAX_COMPARISONS = 10

_SCORECARD_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Vehicle", "Vehicle / Program"),
    ("Powertrain", "Powertrain"),
    ("Physical setup", "Physical Setup"),
    ("Roadload", "Roadload"),
    ("Vehicle demand", "Vehicle Demand"),
    ("Fuel / Energy / CO2", "Fuel / Energy / Emissions"),
    ("Efficiency", "Efficiency"),
)

_PROVENANCE_ROWS: tuple[tuple[str, str], ...] = (
    ("record_origin", "Record origin"),
    ("scenario_intent", "Scenario intent"),
    ("method", "Method"),
    ("model_version", "Model version"),
    ("source_label", "Source"),
    ("source_vde_revision", "Source VDE revision"),
    ("current_vde_revision", "Current VDE revision"),
    ("revision_status", "Revision status"),
    ("confidence", "Confidence"),
)

_PROVENANCE_SECTION_TITLE = "Data Status / Provenance"

_UNIT_QUANTITY_MAP = {
    "mass_kg": "mass",
    "area_m2": "cda",
    "force_n": "force",
    "force_n_per_kph": "force_per_speed",
    "force_n_per_kph2": "force_per_speed_squared",
    "rrc_n_per_kn": "rrc",
    "energy_mj_per_km": "energy_per_distance",
    "wh_per_km": "energy_wh_per_distance",
    "gco2_per_km": "co2_per_distance",
    "ratio": "fraction",
}

# pwt_fuel_energy.py's private _fuel_display_value() applies this same factor for
# L/100km -> gal/100mi. Replicated here (not imported) to keep this module
# Streamlit/session-state free -- see Package 8B docs for why.
_GAL_PER_100MI_PER_L_PER_100KM = 0.425143707


# -----------------------------------------------------------------------------
# Scenario selection (Sec 10-14, 42, 49)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioOption:
    fuelcons_id: int
    vde_id: int | None
    label: str
    provenance_label: str
    make: str | None
    model: str | None
    year: int | None
    legislation: str | None
    electrification: str | None
    record_origin: str | None


def build_scenario_options(catalog_rows: Sequence[dict[str, Any]]) -> list[ScenarioOption]:
    options: list[ScenarioOption] = []
    for row in catalog_rows:
        make = str(row.get("make") or "").strip()
        model = str(row.get("model") or "").strip()
        base_label = " ".join(part for part in (make, model) if part) or f"VDE #{row.get('vde_id')}"
        year = row.get("year")
        legislation = row.get("legislation")
        electrification = row.get("electrification")
        meta = " · ".join(str(x) for x in (year, legislation, electrification) if x)
        label = f"{base_label} · {meta}" if meta else base_label
        options.append(
            ScenarioOption(
                fuelcons_id=int(row["fuelcons_id"]),
                vde_id=int(row["vde_id"]) if row.get("vde_id") is not None else None,
                label=label,
                provenance_label=str(row.get("record_origin") or "UNKNOWN"),
                make=row.get("make"),
                model=row.get("model"),
                year=year,
                legislation=legislation,
                electrification=electrification,
                record_origin=row.get("record_origin"),
            )
        )
    return options


@dataclass(frozen=True)
class SelectionState:
    reference_fuelcons_id: int | None = None
    comparison_fuelcons_ids: tuple[int, ...] = ()


def set_reference(state: SelectionState, fuelcons_id: int | None) -> SelectionState:
    if fuelcons_id is None:
        return SelectionState(reference_fuelcons_id=None, comparison_fuelcons_ids=state.comparison_fuelcons_ids)
    fuelcons_id = int(fuelcons_id)
    comparisons = tuple(cid for cid in state.comparison_fuelcons_ids if cid != fuelcons_id)
    return SelectionState(reference_fuelcons_id=fuelcons_id, comparison_fuelcons_ids=comparisons)


def add_comparison(state: SelectionState, fuelcons_id: int) -> tuple[SelectionState, str | None]:
    fuelcons_id = int(fuelcons_id)
    if fuelcons_id == state.reference_fuelcons_id:
        return state, "This scenario is already selected as Reference."
    if fuelcons_id in state.comparison_fuelcons_ids:
        return state, None
    if len(state.comparison_fuelcons_ids) >= MAX_COMPARISONS:
        return state, f"Maximum {MAX_COMPARISONS} comparison scenarios reached. Remove one before adding another."
    return (
        SelectionState(state.reference_fuelcons_id, state.comparison_fuelcons_ids + (fuelcons_id,)),
        None,
    )


def remove_comparison(state: SelectionState, fuelcons_id: int) -> SelectionState:
    fuelcons_id = int(fuelcons_id)
    return SelectionState(
        state.reference_fuelcons_id,
        tuple(cid for cid in state.comparison_fuelcons_ids if cid != fuelcons_id),
    )


def sync_comparisons_from_widget(
    state: SelectionState, widget_selected_ids: Sequence[int]
) -> tuple[SelectionState, tuple[str, ...]]:
    """Reconcile a multiselect widget's current value into SelectionState while
    preserving the *original* selection order (Sec 49) rather than trusting the
    widget's own return order, which Streamlit does not guarantee to be
    click-order. Newly-appeared ids are appended via add_comparison (enforcing
    the 10-item cap with an explicit message); disappeared ids are dropped.
    """
    widget_ids = [int(i) for i in widget_selected_ids]
    kept = tuple(cid for cid in state.comparison_fuelcons_ids if cid in widget_ids)
    state = SelectionState(state.reference_fuelcons_id, kept)
    errors: list[str] = []
    for cid in widget_ids:
        if cid in state.comparison_fuelcons_ids:
            continue
        state, error = add_comparison(state, cid)
        if error:
            errors.append(error)
    return state, tuple(errors)


# -----------------------------------------------------------------------------
# Value formatting (Sec 31-32, 36)
# -----------------------------------------------------------------------------


def format_value(raw_value: Any, unit_family: str, unit_system: str, *, unavailable: str = "-", signed: bool = False) -> str:
    if raw_value is None:
        return unavailable
    if unit_family in ("text", "count"):
        return str(raw_value)
    if unit_family == "l_per_100km":
        value = float(raw_value)
        unit = "L/100km"
        if unit_system == "US customary":
            value *= _GAL_PER_100MI_PER_L_PER_100KM
            unit = "gal/100mi"
        text = f"{value:.2f} {unit}"
    elif unit_family == "km_per_l":
        text = f"{float(raw_value):.2f} km/l"
    else:
        quantity = _UNIT_QUANTITY_MAP.get(unit_family)
        text = format_quantity(raw_value, quantity, unit_system, unavailable=unavailable) if quantity else str(raw_value)
    if signed and isinstance(raw_value, (int, float)) and raw_value > 0 and not text.startswith("+"):
        text = f"+{text}"
    return text


def _format_delta(absolute_delta: float | None, percent_delta: float | None, unit_family: str, unit_system: str) -> str | None:
    if absolute_delta is None:
        return None
    abs_text = format_value(absolute_delta, unit_family, unit_system, signed=True)
    if percent_delta is None:
        return abs_text
    sign = "+" if percent_delta > 0 else ""
    return f"{abs_text} · {sign}{percent_delta:.1f}%"


# -----------------------------------------------------------------------------
# Scorecard construction (Sec 17-27, 47-48)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorecardCell:
    raw_value: Any
    formatted_value: str
    absolute_delta: float | None
    formatted_delta: str | None
    percent_delta: float | None
    semantic: str | None  # "BETTER" | "WORSE" | None (SAME/CONTEXT_DEPENDENT collapse to None)
    compatible: bool
    available: bool
    basis_mismatch: bool
    warning: str | None


@dataclass(frozen=True)
class ScorecardRow:
    metric_key: str
    label: str
    reference_cell: ScorecardCell
    comparison_cells: tuple[ScorecardCell, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScorecardSection:
    title: str
    rows: tuple[ScorecardRow, ...] = field(default_factory=tuple)


def _semantic_for_display(semantic: str | None) -> str | None:
    return semantic if semantic in ("BETTER", "WORSE") else None


def _metric_row(dataset: ComparisonDataset, metric_key: str, label: str, unit_family: str, unit_system: str) -> ScorecardRow:
    self_result = compare_metric(dataset.reference, dataset.reference, metric_key)
    reference_cell = ScorecardCell(
        raw_value=self_result["reference_value"],
        formatted_value=format_value(self_result["reference_value"], unit_family, unit_system),
        absolute_delta=None,
        formatted_delta=None,
        percent_delta=None,
        semantic=None,
        compatible=True,
        available=self_result["available"],
        basis_mismatch=False,
        warning=None,
    )
    comparison_cells = []
    for item in dataset.comparisons:
        result = compare_metric(dataset.reference, item, metric_key)
        compatible = result["compatible"]
        warning = None if compatible else "Different cycle / incompatible basis"
        comparison_cells.append(
            ScorecardCell(
                raw_value=result["comparison_value"],
                formatted_value=format_value(result["comparison_value"], unit_family, unit_system),
                absolute_delta=result["absolute_delta"] if compatible else None,
                formatted_delta=_format_delta(result["absolute_delta"], result["percent_delta"], unit_family, unit_system)
                if compatible
                else None,
                percent_delta=result["percent_delta"] if compatible else None,
                semantic=_semantic_for_display(result["semantic"]) if compatible else None,
                compatible=compatible,
                available=result["available"],
                basis_mismatch=result["basis_mismatch"],
                warning=warning,
            )
        )
    return ScorecardRow(metric_key=metric_key, label=label, reference_cell=reference_cell, comparison_cells=tuple(comparison_cells))


def _provenance_value(item: ComparisonItem, field_name: str) -> Any:
    value = getattr(item.provenance, field_name)
    return value.value if hasattr(value, "value") else value


def _provenance_cell(value: Any) -> ScorecardCell:
    return ScorecardCell(
        raw_value=value,
        formatted_value=str(value) if value not in (None, "") else "-",
        absolute_delta=None,
        formatted_delta=None,
        percent_delta=None,
        semantic=None,
        compatible=True,
        available=value not in (None, ""),
        basis_mismatch=False,
        warning=None,
    )


def _provenance_section(dataset: ComparisonDataset) -> ScorecardSection:
    rows = []
    for field_name, label in _PROVENANCE_ROWS:
        reference_cell = _provenance_cell(_provenance_value(dataset.reference, field_name))
        comparison_cells = tuple(_provenance_cell(_provenance_value(item, field_name)) for item in dataset.comparisons)
        rows.append(
            ScorecardRow(
                metric_key=f"provenance_{field_name}",
                label=label,
                reference_cell=reference_cell,
                comparison_cells=comparison_cells,
            )
        )
    return ScorecardSection(title=_PROVENANCE_SECTION_TITLE, rows=tuple(rows))


def build_scorecard_sections(dataset: ComparisonDataset, *, unit_system: str = "Metric") -> list[ScorecardSection]:
    sections: list[ScorecardSection] = []
    for group_key, title in _SCORECARD_SECTIONS:
        rows = tuple(
            _metric_row(dataset, metric.key, metric.label, metric.unit_family, unit_system)
            for metric in list_metrics(group_key)
        )
        sections.append(ScorecardSection(title=title, rows=rows))
    sections.append(_provenance_section(dataset))
    return sections


# -----------------------------------------------------------------------------
# Column headers and dataset-level warnings (Sec 12, 27, 38)
# -----------------------------------------------------------------------------


def build_scenario_header(item: ComparisonItem) -> dict[str, Any]:
    provenance = item.provenance
    origin = provenance.record_origin or "UNKNOWN"
    badge_parts = [origin]
    if provenance.scenario_intent and provenance.scenario_intent != provenance.record_origin:
        badge_parts.append(str(provenance.scenario_intent))
    is_stale = provenance.revision_status is RevisionStatus.STALE
    if is_stale:
        badge_parts.append("STALE SOURCE")
    provenance_text = " · ".join(badge_parts)
    is_reference = item.role is ComparisonRole.REFERENCE
    label = item.label or "Unknown vehicle"
    title_lines = [label, ("REFERENCE · " if is_reference else "") + provenance_text]
    return {
        "label": label,
        "role": "REFERENCE" if is_reference else "COMPARISON",
        "provenance_text": provenance_text,
        "is_stale": is_stale,
        "column_title": "\n".join(title_lines),
    }


def build_reference_summary(item: ComparisonItem) -> dict[str, Any]:
    """Compact Reference summary for the Dashboard tab (Sec 7). No score, no badge."""
    aggregate = item.vde["aggregate"]
    fuel = item.fuel_energy or {}
    return {
        "label": item.label or "Unknown vehicle",
        "legislation": item.vehicle.get("legislation"),
        "cycle_name": item.vehicle.get("cycle_name"),
        "record_origin": item.provenance.record_origin,
        "vde_total": aggregate.get("total"),
        "vde_net": aggregate.get("net"),
        "fuel_l_per_100km": fuel.get("fuel_l_per_100km"),
        "fuel_km_per_l": fuel.get("fuel_km_per_l"),
        "energy_wh_per_km": fuel.get("energy_Wh_per_km"),
        "gco2_per_km": fuel.get("gco2_per_km"),
        "eta_pt_est": fuel.get("eta_pt_est"),
    }


@dataclass(frozen=True)
class BarRow:
    label: str
    value: float
    formatted_value: str
    semantic: str | None


def build_metric_bar_rows(
    dataset: ComparisonDataset, metric_key: str, *, unit_system: str = "Metric"
) -> dict[str, list]:
    """One generic row-builder reused for VDE/fuel/energy/CO2/mass/CdA/RRC bars.
    Always routes through compare_metric() -- never re-implements delta/compatibility.
    Nothing is silently dropped: excluded items carry an explicit reason (Sec 10, 41).
    """
    metric = get_metric(metric_key)
    if metric is None:
        return {"rows": [], "excluded": []}

    rows: list[BarRow] = []
    excluded: list[dict[str, str]] = []

    self_result = compare_metric(dataset.reference, dataset.reference, metric_key)
    ref_label = dataset.reference.label or "Unknown vehicle"
    if self_result["available"]:
        rows.append(
            BarRow(
                label=ref_label,
                value=self_result["reference_value"],
                formatted_value=format_value(self_result["reference_value"], metric.unit_family, unit_system),
                semantic=None,
            )
        )
    else:
        excluded.append({"label": ref_label, "reason": f"{metric.label} unavailable"})

    for item in dataset.comparisons:
        result = compare_metric(dataset.reference, item, metric_key)
        label = item.label or "Unknown vehicle"
        if not result["compatible"]:
            excluded.append({"label": label, "reason": "Different cycle / incompatible basis"})
            continue
        if not result["available"]:
            excluded.append({"label": label, "reason": f"{metric.label} unavailable"})
            continue
        rows.append(
            BarRow(
                label=label,
                value=result["comparison_value"],
                formatted_value=format_value(result["comparison_value"], metric.unit_family, unit_system),
                semantic=_semantic_for_display(result["semantic"]),
            )
        )
    return {"rows": rows, "excluded": excluded}


# -----------------------------------------------------------------------------
# Roadload & VDE physical chart data preparation (Sec 22-38, Package 8C)
#
# Physical traces (ABC, roadload curve, cycle demand) are a pure function of
# vde_id + boundary -- two FuelCons scenarios sharing one VDE would otherwise
# draw two identical overlapping traces. This dedup applies ONLY here, never
# to Scorecard/Dashboard scenario-level rows (Sec 25).
# -----------------------------------------------------------------------------


def _scenario_identity(item: ComparisonItem) -> str:
    if item.fuelcons_id is not None:
        origin = item.provenance.record_origin or "UNKNOWN"
        return f"{origin} (#{item.fuelcons_id})"
    return item.label or f"VDE #{item.vde_id}"


@dataclass(frozen=True)
class DedupedVdeGroup:
    vde_id: int
    label: str
    used_by: tuple[str, ...]
    item: ComparisonItem


def deduplicate_by_vde_id(items: Sequence[ComparisonItem]) -> list[DedupedVdeGroup]:
    groups: dict[int, DedupedVdeGroup] = {}
    order: list[int] = []
    for item in items:
        if item.vde_id is None:
            continue
        identity = _scenario_identity(item)
        if item.vde_id not in groups:
            groups[item.vde_id] = DedupedVdeGroup(
                vde_id=item.vde_id, label=item.label or f"VDE #{item.vde_id}", used_by=(identity,), item=item
            )
            order.append(item.vde_id)
        else:
            existing = groups[item.vde_id]
            if identity not in existing.used_by:
                groups[item.vde_id] = DedupedVdeGroup(
                    vde_id=existing.vde_id, label=existing.label, used_by=existing.used_by + (identity,), item=existing.item
                )
    return [groups[vde_id] for vde_id in order]


def build_abc_rows(dataset: ComparisonDataset, boundary: str) -> dict[str, list]:
    boundary_key = boundary.lower()
    groups = deduplicate_by_vde_id((dataset.reference, *dataset.comparisons))
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for group in groups:
        rb = group.item.roadload[boundary_key]
        if not rb.available:
            excluded.append({"label": group.label, "reason": f"Roadload {boundary} unavailable"})
            continue
        rows.append({"label": group.label, "used_by": group.used_by, "A": rb.A, "B": rb.B, "C": rb.C})
    return {"rows": rows, "excluded": excluded}


def build_roadload_curve_rows(dataset: ComparisonDataset, boundary: str) -> dict[str, list]:
    """Row shape matches plots.roadload_curve_comparison_chart's expected input
    exactly ({label, A_N, B_N_per_kph, C_N_per_kph2}) -- no adapter needed at
    the render layer.
    """
    boundary_key = boundary.lower()
    groups = deduplicate_by_vde_id((dataset.reference, *dataset.comparisons))
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for group in groups:
        rb = group.item.roadload[boundary_key]
        if not rb.available:
            excluded.append({"label": group.label, "reason": f"Roadload {boundary} unavailable"})
            continue
        rows.append({"label": group.label, "A_N": rb.A, "B_N_per_kph": rb.B, "C_N_per_kph2": rb.C})
    return {"rows": rows, "excluded": excluded}


def build_cycle_demand_rows(dataset: ComparisonDataset, cycle_frame, boundaries: Sequence[str]) -> dict[str, Any]:
    """Thin adapter: builds the `scenarios` list roadload_analysis.build_cycle_power_analysis
    expects from deduped ComparisonItems, then returns its `series` unmodified --
    the power/energy math itself is never re-implemented here.
    """
    from src.vde_core.roadload_analysis import build_cycle_power_analysis

    groups = deduplicate_by_vde_id((dataset.reference, *dataset.comparisons))
    scenarios = []
    for group in groups:
        total_rb = group.item.roadload["total"]
        net_rb = group.item.roadload["net"]
        scenarios.append(
            {
                "id": str(group.vde_id),
                "label": group.label,
                "mass_kg": group.item.vehicle.get("mass_kg"),
                "total": {"A": total_rb.A, "B": total_rb.B, "C": total_rb.C} if total_rb.available else {},
                "net": {"A": net_rb.A, "B": net_rb.B, "C": net_rb.C} if net_rb.available else {},
            }
        )
    analysis = build_cycle_power_analysis(cycle_frame, scenarios)

    resolved = {(s["scenario_id"], s["boundary"]) for s in analysis["series"]}
    excluded: list[dict[str, str]] = []
    for group in groups:
        for boundary in boundaries:
            if (str(group.vde_id), boundary) not in resolved:
                excluded.append({"label": group.label, "reason": f"Roadload {boundary} or mass unavailable"})

    return {
        "time_s": analysis["time_s"],
        "speed_kph": analysis["speed_kph"],
        "series": [s for s in analysis["series"] if s["boundary"] in boundaries],
        "excluded": excluded,
    }


# -----------------------------------------------------------------------------
# FE x VDE / equi-PSE and competitor delta (Sec 15-21, Package 8C)
#
# The three inconsistent LHV tables found in the repo (fuel_energy.py,
# derivatives.py, plots.py's hardcoded default) are resolved by treating
# fuel_energy.py::LHV_MJ_PER_L/MJ_TO_Wh as canonical -- it is what actually
# backs the stored consumption numbers, unlike the other two display-only
# tables. "Flex"/"Electric"/unknown fuel_type is not LHV-mappable and is
# excluded rather than guessed (never invent a blend percentage).
# -----------------------------------------------------------------------------

_LHV_MAPPABLE_FUEL_TYPES = {"Gasoline": "Gasoline", "Diesel": "Diesel", "Ethanol": "E100"}


def is_temporary_net(item: ComparisonItem) -> bool:
    return "temporary_transmission_used" in item.warnings


def _consumed_energy_mj_per_km(fuel: Mapping[str, Any]) -> float | None:
    if fuel.get("electrification") == "BEV":
        wh = fuel.get("energy_Wh_per_km")
        return (wh / MJ_TO_Wh) if wh is not None else None
    lhv_key = _LHV_MAPPABLE_FUEL_TYPES.get(fuel.get("fuel_type"))
    l100 = fuel.get("fuel_l_per_100km")
    if lhv_key is None or l100 is None:
        return None
    return (l100 / 100.0) * LHV_MJ_PER_L[lhv_key]


def build_fe_vde_points(dataset: ComparisonDataset, *, boundary: str, mode: str) -> dict[str, list]:
    """mode in {"volumetric", "energy_normalized", "electrical"}. x is always
    item.vde["aggregate"][boundary] -- never a fallback boundary. Excluded
    points always carry a reason (Sec 16, 41); nothing is silently dropped.
    """
    boundary_key = boundary.lower()
    reference = dataset.reference
    ref_fuel_type = (reference.fuel_energy or {}).get("fuel_type")
    points: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    for item in (reference, *dataset.comparisons):
        label = item.label or "Unknown vehicle"
        x = item.vde["aggregate"].get(boundary_key)
        if x is None:
            excluded.append({"label": label, "reason": f"VDE {boundary} unavailable"})
            continue

        fuel = item.fuel_energy or {}
        if mode == "volumetric":
            fuel_type = fuel.get("fuel_type")
            if not fuel_type or (item is not reference and fuel_type != ref_fuel_type):
                excluded.append({"label": label, "reason": "Different fuel family - use energy-normalized mode"})
                continue
            y = fuel.get("fuel_l_per_100km")
            if y is None:
                excluded.append({"label": label, "reason": "Fuel consumption unavailable"})
                continue
        elif mode == "energy_normalized":
            y = _consumed_energy_mj_per_km(fuel)
            if y is None:
                excluded.append({"label": label, "reason": "Fuel blend not resolvable to an energy basis"})
                continue
        elif mode == "electrical":
            if fuel.get("electrification") != "BEV":
                excluded.append({"label": label, "reason": "Not a BEV scenario"})
                continue
            y = fuel.get("energy_Wh_per_km")
            if y is None:
                excluded.append({"label": label, "reason": "Electrical energy unavailable"})
                continue
        else:
            raise ValueError(f"Unknown FE x VDE mode: {mode!r}")

        points.append(
            {
                "label": label,
                "role": item.role.value,
                "x": x,
                "y": y,
                "provenance": item.provenance.record_origin,
                "is_temporary_net": is_temporary_net(item),
                "revision_status": item.provenance.revision_status.value if item.provenance.revision_status else None,
            }
        )
    return {"points": points, "excluded": excluded}


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num < 2 or start == stop:
        return [start] * max(num, 1)
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def build_iso_pse_lines(
    x_min: float, x_max: float, eta_values: Sequence[float], *, mode: str, fuel_type: str | None = None
) -> list[dict[str, Any]]:
    """Reuses the same PSE ratio (demand/consumed) as powertrain_efficiency.py,
    inverted to solve for y given x and eta -- never a duplicated equation.
    Returns [] (no lines, not fake ones) when the dataset doesn't support a
    defensible line for this mode/fuel_type (Sec 18-19).
    """
    lhv = None
    if mode == "volumetric":
        lhv_key = _LHV_MAPPABLE_FUEL_TYPES.get(fuel_type)
        if lhv_key is None:
            return []
        lhv = LHV_MJ_PER_L[lhv_key]
    elif mode not in ("energy_normalized", "electrical"):
        raise ValueError(f"Unknown FE x VDE mode: {mode!r}")

    xs = _linspace(x_min, x_max, 40)
    lines: list[dict[str, Any]] = []
    for eta in eta_values:
        if eta is None or eta <= 0:
            continue
        if mode == "volumetric":
            ys = [x / eta / lhv * 100.0 for x in xs]
        elif mode == "energy_normalized":
            ys = [x / eta for x in xs]
        else:  # electrical
            ys = [x * MJ_TO_Wh / eta for x in xs]
        lines.append({"eta": eta, "x": list(xs), "y": ys})
    return lines


def build_competitor_delta_rows(
    dataset: ComparisonDataset, metric_key: str
) -> dict[str, list]:
    """Reference is fixed at 0%/no-verdict; comparisons come straight from
    compare_metric() -- delta semantics are never recomputed here (Sec 20-21).
    """
    metric = get_metric(metric_key)
    if metric is None:
        return {"rows": [], "excluded": []}

    rows: list[dict[str, Any]] = [
        {
            "label": dataset.reference.label or "Unknown vehicle",
            "role": "REFERENCE",
            "percent_delta": 0.0,
            "absolute_delta": 0.0,
            "semantic": None,
        }
    ]
    excluded: list[dict[str, str]] = []
    for item in dataset.comparisons:
        result = compare_metric(dataset.reference, item, metric_key)
        label = item.label or "Unknown vehicle"
        if not result["compatible"]:
            excluded.append({"label": label, "reason": "Different cycle / incompatible basis"})
            continue
        if not result["available"]:
            excluded.append({"label": label, "reason": f"{metric.label} unavailable"})
            continue
        rows.append(
            {
                "label": label,
                "role": "COMPARISON",
                "percent_delta": result["percent_delta"],
                "absolute_delta": result["absolute_delta"],
                "semantic": _semantic_for_display(result["semantic"]),
            }
        )
    return {"rows": rows, "excluded": excluded}


def dataset_warnings_summary(dataset: ComparisonDataset) -> list[str]:
    items = (dataset.reference, *dataset.comparisons)
    warnings: list[str] = []

    stale_count = sum(1 for item in items if item.provenance.revision_status is RevisionStatus.STALE)
    if stale_count:
        plural = "s" if stale_count != 1 else ""
        warnings.append(f"{stale_count} scenario{plural} use{'s' if stale_count == 1 else ''} a stale VDE revision.")

    no_net_count = sum(1 for item in items if item.vde["aggregate"]["net"] is None)
    if no_net_count:
        plural = "s" if no_net_count != 1 else ""
        verb = "has" if no_net_count == 1 else "have"
        warnings.append(f"{no_net_count} scenario{plural} {verb} no NET boundary.")

    legislations = {item.vehicle.get("legislation") for item in items if item.vehicle.get("legislation")}
    if len(legislations) > 1:
        warnings.append(f"Mixed legislations selected: {', '.join(sorted(legislations))}.")

    return warnings


__all__ = [
    "MAX_COMPARISONS",
    "ScenarioOption",
    "build_scenario_options",
    "SelectionState",
    "set_reference",
    "add_comparison",
    "remove_comparison",
    "sync_comparisons_from_widget",
    "format_value",
    "ScorecardCell",
    "ScorecardRow",
    "ScorecardSection",
    "build_scorecard_sections",
    "build_scenario_header",
    "dataset_warnings_summary",
    "build_reference_summary",
    "BarRow",
    "build_metric_bar_rows",
    "DedupedVdeGroup",
    "deduplicate_by_vde_id",
    "build_abc_rows",
    "build_roadload_curve_rows",
    "build_cycle_demand_rows",
    "is_temporary_net",
    "build_fe_vde_points",
    "build_iso_pse_lines",
    "build_competitor_delta_rows",
]
