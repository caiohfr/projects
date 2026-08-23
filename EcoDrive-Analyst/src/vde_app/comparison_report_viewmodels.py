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
from src.vde_core.comparison_metric_registry import list_metrics
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
]
