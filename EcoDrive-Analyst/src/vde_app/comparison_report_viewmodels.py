# src/vde_app/comparison_report_viewmodels.py
# -----------------------------------------------------------------------------
# Package 8B - pure view-model layer for the Comparison Scorecard. No Streamlit
# import. Never re-implements physics or compatibility rules -- those stay in
# src/vde_core/comparison_report_service.py and comparison_metric_registry.py;
# this module only selects/formats/groups what those already computed.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from src.vde_app.units import format_quantity
from src.vde_app.units import unit_label as _unit_label
from src.vde_core.comparison_metric_registry import MetricDefinition, get_metric, list_metrics
from src.vde_core.fuel_energy import MJ_TO_Wh, LhvBasis, resolve_fuel_energy_basis
from src.vde_core.comparison_report_service import (
    ComparisonDataset,
    ComparisonItem,
    ComparisonRole,
    LineageChainResult,
    LineageChainStatus,
    RevisionStatus,
    build_vde_comparison_item,
    compare_metric,
    extract_metric_value,
    resolve_lineage_chain,
    semantic_for_delta,
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
    "energy_mj": "energy_mj",
    "wh_per_km": "energy_wh_per_distance",
    "gco2_per_km": "co2_per_distance",
    "ratio": "fraction",
}

# pwt_fuel_energy.py's private _fuel_display_value() applies this same factor for
# L/100km -> gal/100mi. Replicated here (not imported) to keep this module
# Streamlit/session-state free -- see Package 8B docs for why.
_GAL_PER_100MI_PER_L_PER_100KM = 0.425143707


# -----------------------------------------------------------------------------
# Dataset-wide item ordering (Package 8F Increment 1 -- optional Reference)
# -----------------------------------------------------------------------------


def dataset_items(dataset: ComparisonDataset) -> tuple[ComparisonItem, ...]:
    """The single canonical "all selected items, Reference first when one
    exists" ordering. Every function that previously hardcoded
    `(dataset.reference, *dataset.comparisons)` routes through this so that a
    Reference-less dataset (dataset.reference is None) is handled in exactly
    one place rather than re-guarded ad hoc at each call site.
    """
    if dataset.reference is not None:
        return (dataset.reference, *dataset.comparisons)
    return dataset.comparisons


# -----------------------------------------------------------------------------
# Scenario selection (Sec 10-14, 42, 49)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Engineering filters (Package 8F) -- engine displacement / rated power
#
# Canonical semantics only: engine_size_l lives on vde_db, engine_max_power_kw
# on fuelcons_db (pwt_fuel_energy.py's filters_bar() already treats these as
# canonical; nothing here duplicates or renames them). HP_PER_KW is the same
# conversion factor pwt_fuel_energy.py's filters_bar() uses (1.34102209) --
# conversion happens only at this UI/filter boundary, never stored as a
# second power field. A missing value is never treated as zero: a row is
# only excluded once the caller actually supplies a (min, max) range for
# that field, and even then only that specific row is dropped, never the
# whole dataset.
# -----------------------------------------------------------------------------

HP_PER_KW = 1.34102209


def kw_to_hp(kw: float | None) -> float | None:
    return None if kw is None else float(kw) * HP_PER_KW


def hp_to_kw(hp: float | None) -> float | None:
    return None if hp is None else float(hp) / HP_PER_KW


def apply_engineering_filters(
    rows: Sequence[Mapping[str, Any]],
    *,
    engine_size_l_range: tuple[float | None, float | None] | None = None,
    engine_max_power_kw_range: tuple[float | None, float | None] | None = None,
) -> list[Mapping[str, Any]]:
    """Numeric range filter over scenario catalog rows. Each range is only
    active when supplied (not None); an active range excludes a row whose
    field is missing (never coerced to 0) or outside [min, max] (either
    bound may itself be None for an open-ended range). Rows are otherwise
    retained unchanged -- this never mutates or reorders them.
    """

    def _in_range(value: Any, range_: tuple[float | None, float | None] | None) -> bool:
        if range_ is None:
            return True
        if value is None:
            return False
        lo, hi = range_
        if lo is not None and value < lo:
            return False
        if hi is not None and value > hi:
            return False
        return True

    return [
        row
        for row in rows
        if _in_range(row.get("engine_size_l"), engine_size_l_range)
        and _in_range(row.get("engine_max_power_kw"), engine_max_power_kw_range)
    ]


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
# Presentation roles + Current designation (Package 8F Increment 2)
#
# A deliberately small, presentation-only overlay keyed by canonical identity
# (fc:<fuelcons_id> / vde:<vde_id>) -- never persisted, never stored on
# ComparisonItem/ComparisonDataset, and never inferred from record_origin,
# method, model_version, timestamp, label, or lineage. The canonical
# ComparisonRole (REFERENCE/COMPARISON) stays exactly as it was in Package 8A
# -- this is a second, independent axis, not an expansion of it. Provenance
# (record_origin etc.) is a third, still-separate axis: a Proposal may be
# Estimated, a Benchmark may be Homologated, and this module never collapses
# the two.
# -----------------------------------------------------------------------------


class PresentationRole(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    PROPOSAL = "PROPOSAL"
    BENCHMARK = "BENCHMARK"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PresentationState:
    """Session-only. `roles` maps a canonical identity to a PresentationRole
    value; an identity absent from the mapping is UNSPECIFIED. `current_item_id`
    is a single optional designation, independent of role -- an item may be
    PROPOSAL *and* Current at once (Sec: "Current is NOT a mutually-exclusive
    role").
    """

    roles: Mapping[str, str] = field(default_factory=dict)
    current_item_id: str | None = None


def set_presentation_role(state: PresentationState, identity: str, role: PresentationRole) -> PresentationState:
    roles = dict(state.roles)
    if role is PresentationRole.UNSPECIFIED:
        roles.pop(identity, None)
    else:
        roles[identity] = role.value
    return PresentationState(roles=roles, current_item_id=state.current_item_id)


def presentation_role_for(state: PresentationState, identity: str) -> PresentationRole:
    raw = state.roles.get(identity)
    try:
        return PresentationRole(raw) if raw else PresentationRole.UNSPECIFIED
    except ValueError:
        return PresentationRole.UNSPECIFIED


def set_current_item(state: PresentationState, identity: str | None) -> PresentationState:
    return PresentationState(roles=state.roles, current_item_id=identity)


def is_current_item(state: PresentationState, identity: str) -> bool:
    return state.current_item_id is not None and state.current_item_id == identity


# -----------------------------------------------------------------------------
# Primary KPI + Target (Package 8F Increment 3)
#
# Session-only, never persisted, never a scenario/comparison item. Keyed by
# metric_key (targets_by_metric) so switching the Primary KPI never
# reinterprets a stored number under a different metric's units -- a target
# set for "Fuel consumption" stays exactly that even if the user switches the
# Primary KPI to "VDE TOTAL" and back. BETTER/WORSE for a gap reuses the same
# semantic_for_delta() rule compare_metric() uses -- no second sign
# convention.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetState:
    targets_by_metric: Mapping[str, float] = field(default_factory=dict)


def set_target(state: TargetState, metric_key: str, value: float | None) -> TargetState:
    targets = dict(state.targets_by_metric)
    if value is None:
        targets.pop(metric_key, None)
    else:
        targets[metric_key] = float(value)
    return TargetState(targets_by_metric=targets)


def get_target(state: TargetState, metric_key: str) -> float | None:
    return state.targets_by_metric.get(metric_key)


@dataclass(frozen=True)
class TargetGap:
    metric_key: str
    target_value: float
    actual_value: float
    absolute_gap: float
    percent_gap: float | None
    semantic: str | None  # "BETTER" | "WORSE" | "SAME" | None


def evaluate_target_gap(metric_key: str, actual_value: float | None, target_value: float | None) -> TargetGap | None:
    """Raw gap = actual - target, always explicit (never hidden inside a
    formatted string only). Returns None when either operand is missing --
    a gap is never fabricated from a partial input, and no target line/gap
    is shown when no target exists for this metric.
    """
    if actual_value is None or target_value is None:
        return None
    metric = get_metric(metric_key)
    if metric is None:
        return None
    absolute_gap = actual_value - target_value
    percent_gap = ((actual_value / target_value) - 1.0) * 100.0 if target_value != 0 else None
    return TargetGap(
        metric_key=metric_key,
        target_value=target_value,
        actual_value=actual_value,
        absolute_gap=absolute_gap,
        percent_gap=percent_gap,
        semantic=semantic_for_delta(metric.direction, absolute_gap),
    )


# -----------------------------------------------------------------------------
# Versatile KPI Walk (Package 8F Increment 4)
#
# NOT a rigid waterfall. Underlying item metric values are always absolute
# (extract_metric_value) -- a WalkStep only chooses how to PRESENT one
# already-selected item: ABSOLUTE, or DELTA against one of three bases
# (PREVIOUS_WALK_STATE / REFERENCE / EXPLICIT_ITEM). Every delta reuses
# compare_metric() unmodified -- semantics/compatibility/basis rules are never
# duplicated here, exactly like build_lineage_waterfall (Package 8D). Unlike
# that lineage-specific walk, this one:
#   - never reads vde_id_parent or any DB lineage,
#   - never infers order from selection order, database timestamps, or ids
#     (the caller's explicit `steps` order is the only order used),
#   - tracks one "active anchor" item that only a step with
#     advances_anchor=True can reassign -- a context-only step (e.g. a
#     Benchmark shown for reference) is fully rendered but never changes what
#     the next PREVIOUS_WALK_STATE step compares against.
# A separate chart builder (comparison_report_charts.py) renders this -- the
# Physical VDE Lineage waterfall is never repurposed or reused for it, since
# the two are different domains (explicit DB lineage vs. presentation intent).
# -----------------------------------------------------------------------------


class WalkDisplayMode(str, Enum):
    ABSOLUTE = "ABSOLUTE"
    DELTA = "DELTA"

    def __str__(self) -> str:
        return self.value


class WalkDeltaBase(str, Enum):
    PREVIOUS_WALK_STATE = "PREVIOUS_WALK_STATE"
    REFERENCE = "REFERENCE"
    EXPLICIT_ITEM = "EXPLICIT_ITEM"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class WalkStep:
    item_id: str  # canonical identity (fc:<id> / vde:<id>)
    display_mode: WalkDisplayMode
    delta_base: WalkDeltaBase | None = None  # required when display_mode is DELTA
    explicit_item_id: str | None = None  # required when delta_base is EXPLICIT_ITEM
    advances_anchor: bool = True


@dataclass(frozen=True)
class WalkViewSpec:
    metric_key: str
    steps: tuple[WalkStep, ...]
    target_value: float | None = None


@dataclass(frozen=True)
class WalkRow:
    item_id: str
    label: str
    display_mode: str  # "ABSOLUTE" | "DELTA"
    status: str  # "OK" | "UNAVAILABLE" | "INCOMPATIBLE" | "INVALID_CONFIG"
    absolute_value: float | None
    formatted_absolute_value: str
    delta_value: float | None
    formatted_delta: str | None
    delta_base_item_id: str | None
    delta_base_label: str | None
    semantic: str | None  # "BETTER" | "WORSE" | None
    advances_anchor: bool
    target_gap: TargetGap | None
    provenance: str | None
    role: str  # canonical ComparisonRole value ("REFERENCE" | "COMPARISON")
    presentation_role: str | None  # PresentationRole value, or None if no PresentationState was supplied
    is_current: bool


@dataclass(frozen=True)
class WalkResult:
    metric_key: str
    rows: tuple[WalkRow, ...]
    target_value: float | None
    has_delta_semantics: bool  # drives "KPI Walk" (True) vs "KPI Comparison" (False) hero title
    warnings: tuple[str, ...]  # unresolvable item_id / invalid delta base config -- never silently dropped


def _resolve_walk_delta_base(
    step: WalkStep,
    items_by_identity: Mapping[str, ComparisonItem],
    dataset: ComparisonDataset,
    active_anchor_item: ComparisonItem | None,
) -> tuple[ComparisonItem | None, str | None]:
    if step.delta_base is WalkDeltaBase.PREVIOUS_WALK_STATE:
        if active_anchor_item is None:
            return None, "DELTA vs previous walk state requested, but no prior step has advanced the anchor yet."
        return active_anchor_item, None
    if step.delta_base is WalkDeltaBase.REFERENCE:
        if dataset.reference is None:
            return None, "DELTA vs Reference requested, but no Reference is selected."
        return dataset.reference, None
    if step.delta_base is WalkDeltaBase.EXPLICIT_ITEM:
        base = items_by_identity.get(step.explicit_item_id) if step.explicit_item_id else None
        if base is None:
            return None, "Explicit delta base item is not part of the current selection."
        return base, None
    return None, "DELTA step is missing a delta_base."


def default_walk_steps(dataset: ComparisonDataset) -> tuple[WalkStep, ...]:
    """The safe default (Sec "SAFE DEFAULT"): when no Walk configuration
    exists, ALL selected items render ABSOLUTE, in dataset_items() order --
    never auto-creates a delta merely because an item is selected or tagged.
    """
    return tuple(
        WalkStep(canonical_identity(item), WalkDisplayMode.ABSOLUTE, advances_anchor=True) for item in dataset_items(dataset)
    )


def sequential_walk_steps(dataset: ComparisonDataset) -> tuple[WalkStep, ...]:
    """"Sequential Walk" preset (Sec "UI" presets): first item ABSOLUTE, every
    subsequent item DELTA vs the previous walk state, all advancing the
    anchor -- an explicit convenience action, never inferred lineage.
    """
    items = dataset_items(dataset)
    steps: list[WalkStep] = []
    for i, item in enumerate(items):
        identity = canonical_identity(item)
        if i == 0:
            steps.append(WalkStep(identity, WalkDisplayMode.ABSOLUTE, advances_anchor=True))
        else:
            steps.append(WalkStep(identity, WalkDisplayMode.DELTA, WalkDeltaBase.PREVIOUS_WALK_STATE, advances_anchor=True))
    return tuple(steps)


def delta_vs_reference_walk_steps(dataset: ComparisonDataset) -> tuple[WalkStep, ...]:
    """"Delta vs Reference" preset -- only meaningful when dataset.reference
    exists (the caller is expected to disable/hide this preset otherwise, per
    Sec "DELTA / REFERENCE: requires dataset.reference. If no Reference: mark
    unavailable / invalid configuration. Do not substitute anything.").
    """
    items = dataset_items(dataset)
    if not items:
        return ()
    anchor_identity = canonical_identity(items[0])
    steps: list[WalkStep] = [WalkStep(anchor_identity, WalkDisplayMode.ABSOLUTE, advances_anchor=False)]
    for item in items[1:]:
        steps.append(WalkStep(canonical_identity(item), WalkDisplayMode.DELTA, WalkDeltaBase.REFERENCE, advances_anchor=False))
    return tuple(steps)


def build_walk_rows(
    dataset: ComparisonDataset,
    spec: WalkViewSpec,
    *,
    presentation: PresentationState | None = None,
    unit_system: str = "Metric",
) -> WalkResult:
    """Sec: baseline/delta values are never recomputed from anything but
    compare_metric()/extract_metric_value() -- this function only sequences
    and labels what those already computed. A step referencing an item not in
    the current dataset, or a DELTA step whose base cannot be resolved, is
    reported via `warnings` and rendered with status="INVALID_CONFIG" -- it
    is never silently skipped or fabricated.
    """
    has_delta_semantics = any(step.display_mode is WalkDisplayMode.DELTA for step in spec.steps)
    metric = get_metric(spec.metric_key)
    if metric is None:
        return WalkResult(
            metric_key=spec.metric_key,
            rows=(),
            target_value=spec.target_value,
            has_delta_semantics=has_delta_semantics,
            warnings=("Unknown Primary KPI metric.",),
        )

    items_by_identity = {canonical_identity(item): item for item in dataset_items(dataset)}
    warnings: list[str] = []
    rows: list[WalkRow] = []
    active_anchor_item: ComparisonItem | None = None

    for step in spec.steps:
        item = items_by_identity.get(step.item_id)
        if item is None:
            warnings.append(f"Selected item not found in the current dataset: {step.item_id}")
            continue

        absolute_raw = extract_metric_value(item, spec.metric_key)
        label = item.label or "Unknown vehicle"

        delta_value: float | None = None
        formatted_delta: str | None = None
        delta_base_identity: str | None = None
        delta_base_label: str | None = None
        semantic: str | None = None

        if step.display_mode is WalkDisplayMode.ABSOLUTE:
            status = "OK" if absolute_raw is not None else "UNAVAILABLE"
        else:
            base_item, base_warning = _resolve_walk_delta_base(step, items_by_identity, dataset, active_anchor_item)
            if base_item is None:
                status = "INVALID_CONFIG"
                warnings.append(f"{label}: {base_warning}")
            else:
                delta_base_identity = canonical_identity(base_item)
                delta_base_label = base_item.label or "Unknown vehicle"
                if absolute_raw is None:
                    status = "UNAVAILABLE"
                else:
                    result = compare_metric(base_item, item, spec.metric_key)
                    if not result["compatible"]:
                        status = "INCOMPATIBLE"
                    elif not result["available"]:
                        status = "UNAVAILABLE"
                    else:
                        status = "OK"
                        delta_value = result["absolute_delta"]
                        formatted_delta = _format_delta(
                            result["absolute_delta"], result["percent_delta"], metric.unit_family, unit_system
                        )
                        semantic = _semantic_for_display(result["semantic"])

        rows.append(
            WalkRow(
                item_id=step.item_id,
                label=label,
                display_mode=step.display_mode.value,
                status=status,
                absolute_value=absolute_raw,
                formatted_absolute_value=format_value(absolute_raw, metric.unit_family, unit_system),
                delta_value=delta_value,
                formatted_delta=formatted_delta,
                delta_base_item_id=delta_base_identity,
                delta_base_label=delta_base_label,
                semantic=semantic,
                advances_anchor=step.advances_anchor,
                target_gap=evaluate_target_gap(spec.metric_key, absolute_raw, spec.target_value),
                provenance=item.provenance.record_origin,
                role=item.role.value,
                presentation_role=(presentation_role_for(presentation, step.item_id).value if presentation is not None else None),
                is_current=(is_current_item(presentation, step.item_id) if presentation is not None else False),
            )
        )
        if step.advances_anchor and status == "OK":
            active_anchor_item = item

    return WalkResult(
        metric_key=spec.metric_key,
        rows=tuple(rows),
        target_value=spec.target_value,
        has_delta_semantics=has_delta_semantics,
        warnings=tuple(warnings),
    )


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


def metric_axis_label(metric: MetricDefinition, unit_system: str) -> str:
    """Single source of truth for "Label [unit]" axis titles (Explore/Lineage,
    Package 8D) -- reuses the same unit_family -> quantity mapping format_value
    already relies on, so a chart axis and a Scorecard cell never disagree.
    """
    if metric.unit_family == "l_per_100km":
        unit = "gal/100mi" if unit_system == "US customary" else "L/100km"
        return f"{metric.label} [{unit}]"
    if metric.unit_family == "km_per_l":
        return f"{metric.label} [km/l]"
    quantity = _UNIT_QUANTITY_MAP.get(metric.unit_family)
    if quantity:
        return f"{metric.label} [{_unit_label(quantity, unit_system)}]"
    return metric.label


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


def _absolute_cell(item: ComparisonItem, metric_key: str, unit_family: str, unit_system: str) -> ScorecardCell:
    """A Reference-less cell: the item's own value, no delta/semantic --
    compare_metric() is a pairwise primitive and is never called with a
    fabricated or None Reference (Package 8F Increment 1).
    """
    value = extract_metric_value(item, metric_key)
    return ScorecardCell(
        raw_value=value,
        formatted_value=format_value(value, unit_family, unit_system),
        absolute_delta=None,
        formatted_delta=None,
        percent_delta=None,
        semantic=None,
        compatible=True,
        available=value is not None,
        basis_mismatch=False,
        warning=None,
    )


def _metric_row(dataset: ComparisonDataset, metric_key: str, label: str, unit_family: str, unit_system: str) -> ScorecardRow:
    if dataset.reference is None:
        # No Reference selected: every item is ABSOLUTE only, never a
        # fabricated delta. The first item fills ScorecardRow's mandatory
        # reference_cell slot purely for shape -- it carries no Reference
        # meaning; build_scenario_header only marks a column REFERENCE from
        # item.role, and no item holds ComparisonRole.REFERENCE here.
        cells = [_absolute_cell(item, metric_key, unit_family, unit_system) for item in dataset.comparisons]
        return ScorecardRow(metric_key=metric_key, label=label, reference_cell=cells[0], comparison_cells=tuple(cells[1:]))

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
    items = dataset_items(dataset)
    rows = []
    for field_name, label in _PROVENANCE_ROWS:
        reference_cell = _provenance_cell(_provenance_value(items[0], field_name))
        comparison_cells = tuple(_provenance_cell(_provenance_value(item, field_name)) for item in items[1:])
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

    if dataset.reference is None:
        # No Reference: every item is an absolute bar, no delta/semantic
        # (Package 8F Increment 1) -- never fabricate a comparison base.
        for item in dataset.comparisons:
            label = item.label or "Unknown vehicle"
            value = extract_metric_value(item, metric_key)
            if value is None:
                excluded.append({"label": label, "reason": f"{metric.label} unavailable"})
                continue
            rows.append(
                BarRow(label=label, value=value, formatted_value=format_value(value, metric.unit_family, unit_system), semantic=None)
            )
        return {"rows": rows, "excluded": excluded}

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


def canonical_identity(item: ComparisonItem) -> str:
    """Sec 10: chart-preparation dictionary keys must be canonical IDs, never
    display labels -- two distinct VDE_ONLY items can legitimately share an
    identical label, unlike _scenario_identity() above (which falls back to
    label and exists only for the Roadload dedup-attribution UI text).
    """
    if item.fuelcons_id is not None:
        return f"fc:{item.fuelcons_id}"
    return f"vde:{item.vde_id}"


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
    groups = deduplicate_by_vde_id(dataset_items(dataset))
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
    groups = deduplicate_by_vde_id(dataset_items(dataset))
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for group in groups:
        rb = group.item.roadload[boundary_key]
        if not rb.available:
            excluded.append({"label": group.label, "reason": f"Roadload {boundary} unavailable"})
            continue
        rows.append({"label": group.label, "A_N": rb.A, "B_N_per_kph": rb.B, "C_N_per_kph2": rb.C})
    return {"rows": rows, "excluded": excluded}


# -----------------------------------------------------------------------------
# True Cycle/Phase VDE (Package 8F Increment 7)
#
# The audit found VDEBoundaryResult.by_phase already carries real per-phase
# VDE data (EPA: city/hwy; WLTP: low/mid/high/xhigh) that no prior package
# ever read -- the old "Cycle / phase VDE" section only ever plotted the
# TOTAL/NET aggregate. This reads by_phase directly (no new VDE computation,
# no new physics) and groups items by the phase-key family their own data
# actually uses -- EPA and WLTP items are never merged into one chart, and an
# item with neither recognizable family, or incomplete phase data, is
# excluded with a reason rather than zero-filled or guessed.
# -----------------------------------------------------------------------------

_EPA_PHASES: tuple[str, ...] = ("city", "hwy")
_EPA_PHASE_LABELS = {"city": "City", "hwy": "Highway"}
_WLTP_PHASES: tuple[str, ...] = ("low", "mid", "high", "xhigh")
_WLTP_PHASE_LABELS = {"low": "Low", "mid": "Mid", "high": "High", "xhigh": "Extra High"}


def _phase_family_for(item: ComparisonItem, boundary_key: str) -> tuple[str, ...] | None:
    by_phase = item.vde["cycle_results"][boundary_key].by_phase
    keys = set(by_phase.keys())
    if not keys:
        return None
    if keys.issubset(set(_EPA_PHASES)):
        return _EPA_PHASES
    if keys.issubset(set(_WLTP_PHASES)):
        return _WLTP_PHASES
    return None


def build_cycle_phase_rows(dataset: ComparisonDataset, boundary: str) -> dict[str, Any]:
    """Returns {"families": [{"family": "EPA"|"WLTP", "rows": [...]}], "excluded": [...]}.
    `rows` entries are {"label", "value", "group"} -- group is the phase's
    display label (City/Highway or Low/Mid/High/Extra High), ready for
    build_grouped_bar_figure. An item is excluded (never zero-filled) when it
    has no by_phase data, or when its by_phase doesn't fully cover its own
    family's phase keys.
    """
    boundary_key = boundary.lower()
    groups: dict[tuple[str, ...], list[tuple[ComparisonItem, Mapping[str, float]]]] = {}
    excluded: list[dict[str, str]] = []
    for item in dataset_items(dataset):
        label = item.label or "Unknown vehicle"
        family = _phase_family_for(item, boundary_key)
        by_phase = item.vde["cycle_results"][boundary_key].by_phase
        if family is None:
            excluded.append({"label": label, "reason": f"No recognized phase breakdown for {boundary}"})
            continue
        if not all(phase_key in by_phase for phase_key in family):
            excluded.append({"label": label, "reason": f"Incomplete phase data for {boundary}"})
            continue
        groups.setdefault(family, []).append((item, by_phase))

    families: list[dict[str, Any]] = []
    for family, entries in groups.items():
        phase_labels = _EPA_PHASE_LABELS if family == _EPA_PHASES else _WLTP_PHASE_LABELS
        rows = [
            {"label": item.label or "Unknown vehicle", "value": by_phase[phase_key], "group": phase_labels[phase_key]}
            for item, by_phase in entries
            for phase_key in family
        ]
        families.append({"family": "EPA" if family == _EPA_PHASES else "WLTP", "rows": rows})

    return {"families": families, "excluded": excluded}


def build_cycle_demand_rows(dataset: ComparisonDataset, cycle_frame, boundaries: Sequence[str]) -> dict[str, Any]:
    """Thin adapter: builds the `scenarios` list roadload_analysis.build_cycle_power_analysis
    expects from deduped ComparisonItems, then returns its `series` unmodified --
    the power/energy math itself is never re-implemented here.
    """
    from src.vde_core.roadload_analysis import build_cycle_power_analysis

    groups = deduplicate_by_vde_id(dataset_items(dataset))
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
# FE x VDE / equi-PSE and competitor delta (Sec 15-21, Package 8C; fuel-basis
# resolution added in Package 8F)
#
# LHV/energy-basis resolution is delegated entirely to
# fuel_energy.resolve_fuel_energy_basis() -- this module never redefines or
# guesses an LHV itself. That resolver reuses fuel_energy.py::LHV_MJ_PER_L as
# the sole canonical numeric source (never derivatives.py's or plots.py's
# duplicate 34.2 values) and never SILENTLY guesses: an unrecognized raw
# fuel_type (Flex, an unmapped blend, CNG/LPG/Hydrogen, empty/None) always
# comes back unavailable, never a fabricated value.
# -----------------------------------------------------------------------------


def is_temporary_net(item: ComparisonItem) -> bool:
    return "temporary_transmission_used" in item.warnings


def _consumed_energy_mj_per_km(fuel: Mapping[str, Any]) -> float | None:
    if fuel.get("electrification") == "BEV":
        wh = fuel.get("energy_Wh_per_km")
        return (wh / MJ_TO_Wh) if wh is not None else None
    basis = resolve_fuel_energy_basis(fuel.get("fuel_type"))
    l100 = fuel.get("fuel_l_per_100km")
    if not basis.available or l100 is None:
        return None
    return (l100 / 100.0) * basis.lhv_mj_per_l


def build_fe_vde_points(dataset: ComparisonDataset, *, boundary: str, mode: str) -> dict[str, list]:
    """mode in {"volumetric", "energy_normalized", "electrical"}. x is always
    item.vde["aggregate"][boundary] -- never a fallback boundary. Excluded
    points always carry a reason (Sec 16, 41); nothing is silently dropped.

    When dataset.reference is None (Package 8F), the first selected item
    (dataset_items order) is used only as the volumetric-mode fuel-family
    anchor for internal consistency -- it is never marked with the REFERENCE
    role/star, which the chart layer derives purely from item.role. The PSE
    energy-basis disclosure and the equi-PSE guide lines the chart layer
    draws are deliberately tied to this same anchor (Reference, when one is
    set) and stay absent -- with an explicit, non-crashing explanation, never
    a silent gap -- when the anchor's own fuel type isn't LHV-mappable (e.g.
    Flex), even if another plotted comparison happens to resolve to a known
    family: the guide lines assert an efficiency basis for the anchor's own
    context, so they must not be inferred from a different scenario the
    analyst didn't anchor the comparison on.
    """
    boundary_key = boundary.lower()
    items = dataset_items(dataset)
    anchor = items[0] if items else None
    anchor_basis = (
        resolve_fuel_energy_basis((anchor.fuel_energy or {}).get("fuel_type")) if anchor is not None else None
    )
    # The volumetric-mode family INCLUSION check (which points are even
    # plottable) is judged against whichever item first establishes a
    # resolvable fuel family, not necessarily `anchor` itself -- an
    # unmappable anchor (e.g. a Flex Reference) must exclude only itself,
    # never poison every other item's family comparison to "no match". This
    # is deliberately independent from the guide-line/assumption-disclosure
    # basis below, which stays anchor-specific (see docstring).
    established_family: str | None = None
    for _item in items:
        _basis = resolve_fuel_energy_basis((_item.fuel_energy or {}).get("fuel_type"))
        if _basis.available:
            established_family = _basis.canonical_fuel_family
            break
    points: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    for item in items:
        label = item.label or "Unknown vehicle"
        x = item.vde["aggregate"].get(boundary_key)
        if x is None:
            excluded.append({"label": label, "reason": f"VDE {boundary} unavailable"})
            continue

        fuel = item.fuel_energy or {}
        fuel_basis_label: str | None = None
        if mode == "volumetric":
            item_basis = resolve_fuel_energy_basis(fuel.get("fuel_type"))
            if not item_basis.available:
                raw = fuel.get("fuel_type") or "unknown"
                excluded.append({"label": label, "reason": f"No LHV assumption available for fuel '{raw}'"})
                continue
            # Compatibility is judged on the RESOLVED canonical family, never
            # the raw label string -- two labels that differ only by case or
            # certification wording (e.g. "GASOLINE" vs "Tier 2 Cert
            # Gasoline") are the same family and must not be treated as a
            # fuel-family mismatch.
            if established_family is not None and item_basis.canonical_fuel_family != established_family:
                excluded.append({"label": label, "reason": "Different fuel family - use energy-normalized mode"})
                continue
            y = fuel.get("fuel_l_per_100km")
            if y is None:
                excluded.append({"label": label, "reason": "Fuel consumption unavailable"})
                continue
            fuel_basis_label = item_basis.basis_label
        elif mode == "energy_normalized":
            y = _consumed_energy_mj_per_km(fuel)
            if y is None:
                excluded.append({"label": label, "reason": "Fuel blend not resolvable to an energy basis"})
                continue
            if fuel.get("electrification") != "BEV":
                fuel_basis_label = resolve_fuel_energy_basis(fuel.get("fuel_type")).basis_label
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
                "fuel_basis_label": fuel_basis_label,
            }
        )

    assumption_label = None
    anchor_fuel_type = anchor_basis.raw_fuel_label if anchor_basis is not None else None
    if mode == "volumetric" and anchor_basis is not None and anchor_basis.lhv_basis in (
        LhvBasis.CANONICAL_ASSUMPTION,
        LhvBasis.REGIONAL_ASSUMPTION,
    ):
        assumption_label = anchor_basis.basis_label
    return {
        "points": points,
        "excluded": excluded,
        "assumption_label": assumption_label,
        "anchor_fuel_type": anchor_fuel_type,
    }


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
        basis = resolve_fuel_energy_basis(fuel_type)
        if not basis.available:
            return []
        lhv = basis.lhv_mj_per_l
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


# -----------------------------------------------------------------------------
# Adaptive equi-PSE guide values (Sprint 8 micro-polish)
#
# Presentation-only sizing of which PSE contours to draw -- reuses the exact
# same PSE ratio build_iso_pse_lines() already draws from (never a new
# physics definition), and the same anchor-specific fuel basis rule: no
# adaptive guides are fabricated when that basis isn't defensible for the
# mode/fuel_type, matching build_iso_pse_lines()'s own empty-list behavior.
# -----------------------------------------------------------------------------


def _pse_for_point(x: Any, y: Any, *, mode: str, lhv: float | None) -> float | None:
    if x is None or y is None:
        return None
    x = float(x)
    y = float(y)
    if x <= 0 or y <= 0:
        return None
    if mode == "volumetric":
        if lhv is None or lhv <= 0:
            return None
        consumed_mj_per_km = (y / 100.0) * lhv
        return x / consumed_mj_per_km if consumed_mj_per_km > 0 else None
    if mode == "energy_normalized":
        return x / y
    if mode == "electrical":
        return x * MJ_TO_Wh / y
    return None


def _nice_pse_guides(pse_values: Sequence[float]) -> tuple[float, ...]:
    """Snaps the actual [min, max] PSE span to a clean 2.5/5/10/20
    percentage-point grid, padding by one grid step on each side so a
    single/narrow point still gets surrounding context rather than sitting
    exactly on the chart edge, and keeps the result to 3-5 guides.
    """
    lo, hi = min(pse_values), max(pse_values)
    for step_pct in (2.5, 5.0, 10.0, 20.0):
        step = step_pct / 100.0
        first = math.floor(lo / step) * step
        last = math.ceil(hi / step) * step
        if first >= lo:
            first -= step
        if last <= hi:
            last += step
        count = round((last - first) / step) + 1
        while count < 3:
            first -= step
            last += step
            count = round((last - first) / step) + 1
        if count <= 5:
            return tuple(round(first + i * step, 6) for i in range(count))
    step = 0.20
    first = math.floor(lo / step) * step
    return tuple(round(first + i * step, 6) for i in range(5))


def compute_adaptive_pse_guides(
    points: Sequence[Mapping[str, Any]], *, mode: str, fuel_type: str | None = None
) -> tuple[float, ...]:
    """Equi-PSE guide values sized to what's actually plotted, replacing a
    fixed guide set with one centered on the plotted scenarios' own PSE
    range. Returns () -- no fabricated guides -- when the mode/fuel_type
    basis can't support a defensible line at all (mirrors
    build_iso_pse_lines()'s own rule) or when no plotted point yields a
    computable PSE.
    """
    lhv = None
    if mode == "volumetric":
        basis = resolve_fuel_energy_basis(fuel_type)
        if not basis.available:
            return ()
        lhv = basis.lhv_mj_per_l
    elif mode not in ("energy_normalized", "electrical"):
        return ()

    pse_values = [
        pse
        for pse in (_pse_for_point(p.get("x"), p.get("y"), mode=mode, lhv=lhv) for p in points)
        if pse is not None
    ]
    if not pse_values:
        return ()
    return _nice_pse_guides(pse_values)


def build_competitor_delta_rows(
    dataset: ComparisonDataset, metric_key: str
) -> dict[str, list]:
    """Reference is fixed at 0%/no-verdict; comparisons come straight from
    compare_metric() -- delta semantics are never recomputed here (Sec 20-21).
    Reference-relative delta is meaningless without a Reference (Package 8F):
    returns no rows at all rather than substituting any other item as a base.
    """
    metric = get_metric(metric_key)
    if metric is None or dataset.reference is None:
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
    items = dataset_items(dataset)
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


# -----------------------------------------------------------------------------
# Explore Lite -- generic chart data prep (Sec 3-25, Package 8D)
#
# Consumes ComparisonDataset + Metric Registry only. Never exposes raw SQLite
# fields, arbitrary expressions, or a second hardcoded KPI list -- numeric axes
# come from comparison_metric_registry.list_metrics(); categorical dimensions
# come from the small curated table below (Sec 11), since Scenario/Vehicle/
# Provenance have no Registry entry of their own but are still legitimate,
# already-computed ComparisonItem fields.
# -----------------------------------------------------------------------------


def _dim_vehicle(item: ComparisonItem) -> str | None:
    make = str(item.vehicle.get("make") or "").strip()
    model = str(item.vehicle.get("model") or "").strip()
    return " ".join(part for part in (make, model) if part) or None


@dataclass(frozen=True)
class ExploreDimension:
    key: str
    label: str
    extractor: Callable[[ComparisonItem], Any]
    roles: frozenset[str]  # subset of {"x", "order", "group", "filter"}


_EXPLORE_DIMENSIONS: tuple[ExploreDimension, ...] = (
    ExploreDimension("scenario", "Scenario", lambda i: i.label or f"VDE #{i.vde_id}", frozenset({"x"})),
    ExploreDimension("vehicle", "Vehicle", _dim_vehicle, frozenset({"x"})),
    ExploreDimension("make", "Make", lambda i: i.vehicle.get("make"), frozenset({"x"})),
    ExploreDimension("model_year", "Model Year", lambda i: i.vehicle.get("year"), frozenset({"x", "order"})),
    ExploreDimension("category", "Category", lambda i: i.vehicle.get("category"), frozenset({"x", "group", "filter"})),
    ExploreDimension(
        "legislation", "Legislation", lambda i: i.vehicle.get("legislation"), frozenset({"x", "group", "filter"})
    ),
    ExploreDimension(
        "electrification",
        "Electrification",
        lambda i: i.powertrain.get("electrification"),
        frozenset({"x", "group", "filter"}),
    ),
    ExploreDimension(
        "fuel_type", "Fuel type", lambda i: i.powertrain.get("fuel_type"), frozenset({"x", "group", "filter"})
    ),
    ExploreDimension(
        "provenance", "Provenance", lambda i: i.provenance.record_origin, frozenset({"x", "group", "filter"})
    ),
)
_EXPLORE_DIMENSIONS_BY_KEY = {d.key: d for d in _EXPLORE_DIMENSIONS}


def _dimension_by_key(key: str | None) -> ExploreDimension | None:
    return _EXPLORE_DIMENSIONS_BY_KEY.get(key) if key else None


def list_explore_dimensions(role: str) -> list[ExploreDimension]:
    return [d for d in _EXPLORE_DIMENSIONS if role in d.roles]


def list_explore_numeric_metrics(chart_type: str) -> list[MetricDefinition]:
    """chart_type in {"bar", "scatter", "line"}. Line reuses Bar-compatible
    metrics -- a Line chart is a Bar chart with an explicit ordering basis
    (Sec 17), not a distinct metric class.
    """
    lookup_type = "bar" if chart_type == "line" else chart_type
    return [m for m in list_metrics() if m.unit_family not in ("text",) and lookup_type in m.compatible_chart_types]


def list_available_explore_metrics(items: Sequence[ComparisonItem], chart_type: str) -> list[MetricDefinition]:
    """Sec 8: a metric with zero available values across the current items is
    not offered at all; partial availability is fine (excluded items are
    reported per-row by the chart builders below).
    """
    candidates = list_explore_numeric_metrics(chart_type)
    return [m for m in candidates if any(extract_metric_value(item, m.key) is not None for item in items)]


def list_explore_dimension_values(items: Sequence[ComparisonItem], dimension_key: str) -> list[str]:
    dim = _dimension_by_key(dimension_key)
    if dim is None:
        return []
    values = {str(dim.extractor(item)) for item in items if dim.extractor(item) not in (None, "")}
    return sorted(values)


def _dedupe_display_labels(labels: list[str]) -> list[str]:
    """Presentation-only disambiguation equivalent to components/comparison_report.py's
    _dedupe_titles (Sec 10) -- kept local so this module stays Streamlit-free
    and dependency-direction-clean (UI imports viewmodels, never the reverse).
    Two distinct scenarios sharing one label must never be merged into one bar.
    """
    seen: dict[str, int] = {}
    result = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        result.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return result


def _apply_dimension_filter(
    items: Sequence[ComparisonItem], filter_dimension_key: str | None, filter_values: Sequence[str]
) -> tuple[list[ComparisonItem], list[dict[str, str]]]:
    if not filter_dimension_key or not filter_values:
        return list(items), []
    dim = _dimension_by_key(filter_dimension_key)
    if dim is None:
        return list(items), []
    allowed = {str(v) for v in filter_values}
    kept: list[ComparisonItem] = []
    excluded: list[dict[str, str]] = []
    for item in items:
        value = dim.extractor(item)
        if value is not None and str(value) in allowed:
            kept.append(item)
        else:
            excluded.append({"label": item.label or "Unknown vehicle", "reason": f"Filtered out by {dim.label}"})
    return kept, excluded


@dataclass(frozen=True)
class ExploreBarRow:
    identity: str
    label: str
    value: float
    formatted_value: str
    role: str
    group: str | None = None


@dataclass(frozen=True)
class ExploreScatterPoint:
    identity: str
    label: str
    role: str
    x: float
    y: float
    provenance: str | None
    is_temporary_net: bool
    revision_status: str | None
    group: str | None = None


@dataclass(frozen=True)
class ExploreLineRow:
    identity: str
    x: Any
    x_label: str
    y: float
    formatted_y: str
    role: str
    group: str | None = None


def build_explore_bar_rows(
    dataset: ComparisonDataset,
    *,
    x_dimension_key: str,
    y_metric_key: str,
    group_dimension_key: str | None = None,
    filter_dimension_key: str | None = None,
    filter_values: Sequence[str] = (),
    unit_system: str = "Metric",
) -> dict[str, Any]:
    metric = get_metric(y_metric_key)
    x_dim = _dimension_by_key(x_dimension_key)
    if metric is None or x_dim is None or "bar" not in metric.compatible_chart_types:
        return {"rows": [], "excluded": []}

    items, filter_excluded = _apply_dimension_filter(
        dataset_items(dataset), filter_dimension_key, filter_values
    )
    group_dim = _dimension_by_key(group_dimension_key) if group_dimension_key else None

    excluded: list[dict[str, str]] = list(filter_excluded)
    kept_items: list[ComparisonItem] = []
    labels: list[str] = []
    for item in items:
        x_value = x_dim.extractor(item)
        if x_value in (None, ""):
            excluded.append({"label": item.label or "Unknown vehicle", "reason": f"{x_dim.label} unavailable"})
            continue
        value = extract_metric_value(item, y_metric_key)
        if value is None:
            excluded.append({"label": item.label or "Unknown vehicle", "reason": f"{metric.label} unavailable"})
            continue
        kept_items.append(item)
        labels.append(str(x_value))

    labels = _dedupe_display_labels(labels)
    rows: list[ExploreBarRow] = []
    for item, label in zip(kept_items, labels):
        value = extract_metric_value(item, y_metric_key)
        group_value = group_dim.extractor(item) if group_dim is not None else None
        rows.append(
            ExploreBarRow(
                identity=canonical_identity(item),
                label=label,
                value=value,
                formatted_value=format_value(value, metric.unit_family, unit_system),
                role=item.role.value,
                group=str(group_value) if group_value not in (None, "") else None,
            )
        )
    return {"rows": rows, "excluded": excluded}


def build_explore_scatter_points(
    dataset: ComparisonDataset,
    *,
    x_metric_key: str,
    y_metric_key: str,
    group_dimension_key: str | None = None,
    filter_dimension_key: str | None = None,
    filter_values: Sequence[str] = (),
) -> dict[str, Any]:
    x_metric = get_metric(x_metric_key)
    y_metric = get_metric(y_metric_key)
    if (
        x_metric is None
        or y_metric is None
        or "scatter" not in x_metric.compatible_chart_types
        or "scatter" not in y_metric.compatible_chart_types
    ):
        return {"points": [], "excluded": []}

    items, excluded = _apply_dimension_filter(dataset_items(dataset), filter_dimension_key, filter_values)
    group_dim = _dimension_by_key(group_dimension_key) if group_dimension_key else None

    points: list[ExploreScatterPoint] = []
    for item in items:
        label = item.label or "Unknown vehicle"
        x = extract_metric_value(item, x_metric_key)
        y = extract_metric_value(item, y_metric_key)
        missing = [m.label for m, v in ((x_metric, x), (y_metric, y)) if v is None]
        if missing:
            excluded.append({"label": label, "reason": f"{' / '.join(missing)} unavailable"})
            continue
        group_value = group_dim.extractor(item) if group_dim is not None else None
        points.append(
            ExploreScatterPoint(
                identity=canonical_identity(item),
                label=label,
                role=item.role.value,
                x=x,
                y=y,
                provenance=item.provenance.record_origin,
                is_temporary_net=is_temporary_net(item),
                revision_status=item.provenance.revision_status.value if item.provenance.revision_status else None,
                group=str(group_value) if group_value not in (None, "") else None,
            )
        )
    return {"points": points, "excluded": excluded}


def build_explore_line_rows(
    dataset: ComparisonDataset,
    *,
    x_dimension_key: str,
    y_metric_key: str,
    group_dimension_key: str | None = None,
    filter_dimension_key: str | None = None,
    filter_values: Sequence[str] = (),
    unit_system: str = "Metric",
) -> dict[str, Any]:
    """Sec 17: Line X must come from an explicitly orderable dimension --
    list_explore_dimensions("order") only ever offers Model Year today, so an
    unordered choice (e.g. selection order) is structurally unavailable
    rather than merely discouraged.
    """
    x_dim = _dimension_by_key(x_dimension_key)
    metric = get_metric(y_metric_key)
    if x_dim is None or "order" not in x_dim.roles or metric is None or "bar" not in metric.compatible_chart_types:
        return {"rows": [], "excluded": [], "unavailable_reason": "Select a valid ordered X dimension for Line charts."}

    items, excluded = _apply_dimension_filter(dataset_items(dataset), filter_dimension_key, filter_values)
    group_dim = _dimension_by_key(group_dimension_key) if group_dimension_key else None

    prepared: list[tuple[Any, ComparisonItem]] = []
    for item in items:
        x_value = x_dim.extractor(item)
        if x_value in (None, ""):
            excluded.append({"label": item.label or "Unknown vehicle", "reason": f"{x_dim.label} unavailable"})
            continue
        y_value = extract_metric_value(item, y_metric_key)
        if y_value is None:
            excluded.append({"label": item.label or "Unknown vehicle", "reason": f"{metric.label} unavailable"})
            continue
        prepared.append((x_value, item))

    prepared.sort(key=lambda pair: (pair[0], canonical_identity(pair[1])))

    rows: list[ExploreLineRow] = []
    for x_value, item in prepared:
        y_value = extract_metric_value(item, y_metric_key)
        group_value = group_dim.extractor(item) if group_dim is not None else None
        rows.append(
            ExploreLineRow(
                identity=canonical_identity(item),
                x=x_value,
                x_label=str(x_value),
                y=y_value,
                formatted_y=format_value(y_value, metric.unit_family, unit_system),
                role=item.role.value,
                group=str(group_value) if group_value not in (None, "") else None,
            )
        )
    return {"rows": rows, "excluded": excluded, "unavailable_reason": None}


# -----------------------------------------------------------------------------
# Physical VDE Lineage (Sec 26-43, Package 8D)
#
# The only explicit lineage source in this repository is vde_db.vde_id_parent
# (confirmed by inspection: fuelcons_db has no parent field of its own). This
# is therefore always Physical VDE Lineage. A FuelCons scenario may enter this
# view by resolving its linked VDE, but the chain identity is the VDE id chain
# -- the originating scenario is kept only as display context (Package 8D
# Investigation Addendum, replacing the original "FuelCons scenario lineage
# uses FuelCons identity" requirement).
# -----------------------------------------------------------------------------

_LINEAGE_INELIGIBLE_SOURCE_REQUIREMENTS = frozenset({"FUEL_CONSUMPTION", "FUEL_ENERGY", "ELECTRICAL_ENERGY", "CO2", "PSE"})


@dataclass(frozen=True)
class LineageContext:
    chain: LineageChainResult
    originating_label: str
    originating_identity: str
    is_fuelcons_scenario: bool


def resolve_lineage_context(item: ComparisonItem) -> LineageContext | None:
    if item.vde_id is None:
        return None
    return LineageContext(
        chain=resolve_lineage_chain(item.vde_id),
        originating_label=item.label or f"VDE #{item.vde_id}",
        originating_identity=_scenario_identity(item),
        is_fuelcons_scenario=item.fuelcons_id is not None,
    )


def list_lineage_capable_metrics() -> list[MetricDefinition]:
    """A lineage chain's nodes are always VDE_ONLY items (Sec 30) -- fuel/
    energy/CO2/PSE metrics require a FuelCons scenario and are never
    populated on a bare VDE, so offering them here would always resolve to
    "unavailable at every node" rather than a usable waterfall.
    """
    return [
        m
        for m in list_metrics()
        if m.unit_family not in ("text",) and m.source_requirement not in _LINEAGE_INELIGIBLE_SOURCE_REQUIREMENTS
    ]


def list_available_lineage_metrics(chain: LineageChainResult) -> list[MetricDefinition]:
    """Sec 31: only offer a metric that every required chain node can
    legitimately expose -- a metric missing at even one node is excluded
    from the selector rather than offered and then failing mid-walk.
    """
    if not chain.nodes:
        return []
    items = [build_vde_comparison_item(node.vde_id, vde_row=node.vde_row) for node in chain.nodes]
    return [m for m in list_lineage_capable_metrics() if all(extract_metric_value(i, m.key) is not None for i in items)]


@dataclass(frozen=True)
class LineageStep:
    vde_id: int
    label: str
    parent_vde_id: int | None
    provenance: str | None
    value: float | None
    formatted_value: str
    delta: float | None
    formatted_delta: str | None
    semantic: str | None  # "BETTER" | "WORSE" | None
    status: str  # "OK" | "UNAVAILABLE" | "INCOMPATIBLE"


@dataclass(frozen=True)
class LineageWaterfallResult:
    metric_key: str
    steps: tuple[LineageStep, ...]
    complete: bool
    incomplete_reason: str | None
    chain_status: str
    chain_warnings: tuple[str, ...]


def build_lineage_waterfall(
    chain: LineageChainResult,
    metric_key: str,
    *,
    unit_system: str = "Metric",
    temporary_transmission_by_vde_id: Mapping[int, Mapping[str, Any]] | None = None,
) -> LineageWaterfallResult:
    """Sec 32: baseline = absolute root value; each subsequent step = child
    value - parent value (never recomputed from anything but compare_metric,
    so semantics/compatibility/basis rules are never duplicated). Sec 35-36:
    the first unavailable or incompatible node truncates the walk -- no
    fallback, no fabricated continuation.
    """
    metric = get_metric(metric_key)
    if metric is None or not chain.nodes:
        return LineageWaterfallResult(
            metric_key=metric_key,
            steps=(),
            complete=False,
            incomplete_reason="Metric not available." if metric is None else "No lineage chain resolved.",
            chain_status=chain.status.value,
            chain_warnings=chain.warnings,
        )

    temp_by_vde = temporary_transmission_by_vde_id or {}
    items = [
        build_vde_comparison_item(node.vde_id, vde_row=node.vde_row, temporary_transmission=temp_by_vde.get(node.vde_id))
        for node in chain.nodes
    ]

    baseline_value = extract_metric_value(items[0], metric_key)
    if baseline_value is None:
        return LineageWaterfallResult(
            metric_key=metric_key,
            steps=(),
            complete=False,
            incomplete_reason=f"{metric.label} unavailable at the root of this chain.",
            chain_status=chain.status.value,
            chain_warnings=chain.warnings,
        )

    steps: list[LineageStep] = [
        LineageStep(
            vde_id=chain.nodes[0].vde_id,
            label=chain.nodes[0].label,
            parent_vde_id=None,
            provenance=items[0].provenance.record_origin,
            value=baseline_value,
            formatted_value=format_value(baseline_value, metric.unit_family, unit_system),
            delta=None,
            formatted_delta=None,
            semantic=None,
            status="OK",
        )
    ]

    complete = True
    incomplete_reason: str | None = None
    for i in range(1, len(items)):
        node = chain.nodes[i]
        result = compare_metric(items[i - 1], items[i], metric_key)
        if not result["compatible"]:
            complete = False
            incomplete_reason = f"{node.label}: different cycle / incompatible basis vs its parent."
            steps.append(
                LineageStep(
                    vde_id=node.vde_id,
                    label=node.label,
                    parent_vde_id=node.parent_vde_id,
                    provenance=items[i].provenance.record_origin,
                    value=None,
                    formatted_value="-",
                    delta=None,
                    formatted_delta=None,
                    semantic=None,
                    status="INCOMPATIBLE",
                )
            )
            break
        if not result["available"]:
            complete = False
            incomplete_reason = f"{node.label}: {metric.label} unavailable."
            steps.append(
                LineageStep(
                    vde_id=node.vde_id,
                    label=node.label,
                    parent_vde_id=node.parent_vde_id,
                    provenance=items[i].provenance.record_origin,
                    value=None,
                    formatted_value="-",
                    delta=None,
                    formatted_delta=None,
                    semantic=None,
                    status="UNAVAILABLE",
                )
            )
            break
        steps.append(
            LineageStep(
                vde_id=node.vde_id,
                label=node.label,
                parent_vde_id=node.parent_vde_id,
                provenance=items[i].provenance.record_origin,
                value=result["comparison_value"],
                formatted_value=format_value(result["comparison_value"], metric.unit_family, unit_system),
                delta=result["absolute_delta"],
                formatted_delta=_format_delta(result["absolute_delta"], result["percent_delta"], metric.unit_family, unit_system),
                semantic=_semantic_for_display(result["semantic"]),
                status="OK",
            )
        )

    if complete and chain.status in (LineageChainStatus.BROKEN, LineageChainStatus.MALFORMED):
        complete = False
        incomplete_reason = "Ancestry walk stopped: " + (chain.warnings[-1] if chain.warnings else chain.status.value)

    return LineageWaterfallResult(
        metric_key=metric_key,
        steps=tuple(steps),
        complete=complete,
        incomplete_reason=incomplete_reason,
        chain_status=chain.status.value,
        chain_warnings=chain.warnings,
    )


__all__ = [
    "MAX_COMPARISONS",
    "HP_PER_KW",
    "kw_to_hp",
    "hp_to_kw",
    "apply_engineering_filters",
    "dataset_items",
    "canonical_identity",
    "PresentationRole",
    "PresentationState",
    "set_presentation_role",
    "presentation_role_for",
    "set_current_item",
    "is_current_item",
    "TargetState",
    "set_target",
    "get_target",
    "TargetGap",
    "evaluate_target_gap",
    "WalkDisplayMode",
    "WalkDeltaBase",
    "WalkStep",
    "WalkViewSpec",
    "WalkRow",
    "WalkResult",
    "build_walk_rows",
    "default_walk_steps",
    "sequential_walk_steps",
    "delta_vs_reference_walk_steps",
    "ScenarioOption",
    "build_scenario_options",
    "SelectionState",
    "set_reference",
    "add_comparison",
    "remove_comparison",
    "sync_comparisons_from_widget",
    "format_value",
    "metric_axis_label",
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
    "build_cycle_phase_rows",
    "build_cycle_demand_rows",
    "is_temporary_net",
    "build_fe_vde_points",
    "build_iso_pse_lines",
    "compute_adaptive_pse_guides",
    "build_competitor_delta_rows",
    "ExploreDimension",
    "list_explore_dimensions",
    "list_explore_numeric_metrics",
    "list_available_explore_metrics",
    "list_explore_dimension_values",
    "ExploreBarRow",
    "ExploreScatterPoint",
    "ExploreLineRow",
    "build_explore_bar_rows",
    "build_explore_scatter_points",
    "build_explore_line_rows",
    "LineageContext",
    "resolve_lineage_context",
    "list_lineage_capable_metrics",
    "list_available_lineage_metrics",
    "LineageStep",
    "LineageWaterfallResult",
    "build_lineage_waterfall",
]
