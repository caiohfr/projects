# src/vde_app/components/comparison_report.py
# -----------------------------------------------------------------------------
# Package 8B-8F - dedicated "Program Energy & Fuel Economy Review" UI owner:
# Program Review, Energy Drivers, Technical Scorecard, Explore. Replaces
# pwt_fuel_energy.py as the entry point for the Comparison product. The old
# renderer's Scenario Compare tab is fully superseded by Technical Scorecard
# above and is not linked from here; its other three sub-tabs (Method
# Analysis, Peers & Outlook, Saved Estimates) are Powertrain-Scenario-owned
# capabilities with no Comparison equivalent, and stay reachable behind
# "Powertrain Scenario Tools" (see _render_legacy_bridge) indefinitely -- not
# as a placeholder pending a future package.
#
# Reference is optional (Package 8F): a dataset may hold only Comparison-role
# items (e.g. a benchmark-only review). Analytical presentation role
# (Proposal/Benchmark) and Current designation are a separate, session-only
# overlay, independent of both the canonical ComparisonRole and of
# provenance -- see PresentationState.
#
# This module never queries SQLite directly -- it only calls
# comparison_report_service.py / comparison_report_viewmodels.py /
# comparison_report_charts.py.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.vde_app.comparison_report_charts import (
    build_cycle_demand_figure,
    build_explore_bar,
    build_explore_line,
    build_explore_scatter,
    build_fe_vde_figure,
    build_grouped_bar_figure,
    build_lineage_waterfall_chart,
    build_vehicle_demand_breakdown_chart,
    build_walk_chart,
)
from src.vde_app.comparison_vehicle_demand_viewmodels import (
    VEHICLE_DEMAND_SECTION_TITLE,
    build_vehicle_demand_breakdown_rows,
    build_vehicle_demand_comparison_rows,
    resolve_vehicle_demand_outcomes,
)
from src.vde_app.comparison_report_viewmodels import (
    ComparisonItem,
    PresentationRole,
    PresentationState,
    ScorecardSection,
    SelectionState,
    TargetState,
    WalkDeltaBase,
    WalkDisplayMode,
    WalkStep,
    WalkViewSpec,
    apply_engineering_filters,
    build_cycle_demand_rows,
    build_cycle_phase_rows,
    build_explore_bar_rows,
    build_explore_line_rows,
    build_explore_scatter_points,
    build_fe_vde_points,
    build_iso_pse_lines,
    compute_adaptive_pse_guides,
    build_lineage_waterfall,
    build_roadload_curve_rows,
    build_scenario_header,
    build_scenario_options,
    build_scorecard_sections,
    build_walk_rows,
    canonical_identity,
    dataset_items,
    dataset_warnings_summary,
    default_walk_steps,
    delta_vs_reference_walk_steps,
    evaluate_target_gap,
    format_value,
    hp_to_kw,
    kw_to_hp,
    list_available_explore_metrics,
    list_available_lineage_metrics,
    list_explore_dimension_values,
    list_explore_dimensions,
    get_target,
    metric_axis_label,
    presentation_role_for,
    resolve_lineage_context,
    sequential_walk_steps,
    set_current_item,
    set_presentation_role,
    set_reference,
    set_target,
    sync_comparisons_from_widget,
)
from src.vde_app.components.pwt_fuel_energy import (
    render_comparison_report_page,
    resolve_comparison_report_anchor,
)
from src.vde_app.plots import roadload_curve_comparison_chart
from src.vde_app.units import normalize_unit_system, quantity_input, unit_label
from src.vde_core.comparison_metric_registry import MetricDirection, get_metric
from src.vde_core.comparison_report_service import (
    ComparisonDataset,
    ComparisonRole,
    LineageChainStatus,
    build_comparison_dataset,
    compare_metric,
    extract_metric_value,
    list_comparison_scenarios,
    list_vde_catalog,
    resolve_temporary_transmission_from_component,
)
from src.vde_core.component_repositories import load_component_repository
from src.vde_core.cycles import use_standard_cycle
from src.vde_core.db import current_db_path
from src.vde_core.vehicle_demand import RoadloadBasis

_CELL_STYLE = {
    "BETTER": "background-color: rgba(34,197,94,0.18)",
    "WORSE": "background-color: rgba(239,68,68,0.18)",
}

_SELECTION_KEY = "comparison_selection"
_DIRECT_VDE_SELECTION_KEY = "comparison_direct_vde_selection"
_TEMP_TRANSMISSION_KEY = "comparison_temporary_transmission_by_vde_id"
_PRESENTATION_KEY = "comparison_presentation_state"
_PRIMARY_KPI_KEY = "comparison_primary_kpi"
_TARGET_KEY = "comparison_target_state"
_WALK_ORDER_KEY = "comparison_walk_order"
_WALK_CONFIG_KEY = "comparison_walk_config"

_WALK_DELTA_BASE_LABELS = {
    WalkDeltaBase.PREVIOUS_WALK_STATE.value: "Previous step",
    WalkDeltaBase.REFERENCE.value: "Reference",
    WalkDeltaBase.EXPLICIT_ITEM.value: "Choose item...",
}

_DEFAULT_ETA_LINES = {
    "volumetric": (0.20, 0.25, 0.30, 0.35),
    "energy_normalized": (0.20, 0.25, 0.30, 0.35),
    "electrical": (0.85, 0.90, 0.95),
}
_FE_VDE_Y_TITLE = {
    "volumetric": "Fuel [L/100km]",
    "energy_normalized": "Consumed energy [MJ/km]",
    "electrical": "Electrical energy [Wh/km]",
}
_DELTA_METRIC_OPTIONS = (
    "vde_total", "vde_net", "fuel_l_per_100km", "fuel_km_per_l",
    "energy_wh_per_km", "gco2_per_km", "eta_pt_est", "mass_kg", "cda_m2", "rrc_n_per_kn",
)
_LINEAGE_WARNING_MESSAGES = {
    "broken_lineage_reference": "This lineage chain references a parent VDE that no longer exists in the database.",
    "self_parent_reference": "This VDE lists itself as its own parent -- the chain stops here.",
    "lineage_cycle_detected": "A cycle was detected in this lineage chain -- the chain stops here.",
    "lineage_chain_truncated_max_depth": "This lineage chain is unusually long and was truncated as a safety guard.",
    "lineage_root_not_found": "The selected VDE record could not be resolved.",
}


# -----------------------------------------------------------------------------
# Catalogs (cached by resolved DB path, same pattern as vde_request_compact.py)
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_catalog_cached(db_path_signature: str) -> list[dict]:
    return list_comparison_scenarios()


def _load_catalog() -> list[dict]:
    return _load_catalog_cached(str(Path(current_db_path()).resolve()))


@st.cache_data(show_spinner=False)
def _load_vde_catalog_cached(db_path_signature: str) -> list[dict]:
    return list_vde_catalog()


def _load_vde_catalog() -> list[dict]:
    return _load_vde_catalog_cached(str(Path(current_db_path()).resolve()))


def _dedupe_titles(titles: list[str]) -> list[str]:
    """pandas.Styler requires unique columns/index. Two comparison items can
    legitimately share the same label+provenance (e.g. two scenarios on the
    same VDE with the same record_origin but different fuel assumptions) --
    disambiguate with a trailing counter rather than letting Styler raise.
    """
    seen: dict[str, int] = {}
    result = []
    for title in titles:
        seen[title] = seen.get(title, 0) + 1
        result.append(title if seen[title] == 1 else f"{title} ({seen[title]})")
    return result


def _index_of(value, ordered_values: list) -> int:
    try:
        return ordered_values.index(value)
    except ValueError:
        return 0


def _no_reference_message(action: str, *, allow_direct_vde: bool = False) -> str:
    """Single wording source for the "nothing selected yet" empty state (Sec 13)
    -- every tab shares the same subject/verb pattern, only the action clause
    (what this section needs the selection for) differs. Reference is
    optional (Package 8F): this message only ever appears when NOTHING at
    all is selected, so it asks for "at least one scenario", not specifically
    a Reference.
    """
    subject = "scenario or VDE" if allow_direct_vde else "scenario"
    return f"Select at least one {subject} to begin {action}."


def _render_exclusions(excluded: list[dict]) -> None:
    if not excluded:
        return
    with st.expander(f"{len(excluded)} scenario(s) not shown here", expanded=False):
        for entry in excluded:
            st.caption(f"• {entry['label']} — {entry['reason']}")


# -----------------------------------------------------------------------------
# Scorecard scenario selection (Package 8B)
# -----------------------------------------------------------------------------


def _render_filters(catalog_rows: list[dict]) -> list[dict]:
    """One flat filter grid -- no expanders. Conceptual category (vehicle vs
    engineering vs data) is not a reason to create a separate visual section
    on its own; grouping is only introduced where it actually reduces
    complexity, and a second row of controls next to the first doesn't need
    one here. Displacement/Power are range sliders whose default spans the
    full dataset -- that full span IS the "All" neutral state, so no
    separate activation checkbox is needed; a scenario missing the field is
    excluded only once the user narrows the slider off its default (Package 8F).
    """
    makes = sorted({r["make"] for r in catalog_rows if r.get("make")})
    categories = sorted({r["category"] for r in catalog_rows if r.get("category")})
    legislations = sorted({r["legislation"] for r in catalog_rows if r.get("legislation")})
    electrifications = sorted({r["electrification"] for r in catalog_rows if r.get("electrification")})
    origins = sorted({r["record_origin"] for r in catalog_rows if r.get("record_origin")})

    col1, col2, col3, col4 = st.columns(4)
    make = col1.selectbox("Make", ["All"] + makes, key="comparison_filter_make")
    category = col2.selectbox("Category", ["All"] + categories, key="comparison_filter_category")
    legislation = col3.selectbox("Legislation", ["All"] + legislations, key="comparison_filter_legislation")
    electrification = col4.selectbox("Electrification", ["All"] + electrifications, key="comparison_filter_electrification")

    rows = catalog_rows
    if make != "All":
        rows = [r for r in rows if r.get("make") == make]
    if category != "All":
        rows = [r for r in rows if r.get("category") == category]
    if legislation != "All":
        rows = [r for r in rows if r.get("legislation") == legislation]
    if electrification != "All":
        rows = [r for r in rows if r.get("electrification") == electrification]

    size_col, power_col, origin_col = st.columns(3)

    sizes = sorted({r["engine_size_l"] for r in catalog_rows if r.get("engine_size_l") is not None})
    engine_size_l_range = None
    if len(sizes) >= 1:
        data_min, data_max = float(sizes[0]), float(sizes[-1])
        if data_min < data_max:
            size_min, size_max = size_col.slider(
                "Displacement [L]",
                min_value=data_min,
                max_value=data_max,
                value=(data_min, data_max),
                step=0.1,
                key="comparison_filter_engine_size_range",
            )
            if (size_min, size_max) != (data_min, data_max):
                engine_size_l_range = (size_min, size_max)
        else:
            size_col.caption(f"Displacement [L]: {data_min:g} (all)")
    else:
        size_col.caption("Displacement [L]: no data")

    powers_hp = sorted({kw_to_hp(r["engine_max_power_kw"]) for r in catalog_rows if r.get("engine_max_power_kw") is not None})
    engine_max_power_kw_range = None
    if len(powers_hp) >= 1:
        data_min_hp, data_max_hp = float(powers_hp[0]), float(powers_hp[-1])
        if data_min_hp < data_max_hp:
            power_min_hp, power_max_hp = power_col.slider(
                "Engine power [hp]",
                min_value=data_min_hp,
                max_value=data_max_hp,
                value=(data_min_hp, data_max_hp),
                step=5.0,
                key="comparison_filter_power_range",
            )
            if (power_min_hp, power_max_hp) != (data_min_hp, data_max_hp):
                engine_max_power_kw_range = (hp_to_kw(power_min_hp), hp_to_kw(power_max_hp))
        else:
            power_col.caption(f"Engine power [hp]: {data_min_hp:g} (all)")
    else:
        power_col.caption("Engine power [hp]: no data")

    record_origin = origin_col.selectbox("Provenance", ["All"] + origins, key="comparison_filter_record_origin")

    rows = apply_engineering_filters(
        rows, engine_size_l_range=engine_size_l_range, engine_max_power_kw_range=engine_max_power_kw_range
    )
    if record_origin != "All":
        rows = [r for r in rows if r.get("record_origin") == record_origin]

    return rows


def _option_label(fid: int, all_options_by_id: dict) -> str:
    option = all_options_by_id.get(fid)
    return option.label if option is not None else f"Unknown scenario #{fid}"


def _render_selection(catalog_rows: list[dict]) -> SelectionState:
    """Filters are a candidate-SEARCH tool only (Package 8F) -- they control
    what's newly offered for selection, never what's already selected. A
    selected Reference or Comparison item remains fully selected and usable
    even when it no longer matches the active filters; there is no
    filter-mismatch warning, because "doesn't match the current search" is
    not an error condition. Each picker's option list is therefore
    `currently_selected UNION filtered_candidates` -- the filters still fully
    control what's newly discoverable, just never what's already chosen.
    """
    all_options_by_id = {opt.fuelcons_id: opt for opt in build_scenario_options(catalog_rows)}
    filtered_rows = _render_filters(catalog_rows)
    filtered_options = build_scenario_options(filtered_rows)

    state: SelectionState = st.session_state.setdefault(_SELECTION_KEY, SelectionState())

    reference_ids = [None] + [opt.fuelcons_id for opt in filtered_options]
    if state.reference_fuelcons_id is not None and state.reference_fuelcons_id not in reference_ids:
        reference_ids.append(state.reference_fuelcons_id)
    reference_choice = st.selectbox(
        "Reference",
        options=reference_ids,
        format_func=lambda fid: "Select a reference scenario..." if fid is None else _option_label(fid, all_options_by_id),
        index=_index_of(state.reference_fuelcons_id, reference_ids),
        key="comparison_reference_select",
    )
    if reference_choice != state.reference_fuelcons_id:
        state = set_reference(state, reference_choice)

    picker_ids = [opt.fuelcons_id for opt in filtered_options if opt.fuelcons_id != state.reference_fuelcons_id]
    for cid in state.comparison_fuelcons_ids:
        if cid != state.reference_fuelcons_id and cid not in picker_ids:
            picker_ids.append(cid)

    comparison_choice_ids = st.multiselect(
        "Compare with (up to 10)",
        options=picker_ids,
        default=[cid for cid in state.comparison_fuelcons_ids if cid != state.reference_fuelcons_id],
        format_func=lambda fid: _option_label(fid, all_options_by_id),
        key="comparison_compare_with_select",
    )
    state, errors = sync_comparisons_from_widget(state, comparison_choice_ids)

    st.session_state[_SELECTION_KEY] = state
    for error in errors:
        st.warning(error)
    return state


def _build_scorecard_dataset(state: SelectionState) -> ComparisonDataset | None:
    """Reference is optional (Package 8F): a benchmark-only selection (no
    Reference, one or more Comparison items) builds a dataset with
    reference=None rather than being blocked outright. Nothing is ever
    substituted into the Reference slot -- an empty selection still renders
    the empty state.
    """
    if state.reference_fuelcons_id is None and not state.comparison_fuelcons_ids:
        return None
    reference_spec = (
        {"kind": "FUELCONS_SCENARIO", "fuelcons_id": state.reference_fuelcons_id}
        if state.reference_fuelcons_id is not None
        else None
    )
    comparison_specs = [{"kind": "FUELCONS_SCENARIO", "fuelcons_id": cid} for cid in state.comparison_fuelcons_ids]
    try:
        return build_comparison_dataset(reference_spec, comparison_specs)
    except ValueError as exc:
        st.error(str(exc))
        return None


# -----------------------------------------------------------------------------
# Presentation roles + Current designation (Package 8F Increment 2)
#
# Purely explicit, session-only overlay -- never inferred from provenance,
# method, model version, timestamp, label, or lineage. Independent of the
# canonical ComparisonRole (REFERENCE/COMPARISON) and of the optional
# Reference selection above; Current is not mutually exclusive with a role.
# -----------------------------------------------------------------------------


def _presentation_display_label(item: ComparisonItem) -> str:
    """Reference is a third, structurally distinct tag from role/Current --
    it must stay visible here too, not just in the Scorecard header, since
    this panel is exactly where a reader decides how each item is presented.
    """
    label = item.label or "Unknown vehicle"
    return f"{label} (Reference)" if item.role is ComparisonRole.REFERENCE else label


def _render_presentation_roles(dataset: ComparisonDataset) -> None:
    items = dataset_items(dataset)
    if not items:
        return

    state: PresentationState = st.session_state.setdefault(_PRESENTATION_KEY, PresentationState())
    role_values = [r.value for r in PresentationRole]

    with st.expander("Presentation roles", expanded=False):
        st.caption(
            "Optional labels for how each selected item is presented (Proposal / Benchmark) and which one "
            "is Current -- never derived from provenance or record origin. Reference (if any) is shown for "
            "context only; it is the canonical selection role, not a presentation tag."
        )
        current_ids = ["None"] + [canonical_identity(item) for item in items]
        current_labels = {"None": "None", **{canonical_identity(item): _presentation_display_label(item) for item in items}}
        current_choice = st.radio(
            "Current",
            current_ids,
            index=_index_of(state.current_item_id or "None", current_ids),
            format_func=lambda k: current_labels[k],
            horizontal=True,
            key="comparison_presentation_current",
        )
        new_current = None if current_choice == "None" else current_choice
        if new_current != state.current_item_id:
            state = set_current_item(state, new_current)

        for item in items:
            identity = canonical_identity(item)
            current_role = presentation_role_for(state, identity)
            role_choice = st.selectbox(
                _presentation_display_label(item),
                role_values,
                index=role_values.index(current_role.value),
                key=f"comparison_presentation_role_{identity}",
            )
            if role_choice != current_role.value:
                state = set_presentation_role(state, identity, PresentationRole(role_choice))

        st.session_state[_PRESENTATION_KEY] = state


# -----------------------------------------------------------------------------
# Primary KPI + Target (Package 8F Increment 3)
#
# Session-only, KPI-specific, optional. Target is never a scenario and never
# fabricated -- it only ever appears where a real actual value exists to
# compare it against.
# -----------------------------------------------------------------------------


def _render_primary_kpi_and_target() -> str:
    """Renders the Primary KPI + Target controls; returns the selected
    Primary KPI metric key so callers (Program Review, once assembled) can
    drive the hero visualization from it.
    """
    with st.expander("Primary KPI & Target", expanded=False):
        metric_key = st.selectbox(
            "Primary KPI",
            _DELTA_METRIC_OPTIONS,
            format_func=lambda key: (get_metric(key).label if get_metric(key) else key),
            key=_PRIMARY_KPI_KEY,
        )
        metric = get_metric(metric_key)
        unit_system = normalize_unit_system(st.session_state.get("unit_system"))
        target_state: TargetState = st.session_state.setdefault(_TARGET_KEY, TargetState())
        current_target = get_target(target_state, metric_key)

        col1, col2 = st.columns([1, 3])
        has_target = col1.checkbox("Set a target", value=current_target is not None, key=f"comparison_target_enabled_{metric_key}")
        new_value = col2.number_input(
            f"Target value ({metric.label if metric else metric_key})",
            value=float(current_target) if current_target is not None else 0.0,
            key=f"comparison_target_value_{metric_key}",
            disabled=not has_target,
        )
        new_target = new_value if has_target else None
        if new_target != current_target:
            target_state = set_target(target_state, metric_key, new_target)
            st.session_state[_TARGET_KEY] = target_state

        if current_target is not None and metric is not None:
            st.caption(f"Target: {format_value(current_target, metric.unit_family, unit_system)}")
    return metric_key


# -----------------------------------------------------------------------------
# Versatile KPI Walk (Package 8F Increment 4-6)
#
# Configuration is session-only, keyed by canonical identity. The safe
# default (no configuration yet) is ALL items ABSOLUTE -- a delta is never
# auto-created merely because an item is selected or tagged with a role.
# -----------------------------------------------------------------------------


def _steps_to_walk_state(steps: tuple) -> tuple[tuple[str, ...], dict[str, dict]]:
    order = tuple(step.item_id for step in steps)
    config = {
        step.item_id: {
            "display_mode": step.display_mode.value,
            "delta_base": step.delta_base.value if step.delta_base else None,
            "explicit_item_id": step.explicit_item_id,
            "advances_anchor": step.advances_anchor,
        }
        for step in steps
    }
    return order, config


def _render_walk_configuration(
    dataset: ComparisonDataset, items_by_identity: dict[str, ComparisonItem], metric_key: str
) -> WalkViewSpec:
    target_state: TargetState = st.session_state.setdefault(_TARGET_KEY, TargetState())
    target_value = get_target(target_state, metric_key)

    current_ids = tuple(items_by_identity.keys())
    stored_order = st.session_state.get(_WALK_ORDER_KEY, ())
    order = tuple(i for i in stored_order if i in current_ids) + tuple(i for i in current_ids if i not in stored_order)
    stored_config = st.session_state.get(_WALK_CONFIG_KEY, {})
    config: dict[str, dict] = {
        identity: dict(
            stored_config.get(
                identity, {"display_mode": "ABSOLUTE", "delta_base": None, "explicit_item_id": None, "advances_anchor": True}
            )
        )
        for identity in order
    }

    with st.expander("Configure Walk", expanded=False):
        st.caption("Every item renders ABSOLUTE unless you choose Delta below -- nothing is inferred automatically.")
        preset_cols = st.columns(3)
        if preset_cols[0].button("All Absolute", key="walk_preset_absolute"):
            new_order, new_config = _steps_to_walk_state(default_walk_steps(dataset))
            st.session_state[_WALK_ORDER_KEY] = new_order
            st.session_state[_WALK_CONFIG_KEY] = new_config
            st.rerun()
        if preset_cols[1].button("Sequential Walk", key="walk_preset_sequential"):
            new_order, new_config = _steps_to_walk_state(sequential_walk_steps(dataset))
            st.session_state[_WALK_ORDER_KEY] = new_order
            st.session_state[_WALK_CONFIG_KEY] = new_config
            st.rerun()
        if preset_cols[2].button("Delta vs Reference", key="walk_preset_delta_ref", disabled=dataset.reference is None):
            new_order, new_config = _steps_to_walk_state(delta_vs_reference_walk_steps(dataset))
            st.session_state[_WALK_ORDER_KEY] = new_order
            st.session_state[_WALK_CONFIG_KEY] = new_config
            st.rerun()

        for position, identity in enumerate(order):
            item = items_by_identity[identity]
            row_cfg = config[identity]
            cols = st.columns([0.4, 0.4, 2.2, 1.3, 1.6, 1])
            if cols[0].button("↑", key=f"walk_up_{identity}", disabled=position == 0):
                new_order = order[: position - 1] + (order[position], order[position - 1]) + order[position + 1 :]
                st.session_state[_WALK_ORDER_KEY] = new_order
                st.rerun()
            if cols[1].button("↓", key=f"walk_down_{identity}", disabled=position == len(order) - 1):
                new_order = order[:position] + (order[position + 1], order[position]) + order[position + 2 :]
                st.session_state[_WALK_ORDER_KEY] = new_order
                st.rerun()
            cols[2].markdown(item.label or "Unknown vehicle")
            mode_choice = cols[3].selectbox(
                "Mode",
                ["ABSOLUTE", "DELTA"],
                index=["ABSOLUTE", "DELTA"].index(row_cfg["display_mode"]),
                key=f"walk_mode_{identity}",
                label_visibility="collapsed",
            )
            row_cfg["display_mode"] = mode_choice
            if mode_choice == "DELTA":
                base_options = ["PREVIOUS_WALK_STATE", "EXPLICIT_ITEM"] + (
                    ["REFERENCE"] if dataset.reference is not None else []
                )
                current_base = row_cfg.get("delta_base") or base_options[0]
                if current_base not in base_options:
                    current_base = base_options[0]
                base_choice = cols[4].selectbox(
                    "vs",
                    base_options,
                    index=base_options.index(current_base),
                    format_func=lambda v: _WALK_DELTA_BASE_LABELS[v],
                    key=f"walk_base_{identity}",
                    label_visibility="collapsed",
                )
                row_cfg["delta_base"] = base_choice
                if base_choice == "EXPLICIT_ITEM":
                    other_ids = [i for i in order if i != identity]
                    if other_ids:
                        default_explicit = row_cfg.get("explicit_item_id")
                        if default_explicit not in other_ids:
                            default_explicit = other_ids[0]
                        explicit_choice = cols[5].selectbox(
                            "base item",
                            other_ids,
                            index=other_ids.index(default_explicit),
                            format_func=lambda i: items_by_identity[i].label or i,
                            key=f"walk_explicit_{identity}",
                            label_visibility="collapsed",
                        )
                        row_cfg["explicit_item_id"] = explicit_choice
                    else:
                        row_cfg["explicit_item_id"] = None
                    row_cfg["advances_anchor"] = row_cfg.get("advances_anchor", True)
                else:
                    row_cfg["explicit_item_id"] = None
                    row_cfg["advances_anchor"] = cols[5].checkbox(
                        "Anchor", value=row_cfg.get("advances_anchor", True), key=f"walk_anchor_{identity}"
                    )
            else:
                row_cfg["delta_base"] = None
                row_cfg["explicit_item_id"] = None
                row_cfg["advances_anchor"] = cols[5].checkbox(
                    "Anchor", value=row_cfg.get("advances_anchor", True), key=f"walk_anchor_{identity}"
                )
            config[identity] = row_cfg

        st.session_state[_WALK_ORDER_KEY] = order
        st.session_state[_WALK_CONFIG_KEY] = config

    steps = tuple(
        WalkStep(
            item_id=identity,
            display_mode=WalkDisplayMode(config[identity]["display_mode"]),
            delta_base=(WalkDeltaBase(config[identity]["delta_base"]) if config[identity]["delta_base"] else None),
            explicit_item_id=config[identity]["explicit_item_id"],
            advances_anchor=config[identity]["advances_anchor"],
        )
        for identity in order
    )
    return WalkViewSpec(metric_key=metric_key, steps=steps, target_value=target_value)


def _render_walk_callouts(result, items_by_identity: dict, dataset: ComparisonDataset, metric, unit_system: str) -> None:
    if metric is None:
        return
    current_row = next((row for row in result.rows if row.is_current), None)
    if current_row is None:
        return

    callouts: list[tuple[str, str]] = [("Current KPI", current_row.formatted_absolute_value)]
    if dataset.reference is not None and current_row.item_id != canonical_identity(dataset.reference):
        current_item = items_by_identity.get(current_row.item_id)
        if current_item is not None:
            ref_result = compare_metric(dataset.reference, current_item, metric.key)
            if ref_result["available"] and ref_result["compatible"]:
                callouts.append(
                    ("Δ vs Reference", format_value(ref_result["absolute_delta"], metric.unit_family, unit_system, signed=True))
                )
    if current_row.target_gap is not None:
        callouts.append(
            ("Gap vs Target", format_value(current_row.target_gap.absolute_gap, metric.unit_family, unit_system, signed=True))
        )
    if current_row.provenance:
        callouts.append(("Provenance", current_row.provenance))

    cols = st.columns(len(callouts))
    for col, (label, value) in zip(cols, callouts):
        col.metric(label, value)


def _render_walk_hero(dataset: ComparisonDataset, metric_key: str, unit_system: str) -> None:
    items = dataset_items(dataset)
    items_by_identity = {canonical_identity(item): item for item in items}
    presentation: PresentationState = st.session_state.setdefault(_PRESENTATION_KEY, PresentationState())

    spec = _render_walk_configuration(dataset, items_by_identity, metric_key)
    result = build_walk_rows(dataset, spec, presentation=presentation, unit_system=unit_system)
    metric = get_metric(metric_key)

    title = "KPI Walk" if result.has_delta_semantics else "KPI Comparison"
    st.markdown(f"**{title}**")
    for warning in result.warnings:
        st.warning(warning)

    if result.rows:
        y_title = metric_axis_label(metric, unit_system) if metric else metric_key
        fig = build_walk_chart(result.rows, y_title=y_title, target_value=result.target_value)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No compatible data for this Primary KPI.")

    _render_walk_callouts(result, items_by_identity, dataset, metric, unit_system)

    if result.rows:
        with st.expander("Walk detail", expanded=False):
            table_rows = [
                {
                    "Scenario": row.label,
                    "Mode": row.display_mode,
                    "Value": row.formatted_absolute_value,
                    "Δ": row.formatted_delta or "-",
                    "vs": row.delta_base_label or "-",
                    "Status": row.status,
                    "Role": row.presentation_role or "UNSPECIFIED",
                    "Current": "Yes" if row.is_current else "",
                }
                for row in result.rows
            ]
            st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")


# -----------------------------------------------------------------------------
# Scorecard tab (Package 8B)
# -----------------------------------------------------------------------------


def _render_section(section: ScorecardSection, header_titles: list[str]) -> None:
    rows = [row for row in section.rows if row.reference_cell.available or any(c.available for c in row.comparison_cells)]
    if not rows:
        return

    st.markdown(f"**{section.title}**")
    values: dict[str, list[str]] = {}
    semantics: dict[str, list[str | None]] = {}
    for row in rows:
        cells = [row.reference_cell, *row.comparison_cells]
        cell_texts = []
        cell_semantics = []
        for cell in cells:
            # Sprint 9D: `if` rather than `elif` so a cell can show a delta
            # AND a short warning together (e.g. a negative-residual "Review"
            # flag next to its Reference delta). Behavior-preserving for
            # every existing caller: warning and formatted_delta have always
            # been mutually exclusive by construction elsewhere in this file
            # (compare_metric's incompatible branch sets warning and leaves
            # formatted_delta None; every other cell never sets warning).
            text = cell.formatted_value
            if cell.formatted_delta:
                text = f"{text}\n{cell.formatted_delta}"
            if cell.warning:
                text = f"{text}\n{cell.warning}"
            cell_texts.append(text)
            cell_semantics.append(cell.semantic)
        values[row.label] = cell_texts
        semantics[row.label] = cell_semantics

    display_df = pd.DataFrame.from_dict(values, orient="index", columns=header_titles)
    semantic_df = pd.DataFrame.from_dict(semantics, orient="index", columns=header_titles)

    def _apply_style(_frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [[_CELL_STYLE.get(value, "") for value in values_row] for values_row in semantic_df.to_numpy()],
            index=semantic_df.index,
            columns=semantic_df.columns,
        )

    styled = display_df.style.apply(_apply_style, axis=None)
    st.dataframe(styled, hide_index=False, width="stretch")


def _render_scorecard_tab(dataset: ComparisonDataset | None) -> None:
    if dataset is None:
        st.info(_no_reference_message("comparison"))
        return

    warnings = dataset_warnings_summary(dataset)
    if warnings:
        with st.expander(f"{len(warnings)} dataset warning(s)", expanded=False):
            for warning in warnings:
                st.warning(warning)

    items = dataset_items(dataset)
    header_titles = _dedupe_titles([build_scenario_header(item)["column_title"] for item in items])

    unit_system = normalize_unit_system(st.session_state.get("unit_system"))
    for section in build_scorecard_sections(dataset, unit_system=unit_system):
        _render_section(section, header_titles)


# -----------------------------------------------------------------------------
# Program Review tab (Package 8F Increment 6)
#
# Story: (A) Are we on target? (B) Is the gap vehicle demand or powertrain
# efficiency? (C) How much vehicle demand remains? One Primary-KPI-driven
# hero drives the narrative -- Fuel Consumption/Economy/Energy/CO2 are never
# rendered here as four mandatory charts (that data remains in Technical
# Scorecard's evidence table).
# -----------------------------------------------------------------------------


_DELTA_COLOR_BY_DIRECTION = {
    MetricDirection.LOWER_IS_BETTER: "inverse",  # a decrease is good -> negative delta shows green
    MetricDirection.HIGHER_IS_BETTER: "normal",  # an increase is good -> positive delta shows green
}


def _summary_metric_cell(
    col, metric, item: ComparisonItem, reference: ComparisonItem | None, unit_system: str
) -> None:
    """One compact st.metric cell: absolute value is the primary number,
    %-delta vs Reference is the secondary/colored figure -- never
    concatenated into one equal-weight string. Absolute delta is available
    via the hover tooltip. Reuses compare_metric()/extract_metric_value()
    and the Registry's own direction for BETTER/WORSE color; no comparison
    semantics are recomputed here.
    """
    value = extract_metric_value(item, metric.key)
    if value is None:
        col.metric(metric.label, "unavailable")
        return
    formatted = format_value(value, metric.unit_family, unit_system)
    if reference is None or item is reference:
        col.metric(metric.label, formatted)
        return

    result = compare_metric(reference, item, metric.key)
    if not result["compatible"] or not result["available"]:
        col.metric(metric.label, formatted, help="Not comparable to Reference (different cycle/basis).")
        return

    abs_delta_text = format_value(result["absolute_delta"], metric.unit_family, unit_system, signed=True)
    if result["percent_delta"] is not None:
        sign = "+" if result["percent_delta"] > 0 else ""
        delta_text = f"{sign}{result['percent_delta']:.1f}% vs Ref"
    else:
        delta_text = f"{abs_delta_text} vs Ref"
    col.metric(
        metric.label,
        formatted,
        delta=delta_text,
        delta_color=_DELTA_COLOR_BY_DIRECTION.get(metric.direction, "off"),
        help=f"Δ {abs_delta_text}",
    )


def _render_energy_demand_summary(dataset: ComparisonDataset, metric_key: str, unit_system: str) -> None:
    """"Who is better, by how much, and does the gap look more like demand or
    efficiency" -- a few state KPIs per item, never a mini Technical
    Scorecard. Mass/RRC/CdA/ABC/phase/power stay in Energy Drivers; this is
    Primary KPI + selected VDE boundary + PSE (when available) + Target gap
    (when set) only.
    """
    boundary_choice = st.radio("VDE boundary", ["TOTAL", "NET"], horizontal=True, key="dashboard_vde_boundary")
    vde_metric = get_metric(f"vde_{boundary_choice.lower()}")
    primary_metric = get_metric(metric_key)
    pse_metric = get_metric("eta_pt_est")
    target_state: TargetState = st.session_state.setdefault(_TARGET_KEY, TargetState())
    target_value = get_target(target_state, metric_key)

    items = dataset_items(dataset)
    reference = dataset.reference

    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.caption(_presentation_display_label(item))
        if primary_metric is not None:
            _summary_metric_cell(col, primary_metric, item, reference, unit_system)
        if vde_metric is not None and vde_metric.key != metric_key:
            _summary_metric_cell(col, vde_metric, item, reference, unit_system)
        if pse_metric is not None and extract_metric_value(item, pse_metric.key) is not None:
            _summary_metric_cell(col, pse_metric, item, reference, unit_system)
        if target_value is not None and primary_metric is not None:
            gap = evaluate_target_gap(metric_key, extract_metric_value(item, metric_key), target_value)
            if gap is not None:
                col.metric("Gap to Target", format_value(gap.absolute_gap, primary_metric.unit_family, unit_system, signed=True))


def _render_fe_vde(dataset: ComparisonDataset, unit_system: str) -> None:
    st.markdown("**Demand vs Efficiency**")
    mode_label = st.selectbox("Mode", ["Volumetric", "Energy-normalized", "Electrical"], key="dashboard_fe_vde_mode")
    mode = {"Volumetric": "volumetric", "Energy-normalized": "energy_normalized", "Electrical": "electrical"}[mode_label]
    boundary = st.radio("VDE boundary", ["TOTAL", "NET"], horizontal=True, key="dashboard_fe_vde_boundary")

    points_result = build_fe_vde_points(dataset, boundary=boundary, mode=mode)
    _render_exclusions(points_result["excluded"])
    points = points_result["points"]
    if not points:
        st.info("No scenarios are compatible with this Demand vs Efficiency mode/boundary.")
        return
    if points_result.get("assumption_label"):
        st.caption(f"PSE energy basis: {points_result['assumption_label']}")

    xs = [p["x"] for p in points]
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min, x_max = x_min * 0.8, x_max * 1.2 + 0.1
    else:
        span = x_max - x_min
        x_min, x_max = x_min - span * 0.2, x_max + span * 0.2
    fuel_type = points_result.get("anchor_fuel_type") if mode == "volumetric" else None
    # Guide values are sized to what's actually plotted (Sprint 8 micro-polish)
    # rather than a fixed 20/25/30/35 set; the fixed set is only a fallback
    # for the rare case no plotted point yields a computable PSE. Either way
    # build_iso_pse_lines() is the sole authority on whether a line is
    # defensible at all for this mode/fuel_type -- it independently returns
    # [] for an unmappable basis regardless of which eta_values were passed,
    # so this fallback never risks fabricating a guide.
    eta_values = compute_adaptive_pse_guides(points, mode=mode, fuel_type=fuel_type) or _DEFAULT_ETA_LINES[mode]
    lines = build_iso_pse_lines(x_min, x_max, eta_values, mode=mode, fuel_type=fuel_type)
    if not lines and mode == "volumetric":
        st.caption(
            "Equi-PSE guides aren't available in Volumetric mode for this fuel family "
            f"({fuel_type or 'unknown'}) -- LHV is never guessed. Try Energy-normalized mode."
        )

    x_title = metric_axis_label(get_metric(f"vde_{boundary.lower()}"), unit_system)
    fig = build_fe_vde_figure(points, lines, x_title=x_title, y_title=_FE_VDE_Y_TITLE[mode])
    st.plotly_chart(fig, width="stretch")


def _render_program_review_tab(dataset: ComparisonDataset | None, metric_key: str) -> None:
    if dataset is None:
        st.info(_no_reference_message("the Program Review"))
        return

    unit_system = normalize_unit_system(st.session_state.get("unit_system"))
    warnings = dataset_warnings_summary(dataset)
    if warnings:
        with st.expander(f"{len(warnings)} dataset warning(s)", expanded=False):
            for warning in warnings:
                st.warning(warning)

    _render_walk_hero(dataset, metric_key, unit_system)
    st.divider()
    _render_fe_vde(dataset, unit_system)
    st.divider()
    st.markdown("**Energy & Demand Summary**")
    _render_energy_demand_summary(dataset, metric_key, unit_system)


# -----------------------------------------------------------------------------
# Roadload & VDE tab (Package 8C)
# -----------------------------------------------------------------------------


def _vde_option_label(row: dict) -> str:
    make = str(row.get("make") or "").strip()
    model = str(row.get("model") or "").strip()
    base = " ".join(part for part in (make, model) if part) or f"VDE #{row.get('vde_id')}"
    meta = " · ".join(str(x) for x in (row.get("year"), row.get("legislation")) if x)
    return f"{base} · {meta}" if meta else base


def _render_direct_vde_selection() -> tuple[dict | None, list[dict]] | None:
    catalog_rows = _load_vde_catalog()
    if not catalog_rows:
        st.info("No VDE records are available yet.")
        return None

    makes = sorted({r["make"] for r in catalog_rows if r.get("make")})
    legislations = sorted({r["legislation"] for r in catalog_rows if r.get("legislation")})
    col1, col2 = st.columns(2)
    make = col1.selectbox("Make", ["All"] + makes, key="roadload_direct_filter_make")
    legislation = col2.selectbox("Legislation", ["All"] + legislations, key="roadload_direct_filter_legislation")
    rows = catalog_rows
    if make != "All":
        rows = [r for r in rows if r.get("make") == make]
    if legislation != "All":
        rows = [r for r in rows if r.get("legislation") == legislation]

    options_by_id = {r["vde_id"]: _vde_option_label(r) for r in catalog_rows}
    visible_ids = [r["vde_id"] for r in rows]

    # Reuses SelectionState/set_reference/sync_comparisons_from_widget from 8B --
    # those helpers are generic over "an int id", not FuelCons-specific, despite
    # the field name. Filters are a candidate-search tool only (Package 8F): a
    # selected Reference/Comparison VDE stays selected and usable even when it
    # no longer matches Make/Legislation -- the picker option list is
    # currently_selected UNION filtered_candidates, never filtered-only.
    state: SelectionState = st.session_state.setdefault(_DIRECT_VDE_SELECTION_KEY, SelectionState())
    reference_ids = [None] + list(visible_ids)
    if state.reference_fuelcons_id is not None and state.reference_fuelcons_id not in reference_ids:
        reference_ids.append(state.reference_fuelcons_id)
    reference_choice = st.selectbox(
        "Reference VDE",
        options=reference_ids,
        format_func=lambda vid: "Select a reference VDE..." if vid is None else options_by_id.get(vid, f"VDE #{vid}"),
        index=_index_of(state.reference_fuelcons_id, reference_ids),
        key="roadload_direct_reference_select",
    )
    if reference_choice != state.reference_fuelcons_id:
        state = set_reference(state, reference_choice)

    picker_ids = [vid for vid in visible_ids if vid != state.reference_fuelcons_id]
    for vid in state.comparison_fuelcons_ids:
        if vid != state.reference_fuelcons_id and vid not in picker_ids:
            picker_ids.append(vid)

    comparison_choice_ids = st.multiselect(
        "Compare with (up to 10)",
        options=picker_ids,
        default=[vid for vid in state.comparison_fuelcons_ids if vid != state.reference_fuelcons_id],
        format_func=lambda vid: options_by_id.get(vid, f"VDE #{vid}"),
        key="roadload_direct_compare_with_select",
    )
    state, errors = sync_comparisons_from_widget(state, comparison_choice_ids)
    st.session_state[_DIRECT_VDE_SELECTION_KEY] = state
    for error in errors:
        st.warning(error)

    if state.reference_fuelcons_id is None and not state.comparison_fuelcons_ids:
        return None
    reference_spec = (
        {"kind": "VDE_ONLY", "vde_id": state.reference_fuelcons_id} if state.reference_fuelcons_id is not None else None
    )
    comparison_specs = [{"kind": "VDE_ONLY", "vde_id": vde_id} for vde_id in state.comparison_fuelcons_ids]
    return reference_spec, comparison_specs


def _linked_vde_specs(scorecard_dataset: ComparisonDataset | None) -> tuple[dict | None, list[dict]] | None:
    if scorecard_dataset is None:
        return None
    seen: set[int] = set()
    reference_spec: dict | None = None
    if scorecard_dataset.reference is not None and scorecard_dataset.reference.vde_id is not None:
        seen.add(scorecard_dataset.reference.vde_id)
        reference_spec = {"kind": "VDE_ONLY", "vde_id": scorecard_dataset.reference.vde_id}
    comparison_specs = []
    for item in scorecard_dataset.comparisons:
        if item.vde_id is None or item.vde_id in seen:
            continue
        seen.add(item.vde_id)
        comparison_specs.append({"kind": "VDE_ONLY", "vde_id": item.vde_id})
    if reference_spec is None and not comparison_specs:
        return None
    return reference_spec, comparison_specs


def _transmission_component_options() -> list[tuple[str, str]]:
    repository = load_component_repository("transmission")
    options = []
    for component in repository.list_components():
        component_id = component.get("component_id") or component.get("component_code")
        if not component_id:
            continue
        name = component.get("component_name") or component_id
        options.append((component_id, f"{name} ({component_id})"))
    return options


def _render_temporary_transmission_controls(dataset: ComparisonDataset, temp_by_vde: dict) -> None:
    items_missing_net = [
        item
        for item in dataset_items(dataset)
        if item.vde_id is not None and not item.roadload["net"].available and item.vde_id not in temp_by_vde
    ]
    active_ids = [
        item.vde_id
        for item in dataset_items(dataset)
        if item.vde_id is not None and item.vde_id in temp_by_vde
    ]

    for vde_id in dict.fromkeys(active_ids):
        item = next(i for i in dataset_items(dataset) if i.vde_id == vde_id)
        with st.expander(f"NET · TEMPORARY for {item.label}", expanded=False):
            st.caption(f"Source: {temp_by_vde[vde_id].get('source')}")
            if st.button("Clear temporary assumption", key=f"clear_temp_trans_{vde_id}"):
                temp_by_vde.pop(vde_id, None)
                st.session_state[_TEMP_TRANSMISSION_KEY] = dict(temp_by_vde)
                st.rerun()

    seen_vde_ids: set[int] = set()
    for item in items_missing_net:
        if item.vde_id in seen_vde_ids:
            continue
        seen_vde_ids.add(item.vde_id)
        with st.expander(f"NET unavailable for {item.label} — transmission data missing", expanded=False):
            source_choice = st.radio(
                "Source", ["Component DB", "Manual ABC"], key=f"temp_trans_source_{item.vde_id}", horizontal=True
            )
            if source_choice == "Component DB":
                options = _transmission_component_options()
                if not options:
                    st.caption("No transmission components available.")
                    continue
                choice = st.selectbox(
                    "Transmission component", options, format_func=lambda opt: opt[1], key=f"temp_trans_component_{item.vde_id}"
                )
                if st.button("Apply", key=f"apply_temp_trans_component_{item.vde_id}"):
                    resolved = resolve_temporary_transmission_from_component(choice[0])
                    if resolved:
                        temp_by_vde[item.vde_id] = resolved
                        st.session_state[_TEMP_TRANSMISSION_KEY] = dict(temp_by_vde)
                        st.rerun()
                    else:
                        st.error("Could not resolve this component's ABC.")
            else:
                unit_system = normalize_unit_system(st.session_state.get("unit_system"))
                a = quantity_input(st, "A", 0.0, "force", key=f"temp_trans_a_{item.vde_id}", unit_system=unit_system)
                b = quantity_input(st, "B", 0.0, "force_per_speed", key=f"temp_trans_b_{item.vde_id}", unit_system=unit_system)
                c = quantity_input(
                    st, "C", 0.0, "force_per_speed_squared", key=f"temp_trans_c_{item.vde_id}", unit_system=unit_system
                )
                if st.button("Apply", key=f"apply_temp_trans_manual_{item.vde_id}"):
                    temp_by_vde[item.vde_id] = {"source": "MANUAL", "A": a, "B": b, "C": c}
                    st.session_state[_TEMP_TRANSMISSION_KEY] = dict(temp_by_vde)
                    st.rerun()


_ROADLOAD_METRIC_SUFFIX = {"TOTAL": "_total", "NET": "_net"}


def _render_physical_setup_section(dataset: ComparisonDataset, unit_system: str) -> None:
    """Table, not bar charts (product feedback after reviewing the rendered
    8F build): Mass/RRC/CdA are single scalars per item, and a compact
    evidence-style table with Δ vs Reference carries more information per
    pixel than three separate bar-chart panels -- reuses the exact same
    ScorecardCell/_render_section machinery Technical Scorecard already uses,
    including its Reference-less absolute-only degrade.
    """
    items = dataset_items(dataset)
    header_titles = _dedupe_titles([build_scenario_header(item)["column_title"] for item in items])
    physical_section = next(
        (s for s in build_scorecard_sections(dataset, unit_system=unit_system) if s.title == "Physical Setup"), None
    )
    if physical_section is None or not physical_section.rows:
        return
    _render_section(physical_section, header_titles)


def _render_abc_section(dataset: ComparisonDataset, boundaries: list[str], unit_system: str) -> None:
    """Table, not three bar-chart panels -- A/B/C are coefficients of the one
    curve already plotted just above (roadload force vs speed); splitting
    them into separate bars forces the reader to mentally reconstruct that
    curve instead of reading it. Reuses the Scorecard's Roadload section,
    filtered to the boundaries currently selected above.
    """
    items = dataset_items(dataset)
    header_titles = _dedupe_titles([build_scenario_header(item)["column_title"] for item in items])
    roadload_section = next(
        (s for s in build_scorecard_sections(dataset, unit_system=unit_system) if s.title == "Roadload"), None
    )
    if roadload_section is None:
        return
    suffixes = tuple(_ROADLOAD_METRIC_SUFFIX[b] for b in boundaries)
    filtered_rows = tuple(row for row in roadload_section.rows if row.metric_key.endswith(suffixes))
    if not filtered_rows:
        return
    _render_section(ScorecardSection(title="Roadload ABC", rows=filtered_rows), header_titles)


def _render_roadload_curve_section(dataset: ComparisonDataset, boundaries: list[str]) -> None:
    st.markdown("**Roadload force curve**")
    curves = []
    excluded_all = []
    for boundary in boundaries:
        result = build_roadload_curve_rows(dataset, boundary)
        excluded_all += result["excluded"]
        suffix = f" · {boundary}" if len(boundaries) > 1 else ""
        for row in result["rows"]:
            curves.append({"label": f"{row['label']}{suffix}", **{k: row[k] for k in ("A_N", "B_N_per_kph", "C_N_per_kph2")}})
    _render_exclusions(excluded_all)
    if not curves:
        return
    unit_system = normalize_unit_system(st.session_state.get("unit_system"))
    fig = roadload_curve_comparison_chart(curves, unit_system=unit_system)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")


def _render_cycle_phase_section(dataset: ComparisonDataset, boundaries: list[str], unit_system: str) -> None:
    """Genuinely phase-aware (Package 8F Increment 7) -- reads
    VDEBoundaryResult.by_phase directly (EPA: City/Highway; WLTP: Low/Mid/
    High/Extra High). EPA and WLTP items are never merged into one chart;
    the TOTAL/NET aggregate itself is shown elsewhere (Program Review's
    Vehicle Demand Status), not duplicated here.
    """
    st.markdown("**VDE by Cycle / Phase**")
    y_title = f"VDE [{unit_label('energy_per_distance', unit_system)}]"
    any_rendered = False
    for boundary in boundaries:
        result = build_cycle_phase_rows(dataset, boundary)
        _render_exclusions(result["excluded"])
        for family_block in result["families"]:
            any_rendered = True
            st.caption(f"{family_block['family']} · {boundary}")
            st.plotly_chart(build_grouped_bar_figure(family_block["rows"], y_title=y_title), width="stretch")
    if not any_rendered:
        st.caption("No recognized phase breakdown (EPA City/Highway or WLTP Low/Mid/High/Extra High) is available for the selected items.")


_ROADLOAD_BASIS_BY_LABEL = {"TOTAL": RoadloadBasis.TOTAL, "NET": RoadloadBasis.NET}


def _render_vehicle_demand_summary_section(dataset: ComparisonDataset, boundaries: list[str], unit_system: str) -> None:
    """"Why is vehicle demand different?" (Sprint 9D) -- a compact table from
    the frozen Vehicle Demand Core (Sprint 9A-9C), complementing (never
    replacing) VDE by Cycle/Phase above. Pure consumer: no roadload/RRC/CdA/
    air-density/inertia/tractive/energy-integration physics is computed
    here -- everything comes from calculate_vehicle_demand() via
    comparison_vehicle_demand_viewmodels.py.
    """
    st.markdown(f"**{VEHICLE_DEMAND_SECTION_TITLE}**")
    st.caption(
        "Top rows: overall demand (VDE, tractive/braking energy). Lower rows: the roadload "
        "explanation behind it (known rolling/aero, residual/unattributed, inertial work)."
    )
    items = dataset_items(dataset)
    header_titles = _dedupe_titles([build_scenario_header(item)["column_title"] for item in items])
    outcomes = resolve_vehicle_demand_outcomes(dataset)
    for boundary in boundaries:
        basis = _ROADLOAD_BASIS_BY_LABEL[boundary]
        if len(boundaries) > 1:
            st.caption(boundary)
        section = build_vehicle_demand_comparison_rows(dataset, basis, unit_system, outcomes=outcomes)
        _render_section(section, header_titles)

        breakdown = build_vehicle_demand_breakdown_rows(dataset, basis, outcomes=outcomes)
        _render_exclusions(breakdown["excluded"])
        if breakdown["rows"]:
            with st.expander("Vehicle Demand Energy Breakdown", expanded=False):
                st.caption("Known Rolling + Known Aero + Residual/Unattributed always sum to Roadload Energy.")
                st.plotly_chart(build_vehicle_demand_breakdown_chart(breakdown["rows"]), width="stretch")


def _render_cycle_demand_section(dataset: ComparisonDataset, boundaries: list[str]) -> None:
    st.markdown("**Demanded power over cycle**")
    if not st.checkbox("Show demanded power over cycle", value=False, key="roadload_show_cycle_demand"):
        st.caption("Calculates cycle-integrated power traces on demand -- opt in above.")
        return

    legislation = dataset_items(dataset)[0].vehicle.get("legislation")
    cycle = use_standard_cycle(legislation)
    if cycle is None:
        st.info("No standard cycle trace available for this legislation.")
        return

    max_traces = st.slider(
        "Comparison traces shown", 0, len(dataset.comparisons), min(len(dataset.comparisons), 3),
        key="roadload_cycle_demand_trace_count",
    )
    limited_dataset = ComparisonDataset(reference=dataset.reference, comparisons=dataset.comparisons[:max_traces])
    result = build_cycle_demand_rows(limited_dataset, cycle, boundaries)
    _render_exclusions(result["excluded"])
    if result["series"]:
        st.plotly_chart(build_cycle_demand_figure(result["series"], result["time_s"]), width="stretch")


def _render_energy_drivers_tab(scorecard_dataset: ComparisonDataset | None) -> None:
    """"Why is vehicle demand different?" (Package 8F, extended Sprint 9D).
    Storytelling order: physical setup -> roadload force curve -> ABC
    coefficients -> VDE by cycle/phase -> Vehicle Demand Summary (compact
    physical explanation table, Sprint 9D) -> demanded power (lower
    priority/expandable). Mass/RRC/CdA stay visually coupled with roadload/
    VDE rather than split into a separate overview.
    """
    source = st.radio(
        "Source",
        ["VDEs linked to selected complete scenarios", "Select physical VDEs directly"],
        key="roadload_source_mode",
    )
    specs = (
        _render_direct_vde_selection()
        if source == "Select physical VDEs directly"
        else _linked_vde_specs(scorecard_dataset)
    )
    if specs is None:
        st.info(_no_reference_message("physical analysis", allow_direct_vde=True))
        return

    temp_by_vde = st.session_state.setdefault(_TEMP_TRANSMISSION_KEY, {})
    reference_spec, comparison_specs = specs
    try:
        dataset = build_comparison_dataset(reference_spec, comparison_specs, temporary_transmission_by_vde_id=temp_by_vde)
    except ValueError as exc:
        st.error(str(exc))
        return

    boundary_choice = st.radio("Roadload basis", ["TOTAL", "NET", "Both"], horizontal=True, key="roadload_basis")
    boundaries = {"TOTAL": ["TOTAL"], "NET": ["NET"], "Both": ["TOTAL", "NET"]}[boundary_choice]

    if "NET" in boundaries:
        _render_temporary_transmission_controls(dataset, temp_by_vde)

    unit_system = normalize_unit_system(st.session_state.get("unit_system"))
    st.divider()
    _render_physical_setup_section(dataset, unit_system)
    st.divider()
    _render_roadload_curve_section(dataset, boundaries)
    st.divider()
    _render_abc_section(dataset, boundaries, unit_system)
    st.divider()
    _render_cycle_phase_section(dataset, boundaries, unit_system)
    st.divider()
    _render_vehicle_demand_summary_section(dataset, boundaries, unit_system)
    st.divider()
    _render_cycle_demand_section(dataset, boundaries)


# -----------------------------------------------------------------------------
# Explore tab: Custom Chart + Lineage (Package 8D)
#
# Two independent sections, never merged into one overloaded form (Sec 2).
# Custom Chart is Metric-Registry-driven generic exploration; Lineage is an
# explicit parent-child VDE walk. Neither infers a relationship or a trend --
# the user interprets the chart (Sec 51).
# -----------------------------------------------------------------------------


def _explore_source_options(scorecard_dataset: ComparisonDataset | None) -> list[str]:
    options: list[str] = []
    if scorecard_dataset is not None:
        options.append("Selected complete scenarios")
    direct_state: SelectionState | None = st.session_state.get(_DIRECT_VDE_SELECTION_KEY)
    if direct_state is not None and (direct_state.reference_fuelcons_id is not None or direct_state.comparison_fuelcons_ids):
        options.append("Selected physical VDEs")
    return options


def _build_explore_dataset(
    source: str, scorecard_dataset: ComparisonDataset | None, temp_by_vde: dict
) -> ComparisonDataset | None:
    """Sec 48: reuses whatever is already selected elsewhere -- the Scorecard
    dataset, or the Roadload tab's existing direct-VDE selection state -- and
    never renders a second, fragile selection UI here.
    """
    if source == "Selected complete scenarios":
        return scorecard_dataset
    if source == "Selected physical VDEs":
        direct_state: SelectionState | None = st.session_state.get(_DIRECT_VDE_SELECTION_KEY)
        if direct_state is None or (direct_state.reference_fuelcons_id is None and not direct_state.comparison_fuelcons_ids):
            return None
        reference_spec = (
            {"kind": "VDE_ONLY", "vde_id": direct_state.reference_fuelcons_id}
            if direct_state.reference_fuelcons_id is not None
            else None
        )
        comparison_specs = [{"kind": "VDE_ONLY", "vde_id": vde_id} for vde_id in direct_state.comparison_fuelcons_ids]
        try:
            return build_comparison_dataset(reference_spec, comparison_specs, temporary_transmission_by_vde_id=temp_by_vde)
        except ValueError as exc:
            st.error(str(exc))
            return None
    return None


def _render_explore_bar(dataset: ComparisonDataset, items: tuple, unit_system: str, group_key: str | None, filter_key: str | None, filter_values: list[str]) -> None:
    x_dims = list_explore_dimensions("x")
    x_label = st.selectbox("X", [d.label for d in x_dims], key="explore_bar_x")
    x_dim = next(d for d in x_dims if d.label == x_label)

    metrics = list_available_explore_metrics(items, "bar")
    if not metrics:
        st.info("No compatible numeric metrics are available for this selection.")
        return
    y_label = st.selectbox("Y", [m.label for m in metrics], key="explore_bar_y")
    y_metric = next(m for m in metrics if m.label == y_label)

    result = build_explore_bar_rows(
        dataset,
        x_dimension_key=x_dim.key,
        y_metric_key=y_metric.key,
        group_dimension_key=group_key,
        filter_dimension_key=filter_key,
        filter_values=filter_values,
        unit_system=unit_system,
    )
    _render_exclusions(result["excluded"])
    if not result["rows"]:
        st.info("No compatible numeric metrics are available for this selection.")
        return
    fig = build_explore_bar(result["rows"], x_title=x_dim.label, y_title=metric_axis_label(y_metric, unit_system))
    st.plotly_chart(fig, width="stretch")


def _render_explore_scatter(items: tuple, dataset: ComparisonDataset, unit_system: str, group_key: str | None, filter_key: str | None, filter_values: list[str]) -> None:
    metrics = list_available_explore_metrics(items, "scatter")
    if not metrics:
        st.info("No compatible numeric metrics are available for this selection.")
        return
    col1, col2 = st.columns(2)
    x_label = col1.selectbox("X", [m.label for m in metrics], key="explore_scatter_x")
    x_metric = next(m for m in metrics if m.label == x_label)
    default_y_index = 1 if len(metrics) > 1 else 0
    y_label = col2.selectbox("Y", [m.label for m in metrics], index=default_y_index, key="explore_scatter_y")
    y_metric = next(m for m in metrics if m.label == y_label)

    result = build_explore_scatter_points(
        dataset,
        x_metric_key=x_metric.key,
        y_metric_key=y_metric.key,
        group_dimension_key=group_key,
        filter_dimension_key=filter_key,
        filter_values=filter_values,
    )
    _render_exclusions(result["excluded"])
    if not result["points"]:
        st.info("No compatible numeric metrics are available for this selection.")
        return
    fig = build_explore_scatter(
        result["points"],
        x_title=metric_axis_label(x_metric, unit_system),
        y_title=metric_axis_label(y_metric, unit_system),
    )
    st.plotly_chart(fig, width="stretch")


def _render_explore_line(dataset: ComparisonDataset, items: tuple, unit_system: str, group_key: str | None, filter_key: str | None, filter_values: list[str]) -> None:
    x_dims = list_explore_dimensions("order")
    if not x_dims:
        st.info("No explicitly ordered dimension is available for a Line chart.")
        return
    x_label = st.selectbox("X (ordered)", [d.label for d in x_dims], key="explore_line_x")
    x_dim = next(d for d in x_dims if d.label == x_label)

    metrics = list_available_explore_metrics(items, "line")
    if not metrics:
        st.info("No compatible numeric metrics are available for this selection.")
        return
    y_label = st.selectbox("Y", [m.label for m in metrics], key="explore_line_y")
    y_metric = next(m for m in metrics if m.label == y_label)

    result = build_explore_line_rows(
        dataset,
        x_dimension_key=x_dim.key,
        y_metric_key=y_metric.key,
        group_dimension_key=group_key,
        filter_dimension_key=filter_key,
        filter_values=filter_values,
        unit_system=unit_system,
    )
    _render_exclusions(result["excluded"])
    if not result["rows"]:
        st.info(result.get("unavailable_reason") or "No compatible numeric metrics are available for this selection.")
        return
    fig = build_explore_line(result["rows"], x_title=x_dim.label, y_title=metric_axis_label(y_metric, unit_system))
    st.plotly_chart(fig, width="stretch")


def _render_explore_custom_chart(dataset: ComparisonDataset, unit_system: str) -> None:
    items = dataset_items(dataset)
    chart_type = st.selectbox("Chart type", ["Bar", "Scatter", "Line"], key="explore_chart_type")

    col1, col2 = st.columns(2)
    filter_dims = list_explore_dimensions("filter")
    filter_label = col1.selectbox("Filter", ["None"] + [d.label for d in filter_dims], key="explore_filter_dimension")
    filter_dim = next((d for d in filter_dims if d.label == filter_label), None)
    filter_values: list[str] = []
    if filter_dim is not None:
        available_values = list_explore_dimension_values(items, filter_dim.key)
        filter_values = col1.multiselect(
            f"{filter_dim.label} values", available_values, key=f"explore_filter_values_{filter_dim.key}"
        )

    group_dims = list_explore_dimensions("group")
    group_label = col2.selectbox("Group / color", ["None"] + [d.label for d in group_dims], key="explore_group_dimension")
    group_dim = next((d for d in group_dims if d.label == group_label), None)

    group_key = group_dim.key if group_dim else None
    filter_key = filter_dim.key if filter_dim else None

    if chart_type == "Bar":
        _render_explore_bar(dataset, items, unit_system, group_key, filter_key, filter_values)
    elif chart_type == "Scatter":
        _render_explore_scatter(items, dataset, unit_system, group_key, filter_key, filter_values)
    else:
        _render_explore_line(dataset, items, unit_system, group_key, filter_key, filter_values)


def _lineage_item_key(item: ComparisonItem) -> str:
    if item.fuelcons_id is not None:
        return f"fc:{item.fuelcons_id}"
    return f"vde:{item.vde_id}"


def _render_lineage_tab(dataset: ComparisonDataset, temp_by_vde: dict, unit_system: str) -> None:
    items = [item for item in dataset_items(dataset) if item.vde_id is not None]
    if not items:
        st.info("No scenarios with a resolvable VDE are available to analyze lineage for.")
        return

    keys = [_lineage_item_key(item) for item in items]
    labels = {key: f"{item.label or 'Unknown vehicle'} (VDE #{item.vde_id})" for key, item in zip(keys, items)}
    selected_key = st.selectbox(
        "Analyze lineage for", keys, format_func=lambda k: labels[k], key="lineage_selected_item"
    )
    selected_item = items[keys.index(selected_key)]

    context = resolve_lineage_context(selected_item)
    if context is None:
        st.info("No explicit parent-child lineage is available for this scenario.")
        return

    caption = "Physical VDE Lineage"
    if context.is_fuelcons_scenario:
        caption += f" -- resolved from FuelCons scenario '{context.originating_label}'"
    st.caption(caption)

    chain = context.chain
    if not chain.nodes:
        for warning in chain.warnings:
            st.warning(_LINEAGE_WARNING_MESSAGES.get(warning.split(":")[0], warning))
        st.info("No explicit parent-child lineage is available for this scenario.")
        return
    if len(chain.nodes) == 1 and chain.status == LineageChainStatus.ROOT:
        st.info(f"'{chain.nodes[0].label}' is a lineage root -- no explicit parent scenario is recorded.")
    for warning in chain.warnings:
        st.warning(_LINEAGE_WARNING_MESSAGES.get(warning.split(":")[0], warning))

    metrics = list_available_lineage_metrics(chain)
    if not metrics:
        st.info("No compatible numeric metrics are available across this lineage chain.")
        return
    metric_label = st.selectbox("Metric", [m.label for m in metrics], key="lineage_metric")
    metric = next(m for m in metrics if m.label == metric_label)

    waterfall = build_lineage_waterfall(
        chain, metric.key, unit_system=unit_system, temporary_transmission_by_vde_id=temp_by_vde
    )
    if waterfall.steps:
        fig = build_lineage_waterfall_chart(waterfall.steps, y_title=metric_axis_label(metric, unit_system))
        st.plotly_chart(fig, width="stretch")
    if not waterfall.complete and waterfall.incomplete_reason:
        st.warning(waterfall.incomplete_reason)

    if waterfall.steps:
        table_rows = [
            {
                "Step": i,
                "Scenario": step.label,
                "Parent": step.parent_vde_id if step.parent_vde_id is not None else "-",
                "Provenance": step.provenance or "UNKNOWN",
                "Metric value": step.formatted_value,
                "Δ vs Parent": step.formatted_delta or "-",
                "Status": step.status,
            }
            for i, step in enumerate(waterfall.steps)
        ]
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")


def _render_explore_tab(scorecard_dataset: ComparisonDataset | None) -> None:
    source_options = _explore_source_options(scorecard_dataset)
    if not source_options:
        st.info("Select scenarios above to explore them here.")
        return

    source = (
        st.radio("Data source", source_options, horizontal=True, key="explore_data_source")
        if len(source_options) > 1
        else source_options[0]
    )
    temp_by_vde = st.session_state.setdefault(_TEMP_TRANSMISSION_KEY, {})
    dataset = _build_explore_dataset(source, scorecard_dataset, temp_by_vde)
    if dataset is None:
        st.info(_no_reference_message("exploring", allow_direct_vde=True))
        return

    warnings = dataset_warnings_summary(dataset)
    if warnings:
        with st.expander(f"{len(warnings)} dataset warning(s)", expanded=False):
            for warning in warnings:
                st.warning(warning)

    unit_system = normalize_unit_system(st.session_state.get("unit_system"))
    custom_chart_tab, lineage_tab = st.tabs(["Custom Chart", "Lineage"])
    with custom_chart_tab:
        _render_explore_custom_chart(dataset, unit_system)
    with lineage_tab:
        _render_lineage_tab(dataset, temp_by_vde, unit_system)


# -----------------------------------------------------------------------------
# Legacy bridge and page entry point
# -----------------------------------------------------------------------------


def _render_legacy_bridge() -> None:
    st.divider()
    with st.expander("Powertrain Scenario Tools", expanded=False):
        st.caption(
            "Method Analysis, Peers & Outlook, and Saved Estimates remain here -- they are Powertrain Scenario "
            "capabilities (ML method explanation, DB-wide peer benchmarking, saved-estimate management) with no "
            "equivalent above. The Scenario Compare tab below is superseded by Technical Scorecard above; prefer "
            "that one for engineering comparison."
        )
        vde_id, vde_row = resolve_comparison_report_anchor()
        if not vde_id:
            st.info("No VDE source could be resolved yet.")
            return
        render_comparison_report_page(vde_id, vde_row)


def render_comparison_report() -> None:
    st.title("Program Energy & Fuel Economy Review")
    st.caption("Reference-optional engineering comparison across FuelCons scenarios and physical VDEs.")

    catalog_rows = _load_catalog()
    if not catalog_rows:
        st.info("No FuelCons scenarios are available yet. Save at least one Powertrain Scenario first.")
        _render_legacy_bridge()
        return

    state = _render_selection(catalog_rows)
    dataset = _build_scorecard_dataset(state)
    primary_kpi = _DELTA_METRIC_OPTIONS[0]
    if dataset is not None:
        _render_presentation_roles(dataset)
        primary_kpi = _render_primary_kpi_and_target()

    program_review_tab, energy_drivers_tab, scorecard_tab, explore_tab = st.tabs(
        ["Program Review", "Energy Drivers", "Technical Scorecard", "Explore"]
    )
    with program_review_tab:
        _render_program_review_tab(dataset, primary_kpi)
    with energy_drivers_tab:
        _render_energy_drivers_tab(dataset)
    with scorecard_tab:
        _render_scorecard_tab(dataset)
    with explore_tab:
        _render_explore_tab(dataset)

    _render_legacy_bridge()


__all__ = ["render_comparison_report"]
