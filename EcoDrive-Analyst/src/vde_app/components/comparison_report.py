# src/vde_app/components/comparison_report.py
# -----------------------------------------------------------------------------
# Package 8B/8C - dedicated Comparison Report UI owner. Replaces
# pwt_fuel_energy.py as the entry point for the Comparison product; the old
# renderer (Scenario Compare / Method Analysis / Peers & Outlook / Saved
# Estimates) stays intact and reachable behind "Legacy comparison tools"
# until 8D absorbs its remaining useful capabilities.
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
    build_fe_vde_figure,
    build_grouped_bar_figure,
)
from src.vde_app.comparison_report_viewmodels import (
    MAX_COMPARISONS,
    ScorecardSection,
    SelectionState,
    build_abc_rows,
    build_competitor_delta_rows,
    build_cycle_demand_rows,
    build_fe_vde_points,
    build_iso_pse_lines,
    build_metric_bar_rows,
    build_reference_summary,
    build_roadload_curve_rows,
    build_scenario_header,
    build_scenario_options,
    build_scorecard_sections,
    dataset_warnings_summary,
    format_value,
    set_reference,
    sync_comparisons_from_widget,
)
from src.vde_app.components.pwt_fuel_energy import (
    render_comparison_report_page,
    resolve_comparison_report_anchor,
)
from src.vde_app.plots import roadload_curve_comparison_chart
from src.vde_app.units import normalize_unit_system, quantity_input, unit_label
from src.vde_core.comparison_metric_registry import get_metric
from src.vde_core.comparison_report_service import (
    ComparisonDataset,
    build_comparison_dataset,
    list_comparison_scenarios,
    list_vde_catalog,
    resolve_temporary_transmission_from_component,
)
from src.vde_core.component_repositories import load_component_repository
from src.vde_core.cycles import use_standard_cycle
from src.vde_core.db import current_db_path

_CELL_STYLE = {
    "BETTER": "background-color: rgba(34,197,94,0.18)",
    "WORSE": "background-color: rgba(239,68,68,0.18)",
}

_SELECTION_KEY = "comparison_selection"
_DIRECT_VDE_SELECTION_KEY = "comparison_direct_vde_selection"
_TEMP_TRANSMISSION_KEY = "comparison_temporary_transmission_by_vde_id"

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
    makes = sorted({r["make"] for r in catalog_rows if r.get("make")})
    legislations = sorted({r["legislation"] for r in catalog_rows if r.get("legislation")})
    electrifications = sorted({r["electrification"] for r in catalog_rows if r.get("electrification")})
    origins = sorted({r["record_origin"] for r in catalog_rows if r.get("record_origin")})

    col1, col2, col3, col4 = st.columns(4)
    make = col1.selectbox("Make", ["All"] + makes, key="comparison_filter_make")
    legislation = col2.selectbox("Legislation", ["All"] + legislations, key="comparison_filter_legislation")
    electrification = col3.selectbox("Electrification", ["All"] + electrifications, key="comparison_filter_electrification")
    record_origin = col4.selectbox("Provenance", ["All"] + origins, key="comparison_filter_record_origin")

    rows = catalog_rows
    if make != "All":
        rows = [r for r in rows if r.get("make") == make]
    if legislation != "All":
        rows = [r for r in rows if r.get("legislation") == legislation]
    if electrification != "All":
        rows = [r for r in rows if r.get("electrification") == electrification]
    if record_origin != "All":
        rows = [r for r in rows if r.get("record_origin") == record_origin]
    return rows


def _render_selection(catalog_rows: list[dict]) -> SelectionState:
    all_options_by_id = {opt.fuelcons_id: opt for opt in build_scenario_options(catalog_rows)}
    filtered_rows = _render_filters(catalog_rows)
    options = build_scenario_options(filtered_rows)
    options_by_id = {opt.fuelcons_id: opt for opt in options} or all_options_by_id

    state: SelectionState = st.session_state.setdefault(_SELECTION_KEY, SelectionState())

    reference_ids = [None] + [opt.fuelcons_id for opt in options]
    reference_choice = st.selectbox(
        "Reference",
        options=reference_ids,
        format_func=lambda fid: "Select a reference scenario..." if fid is None else all_options_by_id[fid].label,
        index=_index_of(state.reference_fuelcons_id, reference_ids),
        key="comparison_reference_select",
    )
    if reference_choice != state.reference_fuelcons_id:
        state = set_reference(state, reference_choice)

    filtered_ids = {opt.fuelcons_id for opt in options}
    visible_ids = [cid for cid in state.comparison_fuelcons_ids if cid in filtered_ids]
    hidden_ids = tuple(cid for cid in state.comparison_fuelcons_ids if cid not in filtered_ids)

    comparison_choice_ids = st.multiselect(
        "Compare with (up to 10)",
        options=[fid for fid in options_by_id if fid != state.reference_fuelcons_id],
        default=[cid for cid in visible_ids if cid != state.reference_fuelcons_id],
        format_func=lambda fid: all_options_by_id[fid].label,
        key="comparison_compare_with_select",
    )
    visible_state = SelectionState(state.reference_fuelcons_id, tuple(visible_ids))
    new_visible_state, errors = sync_comparisons_from_widget(visible_state, comparison_choice_ids)
    merged = hidden_ids + new_visible_state.comparison_fuelcons_ids
    errors = list(errors)
    if len(merged) > MAX_COMPARISONS:
        errors.append(f"Maximum {MAX_COMPARISONS} comparison scenarios reached. Remove one before adding another.")
        merged = merged[:MAX_COMPARISONS]
    state = SelectionState(state.reference_fuelcons_id, merged)

    st.session_state[_SELECTION_KEY] = state
    for error in errors:
        st.warning(error)
    return state


def _build_scorecard_dataset(state: SelectionState) -> ComparisonDataset | None:
    if state.reference_fuelcons_id is None:
        return None
    reference_spec = {"kind": "FUELCONS_SCENARIO", "fuelcons_id": state.reference_fuelcons_id}
    comparison_specs = [{"kind": "FUELCONS_SCENARIO", "fuelcons_id": cid} for cid in state.comparison_fuelcons_ids]
    try:
        return build_comparison_dataset(reference_spec, comparison_specs)
    except ValueError as exc:
        st.error(str(exc))
        return None


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
            text = cell.formatted_value
            if cell.formatted_delta:
                text = f"{text}\n{cell.formatted_delta}"
            elif cell.warning:
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
        st.info("Select a reference scenario to begin comparison.")
        return

    warnings = dataset_warnings_summary(dataset)
    if warnings:
        with st.expander(f"{len(warnings)} dataset warning(s)", expanded=False):
            for warning in warnings:
                st.warning(warning)

    items = (dataset.reference, *dataset.comparisons)
    header_titles = _dedupe_titles([build_scenario_header(item)["column_title"] for item in items])

    unit_system = normalize_unit_system(st.session_state.get("unit_system"))
    for section in build_scorecard_sections(dataset, unit_system=unit_system):
        _render_section(section, header_titles)


# -----------------------------------------------------------------------------
# Dashboard tab (Package 8C)
# -----------------------------------------------------------------------------


def _render_reference_summary(dataset: ComparisonDataset, unit_system: str) -> None:
    summary = build_reference_summary(dataset.reference)
    st.markdown("**Reference summary**")
    cols = st.columns(4)
    cols[0].metric("Vehicle", summary["label"])
    cols[1].metric("Legislation", summary["legislation"] or "–")
    cols[2].metric("Provenance", summary["record_origin"] or "–")
    cols[3].metric("VDE TOTAL", format_value(summary["vde_total"], "energy_mj_per_km", unit_system))
    caption_parts = []
    if summary["vde_net"] is not None:
        caption_parts.append(f"VDE NET: {format_value(summary['vde_net'], 'energy_mj_per_km', unit_system)}")
    if summary["fuel_l_per_100km"] is not None:
        caption_parts.append(f"Fuel: {format_value(summary['fuel_l_per_100km'], 'l_per_100km', unit_system)}")
    if summary["gco2_per_km"] is not None:
        caption_parts.append(f"CO2: {format_value(summary['gco2_per_km'], 'gco2_per_km', unit_system)}")
    if summary["eta_pt_est"] is not None:
        caption_parts.append(f"Efficiency: {format_value(summary['eta_pt_est'], 'ratio', unit_system)}")
    if caption_parts:
        st.caption(" · ".join(caption_parts))


def _render_vde_comparison(dataset: ComparisonDataset, unit_system: str) -> None:
    st.markdown("**Vehicle Demand**")
    boundary_choice = st.radio("Boundary", ["TOTAL", "NET", "Both"], horizontal=True, key="dashboard_vde_boundary")
    boundaries = {"TOTAL": ["vde_total"], "NET": ["vde_net"], "Both": ["vde_total", "vde_net"]}[boundary_choice]
    rows = []
    excluded = []
    for metric_key in boundaries:
        result = build_metric_bar_rows(dataset, metric_key, unit_system=unit_system)
        group = "TOTAL" if metric_key == "vde_total" else "NET"
        rows += [{"label": r.label, "value": r.value, "group": group} for r in result["rows"]]
        excluded += result["excluded"]
    _render_exclusions(excluded)
    if rows:
        y_title = f"VDE [{unit_label('energy_per_distance', unit_system)}]"
        st.plotly_chart(build_grouped_bar_figure(rows, y_title=y_title), width="stretch")


def _render_fuel_energy_co2(dataset: ComparisonDataset, unit_system: str) -> None:
    st.markdown("**Fuel / Energy / Emissions**")
    any_rendered = False
    for metric_key, label in (
        ("fuel_l_per_100km", "Fuel consumption"),
        ("fuel_km_per_l", "Fuel economy"),
        ("energy_wh_per_km", "Energy consumption"),
        ("gco2_per_km", "CO2"),
    ):
        result = build_metric_bar_rows(dataset, metric_key, unit_system=unit_system)
        if not result["rows"]:
            continue  # dataset-aware: omit entirely rather than showing an empty chart (Sec 44)
        any_rendered = True
        st.caption(label)
        rows = [{"label": r.label, "value": r.value} for r in result["rows"]]
        st.plotly_chart(build_grouped_bar_figure(rows, y_title=label), width="stretch")
        _render_exclusions(result["excluded"])
    if not any_rendered:
        st.caption("No fuel/energy/CO2 data available for the selected scenarios.")


def _render_fe_vde(dataset: ComparisonDataset, unit_system: str) -> None:
    st.markdown("**FE × VDE**")
    mode_label = st.selectbox("Mode", ["Volumetric", "Energy-normalized", "Electrical"], key="dashboard_fe_vde_mode")
    mode = {"Volumetric": "volumetric", "Energy-normalized": "energy_normalized", "Electrical": "electrical"}[mode_label]
    boundary = st.radio("VDE boundary", ["TOTAL", "NET"], horizontal=True, key="dashboard_fe_vde_boundary")

    points_result = build_fe_vde_points(dataset, boundary=boundary, mode=mode)
    _render_exclusions(points_result["excluded"])
    points = points_result["points"]
    if not points:
        st.info("No scenarios are compatible with this FE × VDE mode/boundary.")
        return

    xs = [p["x"] for p in points]
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min, x_max = x_min * 0.8, x_max * 1.2 + 0.1
    else:
        span = x_max - x_min
        x_min, x_max = x_min - span * 0.2, x_max + span * 0.2
    fuel_type = (dataset.reference.fuel_energy or {}).get("fuel_type") if mode == "volumetric" else None
    lines = build_iso_pse_lines(x_min, x_max, _DEFAULT_ETA_LINES[mode], mode=mode, fuel_type=fuel_type)

    x_title = f"VDE {boundary} [{unit_label('energy_per_distance', unit_system)}]"
    fig = build_fe_vde_figure(points, lines, x_title=x_title, y_title=_FE_VDE_Y_TITLE[mode])
    st.plotly_chart(fig, width="stretch")


def _render_competitor_delta(dataset: ComparisonDataset) -> None:
    st.markdown("**Reference-relative KPI delta**")
    metric_key = st.selectbox(
        "KPI",
        _DELTA_METRIC_OPTIONS,
        format_func=lambda key: (get_metric(key).label if get_metric(key) else key),
        key="dashboard_delta_metric",
    )
    result = build_competitor_delta_rows(dataset, metric_key)
    _render_exclusions(result["excluded"])
    rows = [
        {"label": r["label"], "value": r["percent_delta"], "semantic": r["semantic"]}
        for r in result["rows"]
        if r["percent_delta"] is not None
    ]
    if not rows:
        st.info("No compatible numeric data for this KPI.")
        return
    st.plotly_chart(build_grouped_bar_figure(rows, y_title="% vs Reference", color_by_semantic=True), width="stretch")


def _render_dashboard_tab(dataset: ComparisonDataset | None) -> None:
    if dataset is None:
        st.info("Select a reference scenario to begin comparison.")
        return

    unit_system = normalize_unit_system(st.session_state.get("unit_system"))
    _render_reference_summary(dataset, unit_system)
    warnings = dataset_warnings_summary(dataset)
    if warnings:
        with st.expander(f"{len(warnings)} dataset warning(s)", expanded=False):
            for warning in warnings:
                st.warning(warning)

    st.divider()
    _render_vde_comparison(dataset, unit_system)
    st.divider()
    _render_fuel_energy_co2(dataset, unit_system)
    st.divider()
    _render_fe_vde(dataset, unit_system)
    st.divider()
    _render_competitor_delta(dataset)


# -----------------------------------------------------------------------------
# Roadload & VDE tab (Package 8C)
# -----------------------------------------------------------------------------


def _vde_option_label(row: dict) -> str:
    make = str(row.get("make") or "").strip()
    model = str(row.get("model") or "").strip()
    base = " ".join(part for part in (make, model) if part) or f"VDE #{row.get('vde_id')}"
    meta = " · ".join(str(x) for x in (row.get("year"), row.get("legislation")) if x)
    return f"{base} · {meta}" if meta else base


def _render_direct_vde_selection() -> tuple[dict, list[dict]] | None:
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
    # the field name.
    state: SelectionState = st.session_state.setdefault(_DIRECT_VDE_SELECTION_KEY, SelectionState())
    reference_ids = [None] + visible_ids
    reference_choice = st.selectbox(
        "Reference VDE",
        options=reference_ids,
        format_func=lambda vid: "Select a reference VDE..." if vid is None else options_by_id.get(vid, f"VDE #{vid}"),
        index=_index_of(state.reference_fuelcons_id, reference_ids),
        key="roadload_direct_reference_select",
    )
    if reference_choice != state.reference_fuelcons_id:
        state = set_reference(state, reference_choice)

    comparison_choice_ids = st.multiselect(
        "Compare with (up to 10)",
        options=[vid for vid in visible_ids if vid != state.reference_fuelcons_id],
        default=[vid for vid in state.comparison_fuelcons_ids if vid in visible_ids and vid != state.reference_fuelcons_id],
        format_func=lambda vid: options_by_id.get(vid, f"VDE #{vid}"),
        key="roadload_direct_compare_with_select",
    )
    state, errors = sync_comparisons_from_widget(state, comparison_choice_ids)
    st.session_state[_DIRECT_VDE_SELECTION_KEY] = state
    for error in errors:
        st.warning(error)

    if state.reference_fuelcons_id is None:
        return None
    reference_spec = {"kind": "VDE_ONLY", "vde_id": state.reference_fuelcons_id}
    comparison_specs = [{"kind": "VDE_ONLY", "vde_id": vde_id} for vde_id in state.comparison_fuelcons_ids]
    return reference_spec, comparison_specs


def _linked_vde_specs(scorecard_dataset: ComparisonDataset | None) -> tuple[dict, list[dict]] | None:
    if scorecard_dataset is None or scorecard_dataset.reference.vde_id is None:
        return None
    seen = {scorecard_dataset.reference.vde_id}
    comparison_specs = []
    for item in scorecard_dataset.comparisons:
        if item.vde_id is None or item.vde_id in seen:
            continue
        seen.add(item.vde_id)
        comparison_specs.append({"kind": "VDE_ONLY", "vde_id": item.vde_id})
    return {"kind": "VDE_ONLY", "vde_id": scorecard_dataset.reference.vde_id}, comparison_specs


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
        for item in (dataset.reference, *dataset.comparisons)
        if item.vde_id is not None and not item.roadload["net"].available and item.vde_id not in temp_by_vde
    ]
    active_ids = [
        item.vde_id
        for item in (dataset.reference, *dataset.comparisons)
        if item.vde_id is not None and item.vde_id in temp_by_vde
    ]

    for vde_id in dict.fromkeys(active_ids):
        item = next(i for i in (dataset.reference, *dataset.comparisons) if i.vde_id == vde_id)
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


def _render_abc_section(dataset: ComparisonDataset, boundaries: list[str]) -> None:
    st.markdown("**Roadload ABC**")
    for boundary in boundaries:
        result = build_abc_rows(dataset, boundary)
        _render_exclusions(result["excluded"])
        if not result["rows"]:
            continue
        cols = st.columns(3)
        for coefficient, col in zip("ABC", cols):
            rows = [{"label": row["label"], "value": row[coefficient]} for row in result["rows"]]
            col.plotly_chart(build_grouped_bar_figure(rows, y_title=f"{coefficient} ({boundary})"), width="stretch")


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
    st.markdown("**Cycle / phase VDE**")
    reference_legislation = dataset.reference.vehicle.get("legislation")
    rows = []
    excluded = []
    for item in (dataset.reference, *dataset.comparisons):
        if item.vehicle.get("legislation") != reference_legislation:
            excluded.append({"label": item.label, "reason": "Different cycle basis"})
            continue
        cycle_results = item.vde["cycle_results"]
        for boundary in boundaries:
            result = cycle_results[boundary.lower()]
            if result.aggregate is None:
                excluded.append({"label": item.label, "reason": f"Cycle {boundary} unavailable"})
                continue
            rows.append({"label": item.label, "value": result.aggregate, "group": boundary})
    _render_exclusions(excluded)
    if rows:
        y_title = f"VDE [{unit_label('energy_per_distance', unit_system)}]"
        st.plotly_chart(build_grouped_bar_figure(rows, y_title=y_title), width="stretch")


def _render_cycle_demand_section(dataset: ComparisonDataset, boundaries: list[str]) -> None:
    st.markdown("**Demanded power over cycle**")
    if not st.checkbox("Show demanded power over cycle", value=False, key="roadload_show_cycle_demand"):
        st.caption("Calculates cycle-integrated power traces on demand -- opt in above.")
        return

    legislation = dataset.reference.vehicle.get("legislation")
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


def _render_roadload_tab(scorecard_dataset: ComparisonDataset | None) -> None:
    source = st.radio(
        "Source", ["VDEs linked to selected scenarios", "Select VDEs directly"], key="roadload_source_mode"
    )
    specs = (
        _render_direct_vde_selection()
        if source == "Select VDEs directly"
        else _linked_vde_specs(scorecard_dataset)
    )
    if specs is None:
        st.info("Select a reference scenario/VDE to begin physical analysis.")
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
    _render_abc_section(dataset, boundaries)
    st.divider()
    _render_roadload_curve_section(dataset, boundaries)
    st.divider()
    _render_cycle_phase_section(dataset, boundaries, unit_system)
    st.divider()
    _render_cycle_demand_section(dataset, boundaries)


# -----------------------------------------------------------------------------
# Legacy bridge and page entry point
# -----------------------------------------------------------------------------


def _render_legacy_bridge() -> None:
    st.divider()
    with st.expander("Legacy comparison tools", expanded=False):
        st.caption(
            "Scenario Compare, Method Analysis, Peers & Outlook, and Saved Estimates from the previous "
            "Comparison Report. Kept temporarily; capabilities move into Scorecard, Dashboard, and "
            "Roadload & VDE as those packages land, and this section will be retired."
        )
        vde_id, vde_row = resolve_comparison_report_anchor()
        if not vde_id:
            st.info("No VDE source could be resolved yet.")
            return
        render_comparison_report_page(vde_id, vde_row)


def render_comparison_report() -> None:
    st.title("Comparison Report")
    st.caption("Reference-relative engineering comparison across complete FuelCons scenarios.")

    catalog_rows = _load_catalog()
    if not catalog_rows:
        st.info("No FuelCons scenarios are available yet. Save at least one Powertrain Scenario first.")
        _render_legacy_bridge()
        return

    state = _render_selection(catalog_rows)
    dataset = _build_scorecard_dataset(state)

    scorecard_tab, dashboard_tab, roadload_tab, explore_tab = st.tabs(
        ["Scorecard", "Dashboard", "Roadload & VDE", "Explore"]
    )
    with scorecard_tab:
        _render_scorecard_tab(dataset)
    with dashboard_tab:
        _render_dashboard_tab(dataset)
    with roadload_tab:
        _render_roadload_tab(dataset)
    with explore_tab:
        st.info("Explore Lite is planned for a future package.")

    _render_legacy_bridge()


__all__ = ["render_comparison_report"]
