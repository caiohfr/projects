# src/vde_app/components/comparison_report.py
# -----------------------------------------------------------------------------
# Package 8B - dedicated Comparison Report UI owner. Replaces
# pwt_fuel_energy.py as the entry point for the Comparison product; the old
# renderer (Scenario Compare / Method Analysis / Peers & Outlook / Saved
# Estimates) stays intact and reachable behind "Legacy comparison tools"
# until 8C/8D absorb its useful capabilities.
#
# This module never queries SQLite directly -- it only calls
# comparison_report_service.py / comparison_report_viewmodels.py.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.vde_app.comparison_report_viewmodels import (
    MAX_COMPARISONS,
    ScorecardSection,
    SelectionState,
    build_scenario_header,
    build_scenario_options,
    build_scorecard_sections,
    dataset_warnings_summary,
    set_reference,
    sync_comparisons_from_widget,
)
from src.vde_app.components.pwt_fuel_energy import (
    render_comparison_report_page,
    resolve_comparison_report_anchor,
)
from src.vde_app.units import normalize_unit_system
from src.vde_core.comparison_report_service import build_comparison_dataset, list_comparison_scenarios
from src.vde_core.db import current_db_path

_CELL_STYLE = {
    "BETTER": "background-color: rgba(34,197,94,0.18)",
    "WORSE": "background-color: rgba(239,68,68,0.18)",
}

_SELECTION_KEY = "comparison_selection"


@st.cache_data(show_spinner=False)
def _load_catalog_cached(db_path_signature: str) -> list[dict]:
    return list_comparison_scenarios()


def _load_catalog() -> list[dict]:
    return _load_catalog_cached(str(Path(current_db_path()).resolve()))


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


def _index_of(fuelcons_id: int | None, ordered_ids: list[int | None]) -> int:
    try:
        return ordered_ids.index(fuelcons_id)
    except ValueError:
        return 0


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


def _render_scorecard_tab(state: SelectionState) -> None:
    if state.reference_fuelcons_id is None:
        st.info("Select a reference scenario to begin comparison.")
        return

    reference_spec = {"kind": "FUELCONS_SCENARIO", "fuelcons_id": state.reference_fuelcons_id}
    comparison_specs = [{"kind": "FUELCONS_SCENARIO", "fuelcons_id": cid} for cid in state.comparison_fuelcons_ids]
    try:
        dataset = build_comparison_dataset(reference_spec, comparison_specs)
    except ValueError as exc:
        st.error(str(exc))
        return

    warnings = dataset_warnings_summary(dataset)
    if warnings:
        with st.expander(f"{len(warnings)} dataset warning(s)", expanded=False):
            for warning in warnings:
                st.warning(warning)

    items = (dataset.reference, *dataset.comparisons)
    header_titles = [build_scenario_header(item)["column_title"] for item in items]

    unit_system = normalize_unit_system(st.session_state.get("unit_system"))
    for section in build_scorecard_sections(dataset, unit_system=unit_system):
        _render_section(section, header_titles)


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

    scorecard_tab, dashboard_tab, roadload_tab, explore_tab = st.tabs(
        ["Scorecard", "Dashboard", "Roadload & VDE", "Explore"]
    )
    with scorecard_tab:
        _render_scorecard_tab(state)
    with dashboard_tab:
        st.info("Engineering Dashboard charts are planned for a future package.")
    with roadload_tab:
        st.info("Roadload & VDE views are planned for a future package.")
    with explore_tab:
        st.info("Explore Lite is planned for a future package.")

    _render_legacy_bridge()


__all__ = ["render_comparison_report"]
