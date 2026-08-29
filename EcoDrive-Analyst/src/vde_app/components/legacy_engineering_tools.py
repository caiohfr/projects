"""Selective entry points for retained legacy and compatibility workflows."""

from __future__ import annotations

import streamlit as st

from src.vde_app.components.pwt_fuel_energy import (
    render_active_vde_source_bar,
    render_legacy_comparison_workspace,
    render_powertrain_conversion_workspace,
    render_powertrain_technical_footer,
    render_saved_scenarios_panel,
    render_technology_proposal_workspace,
    resolve_comparison_report_anchor,
)


_PLACEHOLDER = "Choose a legacy area"
_POWERTRAIN = "Powertrain Legacy"
_COMPARISON = "Comparison Legacy"
_ESTIMATES = "Estimate / Snapshot Management"


def _legacy_notice() -> None:
    st.warning(
        "Legacy / compatibility workspace. Retained tools support investigation, "
        "backward compatibility, and reference; new analysis should use the canonical EcoDrive workflows.",
        icon="⚠️",
    )


def _render_powertrain_legacy() -> None:
    st.subheader("Powertrain Legacy")
    st.caption("Scenario pairing, baseline methods, metadata review, and Technology Delta staging.")
    vde_id, vde_row = render_active_vde_source_bar()
    if not vde_id or not vde_row:
        return
    render_powertrain_conversion_workspace(vde_id, vde_row)
    st.divider()
    render_technology_proposal_workspace(vde_id, vde_row)
    if st.checkbox("Load legacy technical diagnostics", key="legacy_powertrain_load_technical"):
        render_powertrain_technical_footer(vde_id, vde_row)


def _render_comparison_legacy() -> None:
    st.subheader("Comparison Legacy")
    st.caption("Historical Scenario Compare, method analysis, peer outlook, and saved-estimate views.")
    vde_id, vde_row = resolve_comparison_report_anchor()
    if not vde_id or not vde_row:
        st.info("No VDE source could be resolved yet.")
        return
    render_legacy_comparison_workspace(vde_id, vde_row)


def _render_estimate_management() -> None:
    st.subheader("Estimate / Snapshot Management")
    st.caption("Inspect and maintain historical FuelCons estimates against their VDE source snapshots.")
    vde_id, vde_row = render_active_vde_source_bar()
    if vde_id and vde_row:
        render_saved_scenarios_panel(vde_id)


def render_legacy_engineering_tools() -> None:
    """Render only the explicitly selected retained workflow."""

    st.title("Legacy & Engineering Tools")
    _legacy_notice()
    area = st.radio(
        "Legacy area",
        [_PLACEHOLDER, _POWERTRAIN, _COMPARISON, _ESTIMATES],
        key="legacy_engineering_area",
        horizontal=True,
    )
    if area == _POWERTRAIN:
        _render_powertrain_legacy()
    elif area == _COMPARISON:
        _render_comparison_legacy()
    elif area == _ESTIMATES:
        _render_estimate_management()
    else:
        st.info("Select an area to load its retained tools. Unselected legacy workflows do not execute.")


__all__ = ["render_legacy_engineering_tools"]
