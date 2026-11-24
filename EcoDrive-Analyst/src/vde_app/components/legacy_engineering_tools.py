"""Selective entry points for retained legacy and compatibility workflows."""

from __future__ import annotations

import streamlit as st

from src.vde_app.components.sprint15d_style import inject_sprint15d_styles
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
        "Experimental / compatibility workspace. Results require their own validation and must not be "
        "presented as canonical EcoDrive outputs without supporting evidence.",
        icon="⚠️",
    )


def _render_capability_map() -> None:
    st.markdown(
        """
        <div class="s15d-card-grid">
          <div class="s15d-card"><span class="s15d-card-icon">PWT</span><strong>Powertrain Legacy</strong><p>Historical pairing, baseline-method and Technology Delta workflows.</p><em>Retained · selectable below</em></div>
          <div class="s15d-card"><span class="s15d-card-icon">CMP</span><strong>Comparison Legacy</strong><p>Historical method analysis, peer outlook and saved-estimate views.</p><em>Retained · selectable below</em></div>
          <div class="s15d-card"><span class="s15d-card-icon">EST</span><strong>Estimate / Snapshot Management</strong><p>Inspect saved FuelCons estimates against their VDE source snapshots.</p><em>Retained · selectable below</em></div>
          <div class="s15d-card is-disabled"><span class="s15d-card-icon">R&amp;D</span><strong>Research capabilities</strong><p>Component estimation and operating-point tools require future contracts and tests.</p><em>Planned / unavailable</em></div>
        </div>
        """,
        unsafe_allow_html=True,
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

    inject_sprint15d_styles()
    st.markdown('<div class="s15d-kicker">Advanced workspace</div>', unsafe_allow_html=True)
    st.title("Engineering Tools / Legacy")
    st.markdown(
        '<div class="s15d-lead">Research, diagnostics and compatibility workflows outside the canonical product path. Prefer Powertrain Scenario, Comparison Report and VDE Setup for supported analysis.</div>',
        unsafe_allow_html=True,
    )
    _legacy_notice()
    _render_capability_map()
    st.markdown('<div class="s15d-section-label">Open one retained area</div>', unsafe_allow_html=True)
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
