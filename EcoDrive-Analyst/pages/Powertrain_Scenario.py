from __future__ import annotations

import streamlit as st

from src.vde_core.db import ensure_db
from src.vde_app.state import ensure_pwt_sidebar_defaults
from src.vde_app.components.pwt_fuel_energy import (
    inject_powertrain_scenario_style,
    render_active_vde_source_bar,
    render_powertrain_conversion_workspace,
    render_powertrain_sidebar_controls,
    render_powertrain_technical_footer,
    render_saved_scenarios_panel,
    render_technology_proposal_workspace,
)
from src.vde_app.components.pwt_system_scenario import render_system_scenario_workspace


st.set_page_config(page_title="EcoDrive - Powertrain Calculation Sheet", layout="wide")
ensure_db()


def main():
    ensure_pwt_sidebar_defaults(st.session_state)
    inject_powertrain_scenario_style()
    st.title("Powertrain System Scenarios")
    render_powertrain_sidebar_controls()
    st.caption(
        "Compose Current and up to three independent multi-domain Proposals, then calculate all ready scenarios through Energy Balance L0."
    )

    vde_id, vde_row = render_active_vde_source_bar()
    if not vde_id:
        st.stop()

    st.session_state["current_vde_id"] = int(vde_id)
    render_system_scenario_workspace(vde_id, vde_row)

    st.divider()
    with st.expander("Evidence and recommendation workbench", expanded=False):
        st.caption(
            "Observed, benchmark, ML, regression, manual and Technology Delta tools remain available as engineering evidence. "
            "They do not change a System Scenario until a value or delta is explicitly adopted in its Domain editor."
        )
        if st.checkbox("Load existing evidence tools", key="pwt_ss_load_evidence_tools"):
            render_powertrain_conversion_workspace(vde_id, vde_row)
            st.divider()
            render_technology_proposal_workspace(vde_id, vde_row)

    with st.expander("Legacy saved estimates", expanded=False):
        st.caption(
            "Existing FuelCons records remain inspectable. Saving a complete multi-domain SystemScenarioResult is deferred because the legacy schema cannot represent its full composition truthfully."
        )
        if st.checkbox("Load legacy saved estimates", key="pwt_ss_load_saved_estimates"):
            render_saved_scenarios_panel(vde_id)

    st.divider()
    render_powertrain_technical_footer(vde_id, vde_row)


if __name__ == "__main__":
    main()
