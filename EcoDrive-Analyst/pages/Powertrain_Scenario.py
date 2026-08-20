from __future__ import annotations

import streamlit as st

from src.vde_core.db import ensure_db
from src.vde_app.state import ensure_pwt_sidebar_defaults
from src.vde_app.components.pwt_fuel_energy import (
    inject_powertrain_scenario_style,
    render_active_vde_source_bar,
    render_powertrain_conversion_workspace,
    render_powertrain_sidebar_controls,
    render_powertrain_scenario_bench,
    render_powertrain_technical_footer,
    render_powertrain_step_header,
    render_powertrain_step_navigation,
    render_results_save_tab,
    render_saved_scenarios_panel,
    render_technology_proposal_workspace,
)


st.set_page_config(page_title="EcoDrive - Powertrain Calculation Sheet", layout="wide")
ensure_db()


def main():
    ensure_pwt_sidebar_defaults(st.session_state)
    inject_powertrain_scenario_style()
    st.title("Powertrain Calculation Sheet")
    input_mode = render_powertrain_sidebar_controls()
    st.caption(
        "Baseline PSE, fuel/CO2 estimate, technology delta and proposal."
        if input_mode == "Guided"
        else "Used by analysts to estimate baseline and proposal fuel/CO2 from active VDE. Program-facing comparison lives in Comparison Report."
    )

    vde_id, vde_row = render_active_vde_source_bar()
    if not vde_id:
        st.stop()

    st.session_state["current_vde_id"] = int(vde_id)
    render_powertrain_scenario_bench(vde_id, vde_row)
    active_step = render_powertrain_step_navigation()

    st.divider()

    if active_step == "Baseline Estimate":
        with st.container(border=True):
            render_powertrain_step_header(
                1,
                "Baseline Estimate",
                "Confirm demand and powertrain reference, review only the active estimation method, and lock the baseline before applying any delta.",
            )
            render_powertrain_conversion_workspace(vde_id, vde_row)
    elif active_step == "Technology Delta":
        with st.container(border=True):
            render_powertrain_step_header(
                2,
                "Technology Delta",
                "Stage technology deltas on top of the baseline and keep registered-only deltas explicit when no quantitative model is available.",
            )
            render_technology_proposal_workspace(vde_id, vde_row)
    else:
        with st.container(border=True):
            render_powertrain_step_header(
                3,
                "Result & Save",
                "Review the final baseline vs proposal comparison and save the scenario.",
            )
            render_results_save_tab(vde_id, vde_row)
            with st.expander("Saved Estimates", expanded=False):
                render_saved_scenarios_panel(vde_id)

    st.divider()
    render_powertrain_technical_footer(vde_id, vde_row)


if __name__ == "__main__":
    main()
