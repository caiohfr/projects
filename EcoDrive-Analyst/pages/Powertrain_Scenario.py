from __future__ import annotations

import streamlit as st

from src.vde_core.db import ensure_db
from src.vde_app.components.pwt_fuel_energy import (
    render_active_vde_source_bar,
    render_estimation_engine_panel,
    render_powertrain_inputs_panel,
    render_results_save_tab,
    render_scenario_definition_section,
    render_saved_scenarios_panel,
)


st.set_page_config(page_title="EcoDrive - Powertrain Scenario", layout="wide")
ensure_db()


def main():
    st.title("Powertrain Scenario")
    st.caption("Context & Energy / Powertrain Inputs / Estimation Engine / Results & Save / Saved Estimates")

    vde_id, vde_row = render_active_vde_source_bar()
    if not vde_id:
        st.stop()

    st.session_state["current_vde_id"] = int(vde_id)

    context_tab, powertrain_tab, engine_tab, results_tab, saved_tab = st.tabs(
        ["Context & Energy", "Powertrain Inputs", "Estimation Engine", "Results & Save", "Saved Estimates"]
    )

    with context_tab:
        render_scenario_definition_section(vde_id, vde_row)

    with powertrain_tab:
        render_powertrain_inputs_panel(vde_id, vde_row)

    with engine_tab:
        render_estimation_engine_panel(vde_id, vde_row)

    with results_tab:
        render_results_save_tab(vde_id, vde_row)

    with saved_tab:
        render_saved_scenarios_panel(vde_id)


if __name__ == "__main__":
    main()
