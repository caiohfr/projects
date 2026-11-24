from __future__ import annotations

import streamlit as st

from src.vde_core.db import ensure_db
from src.vde_app.state import ensure_pwt_sidebar_defaults
from src.vde_app.components.pwt_fuel_energy import (
    inject_powertrain_scenario_style,
    render_powertrain_sidebar_controls,
)
from src.vde_app.components.pwt_system_scenario import render_system_scenario_workspace


st.set_page_config(page_title="EcoDrive - Powertrain Calculation Sheet", layout="wide")
ensure_db()


def main():
    ensure_pwt_sidebar_defaults(st.session_state)
    inject_powertrain_scenario_style()
    render_powertrain_sidebar_controls()
    st.markdown(
        """
        <div class="pwt-page-intro">
            <div>
                <span class="pwt-page-kicker">Canonical Energy Balance L0</span>
                <strong>Powertrain System Scenarios</strong>
                <small>Compose and compare Current plus up to three in-memory proposals from a persisted FuelCons reference.</small>
            </div>
            <div class="pwt-page-pills">
                <span class="pwt-page-pill is-green">Persisted reference required</span>
                <span class="pwt-page-pill">Proposals · in memory</span>
                <span class="pwt-page-pill">No implicit save</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_system_scenario_workspace()


if __name__ == "__main__":
    main()
