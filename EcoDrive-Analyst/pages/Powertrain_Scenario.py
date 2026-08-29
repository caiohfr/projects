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
    st.title("Powertrain System Scenarios")
    render_powertrain_sidebar_controls()
    st.caption(
        "Compose Current and up to three independent multi-domain Proposals, then calculate all ready scenarios through Energy Balance L0."
    )

    render_system_scenario_workspace()


if __name__ == "__main__":
    main()
