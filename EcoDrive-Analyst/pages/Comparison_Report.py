from __future__ import annotations

import streamlit as st

from src.vde_core.db import ensure_db
from src.vde_app.components.pwt_fuel_energy import (
    render_comparison_report_page,
    resolve_comparison_report_anchor,
)


st.set_page_config(page_title="EcoDrive - Powertrain Comparison", layout="wide")
ensure_db()


def main():
    st.title("Powertrain Comparison")
    st.caption("Scenario-first comparison, method analysis, and peer outlook are separated from the estimation workflow.")

    vde_id, vde_row = resolve_comparison_report_anchor()
    if not vde_id:
        st.info("No VDE source could be resolved yet. Save at least one Powertrain Scenario or create a VDE snapshot first.")
        st.stop()

    st.session_state["current_vde_id"] = int(vde_id)
    render_comparison_report_page(vde_id, vde_row)


if __name__ == "__main__":
    main()
