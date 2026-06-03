from __future__ import annotations

import streamlit as st

from src.vde_core.db import ensure_db
from src.vde_core.pwt_fuel_energy_service import compute_vde_total_from_ctx, fetch_vde_row
from src.vde_app.components.shared import get_legislation_icon, search_logo
from src.vde_app.components.pwt_fuel_energy import (
    apply_bev_placeholders_if_needed,
    fixed_header as render_fixed_header,
    render_sidebar_vde_selector_and_context,
    run_regression_panel as render_regression_panel,
    run_view_panel as render_view_panel,
    section_parameters_card as render_parameters_card,
)


st.set_page_config(page_title="EcoDrive - PWT & Fuel/Energy", layout="wide")
ensure_db()


def main():
    st.title("EcoDrive Analyzer 2 - PWT & Fuel/Energy")

    vde_id, ctx = render_sidebar_vde_selector_and_context()
    if not vde_id:
        st.stop()

    apply_bev_placeholders_if_needed(vde_id, ctx["electrification"])

    vde_row = fetch_vde_row(vde_id)
    totals = compute_vde_total_from_ctx(vde_row, ctx)
    vde_net = totals["vde_net_mj_per_km"]

    vde_row["brand_icon"] = search_logo(vde_row, base_dir="data/images/logos", fallback="_unknown.png") or ""
    vde_row["leg_icon"] = get_legislation_icon(vde_row, base_dir="data/images") or ""
    render_fixed_header(vde_row)

    st.session_state["current_vde_id"] = int(vde_id)
    mode = st.radio("Mode", ["View", "Parameters", "Regression"], horizontal=True, key="mode_sel")

    if mode == "View":
        render_view_panel(vde_id, vde_row, ctx)
    elif mode == "Parameters":
        render_parameters_card(vde_id, vde_net, ctx["electrification"])
    else:
        render_regression_panel(vde_id, vde_row, ctx, vde_net)

    st.markdown("---")


if __name__ == "__main__":
    main()
