import streamlit as st
from pathlib import Path

from src.vde_core.db import ensure_db
from src.vde_core.services import load_vde_defaults
from src.vde_app.state import ensure_vde_setup_state, reset_vde_setup_state
from src.vde_app.components.shared import search_logo, get_legislation_icon
from src.vde_app.components.vde_setup import (
    render_auxiliaries_section,
    render_aero_section,
    render_baseline_picker_and_editor_panel,
    render_cycle_section,
    render_compute_and_save_panel,
    render_from_test_section,
    render_live_vde_preview_panel,
    render_parasitic_brake_section,
    render_rr_section,
    render_vehicle_basics_sidebar,
    render_vde_edit_delete_panel,
)

st.set_page_config(page_title="Mock Data / Editor", layout="wide")
ensure_db()


ABS_DIR = Path(__file__).resolve().parent
default_path = ABS_DIR.parent / 'data' / 'standards' / 'vde_defaults_by_category_trans_elec.csv'
DEFAULTS_PATH = Path(default_path)

tire_path = ABS_DIR.parent / 'data' / 'standards' / 'tiresize_fromcode_table.csv'
TIRE_CSV = Path(tire_path)


@st.cache_resource(show_spinner=False)
def get_defaults_df():
    return load_vde_defaults(DEFAULTS_PATH)


def show_if_exists(col, path, *, width=64, caption=None):
    p = Path(path) if path else None
    with col:
        if p and p.exists():
            st.image(str(p), width=width, caption=caption)

def main():
    ensure_vde_setup_state(st.session_state)

    ctx = st.session_state.ctx
    h1, i1, i2, i3 = st.columns([1.0, 0.12, 0.12, 0.12])
    with h1:
        st.title("EcoDrive Analyst - VDE")
        st.caption("Quick setup - clean preview - save/edit snapshots")
    st.divider()

    render_vehicle_basics_sidebar(reset_ctx=lambda preserve_meta=True: reset_vde_setup_state(st.session_state, preserve_meta=preserve_meta))

    logo_path = search_logo(ctx, base_dir="data/images/logos", fallback="_unknown.png") or ""
    leg_icon = get_legislation_icon(ctx, base_dir="data/images") or ""

    ctx["brand_icon"] = logo_path
    ctx["leg_icon"] = leg_icon

    show_if_exists(i1, ctx["brand_icon"], width=50, caption=ctx["make"])
    show_if_exists(i2, ctx["leg_icon"], width=50, caption=ctx["legislation"])

    if ctx["mode"] == "From baseline (editable)":
        render_baseline_picker_and_editor_panel(
            tire_csv=TIRE_CSV,
            rr_section=render_rr_section,
            aero_section=render_aero_section,
            parasitic_brake_section=render_parasitic_brake_section,
        )
    elif ctx["mode"] == "Define all parameters (no baseline)":
        with st.expander("Road load & Curb weight", expanded=True):
            render_rr_section(prefill=None)
        with st.expander("Aerodynamics", expanded=False):
            render_aero_section(prefill=None)
        with st.expander("Parasitic & Brake", expanded=False):
            render_parasitic_brake_section(prefill=None)

    else:
        render_from_test_section()
        render_auxiliaries_section(defaults_df_getter=get_defaults_df)

    render_cycle_section()
    render_live_vde_preview_panel()

    render_compute_and_save_panel(
        defaults_df_getter=get_defaults_df,
        reset_ctx=lambda preserve_meta=True: reset_vde_setup_state(st.session_state, preserve_meta=preserve_meta),
    )
    render_vde_edit_delete_panel(
        defaults_path=DEFAULTS_PATH,
        defaults_df_getter=get_defaults_df,
        reset_ctx=lambda preserve_meta=True: reset_vde_setup_state(st.session_state, preserve_meta=preserve_meta),
    )


if __name__ == "__main__":
    main()
