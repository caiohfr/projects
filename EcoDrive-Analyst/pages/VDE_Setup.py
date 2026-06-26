import streamlit as st
from pathlib import Path

from src.vde_core.db import ensure_db
from src.vde_core.services import load_vde_defaults
from src.vde_app.state import ensure_vde_setup_state, reset_vde_setup_state
from src.vde_app.components.shared import search_logo, get_legislation_icon
from src.vde_app.components.vde_setup import (
    apply_summary_navigation_from_query_params,
    render_auxiliaries_section,
    render_baseline_picker_and_editor_panel,
    render_component_build_up_panel,
    render_cycle_section,
    render_compute_and_save_panel,
    render_executive_summary_panel,
    render_from_test_section,
    render_mass_setup_section,
    render_vde_results_review_panel,
    render_scenario_origin_section,
    render_technical_build_up_view_selector,
    render_transmission_losses_section,
    render_vehicle_aero_section,
    render_vehicle_meta_header,
    render_vehicle_basics_sidebar,
    render_vde_setup_view_selector,
    render_vde_edit_delete_panel,
)

st.set_page_config(page_title="Mock Data / Editor", layout="wide")
ensure_db()


ABS_DIR = Path(__file__).resolve().parent
default_path = ABS_DIR.parent / 'data' / 'standards' / 'vde_defaults_by_category_trans_elec.csv'
DEFAULTS_PATH = Path(default_path)


@st.cache_resource(show_spinner=False)
def get_defaults_df():
    return load_vde_defaults(DEFAULTS_PATH)


def show_if_exists(col, path, *, width=64, caption=None):
    p = Path(path) if path else None
    with col:
        if p and p.exists():
            st.image(str(p), width=width, caption=caption)


def inject_vde_setup_style():
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px;
        }
        .vde-step-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.15rem;
        }
        .vde-step-caption {
            color: #667085;
            font-size: 0.92rem;
            margin-bottom: 0.85rem;
        }
        .vde-summary-chip {
            padding: 0.45rem 0.7rem;
            border: 1px solid rgba(49, 130, 246, 0.18);
            border-radius: 8px;
            background: rgba(248, 250, 252, 0.95);
            margin-bottom: 0.35rem;
            min-height: 5.8rem;
        }
        .vde-summary-chip.is-ok {
            border-color: rgba(34, 197, 94, 0.28);
            background: rgba(240, 253, 244, 0.95);
        }
        .vde-summary-chip.is-pending {
            border-color: rgba(245, 158, 11, 0.28);
            background: rgba(255, 251, 235, 0.98);
        }
        .vde-summary-chip.is-warn {
            border-color: rgba(239, 68, 68, 0.24);
            background: rgba(254, 242, 242, 0.98);
        }
        .vde-summary-chip-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.45rem;
            margin-bottom: 0.2rem;
        }
        .vde-summary-chip strong {
            display: block;
            font-size: 0.78rem;
            color: #475467;
        }
        .vde-summary-chip span {
            font-size: 0.95rem;
            color: #101828;
        }
        .vde-summary-status {
            display: inline-flex;
            align-items: center;
            gap: 0.2rem;
            font-size: 0.72rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .vde-summary-status.is-ok {
            color: #166534;
        }
        .vde-summary-status.is-pending {
            color: #b45309;
        }
        .vde-summary-status.is-warn {
            color: #b42318;
        }
        .vde-summary-status-icon {
            display: inline-flex;
            width: 0.95rem;
            justify-content: center;
        }
        .vde-summary-chip-detail {
            margin-top: 0.3rem;
            font-size: 0.76rem;
            color: #667085;
            line-height: 1.3;
        }
        .vde-summary-link {
            display: block;
            text-decoration: none !important;
        }
        .vde-summary-link:hover .vde-summary-chip {
            box-shadow: 0 0 0 1px rgba(49, 130, 246, 0.16);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_step_header(number: int, title: str, caption: str):
    st.markdown(f"<div class='vde-step-title'>{number}. {title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='vde-step-caption'>{caption}</div>", unsafe_allow_html=True)


def main():
    ensure_vde_setup_state(st.session_state)
    apply_summary_navigation_from_query_params()
    inject_vde_setup_style()

    ctx = st.session_state.ctx
    baseline_mode = ctx["mode"] == "From baseline (editable)"
    roadload_base_row = dict(ctx.get("selected_baseline_row") or ctx.get("baseline_dict") or {}) if baseline_mode else None
    roadload_saved_vde_id = ctx.get("vde_id_parent") if baseline_mode else None
    roadload_transmission_prefill = dict(ctx.get("baseline_dict") or {}) if baseline_mode else None

    h1, i1, i2 = st.columns([1.0, 0.12, 0.12])
    with h1:
        st.title("VDE Setup")
        st.caption("Roadload / Component Build-up / Transmission Losses / VDE TOTAL-NET workflow")
    render_vehicle_basics_sidebar(reset_ctx=lambda preserve_meta=True: reset_vde_setup_state(st.session_state, preserve_meta=preserve_meta))

    logo_path = search_logo(ctx, base_dir="data/images/logos", fallback="_unknown.png") or ""
    leg_icon = get_legislation_icon(ctx, base_dir="data/images") or ""

    ctx["brand_icon"] = logo_path
    ctx["leg_icon"] = leg_icon

    show_if_exists(i1, ctx["brand_icon"], width=50, caption=ctx["make"])
    show_if_exists(i2, ctx["leg_icon"], width=50, caption=ctx["legislation"])

    render_executive_summary_panel()
    active_view = render_vde_setup_view_selector()

    st.divider()

    if active_view == "Scenario Setup":
        with st.container(border=True):
            st.subheader("Scenario Setup")
            st.caption("Define the scenario identity, vehicle context, and the origin path that anchors this VDE workflow.")
            render_vehicle_meta_header()
        with st.container(border=True):
            render_step_header(1, "Scenario Origin", "Choose whether this scenario starts from a saved baseline snapshot or from a brand-new manual/test path.")
            render_scenario_origin_section(
                reset_ctx=lambda preserve_meta=True: reset_vde_setup_state(st.session_state, preserve_meta=preserve_meta),
            )
            if ctx["mode"] == "From baseline (editable)":
                render_baseline_picker_and_editor_panel()
            else:
                st.info("Manual/test origin is active. This scenario will be built from the current page state without loading a baseline snapshot.")
            if ctx.get("abc_total_source_ui") == "From test coastdown":
                st.divider()
                render_from_test_section()
                render_auxiliaries_section(defaults_df_getter=get_defaults_df)
    elif active_view == "Vehicle Parameters":
        with st.container(border=True):
            render_step_header(2, "Vehicle Parameters", "Define the vehicle-level physical parameters that feed the roadload workflow.")
            render_mass_setup_section(
                prefill=dict(ctx.get("selected_baseline_row") or ctx.get("baseline_dict") or {}) if ctx["mode"] == "From baseline (editable)" else None
            )
            st.divider()
            render_vehicle_aero_section(
                base_row=dict(ctx.get("selected_baseline_row") or ctx.get("baseline_dict") or {}) if ctx["mode"] == "From baseline (editable)" else None
            )
    elif active_view == "Roadload Build-up":
        with st.container(border=True):
            render_step_header(3, "Roadload Build-up", "Configure tires, brake drag, parasitics, trailer placeholder, and transmission for the roadload path selected in Scenario Setup.")
            roadload_view = render_technical_build_up_view_selector()
            if roadload_view == "Transmission":
                render_transmission_losses_section(prefill=roadload_transmission_prefill)
            else:
                ctx["component_editor_active"] = roadload_view
                render_component_build_up_panel(
                    base_row=roadload_base_row,
                    saved_vde_id=roadload_saved_vde_id,
                )

    elif active_view == "Cycle & Preview":
        with st.container(border=True):
            render_step_header(4, "Cycle & Preview", "Choose the standard cycle or upload a custom cycle, then review the phase-sensitive VDE preview.")
            render_cycle_section()
    elif active_view == "Results":
        with st.container(border=True):
            render_step_header(5, "Results", "Review the resolved preview, reference context, staged payload, and technical changes before deciding whether to persist the snapshot.")
            render_vde_results_review_panel()
    elif active_view == "Save / Edit":
        with st.container(border=True):
            render_step_header(6, "Save / Edit", "Persist the current snapshot, update an existing one, or manage saved scenarios only after the preview looks right.")
            render_compute_and_save_panel(
                defaults_df_getter=get_defaults_df,
                reset_ctx=lambda preserve_meta=True: reset_vde_setup_state(st.session_state, preserve_meta=preserve_meta),
            )
            st.divider()
            st.caption("Saved scenario maintenance and destructive actions remain confirmation-protected, but now live in the same workflow area as save/update.")
            render_vde_edit_delete_panel(
                defaults_path=DEFAULTS_PATH,
                defaults_df_getter=get_defaults_df,
                reset_ctx=lambda preserve_meta=True: reset_vde_setup_state(st.session_state, preserve_meta=preserve_meta),
            )


if __name__ == "__main__":
    main()
