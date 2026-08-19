import streamlit as st
from pathlib import Path

from src.vde_core.db import ensure_db
from src.vde_core.services import load_vde_defaults
from src.vde_app.state import ensure_vde_setup_state, reset_vde_setup_state
from src.vde_app.components.vde_setup import render_vde_setup_workbook_v21


st.set_page_config(page_title="EcoDrive - VDE Calculation Sheet v2.1", layout="wide")
ensure_db()

ABS_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = ABS_DIR.parent / "data" / "standards" / "vde_defaults_by_category_trans_elec.csv"


@st.cache_resource(show_spinner=False)
def get_defaults_df():
    return load_vde_defaults(DEFAULTS_PATH)


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
        .vde-context-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.5rem;
            margin: 0.35rem 0 0.75rem 0;
        }
        .vde-context-item,
        .vde-status-chip {
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 0.45rem 0.6rem;
            background: #fbfdff;
            min-height: 4.2rem;
        }
        .vde-context-label,
        .vde-status-label {
            color: #667085;
            font-size: 0.72rem;
            font-weight: 600;
            margin-bottom: 0.15rem;
        }
        .vde-context-value {
            color: #101828;
            font-size: 0.9rem;
            font-weight: 600;
            overflow-wrap: anywhere;
        }
        .vde-status-chip {
            min-height: 4.55rem;
        }
        .vde-status-chip.is-ok {
            border-color: rgba(34, 197, 94, 0.28);
            background: rgba(240, 253, 244, 0.95);
        }
        .vde-status-chip.is-pending {
            border-color: rgba(245, 158, 11, 0.28);
            background: rgba(255, 251, 235, 0.98);
        }
        .vde-status-chip.is-warn {
            border-color: rgba(239, 68, 68, 0.24);
            background: rgba(254, 242, 242, 0.98);
        }
        .vde-status-chip.is-neutral {
            border-color: #d0d7de;
            background: #f8fafc;
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
        .vde-summary-status.is-neutral {
            color: #475467;
        }
        .vde-summary-status-icon {
            display: inline-flex;
            width: 0.95rem;
            justify-content: center;
        }
        .vde-status-detail {
            margin-top: 0.16rem;
            color: #667085;
            font-size: 0.72rem;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        @media (max-width: 900px) {
            .vde-context-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    ensure_vde_setup_state(st.session_state)
    inject_vde_setup_style()

    with st.sidebar:
        st.header("VDE Setup v2.1")
        st.caption("Single proposal workbook page. VDE Setup v2 remains available separately.")
        if st.button("Reset v2.1 draft", key="reset_vde_setup_v21_draft"):
            st.session_state.pop("vde_setup_workbook_v21", None)
            reset_vde_setup_state(st.session_state, preserve_meta=True)
            st.rerun()

    st.title("VDE Calculation Sheet")
    st.caption("Scenario proposal workbook with hierarchical domain requests.")
    render_vde_setup_workbook_v21(
        defaults_df_getter=get_defaults_df,
        defaults_path=DEFAULTS_PATH,
        reset_ctx=lambda preserve_meta=True: reset_vde_setup_state(st.session_state, preserve_meta=preserve_meta),
    )


if __name__ == "__main__":
    main()
