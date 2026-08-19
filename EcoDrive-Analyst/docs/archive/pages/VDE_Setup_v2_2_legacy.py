import streamlit as st

from src.vde_app.components.vde_request_compact import (
    V22_SESSION_KEY,
    ensure_v22_session_state,
    render_v22_sidebar_navigation,
    render_vde_request_compact,
)
from src.vde_app.components.vde_request_compact_style import inject_v22_styles
from src.vde_app.units import UNIT_SYSTEM_OPTIONS, normalize_unit_system
from src.vde_core.db import configure_db_path, current_db_path, ensure_db


st.set_page_config(page_title="EcoDrive - VDE Setup v2.2", layout="wide")


def _sync_runtime_db_path() -> str:
    ctx = st.session_state.get("ctx")
    candidate = None
    if isinstance(ctx, dict):
        candidate = str(ctx.get("db_path") or "").strip() or None
    resolved = str(configure_db_path(candidate or str(current_db_path())))
    if isinstance(ctx, dict):
        ctx["db_path"] = resolved
        st.session_state.ctx = ctx
    st.session_state["_active_runtime_db_path"] = resolved
    return resolved


def main() -> None:
    inject_v22_styles()
    state = ensure_v22_session_state(st.session_state)
    with st.sidebar:
        st.header("VDE Setup v2.2")
        st.caption("Experimental compact request flow")
        ctx = st.session_state.get("ctx")
        current_db = None
        if isinstance(ctx, dict):
            current_db = str(ctx.get("db_path") or "").strip() or None
        if not current_db:
            current_db = str(current_db_path())
        selected_db_path = st.text_input("DB path", value=current_db, key="v22_runtime_db_path")
        if not isinstance(ctx, dict):
            ctx = {}
        ctx["db_path"] = selected_db_path
        st.session_state.ctx = ctx
        active_db_path = _sync_runtime_db_path()
        ensure_db()
        st.caption(f"Runtime DB: {active_db_path}")
        if st.button("Reset v2.2 request", key="reset_vde_setup_v22"):
            st.session_state.pop(V22_SESSION_KEY, None)
            st.rerun()
        st.divider()
        render_v22_sidebar_navigation(state)
        st.selectbox(
            "Display units",
            UNIT_SYSTEM_OPTIONS,
            index=UNIT_SYSTEM_OPTIONS.index(normalize_unit_system(st.session_state.get("unit_system"))),
            key="unit_system",
        )

    st.title("VDE Request Builder")
    st.caption("Compact baseline, request and scenario workflow.")
    render_vde_request_compact()


if __name__ == "__main__":
    main()
