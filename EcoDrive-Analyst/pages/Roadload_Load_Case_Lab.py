from __future__ import annotations

import streamlit as st

from src.vde_app.components.roadload_load_case_lab import render_roadload_load_case_lab
from src.vde_app.units import UNIT_SYSTEM_OPTIONS, normalize_unit_system
from src.vde_core.db import configure_db_path, current_db_path, ensure_db


st.set_page_config(page_title="EcoDrive - Roadload & Load Case Lab", layout="wide")


def _sync_runtime_db_path() -> str:
    ctx = st.session_state.get("ctx")
    candidate = None
    if isinstance(ctx, dict):
        candidate = str(ctx.get("db_path") or "").strip() or None
    resolved = str(configure_db_path(candidate or str(current_db_path())))
    if not isinstance(ctx, dict):
        ctx = {}
    ctx["db_path"] = resolved
    st.session_state.ctx = ctx
    return resolved


def main() -> None:
    with st.sidebar:
        st.header("Vehicle Demand")
        st.markdown("**Roadload & Load Case Lab**")
        st.caption("Temporary environment, load, towing, grade and operating-point analysis.")
        active_db_path = _sync_runtime_db_path()
        ensure_db()
        st.caption(f"Runtime DB: {active_db_path}")
        st.selectbox(
            "Display units",
            UNIT_SYSTEM_OPTIONS,
            index=UNIT_SYSTEM_OPTIONS.index(normalize_unit_system(st.session_state.get("unit_system"))),
            key="unit_system",
        )
        st.info("In-memory analysis · no Save action")

    st.title("Roadload & Load Case Lab")
    st.caption("Temporary Vehicle Demand sensitivity without modifying the persisted VDE baseline.")
    render_roadload_load_case_lab()


if __name__ == "__main__":
    main()
