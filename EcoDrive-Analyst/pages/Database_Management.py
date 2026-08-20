from __future__ import annotations

import streamlit as st

from src.vde_app.components.database_management import render_database_management
from src.vde_core.db import configure_db_path, current_db_path, ensure_db


st.set_page_config(page_title="EcoDrive - Database Management", layout="wide")


def _sync_runtime_db_path() -> str:
    context = st.session_state.get("ctx")
    current = str(context.get("db_path") or "").strip() if isinstance(context, dict) else ""
    selected = st.sidebar.text_input("DB path", value=current or str(current_db_path()), key="database_management_runtime_db_path")
    resolved = str(configure_db_path(selected or str(current_db_path())))
    updated_context = dict(context or {})
    updated_context["db_path"] = resolved
    st.session_state.ctx = updated_context
    return resolved


def main() -> None:
    with st.sidebar:
        st.header("Database Management")
        st.caption("Staged catalog administration with explicit review and audit receipts.")
        runtime_path = _sync_runtime_db_path()
        ensure_db()
        st.caption(f"Runtime DB: {runtime_path}")
    render_database_management()


if __name__ == "__main__":
    main()
