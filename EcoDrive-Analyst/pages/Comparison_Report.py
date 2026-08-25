from __future__ import annotations

import streamlit as st

from src.vde_app.components.comparison_report import render_comparison_report
from src.vde_core.db import DEFAULT_DB_PATH, configure_db_path, current_db_path, ensure_db
from src.vde_core.qa_mock_data import DEFAULT_QA_DB_PATH, seed_qa_database


st.set_page_config(page_title="EcoDrive - Comparison Report", layout="wide")

_DB_PATH_WIDGET_KEY = "comparison_report_runtime_db_path"


def _sync_runtime_db_path() -> str:
    """Same apply pattern VDE Setup's own sidebar uses: read whatever the
    text input currently holds (via the shared ctx dict), configure the
    active SQLite path from it, and write the resolved value back so every
    page sharing st.session_state.ctx stays in sync.
    """
    ctx = st.session_state.get("ctx")
    candidate = None
    if isinstance(ctx, dict):
        candidate = str(ctx.get("db_path") or "").strip() or None
    resolved = str(configure_db_path(candidate or str(current_db_path())))
    if isinstance(ctx, dict):
        ctx["db_path"] = resolved
        st.session_state.ctx = ctx
    return resolved


def _switch_db_path(target_path: str) -> None:
    """Programmatically point the DB path text input at `target_path`.
    Streamlit forbids assigning a new value directly into a widget's
    session_state key once that widget has already been instantiated this
    run, so this pops the key instead (same mechanism the Comparison
    Browse "Clear Filters" button uses) -- on the next rerun the input has
    no cached value and falls back to its `value=` argument, which reads
    the just-updated ctx["db_path"].
    """
    ctx = st.session_state.get("ctx")
    if not isinstance(ctx, dict):
        ctx = {}
    ctx["db_path"] = target_path
    st.session_state.ctx = ctx
    st.session_state.pop(_DB_PATH_WIDGET_KEY, None)
    st.rerun()


def main() -> None:
    with st.sidebar:
        st.header("Comparison Report")
        ctx = st.session_state.get("ctx")
        current_db = None
        if isinstance(ctx, dict):
            current_db = str(ctx.get("db_path") or "").strip() or None
        if not current_db:
            current_db = str(current_db_path())
        selected_db_path = st.text_input("DB path", value=current_db, key=_DB_PATH_WIDGET_KEY)
        if not isinstance(ctx, dict):
            ctx = {}
        ctx["db_path"] = selected_db_path
        st.session_state.ctx = ctx

        qa_col, default_col = st.columns(2)
        if qa_col.button("Switch to QA data", key="comparison_report_switch_to_qa", width="stretch"):
            if not DEFAULT_QA_DB_PATH.exists():
                seed_qa_database()
            _switch_db_path(str(DEFAULT_QA_DB_PATH))
        if default_col.button("Switch to default DB", key="comparison_report_switch_to_default", width="stretch"):
            _switch_db_path(str(DEFAULT_DB_PATH))

        active_db_path = _sync_runtime_db_path()
        ensure_db()
        st.caption(f"Runtime DB: {active_db_path}")
        st.divider()

    render_comparison_report()


if __name__ == "__main__":
    main()
