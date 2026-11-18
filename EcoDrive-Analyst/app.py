# app.py
import platform
import sys

import streamlit as st
import pages.home_page as home
from src.vde_core.db import configure_db_path, current_db_path

APP_NAME, APP_ICON, APP_VER = "EcoDrive Analyst", "⚡", "0.7.2"
DB_DEFAULT = str(current_db_path())

st.set_page_config(page_title=f"{APP_NAME} {APP_VER}", page_icon=APP_ICON, layout="wide")


def _bootstrap_ctx():
    if "ctx" not in st.session_state or not isinstance(st.session_state.ctx, dict):
        st.session_state.ctx = {"db_path": DB_DEFAULT}
    st.session_state.ctx.setdefault("db_path", DB_DEFAULT)
    return st.session_state.ctx


def get_ctx():
    return _bootstrap_ctx()


def _sidebar(ctx):
    st.sidebar.title(f"{APP_ICON} {APP_NAME}")
    st.sidebar.caption("Engineering workspace")
    st.sidebar.subheader("Workspace")

    st.page_link("app.py", label="Home")
    st.page_link("pages/VDE_Setup.py", label="VDE Setup")
    st.page_link("pages/Roadload_Load_Case_Lab.py", label="Roadload & Load Case Lab")
    st.page_link("pages/Powertrain_Scenario.py", label="Powertrain Scenario")
    st.page_link("pages/Comparison_Report.py", label="Comparison Report")
    st.page_link("pages/Database_Management.py", label="Database Management")
    st.sidebar.subheader("Reference & Tools")
    st.page_link("pages/Methodology_Models.py", label="Methodology & Models")
    st.page_link("pages/Legacy_Engineering_Tools.py", label="Engineering Tools / Legacy")
    st.sidebar.divider()
    with st.sidebar.expander("Runtime database"):
        ctx["db_path"] = st.text_input("DB path", value=ctx.get("db_path", DB_DEFAULT))
        st.caption("Use a stable path under data/db for reproducibility.")
    st.sidebar.divider()
    st.sidebar.caption(f"EcoDrive Analyst · v{APP_VER}")


def main():
    ctx = _bootstrap_ctx()
    _sidebar(ctx)
    ctx["db_path"] = str(configure_db_path(ctx.get("db_path") or DB_DEFAULT))

    try:
        home.page_home()
    except Exception as e:
        st.error(f"Home page rendering failed: {e}")
        st.exception(e)

    st.markdown("---")
    st.caption(f"{APP_ICON} {APP_NAME} v{APP_VER} · Streamlit · Python {sys.version.split()[0]} · {platform.system()}")


if __name__ == "__main__":
    main()
