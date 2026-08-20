# app.py
import os
import platform
import sys

import streamlit as st
import pages.home_page as home
from src.vde_core.db import configure_db_path

APP_NAME, APP_ICON, APP_VER = "EcoDrive Analyzer", "⚡", "0.7.2"
DB_DEFAULT = "data/db/eco_drive.db"

st.set_page_config(page_title=f"{APP_NAME} {APP_VER}", page_icon=APP_ICON, layout="wide")


def _bootstrap_ctx():
    if "ctx" not in st.session_state or not isinstance(st.session_state.ctx, dict):
        st.session_state.ctx = {"db_path": DB_DEFAULT}
    st.session_state.ctx.setdefault("db_path", DB_DEFAULT)
    return st.session_state.ctx


def get_ctx():
    return _bootstrap_ctx()


def _quick_checks(db_path: str):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("App", f"{APP_NAME}", delta=f"v{APP_VER}")
    with c2:
        st.metric("Python", sys.version.split()[0], delta=platform.system())
    with c3:
        try:
            st.metric("Database", "OK" if os.path.exists(db_path) else "Missing", delta=os.path.basename(db_path))
        except Exception as e:
            st.metric("Database", "Error", delta=str(e)[:18])
    with c4:
        try:
            import pandas as pd  # noqa: F401

            st.metric("Core libs", "Loaded", delta="pandas OK")
        except Exception:
            st.metric("Core libs", "Check", delta="pandas ?")


def _sidebar(ctx):
    st.sidebar.title(f"{APP_ICON} {APP_NAME}")
    st.sidebar.caption("Vehicle Demanded Energy - scientific analyzer")
    st.sidebar.subheader("Navigation")

    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/VDE_Setup.py", label="VDE Setup", icon="📥")
    st.page_link("pages/Powertrain_Scenario.py", label="Powertrain Scenario", icon="⚙️")
    st.page_link("pages/Comparison_Report.py", label="Powertrain Comparison", icon="📊")

    st.page_link("pages/Database_Management.py", label="Database Management")
    st.sidebar.divider()
    ctx["db_path"] = st.sidebar.text_input("DB path", value=ctx.get("db_path", DB_DEFAULT))
    st.sidebar.caption("Tip: keep a stable path under /data/db for reproducibility.")
    st.sidebar.divider()
    st.sidebar.caption("© 2025 - EcoDrive Analyzer | CS50 project")


def main():
    ctx = _bootstrap_ctx()
    _sidebar(ctx)
    ctx["db_path"] = str(configure_db_path(ctx.get("db_path") or DB_DEFAULT))

    st.title(f"{APP_ICON} EcoDrive Analyzer")
    st.caption("Transparent, physics-based, and reproducible benchmarking")

    _quick_checks(ctx["db_path"])

    st.markdown(
        """
> **Start here:**  
> 1) **📥 VDE Setup** - resolve a baseline, requested scenarios, and roadload demand.
> 2) **⚙️ Powertrain Scenario** - estimate fuel / energy / CO2 from resolved VDE.  
> 3) **📊 Powertrain Comparison** - compare saved scenarios, methods, and peer outlook.
        """
    )

    st.divider()
    try:
        home.page_home()
    except Exception as e:
        st.error(f"Home page rendering failed: {e}")
        st.exception(e)

    st.markdown("---")
    st.caption(f"{APP_ICON} {APP_NAME} v{APP_VER} · Streamlit · Python {sys.version.split()[0]} · {platform.system()}")


if __name__ == "__main__":
    main()
