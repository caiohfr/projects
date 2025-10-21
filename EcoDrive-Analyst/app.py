# app.py
import os, sys, platform
import streamlit as st
import pages.home_page as home

APP_NAME, APP_ICON, APP_VER = "EcoDrive Analyzer", "⚡", "0.7.2"
DB_DEFAULT = "data/db/eco_drive.db"

st.set_page_config(page_title=f"{APP_NAME} {APP_VER}", page_icon=APP_ICON, layout="wide")

# ---------- Session bootstrap ----------
def _bootstrap_ctx():
    # garante que 'ctx' exista SEMPRE que chamarmos
    if "ctx" not in st.session_state or not isinstance(st.session_state.ctx, dict):
        st.session_state.ctx = {"db_path": DB_DEFAULT}
    # defaults idempotentes
    st.session_state.ctx.setdefault("db_path", DB_DEFAULT)
    return st.session_state.ctx

def get_ctx():
    """Helper para outras páginas importarem, se quiser."""
    return _bootstrap_ctx()

# ---------- Quick checks ----------
def _quick_checks(db_path: str):
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("App", f"{APP_NAME}", delta=f"v{APP_VER}")
    with c2: st.metric("Python", sys.version.split()[0], delta=platform.system())
    with c3:
        try:
            st.metric("Database", "OK" if os.path.exists(db_path) else "Missing", delta=os.path.basename(db_path))
        except Exception as e:
            st.metric("Database", "Error", delta=str(e)[:18])
    with c4:
        try:
            import pandas as pd  # noqa
            st.metric("Core libs", "Loaded", delta="pandas ✓")
        except Exception:
            st.metric("Core libs", "Check", delta="pandas ?")

# ---------- Sidebar ----------
def _sidebar(ctx):
    st.sidebar.title(f"{APP_ICON} {APP_NAME}")
    st.sidebar.caption("Vehicle Demanded Energy – scientific analyzer")
    st.sidebar.subheader("Navigation")

    st.page_link("app.py",                     label="Home", icon="🏠")
    st.page_link("pages/vde_setup.py",         label="Vehicle Setup", icon="📥")
    st.page_link("pages/pwt_fuel_energy.py",   label="PWT Fuel & Energy", icon="⚙️")
    #st.page_link("pages/operating_points.py",  label="Operating Points / Report", icon="📊")

    st.sidebar.divider()
    # usa o ctx passado (já inicializado)
    ctx["db_path"] = st.sidebar.text_input("DB path", value=ctx.get("db_path", DB_DEFAULT))
    st.sidebar.caption("Tip: keep a stable path under /data/db for reproducibility.")
    st.sidebar.divider()
    st.sidebar.caption("© 2025 – EcoDrive Analyzer | CS50 project")

# ---------- Main ----------
def main():
    # 1) bootstrap ANTES de qualquer acesso ao ctx
    ctx = _bootstrap_ctx()

    # 2) agora é seguro montar a sidebar
    _sidebar(ctx)

    st.title(f"{APP_ICON} EcoDrive Analyzer")
    st.caption("Transparent, physics-based, and reproducible benchmarking")

    _quick_checks(ctx["db_path"])

    st.markdown("""
> **Start here:**  
> 1) **📥 Vehicle Setup** – load cycle and parameters.  
> 2) **⚙️ PWT Fuel & Energy** – compute VDE and deltas.  
> 3) **📊 Operating Points / Report (TO DO)** – visualize & export.
    """)

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
