import streamlit as st

import pages.home_page as home
from src.vde_app.visual_theme import inject_shell_styles, render_context_bar, render_sidebar_brand
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


def _navigation():
    return st.navigation(
        {
            "Engineering workspace": [
                st.Page(home.page_home, title="Home", icon=":material/home:", default=True),
                st.Page("pages/VDE_Setup.py", title="VDE Setup", icon=":material/directions_car:"),
                st.Page(
                    "pages/Roadload_Load_Case_Lab.py",
                    title="Roadload & Load Case Lab",
                    icon=":material/monitoring:",
                ),
                st.Page("pages/Powertrain_Scenario.py", title="Powertrain Scenario", icon=":material/hub:"),
                st.Page("pages/Comparison_Report.py", title="Comparison Report", icon=":material/analytics:"),
                st.Page("pages/Database_Management.py", title="Database Management", icon=":material/database:"),
            ],
            "Reference & tools": [
                st.Page("pages/Methodology_Models.py", title="Methodology & Models", icon=":material/menu_book:"),
                st.Page(
                    "pages/Legacy_Engineering_Tools.py",
                    title="Engineering Tools / Legacy",
                    icon=":material/build:",
                ),
            ],
        },
        position="sidebar",
        expanded=True,
    )


def main():
    ctx = _bootstrap_ctx()
    inject_shell_styles()
    render_sidebar_brand()
    page = _navigation()
    with st.sidebar.expander("Runtime database"):
        ctx["db_path"] = st.text_input("DB path", value=ctx.get("db_path", DB_DEFAULT))
        st.caption("Use a stable path under data/db for reproducibility.")
    st.sidebar.markdown(
        f'<div class="ed-shell-meta">EcoDrive Analyst &middot; v{APP_VER}</div>',
        unsafe_allow_html=True,
    )
    ctx["db_path"] = str(configure_db_path(ctx.get("db_path") or DB_DEFAULT))
    render_context_bar(page.title, ctx["db_path"])
    page.run()


if __name__ == "__main__":
    main()
