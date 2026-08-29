from __future__ import annotations

import streamlit as st

from src.vde_app.components.legacy_engineering_tools import render_legacy_engineering_tools
from src.vde_core.db import ensure_db


st.set_page_config(page_title="EcoDrive - Legacy & Engineering Tools", layout="wide")
ensure_db()


def main() -> None:
    render_legacy_engineering_tools()


if __name__ == "__main__":
    main()
