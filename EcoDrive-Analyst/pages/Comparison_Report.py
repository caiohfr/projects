from __future__ import annotations

import streamlit as st

from src.vde_app.components.comparison_report import render_comparison_report
from src.vde_core.db import ensure_db


st.set_page_config(page_title="EcoDrive - Comparison Report", layout="wide")


def main() -> None:
    ensure_db()
    render_comparison_report()


if __name__ == "__main__":
    main()
