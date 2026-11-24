from __future__ import annotations

import streamlit as st

from src.vde_app.components.methodology_models import render_methodology_models


st.set_page_config(page_title="EcoDrive - Methodology & Models", layout="wide")


def main() -> None:
    render_methodology_models()


if __name__ == "__main__":
    main()
