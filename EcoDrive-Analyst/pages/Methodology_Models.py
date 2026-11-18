from __future__ import annotations

import streamlit as st

from pages.home_page import page_methodology


st.set_page_config(page_title="EcoDrive - Methodology & Models", layout="wide")


def main() -> None:
    page_methodology()


if __name__ == "__main__":
    main()
