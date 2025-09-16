import streamlit as st
from .state import ensure_defaults

def sidebar_inputs():
    ensure_defaults(st.session_state)
    st.sidebar.header("Parameters (sidebar)")
    # Return values to be used in pages
    return st.session_state["roadload_params"]
