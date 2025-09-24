import streamlit as st
from .state import ensure_defaults

def sidebar_inputs():
    ensure_defaults(st.session_state)
    st.sidebar.header("Parameters (sidebar)")
    # Return values to be used in pages
    return st.session_state["roadload_params"]


def pressure_input_with_units(key_prefix=""):
    unit = st.radio("Unit", ["kPa","psi"], key=f"{key_prefix}press_unit", horizontal=True)
    base_kpa = float(st.session_state.get(f"{key_prefix}press_kpa", 230.0))
    default_display = base_kpa if unit=="kPa" else base_kpa/6.89475729
    val = st.number_input(f"Pressure [{unit}]", 0.0, 500.0 if unit=="kPa" else 100.0, default_display, step=1.0 if unit=="kPa" else 0.5, key=f"{key_prefix}press_val")
    kpa = val if unit=="kPa" else val*6.89475729
    st.session_state[f"{key_prefix}press_kpa"] = kpa
    st.caption(f"{kpa:.1f} kPa ≈ {kpa/6.89475729:.1f} psi")
    return kpa
