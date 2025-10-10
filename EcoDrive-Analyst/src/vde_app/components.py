import streamlit as st
from .state import ensure_defaults
from src.vde_core.services import  compute_vde_net_mj_per_km, epa_city_hwy_from_phase, wltp_phases_from_phase
import pandas as pd
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


# ---------- helpers (place near your other small utils) ----------
def vde_by_phase(df_cycle, leg, A, B, C, mass_kg):
    """Return dict {phase_label: MJ/km} if 'phase' column exists."""
    out = {}
    if not isinstance(df_cycle, pd.DataFrame) or "phase" not in df_cycle.columns:
        return out

    def _norm(p):
        p = str(p).strip().lower()
        if leg == "EPA":
            if "city" in p or "ftp" in p: return "city"
            if "hwy" in p or "hwfet" in p or "highway" in p: return "hwy"
        else:  # WLTP
            if "low" in p: return "low"
            if "mid" in p or "medium" in p: return "mid"
            if "high" in p and "extra" not in p: return "high"
            if "xhigh" in p or "extra" in p: return "xhigh"
        return p

    for ph in df_cycle["phase"].unique():
        sub = df_cycle[df_cycle["phase"] == ph]
        r = compute_vde_net_mj_per_km(sub, A, B, C, mass_kg)
        out[_norm(ph)] = float(r["MJ_km"]) if isinstance(r, dict) else float(r)
    return out


def show_vde_feedback(overall_mj_km, by_phase):
    """Render total and per-phase numbers immediately."""
    st.success(f"VDE_NET (cycle total) ≈ {overall_mj_km:.4f} MJ/km")
    if by_phase:
        pref = ["city","hwy","low","mid","high","xhigh"]
        ordered = [k for k in pref if k in by_phase] + [k for k in by_phase if k not in pref]
        cols = st.columns(min(4, len(ordered)))
        for i, k in enumerate(ordered):
            cols[i % len(cols)].metric(k.upper(), f"{by_phase[k]:.4f} MJ/km")

