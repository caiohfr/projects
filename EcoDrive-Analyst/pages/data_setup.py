import streamlit as st
import pandas as pd
from src.vde_app.state import ensure_defaults
from src.vde_core import loaders
from src.vde_core.utils import cycle_kpis

def main():
    st.title("📥 Data & Setup")
    ensure_defaults(st.session_state)

    st.sidebar.header("Vehicle Parameters")
    rp = st.session_state["roadload_params"]
    rp["f0"]   = st.sidebar.number_input("f0 [N]", 0.0, 200.0, rp["f0"])
    rp["f1"]   = st.sidebar.number_input("f1 [N·s/m]", 0.0, 5.0, rp["f1"])
    rp["f2"]   = st.sidebar.number_input("f2 [N·s²/m²]", 0.0, 1.0, rp["f2"], step=0.01)
    rp["mass"] = st.sidebar.number_input("Mass [kg]", 600.0, 3500.0, rp["mass"], step=5.0)

    st.sidebar.subheader("Fuel/Energy (estimation)")
    st.session_state["eta_pt"] = st.sidebar.slider("η_pt (ICE)", 0.15, 0.35, float(st.session_state["eta_pt"]), 0.01)
    st.session_state["lhv"] = st.sidebar.number_input("LHV [MJ/L]", 28.0, 36.0, float(st.session_state["lhv"]), 0.1)

    st.subheader("Choose a built-in cycle")
    available = loaders.list_cycles()
    choice = st.selectbox("Cycles in data/cycles/", ["--"] + available)

    if choice != "--":
        df = loaders.load_cycle(choice)
        st.session_state["cycle_df"] = df
        st.success(f"Cycle '{choice}' loaded from data/cycles/.")
        st.dataframe(df.head())

        k = cycle_kpis(df)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Duration", f"{k['duration_s']:.0f} s")
        c2.metric("Distance", f"{k['distance_km']:.2f} km")
        c3.metric("Avg Speed", f"{k['v_mean_kmh']:.1f} km/h")
        c4.metric("Samples", f"{k['n_points']}")

    st.markdown("---")
    st.subheader("Or upload a CSV (columns: t, v)")
    upl = st.file_uploader("Upload CSV", type=["csv"])
    if upl:
        try:
            df = pd.read_csv(upl)
            if not {"t","v"} <= set(df.columns):
                st.error("CSV must have columns: t, v")
            else:
                st.session_state["cycle_df"] = df
                st.success("Cycle loaded from upload.")
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    with st.expander("Session state (debug)"):
        keys = ["roadload_params","eta_pt","lhv","cycle_df"]
        st.write({k: st.session_state.get(k, None) for k in keys})

if __name__ == "__main__":
    main()
