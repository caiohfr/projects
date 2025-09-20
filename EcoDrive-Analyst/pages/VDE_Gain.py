import streamlit as st
from src.vde_app.state import ensure_defaults
from src.vde_core.physics import compute_vde_net
from src.vde_core.conversion import estimate_fuel_from_vde

def main():
    st.title("⚙️ VDE & Gain")
    ensure_defaults(st.session_state)

    df = st.session_state.get("cycle_df")
    rp = st.session_state.get("roadload_params")
    eta, lhv = st.session_state.get("eta_pt", 0.24), st.session_state.get("lhv", 32.0)

    if df is None:
        st.warning("Load a cycle in **📥 Data & Setup** first.")
        st.stop()

    vde = compute_vde_net(df, rp["f0"], rp["f1"], rp["f2"], rp["mass"])
    fuel = estimate_fuel_from_vde(vde, eta, lhv)

    col1, col2 = st.columns(2)
    col1.metric("VDE_NET [kWh/100km]", f"{vde:.3f}")
    col2.metric("Fuel (est.) [L/100km]", f"{fuel:.3f}")

    with st.expander("Debug"):
        st.write("Params:", rp)
        st.write("η_pt, LHV:", eta, lhv)
        st.write(df.head())

if __name__ == "__main__":
    main()
