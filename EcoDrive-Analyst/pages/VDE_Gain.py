import streamlit as st
from src.vde_app.state import ensure_defaults
from src.vde_core.models import Scenario, RoadLoadABC, InertiaSpec, VehicleAero, Parasitic
from src.vde_core.services import compute_vde_from_scenario

def main():
    st.title("⚙️ VDE & Gain — Scenario")
    ensure_defaults(st.session_state)

    df = st.session_state.get("cycle_df")
    if df is None:
        st.warning("Load a cycle in **Data & Setup**.")
        st.stop()

    # Build Scenario
    mode = st.session_state.get("mode", "BASELINE")
    baseline = st.session_state.get("baseline", None)
    rp = st.session_state.get("roadload_params", {"mass": 1500.0})
    mass = baseline["mass"] if (baseline and "mass" in baseline) else rp.get("mass", 1500.0)

    if mode == "BASELINE" and baseline:
        A,B,C = baseline["A"], baseline["B"], baseline["C"]
        scn = Scenario(
            mode="BASELINE",
            roadload=RoadLoadABC(A,B,C),
            inertia=InertiaSpec(mass, "User")
        )
        st.info(f"Using baseline A/B/C from: {baseline.get('label','(custom)')}")
    elif mode == "SEMI_PARAM":
        rho = st.session_state.get("rho_air", 1.20)
        Cx  = st.session_state.get("Cx", 0.30)
        Af  = st.session_state.get("Af_m2", 2.2)
        scn = Scenario(
            mode="SEMI_PARAM",
            pressure_kpa=st.session_state.get("pressure_kpa", 230.0),
            frac_front=st.session_state.get("frac_front", 0.5),
            aero=VehicleAero(rho=rho, Cx=Cx, Af_m2=Af),
            parasitic=Parasitic(
                A_par=st.session_state.get("A_par", 0.0),
                B_par=st.session_state.get("B_par", 0.0),
                C_par=st.session_state.get("C_par", 0.0),
            ),
            inertia=InertiaSpec(mass, "User"),
        )
        st.info("Using semi-parametric A/B/C composition.")
    else:
        # fallback: BASELINE with manual A/B/C if the user didn’t select a baseline vehicle
        manual = st.session_state.get("abc", {"A":30.0,"B":0.8,"C":0.12})
        scn = Scenario(
            mode="BASELINE",
            roadload=RoadLoadABC(manual["A"], manual["B"], manual["C"]),
            inertia=InertiaSpec(mass, "User")
        )
        st.warning("No baseline vehicle selected. Using manual A/B/C values.")

    # Run computation
    res = compute_vde_from_scenario(df, scn)
    c1,c2,c3 = st.columns(3)
    c1.metric("A/B/C [N, N/kph, N/kph²]", f"{res['abc'].A:.2f} / {res['abc'].B:.3f} / {res['abc'].C:.4f}")
    c2.metric("Distance [km]", f"{res['dist_km']:.2f}")
    c3.metric("VDE_NET [Wh/km]", f"{res['VDE_Wh_per_km']:.1f}")

    with st.expander("Details"):
        st.write("Scenario:", vars(scn.inertia))
        st.write("ABC:", vars(res["abc"]))

if __name__ == "__main__":
    main()
