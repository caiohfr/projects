import streamlit as st 
import pandas as pd
from src.vde_app.state import ensure_defaults
from src.vde_core import loaders
from src.vde_core.utils import cycle_kpis
from src.vde_core.veh_db import load_vehicle_db, list_standards, list_size_classes, list_models, pick_vehicle_row

def main():
    st.title("📥 Data & Setup")
    ensure_defaults(st.session_state)

    # =========================
    # Scenario mode
    # =========================
    st.subheader("Scenario Mode")
    mode = st.radio("Select calculation mode", ["BASELINE", "SEMI_PARAM"], horizontal=True)
    st.session_state["mode"] = mode

    # =========================
    # Standard -> Size -> Baseline vehicle
    # =========================
    st.subheader("Baseline selection (optional)")
    try:
        vdf = load_vehicle_db()
        stds = list_standards(vdf)
        standard = st.selectbox("Standard", ["--"] + stds)
        selected_label = None

        if standard != "--":
            sizes = list_size_classes(vdf, standard)
            size = st.selectbox("Vehicle size/class", ["--"] + sizes)

            if size != "--":
                models_df = list_models(vdf, standard, size)
                labels = ["--"] + models_df["label"].tolist()
                selected_label = st.selectbox("Pick a baseline vehicle", labels)

                if selected_label and selected_label != "--":
                    row = pick_vehicle_row(vdf, selected_label)
                    if row:
                        st.success(f"Loaded baseline: {selected_label}")
                        # Write baseline into session_state for page 2
                        st.session_state["baseline"] = {
                            "standard": row.get("standard", standard),
                            "size_class": row.get("size_class", size),
                            "label": selected_label,
                            "A": float(row["A"]),
                            "B": float(row["B"]),
                            "C": float(row["C"]),
                            "mass": float(row["mass_kg"]),
                            "Cx": float(row["Cx"]) if pd.notnull(row.get("Cx", None)) else 0.30,
                            "Af_m2": float(row["Af_m2"]) if pd.notnull(row.get("Af_m2", None)) else 2.20,
                        }
                        st.json(st.session_state["baseline"])
    except Exception as e:
        st.info("Vehicle DB not found or invalid (create data/vehicles/vehicles.csv).")
        st.caption(str(e))

    # =========================
    # Vehicle parameters (legacy inputs kept)
    # =========================
    st.sidebar.header("Vehicle Parameters (legacy)")
    rp = st.session_state["roadload_params"]
    rp["f0"]   = st.sidebar.number_input("f0 [N]", 0.0, 200.0, rp["f0"])
    rp["f1"]   = st.sidebar.number_input("f1 [N·s/m]", 0.0, 5.0, rp["f1"])
    rp["f2"]   = st.sidebar.number_input("f2 [N·s²/m²]", 0.0, 1.0, rp["f2"], step=0.01)
    rp["mass"] = st.sidebar.number_input("Mass [kg]", 600.0, 3500.0, rp["mass"], step=5.0)

    st.sidebar.subheader("Fuel/Energy (estimation)")
    st.session_state["eta_pt"] = st.sidebar.slider("η_pt (ICE)", 0.15, 0.35, float(st.session_state["eta_pt"]), 0.01)
    st.session_state["lhv"] = st.sidebar.number_input("LHV [MJ/L]", 28.0, 36.0, float(st.session_state["lhv"]), 0.1)

    # =========================
    # Semi-parametric inputs (only shown if chosen)
    # =========================
    if mode == "SEMI_PARAM":
        st.markdown("**Semi-parametric inputs**")
        col1, col2, col3 = st.columns(3)
        st.session_state["pressure_kpa"] = col1.number_input("Tire pressure [kPa]", 150.0, 350.0, 230.0, 1.0)
        st.session_state["frac_front"]   = col2.slider("Front axle load fraction", 0.2, 0.8, 0.5, 0.01)
        st.session_state["Cx"]           = col3.number_input("Cx [-]", 0.10, 0.60, 0.30, 0.01)

        col4, col5 = st.columns(2)
        st.session_state["Af_m2"]  = col4.number_input("Frontal area [m²]", 1.5, 3.5, 2.2, 0.05)
        st.session_state["rho_air"] = col5.number_input("Air density ρ [kg/m³]", 1.0, 1.4, 1.20, 0.01)

        st.markdown("**Parasitic terms (optional)**")
        colA, colB, colC = st.columns(3)
        st.session_state["A_par"] = colA.number_input("Parasitic A [N]", 0.0, 50.0, 0.0, 0.1)
        st.session_state["B_par"] = colB.number_input("Parasitic B [N/kph]", 0.0, 1.0, 0.0, 0.01)
        st.session_state["C_par"] = colC.number_input("Parasitic C [N/kph²]", 0.0, 0.2, 0.0, 0.001)

    st.markdown("---")

    # =========================
    # Cycles
    # =========================
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
        keys = ["mode","baseline","pressure_kpa","frac_front","Cx","Af_m2","rho_air",
                "A_par","B_par","C_par",
                "roadload_params","eta_pt","lhv","cycle_df"]
        st.write({k: st.session_state.get(k, None) for k in keys})

if __name__ == "__main__":
    main()
