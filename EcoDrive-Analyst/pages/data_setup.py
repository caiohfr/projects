import streamlit as st
import pandas as pd
from pathlib import Path

from src.vde_app.state import ensure_defaults
from src.vde_core.utils import cycle_kpis
from src.vde_core.veh_db import (
    load_vehicle_db, list_standards, list_size_classes, list_models, pick_vehicle_row
)

# --- Vehicle basics ------------------------------------------------------------
def _vehicle_basics():
    st.subheader("Vehicle basics")
    col1, col2, col3 = st.columns(3)
    # lista expandida de marcas
    default_makes = [
        "Toyota", "Honda", "Nissan", "Mitsubishi", "Mazda", "Subaru",
        "Hyundai", "Kia",
        "Volkswagen", "Audi", "BMW", "Mercedes-Benz", "Porsche", "Peugeot", 
        "Renault", "Citroën", "Fiat", "Alfa Romeo", "Volvo", "Jaguar", "Land Rover",
        "Skoda", "Seat", "Opel",
        "Ford", "Chevrolet", "Dodge", "Chrysler", "Jeep", "Ram", "Cadillac", 
        "Buick", "GMC", "Lincoln", "Tesla",
        "Suzuki", "Mini", "Smart", "Lexus", "Infinity", "Acura",
        "Other (type manually)"
    ]

    make_choice = col1.selectbox(
        "Make/Brand",
        default_makes,
        index=default_makes.index(st.session_state.get("vb_make", "Toyota"))
        if st.session_state.get("vb_make") in default_makes else len(default_makes) - 1
    )

    if make_choice == "Other (type manually)":
        make = st.text_input(
            "Enter custom brand",
            value=st.session_state.get("vb_make_custom", "")
        )
        st.session_state["vb_make_custom"] = make
    else:
        make = make_choice

    desc = col2.text_input("Model / Description", value=st.session_state.get("vb_desc", ""))
    year = col3.number_input("Year", 1990, 2100, int(st.session_state.get("vb_year", 2020)))

    col4, col5 = st.columns(2)
    legislation = col5.selectbox(
        "Legislation / Standard",
        ["EPA", "WLTP"],
        index=0 if st.session_state.get("vb_legislation", "EPA") == "EPA" else 1
    )

    # categorias oficiais
    epa_classes = [
        "Two Seaters", "Minicompact Cars", "Subcompact Cars", "Compact Cars",
        "Midsize Cars", "Large Cars",
        "Small Station Wagons", "Midsize Station Wagons",
        "Small SUVs", "Standard SUVs",
        "Minivans", "Vans",
        "Small Pickup Trucks", "Standard Pickup Trucks"
    ]
    wltp_classes = [
        "Class 1 (<850 kg)", "Class 2 (850–1220 kg)", "Class 3 (>1220 kg)"
    ]

    if legislation == "EPA":
        category_list = epa_classes
    else:
        category_list = wltp_classes

    category = col4.selectbox(
        "Category / Size class",
        category_list,
        index=category_list.index(st.session_state.get("vb_category", category_list[0]))
        if st.session_state.get("vb_category") in category_list else 0
    )

    notes = st.text_area("Notes / Proposal description", value=st.session_state.get("vb_notes", ""))

    st.session_state["vehicle_meta"] = {
        "make": make, "desc": desc, "year": int(year),
        "category": category, "legislation": legislation, "notes": notes
    }
    st.session_state["vb_make"] = make
    st.session_state["vb_desc"] = desc
    st.session_state["vb_year"] = int(year)
    st.session_state["vb_category"] = category
    st.session_state["vb_legislation"] = legislation
    st.session_state["vb_notes"] = notes


# --- Tire pressure w/ units ----------------------------------------------------
def _pressure_input_with_units(prefix=""):
    st.markdown("**Tire pressure (with unit selector)**")
    ucol1, ucol2 = st.columns([1, 2])
    unit = ucol1.radio(
        "Unit", ["kPa", "psi"],
        horizontal=True,
        index=0 if st.session_state.get(f"{prefix}pressure_unit", "kPa") == "kPa" else 1,
    )
    st.session_state[f"{prefix}pressure_unit"] = unit

    default_kpa = float(st.session_state.get(f"{prefix}pressure_kpa", 230.0))
    default_display = default_kpa if unit == "kPa" else (default_kpa / 6.89475729)

    val = ucol2.number_input(
        f"Pressure [{unit}]",
        0.0, 500.0 if unit == "kPa" else 100.0,
        float(default_display),
        step=1.0 if unit == "kPa" else 0.5
    )

    kpa = val if unit == "kPa" else (val * 6.89475729)
    st.session_state[f"{prefix}pressure_kpa"] = kpa
    st.caption(f"{kpa:.1f} kPa ≈ {kpa/6.89475729:.1f} psi")
    return kpa

# --- Semi-parametric inputs ----------------------------------------------------
def _inputs_semi_param(prefix=""):
    st.markdown("**Semi-parametric inputs (to build A/B/C)**")
    kpa = _pressure_input_with_units(prefix=prefix)

    c1, c2, c3 = st.columns(3)
    frac_front = c2.slider("Front axle load fraction", 0.2, 0.8,
                           float(st.session_state.get(f"{prefix}frac_front", 0.5)), 0.01)
    mass = c3.number_input("Test mass [kg]", 600.0, 3500.0,
                           float(st.session_state.get(f"{prefix}mass", 1500.0)), 5.0)

    c4, c5, c6 = st.columns(3)
    rho_air = c4.number_input("Air density ρ [kg/m³]", 1.0, 1.4,
                              float(st.session_state.get(f"{prefix}rho_air", 1.20)), 0.01)
    Cx = c5.number_input("Cx [-]", 0.10, 0.60,
                         float(st.session_state.get(f"{prefix}Cx", 0.30)), 0.01)
    Af_m2 = c6.number_input("Frontal area [m²]", 1.5, 3.5,
                            float(st.session_state.get(f"{prefix}Af_m2", 2.2)), 0.05)

    st.markdown("**Parasitic terms (optional)**")
    cA, cB, cC = st.columns(3)
    A_par = cA.number_input("Parasitic A [N]", 0.0, 50.0,
                            float(st.session_state.get(f"{prefix}A_par", 0.0)), 0.1)
    B_par = cB.number_input("Parasitic B [N/kph]", 0.0, 1.0,
                            float(st.session_state.get(f"{prefix}B_par", 0.0)), 0.01)
    C_par = cC.number_input("Parasitic C [N/kph²]", 0.0, 0.2,
                            float(st.session_state.get(f"{prefix}C_par", 0.0)), 0.001)

    semip = {"pressure_kpa": kpa, "frac_front": frac_front, "mass": mass,
             "rho_air": rho_air, "Cx": Cx, "Af_m2": Af_m2,
             "A_par": A_par, "B_par": B_par, "C_par": C_par}
    st.session_state[f"{prefix}semi_params"] = semip
    return semip

# --- Powertrain section --------------------------------------------------------
def _pwt_section():
    st.subheader("Powertrain (PWT)")
    st.caption("Select minimal PWT data. Later this will link to Operating Points.")
    col1, col2, col3 = st.columns(3)

    engine_type = col1.selectbox("Engine type",
                                 ["Spark-ignition (Gasoline/Flex)", "Compression-ignition (Diesel)"],
                                 index=0)
    electrif = col2.selectbox("Electrification",
                              ["None", "MHEV (48V)", "HEV", "PHEV", "BEV"], index=0)
    trans = col3.selectbox("Transmission",
                           ["AT (auto)", "DCT", "CVT", "MT"], index=0)

    st.session_state["pwt_engine"] = engine_type
    st.session_state["pwt_electrif"] = electrif
    st.session_state["pwt_trans"] = trans

    base_eta = 0.24
    electrif_mult = {"None": 1.00, "MHEV (48V)": 1.05, "HEV": 1.25,
                     "PHEV": 1.30, "BEV": 3.60}[electrif]
    trans_mult = {"AT (auto)": 1.00, "DCT": 1.03, "CVT": 1.02, "MT": 1.00}[trans]

    eta_pt_est = base_eta * electrif_mult * trans_mult
    st.session_state["eta_pt_est"] = eta_pt_est
    st.session_state["bev_eff_drive"] = 0.88

    st.caption(f"Estimated η_pt ≈ {eta_pt_est:.3f} (ICE proxy). "
               "BEV handled separately. [Operating Points (future)](#)")

# --- Drive cycle helpers -------------------------------------------------------
def _default_cycle_filename(legislation: str) -> str:
    mapping = {"EPA": "ftp75", "WLTP": "wltc_class3"}
    return mapping.get(legislation, "ftp75")

def _load_cycle_csv(filename_no_ext: str) -> pd.DataFrame:
    path = Path("data/cycles") / f"{filename_no_ext}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cycle CSV not found: {path}")
    df = pd.read_csv(path)
    if not {"t", "v"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: t, v (v in m/s)")
    return df

def _use_standard_cycle():
    leg = st.session_state.get("vb_legislation", "EPA")
    fname = _default_cycle_filename(leg)
    try:
        df = _load_cycle_csv(fname)
        st.session_state["cycle_df"] = df
        st.session_state["cycle_source"] = f"standard:{leg}"
        st.success(f"Using default **{leg}** cycle: `{fname}.csv`")
        k = cycle_kpis(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{k['duration_s']:.0f} s")
        c2.metric("Distance", f"{k['distance_km']:.2f} km")
        c3.metric("Avg Speed", f"{k['v_mean_kmh']:.1f} km/h")
        c4.metric("Samples", f"{k['n_points']}")
    except Exception as e:
        st.warning(f"Default cycle for **{leg}** not found. {e}")
        st.info("Please upload a custom cycle below.")

# --- Main page -----------------------------------------------------------------
def main():
    st.title("📥 Data & Setup")
    ensure_defaults(st.session_state)

    _vehicle_basics()

    st.subheader("VDE calculation mode")
    mode_options = {
        "Insert final coastdown (A/B/C + mass) from test": "FROM_TEST",
        "Use a vehicle as Baseline (DB) – or estimate from it": "BASELINE",
        "Define all parameters (no baseline)": "SEMI_PARAM"
    }
    mode_label = st.radio("Choose one:", list(mode_options.keys()), horizontal=False)
    st.session_state["mode"] = mode_options[mode_label]

    _pwt_section()

    # Mode blocks
    if st.session_state["mode"] == "FROM_TEST":
        st.info("Enter coastdown results directly (as obtained from test).")
        colA, colB, colC, colM = st.columns(4)
        A = colA.number_input("A [N]", 0.0, 500.0, float(st.session_state.get("fromtest_A", 30.0)), 0.1)
        B = colB.number_input("B [N/kph]", 0.0, 5.0, float(st.session_state.get("fromtest_B", 0.80)), 0.01)
        C = colC.number_input("C [N/kph²]", 0.0, 1.0, float(st.session_state.get("fromtest_C", 0.12)), 0.001)
        mass = colM.number_input("Test mass [kg]", 600.0, 3500.0, float(st.session_state.get("fromtest_mass", 1500.0)), 5.0)
        st.session_state["from_test"] = {"A": A, "B": B, "C": C, "mass": mass}
        st.session_state["abc"] = {"A": A, "B": B, "C": C}
        st.session_state["manual_mass"] = mass

    elif st.session_state["mode"] == "BASELINE":
        st.info("Pick a baseline vehicle from DB (or opt-out).")
        try:
            vdf = load_vehicle_db()
            stds = list_standards(vdf)
            standard = st.selectbox("DB Standard", ["--"] + stds)
            use_db_baseline = st.checkbox("Use a vehicle from DB", value=True if standard != "--" else False)

            if use_db_baseline and standard != "--":
                sizes = list_size_classes(vdf, standard)
                size = st.selectbox("DB Vehicle size/class", ["--"] + sizes)
                if size != "--":
                    models_df = list_models(vdf, standard, size)
                    labels = ["--"] + models_df["label"].tolist()
                    selected_label = st.selectbox("Pick a baseline vehicle", labels)
                    if selected_label and selected_label != "--":
                        row = pick_vehicle_row(vdf, selected_label)
                        if row:
                            st.success(f"Baseline loaded: {selected_label}")
                            st.session_state["baseline"] = {
                                "standard": row.get("standard", standard),
                                "size_class": row.get("size_class", size),
                                "label": selected_label,
                                "A": float(row["A"]), "B": float(row["B"]), "C": float(row["C"]),
                                "mass": float(row["mass_kg"]),
                                "Cx": float(row["Cx"]) if pd.notnull(row.get("Cx", None)) else 0.30,
                                "Af_m2": float(row["Af_m2"]) if pd.notnull(row.get("Af_m2", None)) else 2.20,
                            }
                            use_semip = st.checkbox("Estimate coastdown from semi-parametric inputs", value=False)
                            st.session_state["use_semiparam_on_baseline"] = use_semip
                            if use_semip:
                                _inputs_semi_param(prefix="baseline_")
                            else:
                                st.caption("Using DB A/B/C directly.")
            else:
                st.warning("No DB baseline selected. Define parameters to estimate coastdown:")
                _inputs_semi_param(prefix="nobase_")

        except Exception as e:
            st.info("Vehicle DB not found or invalid.")
            st.caption(str(e))
            _inputs_semi_param(prefix="nobase_")

    elif st.session_state["mode"] == "SEMI_PARAM":
        st.info("No baseline. Define everything to estimate A/B/C.")
        _inputs_semi_param(prefix="free_")

    st.markdown("---")

    # Drive cycle (auto by legislation, optional upload)
    st.subheader("Drive cycle")
    _use_standard_cycle()
    with st.expander("Upload a different cycle (optional)"):
        upl = st.file_uploader("Upload custom cycle CSV", type=["csv"])
        if upl:
            try:
                df = pd.read_csv(upl)
                if not {"t", "v"}.issubset(df.columns):
                    st.error("CSV must have columns: t, v (v in m/s)")
                else:
                    st.session_state["cycle_df"] = df
                    st.session_state["cycle_source"] = "custom:upload"
                    st.success("Custom cycle loaded.")
                    st.dataframe(df.head())
                    k = cycle_kpis(df)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Duration", f"{k['duration_s']:.0f} s")
                    c2.metric("Distance", f"{k['distance_km']:.2f} km")
                    c3.metric("Avg Speed", f"{k['v_mean_kmh']:.1f} km/h")
                    c4.metric("Samples", f"{k['n_points']}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with st.expander("Session state (debug)"):
        keys = sorted(list(st.session_state.keys()))
        st.write({k: st.session_state.get(k, None) for k in keys})

if __name__ == "__main__":
    main()
