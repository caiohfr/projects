# pages/Data_&_Setup.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.vde_app.state import ensure_defaults
from src.vde_app.plots import cycle_chart
from src.vde_core.utils import cycle_kpis
from src.vde_core.db import ensure_db, fetchall, fetchone, insert_vde, delete_row, update_vde
from src.vde_core.services import (
    default_cycle_for_legislation,
    load_cycle_csv,
    compute_vde_net_mj_per_km,  # retorna dict: {"MJ_km","Wh_km","km"}
)
import plotly.express as px

# ===========================
# Helpers de baseline via DB
# ===========================
def db_list_makes(legislation: str, category: str) -> list[str]:
    rows = fetchall("""
        SELECT DISTINCT make FROM vde_db
        WHERE legislation=? AND category=?
        ORDER BY make
    """, (legislation, category))
    return [r["make"] for r in rows]

def db_list_models(legislation: str, category: str, make: str) -> list[tuple[str,int]]:
    rows = fetchall("""
        SELECT id, model, year FROM vde_db
        WHERE legislation=? AND category=? AND make=?
        ORDER BY year DESC, model
    """, (legislation, category, make))
    out = []
    for r in rows:
        label = f'{make} {r["model"]} ({r["year"]}) [id={r["id"]}]'
        out.append((label, r["id"]))
    return out

def db_pick_vde_row(vde_id: int) -> dict | None:
    return fetchone("SELECT * FROM vde_db WHERE id=?", (vde_id,))

# ===========================
# Vehicle basics
# ===========================
def _vehicle_basics():
    st.subheader("Vehicle basics")
    col1, col2, col3 = st.columns(3)

    # marcas sugeridas (mantidas p/ consistência)
    default_makes = [
        "Toyota", "Honda", "Nissan", "Mitsubishi", "Mazda", "Subaru",
        "Hyundai", "Kia",
        "Volkswagen", "Audi", "BMW", "Mercedes-Benz", "Porsche", "Peugeot",
        "Renault", "Citroën", "Fiat", "Alfa Romeo", "Volvo", "Jaguar", "Land Rover",
        "Skoda", "Seat", "Opel",
        "Ford", "Chevrolet", "Dodge", "Chrysler", "Jeep", "Ram", "Cadillac",
        "Buick", "GMC", "Lincoln", "Tesla",
        "Suzuki", "Mini", "Smart", "Lexus", "Infiniti", "Acura",
        "Other (type manually)"
    ]

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
    category_list = epa_classes if legislation == "EPA" else wltp_classes
    category = col4.selectbox(
        "Category / Size class",
        category_list,
        index=category_list.index(st.session_state.get("vb_category", category_list[0]))
        if st.session_state.get("vb_category") in category_list else 0
    )

    # juntar marcas do DB + sugeridas (sem duplicar) + opção Other
    ensure_db()
    makes_db = db_list_makes(legislation, category)
    merged_makes = list(dict.fromkeys(makes_db + [m for m in default_makes if m not in makes_db]))
    if "Other (type manually)" not in merged_makes:
        merged_makes.append("Other (type manually)")

    make_choice = col1.selectbox(
        "Make/Brand",
        merged_makes,
        index=merged_makes.index(st.session_state.get("vb_make", merged_makes[0]))
        if st.session_state.get("vb_make") in merged_makes else 0
    )
    if make_choice == "Other (type manually)":
        make = st.text_input("Enter custom brand", value=st.session_state.get("vb_make_custom", ""))
        st.session_state["vb_make_custom"] = make
    else:
        make = make_choice

    notes = st.text_area("Notes / Proposal description", value=st.session_state.get("vb_notes", ""))

    # persist meta
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

# ===========================
# PWT mínimo (sem sidebar)
# ===========================
def _pwt_section():
    st.subheader("Powertrain (PWT)")
    st.caption("Minimal PWT inputs. Later this will list engines/transmissions from DB and link to Operating Points.")
    col1, col2, col3 = st.columns(3)

    engine_type = col1.selectbox("Engine type",
                                 ["Spark-ignition (Gasoline/Flex)", "Compression-ignition (Diesel)"],
                                 index=0)
    electrif = col2.selectbox("Electrification",
                              ["None", "MHEV (48V)", "HEV", "PHEV", "BEV"], index=0)
    trans = col3.selectbox("Transmission",
                           ["AT (auto)", "DCT", "CVT", "MT"], index=0)

    # guarda somente para página 2; Page 1 não grava eficiência
    st.session_state["pwt_engine"] = engine_type
    st.session_state["pwt_electrif"] = electrif
    st.session_state["pwt_trans"] = trans

# ===========================
# Semi-param (inputs úteis)
# ===========================
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

    st.session_state[f"{prefix}semi_params"] = {
        "pressure_kpa": kpa, "frac_front": frac_front, "mass": mass,
        "rho_air": rho_air, "Cx": Cx, "Af_m2": Af_m2,
        "A_par": A_par, "B_par": B_par, "C_par": C_par
    }

# ===========================
# Drive cycle (auto + upload)
# ===========================
def _use_standard_cycle():
    leg = st.session_state.get("vb_legislation", "EPA")
    fname = default_cycle_for_legislation(leg)
    try:
        df = load_cycle_csv(fname)
        st.session_state["cycle_df"] = df
        st.session_state["cycle_source"] = f"standard:{leg}"
        st.success(f"Using default **{leg}** cycle: `{fname}.csv`")
        k = cycle_kpis(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{k['duration_s']:.0f} s")
        c2.metric("Distance", f"{k['distance_km']:.2f} km")
        c3.metric("Avg Speed", f"{k['v_mean_kmh']:.1f} km/h")
        c4.metric("Samples", f"{k['n_points']}")
        return df
    except Exception as e:
        st.warning(f"Default cycle for **{leg}** not found. {e}")
        st.info("Please upload a custom cycle below.")
        return None

# ===========================
# Main
# ===========================
def main():
    st.title("📥 Data & Setup")
    ensure_defaults(st.session_state)
    ensure_db()

    # 1) Vehicle basics
    _vehicle_basics()

    # 2) Modo de cálculo
    st.subheader("VDE calculation mode")
    mode_options = {
        "Insert final coastdown (A/B/C + mass) from test": "FROM_TEST",
        "Use a vehicle as Baseline (DB) – or estimate from it": "BASELINE",
        "Define all parameters (no baseline)": "SEMI_PARAM"
    }
    mode_label = st.radio("Choose one:", list(mode_options.keys()), horizontal=False)
    st.session_state["mode"] = mode_options[mode_label]

    # 3) PWT mínimo
    _pwt_section()

    # 4) Blocos por modo
    if st.session_state["mode"] == "FROM_TEST":
        st.info("Enter coastdown results directly (as obtained from test).")
        colA, colB, colC, colM = st.columns(4)
        A = colA.number_input("A [N]", 0.0, 500.0, float(st.session_state.get("fromtest_A", 30.0)), 0.1)
        B = colB.number_input("B [N/kph]", 0.0, 5.0, float(st.session_state.get("fromtest_B", 0.80)), 0.01)
        C = colC.number_input("C [N/kph²]", 0.000, 0.100, float(st.session_state.get("fromtest_C", 0.011)), 0.001)
        mass = colM.number_input("Test mass [kg]", 600.0, 3500.0, float(st.session_state.get("fromtest_mass", 1500.0)), 5.0)
        st.session_state["abc"] = {"A": A, "B": B, "C": C}
        st.session_state["manual_mass"] = mass

    elif st.session_state["mode"] == "BASELINE":
        st.info("Pick a baseline vehicle already saved in the DB (same legislation & category).")
        leg = st.session_state["vehicle_meta"]["legislation"]
        cat = st.session_state["vehicle_meta"]["category"]

        makes_db = db_list_makes(leg, cat)
        if makes_db:
            make_sel = st.selectbox("Baseline: Make (from DB)", makes_db)
            models = db_list_models(leg, cat, make_sel)
            labels = ["--"] + [lbl for (lbl, _vid) in models]
            pick = st.selectbox("Pick baseline vehicle", labels)
            if pick != "--":
                pick_id = next((_vid for (lbl, _vid) in models if lbl == pick), None)
                row = db_pick_vde_row(pick_id) if pick_id else None
                if row:
                    st.success(f"Baseline loaded from DB: id={pick_id}")
                    A = float(row["coast_A_N"]) if row.get("coast_A_N") is not None else 30.0
                    B = float(row["coast_B_N_per_kph"]) if row.get("coast_B_N_per_kph") is not None else 0.8
                    C = float(row["coast_C_N_per_kph2"]) if row.get("coast_C_N_per_kph2") is not None else 0.02
                    mass = float(row["mass_kg"]) if row.get("mass_kg") is not None else 1500.0
                    st.session_state["abc"] = {"A": A, "B": B, "C": C}
                    st.session_state["manual_mass"] = mass
                    st.json({"A": A, "B": B, "C": C, "mass": mass})
                else:
                    st.warning("Could not load selected baseline from DB.")
        else:
            st.warning("No baselines found in DB for this legislation & category. Define parameters below.")
            _inputs_semi_param(prefix="nobase_")

    elif st.session_state["mode"] == "SEMI_PARAM":
        st.info("No baseline. Define everything to estimate A/B/C.")
        _inputs_semi_param(prefix="free_")

    st.markdown("---")

    # 5) Drive cycle
    st.subheader("Drive cycle")
    df = _use_standard_cycle()
    if df is not None:
        fig = cycle_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


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
                    fig = cycle_chart(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.markdown("---")

    # 6) Compute & Save
    st.subheader("Compute VDE and Save to DB")

    meta = st.session_state["vehicle_meta"]
    leg = meta["legislation"]; cat = meta["category"]
    make = meta["make"]; model = meta["desc"]; year = meta["year"]; notes = meta["notes"]
    cycle_name = "FTP-75" if leg == "EPA" else "WLTC Class 3"
    cycle_source = st.session_state.get("cycle_source", "standard:"+leg)
    abc = st.session_state.get("abc"); mass = st.session_state.get("manual_mass")

    if st.button("Compute VDE_NET and Save"):
        try:
            if st.session_state.get("cycle_df") is None:
                st.error("Cycle not loaded. Pick standard or upload a CSV.")
                return
            if not abc or mass is None:
                st.error("Missing A/B/C or mass. Provide in FROM_TEST or pick a DB baseline.")
                return

            A = float(abc["A"]); B = float(abc["B"]); C = float(abc["C"]); mass_kg = float(mass)
            r = compute_vde_net_mj_per_km(st.session_state["cycle_df"], A, B, C, mass_kg)
            vde_mjkm, wh_km, dist_km = r["MJ_km"], r["Wh_km"], r["km"]

            st.success(f"VDE_NET ≈ {vde_mjkm:.4f} MJ/km  ({wh_km:.1f} Wh/km)  | Distance: {dist_km:.2f} km")

            row = {
                "legislation": leg, "category": cat,
                "make": make, "model": model, "year": year, "notes": notes,
                "engine_type": st.session_state.get("pwt_engine"),
                "transmission_type": st.session_state.get("pwt_trans"),
                "mass_kg": mass_kg,
                "coast_A_N": A, "coast_B_N_per_kph": B, "coast_C_N_per_kph2": C,
                "cycle_name": cycle_name, "cycle_source": cycle_source,
                "vde_net_mj_per_km": vde_mjkm
                # campos de fase ficam None por enquanto
            }
            vde_id = insert_vde({k:v for k,v in row.items() if v is not None})
            st.session_state["vde_id"] = vde_id
            st.success(f"VDE snapshot saved to DB (id={vde_id}). Go to Page 2 to estimate fuel/CO₂.")

        except Exception as e:
            st.error(f"Failed to compute/save VDE: {e}")
        st.markdown("---")
    # 7) Edit / Delete existing VDE rows
    st.markdown("---")
    st.subheader("✏️ Edit / Delete an existing VDE row")

    rows = fetchall("""
        SELECT id, legislation, category, make, model, year,
            coast_A_N, coast_B_N_per_kph, coast_C_N_per_kph2, mass_kg,
            notes
        FROM vde_db
        ORDER BY id DESC
        LIMIT 100
    """)

    if not rows:
        st.info("No VDE rows saved yet.")
    else:
        labels = [
            f'#{r["id"]} — {r["legislation"]} | {r["category"]} | '
            f'{r["make"]} {r["model"]} ({r.get("year","")})'
            for r in rows
        ]
        idx = st.selectbox("Pick a VDE to edit/delete", list(range(len(labels))), format_func=lambda i: labels[i])

        sel = rows[idx]
        vde_id_edit = sel["id"]
        st.caption(f"Editing VDE id: {vde_id_edit}")

        # (Opcional) mostrar quantos cenários em fuelcons_db dependem deste VDE
        dep = fetchall("SELECT COUNT(*) AS n FROM fuelcons_db WHERE vde_id=?", (vde_id_edit,))
        st.caption(f'Linked scenarios in fuelcons_db: {dep[0]["n"] if dep else 0}')

        # --- Form de edição ---
        with st.form(key=f"edit_vde_{vde_id_edit}"):
            c1, c2, c3, c4 = st.columns(4)
            A_edit = c1.number_input("A [N]", 0.0, 500.0, float(sel["coast_A_N"] or 0.0), 0.1)
            B_edit = c2.number_input("B [N/kph]", 0.0, 5.0, float(sel["coast_B_N_per_kph"] or 0.0), 0.01)
            C_edit = c3.number_input("C [N/kph²]", 0.000, 0.100, float(sel["coast_C_N_per_kph2"] or 0.0), 0.001)
            M_edit = c4.number_input("Mass [kg]", 0.0, 4000.0, float(sel["mass_kg"] or 0.0), 1.0)

            c5, c6, c7 = st.columns(3)
            make_edit  = c5.text_input("Make",  value=sel["make"] or "")
            model_edit = c6.text_input("Model", value=sel["model"] or "")
            year_edit  = c7.number_input("Year", 1990, 2100, int(sel["year"] or 2020))

            notes_edit = st.text_area("Notes", value=sel["notes"] or "")

            save_btn = st.form_submit_button("💾 Save changes")
            if save_btn:
                try:
                    update_vde(vde_id_edit, {
                        "coast_A_N": A_edit,
                        "coast_B_N_per_kph": B_edit,
                        "coast_C_N_per_kph2": C_edit,
                        "mass_kg": M_edit,
                        "make": make_edit,
                        "model": model_edit,
                        "year": int(year_edit),
                        "notes": notes_edit,
                    })
                    st.success("Row updated.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to update: {e}")


        # 8) bloco de delete (com confirmação)
        with st.expander("🗑️ Delete this VDE row"):
            st.warning("This action is irreversible. Linked fuelcons_db rows will be deleted (ON DELETE CASCADE).")
            confirm_text = st.text_input("Type DELETE to confirm:")
            delete_disabled = (confirm_text != "DELETE")
            if st.button(f"Delete VDE id={vde_id_edit}", type="secondary", disabled=delete_disabled):
                try:
                    delete_row("vde_db", vde_id_edit)
                    st.success(f"VDE id={vde_id_edit} deleted.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")

        with st.expander("Session state (debug)"):
            keys = sorted(list(st.session_state.keys()))
            st.write({k: st.session_state.get(k, None) for k in keys})

if __name__ == "__main__":
    main()
