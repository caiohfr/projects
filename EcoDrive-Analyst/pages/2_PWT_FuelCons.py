# pages/VDE_Gain.py  (PWT & Fuel/Energy → grava fuelcons_db)
import streamlit as st
import pandas as pd

from src.vde_core.db import ensure_db, fetchall, fetchone, insert_fuelcons, update_vde
from src.vde_core.utils import epa_combined_eff_kmpl, epa_combined_cons_l100  # se quiser usar depois

# ---------------------------
# Pequenas tabelas de apoio
# ---------------------------
FUEL_DEFAULTS = {
    # LHV (MJ/L), CO2 g/L (aprox. típico; ajuste conforme necessidade/região)
    "Gasoline": {"LHV": 32.0, "gCO2_per_L": 2340.0},
    "Ethanol":  {"LHV": 21.1, "gCO2_per_L": 1500.0},   # use valor de política do seu país se quiser
    "Diesel":   {"LHV": 36.0, "gCO2_per_L": 2640.0},
    "Flex":     {"LHV": 28.0, "gCO2_per_L": 2200.0},   # média simples (exemplo)
    "CNG":      {"LHV": 21.0, "gCO2_per_L": 2000.0},   # placeholder; CNG costuma ser por kg/Nm³
}

GRID_DEFAULT_gCO2_per_kWh = 0.0  # ajuste se quiser contabilizar CO2 da rede (BEV)


# ---------------------------
# Helpers para escolher VDE
# ---------------------------
def list_vde_snapshots():
    return fetchall("""
        SELECT id, legislation, category, make, model, year, vde_net_mj_per_km
        FROM vde_db
        ORDER BY id DESC
        LIMIT 200
    """)

def pick_vde_from_db():
    rows = list_vde_snapshots()
    if not rows:
        st.warning("No VDE snapshots found. Go back to Page 1 and save one.")
        return None
    labels = [f'#{r["id"]} — {r["legislation"]} | {r["category"]} | '
              f'{r["make"]} {r["model"]} ({r.get("year","")})'
              for r in rows]
    idx = st.selectbox("Pick a VDE snapshot", list(range(len(labels))), format_func=lambda i: labels[i])
    return rows[idx]["id"]


# ---------------------------
# Cálculos principais
# ---------------------------
def whkm_from_mjkm(mj_per_km: float) -> float:
    return mj_per_km / 0.0036

def compute_ice_fuel_from_vde(vde_net_mj_per_km: float,
                              eta_pt_est: float,
                              LHV_MJ_per_L: float,
                              gCO2_per_L: float) -> dict:
    """Retorna: L/100km, km/L, gCO2/km, energy_Wh/km"""
    wh_per_km = whkm_from_mjkm(vde_net_mj_per_km)
    wheel_MJ_per_km = vde_net_mj_per_km
    tank_MJ_per_km = wheel_MJ_per_km / max(eta_pt_est, 1e-6)
    L_per_km = tank_MJ_per_km / max(LHV_MJ_per_L, 1e-6)

    L100 = 100.0 * L_per_km
    kmL  = 100.0 / L100 if L100 > 0 else None
    gCO2 = gCO2_per_L * L_per_km
    return {
        "energy_Wh_per_km": wh_per_km,
        "fuel_l_per_100km": L100,
        "fuel_km_per_l": kmL,
        "gco2_per_km": gCO2
    }

def compute_bev_from_vde(vde_net_mj_per_km: float,
                         driveline_eff: float,
                         grid_gCO2_per_kWh: float) -> dict:
    """BEV: ajusta por eficiência de tração (se quiser aplicar) e CO2 da rede."""
    wheel_Wh_per_km = whkm_from_mjkm(vde_net_mj_per_km)
    batt_Wh_per_km = wheel_Wh_per_km / max(driveline_eff, 1e-6)
    gCO2_per_km = (batt_Wh_per_km / 1000.0) * grid_gCO2_per_kWh
    return {
        "energy_Wh_per_km": batt_Wh_per_km,
        "fuel_l_per_100km": None,
        "fuel_km_per_l": None,
        "gco2_per_km": gCO2_per_km
    }


# ---------------------------
# UI Principal
# ---------------------------
def main():
    st.title("⚙️ PWT & Fuel/Energy (Page 2)")
    ensure_db()

    # 1) Escolher VDE snapshot (pega do session_state ou escolha manual)
    vde_id = st.session_state.get("vde_id")
    if not vde_id:
        st.info("No vde_id in session. Pick one from DB:")
        vde_id = pick_vde_from_db()
        if vde_id:
            st.session_state["vde_id"] = vde_id
        else:
            return

    vde = fetchone("SELECT * FROM vde_db WHERE id=?", (vde_id,))
    if not vde:
        st.error("VDE not found.")
        return

    st.success(f'Using VDE #{vde_id} — {vde["legislation"]} | {vde["category"]} | '
               f'{vde["make"]} {vde["model"]} ({vde.get("year","")})')
    st.metric("VDE_NET", f'{vde["vde_net_mj_per_km"]:.4f} MJ/km')

    st.markdown("---")

    # 2) PWT details → opcionalmente salvar no VDE_DB
    st.subheader("PWT Details (save to VDE_DB)")
    c1, c2 = st.columns(2)
    engine_model = c1.text_input("Engine model", value=vde.get("engine_model") or "")
    engine_size_l = c2.number_input("Engine size [L]", 0.0, 10.0, float(vde.get("engine_size_l") or 0.0), 0.1)

    c3, c4 = st.columns(2)
    engine_aspiration = c3.selectbox(
        "Aspiration", ["NA", "Turbo", "Supercharged"],
        index= ["NA","Turbo","Supercharged"].index(vde.get("engine_aspiration") or "NA")
    )
    transmission_model = c4.text_input("Transmission model", value=vde.get("transmission_model") or "")

    if st.button("💾 Save PWT details to VDE_DB"):
        try:
            update_vde(vde_id, {
                "engine_model": engine_model or None,
                "engine_size_l": engine_size_l if engine_size_l > 0 else None,
                "engine_aspiration": engine_aspiration or None,
                "transmission_model": transmission_model or None
            })
            st.success("PWT details saved to VDE_DB.")
        except Exception as e:
            st.error(f"Failed to save PWT details: {e}")

    st.markdown("---")

    # 3) Eficiência & cenário de consumo/energia → salvar em FUELCONS_DB
    st.subheader("Efficiency & Fuel/Energy scenario (save to FUELCONS_DB)")

    # inputs principais
    colA, colB, colC = st.columns(3)
    electrification = colA.selectbox(
        "Electrification", ["None","MHEV","HEV","PHEV","BEV"],
        index=["None","MHEV","HEV","PHEV","BEV"].index(
            (st.session_state.get("pwt_electrif") or "None").replace(" (48V)", "").replace(" (auto)", "")
        )
        if st.session_state.get("pwt_electrif") else 0
    )
    fuel_type = colB.selectbox("Fuel type", list(FUEL_DEFAULTS.keys()), index=0)

    # default η_pt ou eficiência BEV
    if electrification != "BEV":
        eta_default = float(st.session_state.get("eta_pt_est") or 0.24)
        eta_pt_est = colC.slider("η_pt (ICE path)", 0.05, 0.45, eta_default, 0.005)
    else:
        bev_eff_drive = colC.slider("Driveline efficiency (BEV)", 0.60, 0.98, float(st.session_state.get("bev_eff_drive") or 0.88), 0.01)

    colD, colE, colF = st.columns(3)

    # Combustível: LHV e gCO2/L (com override)
    LHV_default = FUEL_DEFAULTS[fuel_type]["LHV"]
    gCO2L_default = FUEL_DEFAULTS[fuel_type]["gCO2_per_L"]
    LHV = colD.number_input("LHV [MJ/L]", 10.0, 45.0, float(LHV_default), 0.1)
    gCO2_per_L = colE.number_input("CO₂ factor [g/L]", 0.0, 4000.0, float(gCO2L_default), 10.0)

    # BEV: fator de CO2 da rede
    grid_gCO2 = colF.number_input("Grid CO₂ [g/kWh] (BEV)", 0.0, 1000.0, float(GRID_DEFAULT_gCO2_per_kWh), 5.0)

    # PHEV: Utility Factor (opcional)
    uf = None
    if electrification == "PHEV":
        uf = st.slider("Utility Factor (PHEV) [% electric miles]", 0.0, 100.0, 50.0, 1.0)

    st.markdown(" ")

    if st.button("🧮 Compute & Save scenario"):
        try:
            vde_net = float(vde["vde_net_mj_per_km"])
            if electrification == "BEV":
                out = compute_bev_from_vde(vde_net, bev_eff_drive, grid_gCO2)
            else:
                out = compute_ice_fuel_from_vde(vde_net, eta_pt_est, LHV, gCO2_per_L)

            st.success("Computed:")
            st.write({
                "energy_Wh_per_km": round(out["energy_Wh_per_km"], 1) if out["energy_Wh_per_km"] is not None else None,
                "fuel_l_per_100km": round(out["fuel_l_per_100km"], 2) if out["fuel_l_per_100km"] is not None else None,
                "fuel_km_per_l": round(out["fuel_km_per_l"], 2) if out["fuel_km_per_l"] is not None else None,
                "gco2_per_km": round(out["gco2_per_km"], 1) if out["gco2_per_km"] is not None else None
            })

            # monta linha p/ fuelcons_db
            row = {
                "vde_id": vde_id,
                "electrification": electrification,
                "fuel_type": fuel_type,
                "eta_pt_est": None if electrification=="BEV" else eta_pt_est,
                "bev_eff_drive": bev_eff_drive if electrification=="BEV" else None,
                "utility_factor_pct": uf if electrification=="PHEV" else None,
                "energy_Wh_per_km": out["energy_Wh_per_km"],
                "fuel_km_per_l": out["fuel_km_per_l"],
                "fuel_l_per_100km": out["fuel_l_per_100km"],
                "gco2_per_km": out["gco2_per_km"],
                "method_note": "MVP formula from VDE_NET"
            }
            from src.vde_core.db import insert_fuelcons
            fuelcons_id = insert_fuelcons({k:v for k,v in row.items() if v is not None})
            st.success(f"Scenario saved to fuelcons_db (id={fuelcons_id}).")

        except Exception as e:
            st.error(f"Failed to compute/save scenario: {e}")

    with st.expander("Current VDE row (debug)"):
        st.json(vde)

if __name__ == "__main__":
    main()
