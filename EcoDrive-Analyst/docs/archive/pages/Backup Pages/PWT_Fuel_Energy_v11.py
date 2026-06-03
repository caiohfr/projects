# pages/2_PWT_FuelCons.py
# Página 2 — PWT & Fuel/Energy
# Referência direta na sua Página 1 (6_mock_Setup.py) e reuso de funções existentes.
# AVISO: este arquivo assume que os imports abaixo existem exatamente como na Página 1.

import json
import math
import pandas as pd
import streamlit as st

# === IMPORTS (estritos, mesmos nomes/caminhos da sua página 1) ===
from src.vde_core.db import fetchall, fetchone, update_vde, insert_fuelcons, ensure_db
from src.vde_core.services import compute_ice_fuel_from_vde, compute_bev_from_vde

# Reuso de utilidades da Página 1 (copie para uma lib comum quando quiser):
from pages.VDE_Setup import (
    to_float,                # -> mover para src.vde_app.shared if desejado
    load_baselines_df,       # usado para listar snapshots
    validate_core,           # validações simples
    vehicle_basics_sidebar,  # mantém meta no sidebar (legislation/category/make...)
)

st.set_page_config(page_title="EcoDrive — Página 2 (PWT & Fuel/Energy)", layout="wide")
ensure_db()

# ==========================
# HELPERS ESPECÍFICOS PÁG.2 (atualizados)
# ==========================

def drop_empty(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if v not in (None, "")}


def select_vde_snapshot() -> int | None:
    """Selector simples baseado em load_baselines_df(). Retorna vde_id ou None."""
    df = load_baselines_df()
    if df.empty:
        st.info("Sem snapshots no VDE_DB. Crie um na Página 1.")
        return None
    # lista compacta
    opts = (
        df.assign(_label=df.apply(lambda r: f"#{int(r['id'])} — {r['make']} {r['model']} {int(r['year']) if pd.notna(r['year']) else ''} [{r['legislation']}]", axis=1))
          .sort_values("id", ascending=False)
          [["id","_label"]]
          .values.tolist()
    )
    label_to_id = {label: int(_id) for _id, label in opts}
    choice = st.selectbox("VDE Snapshot", list(label_to_id.keys()))
    return label_to_id.get(choice)


def fixed_header(vde_row: dict):
    """Header fixo com dados do VDE + ícones, somente exibição."""
    st.markdown("### Baseline selecionado")
    i1, i2, i3, i4 = st.columns([1,1,4,2])
    # Ícones (se existirem na row)
    with i1:
        brand_icon = vde_row.get("brand_icon") or vde_row.get("brand_logo")
        if brand_icon:
            st.image(brand_icon, width=64, caption="brand")
    with i2:
        leg_icon = vde_row.get("leg_icon") or vde_row.get("legislation_icon")
        if leg_icon:
            st.image(leg_icon, width=64, caption="leg")
    with i3:
        title = f"**{vde_row.get('make','?')} {vde_row.get('model','?')}**"
        subtitle = f"{vde_row.get('year','?')} · {vde_row.get('category','?')} · {vde_row.get('legislation','?')}"
        st.markdown(f"{title}{subtitle}")
        vde = vde_row.get("vde_net_mj_per_km")
        if vde is not None:
            st.caption(f"VDE_NET: {vde:.3f} MJ/km")
    with i4:
        mass = vde_row.get("mass_kg")
        cda  = vde_row.get("cda_m2")
        st.metric("Mass [kg]", f"{mass:.0f}" if mass else "—")
        st.metric("CdA [m²]", f"{cda:.3f}" if cda else "—")


def fuelcons_table(vde_id: int):
    """Mostra os cenários do fuelcons_db para o vde_id selecionado (somente exibição)."""
    st.subheader("Cenários salvos em FUELCONS_DB (somente leitura)")
    q = (
        "SELECT id, created_at, method_note, fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km "
        "FROM fuelcons_db WHERE vde_id=? ORDER BY created_at DESC"
    )
    rows = fetchall(q, (vde_id,))
    if not rows:
        st.info("Nenhum cenário salvo para este VDE.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

def drop_empty(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if v not in (None, "")}


def select_vde_snapshot() -> int | None:
    """Selector simples baseado em load_baselines_df(). Retorna vde_id ou None."""
    df = load_baselines_df()
    if df.empty:
        st.info("Sem snapshots no VDE_DB. Crie um na Página 1.")
        return None
    # lista compacta
    opts = (
        df.assign(_label=df.apply(lambda r: f"#{int(r['id'])} — {r['make']} {r['model']} {int(r['year']) if pd.notna(r['year']) else ''} [{r['legislation']}]", axis=1))
          .sort_values("id", ascending=False)
          [["id","_label"]]
          .values.tolist()
    )
    label_to_id = {label: int(_id) for _id, label in opts}
    choice = st.selectbox("VDE Snapshot", list(label_to_id.keys()))
    return label_to_id.get(choice)

def _apply_scenario_extras(d: dict) -> dict:
    """Acopla gear_count e final_drive_ratio (se houver) ao payload do fuelcons_db."""
    d = dict(d)  # cópia
    g = st.session_state.get("pwt_gears")
    f = st.session_state.get("pwt_fdr")
    if g not in (None, ""):
        d["gear_count"] = g
    if f not in (None, ""):
        d["final_drive_ratio"] = f
    return d

# =================
# SEÇÕES PRINCIPAIS
# =================

def pwt_sidebar(vde_id: int):
    """Sidebar para editar/completar PWT no vde_db; corpo da página só exibe.
    ATENÇÃO: só gravamos colunas que EXISTEM em vde_db (ver schema em db.py).
    """
    st.sidebar.header("PWT — editar e salvar")
    engine_model   = st.sidebar.text_input("Engine model (opcional)")
    engine_size_l  = st.sidebar.number_input("Engine size [L]", min_value=0.0, step=0.1, format="%.2f")
    engine_asp     = st.sidebar.selectbox("Aspiration", ["", "NA", "Turbo", "Supercharged"])
    transmission_model = st.sidebar.text_input("Transmission model (opcional)")
    transmission_type  = st.sidebar.selectbox("Transmission type", ["", "AT", "DCT", "CVT", "MT", "eCVT", "Direct"]) 
    drive_type         = st.sidebar.selectbox("Drive type", ["", "FWD", "RWD", "AWD", "4WD"]) 

    if st.sidebar.button("💾 Save PWT → VDE_DB", use_container_width=True):
        # Somente colunas existentes em vde_db:
        payload = drop_empty({
            "engine_model": engine_model,
            "engine_size_l": engine_size_l,
            "engine_aspiration": engine_asp,
            "transmission_model": transmission_model,
            "transmission_type": transmission_type,
            "drive_type": drive_type,
        })
        payload = {k: v for k, v in payload.items() if v not in ("", None)}
        update_vde(vde_id, payload)
        st.sidebar.success("VDE_DB atualizado.")
        st.sidebar.divider()
        st.sidebar.number_input("Gears (scenario)", min_value=0, step=1, key="pwt_gears")
        st.sidebar.number_input("Final drive ratio (scenario)", min_value=0.0, step=0.01, format="%.2f", key="pwt_fdr")




    with st.expander("Debug / VDE row"):
        row = fetchone("SELECT * FROM vde_db WHERE id=?;", (vde_id,))
        st.write(row)


def section_fuel_energy(vde_id: int):
    st.subheader("Adicionar/Calcular cenários (grava em FUELCONS_DB)")
    tab_infer, tab_official = st.tabs(["Infer from VDE_NET", "Official / Tested values"]) 

    # Helper: mapear electrification padrão a partir de engine_type do snapshot
    def _default_electrification() -> str:
        row = fetchone("SELECT engine_type FROM vde_db WHERE id=?;", (vde_id,)) or {}
        et = (row.get("engine_type") or "").upper()
        if et == "BEV":
            return "BEV"
        if et == "HEV":
            return "HEV"
        # PHEV não tem marcador em vde_db; usuário escolhe manualmente
        return "ICE"

    with tab_infer:
        electrif       = st.selectbox("Electrification", ["ICE", "MHEV", "HEV", "PHEV", "BEV"], index=["ICE","MHEV","HEV","PHEV","BEV"].index(_default_electrification()))
        fuel_type      = st.selectbox("Fuel type", ["Gasoline", "E10", "E22", "E100", "Diesel", "Other"]) 
        lhv_mj_per_l   = st.number_input("LHV [MJ/L] (override opcional)", min_value=0.0, step=0.1, format="%.2f")
        c1, c2, c3 = st.columns(3)
        with c1:
            eta_pt = st.number_input("η_pt (ICE/MHEV/HEV/PHEV)", min_value=0.0, max_value=1.0, step=0.005, format="%.3f")
        with c2:
            driveline_eff = st.number_input("Driveline eff (BEV/PHEV elétrico)", min_value=0.0, max_value=1.0, step=0.005, format="%.3f")
        with c3:
            grid_gco2_per_kwh = st.number_input("Grid [gCO₂/kWh] (BEV/PHEV)", min_value=0.0, step=1.0, format="%.0f")
        uf_phev = st.number_input("UF PHEV (0-1)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

        if st.button("⚙️ Compute & Save (Infer)", use_container_width=True):
            try:
                if electrif == "BEV":
                    res = compute_bev_from_vde(vde_id, driveline_eff=driveline_eff, grid_gco2_per_kwh=grid_gco2_per_kwh)
                    # payload alinhado ao schema de fuelcons_db (electrification NOT NULL)
                    payload = drop_empty({
                        **res,
                        "vde_id": vde_id,
                        "electrification": "BEV",
                        "bev_eff_drive": driveline_eff or None,
                        "method_note": "MVP formula from VDE_NET",
                    })
                else:
                    res = compute_ice_fuel_from_vde(
                        vde_id,
                        fuel_type=fuel_type,
                        eta_pt=eta_pt,
                        lhv_mj_per_l=lhv_mj_per_l or None,
                        electrification=electrif,
                        uf_phev=uf_phev if electrif == "PHEV" else None,
                        driveline_eff=driveline_eff if electrif == "PHEV" else None,
                        grid_gco2_per_kwh=grid_gco2_per_kwh if electrif == "PHEV" else None,
                    )
                    payload = drop_empty({
                        **res,
                        "vde_id": vde_id,
                        "electrification": electrif,      # REQUIRED pelo schema
                        "fuel_type": fuel_type or None,
                        "eta_pt_est": eta_pt or None,
                        "utility_factor_pct": (uf_phev*100.0) if (electrif=="PHEV" and uf_phev is not None) else None,
                        "method_note": "MVP formula from VDE_NET",
                    })
                payload = _apply_scenario_extras(payload)
                insert_fuelcons(payload)
                st.success("Cenário inferido salvo em FUELCONS_DB.")
                st.write(res)
            except Exception as e:
                st.error(f"Falha no compute/save: {e}")

    with tab_official:
        # eletrificação obrigatória no schema; permitir escolher aqui
        electrif_form = st.selectbox("Electrification (required)", ["ICE", "MHEV", "HEV", "PHEV", "BEV"], index=["ICE","MHEV","HEV","PHEV","BEV"].index(_default_electrification()))
        cycle      = st.selectbox("Cycle", ["FTP-75", "HWFET", "US06", "WLTC", "NBR6601", "Other"])  # apenas UI (não gravamos ainda)
        procedure  = st.text_input("Test procedure / label")  # apenas UI por enquanto
        source     = st.text_input("Source (EPA label / INMETRO / CoC / Internal)")  # apenas UI por enquanto
        test_year  = st.number_input("Test year", min_value=1990, max_value=2100, step=1)  # apenas UI por enquanto
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            L_100 = st.number_input("Fuel [L/100km]", min_value=0.0, step=0.01)
        with d2:
            km_L = st.number_input("Fuel [km/L]", min_value=0.0, step=0.01)
        with d3:
            Wh_km = st.number_input("Energy [Wh/km]", min_value=0.0, step=1.0)
        with d4:
            gCO2_km = st.number_input("CO₂ [g/km]", min_value=0.0, step=0.1)

        if st.button("💾 Save Official/Tested", use_container_width=True):
            row = drop_empty({
                "vde_id": vde_id,
                "electrification": electrif_form,   # REQUIRED no schema
                # Campos de metadados (cycle/procedure/source/test_year) ficam só na UI por enquanto
                "fuel_l_per_100km": L_100 or None,
                "fuel_km_per_l": km_L or None,
                "energy_Wh_per_km": Wh_km or None,
                "gco2_per_km": gCO2_km or None,
                "method_note": "Official/Direct entry",
            })
            # ... você monta o row ...
            row = _apply_scenario_extras(row)        
            insert_fuelcons(row)
            st.success("Valores oficiais salvos em FUELCONS_DB.")


    with tab_official:
        cycle      = st.selectbox("Cycle", ["FTP-75", "HWFET", "US06", "WLTC", "NBR6601", "Other"], key="official_cycle")
        procedure  = st.text_input("Test procedure / label", key="official_procedure")
        source     = st.text_input("Source (EPA label / INMETRO / CoC / Internal)" , key="official_source")
        test_year  = st.number_input("Test year", min_value=1990, max_value=2100, step=1, key="official_test_year")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            L_100 = st.number_input("Fuel [L/100km]", min_value=0.0, step=0.01, key="official_L_100")
        with d2:
            km_L = st.number_input("Fuel [km/L]", min_value=0.0, step=0.01, key="official_km_L")
        with d3:
            Wh_km = st.number_input("Energy [Wh/km]", min_value=0.0, step=1.0, key="official_Wh_km")
        with d4:
            gCO2_km = st.number_input("CO₂ [g/km]", min_value=0.0, step=0.1, key="official_gCO2_km")

        if st.button("💾 Save Official/Tested", use_container_width=True, key= "official_save_button"):
            row = drop_empty({
                "vde_id": vde_id,
                # campos opcionais do formulário ainda não existem no schema; mantemos no UI por futuro
                # "cycle": cycle,
                # "test_procedure": procedure,
                # "test_year": int(test_year) if test_year else None,
                # "source": source or None,
                "fuel_l_per_100km": L_100 or None,
                "fuel_km_per_l": km_L or None,
                "energy_Wh_per_km": Wh_km or None,
                "gco2_per_km": gCO2_km or None,
                "method_note": "Official/Direct entry",
            })
            insert_fuelcons(row)
            st.success("Valores oficiais salvos em FUELCONS_DB.")
        if st.button("⚙️ Compute & Save (Infer)", use_container_width=True, key="infer_save_button"):
            try:
                if electrif == "BEV":
                    res = compute_bev_from_vde(vde_id, driveline_eff=driveline_eff, grid_gco2_per_kwh=grid_gco2_per_kwh)
                else:
                    res = compute_ice_fuel_from_vde(
                        vde_id,
                        fuel_type=fuel_type,
                        eta_pt=eta_pt,
                        lhv_mj_per_l=lhv_mj_per_l or None,
                        electrification=electrif,
                        uf_phev=uf_phev if electrif == "PHEV" else None,
                    )
                payload = drop_empty({
                    **res,
                    "vde_id": vde_id,
                    "method_note": "MVP formula from VDE_NET",
                    "source": None,
                    "assumptions_json": res.get("assumptions_json"),
                })
                # NOTE: reuso de insert_fuelcons (já importado)
                insert_fuelcons(payload)
                st.success("Cenário inferido salvo em FUELCONS_DB.")
                st.write(res)
            except Exception as e:
                st.error(f"Falha no compute/save: {e}")

    with tab_official:
        cycle      = st.selectbox("Cycle", ["FTP-75", "HWFET", "US06", "WLTC", "NBR6601", "Other"], key="official_cycle2")
        procedure  = st.text_input("Test procedure / label", key="official_procedure2")
        source     = st.text_input("Source (EPA label / INMETRO / CoC / Internal)", key="official_source2")
        test_year  = st.number_input("Test year", min_value=1990, max_value=2100, step=1, key="official_test_year2")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            L_100 = st.number_input("Fuel [L/100km]", min_value=0.0, step=0.01, key="official_L_100_2")
        with d2:
            km_L = st.number_input("Fuel [km/L]", min_value=0.0, step=0.01, key="official_km_L_2")
        with d3:
            Wh_km = st.number_input("Energy [Wh/km]", min_value=0.0, step=1.0, key="official_Wh_km_2")
        with d4:
            gCO2_km = st.number_input("CO₂ [g/km]", min_value=0.0, step=0.1, key="official_gCO2_km_2")

        if st.button("💾 Save Official/Tested", use_container_width=True,key="infer_save_button2"):
            row = drop_empty({
                "vde_id": vde_id,
                "cycle": cycle,
                "test_procedure": procedure,
                "test_year": int(test_year) if test_year else None,
                "fuel_l_per_100km": L_100 or None,
                "fuel_km_per_l": km_L or None,
                "energy_Wh_per_km": Wh_km or None,
                "gco2_per_km": gCO2_km or None,
                "source": source or None,
                "method_note": "Official/Direct entry",
            })
            insert_fuelcons(row)
            st.success("Valores oficiais salvos em FUELCONS_DB.")


# =============== SCATTER (final da página) ===============


def build_scatter_data() -> pd.DataFrame:
    """
    Usa somente colunas que existem hoje no seu DB.
    - fuelcons_db: fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km, method_note
    - vde_db: vde_net_mj_per_km, engine_size_l, transmission_type, drive_type, category, make, model, year
    """
    q = """
    SELECT 
        f.vde_id,
        f.fuel_l_per_100km,
        f.fuel_km_per_l,
        f.energy_Wh_per_km,
        f.gco2_per_km,
        f.method_note,
        v.vde_net_mj_per_km,
        v.engine_size_l,
        v.transmission_type,
        v.drive_type,
        v.category,
        v.make,
        v.model,
        v.year
    FROM fuelcons_db f
    JOIN vde_db v ON v.id = f.vde_id
    """
    rows = fetchall(q)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # renomeia para manter compatibilidade com o restante dos plots
    if "transmission_type" in df.columns:
        df.rename(columns={"transmission_type": "trans_topology"}, inplace=True)
    if "category" in df.columns:
        df.rename(columns={"category": "vehicle_class"}, inplace=True)

    # grupo de cor
    def _map_color_group(r):
        if pd.notna(r.get("energy_Wh_per_km")) and pd.isna(r.get("fuel_l_per_100km")):
            return r.get("vehicle_class") or r.get("drive_type") or "BEV"
        size = r.get("engine_size_l")
        try:
            size = float(size) if size is not None else None
        except Exception:
            size = None
        if size is None:
            return r.get("trans_topology") or "Unknown"
        if size <= 1.0:  return "≤1.0L"
        if size <= 1.6:  return "1.1–1.6L"
        if size <= 2.0:  return "1.7–2.0L"
        return ">2.0L"

    df["color_group"] = df.apply(_map_color_group, axis=1)
    df["is_bev"] = df["energy_Wh_per_km"].notna() & df["fuel_l_per_100km"].isna()
    df = df[df["vde_net_mj_per_km"].notna()]
    return df



def show_summary_plots(df: pd.DataFrame):
    st.subheader("Summary — Energy/Consumption vs VDE_NET")
    if df.empty:
        st.info("Sem dados para plot.")
        return

    df_bev = df[(df["is_bev"] == True) & df["energy_Wh_per_km"].notna()]
    df_ice = df[(df["is_bev"] == False) & df["fuel_l_per_100km"].notna()]

    try:
        import plotly.express as px
        if not df_bev.empty:
            st.markdown("**BEV: Energy [Wh/km] vs VDE_NET [MJ/km]**")
            fig1 = px.scatter(
                df_bev, x="vde_net_mj_per_km", y="energy_Wh_per_km", color="color_group",
                hover_data=["make","model","year","category","drive_type"]
            )
            fig1.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Sem dados BEV válidos.")

        if not df_ice.empty:
            st.markdown("**ICE/MxHEV: Fuel [L/100km] vs VDE_NET [MJ/km]**")
            fig2 = px.scatter(
                df_ice, x="vde_net_mj_per_km", y="fuel_l_per_100km", color="color_group",
                hover_data=["make","model","model_year","engine_size_l","drive_type"]
            )
            fig2.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Sem dados ICE/Híbridos válidos.")

    except Exception as e:
        st.warning(f"Plotly indisponível ({e}).")


# =============== MAIN ===============

def main():
    # Sidebar agora é dedicado a editar PWT do snapshot
    st.title("EcoDrive Analyst · Página 2 — PWT & Fuel/Energy")

    st.markdown("---")
    st.subheader("Selecionar baseline / snapshot")
    vde_id = select_vde_snapshot()
    if not vde_id:
        st.stop()

    # Carrega a row e fixa header SOMENTE EXIBIÇÃO
    vde_row = fetchone("SELECT * FROM vde_db WHERE id=?;", (vde_id,)) or {}
    fixed_header(vde_row)

    # Sidebar: edição PWT (salva no vde_db)
    pwt_sidebar(vde_id)

    st.markdown("---")
    # Lista de cenários já salvos para este VDE (somente exibição)
    fuelcons_table(vde_id)

    st.markdown("---")
    # Abaixo: blocos para adicionar novos cenários
    section_fuel_energy(vde_id)

    st.markdown("---")
    # Gráficos finais
    df = build_scatter_data()
    show_summary_plots(df)


if __name__ == "__main__":
    main()