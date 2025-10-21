# pages/PWT_Fuel_Energy_v1_2.py
# EcoDrive Analyst — Página 2 (PWT & Fuel/Energy) — v1.2 modular
# Observação: preserva campos e nomes existentes no seu DB. Novas partes estão isoladas em funções stubs.

from __future__ import annotations
import json
import math
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --- DB / services (use os caminhos reais do seu projeto) ---
from src.vde_core.db import fetchall, fetchone, update_vde, insert_fuelcons, ensure_db, update_row, delete_row
from src.vde_core.services import compute_ice_fuel_from_vde, compute_bev_from_vde
from src.vde_app.plots import plot_scatter_with_overlays, build_scatter_data, build_scatter_from_fuel, show_summary_plots
from src.vde_core.regression import  load_regression_dataset, fit_regression_y_vs_vde, predict_current_consumption, build_payload_from_regression
# --- Reuso Página 1 ---
from pages.VDE_Setup import load_baselines_df, to_float
from src.vde_app.components import  fetch_fuelcons_all, fetch_fuelcons_by_vde, render_fuelcons_table,filters_bar, search_logo, get_legislation_icon
from src.vde_app.derivatives import build_min_payload, enrich_with_derivatives, filter_payload, load_fuelcons_allowed



# Depois (dinâmico):


st.set_page_config(page_title="EcoDrive — Página 2 (PWT & Fuel/Energy)", layout="wide")
ensure_db()

FUELCONS_ALLOWED = load_fuelcons_allowed()

# =============================================================================
# Helpers gerais
# =============================================================================

def drop_empty(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if v not in (None, "")}


def fetch_distinct_transmission_models() -> list[str]:
    rows = fetchall(
        "SELECT DISTINCT transmission_model "
        "FROM vde_db WHERE transmission_model IS NOT NULL AND transmission_model <> '' "
        "ORDER BY transmission_model;"
    )
    return [r["transmission_model"] for r in rows] if rows else []


def _apply_scenario_extras(d: dict) -> dict:
    """Acopla gear_count e final_drive_ratio (se houver) ao payload do fuelcons_db."""
    d = dict(d)
    g = st.session_state.get("pwt_gears")
    f = st.session_state.get("pwt_fdr")
    if g not in (None, ""):
        d["gear_count"] = g
    if f not in (None, ""):
        d["final_drive_ratio"] = f
    return d

def _init_sidebar_defaults():
    if st.session_state.get("_sb_defaults_done"):
        return
    st.session_state.update({
        "sb_eta_trans": 0.92,
        "sb_eta_pt": 0.35,
        "sb_fuel_type": "Gasoline",
        "sb_lhv_override": 34.2,
        "sb_uf": 0.50,
        "sb_eta_drive": 0.88,
        "sb_grid": 400.0,
        "pwt_gears": 6,
        "pwt_fdr": 3.91,
    })
    st.session_state["_sb_defaults_done"] = True

# =============================================================================
# Sidebar — orquestração
# =============================================================================


def sidebar_vde_selector_and_context() -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Sidebar central:
      - Seleciona VDE_ID
      - Confirma electrification (default do engine_type)
      - Entradas de transmissão: ou eficiência (eta_trans) OU coeficientes A/B/C (por enquanto, só guardamos no ctx)
      - PARÂMETROS para cálculo: eta_pt, eta_drive, grid, UF, LHV override
      - Captura extras do cenário: gear_count, final_drive_ratio
      - Filtros de regressão + toggle para usar regressão
    Retorna (vde_id, ctx)
    """
    st.sidebar.header("Select your VDE & Parameters (Required)")

    # 1) Seleção de snapshot (VDE_ID)
    vde_id = _sidebar_select_vde_id()
    eta_trans = None
    trans_ABC = None

    
    # 2) Electrification (obrigatório)
    electrif_default = _default_electrification_from_vde(vde_id)
    electrification = st.sidebar.selectbox(
        "Electrification", ["ICE", "HEV", "PHEV", "BEV"],
        index=["ICE","HEV","PHEV","BEV"].index(electrif_default),
        key="sb_electrification",
    )

    # 3) BEV placeholders (opcional)
    apply_bev_placeholders = False
    # defaults antes dos blocos condicionais
    _init_sidebar_defaults()
    if electrification == "BEV":
        apply_bev_placeholders = st.sidebar.checkbox(
            "Apply BEV Place Holdes (engine_size=0.001 etc.)",
            value=True, key="sb_bev_placeholders"
        )
    if vde_id:

        #st.sidebar.divider()
        #analysis_mode = st.sidebar.radio("Analysis mode", ["Parameters", "Regression"], index=1, key="sb_analysis_mode")


        st.sidebar.divider()
        st.sidebar.subheader("Transmission model for VDE_TOTAL")
        trans_mode = st.sidebar.radio(
            "Pick a Mode",
            ["Use transmission global efficiency (η_trans)", "Set transmission drag coefs as velocity function (kph) A/B/C"],
            index=0,
            key="sb_trans_mode",
        )
        if trans_mode == "Use transmission global efficiency (η_trans)":
            eta_trans = st.sidebar.number_input("η_trans (0–1)", min_value=0.0, max_value=1.0, placeholder="0.9",
                                                step=0.005, format="%.3f", key="sb_eta_trans")
        else:
            c1, c2, c3 = st.sidebar.columns(3)
            with c1:
                A = st.number_input("A_trans [N]", min_value=0.0, step=0.1, format="%.2f", key="sb_A_trans")
            with c2:
                B = st.number_input("B_trans [N/kph]", min_value=0.0, step=0.001, format="%.3f", key="sb_B_trans")
            with c3:
                C = st.number_input("C_trans [N/kph²]", min_value=0.0, step=0.0001, format="%.4f", key="sb_C_trans")
            trans_ABC = (A, B, C)

        st.sidebar.caption("Pode usar 'Defaults' por category/transmission_type mais tarde (to-do)")


    ctx: Dict[str, Any] = {
        "electrification": electrification,
        "apply_bev_placeholders": apply_bev_placeholders,
        "eta_trans": eta_trans,
        "trans_ABC": trans_ABC,
    }
    return vde_id, ctx

def _sidebar_select_vde_id() -> Optional[int]:
    df = load_baselines_df()
    if df.empty:
        st.sidebar.info("No VDE_DB Snapshots. Create one on Page VDE Setup.")
        return None
    opts = (
        df.assign(_label=df.apply(
            lambda r: f"#{int(r['id'])} — {r['make']} {r['model']} "
                      f"{int(r['year']) if pd.notna(r['year']) else ''} [{r['legislation']}]",
            axis=1))
          .sort_values("id", ascending=False)[["id", "_label"]]
          .values.tolist()
    )
    label_to_id = {label: int(_id) for _id, label in opts}
    choice = st.sidebar.selectbox("VDE Snapshot", list(label_to_id.keys()), key="sb_snap_selector")
    return label_to_id.get(choice)


def _default_electrification_from_vde(vde_id: Optional[int]) -> str:
    if not vde_id:
        return "ICE"
    row = fetchone("SELECT engine_type FROM vde_db WHERE id=?;", (vde_id,)) or {}
    et = (row.get("engine_type") or "").upper()
    if et == "BEV":
        return "BEV"
    if et == "HEV":
        return "HEV"
    return "ICE"


def apply_bev_placeholders_if_needed(vde_id: int, electrification: str) -> None:
    """Se usuário marcou e BEV foi confirmado, aplica placeholders mínimos no snapshot."""
    if electrification != "BEV":
        return
    if not st.session_state.get("sb_bev_placeholders"):
        return

    payload = drop_empty({
        "engine_model": "",
        "engine_size_l": 0.001,
        "engine_aspiration": "",
        "transmission_type": "SS",
    })
    try:
        update_vde(vde_id, payload)
        st.sidebar.success("Placeholders BEV applied to Snapshot.")
    except Exception as e:
        st.sidebar.warning(f"Não foi possível aplicar placeholders BEV: {e}")


# =============================================================================
# Header fixo (somente leitura)
# =============================================================================

def fixed_header(vde_row: dict):
    st.markdown("### Baseline selected")
    i1, i2, i3, i4 = st.columns([1, 1, 4, 2])


    with i1:
        brand_icon = vde_row.get("brand_icon") or vde_row.get("brand_logo")
        if brand_icon:
            st.image(brand_icon, width=64, caption=vde_row.get('make'))
    with i2:
        leg_icon = vde_row.get("leg_icon") or vde_row.get("legislation_icon")
        if leg_icon:
            st.image(leg_icon, width=64, caption=vde_row.get('legislation'))
    with i3:
        title = f"**{vde_row.get('make','?')} {vde_row.get('model','?')}**"
        subtitle = f"{vde_row.get('year','?')} · {vde_row.get('category','?')} · {vde_row.get('legislation','?')}"
        st.markdown(f"{title}\n\n{subtitle}")
        vde = to_float(vde_row.get("vde_net_mj_per_km"))
        if vde is not None:
            st.caption(f"VDE_NET: {vde:.3f} MJ/km")
    with i4:
        mass = to_float(vde_row.get("mass_kg"))
        cda  = to_float(vde_row.get("cda_m2"))
        st.metric("Mass [kg]", f"{mass:.0f}" if mass is not None else "—")
        st.metric("CdA [m²]",  f"{cda:.3f}"  if cda  is not None else "—")



# =============================================================================
# Cálculo VDE_TOTAL (stub) e criação de cenários
# =============================================================================

def compute_vde_total_from_ctx(vde_row: dict, ctx: Dict[str, Any]) -> Dict[str, float]:
    """
    Por enquanto, retorna VDE_NET sem alteração.
    Futuro: se ctx["eta_trans"] -> VDE_TOTAL = VDE_NET / eta_trans
            se ctx["trans_ABC"]   -> integrar perdas A/B/C ao VDE (conforme sua convenção)
    """
    vde_net = to_float(vde_row.get("vde_net_mj_per_km")) or 0.0
    result = {
        "vde_net_mj_per_km": vde_net,
        "vde_total_mj_per_km": (vde_net / ctx["eta_trans"]) if ctx.get("eta_trans") else None,
    }
    return result

# =============================================================================
# Sections & tabs
# =============================================================================

def section_parameters_card(vde_id: int, vde_net_mj_per_km: float, electrification: str) -> Dict[str, Any]:
    """
    Substitui 'calculation_parameters_block'. Renderiza parâmetros e salva um cenário
    baseado neles (igual ao fluxo da seção de regressão).
    Retorna um pequeno ctx com os valores escolhidos (se você quiser reaproveitar).
    """
    st.header("Parameters-based estimation")

    # ---------- Inputs ----------
    eta_pt = fuel_type = lhv_override = uf_phev = eta_drive = grid = None

    if electrification in ("ICE", "HEV", "PHEV"):
        st.markdown("**Parameters for ICE / MHEV / HEV / PHEV**")
        c1, c2, c3, c4 = st.columns(4)

        eta_pt = c1.number_input("η_pt (ICE/MHEV/HEV/PHEV)",
                                 min_value=0.0, max_value=1.0, step=0.005, format="%.3f",
                                 key="sb_eta_pt")
        fuel_type = c2.selectbox("Fuel type", ["Gasoline", "E10", "E22", "E100", "Diesel", "Other"],
                                 key="sb_fuel_type")
        lhv_override = c3.number_input("LHV [MJ/L] (override opcional)",
                                       min_value=0.0, step=0.1, format="%.2f", key="sb_lhv_override")
        uf_phev = c4.number_input("UF PHEV (0–1)",
                                  min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="sb_uf")
    else:
        # BEV / PHEV elétrico
        if electrification == "PHEV":
            st.session_state["sb_uf"] = 0.50
        eta_drive = st.number_input("Driveline efficiency (BEV/PHEV elétrico)",
                                    min_value=0.0, max_value=1.0, step=0.005, format="%.3f",
                                    key="sb_eta_drive")
        grid = st.number_input("Grid [gCO₂/kWh]", min_value=0.0, step=1.0, format="%.0f",
                               key="sb_grid")

    # ---------- Scenario Extras (mesmos keys que você já usa) ----------
    st.divider()
    st.subheader("Scenario Extras (fuelcons_db)")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Gears (scenario)", min_value=0, step=1, key="pwt_gears", placeholder="6")
    c2.number_input("Final drive ratio (scenario)", min_value=0.0, step=0.01, format="%.2f",
                    key="pwt_fdr", placeholder="3.91")
    # Transmission model picker (via DISTINCT)
    trans_models = fetch_distinct_transmission_models() or []
    trans_models.append("Other...")
    choice = c3.selectbox("Transmission model (scenario)", trans_models, key="pwt_trans_model_choice")
    if choice == "Other...":
        tm_value = st.text_input("Type transmission model", key="pwt_trans_model_custom")
    else:
        tm_value = choice
    st.session_state["pwt_trans_model"] = (tm_value or "").strip() or None

    # ---------- Estimativa (param-based) ----------
    # BEV: E = VDE_NET / η_drive  -> Wh/km = (MJ/km)/η * 277.7778
    # ICE/HEV: L/100 = (VDE_NET/η_pt)/LHV * 100
    # PHEV (simplificado): energy = UF*E_elec; fuel = (1-UF)*L/100
    def _lhv_default(ft: str) -> float:
        # simples: se você já tem um mapa global, substitua aqui
        base = {"GASOLINE": 34.2, "E10": 33.2, "E22": 30.5, "E100": 21.2, "DIESEL": 38.6, "OTHER": 34.2}
        return base.get((ft or "GASOLINE").upper(), 34.2)

    yhat: Dict[str, float] = {}

    if electrification == "BEV":
        if eta_drive and eta_drive > 0:
            yhat["energy_Wh_per_km"] = (vde_net_mj_per_km / eta_drive) * 277.7778

    elif electrification in ("ICE", "HEV"):
        LHV = float(lhv_override) if (lhv_override and lhv_override > 0) else _lhv_default(fuel_type or "Gasoline")
        if eta_pt and eta_pt > 0 and LHV > 0:
            yhat["fuel_l_per_100km"] = (vde_net_mj_per_km / eta_pt) / LHV * 100.0

    elif electrification == "PHEV":
        uf = float(uf_phev) if uf_phev is not None else 0.5
        uf = max(0.0, min(1.0, uf))
        # elétrico
        if eta_drive and eta_drive > 0:
            e_elec = (vde_net_mj_per_km / eta_drive) * 277.7778
            yhat["energy_Wh_per_km"] = uf * e_elec
        # combustão
        LHV = float(lhv_override) if (lhv_override and lhv_override > 0) else _lhv_default(fuel_type or "Gasoline")
        if eta_pt and eta_pt > 0 and LHV > 0:
            l100_fuel = (vde_net_mj_per_km / eta_pt) / LHV * 100.0
            yhat["fuel_l_per_100km"] = (1.0 - uf) * l100_fuel

    # Preview rápido
    st.caption("Preview (parameters-based ŷ):")
    st.write(yhat)

    # ---------- Montagem do payload ----------
    # Para param-based não precisamos mapear por legislação; salvamos o combinado nas chaves base
    payload = {"vde_id": vde_id, "electrification": electrification, "method_note": "Parameters-based estimation"}
    # injeta ŷ base
    for k in ("energy_Wh_per_km", "fuel_l_per_100km"):
        if yhat.get(k) is not None:
            payload[k] = float(yhat[k])

    # Derivados (km/L, gCO2, energia a partir de L/100 etc.)
    payload = enrich_with_derivatives(payload, electrification, fuel_type=fuel_type or "Gasoline")

    # Scenario extras (gears/fdr/trans_model, etc.)
    payload = _apply_scenario_extras(payload)

    payload = filter_payload(payload)
    st.write(payload)
    # ---------- Salvar ----------
    csave1, csave2 = st.columns([1, 3])
    with csave1:
        if st.button("💾 Save (Parameters)", use_container_width=True, key="btn_save_params"):
            to_save = payload
            # garantias mínimas
            to_save.setdefault("vde_id", vde_id)
            to_save.setdefault("electrification", electrification)
            to_save.setdefault("method_note", "Parameters-based estimation")

            insert_fuelcons(to_save)
            st.success("Scenario saved (parameters).")

    # ctx opcional de retorno
    ctx: Dict[str, Any] = {
        "eta_pt": eta_pt,
        "eta_drive": eta_drive,
        "grid_gco2_per_kwh": grid,
        "uf_phev": uf_phev,
        "fuel_type": fuel_type,
        "lhv_override": lhv_override or None,
    }
    return ctx

def section_regression_card(vde_id: int, electrification: str, filters: Dict[str, Any], vde_net: float):
    # ---------- Scenario Extras (mesmos keys que você já usa) ----------
    st.divider()
    st.subheader("Scenario Extras (fuelcons_db)")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Gears (scenario)", min_value=0, step=1, key="pwt_gears", placeholder="6")
    c2.number_input("Final drive ratio (scenario)", min_value=0.0, step=0.01, format="%.2f",
                    key="pwt_fdr", placeholder="3.91")
    # Transmission model picker (via DISTINCT)
    trans_models = fetch_distinct_transmission_models() or []
    trans_models.append("Other...")
    choice = c3.selectbox("Transmission model (scenario)", trans_models, key="pwt_trans_model_choice")
    if choice == "Other...":
        tm_value = st.text_input("Type transmission model", key="pwt_trans_model_custom")
    else:
        tm_value = choice
    st.session_state["pwt_trans_model"] = (tm_value or "").strip() or None
    st.subheader("Regression (aligned with filters above)")
    # dataset com MESMOS filtros da visualização:
    regdf = load_regression_dataset(filters, current_vde_id=vde_id)  # <- sua função (já com power_bin e vde_id se você adicionou)
    model = fit_regression_y_vs_vde(regdf, y_col=None, electrification=electrification)  # modo split (urb/HW) embutido
    yhat = predict_current_consumption(model, vde_net, electrification)
    payload = build_min_payload(vde_id, electrification, yhat, method_note="EPA/WLTP regression (split urb/hw)")

    payload = enrich_with_derivatives(payload, electrification, fuel_type="Gasoline")  # ajuste fuel_type se for o caso

    payload = _apply_scenario_extras(payload)

    payload = filter_payload(payload)

    # diagnóstico rápido
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Model (Urban):", model.get("urb"))
    with col2:
        st.write("Model (Highway):", model.get("hw"))
    with col2:
        st.write("Model (Combined):", model.get("combined"))

    st.write("Estimate for current snapshot:", yhat)
    payload = _apply_scenario_extras(payload)
    st.write(payload)

    if st.button("💾 Save (Regression)", use_container_width=True, key="btn_save_regression"):
        #payload = {"vde_id": vde_id, "electrification": electrification, "method_note": "EPA/WLTP regression (split urb/hw)"}

        insert_fuelcons(payload)
        st.success("Scenario saved (regression).")

    return model  # devolve o modelo para desenhar as linhas no gráfico

def run_view_panel(vde_id: int, vde_row: dict, ctx: dict) -> None:
    st.subheader("View Filters")
    view_filters = filters_bar(vde_id, ctx["electrification"], key_ns="view")
    view_filters["legislation"] = vde_row.get("legislation")
    ctx["view_filters"] = view_filters  # << corrigido (antes estava com ==)

    # LINHAS de eficiência (fixas por tipo)
    eta_lines = [0.20, 0.25, 0.30, 0.35] if ctx["electrification"] != "BEV" else [0.85, 0.90, 0.95]

    # Dataset único para TABELA + GRÁFICO
    if "vde_id" in view_filters:
        # modo: apenas este VDE
        df_fuel = fetch_fuelcons_all(view_filters)
        df_fuel_table = fetch_fuelcons_by_vde(vde_id)
    else:
        df_fuel = fetch_fuelcons_all(view_filters)
        df_fuel_table = df_fuel

    st.markdown("---")
    st.subheader("Fuel/Energy scenarios")

    # Gráfico primeiro (com mesmos filtros)
    df_plot = build_scatter_from_fuel(df_fuel)
    plot_scatter_with_overlays(df_plot, ctx["electrification"], model=None, eta_lines=eta_lines)

    # Depois a tabela
    render_fuelcons_table(df_fuel_table, editable=True)

def run_regression_panel(vde_id: int, vde_row: dict, ctx: dict, vde_net: float) -> None:
    # Filtros próprios da regressão (independentes dos da View)
    st.subheader("Regression Filters")
    reg_filters = filters_bar(vde_id, ctx["electrification"], key_ns="reg")
    reg_filters["legislation"] = vde_row.get("legislation")
    ctx["reg_filters"] = reg_filters
    eta_lines = [0.20, 0.25, 0.30, 0.35] if ctx["electrification"] != "BEV" else [0.85, 0.90, 0.95, 0.98, 1, 1.05]
    # Dataset de regressão
    regdf = load_regression_dataset(reg_filters, current_vde_id=vde_id)
    model = fit_regression_y_vs_vde(regdf, y_col=None, electrification=ctx["electrification"])

    df_fuel_reg = fetch_fuelcons_all(reg_filters)
    df_plot = build_scatter_from_fuel(df_fuel_reg)  
    plot_scatter_with_overlays(df_plot, ctx["electrification"], model, eta_lines)

    # Card da regressão (estima + botão salvar)
    section_regression_card(vde_id, ctx["electrification"], reg_filters, vde_net)


def main():
    st.title("EcoDrive Analyzer 2 — PWT & Fuel/Energy")

    # Sidebar & snapshot
    vde_id, ctx = sidebar_vde_selector_and_context()
    if not vde_id:
        st.stop()

    apply_bev_placeholders_if_needed(vde_id, ctx["electrification"])

    vde_row = fetchone("SELECT * FROM vde_db WHERE id=?;", (vde_id,)) or {}
    totals = compute_vde_total_from_ctx(vde_row, ctx)
    vde_net = totals["vde_net_mj_per_km"]

    # ícones automáticos no header
    vde_row["brand_icon"] = search_logo(vde_row, base_dir="data/images/logos", fallback="_unknown.png") or ""
    vde_row["leg_icon"]   = get_legislation_icon(vde_row, base_dir="data/images") or ""
    fixed_header(vde_row)

    # ADD (uma linha)
    st.session_state["current_vde_id"] = int(vde_id)
    mode = st.radio(
    "Mode", ["🔎 View", "🧮 Parameters", "📈 Regression"],
    horizontal=True, key="mode_sel"
)

    if mode == "🔎 View":
        run_view_panel(vde_id, vde_row, ctx)
    elif mode == "🧮 Parameters":
        section_parameters_card(vde_id, vde_net, ctx["electrification"])
    else:
        run_regression_panel(vde_id, vde_row, ctx, vde_net)

    # Abas: VIEW vs REGRESSION (claridade máxima)
    #tab_view, tab_params, tab_reg = st.tabs(["🔎 View all vehicles", "🧮 Parameters", "📈 Regression"])

    #with tab_view:
        #run_view_panel(vde_id, vde_row, ctx)
    
    #with tab_params:
        # usa sua seção de parâmetros com botão de salvar
        #section_parameters_card(vde_id, vde_net, ctx["electrification"])

    #with tab_reg:
        #run_regression_panel(vde_id, vde_row, ctx, vde_net)

    st.markdown("---")



if __name__ == "__main__":
    main()



