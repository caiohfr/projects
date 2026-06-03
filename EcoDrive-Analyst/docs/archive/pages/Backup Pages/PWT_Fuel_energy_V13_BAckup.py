# PWT_Fuel_Energy_redux.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st
from math import isfinite
# === SUAS FUNÇÕES EXISTENTES (importações reais do seu projeto) =================
from src.vde_core.db import fetchall, fetchone, update_vde, insert_fuelcons, ensure_db
from pages.VDE_Setup import load_baselines_df, to_float
from src.vde_app.plots import build_scatter_data, build_scatter_from_fuel, show_summary_plots
from src.vde_app.derivatives import build_min_payload, enrich_with_derivatives, filter_payload
# Regressão (mesmos nomes que você já tem, inclusive o modo split embutido):
from pages.PWT_Fuel_Energy import (  # ajuste o módulo se o nome for outro
    load_regression_dataset, fit_regression_y_vs_vde, predict_current_consumption
)
# Plots base (vamos reaproveitar estilo, mas o overlay é novo)
# from pages.PWT_Fuel_Energy import build_scatter_data  # opcional, podemos filtrar via SQL aqui
# Suas helpers:
from pages.PWT_Fuel_Energy import (
    fixed_header,                     # <- sua função (header)
    apply_bev_placeholders_if_needed, # <- sua função BEV placeholders
    fetch_fuelcons_by_vde,            # <- sua função tabela por vde
    fetch_fuelcons_all,               # <- sua função tabela "all"
    render_fuelcons_table,            # <- sua função render tabela
    _apply_scenario_extras            # <- sua função: gears/fdr/trans_model → payload
)

st.set_page_config("EcoDrive — Página 2 (Redux)", layout="wide")
ensure_db()

# =============================================================================
# 0) Helpers novos (pequenos)
# =============================================================================

def _sidebar_minimal() -> Tuple[Optional[int], Dict[str, Any]]:
    """Sidebar slim: escolhe VDE + electrification; define transmissão (η_trans ou A/B/C);
       escolhe o modo de análise (Parameters vs Regression). Extras de cenário permanecem aqui."""
    st.sidebar.header("Select your VDE & Electrification")

    vde_id = _sidebar_select_vde_id()

    electrif_default = _default_electrification_from_vde(vde_id)
    electrification = st.sidebar.selectbox(
        "Electrification", ["ICE", "MHEV", "HEV", "PHEV", "BEV"],
        index=["ICE","MHEV","HEV","PHEV","BEV"].index(electrif_default),
        key="sb_electrification",
    )

    # BEV placeholders toggle (igual ao seu fluxo)
    apply_bev = False
    if electrification == "BEV":
        apply_bev = st.sidebar.checkbox("Apply BEV placeholders (engine_size=0.001, SS)", value=False, key="sb_bev_placeholders")

    st.sidebar.divider()
    st.sidebar.subheader("Transmission model for VDE_TOTAL")
    trans_mode = st.sidebar.radio(
        "Pick a Mode",
        ["Use transmission global efficiency (η_trans)", "Set transmission drag coefs as A/B/C (N, N/kph, N/kph²)"],
        index=0, key="sb_trans_mode"
    )

    eta_trans = None
    trans_ABC = None
    if trans_mode.startswith("Use transmission"):
        eta_trans = st.sidebar.number_input("η_trans (0–1)", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="sb_eta_trans")
    else:
        c1, c2, c3 = st.sidebar.columns(3)
        with c1:
            A = st.number_input("A_trans [N]", min_value=0.0, step=0.1, format="%.2f", key="sb_A_trans")
        with c2:
            B = st.number_input("B_trans [N/kph]", min_value=0.0, step=0.001, format="%.3f", key="sb_B_trans")
        with c3:
            C = st.number_input("C_trans [N/kph²]", min_value=0.0, step=0.0001, format="%.4f", key="sb_C_trans")
        trans_ABC = (A, B, C)

    # Scenario extras (mantidos no sidebar como você queria)
    st.sidebar.divider()
    st.sidebar.subheader("Scenario Extras (fuelcons_db)")
    st.sidebar.number_input("Gears (scenario)", min_value=0, step=1, key="pwt_gears")
    st.sidebar.number_input("Final drive ratio (scenario)", min_value=0.0, step=0.01, format="%.2f", key="pwt_fdr")
    # transmission_model (select distinct + Other…)
    _scenario_transmission_model_picker()

    st.sidebar.divider()
    analysis_mode = st.sidebar.radio("Analysis mode", ["Parameters", "Regression"], index=1, key="sb_analysis_mode")

    ctx = {
        "electrification": electrification,
        "apply_bev_placeholders": apply_bev,
        "eta_trans": eta_trans,
        "trans_ABC": trans_ABC,
        "analysis_mode": analysis_mode,
    }
    return vde_id, ctx


def _sidebar_select_vde_id() -> Optional[int]:
    df = load_baselines_df()
    if df.empty:
        st.sidebar.info("No VDE snapshots. Go to Page 1 to create one.")
        return None
    opts = (
        df.assign(_label=df.apply(lambda r: f"#{int(r['id'])} — {r['make']} {r['model']} {int(r['year']) if pd.notna(r['year']) else ''} [{r['legislation']}]", axis=1))
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
    if et == "BEV": return "BEV"
    if et == "HEV": return "HEV"
    if et == "MHEV": return "MHEV"
    if et == "PHEV": return "PHEV"
    return "ICE"


def _scenario_transmission_model_picker() -> None:
    """DISTINCT transmission_model do vde_db + 'Other…'; guarda em session_state."""
    try:
        rows = fetchall(
            "SELECT DISTINCT transmission_model FROM vde_db "
            "WHERE transmission_model IS NOT NULL AND transmission_model <> '' "
            "ORDER BY transmission_model;"
        )
        models = [r["transmission_model"] for r in rows] if rows else []
    except Exception:
        models = []

    models.append("Other…")
    choice = st.sidebar.selectbox("Transmission model (scenario)", models, key="pwt_trans_model_choice")
    if choice == "Other…":
        tm = st.sidebar.text_input("Type transmission model", key="pwt_trans_model_custom")
    else:
        tm = choice
    st.session_state["pwt_trans_model"] = (tm or "").strip() or None


def _filters_bar(vde_id: int, electrification: str) -> Dict[str, Any]:
    """Barra de filtros única (os mesmos filtros valem para visualização e regressão)."""
    st.markdown("### Filters")
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1])

    # options via DISTINCT
    cats = [r["category"] for r in fetchall("SELECT DISTINCT category FROM vde_db WHERE category IS NOT NULL AND category<>'' ORDER BY category;")] or []
    makes = [r["make"] for r in fetchall("SELECT DISTINCT make FROM vde_db WHERE make IS NOT NULL AND make<>'' ORDER BY make;")] or []
    elecs = [r["electrification"] for r in fetchall("SELECT DISTINCT electrification FROM fuelcons_db WHERE electrification IS NOT NULL AND electrification<>'' ORDER BY electrification;")] or ["ICE","MHEV","HEV","PHEV","BEV"]

    with c1:
        view_scope = st.selectbox("View", ["Only this Vehicle id", "All"], index=1, key="fl_scope")
    with c2:
        elec_choice = st.selectbox("Electrification", ["(all)", f"(current: {electrification})"] + [e for e in elecs if e != electrification], key="fl_elec")
    with c3:
        cat_choice = st.selectbox("Category", ["(all)"] + cats, key="fl_cat")
    with c4:
        make_choice = st.selectbox("Make", ["(all)"] + makes, key="fl_make")
    with c5:
        p_choice = st.selectbox("Power (hp)", ["(all)", "≤160", "161–270", "271–470", "471–670", ">670"], key="fl_pbin")

    filters: Dict[str, Any] = {}
    if view_scope == "Only this Vehicle id":
        filters["vde_id"] = vde_id

    if elec_choice not in ("(all)", f"(current: {electrification})"):
        filters["electrification"] = elec_choice
    elif elec_choice.startswith("(current:"):
        filters["electrification"] = electrification

    if cat_choice != "(all)":
        filters["category"] = cat_choice
    if make_choice != "(all)":
        filters["make"] = make_choice

    # power bin em hp → converte p/ kW na query
    hp_to_kw = lambda hp: float(hp) / 1.34102209
    pmap = {
        "≤160": (0, 160), "161–270": (161, 270),
        "271–470": (271, 470), "471–670": (471, 670), ">670": (671, None)
    }
    if p_choice in pmap:
        lo_hp, hi_hp = pmap[p_choice]
        lo_kw = hp_to_kw(lo_hp) if lo_hp is not None else None
        hi_kw = hp_to_kw(hi_hp) if hi_hp is not None else None
        filters["power_kw_range"] = (lo_kw, hi_kw)

    return filters


def _compute_vde_total_from_ctx(vde_row: dict, ctx: Dict[str, Any]) -> Dict[str, float]:
    """Stub simples: se η_trans existir, VDE_TOTAL = VDE_NET/η_trans; senão, retorna só NET."""
    vde_net = to_float(vde_row.get("vde_net_mj_per_km")) or 0.0
    vde_total = (vde_net / ctx["eta_trans"]) if ctx.get("eta_trans") else None
    return {"vde_net_mj_per_km": vde_net, "vde_total_mj_per_km": vde_total}


# =============================================================================
# 1) Plot com overlays (regressão + linhas de eficiência)
# =============================================================================

def plot_scatter_with_overlays(df: pd.DataFrame, electrification: str, model: Dict[str, Any] | None, eta_lines: List[float] | None):
    """Mostra scatter filtrado + reta(s) de regressão + linhas iso-eficiência."""
    if df.empty:
        st.info("No data for the selected filters.")
        return
    import plotly.express as px
    import plotly.graph_objects as go

    # separa BEV vs ICE/MxHEV
    df_bev = df[(df.get("is_bev", False) == True) & df["energy_Wh_per_km"].notna()]
    df_ice = df[(df.get("is_bev", False) == False) & df["fuel_l_per_100km"].notna()]

    figs = []

    if not df_bev.empty:
        fig = px.scatter(
            df_bev, x="vde_net_mj_per_km", y="energy_Wh_per_km",
            color="color_group", hover_data=["make","model","year","vehicle_class"]
        )
        # linhas de regressão (split: urb/hw → duas linhas)
        if model:
            _add_regression_lines(fig, model, electrification, y_kind="bev")
        # iso-η (driveline)
        if eta_lines:
            _add_eta_lines_bev(fig, eta_lines)
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), title="BEV: Energy [Wh/km] vs VDE_NET [MJ/km]")
        figs.append(fig)

    if not df_ice.empty:
        fig = px.scatter(
            df_ice, x="vde_net_mj_per_km", y="fuel_l_per_100km",
            color="color_group", hover_data=["make","model","year","engine_size_l","trans_topology"]
        )
        if model:
            _add_regression_lines(fig, model, electrification, y_kind="ice")
        if eta_lines:
            _add_eta_lines_ice(fig, eta_lines)  # usa LHV default/override se quiser
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), title="ICE/MxHEV: Fuel [L/100km] vs VDE_NET [MJ/km]")
        figs.append(fig)

    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)


def _add_regression_lines(fig, model: Dict[str, Any], electrification: str, y_kind: str):
    """Desenha linha única ou duas linhas (urb/hw)."""
    import numpy as np
    xs = np.linspace(0.2, 1.2, 50)  # faixa típica de VDE_NET; ajuste se quiser

    def _line(a, b):
        return a + b * xs

    if "a" in model and "b" in model:
        ys = _line(model["a"], model["b"])
        fig.add_scatter(x=xs, y=ys, mode="lines", name="Regression", line=dict(dash="solid"))
        return

    if model.get("_is_split"):
        urb, hw = model.get("urb", {}), model.get("hw", {})
        if urb.get("a") is not None:
            fig.add_scatter(x=xs, y=_line(urb["a"], urb["b"]), mode="lines", name="Reg. Urban", line=dict(dash="solid"))
        if hw.get("a") is not None:
            fig.add_scatter(x=xs, y=_line(hw["a"], hw["b"]), mode="lines", name="Reg. Highway", line=dict(dash="dash"))


def _add_eta_lines_bev(fig, eta_list: List[float]):
    """Wh/km = VDE_NET [MJ/km] * 277.78 / η_drive"""
    import numpy as np
    xs = np.linspace(0.2, 1.2, 40)
    for eta in eta_list:
        ys = xs * 277.7778 / max(eta, 1e-9)
        fig.add_scatter(x=xs, y=ys, mode="lines", name=f"η_drive={eta:.2f}", line=dict(width=1, dash="dot"))


def _add_eta_lines_ice(fig, eta_list: List[float], lhv_mj_per_l: float = 34.2):
    """L/100km = VDE_NET [MJ/km] / (η_pt * LHV) * 100"""
    import numpy as np
    xs = np.linspace(0.2, 1.2, 40)
    for eta in eta_list:
        ys = xs / max(eta, 1e-9) / max(lhv_mj_per_l, 1e-9) * 100.0
        fig.add_scatter(x=xs, y=ys, mode="lines", name=f"η_pt={eta:.2f}", line=dict(width=1, dash="dot"))


# =============================================================================
# 2) Seções principais (cards)
# =============================================================================
def section_parameters_card(vde_id: int, electrification: str, results: Dict[str, float]):
    bev = (electrification or "").upper() == "BEV"
    if bev:
        yhat = {
            "urb_Wh_per_km":  results.get("energy_city_Wh_per_km"),
            "hw_Wh_per_km":   results.get("energy_highway_Wh_per_km"),
            "comb_Wh_per_km": results.get("energy_combined_Wh_per_km"),
        }
    else:
        yhat = {
            "urb_l_per_100km":  results.get("fuel_city_l_per_100km"),
            "hw_l_per_100km":   results.get("fuel_highway_l_per_100km"),
            "comb_l_per_100km": results.get("fuel_combined_l_per_100km"),
        }

    if st.button("💾 Save (Parameters)", use_container_width=True, key="btn_save_params"):


        payload = build_min_payload(vde_id, electrification, yhat, method_note="Direct parameters")
        payload = enrich_with_derivatives(payload, electrification, fuel_type="Gasoline")
        payload = _apply_scenario_extras(payload)
        payload = filter_payload(payload)

        st.caption("Payload → fuelcons_db")
        st.json(payload)

        insert_fuelcons(payload)
        st.success("Scenario saved (parameters).")

def section_regression_card(vde_id: int, electrification: str, filters: Dict[str, Any], vde_net: float):
    st.subheader("Regression (aligned with filters above)")

    regdf = load_regression_dataset(filters, current_vde_id=vde_id)
    model = fit_regression_y_vs_vde(regdf, y_col=None, electrification=electrification)
    yhat  = predict_current_consumption(model, vde_net, electrification)

    c1, c2 = st.columns(2)
    with c1: st.write("Model (Urban):",  model.get("urb"))
    with c2: st.write("Model (Highway):", model.get("hw"))
    st.write("Estimate for current snapshot:", yhat)

    if st.button("💾 Save (Regression)", use_container_width=True, key="btn_save_regression"):
        payload = build_min_payload(vde_id, electrification, yhat, method_note="EPA/WLTP regression (split)")
        payload = enrich_with_derivatives(payload, electrification, fuel_type="Gasoline")  # ajuste fuel_type se for o caso
        payload = _apply_scenario_extras(payload)  # seu hook
        payload = filter_payload(payload)

        st.caption("Payload → fuelcons_db")
        st.json(payload)  # preview transparente

        insert_fuelcons(payload)
        st.success("Scenario saved (regression).")

    return model



def sidebar_vde_selector_and_context2() -> Tuple[Optional[int], Dict[str, Any]]:
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
    eta_pt = None
    fuel_type = None
    lhv_override = None
    uf_phev = None
    eta_drive = None
    grid = None
    
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

    st.sidebar.divider()
    st.sidebar.subheader("Calculation Parameters")

    if electrification in ("ICE", "HEV", "PHEV"):
        
        st.sidebar.markdown("**Parameters for ICE / MHEV / HEV / PHEV**")
        # ICE / híbridos
        eta_pt = st.sidebar.number_input("η_pt (ICE/MHEV/HEV/PHEV)", min_value=0.0, max_value=1.0, placeholder="0.35",
                                        step=0.005, format="%.3f", key="sb_eta_pt")
        fuel_type = st.sidebar.selectbox("Fuel type", ["Gasoline", "E10", "E22", "E100", "Diesel", "Other"],
                                        key="sb_fuel_type")
        lhv_override = st.sidebar.number_input("LHV [MJ/L] (override opcional)", min_value=0.0, step=0.1, placeholder="34.2",
                                            format="%.2f", key="sb_lhv_override")
        # PHEV
        uf_phev = st.sidebar.number_input("UF PHEV (0–1)", min_value=0.0, max_value=1.0, step=0.01, placeholder="0.5",
                                        format="%.2f", key="sb_uf")
    else:    
        # BEV / PHEV elétrico
        st.session_state["sb_uf"] = 0.50 if electrification == "PHEV" else 1.0
        # (opcional) zere BEV
        st.session_state["sb_eta_drive"] = 0.88
        st.session_state["sb_grid"] = 400.0
        eta_drive = st.sidebar.number_input("Driveline efficiency (BEV/PHEV elétrico)", min_value=0.0, max_value=1.0, placeholder="0.88",
                                            step=0.005, format="%.3f", key="sb_eta_drive")
        grid = st.sidebar.number_input("Grid [gCO₂/kWh]", min_value=0.0, step=1.0, format="%.0f", key="sb_grid", placeholder="400")

    # Extras do cenário (armazenados no fuelcons_db)
    st.sidebar.divider()
    st.sidebar.subheader("Scenario Extras (fuelcons_db)")
    st.sidebar.number_input("Gears (scenario)", min_value=0, step=1, key="pwt_gears", placeholder="6")
    st.sidebar.number_input("Final drive ratio (scenario)", min_value=0.0, step=0.01, placeholder="3.91",
                            format="%.2f", key="pwt_fdr")
    # --- Transmission model picker (salva em vde_db) ---
    trans_models = fetch_distinct_transmission_models()

    # adiciona opção manual no final
    trans_models.append("Other...")

    choice = st.sidebar.selectbox(
        "Transmission model (scenario)",
        trans_models,
        key="pwt_trans_model_choice"
    )

    if choice == "Other...":
        tm_value = st.sidebar.text_input("Type transmission model", key="pwt_trans_model_custom")
    else:
        tm_value = choice

    # guarda no session_state
    st.session_state["pwt_trans_model"] = tm_value


    # Filtros e uso de regressão
    st.sidebar.divider()
    use_regression = st.sidebar.checkbox("Use linear regression vs VDE (EPA)", value=False, key="sb_use_reg")
    reg_filters = {}
    if use_regression:
        reg_filters["electrification"] = electrification  # já vem do seletor principal
        # --- carregar opções do DB ---
        try:
            cat_rows  = fetchall("SELECT DISTINCT category FROM vde_db WHERE category IS NOT NULL AND category <> '' ORDER BY category;")
            make_rows = fetchall("SELECT DISTINCT make     FROM vde_db WHERE make     IS NOT NULL AND make     <> '' ORDER BY make;")
 
            
            categories = [r["category"] for r in cat_rows] if cat_rows else []
            makes      = [r["make"]     for r in make_rows] if make_rows else []
        except Exception:
            categories, makes = [], []

        # defaults do snapshot atual
        snap_cat = snap_make = None
        if vde_id:
            _r = fetchone("SELECT category, make FROM vde_db WHERE id=?;", (vde_id,)) or {}
            snap_cat  = (_r.get("category") or "").strip() or None
            snap_make = (_r.get("make")     or "").strip() or None

        # opções com "(all)" na frente
        cat_options  = ["(all)"] + categories
        make_options = ["(all)"] + makes

        cat_index  = cat_options.index(snap_cat)  if snap_cat  in cat_options  else 0
        make_index = make_options.index(snap_make) if snap_make in make_options else 0

        cat_choice = st.sidebar.selectbox("Filter: category", cat_options, index=cat_index, key="sb_reg_cat")
        make_choice = st.sidebar.selectbox("Filter: make", make_options, index=make_index, key="sb_reg_make")

        # só grava no filtro se o usuário não deixou "(all)"
        if cat_choice != "(all)":
            reg_filters["category"] = cat_choice
        if make_choice != "(all)":
            reg_filters["make"] = make_choice
            # --- POWER BIN (engine_max_power_kw em fuelcons_db) ---
        # Faixas fixas (ajuste se quiser)
        power_bins = [
            ("(all)", None),
            ("≤150 HP", (0, 150*0.7457)),
            ("151–300 HP", ((150+1)*0.7457, 300*0.7457)),
            ("301–500 HP", ((300+1)*0.7457, 500*0.7457)),
            ("501–700 HP", ((500+1)*0.7457, 700*0.7457)),
            (">700 HP", ((700+1)*0.7457, None)),
        ]


        p_labels = [lbl for (lbl, _) in power_bins]
        p_choice = st.sidebar.selectbox("Filter: engine max power", p_labels, index=0, key="sb_reg_pbin")

        # guarda no filtro apenas se não for (all)
        if p_choice != "(all)":
            # salva o intervalo como tupla (min, max) para usar na query
            p_range = dict(power_bins)[p_choice]
            reg_filters["power_kw_range"] = p_range  # ex.: (121, 150) ou (251, None)

        st.sidebar.checkbox("Include category neighbors", value=False, key="sb_reg_neighbors")
        st.sidebar.checkbox("Remove outliers (IQR)", value=True, key="sb_reg_rm_outliers")

    ctx: Dict[str, Any] = {
        "electrification": electrification,
        "apply_bev_placeholders": apply_bev_placeholders,
        "eta_trans": eta_trans,
        "trans_ABC": trans_ABC,
        "eta_pt": eta_pt,
        "eta_drive": eta_drive,
        "grid_gco2_per_kwh": grid,
        "uf_phev": uf_phev,
        "fuel_type": fuel_type,
        "lhv_override": lhv_override or None,
        "use_regression": use_regression,
        "filters": reg_filters,
    }
    return vde_id, ctx


def calculation_parameters_block(electrification: str) -> None:
        st.header("Parameters-based estimation")
        eta_pt = None
        fuel_type = None
        lhv_override = None
        uf_phev = None
        eta_drive = None
        grid = None

        if electrification in ("ICE", "HEV", "PHEV"):
            
            st.markdown("**Parameters for ICE / MHEV / HEV / PHEV**")
            c1,c2,c3,c4 = st.columns (4)
            # ICE / híbridos
    
            eta_pt = c1.number_input("η_pt (ICE/MHEV/HEV/PHEV)", min_value=0.0, max_value=1.0, placeholder="0.35",
                                            step=0.005, format="%.3f", key="sb_eta_pt")
            fuel_type = c2.selectbox("Fuel type", ["Gasoline", "E10", "E22", "E100", "Diesel", "Other"],
                                            key="sb_fuel_type")
            lhv_override = c3.number_input("LHV [MJ/L] (override opcional)", min_value=0.0, step=0.1, placeholder="34.2",
                                                format="%.2f", key="sb_lhv_override")
            # PHEV
            uf_phev = c4.number_input("UF PHEV (0–1)", min_value=0.0, max_value=1.0, step=0.01, placeholder="0.5",
                                            format="%.2f", key="sb_uf")
        else:    
            # BEV / PHEV elétrico
            st.session_state["sb_uf"] = 0.50 if electrification == "PHEV" else 1.0
            # (opcional) zere BEV
            st.session_state["sb_eta_drive"] = 0.88
            st.session_state["sb_grid"] = 400.0
            eta_drive = st.number_input("Driveline efficiency (BEV/PHEV elétrico)", min_value=0.0, max_value=1.0, placeholder="0.88",
                                                step=0.005, format="%.3f", key="sb_eta_drive")
            grid = st.number_input("Grid [gCO₂/kWh]", min_value=0.0, step=1.0, format="%.0f", key="sb_grid", placeholder="400")

        # Extras do cenário (armazenados no fuelcons_db)
        st.divider()
        st.subheader("Scenario Extras (fuelcons_db)")
        c1,c2,c3 = st.columns (3)
        c1.number_input("Gears (scenario)", min_value=0, step=1, key="pwt_gears", placeholder="6")
        c2.number_input("Final drive ratio (scenario)", min_value=0.0, step=0.01, placeholder="3.91",
                                format="%.2f", key="pwt_fdr")
        # --- Transmission model picker (salva em vde_db) ---
        trans_models = fetch_distinct_transmission_models()

        # adiciona opção manual no final
        trans_models.append("Other...")

        choice = c3.selectbox(
            "Transmission model (scenario)",
            trans_models,
            key="pwt_trans_model_choice"
        )

        if choice == "Other...":
            tm_value = st.text_input("Type transmission model", key="pwt_trans_model_custom")
        else:
            tm_value = choice

        # guarda no session_state
        st.session_state["pwt_trans_model"] = tm_value
        

        ctx: Dict[str, Any] = {
        "eta_pt": eta_pt,
        "eta_drive": eta_drive,
        "grid_gco2_per_kwh": grid,
        "uf_phev": uf_phev,
        "fuel_type": fuel_type,
        "lhv_override": lhv_override or None,
        }

        return ctx
    

def build_payload_from_regression(yhat: Dict[str, Any], model: Dict[str, Any], vde_id: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"vde_id": vde_id, "electrification": ctx.get("electrification"), "method_note": "EPA-regression"}
    payload.update(yhat)
    return drop_empty(payload)


def fit_regression_split_urb_hw(df: pd.DataFrame, electrification: str) -> Dict[str, Any]:
    """Ajusta dois modelos: urb e hw. Retorna dict {'urb': {...}, 'hw': {...}}."""
    if df.empty:
        return {"urb": {"a": None, "b": None, "n": 0, "r2": None},
                "hw":  {"a": None, "b": None, "n": 0, "r2": None}}

    work = _prepare_phase_targets(df, electrification).dropna(subset=["vde_net_mj_per_km"], how="any")

    models = {}
    for label in ["urb", "hw"]:
        if label not in work.columns:
            models[label] = {"a": None, "b": None, "n": 0, "r2": None}
            continue
        sub = work[["vde_net_mj_per_km", f"y_{label}"]].dropna()
        if sub.shape[0] < 3:
            models[label] = {"a": None, "b": None, "n": int(sub.shape[0]), "r2": None}
            continue
        x = sub["vde_net_mj_per_km"].values.astype(float)
        y = sub[f"y_{label}"].values.astype(float)
        X = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(beta[0]), float(beta[1])
        yhat = a + b*x
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - y.mean())**2)) or 1.0
        r2 = 1.0 - ss_res/ss_tot
        models[label] = {"a": a, "b": b, "n": int(sub.shape[0]), "r2": r2, "y_col": f"y_{label}"}

    return models

def fuel_table_filter_toggle() -> str:
    return st.radio("Scenario View:", ["Only for this Vehicle id", "All"], index=0, key="fuel_view_mode")

def render_fuelcons_table2(df: pd.DataFrame):
    if df.empty:
        st.info("Nenhum cenário encontrado.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)

def section_fuel_energy(vde_id: int, ctx: Dict[str, Any]):
    st.subheader("Adicionar/Calcular cenários (grava em FUELCONS_DB)")
    tab_infer, tab_official = st.tabs(["Infer from VDE_NET", "Official / Tested values"]) 

    # --- INFER ---
    with tab_infer:
        electrif = ctx.get("electrification", "ICE")
        eta_pt = ctx.get("eta_pt")
        eta_drive = ctx.get("eta_drive")
        grid = ctx.get("grid_gco2_per_kwh")
        uf_phev = ctx.get("uf_phev")
        lhv_override = ctx.get("lhv_override")
        fuel_type = ctx.get("fuel_type")

        totals = compute_vde_total_from_ctx(fetchone("SELECT * FROM vde_db WHERE id=?;", (vde_id,)) or {}, ctx)
        st.caption(f"VDE_NET usado: {totals['vde_net_mj_per_km']:.3f} MJ/km" + (f" · VDE_TOTAL: {totals['vde_total_mj_per_km']:.3f} MJ/km" if totals.get('vde_total_mj_per_km') else ""))

        # Switch: regressão vs física
        if ctx.get("use_regression") == "Regression":
            st.markdown("**Regression mode (EPA):** estimates fuel consumption from VDE using filtered linear regression. Pick your filters:")

            regdf = load_regression_dataset(ctx.get("filters", {}), current_vde_id=vde_id)
            y_col = "energy_Wh_per_km" if electrif == "BEV" else "fuel_l_per_100km"
            model = fit_regression_y_vs_vde(regdf, y_col=y_col)
            if model.get("n", 0) >= 3:
                yhat = predict_current_consumption(model, totals["vde_net_mj_per_km"], electrif)
                st.write({"model": model, "estimate": yhat})
                payload = build_payload_from_regression(yhat, model, vde_id, ctx)
                if st.button("💾 Save (Regression)", use_container_width=True, key="btn_save_reg"):
                    payload = _apply_scenario_extras(payload)
                    insert_fuelcons(payload)
                    st.success("Regression scenario saved at FUELCONS_DB.")
            else:
                st.info("Unsufficient data for regression (need at least 3 points).")
        else:
            st.markdown("**PWT efficiency mode for estimate** use η_pt/η_drive/UF to estimate.")
            try:
                if electrif == "BEV":
                    res = compute_bev_from_vde(vde_id, driveline_eff=eta_drive or 0.9, grid_gco2_per_kwh=grid or 0.0)
                    payload = drop_empty({
                        **res,
                        "vde_id": vde_id,
                        "electrification": "BEV",
                        "bev_eff_drive": eta_drive or None,
                        "method_note": "MVP formula from VDE_NET",
                    })
                else:
                    res = compute_ice_fuel_from_vde(
                        vde_id,
                        fuel_type=fuel_type,
                        eta_pt=eta_pt or 0.35,
                        lhv_mj_per_l=lhv_override or None,
                        electrification=electrif,
                        uf_phev=uf_phev if electrif == "PHEV" else None,
                        driveline_eff=eta_drive if electrif == "PHEV" else None,
                        grid_gco2_per_kwh=grid if electrif == "PHEV" else None,
                    )
                    payload = drop_empty({
                        **res,
                        "vde_id": vde_id,
                        "electrification": electrif,
                        "fuel_type": fuel_type or None,
                        "eta_pt_est": eta_pt or None,
                        "utility_factor_pct": (uf_phev * 100.0) if (electrif == "PHEV" and uf_phev is not None) else None,
                        "method_note": "MVP formula from VDE_NET",
                    })
                if st.button("💾 Save (Infer)", use_container_width=True, key="btn_save_infer"):
                    payload = _apply_scenario_extras(payload)
                    insert_fuelcons(payload)
                    st.success("Cenário inferido salvo em FUELCONS_DB.")
                    st.write(res)
            except Exception as e:
                st.error(f"Falha no compute/save: {e}")

    # --- OFFICIAL ---
    with tab_official:
        electrif_form = st.selectbox(
            "Electrification (required)", ["ICE", "MHEV", "HEV", "PHEV", "BEV"],
            index=["ICE","MHEV","HEV","PHEV","BEV"].index(ctx.get("electrification", "ICE")),
            key="official_electrification",
        )
        cycle = st.selectbox("Cycle", ["FTP-75", "HWFET", "US06", "WLTC", "NBR6601", "Other"], key="official_cycle")
        procedure = st.text_input("Test procedure / label", key="official_proc")
        source = st.text_input("Source (EPA label / INMETRO / CoC / Internal)", key="official_source")
        test_year = st.number_input("Test year", min_value=1990, max_value=2100, step=1, key="official_year")

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            L_100 = st.number_input("Fuel [L/100km]", min_value=0.0, step=0.01, key="official_L100")
        with d2:
            km_L = st.number_input("Fuel [km/L]", min_value=0.0, step=0.01, key="official_kmL")
        with d3:
            Wh_km = st.number_input("Energy [Wh/km]", min_value=0.0, step=1.0, key="official_Whkm")
        with d4:
            gCO2_km = st.number_input("CO₂ [g/km]", min_value=0.0, step=0.1, key="official_CO2km")

        if st.button("💾 Save Official/Tested", use_container_width=True, key="btn_official"):
            row = drop_empty({
                "vde_id": vde_id,
                "electrification": electrif_form,   # REQUIRED no schema
                # Metadados (cycle/procedure/source/test_year) ainda não existem no schema — guardamos só na UI.
                "fuel_l_per_100km": L_100 or None,
                "fuel_km_per_l": km_L or None,
                "energy_Wh_per_km": Wh_km or None,
                "gco2_per_km": gCO2_km or None,
                "method_note": "Official/Direct entry",
            })
            row = _apply_scenario_extras(row)
            insert_fuelcons(row)
            st.success("Official values saved at FUELCONS_DB.")



def _prepare_phase_targets(df: pd.DataFrame, electrification: str) -> pd.DataFrame:
    """Cria colunas y_urb e y_hw, usando EPA (ftp75/hwfet) se houver; senão WLTP (médias)."""
    if df.empty:
        return df.copy()

    out = df.copy()

    if electrification == "BEV":
        # kWh/100 km → aqui já está em Wh/km no seu schema
        # Preferência EPA
        if {"energy_ftp75_Wh_per_km", "energy_hwfet_Wh_per_km"}.issubset(out.columns):
            out["y_urb"] = out["energy_ftp75_Wh_per_km"]
            out["y_hw"]  = out["energy_hwfet_Wh_per_km"]
        else:
            # Fallback WLTP: média LOW+MID para urbano, HIGH+XHIGH para highway
            low = out.get("energy_low_Wh_per_km")
            mid = out.get("energy_mid_Wh_per_km")
            high = out.get("energy_high_Wh_per_km")
            xhi = out.get("energy_xhigh_Wh_per_km")
            out["y_urb"] = pd.concat([low, mid], axis=1).mean(axis=1, skipna=True)
            out["y_hw"]  = pd.concat([high, xhi], axis=1).mean(axis=1, skipna=True)
    else:
        # ICE / HEV / PHEV — combustíveis em L/100 km
        if {"fuel_ftp75_l_per_100km", "fuel_hwfet_l_per_100km"}.issubset(out.columns):
            out["y_urb"] = out["fuel_ftp75_l_per_100km"]
            out["y_hw"]  = out["fuel_hwfet_l_per_100km"]
        else:
            low = out.get("fuel_low_l_per_100km")
            mid = out.get("fuel_mid_l_per_100km")
            high = out.get("fuel_high_l_per_100km")
            xhi = out.get("fuel_xhigh_l_per_100km")
            out["y_urb"] = pd.concat([low, mid], axis=1).mean(axis=1, skipna=True)
            out["y_hw"]  = pd.concat([high, xhi], axis=1).mean(axis=1, skipna=True)

    return out



def build_payload_from_physics(res: Dict[str, Any], vde_id: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"vde_id": vde_id, "electrification": ctx.get("electrification"), "method_note": "MVP formula from VDE_NET"}
    payload.update(res)
    # extras típicos
    if ctx.get("electrification") == "BEV":
        payload["bev_eff_drive"] = ctx.get("eta_drive") or None
    else:
        payload["eta_pt_est"] = ctx.get("eta_pt") or None
        if ctx.get("electrification") == "PHEV" and ctx.get("uf_phev") is not None:
            payload["utility_factor_pct"] = ctx["uf_phev"]*100.0
    return drop_empty(payload)

# =============================================================================
# 3) Main
# =============================================================================

def main():
    st.title("EcoDrive Analyst · Página 2 — Redux")

    # Sidebar slim
    vde_id, ctx = _sidebar_minimal()
    if not vde_id:
        st.stop()

    # BEV placeholders (opcional)
    apply_bev_placeholders_if_needed(vde_id, ctx["electrification"])

    # Header
    vde_row = fetchone("SELECT * FROM vde_db WHERE id=?;", (vde_id,)) or {}
    fixed_header(vde_row)

    # Filters (únicos e compartilhados)
    filters = _filters_bar(vde_id, ctx["electrification"])

    # Consulta para tabela E para gráfico — mesmo conjunto (se quiser, reuse uma única função)
    if "vde_id" in filters:
        df_fuel = fetch_fuelcons_by_vde(vde_id)  # <- sua função
    else:
        df_fuel = fetch_fuelcons_all(filters)    # <- sua função (aceita elec/cat/make/power/vde_id)

    # Construção do DF para scatter a partir do mesmo conjunto (simples: use df_fuel + join de vde se necessário)
    df_plot = build_scatter_from_fuel(df_fuel)

    # Cálculo de VDE_NET/TOTAL atual (para previsão)
    totals = _compute_vde_total_from_ctx(vde_row, ctx)
    vde_net = totals["vde_net_mj_per_km"]

    # Gráfico no topo com overlays
    st.markdown("---")
    # regressão: só calculamos o modelo aqui se o modo for Regression (pra desenhar a(s) reta(s))
    model_for_plot = None
    if ctx["analysis_mode"] == "Regression":
        regdf = load_regression_dataset(filters, current_vde_id=vde_id)
        model_for_plot = fit_regression_y_vs_vde(regdf, y_col=None, electrification=ctx["electrification"])

    # Linhas de eficiência padrão para overlay (ajuste à vontade)
    eta_lines = [0.30, 0.35, 0.40, 0.45] if ctx["electrification"] != "BEV" else [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    plot_scatter_with_overlays(df_plot, ctx["electrification"], model_for_plot, eta_lines)

    # Cards conforme modo
    st.markdown("---")
    if ctx["analysis_mode"] == "Parameters":
        section_parameters_card(vde_id, ctx["electrification"])
    else:
        _ = section_regression_card(vde_id, ctx["electrification"], filters, vde_net)

    # Tabela com o MESMO filtro
    st.markdown("---")
    st.subheader("Fuel/Energy scenarios")
    render_fuelcons_table(df_fuel)  # <- sua função

# =============================================================================
# Main
# =============================================================================

def main2():
    st.title("EcoDrive Analyst · Página 2 — PWT & Fuel/Energy (v1.2)")

    # 1) Sidebar: seleção + contexto
    vde_id, ctx = sidebar_vde_selector_and_context()
    if not vde_id:
        st.stop()

    # 2) Aplicar placeholders BEV (opcional)
    apply_bev_placeholders_if_needed(vde_id, ctx["electrification"])

    vde_row = fetchone("SELECT * FROM vde_db WHERE id=?;", (vde_id,)) or {}
    # Cálculo de VDE_NET/TOTAL atual (para previsão)

    totals = compute_vde_total_from_ctx(vde_row, ctx)
    vde_net = totals["vde_net_mj_per_km"]

    # --- ícones automáticos (sem inputs) ---
    logo_path = search_logo(vde_row, base_dir="data/images/logos", fallback="_unknown.png") or ""
    leg_icon  = get_legislation_icon(vde_row, base_dir="data/images") or ""

    # injeta nas chaves que o fixed_header já usa
    vde_row["brand_icon"] = logo_path
    vde_row["leg_icon"]   = leg_icon

    fixed_header(vde_row)

    #if ctx["use_regression"] not in "Regression":
        #calculation_parameters_block(ctx["electrification"])

    # 4) Tabela fuelcons — ver por VDE ou Todos
    st.markdown("---")
    #view_mode = fuel_table_filter_toggle()

    # Filters (únicos e compartilhados)
    st.markdown("View Filters")
    view_filters = filters_bar(vde_id, ctx["electrification"], key_ns="view")
    view_filters["legislation"] = vde_row['legislation']
    ctx.get("view_filters") == view_filters
    st.write(view_filters)

    eta_lines = [0.20, 0.25, 0.30, 0.35] if ctx["electrification"] != "BEV" else [0.85, 0.90, 0.95, 0.98, 1, 1.05]


    # Consulta para tabela E para gráfico — mesmo conjunto (se quiser, reuse uma única função)
    if "vde_id" in view_filters:
        df_fuel_by_vde = fetch_fuelcons_by_vde(vde_id)  
        # Tabela com o MESMO filtro
        st.markdown("---")
        st.subheader("Fuel/Energy scenarios")

        df_fuel = fetch_fuelcons_all(view_filters)
        df_plot = build_scatter_from_fuel(df_fuel)
            # Tabela com o MESMO filtro
        st.markdown("---")
        st.subheader("Fuel/Energy scenarios")
        show_summary_plots(df_plot)
        render_fuelcons_table(df_fuel_by_vde)
         
    else:
        df_fuel = fetch_fuelcons_all(view_filters)    # <- sua função (aceita elec/cat/make/power/vde_id)
        # 6) Gráficos finais
        st.markdown("---")
        #df_plot = build_scatter_data()
        df_plot = build_scatter_from_fuel(df_fuel)     
        regdf_view = load_regression_dataset(view_filters, current_vde_id=vde_id)
        model_for_view = fit_regression_y_vs_vde(regdf_view, y_col=None, electrification=ctx["electrification"])
        
        st.markdown("---")
        st.subheader("Fuel/Energy scenarios")
        plot_scatter_with_overlays(df_plot, ctx["electrification"], None, eta_lines)
        render_fuelcons_table(df_fuel)
    #show_summary_plots(df_plot)
    if ctx["use_regression"] == "Parameters":

        section_parameters_card(vde_id, vde_net, ctx["electrification"])
    else:
        reg_filters = filters_bar(vde_id, ctx["electrification"], key_ns="reg")
        reg_filters["legislation"] = vde_row['legislation']
        ctx.get("reg_filters") == reg_filters
        model_for_plot = section_regression_card(vde_id, ctx["electrification"], reg_filters, vde_net)
        df_fuel_reg = fetch_fuelcons_all(reg_filters)
        df_plot = build_scatter_from_fuel(df_fuel_reg)  
        plot_scatter_with_overlays(df_plot, ctx["electrification"], model_for_plot, eta_lines)

    # Cards conforme modo
    st.markdown("---")



if __name__ == "__main__":
    main()
