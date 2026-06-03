from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from src.vde_app.state import ensure_pwt_sidebar_defaults
from src.vde_core.regression import load_regression_dataset, fit_regression_y_vs_vde, predict_current_consumption
from src.vde_core.pwt_fuel_energy_service import (
    apply_bev_placeholders,
    default_electrification_from_vde,
    delete_fuelcons_row,
    fetch_filter_values,
    fetch_fuelcons_all,
    fetch_fuelcons_by_vde,
    fetch_fuelcons_allowed,
    fetch_distinct_transmission_models,
    save_fuelcons_payload,
    update_fuelcons_payload,
)
from src.vde_core.vde_setup_service import load_baselines_df, to_float
from src.vde_app.derivatives import build_min_payload, enrich_with_derivatives, filter_payload
from src.vde_app.plots import build_scatter_from_fuel, plot_scatter_with_overlays


def _apply_scenario_extras(d: dict) -> dict:
    d = dict(d)
    g = st.session_state.get("pwt_gears")
    f = st.session_state.get("pwt_fdr")
    if g not in (None, ""):
        d["gear_count"] = g
    if f not in (None, ""):
        d["final_drive_ratio"] = f
    return d


def _sidebar_select_vde_id() -> Optional[int]:
    df = load_baselines_df()
    if df.empty:
        st.sidebar.info("No VDE_DB Snapshots. Create one on Page VDE Setup.")
        return None

    opts = (
        df.assign(
            _label=df.apply(
                lambda r: f"#{int(r['id'])} - {r['make']} {r['model']} "
                f"{int(r['year']) if pd.notna(r['year']) else ''} [{r['legislation']}]",
                axis=1,
            )
        )
        .sort_values("id", ascending=False)[["id", "_label"]]
        .values.tolist()
    )
    label_to_id = {label: int(_id) for _id, label in opts}
    choice = st.sidebar.selectbox("VDE Snapshot", list(label_to_id.keys()), key="sb_snap_selector")
    return label_to_id.get(choice)


def render_sidebar_vde_selector_and_context() -> Tuple[Optional[int], Dict[str, Any]]:
    st.sidebar.header("Select your VDE & Parameters (Required)")
    ensure_pwt_sidebar_defaults(st.session_state)

    vde_id = _sidebar_select_vde_id()
    eta_trans = None
    trans_abc = None

    electrif_default = default_electrification_from_vde(vde_id)
    electrification = st.sidebar.selectbox(
        "Electrification",
        ["ICE", "HEV", "PHEV", "BEV"],
        index=["ICE", "HEV", "PHEV", "BEV"].index(electrif_default),
        key="sb_electrification",
    )

    apply_bev_placeholders_flag = False
    if electrification == "BEV":
        apply_bev_placeholders_flag = st.sidebar.checkbox(
            "Apply BEV Place Holdes (engine_size=0.001 etc.)",
            value=True,
            key="sb_bev_placeholders",
        )

    if vde_id:
        st.sidebar.divider()
        st.sidebar.subheader("Transmission model for VDE_TOTAL")
        trans_mode = st.sidebar.radio(
            "Pick a Mode",
            [
                "Use transmission global efficiency (eta_trans)",
                "Set transmission drag coefs as velocity function (kph) A/B/C",
            ],
            index=0,
            key="sb_trans_mode",
        )
        if trans_mode == "Use transmission global efficiency (eta_trans)":
            eta_trans = st.sidebar.number_input(
                "eta_trans (0-1)",
                min_value=0.0,
                max_value=1.0,
                placeholder="0.9",
                step=0.005,
                format="%.3f",
                key="sb_eta_trans",
            )
        else:
            c1, c2, c3 = st.sidebar.columns(3)
            with c1:
                A = st.number_input("A_trans [N]", min_value=0.0, step=0.1, format="%.2f", key="sb_A_trans")
            with c2:
                B = st.number_input("B_trans [N/kph]", min_value=0.0, step=0.001, format="%.3f", key="sb_B_trans")
            with c3:
                C = st.number_input("C_trans [N/kph^2]", min_value=0.0, step=0.0001, format="%.4f", key="sb_C_trans")
            trans_abc = (A, B, C)

        st.sidebar.caption("Can use defaults by category/transmission_type later (to-do)")

    return vde_id, {
        "electrification": electrification,
        "apply_bev_placeholders": apply_bev_placeholders_flag,
        "eta_trans": eta_trans,
        "trans_ABC": trans_abc,
    }


def apply_bev_placeholders_if_needed(vde_id: int, electrification: str) -> None:
    if electrification != "BEV":
        return
    if not st.session_state.get("sb_bev_placeholders"):
        return

    try:
        apply_bev_placeholders(vde_id)
        st.sidebar.success("Placeholders BEV applied to Snapshot.")
    except Exception as e:
        st.sidebar.warning(f"Could not apply BEV placeholders: {e}")


def filters_bar(vde_id: int, electrification: str, key_ns: str = "fb") -> Dict[str, Any]:
    k = lambda name: f"{key_ns}_{name}"
    st.markdown("### Filters")
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1])

    filter_values = fetch_filter_values()
    cats = filter_values["categories"]
    makes = filter_values["makes"]
    elecs = filter_values["electrifications"]

    with c1:
        view_scope = st.selectbox("View", ["Only this Vehicle id", "All"], index=1, key=k("fl_scope"))
    with c2:
        elec_choice = st.selectbox(
            "Electrification",
            ["(all)", f"(current: {electrification})"] + [e for e in elecs if e != electrification],
            key=k("fl_elec"),
        )
    with c3:
        cat_choice = st.selectbox("Category", ["(all)"] + cats, key=k("fl_cat"))
    with c4:
        make_choice = st.selectbox("Make", ["(all)"] + makes, key=k("fl_make"))
    with c5:
        p_choice = st.selectbox("Power (hp)", ["(all)", "<=150 HP", "151-300 HP", "301-500 HP", "501-700 HP", ">700 HP"], key=k("fl_pbin"))

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

    hp_to_kw = lambda hp: float(hp) / 1.34102209
    pmap = {
        "<=150 HP": (None, 150),
        "151-300 HP": (151, 300),
        "301-500 HP": (301, 500),
        "501-700 HP": (501, 700),
        ">700 HP": (701, None),
    }

    if p_choice in pmap:
        lo_hp, hi_hp = pmap[p_choice]
        lo_kw = hp_to_kw(lo_hp) if lo_hp is not None else None
        hi_kw = hp_to_kw(hi_hp) if hi_hp is not None else None
        filters["power_kw_range"] = (lo_kw, hi_kw)

    return filters
def render_fuelcons_table(df: pd.DataFrame, editable: bool = False) -> None:
    if df is None or df.empty:
        st.info("No scenarios.")
        return

    show_cols = [
        c
        for c in [
            "id",
            "vde_id",
            "electrification",
            "fuel_l_per_100km",
            "energy_Wh_per_km",
            "fuel_ftp75_l_per_100km",
            "fuel_hwfet_l_per_100km",
            "energy_ftp75_Wh_per_km",
            "energy_hwfet_Wh_per_km",
            "method_note",
            "created_at",
        ]
        if c in df.columns
    ]
    st.dataframe(df[show_cols].sort_values("id", ascending=False), use_container_width=True)

    if not editable:
        return

    allowed = set(fetch_fuelcons_allowed())
    st.markdown("#### Edit / Delete")
    for _, row in df.sort_values("id", ascending=False).iterrows():
        rid = int(row["id"])
        title_value = row.get("fuel_l_per_100km") or row.get("energy_Wh_per_km")
        with st.expander(f"#{rid} · {row.get('electrification', '?')} · y={title_value}", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.1, 1.1, 1.1, 2, 1.2])
            elec = c1.selectbox(
                "Electrification",
                ["ICE", "MHEV", "HEV", "PHEV", "BEV"],
                index=["ICE", "MHEV", "HEV", "PHEV", "BEV"].index(str(row.get("electrification", "ICE"))),
                key=f"fc_elec_{rid}",
            )
            f_comb = c2.number_input("fuel L/100km", value=float(row["fuel_l_per_100km"]) if pd.notna(row.get("fuel_l_per_100km")) else 0.0, step=0.01, format="%.2f", key=f"fc_fcomb_{rid}")
            e_comb = c3.number_input("energy Wh/km", value=float(row["energy_Wh_per_km"]) if pd.notna(row.get("energy_Wh_per_km")) else 0.0, step=1.0, format="%.0f", key=f"fc_ecomb_{rid}")
            f_ftp = c4.number_input("FTP-75 L/100", value=float(row["fuel_ftp75_l_per_100km"]) if pd.notna(row.get("fuel_ftp75_l_per_100km")) else 0.0, step=0.01, format="%.2f", key=f"fc_fftp_{rid}")
            e_ftp = c5.number_input("FTP-75 Wh/km", value=float(row["energy_ftp75_Wh_per_km"]) if pd.notna(row.get("energy_ftp75_Wh_per_km")) else 0.0, step=1.0, format="%.0f", key=f"fc_eftp_{rid}")
            note = c6.text_input("Note", value=str(row.get("method_note") or ""), key=f"fc_note_{rid}")

            c7, c8, c9 = st.columns([1, 1, 6])
            f_hwy = c7.number_input("HWFET L/100", value=float(row["fuel_hwfet_l_per_100km"]) if pd.notna(row.get("fuel_hwfet_l_per_100km")) else 0.0, step=0.01, format="%.2f", key=f"fc_fhwy_{rid}")
            e_hwy = c8.number_input("HWFET Wh/km", value=float(row["energy_hwfet_Wh_per_km"]) if pd.notna(row.get("energy_hwfet_Wh_per_km")) else 0.0, step=1.0, format="%.0f", key=f"fc_ehwy_{rid}")
            st.caption("Fill only the fields you want to change; empty values do not overwrite.")

            a1, a2, a3 = st.columns([1.1, 1.1, 6])
            if a1.button("Save", key=f"fc_save_{rid}"):
                payload = {}
                if elec:
                    payload["electrification"] = elec
                if f_comb and f_comb > 0:
                    payload["fuel_l_per_100km"] = float(f_comb)
                if e_comb and e_comb > 0:
                    payload["energy_Wh_per_km"] = float(e_comb)
                if f_ftp and f_ftp > 0:
                    payload["fuel_ftp75_l_per_100km"] = float(f_ftp)
                if e_ftp and e_ftp > 0:
                    payload["energy_ftp75_Wh_per_km"] = float(e_ftp)
                if f_hwy and f_hwy > 0:
                    payload["fuel_hwfet_l_per_100km"] = float(f_hwy)
                if e_hwy and e_hwy > 0:
                    payload["energy_hwfet_Wh_per_km"] = float(e_hwy)
                if note is not None:
                    payload["method_note"] = note

                payload = {k: v for k, v in payload.items() if k in allowed}
                if payload:
                    try:
                        update_fuelcons_payload(rid, payload)
                        st.success("Saved.")
                    except Exception as e:
                        st.error(f"Update failed: {e}")
                else:
                    st.info("Nothing to save.")

            confirm_key = f"fc_confirm_{rid}"
            if a2.button("Delete", key=f"fc_del_{rid}"):
                st.session_state[confirm_key] = True

            if st.session_state.get(confirm_key):
                b1, b2 = st.columns([1, 6])
                b1.warning("Confirm delete?")
                if b1.button("Confirm", key=f"fc_del_ok_{rid}"):
                    try:
                        delete_fuelcons_row(rid)
                        st.success("Deleted.")
                        st.session_state.pop(confirm_key, None)
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                if b2.button("Cancel", key=f"fc_del_cancel_{rid}"):
                    st.session_state.pop(confirm_key, None)


def fixed_header(vde_row: dict):
    st.markdown("### Baseline selected")
    i1, i2, i3, i4 = st.columns([1, 1, 4, 2])

    with i1:
        brand_icon = vde_row.get("brand_icon") or vde_row.get("brand_logo")
        if brand_icon:
            st.image(brand_icon, width=64, caption=vde_row.get("make"))
    with i2:
        leg_icon = vde_row.get("leg_icon") or vde_row.get("legislation_icon")
        if leg_icon:
            st.image(leg_icon, width=64, caption=vde_row.get("legislation"))
    with i3:
        title = f"**{vde_row.get('make','?')} {vde_row.get('model','?')}**"
        subtitle = f"{vde_row.get('year','?')} · {vde_row.get('category','?')} · {vde_row.get('legislation','?')}"
        st.markdown(f"{title}\n\n{subtitle}")
        vde = to_float(vde_row.get("vde_net_mj_per_km"))
        if vde is not None:
            st.caption(f"VDE_NET: {vde:.3f} MJ/km")
    with i4:
        mass = to_float(vde_row.get("mass_kg"))
        cda = to_float(vde_row.get("cda_m2"))
        st.metric("Mass [kg]", f"{mass:.0f}" if mass is not None else "—")
        st.metric("CdA [m²]", f"{cda:.3f}" if cda is not None else "—")


def section_parameters_card(vde_id: int, vde_net_mj_per_km: float, electrification: str) -> Dict[str, Any]:
    st.header("Parameters-based estimation")
    eta_pt = fuel_type = lhv_override = uf_phev = eta_drive = grid = None

    if electrification in ("ICE", "HEV", "PHEV"):
        st.markdown("**Parameters for ICE / MHEV / HEV / PHEV**")
        c1, c2, c3, c4 = st.columns(4)
        eta_pt = c1.number_input("η_pt (ICE/MHEV/HEV/PHEV)", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="sb_eta_pt")
        fuel_type = c2.selectbox("Fuel type", ["Gasoline", "E10", "E22", "E100", "Diesel", "Other"], key="sb_fuel_type")
        lhv_override = c3.number_input("LHV [MJ/L] (override opcional)", min_value=0.0, step=0.1, format="%.2f", key="sb_lhv_override")
        uf_phev = c4.number_input("UF PHEV (0-1)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="sb_uf")
    else:
        if electrification == "PHEV":
            st.session_state["sb_uf"] = 0.50
        eta_drive = st.number_input("Driveline efficiency (BEV/PHEV elétrico)", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="sb_eta_drive")
        grid = st.number_input("Grid [gCO₂/kWh]", min_value=0.0, step=1.0, format="%.0f", key="sb_grid")

    st.divider()
    st.subheader("Scenario Extras (fuelcons_db)")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Gears (scenario)", min_value=0, step=1, key="pwt_gears", placeholder="6")
    c2.number_input("Final drive ratio (scenario)", min_value=0.0, step=0.01, format="%.2f", key="pwt_fdr", placeholder="3.91")
    trans_models = fetch_distinct_transmission_models()
    trans_models.append("Other...")
    choice = c3.selectbox("Transmission model (scenario)", trans_models, key="pwt_trans_model_choice")
    tm_value = st.text_input("Type transmission model", key="pwt_trans_model_custom") if choice == "Other..." else choice
    st.session_state["pwt_trans_model"] = (tm_value or "").strip() or None

    def _lhv_default(ft: str) -> float:
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
        if eta_drive and eta_drive > 0:
            e_elec = (vde_net_mj_per_km / eta_drive) * 277.7778
            yhat["energy_Wh_per_km"] = uf * e_elec
        LHV = float(lhv_override) if (lhv_override and lhv_override > 0) else _lhv_default(fuel_type or "Gasoline")
        if eta_pt and eta_pt > 0 and LHV > 0:
            l100_fuel = (vde_net_mj_per_km / eta_pt) / LHV * 100.0
            yhat["fuel_l_per_100km"] = (1.0 - uf) * l100_fuel

    st.caption("Preview (parameters-based yhat):")
    st.write(yhat)

    payload = {"vde_id": vde_id, "electrification": electrification, "method_note": "Parameters-based estimation"}
    for k in ("energy_Wh_per_km", "fuel_l_per_100km"):
        if yhat.get(k) is not None:
            payload[k] = float(yhat[k])
    payload = enrich_with_derivatives(payload, electrification, fuel_type=fuel_type or "Gasoline")
    payload = _apply_scenario_extras(payload)
    payload = filter_payload(payload)
    st.write(payload)

    csave1, csave2 = st.columns([1, 3])
    with csave1:
        if st.button("💾 Save (Parameters)", use_container_width=True, key="btn_save_params"):
            to_save = payload
            to_save.setdefault("vde_id", vde_id)
            to_save.setdefault("electrification", electrification)
            to_save.setdefault("method_note", "Parameters-based estimation")
            save_fuelcons_payload(to_save)
            st.success("Scenario saved (parameters).")

    return {
        "eta_pt": eta_pt,
        "eta_drive": eta_drive,
        "grid_gco2_per_kwh": grid,
        "uf_phev": uf_phev,
        "fuel_type": fuel_type,
        "lhv_override": lhv_override or None,
    }


def section_regression_card(vde_id: int, electrification: str, filters: Dict[str, Any], vde_net: float):
    st.divider()
    st.subheader("Scenario Extras (fuelcons_db)")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Gears (scenario)", min_value=0, step=1, key="pwt_gears", placeholder="6")
    c2.number_input("Final drive ratio (scenario)", min_value=0.0, step=0.01, format="%.2f", key="pwt_fdr", placeholder="3.91")
    trans_models = fetch_distinct_transmission_models()
    trans_models.append("Other...")
    choice = c3.selectbox("Transmission model (scenario)", trans_models, key="pwt_trans_model_choice")
    tm_value = st.text_input("Type transmission model", key="pwt_trans_model_custom") if choice == "Other..." else choice
    st.session_state["pwt_trans_model"] = (tm_value or "").strip() or None

    st.subheader("Regression (aligned with filters above)")
    regdf = load_regression_dataset(filters, current_vde_id=vde_id)
    model = fit_regression_y_vs_vde(regdf, y_col=None, electrification=electrification)
    yhat = predict_current_consumption(model, vde_net, electrification)
    payload = build_min_payload(vde_id, electrification, yhat, method_note="EPA/WLTP regression (split urb/hw)")
    payload = enrich_with_derivatives(payload, electrification, fuel_type="Gasoline")
    payload = _apply_scenario_extras(payload)
    payload = filter_payload(payload)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Model (Urban):", model.get("urb"))
    with col2:
        st.write("Model (Highway):", model.get("hw"))
    with col3:
        st.write("Model (Combined):", model.get("combined"))

    st.write("Estimate for current snapshot:", yhat)
    st.write(payload)

    if st.button("💾 Save (Regression)", use_container_width=True, key="btn_save_regression"):
        save_fuelcons_payload(payload)
        st.success("Scenario saved (regression).")

    return model


def run_view_panel(vde_id: int, vde_row: dict, ctx: dict) -> None:
    st.subheader("View Filters")
    view_filters = filters_bar(vde_id, ctx["electrification"], key_ns="view")
    view_filters["legislation"] = vde_row.get("legislation")
    ctx["view_filters"] = view_filters

    eta_lines = [0.20, 0.25, 0.30, 0.35] if ctx["electrification"] != "BEV" else [0.85, 0.90, 0.95]
    if "vde_id" in view_filters:
        df_fuel = fetch_fuelcons_all(view_filters)
        df_fuel_table = fetch_fuelcons_by_vde(vde_id)
    else:
        df_fuel = fetch_fuelcons_all(view_filters)
        df_fuel_table = df_fuel

    st.markdown("---")
    st.subheader("Fuel/Energy scenarios")
    df_plot = build_scatter_from_fuel(df_fuel)
    plot_scatter_with_overlays(df_plot, ctx["electrification"], model=None, eta_lines=eta_lines)
    render_fuelcons_table(df_fuel_table, editable=True)


def run_regression_panel(vde_id: int, vde_row: dict, ctx: dict, vde_net: float) -> None:
    st.subheader("Regression Filters")
    reg_filters = filters_bar(vde_id, ctx["electrification"], key_ns="reg")
    reg_filters["legislation"] = vde_row.get("legislation")
    ctx["reg_filters"] = reg_filters
    eta_lines = [0.20, 0.25, 0.30, 0.35] if ctx["electrification"] != "BEV" else [0.85, 0.90, 0.95, 0.98, 1, 1.05]
    regdf = load_regression_dataset(reg_filters, current_vde_id=vde_id)
    model = fit_regression_y_vs_vde(regdf, y_col=None, electrification=ctx["electrification"])
    df_fuel_reg = fetch_fuelcons_all(reg_filters)
    df_plot = build_scatter_from_fuel(df_fuel_reg)
    plot_scatter_with_overlays(df_plot, ctx["electrification"], model, eta_lines)
    section_regression_card(vde_id, ctx["electrification"], reg_filters, vde_net)
