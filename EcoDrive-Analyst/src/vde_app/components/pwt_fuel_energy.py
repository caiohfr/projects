from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.vde_app.derivatives import build_min_payload, enrich_with_derivatives
from src.vde_app.plots import build_scatter_from_fuel, plot_scatter_with_overlays
from src.vde_core.fuel_energy import GCO2_PER_L, LHV_MJ_PER_L
from src.vde_core.fuel_estimation import (
    FuelEstimateRequest,
    build_fuel_scenario_save_payload,
    run_fuel_estimation,
    save_fuel_estimate_result,
)
from src.vde_core.ml_prediction import describe_ml_prediction_setup
from src.vde_core.nearest_peers import build_peer_analysis_for_request
from src.vde_core.pwt_fuel_energy_service import (
    apply_bev_placeholders,
    build_fuel_estimate_request_from_vde,
    compare_saved_scenario_revision,
    default_electrification_from_vde,
    delete_fuelcons_row,
    fetch_distinct_transmission_models,
    fetch_filter_values,
    fetch_fuelcons_all,
    fetch_fuelcons_allowed,
    fetch_fuelcons_by_vde,
    fetch_vde_row,
    fetch_vde_rows_by_ids,
    resolve_vde_energy_values,
    resolve_vde_source_revision,
    update_fuelcons_payload,
)
from src.vde_core.regression import fit_regression_y_vs_vde, load_regression_dataset, predict_current_consumption
from src.vde_core.vde_setup_service import load_baselines_df, to_float

PWT_ESTIMATION_METHODS = [
    "Manual / Imported",
    "Physics Simple",
    "Regression",
    "ML Prediction",
    "Physics + ML Residual",
    "Map-Based Simulation",
]

SCENARIO_INTENTS = [
    "Baseline",
    "Proposal",
    "Imported / Reference",
]

PWT_DRAFT_RESET_KEYS = [
    "pwt_scenario_name",
    "pwt_scenario_intent",
    "pwt_scenario_electrification",
    "pwt_energy_basis",
    "pwt_energy_basis_label",
    "pwt_bev_draft_placeholders",
    "pwt_setup_method",
    "pwt_manual_fuel_l100",
    "pwt_manual_energy_whkm",
    "pwt_manual_gco2_km",
    "pwt_manual_source",
    "pwt_regression_filters",
    "pwt_gears",
    "pwt_fdr",
    "pwt_trans_model",
    "pwt_trans_model_choice",
    "pwt_trans_model_custom",
    "pwt_common_save_mode",
    "pwt_common_update_target",
    "sb_eta_pt",
    "sb_fuel_type",
    "sb_lhv_override",
    "sb_uf",
    "sb_eta_drive",
    "sb_grid",
]


def _reset_pwt_draft_state_for_vde_change(*, old_vde_id: int | None, new_vde_id: int) -> None:
    for key in PWT_DRAFT_RESET_KEYS:
        st.session_state.pop(key, None)
    st.session_state["pwt_draft_source_vde_id"] = int(new_vde_id)
    st.session_state["pwt_last_source_change_notice"] = (
        f"Powertrain Scenario draft reset after source change: "
        f"{('#' + str(old_vde_id)) if old_vde_id else 'none'} -> #{int(new_vde_id)}."
    )


def _vde_snapshot_options() -> tuple[list[str], dict[str, int]]:
    df = load_baselines_df()
    if df.empty:
        return [], {}

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
    labels = [label for _, label in opts]
    label_to_id = {label: int(_id) for _id, label in opts}
    return labels, label_to_id


def render_active_vde_source_bar() -> tuple[Optional[int], Optional[dict[str, Any]]]:
    labels, label_to_id = _vde_snapshot_options()
    if not labels:
        st.info("No VDE_DB Snapshots. Create one on Page VDE Setup.")
        return None, None

    current_label = st.session_state.get("pwt_active_vde_source")
    if current_label not in label_to_id:
        current_label = labels[0]
        st.session_state["pwt_active_vde_source"] = current_label

    with st.container(border=True):
        st.markdown("### Active VDE Source")
        top_left, top_right = st.columns([3, 2])
        with top_left:
            selected_label = st.selectbox(
                "Active VDE snapshot",
                labels,
                index=labels.index(current_label),
                key="pwt_active_vde_source",
            )
        vde_id = label_to_id[selected_label]
        previous_draft_vde_id = st.session_state.get("pwt_draft_source_vde_id")
        if previous_draft_vde_id not in (None, ""):
            try:
                previous_draft_vde_id = int(previous_draft_vde_id)
            except Exception:
                previous_draft_vde_id = None
        if previous_draft_vde_id is None:
            st.session_state["pwt_draft_source_vde_id"] = int(vde_id)
        elif previous_draft_vde_id != int(vde_id):
            _reset_pwt_draft_state_for_vde_change(old_vde_id=previous_draft_vde_id, new_vde_id=int(vde_id))
        vde_row = fetch_vde_row(vde_id)
        energy_values = resolve_vde_energy_values(vde_row)
        with top_right:
            net_state = "Available" if energy_values["vde_net_mj_per_km"] is not None else "Unavailable"
            st.metric("NET / Transmission", net_state)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ID", f"#{int(vde_id)}")
        c2.metric("Vehicle", f"{vde_row.get('make', '-')} {vde_row.get('model', '-')}".strip())
        c3.metric("Year / Legislation", f"{vde_row.get('year', '-')} | {vde_row.get('legislation', '-')}")
        c4.metric(
            "VDE_TOTAL [MJ/km]",
            f"{energy_values['vde_total_mj_per_km']:.3f}" if energy_values["vde_total_mj_per_km"] is not None else "-",
        )
        c5.metric(
            "VDE_NET [MJ/km]",
            f"{energy_values['vde_net_mj_per_km']:.3f}" if energy_values["vde_net_mj_per_km"] is not None else "-",
        )
        if energy_values["warnings"]:
            st.caption(f"Energy warnings: {', '.join(energy_values['warnings'])}")
        if st.session_state.get("pwt_last_source_change_notice"):
            st.info(st.session_state.pop("pwt_last_source_change_notice"))

    st.session_state["current_vde_id"] = int(vde_id)
    return int(vde_id), vde_row


def resolve_comparison_report_anchor() -> tuple[Optional[int], Optional[dict[str, Any]]]:
    preferred_vde_id = st.session_state.get("current_vde_id")
    saved_df = fetch_fuelcons_all({})
    source_ids: list[int] = []

    if saved_df is not None and not saved_df.empty and "vde_id" in saved_df.columns:
        for value in saved_df["vde_id"].dropna().tolist():
            try:
                source_ids.append(int(value))
            except Exception:
                continue

    if preferred_vde_id not in (None, ""):
        try:
            preferred_vde_id = int(preferred_vde_id)
            if preferred_vde_id in source_ids:
                return preferred_vde_id, fetch_vde_row(preferred_vde_id)
        except Exception:
            pass

    if source_ids:
        anchor_id = source_ids[0]
        return anchor_id, fetch_vde_row(anchor_id)

    baseline_df = load_baselines_df()
    if baseline_df is not None and not baseline_df.empty and "id" in baseline_df.columns:
        try:
            anchor_id = int(baseline_df.sort_values("id", ascending=False).iloc[0]["id"])
            return anchor_id, fetch_vde_row(anchor_id)
        except Exception:
            pass

    return None, None


def _ensure_build_scenario_defaults(vde_id: int, vde_row: dict) -> None:
    default_name = f"{vde_row.get('make', '')} {vde_row.get('model', '')} Fuel Estimate".strip()
    default_electrification = default_electrification_from_vde(vde_id)
    energy_values = resolve_vde_energy_values(vde_row)
    default_basis = "VDE_NET" if energy_values["vde_net_mj_per_km"] is not None else "VDE_TOTAL"

    st.session_state.setdefault("pwt_scenario_name", default_name)
    st.session_state.setdefault("pwt_scenario_intent", SCENARIO_INTENTS[1])
    st.session_state.setdefault("pwt_scenario_electrification", default_electrification)
    st.session_state.setdefault("pwt_energy_basis", default_basis)
    st.session_state.setdefault("pwt_bev_draft_placeholders", False)
    st.session_state.setdefault("pwt_setup_method", PWT_ESTIMATION_METHODS[0])

    if st.session_state.get("pwt_scenario_electrification") not in ("ICE", "HEV", "PHEV", "BEV"):
        st.session_state["pwt_scenario_electrification"] = default_electrification
    if energy_values["vde_net_mj_per_km"] is None and st.session_state.get("pwt_energy_basis") == "VDE_NET":
        st.session_state["pwt_energy_basis"] = "VDE_TOTAL"


def get_build_scenario_context(vde_id: int, vde_row: dict) -> Dict[str, Any]:
    _ensure_build_scenario_defaults(vde_id, vde_row)
    energy_values = resolve_vde_energy_values(vde_row)
    energy_basis = str(st.session_state.get("pwt_energy_basis") or "VDE_TOTAL").upper()
    energy_value = energy_values["vde_net_mj_per_km"] if energy_basis == "VDE_NET" else energy_values["vde_total_mj_per_km"]
    return {
        "scenario_name": str(st.session_state.get("pwt_scenario_name") or ""),
        "scenario_intent": str(st.session_state.get("pwt_scenario_intent") or SCENARIO_INTENTS[1]),
        "electrification": str(st.session_state.get("pwt_scenario_electrification") or default_electrification_from_vde(vde_id)).upper(),
        "energy_basis": energy_basis,
        "energy_value_mj_per_km": energy_value,
        "draft_bev_placeholders": bool(st.session_state.get("pwt_bev_draft_placeholders")),
    }


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


def _build_vde_row_lookup(df: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    if df is None or df.empty or "vde_id" not in df.columns:
        return {}
    vde_ids = sorted(
        {
            int(vde_id)
            for vde_id in df["vde_id"].dropna().tolist()
            if str(vde_id).strip()
        }
    )
    if not vde_ids:
        return {}
    rows_df = fetch_vde_rows_by_ids(vde_ids)
    if rows_df.empty:
        return {}
    return {
        int(row["id"]): row.to_dict()
        for _, row in rows_df.iterrows()
        if pd.notna(row.get("id"))
    }


def _resolve_scenario_vde_row(
    row: pd.Series | dict,
    vde_row_lookup: dict[int, dict[str, Any]] | None = None,
    fallback_current_vde_row: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    data = dict(row)
    lookup = dict(vde_row_lookup or {})
    vde_id = data.get("vde_id")
    if vde_id not in (None, ""):
        try:
            return lookup.get(int(vde_id))
        except Exception:
            pass
    return fallback_current_vde_row


def _resolve_scenario_revision_state(
    row: pd.Series | dict,
    vde_row_lookup: dict[int, dict[str, Any]] | None = None,
    fallback_current_vde_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenario_vde_row = _resolve_scenario_vde_row(row, vde_row_lookup, fallback_current_vde_row)
    return compare_saved_scenario_revision(dict(row).get("source_vde_revision"), scenario_vde_row)


def render_fuelcons_table(
    df: pd.DataFrame,
    editable: bool = False,
    current_vde_row: dict | None = None,
    vde_row_lookup: dict[int, dict[str, Any]] | None = None,
) -> None:
    if df is None or df.empty:
        st.info("No scenarios.")
        return

    resolved_lookup = dict(vde_row_lookup or {})
    if current_vde_row is not None and current_vde_row.get("id") not in (None, ""):
        try:
            resolved_lookup.setdefault(int(current_vde_row["id"]), current_vde_row)
        except Exception:
            pass

    if not resolved_lookup and current_vde_row is None and "vde_id" in df.columns:
        candidate_ids = [int(v) for v in df["vde_id"].dropna().unique().tolist() if str(v).strip()]
        if len(candidate_ids) == 1:
            current_vde_row = fetch_vde_row(candidate_ids[0])
            resolved_lookup[candidate_ids[0]] = current_vde_row

    if "source_vde_revision" in df.columns:
        statuses = []
        for _, row in df.iterrows():
            revision_state = _resolve_scenario_revision_state(row, resolved_lookup, current_vde_row)
            statuses.append(_link_state_table_label(revision_state["status"]))
        df = df.copy()
        df["vde_link_state"] = statuses

    show_cols = [
        c
        for c in [
            "id",
            "vde_id",
            "electrification",
            "energy_basis",
            "engine_method",
            "vde_link_state",
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
        with st.expander(f"#{rid} - {row.get('electrification', '?')} - y={title_value}", expanded=False):
            scenario_vde_row = _resolve_scenario_vde_row(row, resolved_lookup, current_vde_row)
            revision_state = _resolve_scenario_revision_state(row, resolved_lookup, current_vde_row)
            _render_link_state_badge(revision_state["status"], context=f"Scenario #{rid}")
            if revision_state["status"] == "changed":
                st.warning(revision_state["message"])
            elif revision_state["status"] == "missing":
                st.info(revision_state["message"])
            elif revision_state["status"] == "current":
                st.caption(revision_state["message"])

            meta1, meta2, meta3, meta4 = st.columns(4)
            source_vde_id = row.get("vde_id")
            meta1.metric("Source VDE", f"#{int(source_vde_id)}" if pd.notna(source_vde_id) else "-")
            meta2.metric("Engine", str(row.get("engine_method") or "-"))
            meta3.metric("Saved VDE rev", str(row.get("source_vde_revision") or "-"))
            meta4.metric("Link state", _link_state_label(revision_state["status"]))
            if scenario_vde_row is not None:
                live_vehicle = f"{scenario_vde_row.get('make', '-')} {scenario_vde_row.get('model', '-')}".strip()
                st.caption(
                    f"Reference live row: {live_vehicle} | rev {resolve_vde_source_revision(scenario_vde_row) or '-'}"
                )

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


def render_scenario_definition_section(vde_id: int, vde_row: dict) -> Dict[str, Any]:
    _ensure_build_scenario_defaults(vde_id, vde_row)
    energy_values = resolve_vde_energy_values(vde_row)
    current_revision = resolve_vde_source_revision(vde_row)

    st.subheader("Context & Energy")
    st.caption("Define the draft scenario identity, override context, and the vehicle energy source used by the estimation engine.")

    read1, read2, read3, read4 = st.columns(4)
    read1.metric("Active VDE", f"#{int(vde_id)}")
    read2.metric("Vehicle", f"{vde_row.get('make', '-')} {vde_row.get('model', '-')}".strip())
    read3.metric("Legislation", str(vde_row.get("legislation") or "-"))
    read4.metric("Source cycle", str(vde_row.get("cycle_name") or "-"))

    src1, src2, src3 = st.columns(3)
    src1.metric("Source revision", str(current_revision or "-"))
    src2.metric("Source status", "Live snapshot")
    src3.metric("Transmission / NET", "Available" if energy_values["vde_net_mj_per_km"] is not None else "Missing / TOTAL only")

    st.text_input("Scenario name", key="pwt_scenario_name")

    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        st.radio(
            "Scenario intent",
            SCENARIO_INTENTS,
            horizontal=True,
            key="pwt_scenario_intent",
        )
    with c2:
        st.selectbox(
            "Electrification override (Powertrain Scenario only)",
            ["ICE", "HEV", "PHEV", "BEV"],
            key="pwt_scenario_electrification",
        )
        st.caption("This override affects only the Powertrain Scenario draft. It does not modify the VDE snapshot.")

    energy_options = ["Use VDE_TOTAL"]
    if energy_values["vde_net_mj_per_km"] is not None:
        energy_options = ["Use VDE_NET (recommended)", "Use VDE_TOTAL"]
        current_basis_label = "Use VDE_NET (recommended)" if st.session_state.get("pwt_energy_basis") == "VDE_NET" else "Use VDE_TOTAL"
    else:
        current_basis_label = "Use VDE_TOTAL"
        st.warning("VDE_NET is unavailable for this VDE source. Use VDE_TOTAL or complete Transmission in VDE Setup.")
    if st.session_state.get("pwt_energy_basis_label") not in energy_options:
        st.session_state["pwt_energy_basis_label"] = current_basis_label

    st.radio(
        "Vehicle energy used for estimation",
        energy_options,
        horizontal=True,
        index=energy_options.index(current_basis_label),
        key="pwt_energy_basis_label",
    )
    st.session_state["pwt_energy_basis"] = "VDE_NET" if st.session_state.get("pwt_energy_basis_label") == "Use VDE_NET (recommended)" else "VDE_TOTAL"

    energy_basis = st.session_state["pwt_energy_basis"]
    selected_value = energy_values["vde_net_mj_per_km"] if energy_basis == "VDE_NET" else energy_values["vde_total_mj_per_km"]
    info1, info2, info3 = st.columns(3)
    info1.metric("Energy path", energy_basis)
    info2.metric("Energy used [MJ/km]", f"{selected_value:.4f}" if selected_value is not None else "-")
    info3.metric("VDE_NET state", "Available" if energy_values["vde_net_mj_per_km"] is not None else "Unavailable")
    st.caption(
        "VDE_NET represents demand without a separate transmission-loss block. "
        "VDE_TOTAL represents the full demand registered on the VDE line."
    )

    if str(st.session_state.get("pwt_scenario_electrification") or "").upper() == "BEV":
        st.checkbox(
            "Use BEV draft placeholders (draft-only)",
            key="pwt_bev_draft_placeholders",
        )
        st.caption("Draft-only placeholders stay in the Powertrain Scenario context and never update the VDE snapshot.")
    else:
        st.session_state["pwt_bev_draft_placeholders"] = False

    return get_build_scenario_context(vde_id, vde_row)


def render_powertrain_inputs_panel(vde_id: int, vde_row: dict) -> Dict[str, Any]:
    ctx = get_build_scenario_context(vde_id, vde_row)
    active_method = str(st.session_state.get("pwt_setup_method") or PWT_ESTIMATION_METHODS[0])

    st.subheader("Powertrain Inputs")
    st.caption("Stage the physical assumptions used by Physics Simple and future hybrid/simulation engines.")
    if active_method != "Physics Simple":
        st.info(
            f"Active estimation method: {active_method}. These inputs remain staged here and will be used when Physics Simple is selected."
        )

    render_powertrain_conversion_inputs(vde_row, ctx)
    _render_scenario_extras_inputs()
    return ctx


def render_estimation_engine_panel(vde_id: int, vde_row: dict) -> Dict[str, Any]:
    ctx = get_build_scenario_context(vde_id, vde_row)
    regression_vde = ctx.get("energy_value_mj_per_km")

    st.subheader("Estimation Engine")
    st.caption("Choose how the current draft will be estimated. Each engine below uses the same Active VDE source and current draft context.")
    method = render_pwt_estimation_method_selector()

    if method == "Manual / Imported":
        render_manual_imported_inputs()
    elif method == "Physics Simple":
        p1, p2, p3 = st.columns(3)
        p1.metric("Energy basis", str(ctx.get("energy_basis") or "-"))
        p2.metric("Energy used [MJ/km]", f"{ctx['energy_value_mj_per_km']:.4f}" if ctx.get("energy_value_mj_per_km") is not None else "-")
        p3.metric("Electrification", str(ctx.get("electrification") or "-"))
        st.caption("Configure eta, LHV, CO2 factors, utility factor, and optional drivetrain metadata in the Powertrain Inputs tab.")
    elif method == "Regression":
        render_regression_inputs(vde_id, vde_row, ctx, regression_vde)
    elif method == "ML Prediction":
        render_ml_prediction_inputs(vde_id, vde_row, ctx, regression_vde)
    else:
        st.warning(f"{method} is planned - engine export/integration pending.")

    return ctx


def render_results_save_tab(vde_id: int, vde_row: dict) -> None:
    ctx = get_build_scenario_context(vde_id, vde_row)
    regression_vde = ctx.get("energy_value_mj_per_km")
    render_fuel_review_save_panel(vde_id, vde_row, ctx, regression_vde)


def render_pwt_estimation_method_selector() -> str:
    current = str(st.session_state.get("pwt_setup_method") or PWT_ESTIMATION_METHODS[0])
    if current not in PWT_ESTIMATION_METHODS:
        current = PWT_ESTIMATION_METHODS[0]
    if "pwt_setup_method" not in st.session_state or st.session_state.get("pwt_setup_method") not in PWT_ESTIMATION_METHODS:
        st.session_state["pwt_setup_method"] = current
    st.radio(
        "Estimation method",
        PWT_ESTIMATION_METHODS,
        horizontal=True,
        index=PWT_ESTIMATION_METHODS.index(current),
        key="pwt_setup_method",
    )
    return str(st.session_state["pwt_setup_method"])


def _default_lhv(fuel_type: str) -> float:
    return float(LHV_MJ_PER_L.get(fuel_type or "Gasoline", LHV_MJ_PER_L["Gasoline"]))


def _default_gco2_per_l(fuel_type: str) -> float:
    return float(GCO2_PER_L.get(fuel_type or "Gasoline", GCO2_PER_L["Gasoline"]))


def _render_scenario_extras_inputs() -> None:
    with st.expander("Vehicle / Drivetrain Data (optional)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Gear count", min_value=0, step=1, key="pwt_gears", placeholder="6")
        c2.number_input("Final drive ratio", min_value=0.0, step=0.01, format="%.2f", key="pwt_fdr", placeholder="3.91")
        trans_models = fetch_distinct_transmission_models()
        trans_models.append("Other...")
        choice = c3.selectbox("Transmission model", trans_models, key="pwt_trans_model_choice")
        tm_value = st.text_input("Custom transmission model", key="pwt_trans_model_custom") if choice == "Other..." else choice
        st.session_state["pwt_trans_model"] = (tm_value or "").strip() or None
        st.caption("Reserved for scenario metadata and future powertrain/performance capability extensions.")


def render_powertrain_conversion_inputs(vde_row: dict, ctx: Dict[str, Any]) -> None:
    st.markdown("#### Physics Simple Inputs")
    electrification = str(ctx.get("electrification") or "ICE").upper()
    energy_basis = str(ctx.get("energy_basis") or "VDE_TOTAL").upper()
    energy_values = resolve_vde_energy_values(vde_row)
    selected_vde = energy_values["vde_total_mj_per_km"] if energy_basis == "VDE_TOTAL" else energy_values["vde_net_mj_per_km"]

    c_info1, c_info2, c_info3 = st.columns(3)
    c_info1.metric("Energy basis", energy_basis)
    c_info2.metric("Selected VDE [MJ/km]", f"{selected_vde:.3f}" if selected_vde is not None else "-")
    c_info3.metric("Electrification", electrification)
    if energy_values["warnings"]:
        st.caption(f"Energy warnings: {', '.join(energy_values['warnings'])}")

    if electrification == "PHEV" and "sb_uf" not in st.session_state:
        st.session_state["sb_uf"] = 0.50

    if electrification in ("ICE", "HEV", "PHEV"):
        st.markdown("**Fuel path parameters**")
        c1, c2, c3 = st.columns(3)
        c1.number_input("eta_pt (fuel path)", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="sb_eta_pt")
        c2.selectbox("Fuel type", ["Gasoline", "E10", "E22", "E100", "Diesel", "Other"], key="sb_fuel_type")
        c3.number_input("LHV [MJ/L] (optional override)", min_value=0.0, step=0.1, format="%.2f", key="sb_lhv_override")
        if electrification == "PHEV":
            st.number_input("Utility factor (0-1)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="sb_uf")

    if electrification in ("BEV", "PHEV"):
        st.markdown("**Electric path parameters**")
        c1, c2 = st.columns(2)
        c1.number_input("Driveline efficiency", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="sb_eta_drive")
        c2.number_input("Grid [gCO2/kWh]", min_value=0.0, step=1.0, format="%.0f", key="sb_grid")


def render_manual_imported_inputs() -> None:
    st.markdown("#### Manual / Imported Inputs")
    st.caption("Register measured, imported or official values without recalculating from VDE energy.")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Fuel [L/100km]", min_value=0.0, step=0.1, format="%.2f", key="pwt_manual_fuel_l100")
    c2.number_input("Energy [Wh/km]", min_value=0.0, step=1.0, format="%.0f", key="pwt_manual_energy_whkm")
    c3.number_input("CO2 [g/km]", min_value=0.0, step=1.0, format="%.1f", key="pwt_manual_gco2_km")
    st.text_input("Source", key="pwt_manual_source", value="user_input")


def _regression_dataset_warnings(regdf: pd.DataFrame) -> list[str]:
    row_count = len(regdf)
    warnings: list[str] = []
    if row_count == 0:
        warnings.append("regression_dataset_empty")
    elif row_count < 10:
        warnings.append("regression_dataset_small")
    return warnings


def _render_regression_dataset_feedback(warnings: list[str], row_count: int) -> None:
    if "regression_dataset_empty" in warnings:
        st.error("No peer records matched the active filters. Adjust the Regression dataset before saving this scenario.")
    elif "regression_dataset_small" in warnings:
        st.warning(f"Small regression dataset: {row_count} records. Review the scatter and model summary before trusting the estimate.")


def _resolve_regression_state(
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    regression_vde: float | None,
    *,
    render_filters: bool,
) -> dict[str, Any]:
    if regression_vde is None:
        return {
            "filters": {},
            "dataset": pd.DataFrame(),
            "model": {},
            "payload": {},
            "warnings": ["regression_energy_missing"],
        }

    if render_filters:
        reg_filters = filters_bar(vde_id, ctx["electrification"], key_ns="regression")
        reg_filters["legislation"] = vde_row.get("legislation")
        st.session_state["pwt_regression_filters"] = dict(reg_filters)
    else:
        reg_filters = dict(st.session_state.get("pwt_regression_filters") or {})
        reg_filters.setdefault("legislation", vde_row.get("legislation"))

    regdf = load_regression_dataset(reg_filters, current_vde_id=vde_id)
    warnings = _regression_dataset_warnings(regdf)
    if regdf.empty:
        return {
            "filters": reg_filters,
            "dataset": regdf,
            "model": {},
            "payload": {},
            "warnings": warnings,
        }

    model = fit_regression_y_vs_vde(regdf, y_col=None, electrification=ctx["electrification"])
    yhat = predict_current_consumption(model, regression_vde, ctx["electrification"])
    payload = build_min_payload(vde_id, ctx["electrification"], yhat, method_note="regression_existing")
    payload = enrich_with_derivatives(payload, ctx["electrification"], fuel_type="Gasoline")
    return {
        "filters": reg_filters,
        "dataset": regdf,
        "model": model,
        "payload": payload,
        "warnings": warnings,
    }


def _build_regression_runner(vde_id: int, electrification: str, filters: Dict[str, Any]):
    def regression_runner(request_dict, vde_mj_per_km):
        regdf = load_regression_dataset(filters, current_vde_id=vde_id)
        model = fit_regression_y_vs_vde(regdf, y_col=None, electrification=electrification)
        yhat = predict_current_consumption(model, vde_mj_per_km, electrification)
        payload = build_min_payload(vde_id, electrification, yhat, method_note="regression_existing")
        payload = enrich_with_derivatives(payload, electrification, fuel_type="Gasoline")
        dataset_warnings = _regression_dataset_warnings(regdf)
        return {
            "fuel_l_100km": to_float(payload.get("fuel_l_per_100km")),
            "energy_Wh_km": to_float(payload.get("energy_Wh_per_km")),
            "gco2_km": to_float(payload.get("gco2_per_km")),
            "fuel_l_per_100km_urb": to_float(payload.get("fuel_ftp75_l_per_100km")),
            "fuel_l_per_100km_hw": to_float(payload.get("fuel_hwfet_l_per_100km")),
            "energy_Wh_km_urb": to_float(payload.get("energy_ftp75_Wh_per_km")),
            "energy_Wh_km_hw": to_float(payload.get("energy_hwfet_Wh_per_km")),
            "gco2_km_urb": to_float(payload.get("gco2_ftp75_per_km")),
            "gco2_km_hw": to_float(payload.get("gco2_hwfet_per_km")),
            "warnings": dataset_warnings,
            "assumptions": {
                "effective_filters": dict(filters),
                "dataset_rows": len(regdf),
                "model_summary": {
                    "urban": model.get("urb"),
                    "highway": model.get("hw"),
                    "combined": model.get("combined"),
                },
            },
            "confidence": "low" if dataset_warnings else "medium",
        }

    return regression_runner


def render_regression_inputs(vde_id: int, vde_row: dict, ctx: Dict[str, Any], regression_vde: float | None) -> None:
    st.markdown("#### Regression Inputs")
    if regression_vde is None:
        st.warning("Regression preview requires VDE energy, but neither VDE_NET nor VDE_TOTAL is available on this snapshot.")
        return
    regression_state = _resolve_regression_state(vde_id, vde_row, ctx, regression_vde, render_filters=True)
    regdf = regression_state["dataset"]
    model = regression_state["model"]
    payload = regression_state["payload"]
    warnings = regression_state["warnings"]
    _render_regression_dataset_feedback(warnings, len(regdf))

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Model (Urban)**")
    c1.write(model.get("urb"))
    c2.markdown("**Model (Highway)**")
    c2.write(model.get("hw"))
    c3.markdown("**Model (Combined)**")
    c3.write(model.get("combined"))
    st.caption(f"Regression dataset rows: {len(regdf)}")

    with st.expander("Regression Preview", expanded=True):
        eta_lines = [0.20, 0.25, 0.30, 0.35] if ctx["electrification"] != "BEV" else [0.85, 0.90, 0.95, 0.98, 1.00, 1.05]
        st.caption("Inline preview uses the same active filters and the same dataset used by the Regression runner.")
        df_plot = build_scatter_from_fuel(regdf)
        plot_scatter_with_overlays(
            df_plot,
            ctx["electrification"],
            model if model else None,
            eta_lines,
            chart_key="pwt_regression_preview",
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Estimated fuel [L/100km]", f"{to_float(payload.get('fuel_l_per_100km')):.2f}" if to_float(payload.get("fuel_l_per_100km")) is not None else "N/A")
        p2.metric("Estimated energy [Wh/km]", f"{to_float(payload.get('energy_Wh_per_km')):.1f}" if to_float(payload.get("energy_Wh_per_km")) is not None else "N/A")
        p3.metric("Estimated CO2 [g/km]", f"{to_float(payload.get('gco2_per_km')):.1f}" if to_float(payload.get("gco2_per_km")) is not None else "N/A")
        phase_urb = payload.get("fuel_ftp75_l_per_100km") or payload.get("energy_ftp75_Wh_per_km")
        phase_hw = payload.get("fuel_hwfet_l_per_100km") or payload.get("energy_hwfet_Wh_per_km")
        if phase_urb is not None or phase_hw is not None:
            s1, s2 = st.columns(2)
            s1.caption(
                "Urban estimate: "
                + (
                    f"{to_float(payload.get('fuel_ftp75_l_per_100km')):.2f} L/100km"
                    if to_float(payload.get("fuel_ftp75_l_per_100km")) is not None
                    else f"{to_float(payload.get('energy_ftp75_Wh_per_km')):.1f} Wh/km"
                )
            )
            s2.caption(
                "Highway estimate: "
                + (
                    f"{to_float(payload.get('fuel_hwfet_l_per_100km')):.2f} L/100km"
                    if to_float(payload.get("fuel_hwfet_l_per_100km")) is not None
                    else f"{to_float(payload.get('energy_hwfet_Wh_per_km')):.1f} Wh/km"
                )
            )


def render_ml_prediction_inputs(vde_id: int, vde_row: dict, ctx: Dict[str, Any], regression_vde: float | None) -> None:
    st.markdown("#### ML Prediction")
    request = _build_active_fuel_estimate_request(vde_id, vde_row, ctx, regression_vde)
    if request is None:
        st.warning("ML preview requires a valid Powertrain Scenario request.")
        return

    setup = describe_ml_prediction_setup(
        request,
        model_artifact_path=st.session_state.get("pwt_ml_artifact_path"),
        predictor=request.model_options.get("ml_predictor"),
    )
    status = str(setup.get("status") or "unknown")

    if status == "export_pending":
        st.info("ML notebook exists, but no exported inference artifact was found.")
    elif status == "artifact_load_failed":
        st.warning("An ML artifact candidate was found, but loading it failed. Review the artifact before using ML Prediction.")
    elif status == "available":
        st.success("ML inference artifact loaded. Preview and save can use the common pipeline.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Artifact status", status.replace("_", " ").title())
    c2.metric("Expected targets", ", ".join(setup.get("targets") or []) or "-")
    c3.metric("Available features", str(len(setup.get("features", {}).get("available_feature_names") or [])))
    c4.metric("Missing features", str(len(setup.get("features", {}).get("missing_features") or [])))
    st.caption(
        "This method reuses the notebook-defined feature family and plugs into the same FuelEstimateResult / save contract used by the other engines."
    )

    with st.expander("ML feature coverage", expanded=False):
        st.write(
            {
                "artifact_path": setup.get("artifact_path"),
                "artifact_candidates": setup.get("artifact_candidates"),
                "available_features": setup.get("features", {}).get("available_features"),
                "missing_features": setup.get("features", {}).get("missing_features"),
                "candidate_models": setup.get("notebook", {}).get("candidate_models"),
                "warnings": setup.get("warnings"),
            }
        )
    peer_analysis = build_peer_analysis_for_request(request, n=5)
    quality = peer_analysis.get("quality") or {}
    summary = peer_analysis.get("summary") or {}
    peer_metrics = pd.DataFrame(summary.get("metrics") or [])
    hints = list(peer_analysis.get("hints") or [])
    guidance_status = "Ready"
    if (setup.get("warnings") or []) or (peer_analysis.get("warnings") or []):
        guidance_status = "Metadata limited"

    st.markdown("#### ML Guidance Preview")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Peer count", str(summary.get("peer_count", 0)))
    g2.metric("Peer quality", str(quality.get("label") or "-"))
    g3.metric("Coverage status", guidance_status)
    g4.metric("Hints triggered", str(len(hints)))
    st.caption("Nearest peers and investigation hints are computed from the current draft context and the saved scenario database.")

    if not peer_metrics.empty:
        compact_metrics = peer_metrics.copy()
        compact_metrics = compact_metrics[compact_metrics["metric"].isin(["fuel_l_per_100km", "gco2_per_km", "energy_Wh_per_km", "vde_total_mj_per_km", "vde_net_mj_per_km"])]
        show_cols = [
            col
            for col in ["label", "median", "std_dev", "min", "max"]
            if col in compact_metrics.columns
        ]
        if show_cols:
            with st.expander("Nearest peers preview", expanded=False):
                st.dataframe(compact_metrics[show_cols], use_container_width=True, hide_index=True)

    if hints:
        with st.expander("Investigation hints preview", expanded=False):
            for hint in hints[:3]:
                st.info(
                    f"{hint.get('hint')}\n\n"
                    f"Evidence: {hint.get('evidence')}\n\n"
                    f"Next data to inspect: {hint.get('next_data')}"
                )


def _fuel_scenario_extra_payload() -> dict:
    payload = {
        "gear_count": st.session_state.get("pwt_gears"),
        "final_drive_ratio": st.session_state.get("pwt_fdr"),
    }
    return {k: v for k, v in payload.items() if v not in (None, "")}


def _build_result_outputs_for_peer_analysis(result) -> dict[str, Any]:
    phase_outputs = dict(result.phase_outputs or {})
    return {
        "fuel_l_100km": result.fuel_l_100km,
        "energy_Wh_km": result.energy_Wh_km,
        "gco2_km": result.gco2_km,
        "fuel_l_per_100km_urb": phase_outputs.get("fuel_ftp75_l_per_100km"),
        "fuel_l_per_100km_hw": phase_outputs.get("fuel_hwfet_l_per_100km"),
        "energy_Wh_km_urb": phase_outputs.get("energy_ftp75_Wh_per_km"),
        "energy_Wh_km_hw": phase_outputs.get("energy_hwfet_Wh_per_km"),
    }


def _render_comparative_guidance(analysis: dict[str, Any], *, title: str = "Comparative Guidance") -> None:
    st.markdown(f"#### {title}")
    quality = analysis.get("quality") or {}
    summary = analysis.get("summary") or {}
    peer_metrics = pd.DataFrame(summary.get("metrics") or [])
    peers_df = pd.DataFrame(analysis.get("peers") or [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Peer count", str(summary.get("peer_count", 0)))
    c2.metric("Peer group quality", str(quality.get("label") or "-"))
    c3.metric("Reason", str(quality.get("reason") or "-"))

    if not peers_df.empty:
        display_cols = [
            col
            for col in [
                "vde_id",
                "make",
                "model",
                "year",
                "category",
                "electrification",
                "fuel_l_per_100km",
                "energy_Wh_per_km",
                "gco2_per_km",
                "vde_total_mj_per_km",
                "vde_net_mj_per_km",
                "peer_similarity",
            ]
            if col in peers_df.columns
        ]
        st.dataframe(peers_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.caption("No comparable saved scenarios were available for peer guidance.")

    if not peer_metrics.empty:
        display_metrics = [
            col
            for col in [
                "label",
                "scenario_value",
                "median",
                "std_dev",
                "delta_vs_median",
                "z_score",
                "min",
                "max",
                "iqr",
            ]
            if col in peer_metrics.columns
        ]
        with st.expander("Peer comparison stats", expanded=False):
            st.dataframe(peer_metrics[display_metrics], use_container_width=True, hide_index=True)

    hints = analysis.get("hints") or []
    st.markdown("#### Investigation Hints")
    if hints:
        for hint in hints:
            st.info(
                f"{hint.get('hint')}\n\n"
                f"Evidence: {hint.get('evidence')}\n\n"
                f"Next data to inspect: {hint.get('next_data')}"
            )
    else:
        st.caption("No investigation hints were triggered from the current peer comparison.")


def _render_phase_outputs_table(phase_outputs: Dict[str, Any]) -> None:
    if not phase_outputs:
        st.caption("No phase outputs were resolved for the active engine.")
        return

    phase_rows = []
    phase_labels = {
        "ftp75": "FTP-75 / Urban",
        "hwfet": "HWFET / Highway",
        "low": "Low",
        "mid": "Mid",
        "high": "High",
        "xhigh": "Extra High",
    }
    for phase_key, phase_label in phase_labels.items():
        phase_rows.append(
            {
                "Phase": phase_label,
                "Fuel [L/100km]": phase_outputs.get(f"fuel_{phase_key}_l_per_100km"),
                "Energy [Wh/km]": phase_outputs.get(f"energy_{phase_key}_Wh_per_km"),
                "CO2 [g/km]": phase_outputs.get(f"gco2_{phase_key}_per_km"),
            }
        )

    phase_df = pd.DataFrame(phase_rows).dropna(how="all", subset=["Fuel [L/100km]", "Energy [Wh/km]", "CO2 [g/km]"])
    if phase_df.empty:
        st.caption("No phase outputs were resolved for the active engine.")
        return
    st.dataframe(phase_df, use_container_width=True, hide_index=True)


def _load_json_blob(raw_value: Any) -> dict[str, Any]:
    if raw_value in (None, ""):
        return {}
    if isinstance(raw_value, dict):
        return dict(raw_value)
    try:
        parsed = json.loads(str(raw_value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _saved_scenario_label(row: pd.Series | dict) -> str:
    data = dict(row)
    scenario_id = int(data.get("id"))
    source_vde_id = data.get("vde_id")
    method = str(data.get("engine_method") or data.get("method_note") or "-")
    created = str(data.get("created_at") or "-")
    source_text = f"VDE #{int(source_vde_id)}" if pd.notna(source_vde_id) else "VDE ?"
    make = str(data.get("make") or "").strip()
    model = str(data.get("model") or "").strip()
    vehicle_text = f"{make} {model}".strip()
    if vehicle_text:
        return f"#{scenario_id} | {source_text} | {vehicle_text} | {method} | {created}"
    return f"#{scenario_id} | {source_text} | {method} | {created}"


def _link_state_label(status: str) -> str:
    labels = {
        "current": "Current",
        "changed": "Refresh required",
        "missing": "Missing provenance",
        "unknown": "Unknown",
    }
    return labels.get(str(status), str(status))


def _link_state_style(status: str) -> tuple[str, str, str]:
    status = str(status)
    mapping = {
        "current": ("Current", "#166534", "#dcfce7"),
        "changed": ("Refresh required", "#9a3412", "#ffedd5"),
        "missing": ("Missing provenance", "#92400e", "#fef3c7"),
        "unknown": ("Unknown", "#374151", "#e5e7eb"),
    }
    return mapping.get(status, (_link_state_label(status), "#374151", "#e5e7eb"))


def _link_state_table_label(status: str) -> str:
    label, _, _ = _link_state_style(status)
    prefixes = {
        "current": "OK",
        "changed": "STALE",
        "missing": "MISSING",
        "unknown": "UNKNOWN",
    }
    return f"{prefixes.get(str(status), 'INFO')} | {label}"


def _render_link_state_badge(status: str, *, context: str | None = None) -> None:
    label, fg, bg = _link_state_style(status)
    body = f"{context}: {label}" if context else label
    st.markdown(
        (
            f"<div style='margin:0.15rem 0 0.5rem 0;'>"
            f"<span style='display:inline-block;padding:0.3rem 0.6rem;border-radius:999px;"
            f"background:{bg};color:{fg};font-size:0.85rem;font-weight:600;'>"
            f"{body}"
            f"</span></div>"
        ),
        unsafe_allow_html=True,
    )


def _scorecard_options(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    options = [_saved_scenario_label(row) for _, row in df.iterrows()]
    label_to_id = {label: int(df.iloc[idx]["id"]) for idx, label in enumerate(options)}
    return options, label_to_id


def _render_scorecard_reference_column(current_vde_row: dict) -> None:
    energy_values = resolve_vde_energy_values(current_vde_row)
    current_revision = current_vde_row.get("updated_at") or current_vde_row.get("created_at") or "-"
    st.markdown("#### Reference VDE")
    st.caption(f"#{int(current_vde_row.get('id'))} | {current_revision}")
    st.metric("Vehicle", f"{current_vde_row.get('make', '-')} {current_vde_row.get('model', '-')}".strip())
    st.metric("Legislation", str(current_vde_row.get("legislation") or "-"))
    st.metric("Category", str(current_vde_row.get("category") or "-"))
    st.metric(
        "VDE_TOTAL [MJ/km]",
        f"{energy_values['vde_total_mj_per_km']:.3f}" if energy_values["vde_total_mj_per_km"] is not None else "-",
    )
    st.metric(
        "VDE_NET [MJ/km]",
        f"{energy_values['vde_net_mj_per_km']:.3f}" if energy_values["vde_net_mj_per_km"] is not None else "-",
    )
    st.metric("Cycle", str(current_vde_row.get("cycle_name") or "-"))
    st.metric("Transmission", str(current_vde_row.get("transmission_type") or "-"))


def _render_scorecard_scope_column(
    current_vde_row: dict,
    *,
    scope_label: str,
    source_vde_count: int,
    current_count: int,
    refresh_count: int,
) -> None:
    current_revision = current_vde_row.get("updated_at") or current_vde_row.get("created_at") or "-"
    st.markdown("#### Scorecard Scope")
    st.caption(scope_label)
    st.metric("Page anchor VDE", f"#{int(current_vde_row.get('id'))}")
    st.metric("Anchor revision", str(current_revision))
    st.metric("Source VDE lines", str(source_vde_count))
    st.metric("Current links", str(current_count))
    st.metric("Refresh required", str(refresh_count))
    st.metric("Anchor vehicle", f"{current_vde_row.get('make', '-')} {current_vde_row.get('model', '-')}".strip())
    st.metric("Anchor legislation", str(current_vde_row.get("legislation") or "-"))


def _render_scorecard_saved_column(
    row: pd.Series | dict,
    current_vde_row: dict,
    vde_row_lookup: dict[int, dict[str, Any]] | None = None,
) -> None:
    data = dict(row)
    scenario_id = int(data.get("id"))
    revision_state = _resolve_scenario_revision_state(data, vde_row_lookup, current_vde_row)
    scenario_vde_row = _resolve_scenario_vde_row(data, vde_row_lookup, current_vde_row)
    assumptions = _load_json_blob(data.get("assumptions_json"))
    provenance = _load_json_blob(data.get("provenance_json"))

    st.markdown(f"#### Scenario #{scenario_id}")
    _render_link_state_badge(revision_state["status"])
    st.caption(str(data.get("created_at") or "-"))
    st.metric("Source VDE", f"#{int(data['vde_id'])}" if pd.notna(data.get("vde_id")) else "-")
    st.metric("Vehicle", f"{(scenario_vde_row or {}).get('make', '-')} {(scenario_vde_row or {}).get('model', '-')}".strip())
    st.metric("Method", str(data.get("engine_method") or data.get("method_note") or "-"))
    st.metric("Energy basis", str(data.get("energy_basis") or "-"))
    st.metric("Fuel [L/100km]", f"{float(data['fuel_l_per_100km']):.2f}" if pd.notna(data.get("fuel_l_per_100km")) else "N/A")
    st.metric("Energy [Wh/km]", f"{float(data['energy_Wh_per_km']):.1f}" if pd.notna(data.get("energy_Wh_per_km")) else "N/A")
    st.metric("CO2 [g/km]", f"{float(data['gco2_per_km']):.1f}" if pd.notna(data.get("gco2_per_km")) else "N/A")
    st.metric("FTP-75", f"{float(data['fuel_ftp75_l_per_100km']):.2f} L/100" if pd.notna(data.get("fuel_ftp75_l_per_100km")) else "N/A")
    st.metric("HWFET", f"{float(data['fuel_hwfet_l_per_100km']):.2f} L/100" if pd.notna(data.get("fuel_hwfet_l_per_100km")) else "N/A")
    with st.expander("Provenance", expanded=False):
        st.caption(revision_state["message"])
        if scenario_vde_row is not None:
            st.caption(f"Live VDE revision: {resolve_vde_source_revision(scenario_vde_row) or '-'}")
        st.write({"provenance": provenance, "assumptions": assumptions})


def _scenario_scorecard_field_value(
    row: pd.Series | dict,
    field_key: str,
    current_vde_row: dict,
    vde_row_lookup: dict[int, dict[str, Any]] | None = None,
) -> Any:
    data = dict(row)
    assumptions = _load_json_blob(data.get("assumptions_json"))
    provenance = _load_json_blob(data.get("provenance_json"))
    revision_state = _resolve_scenario_revision_state(data, vde_row_lookup, current_vde_row)
    scenario_vde_row = _resolve_scenario_vde_row(data, vde_row_lookup, current_vde_row)
    scenario_energy = resolve_vde_energy_values(scenario_vde_row or {})

    mapping = {
        "source_vde": f"#{int(data['vde_id'])}" if pd.notna(data.get("vde_id")) else "-",
        "vehicle": f"{(scenario_vde_row or {}).get('make', '-')} {(scenario_vde_row or {}).get('model', '-')}".strip(),
        "legislation": (scenario_vde_row or {}).get("legislation") or "-",
        "category": (scenario_vde_row or {}).get("category") or "-",
        "cycle": (scenario_vde_row or {}).get("cycle_name") or "-",
        "transmission": (scenario_vde_row or {}).get("transmission_type") or "-",
        "mass_kg": (scenario_vde_row or {}).get("mass_kg"),
        "live_revision": resolve_vde_source_revision(scenario_vde_row) or "-",
        "live_vde_total": scenario_energy.get("vde_total_mj_per_km"),
        "live_vde_net": scenario_energy.get("vde_net_mj_per_km"),
        "link_state": _link_state_table_label(revision_state["status"]),
        "method": data.get("engine_method") or data.get("method_note") or "-",
        "energy_basis": data.get("energy_basis") or "-",
        "engine_version": data.get("engine_version") or "-",
        "fuel_l_100km": data.get("fuel_l_per_100km"),
        "energy_Wh_km": data.get("energy_Wh_per_km"),
        "gco2_km": data.get("gco2_per_km"),
        "fuel_ftp75": data.get("fuel_ftp75_l_per_100km"),
        "fuel_hwfet": data.get("fuel_hwfet_l_per_100km"),
        "energy_ftp75": data.get("energy_ftp75_Wh_per_km"),
        "energy_hwfet": data.get("energy_hwfet_Wh_per_km"),
        "electrification": data.get("electrification") or "-",
        "gear_count": data.get("gear_count"),
        "final_drive_ratio": data.get("final_drive_ratio"),
        "fuel_type": assumptions.get("fuel_type") or provenance.get("fuel_type") or "-",
        "eta_pt_est": assumptions.get("eta_pt_est") or assumptions.get("eta_pt") or "-",
        "bev_eff_drive": assumptions.get("bev_eff_drive") or assumptions.get("driveline_eff") or "-",
        "utility_factor": assumptions.get("utility_factor") or assumptions.get("uf_phev") or "-",
        "energy_basis_value": provenance.get("energy_basis_value"),
        "data_origin": provenance.get("data_origin") or "-",
        "confidence": provenance.get("confidence") or "-",
        "saved_revision": data.get("source_vde_revision") or "-",
        "warnings": ", ".join(provenance.get("warnings") or []) or "-",
        "created_at": data.get("created_at") or "-",
    }
    return mapping.get(field_key, "-")


def _render_scorecard_group(
    title: str,
    rows: list[tuple[str, Any, str]],
    current_vde_row: dict,
    selected_df: pd.DataFrame,
    vde_row_lookup: dict[int, dict[str, Any]] | None = None,
    reference_label: str = "Reference VDE",
) -> None:
    table = {
        "Field": [label for label, _, _ in rows],
        reference_label: [ref for _, ref, _ in rows],
    }
    for _, row in selected_df.iterrows():
        col = f"#{int(row['id'])}"
        table[col] = [
            _scenario_scorecard_field_value(row, scenario_key, current_vde_row, vde_row_lookup)
            for _, _, scenario_key in rows
        ]
    st.markdown(f"#### {title}")
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


def render_scorecard_panel(vde_id: int, vde_row: dict) -> None:
    st.subheader("Scenario Compare")
    current_vde_row = fetch_vde_row(vde_id)
    scope = st.radio(
        "Comparison source scope",
        ["Current VDE only", "All saved scenarios"],
        horizontal=True,
        key="pwt_scorecard_scope",
    )
    saved_df = fetch_fuelcons_by_vde(vde_id) if scope == "Current VDE only" else fetch_fuelcons_all({})
    vde_row_lookup = _build_vde_row_lookup(saved_df)
    is_global_scope = scope == "All saved scenarios"

    if saved_df.empty:
        st.info("No saved Powertrain Scenarios yet. The scorecard will expand as soon as scenarios are saved.")
        if is_global_scope:
            _render_scorecard_scope_column(
                current_vde_row,
                scope_label=scope,
                source_vde_count=0,
                current_count=0,
                refresh_count=0,
            )
        else:
            _render_scorecard_reference_column(current_vde_row)
        return

    options, label_to_id = _scorecard_options(saved_df)
    default_labels = options[: min(3, len(options))]
    selected_labels = st.multiselect(
        "Scenarios in comparison",
        options,
        default=default_labels,
        key="pwt_scorecard_selection",
    )
    if not selected_labels:
        st.info("Select at least one scenario to populate the scorecard.")
        return

    selected_ids = [label_to_id[label] for label in selected_labels[:3]]
    selected_df = saved_df[saved_df["id"].isin(selected_ids)].copy().sort_values("id", ascending=False)

    statuses = [
        _resolve_scenario_revision_state(row, vde_row_lookup, current_vde_row)["status"]
        for _, row in selected_df.iterrows()
    ]
    current_count = sum(1 for status in statuses if status == "current")
    changed_count = sum(1 for status in statuses if status == "changed")
    missing_count = sum(1 for status in statuses if status == "missing")
    source_vde_count = int(selected_df["vde_id"].dropna().nunique()) if "vde_id" in selected_df.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Saved scenarios", str(len(saved_df)))
    m2.metric("On scorecard", str(len(selected_df)))
    m3.metric("Current", str(current_count))
    m4.metric("Refresh required", str(changed_count + missing_count))

    cols = st.columns(len(selected_df) + 1)
    with cols[0]:
        if is_global_scope:
            _render_scorecard_scope_column(
                current_vde_row,
                scope_label=scope,
                source_vde_count=source_vde_count,
                current_count=current_count,
                refresh_count=changed_count + missing_count,
            )
        else:
            _render_scorecard_reference_column(current_vde_row)
    for idx, (_, row) in enumerate(selected_df.iterrows(), start=1):
        with cols[idx]:
            _render_scorecard_saved_column(row, current_vde_row, vde_row_lookup)

    compare_rows = []
    for _, row in selected_df.iterrows():
        revision_state = _resolve_scenario_revision_state(row, vde_row_lookup, current_vde_row)
        compare_rows.append(
            {
                "Scenario": f"#{int(row['id'])}",
                "Source VDE": f"#{int(row['vde_id'])}" if pd.notna(row.get("vde_id")) else "-",
                "State": _link_state_table_label(revision_state["status"]),
                "Method": row.get("engine_method"),
                "Basis": row.get("energy_basis"),
                "Fuel [L/100km]": row.get("fuel_l_per_100km"),
                "Energy [Wh/km]": row.get("energy_Wh_per_km"),
                "CO2 [g/km]": row.get("gco2_per_km"),
                "Saved rev": row.get("source_vde_revision"),
            }
        )
    st.markdown("#### Quick Compare")
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    energy_values = resolve_vde_energy_values(current_vde_row)
    current_revision = current_vde_row.get("updated_at") or current_vde_row.get("created_at") or "-"
    reference_label = "Reference VDE" if not is_global_scope else "Reference / Scope"

    _render_scorecard_group(
        "Key Results",
        [
            ("Link state", "-", "link_state"),
            ("Method", "-", "method"),
            ("Energy basis", "-", "energy_basis"),
            (
                "Live VDE_TOTAL [MJ/km]",
                energy_values.get("vde_total_mj_per_km") if not is_global_scope else "Own source VDE",
                "live_vde_total",
            ),
            (
                "Live VDE_NET [MJ/km]",
                energy_values.get("vde_net_mj_per_km") if not is_global_scope else "Own source VDE",
                "live_vde_net",
            ),
            ("Basis value used [MJ/km]", "-" if not is_global_scope else "Scenario provenance", "energy_basis_value"),
            ("Fuel [L/100km]", "-", "fuel_l_100km"),
            ("Energy [Wh/km]", "-", "energy_Wh_km"),
            ("CO2 [g/km]", "-", "gco2_km"),
            ("FTP-75 fuel [L/100km]", "-", "fuel_ftp75"),
            ("HWFET fuel [L/100km]", "-", "fuel_hwfet"),
        ],
        current_vde_row,
        selected_df,
        vde_row_lookup,
        reference_label,
    )

    _render_scorecard_group(
        "What Changed / Scenario State",
        [
            ("Source VDE", f"#{int(current_vde_row.get('id'))}" if not is_global_scope else "Scenario-specific", "source_vde"),
            ("Link state", "-", "link_state"),
            ("Engine method", "-", "method"),
            ("Energy basis", "-", "energy_basis"),
            ("Live revision", current_revision if not is_global_scope else "Scenario-specific", "live_revision"),
            ("Gears", "-", "gear_count"),
            ("Final drive ratio", "-", "final_drive_ratio"),
            ("Saved revision", current_revision if not is_global_scope else "Saved with each scenario", "saved_revision"),
            ("Created at", "-", "created_at"),
        ],
        current_vde_row,
        selected_df,
        vde_row_lookup,
        reference_label,
    )

    _render_scorecard_group(
        "Vehicle / Powertrain",
        [
            ("Source VDE", f"#{int(current_vde_row.get('id'))}" if not is_global_scope else "Scenario-specific", "source_vde"),
            ("Vehicle", f"{current_vde_row.get('make', '-')} {current_vde_row.get('model', '-')}".strip() if not is_global_scope else "Own source VDE", "vehicle"),
            ("Electrification", current_vde_row.get("engine_type") if not is_global_scope else "Scenario-specific", "electrification"),
            ("Legislation", current_vde_row.get("legislation") if not is_global_scope else "Own source VDE", "legislation"),
            ("Category", current_vde_row.get("category") if not is_global_scope else "Own source VDE", "category"),
            ("Cycle", current_vde_row.get("cycle_name") if not is_global_scope else "Own source VDE", "cycle"),
            ("Transmission", current_vde_row.get("transmission_type") if not is_global_scope else "Own source VDE", "transmission"),
            ("Mass [kg]", current_vde_row.get("mass_kg") if not is_global_scope else "Own source VDE", "mass_kg"),
            ("Fuel type", "-", "fuel_type"),
        ],
        current_vde_row,
        selected_df,
        vde_row_lookup,
        reference_label,
    )

    _render_scorecard_group(
        "Assumptions / Provenance",
        [
            ("Data origin", "-", "data_origin"),
            ("Confidence", "-", "confidence"),
            ("Engine version", "-", "engine_version"),
            ("eta_pt", "-", "eta_pt_est"),
            ("BEV driveline eff.", "-", "bev_eff_drive"),
            ("Utility factor", "-", "utility_factor"),
            ("Warnings", "-", "warnings"),
            ("Saved revision", current_revision if not is_global_scope else "Saved with each scenario", "saved_revision"),
        ],
        current_vde_row,
        selected_df,
        vde_row_lookup,
        reference_label,
    )


def _render_saved_compare_panel(
    df: pd.DataFrame,
    current_vde_row: dict,
    vde_row_lookup: dict[int, dict[str, Any]],
    scope_label: str,
) -> None:
    if df.empty:
        return

    options = [_saved_scenario_label(row) for _, row in df.iterrows()]
    label_to_id = {label: int(df.iloc[idx]["id"]) for idx, label in enumerate(options)}
    default_labels = options[: min(3, len(options))]
    selected_labels = st.multiselect(
        "Scenarios to compare",
        options,
        default=default_labels,
        key="pwt_saved_compare_selection",
    )
    if not selected_labels:
        st.info("Select at least one saved scenario to compare.")
        return

    selected_ids = [label_to_id[label] for label in selected_labels]
    compare_df = df[df["id"].isin(selected_ids)].copy().sort_values("id", ascending=False)

    revision_states = [
        _resolve_scenario_revision_state(row, vde_row_lookup, current_vde_row)["status"]
        for _, row in compare_df.iterrows()
    ]
    changed_count = sum(1 for status in revision_states if status == "changed")
    current_count = sum(1 for status in revision_states if status == "current")
    missing_count = sum(1 for status in revision_states if status == "missing")
    source_vde_count = int(compare_df["vde_id"].dropna().nunique()) if "vde_id" in compare_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenario scope", scope_label)
    c2.metric("Source VDE lines", str(source_vde_count))
    c3.metric("Compared scenarios", str(len(compare_df)))
    c4.metric("Refresh required", str(changed_count + missing_count))

    st.markdown("#### Source Snapshot")
    ref1, ref2, ref3, ref4 = st.columns(4)
    ref1.metric("Page anchor VDE", f"#{int(current_vde_row.get('id'))}")
    ref2.metric("Current links", f"{current_count}/{len(compare_df)}")
    ref3.metric("Missing provenance", str(missing_count))
    ref4.metric("Changed links", str(changed_count))
    source_snapshot_rows = []
    for _, row in compare_df.iterrows():
        scenario_vde_row = _resolve_scenario_vde_row(row, vde_row_lookup, current_vde_row)
        energy_values = resolve_vde_energy_values(scenario_vde_row or {})
        source_snapshot_rows.append(
            {
                "Scenario": f"#{int(row['id'])}",
                "Source VDE": f"#{int(row['vde_id'])}" if pd.notna(row.get("vde_id")) else "-",
                "Vehicle": f"{(scenario_vde_row or {}).get('make', '-')} {(scenario_vde_row or {}).get('model', '-')}".strip(),
                "Revision": resolve_vde_source_revision(scenario_vde_row) or "-",
                "Cycle": (scenario_vde_row or {}).get("cycle_name") or "-",
                "VDE_TOTAL [MJ/km]": energy_values.get("vde_total_mj_per_km"),
                "VDE_NET [MJ/km]": energy_values.get("vde_net_mj_per_km"),
            }
        )
    st.dataframe(pd.DataFrame(source_snapshot_rows), use_container_width=True, hide_index=True)

    summary_rows = []
    for _, row in compare_df.iterrows():
        revision_state = _resolve_scenario_revision_state(row, vde_row_lookup, current_vde_row)
        summary_rows.append(
            {
                "Scenario": f"#{int(row['id'])}",
                "Source VDE": f"#{int(row['vde_id'])}" if pd.notna(row.get("vde_id")) else "-",
                "Link state": _link_state_table_label(revision_state["status"]),
                "Method": row.get("engine_method"),
                "Basis": row.get("energy_basis"),
                "Fuel [L/100km]": row.get("fuel_l_per_100km"),
                "Energy [Wh/km]": row.get("energy_Wh_per_km"),
                "CO2 [g/km]": row.get("gco2_per_km"),
                "Saved revision": row.get("source_vde_revision"),
                "Created": row.get("created_at"),
            }
        )
    st.markdown("#### Compare Summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Scenario Status")
    badge_cols = st.columns(min(4, max(1, len(compare_df))))
    for idx, (_, row) in enumerate(compare_df.iterrows()):
        revision_state = _resolve_scenario_revision_state(row, vde_row_lookup, current_vde_row)
        scenario_vde_row = _resolve_scenario_vde_row(row, vde_row_lookup, current_vde_row)
        with badge_cols[idx % len(badge_cols)]:
            st.caption(f"Scenario #{int(row['id'])}")
            _render_link_state_badge(revision_state["status"])
            st.caption(
                f"Source VDE #{int(row['vde_id'])}" if pd.notna(row.get("vde_id")) else "Source VDE unavailable"
            )
            st.caption(str(row.get("engine_method") or row.get("method_note") or "-"))
            if scenario_vde_row is not None:
                st.caption(f"{scenario_vde_row.get('make', '-')} {scenario_vde_row.get('model', '-')}".strip())

    compare_fields = [
        ("Source VDE", lambda row, source_row, _: f"#{int(row['vde_id'])}" if pd.notna(row.get("vde_id")) else "-"),
        ("Vehicle", lambda _, source_row, __: f"{(source_row or {}).get('make', '-')} {(source_row or {}).get('model', '-')}".strip()),
        ("Live VDE revision", lambda _, source_row, __: resolve_vde_source_revision(source_row) or "-"),
        ("Saved VDE revision", lambda row, _, __: row.get("source_vde_revision") or "-"),
        ("Link state", lambda _, __, state: _link_state_table_label(state["status"])),
        ("Cycle", lambda _, source_row, __: (source_row or {}).get("cycle_name") or "-"),
        ("Method", lambda row, _, __: row.get("engine_method") or "-"),
        ("Method note", lambda row, _, __: row.get("method_note") or "-"),
        ("Energy basis", lambda row, _, __: row.get("energy_basis") or "-"),
        ("Engine version", lambda row, _, __: row.get("engine_version") or "-"),
        ("Fuel [L/100km]", lambda row, _, __: row.get("fuel_l_per_100km")),
        ("Energy [Wh/km]", lambda row, _, __: row.get("energy_Wh_per_km")),
        ("CO2 [g/km]", lambda row, _, __: row.get("gco2_per_km")),
        ("Fuel FTP-75 [L/100km]", lambda row, _, __: row.get("fuel_ftp75_l_per_100km")),
        ("Fuel HWFET [L/100km]", lambda row, _, __: row.get("fuel_hwfet_l_per_100km")),
        ("Energy FTP-75 [Wh/km]", lambda row, _, __: row.get("energy_ftp75_Wh_per_km")),
        ("Energy HWFET [Wh/km]", lambda row, _, __: row.get("energy_hwfet_Wh_per_km")),
        ("Gears", lambda row, _, __: row.get("gear_count")),
        ("Final drive ratio", lambda row, _, __: row.get("final_drive_ratio")),
        ("Created at", lambda row, _, __: row.get("created_at") or "-"),
    ]
    compare_matrix = {"Field": [field for field, _ in compare_fields]}
    for _, row in compare_df.iterrows():
        column_name = f"#{int(row['id'])}"
        scenario_vde_row = _resolve_scenario_vde_row(row, vde_row_lookup, current_vde_row)
        revision_state = _resolve_scenario_revision_state(row, vde_row_lookup, current_vde_row)
        compare_matrix[column_name] = [getter(row, scenario_vde_row, revision_state) for _, getter in compare_fields]
    st.markdown("#### Field-by-Field Comparison")
    st.dataframe(pd.DataFrame(compare_matrix), use_container_width=True, hide_index=True)

    st.markdown("#### Provenance Details")
    for _, row in compare_df.iterrows():
        assumptions = _load_json_blob(row.get("assumptions_json"))
        provenance = _load_json_blob(row.get("provenance_json"))
        revision_state = _resolve_scenario_revision_state(row, vde_row_lookup, current_vde_row)
        scenario_vde_row = _resolve_scenario_vde_row(row, vde_row_lookup, current_vde_row)
        with st.expander(f"#{int(row['id'])} provenance", expanded=False):
            _render_link_state_badge(revision_state["status"])
            st.caption(revision_state["message"])
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Source VDE", f"#{int(row['vde_id'])}" if pd.notna(row.get("vde_id")) else "-")
            p2.metric("Basis", str(row.get("energy_basis") or "-"))
            p3.metric("Engine version", str(row.get("engine_version") or "-"))
            p4.metric("Link state", _link_state_label(revision_state["status"]))
            if scenario_vde_row is not None:
                live_vehicle = f"{scenario_vde_row.get('make', '-')} {scenario_vde_row.get('model', '-')}".strip()
                st.caption(
                    f"Live row: {live_vehicle} | rev {resolve_vde_source_revision(scenario_vde_row) or '-'}"
                )
            st.write(
                {
                    "provenance": provenance,
                    "assumptions": assumptions,
                }
            )


def _render_common_save_mode(vde_id: int) -> tuple[str, int | None]:
    existing_rows = fetch_fuelcons_by_vde(vde_id)
    save_mode_label = st.radio(
        "Save mode",
        ["Create new scenario", "Update existing scenario"],
        horizontal=True,
        key="pwt_common_save_mode",
    )

    if save_mode_label == "Create new scenario":
        return "insert_new", None

    if existing_rows.empty:
        st.info("No saved Powertrain Scenario exists for this VDE yet. The current review can only be saved as a new scenario.")
        return "insert_new", None

    options = []
    for _, row in existing_rows.sort_values("id", ascending=False).iterrows():
        result_value = row.get("fuel_l_per_100km")
        if pd.isna(result_value):
            result_value = row.get("energy_Wh_per_km")
        result_text = "-" if pd.isna(result_value) else str(result_value)
        options.append((int(row["id"]), f"#{int(row['id'])} | {row.get('method_note', '-') or '-'} | {result_text}"))

    selected_label = st.selectbox(
        "Target saved scenario",
        [label for _, label in options],
        key="pwt_common_update_target",
    )
    selected_row_id = next(row_id for row_id, label in options if label == selected_label)
    return "update_existing", selected_row_id


def _build_active_fuel_estimate_request(vde_id: int, vde_row: dict, ctx: Dict[str, Any], regression_vde: float | None) -> FuelEstimateRequest | None:
    method = str(st.session_state.get("pwt_setup_method") or PWT_ESTIMATION_METHODS[0])
    electrification = str(ctx.get("electrification") or "ICE").upper()
    energy_basis = str(ctx.get("energy_basis") or "VDE_TOTAL").upper()

    if method == "Manual / Imported":
        return build_fuel_estimate_request_from_vde(
            vde_row,
            electrification=electrification,
            energy_basis=energy_basis,
            method="manual_imported",
            manual_inputs={
                "source": st.session_state.get("pwt_manual_source") or "user_input",
                "fuel_l_100km": to_float(st.session_state.get("pwt_manual_fuel_l100")) or None,
                "energy_Wh_km": to_float(st.session_state.get("pwt_manual_energy_whkm")) or None,
                "gco2_km": to_float(st.session_state.get("pwt_manual_gco2_km")) or None,
            },
        )

    if method == "Physics Simple":
        fuel_type = st.session_state.get("sb_fuel_type") or "Gasoline"
        powertrain_features: Dict[str, Any] = {
            "fuel_type": fuel_type,
            "LHV_MJ_per_L": float(to_float(st.session_state.get("sb_lhv_override")) or _default_lhv(fuel_type)),
            "gCO2_per_L": _default_gco2_per_l(fuel_type),
        }
        eta_pt = to_float(st.session_state.get("sb_eta_pt"))
        eta_drive = to_float(st.session_state.get("sb_eta_drive"))
        grid = to_float(st.session_state.get("sb_grid"))
        uf_phev = to_float(st.session_state.get("sb_uf"))
        if eta_pt and eta_pt > 0:
            powertrain_features["eta_pt_est"] = float(eta_pt)
        if eta_drive and eta_drive > 0:
            powertrain_features["bev_eff_drive"] = float(eta_drive)
        if grid is not None:
            powertrain_features["grid_gco2_per_kwh"] = float(grid)
        if uf_phev is not None:
            powertrain_features["utility_factor"] = max(0.0, min(1.0, float(uf_phev)))

        request = build_fuel_estimate_request_from_vde(
            vde_row,
            electrification=electrification,
            energy_basis=energy_basis,
            method="physics_simple",
            powertrain_features=powertrain_features,
        )
        if electrification == "BEV" and bool(ctx.get("draft_bev_placeholders")):
            request.vehicle_features.update(apply_bev_placeholders(vde_id))
        return request

    if method == "Regression":
        if regression_vde is None:
            return None
        filters = dict(st.session_state.get("pwt_regression_filters") or {})
        filters.setdefault("legislation", vde_row.get("legislation"))
        return build_fuel_estimate_request_from_vde(
            vde_row,
            electrification=electrification,
            energy_basis=energy_basis,
            method="regression_existing",
            model_options={
                "regression_runner": _build_regression_runner(vde_id, electrification, filters),
            },
        )

    if method == "ML Prediction":
        return build_fuel_estimate_request_from_vde(
            vde_row,
            electrification=electrification,
            energy_basis=energy_basis,
            method="ml_prediction",
            powertrain_features={
                "gear_count": st.session_state.get("pwt_gears") or vde_row.get("gear_count"),
                "final_drive_ratio": st.session_state.get("pwt_fdr") or vde_row.get("final_drive_ratio"),
            },
            model_options={
                "ml_artifact_path": st.session_state.get("pwt_ml_artifact_path"),
            },
        )

    return None


def render_fuel_review_save_panel(vde_id: int, vde_row: dict, ctx: Dict[str, Any], regression_vde: float | None) -> None:
    st.subheader("Results & Save")
    request = _build_active_fuel_estimate_request(vde_id, vde_row, ctx, regression_vde)
    if request is None:
        st.info("Select and configure a functional estimation path before previewing the Powertrain Scenario.")
        return

    result = run_fuel_estimation(request)
    staged = build_fuel_scenario_save_payload(result, extra_payload=_fuel_scenario_extra_payload())
    save_mode, target_row_id = _render_common_save_mode(vde_id)
    current_revision = result.request.vehicle_features.get("source_vde_revision")
    assumptions = result.assumptions or {}
    comparative_analysis = (
        {
            "peers": list(assumptions.get("nearest_peers") or []),
            "summary": dict(assumptions.get("nearest_peer_summary") or {}),
            "quality": dict(assumptions.get("peer_group_quality") or {}),
            "hints": list(assumptions.get("investigation_hints") or []),
        }
        if result.method == "ml_prediction"
        else build_peer_analysis_for_request(
            request,
            outputs=_build_result_outputs_for_peer_analysis(result),
            n=5,
        )
    )

    st.markdown("#### Performance Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Method", str(result.method or "-"))
    s2.metric("Energy basis", str(result.energy_basis_used or "-"))
    s3.metric("Fuel", f"{result.fuel_l_100km:.2f} L/100km" if result.fuel_l_100km is not None else "N/A")
    s4.metric("Energy", f"{result.energy_Wh_km:.1f} Wh/km" if result.energy_Wh_km is not None else "N/A")
    s5.metric("CO2", f"{result.gco2_km:.1f} g/km" if result.gco2_km is not None else "N/A")

    st.markdown("#### Review Status")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Scenario", str(ctx.get("scenario_name") or "-"))
    r2.metric("Save action", "Update existing" if save_mode == "update_existing" else "Create new")
    r3.metric("Data origin", staged.data_origin)
    r4.metric("Confidence", str(result.confidence or "-"))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Intent", str(ctx.get("scenario_intent") or "-"))
    d2.metric("Electrification", str(ctx.get("electrification") or "-"))
    d3.metric("VDE source", f"#{vde_id}")
    d4.metric("Source revision", str(current_revision or "-"))
    if ctx.get("energy_value_mj_per_km") is not None:
        st.caption(f"Vehicle energy used for estimation: {ctx['energy_value_mj_per_km']:.4f} MJ/km")

    if result.warnings:
        st.warning("Preview warnings: " + ", ".join(result.warnings))

    st.markdown("#### Working Assumptions")
    if assumptions:
        summary_cards = []
        for key in ("source", "fuel_type", "eta_pt_est", "bev_eff_drive", "utility_factor", "dataset_rows"):
            if assumptions.get(key) not in (None, "", {}, []):
                summary_cards.append((key, assumptions.get(key)))
        if summary_cards:
            cols = st.columns(min(4, len(summary_cards)))
            for idx, (label, value) in enumerate(summary_cards):
                cols[idx % len(cols)].metric(label.replace("_", " ").title(), str(value))
        with st.expander("Full assumptions", expanded=False):
            st.write(assumptions)
    else:
        st.caption("No extra assumptions were required beyond the selected method inputs.")

    if result.method == "regression_existing":
        st.markdown("#### Regression Provenance")
        effective_filters = assumptions.get("effective_filters") or {}
        model_summary = assumptions.get("model_summary") or {}
        dataset_rows = assumptions.get("dataset_rows")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Energy basis", str(result.energy_basis_used or "-"))
        g2.metric("Dataset rows", str(dataset_rows if dataset_rows is not None else "-"))
        g3.metric("Method", "Regression")
        g4.metric("VDE source revision", str(current_revision or "-"))
        if result.warnings:
            st.caption("Regression warnings: " + ", ".join(result.warnings))
        with st.expander("Effective Regression Filters", expanded=False):
            st.write(effective_filters or {"info": "No explicit filter overrides were staged."})
        with st.expander("Regression Model Summary", expanded=False):
            st.write(model_summary or {"info": "Model summary unavailable."})

    if result.method in ("manual_imported", "physics_simple", "regression_existing"):
        _render_comparative_guidance(comparative_analysis)

    if result.method == "ml_prediction":
        st.markdown("#### ML Provenance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Artifact status", str(assumptions.get("integration_status") or "-"))
        m2.metric("Model", str(assumptions.get("model_name") or "Export pending"))
        m3.metric("SHAP", str(assumptions.get("shap_status") or "Pending"))
        m4.metric("Targets", ", ".join(assumptions.get("expected_targets") or []) or "-")
        st.caption("These ML signals are model/integration metadata, not proof of physical causality.")
        with st.expander("ML feature coverage", expanded=False):
            st.write(
                {
                    "features_used": assumptions.get("features_used"),
                    "missing_features": assumptions.get("missing_features"),
                    "artifact_candidates": assumptions.get("artifact_candidates"),
                    "coverage_status": assumptions.get("coverage_status", "unknown"),
                    "coverage_details": assumptions.get("coverage_details"),
                    "nearest_peers_available": assumptions.get("nearest_peers_available", False),
                }
            )
        _render_comparative_guidance(comparative_analysis, title="Nearest Peers")

        explanation = assumptions.get("ml_explanation") or {}
        st.markdown("#### Why did ML predict this?")
        if explanation.get("status") == "available":
            st.caption(str(explanation.get("message") or ""))
            explanation_df = pd.DataFrame(explanation.get("grouped_blocks") or [])
            if not explanation_df.empty:
                show_cols = [
                    col
                    for col in [
                        "engineering_block",
                        "contribution",
                        "main_features",
                        "interpretation",
                    ]
                    if col in explanation_df.columns
                ]
                st.dataframe(explanation_df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.caption(str(explanation.get("message") or "SHAP not available for this model in the current integration."))

    with st.expander("Phase Outputs", expanded=False):
        _render_phase_outputs_table(result.phase_outputs)

    if save_mode == "update_existing" and target_row_id is not None:
        st.caption(f"Selected saved Powertrain Scenario for update: #{target_row_id}")
        existing_rows = fetch_fuelcons_by_vde(vde_id)
        target_match = existing_rows.loc[existing_rows["id"] == target_row_id]
        if not target_match.empty:
            revision_state = compare_saved_scenario_revision(target_match.iloc[0].get("source_vde_revision"), vde_row)
            if revision_state["status"] == "changed":
                st.warning(
                    "You are updating a scenario whose saved VDE source is older than the current VDE. "
                    "Saving now will refresh it to the current VDE revision."
                )

    with st.expander("Provenance & technical review", expanded=False):
        st.write(
            {
                "request": result.request.to_dict(),
                "assumptions": result.assumptions,
                "confidence": result.confidence,
                "phase_outputs": result.phase_outputs,
                "warnings": result.warnings,
            }
        )

    with st.expander("Staged Save Payload", expanded=False):
        st.caption("This is the payload that the common Powertrain Scenario save flow will persist.")
        st.write(staged.payload)

    button_label = "Update Powertrain Scenario" if save_mode == "update_existing" else "Save Powertrain Scenario"
    if st.button(button_label, use_container_width=True, key="btn_save_fuel_scenario_common"):
        if not any(staged.payload.get(key) is not None for key in ("fuel_l_per_100km", "energy_Wh_per_km", "gco2_per_km")):
            st.warning("Nothing to save for the current powertrain scenario.")
        else:
            saved = save_fuel_estimate_result(
                result,
                save_mode,
                row_id=target_row_id,
                extra_payload=_fuel_scenario_extra_payload(),
            )
            if saved["action"] == "update_existing":
                st.success(f"Powertrain Scenario updated (id={saved['row_id']}).")
            else:
                st.success(f"Powertrain Scenario saved (id={saved['row_id']}).")


def render_comparison_report_page(vde_id: int, vde_row: dict) -> None:
    report_tab, analysis_tab, benchmark_tab = st.tabs(
        ["Scenario Compare", "Method Analysis", "Peers & Outlook"]
    )

    with report_tab:
        render_scorecard_panel(vde_id, vde_row)

    with analysis_tab:
        analysis_ctx = get_build_scenario_context(vde_id, vde_row)
        render_analysis_lab_panel(vde_id, vde_row, analysis_ctx, analysis_ctx.get("energy_value_mj_per_km"))

    with benchmark_tab:
        render_benchmark_regulatory_panel(vde_id, vde_row, get_build_scenario_context(vde_id, vde_row))


def render_benchmark_regulatory_panel(vde_id: int, vde_row: dict, ctx: dict) -> None:
    st.subheader("Peers & Outlook")

    st.markdown("#### A. Peer Benchmark")
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

    st.caption("Peer view for similar scenarios and saved estimates. This area is external to draft editing.")
    df_plot = build_scatter_from_fuel(df_fuel)
    plot_scatter_with_overlays(
        df_plot,
        ctx["electrification"],
        model=None,
        eta_lines=eta_lines,
        chart_key="pwt_benchmark_scatter",
    )
    render_fuelcons_table(df_fuel_table, editable=False, current_vde_row=vde_row)

    st.markdown("#### B. Regulatory Outlook")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**Inmetro PBEV**")
        st.caption("Indicative Projection - method validation pending.")
    with r2:
        st.markdown("**WLTP / UNECE**")
        st.caption("Indicative Projection - regulatory workflow not wired yet.")
    with r3:
        st.markdown("**CAFE Fleet Projection**")
        st.caption("Indicative Projection - fleet aggregation logic pending.")


def render_prediction_lab_panel() -> None:
    st.markdown("#### Future Capability Slots")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**ML Prediction**")
        st.caption("Planned - exploratory notebook exists, but no exported inference artifact is integrated yet.")
        st.button("Open ML Prediction", disabled=True, key="btn_pred_ml")
    with c2:
        st.markdown("**Physics + ML Residual**")
        st.caption("Planned - hybrid residual engine export/integration is still pending.")
        st.button("Open Hybrid Model", disabled=True, key="btn_pred_hybrid")
    with c3:
        st.markdown("**Map-Based Simulation**")
        st.caption("Planned - simulation integration and trace outputs are still pending.")
        st.button("Open Simulation", disabled=True, key="btn_pred_map")


def render_analysis_lab_panel(vde_id: int, vde_row: dict, ctx: Dict[str, Any], regression_vde: float | None) -> None:
    st.subheader("Method Analysis")
    active_method = str(st.session_state.get("pwt_setup_method") or PWT_ESTIMATION_METHODS[0])
    if active_method == "Regression":
        st.markdown("#### Regression Review")
        if regression_vde is None:
            st.warning("Regression analysis requires a resolved VDE energy input.")
            return
        regression_state = _resolve_regression_state(vde_id, vde_row, ctx, regression_vde, render_filters=False)
        reg_filters = regression_state["filters"]
        regdf = regression_state["dataset"]
        model = regression_state["model"]
        warnings = regression_state["warnings"]
        eta_lines = [0.20, 0.25, 0.30, 0.35] if ctx["electrification"] != "BEV" else [0.85, 0.90, 0.95, 0.98, 1.00, 1.05]

        st.caption("Method Analysis consumes the active Regression filters defined in Powertrain Scenario.")
        with st.expander("Effective Regression Filters", expanded=False):
            st.write(reg_filters or {"info": "No regression filters staged yet. Configure Regression in Powertrain Scenario first."})

        _render_regression_dataset_feedback(warnings, len(regdf))
        if "regression_dataset_empty" in warnings:
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset rows", str(len(regdf)))
        c2.markdown("**Model (Urban)**")
        c2.write(model.get("urb"))
        c3.markdown("**Model (Highway)**")
        c3.write(model.get("hw"))
        c4.markdown("**Model (Combined)**")
        c4.write(model.get("combined"))

        df_plot = build_scatter_from_fuel(regdf)
        plot_scatter_with_overlays(
            df_plot,
            ctx["electrification"],
            model,
            eta_lines,
            chart_key="pwt_analysis_regression",
        )
        st.caption("Regression uses the active builder filters and the currently selected draft electrification.")
        return

    request = _build_active_fuel_estimate_request(vde_id, vde_row, ctx, regression_vde)
    if request is not None and active_method in ("Manual / Imported", "Physics Simple"):
        result = run_fuel_estimation(request)
        st.markdown("#### Technical Diagnostics")
        with st.expander("Current draft diagnostics", expanded=True):
            st.write(
                {
                    "method": result.method,
                    "assumptions": result.assumptions,
                    "confidence": result.confidence,
                    "warnings": result.warnings,
                    "phase_outputs": result.phase_outputs,
                }
            )
        st.caption("Method Analysis explains the current method. Powertrain Scenario remains the only place where draft inputs are edited.")
        return

    render_prediction_lab_panel()


def render_saved_scenarios_panel(vde_id: int) -> None:
    st.subheader("Saved Estimates")
    current_vde_row = fetch_vde_row(vde_id)
    scope = st.radio(
        "Scenario source scope",
        ["Current VDE only", "All saved scenarios"],
        horizontal=True,
        key="pwt_saved_compare_scope",
    )
    saved_df = fetch_fuelcons_by_vde(vde_id) if scope == "Current VDE only" else fetch_fuelcons_all({})
    vde_row_lookup = _build_vde_row_lookup(saved_df)
    if scope == "All saved scenarios":
        st.caption("Comparison is now scenario-first: each saved scenario is checked against its own live VDE source row.")
    _render_saved_compare_panel(saved_df, current_vde_row, vde_row_lookup, scope)
    st.markdown("---")
    render_fuelcons_table(
        saved_df,
        editable=True,
        current_vde_row=current_vde_row if scope == "Current VDE only" else None,
        vde_row_lookup=vde_row_lookup,
    )
