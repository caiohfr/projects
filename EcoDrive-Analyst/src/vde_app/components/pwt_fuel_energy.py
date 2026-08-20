from __future__ import annotations

import copy
import html
import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.vde_app.derivatives import build_min_payload, enrich_with_derivatives
from src.vde_app.plots import build_scatter_from_fuel, plot_scatter_with_overlays
from src.vde_app.units import format_quantity, normalize_unit_system, unit_label
from src.vde_core.estimate_confidence import build_estimate_confidence_summary
from src.vde_core.fuel_energy import GCO2_PER_L, LHV_MJ_PER_L, MJ_TO_Wh
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
    summarize_saved_scenario_revision_states,
    update_fuelcons_payload,
)
from src.vde_core.regression import fit_regression_y_vs_vde, load_regression_dataset, predict_current_consumption
from src.vde_core.vde_setup_service import load_baselines_df, to_float

PWT_ESTIMATION_METHODS = [
    "Observed / Derived PSE",
    "ML Prediction",
    "Regression",
    "Physics Simple",
    "Manual / Imported",
    "Physics + ML Residual",
    "Map-Based Simulation",
]

SCENARIO_INTENTS = [
    "Baseline",
    "Proposal",
    "Imported / Reference",
]

PWT_SCENARIO_STEPS = [
    "Baseline Estimate",
    "Technology Delta",
    "Result & Save",
]

PWT_BASELINE_GUIDED_STAGES = [
    "Select demand",
    "Select baseline source",
    "Confirm baseline",
]

PWT_INPUT_MODES = [
    "Guided",
    "Spreadsheet Assist",
]

PWT_METHOD_DISPLAY_TO_INTERNAL = {
    "Reuse observed reference PSE": "Observed / Derived PSE",
    "ML prediction": "ML Prediction",
    "Regression estimate": "Regression",
    "Assume efficiency": "Physics Simple",
    "Enter observed/imported result": "Manual / Imported",
}

PWT_BENCH_HOTSPOTS = [
    {"key": "driver_cycle", "label": "Demand Source", "tab": "Baseline Estimate"},
    {"key": "roadload_vde", "label": "Roadload / VDE", "tab": "Baseline Estimate"},
    {"key": "powertrain_efficiency", "label": "Baseline Estimate", "tab": "Baseline Estimate"},
    {"key": "transmission", "label": "Reference Metadata", "tab": "Baseline Estimate"},
    {"key": "engine_fuel", "label": "Engine / Fuel", "tab": "Baseline Estimate"},
    {"key": "electric_battery", "label": "Electric Path", "tab": "Baseline Estimate"},
    {"key": "ml_peers", "label": "ML / SHAP / Peers", "tab": "Baseline Estimate"},
    {"key": "results", "label": "Proposal Result", "tab": "Result & Save"},
]

POWERTRAIN_REFERENCE_SOURCE_TYPES = [
    "Same vehicle fuelcons_db line",
    "Another fuelcons_db line",
    "Saved powertrain scenario",
    "Manual definition",
    "Imported map/simulation metadata",
    "Supplier data",
    "Engineering assumption",
]

POWERTRAIN_REFERENCE_SOURCE_LABELS = {
    "Same vehicle fuelcons_db line": "Use observed data from same vehicle",
    "Another fuelcons_db line": "Use observed data from another vehicle",
    "Saved powertrain scenario": "Reuse saved powertrain scenario",
    "Manual definition": "Enter manual baseline",
}

POWERTRAIN_REFERENCE_SOURCE_HELP = {
    "Same vehicle fuelcons_db line": "Uses observed fuel/PSE from the matching fuelcons_db line when available.",
    "Another fuelcons_db line": "Uses observed PSE from another fuelcons_db line and rebases it on the active VDE demand.",
    "Saved powertrain scenario": "Reuses a previously saved powertrain scenario as the baseline source.",
    "Manual definition": "Starts a new scenario draft so you can continue with ML prediction, regression estimate, assume efficiency, or enter an observed/imported result.",
}

DELTA_SUBSYSTEM_OPTIONS = [
    "engine",
    "transmission",
    "hybrid/ESS",
    "electrical/alternator",
    "auxiliary loads",
    "calibration",
    "thermal",
    "fuel system",
    "whole powertrain",
]

DELTA_SOURCE_TYPE_OPTIONS = [
    "engineering_assumption",
    "supplier_data",
    "imported_map",
    "simulation_result",
    "map_analysis",
    "test_data",
    "manual",
    "metadata_only",
]

DELTA_MATURITY_OPTIONS = [
    "metadata_only",
    "engineering_assumption",
    "supplier_data",
    "imported_map",
    "simulation_ready",
    "simulation_result",
    "correlated_model",
    "validated_against_test",
]

DELTA_EFFECT_BASIS_OPTIONS = [
    "Fuel consumption percent delta",
    "PSE percent delta",
    "CO2 percent delta",
    "Efficiency multiplier",
    "Metadata-only / registered-only",
]

DELTA_EFFECT_BASIS_ADVANCED_OPTIONS = [
    "PSE delta",
    "PSE multiplier",
    "fuel delta",
    "CO2 delta",
    "energy delta",
    "map-based effect",
]

DELTA_CONFIDENCE_OPTIONS = ["unknown", "low", "medium", "high"]

PWT_DRAFT_RESET_KEYS = [
    "pwt_scenario_name",
    "pwt_scenario_intent",
    "pwt_scenario_electrification",
    "pwt_energy_basis",
    "pwt_energy_basis_label",
    "pwt_bev_draft_placeholders",
    "pwt_setup_method",
    "pwt_setup_method_explicit",
    "pwt_deterministic_submethod",
    "pwt_manual_fuel_l100",
    "pwt_manual_energy_whkm",
    "pwt_manual_gco2_km",
    "pwt_manual_source",
    "pwt_manual_confidence",
    "pwt_manual_notes",
    "pwt_regression_filters",
    "pwt_feature_category",
    "pwt_feature_transmission_type",
    "pwt_feature_drive_type",
    "pwt_feature_engine_size_l",
    "pwt_feature_power_hp",
    "pwt_gears",
    "pwt_fdr",
    "pwt_trans_model",
    "pwt_trans_model_choice",
    "pwt_trans_model_custom",
    "pwt_common_save_mode",
    "pwt_common_update_target",
    "pwt_feature_inputs_initialized",
    "pwt_reference_source_type",
    "pwt_reference_same_row_id",
    "pwt_reference_other_row_id",
    "pwt_reference_saved_row_id",
    "pwt_reference_manual_label",
    "pwt_reference_manual_maturity",
    "pwt_reference_manual_note",
    "pwt_reference_inputs_initialized",
    "pwt_baseline_confirmed_method",
    "pwt_confirmed_baseline_snapshot",
    "pwt_guided_baseline_stage",
    "pwt_technology_deltas",
    "pwt_delta_name",
    "pwt_delta_subsystem",
    "pwt_delta_source_type",
    "pwt_delta_maturity",
    "pwt_delta_effect_basis",
    "pwt_delta_confidence",
    "pwt_delta_value",
    "pwt_delta_apply_toggle",
    "pwt_delta_notes",
    "pwt_delta_reference",
    "pwt_delta_remove_id",
    "sb_eta_pt",
    "sb_fuel_type",
    "sb_lhv_override",
    "sb_uf",
    "sb_eta_drive",
    "sb_grid",
]

FEATURE_READINESS_FIELDS = [
    {
        "key": "mass_kg",
        "label": "mass_kg",
        "importance": "useful_for_peers",
        "allow_imputed": False,
    },
    {
        "key": "test_mass_kg",
        "label": "test_mass_kg",
        "importance": "useful_for_peers",
        "allow_imputed": False,
    },
    {
        "key": "category",
        "label": "category",
        "session_key": "pwt_feature_category",
        "importance": "critical_for_ml",
        "allow_imputed": False,
    },
    {
        "key": "electrification",
        "label": "electrification",
        "session_key": "pwt_scenario_electrification",
        "importance": "critical_for_ml",
        "allow_imputed": False,
    },
    {
        "key": "fuel_type",
        "label": "fuel_type",
        "session_key": "sb_fuel_type",
        "importance": "useful_for_peers",
        "allow_imputed": True,
    },
    {
        "key": "transmission_type",
        "label": "transmission",
        "session_key": "pwt_feature_transmission_type",
        "importance": "critical_for_ml",
        "allow_imputed": True,
    },
    {
        "key": "drive_type",
        "label": "drive_type",
        "session_key": "pwt_feature_drive_type",
        "importance": "critical_for_ml",
        "allow_imputed": True,
    },
    {
        "key": "engine_max_power_kw",
        "label": "power_hp",
        "session_key": "pwt_feature_power_hp",
        "importance": "useful_for_regression",
        "allow_imputed": True,
    },
    {
        "key": "engine_size_l",
        "label": "engine_size_l",
        "session_key": "pwt_feature_engine_size_l",
        "importance": "critical_for_ml",
        "allow_imputed": True,
    },
    {
        "key": "gear_count",
        "label": "gear_count",
        "session_key": "pwt_gears",
        "importance": "critical_for_ml",
        "allow_imputed": True,
    },
    {
        "key": "final_drive_ratio",
        "label": "final_drive_ratio",
        "session_key": "pwt_fdr",
        "importance": "critical_for_ml",
        "allow_imputed": True,
    },
]

TRANSMISSION_TYPE_OPTIONS = ["AT", "AMT", "CVT", "MT", "OT", "SS"]
DRIVE_TYPE_OPTIONS = ["FWD", "RWD", "AWD", "4WD"]
FUEL_TYPE_OPTIONS = ["Gasoline", "E10", "E22", "E100", "Diesel", "Other"]


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
        st.markdown("### Scenario Pairing")
        st.caption("Vehicle Demand + Baseline Source")
        selected_label = st.selectbox(
            "Active VDE snapshot",
            labels,
            index=labels.index(current_label),
            key="pwt_active_vde_source",
            label_visibility="collapsed",
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
        _ensure_build_scenario_defaults(int(vde_id), vde_row)
        energy_values = resolve_vde_energy_values(vde_row)
        show_technical = _show_technical_details()
        ctx = get_build_scenario_context(vde_id, vde_row)

        left, right = st.columns([1.1, 1.3])
        with left:
            vehicle_text = f"{vde_row.get('make', '-')} {vde_row.get('model', '-')}".strip()
            st.caption("Vehicle Demand")
            st.markdown(
                f"`VDE #{int(vde_id)}` {vehicle_text} {str(vde_row.get('year') or '-')}"
            )
            st.caption(
                f"{str(ctx.get('energy_basis') or '-')} { _format_metric_value(ctx.get('energy_value_mj_per_km'), format_str='%.4f', suffix=' MJ/km')} | "
                f"{str(vde_row.get('cycle_name') or '-')}"
            )
            st.caption("NET available" if energy_values.get("vde_net_mj_per_km") is not None else "TOTAL only")

        with right:
            reference_options = _available_reference_source_types(vde_id)
            current_reference_type = st.session_state.get("pwt_reference_source_type")
            if current_reference_type not in reference_options:
                st.session_state["pwt_reference_source_type"] = "Manual definition"
            st.selectbox(
                "Baseline powertrain source",
                reference_options,
                key="pwt_reference_source_type",
                format_func=_reference_type_display_label,
            )
            reference_type = _reference_type_key(st.session_state.get("pwt_reference_source_type"))
            help_text = _reference_type_help_text(reference_type)
            if help_text:
                st.caption(help_text)
            if "Same vehicle fuelcons_db line" not in reference_options:
                st.caption("No same-vehicle fuelcons data is available for this VDE. Choose another source, assume efficiency, or enter an observed/imported result.")
            if reference_type in {"Same vehicle fuelcons_db line", "Another fuelcons_db line", "Saved powertrain scenario"}:
                candidate_df = _reference_candidates_for_type(vde_id, reference_type)
                if candidate_df.empty:
                    if reference_type == "Same vehicle fuelcons_db line":
                        st.caption("No same-vehicle observed data found. Choose another baseline source, assume efficiency, or enter an observed/imported result.")
                    else:
                        st.caption("No reference row found for the selected source.")
                else:
                    if reference_type == "Same vehicle fuelcons_db line":
                        selection_key = "pwt_reference_same_row_id"
                    elif reference_type == "Another fuelcons_db line":
                        selection_key = "pwt_reference_other_row_id"
                    else:
                        selection_key = "pwt_reference_saved_row_id"
                    labels_for_rows = []
                    label_to_id: dict[str, int] = {}
                    for _, row in candidate_df.iterrows():
                        try:
                            row_id = int(row["id"])
                        except Exception:
                            continue
                        row_label = _reference_candidate_label(row)
                        labels_for_rows.append(row_label)
                        label_to_id[row_label] = row_id
                    if labels_for_rows:
                        current_id = st.session_state.get(selection_key)
                        current_row_label = next((label for label, row_id in label_to_id.items() if row_id == current_id), labels_for_rows[0])
                        selected_row_label = st.selectbox("Reference row", labels_for_rows, index=labels_for_rows.index(current_row_label), key=f"{selection_key}_label")
                        st.session_state[selection_key] = label_to_id[selected_row_label]
            else:
                r1, r2 = st.columns(2)
                r1.text_input("Reference label", key="pwt_reference_manual_label")
                if show_technical:
                    r2.selectbox("Source maturity", DELTA_MATURITY_OPTIONS, key="pwt_reference_manual_maturity")
                    st.text_area("Reference note", key="pwt_reference_manual_note", height=70)

            reference_summary = _selected_powertrain_reference(vde_id, vde_row)
            st.caption("Baseline powertrain source")
            st.markdown(str(reference_summary.get("source_label") or "Reference pending"))
            compact_reference_parts = [_reference_type_display_label(reference_summary.get("source_type") or "Reference pending")]
            if reference_summary.get("observed_fuel") is not None:
                compact_reference_parts.append(
                    "Observed "
                    + _format_metric_value(reference_summary.get("observed_fuel"), format_str="%.2f", suffix=" L/100km")
                )
            if reference_summary.get("observed_pse") is not None:
                compact_reference_parts.append(
                    "PSE "
                    + _format_metric_value(reference_summary.get("observed_pse"), format_str="%.3f")
                )
            st.caption(" | ".join(part for part in compact_reference_parts if part))
            if show_technical and reference_summary.get("reference_vehicle_label") not in (None, "", "-"):
                st.caption(str(reference_summary.get("reference_vehicle_label")))
            if show_technical and (reference_summary.get("observed_fuel") is not None or reference_summary.get("observed_pse") is not None):
                observed_parts = []
                if reference_summary.get("observed_fuel") is not None:
                    observed_parts.append(f"{to_float(reference_summary.get('observed_fuel')):.2f} L/100km")
                if reference_summary.get("observed_pse") is not None:
                    observed_parts.append(f"PSE {_format_metric_value(reference_summary.get('observed_pse'), format_str='%.3f')}")
                st.caption("Observed: " + " | ".join(observed_parts))

        draft = _build_powertrain_scenario_draft(vde_id, vde_row)
        baseline_summary = dict((draft.get("proposal_result") or {}).get("baseline") or {})
        proposal_status = _current_proposal_status_label(draft)
        readiness = _scenario_feature_readiness_snapshot(vde_id, vde_row, ctx, regression_vde=ctx.get("energy_value_mj_per_km"), reference_summary=reference_summary)
        confidence_reason = _confidence_reason_label(
            readiness=readiness,
            active_method=str(draft.get("baseline_estimate", {}).get("method") or ""),
            reference_summary=reference_summary,
            regression_state=_regression_method_option_state(vde_id, vde_row, ctx, ctx.get("energy_value_mj_per_km")),
        )
        strip1, strip2, strip3 = st.columns(3)
        strip1.caption(
            "Baseline: "
            + (
                _format_metric_value(baseline_summary.get("fuel_l_100km"), format_str="%.2f", suffix=" L/100km")
                if baseline_summary.get("fuel_l_100km") is not None
                else _baseline_summary_empty_state(reference_summary)
            )
        )
        strip2.caption("Proposal: " + proposal_status)
        strip3.caption("Confidence: " + confidence_reason)

        st.caption("Metadata: " + _compact_metadata_line(readiness, ctx))

        if str(reference_summary.get("source_label") or "") == "No reference row available":
            message, actions = _powertrain_reference_empty_state(reference_summary)
            st.info(message + " " + " | ".join(actions))

        with st.expander("Demand technical details", expanded=False):
            a1, a2, a3 = st.columns(3)
            a1.metric("VDE_TOTAL", _format_metric_value(energy_values.get("vde_total_mj_per_km"), format_str="%.4f", suffix=" MJ/km"))
            a2.metric("VDE_NET", _format_metric_value(energy_values.get("vde_net_mj_per_km"), format_str="%.4f", suffix=" MJ/km"))
            a3.metric("Revision", str(resolve_vde_source_revision(vde_row) or "-"))
            b1, b2, b3 = st.columns(3)
            b1.metric("ABC A", _format_metric_value(vde_row.get("coast_A_N"), format_str="%.2f"))
            b2.metric("ABC B", _format_metric_value(vde_row.get("coast_B_N_per_kph"), format_str="%.4f"))
            b3.metric("ABC C", _format_metric_value(vde_row.get("coast_C_N_per_kph2"), format_str="%.5f"))
            st.caption("Edit vehicle demand in VDE Setup.")

        with st.expander("Review / edit metadata", expanded=False):
            _render_powertrain_metadata_review(vde_id, vde_row, ctx, readiness, expanded=False)

        if str(st.session_state.get("pwt_scenario_electrification") or "").upper() != "BEV":
            st.session_state["pwt_bev_draft_placeholders"] = False

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
    same_vehicle_rows = fetch_fuelcons_by_vde(vde_id)
    default_reference_type = "Same vehicle fuelcons_db line" if not same_vehicle_rows.empty else "Manual definition"

    st.session_state.setdefault("pwt_scenario_name", default_name)
    st.session_state.setdefault("pwt_scenario_intent", SCENARIO_INTENTS[1])
    st.session_state.setdefault("pwt_scenario_electrification", default_electrification)
    st.session_state.setdefault("pwt_energy_basis", default_basis)
    st.session_state.setdefault("pwt_bev_draft_placeholders", False)
    st.session_state.setdefault("pwt_setup_method_explicit", False)
    st.session_state.setdefault("pwt_manual_confidence", "unknown")
    st.session_state.setdefault("pwt_manual_notes", "")
    if st.session_state.get("pwt_setup_method") not in PWT_ESTIMATION_METHODS:
        st.session_state["pwt_setup_method"] = PWT_ESTIMATION_METHODS[0]
    if not st.session_state.get("pwt_feature_inputs_initialized"):
        st.session_state["pwt_gears"] = ""
        st.session_state["pwt_fdr"] = ""
        st.session_state["sb_fuel_type"] = "(leave missing)"
        st.session_state["pwt_feature_inputs_initialized"] = True
    if not st.session_state.get("pwt_reference_inputs_initialized"):
        st.session_state["pwt_reference_source_type"] = default_reference_type
        st.session_state["pwt_reference_same_row_id"] = None
        st.session_state["pwt_reference_other_row_id"] = None
        st.session_state["pwt_reference_saved_row_id"] = None
        st.session_state["pwt_reference_manual_label"] = "Scenario-local reference"
        st.session_state["pwt_reference_manual_maturity"] = "engineering_assumption"
        st.session_state["pwt_reference_manual_note"] = ""
        st.session_state["pwt_technology_deltas"] = []
        st.session_state["pwt_delta_name"] = ""
        st.session_state["pwt_delta_subsystem"] = DELTA_SUBSYSTEM_OPTIONS[-1]
        st.session_state["pwt_delta_source_type"] = "manual"
        st.session_state["pwt_delta_maturity"] = "engineering_assumption"
        st.session_state["pwt_delta_effect_basis"] = "Metadata-only / registered-only"
        st.session_state["pwt_delta_confidence"] = "unknown"
        st.session_state["pwt_delta_value"] = 0.0
        st.session_state["pwt_delta_apply_toggle"] = True
        st.session_state["pwt_delta_notes"] = ""
        st.session_state["pwt_delta_reference"] = ""
        st.session_state["pwt_reference_inputs_initialized"] = True
    if st.session_state.get("pwt_guided_baseline_stage") not in PWT_BASELINE_GUIDED_STAGES:
        st.session_state["pwt_guided_baseline_stage"] = PWT_BASELINE_GUIDED_STAGES[0]

    if st.session_state.get("pwt_scenario_electrification") not in ("ICE", "HEV", "PHEV", "BEV"):
        st.session_state["pwt_scenario_electrification"] = default_electrification
    if energy_values["vde_net_mj_per_km"] is None and st.session_state.get("pwt_energy_basis") == "VDE_NET":
        st.session_state["pwt_energy_basis"] = "VDE_TOTAL"


def _available_reference_source_types(vde_id: int) -> list[str]:
    options: list[str] = []
    same_vehicle_rows = fetch_fuelcons_by_vde(vde_id)
    if same_vehicle_rows is not None and not same_vehicle_rows.empty:
        options.append("Same vehicle fuelcons_db line")
    options.extend(
        [
            "Another fuelcons_db line",
            "Saved powertrain scenario",
            "Manual definition",
        ]
    )
    return options


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


def _reference_type_key(label: str | None) -> str:
    value = str(label or "Manual definition").strip()
    reverse_lookup = {display: source for source, display in POWERTRAIN_REFERENCE_SOURCE_LABELS.items()}
    return reverse_lookup.get(value, value)


def _reference_type_display_label(source_type: str | None) -> str:
    key = _reference_type_key(source_type)
    return POWERTRAIN_REFERENCE_SOURCE_LABELS.get(key, key or "-")


def _reference_type_help_text(source_type: str | None) -> str:
    key = _reference_type_key(source_type)
    return POWERTRAIN_REFERENCE_SOURCE_HELP.get(key, "")


def _maturity_rank(level: str | None) -> int:
    try:
        return DELTA_MATURITY_OPTIONS.index(str(level or "").strip())
    except ValueError:
        return -1


def _clean_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in dict(value or {}).items() if item not in (None, "", [], {})}


def _reference_candidate_label(row: pd.Series | dict) -> str:
    data = dict(row)
    vehicle = f"{str(data.get('make') or '-')} {str(data.get('model') or '-')}".strip()
    year = str(data.get("year") or "-")
    fuel_text = "-"
    if pd.notna(data.get("fuel_l_per_100km")):
        fuel_text = f"{float(data['fuel_l_per_100km']):.2f} L/100km"
    elif pd.notna(data.get("energy_Wh_per_km")):
        fuel_text = f"{float(data['energy_Wh_per_km']):.1f} Wh/km"
    return (
        f"FC #{int(data['id'])} | VDE #{int(data['vde_id']) if pd.notna(data.get('vde_id')) else '-'} | "
        f"{vehicle} {year} | {str(data.get('engine_method') or data.get('method_note') or '-')} | {fuel_text}"
    )


def _reference_candidates_for_type(vde_id: int, source_type: str) -> pd.DataFrame:
    if source_type == "Same vehicle fuelcons_db line":
        df = fetch_fuelcons_by_vde(vde_id)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["make"] = ""
        df["model"] = ""
        df["year"] = ""
        df["vde_id"] = int(vde_id)
        return df
    if source_type == "Another fuelcons_db line":
        df = fetch_fuelcons_all({})
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.loc[df["vde_id"] != int(vde_id)].copy()
        return df
    if source_type == "Saved powertrain scenario":
        df = fetch_fuelcons_all({})
        if df is None or df.empty:
            return pd.DataFrame()
        if "engine_method" in df.columns:
            scenario_df = df.loc[df["engine_method"].astype(str).str.contains("scenario", case=False, na=False)].copy()
            if not scenario_df.empty:
                return scenario_df
        return pd.DataFrame()
    return pd.DataFrame()


def _fuel_type_from_reference_row(row: dict[str, Any]) -> str | None:
    assumptions = _load_json_blob(row.get("assumptions_json"))
    provenance = _load_json_blob(row.get("provenance_json"))
    fuel_type = assumptions.get("fuel_type")
    if fuel_type in (None, ""):
        fuel_type = dict(provenance.get("scenario_feature_values") or {}).get("fuel_type")
    text = str(fuel_type).strip() if fuel_type not in (None, "") else None
    return text or None


def _derive_reference_pse(reference_row: dict[str, Any]) -> dict[str, Any]:
    source_vde_id = reference_row.get("vde_id")
    if source_vde_id in (None, ""):
        return {"value": None, "status": "unavailable", "basis": None}
    try:
        source_vde = fetch_vde_row(int(source_vde_id))
    except Exception:
        return {"value": None, "status": "unavailable", "basis": None}
    energy_values = resolve_vde_energy_values(source_vde)
    energy_basis = str(reference_row.get("energy_basis") or "VDE_TOTAL").upper()
    demand_value = energy_values["vde_net_mj_per_km"] if energy_basis == "VDE_NET" else energy_values["vde_total_mj_per_km"]
    if demand_value is None:
        return {"value": None, "status": "missing_demand", "basis": energy_basis}

    fuel_l_100km = to_float(reference_row.get("fuel_l_per_100km"))
    if fuel_l_100km is not None:
        fuel_type = _fuel_type_from_reference_row(reference_row) or "Gasoline"
        lhv = float(LHV_MJ_PER_L.get(fuel_type, LHV_MJ_PER_L["Gasoline"]))
        consumed = (fuel_l_100km / 100.0) * lhv
        if consumed > 0:
            return {"value": float(demand_value) / consumed, "status": "available", "basis": energy_basis}

    energy_wh_km = to_float(reference_row.get("energy_Wh_per_km"))
    if energy_wh_km is not None:
        consumed = float(energy_wh_km) / MJ_TO_Wh
        if consumed > 0:
            return {"value": float(demand_value) / consumed, "status": "available", "basis": energy_basis}
    return {"value": None, "status": "missing_observed_result", "basis": energy_basis}


def _reference_metadata_from_row(reference_row: dict[str, Any]) -> dict[str, Any]:
    assumptions = _load_json_blob(reference_row.get("assumptions_json"))
    provenance = _load_json_blob(reference_row.get("provenance_json"))
    scenario_values = dict(provenance.get("scenario_feature_values") or {})
    scenario_sources = dict(provenance.get("scenario_feature_sources") or {})
    powertrain = {
        "fuel_type": _fuel_type_from_reference_row(reference_row),
        "electrification": reference_row.get("electrification"),
        "engine_max_power_kw": reference_row.get("engine_max_power_kw"),
        "gear_count": reference_row.get("gear_count"),
        "final_drive_ratio": reference_row.get("final_drive_ratio"),
    }
    for key in ("engine_size_l", "transmission_type", "drive_type", "category", "electrification", "fuel_type", "gear_count", "final_drive_ratio", "engine_max_power_kw"):
        if scenario_values.get(key) not in (None, ""):
            powertrain[key] = scenario_values.get(key)
    source_kind = "inherited_from_reference_scenario" if "scenario" in str(reference_row.get("engine_method") or "").lower() else "inherited_from_fuelcons"
    sources = {key: scenario_sources.get(key) or source_kind for key, value in powertrain.items() if value not in (None, "")}
    powertrain["sources"] = sources
    powertrain["assumptions"] = assumptions
    return powertrain


def _selected_powertrain_reference(vde_id: int, vde_row: dict) -> dict[str, Any]:
    source_type = _reference_type_key(st.session_state.get("pwt_reference_source_type"))
    summary: dict[str, Any] = {
        "source_type": source_type,
        "source_id": None,
        "source_label": "Reference pending",
        "reference_vehicle_label": "-",
        "observed_fuel": None,
        "observed_energy": None,
        "observed_co2": None,
        "observed_pse": None,
        "maturity_level": "engineering_assumption" if source_type == "Manual definition" else "simulation-ready metadata",
        "metadata": {},
        "note": "",
    }

    if source_type in {"Same vehicle fuelcons_db line", "Another fuelcons_db line", "Saved powertrain scenario"}:
        candidates = _reference_candidates_for_type(vde_id, source_type)
        if candidates.empty:
            summary["source_label"] = "No reference row available"
            summary["maturity_level"] = "missing"
            return summary
        if source_type == "Same vehicle fuelcons_db line":
            selected_key = "pwt_reference_same_row_id"
        elif source_type == "Another fuelcons_db line":
            selected_key = "pwt_reference_other_row_id"
        else:
            selected_key = "pwt_reference_saved_row_id"
        selected_row_id = st.session_state.get(selected_key)
        if selected_row_id in (None, ""):
            try:
                selected_row_id = int(candidates.iloc[0]["id"])
                st.session_state[selected_key] = selected_row_id
            except Exception:
                selected_row_id = None
        row_match = candidates.loc[candidates["id"] == selected_row_id] if selected_row_id is not None else candidates.iloc[:1]
        if row_match.empty:
            row_match = candidates.iloc[:1]
        reference_row = dict(row_match.iloc[0])
        derived_pse = _derive_reference_pse(reference_row)
        summary.update(
            {
                "source_id": int(reference_row["id"]) if pd.notna(reference_row.get("id")) else None,
                "source_label": _reference_candidate_label(reference_row),
                "reference_vehicle_label": f"{str(reference_row.get('make') or vde_row.get('make') or '-')} {str(reference_row.get('model') or vde_row.get('model') or '-')}".strip(),
                "observed_fuel": to_float(reference_row.get("fuel_l_per_100km")),
                "observed_energy": to_float(reference_row.get("energy_Wh_per_km")),
                "observed_co2": to_float(reference_row.get("gco2_per_km")),
                "observed_pse": to_float(derived_pse.get("value")),
                "maturity_level": "simulation-ready metadata",
                "metadata": _reference_metadata_from_row(reference_row),
                "note": "Reference fuel result belongs to the source vehicle. Baseline estimate applies the selected conversion layer to the active VDE demand.",
                "row": reference_row,
            }
        )
        return summary

    if source_type == "Manual definition":
        label = str(st.session_state.get("pwt_reference_manual_label") or "Scenario-local reference").strip()
        maturity = str(st.session_state.get("pwt_reference_manual_maturity") or "engineering_assumption").strip()
        note = str(st.session_state.get("pwt_reference_manual_note") or "").strip()
        summary.update(
            {
                "source_label": label,
                "reference_vehicle_label": f"{str(vde_row.get('make') or '-')} {str(vde_row.get('model') or '-')}".strip(),
                "maturity_level": maturity,
                "note": note or "Manual reference uses the scenario-local metadata and assumptions only.",
            }
        )
        return summary

    summary["source_label"] = f"{source_type} - UI ready, backend pending"
    summary["maturity_level"] = "correlated model unavailable"
    summary["note"] = "This reference type is staged in the UI/state model, but its backend is not implemented yet."
    return summary


def _readiness_source_label(feature_key: str, reference_summary: dict[str, Any] | None) -> str | None:
    metadata = dict((reference_summary or {}).get("metadata") or {})
    sources = dict(metadata.get("sources") or {})
    return sources.get(feature_key)


def _technology_deltas(*, include_form_preview: bool = False) -> list[dict[str, Any]]:
    raw = list(st.session_state.get("pwt_technology_deltas") or [])
    if include_form_preview:
        preview_delta = _draft_delta_from_form()
        if preview_delta is not None:
            raw = raw + [preview_delta]
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        row = dict(item or {})
        row.setdefault("id", index)
        row.setdefault("name", f"Delta {index}")
        row.setdefault("affected_subsystem", "whole powertrain")
        row.setdefault("source_type", "manual")
        row.setdefault("maturity_level", "engineering_assumption")
        row.setdefault("effect_basis", "metadata only")
        row.setdefault("confidence", "unknown")
        row.setdefault("enabled", True)
        row.setdefault("notes", "")
        row.setdefault("reference_description", "")
        effect_value = to_float(row.get("effect_value"))
        row["effect_value"] = effect_value
        effect_basis = _normalize_delta_effect_basis(row.get("effect_basis") or "metadata_only")
        row["effect_basis"] = effect_basis
        if not bool(row.get("enabled")):
            row["quantitative_status"] = "disabled"
        elif effect_basis in {"map_based_effect"}:
            row["quantitative_status"] = "pending_model"
        elif effect_basis in {"metadata_only"} or str(row.get("source_type") or "") == "metadata_only":
            row["quantitative_status"] = "registered_only"
        elif effect_value is None:
            row["quantitative_status"] = "registered_only"
        else:
            row["quantitative_status"] = "applied"
        normalized.append(row)
    return normalized


def _reset_delta_form() -> None:
    st.session_state["pwt_delta_name"] = ""
    st.session_state["pwt_delta_subsystem"] = DELTA_SUBSYSTEM_OPTIONS[-1]
    st.session_state["pwt_delta_source_type"] = "manual"
    st.session_state["pwt_delta_maturity"] = "engineering_assumption"
    st.session_state["pwt_delta_effect_basis"] = "Metadata-only / registered-only"
    st.session_state["pwt_delta_confidence"] = "unknown"
    st.session_state["pwt_delta_value"] = 0.0
    st.session_state["pwt_delta_apply_toggle"] = True
    st.session_state["pwt_delta_notes"] = ""
    st.session_state["pwt_delta_reference"] = ""


def _delta_status_counts(deltas: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"applied": 0, "registered_only": 0, "pending_model": 0, "disabled": 0}
    for delta in deltas:
        status = str(delta.get("quantitative_status") or "registered_only")
        if status in summary:
            summary[status] += 1
    return summary


def _proposal_confidence_label(baseline_confidence: str | None, deltas: list[dict[str, Any]]) -> str:
    level = str(baseline_confidence or "low").strip().lower()
    if any(str(delta.get("confidence") or "").lower() == "low" for delta in deltas if delta.get("quantitative_status") == "applied"):
        return "low"
    if any(str(delta.get("quantitative_status") or "") in {"registered_only", "pending_model"} for delta in deltas):
        return "medium" if level == "high" else level or "medium"
    return level or "low"


def _apply_delta_stack_to_baseline(
    baseline_result: Any,
    *,
    ctx: dict[str, Any],
    deltas: list[dict[str, Any]],
) -> dict[str, Any]:
    if baseline_result is None:
        return {
            "status": "Proposal pending",
            "baseline": {},
            "proposal": {},
            "applied_deltas": [],
            "registered_only_deltas": list(deltas),
            "confidence": "low",
            "warnings": ["baseline_pending"],
            "delta_counts": _delta_status_counts(deltas),
        }

    assumptions = dict((baseline_result.assumptions or {}) or {})
    pse_summary = dict(assumptions.get("pse_summary") or {})
    demand_mj_per_km = to_float(ctx.get("energy_value_mj_per_km"))
    baseline = {
        "pse": to_float(pse_summary.get("value")),
        "fuel_l_100km": to_float(baseline_result.fuel_l_100km),
        "energy_Wh_km": to_float(baseline_result.energy_Wh_km),
        "gco2_km": to_float(baseline_result.gco2_km),
        "method": _pwt_method_label(baseline_result.method),
        "confidence": str(baseline_result.confidence or "-"),
    }
    proposal = dict(baseline)
    applied_deltas: list[dict[str, Any]] = []
    registered_only: list[dict[str, Any]] = []
    warnings: list[str] = []
    fuel_type = str(baseline_result.request.powertrain_features.get("fuel_type") or "Gasoline")
    lhv = float(baseline_result.request.powertrain_features.get("LHV_MJ_per_L") or LHV_MJ_PER_L.get(fuel_type, LHV_MJ_PER_L["Gasoline"]))
    gco2_per_l = float(baseline_result.request.powertrain_features.get("gCO2_per_L") or GCO2_PER_L.get(fuel_type, GCO2_PER_L["Gasoline"]))

    for delta in deltas:
        status = str(delta.get("quantitative_status") or "registered_only")
        if status != "applied":
            registered_only.append(delta)
            continue
        effect_basis = _normalize_delta_effect_basis(delta.get("effect_basis") or "")
        value = to_float(delta.get("effect_value"))
        if value is None:
            registered_only.append(delta)
            continue
        if effect_basis == "pse_delta" and proposal.get("pse") is not None:
            proposal["pse"] = float(proposal["pse"]) + float(value)
        elif effect_basis == "pse_percent_delta" and proposal.get("pse") is not None:
            proposal["pse"] = float(proposal["pse"]) * (1.0 + float(value) / 100.0)
        elif effect_basis in {"pse_multiplier", "efficiency_multiplier"} and proposal.get("pse") is not None:
            proposal["pse"] = float(proposal["pse"]) * float(value)
        elif effect_basis == "fuel_delta" and proposal.get("fuel_l_100km") is not None:
            proposal["fuel_l_100km"] = float(proposal["fuel_l_100km"]) + float(value)
        elif effect_basis == "fuel_percent_delta" and proposal.get("fuel_l_100km") is not None:
            proposal["fuel_l_100km"] = float(proposal["fuel_l_100km"]) * (1.0 + float(value) / 100.0)
        elif effect_basis == "co2_delta" and proposal.get("gco2_km") is not None:
            proposal["gco2_km"] = float(proposal["gco2_km"]) + float(value)
        elif effect_basis == "co2_percent_delta" and proposal.get("gco2_km") is not None:
            proposal["gco2_km"] = float(proposal["gco2_km"]) * (1.0 + float(value) / 100.0)
        elif effect_basis == "energy_delta" and proposal.get("energy_Wh_km") is not None:
            proposal["energy_Wh_km"] = float(proposal["energy_Wh_km"]) + float(value)
        else:
            delta = dict(delta)
            delta["quantitative_status"] = "registered_only"
            registered_only.append(delta)
            continue
        applied_deltas.append(delta)

    if proposal.get("pse") is not None and demand_mj_per_km is not None and proposal["pse"] > 0:
        if baseline_result.request.vehicle_features.get("electrification") == "BEV":
            proposal["energy_Wh_km"] = demand_mj_per_km / proposal["pse"] * MJ_TO_Wh
        elif proposal.get("fuel_l_100km") is None or any(
            _normalize_delta_effect_basis(delta.get("effect_basis") or "") in {"pse_delta", "pse_multiplier", "efficiency_multiplier", "pse_percent_delta"}
            for delta in applied_deltas
        ):
            proposal["fuel_l_100km"] = (demand_mj_per_km / proposal["pse"]) / lhv * 100.0
        if proposal.get("fuel_l_100km") is not None:
            proposal["gco2_km"] = (proposal["fuel_l_100km"] / 100.0) * gco2_per_l

    if proposal.get("fuel_l_100km") is not None and demand_mj_per_km is not None and lhv > 0:
        consumed_mj = (proposal["fuel_l_100km"] / 100.0) * lhv
        if consumed_mj > 0:
            proposal["pse"] = demand_mj_per_km / consumed_mj
        proposal["gco2_km"] = (proposal["fuel_l_100km"] / 100.0) * gco2_per_l

    if proposal.get("energy_Wh_km") is not None and demand_mj_per_km is not None and proposal["energy_Wh_km"] > 0 and baseline_result.request.vehicle_features.get("electrification") == "BEV":
        proposal["pse"] = demand_mj_per_km / (proposal["energy_Wh_km"] / MJ_TO_Wh)

    counts = _delta_status_counts(deltas)
    if not applied_deltas and registered_only:
        status = "No quantitative delta"
        warnings.append("registered_only_deltas")
    elif applied_deltas:
        status = "Estimated"
    else:
        status = "Proposal pending"

    highest_maturity = "-"
    if deltas:
        highest_maturity = max((str(delta.get("maturity_level") or "-") for delta in deltas), key=_maturity_rank)
    return {
        "status": status,
        "baseline": baseline,
        "proposal": proposal,
        "applied_deltas": applied_deltas,
        "registered_only_deltas": registered_only,
        "confidence": _proposal_confidence_label(baseline_result.confidence, deltas),
        "warnings": warnings,
        "delta_counts": counts,
        "highest_maturity": highest_maturity,
    }


def _scenario_override_label() -> str:
    return "Powertrain Scenario override - does not modify VDE Setup."


def _show_technical_details() -> bool:
    return bool(st.session_state.get("pwt_show_technical", False))


def _simple_readiness_status(readiness: dict[str, Any]) -> tuple[str, str]:
    status = str(readiness.get("status_label") or readiness.get("status") or "Pending")
    detail = str(readiness.get("status_detail") or "")
    mapping = {
        "Ready for ML": ("Ready", "Baseline ready."),
        "ML available with imputed features": ("Ready", "ML estimate with imputed features."),
        "Regression recommended": ("ML metadata incomplete", "Regression fallback available."),
        "Deterministic fallback recommended": ("Manual baseline", "Use manual efficiency or imported result."),
        "Missing critical metadata": ("Missing metadata", "Complete powertrain metadata."),
    }
    return mapping.get(status, (status, detail))


def _compact_delta_basis_label(effect_basis: str | None) -> str:
    mapping = {
        "fuel_percent_delta": "Fuel consumption percent delta",
        "pse_percent_delta": "PSE percent delta",
        "co2_percent_delta": "CO2 percent delta",
        "efficiency_multiplier": "Efficiency multiplier",
        "metadata_only": "Metadata-only / registered-only",
        "fuel percent delta": "Fuel consumption percent delta",
        "PSE multiplier": "Efficiency multiplier",
        "metadata only": "Metadata-only / registered-only",
    }
    return mapping.get(str(effect_basis or "").strip(), str(effect_basis or "-"))


def _normalize_delta_effect_basis(effect_basis: str | None) -> str:
    mapping = {
        "Fuel consumption percent delta": "fuel_percent_delta",
        "PSE percent delta": "pse_percent_delta",
        "CO2 percent delta": "co2_percent_delta",
        "Efficiency multiplier": "efficiency_multiplier",
        "Metadata-only / registered-only": "metadata_only",
        "fuel delta": "fuel_delta",
        "fuel percent delta": "fuel_percent_delta",
        "PSE delta": "pse_delta",
        "PSE multiplier": "pse_multiplier",
        "CO2 percent delta": "co2_percent_delta",
        "CO2 delta": "co2_delta",
        "energy delta": "energy_delta",
        "efficiency multiplier": "efficiency_multiplier",
        "metadata only": "metadata_only",
        "map-based effect": "map_based_effect",
    }
    return mapping.get(str(effect_basis or "").strip(), str(effect_basis or "").strip())


def _draft_delta_from_form() -> dict[str, Any] | None:
    basis = _normalize_delta_effect_basis(st.session_state.get("pwt_delta_effect_basis"))
    name = str(st.session_state.get("pwt_delta_name") or "").strip()
    value = to_float(st.session_state.get("pwt_delta_value"))
    enabled = bool(st.session_state.get("pwt_delta_apply_toggle"))
    if not name and basis in {"metadata_only", ""} and (value in (None, 0.0)) and not str(st.session_state.get("pwt_delta_notes") or "").strip():
        return None
    return {
        "id": len(list(st.session_state.get("pwt_technology_deltas") or [])) + 1,
        "name": name or "Draft delta",
        "affected_subsystem": st.session_state.get("pwt_delta_subsystem") or "whole powertrain",
        "source_type": st.session_state.get("pwt_delta_source_type") or "manual",
        "maturity_level": st.session_state.get("pwt_delta_maturity") or "engineering_assumption",
        "effect_basis": basis or "metadata_only",
        "confidence": st.session_state.get("pwt_delta_confidence") or "unknown",
        "effect_value": value,
        "enabled": enabled,
        "notes": str(st.session_state.get("pwt_delta_notes") or "").strip(),
        "reference_description": str(st.session_state.get("pwt_delta_reference") or "").strip(),
        "is_preview_only": True,
    }


def _delta_basis_select_options() -> list[str]:
    options = list(DELTA_EFFECT_BASIS_OPTIONS)
    if _show_technical_details():
        options.extend(DELTA_EFFECT_BASIS_ADVANCED_OPTIONS)
    current = _compact_delta_basis_label(st.session_state.get("pwt_delta_effect_basis"))
    if current not in options and current != "-":
        options.append(current)
    return options


def _format_delta_change(
    baseline_value: float | None,
    proposal_value: float | None,
    *,
    format_str: str,
    suffix: str = "",
) -> str:
    if baseline_value is None or proposal_value is None:
        return "-"
    absolute_delta = proposal_value - baseline_value
    delta_text = f"{absolute_delta:{format_str}}{suffix}"
    if baseline_value == 0:
        return delta_text
    percent_delta = (absolute_delta / baseline_value) * 100.0
    return f"{delta_text} / {percent_delta:+.1f}%"


def _build_baseline_proposal_rows(
    baseline_metrics: dict[str, Any],
    proposal_metrics: dict[str, Any],
) -> list[dict[str, str]]:
    specs = [
        (f"Fuel [{_fuel_display_unit()}]", "fuel_l_100km"),
        ("PSE", "pse"),
        (f"CO2 [{unit_label('co2_per_distance', _current_unit_system())}]", "gco2_km"),
        (f"Energy [{unit_label('energy_wh_per_distance', _current_unit_system())}]", "energy_Wh_km"),
    ]
    rows: list[dict[str, str]] = []
    for label, key in specs:
        baseline_value = to_float(baseline_metrics.get(key))
        proposal_value = to_float(proposal_metrics.get(key))
        if baseline_value is None and proposal_value is None:
            continue
        if key == "fuel_l_100km":
            baseline_text = _format_fuel_value(baseline_value)
            proposal_text = _format_fuel_value(proposal_value)
            delta_text = "-" if baseline_value is None or proposal_value is None else _format_fuel_value(proposal_value - baseline_value)
        elif key == "gco2_km":
            baseline_text = _format_co2_value(baseline_value)
            proposal_text = _format_co2_value(proposal_value)
            delta_text = "-" if baseline_value is None or proposal_value is None else _format_co2_value(proposal_value - baseline_value)
        elif key == "energy_Wh_km":
            baseline_text = _format_energy_value(baseline_value)
            proposal_text = _format_energy_value(proposal_value)
            delta_text = "-" if baseline_value is None or proposal_value is None else _format_energy_value(proposal_value - baseline_value)
        else:
            baseline_text = _format_metric_value(baseline_value, format_str="%.3f")
            proposal_text = _format_metric_value(proposal_value, format_str="%.3f")
            delta_text = "-" if baseline_value is None or proposal_value is None else _format_metric_value(proposal_value - baseline_value, format_str="%.3f")
        rows.append(
            {
                "Metric": label,
                "Baseline": baseline_text,
                "Proposal": proposal_text,
                "Delta": delta_text,
                "Delta %": (
                    "-"
                    if baseline_value in (None, 0) or proposal_value is None
                    else f"{((proposal_value - baseline_value) / baseline_value) * 100.0:+.1f}%"
                ),
            }
        )
    return rows


def _metadata_chip_items(readiness: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
    values = dict(readiness.get("values") or {})
    items = [
        f"Fuel: {_format_feature_value(values.get('fuel_type'), feature_key='fuel_type')}",
        f"Electrification: {str(ctx.get('electrification') or '-')}",
        f"Engine: {_format_feature_value(values.get('engine_size_l'), feature_key='engine_size_l')} L" if values.get("engine_size_l") not in (None, "") else "Engine: missing",
        f"Power: {_format_feature_value(values.get('engine_max_power_kw'), feature_key='engine_max_power_kw')}",
        f"Transmission: {_format_feature_value(values.get('transmission_type'), feature_key='transmission_type')}",
        f"Drive: {_format_feature_value(values.get('drive_type'), feature_key='drive_type')}",
        f"Gear count: {_format_feature_value(values.get('gear_count'), feature_key='gear_count')}",
        f"Final drive: {_format_feature_value(values.get('final_drive_ratio'), feature_key='final_drive_ratio')}",
    ]
    return items


def _compact_metadata_line(readiness: dict[str, Any], ctx: dict[str, Any], *, max_items: int = 8) -> str:
    items = [item for item in _metadata_chip_items(readiness, ctx) if item]
    return " | ".join(items[:max_items]) if items else "Metadata pending"


def _powertrain_reference_empty_state(reference_summary: dict[str, Any]) -> tuple[str, list[str]]:
    source_type = str(reference_summary.get("source_type") or "")
    if source_type == "Same vehicle fuelcons_db line":
        return (
            "No same-vehicle observed data found.",
            [
                "Choose another fuelcons_db line",
                "Use regression baseline",
                "Create new scenario",
            ],
        )
    return (
        "Missing reference",
        [
            "Choose baseline source",
            "Use regression baseline",
            "Create new scenario",
        ],
    )


def _baseline_summary_empty_state(reference_summary: dict[str, Any]) -> str:
    if str(reference_summary.get("source_type") or "") == "Same vehicle fuelcons_db line":
        return "Missing reference"
    return "Choose baseline source"


def _metadata_field_note(feature_key: str) -> str:
    notes = {
        "fuel_type": "Local Powertrain Scenario override.",
        "electrification": "Scenario-only override.",
        "engine_size_l": "Critical for ML coverage.",
        "engine_max_power_kw": "Stored/displayed as power_hp in the editor.",
        "transmission_type": "Used by ML / regression / peers.",
        "gear_count": "Scenario-only override.",
        "drive_type": "Used by ML / regression / peers.",
        "final_drive_ratio": "Scenario-only override.",
        "mass_kg": "Edit mass in VDE Setup.",
        "test_mass_kg": "Edit mass in VDE Setup.",
        "category": "Used by ML / regression / peers.",
    }
    return notes.get(feature_key, "")


def _build_powertrain_metadata_editor_df(
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    readiness: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    values = dict(readiness.get("values") or {})
    sources = dict(readiness.get("sources") or {})
    importance_map = {field["key"]: field["importance"] for field in FEATURE_READINESS_FIELDS}
    label_map = {field["key"]: field["label"] for field in FEATURE_READINESS_FIELDS}
    ordered_keys = [
        "fuel_type",
        "electrification",
        "engine_size_l",
        "engine_max_power_kw",
        "transmission_type",
        "gear_count",
        "drive_type",
        "final_drive_ratio",
        "mass_kg",
        "test_mass_kg",
        "category",
    ]
    for key in ordered_keys:
        value = values.get(key)
        source = str(sources.get(key) or "missing")
        display_value = _format_feature_value(value, feature_key=key)
        if key == "engine_max_power_kw" and value not in (None, ""):
            display_value = _format_feature_value(value, feature_key=key)
        if source in {"missing", "imputed_later"}:
            display_value = ""
        rows.append(
            {
                "field": label_map.get(key, key),
                "value": display_value if key not in {"mass_kg", "test_mass_kg"} else _format_feature_value(value, feature_key=key),
                "source": source,
                "local_override": bool(source == "scenario_override"),
                "required_for_ml": bool(importance_map.get(key) == "critical_for_ml"),
                "notes": _metadata_field_note(key),
                "_feature_key": key,
            }
        )
    return pd.DataFrame(rows)


def _apply_powertrain_metadata_editor_df(editor_df: pd.DataFrame, vde_row: dict) -> list[str]:
    errors: list[str] = []
    if editor_df is None or editor_df.empty:
        return ["Powertrain metadata table is empty."]

    feature_rows = {
        str(row.get("_feature_key") or ""): dict(row)
        for row in editor_df.to_dict(orient="records")
    }

    def _text_value(key: str) -> str:
        return str((feature_rows.get(key) or {}).get("value") or "").strip()

    def _numeric_value(key: str) -> float | None:
        return to_float(_text_value(key))

    category = _text_value("category")
    st.session_state["pwt_feature_category"] = category or "(inherit)"

    transmission_type = _text_value("transmission_type")
    st.session_state["pwt_feature_transmission_type"] = transmission_type or "(inherit)"

    drive_type = _text_value("drive_type")
    st.session_state["pwt_feature_drive_type"] = drive_type or "(inherit)"

    fuel_type = _text_value("fuel_type")
    st.session_state["sb_fuel_type"] = fuel_type or "(leave missing)"

    electrification = _text_value("electrification").upper()
    if electrification:
        if electrification not in {"ICE", "HEV", "PHEV", "BEV"}:
            errors.append("Electrification must be ICE, HEV, PHEV, or BEV.")
        else:
            st.session_state["pwt_scenario_electrification"] = electrification

    engine_size_l = _numeric_value("engine_size_l")
    st.session_state["pwt_feature_engine_size_l"] = "" if engine_size_l is None else f"{engine_size_l:.3f}".rstrip("0").rstrip(".")
    if _text_value("engine_size_l") and engine_size_l is None:
        errors.append("Engine size must be numeric.")

    power_hp = _numeric_value("engine_max_power_kw")
    st.session_state["pwt_feature_power_hp"] = "" if power_hp is None else f"{power_hp:.0f}"
    if _text_value("engine_max_power_kw") and power_hp is None:
        errors.append("Power [hp] must be numeric.")

    gear_count = _numeric_value("gear_count")
    st.session_state["pwt_gears"] = "" if gear_count is None else str(int(round(gear_count)))
    if _text_value("gear_count") and gear_count is None:
        errors.append("Gear count must be numeric.")

    final_drive_ratio = _numeric_value("final_drive_ratio")
    st.session_state["pwt_fdr"] = "" if final_drive_ratio is None else f"{final_drive_ratio:.3f}".rstrip("0").rstrip(".")
    if _text_value("final_drive_ratio") and final_drive_ratio is None:
        errors.append("Final drive ratio must be numeric.")

    if _text_value("mass_kg") and _text_value("mass_kg") != _format_feature_value(vde_row.get("mass_kg"), feature_key="mass_kg"):
        errors.append("Mass is review-only here. Edit mass in VDE Setup.")
    if _text_value("test_mass_kg") and _text_value("test_mass_kg") != _format_feature_value(vde_row.get("test_mass_kg"), feature_key="test_mass_kg"):
        errors.append("Test mass is review-only here. Edit mass in VDE Setup.")

    critical_missing = []
    for key in ("engine_size_l", "drive_type", "gear_count", "final_drive_ratio"):
        if feature_rows.get(key) and not _text_value(key):
            critical_missing.append(str((feature_rows.get(key) or {}).get("field") or key))
    if critical_missing:
        errors.append("Critical metadata still missing: " + ", ".join(critical_missing) + ".")
    return errors


def _build_manual_baseline_editor_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fuel_l_100km": to_float(st.session_state.get("pwt_manual_fuel_l100")),
                "energy_Wh_km": to_float(st.session_state.get("pwt_manual_energy_whkm")),
                "co2_g_km": to_float(st.session_state.get("pwt_manual_gco2_km")),
                "source": str(st.session_state.get("pwt_manual_source") or "user_input"),
                "confidence": str(st.session_state.get("pwt_manual_confidence") or "unknown"),
                "status": "OK" if any(
                    (to_float(st.session_state.get(key)) or 0.0) > 0.0
                    for key in ("pwt_manual_fuel_l100", "pwt_manual_energy_whkm", "pwt_manual_gco2_km")
                ) else "Pending",
                "notes": str(st.session_state.get("pwt_manual_notes") or ""),
            }
        ]
    )


def _apply_manual_baseline_editor_df(editor_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if editor_df is None or editor_df.empty:
        return ["Manual baseline table is empty."]
    row = dict(editor_df.to_dict(orient="records")[0] or {})
    fuel = to_float(row.get("fuel_l_100km"))
    energy = to_float(row.get("energy_Wh_km"))
    co2 = to_float(row.get("co2_g_km"))
    if fuel is not None and fuel <= 0:
        errors.append("Manual baseline fuel must be positive.")
    if energy is not None and energy <= 0:
        errors.append("Manual baseline energy must be positive.")
    if co2 is not None and co2 < 0:
        errors.append("Manual baseline CO2 cannot be negative.")
    st.session_state["pwt_manual_fuel_l100"] = 0.0 if fuel is None else float(fuel)
    st.session_state["pwt_manual_energy_whkm"] = 0.0 if energy is None else float(energy)
    st.session_state["pwt_manual_gco2_km"] = 0.0 if co2 is None else float(co2)
    st.session_state["pwt_manual_source"] = str(row.get("source") or "user_input").strip() or "user_input"
    st.session_state["pwt_manual_confidence"] = str(row.get("confidence") or "unknown").strip() or "unknown"
    st.session_state["pwt_manual_notes"] = str(row.get("notes") or "").strip()
    return errors


def _build_delta_editor_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "delta_label": str(st.session_state.get("pwt_delta_name") or ""),
                "delta_basis": _compact_delta_basis_label(st.session_state.get("pwt_delta_effect_basis")),
                "delta_value": to_float(st.session_state.get("pwt_delta_value")),
                "apply_quantitatively": bool(st.session_state.get("pwt_delta_apply_toggle", True)),
                "source_method": str(st.session_state.get("pwt_delta_source_type") or "manual"),
                "confidence": str(st.session_state.get("pwt_delta_confidence") or "unknown"),
                "status": "Draft" if bool(st.session_state.get("pwt_delta_apply_toggle", True)) else "Registered only",
                "notes": str(st.session_state.get("pwt_delta_notes") or ""),
            }
        ]
    )


def _apply_delta_editor_df(editor_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if editor_df is None or editor_df.empty:
        return ["Technology delta table is empty."]
    row = dict(editor_df.to_dict(orient="records")[0] or {})
    basis = _normalize_delta_effect_basis(row.get("delta_basis"))
    value = to_float(row.get("delta_value"))
    apply_quantitatively = bool(row.get("apply_quantitatively"))
    st.session_state["pwt_delta_name"] = str(row.get("delta_label") or "").strip()
    st.session_state["pwt_delta_effect_basis"] = basis or "metadata_only"
    st.session_state["pwt_delta_value"] = 0.0 if value is None else float(value)
    st.session_state["pwt_delta_apply_toggle"] = apply_quantitatively
    st.session_state["pwt_delta_confidence"] = str(row.get("confidence") or "unknown").strip() or "unknown"
    st.session_state["pwt_delta_notes"] = str(row.get("notes") or "").strip()
    if apply_quantitatively and basis != "metadata_only" and value is None:
        errors.append("Technology delta value is required when quantitative application is enabled.")
    return errors


def _render_powertrain_metadata_review(
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    readiness: dict[str, Any],
    *,
    expanded: bool = False,
    editable: bool = True,
) -> None:
    st.caption("Scenario override - does not modify VDE Setup or source fuelcons line.")
    readiness_df = pd.DataFrame(readiness["rows"])
    if not readiness_df.empty:
        st.dataframe(readiness_df, use_container_width=True, hide_index=True)
    if not editable:
        values = dict(readiness.get("values") or {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", _format_feature_value(values.get("category"), feature_key="category"))
        c2.metric("Transmission", _format_feature_value(values.get("transmission_type"), feature_key="transmission_type"))
        c3.metric("Drive type", _format_feature_value(values.get("drive_type"), feature_key="drive_type"))
        d1, d2, d3 = st.columns(3)
        d1.metric("Engine size [L]", _format_feature_value(values.get("engine_size_l"), feature_key="engine_size_l"))
        d2.metric("Gear count", _format_feature_value(values.get("gear_count"), feature_key="gear_count"))
        d3.metric("Final drive ratio", _format_feature_value(values.get("final_drive_ratio"), feature_key="final_drive_ratio"))
        e1, e2 = st.columns(2)
        e1.metric("Power [hp]", _format_feature_value(values.get("engine_max_power_kw"), feature_key="engine_max_power_kw"))
        e2.metric("Fuel type", _format_feature_value(values.get("fuel_type"), feature_key="fuel_type"))
        st.caption("Edit these overrides in Scenario Pairing.")
        st.caption(_scenario_override_label())
        return
    metadata_df = _build_powertrain_metadata_editor_df(vde_id, vde_row, ctx, readiness)
    edited_df = st.data_editor(
        metadata_df,
        key="pwt_metadata_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["field", "source", "local_override", "required_for_ml", "notes", "_feature_key"],
        column_config={
            "field": st.column_config.TextColumn("field"),
            "value": st.column_config.TextColumn("value"),
            "source": st.column_config.TextColumn("source"),
            "local_override": st.column_config.CheckboxColumn("local_override"),
            "required_for_ml": st.column_config.CheckboxColumn("required_for_ml"),
            "notes": st.column_config.TextColumn("notes"),
            "_feature_key": None,
        },
    )
    metadata_errors = _apply_powertrain_metadata_editor_df(edited_df, vde_row)
    for error in metadata_errors:
        st.warning(error)
    st.caption(_scenario_override_label())


def _render_reference_rebase_explanation(
    *,
    reference_summary: dict[str, Any],
    ctx: dict[str, Any],
    baseline_fuel: float | None,
    baseline_pse: float | None,
) -> None:
    st.markdown("#### Reference observed vs rebased baseline")
    rows = [
        {
            "item": "Reference observed",
            "fuel_l_100km": to_float(reference_summary.get("observed_fuel")),
            "energy_Wh_km": to_float(reference_summary.get("observed_energy_Wh_km")),
            "co2_g_km": to_float(reference_summary.get("observed_co2")),
            "pse": to_float(reference_summary.get("observed_pse")),
            "source/method": "fuelcons_db reference",
            "status": "OK" if to_float(reference_summary.get("observed_pse")) is not None else "Review",
        },
        {
            "item": "Rebased baseline",
            "fuel_l_100km": baseline_fuel,
            "energy_Wh_km": None,
            "co2_g_km": None,
            "pse": baseline_pse,
            "source/method": f"active {str(ctx.get('energy_basis') or '-')} + reference PSE",
            "status": "OK" if baseline_fuel is not None and baseline_pse is not None else "Pending",
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Observed reference -> derive PSE -> apply to active VDE.")


def _metadata_status_summary(readiness: dict[str, Any]) -> tuple[int, int, list[str]]:
    rows = list(readiness.get("rows") or [])
    total = len(rows)
    complete = 0
    missing_labels: list[str] = []
    for row in rows:
        source = str(row.get("source") or "")
        label = str(row.get("feature") or "")
        if source not in {"missing", "imputed_later"}:
            complete += 1
        else:
            if label:
                missing_labels.append(label)
    return complete, total, missing_labels[:3]


def _render_feature_readiness_highlight(readiness: dict[str, Any]) -> None:
    complete, total, missing_labels = _metadata_status_summary(readiness)
    status = str(readiness.get("status_label") or "-")
    detail = str(readiness.get("status_detail") or "")
    missing_count = len(list(readiness.get("critical_ml_missing") or [])) + len(list(readiness.get("regression_useful_missing") or []))

    if missing_labels:
        st.warning(
            "Complete scenario metadata before estimation: "
            + ", ".join(missing_labels)
            + "."
        )
        c1, c2 = st.columns([1.4, 1])
        c1.caption(f"{status} | {complete}/{total} features ready")
        c2.caption(f"{missing_count} feature(s) still need attention")
    else:
        st.success("Scenario metadata is ready for estimation.")
        st.caption(f"{status} | {complete}/{total} features ready")

    if detail:
        st.caption(detail)


def _baseline_pending_message(
    *,
    active_method: str,
    observed_available: bool,
    readiness: dict[str, Any],
    ml_state: dict[str, Any],
    regression_state: dict[str, Any],
) -> str:
    if active_method == "Observed / Derived PSE" and not observed_available:
        return "Observed reference PSE unavailable - choose another reference, use regression, assume efficiency, or enter an observed/imported result."
    if active_method == "ML Prediction":
        if regression_state.get("status") in {"Recommended fallback", "Available"}:
            return "ML prediction is not ready for this scenario. Use regression estimate or complete Scenario Pairing metadata."
        return "ML prediction is not ready for this scenario. Complete Scenario Pairing metadata, assume efficiency, or enter an observed/imported result."
    if active_method == "Regression":
        return "Regression estimate is pending. Review the peer filters, assume efficiency, or enter an observed/imported result."
    if readiness.get("status_label") in {"ML available with imputed features", "Regression recommended"}:
        return "Scenario Pairing still has missing metadata. Complete the overrides you need, or continue with the available fallback."
    return "Baseline unavailable. Choose a reference, run regression estimate, assume efficiency, or enter an observed/imported result."


def _current_proposal_status_label(draft: dict[str, Any]) -> str:
    proposal = dict(draft.get("proposal_result") or {})
    deltas = list(draft.get("technology_deltas") or [])
    proposal_metrics = dict(proposal.get("proposal") or {})
    if not deltas:
        if proposal_metrics.get("fuel_l_100km") is not None:
            return "Same as baseline"
        return "Pending"
    if proposal_metrics.get("fuel_l_100km") is not None:
        return _format_metric_value(proposal_metrics.get("fuel_l_100km"), format_str="%.2f", suffix=" L/100km")
    return str(proposal.get("status") or "Pending")


def _render_baseline_side_options_panel(
    *,
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    readiness: dict[str, Any],
    regression_vde: float | None,
    reference_summary: dict[str, Any],
) -> None:
    st.markdown("#### Baseline Method")
    render_powertrain_method_cards(vde_id, vde_row, ctx, regression_vde)
    st.caption("Choose how the baseline is produced. Complete scenario metadata in the highlighted block below when needed.")


def render_powertrain_technical_footer(vde_id: int, vde_row: dict) -> None:
    ctx = get_build_scenario_context(vde_id, vde_row)
    reference_summary = _selected_powertrain_reference(vde_id, vde_row)
    regression_vde = ctx.get("energy_value_mj_per_km")
    readiness = _scenario_feature_readiness_snapshot(
        vde_id,
        vde_row,
        ctx,
        regression_vde=regression_vde,
        reference_summary=reference_summary,
    )
    draft = _build_powertrain_scenario_draft(vde_id, vde_row)
    show_technical = _show_technical_details()
    energy_values = resolve_vde_energy_values(vde_row)
    active_method = _resolve_active_pwt_setup_method(vde_id, vde_row, ctx)
    regression_state = _resolve_regression_state(vde_id, vde_row, ctx, regression_vde, render_filters=False) if regression_vde is not None else {}
    result = draft.get("baseline_estimate", {}).get("result")
    assumptions = dict((result.assumptions if result else {}) or {})
    proposal_summary = dict(draft.get("proposal_result") or {})

    st.markdown("#### Technical Audit")
    with st.expander("Demand audit", expanded=show_technical):
        revision_text = resolve_vde_source_revision(vde_row) or "-"
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("VDE_TOTAL", _format_metric_value(energy_values.get("vde_total_mj_per_km"), format_str="%.4f", suffix=" MJ/km"))
        d2.metric("VDE_NET", _format_metric_value(energy_values.get("vde_net_mj_per_km"), format_str="%.4f", suffix=" MJ/km"))
        d3.metric("Mass basis", "test_mass_kg" if vde_row.get("test_mass_kg") not in (None, "") else "mass_kg")
        d4.metric("Revision", revision_text)
        d5, d6, d7 = st.columns(3)
        d5.metric("ABC A", _format_metric_value(vde_row.get("coast_A_N"), format_str="%.2f"))
        d6.metric("ABC B", _format_metric_value(vde_row.get("coast_B_N_per_kph"), format_str="%.4f"))
        d7.metric("ABC C", _format_metric_value(vde_row.get("coast_C_N_per_kph2"), format_str="%.5f"))
        st.caption("Source state: " + ("NET available" if energy_values.get("vde_net_mj_per_km") is not None else "TOTAL only"))

    with st.expander("Baseline source audit", expanded=False):
        st.write(reference_summary)

    with st.expander("Metadata audit", expanded=False):
        _render_powertrain_metadata_review(vde_id, vde_row, ctx, readiness, expanded=False, editable=False)

    with st.expander("Feature readiness", expanded=False):
        st.dataframe(pd.DataFrame(readiness.get("rows") or []), use_container_width=True, hide_index=True)

    with st.expander("ML diagnostics / regression / peers / SHAP", expanded=show_technical):
        m1, m2, m3 = st.columns(3)
        ml_state = _ml_method_option_state(vde_id, vde_row, ctx, regression_vde)
        reg_state = _regression_method_option_state(vde_id, vde_row, ctx, regression_vde)
        m1.metric("Active method", _pwt_method_label(active_method))
        m2.metric("ML", str(ml_state.get("status") or "-"))
        m3.metric("Regression", str(reg_state.get("status") or "-"))
        if regression_state:
            dataset = regression_state.get("dataset")
            st.write(
                {
                    "regression_rows": len(dataset) if dataset is not None else 0,
                    "regression_warnings": regression_state.get("warnings"),
                    "regression_model": regression_state.get("model"),
                }
            )
        if assumptions:
            st.write(
                {
                    "shap_status": assumptions.get("shap_status"),
                    "nearest_peers": assumptions.get("nearest_peers"),
                    "peer_group_quality": assumptions.get("peer_group_quality"),
                    "integration_status": assumptions.get("integration_status"),
                }
            )

    with st.expander("Delta provenance", expanded=False):
        st.write(
            {
                "applied_deltas": proposal_summary.get("applied_deltas"),
                "registered_only_deltas": proposal_summary.get("registered_only_deltas"),
                "delta_counts": proposal_summary.get("delta_counts"),
            }
        )

    with st.expander("Scenario/save provenance", expanded=False):
        st.write(
            {
                "powertrain_reference": draft.get("powertrain_reference"),
                "baseline_estimate": draft.get("baseline_estimate"),
                "proposal_result": draft.get("proposal_result"),
                "provenance": draft.get("provenance"),
            }
        )


def _format_feature_value(value: Any, *, feature_key: str | None = None) -> str:
    if value in (None, ""):
        return "-"
    if feature_key == "engine_max_power_kw":
        hp = float(value) * 1.34102209
        return f"{hp:.0f} hp"
    numeric = to_float(value)
    if numeric is None:
        return str(value)
    if feature_key in {"mass_kg", "test_mass_kg", "gear_count"}:
        return str(int(round(numeric)))
    if feature_key in {"engine_size_l", "final_drive_ratio"}:
        return f"{numeric:.3f}"
    return f"{numeric:.2f}"


def _readiness_action_for_source(source: str) -> str:
    mapping = {
        "scenario_override": "override",
        "inherited_from_vde": "confirm",
        "missing": "leave missing",
        "imputed_later": "leave missing",
    }
    return mapping.get(source, "leave missing")


def _readiness_action_for_feature(feature_key: str, source: str) -> str:
    if feature_key in {"mass_kg", "test_mass_kg"}:
        return "confirm" if source == "inherited_from_vde" else "leave missing"
    return _readiness_action_for_source(source)


def _feature_importance_label(importance: str) -> str:
    return str(importance or "optional")


def _scenario_default_value(feature_key: str, vde_id: int, vde_row: dict, ctx: Dict[str, Any]) -> Any:
    if feature_key == "mass_kg":
        return vde_row.get("mass_kg")
    if feature_key == "test_mass_kg":
        return vde_row.get("test_mass_kg")
    if feature_key == "category":
        return vde_row.get("category")
    if feature_key == "electrification":
        return default_electrification_from_vde(vde_id)
    if feature_key == "fuel_type":
        return None
    if feature_key == "transmission_type":
        return vde_row.get("transmission_type")
    if feature_key == "drive_type":
        return vde_row.get("drive_type")
    if feature_key == "engine_max_power_kw":
        return None
    if feature_key == "engine_size_l":
        return vde_row.get("engine_size_l")
    if feature_key == "gear_count":
        return vde_row.get("gear_count")
    if feature_key == "final_drive_ratio":
        return vde_row.get("final_drive_ratio")
    return ctx.get(feature_key)


def _scenario_override_value(feature_key: str, vde_id: int, vde_row: dict, ctx: Dict[str, Any]) -> Any:
    def _same_numeric(left: Any, right: Any) -> bool:
        left_num = to_float(left)
        right_num = to_float(right)
        if left_num is None or right_num is None:
            return False
        return abs(float(left_num) - float(right_num)) < 1e-9

    if feature_key == "electrification":
        current = str(ctx.get("electrification") or "").upper() or None
        default = str(default_electrification_from_vde(vde_id) or "").upper() or None
        return current if current and current != default else None
    if feature_key == "fuel_type":
        current = st.session_state.get("sb_fuel_type")
        return str(current).strip() if current not in (None, "") else None
    if feature_key == "engine_max_power_kw":
        power_hp = to_float(st.session_state.get("pwt_feature_power_hp"))
        return None if power_hp is None else float(power_hp) / 1.34102209
    if feature_key == "gear_count":
        current = st.session_state.get("pwt_gears")
        if current in (None, ""):
            return None
        return None if _same_numeric(current, vde_row.get("gear_count")) else int(current)
    if feature_key == "final_drive_ratio":
        current = to_float(st.session_state.get("pwt_fdr"))
        return None if _same_numeric(current, vde_row.get("final_drive_ratio")) else current
    if feature_key == "engine_size_l":
        current = to_float(st.session_state.get("pwt_feature_engine_size_l"))
        return None if _same_numeric(current, vde_row.get("engine_size_l")) else current
    if feature_key == "category":
        current = st.session_state.get("pwt_feature_category")
        if current in (None, "", "(inherit)"):
            return None
        current = str(current).strip()
        return None if current == str(vde_row.get("category") or "").strip() else current
    if feature_key == "transmission_type":
        current = st.session_state.get("pwt_feature_transmission_type")
        if current in (None, "", "(inherit)"):
            return None
        current = str(current).strip()
        return None if current == str(vde_row.get("transmission_type") or "").strip() else current
    if feature_key == "drive_type":
        current = st.session_state.get("pwt_feature_drive_type")
        if current in (None, "", "(inherit)"):
            return None
        current = str(current).strip()
        return None if current == str(vde_row.get("drive_type") or "").strip() else current
    return None


def _scenario_feature_readiness_snapshot(
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    *,
    regression_vde: float | None = None,
    reference_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    overrides: dict[str, Any] = {}
    reference_metadata = dict((reference_summary or {}).get("metadata") or {})

    for field in FEATURE_READINESS_FIELDS:
        key = field["key"]
        default_value = _scenario_default_value(key, vde_id, vde_row, ctx)
        reference_value = reference_metadata.get(key)
        inherited_source = _readiness_source_label(key, reference_summary)
        override_value = _scenario_override_value(key, vde_id, vde_row, ctx)
        if override_value not in (None, ""):
            value = override_value
            source = "scenario_override"
            overrides[key] = override_value
        elif reference_value not in (None, ""):
            value = reference_value
            source = inherited_source or "inherited_from_fuelcons"
        elif default_value not in (None, ""):
            value = default_value
            source = "inherited_from_vde"
        elif field.get("allow_imputed"):
            value = None
            source = "imputed_later"
        else:
            value = None
            source = "missing"

        values[key] = value
        sources[key] = source
        rows.append(
            {
                "feature": field["label"],
                "current_value": _format_feature_value(value, feature_key=key),
                "source": source,
                "importance": _feature_importance_label(str(field.get("importance") or "optional")),
                "action": _readiness_action_for_feature(key, source),
            }
        )

    critical_ml_missing = [
        field["key"]
        for field in FEATURE_READINESS_FIELDS
        if field["importance"] == "critical_for_ml" and sources.get(field["key"]) in {"missing", "imputed_later"}
    ]
    regression_useful_missing = [
        field["key"]
        for field in FEATURE_READINESS_FIELDS
        if field["importance"] == "useful_for_regression" and sources.get(field["key"]) in {"missing", "imputed_later"}
    ]
    peer_useful_missing = [
        field["key"]
        for field in FEATURE_READINESS_FIELDS
        if field["importance"] == "useful_for_peers" and sources.get(field["key"]) in {"missing", "imputed_later"}
    ]

    regression_can_fit = False
    if regression_vde is not None:
        regression_state = _resolve_regression_state(vde_id, vde_row, ctx, regression_vde, render_filters=False)
        regression_can_fit = bool((regression_state.get("sample_quality") or {}).get("can_fit"))

    if ctx.get("energy_value_mj_per_km") is None or sources.get("category") == "missing" or sources.get("electrification") == "missing":
        status_label = "Missing critical metadata"
        status_detail = "Resolve vehicle demand and core scenario classification before estimation."
    elif not critical_ml_missing:
        status_label = "Ready for ML"
        status_detail = "The core ML feature set is confirmed in the scenario draft."
    elif regression_can_fit and len(critical_ml_missing) >= 2:
        status_label = "Regression recommended"
        status_detail = "Regression can run with the current scenario while ML-critical features remain incomplete."
    elif critical_ml_missing:
        status_label = "ML available with imputed features"
        status_detail = "Complete powertrain metadata before running ML."
    elif regression_can_fit:
        status_label = "Regression recommended"
        status_detail = "Peer sample quality is stronger than the current feature coverage."
    else:
        status_label = "Deterministic fallback recommended"
        status_detail = "Use an engineering assumption until the scenario feature set is completed."

    confidence_impacts = sorted(set(critical_ml_missing + regression_useful_missing + peer_useful_missing))
    return {
        "rows": rows,
        "values": values,
        "sources": sources,
        "overrides": overrides,
        "critical_ml_missing": critical_ml_missing,
        "regression_useful_missing": regression_useful_missing,
        "peer_useful_missing": peer_useful_missing,
        "confidence_impacts": confidence_impacts,
        "status_label": status_label,
        "status_detail": status_detail,
        "regression_can_fit": regression_can_fit,
    }


def _apply_scenario_feature_overrides(
    request: FuelEstimateRequest,
    *,
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    reference_summary: dict[str, Any] | None = None,
) -> FuelEstimateRequest:
    snapshot = _scenario_feature_readiness_snapshot(
        vde_id,
        vde_row,
        ctx,
        regression_vde=ctx.get("energy_value_mj_per_km"),
        reference_summary=reference_summary,
    )
    vehicle = dict(request.vehicle_features or {})
    powertrain = dict(request.powertrain_features or {})
    values = dict(snapshot["values"])
    sources = dict(snapshot["sources"])

    for key in ("mass_kg", "test_mass_kg", "category", "electrification", "transmission_type", "drive_type", "engine_size_l"):
        if values.get(key) not in (None, ""):
            vehicle[key] = values.get(key)
    for key in ("fuel_type", "gear_count", "final_drive_ratio", "engine_max_power_kw", "engine_size_l", "transmission_type", "drive_type"):
        if values.get(key) not in (None, ""):
            powertrain[key] = values.get(key)

    vehicle["scenario_feature_sources"] = sources
    vehicle["scenario_feature_values"] = values
    vehicle["scenario_feature_overrides"] = dict(snapshot["overrides"])
    vehicle["scenario_feature_missing"] = [
        key for key, source in sources.items() if source == "missing"
    ]
    vehicle["scenario_feature_imputed"] = [
        key for key, source in sources.items() if source == "imputed_later"
    ]
    vehicle["scenario_feature_confidence_impacts"] = list(snapshot["confidence_impacts"])
    vehicle["scenario_feature_readiness"] = {
        "status_label": snapshot["status_label"],
        "status_detail": snapshot["status_detail"],
        "critical_ml_missing": list(snapshot["critical_ml_missing"]),
        "regression_useful_missing": list(snapshot["regression_useful_missing"]),
        "peer_useful_missing": list(snapshot["peer_useful_missing"]),
    }
    request.vehicle_features = vehicle
    request.powertrain_features = powertrain
    return request


def _build_powertrain_scenario_draft(vde_id: int, vde_row: dict) -> dict[str, Any]:
    ctx = get_build_scenario_context(vde_id, vde_row)
    reference_summary = _selected_powertrain_reference(vde_id, vde_row)
    readiness = _scenario_feature_readiness_snapshot(
        vde_id,
        vde_row,
        ctx,
        regression_vde=ctx.get("energy_value_mj_per_km"),
        reference_summary=reference_summary,
    )
    active_method = _resolve_active_pwt_setup_method(vde_id, vde_row, ctx)
    confirmed_snapshot = _confirmed_baseline_snapshot(vde_id)
    baseline_method = _effective_baseline_method(active_method, _confirmed_pwt_setup_method())
    if confirmed_snapshot is not None:
        baseline_method = str(confirmed_snapshot.get("method") or baseline_method)
        baseline_request = confirmed_snapshot.get("request")
        baseline_result = confirmed_snapshot.get("result")
        baseline_reference = dict(confirmed_snapshot.get("reference_summary") or reference_summary)
    else:
        baseline_request = _build_active_fuel_estimate_request(
            vde_id,
            vde_row,
            ctx,
            ctx.get("energy_value_mj_per_km"),
            method_label=baseline_method,
            reference_summary=reference_summary,
        )
        baseline_result = run_fuel_estimation(baseline_request) if baseline_request is not None else None
        baseline_reference = reference_summary
    deltas = _technology_deltas(include_form_preview=True)
    proposal_result = _apply_delta_stack_to_baseline(baseline_result, ctx=ctx, deltas=deltas)
    metadata_values = dict(readiness.get("values") or {})
    metadata_sources = dict(readiness.get("sources") or {})
    metadata_rows = []
    for field in FEATURE_READINESS_FIELDS:
        key = field["key"]
        metadata_rows.append(
            {
                "feature": field["label"],
                "value": _format_feature_value(metadata_values.get(key), feature_key=key),
                "source": metadata_sources.get(key, "missing"),
                "readiness": field["importance"],
                "action": _readiness_action_for_feature(key, metadata_sources.get(key, "missing")),
            }
        )

    return {
        "ctx": ctx,
        "vde_source": {
            "vde_id": int(vde_id),
            "vehicle_label": f"{str(vde_row.get('make') or '-')} {str(vde_row.get('model') or '-')}".strip(),
            "cycle": vde_row.get("cycle_name"),
            "demand_basis": ctx.get("energy_basis"),
            "vde_total": resolve_vde_energy_values(vde_row).get("vde_total_mj_per_km"),
            "vde_net": resolve_vde_energy_values(vde_row).get("vde_net_mj_per_km"),
            "selected_demand": ctx.get("energy_value_mj_per_km"),
            "mass_basis": "test_mass_kg" if vde_row.get("test_mass_kg") not in (None, "") else "mass_kg",
            "revision": resolve_vde_source_revision(vde_row),
        },
        "powertrain_reference": reference_summary,
        "metadata": {
            "values": metadata_values,
            "sources": metadata_sources,
            "readiness_rows": metadata_rows,
            "overrides": dict(readiness.get("overrides") or {}),
        },
        "feature_readiness": {
            "status": readiness.get("status_label"),
            "status_detail": readiness.get("status_detail"),
            "missing_features": list(readiness.get("critical_ml_missing") or []),
            "imputed_features": list(readiness.get("regression_useful_missing") or []) + list(readiness.get("peer_useful_missing") or []),
            "recommended_method": active_method,
            "warnings": list(readiness.get("confidence_impacts") or []),
        },
        "baseline_estimate": {
            "method": baseline_method,
            "preview_method": active_method,
            "confirmed_method": _confirmed_pwt_setup_method(),
            "request": baseline_request,
            "result": baseline_result,
            "reference_summary": baseline_reference,
            "confidence": str((baseline_result.confidence if baseline_result else "-") or "-"),
            "warnings": list((baseline_result.warnings if baseline_result else []) or []),
            "provenance": dict(((baseline_result.assumptions if baseline_result else {}) or {}).get("confidence_summary") or {}),
        },
        "technology_deltas": deltas,
        "proposal_result": proposal_result,
        "provenance": {
            "inherited_fields": [key for key, value in metadata_sources.items() if str(value).startswith("inherited_")],
            "overridden_fields": list((readiness.get("overrides") or {}).keys()),
            "assumptions": dict((baseline_result.assumptions if baseline_result else {}) or {}),
            "warnings": list((baseline_result.warnings if baseline_result else []) or []) + list(proposal_result.get("warnings") or []),
        },
    }


def _bench_hotspot_map() -> dict[str, dict[str, str]]:
    return {item["key"]: item for item in PWT_BENCH_HOTSPOTS}


def _active_bench_hotspot() -> str:
    valid_keys = {item["key"] for item in PWT_BENCH_HOTSPOTS}
    current = str(st.session_state.get("pwt_bench_hotspot") or "driver_cycle")
    if current not in valid_keys:
        current = "driver_cycle"
        st.session_state["pwt_bench_hotspot"] = current
    return current


def _pwt_method_label(method: str | None) -> str:
    mapping = {
        "Observed / Derived PSE": "Reuse observed reference PSE",
        "manual_imported": "Enter observed/imported result",
        "physics_simple": "Assume efficiency",
        "regression_existing": "Regression estimate",
        "ml_prediction": "ML prediction",
        "Manual / Imported": "Enter observed/imported result",
        "Physics Simple": "Assume efficiency",
        "Regression": "Regression estimate",
        "ML Prediction": "ML prediction",
    }
    method_key = str(method or "").strip()
    return mapping.get(method_key, method_key or str(st.session_state.get("pwt_setup_method") or "-"))


def _pse_help_text() -> str:
    return "PSE = vehicle demand energy / consumed energy. It is derived from the current scenario result."


def _pse_pending_message(method_label: str) -> str:
    mapping = {
        "Observed / Derived PSE": "Observed reference PSE unavailable - choose another reference, use regression, assume efficiency, or enter an observed/imported result.",
        "Reuse observed reference PSE": "Observed reference PSE unavailable - choose another reference, use regression, assume efficiency, or enter an observed/imported result.",
        "Physics Simple": "PSE pending - enter an assumed efficiency/PSE.",
        "Assume efficiency": "PSE pending - enter an assumed efficiency/PSE.",
        "Manual / Imported": "PSE pending - enter fuel, energy, or CO2 to derive PSE afterward.",
        "Enter observed/imported result": "PSE pending - enter fuel, energy, or CO2 to derive PSE afterward.",
        "Regression": "PSE pending - run the regression estimate.",
        "Regression estimate": "PSE pending - run the regression estimate.",
        "ML Prediction": "PSE pending - run the ML prediction.",
        "ML prediction": "PSE pending - run the ML prediction.",
        "Physics + ML Residual": "PSE pending - this method is planned and cannot produce a result yet.",
        "Map-Based Simulation": "PSE pending - this method is planned and cannot produce a result yet.",
    }
    return mapping.get(method_label, "PSE pending - choose a conversion path.")


def _method_storyline(method_label: str) -> str:
    mapping = {
        "Observed / Derived PSE": "Observed reference -> derive PSE -> apply to active VDE.",
        "Reuse observed reference PSE": "Observed reference -> derive PSE -> apply to active VDE.",
        "Manual / Imported": "Entered result -> derive PSE for diagnostics.",
        "Enter observed/imported result": "Entered result -> derive PSE for diagnostics.",
        "Physics Simple": "Assumed efficiency -> compute fuel/CO2 from active VDE.",
        "Assume efficiency": "Assumed efficiency -> compute fuel/CO2 from active VDE.",
        "Regression": "Data-driven estimate from comparable records.",
        "Regression estimate": "Data-driven estimate from comparable records.",
        "ML Prediction": "Model prediction based on available metadata/features.",
        "ML prediction": "Model prediction based on available metadata/features.",
        "Physics + ML Residual": "Physics+ML residual and map-based simulation",
        "Map-Based Simulation": "Physics+ML residual and map-based simulation",
    }
    return mapping.get(method_label, method_label or "-")


def _final_result_pending_message(method_label: str) -> str:
    mapping = {
        "Observed / Derived PSE": "Waiting for a valid observed reference result.",
        "Reuse observed reference PSE": "Waiting for a valid observed reference result.",
        "Manual / Imported": "Waiting for an entered observed/imported result.",
        "Enter observed/imported result": "Waiting for an entered observed/imported result.",
        "Physics Simple": "Waiting for an assumed efficiency/PSE.",
        "Assume efficiency": "Waiting for an assumed efficiency/PSE.",
        "Regression": "Waiting for the regression estimate.",
        "Regression estimate": "Waiting for the regression estimate.",
        "ML Prediction": "Waiting for the ML prediction.",
        "ML prediction": "Waiting for the ML prediction.",
        "Physics + ML Residual": "Hybrid residual method is planned.",
        "Map-Based Simulation": "Map-based simulation is planned.",
    }
    return mapping.get(method_label, "Waiting for a conversion path.")


def _pse_pending_summary_detail(method_label: str) -> str:
    mapping = {
        "Observed / Derived PSE": "PSE pending - Observed reference PSE",
        "Reuse observed reference PSE": "PSE pending - Observed reference PSE",
        "ML Prediction": "PSE pending - ML prediction",
        "ML prediction": "PSE pending - ML prediction",
        "Regression": "PSE pending - Regression estimate",
        "Regression estimate": "PSE pending - Regression estimate",
        "Physics Simple": "PSE pending - Assumed efficiency",
        "Assume efficiency": "PSE pending - Assumed efficiency",
        "Manual / Imported": "PSE pending - Entered observed/imported result",
        "Enter observed/imported result": "PSE pending - Entered observed/imported result",
        "Physics + ML Residual": "PSE pending - Planned method",
        "Map-Based Simulation": "PSE pending - Planned method",
    }
    return mapping.get(method_label, "PSE pending")


def _has_imported_observed_inputs() -> bool:
    return any(
        to_float(st.session_state.get(key)) is not None
        for key in ("pwt_manual_fuel_l100", "pwt_manual_energy_whkm", "pwt_manual_gco2_km")
    )


def _confidence_preview_label(result: Any, confidence_summary: dict[str, Any], assumptions: dict[str, Any]) -> str:
    if result is None:
        return "Pending result"

    has_output = any(
        value is not None
        for value in (result.fuel_l_100km, result.energy_Wh_km, result.gco2_km)
    )
    if not has_output:
        return "Pending result"

    confidence_level = str(confidence_summary.get("level") or result.confidence or "").strip().lower()
    if confidence_level == "low":
        return "Low confidence"

    peer_quality = dict(assumptions.get("peer_group_quality") or {})
    nearest_peers = list(assumptions.get("nearest_peers") or [])
    if result.method == "ml_prediction" or peer_quality.get("label") or nearest_peers:
        return "Guidance available"

    return "Pre-check OK"


def _confidence_reason_label(
    *,
    readiness: dict[str, Any] | None = None,
    active_method: str | None = None,
    reference_summary: dict[str, Any] | None = None,
    regression_state: dict[str, Any] | None = None,
) -> str:
    status = str((readiness or {}).get("status_label") or "")
    if active_method == "Observed / Derived PSE" and to_float((reference_summary or {}).get("observed_pse")) is not None:
        if status == "Regression recommended":
            return "Observed reference PSE selected · ML metadata incomplete"
        if status == "ML available with imputed features":
            return "Observed reference PSE selected · ML metadata incomplete"
        return "Observed reference PSE selected"
    if status == "Regression recommended":
        return "Regression fallback available"
    if status == "ML available with imputed features":
        return "ML metadata incomplete"
    if regression_state and str(regression_state.get("status") or "") == "Recommended fallback":
        return "Regression fallback available"
    return str((readiness or {}).get("status_detail") or "Baseline ready.")


def _preferred_pwt_setup_method(vde_id: int, vde_row: dict, ctx: Dict[str, Any]) -> str:
    reference_summary = _selected_powertrain_reference(vde_id, vde_row)
    if to_float(reference_summary.get("observed_pse")) is not None:
        return "Observed / Derived PSE"
    regression_vde = ctx.get("energy_value_mj_per_km")
    ml_state = _ml_method_option_state(vde_id, vde_row, ctx, regression_vde)
    if ml_state["status"] == "Recommended":
        return "ML Prediction"
    regression_state = _regression_method_option_state(vde_id, vde_row, ctx, regression_vde)
    if regression_state["status"] in {"Recommended fallback", "Available"}:
        return "Regression"
    if _has_imported_observed_inputs():
        return "Manual / Imported"
    if ctx.get("energy_value_mj_per_km") is not None:
        return "Physics Simple"
    return "Manual / Imported"


def _resolve_active_pwt_setup_method(vde_id: int, vde_row: dict, ctx: Dict[str, Any]) -> str:
    current = str(st.session_state.get("pwt_setup_method") or "").strip()
    explicit = bool(st.session_state.get("pwt_setup_method_explicit"))
    if current in PWT_ESTIMATION_METHODS:
        if current != "Manual / Imported" or explicit or _has_imported_observed_inputs():
            return current

    preferred = _preferred_pwt_setup_method(vde_id, vde_row, ctx)
    st.session_state["pwt_setup_method"] = preferred
    return preferred


def _select_pwt_setup_method(method_label: str) -> None:
    st.session_state["pwt_setup_method"] = method_label
    st.session_state["pwt_setup_method_explicit"] = True


def _select_deterministic_submethod() -> None:
    choice = str(st.session_state.get("pwt_deterministic_submethod") or "Physics Efficiency Assumption")
    method_label = "Manual / Imported" if choice == "Imported / Observed Result" else "Physics Simple"
    _select_pwt_setup_method(method_label)


def _confirmed_pwt_setup_method() -> str | None:
    method = str(st.session_state.get("pwt_baseline_confirmed_method") or "").strip()
    return method if method in PWT_ESTIMATION_METHODS else None


def _confirmed_baseline_snapshot(vde_id: int | None = None) -> dict[str, Any] | None:
    snapshot = st.session_state.get("pwt_confirmed_baseline_snapshot")
    if not isinstance(snapshot, dict):
        return None
    method = str(snapshot.get("method") or "").strip()
    if method not in PWT_ESTIMATION_METHODS:
        return None
    if vde_id is not None:
        try:
            if int(snapshot.get("vde_id")) != int(vde_id):
                return None
        except Exception:
            return None
    return snapshot


def _effective_baseline_method(active_method: str, confirmed_method: str | None) -> str:
    confirmed = str(confirmed_method or "").strip()
    if confirmed in PWT_ESTIMATION_METHODS:
        return confirmed
    return active_method


def _ml_method_option_state(
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    regression_vde: float | None,
) -> dict[str, Any]:
    request = _build_active_fuel_estimate_request(
        vde_id,
        vde_row,
        ctx,
        regression_vde,
        method_label="ML Prediction",
    )
    if request is None:
        return {"status": "Unavailable", "detail": "Needs a valid scenario request.", "peer_count": 0}

    setup = describe_ml_prediction_setup(
        request,
        model_artifact_path=st.session_state.get("pwt_ml_artifact_path"),
        predictor=request.model_options.get("ml_predictor"),
    )
    readiness = _scenario_feature_readiness_snapshot(vde_id, vde_row, ctx, regression_vde=regression_vde)
    missing_count = len(setup.get("features", {}).get("missing_features") or [])
    peer_analysis = build_peer_analysis_for_request(request, n=5)
    peer_count = int((peer_analysis.get("summary") or {}).get("peer_count") or 0)
    status = str(setup.get("status") or "unknown")

    if status == "available" and missing_count == 0 and peer_count == 0:
        return {"status": "Out of domain", "detail": "Artifact is loaded, but no comparable peer context was found.", "peer_count": peer_count}
    if status == "available" and missing_count == 0:
        return {"status": "Recommended", "detail": "Artifact loaded and feature coverage is sufficient.", "peer_count": peer_count}
    if status == "available":
        return {
            "status": "Missing features",
            "detail": f"Complete powertrain metadata in Scenario Pairing. {missing_count} missing feature(s).",
            "peer_count": peer_count,
            "readiness": readiness["status_label"],
        }
    if status == "export_pending":
        return {"status": "Needs artifact", "detail": "Notebook exists, but no inference artifact is wired.", "peer_count": peer_count}
    return {"status": "Unavailable", "detail": "ML runtime inference is not ready for this scenario.", "peer_count": peer_count}


def _regression_method_option_state(
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    regression_vde: float | None,
) -> dict[str, Any]:
    if regression_vde is None:
        return {"status": "Unavailable", "detail": "Needs resolved vehicle demand.", "row_count": 0}

    regression_state = _resolve_regression_state(vde_id, vde_row, ctx, regression_vde, render_filters=False)
    readiness = _scenario_feature_readiness_snapshot(vde_id, vde_row, ctx, regression_vde=regression_vde)
    warnings = list(regression_state.get("warnings") or [])
    dataset = regression_state.get("dataset")
    row_count = len(dataset) if dataset is not None else 0
    ml_state = _ml_method_option_state(vde_id, vde_row, ctx, regression_vde)

    if "regression_dataset_empty" in warnings:
        return {"status": "Insufficient dataset", "detail": "No peer records matched the current scenario filters.", "row_count": row_count}
    if "regression_dataset_insufficient" in warnings:
        return {"status": "Insufficient dataset", "detail": f"Only {row_count} peer records are available after filtering.", "row_count": row_count}
    if "regression_dataset_small" in warnings:
        return {"status": "Needs enough peer data", "detail": f"Only {row_count} peer records are available for regression.", "row_count": row_count}
    if "regression_dataset_moderate" in warnings and ml_state["status"] != "Recommended":
        return {"status": "Recommended fallback", "detail": f"{row_count} peer records are available for the data-driven fallback. Readiness: {readiness['status_label']}.", "row_count": row_count}
    if ml_state["status"] != "Recommended":
        return {"status": "Recommended fallback", "detail": f"{row_count} peer records are available for the data-driven fallback. Readiness: {readiness['status_label']}.", "row_count": row_count}
    return {"status": "Available", "detail": f"{row_count} peer records are available for regression.", "row_count": row_count}


def _method_option_state(
    method_label: str,
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    regression_vde: float | None,
) -> dict[str, str]:
    if method_label == "ML Prediction":
        request = _build_active_fuel_estimate_request(
            vde_id,
            vde_row,
            ctx,
            regression_vde,
            method_label=method_label,
        )
        if request is None:
            return {"status": "Pending", "detail": "Needs a valid scenario request."}
        setup = describe_ml_prediction_setup(
            request,
            model_artifact_path=st.session_state.get("pwt_ml_artifact_path"),
            predictor=request.model_options.get("ml_predictor"),
        )
        ml_status = str(setup.get("status") or "unknown")
        missing_count = len(setup.get("features", {}).get("missing_features") or [])
        if ml_status == "available" and missing_count == 0:
            return {"status": "Recommended", "detail": "Artifact loaded and feature coverage is complete."}
        if ml_status == "available":
            return {"status": "Missing features", "detail": f"Artifact loaded with {missing_count} missing feature(s)."}
        if ml_status == "export_pending":
            return {"status": "Needs artifact", "detail": "Notebook exists, but no inference artifact is wired."}
        if ml_status == "artifact_load_failed":
            return {"status": "Artifact issue", "detail": "An artifact candidate exists, but loading failed."}
        return {"status": "Planned", "detail": "ML setup metadata is present, but runtime inference is not ready."}

    if method_label == "Physics Simple":
        if ctx.get("energy_value_mj_per_km") is not None:
            return {"status": "Ready", "detail": "Best fallback when ML is unavailable."}
        return {"status": "Pending", "detail": "Needs resolved vehicle demand."}

    if method_label == "Regression":
        if regression_vde is not None:
            return {"status": "Ready", "detail": "Uses peer dataset fit from the current demand basis."}
        return {"status": "Pending", "detail": "Needs resolved vehicle demand."}

    if method_label == "Manual / Imported":
        return {"status": "Optional", "detail": "Use only when you want to stage an observed/imported result."}

    if method_label == "Physics + ML Residual":
        return {"status": "Planned", "detail": "Physics baseline plus ML residual correction is not integrated yet."}

    return {"status": "Planned", "detail": "Cycle/map simulation is reserved for a future release."}


def inject_powertrain_scenario_style() -> None:
    st.markdown(
        """
        <style>
        .pwt-step-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.15rem;
        }
        .pwt-step-caption {
            color: #667085;
            font-size: 0.92rem;
            margin-bottom: 0.85rem;
        }
        .pwt-summary-chip {
            padding: 0.45rem 0.7rem;
            border: 1px solid rgba(49, 130, 246, 0.18);
            border-radius: 8px;
            background: rgba(248, 250, 252, 0.95);
            margin-bottom: 0.35rem;
            min-height: 6.2rem;
        }
        .pwt-summary-chip.is-ok {
            border-color: rgba(34, 197, 94, 0.28);
            background: rgba(240, 253, 244, 0.95);
        }
        .pwt-summary-chip.is-pending {
            border-color: rgba(245, 158, 11, 0.28);
            background: rgba(255, 251, 235, 0.98);
        }
        .pwt-summary-chip.is-warn {
            border-color: rgba(239, 68, 68, 0.24);
            background: rgba(254, 242, 242, 0.98);
        }
        .pwt-summary-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.45rem;
            margin-bottom: 0.2rem;
        }
        .pwt-summary-chip strong {
            display: block;
            font-size: 0.78rem;
            color: #475467;
        }
        .pwt-summary-chip span {
            font-size: 0.95rem;
            color: #101828;
        }
        .pwt-summary-status {
            display: inline-flex;
            align-items: center;
            gap: 0.2rem;
            font-size: 0.72rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .pwt-summary-status.is-ok { color: #166534; }
        .pwt-summary-status.is-pending { color: #b45309; }
        .pwt-summary-status.is-warn { color: #b42318; }
        .pwt-summary-status.is-neutral { color: #475467; }
        .pwt-summary-detail {
            margin-top: 0.28rem;
            font-size: 0.76rem;
            color: #667085;
            line-height: 1.3;
        }
        .pwt-context-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.5rem;
            margin: 0.35rem 0 0.75rem 0;
        }
        .pwt-context-item {
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 0.45rem 0.6rem;
            background: #fbfdff;
            min-height: 4.2rem;
        }
        .pwt-context-label {
            color: #667085;
            font-size: 0.72rem;
            font-weight: 600;
            margin-bottom: 0.15rem;
        }
        .pwt-context-value {
            color: #101828;
            font-size: 0.9rem;
            font-weight: 600;
            overflow-wrap: anywhere;
        }
        .pwt-summary-chip.is-neutral {
            border-color: #d0d7de;
            background: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_powertrain_step_header(number: int, title: str, caption: str) -> None:
    st.markdown(f"<div class='pwt-step-title'>{number}. {title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pwt-step-caption'>{caption}</div>", unsafe_allow_html=True)


def render_powertrain_step_navigation() -> str:
    current = str(st.session_state.get("pwt_active_step") or PWT_SCENARIO_STEPS[0])
    if current not in PWT_SCENARIO_STEPS:
        current = PWT_SCENARIO_STEPS[0]
        st.session_state["pwt_active_step"] = current
    selected = st.radio(
        "Powertrain Scenario step",
        PWT_SCENARIO_STEPS,
        horizontal=True,
        index=PWT_SCENARIO_STEPS.index(current),
        key="pwt_active_step",
    )
    return str(selected)


def _current_unit_system() -> str:
    return normalize_unit_system(st.session_state.get("unit_system"))


def _current_pwt_input_mode() -> str:
    mode = str(st.session_state.get("pwt_input_mode") or PWT_INPUT_MODES[0]).strip()
    return mode if mode in PWT_INPUT_MODES else PWT_INPUT_MODES[0]


def render_powertrain_sidebar_controls() -> str:
    with st.sidebar:
        st.header("Powertrain Scenario")
        st.caption("Choose the workspace style and display units for the active scenario draft.")
        st.radio(
            "Display units",
            ["Metric", "US customary"],
            horizontal=True,
            key="unit_system",
        )
        st.radio(
            "Input mode",
            PWT_INPUT_MODES,
            index=PWT_INPUT_MODES.index(_current_pwt_input_mode()),
            key="pwt_input_mode",
        )
        st.toggle("Advanced details", key="pwt_show_technical")
    return _current_pwt_input_mode()


def _format_demand_value(value: Any, *, unavailable: str = "-") -> str:
    return format_quantity(value, "energy_per_distance", _current_unit_system(), unavailable=unavailable)


def _fuel_display_value(value_l_100km: Any) -> float | None:
    numeric = to_float(value_l_100km)
    if numeric is None:
        return None
    if _current_unit_system() == "US customary":
        return float(numeric) * 0.425143707
    return float(numeric)


def _fuel_display_unit() -> str:
    return "gal/100mi" if _current_unit_system() == "US customary" else "L/100km"


def _format_fuel_value(value_l_100km: Any, *, unavailable: str = "-") -> str:
    numeric = _fuel_display_value(value_l_100km)
    if numeric is None:
        return unavailable
    return f"{numeric:.2f} {_fuel_display_unit()}"


def _format_energy_value(value_wh_km: Any, *, unavailable: str = "-") -> str:
    return format_quantity(value_wh_km, "energy_wh_per_distance", _current_unit_system(), unavailable=unavailable)


def _format_co2_value(value_g_km: Any, *, unavailable: str = "-") -> str:
    return format_quantity(value_g_km, "co2_per_distance", _current_unit_system(), unavailable=unavailable)


def _format_metric_value(value: Any, *, format_str: str = "%.2f", suffix: str = "") -> str:
    numeric = to_float(value)
    if numeric is None:
        return "-"
    rendered = format_str % float(numeric)
    return f"{rendered}{suffix}" if suffix else rendered


def _build_bench_snapshot(vde_id: int, vde_row: dict) -> dict[str, Any]:
    ctx = get_build_scenario_context(vde_id, vde_row)
    _resolve_active_pwt_setup_method(vde_id, vde_row, ctx)
    energy_values = resolve_vde_energy_values(vde_row)
    regression_vde = ctx.get("energy_value_mj_per_km")
    request = _build_active_fuel_estimate_request(vde_id, vde_row, ctx, regression_vde)
    result = run_fuel_estimation(request) if request is not None else None
    return {
        "ctx": ctx,
        "energy_values": energy_values,
        "request": request,
        "result": result,
    }


def _build_bench_status_items(snapshot: dict[str, Any]) -> list[str]:
    request = snapshot.get("request")
    result = snapshot.get("result")
    if request is None or result is None:
        return ["Preview Pending"]
    summary = dict((result.assumptions or {}).get("confidence_summary") or {})
    if not summary:
        summary = build_estimate_confidence_summary(
            request=request,
            method=result.method,
            confidence=result.confidence,
            warnings=result.warnings,
            assumptions=result.assumptions,
        )
    return list(summary.get("status_items") or [])


def _render_bench_badges(items: list[str]) -> None:
    if not items:
        return
    chips = "".join(
        (
            "<span style=\"display:inline-block;margin:0 0.35rem 0.35rem 0;"
            "padding:0.2rem 0.6rem;border:1px solid #d0d7de;border-radius:999px;"
            "background:#f7f9fb;font-size:0.85rem;\">"
            f"{html.escape(str(item))}</span>"
        )
        for item in items
    )
    st.markdown(chips, unsafe_allow_html=True)


def _confidence_summary_from_saved_row(row: pd.Series | dict) -> dict[str, Any]:
    data = dict(row)
    assumptions = _load_json_blob(data.get("assumptions_json"))
    provenance = _load_json_blob(data.get("provenance_json"))
    summary = dict(provenance.get("confidence_summary") or assumptions.get("confidence_summary") or {})
    if summary:
        return summary
    confidence = provenance.get("confidence") or data.get("confidence")
    warnings = list(provenance.get("warnings") or [])
    method = str(data.get("engine_method") or "").strip()
    if method:
        return {
            "level": confidence or "-",
            "label": str(confidence or "-").replace("_", " ").title() if confidence else "-",
            "method_status": method,
            "status_items": [method],
            "reasons": [],
            "warning_count": len(warnings),
        }
    return {}


def _pse_summary_from_saved_row(row: pd.Series | dict) -> dict[str, Any]:
    data = dict(row)
    assumptions = _load_json_blob(data.get("assumptions_json"))
    provenance = _load_json_blob(data.get("provenance_json"))
    summary = dict(provenance.get("pse_summary") or assumptions.get("pse_summary") or {})
    return summary


def _render_scenario_bench_visual(vde_row: dict, snapshot: dict[str, Any], status_items: list[str]) -> None:
    ctx = dict(snapshot.get("ctx") or {})
    result = snapshot.get("result")
    pse_summary = dict((result.assumptions if result else {}).get("pse_summary") or {})
    method_label = _pwt_method_label(result.method if result else None)
    demand_value = _format_demand_value(ctx.get("energy_value_mj_per_km"), unavailable="Pending")
    confidence_label = str((result.confidence if result else "pending") or "pending").replace("_", " ").title()
    if pse_summary.get("value") is not None:
        pse_title = _format_metric_value(pse_summary.get("percent_value"), format_str="%.1f", suffix="%")
        pse_caption = str(pse_summary.get("source_label") or "Cycle-effective PSE")
    else:
        pse_title = "PSE pending"
        pse_caption = _pse_pending_message(method_label)
    if result and result.fuel_l_100km is not None:
        result_title = f"{result.fuel_l_100km:.2f} L/100km"
    elif result and result.energy_Wh_km is not None:
        result_title = f"{result.energy_Wh_km:.1f} Wh/km"
    elif result and result.gco2_km is not None:
        result_title = f"{result.gco2_km:.1f} g/km"
    else:
        result_title = "Result pending"

    st.markdown("### Scenario Bench - Virtual Dyno")
    st.caption(
        "Maneuver / Cycle -> Vehicle Demand -> PSE -> Final Result -> Confidence"
    )
    st.markdown(
        (
            "<div style='border:1px solid #d0d7de;border-radius:8px;padding:1rem 1rem 0.8rem 1rem;"
            "background:#fbfdff;'>"
            f"<div style='font-weight:600;font-size:1rem;margin-bottom:0.35rem;'>"
            f"Maneuver: {html.escape(str(ctx.get('scenario_intent') or '-'))} | "
            f"Cycle: {html.escape(str(vde_row.get('cycle_name') or '-'))} | "
            f"Method: {html.escape(method_label)}</div>"
            "<div style='font-size:0.95rem;color:#475569;margin-bottom:0.8rem;'>"
            "Maneuver / Cycle &nbsp;&rarr;&nbsp; Vehicle Demand &nbsp;&rarr;&nbsp; "
            "PSE &nbsp;&rarr;&nbsp; Final Result &nbsp;&rarr;&nbsp; Confidence"
            "</div>"
            "<div style='border:1px dashed #cbd5e1;border-radius:8px;padding:0.9rem;text-align:center;"
            "font-weight:600;background:#ffffff;margin-bottom:0.8rem;'>"
            "[ Vehicle on Roller Bench / Virtual Dyno ]"
            "</div>"
            "<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.75rem;'>"
            f"<div style='border:1px solid #e2e8f0;border-radius:8px;padding:0.75rem;background:#ffffff;'>"
            f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:0.25rem;'>Maneuver / Cycle</div>"
            f"<div style='font-weight:600;'>{html.escape(str(vde_row.get('cycle_name') or '-'))}</div>"
            f"<div style='font-size:0.85rem;color:#475569;'>{html.escape(str(ctx.get('scenario_name') or '-'))}</div>"
            "</div>"
            f"<div style='border:1px solid #e2e8f0;border-radius:8px;padding:0.75rem;background:#ffffff;'>"
            f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:0.25rem;'>Vehicle Demand</div>"
            f"<div style='font-weight:600;'>{html.escape(demand_value)}</div>"
            f"<div style='font-size:0.85rem;color:#475569;'>{html.escape(str(ctx.get('energy_basis') or '-'))}</div>"
            "</div>"
            f"<div style='border:1px solid #e2e8f0;border-radius:8px;padding:0.75rem;background:#ffffff;'>"
            f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:0.25rem;'>PSE - Powertrain System Efficiency</div>"
            f"<div style='font-weight:600;'>{html.escape(pse_title)}</div>"
            f"<div style='font-size:0.85rem;color:#475569;'>{html.escape(pse_caption)}</div>"
            "</div>"
            f"<div style='border:1px solid #e2e8f0;border-radius:8px;padding:0.75rem;background:#ffffff;'>"
            f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:0.25rem;'>Final Result</div>"
            f"<div style='font-weight:600;'>{html.escape(result_title)}</div>"
            f"<div style='font-size:0.85rem;color:#475569;'>Confidence: {html.escape(confidence_label)}</div>"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    badge_items = [
        f"Scenario: {str(ctx.get('scenario_name') or '-')}",
        f"Method: {method_label}",
        f"Confidence: {confidence_label}",
        f"Demand basis: {str(ctx.get('energy_basis') or '-')}",
    ]
    if pse_summary.get("value") is not None:
        badge_items.append(f"PSE: {pse_summary['value']:.3f}")
    else:
        badge_items.append("PSE pending")
    if result and result.fuel_l_100km is not None:
        badge_items.append(f"Fuel: {result.fuel_l_100km:.2f} L/100km")
    if result and result.energy_Wh_km is not None:
        badge_items.append(f"Energy: {result.energy_Wh_km:.1f} Wh/km")
    if result and result.gco2_km is not None:
        badge_items.append(f"CO2: {result.gco2_km:.1f} g/km")
    if all(vde_row.get(key) not in (None, "") for key in ("coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2")):
        badge_items.append("Roadload ABC available")
    _render_bench_badges(badge_items + status_items)


def _summary_card_status_class(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"ok", "available", "current", "pre-check ok", "guidance available", "recommended", "ready", "optional", "confirmed", "derived"}:
        return "is-ok"
    if normalized in {"missing", "warn", "warning", "low confidence", "artifact issue", "not ready"}:
        return "is-warn"
    if normalized in {"locked", "not used", "registered only"}:
        return "is-neutral"
    return "is-pending"


def _status_text_for_display(status: str) -> str:
    normalized = str(status or "pending").strip().lower()
    mapping = {
        "ok": "OK",
        "ready": "Ready",
        "confirmed": "Confirmed",
        "derived": "Derived",
        "missing": "Missing",
        "pending": "Pending",
        "locked": "Locked",
        "draft": "Draft",
        "registered only": "Registered only",
        "not ready": "Not ready",
    }
    return mapping.get(normalized, str(status or "Pending"))


def _summary_card_markup(*, title: str, value: str, status: str, detail: str) -> str:
    status_text = html.escape(_status_text_for_display(status))
    status_class = _summary_card_status_class(status)
    return (
        f"<div class='pwt-summary-chip {status_class}'>"
        f"<div class='pwt-summary-top'>"
        f"<strong>{html.escape(title)}</strong>"
        f"<div class='pwt-summary-status {status_class}'>{status_text}</div>"
        "</div>"
        f"<span>{html.escape(value)}</span>"
        f"<div class='pwt-summary-detail'>{html.escape(detail)}</div>"
        "</div>"
    )


def _render_bench_hotspots() -> None:
    active = _active_bench_hotspot()
    rows = [PWT_BENCH_HOTSPOTS[:4], PWT_BENCH_HOTSPOTS[4:]]
    for row_index, row in enumerate(rows):
        cols = st.columns(len(row))
        for col, hotspot in zip(cols, row):
            if col.button(
                hotspot["label"],
                key=f"pwt_bench_hotspot_{row_index}_{hotspot['key']}",
                use_container_width=True,
                type="primary" if active == hotspot["key"] else "secondary",
            ):
                st.session_state["pwt_bench_hotspot"] = hotspot["key"]


def _render_bench_detail_panel(vde_id: int, vde_row: dict, snapshot: dict[str, Any], status_items: list[str]) -> None:
    active = _active_bench_hotspot()
    hotspot = _bench_hotspot_map()[active]
    ctx = dict(snapshot.get("ctx") or {})
    energy_values = dict(snapshot.get("energy_values") or {})
    request = snapshot.get("request")
    result = snapshot.get("result")
    assumptions = dict((result.assumptions if result else {}) or {})
    method_label = _pwt_method_label(result.method if result else None)

    with st.container():
        st.markdown(f"#### Active subsystem: {hotspot['label']}")

        if active == "driver_cycle":
            st.markdown("**Current state**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Cycle", str(vde_row.get("cycle_name") or "-"))
            c2.metric("Scenario", str(ctx.get("scenario_name") or "-"))
            c3.metric("Intent", str(ctx.get("scenario_intent") or "-"))
            st.markdown("**Next action**")
            st.caption("Confirm cycle and scenario intent before reviewing demand and powertrain conversion.")
            st.markdown(f"**Related workspace:** `{hotspot['tab']}`")
            return

        if active == "roadload_vde":
            st.markdown("**Current state**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Selected basis", str(ctx.get("energy_basis") or "-"))
            c2.metric(f"Demand used [{unit_label('energy_per_distance', _current_unit_system())}]", _format_demand_value(ctx.get("energy_value_mj_per_km"), unavailable="Pending").replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""))
            c3.metric(f"VDE_TOTAL [{unit_label('energy_per_distance', _current_unit_system())}]", _format_demand_value(energy_values.get("vde_total_mj_per_km")).replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""))
            c4.metric(f"VDE_NET [{unit_label('energy_per_distance', _current_unit_system())}]", _format_demand_value(energy_values.get("vde_net_mj_per_km")).replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""))
            abc1, abc2, abc3 = st.columns(3)
            abc1.metric("ABC A", _format_metric_value(vde_row.get("coast_A_N"), format_str="%.2f"))
            abc2.metric("ABC B", _format_metric_value(vde_row.get("coast_B_N_per_kph"), format_str="%.4f"))
            abc3.metric("ABC C", _format_metric_value(vde_row.get("coast_C_N_per_kph2"), format_str="%.5f"))
            net_status = "NET available" if energy_values.get("vde_net_mj_per_km") is not None else "TOTAL only"
            st.caption(f"Transmission / NET status: {net_status}")
            st.markdown("**Next action**")
            st.caption("Review Vehicle Demand here; edit roadload in VDE Setup if the source itself must change.")
            st.caption("Vehicle Demand is the input to the PSE conversion.")
            st.markdown(f"**Related workspace:** `{hotspot['tab']}`")
            if energy_values.get("warnings"):
                st.warning("Roadload / VDE warnings: " + ", ".join(energy_values["warnings"]))
            return

        if active == "powertrain_efficiency":
            pse_summary = dict((result.assumptions if result else {}).get("pse_summary") or {})
            st.markdown("**Current state**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PSE", _format_metric_value(pse_summary.get("value"), format_str="%.3f") if pse_summary.get("value") is not None else "PSE pending")
            c2.metric("Mode", str(pse_summary.get("mode") or "-").title())
            c3.metric("Source", str(pse_summary.get("source_label") or "Unavailable"))
            c4.metric("Target type", str(pse_summary.get("target_type") or "-"))
            d1, d2 = st.columns(2)
            d1.metric(f"Demand [{unit_label('energy_per_distance', _current_unit_system())}]", _format_demand_value(pse_summary.get("demand_mj_per_km")).replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""))
            d2.metric(f"Consumed energy [{unit_label('energy_per_distance', _current_unit_system())}]", _format_demand_value(pse_summary.get("consumed_energy_mj_per_km")).replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""))
            st.caption(_pse_help_text())
            st.markdown("**Next action**")
            st.caption(_pse_pending_message(method_label) if pse_summary.get("value") is None else "Review the PSE source and decide whether to save or refine assumptions.")
            st.markdown(f"**Related workspace:** `{hotspot['tab']}`")
            st.caption(str(pse_summary.get("warning") or "PSE is cycle-effective and should not be interpreted as pure engine efficiency."))
            if pse_summary.get("source") == "ml_fuel_prediction":
                st.caption("Current ML artifact predicts final fuel/energy outputs. PSE is derived from that result; direct PSE prediction is planned/future.")
            return

        if active == "transmission":
            transmission_model = st.session_state.get("pwt_trans_model") or vde_row.get("transmission_model")
            st.markdown("**Current state**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transmission", str(vde_row.get("transmission_type") or "-"))
            c2.metric("Model", str(transmission_model or "-"))
            c3.metric("Gears", str(st.session_state.get("pwt_gears") or vde_row.get("gear_count") or "-"))
            c4.metric("Final drive", _format_metric_value(st.session_state.get("pwt_fdr") or vde_row.get("final_drive_ratio"), format_str="%.3f"))
            st.markdown("**Next action**")
            st.caption("Complete drivetrain metadata if needed.")
            st.markdown(f"**Related workspace:** `{hotspot['tab']}`")
            return

        if active == "engine_fuel":
            powertrain_features = _build_powertrain_features_from_state(vde_row, ctx)
            st.markdown("**Current state**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fuel type", str(powertrain_features.get("fuel_type") or "-"))
            c2.metric("Eta PT", _format_metric_value(powertrain_features.get("eta_pt_est"), format_str="%.3f"))
            c3.metric("LHV", _format_metric_value(powertrain_features.get("LHV_MJ_per_L"), format_str="%.2f", suffix=" MJ/L"))
            c4.metric("gCO2 / L", _format_metric_value(powertrain_features.get("gCO2_per_L"), format_str="%.1f"))
            st.markdown("**Next action**")
            st.caption("Review fuel and efficiency assumptions.")
            st.markdown(f"**Related workspace:** `{hotspot['tab']}`")
            return

        if active == "electric_battery":
            powertrain_features = _build_powertrain_features_from_state(vde_row, ctx)
            st.markdown("**Current state**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Electrification", str(ctx.get("electrification") or "-"))
            c2.metric("Drive efficiency", _format_metric_value(powertrain_features.get("bev_eff_drive"), format_str="%.3f"))
            c3.metric("Utility factor", _format_metric_value(powertrain_features.get("utility_factor"), format_str="%.2f"))
            c4.metric("Grid CO2", _format_metric_value(powertrain_features.get("grid_gco2_per_kwh"), format_str="%.1f", suffix=" g/kWh"))
            st.markdown("**Next action**")
            st.caption("Review electric path assumptions.")
            st.markdown(f"**Related workspace:** `{hotspot['tab']}`")
            return

        if active == "ml_peers":
            if request is None:
                st.info("Select a functional estimation path before reviewing ML / peers details.")
                return
            ml_setup = describe_ml_prediction_setup(
                request,
                model_artifact_path=st.session_state.get("pwt_ml_artifact_path"),
                predictor=request.model_options.get("ml_predictor"),
            )
            peer_quality = dict(assumptions.get("peer_group_quality") or {})
            st.markdown("**Current state**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Method", _pwt_method_label(result.method if result else None))
            c2.metric("ML artifact", str(assumptions.get("integration_status") or ml_setup.get("status") or "-"))
            c3.metric("SHAP", str(assumptions.get("shap_status") or "Unavailable"))
            c4.metric("Peer quality", str(peer_quality.get("label") or "-"))
            st.markdown("**Next action**")
            st.caption("Run ML/peers or review coverage.")
            st.markdown(f"**Related workspace:** `{hotspot['tab']}`")
            return

        st.markdown("**Current state**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Method", method_label)
        c2.metric(f"Fuel [{_fuel_display_unit()}]", _format_fuel_value(result.fuel_l_100km, unavailable="Pending").replace(f" {_fuel_display_unit()}", "") if result else "Pending")
        c3.metric(f"Energy [{unit_label('energy_wh_per_distance', _current_unit_system())}]", _format_energy_value(result.energy_Wh_km, unavailable="Pending").replace(f" {unit_label('energy_wh_per_distance', _current_unit_system())}", "") if result else "Pending")
        c4.metric(f"CO2 [{unit_label('co2_per_distance', _current_unit_system())}]", _format_co2_value(result.gco2_km, unavailable="Pending").replace(f" {unit_label('co2_per_distance', _current_unit_system())}", "") if result else "Pending")
        if result is not None:
            st.markdown("**Next action**")
            st.caption("Review and save.")
            st.markdown(f"**Related workspace:** `{hotspot['tab']}`")
            st.caption(f"Confidence: {str(result.confidence or '-')} | Warnings: {len(result.warnings)}")
        else:
            st.info("No active preview yet for the selected method.")


def render_powertrain_scenario_bench(vde_id: int, vde_row: dict) -> None:
    draft = _build_powertrain_scenario_draft(vde_id, vde_row)
    snapshot = _build_bench_snapshot(vde_id, vde_row)
    ctx = dict(draft.get("ctx") or snapshot.get("ctx") or {})
    result = draft.get("baseline_estimate", {}).get("result") or snapshot.get("result")
    assumptions = dict((result.assumptions if result else {}) or {})
    confidence_summary = dict(assumptions.get("confidence_summary") or {})
    pse_summary = dict(assumptions.get("pse_summary") or {})
    energy_values = dict(snapshot.get("energy_values") or {})
    readiness = dict(draft.get("feature_readiness") or {})
    reference_summary = dict(draft.get("powertrain_reference") or {})
    proposal_summary = dict(draft.get("proposal_result") or {})
    method_label = str(draft.get("baseline_estimate", {}).get("method") or _pwt_method_label(result.method if result else None))
    demand_value = _format_demand_value(ctx.get("energy_value_mj_per_km"), unavailable="Pending")
    baseline_summary = dict(proposal_summary.get("baseline") or {})
    proposal_metrics = dict(proposal_summary.get("proposal") or {})
    tech_deltas = list(draft.get("technology_deltas") or [])
    lead_delta = tech_deltas[0] if tech_deltas else None
    regression_state = _regression_method_option_state(vde_id, vde_row, ctx, ctx.get("energy_value_mj_per_km"))
    confirmed_method = _confirmed_pwt_setup_method()

    if proposal_metrics.get("fuel_l_100km") is not None:
        final_result_value = _format_fuel_value(proposal_metrics["fuel_l_100km"], unavailable="Proposal pending")
    elif proposal_metrics.get("energy_Wh_km") is not None:
        final_result_value = _format_energy_value(proposal_metrics["energy_Wh_km"], unavailable="Proposal pending")
    elif proposal_metrics.get("gco2_km") is not None:
        final_result_value = _format_co2_value(proposal_metrics["gco2_km"], unavailable="Proposal pending")
    else:
        final_result_value = "Proposal pending"

    demand_status = "OK" if ctx.get("energy_value_mj_per_km") is not None else "Missing"
    source_status = "OK" if str(reference_summary.get("source_label") or "").strip() and str(reference_summary.get("source_label") or "") != "No reference row available" else "Missing"
    baseline_status = "Confirmed" if confirmed_method is not None else ("Ready" if baseline_summary.get("fuel_l_100km") is not None else "Pending")
    pse_status = "Derived" if pse_summary.get("value") is not None or baseline_summary.get("pse") is not None else "Pending"
    delta_counts = dict(proposal_summary.get("delta_counts") or {})
    delta_status = "Registered only" if delta_counts.get("applied", 0) == 0 and len(draft.get("technology_deltas") or []) > 0 else ("Draft" if delta_counts.get("applied", 0) > 0 else "Locked" if confirmed_method is None else "Pending")
    final_status = "Ready" if final_result_value != "Proposal pending" else "Pending"
    save_status = "Ready" if final_result_value != "Proposal pending" else "Not ready"
    peer_quality = dict(assumptions.get("peer_group_quality") or {})
    confidence_value = str(proposal_summary.get("confidence") or _confidence_preview_label(result, confidence_summary, assumptions)).replace("_", " ").title()
    confidence_status = "OK" if confidence_value in {"Pre-check OK", "Guidance available"} else confidence_value
    delta_value_text = "No delta"
    delta_detail = "Proposal equals baseline"
    if lead_delta is not None:
        effect_basis = _normalize_delta_effect_basis(lead_delta.get("effect_basis"))
        effect_value = to_float(lead_delta.get("effect_value"))
        if effect_basis == "fuel_percent_delta" and effect_value is not None:
            delta_value_text = f"{effect_value:+.1f}% fuel"
        elif effect_basis == "pse_percent_delta" and effect_value is not None:
            delta_value_text = f"{effect_value:+.1f}% PSE"
        elif effect_basis == "co2_percent_delta" and effect_value is not None:
            delta_value_text = f"{effect_value:+.1f}% CO2"
        elif effect_basis == "efficiency_multiplier" and effect_value is not None:
            delta_value_text = f"x{effect_value:.3f} PSE"
        else:
            delta_value_text = str(lead_delta.get("name") or "Metadata only")
        delta_detail = "Applied" if str(lead_delta.get("quantitative_status") or "") == "applied" else "Registered only"

    baseline_detail = "Baseline pending"
    if baseline_summary.get("pse") is not None:
        if method_label == "Observed / Derived PSE":
            baseline_detail = "Derived from reference PSE"
        else:
            baseline_detail = f"PSE {_format_metric_value(baseline_summary.get('pse'), format_str='%.3f')}"

    proposal_detail = "Same as baseline" if delta_value_text == "No delta" else f"Delta {_format_fuel_value((to_float(proposal_metrics.get('fuel_l_100km')) or 0.0) - (to_float(baseline_summary.get('fuel_l_100km')) or 0.0))}"
    proposal_value = "Same as baseline" if delta_value_text == "No delta" and baseline_summary.get("fuel_l_100km") is not None else final_result_value
    confidence_detail = _confidence_reason_label(
        readiness=readiness,
        active_method=method_label,
        reference_summary=reference_summary,
        regression_state=regression_state,
    )

    context_items = [
        ("Demand", f"{str(ctx.get('energy_basis') or '-')} {demand_value}"),
        ("Baseline source", str(reference_summary.get("source_label") or "Reference pending")),
        ("Baseline method", _pwt_method_label(method_label)),
        ("Delta", delta_value_text),
        ("Proposal", proposal_value),
        ("Confidence", confidence_value),
        ("Save", save_status),
    ]
    context_body = "".join(
        (
            "<div class='pwt-context-item'>"
            f"<div class='pwt-context-label'>{html.escape(label)}</div>"
            f"<div class='pwt-context-value'>{html.escape(str(value or '-'))}</div>"
            "</div>"
        )
        for label, value in context_items
    )
    st.markdown(f"<div class='pwt-context-strip'>{context_body}</div>", unsafe_allow_html=True)

    st.markdown("#### Powertrain Status Bar")
    cols = st.columns(7)
    cards = [
        _summary_card_markup(
            title="Demand",
            value=f"{str(ctx.get('energy_basis') or '-')} {demand_value}",
            status=demand_status,
            detail=str(vde_row.get("cycle_name") or "-"),
        ),
        _summary_card_markup(
            title="Source",
            value=_reference_type_display_label(reference_summary.get("source_type") or "-"),
            status=source_status,
            detail=str(reference_summary.get("source_label") or "Reference pending"),
        ),
        _summary_card_markup(
            title="Baseline",
            value=_format_fuel_value(baseline_summary.get("fuel_l_100km"), unavailable="Baseline pending") if baseline_summary.get("fuel_l_100km") is not None else "Baseline pending",
            status=baseline_status,
            detail=baseline_detail,
        ),
        _summary_card_markup(
            title="PSE",
            value=_format_metric_value(pse_summary.get("value") if pse_summary.get("value") is not None else baseline_summary.get("pse"), format_str="%.3f"),
            status=pse_status,
            detail="Powertrain System Efficiency",
        ),
        _summary_card_markup(
            title="Delta",
            value=delta_value_text,
            status=delta_status,
            detail=delta_detail,
        ),
        _summary_card_markup(
            title="Proposal",
            value=proposal_value,
            status=final_status,
            detail=proposal_detail,
        ),
        _summary_card_markup(
            title="Save",
            value=save_status,
            status=save_status,
            detail=confidence_detail,
        ),
    ]
    for col, card in zip(cols, cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)


def filters_bar(vde_id: int, electrification: str, key_ns: str = "fb", *, allow_current_vehicle_scope: bool = True) -> Dict[str, Any]:
    k = lambda name: f"{key_ns}_{name}"
    st.markdown("### Filters")
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1])

    filter_values = fetch_filter_values()
    cats = filter_values["categories"]
    makes = filter_values["makes"]
    elecs = filter_values["electrifications"]

    with c1:
        if allow_current_vehicle_scope:
            view_scope = st.selectbox("View", ["Only this Vehicle id", "All"], index=1, key=k("fl_scope"))
        else:
            all_fuelcons = fetch_fuelcons_all({})
            id_options: list[str] = []
            id_lookup: dict[str, int] = {}
            if all_fuelcons is not None and not all_fuelcons.empty:
                ordered_df = all_fuelcons.sort_values("id", ascending=False).drop_duplicates(subset=["id"])
                for _, row in ordered_df.iterrows():
                    try:
                        fuelcons_id = int(row["id"])
                    except Exception:
                        continue
                    vehicle = f"{str(row.get('make') or '-')} {str(row.get('model') or '-')}".strip()
                    year = str(row.get("year") or "-")
                    label = (
                        f"FC #{fuelcons_id} | VDE #{int(row['vde_id']) if pd.notna(row.get('vde_id')) else '-'} | "
                        f"{vehicle} {year} | {str(row.get('engine_method') or row.get('method_note') or '-')}"
                    )
                    id_options.append(label)
                    id_lookup[label] = fuelcons_id
            selected_fuelcons_labels = st.multiselect(
                "FuelCons ids",
                id_options,
                key=k("fl_fuelcons_ids"),
                placeholder="All benchmark records",
            )
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
    if allow_current_vehicle_scope and view_scope == "Only this Vehicle id":
        filters["vde_id"] = vde_id
    elif not allow_current_vehicle_scope and selected_fuelcons_labels:
        filters["fuelcons_ids"] = [id_lookup[label] for label in selected_fuelcons_labels]

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


def _regression_candidate_pool_filters(reg_filters: Dict[str, Any]) -> Dict[str, Any]:
    candidate_filters: Dict[str, Any] = {}
    if reg_filters.get("legislation"):
        candidate_filters["legislation"] = reg_filters["legislation"]
    if reg_filters.get("electrification"):
        candidate_filters["electrification"] = reg_filters["electrification"]
    return candidate_filters


def _regression_sample_quality(row_count: int) -> dict[str, Any]:
    if row_count < 5:
        return {"label": "Insufficient sample", "can_fit": False}
    if row_count < 15:
        return {"label": "Low confidence / small sample", "can_fit": True}
    if row_count < 30:
        return {"label": "Usable sample", "can_fit": True}
    return {"label": "Stronger sample", "can_fit": True}


def _regression_filters_summary(filters: Dict[str, Any]) -> str:
    summary: list[str] = []
    if filters.get("fuelcons_ids"):
        summary.append(f"{len(filters['fuelcons_ids'])} fuelcons ids")
    if filters.get("legislation"):
        summary.append(str(filters["legislation"]))
    if filters.get("electrification"):
        summary.append(str(filters["electrification"]))
    if filters.get("category"):
        summary.append(f"category {filters['category']}")
    if filters.get("make"):
        summary.append(f"make {filters['make']}")
    if filters.get("power_kw_range") is not None:
        lo_kw, hi_kw = filters["power_kw_range"]
        lo_hp = int(round(float(lo_kw) * 1.34102209)) if lo_kw is not None else None
        hi_hp = int(round(float(hi_kw) * 1.34102209)) if hi_kw is not None else None
        if lo_hp is not None and hi_hp is not None:
            summary.append(f"power {lo_hp}-{hi_hp} hp")
        elif lo_hp is not None:
            summary.append(f"power >= {lo_hp} hp")
        elif hi_hp is not None:
            summary.append(f"power <= {hi_hp} hp")
    return " | ".join(summary) if summary else "All benchmark records"


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

    st.subheader("Scenario Definition")
    st.caption("Define demand and reference context before baseline estimation.")
    st.caption(
        f"Building scenario from VDE #{int(vde_id)} - "
        f"{vde_row.get('make', '-')} {vde_row.get('model', '-')}".strip()
        + f" {vde_row.get('year', '-')}"
        + f" | {vde_row.get('cycle_name', '-')}"
        + f" | source revision {current_revision or '-'}"
    )

    st.markdown("#### Scenario")
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

    show_technical = _show_technical_details()
    st.markdown("#### Demand")
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
        "Vehicle demand basis for PSE conversion",
        energy_options,
        horizontal=True,
        index=energy_options.index(current_basis_label),
        key="pwt_energy_basis_label",
    )
    st.session_state["pwt_energy_basis"] = "VDE_NET" if st.session_state.get("pwt_energy_basis_label") == "Use VDE_NET (recommended)" else "VDE_TOTAL"

    energy_basis = st.session_state["pwt_energy_basis"]
    selected_value = energy_values["vde_net_mj_per_km"] if energy_basis == "VDE_NET" else energy_values["vde_total_mj_per_km"]
    info1, info2, info3 = st.columns(3)
    info1.metric("Demand basis", energy_basis)
    info2.metric(
        f"Demand used [{unit_label('energy_per_distance', _current_unit_system())}]",
        _format_demand_value(selected_value, unavailable="-").replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""),
    )
    info3.metric(
        "Baseline source",
        _reference_type_display_label(st.session_state.get("pwt_reference_source_type")) or "-",
    )
    with st.expander("Advanced details", expanded=show_technical):
        a1, a2, a3 = st.columns(3)
        a1.metric(
            f"VDE_TOTAL [{unit_label('energy_per_distance', _current_unit_system())}]",
            _format_demand_value(energy_values.get("vde_total_mj_per_km")).replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""),
        )
        a2.metric(
            f"VDE_NET [{unit_label('energy_per_distance', _current_unit_system())}]",
            _format_demand_value(energy_values.get("vde_net_mj_per_km")).replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""),
        )
        a3.metric("Source status", "Live snapshot")
        b1, b2, b3 = st.columns(3)
        b1.metric("Vehicle", f"{str(vde_row.get('make') or '-')} {str(vde_row.get('model') or '-')}".strip())
        b2.metric("Mass basis", "test_mass_kg" if vde_row.get("test_mass_kg") not in (None, "") else "mass_kg")
        b3.metric("Revision", str(current_revision or "-"))
        a1.metric("ABC A", _format_metric_value(vde_row.get("coast_A_N"), format_str="%.2f"))
        a2.metric("ABC B", _format_metric_value(vde_row.get("coast_B_N_per_kph"), format_str="%.4f"))
        a3.metric("ABC C", _format_metric_value(vde_row.get("coast_C_N_per_kph2"), format_str="%.5f"))
        st.caption("Technical vehicle-demand details for the active source.")

    st.markdown("#### Baseline powertrain source")
    reference_options = _available_reference_source_types(vde_id)
    if st.session_state.get("pwt_reference_source_type") not in reference_options:
        st.session_state["pwt_reference_source_type"] = "Manual definition"
    st.selectbox(
        "Baseline powertrain source",
        reference_options,
        key="pwt_reference_source_type",
        format_func=_reference_type_display_label,
    )
    reference_type = _reference_type_key(st.session_state.get("pwt_reference_source_type"))
    if "Same vehicle fuelcons_db line" not in reference_options:
        st.caption("No same-vehicle fuelcons data is available for this VDE. Create a new scenario or choose another source.")
    if reference_type in {"Same vehicle fuelcons_db line", "Another fuelcons_db line"}:
        candidate_df = _reference_candidates_for_type(vde_id, reference_type)
        if candidate_df.empty:
            st.info("Baseline source: Create new scenario")
        else:
            selection_key = "pwt_reference_same_row_id" if reference_type == "Same vehicle fuelcons_db line" else "pwt_reference_other_row_id"
            labels = []
            label_to_id: dict[str, int] = {}
            for _, row in candidate_df.iterrows():
                try:
                    row_id = int(row["id"])
                except Exception:
                    continue
                label = _reference_candidate_label(row)
                labels.append(label)
                label_to_id[label] = row_id
            if labels:
                current_id = st.session_state.get(selection_key)
                current_label = next((label for label, row_id in label_to_id.items() if row_id == current_id), labels[0])
                selected_label = st.selectbox("Reference row", labels, index=labels.index(current_label), key=f"{selection_key}_label")
                st.session_state[selection_key] = label_to_id[selected_label]
    elif reference_type == "Manual definition":
        r1, r2 = st.columns(2)
        r1.text_input("Reference label", key="pwt_reference_manual_label")
        if show_technical:
            r2.selectbox("Source maturity", DELTA_MATURITY_OPTIONS, key="pwt_reference_manual_maturity")
            st.text_area("Reference note", key="pwt_reference_manual_note", height=70)

    if str(st.session_state.get("pwt_scenario_electrification") or "").upper() == "BEV":
        st.checkbox(
            "Use BEV draft placeholders (draft-only)",
            key="pwt_bev_draft_placeholders",
        )
        st.caption("Draft-only placeholders stay in the Powertrain Scenario context and never update the VDE snapshot.")
    else:
        st.session_state["pwt_bev_draft_placeholders"] = False
    ctx = get_build_scenario_context(vde_id, vde_row)
    reference_summary = _selected_powertrain_reference(vde_id, vde_row)
    readiness = _scenario_feature_readiness_snapshot(
        vde_id,
        vde_row,
        ctx,
        regression_vde=ctx.get("energy_value_mj_per_km"),
        reference_summary=reference_summary,
    )

    _render_feature_readiness_highlight(readiness)

    if show_technical:
        with st.expander("Advanced: selected reference", expanded=False):
            ref1, ref2, ref3 = st.columns(3)
            ref1.metric("Reference label", str(reference_summary.get("source_label") or "-"))
            ref2.metric(
                "Observed baseline",
                f"{reference_summary['observed_fuel']:.2f} L/100km" if reference_summary.get("observed_fuel") is not None else "Pending",
            )
            ref3.metric(
                "Observed PSE",
                _format_metric_value(reference_summary.get("observed_pse"), format_str="%.3f") if reference_summary.get("observed_pse") is not None else "Pending",
            )
            st.caption(str(reference_summary.get("note") or "Reference fuel result belongs to the source vehicle. Baseline estimate applies the selected conversion layer to the active VDE demand."))

    metadata_complete, metadata_total, metadata_missing = _metadata_status_summary(readiness)
    metadata_title = "Complete scenario metadata"
    if metadata_missing:
        metadata_title += f" ({len(metadata_missing)} highlighted)"
    else:
        metadata_title += f" ({metadata_complete}/{metadata_total} ready)"

    with st.expander(metadata_title, expanded=False):
        _render_powertrain_metadata_review(vde_id, vde_row, ctx, readiness, expanded=show_technical)

    return ctx


def render_powertrain_inputs_panel(vde_id: int, vde_row: dict) -> Dict[str, Any]:
    ctx = get_build_scenario_context(vde_id, vde_row)
    active_method = _resolve_active_pwt_setup_method(vde_id, vde_row, ctx)

    st.subheader("Baseline Estimation Context")
    st.caption("Configure the shared drivetrain and energy-conversion context used by the baseline estimation method.")
    if active_method != "Physics Simple":
        st.info(
            "These inputs stay shared across the conversion workspace. Assume efficiency uses them directly, "
            "while the other methods reuse them as context and provenance."
        )

    render_powertrain_conversion_inputs(vde_row, ctx)
    _render_scenario_extras_inputs(vde_id, vde_row, ctx)
    return ctx


def render_powertrain_method_cards(vde_id: int, vde_row: dict, ctx: Dict[str, Any], regression_vde: float | None) -> str:
    active_method = _resolve_active_pwt_setup_method(vde_id, vde_row, ctx)
    reference_summary = _selected_powertrain_reference(vde_id, vde_row)
    readiness = _scenario_feature_readiness_snapshot(vde_id, vde_row, ctx, regression_vde=regression_vde, reference_summary=reference_summary)
    observed_available = to_float(reference_summary.get("observed_pse")) is not None
    ml_state = _ml_method_option_state(vde_id, vde_row, ctx, regression_vde)
    regression_state = _regression_method_option_state(vde_id, vde_row, ctx, regression_vde)
    method_options = []
    if observed_available:
        method_options.append("Reuse observed reference PSE")
    method_options.extend([
        "ML prediction",
        "Regression estimate",
        "Assume efficiency",
        "Enter observed/imported result",
    ])
    current_method = _pwt_method_label(active_method)
    if current_method not in method_options:
        current_method = "Regression estimate" if "Regression estimate" in method_options else method_options[0]
    selected_method = st.selectbox(
        "Baseline method",
        method_options,
        index=method_options.index(current_method),
        key="pwt_baseline_method_select",
    )
    if selected_method != current_method:
        _select_pwt_setup_method(PWT_METHOD_DISPLAY_TO_INTERNAL[selected_method])
        active_method = str(st.session_state.get("pwt_setup_method") or active_method)
    simple_status, simple_detail = _simple_readiness_status(readiness)
    st.caption(f"{simple_status} - {simple_detail}")

    if active_method == "Observed / Derived PSE":
        if observed_available:
            st.caption("Derives PSE from an observed reference result and applies it to the active VDE demand.")
        else:
            st.warning("Observed reference PSE unavailable - choose another reference, use regression, assume efficiency, or enter an observed/imported result.")
    elif active_method == "ML Prediction":
        st.caption("Model prediction based on available metadata/features.")
    elif active_method == "Regression":
        st.caption("Data-driven estimate from comparable records.")
    elif active_method == "Manual / Imported":
        st.caption("Enter fuel, energy or CO2 directly. PSE is derived afterward for diagnostics and provenance.")
    else:
        st.caption("Enter an assumed powertrain efficiency/PSE. The system computes fuel/CO2 from the active VDE demand.")

    with st.expander("Advanced: method details", expanded=_show_technical_details()):
        if active_method == "Observed / Derived PSE":
            st.caption("Observed reference -> derive PSE -> apply to active VDE.")
            st.caption(str(reference_summary.get("source_label") or "-"))
        elif active_method == "ML Prediction":
            st.caption("Model prediction based on available metadata/features.")
            st.caption(f"{ml_state['status']} | {ml_state['detail']}")
        elif active_method == "Regression":
            st.caption("Data-driven estimate from comparable records.")
            st.caption(f"{regression_state['status']} | {regression_state['detail']}")
        elif active_method == "Manual / Imported":
            st.caption("Entered result -> derive PSE for diagnostics.")
        else:
            st.caption("Assumed efficiency -> compute fuel/CO2 from active VDE.")
    return active_method


def _render_baseline_demand_stage(vde_id: int, vde_row: dict) -> dict[str, Any]:
    st.markdown("#### Choose demand")
    energy_options = ["Use VDE_TOTAL"]
    if resolve_vde_energy_values(vde_row)["vde_net_mj_per_km"] is not None:
        energy_options = ["Use VDE_NET (recommended)", "Use VDE_TOTAL"]
        current_basis_label = "Use VDE_NET (recommended)" if st.session_state.get("pwt_energy_basis") == "VDE_NET" else "Use VDE_TOTAL"
    else:
        current_basis_label = "Use VDE_TOTAL"
    if st.session_state.get("pwt_energy_basis_label") not in energy_options:
        st.session_state["pwt_energy_basis_label"] = current_basis_label
    st.radio(
        "Vehicle demand basis for PSE conversion",
        energy_options,
        horizontal=True,
        index=energy_options.index(current_basis_label),
        key="pwt_energy_basis_label",
    )
    st.session_state["pwt_energy_basis"] = "VDE_NET" if st.session_state.get("pwt_energy_basis_label") == "Use VDE_NET (recommended)" else "VDE_TOTAL"
    ctx = get_build_scenario_context(vde_id, vde_row)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("VDE", f"#{int(vde_id)}")
    d2.metric("Cycle", str(vde_row.get("cycle_name") or "-"))
    d3.metric("Demand basis", str(ctx.get("energy_basis") or "-"))
    d4.metric(f"Demand [{unit_label('energy_per_distance', _current_unit_system())}]", _format_demand_value(ctx.get("energy_value_mj_per_km"), unavailable="Pending").replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""))
    return ctx


def _render_baseline_source_stage(reference_summary: dict[str, Any], *, observed_available: bool) -> None:
    st.markdown("#### Choose baseline")
    selected_source = _reference_type_display_label(reference_summary.get("source_type") or "-")
    s1, s2 = st.columns(2)
    s1.metric("Selected source", selected_source)
    s2.metric("Reference vehicle", str(reference_summary.get("reference_vehicle_label") or "-"))
    if reference_summary.get("source_type") == "Same vehicle fuelcons_db line" and not observed_available:
        st.caption("No same-vehicle observed data found. Choose another baseline source, assume efficiency, or enter an observed/imported result.")
    if reference_summary.get("observed_fuel") is not None or reference_summary.get("observed_pse") is not None:
        r1, r2 = st.columns(2)
        r1.metric(f"Observed fuel [{_fuel_display_unit()}]", _format_fuel_value(reference_summary.get("observed_fuel"), unavailable="Pending").replace(f" {_fuel_display_unit()}", ""))
        r2.metric("Observed PSE", _format_metric_value(reference_summary.get("observed_pse"), format_str="%.3f") if reference_summary.get("observed_pse") is not None else "Pending")
    else:
        st.caption("Choose another baseline source, assume efficiency, or enter an observed/imported result.")
def _render_confirmed_baseline_section(
    *,
    vde_id: int,
    vde_row: dict,
    ctx: dict[str, Any],
    active_method: str,
    observed_available: bool,
    readiness: dict[str, Any],
    ml_state: dict[str, Any],
    regression_state: dict[str, Any],
    request: Any,
    result: Any,
    reference_summary: dict[str, Any],
    pse_summary: dict[str, Any],
    confirmed_method: str | None,
    confirmed_snapshot: dict[str, Any] | None,
    confirmed_result: Any,
    confirmed_pse_summary: dict[str, Any],
    confidence_reason: str,
) -> None:
    st.markdown("#### Confirm baseline")
    st.caption(
        "Using: "
        + f"{str(ctx.get('energy_basis') or '-')} {_format_demand_value(ctx.get('energy_value_mj_per_km'), unavailable='Pending')}"
        + " + "
        + _reference_type_display_label(reference_summary.get("source_type") or "-")
    )
    if active_method == "Manual / Imported":
        render_manual_imported_inputs()
    elif active_method == "Observed / Derived PSE":
        _render_reference_rebase_explanation(
            reference_summary=reference_summary,
            ctx=ctx,
            baseline_fuel=to_float(result.fuel_l_100km) if result else None,
            baseline_pse=to_float(pse_summary.get("value")),
        )
    elif active_method == "Physics Simple":
        p1, p2, p3 = st.columns(3)
        p1.metric("Energy basis", str(ctx.get("energy_basis") or "-"))
        p2.metric(
            f"Energy used [{unit_label('energy_per_distance', _current_unit_system())}]",
            _format_demand_value(ctx.get("energy_value_mj_per_km"), unavailable="-").replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""),
        )
        p3.metric("Electrification", str(ctx.get("electrification") or "-"))
        st.caption("Assume efficiency reads the shared powertrain context below and converts vehicle demand with explicit efficiency assumptions.")
    elif active_method == "Regression":
        render_regression_inputs(vde_id, vde_row, ctx, ctx.get("energy_value_mj_per_km"))
    elif active_method == "ML Prediction":
        render_ml_prediction_inputs(vde_id, vde_row, ctx, ctx.get("energy_value_mj_per_km"))
    else:
        st.warning(_method_storyline(active_method))

    st.markdown("#### Baseline result")
    if request is None or result is None:
        st.info(
            _baseline_pending_message(
                active_method=active_method,
                observed_available=observed_available,
                readiness=readiness,
                ml_state=ml_state,
                regression_state=regression_state,
            )
        )
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric(f"Fuel [{_fuel_display_unit()}]", _format_fuel_value(result.fuel_l_100km, unavailable="Pending").replace(f" {_fuel_display_unit()}", ""))
        p2.metric(
            "PSE",
            _format_metric_value(pse_summary.get("value"), format_str="%.3f")
            if pse_summary.get("value") is not None
            else "Pending",
        )
        p3.metric(f"CO2 [{unit_label('co2_per_distance', _current_unit_system())}]", _format_co2_value(result.gco2_km, unavailable="Pending").replace(f" {unit_label('co2_per_distance', _current_unit_system())}", ""))
        p4.metric("Confidence", str((result.confidence if result else "-") or "-").replace("_", " ").title())
        st.caption(_pse_help_text())
        st.caption(_pwt_method_label(active_method))
        st.caption(confidence_reason)
        if pse_summary.get("value") is None:
            st.caption(_pse_pending_message(active_method))

    st.markdown("#### Chosen baseline")
    if confirmed_method is None:
        st.info("No baseline confirmed yet. Review the baseline result and click `Confirm baseline`.")
    else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Method", _pwt_method_label(confirmed_method))
        b2.metric(
            f"Fuel [{_fuel_display_unit()}]",
            _format_fuel_value(confirmed_result.fuel_l_100km if confirmed_result else None, unavailable="-").replace(f" {_fuel_display_unit()}", ""),
        )
        b3.metric(
            "PSE",
            _format_metric_value(confirmed_pse_summary.get("value"), format_str="%.3f")
            if confirmed_pse_summary.get("value") is not None
            else "Pending",
        )
        b4.metric("Confidence", str((confirmed_result.confidence if confirmed_result else "-") or "-").replace("_", " ").title())
        confirmed_reference_label = str((confirmed_snapshot or {}).get("reference_summary", {}).get("source_label") or "")
        if confirmed_reference_label:
            st.caption(f"Reference locked: {confirmed_reference_label}")
        if confirmed_method != active_method:
            st.caption(
                f"Chosen baseline stays locked to `{_pwt_method_label(confirmed_method)}` while the active workspace is previewing `{_pwt_method_label(active_method)}`."
            )

    cta = st.button("Confirm baseline", use_container_width=True, key="btn_use_this_baseline")
    if cta:
        if request is None or result is None:
            st.warning("Baseline is still pending. Complete the selected method before confirming.")
        else:
            st.session_state["pwt_baseline_confirmed_method"] = active_method
            st.session_state["pwt_confirmed_baseline_snapshot"] = {
                "vde_id": int(vde_id),
                "method": active_method,
                "request": copy.deepcopy(request),
                "result": copy.deepcopy(result),
                "ctx": copy.deepcopy(ctx),
                "reference_summary": copy.deepcopy(reference_summary),
                "readiness": copy.deepcopy(readiness),
            }
            st.success(f"Baseline confirmed with `{_pwt_method_label(active_method)}`.")


def render_powertrain_conversion_workspace(vde_id: int, vde_row: dict) -> Dict[str, Any]:
    ctx = get_build_scenario_context(vde_id, vde_row)
    reference_summary = _selected_powertrain_reference(vde_id, vde_row)
    observed_available = to_float(reference_summary.get("observed_pse")) is not None
    regression_vde = ctx.get("energy_value_mj_per_km")
    readiness = _scenario_feature_readiness_snapshot(vde_id, vde_row, ctx, regression_vde=regression_vde, reference_summary=reference_summary)
    active_method = _resolve_active_pwt_setup_method(vde_id, vde_row, ctx)
    confirmed_method = _confirmed_pwt_setup_method()
    confirmed_snapshot = _confirmed_baseline_snapshot(vde_id)
    ml_state = _ml_method_option_state(vde_id, vde_row, ctx, regression_vde)
    regression_state = _regression_method_option_state(vde_id, vde_row, ctx, regression_vde)
    request = _build_active_fuel_estimate_request(vde_id, vde_row, ctx, regression_vde, reference_summary=reference_summary)
    result = run_fuel_estimation(request) if request is not None else None
    pse_summary = dict((result.assumptions if result else {}).get("pse_summary") or {})
    confirmed_request = confirmed_snapshot.get("request") if confirmed_snapshot else None
    confirmed_result = confirmed_snapshot.get("result") if confirmed_snapshot else None
    if confirmed_method is not None and confirmed_result is None:
        confirmed_request = _build_active_fuel_estimate_request(
            vde_id,
            vde_row,
            ctx,
            regression_vde,
            method_label=confirmed_method,
            reference_summary=reference_summary,
        )
        confirmed_result = run_fuel_estimation(confirmed_request) if confirmed_request is not None else None
    confirmed_pse_summary: dict[str, Any] = dict((confirmed_result.assumptions if confirmed_result else {}).get("pse_summary") or {})
    recommended_method = _preferred_pwt_setup_method(vde_id, vde_row, ctx)
    confidence_reason = _confidence_reason_label(
        readiness=readiness,
        active_method=active_method,
        reference_summary=reference_summary,
        regression_state=regression_state,
    )
    main_col, options_col = st.columns([0.72, 0.28])
    with options_col:
        _render_baseline_side_options_panel(
            vde_id=vde_id,
            vde_row=vde_row,
            ctx=ctx,
            readiness=readiness,
            regression_vde=regression_vde,
            reference_summary=reference_summary,
        )

    with main_col:
        if _current_pwt_input_mode() == "Guided":
            st.caption("Follow the baseline calculation top to bottom: define demand, choose the baseline source, review the active method, then lock the chosen baseline.")
        else:
            st.caption("Spreadsheet Assist keeps demand, source, and baseline confirmation visible in one compact workspace.")

        ctx = _render_baseline_demand_stage(vde_id, vde_row)
        st.divider()
        _render_baseline_source_stage(reference_summary, observed_available=observed_available)
        st.divider()
        _render_confirmed_baseline_section(
            vde_id=vde_id,
            vde_row=vde_row,
            ctx=ctx,
            active_method=active_method,
            observed_available=observed_available,
            readiness=readiness,
            ml_state=ml_state,
            regression_state=regression_state,
            request=request,
            result=result,
            reference_summary=reference_summary,
            pse_summary=pse_summary,
            confirmed_method=confirmed_method,
            confirmed_snapshot=confirmed_snapshot,
            confirmed_result=confirmed_result,
            confirmed_pse_summary=confirmed_pse_summary,
            confidence_reason=confidence_reason,
        )

    return ctx


def render_estimation_engine_panel(vde_id: int, vde_row: dict) -> Dict[str, Any]:
    return render_powertrain_conversion_workspace(vde_id, vde_row)


def render_technology_proposal_workspace(vde_id: int, vde_row: dict) -> None:
    draft = _build_powertrain_scenario_draft(vde_id, vde_row)
    confirmed_method = _confirmed_pwt_setup_method()
    baseline_result = draft["baseline_estimate"]["result"]
    proposal = dict(draft.get("proposal_result") or {})
    deltas = list(draft.get("technology_deltas") or [])
    counts = dict(proposal.get("delta_counts") or _delta_status_counts(deltas))
    baseline = dict(proposal.get("baseline") or {})
    proposal_metrics = dict(proposal.get("proposal") or {})
    basis_options = _delta_basis_select_options()
    current_basis = _compact_delta_basis_label(st.session_state.get("pwt_delta_effect_basis"))
    if current_basis in basis_options:
        st.session_state["pwt_delta_effect_basis"] = current_basis

    st.subheader("Technology Proposal")
    st.caption("Apply a simple technology delta on top of the baseline and preview the proposal instantly.")
    if confirmed_method is None:
        st.info("Confirm baseline before applying a technology delta.")
        return
    if baseline_result is None:
        st.info("Baseline pending - estimate the baseline powertrain before staging technology deltas.")
        return
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(f"Baseline [{_fuel_display_unit()}]", _format_fuel_value(baseline.get("fuel_l_100km"), unavailable="-").replace(f" {_fuel_display_unit()}", ""))
    s2.metric("Delta status", "No delta" if not deltas else str(proposal.get("status") or "Pending"))
    s3.metric(f"Proposal [{_fuel_display_unit()}]", "Same as baseline" if not deltas else _format_fuel_value(proposal_metrics.get("fuel_l_100km"), unavailable="-").replace(f" {_fuel_display_unit()}", ""))
    s4.metric("Confidence", str(proposal.get("confidence") or "-").replace("_", " ").title())

    st.markdown("#### Delta")
    delta_df = _build_delta_editor_df()
    edited_delta_df = st.data_editor(
        delta_df,
        key="pwt_delta_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["source_method", "status"],
        column_config={
            "delta_label": st.column_config.TextColumn("delta_label"),
            "delta_basis": st.column_config.SelectboxColumn("basis", options=basis_options),
            "delta_value": st.column_config.NumberColumn("value", format="%.3f"),
            "apply_quantitatively": st.column_config.CheckboxColumn("apply"),
            "source_method": st.column_config.TextColumn("source/method"),
            "confidence": st.column_config.SelectboxColumn("confidence", options=DELTA_CONFIDENCE_OPTIONS),
            "status": st.column_config.TextColumn("status"),
            "notes": st.column_config.TextColumn("notes"),
        },
    )
    delta_errors = _apply_delta_editor_df(edited_delta_df)
    for error in delta_errors:
        st.warning(error)

    with st.expander("Advanced delta metadata", expanded=_show_technical_details()):
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Affected subsystem", DELTA_SUBSYSTEM_OPTIONS, key="pwt_delta_subsystem")
        c2.selectbox("Source type", DELTA_SOURCE_TYPE_OPTIONS, key="pwt_delta_source_type")
        c3.selectbox("Maturity level", DELTA_MATURITY_OPTIONS, key="pwt_delta_maturity")
        st.text_input("Reference / source description", key="pwt_delta_reference")
        st.text_area("Notes", key="pwt_delta_notes", height=70)

    d1, d2 = st.columns(2)
    if d1.button("Add delta", use_container_width=True, key="btn_add_pwt_delta"):
        deltas = list(st.session_state.get("pwt_technology_deltas") or [])
        deltas.append(
            {
                "id": len(deltas) + 1,
                "name": str(st.session_state.get("pwt_delta_name") or f"Delta {len(deltas) + 1}").strip(),
                "affected_subsystem": st.session_state.get("pwt_delta_subsystem"),
                "source_type": st.session_state.get("pwt_delta_source_type"),
                "maturity_level": st.session_state.get("pwt_delta_maturity"),
                "effect_basis": st.session_state.get("pwt_delta_effect_basis"),
                "confidence": st.session_state.get("pwt_delta_confidence"),
                "effect_value": to_float(st.session_state.get("pwt_delta_value")),
                "enabled": bool(st.session_state.get("pwt_delta_apply_toggle")),
                "notes": str(st.session_state.get("pwt_delta_notes") or "").strip(),
                "reference_description": str(st.session_state.get("pwt_delta_reference") or "").strip(),
            }
        )
        st.session_state["pwt_technology_deltas"] = deltas
        _reset_delta_form()
        st.success("Technology delta staged.")
    if d2.button("Reset draft delta", use_container_width=True, key="btn_reset_pwt_delta"):
        _reset_delta_form()
        st.success("Draft delta cleared.")

    removable = [f"#{delta.get('id')} - {delta.get('name')}" for delta in deltas if not delta.get("is_preview_only")]
    if removable:
        with st.expander("Advanced: staged deltas", expanded=_show_technical_details()):
            rows = []
            for delta in deltas:
                if delta.get("is_preview_only"):
                    continue
                rows.append(
                    {
                        "delta": delta.get("name"),
                        "basis": _compact_delta_basis_label(delta.get("effect_basis")),
                        "value": delta.get("effect_value"),
                        "confidence": str(delta.get("confidence") or "-").title(),
                        "status": str(delta.get("quantitative_status") or "-").replace("_", " "),
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.selectbox("Remove delta", removable, key="pwt_delta_remove_id")
            if st.button("Remove selected", use_container_width=True, key="btn_remove_pwt_delta"):
                selected_label = str(st.session_state.get("pwt_delta_remove_id") or "")
                keep = []
                for delta in deltas:
                    if delta.get("is_preview_only"):
                        continue
                    label = f"#{delta.get('id')} - {delta.get('name')}"
                    if label != selected_label:
                        keep.append(delta)
                for index, delta in enumerate(keep, start=1):
                    delta["id"] = index
                st.session_state["pwt_technology_deltas"] = keep
                st.success("Selected delta removed.")

    st.markdown("#### Proposal Preview")
    lead_delta = deltas[-1] if deltas else _draft_delta_from_form()
    lead_delta_value = to_float((lead_delta or {}).get("effect_value"))
    lead_delta_label = _compact_delta_basis_label((lead_delta or {}).get("effect_basis"))
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(f"Baseline [{_fuel_display_unit()}]", _format_fuel_value(baseline.get("fuel_l_100km"), unavailable="-").replace(f" {_fuel_display_unit()}", ""))
    p2.metric(
        "Delta",
        f"{lead_delta_value:+.1f}%" if lead_delta_value is not None and "multiplier" not in lead_delta_label.lower() else (
            f"{lead_delta_value:.3f}" if lead_delta_value is not None else "0"
        ),
        lead_delta_label if lead_delta else "Proposal equals baseline",
    )
    p3.metric(f"Proposal [{_fuel_display_unit()}]", "Same as baseline" if not deltas else _format_fuel_value(proposal_metrics.get("fuel_l_100km"), unavailable="-").replace(f" {_fuel_display_unit()}", ""))
    fuel_delta = None
    if to_float(baseline.get("fuel_l_100km")) is not None and to_float(proposal_metrics.get("fuel_l_100km")) is not None:
        fuel_delta = to_float(proposal_metrics.get("fuel_l_100km")) - to_float(baseline.get("fuel_l_100km"))
    p4.metric(f"Delta absolute [{_fuel_display_unit()}]", _format_fuel_value(fuel_delta, unavailable="-").replace(f" {_fuel_display_unit()}", ""))

    preview_rows = _build_baseline_proposal_rows(baseline, proposal_metrics)
    if preview_rows:
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
    if proposal.get("status") == "No quantitative delta" or (not deltas and baseline_result is not None):
        st.info("Proposal equals baseline.")
    elif proposal.get("registered_only_deltas"):
        st.caption("Registered only - no quantitative effect applied.")


def render_results_save_tab(vde_id: int, vde_row: dict) -> None:
    ctx = get_build_scenario_context(vde_id, vde_row)
    _resolve_active_pwt_setup_method(vde_id, vde_row, ctx)
    regression_vde = ctx.get("energy_value_mj_per_km")
    render_fuel_review_save_panel(vde_id, vde_row, ctx, regression_vde)


def render_pwt_estimation_method_selector() -> str:
    current = str(st.session_state.get("pwt_setup_method") or PWT_ESTIMATION_METHODS[0])
    if current not in PWT_ESTIMATION_METHODS:
        current = PWT_ESTIMATION_METHODS[0]
        st.session_state["pwt_setup_method"] = current
    return current


def _default_lhv(fuel_type: str) -> float:
    return float(LHV_MJ_PER_L.get(fuel_type or "Gasoline", LHV_MJ_PER_L["Gasoline"]))


def _default_gco2_per_l(fuel_type: str) -> float:
    return float(GCO2_PER_L.get(fuel_type or "Gasoline", GCO2_PER_L["Gasoline"]))


def _build_powertrain_features_from_state(vde_row: dict, ctx: Dict[str, Any]) -> Dict[str, Any]:
    electrification = str(ctx.get("electrification") or "ICE").upper()
    powertrain_features: Dict[str, Any] = {}

    gear_count = st.session_state.get("pwt_gears") or vde_row.get("gear_count")
    final_drive_ratio = st.session_state.get("pwt_fdr") or vde_row.get("final_drive_ratio")
    transmission_model = st.session_state.get("pwt_trans_model") or vde_row.get("transmission_model")
    transmission_type = st.session_state.get("pwt_feature_transmission_type")
    drive_type = st.session_state.get("pwt_feature_drive_type")
    engine_size_l = to_float(st.session_state.get("pwt_feature_engine_size_l"))
    power_hp = to_float(st.session_state.get("pwt_feature_power_hp"))

    if gear_count not in (None, ""):
        powertrain_features["gear_count"] = int(gear_count)
    if final_drive_ratio not in (None, ""):
        powertrain_features["final_drive_ratio"] = float(final_drive_ratio)
    if transmission_model not in (None, ""):
        powertrain_features["transmission_model"] = transmission_model
    if transmission_type not in (None, "", "(inherit)"):
        powertrain_features["transmission_type"] = transmission_type
    if drive_type not in (None, "", "(inherit)"):
        powertrain_features["drive_type"] = drive_type
    if engine_size_l is not None:
        powertrain_features["engine_size_l"] = float(engine_size_l)
    if power_hp is not None:
        powertrain_features["engine_max_power_kw"] = float(power_hp) / 1.34102209

    if electrification in ("ICE", "HEV", "PHEV"):
        selected_fuel_type = st.session_state.get("sb_fuel_type")
        explicit_fuel_type = None if selected_fuel_type in (None, "", "(leave missing)") else selected_fuel_type
        fuel_type_for_calcs = explicit_fuel_type or "Gasoline"
        if explicit_fuel_type is not None:
            powertrain_features["fuel_type"] = explicit_fuel_type
        powertrain_features["LHV_MJ_per_L"] = float(
            to_float(st.session_state.get("sb_lhv_override")) or _default_lhv(fuel_type_for_calcs)
        )
        powertrain_features["gCO2_per_L"] = _default_gco2_per_l(fuel_type_for_calcs)
        eta_pt = to_float(st.session_state.get("sb_eta_pt"))
        if eta_pt and eta_pt > 0:
            powertrain_features["eta_pt_est"] = float(eta_pt)
        uf_phev = to_float(st.session_state.get("sb_uf"))
        if uf_phev is not None:
            powertrain_features["utility_factor"] = max(0.0, min(1.0, float(uf_phev)))

    if electrification in ("BEV", "PHEV"):
        eta_drive = to_float(st.session_state.get("sb_eta_drive"))
        grid = to_float(st.session_state.get("sb_grid"))
        if eta_drive and eta_drive > 0:
            powertrain_features["bev_eff_drive"] = float(eta_drive)
        if grid is not None:
            powertrain_features["grid_gco2_per_kwh"] = float(grid)

    return powertrain_features


def _render_scenario_extras_inputs(vde_id: int, vde_row: dict, ctx: Dict[str, Any]) -> None:
    readiness = _scenario_feature_readiness_snapshot(
        vde_id,
        vde_row,
        ctx,
        regression_vde=ctx.get("energy_value_mj_per_km"),
    )
    values = dict(readiness.get("values") or {})
    with st.expander("Scenario feature context", expanded=False):
        st.caption("Scenario features are confirmed in Scenario Pairing and reused here by ML, Regression, and Peers.")
        st.caption(_scenario_override_label())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gear count", _format_feature_value(values.get("gear_count"), feature_key="gear_count"))
        c2.metric("Final drive", _format_feature_value(values.get("final_drive_ratio"), feature_key="final_drive_ratio"))
        c3.metric("Transmission", _format_feature_value(values.get("transmission_type"), feature_key="transmission_type"))
        c4.metric("Drive type", _format_feature_value(values.get("drive_type"), feature_key="drive_type"))
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Engine size", _format_feature_value(values.get("engine_size_l"), feature_key="engine_size_l"))
        d2.metric("Power", _format_feature_value(values.get("engine_max_power_kw"), feature_key="engine_max_power_kw"))
        d3.metric("Fuel type", _format_feature_value(values.get("fuel_type"), feature_key="fuel_type"))
        d4.metric("Readiness", str(readiness.get("status_label") or "-"))
        st.caption(str(readiness.get("status_detail") or ""))
        trans_models = fetch_distinct_transmission_models()
        trans_models.append("Other...")
        t1, t2 = st.columns([1.2, 1.8])
        choice = t1.selectbox("Transmission model", trans_models, key="pwt_trans_model_choice")
        tm_value = t2.text_input("Custom transmission model", key="pwt_trans_model_custom") if choice == "Other..." else choice
        st.session_state["pwt_trans_model"] = (tm_value or "").strip() or None


def render_powertrain_conversion_inputs(vde_row: dict, ctx: Dict[str, Any]) -> None:
    st.markdown("#### Efficiency / Energy Conversion Inputs")
    electrification = str(ctx.get("electrification") or "ICE").upper()
    energy_basis = str(ctx.get("energy_basis") or "VDE_TOTAL").upper()
    energy_values = resolve_vde_energy_values(vde_row)
    selected_vde = energy_values["vde_total_mj_per_km"] if energy_basis == "VDE_TOTAL" else energy_values["vde_net_mj_per_km"]

    c_info1, c_info2, c_info3 = st.columns(3)
    c_info1.metric("Energy basis", energy_basis)
    c_info2.metric(
        f"Selected VDE [{unit_label('energy_per_distance', _current_unit_system())}]",
        _format_demand_value(selected_vde, unavailable="-").replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""),
    )
    c_info3.metric("Electrification", electrification)
    if energy_values["warnings"]:
        st.caption(f"Energy warnings: {', '.join(energy_values['warnings'])}")

    if electrification == "PHEV" and "sb_uf" not in st.session_state:
        st.session_state["sb_uf"] = 0.50

    if electrification in ("ICE", "HEV", "PHEV"):
        st.markdown("**Fuel path parameters**")
        c1, c2, c3 = st.columns(3)
        c1.number_input("eta_pt (fuel path)", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="sb_eta_pt")
        c2.selectbox("Fuel type", ["(leave missing)"] + FUEL_TYPE_OPTIONS, key="sb_fuel_type")
        c3.number_input("LHV [MJ/L] (optional override)", min_value=0.0, step=0.1, format="%.2f", key="sb_lhv_override")
        if electrification == "PHEV":
            st.number_input("Utility factor (0-1)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="sb_uf")

    if electrification in ("BEV", "PHEV"):
        st.markdown("**Electric path parameters**")
        c1, c2 = st.columns(2)
        c1.number_input("Driveline efficiency", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="sb_eta_drive")
        c2.number_input("Grid [gCO2/kWh]", min_value=0.0, step=1.0, format="%.0f", key="sb_grid")


def render_manual_imported_inputs() -> None:
    st.markdown("#### Enter Observed/Imported Result")
    st.caption("Register measured, imported or official values without recalculating from VDE energy.")
    manual_df = _build_manual_baseline_editor_df()
    edited_df = st.data_editor(
        manual_df,
        key="pwt_manual_baseline_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["status"],
        column_config={
            "fuel_l_100km": st.column_config.NumberColumn("fuel_l_100km", min_value=0.0, format="%.2f"),
            "energy_Wh_km": st.column_config.NumberColumn("energy_Wh_km", min_value=0.0, format="%.1f"),
            "co2_g_km": st.column_config.NumberColumn("co2_g_km", min_value=0.0, format="%.1f"),
            "source": st.column_config.TextColumn("source/method"),
            "confidence": st.column_config.SelectboxColumn("confidence", options=DELTA_CONFIDENCE_OPTIONS),
            "status": st.column_config.TextColumn("status"),
            "notes": st.column_config.TextColumn("notes"),
        },
    )
    errors = _apply_manual_baseline_editor_df(edited_df)
    for error in errors:
        st.warning(error)


def _regression_dataset_warnings(regdf: pd.DataFrame) -> list[str]:
    row_count = len(regdf)
    warnings: list[str] = []
    if row_count == 0:
        warnings.append("regression_dataset_empty")
    elif row_count < 5:
        warnings.append("regression_dataset_insufficient")
    elif row_count < 15:
        warnings.append("regression_dataset_small")
    elif row_count < 30:
        warnings.append("regression_dataset_moderate")
    return warnings


def _render_regression_dataset_feedback(warnings: list[str], row_count: int) -> None:
    if "regression_dataset_empty" in warnings:
        st.error("No peer records matched the active filters. Adjust the Regression dataset before saving this scenario.")
    elif "regression_dataset_insufficient" in warnings:
        st.error(
            f"Regression sample is too small ({row_count} records). Relax filters or use ML/Physics fallback."
        )
    elif "regression_dataset_small" in warnings:
        st.warning(f"Small regression dataset: {row_count} records. Review the scatter and model summary before trusting the estimate.")
    elif "regression_dataset_moderate" in warnings:
        st.info(f"Regression sample is usable ({row_count} records), but a broader peer sample would improve confidence.")


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
            "candidate_filters": {},
            "candidate_dataset": pd.DataFrame(),
            "dataset": pd.DataFrame(),
            "model": {},
            "payload": {},
            "sample_quality": _regression_sample_quality(0),
            "warnings": ["regression_energy_missing"],
        }

    if render_filters:
        reg_filters = filters_bar(
            vde_id,
            ctx["electrification"],
            key_ns="regression",
            allow_current_vehicle_scope=False,
        )
        reg_filters["legislation"] = vde_row.get("legislation")
        st.session_state["pwt_regression_filters"] = dict(reg_filters)
    else:
        reg_filters = dict(st.session_state.get("pwt_regression_filters") or {})
        reg_filters.setdefault("legislation", vde_row.get("legislation"))

    candidate_filters = _regression_candidate_pool_filters(reg_filters)
    candidate_df = load_regression_dataset(candidate_filters)
    regdf = load_regression_dataset(reg_filters)
    warnings = _regression_dataset_warnings(regdf)
    sample_quality = _regression_sample_quality(len(regdf))
    if regdf.empty:
        return {
            "filters": reg_filters,
            "candidate_filters": candidate_filters,
            "candidate_dataset": candidate_df,
            "dataset": regdf,
            "model": {},
            "payload": {},
            "sample_quality": sample_quality,
            "warnings": warnings,
        }
    if not sample_quality["can_fit"]:
        return {
            "filters": reg_filters,
            "candidate_filters": candidate_filters,
            "candidate_dataset": candidate_df,
            "dataset": regdf,
            "model": {},
            "payload": {},
            "sample_quality": sample_quality,
            "warnings": warnings,
        }

    model = fit_regression_y_vs_vde(regdf, y_col=None, electrification=ctx["electrification"])
    yhat = predict_current_consumption(model, regression_vde, ctx["electrification"])
    payload = build_min_payload(vde_id, ctx["electrification"], yhat, method_note="regression_existing")
    payload = enrich_with_derivatives(payload, ctx["electrification"], fuel_type="Gasoline")
    return {
        "filters": reg_filters,
        "candidate_filters": candidate_filters,
        "candidate_dataset": candidate_df,
        "dataset": regdf,
        "model": model,
        "payload": payload,
        "sample_quality": sample_quality,
        "warnings": warnings,
    }


def _build_regression_runner(vde_id: int, electrification: str, filters: Dict[str, Any]):
    def regression_runner(request_dict, vde_mj_per_km):
        candidate_filters = _regression_candidate_pool_filters(filters)
        candidate_df = load_regression_dataset(candidate_filters)
        regdf = load_regression_dataset(filters)
        dataset_warnings = _regression_dataset_warnings(regdf)
        sample_quality = _regression_sample_quality(len(regdf))
        if sample_quality["can_fit"]:
            model = fit_regression_y_vs_vde(regdf, y_col=None, electrification=electrification)
            yhat = predict_current_consumption(model, vde_mj_per_km, electrification)
            payload = build_min_payload(vde_id, electrification, yhat, method_note="regression_existing")
            payload = enrich_with_derivatives(payload, electrification, fuel_type="Gasoline")
        else:
            model = {}
            payload = {}
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
                "candidate_pool_rows": len(candidate_df),
                "dataset_rows": len(regdf),
                "sample_quality": sample_quality["label"],
                "model_summary": {
                    "urban": model.get("urb"),
                    "highway": model.get("hw"),
                    "combined": model.get("combined"),
                },
            },
            "confidence": "low" if dataset_warnings else "high",
        }

    return regression_runner


def render_regression_inputs(vde_id: int, vde_row: dict, ctx: Dict[str, Any], regression_vde: float | None) -> None:
    st.markdown("#### Regression Inputs")
    if regression_vde is None:
        st.warning("Regression preview requires VDE energy, but neither VDE_NET nor VDE_TOTAL is available on this snapshot.")
        return
    regression_state = _resolve_regression_state(vde_id, vde_row, ctx, regression_vde, render_filters=True)
    candidate_df = regression_state["candidate_dataset"]
    regdf = regression_state["dataset"]
    model = regression_state["model"]
    payload = regression_state["payload"]
    warnings = regression_state["warnings"]
    sample_quality = regression_state["sample_quality"]
    _render_regression_dataset_feedback(warnings, len(regdf))

    s1, s2, s3 = st.columns(3)
    s1.metric("Filtered sample", str(len(regdf)))
    s2.metric("Sample quality", str(sample_quality["label"]))
    s3.metric(
        f"Estimated fuel [{_fuel_display_unit()}]",
        _format_fuel_value(payload.get("fuel_l_per_100km"), unavailable="Pending").replace(f" {_fuel_display_unit()}", ""),
    )

    if not sample_quality["can_fit"]:
        st.info("Regression does not run below 5 filtered records. Relax filters or use ML/Physics fallback.")
        return

    with st.expander("Advanced: peer/regression details", expanded=_show_technical_details()):
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Target VDE", f"#{int(vde_id)}")
        t2.metric("Vehicle", f"{str(vde_row.get('make') or '-')} {str(vde_row.get('model') or '-')}".strip())
        t3.metric("Demand basis", str(ctx.get("energy_basis") or "-"))
        t4.metric(
            f"Target demand [{unit_label('energy_per_distance', _current_unit_system())}]",
            _format_demand_value(regression_vde, unavailable="Pending").replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""),
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("Candidate pool", str(len(candidate_df)))
        d2.metric("Filters", str(len(regression_state["filters"])))
        d3.metric("Applied filters", _regression_filters_summary(regression_state["filters"]))
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Model (Urban)**")
        c1.write(model.get("urb"))
        c2.markdown("**Model (Highway)**")
        c2.write(model.get("hw"))
        c3.markdown("**Model (Combined)**")
        c3.write(model.get("combined"))
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
        p1.metric(f"Estimated fuel [{_fuel_display_unit()}]", _format_fuel_value(payload.get("fuel_l_per_100km"), unavailable="Pending").replace(f" {_fuel_display_unit()}", ""))
        p2.metric(f"Estimated energy [{unit_label('energy_wh_per_distance', _current_unit_system())}]", _format_energy_value(payload.get("energy_Wh_per_km"), unavailable="Pending").replace(f" {unit_label('energy_wh_per_distance', _current_unit_system())}", ""))
        p3.metric(f"Estimated CO2 [{unit_label('co2_per_distance', _current_unit_system())}]", _format_co2_value(payload.get("gco2_per_km"), unavailable="Pending").replace(f" {unit_label('co2_per_distance', _current_unit_system())}", ""))
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
    readiness = _scenario_feature_readiness_snapshot(vde_id, vde_row, ctx, regression_vde=regression_vde)

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

    c1, c2, c3 = st.columns(3)
    c1.metric("Artifact", status.replace("_", " ").title())
    c2.metric("Available features", str(len(setup.get("features", {}).get("available_feature_names") or [])))
    c3.metric("Missing features", str(len(setup.get("features", {}).get("missing_features") or [])))
    if (setup.get("features", {}).get("missing_features") or []):
        st.info("Complete powertrain metadata in Scenario Pairing.")
    st.caption(_simple_readiness_status(readiness)[1])

    with st.expander("Advanced: model diagnostics", expanded=_show_technical_details()):
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

    g1, g2, g3 = st.columns(3)
    g1.metric("Peer count", str(summary.get("peer_count", 0)))
    g2.metric("Peer quality", str(quality.get("label") or "-"))
    g3.metric("Coverage", guidance_status)

    if not peer_metrics.empty:
        compact_metrics = peer_metrics.copy()
        compact_metrics = compact_metrics[compact_metrics["metric"].isin(["fuel_l_per_100km", "gco2_per_km", "energy_Wh_per_km", "vde_total_mj_per_km", "vde_net_mj_per_km"])]
        show_cols = [
            col
            for col in ["label", "median", "std_dev", "min", "max"]
            if col in compact_metrics.columns
        ]
        if show_cols:
            with st.expander("Advanced: nearest peers", expanded=_show_technical_details()):
                st.dataframe(compact_metrics[show_cols], use_container_width=True, hide_index=True)

    if hints:
        with st.expander("Advanced: investigation hints", expanded=_show_technical_details()):
            for hint in hints[:3]:
                st.info(
                    f"{hint.get('hint')}\n\n"
                    f"Evidence: {hint.get('evidence')}\n\n"
                    f"Next data to inspect: {hint.get('next_data')}"
                )


def _fuel_scenario_extra_payload() -> dict:
    payload = {
        "gear_count": int(to_float(st.session_state.get("pwt_gears"))) if to_float(st.session_state.get("pwt_gears")) is not None else None,
        "final_drive_ratio": to_float(st.session_state.get("pwt_fdr")),
    }
    return {k: v for k, v in payload.items() if v not in (None, "")}


def _proposal_save_overrides(result: Any, draft: dict[str, Any]) -> dict[str, Any]:
    proposal_summary = dict(draft.get("proposal_result") or {})
    proposal_metrics = dict(proposal_summary.get("proposal") or {})
    baseline_summary = dict(draft.get("baseline_estimate") or {})
    result.request.vehicle_features["powertrain_reference"] = dict(draft.get("powertrain_reference") or {})
    result.request.vehicle_features["baseline_estimate"] = {
        "method": baseline_summary.get("method"),
        "confidence": baseline_summary.get("confidence"),
        "warnings": baseline_summary.get("warnings"),
    }
    result.request.vehicle_features["technology_deltas"] = list(draft.get("technology_deltas") or [])
    result.request.vehicle_features["proposal_result"] = proposal_summary
    result.request.vehicle_features["scenario_lineage"] = {
        "vde_source": draft.get("vde_source"),
        "powertrain_reference": draft.get("powertrain_reference", {}).get("source_label"),
        "baseline_method": baseline_summary.get("method"),
        "technology_delta_count": len(draft.get("technology_deltas") or []),
    }
    result.assumptions["powertrain_reference"] = dict(draft.get("powertrain_reference") or {})
    result.assumptions["baseline_estimate"] = {
        "method": baseline_summary.get("method"),
        "confidence": baseline_summary.get("confidence"),
        "warnings": baseline_summary.get("warnings"),
    }
    result.assumptions["technology_deltas"] = list(draft.get("technology_deltas") or [])
    result.assumptions["proposal_result"] = proposal_summary
    payload = dict(_fuel_scenario_extra_payload())
    payload.update(
        {
            "fuel_l_per_100km": proposal_metrics.get("fuel_l_100km"),
            "energy_Wh_per_km": proposal_metrics.get("energy_Wh_km"),
            "gco2_per_km": proposal_metrics.get("gco2_km"),
            "method_note": f"baseline={baseline_summary.get('method') or '-'} | proposal={proposal_summary.get('status') or '-'}",
        }
    )
    return {key: value for key, value in payload.items() if value not in (None, "")}


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


def _saved_update_target_options(
    df: pd.DataFrame,
    current_vde_row: dict,
) -> tuple[list[str], dict[str, int], dict[int, dict[str, Any]]]:
    options: list[str] = []
    label_to_id: dict[str, int] = {}
    row_lookup: dict[int, dict[str, Any]] = {}
    for _, row in df.sort_values("id", ascending=False).iterrows():
        row_dict = row.to_dict()
        row_id = int(row_dict["id"])
        revision_state = compare_saved_scenario_revision(row_dict.get("source_vde_revision"), current_vde_row)
        result_value = row_dict.get("fuel_l_per_100km")
        result_suffix = "L/100km"
        if pd.isna(result_value):
            result_value = row_dict.get("energy_Wh_per_km")
            result_suffix = "Wh/km"
        result_text = "-" if pd.isna(result_value) else f"{result_value} {result_suffix}"
        method_text = str(row_dict.get("engine_method") or row_dict.get("method_note") or "-")
        state_text = _link_state_label(revision_state["status"])
        label = f"#{row_id} | {state_text} | {method_text} | {result_text}"
        options.append(label)
        label_to_id[label] = row_id
        row_lookup[row_id] = row_dict
    return options, label_to_id, row_lookup


def _render_update_target_status(row: pd.Series | dict, current_vde_row: dict) -> None:
    data = dict(row)
    revision_state = compare_saved_scenario_revision(data.get("source_vde_revision"), current_vde_row)
    _render_link_state_badge(revision_state["status"], context=f"Target #{int(data['id'])}")
    if revision_state["status"] == "changed":
        st.warning(
            "This saved scenario points to an older VDE revision. "
            "Updating it now will refresh the saved scenario to the current VDE source."
        )
    elif revision_state["status"] == "missing":
        st.info(
            "This saved scenario has no recorded source VDE revision. "
            "Updating it now will write the current VDE revision into the saved scenario."
        )
    elif revision_state["status"] == "current":
        st.caption("This saved scenario already matches the current VDE revision.")
    else:
        st.caption(revision_state["message"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target scenario", f"#{int(data['id'])}")
    c2.metric("Method", str(data.get("engine_method") or data.get("method_note") or "-"))
    c3.metric("Saved VDE rev", str(data.get("source_vde_revision") or "-"))
    c4.metric("Link state", _link_state_label(revision_state["status"]))


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
    confidence_summary = _confidence_summary_from_saved_row(data)
    pse_summary = _pse_summary_from_saved_row(data)

    st.markdown(f"#### Scenario #{scenario_id}")
    _render_link_state_badge(revision_state["status"])
    st.caption(str(data.get("created_at") or "-"))
    st.metric("Source VDE", f"#{int(data['vde_id'])}" if pd.notna(data.get("vde_id")) else "-")
    st.metric("Vehicle", f"{(scenario_vde_row or {}).get('make', '-')} {(scenario_vde_row or {}).get('model', '-')}".strip())
    st.metric("Method", str(data.get("engine_method") or data.get("method_note") or "-"))
    st.metric("Energy basis", str(data.get("energy_basis") or "-"))
    st.metric("Confidence", str(confidence_summary.get("label") or provenance.get("confidence") or "-"))
    st.metric("PSE", _format_metric_value(pse_summary.get("value"), format_str="%.3f"))
    st.metric("Fuel [L/100km]", f"{float(data['fuel_l_per_100km']):.2f}" if pd.notna(data.get("fuel_l_per_100km")) else "N/A")
    st.metric("Energy [Wh/km]", f"{float(data['energy_Wh_per_km']):.1f}" if pd.notna(data.get("energy_Wh_per_km")) else "N/A")
    st.metric("CO2 [g/km]", f"{float(data['gco2_per_km']):.1f}" if pd.notna(data.get("gco2_per_km")) else "N/A")
    st.metric("FTP-75", f"{float(data['fuel_ftp75_l_per_100km']):.2f} L/100" if pd.notna(data.get("fuel_ftp75_l_per_100km")) else "N/A")
    st.metric("HWFET", f"{float(data['fuel_hwfet_l_per_100km']):.2f} L/100" if pd.notna(data.get("fuel_hwfet_l_per_100km")) else "N/A")
    _render_bench_badges(list(confidence_summary.get("status_items") or []))
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
    confidence_summary = _confidence_summary_from_saved_row(data)
    pse_summary = _pse_summary_from_saved_row(data)
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
        "confidence_label": confidence_summary.get("label") or provenance.get("confidence") or "-",
        "confidence_statuses": ", ".join(confidence_summary.get("status_items") or []) or "-",
        "pse_value": pse_summary.get("value"),
        "pse_source": pse_summary.get("source") or "-",
        "pse_source_label": pse_summary.get("source_label") or "-",
        "pse_mode": pse_summary.get("mode") or "-",
        "pse_target_type": pse_summary.get("target_type") or "-",
        "pse_status": pse_summary.get("status") or "-",
        "pse_cycle_basis": pse_summary.get("cycle_basis") or "-",
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


def _render_comparison_report_overview(vde_id: int, vde_row: dict) -> None:
    current_vde_row = fetch_vde_row(vde_id) or vde_row
    saved_df = fetch_fuelcons_by_vde(vde_id)
    vde_row_lookup = _build_vde_row_lookup(saved_df)
    energy_values = resolve_vde_energy_values(current_vde_row)
    link_summary = summarize_saved_scenario_revision_states(
        saved_df.to_dict("records") if saved_df is not None and not saved_df.empty else [],
        current_vde_row,
    )

    st.subheader("Executive Summary")
    st.caption("Live report view for the current anchor scenario. Read left to right: Vehicle Demand -> PSE -> Final Result -> Confidence.")

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Anchor VDE", f"#{int(current_vde_row.get('id'))}")
    top2.metric("Saved scenarios", str(len(saved_df)))
    top3.metric("Current links", str(link_summary["current"]))
    top4.metric("Refresh required", str(link_summary["refresh_required"]))

    base1, base2, base3, base4 = st.columns(4)
    base1.metric("Vehicle", f"{current_vde_row.get('make', '-')} {current_vde_row.get('model', '-')}".strip())
    base2.metric("Cycle", str(current_vde_row.get("cycle_name") or "-"))
    base3.metric(f"VDE_TOTAL [{unit_label('energy_per_distance', _current_unit_system())}]", _format_demand_value(energy_values.get("vde_total_mj_per_km")).replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""))
    base4.metric(f"VDE_NET [{unit_label('energy_per_distance', _current_unit_system())}]", _format_demand_value(energy_values.get("vde_net_mj_per_km")).replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""))
    if link_summary["total"]:
        st.caption(
            f"Anchor link health: {link_summary['current']} current, "
            f"{link_summary['changed']} changed, {link_summary['missing']} missing revision metadata."
        )

    if saved_df.empty:
        st.info("No saved Powertrain Scenario is linked to this VDE yet. Save at least one scenario to populate the live report story.")
        return

    latest_row = saved_df.sort_values("id", ascending=False).iloc[0]
    provenance = _load_json_blob(latest_row.get("provenance_json"))
    confidence_summary = _confidence_summary_from_saved_row(latest_row)
    pse_summary = _pse_summary_from_saved_row(latest_row)
    basis_label = str(latest_row.get("energy_basis") or "-")
    basis_value = provenance.get("energy_basis_value")
    if basis_value is None:
        basis_value = energy_values.get("vde_net_mj_per_km") if basis_label == "VDE_NET" else energy_values.get("vde_total_mj_per_km")

    if pd.notna(latest_row.get("fuel_l_per_100km")):
        result_label = "Fuel"
        result_value = _format_fuel_value(latest_row["fuel_l_per_100km"], unavailable="-")
    elif pd.notna(latest_row.get("energy_Wh_per_km")):
        result_label = "Energy"
        result_value = _format_energy_value(latest_row["energy_Wh_per_km"], unavailable="-")
    else:
        result_label = "Final Result"
        result_value = "-"

    story1, story2, story3, story4 = st.columns(4)
    story1.metric(
        f"Vehicle Demand ({basis_label}) [{unit_label('energy_per_distance', _current_unit_system())}]",
        _format_demand_value(basis_value, unavailable="-").replace(f" {unit_label('energy_per_distance', _current_unit_system())}", ""),
    )
    story2.metric("PSE", _format_metric_value(pse_summary.get("value"), format_str="%.3f"))
    story3.metric(result_label, result_value)
    story4.metric("Confidence", str(confidence_summary.get("label") or provenance.get("confidence") or "-"))
    _render_bench_badges(list(confidence_summary.get("status_items") or []))

    method_text = str(latest_row.get("engine_method") or latest_row.get("method_note") or "-")
    pse_source = str(pse_summary.get("source_label") or "Unavailable")
    st.caption(
        f"Latest saved scenario: #{int(latest_row['id'])} | Method: {method_text} | "
        f"PSE source: {pse_source} | Basis: {basis_label}"
    )
    if pse_summary.get("source") == "ml_fuel_prediction":
        st.caption("Current ML artifact predicts final fuel/energy outputs. PSE in this report is derived from that result.")


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
                "PSE": _scenario_scorecard_field_value(row, "pse_value", current_vde_row, vde_row_lookup),
                "PSE Source": _scenario_scorecard_field_value(row, "pse_source_label", current_vde_row, vde_row_lookup),
                "Confidence": _scenario_scorecard_field_value(row, "confidence_label", current_vde_row, vde_row_lookup),
                "Statuses": _scenario_scorecard_field_value(row, "confidence_statuses", current_vde_row, vde_row_lookup),
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
            ("PSE", "-", "pse_value"),
            ("PSE mode", "-", "pse_mode"),
            ("PSE source", "-", "pse_source_label"),
            ("PSE target type", "-", "pse_target_type"),
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
        "Powertrain Efficiency / Confidence",
        [
            ("PSE", "-", "pse_value"),
            ("PSE mode", "-", "pse_mode"),
            ("PSE source", "-", "pse_source_label"),
            ("PSE status", "-", "pse_status"),
            ("PSE cycle basis", "-", "pse_cycle_basis"),
            ("Data origin", "-", "data_origin"),
            ("Confidence", "-", "confidence"),
            ("Confidence label", "-", "confidence_label"),
            ("Confidence statuses", "-", "confidence_statuses"),
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
                "PSE": _scenario_scorecard_field_value(row, "pse_value", current_vde_row, vde_row_lookup),
                "Confidence": _scenario_scorecard_field_value(row, "confidence_label", current_vde_row, vde_row_lookup),
                "Statuses": _scenario_scorecard_field_value(row, "confidence_statuses", current_vde_row, vde_row_lookup),
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
            _render_bench_badges(list(_confidence_summary_from_saved_row(row).get("status_items") or []))

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
        ("PSE", lambda row, _, __: _pse_summary_from_saved_row(row).get("value")),
        ("PSE mode", lambda row, _, __: _pse_summary_from_saved_row(row).get("mode") or "-"),
        ("PSE source", lambda row, _, __: _pse_summary_from_saved_row(row).get("source_label") or "-"),
        ("PSE status", lambda row, _, __: _pse_summary_from_saved_row(row).get("status") or "-"),
        ("Confidence", lambda row, _, __: _confidence_summary_from_saved_row(row).get("label") or "-"),
        ("Confidence statuses", lambda row, _, __: ", ".join(_confidence_summary_from_saved_row(row).get("status_items") or []) or "-"),
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
        pse_summary = _pse_summary_from_saved_row(row)
        revision_state = _resolve_scenario_revision_state(row, vde_row_lookup, current_vde_row)
        scenario_vde_row = _resolve_scenario_vde_row(row, vde_row_lookup, current_vde_row)
        with st.expander(f"#{int(row['id'])} provenance", expanded=False):
            _render_link_state_badge(revision_state["status"])
            st.caption(revision_state["message"])
            _render_bench_badges(list(_confidence_summary_from_saved_row(row).get("status_items") or []))
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Source VDE", f"#{int(row['vde_id'])}" if pd.notna(row.get("vde_id")) else "-")
            p2.metric("Basis", str(row.get("energy_basis") or "-"))
            p3.metric("PSE", _format_metric_value(pse_summary.get("value"), format_str="%.3f"))
            p4.metric("Confidence", str(_confidence_summary_from_saved_row(row).get("label") or provenance.get("confidence") or "-"))
            if scenario_vde_row is not None:
                live_vehicle = f"{scenario_vde_row.get('make', '-')} {scenario_vde_row.get('model', '-')}".strip()
                st.caption(
                    f"Live row: {live_vehicle} | rev {resolve_vde_source_revision(scenario_vde_row) or '-'}"
                )
            st.caption(str(pse_summary.get("warning") or "PSE is cycle-effective and should not be interpreted as pure engine efficiency."))
            st.write(
                {
                    "provenance": provenance,
                    "assumptions": assumptions,
                }
            )


def _render_common_save_mode(vde_id: int, current_vde_row: dict) -> tuple[str, int | None]:
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

    options, label_to_id, row_lookup = _saved_update_target_options(existing_rows, current_vde_row)

    selected_label = st.selectbox(
        "Target saved scenario",
        options,
        key="pwt_common_update_target",
    )
    selected_row_id = label_to_id[selected_label]
    _render_update_target_status(row_lookup[selected_row_id], current_vde_row)
    return "update_existing", selected_row_id


def _build_observed_reference_request(
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    reference_summary: dict[str, Any],
) -> FuelEstimateRequest | None:
    observed_pse = to_float(reference_summary.get("observed_pse"))
    if observed_pse is None:
        return None
    electrification = str(ctx.get("electrification") or "ICE").upper()
    energy_basis = str(ctx.get("energy_basis") or "VDE_TOTAL").upper()
    powertrain_features = _build_powertrain_features_from_state(vde_row, ctx)
    metadata = dict(reference_summary.get("metadata") or {})
    if electrification == "BEV":
        powertrain_features["bev_eff_drive"] = observed_pse
    else:
        powertrain_features["eta_pt_est"] = observed_pse
    if metadata.get("fuel_type") not in (None, "") and powertrain_features.get("fuel_type") in (None, "", "(leave missing)"):
        powertrain_features["fuel_type"] = metadata.get("fuel_type")
    request = build_fuel_estimate_request_from_vde(
        vde_row,
        electrification=electrification,
        energy_basis=energy_basis,
        method="physics_simple",
        powertrain_features=powertrain_features,
    )
    request.vehicle_features["reference_source_type"] = reference_summary.get("source_type")
    request.vehicle_features["reference_source_id"] = reference_summary.get("source_id")
    request.vehicle_features["reference_source_label"] = reference_summary.get("source_label")
    request.vehicle_features["reference_observed_pse"] = observed_pse
    request.vehicle_features["baseline_method_label"] = "Observed / Derived PSE"
    return request


def _build_active_fuel_estimate_request(
    vde_id: int,
    vde_row: dict,
    ctx: Dict[str, Any],
    regression_vde: float | None,
    *,
    method_label: str | None = None,
    reference_summary: dict[str, Any] | None = None,
) -> FuelEstimateRequest | None:
    method = str(method_label or st.session_state.get("pwt_setup_method") or PWT_ESTIMATION_METHODS[0])
    electrification = str(ctx.get("electrification") or "ICE").upper()
    energy_basis = str(ctx.get("energy_basis") or "VDE_TOTAL").upper()
    reference_summary = reference_summary or _selected_powertrain_reference(vde_id, vde_row)

    if method == "Observed / Derived PSE":
        request = _build_observed_reference_request(vde_id, vde_row, ctx, reference_summary)
        if request is None:
            return None
        return _apply_scenario_feature_overrides(request, vde_id=vde_id, vde_row=vde_row, ctx=ctx, reference_summary=reference_summary)

    if method == "Manual / Imported":
        request = build_fuel_estimate_request_from_vde(
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
        return _apply_scenario_feature_overrides(request, vde_id=vde_id, vde_row=vde_row, ctx=ctx, reference_summary=reference_summary)

    if method == "Physics Simple":
        powertrain_features = _build_powertrain_features_from_state(vde_row, ctx)
        request = build_fuel_estimate_request_from_vde(
            vde_row,
            electrification=electrification,
            energy_basis=energy_basis,
            method="physics_simple",
            powertrain_features=powertrain_features,
        )
        if electrification == "BEV" and bool(ctx.get("draft_bev_placeholders")):
            request.vehicle_features.update(apply_bev_placeholders(vde_id))
            request.vehicle_features["draft_bev_placeholders"] = True
        return _apply_scenario_feature_overrides(request, vde_id=vde_id, vde_row=vde_row, ctx=ctx, reference_summary=reference_summary)

    if method == "Regression":
        if regression_vde is None:
            return None
        filters = dict(st.session_state.get("pwt_regression_filters") or {})
        filters.setdefault("legislation", vde_row.get("legislation"))
        request = build_fuel_estimate_request_from_vde(
            vde_row,
            electrification=electrification,
            energy_basis=energy_basis,
            method="regression_existing",
            powertrain_features=_build_powertrain_features_from_state(vde_row, ctx),
            model_options={
                "regression_runner": _build_regression_runner(vde_id, electrification, filters),
            },
        )
        return _apply_scenario_feature_overrides(request, vde_id=vde_id, vde_row=vde_row, ctx=ctx, reference_summary=reference_summary)

    if method == "ML Prediction":
        request = build_fuel_estimate_request_from_vde(
            vde_row,
            electrification=electrification,
            energy_basis=energy_basis,
            method="ml_prediction",
            powertrain_features=_build_powertrain_features_from_state(vde_row, ctx),
            model_options={
                "ml_artifact_path": st.session_state.get("pwt_ml_artifact_path"),
            },
        )
        if electrification == "BEV" and bool(ctx.get("draft_bev_placeholders")):
            request.vehicle_features.update(apply_bev_placeholders(vde_id))
            request.vehicle_features["draft_bev_placeholders"] = True
        return _apply_scenario_feature_overrides(request, vde_id=vde_id, vde_row=vde_row, ctx=ctx, reference_summary=reference_summary)

    return None


def render_fuel_review_save_panel(vde_id: int, vde_row: dict, ctx: Dict[str, Any], regression_vde: float | None) -> None:
    st.subheader("Result & Save")
    draft = _build_powertrain_scenario_draft(vde_id, vde_row)
    baseline_method = str(draft.get("baseline_estimate", {}).get("method") or st.session_state.get("pwt_setup_method") or "")
    request = _build_active_fuel_estimate_request(
        vde_id,
        vde_row,
        ctx,
        regression_vde,
        method_label=baseline_method,
        reference_summary=draft.get("powertrain_reference"),
    )
    if request is None:
        method_label = str(st.session_state.get("pwt_setup_method") or "Baseline Estimation")
        st.info(_final_result_pending_message(method_label))
        return

    result = run_fuel_estimation(request)
    staged = build_fuel_scenario_save_payload(result, extra_payload=_proposal_save_overrides(result, draft))
    save_mode, target_row_id = _render_common_save_mode(vde_id, vde_row)
    current_revision = result.request.vehicle_features.get("source_vde_revision")
    assumptions = result.assumptions or {}
    confidence_summary = dict(assumptions.get("confidence_summary") or {})
    pse_summary = dict(assumptions.get("pse_summary") or {})
    proposal_summary = dict(draft.get("proposal_result") or {})
    baseline_summary = dict(proposal_summary.get("baseline") or {})
    proposal_metrics = dict(proposal_summary.get("proposal") or {})
    show_technical = _show_technical_details()
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
    status_label = str(proposal_summary.get("status") or "Pending")
    confidence_label = str(proposal_summary.get("confidence") or confidence_summary.get("label") or result.confidence or "-")
    summary1, summary2, summary3 = st.columns(3)
    summary1.metric("Proposal", "Equals baseline" if status_label == "No quantitative delta" else status_label)
    summary2.metric("Confidence", confidence_label.replace("_", " ").title())
    summary3.metric("Save action", "Update existing" if save_mode == "update_existing" else "Create new")
    if result.warnings:
        st.warning("Preview warnings: " + ", ".join(result.warnings))

    st.markdown("#### Baseline vs Proposal")
    comparison_rows = _build_baseline_proposal_rows(baseline_summary, proposal_metrics)
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        if not (draft.get("technology_deltas") or []):
            comparison_df["Proposal"] = "Same as baseline"
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.info("Proposal preview is pending baseline metrics.")

    st.markdown("#### Applied Delta Summary")
    delta_rows = []
    for delta in draft.get("technology_deltas") or []:
        delta_rows.append(
            {
                "Delta": delta.get("name"),
                "Basis": _compact_delta_basis_label(delta.get("effect_basis")),
                "Value": delta.get("effect_value"),
                "Status": "Draft preview" if delta.get("is_preview_only") else str(delta.get("quantitative_status") or "-").replace("_", " "),
            }
        )
    if delta_rows:
        st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No delta applied.")

    if proposal_summary.get("status") == "No quantitative delta":
        st.info("Proposal equals baseline because no quantitative technology delta is applied.")

    st.markdown("#### Save Controls")
    if save_mode == "update_existing" and target_row_id is not None:
        st.caption(f"Selected saved Powertrain Scenario for update: #{target_row_id}")
    button_label = "Update Powertrain Scenario" if save_mode == "update_existing" else "Save Powertrain Scenario"
    if st.button(button_label, use_container_width=True, key="btn_save_fuel_scenario_common"):
        if not any(staged.payload.get(key) is not None for key in ("fuel_l_per_100km", "energy_Wh_per_km", "gco2_per_km")):
            st.warning("Nothing to save for the current powertrain scenario.")
        else:
            saved = save_fuel_estimate_result(
                result,
                save_mode,
                row_id=target_row_id,
                extra_payload=_proposal_save_overrides(result, draft),
            )
            if saved["action"] == "update_existing":
                st.success(f"Powertrain Scenario updated (id={saved['row_id']}).")
            else:
                st.success(f"Powertrain Scenario saved (id={saved['row_id']}).")

    feature_readiness = dict(result.request.vehicle_features.get("scenario_feature_readiness") or {})
    feature_sources = dict(result.request.vehicle_features.get("scenario_feature_sources") or {})
    feature_values = dict(result.request.vehicle_features.get("scenario_feature_values") or {})

    with st.expander("Advanced: confidence and lineage", expanded=show_technical):
        l1, l2, l3, l4, l5 = st.columns(5)
        l1.metric("VDE source", f"#{int(vde_id)}")
        l2.metric("Baseline source", str(draft.get("powertrain_reference", {}).get("source_label") or "-"))
        l3.metric("Baseline method", str(draft.get("baseline_estimate", {}).get("method") or "-"))
        l4.metric("Technology deltas", str(len(draft.get("technology_deltas") or [])))
        l5.metric("Data origin", staged.data_origin)
        confidence_badges = list(confidence_summary.get("status_items") or [])
        if proposal_summary.get("registered_only_deltas"):
            confidence_badges.append("Registered-only deltas")
        if proposal_summary.get("status") == "No quantitative delta":
            confidence_badges.append("Simulation not applied")
        _render_bench_badges(confidence_badges)
        if result.method == "ml_prediction":
            c1, c2 = st.columns(2)
            c1.metric("SHAP", str(assumptions.get("shap_status") or "Unavailable"))
            c2.metric("Peer group quality", str((assumptions.get("peer_group_quality") or {}).get("label") or "-"))
        if save_mode == "update_existing":
            st.caption("Update mode reuses the selected saved scenario row and refreshes its persisted result/provenance to the current draft.")
        else:
            st.caption("Create mode preserves existing saved scenarios and writes the current draft as a new scenario row.")

    if feature_readiness or feature_sources:
        with st.expander("Advanced: feature readiness", expanded=show_technical):
            f1, f2, f3 = st.columns(3)
            f1.metric("Status", str(feature_readiness.get("status_label") or "-"))
            f2.metric("Overrides", str(len(result.request.vehicle_features.get("scenario_feature_overrides") or {})))
            f3.metric("Confidence impacts", str(len(result.request.vehicle_features.get("scenario_feature_confidence_impacts") or [])))
            st.caption(str(feature_readiness.get("status_detail") or _scenario_override_label()))
            rows = []
            for field in FEATURE_READINESS_FIELDS:
                key = field["key"]
                rows.append(
                    {
                        "feature": field["label"],
                        "current_value": _format_feature_value(feature_values.get(key), feature_key=key),
                        "source": feature_sources.get(key, "missing"),
                        "importance": field["importance"],
                        "action": _readiness_action_for_source(feature_sources.get(key, "missing")),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Advanced: working assumptions", expanded=show_technical):
        if assumptions:
            summary_cards = []
            for key in ("source", "fuel_type", "eta_pt_est", "bev_eff_drive", "utility_factor", "dataset_rows"):
                if assumptions.get(key) not in (None, "", {}, []):
                    summary_cards.append((key, assumptions.get(key)))
            if summary_cards:
                cols = st.columns(min(4, len(summary_cards)))
                for idx, (label, value) in enumerate(summary_cards):
                    cols[idx % len(cols)].metric(label.replace("_", " ").title(), str(value))
            st.write(assumptions)
        else:
            st.caption("No extra assumptions were required beyond the selected method inputs.")

    if result.method == "regression_existing":
        with st.expander("Advanced: regression provenance", expanded=show_technical):
            effective_filters = assumptions.get("effective_filters") or {}
            model_summary = assumptions.get("model_summary") or {}
            candidate_pool_rows = assumptions.get("candidate_pool_rows")
            dataset_rows = assumptions.get("dataset_rows")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Energy basis", str(result.energy_basis_used or "-"))
            g2.metric("Candidate pool", str(candidate_pool_rows if candidate_pool_rows is not None else "-"))
            g3.metric("Filtered sample", str(dataset_rows if dataset_rows is not None else "-"))
            g4.metric("Sample quality", str(assumptions.get("sample_quality") or "-"))
            h1, h2 = st.columns(2)
            h1.metric("Method", "Regression")
            h2.metric("VDE source revision", str(current_revision or "-"))
            if result.warnings:
                st.caption("Regression warnings: " + ", ".join(result.warnings))
            st.write({"effective_filters": effective_filters, "model_summary": model_summary})

    if result.method in ("manual_imported", "physics_simple", "regression_existing"):
        with st.expander("Advanced: comparative guidance", expanded=show_technical):
            _render_comparative_guidance(comparative_analysis)

    if result.method == "ml_prediction":
        with st.expander("Advanced: ML provenance", expanded=show_technical):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Artifact status", str(assumptions.get("integration_status") or "-"))
            m2.metric("Model", str(assumptions.get("model_name") or "Export pending"))
            m3.metric("SHAP", str(assumptions.get("shap_status") or "Pending"))
            m4.metric("Targets", ", ".join(assumptions.get("expected_targets") or []) or "-")
            st.caption("These ML signals are model/integration metadata, not proof of physical causality.")
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
            if pse_summary.get("source") == "ml_fuel_prediction":
                st.caption("PSE is derived from the ML fuel prediction explained above. It is not a direct ML PSE prediction in the current artifact.")

    with st.expander("Advanced: phase outputs", expanded=show_technical):
        _render_phase_outputs_table(result.phase_outputs)

    with st.expander("Advanced: provenance payload", expanded=show_technical):
        st.write(
            {
                "request": result.request.to_dict(),
                "assumptions": result.assumptions,
                "confidence": result.confidence,
                "phase_outputs": result.phase_outputs,
                "warnings": result.warnings,
            }
        )

    with st.expander("Advanced: staged save payload", expanded=show_technical):
        st.caption("This is the payload that the common Powertrain Scenario save flow will persist.")
        st.write(staged.payload)


def render_comparison_report_page(vde_id: int, vde_row: dict) -> None:
    _render_comparison_report_overview(vde_id, vde_row)
    st.markdown("---")

    report_tab, analysis_tab, benchmark_tab, saved_tab = st.tabs(
        ["Scenario Compare", "Method Analysis", "Peers & Outlook", "Saved Estimates"]
    )

    with report_tab:
        render_scorecard_panel(vde_id, vde_row)

    with analysis_tab:
        analysis_ctx = get_build_scenario_context(vde_id, vde_row)
        render_analysis_lab_panel(vde_id, vde_row, analysis_ctx, analysis_ctx.get("energy_value_mj_per_km"))

    with benchmark_tab:
        render_benchmark_regulatory_panel(vde_id, vde_row, get_build_scenario_context(vde_id, vde_row))

    with saved_tab:
        render_saved_scenarios_panel(vde_id)


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


def _render_method_analysis_summary(result: Any, ctx: Dict[str, Any]) -> None:
    assumptions = dict((result.assumptions or {}) if result else {})
    confidence_summary = dict(assumptions.get("confidence_summary") or {})
    pse_summary = dict(assumptions.get("pse_summary") or {})

    st.markdown("#### Current Method Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Method", _pwt_method_label(result.method if result else None))
    s2.metric("Energy basis", str(result.energy_basis_used or "-") if result else "-")
    s3.metric("PSE", _format_metric_value(pse_summary.get("value"), format_str="%.3f"))
    s4.metric("Confidence", str(confidence_summary.get("label") or result.confidence or "-") if result else "-")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(f"Fuel [{_fuel_display_unit()}]", _format_fuel_value(result.fuel_l_100km, unavailable="Pending").replace(f" {_fuel_display_unit()}", "") if result else "Pending")
    r2.metric(f"Energy [{unit_label('energy_wh_per_distance', _current_unit_system())}]", _format_energy_value(result.energy_Wh_km, unavailable="Pending").replace(f" {unit_label('energy_wh_per_distance', _current_unit_system())}", "") if result else "Pending")
    r3.metric(f"CO2 [{unit_label('co2_per_distance', _current_unit_system())}]", _format_co2_value(result.gco2_km, unavailable="Pending").replace(f" {unit_label('co2_per_distance', _current_unit_system())}", "") if result else "Pending")
    r4.metric("Warnings", str(len(result.warnings or [])) if result else "0")
    _render_bench_badges(list(confidence_summary.get("status_items") or []))
    st.caption(
        f"Draft context: {str(ctx.get('scenario_name') or '-')} | "
        f"{str(ctx.get('electrification') or '-')} | "
        f"{str(ctx.get('energy_basis') or '-')}"
    )
    if result and result.warnings:
        st.warning("Method warnings: " + ", ".join(result.warnings))


def _render_method_analysis_diagnostics(result: Any, *, title: str = "Current draft diagnostics") -> None:
    st.markdown("#### Technical Diagnostics")
    with st.expander(title, expanded=True):
        st.write(
            {
                "method": result.method,
                "assumptions": result.assumptions,
                "confidence": result.confidence,
                "warnings": result.warnings,
                "phase_outputs": result.phase_outputs,
            }
        )


def _render_ml_method_analysis(request: FuelEstimateRequest, result: Any) -> None:
    assumptions = dict((result.assumptions or {}) if result else {})
    setup = describe_ml_prediction_setup(
        request,
        model_artifact_path=st.session_state.get("pwt_ml_artifact_path"),
        predictor=request.model_options.get("ml_predictor"),
    )
    peer_quality = dict(assumptions.get("peer_group_quality") or {})

    st.markdown("#### ML Diagnostics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Artifact status", str(assumptions.get("integration_status") or setup.get("status") or "-"))
    m2.metric("Model", str(assumptions.get("model_name") or setup.get("model_name") or "Export pending"))
    m3.metric("SHAP", str(assumptions.get("shap_status") or "Unavailable"))
    m4.metric("Peer quality", str(peer_quality.get("label") or "-"))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Coverage", str(assumptions.get("coverage_status") or "unknown"))
    m6.metric("Expected targets", ", ".join(assumptions.get("expected_targets") or setup.get("targets") or []) or "-")
    m7.metric("Features used", str(len(assumptions.get("features_used") or setup.get("features", {}).get("available_feature_names") or [])))
    m8.metric("Missing features", str(len(assumptions.get("missing_features") or setup.get("features", {}).get("missing_features") or [])))
    st.caption("ML analysis reads the active draft through the same estimation contract used by review/save. SHAP and peers remain advisory.")

    explanation = assumptions.get("ml_explanation") or {}
    if explanation.get("status") == "available":
        explanation_df = pd.DataFrame(explanation.get("grouped_blocks") or [])
        if not explanation_df.empty:
            st.markdown("#### SHAP Grouped View")
            show_cols = [
                col
                for col in ["engineering_block", "contribution", "main_features", "interpretation"]
                if col in explanation_df.columns
            ]
            st.dataframe(explanation_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.caption(str(explanation.get("message") or "SHAP not available for this model in the current integration."))

    with st.expander("ML feature coverage", expanded=False):
        st.write(
            {
                "artifact_path": setup.get("artifact_path"),
                "artifact_candidates": setup.get("artifact_candidates"),
                "coverage_status": assumptions.get("coverage_status") or "unknown",
                "coverage_details": assumptions.get("coverage_details"),
                "features_used": assumptions.get("features_used") or setup.get("features", {}).get("available_feature_names"),
                "missing_features": assumptions.get("missing_features") or setup.get("features", {}).get("missing_features"),
                "peer_group_quality": peer_quality,
                "warnings": result.warnings if result else [],
            }
        )
    if dict(assumptions.get("pse_summary") or {}).get("source") == "ml_fuel_prediction":
        st.caption("PSE is derived from the ML fuel prediction shown above. It is not a direct ML PSE target in the current artifact.")


def render_analysis_lab_panel(vde_id: int, vde_row: dict, ctx: Dict[str, Any], regression_vde: float | None) -> None:
    st.subheader("Method Analysis")
    active_method = _resolve_active_pwt_setup_method(vde_id, vde_row, ctx)
    request = _build_active_fuel_estimate_request(vde_id, vde_row, ctx, regression_vde)
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
        result = run_fuel_estimation(request) if request is not None else None

        if result is not None:
            _render_method_analysis_summary(result, ctx)

        st.caption("Method Analysis consumes the active Regression filters defined in Powertrain Scenario.")
        with st.expander("Effective Regression Filters", expanded=False):
            st.write(reg_filters or {"info": "No regression filters staged yet. Configure Regression in Powertrain Scenario first."})

        _render_regression_dataset_feedback(warnings, len(regdf))
        if "regression_dataset_empty" in warnings or "regression_dataset_insufficient" in warnings:
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

    if request is not None and active_method in ("Manual / Imported", "Physics Simple", "ML Prediction"):
        result = run_fuel_estimation(request)
        _render_method_analysis_summary(result, ctx)
        if active_method == "ML Prediction":
            _render_ml_method_analysis(request, result)
            _render_method_analysis_diagnostics(result, title="ML draft diagnostics")
            st.caption("Method Analysis explains the active ML draft without changing any staged Powertrain Scenario inputs.")
            return

        _render_method_analysis_diagnostics(result)
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
    else:
        st.caption("Use this area to review saved scenarios before editing or updating them. Link state shows whether the saved scenario still matches the live VDE source.")
    _render_saved_compare_panel(saved_df, current_vde_row, vde_row_lookup, scope)
    st.markdown("---")
    render_fuelcons_table(
        saved_df,
        editable=True,
        current_vde_row=current_vde_row if scope == "Current VDE only" else None,
        vde_row_lookup=vde_row_lookup,
    )

