import html
import json
import re
import tempfile
from copy import deepcopy
from datetime import datetime

import pandas as pd
import streamlit as st
import numpy as np
from urllib.parse import quote_plus
from pathlib import Path

from src.vde_core.db import ensure_db
from src.vde_core.cycles import default_cycle_for_legislation, load_cycle_csv, use_standard_cycle
from src.vde_core.services import estimate_aux_from_coastdown, inertia_class_from_mass, load_vde_defaults
from src.vde_core.tire_roadload_service import (
    create_tire_from_form,
    get_available_tires,
    get_tire_by_code,
    get_tire_by_id,
    preview_tire_roadload_from_row,
    save_tire_roadload_to_vde,
    summarize_tire_rr,
)
from src.vde_core.vde_setup_service import (
    build_decomp_update_for_edit,
    build_edit_core_update,
    build_test_mass_hint,
    build_vde_phase_update,
    build_live_vde_preview,
    collect_ctx_updates,
    compute_vde_preview_from_inputs,
    delete_vde_snapshot,
    fetch_linked_fuelcons_count,
    fetch_vde_edit_rows,
    fetch_vde_rows_full,
    ensure_baseline_aliases,
    baseline_filter_options,
    apply_baseline_filters,
    build_baseline_state_payload,
    build_delta_mode_ctx_updates,
    db_list_makes,
    merge_update_payloads,
    resolve_tire_calculation_mass,
    resolve_tire_load_mass_basis,
    resolve_test_mass_kg,
    resolve_test_mass_state,
    to_float,
    validate_core,
    update_vde_snapshot,
)
from src.vde_core.vde_request_adapter import (
    build_v21_request_import_summary,
    build_v21_workbook_state_from_request_draft,
)
from src.vde_core.vde_request_contract import is_blank, resolve_effective_baseline
from src.vde_core.vde_request_parser import (
    parse_vde_request_workbook,
    validate_vde_request_workbook,
)
from src.vde_core.vde_request_preview import (
    build_component_action_rows,
    build_proposal_preview_model,
    build_request_audit_rows,
    build_request_comparison_rows,
    build_request_resolution_fingerprint,
    build_validation_summary,
)
from src.vde_core.vde_request_report import (
    build_request_equivalent_draft_from_state,
    build_vde_request_report_filename,
    build_vde_request_report_model,
    generate_vde_request_report_xlsx,
)
from src.vde_core.vde_request_resolver import resolve_vde_request
from src.vde_core.vde_request_save import (
    SAVE_MODE_SELECTED,
    SAVE_MODES,
    build_vde_request_save_plan,
    build_vde_request_save_plan_rows,
    build_vde_request_save_result_rows,
    execute_vde_request_save_plan,
)
from src.vde_core.vde_request_template import (
    build_canonical_baseline_payload,
    build_prefilled_ppe_template,
    build_prefilled_ppe_template_filename,
    compare_printed_snapshot,
    extract_referenced_baseline_id,
    resolve_imported_baseline_status,
    sanitize_request_filename_token,
)
from src.vde_core.vde_workbook_v21 import (
    build_v21_save_plan,
    resolve_v21_workbook as resolve_v21_workbook_model,
    resolve_v21_delta_scalar,
    resolve_v21_delta_triplet,
    resolve_v21_reference_triplet,
    resolve_v21_reference_value,
    rollup_v21_statuses as rollup_v21_statuses_model,
    validate_v21_absolute_reference,
)
from src.vde_core.vde_workflow_service import (
    build_vde_pre_save_review,
    build_vde_workflow_payload_from_ctx,
    build_vde_setup_preview_from_ctx,
    save_vde_setup_result,
    summarize_component_build_up_from_ctx,
)
from src.vde_app.plots import (
    compute_roadload_curve,
    cycle_chart,
    roadload_curve_chart,
    roadload_curve_comparison_chart,
)
from src.vde_app.components.shared import show_vde_feedback
from src.vde_app.units import (
    format_quantity,
    normalize_unit_system,
    quantity_input,
    quantity_metric,
    to_canonical,
    to_display,
    unit_label,
)
from src.vde_app.state import VDE_SETUP_CTX_DEFAULTS
from src.vde_core.loaders import load_tire_size_reference, lookup_tire_size_reference

TIRE_KPA_PER_PSI = 6.89475729317


def _tire_label(row: dict) -> str:
    return (
        f"#{row.get('id')} | "
        f"{row.get('manufacturer', '')} {row.get('model', '')} | "
        f"{row.get('size_code', '')} | "
        f"{str(row.get('standard_family', '')).upper()} | "
        f"{row.get('tire_test_code', '')}"
    )


def _current_unit_system() -> str:
    return normalize_unit_system(st.session_state.get("unit_system"))


def _current_vde_input_mode(ctx: dict | None = None) -> str:
    data = dict(ctx or st.session_state.get("ctx", {}))
    mode = str(data.get("vde_setup_input_mode") or "Spreadsheet").strip()
    return mode if mode in {"Guided", "Spreadsheet"} else "Spreadsheet"


def render_vde_setup_input_mode_selector(*, host=st) -> str:
    ctx = st.session_state.ctx
    current = _current_vde_input_mode(ctx)
    options = ["Spreadsheet", "Guided"]
    ctx["vde_setup_input_mode"] = host.radio(
        "Input mode",
        options,
        horizontal=True,
        index=options.index(current),
        key="vde_setup_input_mode_selector",
    )
    st.session_state["vde_spreadsheet_mode"] = ctx["vde_setup_input_mode"]
    host.caption("Spreadsheet is now the default engineering editor. Guided remains available for step-by-step review.")
    if ctx["vde_setup_input_mode"] == "Guided":
        ctx["spreadsheet_vehicle_errors"] = []
        ctx["spreadsheet_roadload_errors"] = []
        ctx["spreadsheet_transmission_errors"] = []
        ctx["spreadsheet_component_errors"] = []
    return str(ctx["vde_setup_input_mode"])


def _render_roadload_plot(a_force, b_force, c_force):
    unit_system = _current_unit_system()
    curve_df = compute_roadload_curve(
        to_float(a_force, 0.0),
        to_float(b_force, 0.0),
        to_float(c_force, 0.0),
        unit_system=unit_system,
    )
    fig = roadload_curve_chart(curve_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    speed_unit = str(curve_df["speed_unit"].iloc[0])
    force_unit = str(curve_df["force_unit"].iloc[0])
    table_points = [0, 30, 60, 80, 100] if unit_system == "US customary" else [0, 50, 100, 130, 160]
    table_df = curve_df[curve_df["speed_display"].round(6).isin([float(value) for value in table_points])][
        ["speed_display", "force_display", "power_kW"]
    ].copy()
    table_df.rename(
        columns={
            "speed_display": f"Speed [{speed_unit}]",
            "force_display": f"Force [{force_unit}]",
            "power_kW": "Power [kW]",
        },
        inplace=True,
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)


def render_step_header(number: int, title: str, caption: str):
    st.markdown(f"<div class='vde-step-title'>{number}. {title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='vde-step-caption'>{caption}</div>", unsafe_allow_html=True)


def _render_preview_roadload_curves(abc_total: dict | None, abc_net: dict | None) -> None:
    curves: list[dict[str, float | str]] = []
    total = dict(abc_total or {})
    net = dict(abc_net or {})

    if any(to_float(total.get(key)) is not None for key in ("A", "B", "C")):
        curves.append(
            {
                "label": "ABC_TOTAL",
                "A_N": float(to_float(total.get("A"), 0.0) or 0.0),
                "B_N_per_kph": float(to_float(total.get("B"), 0.0) or 0.0),
                "C_N_per_kph2": float(to_float(total.get("C"), 0.0) or 0.0),
            }
        )

    if any(to_float(net.get(key)) is not None for key in ("A", "B", "C")):
        curves.append(
            {
                "label": "ABC_NET",
                "A_N": float(to_float(net.get("A"), 0.0) or 0.0),
                "B_N_per_kph": float(to_float(net.get("B"), 0.0) or 0.0),
                "C_N_per_kph2": float(to_float(net.get("C"), 0.0) or 0.0),
            }
        )

    fig = roadload_curve_comparison_chart(curves, unit_system=_current_unit_system())
    if fig is not None:
        st.caption("Current Roadload Curves")
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def _cycle_speed_moments_kph(cycle_name: str) -> tuple[float | None, float | None]:
    try:
        df = load_cycle_csv(cycle_name)
    except Exception:
        return None, None
    if df is None or df.empty or "v" not in df.columns:
        return None, None
    v_vals = pd.to_numeric(df["v"], errors="coerce").dropna()
    if v_vals.empty:
        return None, None
    v_kph = v_vals * 3.6
    return float(v_kph.mean()), float((v_kph ** 2).mean())


def _epa_weighted_speed_moments_kph() -> tuple[float | None, float | None]:
    ftp_avg, ftp_avg2 = _cycle_speed_moments_kph("FTP75")
    hwfet_avg, hwfet_avg2 = _cycle_speed_moments_kph("HWFET")
    if ftp_avg is None or ftp_avg2 is None or hwfet_avg is None or hwfet_avg2 is None:
        return None, None
    return (
        (0.55 * ftp_avg) + (0.45 * hwfet_avg),
        (0.55 * ftp_avg2) + (0.45 * hwfet_avg2),
    )


def _equivalent_rr_from_abc(
    a_force,
    b_force,
    c_force,
    *,
    load_kN,
) -> tuple[float | None, float | None, float | None]:
    epa_v_avg, epa_v2_avg = _epa_weighted_speed_moments_kph()
    if load_kN is None or float(load_kN or 0.0) <= 0.0:
        return None, epa_v_avg, epa_v2_avg
    effective_force = float(to_float(a_force, 0.0) or 0.0)
    if epa_v_avg is not None and epa_v2_avg is not None:
        effective_force = (
            float(to_float(a_force, 0.0) or 0.0)
            + (float(to_float(b_force, 0.0) or 0.0) * epa_v_avg)
            + (float(to_float(c_force, 0.0) or 0.0) * epa_v2_avg)
        )
    return effective_force / float(load_kN), epa_v_avg, epa_v2_avg


def _abc_from_final_rr_target(
    rr_n_per_kn,
    *,
    load_kN,
    crr_frac_120,
) -> tuple[float, float, float, float | None]:
    target_rr = float(to_float(rr_n_per_kn, 0.0) or 0.0)
    target_force = target_rr * float(to_float(load_kN, 0.0) or 0.0)
    crr_frac = float(to_float(crr_frac_120, 0.0) or 0.0)
    epa_v_avg, _ = _epa_weighted_speed_moments_kph()
    if epa_v_avg is not None:
        denom = 1.0 + ((crr_frac * epa_v_avg) / 120.0)
        a_rr = target_force / denom if abs(denom) > 1e-12 else target_force
    else:
        a_rr = target_force
    b_rr = (crr_frac * a_rr) / 120.0 if a_rr else 0.0
    return a_rr, b_rr, 0.0, epa_v_avg


def _string_or_none(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _truthy_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _editor_float_or_none(value):
    numeric = to_float(value)
    if numeric is None:
        return None
    try:
        if pd.isna(numeric):
            return None
    except Exception:
        pass
    return float(numeric)


def _editor_canonical_or_none(value, quantity: str):
    numeric = _editor_float_or_none(value)
    if numeric is None:
        return None
    return float(to_canonical(numeric, quantity, _current_unit_system()))


def _build_roadload_spreadsheet_df(ctx: dict) -> pd.DataFrame:
    preview = _safe_workflow_preview(ctx)
    derived_net = dict(preview.get("abc_net") or {})
    manual_net_apply = bool(ctx.get("spreadsheet_abc_net_apply"))

    manual_net = {
        "A": to_float(ctx.get("spreadsheet_abc_net_A")),
        "B": to_float(ctx.get("spreadsheet_abc_net_B")),
        "C": to_float(ctx.get("spreadsheet_abc_net_C")),
    }
    if manual_net_apply and any(value is not None for value in manual_net.values()):
        net_values = manual_net
        net_basis = str(ctx.get("spreadsheet_abc_net_basis") or "manual")
        net_source = str(ctx.get("spreadsheet_abc_net_source") or "manual")
    else:
        net_values = derived_net
        net_basis = str(ctx.get("spreadsheet_abc_net_basis") or ("derived/manual" if derived_net else "derived/manual"))
        net_source = str(ctx.get("spreadsheet_abc_net_source") or ("trans loss removed" if derived_net else "trans loss removed"))

    return pd.DataFrame(
        [
            {
                "roadload_set": "ABC_TOTAL",
                "A": to_display(to_float(ctx.get("A")), "force", _current_unit_system()),
                "B": to_display(to_float(ctx.get("B")), "force_per_speed", _current_unit_system()),
                "C": to_display(to_float(ctx.get("C")), "force_per_speed_squared", _current_unit_system()),
                "basis": str(ctx.get("spreadsheet_roadload_total_basis") or "coastdown"),
                "source": str(ctx.get("spreadsheet_roadload_total_source") or "measured/manual"),
                "apply": True,
                "status": "OK" if all(to_float(ctx.get(key)) is not None for key in ("A", "B", "C")) else "Missing",
                "notes": "Applied roadload set",
            },
            {
                "roadload_set": "ABC_NET",
                "A": to_display(to_float(net_values.get("A")), "force", _current_unit_system()),
                "B": to_display(to_float(net_values.get("B")), "force_per_speed", _current_unit_system()),
                "C": to_display(to_float(net_values.get("C")), "force_per_speed_squared", _current_unit_system()),
                "basis": net_basis,
                "source": net_source,
                "apply": manual_net_apply,
                "status": "Derived" if any(value is not None for value in net_values.values()) else "Pending",
                "notes": "Manual NET if apply is checked; otherwise derived from transmission losses",
            },
        ]
    )


def _apply_roadload_spreadsheet_changes(editor_df: pd.DataFrame) -> list[str]:
    ctx = st.session_state.ctx
    errors: list[str] = []
    if editor_df is None or editor_df.empty:
        errors.append("Roadload spreadsheet is empty.")
        ctx["spreadsheet_roadload_errors"] = errors
        return errors

    rows = {
        str(row.get("roadload_set") or "").strip(): row
        for row in editor_df.to_dict(orient="records")
    }
    total_row = rows.get("ABC_TOTAL")
    net_row = rows.get("ABC_NET")
    if not total_row:
        errors.append("ABC_TOTAL row is required.")
        ctx["spreadsheet_roadload_errors"] = errors
        return errors

    total_apply = bool(total_row.get("apply"))
    if not total_apply:
        errors.append("ABC_TOTAL must stay applied.")

    total_a = _editor_canonical_or_none(total_row.get("A"), "force")
    total_b = _editor_canonical_or_none(total_row.get("B"), "force_per_speed")
    total_c = _editor_canonical_or_none(total_row.get("C"), "force_per_speed_squared")
    if total_a is None or total_a < 0.0:
        errors.append("ABC_TOTAL A must be a valid non-negative number.")
    if total_b is None:
        errors.append("ABC_TOTAL B must be a valid number.")
    if total_c is None or total_c < 0.0:
        errors.append("ABC_TOTAL C must be a valid non-negative number.")

    if not errors:
        ctx["A"] = total_a
        ctx["B"] = total_b
        ctx["C"] = total_c
        st.session_state["abc"] = {"A": float(total_a), "B": float(total_b), "C": float(total_c)}

    ctx["spreadsheet_roadload_total_basis"] = str(total_row.get("basis") or "coastdown").strip() or "coastdown"
    ctx["spreadsheet_roadload_total_source"] = str(total_row.get("source") or "measured/manual").strip() or "measured/manual"

    manual_net_apply = bool((net_row or {}).get("apply"))
    ctx["spreadsheet_abc_net_apply"] = manual_net_apply
    ctx["spreadsheet_abc_net_basis"] = str((net_row or {}).get("basis") or "derived/manual").strip() or "derived/manual"
    ctx["spreadsheet_abc_net_source"] = str((net_row or {}).get("source") or "trans loss removed").strip() or "trans loss removed"

    if manual_net_apply:
        net_a = _editor_canonical_or_none((net_row or {}).get("A"), "force")
        net_b = _editor_canonical_or_none((net_row or {}).get("B"), "force_per_speed")
        net_c = _editor_canonical_or_none((net_row or {}).get("C"), "force_per_speed_squared")
        ctx["spreadsheet_abc_net_A"] = net_a
        ctx["spreadsheet_abc_net_B"] = net_b
        ctx["spreadsheet_abc_net_C"] = net_c
        if net_a is None or net_a < 0.0:
            errors.append("ABC_NET A must be a valid non-negative number when apply is enabled.")
        if net_b is None:
            errors.append("ABC_NET B must be a valid number when apply is enabled.")
        if net_c is None or net_c < 0.0:
            errors.append("ABC_NET C must be a valid non-negative number when apply is enabled.")
        if not errors:
            trans_a = float(total_a - net_a)
            trans_b = float(total_b - net_b)
            trans_c = float(total_c - net_c)
            if min(trans_a, trans_b, trans_c) < -1e-9:
                errors.append("ABC_NET cannot exceed ABC_TOTAL. Manual ABC_NET would imply negative transmission losses.")
            else:
                ctx["transmission_losses_source"] = "Manual"
                ctx["trans_A_coef_N"] = trans_a
                ctx["trans_B_coef_Npkph"] = trans_b
                ctx["trans_C_coef_Npkph2"] = trans_c
                ctx["trans_B_Npkph"] = trans_b
    else:
        preview = _safe_workflow_preview(ctx)
        derived_net = dict(preview.get("abc_net") or {})
        ctx["spreadsheet_abc_net_A"] = to_float(derived_net.get("A"))
        ctx["spreadsheet_abc_net_B"] = to_float(derived_net.get("B"))
        ctx["spreadsheet_abc_net_C"] = to_float(derived_net.get("C"))

    ctx["spreadsheet_roadload_errors"] = errors
    return errors


def _build_transmission_spreadsheet_df(ctx: dict, *, prefill: dict | None = None) -> pd.DataFrame:
    base = dict(prefill or {})
    current_source = str(ctx.get("transmission_losses_source") or ("Baseline" if base.get("trans_A_coef_N") is not None else "Missing")).strip().title()
    if current_source == "Baseline":
        a_val = to_float(base.get("trans_A_coef_N"))
        b_val = to_float(base.get("trans_B_coef_Npkph", base.get("trans_B_Npkph")))
        c_val = to_float(base.get("trans_C_coef_Npkph2"))
        source_label = "baseline"
        apply_flag = True
    elif current_source == "Manual":
        a_val = to_float(ctx.get("trans_A_coef_N"))
        b_val = to_float(ctx.get("trans_B_coef_Npkph"))
        c_val = to_float(ctx.get("trans_C_coef_Npkph2"))
        source_label = str(ctx.get("spreadsheet_transmission_source_label") or "test/manual")
        apply_flag = True
    else:
        a_val = to_float(ctx.get("trans_A_coef_N"))
        b_val = to_float(ctx.get("trans_B_coef_Npkph"))
        c_val = to_float(ctx.get("trans_C_coef_Npkph2"))
        source_label = str(ctx.get("spreadsheet_transmission_source_label") or "test/manual")
        apply_flag = False

    return pd.DataFrame(
        [
            {
                "loss_set": "Neutral drag",
                "A_loss": to_display(a_val, "force", _current_unit_system()),
                "B_loss": to_display(b_val, "force_per_speed", _current_unit_system()),
                "C_loss": to_display(c_val, "force_per_speed_squared", _current_unit_system()),
                "source": source_label,
                "apply": apply_flag,
                "status": "OK" if apply_flag and all(value is not None for value in (a_val, b_val, c_val)) else "Review",
                "notes": "Subtracts ABC_TRANS from ABC_TOTAL to resolve ABC_NET",
            }
        ]
    )


def _apply_transmission_spreadsheet_changes(editor_df: pd.DataFrame) -> list[str]:
    ctx = st.session_state.ctx
    errors: list[str] = []
    if editor_df is None or editor_df.empty:
        errors.append("Transmission spreadsheet is empty.")
        ctx["spreadsheet_transmission_errors"] = errors
        return errors

    row = dict(editor_df.to_dict(orient="records")[0] or {})
    apply_flag = bool(row.get("apply"))
    ctx["spreadsheet_transmission_source_label"] = str(row.get("source") or "test/manual").strip() or "test/manual"

    if not apply_flag:
        ctx["transmission_losses_source"] = "Missing"
        ctx["trans_A_coef_N"] = 0.0
        ctx["trans_B_coef_Npkph"] = 0.0
        ctx["trans_C_coef_Npkph2"] = 0.0
        ctx["trans_B_Npkph"] = 0.0
        ctx["spreadsheet_abc_net_apply"] = False
        ctx["spreadsheet_transmission_errors"] = errors
        return errors

    a_loss = _editor_canonical_or_none(row.get("A_loss"), "force")
    b_loss = _editor_canonical_or_none(row.get("B_loss"), "force_per_speed")
    c_loss = _editor_canonical_or_none(row.get("C_loss"), "force_per_speed_squared")
    if a_loss is None or a_loss < 0.0:
        errors.append("Transmission A_loss must be a valid non-negative number.")
    if b_loss is None:
        errors.append("Transmission B_loss must be a valid number.")
    if c_loss is None or c_loss < 0.0:
        errors.append("Transmission C_loss must be a valid non-negative number.")

    if not errors:
        ctx["transmission_losses_source"] = "Manual"
        ctx["trans_A_coef_N"] = float(a_loss)
        ctx["trans_B_coef_Npkph"] = float(b_loss)
        ctx["trans_C_coef_Npkph2"] = float(c_loss)
        ctx["trans_B_Npkph"] = float(b_loss)
        ctx["spreadsheet_abc_net_apply"] = False

    ctx["spreadsheet_transmission_errors"] = errors
    return errors


def _spreadsheet_validation_errors(ctx: dict | None = None) -> list[str]:
    data = dict(ctx or st.session_state.get("ctx", {}))
    return (
        list(data.get("spreadsheet_vehicle_errors") or [])
        + list(data.get("spreadsheet_roadload_errors") or [])
        + list(data.get("spreadsheet_transmission_errors") or [])
        + list(data.get("spreadsheet_component_errors") or [])
    )


def _cycle_distance_km(ctx: dict) -> float | None:
    cycle_df = ctx.get("cycle_df")
    if cycle_df is None:
        return None
    try:
        df_cycle = cycle_df.copy()
        t_vals = pd.to_numeric(df_cycle["t"], errors="coerce")
        v_vals = pd.to_numeric(df_cycle["v"], errors="coerce")
        if t_vals.isna().all() or v_vals.isna().all():
            return None
        return float(np.trapezoid(v_vals, t_vals) / 1000.0)
    except Exception:
        return None


def _source_label_for_field(ctx: dict, field_name: str) -> str:
    baseline = dict(ctx.get("selected_baseline_row") or ctx.get("baseline_dict") or {})
    if field_name in baseline and baseline.get(field_name) not in (None, "", []):
        return "baseline"
    return "scenario"


def _metadata_baseline_value(ctx: dict, field_name: str):
    baseline = dict(ctx.get("selected_baseline_row") or ctx.get("baseline_dict") or {})
    field_map = {
        "legislation": "legislation",
        "category": "category",
        "manufacturer": "make",
        "model": "model",
        "model_year": "year",
        "electrification": "electrification",
        "transmission_type": "transmission_type",
        "drive_type": "drive_type",
        "fuel_type": "fuel_type",
        "proposal": "notes",
    }
    baseline_key = field_map.get(field_name)
    if baseline_key is None:
        return None
    value = baseline.get(baseline_key)
    if field_name == "model_year":
        year_value = to_float(value)
        return int(year_value) if year_value is not None else None
    return value


def _metadata_inherit_default(ctx: dict, field_name: str) -> bool:
    if field_name in {"vehicle_label", "proposal"}:
        return False
    return _metadata_baseline_value(ctx, field_name) not in (None, "", [])


def _metadata_source_label(ctx: dict, field_name: str, value) -> str:
    baseline_value = _metadata_baseline_value(ctx, field_name)
    if baseline_value in (None, "", []):
        return "scenario"
    if field_name == "model_year":
        current_year = to_float(value)
        baseline_year = to_float(baseline_value)
        if current_year is not None and baseline_year is not None and int(current_year) == int(baseline_year):
            return "baseline"
        return "scenario"
    if str(value or "").strip().upper() == str(baseline_value or "").strip().upper():
        return "baseline"
    return "scenario"


def _suggest_scenario_notes_from_baseline(baseline: dict, baseline_id: int | None) -> str:
    baseline_notes = str(baseline.get("notes") or "").strip()
    make = str(baseline.get("make") or "").strip().upper()
    model = str(baseline.get("model") or baseline.get("desc") or "").strip()
    year = str(baseline.get("year") or "").strip()
    vehicle_label = " ".join(part for part in (make, model, year) if part).strip()
    if baseline_notes:
        return f"{baseline_notes} - scenario"
    if vehicle_label:
        return f"{vehicle_label} - scenario"
    if baseline_id is not None:
        return f"Scenario based on VDE-{int(baseline_id):03d}"
    return "New scenario"


def _metadata_category_options(legislation: str) -> list[str]:
    from src.vde_app.components.vde_request_metadata_options import metadata_category_options

    return metadata_category_options(legislation)


def _metadata_choice_options(field_name: str, *, legislation: str, current_value: str) -> list[str] | None:
    from src.vde_app.components.vde_request_metadata_options import metadata_choice_options

    return metadata_choice_options(field_name, legislation=legislation, current_value=current_value)


def _build_vehicle_scenario_spreadsheet_df(ctx: dict) -> pd.DataFrame:
    _ensure_vehicle_metadata_defaults(ctx)
    rows = [
        {"field": "vehicle_label", "value": f"{ctx.get('make', '')} {ctx.get('model', '')}".strip() or "-", "source": "review", "inherit": False, "notes": "Review only"},
        {"field": "legislation", "value": str(ctx.get("legislation") or ""), "source": _metadata_source_label(ctx, "legislation", ctx.get("legislation")), "inherit": _metadata_inherit_default(ctx, "legislation"), "notes": "EPA / WLTP / ABNT"},
        {"field": "category", "value": str(ctx.get("category") or ""), "source": _metadata_source_label(ctx, "category", ctx.get("category")), "inherit": _metadata_inherit_default(ctx, "category"), "notes": "Vehicle class/category"},
        {"field": "manufacturer", "value": str(ctx.get("make") or ""), "source": _metadata_source_label(ctx, "manufacturer", ctx.get("make")), "inherit": _metadata_inherit_default(ctx, "manufacturer"), "notes": "Saved as make"},
        {"field": "model", "value": str(ctx.get("model") or ""), "source": _metadata_source_label(ctx, "model", ctx.get("model")), "inherit": _metadata_inherit_default(ctx, "model"), "notes": "Model / description"},
        {"field": "model_year", "value": int(to_float(ctx.get("year"), 2024) or 2024), "source": _metadata_source_label(ctx, "model_year", ctx.get("year")), "inherit": _metadata_inherit_default(ctx, "model_year"), "notes": "Saved as year"},
        {"field": "electrification", "value": str(ctx.get("electrification") or "ICE"), "source": _metadata_source_label(ctx, "electrification", ctx.get("electrification")), "inherit": _metadata_inherit_default(ctx, "electrification"), "notes": "ICE / HEV / PHEV / BEV"},
        {"field": "transmission_type", "value": str(ctx.get("transmission_type") or "AT"), "source": _metadata_source_label(ctx, "transmission_type", ctx.get("transmission_type")), "inherit": _metadata_inherit_default(ctx, "transmission_type"), "notes": "AT / AMT / CVT / MT / OT"},
        {"field": "drive_type", "value": str(ctx.get("drive_type") or ""), "source": _metadata_source_label(ctx, "drive_type", ctx.get("drive_type")), "inherit": _metadata_inherit_default(ctx, "drive_type"), "notes": "Optional review/edit"},
        {"field": "fuel_type", "value": str(ctx.get("fuel_type") or ""), "source": _metadata_source_label(ctx, "fuel_type", ctx.get("fuel_type")), "inherit": _metadata_inherit_default(ctx, "fuel_type"), "notes": "Review only if not used in this flow"},
    ]
    return pd.DataFrame(rows)


def _apply_vehicle_scenario_spreadsheet_changes(editor_df: pd.DataFrame) -> list[str]:
    ctx = st.session_state.ctx
    errors: list[str] = []
    if editor_df is None or editor_df.empty:
        errors = ["Vehicle / Scenario spreadsheet is empty."]
        ctx["spreadsheet_vehicle_errors"] = errors
        return errors

    st.session_state["vde_vehicle_table"] = editor_df.copy()
    rows = {
        str(row.get("field") or "").strip(): row
        for row in editor_df.to_dict(orient="records")
    }

    year_row = rows.get("model_year") or {}
    inherit_year = bool(year_row.get("inherit")) and _metadata_baseline_value(ctx, "model_year") not in (None, "", [])
    year_candidate = _metadata_baseline_value(ctx, "model_year") if inherit_year else year_row.get("value")
    year_value = to_float(year_candidate)
    if year_value is None or int(year_value) < 1900 or int(year_value) > 2100:
        errors.append("model_year must be a valid year between 1900 and 2100.")
    else:
        ctx["year"] = int(year_value)

    field_map = {
        "legislation": "legislation",
        "category": "category",
        "manufacturer": "make",
        "model": "model",
        "electrification": "electrification",
        "transmission_type": "transmission_type",
        "drive_type": "drive_type",
        "fuel_type": "fuel_type",
    }
    for source_field, target_field in field_map.items():
        row = rows.get(source_field) or {}
        if source_field == "proposal" and bool(row.get("inherit")):
            errors.append("proposal cannot inherit baseline notes. Enter a scenario-specific proposal.")
        baseline_value = _metadata_baseline_value(ctx, source_field)
        inherit_baseline = source_field != "proposal" and bool(row.get("inherit")) and baseline_value not in (None, "", [])
        if source_field == "fuel_type" and "fuel_type" not in ctx and not inherit_baseline:
            continue
        value = str((baseline_value if inherit_baseline else row.get("value")) or "").strip()
        if target_field in {"category", "make", "electrification", "transmission_type", "drive_type", "fuel_type"}:
            value = value.upper()
        ctx[target_field] = value

    required_fields = {
        "legislation": str(ctx.get("legislation") or "").strip(),
        "category": str(ctx.get("category") or "").strip(),
        "make": str(ctx.get("make") or "").strip(),
        "model": str(ctx.get("model") or "").strip(),
        "electrification": str(ctx.get("electrification") or "").strip(),
        "transmission_type": str(ctx.get("transmission_type") or "").strip(),
    }
    missing = [field for field, value in required_fields.items() if not value]
    if missing:
        errors.append("Vehicle / Scenario is missing required fields: " + ", ".join(missing) + ".")
    if ctx.get("mode") == "From baseline (editable)":
        proposal = str(ctx.get("notes") or "").strip()
        baseline_notes = str(_metadata_baseline_value(ctx, "proposal") or "").strip()
        if not proposal:
            errors.append("proposal must be defined before saving a baseline-derived scenario.")
        elif baseline_notes and proposal.upper() == baseline_notes.upper():
            errors.append("proposal must differ from the inherited baseline notes before saving a new scenario.")
    ctx["spreadsheet_vehicle_errors"] = errors
    return errors


def _build_mass_spreadsheet_df(ctx: dict) -> pd.DataFrame:
    test_mass_state = resolve_test_mass_state(dict(ctx))
    rows = [
        {"parameter": "mass_kg", "value": to_display(to_float(ctx.get("mass_kg")), "mass", _current_unit_system()), "unit": unit_label("mass"), "source": _source_label_for_field(ctx, "mass_kg"), "apply": True, "notes": "Curb/base vehicle mass"},
        {"parameter": "test_mass_basis", "value": str(test_mass_state.get("test_mass_basis") or ctx.get("test_mass_basis") or ""), "unit": "-", "source": "resolved", "apply": True, "notes": "WLTP_TMH / WLTP_TML / CURB / EPA_INERTIA_CLASS / PHYSICAL_TEST_MASS / CUSTOM"},
        {"parameter": "test_mass_kg", "value": to_display(to_float(test_mass_state.get("test_mass_kg")), "mass", _current_unit_system()), "unit": unit_label("mass"), "source": "resolved", "apply": True, "notes": "Preferred VDE mass when available"},
        {"parameter": "payload_kg", "value": to_display(to_float(ctx.get("payload_kg")), "mass", _current_unit_system()), "unit": unit_label("mass"), "source": _source_label_for_field(ctx, "payload_kg"), "apply": True, "notes": "Payload / added load"},
        {"parameter": "options_kg", "value": to_display(to_float(ctx.get("options_kg")), "mass", _current_unit_system()), "unit": unit_label("mass"), "source": _source_label_for_field(ctx, "options_kg"), "apply": True, "notes": "Optional equipment mass"},
        {"parameter": "weight_dist_fr_pct", "value": float(to_float(ctx.get("weight_dist_fr_pct"), 50.0) or 50.0), "unit": "%", "source": _source_label_for_field(ctx, "weight_dist_fr_pct"), "apply": True, "notes": "Front weight distribution"},
        {"parameter": "wltp_category", "value": str(ctx.get("wltp_category") or "M1"), "unit": "-", "source": _source_label_for_field(ctx, "wltp_category"), "apply": True, "notes": "WLTP category"},
        {"parameter": "tire_load_mass_basis", "value": str(ctx.get("tire_load_mass_basis") or "TEST_MASS"), "unit": "-", "source": "roadload", "apply": True, "notes": "TEST_MASS / TWC"},
    ]
    return pd.DataFrame(rows)


def _mass_regulatory_mode(ctx: dict) -> str:
    legislation = str(ctx.get("legislation") or "").strip().upper()
    if legislation == "EPA":
        return "EPA"
    if legislation == "WLTP":
        return "WLTP"
    return "CUSTOM"


def _build_mass_audit_df(ctx: dict) -> pd.DataFrame:
    regulatory_mode = _mass_regulatory_mode(ctx)
    test_mass_state = resolve_test_mass_state(dict(ctx))
    resolved_mass = resolve_tire_calculation_mass(dict(ctx)).get("mass_kg")
    rows = [
        {
            "parameter": "mass_kg",
            "value": to_display(to_float(ctx.get("mass_kg")), "mass", _current_unit_system()),
            "unit": unit_label("mass"),
            "source": _source_label_for_field(ctx, "mass_kg"),
            "role": "baseline_reference",
            "used": True,
            "notes": "Curb/base vehicle mass",
        },
        {
            "parameter": "payload_kg",
            "value": to_display(to_float(ctx.get("payload_kg")), "mass", _current_unit_system()),
            "unit": unit_label("mass"),
            "source": _source_label_for_field(ctx, "payload_kg"),
            "role": "profile_input",
            "used": True,
            "notes": "Payload / added load",
        },
        {
            "parameter": "options_kg",
            "value": to_display(to_float(ctx.get("options_kg")), "mass", _current_unit_system()),
            "unit": unit_label("mass"),
            "source": _source_label_for_field(ctx, "options_kg"),
            "role": "profile_input",
            "used": True,
            "notes": "Optional equipment mass",
        },
        {
            "parameter": "weight_dist_fr_pct",
            "value": float(to_float(ctx.get("weight_dist_fr_pct"), 50.0) or 50.0),
            "unit": "%",
            "source": _source_label_for_field(ctx, "weight_dist_fr_pct"),
            "role": "profile_input",
            "used": True,
            "notes": "Front weight distribution",
        },
        {
            "parameter": "test_mass_basis",
            "value": str(test_mass_state.get("test_mass_basis") or ctx.get("test_mass_basis") or ""),
            "unit": "-",
            "source": "resolved",
            "role": "scenario_calculation",
            "used": True,
            "notes": "Resolved test mass basis",
        },
        {
            "parameter": "test_mass_kg",
            "value": to_display(to_float(test_mass_state.get("test_mass_kg")), "mass", _current_unit_system()),
            "unit": unit_label("mass"),
            "source": "resolved",
            "role": "scenario_calculation",
            "used": True,
            "notes": "Resolved test mass",
        },
        {
            "parameter": "tire_load_mass_basis",
            "value": str(ctx.get("tire_load_mass_basis") or "TEST_MASS"),
            "unit": "-",
            "source": "roadload",
            "role": "fuelcons_calculation",
            "used": True,
            "notes": "Roadload / VDE calculation mass basis",
        },
        {
            "parameter": "resolved_calc_mass_kg",
            "value": to_display(to_float(resolved_mass), "mass", _current_unit_system()),
            "unit": unit_label("mass"),
            "source": "resolved",
            "role": "scenario_calculation",
            "used": True,
            "notes": "Resolved mass used by roadload/VDE",
        },
        {
            "parameter": "wltp_category",
            "value": str(ctx.get("wltp_category") or ""),
            "unit": "-",
            "source": _source_label_for_field(ctx, "wltp_category"),
            "role": "traceability" if regulatory_mode == "WLTP" else "not_applicable",
            "used": regulatory_mode == "WLTP",
            "notes": "WLTP category",
        },
        {
            "parameter": "test_mass_low_kg",
            "value": to_display(to_float(test_mass_state.get("test_mass_low_kg")), "mass", _current_unit_system()),
            "unit": unit_label("mass"),
            "source": "resolved",
            "role": "traceability" if regulatory_mode == "WLTP" else "not_applicable",
            "used": regulatory_mode == "WLTP",
            "notes": "WLTP Test Mass Low",
        },
        {
            "parameter": "test_mass_high_kg",
            "value": to_display(to_float(test_mass_state.get("test_mass_high_kg")), "mass", _current_unit_system()),
            "unit": unit_label("mass"),
            "source": "resolved",
            "role": "traceability" if regulatory_mode == "WLTP" else "not_applicable",
            "used": regulatory_mode == "WLTP",
            "notes": "WLTP Test Mass High",
        },
        {
            "parameter": "inertia_class",
            "value": to_display(to_float(ctx.get("inertia_class")), "mass", _current_unit_system()),
            "unit": unit_label("mass"),
            "source": _source_label_for_field(ctx, "inertia_class"),
            "role": "diagnostic" if regulatory_mode == "EPA" else "not_applicable",
            "used": regulatory_mode == "EPA",
            "notes": "EPA inertia / TWC mass when applicable",
        },
    ]
    return pd.DataFrame(rows)


def _mass_snapshot(ctx: dict, *, prefill=None) -> dict:
    base = dict(prefill or {})
    legislation = str(ctx.get("legislation") or base.get("legislation") or "").strip().upper()
    regulatory_mode = _mass_regulatory_mode({"legislation": legislation})
    baseline_curb_mass = to_float(base.get("mass_kg"))
    proposal_curb_mass = to_float(ctx.get("mass_kg"), baseline_curb_mass if baseline_curb_mass is not None else 1550.0)
    baseline_payload = to_float(base.get("payload_kg"), 0.0)
    proposal_payload = to_float(ctx.get("payload_kg"), baseline_payload)
    baseline_options = to_float(base.get("options_kg"), 0.0)
    proposal_options = to_float(ctx.get("options_kg"), baseline_options)
    baseline_weight_dist = to_float(base.get("weight_dist_fr_pct"), 50.0)
    proposal_weight_dist = to_float(ctx.get("weight_dist_fr_pct"), baseline_weight_dist)
    baseline_wltp_category = str(base.get("wltp_category") or "").strip().upper()
    proposal_wltp_category = str(ctx.get("wltp_category") or baseline_wltp_category).strip().upper()

    baseline_tire_basis = str(
        resolve_tire_load_mass_basis(
            {
                "legislation": legislation,
                "tire_load_mass_basis": base.get("tire_load_mass_basis"),
            }
        )
        or "TEST_MASS"
    ).upper()
    proposal_tire_basis = str(
        resolve_tire_load_mass_basis(
            {
                "legislation": legislation,
                "tire_load_mass_basis": ctx.get("tire_load_mass_basis") or base.get("tire_load_mass_basis"),
            }
        )
        or ("TWC" if legislation == "EPA" else "TEST_MASS")
    ).upper()

    baseline_test_state = resolve_test_mass_state(
        {
            **base,
            "legislation": legislation,
            "mass_kg": baseline_curb_mass,
            "payload_kg": baseline_payload,
            "options_kg": baseline_options,
            "wltp_category": baseline_wltp_category,
            "tire_load_mass_basis": baseline_tire_basis,
        }
    )
    proposal_default_state = resolve_test_mass_state(
        {
            "legislation": legislation,
            "mass_kg": proposal_curb_mass,
            "payload_kg": proposal_payload,
            "options_kg": proposal_options,
            "wltp_category": proposal_wltp_category,
            "tire_load_mass_basis": proposal_tire_basis,
        }
    )
    baseline_twc_mass = resolve_tire_calculation_mass(
        {
            **base,
            "legislation": legislation,
            "mass_kg": baseline_curb_mass,
            "payload_kg": baseline_payload,
            "options_kg": baseline_options,
            "wltp_category": baseline_wltp_category,
            "tire_load_mass_basis": "TWC",
        }
    ).get("mass_kg")
    proposal_twc_mass = resolve_tire_calculation_mass(
        {
            "legislation": legislation,
            "mass_kg": proposal_curb_mass,
            "payload_kg": proposal_payload,
            "options_kg": proposal_options,
            "wltp_category": proposal_wltp_category,
            "tire_load_mass_basis": "TWC",
        }
    ).get("mass_kg")

    return {
        "legislation": legislation,
        "regulatory_mode": regulatory_mode,
        "baseline_curb_mass": baseline_curb_mass,
        "proposal_curb_mass": proposal_curb_mass,
        "baseline_payload": baseline_payload,
        "proposal_payload": proposal_payload,
        "baseline_options": baseline_options,
        "proposal_options": proposal_options,
        "baseline_weight_dist": baseline_weight_dist,
        "proposal_weight_dist": proposal_weight_dist,
        "baseline_wltp_category": baseline_wltp_category,
        "proposal_wltp_category": proposal_wltp_category,
        "baseline_tire_basis": baseline_tire_basis,
        "proposal_tire_basis": proposal_tire_basis,
        "baseline_test_state": baseline_test_state,
        "proposal_default_state": proposal_default_state,
        "baseline_twc_mass": baseline_twc_mass,
        "proposal_twc_mass": proposal_twc_mass,
        "gvwr_mass": to_float(ctx.get("mass_profile_gvwr_kg"), to_float(base.get("mass_profile_gvwr_kg"), to_float(base.get("gvwr_kg"), to_float(base.get("gvwr"))))),
        "gcwr_mass": to_float(ctx.get("mass_profile_gcwr_kg"), to_float(base.get("mass_profile_gcwr_kg"), to_float(base.get("gcwr_kg"), to_float(base.get("gcwr"))))),
        "trailer_mass": to_float(ctx.get("mass_profile_trailer_mass_kg"), to_float(base.get("mass_profile_trailer_mass_kg"), to_float(base.get("trailer_mass_kg"), to_float(base.get("trailer_mass"))))),
        "custom_mass": to_float(ctx.get("mass_profile_custom_input_kg"), to_float(ctx.get("test_mass_kg"))),
        "custom_fuelcons_basis": str(ctx.get("mass_profile_custom_fuelcons_basis") or "TEST_MASS").strip().upper() or "TEST_MASS",
    }


def _mass_ref_status(value) -> str:
    return "OK" if value not in (None, "", []) else "Missing"


def _display_quantity_text(value, quantity: str, *, unavailable: str = "-") -> str:
    return format_quantity(
        to_float(value),
        quantity,
        _current_unit_system(),
        include_unit=False,
        unavailable=unavailable,
    )


def _build_baseline_mass_reference_df(snapshot: dict) -> pd.DataFrame:
    base_state = dict(snapshot.get("baseline_test_state") or {})
    rows = [
        {"field": "baseline_curb_mass", "value": _display_quantity_text(snapshot.get("proposal_curb_mass"), "mass"), "unit": unit_label("mass"), "source": "baseline" if snapshot.get("baseline_curb_mass") is not None else "scenario", "status": _mass_ref_status(snapshot.get("proposal_curb_mass")), "notes": "curb/base mass"},
        {"field": "epa_default_test_mass", "value": _display_quantity_text(base_state.get("test_mass_kg"), "mass"), "unit": unit_label("mass"), "source": "resolved", "status": _mass_ref_status(base_state.get("test_mass_kg")), "notes": "EPA default"},
        {"field": "twc_mass", "value": _display_quantity_text(snapshot.get("proposal_twc_mass"), "mass"), "unit": unit_label("mass"), "source": "calculated", "status": _mass_ref_status(snapshot.get("proposal_twc_mass")), "notes": "test weight for consumption"},
        {"field": "tml_mass", "value": _display_quantity_text(base_state.get("test_mass_low_kg"), "mass"), "unit": unit_label("mass"), "source": "resolved" if snapshot.get("regulatory_mode") == "WLTP" else "not set", "status": _mass_ref_status(base_state.get("test_mass_low_kg")), "notes": "WLTP traceability"},
        {"field": "tmh_mass", "value": _display_quantity_text(base_state.get("test_mass_high_kg"), "mass"), "unit": unit_label("mass"), "source": "resolved" if snapshot.get("regulatory_mode") == "WLTP" else "not set", "status": _mass_ref_status(base_state.get("test_mass_high_kg")), "notes": "WLTP traceability"},
        {"field": "gvwr", "value": _display_quantity_text(snapshot.get("gvwr_mass"), "mass"), "unit": unit_label("mass"), "source": "scenario", "status": _mass_ref_status(snapshot.get("gvwr_mass")), "notes": "gross vehicle weight rating"},
        {"field": "gcwr", "value": _display_quantity_text(snapshot.get("gcwr_mass"), "mass"), "unit": unit_label("mass"), "source": "scenario", "status": _mass_ref_status(snapshot.get("gcwr_mass")), "notes": "gross combined weight rating"},
        {"field": "trailer_mass", "value": _display_quantity_text(snapshot.get("trailer_mass"), "mass"), "unit": unit_label("mass"), "source": "scenario", "status": _mass_ref_status(snapshot.get("trailer_mass")), "notes": "trailer what-if mass"},
        {"field": "payload_mass", "value": _display_quantity_text(snapshot.get("proposal_payload"), "mass"), "unit": unit_label("mass"), "source": "scenario", "status": _mass_ref_status(snapshot.get("proposal_payload")), "notes": "payload mass"},
        {"field": "optional_equipment_mass", "value": _display_quantity_text(snapshot.get("proposal_options"), "mass"), "unit": unit_label("mass"), "source": "scenario", "status": _mass_ref_status(snapshot.get("proposal_options")), "notes": "optional equipment"},
    ]
    if snapshot.get("regulatory_mode") == "WLTP":
        rows.append(
            {"field": "wltp_category", "value": str(snapshot.get("proposal_wltp_category") or ""), "unit": "-", "source": "scenario", "status": "OK" if snapshot.get("proposal_wltp_category") else "Missing", "notes": "WLTP category"}
        )
    return pd.DataFrame(rows)


def _apply_baseline_mass_reference_changes(editor_df: pd.DataFrame) -> None:
    ctx = st.session_state.ctx
    if editor_df is None or editor_df.empty:
        return
    st.session_state["vde_mass_reference_table"] = editor_df.copy()
    rows = {str(row.get("field") or "").strip(): row for row in editor_df.to_dict(orient="records")}

    field_map = {
        "baseline_curb_mass": "mass_kg",
        "payload_mass": "payload_kg",
        "optional_equipment_mass": "options_kg",
        "gvwr": "mass_profile_gvwr_kg",
        "gcwr": "mass_profile_gcwr_kg",
        "trailer_mass": "mass_profile_trailer_mass_kg",
    }
    for field_name, ctx_key in field_map.items():
        value = _editor_canonical_or_none((rows.get(field_name) or {}).get("value"), "mass")
        if value is not None:
            ctx[ctx_key] = float(value)
    if "wltp_category" in rows:
        ctx["wltp_category"] = str((rows.get("wltp_category") or {}).get("value") or "").strip().upper()


def _mass_profile_status(required: list[str], primary_mass) -> tuple[str, str]:
    missing = [item for item in required if not item]
    if primary_mass is None:
        return "Missing", "Required inputs incomplete"
    if missing:
        return "Review", ", ".join(missing)
    return "OK", "Resolved"


def _build_mass_profiles_df(snapshot: dict, selected_profile: str) -> pd.DataFrame:
    proposal_curb_mass = snapshot.get("proposal_curb_mass")
    proposal_default_state = dict(snapshot.get("proposal_default_state") or {})
    proposal_twc_mass = snapshot.get("proposal_twc_mass")
    regulatory_mode = str(snapshot.get("regulatory_mode") or "CUSTOM")
    custom_mass = snapshot.get("custom_mass")
    custom_fuel_basis = str(snapshot.get("custom_fuelcons_basis") or "TEST_MASS").upper()
    gvwr_mass = snapshot.get("gvwr_mass")
    gcwr_mass = snapshot.get("gcwr_mass")
    trailer_mass = snapshot.get("trailer_mass")

    def _profile_row(profile: str, rule: str, vde_mass, fuelcons_basis: str, fuelcons_mass, required_inputs: str, status: str, notes: str) -> dict:
        return {
            "profile": profile,
            "enabled": profile == selected_profile,
            "rule": rule,
            "vde_mass_kg": _display_quantity_text(vde_mass, "mass"),
            "fuelcons_mass_basis": fuelcons_basis,
            "fuelcons_mass_kg": _display_quantity_text(fuelcons_mass, "mass"),
            "required_inputs": required_inputs,
            "status": status,
            "notes": notes,
        }

    epa_status_mass = proposal_default_state.get("test_mass_kg")
    if regulatory_mode == "EPA" and epa_status_mass is None and proposal_curb_mass is not None:
        epa_status_mass = float(proposal_curb_mass) + 136.1
    epa_status, epa_note = _mass_profile_status(
        ["curb_mass" if proposal_curb_mass is None else "", "TWC" if proposal_twc_mass is None else ""],
        epa_status_mass,
    )

    perf_100_mass = float(proposal_curb_mass) + 100.0 if proposal_curb_mass is not None else None
    perf_100_status, perf_100_note = _mass_profile_status(["curb_mass" if proposal_curb_mass is None else ""], perf_100_mass)
    perf_300_mass = float(proposal_curb_mass) + 136.1 if proposal_curb_mass is not None else None
    perf_300_status, perf_300_note = _mass_profile_status(["curb_mass" if proposal_curb_mass is None else ""], perf_300_mass)
    tml_mass = proposal_default_state.get("test_mass_low_kg")
    tml_status, tml_note = _mass_profile_status(["TML" if tml_mass is None else ""], tml_mass)
    tmh_mass = proposal_default_state.get("test_mass_high_kg")
    tmh_status, tmh_note = _mass_profile_status(["TMH" if tmh_mass is None else ""], tmh_mass)
    gvwr_status, gvwr_note = _mass_profile_status(["GVWR" if gvwr_mass is None else ""], gvwr_mass)
    gcwr_required = []
    if gcwr_mass is None:
        gcwr_required.append("GCWR")
    if trailer_mass is None:
        gcwr_required.append("trailer_mass")
    gcwr_status, gcwr_note = _mass_profile_status(gcwr_required, gcwr_mass)
    custom_status, custom_note = _mass_profile_status(["manual_test_mass_kg" if custom_mass is None else ""], custom_mass)

    rows = [
        _profile_row("EPA Status", "EPA default test mass or curb +300 lb fallback", epa_status_mass, "TWC", proposal_twc_mass, "curb_mass + TWC", epa_status, epa_note),
        _profile_row("Performance Curb +100 kg", "curb_mass + 100 kg", perf_100_mass, "TEST_MASS", perf_100_mass, "curb_mass", perf_100_status, perf_100_note),
        _profile_row("Performance Curb +300 lb", "curb_mass + 136.1 kg", perf_300_mass, "TEST_MASS", perf_300_mass, "curb_mass", perf_300_status, perf_300_note),
        _profile_row("WLTP TML", "Resolved WLTP TML traceability line", tml_mass, "TEST_MASS", tml_mass, "TML", tml_status, tml_note),
        _profile_row("WLTP TMH", "Resolved WLTP TMH traceability line", tmh_mass, "TEST_MASS", tmh_mass, "TMH", tmh_status, tmh_note),
        _profile_row("GVWR", "GVWR scenario line", gvwr_mass, "TEST_MASS", gvwr_mass, "GVWR", gvwr_status, gvwr_note),
        _profile_row("GCWR", "GCWR / trailer what-if line", gcwr_mass, "TEST_MASS", gcwr_mass, "GCWR + trailer_mass + payload", gcwr_status, "Engineering what-if" if gcwr_status != "OK" else gcwr_note),
        _profile_row("Custom", "Manual/custom test mass scenario", custom_mass, custom_fuel_basis, proposal_twc_mass if custom_fuel_basis == "TWC" else custom_mass, "manual_test_mass_kg", custom_status, custom_note),
    ]
    return pd.DataFrame(rows)


def _default_mass_profile(snapshot: dict) -> str:
    regulatory_mode = str(snapshot.get("regulatory_mode") or "CUSTOM")
    if regulatory_mode == "EPA":
        return "EPA Status"
    if regulatory_mode == "WLTP":
        return "WLTP TMH"
    return "Custom"


def _resolve_enabled_mass_profile(editor_df: pd.DataFrame, *, default_profile: str) -> str:
    if editor_df is None or editor_df.empty:
        return default_profile
    enabled_rows = editor_df[editor_df["enabled"] == True]
    if enabled_rows.empty:
        return default_profile
    return str(enabled_rows.iloc[-1]["profile"])


def _apply_mass_profile_selection(ctx: dict, snapshot: dict, selected_profile: str) -> dict:
    proposal_curb_mass = snapshot.get("proposal_curb_mass")
    proposal_default_state = dict(snapshot.get("proposal_default_state") or {})
    custom_mass = snapshot.get("custom_mass")
    selected_profile = str(selected_profile or _default_mass_profile(snapshot))
    ctx["mass_profile_selected"] = selected_profile
    ctx["tire_load_mass_basis"] = "TEST_MASS"

    if selected_profile == "EPA Status":
        target_mass = proposal_default_state.get("test_mass_kg")
        if target_mass is None and proposal_curb_mass is not None:
            target_mass = float(proposal_curb_mass) + 136.1
        ctx["test_mass_basis"] = "PHYSICAL_TEST_MASS"
        ctx["test_mass_kg"] = target_mass
    elif selected_profile == "Performance Curb +100 kg":
        ctx["test_mass_basis"] = "PHYSICAL_TEST_MASS"
        ctx["test_mass_kg"] = float(proposal_curb_mass) + 100.0 if proposal_curb_mass is not None else None
    elif selected_profile == "Performance Curb +300 lb":
        ctx["test_mass_basis"] = "PHYSICAL_TEST_MASS"
        ctx["test_mass_kg"] = float(proposal_curb_mass) + 136.1 if proposal_curb_mass is not None else None
    elif selected_profile == "WLTP TML":
        ctx["test_mass_basis"] = "WLTP_TML"
        ctx["test_mass_kg"] = None
    elif selected_profile == "WLTP TMH":
        ctx["test_mass_basis"] = "WLTP_TMH"
        ctx["test_mass_kg"] = None
    elif selected_profile == "GVWR":
        ctx["test_mass_basis"] = "PHYSICAL_TEST_MASS"
        ctx["test_mass_kg"] = snapshot.get("gvwr_mass")
    elif selected_profile == "GCWR":
        ctx["test_mass_basis"] = "PHYSICAL_TEST_MASS"
        ctx["test_mass_kg"] = snapshot.get("gcwr_mass")
    else:
        ctx["test_mass_basis"] = "CUSTOM"
        ctx["test_mass_kg"] = custom_mass

    final_test_state = resolve_test_mass_state(dict(ctx))
    ctx["test_mass_kg"] = final_test_state.get("test_mass_kg")
    ctx["test_mass_low_kg"] = final_test_state.get("test_mass_low_kg")
    ctx["test_mass_high_kg"] = final_test_state.get("test_mass_high_kg")
    ctx["test_mass_basis"] = final_test_state.get("test_mass_basis")
    ctx["test_mass_use_default"] = ctx.get("test_mass_basis") not in {"CUSTOM", "PHYSICAL_TEST_MASS"}

    resolved_vde = resolve_tire_calculation_mass(dict(ctx))
    fuelcons_basis = "TEST_MASS"
    fuelcons_mass = resolved_vde.get("mass_kg")
    calc_status = "OK"
    notes = "used by VDE/roadload"
    if selected_profile == "EPA Status":
        fuelcons_basis = "TWC"
        fuelcons_mass = snapshot.get("proposal_twc_mass")
        calc_status = "Review"
        notes = "Fuelcons mass basis changes require VDE recalculation at the selected mass before fuel/CO2 conversion."
    elif selected_profile == "Custom":
        fuelcons_basis = str(snapshot.get("custom_fuelcons_basis") or "TEST_MASS").upper()
        if fuelcons_basis == "TWC":
            fuelcons_mass = snapshot.get("proposal_twc_mass")
            calc_status = "Review"
            notes = "Fuelcons mass basis changes require VDE recalculation at the selected mass before fuel/CO2 conversion."

    return {
        "selected_profile": selected_profile,
        "resolved_vde_mass_kg": resolved_vde.get("mass_kg"),
        "roadload_mass_basis": resolve_tire_load_mass_basis(ctx),
        "fuelcons_mass_basis": fuelcons_basis,
        "fuelcons_mass_kg": fuelcons_mass,
        "calculation_status": calc_status,
        "notes": notes,
    }


def _build_selected_mass_calculation_df(calc: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"output": "selected_profile", "value": str(calc.get("selected_profile") or "-"), "basis": "preset", "status": "OK", "notes": "-"},
            {"output": "resolved_vde_mass_kg", "value": _display_quantity_text(calc.get("resolved_vde_mass_kg"), "mass"), "basis": str(calc.get("roadload_mass_basis") or "-"), "status": "OK" if calc.get("resolved_vde_mass_kg") is not None else "Missing", "notes": "used by VDE/roadload"},
            {"output": "roadload_mass_basis", "value": str(calc.get("roadload_mass_basis") or "-"), "basis": "selected profile", "status": "OK", "notes": "-"},
            {"output": "fuelcons_mass_basis", "value": str(calc.get("fuelcons_mass_basis") or "-"), "basis": "selected profile", "status": calc.get("calculation_status") or "Review", "notes": calc.get("notes") or "-"},
            {"output": "fuelcons_mass_kg", "value": _display_quantity_text(calc.get("fuelcons_mass_kg"), "mass"), "basis": str(calc.get("fuelcons_mass_basis") or "-"), "status": "OK" if calc.get("fuelcons_mass_kg") is not None else "Review", "notes": "-"},
            {"output": "calculation_status", "value": str(calc.get("calculation_status") or "Review"), "basis": "selected profile", "status": str(calc.get("calculation_status") or "Review"), "notes": calc.get("notes") or "-"},
        ]
    )


def _apply_mass_spreadsheet_changes(editor_df: pd.DataFrame) -> list[str]:
    ctx = st.session_state.ctx
    errors: list[str] = []
    if editor_df is None or editor_df.empty:
        return ["Mass spreadsheet is empty."]

    st.session_state["vde_mass_cycle_table"] = editor_df.copy()
    rows = {
        str(row.get("parameter") or "").strip(): row
        for row in editor_df.to_dict(orient="records")
    }

    mass_kg = _editor_canonical_or_none((rows.get("mass_kg") or {}).get("value"), "mass")
    if mass_kg is None or mass_kg <= 0.0:
        errors.append("mass_kg must be a valid positive number.")
    else:
        ctx["mass_kg"] = float(mass_kg)

    payload_kg = _editor_canonical_or_none((rows.get("payload_kg") or {}).get("value"), "mass")
    options_kg = _editor_canonical_or_none((rows.get("options_kg") or {}).get("value"), "mass")
    if payload_kg is not None:
        ctx["payload_kg"] = float(payload_kg)
    if options_kg is not None:
        ctx["options_kg"] = float(options_kg)

    weight_dist = to_float((rows.get("weight_dist_fr_pct") or {}).get("value"))
    if weight_dist is None or weight_dist < 0.0 or weight_dist > 100.0:
        errors.append("weight_dist_fr_pct must be between 0 and 100.")
    else:
        ctx["weight_dist_fr_pct"] = float(weight_dist)

    wltp_category = str((rows.get("wltp_category") or {}).get("value") or "").strip().upper()
    if wltp_category:
        ctx["wltp_category"] = wltp_category

    tire_load_mass_basis = str((rows.get("tire_load_mass_basis") or {}).get("value") or "").strip().upper()
    if tire_load_mass_basis in {"TEST_MASS", "TWC"}:
        ctx["tire_load_mass_basis"] = tire_load_mass_basis
    elif tire_load_mass_basis:
        errors.append("tire_load_mass_basis must be TEST_MASS or TWC.")

    allowed_test_mass_basis = {"WLTP_TMH", "WLTP_TML", "CURB_PLUS_DRIVER", "CURB", "EPA_INERTIA_CLASS", "PHYSICAL_TEST_MASS", "CUSTOM", "GVWR", "GCWR_TRAILER"}
    selected_test_mass_basis = str((rows.get("test_mass_basis") or {}).get("value") or "").strip().upper()
    if selected_test_mass_basis:
        if selected_test_mass_basis not in allowed_test_mass_basis:
            errors.append("test_mass_basis is invalid.")
        else:
            ctx["test_mass_basis"] = selected_test_mass_basis

    edited_test_mass = _editor_canonical_or_none((rows.get("test_mass_kg") or {}).get("value"), "mass")
    if ctx.get("test_mass_basis") in {"CUSTOM", "PHYSICAL_TEST_MASS"}:
        if edited_test_mass is None or edited_test_mass <= 0.0:
            errors.append("test_mass_kg is required when test_mass_basis is CUSTOM or PHYSICAL_TEST_MASS.")
        else:
            ctx["test_mass_kg"] = float(edited_test_mass)
    elif edited_test_mass is not None:
        ctx["test_mass_kg"] = float(edited_test_mass)

    final_test_mass_state = resolve_test_mass_state(dict(ctx))
    ctx["test_mass_kg"] = final_test_mass_state.get("test_mass_kg")
    ctx["test_mass_low_kg"] = final_test_mass_state.get("test_mass_low_kg")
    ctx["test_mass_high_kg"] = final_test_mass_state.get("test_mass_high_kg")
    ctx["test_mass_basis"] = final_test_mass_state.get("test_mass_basis")
    ctx["test_mass_use_default"] = ctx.get("test_mass_basis") not in {"CUSTOM", "PHYSICAL_TEST_MASS"}

    return errors


def _spreadsheet_source_signature(ctx: dict) -> tuple[str, object, str]:
    return (
        str(ctx.get("mode") or ""),
        ctx.get("baseline_id") or ctx.get("vde_id_parent"),
        str(ctx.get("abc_total_source_ui") or ""),
    )


def _reset_spreadsheet_editor_state(ctx: dict) -> None:
    mode_slug = str(ctx.get("mode") or "default").replace(" ", "_")
    # Streamlit data editors keep their own widget state; clear it when the active source changes.
    for key in (
        "vde_vehicle_scenario_spreadsheet_editor",
        "vde_mass_cycle_spreadsheet_editor",
        "vde_mass_spreadsheet_editor",
        "vde_mass_reference_editor",
        "vde_mass_profiles_editor",
        f"vde_roadload_spreadsheet_{mode_slug}",
        f"vde_transmission_spreadsheet_{mode_slug}",
        f"vde_component_spreadsheet_{mode_slug}",
        "vde_vehicle_table",
        "vde_mass_cycle_table",
        "vde_mass_reference_table",
        "vde_abc_table",
        "vde_trans_loss_table",
        "vde_component_table",
    ):
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if str(key).startswith("vde_vehicle_info_"):
            st.session_state.pop(key, None)


SCENARIO_WORKBOOK_TRACKED_KEYS = tuple(
    dict.fromkeys(
        list(VDE_SETUP_CTX_DEFAULTS.keys())
        + [
            "id",
            "selected_vehicle_id",
            "selected_baseline_vde_id",
            "available_saved_vde_lines",
            "selected_reference_line",
            "vehicle_label",
            "electrification",
            "transmission_type",
            "drive_type",
            "fuel_type",
            "model_year",
            "cda_m2",
            "frontal_area_m2",
            "cd",
            "inertia_class",
            "prep_inertia_class",
            "mass_intention",
            "mass_profile_selected",
            "mass_profile_gvwr_kg",
            "mass_profile_gcwr_kg",
            "mass_profile_trailer_mass_kg",
            "mass_profile_custom_input_kg",
            "trailer_code",
            "trailer_A",
            "trailer_B",
            "trailer_C",
            "front_pressure_psi",
            "rear_pressure_psi",
            "tire_db_id",
            "tire_code",
            "neutral_drag_source",
            "residual_torque_front_Nm",
            "residual_torque_rear_Nm",
            "residual_torque_total_Nm",
            "wheel_radius_m",
            "brake_drag_force_N",
            "brake_temp_condition",
            "brake_release_condition",
            "caliper_drag_status",
            "pad_drag_status",
            "parking_brake_drag_flag",
            "axle_hub_mode",
            "axle_hub_A",
            "axle_hub_B",
            "axle_hub_C",
            "axle_hub_delta_A",
            "axle_hub_delta_B",
            "axle_hub_delta_C",
            "parasitic_mode",
            "parasitic_delta_A",
            "parasitic_delta_B",
            "parasitic_delta_C",
        ]
    )
)

SCENARIO_WORKBOOK_WIDGET_PREFIXES = (
    "hdr_",
    "vde_vehicle_info_",
    "mass_profile_",
    "aero_",
    "rr_",
    "brake_",
    "parasitic_",
    "tire_",
    "transmission_",
    "component_",
    "technical_build_up_",
    "from_test_",
    "trailer_",
    "axle_hub_",
)

SCENARIO_WORKBOOK_TRAILER_PRESETS = {
    "TRAILER_LIGHT": {"weight_kg": 750.0, "A": 18.0, "B": 0.0500, "C": 0.0012},
    "TRAILER_BOX": {"weight_kg": 1800.0, "A": 34.0, "B": 0.0820, "C": 0.0027},
    "TRAILER_HEAVY": {"weight_kg": 3200.0, "A": 52.0, "B": 0.1140, "C": 0.0042},
}


def _scenario_workbook_label(column_id: str) -> str:
    return {
        "baseline": "Baseline",
        "walked_1": "Walked #1",
        "walked_2": "Walked #2",
    }.get(str(column_id), str(column_id))


def _sync_scenario_workbook_unit_system() -> None:
    st.session_state["unit_system"] = normalize_unit_system(
        st.session_state.get("scenario_workbook_display_units")
    )


def _reset_scenario_workbook_widget_state(ctx: dict) -> None:
    _reset_spreadsheet_editor_state(ctx)
    explicit_keys = {
        "hdr_leg",
        "hdr_cat",
        "hdr_make_sel",
        "hdr_make_text",
        "hdr_model",
        "hdr_year",
        "hdr_elec",
        "hdr_trans",
        "hdr_notes",
        "mass_profile_custom_fuelcons_basis_selector",
        "mass_profile_custom_input",
        "scenario_workbook_active_column_radio",
        "scenario_workbook_section_selector",
        "scenario_workbook_walk_from_selector",
        "scenario_workbook_display_units",
    }
    for key in list(st.session_state.keys()):
        if key in explicit_keys or any(str(key).startswith(prefix) for prefix in SCENARIO_WORKBOOK_WIDGET_PREFIXES):
            st.session_state.pop(key, None)


def _ensure_scenario_workbook_state(ctx: dict) -> dict:
    columns = ctx.get("scenario_workbook_columns")
    if not isinstance(columns, dict):
        columns = {}
    defaults = {
        "baseline": {"label": "Baseline", "kind": "baseline", "walk_from": None, "direct": {}},
        "walked_1": {"label": "Walked #1", "kind": "walked", "walk_from": "baseline", "direct": {}},
        "walked_2": {"label": "Walked #2", "kind": "walked", "walk_from": "walked_1", "direct": {}},
    }
    for column_id, payload in defaults.items():
        item = dict(columns.get(column_id) or {})
        item.setdefault("label", payload["label"])
        item.setdefault("kind", payload["kind"])
        item.setdefault("walk_from", payload["walk_from"])
        if not isinstance(item.get("direct"), dict):
            item["direct"] = {}
        columns[column_id] = item
    ctx["scenario_workbook_columns"] = columns
    active_column = str(ctx.get("scenario_workbook_active_column") or "walked_1")
    if active_column not in {"walked_1", "walked_2"}:
        active_column = "walked_1"
    ctx["scenario_workbook_active_column"] = active_column
    ctx.setdefault("axle_hub_mode", "Inherit")
    ctx.setdefault("parasitic_mode", "Inherit")
    return columns


def _scenario_workbook_baseline_state(ctx: dict) -> dict:
    base = dict(ctx.get("selected_baseline_row") or ctx.get("baseline_dict") or {})
    state = {}
    alias_map = {
        "id": ("id",),
        "legislation": ("legislation",),
        "category": ("category",),
        "make": ("make", "manufacturer"),
        "model": ("model", "vehicle_label"),
        "year": ("year", "model_year"),
        "notes": ("proposal", "notes"),
        "mass_kg": ("mass_kg", "baseline_mass_kg"),
        "payload_kg": ("payload_kg",),
        "options_kg": ("options_kg",),
        "test_mass_kg": ("test_mass_kg",),
        "test_mass_low_kg": ("test_mass_low_kg",),
        "test_mass_high_kg": ("test_mass_high_kg",),
        "weight_dist_fr_pct": ("weight_dist_fr_pct",),
        "wltp_category": ("wltp_category",),
        "electrification": ("electrification",),
        "transmission_type": ("transmission_type",),
        "drive_type": ("drive_type",),
        "fuel_type": ("fuel_type",),
        "A": ("A", "baseline_A_N"),
        "B": ("B", "baseline_B_N_per_kph"),
        "C": ("C", "baseline_C_N_per_kph2"),
        "cda_m2": ("cda_m2",),
        "cd": ("cd",),
        "frontal_area_m2": ("frontal_area_m2",),
        "rrc_N_per_kN": ("rrc_N_per_kN",),
        "front_pressure_psi": ("front_pressure_psi",),
        "rear_pressure_psi": ("rear_pressure_psi",),
        "tire_A_final": ("tire_A_final",),
        "tire_B_final": ("tire_B_final",),
        "tire_C_final": ("tire_C_final",),
        "trans_A_coef_N": ("trans_A_coef_N",),
        "trans_B_coef_Npkph": ("trans_B_coef_Npkph", "trans_B_Npkph"),
        "trans_C_coef_Npkph2": ("trans_C_coef_Npkph2",),
        "brake_A_coef_N": ("brake_A_coef_N",),
        "brake_B_Npkph": ("brake_B_Npkph",),
        "brake_C_coef_Npkph2": ("brake_C_coef_Npkph2",),
        "parasitic_A_coef_N": ("parasitic_A_coef_N",),
        "parasitic_B_Npkph": ("parasitic_B_Npkph",),
        "parasitic_C_coef_Npkph2": ("parasitic_C_coef_Npkph2",),
    }
    for key in SCENARIO_WORKBOOK_TRACKED_KEYS:
        for candidate in alias_map.get(key, (key,)):
            if candidate in base and base.get(candidate) not in (None, ""):
                state[key] = base.get(candidate)
                break
        else:
            if key in ctx and ctx.get(key) not in (None, ""):
                state[key] = ctx.get(key)
    state["selected_baseline_vde_id"] = state.get("id") or ctx.get("baseline_id") or ctx.get("vde_id_parent")
    state["selected_reference_line"] = "baseline"
    if state.get("year") is not None:
        state["year"] = int(to_float(state.get("year"), ctx.get("year", 2024)) or 2024)
    state.setdefault("notes", str(ctx.get("notes") or ""))
    return state


def _scenario_workbook_source_options(column_id: str) -> list[str]:
    if column_id == "walked_1":
        return ["baseline"]
    if column_id == "walked_2":
        return ["baseline", "walked_1"]
    return ["baseline"]


def _resolve_scenario_workbook_state(ctx: dict, column_id: str, *, _stack: tuple[str, ...] = ()) -> dict:
    _ensure_scenario_workbook_state(ctx)
    if column_id == "baseline":
        return _scenario_workbook_baseline_state(ctx)
    if column_id in _stack:
        return _scenario_workbook_baseline_state(ctx)
    column = dict((ctx.get("scenario_workbook_columns") or {}).get(column_id) or {})
    source_id = str(column.get("walk_from") or "baseline")
    if source_id not in _scenario_workbook_source_options(column_id):
        source_id = "baseline"
    source_state = _resolve_scenario_workbook_state(ctx, source_id, _stack=_stack + (column_id,))
    effective = dict(source_state)
    for key, value in dict(column.get("direct") or {}).items():
        if value is None:
            effective.pop(key, None)
        else:
            effective[key] = value
    effective["selected_reference_line"] = source_id
    effective["walk_from"] = source_id
    effective["id"] = None
    effective["vde_id_parent"] = source_state.get("id") or ctx.get("baseline_id") or ctx.get("vde_id_parent")
    if effective.get("year") is not None:
        effective["year"] = int(to_float(effective.get("year"), ctx.get("year", 2024)) or 2024)
    return effective


def _apply_scenario_workbook_state_to_ctx(ctx: dict, column_id: str) -> None:
    effective = _resolve_scenario_workbook_state(ctx, column_id)
    for key in SCENARIO_WORKBOOK_TRACKED_KEYS:
        if key in effective:
            ctx[key] = effective.get(key)
    if effective.get("year") is not None:
        ctx["year"] = int(to_float(effective.get("year"), ctx.get("year", 2024)) or 2024)
    ctx["notes"] = str(effective.get("notes") or "")
    ctx["selected_reference_line"] = str(effective.get("walk_from") or "baseline")


def _scenario_workbook_column_status(ctx: dict, column_id: str) -> tuple[str, str]:
    effective = _resolve_scenario_workbook_state(ctx, column_id)
    if column_id == "baseline":
        return ("Saved" if effective.get("selected_baseline_vde_id") else "Pending", str(effective.get("selected_baseline_vde_id") or "Select baseline"))
    source_id = str(effective.get("walk_from") or "baseline")
    if source_id not in _scenario_workbook_source_options(column_id):
        return ("Blocked", "Walk From invalid")
    required_meta = ["legislation", "category", "make", "model", "year", "electrification", "transmission_type"]
    missing_meta = [key for key in required_meta if effective.get(key) in (None, "", [])]
    if not str(effective.get("notes") or "").strip():
        return ("Missing", "Proposal required")
    if missing_meta:
        return ("Review", ", ".join(missing_meta[:3]))
    return ("Ready", str(effective.get("notes") or "Ready"))


def _capture_active_scenario_workbook_direct(ctx: dict) -> bool:
    columns = _ensure_scenario_workbook_state(ctx)
    active_column = str(ctx.get("scenario_workbook_active_column") or "walked_1")
    if active_column not in {"walked_1", "walked_2"}:
        return False
    source_id = str((columns.get(active_column) or {}).get("walk_from") or "baseline")
    if source_id not in _scenario_workbook_source_options(active_column):
        source_id = "baseline"
    source_state = _resolve_scenario_workbook_state(ctx, source_id)
    direct: dict[str, object] = {}
    ignored = {"id", "selected_baseline_vde_id", "vde_id_parent", "selected_reference_line"}
    for key in SCENARIO_WORKBOOK_TRACKED_KEYS:
        if key in ignored:
            continue
        current_value = ctx.get(key)
        source_value = source_state.get(key)
        if key == "year":
            current_value = int(to_float(current_value, ctx.get("year", 2024)) or 2024) if current_value not in (None, "") else None
            source_value = int(to_float(source_value, ctx.get("year", 2024)) or 2024) if source_value not in (None, "") else None
        if current_value != source_value:
            direct[key] = current_value
    existing = dict((columns.get(active_column) or {}).get("direct") or {})
    if direct != existing:
        columns[active_column]["direct"] = direct
        ctx["scenario_workbook_columns"] = columns
        return True
    return False


def _build_scenario_workbook_matrix_df(ctx: dict) -> pd.DataFrame:
    columns = _ensure_scenario_workbook_state(ctx)
    effective_states = {
        "baseline": _resolve_scenario_workbook_state(ctx, "baseline"),
        "walked_1": _resolve_scenario_workbook_state(ctx, "walked_1"),
        "walked_2": _resolve_scenario_workbook_state(ctx, "walked_2"),
    }
    matrix_rows = [
        ("VDE-ID", lambda col_id: effective_states[col_id].get("selected_baseline_vde_id") if col_id == "baseline" else "New / Insert"),
        ("Legislation", lambda col_id: effective_states[col_id].get("legislation") or "-"),
        ("Model Year", lambda col_id: effective_states[col_id].get("year") or "-"),
        ("Make", lambda col_id: effective_states[col_id].get("make") or "-"),
        ("Model", lambda col_id: effective_states[col_id].get("model") or "-"),
        ("Description", lambda col_id: effective_states[col_id].get("notes") or "-"),
        ("Status", lambda col_id: _scenario_workbook_column_status(ctx, col_id)[0]),
        ("Walk From", lambda col_id: "-" if col_id == "baseline" else _scenario_workbook_label(str((columns.get(col_id) or {}).get("walk_from") or "baseline"))),
        ("Proposal Direct", lambda col_id: "-" if col_id == "baseline" else str(dict((columns.get(col_id) or {}).get("direct") or {}).get("notes") or "-")),
        ("Proposal Effective", lambda col_id: effective_states[col_id].get("notes") or "-"),
    ]
    rows = []
    for label, getter in matrix_rows:
        rows.append(
            {
                "field": label,
                "Baseline": getter("baseline"),
                "Walked #1": getter("walked_1"),
                "Walked #2": getter("walked_2"),
            }
        )
    return pd.DataFrame(rows)


def _render_scenario_workbook_matrix(ctx: dict) -> None:
    columns = _ensure_scenario_workbook_state(ctx)
    render_vde_workbook_table(
        _build_scenario_workbook_matrix_df(ctx),
        title="Scenario Matrix",
        table_id="scenario-workbook-matrix",
    )

    c1, c2, c3, c4 = st.columns([1.0, 1.2, 1.2, 0.8])
    active_label = _scenario_workbook_label(str(ctx.get("scenario_workbook_active_column") or "walked_1"))
    selected_label = c1.radio(
        "Active walked column",
        ["Walked #1", "Walked #2"],
        horizontal=True,
        index=["Walked #1", "Walked #2"].index(active_label if active_label in {"Walked #1", "Walked #2"} else "Walked #1"),
        key="scenario_workbook_active_column_radio",
    )
    selected_column = "walked_1" if selected_label == "Walked #1" else "walked_2"
    if selected_column != ctx.get("scenario_workbook_active_column"):
        ctx["scenario_workbook_active_column"] = selected_column
        _apply_scenario_workbook_state_to_ctx(ctx, selected_column)
        _reset_scenario_workbook_widget_state(ctx)
        st.rerun()

    source_options = _scenario_workbook_source_options(selected_column)
    current_source = str((columns.get(selected_column) or {}).get("walk_from") or source_options[0])
    if current_source not in source_options:
        current_source = source_options[0]
    chosen_source = c2.selectbox(
        "Walk From",
        source_options,
        index=source_options.index(current_source),
        key="scenario_workbook_walk_from_selector",
        format_func=_scenario_workbook_label,
    )
    if chosen_source != current_source:
        columns[selected_column]["walk_from"] = chosen_source
        ctx["scenario_workbook_columns"] = columns
        _apply_scenario_workbook_state_to_ctx(ctx, selected_column)
        _reset_scenario_workbook_widget_state(ctx)
        st.rerun()

    status_label, status_detail = _scenario_workbook_column_status(ctx, selected_column)
    c3.metric("Selected column status", status_label)
    c4.button("+ Add Column", disabled=True, help="Future expansion after Baseline / Walked #1 / Walked #2 settles.")
    st.caption(f"{_scenario_workbook_label(selected_column)} inherits from {_scenario_workbook_label(chosen_source)}. Preview & Save always uses the effective accumulated state.")
    if status_detail:
        st.caption(status_detail)


def _render_scenario_origin_metadata_menu(*, reset_ctx) -> None:
    ctx = st.session_state.ctx
    render_scenario_origin_spreadsheet_section(reset_ctx=reset_ctx)
    st.divider()
    unit_options = ["Metric", "US customary"]
    current_units = _current_unit_system()
    if st.session_state.get("scenario_workbook_display_units") != current_units:
        st.session_state["scenario_workbook_display_units"] = current_units
    selector_cols = st.columns([1.0, 1.0, 1.3])
    selector_cols[0].radio(
        "Display units",
        unit_options,
        horizontal=True,
        key="scenario_workbook_display_units",
        on_change=_sync_scenario_workbook_unit_system,
    )
    selector_cols[1].metric("Reference line", _scenario_workbook_label(str(ctx.get("selected_reference_line") or "baseline")))
    selector_cols[2].metric("Saved baseline lines", len(ctx.get("available_saved_vde_lines") or []))
    render_vehicle_metadata_spreadsheet_section()
    st.divider()
    _render_workbook_field_editor(
        title="Cycle",
        specs=[
            {"label": "selected_vehicle_id", "ctx_key": "selected_vehicle_id", "kind": "int", "notes": "Optional vehicle link"},
            {"label": "selected_baseline_vde_id", "ctx_key": "selected_baseline_vde_id", "kind": "int", "notes": "Loaded baseline VDE id"},
            {"label": "cycle", "ctx_key": "cycle_name", "kind": "text", "notes": "Cycle selection is temporary text here; legacy footer still hosts upload/manage flow"},
            {"label": "cycle_source", "ctx_key": "cycle_source", "kind": "text", "notes": "standard / uploaded / pending"},
            {"label": "selected_reference_line", "ctx_key": "selected_reference_line", "kind": "text", "notes": "Current Walk From source"},
        ],
        key="scenario_workbook_cycle_editor",
        caption="Cycle-related fields stay in spreadsheet form here. The old cycle uploader remains in the Legacy footer until this menu fully replaces it.",
    )


def _render_mass_aero_workbook_menu(*, prefill=None) -> None:
    ctx = st.session_state.ctx
    mass_options = [
        "EPA_STATUS",
        "EPA_PLUS_1_TWC",
        "WLTP_TML",
        "WLTP_TMH",
        "GVWR",
        "GCWR",
        "PERF_CURB_100KG",
        "PERF_CURB_300LB",
        "CUSTOM",
    ]
    current_intention = str(ctx.get("mass_intention") or "EPA_STATUS")
    if current_intention not in mass_options:
        current_intention = "EPA_STATUS"
    top_cols = st.columns([1.0, 1.0, 1.0])
    ctx["mass_intention"] = top_cols[0].selectbox("Mass intention", mass_options, index=mass_options.index(current_intention), key="scenario_workbook_mass_intention")
    current_aero_mode = str(ctx.get("aero_calculation_mode") or "Inherited")
    aero_modes = ["Inherited", "Absolute", "Delta"]
    if current_aero_mode not in aero_modes:
        current_aero_mode = "Inherited"
    ctx["aero_calculation_mode"] = top_cols[1].selectbox("Aero mode", aero_modes, index=aero_modes.index(current_aero_mode), key="scenario_workbook_aero_mode")
    trailer_options = ["MANUAL"] + list(SCENARIO_WORKBOOK_TRAILER_PRESETS.keys())
    trailer_code = str(ctx.get("trailer_code") or "MANUAL")
    if trailer_code not in trailer_options:
        trailer_code = "MANUAL"
    ctx["trailer_code"] = top_cols[2].selectbox("Trailer code", trailer_options, index=trailer_options.index(trailer_code), key="scenario_workbook_trailer_code")
    preset = SCENARIO_WORKBOOK_TRAILER_PRESETS.get(ctx["trailer_code"])
    if preset:
        ctx["mass_profile_trailer_mass_kg"] = float(preset["weight_kg"])
        ctx["trailer_A"] = float(preset["A"])
        ctx["trailer_B"] = float(preset["B"])
        ctx["trailer_C"] = float(preset["C"])
    render_mass_spreadsheet_section(prefill=prefill)
    st.divider()
    _render_workbook_field_editor(
        title="Aero / Trailer",
        specs=[
            {"label": "aero_mode", "ctx_key": "aero_calculation_mode", "kind": "text", "notes": "Inherit / Absolute / Delta"},
            {"label": "Cd", "ctx_key": "cd", "kind": "float", "notes": "Aerodynamic coefficient"},
            {"label": "frontal_area_m2", "ctx_key": "frontal_area_m2", "kind": "float", "notes": "Frontal area"},
            {"label": "CdA", "ctx_key": "cda_m2", "kind": "float", "notes": "Explicit CdA overrides derived Cd * area"},
            {"label": "trailer_code", "ctx_key": "trailer_code", "kind": "text", "notes": "Use preset code or MANUAL"},
            {"label": "trailer_weight_kg", "ctx_key": "mass_profile_trailer_mass_kg", "kind": "mass", "notes": "Required for trailer / GCWR scenarios"},
            {"label": "trailer_A", "ctx_key": "trailer_A", "kind": "force", "notes": "Trailer drag A"},
            {"label": "trailer_B", "ctx_key": "trailer_B", "kind": "force_per_speed", "notes": "Trailer drag B"},
            {"label": "trailer_C", "ctx_key": "trailer_C", "kind": "force_per_speed_squared", "notes": "Trailer drag C"},
        ],
        key="scenario_workbook_mass_aero_editor",
        caption="Mass & Aero stays in spreadsheet form. The old aero widgets are now footer-only legacy.",
    )
    if str(ctx.get("trailer_code") or "MANUAL") in SCENARIO_WORKBOOK_TRAILER_PRESETS:
        preset = SCENARIO_WORKBOOK_TRAILER_PRESETS[str(ctx.get("trailer_code"))]
        ctx["mass_profile_trailer_mass_kg"] = float(preset["weight_kg"])
        ctx["trailer_A"] = float(preset["A"])
        ctx["trailer_B"] = float(preset["B"])
        ctx["trailer_C"] = float(preset["C"])
    cda_value = to_float(ctx.get("cda_m2"))
    if cda_value is not None:
        ctx["aero_C_coef_Npkph2"] = 0.5 * 1.2 * float(cda_value) * (1 / 3.6) ** 2
    if ctx.get("trailer_code") != "MANUAL":
        st.caption(f"Mock trailer preset `{ctx.get('trailer_code')}` filled the trailer row. You can still override A/B/C manually here.")


def _render_tire_workbook_menu(*, base_row: dict | None = None, saved_vde_id: int | None = None) -> None:
    _render_workbook_component_row(
        component_label="Tires",
        key="scenario_workbook_tire_component_row",
        base_row=base_row,
        saved_vde_id=saved_vde_id,
    )
    st.divider()
    _render_workbook_field_editor(
        title="Tire workbook",
        specs=[
            {"label": "tire_mode", "ctx_key": "component_mode_tires", "kind": "text", "notes": "Spreadsheet-controlled tire mode"},
            {"label": "tire_code", "ctx_key": "tire_code", "kind": "text", "notes": "Optional tire code"},
            {"label": "tire_db_id", "ctx_key": "tire_db_id", "kind": "int", "notes": "Optional Tire DB id"},
            {"label": "tire_size", "ctx_key": "tire_size", "kind": "text", "notes": "Scenario tire size"},
            {"label": "psi_front", "ctx_key": "front_pressure_psi", "kind": "float", "notes": "Front pressure"},
            {"label": "psi_rear", "ctx_key": "rear_pressure_psi", "kind": "float", "notes": "Rear pressure"},
            {"label": "SMERF", "ctx_key": "smerf", "kind": "float", "notes": "Optional audit field"},
            {"label": "rrc_N_per_kN", "ctx_key": "rrc_N_per_kN", "kind": "rrc", "notes": "Final rolling resistance coefficient"},
            {"label": "improvement_pct", "ctx_key": "tire_improvement_pct", "kind": "float", "notes": "Improvement scenario"},
            {"label": "notes", "ctx_key": "tire_manual_delta_rr_notes", "kind": "text", "notes": "Tire provenance / notes"},
        ],
        key="scenario_workbook_tire_fields",
        caption="Tire menu now stays in spreadsheet format. The older tire scenario wizard is footer-only legacy.",
    )


def _render_brake_workbook_menu(*, base_row: dict | None = None) -> None:
    _render_workbook_component_row(
        component_label="Brakes",
        key="scenario_workbook_brake_component_row",
        base_row=base_row,
    )
    st.divider()
    _render_workbook_field_editor(
        title="Brake workbook",
        specs=[
            {"label": "brake_mode", "ctx_key": "component_mode_brakes", "kind": "text", "notes": "Inherit / Absolute / Delta / Not used"},
            {"label": "residual_torque_front_Nm", "ctx_key": "residual_torque_front_Nm", "kind": "float", "notes": "Residual torque front"},
            {"label": "residual_torque_rear_Nm", "ctx_key": "residual_torque_rear_Nm", "kind": "float", "notes": "Residual torque rear"},
            {"label": "residual_torque_total_Nm", "ctx_key": "residual_torque_total_Nm", "kind": "float", "notes": "Audit / future calc"},
            {"label": "wheel_radius_m", "ctx_key": "wheel_radius_m", "kind": "float", "notes": "Wheel radius"},
            {"label": "brake_drag_force_N", "ctx_key": "brake_drag_force_N", "kind": "float", "notes": "Brake drag force"},
            {"label": "brake_temp_condition", "ctx_key": "brake_temp_condition", "kind": "text", "notes": "Brake temperature condition"},
            {"label": "brake_release_condition", "ctx_key": "brake_release_condition", "kind": "text", "notes": "Brake release condition"},
            {"label": "caliper_drag_status", "ctx_key": "caliper_drag_status", "kind": "text", "notes": "Caliper drag status"},
            {"label": "pad_drag_status", "ctx_key": "pad_drag_status", "kind": "text", "notes": "Pad drag status"},
            {"label": "parking_brake_drag_flag", "ctx_key": "parking_brake_drag_flag", "kind": "text", "notes": "true / false"},
        ],
        key="scenario_workbook_brake_fields",
        caption="Brake menu stays in spreadsheet form. The old brake wizard is now legacy-only.",
    )


def _render_axle_hub_workbook_menu() -> None:
    _render_workbook_field_editor(
        title="Axle & Hubs workbook",
        specs=[
            {"label": "axle_hub_mode", "ctx_key": "axle_hub_mode", "kind": "text", "notes": "Inherit / Absolute / Delta ABC / Not used"},
            {"label": "axle_hub_A", "ctx_key": "axle_hub_A", "kind": "force", "notes": "Absolute A"},
            {"label": "axle_hub_B", "ctx_key": "axle_hub_B", "kind": "force_per_speed", "notes": "Absolute B"},
            {"label": "axle_hub_C", "ctx_key": "axle_hub_C", "kind": "force_per_speed_squared", "notes": "Absolute C"},
            {"label": "delta_A", "ctx_key": "axle_hub_delta_A", "kind": "force", "notes": "Delta A"},
            {"label": "delta_B", "ctx_key": "axle_hub_delta_B", "kind": "force_per_speed", "notes": "Delta B"},
            {"label": "delta_C", "ctx_key": "axle_hub_delta_C", "kind": "force_per_speed_squared", "notes": "Delta C"},
        ],
        key="scenario_workbook_axle_hub_fields",
        caption="Axle & Hubs now lives in spreadsheet form. It is staged separately even while the active physics path still treats it as future-ready metadata.",
    )


def _render_parasitic_workbook_menu(*, base_row: dict | None = None) -> None:
    _render_workbook_component_row(
        component_label="Parasitics / Hubs / Axle",
        key="scenario_workbook_parasitic_component_row",
        base_row=base_row,
    )
    st.divider()
    _render_workbook_field_editor(
        title="Parasitic workbook",
        specs=[
            {"label": "parasitic_mode", "ctx_key": "parasitic_mode", "kind": "text", "notes": "Inherit / Absolute / Delta ABC / Not used"},
            {"label": "delta_A", "ctx_key": "parasitic_delta_A", "kind": "force", "notes": "Future explicit delta A"},
            {"label": "delta_B", "ctx_key": "parasitic_delta_B", "kind": "force_per_speed", "notes": "Future explicit delta B"},
            {"label": "delta_C", "ctx_key": "parasitic_delta_C", "kind": "force_per_speed_squared", "notes": "Future explicit delta C"},
        ],
        key="scenario_workbook_parasitic_fields",
        caption="Parasitic Losses stay in spreadsheet form here. The old parasitic wizard is now footer-only legacy.",
    )


def _build_scenario_preview_matrix_df(ctx: dict) -> pd.DataFrame:
    active_column = str(ctx.get("scenario_workbook_active_column") or "walked_1")
    preview = _safe_workflow_preview(ctx)
    baseline = _resolve_scenario_workbook_state(ctx, "baseline")
    columns = _ensure_scenario_workbook_state(ctx)
    active_direct = dict((columns.get(active_column) or {}).get("direct") or {})
    abc_total = dict(preview.get("abc_total") or {})
    abc_net = dict(preview.get("abc_net") or {})
    transmission = dict(preview.get("transmission_losses") or {})
    vde_total = dict(preview.get("vde_total") or {})
    vde_net = dict(preview.get("vde_net") or {})
    rows = [
        {"field": "Walk From", "value": _scenario_workbook_label(str(ctx.get("selected_reference_line") or "baseline"))},
        {"field": "Proposal Direct", "value": str(active_direct.get("notes") or "-")},
        {"field": "Proposal Effective", "value": str(ctx.get("notes") or "-")},
        {"field": "ABC_TOTAL_A", "value": _display_quantity_text(abc_total.get("A"), "force")},
        {"field": "ABC_TOTAL_B", "value": _display_quantity_text(abc_total.get("B"), "force_per_speed")},
        {"field": "ABC_TOTAL_C", "value": _display_quantity_text(abc_total.get("C"), "force_per_speed_squared")},
        {"field": "Transmission_A_loss", "value": _display_quantity_text(ctx.get("trans_A_coef_N"), "force")},
        {"field": "Transmission_B_loss", "value": _display_quantity_text(ctx.get("trans_B_coef_Npkph"), "force_per_speed")},
        {"field": "Transmission_C_loss", "value": _display_quantity_text(ctx.get("trans_C_coef_Npkph2"), "force_per_speed_squared")},
        {"field": "ABC_NET_A", "value": _display_quantity_text(abc_net.get("A"), "force")},
        {"field": "ABC_NET_B", "value": _display_quantity_text(abc_net.get("B"), "force_per_speed")},
        {"field": "ABC_NET_C", "value": _display_quantity_text(abc_net.get("C"), "force_per_speed_squared")},
        {"field": "VDE_TOTAL", "value": format_quantity(vde_total.get("mj_per_km"), "energy_per_distance", include_unit=True, unavailable="-", format_str="%.3f")},
        {"field": "VDE_NET", "value": format_quantity(vde_net.get("mj_per_km"), "energy_per_distance", include_unit=True, unavailable="-", format_str="%.3f")},
        {"field": "Delta vs Baseline", "value": format_quantity((to_float(vde_total.get("mj_per_km")) or 0.0) - (to_float(baseline.get("vde_total_mj_per_km")) or 0.0), "energy_per_distance", include_unit=True, unavailable="-", format_str="%.3f")},
        {"field": "Mass & Aero Status", "value": _scenario_workbook_column_status(ctx, active_column)[0]},
        {"field": "Transmission Status", "value": "OK" if str(transmission.get("status") or "").lower() == "available" else "Review"},
        {"field": "Save Status", "value": "Ready" if preview.get("ok") and vde_total.get("mj_per_km") is not None else "Pending"},
    ]
    return pd.DataFrame(rows)


def _build_scenario_workbook_audit_df(ctx: dict) -> pd.DataFrame:
    active_column = str(ctx.get("scenario_workbook_active_column") or "walked_1")
    columns = _ensure_scenario_workbook_state(ctx)
    source_id = str((columns.get(active_column) or {}).get("walk_from") or "baseline")
    source_state = _resolve_scenario_workbook_state(ctx, source_id)
    effective_state = _resolve_scenario_workbook_state(ctx, active_column)
    direct = dict((columns.get(active_column) or {}).get("direct") or {})
    audit_fields = [
        "legislation",
        "category",
        "make",
        "model",
        "year",
        "notes",
        "mass_kg",
        "payload_kg",
        "options_kg",
        "test_mass_kg",
        "weight_dist_fr_pct",
        "cda_m2",
        "A",
        "B",
        "C",
        "trans_A_coef_N",
        "trans_B_coef_Npkph",
        "trans_C_coef_Npkph2",
        "brake_A_coef_N",
        "brake_B_Npkph",
        "brake_C_coef_Npkph2",
        "axle_hub_A",
        "axle_hub_B",
        "axle_hub_C",
        "parasitic_A_coef_N",
        "parasitic_B_Npkph",
        "parasitic_C_coef_Npkph2",
        "residual_torque_front_Nm",
        "residual_torque_rear_Nm",
        "trailer_code",
    ]
    rows = []
    for field in audit_fields:
        raw_value = direct.get(field, "")
        effective_value = effective_state.get(field)
        mode = "inherit" if field not in direct else "direct"
        status = "OK"
        if field in {"notes", "legislation", "make", "model"} and effective_value in (None, "", []):
            status = "Missing"
        rows.append(
            {
                "field": field,
                "raw_value": raw_value if raw_value not in (None, "") else "blank",
                "effective_value": effective_value if effective_value not in (None, "") else "-",
                "source_column": _scenario_workbook_label(active_column if field in direct else source_id),
                "mode": mode,
                "source": "manual" if field in direct else "inherit",
                "status": status,
                "notes": "" if field in direct else f"inherited from {_scenario_workbook_label(source_id)}",
            }
        )
    return pd.DataFrame(rows)


def _format_workbook_scalar(value, kind: str) -> str:
    if value in (None, ""):
        return ""
    if kind == "mass":
        return _display_quantity_text(value, "mass", unavailable="")
    if kind == "force":
        return _display_quantity_text(value, "force", unavailable="")
    if kind == "force_per_speed":
        return _display_quantity_text(value, "force_per_speed", unavailable="")
    if kind == "force_per_speed_squared":
        return _display_quantity_text(value, "force_per_speed_squared", unavailable="")
    if kind == "rrc":
        return format_quantity(value, "rrc", _current_unit_system(), include_unit=False, unavailable="", format_str="%.3f")
    if kind == "int":
        numeric = to_float(value)
        return "" if numeric is None else str(int(numeric))
    if kind == "float":
        numeric = to_float(value)
        return "" if numeric is None else f"{float(numeric):.6f}".rstrip("0").rstrip(".")
    return str(value)


def _parse_workbook_scalar(value, kind: str):
    if value in (None, ""):
        return "" if kind == "text" else None
    if kind == "mass":
        return _editor_canonical_or_none(value, "mass")
    if kind == "force":
        return _editor_canonical_or_none(value, "force")
    if kind == "force_per_speed":
        return _editor_canonical_or_none(value, "force_per_speed")
    if kind == "force_per_speed_squared":
        return _editor_canonical_or_none(value, "force_per_speed_squared")
    if kind == "rrc":
        return _editor_canonical_or_none(value, "rrc")
    if kind == "int":
        numeric = _editor_float_or_none(value)
        return None if numeric is None else int(numeric)
    if kind == "float":
        return _editor_float_or_none(value)
    return str(value).strip()


def _build_workbook_field_df(ctx: dict, specs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in specs:
        raw_value = ctx.get(spec["ctx_key"], spec.get("default"))
        required = bool(spec.get("required"))
        rows.append(
            {
                "field": spec["label"],
                "value": _format_workbook_scalar(raw_value, str(spec.get("kind") or "text")),
                "source": str(spec.get("source") or "scenario"),
                "status": "Missing" if required and raw_value in (None, "", []) else "OK",
                "notes": str(spec.get("notes") or ""),
            }
        )
    return pd.DataFrame(rows)


def _apply_workbook_field_df(ctx: dict, editor_df: pd.DataFrame, specs: list[dict]) -> list[str]:
    if editor_df is None or editor_df.empty:
        return []
    spec_by_label = {str(spec["label"]): spec for spec in specs}
    errors: list[str] = []
    for row in editor_df.to_dict(orient="records"):
        label = str(row.get("field") or "")
        spec = spec_by_label.get(label)
        if not spec:
            continue
        parsed = _parse_workbook_scalar(row.get("value"), str(spec.get("kind") or "text"))
        if spec.get("required") and parsed in (None, "", []):
            errors.append(f"{label} is required.")
            continue
        ctx[spec["ctx_key"]] = parsed
    return errors


def _render_workbook_field_editor(*, title: str, specs: list[dict], key: str, caption: str | None = None) -> None:
    ctx = st.session_state.ctx
    if caption:
        st.caption(caption)
    editor_df = st.data_editor(
        _build_workbook_field_df(ctx, specs),
        key=key,
        hide_index=True,
        use_container_width=True,
        disabled=["field", "source", "status", "notes"],
        column_config={
            "field": st.column_config.TextColumn("field"),
            "value": st.column_config.TextColumn("value"),
            "source": st.column_config.TextColumn("source"),
            "status": st.column_config.TextColumn("status"),
            "notes": st.column_config.TextColumn("notes"),
        },
    )
    errors = _apply_workbook_field_df(ctx, editor_df, specs)
    for error in errors:
        st.warning(error)


def _render_workbook_component_row(*, component_label: str, key: str, base_row: dict | None = None, saved_vde_id: int | None = None, tires_df=None) -> None:
    ctx = st.session_state.ctx
    full_df = _build_component_spreadsheet_df(ctx)
    filtered_df = full_df.loc[full_df["component"] == component_label].copy()
    edited_df = st.data_editor(
        filtered_df,
        key=key,
        hide_index=True,
        use_container_width=True,
        disabled=["component", "status", "notes"],
        column_config={
            "component": st.column_config.TextColumn("component"),
            "source": st.column_config.TextColumn("source"),
            "A": st.column_config.NumberColumn(f"A [{unit_label('force')}]", format="%.6f"),
            "B": st.column_config.NumberColumn(f"B [{unit_label('force_per_speed')}]", format="%.6f"),
            "C": st.column_config.NumberColumn(f"C [{unit_label('force_per_speed_squared')}]", format="%.6f"),
            "apply": st.column_config.CheckboxColumn("apply"),
            "status": st.column_config.TextColumn("status"),
            "notes": st.column_config.TextColumn("notes"),
        },
    )
    if not edited_df.empty:
        full_df.loc[full_df["component"] == component_label, edited_df.columns] = edited_df.iloc[0].values
    errors = _apply_component_spreadsheet_changes(full_df)
    for error in errors:
        if component_label in error or component_label == "Tires":
            st.warning(error)
    if component_label == "Tires":
        _render_component_spreadsheet_tire_db_block(base_row=base_row, saved_vde_id=saved_vde_id, tires_df=tires_df)


def _component_spreadsheet_apply_value(ctx: dict, component: str) -> bool:
    key = f"spreadsheet_component_{component}_apply"
    stored = ctx.get(key)
    if isinstance(stored, bool):
        return stored
    if component == "tires":
        source = str(ctx.get("tire_component_source") or "Manual RR").strip()
        if source == "Tire DB":
            return bool(ctx.get("include_tire_component")) and bool(ctx.get("tire_preview_result"))
        return any(abs(float(to_float(ctx.get(name), 0.0) or 0.0)) > 0.0 for name in ("rr_alpha_N", "rr_beta_Npkph"))
    if component == "brakes":
        return any(abs(float(to_float(ctx.get(name), 0.0) or 0.0)) > 0.0 for name in ("brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"))
    if component == "parasitics":
        return any(abs(float(to_float(ctx.get(name), 0.0) or 0.0)) > 0.0 for name in ("parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"))
    return False


def _resolved_tire_spreadsheet_values(ctx: dict) -> tuple[float, float, float]:
    source = str(ctx.get("tire_component_source") or "Manual RR").strip()
    if source == "Tire DB":
        preview = dict(ctx.get("tire_preview_result") or {})
        component = dict(preview.get("component_dict") or {})
        return (
            float(to_float(component.get("A"), 0.0) or 0.0),
            float(to_float(component.get("B"), 0.0) or 0.0),
            float(to_float(component.get("C"), 0.0) or 0.0),
        )
    return (
        float(to_float(ctx.get("rr_alpha_N"), 0.0) or 0.0),
        float(to_float(ctx.get("rr_beta_Npkph"), 0.0) or 0.0),
        0.0,
    )


def _build_component_spreadsheet_df(ctx: dict) -> pd.DataFrame:
    tire_a, tire_b, tire_c = _resolved_tire_spreadsheet_values(ctx)
    return pd.DataFrame(
        [
            {
                "component": "Tires",
                "source": str(ctx.get("tire_component_source") or "Manual RR").strip() or "Manual RR",
                "A": to_display(tire_a, "force", _current_unit_system()),
                "B": to_display(tire_b, "force_per_speed", _current_unit_system()),
                "C": to_display(tire_c, "force_per_speed_squared", _current_unit_system()),
                "apply": _component_spreadsheet_apply_value(ctx, "tires"),
                "status": "OK" if _component_spreadsheet_apply_value(ctx, "tires") else "Not used",
                "notes": "Tire rolling resistance contribution",
            },
            {
                "component": "Brakes",
                "source": "Manual",
                "A": to_display(to_float(ctx.get("brake_A_coef_N")), "force", _current_unit_system()),
                "B": to_display(to_float(ctx.get("brake_B_Npkph")), "force_per_speed", _current_unit_system()),
                "C": to_display(to_float(ctx.get("brake_C_coef_Npkph2")), "force_per_speed_squared", _current_unit_system()),
                "apply": _component_spreadsheet_apply_value(ctx, "brakes"),
                "status": "OK" if _component_spreadsheet_apply_value(ctx, "brakes") else "Not used",
                "notes": "Brake drag contribution",
            },
            {
                "component": "Parasitics / Hubs / Axle",
                "source": "Manual",
                "A": to_display(to_float(ctx.get("parasitic_A_coef_N")), "force", _current_unit_system()),
                "B": to_display(to_float(ctx.get("parasitic_B_Npkph")), "force_per_speed", _current_unit_system()),
                "C": to_display(to_float(ctx.get("parasitic_C_coef_Npkph2")), "force_per_speed_squared", _current_unit_system()),
                "apply": _component_spreadsheet_apply_value(ctx, "parasitics"),
                "status": "OK" if _component_spreadsheet_apply_value(ctx, "parasitics") else "Not used",
                "notes": "Accessory, hub and axle losses",
            },
            {
                "component": "Trailer",
                "source": "Reserved",
                "A": to_display(0.0, "force", _current_unit_system()),
                "B": to_display(0.0, "force_per_speed", _current_unit_system()),
                "C": to_display(0.0, "force_per_speed_squared", _current_unit_system()),
                "apply": False,
                "status": "Not used",
                "notes": "Reserved placeholder",
            },
        ]
    )


def _set_component_spreadsheet_disabled_state(ctx: dict, component_key: str) -> None:
    if component_key == "tires":
        ctx["component_mode_tires"] = "Spreadsheet excluded"
        ctx["include_tire_component"] = False
    elif component_key == "brakes":
        ctx["component_mode_brakes"] = "Spreadsheet excluded"
    elif component_key == "parasitics":
        ctx["component_mode_parasitics_hubs_axle"] = "Spreadsheet excluded"


def _apply_component_spreadsheet_changes(editor_df: pd.DataFrame) -> list[str]:
    ctx = st.session_state.ctx
    errors: list[str] = []
    if editor_df is None or editor_df.empty:
        errors.append("Component spreadsheet is empty.")
        ctx["spreadsheet_component_errors"] = errors
        return errors

    rows = {
        str(row.get("component") or "").strip(): row
        for row in editor_df.to_dict(orient="records")
    }

    tire_row = dict(rows.get("Tires") or {})
    tire_source = str(tire_row.get("source") or "Manual RR").strip()
    if tire_source not in {"Manual RR", "Tire DB"}:
        errors.append("Tires source must be Manual RR or Tire DB.")
        tire_source = "Manual RR"
    tire_apply = bool(tire_row.get("apply"))
    ctx["spreadsheet_component_tires_apply"] = tire_apply
    ctx["tire_component_source"] = tire_source
    if tire_source == "Manual RR":
        tire_a = _editor_canonical_or_none(tire_row.get("A"), "force")
        tire_b = _editor_canonical_or_none(tire_row.get("B"), "force_per_speed")
        tire_c = _editor_canonical_or_none(tire_row.get("C"), "force_per_speed_squared")
        if tire_apply:
            if tire_a is None or tire_a < 0.0:
                errors.append("Tires A must be a valid non-negative number when Manual RR is applied.")
            if tire_b is None:
                errors.append("Tires B must be a valid number when Manual RR is applied.")
            if tire_c is None:
                errors.append("Tires C must be a valid number when Manual RR is applied.")
        if tire_a is not None:
            ctx["rr_alpha_N"] = float(tire_a)
        if tire_b is not None:
            ctx["rr_beta_Npkph"] = float(tire_b)
        if tire_apply:
            ctx["component_mode_tires"] = "Replace / manual input"
            ctx["include_tire_component"] = False
            if abs(float(to_float(tire_c, 0.0) or 0.0)) > 1e-9:
                errors.append("Tires C is not used for Manual RR in the current workflow. Keep it at zero.")
        else:
            _set_component_spreadsheet_disabled_state(ctx, "tires")
    else:
        if tire_apply and not ctx.get("tire_preview_result"):
            errors.append("Preview a Tire DB component before applying Tires from Tire DB.")
        ctx["component_mode_tires"] = "Lookup from DB" if tire_apply else "Spreadsheet excluded"
        ctx["include_tire_component"] = bool(tire_apply and ctx.get("tire_preview_result"))

    component_specs = [
        ("Brakes", "brakes", ("brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"), "component_mode_brakes"),
        ("Parasitics / Hubs / Axle", "parasitics", ("parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"), "component_mode_parasitics_hubs_axle"),
    ]
    for label, slug, field_names, mode_key in component_specs:
        row = dict(rows.get(label) or {})
        apply_flag = bool(row.get("apply"))
        ctx[f"spreadsheet_component_{slug}_apply"] = apply_flag
        a_value = _editor_canonical_or_none(row.get("A"), "force")
        b_value = _editor_canonical_or_none(row.get("B"), "force_per_speed")
        c_value = _editor_canonical_or_none(row.get("C"), "force_per_speed_squared")
        if a_value is not None:
            ctx[field_names[0]] = float(a_value)
        if b_value is not None:
            ctx[field_names[1]] = float(b_value)
        if c_value is not None:
            ctx[field_names[2]] = float(c_value)
        if apply_flag:
            if a_value is None:
                errors.append(f"{label} A must be a valid number when applied.")
            if b_value is None:
                errors.append(f"{label} B must be a valid number when applied.")
            if c_value is None:
                errors.append(f"{label} C must be a valid number when applied.")
            ctx[mode_key] = "Replace / manual input"
        else:
            _set_component_spreadsheet_disabled_state(ctx, slug)

    ctx["spreadsheet_component_errors"] = errors
    return errors


def _render_component_spreadsheet_tire_db_block(*, base_row: dict | None = None, saved_vde_id: int | None = None, tires_df=None) -> None:
    ctx = st.session_state.ctx
    if str(ctx.get("tire_component_source") or "Manual RR").strip() != "Tire DB":
        return
    with st.expander("Tire DB selector / preview", expanded=False):
        st.caption("Use the existing tire preview flow to resolve the tire component, then the spreadsheet row will consume that preview as the active Tires source.")
        render_tire_component_section(
            base_row=base_row,
            saved_vde_id=saved_vde_id,
            tires_df=tires_df,
            source_mode_override="Tire DB",
            show_source_selector=False,
        )



def _mm_display(value) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "Pending"
    if _current_unit_system() == "US customary":
        return f"{numeric / 25.4:.2f} in"
    return f"{numeric:.0f} mm"


def _tire_pressure_display_unit() -> str:
    ctx = st.session_state.ctx
    unit = str(ctx.get("tire_pressure_display_unit") or "psi").strip()
    return unit if unit in {"psi", "kPa"} else "psi"


def _format_tire_pressure(value_psi) -> str:
    numeric = to_float(value_psi)
    if numeric is None:
        return "Pending"
    unit = _tire_pressure_display_unit()
    display_value = numeric if unit == "psi" else numeric * TIRE_KPA_PER_PSI
    return f"{display_value:.1f} {unit}"


def _render_tire_pressure_unit_toggle(*, key: str = "tire_pressure_display_unit_toggle"):
    ctx = st.session_state.ctx
    current = _tire_pressure_display_unit()
    ctx["tire_pressure_display_unit"] = st.radio(
        "Tire pressure unit",
        ["psi", "kPa"],
        horizontal=True,
        index=(0 if current == "psi" else 1),
        key=key,
    )
    return ctx["tire_pressure_display_unit"]


def _render_tire_pressure_input(container, label: str, value_psi, *, key_base: str, default_psi: float = 32.0) -> float:
    canonical_psi = to_float(value_psi, default_psi)
    unit = _tire_pressure_display_unit()
    factor = 1.0 if unit == "psi" else TIRE_KPA_PER_PSI
    display_value = float(canonical_psi or 0.0) * factor
    display_input = container.number_input(
        f"{label} [{unit}]",
        value=float(display_value),
        step=0.5 if unit == "psi" else 5.0,
        format="%.1f",
        key=f"{key_base}_{unit.lower()}",
    )
    return float(display_input) / factor


def _distance_display_km(value) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "Pending"
    if _current_unit_system() == "US customary":
        return f"{numeric * 0.621371192237:.0f} mi"
    return f"{numeric:.0f} km"


def _tire_size_reference_details(size_code: str | None) -> dict:
    size_code = _string_or_none(size_code)
    if not size_code:
        return {}
    try:
        return dict(lookup_tire_size_reference(size_code) or {})
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_tire_size_reference_df():
    return load_tire_size_reference()


def _tire_size_options() -> list[str]:
    try:
        df = _load_tire_size_reference_df()
    except Exception:
        return []
    if df.empty or "size_code" not in df.columns:
        return []
    return sorted(df["size_code"].dropna().astype(str).unique().tolist())


def _render_tire_size_selector(*, field_key: str, label: str = "Tire size") -> str:
    ctx = st.session_state.ctx
    size_options = _tire_size_options()
    current_value = str(ctx.get(field_key) or "").strip()
    select_key = f"{field_key}_select"
    custom_key = f"{field_key}_custom"

    if size_options:
        select_options = [""] + size_options
        selected_option = current_value if current_value in size_options else ""
        custom_default = current_value if current_value and current_value not in size_options else str(ctx.get(custom_key) or "")
        c1, c2 = st.columns([1.3, 1.0])
        chosen = c1.selectbox(
            label,
            select_options,
            index=select_options.index(selected_option),
            key=select_key,
        )
        custom_value = c2.text_input(
            "Custom size code",
            value=custom_default,
            key=custom_key,
        )
        resolved = str(custom_value or chosen or "").strip()
    else:
        resolved = st.text_input(
            label,
            value=current_value,
            key=f"{field_key}_text",
        ).strip()
    ctx[field_key] = resolved
    return resolved


def _prefill_circumference_from_size(*, size_field_key: str, circumference_field_key: str) -> float | None:
    ctx = st.session_state.ctx
    size_code = _string_or_none(ctx.get(size_field_key))
    if not size_code:
        return to_float(ctx.get(circumference_field_key))
    size_ref = _tire_size_reference_details(size_code)
    target_mm = to_float(size_ref.get("expected_effective_circumference_mm"))
    if target_mm is None:
        target_mm = to_float(size_ref.get("unloaded_circumference_mm"))
    if target_mm is None or target_mm <= 0:
        return to_float(ctx.get(circumference_field_key))

    previous_size = _string_or_none(ctx.get(f"{circumference_field_key}_prefill_size"))
    previous_value = to_float(ctx.get(f"{circumference_field_key}_prefill_value"))
    current_value = to_float(ctx.get(circumference_field_key))
    should_prefill = (
        current_value is None
        or abs(float(current_value or 0.0)) <= 0.0
        or (
            size_code != previous_size
            and previous_value is not None
            and current_value is not None
            and abs(float(current_value) - float(previous_value)) <= 1e-6
        )
        or (size_code != previous_size and previous_size is None)
    )
    if should_prefill:
        ctx[circumference_field_key] = float(target_mm)
    ctx[f"{circumference_field_key}_prefill_size"] = size_code
    ctx[f"{circumference_field_key}_prefill_value"] = float(target_mm)
    return to_float(ctx.get(circumference_field_key))


def _tire_circumference_mm(reference: dict | None) -> float | None:
    data = dict(reference or {})
    explicit = to_float(data.get("effective_circumference_override_mm"))
    if explicit is not None and explicit > 0:
        return explicit
    size_ref = _tire_size_reference_details(data.get("size_code"))
    for key in ("expected_effective_circumference_mm", "unloaded_circumference_mm"):
        numeric = to_float(size_ref.get(key))
        if numeric is not None and numeric > 0:
            return numeric
    return None


def _reference_has_nonzero_abc(reference: dict | None) -> bool:
    data = dict(reference or {})
    return any(abs(float(to_float(data.get(key), 0.0) or 0.0)) > 0.0 for key in ("A", "B", "C"))


def _build_tire_reference(
    *,
    source: str,
    abc: dict | None = None,
    front_tire: dict | None = None,
    rear_tire: dict | None = None,
    front_pressure_psi=None,
    rear_pressure_psi=None,
    tire_load_mass_basis=None,
    tire_load_mass_used_kg=None,
    tire_calc_source=None,
    tire_calc_notes=None,
    extra: dict | None = None,
) -> dict:
    ref = {
        "source": source,
        "A": float(to_float((abc or {}).get("A"), 0.0) or 0.0),
        "B": float(to_float((abc or {}).get("B"), 0.0) or 0.0),
        "C": float(to_float((abc or {}).get("C"), 0.0) or 0.0),
        "front_tire": dict(front_tire or {}),
        "rear_tire": dict(rear_tire or {}),
        "front_tire_id": (front_tire or {}).get("id"),
        "rear_tire_id": (rear_tire or {}).get("id"),
        "front_pressure_psi": to_float(front_pressure_psi),
        "rear_pressure_psi": to_float(rear_pressure_psi),
        "tire_load_mass_basis": tire_load_mass_basis,
        "tire_load_mass_used_kg": to_float(tire_load_mass_used_kg),
        "tire_calc_source": tire_calc_source,
        "tire_calc_notes": tire_calc_notes,
    }
    if front_tire:
        ref["tire_test_code"] = _string_or_none(front_tire.get("tire_test_code"))
        ref["manufacturer"] = _string_or_none(front_tire.get("manufacturer"))
        ref["model"] = _string_or_none(front_tire.get("model"))
        ref["size_code"] = _string_or_none(front_tire.get("size_code"))
        ref["load_index"] = _string_or_none(front_tire.get("load_index"))
        ref["speed_rating"] = _string_or_none(front_tire.get("speed_rating"))
        ref["standard_family"] = _string_or_none(front_tire.get("standard_family"))
        ref["rr_n_per_kn"] = to_float(front_tire.get("rr_n_per_kn"))
        ref["smerf"] = to_float(front_tire.get("smerf"))
        ref["test_mileage_km"] = to_float(front_tire.get("test_mileage_km"))
        ref["test_method"] = _string_or_none(front_tire.get("test_method"))
        ref["test_source"] = _string_or_none(front_tire.get("test_source"))
        ref["is_tested_value"] = _truthy_flag(front_tire.get("is_tested_value"))
        ref["effective_circumference_override_mm"] = to_float(front_tire.get("effective_circumference_override_mm"))
    if extra:
        ref.update({k: v for k, v in extra.items() if v not in (None, "")})
    ref["circumference_mm"] = _tire_circumference_mm(ref)
    return ref


def _manual_tire_reference_from_ctx(prefix: str, *, source_label: str) -> dict | None:
    ctx = st.session_state.ctx
    manual_basis = str(ctx.get(f"{prefix}_manual_basis") or "RRC-based reference").strip()
    user_notes = _string_or_none(ctx.get(f"{prefix}_notes"))
    load_kN = _resolved_tire_load_kN(ctx)
    rr_n_per_kn = to_float(ctx.get(f"{prefix}_rr_n_per_kn"))

    if manual_basis == "Direct ABC override":
        abc = {
            "A": to_float(ctx.get(f"{prefix}_A")),
            "B": to_float(ctx.get(f"{prefix}_B")),
            "C": to_float(ctx.get(f"{prefix}_C")),
        }
        if not any(value is not None and abs(float(value)) > 0.0 for value in abc.values()):
            return None
        equivalent_rr, epa_v_avg, epa_v2_avg = _equivalent_rr_from_abc(
            abc.get("A"),
            abc.get("B"),
            abc.get("C"),
            load_kN=load_kN,
        )
        rr_n_per_kn = equivalent_rr
        tire_calc_source = "scenario_manual_reference_abc_override"
        derived_note = None
        if equivalent_rr is not None:
            if epa_v_avg is not None and epa_v2_avg is not None:
                derived_note = (
                    f"Equivalent RRC derived from direct ABC using EPA weighted v_avg={epa_v_avg:.2f} kph, "
                    f"v2_avg={epa_v2_avg:.2f} kph^2 and load={load_kN:.3f} kN"
                )
            else:
                derived_note = f"Equivalent RRC derived from direct ABC using load={load_kN:.3f} kN"
        tire_calc_notes = (
            derived_note if not user_notes else f"{derived_note} | {user_notes}"
        ) if derived_note else user_notes
    else:
        if rr_n_per_kn is None or rr_n_per_kn <= 0.0:
            return None
        crr_frac = float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0)
        a_rr, b_rr, c_rr, epa_v_avg = _abc_from_final_rr_target(
            rr_n_per_kn,
            load_kN=load_kN,
            crr_frac_120=crr_frac,
        )
        abc = {"A": a_rr, "B": b_rr, "C": c_rr}
        tire_calc_source = "scenario_manual_reference_rrc"
        if epa_v_avg is not None:
            derived_note = (
                f"Derived from final RRC target using EPA v_avg={epa_v_avg:.2f} kph, "
                f"load={load_kN:.3f} kN and crr1@120={crr_frac:.5f}"
            )
        else:
            derived_note = (
                f"Derived from manual RRC using load={load_kN:.3f} kN and crr1@120={crr_frac:.5f}"
            )
        tire_calc_notes = derived_note if not user_notes else f"{derived_note} | {user_notes}"

    extra = {
        "tire_test_code": _string_or_none(ctx.get(f"{prefix}_tire_test_code")),
        "manufacturer": _string_or_none(ctx.get(f"{prefix}_manufacturer")),
        "model": _string_or_none(ctx.get(f"{prefix}_model")),
        "size_code": _string_or_none(ctx.get(f"{prefix}_size_code")),
        "standard_family": _string_or_none(ctx.get(f"{prefix}_standard_family")),
        "rr_n_per_kn": rr_n_per_kn,
        "smerf": to_float(ctx.get(f"{prefix}_smerf")),
        "test_mileage_km": to_float(ctx.get(f"{prefix}_test_mileage_km")),
        "test_method": _string_or_none(ctx.get(f"{prefix}_test_method")),
        "test_source": _string_or_none(ctx.get(f"{prefix}_test_source")),
        "is_tested_value": _truthy_flag(ctx.get(f"{prefix}_is_tested_value")),
        "effective_circumference_override_mm": to_float(ctx.get(f"{prefix}_effective_circumference_override_mm")),
        "front_pressure_psi": to_float(ctx.get(f"{prefix}_front_pressure_psi")),
        "rear_pressure_psi": to_float(ctx.get(f"{prefix}_rear_pressure_psi")),
        "tire_calc_source": tire_calc_source,
        "tire_calc_notes": tire_calc_notes,
    }
    return _build_tire_reference(
        source=source_label,
        abc=abc,
        front_pressure_psi=extra.pop("front_pressure_psi"),
        rear_pressure_psi=extra.pop("rear_pressure_psi"),
        tire_calc_source=extra.pop("tire_calc_source"),
        tire_calc_notes=extra.pop("tire_calc_notes"),
        extra=extra,
    )


def _tire_provenance_note(reference: dict | None) -> str:
    data = dict(reference or {})
    parts = []
    for key in ("source", "tire_test_code", "manufacturer", "model", "size_code", "standard_family"):
        value = _string_or_none(data.get(key))
        if value:
            parts.append(f"{key}={value}")
    rr_value = to_float(data.get("rr_n_per_kn"))
    if rr_value is not None:
        parts.append(f"rr_n_per_kn={rr_value:.3f}")
    if data.get("tire_calc_notes"):
        parts.append(str(data.get("tire_calc_notes")))
    return "; ".join(parts)

def _compact_abc(values: dict | None) -> str:
    data = dict(values or {})
    return (
        f"{format_quantity(data.get('A'), 'force', include_unit=False, unavailable='0.00', format_str='%.2f')} / "
        f"{format_quantity(data.get('B'), 'force_per_speed', include_unit=False, unavailable='0.00000', format_str='%.5f')} / "
        f"{format_quantity(data.get('C'), 'force_per_speed_squared', include_unit=False, unavailable='0.000000', format_str='%.6f')}"
    )


def _format_energy_value(value, *, unavailable: str = "Unavailable") -> str:
    return format_quantity(value, "energy_per_distance", unavailable=unavailable)


def _component_build_up_enabled(ctx: dict | None = None) -> bool:
    data = dict(ctx or st.session_state.get("ctx", {}))
    source_ui = str(data.get("abc_total_source_ui") or "").strip()
    if source_ui:
        return source_ui == "Component Build-up"
    payload = build_vde_workflow_payload_from_ctx(data)
    return str(payload.get("initial_abc_total_source") or "").strip().upper() == "COMPONENT_BUILD_UP"


def _roadload_basis_value(ctx: dict | None = None) -> str:
    data = dict(ctx or st.session_state.get("ctx", {}))
    source_ui = str(data.get("abc_total_source_ui") or "").strip()
    if source_ui:
        return source_ui
    mode = str(data.get("mode") or "").strip()
    if mode == "From baseline (editable)":
        return "Baseline ABC"
    return "Component Build-up"


def _component_mode_key(component_label: str) -> str:
    slug = (
        component_label.lower()
        .replace(" / ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"component_mode_{slug}"


def _component_mode_options(component_label: str, basis: str) -> tuple[list[str], str]:
    build_mode = basis == "Component Build-up"
    if build_mode:
        if component_label == "Tires":
            return ["Replace / manual input", "Lookup from DB"], "Replace / manual input"
        return ["Replace / manual input"], "Replace / manual input"
    return ["Keep inherited", "Apply delta"], "Keep inherited"


def _component_mode_default(component_label: str, basis: str, ctx: dict) -> str:
    options, fallback = _component_mode_options(component_label, basis)
    key = _component_mode_key(component_label)
    current = str(ctx.get(key) or "").strip()
    if current in options:
        return current

    if basis != "Component Build-up":
        if component_label == "Tires" and abs(float(to_float(ctx.get("delta_rr_N"), 0.0) or 0.0)) > 0.0:
            return "Apply delta"
        if component_label == "Aerodynamics" and abs(float(to_float(ctx.get("delta_aero_cdA"), 0.0) or 0.0)) > 0.0:
            return "Apply delta"
        if component_label == "Brakes" and abs(float(to_float(ctx.get("delta_brake_N"), 0.0) or 0.0)) > 0.0:
            return "Apply delta"
        if component_label == "Parasitics / Hubs / Axle" and abs(float(to_float(ctx.get("delta_parasitics_N"), 0.0) or 0.0)) > 0.0:
            return "Apply delta"
    elif component_label == "Tires" and str(ctx.get("tire_component_source") or "").strip() == "Tire DB":
        return "Lookup from DB"

    return fallback


def _resolve_component_mode(component_label: str, ctx: dict | None = None) -> str:
    data = dict(ctx or st.session_state.get("ctx", {}))
    basis = _roadload_basis_value(data)
    key = _component_mode_key(component_label)
    resolved = _component_mode_default(component_label, basis, data)
    if ctx is None:
        st.session_state.ctx[key] = resolved
    return resolved


def _component_overview_rows(ctx: dict) -> list[dict]:
    basis = _roadload_basis_value(ctx)
    active = str(ctx.get("component_editor_active") or "Tires")
    tire_mode = _resolve_component_mode("Tires", ctx)
    brake_mode = _resolve_component_mode("Brakes", ctx)
    parasitic_mode = _resolve_component_mode("Parasitics / Hubs / Axle", ctx)
    selected_baseline_row = dict(ctx.get("selected_baseline_row") or {})
    baseline_mode = str(ctx.get("mode") or "") == "From baseline (editable)"
    baseline_tire_reference = _saved_tire_reference_from_row(selected_baseline_row)
    scenario_current_reference = _resolve_tire_reference_from_ctx(
        "tire_current_reference",
        source_label="Scenario Current reference",
        preview_ctx_key="tire_current_reference_preview_result",
    )
    current_tire_reference = scenario_current_reference or baseline_tire_reference
    walked_tire_reference = _resolve_tire_reference_from_ctx(
        "tire_walked_reference",
        source_label="Scenario Walked reference",
        preview_ctx_key="tire_walked_reference_preview_result",
    )
    tire_application = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))

    if basis == "Component Build-up":
        tire_reference = "Absolute source"
        tire_active_change = "Absolute tire source" if (
            (str(ctx.get("tire_component_source") or "").strip() == "Tire DB" and ctx.get("tire_preview_result"))
            or any(abs(float(to_float(ctx.get(key), 0.0) or 0.0)) > 0.0 for key in ("rr_alpha_N", "rr_beta_Npkph"))
        ) else "Excluded"
        tire_status = "Applied" if tire_active_change == "Absolute tire source" else "Pending"
    else:
        if current_tire_reference:
            if baseline_tire_reference and not scenario_current_reference and not any(
                ctx.get(key) not in (None, "", 0, 0.0) for key in ("tire_current_reference_A", "tire_current_reference_B", "tire_current_reference_C")
            ):
                tire_reference = "Inherited current"
            else:
                tire_reference = "Scenario current"
        elif baseline_mode:
            tire_reference = "Missing"
        else:
            tire_reference = "Optional"

        if tire_mode == "Keep inherited" or tire_application == "Keep inherited":
            tire_active_change = "Inherited"
            tire_status = "OK"
        elif tire_application == "Manual Delta RR":
            manual_delta_rr = float(to_float(ctx.get("tire_manual_delta_rr_n_per_kn"), 0.0) or 0.0)
            tire_active_change = f"Manual delta final RRC ({format_quantity(manual_delta_rr, 'rrc', format_str='%.2f')})"
            tire_status = "Applied" if abs(float(to_float(ctx.get("delta_rr_N"), 0.0) or 0.0)) > 0.0 else "Pending"
        elif tire_application == "Tire Improvement %":
            improvement_pct = float(to_float(ctx.get("tire_improvement_pct"), 0.0) or 0.0)
            tire_active_change = f"Tire Improvement ({improvement_pct:.1f}%)"
            if improvement_pct == 0.0:
                tire_status = "Pending"
            elif current_tire_reference:
                tire_status = "Applied"
            else:
                tire_status = "Pending current"
        else:
            tire_active_change = "Walked comparison"
            if walked_tire_reference and current_tire_reference and abs(float(to_float(ctx.get("delta_rr_N"), 0.0) or 0.0)) > 0.0:
                tire_status = "Applied"
            elif walked_tire_reference and not current_tire_reference:
                tire_status = "Pending current"
            elif not walked_tire_reference:
                tire_status = "Pending walked"
            else:
                tire_status = "Pending"

    rows = [
        {
            "Component": "Tires",
            "Reference": tire_reference,
            "Active change": tire_active_change,
            "Status": tire_status,
            "Action": "Edit below" if active == "Tires" else "Configure below",
        }
    ]

    component_specs = [
        ("Brakes", brake_mode, "delta_brake_N", "Absolute brake input"),
        ("Parasitics / Hubs / Axle", parasitic_mode, "delta_parasitics_N", "Absolute loss input"),
    ]
    for label, mode, delta_key, absolute_text in component_specs:
        if basis == "Component Build-up":
            reference = "Absolute source"
            if label == "Brakes":
                configured = any(abs(float(to_float(ctx.get(key), 0.0) or 0.0)) > 0.0 for key in ("brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"))
            else:
                configured = any(abs(float(to_float(ctx.get(key), 0.0) or 0.0)) > 0.0 for key in ("parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"))
            active_change = "Absolute value" if configured else "Excluded"
            status = "Applied" if configured else "Pending"
        else:
            if label == "Brakes":
                has_reference = any(to_float(selected_baseline_row.get(key)) is not None for key in ("brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2")) or not baseline_mode
                calc_mode = str(ctx.get("brake_calculation_mode") or "Inherited")
            else:
                has_reference = any(to_float(selected_baseline_row.get(key)) is not None for key in ("parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2")) or not baseline_mode
                calc_mode = str(ctx.get("parasitic_calculation_mode") or "Inherited")
            reference = "Inherited" if has_reference else "Missing"
            if calc_mode == "Use candidate vs reference":
                active_change = "Derived delta"
                status = "Applied"
            elif calc_mode == "Manual delta A" and abs(float(to_float(ctx.get(delta_key), 0.0) or 0.0)) > 0.0:
                active_change = "Manual delta"
                status = "Applied"
            elif mode == "Keep inherited":
                active_change = "Inherited"
                status = "OK"
            else:
                active_change = "Excluded"
                status = "Reference missing" if reference == "Missing" else "Pending"
        rows.append(
            {
                "Component": label,
                "Reference": reference,
                "Active change": active_change,
                "Status": status,
                "Action": "Edit below" if active == label else "Configure below",
            }
        )

    rows.append(
        {
            "Component": "Trailer",
            "Reference": "Placeholder",
            "Active change": "Excluded",
            "Status": "Placeholder",
            "Action": "Reserved",
        }
    )
    return rows


def _render_component_overview_table(ctx: dict):
    rows = _component_overview_rows(ctx)
    icon_map = {
        "Tires": "[T]",
        "Brakes": "[B]",
        "Parasitics / Hubs / Axle": "[P]",
        "Trailer": "[ ]",
    }
    st.caption("Components Overview")
    active = str(ctx.get("component_editor_active") or "Tires")
    for row in rows:
        component = str(row.get("Component") or "")
        icon = icon_map.get(component, "[?]")
        c1, c2, c3, c4 = st.columns([1.25, 1.15, 1.2, 0.9])
        is_active = component == active
        button_label = f"{icon} {component}"
        if c1.button(
            button_label,
            key=f"component_overview_open_{component}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            ctx["component_editor_active"] = component
            st.rerun()
        c2.caption("Reference")
        c2.write(str(row.get("Reference") or "-"))
        c3.caption("Active change")
        c3.write(str(row.get("Active change") or "-"))
        c4.caption("Status")
        c4.write(str(row.get("Status") or "-"))
    st.caption("Reference = current/base state. Active change = what is actually affecting the scenario now.")
    st.caption("Transmission stays outside this grid because it acts on the TOTAL -> NET bridge, but it should be read with the same discipline: clear reference, explicit application, and no silent effect on calculation.")


def _render_tire_editor_block_header(title: str, caption: str):
    st.markdown(f"**{title}**")
    st.caption(caption)


def _render_applied_effect_block(*, title: str, caption: str, inherited_message: str, metrics: list[tuple[str, str]] | None = None):
    with st.container(border=True):
        _render_tire_editor_block_header(title, caption)
        if not metrics:
            st.info(inherited_message)
            return
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)


def _render_scalar_reference_metric(*, label: str, value, quantity: str, status: str):
    st.caption(f"Reference status: `{status}`")
    quantity_metric(st, label, value, quantity, format_str="%.4f")


def _render_abc_reference_metrics(*, title: str, a_value, b_value, c_value, status: str, caption: str | None = None):
    st.caption(f"{title} | status: `{status}`")
    c1, c2, c3 = st.columns(3)
    quantity_metric(c1, "A", a_value, "force", format_str="%.3f")
    quantity_metric(c2, "B", b_value, "force_per_speed", format_str="%.6f")
    quantity_metric(c3, "C", c_value, "force_per_speed_squared", format_str="%.8f")
    if caption:
        st.caption(caption)


def _render_candidate_cda_input(*, key_prefix: str, title: str) -> float | None:
    ctx = st.session_state.ctx
    st.caption(title)
    value = quantity_input(
        st,
        "Candidate CdA",
        to_float(ctx.get(f"{key_prefix}_cda"), 0.0),
        "cda",
        key=f"{key_prefix}_cda_input",
        step_canonical=0.01,
        format_str="%.3f",
    )
    ctx[f"{key_prefix}_cda"] = value
    if abs(float(value)) <= 0.0:
        st.caption("Candidate still pending. Enter a proposed CdA to stage it.")
        return None
    return float(value)


def _render_candidate_abc_inputs(*, key_prefix: str, title: str) -> dict | None:
    ctx = st.session_state.ctx
    st.caption(title)
    c1, c2, c3 = st.columns(3)
    a_val = quantity_input(
        c1,
        "Candidate A",
        to_float(ctx.get(f"{key_prefix}_A"), 0.0),
        "force",
        key=f"{key_prefix}_A_input",
        step_canonical=0.1,
        format_str="%.2f",
    )
    b_val = quantity_input(
        c2,
        "Candidate B",
        to_float(ctx.get(f"{key_prefix}_B"), 0.0),
        "force_per_speed",
        key=f"{key_prefix}_B_input",
        step_canonical=0.001,
        format_str="%.5f",
    )
    c_val = quantity_input(
        c3,
        "Candidate C",
        to_float(ctx.get(f"{key_prefix}_C"), 0.0),
        "force_per_speed_squared",
        key=f"{key_prefix}_C_input",
        step_canonical=0.0001,
        format_str="%.6f",
    )
    ctx[f"{key_prefix}_A"] = a_val
    ctx[f"{key_prefix}_B"] = b_val
    ctx[f"{key_prefix}_C"] = c_val
    if abs(float(a_val)) <= 0.0 and abs(float(b_val)) <= 0.0 and abs(float(c_val)) <= 0.0:
        st.caption("Candidate still pending. Enter at least one non-zero ABC value to stage it.")
        return None
    return {"A": float(a_val), "B": float(b_val), "C": float(c_val)}


def _transmission_component_row(ctx: dict, prefill: dict | None = None) -> dict:
    base = dict(prefill or {})
    source = str(ctx.get("transmission_losses_source") or ("Baseline" if base.get("trans_A_coef_N") is not None else "Missing")).strip().title()
    if source == "Baseline":
        reference = "Inherited"
        calc_now = "Applied to NET"
        status = "Applied"
    elif source == "Manual":
        reference = "Manual"
        calc_now = "Applied to NET"
        status = "Applied" if any(abs(float(to_float(ctx.get(key), 0.0) or 0.0)) > 0.0 for key in ("trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2")) else "Pending"
    else:
        reference = "Missing"
        calc_now = "Excluded"
        status = "Pending"
    return {
        "Component": "Transmission / Neutral Drag",
        "Reference": reference,
        "Candidate": "N/A",
        "Calculation now": calc_now,
        "Status": status,
    }


def _safe_workflow_preview(ctx: dict | None = None) -> dict:
    data = dict(ctx or st.session_state.get("ctx", {}))
    try:
        return dict(build_vde_setup_preview_from_ctx(data) or {})
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _line_source_summary(ctx: dict) -> str:
    mode = str(ctx.get("mode") or "").strip()
    if mode == "From baseline (editable)":
        baseline_id = ctx.get("vde_id_parent") or ctx.get("baseline_id")
        return f"Baseline #{baseline_id}" if baseline_id else "Baseline flow"
    if mode == "New line (manual / test)":
        return "New line"
    return "-"


def _total_source_summary(ctx: dict, preview: dict | None = None) -> str:
    data = dict(preview or {})
    line_source = dict(data.get("line_source") or {})
    source_ui = str(ctx.get("abc_total_source_ui") or "").strip()
    mode = str(ctx.get("mode") or "").strip()
    from_delta = str(ctx.get("from_delta") or "").strip()
    if source_ui:
        return source_ui
    if mode == "From baseline (editable)" and from_delta == "Deltas":
        baseline_id = ctx.get("vde_id_parent") or ctx.get("baseline_id")
        return f"Baseline ABC_TOTAL #{baseline_id}" if baseline_id else "Baseline ABC_TOTAL"
    if _component_build_up_enabled(ctx):
        return "Component Build-up"
    return str(line_source.get("mode") or "Manual inputs")


def _inertia_class_table() -> list[dict]:
    csv_path = Path(__file__).resolve().parents[3] / "data" / "standards" / "inertia_classes_by_mass.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "mass_min_kg_exclusive": to_float(row.get("mass_min_kg_exclusive")),
                "mass_max_kg_inclusive": to_float(row.get("mass_max_kg_inclusive")),
                "inertia_class_kg": to_float(row.get("inertia_class_kg")),
            }
        )
    return rows


def _inertia_step_for_mass(mass_kg: float | None) -> tuple[int | None, dict | None]:
    if mass_kg is None:
        return None, None
    rows = _inertia_class_table()
    for idx, row in enumerate(rows):
        lo = row.get("mass_min_kg_exclusive")
        hi = row.get("mass_max_kg_inclusive")
        if lo is None and hi is not None and mass_kg <= hi:
            return idx, row
        if hi is None and lo is not None and mass_kg > lo:
            return idx, row
        if lo is not None and hi is not None and mass_kg > lo and mass_kg <= hi:
            return idx, row
    return None, None


def _mass_point_for_inertia_step(step: dict | None, point: str) -> float | None:
    if not step:
        return None
    lo = step.get("mass_min_kg_exclusive")
    hi = step.get("mass_max_kg_inclusive")
    if hi is None:
        return None
    low = float(lo + 1.0) if lo is not None else float(hi)
    top = float(hi)
    mid = float(round((low + top) / 2.0, 1))
    if point == "low":
        return low
    if point == "mid":
        return mid
    if point == "top":
        return top
    return None


def _relative_twc_mass_targets(baseline_mass_kg: float | None) -> list[dict]:
    idx, _ = _inertia_step_for_mass(baseline_mass_kg)
    rows = _inertia_class_table()
    if idx is None or not rows:
        return []
    targets = []
    for offset, prefix in [(-1, "-1"), (0, "Current"), (1, "+1")]:
        target_idx = idx + offset
        if target_idx < 0 or target_idx >= len(rows):
            continue
        step = rows[target_idx]
        twc_value = step.get("inertia_class_kg")
        for point, suffix in [("low", "Low"), ("mid", "Mid"), ("top", "Top")]:
            target_mass = _mass_point_for_inertia_step(step, point)
            if target_mass is None:
                continue
            targets.append(
                {
                    "label": f"{prefix} TWC {suffix}",
                    "mass_kg": float(target_mass),
                    "twc_kg": float(twc_value) if twc_value is not None else None,
                }
            )
    return targets


def _relative_twc_target(baseline_mass_kg: float | None, class_offset: int, point: str) -> dict | None:
    idx, _ = _inertia_step_for_mass(baseline_mass_kg)
    rows = _inertia_class_table()
    if idx is None or not rows:
        return None
    target_idx = idx + int(class_offset or 0)
    if target_idx < 0 or target_idx >= len(rows):
        return None
    step = rows[target_idx]
    target_mass = _mass_point_for_inertia_step(step, point)
    if target_mass is None:
        return None
    prefix = "Current" if class_offset == 0 else f"{class_offset:+d}"
    return {
        "label": f"{prefix} TWC {str(point).title()}",
        "mass_kg": float(target_mass),
        "twc_kg": float(step.get("inertia_class_kg")) if step.get("inertia_class_kg") is not None else None,
        "target_index": target_idx,
    }


def _mass_setup_summary(preview: dict | None = None, ctx: dict | None = None) -> str:
    data = dict(preview or {})
    mass_setup = dict(data.get("mass_setup") or {})
    source_ctx = dict(ctx or st.session_state.get("ctx", {}))
    basis = str(mass_setup.get("mass_basis") or source_ctx.get("tire_load_mass_basis") or "TEST_MASS")
    resolved = to_float(mass_setup.get("resolved_mass_used_kg"), to_float(source_ctx.get("mass_kg")))
    if resolved is None:
        return basis
    return f"{basis} | {format_quantity(resolved, 'mass', include_unit=True, format_str='%.0f')}"


def _abc_unit_triplet_label() -> str:
    return (
        f"{unit_label('force')} / "
        f"{unit_label('force_per_speed')} / "
        f"{unit_label('force_per_speed_squared')}"
    )


def _transmission_summary(preview: dict | None = None, ctx: dict | None = None) -> str:
    data = dict(preview or {})
    transmission = dict(data.get("transmission_losses") or {})
    source_ctx = dict(ctx or st.session_state.get("ctx", {}))
    status = str(transmission.get("status") or "").strip().lower()
    source = str(source_ctx.get("transmission_losses_source") or "Missing").strip()
    if status == "available":
        return f"{source} | NET available"
    return f"{source} | NET pending"


def _summary_status_payload(state: str) -> dict:
    normalized = str(state or "pending").strip().lower()
    if normalized in {"ok", "ready", "derived", "defined"}:
        return {"label": "OK", "class_name": "is-ok", "icon_html": "&#10003;"}
    if normalized in {"partial"}:
        return {"label": "Partial", "class_name": "is-pending", "icon_html": "&#9679;"}
    if normalized in {"warn", "review", "missing", "not ready"}:
        label = "Missing" if normalized == "missing" else "Review"
        return {"label": label, "class_name": "is-warn", "icon_html": "&#33;"}
    if normalized in {"not used", "not_used"}:
        return {"label": "Not used", "class_name": "is-neutral", "icon_html": "&#9675;"}
    return {"label": "Pending", "class_name": "is-pending", "icon_html": "&#9679;"}


def _overview_icon(label: str) -> str:
    icons = {
        "Vehicle Data": "&#128663;",
        "Scenario Origin": "&#9873;",
        "ABC_TOTAL Basis": "&#8776;",
        "Mass": "&#9878;",
        "Transmission": "&#8644;",
        "Cycle": "&#10227;",
        "VDE_TOTAL": "&#9312;",
        "VDE_NET": "&#9313;",
    }
    return icons.get(label, "&#9679;")


def _status_label_for_display(state: str) -> str:
    normalized = str(state or "pending").strip().lower()
    mapping = {
        "ok": "OK",
        "ready": "Ready",
        "derived": "Derived",
        "defined": "Defined",
        "partial": "Partial",
        "review": "Review",
        "missing": "Missing",
        "not used": "Not used",
        "not_used": "Not used",
        "pending": "Pending",
    }
    return mapping.get(normalized, str(state or "Pending"))


def _render_status_bar_item(label: str, status: str, detail: str = "") -> None:
    payload = _summary_status_payload(status)
    st.markdown(
        (
            f"<div class='vde-status-chip {payload['class_name']}'>"
            f"<div class='vde-status-label'>{html.escape(label)}</div>"
            f"<div class='vde-summary-status {payload['class_name']}'>"
            f"<span class='vde-summary-status-icon'>{payload['icon_html']}</span>"
            f"<span>{html.escape(_status_label_for_display(status))}</span>"
            "</div>"
            f"<div class='vde-status-detail'>{html.escape(detail)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _normalize_tire_scenario_application(value: str | None) -> str:
    current = str(value or "").strip()
    mapping = {
        "Apply manual delta A": "Manual Delta RR",
        "Apply derived comparison": "Walked tire comparison",
        "Apply Tire Improvement %": "Tire Improvement %",
        "Stage candidate only / do not apply": "Keep inherited",
    }
    normalized = mapping.get(current, current)
    if normalized not in {
        "Keep inherited",
        "Manual Delta RR",
        "Tire Improvement %",
        "Walked tire comparison",
    }:
        return "Keep inherited"
    return normalized


def _display_tire_application_method(value: str | None) -> str:
    current = str(value or "").strip()
    mapping = {
        "Manual Delta RR": "Manual delta final RRC",
        "Tire Improvement %": "Tire Improvement %",
        "Walked tire comparison": "Walked tire comparison",
        "Keep inherited": "Inherited",
    }
    return mapping.get(current, current or "Inherited")


def _display_data_role(value: str | None) -> str:
    current = str(value or "").strip()
    mapping = {
        "engineering_target": "Engineering target",
        "scenario_assumption": "Scenario assumption",
    }
    return mapping.get(current, current or "-")


def _tire_reference_brief(reference: dict | None) -> str:
    data = dict(reference or {})
    identity = " ".join(
        part for part in (
            _string_or_none(data.get("manufacturer")),
            _string_or_none(data.get("model")),
        ) if part
    ) or _string_or_none(data.get("tire_test_code")) or "Scenario reference"
    size_code = _string_or_none(data.get("size_code"))
    rr_text = format_quantity(data.get("rr_n_per_kn"), "rrc", unavailable="Pending", format_str="%.3f")
    if size_code:
        return f"{identity} | {size_code} | {rr_text}"
    return f"{identity} | {rr_text}"


def _default_tire_change_method(ctx: dict) -> str:
    explicit = str(ctx.get("tire_change_method") or "").strip()
    if explicit in {"Manual tire adjustment", "Walked tire comparison"}:
        return explicit
    application = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))
    if application == "Walked tire comparison":
        return "Walked tire comparison"
    return "Manual tire adjustment"


def _default_tire_manual_input_type(ctx: dict) -> str:
    explicit = str(ctx.get("tire_manual_adjustment_input_type") or "").strip()
    if explicit in {"Delta RR", "Delta final RRC"}:
        return "Delta final RRC"
    if explicit in {"Target RRC", "Target final RRC"}:
        return "Target final RRC"
    if explicit == "Tire Improvement %":
        return explicit
    application = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))
    if application == "Tire Improvement %":
        return "Tire Improvement %"
    return "Delta final RRC"


def _resolved_tire_load_kN(ctx: dict) -> float:
    tire_mass_resolution = resolve_tire_calculation_mass(ctx)
    calc_mass_kg = to_float(tire_mass_resolution.get("mass_kg"))
    if calc_mass_kg is None:
        calc_mass_kg = to_float(ctx.get("mass_kg"), 0.0)
    return (float(calc_mass_kg or 0.0) * 9.80665) / 1000.0


def _resolve_tire_reference_from_ctx(
    prefix: str,
    *,
    source_label: str,
    preview_ctx_key: str | None = None,
) -> dict | None:
    ctx = st.session_state.ctx
    preview_key = preview_ctx_key or f"{prefix}_preview_result"
    preview_result = ctx.get(preview_key)
    if isinstance(preview_result, dict):
        reference = _reference_from_preview_result(preview_result, source_label=source_label)
        if reference:
            return reference
    return _manual_tire_reference_from_ctx(prefix, source_label=source_label)


def _ensure_vehicle_metadata_defaults(ctx: dict):
    leg_opts = ["WLTP", "EPA", "ABNT (Brazil)"]
    if ctx.get("legislation") not in leg_opts:
        ctx["legislation"] = "WLTP"

    epa_classes = [
        "Unknown", "Two Seaters", "Minicompact Cars", "Subcompact Cars", "Compact Cars",
        "Midsize Cars", "Large Cars", "Small Station Wagons", "Midsize Station Wagons",
        "Small SUVs", "Standard SUVs", "Minivans", "Vans", "Small Pickup Trucks", "Standard Pickup Trucks",
    ]
    wltp_classes = ["Class 1 (<850 kg)", "Class 2 (850-1220 kg)", "Class 3 (>1220 kg)"]
    category_list = epa_classes if ctx["legislation"] == "EPA" else wltp_classes
    category_list_upper = [category.upper() for category in category_list]
    if ctx.get("category") not in category_list_upper:
        ctx["category"] = category_list_upper[0]

    if not str(ctx.get("electrification") or "").strip():
        ctx["electrification"] = "ICE"
    if not str(ctx.get("transmission_type") or "").strip():
        ctx["transmission_type"] = "AT"
    if not str(ctx.get("year") or "").strip().isdigit():
        ctx["year"] = 2024


def _metadata_status(ctx: dict) -> dict:
    legislation = str(ctx.get("legislation") or "").strip()
    category = str(ctx.get("category") or "").strip()
    make = str(ctx.get("make") or "").strip()
    model = str(ctx.get("model") or "").strip()
    electrification = str(ctx.get("electrification") or "").strip()
    transmission_type = str(ctx.get("transmission_type") or "").strip()
    year_raw = ctx.get("year")

    missing = []
    if not legislation:
        missing.append("legislation")
    if not category or category.upper() == "UNKNOWN":
        missing.append("category")
    if not make:
        missing.append("make")
    if not model:
        missing.append("model")
    if not str(year_raw).strip().isdigit():
        missing.append("year")
    if not electrification:
        missing.append("electrification")
    if not transmission_type:
        missing.append("transmission")

    year_text = str(int(year_raw)) if str(year_raw).strip().isdigit() else "Year pending"
    headline_parts = [part for part in (make.upper(), model) if part]
    headline = " ".join(headline_parts).strip() or "Vehicle data pending"
    value = f"{headline} | {year_text}"
    detail = f"{legislation} | {category} | {electrification or '-'} | {transmission_type or '-'}"

    if missing:
        missing_text = ", ".join(missing[:3])
        if len(missing) > 3:
            missing_text += ", ..."
        return {
            "status": "pending",
            "value": value,
            "detail": f"{detail} | Missing: {missing_text}",
            "missing": missing,
        }

    return {
        "status": "ok",
        "value": value,
        "detail": detail,
        "missing": [],
    }


def _line_source_status(ctx: dict) -> str:
    mode = str(ctx.get("mode") or "").strip()
    if mode == "From baseline (editable)":
        return "ok" if (ctx.get("vde_id_parent") or ctx.get("baseline_id")) else "pending"
    if mode == "New line (manual / test)":
        return "ok"
    return "pending"


def _total_source_status(ctx: dict) -> str:
    source_ui = str(ctx.get("abc_total_source_ui") or "").strip()
    if source_ui == "Baseline ABC":
        return "ok" if (ctx.get("vde_id_parent") or ctx.get("baseline_id")) else "pending"
    if source_ui == "From test coastdown":
        errs, _ = validate_core(ctx.get("A"), ctx.get("B"), ctx.get("C"), ctx.get("mass_kg"))
        return "ok" if not errs else "pending"
    if source_ui == "Component Build-up":
        return "ok"
    return "pending"


def _mass_setup_status(preview: dict | None = None, ctx: dict | None = None) -> str:
    data = dict(preview or {})
    source_ctx = dict(ctx or st.session_state.get("ctx", {}))
    mass_setup = dict(data.get("mass_setup") or {})
    resolved_mass = to_float(mass_setup.get("resolved_mass_used_kg"), to_float(source_ctx.get("mass_kg")))
    weight_dist = to_float(source_ctx.get("weight_dist_fr_pct"))
    if resolved_mass is None or resolved_mass <= 0:
        return "pending"
    if weight_dist is None or weight_dist < 0 or weight_dist > 100:
        return "pending"
    return "ok"


def _cycle_status(ctx: dict) -> str:
    return "ok" if (ctx.get("cycle_df") is not None or str(ctx.get("cycle_name") or "").strip()) else "pending"


def _energy_status(value) -> str:
    return "ok" if value is not None else "pending"


def _summary_action_href(view: str, technical_view: str | None = None) -> str:
    parts = [f"vde_setup_view_target={quote_plus(view)}"]
    if technical_view:
        parts.append(f"technical_build_up_view_target={quote_plus(technical_view)}")
    return "?" + "&".join(parts)


def apply_summary_navigation_from_query_params():
    params = st.query_params
    target_view = params.get("vde_setup_view_target")
    target_technical = params.get("technical_build_up_view_target")

    if isinstance(target_view, list):
        target_view = target_view[0] if target_view else None
    if isinstance(target_technical, list):
        target_technical = target_technical[0] if target_technical else None

    target_view = str(target_view or "").strip()
    target_technical = str(target_technical or "").strip()
    if target_view == "Cycle & Results":
        target_view = "Results"
    if target_view == "Drive Cycle":
        target_view = "Cycle & Preview"
    if target_view == "Technical Build-up":
        target_view = "Roadload Build-up"
    if not target_view:
        return

    valid_views = {"Scenario Setup", "Vehicle Parameters", "Roadload Build-up", "Cycle & Preview", "Results", "Save / Edit"}
    valid_technical = {"Tires", "Brakes", "Parasitics / Hubs / Axle", "Trailer", "Transmission"}

    ctx = st.session_state.ctx
    if target_view in valid_views:
        ctx["vde_setup_view"] = target_view
        st.session_state["vde_setup_view_selector"] = target_view
    if target_technical in valid_technical:
        ctx["technical_build_up_view"] = target_technical
        st.session_state["technical_build_up_view_selector"] = target_technical

    for key in ("vde_setup_view_target", "technical_build_up_view_target"):
        try:
            del st.query_params[key]
        except Exception:
            pass


def _render_summary_chip(
    label: str,
    value: str,
    *,
    status: str = "pending",
    detail: str | None = None,
    action_view: str | None = None,
    action_technical_view: str | None = None,
):
    status_payload = _summary_status_payload(status)
    overview_icon = _overview_icon(label)
    detail_html = f"<div class='vde-summary-chip-detail'>{detail}</div>" if detail else ""
    open_link = ""
    close_link = ""
    if action_view:
        href = _summary_action_href(action_view, action_technical_view)
        open_link = f'<a class="vde-summary-link" href="{href}" target="_self">'
        close_link = "</a>"
    st.markdown(
        f"""
        {open_link}
        <div class="vde-summary-chip {status_payload['class_name']}">
            <div class="vde-summary-chip-top">
                <strong><span class="vde-summary-status-icon">{overview_icon}</span> {label}</strong>
                <span class="vde-summary-status {status_payload['class_name']}">
                    <span class="vde-summary-status-icon">{status_payload['icon_html']}</span>
                    <span>{status_payload['label']}</span>
                </span>
            </div>
            <span>{value}</span>
            {detail_html}
        </div>
        {close_link}
        """,
        unsafe_allow_html=True,
    )


def _render_component_mode_selector(component_label: str) -> str:
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    options, _ = _component_mode_options(component_label, basis)
    current = _component_mode_default(component_label, basis, ctx)
    key = _component_mode_key(component_label)
    if len(options) == 1:
        ctx[key] = options[0]
        st.caption(f"{component_label} mode: `{options[0]}`")
        return ctx[key]
    ctx[key] = st.radio(
        f"{component_label} mode",
        options,
        horizontal=True,
        index=options.index(current),
        key=f"{key}_radio",
    )
    return ctx[key]


def render_vde_setup_view_selector() -> str:
    ctx = st.session_state.ctx
    options = ["Scenario Setup", "Vehicle Parameters", "Roadload Build-up", "Cycle & Preview", "Results", "Save / Edit"]
    current = str(ctx.get("vde_setup_view") or "Scenario Setup")
    if current == "Cycle & Results":
        current = "Results"
    if current == "Drive Cycle":
        current = "Cycle & Preview"
    if current == "Technical Build-up":
        current = "Roadload Build-up"
    if current not in options:
        current = "Scenario Setup"
    ctx["vde_setup_view"] = st.radio(
        "View",
        options,
        horizontal=True,
        index=options.index(current),
        key="vde_setup_view_selector",
    )
    return ctx["vde_setup_view"]


def render_technical_build_up_view_selector() -> str:
    ctx = st.session_state.ctx
    options = ["Tires", "Brakes", "Parasitics / Hubs / Axle", "Trailer", "Transmission"]
    current = str(ctx.get("technical_build_up_view") or "Tires")
    if current in {"Mass & Axle Load", "Components"}:
        current = "Tires"
    if current not in options:
        current = "Tires"
    ctx["technical_build_up_view"] = st.radio(
        "Roadload section",
        options,
        horizontal=True,
        index=options.index(current),
        key="technical_build_up_view_selector",
    )
    return ctx["technical_build_up_view"]


def render_initial_abc_total_source_section(*, defaults_df_getter=None):
    ctx = st.session_state.ctx
    mode = str(ctx.get("mode") or "From baseline (editable)")
    options = (
        ["Baseline ABC", "From test coastdown", "Component Build-up"]
        if mode == "From baseline (editable)"
        else ["From test coastdown", "Component Build-up"]
    )
    default_value = str(ctx.get("abc_total_source_ui") or "").strip()
    if default_value not in options:
        default_value = options[0] if mode == "From baseline (editable)" else "Component Build-up"

    ctx["abc_total_source_ui"] = st.radio(
        "ABC_TOTAL basis",
        options,
        horizontal=True,
        index=options.index(default_value),
        key="abc_total_source_ui_radio",
        format_func=lambda value: {
            "Baseline ABC": "Inherit baseline ABC_TOTAL",
            "From test coastdown": "Measured/test coastdown ABC_TOTAL",
            "Component Build-up": "Build/synthesize ABC_TOTAL from components",
        }.get(value, value),
    )

    selected = ctx["abc_total_source_ui"]
    if selected == "Baseline ABC":
        ctx["from_delta"] = "Deltas"
        baseline_id = ctx.get("vde_id_parent") or ctx.get("baseline_id")
        if not baseline_id:
            st.warning("Select a baseline scenario in Scenario Setup before using the inherited baseline ABC_TOTAL basis.")
    elif selected == "From test coastdown":
        ctx["from_delta"] = "From test"
        render_from_test_section()
        render_auxiliaries_section(defaults_df_getter=defaults_df_getter)
    else:
        ctx["from_delta"] = "Change Parameters"


def render_mass_setup_section(*, prefill=None):
    ctx = st.session_state.ctx
    base = dict(prefill or {})
    mode = str(ctx.get("mode") or "")
    abc_total_source_ui = str(ctx.get("abc_total_source_ui") or "")
    baseline_inherited_mode = mode == "From baseline (editable)" and abc_total_source_ui == "Baseline ABC"
    regulatory_mode = _mass_regulatory_mode(ctx)

    mass_default = to_float(base.get("mass_kg"), to_float(ctx.get("mass_kg"), 1550.0))
    test_mass_prefill = to_float(base.get("test_mass_kg"), to_float(ctx.get("test_mass_kg")))
    weight_dist_default = to_float(base.get("weight_dist_fr_pct"), to_float(ctx.get("weight_dist_fr_pct"), 50.0))
    baseline_mass_reference = to_float(base.get("mass_kg"), mass_default)
    existing_delta_mass = float(to_float(ctx.get("delta_mass_kg"), 0.0) or 0.0)
    legislation = str(ctx.get("legislation") or "").strip().upper()
    scenario_mass_for_calcs = to_float(ctx.get("mass_kg"), mass_default)
    force_twc_basis = False

    if baseline_inherited_mode and baseline_mass_reference is not None:
        baseline_test_mass = resolve_test_mass_kg(
            {
                **ctx,
                "mass_kg": baseline_mass_reference,
                "test_mass_kg": to_float(base.get("test_mass_kg")),
            }
        )
        baseline_twc = resolve_tire_calculation_mass(
            {
                **ctx,
                "mass_kg": baseline_mass_reference,
                "test_mass_kg": to_float(base.get("test_mass_kg")),
                "tire_load_mass_basis": "TWC",
                "twc_kg": to_float(base.get("twc_kg")),
                "etw_kg": to_float(base.get("etw_kg")),
                "inertia_class": to_float(base.get("inertia_class")),
            }
        ).get("mass_kg")

        ctx["mass_kg"] = float(baseline_mass_reference)
        base1, base2, base3 = st.columns(3)
        quantity_metric(base1, "Baseline curb mass", baseline_mass_reference, "mass", format_str="%.1f")
        quantity_metric(base2, "Baseline test mass", baseline_test_mass, "mass", format_str="%.1f")
        quantity_metric(base3, "Baseline TWC", baseline_twc, "mass", format_str="%.1f")

        st.caption("Baseline mass snapshot is read-only here. Choose how this scenario changes mass relative to the inherited baseline.")

        mass_change_mode_options = ["Input new Curb / Test Mass", "Delta"]
        selected_mass_change_mode = str(ctx.get("mass_change_mode") or mass_change_mode_options[0])
        if selected_mass_change_mode not in mass_change_mode_options:
            selected_mass_change_mode = mass_change_mode_options[0]
        ctx["mass_change_mode"] = st.radio(
            "Mass scenario treatment",
            mass_change_mode_options,
            horizontal=True,
            index=mass_change_mode_options.index(selected_mass_change_mode),
            key="mass_change_mode_radio",
        )

        if ctx["mass_change_mode"] == "Input new Curb / Test Mass":
            new_curb_mass = quantity_input(
                st,
                "New curb weight",
                to_float(ctx.get("mass_walked_reference_kg"), scenario_mass_for_calcs),
                "mass",
                key="mass_setup_new_curb_weight",
                min_canonical=300.0,
                max_canonical=5000.0,
                step_canonical=1.0,
                format_str="%.1f",
            )
            scenario_mass_for_calcs = float(new_curb_mass)
            ctx["delta_mass_kg"] = float(scenario_mass_for_calcs) - float(baseline_mass_reference)

            tm1, tm2 = st.columns([1.2, 1.2])
            default_new_test_mass = resolve_test_mass_kg(
                {
                    **ctx,
                    "mass_kg": scenario_mass_for_calcs,
                    "test_mass_kg": None,
                }
            )
            use_default = tm1.checkbox("Use default test mass", value=bool(ctx.get("test_mass_use_default", True)), key="mass_setup_inherited_use_default_test_mass")
            ctx["test_mass_use_default"] = use_default
            if use_default:
                ctx["test_mass_kg"] = None
                quantity_metric(tm2, "Scenario test mass", default_new_test_mass, "mass", format_str="%.1f")
            else:
                scenario_test_mass = quantity_input(
                    tm2,
                    "New test mass",
                    max(to_float(ctx.get("test_mass_kg"), default_new_test_mass or scenario_mass_for_calcs), scenario_mass_for_calcs),
                    "mass",
                    key="mass_setup_new_test_mass",
                    min_canonical=float(scenario_mass_for_calcs),
                    max_canonical=5000.0,
                    step_canonical=1.0,
                    format_str="%.1f",
                )
                ctx["test_mass_kg"] = scenario_test_mass
        else:
            delta_mode_options = ["TWC class target", "Manual delta"]
            selected_delta_mode = str(ctx.get("mass_delta_mode") or delta_mode_options[0])
            if selected_delta_mode not in delta_mode_options:
                selected_delta_mode = delta_mode_options[0]
            ctx["mass_delta_mode"] = st.radio(
                "Delta mode",
                delta_mode_options,
                horizontal=True,
                index=delta_mode_options.index(selected_delta_mode),
                key="mass_delta_mode_radio",
            )

            if ctx["mass_delta_mode"] == "TWC class target":
                base_idx, _ = _inertia_step_for_mass(baseline_mass_reference)
                rows = _inertia_class_table()
                if base_idx is not None and rows:
                    min_offset = -base_idx
                    max_offset = len(rows) - 1 - base_idx
                    offset_col, point_col = st.columns([1, 1.2])
                    class_offset = int(
                        offset_col.number_input(
                            "TWC class offset",
                            min_value=int(min_offset),
                            max_value=int(max_offset),
                            value=int(ctx.get("mass_delta_class_offset", 0) or 0),
                            step=1,
                            key="mass_delta_class_offset_input",
                        )
                    )
                    point_options = ["Low", "Mid", "Top"]
                    selected_point = str(ctx.get("mass_delta_class_point") or "Mid").title()
                    if selected_point not in point_options:
                        selected_point = "Mid"
                    point_value = point_col.selectbox(
                        "Target within class",
                        point_options,
                        index=point_options.index(selected_point),
                        key="mass_delta_class_point_select",
                    )
                    ctx["mass_delta_class_offset"] = class_offset
                    ctx["mass_delta_class_point"] = point_value
                    selected_target = _relative_twc_target(baseline_mass_reference, class_offset, point_value.lower())
                    if selected_target:
                        force_twc_basis = True
                        scenario_mass_for_calcs = float(selected_target["mass_kg"])
                        ctx["delta_mass_kg"] = scenario_mass_for_calcs - float(baseline_mass_reference)
                        ctx["test_mass_kg"] = None
                        ctx["test_mass_use_default"] = True
                        ctx["twc_kg"] = selected_target.get("twc_kg")
                        st.caption(f"Resolved target: `{selected_target['label']}`")
                        adopted_test_mass = resolve_test_mass_kg(
                            {
                                **ctx,
                                "mass_kg": scenario_mass_for_calcs,
                                "test_mass_kg": None,
                            }
                        )
                        adopted1, adopted2, adopted3 = st.columns(3)
                        quantity_metric(adopted1, "Target TWC", selected_target.get("twc_kg"), "mass", format_str="%.1f")
                        quantity_metric(adopted2, "Adopted curb mass", scenario_mass_for_calcs, "mass", format_str="%.1f")
                        quantity_metric(adopted3, "Adopted test mass", adopted_test_mass, "mass", format_str="%.1f")
                        st.caption("This delta-class path auto-adopts the selected TWC target and keeps the VDE calculation mass basis on TWC.")
                    else:
                        st.warning("Could not resolve the selected TWC class target.")
                else:
                    st.warning("Could not resolve TWC class targets from the inertia-class table.")
                    scenario_mass_for_calcs = float(baseline_mass_reference + existing_delta_mass)
            else:
                manual_delta_mass = quantity_input(
                    st,
                    "Delta Mass",
                    existing_delta_mass,
                    "mass",
                    key="mass_setup_delta_mass_kg",
                    step_canonical=1.0,
                    format_str="%.1f",
                )
                ctx["delta_mass_kg"] = float(manual_delta_mass)
                scenario_mass_for_calcs = float(baseline_mass_reference) + float(manual_delta_mass)
                ctx["test_mass_kg"] = None
                ctx["test_mass_use_default"] = True

        ctx["mass_walked_reference_kg"] = scenario_mass_for_calcs
    else:
        ctx["mass_kg"] = quantity_input(
            st,
            "Curb weight",
            to_float(mass_default, 1550.0),
            "mass",
            key="mass_setup_curb_weight",
            min_canonical=300.0,
            max_canonical=5000.0,
            step_canonical=1.0,
            format_str="%.1f",
        )
        scenario_mass_for_calcs = float(ctx["mass_kg"])
        ctx["delta_mass_kg"] = float(to_float(ctx.get("delta_mass_kg"), 0.0) or 0.0)

    payload_default = to_float(base.get("payload_kg"), to_float(ctx.get("payload_kg"), 0.0))
    options_default = to_float(base.get("options_kg"), to_float(ctx.get("options_kg"), 0.0))
    wltp_category_default = str(base.get("wltp_category") or ctx.get("wltp_category") or "").strip().upper()

    input_cols = st.columns(3 if regulatory_mode == "WLTP" else 2)
    ctx["payload_kg"] = quantity_input(
        input_cols[0],
        "Payload",
        to_float(payload_default, 0.0),
        "mass",
        key="mass_setup_payload",
        min_canonical=0.0,
        max_canonical=5000.0,
        step_canonical=1.0,
        format_str="%.1f",
    )
    ctx["options_kg"] = quantity_input(
        input_cols[1],
        "Optional equipment mass",
        to_float(options_default, 0.0),
        "mass",
        key="mass_setup_options",
        min_canonical=0.0,
        max_canonical=1000.0,
        step_canonical=1.0,
        format_str="%.1f",
    )
    if regulatory_mode == "WLTP":
        wltp_category_options = ["", "M1", "M2", "N1", "N2"]
        if wltp_category_default not in wltp_category_options:
            wltp_category_options.append(wltp_category_default)
        ctx["wltp_category"] = input_cols[2].selectbox(
            "WLTP category",
            wltp_category_options,
            index=wltp_category_options.index(wltp_category_default if wltp_category_default in wltp_category_options else ""),
            key="mass_setup_wltp_category",
        )

    row1c1, row1c2, row1c3 = st.columns(3)
    row1c1.metric("Legislation", str(ctx.get("legislation") or "-"))
    ctx["weight_dist_fr_pct"] = row1c2.number_input(
        "Front weight distribution [%]",
        min_value=0.0,
        max_value=100.0,
        value=float(weight_dist_default or 50.0),
        step=0.5,
        format="%.1f",
        key="mass_setup_weight_dist",
    )

    current_basis = resolve_tire_load_mass_basis(
        {
            "legislation": ctx.get("legislation"),
            "tire_load_mass_basis": ctx.get("tire_load_mass_basis") or base.get("tire_load_mass_basis"),
        }
    )
    if legislation == "EPA":
        if force_twc_basis:
            ctx["tire_load_mass_basis"] = "TWC"
            row1c3.metric("VDE calculation mass", "TWC")
        else:
            ctx["tire_load_mass_basis"] = row1c3.selectbox(
                "VDE calculation mass",
                ["TWC", "TEST_MASS"],
                index=["TWC", "TEST_MASS"].index(current_basis if current_basis in {"TWC", "TEST_MASS"} else "TWC"),
                key="mass_setup_tire_basis",
            )
    else:
        ctx["tire_load_mass_basis"] = "TEST_MASS"
        row1c3.metric("VDE calculation mass", "TEST_MASS")

    test_mass_context = {**ctx, "mass_kg": scenario_mass_for_calcs}
    initial_test_mass_state = resolve_test_mass_state(test_mass_context)
    ctx["test_mass_low_kg"] = initial_test_mass_state.get("test_mass_low_kg")
    ctx["test_mass_high_kg"] = initial_test_mass_state.get("test_mass_high_kg")

    derived_laden_mass = initial_test_mass_state.get("laden_mass_kg")
    if regulatory_mode == "WLTP":
        row2c1, row2c2, row2c3 = st.columns(3)
        quantity_metric(row2c1, "Derived laden mass", derived_laden_mass, "mass", format_str="%.1f")
        quantity_metric(row2c2, "WLTP Test Mass Low", initial_test_mass_state.get("test_mass_low_kg"), "mass", format_str="%.1f")
        quantity_metric(row2c3, "WLTP Test Mass High", initial_test_mass_state.get("test_mass_high_kg"), "mass", format_str="%.1f")
    elif regulatory_mode == "EPA":
        row2c1, row2c2, row2c3 = st.columns(3)
        quantity_metric(row2c1, "Resolved test mass", initial_test_mass_state.get("test_mass_kg"), "mass", format_str="%.1f")
        quantity_metric(row2c2, "EPA inertia / TWC mass", initial_test_mass_state.get("test_mass_kg") if ctx.get("tire_load_mass_basis") == "TWC" else None, "mass", format_str="%.1f")
        quantity_metric(row2c3, "Baseline test mass", test_mass_prefill, "mass", format_str="%.1f")
    else:
        row2c1, row2c2 = st.columns(2)
        quantity_metric(row2c1, "Resolved test mass", initial_test_mass_state.get("test_mass_kg"), "mass", format_str="%.1f")
        quantity_metric(row2c2, "Derived laden mass", derived_laden_mass, "mass", format_str="%.1f")
    scope_warning = initial_test_mass_state.get("light_duty_scope_warning")
    if scope_warning:
        st.warning(scope_warning, icon=":material/warning:")

    if regulatory_mode == "WLTP":
        test_mass_basis_options = ["WLTP_TMH", "WLTP_TML", "GVWR", "GCWR_TRAILER", "CURB_PLUS_DRIVER", "CURB", "PHYSICAL_TEST_MASS", "CUSTOM"]
    elif regulatory_mode == "EPA":
        test_mass_basis_options = ["EPA_INERTIA_CLASS", "GVWR", "GCWR_TRAILER", "CURB_PLUS_DRIVER", "CURB", "PHYSICAL_TEST_MASS", "CUSTOM"]
    else:
        test_mass_basis_options = ["GVWR", "GCWR_TRAILER", "CUSTOM", "PHYSICAL_TEST_MASS", "CURB"]
    selected_test_mass_basis = str(
        ctx.get("test_mass_basis")
        or initial_test_mass_state.get("test_mass_basis")
        or ""
    ).strip().upper()
    if selected_test_mass_basis not in test_mass_basis_options:
        if regulatory_mode == "WLTP" and initial_test_mass_state.get("test_mass_high_kg") is not None:
            selected_test_mass_basis = "WLTP_TMH"
        elif regulatory_mode == "EPA" and ctx.get("tire_load_mass_basis") == "TWC":
            selected_test_mass_basis = "EPA_INERTIA_CLASS"
        elif regulatory_mode == "CUSTOM":
            selected_test_mass_basis = "CUSTOM"
        else:
            selected_test_mass_basis = "CURB"

    selector1, selector2 = st.columns([1.2, 1.2])
    ctx["test_mass_basis"] = selector1.selectbox(
        "Test mass used for calculation",
        test_mass_basis_options,
        index=test_mass_basis_options.index(selected_test_mass_basis),
        key="mass_setup_test_mass_basis",
    )

    manual_test_mass_input = None
    if ctx["test_mass_basis"] in {"CUSTOM", "PHYSICAL_TEST_MASS"}:
        manual_default = max(
            to_float(ctx.get("test_mass_kg"), initial_test_mass_state.get("test_mass_kg") or scenario_mass_for_calcs or 0.0),
            scenario_mass_for_calcs or 0.0,
        )
        manual_test_mass_input = quantity_input(
            selector2,
            "Manual test mass",
            manual_default,
            "mass",
            key="mass_setup_manual_test_mass",
            min_canonical=float(scenario_mass_for_calcs or 0.0),
            max_canonical=5000.0,
            step_canonical=1.0,
            format_str="%.1f",
        )
        ctx["test_mass_kg"] = manual_test_mass_input
    else:
        ctx["test_mass_kg"] = None
        selector2.caption(build_test_mass_hint({**ctx, "legislation": legislation}))

    final_test_mass_state = resolve_test_mass_state(
        {
            **ctx,
            "mass_kg": scenario_mass_for_calcs,
            "test_mass_basis": ctx.get("test_mass_basis"),
            "test_mass_kg": ctx.get("test_mass_kg"),
        }
    )
    ctx["test_mass_kg"] = final_test_mass_state.get("test_mass_kg")
    ctx["test_mass_low_kg"] = final_test_mass_state.get("test_mass_low_kg")
    ctx["test_mass_high_kg"] = final_test_mass_state.get("test_mass_high_kg")
    ctx["test_mass_basis"] = final_test_mass_state.get("test_mass_basis")
    ctx["test_mass_use_default"] = ctx.get("test_mass_basis") not in {"CUSTOM", "PHYSICAL_TEST_MASS"}

    tire_mass_resolution = resolve_tire_calculation_mass({**ctx, "mass_kg": scenario_mass_for_calcs})
    calc_mass_kg = tire_mass_resolution.get("mass_kg")
    if legislation == "EPA" and ctx.get("tire_load_mass_basis") == "TWC":
        ctx["inertia_class"] = calc_mass_kg
        ctx["twc_kg"] = calc_mass_kg

    st.info(
        " | ".join(
            [
                "Mass OK",
                f"Legislation {regulatory_mode}",
                f"Basis {str(ctx.get('test_mass_basis') or '-')}",
                f"Resolved VDE mass {format_quantity(calc_mass_kg, 'mass', format_str='%.0f', unavailable='-')}",
                f"Roadload basis {str(ctx.get('tire_load_mass_basis') or 'TEST_MASS')}",
            ]
        )
    )

    row3c1, row3c2, row3c3, row3c4 = st.columns(4)
    quantity_metric(row3c1, "Resolved calc mass", calc_mass_kg, "mass", format_str="%.1f")
    row3c2.metric("Roadload basis", str(ctx.get("tire_load_mass_basis") or "TEST_MASS"))
    row3c3.metric("Test mass basis", str(ctx.get("test_mass_basis") or "-"))
    row3c4.metric("Weight distribution", f"{float(ctx.get('weight_dist_fr_pct') or 50.0):.1f}%")
    st.caption("Mass setup is centralized here so tire, preview, and transmission sections can reuse the same resolved vehicle state.")


def render_executive_summary_panel():
    ctx = st.session_state.ctx
    _ensure_vehicle_metadata_defaults(ctx)
    preview = _safe_workflow_preview(ctx)
    vde_total = dict(preview.get("vde_total") or {})
    vde_net = dict(preview.get("vde_net") or {})
    metadata = _metadata_status(ctx)
    transmission_status = str(dict(preview.get("transmission_losses") or {}).get("status") or "").strip().lower()
    component_active = _component_build_up_enabled(ctx)
    save_status = "ready" if preview.get("ok") and vde_total.get("mj_per_km") is not None and not _spreadsheet_validation_errors(ctx) else "pending"

    items = [
        {
            "label": "Vehicle",
            "detail": metadata["value"],
            "status": metadata["status"],
            "action_view": "Scenario Setup",
        },
        {
            "label": "Mass",
            "detail": _mass_setup_summary(preview, ctx),
            "status": _mass_setup_status(preview, ctx),
            "action_view": "Vehicle Parameters",
        },
        {
            "label": "Cycle",
            "detail": str(ctx.get("cycle_name") or "Standard / pending"),
            "status": _cycle_status(ctx),
            "action_view": "Cycle & Preview",
        },
        {
            "label": "ABC_TOTAL",
            "detail": _total_source_summary(ctx, preview),
            "status": _total_source_status(ctx),
            "action_view": "Roadload Build-up",
        },
        {
            "label": "Trans Loss",
            "detail": _transmission_summary(preview, ctx),
            "status": "ok" if transmission_status == "available" else "review",
            "action_view": "Roadload Build-up",
            "action_technical_view": "Transmission",
        },
        {
            "label": "ABC_NET",
            "detail": format_quantity(vde_net.get("mj_per_km"), "energy_per_distance", unavailable="Pending", format_str="%.3f"),
            "status": "derived" if vde_net.get("mj_per_km") is not None else "pending",
            "action_view": "Results",
        },
        {
            "label": "Components",
            "detail": "Component build-up active" if component_active else "Not applied to ABC_TOTAL",
            "status": "ok" if component_active else "not used",
            "action_view": "Roadload Build-up",
        },
        {
            "label": "Save",
            "detail": format_quantity(vde_total.get("mj_per_km"), "energy_per_distance", unavailable="Preview pending", format_str="%.3f"),
            "status": save_status,
            "action_view": "Save / Edit",
        },
    ]

    st.caption("VDE Status Bar")
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            _render_status_bar_item(
                item["label"],
                item.get("status", "pending"),
                detail=item.get("detail", ""),
            )

    warnings = list(preview.get("warnings") or [])
    spreadsheet_errors = _spreadsheet_validation_errors(ctx)
    if warnings or spreadsheet_errors:
        with st.expander("Status details", expanded=False):
            for warning in warnings:
                st.warning(warning)
            for error in spreadsheet_errors:
                st.warning(error)


def render_vde_calculation_context_header() -> None:
    ctx = st.session_state.ctx
    preview = _safe_workflow_preview(ctx)
    metadata = _metadata_status(ctx)
    vde_total = dict(preview.get("vde_total") or {})
    vde_net = dict(preview.get("vde_net") or {})
    net_text = "available" if vde_net.get("mj_per_km") is not None else "unavailable"
    save_text = "ready" if preview.get("ok") and vde_total.get("mj_per_km") is not None and not _spreadsheet_validation_errors(ctx) else "pending"
    items = [
        ("Vehicle", metadata["value"]),
        ("Cycle", str(ctx.get("cycle_name") or "standard / pending")),
        ("Mass basis", _mass_setup_summary(preview, ctx)),
        ("Roadload", _total_source_summary(ctx, preview)),
        ("NET", net_text),
        ("Save", save_text),
    ]
    body = "".join(
        (
            "<div class='vde-context-item'>"
            f"<div class='vde-context-label'>{html.escape(label)}</div>"
            f"<div class='vde-context-value'>{html.escape(str(value or '-'))}</div>"
            "</div>"
        )
        for label, value in items
    )
    st.markdown(f"<div class='vde-context-strip'>{body}</div>", unsafe_allow_html=True)
    st.caption("Used by analysts to feed vehicle demand. Program-facing comparison lives in Comparison Report.")


def render_executive_summary_panel_legacy():
    ctx = st.session_state.ctx
    _ensure_vehicle_metadata_defaults(ctx)
    preview = _safe_workflow_preview(ctx)
    vde_total = dict(preview.get("vde_total") or {})
    vde_net = dict(preview.get("vde_net") or {})
    metadata = _metadata_status(ctx)
    transmission_status = str(dict(preview.get("transmission_losses") or {}).get("status") or "").strip().lower()

    items = [
        {
            "label": "Vehicle Data",
            "value": metadata["value"],
            "detail": metadata["detail"],
            "status": metadata["status"],
            "action_view": "Scenario Setup",
        },
        {
            "label": "Scenario Origin",
            "value": _line_source_summary(ctx),
            "status": _line_source_status(ctx),
            "action_view": "Scenario Setup",
        },
        {
            "label": "ABC_TOTAL Basis",
            "value": _total_source_summary(ctx, preview),
            "status": _total_source_status(ctx),
            "action_view": "Roadload Build-up",
        },
        {
            "label": "Mass",
            "value": _mass_setup_summary(preview, ctx),
            "status": _mass_setup_status(preview, ctx),
            "action_view": "Vehicle Parameters",
        },
        {
            "label": "Transmission",
            "value": _transmission_summary(preview, ctx),
            "status": "ok" if transmission_status == "available" else "pending",
            "action_view": "Roadload Build-up",
            "action_technical_view": "Transmission",
        },
        {
            "label": "Cycle",
            "value": str(ctx.get("cycle_name") or "Standard / pending"),
            "status": _cycle_status(ctx),
            "action_view": "Cycle & Preview",
        },
        {
            "label": "VDE_TOTAL",
            "value": format_quantity(vde_total.get("mj_per_km"), "energy_per_distance", unavailable="Pending", format_str="%.3f"),
            "status": _energy_status(vde_total.get("mj_per_km")),
            "action_view": "Results",
        },
        {
            "label": "VDE_NET",
            "value": format_quantity(vde_net.get("mj_per_km"), "energy_per_distance", unavailable="Pending", format_str="%.3f"),
            "status": _energy_status(vde_net.get("mj_per_km")),
            "action_view": "Results",
        },
    ]

    st.caption("Executive Summary")
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            _render_summary_chip(
                item["label"],
                item["value"],
                status=item.get("status", "pending"),
                detail=item.get("detail"),
                action_view=item.get("action_view"),
                action_technical_view=item.get("action_technical_view"),
            )

    warnings = list(preview.get("warnings") or [])
    if warnings:
        st.warning("Current workflow warnings: " + ", ".join(warnings[:3]))


def _build_tire_context_row(base_row: dict | None = None) -> dict:
    ctx = st.session_state.ctx
    row = dict(base_row or {})
    for key in (
        "id",
        "legislation",
        "mass_kg",
        "test_mass_kg",
        "inertia_class",
        "twc_kg",
        "etw_kg",
        "weight_dist_fr_pct",
        "front_tire_id",
        "rear_tire_id",
        "front_pressure_psi",
        "rear_pressure_psi",
        "tire_improvement_pct",
        "tire_load_mass_basis",
        "tire_A_final",
        "tire_B_final",
        "tire_C_final",
    ):
        value = ctx.get(key)
        if value not in (None, ""):
            row[key] = value
    return row


def _saved_tire_reference_from_row(base_row: dict | None = None) -> dict | None:
    base = dict(base_row or {})
    abc = {
        "A": to_float(base.get("tire_A_final")),
        "B": to_float(base.get("tire_B_final")),
        "C": to_float(base.get("tire_C_final")),
    }
    if not any(value is not None for value in abc.values()):
        return None
    front_tire = {}
    rear_tire = {}
    if base.get("front_tire_id"):
        try:
            front_tire = get_tire_by_id(int(base.get("front_tire_id")))
        except Exception:
            front_tire = {}
    if base.get("rear_tire_id"):
        try:
            rear_tire = get_tire_by_id(int(base.get("rear_tire_id")))
        except Exception:
            rear_tire = {}
    return _build_tire_reference(
        source="Baseline saved tire reference",
        abc=abc,
        front_tire=front_tire,
        rear_tire=rear_tire or front_tire,
        front_pressure_psi=base.get("front_pressure_psi"),
        rear_pressure_psi=base.get("rear_pressure_psi"),
        tire_load_mass_basis=base.get("tire_load_mass_basis"),
        tire_load_mass_used_kg=base.get("tire_load_mass_used_kg"),
        tire_calc_source=base.get("tire_calc_source"),
        tire_calc_notes=base.get("tire_calc_notes"),
        extra={
            "size_code": base.get("tire_size"),
            "rr_n_per_kn": base.get("rrc_N_per_kN"),
        },
    )


def _reference_from_preview_result(preview_result: dict | None, *, source_label: str) -> dict | None:
    if not isinstance(preview_result, dict):
        return None
    payload = dict(preview_result.get("save_payload") or {})
    if not payload:
        return None
    return _build_tire_reference(
        source=source_label,
        abc={
            "A": payload.get("tire_A_final"),
            "B": payload.get("tire_B_final"),
            "C": payload.get("tire_C_final"),
        },
        front_tire=preview_result.get("front_tire"),
        rear_tire=preview_result.get("rear_tire"),
        front_pressure_psi=payload.get("front_pressure_psi"),
        rear_pressure_psi=payload.get("rear_pressure_psi"),
        tire_load_mass_basis=payload.get("tire_load_mass_basis"),
        tire_load_mass_used_kg=payload.get("tire_load_mass_used_kg"),
        tire_calc_source=payload.get("tire_calc_source"),
        tire_calc_notes=payload.get("tire_calc_notes"),
    )


def _render_tire_reference_metrics(reference: dict, *, title: str):
    st.caption(title)
    identity = " | ".join(
        part
        for part in (
            _string_or_none(reference.get("tire_test_code")),
            " ".join(part for part in (_string_or_none(reference.get("manufacturer")), _string_or_none(reference.get("model"))) if part),
            _string_or_none(reference.get("size_code")),
        )
        if part
    ) or "Scenario-only tire reference"
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Identity", identity)
    i2.metric("Standard", _string_or_none(reference.get("standard_family")) or "Scenario-only")
    i3.metric("RRC", format_quantity(reference.get("rr_n_per_kn"), "rrc", format_str="%.3f"))
    i4.metric("Circumference", _mm_display(reference.get("circumference_mm")))

    d1, d2, d3 = st.columns(3)
    d1.metric("Front pressure", _format_tire_pressure(reference.get("front_pressure_psi")))
    d2.metric("Rear pressure", _format_tire_pressure(reference.get("rear_pressure_psi")))
    d3.metric("Notes", "Available" if _string_or_none(reference.get("tire_calc_notes")) else "Optional")

    detail_parts = [str(reference.get("source") or "").strip()]
    if reference.get("tire_load_mass_basis"):
        detail_parts.append(f"basis={reference.get('tire_load_mass_basis')}")
    if reference.get("tire_load_mass_used_kg") is not None:
        detail_parts.append(
            "mass="
            + format_quantity(reference.get("tire_load_mass_used_kg"), "mass", format_str="%.1f")
        )
    if reference.get("test_method"):
        detail_parts.append(f"method={reference.get('test_method')}")
    if reference.get("test_source"):
        detail_parts.append(f"source={reference.get('test_source')}")
    if reference.get("tire_calc_source"):
        detail_parts.append(str(reference.get("tire_calc_source")))
    if detail_parts:
        st.caption(" | ".join(part for part in detail_parts if part))

    with st.expander("Roadload details", expanded=False):
        m1, m2, m3 = st.columns(3)
        quantity_metric(m1, "A", reference.get("A"), "force", format_str="%.3f")
        quantity_metric(m2, "B", reference.get("B"), "force_per_speed", format_str="%.6f")
        quantity_metric(m3, "C", reference.get("C"), "force_per_speed_squared", format_str="%.8f")
        if reference.get("tire_calc_notes"):
            st.caption(str(reference.get("tire_calc_notes")))


def _render_manual_tire_reference_inputs(*, prefix: str, title: str) -> dict | None:
    ctx = st.session_state.ctx
    st.caption(title)
    ctx[f"{prefix}_tire_test_code"] = ""
    ctx[f"{prefix}_manufacturer"] = ""
    ctx[f"{prefix}_model"] = ""

    size_col, note_col = st.columns([1.2, 1.8])
    with size_col:
        ctx[f"{prefix}_size_code"] = _render_tire_size_selector(
            field_key=f"{prefix}_size_code",
            label="Tire size",
        )
    ctx[f"{prefix}_notes"] = note_col.text_area(
        "Notes",
        value=str(ctx.get(f"{prefix}_notes") or ""),
        key=f"{prefix}_notes_input",
        height=68,
    )
    _prefill_circumference_from_size(
        size_field_key=f"{prefix}_size_code",
        circumference_field_key=f"{prefix}_effective_circumference_override_mm",
    )

    basis_options = ["RRC-based reference", "Direct ABC override"]
    current_basis = str(ctx.get(f"{prefix}_manual_basis") or basis_options[0]).strip()
    if current_basis not in basis_options:
        current_basis = basis_options[0]
    ctx[f"{prefix}_manual_basis"] = st.radio(
        "Roadload representation",
        basis_options,
        horizontal=True,
        index=basis_options.index(current_basis),
        key=f"{prefix}_manual_basis_radio",
    )

    n1, n2, n3, n4 = st.columns(4)
    ctx[f"{prefix}_standard_family"] = "CUSTOM"
    n1.caption("Reference family")
    n1.caption("Scenario-only")
    if ctx[f"{prefix}_manual_basis"] == "RRC-based reference":
        ctx[f"{prefix}_rr_n_per_kn"] = quantity_input(
            n2,
            "Final RRC",
            to_float(ctx.get(f"{prefix}_rr_n_per_kn"), 0.0),
            "rrc",
            key=f"{prefix}_rr_n_per_kn_input",
            step_canonical=0.05,
            format_str="%.3f",
        )
    else:
        n2.caption("Equivalent RRC")
        n2.caption("Will be derived from A after ABC is entered.")
    ctx[f"{prefix}_front_pressure_psi"] = _render_tire_pressure_input(
        n3,
        "Front pressure",
        to_float(ctx.get(f"{prefix}_front_pressure_psi"), 32.0),
        key_base=f"{prefix}_front_pressure_input",
    )
    ctx[f"{prefix}_rear_pressure_psi"] = _render_tire_pressure_input(
        n4,
        "Rear pressure",
        to_float(ctx.get(f"{prefix}_rear_pressure_psi"), 32.0),
        key_base=f"{prefix}_rear_pressure_input",
    )

    p1, p2, p3, p4 = st.columns(4)
    default_circumference_mm = _tire_circumference_mm({"size_code": ctx.get(f"{prefix}_size_code")})
    display_circumference = to_float(ctx.get(f"{prefix}_effective_circumference_override_mm"))
    if display_circumference is None:
        display_circumference = default_circumference_mm
    display_circumference = float((display_circumference or 0.0) / (25.4 if _current_unit_system() == "US customary" else 1.0))
    ctx[f"{prefix}_effective_circumference_override_mm"] = p1.number_input(
        f"Circumference [{('mm' if _current_unit_system() == 'Metric' else 'in')}]",
        value=display_circumference,
        step=1.0 if _current_unit_system() == "Metric" else 0.1,
        format="%.1f",
        key=f"{prefix}_effective_circumference_input",
    )
    if _current_unit_system() == "US customary":
        ctx[f"{prefix}_effective_circumference_override_mm"] = float(ctx[f"{prefix}_effective_circumference_override_mm"]) * 25.4
    ctx[f"{prefix}_test_mileage_km"] = 0.0
    ctx[f"{prefix}_test_method"] = ""
    ctx[f"{prefix}_is_tested_value"] = False
    ctx[f"{prefix}_test_source"] = ""
    p2.caption("Scenario-only reference")
    p3.caption("No DB metadata required")
    p4.caption("Notes + tire size are enough here")

    if ctx[f"{prefix}_manual_basis"] == "Direct ABC override":
        st.caption("Use this only when you want to override the reference roadload directly instead of deriving it from RRC.")
        c1, c2, c3 = st.columns(3)
        a_val = quantity_input(c1, "A", to_float(ctx.get(f"{prefix}_A"), 0.0), "force", key=f"{prefix}_A_input", step_canonical=0.1)
        b_val = quantity_input(c2, "B", to_float(ctx.get(f"{prefix}_B"), 0.0), "force_per_speed", key=f"{prefix}_B_input", step_canonical=0.0001, format_str="%.6f")
        c_val = quantity_input(c3, "C", to_float(ctx.get(f"{prefix}_C"), 0.0), "force_per_speed_squared", key=f"{prefix}_C_input", step_canonical=0.000001, format_str="%.8f")
        ctx[f"{prefix}_A"] = a_val
        ctx[f"{prefix}_B"] = b_val
        ctx[f"{prefix}_C"] = c_val
        if abs(float(a_val)) <= 0.0 and abs(float(b_val)) <= 0.0 and abs(float(c_val)) <= 0.0:
            st.caption("Reference still pending. Enter at least one non-zero ABC value.")
            return None
        load_kN = _resolved_tire_load_kN(ctx)
        equivalent_rr, epa_v_avg, epa_v2_avg = _equivalent_rr_from_abc(
            a_val,
            b_val,
            c_val,
            load_kN=load_kN,
        )
        expected_b = ((float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0) * float(a_val)) / 120.0) if abs(float(a_val)) > 0.0 else 0.0
        d1, d2, d3, d4 = st.columns(4)
        quantity_metric(d1, "Equivalent RRC", equivalent_rr, "rrc", format_str="%.3f")
        quantity_metric(d2, "B expected from A", expected_b, "force_per_speed", format_str="%.6f")
        quantity_metric(d3, "B entered", b_val, "force_per_speed", format_str="%.6f")
        d4.caption(
            (
                f"EPA v_avg: {epa_v_avg:.2f} kph | EPA v2_avg: {epa_v2_avg:.2f} kph^2"
                if epa_v_avg is not None and epa_v2_avg is not None
                else "EPA speed moments unavailable"
            )
        )
    else:
        rr_value = to_float(ctx.get(f"{prefix}_rr_n_per_kn"))
        if rr_value is None or rr_value <= 0.0:
            st.caption("Reference still pending. Enter a non-zero RRC value to derive the roadload.")
            return None
        ctx["crr1_frac_at_120kph"] = st.number_input(
            "crr1 @ 120 kph [-]",
            value=float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0),
            step=0.001,
            format="%.5f",
            key=f"{prefix}_crr1_frac_at_120kph_input",
        )
        load_kN = _resolved_tire_load_kN(ctx)
        crr_frac = float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0)
        derived_a, derived_b, derived_c, epa_v_avg = _abc_from_final_rr_target(
            rr_value,
            load_kN=load_kN,
            crr_frac_120=crr_frac,
        )
        ctx[f"{prefix}_A"] = derived_a
        ctx[f"{prefix}_B"] = derived_b
        ctx[f"{prefix}_C"] = derived_c
        st.caption("Reference roadload is derived automatically from final RRC and crr1@120 so the resolved A/B matches the EPA-equivalent target.")
        d1, d2, d3, d4 = st.columns(4)
        quantity_metric(d1, "Derived A", derived_a, "force", format_str="%.3f")
        quantity_metric(d2, "Derived B", derived_b, "force_per_speed", format_str="%.6f")
        quantity_metric(d3, "Derived C", derived_c, "force_per_speed_squared", format_str="%.8f")
        d4.caption(f"Load used: {load_kN:.3f} kN")
        d4.caption(f"crr1@120: {crr_frac:.5f}")
        if epa_v_avg is not None:
            d4.caption(f"EPA v_avg: {epa_v_avg:.2f} kph")

    return _manual_tire_reference_from_ctx(prefix, source_label="Scenario-only manual reference")


def _clear_tire_reference_ctx(prefix: str, *, preview_ctx_key: str | None = None, widget_prefix: str | None = None):
    ctx = st.session_state.ctx
    for suffix in (
        "_manual_basis",
        "_A",
        "_B",
        "_C",
        "_tire_test_code",
        "_manufacturer",
        "_model",
        "_size_code",
        "_standard_family",
        "_rr_n_per_kn",
        "_smerf",
        "_front_pressure_psi",
        "_rear_pressure_psi",
        "_effective_circumference_override_mm",
        "_effective_circumference_override_mm_prefill_size",
        "_effective_circumference_override_mm_prefill_value",
        "_test_mileage_km",
        "_test_method",
        "_test_source",
        "_is_tested_value",
        "_notes",
    ):
        ctx.pop(f"{prefix}{suffix}", None)
    if preview_ctx_key:
        ctx.pop(preview_ctx_key, None)
    if widget_prefix:
        for suffix in (
            "_front_tire_id",
            "_rear_tire_id",
            "_front_pressure_psi",
            "_rear_pressure_psi",
            "_weight_dist_fr_pct",
            "_tire_improvement_pct",
            "_tire_load_mass_basis",
            "_same_tire_front_rear",
            "_quick_add_created_id",
        ):
            ctx.pop(f"{widget_prefix}{suffix}", None)


def _sync_baseline_cda_in_ctx(cda_value: float, *, baseline_target_id: int | None = None):
    ctx = st.session_state.ctx
    if isinstance(ctx.get("selected_baseline_row"), dict):
        ctx["selected_baseline_row"]["cda_m2"] = float(cda_value)
    if isinstance(ctx.get("baseline_dict"), dict):
        ctx["baseline_dict"]["cda_m2"] = float(cda_value)
    if baseline_target_id and int(ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) == int(baseline_target_id):
        ctx["cda_m2"] = float(cda_value)


def _sync_baseline_abc_reference_in_ctx(
    *,
    a_key: str,
    b_key: str,
    c_key: str,
    a_value: float,
    b_value: float,
    c_value: float,
    baseline_target_id: int | None = None,
):
    ctx = st.session_state.ctx
    if isinstance(ctx.get("selected_baseline_row"), dict):
        ctx["selected_baseline_row"][a_key] = float(a_value)
        ctx["selected_baseline_row"][b_key] = float(b_value)
        ctx["selected_baseline_row"][c_key] = float(c_value)
    if isinstance(ctx.get("baseline_dict"), dict):
        ctx["baseline_dict"][a_key] = float(a_value)
        ctx["baseline_dict"][b_key] = float(b_value)
        ctx["baseline_dict"][c_key] = float(c_value)
    if baseline_target_id and int(ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) == int(baseline_target_id):
        ctx[a_key] = float(a_value)
        ctx[b_key] = float(b_value)
        ctx[c_key] = float(c_value)


def _characterization_basis_options() -> list[str]:
    return ["ISO 28580", "SAE J2452 / SMERF", "Custom / Manual measured RR"]


def _characterization_basis_to_mode(basis: str) -> str:
    normalized = str(basis or "").strip()
    if normalized == "ISO 28580":
        return "ISO_28580"
    if normalized == "SAE J2452 / SMERF":
        return "SAE_J2452"
    return "CUSTOM"


def _quick_add_tire_payload_from_ctx(form_prefix: str) -> tuple[dict, list[str], dict]:
    ctx = st.session_state.ctx
    basis = str(ctx.get(f"{form_prefix}_characterization_basis") or "Custom / Manual measured RR").strip()
    calculation_mode = _characterization_basis_to_mode(basis)
    family = "ISO" if calculation_mode == "ISO_28580" else "SAE" if calculation_mode == "SAE_J2452" else "CUSTOM"
    payload = {
        "tire_test_code": _string_or_none(ctx.get(f"{form_prefix}_tire_test_code")) or "",
        "manufacturer": _string_or_none(ctx.get(f"{form_prefix}_manufacturer")) or "",
        "model": _string_or_none(ctx.get(f"{form_prefix}_model")) or "",
        "test_date": _string_or_none(ctx.get(f"{form_prefix}_test_date")) or "",
        "calculation_mode": calculation_mode,
        "standard_family": family,
        "size_code": _string_or_none(ctx.get(f"{form_prefix}_size_code")),
        "load_index": _string_or_none(ctx.get(f"{form_prefix}_load_index")),
        "speed_rating": _string_or_none(ctx.get(f"{form_prefix}_speed_rating")),
        "effective_circumference_override_mm": to_float(ctx.get(f"{form_prefix}_effective_circumference_override_mm")),
        "test_method": _string_or_none(ctx.get(f"{form_prefix}_test_method")),
        "test_source": _string_or_none(ctx.get(f"{form_prefix}_test_source")),
        "test_mileage_km": to_float(ctx.get(f"{form_prefix}_test_mileage_km")),
        "is_tested_value": 1 if _truthy_flag(ctx.get(f"{form_prefix}_is_tested_value")) else 0,
        "notes": _string_or_none(ctx.get(f"{form_prefix}_notes")),
        "rr_source": _string_or_none(ctx.get(f"{form_prefix}_test_source")),
        "rr_value_source_note": _string_or_none(ctx.get(f"{form_prefix}_test_source")),
    }

    if calculation_mode == "ISO_28580":
        payload.update(
            {
                "iso_rrc_n_per_kn": to_float(ctx.get(f"{form_prefix}_iso_rrc_n_per_kn")),
                "iso_corrected_rrc_n_per_kn": to_float(ctx.get(f"{form_prefix}_iso_corrected_rrc_n_per_kn")),
                "iso_test_pressure_kpa": to_float(ctx.get(f"{form_prefix}_iso_test_pressure_kpa")),
                "iso_test_load_n": to_float(ctx.get(f"{form_prefix}_iso_test_load_n")),
                "iso_test_speed_kph": to_float(ctx.get(f"{form_prefix}_iso_test_speed_kph")),
                "iso_rolling_resistance_force_n": to_float(ctx.get(f"{form_prefix}_iso_rolling_resistance_force_n")),
                "iso_condition_notes": _string_or_none(ctx.get(f"{form_prefix}_iso_condition_notes")),
            }
        )
    elif calculation_mode == "SAE_J2452":
        payload.update(
            {
                "sae_alpha": to_float(ctx.get(f"{form_prefix}_sae_alpha")),
                "sae_beta": to_float(ctx.get(f"{form_prefix}_sae_beta")),
                "sae_a": to_float(ctx.get(f"{form_prefix}_sae_a")),
                "sae_b": to_float(ctx.get(f"{form_prefix}_sae_b")),
                "sae_c": to_float(ctx.get(f"{form_prefix}_sae_c")),
                "smerf": to_float(ctx.get(f"{form_prefix}_smerf")),
                "test_pressure_value": to_float(ctx.get(f"{form_prefix}_sae_test_pressure_value")),
                "pressure_unit": str(ctx.get(f"{form_prefix}_sae_pressure_unit") or "kPa"),
                "test_load_value": to_float(ctx.get(f"{form_prefix}_sae_test_load_value")),
                "load_unit": str(ctx.get(f"{form_prefix}_sae_load_unit") or "N"),
            }
        )
    else:
        payload.update(
            {
                "rr_n_per_kn": to_float(ctx.get(f"{form_prefix}_rr_n_per_kn")),
                "rr_quality": "manual_input",
                "rr_method": "MANUAL_ESTIMATED",
            }
        )

    summary = summarize_tire_rr(payload)
    rr_valid = to_float(summary.get("rr_n_per_kn")) is not None and float(to_float(summary.get("rr_n_per_kn"), 0.0) or 0.0) > 0.0
    errors = []
    if not payload["tire_test_code"]:
        errors.append("Tire test code is required.")
    if not payload["manufacturer"]:
        errors.append("Manufacturer is required.")
    if not payload["model"]:
        errors.append("Model is required.")
    if not payload["test_date"]:
        errors.append("Test date is required.")
    if calculation_mode == "CUSTOM":
        if payload.get("rr_n_per_kn") is None or payload.get("rr_n_per_kn") <= 0:
            errors.append("Custom / Manual measured RR requires a valid final / equivalent RRC value.")
        if not _string_or_none(payload.get("test_source")):
            errors.append("Custom / Manual measured RR requires source / provenance.")
        if not _string_or_none(payload.get("notes")):
            errors.append("Custom / Manual measured RR requires notes / rationale.")
    elif not rr_valid:
        errors.append("Preview characterization must resolve a valid RR before this tire can be created.")
    return payload, errors, summary


def _render_characterization_preview(summary: dict, *, basis: str):
    rr_value = to_float(summary.get("rr_n_per_kn"))
    smerf_value = to_float(summary.get("smerf"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Basis", basis)
    c2.metric("Resolved final RRC", format_quantity(rr_value, "rrc", format_str="%.3f"))
    c3.metric("SMERF", "-" if smerf_value is None else f"{float(smerf_value):.4f}")
    c4.metric("Method", str(summary.get("rr_method") or "Pending"))
    quality = str(summary.get("rr_quality") or "").strip()
    source = str(summary.get("rr_source") or "").strip()
    if quality or source:
        st.caption(" | ".join(part for part in (f"quality={quality}" if quality else "", f"source={source}" if source else "") if part))


def _render_quick_add_characterization_fields(form_prefix: str):
    ctx = st.session_state.ctx
    basis_options = _characterization_basis_options()
    current_basis = str(ctx.get(f"{form_prefix}_characterization_basis") or basis_options[2]).strip()
    if current_basis not in basis_options:
        current_basis = basis_options[2]
    ctx[f"{form_prefix}_characterization_basis"] = st.radio(
        "Tire characterization basis",
        basis_options,
        horizontal=True,
        index=basis_options.index(current_basis),
        key=f"{form_prefix}_characterization_basis_radio",
    )
    basis = ctx[f"{form_prefix}_characterization_basis"]

    if basis == "ISO 28580":
        st.caption("Use ISO fields to resolve the canonical RR through the existing backend normalization path.")
        i1, i2, i3, i4 = st.columns(4)
        ctx[f"{form_prefix}_iso_rrc_n_per_kn"] = quantity_input(i1, "ISO RRC", to_float(ctx.get(f"{form_prefix}_iso_rrc_n_per_kn"), 0.0), "rrc", key=f"{form_prefix}_iso_rrc_input", step_canonical=0.05, format_str="%.3f")
        ctx[f"{form_prefix}_iso_corrected_rrc_n_per_kn"] = quantity_input(i2, "Corrected ISO RRC", to_float(ctx.get(f"{form_prefix}_iso_corrected_rrc_n_per_kn"), 0.0), "rrc", key=f"{form_prefix}_iso_corrected_rrc_input", step_canonical=0.05, format_str="%.3f")
        ctx[f"{form_prefix}_iso_test_pressure_kpa"] = i3.number_input("ISO pressure [kPa]", value=float(to_float(ctx.get(f"{form_prefix}_iso_test_pressure_kpa"), 0.0) or 0.0), step=5.0, format="%.1f", key=f"{form_prefix}_iso_pressure_input")
        ctx[f"{form_prefix}_iso_test_load_n"] = i4.number_input("ISO load [N]", value=float(to_float(ctx.get(f"{form_prefix}_iso_test_load_n"), 0.0) or 0.0), step=50.0, format="%.1f", key=f"{form_prefix}_iso_load_input")
        i5, i6 = st.columns(2)
        ctx[f"{form_prefix}_iso_test_speed_kph"] = i5.number_input("ISO speed [km/h]", value=float(to_float(ctx.get(f"{form_prefix}_iso_test_speed_kph"), 0.0) or 0.0), step=1.0, format="%.1f", key=f"{form_prefix}_iso_speed_input")
        ctx[f"{form_prefix}_iso_rolling_resistance_force_n"] = i6.number_input("ISO rolling resistance force [N]", value=float(to_float(ctx.get(f"{form_prefix}_iso_rolling_resistance_force_n"), 0.0) or 0.0), step=1.0, format="%.2f", key=f"{form_prefix}_iso_force_input")
        ctx[f"{form_prefix}_iso_condition_notes"] = st.text_input("ISO condition notes", value=str(ctx.get(f"{form_prefix}_iso_condition_notes") or ""), key=f"{form_prefix}_iso_condition_notes_input")
    elif basis == "SAE J2452 / SMERF":
        st.caption("Use official SAE/J2452 characterization inputs to calculate SMERF and reference RRC.")
        s1, s2, s3, s4, s5 = st.columns(5)
        ctx[f"{form_prefix}_sae_alpha"] = s1.number_input("alpha", value=float(to_float(ctx.get(f"{form_prefix}_sae_alpha"), 0.0) or 0.0), step=0.01, format="%.6f", key=f"{form_prefix}_sae_alpha_input")
        ctx[f"{form_prefix}_sae_beta"] = s2.number_input("beta", value=float(to_float(ctx.get(f"{form_prefix}_sae_beta"), 0.0) or 0.0), step=0.01, format="%.6f", key=f"{form_prefix}_sae_beta_input")
        ctx[f"{form_prefix}_sae_a"] = s3.number_input("a", value=float(to_float(ctx.get(f"{form_prefix}_sae_a"), 0.0) or 0.0), step=0.0001, format="%.6f", key=f"{form_prefix}_sae_a_input")
        ctx[f"{form_prefix}_sae_b"] = s4.number_input("b", value=float(to_float(ctx.get(f"{form_prefix}_sae_b"), 0.0) or 0.0), step=0.0001, format="%.6f", key=f"{form_prefix}_sae_b_input")
        ctx[f"{form_prefix}_sae_c"] = s5.number_input("c", value=float(to_float(ctx.get(f"{form_prefix}_sae_c"), 0.0) or 0.0), step=0.0001, format="%.6f", key=f"{form_prefix}_sae_c_input")
        s6, s7, s8, s9 = st.columns(4)
        pressure_units = ["kPa", "psi"]
        load_units = ["N", "kg", "lbf"]
        pressure_unit = str(ctx.get(f"{form_prefix}_sae_pressure_unit") or "kPa")
        load_unit = str(ctx.get(f"{form_prefix}_sae_load_unit") or "N")
        if pressure_unit not in pressure_units:
            pressure_unit = "kPa"
        if load_unit not in load_units:
            load_unit = "N"
        ctx[f"{form_prefix}_sae_pressure_unit"] = s6.selectbox("Pressure unit", pressure_units, index=pressure_units.index(pressure_unit), key=f"{form_prefix}_sae_pressure_unit_input")
        ctx[f"{form_prefix}_sae_test_pressure_value"] = s7.number_input(f"SAE pressure [{ctx[f'{form_prefix}_sae_pressure_unit']}]", value=float(to_float(ctx.get(f"{form_prefix}_sae_test_pressure_value"), 0.0) or 0.0), step=1.0, format="%.2f", key=f"{form_prefix}_sae_pressure_value_input")
        ctx[f"{form_prefix}_sae_load_unit"] = s8.selectbox("Load unit", load_units, index=load_units.index(load_unit), key=f"{form_prefix}_sae_load_unit_input")
        ctx[f"{form_prefix}_sae_test_load_value"] = s9.number_input(f"SAE load [{ctx[f'{form_prefix}_sae_load_unit']}]", value=float(to_float(ctx.get(f"{form_prefix}_sae_test_load_value"), 0.0) or 0.0), step=1.0, format="%.2f", key=f"{form_prefix}_sae_load_value_input")
        ctx[f"{form_prefix}_smerf"] = st.number_input("Optional SMERF override", value=float(to_float(ctx.get(f"{form_prefix}_smerf"), 0.0) or 0.0), step=0.1, format="%.4f", key=f"{form_prefix}_smerf_input")
    else:
        st.caption("Custom data stores a measured or engineering RR directly without claiming SAE / ISO methodology.")
        st.caption("Use this path when you have a final / equivalent RRC from manual measurement, estimate, or external engineering input.")
        ctx[f"{form_prefix}_rr_n_per_kn"] = quantity_input(st, "Final / equivalent RRC", to_float(ctx.get(f"{form_prefix}_rr_n_per_kn"), 0.0), "rrc", key=f"{form_prefix}_rr_input", step_canonical=0.05, format_str="%.3f")
        st.info("For custom data, source / provenance and notes are required before creating a physical Tire DB record.")


def _render_quick_add_tire_reference_builder(
    *,
    widget_prefix: str,
    preview_ctx_key: str,
    base_row: dict | None = None,
) -> dict | None:
    ctx = st.session_state.ctx
    form_prefix = f"{widget_prefix}_quick_add"
    ctx.setdefault(f"{form_prefix}_test_date", pd.Timestamp.today().strftime("%Y-%m-%d"))
    ctx.setdefault(f"{form_prefix}_characterization_basis", "Custom / Manual measured RR")
    st.caption("Create a compact tire record in Tire DB without leaving VDE Setup. After save, it becomes the selected reference here.")

    q1, q2, q3, q4 = st.columns(4)
    ctx[f"{form_prefix}_tire_test_code"] = q1.text_input("Tire test code *", value=str(ctx.get(f"{form_prefix}_tire_test_code") or ""), key=f"{form_prefix}_tire_test_code_input")
    ctx[f"{form_prefix}_manufacturer"] = q2.text_input("Manufacturer *", value=str(ctx.get(f"{form_prefix}_manufacturer") or ""), key=f"{form_prefix}_manufacturer_input")
    ctx[f"{form_prefix}_model"] = q3.text_input("Model *", value=str(ctx.get(f"{form_prefix}_model") or ""), key=f"{form_prefix}_model_input")
    ctx[f"{form_prefix}_test_date"] = q4.text_input("Test date *", value=str(ctx.get(f"{form_prefix}_test_date") or ""), key=f"{form_prefix}_test_date_input")

    q5, q6, q7, q8 = st.columns(4)
    with q5:
        ctx[f"{form_prefix}_size_code"] = _render_tire_size_selector(
            field_key=f"{form_prefix}_size_code",
            label="Tire size",
        )
    _prefill_circumference_from_size(
        size_field_key=f"{form_prefix}_size_code",
        circumference_field_key=f"{form_prefix}_effective_circumference_override_mm",
    )
    ctx[f"{form_prefix}_load_index"] = q6.text_input("Load index", value=str(ctx.get(f"{form_prefix}_load_index") or ""), key=f"{form_prefix}_load_index_input")
    ctx[f"{form_prefix}_speed_rating"] = q7.text_input("Speed rating", value=str(ctx.get(f"{form_prefix}_speed_rating") or ""), key=f"{form_prefix}_speed_rating_input")
    ctx[f"{form_prefix}_test_method"] = q8.text_input("Test type", value=str(ctx.get(f"{form_prefix}_test_method") or ""), key=f"{form_prefix}_test_method_input")

    q9, q10, q11, q12 = st.columns(4)
    ctx[f"{form_prefix}_front_pressure_psi"] = _render_tire_pressure_input(
        q9,
        "Front pressure",
        to_float(ctx.get(f"{form_prefix}_front_pressure_psi"), 32.0),
        key_base=f"{form_prefix}_front_pressure_input",
    )
    ctx[f"{form_prefix}_rear_pressure_psi"] = _render_tire_pressure_input(
        q10,
        "Rear pressure",
        to_float(ctx.get(f"{form_prefix}_rear_pressure_psi"), to_float(ctx.get(f"{form_prefix}_front_pressure_psi"), 32.0)),
        key_base=f"{form_prefix}_rear_pressure_input",
    )
    ctx[f"{form_prefix}_test_mileage_km"] = q11.number_input(
        f"Mileage [{'km' if _current_unit_system() == 'Metric' else 'mi'}]",
        value=float((to_float(ctx.get(f"{form_prefix}_test_mileage_km"), 0.0) or 0.0) * (0.621371192237 if _current_unit_system() == "US customary" else 1.0)),
        step=100.0,
        format="%.0f",
        key=f"{form_prefix}_test_mileage_input",
    )
    if _current_unit_system() == "US customary":
        ctx[f"{form_prefix}_test_mileage_km"] = float(ctx[f"{form_prefix}_test_mileage_km"]) / 0.621371192237
    circumference_default_mm = _tire_circumference_mm({"size_code": ctx.get(f"{form_prefix}_size_code")})
    circumference_value = to_float(ctx.get(f"{form_prefix}_effective_circumference_override_mm"), circumference_default_mm)
    display_circumference = float((circumference_value or 0.0) / (25.4 if _current_unit_system() == "US customary" else 1.0))
    ctx[f"{form_prefix}_effective_circumference_override_mm"] = q12.number_input(
        f"Circumference [{'mm' if _current_unit_system() == 'Metric' else 'in'}]",
        value=display_circumference,
        step=1.0 if _current_unit_system() == "Metric" else 0.1,
        format="%.1f",
        key=f"{form_prefix}_circumference_input",
    )
    if _current_unit_system() == "US customary":
        ctx[f"{form_prefix}_effective_circumference_override_mm"] = float(ctx[f"{form_prefix}_effective_circumference_override_mm"]) * 25.4

    ctx[f"{form_prefix}_is_tested_value"] = st.checkbox("Tested value", value=bool(ctx.get(f"{form_prefix}_is_tested_value", False)), key=f"{form_prefix}_is_tested_value_input")
    ctx[f"{form_prefix}_test_source"] = st.text_input("Source / provenance", value=str(ctx.get(f"{form_prefix}_test_source") or ""), key=f"{form_prefix}_test_source_input")
    ctx[f"{form_prefix}_notes"] = st.text_area("Notes", value=str(ctx.get(f"{form_prefix}_notes") or ""), key=f"{form_prefix}_notes_input", height=80)
    _render_quick_add_characterization_fields(form_prefix)

    preview_summary = ctx.get(f"{form_prefix}_characterization_preview")
    preview_col, create_hint_col = st.columns([1, 4])
    if preview_col.button("Preview characterization", key=f"{form_prefix}_preview_characterization_button"):
        payload, _, summary = _quick_add_tire_payload_from_ctx(form_prefix)
        preview_summary = {"payload": payload, "summary": summary}
        ctx[f"{form_prefix}_characterization_preview"] = preview_summary

    if isinstance(preview_summary, dict):
        summary = dict(preview_summary.get("summary") or {})
        _render_characterization_preview(summary, basis=str(ctx.get(f"{form_prefix}_characterization_basis") or ""))
        rr_preview = to_float(summary.get("rr_n_per_kn"))
        if rr_preview is None or rr_preview <= 0:
            create_hint_col.caption("Characterization preview is still missing a valid RR result.")
            st.warning("Preview characterization did not resolve a valid RR yet. Complete the required characterization inputs first.")
    else:
        create_hint_col.caption("Run Preview characterization before creating a Tire DB record from this compact form.")

    preview_result = ctx.get(preview_ctx_key)
    code = _string_or_none(ctx.get(f"{form_prefix}_tire_test_code"))
    existing = {}
    if code:
        try:
            existing = get_tire_by_code(code) or {}
        except Exception:
            existing = {}

    if existing:
        st.warning(f"Tire test code `{code}` already exists in Tire DB as id={existing.get('id')}.")
        if st.button("Use existing tire", key=f"{form_prefix}_use_existing_button"):
            ctx[f"{widget_prefix}_front_tire_id"] = int(existing["id"])
            ctx[f"{widget_prefix}_rear_tire_id"] = int(existing["id"])
            ctx[f"{widget_prefix}_same_tire_front_rear"] = True
            try:
                preview_result = preview_tire_roadload_from_row(
                    _build_tire_context_row(base_row),
                    {
                        "front_tire_id": int(existing["id"]),
                        "rear_tire_id": int(existing["id"]),
                        "same_tire_front_rear": True,
                        "front_pressure_psi": to_float(ctx.get(f"{form_prefix}_front_pressure_psi"), 32.0),
                        "rear_pressure_psi": to_float(ctx.get(f"{form_prefix}_rear_pressure_psi"), 32.0),
                        "front_weight_distribution_pct": to_float(ctx.get("weight_dist_fr_pct"), 50.0),
                        "tire_improvement_pct": 0.0,
                        "tire_load_mass_basis": resolve_tire_load_mass_basis(ctx),
                        "mass_kg": ctx.get("mass_kg"),
                        "test_mass_kg": ctx.get("test_mass_kg"),
                        "inertia_class": ctx.get("inertia_class"),
                        "twc_kg": ctx.get("twc_kg"),
                        "etw_kg": ctx.get("etw_kg"),
                    },
                )
                ctx[preview_ctx_key] = preview_result
            except Exception as exc:
                st.error(f"Could not preview the existing tire: {exc}")

    preview_rr_ready = False
    if isinstance(preview_summary, dict):
        preview_rr = to_float(((preview_summary.get("summary") or {}).get("rr_n_per_kn")))
        preview_rr_ready = preview_rr is not None and preview_rr > 0

    create_col, _ = st.columns([1, 4])
    if create_col.button("Create and select tire", key=f"{form_prefix}_create_button", disabled=bool(existing) or not preview_rr_ready):
        payload, errors, summary = _quick_add_tire_payload_from_ctx(form_prefix)
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                new_id = int(create_tire_from_form(payload))
                ctx[f"{widget_prefix}_quick_add_created_id"] = new_id
                ctx[f"{widget_prefix}_front_tire_id"] = new_id
                ctx[f"{widget_prefix}_rear_tire_id"] = new_id
                ctx[f"{widget_prefix}_same_tire_front_rear"] = True
                preview_result = preview_tire_roadload_from_row(
                    _build_tire_context_row(base_row),
                    {
                        "front_tire_id": new_id,
                        "rear_tire_id": new_id,
                        "same_tire_front_rear": True,
                        "front_pressure_psi": to_float(ctx.get(f"{form_prefix}_front_pressure_psi"), 32.0),
                        "rear_pressure_psi": to_float(ctx.get(f"{form_prefix}_rear_pressure_psi"), 32.0),
                        "front_weight_distribution_pct": to_float(ctx.get("weight_dist_fr_pct"), 50.0),
                        "tire_improvement_pct": 0.0,
                        "tire_load_mass_basis": resolve_tire_load_mass_basis(ctx),
                        "mass_kg": ctx.get("mass_kg"),
                        "test_mass_kg": ctx.get("test_mass_kg"),
                        "inertia_class": ctx.get("inertia_class"),
                        "twc_kg": ctx.get("twc_kg"),
                        "etw_kg": ctx.get("etw_kg"),
                    },
                )
                ctx[preview_ctx_key] = preview_result
                st.success(f"Tire created in Tire DB with id={new_id} and selected here.")
            except Exception as exc:
                st.error(f"Could not create tire: {exc}")

    if not isinstance(preview_result, dict):
        return None
    reference = _reference_from_preview_result(preview_result, source_label="Quick-added Tire DB reference")
    if reference:
        _render_tire_reference_metrics(reference, title="Resolved quick-add reference")
    return preview_result


def _render_tire_reference_selector(
    *,
    title: str,
    helper_text: str,
    mode_key: str,
    prefix: str,
    widget_prefix: str,
    source_label: str,
    base_row: dict | None = None,
    inherited_reference: dict | None = None,
    allow_unset: bool = False,
    save_to_baseline_label: str | None = None,
    baseline_target_id: int | None = None,
) -> dict | None:
    ctx = st.session_state.ctx
    with st.container(border=True):
        _render_tire_editor_block_header(title, helper_text)
        options = []
        if inherited_reference:
            options.append("Inherited baseline")
        if allow_unset:
            options.append("Not set")
        options.extend(["Select from Tire DB", "Quick add tire to DB", "Scenario-only manual reference"])
        default_mode = str(ctx.get(mode_key) or ("Inherited baseline" if inherited_reference else options[0]))
        if default_mode not in options:
            default_mode = "Inherited baseline" if inherited_reference else options[0]
        ctx[mode_key] = st.radio(
            f"{title} source",
            options,
            horizontal=False,
            index=options.index(default_mode),
            key=f"{mode_key}_radio",
        )

        if ctx[mode_key] == "Inherited baseline":
            _clear_tire_reference_ctx(prefix, preview_ctx_key=f"{prefix}_preview_result", widget_prefix=widget_prefix)
            reference = inherited_reference
        elif ctx[mode_key] == "Not set":
            _clear_tire_reference_ctx(prefix, preview_ctx_key=f"{prefix}_preview_result", widget_prefix=widget_prefix)
            reference = None
            st.caption("Reference is staged only when you choose a source below.")
        elif ctx[mode_key] == "Scenario-only manual reference":
            reference = _render_manual_tire_reference_inputs(prefix=prefix, title="Scenario-only tire reference")
        elif ctx[mode_key] == "Quick add tire to DB":
            preview = _render_quick_add_tire_reference_builder(
                widget_prefix=widget_prefix,
                preview_ctx_key=f"{prefix}_preview_result",
                base_row=base_row,
            )
            reference = _reference_from_preview_result(preview, source_label=source_label)
        else:
            preview = _render_tire_db_reference_builder(
                widget_prefix=widget_prefix,
                preview_ctx_key=f"{prefix}_preview_result",
                base_row=base_row,
                save_to_baseline_label=save_to_baseline_label,
                baseline_target_id=baseline_target_id,
            )
            reference = _reference_from_preview_result(preview, source_label=source_label)

        if reference:
            _render_tire_reference_metrics(reference, title="Active reference")
        return reference


def _sync_tire_scenario_snapshot_fields(
    *,
    applied_method: str,
    current_reference: dict | None,
    walked_reference: dict | None,
    equivalent_a,
    equivalent_b,
    equivalent_c,
):
    ctx = st.session_state.ctx
    if applied_method == "Keep inherited" or str(ctx.get("component_mode_tires") or "").strip() == "Keep inherited":
        for key in (
            "front_tire_id",
            "rear_tire_id",
            "tire_load_mass_used_kg",
            "tire_A_final",
            "tire_B_final",
            "tire_C_final",
        ):
            ctx.pop(key, None)
        ctx["tire_rr_note"] = "Keep inherited"
        ctx["tire_calc_source"] = "keep_inherited"
        ctx["tire_calc_notes"] = json.dumps({"applied_method": "Keep inherited"}, ensure_ascii=True)
        return

    for key in (
        "front_tire_id",
        "rear_tire_id",
        "tire_load_mass_basis",
        "tire_load_mass_used_kg",
        "tire_A_final",
        "tire_B_final",
        "tire_C_final",
    ):
        ctx.pop(key, None)

    active_reference = None
    if applied_method == "Walked tire comparison" and walked_reference:
        active_reference = walked_reference
    elif applied_method == "Tire Improvement %" and current_reference:
        active_reference = current_reference
    elif applied_method == "Manual Delta RR" and current_reference:
        active_reference = current_reference

    if active_reference:
        for key in (
            "front_tire_id",
            "rear_tire_id",
            "front_pressure_psi",
            "rear_pressure_psi",
            "tire_load_mass_basis",
            "tire_load_mass_used_kg",
            "tire_calc_source",
        ):
            value = active_reference.get(key)
            if value not in (None, ""):
                ctx[key] = value
        if active_reference.get("size_code"):
            ctx["tire_size"] = active_reference.get("size_code")
        if active_reference.get("rr_n_per_kn") is not None:
            ctx["rrc_N_per_kN"] = active_reference.get("rr_n_per_kn")
        if _reference_has_nonzero_abc(active_reference):
            ctx["tire_A_final"] = active_reference.get("A")
            ctx["tire_B_final"] = active_reference.get("B")
            ctx["tire_C_final"] = active_reference.get("C")

    note_payload = {
        "applied_method": applied_method,
        "change_intent": _string_or_none(ctx.get("tire_manual_change_intent")) or "Scenario-only engineering adjustment",
        "data_role": (
            "engineering_target"
            if str(ctx.get("tire_manual_change_intent") or "").strip() == "Engineering target / supplier request"
            else "scenario_assumption"
        ),
        "evidence_basis": (
            "engineering_estimate"
            if str(ctx.get("tire_manual_change_intent") or "").strip() == "Engineering target / supplier request"
            else "manual_input"
        ),
        "current_reference": _tire_provenance_note(current_reference),
        "walked_reference": _tire_provenance_note(walked_reference),
        "equivalent_delta_abc": {
            "A": equivalent_a,
            "B": equivalent_b,
            "C": equivalent_c,
        },
        "reference_rrc_n_per_kn": to_float((current_reference or {}).get("rr_n_per_kn"), to_float(ctx.get("rrc_N_per_kN"))),
        "target_rrc_n_per_kn": to_float(ctx.get("tire_manual_target_rr_n_per_kn")),
        "manual_delta_rr_n_per_kn": to_float(ctx.get("tire_manual_delta_rr_n_per_kn")),
        "tire_improvement_pct": to_float(ctx.get("tire_improvement_pct")),
        "manual_source": _string_or_none(ctx.get("tire_manual_delta_rr_source")),
        "manual_target_label": _string_or_none(ctx.get("tire_manual_delta_rr_label")),
        "manual_size_code": _string_or_none(ctx.get("tire_manual_delta_rr_size_code")),
        "manual_notes": _string_or_none(ctx.get("tire_manual_delta_rr_notes")),
        "input_basis": _string_or_none(ctx.get("tire_manual_adjustment_input_type")),
    }
    ctx["tire_calc_source"] = (
        "engineering_target"
        if str(ctx.get("tire_manual_change_intent") or "").strip() == "Engineering target / supplier request"
        else
        "tire_walked_comparison" if applied_method == "Walked tire comparison"
        else "tire_improvement_pct" if applied_method == "Tire Improvement %"
        else "manual_delta_rr"
    )
    ctx["tire_calc_notes"] = json.dumps(note_payload, ensure_ascii=True)
    if _string_or_none(note_payload.get("manual_size_code")):
        ctx["tire_size"] = note_payload["manual_size_code"]
    ctx["tire_rr_note"] = applied_method


def _render_tire_db_reference_builder(
    *,
    widget_prefix: str,
    preview_ctx_key: str,
    base_row: dict | None = None,
    save_to_baseline_label: str | None = None,
    baseline_target_id: int | None = None,
) -> dict | None:
    ctx = st.session_state.ctx
    try:
        tires = get_available_tires()
    except Exception as e:
        st.error(f"Could not load tire database: {e}")
        return None

    if not tires:
        st.warning("No active tires found in the tire database.")
        return None

    tire_by_id = {int(r["id"]): r for r in tires if r.get("id") is not None}
    tire_by_code = {
        str(r.get("tire_test_code") or "").strip(): r
        for r in tires
        if str(r.get("tire_test_code") or "").strip()
    }
    tire_codes = sorted(tire_by_code.keys())
    if not tire_codes:
        st.warning("Active tire rows need tire_test_code before they can be used here.")
        return None

    def _code_for_tire_id(tire_id) -> str:
        try:
            row = tire_by_id.get(int(tire_id))
        except Exception:
            row = None
        return str((row or {}).get("tire_test_code") or "").strip()

    base = dict(base_row or {})
    front_default = _code_for_tire_id(ctx.get(f"{widget_prefix}_front_tire_id") or base.get("front_tire_id"))
    rear_default = _code_for_tire_id(ctx.get(f"{widget_prefix}_rear_tire_id") or base.get("rear_tire_id"))
    if front_default not in tire_codes:
        front_default = tire_codes[0]
    if rear_default not in tire_codes:
        rear_default = front_default

    same_default = bool(ctx.get(f"{widget_prefix}_same_tire_front_rear", True) if ctx.get(f"{widget_prefix}_same_tire_front_rear") is not None else (front_default == rear_default))
    legislation = str(base.get("legislation") or ctx.get("legislation") or "").strip().upper()
    basis_default = resolve_tire_load_mass_basis(
        {
            "legislation": legislation,
            "tire_load_mass_basis": ctx.get(f"{widget_prefix}_tire_load_mass_basis") or ctx.get("tire_load_mass_basis") or base.get("tire_load_mass_basis"),
        }
    )

    c1, c2 = st.columns(2)
    front_code = c1.selectbox(
        "Front tire",
        tire_codes,
        index=tire_codes.index(front_default),
        key=f"{widget_prefix}_front_tire_code",
    )
    front_tire_id = int(tire_by_code[front_code]["id"])
    c1.caption(_tire_label(tire_by_code[front_code]))
    same_tire = c2.checkbox(
        "Same tire front/rear",
        value=same_default,
        key=f"{widget_prefix}_same_tire_checkbox",
    )

    rear_tire_id = front_tire_id
    if same_tire:
        rear_code = front_code
        st.caption(f"Rear tire mirrors front tire: {_tire_label(tire_by_id[front_tire_id])}")
    else:
        rear_code = st.selectbox(
            "Rear tire",
            tire_codes,
            index=tire_codes.index(rear_default),
            key=f"{widget_prefix}_rear_tire_code",
        )
        rear_tire_id = int(tire_by_code[rear_code]["id"])
        st.caption(_tire_label(tire_by_code[rear_code]))

    p1, p2, p3, p4 = st.columns(4)
    front_pressure_psi = _render_tire_pressure_input(
        p1,
        "Front pressure",
        to_float(ctx.get(f"{widget_prefix}_front_pressure_psi", ctx.get("front_pressure_psi", base.get("front_pressure_psi", 32.0))), 32.0),
        key_base=f"{widget_prefix}_front_pressure_input",
    )
    rear_pressure_psi = _render_tire_pressure_input(
        p2,
        "Rear pressure",
        to_float(ctx.get(f"{widget_prefix}_rear_pressure_psi", ctx.get("rear_pressure_psi", base.get("rear_pressure_psi", 32.0))), 32.0),
        key_base=f"{widget_prefix}_rear_pressure_input",
    )
    front_weight_distribution_pct = p3.number_input(
        "Front weight distribution [%]",
        value=float(ctx.get(f"{widget_prefix}_weight_dist_fr_pct", ctx.get("weight_dist_fr_pct", base.get("weight_dist_fr_pct", 50.0))) or 50.0),
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        format="%.1f",
        key=f"{widget_prefix}_weight_dist_input",
    )
    tire_improvement_pct = p4.number_input(
        "Tire improvement [%]",
        value=float(ctx.get(f"{widget_prefix}_tire_improvement_pct", ctx.get("tire_improvement_pct", base.get("tire_improvement_pct", 0.0))) or 0.0),
        step=0.5,
        format="%.1f",
        key=f"{widget_prefix}_improvement_input",
    )

    tire_load_mass_basis = str(ctx.get(f"{widget_prefix}_tire_load_mass_basis") or basis_default).strip().upper()
    st.caption(f"Reference mass basis: `{tire_load_mass_basis}`")

    preview_result = None
    if st.button("Preview tire reference", key=f"{widget_prefix}_preview_button"):
        try:
            row_context = _build_tire_context_row(base_row)
            preview_result = preview_tire_roadload_from_row(
                row_context,
                {
                    "front_tire_id": front_tire_id,
                    "rear_tire_id": rear_tire_id,
                    "same_tire_front_rear": same_tire,
                    "front_pressure_psi": front_pressure_psi,
                    "rear_pressure_psi": rear_pressure_psi,
                    "front_weight_distribution_pct": front_weight_distribution_pct,
                    "tire_improvement_pct": tire_improvement_pct,
                    "tire_load_mass_basis": tire_load_mass_basis,
                    "mass_kg": ctx.get("mass_kg"),
                    "test_mass_kg": ctx.get("test_mass_kg"),
                    "inertia_class": ctx.get("inertia_class"),
                    "twc_kg": ctx.get("twc_kg"),
                    "etw_kg": ctx.get("etw_kg"),
                },
            )
            ctx[f"{widget_prefix}_front_tire_id"] = front_tire_id
            ctx[f"{widget_prefix}_rear_tire_id"] = rear_tire_id
            ctx[f"{widget_prefix}_front_pressure_psi"] = front_pressure_psi
            ctx[f"{widget_prefix}_rear_pressure_psi"] = rear_pressure_psi
            ctx[f"{widget_prefix}_weight_dist_fr_pct"] = front_weight_distribution_pct
            ctx[f"{widget_prefix}_tire_improvement_pct"] = tire_improvement_pct
            ctx[f"{widget_prefix}_tire_load_mass_basis"] = tire_load_mass_basis
            ctx[preview_ctx_key] = preview_result
        except Exception as e:
            st.error(f"Failed to preview tire reference: {e}")
            return None

    if preview_result is None:
        cached = ctx.get(preview_ctx_key)
        if isinstance(cached, dict) and cached.get("save_payload"):
            preview_result = cached

    if not preview_result:
        return None

    reference = _reference_from_preview_result(preview_result, source_label="Tire DB reference")
    if reference:
        _render_tire_reference_metrics(reference, title="Resolved Tire DB reference")

    if save_to_baseline_label and baseline_target_id:
        if st.button(save_to_baseline_label, key=f"{widget_prefix}_save_to_baseline_button"):
            try:
                payload = save_tire_roadload_to_vde(int(baseline_target_id), preview_result)
                ctx["tire_saved_payload"] = payload
                st.success(f"Tire reference saved to VDE id={baseline_target_id}.")
            except Exception as e:
                st.error(f"Failed to update baseline tire reference: {e}")

    with st.expander("Reference preview details", expanded=False):
        st.write(
            {
                "application": preview_result.get("application"),
                "mass_resolution": preview_result.get("mass_resolution"),
                "calculation": preview_result.get("calculation"),
                "save_payload": preview_result.get("save_payload"),
            }
        )
    return preview_result


def _render_tire_preview_result(preview_result: dict, *, saved_vde_id: int | None = None):
    ctx = st.session_state.ctx
    calc = preview_result.get("calculation", {})
    loads = calc.get("loads", {})
    total_final = calc.get("total_final_abc", {})
    mass_resolution = preview_result.get("mass_resolution", {})
    delta_vs_saved = preview_result.get("delta_vs_saved", {})

    st.success("Tire component preview ready.")
    m1, m2, m3, m4 = st.columns(4)
    quantity_metric(m1, "Mass used", calc.get("tire_load_mass_used_kg"), "mass", format_str="%.1f")
    quantity_metric(m2, "Front axle load", loads.get("front_axle_load_n"), "force", format_str="%.1f")
    quantity_metric(m3, "Rear axle load", loads.get("rear_axle_load_n"), "force", format_str="%.1f")
    m4.metric("Tire ABC", _compact_abc(total_final))

    include_default = bool(ctx.get("include_tire_component", True))
    ctx["include_tire_component"] = st.checkbox(
        "Include tire component in TOTAL preview",
        value=include_default,
        key="tire_component_include_checkbox",
    )
    st.caption(
        f"Mass basis: {mass_resolution.get('basis')} ({mass_resolution.get('source_field')}). "
        "When enabled, the workflow preview adds this tire component into ABC_TOTAL."
    )

    if delta_vs_saved:
        st.caption(f"Delta vs saved tire ABC: {_compact_abc(delta_vs_saved)}")

    with st.expander("Tire component details", expanded=False):
        st.write(
            {
                "application": preview_result.get("application"),
                "mass_resolution": preview_result.get("mass_resolution"),
                "calculation": preview_result.get("calculation"),
                "save_payload": preview_result.get("save_payload"),
                "component": preview_result.get("component_dict"),
            }
        )

    if saved_vde_id:
        save_col, note_col = st.columns([1, 3])
        if save_col.button("Apply tire data to saved baseline", key=f"btn_tire_save_component_{saved_vde_id}"):
            try:
                payload = save_tire_roadload_to_vde(int(saved_vde_id), preview_result)
                ctx["tire_saved_payload"] = payload
                st.success(f"Tire roadload saved to VDE id={saved_vde_id}.")
            except Exception as e:
                st.error(f"Failed to save tire roadload to VDE: {e}")

        saved_payload = ctx.get("tire_saved_payload")
        if isinstance(saved_payload, dict):
            note_col.caption(
                "Saved tire application: "
                f"front={saved_payload.get('front_tire_id')} | "
                f"rear={saved_payload.get('rear_tire_id')} | "
                f"ABC={_compact_abc({'A': saved_payload.get('tire_A_final'), 'B': saved_payload.get('tire_B_final'), 'C': saved_payload.get('tire_C_final')})}"
            )


def render_tire_component_section(
    *,
    base_row: dict | None = None,
    saved_vde_id: int | None = None,
    tires_df=None,
    source_mode_override: str | None = None,
    show_source_selector: bool = True,
):
    ctx = st.session_state.ctx
    _render_tire_pressure_unit_toggle(key="tire_component_section_pressure_unit_toggle")
    source_options = ["Tire DB", "Manual RR"]
    current_source = str(source_mode_override or ctx.get("tire_component_source") or "Manual RR")
    if current_source not in source_options:
        current_source = "Manual RR"
    if show_source_selector:
        ctx["tire_component_source"] = st.radio(
            "Tire component source",
            source_options,
            horizontal=True,
            index=source_options.index(current_source),
            key="tire_component_source_radio",
        )
    else:
        ctx["tire_component_source"] = current_source
        st.caption(f"Tire source mode: `{current_source}`")

    if ctx["tire_component_source"] == "Manual RR":
        ctx["include_tire_component"] = False
        st.caption("Manual RR stays inside the Tires component as a fallback path when no tire DB component is selected.")
        render_rr_section(prefill=base_row, tires_df=tires_df)
        return

    try:
        tires = get_available_tires()
    except Exception as e:
        st.error(f"Could not load tire database: {e}")
        ctx["include_tire_component"] = False
        return

    if not tires:
        st.warning("No active tires found in the tire database. Use Manual RR for now.")
        ctx["include_tire_component"] = False
        return

    tire_by_id = {int(r["id"]): r for r in tires if r.get("id") is not None}
    tire_by_code = {
        str(r.get("tire_test_code") or "").strip(): r
        for r in tires
        if str(r.get("tire_test_code") or "").strip()
    }
    tire_codes = sorted(tire_by_code.keys())
    if not tire_codes:
        st.warning("Active tire rows need tire_test_code before they can be used in VDE Setup.")
        ctx["include_tire_component"] = False
        return

    def _code_for_tire_id(tire_id) -> str:
        try:
            row = tire_by_id.get(int(tire_id))
        except Exception:
            row = None
        return str((row or {}).get("tire_test_code") or "").strip()

    base = dict(base_row or {})
    front_default = _code_for_tire_id(ctx.get("front_tire_id") or base.get("front_tire_id"))
    rear_default = _code_for_tire_id(ctx.get("rear_tire_id") or base.get("rear_tire_id"))
    if front_default not in tire_codes:
        front_default = tire_codes[0]
    if rear_default not in tire_codes:
        rear_default = front_default

    same_default = bool(ctx.get("same_tire_front_rear", False) or (front_default == rear_default))
    legislation = str(base.get("legislation") or ctx.get("legislation") or "").strip().upper()
    basis_default = resolve_tire_load_mass_basis(
        {
            "legislation": legislation,
            "tire_load_mass_basis": ctx.get("tire_load_mass_basis") or base.get("tire_load_mass_basis"),
        }
    )

    c1, c2 = st.columns(2)
    front_tire_code = c1.selectbox("Front tire", tire_codes, index=tire_codes.index(front_default), key="tire_component_front_code")
    front_tire_id = int(tire_by_code[front_tire_code]["id"])
    c1.caption(_tire_label(tire_by_code[front_tire_code]))
    same_tire = c2.checkbox("Same tire front/rear", value=same_default, key="tire_component_same_checkbox")

    rear_tire_id = front_tire_id
    if same_tire:
        st.caption(f"Rear tire mirrors front tire: {_tire_label(tire_by_id[front_tire_id])}")
        rear_tire_code = front_tire_code
    else:
        rear_tire_code = st.selectbox("Rear tire", tire_codes, index=tire_codes.index(rear_default), key="tire_component_rear_code")
        rear_tire_id = int(tire_by_code[rear_tire_code]["id"])
        st.caption(_tire_label(tire_by_code[rear_tire_code]))

    p1, p2, p3, p4 = st.columns(4)
    front_pressure_psi = _render_tire_pressure_input(
        p1,
        "Front pressure",
        to_float(ctx.get("front_pressure_psi", base.get("front_pressure_psi", 32.0)), 32.0),
        key_base="tire_component_front_psi",
    )
    rear_pressure_psi = _render_tire_pressure_input(
        p2,
        "Rear pressure",
        to_float(ctx.get("rear_pressure_psi", base.get("rear_pressure_psi", 32.0)), 32.0),
        key_base="tire_component_rear_psi",
    )
    front_weight_distribution_pct = p3.number_input(
        "Front weight distribution [%]",
        value=float(ctx.get("weight_dist_fr_pct", base.get("weight_dist_fr_pct", 50.0)) or 50.0),
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        format="%.1f",
        key="tire_component_weight_dist",
    )
    tire_improvement_pct = p4.number_input(
        "Tire improvement [%]",
        value=float(ctx.get("tire_improvement_pct", base.get("tire_improvement_pct", 0.0)) or 0.0),
        step=0.5,
        format="%.1f",
        key="tire_component_improvement",
    )

    tire_load_mass_basis = str(ctx.get("tire_load_mass_basis") or basis_default).strip().upper()
    st.caption(f"VDE calculation mass basis: `{tire_load_mass_basis}`")

    preview_result = None
    if st.button("Preview tire component", key="btn_tire_component_preview"):
        try:
            row_context = _build_tire_context_row(base_row)
            preview_result = preview_tire_roadload_from_row(
                row_context,
                {
                    "front_tire_id": front_tire_id,
                    "rear_tire_id": rear_tire_id,
                    "same_tire_front_rear": same_tire,
                    "front_pressure_psi": front_pressure_psi,
                    "rear_pressure_psi": rear_pressure_psi,
                    "front_weight_distribution_pct": front_weight_distribution_pct,
                    "tire_improvement_pct": tire_improvement_pct,
                    "tire_load_mass_basis": tire_load_mass_basis,
                    "mass_kg": ctx.get("mass_kg"),
                    "test_mass_kg": ctx.get("test_mass_kg"),
                    "inertia_class": ctx.get("inertia_class"),
                    "twc_kg": ctx.get("twc_kg"),
                    "etw_kg": ctx.get("etw_kg"),
                },
            )
            ctx["front_tire_id"] = front_tire_id
            ctx["rear_tire_id"] = rear_tire_id
            ctx["front_tire_test_code"] = front_tire_code
            ctx["rear_tire_test_code"] = rear_tire_code
            ctx["same_tire_front_rear"] = same_tire
            ctx["front_pressure_psi"] = front_pressure_psi
            ctx["rear_pressure_psi"] = rear_pressure_psi
            ctx["weight_dist_fr_pct"] = front_weight_distribution_pct
            ctx["tire_improvement_pct"] = tire_improvement_pct
            ctx["tire_load_mass_basis"] = tire_load_mass_basis
            ctx["tire_preview_result"] = preview_result
            ctx["include_tire_component"] = True
        except Exception as e:
            st.error(f"Failed to preview tire component: {e}")
            ctx["include_tire_component"] = False
            return

    if preview_result is None:
        cached = ctx.get("tire_preview_result")
        if isinstance(cached, dict) and cached.get("save_payload"):
            preview_result = cached

    if preview_result:
        _render_tire_preview_result(preview_result, saved_vde_id=saved_vde_id)
    else:
        ctx["include_tire_component"] = False
        st.info("Choose tires and preview the component to feed ABC_TOTAL from tire_db.")


def render_vehicle_basics_sidebar(*, reset_ctx):
    with st.sidebar:
        st.header("VDE Setup")
        st.caption("Scenario origin, roadload basis, mass setup, and technical configuration now live inside the main workflow.")
        st.radio(
            "Display units",
            ["Metric", "US customary"],
            horizontal=True,
            key="unit_system",
        )
        input_mode = render_vde_setup_input_mode_selector(host=st)
    return input_mode


def render_scenario_origin_section(*, reset_ctx):
    ctx = st.session_state.ctx
    mode_options = [
        "From baseline (editable)",
        "New line (manual / test)",
    ]
    labels = {
        "From baseline (editable)": "From baseline program",
        "New line (manual / test)": "New Vehicle Program",
    }

    prev_mode = ctx.get("mode", "From baseline (editable)")
    if prev_mode not in mode_options:
        prev_mode = mode_options[0]
    ctx["mode"] = st.radio(
        "Scenario Origin",
        mode_options,
        horizontal=True,
        index=mode_options.index(prev_mode),
        key="scenario_origin_radio",
        format_func=lambda value: labels.get(value, value),
    )
    if ctx["mode"] != prev_mode:
        reset_ctx(preserve_meta=True)
        st.rerun()

    if ctx["mode"] == "From baseline (editable)":
        source_options = [
            "Inherit baseline ABC_TOTAL",
            "New test ABC_TOTAL",
        ]
        option_map = {
            "Inherit baseline ABC_TOTAL": "Baseline ABC",
            "New test ABC_TOTAL": "From test coastdown",
        }
    else:
        source_options = [
            "Insert new test final ABC & define all current/walk components",
            "Calculate final ABC from components",
        ]
        option_map = {
            "Insert new test final ABC & define all current/walk components": "From test coastdown",
            "Calculate final ABC from components": "Component Build-up",
        }

    reverse_map = {value: key for key, value in option_map.items()}
    current_source_ui = str(ctx.get("abc_total_source_ui") or "").strip()
    default_source_option = reverse_map.get(current_source_ui, source_options[0])
    selected_source_option = st.radio(
        "Roadload source path",
        source_options,
        horizontal=True,
        index=source_options.index(default_source_option),
        key="scenario_source_path_radio",
    )
    ctx["abc_total_source_ui"] = option_map[selected_source_option]

    selected = ctx["abc_total_source_ui"]
    if selected == "Baseline ABC":
        ctx["from_delta"] = "Deltas"
        baseline_id = ctx.get("vde_id_parent") or ctx.get("baseline_id")
        if not baseline_id:
            st.warning("Select a baseline scenario before using inherited baseline ABC_TOTAL.")
    elif selected == "From test coastdown":
        ctx["from_delta"] = "From test"
    else:
        ctx["from_delta"] = "Change Parameters"


def render_vehicle_meta_header():
    ctx = st.session_state.ctx
    _ensure_vehicle_metadata_defaults(ctx)

    leg_opts = ["WLTP", "EPA", "ABNT (Brazil)"]

    epa_classes = [
        "Unknown", "Two Seaters", "Minicompact Cars", "Subcompact Cars", "Compact Cars",
        "Midsize Cars", "Large Cars", "Small Station Wagons", "Midsize Station Wagons",
        "Small SUVs", "Standard SUVs", "Minivans", "Vans", "Small Pickup Trucks", "Standard Pickup Trucks",
    ]
    wltp_classes = ["Class 1 (<850 kg)", "Class 2 (850-1220 kg)", "Class 3 (>1220 kg)"]
    category_list = epa_classes if ctx["legislation"] == "EPA" else wltp_classes
    category_list_upper = [category.upper() for category in category_list]

    if ctx.get("category") not in category_list_upper:
        ctx["category"] = category_list_upper[0]

    elec_opts = ["ICE", "HEV", "PHEV", "BEV"]
    trans_opts = ["AT", "AMT", "CVT", "MT", "OT"]

    r1c1, r1c2, r1c3 = st.columns([1.1, 1.6, 1.3])
    ctx["legislation"] = r1c1.selectbox(
        "Legislation",
        leg_opts,
        index=leg_opts.index(ctx["legislation"]),
        key="hdr_leg",
    )
    category_list = epa_classes if ctx["legislation"] == "EPA" else wltp_classes
    category_list_upper = [category.upper() for category in category_list]
    if ctx.get("category") not in category_list_upper:
        ctx["category"] = category_list_upper[0]
    ctx["category"] = r1c2.selectbox(
        "Category",
        category_list_upper,
        index=category_list_upper.index(ctx["category"]),
        key="hdr_cat",
    )

    default_makes = [
        "Toyota", "Honda", "Nissan", "Mitsubishi", "Mazda", "Subaru", "Hyundai", "Kia",
        "Volkswagen", "Audi", "BMW", "Mercedes-Benz", "Porsche", "Peugeot", "Renault", "Citroen",
        "Fiat", "Alfa Romeo", "Volvo", "Jaguar", "Land Rover", "Skoda", "Seat", "Opel",
        "Ford", "Chevrolet", "Dodge", "Chrysler", "Jeep", "Ram", "Cadillac", "Buick", "GMC",
        "Lincoln", "Tesla", "Suzuki", "Mini", "Smart", "Lexus", "Infiniti", "Acura",
    ]
    default_makes_upper = [make.upper() for make in default_makes]
    try:
        ensure_db()
        makes_db = db_list_makes(ctx["legislation"], ctx["category"])
        makes_db = [make.upper() for make in makes_db]
    except Exception:
        makes_db = []

    merged_makes = list(dict.fromkeys(makes_db + [make for make in default_makes_upper if make not in makes_db]))
    if "OTHER (TYPE MANUALLY)" not in merged_makes:
        merged_makes.append("OTHER (TYPE MANUALLY)")

    selected_make = str(ctx.get("make", "")).upper()
    make_choice = r1c3.selectbox(
        "Make/Brand",
        merged_makes,
        index=(merged_makes.index(selected_make) if selected_make in merged_makes else 0),
        key="hdr_make_sel",
    )
    if make_choice == "OTHER (TYPE MANUALLY)":
        ctx["make"] = r1c3.text_input("Enter custom brand", value=ctx.get("make", ""), key="hdr_make_text").upper()
    else:
        ctx["make"] = make_choice

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1.6, 0.8, 1.0, 1.0, 1.2])
    ctx["model"] = r2c1.text_input("Model/Desc.", value=ctx.get("model", ""), key="hdr_model")
    ctx["year"] = r2c2.number_input("Year", 1900, 2100, int(ctx.get("year", 2024)), step=1, key="hdr_year")
    ctx["electrification"] = r2c3.selectbox(
        "Electrification",
        elec_opts,
        index=elec_opts.index(ctx.get("electrification", "ICE")),
        key="hdr_elec",
    )
    ctx["transmission_type"] = r2c4.selectbox(
        "Transmission",
        trans_opts,
        index=trans_opts.index(ctx.get("transmission_type", "AT")),
        key="hdr_trans",
    )
    ctx["notes"] = r2c5.text_input("Proposal / Scenario", value=ctx.get("notes", ""), key="hdr_notes")

    st.caption("Required for Vehicle Data = OK: legislation, category, make, model, year, electrification, and transmission.")


def render_vehicle_metadata_spreadsheet_section() -> None:
    ctx = st.session_state.ctx
    st.caption("Vehicle Info workbook. Baseline values can stay inherited field-by-field while you tailor the scenario metadata.")
    proposal_cols = st.columns([1.6, 1.0])
    ctx["notes"] = proposal_cols[0].text_input(
        "Proposal / Scenario",
        value=str(ctx.get("notes") or ""),
        key="vde_vehicle_info_proposal",
        help="Required for save. For baseline-derived scenarios, use a scenario-specific label different from the inherited baseline notes.",
    )
    baseline_notes = str(_metadata_baseline_value(ctx, "proposal") or "").strip()
    proposal_source = _metadata_source_label(ctx, "proposal", ctx.get("notes"))
    proposal_cols[1].metric("Proposal source", proposal_source)
    if baseline_notes:
        st.caption(f"Inherited baseline notes: {baseline_notes}")
    vehicle_df = _build_vehicle_scenario_spreadsheet_df(ctx)
    header = st.columns([1.2, 1.25, 1.0, 0.9, 1.3])
    header[0].caption("field")
    header[1].caption("value")
    header[2].caption("source")
    header[3].caption("inherit baseline")
    header[4].caption("notes")

    edited_rows: list[dict] = []
    active_legislation = str(ctx.get("legislation") or "WLTP").strip().upper()
    for row in vehicle_df.to_dict(orient="records"):
        field_name = str(row.get("field") or "")
        baseline_value = _metadata_baseline_value(ctx, field_name)
        can_inherit = field_name != "proposal" and baseline_value not in (None, "", [])
        widget_key = f"vde_vehicle_info_{field_name}"
        value_key = f"{widget_key}_value"
        inherit_key = f"{widget_key}_inherit"
        if can_inherit:
            inherit_value = bool(st.session_state.get(inherit_key, row.get("inherit")))
            st.session_state[inherit_key] = inherit_value
        else:
            inherit_value = False
            st.session_state.pop(inherit_key, None)

        if inherit_value and can_inherit:
            st.session_state[value_key] = baseline_value
        elif value_key not in st.session_state:
            st.session_state[value_key] = row.get("value")
        resolved_value = st.session_state.get(value_key)

        c1, c2, c3, c4, c5 = st.columns([1.2, 1.25, 1.0, 0.9, 1.3])
        c1.write(field_name)

        if field_name == "vehicle_label":
            c2.write(str(resolved_value or "-"))
        elif field_name == "model_year":
            year_default = int(to_float(resolved_value, 2024) or 2024)
            resolved_value = c2.number_input(
                "model_year",
                min_value=1900,
                max_value=2100,
                value=year_default,
                step=1,
                key=f"{widget_key}_value",
                label_visibility="collapsed",
                disabled=inherit_value,
            )
        else:
            current_value = str(resolved_value or "").strip()
            options = _metadata_choice_options(field_name, legislation=active_legislation, current_value=current_value)
            if options is not None:
                normalized_value = current_value if field_name == "legislation" else current_value.upper()
                if normalized_value not in options:
                    options = options + [normalized_value] if normalized_value else options
                if not normalized_value and options:
                    normalized_value = str(options[0])
                resolved_value = c2.selectbox(
                    field_name,
                    options,
                    index=options.index(normalized_value),
                    key=f"{widget_key}_value",
                    label_visibility="collapsed",
                    disabled=inherit_value,
                )
            else:
                resolved_value = c2.text_input(
                    field_name,
                    value=current_value,
                    key=f"{widget_key}_value",
                    label_visibility="collapsed",
                    disabled=inherit_value,
                )

        c3.write(str(row.get("source") or ""))
        if can_inherit:
            inherit_value = c4.checkbox(
                field_name,
                value=inherit_value,
                key=f"{widget_key}_inherit",
                label_visibility="collapsed",
            )
        else:
            c4.write("-")
        c5.caption(str(row.get("notes") or ""))

        if field_name == "legislation":
            active_legislation = str(baseline_value if inherit_value and can_inherit else resolved_value or active_legislation).strip().upper()

        edited_row = dict(row)
        edited_row["inherit"] = inherit_value
        edited_row["value"] = resolved_value
        edited_rows.append(edited_row)

    edited_df = pd.DataFrame(edited_rows)
    errors = _apply_vehicle_scenario_spreadsheet_changes(edited_df)
    for error in errors:
        st.warning(error)


def render_scenario_origin_spreadsheet_section(*, reset_ctx) -> None:
    ctx = st.session_state.ctx
    render_scenario_origin_section(reset_ctx=reset_ctx)
    if ctx["mode"] == "From baseline (editable)":
        render_baseline_picker_and_editor_panel()
    else:
        st.info("Manual/test origin is active. This scenario will be built from the current workbook state without loading a baseline snapshot.")


def render_mass_spreadsheet_section(*, prefill=None) -> None:
    ctx = st.session_state.ctx
    snapshot = _mass_snapshot(ctx, prefill=prefill)
    legislation_mode = str(snapshot.get("regulatory_mode") or "CUSTOM").upper()
    active_cycle = str(ctx.get("cycle_name") or "Pending")
    st.caption("Active cycle: " + active_cycle + ". Edit cycle in `Cycle & Preview`.")

    ctx["mass_profile_custom_fuelcons_basis"] = st.selectbox(
        "Custom fuelcons mass basis",
        ["TEST_MASS", "TWC"],
        index=["TEST_MASS", "TWC"].index(str(ctx.get("mass_profile_custom_fuelcons_basis") or "TEST_MASS").upper() if str(ctx.get("mass_profile_custom_fuelcons_basis") or "TEST_MASS").upper() in {"TEST_MASS", "TWC"} else "TEST_MASS"),
        key="mass_profile_custom_fuelcons_basis_selector",
    )

    st.caption("Baseline Mass Reference")
    reference_df = _build_baseline_mass_reference_df(snapshot)
    with st.container(border=True):
        reference_edit = st.data_editor(
            reference_df,
            key="vde_mass_reference_editor",
            hide_index=True,
            use_container_width=True,
            height=min(420, 44 + (len(reference_df) * 35)),
            disabled=["field", "unit", "source", "status", "notes"],
            column_config={
                "field": st.column_config.TextColumn("field"),
                "value": st.column_config.TextColumn("value"),
                "unit": st.column_config.TextColumn("unit"),
                "source": st.column_config.TextColumn("source"),
                "status": st.column_config.TextColumn("status"),
                "notes": st.column_config.TextColumn("notes"),
            },
        )
    _apply_baseline_mass_reference_changes(reference_edit)
    snapshot = _mass_snapshot(ctx, prefill=prefill)

    custom_default = to_float(snapshot.get("custom_mass"), snapshot.get("proposal_curb_mass"))
    ctx["mass_profile_custom_input_kg"] = quantity_input(
        st,
        "Custom profile mass",
        to_float(custom_default, 0.0),
        "mass",
        key="mass_profile_custom_input",
        min_canonical=0.0,
        max_canonical=5000.0,
        step_canonical=1.0,
        format_str="%.1f",
    )
    snapshot = _mass_snapshot(ctx, prefill=prefill)

    st.caption("Mass Scenario Profiles")
    default_profile = str(ctx.get("mass_profile_selected") or _default_mass_profile(snapshot))
    profiles_df = _build_mass_profiles_df(snapshot, default_profile)
    with st.container(border=True):
        profiles_edit = st.data_editor(
            profiles_df,
            key="vde_mass_profiles_editor",
            hide_index=True,
            use_container_width=True,
            height=min(460, 44 + (len(profiles_df) * 35)),
            disabled=["profile", "rule", "vde_mass_kg", "fuelcons_mass_basis", "fuelcons_mass_kg", "required_inputs", "status", "notes"],
            column_config={
                "profile": st.column_config.TextColumn("profile"),
                "enabled": st.column_config.CheckboxColumn("enabled"),
                "rule": st.column_config.TextColumn("rule"),
                "vde_mass_kg": st.column_config.TextColumn("vde_mass_kg"),
                "fuelcons_mass_basis": st.column_config.TextColumn("fuelcons_mass_basis"),
                "fuelcons_mass_kg": st.column_config.TextColumn("fuelcons_mass_kg"),
                "required_inputs": st.column_config.TextColumn("required_inputs"),
                "status": st.column_config.TextColumn("status"),
                "notes": st.column_config.TextColumn("notes"),
            },
        )
    selected_profile = _resolve_enabled_mass_profile(profiles_edit, default_profile=default_profile)
    enabled_count = int(pd.to_numeric(profiles_edit["enabled"], errors="coerce").fillna(False).astype(bool).sum()) if profiles_edit is not None and not profiles_edit.empty else 0
    if enabled_count > 1:
        st.warning("Only one mass scenario profile can stay active. The last enabled row was kept as the selected profile.")

    selected_calc = _apply_mass_profile_selection(ctx, snapshot, selected_profile)
    render_vde_workbook_table(
        _build_selected_mass_calculation_df(selected_calc),
        title="Selected Scenario Calculation",
        table_id="selected-scenario-calculation",
    )
    if str(selected_calc.get("fuelcons_mass_basis") or "").upper() == "TWC":
        st.info("Fuelcons mass basis changes require VDE recalculation at the selected mass before fuel/CO2 conversion.")

    with st.expander("Advanced Mass Audit", expanded=False):
        audit_df = _build_mass_audit_df(ctx)
        render_vde_workbook_table(
            audit_df,
            title="Advanced Mass Audit",
            table_id="advanced-mass-audit",
        )
        if legislation_mode == "EPA":
            st.caption("WLTP-only rows remain in the audit as not applicable.")


def render_cycle_spreadsheet_section() -> None:
    ctx = st.session_state.ctx
    st.caption("Cycle selection and upload stay here so the drive trace is managed independently from vehicle mass.")
    cycle_distance_km = _cycle_distance_km(ctx)
    s1, s2, s3 = st.columns(3)
    s1.metric("Cycle name", str(ctx.get("cycle_name") or "Pending"))
    s2.metric("Cycle source", str(ctx.get("cycle_source") or ("loaded" if ctx.get("cycle_df") is not None else "Pending")))
    s3.metric(
        f"Cycle distance [{'km' if _current_unit_system() == 'Metric' else 'mi'}]",
        "-" if cycle_distance_km is None else f"{(cycle_distance_km if _current_unit_system() == 'Metric' else cycle_distance_km * 0.621371192237):.2f}",
    )
    render_cycle_section(include_preview_snapshot=False)


def render_vde_setup_spreadsheet_workbook(
    *,
    defaults_df_getter,
    defaults_path,
    reset_ctx,
    roadload_base_row: dict | None = None,
    roadload_saved_vde_id: int | None = None,
    roadload_transmission_prefill: dict | None = None,
) -> None:
    ctx = st.session_state.ctx
    _ensure_scenario_workbook_state(ctx)
    signature = _spreadsheet_source_signature(ctx)
    previous_signature = st.session_state.get("vde_spreadsheet_source_signature")
    st.session_state["vde_spreadsheet_source_signature"] = signature
    if previous_signature is not None and previous_signature != signature:
        _reset_scenario_workbook_widget_state(ctx)
        st.info("Active VDE source changed. Spreadsheet inputs were refreshed from the selected source.")

    _apply_scenario_workbook_state_to_ctx(ctx, str(ctx.get("scenario_workbook_active_column") or "walked_1"))
    st.caption("Scenario Workbook: compare Baseline with walked requests, inherit with Walk From, and preview the effective snapshot before save.")
    _render_scenario_workbook_matrix(ctx)

    workbook_sections = [
        "Scenario Origin / Data & Metadata",
        "Mass & Aero",
        "Tire",
        "Transmission",
        "Brake",
        "Axle & Hubs",
        "Parasitic Losses",
        "Preview & Save",
        "Technical Audit",
    ]
    current_section = str(ctx.get("vde_workbook_section") or workbook_sections[0])
    if current_section not in workbook_sections:
        current_section = workbook_sections[0]
    ctx["vde_workbook_section"] = st.radio(
        "Workbook section",
        workbook_sections,
        horizontal=True,
        index=workbook_sections.index(current_section),
        key="vde_workbook_section_selector",
        label_visibility="collapsed",
    )

    baseline_prefill = dict(ctx.get("selected_baseline_row") or ctx.get("baseline_dict") or {}) if ctx["mode"] == "From baseline (editable)" else None

    if ctx["vde_workbook_section"] == "Scenario Origin / Data & Metadata":
        render_step_header(1, "Scenario Origin / Data & Metadata", "Pick the baseline, confirm the walked column inheritance path, then define the vehicle metadata and cycle for the selected walked request.")
        _render_scenario_origin_metadata_menu(reset_ctx=reset_ctx)

    elif ctx["vde_workbook_section"] == "Mass & Aero":
        render_step_header(2, "Mass & Aero", "Resolve the active walked column mass intention, trailer context, and aero proposal in one place.")
        _render_mass_aero_workbook_menu(prefill=baseline_prefill)

    elif ctx["vde_workbook_section"] == "Tire":
        render_step_header(3, "Tire", "Tire treatment stays scenario-aware: inherit, lookup, or apply a walked change and keep the preview explicit.")
        _render_tire_workbook_menu(base_row=roadload_base_row, saved_vde_id=roadload_saved_vde_id)

    elif ctx["vde_workbook_section"] == "Transmission":
        render_step_header(4, "Transmission", "Transmission Losses remain the explicit TOTAL -> NET bridge for the selected walked column.")
        render_transmission_losses_section(prefill=roadload_transmission_prefill)

    elif ctx["vde_workbook_section"] == "Brake":
        render_step_header(5, "Brake", "Brake drag stays separate from the other losses, and the workbook now tracks residual torque fields with it.")
        _render_brake_workbook_menu(base_row=roadload_base_row)

    elif ctx["vde_workbook_section"] == "Axle & Hubs":
        render_step_header(6, "Axle & Hubs", "Track axle and hub losses with the same inherit / absolute / delta pattern used elsewhere in the workbook.")
        _render_axle_hub_workbook_menu()

    elif ctx["vde_workbook_section"] == "Parasitic Losses":
        render_step_header(7, "Parasitic Losses", "Parasitics stay independent from Transmission Losses so the build-up remains legible.")
        _render_parasitic_workbook_menu(base_row=roadload_base_row)

    elif ctx["vde_workbook_section"] == "Preview & Save":
        render_step_header(8, "Preview & Save", "Preview remains the source of truth. Save/update still uses the existing workflow payload and persistence path.")
        render_vde_workbook_table(
            _build_scenario_preview_matrix_df(ctx),
            title="Scenario Preview Matrix",
            table_id="scenario-preview-matrix-legacy",
        )
        action_cols = st.columns(3)
        action_cols[0].button("Compute selected column", disabled=True, help="The live preview already computes the selected walked column continuously.")
        action_cols[1].button("Save all ready columns", disabled=True, help="Future workflow once multi-column persistence is hardened.")
        if action_cols[2].button("Open Technical Audit", key="open_scenario_workbook_audit"):
            ctx["vde_workbook_section"] = "Technical Audit"
            st.rerun()
        st.divider()
        render_vde_results_review_panel()
        st.divider()
        render_compute_and_save_panel(
            defaults_df_getter=defaults_df_getter,
            reset_ctx=reset_ctx,
        )
        st.divider()
        render_vde_edit_delete_panel(
            defaults_path=defaults_path,
            defaults_df_getter=defaults_df_getter,
            reset_ctx=reset_ctx,
        )

    elif ctx["vde_workbook_section"] == "Technical Audit":
        render_step_header(9, "Technical Audit", "Inspect raw vs effective workbook values, inheritance source, and workflow provenance before persisting the selected walked column.")
        render_vde_setup_technical_audit()

    render_vde_setup_legacy_footer(
        defaults_df_getter=defaults_df_getter,
        defaults_path=defaults_path,
        reset_ctx=reset_ctx,
        roadload_base_row=roadload_base_row,
        roadload_saved_vde_id=roadload_saved_vde_id,
        roadload_transmission_prefill=roadload_transmission_prefill,
    )

    if _capture_active_scenario_workbook_direct(ctx):
        st.rerun()


def render_vde_setup_technical_audit() -> None:
    ctx = st.session_state.ctx
    preview = _safe_workflow_preview(ctx)
    st.dataframe(_build_scenario_workbook_audit_df(ctx), use_container_width=True, hide_index=True)
    with st.expander("Workflow provenance", expanded=False):
        with st.expander("Demand source audit", expanded=False):
            st.write(preview.get("line_source") or {"scenario_origin": _line_source_summary(ctx)})
        with st.expander("Mass audit", expanded=False):
            st.write(preview.get("mass_setup") or {})
        with st.expander("Roadload ABC audit", expanded=False):
            st.write(
                {
                    "initial_abc_total_base": preview.get("initial_abc_total_base"),
                    "component_abc_total": preview.get("component_abc_total"),
                    "abc_total": preview.get("abc_total"),
                    "abc_net": preview.get("abc_net"),
                }
            )
        with st.expander("Transmission loss audit", expanded=False):
            st.write(preview.get("transmission_losses") or {})
        with st.expander("Component build-up audit", expanded=False):
            components = list(preview.get("components") or [])
            if components:
                st.dataframe(pd.DataFrame(components), use_container_width=True, hide_index=True)
            else:
                st.caption("No component build-up records are active.")
        with st.expander("Save payload / provenance", expanded=False):
            st.write(preview.get("save_payload") or {})


def render_vde_setup_legacy_footer(
    *,
    defaults_df_getter,
    defaults_path,
    reset_ctx,
    roadload_base_row: dict | None = None,
    roadload_saved_vde_id: int | None = None,
    roadload_transmission_prefill: dict | None = None,
) -> None:
    ctx = st.session_state.ctx
    legacy_owner = {
        "Scenario Origin": "Scenario Origin / Data & Metadata",
        "Vehicle Info": "Scenario Origin / Data & Metadata",
        "Cycle & Preview": "Scenario Origin / Data & Metadata",
        "Mass Setup": "Mass & Aero",
        "Roadload ABC": "Mass & Aero",
        "Transmission Losses": "Transmission",
        "Component Build-up": "Tire",
        "Brake Editor": "Brake",
        "Parasitic Editor": "Parasitic Losses",
        "Results / Save": "Preview & Save",
    }
    legacy_options = [
        "Hidden",
        "Scenario Origin",
        "Vehicle Info",
        "Cycle & Preview",
        "Mass Setup",
        "Roadload ABC",
        "Transmission Losses",
        "Component Build-up",
        "Brake Editor",
        "Parasitic Editor",
        "Results / Save",
    ]

    st.divider()
    with st.expander("Legacy Sections", expanded=False):
        st.caption("Temporary footer for original VDE Setup panels that are being replaced by the workbook flow above.")
        selected = st.selectbox(
            "Legacy panel",
            legacy_options,
            index=legacy_options.index(str(st.session_state.get("vde_legacy_footer_panel") or "Hidden")) if str(st.session_state.get("vde_legacy_footer_panel") or "Hidden") in legacy_options else 0,
            key="vde_legacy_footer_panel",
        )
        if selected == "Hidden":
            st.caption("Legacy panels stay out of the main workflow and only remain here while the new workbook replaces them section by section.")
            return

        owner = legacy_owner.get(selected)
        if owner and str(ctx.get("vde_workbook_section") or "") == owner:
            st.info("This legacy panel overlaps the active workbook section above. Switch the main workbook section first if you want to compare the legacy view.")
            return

        st.warning(f"Legacy panel: {selected}. This area is temporary and will be removed after the workbook fully replaces the original flow.")

        if selected == "Scenario Origin":
            render_step_header(0, "Legacy Scenario Origin", "Original baseline/manual scenario-origin block kept temporarily at the footer.")
            render_scenario_origin_spreadsheet_section(reset_ctx=reset_ctx)
        elif selected == "Vehicle Info":
            render_step_header(0, "Legacy Vehicle Info", "Original vehicle metadata spreadsheet kept temporarily at the footer.")
            render_vehicle_metadata_spreadsheet_section()
        elif selected == "Cycle & Preview":
            render_step_header(0, "Legacy Cycle & Preview", "Original cycle block kept temporarily at the footer.")
            render_cycle_spreadsheet_section()
        elif selected == "Mass Setup":
            render_step_header(0, "Legacy Mass Setup", "Original mass spreadsheet kept temporarily at the footer.")
            render_mass_spreadsheet_section(
                prefill=dict(ctx.get("selected_baseline_row") or ctx.get("baseline_dict") or {}) if ctx["mode"] == "From baseline (editable)" else None
            )
        elif selected == "Roadload ABC":
            render_step_header(0, "Legacy Roadload ABC", "Original roadload block kept temporarily at the footer.")
            render_aero_section(prefill=dict(ctx.get("selected_baseline_row") or ctx.get("baseline_dict") or {}))
            st.divider()
            render_from_test_section()
            with st.expander("Advanced NET auxiliary estimate", expanded=False):
                render_auxiliaries_section(defaults_df_getter=defaults_df_getter)
        elif selected == "Transmission Losses":
            render_step_header(0, "Legacy Transmission Losses", "Original transmission-loss editor kept temporarily at the footer.")
            render_transmission_losses_section(prefill=roadload_transmission_prefill)
        elif selected == "Component Build-up":
            render_step_header(0, "Legacy Component Build-up", "Original component build-up panel kept temporarily at the footer.")
            render_component_build_up_panel(
                base_row=roadload_base_row,
                saved_vde_id=roadload_saved_vde_id,
            )
        elif selected == "Brake Editor":
            render_step_header(0, "Legacy Brake Editor", "Original brake component editor kept temporarily at the footer.")
            _render_brake_component_editor(base_row=roadload_base_row)
        elif selected == "Parasitic Editor":
            render_step_header(0, "Legacy Parasitic Editor", "Original parasitic component editor kept temporarily at the footer.")
            _render_parasitic_component_editor(base_row=roadload_base_row)
        elif selected == "Results / Save":
            render_step_header(0, "Legacy Results / Save", "Original preview/save blocks kept temporarily at the footer.")
            render_vde_results_review_panel()
            st.divider()
            render_compute_and_save_panel(
                defaults_df_getter=defaults_df_getter,
                reset_ctx=reset_ctx,
            )
            st.divider()
            render_vde_edit_delete_panel(
                defaults_path=defaults_path,
                defaults_df_getter=defaults_df_getter,
                reset_ctx=reset_ctx,
            )


VDE_WORKBOOK_V2_COLUMNS = ("baseline", "walked_1", "walked_2")
VDE_WORKBOOK_V2_LABELS = {
    "baseline": "Baseline",
    "walked_1": "Walked #1",
    "walked_2": "Walked #2",
}
VDE_WORKBOOK_V2_DEFAULT_SCENARIOS = [
    {"key": "baseline", "label": "Baseline", "role": "baseline"},
    {"key": "walked_1", "label": "Walked #1", "role": "walked"},
    {"key": "walked_2", "label": "Walked #2", "role": "walked"},
]
VDE_WORKBOOK_V2_MENUS = [
    "Scenario Workbook",
    "Mass & Aero",
    "Tire",
    "Transmission",
    "Brake",
    "Axle & Hubs",
    "Parasitic Losses",
    "Preview & Save",
    "Technical Audit",
]

VDE_WORKBOOK_V2_MATRIX_SPECS = [
    {"id": "line_source", "label": "Line source", "kind": "text", "notes": "Existing DB or new line"},
    {"id": "vde_id", "label": "VDE-ID", "kind": "readonly", "notes": "Saved id or pending"},
    {"id": "baseline_selector", "label": "Baseline selector", "kind": "text", "notes": "Only for baseline"},
    {"id": "description", "label": "Description", "kind": "text", "notes": "Scenario label"},
    {"id": "status", "label": "Status", "kind": "readonly", "notes": "Calculated"},
    {"id": "walk_from", "label": "Walk From", "kind": "text", "notes": "Required for walked"},
    {"id": "proposal_direct", "label": "Proposal Direct", "kind": "text", "notes": "Direct change label"},
    {"id": "proposal_effective", "label": "Proposal Effective", "kind": "readonly", "notes": "Calculated"},
    {"id": "legislation", "label": "Legislation", "kind": "text", "notes": "Inherited if blank"},
    {"id": "model_year", "label": "Model Year", "kind": "int", "notes": "Inherited if blank"},
    {"id": "make", "label": "Make", "kind": "text", "notes": "Inherited if blank"},
    {"id": "model", "label": "Model", "kind": "text", "notes": "Inherited if blank"},
    {"id": "cycle", "label": "Cycle", "kind": "text", "notes": "Inherited if blank"},
    {"id": "display_units", "label": "Display Units", "kind": "text", "notes": "Inherited if blank"},
    {"id": "roadload_source_type", "label": "Roadload Source Type", "kind": "text", "notes": "Source mode"},
    {"id": "save_target", "label": "Save / Update Target", "kind": "text", "notes": "Selected save target"},
    {"id": "scenario_notes", "label": "Notes", "kind": "text", "notes": "Optional notes"},
]

VDE_WORKBOOK_V2_SECTION_SPECS = {
    "Mass & Aero": [
        {"id": "mass_intention", "label": "mass_intention", "kind": "select", "options": ["INHERIT", "EPA_STATUS", "EPA_PLUS_1_TWC", "WLTP_TML", "WLTP_TMH", "GVWR", "GCWR", "PERF_CURB_100KG", "PERF_CURB_300LB", "CUSTOM", "FROM_BASELINE", "NEW_TEST_MASS", "MANUAL"], "ctx_key": "mass_intention", "notes": "Controls required mass inputs"},
        {"id": "curb_mass_kg", "label": "Base / curb mass [kg]", "kind": "mass", "ctx_key": "mass_kg", "notes": "Base curb mass"},
        {"id": "test_mass_kg", "label": "Resolved VDE test mass [kg]", "kind": "mass", "ctx_key": "test_mass_kg", "notes": "Resolved final test mass used in VDE"},
        {"id": "inertia_class", "label": "EPA ETW / TWC [kg]", "kind": "mass", "ctx_key": "inertia_class", "notes": "Legacy EPA Equivalent Test Weight / TWC proxy"},
        {"id": "prep_inertia_class", "label": "prep_inertia_class", "kind": "text", "ctx_key": "prep_inertia_class", "notes": "Free text audit field"},
        {"id": "TWC_kg", "label": "TWC_kg", "kind": "mass", "ctx_key": "twc_kg", "notes": "EPA TWC / inertia reference"},
        {"id": "TML_kg", "label": "WLTP TML [kg]", "kind": "mass", "ctx_key": "test_mass_low_kg", "notes": "WLTP test mass low"},
        {"id": "TMH_kg", "label": "WLTP TMH [kg]", "kind": "mass", "ctx_key": "test_mass_high_kg", "notes": "WLTP test mass high"},
        {"id": "fr_weight_pct", "label": "fr_weight_pct", "kind": "float", "ctx_key": "weight_dist_fr_pct", "notes": "Front weight distribution [%]"},
        {"id": "payload_kg", "label": "payload_kg", "kind": "mass", "ctx_key": "payload_kg", "notes": "Payload"},
        {"id": "payload_display_kg", "label": "GVWR payload display [kg]", "kind": "mass", "readonly": True, "notes": "Calculated as GVWR - curb mass when applicable"},
        {"id": "GVWR_kg", "label": "GVWR [kg]", "kind": "mass", "ctx_key": "mass_profile_gvwr_kg", "notes": "Required only for GVWR mode"},
        {"id": "GCWR_kg", "label": "GCWR [kg]", "kind": "mass", "ctx_key": "mass_profile_gcwr_kg", "notes": "Required only for GCWR mode"},
        {"id": "trailer_code", "label": "trailer_code", "kind": "select", "options": ["", "TRAILER_LIGHT", "TRAILER_BOX", "TRAILER_HEAVY", "MANUAL"], "ctx_key": "trailer_code", "notes": "Mock preset or manual"},
        {"id": "trailer_weight_kg", "label": "Trailer mass [kg]", "kind": "mass", "ctx_key": "mass_profile_trailer_mass_kg", "notes": "Trailer mass for GCWR scenarios"},
        {"id": "vehicle_mass_at_gcwr", "label": "Vehicle mass at GCWR [kg]", "kind": "mass", "readonly": True, "notes": "Calculated as GCWR - trailer mass"},
        {"id": "trailer_roadload_source", "label": "trailer_roadload_source", "kind": "select", "options": ["None", "Trailer DB", "Manual ABC"], "ctx_key": "trailer_roadload_source", "notes": "Trailer roadload source"},
        {"id": "trailer_roadload_status", "label": "trailer_roadload_status", "kind": "text", "readonly": True, "notes": "Trailer roadload completeness status"},
        {"id": "trailer_A", "label": "trailer_A", "kind": "force", "ctx_key": "trailer_A", "notes": "Trailer A"},
        {"id": "trailer_B", "label": "trailer_B", "kind": "force_per_speed", "ctx_key": "trailer_B", "notes": "Trailer B"},
        {"id": "trailer_C", "label": "trailer_C", "kind": "force_per_speed_squared", "ctx_key": "trailer_C", "notes": "Trailer C"},
        {"id": "aero_mode", "label": "aero_mode", "kind": "select", "options": ["Inherit", "Absolute", "Delta"], "ctx_key": "aero_calculation_mode", "notes": "Explicit aero mode"},
        {"id": "Cd", "label": "Cd", "kind": "float", "ctx_key": "cd", "notes": "Aerodynamic coefficient"},
        {"id": "frontal_area_m2", "label": "frontal_area_m2", "kind": "float", "ctx_key": "frontal_area_m2", "notes": "Frontal area"},
        {"id": "CdA", "label": "CdA", "kind": "float", "ctx_key": "cda_m2", "notes": "Explicit CdA overrides Cd * area"},
        {"id": "ABC_TOTAL_A", "label": "ABC_TOTAL_A", "kind": "force", "ctx_key": "A", "notes": "Primary roadload A"},
        {"id": "ABC_TOTAL_B", "label": "ABC_TOTAL_B", "kind": "force_per_speed", "ctx_key": "B", "notes": "Primary roadload B"},
        {"id": "ABC_TOTAL_C", "label": "ABC_TOTAL_C", "kind": "force_per_speed_squared", "ctx_key": "C", "notes": "Primary roadload C"},
        {"id": "effective_test_mass_kg", "label": "effective_test_mass_kg", "kind": "mass", "ctx_key": "test_mass_kg", "readonly": True, "notes": "Calculated effective test mass"},
        {"id": "vde_mass_basis", "label": "Test mass basis", "kind": "text", "ctx_key": "test_mass_basis", "readonly": True, "notes": "Resolved VDE mass basis"},
        {"id": "fuelcons_mass_basis", "label": "fuelcons_mass_basis", "kind": "text", "readonly": True, "notes": "Resolved fuel/consumption mass basis"},
        {"id": "mass_rule_status", "label": "Mass rule status", "kind": "text", "readonly": True, "notes": "Resolved mass rule status"},
        {"id": "mass_rule_notes", "label": "Mass rule notes", "kind": "text", "readonly": True, "ctx_key": "mass_rule_notes", "notes": "Resolved mass rule notes"},
    ],
    "Tire": [
        {"id": "tire_mode", "label": "tire_mode", "kind": "text", "ctx_key": "tire_mode_v2", "notes": "Inherit / Absolute / Delta / SMERF / Lookup"},
        {"id": "tire_code", "label": "tire_code", "kind": "text", "ctx_key": "tire_code", "notes": "Optional tire code"},
        {"id": "tire_db_id", "label": "tire_db_id", "kind": "int", "ctx_key": "tire_db_id", "notes": "Optional Tire DB id"},
        {"id": "tire_size", "label": "tire_size", "kind": "text", "ctx_key": "tire_size", "notes": "Tire size"},
        {"id": "psi_front", "label": "psi_front", "kind": "float", "ctx_key": "front_pressure_psi", "notes": "Front pressure"},
        {"id": "psi_rear", "label": "psi_rear", "kind": "float", "ctx_key": "rear_pressure_psi", "notes": "Rear pressure"},
        {"id": "SMERF", "label": "SMERF", "kind": "float", "ctx_key": "smerf", "notes": "Optional audit value"},
        {"id": "rrc_N_per_kN", "label": "rrc_N_per_kN", "kind": "rrc", "ctx_key": "rrc_N_per_kN", "notes": "Rolling resistance coefficient"},
        {"id": "tire_A", "label": "tire_A", "kind": "force", "ctx_key": "tire_A_final", "notes": "Effective tire A"},
        {"id": "tire_B", "label": "tire_B", "kind": "force_per_speed", "ctx_key": "tire_B_final", "notes": "Effective tire B"},
        {"id": "tire_C", "label": "tire_C", "kind": "force_per_speed_squared", "ctx_key": "tire_C_final", "notes": "Effective tire C"},
        {"id": "tire_delta_A", "label": "delta_A", "kind": "force", "ctx_key": "tire_delta_A_v2", "notes": "Optional delta A"},
        {"id": "tire_delta_B", "label": "delta_B", "kind": "force_per_speed", "ctx_key": "tire_delta_B_v2", "notes": "Optional delta B"},
        {"id": "tire_delta_C", "label": "delta_C", "kind": "force_per_speed_squared", "ctx_key": "tire_delta_C_v2", "notes": "Optional delta C"},
        {"id": "improvement_pct", "label": "improvement_pct", "kind": "float", "ctx_key": "tire_improvement_pct", "notes": "Improvement percentage"},
        {"id": "tire_source", "label": "source", "kind": "text", "ctx_key": "tire_source_v2", "notes": "tire_db / manual / inherit"},
        {"id": "tire_notes", "label": "notes", "kind": "text", "ctx_key": "tire_notes_v2", "notes": "Tire notes"},
    ],
    "Transmission": [
        {"id": "trans_loss_mode", "label": "trans_loss_mode", "kind": "text", "ctx_key": "trans_loss_mode_v2", "notes": "Inherit / Absolute / Delta / Not available"},
        {"id": "trans_A_loss", "label": "trans_A_loss", "kind": "force", "ctx_key": "trans_A_coef_N", "notes": "Transmission A loss"},
        {"id": "trans_B_loss", "label": "trans_B_loss", "kind": "force_per_speed", "ctx_key": "trans_B_coef_Npkph", "notes": "Transmission B loss"},
        {"id": "trans_C_loss", "label": "trans_C_loss", "kind": "force_per_speed_squared", "ctx_key": "trans_C_coef_Npkph2", "notes": "Transmission C loss"},
        {"id": "trans_delta_A_loss", "label": "delta_A_loss", "kind": "force", "ctx_key": "trans_delta_A_v2", "notes": "Optional delta A"},
        {"id": "trans_delta_B_loss", "label": "delta_B_loss", "kind": "force_per_speed", "ctx_key": "trans_delta_B_v2", "notes": "Optional delta B"},
        {"id": "trans_delta_C_loss", "label": "delta_C_loss", "kind": "force_per_speed_squared", "ctx_key": "trans_delta_C_v2", "notes": "Optional delta C"},
        {"id": "neutral_drag_source", "label": "neutral_drag_source", "kind": "text", "ctx_key": "neutral_drag_source", "notes": "Reference source"},
        {"id": "trans_vde_source_id", "label": "vde_source_id", "kind": "int", "ctx_key": "trans_vde_source_id", "notes": "Optional linked VDE id"},
        {"id": "trans_component_source_id", "label": "component_source_id", "kind": "int", "ctx_key": "trans_component_source_id", "notes": "Optional component id"},
        {"id": "net_available", "label": "net_available", "kind": "text", "ctx_key": "trans_net_available_v2", "notes": "Computed / expected flag"},
        {"id": "trans_source", "label": "source", "kind": "text", "ctx_key": "transmission_losses_source", "notes": "baseline / manual / missing"},
        {"id": "trans_notes", "label": "notes", "kind": "text", "ctx_key": "trans_notes_v2", "notes": "Transmission notes"},
    ],
    "Brake": [
        {"id": "brake_mode", "label": "brake_mode", "kind": "text", "ctx_key": "brake_mode_v2", "notes": "Inherit / Absolute / Delta / Not used"},
        {"id": "brake_A", "label": "brake_A", "kind": "force", "ctx_key": "brake_A_coef_N", "notes": "Brake A"},
        {"id": "brake_B", "label": "brake_B", "kind": "force_per_speed", "ctx_key": "brake_B_Npkph", "notes": "Brake B"},
        {"id": "brake_C", "label": "brake_C", "kind": "force_per_speed_squared", "ctx_key": "brake_C_coef_Npkph2", "notes": "Brake C"},
        {"id": "brake_delta_A", "label": "delta_A", "kind": "force", "ctx_key": "brake_delta_A_v2", "notes": "Optional delta A"},
        {"id": "brake_delta_B", "label": "delta_B", "kind": "force_per_speed", "ctx_key": "brake_delta_B_v2", "notes": "Optional delta B"},
        {"id": "brake_delta_C", "label": "delta_C", "kind": "force_per_speed_squared", "ctx_key": "brake_delta_C_v2", "notes": "Optional delta C"},
        {"id": "residual_torque_front_Nm", "label": "residual_torque_front_Nm", "kind": "float", "ctx_key": "residual_torque_front_Nm", "notes": "Residual torque front"},
        {"id": "residual_torque_rear_Nm", "label": "residual_torque_rear_Nm", "kind": "float", "ctx_key": "residual_torque_rear_Nm", "notes": "Residual torque rear"},
        {"id": "residual_torque_total_Nm", "label": "residual_torque_total_Nm", "kind": "float", "ctx_key": "residual_torque_total_Nm", "notes": "Residual torque total"},
        {"id": "wheel_radius_m", "label": "wheel_radius_m", "kind": "float", "ctx_key": "wheel_radius_m", "notes": "Wheel radius"},
        {"id": "brake_drag_force_N", "label": "brake_drag_force_N", "kind": "float", "ctx_key": "brake_drag_force_N", "notes": "Brake drag force"},
        {"id": "brake_temp_condition", "label": "brake_temp_condition", "kind": "text", "ctx_key": "brake_temp_condition", "notes": "Brake temperature condition"},
        {"id": "brake_release_condition", "label": "brake_release_condition", "kind": "text", "ctx_key": "brake_release_condition", "notes": "Brake release condition"},
        {"id": "caliper_drag_status", "label": "caliper_drag_status", "kind": "text", "ctx_key": "caliper_drag_status", "notes": "Caliper drag status"},
        {"id": "pad_drag_status", "label": "pad_drag_status", "kind": "text", "ctx_key": "pad_drag_status", "notes": "Pad drag status"},
        {"id": "parking_brake_drag_flag", "label": "parking_brake_drag_flag", "kind": "text", "ctx_key": "parking_brake_drag_flag", "notes": "true / false"},
    ],
    "Axle & Hubs": [
        {"id": "axle_hub_mode", "label": "axle_hub_mode", "kind": "text", "ctx_key": "axle_hub_mode", "notes": "Inherit / Absolute / Delta / Not used"},
        {"id": "axle_hub_A", "label": "axle_hub_A", "kind": "force", "ctx_key": "axle_hub_A", "notes": "Axle & Hubs A"},
        {"id": "axle_hub_B", "label": "axle_hub_B", "kind": "force_per_speed", "ctx_key": "axle_hub_B", "notes": "Axle & Hubs B"},
        {"id": "axle_hub_C", "label": "axle_hub_C", "kind": "force_per_speed_squared", "ctx_key": "axle_hub_C", "notes": "Axle & Hubs C"},
        {"id": "axle_delta_A", "label": "delta_A", "kind": "force", "ctx_key": "axle_hub_delta_A", "notes": "Delta A"},
        {"id": "axle_delta_B", "label": "delta_B", "kind": "force_per_speed", "ctx_key": "axle_hub_delta_B", "notes": "Delta B"},
        {"id": "axle_delta_C", "label": "delta_C", "kind": "force_per_speed_squared", "ctx_key": "axle_hub_delta_C", "notes": "Delta C"},
        {"id": "axle_source", "label": "source", "kind": "text", "ctx_key": "axle_hub_source_v2", "notes": "Reference source"},
        {"id": "axle_notes", "label": "notes", "kind": "text", "ctx_key": "axle_hub_notes_v2", "notes": "Axle / hub notes"},
    ],
    "Parasitic Losses": [
        {"id": "parasitic_mode", "label": "parasitic_mode", "kind": "text", "ctx_key": "parasitic_mode", "notes": "Inherit / Absolute / Delta / Not used"},
        {"id": "parasitic_A", "label": "parasitic_A", "kind": "force", "ctx_key": "parasitic_A_coef_N", "notes": "Parasitic A"},
        {"id": "parasitic_B", "label": "parasitic_B", "kind": "force_per_speed", "ctx_key": "parasitic_B_Npkph", "notes": "Parasitic B"},
        {"id": "parasitic_C", "label": "parasitic_C", "kind": "force_per_speed_squared", "ctx_key": "parasitic_C_coef_Npkph2", "notes": "Parasitic C"},
        {"id": "parasitic_delta_A_row", "label": "delta_A", "kind": "force", "ctx_key": "parasitic_delta_A", "notes": "Delta A"},
        {"id": "parasitic_delta_B_row", "label": "delta_B", "kind": "force_per_speed", "ctx_key": "parasitic_delta_B", "notes": "Delta B"},
        {"id": "parasitic_delta_C_row", "label": "delta_C", "kind": "force_per_speed_squared", "ctx_key": "parasitic_delta_C", "notes": "Delta C"},
        {"id": "parasitic_source", "label": "source", "kind": "text", "ctx_key": "parasitic_source_v2", "notes": "Reference source"},
        {"id": "parasitic_notes", "label": "notes", "kind": "text", "ctx_key": "parasitic_notes_v2", "notes": "Parasitic notes"},
    ],
}

VDE_CELL_SPECIAL_TOKENS = {
    "inherit": "inherit",
    "inherited": "inherit",
    "blank": "inherit",
    "not used": "not_used",
    "not_used": "not_used",
    "missing": "missing",
    "unavailable": "unavailable",
    "calculated": "calculated",
}

VDE_WORKBOOK_V2_STATE_KEY = "vde_setup_workbook_v2"


def _v2_state_key() -> str:
    return str(st.session_state.get("_vde_workbook_active_state_key") or VDE_WORKBOOK_V2_STATE_KEY)


def _v2_set_state(state: dict) -> None:
    st.session_state[_v2_state_key()] = state


def _v2_field_spec_map() -> dict[str, dict]:
    mapping: dict[str, dict] = {str(spec["id"]): spec for spec in VDE_WORKBOOK_V2_MATRIX_SPECS}
    for section_specs in VDE_WORKBOOK_V2_SECTION_SPECS.values():
        for spec in section_specs:
            mapping[str(spec["id"])] = spec
    return mapping


def _v2_default_scenarios() -> list[dict]:
    return [dict(item) for item in VDE_WORKBOOK_V2_DEFAULT_SCENARIOS]


def _v2_normalize_scenarios(raw_scenarios) -> list[dict]:
    normalized: list[dict] = []
    seen: set[str] = set()
    for raw in raw_scenarios or []:
        key = str((raw or {}).get("key") or "").strip()
        if not key or key in seen:
            continue
        role = "baseline" if key == "baseline" else str((raw or {}).get("role") or "walked").strip().lower() or "walked"
        if role not in {"baseline", "walked"}:
            role = "walked"
        if role == "baseline":
            label = "Baseline"
        else:
            match = re.search(r"(\d+)$", key)
            suffix = match.group(1) if match else str(len([item for item in normalized if item.get("role") == "walked"]) + 1)
            label = str((raw or {}).get("label") or f"Walked #{suffix}").strip() or f"Walked #{suffix}"
        normalized.append({"key": key, "label": label, "role": role})
        seen.add(key)
    if "baseline" not in seen:
        normalized.insert(0, {"key": "baseline", "label": "Baseline", "role": "baseline"})
    if len(normalized) == 1:
        normalized.extend(_v2_default_scenarios()[1:])
    return normalized


def _v2_scenarios(state: dict | None = None) -> list[dict]:
    owned_state = state is not None
    state = state or _v2_state()
    scenarios = state.get("scenarios")
    normalized = _v2_normalize_scenarios(scenarios)
    if scenarios != normalized:
        state["scenarios"] = normalized
        if not owned_state:
            _v2_set_state(state)
    return normalized


def _v2_column_ids(state: dict | None = None) -> list[str]:
    return [str(item["key"]) for item in _v2_scenarios(state)]


def _v2_walked_column_ids(state: dict | None = None) -> list[str]:
    return [str(item["key"]) for item in _v2_scenarios(state) if str(item.get("role") or "") == "walked"]


def _v2_column_label(column_id: str, state: dict | None = None) -> str:
    return _v2_workbook_column_names(state).get(column_id, column_id)


def _v2_allowed_walk_from_ids(column_id: str, state: dict | None = None) -> list[str]:
    if column_id == "baseline":
        return []
    ordered_ids = _v2_column_ids(state)
    if column_id not in ordered_ids:
        return ["baseline"]
    column_index = ordered_ids.index(column_id)
    return ordered_ids[:column_index]


def _v2_last_column_id(state: dict | None = None) -> str:
    column_ids = _v2_column_ids(state)
    return column_ids[-1] if column_ids else "baseline"


def _v2_add_walked_column() -> str:
    state = _v2_state()
    scenarios = list(_v2_scenarios(state))
    walked = [item for item in scenarios if str(item.get("role") or "") == "walked"]
    next_index = len(walked) + 1
    next_key = f"walked_{next_index}"
    previous_key = walked[-1]["key"] if walked else "baseline"
    scenarios.append({"key": next_key, "label": f"Walked #{next_index}", "role": "walked"})
    columns = dict(state.get("columns") or {})
    columns[next_key] = {
        "walk_from": previous_key,
        "line_source": "New / Insert",
        "direct": {
            "line_source": "New / Insert",
        },
    }
    state["scenarios"] = scenarios
    state["columns"] = columns
    state["save_target"] = next_key
    state["audit_target"] = next_key
    metadata = dict(state.get("metadata") or {})
    metadata["save_target"] = next_key
    state["metadata"] = metadata
    _v2_set_state(state)
    return next_key


def _v2_reindex_walked_columns(scenarios: list[dict], columns: dict[str, dict]) -> tuple[list[dict], dict[str, dict]]:
    scenario_map: dict[str, str] = {"baseline": "baseline"}
    new_scenarios = [{"key": "baseline", "label": "Baseline", "role": "baseline"}]
    walked_source_keys: list[str] = []
    for index, scenario in enumerate((item for item in scenarios if str(item.get("key") or "") != "baseline"), start=1):
        old_key = str(scenario.get("key") or "")
        new_key = f"walked_{index}"
        scenario_map[old_key] = new_key
        walked_source_keys.append(old_key)
        new_scenarios.append({"key": new_key, "label": f"Walked #{index}", "role": "walked"})

    new_columns: dict[str, dict] = {"baseline": dict(columns.get("baseline") or {})}
    previous_key = "baseline"
    for old_key in walked_source_keys:
        new_key = scenario_map[old_key]
        column = dict(columns.get(old_key) or {})
        old_walk_from = str(column.get("walk_from") or previous_key or "baseline")
        remapped_walk_from = scenario_map.get(old_walk_from, previous_key if previous_key in [item["key"] for item in new_scenarios] else "baseline")
        if remapped_walk_from == new_key:
            remapped_walk_from = previous_key
        column["walk_from"] = remapped_walk_from or "baseline"
        new_columns[new_key] = column
        previous_key = new_key
    return new_scenarios, new_columns


def _v2_remove_walked_column(column_id: str) -> None:
    if column_id == "baseline":
        return
    state = _v2_state()
    scenarios = [dict(item) for item in _v2_scenarios(state) if str(item.get("key") or "") != column_id]
    columns = dict(state.get("columns") or {})
    columns.pop(column_id, None)
    scenarios, columns = _v2_reindex_walked_columns(scenarios, columns)
    valid_keys = [str(item["key"]) for item in scenarios]
    preview_cache = dict(state.get("preview_cache") or {})
    state["preview_cache"] = {key: value for key, value in preview_cache.items() if key in valid_keys}
    state["scenarios"] = scenarios
    state["columns"] = columns
    state["save_target"] = state.get("save_target") if str(state.get("save_target") or "") in valid_keys else valid_keys[-1]
    state["audit_target"] = state.get("audit_target") if str(state.get("audit_target") or "") in valid_keys else valid_keys[-1]
    metadata = dict(state.get("metadata") or {})
    if str(metadata.get("save_target") or "") not in valid_keys:
        metadata["save_target"] = state["save_target"]
    state["metadata"] = metadata
    _v2_set_state(state)


def _v2_apply_selected_baseline_row(state: dict, selected_row: dict | None) -> dict:
    metadata = dict(state.get("metadata") or {})
    columns = dict(state.get("columns") or {})
    baseline = dict(columns.get("baseline") or {})
    if selected_row:
        metadata.update(_v2_row_metadata_defaults(selected_row))
        metadata["selected_baseline_vde_id"] = int(to_float(selected_row.get("id"), 0) or 0) or None
        metadata["selected_baseline_label"] = _v2_row_label(selected_row)
        metadata["line_source"] = "Existing VDE DB"
        metadata["roadload_source_type"] = "Baseline ABC"
        baseline["selected_vde_id"] = metadata["selected_baseline_vde_id"]
        baseline["line_source"] = "Existing VDE DB"
        baseline["direct"] = {}
    else:
        metadata["selected_baseline_vde_id"] = None
        metadata["selected_baseline_label"] = ""
        baseline["selected_vde_id"] = None
    columns["baseline"] = baseline
    state["columns"] = columns
    state["metadata"] = metadata
    preview_cache = dict(state.get("preview_cache") or {})
    preview_cache.pop("baseline", None)
    state["preview_cache"] = preview_cache
    return state


def parse_vde_cell_value(raw_text, expected_type: str | None = None) -> dict:
    raw_value = "" if raw_text is None else str(raw_text).strip()
    if raw_value == "":
        return {
            "raw_value": "",
            "parsed_value": None,
            "parse_status": "blank",
        }

    normalized = re.sub(r"\s+", " ", raw_value).strip().lower()
    if normalized in VDE_CELL_SPECIAL_TOKENS:
        return {
            "raw_value": raw_value,
            "parsed_value": VDE_CELL_SPECIAL_TOKENS[normalized],
            "parse_status": "token",
        }

    if expected_type in {"int", "float", "mass", "force", "force_per_speed", "force_per_speed_squared", "rrc"}:
        compact = raw_value.replace(" ", "")
        if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", compact):
            parsed = float(compact.replace(",", "."))
            if expected_type == "int":
                parsed = int(parsed)
            return {
                "raw_value": raw_value,
                "parsed_value": parsed,
                "parse_status": "numeric",
            }

    return {
        "raw_value": raw_value,
        "parsed_value": raw_value,
        "parse_status": "text",
    }


@st.cache_data(show_spinner=False)
def _v2_fetch_vde_rows() -> list[dict]:
    try:
        rows = fetch_vde_rows_full()
    except Exception:
        return []
    df = ensure_baseline_aliases(pd.DataFrame(rows))
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _v2_row_label(row: dict) -> str:
    proposal = str(row.get("proposal") or row.get("notes") or "").strip()
    parts = [
        f"VDE-{row.get('id')}",
        str(row.get("make") or row.get("manufacturer") or "").strip(),
        str(row.get("model") or row.get("vehicle_label") or "").strip(),
        str(int(to_float(row.get("year"), row.get("model_year", 0)) or 0) or ""),
        str(row.get("legislation") or "").strip(),
    ]
    label = " | ".join(part for part in parts if part)
    if proposal:
        return f"{label} | {proposal}"
    return label


def _v2_resolve_baseline_selector(raw_value: str, rows: list[dict]) -> dict | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    exact = next((row for row in rows if _v2_row_label(row) == text), None)
    if exact:
        return exact
    upper_text = text.upper()
    marker = upper_text.find("VDE-")
    if marker >= 0:
        suffix = text[marker + 4 :]
        digits = []
        for char in suffix:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if digits:
            return _v2_find_row_by_id(rows, int("".join(digits)))
    return None


def _v2_find_row_by_id(rows: list[dict], row_id) -> dict | None:
    numeric_id = to_float(row_id)
    if numeric_id is None:
        return None
    target_id = int(numeric_id)
    return next((row for row in rows if int(to_float(row.get("id"), 0) or 0) == target_id), None)


def _v2_row_metadata_defaults(row: dict | None) -> dict:
    if not row:
        return {}
    legislation = str(row.get("legislation") or "").strip()
    return {
        "selected_baseline_vde_id": int(to_float(row.get("id"), 0) or 0) or None,
        "legislation": legislation,
        "model_year": int(to_float(row.get("year"), row.get("model_year", 0)) or 0) or None,
        "make": str(row.get("make") or row.get("manufacturer") or "").strip(),
        "model": str(row.get("model") or row.get("vehicle_label") or "").strip(),
        "cycle": str(row.get("cycle_name") or default_cycle_for_legislation(legislation or "EPA") or "").strip(),
        "description": str(row.get("proposal") or row.get("notes") or row.get("vehicle_label") or "").strip(),
        "roadload_source_type": "Baseline ABC",
    }


def _v2_metadata_effective(state: dict | None = None) -> dict:
    state = state or _v2_state()
    metadata = dict(state.get("metadata") or {})
    rows = list(state.get("rows") or [])
    line_source = str(metadata.get("line_source") or "Existing VDE DB")
    selected_row = _v2_find_row_by_id(rows, metadata.get("selected_baseline_vde_id")) if line_source == "Existing VDE DB" else None
    effective = _v2_row_metadata_defaults(selected_row)
    effective["line_source"] = line_source
    effective["selected_baseline_vde_id"] = metadata.get("selected_baseline_vde_id")
    for field_id in ("legislation", "model_year", "make", "model", "cycle", "description", "roadload_source_type", "display_units"):
        value = metadata.get(field_id)
        if value not in (None, ""):
            effective[field_id] = value
    if not str(effective.get("cycle") or "").strip():
        effective["cycle"] = str(default_cycle_for_legislation(str(effective.get("legislation") or "EPA")) or "").strip()
    if not str(effective.get("roadload_source_type") or "").strip():
        effective["roadload_source_type"] = "Baseline ABC" if line_source == "Existing VDE DB" else "From test coastdown"
    effective["display_units"] = str(effective.get("display_units") or normalize_unit_system(st.session_state.get("unit_system") or "Metric"))
    effective["selected_row"] = selected_row
    effective["selected_baseline_label"] = _v2_row_label(selected_row) if selected_row else ""
    return effective


def _v2_metadata_status() -> tuple[str, str]:
    effective = _v2_metadata_effective()
    required = [
        effective.get("legislation"),
        effective.get("model_year"),
        effective.get("make"),
        effective.get("model"),
    ]
    filled = sum(1 for value in required if value not in (None, "", 0))
    detail = " | ".join(
        part for part in [
            str(effective.get("make") or "").strip(),
            str(effective.get("model") or "").strip(),
            str(effective.get("model_year") or "").strip(),
            str(effective.get("legislation") or "").strip(),
        ] if part
    ) or "Metadata pending"
    if filled == 0:
        return "Pending", detail
    if filled < len(required):
        return "Partial", detail
    return "Defined", detail


def _v2_state() -> dict:
    state = st.session_state.get(_v2_state_key())
    if not isinstance(state, dict):
        state = {}
    rows = _v2_fetch_vde_rows()
    row_map = {int(row.get("id")): row for row in rows if row.get("id") is not None}
    default_baseline_id = next(iter(row_map.keys()), None)
    columns = state.get("columns")
    if not isinstance(columns, dict):
        columns = {}
    scenarios = _v2_normalize_scenarios(state.get("scenarios") or _v2_default_scenarios())
    state["scenarios"] = scenarios
    scenario_keys = [str(item["key"]) for item in scenarios]
    baseline = dict(columns.get("baseline") or {})
    baseline.setdefault("line_source", "Existing VDE DB" if default_baseline_id else "New test ABC_TOTAL")
    if str(baseline.get("line_source") or "") == "New test / New ABC_TOTAL line":
        baseline["line_source"] = "New test ABC_TOTAL"
    baseline.setdefault("selected_vde_id", default_baseline_id)
    baseline.setdefault("direct", {})
    normalized_columns: dict[str, dict] = {"baseline": baseline}
    previous_key = "baseline"
    for scenario in scenarios:
        key = str(scenario["key"])
        if key == "baseline":
            continue
        column = dict(columns.get(key) or {})
        column.setdefault("walk_from", previous_key)
        column.setdefault("line_source", "New / Insert")
        column.setdefault("direct", {})
        normalized_columns[key] = column
        previous_key = key
    metadata = dict(state.get("metadata") or {})
    metadata.setdefault("line_source", str(baseline.get("line_source") or ("Existing VDE DB" if default_baseline_id else "New test ABC_TOTAL")))
    if metadata["line_source"] == "New test / New ABC_TOTAL line":
        metadata["line_source"] = "New test ABC_TOTAL"
    if metadata.get("selected_baseline_vde_id") in (None, "") and default_baseline_id is not None:
        metadata["selected_baseline_vde_id"] = default_baseline_id
    selected_row = _v2_find_row_by_id(rows, metadata.get("selected_baseline_vde_id"))
    row_defaults = _v2_row_metadata_defaults(selected_row)
    for field_id in ("legislation", "model_year", "make", "model", "cycle", "description", "roadload_source_type"):
        metadata.setdefault(field_id, row_defaults.get(field_id))
    metadata.setdefault("display_units", normalize_unit_system(st.session_state.get("unit_system") or "Metric"))
    metadata.setdefault("save_target", str(state.get("save_target") or _v2_last_column_id({"scenarios": scenarios})))
    baseline["line_source"] = str(metadata.get("line_source") or baseline.get("line_source") or "Existing VDE DB")
    baseline["selected_vde_id"] = metadata.get("selected_baseline_vde_id") if baseline["line_source"] == "Existing VDE DB" else None
    state["columns"] = {key: normalized_columns.get(key, {"direct": {}}) for key in scenario_keys}
    state["metadata"] = metadata
    state["rows"] = rows
    state.setdefault("menu", VDE_WORKBOOK_V2_MENUS[0])
    if str(metadata.get("save_target") or "") not in scenario_keys:
        metadata["save_target"] = _v2_last_column_id({"scenarios": scenarios})
    state["save_target"] = str(metadata.get("save_target") or state.get("save_target") or _v2_last_column_id({"scenarios": scenarios}))
    audit_target = str(state.get("audit_target") or _v2_last_column_id({"scenarios": scenarios}))
    if audit_target not in scenario_keys:
        audit_target = _v2_last_column_id({"scenarios": scenarios})
    state["audit_target"] = audit_target
    st.session_state["unit_system"] = str(metadata.get("display_units") or "Metric")
    if _v2_state_key() == VDE_WORKBOOK_V21_STATE_KEY:
        state = _v21_ensure_workbook_state(state)
    _v2_set_state(state)
    return state


def _v2_parse_value(raw_value, kind: str):
    parsed_meta = parse_vde_cell_value(raw_value, kind)
    value = parsed_meta.get("parsed_value")
    if parsed_meta.get("parse_status") == "blank":
        return None
    if parsed_meta.get("parse_status") in {"token", "text"}:
        return value
    if kind == "mass":
        return _editor_canonical_or_none(value, "mass")
    if kind == "force":
        return _editor_canonical_or_none(value, "force")
    if kind == "force_per_speed":
        return _editor_canonical_or_none(value, "force_per_speed")
    if kind == "force_per_speed_squared":
        return _editor_canonical_or_none(value, "force_per_speed_squared")
    if kind == "rrc":
        return _editor_canonical_or_none(value, "rrc")
    if kind == "int":
        numeric = _editor_float_or_none(value)
        return None if numeric is None else int(numeric)
    if kind == "float":
        return _editor_float_or_none(value)
    return str(value).strip()


def _v2_format_value(value, kind: str) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    if kind == "mass":
        return _display_quantity_text(value, "mass", unavailable="")
    if kind == "force":
        return _display_quantity_text(value, "force", unavailable="")
    if kind == "force_per_speed":
        return _display_quantity_text(value, "force_per_speed", unavailable="")
    if kind == "force_per_speed_squared":
        return _display_quantity_text(value, "force_per_speed_squared", unavailable="")
    if kind == "rrc":
        return format_quantity(value, "rrc", _current_unit_system(), include_unit=False, unavailable="", format_str="%.3f")
    if kind == "int":
        numeric = to_float(value)
        return "" if numeric is None else str(int(numeric))
    if kind == "float":
        numeric = to_float(value)
        return "" if numeric is None else f"{float(numeric):.6f}".rstrip("0").rstrip(".")
    return str(value)


def _v2_row_to_effective_state(row: dict) -> dict:
    legislation = str(row.get("legislation") or "").strip()
    default_mass_intention = "FROM_BASELINE"
    if legislation.upper() == "EPA":
        default_mass_intention = "EPA_STATUS"
    elif legislation.upper() == "WLTP":
        default_mass_intention = "WLTP_TMH" if row.get("test_mass_high_kg") not in (None, "") else "WLTP_TML"
    effective: dict[str, object] = {
        "line_source": "Existing VDE DB",
        "vde_id": row.get("id"),
        "baseline_selector": _v2_row_label(row),
        "description": str(row.get("proposal") or row.get("notes") or row.get("vehicle_label") or "").strip(),
        "proposal_direct": "",
        "proposal_effective": str(row.get("proposal") or row.get("notes") or "").strip(),
        "legislation": legislation,
        "model_year": int(to_float(row.get("year"), row.get("model_year", 0)) or 0) or None,
        "make": str(row.get("make") or row.get("manufacturer") or "").strip(),
        "model": str(row.get("model") or row.get("vehicle_label") or "").strip(),
        "cycle": str(row.get("cycle_name") or "").strip(),
        "mass_intention": default_mass_intention,
    }
    alias_map = {
        "curb_mass_kg": row.get("mass_kg", row.get("baseline_mass_kg")),
        "test_mass_kg": row.get("test_mass_kg"),
        "inertia_class": row.get("inertia_class"),
        "TWC_kg": row.get("twc_kg", row.get("etw_kg", row.get("inertia_class"))),
        "TML_kg": row.get("test_mass_low_kg"),
        "TMH_kg": row.get("test_mass_high_kg"),
        "fr_weight_pct": row.get("weight_dist_fr_pct"),
        "payload_kg": row.get("payload_kg"),
        "GVWR_kg": row.get("gvwr_kg", row.get("mass_profile_gvwr_kg")),
        "GCWR_kg": row.get("gcwr_kg", row.get("mass_profile_gcwr_kg")),
        "trailer_weight_kg": row.get("trailer_mass_kg"),
        "trailer_code": row.get("trailer_code"),
        "trailer_roadload_source": row.get("trailer_roadload_source"),
        "trailer_A": row.get("trailer_A_coef_N", row.get("trailer_A")),
        "trailer_B": row.get("trailer_B_coef_Npkph", row.get("trailer_B")),
        "trailer_C": row.get("trailer_C_coef_Npkph2", row.get("trailer_C")),
        "mass_rule_status": row.get("mass_rule_status"),
        "mass_rule_notes": row.get("mass_rule_notes"),
        "Cd": row.get("cd"),
        "frontal_area_m2": row.get("frontal_area_m2"),
        "CdA": row.get("cda_m2"),
        "ABC_TOTAL_A": row.get("A", row.get("baseline_A_N")),
        "ABC_TOTAL_B": row.get("B", row.get("baseline_B_N_per_kph")),
        "ABC_TOTAL_C": row.get("C", row.get("baseline_C_N_per_kph2")),
        "rrc_N_per_kN": row.get("rrc_N_per_kN"),
        "psi_front": row.get("front_pressure_psi"),
        "psi_rear": row.get("rear_pressure_psi"),
        "tire_A": row.get("tire_A_final"),
        "tire_B": row.get("tire_B_final"),
        "tire_C": row.get("tire_C_final"),
        "trans_A_loss": row.get("trans_A_coef_N"),
        "trans_B_loss": row.get("trans_B_coef_Npkph", row.get("trans_B_Npkph")),
        "trans_C_loss": row.get("trans_C_coef_Npkph2"),
        "brake_A": row.get("brake_A_coef_N"),
        "brake_B": row.get("brake_B_Npkph"),
        "brake_C": row.get("brake_C_coef_Npkph2"),
        "parasitic_A": row.get("parasitic_A_coef_N"),
        "parasitic_B": row.get("parasitic_B_Npkph"),
        "parasitic_C": row.get("parasitic_C_coef_Npkph2"),
    }
    effective.update({key: value for key, value in alias_map.items() if value not in (None, "")})
    return effective


def _v2_apply_trailer_preset(effective: dict) -> None:
    preset = SCENARIO_WORKBOOK_TRAILER_PRESETS.get(str(effective.get("trailer_code") or "").strip())
    if not preset:
        return
    effective.setdefault("trailer_weight_kg", preset["weight_kg"])
    effective.setdefault("trailer_A", preset["A"])
    effective.setdefault("trailer_B", preset["B"])
    effective.setdefault("trailer_C", preset["C"])


def _v2_numeric_value(value):
    if isinstance(value, str) and value.strip().lower() in set(VDE_CELL_SPECIAL_TOKENS.values()):
        return None
    return to_float(value)


def _v2_apply_mass_intention(effective: dict, *, inherited: dict | None = None) -> None:
    inherited = dict(inherited or {})
    legislation = str(effective.get("legislation") or inherited.get("legislation") or "").strip().upper()
    requested = str(effective.get("mass_intention") or "").strip().upper()
    inherited_mode = str(inherited.get("mass_intention") or "").strip().upper()
    if not requested:
        if legislation == "EPA":
            requested = "EPA_STATUS"
        elif legislation == "WLTP":
            requested = "WLTP_TMH" if _v2_numeric_value(effective.get("TMH_kg")) is not None else "WLTP_TML"
        else:
            requested = inherited_mode or "FROM_BASELINE"
    if requested in {"INHERIT", "FROM_BASELINE"} and inherited_mode:
        requested = inherited_mode
    if requested == "MANUAL":
        requested = "CUSTOM"
    if requested == "NEW_TEST_MASS":
        requested = "CUSTOM"
    effective["mass_intention"] = requested

    curb_mass = _v2_numeric_value(effective.get("curb_mass_kg"))
    manual_test_mass = _v2_numeric_value(effective.get("test_mass_kg"))
    twc_mass = _v2_numeric_value(effective.get("TWC_kg", effective.get("inertia_class")))
    if twc_mass is None:
        twc_mass = _v2_numeric_value(effective.get("inertia_class"))
    tml_mass = _v2_numeric_value(effective.get("TML_kg"))
    tmh_mass = _v2_numeric_value(effective.get("TMH_kg"))
    gvwr_mass = _v2_numeric_value(effective.get("GVWR_kg"))
    gcwr_mass = _v2_numeric_value(effective.get("GCWR_kg"))
    trailer_weight = _v2_numeric_value(effective.get("trailer_weight_kg"))
    trailer_a = _v2_numeric_value(effective.get("trailer_A"))
    trailer_b = _v2_numeric_value(effective.get("trailer_B"))
    trailer_c = _v2_numeric_value(effective.get("trailer_C"))
    trailer_source = str(effective.get("trailer_roadload_source") or "").strip()
    prep_inertia = _v2_numeric_value(effective.get("prep_inertia_class"))

    mass_status = "OK"
    mass_notes = "Resolved"
    effective_test_mass = manual_test_mass
    vde_mass_basis = str(effective.get("vde_mass_basis") or inherited.get("vde_mass_basis") or "PHYSICAL_TEST_MASS")
    fuelcons_basis = str(effective.get("fuelcons_mass_basis") or inherited.get("fuelcons_mass_basis") or "TEST_MASS")
    payload_display_kg = None
    vehicle_mass_at_gcwr = None
    trailer_roadload_status = "Not used"

    if requested == "EPA_STATUS":
        if twc_mass is not None:
            effective_test_mass = twc_mass
            mass_status = "OK"
            mass_notes = "Using EPA ETW / TWC from inertia_class"
        elif curb_mass is not None:
            loaded_vehicle_weight = curb_mass + 136.1
            fallback_twc = inertia_class_from_mass(loaded_vehicle_weight)
            effective_test_mass = fallback_twc
            mass_status = "Review" if fallback_twc is not None else "Missing"
            mass_notes = "EPA fallback from curb + 300 lb requires ETW/TWC lookup"
        else:
            effective_test_mass = None
            mass_status = "Missing"
            mass_notes = "Need inertia_class or curb_mass_kg"
        vde_mass_basis = "EPA_INERTIA_CLASS"
        fuelcons_basis = "TWC"
    elif requested == "EPA_PLUS_1_TWC":
        effective_test_mass = prep_inertia if prep_inertia is not None else twc_mass
        vde_mass_basis = "EPA_INERTIA_CLASS"
        fuelcons_basis = "TWC"
        if effective_test_mass is None:
            mass_status = "Missing"
            mass_notes = "EPA ETW / TWC target unavailable"
        else:
            mass_status = "OK"
            mass_notes = "Resolved EPA TWC target"
    elif requested == "WLTP_TML":
        effective_test_mass = tml_mass
        vde_mass_basis = "WLTP_TML"
        fuelcons_basis = "TEST_MASS"
        if effective_test_mass is None:
            mass_status = "Missing"
            mass_notes = "TML_kg required"
    elif requested == "WLTP_TMH":
        effective_test_mass = tmh_mass
        vde_mass_basis = "WLTP_TMH"
        fuelcons_basis = "TEST_MASS"
        if effective_test_mass is None:
            mass_status = "Missing"
            mass_notes = "TMH_kg required"
    elif requested == "GVWR":
        effective_test_mass = gvwr_mass
        vde_mass_basis = "GVWR"
        fuelcons_basis = "TEST_MASS"
        if effective_test_mass is None:
            mass_status = "Missing"
            mass_notes = "GVWR_kg required"
        elif curb_mass is not None and gvwr_mass is not None and gvwr_mass < curb_mass:
            mass_status = "Invalid"
            mass_notes = "GVWR_kg cannot be lower than curb_mass_kg"
        elif curb_mass is None:
            mass_status = "Review"
            mass_notes = "GVWR resolved but curb_mass_kg is unavailable for payload display"
        if gvwr_mass is not None and curb_mass is not None:
            payload_display_kg = gvwr_mass - curb_mass
    elif requested == "GCWR":
        effective_test_mass = gcwr_mass
        vde_mass_basis = "GCWR_TRAILER"
        fuelcons_basis = "TEST_MASS"
        if effective_test_mass is None:
            mass_status = "Missing"
            mass_notes = "GCWR_kg required"
        elif trailer_weight is None:
            mass_status = "Missing"
            mass_notes = "trailer_weight_kg required"
        else:
            vehicle_mass_at_gcwr = gcwr_mass - trailer_weight
            trailer_abc_complete = all(value is not None for value in (trailer_a, trailer_b, trailer_c))
            if trailer_source == "Trailer DB":
                trailer_roadload_status = "OK" if str(effective.get("trailer_code") or "").strip() and trailer_abc_complete else "Review"
            elif trailer_source == "Manual ABC":
                trailer_roadload_status = "OK" if trailer_abc_complete else "Review"
            else:
                trailer_roadload_status = "Review"
            if trailer_weight >= gcwr_mass:
                mass_status = "Invalid"
                mass_notes = "trailer_weight_kg must be lower than GCWR_kg"
            elif curb_mass is not None and vehicle_mass_at_gcwr < curb_mass:
                mass_status = "Invalid"
                mass_notes = "vehicle_mass_at_gcwr cannot be lower than curb_mass_kg"
            elif gvwr_mass is not None and vehicle_mass_at_gcwr > gvwr_mass:
                mass_status = "Review"
                mass_notes = "vehicle_mass_at_gcwr exceeds GVWR_kg"
            elif trailer_roadload_status != "OK":
                mass_status = "Review"
                mass_notes = "Trailer roadload missing; only trailer mass/inertia included"
            else:
                mass_status = "OK"
                mass_notes = "Resolved GCWR / trailer mass"
            if trailer_roadload_status == "OK":
                base_a = _v2_numeric_value(effective.get("ABC_TOTAL_A"))
                base_b = _v2_numeric_value(effective.get("ABC_TOTAL_B"))
                base_c = _v2_numeric_value(effective.get("ABC_TOTAL_C"))
                if base_a is not None and trailer_a is not None:
                    effective["ABC_TOTAL_A"] = base_a + trailer_a
                if base_b is not None and trailer_b is not None:
                    effective["ABC_TOTAL_B"] = base_b + trailer_b
                if base_c is not None and trailer_c is not None:
                    effective["ABC_TOTAL_C"] = base_c + trailer_c
    elif requested == "PERF_CURB_100KG":
        effective_test_mass = None if curb_mass is None else curb_mass + 100.0
        vde_mass_basis = "CURB_PLUS_DRIVER"
        fuelcons_basis = "TEST_MASS"
        if effective_test_mass is None:
            mass_status = "Missing"
            mass_notes = "curb_mass_kg required"
    elif requested == "PERF_CURB_300LB":
        effective_test_mass = None if curb_mass is None else curb_mass + 136.1
        vde_mass_basis = "CURB_PLUS_DRIVER"
        fuelcons_basis = "TEST_MASS"
        if effective_test_mass is None:
            mass_status = "Missing"
            mass_notes = "curb_mass_kg required"
    elif requested == "CUSTOM":
        effective_test_mass = manual_test_mass
        vde_mass_basis = "CUSTOM"
        fuelcons_basis = "TEST_MASS"
        if effective_test_mass is None:
            mass_status = "Missing"
            mass_notes = "test_mass_kg required"

    if requested in {"INHERIT", "FROM_BASELINE"} and inherited:
        effective_test_mass = inherited.get("effective_test_mass_kg")
        vde_mass_basis = str(inherited.get("vde_mass_basis") or vde_mass_basis)
        fuelcons_basis = str(inherited.get("fuelcons_mass_basis") or fuelcons_basis)
        mass_status = str(inherited.get("mass_rule_status") or mass_status)
        mass_notes = str(inherited.get("mass_rule_notes") or mass_notes)

    effective["effective_test_mass_kg"] = effective_test_mass
    effective["vde_mass_basis"] = vde_mass_basis
    effective["fuelcons_mass_basis"] = fuelcons_basis
    effective["mass_rule_status"] = mass_status
    effective["mass_rule_notes"] = mass_notes
    effective["payload_display_kg"] = payload_display_kg
    effective["vehicle_mass_at_gcwr"] = vehicle_mass_at_gcwr
    effective["trailer_roadload_status"] = trailer_roadload_status


def _v2_effective_state(column_id: str, *, _stack: tuple[str, ...] = ()) -> dict:
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    valid_columns = set(_v2_column_ids(state))
    spec_map = _v2_field_spec_map()
    metadata = _v2_metadata_effective(state)
    if column_id == "baseline":
        baseline = dict(columns.get("baseline") or {})
        if str(baseline.get("line_source") or "Existing VDE DB") == "Existing VDE DB":
            row = next((item for item in state.get("rows") or [] if int(item.get("id")) == int(baseline.get("selected_vde_id") or 0)), None)
            effective = _v2_row_to_effective_state(row or {})
        else:
            effective = {
                "line_source": "New test ABC_TOTAL",
                "vde_id": "New / Insert",
                "baseline_selector": "New test ABC_TOTAL",
                "walk_from": "",
                "proposal_direct": "",
                "proposal_effective": "",
            }
        effective["line_source"] = str(metadata.get("line_source") or effective.get("line_source") or "Existing VDE DB")
        effective["baseline_selector"] = str(metadata.get("selected_baseline_label") or effective.get("baseline_selector") or "")
        for field_id in ("legislation", "model_year", "make", "model", "cycle"):
            if metadata.get(field_id) not in (None, "", 0):
                effective[field_id] = metadata.get(field_id)
        effective["description"] = str(metadata.get("description") or effective.get("description") or "")
        for field_id, raw_value in dict(baseline.get("direct") or {}).items():
            spec = spec_map.get(str(field_id), {"kind": "text"})
            parsed = _v2_parse_value(raw_value, str(spec.get("kind") or "text"))
            effective[field_id] = parsed if parsed is not None else ""
        effective["description"] = str(effective.get("description") or "")
        effective["proposal_effective"] = str(effective.get("proposal_direct") or effective.get("proposal_effective") or "").strip()
        if str(metadata.get("description") or "").strip() and not str(effective.get("proposal_direct") or "").strip():
            effective["proposal_effective"] = str(metadata.get("description") or "").strip()
        effective["save_target"] = "Selected" if str(state.get("save_target") or "") == column_id else ""
        effective["scenario_notes"] = str(dict(baseline.get("direct") or {}).get("scenario_notes") or "")
        _v21_apply_proposals_to_effective(effective, column_id, state)
        _v2_apply_trailer_preset(effective)
        _v2_apply_mass_intention(effective)
        return effective

    if column_id in _stack:
        return _v2_effective_state("baseline")
    column = dict(columns.get(column_id) or {})
    source_id = str(column.get("walk_from") or "baseline")
    if source_id not in valid_columns:
        source_id = "baseline"
    source_state = _v2_effective_state(source_id, _stack=_stack + (column_id,))
    effective = deepcopy(source_state)
    direct = dict(column.get("direct") or {})
    for field_id, raw_value in direct.items():
        if raw_value in (None, ""):
            continue
        spec = spec_map.get(str(field_id), {"kind": "text"})
        parsed = _v2_parse_value(raw_value, str(spec.get("kind") or "text"))
        if parsed is not None:
            effective[field_id] = parsed
    direct_proposal = str(direct.get("proposal_direct") or "").strip()
    inherited_proposal = str(source_state.get("proposal_effective") or "").strip()
    effective["proposal_direct"] = direct_proposal
    effective["proposal_effective"] = " + ".join(part for part in [inherited_proposal, direct_proposal] if part)
    effective["walk_from"] = source_id
    effective["vde_id"] = "New / Insert"
    effective["line_source"] = str(direct.get("line_source") or "New / Insert")
    effective["save_target"] = "Selected" if str(state.get("save_target") or "") == column_id else ""
    effective["scenario_notes"] = str(direct.get("scenario_notes") or "")
    _v21_apply_proposals_to_effective(effective, column_id, state)
    _v2_apply_trailer_preset(effective)
    _v2_apply_mass_intention(effective, inherited=source_state)
    return effective


def _v2_column_status(column_id: str) -> tuple[str, str]:
    effective = _v2_effective_state(column_id)
    if column_id == "baseline":
        metadata_status, metadata_detail = _v2_metadata_status()
        if str(effective.get("line_source") or "").startswith("Existing") and effective.get("vde_id") not in (None, "", "New / Insert"):
            return "OK", "Loaded from VDE DB"
        if metadata_status == "Defined":
            return "Ready", metadata_detail
        if metadata_status == "Partial":
            return "Missing", metadata_detail
        if metadata_status == "Pending":
            return "Missing", "New baseline line needs core metadata"
        return "Ready", "New test / ABC_TOTAL baseline"
    if not str(effective.get("walk_from") or "").strip():
        return "Blocked", "Walk From missing"
    if not str(effective.get("proposal_effective") or "").strip():
        return "Review", "Proposal missing"
    if str(effective.get("mass_rule_status") or "").strip() in {"Missing", "Review"}:
        return str(effective.get("mass_rule_status")), str(effective.get("mass_rule_notes") or "Mass rule needs review")
    required = [effective.get("legislation"), effective.get("model_year"), effective.get("make"), effective.get("model")]
    if any(value in (None, "", 0) for value in required):
        return "Missing", "Metadata incomplete"
    return "Ready", "Effective snapshot resolved"


def _v2_domain_statuses(column_id: str, preview: dict | None = None) -> dict[str, tuple[str, str]]:
    effective = _v2_effective_state(column_id)
    preview_data = dict(preview or {})
    trans = dict(preview_data.get("transmission_losses") or {})
    vde_net = dict(preview_data.get("vde_net") or {})
    preview_ok = bool(preview_data.get("ok"))
    metadata_status, metadata_detail = _v2_metadata_status()
    state = _v2_state()
    workbook_detail = metadata_detail
    if column_id == "baseline":
        walked_count = sum(1 for col_id in _v2_walked_column_ids(state) if dict((state.get("columns") or {}).get(col_id) or {}).get("direct"))
        selector_hint = str(_v2_metadata_effective(state).get("selected_baseline_label") or "Baseline pending")
        workbook_detail = f"{selector_hint} | {walked_count} requested proposal(s)"
    mass_status = "OK" if str(effective.get("mass_rule_status") or "").strip() in {"Ready", "OK"} else str(effective.get("mass_rule_status") or "Pending")
    mass_detail = str(effective.get("mass_rule_notes") or effective.get("mass_intention") or "Pending")

    tire_status = "OK"
    tire_detail = str(effective.get("tire_mode") or "Inherited")
    if str(effective.get("tire_code") or "").strip() and (effective.get("psi_front") in (None, "") or effective.get("psi_rear") in (None, "")):
        tire_status, tire_detail = "Review", "Tire changed with inherited pressure"

    transmission_status = "OK" if str(trans.get("status") or "").lower() == "available" else "Review"
    transmission_detail = "NET available" if transmission_status == "OK" else "NET pending / missing losses"

    brake_status = "OK" if effective.get("brake_A") not in (None, "") or str(effective.get("brake_mode") or "").strip() in {"", "Inherit"} else "Pending"
    axle_status = "OK" if str(effective.get("axle_hub_mode") or "").strip() not in {"Blocked"} else "Blocked"
    parasitic_status = "OK" if str(effective.get("parasitic_mode") or "").strip() not in {"Blocked"} else "Blocked"
    preview_status = "Ready" if preview_ok and vde_net.get("mj_per_km") is not None else ("Review" if preview_ok else "Pending")
    preview_detail = "Preview resolved" if preview_ok else "Preview pending"

    return {
        "Scenario Workbook": (metadata_status, workbook_detail),
        "Mass & Aero": (mass_status, mass_detail),
        "Tire": (tire_status, tire_detail),
        "Transmission": (transmission_status, transmission_detail),
        "Brake": (brake_status, str(effective.get("brake_mode") or "Inherited")),
        "Axle & Hubs": (axle_status, str(effective.get("axle_hub_mode") or "Inherited")),
        "Parasitic Losses": (parasitic_status, str(effective.get("parasitic_mode") or "Inherited")),
        "Preview / Save": (preview_status, preview_detail),
    }


def _v2_state_to_ctx(column_id: str) -> dict:
    effective = _v2_effective_state(column_id)
    ctx = deepcopy(VDE_SETUP_CTX_DEFAULTS)
    current_ctx = dict(st.session_state.get("ctx") or {})
    for key in ("cycle_df", "cycle_source"):
        if key in current_ctx:
            ctx[key] = current_ctx.get(key)
    state = _v2_state()
    metadata = _v2_metadata_effective(state)
    selected_row = metadata.get("selected_row")
    if str(metadata.get("line_source") or "Existing VDE DB") == "Existing VDE DB":
        ctx["mode"] = "From baseline (editable)"
        ctx["selected_baseline_row"] = selected_row
        ctx["baseline_dict"] = selected_row
        ctx["baseline_id"] = selected_row.get("id") if selected_row else None
        ctx["vde_id_parent"] = selected_row.get("id") if selected_row else None
        ctx["abc_total_source_ui"] = "Baseline ABC"
    else:
        ctx["mode"] = "New line (manual / test)"
        ctx["selected_baseline_row"] = None
        ctx["baseline_dict"] = None
        ctx["baseline_id"] = None
        ctx["vde_id_parent"] = None
        ctx["abc_total_source_ui"] = "From test coastdown"

    spec_map = _v2_field_spec_map()
    for field_id, spec in spec_map.items():
        ctx_key = spec.get("ctx_key")
        if ctx_key and field_id in effective and effective.get(field_id) not in (None, ""):
            ctx[ctx_key] = effective.get(field_id)

    mass_mode = str(effective.get("mass_intention") or "").strip().upper()
    effective_test_mass = _v2_numeric_value(effective.get("effective_test_mass_kg"))
    if _v2_numeric_value(effective.get("TWC_kg")) is not None:
        ctx["twc_kg"] = _v2_numeric_value(effective.get("TWC_kg"))
    if mass_mode == "WLTP_TML":
        ctx["test_mass_basis"] = "WLTP_TML"
    elif mass_mode == "WLTP_TMH":
        ctx["test_mass_basis"] = "WLTP_TMH"
    elif mass_mode in {"EPA_STATUS", "EPA_PLUS_1_TWC"}:
        ctx["test_mass_basis"] = "EPA_INERTIA_CLASS"
    elif mass_mode == "GVWR":
        ctx["test_mass_basis"] = "GVWR"
    elif mass_mode == "GCWR":
        ctx["test_mass_basis"] = "GCWR_TRAILER"
    elif mass_mode in {"PERF_CURB_100KG", "PERF_CURB_300LB"}:
        ctx["test_mass_basis"] = "CURB_PLUS_DRIVER"
    elif effective_test_mass is not None:
        ctx["test_mass_basis"] = "PHYSICAL_TEST_MASS"
        ctx["test_mass_kg"] = effective_test_mass
    ctx["tire_load_mass_basis"] = "TEST_MASS"

    ctx["legislation"] = str(metadata.get("legislation") or effective.get("legislation") or ctx.get("legislation") or "EPA")
    if metadata.get("model_year") not in (None, "", 0):
        ctx["year"] = int(to_float(metadata.get("model_year"), ctx.get("year", 2024)) or 2024)
    ctx["make"] = str(metadata.get("make") or effective.get("make") or ctx.get("make") or "")
    ctx["model"] = str(metadata.get("model") or effective.get("model") or ctx.get("model") or "")
    ctx["cycle_name"] = str(metadata.get("cycle") or effective.get("cycle") or current_ctx.get("cycle_name") or "Pending")
    ctx["notes"] = str(effective.get("proposal_effective") or metadata.get("description") or effective.get("description") or "")
    if effective.get("CdA") not in (None, ""):
        ctx["aero_C_coef_Npkph2"] = 0.5 * 1.2 * float(to_float(effective.get("CdA"), 0.0) or 0.0) * (1 / 3.6) ** 2
    return ctx


def _v2_preview(column_id: str) -> dict:
    ctx = _v2_state_to_ctx(column_id)
    try:
        return build_vde_setup_preview_from_ctx(ctx)
    except Exception as exc:
        return {"ok": False, "warnings": [str(exc)]}


def _v2_cached_preview(column_id: str) -> dict:
    state = _v2_state()
    cache = dict(state.get("preview_cache") or {})
    cached = dict(cache.get(column_id) or {})
    return cached or {"ok": False, "warnings": [], "cached": False}


def _v2_store_previews(previews: dict[str, dict]) -> None:
    state = _v2_state()
    cache = dict(state.get("preview_cache") or {})
    for column_id, preview in previews.items():
        payload = dict(preview or {})
        payload["cached"] = True
        cache[column_id] = payload
    state["preview_cache"] = cache
    _v2_set_state(state)


def _v2_default_focus_column() -> str:
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    for column_id in reversed(_v2_walked_column_ids(state)):
        if dict(columns.get(column_id) or {}).get("direct"):
            return column_id
    return "baseline"


def _v2_render_context_header() -> None:
    column_id = _v2_default_focus_column()
    effective = _v2_effective_state(column_id)
    metadata = _v2_metadata_effective()
    preview = _v2_cached_preview(column_id)
    vde_net = dict(preview.get("vde_net") or {})
    save_status = "ready" if preview.get("ok") else "pending"
    if _v2_state_key() == VDE_WORKBOOK_V21_STATE_KEY:
        baseline = _v2_effective_state("baseline")
        baseline_preview = _v2_cached_preview("baseline")
        baseline_net = dict(baseline_preview.get("vde_net") or {})
        baseline_legislation = str(baseline.get("legislation") or metadata.get("legislation") or "Pending")
        etw_label = "EPA ETW / TWC" if baseline_legislation.upper() == "EPA" else "Reference test mass"
        items = [
            ("Vehicle", " | ".join(part for part in [str(baseline.get("make") or metadata.get("make") or ""), str(baseline.get("model") or metadata.get("model") or ""), str(baseline.get("model_year") or metadata.get("model_year") or "")] if part) or "Pending"),
            ("Baseline VDE ID", str(baseline.get("vde_id") or metadata.get("selected_baseline_vde_id") or "Pending")),
            ("Legislation", baseline_legislation),
            ("Cycle", str(baseline.get("cycle") or metadata.get("cycle") or "Pending")),
            ("Mass basis", str(baseline.get("vde_mass_basis") or baseline.get("test_mass_basis") or baseline.get("mass_intention") or "Pending")),
            ("Curb / base mass", _v2_format_value(baseline.get("curb_mass_kg"), "mass") or "-"),
            (etw_label, _v2_format_value(baseline.get("inertia_class") if baseline_legislation.upper() == "EPA" else baseline.get("test_mass_kg"), "mass") or "-"),
            ("Resolved VDE test mass", _v2_format_value(baseline.get("effective_test_mass_kg") or baseline.get("test_mass_kg"), "mass") or "-"),
            ("Roadload / ABC source", str(baseline.get("roadload_source_type") or metadata.get("roadload_source_type") or ("Baseline ABC" if str(metadata.get("line_source") or "").startswith("Existing") else "New ABC_TOTAL line"))),
            ("ABC_TOTAL A/B/C", " / ".join(_v2_format_value(baseline.get(field), "float") or "-" for field in ["ABC_TOTAL_A", "ABC_TOTAL_B", "ABC_TOTAL_C"])),
            ("NET status", "available" if baseline_net.get("mj_per_km") is not None else "unavailable"),
            ("Save status", save_status),
        ]
    else:
        items = [
            ("Vehicle", " | ".join(part for part in [str(metadata.get("make") or ""), str(metadata.get("model") or ""), str(metadata.get("model_year") or "")] if part) or "Pending"),
            ("Cycle", str(metadata.get("cycle") or effective.get("cycle") or "standard / pending")),
            ("Mass basis", str(effective.get("mass_intention") or "Pending")),
            ("Roadload / ABC Source", str(metadata.get("roadload_source_type") or ("Baseline ABC" if str(metadata.get("line_source") or "").startswith("Existing") else "New ABC_TOTAL line"))),
            ("NET", "available" if vde_net.get("mj_per_km") is not None else "unavailable"),
            ("Save", save_status),
        ]
    body = "".join(
        (
            "<div class='vde-context-item'>"
            f"<div class='vde-context-label'>{html.escape(label)}</div>"
            f"<div class='vde-context-value'>{html.escape(str(value or '-'))}</div>"
            "</div>"
        )
        for label, value in items
    )
    st.markdown(f"<div class='vde-context-strip'>{body}</div>", unsafe_allow_html=True)
    walked_count = len(_v2_walked_column_ids())
    if _v2_state_key() == VDE_WORKBOOK_V21_STATE_KEY:
        st.caption(f"Scenario Workbook keeps Baseline plus {walked_count} requested proposal(s) side by side while the proposal matrix stays global.")
    else:
        st.caption(f"Workbook v2 keeps Baseline plus {walked_count} walked column(s) side by side while the active menu uses a lightweight grid.")


def _v2_render_status_bar() -> None:
    column_id = _v2_default_focus_column()
    statuses = _v2_domain_statuses(column_id, _v2_cached_preview(column_id))
    st.caption("VDE Status Bar")
    cols = st.columns(len(statuses))
    for col, (label, payload) in zip(cols, statuses.items()):
        with col:
            _render_status_bar_item(label, payload[0], payload[1])


def _v2_workbook_column_names(state: dict | None = None) -> dict[str, str]:
    state = state or st.session_state.get(_v2_state_key())
    scenarios = _v2_normalize_scenarios((state or {}).get("scenarios") or _v2_default_scenarios())
    return {
        str(item["key"]): str(item["label"])
        for item in scenarios
    }


def _ppe_column_label(column_id: str) -> str:
    column_id = str(column_id or "").strip()
    if column_id == "baseline":
        return "Baseline / Printed"
    match = re.fullmatch(r"walked_(\d+)", column_id)
    if match:
        return f"Requested #{match.group(1)}"
    return _v2_column_label(column_id, _v2_state())


def _v21_request_column_labels(state: dict | None = None) -> dict[str, str]:
    labels = _v2_workbook_column_names(state)
    return {
        column_id: _ppe_column_label(column_id)
        for column_id in labels
    }


def _v21_display_column_label(column_id: str, state: dict | None = None, *, baseline_printed: bool = False) -> str:
    column_id = str(column_id or "").strip()
    if column_id == "baseline":
        return "Baseline / Printed" if baseline_printed else "Baseline"
    match = re.fullmatch(r"walked_(\d+)", column_id)
    if match:
        return f"Requested #{match.group(1)}"
    return _v2_column_label(column_id, state or _v2_state())


def _v21_render_context_header() -> None:
    state = _v2_state()
    metadata = dict(_v2_metadata_effective(state) or {})
    baseline = dict(_v2_effective_state("baseline") or {})
    vehicle = " | ".join(
        part
        for part in [
            str(baseline.get("make") or metadata.get("make") or ""),
            str(baseline.get("model") or metadata.get("model") or ""),
            str(baseline.get("model_year") or metadata.get("model_year") or ""),
        ]
        if part
    ) or "Pending"
    abc_total = " / ".join(
        _v2_format_value(baseline.get(field), "float") or "-"
        for field in ["ABC_TOTAL_A", "ABC_TOTAL_B", "ABC_TOTAL_C"]
    )
    net_available = any(
        baseline.get(field) not in (None, "")
        for field in ("ABC_NET_A", "ABC_NET_B", "ABC_NET_C", "A_NET", "B_NET", "C_NET")
    )
    transmission_available = any(
        baseline.get(field) not in (None, "")
        for field in ("trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2")
    )
    items = [
        ("Vehicle", vehicle),
        ("Baseline VDE ID", str(baseline.get("vde_id") or metadata.get("selected_baseline_vde_id") or "Pending")),
        ("Cycle", str(baseline.get("cycle") or baseline.get("cycle_name") or metadata.get("cycle") or "Pending")),
        (
            "Test mass / basis",
            (
                (_v2_format_value(baseline.get("effective_test_mass_kg") or baseline.get("test_mass_kg"), "mass") or "-")
                + " / "
                + str(baseline.get("vde_mass_basis") or baseline.get("test_mass_basis") or baseline.get("mass_intention") or "Pending")
            ),
        ),
        ("ABC_TOTAL A/B/C", abc_total),
        ("NET status", "Available" if net_available or transmission_available else "Pending"),
    ]
    body = "".join(
        (
            "<div class='vde-context-item'>"
            f"<div class='vde-context-label'>{html.escape(label)}</div>"
            f"<div class='vde-context-value'>{html.escape(str(value or '-'))}</div>"
            "</div>"
        )
        for label, value in items
    )
    st.markdown(f"<div class='vde-context-strip'>{body}</div>", unsafe_allow_html=True)
    with st.expander("Baseline details", expanded=False):
        detail_rows = [
            {"field": "Legislation", "value": str(baseline.get("legislation") or metadata.get("legislation") or "-")},
            {"field": "Roadload source", "value": str(baseline.get("roadload_source_type") or metadata.get("roadload_source_type") or "-")},
            {"field": "Curb mass", "value": _v2_format_value(baseline.get("curb_mass_kg") or baseline.get("mass_kg"), "mass") or "-"},
            {"field": "EPA ETW / TWC", "value": _v2_format_value(baseline.get("inertia_class") or baseline.get("TWC_kg"), "mass") or "-"},
            {"field": "Reference line", "value": str(metadata.get("selected_baseline_label") or baseline.get("baseline_selector") or "-")},
        ]
        render_vde_workbook_table(
            pd.DataFrame(detail_rows),
            title="Baseline details",
            table_id="v21-compact-baseline-details",
        )


def _v2_mass_field_display_override(effective: dict, field_id: str):
    mode = str(effective.get("mass_intention") or "").strip().upper()
    if field_id == "GVWR_kg" and mode != "GVWR":
        return "not used"
    if field_id == "payload_display_kg":
        if mode != "GVWR":
            return "not used"
        if effective.get("payload_display_kg") in (None, ""):
            return "review"
    if field_id in {"GCWR_kg", "trailer_code", "trailer_weight_kg", "trailer_roadload_source", "trailer_A", "trailer_B", "trailer_C", "vehicle_mass_at_gcwr", "trailer_roadload_status"} and mode != "GCWR":
        return "not used"
    if field_id == "TML_kg" and mode != "WLTP_TML":
        return "not used" if str(effective.get("legislation") or "").strip().upper() != "WLTP" else ("unavailable" if effective.get("TML_kg") in (None, "") else None)
    if field_id == "TMH_kg" and mode != "WLTP_TMH":
        return "not used" if str(effective.get("legislation") or "").strip().upper() != "WLTP" else ("unavailable" if effective.get("TMH_kg") in (None, "") else None)
    if field_id == "TWC_kg" and effective.get("TWC_kg") in (None, ""):
        return "unavailable"
    return None


def export_vde_workbook_template() -> pd.DataFrame:
    return pd.DataFrame(columns=["field", *_v2_workbook_column_names().values(), "notes"])


def import_vde_workbook_template(template_df: pd.DataFrame) -> dict:
    # TODO: support CSV/XLSX round-trip into workbook v2 state.
    return {"ok": False, "rows": 0 if template_df is None else len(template_df), "message": "TODO"}


def _v2_field_oriented_display_value(column_id: str, spec: dict) -> str:
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    direct = dict((columns.get(column_id) or {}).get("direct") or {})
    effective = _v2_effective_state(column_id)
    metadata = _v2_metadata_effective(state)
    field_id = str(spec["id"])
    kind = str(spec.get("kind") or "text")
    if field_id == "line_source":
        if column_id == "baseline":
            return str(metadata.get("line_source") or "Existing VDE DB")
        return str(direct.get("line_source") or "New / Insert")
    if field_id == "vde_id":
        return str(effective.get("vde_id") or ("New / Insert" if column_id != "baseline" else ""))
    if field_id == "baseline_selector":
        if column_id == "baseline":
            return str(metadata.get("selected_baseline_label") or effective.get("baseline_selector") or "")
        return str(direct.get("baseline_selector") or "")
    if field_id == "description":
        if column_id == "baseline":
            return str(metadata.get("description") or effective.get("description") or "")
        return str(direct.get("description") or "")
    if field_id == "status":
        return _v2_column_status(column_id)[0]
    if field_id == "walk_from":
        if column_id == "baseline":
            return "-"
        walk_from = str((columns.get(column_id) or {}).get("walk_from") or "")
        return _v2_column_label(walk_from, state) if walk_from else ""
    if field_id == "proposal_direct":
        return str(direct.get("proposal_direct") or "")
    if field_id == "proposal_effective":
        return str(effective.get("proposal_effective") or "")
    if field_id == "legislation":
        if column_id == "baseline":
            return str(metadata.get("legislation") or effective.get("legislation") or "")
        return str(direct.get("legislation") or "")
    if field_id == "model_year":
        if column_id == "baseline":
            return _v2_format_value(metadata.get("model_year") or effective.get("model_year"), kind)
        return _v2_format_value(direct.get("model_year"), kind)
    if field_id == "make":
        if column_id == "baseline":
            return str(metadata.get("make") or effective.get("make") or "")
        return str(direct.get("make") or "")
    if field_id == "model":
        if column_id == "baseline":
            return str(metadata.get("model") or effective.get("model") or "")
        return str(direct.get("model") or "")
    if field_id == "cycle":
        if column_id == "baseline":
            return str(metadata.get("cycle") or effective.get("cycle") or "")
        return str(direct.get("cycle") or "")
    if field_id == "display_units":
        if column_id == "baseline":
            return str(metadata.get("display_units") or "Metric")
        return str(direct.get("display_units") or "")
    if field_id == "roadload_source_type":
        if column_id == "baseline":
            return str(metadata.get("roadload_source_type") or "")
        return str(direct.get("roadload_source_type") or "")
    if field_id == "save_target":
        return _v2_column_label(column_id, state) if str(state.get("save_target") or "") == column_id else ""
    if field_id == "scenario_notes":
        return str(direct.get("scenario_notes") or "")
    if bool(spec.get("readonly")) or kind == "readonly":
        return _v2_format_value(effective.get(field_id), kind)
    display_override = _v2_mass_field_display_override(effective, field_id)
    if display_override is not None:
        return str(display_override)
    if column_id == "baseline":
        return _v2_format_value(direct.get(field_id, effective.get(field_id)), kind)
    return _v2_format_value(direct.get(field_id), kind)


def _v2_build_field_oriented_editor_df(specs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    column_labels = _v2_workbook_column_names()
    for spec in specs:
        row = {
            "field": str(spec["label"]),
            "notes": str(spec.get("notes") or ""),
        }
        for column_id, label in column_labels.items():
            row[label] = _v2_field_oriented_display_value(column_id, spec)
        rows.append(row)
    return pd.DataFrame(rows)


def _v2_apply_field_oriented_editor_df(editor_df: pd.DataFrame, specs: list[dict], *, scenario_workbook: bool = False) -> None:
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    valid_columns = set(_v2_column_ids(state))
    metadata = dict(state.get("metadata") or {})
    rows = list(state.get("rows") or [])
    spec_map = {str(spec["label"]): spec for spec in specs}
    label_to_column = {label: column_id for column_id, label in _v2_workbook_column_names().items()}
    label_to_column.update({label: column_id for column_id, label in _v21_request_column_labels(state).items()})
    save_target = str(state.get("save_target") or _v2_last_column_id(state))

    for row in editor_df.to_dict(orient="records"):
        spec = spec_map.get(str(row.get("field") or ""))
        if not spec:
            continue
        field_id = str(spec["id"])
        kind = str(spec.get("kind") or "text")
        for display_label, column_id in label_to_column.items():
            column = dict(columns.get(column_id) or {})
            direct = dict(column.get("direct") or {})
            raw_value = row.get(display_label)
            raw_text = "" if raw_value is None else str(raw_value).strip()

            if field_id == "save_target":
                normalized_target = _v2_label_to_column_id(raw_text)
                if normalized_target == column_id or raw_text.lower() in {"selected", "save", "x", "true", "yes"}:
                    save_target = column_id
                continue
            if field_id in {"vde_id", "status", "proposal_effective"}:
                continue
            if scenario_workbook and field_id == "line_source":
                if column_id == "baseline":
                    if raw_text:
                        metadata["line_source"] = raw_text
                        baseline = dict(columns.get("baseline") or {})
                        baseline["line_source"] = raw_text
                        if raw_text != "Existing VDE DB":
                            metadata["selected_baseline_vde_id"] = None
                            metadata["selected_baseline_label"] = ""
                            baseline["selected_vde_id"] = None
                        columns["baseline"] = baseline
                    continue
                direct["line_source"] = "New / Insert"
                column["direct"] = direct
                columns[column_id] = column
                continue
            if scenario_workbook and field_id == "baseline_selector":
                if column_id == "baseline":
                    selected_row = _v2_resolve_baseline_selector(raw_text, rows)
                    if selected_row and str(metadata.get("line_source") or "").startswith("Existing"):
                        updated_state = _v2_apply_selected_baseline_row(
                            {
                                **state,
                                "columns": columns,
                                "metadata": metadata,
                            },
                            selected_row,
                        )
                        columns = dict(updated_state.get("columns") or {})
                        metadata = dict(updated_state.get("metadata") or {})
                    else:
                        metadata["selected_baseline_label"] = raw_text
                        metadata["selected_baseline_vde_id"] = None
                    continue
                if raw_text in {"", "-", "inherit", "not used"}:
                    direct.pop("baseline_selector", None)
                else:
                    direct["baseline_selector"] = raw_text
                column["direct"] = direct
                columns[column_id] = column
                continue
            if scenario_workbook and field_id == "walk_from":
                if column_id == "baseline":
                    continue
                walk_from = _v2_label_to_column_id(raw_text)
                allowed_walk_from = set(_v2_allowed_walk_from_ids(column_id, state))
                column["walk_from"] = walk_from if walk_from in allowed_walk_from else ""
                columns[column_id] = column
                continue
            if scenario_workbook and column_id == "baseline" and field_id in {"description", "legislation", "model_year", "make", "model", "cycle", "display_units", "roadload_source_type"}:
                if field_id == "model_year":
                    parsed_year = _v2_parse_value(raw_value, kind)
                    metadata["model_year"] = parsed_year
                elif field_id == "display_units":
                    metadata["display_units"] = raw_text or metadata.get("display_units") or "Metric"
                    st.session_state["unit_system"] = metadata["display_units"]
                else:
                    metadata[field_id] = raw_text
                continue
            if scenario_workbook and column_id == "baseline" and field_id == "proposal_direct":
                continue

            parsed = _v2_parse_value(raw_value, kind)
            if column_id == "baseline":
                if parsed in (None, "", "inherit"):
                    direct.pop(field_id, None)
                else:
                    direct[field_id] = parsed
            else:
                base_value = _v2_base_reference_value(column_id, field_id)
                if parsed in (None, "", "inherit") or parsed == base_value:
                    direct.pop(field_id, None)
                else:
                    direct[field_id] = parsed
            column["direct"] = direct
            columns[column_id] = column

    state["columns"] = columns
    state["save_target"] = save_target
    metadata["save_target"] = save_target
    state["metadata"] = metadata
    _v2_set_state(state)


def _v2_field_config(spec: dict, *, section_key: str, scenario_key: str | None = None, state: dict | None = None) -> dict:
    field_id = str(spec["id"])
    kind = str(spec.get("kind") or "text")
    current_state = state or _v2_state()
    config = {
        "id": field_id,
        "label": str(spec.get("label") or field_id),
        "type": "text",
        "options": [],
        "editable": kind != "readonly" and not bool(spec.get("readonly")),
        "notes": str(spec.get("notes") or ""),
    }
    if kind == "select":
        config["type"] = "select"
        config["options"] = list(spec.get("options") or [])
    elif kind in {"bool", "boolean"}:
        config["type"] = "bool"
    if section_key == "Scenario Workbook":
        if field_id == "baseline_selector":
            config["type"] = "select"
            config["options"] = [""] + [_v2_row_label(row) for row in current_state.get("rows") or []]
        elif field_id == "walk_from":
            config["type"] = "select"
            allowed_ids = _v2_allowed_walk_from_ids(str(scenario_key or ""), current_state)
            config["options"] = [""] + [_v2_column_label(item, current_state) for item in allowed_ids]
        elif field_id == "line_source":
            config["type"] = "select"
            config["options"] = ["Existing VDE DB", "New test ABC_TOTAL"]
        elif field_id == "display_units":
            config["type"] = "select"
            config["options"] = ["Metric", "US customary"]
        elif field_id == "save_target":
            config["type"] = "select"
            config["options"] = [""] + list(_v2_workbook_column_names().values())
        elif field_id in {"vde_id", "status", "proposal_effective"}:
            config["editable"] = False
    return config


def _v2_cell_display_text(value) -> str:
    text = str(value if value not in (None, "") else "-")
    return text


def _v2_cell_class_name(text: str) -> str:
    normalized = str(text or "").strip().lower()
    if normalized in {"-", "", "inherit", "inherited"} or normalized.startswith("inherit from "):
        return "is-inherit"
    if "invalid" in normalized:
        return "is-invalid"
    if any(token in normalized for token in ["missing", "blocked"]):
        return "is-missing"
    if any(token in normalized for token in ["review", "pending", "stale", "partial"]):
        return "is-review"
    if any(token in normalized for token in ["ok", "ready", "defined", "selected", "loaded", "current", "saved available", "draft available", "saved"]):
        return "is-ok"
    if any(token in normalized for token in ["calculated", "new / insert", "not used", "unavailable"]):
        return "is-neutral"
    return ""


def _inject_vde_workbook_table_css() -> None:
    st.markdown(
        """
        <style>
        .vde-workbook-table-card { margin: 0.45rem 0 0.85rem 0; border: 1px solid #cbd5e1; border-radius: 0.55rem; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); background: #ffffff; overflow: hidden; }
        .vde-workbook-table-title { padding: 0.52rem 0.72rem; font-size: 0.83rem; font-weight: 700; color: #15304b; background: linear-gradient(180deg, #f8fbff 0%, #eef5fc 100%); border-bottom: 1px solid #dbe3ee; }
        .vde-workbook-table-wrap, .v2-workbook-table-wrap { overflow-x: auto; }
        table.vde-workbook-table, table.v2-workbook-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; background: #ffffff; }
        .vde-workbook-table th, .vde-workbook-table td, .v2-workbook-table th, .v2-workbook-table td { border: 1px solid #dbe3ee; padding: 0.48rem 0.58rem; vertical-align: top; text-align: left; }
        .vde-workbook-table thead th, .v2-workbook-table thead th { background: #1f3a5f; color: #f8fafc; font-weight: 700; }
        .vde-workbook-table tbody tr:nth-child(even) td, .v2-workbook-table tbody tr:nth-child(even) td { background: #f8fafc; }
        .vde-workbook-table td.is-field, .v2-workbook-table tbody td:first-child { font-weight: 700; color: #15304b; min-width: 12rem; background: #e8f1fb; }
        .vde-workbook-table td.is-notes, .v2-workbook-table tbody td:last-child { color: #7c5a18; min-width: 14rem; background: #fff9e8; }
        .vde-workbook-table td.is-baseline-col { background: #eef2f6; }
        .vde-workbook-table td.is-walked-col { background: #f4f8ff; }
        .vde-workbook-chip, .v2-cell-chip { display: inline-block; padding: 0.1rem 0.46rem; border-radius: 999px; font-size: 0.79rem; border: 1px solid transparent; background: #edf2f7; color: #475467; }
        .vde-workbook-table td.is-source { background: #eef2f7; }
        .vde-workbook-table td.is-inherit, .vde-workbook-table td.is-source.is-inherit { background: #f8fafc; }
        .vde-workbook-table td.is-neutral { background: #eef2ff; }
        .vde-workbook-table td.is-review { background: #fff8db; }
        .vde-workbook-table td.is-missing { background: #fff1f2; }
        .vde-workbook-table td.is-invalid { background: #fee2e2; }
        .vde-workbook-table td.is-ok { background: #ecfdf3; }
        .v2-cell-chip.is-inherit { background: #eef4ff; color: #31507a; border-color: #c9d9f1; }
        .v2-cell-chip.is-ok { background: #e8f7ee; color: #166534; border-color: #86efac; }
        .v2-cell-chip.is-review { background: #fff4d6; color: #a16207; border-color: #facc15; }
        .v2-cell-chip.is-missing { background: #fee2e2; color: #b42318; border-color: #fca5a5; }
        .v2-cell-chip.is-invalid { background: #fecaca; color: #991b1b; border-color: #ef4444; }
        .v2-cell-chip.is-neutral { background: #e7eefc; color: #1d4ed8; border-color: #bfdbfe; }
        .vde-workbook-chip.is-inherit { background: #eef4ff; color: #31507a; border-color: #c9d9f1; }
        .vde-workbook-chip.is-ok { background: #e8f7ee; color: #166534; border-color: #86efac; }
        .vde-workbook-chip.is-review { background: #fff4d6; color: #a16207; border-color: #facc15; }
        .vde-workbook-chip.is-missing { background: #fee2e2; color: #b42318; border-color: #fca5a5; }
        .vde-workbook-chip.is-invalid { background: #fecaca; color: #991b1b; border-color: #ef4444; }
        .vde-workbook-chip.is-neutral { background: #ecebff; color: #5b21b6; border-color: #c4b5fd; }
        .vde-workbook-chip.is-direct { background: #e0ecff; color: #1547a8; border-color: #93c5fd; }
        .vde-workbook-chip.is-inherit, .v2-cell-chip.is-inherit { font-style: normal; font-weight: 600; }
        .v21-detail-banner { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.15rem 0 0.85rem 0; }
        .v21-detail-banner .v2-cell-chip { font-size: 0.78rem; padding: 0.15rem 0.52rem; }
        .v21-detail-head { font-size: 0.8rem; font-weight: 700; color: #475467; padding: 0.42rem 0.56rem; border-radius: 0.38rem; background: #f8fafc; border: 1px solid #d7e0ea; }
        .v21-detail-head.is-baseline { background: #eef2f6; color: #475467; border-color: #d1d9e6; }
        .v21-detail-head.is-target { background: #e3f0ff; color: #1547a8; border-color: #8eb7f2; }
        .v21-detail-head.is-ok { background: #eef8f1; color: #17603a; border-color: #b7e1c2; }
        .v21-detail-head.is-other { background: #f8fafc; color: #667085; }
        .v21-detail-head.is-notes { background: #fff6dc; color: #8a5a10; border-color: #f5d48f; }
        .v21-detail-field { font-weight: 700; color: #15304b; background: #e8f1fb; border: 1px solid #d1e0f0; border-radius: 0.36rem; padding: 0.48rem 0.54rem; }
        .v21-detail-field-sub { font-size: 0.75rem; color: #5b6b82; margin-top: 0.12rem; }
        .v21-detail-note { font-size: 0.79rem; color: #73510d; background: #fff9e8; border: 1px solid #f3dfaa; border-radius: 0.36rem; padding: 0.48rem 0.54rem; }
        .v21-detail-row { padding: 0.14rem 0 0.28rem 0; }
        .v21-detail-readonly { min-height: 2.2rem; display: flex; align-items: center; padding: 0.18rem 0; }
        .v21-workbook-frame { border: 1px solid #cbd5e1; border-radius: 0.55rem; overflow: hidden; margin: 0.45rem 0 0.85rem 0; background: #ffffff; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); }
        .v21-workbook-header { background: #1f3a5f; color: #f8fafc; font-size: 0.8rem; font-weight: 700; padding: 0.52rem 0.58rem; border-right: 1px solid rgba(255,255,255,0.08); border-bottom: 1px solid #dbe3ee; border-radius: 0; min-height: 2.35rem; display: flex; align-items: center; }
        .v21-workbook-fieldcell { font-weight: 700; color: #15304b; background: #e8f1fb; border: 1px solid #d1e0f0; border-radius: 0.36rem; padding: 0.48rem 0.54rem; min-height: 2.35rem; display: flex; align-items: center; }
        .v21-workbook-notecell { font-size: 0.79rem; color: #73510d; background: #fff9e8; border: 1px solid #f3dfaa; border-radius: 0.36rem; padding: 0.48rem 0.54rem; min-height: 2.35rem; display: flex; align-items: center; }
        .v21-workbook-row { padding: 0.2rem 0.2rem 0.05rem 0.2rem; border-top: 1px solid #edf2f7; }
        .v21-workbook-row.is-proposal { background: linear-gradient(180deg, #fbfdff 0%, #f6f9fe 100%); }
        .v21-workbook-cell { min-height: 2.35rem; display: flex; align-items: center; padding: 0.1rem 0; }
        .v21-workbook-cell.is-baseline { background: #eef2f6; border: 1px solid #d1d9e6; border-radius: 0.36rem; padding: 0.35rem 0.45rem; }
        .v21-workbook-cell.is-walked { background: #f4f8ff; border: 1px solid #d7e6fb; border-radius: 0.36rem; padding: 0.35rem 0.45rem; }
        .v21-workbook-subnote { font-size: 0.74rem; color: #667085; margin-top: 0.12rem; }
        .v21-setup-card { border: 1px solid #dbe3ee; border-radius: 0.55rem; background: #ffffff; padding: 0.8rem 0.85rem; min-height: 100%; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
        .v21-setup-card.is-baseline { background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%); }
        .v21-setup-card.is-walked { background: linear-gradient(180deg, #fbfdff 0%, #f6f9fe 100%); }
        .v21-setup-card-title { font-size: 0.9rem; font-weight: 700; color: #15304b; margin-bottom: 0.65rem; }
        .v21-setup-label { font-size: 0.74rem; font-weight: 700; color: #5b6b82; text-transform: uppercase; letter-spacing: 0.02em; margin: 0.2rem 0 0.18rem 0; }
        .v21-setup-readonly { min-height: 2.25rem; display: flex; align-items: center; padding: 0.45rem 0.58rem; border-radius: 0.4rem; border: 1px solid #d7e0ea; background: #f8fafc; color: #15304b; }
        .v21-setup-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.45rem; }
        .v21-domain-card { border: 1px solid #dbe3ee; border-radius: 0.5rem; background: #fbfdff; padding: 0.62rem 0.72rem; min-height: 5.1rem; margin-bottom: 0.35rem; }
        .v21-domain-card.is-active { border-color: #8eb7f2; background: linear-gradient(180deg, #eef6ff 0%, #f8fbff 100%); box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.12); }
        .v21-domain-card-title { font-size: 0.82rem; font-weight: 700; color: #15304b; margin-bottom: 0.18rem; }
        .v21-domain-card-detail { margin-top: 0.18rem; color: #667085; font-size: 0.72rem; line-height: 1.25; overflow-wrap: anywhere; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _v2_render_light_workbook_styles() -> None:
    _inject_vde_workbook_table_css()


def _vde_cell_class(column_name, value) -> str:
    classes: list[str] = []
    text = _v2_cell_display_text("-" if pd.isna(value) else value)
    value_class = _v2_cell_class_name(text)
    if value_class:
        classes.append(value_class)
    normalized_text = text.strip().lower()
    normalized_column = str(column_name or "").strip().lower()
    if normalized_column == "baseline":
        classes.append("is-baseline-col")
    elif normalized_column.startswith("walked"):
        classes.append("is-walked-col")
    if "source" in normalized_column or normalized_text in {"manual", "workbook", "vde_db", "existing vde db"}:
        classes.append("is-source")
    if any(token in normalized_text for token in ["calculated", "computed"]):
        classes.append("is-neutral")
    if normalized_text.startswith("inherited from ") or normalized_text in {"inherit", "inherited"}:
        classes.append("is-inherit")
    if normalized_text.startswith("prop #") or normalized_column in {"proposal direct", "field / proposal"} and normalized_text.startswith("prop #"):
        classes.append("is-direct")
    return " ".join(dict.fromkeys(item for item in classes if item))


def render_vde_workbook_table(df, *, title: str | None = None, table_id: str | None = None) -> None:
    _inject_vde_workbook_table_css()
    if isinstance(df, pd.Series):
        df = df.to_frame().reset_index()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    working_df = df.copy()
    headers = [str(column) for column in working_df.columns]
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in headers)
    body_rows: list[str] = []
    for row in working_df.itertuples(index=False, name=None):
        cell_html: list[str] = []
        last_index = len(headers) - 1
        for index, (column_name, value) in enumerate(zip(headers, row)):
            display_text = "-" if pd.isna(value) or value in (None, "") else str(value)
            cell_class = _vde_cell_class(column_name, display_text)
            if index == 0:
                cell_class = f"{cell_class} is-field".strip()
            if str(column_name).strip().lower() == "notes" or index == last_index and str(column_name).strip().lower() in {"notes", "detail"}:
                cell_class = f"{cell_class} is-notes".strip()
            chip_class = _v2_cell_class_name(display_text)
            if "is-direct" in cell_class and chip_class in {"", None}:
                chip_class = "is-direct"
            chip_html = (
                f"<span class='vde-workbook-chip {chip_class}'>{html.escape(display_text)}</span>"
                if chip_class
                else html.escape(display_text)
            )
            class_attr = f" class='{cell_class}'" if cell_class else ""
            cell_html.append(f"<td{class_attr}>{chip_html}</td>")
        body_rows.append("<tr>" + "".join(cell_html) + "</tr>")
    safe_table_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(table_id or "").strip()).strip("-")
    id_attr = f" id='{safe_table_id}'" if safe_table_id else ""
    title_html = f"<div class='vde-workbook-table-title'>{html.escape(str(title))}</div>" if title else ""
    st.markdown(
        "<div class='vde-workbook-table-card'>"
        f"{title_html}"
        f"<div class='vde-workbook-table-wrap'><table{id_attr} class='vde-workbook-table'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div></div>",
        unsafe_allow_html=True,
    )


def _v2_render_light_workbook_table(section_key: str, specs: list[dict]) -> None:
    column_labels = _v2_workbook_column_names()
    rows: list[dict] = []
    for spec in specs:
        field_label = str(spec.get("label") or spec["id"])
        notes = str(spec.get("notes") or "")
        row = {"field": field_label}
        for column_id in column_labels:
            display_text = _v2_cell_display_text(_v2_field_oriented_display_value(column_id, spec))
            row[column_labels[column_id]] = display_text
        row["notes"] = notes
        rows.append(row)
    render_vde_workbook_table(
        pd.DataFrame(rows),
        title=section_key,
        table_id=f"vde-workbook-{section_key.lower().replace(' ', '-')}",
    )


def _v2_apply_single_field_value(section_key: str, scenario_key: str, field_id: str, raw_value) -> None:
    specs = VDE_WORKBOOK_V2_MATRIX_SPECS if section_key == "Scenario Workbook" else list(VDE_WORKBOOK_V2_SECTION_SPECS.get(section_key) or [])
    spec = next((item for item in specs if str(item["id"]) == field_id), None)
    if not spec:
        return
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    valid_columns = set(_v2_column_ids(state))
    metadata = dict(state.get("metadata") or {})
    rows = list(state.get("rows") or [])
    column = dict(columns.get(scenario_key) or {})
    direct = dict(column.get("direct") or {})
    kind = str(spec.get("kind") or "text")
    raw_text = "" if raw_value is None else str(raw_value).strip()

    if field_id == "save_target":
        state["save_target"] = _v2_label_to_column_id(raw_text) if raw_text else state.get("save_target", _v2_last_column_id(state))
        metadata["save_target"] = state["save_target"]
        state["metadata"] = metadata
        _v2_set_state(state)
        return
    if field_id in {"vde_id", "status", "proposal_effective"}:
        return
    if section_key == "Scenario Workbook" and field_id == "line_source":
        if scenario_key == "baseline":
            if raw_text:
                metadata["line_source"] = raw_text
                column["line_source"] = raw_text
                if raw_text != "Existing VDE DB":
                    metadata["selected_baseline_vde_id"] = None
                    metadata["selected_baseline_label"] = ""
                    column["selected_vde_id"] = None
                columns["baseline"] = column
        else:
            direct["line_source"] = "New / Insert"
            column["direct"] = direct
            columns[scenario_key] = column
        state["columns"] = columns
        state["metadata"] = metadata
        _v2_set_state(state)
        return
    if section_key == "Scenario Workbook" and field_id == "baseline_selector":
        if scenario_key == "baseline":
            selected_row = _v2_resolve_baseline_selector(raw_text, rows)
            if selected_row and str(metadata.get("line_source") or "").startswith("Existing"):
                updated_state = _v2_apply_selected_baseline_row(
                    {
                        **state,
                        "columns": columns,
                        "metadata": metadata,
                    },
                    selected_row,
                )
                columns = dict(updated_state.get("columns") or {})
                metadata = dict(updated_state.get("metadata") or {})
            else:
                metadata["selected_baseline_label"] = raw_text
                metadata["selected_baseline_vde_id"] = None
        else:
            if raw_text in {"", "-", "inherit", "not used"}:
                direct.pop("baseline_selector", None)
            else:
                direct["baseline_selector"] = raw_text
            column["direct"] = direct
            columns[scenario_key] = column
        state["columns"] = columns
        state["metadata"] = metadata
        _v2_set_state(state)
        return
    if section_key == "Scenario Workbook" and field_id == "walk_from":
        if scenario_key == "baseline":
            return
        walk_from = _v2_label_to_column_id(raw_text)
        allowed_walk_from = set(_v2_allowed_walk_from_ids(scenario_key, state))
        column["walk_from"] = walk_from if walk_from in allowed_walk_from else ""
        columns[scenario_key] = column
        state["columns"] = columns
        _v2_set_state(state)
        return
    if section_key == "Scenario Workbook" and scenario_key == "baseline" and field_id in {"description", "legislation", "model_year", "make", "model", "cycle", "display_units", "roadload_source_type"}:
        parsed = _v2_parse_value(raw_value, kind)
        if field_id == "display_units":
            metadata["display_units"] = raw_text or metadata.get("display_units") or "Metric"
            st.session_state["unit_system"] = metadata["display_units"]
        elif field_id == "model_year":
            metadata["model_year"] = parsed
        else:
            metadata[field_id] = parsed if parsed is not None else raw_text
        state["metadata"] = metadata
        _v2_set_state(state)
        return
    if section_key == "Scenario Workbook" and scenario_key == "baseline" and field_id == "proposal_direct":
        return

    parsed = _v2_parse_value(raw_value, kind)
    if scenario_key == "baseline":
        if parsed in (None, "", "inherit"):
            direct.pop(field_id, None)
        else:
            direct[field_id] = parsed
    else:
        base_value = _v2_base_reference_value(scenario_key, field_id)
        if parsed in (None, "", "inherit") or parsed == base_value:
            direct.pop(field_id, None)
        else:
            direct[field_id] = parsed
    column["direct"] = direct
    columns[scenario_key] = column
    state["columns"] = columns
    state["metadata"] = metadata
    _v2_set_state(state)


def _v2_render_light_editor_panel(section_key: str, specs: list[dict]) -> None:
    st.caption("Edit selected field")
    state = _v2_state()
    scenario_labels = _v2_workbook_column_names()
    scenario_options = list(scenario_labels.keys())
    current_scenario = str(st.session_state.get(f"v2_{section_key}_scenario", "baseline"))
    if current_scenario not in scenario_options:
        current_scenario = "baseline"
    field_options = [str(spec["id"]) for spec in specs]
    current_field = str(st.session_state.get(f"v2_{section_key}_field", field_options[0] if field_options else ""))
    if current_field not in field_options and field_options:
        current_field = field_options[0]
    selector_cols = st.columns([1, 1])
    chosen_scenario = selector_cols[0].selectbox(
        "Select scenario column",
        scenario_options,
        index=scenario_options.index(current_scenario),
        format_func=lambda value: scenario_labels.get(value, value),
        key=f"v2_{section_key}_scenario",
    )
    chosen_field = selector_cols[1].selectbox(
        "Select field",
        field_options,
        index=field_options.index(current_field) if current_field in field_options else 0,
        format_func=lambda value: next((str(spec.get('label') or value) for spec in specs if str(spec['id']) == value), value),
        key=f"v2_{section_key}_field",
    )
    if not chosen_field:
        return
    spec = next(spec for spec in specs if str(spec["id"]) == chosen_field)
    config = _v2_field_config(spec, section_key=section_key, scenario_key=chosen_scenario)
    current_display = _v2_field_oriented_display_value(chosen_scenario, spec)
    st.caption(config["notes"] or " ")
    value_container = st.container()
    new_value = current_display
    if not config["editable"]:
        value_container.text_input("Value", value=str(current_display), disabled=True, key=f"v2_{section_key}_{chosen_scenario}_{chosen_field}_readonly")
        st.info("Calculated field.")
        return
    if config["type"] == "select":
        options = list(config["options"] or [""])
        normalized_display = "" if str(current_display) == "-" else str(current_display)
        if chosen_field == "walk_from" and normalized_display not in options:
            normalized_display = ""
        elif str(current_display) not in options and str(current_display) not in {"-", ""}:
            options.append(str(current_display))
        selected = value_container.selectbox(
            "Value",
            options,
            index=options.index(normalized_display) if normalized_display in options else 0,
            key=f"v2_{section_key}_{chosen_scenario}_{chosen_field}_select",
        )
        new_value = selected
    elif config["type"] == "bool":
        checked = str(current_display).strip().lower() in {"true", "1", "yes", "x"}
        new_value = value_container.checkbox(
            "Value",
            value=checked,
            key=f"v2_{section_key}_{chosen_scenario}_{chosen_field}_bool",
        )
    else:
        entered = "" if str(current_display) == "-" else str(current_display)
        new_value = value_container.text_input(
            "Value",
            value=entered,
            key=f"v2_{section_key}_{chosen_scenario}_{chosen_field}_text",
        )
    if st.button("Apply value", key=f"v2_{section_key}_{chosen_scenario}_{chosen_field}_apply"):
        _v2_apply_single_field_value(section_key, chosen_scenario, chosen_field, new_value)
        st.success(f"Updated {config['label']} for {scenario_labels.get(chosen_scenario, chosen_scenario)}.")


def _v2_is_cell_editable(section_key: str, scenario_key: str, spec: dict) -> bool:
    config = _v2_field_config(spec, section_key=section_key, scenario_key=scenario_key)
    if not config["editable"]:
        return False
    field_id = str(spec["id"])
    if section_key == "Scenario Workbook":
        if field_id in {"vde_id", "status", "proposal_effective"}:
            return False
        if field_id == "baseline_selector" and scenario_key != "baseline":
            return False
        if field_id == "line_source" and scenario_key != "baseline":
            return False
        if field_id == "walk_from" and scenario_key == "baseline":
            return False
        if field_id == "proposal_direct" and scenario_key == "baseline":
            return False
    return True


def _v2_render_form_cell(*, section_key: str, scenario_key: str, spec: dict, current_value, host):
    field_id = str(spec["id"])
    config = _v2_field_config(spec, section_key=section_key, scenario_key=scenario_key)
    widget_key = f"v2_grid_{section_key}_{field_id}_{scenario_key}"
    if not _v2_is_cell_editable(section_key, scenario_key, spec):
        display_text = _v2_cell_display_text(current_value)
        class_name = _v2_cell_class_name(display_text)
        host.markdown(
            f"<span class='v2-cell-chip {class_name}'>{html.escape(display_text)}</span>",
            unsafe_allow_html=True,
        )
        return current_value
    if config["type"] == "select":
        options = list(config["options"] or [""])
        normalized = "" if str(current_value) == "-" else str(current_value)
        if field_id == "walk_from" and normalized not in options:
            normalized = ""
        elif str(current_value) not in options and str(current_value) not in {"", "-"}:
            options.append(str(current_value))
        if normalized not in options:
            options = [""] + options
        return host.selectbox(
            field_id,
            options,
            index=options.index(normalized) if normalized in options else 0,
            key=widget_key,
            label_visibility="collapsed",
        )
    if config["type"] == "bool":
        checked = str(current_value).strip().lower() in {"true", "1", "yes", "x"}
        return host.checkbox(
            field_id,
            value=checked,
            key=widget_key,
            label_visibility="collapsed",
        )
    text_value = "" if str(current_value) == "-" else str(current_value)
    return host.text_input(
        field_id,
        value=text_value,
        key=widget_key,
        label_visibility="collapsed",
        placeholder="inherit" if scenario_key != "baseline" else "",
    )


def _v2_render_lightweight_workbook_grid(section_key: str, specs: list[dict], *, scenario_workbook: bool = False) -> None:
    _v2_render_light_workbook_styles()
    section_df = _v2_build_field_oriented_editor_df(specs)
    row_lookup = {str(row.get("field") or ""): row for row in section_df.to_dict(orient="records")}
    column_labels = _v2_workbook_column_names()
    width_spec = [1.4] + [1.5 for _ in column_labels] + [1.2]
    form_key = re.sub(r"[^a-z0-9]+", "_", section_key.lower()).strip("_")
    with st.form(f"v2_{form_key}_grid_form"):
        header = st.columns(width_spec)
        header[0].markdown("**field**")
        for index, display_label in enumerate(column_labels.values(), start=1):
            header[index].markdown(f"**{display_label}**")
        header[-1].markdown("**notes**")
        edited_rows: list[dict] = []
        for spec in specs:
            field_label = str(spec.get("label") or spec["id"])
            row = dict(row_lookup.get(field_label) or {})
            cols = st.columns(width_spec)
            cols[0].markdown(field_label)
            edited_row = {"field": field_label, "notes": str(spec.get("notes") or "")}
            for index, (scenario_key, display_label) in enumerate(column_labels.items(), start=1):
                current_value = row.get(display_label, "")
                edited_row[display_label] = _v2_render_form_cell(
                    section_key=section_key,
                    scenario_key=scenario_key,
                    spec=spec,
                    current_value=current_value,
                    host=cols[index],
                )
            cols[-1].caption(str(spec.get("notes") or ""))
            edited_rows.append(edited_row)
        apply = st.form_submit_button("Apply changes")
    if apply:
        _v2_apply_field_oriented_editor_df(pd.DataFrame(edited_rows), specs, scenario_workbook=scenario_workbook)
        st.session_state["v2_flash_message"] = f"{section_key} updated."
        st.rerun()


def _v2_render_matrix_column_actions() -> None:
    column_labels = _v2_workbook_column_names()
    width_spec = [1.4] + [1.5 for _ in column_labels] + [1.2]
    cols = st.columns(width_spec)
    cols[0].caption("column actions")
    for index, scenario_key in enumerate(column_labels.keys(), start=1):
        if scenario_key == "baseline":
            cols[index].caption("reference line")
            continue
        if cols[index].button("Remove Column", key=f"v2_remove_column_{scenario_key}", use_container_width=True):
            removed_label = _v2_column_label(scenario_key)
            _v2_remove_walked_column(scenario_key)
            st.session_state["v2_flash_message"] = f"Removed {removed_label}."
            st.rerun()
    cols[-1].caption(" ")


def _v2_render_metadata_menu() -> None:
    state = _v2_state()
    metadata = _v2_metadata_effective(state)
    st.caption("Define metadata once here. The workbook below stays light and only carries scenario/proposal deltas.")
    with st.form("vde_v2_metadata_form"):
        row1 = st.columns(3)
        line_source = row1[0].selectbox(
            "Line source",
            ["Existing VDE DB", "New test ABC_TOTAL"],
            index=["Existing VDE DB", "New test ABC_TOTAL"].index(str(metadata.get("line_source") or "Existing VDE DB")) if str(metadata.get("line_source") or "Existing VDE DB") in ["Existing VDE DB", "New test ABC_TOTAL"] else 0,
        )
        selected_id = row1[1].number_input(
            "selected_baseline_vde_id",
            min_value=0,
            step=1,
            value=int(to_float(metadata.get("selected_baseline_vde_id"), 0) or 0),
            disabled=line_source != "Existing VDE DB",
        )
        display_units = row1[2].selectbox(
            "display_units",
            ["Metric", "US customary"],
            index=["Metric", "US customary"].index(str(metadata.get("display_units") or "Metric")) if str(metadata.get("display_units") or "Metric") in ["Metric", "US customary"] else 0,
        )

        row2 = st.columns(4)
        legislation = row2[0].selectbox(
            "legislation",
            ["", "EPA", "WLTP", "BRA"],
            index=["", "EPA", "WLTP", "BRA"].index(str(metadata.get("legislation") or "")) if str(metadata.get("legislation") or "") in ["", "EPA", "WLTP", "BRA"] else 0,
        )
        model_year = row2[1].number_input(
            "model_year",
            min_value=0,
            step=1,
            value=int(to_float(metadata.get("model_year"), 0) or 0),
        )
        make = row2[2].text_input("make", value=str(metadata.get("make") or ""))
        model = row2[3].text_input("model", value=str(metadata.get("model") or ""))

        row3 = st.columns(4)
        cycle = row3[0].text_input("cycle", value=str(metadata.get("cycle") or ""))
        description = row3[1].text_input("description / baseline proposal label", value=str(metadata.get("description") or ""))
        roadload_source_type = row3[2].text_input("roadload source type", value=str(metadata.get("roadload_source_type") or ""))
        save_target = row3[3].selectbox(
            "save/update target",
            list(_v2_column_ids(state)),
            index=list(_v2_column_ids(state)).index(str(state.get("save_target") or _v2_last_column_id(state))) if str(state.get("save_target") or _v2_last_column_id(state)) in _v2_column_ids(state) else max(len(_v2_column_ids(state)) - 1, 0),
            format_func=lambda value: _v2_column_label(value, state),
        )

        submitted = st.form_submit_button("Apply metadata")

    if submitted:
        raw_state = _v2_state()
        next_metadata = dict(raw_state.get("metadata") or {})
        next_metadata["line_source"] = line_source
        next_metadata["selected_baseline_vde_id"] = int(selected_id) or None
        next_metadata["legislation"] = legislation
        next_metadata["model_year"] = int(model_year) if int(model_year) > 0 else None
        next_metadata["make"] = str(make).strip()
        next_metadata["model"] = str(model).strip()
        next_metadata["cycle"] = str(cycle).strip()
        next_metadata["display_units"] = display_units
        next_metadata["description"] = str(description).strip()
        next_metadata["roadload_source_type"] = str(roadload_source_type).strip()
        next_metadata["save_target"] = save_target
        if line_source == "Existing VDE DB":
            selected_row = _v2_find_row_by_id(raw_state.get("rows") or [], next_metadata.get("selected_baseline_vde_id"))
            if selected_row:
                auto_defaults = _v2_row_metadata_defaults(selected_row)
                for field_id in ("legislation", "model_year", "make", "model", "cycle"):
                    if next_metadata.get(field_id) in (None, "", 0):
                        next_metadata[field_id] = auto_defaults.get(field_id)
                if next_metadata.get("description") in (None, ""):
                    next_metadata["description"] = auto_defaults.get("description") or ""
                if next_metadata.get("roadload_source_type") in (None, ""):
                    next_metadata["roadload_source_type"] = "Baseline ABC"
        raw_state["metadata"] = next_metadata
        raw_state["save_target"] = save_target
        st.session_state["unit_system"] = display_units
        _v2_set_state(raw_state)
        st.success("Metadata applied to VDE Setup v2.")


def _v2_render_db_browser() -> None:
    state = _v2_state()
    rows = list(state.get("rows") or [])
    if not rows:
        st.info("No VDE rows available.")
        return
    df = pd.DataFrame(rows)
    filter_cols = st.columns(4)
    make_filter = filter_cols[0].text_input("Filter make", value=str(st.session_state.get("v2_db_browser_make", "")))
    model_filter = filter_cols[1].text_input("Filter model", value=str(st.session_state.get("v2_db_browser_model", "")))
    year_filter = filter_cols[2].text_input("Filter model_year", value=str(st.session_state.get("v2_db_browser_year", "")))
    legislation_filter = filter_cols[3].text_input("Filter legislation", value=str(st.session_state.get("v2_db_browser_legislation", "")))
    st.session_state["v2_db_browser_make"] = make_filter
    st.session_state["v2_db_browser_model"] = model_filter
    st.session_state["v2_db_browser_year"] = year_filter
    st.session_state["v2_db_browser_legislation"] = legislation_filter

    filtered = df.copy()
    if make_filter.strip():
        filtered = filtered[filtered["make"].fillna("").astype(str).str.contains(make_filter.strip(), case=False, na=False)]
    if model_filter.strip():
        filtered = filtered[filtered["model"].fillna("").astype(str).str.contains(model_filter.strip(), case=False, na=False)]
    if year_filter.strip():
        filtered = filtered[filtered["year"].fillna("").astype(str).str.contains(year_filter.strip(), case=False, na=False)]
    if legislation_filter.strip():
        filtered = filtered[filtered["legislation"].fillna("").astype(str).str.contains(legislation_filter.strip(), case=False, na=False)]

    browser_cols = [col for col in ["id", "make", "model", "year", "legislation", "mass_kg", "test_mass_kg", "A", "B", "C", "cycle_name", "notes"] if col in filtered.columns]
    st.dataframe(filtered[browser_cols], use_container_width=True, hide_index=True)

    filtered_rows = filtered.to_dict(orient="records")
    labels = [_v2_row_label(row) for row in filtered_rows]
    selected_label = st.selectbox(
        "Choose baseline row",
        options=[""] + labels,
        key="v2_db_browser_selected_label",
    )
    if st.button("Use selected VDE row", key="v2_db_browser_use_selected", disabled=not selected_label):
        selected_row = next((row for row in filtered_rows if _v2_row_label(row) == selected_label), None)
        if selected_row:
            next_state = _v2_state()
            next_state = _v2_apply_selected_baseline_row(next_state, selected_row)
            metadata = dict(next_state.get("metadata") or {})
            metadata["display_units"] = str(metadata.get("display_units") or normalize_unit_system(st.session_state.get("unit_system") or "Metric"))
            next_state["metadata"] = metadata
            st.session_state["unit_system"] = metadata["display_units"]
            _v2_set_state(next_state)
            st.success(f"Baseline VDE-{int(to_float(selected_row.get('id'), 0) or 0)} loaded into metadata.")


def _v2_label_to_column_id(label: str) -> str:
    normalized = str(label or "").strip()
    request_labels = _v21_request_column_labels()
    for column_id, display in _v2_workbook_column_names().items():
        if normalized == column_id:
            return column_id
        if normalized == display:
            return column_id
        if normalized == request_labels.get(column_id):
            return column_id
    return normalized.lower().replace(" ", "_").replace("#", "")


def _v2_editor_display_value(value, kind: str):
    if isinstance(value, str):
        return value
    if kind == "mass":
        display = to_display(value, "mass", _current_unit_system())
        return None if display is None else float(display)
    if kind == "force":
        display = to_display(value, "force", _current_unit_system())
        return None if display is None else float(display)
    if kind == "force_per_speed":
        display = to_display(value, "force_per_speed", _current_unit_system())
        return None if display is None else float(display)
    if kind == "force_per_speed_squared":
        display = to_display(value, "force_per_speed_squared", _current_unit_system())
        return None if display is None else float(display)
    if kind == "rrc":
        display = to_display(value, "rrc", _current_unit_system())
        return None if display is None else float(display)
    if kind == "int":
        numeric = to_float(value)
        return None if numeric is None else int(numeric)
    if kind == "float":
        numeric = to_float(value)
        return None if numeric is None else float(numeric)
    return "" if value in (None, "") else str(value)


def _v2_base_reference_value(column_id: str, field_id: str):
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    if column_id == "baseline":
        baseline = dict(columns.get("baseline") or {})
        if str(baseline.get("line_source") or "Existing VDE DB") == "Existing VDE DB":
            selected_id = baseline.get("selected_vde_id")
            row = next((item for item in state.get("rows") or [] if int(item.get("id")) == int(selected_id or 0)), None)
            if row:
                return _v2_row_to_effective_state(row).get(field_id)
        return None
    walk_from = str(dict(columns.get(column_id) or {}).get("walk_from") or "baseline")
    if walk_from not in _v2_column_ids(state):
        walk_from = "baseline"
    return _v2_effective_state(walk_from).get(field_id)


def _v2_direct_value(column_id: str, field_id: str, kind: str):
    state = _v2_state()
    column = dict((state.get("columns") or {}).get(column_id) or {})
    direct = dict(column.get("direct") or {})
    if field_id in {"line_source", "baseline_selector", "walk_from"}:
        if column_id == "baseline" and field_id == "line_source":
            return str(column.get("line_source") or "Existing VDE DB")
        if column_id == "baseline" and field_id == "baseline_selector":
            selected_id = column.get("selected_vde_id")
            row = next((item for item in state.get("rows") or [] if int(item.get("id")) == int(selected_id or 0)), None)
            return _v2_row_label(row) if row else ""
        if field_id == "walk_from":
            return _v2_column_label(str(column.get("walk_from") or ""), state)
    if column_id == "baseline":
        return _v2_editor_display_value(direct.get(field_id, _v2_effective_state(column_id).get(field_id)), kind)
    return _v2_editor_display_value(direct.get(field_id), kind)


def _v2_build_row_oriented_editor_df(specs: list[dict], *, include_matrix_fields: bool = False) -> pd.DataFrame:
    rows: list[dict] = []
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    for column_id in _v2_column_ids(state):
        effective = _v2_effective_state(column_id)
        role = "Baseline" if column_id == "baseline" else "Walked"
        row = {
            "scenario": _v2_column_label(column_id, state),
            "role": role,
        }
        if include_matrix_fields:
            row["line_source"] = str(dict(columns.get(column_id) or {}).get("line_source") or ("Existing VDE DB" if column_id == "baseline" else "New / Insert"))
            row["vde_id"] = str(effective.get("vde_id") or ("New / Insert" if column_id != "baseline" else ""))
            row["baseline_selector"] = _v2_direct_value(column_id, "baseline_selector", "text")
            row["walk_from"] = "" if column_id == "baseline" else _v2_column_label(str(dict(columns.get(column_id) or {}).get("walk_from") or ""), state)
            row["description"] = _v2_direct_value(column_id, "description", "text")
            status_label, _ = _v2_column_status(column_id)
            row["status"] = status_label
            row["proposal_direct"] = _v2_direct_value(column_id, "proposal_direct", "text")
            row["proposal_effective"] = str(effective.get("proposal_effective") or "")
            row["legislation"] = _v2_direct_value(column_id, "legislation", "select")
            row["model_year"] = _v2_direct_value(column_id, "model_year", "int")
            row["make"] = _v2_direct_value(column_id, "make", "text")
            row["model"] = _v2_direct_value(column_id, "model", "text")
            row["cycle"] = _v2_direct_value(column_id, "cycle", "text")
        for spec in specs:
            field_id = str(spec["id"])
            if field_id in row:
                continue
            row[field_id] = _v2_direct_value(column_id, field_id, str(spec.get("kind") or "text"))
        rows.append(row)
    return pd.DataFrame(rows)


def _v2_apply_row_oriented_editor_df(editor_df: pd.DataFrame, specs: list[dict], *, include_matrix_fields: bool = False) -> None:
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    label_to_row = {_v2_row_label(row): row for row in state.get("rows") or []}
    spec_map = {str(spec["id"]): spec for spec in specs}

    for row in editor_df.to_dict(orient="records"):
        column_id = _v2_label_to_column_id(row.get("scenario"))
        if column_id not in _v2_column_ids(state):
            continue
        column = dict(columns.get(column_id) or {})
        direct = dict(column.get("direct") or {})

        if include_matrix_fields:
            line_source = str(row.get("line_source") or "").strip() or ("Existing VDE DB" if column_id == "baseline" else "New / Insert")
            if column_id == "baseline":
                if line_source in {"New / Insert", "New test / New ABC_TOTAL line"}:
                    line_source = "New test ABC_TOTAL"
                column["line_source"] = line_source
                selector_label = str(row.get("baseline_selector") or "").strip()
                selected_row = label_to_row.get(selector_label)
                column["selected_vde_id"] = int(selected_row.get("id")) if selected_row else None
            else:
                direct["line_source"] = line_source
                walk_from_label = str(row.get("walk_from") or "").strip()
                walk_from = _v2_label_to_column_id(walk_from_label)
                if walk_from in _v2_column_ids(state) and walk_from != column_id:
                    column["walk_from"] = walk_from
                else:
                    column["walk_from"] = ""
            for field_id, kind in [
                ("description", "text"),
                ("proposal_direct", "text"),
                ("legislation", "select"),
                ("model_year", "int"),
                ("make", "text"),
                ("model", "text"),
                ("cycle", "text"),
            ]:
                parsed = _v2_parse_value(row.get(field_id), kind)
                base_value = _v2_base_reference_value(column_id, field_id)
                if column_id != "baseline" and (parsed in (None, "", "inherit") or parsed == base_value):
                    direct.pop(field_id, None)
                else:
                    direct[field_id] = parsed if parsed is not None else ""

        for field_id, spec in spec_map.items():
            parsed = _v2_parse_value(row.get(field_id), str(spec.get("kind") or "text"))
            base_value = _v2_base_reference_value(column_id, field_id)
            if column_id != "baseline" and (parsed in (None, "", "inherit") or parsed == base_value):
                direct.pop(field_id, None)
            else:
                direct[field_id] = parsed if parsed is not None else ""

        column["direct"] = direct
        columns[column_id] = column

    state["columns"] = columns
    _v2_set_state(state)


def _v2_render_transposed_view(editor_df: pd.DataFrame, fields: list[str], *, caption: str) -> None:
    workbook_view = editor_df.set_index("scenario")[fields].T.reset_index()
    workbook_view.rename(columns={"index": "field"}, inplace=True)
    render_vde_workbook_table(workbook_view, title=caption, table_id=f"transposed-{caption.lower().replace(' ', '-')}")


def _v2_render_matrix_editor() -> None:
    flash_message = str(st.session_state.pop("v2_flash_message", "") or "").strip()
    if flash_message:
        st.success(flash_message)
    st.caption("Scenario Workbook is the source of truth for metadata, baseline selection, proposals, and walked inheritance.")
    action_cols = st.columns([1, 6])
    if action_cols[0].button("+ Add Column", key="v2_add_worked_column_button"):
        new_key = _v2_add_walked_column()
        st.session_state["v2_flash_message"] = f"Added {_v2_column_label(new_key)}."
        st.rerun()
    action_cols[1].caption("Line source is only chosen on Baseline. Every walked column is treated as a new proposal line and saves as `New / Insert`.")
    _v2_render_matrix_column_actions()
    _v2_render_lightweight_workbook_grid("Scenario Workbook", VDE_WORKBOOK_V2_MATRIX_SPECS, scenario_workbook=True)
    with st.expander("Browse VDE DB lines"):
        st.caption("Use this browser only as lookup support, then copy or apply the baseline line back into the Scenario Workbook.")
        if st.button("Load DB preview", key="v2_load_db_preview_button"):
            st.session_state["v2_show_db_preview"] = True
        if st.session_state.get("v2_show_db_preview"):
            _v2_render_db_browser()


def _v2_render_section_editor(section_name: str) -> None:
    specs = list(VDE_WORKBOOK_V2_SECTION_SPECS.get(section_name) or [])
    st.caption(f"{section_name} workbook. Baseline shows the effective reference state; walked columns capture only direct deltas.")
    _v2_render_lightweight_workbook_grid(section_name, specs)
    column_ids = _v2_column_ids()
    preview_map = {column_id: _v2_cached_preview(column_id) for column_id in column_ids}
    status_rows = [
        {
            "scenario": _v2_column_label(column_id),
            "status": _v2_domain_statuses(column_id, preview_map[column_id]).get(section_name, ("Pending", ""))[0],
            "detail": _v2_domain_statuses(column_id, preview_map[column_id]).get(section_name, ("Pending", ""))[1],
        }
        for column_id in column_ids
    ]
    render_vde_workbook_table(
        pd.DataFrame(status_rows),
        title=f"{section_name} status summary",
        table_id=f"{section_name.lower().replace(' ', '-')}-status-summary",
    )


def _v2_render_preview_save(defaults_df_getter) -> None:
    metadata = _v2_metadata_effective()
    metadata_rows = [
        {"field": "line_source", "value": str(metadata.get("line_source") or "-")},
        {"field": "selected_baseline_vde_id", "value": str(metadata.get("selected_baseline_vde_id") or "-")},
        {"field": "vehicle", "value": " | ".join(part for part in [str(metadata.get("make") or ""), str(metadata.get("model") or ""), str(metadata.get("model_year") or ""), str(metadata.get("legislation") or "")] if part) or "-"},
        {"field": "cycle", "value": str(metadata.get("cycle") or "-")},
        {"field": "description", "value": str(metadata.get("description") or "-")},
    ]
    render_vde_workbook_table(
        pd.DataFrame(metadata_rows),
        title="Metadata effective",
        table_id="preview-metadata-effective",
    )

    state = _v2_state()
    column_ids = _v2_column_ids(state)
    column_labels = _v2_workbook_column_names()
    previews = {column_id: _v2_cached_preview(column_id) for column_id in column_ids}
    preview_specs = []
    preview_specs.extend(
        [
            {"id": "walk_from", "label": "Walk From", "kind": "readonly", "notes": "-"},
            {"id": "proposal_direct", "label": "Proposal Direct", "kind": "readonly", "notes": "-"},
            {"id": "proposal_effective", "label": "Proposal Effective", "kind": "readonly", "notes": "-"},
            {"id": "base_mass", "label": "Base / curb mass [kg]", "kind": "readonly", "notes": "-"},
            {"id": "epa_twc", "label": "EPA ETW / TWC [kg]", "kind": "readonly", "notes": "-"},
            {"id": "test_mass", "label": "Resolved VDE test mass [kg]", "kind": "readonly", "notes": "-"},
            {"id": "test_mass_basis", "label": "Test mass basis", "kind": "readonly", "notes": "-"},
            {"id": "gvwr", "label": "GVWR [kg]", "kind": "readonly", "notes": "-"},
            {"id": "gcwr", "label": "GCWR [kg]", "kind": "readonly", "notes": "-"},
            {"id": "trailer_mass", "label": "Trailer mass [kg]", "kind": "readonly", "notes": "-"},
            {"id": "vehicle_mass_at_gcwr", "label": "Vehicle mass at GCWR [kg]", "kind": "readonly", "notes": "-"},
            {"id": "trailer_roadload_status", "label": "Trailer roadload status", "kind": "readonly", "notes": "-"},
            {"id": "mass_rule_status", "label": "Mass rule status", "kind": "readonly", "notes": "-"},
            {"id": "mass_rule_notes", "label": "Mass rule notes", "kind": "readonly", "notes": "-"},
            {"id": "abc_total", "label": "ABC_TOTAL", "kind": "readonly", "notes": "-"},
            {"id": "transmission_losses", "label": "Transmission Losses", "kind": "readonly", "notes": "-"},
            {"id": "abc_net", "label": "ABC_NET", "kind": "readonly", "notes": "-"},
            {"id": "vde_total", "label": "VDE_TOTAL", "kind": "readonly", "notes": "-"},
            {"id": "vde_net", "label": "VDE_NET", "kind": "readonly", "notes": "-"},
            {"id": "delta_vs_baseline", "label": "Delta vs Baseline", "kind": "readonly", "notes": "-"},
            {"id": "mass_status", "label": "Mass & Aero Status", "kind": "readonly", "notes": "-"},
            {"id": "tire_status", "label": "Tire Status", "kind": "readonly", "notes": "-"},
            {"id": "transmission_status", "label": "Transmission Status", "kind": "readonly", "notes": "-"},
            {"id": "brake_status", "label": "Brake Status", "kind": "readonly", "notes": "-"},
            {"id": "axle_status", "label": "Axle & Hubs Status", "kind": "readonly", "notes": "-"},
            {"id": "parasitic_status", "label": "Parasitic Status", "kind": "readonly", "notes": "-"},
            {"id": "save_status", "label": "Save Status", "kind": "readonly", "notes": "-"},
        ]
    )
    def preview_value(field_id: str, column_id: str) -> str:
        statuses = _v2_domain_statuses(column_id, previews[column_id])
        preview = previews[column_id]
        if field_id == "walk_from":
            return "-" if column_id == "baseline" else _v2_column_label(str(_v2_effective_state(column_id).get("walk_from") or "baseline"), state)
        if field_id == "proposal_direct":
            return str(_v2_effective_state(column_id).get("proposal_direct") or "")
        if field_id == "proposal_effective":
            return str(_v2_effective_state(column_id).get("proposal_effective") or "")
        if field_id == "base_mass":
            return _v2_format_value(_v2_effective_state(column_id).get("curb_mass_kg"), "mass") or "-"
        if field_id == "epa_twc":
            return _v2_format_value(_v2_effective_state(column_id).get("inertia_class"), "mass") or "-"
        if field_id == "test_mass":
            return _v2_format_value(_v2_effective_state(column_id).get("effective_test_mass_kg"), "mass") or "-"
        if field_id == "test_mass_basis":
            return str(_v2_effective_state(column_id).get("vde_mass_basis") or "-")
        if field_id == "gvwr":
            return _v2_format_value(_v2_effective_state(column_id).get("GVWR_kg"), "mass") or "-"
        if field_id == "gcwr":
            return _v2_format_value(_v2_effective_state(column_id).get("GCWR_kg"), "mass") or "-"
        if field_id == "trailer_mass":
            return _v2_format_value(_v2_effective_state(column_id).get("trailer_weight_kg"), "mass") or "-"
        if field_id == "vehicle_mass_at_gcwr":
            return _v2_format_value(_v2_effective_state(column_id).get("vehicle_mass_at_gcwr"), "mass") or "-"
        if field_id == "trailer_roadload_status":
            return str(_v2_effective_state(column_id).get("trailer_roadload_status") or "-")
        if field_id == "mass_rule_status":
            return str(_v2_effective_state(column_id).get("mass_rule_status") or "-")
        if field_id == "mass_rule_notes":
            return str(_v2_effective_state(column_id).get("mass_rule_notes") or "-")
        if field_id == "abc_total":
            return _compact_abc(dict(preview.get("abc_total") or {}))
        if field_id == "transmission_losses":
            return _compact_abc(dict((preview.get("transmission_losses") or {}).get("abc") or {})) if (preview.get("transmission_losses") or {}).get("abc") else "Unavailable"
        if field_id == "abc_net":
            return _compact_abc(dict(preview.get("abc_net") or {})) if preview.get("abc_net") else "Unavailable"
        if field_id == "vde_total":
            return format_quantity(dict(preview.get("vde_total") or {}).get("mj_per_km"), "energy_per_distance", include_unit=True, unavailable="-", format_str="%.3f")
        if field_id == "vde_net":
            return format_quantity(dict(preview.get("vde_net") or {}).get("mj_per_km"), "energy_per_distance", include_unit=True, unavailable="-", format_str="%.3f")
        if field_id == "delta_vs_baseline":
            baseline_val = dict(previews["baseline"].get("vde_net") or {}).get("mj_per_km")
            current_val = dict(preview.get("vde_net") or {}).get("mj_per_km")
            if baseline_val is None or current_val is None:
                return "-"
            return f"{(current_val - baseline_val):.3f} MJ/km"
        if field_id == "mass_status":
            return statuses["Mass & Aero"][0]
        if field_id == "tire_status":
            return statuses["Tire"][0]
        if field_id == "transmission_status":
            return statuses["Transmission"][0]
        if field_id == "brake_status":
            return statuses["Brake"][0]
        if field_id == "axle_status":
            return statuses["Axle & Hubs"][0]
        if field_id == "parasitic_status":
            return statuses["Parasitic Losses"][0]
        if field_id == "save_status":
            return "Ready" if preview.get("ok") else "Pending"
        return "-"
    preview_rows: list[dict] = []
    for spec in preview_specs:
        row = {"field": str(spec["label"])}
        for column_id in column_ids:
            text = preview_value(str(spec["id"]), column_id)
            row[column_labels[column_id]] = text
        row["notes"] = str(spec.get("notes") or "")
        preview_rows.append(row)
    render_vde_workbook_table(
        pd.DataFrame(preview_rows),
        title="Scenario Preview Matrix",
        table_id="scenario-preview-matrix",
    )

    target_options = list(column_ids)
    current_target = str(state.get("save_target") or _v2_last_column_id(state))
    if current_target not in target_options:
        current_target = _v2_last_column_id(state)
    save_cols = st.columns([1.2, 1.0, 1.0])
    chosen_target = save_cols[0].selectbox(
        "Save target",
        target_options,
        index=target_options.index(current_target),
        format_func=lambda value: _v2_column_label(value, state),
        key="v2_save_target_selector",
    )
    state["save_target"] = chosen_target
    _v2_set_state(state)
    save_mode = "update_existing" if chosen_target == "baseline" and str(_v2_effective_state("baseline").get("line_source") or "").startswith("Existing") else "insert_new"
    if save_cols[1].button("Compute effective snapshot", key="v2_compute_selected_column"):
        fresh = {column_id: _v2_preview(column_id) for column_id in column_ids}
        _v2_store_previews(fresh)
        st.success("Effective snapshot refreshed.")
    if save_cols[2].button("Save selected column", key="v2_save_selected_column"):
        preview = _v2_cached_preview(chosen_target)
        if not preview.get("ok"):
            st.error("Preview is not ready to save.")
        else:
            result = save_vde_setup_result(
                preview,
                save_mode,
                ctx=_v2_state_to_ctx(chosen_target),
                defaults_df=defaults_df_getter(),
            )
            st.success(f"Saved {_v2_column_label(chosen_target, state)} via `{result.get('action')}` on VDE id={result.get('vde_id')}.")


def _v2_build_audit_df(column_id: str) -> pd.DataFrame:
    spec_map = _v2_field_spec_map()
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    effective = _v2_effective_state(column_id)
    direct = dict((columns.get(column_id) or {}).get("direct") or {})
    source_id = str((columns.get(column_id) or {}).get("walk_from") or "baseline") if column_id != "baseline" else ""
    rows = []
    for field_id, spec in spec_map.items():
        rows.append(
            {
                "field": field_id,
                "raw_value": direct.get(field_id, "blank") if column_id != "baseline" or str((columns.get("baseline") or {}).get("line_source") or "") != "Existing VDE DB" else "loaded",
                "effective_value": effective.get(field_id) if effective.get(field_id) not in (None, "") else "-",
                "source_column": "Baseline" if column_id == "baseline" else _v2_column_label(source_id if field_id not in direct else column_id, state),
                "mode": "inherit" if field_id not in direct else "direct",
                "source": "vde_db" if column_id == "baseline" and str((columns.get("baseline") or {}).get("line_source") or "").startswith("Existing") else "manual",
                "status": "OK" if effective.get(field_id) not in (None, "") else "Missing",
                "notes": str(spec.get("notes") or ""),
            }
        )
    return pd.DataFrame(rows)


def _v2_render_technical_audit() -> None:
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    column_ids = _v2_column_ids(state)
    summary_rows = []
    for column_id in column_ids:
        effective = _v2_effective_state(column_id)
        direct = dict((columns.get(column_id) or {}).get("direct") or {})
        audit_df = _v2_build_audit_df(column_id)
        missing_count = int((audit_df["status"] == "Missing").sum()) if not audit_df.empty else 0
        summary_rows.append(
            {
                "scenario": _v2_column_label(column_id, state),
                "walk_from": "-" if column_id == "baseline" else _v2_column_label(str(effective.get("walk_from") or "baseline"), state),
                "proposal_effective": str(effective.get("proposal_effective") or ""),
                "direct_fields": len(direct),
                "missing_fields": missing_count,
                "source": "Existing VDE DB" if column_id == "baseline" and str(effective.get("line_source") or "").startswith("Existing") else "Workbook",
                "status": _v2_column_status(column_id)[0],
            }
        )
    render_vde_workbook_table(
        pd.DataFrame(summary_rows),
        title="Technical Audit summary",
        table_id="technical-audit-summary",
    )

    state = _v2_state()
    target_options = list(_v2_column_ids(state))
    current_target = str(state.get("audit_target") or _v2_last_column_id(state))
    if current_target not in target_options:
        current_target = _v2_last_column_id(state)
    state["audit_target"] = st.selectbox(
        "Audit target",
        target_options,
        index=target_options.index(current_target),
        format_func=lambda value: _v2_column_label(value, state),
        key="v2_audit_target_selector",
    )
    _v2_set_state(state)
    render_vde_workbook_table(
        _v2_build_audit_df(state["audit_target"]),
        title=f"Technical Audit - {_v2_column_label(state['audit_target'], state)}",
        table_id=f"technical-audit-{state['audit_target']}",
    )


VDE_WORKBOOK_V21_STATE_KEY = "vde_setup_workbook_v21"
VDE_WORKBOOK_V21_MENUS = [
    "Scenario Workbook",
    "Preview & Save",
]

# V2.1 workbook refactor map:
# - `render_vde_setup_workbook_v21` and the `_v21_render_*` helpers below remain the
#   active workbook surface for the new page.
# - `_resolve_scenario_workbook_state`, `_build_scenario_workbook_matrix_df`,
#   `_render_scenario_workbook_matrix`, and `render_vde_setup_spreadsheet_workbook`
#   remain legacy/reference workflow helpers for now.
# - During migration, top-level `state["proposals"]` stays alive for preview/save
#   compatibility while `state["columns"][column_id]["domains"]` becomes the explicit
#   workbook-facing schema for walked-column domain proposals.

V21_STATUS = ("Inherited", "OK", "Missing", "Review", "Invalid")

VDE_WORKBOOK_V21_COMPONENT_GROUPS = [
    {"key": "mass_aero", "label": "Mass & Aero", "domains": ["mass", "aero"]},
    {"key": "tire", "label": "Tire", "domains": ["tire"]},
    {"key": "transmission", "label": "Transmission", "domains": ["transmission"]},
    {"key": "brake", "label": "Brake", "domains": ["brake"]},
    {"key": "axle_hubs", "label": "Axle & Hubs", "domains": ["axle_hubs"]},
    {"key": "parasitic", "label": "Parasitic Losses", "domains": ["parasitic"]},
]

VDE_WORKBOOK_V21_COMPACT_SPECS = [
    {"id": "description", "label": "Description", "kind": "text", "notes": "Short scenario label"},
    {"id": "walk_from", "label": "Walk From", "kind": "text", "notes": "Effective inheritance source"},
    {"id": "proposal_direct", "label": "Proposal Direct", "kind": "readonly", "notes": "Direct proposal labels"},
    {"id": "proposal_effective", "label": "Proposal Effective", "kind": "readonly", "notes": "Accumulated proposal labels"},
]

VDE_WORKBOOK_V21_DOMAINS = {
    "mass": {
        "label": "Mass proposal",
        "section": "Mass",
        "short_label": "Mass",
        "types": ["INHERIT", "EPA_STATUS", "MASS_TWC_SHIFT", "PERFORMANCE_CURB_MASS", "WLTP_MASS_LINE", "GVWR", "GCWR", "CUSTOM_MASS"],
        "details": ["curb_mass_kg", "test_mass_kg", "payload_kg", "GVWR_kg", "GCWR_kg", "trailer_code", "trailer_weight_kg", "trailer_roadload_source", "trailer_A", "trailer_B", "trailer_C", "shift_steps", "target_side", "reference_source", "reference_mass_kg", "target_mass_kg", "line_type", "mass_kg", "optional_weight_kg", "laden_mass_kg", "wltp_mass_pair_id", "preset", "custom_delta_kg", "effective_test_mass_kg", "source", "notes"],
    },
    "aero": {
        "label": "Aero proposal",
        "section": "Aero",
        "short_label": "Aero",
        "types": ["INHERIT", "AERO_DELTA_CDA", "AERO_ABSOLUTE_CDA"],
        "details": ["delta_CdA", "new_CdA", "baseline_CdA", "Af_optional", "Cd_display", "source", "notes"],
    },
    "tire": {
        "label": "Tire proposal",
        "section": "Tire",
        "short_label": "Tire",
        "types": ["INHERIT", "TIRE_DB_LOOKUP", "TIRE_SMERF_RRC_CHANGE"],
        "details": ["baseline_tire_code", "new_tire_code", "tire_db_id", "tire_size", "psi_front", "psi_rear", "load_basis", "improvement_pct", "baseline_SMERF_optional", "delta_SMERF_optional", "baseline_RRC_optional", "delta_RRC_optional", "pressure_basis", "source", "notes"],
    },
    "transmission": {
        "label": "Transmission proposal",
        "section": "Transmission",
        "short_label": "Transmission",
        "types": ["INHERIT", "UPDATE_TRANS_DRAG_ABC", "TRANS_LOSS_PCT"],
        "details": ["change_mode", "baseline_component_reference_mode", "baseline_trans_A", "baseline_trans_B", "baseline_trans_C", "new_trans_A", "new_trans_B", "new_trans_C", "delta_A", "delta_B", "delta_C", "neutral_drag_source", "loss_pct", "percent_basis", "rule_version", "source", "notes"],
    },
    "brake": {
        "label": "Brake proposal",
        "section": "Brake",
        "short_label": "Brake",
        "types": ["INHERIT", "BRAKE_DRAG_CHANGE"],
        "details": ["method", "change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "brake_A", "brake_B", "brake_C", "delta_A", "delta_B", "delta_C", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m", "brake_drag_force_N", "brake_temp_condition", "brake_release_condition", "source", "notes"],
    },
    "axle_hubs": {
        "label": "Axle & Hubs proposal",
        "section": "Axle & Hubs",
        "short_label": "Axle & Hubs",
        "types": ["INHERIT", "AXLE_HUB_DRAG_CHANGE"],
        "details": ["change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "axle_hub_A", "axle_hub_B", "axle_hub_C", "delta_A", "delta_B", "delta_C", "source", "notes"],
    },
    "parasitic": {
        "label": "Parasitic Losses proposal",
        "section": "Parasitic Losses",
        "short_label": "Parasitic Losses",
        "types": ["INHERIT", "PARASITIC_LOSS_CHANGE"],
        "details": ["change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "parasitic_A", "parasitic_B", "parasitic_C", "delta_A", "delta_B", "delta_C", "source", "notes"],
    },
}

V21_DOMAIN_SCHEMAS = {
    domain_key: {
        "label": str(config.get("label") or domain_key),
        "section": str(config.get("section") or config.get("label") or domain_key),
        "proposal_types": [item for item in list(config.get("types") or []) if item != "INHERIT"],
        "detail_fields": list(config.get("details") or []),
    }
    for domain_key, config in VDE_WORKBOOK_V21_DOMAINS.items()
}

VDE_WORKBOOK_V21_DETAIL_FIELDS = {
    "mass": {
        "INHERIT": ["notes"],
        "EPA_STATUS": ["inertia_class", "curb_mass_kg", "source", "notes"],
        "MASS_TWC_SHIFT": ["shift_steps", "target_side", "reference_source", "reference_mass_kg", "target_mass_kg", "notes"],
        "EPA_PLUS_1_TWC": ["shift_steps", "target_side", "reference_source", "reference_mass_kg", "target_mass_kg", "notes"],
        "PERFORMANCE_CURB_MASS": ["preset", "custom_delta_kg", "curb_mass_kg", "effective_test_mass_kg", "notes"],
        "WLTP_MASS_LINE": ["line_type", "mass_kg", "optional_weight_kg", "laden_mass_kg", "wltp_mass_pair_id", "source", "notes"],
        "GVWR": ["GVWR_kg", "payload_kg", "notes"],
        "GCWR": ["GCWR_kg", "trailer_weight_kg", "trailer_code", "trailer_roadload_source", "trailer_A", "trailer_B", "trailer_C", "notes"],
        "CUSTOM_MASS": ["test_mass_kg", "notes"],
    },
    "aero": {
        "INHERIT": ["notes"],
        "AERO_ABSOLUTE_CDA": ["new_CdA", "baseline_CdA", "Af_optional", "Cd_display", "source", "notes"],
        "AERO_DELTA_CDA": ["delta_CdA", "source", "notes"],
    },
    "tire": {
        "INHERIT": ["tire_notes"],
        "TIRE_DB_LOOKUP": ["baseline_tire_code", "new_tire_code", "tire_db_id", "tire_size", "psi_front", "psi_rear", "load_basis", "improvement_pct", "source", "notes"],
        "TIRE_SMERF_RRC_CHANGE": ["baseline_SMERF_optional", "delta_SMERF_optional", "baseline_RRC_optional", "delta_RRC_optional", "pressure_basis", "load_basis", "source", "notes"],
    },
    "transmission": {
        "INHERIT": ["trans_notes"],
        "UPDATE_TRANS_DRAG_ABC": ["change_mode", "baseline_component_reference_mode", "baseline_trans_A", "baseline_trans_B", "baseline_trans_C", "new_trans_A", "new_trans_B", "new_trans_C", "delta_A", "delta_B", "delta_C", "neutral_drag_source", "source", "notes"],
        "TRANS_LOSS_PCT": ["loss_pct", "percent_basis", "rule_version", "source", "notes"],
    },
    "brake": {
        "INHERIT": ["brake_notes"],
        "BRAKE_DRAG_CHANGE": ["method", "change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "brake_A", "brake_B", "brake_C", "delta_A", "delta_B", "delta_C", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m", "brake_drag_force_N", "brake_temp_condition", "brake_release_condition", "source", "notes"],
    },
    "axle_hubs": {
        "INHERIT": ["axle_notes"],
        "AXLE_HUB_DRAG_CHANGE": ["change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "axle_hub_A", "axle_hub_B", "axle_hub_C", "delta_A", "delta_B", "delta_C", "source", "notes"],
    },
    "parasitic": {
        "INHERIT": ["parasitic_notes"],
        "PARASITIC_LOSS_CHANGE": ["change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "parasitic_A", "parasitic_B", "parasitic_C", "delta_A", "delta_B", "delta_C", "source", "notes"],
    },
}

VDE_WORKBOOK_V21_LEGACY_DOMAIN_MAP = {
    "mass_aero": "mass",
}

VDE_WORKBOOK_V21_TYPE_LABELS = {
    "EPA_STATUS": "EPA status mass",
    "MASS_TWC_SHIFT": "TWC shift",
    "EPA_PLUS_1_TWC": "TWC shift",
    "PERFORMANCE_CURB_MASS": "Performance curb mass",
    "PERF_CURB_100KG": "Performance curb mass",
    "PERF_CURB_300LB": "Performance curb mass",
    "WLTP_MASS_LINE": "WLTP mass line",
    "WLTP_TML": "WLTP mass line",
    "WLTP_TMH": "WLTP mass line",
    "GVWR": "GVWR loaded mass",
    "GCWR": "GCWR / trailer mass",
    "TRAILER_GCWR": "GCWR / trailer",
    "CUSTOM_MASS": "Custom test mass",
    "AERO_ABSOLUTE_CDA": "Absolute CdA",
    "AERO_DELTA_CDA": "Delta CdA",
    "AERO_CD_AREA": "Absolute CdA",
    "AERO_DELTA_ABC": "Delta CdA",
    "TIRE_DB_LOOKUP": "Tire DB lookup",
    "TIRE_ABSOLUTE_ABC": "SMERF / RRC change",
    "TIRE_DELTA_ABC": "SMERF / RRC change",
    "TIRE_DELTA_SMERF": "SMERF / RRC change",
    "TIRE_IMPROVEMENT_PCT": "Tire DB lookup",
    "TIRE_METADATA_ONLY": "SMERF / RRC change",
    "TIRE_SMERF_RRC_CHANGE": "SMERF / RRC change",
    "TRANS_LOSS_ABSOLUTE": "Update trans drag ABC",
    "TRANS_LOSS_DELTA_ABC": "Update trans drag ABC",
    "TRANS_LOSS_NOT_AVAILABLE": "Update trans drag ABC",
    "TRANS_METADATA_ONLY": "Update trans drag ABC",
    "UPDATE_TRANS_DRAG_ABC": "Update trans drag ABC",
    "TRANS_LOSS_PCT": "Transmission loss %",
    "BRAKE_ABSOLUTE_ABC": "Brake drag change",
    "BRAKE_DELTA_ABC": "Brake drag change",
    "RESIDUAL_TORQUE_DELTA": "Brake drag change",
    "BRAKE_METADATA_ONLY": "Brake drag change",
    "BRAKE_NOT_USED": "Brake drag change",
    "BRAKE_DRAG_CHANGE": "Brake drag change",
    "AXLE_HUB_ABSOLUTE_ABC": "Axle / hub drag change",
    "AXLE_HUB_DELTA_ABC": "Axle / hub drag change",
    "AXLE_HUB_METADATA_ONLY": "Axle / hub drag change",
    "AXLE_HUB_NOT_USED": "Axle / hub drag change",
    "AXLE_HUB_DRAG_CHANGE": "Axle / hub drag change",
    "PARASITIC_ABSOLUTE_ABC": "Parasitic loss change",
    "PARASITIC_DELTA_ABC": "Parasitic loss change",
    "PARASITIC_METADATA_ONLY": "Parasitic loss change",
    "PARASITIC_NOT_USED": "Parasitic loss change",
    "PARASITIC_LOSS_CHANGE": "Parasitic loss change",
}

VDE_WORKBOOK_V21_MATRIX_SELECTIONS = {
    "transmission": [
        {"value": "UPDATE_TRANS_DRAG_ABC__DELTA_ABC", "proposal_type": "UPDATE_TRANS_DRAG_ABC", "label": "Update trans drag ABC - Delta ABC", "seed": {"change_mode": "Delta ABC", "baseline_update_requested": False}},
        {"value": "UPDATE_TRANS_DRAG_ABC__ABSOLUTE_ABC", "proposal_type": "UPDATE_TRANS_DRAG_ABC", "label": "Update trans drag ABC - Absolute ABC", "seed": {"change_mode": "Absolute ABC", "baseline_update_requested": False}},
        {"value": "TRANS_LOSS_PCT", "proposal_type": "TRANS_LOSS_PCT", "label": "Transmission loss %", "seed": {}},
    ],
    "brake": [
        {"value": "BRAKE_DRAG_CHANGE__DELTA_ABC", "proposal_type": "BRAKE_DRAG_CHANGE", "label": "Brake drag change - Delta ABC", "seed": {"method": "Brake ABC", "change_mode": "Delta ABC", "baseline_update_requested": False}},
        {"value": "BRAKE_DRAG_CHANGE__ABSOLUTE_ABC", "proposal_type": "BRAKE_DRAG_CHANGE", "label": "Brake drag change - Absolute ABC", "seed": {"method": "Brake ABC", "change_mode": "Absolute ABC", "baseline_update_requested": False}},
        {"value": "BRAKE_DRAG_CHANGE__RESIDUAL_TORQUE", "proposal_type": "BRAKE_DRAG_CHANGE", "label": "Brake drag change - Residual torque", "seed": {"method": "Residual torque", "baseline_update_requested": False}},
    ],
    "axle_hubs": [
        {"value": "AXLE_HUB_DRAG_CHANGE__DELTA_ABC", "proposal_type": "AXLE_HUB_DRAG_CHANGE", "label": "Axle / hub drag change - Delta ABC", "seed": {"change_mode": "Delta ABC", "baseline_update_requested": False}},
        {"value": "AXLE_HUB_DRAG_CHANGE__ABSOLUTE_ABC", "proposal_type": "AXLE_HUB_DRAG_CHANGE", "label": "Axle / hub drag change - Absolute ABC", "seed": {"change_mode": "Absolute ABC", "baseline_update_requested": False}},
    ],
    "parasitic": [
        {"value": "PARASITIC_LOSS_CHANGE__DELTA_ABC", "proposal_type": "PARASITIC_LOSS_CHANGE", "label": "Parasitic loss change - Delta ABC", "seed": {"change_mode": "Delta ABC", "baseline_update_requested": False}},
        {"value": "PARASITIC_LOSS_CHANGE__ABSOLUTE_ABC", "proposal_type": "PARASITIC_LOSS_CHANGE", "label": "Parasitic loss change - Absolute ABC", "seed": {"change_mode": "Absolute ABC", "baseline_update_requested": False}},
    ],
}


def _v21_proposals(state: dict | None = None) -> dict:
    owned_state = state is not None
    state = state or _v2_state()
    proposals = state.get("proposals")
    normalized: dict[str, dict] = {}
    changed = not isinstance(proposals, dict)
    next_seq = int(to_float(state.get("proposal_seq")) or 0)
    for column_id, raw_column_proposals in dict(proposals or {}).items():
        if not isinstance(raw_column_proposals, dict):
            changed = True
            continue
        column_proposals: dict[str, dict] = {}
        for domain_key, raw_proposal in raw_column_proposals.items():
            if not isinstance(raw_proposal, dict):
                changed = True
                continue
            proposal = dict(raw_proposal)
            proposal_type = str(proposal.get("proposal_type") or proposal.get("type") or "INHERIT").strip().upper() or "INHERIT"
            if proposal_type == "INHERIT":
                changed = True
                continue
            normalized_domain_key = str(domain_key)
            if normalized_domain_key == "mass_aero":
                normalized_domain_key = "aero" if proposal_type.startswith("AERO_") else "mass"
                changed = True
            normalized_domain_key = VDE_WORKBOOK_V21_LEGACY_DOMAIN_MAP.get(normalized_domain_key, normalized_domain_key)
            proposal_id = str(proposal.get("id") or "").strip()
            match = re.search(r"(\d+)$", proposal_id)
            if match:
                next_seq = max(next_seq, int(match.group(1)))
            else:
                next_seq += 1
                proposal_id = f"prop_{next_seq}"
                changed = True
            normalized_proposal = {
                "id": proposal_id,
                "domain": str(normalized_domain_key),
                "type": proposal_type,
                "proposal_type": proposal_type,
                "label": str(proposal.get("label") or "").strip(),
                "details": {key: value for key, value in dict(proposal.get("details") or {}).items() if value not in (None, "")},
                "status": str(proposal.get("status") or "Draft").strip() or "Draft",
            }
            if normalized_proposal != proposal:
                changed = True
            column_proposals[str(normalized_domain_key)] = normalized_proposal
        if column_proposals:
            normalized[str(column_id)] = column_proposals
    columns = dict(state.get("columns") or {})
    for column_id, raw_column in columns.items():
        if not isinstance(raw_column, dict):
            continue
        domains = dict(raw_column.get("domains") or {})
        if not domains:
            continue
        column_proposals = dict(normalized.get(str(column_id)) or {})
        for domain_key, raw_domain_state in domains.items():
            if not isinstance(raw_domain_state, dict):
                changed = True
                continue
            domain_state = dict(raw_domain_state)
            proposal_type = str(domain_state.get("proposal_type") or domain_state.get("type") or "INHERIT").strip().upper() or "INHERIT"
            mode = str(domain_state.get("mode") or ("direct" if proposal_type != "INHERIT" else "inherited")).strip().lower()
            if mode != "direct" or proposal_type == "INHERIT":
                continue
            normalized_domain_key = str(domain_key)
            if normalized_domain_key == "mass_aero":
                normalized_domain_key = "aero" if proposal_type.startswith("AERO_") else "mass"
                changed = True
            normalized_domain_key = VDE_WORKBOOK_V21_LEGACY_DOMAIN_MAP.get(normalized_domain_key, normalized_domain_key)
            proposal_id = str(domain_state.get("id") or "").strip()
            match = re.search(r"(\d+)$", proposal_id)
            if match:
                next_seq = max(next_seq, int(match.group(1)))
            else:
                next_seq += 1
                proposal_id = f"prop_{next_seq}"
                changed = True
            notes = domain_state.get("notes") or []
            if not isinstance(notes, list):
                notes = [str(notes)]
            normalized_proposal = {
                "id": proposal_id,
                "domain": str(normalized_domain_key),
                "type": proposal_type,
                "proposal_type": proposal_type,
                "label": str(domain_state.get("label") or "").strip(),
                "details": {key: value for key, value in dict(domain_state.get("details") or {}).items() if value not in (None, "")},
                "status": str(domain_state.get("status") or "Draft").strip() or "Draft",
                "notes": [str(item).strip() for item in notes if str(item).strip()],
            }
            if column_proposals.get(str(normalized_domain_key)) != normalized_proposal:
                changed = True
            column_proposals[str(normalized_domain_key)] = normalized_proposal
        if column_proposals:
            normalized[str(column_id)] = column_proposals
    if changed or state.get("proposal_seq") != next_seq or state.get("proposals") != normalized:
        state["proposals"] = normalized
        state["proposal_seq"] = next_seq
        if not owned_state:
            _v2_set_state(state)
    return normalized


def _v21_domain_note_list(proposal: dict | None) -> list[str]:
    proposal = dict(proposal or {})
    details = _v21_normalize_details(proposal.get("details") or {})
    notes: list[str] = []
    for candidate in (
        proposal.get("notes"),
        details.get("notes"),
        details.get("tire_notes"),
        details.get("trans_notes"),
        details.get("brake_notes"),
        details.get("axle_notes"),
        details.get("parasitic_notes"),
    ):
        text = str(candidate or "").strip()
        if text and text not in notes:
            notes.append(text)
    return notes


def _v21_domain_state_from_proposal(column_id: str, domain_key: str, proposal: dict | None, state: dict) -> dict:
    if not proposal:
        return {
            "mode": "inherited",
            "id": "",
            "domain": domain_key,
            "proposal_type": "INHERIT",
            "label": "",
            "details": {},
            "status": "Inherited",
            "notes": [],
        }
    proposal_type = str(proposal.get("proposal_type") or proposal.get("type") or "INHERIT").strip().upper() or "INHERIT"
    resolved_status = _v21_resolved_proposal_status(column_id, domain_key, proposal, state)
    if resolved_status not in V21_STATUS:
        resolved_status = "Invalid"
    return {
        "mode": "direct",
        "id": str(proposal.get("id") or "").strip(),
        "domain": domain_key,
        "proposal_type": proposal_type,
        "label": str(proposal.get("label") or "").strip(),
        "details": _v21_normalize_details(proposal.get("details") or {}),
        "status": resolved_status,
        "notes": _v21_domain_note_list(proposal),
    }


def _v21_domain_state(column_id: str, domain_key: str, state: dict | None = None) -> dict:
    state = state or _v2_state()
    columns = dict(state.get("columns") or {})
    column = dict(columns.get(column_id) or {})
    domains = dict(column.get("domains") or {})
    domain_state = dict(domains.get(domain_key) or {})
    if domain_state:
        default_mode = "inherited" if str(domain_state.get("proposal_type") or "INHERIT").strip().upper() == "INHERIT" else "direct"
        domain_state.setdefault("mode", default_mode)
        domain_state.setdefault("id", "")
        domain_state.setdefault("domain", domain_key)
        domain_state.setdefault("proposal_type", "INHERIT" if domain_state.get("mode") != "direct" else "")
        domain_state.setdefault("label", "")
        domain_state["details"] = _v21_normalize_details(domain_state.get("details") or {})
        status_value = str(domain_state.get("status") or ("Inherited" if domain_state.get("mode") != "direct" else "Draft")).strip() or "Inherited"
        domain_state["status"] = status_value if status_value in (*V21_STATUS, "Draft") else "Invalid"
        notes = domain_state.get("notes") or []
        if not isinstance(notes, list):
            notes = [str(notes)]
        domain_state["notes"] = [str(item).strip() for item in notes if str(item).strip()]
        return domain_state
    proposal = dict((_v21_proposals(state).get(column_id) or {}).get(domain_key) or {})
    if proposal:
        return _v21_domain_state_from_proposal(column_id, domain_key, proposal, state)
    return {
        "mode": "inherited",
        "id": "",
        "domain": domain_key,
        "proposal_type": "INHERIT",
        "label": "",
        "details": {},
        "status": "Inherited",
        "notes": [],
    }


def _v21_domain_state_as_proposal(column_id: str, domain_key: str, state: dict | None = None) -> dict:
    domain_state = _v21_domain_state(column_id, domain_key, state)
    if str(domain_state.get("mode") or "") != "direct":
        return {}
    proposal_type = str(domain_state.get("proposal_type") or "").strip().upper() or "INHERIT"
    return {
        "id": str(domain_state.get("id") or "").strip(),
        "domain": domain_key,
        "type": proposal_type,
        "proposal_type": proposal_type,
        "label": str(domain_state.get("label") or "").strip(),
        "details": _v21_normalize_details(domain_state.get("details") or {}),
        "status": str(domain_state.get("status") or "Draft").strip() or "Draft",
        "notes": list(domain_state.get("notes") or []),
    }


def _v21_sync_domain_state_for_column(state: dict, column_id: str, domain_key: str, proposal: dict | None) -> dict:
    columns = {str(key): dict(value or {}) for key, value in dict(state.get("columns") or {}).items()}
    column = dict(columns.get(column_id) or {})
    domains = dict(column.get("domains") or {})
    domains[domain_key] = _v21_domain_state_from_proposal(column_id, domain_key, proposal, state)
    column["domains"] = domains
    columns[column_id] = column
    state["columns"] = columns
    return state


def _v21_ensure_workbook_state(state: dict | None = None) -> dict:
    owned_state = state is not None
    state = state or _v2_state()
    scenarios = _v2_scenarios(state)
    columns = {str(key): dict(value or {}) for key, value in dict(state.get("columns") or {}).items()}
    proposals = _v21_proposals(state)
    changed = False

    for scenario in scenarios:
        column_id = str(scenario.get("key") or "")
        if not column_id:
            continue
        column = dict(columns.get(column_id) or {})
        expected_kind = "baseline" if column_id == "baseline" else "walked"
        expected_label = str(scenario.get("label") or ("Baseline" if expected_kind == "baseline" else column_id))
        if column.get("kind") != expected_kind:
            column["kind"] = expected_kind
            changed = True
        if str(column.get("label") or "") != expected_label:
            column["label"] = expected_label
            changed = True
        if expected_kind == "baseline":
            if column.get("walk_from", "__missing__") is not None:
                column["walk_from"] = None
                changed = True
            baseline_overrides = dict(column.get("baseline_overrides") or {})
            if column.get("baseline_overrides") != baseline_overrides:
                column["baseline_overrides"] = baseline_overrides
                changed = True
            printed_overrides = dict(column.get("printed_overrides") or {})
            if column.get("printed_overrides") != printed_overrides:
                column["printed_overrides"] = printed_overrides
                changed = True
            if column.get("domains") not in (None, {}):
                column["domains"] = {}
                changed = True
            else:
                column.setdefault("domains", {})
        else:
            allowed_sources = _v2_allowed_walk_from_ids(column_id, state)
            fallback_source = allowed_sources[-1] if allowed_sources else "baseline"
            walk_from = str(column.get("walk_from") or fallback_source or "baseline")
            if walk_from not in allowed_sources:
                walk_from = fallback_source or "baseline"
            if str(column.get("walk_from") or "") != walk_from:
                column["walk_from"] = walk_from
                changed = True
            domain_states = {
                domain_key: _v21_domain_state_from_proposal(
                    column_id,
                    domain_key,
                    dict((proposals.get(column_id) or {}).get(domain_key) or {}),
                    state,
                )
                for domain_key in VDE_WORKBOOK_V21_DOMAINS
            }
            if column.get("domains") != domain_states:
                column["domains"] = domain_states
                changed = True
        columns[column_id] = column

    normalized_columns = {column_id: columns.get(column_id, {}) for column_id in _v2_column_ids(state)}
    if state.get("columns") != normalized_columns:
        state["columns"] = normalized_columns
        changed = True
    if "baseline_override_enabled" not in state:
        state["baseline_override_enabled"] = False
        changed = True
    if changed and not owned_state:
        _v2_set_state(state)
    return state


V21_BASELINE_PRINTED_GLOBAL_SCOPE = "__global__"

VDE_WORKBOOK_V21_PPE_BASELINE_EDITABLE_FIELDS = {
    "mass_kg",
    "inertia_class",
    "test_mass_kg",
    "payload_kg",
    "options_kg",
    "weight_dist_fr_pct",
    "gvwr_kg",
    "gcwr_kg",
    "trailer_mass_kg",
    "trailer_code",
    "trailer_A",
    "trailer_B",
    "trailer_C",
    "baseline_CdA",
    "Cd",
    "frontal_area_m2",
    "baseline_tire_code",
    "tire_db_id",
    "tire_size",
    "front_pressure_psi",
    "rear_pressure_psi",
    "hot_front_pressure_psi",
    "hot_rear_pressure_psi",
    "baseline_RRC_optional",
    "baseline_SMERF_optional",
    "rrc_N_per_kN",
    "baseline_trans_A",
    "baseline_trans_B",
    "baseline_trans_C",
    "baseline_component_A",
    "baseline_component_B",
    "baseline_component_C",
    "residual_torque_front_Nm",
    "residual_torque_rear_Nm",
    "residual_torque_total_Nm",
    "wheel_radius_m",
    "axle_hub_delta_A",
    "axle_hub_delta_B",
    "axle_hub_delta_C",
    "parasitic_delta_A",
    "parasitic_delta_B",
    "parasitic_delta_C",
    "mass_profile_gvwr_kg",
    "mass_profile_gcwr_kg",
    "mass_profile_trailer_mass_kg",
    "mass_profile_custom_input_kg",
}


def _v21_baseline_override_enabled(state: dict | None = None) -> bool:
    state = state or _v2_state()
    bucket = _v21_baseline_printed_override_bucket(state)
    has_overrides = any(dict(scope_bucket or {}) for scope_bucket in dict(bucket or {}).values())
    return bool(state.get("baseline_override_enabled")) or has_overrides


def _v21_baseline_printed_override_bucket(state: dict | None = None) -> dict:
    state = state or _v2_state()
    columns = dict(state.get("columns") or {})
    baseline = dict(columns.get("baseline") or {})
    return dict(baseline.get("printed_overrides") or {})


def _v21_baseline_printed_override_scope(domain_key: str | None, field_id: str) -> str:
    canonical = _v21_canonical_field_id(field_id)
    if canonical in {"baseline_component_A", "baseline_component_B", "baseline_component_C"}:
        return str(domain_key or V21_BASELINE_PRINTED_GLOBAL_SCOPE)
    return V21_BASELINE_PRINTED_GLOBAL_SCOPE


def _v21_baseline_printed_override_value(field_id: str, domain_key: str | None = None, state: dict | None = None):
    canonical = _v21_canonical_field_id(field_id)
    bucket = _v21_baseline_printed_override_bucket(state)
    scoped = dict(bucket.get(_v21_baseline_printed_override_scope(domain_key, canonical)) or {})
    if canonical in scoped:
        return scoped.get(canonical)
    global_bucket = dict(bucket.get(V21_BASELINE_PRINTED_GLOBAL_SCOPE) or {})
    return global_bucket.get(canonical)


def _v21_set_baseline_printed_override_values(domain_key: str | None, values: dict[str, object], state: dict | None = None) -> dict:
    state = deepcopy(state or _v2_state())
    columns = {str(key): dict(value or {}) for key, value in dict(state.get("columns") or {}).items()}
    baseline = dict(columns.get("baseline") or {})
    bucket = dict(baseline.get("printed_overrides") or {})
    per_scope_updates: dict[str, dict[str, object]] = {}
    for field_id, value in dict(values or {}).items():
        canonical = _v21_canonical_field_id(field_id)
        scope = _v21_baseline_printed_override_scope(domain_key, canonical)
        scope_bucket = dict(per_scope_updates.get(scope) or dict(bucket.get(scope) or {}))
        if value in (None, ""):
            scope_bucket.pop(canonical, None)
        else:
            scope_bucket[canonical] = value
        per_scope_updates[scope] = scope_bucket
    for scope, scope_bucket in per_scope_updates.items():
        if scope_bucket:
            bucket[scope] = scope_bucket
        else:
            bucket.pop(scope, None)
    baseline["printed_overrides"] = bucket
    columns["baseline"] = baseline
    state["columns"] = columns
    preview_cache = dict(state.get("preview_cache") or {})
    preview_cache.clear()
    state["preview_cache"] = preview_cache
    return state


def _v21_mass_twc_shift_label(details: dict) -> str:
    details = dict(details or {})
    steps_raw = str(details.get("twc_shift_steps") or "+1").strip() or "+1"
    if re.fullmatch(r"[+-]?\d+", steps_raw):
        value = int(steps_raw)
        if value > 0:
            steps_text = f"+{value}"
        else:
            steps_text = str(value)
    else:
        steps_text = steps_raw
    side = str(details.get("twc_target_side") or "").strip().title()
    if side in {"", "Nominal"}:
        return f"{steps_text} TWC"
    return f"{steps_text} TWC {side}"


def _v21_proposal_type_label(proposal_type: str) -> str:
    proposal_type = str(proposal_type or "").strip().upper()
    return str(VDE_WORKBOOK_V21_TYPE_LABELS.get(proposal_type) or proposal_type)


def _v21_effective_proposal_label(proposal: dict | None) -> str:
    proposal = dict(proposal or {})
    label = str(proposal.get("label") or "").strip()
    if label:
        return label
    proposal_type = str(proposal.get("proposal_type") or proposal.get("type") or "").strip().upper()
    details = dict(proposal.get("details") or {})
    if proposal_type in {"MASS_TWC_SHIFT", "EPA_PLUS_1_TWC"}:
        if proposal_type == "EPA_PLUS_1_TWC":
            details.setdefault("twc_shift_steps", "+1")
            details.setdefault("twc_target_side", "Nominal")
        return _v21_mass_twc_shift_label(details)
    return _v21_proposal_type_label(proposal_type)


def _v21_walk_from_label(column_id: str, state: dict | None = None) -> str:
    state = state or _v2_state()
    if column_id == "baseline":
        return "Baseline"
    columns = dict(state.get("columns") or {})
    source_id = str(dict(columns.get(column_id) or {}).get("walk_from") or "baseline")
    if source_id not in set(_v2_column_ids(state)):
        source_id = "baseline"
    return _v21_display_column_label(source_id, state)


def _v21_proposal_badge_text(proposal: dict | None) -> str:
    proposal = dict(proposal or {})
    proposal_id = str(proposal.get("id") or "").strip()
    match = re.search(r"(\d+)$", proposal_id)
    prefix = f"Prop #{match.group(1)}" if match else "Prop"
    label = _v21_effective_proposal_label(proposal)
    separator = "\u00b7"
    return f"{prefix} {separator} {label}".strip(f" {separator}")


def _v21_domain_status(proposal: dict | None) -> str:
    if not proposal:
        return "inherit"
    proposal_type = str(proposal.get("proposal_type") or "INHERIT").strip().upper()
    details = _v21_normalize_details(proposal.get("details") or {})
    if proposal_type == "INHERIT":
        return "inherit"
    if proposal_type in {"MASS_TWC_SHIFT", "EPA_PLUS_1_TWC"}:
        if str(details.get("target_mass_kg") or details.get("custom_target_mass_kg") or details.get("target_test_mass_kg") or "").strip():
            return "Draft"
        if str(details.get("shift_steps") or details.get("twc_shift_steps") or "").strip():
            return "Draft"
        return "Missing"
    if proposal_type == "PERFORMANCE_CURB_MASS":
        if not str(details.get("preset") or "").strip():
            return "Missing"
        if str(details.get("preset") or "").strip().lower() == "custom" and not str(details.get("custom_delta_kg") or "").strip():
            return "Missing"
    if proposal_type == "WLTP_MASS_LINE" and (
        not str(details.get("line_type") or "").strip()
        or (
            str(details.get("line_type") or "").strip().upper() == "TMH"
            and not str(details.get("test_mass_high_kg") or "").strip()
        )
        or (
            str(details.get("line_type") or "").strip().upper() != "TMH"
            and not str(details.get("test_mass_low_kg") or "").strip()
        )
    ):
        return "Missing"
    if proposal_type in {"GVWR"} and not str(details.get("gvwr_kg") or "").strip():
        return "Missing"
    if proposal_type in {"GCWR", "TRAILER_GCWR"} and (not str(details.get("gcwr_kg") or "").strip() or not str(details.get("trailer_mass_kg") or "").strip()):
        return "Missing"
    if proposal_type in {"CUSTOM_MASS"} and not str(details.get("test_mass_kg") or "").strip():
        return "Missing"
    if proposal_type == "AERO_DELTA_CDA" and not str(details.get("delta_CdA") or "").strip():
        return "Missing"
    if proposal_type == "AERO_ABSOLUTE_CDA" and not str(details.get("new_CdA") or "").strip():
        return "Missing"
    if proposal_type == "TIRE_DB_LOOKUP" and not str(details.get("new_tire_code") or "").strip():
        return "Missing"
    if proposal_type == "TIRE_SMERF_RRC_CHANGE" and not any(str(details.get(field) or "").strip() for field in ("delta_SMERF_optional", "delta_RRC_optional")):
        return "Missing"
    if proposal_type == "UPDATE_TRANS_DRAG_ABC" and not any(str(details.get(field) or "").strip() for field in ("new_trans_A", "new_trans_B", "new_trans_C", "delta_A", "delta_B", "delta_C")):
        return "Missing"
    if proposal_type == "TRANS_LOSS_PCT" and not str(details.get("loss_pct") or "").strip():
        return "Missing"
    if proposal_type == "BRAKE_DRAG_CHANGE" and not any(str(details.get(field) or "").strip() for field in ("brake_A", "brake_B", "brake_C", "delta_A", "delta_B", "delta_C", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "brake_drag_force_N")):
        return "Missing"
    if proposal_type == "AXLE_HUB_DRAG_CHANGE" and not any(str(details.get(field) or "").strip() for field in ("axle_hub_A", "axle_hub_B", "axle_hub_C", "delta_A", "delta_B", "delta_C")):
        return "Missing"
    if proposal_type == "PARASITIC_LOSS_CHANGE" and not any(str(details.get(field) or "").strip() for field in ("parasitic_A", "parasitic_B", "parasitic_C", "delta_A", "delta_B", "delta_C")):
        return "Missing"
    return "Review" if not str(proposal.get("label") or "").strip() else "Draft"


def _v21_summary_text(column_id: str, domain_key: str, state: dict | None = None) -> str:
    if column_id == "baseline":
        return "baseline"
    proposal = _v21_domain_state_as_proposal(column_id, domain_key, state)
    if not proposal:
        return f"Inherited from {_v21_walk_from_label(column_id, state)}"
    return _v21_proposal_badge_text(proposal)


def _v21_direct_fields_from_proposal(domain_key: str, proposal: dict, column_id: str | None = None, state: dict | None = None) -> dict:
    proposal_type = str(proposal.get("proposal_type") or "INHERIT").strip().upper()
    details = {
        key: value
        for key, value in _v21_normalize_details(proposal.get("details") or {}).items()
        if value not in (None, "", "inherit")
    }
    details = _v21_sync_component_reference_details(details)
    if proposal_type == "INHERIT":
        return {}
    out = dict(details)
    if domain_key == "mass":
        if details.get("mass_kg") not in (None, ""):
            out["mass_kg"] = details.get("mass_kg")
        if details.get("gvwr_kg") not in (None, ""):
            out["GVWR_kg"] = details.get("gvwr_kg")
        if details.get("gcwr_kg") not in (None, ""):
            out["GCWR_kg"] = details.get("gcwr_kg")
        if details.get("trailer_mass_kg") not in (None, ""):
            out["trailer_weight_kg"] = details.get("trailer_mass_kg")
        if proposal_type in {"MASS_TWC_SHIFT", "EPA_PLUS_1_TWC"}:
            out["mass_intention"] = "EPA_PLUS_1_TWC"
            target_mass = details.get("target_mass_kg") or details.get("custom_target_mass_kg") or details.get("target_test_mass_kg")
            if target_mass not in (None, ""):
                out["prep_inertia_class"] = target_mass
        elif proposal_type == "PERFORMANCE_CURB_MASS":
            curb_mass = to_float(details.get("mass_kg"))
            preset = str(details.get("preset") or "").strip().lower()
            delta_kg = to_float(details.get("custom_delta_kg"))
            if preset == "+300 lb":
                out["mass_intention"] = "PERF_CURB_300LB"
            elif preset == "custom" and delta_kg is not None and curb_mass is not None:
                out["mass_intention"] = "CUSTOM"
                out["test_mass_kg"] = curb_mass + delta_kg
            else:
                out["mass_intention"] = "PERF_CURB_100KG"
        elif proposal_type == "WLTP_MASS_LINE":
            line_type = str(details.get("line_type") or "").strip().upper()
            mass_value = details.get("test_mass_high_kg") if line_type == "TMH" else details.get("test_mass_low_kg")
            if line_type == "TMH":
                out["mass_intention"] = "WLTP_TMH"
                out["TMH_kg"] = mass_value
            else:
                out["mass_intention"] = "WLTP_TML"
                out["TML_kg"] = mass_value
            if details.get("optional_weight_kg") not in (None, ""):
                out["payload_kg"] = details.get("optional_weight_kg")
        elif proposal_type == "CUSTOM_MASS":
            out["mass_intention"] = "CUSTOM"
        else:
            out["mass_intention"] = {
                "TRAILER_GCWR": "GCWR",
            }.get(proposal_type, proposal_type)
    elif domain_key == "aero":
        if proposal_type in {"AERO_ABSOLUTE_CDA", "AERO_CD_AREA"}:
            out["aero_mode"] = "Absolute"
            if details.get("new_CdA") not in (None, ""):
                out["CdA"] = details.get("new_CdA")
        elif proposal_type in {"AERO_DELTA_CDA", "AERO_DELTA_ABC"}:
            out["aero_mode"] = "Delta"
    elif domain_key == "mass_aero":
        out["mass_intention"] = {
            "CUSTOM_MASS": "CUSTOM",
            "TRAILER_GCWR": "GCWR",
        }.get(proposal_type, proposal_type)
        if proposal_type == "AERO_ABSOLUTE":
            out["aero_mode"] = "Absolute"
        elif proposal_type == "AERO_DELTA":
            out["aero_mode"] = "Delta"
    elif domain_key == "tire":
        if details.get("front_pressure_psi") not in (None, ""):
            out["front_pressure_psi"] = details.get("front_pressure_psi")
        if details.get("rear_pressure_psi") not in (None, ""):
            out["rear_pressure_psi"] = details.get("rear_pressure_psi")
        if details.get("tire_improvement_pct") not in (None, ""):
            out["tire_improvement_pct"] = details.get("tire_improvement_pct")
        if details.get("tire_load_mass_basis") not in (None, ""):
            out["load_basis"] = details.get("tire_load_mass_basis")
        out["tire_mode"] = "TIRE_DB_LOOKUP" if proposal_type == "TIRE_DB_LOOKUP" else "TIRE_DELTA_SMERF"
    elif domain_key == "transmission":
        if proposal_type == "UPDATE_TRANS_DRAG_ABC":
            out["trans_loss_mode"] = "TRANS_LOSS_DELTA_ABC"
            if str(details.get("change_mode") or "").strip() == "Absolute ABC" and column_id:
                computed = _v21_component_delta_from_absolute("transmission", column_id, details, state)
                if computed:
                    details.update(computed)
                    out.update(computed)
            delta_map = {
                "delta_A": "trans_delta_A_loss",
                "delta_B": "trans_delta_B_loss",
                "delta_C": "trans_delta_C_loss",
            }
            for source_field, target_field in delta_map.items():
                if details.get(source_field) not in (None, ""):
                    out[target_field] = details.get(source_field)
        else:
            out["trans_loss_mode"] = proposal_type
    elif domain_key == "brake":
        if str(details.get("method") or "").strip().lower() == "residual torque":
            out["brake_mode"] = "RESIDUAL_TORQUE_DELTA"
        else:
            out["brake_mode"] = "BRAKE_DELTA_ABC"
            if str(details.get("change_mode") or "").strip() == "Absolute ABC" and column_id:
                computed = _v21_component_delta_from_absolute("brake", column_id, details, state)
                if computed:
                    details.update(computed)
                    out.update(computed)
            delta_map = {
                "delta_A": "brake_delta_A",
                "delta_B": "brake_delta_B",
                "delta_C": "brake_delta_C",
            }
            for source_field, target_field in delta_map.items():
                if details.get(source_field) not in (None, ""):
                    out[target_field] = details.get(source_field)
    elif domain_key == "axle_hubs":
        out["axle_hub_mode"] = "AXLE_HUB_DELTA_ABC"
        if str(details.get("change_mode") or "").strip() == "Absolute ABC" and column_id:
            computed = _v21_component_delta_from_absolute("axle_hubs", column_id, details, state)
            if computed:
                details.update(computed)
                out.update(computed)
        delta_map = {
            "delta_A": "axle_delta_A",
            "delta_B": "axle_delta_B",
            "delta_C": "axle_delta_C",
        }
        for source_field, target_field in delta_map.items():
            if details.get(source_field) not in (None, ""):
                out[target_field] = details.get(source_field)
    elif domain_key == "parasitic":
        out["parasitic_mode"] = "PARASITIC_DELTA_ABC"
        if str(details.get("change_mode") or "").strip() == "Absolute ABC" and column_id:
            computed = _v21_component_delta_from_absolute("parasitic", column_id, details, state)
            if computed:
                details.update(computed)
                out.update(computed)
        delta_map = {
            "delta_A": "parasitic_delta_A_row",
            "delta_B": "parasitic_delta_B_row",
            "delta_C": "parasitic_delta_C_row",
        }
        for source_field, target_field in delta_map.items():
            if details.get(source_field) not in (None, ""):
                out[target_field] = details.get(source_field)
    return out


def _v21_column_proposal_label(column_id: str, state: dict | None = None) -> str:
    labels: list[str] = []
    proposals = [
        _v21_domain_state_as_proposal(column_id, domain_key, state)
        for domain_key in VDE_WORKBOOK_V21_DOMAINS
    ]
    proposals = [proposal for proposal in proposals if proposal]
    def _proposal_order(item: dict) -> int:
        match = re.search(r"(\d+)$", str((item or {}).get("id") or ""))
        return int(match.group(1)) if match else 0
    proposals.sort(key=_proposal_order)
    for proposal in proposals:
        proposal = dict(proposal or {})
        proposal_type = str(proposal.get("proposal_type") or "INHERIT").strip().upper()
        if proposal_type == "INHERIT":
            continue
        label = _v21_effective_proposal_label(proposal)
        labels.append(label)
    return " + ".join(labels)


def _v21_apply_proposals_to_effective(effective: dict, column_id: str, state: dict | None = None) -> None:
    proposals = {
        domain_key: _v21_domain_state_as_proposal(column_id, domain_key, state)
        for domain_key in VDE_WORKBOOK_V21_DOMAINS
    }
    proposals = {domain_key: proposal for domain_key, proposal in proposals.items() if proposal}
    for domain_key, proposal in proposals.items():
        proposal = dict(proposal or {})
        proposal_type = str(proposal.get("proposal_type") or proposal.get("type") or "").strip().upper()
        direct_fields = _v21_direct_fields_from_proposal(str(domain_key), proposal, column_id, state)
        if str(domain_key) == "aero":
            details = _v21_normalize_details(proposal.get("details") or {})
            inherited_cda = to_float(effective.get("CdA"))
            inherited_a = to_float(effective.get("ABC_TOTAL_A"))
            inherited_b = to_float(effective.get("ABC_TOTAL_B"))
            inherited_c = to_float(effective.get("ABC_TOTAL_C"))
            _, _, has_reference = _v21_aero_reference_value(column_id, details, state)
            if proposal_type == "AERO_CD_AREA":
                cd = to_float(details.get("Cd"))
                area = to_float(details.get("frontal_area_m2"))
                if cd is not None and area is not None:
                    direct_fields["CdA"] = cd * area
            elif proposal_type == "AERO_ABSOLUTE_CDA":
                if not has_reference:
                    direct_fields.pop("CdA", None)
                elif details.get("CdA") in (None, ""):
                    cd = to_float(details.get("Cd"))
                    area = to_float(details.get("frontal_area_m2"))
                    if cd is not None and area is not None:
                        direct_fields["CdA"] = cd * area
            elif proposal_type == "AERO_DELTA_CDA":
                delta_cda = to_float(details.get("delta_CdA"))
                if inherited_cda is not None and delta_cda is not None:
                    direct_fields["CdA"] = inherited_cda + delta_cda
            elif proposal_type == "AERO_DELTA_ABC":
                delta_a = to_float(details.get("delta_A"))
                delta_b = to_float(details.get("delta_B"))
                delta_c = to_float(details.get("delta_C"))
                if inherited_a is not None and delta_a is not None:
                    direct_fields["ABC_TOTAL_A"] = inherited_a + delta_a
                if inherited_b is not None and delta_b is not None:
                    direct_fields["ABC_TOTAL_B"] = inherited_b + delta_b
                if inherited_c is not None and delta_c is not None:
                    direct_fields["ABC_TOTAL_C"] = inherited_c + delta_c
        for field_id, value in direct_fields.items():
            parsed = _v2_parse_value(value, str((_v2_field_spec_map().get(field_id) or {}).get("kind") or "text"))
            if parsed not in (None, "", "inherit"):
                effective[field_id] = parsed
    proposal_label = _v21_column_proposal_label(column_id, state)
    if column_id != "baseline" and proposal_label:
        direct_label = str(effective.get("proposal_direct") or "").strip()
        if not direct_label:
            effective["proposal_direct"] = proposal_label
        effective_label = str(effective.get("proposal_effective") or "").strip()
        if proposal_label not in effective_label.split(" + "):
            effective["proposal_effective"] = " + ".join(part for part in [effective_label, proposal_label] if part)


def _v21_proposal_summary_specs() -> list[dict]:
    return [
        {"id": f"proposal_{domain_key}", "label": str(config["label"]), "kind": "readonly", "notes": "Direct proposal summary"}
        for domain_key, config in VDE_WORKBOOK_V21_DOMAINS.items()
    ]


def _v21_status_specs() -> list[dict]:
    return [
        {"id": "save_status", "label": "Save Status", "kind": "readonly", "notes": "Preview/save readiness"},
        {"id": "review_status", "label": "Review Status", "kind": "readonly", "notes": "Request completeness"},
    ]


def _v21_is_domain_summary_row(spec_id: str) -> bool:
    text = str(spec_id or "").strip()
    if not text.startswith("proposal_"):
        return False
    return text.removeprefix("proposal_") in VDE_WORKBOOK_V21_DOMAINS


def _v21_review_status(column_id: str, state: dict | None = None) -> str:
    if column_id == "baseline":
        return _v2_column_status(column_id)[0]
    state = state or _v2_state()
    resolved = _v21_resolved_workbook_model(state)
    return str(dict(resolved.get("columns") or {}).get(column_id, {}).get("review_status") or "Inherited")


def _v21_component_group_specs() -> list[dict]:
    return [dict(item) for item in VDE_WORKBOOK_V21_COMPONENT_GROUPS]


def _v21_component_group_domains(group_key: str) -> list[str]:
    for item in _v21_component_group_specs():
        if str(item.get("key") or "") == str(group_key or ""):
            return list(item.get("domains") or [])
    return ["mass", "aero"] if str(group_key or "") == "mass_aero" else []


def _v21_component_group_status(group_key: str, state: dict | None = None) -> tuple[str, str, int]:
    state = state or _v2_state()
    resolved = _v21_resolved_workbook_model(state)
    walked_columns = [column_id for column_id in _v2_column_ids(state) if column_id != "baseline"]
    domains = _v21_component_group_domains(group_key)
    statuses: list[str] = []
    proposal_count = 0
    for domain_key in domains:
        for column_id in walked_columns:
            column_state = dict(dict(resolved.get("columns") or {}).get(column_id) or {})
            direct_domain = dict(dict(column_state.get("direct_domains") or {}).get(domain_key) or {})
            effective_domain = dict(dict(column_state.get("effective_domains") or {}).get(domain_key) or {})
            if not direct_domain:
                continue
            proposal_count += 1
            statuses.append(str(effective_domain.get("status") or direct_domain.get("status") or "Inherited"))
    if not statuses:
        return "Inherited", "No direct proposal", 0
    return rollup_v21_statuses_model(statuses, default="Inherited"), f"{proposal_count} direct proposal(s)", proposal_count


def _v21_default_component_group(state: dict | None = None) -> str:
    state = state or _v2_state()
    current = str(st.session_state.get("v21_component_group") or state.get("v21_component_group") or "").strip()
    valid_keys = [str(item["key"]) for item in _v21_component_group_specs()]
    if current in valid_keys:
        return current
    for item in _v21_component_group_specs():
        _, _, proposal_count = _v21_component_group_status(str(item["key"]), state)
        if proposal_count > 0:
            return str(item["key"])
    for item in _v21_component_group_specs():
        status_value, _, _ = _v21_component_group_status(str(item["key"]), state)
        if status_value in {"Invalid", "Missing", "Review", "Draft"}:
            return str(item["key"])
    return "mass_aero"


def _v21_set_component_group(group_key: str) -> None:
    state = _v2_state()
    state["v21_component_group"] = str(group_key or "mass_aero")
    _v2_set_state(state)
    st.session_state["v21_component_group"] = str(group_key or "mass_aero")


def _v21_default_detail_domain(allowed_domains: list[str] | None = None, state: dict | None = None) -> str:
    state = state or _v2_state()
    domain_order = [domain for domain in (allowed_domains or list(VDE_WORKBOOK_V21_DOMAINS)) if domain in VDE_WORKBOOK_V21_DOMAINS]
    current = str(st.session_state.get("v21_detail_domain") or "").strip()
    if current in domain_order:
        return current
    walked_columns = [column_id for column_id in _v2_column_ids(state) if column_id != "baseline"]
    for domain_key in domain_order:
        for column_id in walked_columns:
            if _v21_get_direct_proposal(column_id, domain_key, state):
                return domain_key
    for domain_key in domain_order:
        for column_id in walked_columns:
            proposal = _v21_get_direct_proposal(column_id, domain_key, state)
            if proposal and _v21_resolved_proposal_status(column_id, domain_key, proposal, state) in {"Invalid", "Missing", "Review", "Draft"}:
                return domain_key
    return domain_order[0] if domain_order else "mass"


def _v21_input_display_value(column_id: str, spec: dict) -> str:
    field_id = str(spec["id"])
    if _v21_is_domain_summary_row(field_id):
        return _v21_summary_text(column_id, field_id.removeprefix("proposal_"))
    if field_id == "proposal_direct":
        if column_id == "baseline":
            return "-"
        return _v21_column_proposal_label(column_id) or str(_v2_effective_state(column_id).get("proposal_direct") or "inherit")
    if field_id == "proposal_effective":
        return str(_v2_effective_state(column_id).get("proposal_effective") or "-")
    if field_id == "save_status":
        preview = _v2_cached_preview(column_id)
        return "Ready" if preview.get("ok") else "Pending"
    if field_id == "review_status":
        return _v21_review_status(column_id)
    return _v2_field_oriented_display_value(column_id, spec)


def _v21_proposal_type_for_cell(column_id: str, domain_key: str, state: dict | None = None) -> str:
    if column_id == "baseline":
        return "baseline"
    proposal = _v21_domain_state_as_proposal(column_id, domain_key, state)
    return str(proposal.get("proposal_type") or "inherit").strip() or "inherit"


def _v21_proposal_select_widget_key(column_id: str, domain_key: str) -> str:
    return f"v21_proposal_select__{column_id}__{domain_key}"


def _v21_matrix_selection_config(domain_key: str, selection_value: str) -> dict:
    selection_value = str(selection_value or "").strip()
    for config in VDE_WORKBOOK_V21_MATRIX_SELECTIONS.get(domain_key, []):
        if str(config.get("value") or "").strip() == selection_value:
            return dict(config)
    return {
        "value": selection_value,
        "proposal_type": selection_value,
        "label": _v21_proposal_type_label(selection_value),
        "seed": {},
    }


def _v21_matrix_selection_value_for_proposal(domain_key: str, proposal: dict | None) -> str:
    proposal = dict(proposal or {})
    proposal_type = str(proposal.get("proposal_type") or proposal.get("type") or "").strip().upper()
    details = _v21_normalize_details(proposal.get("details") or {})
    if domain_key == "transmission" and proposal_type == "UPDATE_TRANS_DRAG_ABC":
        return "UPDATE_TRANS_DRAG_ABC__ABSOLUTE_ABC" if str(details.get("change_mode") or "").strip() == "Absolute ABC" else "UPDATE_TRANS_DRAG_ABC__DELTA_ABC"
    if domain_key == "brake" and proposal_type == "BRAKE_DRAG_CHANGE":
        if str(details.get("method") or "").strip() == "Residual torque":
            return "BRAKE_DRAG_CHANGE__RESIDUAL_TORQUE"
        return "BRAKE_DRAG_CHANGE__ABSOLUTE_ABC" if str(details.get("change_mode") or "").strip() == "Absolute ABC" else "BRAKE_DRAG_CHANGE__DELTA_ABC"
    if domain_key == "axle_hubs" and proposal_type == "AXLE_HUB_DRAG_CHANGE":
        return "AXLE_HUB_DRAG_CHANGE__ABSOLUTE_ABC" if str(details.get("change_mode") or "").strip() == "Absolute ABC" else "AXLE_HUB_DRAG_CHANGE__DELTA_ABC"
    if domain_key == "parasitic" and proposal_type == "PARASITIC_LOSS_CHANGE":
        return "PARASITIC_LOSS_CHANGE__ABSOLUTE_ABC" if str(details.get("change_mode") or "").strip() == "Absolute ABC" else "PARASITIC_LOSS_CHANGE__DELTA_ABC"
    return proposal_type


def _v21_apply_proposal_summary_rows(edited_rows: list[dict], specs: list[dict]) -> None:
    state = _v2_state()
    proposals = dict(_v21_proposals(state))
    labels = _v21_request_column_labels(state)
    label_to_column = {label: key for key, label in labels.items()}
    spec_by_label = {str(spec.get("label") or spec["id"]): spec for spec in specs}
    changed_target = ""
    changed_domain = ""
    for row in edited_rows:
        spec = spec_by_label.get(str(row.get("field") or ""))
        if not spec:
            continue
        spec_id = str(spec.get("id") or "")
        if not _v21_is_domain_summary_row(spec_id):
            continue
        domain_key = spec_id.removeprefix("proposal_")
        for display_label, column_id in label_to_column.items():
            if column_id == "baseline":
                continue
            raw_value = str(row.get(display_label) or "").strip()
            proposal_type = raw_value.upper()
            column_proposals = dict(proposals.get(column_id) or {})
            if proposal_type in {"", "-", "INHERIT"}:
                column_proposals.pop(domain_key, None)
                state = _v21_sync_domain_state_for_column(state, column_id, domain_key, None)
            else:
                proposal = dict(column_proposals.get(domain_key) or {})
                proposal["proposal_type"] = proposal_type
                proposal.setdefault("label", "")
                proposal.setdefault("details", {})
                column_proposals[domain_key] = proposal
                state = _v21_sync_domain_state_for_column(state, column_id, domain_key, proposal)
                changed_target = column_id
                changed_domain = domain_key
            if column_proposals:
                proposals[column_id] = column_proposals
            else:
                proposals.pop(column_id, None)
    state["proposals"] = proposals
    state["preview_cache"] = {}
    _v2_set_state(state)
    _v21_mark_request_preview_stale()
    if changed_target and changed_domain:
        st.session_state["v21_detail_target"] = changed_target
        st.session_state["v21_detail_domain"] = changed_domain


def _v21_render_baseline_reference() -> None:
    state = _v2_state()
    metadata = _v2_metadata_effective(state)
    rows = list(state.get("rows") or [])
    labels = [_v2_row_label(row) for row in rows]
    current_source = str(metadata.get("line_source") or "Existing VDE DB")
    source_options = ["Existing VDE DB", "New test ABC_TOTAL", "Manual / New line"]
    if current_source not in source_options:
        current_source = "Existing VDE DB" if current_source.startswith("Existing") else "New test ABC_TOTAL"
    current_label = str(metadata.get("selected_baseline_label") or "")
    with st.container(border=True):
        st.markdown("**Baseline Reference**")
        with st.form("v21_baseline_reference_form"):
            cols = st.columns([1.0, 3.0, 0.9])
            line_source = cols[0].selectbox(
                "Line source",
                source_options,
                index=source_options.index(current_source),
            )
            baseline_label = cols[1].selectbox(
                "Baseline VDE selector",
                [""] + labels,
                index=([""] + labels).index(current_label) if current_label in labels else 0,
                disabled=line_source != "Existing VDE DB",
            )
            submitted = cols[2].form_submit_button("Load baseline")
        if submitted:
            next_state = _v2_state()
            columns = dict(next_state.get("columns") or {})
            baseline = dict(columns.get("baseline") or {})
            next_metadata = dict(next_state.get("metadata") or {})
            if line_source == "Existing VDE DB":
                selected_row = _v2_resolve_baseline_selector(baseline_label, rows)
                if selected_row:
                    next_state = _v2_apply_selected_baseline_row(next_state, selected_row)
                else:
                    next_metadata["line_source"] = line_source
                    next_metadata["selected_baseline_vde_id"] = None
                    next_metadata["selected_baseline_label"] = ""
                    baseline["line_source"] = line_source
                    baseline["selected_vde_id"] = None
                    columns["baseline"] = baseline
                    next_state["columns"] = columns
                    next_state["metadata"] = next_metadata
            else:
                next_metadata["line_source"] = line_source
                next_metadata["selected_baseline_vde_id"] = None
                next_metadata["selected_baseline_label"] = ""
                baseline["line_source"] = line_source
                baseline["selected_vde_id"] = None
                columns["baseline"] = baseline
                next_state["columns"] = columns
                next_state["metadata"] = next_metadata
            next_state["preview_cache"] = {}
            _v2_set_state(next_state)
            _v21_clear_request_runtime_state(clear_resolution=True)
            st.session_state["v21_flash_message"] = "Baseline reference updated."
            st.rerun()

        effective = _v2_effective_state("baseline")
        summary_rows = [
            {"field": "Baseline ID", "value": str(effective.get("vde_id") or "-")},
            {"field": "Vehicle", "value": " | ".join(part for part in [str(effective.get("make") or ""), str(effective.get("model") or ""), str(effective.get("model_year") or "")] if part) or "-"},
            {"field": "Legislation", "value": str(effective.get("legislation") or "-")},
            {"field": "Cycle", "value": str(effective.get("cycle") or "-")},
            {"field": "ABC_TOTAL_A/B/C", "value": " / ".join(_v2_format_value(effective.get(field), "float") or "-" for field in ["ABC_TOTAL_A", "ABC_TOTAL_B", "ABC_TOTAL_C"])},
            {"field": "Base / curb mass [kg]", "value": _v2_format_value(effective.get("curb_mass_kg"), "mass") or "-"},
            {"field": "EPA ETW / TWC [kg]", "value": _v2_format_value(effective.get("inertia_class"), "mass") or "-"},
            {"field": "Resolved VDE test mass [kg]", "value": _v2_format_value(effective.get("test_mass_kg"), "mass") or "-"},
            {"field": "Test mass basis", "value": str(effective.get("test_mass_basis") or "-")},
            {"field": "Roadload source", "value": str(effective.get("roadload_source_type") or metadata.get("roadload_source_type") or "-")},
        ]
        with st.expander("Show baseline reference details", expanded=False):
            render_vde_workbook_table(
                pd.DataFrame(summary_rows),
                title="Baseline Reference",
                table_id="baseline-reference-summary",
            )


def _v21_render_ppe_request_actions() -> None:
    state = _v2_state()
    baseline_ready = _v21_baseline_is_loaded(state)
    with st.container(border=True):
        st.markdown("**PPE Request Actions**")
        action_cols = st.columns([1.2, 2.8])
        if baseline_ready:
            baseline_snapshot = _v21_current_baseline_snapshot(state)
            has_request_content = bool(dict(state.get("vde_request_import") or {})) or any(
                dict(domain_map or {}) for domain_map in dict(state.get("proposals") or {}).values()
            )
            request_draft = _v21_report_request_draft(state) if has_request_content else None
            try:
                template_bytes = build_prefilled_ppe_template(
                    V21_REQUEST_TEMPLATE_PATH,
                    baseline_snapshot,
                    request_draft=request_draft,
                    metadata={
                        "baseline_source": baseline_snapshot.get("line_source") or "Existing VDE DB",
                        "request_schema_version": "0.1",
                        "source_type": str(dict(dict(request_draft or {}).get("source") or {}).get("source_type") or "UI"),
                    },
                )
                filename = build_prefilled_ppe_template_filename(
                    baseline_snapshot.get("selected_baseline_vde_id") or "baseline",
                    _v21_template_vehicle_label(baseline_snapshot),
                    datetime.now().strftime("%Y-%m-%d"),
                )
                action_cols[0].download_button(
                    "Fill template with request",
                    data=template_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                action_cols[1].caption("Exports the official PPE template with Baseline / Printed filled from the loaded DB baseline. If a request draft already exists, its Baseline Correction and Requested columns are also staged.")
            except Exception as exc:
                action_cols[0].button("Fill template with request", disabled=True, use_container_width=True)
                action_cols[1].error(f"Could not build the PPE template: {exc}")
        else:
            action_cols[0].button("Fill template with request", disabled=True, use_container_width=True)
            action_cols[1].caption("Load a baseline before generating the PPE template.")
    _v21_render_request_import_panel()


def _v21_render_input_workbook_table(specs: list[dict]) -> None:
    _v2_render_light_workbook_styles()
    column_labels = _v21_request_column_labels()
    width_spec = [1.45] + [1.55 for _ in column_labels] + [1.2]
    normal_specs = [spec for spec in specs if not _v21_is_domain_summary_row(str(spec["id"]))]
    proposal_specs = [spec for spec in specs if _v21_is_domain_summary_row(str(spec["id"]))]
    if normal_specs:
        with st.container(border=True):
            with st.form("v21_input_workbook_form"):
                st.markdown("<div class='v21-workbook-frame'>", unsafe_allow_html=True)
                header = st.columns(width_spec)
                header[0].markdown("<div class='v21-workbook-header'>field / proposal</div>", unsafe_allow_html=True)
                for index, display_label in enumerate(column_labels.values(), start=1):
                    header[index].markdown(f"<div class='v21-workbook-header'>{html.escape(display_label)}</div>", unsafe_allow_html=True)
                header[-1].markdown("<div class='v21-workbook-header'>notes</div>", unsafe_allow_html=True)
                edited_rows: list[dict] = []
                for spec in normal_specs:
                    field_label = str(spec.get("label") or spec["id"])
                    cols = st.columns(width_spec)
                    cols[0].markdown(
                        f"<div class='v21-workbook-row'><div class='v21-workbook-fieldcell'>{html.escape(field_label)}</div></div>",
                        unsafe_allow_html=True,
                    )
                    edited_row = {"field": field_label, "notes": str(spec.get("notes") or "")}
                    for index, (scenario_key, display_label) in enumerate(column_labels.items(), start=1):
                        current_value = _v21_input_display_value(scenario_key, spec)
                        cell_class = "is-baseline" if scenario_key == "baseline" else "is-walked"
                        cols[index].markdown(f"<div class='v21-workbook-row'><div class='v21-workbook-cell {cell_class}'>", unsafe_allow_html=True)
                        edited_row[display_label] = _v2_render_form_cell(
                            section_key="Scenario Workbook",
                            scenario_key=scenario_key,
                            spec=spec,
                            current_value=current_value,
                            host=cols[index],
                        )
                        cols[index].markdown("</div></div>", unsafe_allow_html=True)
                    cols[-1].markdown(
                        f"<div class='v21-workbook-row'><div class='v21-workbook-notecell'>{html.escape(str(spec.get('notes') or ''))}</div></div>",
                        unsafe_allow_html=True,
                    )
                    edited_rows.append(edited_row)
                apply = st.form_submit_button("Apply workbook changes")
                st.markdown("</div>", unsafe_allow_html=True)
        if apply:
            editable_matrix_ids = {str(spec["id"]) for spec in VDE_WORKBOOK_V2_MATRIX_SPECS}
            matrix_specs = [
                spec
                for spec in specs
                if str(spec["id"]) in editable_matrix_ids and not str(spec["id"]).startswith("proposal_") and str(spec.get("kind") or "") != "readonly"
            ]
            _v2_apply_field_oriented_editor_df(pd.DataFrame(edited_rows), matrix_specs, scenario_workbook=True)
            _v21_mark_request_preview_stale()
            st.session_state["v21_flash_message"] = "Scenario Input Workbook updated."
            st.rerun()
    if proposal_specs:
        st.caption("Proposal rows")
        with st.container(border=True):
            with st.form("v21_proposal_matrix_form"):
                st.markdown("<div class='v21-workbook-frame'>", unsafe_allow_html=True)
                header = st.columns(width_spec)
                header[0].markdown("<div class='v21-workbook-header'>field / proposal</div>", unsafe_allow_html=True)
                for index, display_label in enumerate(column_labels.values(), start=1):
                    header[index].markdown(f"<div class='v21-workbook-header'>{html.escape(display_label)}</div>", unsafe_allow_html=True)
                header[-1].markdown("<div class='v21-workbook-header'>notes</div>", unsafe_allow_html=True)
                selection_map: dict[tuple[str, str], str] = {}
                for spec in proposal_specs:
                    field_label = str(spec.get("label") or spec["id"])
                    domain_key = str(spec["id"]).removeprefix("proposal_")
                    cols = st.columns(width_spec)
                    cols[0].markdown(
                        f"<div class='v21-workbook-row is-proposal'><div class='v21-workbook-fieldcell'>{html.escape(field_label)}</div></div>",
                        unsafe_allow_html=True,
                    )
                    for index, scenario_key in enumerate(column_labels.keys(), start=1):
                        cell_class = "is-baseline" if scenario_key == "baseline" else "is-walked"
                        cols[index].markdown(f"<div class='v21-workbook-row is-proposal'><div class='v21-workbook-cell {cell_class}'>", unsafe_allow_html=True)
                        selected = _v21_render_proposal_cell(scenario_key, domain_key, cols[index])
                        cols[index].markdown("</div></div>", unsafe_allow_html=True)
                        if selected is not None:
                            selection_map[(scenario_key, domain_key)] = selected
                    cols[-1].markdown(
                        f"<div class='v21-workbook-row is-proposal'><div class='v21-workbook-notecell'>{html.escape(str(spec.get('notes') or ''))}</div></div>",
                        unsafe_allow_html=True,
                    )
                apply_selection = st.form_submit_button("Apply proposal selections")
                st.markdown("</div>", unsafe_allow_html=True)
        if apply_selection:
            _v21_apply_proposal_selection_changes(selection_map)
            _v21_mark_request_preview_stale()
            st.rerun()


def _v21_render_setup_readonly(label: str, value: str) -> None:
    st.markdown(f"<div class='v21-setup-label'>{html.escape(label)}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='v21-setup-readonly'>{html.escape(str(value or '-'))}</div>",
        unsafe_allow_html=True,
    )


def _v21_render_status_chips(column_id: str, state: dict | None = None) -> None:
    state = state or _v2_state()
    column_status = _v2_column_status(column_id)[0]
    proposal_status = _v21_review_status(column_id)
    st.markdown(
        "<div class='v21-setup-chips'>"
        f"<span class='v2-cell-chip {_v2_cell_class_name(column_status)}'>Column | {html.escape(column_status)}</span>"
        f"<span class='v2-cell-chip {_v2_cell_class_name(proposal_status)}'>Proposal | {html.escape(proposal_status)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _v21_baseline_source_summary(state: dict | None = None) -> str:
    state = state or _v2_state()
    metadata = dict(state.get("metadata") or {})
    effective = _v2_effective_state("baseline")
    line_source = str(effective.get("line_source") or metadata.get("line_source") or "-").strip() or "-"
    baseline_label = str(metadata.get("selected_baseline_label") or effective.get("baseline_selector") or "").strip()
    if baseline_label:
        return f"{line_source} | {baseline_label}"
    return line_source


def _v21_render_scenario_column_setup() -> None:
    _v2_render_light_workbook_styles()
    state = _v2_state()
    columns = dict(state.get("columns") or {})
    scenarios = ["baseline", *[column_id for column_id in _v2_column_ids(state) if column_id != "baseline"]]
    request_labels = _v21_request_column_labels(state)
    st.caption("Scenario Column Setup")
    with st.container(border=True):
        with st.form("v21_scenario_column_setup_form"):
            card_cols = st.columns(len(scenarios))
            updates: dict[str, dict[str, str]] = {}
            for index, column_id in enumerate(scenarios):
                column = dict(columns.get(column_id) or {})
                effective = _v2_effective_state(column_id)
                role_class = "is-baseline" if column_id == "baseline" else "is-walked"
                with card_cols[index]:
                    st.markdown(f"<div class='v21-setup-card {role_class}'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='v21-setup-card-title'>{html.escape(request_labels.get(column_id, _v2_column_label(column_id, state)))}</div>",
                        unsafe_allow_html=True,
                    )
                    if column_id == "baseline":
                        _v21_render_setup_readonly("Description", str((state.get("metadata") or {}).get("description") or effective.get("description") or "-"))
                        _v21_render_setup_readonly("Source", _v21_baseline_source_summary(state))
                        _v21_render_setup_readonly("Status", _v2_column_status(column_id)[0])
                    else:
                        description_key = f"v21_setup_description__{column_id}"
                        walk_from_key = f"v21_setup_walk_from__{column_id}"
                        current_description = str(dict(column.get("direct") or {}).get("description") or "")
                        allowed_walk_from = _v2_allowed_walk_from_ids(column_id, state)
                        current_walk_from = str(column.get("walk_from") or "baseline")
                        if current_walk_from not in allowed_walk_from and allowed_walk_from:
                            current_walk_from = allowed_walk_from[0]
                        _v21_widget_default(description_key, current_description)
                        _v21_widget_default(walk_from_key, current_walk_from)
                        st.markdown("<div class='v21-setup-label'>Description</div>", unsafe_allow_html=True)
                        description_value = st.text_input(
                            f"{_v2_column_label(column_id, state)} description",
                            key=description_key,
                            label_visibility="collapsed",
                        )
                        st.markdown("<div class='v21-setup-label'>Walk From</div>", unsafe_allow_html=True)
                        walk_from_value = st.selectbox(
                            f"{_v2_column_label(column_id, state)} walk from",
                            allowed_walk_from,
                            index=allowed_walk_from.index(st.session_state.get(walk_from_key, current_walk_from)) if st.session_state.get(walk_from_key, current_walk_from) in allowed_walk_from else 0,
                            key=walk_from_key,
                            format_func=lambda value: _v2_column_label(value, state),
                            label_visibility="collapsed",
                        )
                        _v21_render_status_chips(column_id, state)
                        updates[column_id] = {
                            "description": str(description_value or "").strip(),
                            "walk_from": str(walk_from_value or "").strip(),
                        }
                    st.markdown("</div>", unsafe_allow_html=True)
            apply = st.form_submit_button("Apply scenario header changes")
    if apply:
        next_state = _v2_state()
        next_columns = dict(next_state.get("columns") or {})
        for column_id, payload in updates.items():
            column = dict(next_columns.get(column_id) or {})
            direct = dict(column.get("direct") or {})
            description = str(payload.get("description") or "").strip()
            if description:
                direct["description"] = description
            else:
                direct.pop("description", None)
            allowed_walk_from = _v2_allowed_walk_from_ids(column_id, next_state)
            walk_from = str(payload.get("walk_from") or "").strip()
            if walk_from not in allowed_walk_from and allowed_walk_from:
                walk_from = allowed_walk_from[0]
            column["direct"] = direct
            column["walk_from"] = walk_from or column.get("walk_from") or "baseline"
            next_columns[column_id] = column
        next_state["columns"] = next_columns
        next_state["preview_cache"] = {}
        _v2_set_state(next_state)
        _v21_mark_request_preview_stale()
        st.session_state["v21_flash_message"] = "Scenario column setup updated."
        st.rerun()
    toggle_cols = st.columns([1.0, 4.4])
    override_bucket = _v21_baseline_printed_override_bucket(state)
    override_count = sum(len(dict(scope_bucket or {})) for scope_bucket in dict(override_bucket or {}).values())
    override_status = "Active" if override_count else "Ready"
    override_class = "is-review" if override_count else "is-neutral"
    toggle_cols[0].markdown(
        f"<div class='v21-detail-head {override_class}'>Baseline Override<br><span class='v21-detail-field-sub'>{html.escape(override_status)} · {override_count} field(s)</span></div>",
        unsafe_allow_html=True,
    )
    toggle_cols[1].caption("Edit Baseline / Printed directly inside the workbook cells below. Overrides apply only to the current request and DB persistence is still decided later in Review & Save.")
    action_cols = st.columns([1, 6])
    if action_cols[0].button("+ Add Column", key="v21_add_column_button"):
        new_key = _v2_add_walked_column()
        st.session_state["v21_flash_message"] = f"Added {request_labels.get(new_key, _v2_column_label(new_key))}."
        st.rerun()
    action_cols[1].caption("Column management lives here so requested proposal columns stay tied to the setup cards above.")
    _v2_render_matrix_column_actions()


def _v21_add_or_update_proposal(column_id: str, domain_key: str, proposal_type: str, label: str, details: dict, *, status: str = "Draft") -> str:
    state = _v2_state()
    proposals = dict(_v21_proposals(state))
    column_proposals = dict(proposals.get(column_id) or {})
    proposal_type = str(proposal_type or "INHERIT").strip().upper() or "INHERIT"
    if proposal_type == "INHERIT":
        column_proposals.pop(domain_key, None)
        if column_proposals:
            proposals[column_id] = column_proposals
        else:
            proposals.pop(column_id, None)
        state["proposals"] = proposals
        state = _v21_sync_domain_state_for_column(state, column_id, domain_key, None)
        preview_cache = dict(state.get("preview_cache") or {})
        preview_cache.pop(column_id, None)
        state["preview_cache"] = preview_cache
        _v2_set_state(state)
        _v21_mark_request_preview_stale()
        return ""
    existing = _v21_get_direct_proposal(column_id, domain_key, state)
    proposal_id = str(existing.get("id") or "").strip()
    if not proposal_id:
        next_seq = int(to_float(state.get("proposal_seq")) or 0) + 1
        state["proposal_seq"] = next_seq
        proposal_id = f"prop_{next_seq}"
    proposal_payload = {
        "id": proposal_id,
        "domain": domain_key,
        "type": proposal_type,
        "proposal_type": proposal_type,
        "label": str(label or "").strip(),
        "details": {key: value for key, value in details.items() if value not in (None, "")},
        "status": str(status or "Draft"),
    }
    column_proposals[domain_key] = proposal_payload
    proposals[column_id] = column_proposals
    state["proposals"] = proposals
    state = _v21_sync_domain_state_for_column(state, column_id, domain_key, proposal_payload)
    preview_cache = dict(state.get("preview_cache") or {})
    preview_cache.pop(column_id, None)
    state["preview_cache"] = preview_cache
    _v2_set_state(state)
    _v21_mark_request_preview_stale()
    return proposal_id


def _v21_remove_proposal(column_id: str, domain_key: str) -> None:
    state = _v2_state()
    proposals = dict(_v21_proposals(state))
    column_proposals = dict(proposals.get(column_id) or {})
    column_proposals.pop(domain_key, None)
    if column_proposals:
        proposals[column_id] = column_proposals
    else:
        proposals.pop(column_id, None)
    state["proposals"] = proposals
    state = _v21_sync_domain_state_for_column(state, column_id, domain_key, None)
    preview_cache = dict(state.get("preview_cache") or {})
    preview_cache.pop(column_id, None)
    state["preview_cache"] = preview_cache
    _v2_set_state(state)
    _v21_mark_request_preview_stale()


def _v21_get_direct_proposal(column_id: str, domain_key: str, state: dict | None = None) -> dict:
    return _v21_domain_state_as_proposal(column_id, domain_key, state)


def _v21_select_proposal(column_id: str, domain_key: str) -> None:
    state = _v2_state()
    state["proposal_target"] = column_id
    _v2_set_state(state)
    st.session_state["v21_detail_target"] = column_id
    st.session_state["v21_detail_domain"] = domain_key
    st.session_state.pop("v21_pending_add_proposal", None)


def _v21_create_direct_proposal(column_id: str, domain_key: str) -> str:
    proposal_type = _v21_domain_proposal_types(domain_key)[0]
    proposal_id = _v21_add_or_update_proposal(column_id, domain_key, proposal_type, "", {})
    _v21_select_proposal(column_id, domain_key)
    return proposal_id


def _v21_start_pending_add_proposal(column_id: str, domain_key: str) -> None:
    st.session_state["v21_pending_add_proposal"] = {
        "target_scenario": column_id,
        "domain": domain_key,
    }


def _v21_pending_add_proposal() -> dict:
    pending = st.session_state.get("v21_pending_add_proposal")
    return dict(pending or {}) if isinstance(pending, dict) else {}


def _v21_domain_proposal_types(domain_key: str) -> list[str]:
    return [item for item in VDE_WORKBOOK_V21_DOMAINS[domain_key]["types"] if item != "INHERIT"]


def _v21_focus_proposal_domain(column_id: str, domain_key: str) -> None:
    state = _v2_state()
    state["proposal_target"] = column_id
    _v2_set_state(state)
    st.session_state["v21_detail_target"] = column_id
    st.session_state["v21_detail_domain"] = domain_key


def _v21_is_pending_for_cell(column_id: str, domain_key: str) -> bool:
    pending = _v21_pending_add_proposal()
    return str(pending.get("target_scenario") or "") == column_id and str(pending.get("domain") or "") == domain_key


def _v21_create_proposal_from_type(column_id: str, domain_key: str, proposal_type: str) -> str:
    state = _v2_state()
    proposals = dict(_v21_proposals(state))
    column_proposals = dict(proposals.get(column_id) or {})
    proposal_type = str(proposal_type or "").strip().upper()
    existing = _v21_get_direct_proposal(column_id, domain_key, state)
    proposal_id = str(existing.get("id") or "").strip()
    if not proposal_id:
        next_seq = int(to_float(state.get("proposal_seq")) or 0) + 1
        proposal_id = f"prop_{next_seq}"
        state["proposal_seq"] = next_seq
    default_label = _v21_proposal_type_label(proposal_type)
    proposal_payload = {
        "id": proposal_id,
        "domain": domain_key,
        "type": proposal_type,
        "proposal_type": proposal_type,
        "label": default_label,
        "details": {},
        "status": "Draft",
    }
    column_proposals[domain_key] = proposal_payload
    proposals[column_id] = column_proposals
    state["proposals"] = proposals
    state = _v21_sync_domain_state_for_column(state, column_id, domain_key, proposal_payload)
    preview_cache = dict(state.get("preview_cache") or {})
    preview_cache.pop(column_id, None)
    state["preview_cache"] = preview_cache
    _v2_set_state(state)
    _v21_mark_request_preview_stale()
    _v21_select_proposal(column_id, domain_key)
    return proposal_id


def _v21_has_direct_proposal(column_id: str, domain_key: str, state: dict | None = None) -> bool:
    return bool(_v21_get_direct_proposal(column_id, domain_key, state))


def _v21_proposal_cell_chip(column_id: str, domain_key: str, state: dict | None = None) -> tuple[str, str]:
    state = state or _v2_state()
    proposal = _v21_get_direct_proposal(column_id, domain_key, state)
    if proposal:
        badge = _v21_proposal_badge_text(proposal)
        if str(st.session_state.get("v21_detail_target") or "") == column_id and str(st.session_state.get("v21_detail_domain") or "") == domain_key:
            return f"Selected | {badge}", "is-neutral"
        status = _v21_resolved_proposal_status(column_id, domain_key, proposal, state)
        if status == "Invalid":
            return _v21_proposal_badge_text({"id": proposal.get("id"), "label": "Invalid", "proposal_type": proposal.get("proposal_type")}), "is-missing"
        if status == "Missing":
            return _v21_proposal_badge_text({"id": proposal.get("id"), "label": "Missing", "proposal_type": proposal.get("proposal_type")}), "is-missing"
        if status == "Review":
            return _v21_proposal_badge_text({"id": proposal.get("id"), "label": "Review", "proposal_type": proposal.get("proposal_type")}), "is-review"
        if status == "Draft":
            return _v21_proposal_badge_text({"id": proposal.get("id"), "label": "Draft", "proposal_type": proposal.get("proposal_type")}), "is-review"
        return badge, "is-ok"
    return f"\u21b3 Inherited from {_v21_walk_from_label(column_id, state)}", "is-inherit"


def _v21_baseline_proposal_cell_summary(domain_key: str, state: dict | None = None) -> tuple[str, str]:
    state = state or _v2_state()
    if domain_key == "aero":
        baseline_cda = to_float(_v21_reference_raw_value("baseline", "baseline_CdA", state))
        if baseline_cda is None:
            return "CdA ref missing", "is-missing"
        return f"CdA ref: {_v2_format_value(baseline_cda, 'float') or baseline_cda}", "is-ok"
    if domain_key == "tire":
        baseline_tire = str(_v21_reference_raw_value("baseline", "baseline_tire_code", state) or "").strip()
        if not baseline_tire:
            return "Tire ref missing", "is-missing"
        return f"Tire ref: {baseline_tire}", "is-ok"
    if domain_key in {"transmission", "brake", "axle_hubs", "parasitic"}:
        baseline_values, has_reference = _v21_component_inherited_baseline(domain_key, "baseline", state)
        if not has_reference:
            return "ABC ref missing", "is-missing"
        formatted = " / ".join(_v2_format_value(value, "float") or "-" for value in baseline_values)
        return f"ABC ref: {formatted}", "is-ok"
    if domain_key == "mass":
        return "Mass baseline context", "is-neutral"
    return "Baseline reference", "is-neutral"


def _v21_proposal_select_options(column_id: str, domain_key: str, state: dict | None = None) -> list[str]:
    state = state or _v2_state()
    if column_id == "baseline":
        return ["baseline"]
    inherited_label = f"inherit::{_v21_walk_from_label(column_id, state)}"
    configured = [str(item.get("value") or "") for item in VDE_WORKBOOK_V21_MATRIX_SELECTIONS.get(domain_key, [])]
    return [inherited_label, *(configured or _v21_domain_proposal_types(domain_key))]


def _v21_proposal_select_label(option: str) -> str:
    text = str(option or "").strip()
    if text == "baseline":
        return "baseline/reference"
    if text.startswith("inherit::"):
        return f"\u21b3 Inherited from {text.split('::', 1)[1]}"
    for domain_options in VDE_WORKBOOK_V21_MATRIX_SELECTIONS.values():
        for config in domain_options:
            if str(config.get("value") or "").strip() == text:
                return str(config.get("label") or text)
    return _v21_proposal_type_label(text)


def _v21_render_proposal_cell(scenario_key: str, domain_key: str, host) -> str | None:
    state = _v2_state()
    if scenario_key == "baseline":
        summary_text, summary_class = _v21_baseline_proposal_cell_summary(domain_key, state)
        host.markdown(
            f"<span class='v2-cell-chip {summary_class}'>{html.escape(str(summary_text))}</span>",
            unsafe_allow_html=True,
        )
        return None
    options = _v21_proposal_select_options(scenario_key, domain_key, state)
    proposal = _v21_get_direct_proposal(scenario_key, domain_key, state)
    current_type = _v21_proposal_type_for_cell(scenario_key, domain_key, state)
    current_value = _v21_matrix_selection_value_for_proposal(domain_key, proposal) if current_type not in {"", "inherit"} else options[0]
    key = _v21_proposal_select_widget_key(scenario_key, domain_key)
    _v21_widget_default(key, current_value)
    selected = host.selectbox(
        "Proposal selection",
        options,
        index=options.index(st.session_state.get(key, current_value)) if st.session_state.get(key, current_value) in options else 0,
        key=key,
        format_func=_v21_proposal_select_label,
        label_visibility="collapsed",
    )
    return str(selected)


def _v21_transition_proposal_details(domain_key: str, old_type: str, new_type: str, details: dict | None) -> dict:
    normalized = _v21_normalize_details(details or {})
    old_type = str(old_type or "").strip().upper()
    new_type = str(new_type or "").strip().upper()
    if not normalized or old_type == new_type:
        return normalized
    keep_fields = {
        "source",
        "notes",
        "mass_kg",
        "test_mass_kg",
        "test_mass_basis",
        "front_pressure_psi",
        "rear_pressure_psi",
        "hot_front_pressure_psi",
        "hot_rear_pressure_psi",
        "weight_dist_fr_pct",
        "baseline_tire_code",
    }
    new_fields = set(_v21_detail_fields_for_type(domain_key, new_type))
    compact_fields = set(_v21_compact_fields_for_proposal(domain_key, new_type, normalized, {}))
    advanced_fields = set(_v21_advanced_fields_for_proposal(domain_key, new_type, normalized, {}))
    allowed = {_v21_canonical_field_id(field_id) for field_id in (new_fields | compact_fields | advanced_fields | keep_fields)}
    return {
        canonical_key: value
        for canonical_key, value in normalized.items()
        if _v21_canonical_field_id(canonical_key) in allowed
    }


def _v21_seed_component_selection_details(domain_key: str, selection_value: str, details: dict | None) -> dict:
    details = _v21_normalize_details(details or {})
    config = _v21_matrix_selection_config(domain_key, selection_value)
    proposal_type = str(config.get("proposal_type") or "").strip().upper()
    seeded = dict(details)
    seed_values = dict(config.get("seed") or {})
    if proposal_type == "UPDATE_TRANS_DRAG_ABC":
        seeded.update(seed_values)
        seeded.pop("method", None)
        if seeded.get("change_mode") == "Absolute ABC":
            for field_id in ("delta_A", "delta_B", "delta_C"):
                seeded.pop(field_id, None)
        else:
            for field_id in ("new_trans_A", "new_trans_B", "new_trans_C", "baseline_component_reference_mode", "baseline_update_requested"):
                seeded.pop(field_id, None)
    elif proposal_type == "BRAKE_DRAG_CHANGE":
        seeded.update(seed_values)
        if seeded.get("method") == "Residual torque":
            for field_id in ("change_mode", "brake_A", "brake_B", "brake_C", "delta_A", "delta_B", "delta_C", "baseline_component_reference_mode", "baseline_update_requested", "baseline_component_A", "baseline_component_B", "baseline_component_C"):
                seeded.pop(field_id, None)
        elif seeded.get("change_mode") == "Absolute ABC":
            for field_id in ("delta_A", "delta_B", "delta_C", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m"):
                seeded.pop(field_id, None)
        else:
            for field_id in ("brake_A", "brake_B", "brake_C", "baseline_component_reference_mode", "baseline_update_requested", "baseline_component_A", "baseline_component_B", "baseline_component_C", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m"):
                seeded.pop(field_id, None)
    elif proposal_type in {"AXLE_HUB_DRAG_CHANGE", "PARASITIC_LOSS_CHANGE"}:
        seeded.update(seed_values)
        absolute_fields = ("axle_hub_A", "axle_hub_B", "axle_hub_C") if proposal_type == "AXLE_HUB_DRAG_CHANGE" else ("parasitic_A", "parasitic_B", "parasitic_C")
        if seeded.get("change_mode") == "Absolute ABC":
            for field_id in ("delta_A", "delta_B", "delta_C"):
                seeded.pop(field_id, None)
        else:
            for field_id in (*absolute_fields, "baseline_component_reference_mode", "baseline_update_requested", "baseline_component_A", "baseline_component_B", "baseline_component_C"):
                seeded.pop(field_id, None)
    return _v21_sync_component_reference_details(seeded)


def _v21_apply_proposal_selection_changes(selection_map: dict[tuple[str, str], str]) -> tuple[int, int, int]:
    state = _v2_state()
    proposals = dict(_v21_proposals(state))
    column_labels = _v21_request_column_labels(state)
    created = 0
    updated = 0
    removed = 0
    focus_target: tuple[str, str] | None = None
    removed_focus: tuple[str, str] | None = None
    for scenario_key in _v2_column_ids(state):
        if scenario_key == "baseline":
            continue
        for domain_key in VDE_WORKBOOK_V21_DOMAINS:
            selected_value = str(selection_map.get((scenario_key, domain_key), "") or "").strip()
            if not selected_value:
                continue
            column_proposals = dict(proposals.get(scenario_key) or {})
            existing = _v21_get_direct_proposal(scenario_key, domain_key, state)
            current_type = str(existing.get("proposal_type") or existing.get("type") or "inherit").strip().upper() or "INHERIT"
            current_selection = _v21_matrix_selection_value_for_proposal(domain_key, existing) if existing else ""
            inherited_selected = selected_value.startswith("inherit::") or selected_value.lower() == "inherit"
            if inherited_selected and not existing:
                continue
            if inherited_selected:
                column_proposals.pop(domain_key, None)
                if column_proposals:
                    proposals[scenario_key] = column_proposals
                else:
                    proposals.pop(scenario_key, None)
                state = _v21_sync_domain_state_for_column(state, scenario_key, domain_key, None)
                removed += 1
                removed_focus = (scenario_key, domain_key)
                continue
            selection_config = _v21_matrix_selection_config(domain_key, selected_value)
            desired_type = str(selection_config.get("proposal_type") or selected_value).upper()
            if existing and current_type == desired_type and current_selection == selected_value:
                continue
            next_seq = int(to_float(state.get("proposal_seq")) or 0)
            proposal_id = str(existing.get("id") or "").strip()
            if not proposal_id:
                next_seq += 1
                state["proposal_seq"] = next_seq
                proposal_id = f"prop_{next_seq}"
                created += 1
            else:
                updated += 1
            label = str(existing.get("label") or "").strip()
            matrix_label = _v21_proposal_select_label(selected_value)
            if not label or label in {_v21_proposal_type_label(current_type), _v21_proposal_select_label(current_selection)}:
                label = matrix_label
            details = _v21_transition_proposal_details(domain_key, current_type, desired_type, existing.get("details") or {})
            details = _v21_seed_component_selection_details(domain_key, selected_value, details)
            status_value, _, _, _ = _v21_validate_proposal_details(scenario_key, domain_key, desired_type, details, state)
            proposal_payload = {
                "id": proposal_id,
                "domain": domain_key,
                "type": desired_type,
                "proposal_type": desired_type,
                "label": label,
                "details": {key: value for key, value in details.items() if value not in (None, "")},
                "status": str(status_value or "Draft"),
            }
            column_proposals[domain_key] = proposal_payload
            proposals[scenario_key] = column_proposals
            state = _v21_sync_domain_state_for_column(state, scenario_key, domain_key, proposal_payload)
            focus_target = (scenario_key, domain_key)
    state["proposals"] = proposals
    preview_cache = dict(state.get("preview_cache") or {})
    for scenario_key, _ in selection_map:
        preview_cache.pop(scenario_key, None)
    state["preview_cache"] = preview_cache
    _v2_set_state(state)
    if focus_target:
        _v21_select_proposal(*focus_target)
        st.session_state["v21_flash_message"] = f"Applied proposal selections. Opened {VDE_WORKBOOK_V21_DOMAINS[focus_target[1]]['label']} for {column_labels.get(focus_target[0], focus_target[0])}."
    elif removed_focus:
        _v21_focus_proposal_domain(*removed_focus)
        inherited_from = _v21_walk_from_label(removed_focus[0], _v2_state())
        st.session_state["v21_flash_message"] = f"Applied proposal selections. {VDE_WORKBOOK_V21_DOMAINS[removed_focus[1]]['label']} now inherits from {inherited_from} in {column_labels.get(removed_focus[0], removed_focus[0])}."
    else:
        st.session_state["v21_flash_message"] = "No proposal selection changes."
    return created, updated, removed


def _v21_detail_fields_for_type(domain_key: str, proposal_type: str) -> list[str]:
    proposal_type = str(proposal_type or "INHERIT").strip().upper()
    domain_fields = VDE_WORKBOOK_V21_DETAIL_FIELDS.get(domain_key) or {}
    configured = domain_fields.get(proposal_type)
    if configured is not None:
        return list(configured)
    return list((VDE_WORKBOOK_V21_DOMAINS.get(domain_key) or {}).get("details") or [])


VDE_WORKBOOK_V21_DETAIL_SELECT_OPTIONS = {
    "shift_steps": ["+1", "-1", "+2", "-2"],
    "target_side": ["Low", "Nominal", "High"],
    "reference_source": ["Inherited reference", "Baseline", "Manual"],
    "line_type": ["TML", "TMH"],
    "preset": ["+100 kg", "+300 lb", "custom"],
    "method": ["Brake ABC", "Residual torque"],
    "change_mode": ["Delta ABC", "Absolute ABC"],
    "baseline_component_reference_mode": [
        "Enter manual baseline component ABC, do not update baseline",
        "Enter manual baseline component ABC and update baseline",
    ],
    "tire_load_mass_basis": ["TEST_MASS", "TWC"],
    "load_basis": ["TEST_MASS", "TWC"],
    "pressure_basis": ["Placard", "Test", "Manual"],
    "neutral_drag_source": ["Baseline", "Manual", "Test"],
    "percent_basis": ["Baseline losses", "Net delta", "Manual"],
    "rule_version": ["Current", "Legacy", "Manual"],
}

VDE_WORKBOOK_V21_REFERENCE_FIELD_RULES = {
    "mass_kg": {"allow_zero": False, "min_value": 0.0, "advanced_only": False, "source_label": "Mass DB/source"},
    "inertia_class": {"allow_zero": False, "min_value": 0.0, "advanced_only": True, "source_label": "EPA ETW/TWC source"},
    "baseline_CdA": {"allow_zero": False, "min_value": 0.0, "advanced_only": False, "source_label": "CdA source"},
    "baseline_tire_code": {"allow_zero": False, "advanced_only": False, "source_label": "Baseline tire source"},
    "baseline_SMERF_optional": {"allow_zero": False, "min_value": 0.0, "advanced_only": False, "source_label": "SMERF source"},
    "baseline_RRC_optional": {"allow_zero": False, "min_value": 0.0, "advanced_only": False, "source_label": "RRC source"},
    "front_pressure_psi": {"allow_zero": False, "min_value": 0.0, "advanced_only": False, "source_label": "Pressure source"},
    "rear_pressure_psi": {"allow_zero": False, "min_value": 0.0, "advanced_only": False, "source_label": "Pressure source"},
    "hot_front_pressure_psi": {"allow_zero": False, "min_value": 0.0, "advanced_only": False, "source_label": "Hot pressure source"},
    "hot_rear_pressure_psi": {"allow_zero": False, "min_value": 0.0, "advanced_only": False, "source_label": "Hot pressure source"},
    "tire_load_mass_basis": {"allow_zero": False, "advanced_only": False, "source_label": "Tire load source"},
    "weight_dist_fr_pct": {"allow_zero": False, "min_value": 0.0, "advanced_only": False, "source_label": "Weight distribution source"},
    "baseline_trans_A": {"allow_zero": True, "min_value": 0.0, "advanced_only": False, "source_label": "Transmission ABC source"},
    "baseline_trans_B": {"allow_zero": True, "min_value": 0.0, "advanced_only": False, "source_label": "Transmission ABC source"},
    "baseline_trans_C": {"allow_zero": True, "min_value": 0.0, "advanced_only": False, "source_label": "Transmission ABC source"},
    "baseline_component_A": {"allow_zero": True, "min_value": 0.0, "advanced_only": False, "source_label": "Component ABC source"},
    "baseline_component_B": {"allow_zero": True, "min_value": 0.0, "advanced_only": False, "source_label": "Component ABC source"},
    "baseline_component_C": {"allow_zero": True, "min_value": 0.0, "advanced_only": False, "source_label": "Component ABC source"},
}

VDE_WORKBOOK_V21_CANONICAL_DETAIL_ALIASES = {
    "mass_kg": ("mass_kg", "curb_mass_kg"),
    "test_mass_kg": ("test_mass_kg", "effective_test_mass_kg"),
    "test_mass_basis": ("test_mass_basis", "vde_mass_basis"),
    "test_mass_low_kg": ("test_mass_low_kg", "TML_kg"),
    "test_mass_high_kg": ("test_mass_high_kg", "TMH_kg"),
    "gvwr_kg": ("gvwr_kg", "GVWR_kg"),
    "gcwr_kg": ("gcwr_kg", "GCWR_kg"),
    "trailer_mass_kg": ("trailer_mass_kg", "trailer_weight_kg"),
    "front_pressure_psi": ("front_pressure_psi", "psi_front"),
    "rear_pressure_psi": ("rear_pressure_psi", "psi_rear"),
    "hot_front_pressure_psi": ("hot_front_pressure_psi", "hot_psi_front"),
    "hot_rear_pressure_psi": ("hot_rear_pressure_psi", "hot_psi_rear"),
    "tire_improvement_pct": ("tire_improvement_pct", "improvement_pct"),
    "tire_load_mass_basis": ("tire_load_mass_basis", "load_basis"),
    "frontal_area_m2": ("frontal_area_m2", "Af_optional"),
}

VDE_WORKBOOK_V21_LEGACY_DETAIL_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in VDE_WORKBOOK_V21_CANONICAL_DETAIL_ALIASES.items()
    for alias in aliases
}

VDE_WORKBOOK_V21_DETAIL_NOTES = {
    "proposal_type": "Proposal method",
    "proposal_label": "Short label",
    "status": "Validation status",
    "baseline_CdA": "Baseline/reference CdA",
    "new_CdA": "Calculated for delta modes; direct input for absolute modes",
    "Cd_display": "Calculated display value",
    "mass_kg": "Base / curb mass [kg]",
    "inertia_class": "EPA ETW / TWC [kg]",
    "test_mass_kg": "Resolved VDE test mass [kg]",
    "test_mass_basis": "Resolved test mass basis",
    "mass_rule_status": "Mass validation status",
    "mass_rule_notes": "Mass validation notes",
    "payload_kg": "Calculated payload display",
    "vehicle_mass_at_gcwr": "Calculated as GCWR - trailer mass",
    "trailer_roadload_status": "Trailer roadload completeness",
    "baseline_tire_code": "Required baseline tire",
    "new_tire_code": "New tire package",
    "tire_db_id": "Optional Tire DB id",
    "tire_size": "Tire size",
    "front_pressure_psi": "Cold front tire pressure",
    "rear_pressure_psi": "Cold rear tire pressure",
    "hot_front_pressure_psi": "Optional hot front pressure",
    "hot_rear_pressure_psi": "Optional hot rear pressure",
    "tire_improvement_pct": "Positive = lower RR in EcoDrive convention",
    "weight_dist_fr_pct": "Front axle weight distribution",
    "tire_load_mass_used_kg": "Applied tire load mass",
    "final_RRC_calculated": "Calculated final RRC",
    "baseline_SMERF_optional": "Reference SMERF",
    "delta_SMERF_optional": "SMERF delta",
    "baseline_RRC_optional": "Reference RRC",
    "delta_RRC_optional": "RRC delta",
    "baseline_trans_A": "Reference transmission A",
    "baseline_trans_B": "Reference transmission B",
    "baseline_trans_C": "Reference transmission C",
    "new_trans_A": "Absolute A if using new ABC",
    "new_trans_B": "Absolute B if using new ABC",
    "new_trans_C": "Absolute C if using new ABC",
    "delta_A": "Delta A",
    "delta_B": "Delta B",
    "delta_C": "Delta C",
    "shift_steps": "TWC step shift",
    "target_side": "Target side / bucket",
    "reference_source": "Reference source for the mass lookup",
    "reference_mass_kg": "Reference mass",
    "target_mass_kg": "Target mass after shift",
    "mass_kg": "Selected WLTP mass",
    "optional_weight_kg": "Optional / payload weight",
    "laden_mass_kg": "Laden mass",
    "wltp_mass_pair_id": "Traceability id",
    "loss_pct": "Transmission loss percentage",
    "brake_drag_force_N": "Resolved drag force",
    "residual_torque_total_Nm": "Total residual torque",
    "source": "Source / provenance",
    "notes": "Scenario notes",
}


def _v21_canonical_field_id(field_id: str) -> str:
    return str(VDE_WORKBOOK_V21_LEGACY_DETAIL_TO_CANONICAL.get(field_id) or field_id)


def _v21_reference_field_rule(field_id: str) -> dict:
    return dict(VDE_WORKBOOK_V21_REFERENCE_FIELD_RULES.get(_v21_canonical_field_id(field_id), {}))


def _v21_requested_field_baseline_field(field_id: str, domain_key: str | None = None) -> str:
    canonical = _v21_canonical_field_id(field_id)
    if canonical == "new_CdA":
        return "baseline_CdA"
    if canonical in {"new_trans_A", "new_trans_B", "new_trans_C"}:
        return canonical.replace("new_trans_", "baseline_trans_")
    if canonical in {"brake_A", "brake_B", "brake_C"}:
        return canonical.replace("brake_", "baseline_component_")
    if canonical in {"axle_hub_A", "axle_hub_B", "axle_hub_C"}:
        return canonical.replace("axle_hub_", "baseline_component_")
    if canonical in {"parasitic_A", "parasitic_B", "parasitic_C"}:
        return canonical.replace("parasitic_", "baseline_component_")
    if canonical == "new_tire_code":
        return "baseline_tire_code"
    return canonical


def _v21_baseline_effective_aliases(field_id: str, domain_key: str | None = None) -> list[str]:
    canonical = _v21_canonical_field_id(field_id)
    aliases = [canonical, _v21_reference_field_alias(canonical)]
    if canonical == "mass_kg":
        aliases.append("curb_mass_kg")
    if canonical == "baseline_CdA":
        aliases.extend(["CdA", "cda_m2"])
    if canonical in {"baseline_component_A", "baseline_component_B", "baseline_component_C"}:
        suffix = canonical.rsplit("_", 1)[-1]
        domain_alias = {
            "brake": f"brake_{suffix}",
            "axle_hubs": f"axle_hub_{suffix}",
            "parasitic": f"parasitic_{suffix}",
        }.get(str(domain_key or ""))
        if domain_alias:
            aliases.append(domain_alias)
    if canonical == "gvwr_kg":
        aliases.append("GVWR_kg")
    if canonical == "gcwr_kg":
        aliases.append("GCWR_kg")
    if canonical == "trailer_mass_kg":
        aliases.append("trailer_weight_kg")
    if canonical == "mass_profile_gvwr_kg":
        aliases.extend(["gvwr_kg", "GVWR_kg"])
    if canonical == "mass_profile_gcwr_kg":
        aliases.extend(["gcwr_kg", "GCWR_kg"])
    if canonical == "mass_profile_trailer_mass_kg":
        aliases.extend(["trailer_mass_kg", "trailer_weight_kg"])
    if canonical == "mass_profile_custom_input_kg":
        aliases.append("test_mass_kg")
    if canonical in {"axle_hub_delta_A", "axle_hub_delta_B", "axle_hub_delta_C"}:
        aliases.append(canonical.replace("axle_hub_", ""))
    if canonical in {"parasitic_delta_A", "parasitic_delta_B", "parasitic_delta_C"}:
        aliases.append(canonical.replace("parasitic_", ""))
    return list(dict.fromkeys([alias for alias in aliases if alias]))


def _v21_baseline_field_kind(field_id: str, domain_key: str | None = None) -> str:
    spec_map = _v2_field_spec_map()
    for alias in _v21_baseline_effective_aliases(field_id, domain_key):
        spec = spec_map.get(alias)
        if spec:
            return str(spec.get("kind") or "text")
    rule = _v21_reference_field_rule(field_id)
    if "min_value" in rule or _v21_canonical_field_id(field_id).startswith("baseline_"):
        return "float"
    return "text"


def _v21_baseline_printed_override_active(field_id: str, domain_key: str | None = None, state: dict | None = None) -> bool:
    return _v21_baseline_printed_override_value(field_id, domain_key, state) not in (None, "")


def _v21_display_baseline_value(field_id: str, domain_key: str | None = None, state: dict | None = None):
    state = state or _v2_state()
    override_value = _v21_baseline_printed_override_value(field_id, domain_key, state)
    if override_value not in (None, ""):
        return override_value
    return _v21_reference_raw_value("baseline", field_id, state, include_override=False)


def _v21_baseline_value_for_delta(field_id: str, domain_key: str | None = None, state: dict | None = None):
    value = _v21_display_baseline_value(field_id, domain_key, state)
    numeric_value = to_float(value)
    return 0.0 if numeric_value is None else numeric_value


def _v21_apply_printed_overrides_to_effective(effective: dict, state: dict | None = None) -> dict:
    state = state or _v2_state()
    bucket = _v21_baseline_printed_override_bucket(state)
    if not bucket:
        return effective
    spec_map = _v2_field_spec_map()
    for scope, scope_bucket in dict(bucket or {}).items():
        domain_key = None if scope == V21_BASELINE_PRINTED_GLOBAL_SCOPE else str(scope)
        for field_id, raw_value in dict(scope_bucket or {}).items():
            aliases = _v21_baseline_effective_aliases(field_id, domain_key)
            kind = _v21_baseline_field_kind(field_id, domain_key)
            parsed = _v2_parse_value(raw_value, kind)
            if parsed in (None, ""):
                continue
            for alias in aliases:
                effective[alias] = parsed
            if _v21_canonical_field_id(field_id) == "mass_kg":
                effective["curb_mass_kg"] = parsed
    return effective


def _v21_ppe_baseline_field_editable(
    domain_key: str,
    proposal_type: str,
    field_id: str,
    details: dict | None = None,
    context: dict | None = None,
    *,
    advanced: bool = False,
) -> bool:
    canonical = _v21_canonical_field_id(field_id)
    if canonical in {"proposal_type", "proposal_label", "status", "baseline_component_reference_mode", "change_mode", "method", "source", "notes"}:
        return False
    if canonical.startswith("delta_") or canonical.startswith("new_"):
        return False
    if canonical in {"Cd_display", "mass_rule_status", "mass_rule_notes", "test_mass_basis", "final_RRC_calculated", "tire_calc_source", "tire_calc_notes", "trailer_roadload_status", "vehicle_mass_at_gcwr", "brake_drag_force_N"}:
        return False
    if canonical not in VDE_WORKBOOK_V21_PPE_BASELINE_EDITABLE_FIELDS and not _v21_is_reference_field_for_proposal(domain_key, proposal_type, canonical, details):
        return False
    return _v21_is_field_used(domain_key, proposal_type, canonical, details, context, advanced=advanced) or _v21_is_reference_field_for_proposal(domain_key, proposal_type, canonical, details)


def _v21_baseline_printed_status(field_id: str, domain_key: str | None = None, state: dict | None = None) -> tuple[str, str]:
    state = state or _v2_state()
    override_value = _v21_baseline_printed_override_value(field_id, domain_key, state)
    inherited_value = _v21_reference_raw_value("baseline", field_id, state, include_override=False)
    if override_value not in (None, ""):
        return "Review", "Manual baseline override"
    if inherited_value not in (None, ""):
        return "OK", "Printed from DB/source"
    return "Missing", "No printed baseline/source value"

def _v21_reference_value_valid(field_id: str, value) -> bool:
    canonical = _v21_canonical_field_id(field_id)
    if value in (None, ""):
        return False
    rule = _v21_reference_field_rule(canonical)
    numeric_value = to_float(value)
    if numeric_value is None:
        return bool(str(value).strip())
    allow_zero = bool(rule.get("allow_zero", True))
    min_value = rule.get("min_value")
    if not allow_zero and abs(float(numeric_value)) <= 1e-12:
        return False
    if min_value is not None and float(numeric_value) < float(min_value):
        return False
    return True


def _v21_mass_inertia_option_values() -> list[str]:
    values = []
    for row in _inertia_class_table():
        mass_value = to_float(row.get("inertia_class_kg"))
        if mass_value is not None:
            values.append(f"{mass_value:.0f}")
    return list(dict.fromkeys(values))


def _v21_detail_aliases(field_id: str) -> tuple[str, ...]:
    canonical = _v21_canonical_field_id(field_id)
    return tuple(VDE_WORKBOOK_V21_CANONICAL_DETAIL_ALIASES.get(canonical, (canonical,)))


def _v21_detail_value(details: dict | None, field_id: str, default=None):
    details = dict(details or {})
    for alias in _v21_detail_aliases(field_id):
        if alias in details and details.get(alias) not in (None, ""):
            return details.get(alias)
    return default


def _v21_normalize_details(details: dict | None) -> dict:
    details = dict(details or {})
    normalized: dict = {}
    for key, value in details.items():
        canonical = _v21_canonical_field_id(str(key))
        if canonical not in normalized or key == canonical:
            normalized[canonical] = value
    return normalized


def _v21_display_fields_for_type(domain_key: str, proposal_type: str) -> list[str]:
    proposal_type = str(proposal_type or "INHERIT").strip().upper()
    custom_fields = {
        ("mass", "MASS_TWC_SHIFT"): ["shift_steps", "target_side", "reference_source", "reference_mass_kg", "target_mass_kg", "source", "notes"],
        ("mass", "EPA_PLUS_1_TWC"): ["shift_steps", "target_side", "reference_source", "reference_mass_kg", "target_mass_kg", "source", "notes"],
        ("aero", "AERO_DELTA_CDA"): ["baseline_CdA", "delta_CdA", "new_CdA", "source", "notes"],
        ("aero", "AERO_ABSOLUTE_CDA"): ["baseline_CdA", "new_CdA", "frontal_area_m2", "Cd_display", "source", "notes"],
    }
    fields = custom_fields.get((domain_key, proposal_type))
    if fields is not None:
        return list(fields)
    return _v21_detail_fields_for_type(domain_key, proposal_type)


def _v21_detail_field_kind(field_id: str) -> str:
    field_id = _v21_canonical_field_id(field_id)
    explicit = {
        "baseline_CdA": "float",
        "new_CdA": "float",
        "Cd_display": "float",
        "tire_improvement_pct": "float",
        "loss_pct": "float",
        "front_pressure_psi": "float",
        "rear_pressure_psi": "float",
        "hot_front_pressure_psi": "float",
        "hot_rear_pressure_psi": "float",
        "frontal_area_m2": "float",
        "weight_dist_fr_pct": "float",
        "wheel_radius_m": "float",
        "baseline_SMERF_optional": "float",
        "delta_SMERF_optional": "float",
        "baseline_RRC_optional": "rrc",
        "delta_RRC_optional": "rrc",
        "final_RRC_calculated": "rrc",
        "baseline_component_A": "force",
        "baseline_component_B": "force_per_speed",
        "baseline_component_C": "force_per_speed_squared",
    }
    if field_id in explicit:
        return explicit[field_id]
    if field_id.endswith("_kg"):
        return "mass"
    if field_id.endswith("_Nm"):
        return "float"
    if field_id in {"brake_A", "delta_A", "new_trans_A", "baseline_trans_A", "axle_hub_A", "parasitic_A", "trailer_A"}:
        return "force"
    if field_id in {"brake_B", "delta_B", "new_trans_B", "baseline_trans_B", "axle_hub_B", "parasitic_B", "trailer_B"}:
        return "force_per_speed"
    if field_id in {"brake_C", "delta_C", "new_trans_C", "baseline_trans_C", "axle_hub_C", "parasitic_C", "trailer_C"}:
        return "force_per_speed_squared"
    if field_id in {"tire_db_id"}:
        return "int"
    return str((_v2_field_spec_map().get(field_id) or {}).get("kind") or "text")


def _v21_detail_field_note(field_id: str) -> str:
    canonical = _v21_canonical_field_id(field_id)
    spec = _v2_field_spec_map().get(canonical) or _v2_field_spec_map().get(field_id) or {}
    return str(VDE_WORKBOOK_V21_DETAIL_NOTES.get(canonical) or VDE_WORKBOOK_V21_DETAIL_NOTES.get(field_id) or spec.get("notes") or "")


def _v21_local_delta_note(column_id: str, domain_key: str, proposal_type: str, field_id: str, details: dict | None = None, state: dict | None = None) -> str:
    state = state or _v2_state()
    proposal_type = str(proposal_type or "").strip().upper()
    field_id = _v21_canonical_field_id(field_id)
    details = _v21_normalize_details(details)
    if proposal_type == "AERO_ABSOLUTE_CDA" and field_id == "delta_CdA":
        return f"Local delta vs {_v21_walk_from_label(column_id, state)}."
    if domain_key in {"transmission", "brake", "axle_hubs", "parasitic"} and field_id in {"delta_A", "delta_B", "delta_C"}:
        if str(details.get("change_mode") or "").strip() == "Absolute ABC":
            return f"Local delta vs {_v21_walk_from_label(column_id, state)}."
    return ""


def _v21_reference_fields_for_proposal(domain_key: str, proposal_type: str, details: dict | None = None) -> list[str]:
    proposal_type = str(proposal_type or "").strip().upper()
    details = _v21_normalize_details(details)
    if domain_key == "mass":
        if proposal_type in {"EPA_STATUS", "MASS_TWC_SHIFT", "EPA_PLUS_1_TWC"}:
            return ["mass_kg", "inertia_class"]
        if proposal_type in {"PERFORMANCE_CURB_MASS", "WLTP_MASS_LINE", "GVWR", "GCWR", "TRAILER_GCWR"}:
            return ["mass_kg"]
        return []
    if domain_key == "aero" and proposal_type in {"AERO_ABSOLUTE_CDA", "AERO_DELTA_CDA"}:
        return ["baseline_CdA"]
    if domain_key == "tire" and proposal_type == "TIRE_DB_LOOKUP":
        return ["baseline_tire_code"]
    if domain_key == "tire" and proposal_type == "TIRE_SMERF_RRC_CHANGE":
        return [
            "baseline_SMERF_optional",
            "baseline_RRC_optional",
            "front_pressure_psi",
            "rear_pressure_psi",
            "tire_load_mass_basis",
            "weight_dist_fr_pct",
        ]
    if domain_key == "transmission" and proposal_type == "UPDATE_TRANS_DRAG_ABC":
        fields = ["baseline_trans_A", "baseline_trans_B", "baseline_trans_C"]
        if str(details.get("change_mode") or "").strip() == "Absolute ABC":
            return ["baseline_component_reference_mode", *fields]
        return fields
    if domain_key == "brake" and proposal_type == "BRAKE_DRAG_CHANGE":
        if str(details.get("method") or "").strip() == "Residual torque":
            return []
        fields = ["baseline_component_A", "baseline_component_B", "baseline_component_C"]
        if str(details.get("change_mode") or "").strip() in {"Absolute", "Absolute ABC"}:
            return ["baseline_component_reference_mode", *fields]
        return fields
    if domain_key in {"axle_hubs", "parasitic"}:
        fields = ["baseline_component_A", "baseline_component_B", "baseline_component_C"]
        if str(details.get("change_mode") or "").strip() in {"Absolute", "Absolute ABC"}:
            return ["baseline_component_reference_mode", *fields]
        return fields
    return []


def _v21_is_reference_field_for_proposal(domain_key: str, proposal_type: str, field_id: str, details: dict | None = None) -> bool:
    return _v21_canonical_field_id(field_id) in set(_v21_reference_fields_for_proposal(domain_key, proposal_type, details))


def _v21_is_new_absolute_field_for_proposal(domain_key: str, proposal_type: str, field_id: str, details: dict | None = None) -> bool:
    proposal_type = str(proposal_type or "").strip().upper()
    field_id = _v21_canonical_field_id(field_id)
    details = _v21_normalize_details(details)
    if domain_key == "aero" and proposal_type == "AERO_ABSOLUTE_CDA":
        return field_id == "new_CdA"
    if domain_key == "transmission" and proposal_type == "UPDATE_TRANS_DRAG_ABC" and str(details.get("change_mode") or "").strip() == "Absolute ABC":
        return field_id in {"new_trans_A", "new_trans_B", "new_trans_C"}
    if domain_key == "brake" and proposal_type == "BRAKE_DRAG_CHANGE" and str(details.get("method") or "").strip() != "Residual torque" and str(details.get("change_mode") or "").strip() in {"Absolute", "Absolute ABC"}:
        return field_id in {"brake_A", "brake_B", "brake_C"}
    if domain_key == "axle_hubs" and proposal_type == "AXLE_HUB_DRAG_CHANGE" and str(details.get("change_mode") or "").strip() in {"Absolute", "Absolute ABC"}:
        return field_id in {"axle_hub_A", "axle_hub_B", "axle_hub_C"}
    if domain_key == "parasitic" and proposal_type == "PARASITIC_LOSS_CHANGE" and str(details.get("change_mode") or "").strip() in {"Absolute", "Absolute ABC"}:
        return field_id in {"parasitic_A", "parasitic_B", "parasitic_C"}
    return False


def _v21_reference_usage_text(column_id: str, domain_key: str, proposal_type: str, field_id: str, details: dict | None = None, state: dict | None = None) -> str:
    state = state or _v2_state()
    field_id = _v21_canonical_field_id(field_id)
    details = _v21_normalize_details(details)
    source_label = _v21_walk_from_label(column_id, state)
    manual_override = _v21_reference_override_value(column_id, domain_key, field_id, state)
    if _v21_reference_value_valid(field_id, manual_override):
        return f"Uses manual reference = {_v21_detail_display_text(manual_override, field_id)}"
    if domain_key == "mass":
        reference_value = _v21_reference_raw_value(column_id, field_id, state, include_override=False)
        if _v21_reference_value_valid(field_id, reference_value):
            return f"Uses {source_label} reference = {_v21_detail_display_text(reference_value, field_id)}"
        return "Missing reference"
    if domain_key == "tire":
        reference_value = _v21_reference_raw_value(column_id, field_id, state, include_override=False)
        if _v21_reference_value_valid(field_id, reference_value):
            return f"Uses {source_label} reference = {_v21_detail_display_text(reference_value, field_id)}"
        return "Missing reference"
    if field_id == "baseline_component_reference_mode":
        if domain_key == "aero":
            return "Reference mode not used"
        if domain_key in {"transmission", "brake", "axle_hubs", "parasitic"}:
            reference_values, reference_source, has_reference = _v21_component_reference_triplet(domain_key, column_id, details, state)
            if reference_source == "manual_override":
                return "Manual reference override"
            if reference_source == "assume_zero":
                return "Assume zero reference"
            if has_reference:
                return f"Uses {source_label} reference"
            return "Missing reference"
    if domain_key == "aero":
        reference_value, reference_source, has_reference = _v21_aero_reference_value(column_id, details, state)
        if reference_source == "manual_override":
            return f"Uses manual reference = {_v21_detail_display_text(reference_value, field_id)}"
        if has_reference:
            return f"Uses {source_label} reference = {_v21_detail_display_text(reference_value, field_id)}"
        return "Missing reference"
    if domain_key in {"transmission", "brake", "axle_hubs", "parasitic"}:
        reference_values, reference_source, has_reference = _v21_component_reference_triplet(domain_key, column_id, details, state)
        field_map = {
            "baseline_trans_A": 0,
            "baseline_trans_B": 1,
            "baseline_trans_C": 2,
            "baseline_component_A": 0,
            "baseline_component_B": 1,
            "baseline_component_C": 2,
        }
        ref_value = reference_values[field_map[field_id]] if field_id in field_map else None
        if reference_source == "manual_override":
            return f"Uses manual reference = {_v21_detail_display_text(ref_value, field_id)}"
        if reference_source == "assume_zero":
            return "Uses assumed zero reference"
        if has_reference:
            return f"Uses {source_label} reference = {_v21_detail_display_text(ref_value, field_id)}"
        return "Missing reference"
    return f"Inherited from {source_label}"


def _v21_reference_override_visible(field_id: str, *, advanced: bool, has_reference: bool) -> bool:
    return True


def _v21_reference_status_class(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"missing", "invalid", "blocked"}:
        return "is-missing"
    if normalized in {"review", "pending", "draft"}:
        return "is-review"
    if normalized in {"inherited"}:
        return "is-inherit"
    return "is-ok"


def _v21_reference_cell_payload(
    column_id: str,
    domain_key: str,
    proposal_type: str,
    field_id: str,
    details: dict | None = None,
    state: dict | None = None,
) -> dict[str, object]:
    state = state or _v2_state()
    canonical = _v21_canonical_field_id(field_id)
    details = _v21_sync_component_reference_details(details or {})
    walk_from_label = _v21_walk_from_label(column_id, state)
    inherited_value = None
    inherited_valid = False
    override_value = _v21_reference_override_value(column_id, domain_key, canonical, state)
    if override_value in (None, ""):
        override_value = _v21_detail_value(details, canonical)
    override_valid = _v21_reference_value_valid(canonical, override_value)
    effective_value = None
    provenance = "missing"
    status = "Missing"
    inherited_label = f"DB/source from {walk_from_label}"
    inherited_note = "Inherited/source reference"
    override_required = False

    if canonical == "baseline_component_reference_mode":
        mode_value = str(override_value or details.get("baseline_component_reference_mode") or "").strip()
        flags = _v21_component_reference_mode_flags(mode_value)
        effective_value = mode_value or None
        if flags["assume_zero"]:
            provenance = "assume_zero"
            status = "Review"
        elif mode_value:
            provenance = "manual_override"
            status = "Review"
        inherited_label = "No DB/source value"
        inherited_note = "Reference handling mode for missing component ABC."
        override_required = False
        return {
            "inherited_value": None,
            "inherited_valid": False,
            "override_value": mode_value,
            "override_valid": bool(mode_value),
            "effective_value": effective_value,
            "provenance": provenance,
            "status": status,
            "inherited_label": inherited_label,
            "inherited_note": inherited_note,
            "effective_note": "Manual mode is tracked and only confirmed later in Preview & Save.",
            "override_required": override_required,
        }

    if domain_key in {"mass", "tire"}:
        inherited_value = _v21_reference_raw_value(column_id, canonical, state, include_override=False)
        inherited_valid = _v21_reference_value_valid(canonical, inherited_value)
    elif domain_key == "aero":
        inherited_value = _v21_reference_raw_value(column_id, canonical, state, include_override=False)
        inherited_valid = _v21_reference_value_valid(canonical, inherited_value)
    elif domain_key in {"transmission", "brake", "axle_hubs", "parasitic"}:
        inherited_values, has_inherited = _v21_component_inherited_baseline(domain_key, column_id, state)
        field_map = {
            "baseline_trans_A": 0,
            "baseline_trans_B": 1,
            "baseline_trans_C": 2,
            "baseline_component_A": 0,
            "baseline_component_B": 1,
            "baseline_component_C": 2,
        }
        inherited_value = inherited_values[field_map[canonical]] if canonical in field_map else None
        inherited_valid = bool(has_inherited) and _v21_reference_value_valid(canonical, inherited_value)
        flags = _v21_component_reference_mode_flags(details.get("baseline_component_reference_mode"))
        if override_valid:
            effective_value = override_value
            provenance = "manual_override"
            status = "Review"
        elif inherited_valid:
            effective_value = inherited_value
            provenance = "inherited"
            status = "OK"
        elif flags["assume_zero"] and bool(_v21_reference_field_rule(canonical).get("allow_zero", True)):
            effective_value = 0.0
            provenance = "assume_zero"
            status = "Review"
        override_required = not inherited_valid and provenance == "missing"
    else:
        inherited_value = _v21_reference_raw_value(column_id, canonical, state, include_override=False)
        inherited_valid = _v21_reference_value_valid(canonical, inherited_value)

    if effective_value is None:
        if override_valid:
            effective_value = override_value
            provenance = "manual_override"
            status = "Review"
        elif inherited_valid:
            effective_value = inherited_value
            provenance = "inherited"
            status = "OK"
        else:
            override_required = True

    provenance_label = {
        "manual_override": "Manual override",
        "inherited": inherited_label,
        "assume_zero": "Assume zero",
        "missing": "Missing",
    }.get(provenance, provenance.replace("_", " ").title())
    effective_note = f"Effective reference used by resolver: {provenance_label}."
    if provenance == "manual_override":
        effective_note = "Manual override is active. Baseline DB is unchanged until Review & Save confirms it."
    elif provenance == "missing":
        effective_note = "Resolver has no valid reference yet."
    elif provenance == "assume_zero":
        effective_note = "Resolver is using an explicit assumed-zero reference."

    return {
        "inherited_value": inherited_value,
        "inherited_valid": inherited_valid,
        "override_value": override_value,
        "override_valid": override_valid,
        "effective_value": effective_value,
        "provenance": provenance,
        "status": status,
        "inherited_label": inherited_label,
        "inherited_note": inherited_note,
        "effective_note": effective_note,
        "override_required": override_required,
    }


def _v21_detail_field_options(field_id: str, domain_key: str | None = None, proposal_type: str | None = None, context: dict | None = None) -> list[str]:
    canonical = _v21_canonical_field_id(field_id)
    proposal_type = str(proposal_type or "").strip().upper()
    if domain_key == "mass" and canonical in {"inertia_class", "reference_mass_kg", "target_mass_kg"}:
        return _v21_mass_inertia_option_values()
    for alias in _v21_detail_aliases(field_id):
        options = VDE_WORKBOOK_V21_DETAIL_SELECT_OPTIONS.get(alias)
        if options:
            return list(options)
    return []


def _v21_manual_baseline_override_allowed(domain_key: str, proposal_type: str, field_id: str, details: dict | None = None, context: dict | None = None) -> bool:
    proposal_type = str(proposal_type or "").strip().upper()
    field_id = _v21_canonical_field_id(field_id)
    context = dict(context or {})
    column_id = str(context.get("column_id") or "")
    state = context.get("state")
    if not column_id:
        return False
    if proposal_type == "AERO_ABSOLUTE_CDA" and field_id == "baseline_CdA":
        _, _, has_reference = _v21_aero_reference_value(column_id, details, state)
        return not has_reference
    if domain_key == "transmission" and proposal_type == "UPDATE_TRANS_DRAG_ABC" and str(_v21_detail_value(details, "change_mode") or "").strip() == "Absolute ABC":
        _, has_inherited = _v21_component_inherited_baseline("transmission", column_id, state)
        return not has_inherited and field_id in {"baseline_trans_A", "baseline_trans_B", "baseline_trans_C"}
    if domain_key == "brake" and proposal_type == "BRAKE_DRAG_CHANGE" and str(_v21_detail_value(details, "change_mode") or "").strip() in {"Absolute", "Absolute ABC"}:
        _, has_inherited = _v21_component_inherited_baseline("brake", column_id, state)
        return not has_inherited and field_id in {"baseline_component_A", "baseline_component_B", "baseline_component_C"}
    if domain_key == "axle_hubs" and proposal_type == "AXLE_HUB_DRAG_CHANGE" and str(_v21_detail_value(details, "change_mode") or "").strip() in {"Absolute", "Absolute ABC"}:
        _, has_inherited = _v21_component_inherited_baseline("axle_hubs", column_id, state)
        return not has_inherited and field_id in {"baseline_component_A", "baseline_component_B", "baseline_component_C"}
    if domain_key == "parasitic" and proposal_type == "PARASITIC_LOSS_CHANGE" and str(_v21_detail_value(details, "change_mode") or "").strip() in {"Absolute", "Absolute ABC"}:
        _, has_inherited = _v21_component_inherited_baseline("parasitic", column_id, state)
        return not has_inherited and field_id in {"baseline_component_A", "baseline_component_B", "baseline_component_C"}
    return False


def _v21_detail_field_editable(domain_key: str, proposal_type: str, field_id: str, details: dict | None = None, context: dict | None = None) -> bool:
    proposal_type = str(proposal_type or "").strip().upper()
    canonical = _v21_canonical_field_id(field_id)
    if _v21_is_reference_field_for_proposal(domain_key, proposal_type, canonical, details):
        return False
    if canonical.startswith("baseline_"):
        return False
    if canonical in {"change_mode", "baseline_update_requested"}:
        return False
    if canonical in {"proposal_type", "proposal_label", "status", "Cd_display", "test_mass_basis", "mass_rule_status", "mass_rule_notes", "payload_kg", "vehicle_mass_at_gcwr", "trailer_roadload_status", "inertia_class"}:
        return False
    if canonical == "test_mass_kg" and not (domain_key == "mass" and proposal_type == "CUSTOM_MASS"):
        return False
    if domain_key == "transmission" and canonical == "method":
        return False
    if domain_key == "brake" and canonical == "method" and str(proposal_type or "").strip().upper() == "BRAKE_DRAG_CHANGE":
        return False
    if proposal_type in {"AERO_DELTA_CDA"} and canonical in {"baseline_CdA", "new_CdA"}:
        return False
    if proposal_type in {"AERO_ABSOLUTE_CDA"} and canonical in {"baseline_CdA"}:
        return False
    return True


VDE_WORKBOOK_V21_DETAIL_FIELD_LABELS = {
    "proposal_type": "proposal_type",
    "proposal_label": "proposal_label",
    "status": "status",
    "mass_kg": "Base / curb mass [kg]",
    "inertia_class": "EPA ETW / TWC [kg]",
    "test_mass_low_kg": "WLTP TML [kg]",
    "test_mass_high_kg": "WLTP TMH [kg]",
    "test_mass_kg": "Resolved VDE test mass [kg]",
    "test_mass_basis": "Test mass basis",
    "gvwr_kg": "GVWR [kg]",
    "gcwr_kg": "GCWR [kg]",
    "trailer_mass_kg": "Trailer mass [kg]",
    "payload_kg": "Payload [kg]",
    "vehicle_mass_at_gcwr": "Vehicle mass at GCWR [kg]",
    "trailer_roadload_status": "Trailer roadload status",
    "mass_rule_status": "Mass rule status",
    "mass_rule_notes": "Mass rule notes",
    "baseline_CdA": "CdA reference",
    "delta_CdA": "Delta CdA",
    "new_CdA": "New CdA",
    "frontal_area_m2": "Frontal area [m^2]",
    "Af_optional": "Frontal area [m²]",
    "Cd_display": "Cd display",
    "baseline_tire_code": "Baseline tire code",
    "new_tire_code": "Tire EPA / DB Code",
    "front_pressure_psi": "Cold Tire Pressure Front [psi]",
    "rear_pressure_psi": "Cold Tire Pressure Rear [psi]",
    "hot_front_pressure_psi": "Hot Tire Pressure Front [psi]",
    "hot_rear_pressure_psi": "Hot Tire Pressure Rear [psi]",
    "tire_improvement_pct": "Tire improvement [%] - positive = lower RR",
    "weight_dist_fr_pct": "Weight distribution front [%]",
    "tire_load_mass_basis": "Tire load mass basis",
    "tire_load_mass_used_kg": "Tire load mass used [kg]",
    "final_RRC_calculated": "Final RRC calculated [N/kN]",
    "baseline_SMERF_optional": "Baseline SMERF",
    "delta_SMERF_optional": "Delta SMERF",
    "baseline_RRC_optional": "Baseline RRC",
    "delta_RRC_optional": "Delta RRC",
    "front_tire_id": "Front tire ID",
    "rear_tire_id": "Rear tire ID",
    "rrc_N_per_kN": "RRC [N/kN]",
    "smerf": "SMERF",
    "tire_calc_source": "Tire calc source",
    "tire_calc_notes": "Tire calc notes",
    "mro_kg": "MRO [kg]",
    "options_kg": "Optional equipment [kg]",
    "wltp_category": "WLTP category",
    "baseline_trans_A": "Transmission reference A",
    "baseline_trans_B": "Transmission reference B",
    "baseline_trans_C": "Transmission reference C",
    "new_trans_A": "New trans A",
    "new_trans_B": "New trans B",
    "new_trans_C": "New trans C",
    "baseline_component_reference_mode": "Baseline component reference mode",
    "baseline_component_A": "Component reference A",
    "baseline_component_B": "Component reference B",
    "baseline_component_C": "Component reference C",
    "baseline_update_requested": "Baseline update requested",
    "change_mode": "Change mode",
    "method": "Method",
}

VDE_WORKBOOK_V21_SPECIAL_DETAIL_FIELDS = ["proposal_type", "proposal_label", "status"]

VDE_WORKBOOK_V21_MASS_COMPUTED_FIELDS = {
    "EPA_STATUS": ["test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    "MASS_TWC_SHIFT": ["test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    "EPA_PLUS_1_TWC": ["test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    "PERFORMANCE_CURB_MASS": ["test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    "WLTP_MASS_LINE": ["test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    "GVWR": ["test_mass_kg", "test_mass_basis", "payload_kg", "mass_rule_status", "mass_rule_notes"],
    "GCWR": ["test_mass_kg", "test_mass_basis", "vehicle_mass_at_gcwr", "trailer_roadload_status", "mass_rule_status", "mass_rule_notes"],
    "TRAILER_GCWR": ["test_mass_kg", "test_mass_basis", "vehicle_mass_at_gcwr", "trailer_roadload_status", "mass_rule_status", "mass_rule_notes"],
    "CUSTOM_MASS": ["test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
}


def _v21_detail_field_label(field_id: str) -> str:
    canonical = _v21_canonical_field_id(field_id)
    return str(VDE_WORKBOOK_V21_DETAIL_FIELD_LABELS.get(canonical) or VDE_WORKBOOK_V21_DETAIL_FIELD_LABELS.get(field_id) or canonical)


VDE_WORKBOOK_V21_COMPACT_FIELDS = {
    ("mass", "EPA_STATUS"): ["mass_kg", "inertia_class", "test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    ("mass", "MASS_TWC_SHIFT"): ["mass_kg", "inertia_class", "shift_steps", "target_side", "target_mass_kg", "test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    ("mass", "EPA_PLUS_1_TWC"): ["mass_kg", "inertia_class", "shift_steps", "target_side", "target_mass_kg", "test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    ("mass", "PERFORMANCE_CURB_MASS"): ["mass_kg", "preset", "custom_delta_kg", "test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    ("mass", "WLTP_MASS_LINE"): ["line_type", "test_mass_low_kg", "test_mass_high_kg", "test_mass_kg", "test_mass_basis", "mass_rule_status"],
    ("mass", "GVWR"): ["mass_kg", "gvwr_kg", "payload_kg", "test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    ("mass", "GCWR"): ["mass_kg", "gvwr_kg", "gcwr_kg", "trailer_mass_kg", "vehicle_mass_at_gcwr", "test_mass_kg", "test_mass_basis", "trailer_roadload_source", "trailer_code", "trailer_A", "trailer_B", "trailer_C", "mass_rule_status", "mass_rule_notes"],
    ("mass", "TRAILER_GCWR"): ["mass_kg", "gvwr_kg", "gcwr_kg", "trailer_mass_kg", "vehicle_mass_at_gcwr", "test_mass_kg", "test_mass_basis", "trailer_roadload_source", "trailer_code", "trailer_A", "trailer_B", "trailer_C", "mass_rule_status", "mass_rule_notes"],
    ("mass", "CUSTOM_MASS"): ["test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes"],
    ("aero", "AERO_DELTA_CDA"): ["baseline_CdA", "delta_CdA", "new_CdA", "source"],
    ("aero", "AERO_ABSOLUTE_CDA"): ["new_CdA", "baseline_CdA", "delta_CdA", "frontal_area_m2", "Cd_display"],
    ("tire", "TIRE_DB_LOOKUP"): ["new_tire_code", "front_pressure_psi", "rear_pressure_psi", "hot_front_pressure_psi", "hot_rear_pressure_psi", "tire_improvement_pct", "weight_dist_fr_pct"],
    ("tire", "TIRE_SMERF_RRC_CHANGE"): ["baseline_SMERF_optional", "delta_SMERF_optional", "baseline_RRC_optional", "delta_RRC_optional", "front_pressure_psi", "rear_pressure_psi", "hot_front_pressure_psi", "hot_rear_pressure_psi", "tire_load_mass_basis", "tire_load_mass_used_kg", "weight_dist_fr_pct", "final_RRC_calculated", "source"],
    ("transmission", "UPDATE_TRANS_DRAG_ABC"): ["change_mode", "baseline_component_reference_mode", "baseline_trans_A", "baseline_trans_B", "baseline_trans_C", "new_trans_A", "new_trans_B", "new_trans_C", "delta_A", "delta_B", "delta_C", "neutral_drag_source"],
    ("transmission", "TRANS_LOSS_PCT"): ["loss_pct", "percent_basis", "rule_version"],
    ("brake", "BRAKE_DRAG_CHANGE"): ["method", "change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "delta_A", "delta_B", "delta_C", "brake_A", "brake_B", "brake_C", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m", "brake_drag_force_N"],
    ("axle_hubs", "AXLE_HUB_DRAG_CHANGE"): ["change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "delta_A", "delta_B", "delta_C", "axle_hub_A", "axle_hub_B", "axle_hub_C"],
    ("parasitic", "PARASITIC_LOSS_CHANGE"): ["change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "delta_A", "delta_B", "delta_C", "parasitic_A", "parasitic_B", "parasitic_C"],
}

VDE_WORKBOOK_V21_ADVANCED_FIELDS = {
    ("mass", "EPA_STATUS"): ["source", "notes"],
    ("mass", "MASS_TWC_SHIFT"): ["source", "notes"],
    ("mass", "EPA_PLUS_1_TWC"): ["source", "notes"],
    ("mass", "PERFORMANCE_CURB_MASS"): ["source", "notes"],
    ("mass", "WLTP_MASS_LINE"): ["optional_weight_kg", "laden_mass_kg", "wltp_mass_pair_id", "mro_kg", "options_kg", "wltp_category", "source", "notes"],
    ("mass", "GVWR"): ["source", "notes"],
    ("mass", "GCWR"): ["source", "notes", "trailer_calc_notes"],
    ("mass", "TRAILER_GCWR"): ["source", "notes", "trailer_calc_notes"],
    ("mass", "CUSTOM_MASS"): ["source", "notes"],
    ("aero", "AERO_DELTA_CDA"): ["source", "notes"],
    ("aero", "AERO_ABSOLUTE_CDA"): ["source", "notes"],
    ("tire", "TIRE_DB_LOOKUP"): ["baseline_tire_code", "tire_db_id", "front_tire_id", "rear_tire_id", "tire_size", "rrc_N_per_kN", "smerf", "tire_load_mass_basis", "tire_load_mass_used_kg", "tire_calc_source", "tire_calc_notes", "source", "notes"],
    ("tire", "TIRE_SMERF_RRC_CHANGE"): ["tire_size", "tire_db_id", "front_tire_id", "rear_tire_id", "tire_calc_source", "tire_calc_notes", "notes"],
    ("transmission", "UPDATE_TRANS_DRAG_ABC"): ["baseline_trans_A", "baseline_trans_B", "baseline_trans_C", "source", "notes"],
    ("transmission", "TRANS_LOSS_PCT"): ["source", "notes"],
    ("brake", "BRAKE_DRAG_CHANGE"): ["source", "notes"],
    ("axle_hubs", "AXLE_HUB_DRAG_CHANGE"): ["source", "notes"],
    ("parasitic", "PARASITIC_LOSS_CHANGE"): ["source", "notes"],
}


def _v21_compact_fields_for_proposal(domain_key: str, proposal_type: str, details: dict | None = None, context: dict | None = None) -> list[str]:
    proposal_type = str(proposal_type or "").strip().upper()
    details = _v21_normalize_details(details)
    context = dict(context or {})
    fields = list(VDE_WORKBOOK_V21_COMPACT_FIELDS.get((domain_key, proposal_type), _v21_display_fields_for_type(domain_key, proposal_type)))
    if domain_key == "tire" and proposal_type == "TIRE_DB_LOOKUP":
        baseline_tire = context.get("baseline_tire_code")
        if baseline_tire in (None, ""):
            fields.insert(0, "baseline_tire_code")
    if domain_key == "tire" and proposal_type == "TIRE_SMERF_RRC_CHANGE":
        if _v21_detail_value(details, "delta_RRC_optional") not in (None, "") and _v21_detail_value(details, "delta_SMERF_optional") in (None, ""):
            fields = [field for field in fields if field not in {"tire_load_mass_basis", "tire_load_mass_used_kg", "weight_dist_fr_pct"}]
    if domain_key == "transmission" and proposal_type == "UPDATE_TRANS_DRAG_ABC":
        change_mode = str(_v21_detail_value(details, "change_mode") or "").strip()
        if change_mode == "Absolute ABC":
            fields = [field for field in fields if field != "baseline_component_reference_mode"]
        else:
            fields = [field for field in fields if field not in {"new_trans_A", "new_trans_B", "new_trans_C", "baseline_component_reference_mode"}]
    if domain_key == "mass" and proposal_type == "WLTP_MASS_LINE":
        line_type = str(_v21_detail_value(details, "line_type") or "").strip().upper()
        if line_type == "TMH":
            fields = [field for field in fields if field != "test_mass_low_kg"]
        else:
            fields = [field for field in fields if field != "test_mass_high_kg"]
    if domain_key == "brake" and proposal_type == "BRAKE_DRAG_CHANGE":
        method = str(_v21_detail_value(details, "method") or "").strip()
        change_mode = str(_v21_detail_value(details, "change_mode") or "").strip()
        if method == "Residual torque":
            fields = [field for field in fields if field not in {"change_mode", "baseline_component_reference_mode", "baseline_component_A", "baseline_component_B", "baseline_component_C", "delta_A", "delta_B", "delta_C", "brake_A", "brake_B", "brake_C"}]
        elif change_mode in {"Absolute", "Absolute ABC"}:
            fields = [field for field in fields if field != "baseline_component_reference_mode"]
        else:
            fields = [field for field in fields if field not in {"baseline_component_reference_mode", "brake_A", "brake_B", "brake_C", "residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m", "brake_drag_force_N"}]
    if domain_key in {"axle_hubs", "parasitic"}:
        change_mode = str(_v21_detail_value(details, "change_mode") or "").strip()
        absolute_fields = {"axle_hub_A", "axle_hub_B", "axle_hub_C", "parasitic_A", "parasitic_B", "parasitic_C"}
        if change_mode == "Absolute ABC":
            fields = [field for field in fields if field != "baseline_component_reference_mode"]
        else:
            fields = [field for field in fields if field not in {"baseline_component_reference_mode", *absolute_fields}]
    return list(dict.fromkeys(fields))


def _v21_advanced_fields_for_proposal(domain_key: str, proposal_type: str, details: dict | None = None, context: dict | None = None) -> list[str]:
    proposal_type = str(proposal_type or "").strip().upper()
    details = _v21_normalize_details(details)
    context = dict(context or {})
    fields = list(VDE_WORKBOOK_V21_ADVANCED_FIELDS.get((domain_key, proposal_type), []))
    if domain_key == "tire" and proposal_type == "TIRE_DB_LOOKUP":
        baseline_tire = context.get("baseline_tire_code")
        if baseline_tire not in (None, ""):
            fields = ["baseline_tire_code", *fields]
    return list(dict.fromkeys(fields))


def _v21_should_show_field(domain_key: str, proposal_type: str, field_id: str, details: dict | None = None, context: dict | None = None, advanced: bool = False) -> bool:
    proposal_type = str(proposal_type or "").strip().upper()
    field_id = _v21_canonical_field_id(field_id)
    if field_id in VDE_WORKBOOK_V21_SPECIAL_DETAIL_FIELDS:
        return True
    compact_fields = _v21_compact_fields_for_proposal(domain_key, proposal_type, details, context)
    advanced_fields = _v21_advanced_fields_for_proposal(domain_key, proposal_type, details, context)
    return field_id in compact_fields or (advanced and field_id in advanced_fields)


def _v21_field_display_state(domain_key: str, proposal_type: str, field_id: str, details: dict | None = None, context: dict | None = None, *, advanced: bool = False) -> str:
    proposal_type = str(proposal_type or "").strip().upper()
    field_id = _v21_canonical_field_id(field_id)
    details = _v21_normalize_details(details)
    context = dict(context or {})
    if not _v21_is_field_used(domain_key, proposal_type, field_id, details, context, advanced=advanced):
        return "not_used"
    calculated_fields = {"test_mass_basis", "mass_rule_status", "mass_rule_notes", "payload_kg", "vehicle_mass_at_gcwr", "trailer_roadload_status", "Cd_display", "final_RRC_calculated", "brake_drag_force_N"}
    if field_id == "test_mass_kg" and not _v21_detail_field_editable(domain_key, proposal_type, field_id, details, context):
        return "calculated"
    if field_id == "new_CdA" and proposal_type == "AERO_DELTA_CDA":
        return "calculated"
    if field_id == "delta_CdA" and proposal_type == "AERO_ABSOLUTE_CDA":
        return "calculated"
    if domain_key in {"transmission", "brake", "axle_hubs", "parasitic"} and field_id in {"delta_A", "delta_B", "delta_C"} and str(_v21_detail_value(details, "change_mode") or "").strip() == "Absolute ABC":
        return "calculated"
    if field_id in calculated_fields:
        return "calculated"
    if field_id.startswith("baseline_") and _v21_is_field_required(domain_key, proposal_type, field_id, details, context):
        return "missing"
    if field_id.startswith("baseline_") and not _v21_detail_field_editable(domain_key, proposal_type, field_id, details, context):
        return "read_only"
    if not _v21_detail_field_editable(domain_key, proposal_type, field_id, details, context):
        return "read_only"
    if _v21_is_field_required(domain_key, proposal_type, field_id, details, context) and _v21_detail_value(details, field_id) in (None, ""):
        return "missing"
    if domain_key == "tire" and proposal_type == "TIRE_SMERF_RRC_CHANGE" and field_id in {"front_pressure_psi", "rear_pressure_psi", "tire_load_mass_basis", "tire_load_mass_used_kg"} and _v21_detail_value(details, "delta_SMERF_optional") not in (None, "") and _v21_detail_value(details, field_id) in (None, "") and context.get(field_id) in (None, ""):
        return "review"
    if domain_key == "tire" and proposal_type == "TIRE_DB_LOOKUP":
        improvement = to_float(_v21_detail_value(details, "tire_improvement_pct"))
        if field_id == "tire_improvement_pct" and improvement is not None and improvement < 0:
            return "review"
    return "editable"


def _v21_active_domain_proposals(domain_key: str, state: dict | None = None) -> dict[str, dict]:
    state = state or _v2_state()
    return {
        column_id: _v21_domain_state_as_proposal(column_id, domain_key, state)
        for column_id in _v2_column_ids(state)
        if column_id != "baseline" and _v21_domain_state_as_proposal(column_id, domain_key, state)
    }


def _v21_detail_fields_for_domain(domain_key: str, active_proposals: dict[str, dict], *, advanced: bool = False, state: dict | None = None) -> list[str]:
    ordered: list[str] = list(VDE_WORKBOOK_V21_SPECIAL_DETAIL_FIELDS)
    seen = set(ordered)
    state = state or _v2_state()
    for column_id, proposal in active_proposals.items():
        proposal_type = str((proposal or {}).get("proposal_type") or (proposal or {}).get("type") or "").strip().upper()
        details = dict((proposal or {}).get("details") or {})
        context = {"baseline_tire_code": _v21_reference_raw_value(column_id, "baseline_tire_code", state), "column_id": column_id, "state": state}
        visible_fields = _v21_compact_fields_for_proposal(domain_key, proposal_type, details, context)
        if advanced:
            visible_fields += _v21_advanced_fields_for_proposal(domain_key, proposal_type, details, context)
        for field_id in visible_fields:
            if field_id not in seen:
                ordered.append(field_id)
                seen.add(field_id)
        if domain_key == "mass":
            for field_id in VDE_WORKBOOK_V21_MASS_COMPUTED_FIELDS.get(proposal_type, []):
                if field_id not in seen:
                    ordered.append(field_id)
                    seen.add(field_id)
    return ordered


def _v21_is_field_used(domain_key: str, proposal_type: str, field_id: str, details: dict | None = None, context: dict | None = None, *, advanced: bool = False) -> bool:
    proposal_type = str(proposal_type or "").strip().upper()
    field_id = _v21_canonical_field_id(field_id)
    if field_id in VDE_WORKBOOK_V21_SPECIAL_DETAIL_FIELDS:
        return True
    if domain_key == "mass" and field_id in set(VDE_WORKBOOK_V21_MASS_COMPUTED_FIELDS.get(proposal_type, [])):
        return True
    return _v21_should_show_field(domain_key, proposal_type, field_id, details, context, advanced=advanced)


def _v21_detail_widget_key(domain_key: str, column_id: str, field_id: str, proposal_id: str | None = None) -> str:
    proposal_token = str(proposal_id or "no_prop").strip() or "no_prop"
    return f"v21_detail__{domain_key}__{column_id}__{proposal_token}__{_v21_canonical_field_id(field_id)}"


def _v21_widget_default(key: str, value: object) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _v21_clean_detail_value(value):
    if value in (None, "", "inherit"):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


def _v21_is_field_required(domain_key: str, proposal_type: str, field_id: str, details: dict | None = None, context: dict | None = None) -> bool:
    proposal_type = str(proposal_type or "").strip().upper()
    field_id = _v21_canonical_field_id(field_id)
    details = _v21_normalize_details(details)
    context = dict(context or {})
    required_map = {
        ("mass", "EPA_STATUS"): set(),
        ("mass", "EPA_PLUS_1_TWC"): set(),
        ("mass", "MASS_TWC_SHIFT"): set(),
        ("mass", "PERFORMANCE_CURB_MASS"): {"preset"},
        ("mass", "WLTP_MASS_LINE"): {"line_type"},
        ("mass", "GVWR"): {"gvwr_kg"},
        ("mass", "GCWR"): {"gvwr_kg", "gcwr_kg", "trailer_mass_kg"},
        ("mass", "TRAILER_GCWR"): {"gvwr_kg", "gcwr_kg", "trailer_mass_kg"},
        ("mass", "CUSTOM_MASS"): {"test_mass_kg"},
        ("aero", "AERO_DELTA_CDA"): {"delta_CdA"},
        ("aero", "AERO_ABSOLUTE_CDA"): {"new_CdA"},
        ("transmission", "TRANS_LOSS_PCT"): {"loss_pct", "percent_basis", "rule_version"},
    }
    if field_id in required_map.get((domain_key, proposal_type), set()):
        return True
    if domain_key == "aero" and proposal_type == "AERO_ABSOLUTE_CDA":
        if field_id == "new_CdA":
            return _v21_aero_absolute_value(details) is None
        reference_value, _, has_reference = _v21_aero_reference_value(str(context.get("column_id") or ""), details, context.get("state"))
        if field_id == "baseline_CdA":
            return not has_reference
        return False
    if domain_key == "mass" and proposal_type == "WLTP_MASS_LINE":
        line_type = str(_v21_detail_value(details, "line_type") or "").strip().upper()
        return field_id == ("test_mass_high_kg" if line_type == "TMH" else "test_mass_low_kg")
    if domain_key == "mass" and proposal_type in {"MASS_TWC_SHIFT", "EPA_PLUS_1_TWC"}:
        has_target_mass = _v21_detail_value(details, "target_mass_kg") not in (None, "")
        if field_id == "target_mass_kg":
            return not bool(_v21_detail_value(details, "shift_steps"))
        if field_id == "shift_steps":
            return not has_target_mass
        if field_id == "target_side":
            return bool(_v21_detail_value(details, "shift_steps")) and not has_target_mass
        return False
    if domain_key == "tire" and proposal_type == "TIRE_DB_LOOKUP":
        if field_id == "new_tire_code":
            return True
        if field_id == "baseline_tire_code":
            baseline_tire = context.get("baseline_tire_code")
            return baseline_tire in (None, "")
    if domain_key == "tire" and proposal_type == "TIRE_SMERF_RRC_CHANGE":
        if field_id in {"delta_SMERF_optional", "delta_RRC_optional"}:
            return False
        if _v21_detail_value(details, "delta_SMERF_optional") not in (None, ""):
            return field_id in {"front_pressure_psi", "rear_pressure_psi", "tire_load_mass_basis", "weight_dist_fr_pct"}
        return False
    if domain_key == "transmission" and proposal_type == "UPDATE_TRANS_DRAG_ABC":
        change_mode = str(_v21_detail_value(details, "change_mode") or "").strip()
        if change_mode == "Absolute ABC":
            reference_values, reference_source, has_reference = _v21_component_reference_triplet(domain_key, str(context.get("column_id") or ""), details, context.get("state"))
            if not has_reference:
                if field_id in {"baseline_trans_A", "baseline_trans_B", "baseline_trans_C"}:
                    return True
            return field_id in {"new_trans_A", "new_trans_B", "new_trans_C"}
        return False
    if domain_key == "brake":
        method = str(_v21_detail_value(details, "method") or "").strip()
        change_mode = str(_v21_detail_value(details, "change_mode") or "").strip()
        if method == "Residual torque":
            if field_id == "wheel_radius_m":
                return True
            has_total = _v21_detail_value(details, "residual_torque_total_Nm") not in (None, "")
            has_split = any(_v21_detail_value(details, item) not in (None, "") for item in ("residual_torque_front_Nm", "residual_torque_rear_Nm"))
            if has_total:
                return False
            return field_id in {"residual_torque_total_Nm", "residual_torque_front_Nm", "residual_torque_rear_Nm"} and not has_split
        if change_mode in {"Absolute", "Absolute ABC"}:
            reference_values, _, has_reference = _v21_component_reference_triplet(domain_key, str(context.get("column_id") or ""), details, context.get("state"))
            if not has_reference:
                if field_id in {"baseline_component_A", "baseline_component_B", "baseline_component_C"}:
                    return True
            return field_id in {"brake_A", "brake_B", "brake_C"}
        return False
    if domain_key in {"axle_hubs", "parasitic"}:
        change_mode = str(_v21_detail_value(details, "change_mode") or "").strip()
        if change_mode in {"Absolute", "Absolute ABC"}:
            reference_values, _, has_reference = _v21_component_reference_triplet(domain_key, str(context.get("column_id") or ""), details, context.get("state"))
            if not has_reference:
                if field_id in {"baseline_component_A", "baseline_component_B", "baseline_component_C"}:
                    return True
            return field_id in {
                "axle_hub_A", "axle_hub_B", "axle_hub_C",
                "parasitic_A", "parasitic_B", "parasitic_C",
            }
        return False
    return False


def _v21_reference_field_alias(field_id: str) -> str:
    field_id = _v21_canonical_field_id(field_id)
    alias_map = {
        "baseline_CdA": "CdA",
        "reference_mass_kg": "test_mass_kg",
        "baseline_tire_code": "tire_code",
        "baseline_SMERF_optional": "SMERF",
        "baseline_RRC_optional": "rrc_N_per_kN",
        "baseline_trans_A": "trans_A_loss",
        "baseline_trans_B": "trans_B_loss",
        "baseline_trans_C": "trans_C_loss",
        "test_mass_kg": "effective_test_mass_kg",
        "test_mass_basis": "vde_mass_basis",
        "gvwr_kg": "GVWR_kg",
        "gcwr_kg": "GCWR_kg",
        "trailer_mass_kg": "trailer_weight_kg",
    }
    return alias_map.get(field_id, field_id)


def _v21_reference_override_lookup_any_domain(target_column_id: str, field_id: str, state: dict | None = None):
    canonical = _v21_canonical_field_id(field_id)
    bucket = _v21_reference_override_bucket(state)
    for domain_bucket in dict(bucket.get(str(target_column_id)) or {}).values():
        values = dict(domain_bucket or {})
        if canonical in values and values.get(canonical) not in (None, ""):
            return values.get(canonical)
    return None


def _v21_reference_raw_value(column_id: str, field_id: str, state: dict | None = None, *, include_override: bool = True):
    state = state or _v2_state()
    canonical = _v21_canonical_field_id(field_id)
    alias = _v21_reference_field_alias(canonical)
    if include_override and column_id != "baseline":
        override_value = _v21_reference_override_lookup_any_domain(column_id, canonical, state)
        if _v21_reference_value_valid(canonical, override_value):
            return override_value
    reference_column_id = "baseline" if column_id == "baseline" else _v21_reference_source_column_id(column_id, state)
    effective = _v21_resolve_effective_state(reference_column_id, state)
    if canonical.startswith("baseline_") and effective.get(alias) not in (None, ""):
        candidate = effective.get(alias)
        if _v21_reference_value_valid(canonical, candidate):
            return candidate
    for candidate in _v21_detail_aliases(canonical):
        if effective.get(candidate) not in (None, ""):
            value = effective.get(candidate)
            if _v21_reference_value_valid(canonical, value):
                return value
    if effective.get(alias) not in (None, ""):
        value = effective.get(alias)
        if _v21_reference_value_valid(canonical, value):
            return value
    preview_cache = dict(state.get("preview_cache") or {})
    source_preview = dict(preview_cache.get(reference_column_id) or {})
    for candidate in _v21_detail_aliases(canonical):
        if source_preview.get(candidate) not in (None, ""):
            value = source_preview.get(candidate)
            if _v21_reference_value_valid(canonical, value):
                return value
    if source_preview.get(alias) not in (None, ""):
        value = source_preview.get(alias)
        if _v21_reference_value_valid(canonical, value):
            return value
    value = effective.get(canonical)
    return value if _v21_reference_value_valid(canonical, value) else None


def _v21_reference_override_bucket(state: dict | None = None) -> dict:
    state = state or _v2_state()
    columns = dict(state.get("columns") or {})
    baseline = dict(columns.get("baseline") or {})
    bucket = dict(baseline.get("baseline_overrides") or {})
    return bucket


def _v21_reference_override_value(
    target_column_id: str,
    domain_key: str,
    field_id: str,
    state: dict | None = None,
):
    bucket = _v21_reference_override_bucket(state)
    direct_value = (
        dict(dict(bucket.get(str(target_column_id)) or {}).get(domain_key) or {}).get(_v21_canonical_field_id(field_id))
    )
    if direct_value not in (None, ""):
        return direct_value
    return _v21_baseline_printed_override_value(field_id, domain_key, state)


def _v21_set_reference_override_values(
    target_column_id: str,
    domain_key: str,
    values: dict[str, object],
    state: dict | None = None,
) -> dict:
    state = deepcopy(state or _v2_state())
    columns = {str(key): dict(value or {}) for key, value in dict(state.get("columns") or {}).items()}
    baseline = dict(columns.get("baseline") or {})
    bucket = dict(baseline.get("baseline_overrides") or {})
    domain_bucket = dict(dict(bucket.get(str(target_column_id)) or {}).get(domain_key) or {})
    for field_id, value in dict(values or {}).items():
        canonical = _v21_canonical_field_id(field_id)
        if value in (None, ""):
            domain_bucket.pop(canonical, None)
        else:
            domain_bucket[canonical] = value
    target_bucket = dict(bucket.get(str(target_column_id)) or {})
    if domain_bucket:
        target_bucket[domain_key] = domain_bucket
        bucket[str(target_column_id)] = target_bucket
    else:
        target_bucket.pop(domain_key, None)
        if target_bucket:
            bucket[str(target_column_id)] = target_bucket
        else:
            bucket.pop(str(target_column_id), None)
    baseline["baseline_overrides"] = bucket
    columns["baseline"] = baseline
    state["columns"] = columns
    preview_cache = dict(state.get("preview_cache") or {})
    preview_cache.clear()
    state["preview_cache"] = preview_cache
    return state


def _v21_component_reference_mode_flags(reference_mode: str) -> dict[str, bool]:
    text = str(reference_mode or "").strip()
    return {
        "assume_zero": text.startswith("Assume baseline component ABC = 0"),
        "manual_reference": text.startswith("Enter manual baseline component ABC"),
        "update_baseline": text.endswith("update baseline"),
    }


def _v21_sync_component_reference_details(details: dict | None) -> dict:
    synced = _v21_normalize_details(details or {})
    reference_mode = str(synced.get("baseline_component_reference_mode") or "").strip()
    if not reference_mode:
        synced.pop("baseline_update_requested", None)
        return synced
    synced["baseline_update_requested"] = bool(_v21_component_reference_mode_flags(reference_mode)["update_baseline"])
    return synced


def _v21_component_absolute_fields(domain_key: str) -> tuple[str, str, str]:
    mapping = {
        "transmission": ("new_trans_A", "new_trans_B", "new_trans_C"),
        "brake": ("brake_A", "brake_B", "brake_C"),
        "axle_hubs": ("axle_hub_A", "axle_hub_B", "axle_hub_C"),
        "parasitic": ("parasitic_A", "parasitic_B", "parasitic_C"),
    }
    return mapping.get(domain_key, ("", "", ""))


def _v21_component_baseline_field_ids(domain_key: str) -> tuple[str, str, str]:
    mapping = {
        "transmission": ("baseline_trans_A", "baseline_trans_B", "baseline_trans_C"),
        "brake": ("baseline_component_A", "baseline_component_B", "baseline_component_C"),
        "axle_hubs": ("baseline_component_A", "baseline_component_B", "baseline_component_C"),
        "parasitic": ("baseline_component_A", "baseline_component_B", "baseline_component_C"),
    }
    return mapping.get(domain_key, ("", "", ""))


def _v21_reference_source_column_id(column_id: str, state: dict | None = None) -> str:
    state = state or _v2_state()
    if column_id == "baseline":
        return "baseline"
    columns = dict(state.get("columns") or {})
    source_id = str(dict(columns.get(column_id) or {}).get("walk_from") or "baseline")
    if source_id not in _v2_column_ids(state):
        source_id = "baseline"
    return source_id


def _v21_resolve_effective_state(column_id: str, state: dict | None = None, *, _stack: tuple[str, ...] = ()) -> dict:
    if state is None:
        return deepcopy(_v2_effective_state(column_id, _stack=_stack))

    columns = dict(state.get("columns") or {})
    valid_columns = set(_v2_column_ids(state))
    spec_map = _v2_field_spec_map()
    metadata = _v2_metadata_effective(state)
    if column_id == "baseline":
        baseline = dict(columns.get("baseline") or {})
        if str(baseline.get("line_source") or "Existing VDE DB") == "Existing VDE DB":
            row = next((item for item in state.get("rows") or [] if int(item.get("id")) == int(baseline.get("selected_vde_id") or 0)), None)
            effective = _v2_row_to_effective_state(row or {})
        else:
            effective = {
                "line_source": "New test ABC_TOTAL",
                "vde_id": "New / Insert",
                "baseline_selector": "New test ABC_TOTAL",
                "walk_from": "",
                "proposal_direct": "",
                "proposal_effective": "",
            }
        effective["line_source"] = str(metadata.get("line_source") or effective.get("line_source") or "Existing VDE DB")
        effective["baseline_selector"] = str(metadata.get("selected_baseline_label") or effective.get("baseline_selector") or "")
        for field_id in ("legislation", "model_year", "make", "model", "cycle"):
            if metadata.get(field_id) not in (None, "", 0):
                effective[field_id] = metadata.get(field_id)
        effective["description"] = str(metadata.get("description") or effective.get("description") or "")
        for field_id, raw_value in dict(baseline.get("direct") or {}).items():
            spec = spec_map.get(str(field_id), {"kind": "text"})
            parsed = _v2_parse_value(raw_value, str(spec.get("kind") or "text"))
            effective[field_id] = parsed if parsed is not None else ""
        effective["description"] = str(effective.get("description") or "")
        effective["proposal_effective"] = str(effective.get("proposal_direct") or effective.get("proposal_effective") or "").strip()
        if str(metadata.get("description") or "").strip() and not str(effective.get("proposal_direct") or "").strip():
            effective["proposal_effective"] = str(metadata.get("description") or "").strip()
        effective["save_target"] = "Selected" if str(state.get("save_target") or "") == column_id else ""
        effective["scenario_notes"] = str(dict(baseline.get("direct") or {}).get("scenario_notes") or "")
        effective = _v21_apply_printed_overrides_to_effective(effective, state)
        _v21_apply_proposals_to_effective(effective, column_id, state)
        _v2_apply_trailer_preset(effective)
        _v2_apply_mass_intention(effective)
        return effective

    if column_id in _stack:
        return _v21_resolve_effective_state("baseline", state, _stack=_stack)

    column = dict(columns.get(column_id) or {})
    source_id = str(column.get("walk_from") or "baseline")
    if source_id not in valid_columns:
        source_id = "baseline"
    source_state = _v21_resolve_effective_state(source_id, state, _stack=_stack + (column_id,))
    effective = deepcopy(source_state)
    direct = dict(column.get("direct") or {})
    for field_id, raw_value in direct.items():
        if raw_value in (None, ""):
            continue
        spec = spec_map.get(str(field_id), {"kind": "text"})
        parsed = _v2_parse_value(raw_value, str(spec.get("kind") or "text"))
        if parsed is not None:
            effective[field_id] = parsed
    direct_proposal = str(direct.get("proposal_direct") or "").strip()
    inherited_proposal = str(source_state.get("proposal_effective") or "").strip()
    effective["proposal_direct"] = direct_proposal
    effective["proposal_effective"] = " + ".join(part for part in [inherited_proposal, direct_proposal] if part)
    effective["walk_from"] = source_id
    effective["vde_id"] = "New / Insert"
    effective["line_source"] = str(direct.get("line_source") or "New / Insert")
    effective["save_target"] = "Selected" if str(state.get("save_target") or "") == column_id else ""
    effective["scenario_notes"] = str(direct.get("scenario_notes") or "")
    _v21_apply_proposals_to_effective(effective, column_id, state)
    _v2_apply_trailer_preset(effective)
    _v2_apply_mass_intention(effective, inherited=source_state)
    return effective


def _v21_has_explicit_domain_reference(domain_key: str, source_column_id: str, state: dict | None = None) -> bool:
    state = state or _v2_state()
    if source_column_id not in _v2_column_ids(state):
        return False
    columns = dict(state.get("columns") or {})
    if source_column_id == "baseline":
        baseline = dict(columns.get("baseline") or {})
        if baseline.get("selected_vde_id") not in (None, "", 0):
            return True
        direct = dict(baseline.get("direct") or {})
        baseline_fields = {
            "aero": {"CdA"},
            "transmission": {"trans_A_loss", "trans_B_loss", "trans_C_loss"},
            "brake": {"brake_A", "brake_B", "brake_C"},
            "axle_hubs": {"axle_hub_A", "axle_hub_B", "axle_hub_C"},
            "parasitic": {"parasitic_A", "parasitic_B", "parasitic_C"},
        }.get(domain_key, set())
        return any(direct.get(field_id) not in (None, "", "inherit") for field_id in baseline_fields)
    proposal = _v21_get_direct_proposal(source_column_id, domain_key, state)
    if proposal:
        status = _v21_resolved_proposal_status(source_column_id, domain_key, proposal, state)
        if status not in {"Missing", "Invalid", "Blocked", "Draft", "Inherited"}:
            return True
    parent_source = _v21_reference_source_column_id(source_column_id, state)
    if parent_source == source_column_id:
        return False
    return _v21_has_explicit_domain_reference(domain_key, parent_source, state)


def _v21_component_inherited_baseline(domain_key: str, column_id: str, state: dict | None = None) -> tuple[tuple[object, object, object], bool]:
    state = state or _v2_state()
    source_column_id = _v21_reference_source_column_id(column_id, state)
    source_context = _v21_detail_source_context(column_id, state)
    field_ids = {
        "transmission": ("trans_A_loss", "trans_B_loss", "trans_C_loss"),
        "brake": ("brake_A", "brake_B", "brake_C"),
        "axle_hubs": ("axle_hub_A", "axle_hub_B", "axle_hub_C"),
        "parasitic": ("parasitic_A", "parasitic_B", "parasitic_C"),
    }.get(domain_key, ("", "", ""))
    values = tuple(source_context.get(field_id) for field_id in field_ids)
    has_reference = _v21_has_explicit_domain_reference(domain_key, source_column_id, state) and any(to_float(item) is not None for item in values)
    return values, has_reference


def _v21_aero_reference_value(column_id: str, details: dict | None, state: dict | None = None) -> tuple[object, str, bool]:
    state = state or _v2_state()
    details = _v21_normalize_details(details or {})
    source_column_id = _v21_reference_source_column_id(column_id, state)
    inherited_value = to_float(_v21_reference_raw_value(column_id, "baseline_CdA", state))
    manual_value = to_float(
        _v21_reference_override_value(column_id, "aero", "baseline_CdA", state),
        to_float(details.get("baseline_CdA")),
    )
    resolved = resolve_v21_reference_value(
        inherited_value if _v21_has_explicit_domain_reference("aero", source_column_id, state) else None,
        manual_value=manual_value,
        assume_zero=False,
    )
    return resolved["value"], str(resolved["source"]), bool(resolved["has_reference"])


def _v21_aero_absolute_value(details: dict | None) -> float | None:
    details = _v21_normalize_details(details or {})
    direct_value = to_float(details.get("new_CdA"))
    if direct_value is not None:
        return direct_value
    cd = to_float(details.get("Cd"))
    area = to_float(details.get("frontal_area_m2"))
    if cd is None or area is None:
        return None
    return float(cd * area)


def _v21_component_reference_triplet(domain_key: str, column_id: str, details: dict | None, state: dict | None = None) -> tuple[tuple[object, object, object], str, bool]:
    details = _v21_sync_component_reference_details(details or {})
    state = state or _v2_state()
    inherited_values, has_inherited = _v21_component_inherited_baseline(domain_key, column_id, state)
    manual_fields = _v21_component_baseline_field_ids(domain_key)
    manual_values = tuple(
        _v21_reference_override_value(column_id, domain_key, field_id, state)
        if _v21_reference_override_value(column_id, domain_key, field_id, state) not in (None, "")
        else details.get(field_id)
        for field_id in manual_fields
    )
    reference_flags = _v21_component_reference_mode_flags(details.get("baseline_component_reference_mode"))
    resolved = resolve_v21_reference_triplet(
        inherited_values if has_inherited else (None, None, None),
        manual_values=manual_values,
        assume_zero=bool(reference_flags["assume_zero"]),
    )
    return tuple(resolved["values"]), str(resolved["source"]), bool(resolved["has_reference"])


def _v21_component_delta_from_absolute(domain_key: str, column_id: str, details: dict | None, state: dict | None = None) -> dict[str, float] | None:
    details = _v21_normalize_details(details or {})
    absolute_fields = _v21_component_absolute_fields(domain_key)
    absolute_values = [to_float(details.get(field_id)) for field_id in absolute_fields]
    if any(value is None for value in absolute_values):
        return None
    baseline_values, _, has_reference = _v21_component_reference_triplet(domain_key, column_id, details, state)
    baseline_numeric = [to_float(value) for value in baseline_values]
    if not has_reference or any(value is None for value in baseline_numeric):
        return None
    delta_state = resolve_v21_delta_triplet(
        new_values=absolute_values,
        reference_values=baseline_numeric,
    )
    return {
        "delta_A": delta_state["local_delta"][0],
        "delta_B": delta_state["local_delta"][1],
        "delta_C": delta_state["local_delta"][2],
    }


def _v21_detail_source_context(column_id: str, state: dict | None = None) -> dict:
    state = state or _v2_state()
    source_id = "baseline" if column_id == "baseline" else _v21_reference_source_column_id(column_id, state)
    return deepcopy(_v21_resolve_effective_state(source_id, state))


def _v21_effective_state_for_proposal(column_id: str, domain_key: str, proposal_type: str, details: dict, state: dict | None = None) -> dict:
    state = state or _v2_state()
    inherited = _v21_detail_source_context(column_id, state)
    effective = deepcopy(inherited)
    details = _v21_normalize_details(details)
    proposal = {
        "domain": domain_key,
        "proposal_type": proposal_type,
        "type": proposal_type,
        "details": dict(details or {}),
    }
    direct_fields = _v21_direct_fields_from_proposal(domain_key, proposal, column_id, state)
    if domain_key == "mass":
        for field_id in _v21_reference_fields_for_proposal(domain_key, proposal_type, details):
            override_value = _v21_reference_override_value(column_id, domain_key, field_id, state)
            inherited_value = _v21_reference_raw_value(column_id, field_id, state, include_override=False)
            source_value = override_value if _v21_reference_value_valid(field_id, override_value) else inherited_value
            if not _v21_reference_value_valid(field_id, source_value):
                continue
            parsed = _v2_parse_value(source_value, str((_v2_field_spec_map().get(field_id) or {}).get("kind") or "text"))
            if parsed in (None, "", "inherit"):
                continue
            effective[field_id] = parsed
            if field_id == "mass_kg":
                effective["curb_mass_kg"] = parsed
    if domain_key == "aero":
        current_effective = deepcopy(effective)
        working_type = str(proposal_type or "").strip().upper()
        if working_type == "AERO_ABSOLUTE_CDA":
            _, _, has_reference = _v21_aero_reference_value(column_id, details, state)
            direct_fields.pop("CdA", None)
            if has_reference:
                absolute_cda = _v21_aero_absolute_value(details)
                if absolute_cda is not None:
                    direct_fields["CdA"] = absolute_cda
        elif working_type == "AERO_DELTA_CDA":
            inherited_cda = to_float(current_effective.get("CdA"))
            delta_cda = to_float(details.get("delta_CdA"))
            if inherited_cda is not None and delta_cda is not None:
                direct_fields["CdA"] = inherited_cda + delta_cda
    for field_id, value in direct_fields.items():
        parsed = _v2_parse_value(value, str((_v2_field_spec_map().get(field_id) or {}).get("kind") or "text"))
        if parsed not in (None, "", "inherit"):
            effective[field_id] = parsed
            if field_id == "mass_kg":
                effective["curb_mass_kg"] = parsed
    if domain_key == "mass":
        _v2_apply_trailer_preset(effective)
        _v2_apply_mass_intention(effective, inherited=inherited)
    return effective


def _v21_has_reference_override_for_proposal(column_id: str, domain_key: str, proposal_type: str, details: dict | None = None, state: dict | None = None) -> bool:
    state = state or _v2_state()
    for field_id in _v21_reference_fields_for_proposal(domain_key, proposal_type, details):
        if field_id == "baseline_component_reference_mode":
            continue
        override_value = _v21_reference_override_value(column_id, domain_key, field_id, state)
        if _v21_reference_value_valid(field_id, override_value):
            return True
    return False


def _v21_validate_proposal_details(column_id: str, domain_key: str, proposal_type: str, details: dict, state: dict | None = None) -> tuple[str, list[str], list[str], dict]:
    state = state or _v2_state()
    proposal_type = str(proposal_type or "INHERIT").strip().upper()
    details = {key: value for key, value in _v21_sync_component_reference_details(details or {}).items() if value not in (None, "")}
    warnings: list[str] = []
    missing_fields: list[str] = []
    reference_override_active = _v21_has_reference_override_for_proposal(column_id, domain_key, proposal_type, details, state)
    if proposal_type == "INHERIT":
        return "Inherited", warnings, missing_fields, {}
    if domain_key == "mass":
        effective = _v21_effective_state_for_proposal(column_id, domain_key, proposal_type, details, state)
        status = str(effective.get("mass_rule_status") or "Draft")
        note = str(effective.get("mass_rule_notes") or "").strip()
        if note:
            warnings.append(note)
        if reference_override_active and status not in {"Missing", "Invalid", "Blocked"}:
            status = "Review"
            warnings.append("Manual reference override in Baseline/reference.")
        return status, warnings, missing_fields, effective
    if domain_key == "aero":
        effective = _v21_effective_state_for_proposal(column_id, domain_key, proposal_type, details, state)
        if proposal_type == "AERO_DELTA_CDA":
            if details.get("delta_CdA") in (None, ""):
                return "Missing", warnings, ["delta_CdA"], effective
            return "OK", warnings, missing_fields, effective
        absolute_cda = _v21_aero_absolute_value(details)
        if absolute_cda is None:
            return "Missing", warnings, ["new_CdA"], effective
        _, reference_source, has_reference = _v21_aero_reference_value(column_id, details, state)
        validation = validate_v21_absolute_reference(
            details,
            new_fields=("new_CdA",),
            baseline_fields=("baseline_CdA",),
            has_reference=has_reference,
            reference_source=reference_source,
            absolute_label="Absolute CdA",
        )
        warnings.extend(validation["warnings"])
        return validation["status"], warnings, list(validation["missing_fields"]), effective
    if domain_key == "tire":
        if proposal_type == "TIRE_DB_LOOKUP":
            baseline_tire = _v21_reference_raw_value(column_id, "baseline_tire_code", state)
            required_fields = ["new_tire_code"]
            if baseline_tire in (None, ""):
                required_fields.insert(0, "baseline_tire_code")
            required = [field for field in required_fields if details.get(field) in (None, "")]
            if required:
                return "Missing", warnings, required, {}
            improvement = to_float(details.get("tire_improvement_pct"))
            if improvement is not None and improvement < 0:
                warnings.append("Negative tire improvement increases RR in EcoDrive convention.")
                return "Review", warnings, missing_fields, {}
            if reference_override_active:
                warnings.append("Manual reference override in Baseline/reference.")
                return "Review", warnings, missing_fields, {}
            return "OK", warnings, missing_fields, {}
        has_smerf_delta = details.get("delta_SMERF_optional") not in (None, "")
        has_rrc_delta = details.get("delta_RRC_optional") not in (None, "")
        if not has_smerf_delta and not has_rrc_delta:
            return "Missing", warnings, ["delta_SMERF_optional", "delta_RRC_optional"], {}
        if has_rrc_delta and not has_smerf_delta:
            if all(details.get(field) in (None, "") for field in ("baseline_SMERF_optional", "baseline_RRC_optional")):
                warnings.append("Baseline SMERF/RRC unavailable; applying direct RRC delta as Review.")
                return "Review", warnings, missing_fields, {}
            return "OK", warnings, missing_fields, {}
        inherited_inputs = {
            "front_pressure_psi": _v21_reference_raw_value(column_id, "front_pressure_psi", state),
            "rear_pressure_psi": _v21_reference_raw_value(column_id, "rear_pressure_psi", state),
            "tire_load_mass_basis": _v21_reference_raw_value(column_id, "tire_load_mass_basis", state),
            "weight_dist_fr_pct": _v21_reference_raw_value(column_id, "weight_dist_fr_pct", state),
        }
        for field_id, inherited_value in inherited_inputs.items():
            if details.get(field_id) in (None, "") and inherited_value in (None, ""):
                missing_fields.append(field_id)
        if missing_fields:
            return "Missing", warnings, missing_fields, {}
        if all(details.get(field) in (None, "") for field in ("baseline_SMERF_optional", "baseline_RRC_optional")):
            warnings.append("Baseline SMERF/RRC unavailable; applying delta as Review.")
            return "Review", warnings, missing_fields, {}
        if reference_override_active:
            warnings.append("Manual reference override in Baseline/reference.")
            return "Review", warnings, missing_fields, {}
        return "OK", warnings, missing_fields, {}
    if domain_key == "transmission":
        if proposal_type == "UPDATE_TRANS_DRAG_ABC":
            change_mode = str(details.get("change_mode") or "").strip()
            if change_mode == "Absolute ABC":
                _, reference_source, has_reference = _v21_component_reference_triplet(domain_key, column_id, details, state)
                validation = validate_v21_absolute_reference(
                    details,
                    new_fields=("new_trans_A", "new_trans_B", "new_trans_C"),
                    baseline_fields=("baseline_trans_A", "baseline_trans_B", "baseline_trans_C"),
                    has_reference=has_reference,
                    reference_source=reference_source,
                    baseline_update_requested=bool(details.get("baseline_update_requested")),
                )
                warnings.extend(validation["warnings"])
                return validation["status"], warnings, list(validation["missing_fields"]), {}
            if not any(details.get(field) not in (None, "") for field in ("delta_A", "delta_B", "delta_C")):
                return "Missing", warnings, ["delta_A", "delta_B", "delta_C"], {}
            return "OK", warnings, missing_fields, {}
        required = [field for field in ("loss_pct", "percent_basis", "rule_version") if details.get(field) in (None, "")]
        if required:
            return "Missing", warnings, required, {}
        warnings.append("Transmission loss percent uses a rule-based conversion.")
        return "Review", warnings, missing_fields, {}
    if domain_key == "brake":
        method = str(details.get("method") or "").strip()
        if method == "Residual torque":
            has_total = details.get("residual_torque_total_Nm") not in (None, "")
            has_split = any(details.get(field) not in (None, "") for field in ("residual_torque_front_Nm", "residual_torque_rear_Nm"))
            if not has_total and not has_split:
                return "Missing", warnings, ["residual_torque_total_Nm"], {}
            if details.get("wheel_radius_m") in (None, ""):
                return "Missing", warnings, ["wheel_radius_m"], {}
            warnings.append("Residual torque conversion may still need technical review.")
            return "Review", warnings, missing_fields, {}
        change_mode = str(details.get("change_mode") or "").strip()
        if change_mode in {"Absolute", "Absolute ABC"}:
            _, reference_source, has_reference = _v21_component_reference_triplet(domain_key, column_id, details, state)
            validation = validate_v21_absolute_reference(
                details,
                new_fields=("brake_A", "brake_B", "brake_C"),
                baseline_fields=("baseline_component_A", "baseline_component_B", "baseline_component_C"),
                has_reference=has_reference,
                reference_source=reference_source,
                baseline_update_requested=bool(details.get("baseline_update_requested")),
            )
            warnings.extend(validation["warnings"])
            return validation["status"], warnings, list(validation["missing_fields"]), {}
        if not any(details.get(field) not in (None, "") for field in ("delta_A", "delta_B", "delta_C")):
            return "Missing", warnings, ["delta_A", "delta_B", "delta_C"], {}
        return "OK", warnings, missing_fields, {}
    if domain_key == "axle_hubs":
        change_mode = str(details.get("change_mode") or "").strip()
        if change_mode in {"Absolute", "Absolute ABC"}:
            _, reference_source, has_reference = _v21_component_reference_triplet(domain_key, column_id, details, state)
            validation = validate_v21_absolute_reference(
                details,
                new_fields=("axle_hub_A", "axle_hub_B", "axle_hub_C"),
                baseline_fields=("baseline_component_A", "baseline_component_B", "baseline_component_C"),
                has_reference=has_reference,
                reference_source=reference_source,
                baseline_update_requested=bool(details.get("baseline_update_requested")),
            )
            warnings.extend(validation["warnings"])
            return validation["status"], warnings, list(validation["missing_fields"]), {}
        if not any(details.get(field) not in (None, "") for field in ("delta_A", "delta_B", "delta_C")):
            return "Missing", warnings, ["delta_A", "delta_B", "delta_C"], {}
        return "OK", warnings, missing_fields, {}
    if domain_key == "parasitic":
        change_mode = str(details.get("change_mode") or "").strip()
        if change_mode in {"Absolute", "Absolute ABC"}:
            _, reference_source, has_reference = _v21_component_reference_triplet(domain_key, column_id, details, state)
            validation = validate_v21_absolute_reference(
                details,
                new_fields=("parasitic_A", "parasitic_B", "parasitic_C"),
                baseline_fields=("baseline_component_A", "baseline_component_B", "baseline_component_C"),
                has_reference=has_reference,
                reference_source=reference_source,
                baseline_update_requested=bool(details.get("baseline_update_requested")),
            )
            warnings.extend(validation["warnings"])
            return validation["status"], warnings, list(validation["missing_fields"]), {}
        if not any(details.get(field) not in (None, "") for field in ("delta_A", "delta_B", "delta_C")):
            return "Missing", warnings, ["delta_A", "delta_B", "delta_C"], {}
        return "OK", warnings, missing_fields, {}
    return "Draft", warnings, missing_fields, {}


def _v21_resolved_proposal_status(column_id: str, domain_key: str, proposal: dict | None, state: dict | None = None) -> str:
    proposal = dict(proposal or {})
    if not proposal:
        return "Inherited"
    proposal_type = str(proposal.get("proposal_type") or proposal.get("type") or "INHERIT").strip().upper()
    status, _, _, _ = _v21_validate_proposal_details(column_id, domain_key, proposal_type, dict(proposal.get("details") or {}), state)
    return status


def _v21_calculated_detail_raw_value(column_id: str, domain_key: str, proposal_type: str, field_id: str, proposal: dict, state: dict | None = None):
    state = state or _v2_state()
    field_id = _v21_canonical_field_id(field_id)
    details = _v21_sync_component_reference_details(proposal.get("details") or {})
    proposal_type = str(proposal_type or "").strip().upper()
    if domain_key == "mass" and field_id in {"test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes", "payload_kg", "vehicle_mass_at_gcwr", "trailer_roadload_status"}:
        effective = _v21_effective_state_for_proposal(column_id, domain_key, proposal_type, details, state)
        alias = {
            "test_mass_kg": "effective_test_mass_kg",
            "test_mass_basis": "vde_mass_basis",
            "payload_kg": "payload_display_kg",
        }.get(field_id, field_id)
        return effective.get(alias, effective.get(field_id))
    if domain_key in {"transmission", "brake", "axle_hubs", "parasitic"} and field_id in {"delta_A", "delta_B", "delta_C"}:
        if str(details.get("change_mode") or "").strip() == "Absolute ABC":
            computed = _v21_component_delta_from_absolute(domain_key, column_id, details, state)
            if computed:
                return computed.get(field_id)
        return None
    if field_id in {"change_mode", "method", "baseline_update_requested"}:
        return details.get(field_id)
    if field_id == "baseline_CdA":
        return _v21_reference_raw_value(column_id, field_id, state)
    if domain_key == "tire" and field_id == "final_RRC_calculated":
        baseline_rrc = to_float(details.get("baseline_RRC_optional"), to_float(_v21_reference_raw_value(column_id, "baseline_RRC_optional", state)))
        delta_rrc = to_float(details.get("delta_RRC_optional"))
        if baseline_rrc is not None and delta_rrc is not None:
            return baseline_rrc + delta_rrc
        baseline_rrc = to_float(details.get("baseline_RRC_optional"), to_float(_v21_reference_raw_value(column_id, "baseline_RRC_optional", state)))
        baseline_smerf = to_float(details.get("baseline_SMERF_optional"), to_float(_v21_reference_raw_value(column_id, "baseline_SMERF_optional", state)))
        delta_smerf = to_float(details.get("delta_SMERF_optional"))
        if baseline_rrc is not None and baseline_smerf not in (None, 0) and delta_smerf is not None:
            new_smerf = baseline_smerf + delta_smerf
            if new_smerf > 0:
                return baseline_rrc * (baseline_smerf / new_smerf)
        return None
    if proposal_type == "AERO_DELTA_CDA" and field_id == "new_CdA":
        base_cda = to_float(_v21_reference_raw_value(column_id, "baseline_CdA", state))
        delta_cda = to_float(details.get("delta_CdA"))
        if base_cda is None or delta_cda is None:
            return None
        return base_cda + delta_cda
    if proposal_type == "AERO_ABSOLUTE_CDA" and field_id == "delta_CdA":
        new_cda = _v21_aero_absolute_value(details)
        base_cda, _, has_reference = _v21_aero_reference_value(column_id, details, state)
        if new_cda is None or not has_reference or base_cda is None:
            return None
        return new_cda - base_cda
    if proposal_type == "AERO_ABSOLUTE_CDA" and field_id == "Cd_display":
        new_cda = to_float(details.get("new_CdA"))
        area = to_float(details.get("frontal_area_m2"))
        if new_cda is None or area in (None, 0):
            return None
        return new_cda / area
    return None


def _v21_detail_raw_value_for_column(column_id: str, domain_key: str, proposal_type: str, field_id: str, state: dict | None = None):
    state = state or _v2_state()
    field_id = _v21_canonical_field_id(field_id)
    if column_id == "baseline":
        return _v21_reference_raw_value("baseline", field_id, state)
    proposal = _v21_get_direct_proposal(column_id, domain_key, state)
    if proposal:
        details = _v21_normalize_details(proposal.get("details") or {})
        detail_value = _v21_detail_value(details, field_id)
        if detail_value not in (None, "", "inherit"):
            return detail_value
        calculated = _v21_calculated_detail_raw_value(column_id, domain_key, proposal_type, field_id, proposal, state)
        if calculated not in (None, ""):
            return calculated
        if field_id.startswith("baseline_"):
            return _v21_reference_raw_value(column_id, field_id, state)
        return None
    return None


def _v21_detail_display_text(value, field_id: str) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, str):
        return value
    return _v2_format_value(value, _v21_detail_field_kind(field_id)) or "-"


def _v21_render_detail_readonly_cell(text: str, host, *, class_name: str | None = None) -> None:
    display_text = str(text or "-")
    resolved_class = class_name or _v2_cell_class_name(display_text)
    host.markdown(
        f"<div class='v21-detail-readonly'><span class='v2-cell-chip {resolved_class}'>{html.escape(display_text)}</span></div>",
        unsafe_allow_html=True,
    )

def _v21_render_proposal_controls() -> None:
    state = _v2_state()
    column_labels = _v21_request_column_labels(state)
    target_options = [column_id for column_id in _v2_column_ids(state) if column_id != "baseline"]
    default_target = str(state.get("proposal_target") or _v2_last_column_id(state))
    if default_target not in target_options:
        default_target = _v2_last_column_id(state)
    st.caption("Add / Edit Proposal")
    with st.form("v21_add_proposal_form"):
        cols = st.columns([1.1, 1.2, 1.5, 1.7, 0.9])
        target = cols[0].selectbox(
            "Target column",
            target_options,
            index=target_options.index(default_target),
            format_func=lambda value: column_labels.get(value, value),
        )
        domain_options = list(VDE_WORKBOOK_V21_DOMAINS)
        domain = cols[1].selectbox(
            "Domain",
            domain_options,
            format_func=lambda value: str(VDE_WORKBOOK_V21_DOMAINS[value]["label"]).replace(" proposal", ""),
        )
        proposal_types = [item for item in VDE_WORKBOOK_V21_DOMAINS[domain]["types"] if item != "INHERIT"]
        default_type = proposal_types[0]
        proposal_type = cols[2].selectbox(
            "Proposal type",
            proposal_types,
            index=proposal_types.index(default_type),
        )
        label = cols[3].text_input("Label", value="")
        submitted = cols[4].form_submit_button("Create proposal")
    if submitted:
        state = _v2_state()
        state["proposal_target"] = target
        _v2_set_state(state)
        proposal_id = _v21_add_or_update_proposal(target, domain, proposal_type, label, {})
        st.session_state["v21_detail_domain"] = domain
        st.session_state["v21_detail_target"] = target
        proposal_badge = _v21_proposal_badge_text({"id": proposal_id, "label": label, "proposal_type": proposal_type})
        st.session_state["v21_flash_message"] = f"Created {proposal_badge} in {column_labels.get(target, target)}."
        st.rerun()


def _v21_render_selected_proposal_details(
    allowed_domains: list[str] | None = None,
    *,
    advanced_override: bool | None = None,
) -> None:
    state = _v2_state()
    column_ids = _v2_column_ids(state)
    column_labels = _v21_request_column_labels(state)
    baseline_override_enabled = _v21_baseline_override_enabled(state)
    st.session_state.pop("v21_pending_add_proposal", None)
    domain_keys = [domain_key for domain_key in (allowed_domains or list(VDE_WORKBOOK_V21_DOMAINS)) if domain_key in VDE_WORKBOOK_V21_DOMAINS]
    if not domain_keys:
        st.info("No proposal domain available for this selection.")
        return
    if len(domain_keys) > 1:
        st.caption("Editing grouped proposal details together.")
        advanced = st.checkbox(
            "Show advanced fields",
            value=bool(st.session_state.get("v21_show_advanced_fields", False)),
            key=f"v21_show_advanced_fields__{'__'.join(domain_keys)}",
        )
        st.session_state["v21_show_advanced_fields"] = advanced
        for index, domain_key in enumerate(domain_keys):
            if index:
                st.divider()
            st.markdown(
                f"**{html.escape(str(VDE_WORKBOOK_V21_DOMAINS[domain_key]['label']).replace(' proposal', ''))}**"
            )
            _v21_render_selected_proposal_details([domain_key], advanced_override=advanced)
        return
    domain = _v21_default_detail_domain(domain_keys, state)
    st.session_state["v21_detail_domain"] = domain
    selector_cols = st.columns([1.15, 4.45])
    selector_cols[0].markdown(
        f"<div class='v21-detail-head is-target'>{html.escape(str(VDE_WORKBOOK_V21_DOMAINS[domain]['label']).replace(' proposal', ''))}</div>",
        unsafe_allow_html=True,
    )
    config = VDE_WORKBOOK_V21_DOMAINS[domain]
    active_proposals = _v21_active_domain_proposals(domain, state)
    direct_columns = [column_id for column_id in column_ids if column_id != "baseline" and column_id in active_proposals]
    target = str(st.session_state.get("v21_detail_target") or state.get("proposal_target") or (direct_columns[0] if direct_columns else _v2_last_column_id(state)))
    if target not in direct_columns:
        target = direct_columns[0] if direct_columns else _v2_last_column_id(state)
    selector_cols[1].caption("Edit every direct proposal for this domain side by side. Baseline/reference follows the active proposal column automatically.")
    st.session_state["v21_detail_target"] = target
    detail_banner = [
        ("Selected Proposal Details", "is-neutral"),
        (f"Domain: {str(config['label']).replace(' proposal', '')}", "is-ok"),
        (f"Direct proposals: {len(direct_columns)}", "is-neutral"),
    ]
    if direct_columns:
        detail_banner.append((f"Baseline Override: {'Yes' if baseline_override_enabled else 'No'}", "is-review" if baseline_override_enabled else "is-neutral"))
        detail_banner.append((f"Editable: {', '.join(column_labels.get(column_id, column_id) for column_id in direct_columns)}", "is-review"))
    else:
        detail_banner.append(("No direct proposal yet", "is-neutral"))
    banner_html = "".join(
        f"<span class='v2-cell-chip {class_name}'>{html.escape(text)}</span>"
        for text, class_name in detail_banner
    )
    st.markdown(f"<div class='v21-detail-banner'>{banner_html}</div>", unsafe_allow_html=True)
    if not direct_columns:
        st.info("No direct proposal. This domain currently inherits from each column's Walk From selection.")
        return
    if advanced_override is None:
        advanced = st.checkbox(
            "Show advanced fields",
            value=bool(st.session_state.get("v21_show_advanced_fields", False)),
            key=f"v21_show_advanced_fields__{domain}",
        )
        st.session_state["v21_show_advanced_fields"] = advanced
    else:
        advanced = advanced_override

    def _proposal_widget_key(column_id: str, field_id: str) -> str:
        proposal = dict(active_proposals.get(column_id) or {})
        return _v21_detail_widget_key(domain, column_id, field_id, str(proposal.get("id") or "no_prop"))

    def _detail_context(column_id: str) -> dict[str, object]:
        return {
            "baseline_tire_code": _v21_reference_raw_value(column_id, "baseline_tire_code", state),
            "front_pressure_psi": _v21_reference_raw_value(column_id, "front_pressure_psi", state),
            "rear_pressure_psi": _v21_reference_raw_value(column_id, "rear_pressure_psi", state),
            "tire_load_mass_basis": _v21_reference_raw_value(column_id, "tire_load_mass_basis", state),
            "weight_dist_fr_pct": _v21_reference_raw_value(column_id, "weight_dist_fr_pct", state),
            "column_id": column_id,
            "state": state,
        }

    def _reference_widget_key(column_id: str, field_id: str) -> str:
        return f"v21_ref::{domain}::{column_id}::{_v21_canonical_field_id(field_id)}"

    def _baseline_widget_key(field_id: str) -> str:
        return f"v21_baseline_printed::{domain}::{_v21_canonical_field_id(field_id)}"

    def _baseline_widget_default(field_id: str) -> None:
        widget_key = _baseline_widget_key(field_id)
        current_value = _v21_baseline_printed_override_value(field_id, domain, state)
        if current_value in (None, ""):
            current_value = _v21_reference_raw_value("baseline", field_id, state, include_override=False)
        _v21_widget_default(widget_key, "" if current_value in (None, "") else str(current_value))

    def _reference_widget_default(column_id: str, field_id: str, fallback=None) -> None:
        widget_key = _reference_widget_key(column_id, field_id)
        override_value = _v21_reference_override_value(column_id, domain, field_id, state)
        value = override_value if override_value not in (None, "") else fallback
        _v21_widget_default(widget_key, "" if value in (None, "") else str(value))

    def _reference_source_snapshot(column_id: str, proposal_type: str, field_id: str, live_details: dict[str, object]) -> tuple[object, bool, str]:
        canonical = _v21_canonical_field_id(field_id)
        if canonical == "baseline_component_reference_mode":
            return None, True, f"Inherited from {_v21_walk_from_label(column_id, state)}"
        if domain in {"mass", "tire"}:
            inherited_value = _v21_reference_raw_value(column_id, canonical, state, include_override=False)
            has_reference = _v21_reference_value_valid(canonical, inherited_value)
            return inherited_value, has_reference, f"Uses {_v21_walk_from_label(column_id, state)} reference"
        if domain == "aero":
            inherited_value = _v21_reference_raw_value(column_id, canonical, state, include_override=False)
            has_reference = _v21_reference_value_valid(canonical, inherited_value)
            return inherited_value, has_reference, f"Uses {_v21_walk_from_label(column_id, state)} reference"
        if domain in {"transmission", "brake", "axle_hubs", "parasitic"}:
            reference_values, reference_source, has_reference = _v21_component_reference_triplet(domain, column_id, live_details, state)
            field_map = {
                "baseline_trans_A": 0,
                "baseline_trans_B": 1,
                "baseline_trans_C": 2,
                "baseline_component_A": 0,
                "baseline_component_B": 1,
                "baseline_component_C": 2,
            }
            reference_value = reference_values[field_map[canonical]] if canonical in field_map else None
            return reference_value, bool(has_reference), "Assume zero reference" if reference_source == "assume_zero" else f"Uses {_v21_walk_from_label(column_id, state)} reference"
        inherited_value = _v21_reference_raw_value(column_id, canonical, state, include_override=False)
        has_reference = _v21_reference_value_valid(canonical, inherited_value)
        return inherited_value, has_reference, f"Uses {_v21_walk_from_label(column_id, state)} reference"

    for column_id in direct_columns:
        proposal = dict(active_proposals.get(column_id) or {})
        normalized_details = _v21_normalize_details(proposal.get("details") or {})
        type_key = _proposal_widget_key(column_id, "proposal_type")
        label_key = _proposal_widget_key(column_id, "proposal_label")
        _v21_widget_default(type_key, str(proposal.get("proposal_type") or proposal.get("type") or _v21_domain_proposal_types(domain)[0]))
        _v21_widget_default(label_key, str(proposal.get("label") or ""))
        for field_id in set(_v21_compact_fields_for_proposal(domain, str(proposal.get("proposal_type") or proposal.get("type") or ""), normalized_details, _detail_context(column_id)) + _v21_advanced_fields_for_proposal(domain, str(proposal.get("proposal_type") or proposal.get("type") or ""), normalized_details, _detail_context(column_id))):
            if not _v21_detail_field_editable(domain, str(proposal.get("proposal_type") or proposal.get("type") or ""), field_id, normalized_details, _detail_context(column_id)):
                continue
            widget_key = _proposal_widget_key(column_id, field_id)
            existing_value = _v21_detail_value(normalized_details, field_id)
            _v21_widget_default(widget_key, "" if existing_value in (None, "") else str(existing_value))

    target_proposal = dict(active_proposals.get(target) or {})
    target_proposal_type = str(
        st.session_state.get(_proposal_widget_key(target, "proposal_type"))
        or target_proposal.get("proposal_type")
        or target_proposal.get("type")
        or _v21_domain_proposal_types(domain)[0]
    ).strip().upper()
    target_details_seed = _v21_normalize_details(target_proposal.get("details") or {})
    target_reference_fields = _v21_reference_fields_for_proposal(domain, target_proposal_type, target_details_seed)
    for field_id in target_reference_fields:
        fallback = target_details_seed.get(_v21_canonical_field_id(field_id))
        if _v21_canonical_field_id(field_id) == "baseline_component_reference_mode":
            fallback = target_details_seed.get("baseline_component_reference_mode")
        _reference_widget_default(target, field_id, fallback=fallback)

    proposal_snapshots = {
        column_id: {
            **dict(active_proposals.get(column_id) or {}),
            "proposal_type": str(st.session_state.get(_proposal_widget_key(column_id, "proposal_type")) or dict(active_proposals.get(column_id) or {}).get("proposal_type") or dict(active_proposals.get(column_id) or {}).get("type") or _v21_domain_proposal_types(domain)[0]),
        }
        for column_id in direct_columns
    }
    display_fields = _v21_detail_fields_for_domain(domain, proposal_snapshots, advanced=advanced, state=state)
    if not display_fields:
        display_fields = list(VDE_WORKBOOK_V21_SPECIAL_DETAIL_FIELDS)
    for field_id in display_fields:
        if _v21_ppe_baseline_field_editable(domain, target_proposal_type, field_id, target_details_seed, _detail_context(target), advanced=advanced):
            _baseline_widget_default(field_id)

    def _live_details(column_id: str) -> dict[str, object]:
        proposal = dict(active_proposals.get(column_id) or {})
        proposal_type = str(st.session_state.get(_proposal_widget_key(column_id, "proposal_type")) or proposal.get("proposal_type") or proposal.get("type") or _v21_domain_proposal_types(domain)[0]).strip().upper()
        details: dict[str, object] = {}
        base_details = _v21_normalize_details(proposal.get("details") or {})
        context = _detail_context(column_id)
        for field_id in display_fields:
            if field_id in VDE_WORKBOOK_V21_SPECIAL_DETAIL_FIELDS:
                continue
            if not _v21_is_field_used(domain, proposal_type, field_id, base_details, context, advanced=advanced):
                continue
            if not _v21_detail_field_editable(domain, proposal_type, field_id, base_details, context):
                continue
            widget_key = _proposal_widget_key(column_id, field_id)
            raw_value = st.session_state.get(widget_key, _v21_detail_value(base_details, field_id))
            cleaned = _v21_clean_detail_value(raw_value)
            if cleaned not in (None, ""):
                details[_v21_canonical_field_id(field_id)] = cleaned
        return _v21_sync_component_reference_details(details)

    def _target_reference_inputs(column_id: str, proposal_type: str) -> dict[str, object]:
        proposal = dict(active_proposals.get(column_id) or {})
        base_details = _v21_normalize_details(proposal.get("details") or {})
        values: dict[str, object] = {}
        reference_fields = (
            target_reference_fields
            if column_id == target and proposal_type == target_proposal_type
            else _v21_reference_fields_for_proposal(domain, proposal_type, base_details)
        )
        for field_id in reference_fields:
            widget_value = st.session_state.get(_reference_widget_key(column_id, field_id), "")
            cleaned = _v21_clean_detail_value(widget_value)
            if cleaned not in (None, ""):
                values[_v21_canonical_field_id(field_id)] = cleaned
        if "baseline_component_reference_mode" in reference_fields:
            mode = str(values.get("baseline_component_reference_mode") or base_details.get("baseline_component_reference_mode") or "").strip()
            if mode:
                values["baseline_component_reference_mode"] = mode
                values["baseline_update_requested"] = bool(_v21_component_reference_mode_flags(mode)["update_baseline"])
        return values

    def _baseline_printed_inputs() -> dict[str, object]:
        values: dict[str, object] = {}
        for field_id in display_fields:
            if not _v21_ppe_baseline_field_editable(domain, target_proposal_type, field_id, target_details_seed, _detail_context(target), advanced=advanced):
                continue
            widget_value = st.session_state.get(_baseline_widget_key(field_id), "")
            cleaned = _v21_clean_detail_value(widget_value)
            values[_v21_canonical_field_id(field_id)] = cleaned
        return values

    with st.form(f"v21_details_form_{domain}"):
        live_details_by_column = {column_id: _live_details(column_id) for column_id in direct_columns}
        target_reference_inputs = _target_reference_inputs(target, target_proposal_type)
        baseline_printed_inputs = _baseline_printed_inputs()
        if target in live_details_by_column:
            merged_target_details = dict(live_details_by_column[target])
            merged_target_details.update(target_reference_inputs)
            live_details_by_column[target] = _v21_sync_component_reference_details(merged_target_details)
        status_cache = {
            column_id: _v21_validate_proposal_details(
                column_id,
                domain,
                str(st.session_state.get(_proposal_widget_key(column_id, "proposal_type")) or proposal_snapshots[column_id].get("proposal_type") or _v21_domain_proposal_types(domain)[0]),
                live_details_by_column[column_id],
                state,
            )
            for column_id in direct_columns
        }

        header_cols = st.columns([1.65] + [1.35 for _ in column_ids] + [1.3])
        header_cols[0].markdown("<div class='v21-detail-head'>field</div>", unsafe_allow_html=True)
        for index, column_id in enumerate(column_ids, start=1):
            if column_id == "baseline":
                head_class = "is-baseline"
                head_text = column_labels.get(column_id, column_id)
            elif column_id in direct_columns:
                head_class = "is-target" if column_id == target else "is-ok"
                badge = _v21_proposal_badge_text(active_proposals.get(column_id))
                head_text = f"{column_labels.get(column_id, column_id)}<br><span class='v21-detail-field-sub'>{html.escape(badge)}</span>"
            else:
                head_class = "is-other"
                head_text = column_labels.get(column_id, column_id)
            header_cols[index].markdown(
                f"<div class='v21-detail-head {head_class}'>{head_text}</div>",
                unsafe_allow_html=True,
            )
        header_cols[-1].markdown("<div class='v21-detail-head is-notes'>notes</div>", unsafe_allow_html=True)

        for field_id in display_fields:
            row_cols = st.columns([1.65] + [1.35 for _ in column_ids] + [1.3])
            row_cols[0].markdown(
                f"<div class='v21-detail-row'><div class='v21-detail-field'>{html.escape(_v21_detail_field_label(field_id))}</div></div>",
                unsafe_allow_html=True,
            )
            target_live_details = dict(live_details_by_column.get(target) or {})
            for index, column_id in enumerate(column_ids, start=1):
                if column_id == "baseline":
                    if field_id == "proposal_type":
                        text = "Printed baseline layer"
                    elif field_id in {"proposal_label", "status"}:
                        text = "-"
                    elif _v21_canonical_field_id(field_id) == "baseline_component_reference_mode":
                        _v21_render_detail_readonly_cell("Derived from Baseline / Printed", row_cols[index], class_name="is-neutral")
                        row_cols[index].caption("Reference handling mode is derived from the selected request and Review & Save.")
                        continue
                    elif _v21_ppe_baseline_field_editable(domain, target_proposal_type, field_id, target_live_details, _detail_context(target), advanced=advanced):
                        widget_key = _baseline_widget_key(field_id)
                        options = _v21_detail_field_options(field_id, domain, target_proposal_type, _detail_context(target))
                        status_text, provenance_text = _v21_baseline_printed_status(field_id, domain, state)
                        inherited_value = _v21_reference_raw_value("baseline", field_id, state, include_override=False)
                        inherited_text = _v21_detail_display_text(inherited_value, field_id) if inherited_value not in (None, "") else "Missing"
                        current_value = st.session_state.get(widget_key, "")
                        if options:
                            select_options = [""] + options
                            if current_value and current_value not in select_options:
                                select_options.append(current_value)
                            row_cols[index].selectbox(
                                field_id,
                                select_options,
                                index=select_options.index(current_value) if current_value in select_options else 0,
                                key=widget_key,
                                label_visibility="collapsed",
                            )
                        else:
                            row_cols[index].text_input(
                                field_id,
                                key=widget_key,
                                label_visibility="collapsed",
                                placeholder="required baseline value" if inherited_value in (None, "") else "optional baseline override",
                            )
                        effective_text = _v21_detail_display_text(st.session_state.get(widget_key, "") or inherited_value, field_id)
                        _v21_render_detail_readonly_cell(
                            f"Printed: {inherited_text}",
                            row_cols[index],
                            class_name="is-inherit" if inherited_value not in (None, "") else "is-missing",
                        )
                        _v21_render_detail_readonly_cell(
                            f"Effective: {effective_text if effective_text not in {'', '-'} else 'Missing'}",
                            row_cols[index],
                            class_name=_v21_reference_status_class(status_text),
                        )
                        row_cols[index].caption(f"Status: {status_text} | Provenance: {provenance_text}")
                        row_cols[index].caption("Baseline Override edits this printed value only for the current request. DB persistence is decided later in Review & Save.")
                        continue
                    elif _v21_is_new_absolute_field_for_proposal(domain, target_proposal_type, field_id, target_live_details) or field_id in {"delta_CdA", "delta_A", "delta_B", "delta_C"}:
                        text = "-"
                    else:
                        raw_value = _v21_display_baseline_value(field_id, domain, state)
                        if raw_value in (None, "") and field_id in {"test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes", "payload_kg", "vehicle_mass_at_gcwr", "trailer_roadload_status"}:
                            baseline_alias = {"test_mass_kg": "effective_test_mass_kg", "test_mass_basis": "vde_mass_basis", "payload_kg": "payload_display_kg"}.get(field_id, field_id)
                            raw_value = _v2_effective_state("baseline").get(baseline_alias)
                        text = _v21_detail_display_text(raw_value, field_id) if raw_value not in (None, "") else "Missing"
                    _v21_render_detail_readonly_cell(text, row_cols[index])
                    continue

                proposal = dict(active_proposals.get(column_id) or {})
                if not proposal:
                    _v21_render_detail_readonly_cell(f"Inherited from {_v21_walk_from_label(column_id, state)}", row_cols[index], class_name="is-inherit")
                    continue

                proposal_type = str(st.session_state.get(_proposal_widget_key(column_id, "proposal_type")) or proposal.get("proposal_type") or proposal.get("type") or _v21_domain_proposal_types(domain)[0]).strip().upper()
                live_details = dict(live_details_by_column.get(column_id) or {})
                status_value, warnings_value, _, effective_value = status_cache.get(column_id, ("Draft", [], [], {}))
                context = _detail_context(column_id)
                display_state = _v21_field_display_state(domain, proposal_type, field_id, live_details, context, advanced=advanced)

                if field_id == "proposal_type":
                    row_cols[index].selectbox(
                        field_id,
                        _v21_domain_proposal_types(domain),
                        index=_v21_domain_proposal_types(domain).index(proposal_type) if proposal_type in _v21_domain_proposal_types(domain) else 0,
                        key=_proposal_widget_key(column_id, field_id),
                        format_func=_v21_proposal_type_label,
                        label_visibility="collapsed",
                    )
                    continue
                if field_id == "proposal_label":
                    row_cols[index].text_input(
                        field_id,
                        key=_proposal_widget_key(column_id, field_id),
                        label_visibility="collapsed",
                        placeholder="label",
                    )
                    continue
                if field_id == "status":
                    status_class = "is-ok"
                    if status_value in {"Missing", "Invalid"}:
                        status_class = "is-missing"
                    elif status_value in {"Draft", "Review"}:
                        status_class = "is-review"
                    _v21_render_detail_readonly_cell(status_value, row_cols[index], class_name=status_class)
                    continue
                if _v21_is_reference_field_for_proposal(domain, proposal_type, field_id, live_details):
                    _v21_render_detail_readonly_cell(
                        _v21_reference_usage_text(column_id, domain, proposal_type, field_id, live_details, state),
                        row_cols[index],
                        class_name="is-inherit" if "Missing" not in _v21_reference_usage_text(column_id, domain, proposal_type, field_id, live_details, state) else "is-missing",
                    )
                    continue
                if display_state == "not_used":
                    _v21_render_detail_readonly_cell("not used", row_cols[index], class_name="is-inherit")
                    continue
                if display_state in {"editable", "missing", "review"}:
                    widget_key = _proposal_widget_key(column_id, field_id)
                    options = _v21_detail_field_options(field_id, domain, proposal_type, context)
                    current_value = st.session_state.get(widget_key, "")
                    if options:
                        select_options = [""] + options
                        if current_value and current_value not in select_options:
                            select_options.append(current_value)
                        row_cols[index].selectbox(
                            field_id,
                            select_options,
                            index=select_options.index(current_value) if current_value in select_options else 0,
                            key=widget_key,
                            label_visibility="collapsed",
                        )
                    else:
                        row_cols[index].text_input(
                            field_id,
                            key=widget_key,
                            label_visibility="collapsed",
                            placeholder="enter value",
                        )
                    continue

                proposal_snapshot = {
                    "proposal_type": proposal_type,
                    "details": live_details,
                }
                raw_value = _v21_calculated_detail_raw_value(column_id, domain, proposal_type, field_id, proposal_snapshot, state)
                if raw_value in (None, "") and _v21_canonical_field_id(field_id).startswith("baseline_"):
                    raw_value = _v21_reference_raw_value(column_id, field_id, state)
                if raw_value in (None, "") and field_id in {"test_mass_kg", "test_mass_basis", "mass_rule_status", "mass_rule_notes", "payload_kg", "vehicle_mass_at_gcwr", "trailer_roadload_status"}:
                    raw_value = dict(effective_value or {}).get(field_id)
                if raw_value in (None, ""):
                    text = "calculated" if display_state == "calculated" else "unavailable"
                else:
                    text = _v21_detail_display_text(raw_value, field_id)
                readonly_class = None
                if display_state == "calculated":
                    readonly_class = "is-neutral"
                elif display_state == "missing":
                    readonly_class = "is-missing"
                elif display_state == "review":
                    readonly_class = "is-review"
                _v21_render_detail_readonly_cell(text, row_cols[index], class_name=readonly_class)

            note_text = _v21_detail_field_note(field_id) or "-"
            required_columns = []
            delta_notes = []
            for column_id in direct_columns:
                proposal_type = str(st.session_state.get(_proposal_widget_key(column_id, "proposal_type")) or proposal_snapshots[column_id].get("proposal_type") or "").strip().upper()
                context = _detail_context(column_id)
                proposal_details = live_details_by_column.get(column_id) or {}
                if _v21_is_field_used(domain, proposal_type, field_id, live_details_by_column.get(column_id) or {}, context, advanced=advanced) and _v21_is_field_required(domain, proposal_type, field_id, live_details_by_column.get(column_id) or {}, context):
                    required_columns.append(column_labels.get(column_id, column_id))
                delta_note = _v21_local_delta_note(column_id, domain, proposal_type, field_id, proposal_details, state)
                if delta_note:
                    delta_notes.append(f"{column_labels.get(column_id, column_id)}: {delta_note}")
            if required_columns:
                note_text = f"{note_text} Required for {', '.join(required_columns)}."
            if delta_notes:
                note_text = f"{note_text} {' '.join(dict.fromkeys(delta_notes))}".strip()
            row_cols[-1].markdown(
                f"<div class='v21-detail-row v21-detail-note'>{html.escape(note_text)}</div>",
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button("Apply proposal details")

    warning_messages = []
    for column_id in direct_columns:
        for warning_text in status_cache.get(column_id, ("", [], [], {}))[1]:
            warning_messages.append(f"{column_labels.get(column_id, column_id)}: {warning_text}")
    if warning_messages:
        st.warning(" | ".join(dict.fromkeys(warning_messages)))

    action_cols = st.columns([1.25, 3.75])
    if submitted:
        next_state = _v2_state()
        if baseline_printed_inputs:
            next_state = _v21_set_baseline_printed_override_values(domain, baseline_printed_inputs, next_state)
        state_for_save = next_state
        updated_count = 0
        for column_id in direct_columns:
            proposal = dict(active_proposals.get(column_id) or {})
            proposal_type = str(st.session_state.get(_proposal_widget_key(column_id, "proposal_type")) or proposal.get("proposal_type") or proposal.get("type") or _v21_domain_proposal_types(domain)[0]).strip().upper()
            label = str(st.session_state.get(_proposal_widget_key(column_id, "proposal_label")) or "").strip()
            details = dict(live_details_by_column.get(column_id) or {})
            if column_id == target:
                for field_id in target_reference_fields:
                    canonical = _v21_canonical_field_id(field_id)
                    if canonical != "baseline_component_reference_mode":
                        details.pop(canonical, None)
                if "baseline_component_reference_mode" in target_reference_fields:
                    mode = str(target_reference_inputs.get("baseline_component_reference_mode") or "").strip()
                    if mode:
                        details["baseline_component_reference_mode"] = mode
                        details["baseline_update_requested"] = bool(_v21_component_reference_mode_flags(mode)["update_baseline"])
                    else:
                        details.pop("baseline_component_reference_mode", None)
                        details.pop("baseline_update_requested", None)
            status_value, _, _, _ = _v21_validate_proposal_details(column_id, domain, proposal_type, details, state_for_save)
            _v21_add_or_update_proposal(column_id, domain, proposal_type, label, details, status=status_value)
            updated_count += 1
        st.session_state["v21_flash_message"] = f"Applied proposal details for {str(config['label']).replace(' proposal', '')} across {updated_count} column(s)."
        st.rerun()
    action_cols[1].caption("Use the proposal select matrix above to set inheritance or choose each direct proposal type. This panel only edits the active domain details.")


def _v21_render_component_detail_cards() -> list[str]:
    _v2_render_light_workbook_styles()
    state = _v2_state()
    active_group = _v21_default_component_group(state)
    _v21_set_component_group(active_group)
    st.caption("Component detail cards")
    card_cols = st.columns(len(VDE_WORKBOOK_V21_COMPONENT_GROUPS))
    for col, config in zip(card_cols, _v21_component_group_specs()):
        group_key = str(config["key"])
        status_value, detail_text, proposal_count = _v21_component_group_status(group_key, state)
        with col:
            st.markdown(
                (
                    f"<div class='v21-domain-card {'is-active' if active_group == group_key else ''}'>"
                    f"<div class='v21-domain-card-title'>{html.escape(str(config['label']))}</div>"
                    f"<div class='vde-summary-status {_v2_cell_class_name(status_value)}'>{html.escape(_status_label_for_display(status_value))}</div>"
                    f"<div class='v21-domain-card-detail'>{html.escape(detail_text if proposal_count else 'Choose this card to edit details for this component group.')}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if st.button("Active" if active_group == group_key else "Edit details", key=f"v21_component_card_{group_key}", use_container_width=True, type="primary" if active_group == group_key else "secondary"):
                _v21_set_component_group(group_key)
                st.rerun()
    return _v21_component_group_domains(active_group)


def _v21_render_scenario_input_workbook() -> None:
    flash_message = str(st.session_state.pop("v21_flash_message", "") or "").strip()
    if flash_message:
        st.success(flash_message)
    _v21_render_baseline_reference()
    _v21_render_ppe_request_actions()
    _v21_render_scenario_column_setup()
    _v21_render_input_workbook_table(_v21_proposal_summary_specs())
    allowed_domains = _v21_render_component_detail_cards()
    with st.expander("Selected proposal details", expanded=True):
        _v21_render_selected_proposal_details(allowed_domains)
    with st.expander("Advanced / Audit", expanded=False):
        st.caption("Database browsing and legacy technical diagnostics are kept here for troubleshooting.")
        if st.button("Load DB preview", key="v21_load_db_preview_button"):
            st.session_state["v2_show_db_preview"] = True
        if st.session_state.get("v2_show_db_preview"):
            _v2_render_db_browser()
        st.divider()
        _v2_render_technical_audit()


def _v21_render_proposal_summary_table() -> None:
    column_labels = _v21_request_column_labels()
    rows: list[dict] = []
    effective_row = {"field / proposal": "Proposal Effective"}
    for column_id in column_labels:
        text = str(_v2_effective_state(column_id).get("proposal_effective") or "-")
        effective_row[column_labels[column_id]] = text
    effective_row["notes"] = "Accumulated request label"
    rows.append(effective_row)
    for domain_key, config in VDE_WORKBOOK_V21_DOMAINS.items():
        row = {"field / proposal": str(config["label"])}
        for column_id in column_labels:
            text = _v21_summary_text(column_id, domain_key)
            row[column_labels[column_id]] = text
        row["notes"] = "Direct proposal summary"
        rows.append(row)
    render_vde_workbook_table(
        pd.DataFrame(rows),
        title="Proposal summary",
        table_id="v21-proposal-summary",
    )


def _v21_status_rank(status: str) -> int:
    normalized = str(status or "Pending").strip().lower()
    if normalized in {"invalid", "blocked"}:
        return 5
    if normalized == "missing":
        return 4
    if normalized in {"review", "pending", "partial"}:
        return 3
    if normalized in {"ready", "ok", "defined", "derived"}:
        return 2
    if normalized in {"inherited", "inherit", "not used", "not_used", "unavailable"}:
        return 1
    return 0


def _v21_resolved_workbook_model(state: dict | None = None) -> dict:
    state = state or _v2_state()
    state = _v21_ensure_workbook_state(deepcopy(state))
    workbook_state = {
        "scenarios": deepcopy(_v2_scenarios(state)),
        "columns": deepcopy(dict(state.get("columns") or {})),
    }
    return resolve_v21_workbook_model(
        workbook_state,
        baseline_state=_v2_effective_state("baseline"),
        domain_keys=list(VDE_WORKBOOK_V21_DOMAINS),
        type_labels=VDE_WORKBOOK_V21_TYPE_LABELS,
    )


def _v21_rollup_statuses(statuses: list[tuple[str, str, str]]) -> tuple[str, str]:
    if not statuses:
        return "Pending", "No columns available"
    ordered = sorted(statuses, key=lambda item: _v21_status_rank(item[1]), reverse=True)
    worst_status = ordered[0][1]
    problem_items = [(label, status) for label, status, _ in ordered if _v21_status_rank(status) >= 3]
    if problem_items:
        detail = " | ".join(f"{label}: {_status_label_for_display(status)}" for label, status in problem_items[:3])
        return worst_status, detail
    ok_count = sum(1 for _, status, _ in statuses if _v21_status_rank(status) == 2)
    inherited_count = sum(1 for _, status, _ in statuses if _v21_status_rank(status) == 1)
    if ok_count:
        detail = f"{ok_count}/{len(statuses)} columns OK"
        if inherited_count:
            detail = f"{detail} | {inherited_count} inherited"
        return "OK", detail
    inherited_labels = [label for label, status, _ in statuses if _v21_status_rank(status) == 1]
    if inherited_labels:
        return "Inherited", ", ".join(inherited_labels[:3])
    return worst_status, " | ".join(f"{label}: {_status_label_for_display(status)}" for label, status, _ in ordered[:3])


def _v21_status_bar_payloads(state: dict | None = None) -> dict[str, tuple[str, str]]:
    state = state or _v2_state()
    return _v21_request_flow_status_payloads(state)


def _v21_request_flow_status_payloads(state: dict | None = None, session_data: dict | None = None) -> dict[str, tuple[str, str]]:
    state = state or _v2_state()
    session_data = session_data if session_data is not None else st.session_state
    proposals = dict(state.get("proposals") or {})
    proposal_count = sum(1 for domain_map in proposals.values() for proposal in dict(domain_map or {}).values() if dict(proposal or {}))
    current_hash = _v21_request_resolution_current_hash(state)
    stored_hash = str(session_data.get(V21_REQUEST_RESOLUTION_HASH_KEY) or "")
    resolution = dict(session_data.get(V21_REQUEST_RESOLUTION_STATE_KEY) or {})
    preview_stale = bool(resolution) and (bool(session_data.get(V21_REQUEST_RESOLUTION_STALE_KEY)) or stored_hash != current_hash)
    if not resolution:
        preview_status = "Not run"
        preview_detail = "Run Validate & Preview"
    elif preview_stale:
        preview_status = "Stale"
        preview_detail = f"Last preview `{stored_hash[:12] or '-'} `"
    else:
        preview_status = "Current"
        preview_detail = f"Fingerprint `{stored_hash[:12]}`"
    save_result = dict(session_data.get(V21_REQUEST_SAVE_RESULT_KEY) or {})
    save_status_raw = str(save_result.get("status") or "").strip().lower()
    if save_status_raw == "success":
        save_status = "Saved"
        save_detail = str(save_result.get("operation_id") or "Completed")
    elif save_status_raw == "partial":
        save_status = "Partial"
        save_detail = str(save_result.get("operation_id") or "Partial save")
    elif save_status_raw:
        save_status = "Failed"
        save_detail = str(save_result.get("operation_id") or "Review save result")
    else:
        save_status = "Pending"
        save_detail = "No canonical save result"
    draft_available = bool(session_data.get(V21_REQUEST_DRAFT_REPORT_BYTES_KEY))
    saved_available = bool(session_data.get(V21_REQUEST_SAVED_REPORT_BYTES_KEY))
    if saved_available:
        report_status = "Saved available"
        report_detail = str(session_data.get(V21_REQUEST_SAVED_REPORT_NAME_KEY) or "Saved report")
    elif draft_available:
        report_status = "Draft available"
        report_detail = str(session_data.get(V21_REQUEST_DRAFT_REPORT_NAME_KEY) or "Draft report")
    else:
        report_status = "Pending"
        report_detail = "No report generated"
    return {
        "Baseline": ("Loaded" if _v21_baseline_is_loaded(state) else "Pending", "Baseline reference selected" if _v21_baseline_is_loaded(state) else "Load a baseline"),
        "Request": ("Defined" if proposal_count else "Pending", f"{proposal_count} proposal(s)"),
        "Preview": (preview_status, preview_detail.strip()),
        "Save": (save_status, save_detail),
        "Report": (report_status, report_detail),
    }


def _v21_render_status_bar() -> None:
    st.caption("Request flow")
    statuses = _v21_status_bar_payloads()
    cols = st.columns(len(statuses))
    for col, (label, payload) in zip(cols, statuses.items()):
        with col:
            _render_status_bar_item(label, payload[0], payload[1])


V21_REQUEST_IMPORT_PENDING_STATE_KEY = "v21_request_import_pending_state"
V21_REQUEST_IMPORT_PENDING_SUMMARY_KEY = "v21_request_import_pending_summary"
V21_REQUEST_IMPORT_PENDING_SOURCE_KEY = "v21_request_import_pending_source"
V21_REQUEST_IMPORT_PENDING_DRAFT_KEY = "v21_request_import_pending_draft"
V21_REQUEST_IMPORT_PENDING_ERROR_KEY = "v21_request_import_pending_error"
V21_REQUEST_IMPORT_CONFIRM_BASELINE_KEY = "v21_request_import_confirm_baseline"
V21_REQUEST_RESOLUTION_STATE_KEY = "v21_request_resolution"
V21_REQUEST_RESOLUTION_HASH_KEY = "v21_request_resolution_input_hash"
V21_REQUEST_RESOLUTION_STALE_KEY = "v21_request_resolution_is_stale"
V21_REQUEST_RESOLUTION_ERROR_KEY = "v21_request_resolution_error"
V21_REQUEST_RESOLUTION_SELECTED_PROPOSAL_KEY = "v21_request_resolution_selected_proposal"
V21_REQUEST_SAVE_RESULT_KEY = "v21_request_save_result"
V21_REQUEST_SAVE_FINGERPRINT_KEY = "v21_request_save_fingerprint"
V21_REQUEST_SAVE_PROPOSALS_KEY = "v21_request_saved_proposals"
V21_REQUEST_DRAFT_REPORT_BYTES_KEY = "v21_request_draft_report_bytes"
V21_REQUEST_DRAFT_REPORT_NAME_KEY = "v21_request_draft_report_name"
V21_REQUEST_DRAFT_REPORT_FINGERPRINT_KEY = "v21_request_draft_report_fingerprint"
V21_REQUEST_SAVED_REPORT_BYTES_KEY = "v21_request_saved_report_bytes"
V21_REQUEST_SAVED_REPORT_NAME_KEY = "v21_request_saved_report_name"
V21_REQUEST_SAVED_REPORT_OPERATION_KEY = "v21_request_saved_report_operation"
V21_REQUEST_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "docs" / "templates" / "EcoDrive_VDE_PPE_Request_Input_template_v01.xlsx"


def _v21_import_summary_payload_from_validation(validation: dict, *, filename: str) -> dict:
    errors = [dict(item) for item in list(validation.get("errors") or [])]
    warnings = [dict(item) for item in list(validation.get("warnings") or [])]
    return {
        "schema_version": None,
        "template_version": None,
        "source": {"filename": filename},
        "proposal_count": 0,
        "baseline_correction_count": 0,
        "issues": [*errors, *warnings],
        "blocking_errors": errors,
        "review_issues": [],
        "warnings": warnings,
        "blocking_count": len(errors),
        "review_count": 0,
        "warning_count": len(warnings),
        "active_columns": [],
    }


def _v21_request_import_summary_rows(summary: dict | None) -> list[dict]:
    summary = dict(summary or {})
    source = dict(summary.get("source") or {})
    baseline_reference = dict(summary.get("baseline_reference") or {})
    rows = [
        {"field": "Template version", "value": str(summary.get("template_version") or "-"), "notes": "Imported workbook template"},
        {"field": "Source file", "value": str(source.get("filename") or "-"), "notes": "Uploaded Excel workbook"},
        {"field": "Proposals detected", "value": str(summary.get("proposal_count") or 0), "notes": "Active Requested columns imported"},
        {"field": "Baseline corrections", "value": str(summary.get("baseline_correction_count") or 0), "notes": "Non-blank baseline correction cells"},
        {"field": "Blocking issues", "value": str(summary.get("blocking_count") or 0), "notes": "Structural errors that block Apply"},
        {"field": "Review issues", "value": str(summary.get("review_count") or 0), "notes": "Column/domain issues preserved for review"},
        {"field": "Warnings", "value": str(summary.get("warning_count") or 0), "notes": "Non-blocking workbook warnings"},
    ]
    if baseline_reference:
        rows.extend(
            [
                {"field": "Referenced baseline", "value": str(baseline_reference.get("referenced_baseline_id") or "-"), "notes": "Baseline VDE ID read from the file"},
                {"field": "Baseline found", "value": "Yes" if baseline_reference.get("baseline_found") else "No", "notes": "Whether the referenced baseline exists in the current DB"},
                {"field": "Vehicle", "value": str(baseline_reference.get("vehicle") or "-"), "notes": "Resolved DB line for the imported baseline"},
                {"field": "Cycle", "value": str(baseline_reference.get("cycle") or "-"), "notes": "Cycle from the referenced baseline"},
                {"field": "Printed snapshot integrity", "value": str(baseline_reference.get("printed_snapshot_integrity") or "-"), "notes": str(baseline_reference.get("printed_snapshot_message") or "Printed vs DB comparison")},
            ]
        )
    return rows

def _v21_request_import_issue_rows(summary: dict | None) -> list[dict]:
    summary = dict(summary or {})
    rows: list[dict] = []
    for bucket_name, label in (
        ("blocking_errors", "Blocking"),
        ("review_issues", "Review"),
        ("warnings", "Warning"),
    ):
        for item in list(summary.get(bucket_name) or []):
            payload = dict(item or {})
            rows.append(
                {
                    "severity": label,
                    "scope": str(payload.get("scope") or payload.get("sheet") or "-"),
                    "column": str(payload.get("column_id") or payload.get("source_column") or "-"),
                    "domain": str(payload.get("domain") or "-"),
                    "code": str(payload.get("code") or "-"),
                    "message": str(payload.get("message") or "-"),
                }
            )
    return rows


def _v21_clear_import_pending_state() -> None:
    for key in (
        V21_REQUEST_IMPORT_PENDING_STATE_KEY,
        V21_REQUEST_IMPORT_PENDING_SUMMARY_KEY,
        V21_REQUEST_IMPORT_PENDING_SOURCE_KEY,
        V21_REQUEST_IMPORT_PENDING_DRAFT_KEY,
        V21_REQUEST_IMPORT_PENDING_ERROR_KEY,
        V21_REQUEST_IMPORT_CONFIRM_BASELINE_KEY,
    ):
        st.session_state.pop(key, None)


def _v21_clear_request_resolution_state() -> None:
    for key in (
        V21_REQUEST_RESOLUTION_STATE_KEY,
        V21_REQUEST_RESOLUTION_HASH_KEY,
        V21_REQUEST_RESOLUTION_STALE_KEY,
        V21_REQUEST_RESOLUTION_ERROR_KEY,
        V21_REQUEST_RESOLUTION_SELECTED_PROPOSAL_KEY,
    ):
        st.session_state.pop(key, None)
    _v21_clear_request_save_state()


def _v21_clear_request_save_state() -> None:
    for key in (
        V21_REQUEST_SAVE_RESULT_KEY,
        V21_REQUEST_SAVE_FINGERPRINT_KEY,
        V21_REQUEST_SAVE_PROPOSALS_KEY,
        V21_REQUEST_DRAFT_REPORT_BYTES_KEY,
        V21_REQUEST_DRAFT_REPORT_NAME_KEY,
        V21_REQUEST_DRAFT_REPORT_FINGERPRINT_KEY,
        V21_REQUEST_SAVED_REPORT_BYTES_KEY,
        V21_REQUEST_SAVED_REPORT_NAME_KEY,
        V21_REQUEST_SAVED_REPORT_OPERATION_KEY,
    ):
        st.session_state.pop(key, None)


def _v21_clear_request_runtime_state(*, clear_resolution: bool = True, clear_import_pending: bool = False) -> None:
    if clear_import_pending:
        _v21_clear_import_pending_state()
    if clear_resolution:
        _v21_clear_request_resolution_state()
    else:
        _v21_clear_request_save_state()


def _v21_mark_request_preview_stale() -> None:
    if st.session_state.get(V21_REQUEST_RESOLUTION_STATE_KEY):
        st.session_state[V21_REQUEST_RESOLUTION_STALE_KEY] = True
    _v21_clear_request_save_state()


def _v21_report_request_draft(state: dict | None = None) -> dict:
    state = deepcopy(dict(state or _v2_state()))
    draft = deepcopy(dict(state.get("vde_request_draft") or {}))
    if draft:
        return draft
    return build_request_equivalent_draft_from_state(state)


def _v21_build_request_report_bytes(
    state: dict | None,
    resolution: dict | None,
    *,
    save_result: dict | None = None,
) -> tuple[bytes, str]:
    report_model = build_vde_request_report_model(
        _v21_report_request_draft(state),
        deepcopy(dict(resolution or {})),
        deepcopy(dict(save_result or {})) if save_result is not None else None,
    )
    return generate_vde_request_report_xlsx(report_model), build_vde_request_report_filename(report_model)


def _v21_post_save_summary(save_result: dict | None) -> dict:
    result = dict(save_result or {})
    component_results = list(result.get("component_results") or [])
    counter = {
        "rows_saved": len(list(result.get("saved_proposals") or [])),
        "rows_skipped": len(list(result.get("skipped_proposals") or [])),
        "baseline_fields_updated": sum(len(list(dict(item or {}).get("updated_fields") or [])) for item in list(result.get("baseline_updates") or [])),
        "components_reused": sum(1 for item in component_results if str(dict(item or {}).get("status") or "") == "reused_existing"),
        "components_created": sum(1 for item in component_results if str(dict(item or {}).get("status") or "") == "created"),
        "snapshot_only_components": sum(1 for item in component_results if str(dict(item or {}).get("status") or "") == "snapshot_only"),
        "failures": len(list(result.get("issues") or [])) + sum(1 for item in component_results if str(dict(item or {}).get("status") or "") in {"component_creation_failed", "unavailable"}),
    }
    counter["operation_status"] = str(result.get("status") or "failed").title()
    return counter


def _v21_reset_request_flow(reset_ctx=None) -> None:
    state = deepcopy(dict(_v2_state() or {}))
    columns = dict(state.get("columns") or {})
    for column_id, column in list(columns.items()):
        if column_id == "baseline":
            continue
        payload = dict(column or {})
        payload["direct"] = {}
        payload["domains"] = {}
        payload["walk_from"] = payload.get("walk_from") or "baseline"
        columns[column_id] = payload
    state["columns"] = columns
    for key in (
        "proposals",
        "proposal_seq",
        "vde_request_import",
        "vde_request_import_summary",
        "vde_request_draft",
        "vde_request_source",
        "preview_cache",
        "saved_targets",
    ):
        state.pop(key, None)
    _v2_set_state(state)
    _v21_clear_request_runtime_state(clear_resolution=True, clear_import_pending=True)
    _v21_clear_incompatible_import_widget_state()
    st.session_state["v21_flash_message"] = "Started a new VDE Request draft. Baseline and database rows were preserved."
    st.rerun()


def _v21_clear_incompatible_import_widget_state() -> None:
    prefixes = (
        "v21_detail__",
        "v21_proposal_select__",
        "v21_setup_",
        "v21_show_advanced_fields__",
    )
    exact = {
        "v21_detail_target",
        "v21_detail_domain",
        "v21_component_group",
        "v21_show_advanced_fields",
        "v2_show_db_preview",
    }
    for key in list(st.session_state.keys()):
        if key in exact or any(str(key).startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)


def _v21_first_import_focus(state: dict) -> tuple[str, str] | tuple[None, None]:
    state = dict(state or {})
    walked_columns = [
        str(item.get("key") or "")
        for item in list(state.get("scenarios") or [])
        if str(item.get("role") or "") == "walked"
    ]
    proposals = dict(state.get("proposals") or {})
    for column_id in walked_columns:
        for domain_key in VDE_WORKBOOK_V21_DOMAINS:
            if dict(dict(proposals.get(column_id) or {}).get(domain_key) or {}):
                return column_id, domain_key
    if walked_columns:
        return walked_columns[0], "mass"
    return None, None


def _v21_request_preview_available(state: dict | None = None) -> bool:
    state = state or _v2_state()
    import_meta = dict(state.get("vde_request_import") or {})
    baseline_reference = dict(import_meta.get("baseline_reference") or {})
    if baseline_reference.get("association_required") and is_blank(dict(_v2_metadata_effective(state) or {}).get("selected_baseline_vde_id")):
        return False
    if dict(state.get("vde_request_import") or {}):
        return True
    for domain_map in dict(state.get("proposals") or {}).values():
        if dict(domain_map or {}):
            return True
    return False


def _v21_baseline_is_loaded(state: dict | None = None) -> bool:
    state = state or _v2_state()
    metadata = dict(_v2_metadata_effective(state) or {})
    return str(metadata.get("line_source") or "").startswith("Existing") and bool(metadata.get("selected_row"))


def _v21_current_baseline_snapshot(state: dict | None = None) -> dict:
    state = state or _v2_state()
    metadata = dict(_v2_metadata_effective(state) or {})
    selected_row = dict(metadata.get("selected_row") or {})
    effective = dict(_v2_effective_state("baseline") or {})
    snapshot: dict[str, object] = {}
    for source in (selected_row, effective, metadata):
        for key, value in dict(source or {}).items():
            if value not in (None, ""):
                snapshot[key] = value
    if metadata.get("selected_baseline_vde_id") not in (None, ""):
        snapshot["selected_baseline_vde_id"] = metadata.get("selected_baseline_vde_id")
    if metadata.get("line_source"):
        snapshot["line_source"] = metadata.get("line_source")
    return snapshot


def _v21_template_vehicle_label(snapshot: dict | None = None) -> str:
    snapshot = dict(snapshot or {})
    return "_".join(
        sanitize_request_filename_token(part, "")
        for part in (
            snapshot.get("make"),
            snapshot.get("model"),
            snapshot.get("year") or snapshot.get("model_year"),
        )
        if str(part or "").strip()
    ) or "vehicle"


def _v21_snapshot_from_vde_row(row: dict | None) -> dict:
    row = dict(row or {})
    effective = dict(_v2_row_to_effective_state(row) or {})
    metadata = _v2_row_metadata_defaults(row)
    snapshot: dict[str, object] = {}
    for source in (row, effective, metadata):
        for key, value in dict(source or {}).items():
            if value not in (None, ""):
                snapshot[key] = value
    return snapshot


def _v21_import_baseline_reference_summary(
    draft: dict,
    *,
    current_state: dict | None = None,
) -> tuple[dict, dict, dict | None, dict]:
    state = current_state or _v2_state()
    rows = list(state.get("rows") or [])
    current_metadata = dict(_v2_metadata_effective(state) or {})
    current_baseline_id = current_metadata.get("selected_baseline_vde_id")
    imported_baseline_id = extract_referenced_baseline_id(draft)
    found_row = _v2_find_row_by_id(rows, imported_baseline_id) if imported_baseline_id not in (None, "") else None
    baseline_status = resolve_imported_baseline_status(current_baseline_id, imported_baseline_id, found_row)
    db_snapshot = _v21_snapshot_from_vde_row(found_row) if found_row else {}
    printed_integrity = compare_printed_snapshot(dict(draft.get("baseline_printed") or {}), db_snapshot) if found_row else {
        "status": "Review",
        "ok": False,
        "compared_fields": 0,
        "divergent_fields": [],
        "message": "Printed snapshot integrity cannot be confirmed until a valid baseline is associated.",
    }
    summary = {
        "status": baseline_status.get("status"),
        "message": baseline_status.get("message"),
        "current_page_baseline_id": current_baseline_id,
        "referenced_baseline_id": imported_baseline_id,
        "baseline_found": bool(found_row),
        "vehicle": _v2_row_label(found_row) if found_row else "-",
        "cycle": str((found_row or {}).get("cycle_name") or "-"),
        "requires_confirmation": bool(baseline_status.get("requires_confirmation")),
        "association_required": bool(baseline_status.get("blocking")),
        "printed_snapshot_integrity": printed_integrity.get("status"),
        "printed_snapshot_message": printed_integrity.get("message"),
        "printed_snapshot_divergent_fields": deepcopy(list(printed_integrity.get("divergent_fields") or [])),
    }
    return summary, printed_integrity, found_row, db_snapshot


def _v21_request_resolution_baseline_context(state: dict | None = None) -> dict:
    state = state or _v2_state()
    baseline_context = deepcopy(dict(st.session_state.get("ctx") or {}))
    try:
        baseline_ctx = dict(_v2_state_to_ctx("baseline") or {})
    except Exception:
        baseline_ctx = {}
    baseline_effective = dict(_v2_effective_state("baseline") or {})
    metadata = dict(_v2_metadata_effective(state) or {})
    selected_row = dict(metadata.get("selected_row") or {})

    for source in (selected_row, baseline_effective, metadata, baseline_ctx):
        for key, value in dict(source or {}).items():
            if value not in (None, ""):
                baseline_context[key] = value

    if metadata.get("selected_baseline_vde_id") not in (None, ""):
        baseline_context["selected_baseline_vde_id"] = metadata.get("selected_baseline_vde_id")
    if baseline_ctx.get("cycle_df") is not None:
        baseline_context["cycle_df"] = baseline_ctx.get("cycle_df")
    elif baseline_context.get("cycle_df") is None:
        leg = str(baseline_context.get("legislation") or metadata.get("legislation") or "EPA")
        baseline_context["cycle_df"] = use_standard_cycle(leg)
    return baseline_context


def _v21_request_resolution_current_hash(state: dict | None = None) -> str:
    state = state or _v2_state()
    return build_request_resolution_fingerprint(state, _v21_request_resolution_baseline_context(state))


def _v21_resolve_request_preview(state: dict | None = None) -> tuple[dict | None, str, str]:
    state = deepcopy(dict(state or _v2_state()))
    baseline_context = _v21_request_resolution_baseline_context(state)
    current_hash = build_request_resolution_fingerprint(state, baseline_context)
    try:
        resolution = resolve_vde_request(state, baseline_context)
    except Exception as exc:
        return None, current_hash, str(exc)
    return resolution, current_hash, ""


def _v21_import_uploaded_request(upload) -> tuple[dict | None, dict | None, str | None]:
    if upload is None:
        return None, None, "Choose an Excel workbook before validating."
    temp_path = None
    try:
        suffix = Path(str(upload.name or "request.xlsx")).suffix or ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(upload.getvalue())
            temp_path = Path(handle.name)
        validation = validate_vde_request_workbook(temp_path)
        if not validation.get("ok"):
            summary = _v21_import_summary_payload_from_validation(validation, filename=str(upload.name or temp_path.name))
            return None, summary, "Workbook validation found structural issues."
        draft = parse_vde_request_workbook(temp_path)
        current_state = _v2_state()
        baseline_summary, printed_integrity, found_row, db_snapshot = _v21_import_baseline_reference_summary(draft, current_state=current_state)
        imported_printed = deepcopy(dict(draft.get("baseline_printed") or {}))
        working_draft = deepcopy(draft)
        if found_row:
            canonical_printed = build_canonical_baseline_payload(db_snapshot)
            corrections = deepcopy(dict(working_draft.get("baseline_corrections") or {}))
            field_keys = set(canonical_printed) | set(corrections)
            working_draft["baseline_printed"] = canonical_printed
            working_draft["effective_baseline"] = {
                field_key: resolve_effective_baseline(canonical_printed.get(field_key), corrections.get(field_key))
                for field_key in field_keys
            }
        import_state_seed = deepcopy(current_state)
        if found_row:
            import_state_seed = _v2_apply_selected_baseline_row(import_state_seed, found_row)
        elif baseline_summary.get("association_required"):
            import_state_seed = _v2_apply_selected_baseline_row(import_state_seed, None)
            next_metadata = dict(import_state_seed.get("metadata") or {})
            next_metadata["line_source"] = "Existing VDE DB"
            import_state_seed["metadata"] = next_metadata
        adapted_state = build_v21_workbook_state_from_request_draft(working_draft, import_state_seed)
        adapted_state["vde_request_import"]["baseline_reference"] = deepcopy(baseline_summary)
        adapted_state["vde_request_import"]["baseline_printed_file"] = imported_printed
        adapted_state["vde_request_import"]["printed_snapshot_integrity"] = deepcopy(printed_integrity)
        summary = build_v21_request_import_summary(working_draft, adapted_state=adapted_state)
        summary["baseline_reference"] = deepcopy(baseline_summary)
        summary["printed_snapshot_integrity"] = deepcopy(printed_integrity)
        review_issues = list(summary.get("review_issues") or [])
        if baseline_summary.get("status") in {"mismatch", "unresolved", "missing_reference"}:
            review_issues.append(
                {
                    "severity": "review",
                    "scope": "baseline",
                    "code": f"baseline_{baseline_summary.get('status')}",
                    "message": str(baseline_summary.get("message") or "Baseline reference needs review."),
                    "source_column": "Baseline",
                }
            )
        divergent_fields = list(printed_integrity.get("divergent_fields") or [])
        if divergent_fields:
            review_issues.append(
                {
                    "severity": "review",
                    "scope": "baseline",
                    "code": "printed_snapshot_differs",
                    "message": str(printed_integrity.get("message") or "Baseline / Printed differs from the current database baseline."),
                    "source_column": "Baseline",
                }
            )
        summary["review_issues"] = review_issues
        summary["review_count"] = len(review_issues)
        summary["issues"] = list(summary.get("issues") or []) + review_issues
        adapted_state["vde_request_import_summary"] = deepcopy(summary)
        return adapted_state, summary, None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _v21_apply_imported_request_state() -> None:
    pending_state = deepcopy(dict(st.session_state.get(V21_REQUEST_IMPORT_PENDING_STATE_KEY) or {}))
    pending_summary = dict(st.session_state.get(V21_REQUEST_IMPORT_PENDING_SUMMARY_KEY) or {})
    if not pending_state:
        st.warning("Validate an uploaded request before applying it.")
        return
    if int(to_float(pending_summary.get("blocking_count"), 0) or 0) > 0:
        st.error("Apply is blocked until structural workbook issues are resolved.")
        return
    baseline_reference = dict(pending_summary.get("baseline_reference") or {})
    if baseline_reference.get("requires_confirmation") and not bool(st.session_state.get(V21_REQUEST_IMPORT_CONFIRM_BASELINE_KEY)):
        st.warning("Confirm 'Use baseline from imported request' before applying this import.")
        return
    _v21_clear_incompatible_import_widget_state()
    target, domain = _v21_first_import_focus(pending_state)
    if target:
        pending_state["proposal_target"] = target
    if domain:
        pending_state["v21_component_group"] = _COMPONENT_GROUP_BY_DOMAIN.get(domain, "mass_aero")
    _v2_set_state(pending_state)
    if target:
        st.session_state["v21_detail_target"] = target
    if domain:
        st.session_state["v21_detail_domain"] = domain
        st.session_state["v21_component_group"] = _COMPONENT_GROUP_BY_DOMAIN.get(domain, "mass_aero")
    if baseline_reference.get("association_required"):
        st.session_state["v21_flash_message"] = "Imported PPE Request applied. Associate a valid baseline before Preview & Save."
    else:
        st.session_state["v21_flash_message"] = "Imported PPE Request applied. The current unsaved draft was replaced."
    _v21_clear_request_runtime_state(clear_resolution=True)
    _v21_clear_import_pending_state()
    st.rerun()


def _v21_render_request_import_panel() -> None:
    applied_state = dict(_v2_state())
    pending_summary = dict(st.session_state.get(V21_REQUEST_IMPORT_PENDING_SUMMARY_KEY) or {})
    applied_summary = dict(applied_state.get("vde_request_import_summary") or {})
    summary = pending_summary or applied_summary
    pending_error = str(st.session_state.get(V21_REQUEST_IMPORT_PENDING_ERROR_KEY) or "").strip()
    baseline_reference = dict(summary.get("baseline_reference") or {})
    needs_confirmation = bool(baseline_reference.get("requires_confirmation"))
    confirmed = bool(st.session_state.get(V21_REQUEST_IMPORT_CONFIRM_BASELINE_KEY))

    with st.expander("Load filled PPE template", expanded=False):
        st.caption("Load a completed PPE Request. After validation, applying it replaces the current unsaved request draft.")
        cols = st.columns([1.5, 0.6, 0.8])
        upload = cols[0].file_uploader(
            "PPE Request workbook",
            type=["xlsx"],
            accept_multiple_files=False,
            key="v21_request_import_file",
        )
        validate_clicked = cols[1].button("Validate", key="v21_request_import_validate_button", use_container_width=True)
        apply_disabled = (
            not bool(st.session_state.get(V21_REQUEST_IMPORT_PENDING_STATE_KEY))
            or int(to_float(dict(st.session_state.get(V21_REQUEST_IMPORT_PENDING_SUMMARY_KEY) or {}).get("blocking_count"), 0) or 0) > 0
            or (needs_confirmation and not confirmed)
        )
        apply_clicked = cols[2].button("Apply imported request", key="v21_request_import_apply_button", use_container_width=True, disabled=apply_disabled)

        if validate_clicked:
            _v21_clear_import_pending_state()
            next_state, next_summary, error_message = _v21_import_uploaded_request(upload)
            if next_summary:
                st.session_state[V21_REQUEST_IMPORT_PENDING_SUMMARY_KEY] = next_summary
            if next_state:
                st.session_state[V21_REQUEST_IMPORT_PENDING_STATE_KEY] = next_state
                st.session_state[V21_REQUEST_IMPORT_PENDING_DRAFT_KEY] = deepcopy(dict(next_state.get("vde_request_draft") or {}))
                st.session_state[V21_REQUEST_IMPORT_PENDING_SOURCE_KEY] = deepcopy(dict(next_state.get("vde_request_source") or {}))
            if error_message:
                st.session_state[V21_REQUEST_IMPORT_PENDING_ERROR_KEY] = error_message
            elif next_state:
                st.success("Workbook validated and adapted. Review the summary below, then apply the import.")

        if apply_clicked:
            _v21_apply_imported_request_state()

        pending_error = str(st.session_state.get(V21_REQUEST_IMPORT_PENDING_ERROR_KEY) or "").strip()
        summary = dict(st.session_state.get(V21_REQUEST_IMPORT_PENDING_SUMMARY_KEY) or applied_summary or {})
        if pending_error:
            st.error(pending_error)
        if summary:
            render_vde_workbook_table(
                pd.DataFrame(_v21_request_import_summary_rows(summary)),
                title="Import summary",
                table_id="v21-request-import-summary",
            )
            issue_rows = _v21_request_import_issue_rows(summary)
            if issue_rows:
                render_vde_workbook_table(
                    pd.DataFrame(issue_rows),
                    title="Import issues",
                    table_id="v21-request-import-issues",
                )
        baseline_reference = dict(summary.get("baseline_reference") or {})
        if baseline_reference.get("requires_confirmation"):
            st.warning(str(baseline_reference.get("message") or "The imported request references a different baseline than the current page."))
            confirm_cols = st.columns([1.1, 0.9, 3.0])
            if confirm_cols[0].button("Use baseline from imported request", key="v21_request_import_confirm_baseline", use_container_width=True):
                st.session_state[V21_REQUEST_IMPORT_CONFIRM_BASELINE_KEY] = True
                st.rerun()
            if confirm_cols[1].button("Cancel import", key="v21_request_import_cancel_baseline_mismatch", use_container_width=True):
                _v21_clear_import_pending_state()
                st.session_state["v21_flash_message"] = "Imported PPE Request was cancelled."
                st.rerun()
            if st.session_state.get(V21_REQUEST_IMPORT_CONFIRM_BASELINE_KEY):
                confirm_cols[2].success("Imported baseline confirmed. Apply imported request is now enabled.")


def _v21_render_request_resolution_preview(state: dict | None = None) -> None:
    state = state or _v2_state()
    available = _v21_request_preview_available(state)
    current_hash = _v21_request_resolution_current_hash(state)
    stored_hash = str(st.session_state.get(V21_REQUEST_RESOLUTION_HASH_KEY) or "")
    resolution = deepcopy(dict(st.session_state.get(V21_REQUEST_RESOLUTION_STATE_KEY) or {}))
    stale = bool(resolution) and stored_hash != current_hash
    st.session_state[V21_REQUEST_RESOLUTION_STALE_KEY] = stale
    error_message = str(st.session_state.get(V21_REQUEST_RESOLUTION_ERROR_KEY) or "").strip()

    with st.container(border=True):
        st.markdown("**VDE Request Preview**")
        st.caption("Validate the current request, resolve the proposals, and review the engineering results before saving.")
        action_cols = st.columns([0.9, 1.4, 1.2])
        validate_disabled = not available
        validate_clicked = action_cols[0].button(
            "Validate & Preview",
            key="v21_request_validate_preview",
            use_container_width=True,
            disabled=validate_disabled,
        )
        if validate_clicked:
            resolution, current_hash, error_message = _v21_resolve_request_preview(state)
            if resolution:
                st.session_state[V21_REQUEST_RESOLUTION_STATE_KEY] = deepcopy(resolution)
                st.session_state[V21_REQUEST_RESOLUTION_HASH_KEY] = current_hash
                st.session_state[V21_REQUEST_RESOLUTION_STALE_KEY] = False
                st.session_state[V21_REQUEST_RESOLUTION_ERROR_KEY] = ""
                proposal_results = list(resolution.get("proposal_results") or [])
                if proposal_results:
                    st.session_state[V21_REQUEST_RESOLUTION_SELECTED_PROPOSAL_KEY] = str(proposal_results[0].get("proposal_id") or "")
                st.success("Request preview validated and refreshed.")
            else:
                st.session_state[V21_REQUEST_RESOLUTION_ERROR_KEY] = error_message
                st.session_state[V21_REQUEST_RESOLUTION_STALE_KEY] = True
                st.error(error_message or "Request preview could not be resolved.")

        baseline_reference = dict(dict(state.get("vde_request_import") or {}).get("baseline_reference") or {})
        if not available and baseline_reference.get("association_required"):
            action_cols[1].warning("Preview blocked until a valid baseline is associated in Baseline Reference.")
        elif not available:
            action_cols[1].info("No request-style draft is staged yet. Import a PPE Request or add requested proposals first.")
        elif stale:
            action_cols[1].warning("Preview outdated — run Validate & Preview again.")
        elif resolution:
            action_cols[1].success("Current request preview is up to date.")
        else:
            action_cols[1].info("Run Validate & Preview to build the comparative request preview.")
        action_cols[2].caption(f"Fingerprint: `{current_hash[:12]}`")

        cached_draft_bytes = st.session_state.get(V21_REQUEST_DRAFT_REPORT_BYTES_KEY)
        cached_draft_name = str(st.session_state.get(V21_REQUEST_DRAFT_REPORT_NAME_KEY) or "EcoDrive_VDE_Request_DRAFT.xlsx")
        cached_draft_fp = str(st.session_state.get(V21_REQUEST_DRAFT_REPORT_FINGERPRINT_KEY) or "")

        if error_message:
            st.error(error_message)
        if resolution and not stale:
            if cached_draft_fp != current_hash or not cached_draft_bytes:
                report_bytes, report_name = _v21_build_request_report_bytes(state, resolution)
                st.session_state[V21_REQUEST_DRAFT_REPORT_BYTES_KEY] = report_bytes
                st.session_state[V21_REQUEST_DRAFT_REPORT_NAME_KEY] = report_name
                st.session_state[V21_REQUEST_DRAFT_REPORT_FINGERPRINT_KEY] = current_hash
                cached_draft_bytes = report_bytes
                cached_draft_name = report_name
                cached_draft_fp = current_hash
            report_cols = st.columns([1.2, 2.8])
            report_cols[0].download_button(
                "Download Draft Report",
                data=cached_draft_bytes,
                file_name=cached_draft_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"v21_request_download_draft__{current_hash[:12]}",
                use_container_width=True,
            )
            report_cols[1].caption("Draft export uses the current validated Preview snapshot only. It is marked as not committed to the VDE Database.")
        elif stale and cached_draft_bytes and cached_draft_fp == stored_hash:
            st.info(f"Draft report for the last validated preview is still available (`{cached_draft_fp[:12]}`). Re-run Validate & Preview to refresh it.")
            st.download_button(
                "Download Last Draft Report",
                data=cached_draft_bytes,
                file_name=cached_draft_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"v21_request_download_last_draft__{cached_draft_fp[:12]}",
                use_container_width=True,
            )
        if not available or stale or not resolution:
            return

        summary = build_validation_summary(resolution)
        summary_cols = st.columns(8)
        summary_cols[0].metric("Overall", str(summary.get("overall_status") or "OK"))
        summary_cols[1].metric("Proposals", int(summary.get("proposal_count") or 0))
        summary_cols[2].metric("OK", int(summary.get("ok_count") or 0))
        summary_cols[3].metric("Review", int(summary.get("review_count") or 0))
        summary_cols[4].metric("Missing", int(summary.get("missing_count") or 0))
        summary_cols[5].metric("Invalid", int(summary.get("invalid_count") or 0))
        summary_cols[6].metric("Blocked", int(summary.get("blocked_count") or 0))
        summary_cols[7].metric("Warnings", int(summary.get("warning_count") or 0))

        comparison_rows = build_request_comparison_rows(resolution)
        if comparison_rows:
            render_vde_workbook_table(
                pd.DataFrame(comparison_rows),
                title="Comparison summary",
                table_id="v21-request-comparison-summary",
            )

        proposal_results = {
            str(item.get("proposal_id") or ""): dict(item or {})
            for item in list(resolution.get("proposal_results") or [])
            if str(item.get("proposal_id") or "").strip()
        }
        if proposal_results:
            proposal_ids = list(proposal_results)
            current_selection = str(st.session_state.get(V21_REQUEST_RESOLUTION_SELECTED_PROPOSAL_KEY) or proposal_ids[0])
            if current_selection not in proposal_results:
                current_selection = proposal_ids[0]
            selected_proposal_id = st.selectbox(
                "Proposal detail",
                proposal_ids,
                index=proposal_ids.index(current_selection),
                format_func=lambda value: f"{proposal_results[value].get('source_column') or value} | {proposal_results[value].get('status') or 'OK'}",
                key="v21_request_resolution_selected_proposal_selector",
            )
            st.session_state[V21_REQUEST_RESOLUTION_SELECTED_PROPOSAL_KEY] = selected_proposal_id
            model = build_proposal_preview_model(proposal_results[selected_proposal_id])
            header = dict(model.get("header") or {})
            header_cols = st.columns(5)
            header_cols[0].metric("Request", str(header.get("requested_label") or header.get("source_column") or selected_proposal_id))
            header_cols[1].metric("Status", str(header.get("status") or "OK"))
            header_cols[2].metric("Walk From", str(header.get("walk_from") or "Baseline"))
            header_cols[3].metric("Source column", str(header.get("source_column") or "—"))
            header_cols[4].metric("Issues", int(header.get("issues_count") or 0))

            render_vde_workbook_table(
                pd.DataFrame(model.get("engineering_rows") or []),
                title="Engineering Result",
                table_id="v21-request-engineering-result",
            )
            if list(model.get("domain_change_rows") or []):
                render_vde_workbook_table(
                    pd.DataFrame(model.get("domain_change_rows") or []),
                    title="Domain Changes",
                    table_id="v21-request-domain-changes",
                )
            component_rows = list(model.get("component_action_rows") or [])
            if component_rows:
                render_vde_workbook_table(
                    pd.DataFrame(component_rows),
                    title="Component Actions",
                    table_id="v21-request-component-actions",
                )
            validation_rows = list(model.get("validation_rows") or [])
            if validation_rows:
                render_vde_workbook_table(
                    pd.DataFrame(validation_rows),
                    title="Validation",
                    table_id="v21-request-validation",
                )
            audit_rows = list(model.get("audit_rows") or [])
            if audit_rows:
                render_vde_workbook_table(
                    pd.DataFrame(audit_rows),
                    title="Audit",
                    table_id="v21-request-proposal-audit",
                )

        with st.expander("Technical Audit", expanded=False):
            audit_rows = build_request_audit_rows(resolution)
            if audit_rows:
                render_vde_workbook_table(
                    pd.DataFrame(audit_rows),
                    title="Request audit",
                    table_id="v21-request-audit",
                )


def _v21_render_request_review_save(state: dict | None = None, *, reset_ctx=None) -> None:
    state = state or _v2_state()
    resolution = deepcopy(dict(st.session_state.get(V21_REQUEST_RESOLUTION_STATE_KEY) or {}))
    current_hash = _v21_request_resolution_current_hash(state)
    resolved_hash = str(st.session_state.get(V21_REQUEST_RESOLUTION_HASH_KEY) or "")
    preview_stale = bool(st.session_state.get(V21_REQUEST_RESOLUTION_STALE_KEY))
    saved_fingerprint = str(st.session_state.get(V21_REQUEST_SAVE_FINGERPRINT_KEY) or "")
    saved_proposals = list(st.session_state.get(V21_REQUEST_SAVE_PROPOSALS_KEY) or [])
    if saved_fingerprint and saved_fingerprint != current_hash:
        saved_proposals = []

    with st.container(border=True):
        st.markdown("**Review & Save**")
        st.caption("Plan the save from the current validated request preview, confirm review and baseline update choices, then persist one row per proposal.")
        if not resolution:
            st.info("Run Validate & Preview before building a Save Plan for the request flow.")
            return
        if preview_stale or resolved_hash != current_hash:
            st.warning("Save is disabled until the Preview is refreshed with Validate & Preview.")
            return

        proposal_results = list(resolution.get("proposal_results") or [])
        proposal_ids = [str(item.get("proposal_id") or "") for item in proposal_results if str(item.get("proposal_id") or "").strip()]
        save_mode = st.selectbox(
            "Save mode",
            list(SAVE_MODES),
            key="v21_request_save_mode",
        )
        if save_mode == SAVE_MODE_SELECTED:
            selected_proposal_ids = st.multiselect(
                "Proposal selection",
                proposal_ids,
                default=[item for item in proposal_ids if item not in saved_proposals],
                format_func=lambda value: next((str(item.get("source_column") or value) for item in proposal_results if str(item.get("proposal_id") or "") == value), value),
                key="v21_request_save_selected_ids",
            )
        else:
            selected_proposal_ids = proposal_ids

        review_confirmations = {}
        review_rows = [
            item for item in proposal_results
            if str(item.get("status") or "").strip().lower() == "review"
            and dict(dict(item.get("vde_results") or {}).get("total") or {}).get("mj_per_km") not in (None, "")
        ]
        if review_rows:
            st.caption("Review confirmations")
            for item in review_rows:
                proposal_id = str(item.get("proposal_id") or "")
                review_confirmations[proposal_id] = st.checkbox(
                    f"Confirm Review save for {str(item.get('source_column') or proposal_id)}",
                    key=f"v21_request_review_confirm__{proposal_id}",
                    value=bool(st.session_state.get(f"v21_request_review_confirm__{proposal_id}", False)),
                )

        baseline_update_choices = {}
        baseline = dict(resolution.get("baseline") or {})
        corrected_fields = list(baseline.get("corrected_fields") or [])
        if corrected_fields:
            st.caption("Baseline correction updates")
            for field_key in corrected_fields:
                choice_key = f"v21_request_baseline_update__{field_key}"
                baseline_update_choices[field_key] = st.checkbox(
                    f"Update original baseline field `{field_key}` with its correction",
                    key=choice_key,
                    value=bool(st.session_state.get(choice_key, True)),
                )

        preliminary_plan = build_vde_request_save_plan(
            resolution,
            selected_proposal_ids=selected_proposal_ids,
            save_mode=save_mode,
            review_confirmations=review_confirmations,
            baseline_update_choices=baseline_update_choices,
            request_state=state,
            current_fingerprint=current_hash,
            resolution_fingerprint=resolved_hash,
            preview_is_stale=preview_stale,
            previously_saved_proposal_ids=saved_proposals,
            previous_save_fingerprint=saved_fingerprint,
        )

        component_creation_confirmations = {}
        component_rows = [
            item for item in list(preliminary_plan.get("component_requests") or [])
            if str(item.get("action") or "") == "eligible_for_new_component"
        ]
        if component_rows:
            st.caption("Component creation confirmations")
            for item in component_rows:
                proposal_id = str(item.get("proposal_id") or "")
                domain_key = str(item.get("domain") or "")
                confirm_key = f"v21_request_component_create__{proposal_id}__{domain_key}"
                component_creation_confirmations[f"{proposal_id}:{domain_key}"] = st.checkbox(
                    f"Create reusable {domain_key.replace('_', ' ').title()} component for {str(item.get('source_column') or proposal_id)}",
                    key=confirm_key,
                    value=bool(st.session_state.get(confirm_key, False)),
                    disabled=not bool(item.get("creation_supported")),
                )

        save_plan = build_vde_request_save_plan(
            resolution,
            selected_proposal_ids=selected_proposal_ids,
            save_mode=save_mode,
            review_confirmations=review_confirmations,
            baseline_update_choices=baseline_update_choices,
            component_creation_confirmations=component_creation_confirmations,
            request_state=state,
            current_fingerprint=current_hash,
            resolution_fingerprint=resolved_hash,
            preview_is_stale=preview_stale,
            previously_saved_proposal_ids=saved_proposals,
            previous_save_fingerprint=saved_fingerprint,
        )

        plan_rows = build_vde_request_save_plan_rows(save_plan)
        if plan_rows:
            render_vde_workbook_table(
                pd.DataFrame(plan_rows),
                title="Final Save Plan",
                table_id="v21-request-save-plan",
            )
        skipped_rows = list(save_plan.get("skipped_proposals") or [])
        if skipped_rows:
            render_vde_workbook_table(
                pd.DataFrame(skipped_rows),
                title="Proposals skipped",
                table_id="v21-request-save-skipped",
            )
        baseline_rows = list(save_plan.get("baseline_updates") or [])
        if baseline_rows:
            render_vde_workbook_table(
                pd.DataFrame(baseline_rows),
                title="Baseline fields to update",
                table_id="v21-request-save-baseline",
            )
        component_plan_rows = list(save_plan.get("component_requests") or [])
        if component_plan_rows:
            render_vde_workbook_table(
                pd.DataFrame(component_plan_rows),
                title="Component actions",
                table_id="v21-request-save-components",
            )
        blocking_rows = list(save_plan.get("blocking_issues") or [])
        if blocking_rows:
            render_vde_workbook_table(
                pd.DataFrame(blocking_rows),
                title="Blocking issues",
                table_id="v21-request-save-blocking",
            )
        warning_rows = list(save_plan.get("warnings") or [])
        if warning_rows:
            render_vde_workbook_table(
                pd.DataFrame(warning_rows),
                title="Save plan warnings",
                table_id="v21-request-save-warnings",
            )

        execute_disabled = not bool(save_plan.get("can_execute"))
        if st.button(
            "Save selected proposals to VDE Database",
            key="v21_request_execute_save",
            use_container_width=True,
            disabled=execute_disabled,
        ):
            result = execute_vde_request_save_plan(save_plan)
            st.session_state[V21_REQUEST_SAVE_RESULT_KEY] = deepcopy(result)
            if str(result.get("status") or "") in {"success", "partial"}:
                saved_now = [str(item.get("proposal_id") or "") for item in list(result.get("saved_proposals") or []) if str(item.get("proposal_id") or "").strip()]
                st.session_state[V21_REQUEST_SAVE_FINGERPRINT_KEY] = current_hash
                st.session_state[V21_REQUEST_SAVE_PROPOSALS_KEY] = sorted(set(saved_proposals) | set(saved_now))
                saved_report_bytes, saved_report_name = _v21_build_request_report_bytes(state, resolution, save_result=result)
                st.session_state[V21_REQUEST_SAVED_REPORT_BYTES_KEY] = saved_report_bytes
                st.session_state[V21_REQUEST_SAVED_REPORT_NAME_KEY] = saved_report_name
                st.session_state[V21_REQUEST_SAVED_REPORT_OPERATION_KEY] = str(result.get("operation_id") or "")
            st.rerun()

        save_result = dict(st.session_state.get(V21_REQUEST_SAVE_RESULT_KEY) or {})
        if save_result:
            if str(save_result.get("status") or "") == "success":
                st.success(f"Save completed. Operation `{save_result.get('operation_id')}`.")
            elif str(save_result.get("status") or "") == "partial":
                st.warning(f"Save partially completed. Operation `{save_result.get('operation_id')}`.")
            else:
                st.error(f"Save failed. Operation `{save_result.get('operation_id')}`.")
            result_rows = build_vde_request_save_result_rows(save_result)
            if result_rows:
                render_vde_workbook_table(
                    pd.DataFrame(result_rows),
                    title="Save Result",
                    table_id="v21-request-save-result",
                )
            post_summary = _v21_post_save_summary(save_result)
            summary_cols = st.columns(7)
            summary_cols[0].metric("Operation status", str(post_summary.get("operation_status") or "-"))
            summary_cols[1].metric("Rows saved", int(post_summary.get("rows_saved") or 0))
            summary_cols[2].metric("Rows skipped", int(post_summary.get("rows_skipped") or 0))
            summary_cols[3].metric("Baseline fields updated", int(post_summary.get("baseline_fields_updated") or 0))
            summary_cols[4].metric("Components reused", int(post_summary.get("components_reused") or 0))
            summary_cols[5].metric("Components created", int(post_summary.get("components_created") or 0))
            summary_cols[6].metric("Failures", int(post_summary.get("failures") or 0))

            saved_report_bytes = st.session_state.get(V21_REQUEST_SAVED_REPORT_BYTES_KEY)
            saved_report_name = str(st.session_state.get(V21_REQUEST_SAVED_REPORT_NAME_KEY) or "EcoDrive_VDE_Request_SAVED.xlsx")
            if not saved_report_bytes:
                saved_report_bytes, saved_report_name = _v21_build_request_report_bytes(state, resolution, save_result=save_result)
                st.session_state[V21_REQUEST_SAVED_REPORT_BYTES_KEY] = saved_report_bytes
                st.session_state[V21_REQUEST_SAVED_REPORT_NAME_KEY] = saved_report_name
                st.session_state[V21_REQUEST_SAVED_REPORT_OPERATION_KEY] = str(save_result.get("operation_id") or "")
            action_cols = st.columns([1.2, 1.1, 1.7])
            if saved_report_bytes:
                action_cols[0].download_button(
                    "Download Saved Report",
                    data=saved_report_bytes,
                    file_name=saved_report_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"v21_request_download_saved__{str(save_result.get('operation_id') or 'save')}",
                    use_container_width=True,
                )
            if action_cols[1].button(
                "Start New Request",
                key="v21_request_start_new_after_save",
                use_container_width=True,
            ):
                _v21_reset_request_flow(reset_ctx=reset_ctx)
            if action_cols[2].button(
                "Keep Reviewing Current Request",
                key="v21_request_keep_reviewing_after_save",
                use_container_width=True,
            ):
                st.session_state["v21_preview_flash"] = "Current request and saved results were kept in place for further review/export."
                st.rerun()


def _v21_preview_rows(state: dict | None = None, previews: dict[str, dict] | None = None) -> list[dict]:
    state = state or _v2_state()
    resolved = _v21_resolved_workbook_model(state)
    resolved_columns = dict(resolved.get("columns") or {})
    column_ids = _v2_column_ids(state)
    column_labels = _v21_request_column_labels(state)
    previews = previews or {column_id: _v2_cached_preview(column_id) for column_id in column_ids}

    def preview_value(field_id: str, column_id: str) -> str:
        preview = dict(previews.get(column_id) or {})
        effective = _v2_effective_state(column_id)
        resolved_column = dict(resolved_columns.get(column_id) or {})
        statuses = _v2_domain_statuses(column_id, preview)
        if field_id == "walk_from":
            walk_from = str(resolved_column.get("walk_from") or effective.get("walk_from") or "baseline")
            return "-" if column_id == "baseline" else _v2_column_label(walk_from, state)
        if field_id == "proposal_direct":
            return str(resolved_column.get("proposal_direct") or effective.get("proposal_direct") or "-")
        if field_id == "proposal_effective":
            return str(resolved_column.get("proposal_effective") or effective.get("proposal_effective") or "-")
        if field_id == "abc_total":
            return _compact_abc(dict(preview.get("abc_total") or {}))
        if field_id == "transmission_losses":
            return _compact_abc(dict((preview.get("transmission_losses") or {}).get("abc") or {})) if (preview.get("transmission_losses") or {}).get("abc") else "Unavailable"
        if field_id == "abc_net":
            return _compact_abc(dict(preview.get("abc_net") or {})) if preview.get("abc_net") else "Unavailable"
        if field_id == "test_mass":
            return _v2_format_value(effective.get("effective_test_mass_kg") or effective.get("test_mass_kg"), "mass") or "-"
        if field_id == "test_mass_basis":
            return str(effective.get("vde_mass_basis") or effective.get("test_mass_basis") or "-")
        if field_id == "vde_total":
            return format_quantity(dict(preview.get("vde_total") or {}).get("mj_per_km"), "energy_per_distance", include_unit=True, unavailable="-", format_str="%.3f")
        if field_id == "vde_net":
            return format_quantity(dict(preview.get("vde_net") or {}).get("mj_per_km"), "energy_per_distance", include_unit=True, unavailable="-", format_str="%.3f")
        if field_id == "delta_vs_baseline":
            baseline_val = dict(previews.get("baseline") or {}).get("vde_net", {}).get("mj_per_km")
            current_val = dict(preview.get("vde_net") or {}).get("mj_per_km")
            if baseline_val is None or current_val is None:
                return "-"
            return format_quantity(current_val - baseline_val, "energy_per_distance", include_unit=True, unavailable="-", format_str="%.3f")
        if field_id == "column_status":
            return str(resolved_column.get("effective_status") or _v2_column_status(column_id)[0])
        if field_id == "warnings":
            warnings = [str(item).strip() for item in list(preview.get("warnings") or []) if str(item).strip()]
            return " | ".join(warnings[:2]) if warnings else "-"
        if field_id == "save_status":
            return "Ready" if preview.get("ok") else "Pending"
        if field_id == "mass_aero_status":
            return statuses["Mass & Aero"][0]
        if field_id == "tire_status":
            return statuses["Tire"][0]
        if field_id == "transmission_status":
            return statuses["Transmission"][0]
        if field_id == "brake_status":
            return statuses["Brake"][0]
        if field_id == "axle_status":
            return statuses["Axle & Hubs"][0]
        if field_id == "parasitic_status":
            return statuses["Parasitic Losses"][0]
        return "-"

    specs = [
        ("walk_from", "Walk From", "Effective inheritance source"),
        ("proposal_direct", "Proposal Direct", "Direct proposal labels"),
        ("proposal_effective", "Proposal Effective", "Accumulated proposal labels"),
        ("abc_total", "ABC_TOTAL A/B/C", "Preview result"),
        ("transmission_losses", "Transmission losses A/B/C", "Preview result"),
        ("abc_net", "ABC_NET A/B/C", "Preview result"),
        ("test_mass", "Resolved VDE test mass [kg]", "Effective snapshot"),
        ("test_mass_basis", "Test mass basis", "Effective snapshot"),
        ("vde_total", "VDE_TOTAL", "Preview result"),
        ("vde_net", "VDE_NET", "Preview result"),
        ("delta_vs_baseline", "Delta vs Baseline", "VDE_NET delta vs Baseline"),
        ("mass_aero_status", "Mass & Aero status", "Domain status"),
        ("tire_status", "Tire status", "Domain status"),
        ("transmission_status", "Transmission status", "Domain status"),
        ("brake_status", "Brake status", "Domain status"),
        ("axle_status", "Axle & Hubs status", "Domain status"),
        ("parasitic_status", "Parasitic status", "Domain status"),
        ("column_status", "Column status", "Overall column resolution"),
        ("warnings", "Warnings", "Preview warnings"),
        ("save_status", "Save status", "Save readiness"),
    ]
    rows: list[dict] = []
    for field_id, label, notes in specs:
        row = {"field": label}
        for column_id in column_ids:
            row[column_labels[column_id]] = preview_value(field_id, column_id)
        row["notes"] = notes
        rows.append(row)
    return rows


def _v21_request_preview_rows(state: dict | None = None) -> list[dict]:
    state = state or _v2_state()
    column_labels = _v21_request_column_labels(state)
    request_columns = [column_id for column_id in _v2_column_ids(state) if column_id != "baseline"]
    rows_by_key: dict[tuple[str, str], dict[str, object]] = {}

    for domain_key, config in VDE_WORKBOOK_V21_DOMAINS.items():
        active = _v21_active_domain_proposals(domain_key, state)
        for column_id in request_columns:
            proposal = dict(active.get(column_id) or {})
            if not proposal:
                continue
            proposal_type = str(proposal.get("proposal_type") or proposal.get("type") or "").strip().upper()
            if not proposal_type:
                continue
            details = _v21_normalize_details(proposal.get("details") or {})
            context = {
                "baseline_tire_code": _v21_reference_raw_value(column_id, "baseline_tire_code", state),
                "front_pressure_psi": _v21_reference_raw_value(column_id, "front_pressure_psi", state),
                "rear_pressure_psi": _v21_reference_raw_value(column_id, "rear_pressure_psi", state),
                "tire_load_mass_basis": _v21_reference_raw_value(column_id, "tire_load_mass_basis", state),
                "weight_dist_fr_pct": _v21_reference_raw_value(column_id, "weight_dist_fr_pct", state),
                "column_id": column_id,
                "state": state,
            }
            status_value, _, _, _ = _v21_validate_proposal_details(column_id, domain_key, proposal_type, details, state)
            display_fields = _v21_compact_fields_for_proposal(domain_key, proposal_type, details, context)
            for field_id in display_fields:
                canonical = _v21_canonical_field_id(field_id)
                if canonical in VDE_WORKBOOK_V21_SPECIAL_DETAIL_FIELDS or canonical == "baseline_component_reference_mode":
                    continue
                if not _v21_detail_field_editable(domain_key, proposal_type, canonical, details, context):
                    continue
                requested_value = _v21_detail_value(details, canonical)
                baseline_field_id = _v21_requested_field_baseline_field(canonical, domain_key)
                baseline_value = _v21_display_baseline_value(baseline_field_id, domain_key, state)
                baseline_override_active = _v21_baseline_printed_override_active(baseline_field_id, domain_key, state)
                if requested_value in (None, "") and not baseline_override_active:
                    continue
                row_key = (domain_key, canonical)
                row = rows_by_key.setdefault(
                    row_key,
                    {
                        "field": _v21_detail_field_label(canonical),
                        "domain": str(config["label"]).replace(" proposal", ""),
                        "Baseline / Printed": _v21_detail_display_text(baseline_value, baseline_field_id) if baseline_value not in (None, "") else "-",
                        "Baseline Override": "Yes" if baseline_override_active else "No",
                        "Requested #1 input": "-",
                        "Requested #1 delta": "-",
                        "Requested #1 status": "-",
                        "Requested #2 input": "-",
                        "Requested #2 delta": "-",
                        "Requested #2 status": "-",
                        "notes": _v21_detail_field_note(canonical) or str(proposal.get("label") or _v21_proposal_type_label(proposal_type)),
                    },
                )
                row["Baseline / Printed"] = _v21_detail_display_text(baseline_value, baseline_field_id) if baseline_value not in (None, "") else "-"
                row["Baseline Override"] = "Yes" if baseline_override_active else "No"
                request_label = column_labels.get(column_id, column_id)
                row[f"{request_label} input"] = _v21_detail_display_text(requested_value, canonical) if requested_value not in (None, "") else "-"
                requested_numeric = to_float(requested_value)
                baseline_numeric = to_float(baseline_value)
                is_delta_input = canonical.startswith("delta_")
                if requested_numeric is None:
                    delta_text = "-"
                    request_status = status_value
                elif is_delta_input:
                    delta_text = _v21_detail_display_text(requested_numeric, canonical)
                    request_status = "Review" if baseline_numeric is None else status_value
                else:
                    delta_value = requested_numeric - (baseline_numeric if baseline_numeric is not None else 0.0)
                    delta_text = _v21_detail_display_text(delta_value, canonical)
                    request_status = "Review" if baseline_numeric is None else status_value
                row[f"{request_label} delta"] = delta_text
                row[f"{request_label} status"] = request_status
                if baseline_numeric is None and requested_numeric is not None and not is_delta_input:
                    row["notes"] = "Missing baseline; delta equals requested absolute."
    return list(rows_by_key.values())


def _v21_domain_label_map() -> dict[str, str]:
    return {
        domain_key: str(config["label"]).replace(" proposal", "")
        for domain_key, config in VDE_WORKBOOK_V21_DOMAINS.items()
    }


def _v21_save_plan_payload(state: dict | None = None, previews: dict[str, dict] | None = None) -> dict:
    state = state or _v2_state()
    resolved = _v21_resolved_workbook_model(state)
    previews = previews or {column_id: _v2_cached_preview(column_id) for column_id in _v2_column_ids(state)}
    metadata = _v2_metadata_effective(state)
    baseline = _v2_effective_state("baseline")
    baseline_is_existing = str(baseline.get("line_source") or metadata.get("line_source") or "").startswith("Existing")
    baseline_target_id = int(to_float(metadata.get("selected_baseline_vde_id"), 0) or 0) or None
    return build_v21_save_plan(
        resolved,
        previews,
        baseline_is_existing=baseline_is_existing,
        baseline_target_id=baseline_target_id,
        selected_target=str(state.get("save_target") or _v2_last_column_id(state)),
        saved_targets=_v21_saved_targets(state),
        domain_labels=_v21_domain_label_map(),
    )


def _v21_save_plan_rows(save_plan: dict) -> list[dict]:
    rows: list[dict] = []
    for item in list(save_plan.get("rows") or []):
        target_id = item.get("target_vde_id")
        target_label = "-" if target_id in (None, "") else f"VDE-{int(target_id):04d}"
        rows.append(
            {
                "column": str(item.get("label") or item.get("column_id") or "-"),
                "action": str(item.get("action") or "-"),
                "target": target_label,
                "status": str(item.get("status") or "-"),
                "confirm": "Required" if item.get("requires_confirmation") else "-",
                "notes": str(item.get("notes") or "-"),
            }
        )
    return rows


def _v21_saved_targets(state: dict | None = None) -> dict[str, int | None]:
    state = state or _v2_state()
    saved = {}
    for key, value in dict(state.get("v21_saved_targets") or {}).items():
        saved[str(key)] = int(to_float(value, 0) or 0) or None
    return saved


def _v21_remember_saved_target(column_id: str, vde_id: int | None) -> None:
    if not vde_id:
        return
    state = _v2_state()
    saved = dict(state.get("v21_saved_targets") or {})
    saved[str(column_id)] = int(vde_id)
    state["v21_saved_targets"] = saved
    _v2_set_state(state)


def _v21_save_mode_from_plan_row(plan_row: dict) -> str | None:
    action = str(plan_row.get("action") or "").strip().lower()
    if action == "update_existing":
        return "update_existing"
    if action == "create_new":
        return "insert_new"
    return None


def _v21_prepare_preview_for_target(preview: dict, target_vde_id: int | None) -> dict:
    prepared = deepcopy(dict(preview or {}))
    save_payload = dict(prepared.get("save_payload") or {})
    save_payload["target_vde_id"] = target_vde_id
    prepared["save_payload"] = save_payload
    return prepared


def _v21_render_preview_save(defaults_df_getter, *, reset_ctx=None) -> None:
    state = _v2_state()
    preview_flash = st.session_state.pop("v21_preview_flash", None)
    if preview_flash:
        st.success(str(preview_flash))

    _v21_render_request_resolution_preview(state)
    _v21_render_request_review_save(state, reset_ctx=reset_ctx)


def _v21_render_advanced_domain_editors() -> None:
    domain = st.selectbox(
        "Advanced editor",
        list(VDE_WORKBOOK_V2_SECTION_SPECS.keys()),
        key="v21_advanced_domain_editor",
    )
    _v2_render_section_editor(domain)


def render_vde_setup_workbook_v21(*, defaults_df_getter, defaults_path=None, reset_ctx=None) -> None:
    st.session_state["_vde_workbook_active_state_key"] = VDE_WORKBOOK_V21_STATE_KEY
    state = _v2_state()
    _v21_render_context_header()
    _v21_render_status_bar()
    st.divider()
    menu = st.radio(
        "Workbook v2.1 menu",
        VDE_WORKBOOK_V21_MENUS,
        horizontal=True,
        index=VDE_WORKBOOK_V21_MENUS.index(str(state.get("menu_v21") or VDE_WORKBOOK_V21_MENUS[0])) if str(state.get("menu_v21") or VDE_WORKBOOK_V21_MENUS[0]) in VDE_WORKBOOK_V21_MENUS else 0,
        key="vde_workbook_v21_menu_selector",
        label_visibility="collapsed",
    )
    state["menu_v21"] = menu
    _v2_set_state(state)
    if menu == "Scenario Workbook":
        _v21_render_scenario_input_workbook()
    elif menu == "Preview & Save":
        _v21_render_preview_save(defaults_df_getter, reset_ctx=reset_ctx)


def render_vde_setup_workbook_v2(*, defaults_df_getter, defaults_path=None, reset_ctx=None) -> None:
    st.session_state["_vde_workbook_active_state_key"] = VDE_WORKBOOK_V2_STATE_KEY
    state = _v2_state()
    _v2_render_context_header()
    _v2_render_status_bar()
    st.divider()
    menu = st.radio(
        "Workbook v2 menu",
        VDE_WORKBOOK_V2_MENUS,
        horizontal=True,
        index=VDE_WORKBOOK_V2_MENUS.index(str(state.get("menu") or VDE_WORKBOOK_V2_MENUS[0])) if str(state.get("menu") or VDE_WORKBOOK_V2_MENUS[0]) in VDE_WORKBOOK_V2_MENUS else 0,
        key="vde_workbook_v2_menu_selector",
        label_visibility="collapsed",
    )
    state["menu"] = menu
    _v2_set_state(state)

    if menu == "Scenario Workbook":
        _v2_render_matrix_editor()
    elif menu in VDE_WORKBOOK_V2_SECTION_SPECS:
        _v2_render_section_editor(menu)
    elif menu == "Preview & Save":
        _v2_render_preview_save(defaults_df_getter)
    elif menu == "Technical Audit":
        _v2_render_technical_audit()


def _render_tire_component_editor(*, base_row: dict | None = None, saved_vde_id: int | None = None, tires_df=None):
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    if basis != "Component Build-up":
        _render_tire_pressure_unit_toggle(key="tire_component_editor_pressure_unit_toggle")
    if basis == "Component Build-up":
        mode = _render_component_mode_selector("Tires")
        if mode == "Lookup from DB":
            render_tire_component_section(
                base_row=base_row,
                saved_vde_id=saved_vde_id,
                tires_df=tires_df,
                source_mode_override="Tire DB",
                show_source_selector=False,
            )
        else:
            render_tire_component_section(
                base_row=base_row,
                saved_vde_id=saved_vde_id,
                tires_df=tires_df,
                source_mode_override="Manual RR",
                show_source_selector=False,
            )
        return

    base = dict(base_row or {})
    baseline_current = _saved_tire_reference_from_row(base)
    current_reference = _resolve_tire_reference_from_ctx(
        "tire_current_reference",
        source_label="Scenario Current reference",
        preview_ctx_key="tire_current_reference_preview_result",
    ) or baseline_current
    walked_reference = _resolve_tire_reference_from_ctx(
        "tire_walked_reference",
        source_label="Scenario Walked reference",
        preview_ctx_key="tire_walked_reference_preview_result",
    )

    treatment_options = ["Keep inherited", "Apply tire change"]
    current_application = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))
    current_mode = str(ctx.get("component_mode_tires") or "")
    if current_mode == "Apply delta" or current_application != "Keep inherited":
        treatment_default = "Apply tire change"
    else:
        treatment_default = "Keep inherited"

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Tire Scenario Treatment",
            "Choose whether tires remain inherited from the selected roadload basis or whether this scenario stages and applies a tire change.",
        )
        tire_treatment = st.radio(
            "Treatment",
            treatment_options,
            horizontal=True,
            index=treatment_options.index(treatment_default),
            key="tire_scenario_treatment_radio",
        )
        if tire_treatment == "Keep inherited":
            ctx["component_mode_tires"] = "Keep inherited"
            ctx["tire_scenario_application"] = "Keep inherited"
            ctx["delta_rr_N"] = 0.0
            ctx["tire_improvement_pct"] = 0.0
            ctx["tire_change_method"] = "Manual tire adjustment"
            if baseline_current:
                st.info("Inherited tire reference remains active. No tire adjustment is applied to this scenario.")
                _render_tire_reference_metrics(baseline_current, title="Inherited tire reference")
            else:
                st.info("No explicit tire reference is saved on the baseline, but this scenario is not applying an active tire adjustment.")
            return
        st.caption("A tire change is active for this scenario. Define the reference on the left and the active change on the right.")

    current_col, change_col = st.columns(2)

    with current_col:
        current_reference = _render_tire_reference_selector(
            title="Current Tire (Reference)",
            helper_text="Defines the current/base tire state used as reference for this scenario. Current is only required for Tire Improvement % and Current vs Walked comparison.",
            mode_key="tire_current_reference_mode",
            prefix="tire_current_reference",
            widget_prefix="tire_current_ref",
            source_label="Scenario Current reference",
            base_row=base_row,
            inherited_reference=baseline_current,
            save_to_baseline_label="Update baseline with this Current reference" if (saved_vde_id or base.get("id")) else None,
            baseline_target_id=int(saved_vde_id or base.get("id")) if (saved_vde_id or base.get("id")) else None,
        )

    crr_frac = float(to_float(ctx.get("crr1_frac_at_120kph"), 0.0) or 0.0)
    load_kN = _resolved_tire_load_kN(ctx)
    suggested_delta_a = None
    suggested_delta_b = None
    suggested_delta_c = None
    if current_reference and walked_reference:
        suggested_delta_a = float(to_float(walked_reference.get("A"), 0.0) or 0.0) - float(to_float(current_reference.get("A"), 0.0) or 0.0)
        suggested_delta_b = float(to_float(walked_reference.get("B"), 0.0) or 0.0) - float(to_float(current_reference.get("B"), 0.0) or 0.0)
        suggested_delta_c = float(to_float(walked_reference.get("C"), 0.0) or 0.0) - float(to_float(current_reference.get("C"), 0.0) or 0.0)

    equivalent_a = None
    equivalent_b = None
    equivalent_c = None

    with change_col:
        with st.container(border=True):
            _render_tire_editor_block_header(
                "Tire Change",
                "Choose how this scenario applies a tire change: direct manual adjustment or walked-vs-current comparison.",
            )
            change_method_options = ["Manual tire adjustment", "Walked tire comparison"]
            change_method_default = _default_tire_change_method(ctx)
            ctx["tire_change_method"] = st.radio(
                "Method",
                change_method_options,
                horizontal=False,
                index=change_method_options.index(change_method_default),
                key="tire_change_method_radio",
            )

            if ctx["tire_change_method"] == "Manual tire adjustment":
                change_intent_options = ["Scenario-only engineering adjustment", "Engineering target / supplier request"]
                current_change_intent = str(ctx.get("tire_manual_change_intent") or change_intent_options[0]).strip()
                if current_change_intent not in change_intent_options:
                    current_change_intent = change_intent_options[0]
                ctx["tire_manual_change_intent"] = st.radio(
                    "Change intent",
                    change_intent_options,
                    horizontal=False,
                    index=change_intent_options.index(current_change_intent),
                    key="tire_manual_change_intent_radio",
                )
                input_type_options = ["Delta final RRC", "Target final RRC", "Tire Improvement %"]
                input_type_default = _default_tire_manual_input_type(ctx)
                ctx["tire_manual_adjustment_input_type"] = st.radio(
                    "Input basis",
                    input_type_options,
                    horizontal=True,
                    index=input_type_options.index(input_type_default),
                    key="tire_manual_adjustment_input_type_radio",
                )

                current_rrc = None
                if current_reference and to_float(current_reference.get("rr_n_per_kn")) is not None:
                    current_rrc = float(to_float(current_reference.get("rr_n_per_kn"), 0.0) or 0.0)
                elif to_float(ctx.get("rrc_N_per_kN")) is not None:
                    current_rrc = float(to_float(ctx.get("rrc_N_per_kN"), 0.0) or 0.0)

                if ctx["tire_manual_adjustment_input_type"] == "Delta final RRC":
                    default_delta_rr = to_float(ctx.get("tire_manual_delta_rr_n_per_kn"))
                    if default_delta_rr is None and load_kN > 0:
                        default_delta_rr, _, _ = _equivalent_rr_from_abc(
                            to_float(ctx.get("delta_rr_N"), 0.0),
                            float(to_float(ctx.get("delta_rr_N"), 0.0) or 0.0) * (crr_frac / 120.0),
                            0.0,
                            load_kN=load_kN,
                        )
                    ctx["tire_manual_delta_rr_n_per_kn"] = quantity_input(
                        st,
                        "Delta final RRC",
                        to_float(default_delta_rr, 0.0),
                        "rrc",
                        key="tire_manual_delta_rr_input",
                        step_canonical=0.05,
                        format_str="%.3f",
                    )
                    md1, md2 = st.columns(2)
                    ctx["tire_manual_delta_rr_label"] = md1.text_input(
                        "Intended tire label",
                        value=str(ctx.get("tire_manual_delta_rr_label") or ""),
                        key="tire_manual_delta_rr_label_input",
                    )
                    with md2:
                        ctx["tire_manual_delta_rr_size_code"] = _render_tire_size_selector(
                            field_key="tire_manual_delta_rr_size_code",
                            label="Intended tire size",
                        )
                    md3, md4 = st.columns(2)
                    ctx["tire_manual_delta_rr_source"] = md3.text_input(
                        "Source / provenance",
                        value=str(ctx.get("tire_manual_delta_rr_source") or ""),
                        key="tire_manual_delta_rr_source_input",
                    )
                    ctx["tire_manual_delta_rr_notes"] = md4.text_input(
                        "Notes",
                        value=str(ctx.get("tire_manual_delta_rr_notes") or ""),
                        key="tire_manual_delta_rr_notes_input",
                    )
                    equivalent_a, equivalent_b, equivalent_c, epa_v_avg = _abc_from_final_rr_target(
                        ctx["tire_manual_delta_rr_n_per_kn"],
                        load_kN=load_kN,
                        crr_frac_120=crr_frac,
                    )

                    ctx["component_mode_tires"] = "Apply delta"
                    ctx["tire_scenario_application"] = "Manual Delta RR"
                    ctx["tire_delta_calculation_mode"] = "Manual delta RR"
                    ctx["tire_improvement_pct"] = 0.0
                    ctx["delta_rr_N"] = equivalent_a

                    if ctx["tire_manual_change_intent"] == "Engineering target / supplier request":
                        if current_rrc is None:
                            ctx["component_mode_tires"] = "Keep inherited"
                            ctx["delta_rr_N"] = 0.0
                            st.warning("Engineering target / supplier request needs a Current Tire reference so the requested delta can be anchored to a reference final RRC.")
                        else:
                            target_rrc = current_rrc + float(ctx["tire_manual_delta_rr_n_per_kn"])
                            t1, t2, t3 = st.columns(3)
                            t1.metric("Reference final RRC", format_quantity(current_rrc, "rrc", format_str="%.3f"))
                            t2.metric("Target final RRC", format_quantity(target_rrc, "rrc", format_str="%.3f"))
                            t3.metric("Delta final RRC", format_quantity(ctx["tire_manual_delta_rr_n_per_kn"], "rrc", format_str="%.3f"))
                    else:
                        st.caption(
                            f"Resolved with VDE tire calculation load = {load_kN:.2f} kN. "
                            "The workflow stores the applied effect internally as equivalent delta A/B for the current EPA-aligned final RRC target."
                        )
                        if epa_v_avg is not None:
                            st.caption(f"EPA v_avg used for delta decomposition: {epa_v_avg:.2f} kph")
                elif ctx["tire_manual_adjustment_input_type"] == "Target final RRC":
                    ctx["tire_manual_target_rr_n_per_kn"] = quantity_input(
                        st,
                        "Target final RRC",
                        to_float(ctx.get("tire_manual_target_rr_n_per_kn"), current_rrc or 0.0),
                        "rrc",
                        key="tire_manual_target_rr_input",
                        step_canonical=0.05,
                        format_str="%.3f",
                    )
                    md1, md2 = st.columns(2)
                    ctx["tire_manual_delta_rr_label"] = md1.text_input(
                        "Intended tire label",
                        value=str(ctx.get("tire_manual_delta_rr_label") or ""),
                        key="tire_manual_target_label_input",
                    )
                    with md2:
                        ctx["tire_manual_delta_rr_size_code"] = _render_tire_size_selector(
                            field_key="tire_manual_delta_rr_size_code",
                            label="Intended tire size",
                        )
                    md3, md4 = st.columns(2)
                    ctx["tire_manual_delta_rr_source"] = md3.text_input(
                        "Source / provenance",
                        value=str(ctx.get("tire_manual_delta_rr_source") or ""),
                        key="tire_manual_target_source_input",
                    )
                    ctx["tire_manual_delta_rr_notes"] = md4.text_input(
                        "Notes",
                        value=str(ctx.get("tire_manual_delta_rr_notes") or ""),
                        key="tire_manual_target_notes_input",
                    )
                    ctx["tire_improvement_pct"] = 0.0
                    if current_rrc is None:
                        ctx["component_mode_tires"] = "Keep inherited"
                        ctx["delta_rr_N"] = 0.0
                        st.warning("Target final RRC needs a Current Tire reference so the scenario can derive a delta from the target.")
                    else:
                        target_rrc = float(to_float(ctx.get("tire_manual_target_rr_n_per_kn"), 0.0) or 0.0)
                        delta_rr_n_per_kn = target_rrc - current_rrc
                        ctx["tire_manual_delta_rr_n_per_kn"] = delta_rr_n_per_kn
                        equivalent_a, equivalent_b, equivalent_c, epa_v_avg = _abc_from_final_rr_target(
                            delta_rr_n_per_kn,
                            load_kN=load_kN,
                            crr_frac_120=crr_frac,
                        )
                        ctx["component_mode_tires"] = "Apply delta"
                        ctx["tire_scenario_application"] = "Manual Delta RR"
                        ctx["tire_delta_calculation_mode"] = "Target RRC"
                        ctx["delta_rr_N"] = equivalent_a
                        t1, t2, t3 = st.columns(3)
                        t1.metric("Reference final RRC", format_quantity(current_rrc, "rrc", format_str="%.3f"))
                        t2.metric("Target final RRC", format_quantity(target_rrc, "rrc", format_str="%.3f"))
                        t3.metric("Delta final RRC", format_quantity(delta_rr_n_per_kn, "rrc", format_str="%.3f"))
                        if epa_v_avg is not None:
                            st.caption(f"EPA v_avg used for target-to-delta decomposition: {epa_v_avg:.2f} kph")
                else:
                    ctx["tire_improvement_pct"] = st.number_input(
                        "Tire Improvement [%]",
                        value=float(ctx.get("tire_improvement_pct", 0.0) or 0.0),
                        step=0.5,
                        format="%.1f",
                        key="tire_improvement_pct_apply_input",
                    )
                    ctx["tire_scenario_application"] = "Tire Improvement %"
                    ctx["tire_delta_calculation_mode"] = "Tire Improvement %"
                    if current_reference:
                        factor = 1.0 - (float(ctx["tire_improvement_pct"]) / 100.0)
                        improved_a = float(to_float(current_reference.get("A"), 0.0) or 0.0) * factor
                        improved_b = float(to_float(current_reference.get("B"), 0.0) or 0.0) * factor
                        improved_c = float(to_float(current_reference.get("C"), 0.0) or 0.0) * factor
                        equivalent_a = improved_a - float(to_float(current_reference.get("A"), 0.0) or 0.0)
                        equivalent_b = improved_b - float(to_float(current_reference.get("B"), 0.0) or 0.0)
                        equivalent_c = improved_c - float(to_float(current_reference.get("C"), 0.0) or 0.0)
                        ctx["component_mode_tires"] = "Apply delta"
                        ctx["delta_rr_N"] = equivalent_a
                        if current_rrc is not None:
                            target_rrc = current_rrc * factor
                            t1, t2, t3 = st.columns(3)
                            t1.metric("Reference final RRC", format_quantity(current_rrc, "rrc", format_str="%.3f"))
                            t2.metric("Target final RRC", format_quantity(target_rrc, "rrc", format_str="%.3f"))
                            t3.metric("Delta final RRC", format_quantity(target_rrc - current_rrc, "rrc", format_str="%.3f"))
                        else:
                            st.caption("Equivalent ABC delta is derived from the Current Tire reference for traceability.")
                    else:
                        ctx["component_mode_tires"] = "Keep inherited"
                        ctx["delta_rr_N"] = 0.0
                        st.warning("Current Tire is optional for manual setup, but it is still needed here to derive an equivalent ABC delta from Tire Improvement %.")
            else:
                walked_reference = _render_tire_reference_selector(
                    title="Walked Tire (Candidate)",
                    helper_text="Stage the walked / proposed tire here. Without a Current reference the candidate can still be registered, but no comparison delta is calculated.",
                    mode_key="tire_walked_reference_mode",
                    prefix="tire_walked_reference",
                    widget_prefix="tire_walked_ref",
                    source_label="Walked tire reference",
                    base_row=base_row,
                    allow_unset=True,
                )

                ctx["tire_scenario_application"] = "Walked tire comparison"
                ctx["tire_delta_calculation_mode"] = "Use Current vs Walked delta"
                ctx["tire_improvement_pct"] = 0.0

                if not current_reference:
                    ctx["component_mode_tires"] = "Keep inherited"
                    ctx["delta_rr_N"] = 0.0
                    st.warning(
                        "Current Tire is required for Walked tire comparison.\n"
                        "Without a Current reference, no comparison delta is calculated."
                    )
                elif walked_reference:
                    suggested_delta_a = float(to_float(walked_reference.get("A"), 0.0) or 0.0) - float(to_float(current_reference.get("A"), 0.0) or 0.0)
                    suggested_delta_b = float(to_float(walked_reference.get("B"), 0.0) or 0.0) - float(to_float(current_reference.get("B"), 0.0) or 0.0)
                    suggested_delta_c = float(to_float(walked_reference.get("C"), 0.0) or 0.0) - float(to_float(current_reference.get("C"), 0.0) or 0.0)
                    equivalent_a = suggested_delta_a
                    equivalent_b = suggested_delta_b
                    equivalent_c = suggested_delta_c
                    ctx["component_mode_tires"] = "Apply delta"
                    ctx["delta_rr_N"] = suggested_delta_a
                    st.caption("Walked comparison is active. The scenario uses the resolved Current vs Walked delta.")
                else:
                    ctx["component_mode_tires"] = "Keep inherited"
                    ctx["delta_rr_N"] = 0.0
                    st.info("Stage a Walked Tire to enable the comparison effect on this scenario.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Applied to scenario",
            "Read-only summary of the tire effect currently resolved for this scenario.",
        )
        applied_method = _normalize_tire_scenario_application(ctx.get("tire_scenario_application"))
        if applied_method == "Keep inherited" or ctx.get("component_mode_tires") == "Keep inherited":
            if applied_method == "Walked tire comparison" and walked_reference and not current_reference:
                st.info("Method: Walked tire comparison. Effect is pending because Current Tire is still missing.")
            elif applied_method == "Walked tire comparison" and not walked_reference:
                st.info("Method: Walked tire comparison. Effect is pending until a Walked Tire is selected.")
            elif applied_method == "Tire Improvement %" and float(to_float(ctx.get("tire_improvement_pct"), 0.0) or 0.0) > 0.0 and not current_reference:
                st.info("Method: Tire Improvement %. Effect is pending until a Current Tire reference is available for equivalent ABC derivation.")
            else:
                st.info("Method: inherited. No tire adjustment is currently affecting the scenario.")
        else:
            if applied_method == "Manual Delta RR":
                effect_text = format_quantity(ctx.get("tire_manual_delta_rr_n_per_kn"), "rrc", format_str="%.3f")
            elif applied_method == "Tire Improvement %":
                effect_text = f"{float(to_float(ctx.get('tire_improvement_pct'), 0.0) or 0.0):.1f}%"
            else:
                effect_text = "Current vs Walked"
            data_role = "engineering_target" if str(ctx.get("tire_manual_change_intent") or "").strip() == "Engineering target / supplier request" else "scenario_assumption"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Method", _display_tire_application_method(applied_method))
            c2.metric("Effect applied", effect_text)
            c3.metric("Status", "Applied")
            c4.metric("Role", _display_data_role(data_role))

            reference_rrc = to_float((current_reference or {}).get("rr_n_per_kn"), to_float(ctx.get("rrc_N_per_kN")))
            target_rrc = to_float(ctx.get("tire_manual_target_rr_n_per_kn"))
            summary_target_text = None
            if applied_method == "Walked tire comparison" and walked_reference:
                summary_target_text = _tire_reference_brief(walked_reference)
            elif target_rrc is not None and abs(float(target_rrc or 0.0)) > 0.0:
                summary_target_text = format_quantity(target_rrc, "rrc", unavailable="-", format_str="%.3f")

            top_summary_parts = [
                f"Reference: {_tire_reference_brief(current_reference)}" if current_reference else None,
                f"Target: {summary_target_text}" if summary_target_text else None,
                f"Intent: {str(ctx.get('tire_manual_change_intent') or '-')}",
            ]
            st.caption(" | ".join(part for part in top_summary_parts if part))

            if equivalent_a is not None:
                e1, e2, e3 = st.columns(3)
                e1.metric(
                    "Equivalent delta ABC",
                    _compact_abc({"A": equivalent_a, "B": equivalent_b or 0.0, "C": equivalent_c or 0.0}),
                )
                e2.metric("Reference final RRC", format_quantity(reference_rrc, "rrc", unavailable="-", format_str="%.3f"))
                e3.metric(
                    "Target / compared final RRC",
                    (
                        format_quantity(target_rrc, "rrc", unavailable="-", format_str="%.3f")
                        if applied_method != "Walked tire comparison"
                        else format_quantity((walked_reference or {}).get("rr_n_per_kn"), "rrc", unavailable="-", format_str="%.3f")
                    ),
                )

            with st.expander("Reference and comparison details", expanded=False):
                ref1, ref2 = st.columns(2)
                with ref1:
                    if current_reference:
                        _render_tire_reference_metrics(current_reference, title="Reference used")
                with ref2:
                    if walked_reference and applied_method == "Walked tire comparison":
                        _render_tire_reference_metrics(walked_reference, title="Candidate used")
                    elif applied_method != "Walked tire comparison":
                        st.caption("No comparison candidate is active for this method.")
                st.caption("Equivalent ABC remains available for traceability. The inherited workflow still consumes the resolved tire effect through the existing delta-based calculation path.")
        _sync_tire_scenario_snapshot_fields(
            applied_method=applied_method,
            current_reference=current_reference,
            walked_reference=walked_reference,
            equivalent_a=equivalent_a,
            equivalent_b=equivalent_b,
            equivalent_c=equivalent_c,
        )


def _render_aero_component_editor(*, base_row: dict | None = None):
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    if basis == "Component Build-up":
        with st.container(border=True):
            _render_tire_editor_block_header(
                "Reference Aero",
                "In component build-up, aerodynamics contributes as an absolute source instead of a delta over inherited roadload.",
            )
            st.caption("Reference status: `Absolute source`")
            render_aero_section(prefill=base_row)
        return
    base = dict(base_row or {})
    reference_cda = to_float(ctx.get("aero_reference_cda_override"), to_float(base.get("cda_m2")))
    reference_defined = reference_cda is not None and abs(float(reference_cda)) > 0.0
    baseline_target_id = int(base.get("id") or ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) or None
    reference_status = "Scenario override" if to_float(ctx.get("aero_reference_cda_override")) is not None else "Inherited"
    candidate_cda = None

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Reference Aero",
            "Defines the current aerodynamic reference used by this scenario when roadload is inherited from baseline or coastdown.",
        )
        if reference_cda is not None:
            _render_scalar_reference_metric(
                label="Reference CdA",
                value=reference_cda,
                quantity="cda",
                status=reference_status,
            )
        else:
            st.warning("Reference status: Missing. No explicit baseline CdA is available for derived comparison.")
        if reference_cda is not None and not reference_defined:
            st.warning("Reference CdA is currently zero. If that is only a missing baseline value, you can correct the baseline row here.")
        if baseline_target_id:
            with st.expander("Set / update baseline reference aero", expanded=not reference_defined):
                baseline_cda_default = to_float(ctx.get("aero_baseline_reference_cda"), reference_cda)
                baseline_cda_value = quantity_input(
                    st,
                    "Baseline reference CdA",
                    to_float(baseline_cda_default, 0.0),
                    "cda",
                    key="aero_baseline_reference_cda_input",
                    step_canonical=0.001,
                    format_str="%.3f",
                )
                ctx["aero_baseline_reference_cda"] = baseline_cda_value
                update_baseline = st.checkbox(
                    "Also update selected baseline row in VDE DB",
                    value=not reference_defined,
                    key="aero_update_baseline_checkbox",
                )
                if st.button("Apply reference aero", key="aero_apply_reference_button"):
                    ctx["cda_m2"] = float(baseline_cda_value)
                    ctx["aero_reference_cda_override"] = float(baseline_cda_value)
                    reference_cda = float(baseline_cda_value)
                    reference_defined = abs(reference_cda) > 0.0
                    if update_baseline:
                        try:
                            update_vde_snapshot(int(baseline_target_id), {"cda_m2": float(baseline_cda_value)})
                            _sync_baseline_cda_in_ctx(float(baseline_cda_value), baseline_target_id=baseline_target_id)
                            ctx["aero_reference_cda_override"] = None
                            st.success(f"Baseline reference aero updated on VDE id={baseline_target_id}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not update baseline reference aero: {exc}")
                    else:
                        st.success("Reference aero staged only for the current scenario context.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Proposed / Applied Aero Change",
            "Define the active aerodynamic change as a manual delta or a proposed-vs-reference comparison.",
        )
        new_cda_label = "New CdA"
        aero_change_options = [
            "Keep inherited",
            new_cda_label,
            "Manual delta CdA",
        ]
        current_candidate_mode = str(ctx.get("aero_candidate_mode") or "Not set")
        current_calc_mode = str(ctx.get("aero_calculation_mode") or "Inherited")
        current_component_mode = str(ctx.get("component_mode_aerodynamics") or "Keep inherited")
        if current_component_mode != "Apply delta" or current_calc_mode == "Inherited":
            current_aero_change = "Keep inherited"
        elif current_candidate_mode == "Manual proposal" or current_calc_mode == "Use candidate vs reference":
            current_aero_change = new_cda_label
        else:
            current_aero_change = "Manual delta CdA"

        selected_aero_change = st.radio(
            "Aero change method",
            aero_change_options,
            horizontal=True,
            index=aero_change_options.index(current_aero_change),
            key="aero_change_mode_radio",
        )

        if selected_aero_change == "Keep inherited":
            ctx["aero_candidate_mode"] = "Not set"
            ctx["aero_calculation_mode"] = "Inherited"
            ctx["component_mode_aerodynamics"] = "Keep inherited"
            ctx["delta_aero_cdA"] = 0.0
            st.info("Calculation status: `Inherited`")
        elif selected_aero_change == new_cda_label:
            ctx["aero_candidate_mode"] = "Manual proposal"
            candidate_cda = _render_candidate_cda_input(
                key_prefix="aero_candidate",
                title="New CdA",
            )
            if candidate_cda is not None and reference_cda is not None:
                suggested_delta_cda = float(candidate_cda) - float(reference_cda)
                ctx["aero_calculation_mode"] = "Use candidate vs reference"
                ctx["component_mode_aerodynamics"] = "Apply delta"
                ctx["delta_aero_cdA"] = suggested_delta_cda
                quantity_metric(st, "Suggested delta CdA", suggested_delta_cda, "cda", format_str="%.3f")
                st.success("Derived change from Proposed Aero vs Reference Aero is active.")
            else:
                ctx["aero_calculation_mode"] = "Use candidate vs reference"
                ctx["component_mode_aerodynamics"] = "Keep inherited"
                ctx["delta_aero_cdA"] = 0.0
                if candidate_cda is None:
                    st.caption("No proposed CdA is staged right now.")
                else:
                    st.warning("Reference Aero is required before a proposed CdA can generate an active delta.")
        else:
            ctx["aero_candidate_mode"] = "Not set"
            ctx["aero_calculation_mode"] = "Manual delta CdA"
            ctx["component_mode_aerodynamics"] = "Apply delta"
            ctx["delta_aero_cdA"] = quantity_input(
                st,
                "Aerodynamics delta CdA",
                to_float(ctx.get("delta_aero_cdA"), 0.0),
                "cda",
                key="aero_delta_cda_input",
                step_canonical=0.001,
                format_str="%.3f",
            )
        st.caption("The inherited/coastdown workflow currently applies aerodynamics through explicit delta CdA only.")
    applied_method = str(ctx.get("aero_calculation_mode") or "Inherited")
    delta_cda = float(to_float(ctx.get("delta_aero_cdA"), 0.0) or 0.0)
    resolved_new_cda = None
    if applied_method == "Use candidate vs reference":
        resolved_new_cda = to_float(ctx.get("aero_candidate_cda"))
    elif reference_cda is not None and ctx.get("component_mode_aerodynamics") == "Apply delta":
        resolved_new_cda = float(reference_cda) + float(delta_cda)

    _render_applied_effect_block(
        title="Applied Aero State",
        caption="Read-only summary of the aerodynamic effect currently resolved for this scenario.",
        inherited_message="Method: inherited. No aerodynamic change is currently affecting the scenario.",
        metrics=[
            ("Method", applied_method),
            ("Effect applied", format_quantity(delta_cda, "cda", format_str="%.3f")),
            ("New CdA", format_quantity(resolved_new_cda, "cda", format_str="%.3f", unavailable="Pending")),
            ("Status", "Applied" if abs(delta_cda) > 0.0 or applied_method == 'Use candidate vs reference' else "Pending"),
        ] if ctx.get("component_mode_aerodynamics") == "Apply delta" else None,
    )


def render_vehicle_aero_section(*, base_row: dict | None = None):
    _render_aero_component_editor(base_row=base_row)


def _render_brake_component_editor(*, base_row: dict | None = None):
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    if basis == "Component Build-up":
        with st.container(border=True):
            _render_tire_editor_block_header(
                "Reference Brake Drag",
                "In component build-up, brake drag contributes as an absolute source instead of an inherited delta.",
            )
            st.caption("Reference status: `Absolute source`")
            render_brake_section(prefill=base_row)
        return
    base = dict(base_row or {})
    ref_a = to_float(ctx.get("brake_reference_A_override"), base.get("brake_A_coef_N"))
    ref_b = to_float(ctx.get("brake_reference_B_override"), base.get("brake_B_Npkph"))
    ref_c = to_float(ctx.get("brake_reference_C_override"), base.get("brake_C_coef_Npkph2"))
    baseline_target_id = int(base.get("id") or ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) or None
    reference_defined = any(abs(float(to_float(value, 0.0) or 0.0)) > 0.0 for value in (ref_a, ref_b, ref_c))
    reference_status = "Scenario override" if any(ctx.get(key) is not None for key in ("brake_reference_A_override", "brake_reference_B_override", "brake_reference_C_override")) else "Inherited"

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Reference Brake Drag",
            "Defines the current brake drag reference used when roadload is inherited from baseline or coastdown.",
        )
        if any(value is not None for value in (ref_a, ref_b, ref_c)):
            _render_abc_reference_metrics(
                title="Reference brake drag",
                a_value=ref_a,
                b_value=ref_b,
                c_value=ref_c,
                status=reference_status,
                caption="Current inherited workflow only consumes brake delta through A when baseline/coastdown is active.",
            )
        else:
            st.warning("Reference status: Missing. No explicit baseline brake split is available for derived comparison.")
        if baseline_target_id:
            with st.expander("Set / update baseline reference brake drag", expanded=not reference_defined):
                b1, b2, b3 = st.columns(3)
                baseline_ref_a = quantity_input(b1, "Baseline brake A", to_float(ctx.get("brake_baseline_reference_A"), ref_a or 0.0), "force", key="brake_baseline_reference_A_input", step_canonical=0.1, format_str="%.3f")
                baseline_ref_b = quantity_input(b2, "Baseline brake B", to_float(ctx.get("brake_baseline_reference_B"), ref_b or 0.0), "force_per_speed", key="brake_baseline_reference_B_input", step_canonical=0.0001, format_str="%.6f")
                baseline_ref_c = quantity_input(b3, "Baseline brake C", to_float(ctx.get("brake_baseline_reference_C"), ref_c or 0.0), "force_per_speed_squared", key="brake_baseline_reference_C_input", step_canonical=0.000001, format_str="%.8f")
                ctx["brake_baseline_reference_A"] = baseline_ref_a
                ctx["brake_baseline_reference_B"] = baseline_ref_b
                ctx["brake_baseline_reference_C"] = baseline_ref_c
                update_baseline = st.checkbox(
                    "Also update selected baseline row in VDE DB",
                    value=not reference_defined,
                    key="brake_update_baseline_checkbox",
                )
                if st.button("Apply reference brake drag", key="brake_apply_reference_button"):
                    ctx["brake_reference_A_override"] = float(baseline_ref_a)
                    ctx["brake_reference_B_override"] = float(baseline_ref_b)
                    ctx["brake_reference_C_override"] = float(baseline_ref_c)
                    if update_baseline:
                        try:
                            update_vde_snapshot(
                                int(baseline_target_id),
                                {
                                    "brake_A_coef_N": float(baseline_ref_a),
                                    "brake_B_Npkph": float(baseline_ref_b),
                                    "brake_C_coef_Npkph2": float(baseline_ref_c),
                                },
                            )
                            _sync_baseline_abc_reference_in_ctx(
                                a_key="brake_A_coef_N",
                                b_key="brake_B_Npkph",
                                c_key="brake_C_coef_Npkph2",
                                a_value=float(baseline_ref_a),
                                b_value=float(baseline_ref_b),
                                c_value=float(baseline_ref_c),
                                baseline_target_id=baseline_target_id,
                            )
                            ctx["brake_reference_A_override"] = None
                            ctx["brake_reference_B_override"] = None
                            ctx["brake_reference_C_override"] = None
                            st.success(f"Baseline reference brake drag updated on VDE id={baseline_target_id}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not update baseline reference brake drag: {exc}")
                    else:
                        st.success("Reference brake drag staged only for the current scenario context.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Change",
            "Choose whether brake drag stays inherited, is calculated from a new brake-drag proposal, or uses a direct delta A.",
        )
        new_brake_label = "New brake drag"
        brake_change_options = ["Keep inherited", new_brake_label, "Manual delta A"]
        current_component_mode = str(ctx.get("component_mode_brakes") or "Keep inherited")
        current_candidate_mode = str(ctx.get("brake_candidate_mode") or "Not set")
        current_calc_mode = str(ctx.get("brake_calculation_mode") or "Inherited")
        if current_component_mode != "Apply delta" or current_calc_mode == "Inherited":
            current_brake_change = "Keep inherited"
        elif current_candidate_mode == "Manual proposal" or current_calc_mode == "Use candidate vs reference":
            current_brake_change = new_brake_label
        else:
            current_brake_change = "Manual delta A"
        if current_brake_change not in brake_change_options:
            current_brake_change = "Keep inherited"

        selected_brake_change = st.radio(
            "Brake drag change",
            brake_change_options,
            horizontal=True,
            index=brake_change_options.index(current_brake_change),
            key="brake_change_mode_radio",
        )

        if selected_brake_change == "Keep inherited":
            ctx["brake_candidate_mode"] = "Not set"
            ctx["brake_calculation_mode"] = "Inherited"
            ctx["component_mode_brakes"] = "Keep inherited"
            ctx["delta_brake_N"] = 0.0
            st.info("Calculation status: inherited. No brake-drag change is currently applied.")
        elif selected_brake_change == new_brake_label:
            ctx["brake_candidate_mode"] = "Manual proposal"
            candidate = _render_candidate_abc_inputs(
                key_prefix="brake_candidate",
                title="New brake drag ABC",
            )
            ctx["brake_calculation_mode"] = "Use candidate vs reference"
            if candidate and ref_a is not None:
                suggested_delta_a = float(candidate.get("A") or 0.0) - float(ref_a or 0.0)
                ctx["component_mode_brakes"] = "Apply delta"
                ctx["delta_brake_N"] = suggested_delta_a
                quantity_metric(st, "Applied brake delta A", suggested_delta_a, "force", format_str="%.3f")
                st.success("Derived change from new brake drag vs reference is active.")
            elif candidate and ref_a is None:
                ctx["component_mode_brakes"] = "Keep inherited"
                ctx["delta_brake_N"] = 0.0
                st.warning("New brake drag is staged, but derived brake delta needs an explicit Reference Brake Drag. Use manual delta A instead.")
            else:
                ctx["component_mode_brakes"] = "Keep inherited"
                ctx["delta_brake_N"] = 0.0
                st.caption("New brake drag is not fully defined yet.")
        else:
            ctx["brake_candidate_mode"] = "Not set"
            ctx["brake_calculation_mode"] = "Manual delta A"
            ctx["component_mode_brakes"] = "Apply delta"
            ctx["delta_brake_N"] = quantity_input(
                st,
                "Brakes delta A",
                to_float(ctx.get("delta_brake_N"), 0.0),
                "force",
                key="brake_delta_input",
                step_canonical=0.1,
                format_str="%.3f",
            )
        st.caption("Current inherited/coastdown workflow uses brake drag adjustment through A only.")
    delta_brake = float(to_float(ctx.get("delta_brake_N"), 0.0) or 0.0)
    _render_applied_effect_block(
        title="Applied effect",
        caption="Read-only summary of the brake-drag effect currently resolved for this scenario.",
        inherited_message="Method: inherited. No brake-drag change is currently affecting the scenario.",
        metrics=[
            ("Method", str(ctx.get("brake_calculation_mode") or "Inherited")),
            ("Effect applied", format_quantity(delta_brake, "force", format_str="%.3f")),
            ("Status", "Applied" if abs(delta_brake) > 0.0 or str(ctx.get("brake_calculation_mode") or "") == "Use candidate vs reference" else "Pending"),
        ] if ctx.get("component_mode_brakes") == "Apply delta" else None,
    )


def _render_parasitic_component_editor(*, base_row: dict | None = None):
    ctx = st.session_state.ctx
    basis = _roadload_basis_value(ctx)
    if basis == "Component Build-up":
        with st.container(border=True):
            _render_tire_editor_block_header(
                "Reference Losses",
                "In component build-up, parasitic losses contribute as an absolute source instead of an inherited delta.",
            )
            st.caption("Reference status: `Absolute source`")
            render_parasitic_section(prefill=base_row)
        return
    base = dict(base_row or {})
    ref_a = to_float(ctx.get("parasitic_reference_A_override"), base.get("parasitic_A_coef_N"))
    ref_b = to_float(ctx.get("parasitic_reference_B_override"), base.get("parasitic_B_Npkph"))
    ref_c = to_float(ctx.get("parasitic_reference_C_override"), base.get("parasitic_C_coef_Npkph2"))
    baseline_target_id = int(base.get("id") or ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) or None
    reference_defined = any(abs(float(to_float(value, 0.0) or 0.0)) > 0.0 for value in (ref_a, ref_b, ref_c))
    reference_status = "Scenario override" if any(ctx.get(key) is not None for key in ("parasitic_reference_A_override", "parasitic_reference_B_override", "parasitic_reference_C_override")) else "Inherited"

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Reference Losses",
            "Defines the current parasitic-loss reference used when roadload is inherited from baseline or coastdown.",
        )
        if any(value is not None for value in (ref_a, ref_b, ref_c)):
            _render_abc_reference_metrics(
                title="Reference parasitic losses",
                a_value=ref_a,
                b_value=ref_b,
                c_value=ref_c,
                status=reference_status,
                caption="Current inherited/coastdown workflow only consumes parasitic adjustment through A.",
            )
        else:
            st.warning("Reference status: Missing. No explicit baseline parasitic split is available for derived comparison.")
        if baseline_target_id:
            with st.expander("Set / update baseline reference parasitic losses", expanded=not reference_defined):
                p1, p2, p3 = st.columns(3)
                baseline_ref_a = quantity_input(p1, "Baseline parasitic A", to_float(ctx.get("parasitic_baseline_reference_A"), ref_a or 0.0), "force", key="parasitic_baseline_reference_A_input", step_canonical=0.1, format_str="%.3f")
                baseline_ref_b = quantity_input(p2, "Baseline parasitic B", to_float(ctx.get("parasitic_baseline_reference_B"), ref_b or 0.0), "force_per_speed", key="parasitic_baseline_reference_B_input", step_canonical=0.0001, format_str="%.6f")
                baseline_ref_c = quantity_input(p3, "Baseline parasitic C", to_float(ctx.get("parasitic_baseline_reference_C"), ref_c or 0.0), "force_per_speed_squared", key="parasitic_baseline_reference_C_input", step_canonical=0.000001, format_str="%.8f")
                ctx["parasitic_baseline_reference_A"] = baseline_ref_a
                ctx["parasitic_baseline_reference_B"] = baseline_ref_b
                ctx["parasitic_baseline_reference_C"] = baseline_ref_c
                update_baseline = st.checkbox(
                    "Also update selected baseline row in VDE DB",
                    value=not reference_defined,
                    key="parasitic_update_baseline_checkbox",
                )
                if st.button("Apply reference parasitic losses", key="parasitic_apply_reference_button"):
                    ctx["parasitic_reference_A_override"] = float(baseline_ref_a)
                    ctx["parasitic_reference_B_override"] = float(baseline_ref_b)
                    ctx["parasitic_reference_C_override"] = float(baseline_ref_c)
                    if update_baseline:
                        try:
                            update_vde_snapshot(
                                int(baseline_target_id),
                                {
                                    "parasitic_A_coef_N": float(baseline_ref_a),
                                    "parasitic_B_Npkph": float(baseline_ref_b),
                                    "parasitic_C_coef_Npkph2": float(baseline_ref_c),
                                },
                            )
                            _sync_baseline_abc_reference_in_ctx(
                                a_key="parasitic_A_coef_N",
                                b_key="parasitic_B_Npkph",
                                c_key="parasitic_C_coef_Npkph2",
                                a_value=float(baseline_ref_a),
                                b_value=float(baseline_ref_b),
                                c_value=float(baseline_ref_c),
                                baseline_target_id=baseline_target_id,
                            )
                            ctx["parasitic_reference_A_override"] = None
                            ctx["parasitic_reference_B_override"] = None
                            ctx["parasitic_reference_C_override"] = None
                            st.success(f"Baseline reference parasitic losses updated on VDE id={baseline_target_id}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not update baseline reference parasitic losses: {exc}")
                    else:
                        st.success("Reference parasitic losses staged only for the current scenario context.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Change",
            "Choose whether parasitic losses stay inherited, are calculated from a new proposal, or use a direct delta A.",
        )
        new_parasitic_label = "New parasitic losses"
        parasitic_change_options = ["Keep inherited", new_parasitic_label, "Manual delta A"]
        current_component_mode = str(ctx.get("component_mode_parasitics_hubs_axle") or "Keep inherited")
        current_candidate_mode = str(ctx.get("parasitic_candidate_mode") or "Not set")
        current_calc_mode = str(ctx.get("parasitic_calculation_mode") or "Inherited")
        if current_component_mode != "Apply delta" or current_calc_mode == "Inherited":
            current_parasitic_change = "Keep inherited"
        elif current_candidate_mode == "Manual proposal" or current_calc_mode == "Use candidate vs reference":
            current_parasitic_change = new_parasitic_label
        else:
            current_parasitic_change = "Manual delta A"
        if current_parasitic_change not in parasitic_change_options:
            current_parasitic_change = "Keep inherited"

        selected_parasitic_change = st.radio(
            "Parasitic losses change",
            parasitic_change_options,
            horizontal=True,
            index=parasitic_change_options.index(current_parasitic_change),
            key="parasitic_change_mode_radio",
        )

        if selected_parasitic_change == "Keep inherited":
            ctx["parasitic_candidate_mode"] = "Not set"
            ctx["parasitic_calculation_mode"] = "Inherited"
            ctx["component_mode_parasitics_hubs_axle"] = "Keep inherited"
            ctx["delta_parasitics_N"] = 0.0
            st.info("Calculation status: inherited. No parasitic-loss change is currently applied.")
        elif selected_parasitic_change == new_parasitic_label:
            ctx["parasitic_candidate_mode"] = "Manual proposal"
            candidate = _render_candidate_abc_inputs(
                key_prefix="parasitic_candidate",
                title="New parasitic-loss ABC",
            )
            ctx["parasitic_calculation_mode"] = "Use candidate vs reference"
            if candidate and ref_a is not None:
                suggested_delta_a = float(candidate.get("A") or 0.0) - float(ref_a or 0.0)
                ctx["component_mode_parasitics_hubs_axle"] = "Apply delta"
                ctx["delta_parasitics_N"] = suggested_delta_a
                quantity_metric(st, "Applied parasitic delta A", suggested_delta_a, "force", format_str="%.3f")
                st.success("Derived change from new parasitic losses vs reference is active.")
            elif candidate and ref_a is None:
                ctx["component_mode_parasitics_hubs_axle"] = "Keep inherited"
                ctx["delta_parasitics_N"] = 0.0
                st.warning("New parasitic losses are staged, but derived parasitic delta needs an explicit Reference Losses setup. Use manual delta A instead.")
            else:
                ctx["component_mode_parasitics_hubs_axle"] = "Keep inherited"
                ctx["delta_parasitics_N"] = 0.0
                st.caption("New parasitic losses are not fully defined yet.")
        else:
            ctx["parasitic_candidate_mode"] = "Not set"
            ctx["parasitic_calculation_mode"] = "Manual delta A"
            ctx["component_mode_parasitics_hubs_axle"] = "Apply delta"
            ctx["delta_parasitics_N"] = quantity_input(
                st,
                "Parasitics delta A",
                to_float(ctx.get("delta_parasitics_N"), 0.0),
                "force",
                key="parasitic_delta_input",
                step_canonical=0.1,
                format_str="%.3f",
            )
        st.caption("Current inherited/coastdown workflow uses parasitic adjustment through A only.")
    delta_parasitics = float(to_float(ctx.get("delta_parasitics_N"), 0.0) or 0.0)
    _render_applied_effect_block(
        title="Applied effect",
        caption="Read-only summary of the parasitic-loss effect currently resolved for this scenario.",
        inherited_message="Method: inherited. No parasitic-loss change is currently affecting the scenario.",
        metrics=[
            ("Method", str(ctx.get("parasitic_calculation_mode") or "Inherited")),
            ("Effect applied", format_quantity(delta_parasitics, "force", format_str="%.3f")),
            ("Status", "Applied" if abs(delta_parasitics) > 0.0 or str(ctx.get("parasitic_calculation_mode") or "") == "Use candidate vs reference" else "Pending"),
        ] if ctx.get("component_mode_parasitics_hubs_axle") == "Apply delta" else None,
    )


def render_component_build_up_panel(*, base_row: dict | None = None, saved_vde_id: int | None = None, tires_df=None):
    ctx = st.session_state.ctx
    options = ["Tires", "Brakes", "Parasitics / Hubs / Axle", "Trailer"]
    active = str(ctx.get("component_editor_active") or "Tires")
    if active not in options:
        active = "Tires"
    ctx["component_editor_active"] = active

    if _current_vde_input_mode(ctx) == "Spreadsheet":
        st.caption("Spreadsheet mode edits the active TOTAL components in one compact table. Tire DB still uses the existing preview flow and feeds the Tires row.")
        component_df = _build_component_spreadsheet_df(ctx)
        editor_key = f"vde_component_spreadsheet_{str(ctx.get('mode') or 'default').replace(' ', '_')}"
        edited_df = st.data_editor(
            component_df,
            key=editor_key,
            hide_index=True,
            use_container_width=True,
            disabled=["component", "status", "notes"],
            column_config={
                "component": st.column_config.TextColumn("component"),
                "source": st.column_config.SelectboxColumn(
                    "source",
                    options=["Manual RR", "Tire DB", "Manual", "Reserved"],
                ),
                "A": st.column_config.NumberColumn(f"A [{unit_label('force')}]", format="%.6f"),
                "B": st.column_config.NumberColumn(f"B [{unit_label('force_per_speed')}]", format="%.6f"),
                "C": st.column_config.NumberColumn(f"C [{unit_label('force_per_speed_squared')}]", format="%.6f"),
                "apply": st.column_config.CheckboxColumn("apply"),
                "status": st.column_config.TextColumn("status"),
                "notes": st.column_config.TextColumn("notes"),
            },
        )
        st.session_state["vde_component_table"] = edited_df.copy()
        errors = _apply_component_spreadsheet_changes(edited_df)
        for error in errors:
            st.warning(error)
        _render_component_spreadsheet_tire_db_block(base_row=base_row, saved_vde_id=saved_vde_id, tires_df=tires_df)
    else:
        _render_component_overview_table(ctx)
        st.divider()

        with st.container(border=True):
            if ctx["component_editor_active"] == "Tires":
                _render_tire_component_editor(base_row=base_row, saved_vde_id=saved_vde_id, tires_df=tires_df)
            elif ctx["component_editor_active"] == "Brakes":
                _render_brake_component_editor(base_row=base_row)
            elif ctx["component_editor_active"] == "Parasitics / Hubs / Axle":
                _render_parasitic_component_editor(base_row=base_row)
            else:
                st.info("Trailer stays as a reserved slot in Sprint 5. The visual slot exists now so future component DBs can follow the same pattern.")

    summary = summarize_component_build_up_from_ctx(ctx)
    if summary.get("enabled"):
        abc_total = dict(summary.get("abc_total") or {})
        ctx["A"] = float(to_float(abc_total.get("A"), 0.0) or 0.0)
        ctx["B"] = float(to_float(abc_total.get("B"), 0.0) or 0.0)
        ctx["C"] = float(to_float(abc_total.get("C"), 0.0) or 0.0)

        st.caption("Current ABC_TOTAL resolved from Component Build-up")
        s1, s2, s3 = st.columns(3)
        quantity_metric(s1, "A_TOTAL", ctx["A"], "force", format_str="%.2f")
        quantity_metric(s2, "B_TOTAL", ctx["B"], "force_per_speed", format_str="%.5f")
        quantity_metric(s3, "C_TOTAL", ctx["C"], "force_per_speed_squared", format_str="%.6f")

        component_rows = [
            {
                "component": component.get("name"),
                "source": component.get("source"),
                "A": float(to_float(component.get("A"), 0.0) or 0.0),
                "B": float(to_float(component.get("B"), 0.0) or 0.0),
                "C": float(to_float(component.get("C"), 0.0) or 0.0),
            }
            for component in summary.get("components") or []
        ]
        if component_rows:
            with st.expander("Resolved components", expanded=False):
                st.dataframe(pd.DataFrame(component_rows), use_container_width=True, hide_index=True)

def render_baseline_picker_and_editor_panel():
    ctx = st.session_state.ctx

    try:
        rows = fetch_vde_rows_full()
    except Exception as e:
        st.error(f"Could not read vde_db: {e}")
        return

    if not rows:
        st.info("No snapshots in vde_db yet. Save a scenario first before trying to inherit from baseline.")
        return

    df = ensure_baseline_aliases(pd.DataFrame(rows))

    filter_opts = baseline_filter_options(df)
    leg = st.session_state.get("vde_baseline_filter_legislation", filter_opts["legislation"][0] if filter_opts["legislation"] else "(all)")
    make = st.session_state.get("vde_baseline_filter_make", filter_opts["make"][0] if filter_opts["make"] else "(all)")
    cat_contains = str(st.session_state.get("vde_baseline_filter_category_contains") or "")
    year_eq = str(st.session_state.get("vde_baseline_filter_year_eq") or "")

    dfv = apply_baseline_filters(
        df,
        legislation=leg,
        make=make,
        category_contains=cat_contains,
        year_eq=year_eq,
    )

    if dfv.empty:
        st.warning("No rows after filters.")
        with st.expander("Show raw columns (debug)"):
            st.write(sorted(df.columns.tolist()))
        return

    cols_to_show = [
        "id", "created_at", "updated_at",
        "legislation", "category", "make", "model", "year", "notes",
        "engine_type", "engine_model", "engine_size_l", "engine_aspiration",
        "transmission_type", "transmission_model", "drive_type",
        "mass_kg", "test_mass_kg", "test_mass_low_kg", "test_mass_high_kg", "test_mass_basis", "inertia_class", "cda_m2", "weight_dist_fr_pct", "payload_kg",
        "mro_kg", "options_kg", "wltp_category",
        "tire_size", "tire_rr_note", "smerf", "front_pressure_psi", "rear_pressure_psi",
        "rrc_N_per_kN", "crr1_frac_at_120kph", "rr_load_kpa",
        "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2",
        "trans_A_coef_N", "trans_B_Npkph", "trans_C_coef_Npkph2",
        "brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2",
        "aero_C_coef_Npkph2",
        "rr_alpha_N", "rr_beta_Npkph", "rr_a_Npkph2", "rr_b_N", "rr_c_Npkph",
        "cycle_name", "cycle_source",
        "vde_urb_mj", "vde_hw_mj",
        "vde_net_mj_per_km", "vde_total_mj_per_km",
        "vde_urb_mj_per_km", "vde_hw_mj_per_km",
        "vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km", "vde_extra_high_mj_per_km",
        "vde_id_parent", "baseline_A_N", "baseline_B_N_per_kph", "baseline_C_N_per_kph2", "baseline_mass_kg",
        "delta_rr_N", "delta_brake_N", "delta_parasitics_N", "delta_aero_Npkph2", "delta_mass_kg",
    ]
    cols_to_show = [c for c in cols_to_show if c in dfv.columns]
    options_df = dfv.sort_values("id", ascending=False).copy()
    options_df["baseline_label"] = options_df.apply(
        lambda row: (
            f"VDE-{int(row['id']):03d} | "
            f"{str(row.get('make') or '').upper()} "
            f"{str(row.get('model') or '')} | "
            f"{str(row.get('year') or '')} | "
            f"{str(row.get('legislation') or '')}"
        ),
        axis=1,
    )
    option_map = {row["baseline_label"]: int(row["id"]) for _, row in options_df.iterrows()}
    labels = list(option_map.keys())
    default_label = labels[0]
    selected_id = int(ctx.get("vde_id_parent") or ctx.get("baseline_id") or option_map[default_label])
    for label, row_id in option_map.items():
        if row_id == selected_id:
            default_label = label
            break

    c_pick, c_hint = st.columns([2, 1])
    sel_label = c_pick.selectbox("Select baseline scenario", labels, index=labels.index(default_label))
    sel_id = option_map[sel_label]
    c_hint.caption("Baseline scenarios follow the active filters below.")

    base = dfv[dfv["id"] == sel_id].iloc[0].to_dict()
    previous_baseline_id = to_float(ctx.get("baseline_id"))
    previous_autofill = str(ctx.get("metadata_proposal_autofill") or "").strip()
    current_notes = str(ctx.get("notes") or "").strip()

    st.session_state.ctx.update(build_baseline_state_payload(base, int(sel_id)))
    st.session_state.ctx["selected_baseline_row"] = base
    baseline_seed = dict(st.session_state.ctx.get("baseline_dict") or {})
    for key in (
        "legislation",
        "category",
        "make",
        "model",
        "year",
        "electrification",
        "transmission_type",
        "drive_type",
        "fuel_type",
    ):
        value = baseline_seed.get(key)
        if value not in (None, "", []):
            st.session_state.ctx[key] = value
    baseline_notes = str(baseline_seed.get("notes") or "").strip()
    scenario_notes = _suggest_scenario_notes_from_baseline(baseline_seed, int(sel_id))
    baseline_changed = previous_baseline_id is None or int(previous_baseline_id) != int(sel_id)
    if baseline_changed or not current_notes or current_notes == previous_autofill or current_notes == baseline_notes:
        st.session_state.ctx["notes"] = scenario_notes
    st.session_state.ctx["metadata_proposal_autofill"] = scenario_notes

    summary1, summary2, summary3, summary4 = st.columns(4)
    summary1.metric("Baseline ID", f"{int(base.get('id', 0))}")
    summary2.metric(
        f"ABC_TOTAL [{_abc_unit_triplet_label()}]",
        _compact_abc({"A": base.get("A"), "B": base.get("B"), "C": base.get("C")}),
    )
    quantity_metric(summary3, "Mass", base.get("mass_kg"), "mass", format_str="%.0f")
    quantity_metric(summary4, "Test mass", base.get("test_mass_kg"), "mass", format_str="%.0f")

    with st.expander("Scenario filters and matching baseline scenarios", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        leg = f1.selectbox(
            "Legislation",
            filter_opts["legislation"],
            key="vde_baseline_filter_legislation",
        )
        make = f2.selectbox(
            "Make",
            filter_opts["make"],
            key="vde_baseline_filter_make",
        )
        cat_contains = f3.text_input(
            "Category contains",
            key="vde_baseline_filter_category_contains",
        )
        year_eq = f4.text_input(
            "Year (=)",
            key="vde_baseline_filter_year_eq",
        )
        st.caption("Matching baseline scenarios")
        display_df = options_df[cols_to_show + ["baseline_label"]] if "baseline_label" not in cols_to_show else options_df[cols_to_show]
        display_df = display_df.copy()
        rename_map = {}
        if "mass_kg" in display_df.columns:
            display_df["mass_kg"] = display_df["mass_kg"].apply(lambda value: to_display(value, "mass"))
            rename_map["mass_kg"] = f"mass [{unit_label('mass')}]"
        if "test_mass_kg" in display_df.columns:
            display_df["test_mass_kg"] = display_df["test_mass_kg"].apply(lambda value: to_display(value, "mass"))
            rename_map["test_mass_kg"] = f"test_mass [{unit_label('mass')}]"
        if "test_mass_low_kg" in display_df.columns:
            display_df["test_mass_low_kg"] = display_df["test_mass_low_kg"].apply(lambda value: to_display(value, "mass"))
            rename_map["test_mass_low_kg"] = f"test_mass_low [{unit_label('mass')}]"
        if "test_mass_high_kg" in display_df.columns:
            display_df["test_mass_high_kg"] = display_df["test_mass_high_kg"].apply(lambda value: to_display(value, "mass"))
            rename_map["test_mass_high_kg"] = f"test_mass_high [{unit_label('mass')}]"
        if "coast_A_N" in display_df.columns:
            display_df["coast_A_N"] = display_df["coast_A_N"].apply(lambda value: to_display(value, "force"))
            rename_map["coast_A_N"] = f"coast_A [{unit_label('force')}]"
        if "coast_B_N_per_kph" in display_df.columns:
            display_df["coast_B_N_per_kph"] = display_df["coast_B_N_per_kph"].apply(lambda value: to_display(value, "force_per_speed"))
            rename_map["coast_B_N_per_kph"] = f"coast_B [{unit_label('force_per_speed')}]"
        if "coast_C_N_per_kph2" in display_df.columns:
            display_df["coast_C_N_per_kph2"] = display_df["coast_C_N_per_kph2"].apply(lambda value: to_display(value, "force_per_speed_squared"))
            rename_map["coast_C_N_per_kph2"] = f"coast_C [{unit_label('force_per_speed_squared')}]"
        if "vde_net_mj_per_km" in display_df.columns:
            display_df["vde_net_mj_per_km"] = display_df["vde_net_mj_per_km"].apply(lambda value: to_display(value, "energy_per_distance"))
            rename_map["vde_net_mj_per_km"] = f"vde_net [{unit_label('energy_per_distance')}]"
        if "vde_total_mj_per_km" in display_df.columns:
            display_df["vde_total_mj_per_km"] = display_df["vde_total_mj_per_km"].apply(lambda value: to_display(value, "energy_per_distance"))
            rename_map["vde_total_mj_per_km"] = f"vde_total [{unit_label('energy_per_distance')}]"
        if rename_map:
            display_df = display_df.rename(columns=rename_map)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    source_ui = str(ctx.get("abc_total_source_ui") or "Baseline ABC")
    if source_ui not in {"Baseline ABC", "From test coastdown", "Component Build-up"}:
        source_ui = "Baseline ABC"
        ctx["abc_total_source_ui"] = source_ui

    if source_ui == "Baseline ABC":
        ctx["from_delta"] = "Deltas"
        ctx.update(build_delta_mode_ctx_updates(base))
        st.info("Baseline scenario loaded. Roadload Basis in Roadload Build-up now decides whether ABC_TOTAL stays inherited, switches to measured/test coastdown, or is rebuilt from components.")
    elif source_ui == "Component Build-up":
        ctx["from_delta"] = "Change Parameters"
        st.success(f"Editing baseline #{base.get('id', '')} with full parameter override.")
        if str(ctx.get("legislation", "")).strip().upper() == "WLTP":
            st.info(
                "WLTP baseline path currently uses test mass for road/tire calculation. "
                "The code already has hooks for MRO/TPMLM-based WLTP mass resolution, "
                "but vehicle-type/cargo and TPMLM UI inputs are still a placeholder."
            )
        st.caption("Roadload component editors are available in Roadload Build-up.")
    else:
        ctx["from_delta"] = "From test"
        st.info("Baseline row remains selected as scenario context, but ABC_TOTAL will be replaced by measured coastdown inputs in Roadload Build-up.")

    with st.expander("Inherited baseline snapshot", expanded=False):
        key_cols = [
            "id", "legislation", "category", "make", "model", "year", "mass_kg", "test_mass_kg", "A", "B", "C",
            "vde_net_mj_per_km", "vde_urb_mj_per_km", "vde_hw_mj_per_km",
            "vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km", "vde_extra_high_mj_per_km",
        ]
        snapshot = {k: base.get(k) for k in key_cols if k in base}
        if "mass_kg" in snapshot:
            snapshot["mass"] = format_quantity(snapshot.pop("mass_kg"), "mass", format_str="%.0f")
        if "test_mass_kg" in snapshot:
            snapshot["test_mass"] = format_quantity(snapshot.pop("test_mass_kg"), "mass", format_str="%.0f")
        if {"A", "B", "C"} <= set(snapshot.keys()):
            snapshot["ABC_TOTAL"] = _compact_abc(snapshot)
        if "vde_net_mj_per_km" in snapshot:
            snapshot["vde_net"] = format_quantity(snapshot.pop("vde_net_mj_per_km"), "energy_per_distance", format_str="%.4f")
        if "vde_urb_mj_per_km" in snapshot:
            snapshot["vde_urb"] = format_quantity(snapshot.pop("vde_urb_mj_per_km"), "energy_per_distance", format_str="%.4f")
        if "vde_hw_mj_per_km" in snapshot:
            snapshot["vde_hw"] = format_quantity(snapshot.pop("vde_hw_mj_per_km"), "energy_per_distance", format_str="%.4f")
        st.write(snapshot)


def render_compute_and_save_panel(*, defaults_df_getter, reset_ctx):
    ctx = st.session_state.ctx
    flash = st.session_state.pop("vde_setup_save_flash", None)
    if isinstance(flash, dict):
        st.success(str(flash.get("message") or "VDE snapshot persisted."))
        total_label = flash.get("vde_total")
        net_label = flash.get("vde_net")
        if total_label:
            st.info(f"Saved VDE_TOTAL: **{total_label}**")
        if net_label:
            st.info(f"Saved VDE_NET: **{net_label}**")

    st.caption(
        "Save uses the Sprint 5 workflow contract and still preserves the legacy snapshot fields needed by the current DB."
    )

    errs, warns = validate_core(ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
    for warning in (warns or []):
        st.warning(warning)
    spreadsheet_errors = _spreadsheet_validation_errors(ctx) if _current_vde_input_mode(ctx) == "Spreadsheet" else []
    for error in spreadsheet_errors:
        st.warning(error)
    errs.extend(spreadsheet_errors)
    if ctx.get("cycle_df") is None:
        errs.append("Cycle not loaded. Pick default or upload a CSV.")
    for error in (errs or []):
        st.error(error)
    disabled_actions = bool(errs)
    workflow_preview = None
    target_vde_id = None

    if not disabled_actions:
        try:
            workflow_preview = build_vde_setup_preview_from_ctx(ctx)
        except Exception as e:
            workflow_preview = {"ok": False, "error": f"Workflow preview not available: {e}"}
        if not workflow_preview.get("ok"):
            st.error(workflow_preview.get("error") or "Workflow preview not available.")
            disabled_actions = True
        else:
            save_payload = dict(workflow_preview.get("save_payload") or {})
            target_vde_id = save_payload.get("target_vde_id") or ctx.get("vde_id_parent") or ctx.get("baseline_id")
            preview_line_source = dict(workflow_preview.get("line_source") or {})
            preview_total = dict(workflow_preview.get("vde_total") or {})
            preview_net = dict(workflow_preview.get("vde_net") or {})
            preview_mass = dict(workflow_preview.get("mass_setup") or {})

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Scenario origin", _line_source_summary(ctx))
            s2.metric("Roadload basis", _total_source_summary(ctx, workflow_preview))
            s3.metric("Preview VDE_TOTAL", _format_energy_value(preview_total.get("mj_per_km"), unavailable="-"))
            s4.metric("Preview VDE_NET", _format_energy_value(preview_net.get("mj_per_km")))

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Update target", f"VDE-{int(target_vde_id):03d}" if target_vde_id is not None else "None")
            p2.metric("Cycle", str(ctx.get("cycle_name") or "-"))
            p3.metric("Mass basis", str(preview_mass.get("mass_basis") or "-"))
            quantity_metric(
                p4,
                "Resolved mass",
                preview_mass.get("resolved_mass_used_kg"),
                "mass",
                format_str="%.1f",
            )

            with st.expander("Snapshot / provenance", expanded=False):
                st.write(
                    {
                        "scenario_origin": _line_source_summary(ctx),
                        "roadload_basis": _total_source_summary(ctx, workflow_preview),
                        "update_target_vde_id": target_vde_id,
                        "line_source": preview_line_source,
                        "save_payload": workflow_preview.get("save_payload"),
                    }
                )

    update_confirm = st.checkbox(
        "Confirm update of selected baseline snapshot",
        value=False,
        disabled=disabled_actions or target_vde_id is None,
        key="vde_save_update_confirm",
    )

    action_cols = st.columns(2)
    save_as_new_clicked = action_cols[0].button(
        "Save as new snapshot",
        key="btn_compute_save_new",
        disabled=disabled_actions,
        use_container_width=True,
    )
    update_existing_clicked = action_cols[1].button(
        "Update existing snapshot",
        key="btn_compute_save_update",
        disabled=disabled_actions or target_vde_id is None or not update_confirm,
        use_container_width=True,
    )

    if target_vde_id is None and not disabled_actions:
        st.caption("No baseline target is selected, so only Save as new is available in this workflow.")
    else:
        st.caption("Use Save as new to preserve provenance as a separate row. Use Update existing only when you intentionally want to overwrite the selected baseline snapshot.")

    save_mode = None
    if save_as_new_clicked:
        save_mode = "insert_new"
    elif update_existing_clicked:
        save_mode = "update_existing"

    if save_mode:
        try:
            defaults_df = defaults_df_getter() if callable(defaults_df_getter) else None
            workflow_preview = workflow_preview or build_vde_setup_preview_from_ctx(ctx)
            saved = save_vde_setup_result(
                workflow_preview,
                save_mode,
                ctx=ctx,
                defaults_df=defaults_df,
            )
            vde_id = int(saved["vde_id"])
            row = dict(saved.get("row") or {})
            vde_total_mj_km = to_float(row.get("vde_total_mj_per_km"))
            vde_net_mj_km = to_float(row.get("vde_net_mj_per_km"))
            by_phase = dict((workflow_preview.get("vde_net") or workflow_preview.get("vde_total") or {}).get("by_phase") or {})

            total_label = format_quantity(vde_total_mj_km, "energy_per_distance") if vde_total_mj_km is not None else "-"
            net_label = format_quantity(vde_net_mj_km, "energy_per_distance", unavailable="Unavailable")
            st.info(f"Saved VDE_TOTAL: **{total_label}**")
            st.info(f"Saved VDE_NET: **{net_label}**")
            if by_phase:
                order = ["city", "hwy", "low", "mid", "high", "xhigh"]
                keys = [k for k in order if k in by_phase] + [k for k in by_phase if k not in order]
                cols = st.columns(min(4, len(keys)))
                for i, key in enumerate(keys):
                    label = {"city": "CITY", "hwy": "HWY"}.get(key, key.upper())
                    quantity_metric(cols[i % len(cols)], label, by_phase[key], "energy_per_distance", format_str="%.4f")
            st.session_state["vde_id"] = vde_id

            action_label = "Updated" if save_mode == "update_existing" else "Saved"
            st.session_state["vde_setup_save_flash"] = {
                "message": f"{action_label} VDE snapshot (id={vde_id}).",
                "vde_total": total_label,
                "vde_net": net_label,
            }
            reset_ctx(preserve_meta=True)
            st.rerun()

        except Exception as e:
            st.error(f"Failed to persist VDE snapshot: {e}")


def _render_phase_metrics(title: str, phases: dict | None) -> None:
    phase_dict = dict(phases or {})
    if not phase_dict:
        return

    st.caption(title)
    order = ["city", "hwy", "low", "mid", "high", "xhigh"]
    ordered = [key for key in order if key in phase_dict] + [key for key in phase_dict if key not in order]
    cols = st.columns(min(4, len(ordered)))
    for i, key in enumerate(ordered):
        quantity_metric(cols[i % len(cols)], key.upper(), phase_dict[key], "energy_per_distance", format_str="%.4f")


def _render_cycle_preview_snapshot() -> None:
    ctx = st.session_state.get("ctx", {})
    workflow_preview = _safe_workflow_preview(ctx)
    if not workflow_preview.get("ok"):
        st.info("Phase preview will appear here after the cycle and upstream inputs are valid.")
        return

    vde_total = dict(workflow_preview.get("vde_total") or {})
    vde_net = dict(workflow_preview.get("vde_net") or {})
    total_phases = dict(vde_total.get("by_phase") or {})
    net_phases = dict(vde_net.get("by_phase") or {})
    warnings = list(workflow_preview.get("warnings") or [])

    top1, top2, top3 = st.columns(3)
    top1.metric("Preview VDE_TOTAL", _format_energy_value(vde_total.get("mj_per_km"), unavailable="-"))
    top2.metric("Preview VDE_NET", _format_energy_value(vde_net.get("mj_per_km")))
    top3.metric("Cycle", str(ctx.get("cycle_name") or "-"))

    _render_preview_roadload_curves(workflow_preview.get("abc_total"), workflow_preview.get("abc_net"))

    phase_order = ["city", "hwy", "low", "mid", "high", "xhigh"]
    present_phases = [key for key in phase_order if key in total_phases or key in net_phases]
    present_phases += [key for key in total_phases if key not in present_phases]
    present_phases += [key for key in net_phases if key not in present_phases]

    if present_phases:
        rows = []
        labels = {
            "city": "URB / CITY",
            "hwy": "HWY",
            "low": "LOW",
            "mid": "MID",
            "high": "HIGH",
            "xhigh": "XHIGH",
        }
        for phase in present_phases:
            rows.append(
                {
                    "Phase": labels.get(phase, phase.upper()),
                    "VDE_TOTAL": _format_energy_value(total_phases.get(phase), unavailable="-"),
                    "VDE_NET": _format_energy_value(net_phases.get(phase)),
                }
            )
        st.caption("Phase Preview")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if warnings:
        st.warning("Preview warnings: " + ", ".join(warnings[:3]))


def _review_value_label(value) -> str:
    if isinstance(value, dict):
        if {"A", "B", "C"} & set(value.keys()):
            return _compact_abc(value)
        if "mj_per_km" in value:
            return _format_energy_value(value.get("mj_per_km"))
        return json.dumps(value, default=str)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    text = str(value or "").strip()
    return text or "-"


def _dict_to_rows(payload: dict | None) -> list[dict[str, str]]:
    data = dict(payload or {})
    rows = []
    for key, value in data.items():
        rows.append({"Field": str(key), "Value": _review_value_label(value)})
    return rows


def _save_mode_label(target_vde_id) -> str:
    if target_vde_id is None:
        return "Create new scenario"
    return "Create new scenario / Update existing VDE"


def render_vde_results_review_panel():
    ctx = st.session_state.get("ctx", {})
    component_build_up_active = _component_build_up_enabled(ctx)

    try:
        workflow_preview = build_vde_setup_preview_from_ctx(ctx)
    except Exception as exc:
        workflow_preview = {"ok": False, "error": f"Workflow preview not available: {exc}"}

    if not workflow_preview.get("ok"):
        st.warning(workflow_preview.get("error") or "Preview not available.")
        return

    save_payload = workflow_preview.get("save_payload") or {}
    review = build_vde_pre_save_review(ctx, workflow_preview, save_payload)
    performance_total = dict(workflow_preview.get("vde_total") or {})
    performance_net = dict(workflow_preview.get("vde_net") or {})
    warnings = list(workflow_preview.get("warnings") or [])
    working_summary = dict(review.get("working_scenario_summary") or {})
    change_summary = dict(review.get("change_summary") or {})
    reference_snapshot = dict(review.get("reference_snapshot") or {})
    baseline_rows = list(review.get("baseline_vs_working_rows") or [])
    staged_payload = dict(review.get("staged_save_payload") or {})
    phase_update_row = dict(workflow_preview.get("phase_update_row") or {})
    legacy_preview = build_live_vde_preview(ctx)

    with st.container(border=True):
        st.subheader("Preview Worksheet")
        p1, p2, p3 = st.columns(3)
        p1.metric("VDE_TOTAL", _format_energy_value(performance_total.get("mj_per_km"), unavailable="-"))
        p2.metric("VDE_NET", _format_energy_value(performance_net.get("mj_per_km")))
        p3.metric("Cycle", str(ctx.get("cycle_name") or working_summary.get("cycle_name") or "-"))

        phase_rows = []
        total_phases = dict(performance_total.get("by_phase") or {})
        net_phases = dict(performance_net.get("by_phase") or {})
        phase_order = ["city", "hwy", "low", "mid", "high", "xhigh"]
        present_phases = [key for key in phase_order if key in total_phases or key in net_phases]
        present_phases += [key for key in total_phases if key not in present_phases]
        present_phases += [key for key in net_phases if key not in present_phases]
        labels = {"city": "URB / CITY", "hwy": "HWY", "low": "LOW", "mid": "MID", "high": "HIGH", "xhigh": "XHIGH"}
        for phase in present_phases:
            phase_rows.append(
                {
                    "Phase": labels.get(phase, phase.upper()),
                    "VDE_TOTAL": _format_energy_value(total_phases.get(phase), unavailable="-"),
                    "VDE_NET": _format_energy_value(net_phases.get(phase)),
                }
            )
        if phase_rows:
            st.caption("Phase outputs")
            st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)
        if warnings:
            st.warning("Warnings: " + ", ".join(warnings))

    with st.container(border=True):
        st.subheader("Review Worksheet")
        target_vde_id = staged_payload.get("target_vde_id")
        basis_label = (
            "Baseline ABC_TOTAL" if str(working_summary.get("roadload_basis") or "") == "BASELINE"
            else "Component Build-up" if str(working_summary.get("roadload_basis") or "") in {"COMPONENT_BUILD_UP", "COMPONENTS"}
            else "Measured/test coastdown"
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("Reference source", str(reference_snapshot.get("kind") or "-"))
        r2.metric("Roadload basis", basis_label)
        r3.metric("Save mode", _save_mode_label(target_vde_id))
        st.caption("Resolved from the current draft state using the existing workflow preview.")

    with st.container(border=True):
        st.subheader("Change Worksheet")
        change_rows = []
        order = [
            ("Mass", "mass"),
            ("Tires", "tires"),
            ("Aerodynamics", "aero"),
            ("Brakes", "brakes"),
            ("Parasitics", "parasitics"),
            ("Trailer", "trailer"),
            ("Transmission", "transmission"),
            ("Cycle", "cycle"),
        ]
        for label, key in order:
            item = dict(change_summary.get(key) or {})
            change_rows.append(
                {
                    "Area": label,
                    "Working state": item.get("working") or "-",
                    "Status": item.get("state") or "-",
                }
            )
        st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Reference vs Working Worksheet")
        main_rows = []
        quiet_rows = []
        for row in baseline_rows:
            compact = dict(row)
            if str(compact.get("Change") or "").strip().lower() == "not applicable" or "not needed" in str(compact.get("Change") or "").strip().lower():
                quiet_rows.append(compact)
            else:
                main_rows.append(compact)
        if main_rows:
            st.dataframe(pd.DataFrame(main_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No major comparison rows are available yet.")
        if quiet_rows:
            with st.expander(f"Not applicable rows ({len(quiet_rows)})", expanded=False):
                st.dataframe(pd.DataFrame(quiet_rows), use_container_width=True, hide_index=True)

    with st.expander("Technical Audit", expanded=False):
        st.caption("Detailed provenance, component resolution, staged payload and debug preview.")
        with st.expander("Component build-up audit", expanded=False):
            components = list(workflow_preview.get("components") or [])
            if components:
                st.dataframe(pd.DataFrame(components), use_container_width=True, hide_index=True)
            else:
                st.caption("No resolved component records are present for this preview.")
        with st.expander("VDE calculation audit", expanded=False):
            st.write(
                {
                    "line_source": workflow_preview.get("line_source"),
                    "initial_abc_total_base": workflow_preview.get("initial_abc_total_base"),
                    "component_abc_total": workflow_preview.get("component_abc_total"),
                    "abc_total": workflow_preview.get("abc_total"),
                    "transmission_losses": workflow_preview.get("transmission_losses"),
                    "abc_net": workflow_preview.get("abc_net"),
                    "mass_setup": workflow_preview.get("mass_setup"),
                    "vde_total": workflow_preview.get("vde_total"),
                    "vde_net": workflow_preview.get("vde_net"),
                    "warnings": workflow_preview.get("warnings"),
                }
            )
        with st.expander("Save payload / provenance", expanded=False):
            st.caption("Insert row")
            st.dataframe(pd.DataFrame(_dict_to_rows(staged_payload.get("insert_row"))), use_container_width=True, hide_index=True)
            st.caption("Update row")
            st.dataframe(pd.DataFrame(_dict_to_rows(staged_payload.get("update_row"))), use_container_width=True, hide_index=True)
            target_label = f"VDE-{int(staged_payload['target_vde_id']):03d}" if staged_payload.get("target_vde_id") is not None else "None"
            st.caption(f"Target VDE id: {target_label}")
            if phase_update_row:
                st.caption("Phase update row")
                st.dataframe(pd.DataFrame(_dict_to_rows(phase_update_row)), use_container_width=True, hide_index=True)
            st.write(
                {
                    "reference_snapshot": review.get("reference_snapshot"),
                    "working_scenario_summary": review.get("working_scenario_summary"),
                    "change_summary": review.get("change_summary"),
                    "staged_save_payload": staged_payload,
                    "phase_update_row": phase_update_row,
                }
            )
        if legacy_preview.get("ok") and not component_build_up_active:
            with st.expander("Legacy preview (debug)", expanded=False):
                total_mj_km = float(legacy_preview["total_mj_km"])
                phases = legacy_preview.get("phases", {})
                st.caption(f"Legacy NET preview: {format_quantity(total_mj_km, 'energy_per_distance', format_str='%.4f')}")
                _render_phase_metrics("Legacy phase outputs", phases)


def render_live_vde_preview_panel():
    ctx = st.session_state.get("ctx", {})
    component_build_up_active = _component_build_up_enabled(ctx)

    try:
        workflow_preview = build_vde_setup_preview_from_ctx(ctx)
    except Exception as e:
        workflow_preview = {"ok": False, "error": f"Workflow preview not available: {e}"}

    legacy_preview = build_live_vde_preview(ctx)
    if not workflow_preview.get("ok") and not legacy_preview.get("ok"):
        st.warning(workflow_preview.get("error") or legacy_preview.get("error") or "Preview not available.")
        return

    if workflow_preview.get("ok"):
        vde_total = workflow_preview.get("vde_total") or {}
        vde_net = workflow_preview.get("vde_net") or {}
        abc_total = workflow_preview.get("abc_total") or {}
        abc_net = workflow_preview.get("abc_net") or {}
        mass_setup = workflow_preview.get("mass_setup") or {}
        transmission = workflow_preview.get("transmission_losses") or {}
        transmission_abc = transmission.get("abc") or {}
        line_source = workflow_preview.get("line_source") or {}
        warnings = list(workflow_preview.get("warnings") or [])

        top1, top2, top3, top4 = st.columns(4)
        top1.metric("VDE_TOTAL", _format_energy_value(vde_total.get("mj_per_km"), unavailable="-"))
        top2.metric("VDE_NET", _format_energy_value(vde_net.get("mj_per_km")))
        top3.metric("Roadload basis", _total_source_summary(ctx, workflow_preview))
        top4.metric("Transmission", str(transmission.get("status") or "missing").replace("_", " ").title())

        preview_rows = [
            {
                "Stage": "TOTAL",
                "ABC": _compact_abc(abc_total),
                "Energy": _format_energy_value(vde_total.get("mj_per_km"), unavailable="-"),
                "Status": "Available",
            },
            {
                "Stage": "TRANS",
                "ABC": _compact_abc(transmission_abc) if transmission.get("status") == "available" else "Unavailable",
                "Energy": "-",
                "Status": str(transmission.get("status") or "missing").replace("_", " ").title(),
            },
            {
                "Stage": "NET",
                "ABC": _compact_abc(abc_net) if abc_net else "Unavailable",
                "Energy": _format_energy_value(vde_net.get("mj_per_km")),
                "Status": "Available" if vde_net.get("mj_per_km") is not None else "Pending",
            },
        ]
        st.caption("Preview Overview")
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Mass basis", str(mass_setup.get("mass_basis") or "-"))
        quantity_metric(d2, "Resolved mass", mass_setup.get("resolved_mass_used_kg"), "mass", format_str="%.1f")
        d3.metric("Scenario origin", _line_source_summary(ctx))
        d4.metric("Cycle", str(ctx.get("cycle_name") or line_source.get("cycle_name") or "-"))

        _render_preview_roadload_curves(abc_total, abc_net)

        if warnings:
            st.warning("Workflow warnings: " + ", ".join(warnings))

        with st.expander("Phase outputs", expanded=False):
            _render_phase_metrics("TOTAL phase outputs", vde_total.get("by_phase"))
            _render_phase_metrics("NET phase outputs", vde_net.get("by_phase"))

        with st.expander("Technical preview details", expanded=False):
            st.write(
                {
                    "initial_abc_total_base": workflow_preview.get("initial_abc_total_base"),
                    "component_abc_total": workflow_preview.get("component_abc_total"),
                    "abc_total": workflow_preview.get("abc_total"),
                    "abc_net": workflow_preview.get("abc_net"),
                    "transmission_losses": transmission,
                    "mass_setup": mass_setup,
                    "components": workflow_preview.get("components"),
                    "save_payload": workflow_preview.get("save_payload"),
                }
            )

    if legacy_preview.get("ok") and not workflow_preview.get("ok"):
        total_mj_km = float(legacy_preview["total_mj_km"])
        phases = legacy_preview.get("phases", {})
        equiv = legacy_preview.get("equiv")

        st.info(f"Legacy NET preview: **{format_quantity(total_mj_km, 'energy_per_distance')}**")
        if equiv is not None:
            with st.expander("RoadLoad breakdown", expanded=False):
                st.dataframe(pd.DataFrame(equiv.component_table), use_container_width=True, hide_index=True)
        _render_phase_metrics("Legacy phase outputs", phases)
    elif legacy_preview.get("ok") and workflow_preview.get("ok") and not component_build_up_active:
        with st.expander("Legacy preview (debug)", expanded=False):
            total_mj_km = float(legacy_preview["total_mj_km"])
            phases = legacy_preview.get("phases", {})
            st.caption(f"Legacy NET preview: {format_quantity(total_mj_km, 'energy_per_distance', format_str='%.4f')}")
            _render_phase_metrics("Legacy phase outputs", phases)


def render_cycle_section(*, include_preview_snapshot: bool = True):
    ctx = st.session_state.ctx

    errors, warns = validate_core(ctx["A"], ctx["B"], ctx["C"], ctx["mass_kg"])
    if warns:
        for warning in warns:
            st.warning(warning)

    cleft, cright = st.columns([1, 1])
    use_default = cleft.button("Use legislation default cycle")
    upload = cright.file_uploader(
        "or upload CSV with columns [t, v] (s, m/s)",
        type=["csv"],
        accept_multiple_files=False,
    )

    if use_default:
        cycle_name = default_cycle_for_legislation(ctx["legislation"])
        df_cycle = use_standard_cycle(ctx["legislation"])
        if df_cycle is None:
            st.warning(f"Default cycle for {ctx['legislation']} not found.")
            st.info("Upload a custom CSV cycle with columns [t, v].")
        else:
            ctx["cycle_df"] = df_cycle
            ctx["cycle_name"] = cycle_name
            st.success(f"Using default cycle: {cycle_name}.csv")

    if upload is not None:
        try:
            df_cycle = pd.read_csv(upload)
            if not {"t", "v"} <= set(df_cycle.columns):
                raise ValueError("Uploaded CSV must have columns: t, v (v in m/s)")
            df_cycle = df_cycle.copy()
            df_cycle["t"] = pd.to_numeric(df_cycle["t"], errors="raise")
            df_cycle["v"] = pd.to_numeric(df_cycle["v"], errors="raise")
            if "phase" in df_cycle.columns:
                df_cycle["phase"] = df_cycle["phase"].astype(str)
            ctx["cycle_df"] = df_cycle
            ctx["cycle_name"] = upload.name
            st.success(f"Cycle loaded: {upload.name}")
        except Exception as e:
            st.error(f"CSV load error: {e}")

    if ctx["cycle_df"] is not None:
        df_cycle = ctx["cycle_df"]
        t_vals = pd.to_numeric(df_cycle["t"], errors="coerce")
        v_vals = pd.to_numeric(df_cycle["v"], errors="coerce")
        duration_s = float(t_vals.iloc[-1] - t_vals.iloc[0]) if len(t_vals) else 0.0
        distance_km = float(np.trapezoid(v_vals, t_vals) / 1000.0)
        avg_speed_kph = float(v_vals.mean() * 3.6) if len(v_vals) else 0.0
        distance_value = distance_km if _current_unit_system() == "Metric" else distance_km * 0.621371192237
        distance_unit = "km" if _current_unit_system() == "Metric" else "mi"
        st.caption(
            f"Duration: {duration_s:.0f} s | Distance: {distance_value:.2f} {distance_unit} | "
            f"v_avg: {format_quantity(avg_speed_kph, 'speed', format_str='%.1f')}"
        )
    else:
        errors.append("No cycle loaded. Use default or upload a CSV.")

    if ctx["cycle_df"] is not None:
        display_cycle_df = ctx["cycle_df"].copy()
        if _current_unit_system() == "US customary":
            display_cycle_df["v"] = pd.to_numeric(display_cycle_df["v"], errors="coerce") * 0.621371192237
        fig = cycle_chart(display_cycle_df, unit_system=_current_unit_system())
        if fig:
            fig.update_yaxes(title_text=f"Speed [{unit_label('speed')}]")
            st.plotly_chart(fig, use_container_width=True)

    if include_preview_snapshot:
        st.divider()
        st.caption("Cycle-linked VDE Preview")
        st.caption("This view keeps cycle selection and phase preview together so URB/CITY and HWY impacts stay visible while you adjust the scenario.")
        _render_cycle_preview_snapshot()


def render_auxiliaries_section(*, defaults_df_getter):
    """
    Uses A/B/C + mass + (category, electrification, transmission_type) from ctx
    to decompose NET coastdown using the defaults CSV.
    """
    ctx = st.session_state.ctx
    st.subheader("Estimate auxiliaries from coastdown (NET)")

    missing = [k for k in ("A", "B", "C", "mass_kg", "category") if ctx.get(k) in (None, "")]
    disabled = len(missing) > 0
    if disabled:
        st.caption(f"Fill first: {', '.join(missing)}")

    if st.button("Estimate using defaults CSV", disabled=disabled):
        defaults_df = defaults_df_getter() if callable(defaults_df_getter) else None
        res = estimate_aux_from_coastdown(
            A_N=ctx["A"],
            B_N_per_kph=ctx["B"],
            C_N_per_kph2=ctx["C"],
            mass_kg=ctx["mass_kg"],
            category=ctx["category"],
            electrification=ctx.get("electrification", "ICE"),
            transmission_type=ctx.get("transmission_type", "AT"),
            cdA_override_m2=ctx.get("cda_m2"),
            defaults_df=defaults_df,
        )

        ctx.update(
            {
                "rr_alpha_N": res["rr_alpha_N"],
                "rr_beta_Npkph": res["rr_beta_Npkph"],
                "aero_C_coef_Npkph2": res["aero_C_coef_Npkph2"],
                "parasitic_A_N": res["parasitic_A_coef_N"],
                "parasitic_B_Npkph": res["parasitic_B_coef_Npkph"],
                "parasitic_C_Npkph2": res["parasitic_C_coef_Npkph2"],
                "decomp_check_ok": res["check_ok"],
                "cda_m2": res["cdA_used_m2"],
            }
        )

        c1, c2, c3 = st.columns(3)
        quantity_metric(c1, "RR alpha", res["rr_alpha_N"], "force", format_str="%.2f")
        quantity_metric(c2, "RR beta", res["rr_beta_Npkph"], "force_per_speed", format_str="%.3f")
        quantity_metric(c3, "Aero C", res["aero_C_coef_Npkph2"], "force_per_speed_squared", format_str="%.3f")
        d1, d2, d3 = st.columns(3)
        quantity_metric(d1, "Parasitic A", res["parasitic_A_coef_N"], "force", format_str="%.2f")
        quantity_metric(d2, "Parasitic B", res["parasitic_B_coef_Npkph"], "force_per_speed", format_str="%.3f")
        d3.metric("Check", "OK" if res["check_ok"] else "Review")


def render_from_test_section():
    """
    Enter coastdown outputs directly.
    Keeps compatibility with legacy session keys as a transition layer.
    """
    ctx = st.session_state.ctx
    st.subheader("From test - direct coastdown (A/B/C)")

    if _current_vde_input_mode(ctx) == "Spreadsheet":
        st.caption(
            "Edit ABC_TOTAL directly in the table. ABC_NET may also be staged manually; when applied, "
            "the app derives transmission losses from ABC_TOTAL - ABC_NET using the existing TOTAL -> NET logic."
        )
        roadload_df = _build_roadload_spreadsheet_df(ctx)
        editor_key = f"vde_roadload_spreadsheet_{str(ctx.get('mode') or 'default').replace(' ', '_')}"
        edited_df = st.data_editor(
            roadload_df,
            key=editor_key,
            hide_index=True,
            use_container_width=True,
            disabled=["roadload_set", "status", "notes"],
            column_config={
                "roadload_set": st.column_config.TextColumn("roadload_set"),
                "A": st.column_config.NumberColumn(f"A [{unit_label('force')}]", format="%.6f"),
                "B": st.column_config.NumberColumn(f"B [{unit_label('force_per_speed')}]", format="%.6f"),
                "C": st.column_config.NumberColumn(f"C [{unit_label('force_per_speed_squared')}]", format="%.6f"),
                "basis": st.column_config.SelectboxColumn("basis", options=["coastdown", "derived/manual", "manual"]),
                "source": st.column_config.TextColumn("source"),
                "apply": st.column_config.CheckboxColumn("apply"),
                "status": st.column_config.TextColumn("status"),
                "notes": st.column_config.TextColumn("notes"),
            },
        )
        st.session_state["vde_abc_table"] = edited_df.copy()
        errors = _apply_roadload_spreadsheet_changes(edited_df)
        for error in errors:
            st.warning(error)

        preview = _safe_workflow_preview(ctx)
        abc_total = dict(preview.get("abc_total") or {})
        abc_net = dict(preview.get("abc_net") or {})
        vde_total = dict(preview.get("vde_total") or {})
        vde_net = dict(preview.get("vde_net") or {})
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("ABC_TOTAL", _compact_abc(abc_total))
        p2.metric("ABC_NET", _compact_abc(abc_net) if abc_net else "Unavailable")
        p3.metric("VDE_TOTAL", _format_energy_value(vde_total.get("mj_per_km"), unavailable="-"))
        p4.metric("VDE_NET", _format_energy_value(vde_net.get("mj_per_km")))
        st.caption("Mass and test-mass inputs are managed in Vehicle Parameters.")
        _render_roadload_plot(ctx.get("A"), ctx.get("B"), ctx.get("C"))
        # TODO: Component build-up spreadsheet stays out of scope for this first spreadsheet-input round.
        return

    colA, colB, colC = st.columns(3)
    A = quantity_input(colA, "A", to_float(ctx.get("A"), 30.0), "force", key="from_test_A", min_canonical=0.0, max_canonical=500.0, step_canonical=0.1, format_str="%.2f")
    B = quantity_input(colB, "B", to_float(ctx.get("B"), 0.80), "force_per_speed", key="from_test_B", min_canonical=-1.0, max_canonical=5.0, step_canonical=0.01, format_str="%.5f")
    C = quantity_input(colC, "C", to_float(ctx.get("C"), 0.011), "force_per_speed_squared", key="from_test_C", min_canonical=0.0, max_canonical=0.100, step_canonical=0.001, format_str="%.6f")
    ctx["A"], ctx["B"], ctx["C"] = to_float(A), to_float(B), to_float(C)

    st.session_state["abc"] = {"A": float(A), "B": float(B), "C": float(C)}
    st.session_state["manual_mass"] = to_float(ctx.get("mass_kg"))
    st.caption("Mass and test-mass inputs are managed in Vehicle Parameters.")
    _render_roadload_plot(ctx["A"], ctx["B"], ctx["C"])


def render_aero_section(*, prefill=None):
    """
    Uses cda_m2 from DB. Shows estimated aero C [N/kph^2] as reference.
    Does not overwrite measured coastdown C.
    """
    ctx = st.session_state.ctx
    st.subheader("Aerodynamics")

    cda0 = to_float(prefill.get("cda_m2"), ctx.get("cda_m2")) if prefill else to_float(ctx.get("cda_m2"), None)
    cda = quantity_input(st, "CdA", to_float(cda0, 0.0), "cda", key="aero_section_cda", step_canonical=0.01, format_str="%.3f")
    ctx["cda_m2"] = to_float(cda)

    rho = 1.2
    C_aero = 0.5 * rho * ctx["cda_m2"] * (1 / 3.6) ** 2
    ctx["aero_C_coef_Npkph2"] = C_aero

    quantity_metric(st, "C_aero (est.)", C_aero, "force_per_speed_squared", format_str="%.6f")
    st.caption("Measured coastdown C remains in 'coast_C_N_per_kph2'; this is only reference.")


def render_rr_section(*, prefill=None, tires_df=None):
    """
    RR only (does not overwrite A/B/C):
      IN: rrc_N_per_kN [N/kN], crr1_frac_at_120kph [-], mass_kg [kg]
      Optional: tire selectbox when tires_df is provided
      OUT: rr_alpha_N [N], rr_beta_Npkph [N/kph]; stores tire_size in ctx
    """
    ctx = st.session_state.ctx
    st.subheader("Rolling Resistance")

    if isinstance(tires_df, pd.DataFrame) and not tires_df.empty:
        sizes = sorted(tires_df["tire_size"].dropna().astype(str).unique().tolist())
        current_size = str(ctx.get("tire_size") or (prefill.get("tire_size") if prefill else "") or "")
        try:
            idx0 = sizes.index(current_size) if current_size in sizes else 0
        except Exception:
            idx0 = 0
        sel = st.selectbox("Tire size", sizes, index=idx0)
        ctx["tire_size"] = sel
        trow = tires_df.loc[tires_df["tire_size"] == sel].iloc[0].to_dict()
        st.caption(f'Diameter {trow["tire_circ_mm"]:.0f} mm ')
        ctx["tire_circ_m"] = float(trow["tire_circ_mm"]) / 1000.0

    if prefill:
        rrc0 = to_float(prefill.get("rrc_N_per_kN"), to_float(ctx.get("rrc_N_per_kN"), 9.5))
        frac0 = to_float(prefill.get("crr1_frac_at_120kph"), to_float(ctx.get("crr1_frac_at_120kph"), 0.10))
        m0 = to_float(prefill.get("mass_kg"), to_float(ctx.get("mass_kg"), 1500.0))
        weight_dist0 = to_float(prefill.get("weight_dist_fr_pct"), to_float(ctx.get("weight_dist_fr_pct"), 50.0))
    else:
        rrc0 = to_float(ctx.get("rrc_N_per_kN"), 9.5)
        frac0 = to_float(ctx.get("crr1_frac_at_120kph"), 0.10)
        m0 = to_float(ctx.get("mass_kg"), 1500.0)
        weight_dist0 = to_float(ctx.get("weight_dist_fr_pct"), 50.0)

    c1, c2 = st.columns(2)
    ctx["rrc_N_per_kN"] = quantity_input(c1, "Final RRC", to_float(rrc0, 0.0), "rrc", key="rr_section_rrc", step_canonical=0.1, format_str="%.2f")
    ctx["crr1_frac_at_120kph"] = c2.number_input(
        "Linear RR fraction at 120 kph [-]",
        value=float(frac0),
        min_value=0.0,
        max_value=1.0,
        step=0.005,
        format="%.3f",
    )

    legislation = str(ctx.get("legislation", "") or "").strip().upper()
    st.caption(
        "Mass setup is controlled in Vehicle Parameters. "
        f"Using curb mass={format_quantity(ctx.get('mass_kg') or m0 or 0.0, 'mass', format_str='%.1f')}, "
        f"front weight distribution={float(ctx.get('weight_dist_fr_pct') or weight_dist0 or 50.0):.1f}%, "
        f"VDE mass basis={str(ctx.get('tire_load_mass_basis') or resolve_tire_load_mass_basis(ctx))}."
    )

    tire_mass_resolution = resolve_tire_calculation_mass(ctx)
    calc_mass_kg = tire_mass_resolution.get("mass_kg")
    if legislation == "EPA" and ctx.get("tire_load_mass_basis") == "TWC":
        ctx["inertia_class"] = calc_mass_kg
        ctx["twc_kg"] = calc_mass_kg
    G = 9.80665
    load_kN = (ctx["mass_kg"] * G) / 1000.0 if ctx.get("mass_kg") else 0.0
    A_rr, B_rr, _, epa_v_avg = _abc_from_final_rr_target(
        ctx.get("rrc_N_per_kN"),
        load_kN=load_kN,
        crr_frac_120=ctx.get("crr1_frac_at_120kph"),
    )

    ctx["rr_alpha_N"] = A_rr
    ctx["rr_beta_Npkph"] = B_rr

    c4, c5, c6 = st.columns(3)
    c4.metric("Load [kN]", f"{load_kN:.2f}")
    quantity_metric(c5, "A_rr ~ SMERF", A_rr, "force", format_str="%.2f")
    quantity_metric(c6, "Calculation mass", calc_mass_kg, "mass", format_str="%.1f")
    st.caption(f"Current VDE mass basis: {tire_mass_resolution.get('basis')} ({tire_mass_resolution.get('source')})")
    st.caption(f"Resolved from final RRC target: F_rr(v) = {A_rr:.2f} + {B_rr:.5f}*v   [v in kph]")
    if epa_v_avg is not None:
        st.caption(f"EPA v_avg used for decomposition: {epa_v_avg:.2f} kph")


def render_parasitic_section(*, prefill=None):
    ctx = st.session_state.ctx
    st.subheader("Parasitics / Hubs / Axle")

    if prefill:
        parA0 = to_float(prefill.get("parasitic_A_coef_N"), ctx.get("parasitic_A_coef_N", 0.0))
        parB0 = to_float(prefill.get("parasitic_B_Npkph"), ctx.get("parasitic_B_Npkph", 0.0))
        parC0 = to_float(prefill.get("parasitic_C_coef_Npkph2"), ctx.get("parasitic_C_coef_Npkph2", 0.0))
    else:
        parA0 = to_float(ctx.get("parasitic_A_coef_N"), 0.0)
        parB0 = to_float(ctx.get("parasitic_B_Npkph"), 0.0)
        parC0 = to_float(ctx.get("parasitic_C_coef_Npkph2"), 0.0)

    p1, p2, p3 = st.columns(3)
    ctx["parasitic_A_coef_N"] = quantity_input(p1, "Parasitic A", to_float(parA0, 0.0), "force", key="parasitic_A_coef_N", step_canonical=0.1, format_str="%.2f")
    ctx["parasitic_B_Npkph"] = quantity_input(p2, "Parasitic B", to_float(parB0, 0.0), "force_per_speed", key="parasitic_B_Npkph", step_canonical=0.001, format_str="%.5f")
    ctx["parasitic_C_coef_Npkph2"] = quantity_input(p3, "Parasitic C", to_float(parC0, 0.0), "force_per_speed_squared", key="parasitic_C_coef_Npkph2", step_canonical=0.0001, format_str="%.6f")

    c1, c2, c3 = st.columns(3)
    quantity_metric(c1, "Parasitic A", ctx["parasitic_A_coef_N"], "force", format_str="%.2f")
    quantity_metric(c2, "Parasitic B", ctx["parasitic_B_Npkph"], "force_per_speed", format_str="%.5f")
    quantity_metric(c3, "Parasitic C", ctx["parasitic_C_coef_Npkph2"], "force_per_speed_squared", format_str="%.6f")

    st.caption("This slot is meant to absorb hub, axle, bearing and other non-brake parasitic losses.")


def render_brake_section(*, prefill=None):
    ctx = st.session_state.ctx
    st.subheader("Brakes")

    if prefill:
        brA0 = to_float(prefill.get("brake_A_coef_N"), ctx.get("brake_A_coef_N", 0.0))
        brB0 = to_float(prefill.get("brake_B_Npkph"), ctx.get("brake_B_Npkph", 0.0))
        brC0 = to_float(prefill.get("brake_C_coef_Npkph2"), ctx.get("brake_C_coef_Npkph2", 0.0))
    else:
        brA0 = to_float(ctx.get("brake_A_coef_N"), 0.0)
        brB0 = to_float(ctx.get("brake_B_Npkph"), 0.0)
        brC0 = to_float(ctx.get("brake_C_coef_Npkph2"), 0.0)

    b1, b2, b3 = st.columns(3)
    ctx["brake_A_coef_N"] = quantity_input(b1, "Brake A", to_float(brA0, 0.0), "force", key="brake_A_coef_N", step_canonical=0.1, format_str="%.2f")
    ctx["brake_B_Npkph"] = quantity_input(b2, "Brake B", to_float(brB0, 0.0), "force_per_speed", key="brake_B_Npkph", step_canonical=0.001, format_str="%.5f")
    ctx["brake_C_coef_Npkph2"] = quantity_input(b3, "Brake C", to_float(brC0, 0.0), "force_per_speed_squared", key="brake_C_coef_Npkph2", step_canonical=0.0001, format_str="%.6f")

    c1, c2, c3 = st.columns(3)
    quantity_metric(c1, "Brake A", ctx["brake_A_coef_N"], "force", format_str="%.2f")
    quantity_metric(c2, "Brake B", ctx["brake_B_Npkph"], "force_per_speed", format_str="%.5f")
    quantity_metric(c3, "Brake C", ctx["brake_C_coef_Npkph2"], "force_per_speed_squared", format_str="%.6f")

    st.caption("Brake drag remains an explicit TOTAL component so future brake DB rows can plug into the same slot.")


def render_transmission_losses_section(*, prefill=None):
    """
    Explicit TOTAL -> NET bridge.

    Keeps transmission losses visible in the workflow so VDE_NET semantics are
    not hidden inside page-specific assumptions.
    """
    ctx = st.session_state.ctx
    st.subheader("Transmission Losses / Neutral Drag")
    st.caption("Transmission is configured separately from Components because it bridges ABC_TOTAL to ABC_NET, but it behaves like a technical component with explicit reference and explicit application.")

    prefill = dict(prefill or {})
    if _current_vde_input_mode(ctx) == "Spreadsheet":
        st.caption("Edit the applied transmission losses in a compact table. The existing preview remains the source of truth for TOTAL -> NET.")
        transmission_df = _build_transmission_spreadsheet_df(ctx, prefill=prefill)
        editor_key = f"vde_transmission_spreadsheet_{str(ctx.get('mode') or 'default').replace(' ', '_')}"
        edited_df = st.data_editor(
            transmission_df,
            key=editor_key,
            hide_index=True,
            use_container_width=True,
            disabled=["loss_set", "status", "notes"],
            column_config={
                "loss_set": st.column_config.TextColumn("loss_set"),
                "A_loss": st.column_config.NumberColumn(f"A_loss [{unit_label('force')}]", format="%.6f"),
                "B_loss": st.column_config.NumberColumn(f"B_loss [{unit_label('force_per_speed')}]", format="%.6f"),
                "C_loss": st.column_config.NumberColumn(f"C_loss [{unit_label('force_per_speed_squared')}]", format="%.6f"),
                "source": st.column_config.TextColumn("source"),
                "apply": st.column_config.CheckboxColumn("apply"),
                "status": st.column_config.TextColumn("status"),
                "notes": st.column_config.TextColumn("notes"),
            },
        )
        st.session_state["vde_trans_loss_table"] = edited_df.copy()
        errors = _apply_transmission_spreadsheet_changes(edited_df)
        for error in errors:
            st.warning(error)

        applied_trans = {
            "A": float(to_float(ctx.get("trans_A_coef_N"), 0.0) or 0.0),
            "B": float(to_float(ctx.get("trans_B_coef_Npkph"), 0.0) or 0.0),
            "C": float(to_float(ctx.get("trans_C_coef_Npkph2"), 0.0) or 0.0),
        }
        with st.container(border=True):
            _render_tire_editor_block_header(
                "Instant TOTAL -> NET Preview",
                "Read-only bridge preview using the current ABC_TOTAL and applied transmission losses.",
            )
            preview = _safe_workflow_preview(ctx)
            abc_total = dict(preview.get("abc_total") or {})
            abc_net = dict(preview.get("abc_net") or {})
            vde_net = dict(preview.get("vde_net") or {})
            transmission_ready = bool((edited_df.to_dict(orient="records")[0] or {}).get("apply")) and not errors
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current ABC_TOTAL", _compact_abc(abc_total))
            c2.metric("Applied ABC_TRANS", _compact_abc(applied_trans))
            if transmission_ready and abc_net:
                c3.metric("Resolved ABC_NET", _compact_abc(abc_net))
                c4.metric("VDE_NET", _format_energy_value(vde_net.get("mj_per_km")))
                st.caption("ABC_NET = ABC_TOTAL - ABC_TRANS. VDE_TOTAL is still based on ABC_TOTAL.")
            else:
                c3.metric("Resolved ABC_NET", "Unavailable")
                c4.metric("VDE_NET", "Unavailable")
                st.caption("VDE_TOTAL remains available; NET outputs require applied transmission losses.")
        return

    source_options = ["Missing", "Baseline", "Manual"]
    source_default = str(
        ctx.get("transmission_losses_source")
        or ("Baseline" if prefill.get("trans_A_coef_N") is not None else "Missing")
    ).strip().title()
    if source_default not in source_options:
        source_default = "Missing"

    base_a = to_float(prefill.get("trans_A_coef_N"), 0.0)
    base_b = to_float(prefill.get("trans_B_coef_Npkph", prefill.get("trans_B_Npkph")), 0.0)
    base_c = to_float(prefill.get("trans_C_coef_Npkph2"), 0.0)
    baseline_available = any(abs(float(value or 0.0)) > 0.0 for value in (base_a, base_b, base_c))
    baseline_target_id = int(prefill.get("id") or ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) or None

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Reference Transmission Losses",
            "Defines the inherited neutral-drag reference available from the selected baseline row.",
        )
        if baseline_available:
            _render_abc_reference_metrics(
                title="Baseline transmission losses",
                a_value=base_a,
                b_value=base_b,
                c_value=base_c,
                status="Loaded",
                caption="This is the current inherited transmission reference before any manual replacement.",
            )
        else:
            st.warning("Reference status: Missing. No transmission losses are stored on the selected baseline row.")
        if baseline_target_id:
            with st.expander("Set / update baseline reference transmission losses", expanded=not baseline_available):
                t1, t2, t3 = st.columns(3)
                baseline_ref_a = quantity_input(
                    t1,
                    "Baseline transmission A",
                    to_float(ctx.get("transmission_baseline_reference_A"), base_a or 0.0),
                    "force",
                    key="transmission_baseline_reference_A_input",
                    step_canonical=0.1,
                    format_str="%.3f",
                )
                baseline_ref_b = quantity_input(
                    t2,
                    "Baseline transmission B",
                    to_float(ctx.get("transmission_baseline_reference_B"), base_b or 0.0),
                    "force_per_speed",
                    key="transmission_baseline_reference_B_input",
                    step_canonical=0.0001,
                    format_str="%.6f",
                )
                baseline_ref_c = quantity_input(
                    t3,
                    "Baseline transmission C",
                    to_float(ctx.get("transmission_baseline_reference_C"), base_c or 0.0),
                    "force_per_speed_squared",
                    key="transmission_baseline_reference_C_input",
                    step_canonical=0.000001,
                    format_str="%.8f",
                )
                ctx["transmission_baseline_reference_A"] = baseline_ref_a
                ctx["transmission_baseline_reference_B"] = baseline_ref_b
                ctx["transmission_baseline_reference_C"] = baseline_ref_c
                update_baseline = st.checkbox(
                    "Also update selected baseline row in VDE DB",
                    value=not baseline_available,
                    key="transmission_update_baseline_checkbox",
                )
                if st.button("Apply reference transmission losses", key="transmission_apply_reference_button"):
                    ctx["trans_A_coef_N"] = float(baseline_ref_a)
                    ctx["trans_B_coef_Npkph"] = float(baseline_ref_b)
                    ctx["trans_C_coef_Npkph2"] = float(baseline_ref_c)
                    ctx["trans_B_Npkph"] = float(baseline_ref_b)
                    if update_baseline:
                        try:
                            update_vde_snapshot(
                                int(baseline_target_id),
                                {
                                    "trans_A_coef_N": float(baseline_ref_a),
                                    "trans_B_Npkph": float(baseline_ref_b),
                                    "trans_C_coef_Npkph2": float(baseline_ref_c),
                                },
                            )
                            _sync_baseline_abc_reference_in_ctx(
                                a_key="trans_A_coef_N",
                                b_key="trans_B_Npkph",
                                c_key="trans_C_coef_Npkph2",
                                a_value=float(baseline_ref_a),
                                b_value=float(baseline_ref_b),
                                c_value=float(baseline_ref_c),
                                baseline_target_id=baseline_target_id,
                            )
                            if baseline_target_id and int(ctx.get("vde_id_parent") or ctx.get("baseline_id") or 0) == int(baseline_target_id):
                                ctx["trans_B_coef_Npkph"] = float(baseline_ref_b)
                            st.success(f"Baseline reference transmission losses updated on VDE id={baseline_target_id}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not update baseline reference transmission losses: {exc}")
                    else:
                        st.success("Reference transmission losses staged only for the current scenario context.")

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Applied Transmission Change",
            "Choose the transmission losses used to subtract ABC_TRANS from ABC_TOTAL and resolve ABC_NET.",
        )
        source = st.radio(
            "Transmission source",
            source_options,
            horizontal=True,
            index=source_options.index(source_default),
            key="transmission_losses_source_radio",
        )
        ctx["transmission_losses_source"] = source

        if source == "Baseline":
            ctx["trans_A_coef_N"] = base_a
            ctx["trans_B_coef_Npkph"] = base_b
            ctx["trans_C_coef_Npkph2"] = base_c
            if baseline_available:
                st.info("Calculation status: baseline transmission losses are active.")
            else:
                st.warning("Baseline source is selected, but no baseline transmission losses are available. VDE_NET will remain unavailable.")
        elif source == "Missing":
            ctx["trans_A_coef_N"] = 0.0
            ctx["trans_B_coef_Npkph"] = 0.0
            ctx["trans_C_coef_Npkph2"] = 0.0
            st.warning("VDE_TOTAL remains available; VDE_NET will remain unavailable until transmission losses are provided.")
        else:
            a0 = to_float(ctx.get("trans_A_coef_N"), base_a)
            b0 = to_float(ctx.get("trans_B_coef_Npkph"), base_b)
            c0 = to_float(ctx.get("trans_C_coef_Npkph2"), base_c)
            c1, c2, c3 = st.columns(3)
            ctx["trans_A_coef_N"] = quantity_input(c1, "A_TRANS", to_float(a0, 0.0), "force", key="trans_A_coef_N", step_canonical=0.1, format_str="%.2f")
            ctx["trans_B_coef_Npkph"] = quantity_input(c2, "B_TRANS", to_float(b0, 0.0), "force_per_speed", key="trans_B_coef_Npkph", step_canonical=0.001, format_str="%.5f")
            ctx["trans_C_coef_Npkph2"] = quantity_input(c3, "C_TRANS", to_float(c0, 0.0), "force_per_speed_squared", key="trans_C_coef_Npkph2", step_canonical=0.0001, format_str="%.6f")
            st.info("Calculation status: manual transmission losses are active.")

    # Keep the legacy alias in sync with the DB-oriented name used by Sprint 5 services.
    ctx["trans_B_Npkph"] = ctx.get("trans_B_coef_Npkph", 0.0)

    applied_trans = {
        "A": float(to_float(ctx.get("trans_A_coef_N"), 0.0) or 0.0),
        "B": float(to_float(ctx.get("trans_B_coef_Npkph"), 0.0) or 0.0),
        "C": float(to_float(ctx.get("trans_C_coef_Npkph2"), 0.0) or 0.0),
    }

    with st.container(border=True):
        _render_tire_editor_block_header(
            "Instant TOTAL -> NET Preview",
            "Read-only bridge preview using the current ABC_TOTAL and applied transmission losses.",
        )
        preview = _safe_workflow_preview(ctx)
        abc_total = dict(preview.get("abc_total") or {})
        abc_net = dict(preview.get("abc_net") or {})
        vde_net = dict(preview.get("vde_net") or {})
        transmission_ready = source == "Manual" or (source == "Baseline" and baseline_available)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current ABC_TOTAL", _compact_abc(abc_total))
        c2.metric("Applied ABC_TRANS", _compact_abc(applied_trans))
        if transmission_ready and abc_net:
            c3.metric("Resolved ABC_NET", _compact_abc(abc_net))
            c4.metric("VDE_NET", _format_energy_value(vde_net.get("mj_per_km")))
            st.caption("ABC_NET = ABC_TOTAL - ABC_TRANS. VDE_TOTAL is still based on ABC_TOTAL.")
        else:
            c3.metric("Resolved ABC_NET", "Unavailable")
            c4.metric("VDE_NET", "Unavailable")
            st.caption("VDE_TOTAL remains available; NET outputs require transmission losses.")

    st.dataframe(
        pd.DataFrame([_transmission_component_row(ctx, prefill)]),
        use_container_width=True,
        hide_index=True,
    )


def render_vde_edit_delete_panel(*, defaults_path, reset_ctx, defaults_df_getter=None):
    st.markdown("---")
    with st.expander("Existing row maintenance", expanded=False):
        st.subheader("Edit / Delete an existing VDE row")

        rows = fetch_vde_edit_rows(limit=100)
        if not rows:
            st.info("No VDE rows saved yet.")
            return

        labels = [
            f'#{r["id"]} - {r["legislation"]} | {r["category"]} | {r["make"]} {r["model"]} ({r.get("year","")})'
            for r in rows
        ]
        idx = st.selectbox("Pick a VDE to edit/delete", list(range(len(labels))), format_func=lambda i: labels[i])
        sel = rows[idx]
        vde_id_edit = sel["id"]
        st.caption(f"Editing VDE id: {vde_id_edit}")

        try:
            linked_n = fetch_linked_fuelcons_count(vde_id_edit)
            st.caption(f"Linked scenarios in fuelcons_db: {linked_n}")
        except Exception:
            pass

        with st.form(key=f"edit_vde_{vde_id_edit}"):
            c1, c2, c3, c4 = st.columns(4)
            a_edit = quantity_input(c1, "A", to_float(sel["coast_A_N"], 0.0), "force", key=f"edit_vde_a_{vde_id_edit}", min_canonical=0.0, max_canonical=5000.0, step_canonical=0.1, format_str="%.2f")
            b_edit = quantity_input(c2, "B", to_float(sel["coast_B_N_per_kph"], 0.0), "force_per_speed", key=f"edit_vde_b_{vde_id_edit}", min_canonical=-5.0, max_canonical=5.0, step_canonical=0.01, format_str="%.5f")
            c_edit = quantity_input(c3, "C", to_float(sel["coast_C_N_per_kph2"], 0.0), "force_per_speed_squared", key=f"edit_vde_c_{vde_id_edit}", min_canonical=0.0, max_canonical=1.0, step_canonical=0.000001, format_str="%.6f")
            m_edit = quantity_input(c4, "Curb weight", to_float(sel["mass_kg"], 0.0), "mass", key=f"edit_vde_mass_{vde_id_edit}", min_canonical=1.0, max_canonical=4000.0, step_canonical=1.0, format_str="%.1f")

            test_mass_prefill = to_float(sel.get("test_mass_kg"))
            test_mass_default = resolve_test_mass_kg(
                {
                    "legislation": sel.get("legislation"),
                    "mass_kg": m_edit,
                    "test_mass_kg": None,
                }
            )
            use_default_edit = st.checkbox(
                "Use default test mass",
                value=not (test_mass_prefill is not None and test_mass_default is not None and abs(test_mass_prefill - test_mass_default) > 1e-9),
                key=f"edit_vde_use_default_{vde_id_edit}",
            )
            if use_default_edit:
                test_mass_edit = None
                st.caption(f"Test mass: {format_quantity(test_mass_default, 'mass', format_str='%.1f')}")
            else:
                test_mass_edit = quantity_input(
                    st,
                    "Test mass",
                    max(test_mass_prefill if test_mass_prefill is not None else (test_mass_default or m_edit), m_edit),
                    "mass",
                    key=f"edit_vde_manual_test_mass_{vde_id_edit}",
                    min_canonical=float(m_edit),
                    max_canonical=4000.0,
                    step_canonical=1.0,
                    format_str="%.1f",
                )
            hint = build_test_mass_hint({"legislation": sel.get("legislation")})
            if hint:
                st.caption(hint)

            meta1, meta2, meta3, meta4 = st.columns(4)
            meta1.metric("Legislation", str(sel.get("legislation") or "-"))
            category_edit = meta2.text_input("Category", value=str(sel.get("category") or ""))
            electrification_options = ["ICE", "HEV", "PHEV", "BEV"]
            electrification_default = str(sel.get("electrification") or "ICE").upper()
            if electrification_default not in electrification_options:
                electrification_options.append(electrification_default)
            electrification_edit = meta3.selectbox(
                "Electrification",
                electrification_options,
                index=electrification_options.index(electrification_default),
            )
            transmission_options = ["AT", "AMT", "CVT", "MT", "OT"]
            transmission_default = str(sel.get("transmission_type") or "AT").upper()
            if transmission_default not in transmission_options:
                transmission_options.append(transmission_default)
            transmission_type_edit = meta4.selectbox(
                "Transmission",
                transmission_options,
                index=transmission_options.index(transmission_default),
            )

            c5, c6, c7, c8 = st.columns(4)
            make_edit = c5.text_input("Make", value=sel["make"] or "")
            model_edit = c6.text_input("Model", value=sel["model"] or "")
            year_edit = c7.number_input("Year", 1990, 2100, int(sel["year"] or 2020))
            cycle_name_edit = c8.text_input("Cycle name", value=str(sel.get("cycle_name") or default_cycle_for_legislation(sel.get("legislation", "EPA")) or ""))

            extra1, extra2, extra3, extra4 = st.columns(4)
            cda_edit = extra1.number_input("CdA [m^2]", value=float(to_float(sel.get("cda_m2"), 0.0) or 0.0), step=0.001, format="%.4f")
            tire_size_edit = extra2.text_input("Tire size", value=str(sel.get("tire_size") or ""))
            weight_dist_edit = extra3.number_input(
                "Front weight distribution [%]",
                min_value=0.0,
                max_value=100.0,
                value=float(to_float(sel.get("weight_dist_fr_pct"), 50.0) or 50.0),
                step=0.5,
                format="%.1f",
            )
            mass_basis_options = ["TWC", "TEST_MASS"]
            mass_basis_default = str(resolve_tire_load_mass_basis(sel) or "TEST_MASS").upper()
            if mass_basis_default not in mass_basis_options:
                mass_basis_default = "TEST_MASS"
            tire_load_mass_basis_edit = extra4.selectbox(
                "VDE mass basis",
                mass_basis_options,
                index=mass_basis_options.index(mass_basis_default),
            )

            notes_edit = st.text_area("Notes", value=sel["notes"] or "")

            if st.form_submit_button("Save changes"):
                try:
                    core_updates = build_edit_core_update(
                        A=a_edit,
                        B=b_edit,
                        C=c_edit,
                        mass_kg=m_edit,
                        test_mass_kg=to_float(test_mass_edit) or None,
                        make=make_edit,
                        model=model_edit,
                        year=int(year_edit),
                        notes=notes_edit,
                    )
                    extra_edit_updates = {
                        "category": str(category_edit or "").strip(),
                        "electrification": str(electrification_edit or "").strip().upper(),
                        "transmission_type": str(transmission_type_edit or "").strip().upper(),
                        "cycle_name": str(cycle_name_edit or "").strip(),
                        "cda_m2": float(to_float(cda_edit, 0.0) or 0.0),
                        "tire_size": str(tire_size_edit or "").strip(),
                        "weight_dist_fr_pct": float(to_float(weight_dist_edit, 50.0) or 50.0),
                        "tire_load_mass_basis": str(tire_load_mass_basis_edit or "TEST_MASS").strip().upper(),
                    }

                    rr_updates = collect_ctx_updates(
                        st.session_state.get("ctx"),
                        [
                            "tire_size",
                            "rrc_N_per_kN",
                            "crr1_frac_at_120kph",
                            "front_pressure_psi",
                            "rear_pressure_psi",
                            "rr_load_kpa",
                            "smerf",
                        ],
                        include_none=False,
                    )
                    if extra_edit_updates.get("tire_size"):
                        rr_updates["tire_size"] = extra_edit_updates["tire_size"]

                    pb_updates = collect_ctx_updates(
                        st.session_state.get("ctx"),
                        [
                            "parasitic_A_coef_N",
                            "parasitic_B_coef_Npkph",
                            "parasitic_C_coef_Npkph2",
                            "brake_A_coef_N",
                            "brake_B_Npkph",
                            "brake_C_coef_Npkph2",
                        ],
                        include_none=True,
                    )
                    decomp_upd = {}

                    try:
                        defaults_df = (
                            defaults_df_getter()
                            if callable(defaults_df_getter)
                            else load_vde_defaults(defaults_path)
                        )
                        decomp = estimate_aux_from_coastdown(
                            A_N=a_edit,
                            B_N_per_kph=b_edit,
                            C_N_per_kph2=c_edit,
                            mass_kg=m_edit,
                            category=extra_edit_updates.get("category") or sel.get("category", ""),
                            electrification=extra_edit_updates.get("electrification") or sel.get("electrification", "ICE"),
                            transmission_type=extra_edit_updates.get("transmission_type") or sel.get("transmission_type", "AT"),
                            cdA_override_m2=extra_edit_updates.get("cda_m2"),
                            defaults_df=defaults_df,
                        )
                        decomp_upd = build_decomp_update_for_edit(decomp)
                    except Exception:
                        decomp_upd = {}

                    leg_row = sel.get("legislation", "EPA")
                    try:
                        cycle_name = extra_edit_updates.get("cycle_name") or default_cycle_for_legislation(leg_row)
                        df_cycle = load_cycle_csv(cycle_name) if cycle_name else None
                    except Exception:
                        df_cycle = None

                    if isinstance(df_cycle, pd.DataFrame) and not df_cycle.empty:
                        preview = compute_vde_preview_from_inputs(
                            df_cycle,
                            leg_row,
                            A=a_edit,
                            B=b_edit,
                            C=c_edit,
                            mass_kg=m_edit,
                        )
                        if not preview.get("ok"):
                            raise ValueError(preview.get("error", "Failed to recompute VDE."))

                        total_mj_km = float(preview["total_mj_km"])
                        by_phase = dict(preview.get("by_phase", {}))
                        show_vde_feedback(total_mj_km, by_phase)

                        phase_updates = merge_update_payloads(
                            {"vde_net_mj_per_km": total_mj_km},
                            build_vde_phase_update(
                                df_cycle,
                                leg_row,
                                A=a_edit,
                                B=b_edit,
                                C=c_edit,
                                mass_kg=m_edit,
                            ),
                        )
                        final_updates = merge_update_payloads(
                            core_updates,
                            extra_edit_updates,
                            rr_updates,
                            pb_updates,
                            decomp_upd,
                            phase_updates,
                        )
                        if final_updates:
                            update_vde_snapshot(vde_id_edit, final_updates)
                    else:
                        final_updates = merge_update_payloads(
                            core_updates,
                            extra_edit_updates,
                            rr_updates,
                            pb_updates,
                            decomp_upd,
                        )
                        if final_updates:
                            update_vde_snapshot(vde_id_edit, final_updates)
                        st.warning("Row updated, but cycle could not be loaded; phase VDE not recomputed.")

                    st.success("Row updated.")
                    reset_ctx(preserve_meta=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update: {e}")

        with st.container(border=True):
            st.warning("Delete is irreversible. Linked fuelcons_db rows will be deleted as well.")
            confirm_text = st.text_input("Type DELETE to confirm:")
            delete_disabled = (str(confirm_text).strip().upper() != "DELETE")
            if st.button(f"Delete VDE id={vde_id_edit}", type="secondary", disabled=delete_disabled):
                try:
                    deleted = delete_vde_snapshot(vde_id_edit)
                    if deleted > 0:
                        st.success(f"VDE id={vde_id_edit} deleted.")
                        reset_ctx(preserve_meta=True)
                        st.rerun()
                    else:
                        st.warning(f"VDE id={vde_id_edit} was not found (no row deleted).")
                except Exception as e:
                    st.error(f"Failed to delete: {e}")


