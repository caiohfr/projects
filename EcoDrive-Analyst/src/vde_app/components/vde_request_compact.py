from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import sqlite3
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.vde_app.components.shared import search_logo
from src.vde_app.components.vde_request_compact_style import (
    render_v22_apply_result,
    render_v22_branding_header,
    render_v22_chip_list,
    render_v22_context_strip,
    render_v22_domain_card_header,
    render_v22_group_header,
    render_v22_notice_strip,
    render_v22_preview_status_strip,
    render_v22_reference_divider,
    render_v22_request_inputs_overview,
    render_v22_scenario_overview_cards,
    render_v22_sidebar_step_meta,
    render_v22_summary_groups,
    render_v22_step_header,
)
from src.vde_app.components.vde_request_compact_viewmodels import (
    build_active_corrections_summary,
    build_baseline_candidate_status_payload,
    build_domain_card_payload,
    build_loaded_baseline_summary_payload,
    build_roadload_analysis_payload,
    build_cycle_power_analysis_payload,
    build_preview_audit_payload,
    build_preview_status_payload,
    build_scenario_overview_payload,
    build_vde_cycle_comparison_payload,
    build_engineering_comparison_payload,
    build_validation_summary_payload,
    build_request_inputs_overview_payload,
    build_v22_branding_payload,
    build_v22_flow_status_payload,
    format_v22_issue_for_display,
    proposal_display_label,
    SECTION_ORDER,
    walk_from_display_label,
)
from src.vde_app.components.vde_request_domain_editors import (
    applicable_fields,
    field_meta,
    field_schema,
    friendly_message,
    is_field_editable,
    is_field_editable_with_inputs,
    proposal_is_not_used,
    proposal_status_label,
    resolve_domain_display,
    rows_for_active_domain,
    sanitize_domain_inputs,
)
from src.vde_app.components.vde_request_lookup import (
    active_domain_has_lookup_requests,
    apply_lookup_to_inputs,
    component_lookup_rows,
    default_lookup_source,
    is_component_lookup_source,
    lookup_empty_message,
    lookup_source_options,
    TIRE_LOOKUP_BROWSE_LIMIT,
    vde_lookup_rows,
)
from src.vde_app.components.vde_request_compact_units import (
    display_unit_for_field,
    field_uses_display_units,
    format_select_option_for_field,
    display_format_for_field,
    display_step_for_field,
    format_display_value_for_field,
    format_value_map_for_display,
    quantity_kind_for_field,
    to_canonical_field_value,
    to_display_field_value,
)
from src.vde_app.components.vde_request_metadata_options import metadata_field_spec, metadata_override_value
from src.vde_app.units import PRESSURE_UNIT_OPTIONS, normalize_pressure_unit, normalize_unit_system
from src.vde_core.cycles import default_cycle_for_legislation
from src.vde_core.db import current_db_path
from src.vde_core.repositories import fetch_vde_all_rows, fetch_vde_browser_runtime_snapshot, fetch_vde_by_id
from src.vde_core.vde_component_modes import canonical_component_mode
from src.vde_core.vde_tire_modes import canonical_tire_proposal_type
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_compact_persistence import (
    build_v22_save_plan,
    save_v22_request,
    saved_component_repositories_from_state,
)
from src.vde_core.vde_request_finalization import build_scenario_configuration_summaries, suggested_scenario_name
from src.vde_core.vde_request_compact_state import (
    V22_BASELINE_FIELDS,
    V22_MAX_PROPOSALS,
    V22_PROPOSAL_DOMAINS,
    V22_PROPOSAL_METADATA_FIELDS,
    V22_TIRE_PRESSURE_UNIT_OPTIONS,
    add_v22_proposal,
    allowed_walk_from_options,
    apply_v22_baseline,
    apply_v22_corrections,
    apply_v22_domain_inputs,
    apply_v22_new_test_baseline,
    apply_v22_proposal_metadata,
    apply_v22_proposal_matrix,
    build_v22_canonical_request_draft,
    create_v22_state,
    mark_v22_preview_stale,
    normalize_v22_state,
    proposal_type_labels_by_domain,
    remove_v22_proposal,
    resolve_v22_baseline_mass_review,
    resolve_v22_effective_baseline,
    resolve_v22_metadata_contexts,
    resolve_v22_tire_pressure_unit,
    set_v22_tire_pressure_unit_preference,
)
from src.vde_core.vde_request_contract import is_blank


V22_SESSION_KEY = "vde_setup_v22"
SECTION_OPTIONS = {
    "baseline": "Baseline & Corrections",
    "matrix": "Proposal Matrix",
    "inputs": "Request Inputs",
    "preview": "Preview & Save",
}
DOMAIN_LABELS = {
    "mass": "Mass",
    "aero": "Aero",
    "tire": "Tire",
    "transmission": "Transmission",
    "brake": "Brake",
    "axle_hubs": "Axle & Hubs",
    "parasitic": "Parasitics",
}
BASELINE_SOURCE_LABELS = {
    "EXISTING_VDE": "Existing VDE",
    "NEW_TEST": "New Test",
}
METADATA_SIMPLE_FIELDS = ("name", "description", "make", "model", "model_year")
METADATA_ADDITIONAL_FIELDS = ("category", "electrification", "transmission_type", "drive_type", "fuel_type")
METADATA_READONLY_FIELDS = ("legislation", "cycle_name")
METADATA_FIELD_LABELS = {
    "name": "Name",
    "description": "Description",
    "make": "Make",
    "model": "Model",
    "model_year": "Model Year",
    "category": "Category",
    "electrification": "Electrification",
    "transmission_type": "Transmission Type",
    "drive_type": "Drive Type",
    "fuel_type": "Fuel Type",
    "legislation": "Legislation",
    "cycle_name": "Cycle",
}
BASELINE_FIELD_META = [
    ("Mass", "mass_kg", "kg"),
    ("Mass", "test_mass_kg", "kg"),
    ("Mass", "test_mass_basis", ""),
    ("Mass", "inertia_class", "kg"),
    ("Mass", "payload_kg", "kg"),
    ("Mass", "options_kg", "kg"),
    ("Mass", "weight_dist_fr_pct", "%"),
    ("Mass", "gvwr_kg", "kg"),
    ("Mass", "gcwr_kg", "kg"),
    ("Mass", "trailer_mass_kg", "kg"),
    ("Roadload", "A", "N"),
    ("Roadload", "B", "N/kph"),
    ("Roadload", "C", "N/kph2"),
    ("Aero", "cda_m2", "m2"),
    ("Tire", "tire_db_id", ""),
    ("Tire", "tire_code", ""),
    ("Tire", "rrc_N_per_kN", "N/kN"),
    ("Tire", "front_pressure_psi", "psi"),
    ("Tire", "rear_pressure_psi", "psi"),
    ("Transmission", "transmission_component_db_id", ""),
    ("Transmission", "trans_A_coef_N", "N"),
    ("Transmission", "trans_B_coef_Npkph", "N/kph"),
    ("Transmission", "trans_C_coef_Npkph2", "N/kph2"),
    ("Brake", "brake_component_db_id", ""),
    ("Brake", "brake_A_coef_N", "N"),
    ("Brake", "brake_B_Npkph", "N/kph"),
    ("Brake", "brake_C_coef_Npkph2", "N/kph2"),
    ("Axle & Hubs", "axle_hubs_component_db_id", ""),
    ("Axle & Hubs", "axle_hub_A", "N"),
    ("Axle & Hubs", "axle_hub_B", "N/kph"),
    ("Axle & Hubs", "axle_hub_C", "N/kph2"),
    ("Parasitics", "parasitic_component_db_id", ""),
    ("Parasitics", "parasitic_A_coef_N", "N"),
    ("Parasitics", "parasitic_B_Npkph", "N/kph"),
    ("Parasitics", "parasitic_C_coef_Npkph2", "N/kph2"),
]
DOMAIN_BASELINE_FIELDS = {
    "mass": {"mass_kg", "test_mass_kg", "test_mass_basis", "weight_dist_fr_pct", "inertia_class", "payload_kg", "options_kg", "gvwr_kg", "gcwr_kg", "trailer_mass_kg"},
    "aero": {"cda_m2"},
    "tire": {"tire_db_id", "tire_code", "rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi"},
    "transmission": {"transmission_component_db_id", "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"},
    "brake": {"brake_component_db_id", "brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"},
    "axle_hubs": {"axle_hubs_component_db_id", "axle_hub_A", "axle_hub_B", "axle_hub_C"},
    "parasitic": {"parasitic_component_db_id", "parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"},
}
MASS_SIMPLE_CORRECTION_FIELDS = (
    "mass_kg",
    "test_mass_kg",
    "test_mass_basis",
    "weight_dist_fr_pct",
    "inertia_class",
    "payload_kg",
    "options_kg",
    "gvwr_kg",
    "gcwr_kg",
    "trailer_mass_kg",
)
AERO_SIMPLE_CORRECTION_FIELDS = ("cda_m2",)
TRANSMISSION_SIMPLE_CORRECTION_FIELDS = ("transmission_component_db_id", "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2")
BRAKE_SIMPLE_CORRECTION_FIELDS = ("brake_component_db_id", "brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2")
AXLE_HUBS_SIMPLE_CORRECTION_FIELDS = ("axle_hubs_component_db_id", "axle_hub_A", "axle_hub_B", "axle_hub_C")
PARASITICS_SIMPLE_CORRECTION_FIELDS = ("parasitic_component_db_id", "parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2")
SIMPLE_SHEET_DOMAINS = {"mass", "aero", "tire", "transmission", "brake", "axle_hubs", "parasitic"}
COMPONENT_SIMPLE_FIELD_CONFIG = {
    "transmission": {
        "lookup_id": "transmission_component_db_id",
        "vde_id": "transmission_vde_db_id",
        "abc": ("trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2"),
        "corrections": TRANSMISSION_SIMPLE_CORRECTION_FIELDS,
        "apply_label": "Apply Transmission",
        "form_key": "v22_transmission_inputs_form",
    },
    "brake": {
        "lookup_id": "brake_component_db_id",
        "vde_id": "brake_vde_db_id",
        "abc": ("brake_A_coef_N", "brake_B_Npkph", "brake_C_coef_Npkph2"),
        "corrections": BRAKE_SIMPLE_CORRECTION_FIELDS,
        "apply_label": "Apply Brake",
        "form_key": "v22_brake_inputs_form",
    },
    "axle_hubs": {
        "lookup_id": "axle_hubs_component_db_id",
        "vde_id": "axle_hubs_vde_db_id",
        "abc": ("axle_hub_A", "axle_hub_B", "axle_hub_C"),
        "corrections": AXLE_HUBS_SIMPLE_CORRECTION_FIELDS,
        "apply_label": "Apply Axle & Hubs",
        "form_key": "v22_axle_hubs_inputs_form",
    },
    "parasitic": {
        "lookup_id": "parasitic_component_db_id",
        "vde_id": "parasitic_vde_db_id",
        "abc": ("parasitic_A_coef_N", "parasitic_B_Npkph", "parasitic_C_coef_Npkph2"),
        "corrections": PARASITICS_SIMPLE_CORRECTION_FIELDS,
        "apply_label": "Apply Parasitics",
        "form_key": "v22_parasitics_inputs_form",
    },
}
EM_DASH = "\u2014"
_COMPARISON_COLUMN_FIELDS = {
    "Mass [kg]": "mass_kg",
    "CdA [m^2]": "CdA",
    "ABC_TOTAL A [N]": "A",
    "ABC_TOTAL B [N/kph]": "B",
    "ABC_TOTAL C [N/kph^2]": "C",
    "ABC_NET A [N]": "A",
    "ABC_NET B [N/kph]": "B",
    "ABC_NET C [N/kph^2]": "C",
    "VDE_TOTAL [MJ/km]": "vde_total_mj_per_km",
    "VDE_NET [MJ/km]": "vde_net_mj_per_km",
}
_ENGINEERING_FIELD_KEYS = {
    "Mass": "mass_kg",
    "CdA": "CdA",
    "VDE_TOTAL": "vde_total_mj_per_km",
    "VDE_NET": "vde_net_mj_per_km",
}
_LOOKUP_DISPLAY_FIELD_KEYS = {
    "RRC": "rrc_N_per_kN",
    "Mass": "mass_kg",
    "CdA": "CdA",
    "A": "A",
    "B": "B",
    "C": "C",
    "Reference pressure": "front_pressure_psi",
    "Test load": "test_mass_kg",
}
_LOOKUP_ENGINEERING_NUMERIC_COLUMNS = {"alpha", "beta", "a", "b", "c"}
V22_LAST_UNIT_SYSTEM_KEY = "_v22_last_display_unit_system"
V22_UNIT_RESET_NOTICE_KEY = "_v22_unit_reset_notice"
V22_UNIT_RESET_NOTICE = "Unapplied edits were reset when display units changed."
V22_UNIT_SENSITIVE_WIDGET_EXCLUSIONS = {"target_mass_kg"}
V22_TIRE_PRESSURE_UNIT_WIDGET_KEY = "v22_tire_pressure_unit"
V22_LAST_TIRE_PRESSURE_UNIT_KEY = "_v22_last_tire_pressure_unit"
V22_ROADLOAD_MAX_SPEED_KEY = "v22_roadload_max_speed"
V22_DOMAIN_WIDGET_DRAFTS_KEY = "_v22_request_inputs_widget_drafts"
V22_RENDERED_DOMAIN_KEY = "_v22_request_inputs_rendered_domain"
V22_RENDERED_SECTION_KEY = "_v22_request_inputs_rendered_section"
V22_COMPONENT_LOOKUP_DRAFTS_KEY = "_v22_component_lookup_drafts"


def _current_unit_system() -> str:
    return normalize_unit_system(st.session_state.get("unit_system"))


def _current_tire_pressure_unit(state: dict | None = None, unit_system: str | None = None) -> str:
    current_state = normalize_v22_state(state if state is not None else st.session_state.get(V22_SESSION_KEY))
    return resolve_v22_tire_pressure_unit(current_state, unit_system or _current_unit_system())


def _simple_widget_domain_scope(domain: str | None) -> str:
    domain_key = str(domain or "").strip()
    return "parasitics" if domain_key == "parasitic" else domain_key


def _normalize_simple_widget_domain_scope(domain: str | None) -> str:
    scope = str(domain or "").strip()
    return "parasitic" if scope == "parasitics" else scope


def _domain_widget_key(domain: str, proposal_id: str, field_key: str) -> str:
    domain_key = str(domain or "").strip()
    if domain_key in SIMPLE_SHEET_DOMAINS:
        return f"v22_simple_{_simple_widget_domain_scope(domain_key)}__{proposal_id}__{field_key}"
    return f"v22_input__{domain_key}__{proposal_id}__{field_key}"


def _candidate_domain_widget_keys(domain: str, proposal_id: str, field_key: str) -> list[str]:
    domain_key = str(domain or "").strip()
    keys = []
    if domain_key in SIMPLE_SHEET_DOMAINS:
        keys.append(f"v22_simple_{_simple_widget_domain_scope(domain_key)}__{proposal_id}__{field_key}")
    keys.append(f"v22_input__{domain_key}__{proposal_id}__{field_key}")
    return keys


def _existing_domain_widget_key(domain: str, proposal_id: str, field_key: str):
    for widget_key in _candidate_domain_widget_keys(domain, proposal_id, field_key):
        if widget_key in st.session_state:
            return widget_key
    return None


def _preserve_domain_widget_draft(domain: str) -> None:
    domain_key = str(domain or "").strip()
    if not domain_key:
        return
    scope = _simple_widget_domain_scope(domain_key)
    prefixes = (f"v22_simple_{scope}__", f"v22_input__{domain_key}__", f"v22_correction__{domain_key}__")
    drafts = dict(st.session_state.get(V22_DOMAIN_WIDGET_DRAFTS_KEY) or {})
    domain_draft = dict(drafts.get(domain_key) or {})
    for key, value in st.session_state.items():
        if str(key).startswith(prefixes):
            domain_draft[str(key)] = value
    drafts[domain_key] = domain_draft
    st.session_state[V22_DOMAIN_WIDGET_DRAFTS_KEY] = drafts
    # Reassigning before the widget is omitted keeps Streamlit from pruning a
    # staged value when the user moves to another domain.
    for key, value in domain_draft.items():
        st.session_state[key] = value


def _restore_domain_widget_draft(domain: str) -> None:
    drafts = dict(st.session_state.get(V22_DOMAIN_WIDGET_DRAFTS_KEY) or {})
    for key, value in dict(drafts.get(str(domain or "")) or {}).items():
        if key not in st.session_state:
            st.session_state[key] = value


def _component_lookup_draft(domain: str, proposal_id: str) -> dict:
    drafts = dict(st.session_state.get(V22_COMPONENT_LOOKUP_DRAFTS_KEY) or {})
    return deepcopy(dict(dict(drafts.get(str(domain or "")) or {}).get(str(proposal_id or "")) or {}))


def _stage_component_lookup_draft(domain: str, proposal_id: str, inputs: dict) -> None:
    drafts = deepcopy(dict(st.session_state.get(V22_COMPONENT_LOOKUP_DRAFTS_KEY) or {}))
    domain_drafts = dict(drafts.get(str(domain or "")) or {})
    domain_drafts[str(proposal_id or "")] = deepcopy(dict(inputs or {}))
    drafts[str(domain or "")] = domain_drafts
    st.session_state[V22_COMPONENT_LOOKUP_DRAFTS_KEY] = drafts


def _widget_target_parts(key: str) -> tuple[str | None, str | None]:
    text = str(key or "")
    if text.startswith("v22_input__"):
        parts = text.split("__")
        if len(parts) >= 4:
            return parts[1], parts[3]
    if text.startswith("v22_simple_"):
        parts = text.split("__")
        if len(parts) >= 3:
            return _normalize_simple_widget_domain_scope(parts[0].replace("v22_simple_", "", 1)), parts[2]
    if text.startswith("v22_correction__"):
        parts = text.split("__")
        if len(parts) >= 3:
            return parts[1], parts[2]
    return None, None


def _is_tire_pressure_widget(domain: str | None, field_key: str | None) -> bool:
    return str(domain or "") == "tire" and quantity_kind_for_field(field_key) == "pressure"


def _display_numeric_value(field_key: str, value, unit_system: str) -> str:
    numeric = _parse_display_number(value)
    return format_display_value_for_field(
        field_key,
        numeric,
        unit_system,
        unavailable=EM_DASH,
        pressure_unit=_current_tire_pressure_unit(unit_system=unit_system),
    )


def _display_readonly_rows(rows: list[dict], column_fields: dict[str, str], unit_system: str) -> list[dict]:
    display_rows: list[dict] = []
    for source_row in list(rows or []):
        current = {}
        for key, value in dict(source_row or {}).items():
            field_key = column_fields.get(key)
            if not field_key:
                current[key] = value
                continue
            display_key = _display_column_label(key, field_key, unit_system)
            current[display_key] = _display_numeric_value(field_key, value, unit_system)
        display_rows.append(current)
    return display_rows


def _display_column_label(label: str, field_key: str, unit_system: str) -> str:
    base = str(label).split("[", 1)[0].strip()
    unit = display_unit_for_field(field_key, unit_system, pressure_unit=_current_tire_pressure_unit(unit_system=unit_system))
    if not unit or quantity_kind_is_text_only(field_key):
        return base
    return f"{base} [{unit}]"


def quantity_kind_is_text_only(field_key: str) -> bool:
    return field_key == "target_twc_interval"


def _display_lookup_rows(rows: list[dict], unit_system: str) -> list[dict]:
    display_rows: list[dict] = []
    for source_row in list(rows or []):
        current = {}
        for key, value in dict(source_row or {}).items():
            if key == "_raw":
                continue
            if key == "Mileage":
                current["Mileage [km]"] = value
                continue
            if key == "SMERF":
                numeric = _lookup_numeric_or_none(value)
                current["SMERF"] = EM_DASH if numeric is None else f"{numeric:.10g}"
                continue
            if key in _LOOKUP_ENGINEERING_NUMERIC_COLUMNS:
                numeric = _lookup_numeric_or_none(value)
                current[key] = EM_DASH if numeric is None else f"{numeric:.10g}"
                continue
            field_key = _LOOKUP_DISPLAY_FIELD_KEYS.get(key)
            if not field_key:
                current[key] = value
                continue
            current[_display_column_label(key, field_key, unit_system)] = _display_numeric_value(field_key, value, unit_system)
        display_rows.append(current)
    return display_rows


def _lookup_numeric_or_none(value):
    try:
        if value is None:
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except Exception:
        return None


def _tire_browser_filter_rows(
    rows: list[dict],
    *,
    code_query: str = "",
    rrc_min=None,
    rrc_max=None,
    pressure_min=None,
    pressure_max=None,
    load_min=None,
    load_max=None,
    mileage_mode: str = "All",
) -> list[dict]:
    needle = str(code_query or "").strip().lower()
    filtered: list[dict] = []
    for row in list(rows or []):
        item = dict(row or {})
        haystack = " ".join(
            str(item.get(key) or "").lower()
            for key in ("lookup_id", "Tire ID", "Tire code", "Notes", "Status", "Source")
        )
        if needle and needle not in haystack:
            continue
        rrc_value = _lookup_numeric_or_none(item.get("RRC"))
        pressure_value = _lookup_numeric_or_none(item.get("Reference pressure"))
        load_value = _lookup_numeric_or_none(item.get("Test load"))
        mileage_value = _lookup_numeric_or_none(item.get("Mileage"))
        if rrc_min is not None and (rrc_value is None or rrc_value < float(rrc_min)):
            continue
        if rrc_max is not None and (rrc_value is None or rrc_value > float(rrc_max)):
            continue
        if pressure_min is not None and (pressure_value is None or pressure_value < float(pressure_min)):
            continue
        if pressure_max is not None and (pressure_value is None or pressure_value > float(pressure_max)):
            continue
        if load_min is not None and (load_value is None or load_value < float(load_min)):
            continue
        if load_max is not None and (load_value is None or load_value > float(load_max)):
            continue
        if mileage_mode == "0 km" and mileage_value != 0.0:
            continue
        if mileage_mode == ">0 km" and not (mileage_value is not None and mileage_value > 0.0):
            continue
        filtered.append(item)
    return filtered


def _paginate_lookup_rows(rows: list[dict], *, page: int, page_size: int) -> dict:
    size = max(int(page_size or 0), 1)
    total = len(list(rows or []))
    total_pages = max(int(math.ceil(total / size)) if total else 1, 1)
    current_page = min(max(int(page or 0), 0), total_pages - 1)
    start = current_page * size if total else 0
    end = min(start + size, total)
    return {
        "rows": list(rows or [])[start:end],
        "page": current_page,
        "page_size": size,
        "total": total,
        "total_pages": total_pages,
        "start": start,
        "end": end,
    }


def _tire_browser_runtime_snapshot(all_rows: list[dict], filtered_rows: list[dict]) -> dict:
    path = Path(current_db_path()).resolve()
    table_total = None
    active_total = None
    qa_codes: list[str] = []
    try:
        with sqlite3.connect(str(path), timeout=30) as con:
            cur = con.cursor()
            cur.execute("select count(*) from sqlite_master where type='table' and name='tire_roadload_db'")
            if int(cur.fetchone()[0] or 0):
                cur.execute("select count(*) from tire_roadload_db")
                table_total = int(cur.fetchone()[0] or 0)
                cur.execute("select count(*) from tire_roadload_db where coalesce(is_active, 1)=1")
                active_total = int(cur.fetchone()[0] or 0)
                cur.execute(
                    "select tire_test_code from tire_roadload_db "
                    "where tire_test_code in ('QA-BASE','QA-ECO','QA-HIGH-RRC','QA-LOAD','QA-NEUTRAL','QA-SAME-RRC-DIFF-SAE','QA-LOW-PRESSURE','QA-HIGH-PRESSURE') "
                    "order by tire_test_code"
                )
                qa_codes = [str(row[0]) for row in cur.fetchall() if row and row[0]]
    except Exception:
        pass
    return {
        "path": str(path),
        "table_total": table_total,
        "active_total": active_total,
        "total_before_filters": len(list(all_rows or [])),
        "filtered_total": len(list(filtered_rows or [])),
        "qa_codes": qa_codes,
    }


def _tire_browser_filters_active(
    *,
    code_query: str,
    rrc_min,
    rrc_max,
    pressure_min,
    pressure_max,
    load_min,
    load_max,
    mileage_mode: str,
) -> bool:
    return any(
        (
            str(code_query or "").strip(),
            rrc_min is not None,
            rrc_max is not None,
            pressure_min is not None,
            pressure_max is not None,
            load_min is not None,
            load_max is not None,
            str(mileage_mode or "All") != "All",
        )
    )


def _display_abc_triplet(payload: dict, unit_system: str) -> str:
    data = dict(payload or {})
    return " / ".join(
        [
            _display_numeric_value("A", data.get("A"), unit_system),
            _display_numeric_value("B", data.get("B"), unit_system),
            _display_numeric_value("C", data.get("C"), unit_system),
        ]
    )


def _preview_audit_rows(resolution_result: dict, unit_system: str) -> list[dict]:
    pressure_unit = _current_tire_pressure_unit(unit_system=unit_system)
    rows: list[dict] = []
    for proposal in list(dict(resolution_result or {}).get("proposal_results") or []):
        for domain_key, payload in dict(proposal.get("domain_results") or {}).items():
            item = dict(payload or {})
            rows.append(
                {
                    "Scenario": str(proposal.get("source_column") or proposal.get("proposal_id") or "Requested"),
                    "Proposal ID": str(proposal.get("proposal_id") or EM_DASH),
                    "Domain": domain_key,
                    "Status": str(item.get("status") or EM_DASH),
                    "Walk From": str(dict(proposal.get("walk_from") or {}).get("label") or dict(proposal.get("walk_from") or {}).get("column_id") or EM_DASH),
                    "Requested": format_value_map_for_display(item.get("requested_values"), unit_system, unavailable=EM_DASH, pressure_unit=pressure_unit),
                    "Resolved": format_value_map_for_display(item.get("resolved_values"), unit_system, unavailable=EM_DASH, pressure_unit=pressure_unit),
                    "Issues": str(len(list(item.get("issues") or []))),
                    "Source": str(item.get("source") or EM_DASH),
                }
            )
    return rows


def _proposal_engineering_rows_for_display(proposal_result: dict, unit_system: str) -> list[dict]:
    pressure_unit = _current_tire_pressure_unit(unit_system=unit_system)
    result = dict(proposal_result or {})
    snapshot = dict(result.get("resolved_snapshot") or {})
    resolved_mass_setup = dict(snapshot.get("resolved_mass_setup") or {})
    vde_results = dict(result.get("vde_results") or {})
    total_vde = dict(vde_results.get("total") or {})
    net_vde = dict(vde_results.get("net") or {})
    return [
        {
            "Field": _display_column_label("Mass [kg]", "mass_kg", unit_system),
            "Value": _display_numeric_value(
                "mass_kg",
                resolved_mass_setup.get("resolved_mass_used_kg") or resolved_mass_setup.get("test_mass_kg") or snapshot.get("test_mass_kg") or snapshot.get("mass_kg"),
                unit_system,
            ),
        },
        {
            "Field": _display_column_label("CdA [m^2]", "CdA", unit_system),
            "Value": format_display_value_for_field("CdA", snapshot.get("CdA"), unit_system, unavailable=EM_DASH, pressure_unit=pressure_unit),
        },
        {"Field": "ABC_TOTAL", "Value": _display_abc_triplet(result.get("abc_total") or {}, unit_system)},
        {
            "Field": _display_column_label("VDE_TOTAL [MJ/km]", "vde_total_mj_per_km", unit_system),
            "Value": _display_numeric_value("vde_total_mj_per_km", total_vde.get("mj_per_km"), unit_system),
        },
        {"Field": "ABC_NET", "Value": _display_abc_triplet(result.get("abc_net") or {}, unit_system)},
        {
            "Field": _display_column_label("VDE_NET [MJ/km]", "vde_net_mj_per_km", unit_system),
            "Value": _display_numeric_value("vde_net_mj_per_km", net_vde.get("mj_per_km"), unit_system),
        },
    ]


def _proposal_domain_change_rows_for_display(proposal_result: dict, unit_system: str) -> list[dict]:
    pressure_unit = _current_tire_pressure_unit(unit_system=unit_system)
    rows: list[dict] = []
    for domain_key in V22_PROPOSAL_DOMAINS:
        payload = dict(dict(proposal_result or {}).get("domain_results") or {}).get(domain_key)
        if not payload:
            continue
        rows.append(
            {
                "Domain": domain_key,
                "Proposal type": str(payload.get("proposal_type") or "INHERIT"),
                "Status": str(payload.get("status") or "OK"),
                "Source": str(payload.get("source") or EM_DASH),
                "Requested": format_value_map_for_display(payload.get("requested_values"), unit_system, unavailable=EM_DASH, pressure_unit=pressure_unit),
                "Resolved": format_value_map_for_display(payload.get("resolved_values"), unit_system, unavailable=EM_DASH, pressure_unit=pressure_unit),
                "Notes": " | ".join(str(item) for item in list(payload.get("notes") or []) if not is_blank(item)) or EM_DASH,
            }
        )
    return rows


def _mass_resolution_rows_for_display(proposal_result: dict, unit_system: str) -> list[dict]:
    pressure_unit = _current_tire_pressure_unit(unit_system=unit_system)
    source = dict(dict(proposal_result or {}).get("source_snapshot") or {})
    resolved = dict(dict(proposal_result or {}).get("resolved_snapshot") or {})
    rows = [
        ("Source curb mass", "mass_kg", source.get("mass_kg")),
        ("Resolved curb mass", "mass_kg", resolved.get("mass_kg")),
        ("Source TWC", "inertia_class", source.get("inertia_class")),
        ("Resolved TWC", "inertia_class", resolved.get("inertia_class")),
        ("Class interval", "target_twc_interval", resolved.get("target_twc_interval")),
        ("Resolved test mass", "test_mass_kg", resolved.get("test_mass_kg")),
    ]
    display_rows = []
    for label, field_key, value in rows:
        if is_blank(value):
            continue
        display_rows.append(
            {
                "Field": label,
                "Value": format_display_value_for_field(field_key, value, unit_system, unavailable=EM_DASH, pressure_unit=pressure_unit),
            }
        )
    return display_rows


def _tire_resolution_rows_for_display(proposal_result: dict, unit_system: str) -> list[dict]:
    tire = dict(dict(proposal_result or {}).get("domain_results") or {}).get("tire") or {}
    values = dict(tire.get("resolved_values") or {})
    if not values.get("source_tire_ABC") and not values.get("resolved_tire_ABC"):
        return []
    return [
        {"Field": "Source Tire ABC", "Value": _display_abc_triplet(values.get("source_tire_ABC") or {}, unit_system)},
        {"Field": "Resolved Tire ABC", "Value": _display_abc_triplet(values.get("resolved_tire_ABC") or {}, unit_system)},
        {"Field": "Delta Tire ABC", "Value": _display_abc_triplet(values.get("delta_tire_ABC") or {}, unit_system)},
        {"Field": "Method", "Value": values.get("tire_abc_method") or EM_DASH},
        {"Field": "Load basis", "Value": values.get("tire_load_mass_basis") or EM_DASH},
        {"Field": "Load used", "Value": _display_numeric_value("test_mass_kg", values.get("tire_load_mass_used_kg"), unit_system)},
    ]


def _parse_display_number(value):
    if value in (None, "", EM_DASH):
        return None
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return value


def _unit_sensitive_widget_field_key(widget_key: str) -> str | None:
    text = str(widget_key or "")
    if text.startswith("v22_input__"):
        parts = text.rsplit("__", 1)
        return parts[1] if len(parts) == 2 else None
    if text.startswith("v22_simple_"):
        parts = text.rsplit("__", 1)
        return parts[1] if len(parts) == 2 else None
    if text.startswith("v22_correction__"):
        parts = text.rsplit("__", 1)
        return parts[1] if len(parts) == 2 else None
    return None


def clear_v22_unit_sensitive_widget_state(session_state) -> None:
    for key in list(session_state.keys()):
        domain, field_key = _widget_target_parts(str(key))
        if not field_key or field_key in V22_UNIT_SENSITIVE_WIDGET_EXCLUSIONS or not field_uses_display_units(field_key):
            continue
        if _is_tire_pressure_widget(domain, field_key):
            continue
        session_state.pop(key, None)


def _convert_tire_pressure_widget_state(session_state, source_unit: str | None, target_unit: str | None) -> None:
    source = normalize_pressure_unit(source_unit, default="kPa")
    target = normalize_pressure_unit(target_unit, default=source)
    if source == target:
        return
    for key in list(session_state.keys()):
        domain, field_key = _widget_target_parts(str(key))
        if not _is_tire_pressure_widget(domain, field_key):
            continue
        value = session_state.get(key)
        canonical = to_canonical_field_value(field_key, value, _current_unit_system(), pressure_unit=source)
        display_value = to_display_field_value(field_key, canonical, _current_unit_system(), pressure_unit=target)
        numeric = _parse_display_number(display_value)
        if numeric is None and display_value not in (None, ""):
            continue
        session_state[key] = None if display_value is None else float(display_value)


def _handle_v22_unit_system_change(session_state) -> str | None:
    current_unit_system = normalize_unit_system(session_state.get("unit_system"))
    previous_unit_system = normalize_unit_system(session_state.get(V22_LAST_UNIT_SYSTEM_KEY))
    state = ensure_v22_session_state(session_state)
    current_pressure_unit = resolve_v22_tire_pressure_unit(state, current_unit_system)
    previous_pressure_unit = normalize_pressure_unit(
        session_state.get(V22_LAST_TIRE_PRESSURE_UNIT_KEY),
        default=current_pressure_unit,
    )
    if V22_LAST_UNIT_SYSTEM_KEY not in session_state:
        session_state[V22_LAST_UNIT_SYSTEM_KEY] = current_unit_system
        session_state[V22_LAST_TIRE_PRESSURE_UNIT_KEY] = current_pressure_unit
        return None
    if current_unit_system == previous_unit_system:
        session_state[V22_LAST_TIRE_PRESSURE_UNIT_KEY] = current_pressure_unit
        return None
    clear_v22_unit_sensitive_widget_state(session_state)
    _convert_tire_pressure_widget_state(session_state, previous_pressure_unit, current_pressure_unit)
    session_state[V22_TIRE_PRESSURE_UNIT_WIDGET_KEY] = current_pressure_unit
    session_state[V22_LAST_UNIT_SYSTEM_KEY] = current_unit_system
    session_state[V22_LAST_TIRE_PRESSURE_UNIT_KEY] = current_pressure_unit
    session_state[V22_UNIT_RESET_NOTICE_KEY] = V22_UNIT_RESET_NOTICE
    return V22_UNIT_RESET_NOTICE


def ensure_v22_session_state(session_state) -> dict:
    session_state[V22_SESSION_KEY] = normalize_v22_state(session_state.get(V22_SESSION_KEY))
    return session_state[V22_SESSION_KEY]


def render_active_v22_section(section: str, renderers: dict[str, object], *args, **kwargs):
    renderer = renderers.get(section) or renderers.get("baseline")
    if renderer is None:
        return None
    return renderer(*args, **kwargs)


def render_v22_sidebar_navigation(state: dict) -> None:
    flow = build_v22_flow_status_payload(state)
    st.markdown("**Request Flow**")
    for step in list(flow.get("steps") or []):
        if st.button(
            f"{step.get('icon')} {step.get('index')}. {step.get('label')}",
            key=f"v22_sidebar_nav__{step.get('key')}",
            use_container_width=True,
            type="primary" if step.get("is_active") else "secondary",
        ):
            _set_v22_active_section(state, str(step.get("key") or "baseline"))
        render_v22_sidebar_step_meta(step)


def render_vde_request_compact() -> None:
    state = ensure_v22_session_state(st.session_state)
    previous_section = str(st.session_state.get(V22_RENDERED_SECTION_KEY) or "")
    if previous_section == "inputs" and str(state.get("active_section") or "") != "inputs":
        _preserve_domain_widget_draft(str(st.session_state.get(V22_RENDERED_DOMAIN_KEY) or ""))
    st.session_state[V22_RENDERED_SECTION_KEY] = str(state.get("active_section") or "baseline")
    unit_reset_notice = _handle_v22_unit_system_change(st.session_state)
    _prime_correction_widget_cache(state)
    flow = build_v22_flow_status_payload(state)
    active_step = next(
        (item for item in list(flow.get("steps") or []) if str(item.get("key") or "") == str(state.get("active_section") or "baseline")),
        {},
    )
    render_v22_branding_header(build_v22_branding_payload(state))
    render_v22_context_strip(list(flow.get("context_strip") or []))
    render_v22_step_header(active_step)
    if unit_reset_notice:
        st.info(unit_reset_notice)

    start = time.perf_counter()
    render_active_v22_section(
        state["active_section"],
        {
            "baseline": render_baseline_section,
            "matrix": render_proposal_matrix_section,
            "inputs": render_request_inputs_section,
            "preview": render_preview_save_section,
        },
        state,
    )
    if str(state.get("active_section") or "") != "inputs":
        render_v22_section_pager(state)
    if st.query_params.get("v22_profile") == "1":
        st.caption(f"Render duration: {(time.perf_counter() - start) * 1000:.0f} ms")


def render_v22_section_pager(state: dict) -> None:
    has_direct_domains = bool(build_request_inputs_overview_payload(state).get("active_domain_keys") or [])
    if str(state.get("active_section") or "") == "inputs" and has_direct_domains:
        _render_v22_inputs_domain_pager(state)
        return
    active_section = str(state.get("active_section") or SECTION_ORDER[0])
    try:
        active_index = SECTION_ORDER.index(active_section)
    except ValueError:
        active_index = 0
        active_section = SECTION_ORDER[0]
    previous_section = SECTION_ORDER[active_index - 1] if active_index > 0 else None
    next_section = SECTION_ORDER[active_index + 1] if active_index < len(SECTION_ORDER) - 1 else None

    prev_col, next_col = st.columns(2)
    if prev_col.button("Previous", key=f"v22_prev__{active_section}", disabled=previous_section is None, use_container_width=True):
        _set_v22_active_section(state, previous_section)
    if next_col.button("Next", key=f"v22_next__{active_section}", disabled=next_section is None, use_container_width=True):
        _set_v22_active_section(state, next_section)
    st.caption("Previous / Next changes only the active section. It does not apply inputs or regenerate preview.")


def _render_v22_inputs_domain_pager(state: dict) -> None:
    domains = list(build_request_inputs_overview_payload(state).get("active_domain_keys") or [])
    if not domains:
        return
    selected = str(st.session_state.get("v22_request_inputs_active_domain") or domains[0])
    if selected not in domains:
        selected = domains[0]
    index = domains.index(selected)
    previous_domain = domains[index - 1] if index else None
    next_domain = domains[index + 1] if index < len(domains) - 1 else None
    previous_label = f"Previous: {DOMAIN_LABELS[previous_domain]}" if previous_domain else "Previous: Proposal Matrix"
    next_label = f"Next: {DOMAIN_LABELS[next_domain]}" if next_domain else "Next: Preview"
    prev_col, next_col = st.columns(2)
    if previous_domain:
        prev_col.button(
            previous_label,
            key=f"v22_inputs_prev__{selected}",
            use_container_width=True,
            on_click=_set_v22_active_domain_from_pager,
            args=(selected, previous_domain),
        )
    elif prev_col.button(previous_label, key=f"v22_inputs_prev__{selected}", use_container_width=True):
        _set_v22_active_section(state, "matrix")
    if next_domain:
        next_col.button(
            next_label,
            key=f"v22_inputs_next__{selected}",
            use_container_width=True,
            on_click=_set_v22_active_domain_from_pager,
            args=(selected, next_domain),
        )
    elif next_col.button(next_label, key=f"v22_inputs_next__{selected}", use_container_width=True):
        _set_v22_active_section(state, "preview")
    st.caption("Previous / Next changes navigation only. It does not apply inputs or regenerate preview.")


def _set_v22_active_domain_from_pager(current_domain: str, target_domain: str) -> None:
    _preserve_domain_widget_draft(current_domain)
    st.session_state["v22_request_inputs_active_domain"] = target_domain


def _set_v22_active_section(state: dict, section: str | None) -> None:
    next_section = str(section or SECTION_ORDER[0])
    if next_section not in SECTION_ORDER:
        next_section = SECTION_ORDER[0]
    current = normalize_v22_state(state)
    if str(current.get("active_section") or SECTION_ORDER[0]) == next_section:
        return
    current["active_section"] = next_section
    st.session_state[V22_SESSION_KEY] = current
    st.rerun()


def _resolved_runtime_db_path() -> str:
    return str(Path(current_db_path()).resolve())


@st.cache_data(show_spinner=False)
def _baseline_summary_rows_cached(db_path_signature: str) -> list[dict]:
    rows = fetch_vde_all_rows()
    summary = []
    for row in rows:
        summary.append(
            {
                "VDE ID": row.get("id"),
                "Make": row.get("make"),
                "Model": row.get("model"),
                "Year": row.get("year"),
                "Legislation": row.get("legislation"),
                "Cycle": row.get("cycle") or row.get("cycle_name"),
                "Test mass": row.get("test_mass_kg") or row.get("mass_kg"),
                "ABC_TOTAL": _abc_label(row),
                "VDE_TOTAL": row.get("vde_total") or row.get("vde_total_mj_per_km") or row.get("vde_net_mj_per_km"),
                "Notes": row.get("notes"),
            }
        )
    return summary


def _baseline_summary_rows() -> list[dict]:
    return _baseline_summary_rows_cached(_resolved_runtime_db_path())


_baseline_summary_rows.clear = _baseline_summary_rows_cached.clear


def _sync_baseline_browser_runtime_db_state() -> str:
    active_path = _resolved_runtime_db_path()
    previous_path = str(st.session_state.get("_v22_baseline_browser_db_path") or "")
    if previous_path and previous_path != active_path:
        _baseline_summary_rows.clear()
        for key in (
            "v22_baseline_selector",
            "v22_baseline_selector_empty",
            "v22_filter_legislation",
            "v22_filter_make",
            "v22_filter_year",
            "v22_filter_model",
        ):
            st.session_state.pop(key, None)
    st.session_state["_v22_baseline_browser_db_path"] = active_path
    return active_path


def _baseline_browser_runtime_snapshot(filtered_rows: pd.DataFrame) -> dict:
    snapshot = dict(fetch_vde_browser_runtime_snapshot() or {})
    snapshot["total_before_filters"] = len(_baseline_summary_rows())
    snapshot["filtered_total"] = 0 if filtered_rows is None else int(len(filtered_rows))
    return snapshot


def _baseline_source_type(state: dict | None) -> str:
    normalized = normalize_v22_state(state)
    return str(dict(normalized.get("baseline") or {}).get("source_type") or "EXISTING_VDE").strip().upper() or "EXISTING_VDE"


def _baseline_input_lane_label(state: dict | None) -> str:
    return "Input" if _baseline_source_type(state) == "NEW_TEST" else "Correction"


def _baseline_new_test_widget_key(field_key: str) -> str:
    return f"v22_baseline_new_test__{field_key}"


def _is_baseline_new_test_numeric_field(field_key: str) -> bool:
    return field_key in {"A", "B", "C", "test_mass_kg"}


def _safe_optional_numeric(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except Exception:
            return None
        return numeric if math.isfinite(numeric) else None
    return None


def _prime_baseline_new_test_widget(field_key: str, canonical_value) -> None:
    widget_key = _baseline_new_test_widget_key(field_key)
    if _is_baseline_new_test_numeric_field(field_key):
        if widget_key in st.session_state:
            st.session_state[widget_key] = _safe_optional_numeric(st.session_state[widget_key])
            return
        display_value = to_display_field_value(field_key, canonical_value, _current_unit_system())
        st.session_state[widget_key] = _safe_optional_numeric(display_value)
        return
    if widget_key in st.session_state:
        return
    st.session_state[widget_key] = "" if is_blank(canonical_value) else str(canonical_value)


def _render_baseline_new_test_form(state: dict) -> tuple[dict, list[str]]:
    normalized = normalize_v22_state(state)
    baseline = dict(normalized.get("baseline") or {})
    source_snapshot = dict(baseline.get("source_snapshot") or {})
    seed = deepcopy(source_snapshot if _baseline_source_type(state) == "NEW_TEST" else {})
    current_legislation = str(
        st.session_state.get(_baseline_new_test_widget_key("legislation"))
        or seed.get("legislation")
        or "EPA"
    ).strip().upper() or "EPA"
    default_cycle = str(seed.get("cycle_name") or default_cycle_for_legislation(current_legislation) or "").strip()
    for field_key, fallback in (
        ("A", seed.get("A")),
        ("B", seed.get("B")),
        ("C", seed.get("C")),
        ("test_mass_kg", seed.get("test_mass_kg")),
        ("legislation", current_legislation),
        ("cycle_name", default_cycle),
        ("notes", seed.get("notes")),
    ):
        _prime_baseline_new_test_widget(field_key, fallback)

    legislation_spec = metadata_field_spec("legislation", legislation=current_legislation, category="", current_value=current_legislation)
    legislation_options = [option for option in list(legislation_spec.get("options") or ["EPA", "WLTP", "ABNT (Brazil)"]) if option != "(inherit)"] or ["EPA", "WLTP", "ABNT (Brazil)"]
    if current_legislation not in legislation_options:
        legislation_options.append(current_legislation)
    cycle_widget_key = _baseline_new_test_widget_key("cycle_name")
    if not str(st.session_state.get(cycle_widget_key) or "").strip():
        suggested_cycle = str(default_cycle_for_legislation(current_legislation) or "").strip()
        if suggested_cycle:
            st.session_state[cycle_widget_key] = suggested_cycle

    unit_system = _current_unit_system()
    top = st.columns([1.0, 1.0, 1.0, 1.0])
    A_value = _safe_optional_numeric(st.session_state.get(_baseline_new_test_widget_key("A")))
    B_value = _safe_optional_numeric(st.session_state.get(_baseline_new_test_widget_key("B")))
    C_value = _safe_optional_numeric(st.session_state.get(_baseline_new_test_widget_key("C")))
    test_mass_value = _safe_optional_numeric(st.session_state.get(_baseline_new_test_widget_key("test_mass_kg")))
    A_display = top[0].number_input(
        "ABC_TOTAL A",
        key=_baseline_new_test_widget_key("A"),
        value=A_value,
        step=display_step_for_field("A", field_meta("A").get("step"), unit_system),
        format=display_format_for_field("A", field_meta("A").get("format"), unit_system),
        placeholder="-",
    )
    B_display = top[1].number_input(
        "ABC_TOTAL B",
        key=_baseline_new_test_widget_key("B"),
        value=B_value,
        step=display_step_for_field("B", field_meta("B").get("step"), unit_system),
        format=display_format_for_field("B", field_meta("B").get("format"), unit_system),
        placeholder="-",
    )
    C_display = top[2].number_input(
        "ABC_TOTAL C",
        key=_baseline_new_test_widget_key("C"),
        value=C_value,
        step=display_step_for_field("C", field_meta("C").get("step"), unit_system),
        format=display_format_for_field("C", field_meta("C").get("format"), unit_system),
        placeholder="-",
    )
    test_mass_display = top[3].number_input(
        "Test Mass",
        key=_baseline_new_test_widget_key("test_mass_kg"),
        value=test_mass_value,
        step=display_step_for_field("test_mass_kg", field_meta("test_mass_kg").get("step"), unit_system),
        format=display_format_for_field("test_mass_kg", field_meta("test_mass_kg").get("format"), unit_system),
        placeholder="-",
    )

    bottom = st.columns([1.0, 1.2, 1.8])
    legislation = bottom[0].selectbox(
        "Legislation",
        legislation_options,
        index=legislation_options.index(current_legislation) if current_legislation in legislation_options else 0,
        key=_baseline_new_test_widget_key("legislation"),
    )
    if not str(st.session_state.get(cycle_widget_key) or "").strip():
        suggested_cycle = str(default_cycle_for_legislation(legislation) or "").strip()
        if suggested_cycle:
            st.session_state[cycle_widget_key] = suggested_cycle
    cycle_name = bottom[1].text_input(
        "Cycle",
        key=cycle_widget_key,
        value=str(st.session_state.get(cycle_widget_key) or ""),
    )
    notes = bottom[2].text_input(
        "Note",
        key=_baseline_new_test_widget_key("notes"),
        value=str(st.session_state.get(_baseline_new_test_widget_key("notes")) or ""),
    )

    payload = {
        "A": to_canonical_field_value("A", A_display, unit_system),
        "B": to_canonical_field_value("B", B_display, unit_system),
        "C": to_canonical_field_value("C", C_display, unit_system),
        "test_mass_kg": to_canonical_field_value("test_mass_kg", test_mass_display, unit_system),
        "legislation": legislation,
        "cycle_name": str(cycle_name or "").strip(),
        "notes": str(notes or "").strip(),
    }
    issues: list[str] = []
    for field_key, label in (("A", "ABC_TOTAL A"), ("B", "ABC_TOTAL B"), ("C", "ABC_TOTAL C"), ("test_mass_kg", "Test Mass")):
        if is_blank(payload.get(field_key)):
            issues.append(f"{label} is required.")
    if is_blank(payload.get("legislation")):
        issues.append("Legislation is required.")
    if is_blank(payload.get("cycle_name")):
        issues.append("Cycle is required.")
    try:
        if payload.get("A") is not None and float(payload["A"]) < 0:
            issues.append("ABC_TOTAL A cannot be negative.")
    except Exception:
        issues.append("ABC_TOTAL A must be numeric.")
    try:
        if payload.get("C") is not None and float(payload["C"]) < 0:
            issues.append("ABC_TOTAL C cannot be negative.")
    except Exception:
        issues.append("ABC_TOTAL C must be numeric.")
    try:
        if payload.get("test_mass_kg") is not None and float(payload["test_mass_kg"]) <= 0:
            issues.append("Test Mass must be greater than zero.")
    except Exception:
        issues.append("Test Mass must be numeric.")
    return payload, issues


def render_baseline_section(state: dict) -> None:
    st.subheader("Baseline")
    _sync_baseline_browser_runtime_db_state()
    rows = _baseline_summary_rows()
    df = pd.DataFrame(rows)
    baseline = dict(state.get("baseline") or {})
    source_type = _baseline_source_type(state)
    source_labels = list(BASELINE_SOURCE_LABELS.values())
    source_label = BASELINE_SOURCE_LABELS.get(source_type, BASELINE_SOURCE_LABELS["EXISTING_VDE"])
    source_cols = st.columns([1.4, 4.6])
    selected_source_label = source_cols[0].radio(
        "Source",
        source_labels,
        index=source_labels.index(source_label) if source_label in source_labels else 0,
        key="v22_baseline_source_selector",
        horizontal=True,
    )
    selected_source_type = next((key for key, value in BASELINE_SOURCE_LABELS.items() if value == selected_source_label), "EXISTING_VDE")
    legislation, make, year, model_text = _current_baseline_filter_values(df)
    filtered = _apply_summary_filters(df, legislation=legislation, make=make, year=year, model_text=model_text)
    selected_vde_id = _current_candidate_selection(filtered)
    selected_label = _baseline_option_label(filtered, selected_vde_id) if not filtered.empty and not is_blank(selected_vde_id) else ""
    candidate_payload = build_baseline_candidate_status_payload(state, selected_vde_id, selected_label=selected_label)
    if selected_source_type != "EXISTING_VDE":
        selected_vde_id = None
        candidate_payload = {"loaded_baseline_id": None, "candidate_differs": False, "warning_message": ""}

    st.markdown("**Candidate baseline**")
    load_pressed = False
    if selected_source_type == "EXISTING_VDE":
        selector_cols = st.columns([4.2, 1.1])
        if filtered.empty:
            selector_cols[0].selectbox(
                "Candidate baseline",
                ["No baseline rows match the current filters."],
                index=0,
                disabled=True,
                key="v22_baseline_selector_empty",
            )
            load_pressed = selector_cols[1].button("Load baseline", key="v22_load_baseline", disabled=True, use_container_width=True)
        else:
            selector_cols[0].selectbox(
                "Candidate baseline",
                filtered["VDE ID"].tolist(),
                format_func=lambda value: _baseline_option_label(filtered, value),
                key="v22_baseline_selector",
                label_visibility="collapsed",
            )
            load_pressed = selector_cols[1].button("Load baseline", key="v22_load_baseline", use_container_width=True)
    else:
        new_test_payload, new_test_issues = _render_baseline_new_test_form(state)
        st.caption("Measured ABC_TOTAL is authoritative for this baseline. Transmission remains unresolved until explicitly provided elsewhere.")
        if st.button("Load baseline", key="v22_load_new_test_baseline", use_container_width=False):
            if new_test_issues:
                for message in new_test_issues:
                    render_v22_notice_strip(message, tone="warning")
            else:
                clear_v22_correction_widget_state(st.session_state)
                st.session_state[V22_SESSION_KEY] = apply_v22_new_test_baseline(state, new_test_payload)
                st.rerun()

    candidate_meta = st.columns(2)
    candidate_meta[0].caption(f"Selected candidate: VDE #{selected_vde_id}" if not is_blank(selected_vde_id) else "Selected candidate: —")
    candidate_meta[1].caption(
        f"Loaded baseline: VDE #{candidate_payload.get('loaded_baseline_id')}"
        if not is_blank(candidate_payload.get("loaded_baseline_id"))
        else "Loaded baseline: —"
    )
    if candidate_payload.get("candidate_differs"):
        render_v22_notice_strip(str(candidate_payload.get("warning_message") or ""), tone="warning")

    if selected_source_type == "EXISTING_VDE" and load_pressed and not filtered.empty and not is_blank(selected_vde_id):
        row = fetch_vde_by_id(int(selected_vde_id))
        clear_v22_correction_widget_state(st.session_state)
        st.session_state[V22_SESSION_KEY] = apply_v22_baseline(state, row)
        st.rerun()

    if not baseline.get("loaded"):
        st.info("No baseline loaded. Select a candidate and load it to start the request.")
    else:
        summary_payload = build_loaded_baseline_summary_payload(state, _current_unit_system())
        st.markdown("**Loaded Baseline**")
        loaded_top = st.columns(2)
        loaded_top[0].caption("New Test baseline" if _baseline_source_type(state) == "NEW_TEST" else f"VDE #{summary_payload.get('baseline_id') or EM_DASH}")
        loaded_top[1].caption(f"Status: {summary_payload.get('status') or 'Loaded'}")
        render_v22_summary_groups(list(summary_payload.get("groups") or []))
        if str(summary_payload.get("notes") or "").strip():
            st.caption(f"Notes: {summary_payload.get('notes')}")

        corrections_payload = build_active_corrections_summary(state, _current_unit_system())
        st.markdown("**Active Baseline Corrections**")
        st.caption("Baseline corrections are edited in Request Inputs.")
        if int(corrections_payload.get("count") or 0) <= 0:
            st.info(str(corrections_payload.get("empty_message") or "No active baseline corrections."))
        else:
            st.caption(f"{int(corrections_payload.get('count') or 0)} active corrections")
            render_v22_chip_list(
                [
                    f"{row.get('domain')} · {', '.join(list(row.get('fields') or []))}"
                    for row in list(corrections_payload.get("domain_rows") or [])
                ]
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Domain": item.get("domain"),
                            "Field": item.get("field_label"),
                            "Printed": item.get("printed_value"),
                            "Effective": item.get("effective_value"),
                        }
                        for item in list(corrections_payload.get("entries") or [])
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    if selected_source_type == "EXISTING_VDE":
        with st.expander("Browse VDE Database", expanded=False):
            runtime_snapshot = _baseline_browser_runtime_snapshot(filtered)
            st.caption(f"VDE Browser DB: {runtime_snapshot.get('path') or EM_DASH}")
            st.caption(
                "VDE rows: "
                f"{runtime_snapshot.get('row_count') if runtime_snapshot.get('row_count') is not None else EM_DASH} "
                f"(showing {runtime_snapshot.get('filtered_total', 0)} of {runtime_snapshot.get('total_before_filters', 0)})"
            )
            filters = st.columns([1.0, 1.0, 1.0, 1.4])
            filters[0].selectbox("Legislation", _filter_options(df, "Legislation"), key="v22_filter_legislation")
            filters[1].selectbox("Make", _filter_options(df, "Make"), key="v22_filter_make")
            filters[2].text_input("Year", key="v22_filter_year")
            filters[3].text_input("Model contains", key="v22_filter_model")
            if filtered.empty:
                st.info("No baseline rows match the current filters.")
            else:
                st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)


def render_proposal_matrix_section(state: dict) -> None:
    st.subheader("Proposal Matrix")
    proposals = list(state.get("proposals") or [])
    top = st.columns([1, 4])
    if top[0].button("+ Add proposal", key="v22_add_proposal", disabled=len(proposals) >= V22_MAX_PROPOSALS):
        st.session_state[V22_SESSION_KEY] = add_v22_proposal(state)
        st.rerun()
    top[1].caption(f"{len(proposals)} of {V22_MAX_PROPOSALS} proposals configured.")

    labels_by_domain = proposal_type_labels_by_domain()
    matrix_rows = []
    with st.form("v22_proposal_matrix_form"):
        header = st.columns([0.9, 1.2, 1, 1, 1, 1.15, 1, 1.1, 1.1, 0.7])
        for cell, label in zip(header, ["Proposal", "Walk From", "Mass", "Aero", "Tire", "Transmission", "Brake", "Axle & Hubs", "Parasitics", "Remove"]):
            cell.caption(label)
        for proposal in proposals:
            proposal_id = str(proposal.get("proposal_id") or "")
            row_cols = st.columns([0.9, 1.2, 1, 1, 1, 1.15, 1, 1.1, 1.1, 0.7])
            row_cols[0].write(proposal_display_label(state, proposal))
            walk_options = allowed_walk_from_options(state, proposal_id)
            current_walk = str(proposal.get("walk_from") or "baseline")
            if current_walk not in walk_options:
                walk_options = [current_walk, *walk_options]
            walk_from = row_cols[1].selectbox(
                "Walk From",
                walk_options,
                index=walk_options.index(current_walk),
                key=f"v22_matrix_walk_{proposal_id}",
                label_visibility="collapsed",
                format_func=_walk_label(state),
            )
            row = {"proposal_id": proposal_id, "walk_from": walk_from}
            for offset, domain in enumerate(V22_PROPOSAL_DOMAINS, start=2):
                options = labels_by_domain.get(domain, ["Inherit"])
                current = _selection_mode_for_domain(proposal, domain)
                if current not in options:
                    current = "Inherit"
                row[domain] = row_cols[offset].selectbox(
                    DOMAIN_LABELS[domain],
                    options,
                    index=options.index(current),
                    key=f"v22_matrix_{proposal_id}_{domain}",
                    label_visibility="collapsed",
                )
            row["remove"] = row_cols[9].checkbox("Remove", key=f"v22_matrix_remove_{proposal_id}", label_visibility="collapsed")
            matrix_rows.append(row)
        submitted = st.form_submit_button("Apply Proposal Matrix")
    if submitted:
        next_state = apply_v22_proposal_matrix(state, matrix_rows)
        _clear_widget_state_after_matrix_change(state, next_state, st.session_state)
        st.session_state[V22_SESSION_KEY] = next_state
        st.rerun()

    invalid = _invalid_walk_from_rows(state)
    if invalid:
        st.warning("Some Walk From references are no longer valid: " + ", ".join(invalid))


def render_request_inputs_section(state: dict) -> None:
    start = time.perf_counter()
    st.subheader("Request Inputs")
    st.caption("Configure and apply one engineering domain at a time.")
    overview_payload = build_request_inputs_overview_payload(state)
    render_v22_request_inputs_overview(overview_payload)

    baseline_payload = dict(state.get("baseline") or {})
    printed_baseline = dict(baseline_payload.get("printed") or {})
    effective_baseline = dict(baseline_payload.get("effective") or {})
    corrections = dict(baseline_payload.get("corrections") or {})
    unit_system = _current_unit_system()
    if not overview_payload.get("has_active_domains"):
        st.info("Direct proposal domains will appear here after the Proposal Matrix is configured.")
        render_v22_section_pager(state)
        _render_request_inputs_secondary(overview_payload, baseline_payload)
        if st.query_params.get("v22_profile") == "1":
            st.caption(f"Request Inputs render: {(time.perf_counter() - start) * 1000:.0f} ms")
        return

    active_domains = list(overview_payload.get("active_domain_keys") or [])
    selected_domain = st.radio(
        "Active domain",
        active_domains,
        horizontal=True,
        format_func=lambda value: DOMAIN_LABELS.get(value, str(value).replace("_", " ").title()),
        key="v22_request_inputs_active_domain",
        label_visibility="collapsed",
    )
    previous_domain = str(st.session_state.get(V22_RENDERED_DOMAIN_KEY) or "")
    if previous_domain and previous_domain != selected_domain:
        _preserve_domain_widget_draft(previous_domain)
    _restore_domain_widget_draft(selected_domain)
    st.session_state[V22_RENDERED_DOMAIN_KEY] = selected_domain
    _render_request_inputs_domain(
        state,
        selected_domain,
        unit_system=unit_system,
        printed_baseline=printed_baseline,
        effective_baseline=effective_baseline,
        corrections=corrections,
    )
    render_v22_section_pager(state)
    _render_request_inputs_secondary(overview_payload, baseline_payload)

    if st.query_params.get("v22_profile") == "1":
        for domain in list(overview_payload.get("active_domain_keys") or []):
            _render_applied_input_debug(state, domain)
        st.caption(f"Request Inputs render: {(time.perf_counter() - start) * 1000:.0f} ms")


def _render_request_inputs_domain(
    state: dict,
    domain: str,
    *,
    unit_system: str,
    printed_baseline: dict,
    effective_baseline: dict,
    corrections: dict,
) -> None:
    card_payload = build_domain_card_payload(state, domain, unit_system)
    render_v22_domain_card_header(card_payload)
    _render_domain_apply_feedback(card_payload)
    if domain == "tire":
        unit_columns = st.columns([4.0, 1.0])
        unit_columns[0].caption("Tire input pressure")
        _render_tire_pressure_unit_selector(state, container=unit_columns[1])
    if active_domain_has_lookup_requests(state, domain):
        render_v22_group_header(f"1. Select {DOMAIN_LABELS[domain]} component")
        _render_lookup_panel(state, domain)
    render_v22_group_header("2. Review requested values")
    renderer = {
        "mass": render_mass_inputs_simple,
        "aero": render_aero_inputs_simple,
        "tire": render_tire_inputs_simple,
        "transmission": render_transmission_inputs_simple,
        "brake": render_brake_inputs_simple,
        "axle_hubs": render_axle_hubs_inputs_simple,
        "parasitic": render_parasitics_inputs_simple,
    }.get(domain)
    if renderer is None:
        render_v22_notice_strip(
            f"No Request Inputs renderer is configured for {DOMAIN_LABELS.get(domain, domain)}.",
            tone="warning",
        )
    else:
        renderer(
            state,
            card_payload,
            printed_baseline=printed_baseline,
            effective_baseline=effective_baseline,
            corrections=corrections,
        )


def _render_request_inputs_reference_controls(baseline_payload: dict) -> None:
    with st.expander("Request options", expanded=False):
        st.selectbox(
            "Reference adjustment scope",
            ["request_only", "save_as_new_baseline"],
            index=["request_only", "save_as_new_baseline"].index(str(baseline_payload.get("correction_disposition") or "request_only")),
            format_func=lambda value: "Use only in this request" if value == "request_only" else "Save corrected baseline as a new VDE line",
            key="v22_correction_disposition",
        )


def _render_inactive_domain_summary(overview_payload: dict) -> None:
    inactive = [dict(item) for item in list(overview_payload.get("inactive_domains") or [])]
    if not inactive:
        return
    rows = []
    for item in inactive:
        summary = str(item.get("proposal_type_summary") or "").strip() or "Inherit"
        if item.get("inactive_summary"):
            summary = "; ".join(list(item.get("inactive_summary") or []))
        rows.append({"Domain": item.get("label"), "Configuration": summary})
    with st.expander(f"Inactive / inherited domains · {len(inactive)}", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_request_inputs_secondary(overview_payload: dict, baseline_payload: dict) -> None:
    _render_request_inputs_reference_controls(baseline_payload)
    _render_inactive_domain_summary(overview_payload)


def _render_domain_apply_feedback(card_payload: dict) -> None:
    status_key = str(card_payload.get("status_key") or "pending")
    if status_key == "stale":
        render_v22_apply_result("Proposal configuration changed. Re-apply this domain.", tone="stale")
        return
    last_message = str(card_payload.get("last_apply_message") or "").strip()
    if last_message:
        if status_key != "ready":
            render_v22_apply_result(last_message, tone="review" if status_key == "review" else "pending")
        return
    if status_key == "pending":
        render_v22_apply_result("Pending. Apply this domain when the inputs are ready.", tone="pending")


def _apply_tire_pressure_unit_override(session_state, pressure_unit: str) -> None:
    state = ensure_v22_session_state(session_state)
    unit_system = normalize_unit_system(session_state.get("unit_system"))
    source_unit = resolve_v22_tire_pressure_unit(state, unit_system)
    target_unit = normalize_pressure_unit(pressure_unit, default=source_unit)
    _convert_tire_pressure_widget_state(session_state, source_unit, target_unit)
    session_state[V22_SESSION_KEY] = set_v22_tire_pressure_unit_preference(state, target_unit)
    session_state[V22_LAST_TIRE_PRESSURE_UNIT_KEY] = target_unit


def _render_tire_pressure_unit_selector(state: dict, *, container=None) -> None:
    current_unit = _current_tire_pressure_unit(state)
    _prime_widget_value(V22_TIRE_PRESSURE_UNIT_WIDGET_KEY, current_unit)
    target = container or st
    preferred_order = ("psi", "bar", "kPa")
    available_units = set(V22_TIRE_PRESSURE_UNIT_OPTIONS or PRESSURE_UNIT_OPTIONS)
    selected_unit = target.radio(
        "Pressure unit",
        [unit for unit in preferred_order if unit in available_units],
        key=V22_TIRE_PRESSURE_UNIT_WIDGET_KEY,
        horizontal=True,
    )
    if selected_unit != current_unit:
        _apply_tire_pressure_unit_override(st.session_state, selected_unit)
        st.rerun()


def render_mass_inputs_simple(
    state: dict,
    card_payload: dict,
    *,
    printed_baseline: dict,
    effective_baseline: dict,
    corrections: dict,
) -> None:
    domain = "mass"
    printed_display = _baseline_domain_display(domain, printed_baseline)
    effective_display, proposal_contexts = _domain_contexts(domain, state, effective_baseline)
    proposals = list(state.get("proposals") or [])
    correction_values = {field_key: corrections.get(field_key) for field_key in MASS_SIMPLE_CORRECTION_FIELDS}
    values_by_proposal: dict[str, dict] = {}
    debug_widget_keys = None
    proposal_specs = []

    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id") or "")
        context = dict(proposal_contexts.get(proposal_id) or {})
        editable_inputs = deepcopy(dict(context.get("inputs") or {}))
        values_by_proposal[proposal_id] = editable_inputs
        proposal_specs.append(
            {
                "proposal": proposal,
                "proposal_id": proposal_id,
                "context": context,
                "editable_inputs": editable_inputs,
                "proposal_type": str(context.get("proposal_type") or "INHERIT"),
                "selection_mode": str(context.get("selection_mode") or context.get("proposal_type") or "INHERIT"),
            }
        )

    with st.form("v22_mass_inputs_form"):
        render_v22_reference_divider()
        _render_simple_sheet_header(state, domain, proposal_specs)
        for row_key in _mass_simple_sheet_rows(proposal_specs):
            _render_mass_simple_sheet_row(
                state,
                row_key,
                proposal_specs,
                printed_display,
                effective_display,
                correction_values,
                debug_widget_keys=debug_widget_keys,
            )
        submitted = st.form_submit_button("Apply Mass")

    _render_mass_resolved_audit(proposal_specs)

    if not submitted:
        return

    merged_corrections = _merge_simple_corrections(corrections, MASS_SIMPLE_CORRECTION_FIELDS, correction_values)
    payload = build_v22_domain_apply_payload(domain, proposals, values_by_proposal)
    next_state = apply_v22_corrections(state, merged_corrections)
    next_state["baseline"]["correction_disposition"] = str(st.session_state.get("v22_correction_disposition", "request_only") or "request_only")
    next_state = apply_v22_domain_inputs(next_state, domain, payload)
    st.session_state[V22_SESSION_KEY] = next_state


def render_aero_inputs_simple(
    state: dict,
    card_payload: dict,
    *,
    printed_baseline: dict,
    effective_baseline: dict,
    corrections: dict,
) -> None:
    domain = "aero"
    printed_display = _baseline_domain_display(domain, printed_baseline)
    effective_display, proposal_contexts = _domain_contexts(domain, state, effective_baseline)
    proposals = list(state.get("proposals") or [])
    correction_values = {field_key: corrections.get(field_key) for field_key in AERO_SIMPLE_CORRECTION_FIELDS}
    values_by_proposal: dict[str, dict] = {}
    debug_widget_keys = None
    proposal_specs = []

    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id") or "")
        context = dict(proposal_contexts.get(proposal_id) or {})
        editable_inputs = deepcopy(dict(context.get("inputs") or {}))
        values_by_proposal[proposal_id] = editable_inputs
        proposal_specs.append(
            {
                "proposal": proposal,
                "proposal_id": proposal_id,
                "context": context,
                "editable_inputs": editable_inputs,
                "proposal_type": str(context.get("proposal_type") or "INHERIT"),
                "selection_mode": str(context.get("selection_mode") or context.get("proposal_type") or "INHERIT"),
            }
        )

    with st.form("v22_aero_inputs_form"):
        render_v22_reference_divider()
        _render_simple_sheet_header(state, domain, proposal_specs)
        for row_key in ("cda_m2", "delta_CdA"):
            _render_aero_simple_sheet_row(
                row_key,
                proposal_specs,
                printed_display,
                effective_display,
                correction_values,
                debug_widget_keys=debug_widget_keys,
            )
        submitted = st.form_submit_button("Apply Aero")

    if not submitted:
        return

    merged_corrections = _merge_simple_corrections(corrections, AERO_SIMPLE_CORRECTION_FIELDS, correction_values)
    payload = build_v22_domain_apply_payload(domain, proposals, values_by_proposal)
    next_state = apply_v22_corrections(state, merged_corrections)
    next_state["baseline"]["correction_disposition"] = str(st.session_state.get("v22_correction_disposition", "request_only") or "request_only")
    next_state = apply_v22_domain_inputs(next_state, domain, payload)
    st.session_state[V22_SESSION_KEY] = next_state


def render_tire_inputs_simple(
    state: dict,
    card_payload: dict,
    *,
    printed_baseline: dict,
    effective_baseline: dict,
    corrections: dict,
) -> None:
    domain = "tire"
    printed_display = _baseline_domain_display(domain, printed_baseline)
    effective_display, proposal_contexts = _domain_contexts(domain, state, effective_baseline)
    proposals = list(state.get("proposals") or [])
    correction_field_keys = tuple(card_payload.get("correction_field_keys") or ())
    correction_values = {field_key: corrections.get(field_key) for field_key in correction_field_keys}
    values_by_proposal: dict[str, dict] = {}
    debug_widget_keys = None
    proposal_specs = []

    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id") or "")
        context = dict(proposal_contexts.get(proposal_id) or {})
        proposal_type = canonical_tire_proposal_type(str(context.get("proposal_type") or "INHERIT"))
        selection_mode = str(context.get("selection_mode") or context.get("proposal_type") or "INHERIT")
        editable_inputs = deepcopy(dict(context.get("inputs") or {}))
        editable_inputs.update(_current_widget_inputs(domain, proposal_id, proposal_type, selection_mode, editable_inputs))
        values_by_proposal[proposal_id] = editable_inputs
        proposal_specs.append(
            {
                "proposal": proposal,
                "proposal_id": proposal_id,
                "context": context,
                "editable_inputs": editable_inputs,
                "proposal_type": proposal_type,
                "selection_mode": selection_mode,
            }
        )

    with st.form("v22_tire_inputs_form"):
        render_v22_reference_divider()
        _render_simple_sheet_header(state, domain, proposal_specs)
        for row_key in _tire_simple_sheet_rows(proposal_specs, correction_field_keys):
            _render_tire_simple_sheet_row(
                row_key,
                proposal_specs,
                printed_display,
                effective_display,
                correction_values,
                debug_widget_keys=debug_widget_keys,
            )
        submitted = st.form_submit_button("Apply Tire")

    _render_tire_resolved_audit(proposal_specs)

    if not submitted:
        return

    merged_corrections = _merge_simple_corrections(corrections, correction_field_keys, correction_values)
    payload = build_v22_domain_apply_payload(domain, proposals, values_by_proposal)
    next_state = apply_v22_corrections(state, merged_corrections)
    next_state["baseline"]["correction_disposition"] = str(st.session_state.get("v22_correction_disposition", "request_only") or "request_only")
    next_state = apply_v22_domain_inputs(next_state, domain, payload)
    st.session_state[V22_SESSION_KEY] = next_state


def render_transmission_inputs_simple(
    state: dict,
    card_payload: dict,
    *,
    printed_baseline: dict,
    effective_baseline: dict,
    corrections: dict,
) -> None:
    _render_component_inputs_simple(
        state,
        domain="transmission",
        printed_baseline=printed_baseline,
        effective_baseline=effective_baseline,
        corrections=corrections,
    )


def render_brake_inputs_simple(
    state: dict,
    card_payload: dict,
    *,
    printed_baseline: dict,
    effective_baseline: dict,
    corrections: dict,
) -> None:
    _render_component_inputs_simple(
        state,
        domain="brake",
        printed_baseline=printed_baseline,
        effective_baseline=effective_baseline,
        corrections=corrections,
    )


def render_axle_hubs_inputs_simple(
    state: dict,
    card_payload: dict,
    *,
    printed_baseline: dict,
    effective_baseline: dict,
    corrections: dict,
) -> None:
    _render_component_inputs_simple(
        state,
        domain="axle_hubs",
        printed_baseline=printed_baseline,
        effective_baseline=effective_baseline,
        corrections=corrections,
    )


def render_parasitics_inputs_simple(
    state: dict,
    card_payload: dict,
    *,
    printed_baseline: dict,
    effective_baseline: dict,
    corrections: dict,
) -> None:
    _render_component_inputs_simple(
        state,
        domain="parasitic",
        printed_baseline=printed_baseline,
        effective_baseline=effective_baseline,
        corrections=corrections,
    )


def _render_component_inputs_simple(
    state: dict,
    *,
    domain: str,
    printed_baseline: dict,
    effective_baseline: dict,
    corrections: dict,
) -> None:
    config = dict(COMPONENT_SIMPLE_FIELD_CONFIG[domain])
    printed_display = _baseline_domain_display(domain, printed_baseline)
    effective_display, proposal_contexts = _domain_contexts(domain, state, effective_baseline)
    proposals = list(state.get("proposals") or [])
    correction_fields = tuple(config.get("corrections") or ())
    correction_values = {field_key: corrections.get(field_key) for field_key in correction_fields}
    values_by_proposal: dict[str, dict] = {}
    debug_widget_keys = None
    proposal_specs = []

    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id") or "")
        context = dict(proposal_contexts.get(proposal_id) or {})
        proposal_type = str(context.get("proposal_type") or "INHERIT")
        selection_mode = str(context.get("selection_mode") or proposal_type)
        editable_inputs = deepcopy(dict(context.get("inputs") or {}))
        editable_inputs.update(
            _current_widget_inputs(
                domain,
                proposal_id,
                proposal_type,
                selection_mode,
                editable_inputs,
            )
        )
        if canonical_component_mode(domain, proposal_type, selection_mode, editable_inputs) == "LOOKUP":
            editable_inputs.update(_component_lookup_draft(domain, proposal_id))
        values_by_proposal[proposal_id] = editable_inputs
        proposal_specs.append(
            {
                "proposal": proposal,
                "proposal_id": proposal_id,
                "context": context,
                "editable_inputs": editable_inputs,
                "proposal_type": proposal_type,
                "selection_mode": selection_mode,
                "component_mode": canonical_component_mode(
                    domain,
                    proposal_type,
                    selection_mode,
                    editable_inputs,
                ),
            }
        )

    with st.form(str(config.get("form_key") or f"v22_{domain}_inputs_form")):
        render_v22_reference_divider()
        _render_simple_sheet_header(state, domain, proposal_specs)
        for row_key in _component_simple_sheet_rows(domain, proposal_specs):
            _render_component_simple_sheet_row(
                domain,
                row_key,
                proposal_specs,
                printed_display,
                effective_display,
                correction_values,
                debug_widget_keys=debug_widget_keys,
            )
        submitted = st.form_submit_button(str(config.get("apply_label") or f"Apply {DOMAIN_LABELS[domain]}"))

    if domain == "transmission":
        _render_transmission_coastdown_audit(proposal_specs)

    if not submitted:
        return

    merged_corrections = _merge_simple_corrections(corrections, correction_fields, correction_values)
    payload = build_v22_domain_apply_payload(domain, proposals, values_by_proposal)
    next_state = apply_v22_corrections(state, merged_corrections)
    next_state["baseline"]["correction_disposition"] = str(st.session_state.get("v22_correction_disposition", "request_only") or "request_only")
    next_state = apply_v22_domain_inputs(next_state, domain, payload)
    st.session_state[V22_SESSION_KEY] = next_state


def _tire_simple_sheet_rows(proposal_specs: list[dict], correction_field_keys: tuple[str, ...]) -> list[str]:
    active_types = {
        canonical_tire_proposal_type(str(spec.get("proposal_type") or "INHERIT"))
        for spec in proposal_specs
        if str(spec.get("proposal_type") or "INHERIT") not in {"", "INHERIT"}
    }
    rows = ["tire_db_id", "tire_source_vde_id", "tire_code"]
    if "TIRE_TARGET_RRC" in active_types:
        rows.append("target_rrc_N_per_kN")
    if "TIRE_IMPROVEMENT_PCT" in active_types:
        rows.append("tire_improvement_pct")
    rows.extend(["rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi", "tire_load_mass_basis", "tire_review_status"])
    for field_key in correction_field_keys:
        if field_key not in rows:
            rows.append(field_key)
    return rows


def _render_tire_simple_sheet_row(
    row_key: str,
    proposal_specs: list[dict],
    printed_display: dict,
    effective_display: dict,
    correction_values: dict,
    *,
    debug_widget_keys: list[str] | None = None,
) -> None:
    correction_field_keys = set(correction_values.keys())
    printed_value = printed_display.get(row_key)
    effective_value = effective_display.get(row_key)

    if row_key in correction_field_keys:

        def correction_renderer(cell):
            correction_values[row_key] = _render_correction_widget("tire", row_key, correction_values.get(row_key), cell)
            if debug_widget_keys is not None:
                debug_widget_keys.append(f"v22_correction__tire__{row_key}")

    else:
        correction_renderer = _render_em_dash

    proposal_renderers = [
        (lambda cell, spec=spec: _render_tire_simple_sheet_cell(cell, row_key, spec, debug_widget_keys=debug_widget_keys))
        for spec in proposal_specs
    ]
    _render_simple_sheet_row(
        label=field_meta(row_key).get("label") or row_key,
        field_key=row_key,
        proposal_specs=proposal_specs,
        printed_value=printed_value,
        effective_value=effective_value,
        correction_renderer=correction_renderer,
        proposal_renderers=proposal_renderers,
    )


def _render_tire_simple_sheet_cell(cell, row_key: str, spec: dict, *, debug_widget_keys: list[str] | None = None) -> None:
    proposal_type = canonical_tire_proposal_type(str(spec.get("proposal_type") or "INHERIT"))
    selection_mode = str(spec.get("selection_mode") or proposal_type)
    proposal_id = str(spec.get("proposal_id") or "")
    context = dict(spec.get("context") or {})
    editable_inputs = spec.get("editable_inputs")
    if editable_inputs is None:
        editable_inputs = {}
    source_display = dict(context.get("source_display") or {})
    resolved_display = dict(context.get("resolved_display") or {})

    if proposal_type == "INHERIT":
        if row_key in {"tire_db_id", "tire_source_vde_id", "tire_code", "rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi", "tire_load_mass_basis", "tire_review_status"}:
            cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        else:
            cell.write(EM_DASH)
        return
    if proposal_is_not_used(proposal_type, selection_mode, domain="tire"):
        cell.write("Not used")
        return

    if row_key in {"tire_db_id", "tire_source_vde_id", "tire_code"}:
        if proposal_type == "TIRE_DB_LOOKUP":
            value = editable_inputs.get(row_key)
            if is_blank(value):
                value = resolved_display.get(row_key)
            if is_blank(value):
                value = source_display.get(row_key)
            cell.write(_display_domain_cell(value, row_key))
        else:
            cell.write(_display_domain_cell(resolved_display.get(row_key) or source_display.get(row_key), row_key))
        return
    if row_key == "target_rrc_N_per_kN":
        if proposal_type == "TIRE_TARGET_RRC":
            editable_inputs[row_key] = _render_simple_number_input("tire", proposal_id, row_key, editable_inputs.get(row_key), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key == "tire_improvement_pct":
        if proposal_type == "TIRE_IMPROVEMENT_PCT":
            editable_inputs[row_key] = _render_simple_number_input("tire", proposal_id, row_key, editable_inputs.get(row_key), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key == "rrc_N_per_kN":
        if proposal_type == "TIRE_DB_LOOKUP":
            cell.write(_display_domain_cell(editable_inputs.get(row_key), row_key))
        else:
            cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        return
    if row_key in {"front_pressure_psi", "rear_pressure_psi"}:
        if proposal_type in {"TIRE_DB_LOOKUP", "TIRE_TARGET_RRC", "TIRE_IMPROVEMENT_PCT"}:
            editable_inputs[row_key] = _render_simple_number_input("tire", proposal_id, row_key, editable_inputs.get(row_key), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        return
    if row_key == "tire_load_mass_basis":
        # Tire load mass is resolved by the Mass proposal.  It is selectable
        # only where a TWC-based mass proposal explicitly permits that choice.
        cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        return
    if row_key == "tire_review_status":
        cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        return

    cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))


def _component_simple_sheet_rows(domain: str, proposal_specs: list[dict]) -> list[str]:
    config = dict(COMPONENT_SIMPLE_FIELD_CONFIG[domain])
    rows = [str(config.get("lookup_id") or ""), str(config.get("vde_id") or ""), *(config.get("abc") or ())]
    component_modes = {
        str(spec.get("component_mode") or "INHERIT")
        for spec in proposal_specs
        if str(spec.get("proposal_type") or "INHERIT") not in {"", "INHERIT"}
    }
    proposal_types = {
        str(spec.get("proposal_type") or "INHERIT")
        for spec in proposal_specs
        if str(spec.get("proposal_type") or "INHERIT") not in {"", "INHERIT"}
    }
    if "DELTA_ABC" in component_modes:
        rows.extend(["delta_A", "delta_B", "delta_C"])
    if domain == "transmission" and proposal_types - {"TRANS_LOSS_PCT"}:
        rows.insert(2, "transmission_application_mode")
    if domain == "transmission" and "TRANS_LOSS_PCT" in proposal_types:
        rows.append("transmission_loss_pct")
    if domain == "brake" and "RESIDUAL_TORQUE" in component_modes:
        rows.extend(["residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m", "brake_drag_force_N"])
    return [row_key for row_key in rows if row_key]


def _render_component_simple_sheet_row(
    domain: str,
    row_key: str,
    proposal_specs: list[dict],
    printed_display: dict,
    effective_display: dict,
    correction_values: dict,
    *,
    debug_widget_keys: list[str] | None = None,
) -> None:
    config = dict(COMPONENT_SIMPLE_FIELD_CONFIG[domain])
    correction_field_keys = set(config.get("corrections") or ())
    printed_value = printed_display.get(row_key)
    effective_value = effective_display.get(row_key)

    if row_key in correction_field_keys:

        def correction_renderer(cell):
            correction_values[row_key] = _render_correction_widget(domain, row_key, correction_values.get(row_key), cell)
            if debug_widget_keys is not None:
                debug_widget_keys.append(f"v22_correction__{domain}__{row_key}")

    else:
        correction_renderer = _render_em_dash

    proposal_renderers = [
        (lambda cell, spec=spec: _render_component_simple_sheet_cell(domain, cell, row_key, spec, debug_widget_keys=debug_widget_keys))
        for spec in proposal_specs
    ]
    _render_simple_sheet_row(
        label=field_meta(row_key).get("label") or row_key,
        field_key=row_key,
        proposal_specs=proposal_specs,
        printed_value=printed_value,
        effective_value=effective_value,
        correction_renderer=correction_renderer,
        proposal_renderers=proposal_renderers,
    )


def _render_component_simple_sheet_cell(
    domain: str,
    cell,
    row_key: str,
    spec: dict,
    *,
    debug_widget_keys: list[str] | None = None,
) -> None:
    config = dict(COMPONENT_SIMPLE_FIELD_CONFIG[domain])
    proposal_type = str(spec.get("proposal_type") or "INHERIT")
    selection_mode = str(spec.get("selection_mode") or proposal_type)
    component_mode = str(spec.get("component_mode") or canonical_component_mode(domain, proposal_type, selection_mode))
    proposal_id = str(spec.get("proposal_id") or "")
    context = dict(spec.get("context") or {})
    editable_inputs = spec.get("editable_inputs")
    if editable_inputs is None:
        editable_inputs = {}
    source_display = dict(context.get("source_display") or {})
    resolved_display = dict(context.get("resolved_display") or {})
    lookup_id_field = str(config.get("lookup_id") or "")
    vde_id_field = str(config.get("vde_id") or "")
    abc_fields = tuple(config.get("abc") or ())

    if proposal_type == "INHERIT":
        if row_key == "transmission_application_mode":
            cell.write(_display_domain_cell(resolved_display.get(row_key) or source_display.get(row_key), row_key))
            return
        if row_key in abc_fields:
            cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        else:
            cell.write(EM_DASH)
        return
    if proposal_is_not_used(proposal_type, selection_mode, domain=domain):
        cell.write("Not used")
        return

    if row_key == lookup_id_field:
        if component_mode == "LOOKUP":
            cell.write(_display_domain_cell(editable_inputs.get(row_key), row_key))
        else:
            cell.write(_display_domain_cell(resolved_display.get(row_key) or source_display.get(row_key), row_key))
        return

    if row_key == vde_id_field:
        cell.write(_display_domain_cell(resolved_display.get(row_key) or source_display.get(row_key), row_key))
        return

    if row_key == "transmission_application_mode":
        if proposal_type == "TRANS_LOSS_PCT":
            cell.write("Fixed measured TOTAL; recalculate NET")
            return
        editable_inputs[row_key] = _render_simple_select_input(
            _simple_widget_domain_scope(domain),
            proposal_id,
            row_key,
            editable_inputs.get(row_key) or resolved_display.get(row_key) or source_display.get(row_key),
            container=cell,
            debug_widget_keys=debug_widget_keys,
        )
        return

    if row_key in abc_fields:
        if component_mode == "LOOKUP":
            cell.write(_display_domain_cell(editable_inputs.get(row_key), row_key))
        elif component_mode == "ABSOLUTE_ABC":
            editable_inputs[row_key] = _render_simple_number_input(
                _simple_widget_domain_scope(domain),
                proposal_id,
                row_key,
                editable_inputs.get(row_key),
                container=cell,
                debug_widget_keys=debug_widget_keys,
            )
        else:
            cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        return

    if row_key in {"delta_A", "delta_B", "delta_C"}:
        if component_mode == "DELTA_ABC":
            editable_inputs[row_key] = _render_simple_number_input(
                _simple_widget_domain_scope(domain),
                proposal_id,
                row_key,
                editable_inputs.get(row_key),
                container=cell,
                debug_widget_keys=debug_widget_keys,
            )
        else:
            cell.write(EM_DASH)
        return

    if row_key == "transmission_loss_pct":
        if proposal_type == "TRANS_LOSS_PCT":
            editable_inputs[row_key] = _render_simple_number_input(
                _simple_widget_domain_scope(domain),
                proposal_id,
                row_key,
                editable_inputs.get(row_key),
                container=cell,
                debug_widget_keys=debug_widget_keys,
            )
            applied_inputs = dict(context.get("inputs") or {})
            if str(applied_inputs.get("rule_version") or "").strip().upper() != "COASTDOWN_SHARE_V1":
                cell.caption("Legacy value applied. Apply Transmission to use Walk From ABC_TOTAL.")
            else:
                cell.caption("Applied against Walk From ABC_TOTAL; TOTAL remains fixed.")
        else:
            cell.write(EM_DASH)
        return

    if row_key in {"percent_basis", "rule_version"}:
        if proposal_type == "TRANS_LOSS_PCT":
            editable_inputs[row_key] = _render_simple_text_input(
                _simple_widget_domain_scope(domain),
                proposal_id,
                row_key,
                editable_inputs.get(row_key),
                container=cell,
                debug_widget_keys=debug_widget_keys,
            )
        else:
            cell.write(EM_DASH)
        return

    if row_key in {"residual_torque_front_Nm", "residual_torque_rear_Nm", "residual_torque_total_Nm", "wheel_radius_m"}:
        if component_mode == "RESIDUAL_TORQUE":
            editable_inputs[row_key] = _render_simple_number_input(
                _simple_widget_domain_scope(domain),
                proposal_id,
                row_key,
                editable_inputs.get(row_key),
                container=cell,
                debug_widget_keys=debug_widget_keys,
            )
        else:
            cell.write(EM_DASH)
        return

    if row_key == "brake_drag_force_N":
        cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        return

    cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))


def _render_simple_reference_section(
    domain: str,
    field_keys: tuple[str, ...],
    printed_display: dict,
    effective_display: dict,
    correction_values: dict,
    *,
    debug_widget_keys: list[str] | None = None,
) -> None:
    header = st.columns([1.35, 0.65, 0.9, 0.95, 0.95])
    header[0].caption("Field")
    header[1].caption("Unit")
    header[2].caption("Printed")
    header[3].caption(_baseline_input_lane_label(st.session_state.get(V22_SESSION_KEY)))
    header[4].caption("Effective")
    for field_key in field_keys:
        meta = field_meta(field_key)
        row = st.columns([1.35, 0.65, 0.9, 0.95, 0.95])
        row[0].markdown(meta.get("label") or field_key)
        row[1].caption(display_unit_for_field(field_key, _current_unit_system(), meta.get("unit"), pressure_unit=_current_tire_pressure_unit(unit_system=_current_unit_system())) or "-")
        row[2].write(_display_domain_cell(printed_display.get(field_key), field_key))
        correction_values[field_key] = _render_correction_widget(domain, field_key, correction_values.get(field_key), row[3])
        row[4].write(_display_domain_cell(effective_display.get(field_key), field_key))
        if debug_widget_keys is not None:
            debug_widget_keys.append(f"v22_correction__{domain}__{field_key}")


def _simple_sheet_column_widths(proposal_count: int) -> list[float]:
    return [1.45, 0.65, 0.9, 0.95, 0.95, *([1.08] * max(proposal_count, 1))]


def _render_simple_sheet_header(state: dict, domain: str, proposal_specs: list[dict]) -> None:
    header = st.columns(_simple_sheet_column_widths(len(proposal_specs)))
    header[0].caption("Field")
    header[1].caption("Unit")
    header[2].caption("Printed")
    header[3].caption(_baseline_input_lane_label(state))
    header[4].caption("Effective")
    proposal_statuses = dict(dict(dict(state.get("domain_input_state") or {}).get(domain) or {}).get("proposal_statuses") or {})
    for index, spec in enumerate(proposal_specs, start=5):
        proposal = dict(spec.get("proposal") or {})
        proposal_id = str(spec.get("proposal_id") or "")
        status_payload = dict(proposal_statuses.get(proposal_id) or {})
        header[index].markdown(
            "\n".join(
                [
                    f"**{proposal_display_label(state, proposal)}**",
                    f"From: {walk_from_display_label(state, proposal.get('walk_from') or 'baseline')}",
                    str(spec.get("selection_mode") or spec.get("proposal_type") or "Inherit"),
                    proposal_status_label(status_payload),
                ]
            )
        )


def _render_simple_sheet_row(
    *,
    label: str,
    field_key: str | None,
    proposal_specs: list[dict],
    printed_value,
    effective_value,
    correction_renderer,
    proposal_renderers: list,
) -> None:
    row = st.columns(_simple_sheet_column_widths(len(proposal_specs)))
    row[0].markdown(label)
    if field_key:
        unit = display_unit_for_field(field_key, _current_unit_system(), field_meta(field_key).get("unit"), pressure_unit=_current_tire_pressure_unit(unit_system=_current_unit_system())) or "-"
    else:
        unit = "-"
    row[1].caption(unit)
    row[2].write(_display_domain_cell(printed_value, field_key) if field_key else _display_editor_value(printed_value))
    correction_renderer(row[3])
    row[4].write(_display_domain_cell(effective_value, field_key) if field_key else _display_editor_value(effective_value))
    for offset, render_cell in enumerate(proposal_renderers, start=5):
        render_cell(row[offset])


def _render_em_dash(cell) -> None:
    cell.write(EM_DASH)


def _mass_simple_sheet_rows(proposal_specs: list[dict]) -> list[str]:
    active_types = {
        str(spec.get("proposal_type") or "INHERIT")
        for spec in proposal_specs
        if str(spec.get("proposal_type") or "INHERIT") not in {"", "INHERIT"}
    }
    rows = [
        "current_curb_mass_kg",
        "mass_kg",
        "weight_dist_fr_pct",
    ]
    if "CUSTOM_MASS" in active_types:
        rows.append("test_mass_kg")
    if active_types & {"EPA_CURB_TO_TWC", "MASS_TWC_SHIFT", "CUSTOM_MASS"}:
        rows.append("test_mass_basis")
    if active_types & {"EPA_CURB_TO_TWC", "MASS_TWC_SHIFT"}:
        rows.append("tire_load_mass_basis")
    if "MASS_TWC_SHIFT" in active_types:
        rows.extend(["shift_steps", "target_mass_kg", "curb_position"])
    if "PERFORMANCE_CURB_MASS" in active_types:
        rows.extend(["preset", "custom_delta_kg"])
    if "WLTP_MASS_LINE" in active_types:
        rows.extend(["line_type", "payload_kg", "options_kg", "test_mass_low_kg", "test_mass_high_kg"])
    if "GVWR" in active_types:
        rows.extend(["payload_kg", "gvwr_kg"])
    if "GCWR" in active_types:
        rows.extend(["gcwr_kg", "trailer_mass_kg", "trailer_A", "trailer_B", "trailer_C"])
    return rows


def _render_mass_resolved_audit(proposal_specs: list[dict]) -> None:
    fields = (
        ("EPA ETW / TWC", "inertia_class"),
        ("VDE mass", "vde_calculation_mass_kg"),
        ("VDE mass basis", "vde_mass_basis"),
        ("Tire calculation mass", "tire_load_mass_used_kg"),
        ("Tire calculation mass basis", "tire_load_mass_basis"),
        ("Status", "mass_rule_status"),
        ("Notes", "mass_rule_notes"),
    )
    with st.expander("Resolved mass audit", expanded=True):
        rows = []
        for label, field_key in fields:
            row = {"Field": label}
            for spec in proposal_specs:
                context = dict(spec.get("context") or {})
                proposal = dict(spec.get("proposal") or {})
                proposal_label = f"Requested #{proposal.get('display_index') or spec.get('proposal_id')}"
                value = dict(context.get("resolved_display") or {}).get(field_key)
                row[proposal_label] = _display_domain_cell(value, field_key)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_tire_resolved_audit(proposal_specs: list[dict]) -> None:
    fields = (
        ("Source RRC", "tire_source_rrc_N_per_kN"),
        ("Resolved RRC", "rrc_N_per_kN"),
        ("Adjusted RRC", "tire_adjusted_rrc_N_per_kN"),
        ("Delta RRC", "tire_delta_rrc_N_per_kN"),
        ("Reference front pressure", "tire_reference_front_pressure_psi"),
        ("Reference rear pressure", "tire_reference_rear_pressure_psi"),
        ("Tire calculation mass", "tire_load_mass_used_kg"),
        ("Tire calculation mass basis", "tire_load_mass_basis"),
        ("Tire ABC A", "tire_A_final"),
        ("Tire ABC B", "tire_B_final"),
        ("Tire ABC C", "tire_C_final"),
        ("Status", "tire_review_status"),
        ("Notes", "tire_rule_notes"),
    )
    with st.expander("Resolved tire audit", expanded=False):
        rows = []
        for label, field_key in fields:
            row = {"Field": label}
            for spec in proposal_specs:
                context = dict(spec.get("context") or {})
                proposal = dict(spec.get("proposal") or {})
                proposal_label = f"Requested #{proposal.get('display_index') or spec.get('proposal_id')}"
                value = dict(context.get("resolved_display") or {}).get(field_key)
                row[proposal_label] = _display_domain_cell(value, field_key)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_transmission_coastdown_audit(proposal_specs: list[dict]) -> None:
    if not any(str(spec.get("proposal_type") or "") == "TRANS_LOSS_PCT" for spec in proposal_specs):
        return
    fields = (
        ("Applied coastdown share", "transmission_loss_pct"),
        ("Calculation basis", "transmission_percent_basis"),
        ("Rule version", "transmission_rule_version"),
        ("Walk From ABC_TOTAL A", "source_abc_total_A"),
        ("Walk From ABC_TOTAL B", "source_abc_total_B"),
        ("Walk From ABC_TOTAL C", "source_abc_total_C"),
        ("Resolved transmission A", "trans_A_coef_N"),
        ("Resolved transmission B", "trans_B_coef_Npkph"),
        ("Resolved transmission C", "trans_C_coef_Npkph2"),
    )
    with st.expander("Transmission coastdown share audit", expanded=False):
        rows = []
        for label, field_key in fields:
            row = {"Field": label}
            for spec in proposal_specs:
                proposal = dict(spec.get("proposal") or {})
                proposal_label = f"Requested #{proposal.get('display_index') or spec.get('proposal_id')}"
                context = dict(spec.get("context") or {})
                inputs = dict(context.get("inputs") or {})
                resolved = dict(context.get("resolved_display") or {})
                if field_key == "transmission_percent_basis":
                    value = inputs.get("percent_basis")
                elif field_key == "transmission_rule_version":
                    value = inputs.get("rule_version")
                else:
                    value = resolved.get(field_key)
                row[proposal_label] = _display_domain_cell(value, field_key)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_mass_simple_sheet_row(
    state: dict,
    row_key: str,
    proposal_specs: list[dict],
    printed_display: dict,
    effective_display: dict,
    correction_values: dict,
    *,
    debug_widget_keys: list[str] | None = None,
) -> None:
    label_overrides = {
        "current_curb_mass_kg": field_meta("current_curb_mass_kg").get("label") or "Current / inherited curb mass",
    }
    display_field = "mass_kg" if row_key == "current_curb_mass_kg" else row_key
    correction_field_keys = set(MASS_SIMPLE_CORRECTION_FIELDS)
    if row_key == "current_curb_mass_kg":
        printed_value = printed_display.get("mass_kg")
        effective_value = effective_display.get("mass_kg")
    else:
        printed_value = printed_display.get(display_field)
        effective_value = effective_display.get(display_field)

    if row_key in correction_field_keys:
        def correction_renderer(cell):
            correction_values[row_key] = _render_correction_widget("mass", row_key, correction_values.get(row_key), cell)
            if debug_widget_keys is not None:
                debug_widget_keys.append(f"v22_correction__mass__{row_key}")
    else:
        correction_renderer = _render_em_dash

    proposal_renderers = [
        (lambda cell, spec=spec: _render_mass_simple_sheet_cell(cell, row_key, spec, debug_widget_keys=debug_widget_keys))
        for spec in proposal_specs
    ]
    _render_simple_sheet_row(
        label=label_overrides.get(row_key, field_meta(row_key).get("label") or row_key),
        field_key=display_field,
        proposal_specs=proposal_specs,
        printed_value=printed_value,
        effective_value=effective_value,
        correction_renderer=correction_renderer,
        proposal_renderers=proposal_renderers,
    )


def _render_mass_simple_sheet_cell(cell, row_key: str, spec: dict, *, debug_widget_keys: list[str] | None = None) -> None:
    proposal_type = str(spec.get("proposal_type") or "INHERIT")
    selection_mode = str(spec.get("selection_mode") or proposal_type)
    proposal_id = str(spec.get("proposal_id") or "")
    context = dict(spec.get("context") or {})
    editable_inputs = spec.get("editable_inputs")
    if editable_inputs is None:
        editable_inputs = {}
    source_display = dict(context.get("source_display") or {})
    resolved_display = dict(context.get("resolved_display") or {})

    if proposal_type == "INHERIT":
        if row_key in {"current_curb_mass_kg", "mass_kg", "inertia_class", "test_mass_kg", "vde_calculation_mass_kg", "vde_mass_basis", "tire_load_mass_used_kg", "test_mass_basis", "weight_dist_fr_pct", "tire_load_mass_basis"}:
            cell.write(_display_domain_cell(resolved_display.get("mass_kg" if row_key == "current_curb_mass_kg" else row_key), "mass_kg" if row_key == "current_curb_mass_kg" else row_key))
        else:
            cell.write(EM_DASH)
        return
    if proposal_is_not_used(proposal_type, selection_mode, domain="mass"):
        cell.write("Not used")
        return

    if row_key == "current_curb_mass_kg":
        cell.write(_display_domain_cell(source_display.get("mass_kg"), "mass_kg"))
        return
    if row_key == "mass_kg":
        if proposal_type in {"EPA_CURB_TO_TWC", "PERFORMANCE_CURB_MASS", "WLTP_MASS_LINE", "GVWR", "GCWR"}:
            editable_inputs["mass_kg"] = _render_simple_number_input("mass", proposal_id, "mass_kg", editable_inputs.get("mass_kg"), container=cell, debug_widget_keys=debug_widget_keys)
            return
        cell.write(_display_domain_cell(resolved_display.get("mass_kg"), "mass_kg"))
        return
    if row_key == "inertia_class":
        cell.write(_display_domain_cell(resolved_display.get("inertia_class"), "inertia_class"))
        return
    if row_key in {"vde_calculation_mass_kg", "vde_mass_basis", "tire_load_mass_used_kg"}:
        cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        return
    if row_key == "test_mass_kg":
        if proposal_type == "CUSTOM_MASS":
            editable_inputs["test_mass_kg"] = _render_simple_number_input("mass", proposal_id, "test_mass_kg", editable_inputs.get("test_mass_kg"), container=cell, debug_widget_keys=debug_widget_keys)
            return
        cell.write(_display_domain_cell(resolved_display.get("test_mass_kg"), "test_mass_kg"))
        return
    if row_key == "weight_dist_fr_pct":
        editable_inputs["weight_dist_fr_pct"] = _render_simple_number_input("mass", proposal_id, "weight_dist_fr_pct", editable_inputs.get("weight_dist_fr_pct"), container=cell, debug_widget_keys=debug_widget_keys)
        return
    if row_key == "tire_load_mass_basis":
        if proposal_type in {"EPA_CURB_TO_TWC", "MASS_TWC_SHIFT"}:
            editable_inputs["tire_load_mass_basis"] = _render_simple_select_input("mass", proposal_id, "tire_load_mass_basis", editable_inputs.get("tire_load_mass_basis"), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(_display_domain_cell(resolved_display.get("tire_load_mass_basis"), "tire_load_mass_basis"))
        return
    if row_key == "test_mass_basis":
        if proposal_type == "CUSTOM_MASS":
            editable_inputs["test_mass_basis"] = _render_simple_text_input("mass", proposal_id, "test_mass_basis", editable_inputs.get("test_mass_basis"), container=cell, debug_widget_keys=debug_widget_keys)
            return
        cell.write(_display_domain_cell(resolved_display.get("test_mass_basis"), "test_mass_basis"))
        return
    if row_key == "shift_steps":
        if proposal_type == "MASS_TWC_SHIFT":
            editable_inputs["shift_steps"] = _render_simple_select_input("mass", proposal_id, "shift_steps", editable_inputs.get("shift_steps"), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key == "target_mass_kg":
        if proposal_type == "MASS_TWC_SHIFT" and str(editable_inputs.get("shift_steps") or "") == "Select target":
            editable_inputs["target_mass_kg"] = _render_simple_select_input("mass", proposal_id, "target_mass_kg", editable_inputs.get("target_mass_kg"), container=cell, debug_widget_keys=debug_widget_keys)
        elif proposal_type == "MASS_TWC_SHIFT":
            cell.write(_display_domain_cell(resolved_display.get("target_mass_kg"), "target_mass_kg"))
        else:
            cell.write(EM_DASH)
        return
    if row_key == "curb_position":
        if proposal_type == "MASS_TWC_SHIFT":
            editable_inputs["curb_position"] = _render_simple_select_input("mass", proposal_id, "curb_position", editable_inputs.get("curb_position"), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key == "preset":
        if proposal_type == "PERFORMANCE_CURB_MASS":
            editable_inputs["preset"] = _render_simple_select_input("mass", proposal_id, "preset", editable_inputs.get("preset"), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key == "custom_delta_kg":
        if proposal_type == "PERFORMANCE_CURB_MASS" and str(editable_inputs.get("preset") or "") == "Custom delta":
            editable_inputs["custom_delta_kg"] = _render_simple_number_input("mass", proposal_id, "custom_delta_kg", editable_inputs.get("custom_delta_kg"), container=cell, debug_widget_keys=debug_widget_keys)
        elif proposal_type == "PERFORMANCE_CURB_MASS":
            cell.write(_display_domain_cell(resolved_display.get("custom_delta_kg"), "custom_delta_kg"))
        else:
            cell.write(EM_DASH)
        return
    if row_key == "line_type":
        if proposal_type == "WLTP_MASS_LINE":
            editable_inputs["line_type"] = _render_simple_select_input("mass", proposal_id, "line_type", editable_inputs.get("line_type"), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key in {"payload_kg", "options_kg"}:
        if proposal_type == "WLTP_MASS_LINE" or (proposal_type == "GVWR" and row_key == "payload_kg"):
            editable_inputs[row_key] = _render_simple_number_input("mass", proposal_id, row_key, editable_inputs.get(row_key), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key in {"test_mass_low_kg", "test_mass_high_kg"}:
        cell.write(_display_domain_cell(resolved_display.get(row_key), row_key) if proposal_type == "WLTP_MASS_LINE" else EM_DASH)
        return
    if row_key == "gvwr_kg":
        if proposal_type == "GVWR":
            cell.write(_display_domain_cell(resolved_display.get("gvwr_kg"), "gvwr_kg"))
        else:
            cell.write(EM_DASH)
        return
    if row_key == "gcwr_kg":
        if proposal_type == "GCWR":
            editable_inputs["gcwr_kg"] = _render_simple_number_input("mass", proposal_id, "gcwr_kg", editable_inputs.get("gcwr_kg"), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key == "trailer_mass_kg":
        if proposal_type == "GCWR":
            editable_inputs["trailer_mass_kg"] = _render_simple_number_input("mass", proposal_id, "trailer_mass_kg", editable_inputs.get("trailer_mass_kg"), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key in {"trailer_A", "trailer_B", "trailer_C"}:
        if proposal_type == "GCWR":
            editable_inputs[row_key] = _render_simple_number_input("mass", proposal_id, row_key, editable_inputs.get(row_key), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(EM_DASH)
        return
    if row_key in {"mass_rule_status", "mass_rule_notes"}:
        cell.write(_display_domain_cell(resolved_display.get(row_key), row_key))
        return

    cell.write(EM_DASH)


def _render_aero_simple_sheet_row(
    row_key: str,
    proposal_specs: list[dict],
    printed_display: dict,
    effective_display: dict,
    correction_values: dict,
    *,
    debug_widget_keys: list[str] | None = None,
) -> None:
    if row_key == "cda_m2":
        def correction_renderer(cell):
            correction_values["cda_m2"] = _render_correction_widget("aero", "cda_m2", correction_values.get("cda_m2"), cell)
            if debug_widget_keys is not None:
                debug_widget_keys.append("v22_correction__aero__cda_m2")
    else:
        correction_renderer = _render_em_dash

    proposal_renderers = [
        (lambda cell, spec=spec: _render_aero_simple_sheet_cell(cell, row_key, spec, debug_widget_keys=debug_widget_keys))
        for spec in proposal_specs
    ]
    _render_simple_sheet_row(
        label=field_meta(row_key).get("label") or row_key,
        field_key=row_key,
        proposal_specs=proposal_specs,
        printed_value=printed_display.get(row_key) if row_key == "cda_m2" else None,
        effective_value=effective_display.get(row_key) if row_key == "cda_m2" else None,
        correction_renderer=correction_renderer,
        proposal_renderers=proposal_renderers,
    )


def _render_aero_simple_sheet_cell(cell, row_key: str, spec: dict, *, debug_widget_keys: list[str] | None = None) -> None:
    proposal_type = str(spec.get("proposal_type") or "INHERIT")
    selection_mode = str(spec.get("selection_mode") or proposal_type)
    proposal_id = str(spec.get("proposal_id") or "")
    context = dict(spec.get("context") or {})
    editable_inputs = spec.get("editable_inputs")
    if editable_inputs is None:
        editable_inputs = {}
    resolved_display = dict(context.get("resolved_display") or {})

    if proposal_type == "INHERIT":
        if row_key == "cda_m2":
            cell.write(_display_domain_cell(resolved_display.get("cda_m2"), "cda_m2"))
        else:
            cell.write(_display_domain_cell(resolved_display.get("delta_CdA"), "delta_CdA"))
        return
    if proposal_is_not_used(proposal_type, selection_mode, domain="aero"):
        cell.write("Not used" if row_key == "cda_m2" else EM_DASH)
        return

    if row_key == "cda_m2":
        if proposal_type == "AERO_ABSOLUTE_CDA":
            editable_inputs["cda_m2"] = _render_simple_number_input("aero", proposal_id, "cda_m2", editable_inputs.get("cda_m2"), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(_display_domain_cell(resolved_display.get("cda_m2"), "cda_m2"))
        return
    if row_key == "delta_CdA":
        if proposal_type == "AERO_DELTA_CDA":
            editable_inputs["delta_CdA"] = _render_simple_number_input("aero", proposal_id, "delta_CdA", editable_inputs.get("delta_CdA"), container=cell, debug_widget_keys=debug_widget_keys)
        else:
            cell.write(_display_domain_cell(resolved_display.get("delta_CdA"), "delta_CdA"))
        return


def _render_simple_proposal_header(state: dict, proposal: dict, domain: str, context: dict) -> None:
    proposal_id = str(proposal.get("proposal_id") or "")
    selection_mode = str(context.get("selection_mode") or context.get("proposal_type") or "Inherit")
    status_payload = dict(dict(dict(state.get("domain_input_state") or {}).get(domain) or {}).get("proposal_statuses") or {}).get(proposal_id) or {}
    st.markdown(f"**{proposal_display_label(state, proposal)}**")
    details = st.columns([1.3, 1.4, 1.1])
    details[0].caption(f"Walk From: {walk_from_display_label(state, proposal.get('walk_from') or 'baseline')}")
    details[1].caption(f"Proposal Type: {selection_mode}")
    details[2].caption(proposal_status_label(status_payload))


def _render_mass_simple_proposal_inputs(
    proposal_id: str,
    context: dict,
    editable_inputs: dict,
    *,
    debug_widget_keys: list[str] | None = None,
) -> None:
    proposal_type = str(context.get("proposal_type") or "INHERIT")
    selection_mode = str(context.get("selection_mode") or proposal_type)
    resolved_display = dict(context.get("resolved_display") or {})
    source_display = dict(context.get("source_display") or {})

    if proposal_type == "INHERIT":
        st.caption("This proposal inherits the resolved Mass state from Walk From.")
        return
    if proposal_is_not_used(proposal_type, selection_mode, domain="mass"):
        st.caption("Mass is marked as not used for this proposal.")
        return

    if proposal_type == "EPA_STATUS":
        st.caption("Legacy compatibility mode. It preserves the inherited EPA/TWC state.")
        editable_inputs["tire_load_mass_basis"] = _render_simple_select_input("mass", proposal_id, "tire_load_mass_basis", editable_inputs.get("tire_load_mass_basis"), debug_widget_keys=debug_widget_keys)
        return

    if proposal_type == "EPA_CURB_TO_TWC":
        cols = st.columns(2)
        cols[0].caption(f"Current / inherited curb mass: {_display_domain_cell(source_display.get('mass_kg'), 'mass_kg')}")
        editable_inputs["mass_kg"] = _render_simple_number_input("mass", proposal_id, "mass_kg", editable_inputs.get("mass_kg"), container=cols[1], debug_widget_keys=debug_widget_keys)
        editable_inputs["tire_load_mass_basis"] = _render_simple_select_input("mass", proposal_id, "tire_load_mass_basis", editable_inputs.get("tire_load_mass_basis"), debug_widget_keys=debug_widget_keys)
        return

    if proposal_type == "MASS_TWC_SHIFT":
        first_row = st.columns(3)
        editable_inputs["shift_steps"] = _render_simple_select_input("mass", proposal_id, "shift_steps", editable_inputs.get("shift_steps"), container=first_row[0], debug_widget_keys=debug_widget_keys)
        if str(editable_inputs.get("shift_steps") or "") == "Select target":
            editable_inputs["target_mass_kg"] = _render_simple_select_input("mass", proposal_id, "target_mass_kg", editable_inputs.get("target_mass_kg"), container=first_row[1], debug_widget_keys=debug_widget_keys)
        else:
            first_row[1].caption(f"Target ETW / TWC: {_display_domain_cell(resolved_display.get('target_mass_kg'), 'target_mass_kg')}")
        editable_inputs["curb_position"] = _render_simple_select_input("mass", proposal_id, "curb_position", editable_inputs.get("curb_position"), container=first_row[2], debug_widget_keys=debug_widget_keys)
        editable_inputs["tire_load_mass_basis"] = _render_simple_select_input("mass", proposal_id, "tire_load_mass_basis", editable_inputs.get("tire_load_mass_basis"), debug_widget_keys=debug_widget_keys)
        return

    if proposal_type == "PERFORMANCE_CURB_MASS":
        first_row = st.columns(2)
        editable_inputs["mass_kg"] = _render_simple_number_input("mass", proposal_id, "mass_kg", editable_inputs.get("mass_kg"), container=first_row[0], debug_widget_keys=debug_widget_keys)
        editable_inputs["preset"] = _render_simple_select_input("mass", proposal_id, "preset", editable_inputs.get("preset"), container=first_row[1], debug_widget_keys=debug_widget_keys)
        if str(editable_inputs.get("preset") or "") == "Custom delta":
            editable_inputs["custom_delta_kg"] = _render_simple_number_input("mass", proposal_id, "custom_delta_kg", editable_inputs.get("custom_delta_kg"), debug_widget_keys=debug_widget_keys)
        editable_inputs["tire_load_mass_basis"] = _render_simple_select_input("mass", proposal_id, "tire_load_mass_basis", editable_inputs.get("tire_load_mass_basis"), debug_widget_keys=debug_widget_keys)
        return

    if proposal_type == "WLTP_MASS_LINE":
        first_row = st.columns(2)
        editable_inputs["line_type"] = _render_simple_select_input("mass", proposal_id, "line_type", editable_inputs.get("line_type"), container=first_row[0], debug_widget_keys=debug_widget_keys)
        editable_inputs["mass_kg"] = _render_simple_number_input("mass", proposal_id, "mass_kg", editable_inputs.get("mass_kg"), container=first_row[1], debug_widget_keys=debug_widget_keys)
        second_row = st.columns(2)
        editable_inputs["payload_kg"] = _render_simple_number_input("mass", proposal_id, "payload_kg", editable_inputs.get("payload_kg"), container=second_row[0], debug_widget_keys=debug_widget_keys)
        editable_inputs["options_kg"] = _render_simple_number_input("mass", proposal_id, "options_kg", editable_inputs.get("options_kg"), container=second_row[1], debug_widget_keys=debug_widget_keys)
        editable_inputs["tire_load_mass_basis"] = _render_simple_select_input("mass", proposal_id, "tire_load_mass_basis", editable_inputs.get("tire_load_mass_basis"), debug_widget_keys=debug_widget_keys)
        return

    if proposal_type == "GVWR":
        editable_inputs["gvwr_kg"] = _render_simple_number_input("mass", proposal_id, "gvwr_kg", editable_inputs.get("gvwr_kg"), debug_widget_keys=debug_widget_keys)
        editable_inputs["tire_load_mass_basis"] = _render_simple_select_input("mass", proposal_id, "tire_load_mass_basis", editable_inputs.get("tire_load_mass_basis"), debug_widget_keys=debug_widget_keys)
        return

    if proposal_type == "GCWR":
        first_row = st.columns(2)
        editable_inputs["gcwr_kg"] = _render_simple_number_input("mass", proposal_id, "gcwr_kg", editable_inputs.get("gcwr_kg"), container=first_row[0], debug_widget_keys=debug_widget_keys)
        editable_inputs["trailer_mass_kg"] = _render_simple_number_input("mass", proposal_id, "trailer_mass_kg", editable_inputs.get("trailer_mass_kg"), container=first_row[1], debug_widget_keys=debug_widget_keys)
        second_row = st.columns(3)
        editable_inputs["trailer_A"] = _render_simple_number_input("mass", proposal_id, "trailer_A", editable_inputs.get("trailer_A"), container=second_row[0], debug_widget_keys=debug_widget_keys)
        editable_inputs["trailer_B"] = _render_simple_number_input("mass", proposal_id, "trailer_B", editable_inputs.get("trailer_B"), container=second_row[1], debug_widget_keys=debug_widget_keys)
        editable_inputs["trailer_C"] = _render_simple_number_input("mass", proposal_id, "trailer_C", editable_inputs.get("trailer_C"), container=second_row[2], debug_widget_keys=debug_widget_keys)
        editable_inputs["tire_load_mass_basis"] = _render_simple_select_input("mass", proposal_id, "tire_load_mass_basis", editable_inputs.get("tire_load_mass_basis"), debug_widget_keys=debug_widget_keys)
        return

    if proposal_type == "CUSTOM_MASS":
        first_row = st.columns(2)
        editable_inputs["test_mass_kg"] = _render_simple_number_input("mass", proposal_id, "test_mass_kg", editable_inputs.get("test_mass_kg"), container=first_row[0], debug_widget_keys=debug_widget_keys)
        editable_inputs["test_mass_basis"] = _render_simple_text_input("mass", proposal_id, "test_mass_basis", editable_inputs.get("test_mass_basis"), container=first_row[1], debug_widget_keys=debug_widget_keys)
        editable_inputs["tire_load_mass_basis"] = _render_simple_select_input("mass", proposal_id, "tire_load_mass_basis", editable_inputs.get("tire_load_mass_basis"), debug_widget_keys=debug_widget_keys)
        return

    st.caption(f"Mass proposal type `{proposal_type}` still falls back to its current applied values.")


def _render_mass_simple_resolved_summary(context: dict) -> None:
    resolved_display = dict(context.get("resolved_display") or {})
    status_row = st.columns(4)
    status_row[0].caption(f"Resolved TWC: {_display_domain_cell(resolved_display.get('inertia_class'), 'inertia_class')}")
    status_row[1].caption(f"Resolved test mass: {_display_domain_cell(resolved_display.get('test_mass_kg'), 'test_mass_kg')}")
    status_row[2].caption(f"Front weight distribution: {_display_domain_cell(resolved_display.get('weight_dist_fr_pct'), 'weight_dist_fr_pct')}")
    status_row[3].caption(f"Status: {_display_domain_cell(resolved_display.get('mass_rule_status'), 'mass_rule_status')}")
    notes = _display_domain_cell(resolved_display.get("mass_rule_notes"), "mass_rule_notes")
    if notes != EM_DASH:
        st.caption(f"Notes: {notes}")


def _render_aero_simple_proposal_inputs(
    proposal_id: str,
    context: dict,
    editable_inputs: dict,
    *,
    debug_widget_keys: list[str] | None = None,
) -> None:
    proposal_type = str(context.get("proposal_type") or "INHERIT")
    selection_mode = str(context.get("selection_mode") or proposal_type)
    source_display = dict(context.get("source_display") or {})

    if proposal_type == "INHERIT":
        st.caption("This proposal inherits the resolved Aero state from Walk From.")
        return
    if proposal_is_not_used(proposal_type, selection_mode, domain="aero"):
        st.caption("Aero is marked as not used for this proposal.")
        return

    st.caption(f"Baseline / inherited CdA: {_display_domain_cell(source_display.get('cda_m2'), 'cda_m2')}")
    if proposal_type == "AERO_ABSOLUTE_CDA":
        editable_inputs["cda_m2"] = _render_simple_number_input("aero", proposal_id, "cda_m2", editable_inputs.get("cda_m2"), debug_widget_keys=debug_widget_keys)
        return
    if proposal_type == "AERO_DELTA_CDA":
        editable_inputs["delta_CdA"] = _render_simple_number_input("aero", proposal_id, "delta_CdA", editable_inputs.get("delta_CdA"), debug_widget_keys=debug_widget_keys)
        return
    if proposal_type == "AERO_NOT_USED":
        st.caption("Aero is intentionally unchanged for this proposal.")
        return

    st.caption(f"Aero proposal type `{proposal_type}` still falls back to its current applied values.")


def _render_aero_simple_resolved_summary(context: dict) -> None:
    resolved_display = dict(context.get("resolved_display") or {})
    row = st.columns(3)
    row[0].caption(f"Resolved CdA: {_display_domain_cell(resolved_display.get('cda_m2'), 'cda_m2')}")
    row[1].caption(f"Delta CdA: {_display_domain_cell(resolved_display.get('delta_CdA'), 'delta_CdA')}")
    row[2].caption(f"Baseline CdA: {_display_domain_cell(resolved_display.get('baseline_CdA'), 'baseline_CdA')}")


def _render_simple_number_input(
    domain: str,
    proposal_id: str,
    field_key: str,
    value,
    *,
    container=None,
    debug_widget_keys: list[str] | None = None,
):
    target = container or st
    key = f"v22_simple_{_simple_widget_domain_scope(domain)}__{proposal_id}__{field_key}"
    unit_system = _current_unit_system()
    pressure_unit = _current_tire_pressure_unit(unit_system=unit_system)
    meta = field_meta(field_key)
    display_value = to_display_field_value(field_key, value, unit_system, pressure_unit=pressure_unit)
    if debug_widget_keys is not None:
        debug_widget_keys.append(key)
    kwargs = {
        "key": key,
        "min_value": float(meta["min"]) if meta.get("min") is not None else None,
        "max_value": float(meta["max"]) if meta.get("max") is not None else None,
        "step": display_step_for_field(field_key, meta.get("step"), unit_system, pressure_unit=pressure_unit),
        "format": display_format_for_field(field_key, meta.get("format"), unit_system, pressure_unit=pressure_unit),
        "placeholder": "-",
    }
    if key not in st.session_state:
        # Initialize through Session State only.  Passing value= as well makes
        # Streamlit warn and can hydrate a form widget with its numeric zero.
        st.session_state[key] = None if is_blank(display_value) else float(display_value)
    raw_value = target.number_input(meta.get("label") or field_key, **kwargs)
    return to_canonical_field_value(field_key, raw_value, unit_system, pressure_unit=pressure_unit)


def _render_simple_select_input(
    domain: str,
    proposal_id: str,
    field_key: str,
    value,
    *,
    container=None,
    debug_widget_keys: list[str] | None = None,
):
    target = container or st
    key = f"v22_simple_{_simple_widget_domain_scope(domain)}__{proposal_id}__{field_key}"
    unit_system = _current_unit_system()
    pressure_unit = _current_tire_pressure_unit(unit_system=unit_system)
    meta = field_meta(field_key)
    options = list(meta.get("options") or [])
    current = _current_select_value(field_key, value, {field_key: value}, options)
    _prime_widget_value(key, current)
    if debug_widget_keys is not None:
        debug_widget_keys.append(key)
    return target.selectbox(
        meta.get("label") or field_key,
        options,
        key=key,
        format_func=lambda item: _format_select_option(field_key, item, unit_system, pressure_unit=pressure_unit),
    )


def _render_simple_text_input(
    domain: str,
    proposal_id: str,
    field_key: str,
    value,
    *,
    container=None,
    debug_widget_keys: list[str] | None = None,
):
    target = container or st
    key = f"v22_simple_{_simple_widget_domain_scope(domain)}__{proposal_id}__{field_key}"
    meta = field_meta(field_key)
    display_value = _display_editor_value(value)
    _prime_widget_value(key, display_value)
    if debug_widget_keys is not None:
        debug_widget_keys.append(key)
    return _parse_editor_value(target.text_input(meta.get("label") or field_key, key=key, value=display_value))


def _merge_simple_corrections(current_corrections: dict, field_keys: tuple[str, ...], correction_values: dict) -> dict:
    merged = dict(current_corrections)
    for field_key in field_keys:
        value = correction_values.get(field_key)
        if is_blank(value):
            merged.pop(field_key, None)
        else:
            merged[field_key] = value
    return merged


def _render_domain_technical_details(card_payload: dict) -> None:
    summaries = list(card_payload.get("proposal_summaries") or [])
    if summaries:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Proposal": item.get("label"),
                        "Walk From": item.get("walk_from_label"),
                        "Mode": item.get("mode_label"),
                        "Kind": item.get("kind"),
                        "Status": item.get("status_label"),
                    }
                    for item in summaries
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    render_v22_chip_list([f"Fields: {', '.join(list(card_payload.get('row_keys') or []))}"])


def _render_vde_cycle_comparison_table(payload: dict) -> None:
    rows = []
    for item in list(payload.get("rows") or []):
        row = {"Result": item.get("label")}
        for column in list(payload.get("columns") or []):
            row[str(column.get("label") or column.get("id") or "")] = dict(item.get("display_values") or {}).get(column.get("id"))
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_preview_save_section(state: dict) -> None:
    st.subheader("Preview & Save")
    unit_system = _current_unit_system()
    preview = dict(state.get("preview") or {})
    component_repositories = saved_component_repositories_from_state(state)
    status_payload = build_preview_status_payload(state)
    render_v22_preview_status_strip(
        [
            {"label": "Preview", "value": status_payload.get("preview_label")},
            {"label": "Proposals", "value": status_payload.get("proposal_count")},
            {"label": "Review", "value": status_payload.get("review_count", 0)},
            {"label": "Save", "value": status_payload.get("save_status")},
        ]
    )
    if status_payload.get("baseline_correction_count"):
        render_v22_chip_list([f"{int(status_payload.get('baseline_correction_count') or 0)} baseline corrections applied"])
    if status_payload.get("stale_message"):
        render_v22_notice_strip(str(status_payload.get("stale_message") or ""), tone="warning")

    if st.button("Validate & Preview", key="v22_validate_preview"):
        started = time.perf_counter()
        bundle = build_v22_preview_bundle(
            state,
            baseline_context=compact_baseline_context(state),
            component_repositories=component_repositories,
        )
        next_state = normalize_v22_state(state)
        next_state["preview"] = {"status": "fresh", "fingerprint": bundle.get("fingerprint"), "result": bundle}
        st.session_state[V22_SESSION_KEY] = next_state
        if st.query_params.get("v22_profile") == "1":
            st.session_state["v22_last_preview_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        st.rerun()

    bundle = dict(preview.get("result") or {})
    if st.query_params.get("v22_profile") == "1" and "v22_last_preview_ms" in st.session_state:
        st.caption(f"Validate & Preview: {st.session_state['v22_last_preview_ms']:.0f} ms")

    if not bundle:
        if status_payload.get("pending_rows"):
            render_v22_group_header("Not applied")
            st.dataframe(pd.DataFrame(status_payload.get("pending_rows") or []), use_container_width=True, hide_index=True)
        if status_payload.get("incomplete_rows"):
            render_v22_group_header("Applied but incomplete")
            st.dataframe(pd.DataFrame(status_payload.get("incomplete_rows") or []), use_container_width=True, hide_index=True)
        st.info(str(status_payload.get("empty_message") or "No preview generated yet."))
        return

    overview_payload = build_scenario_overview_payload(state)
    comparison_payload = build_engineering_comparison_payload(state, unit_system)
    cycle_comparison_payload = build_vde_cycle_comparison_payload(state, unit_system)
    validation_payload = build_validation_summary_payload(state, unit_system)
    audit_payload = build_preview_audit_payload(state, unit_system)
    tabs = st.tabs(["Overview", "Engineering Comparison", "DB Preview & Save", "Technical Audit"])

    with tabs[0]:
        render_v22_group_header("Scenario Overview")
        render_v22_scenario_overview_cards(overview_payload.get("scenarios") or [])
        correction_summary = dict(audit_payload.get("baseline_corrections") or {})
        if correction_summary.get("entries"):
            render_v22_group_header("Baseline adjustments")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Domain": item.get("domain"),
                            "Field": item.get("field_label"),
                            "Printed": item.get("printed_value"),
                            "Effective": item.get("effective_value"),
                        }
                        for item in list(correction_summary.get("entries") or [])
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[1]:
        _render_roadload_analysis_block(state, unit_system)
        _render_cycle_power_analysis_block(state)
        comparison_mode = st.radio(
            "Show",
            ["Changed domains", "All domains"],
            horizontal=True,
            key="v22_engineering_comparison_scope",
        )
        changed_group_titles = set(comparison_payload.get("changed_group_titles") or [])
        groups = list(comparison_payload.get("groups") or [])
        if comparison_mode == "Changed domains":
            groups = [group for group in groups if group.get("title") in changed_group_titles]
        for group in groups:
            render_v22_group_header(group.get("title"))
            rows = []
            for row in list(group.get("rows") or []):
                display_row = {"Field": row.get("label"), "Unit": row.get("unit")}
                for column in list(comparison_payload.get("columns") or []):
                    display_row[str(column.get("label") or column.get("id") or "")] = dict(row.get("display_values") or {}).get(column.get("id"))
                rows.append(display_row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if cycle_comparison_payload.get("has_cycle_results"):
            render_v22_group_header("VDE by cycle")
            _render_vde_cycle_comparison_table(cycle_comparison_payload)

    with tabs[2]:
        _render_db_preview_and_save(state)

    with tabs[3]:
        _render_validation_details(validation_payload)

        if audit_payload.get("fingerprint"):
            st.caption(f"Fingerprint: {audit_payload.get('fingerprint')}")
        if audit_payload.get("audit_rows"):
            render_v22_group_header("Per-domain audit")
            st.dataframe(pd.DataFrame(audit_payload.get("audit_rows") or []), use_container_width=True, hide_index=True)
        resolution_result = dict(bundle.get("resolution_result") or {})
        proposal_results_by_id = {
            str(item.get("proposal_id") or ""): dict(item)
            for item in list(resolution_result.get("proposal_results") or [])
        }
        for model in list(audit_payload.get("proposal_models") or []):
            header = dict(model.get("header") or {})
            proposal_result = proposal_results_by_id.get(str(header.get("proposal_id") or ""), {})
            label = f"{header.get('requested_label') or header.get('source_column') or 'Requested'} | {header.get('status') or 'OK'}"
            with st.expander(label, expanded=False):
                st.caption(f"Walk From: {header.get('walk_from') or EM_DASH}")
                st.dataframe(
                    pd.DataFrame(_proposal_engineering_rows_for_display(proposal_result, unit_system)),
                    use_container_width=True,
                    hide_index=True,
                )
                mass_resolution_rows = _mass_resolution_rows_for_display(proposal_result, unit_system)
                if mass_resolution_rows:
                    render_v22_group_header("Mass Resolution")
                    st.dataframe(pd.DataFrame(mass_resolution_rows), use_container_width=True, hide_index=True)
                tire_resolution_rows = _tire_resolution_rows_for_display(proposal_result, unit_system)
                if tire_resolution_rows:
                    render_v22_group_header("Tire Resolution")
                    st.dataframe(pd.DataFrame(tire_resolution_rows), use_container_width=True, hide_index=True)
                domain_change_rows = _proposal_domain_change_rows_for_display(proposal_result, unit_system)
                if domain_change_rows:
                    render_v22_group_header("Domain Changes")
                    st.dataframe(pd.DataFrame(domain_change_rows), use_container_width=True, hide_index=True)
                if model.get("validation_rows"):
                    render_v22_group_header("Warnings & Issues")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Severity": row.get("Severity"),
                                    "Domain": row.get("Domain"),
                                    "Field": row.get("Field"),
                                    "Message": format_v22_issue_for_display(row, unit_system),
                                }
                                for row in list(model.get("validation_rows") or [])
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                if model.get("component_action_rows"):
                    render_v22_group_header("Component Actions")
                    st.dataframe(pd.DataFrame(model.get("component_action_rows") or []), use_container_width=True, hide_index=True)
        with st.expander("Effective metadata", expanded=False):
            _render_scenario_metadata_audit(state)


def _render_validation_details(validation_payload: dict) -> None:
        summary = dict(validation_payload.get("summary") or {})
        if summary:
            metrics = st.columns(5)
            metric_labels = [
                ("Overall", summary.get("overall_status", "Pending")),
                ("Ready proposals", f"{summary.get('ready_count', 0)} / {summary.get('proposal_count', 0)}"),
                ("Review", summary.get("review_count", 0)),
                ("Invalid", summary.get("invalid_count", 0)),
                ("Missing", summary.get("missing_count", 0)),
            ]
            for cell, (label, value) in zip(metrics, metric_labels):
                cell.metric(label, value)
        if validation_payload.get("pending_rows"):
            render_v22_group_header("Not applied")
            st.dataframe(pd.DataFrame(validation_payload.get("pending_rows") or []), use_container_width=True, hide_index=True)
        if validation_payload.get("incomplete_rows"):
            render_v22_group_header("Applied but incomplete")
            st.dataframe(pd.DataFrame(validation_payload.get("incomplete_rows") or []), use_container_width=True, hide_index=True)
        if validation_payload.get("root_issue_rows"):
            render_v22_group_header("Validation Issues")
            st.dataframe(pd.DataFrame(validation_payload.get("root_issue_rows") or []), use_container_width=True, hide_index=True)
        issue_sections = [
            section for section in list(validation_payload.get("scenario_sections") or [])
            if section.get("issue_rows") or str(section.get("status") or "") not in {"OK", "Ready"}
        ]
        for section in issue_sections:
            with st.expander(f"{section.get('label')} | {section.get('status')}", expanded=False):
                if section.get("issue_rows"):
                    st.dataframe(pd.DataFrame(section.get("issue_rows") or []), use_container_width=True, hide_index=True)
                if section.get("domain_rows"):
                    st.dataframe(pd.DataFrame(section.get("domain_rows") or []), use_container_width=True, hide_index=True)
        with st.expander("All domain statuses", expanded=False):
            for section in list(validation_payload.get("scenario_sections") or []):
                if section.get("domain_rows"):
                    st.caption(f"{section.get('label')} | {section.get('status')}")
                    st.dataframe(pd.DataFrame(section.get("domain_rows") or []), use_container_width=True, hide_index=True)


def _render_roadload_analysis_block(state: dict, unit_system: str) -> None:
    render_v22_group_header("Roadload Analysis")
    controls = st.columns([1.0, 1.2, 3.0])
    speed_option = controls[0].selectbox(
        "Maximum speed",
        [120, 140, 160],
        index=[120, 140, 160].index(int(st.session_state.get(V22_ROADLOAD_MAX_SPEED_KEY) or 140))
        if int(st.session_state.get(V22_ROADLOAD_MAX_SPEED_KEY) or 140) in {120, 140, 160}
        else 1,
        key=V22_ROADLOAD_MAX_SPEED_KEY,
    )
    boundary_mode = controls[1].radio("Show", ["TOTAL", "NET", "Both"], horizontal=True, key="v22_roadload_boundary")
    payload = build_roadload_analysis_payload(state, unit_system, speed_max_kph=int(speed_option))
    if not payload.get("has_bundle") or not payload.get("is_fresh"):
        render_v22_notice_strip(str(payload.get("message") or "Roadload curves require a fresh preview."), tone="warning")
        return
    visible_series = [
        item for item in list(payload.get("series") or [])
        if boundary_mode == "Both" or str(item.get("state_label") or "") == boundary_mode
    ]
    if not visible_series:
        st.info(str(payload.get("message") or "No resolved roadload curves are available in the fresh preview."))
        return
    visible_payload = dict(payload)
    visible_payload["series"] = visible_series
    st.plotly_chart(_build_roadload_analysis_figure(visible_payload), use_container_width=True, key="v22_roadload_curves")
    with st.expander("Roadload checkpoints", expanded=False):
        st.caption(
            f"Checkpoint speeds stay canonical in km/h. Force values are displayed in {payload.get('force_unit') or 'N'}."
        )
        checkpoint_rows = [row for row in list(payload.get("checkpoint_rows") or []) if boundary_mode == "Both" or str(row.get("State") or "") == boundary_mode]
        st.dataframe(pd.DataFrame(checkpoint_rows), use_container_width=True, hide_index=True)


def _build_roadload_analysis_figure(payload: dict) -> go.Figure:
    fig = go.Figure()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    color_by_scenario: dict[str, str] = {}
    for item in list(payload.get("series") or []):
        scenario_id = str(item.get("scenario_id") or "")
        if scenario_id not in color_by_scenario:
            color_by_scenario[scenario_id] = colors[len(color_by_scenario) % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=list(item.get("speed_display") or []),
                y=list(item.get("force_display") or []),
                mode="lines",
                name=str(item.get("legend_label") or ""),
                line={
                    "color": color_by_scenario[scenario_id],
                    "dash": str(item.get("line_dash") or "solid"),
                    "width": 2,
                },
                customdata=[
                    [str(item.get("scenario_label") or ""), str(item.get("state_label") or ""), float(speed_kph), float(force_n)]
                    for speed_kph, force_n in zip(list(item.get("speed_kph") or []), list(item.get("force_N") or []))
                ],
                hovertemplate=(
                    "%{customdata[0]} - %{customdata[1]}<br>"
                    "%{customdata[2]:.0f} km/h<br>"
                    "%{customdata[3]:.2f} N"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    fig.update_xaxes(title_text=f"Vehicle speed [{payload.get('speed_unit') or 'km/h'}]")
    fig.update_yaxes(title_text=f"Roadload force [{payload.get('force_unit') or 'N'}]")
    return fig


def _render_cycle_power_analysis_block(state: dict) -> None:
    render_v22_group_header("Cycle Power Analysis")
    initial = build_cycle_power_analysis_payload(state)
    if not initial.get("has_bundle") or not initial.get("is_fresh"):
        render_v22_notice_strip(str(initial.get("message") or "Cycle power analysis requires a fresh preview."), tone="warning")
        return
    options = list(initial.get("cycle_options") or [])
    if not options:
        st.info(str(initial.get("message") or "No physical cycle is available."))
        return
    controls = st.columns([1.0, 1.2, 3.0])
    cycle = controls[0].selectbox("Cycle", options, key="v22_cycle_power_cycle")
    boundary_mode = controls[1].radio("Show", ["TOTAL", "NET", "Both"], horizontal=True, key="v22_cycle_power_boundary")
    payload = build_cycle_power_analysis_payload(state, selected_cycle=cycle)
    visible_series = [
        item for item in list(payload.get("series") or [])
        if boundary_mode == "Both" or str(item.get("boundary") or "") == boundary_mode
    ]
    if not visible_series:
        st.info(str(payload.get("message") or "No resolved power series is available."))
        return
    st.plotly_chart(_build_cycle_speed_figure(payload), use_container_width=True, key="v22_cycle_speed")
    st.plotly_chart(_build_cycle_power_figure(payload, visible_series), use_container_width=True, key="v22_cycle_power")
    with st.expander("Component breakdown", expanded=False):
        st.info(str(payload.get("decomposition_note") or "Component attribution is not available for this resolved roadload."))


def _build_cycle_speed_figure(payload: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(payload.get("time_s") or []),
            y=list(payload.get("speed_kph") or []),
            mode="lines",
            name="Vehicle speed",
            line={"color": "#1f77b4", "width": 2},
        )
    )
    fig.update_layout(height=250, margin={"l": 10, "r": 10, "t": 20, "b": 10}, showlegend=False)
    fig.update_xaxes(title_text="Time [s]")
    fig.update_yaxes(title_text="Vehicle speed [km/h]")
    return fig


def _build_cycle_power_figure(payload: dict, series: list[dict]) -> go.Figure:
    fig = go.Figure()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    color_by_scenario: dict[str, str] = {}
    for item in series:
        scenario_id = str(item.get("scenario_id") or "")
        if scenario_id not in color_by_scenario:
            color_by_scenario[scenario_id] = colors[len(color_by_scenario) % len(colors)]
        boundary = str(item.get("boundary") or "TOTAL")
        fig.add_trace(
            go.Scatter(
                x=list(payload.get("time_s") or []),
                y=list(item.get("demanded_power_kw") or []),
                mode="lines",
                name=f"{item.get('scenario_label')} {boundary}",
                line={"color": color_by_scenario[scenario_id], "dash": "solid" if boundary == "TOTAL" else "dash", "width": 2},
            )
        )
    fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 20, "b": 10}, legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0})
    fig.update_xaxes(title_text="Time [s]")
    fig.update_yaxes(title_text="Demanded power [kW]")
    return fig


def _render_db_preview_and_save(state: dict) -> None:
    save_plan = build_v22_save_plan(state)
    summaries = list(save_plan.get("configuration_summaries") or build_scenario_configuration_summaries(state))
    summaries_by_id = {str(item.get("proposal_id") or ""): dict(item) for item in summaries}

    render_v22_group_header("Finalize configurations")
    _render_scenario_metadata_section(state, summaries_by_id)

    render_v22_group_header("Save readiness")
    proposal_rows = list(save_plan.get("proposals") or [])
    if save_plan.get("can_execute"):
        st.success(f"{len(proposal_rows)} scenario(s) ready to save.")
    else:
        for row in proposal_rows:
            if row.get("eligible"):
                continue
            st.warning(
                f"{row.get('source_column') or row.get('proposal_id')}: cannot save. "
                f"{_save_readiness_message(row)}"
            )
        for issue in list(save_plan.get("blocking_issues") or []):
            st.warning(_save_issue_message(dict(issue)))
    for issue in list(save_plan.get("warnings") or []):
        st.info(_save_issue_message(dict(issue)))

    render_v22_group_header("Configurations to save")
    for proposal_row in proposal_rows:
        proposal_id = str(proposal_row.get("proposal_id") or "")
        summary = summaries_by_id.get(proposal_id, {})
        row = dict(proposal_row.get("row_payload") or {})
        with st.container(border=True):
            st.markdown(f"**{summary.get('proposal_label') or proposal_row.get('source_column') or proposal_id}**")
            st.markdown(f"**{summary.get('program_label') or EM_DASH}**")
            st.write(summary.get("engineering_summary") or "No direct engineering changes")
            st.caption(f"Based on {summary.get('based_on') or 'Baseline'}")
            st.caption(
                f"Curb {_save_mass_display(row.get('mass_kg'))} · "
                f"VDE mass {_save_mass_display(row.get('test_mass_kg'))} · "
                f"VDE TOTAL {_save_vde_display(row.get('vde_total_mj_per_km'))} · "
                f"NET {_save_vde_display(row.get('vde_net_mj_per_km'))}"
            )
            st.caption("Ready" if proposal_row.get("eligible") else f"Cannot save: {_save_readiness_message(proposal_row)}")

    compact_rows = []
    for proposal_row in proposal_rows:
        row = dict(proposal_row.get("row_payload") or {})
        compact_rows.append(
            {
                "Proposal": proposal_row.get("source_column") or proposal_row.get("proposal_id"),
                "Name": proposal_row.get("final_name"),
                "Make": row.get("make"),
                "Model": row.get("model"),
                "Model Year": row.get("year"),
                "Legislation": row.get("legislation"),
                "Cycle": row.get("cycle_name"),
                "Curb mass": row.get("mass_kg"),
                "VDE mass": row.get("test_mass_kg"),
                "ABC TOTAL": _format_abc_row_for_display(row),
                "VDE TOTAL": row.get("vde_total_mj_per_km"),
                "VDE NET": row.get("vde_net_mj_per_km"),
                "Save status": "Ready" if proposal_row.get("eligible") else "Blocked",
            }
        )
    with st.expander("DB row preview", expanded=False):
        if compact_rows:
            st.dataframe(pd.DataFrame(compact_rows), use_container_width=True, hide_index=True)
    with st.expander("Full DB row", expanded=False):
        full_rows = []
        for proposal_row in proposal_rows:
            full_rows.append(
                {
                    "Proposal": proposal_row.get("source_column") or proposal_row.get("proposal_id"),
                    **dict(proposal_row.get("row_payload") or {}),
                }
            )
        if full_rows:
            st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)

    _render_save_panel(state, save_plan)


def _format_abc_row_for_display(row: dict) -> str:
    values = (row.get("coast_A_N"), row.get("coast_B_N_per_kph"), row.get("coast_C_N_per_kph2"))
    return " / ".join(EM_DASH if value in (None, "") else f"{float(value):.6g}" for value in values)


def _render_save_panel(state: dict, save_plan: dict) -> None:
    st.divider()
    render_v22_group_header("Save Request")
    save_state = dict(dict(state or {}).get("save") or {})
    save_result = dict(save_state.get("result") or {})
    save_status = str(save_state.get("status") or "pending").lower()
    if save_status == "success":
        st.success(
            f"Saved successfully. Record #{save_result.get('record_id')} | "
            f"{len(list(save_result.get('saved_proposals') or []))} proposal(s)."
        )
    elif save_status == "failed" and save_result:
        issues = list(save_result.get("issues") or [])
        message = str(issues[0].get("message") or "Save failed.") if issues else "Save failed."
        st.error(message)
    can_save = bool(save_plan.get("can_execute"))
    if st.button("Save request", disabled=not can_save, key="v22_save_request"):
        result = save_v22_request(state)
        next_state = normalize_v22_state(state)
        next_state["save"] = {
            "status": str(result.get("status") or "failed"),
            "result": result,
        }
        st.session_state[V22_SESSION_KEY] = next_state
        st.rerun()
    if can_save:
        st.caption("The displayed DB rows are the exact payload submitted by Save Request.")
    else:
        st.caption("Save becomes available after a fresh preview and resolution of save blockers.")


def _save_mass_display(value) -> str:
    return EM_DASH if value in (None, "") else f"{float(value):.0f} kg"


def _save_vde_display(value) -> str:
    return EM_DASH if value in (None, "") else f"{float(value):.4f} MJ/km"


def _save_readiness_message(row: dict) -> str:
    reasons = list(dict(row or {}).get("ineligible_reasons") or [])
    if reasons:
        return _save_issue_message({"message": reasons[0]})
    return _save_issue_message({"message": dict(row or {}).get("status") or "Required information needs attention."})


def _save_issue_message(issue: dict) -> str:
    message = str(dict(issue or {}).get("message") or "Save requires attention.").strip()
    translations = {
        "status_missing": "Required information is missing.",
        "status_invalid": "Some information is invalid.",
        "status_blocked": "A required dependency is blocked.",
        "preview_not_fresh": "Run Validate & Preview before saving.",
    }
    return translations.get(message.lower(), message.replace("_", " ").capitalize())


def _render_scenario_metadata_section(state: dict, summaries_by_id: dict[str, dict]) -> None:
    contexts = resolve_v22_metadata_contexts(state)
    for proposal in list(state.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        context = dict(contexts.get(proposal_id) or {})
        effective = dict(context.get("effective_metadata") or {})
        source_metadata = dict(context.get("source_metadata") or {})
        overrides = dict(context.get("overrides") or {})
        summary = dict(summaries_by_id.get(proposal_id) or {})
        current_name = suggested_scenario_name(summary, effective.get("name"))
        with st.container(border=True):
            st.markdown(f"**{summary.get('proposal_label') or proposal_display_label(state, proposal)}**")
            st.markdown(f"**{summary.get('program_label') or EM_DASH}**")
            st.write(summary.get("engineering_summary") or "No direct engineering changes")
            st.caption(f"Based on {summary.get('based_on') or 'Baseline'}")

            st.markdown("**Metadata**")
            source_values = {"inherit": "Inherit from Walk From", "existing_vde": "Copy from existing VDE", "custom": "Custom"}
            source_key = str(proposal.get("metadata_source") or "inherit")
            source = st.radio(
                "Metadata source",
                list(source_values),
                index=list(source_values).index(source_key) if source_key in source_values else 0,
                format_func=source_values.get,
                horizontal=True,
                key=f"v22_metadata_source__{proposal_id}",
            )
            source_overrides = _render_existing_vde_metadata_picker(proposal_id) if source == "existing_vde" else dict(overrides)
            if st.button("Use metadata", key=f"v22_metadata_source_apply__{proposal_id}"):
                st.session_state[V22_SESSION_KEY] = apply_v22_proposal_metadata(
                    state,
                    proposal_id,
                    source_overrides,
                    metadata_source=source,
                )
                st.rerun()

            with st.form(f"v22_metadata_form__{proposal_id}"):
                edited = {
                    "name": st.text_input(
                        "Scenario name",
                        value=current_name,
                        key=f"v22_final_name__{proposal_id}",
                    )
                }
                primary_cols = st.columns(4)
                make_value = _display_editor_value(effective.get("make"))
                logo_path = search_logo(
                    {"make": make_value},
                    base_dir=str(Path(__file__).resolve().parents[3] / "data" / "images" / "logos"),
                )
                if logo_path:
                    primary_cols[0].image(logo_path, width=34)
                db_rows = fetch_vde_all_rows()
                edited_make = _render_db_metadata_choice(primary_cols[0], proposal_id, "make", overrides, effective, db_rows)
                edited["make"] = edited_make
                selected_make = edited_make or make_value
                edited["model"] = _render_db_metadata_choice(primary_cols[1], proposal_id, "model", overrides, effective, db_rows, make=selected_make)
                selected_model = edited["model"] or str(effective.get("model") or "")
                edited["model_year"] = _render_db_metadata_choice(primary_cols[2], proposal_id, "model_year", overrides, effective, db_rows, make=selected_make, model=selected_model)
                edited["category"] = _render_metadata_override_widget(primary_cols[3], proposal_id, "category", overrides=overrides, effective=effective)
                for index, field_key in enumerate(("make", "model", "model_year", "category")):
                    primary_cols[index].caption(_metadata_provenance(field_key, overrides, proposal))

                technical_cols = st.columns(4)
                for index, field_key in enumerate(("electrification", "transmission_type", "drive_type", "fuel_type")):
                    edited[field_key] = _render_metadata_override_widget(
                        technical_cols[index],
                        proposal_id,
                        field_key,
                        overrides=overrides,
                        effective=effective,
                    )
                    technical_cols[index].caption(_metadata_provenance(field_key, overrides, proposal))

                edited["description"] = st.text_area(
                    "Description",
                    value=_display_editor_value(effective.get("description")),
                    key=f"v22_meta__{proposal_id}__description",
                )
                st.caption(
                    f"{_display_editor_value(effective.get('legislation')) or EM_DASH} | "
                    f"{_display_editor_value(effective.get('cycle_name')) or EM_DASH}"
                )
                submitted = st.form_submit_button("Apply metadata")
            if submitted:
                next_state = apply_v22_proposal_metadata(
                    state,
                    proposal_id,
                    _normalize_metadata_editor_overrides(
                        edited,
                        source_metadata=source_metadata,
                        effective_metadata=effective,
                        existing_overrides=overrides,
                        metadata_source=str(proposal.get("metadata_source") or "inherit"),
                    ),
                    metadata_source=str(proposal.get("metadata_source") or "inherit"),
                )
                st.session_state[V22_SESSION_KEY] = next_state
                st.rerun()


def _render_existing_vde_metadata_picker(proposal_id: str) -> dict:
    rows = fetch_vde_all_rows()
    makes = ["All"] + sorted({str(row.get("make") or "").strip() for row in rows if str(row.get("make") or "").strip()})
    selected_make = st.selectbox("Make", makes, key=f"v22_meta_copy_make__{proposal_id}")
    filtered = [row for row in rows if selected_make == "All" or str(row.get("make") or "").strip() == selected_make]
    option_ids = [str(row.get("id")) for row in filtered if row.get("id") is not None]
    if not option_ids:
        st.info("No existing VDE rows match this metadata filter.")
        return {}
    selected_id = st.selectbox(
        "Reference VDE / program",
        option_ids,
        format_func=lambda value: _existing_vde_metadata_label(next((row for row in filtered if str(row.get("id")) == value), {})),
        key=f"v22_meta_copy_vde__{proposal_id}",
    )
    row = next((dict(item) for item in filtered if str(item.get("id")) == str(selected_id)), {})
    return {
        "description": row.get("notes"),
        "make": row.get("make"),
        "model": row.get("model"),
        "model_year": row.get("year"),
        "category": row.get("category"),
        "electrification": row.get("electrification"),
        "transmission_type": row.get("transmission_type"),
        "drive_type": row.get("drive_type"),
        "fuel_type": row.get("fuel_type"),
    }


def _metadata_provenance(field_key: str, overrides: dict, proposal: dict) -> str:
    if field_key in dict(overrides or {}):
        return "copied" if str(dict(proposal or {}).get("metadata_source") or "") == "existing_vde" else "edited"
    return "inherited"


def _existing_vde_metadata_label(row: dict) -> str:
    return " | ".join(str(row.get(key) or EM_DASH) for key in ("id", "make", "model", "year"))


def _render_db_metadata_choice(cell, proposal_id: str, field_key: str, overrides: dict, effective: dict, rows: list[dict], *, make: str = "", model: str = "") -> str:
    column = {"make": "make", "model": "model", "model_year": "year"}[field_key]
    candidates = []
    for row in rows:
        if make and field_key in {"model", "model_year"} and str(row.get("make") or "").strip() != make:
            continue
        if model and field_key == "model_year" and str(row.get("model") or "").strip() != model:
            continue
        value = str(row.get(column) or "").strip()
        if value:
            candidates.append(value)
    current = _display_editor_value(effective.get(field_key))
    options = [*sorted(set(candidates)), "OTHER / CUSTOM"]
    if current and current not in options:
        options.insert(-1, current)
    selected = cell.selectbox(METADATA_FIELD_LABELS[field_key], options, index=options.index(current) if current in options else 0, key=f"v22_meta_db__{proposal_id}__{field_key}")
    if selected == "OTHER / CUSTOM":
        return cell.text_input(f"{METADATA_FIELD_LABELS[field_key]} custom", key=f"v22_meta_db_custom__{proposal_id}__{field_key}", placeholder=_display_editor_value(effective.get(field_key)))
    return selected


def _render_scenario_metadata_audit(state: dict) -> None:
    contexts = resolve_v22_metadata_contexts(state)
    rows = []
    for proposal in list(state.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        effective = dict(dict(contexts.get(proposal_id) or {}).get("effective_metadata") or {})
        rows.append({"Proposal": proposal_display_label(state, proposal), **{METADATA_FIELD_LABELS.get(key, key): _display_editor_value(effective.get(key)) for key in V22_PROPOSAL_METADATA_FIELDS}})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_metadata_override_widget(cell, proposal_id: str, field_key: str, *, overrides: dict, effective: dict):
    current_effective = effective.get(field_key)
    legislation = str(effective.get("legislation") or "")
    category = str(effective.get("category") or "")
    spec = metadata_field_spec(
        field_key,
        legislation=legislation,
        category=category,
        current_value=_display_editor_value(current_effective),
    )
    widget_key = f"v22_meta__{proposal_id}__{field_key}"
    label = METADATA_FIELD_LABELS[field_key]
    if spec.get("widget") == "select":
        options = [option for option in list(spec.get("options") or []) if option != "(inherit)"]
        selected_default = _metadata_selected_default(field_key, current_effective, options)
        selected = cell.selectbox(
            label,
            options,
            index=options.index(selected_default),
            key=widget_key,
        )
        custom_value = None
        if spec.get("allow_custom") and selected == spec.get("custom_option"):
            custom_value = cell.text_input(
                f"{label} custom",
                value="",
                key=f"{widget_key}__custom",
                placeholder=_display_editor_value(current_effective),
            )
        return metadata_override_value(field_key, selected, custom_value=custom_value)
    return cell.text_input(
        label,
        value=_display_editor_value(current_effective),
        key=widget_key,
    )


def _metadata_selected_default(field_key: str, effective_value, options: list[str]) -> str:
    current = metadata_override_value(field_key, _display_editor_value(effective_value))
    if current and current in options:
        return current
    return options[0] if options else ""


def _normalize_metadata_editor_overrides(
    edited: dict,
    *,
    source_metadata: dict,
    effective_metadata: dict,
    existing_overrides: dict,
    metadata_source: str,
) -> dict:
    normalized = {}
    for field_key, raw_value in dict(edited or {}).items():
        value = metadata_override_value(field_key, raw_value)
        source_value = metadata_override_value(field_key, source_metadata.get(field_key))
        effective_value = metadata_override_value(field_key, effective_metadata.get(field_key))
        existing_value = metadata_override_value(field_key, existing_overrides.get(field_key))
        if not value:
            continue
        if str(metadata_source or "") == "existing_vde" and existing_value and value == effective_value:
            normalized[field_key] = existing_value
        elif value != source_value:
            normalized[field_key] = value
    return normalized


def _render_effective_metadata_table(effective: dict) -> None:
    rows = []
    for field_key in V22_PROPOSAL_METADATA_FIELDS:
        rows.append(
            {
                "Field": METADATA_FIELD_LABELS.get(field_key, field_key),
                "Effective": _display_editor_value(effective.get(field_key)),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_lookup_panel(state: dict, domain: str) -> None:
    unit_system = _current_unit_system()
    st.markdown("**Browse source**")
    target_rows = _lookup_targets(state, domain)
    target_options = [row["proposal_id"] for row in target_rows]
    target_key = f"v22_lookup_target__{domain}"
    source_options = lookup_source_options(domain)
    source_key = f"v22_lookup_source__{domain}"
    if st.session_state.get(source_key) not in source_options:
        st.session_state[source_key] = default_lookup_source(domain)
    if target_options and st.session_state.get(target_key) not in target_options:
        st.session_state[target_key] = target_options[0]
    if domain == "tire":
        source_kind = st.radio(
            "Source",
            source_options,
            horizontal=True,
            key=source_key,
        )
        target_id = str(st.session_state.get(target_key) or "")
        if is_component_lookup_source(domain, source_kind):
            _render_tire_database_browser(state, target_rows, target_id, unit_system)
            return
        _render_existing_vde_tire_lookup(state, target_rows, target_id, unit_system)
        return
    else:
        cols = st.columns([1.0, 1.8])
        source_kind = cols[0].radio(
            "Source",
            source_options,
            horizontal=True,
            key=source_key,
        )
        query = cols[1].text_input("Search", key=f"v22_lookup_query__{domain}")
        target_id = str(st.session_state.get(target_key) or "")

    query = str(st.session_state.get(f"v22_lookup_query__{domain}") or "")
    source_kind = str(st.session_state.get(source_key) or default_lookup_source(domain))
    results = component_lookup_rows(domain, query) if is_component_lookup_source(domain, source_kind) else vde_lookup_rows(domain, query)
    st.session_state[f"v22_lookup_results__{domain}"] = results
    st.session_state[f"v22_lookup_results_source__{domain}"] = source_kind
    if not results:
        message = lookup_empty_message(
            domain,
            st.session_state.get(f"v22_lookup_results_source__{domain}", st.session_state.get(source_key, default_lookup_source(domain))),
            st.session_state.get(f"v22_lookup_query__{domain}", ""),
            results,
        )
        if message:
            st.info(message)
        return

    display_rows = _display_lookup_rows(results, unit_system)
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
    option_ids = [str(dict(row).get("lookup_id") or "") for row in results if str(dict(row).get("lookup_id") or "").strip()]
    if not option_ids:
        st.info("No selectable rows were returned for this query.")
        return
    selected_id = st.selectbox(
        "Selected tire" if domain == "tire" else "Selected row",
        option_ids,
        format_func=lambda value: _lookup_result_label(results, value),
        key=f"v22_lookup_selected__{domain}",
    )
    selected_row = next((dict(row) for row in results if str(dict(row).get("lookup_id") or "") == str(selected_id)), None)
    if domain == "tire" and selected_row is not None:
        smerf_value = _lookup_numeric_or_none(selected_row.get("SMERF"))
        smerf_label = EM_DASH if smerf_value is None else f"{smerf_value:.10g}"
        summary_items = [
            f"Tire ID: {selected_row.get('Tire ID') or EM_DASH}",
            f"Tire code: {selected_row.get('Tire code') or EM_DASH}",
            f"RRC: {_display_domain_cell(selected_row.get('RRC'), 'rrc_N_per_kN')}",
            f"SMERF: {smerf_label}",
            f"Reference pressure: {_display_domain_cell(selected_row.get('Reference pressure'), 'front_pressure_psi')}",
            f"Test load: {_display_domain_cell(selected_row.get('Test load'), 'test_mass_kg')}",
        ]
        st.caption("Selected reference")
        render_v22_chip_list(summary_items)
    action_cols = st.columns([1.8, 1.0])
    target_id = action_cols[0].selectbox(
        "Apply to",
        target_options,
        format_func=lambda value: _lookup_target_label(target_rows, value),
        key=target_key,
        disabled=not target_options,
    ) if target_options else ""
    if action_cols[1].button("Use selected tire" if domain == "tire" else "Use selected row", key=f"v22_lookup_apply__{domain}", use_container_width=True):
        source_kind = str(st.session_state.get(f"v22_lookup_results_source__{domain}", st.session_state.get(f"v22_lookup_source__{domain}", "Component DB")))
        target_id = str(st.session_state.get(f"v22_lookup_target__{domain}") or "")
        normalized = normalize_v22_state(state)
        proposal = next((item for item in normalized.get("proposals") or [] if str(item.get("proposal_id") or "") == target_id), None)
        if proposal is None or selected_row is None:
            st.warning("Pick a valid target proposal and lookup row.")
            return
        domain_payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
        selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
        lookup_inputs = apply_lookup_to_inputs(domain, source_kind, selected_row)
        populated = _apply_lookup_to_widget_state(
            st.session_state,
            domain,
            target_id,
            proposal_type,
            selection_mode,
            lookup_inputs,
            unit_system=unit_system,
            pressure_unit=_current_tire_pressure_unit(state, unit_system),
        )
        if populated:
            if domain in {"transmission", "brake", "axle_hubs", "parasitic"}:
                _stage_component_lookup_draft(domain, target_id, populated)
            st.info("Lookup row loaded into editable inputs. Apply this domain to commit it.")
        else:
            st.warning("The selected row does not provide editable inputs for this proposal.")
        st.rerun()


def _render_tire_database_browser(state: dict, target_rows: list[dict], target_id: str, unit_system: str) -> None:
    all_rows = list(component_lookup_rows("tire", "", limit=None))
    with st.expander("Browse Tire Database", expanded=True):
        filter_row = st.columns([1.8, 0.9, 0.9, 1.0])
        code_query = filter_row[0].text_input("Tire code contains", key="v22_tire_browser_code_query")
        rrc_min = filter_row[1].number_input("RRC min", key="v22_tire_browser_rrc_min", value=None, step=0.1, format="%.1f", placeholder="-")
        rrc_max = filter_row[2].number_input("RRC max", key="v22_tire_browser_rrc_max", value=None, step=0.1, format="%.1f", placeholder="-")
        mileage_mode = filter_row[3].selectbox("Mileage", ["All", "0 km", ">0 km"], key="v22_tire_browser_mileage_mode")
        with st.expander("Advanced filters & details", expanded=False):
            advanced_row = st.columns([1.0, 1.0, 1.0, 1.0])
            pressure_min = advanced_row[0].number_input("Pressure min", key="v22_tire_browser_pressure_min", value=None, step=0.5, format="%.1f", placeholder="-")
            pressure_max = advanced_row[1].number_input("Pressure max", key="v22_tire_browser_pressure_max", value=None, step=0.5, format="%.1f", placeholder="-")
            load_min = advanced_row[2].number_input("Load min", key="v22_tire_browser_load_min", value=None, step=1.0, format="%.0f", placeholder="-")
            load_max = advanced_row[3].number_input("Load max", key="v22_tire_browser_load_max", value=None, step=1.0, format="%.0f", placeholder="-")
            filtered_rows = _tire_browser_filter_rows(
                all_rows,
                code_query=code_query,
                rrc_min=rrc_min,
                rrc_max=rrc_max,
                pressure_min=pressure_min,
                pressure_max=pressure_max,
                load_min=load_min,
                load_max=load_max,
                mileage_mode=mileage_mode,
            )
            runtime_snapshot = _tire_browser_runtime_snapshot(all_rows, filtered_rows)
            st.caption(f"Tire DB: {runtime_snapshot['path']}")
            st.caption(
                "Records: "
                f"table={runtime_snapshot['table_total'] if runtime_snapshot['table_total'] is not None else EM_DASH}, "
                f"active={runtime_snapshot['active_total'] if runtime_snapshot['active_total'] is not None else EM_DASH}, "
                f"before filters={runtime_snapshot['total_before_filters']}, "
                f"after filters={runtime_snapshot['filtered_total']}"
            )
            st.caption(
                "QA codes present: "
                + (", ".join(runtime_snapshot["qa_codes"]) if runtime_snapshot["qa_codes"] else "none")
            )
        st.session_state["v22_lookup_results_source__tire"] = "Tire Database"
        filters_active = _tire_browser_filters_active(
            code_query=code_query,
            rrc_min=rrc_min,
            rrc_max=rrc_max,
            pressure_min=pressure_min,
            pressure_max=pressure_max,
            load_min=load_min,
            load_max=load_max,
            mileage_mode=mileage_mode,
        )

        if not filtered_rows:
            st.session_state["v22_lookup_results__tire"] = []
            message = lookup_empty_message(
                "tire",
                "Tire Database",
                code_query,
                filtered_rows,
                filters_active=filters_active,
            )
            st.info(message or "No matching Tire Database records.")
            return

        signature = tuple(str(dict(row).get("lookup_id") or "") for row in filtered_rows)
        if st.session_state.get("v22_tire_browser_signature") != signature:
            st.session_state["v22_tire_browser_signature"] = signature
            st.session_state["v22_tire_browser_page"] = 0
        page_payload = _paginate_lookup_rows(
            filtered_rows,
            page=int(st.session_state.get("v22_tire_browser_page", 0) or 0),
            page_size=TIRE_LOOKUP_BROWSE_LIMIT,
        )
        st.session_state["v22_tire_browser_page"] = int(page_payload["page"])
        visible_rows = list(page_payload["rows"])
        st.session_state["v22_lookup_results__tire"] = visible_rows

        start = int(page_payload["start"]) + 1
        end = int(page_payload["end"])
        total = int(page_payload["total"])
        st.caption(f"Showing {start}-{end} of {total} records")
        display_rows = _display_lookup_rows(visible_rows, unit_system)
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        if int(page_payload["total_pages"]) > 1:
            pager = st.columns([0.8, 1.8, 0.8])
            if pager[0].button(
                "Previous",
                key="v22_tire_browser_prev",
                disabled=int(page_payload["page"]) <= 0,
                use_container_width=True,
            ):
                st.session_state["v22_tire_browser_page"] = max(int(page_payload["page"]) - 1, 0)
                st.rerun()
            pager[1].caption(f"Page {int(page_payload['page']) + 1} of {int(page_payload['total_pages'])}")
            if pager[2].button(
                "Next",
                key="v22_tire_browser_next",
                disabled=int(page_payload["page"]) >= int(page_payload["total_pages"]) - 1,
                use_container_width=True,
            ):
                st.session_state["v22_tire_browser_page"] = min(
                    int(page_payload["page"]) + 1,
                    int(page_payload["total_pages"]) - 1,
                )
                st.rerun()

        option_ids = [str(dict(row).get("lookup_id") or "") for row in visible_rows if str(dict(row).get("lookup_id") or "").strip()]
        if not option_ids:
            st.info("No selectable tire rows were returned.")
            return
        if str(st.session_state.get("v22_lookup_selected__tire") or "") not in option_ids:
            st.session_state["v22_lookup_selected__tire"] = option_ids[0]
        selected_id = st.selectbox(
            "Select tire record",
            option_ids,
            format_func=lambda value: _lookup_result_label(visible_rows, value),
            key="v22_lookup_selected__tire",
        )
        selected_row = next((dict(row) for row in visible_rows if str(dict(row).get("lookup_id") or "") == str(selected_id)), None)
        if selected_row is not None:
            raw_selected = dict(selected_row.get("_raw") or {})
            smerf_value = _lookup_numeric_or_none(selected_row.get("SMERF"))
            smerf_label = EM_DASH if smerf_value is None else f"{smerf_value:.10g}"
            summary_items = [
                f"Tire ID: {selected_row.get('Tire ID') or EM_DASH}",
                f"Tire code: {selected_row.get('Tire code') or EM_DASH}",
                f"RRC: {_display_domain_cell(selected_row.get('RRC'), 'rrc_N_per_kN')}",
                f"SMERF: {smerf_label}",
                f"Reference pressure: {_display_domain_cell(selected_row.get('Reference pressure'), 'front_pressure_psi')}",
                f"Test load: {_display_domain_cell(selected_row.get('Test load'), 'test_mass_kg')}",
                f"Mileage: {selected_row.get('Mileage') if not is_blank(selected_row.get('Mileage')) else EM_DASH}",
            ]
            st.caption("Selected Tire")
            render_v22_chip_list(summary_items)
            sae_items = [
                ("alpha", raw_selected.get("sae_alpha")),
                ("beta", raw_selected.get("sae_beta")),
                ("a", raw_selected.get("sae_a")),
                ("b", raw_selected.get("sae_b")),
                ("c", raw_selected.get("sae_c")),
            ]
            if any(not is_blank(value) for _, value in sae_items):
                st.caption("SAE coefficients")
                cols = st.columns(len(sae_items))
                for col, (label, value) in zip(cols, sae_items):
                    display_value = EM_DASH if is_blank(value) else f"{float(value):.10g}"
                    col.caption(label)
                    col.write(display_value)
        action_cols = st.columns([1.8, 1.0])
        target_id = action_cols[0].selectbox(
            "Apply to",
            [row["proposal_id"] for row in target_rows],
            format_func=lambda value: _lookup_target_label(target_rows, value),
            key="v22_lookup_target__tire",
        )
        if action_cols[1].button("Use selected tire", key="v22_lookup_apply__tire", use_container_width=True):
            _apply_selected_lookup_row(state, "tire", target_id, "Tire Database", selected_row, unit_system)


def _render_existing_vde_tire_lookup(state: dict, target_rows: list[dict], target_id: str, unit_system: str) -> None:
    query = st.text_input("Search", key="v22_lookup_query__tire")
    results = list(vde_lookup_rows("tire", query, limit=25))
    st.session_state["v22_lookup_results__tire"] = results
    st.session_state["v22_lookup_results_source__tire"] = "Existing VDE"
    if not results:
        message = lookup_empty_message("tire", "Existing VDE", st.session_state.get("v22_lookup_query__tire", ""), results)
        st.info(message)
        return

    st.dataframe(pd.DataFrame(_display_lookup_rows(results, unit_system)), use_container_width=True, hide_index=True)
    option_ids = [str(dict(row).get("lookup_id") or "") for row in results if str(dict(row).get("lookup_id") or "").strip()]
    if not option_ids:
        st.info("No selectable rows were returned for this query.")
        return
    selected_id = st.selectbox(
        "Selected row",
        option_ids,
        format_func=lambda value: _lookup_result_label(results, value),
        key="v22_lookup_selected__tire",
    )
    selected_row = next((dict(row) for row in results if str(dict(row).get("lookup_id") or "") == str(selected_id)), None)
    action_cols = st.columns([1.8, 1.0])
    target_id = action_cols[0].selectbox(
        "Apply to",
        [row["proposal_id"] for row in target_rows],
        format_func=lambda value: _lookup_target_label(target_rows, value),
        key="v22_lookup_target__tire",
    )
    if action_cols[1].button("Use selected tire", key="v22_lookup_apply__tire", use_container_width=True):
        _apply_selected_lookup_row(state, "tire", target_id, "Existing VDE", selected_row, unit_system)


def _apply_selected_lookup_row(state: dict, domain: str, target_id: str, source_kind: str, selected_row: dict | None, unit_system: str) -> None:
    normalized = normalize_v22_state(state)
    proposal = next((item for item in normalized.get("proposals") or [] if str(item.get("proposal_id") or "") == str(target_id or "")), None)
    if proposal is None or selected_row is None:
        st.warning("Pick a valid target proposal and lookup row.")
        return
    domain_payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
    proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
    selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
    lookup_inputs = apply_lookup_to_inputs(domain, source_kind, selected_row)
    populated = _apply_lookup_to_widget_state(
        st.session_state,
        domain,
        target_id,
        proposal_type,
        selection_mode,
        lookup_inputs,
        unit_system=unit_system,
        pressure_unit=_current_tire_pressure_unit(state, unit_system),
    )
    if populated:
        if domain in {"transmission", "brake", "axle_hubs", "parasitic"}:
            _stage_component_lookup_draft(domain, target_id, populated)
        st.info("Lookup row loaded into editable inputs. Apply this domain to commit it.")
    else:
        st.warning("The selected row does not provide editable inputs for this proposal.")
    st.rerun()


def _apply_lookup_to_widget_state(
    session_state,
    domain: str,
    proposal_id: str,
    proposal_type: str,
    selection_mode: str,
    lookup_inputs: dict | None,
    *,
    unit_system: str | None = None,
    pressure_unit: str | None = None,
) -> dict:
    populated = {}
    current_inputs = {}
    fields = set(applicable_fields(domain, proposal_type, selection_mode))
    snapshot_key = _domain_widget_key(domain, proposal_id, "tire_snapshot")
    for field_key, value in dict(lookup_inputs or {}).items():
        if field_key not in fields:
            continue
        if is_blank(value):
            if domain in COMPONENT_SIMPLE_FIELD_CONFIG and field_key in {
                COMPONENT_SIMPLE_FIELD_CONFIG[domain]["lookup_id"],
                COMPONENT_SIMPLE_FIELD_CONFIG[domain]["vde_id"],
            }:
                session_state[_domain_widget_key(domain, proposal_id, field_key)] = ""
                current_inputs[field_key] = ""
                populated[field_key] = ""
            continue
        schema = field_schema(domain, proposal_type, selection_mode, field_key, inputs=current_inputs)
        widget = schema.get("widget") or schema.get("kind") or "text"
        key = _domain_widget_key(domain, proposal_id, field_key)
        if widget == "select":
            stored_value = _current_select_value(field_key, value, current_inputs, list(schema.get("options") or []))
        elif field_meta(field_key).get("kind") == "number":
            display_value = to_display_field_value(field_key, value, unit_system or _current_unit_system(), pressure_unit=pressure_unit)
            stored_value = None if is_blank(display_value) else float(display_value)
        else:
            stored_value = _display_editor_value(value)
        session_state[key] = stored_value
        current_inputs[field_key] = value
        populated[field_key] = value
    if domain == "tire":
        tire_snapshot = dict(lookup_inputs or {}).get("tire_snapshot")
        if isinstance(tire_snapshot, dict) and tire_snapshot:
            session_state[snapshot_key] = deepcopy(tire_snapshot)
    return populated


def _lookup_targets(state: dict, domain: str) -> list[dict]:
    rows = []
    for proposal in list(state.get("proposals") or []):
        payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        proposal_type = str(payload.get("proposal_type") or "INHERIT")
        if proposal_type == "INHERIT":
            continue
        rows.append(
            {
                "proposal_id": str(proposal.get("proposal_id") or ""),
                "display_index": int(proposal.get("display_index") or 0),
                "selection_mode": str(payload.get("selection_mode") or proposal_type),
            }
        )
    return rows


def _lookup_target_label(rows: list[dict], proposal_id: str) -> str:
    for row in rows:
        if str(row.get("proposal_id") or "") == str(proposal_id or ""):
            return f"Requested #{row.get('display_index')} | {row.get('selection_mode')}"
    return str(proposal_id or "")


def _lookup_result_label(rows: list[dict], lookup_id: str) -> str:
    for row in rows:
        payload = dict(row or {})
        if str(payload.get("lookup_id") or "") != str(lookup_id or ""):
            continue
        for key in ("Tire ID", "VDE ID", "Tire code", "ID", "Code / Name", "Make", "Model"):
            if not is_blank(payload.get(key)):
                head = payload.get(key)
                break
        else:
            head = lookup_id
        description = payload.get("Tire code") or payload.get("Description") or payload.get("Notes") or payload.get("Model") or payload.get("Code / Name")
        return f"{head} | {description}" if not is_blank(description) else str(head)
    return str(lookup_id or "")


def _domain_contexts(domain: str, state: dict, baseline: dict) -> tuple[dict, dict[str, dict]]:
    baseline_display = _baseline_domain_display(domain, baseline)
    mass_contexts = {}
    if domain == "tire":
        _, mass_contexts = _domain_contexts("mass", state, baseline)
    resolved_by_id = {}
    contexts = {}
    for proposal in list(state.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        walk_from_id = str(proposal.get("walk_from") or "baseline")
        source_display = baseline_display if walk_from_id == "baseline" else deepcopy(dict(resolved_by_id.get(walk_from_id) or baseline_display))
        domain_payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
        selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
        inputs = deepcopy(dict(dict(proposal.get("inputs") or {}).get(domain) or {}))
        resolved_display = resolve_domain_display(
            domain,
            source_display,
            {
                "domains": {domain: domain_payload},
                "inputs": {domain: inputs},
            },
        )
        if domain == "tire":
            mass_display = dict(dict(mass_contexts.get(proposal_id) or {}).get("resolved_display") or {})
            for field_key in ("tire_load_mass_used_kg", "tire_load_mass_basis"):
                if mass_display.get(field_key) is not None:
                    resolved_display[field_key] = mass_display[field_key]
        resolved_by_id[proposal_id] = deepcopy(resolved_display)
        contexts[proposal_id] = {
            "proposal_type": proposal_type,
            "selection_mode": selection_mode,
            "source_display": source_display,
            "resolved_display": resolved_display,
            "inputs": inputs,
        }
    return baseline_display, contexts


def _baseline_domain_display(domain: str, baseline: dict) -> dict:
    display = resolve_domain_display(
        domain,
        baseline,
        {
            "domains": {domain: {"proposal_type": "INHERIT", "selection_mode": "Inherit"}},
            "inputs": {},
        },
    )
    if domain == "transmission":
        total = dict(baseline.get("abc_total") or baseline.get("initial_abc_total") or {})
        for component, field_key in (("A", "source_abc_total_A"), ("B", "source_abc_total_B"), ("C", "source_abc_total_C")):
            value = baseline.get(component)
            if is_blank(value):
                value = total.get(component)
            display[field_key] = value
    return display


def _render_domain_apply_status(state: dict, domain: str) -> None:
    payload = dict(dict(state.get("domain_input_state") or {}).get(domain) or {})
    proposal_statuses = dict(payload.get("proposal_statuses") or {})
    ready = sum(1 for item in proposal_statuses.values() if str(dict(item or {}).get("status") or "") == "applied_ready")
    incomplete = sum(1 for item in proposal_statuses.values() if str(dict(item or {}).get("status") or "") == "applied_incomplete")
    status = str(payload.get("status") or "not_configured")
    status_label = {
        "not_configured": "Not configured",
        "applied_ready": f"Applied â€” {ready} ready, {incomplete} incomplete",
        "applied_incomplete": f"Applied â€” {ready} ready, {incomplete} incomplete",
        "stale_after_matrix_change": "Stale after Proposal Matrix change",
    }.get(status, status.replace("_", " ").title())
    st.caption(f"{DOMAIN_LABELS[domain]} inputs: {status_label}")
    st.caption(f"Preview: {dict(state.get('preview') or {}).get('status') or 'not_run'}")
    if payload.get("last_applied_at"):
        st.caption(f"Last applied: {payload.get('last_applied_at')}")
    if payload.get("last_apply_message") and status in {"applied_ready", "applied_incomplete"}:
        st.success(str(payload.get("last_apply_message")))
    elif payload.get("last_apply_message") and status == "stale_after_matrix_change":
        st.warning(str(payload.get("last_apply_message")))


def _current_widget_inputs(
    domain: str,
    proposal_id: str,
    proposal_type: str,
    selection_mode: str,
    editable_inputs: dict,
) -> dict:
    current = deepcopy(dict(editable_inputs or {}))
    unit_system = _current_unit_system()
    pressure_unit = _current_tire_pressure_unit(unit_system=unit_system)
    for input_field_key in applicable_fields(domain, proposal_type, selection_mode):
        widget_key = _existing_domain_widget_key(domain, proposal_id, input_field_key)
        if widget_key not in st.session_state:
            continue
        schema = field_schema(domain, proposal_type, selection_mode, input_field_key, inputs=current)
        current[input_field_key] = _widget_state_value(
            st.session_state.get(widget_key),
            schema.get("widget"),
            input_field_key,
            unit_system,
            pressure_unit,
        )
    if domain == "tire" and proposal_type in {"TIRE_DB_LOOKUP", "TIRE_METADATA_ONLY"}:
        snapshot_key = _existing_domain_widget_key(domain, proposal_id, "tire_snapshot")
        snapshot_value = st.session_state.get(snapshot_key)
        if isinstance(snapshot_value, dict) and snapshot_value:
            current["tire_snapshot"] = deepcopy(snapshot_value)
    return current


def _widget_state_value(value, widget_type: str | None = None, field_key: str | None = None, unit_system: str | None = None, pressure_unit: str | None = None):
    if widget_type == "select":
        return value
    if widget_type == "number":
        return to_canonical_field_value(field_key, value, unit_system, pressure_unit=pressure_unit)
    if isinstance(value, str):
        return _parse_editor_value(value)
    return value


def _prime_widget_value(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _prime_correction_widget_cache(state: dict) -> None:
    corrections = dict(dict(state.get("baseline") or {}).get("corrections") or {})
    correction_disposition = str(dict(state.get("baseline") or {}).get("correction_disposition") or "request_only")
    unit_system = _current_unit_system()
    pressure_unit = _current_tire_pressure_unit(state, unit_system)
    _prime_widget_value("v22_correction_disposition", correction_disposition)
    for domain, field_keys in DOMAIN_BASELINE_FIELDS.items():
        _prime_widget_value(f"v22_correction_disposition__{domain}", correction_disposition)
        for field_key in field_keys:
            key = f"v22_correction__{domain}__{field_key}"
            if field_meta(field_key).get("kind") == "number":
                display_value = to_display_field_value(field_key, corrections.get(field_key), unit_system, pressure_unit=pressure_unit)
                _prime_widget_value(key, None if is_blank(display_value) else float(display_value))
            else:
                _prime_widget_value(key, _display_editor_value(corrections.get(field_key)))


def _current_select_value(field_key: str, value, editable_inputs: dict, options: list):
    if field_key == "shift_steps":
        shift = value
        if is_blank(shift):
            return "Select target" if not is_blank(dict(editable_inputs or {}).get("target_mass_kg")) else (options[0] if options else None)
        try:
            numeric = int(float(shift))
            label = f"{numeric:+d}"
            return label if label in options else (options[0] if options else None)
        except Exception:
            text = str(shift)
            return text if text in options else (options[0] if options else None)
    if value in options:
        return value
    return options[0] if options else None


def _format_select_option(field_key: str, value, unit_system: str | None = None, pressure_unit: str | None = None) -> str:
    if field_key == "transmission_application_mode":
        return {
            "APPLY_DELTA_TO_TOTAL": "Vehicle change - apply transmission delta to TOTAL",
            "KEEP_TOTAL_FIXED": "Fixed measured TOTAL - recalculate NET only",
        }.get(str(value or "").strip().upper(), str(value))
    if quantity_kind_for_field(field_key):
        return format_select_option_for_field(field_key, value, unit_system or _current_unit_system(), pressure_unit=pressure_unit or _current_tire_pressure_unit(unit_system=unit_system or _current_unit_system()))
    return str(value)


def build_v22_domain_apply_payload(domain: str, proposals: list[dict], values_by_proposal: dict | None) -> dict:
    payload = {}
    for proposal in list(proposals or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        domain_payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
        selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
        editable_inputs = dict(dict(values_by_proposal or {}).get(proposal_id) or {})
        payload[proposal_id] = sanitize_domain_inputs(
            domain,
            proposal_type,
            selection_mode,
            editable_inputs,
        )
    return payload


def clear_v22_domain_widget_state(session_state, domain: str, proposal_id: str, allowed_field_keys=None) -> None:
    allowed = None if allowed_field_keys is None else {str(field_key) for field_key in allowed_field_keys}
    prefixes = [
        f"v22_input__{domain}__{proposal_id}__",
        f"v22_simple_{_simple_widget_domain_scope(domain)}__{proposal_id}__",
    ]
    for key in list(session_state.keys()):
        widget_key = str(key)
        matched_prefix = next((prefix for prefix in prefixes if widget_key.startswith(prefix)), None)
        if matched_prefix is None:
            continue
        field_key = widget_key[len(matched_prefix):]
        if allowed is not None and field_key in allowed:
            continue
        session_state.pop(key, None)


def clear_v22_correction_widget_state(session_state, domain: str | None = None) -> None:
    prefix = f"v22_correction__{domain}__" if domain else "v22_correction__"
    for key in list(session_state.keys()):
        if str(key).startswith(prefix):
            session_state.pop(key, None)


def _clear_widget_state_after_matrix_change(previous_state: dict, next_state: dict, session_state) -> None:
    previous_by_id = {str(proposal.get("proposal_id") or ""): proposal for proposal in list(previous_state.get("proposals") or [])}
    next_by_id = {str(proposal.get("proposal_id") or ""): proposal for proposal in list(next_state.get("proposals") or [])}

    for proposal_id, proposal in previous_by_id.items():
        if proposal_id in next_by_id:
            continue
        for domain in V22_PROPOSAL_DOMAINS:
            clear_v22_domain_widget_state(session_state, domain, proposal_id)
            _clear_component_lookup_draft(session_state, domain, proposal_id)

    for proposal_id, next_proposal in next_by_id.items():
        previous_proposal = previous_by_id.get(proposal_id)
        if previous_proposal is None:
            continue
        for domain in V22_PROPOSAL_DOMAINS:
            if _domain_widget_signature(previous_proposal, domain) == _domain_widget_signature(next_proposal, domain):
                continue
            payload = dict(dict(next_proposal.get("domains") or {}).get(domain) or {})
            proposal_type = str(payload.get("proposal_type") or "INHERIT")
            selection_mode = str(payload.get("selection_mode") or proposal_type)
            clear_v22_domain_widget_state(
                session_state,
                domain,
                proposal_id,
                allowed_field_keys=applicable_fields(domain, proposal_type, selection_mode),
            )
            _clear_component_lookup_draft(session_state, domain, proposal_id)


def _clear_component_lookup_draft(session_state, domain: str, proposal_id: str) -> None:
    drafts = deepcopy(dict(session_state.get(V22_COMPONENT_LOOKUP_DRAFTS_KEY) or {}))
    domain_drafts = dict(drafts.get(str(domain or "")) or {})
    domain_drafts.pop(str(proposal_id or ""), None)
    if domain_drafts:
        drafts[str(domain or "")] = domain_drafts
    else:
        drafts.pop(str(domain or ""), None)
    session_state[V22_COMPONENT_LOOKUP_DRAFTS_KEY] = drafts


def _domain_widget_signature(proposal: dict, domain: str) -> tuple[str, str]:
    payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
    proposal_type = str(payload.get("proposal_type") or "INHERIT")
    selection_mode = str(payload.get("selection_mode") or proposal_type)
    return (proposal_type, selection_mode)


def _render_applied_input_debug(state: dict, domain: str) -> None:
    draft = build_v22_canonical_request_draft(state)
    proposals_by_id = {
        str(proposal.get("proposal_id") or ""): proposal
        for proposal in list(normalize_v22_state(state).get("proposals") or [])
    }
    with st.expander("Applied Input Debug", expanded=False):
        for proposal in list(draft.get("proposals") or []):
            proposal_id = str(proposal.get("proposal_id") or "")
            request = dict(dict(proposal.get("domain_requests") or {}).get(domain) or {})
            applied_inputs = dict(dict(dict(proposals_by_id.get(proposal_id) or {}).get("inputs") or {}).get(domain) or {})
            st.markdown(f"**{proposal_id}**")
            st.json(
                {
                    "proposal_id": proposal_id,
                    "domain": domain,
                    "applied_inputs": applied_inputs,
                    "canonical_raw_values": dict(request.get("raw_values") or {}),
                    "canonical_proposal_details_seed": dict(request.get("proposal_details_seed") or {}),
                }
            )


def _baseline_compact_summary_frame(state: dict) -> pd.DataFrame:
    baseline = dict(state.get("baseline") or {})
    printed = dict(baseline.get("printed") or {})
    effective = dict(baseline.get("effective") or {})
    mass_review = resolve_v22_baseline_mass_review(state)
    unit_system = _current_unit_system()
    rows = [
        {"Field": "Loaded baseline ID", "Printed": _display_editor_value(printed.get("selected_baseline_vde_id")), "Effective": _display_editor_value(effective.get("selected_baseline_vde_id"))},
        {"Field": "Legislation", "Printed": _display_editor_value(printed.get("legislation")), "Effective": _display_editor_value(effective.get("legislation"))},
        {"Field": "Vehicle", "Printed": f"{printed.get('make') or ''} {printed.get('model') or ''} {printed.get('year') or ''}".strip(), "Effective": f"{effective.get('make') or ''} {effective.get('model') or ''} {effective.get('year') or ''}".strip()},
        {"Field": "Cycle", "Printed": _display_editor_value(printed.get("cycle_name")), "Effective": _display_editor_value(effective.get("cycle_name"))},
        {"Field": _display_column_label("Mass [kg]", "mass_kg", unit_system), "Printed": format_display_value_for_field("mass_kg", printed.get("mass_kg"), unit_system, unavailable=""), "Effective": format_display_value_for_field("mass_kg", effective.get("mass_kg"), unit_system, unavailable="")},
        {"Field": _display_column_label("Test mass [kg]", "test_mass_kg", unit_system), "Printed": format_display_value_for_field("test_mass_kg", printed.get("test_mass_kg"), unit_system, unavailable=""), "Effective": format_display_value_for_field("test_mass_kg", effective.get("test_mass_kg"), unit_system, unavailable="")},
        {"Field": _display_column_label("EPA ETW / TWC [kg]", "inertia_class", unit_system), "Printed": format_display_value_for_field("inertia_class", printed.get("inertia_class"), unit_system, unavailable=""), "Effective": format_display_value_for_field("inertia_class", effective.get("inertia_class"), unit_system, unavailable="")},
        {"Field": _display_column_label("CdA [m^2]", "CdA", unit_system), "Printed": format_display_value_for_field("CdA", printed.get("cda_m2"), unit_system, unavailable=""), "Effective": format_display_value_for_field("CdA", effective.get("cda_m2"), unit_system, unavailable="")},
        {"Field": "ABC_TOTAL", "Printed": _display_abc_triplet(printed, unit_system), "Effective": _display_abc_triplet(effective, unit_system)},
    ]
    if mass_review:
        rows.extend(
            [
                {"Field": "Mass review status", "Printed": "", "Effective": _display_editor_value(mass_review.get("baseline_mass_review_status"))},
                {"Field": _display_column_label("Suggested EPA ETW / TWC [kg]", "inertia_class", unit_system), "Printed": "", "Effective": format_display_value_for_field("inertia_class", mass_review.get("baseline_mass_suggested_inertia_class"), unit_system, unavailable="")},
                {"Field": "Suggested TWC interval", "Printed": "", "Effective": format_display_value_for_field("baseline_mass_target_twc_interval", mass_review.get("baseline_mass_target_twc_interval"), unit_system, unavailable="")},
                {"Field": "Mass review notes", "Printed": "", "Effective": _display_editor_value(mass_review.get("baseline_mass_review_notes"))},
            ]
        )
    return pd.DataFrame(rows)


def _render_correction_form(state: dict) -> None:
    baseline = dict(state.get("baseline") or {})
    printed = dict(baseline.get("printed") or {})
    corrections = dict(baseline.get("corrections") or {})
    effective = resolve_v22_effective_baseline(state)
    rows = [
        {
            "Domain": domain,
            "Field": field,
            "Unit": unit,
            "Baseline / Printed": _display_editor_value(printed.get(field)),
            "Baseline Correction": _display_editor_value(corrections.get(field)),
            "Effective": _display_editor_value(effective.get(field)),
        }
        for domain, field, unit in BASELINE_FIELD_META
    ]
    with st.form("v22_baseline_corrections_form"):
        disposition = st.selectbox(
            "Correction usage",
            ["request_only", "save_as_new_baseline"],
            index=["request_only", "save_as_new_baseline"].index(str(baseline.get("correction_disposition") or "request_only")),
            format_func=lambda value: "Use only in this request" if value == "request_only" else "Save corrected baseline as a new VDE line",
            key="v22_correction_disposition",
        )
        edited = st.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            disabled=["Domain", "Field", "Unit", "Baseline / Printed", "Effective"],
            key="v22_corrections_editor",
        )
        submitted = st.form_submit_button("Apply Baseline Corrections")
    if submitted:
        next_corrections = {
            str(row.get("Field")): _parse_editor_value(row.get("Baseline Correction"))
            for row in edited.to_dict("records")
            if not is_blank(row.get("Baseline Correction"))
        }
        next_state = apply_v22_corrections(state, next_corrections)
        next_state["baseline"]["correction_disposition"] = disposition
        next_state["baseline"]["effective"] = resolve_v22_effective_baseline(next_state)
        st.session_state[V22_SESSION_KEY] = next_state
        st.rerun()

    mass_review = resolve_v22_baseline_mass_review({"baseline": {"effective": effective}})
    if mass_review:
        st.markdown("**EPA Mass Review**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": "Status", "Value": _display_editor_value(mass_review.get("baseline_mass_review_status"))},
                    {"Field": "Suggested EPA ETW / TWC", "Value": _display_editor_value(mass_review.get("baseline_mass_suggested_inertia_class"))},
                    {"Field": "Suggested TWC interval", "Value": _display_editor_value(mass_review.get("baseline_mass_target_twc_interval"))},
                    {"Field": "Notes", "Value": _display_editor_value(mass_review.get("baseline_mass_review_notes"))},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Effective Baseline**")
    st.dataframe(_snapshot_frame(dict(state.get("baseline", {}).get("effective") or {})), use_container_width=True, hide_index=True)


def _snapshot_frame(snapshot: dict) -> pd.DataFrame:
    rows = [{"Field": field, "Value": _display_editor_value(snapshot.get(field))} for field in V22_BASELINE_FIELDS if field in snapshot]
    return pd.DataFrame(rows)


def _render_correction_widget(domain: str, field_key: str, value, cell):
    key = f"v22_correction__{domain}__{field_key}"
    unit_system = _current_unit_system()
    pressure_unit = _current_tire_pressure_unit(unit_system=unit_system)
    meta = field_meta(field_key)
    if meta.get("kind") == "number":
        display_value = to_display_field_value(field_key, value, unit_system, pressure_unit=pressure_unit)
        _prime_widget_value(key, None if is_blank(display_value) else float(display_value))
        raw_value = cell.number_input(
            meta.get("label") or field_key,
            key=key,
            value=None if is_blank(display_value) else float(display_value),
            min_value=float(meta["min"]) if meta.get("min") is not None else None,
            max_value=float(meta["max"]) if meta.get("max") is not None else None,
            step=display_step_for_field(field_key, meta.get("step"), unit_system, pressure_unit=pressure_unit),
            format=display_format_for_field(field_key, meta.get("format"), unit_system, pressure_unit=pressure_unit),
            placeholder="-",
            label_visibility="collapsed",
        )
        return to_canonical_field_value(field_key, raw_value, unit_system, pressure_unit=pressure_unit)
    display_value = _display_editor_value(value)
    _prime_widget_value(key, display_value)
    raw = cell.text_input(
        meta.get("label") or field_key,
        key=key,
        value=display_value,
        label_visibility="collapsed",
    )
    return _parse_editor_value(raw)


def _input_application_rows(state: dict) -> tuple[list[dict], list[dict]]:
    not_applied = []
    incomplete = []
    domain_input_state = dict(state.get("domain_input_state") or {})
    for proposal in list(state.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        for domain in V22_PROPOSAL_DOMAINS:
            payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
            proposal_type = str(payload.get("proposal_type") or "INHERIT")
            selection_mode = str(payload.get("selection_mode") or proposal_type)
            if proposal_type == "INHERIT" or proposal_is_not_used(proposal_type, selection_mode):
                continue
            domain_state = dict(domain_input_state.get(domain) or {})
            proposal_status = dict(dict(domain_state.get("proposal_statuses") or {}).get(proposal_id) or {})
            status = str(proposal_status.get("status") or "not_configured")
            row = {
                "Proposal": proposal_display_label(state, proposal),
                "Domain": DOMAIN_LABELS[domain],
                "Type": selection_mode,
                "Status": proposal_status_label(proposal_status),
            }
            if status == "applied_incomplete":
                row["Issue"] = friendly_message(" | ".join(list(proposal_status.get("issues") or [])))
                incomplete.append(row)
            elif status == "not_configured" or str(domain_state.get("status") or "") == "stale_after_matrix_change":
                not_applied.append(row)
    return not_applied, incomplete


def _invalid_walk_from_rows(state: dict) -> list[str]:
    invalid = []
    for proposal in list(state.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        if str(proposal.get("walk_from") or "baseline") not in allowed_walk_from_options(state, proposal_id):
            invalid.append(proposal_display_label(state, proposal))
    return invalid


def _selection_mode_for_domain(proposal: dict, domain: str) -> str:
    return str(dict(dict(proposal.get("domains") or {}).get(domain) or {}).get("selection_mode") or "Inherit")


def _walk_label(state: dict):
    return lambda value: walk_from_display_label(state, value)


def _filter_options(df: pd.DataFrame, column: str) -> list:
    if column not in df:
        return ["(all)"]
    values = sorted(value for value in df[column].dropna().unique().tolist() if str(value).strip())
    return ["(all)", *values]


def _current_baseline_filter_values(df: pd.DataFrame) -> tuple[str, str, str, str]:
    legislation_options = _filter_options(df, "Legislation")
    make_options = _filter_options(df, "Make")
    legislation = str(st.session_state.get("v22_filter_legislation") or "(all)")
    make = str(st.session_state.get("v22_filter_make") or "(all)")
    if legislation not in legislation_options:
        legislation = "(all)"
        st.session_state["v22_filter_legislation"] = legislation
    if make not in make_options:
        make = "(all)"
        st.session_state["v22_filter_make"] = make
    year = str(st.session_state.get("v22_filter_year") or "")
    model_text = str(st.session_state.get("v22_filter_model") or "")
    return legislation, make, year, model_text


def _apply_summary_filters(df: pd.DataFrame, *, legislation: str, make: str, year: str, model_text: str) -> pd.DataFrame:
    out = df.copy()
    if legislation != "(all)":
        out = out[out["Legislation"] == legislation]
    if make != "(all)":
        out = out[out["Make"] == make]
    if str(year or "").strip():
        out = out[out["Year"].astype(str) == str(year).strip()]
    if str(model_text or "").strip():
        out = out[out["Model"].astype(str).str.contains(str(model_text).strip(), case=False, na=False)]
    return out


def _current_candidate_selection(df: pd.DataFrame):
    if df.empty or "VDE ID" not in df:
        return None
    options = df["VDE ID"].tolist()
    current = st.session_state.get("v22_baseline_selector")
    if current in options:
        return current
    st.session_state["v22_baseline_selector"] = options[0]
    return options[0]


def _baseline_option_label(df: pd.DataFrame, value) -> str:
    row = df[df["VDE ID"] == value].head(1)
    if row.empty:
        return str(value)
    item = row.iloc[0].to_dict()
    year = item.get("Year")
    year_label = f"MY{year}" if not is_blank(year) else ""
    parts = [
        f"VDE #{item.get('VDE ID')}",
        " ".join(part for part in [str(item.get("Make") or "").strip(), str(item.get("Model") or "").strip()] if part).strip(),
        year_label,
        str(item.get("Legislation") or "").strip(),
        str(item.get("Cycle") or "").strip(),
    ]
    return " · ".join(part for part in parts if part)


def _abc_label(row: dict) -> str:
    values = [row.get("coast_A_N") or row.get("A"), row.get("coast_B_N_per_kph") or row.get("B"), row.get("coast_C_N_per_kph2") or row.get("C")]
    return " / ".join(_display_editor_value(value) for value in values)


def _display_editor_value(value) -> str:
    if is_blank(value):
        return ""
    return str(value)


def _display_domain_cell(value, field_key: str) -> str:
    if is_blank(value):
        return EM_DASH
    if field_key == "transmission_application_mode":
        return _format_select_option(field_key, value)
    if quantity_kind_for_field(field_key):
        return _trim_display_numeric_text(
            format_display_value_for_field(
                field_key,
                value,
                _current_unit_system(),
                unavailable=EM_DASH,
                pressure_unit=_current_tire_pressure_unit(),
            )
        )
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text
    return str(value)


def _parse_editor_value(value):
    if is_blank(value):
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        try:
            return float(text)
        except Exception:
            return value.strip()
    return value


def _trim_display_numeric_text(value) -> str:
    text = str(value)
    try:
        if any(token in text for token in ("(", "]", " ", "|", "=")):
            return text
        numeric = float(text)
        return f"{numeric:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return text
