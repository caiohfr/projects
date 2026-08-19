from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from typing import Any

from src.vde_core.component_repositories import COMPONENT_PROVENANCE_FIELDS, create_component
from src.vde_core.db import DB_PATH, ensure_db, table_columns
from src.vde_core.services import autoresolve_test_mass
from src.vde_core.vde_request_contract import VDE_REQUEST_SCHEMA_VERSION, is_blank


SAVE_MODE_VALID_ONLY = "Save valid only"
SAVE_MODE_SELECTED = "Save selected"
SAVE_MODE_REQUIRE_ALL_VALID = "Require all valid"
SAVE_MODES = (
    SAVE_MODE_VALID_ONLY,
    SAVE_MODE_SELECTED,
    SAVE_MODE_REQUIRE_ALL_VALID,
)

BASELINE_DB_FIELD_MAP = {
    "notes": "notes",
    "legislation": "legislation",
    "category": "category",
    "make": "make",
    "model": "model",
    "year": "year",
    "cycle_name": "cycle_name",
    "cycle_source": "cycle_source",
    "mass_kg": "mass_kg",
    "test_mass_kg": "test_mass_kg",
    "test_mass_low_kg": "test_mass_low_kg",
    "test_mass_high_kg": "test_mass_high_kg",
    "test_mass_basis": "test_mass_basis",
    "payload_kg": "payload_kg",
    "weight_dist_fr_pct": "weight_dist_fr_pct",
    "inertia_class": "inertia_class",
    "cda_m2": "cda_m2",
    "A": "coast_A_N",
    "B": "coast_B_N_per_kph",
    "C": "coast_C_N_per_kph2",
    "rrc_N_per_kN": "rrc_N_per_kN",
    "smerf": "smerf",
    "front_pressure_psi": "front_pressure_psi",
    "rear_pressure_psi": "rear_pressure_psi",
    "front_tire_id": "front_tire_id",
    "rear_tire_id": "rear_tire_id",
    "tire_load_mass_basis": "tire_load_mass_basis",
    "tire_A_final": "tire_A_final",
    "tire_B_final": "tire_B_final",
    "tire_C_final": "tire_C_final",
    "tire_calc_source": "tire_calc_source",
    "tire_calc_notes": "tire_calc_notes",
    "trans_A_coef_N": "trans_A_coef_N",
    "trans_B_coef_Npkph": "trans_B_coef_Npkph",
    "trans_C_coef_Npkph2": "trans_C_coef_Npkph2",
    "brake_A_coef_N": "brake_A_coef_N",
    "brake_B_Npkph": "brake_B_coef_Npkph",
    "brake_C_coef_Npkph2": "brake_C_coef_Npkph2",
    "parasitic_A_coef_N": "parasitic_A_coef_N",
    "parasitic_B_Npkph": "parasitic_B_coef_Npkph",
    "parasitic_C_coef_Npkph2": "parasitic_C_coef_Npkph2",
    "wltp_category": "wltp_category",
    "GVWR_kg": "gvwr_kg",
    "GCWR_kg": "gcwr_kg",
    "trailer_weight_kg": "trailer_mass_kg",
    "trailer_code": "trailer_code",
    "trailer_A": "trailer_A_coef_N",
    "trailer_B": "trailer_B_coef_Npkph",
    "trailer_C": "trailer_C_coef_Npkph2",
}

MOCK_COMPONENT_SNAPSHOT_FIELDS = {
    "transmission": ("new_trans_A", "new_trans_B", "new_trans_C", "loss_pct"),
    "brake": ("brake_A", "brake_B", "brake_C"),
    "axle_hubs": ("axle_hub_A", "axle_hub_B", "axle_hub_C"),
    "parasitic": ("parasitic_A", "parasitic_B", "parasitic_C"),
}


def _issue(code: str, severity: str, message: str, **extra) -> dict:
    payload = {"code": code, "severity": severity, "message": message}
    payload.update({key: value for key, value in extra.items() if value not in (None, "")})
    return payload


def _clean_text(value) -> str:
    return str(value or "").strip()


def _normalize_save_mode(save_mode: str | None) -> str:
    text = _clean_text(save_mode)
    if text in SAVE_MODES:
        return text
    lowered = text.lower()
    mapping = {
        "save valid only": SAVE_MODE_VALID_ONLY,
        "save_valid_only": SAVE_MODE_VALID_ONLY,
        "valid_only": SAVE_MODE_VALID_ONLY,
        "save selected": SAVE_MODE_SELECTED,
        "save_selected": SAVE_MODE_SELECTED,
        "selected": SAVE_MODE_SELECTED,
        "require all valid": SAVE_MODE_REQUIRE_ALL_VALID,
        "require_all_valid": SAVE_MODE_REQUIRE_ALL_VALID,
        "all_valid": SAVE_MODE_REQUIRE_ALL_VALID,
    }
    return mapping.get(lowered, SAVE_MODE_VALID_ONLY)


def _total_vde_mj_per_km(proposal_result: dict | None):
    total = dict(dict(proposal_result or {}).get("vde_results") or {}).get("total") or {}
    value = total.get("mj_per_km")
    return None if is_blank(value) else float(value)


def _net_vde_mj_per_km(proposal_result: dict | None):
    net = dict(dict(proposal_result or {}).get("vde_results") or {}).get("net") or {}
    value = net.get("mj_per_km")
    return None if is_blank(value) else float(value)


def _rollup_status(statuses: list[str], default: str = "OK") -> str:
    order = {"OK": 0, "Review": 1, "Missing": 2, "Invalid": 3, "Blocked": 4}
    cleaned = [str(item or "").strip().title() for item in statuses if str(item or "").strip()]
    if not cleaned:
        return default
    return max(cleaned, key=lambda item: order.get(item, -1))


def _request_source(state: dict | None) -> dict:
    data = dict(state or {})
    return dict(data.get("vde_request_source") or dict(data.get("vde_request_import") or {}).get("source") or {})


def _import_columns(state: dict | None) -> dict:
    data = dict(state or {})
    return dict(dict(data.get("vde_request_import") or {}).get("columns") or {})


def _proposal_column_state(state: dict | None, proposal_id: str) -> dict:
    data = dict(state or {})
    columns = dict(data.get("columns") or {})
    return dict(columns.get(proposal_id) or {})


def _proposal_import_meta(state: dict | None, proposal_id: str) -> dict:
    return dict(_import_columns(state).get(proposal_id) or {})


def _proposal_user_label(state: dict | None, proposal_id: str, proposal_result: dict) -> str:
    column = _proposal_column_state(state, proposal_id)
    direct = dict(column.get("direct") or {})
    return _clean_text(direct.get("description"))


def _proposal_user_notes(state: dict | None, proposal_id: str) -> str:
    column = _proposal_column_state(state, proposal_id)
    direct = dict(column.get("direct") or {})
    return _clean_text(direct.get("description"))


def _proposal_domain_types(proposal_result: dict | None) -> dict[str, str]:
    result = {}
    for domain_key, payload in dict(dict(proposal_result or {}).get("domain_results") or {}).items():
        proposal_type = _clean_text(dict(payload or {}).get("proposal_type") or "INHERIT") or "INHERIT"
        result[str(domain_key)] = proposal_type
    return result


def _domain_summary_label(domain_key: str, proposal_type: str, proposal_result: dict) -> str:
    proposal_type = _clean_text(proposal_type).upper()
    if proposal_type in {"INHERIT", ""}:
        return ""
    if domain_key == "mass":
        source_mass = dict(dict(proposal_result or {}).get("source_snapshot") or {}).get("mass_kg")
        resolved_mass = dict(dict(proposal_result or {}).get("resolved_snapshot") or {}).get("resolved_mass_setup") or {}
        target_mass = dict(dict(proposal_result or {}).get("resolved_snapshot") or {}).get("mass_kg") or resolved_mass.get("resolved_mass_used_kg") or resolved_mass.get("test_mass_kg")
        if source_mass not in (None, "") and target_mass not in (None, ""):
            delta = float(target_mass) - float(source_mass)
            if abs(delta) < 0.5:
                return "Mass"
            return f"Mass {delta:+.0f} kg"
        return "Mass"
    if domain_key == "aero":
        return "CdA Delta" if "DELTA" in proposal_type else "CdA Absolute"
    if domain_key == "tire":
        return "Tire Lookup" if "LOOKUP" in proposal_type else "Tire"
    if domain_key == "transmission":
        if "LOSS_PCT" in proposal_type:
            return "Transmission coastdown share"
        return "Trans Delta" if "DELTA" in proposal_type else "Trans Absolute"
    if domain_key == "brake":
        if "RESIDUAL" in proposal_type:
            return "Brake Residual"
        return "Brake Delta" if "DELTA" in proposal_type else "Brake Absolute"
    if domain_key == "axle_hubs":
        return "Axle Delta" if "DELTA" in proposal_type else "Axle Absolute"
    if domain_key == "parasitic":
        return "Parasitic Delta" if "DELTA" in proposal_type else "Parasitic Absolute"
    return domain_key.replace("_", " ").title()


def generate_auto_proposal_name(proposal_result: dict | None) -> str:
    proposal = dict(proposal_result or {})
    pieces = []
    for domain_key, proposal_type in _proposal_domain_types(proposal).items():
        label = _domain_summary_label(domain_key, proposal_type, proposal)
        if label:
            pieces.append(label)
    if pieces:
        return " + ".join(pieces[:3])[:120]
    display_index = proposal.get("display_index")
    if display_index not in (None, ""):
        return f"Requested #{int(display_index)}"
    return _clean_text(proposal.get("source_column")) or _clean_text(proposal.get("proposal_id")) or "Requested proposal"


def _baseline_reference_snapshot(resolution_result: dict | None) -> dict:
    resolved = dict(dict(resolution_result or {}).get("resolved_columns") or {}).get("baseline") or {}
    baseline_effective = dict(dict(resolution_result or {}).get("baseline") or {}).get("effective") or {}
    snapshot = dict(baseline_effective)
    snapshot.update({key: value for key, value in resolved.items() if value not in (None, "")})
    return snapshot


def _proposal_row_payload(
    resolution_result: dict | None,
    proposal_result: dict | None,
    request_state: dict | None,
    *,
    final_name: str,
    note_text: str,
) -> dict:
    resolution = dict(resolution_result or {})
    proposal = dict(proposal_result or {})
    snapshot = dict(proposal.get("resolved_snapshot") or {})
    mass_setup = dict(snapshot.get("resolved_mass_setup") or {})
    baseline_ref = _baseline_reference_snapshot(resolution)
    transmission = dict(snapshot.get("transmission_losses") or {})
    total_abc = dict(proposal.get("abc_total") or {})
    vde_total = dict(dict(proposal.get("vde_results") or {}).get("total") or {})
    vde_net = dict(dict(proposal.get("vde_results") or {}).get("net") or {})
    request_source = _request_source(request_state)

    payload = {
        "legislation": snapshot.get("legislation") or baseline_ref.get("legislation"),
        "category": snapshot.get("category") or baseline_ref.get("category"),
        "make": snapshot.get("make") or baseline_ref.get("make") or "NEW TEST",
        "model": snapshot.get("model") or baseline_ref.get("model") or final_name or proposal.get("proposal_id") or "Requested proposal",
        "year": snapshot.get("year") or baseline_ref.get("year") or datetime.now(timezone.utc).year,
        "notes": note_text or final_name,
        "mass_kg": snapshot.get("mass_kg"),
        "test_mass_kg": mass_setup.get("test_mass_kg") or snapshot.get("test_mass_kg"),
        "test_mass_low_kg": mass_setup.get("test_mass_low_kg") or snapshot.get("test_mass_low_kg"),
        "test_mass_high_kg": mass_setup.get("test_mass_high_kg") or snapshot.get("test_mass_high_kg"),
        "test_mass_basis": mass_setup.get("test_mass_basis") or snapshot.get("test_mass_basis"),
        "inertia_class": mass_setup.get("inertia_class") or snapshot.get("inertia_class"),
        "weight_dist_fr_pct": mass_setup.get("weight_dist_fr_pct") or snapshot.get("weight_dist_fr_pct"),
        "payload_kg": mass_setup.get("payload_kg") or snapshot.get("payload_kg"),
        "wltp_category": snapshot.get("wltp_category"),
        "cda_m2": snapshot.get("CdA"),
        "coast_A_N": total_abc.get("A"),
        "coast_B_N_per_kph": total_abc.get("B"),
        "coast_C_N_per_kph2": total_abc.get("C"),
        "vde_total_mj_per_km": vde_total.get("mj_per_km"),
        "vde_net_mj_per_km": vde_net.get("mj_per_km"),
        "cycle_name": snapshot.get("cycle_name") or baseline_ref.get("cycle_name"),
        "cycle_source": snapshot.get("cycle_source") or baseline_ref.get("cycle_source") or "request_preview",
        "vde_id_parent": baseline_ref.get("selected_baseline_vde_id"),
        "baseline_A_N": dict(baseline_ref.get("initial_abc_total") or {}).get("A") or baseline_ref.get("A"),
        "baseline_B_N_per_kph": dict(baseline_ref.get("initial_abc_total") or {}).get("B") or baseline_ref.get("B"),
        "baseline_C_N_per_kph2": dict(baseline_ref.get("initial_abc_total") or {}).get("C") or baseline_ref.get("C"),
        "baseline_mass_kg": baseline_ref.get("mass_kg"),
        "front_tire_id": snapshot.get("front_tire_id"),
        "rear_tire_id": snapshot.get("rear_tire_id"),
        "tire_A_final": snapshot.get("tire_A_final"),
        "tire_B_final": snapshot.get("tire_B_final"),
        "tire_C_final": snapshot.get("tire_C_final"),
        "tire_calc_source": snapshot.get("tire_calc_source"),
        "tire_load_mass_basis": snapshot.get("tire_load_mass_basis"),
        "tire_improvement_pct": snapshot.get("tire_improvement_pct"),
        "rrc_N_per_kN": snapshot.get("rrc_N_per_kN"),
        "smerf": snapshot.get("smerf"),
        "front_pressure_psi": snapshot.get("front_pressure_psi"),
        "rear_pressure_psi": snapshot.get("rear_pressure_psi"),
        "trans_A_coef_N": transmission.get("A_TRANS") or dict(transmission.get("abc") or {}).get("A"),
        "trans_B_coef_Npkph": transmission.get("B_TRANS") or dict(transmission.get("abc") or {}).get("B"),
        "trans_C_coef_Npkph2": transmission.get("C_TRANS") or dict(transmission.get("abc") or {}).get("C"),
        "brake_A_coef_N": snapshot.get("brake_A"),
        "brake_B_coef_Npkph": snapshot.get("brake_B"),
        "brake_C_coef_Npkph2": snapshot.get("brake_C"),
        "parasitic_A_coef_N": snapshot.get("parasitic_A"),
        "parasitic_B_coef_Npkph": snapshot.get("parasitic_B"),
        "parasitic_C_coef_Npkph2": snapshot.get("parasitic_C"),
        "gvwr_kg": snapshot.get("GVWR_kg"),
        "gcwr_kg": snapshot.get("GCWR_kg"),
        "trailer_mass_kg": snapshot.get("trailer_weight_kg"),
        "trailer_code": snapshot.get("trailer_code"),
        "trailer_roadload_source": snapshot.get("trailer_roadload_source") or request_source.get("filename"),
        "trailer_A_coef_N": snapshot.get("trailer_A"),
        "trailer_B_coef_Npkph": snapshot.get("trailer_B"),
        "trailer_C_coef_Npkph2": snapshot.get("trailer_C"),
        "mass_rule_status": mass_setup.get("mass_rule_status"),
        "mass_rule_notes": mass_setup.get("mass_rule_notes"),
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _component_creation_payload(domain_key: str, component_action: dict, proposal_row: dict) -> tuple[dict | None, str | None]:
    snapshot = dict(component_action.get("component_snapshot") or {})
    required_fields = MOCK_COMPONENT_SNAPSHOT_FIELDS.get(domain_key)
    if not required_fields:
        return None, "unsupported_domain"
    values: dict[str, Any] = {}
    if domain_key == "transmission":
        values["trans_A"] = snapshot.get("new_trans_A")
        values["trans_B"] = snapshot.get("new_trans_B")
        values["trans_C"] = snapshot.get("new_trans_C")
        values["loss_pct"] = snapshot.get("loss_pct")
    elif domain_key == "brake":
        values["brake_A"] = snapshot.get("brake_A")
        values["brake_B"] = snapshot.get("brake_B")
        values["brake_C"] = snapshot.get("brake_C")
        values["residual_torque_front_nm"] = snapshot.get("residual_torque_front_nm", 0.0)
        values["residual_torque_rear_nm"] = snapshot.get("residual_torque_rear_nm", 0.0)
        values["wheel_radius_m"] = snapshot.get("wheel_radius_m", 1.0)
    elif domain_key == "axle_hubs":
        values["axle_hubs_A"] = snapshot.get("axle_hub_A")
        values["axle_hubs_B"] = snapshot.get("axle_hub_B")
        values["axle_hubs_C"] = snapshot.get("axle_hub_C")
    elif domain_key == "parasitic":
        values["parasitic_A"] = snapshot.get("parasitic_A")
        values["parasitic_B"] = snapshot.get("parasitic_B")
        values["parasitic_C"] = snapshot.get("parasitic_C")
    if any(is_blank(values.get(field_name)) for field_name in values if field_name not in {"loss_pct", "residual_torque_front_nm", "residual_torque_rear_nm", "wheel_radius_m"}):
        return None, "insufficient_snapshot"
    proposal_id = _clean_text(proposal_row.get("proposal_id"))
    digest = hashlib.sha1(f"{domain_key}|{proposal_id}|{json.dumps(values, sort_keys=True, default=str)}".encode("utf-8")).hexdigest()[:10].upper()
    payload = {
        "component_id": f"{domain_key.upper()}-USER-{digest}",
        "component_name": _clean_text(proposal_row.get("final_name")) or f"{domain_key.title()} component",
        "status": "user_created",
        "source": "manual_request",
        "notes": f"Created from {proposal_row.get('source_column') or proposal_id}",
    }
    for field_name in COMPONENT_PROVENANCE_FIELDS:
        if not is_blank(snapshot.get(field_name)):
            payload[field_name] = snapshot.get(field_name)
    payload.update(values)
    return payload, None


def _component_plan_rows(proposal_row: dict) -> list[dict]:
    rows = []
    for action in list(proposal_row.get("component_actions") or []):
        payload = dict(action or {})
        domain_key = _clean_text(payload.get("domain"))
        create_payload, create_error = _component_creation_payload(domain_key, payload, proposal_row)
        row = {
            "proposal_id": proposal_row.get("proposal_id"),
            "source_column": proposal_row.get("source_column"),
            "domain": domain_key,
            "action": payload.get("action"),
            "component_id": payload.get("component_id"),
            "requires_confirmation": bool(payload.get("requires_confirmation")),
            "component_snapshot": deepcopy(payload.get("component_snapshot")),
            "issues": deepcopy(list(payload.get("issues") or [])),
            "create_payload": create_payload,
            "creation_supported": create_payload is not None,
            "creation_error": create_error,
        }
        rows.append(row)
    return rows


def _combine_notes(final_name: str, user_notes: str, audit_lines: list[str]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for item in [final_name, user_notes, *audit_lines]:
        text = _clean_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        lines.append(text)
    return "\n".join(lines)


def _audit_note_lines(
    resolution_result: dict | None,
    proposal_result: dict | None,
    request_state: dict | None,
    *,
    final_name: str,
    baseline_update_rows: list[dict],
    confirmed_review: bool,
) -> list[str]:
    resolution = dict(resolution_result or {})
    proposal = dict(proposal_result or {})
    baseline = dict(resolution.get("baseline") or {})
    request_source = _request_source(request_state)
    component_actions = list(proposal.get("component_actions") or [])
    correction_keys = [row["field_key"] for row in baseline_update_rows if row.get("selected")]
    domain_types = _proposal_domain_types(proposal)
    explicit_not_used = sorted(
        domain_key
        for domain_key, proposal_type in domain_types.items()
        if proposal_type.endswith("NOT_USED") or proposal_type in {"TIRE_METADATA_ONLY", "TRANS_LOSS_NOT_AVAILABLE"}
    )
    component_bits = []
    for action in component_actions:
        payload = dict(action or {})
        domain = _clean_text(payload.get("domain"))
        act = _clean_text(payload.get("action"))
        comp = _clean_text(payload.get("component_id"))
        text = f"{domain}:{act}"
        if comp:
            text += f"({comp})"
        component_bits.append(text)
    domain_bits = [f"{key}={value}" for key, value in domain_types.items() if _clean_text(value) and value != "INHERIT"]
    review_codes = [dict(item).get("code") for item in list(proposal.get("issues") or []) if str(dict(item).get("severity") or "").strip().lower() == "review"]
    lines = [
        f"Request schema {request_source.get('schema_version') or VDE_REQUEST_SCHEMA_VERSION}",
        f"Request source: {_clean_text(request_source.get('filename') or request_source.get('workbook_name') or 'manual_v21_request')}",
        f"Proposal label: {final_name}",
        f"Original source column: {_clean_text(proposal.get('source_column')) or proposal.get('proposal_id')}",
        f"Walk From: {_clean_text(dict(proposal.get('walk_from') or {}).get('label') or dict(proposal.get('walk_from') or {}).get('column_id') or 'Baseline')}",
        f"Original baseline ID: {baseline.get('effective', {}).get('selected_baseline_vde_id') or baseline.get('printed', {}).get('selected_baseline_vde_id') or '-'}",
        "Proposal types: " + (", ".join(domain_bits) if domain_bits else "inherit only"),
        "Baseline corrections used: " + (", ".join(correction_keys) if correction_keys else "none"),
        "Component actions: " + (", ".join(component_bits) if component_bits else "none"),
        "Explicit Not used: " + (", ".join(explicit_not_used) if explicit_not_used else "none"),
        f"Status at save: {_clean_text(proposal.get('status')) or 'OK'}",
        "Review issues confirmed: " + (", ".join(str(code) for code in review_codes if code) if confirmed_review else "none"),
    ]
    return lines


def _baseline_update_rows(resolution_result: dict | None, baseline_update_choices: dict | None = None) -> list[dict]:
    baseline = dict(dict(resolution_result or {}).get("baseline") or {})
    printed = dict(baseline.get("printed") or {})
    correction = dict(baseline.get("correction") or {})
    effective = dict(baseline.get("effective") or {})
    choices = dict(baseline_update_choices or {})
    rows = []
    seen_db_fields: set[str] = set()
    for field_key in list(baseline.get("corrected_fields") or []):
        db_field = BASELINE_DB_FIELD_MAP.get(field_key)
        supported = bool(db_field)
        duplicate_alias = bool(db_field and db_field in seen_db_fields)
        if db_field:
            seen_db_fields.add(db_field)
        default_selected = supported and not duplicate_alias
        selected = bool(choices.get(field_key, default_selected)) if supported and not duplicate_alias else False
        rows.append(
            {
                "field_key": field_key,
                "db_field": db_field,
                "printed": printed.get(field_key),
                "correction": correction.get(field_key),
                "effective": effective.get(field_key),
                "supported": supported,
                "duplicate_alias": duplicate_alias,
                "default_selected": default_selected,
                "selected": selected,
            }
        )
    return rows


def _proposal_plan_row(
    resolution_result: dict,
    proposal_result: dict,
    request_state: dict | None,
    baseline_update_rows: list[dict],
    review_confirmations: dict | None,
) -> dict:
    proposal = dict(proposal_result or {})
    proposal_id = _clean_text(proposal.get("proposal_id"))
    status_raw = _clean_text(proposal.get("status")) or "OK"
    normalized_status = {
        "ok": "OK",
        "review": "Review",
        "missing": "Missing",
        "invalid": "Invalid",
        "blocked": "Blocked",
    }
    status = normalized_status.get(status_raw.lower(), status_raw)
    total_vde = _total_vde_mj_per_km(proposal)
    net_vde = _net_vde_mj_per_km(proposal)
    review_required = status == "Review"
    review_confirmed = bool(dict(review_confirmations or {}).get(proposal_id))
    vde_available = total_vde is not None
    base_eligible = status == "OK" and vde_available
    review_eligible = status == "Review" and vde_available and review_confirmed
    eligible = base_eligible or review_eligible
    final_name = _proposal_user_label(request_state, proposal_id, proposal)
    auto_name = generate_auto_proposal_name(proposal)
    if not _clean_text(final_name):
        final_name = auto_name
    user_notes = _proposal_user_notes(request_state, proposal_id)
    notes = _combine_notes(
        final_name,
        user_notes,
        _audit_note_lines(
            resolution_result,
            proposal,
            request_state,
            final_name=final_name,
            baseline_update_rows=baseline_update_rows,
            confirmed_review=review_confirmed,
        ),
    )
    ineligible_reasons = []
    if status in {"Missing", "Invalid", "Blocked"}:
        ineligible_reasons.append(f"status_{status.lower()}")
    if not vde_available:
        ineligible_reasons.append("missing_vde_total")
    if review_required and not review_confirmed:
        ineligible_reasons.append("review_confirmation_required")

    row = {
        "proposal_id": proposal_id,
        "display_index": proposal.get("display_index"),
        "source_column": proposal.get("source_column"),
        "status": status,
        "walk_from": dict(proposal.get("walk_from") or {}).get("label") or dict(proposal.get("walk_from") or {}).get("column_id") or "Baseline",
        "review_required": review_required,
        "review_confirmed": review_confirmed,
        "vde_total_mj_per_km": total_vde,
        "vde_net_mj_per_km": net_vde,
        "vde_available": vde_available,
        "eligible": eligible,
        "ineligible_reasons": ineligible_reasons,
        "final_name": final_name,
        "auto_name": auto_name,
        "user_notes": user_notes,
        "note_text": notes,
        "issues": deepcopy(list(proposal.get("issues") or [])),
        "domain_types": _proposal_domain_types(proposal),
        "component_actions": _component_plan_rows(
            {
                "proposal_id": proposal_id,
                "source_column": proposal.get("source_column"),
                "final_name": final_name,
                "component_actions": list(proposal.get("component_actions") or []),
            }
        ),
        "row_payload": _proposal_row_payload(
            resolution_result,
            proposal,
            request_state,
            final_name=final_name,
            note_text=notes,
        ),
    }
    return row


def build_vde_request_save_plan(
    resolution_result,
    selected_proposal_ids=None,
    save_mode=None,
    review_confirmations=None,
    baseline_update_choices=None,
    component_creation_confirmations=None,
    *,
    request_state=None,
    current_fingerprint: str | None = None,
    resolution_fingerprint: str | None = None,
    preview_is_stale: bool = False,
    previously_saved_proposal_ids: list[str] | None = None,
    previous_save_fingerprint: str | None = None,
) -> dict:
    resolution = deepcopy(dict(resolution_result or {}))
    review_confirmations = dict(review_confirmations or {})
    component_creation_confirmations = dict(component_creation_confirmations or {})
    normalized_mode = _normalize_save_mode(save_mode)
    selected_ids = [str(item) for item in list(selected_proposal_ids or []) if _clean_text(item)]
    baseline_rows = _baseline_update_rows(resolution, baseline_update_choices)
    proposal_rows = [
        _proposal_plan_row(resolution, proposal_result, request_state, baseline_rows, review_confirmations)
        for proposal_result in list(resolution.get("proposal_results") or [])
    ]

    blocking_issues: list[dict] = []
    warnings: list[dict] = []
    current_fp = _clean_text(current_fingerprint)
    resolved_fp = _clean_text(resolution_fingerprint)
    previous_fp = _clean_text(previous_save_fingerprint)
    saved_ids = {str(item) for item in list(previously_saved_proposal_ids or []) if _clean_text(item)}

    if not resolution:
        blocking_issues.append(_issue("missing_resolution", "blocked", "Validate & Preview must run before building a Save Plan."))
    if preview_is_stale:
        blocking_issues.append(_issue("preview_stale", "blocked", "Preview outdated — run Validate & Preview again."))
    if current_fp and resolved_fp and current_fp != resolved_fp:
        blocking_issues.append(_issue("fingerprint_mismatch", "blocked", "Current workbook fingerprint does not match the validated Preview."))

    all_ids = [row["proposal_id"] for row in proposal_rows]
    if normalized_mode == SAVE_MODE_SELECTED:
        requested_ids = selected_ids
        if not requested_ids:
            blocking_issues.append(_issue("no_selected_proposals", "blocked", "Choose at least one proposal for Save selected mode."))
    elif normalized_mode == SAVE_MODE_REQUIRE_ALL_VALID:
        requested_ids = all_ids
    else:
        requested_ids = [row["proposal_id"] for row in proposal_rows if row["eligible"]]

    plan_rows = []
    skipped_rows = []
    requested_set = set(requested_ids)

    if normalized_mode == SAVE_MODE_REQUIRE_ALL_VALID:
        ineligible = [row for row in proposal_rows if not row["eligible"]]
        if ineligible:
            blocking_issues.append(
                _issue(
                    "require_all_valid_blocked",
                    "blocked",
                    "Require all valid blocks execution until every proposal is eligible.",
                    proposal_ids=[row["proposal_id"] for row in ineligible],
                )
            )

    for row in proposal_rows:
        proposal_id = row["proposal_id"]
        requested = proposal_id in requested_set
        duplicate = bool(current_fp and previous_fp and current_fp == previous_fp and proposal_id in saved_ids)
        row["requested"] = requested
        row["already_saved"] = duplicate
        row["component_plan"] = []

        for component_row in list(row.get("component_actions") or []):
            key = f"{proposal_id}:{component_row['domain']}"
            wants_create = bool(component_creation_confirmations.get(key))
            component_row["requested_create"] = wants_create
            component_row["will_create"] = (
                requested
                and wants_create
                and component_row.get("action") == "eligible_for_new_component"
                and component_row.get("creation_supported")
            )
            row["component_plan"].append(component_row)
            if wants_create and component_row.get("action") == "eligible_for_new_component" and not component_row.get("creation_supported"):
                warnings.append(
                    _issue(
                        "component_creation_unavailable",
                        "review",
                        f"{component_row['domain']} component creation was requested but the snapshot is incomplete; the proposal will remain snapshot-only.",
                        proposal_id=proposal_id,
                        domain=component_row["domain"],
                    )
                )

        if not requested:
            skipped_rows.append({"proposal_id": proposal_id, "reason": "not_selected", "status": row["status"]})
            continue
        if duplicate:
            skipped_rows.append({"proposal_id": proposal_id, "reason": "already_saved_for_current_preview", "status": row["status"]})
            continue
        if not row["eligible"]:
            blocking_issues.append(
                _issue(
                    "proposal_not_eligible",
                    "blocked",
                    f"{proposal_id} is not eligible for save.",
                    proposal_id=proposal_id,
                    reasons=list(row["ineligible_reasons"]),
                )
            )
            skipped_rows.append({"proposal_id": proposal_id, "reason": "not_eligible", "status": row["status"]})
            continue
        plan_rows.append(row)

    baseline_update_requests = [row for row in baseline_rows if row.get("selected") and row.get("supported") and not row.get("duplicate_alias")]
    if baseline_update_requests and is_blank(_baseline_reference_snapshot(resolution).get("selected_baseline_vde_id")):
        blocking_issues.append(_issue("baseline_update_without_baseline_id", "blocked", "Baseline updates were selected but the original baseline ID is unavailable."))

    if not plan_rows and not blocking_issues:
        warnings.append(_issue("nothing_to_save", "review", "No eligible proposals remain in the current Save Plan."))

    operation_seed = {
        "fingerprint": resolved_fp or current_fp,
        "proposal_ids": [row["proposal_id"] for row in plan_rows],
        "mode": normalized_mode,
        "baseline_updates": [row["field_key"] for row in baseline_update_requests],
    }
    operation_id = "saveop_" + hashlib.sha1(json.dumps(operation_seed, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]

    plan = {
        "operation_id": operation_id,
        "save_mode": normalized_mode,
        "preview_is_stale": bool(preview_is_stale),
        "current_fingerprint": current_fp or None,
        "resolution_fingerprint": resolved_fp or None,
        "status": "blocked" if blocking_issues else ("ready" if plan_rows else "empty"),
        "can_execute": bool(plan_rows) and not blocking_issues,
        "proposals": proposal_rows,
        "selected_proposal_ids": requested_ids,
        "proposals_to_save": plan_rows,
        "skipped_proposals": skipped_rows,
        "baseline_updates": baseline_rows,
        "baseline_update_requests": baseline_update_requests,
        "component_requests": [item for row in plan_rows for item in list(row.get("component_plan") or [])],
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "already_saved_proposal_ids": sorted(saved_ids),
    }
    json.dumps(plan, default=str)
    return plan


def _insert_row_in_transaction(con, row: dict) -> int:
    payload = autoresolve_test_mass(dict(row or {}))
    columns = list(payload.keys())
    values = [payload[column] for column in columns]
    placeholders = ",".join("?" for _ in columns)
    cur = con.cursor()
    cur.execute(f"INSERT INTO vde_db ({','.join(columns)}) VALUES ({placeholders})", values)
    return int(cur.lastrowid)


def _update_row_in_transaction(con, row_id: int, updates: dict) -> None:
    payload = autoresolve_test_mass(dict(updates or {}))
    payload["updated_at"] = payload.get("updated_at") or None
    columns = [column for column in payload.keys()]
    values = [payload[column] for column in columns]
    set_clause = ", ".join(f"{column}=?" for column in columns)
    con.execute(f"UPDATE vde_db SET {set_clause} WHERE id=?", [*values, int(row_id)])


def _default_services() -> dict:
    return {
        "ensure_db": ensure_db,
        "connect_db": lambda: sqlite3.connect(str(DB_PATH), timeout=30),
        "create_component": create_component,
        "table_columns": table_columns,
    }


def execute_vde_request_save_plan(save_plan, repositories=None, services=None) -> dict:
    plan = deepcopy(dict(save_plan or {}))
    result = {
        "operation_id": plan.get("operation_id"),
        "status": "failed",
        "executed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "saved_proposals": [],
        "skipped_proposals": deepcopy(list(plan.get("skipped_proposals") or [])),
        "baseline_updates": [],
        "component_results": [],
        "issues": [],
    }
    if plan.get("status") == "blocked" or not plan.get("can_execute"):
        result["issues"] = deepcopy(list(plan.get("blocking_issues") or []))
        return result

    service_map = _default_services()
    service_map.update(dict(services or {}))
    service_map.update(dict(repositories or {}))

    try:
        service_map["ensure_db"]()
        supported_columns = set(service_map["table_columns"]("vde_db"))
        con = service_map["connect_db"]()
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("BEGIN")
        try:
            baseline_reference = _baseline_reference_snapshot({"baseline": {"effective": {}}})
            baseline_id = None
            if list(plan.get("baseline_update_requests") or []):
                first_row = next(iter(plan.get("proposals_to_save") or []), {})
                row_payload = dict(first_row.get("row_payload") or {})
                baseline_id = row_payload.get("vde_id_parent")
                if is_blank(baseline_id):
                    raise ValueError("Baseline updates were requested but the original baseline ID is unavailable.")
                update_payload = {}
                for item in list(plan.get("baseline_update_requests") or []):
                    db_field = item.get("db_field")
                    if db_field and db_field in supported_columns:
                        update_payload[db_field] = item.get("correction")
                if update_payload:
                    _update_row_in_transaction(con, int(baseline_id), update_payload)
                    result["baseline_updates"].append(
                        {
                            "baseline_id": int(baseline_id),
                            "updated_fields": sorted(update_payload.keys()),
                            "status": "updated",
                        }
                    )

            for proposal_row in list(plan.get("proposals_to_save") or []):
                row_payload = {key: value for key, value in dict(proposal_row.get("row_payload") or {}).items() if key in supported_columns}
                inserted_id = _insert_row_in_transaction(con, row_payload)
                result["saved_proposals"].append(
                    {
                        "proposal_id": proposal_row.get("proposal_id"),
                        "vde_row_id": inserted_id,
                        "status": "saved",
                        "name": proposal_row.get("final_name"),
                    }
                )
            con.commit()
        except Exception as exc:
            con.rollback()
            raise exc
        finally:
            con.close()
    except Exception as exc:
        result["issues"].append(_issue("db_save_failed", "error", str(exc)))
        result["status"] = "failed"
        return result

    for proposal_row in list(plan.get("proposals_to_save") or []):
        for component_row in list(proposal_row.get("component_plan") or []):
            domain_key = component_row.get("domain")
            action = component_row.get("action")
            requested_create = bool(component_row.get("requested_create"))
            if action == "reuse_existing":
                result["component_results"].append(
                    {
                        "proposal_id": proposal_row.get("proposal_id"),
                        "domain": domain_key,
                        "status": "reused_existing",
                        "component_id": component_row.get("component_id"),
                    }
                )
                continue
            if action in {"snapshot_only", "unavailable"} or not requested_create:
                result["component_results"].append(
                    {
                        "proposal_id": proposal_row.get("proposal_id"),
                        "domain": domain_key,
                        "status": "snapshot_only" if action != "unavailable" else "unavailable",
                        "component_id": None,
                        "reason": component_row.get("creation_error") or action,
                    }
                )
                continue
            if action == "eligible_for_new_component":
                create_payload = dict(component_row.get("create_payload") or {})
                if not create_payload:
                    result["component_results"].append(
                        {
                            "proposal_id": proposal_row.get("proposal_id"),
                            "domain": domain_key,
                            "status": "component_creation_failed",
                            "component_id": None,
                            "reason": component_row.get("creation_error") or "missing_create_payload",
                        }
                    )
                    continue
                try:
                    created = service_map["create_component"](domain_key, create_payload)
                    result["component_results"].append(
                        {
                            "proposal_id": proposal_row.get("proposal_id"),
                            "domain": domain_key,
                            "status": "created",
                            "component_id": created.get("component_id"),
                        }
                    )
                except Exception as exc:
                    result["component_results"].append(
                        {
                            "proposal_id": proposal_row.get("proposal_id"),
                            "domain": domain_key,
                            "status": "component_creation_failed",
                            "component_id": None,
                            "reason": str(exc),
                        }
                    )

    component_failed = any(item.get("status") == "component_creation_failed" for item in result["component_results"])
    if result["saved_proposals"] and component_failed:
        result["status"] = "partial"
    elif result["saved_proposals"]:
        result["status"] = "success"
    else:
        result["status"] = "failed"
    json.dumps(result, default=str)
    return result


def build_vde_request_save_plan_rows(save_plan: dict | None) -> list[dict]:
    rows = []
    for item in list(dict(save_plan or {}).get("proposals_to_save") or []):
        rows.append(
            {
                "Proposal": item.get("source_column") or item.get("proposal_id"),
                "Status": item.get("status"),
                "Walk From": item.get("walk_from"),
                "Name": item.get("final_name"),
                "VDE_TOTAL [MJ/km]": item.get("vde_total_mj_per_km"),
                "Review confirmed": "Yes" if item.get("review_confirmed") else "No",
            }
        )
    return rows


def build_vde_request_save_result_rows(save_result: dict | None) -> list[dict]:
    rows = []
    result = dict(save_result or {})
    for item in list(result.get("saved_proposals") or []):
        rows.append(
            {
                "kind": "saved_proposal",
                "proposal_id": item.get("proposal_id"),
                "status": item.get("status"),
                "details": f"VDE row {item.get('vde_row_id')}",
            }
        )
    for item in list(result.get("baseline_updates") or []):
        rows.append(
            {
                "kind": "baseline_update",
                "proposal_id": item.get("baseline_id"),
                "status": item.get("status"),
                "details": ", ".join(item.get("updated_fields") or []),
            }
        )
    for item in list(result.get("component_results") or []):
        rows.append(
            {
                "kind": "component",
                "proposal_id": item.get("proposal_id"),
                "status": item.get("status"),
                "details": f"{item.get('domain')}: {item.get('component_id') or item.get('reason') or '-'}",
            }
        )
    for item in list(result.get("issues") or []):
        rows.append(
            {
                "kind": "issue",
                "proposal_id": item.get("proposal_id"),
                "status": item.get("severity"),
                "details": item.get("message"),
            }
        )
    return rows


__all__ = [
    "SAVE_MODE_REQUIRE_ALL_VALID",
    "SAVE_MODE_SELECTED",
    "SAVE_MODE_VALID_ONLY",
    "SAVE_MODES",
    "build_vde_request_save_plan",
    "build_vde_request_save_plan_rows",
    "build_vde_request_save_result_rows",
    "execute_vde_request_save_plan",
    "generate_auto_proposal_name",
]
