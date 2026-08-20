from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any

from src.vde_core.component_repositories import COMPONENT_PROVENANCE_FIELDS, load_component_repository, lookup_component
from src.vde_core.vde_component_modes import canonical_component_mode
from src.vde_core.cycles import use_standard_cycle
from src.vde_core.vde_not_used_modes import is_not_used_proposal
from src.vde_core.roadload import cdA_to_C
from src.vde_core.vde_mass_proposal_resolver import resolve_mass_proposal
from src.vde_core.vde_request_contract import is_blank, normalize_domain, resolve_effective_baseline
from src.vde_core.vde_tire_modes import tire_resolver_inputs_from_details
from src.vde_core.vde_tire_proposal_resolver import resolve_tire_proposal
from src.vde_core.vde_workflow_service import build_vde_setup_preview


VDE_REQUEST_RESOLVER_VERSION = "0.1"

_STATUS_PRIORITY = {
    "OK": 0,
    "Review": 1,
    "Missing": 2,
    "Invalid": 3,
    "Blocked": 4,
}
_NON_BLOCKING_PREVIEW_WARNINGS = {
    "weight_distribution_missing_default_50pct",
}
_COMPONENT_DOMAINS = ("tire", "transmission", "brake", "axle_hubs", "parasitic")
TRANSMISSION_APPLICATION_MODE_DEFAULT = "APPLY_DELTA_TO_TOTAL"
TRANSMISSION_APPLICATION_MODE_KEEP_TOTAL_FIXED = "KEEP_TOTAL_FIXED"
_MANUAL_COMPONENT_FIELD_MAP = {
    "transmission": ("new_trans_A", "new_trans_B", "new_trans_C"),
    "brake": ("brake_A", "brake_B", "brake_C"),
    "axle_hubs": ("axle_hub_A", "axle_hub_B", "axle_hub_C"),
    "parasitic": ("parasitic_A", "parasitic_B", "parasitic_C"),
}


def _issue(
    code: str,
    severity: str,
    message: str,
    *,
    domain: str | None = None,
    field_key: str | None = None,
    proposal_id: str | None = None,
    source_column: str | None = None,
) -> dict:
    return {
        "code": code,
        "severity": severity,
        "domain": domain,
        "field_key": field_key,
        "proposal_id": proposal_id,
        "source_column": source_column,
        "message": message,
    }


def _rollup_statuses(statuses: list[str], *, default: str = "OK") -> str:
    cleaned = [str(item or "").strip().title() for item in statuses if str(item or "").strip()]
    ranked = [item for item in cleaned if item in _STATUS_PRIORITY]
    if not ranked:
        return default
    return max(ranked, key=lambda item: _STATUS_PRIORITY.get(item, -1))


def _to_float(value):
    if is_blank(value):
        return None
    return float(value)


def _first_nonblank(*values):
    for value in values:
        if not is_blank(value):
            return value
    return None


def _copy_abc(payload: dict | None) -> dict[str, float | None]:
    data = dict(payload or {})
    return {
        "A": _to_float(data.get("A")),
        "B": _to_float(data.get("B")),
        "C": _to_float(data.get("C")),
    }


def _abc_complete(payload: dict | None) -> bool:
    data = _copy_abc(payload)
    return all(data[key] is not None for key in ("A", "B", "C"))


def _abc_add(lhs: dict | None, rhs: dict | None) -> dict[str, float | None]:
    left = _copy_abc(lhs)
    right = _copy_abc(rhs)
    return {
        "A": (left["A"] or 0.0) + (right["A"] or 0.0),
        "B": (left["B"] or 0.0) + (right["B"] or 0.0),
        "C": (left["C"] or 0.0) + (right["C"] or 0.0),
    }


def _abc_subtract(lhs: dict | None, rhs: dict | None) -> dict[str, float | None]:
    left = _copy_abc(lhs)
    right = _copy_abc(rhs)
    return {
        "A": (left["A"] or 0.0) - (right["A"] or 0.0),
        "B": (left["B"] or 0.0) - (right["B"] or 0.0),
        "C": (left["C"] or 0.0) - (right["C"] or 0.0),
    }


def _abc_from_sequence(values: tuple[Any, Any, Any] | list[Any]) -> dict[str, float | None]:
    data = list(values or [None, None, None])
    while len(data) < 3:
        data.append(None)
    return {"A": _to_float(data[0]), "B": _to_float(data[1]), "C": _to_float(data[2])}


def _transmission_triplet(payload: dict | None) -> dict[str, float | None]:
    data = dict(payload or {})
    abc = dict(data.get("abc") or {})
    return _abc_from_sequence(
        (
            _first_nonblank(data.get("A_TRANS"), data.get("trans_A_coef_N"), abc.get("A"), data.get("A")),
            _first_nonblank(data.get("B_TRANS"), data.get("trans_B_coef_Npkph"), data.get("trans_B_Npkph"), abc.get("B"), data.get("B")),
            _first_nonblank(data.get("C_TRANS"), data.get("trans_C_coef_Npkph2"), abc.get("C"), data.get("C")),
        )
    )


def _normalize_transmission_application_mode(value) -> str:
    text = str(value or "").strip().upper()
    if text == TRANSMISSION_APPLICATION_MODE_KEEP_TOTAL_FIXED:
        return TRANSMISSION_APPLICATION_MODE_KEEP_TOTAL_FIXED
    return TRANSMISSION_APPLICATION_MODE_DEFAULT


def _transmission_mode_label(value) -> str:
    mode = _normalize_transmission_application_mode(value)
    if mode == TRANSMISSION_APPLICATION_MODE_KEEP_TOTAL_FIXED:
        return "Fixed measured TOTAL - NET recalculated"
    return "Vehicle change - TOTAL updated"


def _transmission_mode_from_snapshot(snapshot: dict | None) -> str:
    payload = dict(snapshot or {})
    return _normalize_transmission_application_mode(
        payload.get("transmission_application_mode")
        or dict(payload.get("transmission_losses") or {}).get("transmission_application_mode")
    )


def _set_transmission_application_mode(snapshot: dict, value) -> str:
    mode = _normalize_transmission_application_mode(value)
    snapshot["transmission_application_mode"] = mode
    transmission = snapshot.get("transmission_losses")
    if isinstance(transmission, dict):
        transmission["transmission_application_mode"] = mode
    return mode


def _json_safe_value(value):
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "to_dict") and not isinstance(value, type):
        try:
            return _json_safe_value(value.to_dict())
        except Exception:
            return str(value)
    if hasattr(value, "__class__") and value.__class__.__name__ == "DataFrame":
        return f"<DataFrame rows={len(value)} cols={len(getattr(value, 'columns', []))}>"
    return value


def _nonblank_mapping(payload: dict | None) -> dict:
    data = dict(payload or {})
    return {str(key): value for key, value in data.items() if not is_blank(value)}


def _component_provenance(payload: dict | None) -> dict:
    data = dict(payload or {})
    return {field_name: data.get(field_name) for field_name in COMPONENT_PROVENANCE_FIELDS if not is_blank(data.get(field_name))}


def _snapshot_view(snapshot: dict) -> dict:
    payload = deepcopy(dict(snapshot or {}))
    payload.pop("cycle_df", None)
    return _json_safe_value(payload)


def _get_import_meta(workbook_state: dict) -> dict:
    return deepcopy(dict(dict(workbook_state or {}).get("vde_request_import") or {}))


def _effective_baseline_payload(workbook_state: dict, baseline_context: dict | None = None) -> tuple[dict, dict]:
    import_meta = _get_import_meta(workbook_state)
    printed = deepcopy(dict(import_meta.get("baseline_printed") or dict(dict(baseline_context or {}).get("printed") or {})))
    correction = deepcopy(dict(import_meta.get("baseline_corrections") or dict(dict(baseline_context or {}).get("correction") or {})))
    effective = deepcopy(dict(import_meta.get("effective_baseline") or {}))
    if not effective:
        field_keys = set(printed) | set(correction)
        effective = {field_key: resolve_effective_baseline(printed.get(field_key), correction.get(field_key)) for field_key in field_keys}
    corrected_fields = [field_key for field_key in sorted(set(printed) | set(correction)) if not is_blank(correction.get(field_key))]
    return effective, {
        "printed": _json_safe_value(printed),
        "correction": _json_safe_value(correction),
        "effective": _json_safe_value(effective),
        "corrected_fields": corrected_fields,
    }


def _build_baseline_snapshot(workbook_state: dict, baseline_context: dict | None = None) -> tuple[dict, dict]:
    effective, baseline_payload = _effective_baseline_payload(workbook_state, baseline_context)
    context = deepcopy(dict(baseline_context or {}))
    legislation = str(effective.get("legislation") or context.get("legislation") or "EPA")
    cycle_df = context.get("cycle_df")
    if cycle_df is None:
        cycle_df = use_standard_cycle(legislation)
    snapshot = {
        "selected_baseline_vde_id": effective.get("selected_baseline_vde_id") or context.get("selected_baseline_vde_id"),
        "legislation": legislation,
        "category": effective.get("category") or context.get("category"),
        "electrification": effective.get("electrification") or context.get("electrification"),
        "transmission_type": effective.get("transmission_type") or context.get("transmission_type"),
        "drive_type": effective.get("drive_type") or context.get("drive_type"),
        "fuel_type": effective.get("fuel_type") or context.get("fuel_type"),
        "make": effective.get("make") or context.get("make"),
        "model": effective.get("model") or context.get("model"),
        "year": effective.get("year") or effective.get("model_year") or context.get("year"),
        "description": effective.get("description") or effective.get("notes") or context.get("description") or context.get("notes"),
        "cycle_name": effective.get("cycle_name") or context.get("cycle_name") or "Pending",
        "mass_kg": _to_float(effective.get("mass_kg") or effective.get("curb_mass_kg") or context.get("mass_kg")),
        "test_mass_kg": _to_float(effective.get("test_mass_kg") or context.get("test_mass_kg")),
        "payload_kg": _to_float(effective.get("payload_kg") or context.get("payload_kg")),
        "weight_dist_fr_pct": _to_float(effective.get("weight_dist_fr_pct") or effective.get("fr_weight_pct") or context.get("weight_dist_fr_pct")),
        "inertia_class": _to_float(effective.get("inertia_class") or context.get("inertia_class")),
        "CdA": _to_float(effective.get("cda_m2") or effective.get("CdA") or context.get("CdA")),
        "frontal_area_m2": _to_float(effective.get("frontal_area_m2") or context.get("frontal_area_m2")),
        "wltp_category": effective.get("wltp_category") or context.get("wltp_category"),
        "front_tire_id": effective.get("front_tire_id") or context.get("front_tire_id"),
        "rear_tire_id": effective.get("rear_tire_id") or context.get("rear_tire_id"),
        "tire_db_id": effective.get("tire_db_id") or context.get("tire_db_id"),
        "tire_code": effective.get("tire_code") or context.get("tire_code"),
        "front_pressure_psi": _to_float(effective.get("front_pressure_psi") or effective.get("psi_front") or context.get("front_pressure_psi")),
        "rear_pressure_psi": _to_float(effective.get("rear_pressure_psi") or effective.get("psi_rear") or context.get("rear_pressure_psi")),
        "rrc_N_per_kN": _to_float(effective.get("rrc_N_per_kN") or context.get("rrc_N_per_kN")),
        "smerf": _to_float(effective.get("smerf") or context.get("smerf")),
        "tire_load_mass_basis": (
            effective.get("tire_load_mass_basis")
            or context.get("tire_load_mass_basis")
            or ("TWC" if legislation == "EPA" else "TEST_MASS")
        ),
        "test_mass_basis": effective.get("test_mass_basis") or context.get("test_mass_basis"),
        "mass_intention": effective.get("mass_intention") or context.get("mass_intention"),
        "GVWR_kg": _to_float(effective.get("gvwr_kg") or effective.get("GVWR_kg") or context.get("GVWR_kg")),
        "GCWR_kg": _to_float(effective.get("mass_profile_gcwr_kg") or effective.get("GCWR_kg") or context.get("GCWR_kg")),
        "trailer_weight_kg": _to_float(effective.get("mass_profile_trailer_mass_kg") or effective.get("trailer_weight_kg") or context.get("trailer_weight_kg")),
        "trailer_code": effective.get("trailer_code") or context.get("trailer_code"),
        "trailer_A": _to_float(effective.get("trailer_A") or context.get("trailer_A")),
        "trailer_B": _to_float(effective.get("trailer_B") or context.get("trailer_B")),
        "trailer_C": _to_float(effective.get("trailer_C") or context.get("trailer_C")),
        "initial_abc_total_source": "manual",
        "initial_abc_total": {
            "A": _to_float(effective.get("A") or effective.get("ABC_TOTAL_A") or context.get("A")),
            "B": _to_float(effective.get("B") or effective.get("ABC_TOTAL_B") or context.get("B")),
            "C": _to_float(effective.get("C") or effective.get("ABC_TOTAL_C") or context.get("C")),
        },
        "transmission_losses": None,
        "transmission_application_mode": TRANSMISSION_APPLICATION_MODE_DEFAULT,
        "tire_A_final": _to_float(effective.get("tire_A_final") or context.get("tire_A_final")),
        "tire_B_final": _to_float(effective.get("tire_B_final") or context.get("tire_B_final")),
        "tire_C_final": _to_float(effective.get("tire_C_final") or context.get("tire_C_final")),
        "tire_calc_source": effective.get("tire_calc_source") or context.get("tire_calc_source"),
        "brake_A": _to_float(effective.get("brake_A_coef_N") or effective.get("brake_A") or context.get("brake_A")),
        "brake_B": _to_float(effective.get("brake_B_Npkph") or effective.get("brake_B") or context.get("brake_B")),
        "brake_C": _to_float(effective.get("brake_C_coef_Npkph2") or effective.get("brake_C") or context.get("brake_C")),
        "axle_hub_A": _to_float(effective.get("axle_hub_A") or context.get("axle_hub_A")),
        "axle_hub_B": _to_float(effective.get("axle_hub_B") or context.get("axle_hub_B")),
        "axle_hub_C": _to_float(effective.get("axle_hub_C") or context.get("axle_hub_C")),
        "parasitic_A": _to_float(effective.get("parasitic_A_coef_N") or effective.get("parasitic_A") or context.get("parasitic_A")),
        "parasitic_B": _to_float(effective.get("parasitic_B_Npkph") or effective.get("parasitic_B") or context.get("parasitic_B")),
        "parasitic_C": _to_float(effective.get("parasitic_C_coef_Npkph2") or effective.get("parasitic_C") or context.get("parasitic_C")),
        "cycle_df": cycle_df,
    }
    trans_abc = _abc_from_sequence(
        (
            effective.get("trans_A_coef_N") or effective.get("trans_A_loss") or context.get("trans_A_coef_N"),
            effective.get("trans_B_coef_Npkph") or effective.get("trans_B_loss") or context.get("trans_B_coef_Npkph") or context.get("trans_B_Npkph"),
            effective.get("trans_C_coef_Npkph2") or effective.get("trans_C_loss") or context.get("trans_C_coef_Npkph2"),
        )
    )
    if _abc_complete(trans_abc):
        snapshot["transmission_losses"] = {
            "source": "baseline_snapshot",
            "A_TRANS": trans_abc["A"],
            "B_TRANS": trans_abc["B"],
            "C_TRANS": trans_abc["C"],
            "transmission_application_mode": TRANSMISSION_APPLICATION_MODE_DEFAULT,
        }
    return snapshot, baseline_payload


def _proposal_columns(workbook_state: dict) -> list[dict]:
    scenarios = list(dict(workbook_state or {}).get("scenarios") or [])
    columns = dict(dict(workbook_state or {}).get("columns") or {})
    import_columns = dict(_get_import_meta(workbook_state).get("columns") or {})
    result: list[dict] = []
    for scenario in scenarios:
        if str(scenario.get("role") or "") != "walked":
            continue
        column_id = str(scenario.get("key") or "")
        column = dict(columns.get(column_id) or {})
        source = dict(import_columns.get(column_id) or {})
        result.append(
            {
                "column_id": column_id,
                "label": str(scenario.get("label") or column_id),
                "proposal_id": str(source.get("proposal_id") or column_id),
                "display_index": source.get("display_index"),
                "source_column": str(source.get("source_column") or str(scenario.get("label") or column_id)),
                "walk_from": str(column.get("walk_from") or "baseline"),
                "direct": deepcopy(dict(column.get("direct") or {})),
                "direct_domains": deepcopy(dict(dict(workbook_state.get("proposals") or {}).get(column_id) or {})),
            }
        )
    return result


def _reference_triplet_from_details(details: dict, domain_key: str) -> dict[str, float | None]:
    if domain_key == "transmission":
        return _abc_from_sequence((details.get("baseline_trans_A"), details.get("baseline_trans_B"), details.get("baseline_trans_C")))
    return _abc_from_sequence((details.get("baseline_component_A"), details.get("baseline_component_B"), details.get("baseline_component_C")))


def _current_component_triplet(snapshot: dict, domain_key: str) -> dict[str, float | None]:
    if domain_key == "transmission":
        current = dict(snapshot.get("transmission_losses") or {})
        return _transmission_triplet(current)
    if domain_key == "brake":
        return _abc_from_sequence((snapshot.get("brake_A"), snapshot.get("brake_B"), snapshot.get("brake_C")))
    if domain_key == "axle_hubs":
        return _abc_from_sequence((snapshot.get("axle_hub_A"), snapshot.get("axle_hub_B"), snapshot.get("axle_hub_C")))
    if domain_key == "parasitic":
        return _abc_from_sequence((snapshot.get("parasitic_A"), snapshot.get("parasitic_B"), snapshot.get("parasitic_C")))
    if domain_key == "tire":
        return _abc_from_sequence((snapshot.get("tire_A_final"), snapshot.get("tire_B_final"), snapshot.get("tire_C_final")))
    return _abc_from_sequence((None, None, None))


def _component_triplet_from_details(domain_key: str, details: dict) -> dict[str, float | None]:
    if domain_key == "transmission":
        return _abc_from_sequence((details.get("trans_A_coef_N"), details.get("trans_B_coef_Npkph"), details.get("trans_C_coef_Npkph2")))
    if domain_key == "brake":
        return _abc_from_sequence((details.get("brake_A_coef_N"), details.get("brake_B_Npkph"), details.get("brake_C_coef_Npkph2")))
    if domain_key == "axle_hubs":
        return _abc_from_sequence((details.get("axle_hub_A"), details.get("axle_hub_B"), details.get("axle_hub_C")))
    if domain_key == "parasitic":
        return _abc_from_sequence((details.get("parasitic_A_coef_N"), details.get("parasitic_B_Npkph"), details.get("parasitic_C_Npkph2")))
    return _abc_from_sequence((None, None, None))


def _set_component_triplet(
    snapshot: dict,
    domain_key: str,
    values: dict[str, float | None],
    *,
    source: str | None = None,
    transmission_application_mode: str | None = None,
) -> None:
    triplet = _copy_abc(values)
    if domain_key == "transmission":
        mode = _normalize_transmission_application_mode(
            transmission_application_mode or _transmission_mode_from_snapshot(snapshot)
        )
        snapshot["transmission_losses"] = {
            "source": source or "request",
            "A_TRANS": triplet["A"],
            "B_TRANS": triplet["B"],
            "C_TRANS": triplet["C"],
            "transmission_application_mode": mode,
        }
        snapshot["transmission_application_mode"] = mode
        return
    if domain_key == "brake":
        snapshot["brake_A"], snapshot["brake_B"], snapshot["brake_C"] = triplet["A"], triplet["B"], triplet["C"]
    elif domain_key == "axle_hubs":
        snapshot["axle_hub_A"], snapshot["axle_hub_B"], snapshot["axle_hub_C"] = triplet["A"], triplet["B"], triplet["C"]
    elif domain_key == "parasitic":
        snapshot["parasitic_A"], snapshot["parasitic_B"], snapshot["parasitic_C"] = triplet["A"], triplet["B"], triplet["C"]
    elif domain_key == "tire":
        snapshot["tire_A_final"], snapshot["tire_B_final"], snapshot["tire_C_final"] = triplet["A"], triplet["B"], triplet["C"]
        if source:
            snapshot["tire_calc_source"] = source


def _apply_total_delta(snapshot: dict, delta: dict[str, float | None]) -> None:
    current = _copy_abc(snapshot.get("initial_abc_total"))
    snapshot["initial_abc_total"] = _abc_add(current, delta)


def _apply_component_delta_to_total(
    snapshot: dict,
    domain_key: str,
    delta: dict[str, float | None],
    *,
    transmission_application_mode: str | None = None,
) -> None:
    if domain_key == "transmission":
        mode = _set_transmission_application_mode(
            snapshot,
            transmission_application_mode or _transmission_mode_from_snapshot(snapshot),
        )
        if mode == TRANSMISSION_APPLICATION_MODE_KEEP_TOTAL_FIXED:
            return
    _apply_total_delta(snapshot, delta)


def _component_action_from_lookup(domain_key: str, lookup_result: dict) -> dict:
    component = deepcopy(dict(lookup_result.get("component") or {}))
    return {
        "action": "reuse_existing" if lookup_result.get("found") else "unavailable",
        "domain": domain_key,
        "component_id": lookup_result.get("component_id"),
        "component_snapshot": _json_safe_value(component) if component else None,
        "requires_confirmation": True,
        "issues": deepcopy(list(lookup_result.get("issues") or [])),
    }


def _component_action_from_manual(domain_key: str, details: dict, *, complete: bool) -> dict:
    fields = _MANUAL_COMPONENT_FIELD_MAP.get(domain_key, ())
    snapshot = {field_name: details.get(field_name) for field_name in fields if details.get(field_name) not in (None, "")}
    snapshot.update(_component_provenance(details))
    return {
        "action": "eligible_for_new_component" if complete else "snapshot_only",
        "domain": domain_key,
        "component_id": None,
        "component_snapshot": _json_safe_value(snapshot),
        "requires_confirmation": True,
        "issues": [],
    }


def _component_action_not_used(domain_key: str) -> dict:
    return {
        "action": "snapshot_only",
        "domain": domain_key,
        "component_id": None,
        "component_snapshot": {"used": False, "reason": "explicit_not_used"},
        "requires_confirmation": True,
        "issues": [],
    }


def _domain_result(
    domain_key: str,
    proposal: dict | None,
    *,
    status: str,
    issues: list[dict] | None = None,
    notes: list[str] | None = None,
    component_action: dict | None = None,
    source_label: str | None = None,
    requested_values: dict | None = None,
    resolved_values: dict | None = None,
) -> dict:
    proposal = dict(proposal or {})
    return {
        "domain": domain_key,
        "proposal_type": str(proposal.get("proposal_type") or "INHERIT"),
        "proposal_label": str(proposal.get("label") or proposal.get("proposal_type") or "Inherit"),
        "status": status,
        "issues": deepcopy(list(issues or [])),
        "notes": deepcopy(list(notes or [])),
        "source": source_label,
        "requested_values": _json_safe_value(deepcopy(dict(requested_values or {}))),
        "resolved_values": _json_safe_value(deepcopy(dict(resolved_values or {}))),
        "component_action": _json_safe_value(component_action) if component_action else None,
    }


def _resolve_inherit(domain_key: str, proposal: dict | None, source_label: str) -> dict:
    return _domain_result(domain_key, proposal, status="OK", notes=[f"Inherited from {source_label}."], source_label=source_label)


def _resolve_mass(domain_key: str, proposal: dict, source_snapshot: dict, working_snapshot: dict, proposal_id: str, source_label: str) -> tuple[dict, list[dict], dict | None]:
    details = deepcopy(dict(proposal.get("details") or {}))
    proposal_type = str(proposal.get("proposal_type") or "").upper()
    explicit_test_mass = details.get("test_mass_kg")
    source_test_mass = source_snapshot.get("test_mass_kg")
    if proposal_type == "EPA_CURB_TO_TWC" and (is_blank(explicit_test_mass) or explicit_test_mass == source_test_mass):
        explicit_test_mass = None
    requested_curb_mass = _first_nonblank(details.get("mass_kg"), details.get("curb_mass_kg"))
    legacy_target_curb_mass = details.get("target_curb_mass_kg")
    if is_blank(requested_curb_mass):
        requested_curb_mass = legacy_target_curb_mass
    inputs = {
        "mass_kg": requested_curb_mass,
        "test_mass_kg": explicit_test_mass,
        "test_mass_basis": details.get("test_mass_basis"),
        "weight_dist_fr_pct": details.get("weight_dist_fr_pct", details.get("fr_weight_pct")),
        "tire_load_mass_basis": details.get("tire_load_mass_basis"),
        "inertia_class": details.get("inertia_class"),
        "shift_steps": details.get("shift_steps"),
        "target_side": details.get("target_side"),
        "curb_position": details.get("curb_position"),
        "target_mass_kg": details.get("target_mass_kg"),
        "preset": details.get("preset"),
        "custom_delta_kg": details.get("custom_delta_kg"),
        "line_type": details.get("line_type"),
        "options_kg": details.get("optional_weight_kg", details.get("options_kg")),
        "payload_kg": details.get("payload_kg"),
        "gvwr_kg": details.get("GVWR_kg", details.get("gvwr_kg")),
        "gcwr_kg": details.get("GCWR_kg", details.get("gcwr_kg")),
        "trailer_mass_kg": details.get("trailer_weight_kg", details.get("trailer_mass_kg")),
        "trailer_A": details.get("trailer_A"),
        "trailer_B": details.get("trailer_B"),
        "trailer_C": details.get("trailer_C"),
        "reference_mass_kg": details.get("reference_mass_kg"),
    }
    if not is_blank(legacy_target_curb_mass):
        inputs["target_curb_mass_kg"] = legacy_target_curb_mass
    outcome = resolve_mass_proposal(source_snapshot, proposal_type, inputs)
    resolved_mass = dict(outcome.get("resolved_snapshot") or {})
    issues = [
        _issue(
            str(item.get("code") or "mass_issue"),
            str(item.get("severity") or "review"),
            str(item.get("message") or "Mass issue."),
            domain=domain_key,
            proposal_id=proposal_id,
            source_column=source_label,
        )
        for item in list(outcome.get("issues") or [])
    ]
    for field_key in (
        "mass_kg",
        "curb_mass_kg",
        "vehicle_loaded_mass_kg",
        "curb_mass_kg",
        "vehicle_loaded_mass_kg",
        "current_curb_mass_kg",
        "target_curb_mass_kg",
        "target_mass_kg",
        "test_mass_kg",
        "test_mass_basis",
        "vde_calculation_mass_kg",
        "vde_mass_basis",
        "weight_dist_fr_pct",
        "inertia_class",
        "payload_kg",
        "vehicle_mass_at_gcwr",
        "trailer_roadload_status",
        "mass_rule_status",
        "mass_rule_notes",
        "test_mass_low_kg",
        "test_mass_high_kg",
        "mass_intention",
        "legislation",
        "tire_load_mass_basis",
        "tire_load_mass_used_kg",
        "target_twc_interval",
        "target_twc_lower_bound_exclusive",
        "target_twc_upper_bound_inclusive",
        "curb_position",
        "gvwr_kg",
        "gcwr_kg",
        "trailer_mass_kg",
    ):
        if field_key in resolved_mass:
            working_snapshot[field_key] = resolved_mass.get(field_key)
    if "gvwr_kg" in resolved_mass:
        working_snapshot["GVWR_kg"] = resolved_mass.get("gvwr_kg")
    if "gcwr_kg" in resolved_mass:
        working_snapshot["GCWR_kg"] = resolved_mass.get("gcwr_kg")
    if "trailer_mass_kg" in resolved_mass:
        working_snapshot["trailer_weight_kg"] = resolved_mass.get("trailer_mass_kg")
    trailer = _abc_from_sequence((resolved_mass.get("trailer_A"), resolved_mass.get("trailer_B"), resolved_mass.get("trailer_C")))
    if _abc_complete(trailer):
        current = _abc_from_sequence((source_snapshot.get("trailer_A"), source_snapshot.get("trailer_B"), source_snapshot.get("trailer_C")))
        delta = trailer if not _abc_complete(current) else _abc_subtract(trailer, current)
        _apply_total_delta(working_snapshot, delta)
        working_snapshot["trailer_A"], working_snapshot["trailer_B"], working_snapshot["trailer_C"] = trailer["A"], trailer["B"], trailer["C"]
    status = _rollup_statuses([issue["severity"].title() for issue in issues], default=str(outcome.get("status") or "OK"))
    if status not in _STATUS_PRIORITY:
        status = "OK"
    if status == "Invalid":
        return _domain_result(domain_key, proposal, status="Invalid", issues=issues, source_label=source_label, requested_values=details), issues, None
    return _domain_result(
        domain_key,
        proposal,
        status=status,
        issues=issues,
        source_label=source_label,
        requested_values=details,
        resolved_values={
            "mass_kg": working_snapshot.get("mass_kg"),
            "current_curb_mass_kg": working_snapshot.get("current_curb_mass_kg"),
            "target_curb_mass_kg": working_snapshot.get("target_curb_mass_kg"),
            "target_mass_kg": working_snapshot.get("target_mass_kg"),
            "inertia_class": working_snapshot.get("inertia_class"),
            "mass_intention": working_snapshot.get("mass_intention"),
            "test_mass_kg": working_snapshot.get("test_mass_kg"),
            "test_mass_basis": working_snapshot.get("test_mass_basis"),
            "weight_dist_fr_pct": working_snapshot.get("weight_dist_fr_pct"),
            "tire_load_mass_basis": working_snapshot.get("tire_load_mass_basis"),
            "payload_kg": working_snapshot.get("payload_kg"),
            "target_twc_interval": working_snapshot.get("target_twc_interval"),
            "target_twc_lower_bound_exclusive": working_snapshot.get("target_twc_lower_bound_exclusive"),
            "target_twc_upper_bound_inclusive": working_snapshot.get("target_twc_upper_bound_inclusive"),
            "mass_rule_status": working_snapshot.get("mass_rule_status"),
            "mass_rule_notes": working_snapshot.get("mass_rule_notes"),
        },
    ), issues, None


def _resolve_aero(domain_key: str, proposal: dict, source_snapshot: dict, working_snapshot: dict, proposal_id: str, source_label: str) -> tuple[dict, list[dict], dict | None]:
    details = deepcopy(dict(proposal.get("details") or {}))
    proposal_type = str(proposal.get("proposal_type") or "").upper()
    issues: list[dict] = []
    notes: list[str] = []
    if proposal_type == "AERO_NOT_USED":
        issues.append(_issue("aero_not_used_review", "review", "Aero Not used remains in Review and does not zero CdA automatically.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
        return _domain_result(domain_key, proposal, status="Review", issues=issues, notes=["Physical aero exclusion is not consolidated yet."], source_label=source_label), issues, _component_action_not_used(domain_key)
    if proposal_type == "AERO_DELTA_CDA":
        delta_cda = _to_float(details.get("delta_CdA"))
        if delta_cda is None:
            issues.append(_issue("missing_delta_cda", "missing", "Delta CdA requires delta_CdA.", domain=domain_key, field_key="delta_CdA", proposal_id=proposal_id, source_column=source_label))
        else:
            source_cda = _to_float(source_snapshot.get("CdA"))
            if source_cda is not None:
                working_snapshot["CdA"] = source_cda + delta_cda
            else:
                issues.append(_issue("missing_source_cda", "review", "Source CdA is missing; delta updates total C but effective CdA display stays unknown.", domain=domain_key, field_key="CdA", proposal_id=proposal_id, source_column=source_label))
            _apply_total_delta(working_snapshot, {"A": 0.0, "B": 0.0, "C": float(cdA_to_C(delta_cda))})
            notes.append(f"Local delta vs {source_label}.")
    elif proposal_type == "AERO_ABSOLUTE_CDA":
        new_cda = _to_float(details.get("new_CdA"))
        reference_cda = _to_float(source_snapshot.get("CdA"))
        provenance = "inherited"
        if reference_cda is None:
            reference_cda = _to_float(details.get("baseline_CdA"))
            provenance = "manual_override"
        if new_cda is None:
            issues.append(_issue("missing_new_cda", "missing", "Absolute CdA requires new_CdA.", domain=domain_key, field_key="new_CdA", proposal_id=proposal_id, source_column=source_label))
        elif reference_cda is None:
            issues.append(_issue("missing_reference_cda", "missing", "Absolute CdA always requires a baseline/reference CdA.", domain=domain_key, field_key="baseline_CdA", proposal_id=proposal_id, source_column=source_label))
        else:
            delta_cda = new_cda - reference_cda
            working_snapshot["CdA"] = new_cda
            _apply_total_delta(working_snapshot, {"A": 0.0, "B": 0.0, "C": float(cdA_to_C(delta_cda))})
            if provenance == "manual_override":
                issues.append(_issue("manual_reference_override", "review", "Manual baseline/reference CdA override was used.", domain=domain_key, field_key="baseline_CdA", proposal_id=proposal_id, source_column=source_label))
            notes.append(f"Local delta vs {source_label}.")
    status = _rollup_statuses([issue["severity"].title() for issue in issues], default="OK")
    if status not in _STATUS_PRIORITY:
        status = "OK"
    return _domain_result(domain_key, proposal, status="Review" if status == "Review" else status, issues=issues, notes=notes, source_label=source_label, requested_values=details, resolved_values={"CdA": working_snapshot.get("CdA")}), issues, {"action": "snapshot_only", "domain": domain_key, "component_id": None, "component_snapshot": {"CdA": working_snapshot.get("CdA")}, "requires_confirmation": True, "issues": []}


def _resolve_component_delta_or_absolute(
    domain_key: str,
    proposal: dict,
    source_snapshot: dict,
    working_snapshot: dict,
    proposal_id: str,
    source_label: str,
    component_repositories: dict | None = None,
) -> tuple[dict, list[dict], dict | None]:
    details = deepcopy(dict(proposal.get("details") or {}))
    proposal_type = str(proposal.get("proposal_type") or "").upper()
    issues: list[dict] = []
    notes: list[str] = []
    transmission_mode = None
    if domain_key == "transmission":
        transmission_mode = _set_transmission_application_mode(
            working_snapshot,
            details.get("transmission_application_mode") or _transmission_mode_from_snapshot(source_snapshot),
        )

    if is_not_used_proposal(domain_key, proposal_type, proposal.get("selection_mode")):
        reference = _current_component_triplet(source_snapshot, domain_key)
        if domain_key == "transmission" and transmission_mode == TRANSMISSION_APPLICATION_MODE_KEEP_TOTAL_FIXED:
            _set_component_triplet(
                working_snapshot,
                domain_key,
                {"A": 0.0, "B": 0.0, "C": 0.0},
                source="explicit_not_used",
                transmission_application_mode=transmission_mode,
            )
            return _domain_result(
                domain_key,
                proposal,
                status="OK",
                issues=[],
                notes=["Explicit Not used applied."],
                source_label=source_label,
                resolved_values={
                    "transmission_application_mode": transmission_mode,
                    "transmission_mode": _transmission_mode_label(transmission_mode),
                },
            ), issues, _component_action_not_used(domain_key)
        if not _abc_complete(reference):
            issues.append(_issue("missing_component_reference", "missing", f"{domain_key} Not used requires a source component reference.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
            return _domain_result(domain_key, proposal, status="Missing", issues=issues, source_label=source_label), issues, _component_action_not_used(domain_key)
        _apply_component_delta_to_total(
            working_snapshot,
            domain_key,
            _abc_subtract({"A": 0.0, "B": 0.0, "C": 0.0}, reference),
            transmission_application_mode=transmission_mode,
        )
        _set_component_triplet(
            working_snapshot,
            domain_key,
            {"A": 0.0, "B": 0.0, "C": 0.0},
            source="explicit_not_used",
            transmission_application_mode=transmission_mode,
        )
        return _domain_result(domain_key, proposal, status="OK", issues=[], notes=["Explicit Not used applied."], source_label=source_label), issues, _component_action_not_used(domain_key)

    if proposal_type in {"TRANS_METADATA_ONLY", "BRAKE_METADATA_ONLY", "AXLE_HUB_METADATA_ONLY", "PARASITIC_METADATA_ONLY"}:
        lookup_id = details.get("component_db_id") or details.get("transmission_component_db_id") or details.get("brake_component_db_id") or details.get("axle_hubs_component_db_id") or details.get("parasitic_component_db_id")
        vde_lookup_id = details.get("transmission_vde_db_id") or details.get("brake_vde_db_id") or details.get("axle_hubs_vde_db_id") or details.get("parasitic_vde_db_id")
        component = {}
        if vde_lookup_id not in (None, ""):
            new_triplet = _component_triplet_from_details(domain_key, details)
            action = _component_action_from_manual(domain_key, details, complete=_abc_complete(new_triplet))
            action["action"] = "reuse_vde_snapshot"
            action["vde_id"] = vde_lookup_id
            action["component_id"] = None
            result_issues = []
            if not _abc_complete(new_triplet):
                issue = _issue("missing_vde_component_reference", "missing", f"Selected VDE does not contain a complete {domain_key} ABC reference.", domain=domain_key, proposal_id=proposal_id, source_column=source_label)
                return _domain_result(domain_key, proposal, status="Missing", issues=[issue], source_label=source_label), [issue], action
        else:
            result = lookup_component(domain_key, str(lookup_id or ""), component_repositories)
            action = _component_action_from_lookup(domain_key, result)
            if not result["found"]:
                return _domain_result(domain_key, proposal, status="Missing", issues=result["issues"], source_label=source_label), result["issues"], action
            component = dict(result["component"] or {})
            result_issues = result["issues"]
            new_triplet = {
                "A": _to_float(
                    _first_nonblank(
                        component.get("trans_A"),
                        component.get("brake_A"),
                        component.get("axle_hubs_A"),
                        component.get("parasitic_A"),
                    )
                ),
                "B": _to_float(
                    _first_nonblank(
                        component.get("trans_B"),
                        component.get("brake_B"),
                        component.get("axle_hubs_B"),
                        component.get("parasitic_B"),
                    )
                ),
                "C": _to_float(
                    _first_nonblank(
                        component.get("trans_C"),
                        component.get("brake_C"),
                        component.get("axle_hubs_C"),
                        component.get("parasitic_C"),
                    )
                ),
            }
        current = _current_component_triplet(source_snapshot, domain_key)
        if not _abc_complete(current) and not (domain_key == "transmission" and transmission_mode == TRANSMISSION_APPLICATION_MODE_KEEP_TOTAL_FIXED):
            issues = result_issues + [_issue("missing_component_reference", "review", f"{domain_key} lookup could not adjust ABC_TOTAL because the inherited component reference is missing.", domain=domain_key, proposal_id=proposal_id, source_column=source_label)]
            return _domain_result(domain_key, proposal, status="Review", issues=issues, source_label=source_label), issues, action
        _apply_component_delta_to_total(
            working_snapshot,
            domain_key,
            _abc_subtract(new_triplet, current),
            transmission_application_mode=transmission_mode,
        )
        _set_component_triplet(
            working_snapshot,
            domain_key,
            new_triplet,
            source=f"lookup:{vde_lookup_id if vde_lookup_id not in (None, '') else lookup_id}",
            transmission_application_mode=transmission_mode,
        )
        if domain_key == "transmission" and component.get("loss_pct") not in (None, ""):
            working_snapshot["transmission_loss_pct"] = _to_float(component.get("loss_pct"))
        resolved_values = dict(new_triplet)
        if domain_key == "transmission":
            resolved_values["transmission_application_mode"] = transmission_mode
            resolved_values["transmission_mode"] = _transmission_mode_label(transmission_mode)
        provenance = _component_provenance(component)
        if provenance:
            resolved_values["component_provenance"] = provenance
        return _domain_result(domain_key, proposal, status="OK", source_label=source_label, resolved_values=resolved_values, component_action=action), [], action

    if proposal_type == "TRANS_LOSS_PCT":
        pct = _to_float(details.get("loss_pct"))
        if pct is None:
            issues.append(_issue("missing_loss_pct", "missing", "Transmission coastdown share requires a percentage.", domain=domain_key, field_key="loss_pct", proposal_id=proposal_id, source_column=source_label))
            return _domain_result(domain_key, proposal, status="Missing", issues=issues, source_label=source_label), issues, _component_action_from_manual(domain_key, details, complete=False)
        rule_version = str(details.get("rule_version") or "").strip().upper()
        if rule_version != "COASTDOWN_SHARE_V1":
            # Preserve unresolved legacy drafts exactly as they were resolved
            # before Coastdown Share v1.  New UI applies the explicit version.
            current = _current_component_triplet(source_snapshot, domain_key)
            if not _abc_complete(current):
                issues.append(_issue("missing_transmission_reference", "missing", "Legacy transmission percentage requires a source transmission reference.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
                return _domain_result(domain_key, proposal, status="Missing", issues=issues, source_label=source_label), issues, _component_action_from_manual(domain_key, details, complete=False)
            factor = 1.0 + (pct / 100.0)
            scaled = {"A": current["A"] * factor, "B": current["B"] * factor, "C": current["C"] * factor}
            _apply_component_delta_to_total(working_snapshot, domain_key, _abc_subtract(scaled, current), transmission_application_mode=transmission_mode)
            _set_component_triplet(working_snapshot, domain_key, scaled, source="legacy_loss_pct", transmission_application_mode=transmission_mode)
            working_snapshot["transmission_loss_pct"] = pct
            scaled["transmission_application_mode"] = transmission_mode
            scaled["transmission_mode"] = _transmission_mode_label(transmission_mode)
            return _domain_result(domain_key, proposal, status="Review", issues=[], notes=["Legacy transmission percentage rule retained; reapply to use Coastdown Share v1."], source_label=source_label, resolved_values=scaled), [], _component_action_from_manual(domain_key, details, complete=True)

        if not math.isfinite(pct) or pct < 0.0 or pct > 100.0:
            issues.append(_issue("invalid_loss_pct", "invalid", "Transmission coastdown share must be between 0 and 100%.", domain=domain_key, field_key="loss_pct", proposal_id=proposal_id, source_column=source_label))
            return _domain_result(domain_key, proposal, status="Invalid", issues=issues, source_label=source_label), issues, _component_action_from_manual(domain_key, details, complete=False)

        source_total = _copy_abc(source_snapshot.get("initial_abc_total") or source_snapshot.get("abc_total"))
        if not _abc_complete(source_total):
            issues.append(_issue("missing_source_abc_total", "missing", "Transmission coastdown share requires the Walk From ABC_TOTAL.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
            return _domain_result(domain_key, proposal, status="Missing", issues=issues, source_label=source_label), issues, _component_action_from_manual(domain_key, details, complete=False)

        transmission_mode = TRANSMISSION_APPLICATION_MODE_KEEP_TOTAL_FIXED
        _set_transmission_application_mode(working_snapshot, transmission_mode)
        scaled = {key: value * (pct / 100.0) for key, value in source_total.items()}
        _set_component_triplet(
            working_snapshot,
            domain_key,
            scaled,
            source="loss_pct",
            transmission_application_mode=transmission_mode,
        )
        working_snapshot["transmission_loss_pct"] = pct
        working_snapshot["transmission_percent_basis"] = "SOURCE_ABC_TOTAL"
        working_snapshot["transmission_rule_version"] = "COASTDOWN_SHARE_V1"
        scaled["transmission_application_mode"] = transmission_mode
        scaled["transmission_mode"] = _transmission_mode_label(transmission_mode)
        scaled["percent_basis"] = "SOURCE_ABC_TOTAL"
        scaled["rule_version"] = "COASTDOWN_SHARE_V1"
        scaled["source_abc_total"] = source_total
        return _domain_result(domain_key, proposal, status="OK", issues=[], notes=["Transmission coastdown share estimated from Walk From ABC_TOTAL; total roadload remains fixed."], source_label=source_label, resolved_values=scaled), [], _component_action_from_manual(domain_key, details, complete=True)

    component_mode = canonical_component_mode(domain_key, proposal_type, proposal.get("selection_mode"), details)
    if domain_key == "brake" and component_mode == "RESIDUAL_TORQUE":
        current = _current_component_triplet(source_snapshot, domain_key)
        torque_front = _to_float(details.get("residual_torque_front_nm", details.get("residual_torque_front_Nm")))
        torque_rear = _to_float(details.get("residual_torque_rear_nm", details.get("residual_torque_rear_Nm")))
        torque_total = _to_float(details.get("residual_torque_total_Nm"))
        wheel_radius = _to_float(details.get("wheel_radius_m"))
        axle_torques = [value for value in (torque_front, torque_rear) if value is not None]
        torque_sum = torque_total if torque_total is not None else (sum(axle_torques) if axle_torques else None)
        if torque_sum is None or wheel_radius in (None, 0.0):
            issues.append(_issue("missing_residual_torque_inputs", "missing", "Residual torque requires total torque or front/rear torque, plus wheel radius.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
            return _domain_result(domain_key, proposal, status="Missing", issues=issues, source_label=source_label), issues, _component_action_from_manual(domain_key, details, complete=False)
        drag_a = torque_sum / wheel_radius
        delta = {"A": drag_a, "B": 0.0, "C": 0.0}
        _apply_total_delta(working_snapshot, delta)
        if _abc_complete(current):
            new_triplet = _abc_add(current, delta)
            _set_component_triplet(working_snapshot, domain_key, new_triplet, source="residual_torque")
        return _domain_result(domain_key, proposal, status="OK", issues=[], notes=["Residual torque converted to delta A contribution."], source_label=source_label, resolved_values=delta), [], _component_action_from_manual(domain_key, details, complete=True)

    if component_mode == "DELTA_ABC":
        delta = _abc_from_sequence((details.get("delta_A"), details.get("delta_B"), details.get("delta_C")))
        if not _abc_complete(delta):
            issues.append(_issue("missing_delta_abc", "missing", "Delta ABC requires delta_A, delta_B and delta_C.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
            return _domain_result(domain_key, proposal, status="Missing", issues=issues, source_label=source_label), issues, _component_action_from_manual(domain_key, details, complete=False)
        _apply_component_delta_to_total(
            working_snapshot,
            domain_key,
            delta,
            transmission_application_mode=transmission_mode,
        )
        current = _current_component_triplet(source_snapshot, domain_key)
        if _abc_complete(current):
            _set_component_triplet(
                working_snapshot,
                domain_key,
                _abc_add(current, delta),
                source="delta",
                transmission_application_mode=transmission_mode,
            )
        notes.append(f"Local delta vs {source_label}.")
        if domain_key == "transmission":
            delta["transmission_application_mode"] = transmission_mode
            delta["transmission_mode"] = _transmission_mode_label(transmission_mode)
        return _domain_result(domain_key, proposal, status="OK", issues=[], notes=notes, source_label=source_label, requested_values=details, resolved_values=delta), [], _component_action_from_manual(domain_key, details, complete=True)

    if component_mode == "ABSOLUTE_ABC":
        if domain_key == "transmission":
            new_triplet = _abc_from_sequence((details.get("new_trans_A"), details.get("new_trans_B"), details.get("new_trans_C")))
        else:
            fields = _MANUAL_COMPONENT_FIELD_MAP[domain_key]
            new_triplet = _abc_from_sequence((details.get(fields[0]), details.get(fields[1]), details.get(fields[2])))
        if not _abc_complete(new_triplet):
            issues.append(_issue("missing_absolute_abc", "missing", "Absolute ABC requires complete new A/B/C values.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
            return _domain_result(domain_key, proposal, status="Missing", issues=issues, source_label=source_label), issues, _component_action_from_manual(domain_key, details, complete=False)
        reference = _current_component_triplet(source_snapshot, domain_key)
        provenance = "inherited"
        if not _abc_complete(reference):
            reference = _reference_triplet_from_details(details, domain_key)
            provenance = "manual_override"
        if not _abc_complete(reference) and not (domain_key == "transmission" and transmission_mode == TRANSMISSION_APPLICATION_MODE_KEEP_TOTAL_FIXED):
            issues.append(_issue("missing_component_reference", "missing", "Absolute proposals always require a baseline/reference component ABC.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
            return _domain_result(domain_key, proposal, status="Missing", issues=issues, source_label=source_label), issues, _component_action_from_manual(domain_key, details, complete=True)
        delta = _abc_subtract(new_triplet, reference)
        _apply_component_delta_to_total(
            working_snapshot,
            domain_key,
            delta,
            transmission_application_mode=transmission_mode,
        )
        _set_component_triplet(
            working_snapshot,
            domain_key,
            new_triplet,
            source=provenance,
            transmission_application_mode=transmission_mode,
        )
        if provenance == "manual_override":
            issues.append(_issue("manual_reference_override", "review", "Manual baseline/reference override was used.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
        notes.append(f"Local delta vs {source_label}.")
        status = "Review" if issues else "OK"
        if domain_key == "transmission":
            new_triplet["transmission_application_mode"] = transmission_mode
            new_triplet["transmission_mode"] = _transmission_mode_label(transmission_mode)
        return _domain_result(domain_key, proposal, status=status, issues=issues, notes=notes, source_label=source_label, requested_values=details, resolved_values=new_triplet), issues, _component_action_from_manual(domain_key, details, complete=True)

    issues.append(_issue("unsupported_component_mode", "review", f"Unsupported component mode for {domain_key}.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
    return _domain_result(domain_key, proposal, status="Review", issues=issues, source_label=source_label), issues, _component_action_from_manual(domain_key, details, complete=False)


def _resolve_tire(domain_key: str, proposal: dict, source_snapshot: dict, working_snapshot: dict, proposal_id: str, source_label: str) -> tuple[dict, list[dict], dict | None]:
    details = deepcopy(dict(proposal.get("details") or {}))
    proposal_type = str(proposal.get("proposal_type") or "").upper()
    issues: list[dict] = []
    if is_not_used_proposal(domain_key, proposal_type, proposal.get("selection_mode")):
        current = _current_component_triplet(source_snapshot, domain_key)
        if not _abc_complete(current):
            issues.append(_issue("missing_tire_reference", "review", "Tire Not used was requested but the inherited tire contribution is unavailable.", domain=domain_key, proposal_id=proposal_id, source_column=source_label))
            return _domain_result(domain_key, proposal, status="Review", issues=issues, source_label=source_label), issues, _component_action_not_used(domain_key)
        _apply_total_delta(working_snapshot, _abc_subtract({"A": 0.0, "B": 0.0, "C": 0.0}, current))
        _set_component_triplet(working_snapshot, domain_key, {"A": 0.0, "B": 0.0, "C": 0.0}, source="explicit_not_used")
        return _domain_result(domain_key, proposal, status="OK", issues=[], notes=["Explicit Not used applied."], source_label=source_label), issues, _component_action_not_used(domain_key)

    inputs = tire_resolver_inputs_from_details(proposal_type, details)
    requested_values = _nonblank_mapping(inputs)
    tire_working_snapshot = deepcopy(dict(working_snapshot or {}))
    for pressure_key in ("front_pressure_psi", "rear_pressure_psi"):
        if not is_blank(inputs.get(pressure_key)):
            tire_working_snapshot[pressure_key] = inputs.get(pressure_key)
    outcome = resolve_tire_proposal(source_snapshot, proposal_type, inputs, current_snapshot=tire_working_snapshot)
    resolved_tire = dict(outcome.get("resolved_snapshot") or {})
    issues.extend(
        _issue(
            str(item.get("code") or "tire_issue"),
            str(item.get("severity") or "review"),
            str(item.get("message") or "Tire issue."),
            domain=domain_key,
            proposal_id=proposal_id,
            source_column=source_label,
        )
        for item in list(outcome.get("issues") or [])
    )
    for field_key in (
        "tire_db_id",
        "tire_code",
        "front_pressure_psi",
        "rear_pressure_psi",
        "tire_load_mass_basis",
        "tire_load_mass_used_kg",
        "source_tire_load_mass_used_kg",
        "rrc_N_per_kN",
        "target_rrc_N_per_kN",
        "tire_source_rrc_N_per_kN",
        "tire_target_rrc_N_per_kN",
        "tire_adjusted_rrc_N_per_kN",
        "tire_delta_rrc_N_per_kN",
        "tire_reference_front_pressure_psi",
        "tire_reference_rear_pressure_psi",
        "tire_requested_front_pressure_psi",
        "tire_requested_rear_pressure_psi",
        "tire_front_weight_fraction",
        "tire_pressure_sensitivity",
        "tire_adjustment_method",
        "tire_abc_method",
        "tire_review_status",
        "tire_rule_status",
        "tire_rule_notes",
    ):
        if field_key in resolved_tire:
            working_snapshot[field_key] = resolved_tire.get(field_key)
    new_triplet = _abc_from_sequence((resolved_tire.get("tire_A_final"), resolved_tire.get("tire_B_final"), resolved_tire.get("tire_C_final")))
    current = _current_component_triplet(source_snapshot, domain_key)
    if not _abc_complete(current):
        delta_triplet = _copy_abc(resolved_tire.get("tire_delta_abc"))
        if _abc_complete(delta_triplet):
            _apply_total_delta(working_snapshot, delta_triplet)
        if _abc_complete(new_triplet):
            _set_component_triplet(working_snapshot, domain_key, new_triplet, source=str(resolved_tire.get("tire_adjustment_method") or "rrc"))
        status = _rollup_statuses([issue["severity"].title() for issue in issues], default=str(outcome.get("status") or "OK"))
        return _domain_result(
            domain_key,
            proposal,
            status=status if status in _STATUS_PRIORITY else "Review",
            issues=issues,
            source_label=source_label,
            requested_values=requested_values,
            resolved_values={
                "resolved_rrc_N_per_kN": resolved_tire.get("rrc_N_per_kN"),
                "adjustment_method": resolved_tire.get("tire_adjustment_method"),
                "delta_rrc_N_per_kN": resolved_tire.get("tire_delta_rrc_N_per_kN"),
                "front_weight_distribution_pct": None if resolved_tire.get("tire_front_weight_fraction") is None else float(resolved_tire.get("tire_front_weight_fraction")) * 100.0,
                "rear_weight_distribution_pct": None if resolved_tire.get("tire_front_weight_fraction") is None else (1.0 - float(resolved_tire.get("tire_front_weight_fraction"))) * 100.0,
                "resolved_tire_ABC": resolved_tire.get("tire_resolved_abc"),
                "delta_tire_ABC": resolved_tire.get("tire_delta_abc"),
            },
        ), issues, {"action": "reuse_existing", "domain": domain_key, "component_id": resolved_tire.get("tire_db_id"), "component_snapshot": _json_safe_value(resolved_tire), "requires_confirmation": True, "issues": []}

    _apply_total_delta(working_snapshot, _abc_subtract(new_triplet, current))
    _set_component_triplet(working_snapshot, domain_key, new_triplet, source=str(resolved_tire.get("tire_adjustment_method") or "rrc"))
    status = _rollup_statuses([issue["severity"].title() for issue in issues], default=str(outcome.get("status") or "OK"))
    if status not in _STATUS_PRIORITY:
        status = "OK"
    return _domain_result(
        domain_key,
        proposal,
        status=status,
        issues=issues,
        notes=[str(resolved_tire.get("tire_adjustment_method") or proposal_type)],
        source_label=source_label,
        requested_values=requested_values,
        resolved_values={
            "source_rrc_N_per_kN": resolved_tire.get("tire_source_rrc_N_per_kN"),
            "resolved_rrc_N_per_kN": resolved_tire.get("rrc_N_per_kN"),
            "delta_rrc_N_per_kN": resolved_tire.get("tire_delta_rrc_N_per_kN"),
            "reference_pressure_front_rear_psi": {
                "front": resolved_tire.get("tire_reference_front_pressure_psi"),
                "rear": resolved_tire.get("tire_reference_rear_pressure_psi"),
            },
            "requested_pressure_front_rear_psi": {
                "front": resolved_tire.get("tire_requested_front_pressure_psi"),
                "rear": resolved_tire.get("tire_requested_rear_pressure_psi"),
            },
            "front_weight_fraction": resolved_tire.get("tire_front_weight_fraction"),
            "front_weight_distribution_pct": None if resolved_tire.get("tire_front_weight_fraction") is None else float(resolved_tire.get("tire_front_weight_fraction")) * 100.0,
            "rear_weight_distribution_pct": None if resolved_tire.get("tire_front_weight_fraction") is None else (1.0 - float(resolved_tire.get("tire_front_weight_fraction"))) * 100.0,
            "sensitivity": resolved_tire.get("tire_pressure_sensitivity"),
            "adjustment_method": resolved_tire.get("tire_adjustment_method"),
            "tire_abc_method": resolved_tire.get("tire_abc_method"),
            "tire_load_mass_basis": resolved_tire.get("tire_load_mass_basis"),
            "tire_load_mass_used_kg": resolved_tire.get("tire_load_mass_used_kg"),
            "source_tire_load_mass_used_kg": resolved_tire.get("source_tire_load_mass_used_kg"),
            "source_tire_ABC": resolved_tire.get("tire_source_abc"),
            "resolved_tire_ABC": resolved_tire.get("tire_resolved_abc"),
            "delta_tire_ABC": resolved_tire.get("tire_delta_abc"),
        },
    ), issues, {"action": "reuse_existing", "domain": domain_key, "component_id": resolved_tire.get("tire_db_id"), "component_snapshot": _json_safe_value(resolved_tire), "requires_confirmation": True, "issues": []}


def _resolve_domain(
    domain_key: str,
    proposal: dict | None,
    source_snapshot: dict,
    working_snapshot: dict,
    proposal_id: str,
    source_label: str,
    component_repositories: dict | None = None,
) -> tuple[dict, list[dict], dict | None]:
    if domain_key == "tire" and (not proposal or str(proposal.get("proposal_type") or "INHERIT").upper() == "INHERIT"):
        inherited_tire = dict(proposal or {"proposal_type": "INHERIT", "label": "Inherit", "details": {}})
        inherited_tire.setdefault("proposal_type", "INHERIT")
        inherited_tire.setdefault("label", "Inherit")
        inherited_tire.setdefault("details", {})
        return _resolve_tire(domain_key, inherited_tire, source_snapshot, working_snapshot, proposal_id, source_label)
    if not proposal or str(proposal.get("proposal_type") or "INHERIT").upper() == "INHERIT":
        return _resolve_inherit(domain_key, proposal, source_label), [], None
    if domain_key == "mass":
        return _resolve_mass(domain_key, proposal, source_snapshot, working_snapshot, proposal_id, source_label)
    if domain_key == "aero":
        return _resolve_aero(domain_key, proposal, source_snapshot, working_snapshot, proposal_id, source_label)
    if domain_key == "tire":
        return _resolve_tire(domain_key, proposal, source_snapshot, working_snapshot, proposal_id, source_label)
    if domain_key in {"transmission", "brake", "axle_hubs", "parasitic"}:
        return _resolve_component_delta_or_absolute(domain_key, proposal, source_snapshot, working_snapshot, proposal_id, source_label, component_repositories)
    issues = [_issue("unsupported_domain", "review", f"Unsupported domain '{domain_key}'.", domain=domain_key, proposal_id=proposal_id, source_column=source_label)]
    return _domain_result(domain_key, proposal, status="Review", issues=issues, source_label=source_label), issues, None


def _build_preview_from_snapshot(snapshot: dict) -> dict:
    payload = deepcopy(dict(snapshot or {}))
    if payload.get("cycle_df") is None:
        payload["cycle_df"] = use_standard_cycle(str(payload.get("legislation") or "EPA"))
    preview = build_vde_setup_preview(payload)
    return {
        "ok": bool(preview.get("ok")),
        "warnings": _json_safe_value(list(preview.get("warnings") or [])),
        "abc_total": _json_safe_value(dict(preview.get("abc_total") or {})),
        "abc_net": _json_safe_value(dict(preview.get("abc_net") or {})) if preview.get("abc_net") is not None else None,
        "vde_total": _json_safe_value(dict(preview.get("vde_total") or {})) if preview.get("vde_total") is not None else None,
        "vde_net": _json_safe_value(dict(preview.get("vde_net") or {})) if preview.get("vde_net") is not None else None,
        "mass_setup": _json_safe_value(dict(preview.get("mass_setup") or {})),
        "transmission_losses": _json_safe_value(dict(preview.get("transmission_losses") or {})),
    }


def _blocking_preview_status(preview_summary: dict | None) -> str | None:
    warnings = [str(item or "").strip() for item in list(dict(preview_summary or {}).get("warnings") or []) if str(item or "").strip()]
    blocking = [warning for warning in warnings if warning not in _NON_BLOCKING_PREVIEW_WARNINGS]
    return "Review" if blocking else None


def _resolved_snapshot_from_preview(snapshot: dict, preview_summary: dict) -> dict:
    resolved = deepcopy(dict(snapshot or {}))
    if preview_summary.get("abc_total") is not None:
        resolved["initial_abc_total"] = deepcopy(dict(preview_summary.get("abc_total") or {}))
    resolved["abc_total"] = deepcopy(dict(preview_summary.get("abc_total") or {}))
    resolved["abc_net"] = deepcopy(preview_summary.get("abc_net"))
    resolved["vde_total"] = deepcopy(preview_summary.get("vde_total"))
    resolved["vde_net"] = deepcopy(preview_summary.get("vde_net"))
    resolved["preview_warnings"] = deepcopy(list(preview_summary.get("warnings") or []))
    resolved["resolved_mass_setup"] = deepcopy(dict(preview_summary.get("mass_setup") or {}))
    mass_setup = dict(preview_summary.get("mass_setup") or {})
    for field_key in (
        "mass_kg",
        "test_mass_kg",
        "test_mass_low_kg",
        "test_mass_high_kg",
        "test_mass_basis",
        "inertia_class",
        "payload_kg",
        "gvwr_kg",
        "gcwr_kg",
        "trailer_mass_kg",
        "mass_basis",
        "vde_mass_basis",
        "vde_calculation_mass_kg",
        "tire_load_mass_basis",
        "tire_load_mass_used_kg",
        "tire_load_mass_source",
        "mass_intention",
    ):
        if field_key in mass_setup:
            resolved[field_key] = mass_setup.get(field_key)
    if "trailer_mass_kg" in mass_setup:
        resolved["trailer_weight_kg"] = mass_setup.get("trailer_mass_kg")
    transmission_losses = dict(preview_summary.get("transmission_losses") or {})
    transmission_mode = _transmission_mode_from_snapshot(snapshot)
    if transmission_losses:
        transmission_losses["transmission_application_mode"] = transmission_mode
        resolved["transmission_losses"] = deepcopy(transmission_losses)
        trans_triplet = _transmission_triplet(transmission_losses)
        if _abc_complete(trans_triplet):
            resolved["trans_A_coef_N"] = trans_triplet["A"]
            resolved["trans_B_coef_Npkph"] = trans_triplet["B"]
            resolved["trans_C_coef_Npkph2"] = trans_triplet["C"]
    resolved["transmission_application_mode"] = transmission_mode
    return resolved


def _proposal_result(column_meta: dict, source_snapshot: dict, requested_snapshot: dict, resolved_snapshot: dict, domain_results: dict, issues: list[dict], component_actions: list[dict], preview_summary: dict) -> dict:
    preview_status = _blocking_preview_status(preview_summary)
    proposal_status = _rollup_statuses(
        [item.get("status") for item in domain_results.values()] +
        ([preview_status] if preview_status else []) +
        [issue.get("severity", "").title() for issue in issues],
        default="OK",
    )
    return {
        "proposal_id": column_meta["proposal_id"],
        "display_index": column_meta["display_index"],
        "source_column": column_meta["source_column"],
        "walk_from": {
            "column_id": column_meta["walk_from"],
            "label": "Baseline" if column_meta["walk_from"] == "baseline" else column_meta["walk_from"],
        },
        "source_snapshot": _snapshot_view(source_snapshot),
        "requested_snapshot": _snapshot_view(requested_snapshot),
        "resolved_snapshot": _snapshot_view(resolved_snapshot),
        "domain_results": _json_safe_value(domain_results),
        "abc_total": _json_safe_value(resolved_snapshot.get("abc_total")),
        "abc_net": _json_safe_value(resolved_snapshot.get("abc_net")),
        "vde_results": {
            "total": _json_safe_value(resolved_snapshot.get("vde_total")),
            "net": _json_safe_value(resolved_snapshot.get("vde_net")),
        },
        "status": proposal_status,
        "issues": _json_safe_value(issues),
        "component_actions": _json_safe_value(component_actions),
        "preview_summary": _json_safe_value(preview_summary),
    }


def _apply_column_direct_metadata(working_snapshot: dict, direct_payload: dict | None) -> None:
    direct = dict(direct_payload or {})
    mapping = {
        "name": "name",
        "description": "description",
        "make": "make",
        "model": "model",
        "year": "year",
        "category": "category",
        "electrification": "electrification",
        "transmission_type": "transmission_type",
        "drive_type": "drive_type",
        "fuel_type": "fuel_type",
        "legislation": "legislation",
        "cycle_name": "cycle_name",
    }
    for source_key, target_key in mapping.items():
        if direct.get(source_key) in (None, ""):
            continue
        working_snapshot[target_key] = direct.get(source_key)


def resolve_vde_request(workbook_state, baseline_context=None, component_repositories=None) -> dict:
    workbook = deepcopy(dict(workbook_state or {}))
    baseline_snapshot, baseline_payload = _build_baseline_snapshot(workbook, baseline_context)
    repositories = dict(component_repositories or {})
    for domain_key in ("transmission", "brake", "axle_hubs", "parasitic"):
        repositories.setdefault(domain_key, load_component_repository(domain_key))

    try:
        baseline_preview = _build_preview_from_snapshot(baseline_snapshot)
        baseline_resolved_snapshot = _resolved_snapshot_from_preview(baseline_snapshot, baseline_preview)
    except ValueError as exc:
        # Preserve the authoritative baseline snapshot when legacy source data
        # cannot satisfy the existing mass contract required by VDE preview.
        baseline_preview = {
            "abc_total": deepcopy(dict(baseline_snapshot.get("initial_abc_total") or {})),
            "abc_net": None,
            "vde_total": None,
            "vde_net": None,
            "mass_setup": {},
            "warnings": [str(exc)],
        }
        baseline_resolved_snapshot = deepcopy(baseline_snapshot)
        baseline_resolved_snapshot["abc_total"] = deepcopy(dict(baseline_snapshot.get("initial_abc_total") or {}))
    baseline_result = {
        "abc_total": _json_safe_value(baseline_resolved_snapshot.get("abc_total")),
        "abc_net": _json_safe_value(baseline_resolved_snapshot.get("abc_net")) if baseline_resolved_snapshot.get("abc_net") is not None else None,
        "vde_results": {
            "total": _json_safe_value(baseline_resolved_snapshot.get("vde_total")) if baseline_resolved_snapshot.get("vde_total") is not None else None,
            "net": _json_safe_value(baseline_resolved_snapshot.get("vde_net")) if baseline_resolved_snapshot.get("vde_net") is not None else None,
        },
        "mass_setup": _json_safe_value(dict(baseline_preview.get("mass_setup") or {})),
        "cycle_name": baseline_resolved_snapshot.get("cycle_name"),
        "warnings": _json_safe_value(list(baseline_preview.get("warnings") or [])),
    }
    resolved_columns: dict[str, dict] = {"baseline": deepcopy(baseline_resolved_snapshot)}
    proposal_results: list[dict] = []
    all_issues: list[dict] = []

    column_order = ["baseline"]
    proposal_columns = _proposal_columns(workbook)

    for index, column_meta in enumerate(proposal_columns, start=1):
        column_id = column_meta["column_id"]
        column_order.append(column_id)
        allowed = column_order[:-1]
        walk_from = column_meta["walk_from"] or "baseline"
        if walk_from not in allowed:
            issue = _issue("invalid_walk_from", "blocked", f"Walk From '{walk_from}' is not a valid prior column.", proposal_id=column_meta["proposal_id"], source_column=column_meta["source_column"])
            proposal_result = {
                "proposal_id": column_meta["proposal_id"],
                "display_index": column_meta["display_index"] or index,
                "source_column": column_meta["source_column"],
                "walk_from": {"column_id": walk_from, "label": walk_from},
                "source_snapshot": None,
                "requested_snapshot": None,
                "resolved_snapshot": None,
                "domain_results": {},
                "abc_total": None,
                "abc_net": None,
                "vde_results": {"total": None, "net": None},
                "status": "Blocked",
                "issues": [issue],
                "component_actions": [],
                "preview_summary": None,
            }
            proposal_results.append(proposal_result)
            all_issues.append(issue)
            continue

        source_snapshot = deepcopy(resolved_columns[walk_from])
        requested_snapshot = deepcopy(source_snapshot)
        _apply_column_direct_metadata(requested_snapshot, column_meta.get("direct"))
        domain_results: dict[str, dict] = {}
        issues: list[dict] = []
        component_actions: list[dict] = []
        for domain_key in ("mass", "aero", "tire", "transmission", "brake", "axle_hubs", "parasitic"):
            proposal = dict(column_meta["direct_domains"].get(domain_key) or {})
            domain_result, domain_issues, component_action = _resolve_domain(
                domain_key,
                proposal,
                source_snapshot,
                requested_snapshot,
                column_meta["proposal_id"],
                "Baseline" if walk_from == "baseline" else walk_from,
                repositories,
            )
            domain_results[domain_key] = domain_result
            issues.extend(domain_issues)
            if component_action:
                component_actions.append(component_action)

        preview_summary = _build_preview_from_snapshot(requested_snapshot)
        resolved_snapshot = _resolved_snapshot_from_preview(requested_snapshot, preview_summary)
        proposal_result = _proposal_result(
            column_meta,
            source_snapshot,
            requested_snapshot,
            resolved_snapshot,
            domain_results,
            issues,
            component_actions,
            preview_summary,
        )
        proposal_results.append(proposal_result)
        resolved_columns[column_id] = deepcopy(resolved_snapshot)
        all_issues.extend(issues)

    result = {
        "resolver_version": VDE_REQUEST_RESOLVER_VERSION,
        "baseline": baseline_payload,
        "baseline_result": baseline_result,
        "column_order": column_order,
        "proposal_results": proposal_results,
        "resolved_columns": {column_id: _snapshot_view(snapshot) for column_id, snapshot in resolved_columns.items()},
        "status": _rollup_statuses([item.get("status") for item in proposal_results] + [issue.get("severity", "").title() for issue in all_issues], default="OK"),
        "issues": _json_safe_value(all_issues),
    }
    json.dumps(result, default=str)
    return result


__all__ = [
    "VDE_REQUEST_RESOLVER_VERSION",
    "resolve_vde_request",
]
