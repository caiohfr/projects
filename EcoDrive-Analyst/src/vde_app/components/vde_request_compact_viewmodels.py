from __future__ import annotations

from copy import deepcopy

from src.vde_app.components.vde_request_domain_editors import proposal_is_not_used
from src.vde_app.components.vde_request_domain_editors import field_meta
from src.vde_app.components.vde_request_domain_editors import friendly_message
from src.vde_app.components.vde_request_domain_editors import proposal_status_label
from src.vde_app.components.vde_request_domain_editors import rows_for_active_domain
from src.vde_app.components.vde_request_compact_units import (
    display_unit_for_field,
    format_display_value_for_field,
    format_value_map_for_display,
    quantity_kind_for_field,
)
from src.vde_app.units import (
    format_quantity,
    normalize_unit_system,
    to_display,
    unit_label,
)
from src.vde_core.cycles import load_cycle_csv
from src.vde_core.roadload_analysis import build_cycle_power_analysis, build_roadload_curve, canonical_cycle_segments, roadload_force_N
from src.vde_core.vde_request_compact_state import normalize_v22_state
from src.vde_core.vde_request_contract import is_blank
from src.vde_core.vde_request_preview import validation_allows_save


SECTION_ORDER = ("baseline", "matrix", "inputs", "preview")
SECTION_LABELS = {
    "baseline": "Baseline & Corrections",
    "matrix": "Proposal Matrix",
    "inputs": "Request Inputs",
    "preview": "Preview & Save",
}
SECTION_CAPTIONS = {
    "baseline": "Select and review the loaded technical baseline.",
    "matrix": "Configure proposal inheritance and direct-edit modes.",
    "inputs": "Edit and apply one engineering domain at a time.",
    "preview": "Validate the canonical draft and inspect the technical preview.",
}
SECTION_ICONS = {
    "complete": "[OK]",
    "active": "[>]",
    "review": "!",
    "stale": "[~]",
    "pending": "[ ]",
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

_STRUCTURED_ISSUE_VALUE_KEYS = ("value", "actual", "expected", "min", "max")
_UNAVAILABLE = "\u2014"
_ROADLOAD_SPEED_OPTIONS_KPH = (120, 140, 160)
_ROADLOAD_CHECKPOINTS_KPH = (0, 50, 100, 120)
_BASELINE_CORRECTION_DOMAIN_BY_FIELD = {
    "mass_kg": "Mass",
    "test_mass_kg": "Mass",
    "test_mass_basis": "Mass",
    "inertia_class": "Mass",
    "payload_kg": "Mass",
    "options_kg": "Mass",
    "weight_dist_fr_pct": "Mass",
    "gvwr_kg": "Mass",
    "gcwr_kg": "Mass",
    "trailer_mass_kg": "Mass",
    "cda_m2": "Aero",
    "CdA": "Aero",
    "tire_db_id": "Tire",
    "tire_code": "Tire",
    "rrc_N_per_kN": "Tire",
    "front_pressure_psi": "Tire",
    "rear_pressure_psi": "Tire",
    "transmission_component_db_id": "Transmission",
    "trans_A_coef_N": "Transmission",
    "trans_B_coef_Npkph": "Transmission",
    "trans_C_coef_Npkph2": "Transmission",
    "brake_component_db_id": "Brake",
    "brake_A_coef_N": "Brake",
    "brake_B_Npkph": "Brake",
    "brake_C_coef_Npkph2": "Brake",
    "axle_hubs_component_db_id": "Axle & Hubs",
    "axle_hub_A": "Axle & Hubs",
    "axle_hub_B": "Axle & Hubs",
    "axle_hub_C": "Axle & Hubs",
    "parasitic_component_db_id": "Parasitics",
    "parasitic_A_coef_N": "Parasitics",
    "parasitic_B_Npkph": "Parasitics",
    "parasitic_C_coef_Npkph2": "Parasitics",
}
_BASELINE_CORRECTION_FIELD_ORDER = {field_key: index for index, field_key in enumerate(_BASELINE_CORRECTION_DOMAIN_BY_FIELD)}
_PREVIEW_COMPARISON_GROUPS = (
    ("Mass", (
        ("mass_kg", "Curb mass", "mass_kg"),
        ("inertia_class", "EPA ETW / TWC", "inertia_class"),
        ("test_mass_kg", "Test mass", "test_mass_kg"),
        ("test_mass_basis", "Test mass basis", "test_mass_basis"),
        ("gvwr_kg", "GVWR", "gvwr_kg"),
        ("gcwr_kg", "GCWR", "gcwr_kg"),
        ("trailer_mass_kg", "Trailer mass", "trailer_mass_kg"),
    )),
    ("Aero", (
        ("CdA", "CdA", "CdA"),
    )),
    ("Tire", (
        ("rrc_N_per_kN", "RRC", "rrc_N_per_kN"),
        ("front_pressure_psi", "Front pressure", "front_pressure_psi"),
        ("rear_pressure_psi", "Rear pressure", "rear_pressure_psi"),
        ("tire_code", "Tire code", "tire_code"),
        ("tire_A_final", "Tire ABC A", "tire_A_final"),
        ("tire_B_final", "Tire ABC B", "tire_B_final"),
        ("tire_C_final", "Tire ABC C", "tire_C_final"),
    )),
    ("Transmission", (
        ("trans_A_coef_N", "A", "trans_A_coef_N"),
        ("trans_B_coef_Npkph", "B", "trans_B_coef_Npkph"),
        ("trans_C_coef_Npkph2", "C", "trans_C_coef_Npkph2"),
    )),
    ("Brake", (
        ("brake_A_coef_N", "A", "brake_A_coef_N"),
        ("brake_B_Npkph", "B", "brake_B_Npkph"),
        ("brake_C_coef_Npkph2", "C", "brake_C_coef_Npkph2"),
        ("residual_torque_total_Nm", "Residual torque", "residual_torque_total_Nm"),
    )),
    ("Axle & Hubs", (
        ("axle_hub_A", "A", "axle_hub_A"),
        ("axle_hub_B", "B", "axle_hub_B"),
        ("axle_hub_C", "C", "axle_hub_C"),
    )),
    ("Parasitics", (
        ("parasitic_A_coef_N", "A", "parasitic_A_coef_N"),
        ("parasitic_B_Npkph", "B", "parasitic_B_Npkph"),
        ("parasitic_C_coef_Npkph2", "C", "parasitic_C_coef_Npkph2"),
    )),
    ("Trailer", (
        ("trailer_A", "A", "trailer_A"),
        ("trailer_B", "B", "trailer_B"),
        ("trailer_C", "C", "trailer_C"),
    )),
    ("Resulting Roadload", (
        ("abc_total_A", "ABC_TOTAL A", "A"),
        ("abc_total_B", "ABC_TOTAL B", "B"),
        ("abc_total_C", "ABC_TOTAL C", "C"),
        ("abc_net_A", "ABC_NET A", "A"),
        ("abc_net_B", "ABC_NET B", "B"),
        ("abc_net_C", "ABC_NET C", "C"),
    )),
    ("VDE", (
        ("vde_total_mj_per_km", "VDE_TOTAL", "vde_total_mj_per_km"),
        ("vde_net_mj_per_km", "VDE_NET", "vde_net_mj_per_km"),
    )),
)
_COMPARISON_GROUP_DOMAIN = {
    "Mass": "mass",
    "Aero": "aero",
    "Tire": "tire",
    "Transmission": "transmission",
    "Brake": "brake",
    "Axle & Hubs": "axle_hubs",
    "Parasitics": "parasitic",
}
_COMPARISON_ALWAYS_VISIBLE_GROUPS = {"Resulting Roadload", "VDE"}

_COMPONENT_PREVIEW_FIELD_ALIASES = {
    "brake_A_coef_N": ("brake_A_coef_N", "brake_A"),
    "brake_B_Npkph": ("brake_B_Npkph", "brake_B"),
    "brake_C_coef_Npkph2": ("brake_C_coef_Npkph2", "brake_C"),
    "residual_torque_total_Nm": ("residual_torque_total_Nm",),
    "axle_hub_A": ("axle_hub_A",),
    "axle_hub_B": ("axle_hub_B",),
    "axle_hub_C": ("axle_hub_C",),
    "parasitic_A_coef_N": ("parasitic_A_coef_N", "parasitic_A"),
    "parasitic_B_Npkph": ("parasitic_B_Npkph", "parasitic_B"),
    "parasitic_C_coef_Npkph2": ("parasitic_C_coef_Npkph2", "parasitic_C"),
}


def build_v22_branding_payload(state: dict | None) -> dict:
    normalized = normalize_v22_state(state)
    baseline = dict(normalized.get("baseline") or {})
    if not baseline.get("loaded"):
        return {
            "loaded": False,
            "baseline_id": None,
            "proposal_count": len(list(normalized.get("proposals") or [])),
            "make": "",
            "legislation": "",
            "logo_path": None,
            "legislation_icon_path": None,
        }
    effective = dict(baseline.get("effective") or {})
    printed = dict(baseline.get("printed") or {})
    return {
        "loaded": True,
        "baseline_id": baseline.get("selected_vde_id"),
        "proposal_count": len(list(normalized.get("proposals") or [])),
        "make": str(effective.get("make") or printed.get("make") or ""),
        "legislation": str(effective.get("legislation") or printed.get("legislation") or ""),
        "logo_path": None,
        "legislation_icon_path": None,
    }


def build_v22_flow_status_payload(state: dict | None) -> dict:
    normalized = normalize_v22_state(state)
    active_section = str(normalized.get("active_section") or "baseline")
    baseline_step = _baseline_step_payload(normalized, active_section)
    matrix_step = _matrix_step_payload(normalized, active_section)
    inputs_step = _inputs_step_payload(normalized, active_section)
    preview_step = _preview_step_payload(normalized, active_section)
    steps = [baseline_step, matrix_step, inputs_step, preview_step]
    return {
        "active_section": active_section,
        "steps": steps,
        "baseline_loaded": bool(dict(normalized.get("baseline") or {}).get("loaded")),
        "proposal_count": len(list(normalized.get("proposals") or [])),
        "preview_status": str(dict(normalized.get("preview") or {}).get("status") or "not_run"),
        "context_strip": _context_strip_payload(normalized, steps),
        "validation_status": preview_step.get("base_status"),
        "save_status": "disabled",
    }


def proposal_display_label(state: dict | None, proposal: dict | None) -> str:
    normalized = normalize_v22_state(state)
    target_id = str(dict(proposal or {}).get("proposal_id") or "")
    for item in list(normalized.get("proposals") or []):
        if str(item.get("proposal_id") or "") != target_id:
            continue
        return f"Requested #{int(item.get('display_index') or 0)}"
    return target_id or "Requested"


def walk_from_display_label(state: dict | None, walk_from) -> str:
    value = str(walk_from or "baseline")
    if value == "baseline":
        return "Baseline"
    normalized = normalize_v22_state(state)
    for proposal in list(normalized.get("proposals") or []):
        if str(proposal.get("proposal_id") or "") == value:
            return proposal_display_label(normalized, proposal)
    return value


def build_v22_domain_status_payload(state: dict | None, domain: str) -> dict:
    normalized = normalize_v22_state(state)
    payload = deepcopy(dict(dict(normalized.get("domain_input_state") or {}).get(domain) or {}))
    proposal_statuses = dict(payload.get("proposal_statuses") or {})
    ready_count = 0
    incomplete_count = 0
    configured_count = 0
    for item in proposal_statuses.values():
        status = str(dict(item or {}).get("status") or "")
        if status in {"applied_ready", "applied_incomplete"}:
            configured_count += 1
        if status == "applied_ready":
            ready_count += 1
        elif status == "applied_incomplete":
            incomplete_count += 1
    return {
        "domain": str(domain or ""),
        "status": str(payload.get("status") or "not_configured"),
        "revision": int(payload.get("revision") or 0),
        "last_applied_at": payload.get("last_applied_at"),
        "last_apply_message": payload.get("last_apply_message"),
        "ready_count": ready_count,
        "incomplete_count": incomplete_count,
        "configured_count": configured_count,
    }


def build_request_inputs_overview_payload(state: dict | None) -> dict:
    normalized = normalize_v22_state(state)
    active_domains = _direct_domain_keys(normalized)
    buckets = _request_inputs_status_counts(normalized, active_domains)
    inactive_domains = [
        build_domain_card_payload(normalized, domain, "Metric")
        for domain in DOMAIN_LABELS
        if domain not in active_domains
    ]
    return {
        "active_domain_keys": list(active_domains),
        "active_domain_count": len(active_domains),
        "ready_count": buckets["ready"],
        "review_count": buckets["review"],
        "stale_count": buckets["stale"],
        "pending_count": buckets["pending"],
        "summary": _request_inputs_summary(len(active_domains), buckets),
        "active_domains": [build_domain_card_payload(normalized, domain, "Metric") for domain in active_domains],
        "inactive_domains": inactive_domains,
        "has_active_domains": bool(active_domains),
    }


def build_domain_card_payload(state: dict | None, domain: str, unit_system) -> dict:
    normalized = normalize_v22_state(state)
    proposals = list(normalized.get("proposals") or [])
    domain_state = dict(dict(normalized.get("domain_input_state") or {}).get(domain) or {})
    proposal_statuses = dict(domain_state.get("proposal_statuses") or {})
    proposal_summaries = []
    active_proposals = []
    inactive_labels = []
    walk_from_lines = []
    active_modes = []
    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id") or "")
        domain_payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
        selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
        walk_from_label = walk_from_display_label(normalized, proposal.get("walk_from") or "baseline")
        label = proposal_display_label(normalized, proposal)
        kind = _proposal_domain_kind(proposal_type, selection_mode)
        mode_label = _selection_mode_label(proposal_type, selection_mode)
        summary = {
            "proposal_id": proposal_id,
            "label": label,
            "walk_from_label": walk_from_label,
            "mode_label": mode_label,
            "kind": kind,
            "status_label": proposal_status_label(proposal_statuses.get(proposal_id)),
            "raw_status": str(dict(proposal_statuses.get(proposal_id) or {}).get("status") or ""),
        }
        proposal_summaries.append(summary)
        if kind == "active":
            active_proposals.append(summary)
            walk_from_lines.append(f"{label} <- {walk_from_label}")
            if mode_label not in active_modes:
                active_modes.append(mode_label)
        else:
            inactive_labels.append(f"{label} - {mode_label}")

    bucket = _request_inputs_status_bucket(domain_state)
    correction_fields = _baseline_correction_entries_for_domain(normalized, domain, unit_system)
    row_keys = rows_for_active_domain(domain, normalized)
    correction_field_keys = [item["field_key"] for item in correction_fields]
    for field_key in correction_field_keys:
        if field_key not in row_keys:
            row_keys.append(field_key)
    return {
        "domain": str(domain or ""),
        "label": DOMAIN_LABELS.get(domain, str(domain or "").replace("_", " ").title()),
        "is_active": bool(active_proposals),
        "status_key": bucket,
        "status_label": _request_inputs_status_label(bucket),
        "status_tone": bucket,
        "proposal_type_summary": " | ".join(active_modes) if active_modes else (_inactive_domain_mode_summary(proposal_summaries) or "Inherit"),
        "active_proposal_count": len(active_proposals),
        "proposal_summaries": proposal_summaries,
        "walk_from_lines": walk_from_lines,
        "inactive_summary": inactive_labels,
        "last_applied_at": domain_state.get("last_applied_at"),
        "last_apply_message": domain_state.get("last_apply_message"),
        "domain_state_status": str(domain_state.get("status") or "not_configured"),
        "revision": int(domain_state.get("revision") or 0),
        "reference_changes": correction_fields,
        "reference_change_count": len(correction_fields),
        "reference_caption": "Corrections modify the effective baseline used by all proposals.",
        "row_keys": row_keys,
        "correction_field_keys": correction_field_keys,
    }


def build_preview_status_payload(state: dict | None) -> dict:
    normalized = normalize_v22_state(state)
    preview = dict(normalized.get("preview") or {})
    bundle = dict(preview.get("result") or {})
    validation = dict(bundle.get("validation_summary") or {})
    save = dict(normalized.get("save") or {})
    baseline = dict(normalized.get("baseline") or {})
    pending_rows, incomplete_rows = _preview_pending_and_incomplete_rows(normalized)
    preview_status = str(preview.get("status") or "not_run")
    return {
        "preview_status": preview_status,
        "preview_label": _preview_state_label(preview_status),
        "validation_status": str(validation.get("overall_status") or "Pending"),
        "baseline_id": baseline.get("selected_vde_id"),
        "proposal_count": len(list(normalized.get("proposals") or [])),
        "review_count": int(validation.get("review_count") or 0),
        "save_status": _save_gate_label(preview_status, validation, save),
        "fingerprint": preview.get("fingerprint") or bundle.get("fingerprint"),
        "has_bundle": bool(bundle),
        "stale_message": "Preview is stale. Request inputs changed after this preview was generated." if preview_status == "stale" else "",
        "empty_message": "No preview generated yet. Apply the required domains and run Validate & Preview.",
        "pending_rows": pending_rows,
        "incomplete_rows": incomplete_rows,
        "baseline_correction_count": sum(1 for value in dict(baseline.get("corrections") or {}).values() if not is_blank(value)),
    }


def build_scenario_overview_payload(state: dict | None) -> dict:
    normalized = normalize_v22_state(state)
    bundle = dict(dict(normalized.get("preview") or {}).get("result") or {})
    if not bundle:
        return {"scenarios": [], "has_bundle": False}
    proposal_results_by_id = {
        str(item.get("proposal_id") or ""): dict(item)
        for item in list(dict(bundle.get("resolution_result") or {}).get("proposal_results") or [])
    }
    scenarios = [
        _baseline_scenario_card(normalized),
    ]
    for proposal in list(normalized.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        scenarios.append(_proposal_scenario_card(normalized, proposal, proposal_results_by_id.get(proposal_id)))
    comparison_scenarios = {
        str(item.get("id") or ""): item
        for item in _comparison_scenarios(normalized, bundle)
    }
    for scenario in scenarios:
        scenario_id = str(scenario.get("proposal_id") or scenario.get("id") or "")
        if scenario.get("label") == "Baseline":
            scenario_id = "baseline"
        comparison = comparison_scenarios.get(scenario_id)
        if comparison is None:
            continue
        scenario["metrics"] = _scenario_result_metrics(comparison)
        scenario["cycle_results"] = _scenario_cycle_results(comparison)
    return {"scenarios": scenarios, "has_bundle": True}


def build_vde_cycle_comparison_payload(state: dict | None, unit_system) -> dict:
    normalized = normalize_v22_state(state)
    bundle = dict(dict(normalized.get("preview") or {}).get("result") or {})
    if not bundle:
        return {"columns": [], "rows": [], "has_cycle_results": False}
    scenarios = _comparison_scenarios(normalized, bundle)
    phase_keys = []
    for scenario in scenarios:
        for item in _scenario_cycle_results(scenario):
            key = str(item.get("key") or "")
            if key and key not in phase_keys:
                phase_keys.append(key)
    if not phase_keys:
        return {"columns": [], "rows": [], "has_cycle_results": False}

    rows = []
    for key in phase_keys:
        for result_kind, label in (("total", "TOTAL"), ("net", "NET")):
            values = {
                scenario["id"]: _scenario_cycle_value(scenario, key, result_kind)
                for scenario in scenarios
            }
            if not _comparison_row_has_values(values):
                continue
            rows.append(
                {
                    "label": f"{_cycle_result_label(key)} {label}",
                    "display_values": {
                        scenario["id"]: _preview_display_value("vde_total_mj_per_km", value, unit_system)
                        for scenario, value in ((item, values[item["id"]]) for item in scenarios)
                    },
                }
            )
    return {
        "columns": [{"id": item["id"], "label": item["label"]} for item in scenarios],
        "rows": rows,
        "has_cycle_results": bool(rows),
    }


def build_engineering_comparison_payload(state: dict | None, unit_system) -> dict:
    normalized = normalize_v22_state(state)
    bundle = dict(dict(normalized.get("preview") or {}).get("result") or {})
    if not bundle:
        return {"columns": [], "groups": [], "has_bundle": False}
    scenarios = _comparison_scenarios(normalized, bundle)
    groups = []
    for group_label, specs in _PREVIEW_COMPARISON_GROUPS:
        rows = []
        for row_key, label, unit_field_key in specs:
            values = {
                scenario["id"]: _preview_scenario_raw_value(scenario, row_key)
                for scenario in scenarios
            }
            if not _comparison_row_has_values(values):
                continue
            rows.append(
                {
                    "group": group_label,
                    "field_key": row_key,
                    "label": label,
                    "unit": display_unit_for_field(unit_field_key, unit_system) or _UNAVAILABLE,
                    "display_values": {
                        scenario["id"]: _preview_display_value(unit_field_key, values.get(scenario["id"]), unit_system)
                        for scenario in scenarios
                    },
                    "raw_values": values,
                }
            )
        if rows:
            groups.append({"title": group_label, "rows": rows})
    changed_domains = {
        domain
        for proposal in list(normalized.get("proposals") or [])
        for domain in DOMAIN_LABELS
        if _proposal_has_direct_domain(proposal, domain)
    }
    return {
        "columns": [{"id": item["id"], "label": item["label"]} for item in scenarios],
        "groups": groups,
        "changed_group_titles": [
            group.get("title")
            for group in groups
            if group.get("title") in _COMPARISON_ALWAYS_VISIBLE_GROUPS
            or _COMPARISON_GROUP_DOMAIN.get(group.get("title")) in changed_domains
        ],
        "has_bundle": True,
    }


def build_roadload_analysis_payload(state: dict | None, unit_system, *, speed_max_kph: int = 140) -> dict:
    normalized = normalize_v22_state(state)
    preview = dict(normalized.get("preview") or {})
    bundle = dict(preview.get("result") or {})
    selected_speed_max = int(speed_max_kph) if int(speed_max_kph) in _ROADLOAD_SPEED_OPTIONS_KPH else 140
    display_system = normalize_unit_system(unit_system)
    base_payload = {
        "has_bundle": bool(bundle),
        "is_fresh": str(preview.get("status") or "") == "fresh",
        "speed_max_options_kph": list(_ROADLOAD_SPEED_OPTIONS_KPH),
        "selected_speed_max_kph": selected_speed_max,
        "speed_unit": unit_label("speed", display_system),
        "force_unit": unit_label("force", display_system),
        "checkpoint_speed_kph": list(_ROADLOAD_CHECKPOINTS_KPH),
        "series": [],
        "checkpoint_rows": [],
        "message": "",
    }
    if not bundle:
        base_payload["message"] = "Run Validate & Preview to build roadload curves."
        return base_payload
    if str(preview.get("status") or "") != "fresh":
        base_payload["message"] = "Preview is stale. Re-run Validate & Preview to update roadload curves."
        return base_payload

    scenarios = _comparison_scenarios(normalized, bundle)
    series = []
    for scenario in scenarios:
        for state_label, abc in _roadload_series_candidates(scenario):
            if not _abc_complete_triplet(abc):
                continue
            series.append(
                _build_roadload_series_payload(
                    scenario,
                    state_label,
                    abc,
                    display_system,
                    speed_max_kph=selected_speed_max,
                )
            )
    base_payload["series"] = series
    base_payload["checkpoint_rows"] = [
        {
            "Scenario": item["scenario_label"],
            "State": item["state_label"],
            **{
                f"{speed_kph} km/h": dict(item.get("checkpoint_display_map") or {}).get(speed_kph, _UNAVAILABLE)
                for speed_kph in _ROADLOAD_CHECKPOINTS_KPH
            },
        }
        for item in series
    ]
    if not series:
        base_payload["message"] = "No resolved ABC curves are available in the fresh preview."
    return base_payload


def build_cycle_power_analysis_payload(state: dict | None, *, selected_cycle: str | None = None) -> dict:
    normalized = normalize_v22_state(state)
    preview = dict(normalized.get("preview") or {})
    bundle = dict(preview.get("result") or {})
    payload = {
        "has_bundle": bool(bundle),
        "is_fresh": str(preview.get("status") or "") == "fresh",
        "cycle_options": [],
        "selected_cycle": "",
        "time_s": [],
        "speed_kph": [],
        "series": [],
        "decomposition_available": False,
        "decomposition_note": "",
        "message": "",
    }
    if not bundle:
        payload["message"] = "Run Validate & Preview to analyze cycle power."
        return payload
    if not payload["is_fresh"]:
        payload["message"] = "Preview is stale. Re-run Validate & Preview to update cycle power."
        return payload
    resolution = dict(bundle.get("resolution_result") or {})
    baseline = dict(dict(resolution.get("baseline") or {}).get("effective") or {})
    cycle_name = str(baseline.get("cycle_name") or baseline.get("cycle") or "").strip()
    if not cycle_name:
        payload["message"] = "No canonical cycle is available for this preview."
        return payload
    cycle_file_name = _canonical_cycle_file_name(cycle_name)
    try:
        cycle_frame = load_cycle_csv(cycle_file_name)
    except (FileNotFoundError, ValueError):
        payload["message"] = f"Canonical cycle data is unavailable for {cycle_name}."
        return payload
    segments = canonical_cycle_segments(cycle_frame)
    options = list(segments)
    if not options:
        payload["message"] = "No physical cycle segments are available."
        return payload
    selected = str(selected_cycle or "")
    if selected not in segments:
        selected = options[0]
    scenarios = []
    for scenario in _comparison_scenarios(normalized, bundle):
        total, net = _roadload_series_candidates(scenario)
        scenarios.append(
            {
                "id": scenario.get("id"),
                "label": scenario.get("label"),
                "mass_kg": _scenario_cycle_mass_kg(scenario),
                "total": total[1],
                "net": net[1],
            }
        )
    analysis = build_cycle_power_analysis(segments[selected], scenarios)
    payload.update(analysis)
    payload["cycle_options"] = options
    payload["selected_cycle"] = selected
    if not payload["series"]:
        payload["message"] = "No resolved TOTAL or NET ABC is available for cycle power analysis."
    return payload


def build_validation_summary_payload(state: dict | None, unit_system) -> dict:
    normalized = normalize_v22_state(state)
    bundle = dict(dict(normalized.get("preview") or {}).get("result") or {})
    if not bundle:
        pending_rows, incomplete_rows = _preview_pending_and_incomplete_rows(normalized)
        return {
            "summary": {},
            "scenario_sections": [],
            "root_issue_rows": [],
            "pending_rows": pending_rows,
            "incomplete_rows": incomplete_rows,
            "has_bundle": False,
        }
    resolution_result = dict(bundle.get("resolution_result") or {})
    validation = dict(bundle.get("validation_summary") or {})
    scenario_sections = []
    for proposal in list(resolution_result.get("proposal_results") or []):
        domain_rows = []
        for domain_key in DOMAIN_LABELS:
            payload = dict(dict(proposal.get("domain_results") or {}).get(domain_key) or {})
            if not payload:
                continue
            domain_rows.append(
                {
                    "Domain": DOMAIN_LABELS.get(domain_key, domain_key),
                    "Status": str(payload.get("status") or _UNAVAILABLE),
                    "Proposal type": str(payload.get("proposal_type") or "INHERIT"),
                }
            )
        issue_rows = [
            {
                "Severity": item.get("severity"),
                "Domain": item.get("domain"),
                "Field": item.get("field_key"),
                "Message": format_v22_issue_for_display(item, unit_system),
            }
            for item in list(proposal.get("issues") or [])
        ]
        preview_warnings = list(dict(proposal.get("preview_summary") or {}).get("warnings") or [])
        for warning in preview_warnings:
            issue_rows.append(
                {
                    "Severity": "warning",
                    "Domain": _UNAVAILABLE,
                    "Field": _UNAVAILABLE,
                    "Message": str(warning),
                }
            )
        scenario_sections.append(
            {
                "label": str(proposal.get("source_column") or proposal.get("proposal_id") or "Requested"),
                "status": str(proposal.get("status") or "OK"),
                "domain_rows": domain_rows,
                "issue_rows": issue_rows,
            }
        )
    pending_rows, incomplete_rows = _preview_pending_and_incomplete_rows(normalized)
    return {
        "summary": {
            "overall_status": str(validation.get("overall_status") or "Pending"),
            "ready_count": int(validation.get("ok_count") or 0),
            "proposal_count": int(validation.get("proposal_count") or 0),
            "review_count": int(validation.get("review_count") or 0),
            "invalid_count": int(validation.get("invalid_count") or 0),
            "missing_count": int(validation.get("missing_count") or 0),
        },
        "scenario_sections": scenario_sections,
        "root_issue_rows": [
            {
                "Severity": item.get("severity"),
                "Domain": item.get("domain"),
                "Field": item.get("field_key"),
                "Message": format_v22_issue_for_display(item, unit_system),
            }
            for item in list(resolution_result.get("issues") or [])
        ],
        "pending_rows": pending_rows,
        "incomplete_rows": incomplete_rows,
        "has_bundle": True,
    }


def build_preview_audit_payload(state: dict | None, unit_system) -> dict:
    normalized = normalize_v22_state(state)
    bundle = dict(dict(normalized.get("preview") or {}).get("result") or {})
    if not bundle:
        return {"has_bundle": False, "audit_rows": [], "proposal_models": [], "baseline_corrections": {}}
    resolution_result = dict(bundle.get("resolution_result") or {})
    audit_rows = []
    for proposal in list(resolution_result.get("proposal_results") or []):
        for domain_key, payload in dict(proposal.get("domain_results") or {}).items():
            item = dict(payload or {})
            audit_rows.append(
                {
                    "Scenario": str(proposal.get("source_column") or proposal.get("proposal_id") or "Requested"),
                    "Proposal ID": str(proposal.get("proposal_id") or _UNAVAILABLE),
                    "Domain": DOMAIN_LABELS.get(domain_key, domain_key),
                    "Status": str(item.get("status") or _UNAVAILABLE),
                    "Walk From": str(dict(proposal.get("walk_from") or {}).get("label") or dict(proposal.get("walk_from") or {}).get("column_id") or _UNAVAILABLE),
                    "Requested": format_value_map_for_display(item.get("requested_values"), unit_system, unavailable=_UNAVAILABLE),
                    "Resolved": format_value_map_for_display(item.get("resolved_values"), unit_system, unavailable=_UNAVAILABLE),
                    "Issues": str(len(list(item.get("issues") or []))),
                    "Source": str(item.get("source") or _UNAVAILABLE),
                }
            )
    return {
        "has_bundle": True,
        "audit_rows": audit_rows,
        "proposal_models": list(bundle.get("proposal_models") or []),
        "fingerprint": bundle.get("fingerprint"),
        "baseline_corrections": build_active_corrections_summary(normalized, unit_system),
    }


def build_baseline_candidate_status_payload(state: dict | None, selected_vde_id, *, selected_label: str | None = None) -> dict:
    normalized = normalize_v22_state(state)
    baseline = dict(normalized.get("baseline") or {})
    printed = dict(baseline.get("printed") or {})
    effective = dict(baseline.get("effective") or {})
    loaded = bool(baseline.get("loaded"))
    loaded_vde_id = baseline.get("selected_vde_id")
    selected_present = not is_blank(selected_vde_id)
    differs = loaded and selected_present and str(selected_vde_id) != str(loaded_vde_id)
    return {
        "loaded": loaded,
        "selected_candidate_id": selected_vde_id,
        "selected_candidate_label": str(selected_label or ""),
        "loaded_baseline_id": loaded_vde_id,
        "loaded_baseline_label": _baseline_loaded_label(effective, printed),
        "status": "Loaded" if loaded else "Pending",
        "candidate_differs": differs,
        "warning_message": (
            f"Selected candidate differs from the loaded baseline. The current request continues using VDE #{loaded_vde_id} until Load baseline is pressed."
            if differs and not is_blank(loaded_vde_id)
            else ""
        ),
    }


def build_loaded_baseline_summary_payload(state: dict | None, unit_system) -> dict:
    normalized = normalize_v22_state(state)
    baseline = dict(normalized.get("baseline") or {})
    if not baseline.get("loaded"):
        return {"loaded": False, "groups": [], "notes": "", "baseline_id": None, "status": "Pending"}

    printed = dict(baseline.get("printed") or {})
    effective = dict(baseline.get("effective") or {})
    groups = [
        {
            "title": "Vehicle",
            "items": [
                {"label": "Make", "value": _display_text_value(effective.get("make") or printed.get("make"))},
                {"label": "Model", "value": _display_text_value(effective.get("model") or printed.get("model"))},
                {"label": "Model Year", "value": _display_text_value(effective.get("year") or printed.get("year"))},
                {"label": "Category", "value": _display_text_value(effective.get("category") or printed.get("category"))},
            ],
        },
        {
            "title": "Regulation",
            "items": [
                {"label": "Legislation", "value": _display_text_value(effective.get("legislation") or printed.get("legislation"))},
                {"label": "Cycle", "value": _display_text_value(effective.get("cycle_name") or printed.get("cycle_name"))},
            ],
        },
        {
            "title": "Mass",
            "items": [
                {"label": "Curb mass", "value": _display_field_value("mass_kg", effective.get("mass_kg"), unit_system)},
                {"label": "EPA ETW / TWC", "value": _display_field_value("inertia_class", effective.get("inertia_class"), unit_system)},
                {"label": "Test mass", "value": _display_field_value("test_mass_kg", effective.get("test_mass_kg"), unit_system)},
                {"label": "Test mass basis", "value": _display_text_value(effective.get("test_mass_basis"))},
            ],
        },
        {
            "title": "Roadload",
            "items": [
                {"label": "A", "value": _display_field_value("A", effective.get("A"), unit_system)},
                {"label": "B", "value": _display_field_value("B", effective.get("B"), unit_system)},
                {"label": "C", "value": _display_field_value("C", effective.get("C"), unit_system)},
            ],
        },
        {
            "title": "VDE",
            "items": [
                {"label": "VDE_TOTAL", "value": _display_field_value("vde_total_mj_per_km", effective.get("vde_total_mj_per_km"), unit_system)},
                {"label": "VDE_NET", "value": _display_field_value("vde_net_mj_per_km", effective.get("vde_net_mj_per_km"), unit_system)},
            ],
        },
    ]
    return {
        "loaded": True,
        "baseline_id": baseline.get("selected_vde_id"),
        "status": "Loaded",
        "groups": groups,
        "notes": _display_text_value(effective.get("notes") or printed.get("notes"), unavailable=""),
    }


def build_active_corrections_summary(state: dict | None, unit_system) -> dict:
    normalized = normalize_v22_state(state)
    baseline = dict(normalized.get("baseline") or {})
    printed = dict(baseline.get("printed") or {})
    effective = dict(baseline.get("effective") or {})
    corrections = dict(baseline.get("corrections") or {})
    entries = []
    for field_key, value in sorted(
        corrections.items(),
        key=lambda item: (_BASELINE_CORRECTION_FIELD_ORDER.get(str(item[0]), 9999), str(item[0])),
    ):
        if is_blank(value):
            continue
        domain = _BASELINE_CORRECTION_DOMAIN_BY_FIELD.get(str(field_key), "Baseline")
        label = str(field_meta(field_key).get("label") or field_key)
        entries.append(
            {
                "domain": domain,
                "field_key": str(field_key),
                "field_label": label,
                "printed_value": _display_field_value(field_key, printed.get(field_key), unit_system),
                "effective_value": _display_field_value(field_key, effective.get(field_key), unit_system),
            }
        )
    domain_rows = []
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry["domain"], []).append(entry["field_label"])
    for domain, labels in grouped.items():
        domain_rows.append({"domain": domain, "fields": labels})
    return {
        "count": len(entries),
        "entries": entries,
        "domain_rows": domain_rows,
        "empty_message": "No active baseline corrections.",
    }


def format_v22_issue_for_display(issue, unit_system) -> str:
    payload = deepcopy(dict(issue or {}))
    message = str(payload.get("message") or payload.get("Message") or "").strip()
    field_key = str(payload.get("field_key") or payload.get("field") or payload.get("Field") or "").strip()
    if not field_key or quantity_kind_for_field(field_key) is None:
        return message
    if not any(not is_blank(payload.get(key)) for key in _STRUCTURED_ISSUE_VALUE_KEYS):
        return message

    label = str(field_meta(field_key).get("label") or field_key.replace("_", " ").title())
    actual = payload.get("actual", payload.get("value"))
    expected = payload.get("expected")
    min_value = payload.get("min")
    max_value = payload.get("max")

    if not is_blank(actual) and not is_blank(min_value) and not is_blank(max_value):
        return f"{label} {_issue_value_text(field_key, actual, unit_system)} is outside the allowed interval {_issue_interval_text(field_key, min_value, max_value, unit_system)}."
    if not is_blank(actual) and not is_blank(expected):
        return f"{label} {_issue_value_text(field_key, actual, unit_system)} does not match expected {_issue_value_text(field_key, expected, unit_system)}."
    if not is_blank(actual) and not is_blank(min_value):
        return f"{label} {_issue_value_text(field_key, actual, unit_system)} must be at least {_issue_value_text(field_key, min_value, unit_system)}."
    if not is_blank(actual) and not is_blank(max_value):
        return f"{label} {_issue_value_text(field_key, actual, unit_system)} must be at most {_issue_value_text(field_key, max_value, unit_system)}."
    return message


def step_payload_by_key(state: dict | None, section_key: str) -> dict:
    flow = build_v22_flow_status_payload(state)
    for item in list(flow.get("steps") or []):
        if str(item.get("key") or "") == str(section_key or ""):
            return item
    return {}


def _baseline_step_payload(state: dict, active_section: str) -> dict:
    baseline = dict(state.get("baseline") or {})
    loaded = bool(baseline.get("loaded"))
    status = "pending"
    summary = "No baseline loaded"
    detail = "Load a baseline to start the request flow."
    if loaded:
        status = "complete"
        loaded_id = baseline.get("selected_vde_id")
        summary = f"VDE #{loaded_id}" if loaded_id not in (None, "") else "Baseline loaded"
        detail = "Technical baseline loaded and ready for proposal configuration."
    return _step_payload("baseline", active_section, status=status, summary=summary, detail=detail)


def _matrix_step_payload(state: dict, active_section: str) -> dict:
    baseline_loaded = bool(dict(state.get("baseline") or {}).get("loaded"))
    direct_domains = _direct_domain_keys(state)
    invalid_walk_from = _invalid_walk_from_labels(state)
    if not baseline_loaded:
        return _step_payload(
            "matrix",
            active_section,
            status="pending",
            summary="Waiting for baseline",
            detail="Load a baseline before configuring proposal relationships.",
        )
    if invalid_walk_from:
        return _step_payload(
            "matrix",
            active_section,
            status="review",
            summary=f"{len(direct_domains)} direct domains",
            detail="Invalid Walk From: " + "; ".join(invalid_walk_from),
        )
    if not direct_domains:
        return _step_payload(
            "matrix",
            active_section,
            status="pending",
            summary="No direct proposals",
            detail="Choose at least one direct domain to make the matrix actionable.",
        )
    return _step_payload(
        "matrix",
        active_section,
        status="complete",
        summary=f"{len(direct_domains)} direct domains",
        detail="Walk From references are valid for the configured proposal modes.",
    )


def _inputs_step_payload(state: dict, active_section: str) -> dict:
    baseline_loaded = bool(dict(state.get("baseline") or {}).get("loaded"))
    direct_domains = _direct_domain_keys(state)
    if not baseline_loaded or not direct_domains:
        return _step_payload(
            "inputs",
            active_section,
            status="pending",
            summary="0/0 domains ready" if baseline_loaded else "Waiting for baseline",
            detail="Direct proposal domains will appear here after the matrix is configured.",
        )

    ready = []
    incomplete = []
    stale = []
    pending = []
    domain_input_state = dict(state.get("domain_input_state") or {})
    for domain in direct_domains:
        bucket = _request_inputs_status_bucket(dict(domain_input_state.get(domain) or {}))
        if bucket == "ready":
            ready.append(domain)
        elif bucket == "review":
            incomplete.append(domain)
        elif bucket == "stale":
            stale.append(domain)
        else:
            pending.append(domain)

    summary = f"{len(ready)}/{len(direct_domains)} inputs applied"
    if stale:
        return _step_payload(
            "inputs",
            active_section,
            status="stale",
            summary=summary,
            detail=_join_domain_detail(stale, suffix="stale"),
        )
    if incomplete:
        return _step_payload(
            "inputs",
            active_section,
            status="review",
            summary=summary,
            detail=_join_domain_detail(incomplete, suffix="incomplete"),
        )
    if pending:
        return _step_payload(
            "inputs",
            active_section,
            status="pending",
            summary=summary,
            detail=_join_domain_detail(pending, suffix="pending apply"),
        )
    return _step_payload(
        "inputs",
        active_section,
        status="complete",
        summary=summary,
        detail="All direct domains are applied.",
    )


def _preview_step_payload(state: dict, active_section: str) -> dict:
    preview = dict(state.get("preview") or {})
    preview_status = str(preview.get("status") or "not_run")
    bundle = dict(preview.get("result") or {})
    validation = dict(bundle.get("validation_summary") or {})
    if preview_status == "stale":
        return _step_payload(
            "preview",
            active_section,
            status="stale",
            summary="Preview stale",
            detail="Re-run Validate & Preview after matrix or input changes.",
        )
    if preview_status in {"not_run", ""} or not bundle:
        return _step_payload(
            "preview",
            active_section,
            status="pending",
            summary="Preview not generated",
            detail="Run Validate & Preview to build the canonical draft and resolver output.",
        )
    overall = str(validation.get("overall_status") or "OK")
    if overall != "OK":
        return _step_payload(
            "preview",
            active_section,
            status="review",
            summary=f"Validation {overall}",
            detail=f"{validation.get('review_count', 0)} review; {validation.get('missing_count', 0)} missing",
        )
    return _step_payload(
        "preview",
        active_section,
        status="complete",
        summary="Preview ready",
        detail="Preview fresh; validation clear.",
    )


def _step_payload(section_key: str, active_section: str, *, status: str, summary: str, detail: str) -> dict:
    is_active = section_key == active_section
    rendered_status = "active" if is_active else status
    return {
        "key": section_key,
        "index": SECTION_ORDER.index(section_key) + 1,
        "label": SECTION_LABELS[section_key],
        "caption": SECTION_CAPTIONS[section_key],
        "status": rendered_status,
        "base_status": status,
        "icon": SECTION_ICONS[rendered_status],
        "summary": summary,
        "detail": detail,
        "is_active": is_active,
    }


def _context_strip_payload(state: dict, steps: list[dict]) -> list[dict]:
    # Preview and save state belongs to Preview & Save, not the global header.
    baseline = dict(state.get("baseline") or {})
    loaded_id = baseline.get("selected_vde_id")
    return [
        {"label": "Baseline", "value": f"VDE #{loaded_id}" if loaded_id not in (None, "") else "Not loaded"},
        {"label": "Proposals", "value": str(len(list(state.get("proposals") or [])))},
    ]


def _direct_domain_keys(state: dict) -> list[str]:
    normalized = normalize_v22_state(state)
    ordered = []
    for domain in DOMAIN_LABELS:
        for proposal in list(normalized.get("proposals") or []):
            payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
            proposal_type = str(payload.get("proposal_type") or "INHERIT")
            selection_mode = str(payload.get("selection_mode") or proposal_type)
            if proposal_type == "INHERIT" or proposal_is_not_used(proposal_type, selection_mode):
                continue
            if domain not in ordered:
                ordered.append(domain)
            break
    return ordered


def _request_inputs_status_counts(state: dict, domains: list[str]) -> dict:
    counts = {"ready": 0, "review": 0, "stale": 0, "pending": 0}
    domain_input_state = dict(state.get("domain_input_state") or {})
    for domain in list(domains or []):
        counts[_request_inputs_status_bucket(dict(domain_input_state.get(domain) or {}))] += 1
    return counts


def _request_inputs_status_bucket(domain_state: dict | None) -> str:
    payload = dict(domain_state or {})
    status = str(payload.get("status") or "not_configured")
    if status == "applied_ready":
        return "ready"
    if status == "applied_incomplete":
        return "review"
    if status == "stale_after_matrix_change":
        if int(payload.get("revision") or 0) > 0 or payload.get("last_applied_at"):
            return "stale"
    return "pending"


def _request_inputs_status_label(bucket: str) -> str:
    return {
        "ready": "Applied",
        "review": "Incomplete",
        "stale": "Pending apply",
        "pending": "Pending apply",
    }.get(str(bucket or "pending"), "Pending apply")


def _request_inputs_summary(active_count: int, buckets: dict) -> str:
    parts = [f"{active_count} direct domains", f"{int(buckets.get('ready') or 0)} applied"]
    incomplete = int(buckets.get("review") or 0)
    pending = int(buckets.get("pending") or 0) + int(buckets.get("stale") or 0)
    if incomplete:
        parts.append(f"{incomplete} incomplete")
    if pending:
        parts.append(f"{pending} pending")
    return " | ".join(parts)


def _proposal_domain_kind(proposal_type: str, selection_mode: str) -> str:
    if str(proposal_type or "").strip().upper() == "INHERIT":
        return "inherit"
    if proposal_is_not_used(proposal_type, selection_mode):
        return "not_used"
    return "active"


def _selection_mode_label(proposal_type: str, selection_mode: str) -> str:
    if str(proposal_type or "").strip().upper() == "INHERIT":
        return "Inherit"
    if proposal_is_not_used(proposal_type, selection_mode):
        return "Not used"
    return str(selection_mode or proposal_type or "").strip() or "Inherit"


def _inactive_domain_mode_summary(proposal_summaries: list[dict]) -> str:
    labels = []
    for item in list(proposal_summaries or []):
        mode_label = str(dict(item).get("mode_label") or "").strip()
        if mode_label and mode_label not in labels:
            labels.append(mode_label)
    return " | ".join(labels)


def _baseline_correction_entries_for_domain(state: dict, domain: str, unit_system) -> list[dict]:
    normalized = normalize_v22_state(state)
    baseline = dict(normalized.get("baseline") or {})
    printed = dict(baseline.get("printed") or {})
    effective = dict(baseline.get("effective") or {})
    corrections = dict(baseline.get("corrections") or {})
    target_label = DOMAIN_LABELS.get(domain, str(domain or "").replace("_", " ").title())
    entries = []
    field_keys = [
        field_key
        for field_key, label in _BASELINE_CORRECTION_DOMAIN_BY_FIELD.items()
        if label == target_label
    ]
    for field_key in sorted(
        field_keys,
        key=lambda key: (_BASELINE_CORRECTION_FIELD_ORDER.get(str(key), 9999), str(key)),
    ):
        if _BASELINE_CORRECTION_DOMAIN_BY_FIELD.get(str(field_key)) != target_label:
            continue
        entries.append(
            {
                "field_key": str(field_key),
                "field_label": str(field_meta(field_key).get("label") or field_key),
                "printed_value": _display_field_value(field_key, printed.get(field_key), unit_system),
                "effective_value": _display_field_value(field_key, effective.get(field_key), unit_system),
                "unit": display_unit_for_field(field_key, unit_system),
            }
        )
    return entries


def _preview_state_label(status: str) -> str:
    mapping = {
        "fresh": "Fresh",
        "stale": "Stale",
        "not_run": "Not generated",
        "": "Not generated",
    }
    return mapping.get(str(status or ""), str(status or "Not generated").replace("_", " ").title())


def _preview_pending_and_incomplete_rows(state: dict) -> tuple[list[dict], list[dict]]:
    normalized = normalize_v22_state(state)
    not_applied = []
    incomplete = []
    domain_input_state = dict(normalized.get("domain_input_state") or {})
    for proposal in list(normalized.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        for domain in DOMAIN_LABELS:
            payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
            proposal_type = str(payload.get("proposal_type") or "INHERIT")
            selection_mode = str(payload.get("selection_mode") or proposal_type)
            if proposal_type == "INHERIT" or proposal_is_not_used(proposal_type, selection_mode):
                continue
            domain_state = dict(domain_input_state.get(domain) or {})
            proposal_status = dict(dict(domain_state.get("proposal_statuses") or {}).get(proposal_id) or {})
            status = str(proposal_status.get("status") or "not_configured")
            row = {
                "Proposal": proposal_display_label(normalized, proposal),
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


def _baseline_scenario_card(state: dict) -> dict:
    normalized = normalize_v22_state(state)
    baseline = dict(normalized.get("baseline") or {})
    corrections = dict(baseline.get("corrections") or {})
    changed = []
    seen = set()
    for field_key, value in corrections.items():
        if is_blank(value):
            continue
        domain = _BASELINE_CORRECTION_DOMAIN_BY_FIELD.get(str(field_key), "Baseline")
        if domain in seen:
            continue
        changed.append(domain)
        seen.add(domain)
    return {
        "id": "baseline",
        "label": "Baseline",
        "walk_from": _UNAVAILABLE,
        "reference_id": f"VDE #{baseline.get('selected_vde_id')}" if baseline.get("selected_vde_id") not in (None, "") else "",
        "status": "Reference",
        "changes": changed,
        "inherited": [],
        "not_used": [],
        "review": [],
        "missing": [],
    }


def _proposal_scenario_card(state: dict, proposal: dict, proposal_result: dict | None) -> dict:
    normalized = normalize_v22_state(state)
    result = dict(proposal_result or {})
    changes = []
    inherited = []
    not_used = []
    review = []
    missing = []
    for domain in DOMAIN_LABELS:
        payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        proposal_type = str(payload.get("proposal_type") or "INHERIT")
        selection_mode = str(payload.get("selection_mode") or proposal_type)
        domain_result = dict(dict(result.get("domain_results") or {}).get(domain) or {})
        domain_status = str(domain_result.get("status") or "")
        label = DOMAIN_LABELS[domain]
        if proposal_type == "INHERIT":
            inherited.append(label)
        elif proposal_is_not_used(proposal_type, selection_mode):
            not_used.append(label)
        else:
            changes.append(f"{label} · {selection_mode}")
        if domain_status == "Review":
            review.append(label)
        elif domain_status in {"Missing", "Invalid", "Blocked"}:
            missing.append(label)
    return {
        "id": str(proposal.get("proposal_id") or ""),
        "label": proposal_display_label(normalized, proposal),
        "walk_from": walk_from_display_label(normalized, proposal.get("walk_from") or "baseline"),
        "status": str(result.get("status") or "Pending"),
        "changes": changes,
        "inherited": inherited,
        "not_used": not_used,
        "review": review,
        "missing": missing,
    }


def _scenario_result_metrics(scenario: dict) -> list[dict]:
    def display_abc(prefix: str) -> str:
        return " / ".join(
            _overview_abc_display(_preview_scenario_raw_value(scenario, field_key))
            for field_key in (f"{prefix}_A", f"{prefix}_B", f"{prefix}_C")
        )

    return [
        {
            "label": "Curb mass",
            "value": _preview_display_value("mass_kg", _preview_scenario_raw_value(scenario, "mass_kg"), "Metric"),
        },
        {
            "label": "VDE mass",
            "value": _preview_display_value(
                "vde_calculation_mass_kg",
                _preview_scenario_raw_value(scenario, "vde_calculation_mass_kg"),
                "Metric",
            ),
        },
        {
            "label": "VDE TOTAL",
            "value": _preview_display_value(
                "vde_total_mj_per_km",
                _preview_scenario_raw_value(scenario, "vde_total_mj_per_km"),
                "Metric",
            ),
        },
        {
            "label": "VDE NET",
            "value": _preview_display_value(
                "vde_net_mj_per_km",
                _preview_scenario_raw_value(scenario, "vde_net_mj_per_km"),
                "Metric",
            ),
        },
        {"label": "ABC TOTAL", "value": display_abc("abc_total")},
        {"label": "ABC NET", "value": display_abc("abc_net")},
    ]


def _overview_abc_display(value) -> str:
    if is_blank(value):
        return _UNAVAILABLE
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _scenario_cycle_results(scenario: dict) -> list[dict]:
    keys = []
    for result_kind in ("total", "net"):
        result = _scenario_vde_result(scenario, result_kind)
        for key in dict(result.get("by_phase") or {}):
            if key not in keys:
                keys.append(key)
    if keys:
        keys.append("combined")
    return [
        {
            "key": key,
            "label": _cycle_result_label(key),
            "total": _scenario_cycle_value(scenario, key, "total"),
            "net": _scenario_cycle_value(scenario, key, "net"),
        }
        for key in keys
    ]


def _scenario_cycle_value(scenario: dict, key: str, result_kind: str):
    result = _scenario_vde_result(scenario, result_kind)
    if key == "combined":
        return result.get("mj_per_km")
    return dict(result.get("by_phase") or {}).get(key)


def _scenario_vde_result(scenario: dict, result_kind: str) -> dict:
    if str(scenario.get("id") or "") == "baseline":
        return dict(dict(scenario.get("baseline_resolved") or {}).get(f"vde_{result_kind}") or {})
    return dict(dict(dict(scenario.get("proposal_result") or {}).get("vde_results") or {}).get(result_kind) or {})


def _cycle_result_label(key: str) -> str:
    return {
        "city": "FTP-75",
        "ftp75": "FTP-75",
        "ftp-75": "FTP-75",
        "hwy": "HWFET",
        "hwfet": "HWFET",
        "low": "Low",
        "mid": "Medium",
        "medium": "Medium",
        "high": "High",
        "xhigh": "Extra High",
        "extra_high": "Extra High",
        "combined": "Combined",
    }.get(str(key or "").strip().lower(), str(key or "").replace("_", " ").title())


def _proposal_has_direct_domain(proposal: dict, domain: str) -> bool:
    payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
    proposal_type = str(payload.get("proposal_type") or "INHERIT")
    selection_mode = str(payload.get("selection_mode") or proposal_type)
    return proposal_type != "INHERIT" and not proposal_is_not_used(proposal_type, selection_mode)


def _comparison_scenarios(state: dict, bundle: dict) -> list[dict]:
    normalized = normalize_v22_state(state)
    resolution = dict(bundle.get("resolution_result") or {})
    scenarios = [
        {
            "id": "baseline",
            "label": "Effective Baseline",
            "baseline_effective": dict(dict(resolution.get("baseline") or {}).get("effective") or {}),
            "baseline_resolved": dict(dict(resolution.get("resolved_columns") or {}).get("baseline") or {}),
        }
    ]
    proposal_results_by_id = {
        str(item.get("proposal_id") or ""): dict(item)
        for item in list(resolution.get("proposal_results") or [])
    }
    for proposal in list(normalized.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        result = dict(proposal_results_by_id.get(proposal_id) or {})
        scenarios.append(
            {
                "id": proposal_id,
                "label": proposal_display_label(normalized, proposal),
                "proposal_result": result,
                "resolved_snapshot": dict(result.get("resolved_snapshot") or {}),
            }
        )
    return scenarios


def _comparison_row_has_values(values: dict[str, object]) -> bool:
    return any(not is_blank(value) for value in values.values())


def _roadload_series_candidates(scenario: dict) -> list[tuple[str, dict | None]]:
    if str(scenario.get("id")) == "baseline":
        resolved = dict(scenario.get("baseline_resolved") or {})
        total = dict(resolved.get("initial_abc_total") or resolved.get("abc_total") or {})
        net = dict(resolved.get("abc_net") or {}) if resolved.get("abc_net") is not None else None
        return [("TOTAL", total), ("NET", net)]
    result = dict(scenario.get("proposal_result") or {})
    total = dict(result.get("abc_total") or {})
    net = dict(result.get("abc_net") or {}) if result.get("abc_net") is not None else None
    return [("TOTAL", total), ("NET", net)]


def _canonical_cycle_file_name(cycle_name: str) -> str:
    normalized = str(cycle_name or "").strip().upper().replace("-", "").replace("_", "")
    if normalized in {"FTP75", "FTP753BAGS"}:
        return "FTP75"
    if normalized == "HWFET":
        return "HWFET"
    if normalized in {"WLTPCLASS3AB", "WLTP3AB"}:
        return "WLTP_Class3ab"
    return str(cycle_name or "").strip()


def _scenario_cycle_mass_kg(scenario: dict) -> float | None:
    if str(scenario.get("id")) == "baseline":
        source = dict(scenario.get("baseline_resolved") or {})
        fallback = dict(scenario.get("baseline_effective") or {})
    else:
        source = dict(scenario.get("resolved_snapshot") or {})
        fallback = {}
    for field_name in ("test_mass_kg", "mass_kg"):
        value = source.get(field_name, fallback.get(field_name))
        if is_blank(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _abc_complete_triplet(abc: dict | None) -> bool:
    triplet = dict(abc or {})
    return all(not is_blank(triplet.get(key)) for key in ("A", "B", "C"))


def _build_roadload_series_payload(
    scenario: dict,
    state_label: str,
    abc: dict,
    unit_system: str,
    *,
    speed_max_kph: int,
) -> dict:
    scenario_id = str(scenario.get("id") or "")
    scenario_label = "Baseline" if scenario_id == "baseline" else str(scenario.get("label") or scenario_id or "Requested")
    curve = build_roadload_curve(abc, speed_min_kph=0, speed_max_kph=speed_max_kph, step_kph=1)
    speed_kph = [float(value) for value in list(curve.get("speed_kph") or [])]
    force_N = [float(value) for value in list(curve.get("force_N") or [])]
    speed_display = [float(to_display(value, "speed", unit_system)) for value in speed_kph]
    force_display = [float(to_display(value, "force", unit_system)) for value in force_N]
    checkpoint_map = {}
    checkpoint_display_map = {}
    for speed_kph_value in _ROADLOAD_CHECKPOINTS_KPH:
        force_value = float(roadload_force_N(abc.get("A"), abc.get("B"), abc.get("C"), speed_kph_value))
        checkpoint_map[speed_kph_value] = force_value
        checkpoint_display_map[speed_kph_value] = format_quantity(
            force_value,
            "force",
            unit_system,
            include_unit=False,
            unavailable=_UNAVAILABLE,
        )
    return {
        "series_id": f"{scenario_id or 'baseline'}_{state_label.lower()}",
        "scenario_id": scenario_id or "baseline",
        "scenario_label": scenario_label,
        "state_label": state_label,
        "legend_label": f"{scenario_label} {state_label}",
        "line_dash": "solid" if state_label == "TOTAL" else "dash",
        "abc": {
            "A": float(abc.get("A")),
            "B": float(abc.get("B")),
            "C": float(abc.get("C")),
        },
        "speed_kph": speed_kph,
        "force_N": force_N,
        "speed_display": speed_display,
        "force_display": force_display,
        "checkpoint_force_map_N": checkpoint_map,
        "checkpoint_display_map": checkpoint_display_map,
    }


def _preview_scenario_raw_value(scenario: dict, row_key: str):
    if str(scenario.get("id")) == "baseline":
        return _baseline_preview_raw_value(scenario, row_key)
    return _proposal_preview_raw_value(scenario, row_key)


def _baseline_preview_raw_value(scenario: dict, row_key: str):
    effective = dict(scenario.get("baseline_effective") or {})
    resolved = dict(scenario.get("baseline_resolved") or {})
    if row_key.startswith("trans_"):
        return _transmission_preview_raw_value(resolved, effective, row_key)
    component_value = _component_preview_raw_value(resolved, effective, row_key)
    if component_value is not None:
        return component_value
    if row_key == "CdA":
        return resolved.get("CdA", effective.get("cda_m2"))
    if row_key.startswith("abc_total_"):
        return dict(resolved.get("initial_abc_total") or resolved.get("abc_total") or {}).get(row_key.rsplit("_", 1)[-1])
    if row_key.startswith("abc_net_"):
        return dict(resolved.get("abc_net") or {}).get(row_key.rsplit("_", 1)[-1])
    if row_key == "vde_total_mj_per_km":
        payload = resolved.get("vde_total") if isinstance(resolved.get("vde_total"), dict) else None
        if payload is None:
            payload = effective.get("vde_total") if isinstance(effective.get("vde_total"), dict) else None
        return (payload or {}).get("mj_per_km", resolved.get("vde_total_mj_per_km", effective.get("vde_total_mj_per_km")))
    if row_key == "vde_net_mj_per_km":
        payload = resolved.get("vde_net") if isinstance(resolved.get("vde_net"), dict) else None
        if payload is None:
            payload = effective.get("vde_net") if isinstance(effective.get("vde_net"), dict) else None
        return (payload or {}).get("mj_per_km", resolved.get("vde_net_mj_per_km", effective.get("vde_net_mj_per_km")))
    return resolved.get(row_key, effective.get(row_key))


def _proposal_preview_raw_value(scenario: dict, row_key: str):
    result = dict(scenario.get("proposal_result") or {})
    snapshot = dict(scenario.get("resolved_snapshot") or {})
    if row_key.startswith("trans_"):
        return _transmission_preview_raw_value(snapshot, {}, row_key)
    component_value = _component_preview_raw_value(snapshot, {}, row_key)
    if component_value is not None:
        return component_value
    if row_key.startswith("abc_total_"):
        return dict(result.get("abc_total") or {}).get(row_key.rsplit("_", 1)[-1])
    if row_key.startswith("abc_net_"):
        return dict(result.get("abc_net") or {}).get(row_key.rsplit("_", 1)[-1])
    if row_key == "vde_total_mj_per_km":
        return dict(dict(result.get("vde_results") or {}).get("total") or {}).get("mj_per_km")
    if row_key == "vde_net_mj_per_km":
        return dict(dict(result.get("vde_results") or {}).get("net") or {}).get("mj_per_km")
    return snapshot.get(row_key)


def _preview_display_value(field_key: str, value, unit_system) -> str:
    if quantity_kind_for_field(field_key) is not None:
        return format_display_value_for_field(field_key, value, unit_system, unavailable=_UNAVAILABLE)
    return _display_text_value(value)


def _component_preview_raw_value(primary: dict, fallback: dict, row_key: str):
    aliases = _COMPONENT_PREVIEW_FIELD_ALIASES.get(row_key)
    if not aliases:
        return None
    for key in aliases:
        value = primary.get(key, fallback.get(key))
        if not is_blank(value):
            return value
    return primary.get(aliases[0], fallback.get(aliases[0]))


def _transmission_preview_raw_value(primary: dict, fallback: dict, row_key: str):
    transmission = dict(primary.get("transmission_losses") or {})
    abc = dict(transmission.get("abc") or {})
    lookup = {
        "trans_A_coef_N": transmission.get("A_TRANS", abc.get("A")),
        "trans_B_coef_Npkph": transmission.get("B_TRANS", abc.get("B")),
        "trans_C_coef_Npkph2": transmission.get("C_TRANS", abc.get("C")),
    }
    if not is_blank(lookup.get(row_key)):
        return lookup.get(row_key)
    return primary.get(row_key, fallback.get(row_key))


def _save_gate_label(preview_status: str, validation: dict | None, save: dict | None = None) -> str:
    save_status = str(dict(save or {}).get("status") or "").lower()
    if save_status == "success":
        return "Saved"
    if save_status == "failed":
        return "Failed"
    if str(preview_status or "") != "fresh":
        return "Pending"
    return "Eligible" if validation_allows_save(validation) else "Blocked"


def _invalid_walk_from_labels(state: dict) -> list[str]:
    normalized = normalize_v22_state(state)
    valid_seen = ["baseline"]
    invalid = []
    for proposal in list(normalized.get("proposals") or []):
        walk_from = str(proposal.get("walk_from") or "baseline")
        if walk_from not in valid_seen:
            invalid.append(proposal_display_label(normalized, proposal))
        valid_seen.append(str(proposal.get("proposal_id") or ""))
    return invalid


def _join_domain_detail(domains: list[str], *, suffix: str) -> str:
    labels = [DOMAIN_LABELS.get(domain, str(domain).replace("_", " ").title()) for domain in domains]
    return "; ".join(f"{label} {suffix}" for label in labels)


def _baseline_loaded_label(effective: dict, printed: dict) -> str:
    make = str(effective.get("make") or printed.get("make") or "").strip()
    model = str(effective.get("model") or printed.get("model") or "").strip()
    year = effective.get("year") or printed.get("year")
    legislation = str(effective.get("legislation") or printed.get("legislation") or "").strip()
    cycle = str(effective.get("cycle_name") or printed.get("cycle_name") or "").strip()
    parts = [part for part in [make, model, f"MY{year}" if not is_blank(year) else "", legislation, cycle] if str(part).strip()]
    return " \u00b7 ".join(parts) if parts else _UNAVAILABLE


def _display_field_value(field_key: str, value, unit_system) -> str:
    if quantity_kind_for_field(field_key):
        return format_display_value_for_field(field_key, value, unit_system, unavailable=_UNAVAILABLE)
    return _display_text_value(value)


def _display_text_value(value, *, unavailable: str = _UNAVAILABLE) -> str:
    return unavailable if is_blank(value) else str(value)


def _issue_value_text(field_key: str, value, unit_system) -> str:
    rendered = format_display_value_for_field(field_key, value, unit_system, unavailable="")
    unit = display_unit_for_field(field_key, unit_system)
    if unit and rendered not in {"", "-"}:
        return f"{rendered} {unit}"
    return rendered


def _issue_interval_text(field_key: str, min_value, max_value, unit_system) -> str:
    lower = format_display_value_for_field(field_key, min_value, unit_system, unavailable="")
    upper = format_display_value_for_field(field_key, max_value, unit_system, unavailable="")
    unit = display_unit_for_field(field_key, unit_system)
    suffix = f" {unit}" if unit else ""
    return f"({lower}, {upper}]{suffix}"
