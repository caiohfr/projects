from __future__ import annotations

from copy import deepcopy
import json

from src.vde_core.vde_not_used_modes import normalize_not_used_proposal_type
from src.vde_core.vde_component_modes import legacy_component_mode_fields
from src.vde_core.vde_request_detail_mapping import detail_key_for_domain_field
from src.vde_core.vde_request_contract import FIELD_KEY_ALIASES, is_blank, normalize_domain
from src.vde_core.vde_tire_modes import canonical_tire_details, canonical_tire_proposal_type, legacy_tire_detail_fields


VDE_REQUEST_IMPORT_VERSION = "0.1"

_PROPOSAL_DOMAINS = ("mass", "aero", "tire", "transmission", "brake", "axle_hubs", "parasitic")
_COMPONENT_GROUP_BY_DOMAIN = {
    "mass": "mass_aero",
    "aero": "mass_aero",
    "tire": "tire",
    "transmission": "transmission",
    "brake": "brake",
    "axle_hubs": "axle_hubs",
    "parasitic": "parasitic",
}
_METADATA_BASELINE_FIELDS = {
    "selected_baseline_vde_id": "selected_baseline_vde_id",
    "legislation": "legislation",
    "make": "make",
    "model": "model",
    "year": "model_year",
    "cycle_name": "cycle",
    "notes": "description",
    "abc_total_source_ui": "roadload_source_type",
}
_BASELINE_DIRECT_FIELDS = {
    "category": "category",
    "electrification": "electrification",
    "transmission_type": "transmission_type",
    "drive_type": "drive_type",
    "fuel_type": "fuel_type",
    "mass_kg": "curb_mass_kg",
    "test_mass_kg": "test_mass_kg",
    "inertia_class": "inertia_class",
    "payload_kg": "payload_kg",
    "weight_dist_fr_pct": "fr_weight_pct",
    "A": "ABC_TOTAL_A",
    "B": "ABC_TOTAL_B",
    "C": "ABC_TOTAL_C",
    "cda_m2": "CdA",
    "frontal_area_m2": "frontal_area_m2",
    "tire_code": "tire_code",
    "tire_db_id": "tire_db_id",
    "tire_size": "tire_size",
    "front_pressure_psi": "psi_front",
    "rear_pressure_psi": "psi_rear",
    "hot_front_pressure_psi": "hot_front_pressure_psi",
    "hot_rear_pressure_psi": "hot_rear_pressure_psi",
    "rrc_N_per_kN": "rrc_N_per_kN",
    "smerf": "smerf",
    "mass_profile_gcwr_kg": "GCWR_kg",
    "mass_profile_trailer_mass_kg": "trailer_weight_kg",
    "trailer_code": "trailer_code",
    "trailer_A": "trailer_A",
    "trailer_B": "trailer_B",
    "trailer_C": "trailer_C",
    "trans_A_coef_N": "trans_A_loss",
    "trans_B_coef_Npkph": "trans_B_loss",
    "trans_C_coef_Npkph2": "trans_C_loss",
    "brake_A_coef_N": "brake_A",
    "brake_B_Npkph": "brake_B",
    "brake_C_coef_Npkph2": "brake_C",
    "axle_hub_A": "axle_hub_A",
    "axle_hub_B": "axle_hub_B",
    "axle_hub_C": "axle_hub_C",
    "parasitic_A_coef_N": "parasitic_A",
    "parasitic_B_Npkph": "parasitic_B",
    "parasitic_C_coef_Npkph2": "parasitic_C",
}
_PROPOSAL_DIRECT_METADATA_FIELDS = {
    "name": "name",
    "description": "description",
    "make": "make",
    "model": "model",
    "model_year": "year",
    "category": "category",
    "electrification": "electrification",
    "transmission_type": "transmission_type",
    "drive_type": "drive_type",
    "fuel_type": "fuel_type",
    "legislation": "legislation",
    "cycle_name": "cycle_name",
}
_INTERNAL_ONLY_PROPOSAL_TYPES = {
    "transmission": {"TRANS_METADATA_ONLY"},
    "brake": {"BRAKE_METADATA_ONLY"},
    "axle_hubs": {"AXLE_HUB_METADATA_ONLY"},
    "parasitic": {"PARASITIC_METADATA_ONLY"},
    "tire": set(),
}
_IMPORT_PLACEHOLDER_TYPES = {
    "aero": "AERO_NOT_USED",
    "mass": "MASS_IMPORTED_REVIEW",
    "tire": "TIRE_IMPORTED_REVIEW",
    "transmission": "TRANS_IMPORTED_REVIEW",
    "brake": "BRAKE_IMPORTED_REVIEW",
    "axle_hubs": "AXLE_IMPORTED_REVIEW",
    "parasitic": "PARASITIC_IMPORTED_REVIEW",
}


def _normalize_issue(issue: dict | None, *, scope: str | None = None, column_id: str | None = None, domain: str | None = None) -> dict:
    payload = deepcopy(dict(issue or {}))
    payload.setdefault("severity", "review")
    if scope:
        payload["scope"] = scope
    if column_id:
        payload["column_id"] = column_id
    if domain:
        payload["domain"] = domain
    return payload


def _group_issues(issues: list[dict]) -> dict:
    blocking = [
        item
        for item in issues
        if str(item.get("severity") or "").strip().lower() == "error"
        and str(item.get("scope") or "").strip().lower() == "workbook"
    ]
    review = [
        item
        for item in issues
        if str(item.get("severity") or "").strip().lower() == "review"
        or (
            str(item.get("severity") or "").strip().lower() == "error"
            and str(item.get("scope") or "").strip().lower() != "workbook"
        )
    ]
    warnings = [item for item in issues if str(item.get("severity") or "").strip().lower() not in {"error", "review"}]
    return {
        "blocking_errors": blocking,
        "review_issues": review,
        "warnings": warnings,
        "blocking_count": len(blocking),
        "review_count": len(review),
        "warning_count": len(warnings),
    }


def _preferred_alias(field_key: str) -> str:
    aliases = FIELD_KEY_ALIASES.get(field_key)
    if aliases:
        return str(aliases[0])
    return str(field_key)


def _proposal_label(proposal: dict, domain_request: dict) -> str:
    for candidate in (
        proposal.get("name"),
        domain_request.get("raw_proposal_type"),
        dict(domain_request.get("normalized_proposal") or {}).get("selection_mode"),
        domain_request.get("proposal_type"),
        proposal.get("source_column"),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return "Imported request"


def _proposal_note_lines(proposal: dict, domain_request: dict) -> list[str]:
    lines: list[str] = []
    raw_type = str(domain_request.get("raw_proposal_type") or "").strip()
    if raw_type:
        lines.append(f"Imported template type: {raw_type}")
    if domain_request.get("has_internal_equivalent") is False:
        lines.append("Imported as compatibility state. Review before using Preview / Save.")
    for issue in list(domain_request.get("issues") or []):
        message = str(dict(issue or {}).get("message") or "").strip()
        if message and message not in lines:
            lines.append(message)
    return lines


def _detail_target_for_field(domain_key: str, proposal_type: str, field_key: str, seed: dict) -> str:
    return detail_key_for_domain_field(domain_key, proposal_type, field_key, seed)


def _build_domain_proposal(proposal: dict, domain_request: dict) -> dict | None:
    raw_type = domain_request.get("raw_proposal_type")
    proposal_type = str(domain_request.get("proposal_type") or "").strip().upper()
    normalized = dict(domain_request.get("normalized_proposal") or {})
    selection_mode = str(domain_request.get("selection_mode") or normalized.get("selection_mode") or "").strip()
    normalized_selection_mode = selection_mode.replace("_", " ").strip().lower()
    if not proposal_type and not is_blank(raw_type):
        proposal_type = _IMPORT_PLACEHOLDER_TYPES.get(normalize_domain(domain_request.get("domain")), "IMPORTED_REVIEW")
    if not proposal_type:
        return None
    if proposal_type == "INHERIT" and is_blank(raw_type):
        return None

    domain_key = normalize_domain(domain_request.get("domain"))
    proposal_type = normalize_not_used_proposal_type(domain_key, proposal_type, selection_mode)
    if domain_key == "tire":
        proposal_type = canonical_tire_proposal_type(proposal_type)
    seed = deepcopy(dict(domain_request.get("proposal_details_seed") or {}))
    details = deepcopy(seed)
    raw_values = dict(domain_request.get("raw_values") or {})
    for field_key, value in raw_values.items():
        if is_blank(value):
            continue
        detail_key = _detail_target_for_field(domain_key, proposal_type, field_key, seed)
        details[detail_key] = value
    if domain_key == "tire":
        details = canonical_tire_details(proposal_type, details)
        details.update(legacy_tire_detail_fields(proposal_type, details))
    details.update(legacy_component_mode_fields(domain_key, proposal_type, selection_mode, details))
    if proposal.get("name") not in (None, ""):
        details.setdefault("notes", str(proposal.get("name")))
    if proposal_type == "AERO_NOT_USED":
        details.setdefault("source", "Aero Not used import")

    notes = _proposal_note_lines(proposal, domain_request)
    if proposal_type == "AERO_NOT_USED":
        notes.append("Aero Not used remains a Review item because the physical exclusion path is not consolidated yet.")

    status = "Draft"
    if domain_request.get("has_internal_equivalent") is False:
        status = "Review"
    if any(str(item.get("severity") or "").strip().lower() in {"error", "review"} for item in list(domain_request.get("issues") or [])):
        status = "Review"

    if proposal_type in _INTERNAL_ONLY_PROPOSAL_TYPES.get(domain_key, set()):
        status = "Review"

    return {
        "id": str(proposal.get("proposal_id") or ""),
        "domain": domain_key,
        "type": proposal_type,
        "proposal_type": proposal_type,
        "label": _proposal_label(proposal, domain_request),
        "selection_mode": selection_mode,
        "details": details,
        "status": status,
        "notes": notes,
    }


def build_v21_request_import_summary(draft: dict, adapted_state: dict | None = None) -> dict:
    draft = deepcopy(dict(draft or {}))
    proposals = list(draft.get("proposals") or [])
    issues: list[dict] = []
    issues.extend(_normalize_issue(item, scope="workbook") for item in list(draft.get("issues") or []))
    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id") or "")
        issues.extend(_normalize_issue(item, scope="proposal", column_id=proposal_id) for item in list(proposal.get("issues") or []))
        for domain_key, domain_request in dict(proposal.get("domain_requests") or {}).items():
            issues.extend(
                _normalize_issue(item, scope="domain", column_id=proposal_id, domain=domain_key)
                for item in list(dict(domain_request or {}).get("issues") or [])
            )
    grouped = _group_issues(issues)
    active_columns = []
    if adapted_state:
        for scenario in list(adapted_state.get("scenarios") or []):
            if str(scenario.get("role") or "") != "walked":
                continue
            active_columns.append(
                {
                    "column_id": str(scenario.get("key") or ""),
                    "label": str(scenario.get("label") or ""),
                }
            )
    baseline_correction_count = sum(1 for value in dict(draft.get("baseline_corrections") or {}).values() if not is_blank(value))
    return {
        "schema_version": draft.get("schema_version"),
        "template_version": draft.get("template_version"),
        "source": deepcopy(dict(draft.get("source") or {})),
        "proposal_count": len(proposals),
        "baseline_correction_count": baseline_correction_count,
        "issues": issues,
        **grouped,
        "active_columns": active_columns,
    }


def build_v21_workbook_state_from_request_draft(draft: dict, current_state: dict | None = None) -> dict:
    draft = deepcopy(dict(draft or {}))
    if not draft.get("proposals") and not draft.get("baseline_printed") and not draft.get("baseline_corrections"):
        raise ValueError("Imported draft is empty or missing required request content.")

    current_state = deepcopy(dict(current_state or {}))
    proposals = list(draft.get("proposals") or [])
    scenarios = [{"key": "baseline", "label": "Baseline", "role": "baseline"}]
    columns: dict[str, dict] = {}
    proposal_columns: dict[str, dict] = {}
    import_columns: dict[str, dict] = {}
    proposal_id_to_column_id: dict[str, str] = {}

    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        if not proposal_id:
            raise ValueError("Imported proposal is missing proposal_id.")
        label = f"Requested #{int(proposal.get('display_index') or len(proposal_id_to_column_id) + 1)}"
        scenarios.append({"key": proposal_id, "label": label, "role": "walked"})
        proposal_id_to_column_id[proposal_id] = proposal_id

    rows = list(current_state.get("rows") or [])
    baseline_printed = deepcopy(dict(draft.get("baseline_printed") or {}))
    baseline_corrections = deepcopy(dict(draft.get("baseline_corrections") or {}))
    effective_baseline = deepcopy(dict(draft.get("effective_baseline") or {}))
    baseline_source_type = str(draft.get("baseline_source_type") or effective_baseline.get("baseline_source_type") or "").strip().upper()
    metadata = deepcopy(dict(current_state.get("metadata") or {}))
    metadata["display_units"] = str(metadata.get("display_units") or "Metric")

    for field_key, target_key in _METADATA_BASELINE_FIELDS.items():
        value = effective_baseline.get(field_key)
        if is_blank(value):
            continue
        metadata[target_key] = value

    if baseline_source_type == "NEW_TEST":
        baseline_line_source = "New test ABC_TOTAL"
    elif baseline_source_type == "EXISTING_VDE":
        baseline_line_source = "Existing VDE DB"
    else:
        baseline_line_source = "Existing VDE DB" if not is_blank(metadata.get("selected_baseline_vde_id")) else "New test ABC_TOTAL"
    baseline_direct: dict[str, object] = {}
    for field_key, target_key in _BASELINE_DIRECT_FIELDS.items():
        value = effective_baseline.get(field_key)
        if is_blank(value):
            continue
        baseline_direct[target_key] = value

    baseline_column = {
        "kind": "baseline",
        "label": "Baseline",
        "walk_from": None,
        "line_source": baseline_line_source,
        "selected_vde_id": metadata.get("selected_baseline_vde_id") if baseline_line_source == "Existing VDE DB" else None,
        "direct": baseline_direct,
        "printed_overrides": {},
        "baseline_overrides": {},
        "domains": {},
    }
    for field_key, value in baseline_printed.items():
        if is_blank(value):
            continue
        target_key = _BASELINE_DIRECT_FIELDS.get(field_key)
        if target_key is None and field_key not in {"selected_baseline_vde_id", "legislation", "make", "model", "year", "cycle_name", "notes", "abc_total_source_ui"}:
            target_key = field_key
        if target_key:
            baseline_column["printed_overrides"].setdefault("__global__", {})[target_key] = value
    columns["baseline"] = baseline_column

    all_issues: list[dict] = [_normalize_issue(item, scope="workbook") for item in list(draft.get("issues") or [])]
    proposal_seq = 0

    for proposal in proposals:
        column_id = proposal_id_to_column_id[str(proposal.get("proposal_id"))]
        display_index = int(proposal.get("display_index") or len(import_columns) + 1)
        walk_from_payload = dict(proposal.get("walk_from") or {})
        walk_from_kind = str(walk_from_payload.get("kind") or "baseline").strip().lower()
        walk_from_raw = str(walk_from_payload.get("source_column") or "Baseline")
        walk_from = "baseline"
        if walk_from_kind == "proposal":
            mapped = proposal_id_to_column_id.get(str(walk_from_payload.get("proposal_id") or ""))
            if mapped:
                walk_from = mapped
        direct = {"line_source": "New / Insert"}
        effective_metadata = deepcopy(dict(proposal.get("effective_metadata") or {}))
        if str(effective_metadata.get("description") or proposal.get("name") or "").strip():
            direct["description"] = str(effective_metadata.get("description") or proposal.get("name") or "").strip()
        for metadata_key, direct_key in _PROPOSAL_DIRECT_METADATA_FIELDS.items():
            value = effective_metadata.get(metadata_key)
            if is_blank(value):
                continue
            direct[direct_key] = value
        columns[column_id] = {
            "kind": "walked",
            "label": next(item["label"] for item in scenarios if str(item["key"]) == column_id),
            "walk_from": walk_from,
            "line_source": "New / Insert",
            "direct": direct,
            "domains": {},
        }
        column_proposals: dict[str, dict] = {}
        column_issues = [_normalize_issue(item, scope="proposal", column_id=column_id) for item in list(proposal.get("issues") or [])]
        all_issues.extend(column_issues)

        for domain_key in _PROPOSAL_DOMAINS:
            domain_request = dict(dict(proposal.get("domain_requests") or {}).get(domain_key) or {})
            if not domain_request:
                continue
            proposal_payload = _build_domain_proposal(proposal, domain_request)
            if proposal_payload:
                column_proposals[domain_key] = proposal_payload
                proposal_seq += 1
            all_issues.extend(
                _normalize_issue(item, scope="domain", column_id=column_id, domain=domain_key)
                for item in list(domain_request.get("issues") or [])
            )

        if column_proposals:
            proposal_columns[column_id] = column_proposals

        import_columns[column_id] = {
            "proposal_id": str(proposal.get("proposal_id") or ""),
            "source_column": str(proposal.get("source_column") or ""),
            "source_index": proposal.get("source_index"),
            "display_index": display_index,
            "requested_label": next(item["label"] for item in scenarios if str(item["key"]) == column_id),
            "walk_from": walk_from,
            "walk_from_requested": walk_from_raw,
            "walk_from_kind": walk_from_kind,
            "raw_values": deepcopy(dict(proposal.get("raw_values") or {})),
            "normalized_values": deepcopy(dict(proposal.get("normalized_values") or {})),
            "issues": column_issues,
            "domains": {
                domain_key: {
                    "raw_proposal_type": dict(domain_request).get("raw_proposal_type"),
                    "normalized_proposal": deepcopy(dict(dict(domain_request).get("normalized_proposal") or {})),
                    "raw_values": deepcopy(dict(dict(domain_request).get("raw_values") or {})),
                    "normalized_values": deepcopy(dict(dict(domain_request).get("normalized_values") or {})),
                    "aliases": deepcopy(dict(dict(domain_request).get("aliases") or {})),
                    "issues": deepcopy(list(dict(domain_request).get("issues") or [])),
                    "proposal_type": dict(domain_request).get("proposal_type"),
                    "selection_mode": dict(domain_request).get("selection_mode"),
                    "has_internal_equivalent": dict(domain_request).get("has_internal_equivalent"),
                }
                for domain_key, domain_request in dict(proposal.get("domain_requests") or {}).items()
            },
        }

    state = {
        "menu": str(current_state.get("menu") or "Scenario Workbook"),
        "rows": rows,
        "scenarios": scenarios,
        "columns": columns,
        "metadata": metadata,
        "proposals": proposal_columns,
        "proposal_seq": proposal_seq,
        "preview_cache": {},
        "save_target": scenarios[-1]["key"] if scenarios else "baseline",
        "audit_target": scenarios[-1]["key"] if scenarios else "baseline",
        "proposal_target": next((scenario["key"] for scenario in scenarios if str(scenario.get("role") or "") == "walked"), "baseline"),
        "baseline_override_enabled": any(
            not is_blank(value)
            for target_bucket in dict(baseline_column.get("baseline_overrides") or {}).values()
            for domain_bucket in dict(target_bucket or {}).values()
            for value in dict(domain_bucket or {}).values()
        ),
        "v21_saved_targets": {},
        "vde_request_draft": draft,
        "vde_request_source": deepcopy(dict(draft.get("source") or {})),
        "vde_request_import": {
            "version": VDE_REQUEST_IMPORT_VERSION,
            "schema_version": draft.get("schema_version"),
            "template_version": draft.get("template_version"),
            "source": deepcopy(dict(draft.get("source") or {})),
            "baseline_printed": baseline_printed,
            "baseline_corrections": baseline_corrections,
            "effective_baseline": effective_baseline,
            "columns": import_columns,
            "issues": all_issues,
        },
    }
    summary = build_v21_request_import_summary(draft, adapted_state=state)
    state["vde_request_import_summary"] = summary
    json.dumps(state, default=str)
    return state


def apply_v21_request_import(current_state: dict, draft: dict) -> dict:
    current_snapshot = deepcopy(dict(current_state or {}))
    next_state = build_v21_workbook_state_from_request_draft(draft, current_snapshot)
    if next_state["vde_request_import_summary"]["blocking_count"] > 0:
        return next_state
    return next_state
