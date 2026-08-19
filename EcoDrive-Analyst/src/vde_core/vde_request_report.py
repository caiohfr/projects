from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

from src.vde_core.vde_request_contract import FIELD_KEY_ALIASES, VDE_REQUEST_SCHEMA_VERSION, is_blank
from src.vde_core.vde_request_preview import build_validation_summary
from src.vde_core.vde_request_save import generate_auto_proposal_name


VDE_REQUEST_REPORT_VERSION = "1.0"

_DASH = "\u2014"
_COMPONENT_DOMAINS = ("tire", "transmission", "brake", "axle_hubs", "parasitic")
_STATUS_STYLE_KEY = {
    "ok": "status_ok",
    "success": "status_ok",
    "saved": "status_ok",
    "updated": "status_ok",
    "reused_existing": "status_ok",
    "created": "status_ok",
    "review": "status_review",
    "pending": "status_review",
    "partial": "status_review",
    "draft": "status_review",
    "missing": "status_missing",
    "invalid": "status_invalid",
    "blocked": "status_invalid",
    "failed": "status_invalid",
    "error": "status_invalid",
    "component_creation_failed": "status_invalid",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _sanitize_filename_token(value, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = Path(text).name
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or fallback


def _fmt_number(value, digits: int = 3):
    if is_blank(value):
        return None
    try:
        number = float(value)
    except Exception:
        return value
    text = f"{number:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _fmt_mass(value):
    return _fmt_number(value, 1)


def _fmt_cda(value):
    return _fmt_number(value, 4)


def _fmt_energy(value):
    if isinstance(value, dict):
        value = value.get("mj_per_km")
    return _fmt_number(value, 3)


def _fmt_abc(value, digits: int):
    return _fmt_number(value, digits)


def _compact(value) -> str:
    if value in (None, "", [], {}):
        return _DASH
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            parts.append(f"{key}={_compact(item)}")
        return " | ".join(parts) if parts else _DASH
    if isinstance(value, list):
        parts = [_compact(item) for item in value if item not in (None, "", [], {})]
        return " | ".join(str(item) for item in parts if item not in ("", _DASH)) or _DASH
    return str(value)


def _proposal_index(request_draft: dict | None) -> tuple[dict[str, dict], dict[str, dict]]:
    by_id: dict[str, dict] = {}
    by_source: dict[str, dict] = {}
    for item in list(dict(request_draft or {}).get("proposals") or []):
        payload = deepcopy(dict(item or {}))
        proposal_id = str(payload.get("proposal_id") or "").strip()
        source_column = str(payload.get("source_column") or "").strip()
        if proposal_id:
            by_id[proposal_id] = payload
        if source_column:
            by_source[source_column] = payload
    return by_id, by_source


def _saved_lookup(save_result: dict | None) -> tuple[dict[str, dict], dict[tuple[str, str], dict], dict[int, dict]]:
    proposal_map: dict[str, dict] = {}
    component_map: dict[tuple[str, str], dict] = {}
    baseline_map: dict[int, dict] = {}
    result = dict(save_result or {})
    for item in list(result.get("saved_proposals") or []):
        proposal_id = str(dict(item or {}).get("proposal_id") or "").strip()
        if proposal_id:
            proposal_map[proposal_id] = deepcopy(dict(item or {}))
    for item in list(result.get("component_results") or []):
        payload = deepcopy(dict(item or {}))
        key = (str(payload.get("proposal_id") or "").strip(), str(payload.get("domain") or "").strip())
        if key[0] and key[1]:
            component_map[key] = payload
    for item in list(result.get("baseline_updates") or []):
        payload = deepcopy(dict(item or {}))
        try:
            baseline_id = int(payload.get("baseline_id"))
        except Exception:
            continue
        baseline_map[baseline_id] = payload
    return proposal_map, component_map, baseline_map


def _baseline_metadata(resolution_result: dict | None) -> dict:
    resolution = dict(resolution_result or {})
    baseline = dict(resolution.get("baseline") or {})
    effective = dict(baseline.get("effective") or {})
    printed = dict(baseline.get("printed") or {})
    baseline_id = effective.get("selected_baseline_vde_id") or printed.get("selected_baseline_vde_id")
    baseline_name = " ".join(
        str(item).strip()
        for item in (effective.get("make"), effective.get("model"), effective.get("year"))
        if str(item or "").strip()
    ).strip()
    return {
        "baseline_id": baseline_id,
        "baseline_name": baseline_name or None,
        "printed": printed,
        "correction": dict(baseline.get("correction") or {}),
        "effective": effective,
        "corrected_fields": list(baseline.get("corrected_fields") or []),
    }


def _value_from_aliases(payload: dict | None, field_key: str):
    data = dict(payload or {})
    aliases = FIELD_KEY_ALIASES.get(field_key, (field_key,))
    for alias in aliases:
        if alias in data and not is_blank(data.get(alias)):
            return data.get(alias)
    return None


def _resolved_field_value(proposal_result: dict, field_key: str):
    resolved = dict(proposal_result.get("resolved_snapshot") or {})
    source = dict(proposal_result.get("source_snapshot") or {})
    value = _value_from_aliases(resolved, field_key)
    if not is_blank(value):
        return value
    mass_setup = dict(resolved.get("resolved_mass_setup") or {})
    transmission = dict(resolved.get("transmission_losses") or {})
    lookup = {
        "test_mass_kg": mass_setup.get("test_mass_kg") or mass_setup.get("resolved_mass_used_kg"),
        "test_mass_basis": mass_setup.get("test_mass_basis"),
        "inertia_class": mass_setup.get("inertia_class"),
        "payload_kg": mass_setup.get("payload_kg"),
        "weight_dist_fr_pct": mass_setup.get("weight_dist_fr_pct"),
        "mass_rule_status": mass_setup.get("mass_rule_status"),
        "mass_rule_notes": mass_setup.get("mass_rule_notes"),
        "cda_m2": resolved.get("CdA"),
        "trans_A_coef_N": transmission.get("A_TRANS"),
        "trans_B_coef_Npkph": transmission.get("B_TRANS"),
        "trans_C_coef_Npkph2": transmission.get("C_TRANS"),
        "A": dict(proposal_result.get("abc_total") or {}).get("A"),
        "B": dict(proposal_result.get("abc_total") or {}).get("B"),
        "C": dict(proposal_result.get("abc_total") or {}).get("C"),
    }
    value = lookup.get(field_key)
    if not is_blank(value):
        return value
    return _value_from_aliases(source, field_key)


def _request_rows_from_original(request_draft: dict, resolution_result: dict | None) -> list[dict]:
    proposal_by_id, proposal_by_source = _proposal_index(request_draft)
    proposal_results = {
        str(item.get("source_column") or ""): dict(item or {})
        for item in list(dict(resolution_result or {}).get("proposal_results") or [])
    }
    baseline_meta = _baseline_metadata(resolution_result)
    rows: list[dict] = []
    for record in list(dict(dict(request_draft or {}).get("original_request") or {}).get("request_rows") or []):
        payload = dict(record or {})
        field_key = str(payload.get("field_key") or "").strip()
        for requested_meta in list(dict(dict(request_draft or {}).get("original_request") or {}).get("requested_columns") or []):
            source_column = str(dict(requested_meta).get("source_column") or "").strip()
            proposal = dict(proposal_by_source.get(source_column) or {})
            proposal_result = dict(proposal_results.get(source_column) or {})
            rows.append(
                {
                    "Scenario": source_column or "Baseline",
                    "Proposal ID": str(proposal.get("proposal_id") or ""),
                    "Section": payload.get("section"),
                    "Field / Parameter": payload.get("field_label"),
                    "Field Key": field_key,
                    "Unit": payload.get("unit"),
                    "Baseline Printed": payload.get("baseline_printed"),
                    "Baseline Correction": payload.get("baseline_correction"),
                    "Baseline Effective": baseline_meta["effective"].get(field_key),
                    "Requested Original": dict(payload.get("requested_values") or {}).get(source_column),
                    "Requested Normalized": dict(proposal.get("normalized_values") or {}).get(field_key),
                    "Source Value": _value_from_aliases(dict(proposal_result.get("source_snapshot") or {}), field_key),
                    "Resolved Value": _resolved_field_value(proposal_result, field_key) if proposal_result else None,
                    "Source Column": source_column,
                }
            )
    return rows


def _manual_request_rows(request_draft: dict, resolution_result: dict | None) -> list[dict]:
    proposal_by_id, _ = _proposal_index(request_draft)
    baseline_meta = _baseline_metadata(resolution_result)
    proposal_results = list(dict(resolution_result or {}).get("proposal_results") or [])
    field_keys = set(baseline_meta["printed"]) | set(baseline_meta["correction"]) | set(baseline_meta["effective"])
    for proposal in proposal_by_id.values():
        field_keys.update(dict(proposal.get("raw_values") or {}).keys())
        field_keys.update(dict(proposal.get("normalized_values") or {}).keys())
        for domain_request in dict(proposal.get("domain_requests") or {}).values():
            field_keys.update(dict(domain_request).get("raw_values", {}).keys())
    ordered_field_keys = ["notes", "walk_from", *sorted(key for key in field_keys if key not in {"notes", "walk_from"})]
    rows: list[dict] = []
    for proposal_result in proposal_results:
        proposal_id = str(dict(proposal_result).get("proposal_id") or "")
        proposal = dict(proposal_by_id.get(proposal_id) or {})
        source_column = str(proposal.get("source_column") or proposal_result.get("source_column") or proposal_id)
        for field_key in ordered_field_keys:
            raw_values = dict(proposal.get("raw_values") or {})
            normalized_values = dict(proposal.get("normalized_values") or {})
            if field_key == "walk_from":
                requested_original = dict(proposal.get("walk_from") or {}).get("source_column")
                requested_normalized = requested_original
            elif field_key == "notes":
                requested_original = proposal.get("name")
                requested_normalized = proposal.get("name")
            else:
                requested_original = raw_values.get(field_key)
                requested_normalized = normalized_values.get(field_key) or requested_original
            if is_blank(requested_original) and is_blank(requested_normalized) and is_blank(baseline_meta["printed"].get(field_key)) and is_blank(baseline_meta["correction"].get(field_key)) and is_blank(baseline_meta["effective"].get(field_key)):
                continue
            rows.append(
                {
                    "Scenario": source_column,
                    "Proposal ID": proposal_id,
                    "Section": "UI",
                    "Field / Parameter": field_key,
                    "Field Key": field_key,
                    "Unit": "",
                    "Baseline Printed": baseline_meta["printed"].get(field_key),
                    "Baseline Correction": baseline_meta["correction"].get(field_key),
                    "Baseline Effective": baseline_meta["effective"].get(field_key),
                    "Requested Original": requested_original,
                    "Requested Normalized": requested_normalized,
                    "Source Value": _value_from_aliases(dict(proposal_result.get("source_snapshot") or {}), field_key),
                    "Resolved Value": _resolved_field_value(dict(proposal_result or {}), field_key),
                    "Source Column": source_column,
                }
            )
    return rows


def build_request_equivalent_draft_from_state(workbook_state: dict) -> dict:
    state = deepcopy(dict(workbook_state or {}))
    import_meta = dict(state.get("vde_request_import") or {})
    source = dict(state.get("vde_request_source") or import_meta.get("source") or {})
    scenarios = list(state.get("scenarios") or [])
    columns = dict(state.get("columns") or {})
    proposals_state = dict(state.get("proposals") or {})
    proposals: list[dict] = []
    baseline_printed = deepcopy(dict(import_meta.get("baseline_printed") or {}))
    baseline_corrections = deepcopy(dict(import_meta.get("baseline_corrections") or {}))
    effective_baseline = deepcopy(dict(import_meta.get("effective_baseline") or {}))
    for index, scenario in enumerate([item for item in scenarios if str(item.get("role") or "") == "walked"], start=1):
        column_id = str(scenario.get("key") or "")
        column = dict(columns.get(column_id) or {})
        domain_requests = {}
        raw_values = {}
        description = str(dict(column.get("direct") or {}).get("description") or "").strip()
        if description:
            raw_values["notes"] = description
        walk_from = str(column.get("walk_from") or "baseline")
        walk_source = "Baseline" if walk_from == "baseline" else walk_from
        raw_values["walk_from"] = walk_source
        for domain_key, proposal in dict(proposals_state.get(column_id) or {}).items():
            payload = dict(proposal or {})
            details = deepcopy(dict(payload.get("details") or {}))
            proposal_type = str(payload.get("proposal_type") or payload.get("type") or "").strip() or None
            domain_raw = {key: value for key, value in details.items() if not is_blank(value)}
            raw_values.update(domain_raw)
            domain_requests[domain_key] = {
                "domain": domain_key,
                "raw_proposal_type": proposal_type,
                "proposal_type": proposal_type,
                "selection_mode": proposal_type,
                "raw_values": deepcopy(domain_raw),
                "normalized_values": deepcopy(domain_raw),
                "proposal_details_seed": deepcopy(details),
                "issues": [],
                "has_internal_equivalent": True,
            }
        proposals.append(
            {
                "proposal_id": column_id,
                "display_index": index,
                "source_column": str(scenario.get("label") or f"Requested #{index}"),
                "source_index": index,
                "name": description or None,
                "walk_from": {
                    "kind": "baseline" if walk_from == "baseline" else "proposal",
                    "proposal_id": None if walk_from == "baseline" else walk_from,
                    "source_column": walk_source,
                },
                "raw_values": raw_values,
                "normalized_values": deepcopy(raw_values),
                "domain_requests": domain_requests,
                "issues": [],
            }
        )
    return {
        "schema_version": str(import_meta.get("schema_version") or VDE_REQUEST_SCHEMA_VERSION),
        "template_version": str(import_meta.get("template_version") or ""),
        "source": {
            "filename": _sanitize_filename_token(source.get("filename") or "UI_Request", "UI_Request"),
            "source_type": "UI",
            "imported_at": str(source.get("imported_at") or ""),
        },
        "baseline_printed": baseline_printed,
        "baseline_corrections": baseline_corrections,
        "effective_baseline": effective_baseline,
        "proposals": proposals,
        "issues": deepcopy(list(import_meta.get("issues") or [])),
        "original_request": {},
    }


def _request_rows(request_draft: dict, resolution_result: dict | None) -> list[dict]:
    original_rows = list(dict(dict(request_draft or {}).get("original_request") or {}).get("request_rows") or [])
    if original_rows:
        rows = _request_rows_from_original(request_draft, resolution_result)
    else:
        rows = _manual_request_rows(request_draft, resolution_result)
    return rows


def _proposal_label(request_proposal: dict, proposal_result: dict) -> str:
    return str(
        request_proposal.get("name")
        or request_proposal.get("source_column")
        or proposal_result.get("source_column")
        or generate_auto_proposal_name(proposal_result)
        or proposal_result.get("proposal_id")
        or "Requested proposal"
    )


def _summary_counts(resolution_result: dict | None, save_result: dict | None) -> dict:
    summary = build_validation_summary(resolution_result)
    result = dict(save_result or {})
    return {
        "Overall status": summary.get("overall_status"),
        "Total proposals": summary.get("proposal_count"),
        "OK": summary.get("ok_count"),
        "Review": summary.get("review_count"),
        "Missing": summary.get("missing_count"),
        "Invalid": summary.get("invalid_count"),
        "Blocked": summary.get("blocked_count"),
        "Saved": len(list(result.get("saved_proposals") or [])),
        "Skipped": len(list(result.get("skipped_proposals") or [])),
    }


def _highlight_rows(request_draft: dict, resolution_result: dict | None) -> list[dict]:
    draft_by_id, _ = _proposal_index(request_draft)
    rows: list[dict] = []
    for proposal_result in list(dict(resolution_result or {}).get("proposal_results") or []):
        proposal = dict(proposal_result or {})
        request_proposal = dict(draft_by_id.get(str(proposal.get("proposal_id") or "")) or {})
        source_snapshot = dict(proposal.get("source_snapshot") or {})
        resolved_snapshot = dict(proposal.get("resolved_snapshot") or {})
        changes: list[str] = []
        source_mass = source_snapshot.get("test_mass_kg") or source_snapshot.get("mass_kg")
        target_mass = dict(resolved_snapshot.get("resolved_mass_setup") or {}).get("resolved_mass_used_kg") or resolved_snapshot.get("test_mass_kg") or resolved_snapshot.get("mass_kg")
        if not is_blank(source_mass) and not is_blank(target_mass):
            delta = float(target_mass) - float(source_mass)
            if abs(delta) >= 0.05:
                changes.append(f"Mass {delta:+.0f} kg")
        source_cda = source_snapshot.get("CdA")
        target_cda = resolved_snapshot.get("CdA")
        if not is_blank(source_cda) and not is_blank(target_cda):
            delta = float(target_cda) - float(source_cda)
            if abs(delta) >= 0.00005:
                changes.append(f"CdA {delta:+.4f} m^2")
        for domain_key, payload in dict(proposal.get("domain_results") or {}).items():
            domain_payload = dict(payload or {})
            proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
            if proposal_type == "INHERIT":
                continue
            if domain_key == "mass" and any(item.startswith("Mass ") for item in changes):
                continue
            if domain_key == "aero" and any(item.startswith("CdA ") for item in changes):
                continue
            changes.append(f"{domain_key.replace('_', ' ').title()} {proposal_type}")
        rows.append(
            {
                "Scenario": str(proposal.get("source_column") or proposal.get("proposal_id") or "Requested"),
                "Proposal Name": _proposal_label(request_proposal, proposal),
                "Highlights": " | ".join(changes) if changes else "Inherited / no direct engineering delta",
            }
        )
    return rows


def _summary_comparison_rows(request_draft: dict, resolution_result: dict | None, save_result: dict | None) -> list[dict]:
    draft_by_id, _ = _proposal_index(request_draft)
    saved_map, _, _ = _saved_lookup(save_result)
    resolution = dict(resolution_result or {})
    rows: list[dict] = []
    baseline = _baseline_metadata(resolution_result)
    baseline_resolved = dict(dict(resolution.get("resolved_columns") or {}).get("baseline") or {})
    rows.append(
        {
            "Scenario": "Baseline",
            "Proposal Name": baseline.get("baseline_name") or "Baseline reference",
            "Status": "Review" if baseline.get("corrected_fields") else "OK",
            "Saved?": "Reference",
            "VDE DB Row ID": baseline.get("baseline_id"),
            "Walk From": _DASH,
            "Source Column": "Baseline",
            "Mass": baseline_resolved.get("test_mass_kg") or baseline_resolved.get("mass_kg"),
            "CdA": baseline_resolved.get("CdA") or baseline["effective"].get("cda_m2"),
            "ABC_TOTAL A": dict(baseline_resolved.get("initial_abc_total") or {}).get("A"),
            "ABC_TOTAL B": dict(baseline_resolved.get("initial_abc_total") or {}).get("B"),
            "ABC_TOTAL C": dict(baseline_resolved.get("initial_abc_total") or {}).get("C"),
            "VDE_TOTAL": None,
            "ABC_NET A": None,
            "ABC_NET B": None,
            "ABC_NET C": None,
            "VDE_NET": None,
            "Issues": " | ".join(baseline.get("corrected_fields") or []) or None,
        }
    )
    for proposal_result in list(resolution.get("proposal_results") or []):
        proposal = dict(proposal_result or {})
        request_proposal = dict(draft_by_id.get(str(proposal.get("proposal_id") or "")) or {})
        saved_payload = dict(saved_map.get(str(proposal.get("proposal_id") or "")) or {})
        resolved_snapshot = dict(proposal.get("resolved_snapshot") or {})
        mass_setup = dict(resolved_snapshot.get("resolved_mass_setup") or {})
        walk_from = dict(proposal.get("walk_from") or {})
        rows.append(
            {
                "Scenario": str(proposal.get("source_column") or proposal.get("proposal_id") or "Requested"),
                "Proposal Name": _proposal_label(request_proposal, proposal),
                "Status": str(proposal.get("status") or "OK"),
                "Saved?": "Yes" if saved_payload else "No",
                "VDE DB Row ID": saved_payload.get("vde_row_id"),
                "Walk From": str(walk_from.get("label") or walk_from.get("column_id") or _DASH),
                "Source Column": str(proposal.get("source_column") or proposal.get("proposal_id") or ""),
                "Mass": mass_setup.get("resolved_mass_used_kg") or mass_setup.get("test_mass_kg") or resolved_snapshot.get("test_mass_kg") or resolved_snapshot.get("mass_kg"),
                "CdA": resolved_snapshot.get("CdA"),
                "ABC_TOTAL A": dict(proposal.get("abc_total") or {}).get("A"),
                "ABC_TOTAL B": dict(proposal.get("abc_total") or {}).get("B"),
                "ABC_TOTAL C": dict(proposal.get("abc_total") or {}).get("C"),
                "VDE_TOTAL": dict(dict(proposal.get("vde_results") or {}).get("total") or {}).get("mj_per_km"),
                "ABC_NET A": dict(proposal.get("abc_net") or {}).get("A"),
                "ABC_NET B": dict(proposal.get("abc_net") or {}).get("B"),
                "ABC_NET C": dict(proposal.get("abc_net") or {}).get("C"),
                "VDE_NET": dict(dict(proposal.get("vde_results") or {}).get("net") or {}).get("mj_per_km"),
                "Issues": " | ".join(
                    str(dict(issue or {}).get("message") or dict(issue or {}).get("code") or "").strip()
                    for issue in list(proposal.get("issues") or [])
                    if str(dict(issue or {}).get("message") or dict(issue or {}).get("code") or "").strip()
                ) or None,
            }
        )
    return rows


def _result_rows(request_draft: dict, resolution_result: dict | None, save_result: dict | None) -> list[dict]:
    draft_by_id, _ = _proposal_index(request_draft)
    saved_map, _, _ = _saved_lookup(save_result)
    rows: list[dict] = []
    for proposal_result in list(dict(resolution_result or {}).get("proposal_results") or []):
        proposal = dict(proposal_result or {})
        request_proposal = dict(draft_by_id.get(str(proposal.get("proposal_id") or "")) or {})
        resolved_snapshot = dict(proposal.get("resolved_snapshot") or {})
        mass_setup = dict(resolved_snapshot.get("resolved_mass_setup") or {})
        transmission = dict(resolved_snapshot.get("transmission_losses") or {})
        saved_payload = dict(saved_map.get(str(proposal.get("proposal_id") or "")) or {})
        rows.append(
            {
                "Proposal ID": str(proposal.get("proposal_id") or ""),
                "Display Label": str(proposal.get("source_column") or proposal.get("proposal_id") or ""),
                "Proposal Name": _proposal_label(request_proposal, proposal),
                "Source Column": str(proposal.get("source_column") or ""),
                "Status": str(proposal.get("status") or ""),
                "Walk From": str(dict(proposal.get("walk_from") or {}).get("label") or dict(proposal.get("walk_from") or {}).get("column_id") or _DASH),
                "Source Proposal": str(dict(proposal.get("walk_from") or {}).get("column_id") or "baseline"),
                "Mass / Test Mass": mass_setup.get("resolved_mass_used_kg") or mass_setup.get("test_mass_kg") or resolved_snapshot.get("test_mass_kg") or resolved_snapshot.get("mass_kg"),
                "CdA": resolved_snapshot.get("CdA"),
                "ABC_TOTAL A": dict(proposal.get("abc_total") or {}).get("A"),
                "ABC_TOTAL B": dict(proposal.get("abc_total") or {}).get("B"),
                "ABC_TOTAL C": dict(proposal.get("abc_total") or {}).get("C"),
                "ABC_NET A": dict(proposal.get("abc_net") or {}).get("A"),
                "ABC_NET B": dict(proposal.get("abc_net") or {}).get("B"),
                "ABC_NET C": dict(proposal.get("abc_net") or {}).get("C"),
                "VDE_TOTAL": dict(dict(proposal.get("vde_results") or {}).get("total") or {}).get("mj_per_km"),
                "VDE_NET": dict(dict(proposal.get("vde_results") or {}).get("net") or {}).get("mj_per_km"),
                "Cycle information": str(resolved_snapshot.get("cycle_name") or resolved_snapshot.get("legislation") or ""),
                "Transmission losses": _compact(transmission),
                "Preview summary": " | ".join(str(item) for item in list(dict(proposal.get("preview_summary") or {}).get("warnings") or [])) or "OK",
                "Saved?": "Yes" if saved_payload else "No",
                "VDE DB Row ID": saved_payload.get("vde_row_id"),
            }
        )
    return rows


def _component_rows(request_draft: dict, resolution_result: dict | None, save_result: dict | None) -> list[dict]:
    draft_by_id, _ = _proposal_index(request_draft)
    _, component_results_map, _ = _saved_lookup(save_result)
    rows: list[dict] = []
    for proposal_result in list(dict(resolution_result or {}).get("proposal_results") or []):
        proposal = dict(proposal_result or {})
        request_proposal = dict(draft_by_id.get(str(proposal.get("proposal_id") or "")) or {})
        direct_types = {
            domain_key: str(dict(payload or {}).get("proposal_type") or "INHERIT")
            for domain_key, payload in dict(proposal.get("domain_results") or {}).items()
        }
        actions = {
            str(dict(item or {}).get("domain") or ""): dict(item or {})
            for item in list(proposal.get("component_actions") or [])
        }
        domain_keys = sorted(set(actions) | {key for key, value in direct_types.items() if value != "INHERIT"})
        for domain_key in domain_keys:
            action = dict(actions.get(domain_key) or {})
            executed = dict(component_results_map.get((str(proposal.get("proposal_id") or ""), domain_key)) or {})
            snapshot = dict(action.get("component_snapshot") or {})
            row = {
                "Proposal": _proposal_label(request_proposal, proposal),
                "Domain": domain_key,
                "Proposal Type": direct_types.get(domain_key),
                "Action Planned": action.get("action") or ("snapshot_only" if domain_key in {"mass", "aero"} else None),
                "Action Executed": executed.get("status"),
                "Component ID Requested": action.get("component_id"),
                "Component ID Resolved": executed.get("component_id") or action.get("component_id"),
                "Component Name": snapshot.get("component_name") or snapshot.get("name"),
                "Repository Source": snapshot.get("source") or action.get("source"),
                "Lookup Found?": "Yes" if action.get("action") == "reuse_existing" else ("No" if action.get("action") == "unavailable" else None),
                "Used?": "No" if snapshot.get("used") is False else ("Yes" if domain_key in direct_types and direct_types.get(domain_key) != "INHERIT" else None),
                "New Component Confirmed?": "Yes" if str(executed.get("status") or "") == "created" else ("No" if action.get("action") == "eligible_for_new_component" else None),
                "Creation Status": executed.get("status"),
                "Snapshot Available?": "Yes" if snapshot else "No",
                "Technical Summary": _compact(snapshot),
                "Issues": " | ".join(
                    str(dict(issue or {}).get("message") or dict(issue or {}).get("code") or "").strip()
                    for issue in list(action.get("issues") or [])
                    if str(dict(issue or {}).get("message") or dict(issue or {}).get("code") or "").strip()
                ) or executed.get("reason"),
            }
            extra_fields = {
                "A": snapshot.get("A") or snapshot.get("new_trans_A") or snapshot.get("brake_A") or snapshot.get("axle_hubs_A") or snapshot.get("parasitic_A"),
                "B": snapshot.get("B") or snapshot.get("new_trans_B") or snapshot.get("brake_B") or snapshot.get("axle_hubs_B") or snapshot.get("parasitic_B"),
                "C": snapshot.get("C") or snapshot.get("new_trans_C") or snapshot.get("brake_C") or snapshot.get("axle_hubs_C") or snapshot.get("parasitic_C"),
                "loss_pct": snapshot.get("loss_pct"),
                "residual torque": snapshot.get("residual_torque_front_nm") or snapshot.get("residual_torque_rear_nm"),
                "wheel radius": snapshot.get("wheel_radius_m"),
                "RRC": snapshot.get("rrc_N_per_kN"),
                "Component Type": snapshot.get("component_type"),
                "Position": snapshot.get("component_position"),
                "Test Condition": snapshot.get("test_condition_type"),
                "Driveline Architecture": snapshot.get("driveline_architecture"),
                "Physical Boundary": snapshot.get("physical_boundary"),
                "Configuration From": snapshot.get("configuration_from"),
                "Configuration To": snapshot.get("configuration_to"),
                "Test Method": snapshot.get("test_method"),
                "Hardware Reference": snapshot.get("hardware_reference"),
                "Source Reference": snapshot.get("source_reference"),
                "NET Bridge Eligible": snapshot.get("net_bridge_eligible"),
            }
            row.update(extra_fields)
            rows.append(row)
    return rows


def _validation_rows(request_draft: dict, resolution_result: dict | None, save_result: dict | None) -> list[dict]:
    rows: list[dict] = []
    result = dict(resolution_result or {})
    saved_map, component_results_map, _ = _saved_lookup(save_result)
    for issue in list(dict(request_draft or {}).get("issues") or []):
        payload = dict(issue or {})
        rows.append(
            {
                "Scope": "request",
                "Proposal": None,
                "Domain": payload.get("domain"),
                "Field Key": payload.get("field_key"),
                "Code": payload.get("code"),
                "Severity": payload.get("severity"),
                "Status": result.get("status"),
                "Message": payload.get("message"),
                "Blocking?": "Yes" if str(payload.get("severity") or "").lower() == "error" else "No",
                "Confirmed for Save?": None,
                "Resolution / Save Outcome": None,
            }
        )
    for proposal in list(dict(request_draft or {}).get("proposals") or []):
        proposal_id = str(dict(proposal or {}).get("proposal_id") or "")
        for issue in list(dict(proposal or {}).get("issues") or []):
            payload = dict(issue or {})
            rows.append(
                {
                    "Scope": "proposal",
                    "Proposal": proposal_id,
                    "Domain": payload.get("domain"),
                    "Field Key": payload.get("field_key"),
                    "Code": payload.get("code"),
                    "Severity": payload.get("severity"),
                    "Status": None,
                    "Message": payload.get("message"),
                    "Blocking?": "Yes" if str(payload.get("severity") or "").lower() == "error" else "No",
                    "Confirmed for Save?": None,
                    "Resolution / Save Outcome": None,
                }
            )
        for domain_key, domain_request in dict(proposal.get("domain_requests") or {}).items():
            for issue in list(dict(domain_request or {}).get("issues") or []):
                payload = dict(issue or {})
                rows.append(
                    {
                        "Scope": "domain_request",
                        "Proposal": proposal_id,
                        "Domain": domain_key,
                        "Field Key": payload.get("field_key"),
                        "Code": payload.get("code"),
                        "Severity": payload.get("severity"),
                        "Status": None,
                        "Message": payload.get("message"),
                        "Blocking?": "Yes" if str(payload.get("severity") or "").lower() == "error" else "No",
                        "Confirmed for Save?": None,
                        "Resolution / Save Outcome": None,
                    }
                )
    for issue in list(result.get("issues") or []):
        payload = dict(issue or {})
        rows.append(
            {
                "Scope": "resolver",
                "Proposal": payload.get("proposal_id"),
                "Domain": payload.get("domain"),
                "Field Key": payload.get("field_key"),
                "Code": payload.get("code"),
                "Severity": payload.get("severity"),
                "Status": result.get("status"),
                "Message": payload.get("message"),
                "Blocking?": "Yes" if str(payload.get("severity") or "").lower() in {"error", "blocked"} else "No",
                "Confirmed for Save?": None,
                "Resolution / Save Outcome": None,
            }
        )
    for proposal_result in list(result.get("proposal_results") or []):
        proposal = dict(proposal_result or {})
        saved_payload = dict(saved_map.get(str(proposal.get("proposal_id") or "")) or {})
        for issue in list(proposal.get("issues") or []):
            payload = dict(issue or {})
            rows.append(
                {
                    "Scope": "proposal_result",
                    "Proposal": proposal.get("proposal_id"),
                    "Domain": payload.get("domain"),
                    "Field Key": payload.get("field_key"),
                    "Code": payload.get("code"),
                    "Severity": payload.get("severity"),
                    "Status": proposal.get("status"),
                    "Message": payload.get("message"),
                    "Blocking?": "Yes" if str(payload.get("severity") or "").lower() in {"error", "blocked"} else "No",
                    "Confirmed for Save?": "Yes" if saved_payload and str(proposal.get("status") or "") == "Review" else None,
                    "Resolution / Save Outcome": saved_payload.get("status"),
                }
            )
        for action in list(proposal.get("component_actions") or []):
            action_payload = dict(action or {})
            for issue in list(action_payload.get("issues") or []):
                payload = dict(issue or {})
                executed = dict(component_results_map.get((str(proposal.get("proposal_id") or ""), str(action_payload.get("domain") or ""))) or {})
                rows.append(
                    {
                        "Scope": "component_action",
                        "Proposal": proposal.get("proposal_id"),
                        "Domain": action_payload.get("domain"),
                        "Field Key": payload.get("field_key"),
                        "Code": payload.get("code"),
                        "Severity": payload.get("severity"),
                        "Status": proposal.get("status"),
                        "Message": payload.get("message"),
                        "Blocking?": "Yes" if str(payload.get("severity") or "").lower() in {"error", "blocked"} else "No",
                        "Confirmed for Save?": None,
                        "Resolution / Save Outcome": executed.get("status"),
                    }
                )
    for skipped in list(dict(save_result or {}).get("skipped_proposals") or []):
        payload = dict(skipped or {})
        rows.append(
            {
                "Scope": "save",
                "Proposal": payload.get("proposal_id"),
                "Domain": None,
                "Field Key": None,
                "Code": "skipped_proposal",
                "Severity": "review",
                "Status": payload.get("status"),
                "Message": payload.get("reason"),
                "Blocking?": "No",
                "Confirmed for Save?": "No",
                "Resolution / Save Outcome": "skipped",
            }
        )
    for issue in list(dict(save_result or {}).get("issues") or []):
        payload = dict(issue or {})
        rows.append(
            {
                "Scope": "save",
                "Proposal": payload.get("proposal_id"),
                "Domain": payload.get("domain"),
                "Field Key": payload.get("field_key"),
                "Code": payload.get("code"),
                "Severity": payload.get("severity"),
                "Status": dict(save_result or {}).get("status"),
                "Message": payload.get("message"),
                "Blocking?": "Yes" if str(payload.get("severity") or "").lower() in {"error", "blocked"} else "No",
                "Confirmed for Save?": None,
                "Resolution / Save Outcome": dict(save_result or {}).get("status"),
            }
        )
    return rows


def build_vde_request_report_model(
    request_draft,
    resolution_result,
    save_result=None,
) -> dict:
    draft = deepcopy(dict(request_draft or {}))
    resolution = deepcopy(dict(resolution_result or {}))
    save = deepcopy(dict(save_result or {})) if save_result is not None else None
    baseline = _baseline_metadata(resolution)
    request_source = dict(draft.get("source") or {})
    report_state = "Draft"
    if save is not None:
        status = str(save.get("status") or "").lower()
        report_state = {
            "success": "Saved",
            "partial": "Partial",
            "failed": "Failed",
        }.get(status, "Saved")
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary_counts = _summary_counts(resolution, save)
    model = {
        "report_version": VDE_REQUEST_REPORT_VERSION,
        "request_schema_version": str(draft.get("schema_version") or VDE_REQUEST_SCHEMA_VERSION),
        "report_state": report_state,
        "generated_at": generated_at,
        "metadata": {
            "template_version": str(draft.get("template_version") or ""),
            "source_filename": _sanitize_filename_token(request_source.get("filename") or "manual_request", "manual_request"),
            "source_type": str(request_source.get("source_type") or ("Excel" if draft.get("original_request") else "UI")),
            "baseline_id": baseline.get("baseline_id"),
            "baseline_name": baseline.get("baseline_name"),
            "save_operation_id": dict(save or {}).get("operation_id"),
            "save_status": dict(save or {}).get("status"),
            "save_executed_at": dict(save or {}).get("executed_at"),
            "generated_at": generated_at,
            "resolution_overall_status": resolution.get("status"),
        },
        "summary_counts": summary_counts,
        "summary_rows": _summary_comparison_rows(draft, resolution, save),
        "highlight_rows": _highlight_rows(draft, resolution),
        "request_rows": _request_rows(draft, resolution),
        "result_rows": _result_rows(draft, resolution, save),
        "component_rows": _component_rows(draft, resolution, save),
        "validation_rows": _validation_rows(draft, resolution, save),
        "draft": _json_safe(draft),
        "resolution_result": _json_safe(resolution),
        "save_result": _json_safe(save),
    }
    json.dumps(model, default=str)
    return model


def build_vde_request_report_filename(report_model: dict | None) -> str:
    model = dict(report_model or {})
    state = str(model.get("report_state") or "Draft").upper()
    metadata = dict(model.get("metadata") or {})
    baseline_token = _sanitize_filename_token(metadata.get("baseline_id") or metadata.get("baseline_name") or "baseline", "baseline")
    if state == "DRAFT":
        date_token = str(model.get("generated_at") or "")[:10].replace("-", "")
        return f"EcoDrive_VDE_Request_DRAFT_{baseline_token}_{date_token or 'report'}.xlsx"
    op_token = _sanitize_filename_token(metadata.get("save_operation_id") or "operation", "operation")
    return f"EcoDrive_VDE_Request_{state}_{op_token}.xlsx"


def _col_letter(index: int) -> str:
    value = index + 1
    letters = []
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _escape_xml(value: str) -> str:
    return html.escape(value, quote=True)


def _cell_xml(ref: str, value, *, style_id: int | None = None) -> str:
    style_attr = f' s="{style_id}"' if style_id not in (None, 0) else ""
    if value is None:
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"{style_attr}><v>{"1" if value else "0"}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{_escape_xml(str(value))}</t></is></c>'


def _style_catalog():
    return {
        "base": 0,
        "title": 1,
        "subtitle": 2,
        "header": 3,
        "meta_label": 4,
        "meta_value": 5,
        "wrap": 6,
        "status_ok": 7,
        "status_review": 8,
        "status_missing": 9,
        "status_invalid": 10,
        "number_1": 11,
        "number_3": 12,
        "number_4": 13,
        "field": 14,
    }


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><sz val="11"/><color rgb="FF101828"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="14"/><color rgb="FF101828"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="9">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF8FAFC"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F3B63"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFEFF6FF"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFECFDF3"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF7D6"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFEF3F2"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFEE4E2"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFD0D5DD"/></left>
      <right style="thin"><color rgb="FFD0D5DD"/></right>
      <top style="thin"><color rgb="FFD0D5DD"/></top>
      <bottom style="thin"><color rgb="FFD0D5DD"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="15">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
    <xf numFmtId="0" fontId="0" fillId="2" borderId="0" xfId="0" applyFill="1"/>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="2" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="2" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>
    <xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="6" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="7" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="8" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
    <xf numFmtId="2" fontId="0" fillId="2" borderId="1" xfId="0" applyNumberFormat="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="4" fontId="0" fillId="2" borderId="1" xfId="0" applyNumberFormat="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="10" fontId="0" fillId="2" borderId="1" xfId="0" applyNumberFormat="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>"""


def _table_header_style(column_name: str) -> str:
    if str(column_name or "").strip().lower() in {"field / parameter", "field key", "section"}:
        return "field"
    return "header"


def _value_style(column_name: str, value) -> str:
    text = str(column_name or "").strip().lower()
    status_text = str(value or "").strip().lower()
    if text in {"status", "overall status", "report state", "save status", "severity", "creation status"} or text.endswith("status"):
        return _STATUS_STYLE_KEY.get(status_text, "meta_value")
    if text in {"message", "issues", "technical summary", "preview summary", "notes", "resolution / save outcome", "highlights"}:
        return "wrap"
    if text in {"field / parameter", "field key", "section"}:
        return "field"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if any(token in text for token in ("cda", " b", " c")):
            return "number_4"
        if any(token in text for token in ("vde", "abc_total", "abc_net")):
            return "number_3"
        return "number_1"
    return "meta_value"


def _sheet_blocks(report_model: dict) -> list[dict]:
    metadata = dict(report_model.get("metadata") or {})
    summary_counts = dict(report_model.get("summary_counts") or {})
    return [
        {
            "name": "SUMMARY",
            "title": "EcoDrive VDE Request Report",
            "metadata": [
                ("Report state", report_model.get("report_state")),
                ("Request schema version", report_model.get("request_schema_version")),
                ("Template version", metadata.get("template_version")),
                ("Source filename", metadata.get("source_filename")),
                ("Baseline ID / name", " | ".join(str(item) for item in (metadata.get("baseline_id"), metadata.get("baseline_name")) if not is_blank(item)) or _DASH),
                ("Generated at", metadata.get("generated_at")),
                ("Save operation ID", metadata.get("save_operation_id")),
                ("Save status", metadata.get("save_status")),
            ],
            "tables": [
                ("Validation summary", [dict(zip(("Metric", "Value"), item)) for item in summary_counts.items()]),
                ("Technical comparison", list(report_model.get("summary_rows") or [])),
                ("Change highlights", list(report_model.get("highlight_rows") or [])),
            ],
        },
        {
            "name": "REQUEST",
            "title": "REQUEST",
            "metadata": [
                ("Source type", metadata.get("source_type")),
                ("Source filename", metadata.get("source_filename")),
                ("Report version", report_model.get("report_version")),
            ],
            "tables": [
                ("Request audit trail", list(report_model.get("request_rows") or [])),
            ],
        },
        {
            "name": "RESULTS",
            "title": "RESULTS",
            "metadata": [
                ("Overall status", metadata.get("resolution_overall_status")),
                ("Save status", metadata.get("save_status")),
            ],
            "tables": [
                ("Technical results", list(report_model.get("result_rows") or [])),
            ],
        },
        {
            "name": "COMPONENTS",
            "title": "COMPONENTS",
            "metadata": [
                ("Save operation ID", metadata.get("save_operation_id")),
                ("Report state", report_model.get("report_state")),
            ],
            "tables": [
                ("Component actions and provenance", list(report_model.get("component_rows") or [])),
            ],
        },
        {
            "name": "VALIDATION",
            "title": "VALIDATION",
            "metadata": [
                ("Overall status", metadata.get("resolution_overall_status")),
                ("Save operation status", metadata.get("save_status")),
                ("Generated at", metadata.get("generated_at")),
            ],
            "tables": [
                ("Issues and outcomes", list(report_model.get("validation_rows") or [])),
            ],
        },
    ]


def _rows_to_sheet_xml(block: dict, style_ids: dict[str, int]) -> tuple[str, int, str]:
    rows_xml: list[str] = []
    max_col = 0
    row_index = 1
    freeze_row = 2
    autofilter_ref = ""
    column_widths: defaultdict[int, int] = defaultdict(lambda: 12)

    def add_row(values: list[tuple[object, int]]):
        nonlocal row_index, max_col
        cells = []
        for column_index, (value, style_id) in enumerate(values):
            ref = f"{_col_letter(column_index)}{row_index}"
            if value is not None:
                column_widths[column_index] = max(column_widths[column_index], min(max(len(str(value)) + 2, 10), 42))
            cells.append(_cell_xml(ref, value, style_id=style_id))
            max_col = max(max_col, column_index + 1)
        rows_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
        row_index += 1

    add_row([(block.get("title"), style_ids["title"])])
    for label, value in list(block.get("metadata") or []):
        add_row([(label, style_ids["meta_label"]), (value if not is_blank(value) else _DASH, style_ids["meta_value"])])
    add_row([(None, style_ids["base"])])

    for table_title, rows in list(block.get("tables") or []):
        add_row([(table_title, style_ids["subtitle"])])
        if not rows:
            add_row([("No rows", style_ids["meta_value"])])
            add_row([(None, style_ids["base"])])
            continue
        headers = list(rows[0].keys())
        header_row_index = row_index
        add_row([(header, style_ids[_table_header_style(header)]) for header in headers])
        for row in rows:
            values = []
            for header in headers:
                value = dict(row or {}).get(header)
                if isinstance(value, str) and value.strip() == "":
                    value = None
                values.append((value if not is_blank(value) else None, style_ids[_value_style(header, value)]))
            add_row(values)
        if not autofilter_ref:
            autofilter_ref = f"A{header_row_index}:{_col_letter(len(headers)-1)}{row_index - 1}"
            freeze_row = header_row_index + 1
        add_row([(None, style_ids["base"])])

    cols_xml = []
    for column_index in range(max_col):
        width = column_widths[column_index]
        cols_xml.append(
            f'<col min="{column_index + 1}" max="{column_index + 1}" width="{max(10, min(width, 42))}" customWidth="1"/>'
        )
    pane = f'<sheetViews><sheetView workbookViewId="0"><pane ySplit="{max(freeze_row - 1, 1)}" topLeftCell="A{freeze_row}" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
    autofilter_xml = f'<autoFilter ref="{autofilter_ref}"/>' if autofilter_ref else ""
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{pane}"
        f"<cols>{''.join(cols_xml)}</cols>"
        f"<sheetData>{''.join(rows_xml)}</sheetData>"
        f"{autofilter_xml}"
        "</worksheet>"
    )
    return xml, freeze_row, autofilter_ref


def generate_vde_request_report_xlsx(
    report_model,
    output=None,
):
    model = deepcopy(dict(report_model or {}))
    style_ids = _style_catalog()
    blocks = _sheet_blocks(model)
    workbook_views = '<bookViews><workbookView activeTab="0"/></bookViews>'
    sheets_xml = []
    rels_xml = []
    worksheet_xml_map: dict[str, str] = {}
    for index, block in enumerate(blocks, start=1):
        sheet_name = str(block.get("name") or f"Sheet{index}")
        sheets_xml.append(
            f'<sheet name="{_escape_xml(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>'
        )
        rels_xml.append(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        )
        worksheet_xml_map[f"xl/worksheets/sheet{index}.xml"], _, _ = _rows_to_sheet_xml(block, style_ids)

    content_types = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(1, len(blocks) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            + "".join(content_types)
            + "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"{workbook_views}<sheets>{''.join(sheets_xml)}</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rels_xml)
            + "</Relationships>",
        )
        archive.writestr("xl/styles.xml", _styles_xml())
        for path, xml in worksheet_xml_map.items():
            archive.writestr(path, xml)

    payload = buffer.getvalue()
    if output is None:
        return payload
    if hasattr(output, "write"):
        output.write(payload)
        return output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return output_path


__all__ = [
    "VDE_REQUEST_REPORT_VERSION",
    "build_request_equivalent_draft_from_state",
    "build_vde_request_report_filename",
    "build_vde_request_report_model",
    "generate_vde_request_report_xlsx",
]
