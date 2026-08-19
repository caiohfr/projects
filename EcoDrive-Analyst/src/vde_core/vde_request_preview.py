from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json


_DASH = "—"
_BASELINE_CONTEXT_KEYS = (
    "selected_baseline_vde_id",
    "legislation",
    "category",
    "electrification",
    "transmission_type",
    "drive_type",
    "fuel_type",
    "make",
    "model",
    "year",
    "cycle_name",
    "mass_kg",
    "test_mass_kg",
    "payload_kg",
    "weight_dist_fr_pct",
    "inertia_class",
    "CdA",
    "frontal_area_m2",
    "front_tire_id",
    "rear_tire_id",
    "tire_db_id",
    "tire_code",
    "rrc_N_per_kN",
    "tire_load_mass_basis",
    "tire_A_final",
    "tire_B_final",
    "tire_C_final",
    "tire_calc_source",
    "smerf",
    "trans_A_coef_N",
    "trans_B_coef_Npkph",
    "trans_C_coef_Npkph2",
    "brake_A",
    "brake_B",
    "brake_C",
    "axle_hub_A",
    "axle_hub_B",
    "axle_hub_C",
    "parasitic_A",
    "parasitic_B",
    "parasitic_C",
    "A",
    "B",
    "C",
)
_DOMAIN_ORDER = ("mass", "aero", "tire", "transmission", "brake", "axle_hubs", "parasitic")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "__class__") and value.__class__.__name__ == "DataFrame":
        return f"<DataFrame rows={len(value)} cols={len(getattr(value, 'columns', []))}>"
    return value


def _sanitize_baseline_context(baseline_context: dict | None) -> dict:
    payload = {}
    source = dict(baseline_context or {})
    for key in _BASELINE_CONTEXT_KEYS:
        if key in source:
            payload[key] = _json_safe(source.get(key))
    return payload


def _sanitize_workbook_state(workbook_state: dict | None) -> dict:
    state = dict(workbook_state or {})
    import_meta = dict(state.get("vde_request_import") or {})
    scenarios = []
    for item in list(state.get("scenarios") or []):
        scenarios.append(
            {
                "key": str(item.get("key") or ""),
                "label": str(item.get("label") or ""),
                "role": str(item.get("role") or ""),
            }
        )
    columns = {}
    for column_id, column in dict(state.get("columns") or {}).items():
        payload = dict(column or {})
        columns[str(column_id)] = {
            "kind": payload.get("kind"),
            "walk_from": payload.get("walk_from"),
            "line_source": payload.get("line_source"),
            "selected_vde_id": payload.get("selected_vde_id"),
            "direct": _json_safe(dict(payload.get("direct") or {})),
            "printed_overrides": _json_safe(dict(payload.get("printed_overrides") or {})),
            "baseline_overrides": _json_safe(dict(payload.get("baseline_overrides") or {})),
        }
    proposals = {}
    for column_id, domain_map in dict(state.get("proposals") or {}).items():
        proposals[str(column_id)] = {}
        for domain_key, proposal in dict(domain_map or {}).items():
            payload = dict(proposal or {})
            proposals[str(column_id)][str(domain_key)] = {
                "id": payload.get("id"),
                "proposal_type": payload.get("proposal_type") or payload.get("type"),
                "label": payload.get("label"),
                "details": _json_safe(dict(payload.get("details") or {})),
            }
    return {
        "baseline_printed": _json_safe(dict(import_meta.get("baseline_printed") or {})),
        "baseline_corrections": _json_safe(dict(import_meta.get("baseline_corrections") or {})),
        "effective_baseline": _json_safe(dict(import_meta.get("effective_baseline") or {})),
        "scenarios": scenarios,
        "columns": columns,
        "proposals": proposals,
    }


def build_request_resolution_fingerprint(workbook_state: dict | None, baseline_context: dict | None = None) -> str:
    payload = {
        "workbook": _sanitize_workbook_state(workbook_state),
        "baseline_context": _sanitize_baseline_context(baseline_context),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _fmt_number(value, *, digits: int = 3, dash: str = _DASH) -> str:
    if value in (None, ""):
        return dash
    try:
        number = float(value)
    except Exception:
        return str(value)
    text = f"{number:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _fmt_mass(value) -> str:
    return _fmt_number(value, digits=1)


def _fmt_cda(value) -> str:
    return _fmt_number(value, digits=4)


def _fmt_energy(value) -> str:
    if isinstance(value, dict):
        return _fmt_number(value.get("mj_per_km"), digits=3)
    return _fmt_number(value, digits=3)


def _fmt_abc_triplet(payload: dict | None) -> tuple[str, str, str]:
    data = dict(payload or {})
    return (
        _fmt_number(data.get("A"), digits=3),
        _fmt_number(data.get("B"), digits=5),
        _fmt_number(data.get("C"), digits=6),
    )


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
        parts = [str(_compact(item)) for item in value if item not in (None, "", [], {})]
        return " | ".join(parts) if parts else _DASH
    return str(value)


def build_validation_summary(resolution_result: dict | None) -> dict:
    result = dict(resolution_result or {})
    proposal_results = list(result.get("proposal_results") or [])
    counter = Counter(str(item.get("status") or "OK") for item in proposal_results)
    warning_count = sum(len(list(dict(item.get("preview_summary") or {}).get("warnings") or [])) for item in proposal_results)
    return {
        "overall_status": str(result.get("status") or "OK"),
        "proposal_count": len(proposal_results),
        "ok_count": counter.get("OK", 0),
        "review_count": counter.get("Review", 0),
        "missing_count": counter.get("Missing", 0),
        "invalid_count": counter.get("Invalid", 0),
        "blocked_count": counter.get("Blocked", 0),
        "warning_count": warning_count,
        "issue_count": len(list(result.get("issues") or [])),
    }


def validation_allows_save(validation_summary: dict | None) -> bool:
    summary = dict(validation_summary or {})
    if not summary:
        return False
    for key in ("missing_count", "invalid_count", "blocked_count"):
        try:
            if int(summary.get(key) or 0) > 0:
                return False
        except Exception:
            return False
    overall = str(summary.get("overall_status") or "").strip()
    if overall in {"Missing", "Invalid", "Blocked", "", "Pending"}:
        return False
    return True


def _proposal_label_map(resolution_result: dict | None) -> dict[str, str]:
    labels = {}
    for item in list(dict(resolution_result or {}).get("proposal_results") or []):
        proposal_id = str(item.get("proposal_id") or "")
        label = str(item.get("source_column") or proposal_id)
        if proposal_id:
            labels[proposal_id] = label
    return labels


def build_request_comparison_rows(resolution_result: dict | None) -> list[dict]:
    result = dict(resolution_result or {})
    rows: list[dict] = []
    baseline = dict(result.get("baseline") or {})
    baseline_effective = dict(baseline.get("effective") or {})
    baseline_resolved = dict(dict(result.get("resolved_columns") or {}).get("baseline") or {})
    total_a, total_b, total_c = _fmt_abc_triplet(baseline_resolved.get("initial_abc_total") or baseline_resolved.get("abc_total"))
    baseline_status = "Review" if list(baseline.get("corrected_fields") or []) else "OK"
    rows.append(
        {
            "Scenario": "Baseline",
            "Status": baseline_status,
            "Walk From": _DASH,
            "Mass [kg]": _fmt_mass(baseline_resolved.get("test_mass_kg") or baseline_resolved.get("mass_kg") or baseline_effective.get("test_mass_kg") or baseline_effective.get("mass_kg")),
            "CdA [m^2]": _fmt_cda(baseline_resolved.get("CdA") or baseline_effective.get("cda_m2")),
            "ABC_TOTAL A [N]": total_a,
            "ABC_TOTAL B [N/kph]": total_b,
            "ABC_TOTAL C [N/kph^2]": total_c,
            "VDE_TOTAL [MJ/km]": _DASH,
            "ABC_NET A [N]": _DASH,
            "ABC_NET B [N/kph]": _DASH,
            "ABC_NET C [N/kph^2]": _DASH,
            "VDE_NET [MJ/km]": _DASH,
            "Issues": "0",
        }
    )
    label_map = _proposal_label_map(result)
    for proposal in list(result.get("proposal_results") or []):
        snapshot = dict(proposal.get("resolved_snapshot") or {})
        total_a, total_b, total_c = _fmt_abc_triplet(proposal.get("abc_total"))
        net_a, net_b, net_c = _fmt_abc_triplet(proposal.get("abc_net"))
        walk_from = dict(proposal.get("walk_from") or {})
        walk_from_label = str(walk_from.get("label") or walk_from.get("column_id") or _DASH)
        if str(walk_from.get("column_id") or "") in label_map:
            walk_from_label = label_map[str(walk_from.get("column_id") or "")]
        rows.append(
            {
                "Scenario": str(proposal.get("source_column") or proposal.get("proposal_id") or "Requested"),
                "Status": str(proposal.get("status") or "OK"),
                "Walk From": walk_from_label,
                "Mass [kg]": _fmt_mass(
                    dict(snapshot.get("resolved_mass_setup") or {}).get("resolved_mass_used_kg")
                    or dict(snapshot.get("resolved_mass_setup") or {}).get("test_mass_kg")
                    or snapshot.get("test_mass_kg")
                    or snapshot.get("mass_kg")
                ),
                "CdA [m^2]": _fmt_cda(snapshot.get("CdA")),
                "ABC_TOTAL A [N]": total_a,
                "ABC_TOTAL B [N/kph]": total_b,
                "ABC_TOTAL C [N/kph^2]": total_c,
                "VDE_TOTAL [MJ/km]": _fmt_energy(dict(proposal.get("vde_results") or {}).get("total")),
                "ABC_NET A [N]": net_a,
                "ABC_NET B [N/kph]": net_b,
                "ABC_NET C [N/kph^2]": net_c,
                "VDE_NET [MJ/km]": _fmt_energy(dict(proposal.get("vde_results") or {}).get("net")),
                "Issues": str(len(list(proposal.get("issues") or []))),
            }
        )
    return rows


def build_component_action_rows(proposal_result: dict | None) -> list[dict]:
    result = dict(proposal_result or {})
    rows: list[dict] = []
    for item in list(result.get("component_actions") or []):
        payload = dict(item or {})
        rows.append(
            {
                "Domain": str(payload.get("domain") or _DASH),
                "Action": str(payload.get("action") or _DASH),
                "Component ID": str(payload.get("component_id") or _DASH),
                "Requires confirmation": "Yes" if payload.get("requires_confirmation") else "No",
                "Issues": str(len(list(payload.get("issues") or []))),
                "Snapshot": _compact(payload.get("component_snapshot")),
            }
        )
    return rows


def build_proposal_preview_model(proposal_result: dict | None) -> dict:
    result = dict(proposal_result or {})
    source_snapshot = dict(result.get("source_snapshot") or {})
    resolved_snapshot = dict(result.get("resolved_snapshot") or {})
    preview_summary = dict(result.get("preview_summary") or {})
    domain_results = dict(result.get("domain_results") or {})
    engineering_rows = [
        {"Field": "Mass [kg]", "Value": _fmt_mass(dict(resolved_snapshot.get("resolved_mass_setup") or {}).get("resolved_mass_used_kg") or resolved_snapshot.get("test_mass_kg") or resolved_snapshot.get("mass_kg"))},
        {"Field": "CdA [m^2]", "Value": _fmt_cda(resolved_snapshot.get("CdA"))},
        {"Field": "ABC_TOTAL", "Value": _compact(result.get("abc_total"))},
        {"Field": "VDE_TOTAL [MJ/km]", "Value": _fmt_energy(dict(result.get("vde_results") or {}).get("total"))},
        {"Field": "ABC_NET", "Value": _compact(result.get("abc_net"))},
        {"Field": "VDE_NET [MJ/km]", "Value": _fmt_energy(dict(result.get("vde_results") or {}).get("net"))},
    ]
    domain_rows = []
    for domain_key in _DOMAIN_ORDER:
        payload = dict(domain_results.get(domain_key) or {})
        if not payload:
            continue
        domain_rows.append(
            {
                "Domain": domain_key,
                "Proposal type": str(payload.get("proposal_type") or "INHERIT"),
                "Status": str(payload.get("status") or "OK"),
                "Source": str(payload.get("source") or _DASH),
                "Requested": _compact(payload.get("requested_values")),
                "Resolved": _compact(payload.get("resolved_values")),
                "Notes": _compact(payload.get("notes")),
            }
        )
    validation_rows = []
    for issue in list(result.get("issues") or []):
        payload = dict(issue or {})
        validation_rows.append(
            {
                "Severity": str(payload.get("severity") or _DASH),
                "Code": str(payload.get("code") or _DASH),
                "Domain": str(payload.get("domain") or _DASH),
                "Field": str(payload.get("field_key") or _DASH),
                "Message": str(payload.get("message") or _DASH),
            }
        )
    for warning in list(preview_summary.get("warnings") or []):
        validation_rows.append(
            {
                "Severity": "warning",
                "Code": "preview_warning",
                "Domain": _DASH,
                "Field": _DASH,
                "Message": str(warning),
            }
        )
    audit_rows = []
    for domain_key in _DOMAIN_ORDER:
        payload = dict(domain_results.get(domain_key) or {})
        if not payload:
            continue
        audit_rows.append(
            {
                "Domain": domain_key,
                "Status": str(payload.get("status") or _DASH),
                "Source": str(payload.get("source") or _DASH),
                "Requested values": _compact(payload.get("requested_values")),
                "Resolved values": _compact(payload.get("resolved_values")),
                "Issues": str(len(list(payload.get("issues") or []))),
            }
        )
    return {
        "header": {
            "proposal_id": str(result.get("proposal_id") or ""),
            "requested_label": (
                f"Requested #{int(result.get('display_index'))}"
                if result.get("display_index") not in (None, "")
                else str(result.get("source_column") or _DASH)
            ),
            "display_index": result.get("display_index"),
            "source_column": str(result.get("source_column") or _DASH),
            "status": str(result.get("status") or "OK"),
            "walk_from": str(dict(result.get("walk_from") or {}).get("label") or dict(result.get("walk_from") or {}).get("column_id") or _DASH),
            "issues_count": len(list(result.get("issues") or [])),
        },
        "engineering_rows": engineering_rows,
        "domain_change_rows": domain_rows,
        "component_action_rows": build_component_action_rows(result),
        "validation_rows": validation_rows,
        "audit_rows": audit_rows,
        "source_snapshot": _compact(source_snapshot),
    }


def build_request_audit_rows(resolution_result: dict | None) -> list[dict]:
    rows: list[dict] = []
    result = dict(resolution_result or {})
    for proposal in list(result.get("proposal_results") or []):
        for domain_key, payload in dict(proposal.get("domain_results") or {}).items():
            item = dict(payload or {})
            rows.append(
                {
                    "Scenario": str(proposal.get("source_column") or proposal.get("proposal_id") or "Requested"),
                    "Proposal ID": str(proposal.get("proposal_id") or _DASH),
                    "Domain": domain_key,
                    "Status": str(item.get("status") or _DASH),
                    "Walk From": str(dict(proposal.get("walk_from") or {}).get("label") or dict(proposal.get("walk_from") or {}).get("column_id") or _DASH),
                    "Requested": _compact(item.get("requested_values")),
                    "Resolved": _compact(item.get("resolved_values")),
                    "Issues": str(len(list(item.get("issues") or []))),
                    "Source": str(item.get("source") or _DASH),
                }
            )
    return rows


__all__ = [
    "build_component_action_rows",
    "build_proposal_preview_model",
    "build_request_audit_rows",
    "build_request_comparison_rows",
    "build_request_resolution_fingerprint",
    "build_validation_summary",
    "validation_allows_save",
]
