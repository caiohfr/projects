from __future__ import annotations

import re
from copy import deepcopy
from typing import Sequence


V21_STATUS_PRIORITY = {
    "invalid": 5,
    "blocked": 5,
    "missing": 4,
    "review": 3,
    "pending": 3,
    "partial": 3,
    "draft": 2,
    "ok": 1,
    "ready": 1,
    "defined": 1,
    "derived": 1,
    "inherited": 0,
    "inherit": 0,
    "not used": 0,
    "not_used": 0,
    "unavailable": 0,
}


def v21_status_rank(status: str) -> int:
    return V21_STATUS_PRIORITY.get(str(status or "Inherited").strip().lower(), 0)


def rollup_v21_statuses(statuses: list[str], *, default: str = "Inherited") -> str:
    cleaned = [str(item or "").strip() for item in statuses if str(item or "").strip()]
    if not cleaned:
        return default
    return max(cleaned, key=v21_status_rank)


def _column_order(workbook_state: dict) -> list[str]:
    scenarios = list(workbook_state.get("scenarios") or [])
    ordered = [str(item.get("key") or "").strip() for item in scenarios if str(item.get("key") or "").strip()]
    if "baseline" in ordered:
        return ordered
    return ["baseline", *[item for item in ordered if item != "baseline"]]


def _column_label(column_id: str, workbook_state: dict) -> str:
    for item in list(workbook_state.get("scenarios") or []):
        if str(item.get("key") or "").strip() == column_id:
            return str(item.get("label") or column_id).strip() or column_id
    if column_id == "baseline":
        return "Baseline"
    if column_id.startswith("walked_"):
        suffix = column_id.removeprefix("walked_")
        return f"Walked #{suffix}"
    return column_id


def _allowed_walk_from(column_id: str, workbook_state: dict) -> list[str]:
    ordered = _column_order(workbook_state)
    if column_id == "baseline" or column_id not in ordered:
        return []
    return ordered[: ordered.index(column_id)]


def _proposal_fallback_label(proposal_type: str, type_labels: dict[str, str] | None = None) -> str:
    proposal_type = str(proposal_type or "").strip().upper()
    if not proposal_type:
        return ""
    if type_labels and proposal_type in type_labels:
        return str(type_labels[proposal_type])
    return proposal_type


def _proposal_badge(domain_state: dict, type_labels: dict[str, str] | None = None) -> str:
    proposal_id = str(domain_state.get("id") or "").strip()
    match = re.search(r"(\d+)$", proposal_id)
    prefix = f"Prop #{match.group(1)}" if match else "Prop"
    label = str(domain_state.get("label") or "").strip() or _proposal_fallback_label(domain_state.get("proposal_type"), type_labels)
    if not label:
        return prefix
    return f"{prefix} · {label}"


def _normalize_domain_state(domain_key: str, payload: dict | None, *, type_labels: dict[str, str] | None = None) -> dict:
    value = dict(payload or {})
    proposal_type = str(value.get("proposal_type") or value.get("type") or "INHERIT").strip().upper() or "INHERIT"
    mode = str(value.get("mode") or ("inherited" if proposal_type == "INHERIT" else "direct")).strip().lower()
    direct = mode == "direct" and proposal_type != "INHERIT"
    status = str(value.get("status") or ("Inherited" if not direct else "Draft")).strip() or ("Inherited" if not direct else "Draft")
    notes = value.get("notes") or []
    if not isinstance(notes, list):
        notes = [notes]
    normalized = {
        "domain": domain_key,
        "id": str(value.get("id") or "").strip(),
        "mode": "direct" if direct else "inherited",
        "proposal_type": proposal_type,
        "label": str(value.get("label") or "").strip(),
        "details": deepcopy(dict(value.get("details") or {})),
        "status": status,
        "notes": [str(item).strip() for item in notes if str(item).strip()],
    }
    normalized["badge_text"] = _proposal_badge(normalized, type_labels)
    return normalized


def resolve_v21_domain(
    domain_key: str,
    source_state: dict | None,
    proposal: dict | None,
    baseline_overrides: dict | None = None,
    *,
    source_column: str | None = None,
    current_column: str | None = None,
    type_labels: dict[str, str] | None = None,
) -> dict:
    direct_state = _normalize_domain_state(domain_key, proposal, type_labels=type_labels)
    if direct_state["mode"] == "direct":
        direct_state["source_column"] = current_column
        direct_state["inherited_from"] = source_column
        direct_state["display_status"] = direct_state["status"]
        return direct_state

    inherited = _normalize_domain_state(domain_key, source_state, type_labels=type_labels)
    if inherited["mode"] == "direct":
        inherited["display_status"] = "Inherited"
    else:
        inherited["status"] = "Inherited"
        inherited["display_status"] = "Inherited"
    inherited["mode"] = "inherited"
    inherited["inherited_from"] = source_column
    inherited["source_column"] = inherited.get("source_column") or source_column
    inherited["baseline_overrides"] = deepcopy(dict(baseline_overrides or {}))
    return inherited


def resolve_v21_column(
    workbook_state: dict,
    column_id: str,
    baseline_state: dict | None = None,
    *,
    resolved_columns: dict[str, dict] | None = None,
    domain_keys: list[str] | tuple[str, ...] | None = None,
    type_labels: dict[str, str] | None = None,
) -> dict:
    resolved_columns = resolved_columns or {}
    columns = {str(key): dict(value or {}) for key, value in dict(workbook_state.get("columns") or {}).items()}
    column = dict(columns.get(column_id) or {})
    kind = "baseline" if column_id == "baseline" else "walked"
    label = str(column.get("label") or _column_label(column_id, workbook_state)).strip() or column_id
    ordered = _column_order(workbook_state)
    requested_walk_from = str(column.get("walk_from") or "").strip() or None
    allowed = _allowed_walk_from(column_id, workbook_state)
    walk_from_status = "OK"
    walk_from_note = ""

    if kind == "baseline":
        resolved_walk_from = None
    else:
        fallback = allowed[-1] if allowed else "baseline"
        if requested_walk_from in allowed:
            resolved_walk_from = requested_walk_from
        else:
            resolved_walk_from = fallback
            if requested_walk_from:
                walk_from_status = "Invalid"
                walk_from_note = f"Invalid Walk From '{requested_walk_from}' resolved to {label if fallback == column_id else _column_label(fallback, workbook_state)}."

    if domain_keys is None:
        inferred = set()
        for value in columns.values():
            inferred.update(dict(value.get("domains") or {}).keys())
        domain_keys = list(inferred)

    direct_domains = {
        domain_key: _normalize_domain_state(domain_key, dict(dict(column.get("domains") or {}).get(domain_key) or {}), type_labels=type_labels)
        for domain_key in domain_keys
    }
    direct_domains = {key: value for key, value in direct_domains.items() if value["mode"] == "direct"}

    if kind == "baseline":
        effective_domains = {}
        for domain_key in domain_keys:
            effective_domains[domain_key] = resolve_v21_domain(
                domain_key,
                None,
                dict(direct_domains.get(domain_key) or {}),
                dict(column.get("baseline_overrides") or {}),
                source_column=None,
                current_column=column_id,
                type_labels=type_labels,
            )
        proposal_effective_labels = [item["badge_text"] for item in direct_domains.values() if item.get("badge_text")]
    else:
        source_column_state = dict(resolved_columns.get(resolved_walk_from or "baseline") or {})
        source_domains = dict(source_column_state.get("effective_domains") or {})
        effective_domains = {}
        for domain_key in domain_keys:
            effective_domains[domain_key] = resolve_v21_domain(
                domain_key,
                dict(source_domains.get(domain_key) or {}),
                dict(direct_domains.get(domain_key) or {}),
                dict(column.get("baseline_overrides") or {}),
                source_column=resolved_walk_from,
                current_column=column_id,
                type_labels=type_labels,
            )
        proposal_effective_labels = list(source_column_state.get("proposal_effective_labels") or [])
        for domain_key in domain_keys:
            direct_domain = direct_domains.get(domain_key)
            if direct_domain and direct_domain.get("badge_text"):
                proposal_effective_labels.append(str(direct_domain["badge_text"]))

    proposal_direct_labels = [item["badge_text"] for item in direct_domains.values() if item.get("badge_text")]
    review_status = rollup_v21_statuses([item.get("status") for item in direct_domains.values()], default="Inherited")
    effective_status = rollup_v21_statuses([item.get("status") for item in effective_domains.values()], default="Inherited")
    if walk_from_status == "Invalid":
        review_status = rollup_v21_statuses([review_status, "Invalid"], default="Invalid")
        effective_status = rollup_v21_statuses([effective_status, "Invalid"], default="Invalid")

    return {
        "column_id": column_id,
        "label": label,
        "kind": kind,
        "order_index": ordered.index(column_id) if column_id in ordered else -1,
        "walk_from_requested": requested_walk_from,
        "walk_from": resolved_walk_from,
        "walk_from_status": walk_from_status,
        "walk_from_note": walk_from_note,
        "direct_domains": direct_domains,
        "effective_domains": effective_domains,
        "proposal_direct_labels": proposal_direct_labels,
        "proposal_effective_labels": proposal_effective_labels,
        "proposal_direct": " + ".join(proposal_direct_labels),
        "proposal_effective": " + ".join(proposal_effective_labels),
        "review_status": review_status,
        "effective_status": effective_status,
        "baseline_state": deepcopy(dict(baseline_state or {})),
    }


def resolve_v21_workbook(
    workbook_state: dict,
    baseline_state: dict | None = None,
    *,
    domain_keys: list[str] | tuple[str, ...] | None = None,
    type_labels: dict[str, str] | None = None,
) -> dict:
    ordered = _column_order(workbook_state)
    resolved_columns: dict[str, dict] = {}
    for column_id in ordered:
        resolved_columns[column_id] = resolve_v21_column(
            workbook_state,
            column_id,
            baseline_state,
            resolved_columns=resolved_columns,
            domain_keys=domain_keys,
            type_labels=type_labels,
        )
    return {
        "column_order": ordered,
        "columns": resolved_columns,
        "baseline_state": deepcopy(dict(baseline_state or {})),
    }


def _save_plan_note_from_requests(requests: list[dict]) -> str:
    labels = []
    for item in requests:
        column_label = str(item.get("column_label") or item.get("column_id") or "").strip()
        domain_label = str(item.get("domain_label") or item.get("domain") or "").strip()
        if column_label and domain_label:
            labels.append(f"{column_label} / {domain_label}")
    if not labels:
        return "Baseline update requested."
    return "Baseline update requested by " + ", ".join(labels) + "."


def _missing_fields(details: dict | None, field_ids: Sequence[str]) -> list[str]:
    payload = dict(details or {})
    return [field_id for field_id in field_ids if payload.get(field_id) in (None, "")]


def validate_v21_absolute_reference(
    details: dict | None,
    *,
    new_fields: Sequence[str],
    baseline_fields: Sequence[str],
    has_reference: bool,
    reference_source: str | None,
    baseline_update_requested: bool = False,
    absolute_label: str = "Absolute ABC",
) -> dict:
    missing_new = _missing_fields(details, new_fields)
    if missing_new:
        return {
            "status": "Missing",
            "warnings": [],
            "missing_fields": missing_new,
        }

    if not has_reference:
        missing_baseline = _missing_fields(details, baseline_fields)
        return {
            "status": "Missing",
            "warnings": [],
            "missing_fields": missing_baseline or list(baseline_fields),
        }

    if str(reference_source or "").strip().lower() == "assume_zero":
        return {
            "status": "Review",
            "warnings": [f"{absolute_label} is assuming zero as the baseline reference."],
            "missing_fields": [],
        }

    if str(reference_source or "").strip().lower() == "manual_override":
        warnings = [f"{absolute_label} is using a manual baseline override."]
        if baseline_update_requested:
            warnings.append("Baseline update requested; Preview & Save must confirm provenance before persisting.")
        return {
            "status": "Review",
            "warnings": warnings,
            "missing_fields": [],
        }

    return {
        "status": "OK",
        "warnings": [],
        "missing_fields": [],
    }


def resolve_v21_reference_value(
    inherited_value,
    *,
    manual_value=None,
    assume_zero: bool = False,
) -> dict:
    if manual_value not in (None, ""):
        return {
            "value": manual_value,
            "source": "manual_override",
            "has_reference": True,
        }
    if inherited_value not in (None, ""):
        return {
            "value": inherited_value,
            "source": "inherited",
            "has_reference": True,
        }
    if assume_zero:
        return {
            "value": 0.0,
            "source": "assume_zero",
            "has_reference": True,
        }
    return {
        "value": None,
        "source": "missing",
        "has_reference": False,
    }


def resolve_v21_reference_triplet(
    inherited_values: Sequence[object] | None,
    *,
    manual_values: Sequence[object] | None = None,
    assume_zero: bool = False,
) -> dict:
    inherited_values = tuple(inherited_values or (None, None, None))
    manual_values = tuple(manual_values or (None, None, None))
    resolved_values: list[object] = []
    used_manual = False
    used_inherited = False
    used_zero = False
    for inherited_value, manual_value in zip(inherited_values, manual_values):
        if manual_value not in (None, ""):
            resolved_values.append(manual_value)
            used_manual = True
        elif inherited_value not in (None, ""):
            resolved_values.append(inherited_value)
            used_inherited = True
        elif assume_zero:
            resolved_values.append(0.0)
            used_zero = True
        else:
            resolved_values.append(None)
    if all(item not in (None, "") for item in resolved_values):
        source = "manual_override" if used_manual else ("assume_zero" if used_zero and not used_inherited else "inherited")
        return {
            "values": tuple(resolved_values),
            "source": source,
            "has_reference": True,
        }
    return {
        "values": tuple(resolved_values),
        "source": "missing",
        "has_reference": False,
    }


def resolve_v21_delta_scalar(
    *,
    new_value,
    reference_value,
    original_baseline_value=None,
) -> dict:
    if new_value in (None, "") or reference_value in (None, ""):
        return {
            "local_delta": None,
            "accumulated_delta": None,
        }
    local_delta = float(new_value) - float(reference_value)
    accumulated_delta = None
    if original_baseline_value not in (None, ""):
        accumulated_delta = float(new_value) - float(original_baseline_value)
    return {
        "local_delta": local_delta,
        "accumulated_delta": accumulated_delta,
    }


def resolve_v21_delta_triplet(
    *,
    new_values: Sequence[object] | None,
    reference_values: Sequence[object] | None,
    original_baseline_values: Sequence[object] | None = None,
) -> dict:
    new_values = tuple(new_values or (None, None, None))
    reference_values = tuple(reference_values or (None, None, None))
    original_baseline_values = tuple(original_baseline_values or (None, None, None))
    if any(item in (None, "") for item in (*new_values, *reference_values)):
        return {
            "local_delta": (None, None, None),
            "accumulated_delta": (None, None, None),
        }
    local_delta = tuple(float(new_value) - float(reference_value) for new_value, reference_value in zip(new_values, reference_values))
    if original_baseline_values and all(item not in (None, "") for item in original_baseline_values):
        accumulated_delta = tuple(float(new_value) - float(baseline_value) for new_value, baseline_value in zip(new_values, original_baseline_values))
    else:
        accumulated_delta = (None, None, None)
    return {
        "local_delta": local_delta,
        "accumulated_delta": accumulated_delta,
    }


def v21_baseline_update_requests(
    resolved_workbook: dict,
    *,
    domain_labels: dict[str, str] | None = None,
) -> list[dict]:
    requests: list[dict] = []
    columns = dict(resolved_workbook.get("columns") or {})
    for column_id, column_state in columns.items():
        for domain_key, domain_state in dict(column_state.get("direct_domains") or {}).items():
            details = dict(domain_state.get("details") or {})
            if not bool(details.get("baseline_update_requested")):
                continue
            requests.append(
                {
                    "column_id": column_id,
                    "column_label": str(column_state.get("label") or column_id),
                    "domain": domain_key,
                    "domain_label": str((domain_labels or {}).get(domain_key) or domain_key),
                    "proposal_type": str(domain_state.get("proposal_type") or ""),
                    "proposal_label": str(domain_state.get("label") or domain_state.get("badge_text") or "").strip(),
                    "status": str(domain_state.get("status") or "Review"),
                }
            )
    return requests


def build_v21_save_plan(
    resolved_workbook: dict,
    previews: dict[str, dict] | None,
    *,
    baseline_is_existing: bool = False,
    baseline_target_id: int | None = None,
    selected_target: str | None = None,
    saved_targets: dict[str, int | None] | None = None,
    domain_labels: dict[str, str] | None = None,
) -> dict:
    previews = {str(key): dict(value or {}) for key, value in dict(previews or {}).items()}
    selected_target = str(selected_target or "").strip() or None
    saved_targets = {str(key): value for key, value in dict(saved_targets or {}).items()}
    columns = dict(resolved_workbook.get("columns") or {})
    order = list(resolved_workbook.get("column_order") or columns.keys())
    baseline_requests = v21_baseline_update_requests(resolved_workbook, domain_labels=domain_labels)
    rows: list[dict] = []
    blocked_labels: list[str] = []
    review_labels: list[str] = []

    for column_id in order:
        column_state = dict(columns.get(column_id) or {})
        preview = dict(previews.get(column_id) or {})
        preview_ok = bool(preview.get("ok"))
        review_status = str(column_state.get("review_status") or "Inherited")
        effective_status = str(column_state.get("effective_status") or review_status)
        direct_domains = dict(column_state.get("direct_domains") or {})
        direct_count = len(direct_domains)
        target_vde_id = saved_targets.get(column_id)
        is_selected = selected_target == column_id
        requires_confirmation = False
        blocked = False

        if column_id == "baseline":
            target_vde_id = baseline_target_id
            if baseline_is_existing and baseline_requests:
                action = "update_existing"
                status = "Review" if preview_ok else "Pending"
                notes = _save_plan_note_from_requests(baseline_requests)
                requires_confirmation = True
                blocked = not preview_ok
            else:
                action = "no_update"
                status = "Inherited" if baseline_is_existing else "Pending"
                notes = "Baseline remains reference-only in the Save All plan."
                if not baseline_is_existing:
                    notes = "Baseline is a new reference line; use Save selected if you want to persist it now."
        else:
            if direct_count == 0:
                action = "skip"
                status = "Inherited"
                notes = "No direct proposal in this walked column."
            elif effective_status in {"Invalid", "Missing"}:
                action = "skip"
                status = effective_status
                notes = "Resolve proposal issues before saving this column."
                blocked = True
            elif not preview_ok:
                action = "skip"
                status = "Pending"
                notes = "Preview is not ready yet; compute Preview All first."
                blocked = True
            else:
                action = "update_existing" if target_vde_id else "create_new"
                status = "Review" if review_status in {"Review", "Draft"} else "OK"
                notes = "Ready to save effective snapshot."
                if review_status in {"Review", "Draft"}:
                    requires_confirmation = True
                    notes = "Column has review items; save only with explicit confirmation."

        if blocked:
            blocked_labels.append(str(column_state.get("label") or column_id))
        elif requires_confirmation:
            review_labels.append(str(column_state.get("label") or column_id))

        rows.append(
            {
                "column_id": column_id,
                "label": str(column_state.get("label") or column_id),
                "action": action,
                "target_vde_id": target_vde_id,
                "status": status,
                "selected": is_selected,
                "requires_confirmation": requires_confirmation,
                "preview_ok": preview_ok,
                "direct_proposal_count": direct_count,
                "review_status": review_status,
                "effective_status": effective_status,
                "notes": notes,
            }
        )

    return {
        "rows": rows,
        "baseline_update_requests": baseline_requests,
        "selected_target": selected_target,
        "can_save_all": not blocked_labels,
        "has_saveable_rows": any(str(item.get("action") or "") in {"create_new", "update_existing"} for item in rows),
        "requires_confirmation": bool(review_labels),
        "blocked_columns": blocked_labels,
        "review_columns": review_labels,
    }
