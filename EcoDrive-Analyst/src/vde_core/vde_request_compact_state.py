from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from src.vde_app.components.vde_request_domain_editors import proposal_application_status, proposal_is_not_used, resolve_domain_display, sanitize_domain_inputs
from src.vde_core.vde_request_detail_mapping import detail_key_for_domain_field
from src.vde_core.vde_request_contract import (
    FIELD_KEY_ALIASES,
    TEMPLATE_PROPOSAL_MAP,
    VISIBLE_TEMPLATE_PROPOSAL_LABELS,
    VDE_REQUEST_SCHEMA_VERSION,
    is_blank,
    normalize_template_proposal_type,
    resolve_effective_baseline,
)
from src.vde_core.vde_not_used_modes import normalize_not_used_proposal_type
from src.vde_core.vde_tire_modes import canonical_tire_details, canonical_tire_proposal_type
from src.vde_core.test_mass import format_inertia_step_interval, inertia_step_for_mass


V22_SCHEMA_VERSION = "0.1"
V22_MAX_PROPOSALS = 30
V22_PROPOSAL_DOMAINS = ("mass", "aero", "tire", "transmission", "brake", "axle_hubs", "parasitic")
V22_SECTION_KEYS = ("baseline", "matrix", "inputs", "preview")
V22_CORRECTION_DISPOSITIONS = ("request_only", "save_as_new_baseline")
V22_TIRE_PRESSURE_UNIT_OPTIONS = ("kPa", "bar", "psi")
V22_BASELINE_SOURCE_TYPES = ("EXISTING_VDE", "NEW_TEST")
V22_PROPOSAL_METADATA_FIELDS = (
    "name",
    "description",
    "make",
    "model",
    "model_year",
    "category",
    "electrification",
    "transmission_type",
    "drive_type",
    "fuel_type",
    "legislation",
    "cycle_name",
)
V22_METADATA_READONLY_FIELDS = {"legislation", "cycle_name"}

V22_BASELINE_FIELDS = (
    "abc_total_source_ui",
    "selected_baseline_vde_id",
    "make",
    "model",
    "year",
    "legislation",
    "category",
    "electrification",
    "transmission_type",
    "drive_type",
    "fuel_type",
    "cycle_name",
    "notes",
    "mass_kg",
    "test_mass_kg",
    "test_mass_basis",
    "inertia_class",
    "payload_kg",
    "options_kg",
    "weight_dist_fr_pct",
    "gvwr_kg",
    "gcwr_kg",
    "trailer_mass_kg",
    "A",
    "B",
    "C",
    "cda_m2",
    "tire_db_id",
    "tire_code",
    "front_tire_id",
    "rear_tire_id",
    "rrc_N_per_kN",
    "front_pressure_psi",
    "rear_pressure_psi",
    "tire_load_mass_basis",
    "tire_A_final",
    "tire_B_final",
    "tire_C_final",
    "tire_calc_source",
    "transmission_component_db_id",
    "trans_A_coef_N",
    "trans_B_coef_Npkph",
    "trans_C_coef_Npkph2",
    "brake_component_db_id",
    "brake_A_coef_N",
    "brake_B_Npkph",
    "brake_C_coef_Npkph2",
    "axle_hubs_component_db_id",
    "axle_hub_A",
    "axle_hub_B",
    "axle_hub_C",
    "parasitic_component_db_id",
    "parasitic_A_coef_N",
    "parasitic_B_Npkph",
    "parasitic_C_coef_Npkph2",
)

_EXTRA_FIELD_ALIASES = {
    "selected_baseline_vde_id": ("id", "vde_id"),
    "A": ("coast_A_N", "ABC_TOTAL_A"),
    "B": ("coast_B_N_per_kph", "ABC_TOTAL_B"),
    "C": ("coast_C_N_per_kph2", "ABC_TOTAL_C"),
    "cycle_name": ("cycle", "cycle_name"),
    "notes": ("notes", "description"),
    "trans_A_coef_N": ("trans_A_loss",),
    "trans_B_coef_Npkph": ("trans_B_loss",),
    "trans_C_coef_Npkph2": ("trans_C_loss",),
    "brake_B_Npkph": ("brake_B_coef_Npkph",),
    "parasitic_B_Npkph": ("parasitic_B_coef_Npkph",),
}

_DETAIL_KEY_BY_DOMAIN_FIELD = {
    ("aero", "cda_m2", "AERO_ABSOLUTE_CDA", "Absolute CdA"): "new_CdA",
    ("aero", "cda_m2", "AERO_DELTA_CDA", "Delta CdA"): "delta_CdA",
    ("transmission", "trans_A_coef_N", "UPDATE_TRANS_DRAG_ABC", "Absolute ABC"): "new_trans_A",
    ("transmission", "trans_B_coef_Npkph", "UPDATE_TRANS_DRAG_ABC", "Absolute ABC"): "new_trans_B",
    ("transmission", "trans_C_coef_Npkph2", "UPDATE_TRANS_DRAG_ABC", "Absolute ABC"): "new_trans_C",
    ("transmission", "trans_A_coef_N", "UPDATE_TRANS_DRAG_ABC", "Delta ABC"): "delta_A",
    ("transmission", "trans_B_coef_Npkph", "UPDATE_TRANS_DRAG_ABC", "Delta ABC"): "delta_B",
    ("transmission", "trans_C_coef_Npkph2", "UPDATE_TRANS_DRAG_ABC", "Delta ABC"): "delta_C",
}


def _blank_proposal(proposal_id: str, display_index: int) -> dict:
    return {
        "proposal_id": proposal_id,
        "display_index": int(display_index),
        "name": "",
        "metadata_overrides": {},
        "metadata_source": "inherit",
        "walk_from": "baseline",
        "domains": {domain: {"proposal_type": "INHERIT"} for domain in V22_PROPOSAL_DOMAINS},
        "inputs": {},
    }


def _default_domain_input_state() -> dict:
    return {
        domain: {
            "revision": 0,
            "last_applied_at": None,
            "status": "not_configured",
            "proposal_statuses": {},
            "last_apply_message": None,
        }
        for domain in V22_PROPOSAL_DOMAINS
    }


def create_v22_state() -> dict:
    return {
        "schema_version": V22_SCHEMA_VERSION,
        "active_section": "baseline",
        "ui_preferences": {"tire_pressure_unit": None},
        "baseline": {
            "source_type": "EXISTING_VDE",
            "source_snapshot": {},
            "selected_vde_id": None,
            "loaded": False,
            "printed": {},
            "corrections": {},
            "effective": {},
            "correction_disposition": "request_only",
        },
        "proposals": [_blank_proposal("requested_1", 1), _blank_proposal("requested_2", 2)],
        "domain_input_state": _default_domain_input_state(),
        "preview": {"status": "not_run", "fingerprint": None, "result": None},
        "save": {"status": "pending", "result": None},
        "_next_proposal_seq": 3,
    }


def normalize_v22_state(state: dict | None) -> dict:
    source = deepcopy(dict(state or {}))
    if not source:
        return create_v22_state()

    default = create_v22_state()
    source["schema_version"] = str(source.get("schema_version") or V22_SCHEMA_VERSION)
    if source.get("active_section") not in V22_SECTION_KEYS:
        source["active_section"] = default["active_section"]

    ui_preferences = deepcopy(dict(default["ui_preferences"]))
    ui_preferences.update(deepcopy(dict(source.get("ui_preferences") or {})))
    tire_pressure_unit = str(ui_preferences.get("tire_pressure_unit") or "").strip()
    ui_preferences["tire_pressure_unit"] = tire_pressure_unit if tire_pressure_unit in V22_TIRE_PRESSURE_UNIT_OPTIONS else None
    source["ui_preferences"] = ui_preferences

    baseline = deepcopy(dict(default["baseline"]))
    baseline.update(deepcopy(dict(source.get("baseline") or {})))
    if baseline.get("source_type") not in V22_BASELINE_SOURCE_TYPES:
        baseline["source_type"] = "EXISTING_VDE"
    baseline["source_snapshot"] = deepcopy(dict(baseline.get("source_snapshot") or {}))
    if baseline.get("correction_disposition") not in V22_CORRECTION_DISPOSITIONS:
        baseline["correction_disposition"] = "request_only"
    source["baseline"] = baseline

    proposals = list(source.get("proposals") or [])
    if not proposals:
        proposals = deepcopy(default["proposals"])
    normalized_proposals = []
    seen_ids: set[str] = set()
    for index, item in enumerate(proposals, start=1):
        proposal = deepcopy(dict(item or {}))
        proposal_id = str(proposal.get("proposal_id") or f"requested_{index}").strip()
        if proposal_id in seen_ids:
            proposal_id = f"requested_{index}"
        seen_ids.add(proposal_id)
        proposal["proposal_id"] = proposal_id
        proposal.setdefault("name", "")
        proposal["metadata_overrides"] = deepcopy(dict(proposal.get("metadata_overrides") or {}))
        proposal["metadata_source"] = str(proposal.get("metadata_source") or "inherit").strip().lower()
        if proposal["metadata_source"] not in {"inherit", "existing_vde", "custom"}:
            proposal["metadata_source"] = "inherit"
        proposal.setdefault("walk_from", "baseline")
        domains = deepcopy(dict(proposal.get("domains") or {}))
        for domain in V22_PROPOSAL_DOMAINS:
            domains[domain] = _normalize_domain_payload(domain, domains.get(domain))
        proposal["domains"] = domains
        proposal["inputs"] = deepcopy(dict(proposal.get("inputs") or {}))
        transmission_payload = dict(domains.get("transmission") or {})
        transmission_inputs = dict(proposal["inputs"].get("transmission") or {})
        if str(transmission_payload.get("proposal_type") or "").upper() == "TRANS_LOSS_PCT":
            legacy_share = transmission_inputs.get("transmission_loss_pct")
            if is_blank(legacy_share):
                legacy_share = transmission_inputs.get("loss_pct")
            if not is_blank(legacy_share):
                # v2.2 Coastdown Share is always a share of Walk From ABC_TOTAL.
                # Migrate earlier compact drafts before they reach either display
                # resolution or the canonical resolver.
                transmission_inputs["transmission_loss_pct"] = legacy_share
                transmission_inputs["transmission_application_mode"] = "KEEP_TOTAL_FIXED"
                transmission_inputs["percent_basis"] = "SOURCE_ABC_TOTAL"
                transmission_inputs["rule_version"] = "COASTDOWN_SHARE_V1"
                proposal["inputs"]["transmission"] = transmission_inputs
        normalized_proposals.append(proposal)
    source["proposals"] = normalized_proposals[:V22_MAX_PROPOSALS]
    renumber_v22_proposals(source)

    domain_input_state = deepcopy(_default_domain_input_state())
    for domain, payload in dict(source.get("domain_input_state") or {}).items():
        if domain not in domain_input_state:
            continue
        item = deepcopy(dict(payload or {}))
        domain_input_state[domain].update(
            {
                "revision": int(item.get("revision") or 0),
                "last_applied_at": item.get("last_applied_at"),
                "status": item.get("status") or "not_configured",
                "proposal_statuses": deepcopy(dict(item.get("proposal_statuses") or {})),
                "last_apply_message": item.get("last_apply_message"),
            }
        )
    source["domain_input_state"] = domain_input_state

    preview = deepcopy(dict(default["preview"]))
    preview.update(deepcopy(dict(source.get("preview") or {})))
    save = deepcopy(dict(default["save"]))
    save.update(deepcopy(dict(source.get("save") or {})))
    source["preview"] = preview
    source["save"] = save

    suffixes = []
    for proposal in source["proposals"]:
        proposal_id = str(proposal.get("proposal_id") or "")
        if proposal_id.startswith("requested_"):
            try:
                suffixes.append(int(proposal_id.rsplit("_", 1)[-1]))
            except Exception:
                pass
    next_seq = source.get("_next_proposal_seq")
    try:
        next_seq = int(next_seq)
    except Exception:
        next_seq = 0
    source["_next_proposal_seq"] = max([next_seq, *(value + 1 for value in suffixes), 1])
    return source


def has_v22_tire_pressure_unit_override(state: dict | None) -> bool:
    normalized = normalize_v22_state(state)
    return str(dict(normalized.get("ui_preferences") or {}).get("tire_pressure_unit") or "").strip() in V22_TIRE_PRESSURE_UNIT_OPTIONS


def resolve_v22_tire_pressure_unit(state: dict | None, unit_system: str | None = None) -> str:
    normalized = normalize_v22_state(state)
    explicit = str(dict(normalized.get("ui_preferences") or {}).get("tire_pressure_unit") or "").strip()
    if explicit in V22_TIRE_PRESSURE_UNIT_OPTIONS:
        return explicit
    return "psi" if str(unit_system or "").strip() == "US customary" else "kPa"


def set_v22_tire_pressure_unit_preference(state: dict | None, pressure_unit: str | None) -> dict:
    next_state = normalize_v22_state(state)
    text = str(pressure_unit or "").strip()
    next_state.setdefault("ui_preferences", {})
    next_state["ui_preferences"]["tire_pressure_unit"] = text if text in V22_TIRE_PRESSURE_UNIT_OPTIONS else None
    return next_state


def renumber_v22_proposals(state: dict) -> dict:
    for index, proposal in enumerate(list(state.get("proposals") or []), start=1):
        dict(proposal)["display_index"] = index
        proposal["display_index"] = index
    return state


def add_v22_proposal(state: dict) -> dict:
    next_state = normalize_v22_state(state)
    proposals = list(next_state.get("proposals") or [])
    if len(proposals) >= V22_MAX_PROPOSALS:
        return next_state
    seq = int(next_state.get("_next_proposal_seq") or len(proposals) + 1)
    existing = {str(proposal.get("proposal_id") or "") for proposal in proposals}
    while f"requested_{seq}" in existing:
        seq += 1
    proposals.append(_blank_proposal(f"requested_{seq}", len(proposals) + 1))
    next_state["proposals"] = proposals
    next_state["_next_proposal_seq"] = seq + 1
    renumber_v22_proposals(next_state)
    mark_v22_preview_stale(next_state)
    return next_state


def remove_v22_proposal(state: dict, proposal_id: str) -> dict:
    next_state = normalize_v22_state(state)
    target = str(proposal_id or "")
    next_state["proposals"] = [proposal for proposal in next_state["proposals"] if str(proposal.get("proposal_id")) != target]
    renumber_v22_proposals(next_state)
    mark_v22_preview_stale(next_state)
    return next_state


def allowed_walk_from_options(state: dict, proposal_id: str) -> list[str]:
    normalized = normalize_v22_state(state)
    options = ["baseline"]
    for proposal in normalized.get("proposals") or []:
        current_id = str(proposal.get("proposal_id") or "")
        if current_id == str(proposal_id or ""):
            break
        options.append(current_id)
    return options


def apply_v22_baseline(state: dict, baseline_row: dict | None) -> dict:
    next_state = normalize_v22_state(state)
    row = deepcopy(dict(baseline_row or {}))
    printed = {}
    for field_key in V22_BASELINE_FIELDS:
        printed[field_key] = _value_from_aliases(row, field_key)
    if row and is_blank(printed.get("abc_total_source_ui")):
        printed["abc_total_source_ui"] = "Baseline ABC"
    selected_id = printed.get("selected_baseline_vde_id")
    source_snapshot = {field_key: printed.get(field_key) for field_key in V22_BASELINE_FIELDS if not is_blank(printed.get(field_key))}
    source_snapshot["baseline_source_type"] = "EXISTING_VDE"
    next_state["baseline"]["selected_vde_id"] = selected_id
    next_state["baseline"]["source_type"] = "EXISTING_VDE"
    next_state["baseline"]["source_snapshot"] = source_snapshot
    next_state["baseline"]["loaded"] = bool(row)
    next_state["baseline"]["printed"] = printed
    next_state["baseline"]["corrections"] = {}
    next_state["baseline"]["effective"] = resolve_v22_effective_baseline(next_state)
    clear_v22_runtime_state(next_state)
    return next_state


def build_new_test_canonical_baseline(source_inputs: dict | None) -> dict:
    inputs = deepcopy(dict(source_inputs or {}))
    legislation = str(inputs.get("legislation") or "").strip().upper()
    test_mass_kg = inputs.get("test_mass_kg")
    corrections = {
        "abc_total_source_ui": "From test coastdown",
        "selected_baseline_vde_id": None,
    }
    for field_key in (
        "legislation",
        "cycle_name",
        "make",
        "model",
        "year",
        "category",
        "electrification",
        "transmission_type",
        "drive_type",
        "fuel_type",
        "notes",
        "A",
        "B",
        "C",
        "test_mass_kg",
    ):
        value = inputs.get(field_key)
        if is_blank(value):
            continue
        corrections[field_key] = value
    if not is_blank(test_mass_kg):
        if legislation == "EPA":
            corrections["inertia_class"] = test_mass_kg
            corrections["test_mass_basis"] = "EPA_INERTIA_CLASS"
        else:
            corrections["test_mass_basis"] = "PHYSICAL_TEST_MASS"
    corrections["baseline_source_type"] = "NEW_TEST"
    next_state = normalize_v22_state({})
    next_state["baseline"]["source_type"] = "NEW_TEST"
    next_state["baseline"]["source_snapshot"] = deepcopy(corrections)
    next_state["baseline"]["loaded"] = bool(corrections)
    next_state["baseline"]["selected_vde_id"] = None
    next_state["baseline"]["printed"] = {}
    next_state["baseline"]["corrections"] = deepcopy(corrections)
    next_state["baseline"]["effective"] = resolve_v22_effective_baseline(next_state)
    return {
        "printed": {},
        "corrections": deepcopy(corrections),
        "effective": deepcopy(dict(next_state["baseline"]["effective"] or {})),
        "source_snapshot": deepcopy(corrections),
    }


def apply_v22_new_test_baseline(state: dict, source_inputs: dict | None) -> dict:
    next_state = normalize_v22_state(state)
    payload = build_new_test_canonical_baseline(source_inputs)
    next_state["baseline"]["source_type"] = "NEW_TEST"
    next_state["baseline"]["source_snapshot"] = deepcopy(dict(payload.get("source_snapshot") or {}))
    next_state["baseline"]["selected_vde_id"] = None
    next_state["baseline"]["loaded"] = bool(payload.get("effective"))
    next_state["baseline"]["printed"] = deepcopy(dict(payload.get("printed") or {}))
    next_state["baseline"]["corrections"] = deepcopy(dict(payload.get("corrections") or {}))
    next_state["baseline"]["effective"] = deepcopy(dict(payload.get("effective") or {}))
    clear_v22_runtime_state(next_state)
    return next_state


def apply_v22_corrections(state: dict, corrections: dict | None) -> dict:
    next_state = normalize_v22_state(state)
    cleaned = {}
    for field_key in V22_BASELINE_FIELDS:
        if field_key not in dict(corrections or {}):
            continue
        value = dict(corrections or {}).get(field_key)
        if is_blank(value):
            continue
        cleaned[field_key] = value
    next_state["baseline"]["corrections"] = cleaned
    next_state["baseline"]["effective"] = resolve_v22_effective_baseline(next_state)
    mark_v22_preview_stale(next_state)
    return next_state


def resolve_v22_effective_baseline(state: dict) -> dict:
    baseline = dict(dict(state or {}).get("baseline") or {})
    printed = deepcopy(dict(baseline.get("printed") or {}))
    corrections = deepcopy(dict(baseline.get("corrections") or {}))
    keys = list(dict.fromkeys([*V22_BASELINE_FIELDS, *printed.keys(), *corrections.keys()]))
    effective = {field_key: resolve_effective_baseline(printed.get(field_key), corrections.get(field_key)) for field_key in keys}
    source_type = str(baseline.get("source_type") or "EXISTING_VDE").strip().upper()
    effective["baseline_source_type"] = source_type if source_type in V22_BASELINE_SOURCE_TYPES else "EXISTING_VDE"
    if is_blank(effective.get("abc_total_source_ui")):
        effective["abc_total_source_ui"] = "From test coastdown" if effective["baseline_source_type"] == "NEW_TEST" else "Baseline ABC"
    if effective["baseline_source_type"] == "NEW_TEST":
        effective["selected_baseline_vde_id"] = None
    effective.update(resolve_v22_baseline_mass_review({"baseline": {"effective": effective}}))
    return effective


def apply_v22_proposal_matrix(state: dict, matrix_rows: list[dict] | dict | None) -> dict:
    next_state = normalize_v22_state(state)
    rows_by_id = _rows_by_proposal_id(matrix_rows)
    proposals = []
    changed_domains: set[str] = set()
    for proposal in next_state["proposals"]:
        proposal = deepcopy(proposal)
        row = rows_by_id.get(str(proposal.get("proposal_id") or ""), {})
        if row.get("remove"):
            continue
        if "walk_from" in row:
            next_walk_from = str(row.get("walk_from") or "baseline")
            if next_walk_from != str(proposal.get("walk_from") or "baseline"):
                changed_domains.update(
                    domain
                    for domain, payload in dict(proposal.get("domains") or {}).items()
                    if str(dict(payload or {}).get("proposal_type") or "INHERIT") != "INHERIT"
                )
            proposal["walk_from"] = next_walk_from
        for domain in V22_PROPOSAL_DOMAINS:
            if domain not in row:
                continue
            previous = _normalize_domain_payload(domain, proposal["domains"].get(domain))
            updated = _normalize_domain_payload(domain, row.get(domain))
            proposal["domains"][domain] = updated
            if _domain_input_signature(previous) != _domain_input_signature(updated):
                proposal["inputs"].pop(domain, None)
                changed_domains.add(domain)
        proposals.append(proposal)
    next_state["proposals"] = proposals
    renumber_v22_proposals(next_state)
    for domain in changed_domains:
        _mark_domain_input_state_stale(next_state, domain)
    mark_v22_preview_stale(next_state)
    return next_state


def apply_v22_domain_inputs(state: dict, domain: str, values_by_proposal: dict | None) -> dict:
    next_state = normalize_v22_state(state)
    domain_key = str(domain or "").strip()
    rows_by_id = {str(key): deepcopy(dict(value or {})) for key, value in dict(values_by_proposal or {}).items()}
    for proposal in next_state["proposals"]:
        proposal_id = str(proposal.get("proposal_id") or "")
        if proposal_id not in rows_by_id:
            continue
        domain_payload = dict(dict(proposal.get("domains") or {}).get(domain_key) or {})
        proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
        selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
        cleaned = sanitize_domain_inputs(
            domain_key,
            proposal_type,
            selection_mode,
            dict(rows_by_id.get(proposal_id) or {}),
        )
        proposal.setdefault("inputs", {})
        if cleaned:
            proposal["inputs"][domain_key] = cleaned
        else:
            proposal["inputs"].pop(domain_key, None)
    _refresh_domain_input_state(next_state, domain_key, applied=True)
    mark_v22_preview_stale(next_state)
    return next_state


def apply_v22_proposal_metadata(
    state: dict,
    proposal_id: str,
    overrides: dict | None,
    *,
    metadata_source: str | None = None,
) -> dict:
    next_state = normalize_v22_state(state)
    target_id = str(proposal_id or "")
    cleaned = {}
    for field_key, value in dict(overrides or {}).items():
        key = str(field_key or "")
        if key not in V22_PROPOSAL_METADATA_FIELDS or key in V22_METADATA_READONLY_FIELDS:
            continue
        if is_blank(value):
            continue
        cleaned[key] = value
    for proposal in list(next_state.get("proposals") or []):
        if str(proposal.get("proposal_id") or "") != target_id:
            continue
        proposal["metadata_overrides"] = cleaned
        if "name" in cleaned:
            proposal["name"] = str(cleaned["name"])
        if metadata_source is not None:
            source = str(metadata_source or "inherit").strip().lower()
            proposal["metadata_source"] = source if source in {"inherit", "existing_vde", "custom"} else "inherit"
        break
    # Persistence identity does not affect the resolved engineering request.
    next_state.setdefault("save", {})
    next_state["save"]["status"] = "pending"
    next_state["save"]["result"] = None
    return next_state


def mark_v22_preview_stale(state: dict) -> dict:
    state.setdefault("preview", {})
    state.setdefault("save", {})
    state["preview"]["status"] = "stale"
    state["preview"]["fingerprint"] = None
    state["preview"]["result"] = None
    state["save"]["status"] = "pending"
    state["save"]["result"] = None
    state.pop("report", None)
    return state


def clear_v22_runtime_state(state: dict) -> dict:
    state["preview"] = {"status": "not_run", "fingerprint": None, "result": None}
    state["save"] = {"status": "pending", "result": None}
    state.pop("report", None)
    return state


def _mark_domain_input_state_stale(state: dict, domain: str) -> None:
    payload = dict(dict(state.get("domain_input_state") or {}).get(domain) or {})
    proposal_statuses = deepcopy(dict(payload.get("proposal_statuses") or {}))
    state.setdefault("domain_input_state", {})
    state["domain_input_state"][domain] = {
        "revision": int(payload.get("revision") or 0),
        "last_applied_at": payload.get("last_applied_at"),
        "status": "stale_after_matrix_change",
        "proposal_statuses": proposal_statuses,
        "last_apply_message": "Proposal Matrix changed. Re-apply this domain.",
    }


def _refresh_domain_input_state(state: dict, domain: str, *, applied: bool) -> None:
    normalized = normalize_v22_state(state)
    baseline = dict(dict(normalized.get("baseline") or {}).get("effective") or {})
    proposal_statuses = {}
    resolved_by_id = {}
    ready_count = 0
    incomplete_count = 0
    direct_count = 0
    for proposal in list(normalized.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        walk_from_id = str(proposal.get("walk_from") or "baseline")
        source_display = deepcopy(dict(baseline)) if walk_from_id == "baseline" else deepcopy(dict(resolved_by_id.get(walk_from_id) or baseline))
        domain_payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
        proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
        selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
        inputs = sanitize_domain_inputs(domain, proposal_type, selection_mode, dict(dict(proposal.get("inputs") or {}).get(domain) or {}))
        resolved_display = resolve_domain_display(
            domain,
            source_display,
            {"domains": {domain: domain_payload}, "inputs": {domain: inputs}},
        )
        resolved_by_id[proposal_id] = deepcopy(resolved_display)
        status_payload = proposal_application_status(domain, proposal_type, selection_mode, inputs, resolved_display)
        if proposal_type == "INHERIT":
            proposal_statuses[proposal_id] = status_payload
            continue
        if proposal_is_not_used(proposal_type, selection_mode):
            proposal_statuses[proposal_id] = status_payload
            continue
        direct_count += 1
        if not applied and not inputs:
            status_payload = {"status": "not_configured", "message": "Not configured", "missing_fields": [], "issues": []}
        if status_payload["status"] == "applied_ready":
            ready_count += 1
        elif status_payload["status"] == "applied_incomplete":
            incomplete_count += 1
        proposal_statuses[proposal_id] = status_payload
    domain_state = dict(dict(normalized.get("domain_input_state") or {}).get(domain) or {})
    status = "not_configured"
    if direct_count:
        status = "applied_incomplete" if incomplete_count else "applied_ready"
        if not applied:
            status = "not_configured"
    normalized["domain_input_state"][domain] = {
        "revision": int(domain_state.get("revision") or 0) + (1 if applied else 0),
        "last_applied_at": datetime.now().strftime("%H:%M:%S") if applied else domain_state.get("last_applied_at"),
        "status": status,
        "proposal_statuses": proposal_statuses,
        "last_apply_message": (
            f"{domain.title()} inputs applied — {ready_count} ready, {incomplete_count} incomplete."
            if applied
            else domain_state.get("last_apply_message")
        ),
    }
    state["domain_input_state"] = normalized["domain_input_state"]


def build_v22_canonical_request_draft(state: dict) -> dict:
    normalized = normalize_v22_state(state)
    baseline = dict(normalized.get("baseline") or {})
    effective_baseline = deepcopy(dict(baseline.get("effective") or {}))
    proposals = []
    resolved_by_id: dict[str, dict[str, dict]] = {}
    resolved_metadata_by_id: dict[str, dict] = {}
    baseline_metadata = _baseline_metadata(effective_baseline)
    for proposal in normalized.get("proposals") or []:
        proposal_id = str(proposal.get("proposal_id") or "")
        walk_from_id = str(proposal.get("walk_from") or "baseline")
        walk_from = {
            "kind": "baseline" if walk_from_id == "baseline" else "proposal",
            "proposal_id": None if walk_from_id == "baseline" else walk_from_id,
            "source_column": "Baseline" if walk_from_id == "baseline" else _source_column_for_id(normalized, walk_from_id),
        }
        source_metadata = deepcopy(baseline_metadata) if walk_from_id == "baseline" else deepcopy(dict(resolved_metadata_by_id.get(walk_from_id) or baseline_metadata))
        effective_metadata = _effective_proposal_metadata(proposal, source_metadata)
        domain_requests = {}
        domain_resolved = {}
        for domain in V22_PROPOSAL_DOMAINS:
            source_display = deepcopy(effective_baseline) if walk_from_id == "baseline" else deepcopy(dict(resolved_by_id.get(walk_from_id, {}).get(domain) or effective_baseline))
            domain_payload = dict(dict(proposal.get("domains") or {}).get(domain) or {})
            proposal_type = str(domain_payload.get("proposal_type") or "INHERIT")
            selection_mode = str(domain_payload.get("selection_mode") or proposal_type)
            inputs = sanitize_domain_inputs(domain, proposal_type, selection_mode, dict(dict(proposal.get("inputs") or {}).get(domain) or {}))
            resolved_display = resolve_domain_display(
                domain,
                source_display,
                {"domains": {domain: domain_payload}, "inputs": {domain: inputs}},
            )
            domain_resolved[domain] = deepcopy(resolved_display)
            domain_requests[domain] = _domain_request_payload(
                domain,
                proposal,
                domain_payload,
                resolved_display=resolved_display,
            )
        resolved_by_id[proposal_id] = domain_resolved
        resolved_metadata_by_id[proposal_id] = deepcopy(effective_metadata)
        proposals.append(
            {
                "proposal_id": proposal_id,
                "display_index": int(proposal.get("display_index") or len(proposals) + 1),
                "source_column": f"Requested #{int(proposal.get('display_index') or len(proposals) + 1)}",
                "name": proposal.get("name", ""),
                "effective_metadata": effective_metadata,
                "metadata_overrides": deepcopy(dict(proposal.get("metadata_overrides") or {})),
                "walk_from": walk_from,
                "domain_requests": domain_requests,
                "issues": _proposal_issues(normalized, proposal),
            }
        )
    return {
        "schema_version": VDE_REQUEST_SCHEMA_VERSION,
        "template_version": V22_SCHEMA_VERSION,
        "source": {"source_type": "VDE Setup v2.2", "interface": "compact_request"},
        "baseline_source_type": baseline.get("source_type") or "EXISTING_VDE",
        "baseline_source_snapshot": deepcopy(dict(baseline.get("source_snapshot") or {})),
        "baseline_printed": deepcopy(dict(baseline.get("printed") or {})),
        "baseline_corrections": deepcopy(dict(baseline.get("corrections") or {})),
        "effective_baseline": effective_baseline,
        "baseline_correction_disposition": baseline.get("correction_disposition") or "request_only",
        "proposals": proposals,
        "issues": [],
    }


def _rows_by_proposal_id(matrix_rows: list[dict] | dict | None) -> dict[str, dict]:
    if isinstance(matrix_rows, dict):
        iterable = list(matrix_rows.values())
    else:
        iterable = list(matrix_rows or [])
    rows = {}
    for row in iterable:
        item = dict(row or {})
        proposal_id = str(item.get("proposal_id") or "").strip()
        if proposal_id:
            rows[proposal_id] = item
    return rows


def _normalize_domain_payload(domain: str, payload: Any) -> dict:
    if isinstance(payload, dict):
        raw_value = payload.get("selection_mode") or payload.get("label") or payload.get("raw_proposal_type") or payload.get("proposal_type")
    else:
        raw_value = payload
    raw_value = "Inherit" if is_blank(raw_value) else raw_value
    normalized = normalize_template_proposal_type(domain, raw_value)
    if not normalized.get("ok"):
        proposal_type = normalize_not_used_proposal_type(domain, str(raw_value or "INHERIT").strip().upper() or "INHERIT", str(raw_value or ""))
        return {"proposal_type": proposal_type, "selection_mode": str(raw_value or proposal_type)}
    out = {
        "proposal_type": normalize_not_used_proposal_type(domain, normalized.get("proposal_type") or "INHERIT", normalized.get("selection_mode") or normalized.get("template_label") or str(raw_value)),
        "selection_mode": normalized.get("selection_mode") or normalized.get("template_label") or str(raw_value),
    }
    details = deepcopy(dict(normalized.get("details") or {}))
    if details:
        out["details_seed"] = details
    if normalized.get("has_internal_equivalent") is False:
        out["has_internal_equivalent"] = False
    return out


def _domain_input_signature(payload: dict) -> tuple:
    item = dict(payload or {})
    return (str(item.get("proposal_type") or "INHERIT"), str(item.get("selection_mode") or ""))


def _domain_request_payload(
    domain: str,
    proposal: dict,
    domain_payload: dict | None,
    *,
    resolved_display: dict | None = None,
) -> dict:
    payload = _normalize_domain_payload(domain, domain_payload)
    selection_mode = str(payload.get("selection_mode") or "Inherit")
    proposal_type = normalize_not_used_proposal_type(domain, str(payload.get("proposal_type") or "INHERIT"), selection_mode)
    if domain == "tire":
        proposal_type = canonical_tire_proposal_type(proposal_type)
    raw_values = deepcopy(dict(dict(proposal.get("inputs") or {}).get(domain) or {}))
    details_seed = deepcopy(dict(payload.get("details_seed") or {}))
    for field_key, value in raw_values.items():
        if is_blank(value):
            continue
        details_seed.setdefault(_detail_key(domain, field_key, proposal_type, selection_mode), value)
    if domain == "tire":
        details_seed = canonical_tire_details(proposal_type, details_seed)
    if domain == "mass" and proposal_type == "MASS_TWC_SHIFT":
        resolved = dict(resolved_display or {})
        if not is_blank(resolved.get("target_mass_kg")):
            details_seed.setdefault("target_mass_kg", resolved.get("target_mass_kg"))
    return {
        "domain": domain,
        "raw_proposal_type": selection_mode,
        "proposal_type": proposal_type,
        "selection_mode": selection_mode,
        "raw_values": raw_values,
        "proposal_details_seed": details_seed,
        "normalized_proposal": deepcopy(payload),
        "has_internal_equivalent": payload.get("has_internal_equivalent", True),
        "issues": [],
    }


def _proposal_issues(state: dict, proposal: dict) -> list[dict]:
    proposal_id = str(proposal.get("proposal_id") or "")
    walk_from = str(proposal.get("walk_from") or "baseline")
    if walk_from in allowed_walk_from_options(state, proposal_id):
        return []
    return [
        {
            "severity": "review",
            "code": "invalid_walk_from",
            "message": f"Walk From '{walk_from}' does not reference Baseline or a previous proposal.",
        }
    ]


def _source_column_for_id(state: dict, proposal_id: str) -> str:
    for proposal in list(state.get("proposals") or []):
        if str(proposal.get("proposal_id") or "") == str(proposal_id or ""):
            return f"Requested #{int(proposal.get('display_index') or 0)}"
    return str(proposal_id or "")


def _detail_key(domain: str, field_key: str, proposal_type: str, selection_mode: str) -> str:
    mapped = _DETAIL_KEY_BY_DOMAIN_FIELD.get((domain, field_key, proposal_type, selection_mode))
    if mapped:
        return mapped
    return detail_key_for_domain_field(domain, proposal_type, field_key, {"selection_mode": selection_mode})


def _value_from_aliases(payload: dict, field_key: str):
    aliases = list(FIELD_KEY_ALIASES.get(field_key, (field_key,))) + list(_EXTRA_FIELD_ALIASES.get(field_key, ()))
    for alias in dict.fromkeys([field_key, *aliases]):
        if alias in payload and not is_blank(payload.get(alias)):
            return payload.get(alias)
    if field_key in payload:
        return payload.get(field_key)
    return None


def proposal_type_labels_by_domain() -> dict[str, list[str]]:
    labels = {}
    for domain, entries in TEMPLATE_PROPOSAL_MAP.items():
        if domain not in V22_PROPOSAL_DOMAINS:
            continue
        visible = list(VISIBLE_TEMPLATE_PROPOSAL_LABELS.get(domain) or entries.keys())
        labels[domain] = [label for label in visible if label in entries]
    return labels


def resolve_v22_baseline_mass_review(state: dict) -> dict:
    baseline = dict(dict(state or {}).get("baseline") or {})
    effective = dict(baseline.get("effective") or {})
    legislation = str(effective.get("legislation") or "").strip().upper()
    if legislation != "EPA":
        return {}
    mass_kg = _safe_float(effective.get("mass_kg"))
    inertia_class = _safe_float(effective.get("inertia_class"))
    if mass_kg is None:
        return {}
    step = inertia_step_for_mass(mass_kg)
    if not step:
        return {
            "baseline_mass_review_status": "Invalid",
            "baseline_mass_review_notes": "Corrected curb mass is outside the canonical EPA TWC table.",
            "baseline_mass_suggested_inertia_class": None,
            "baseline_mass_target_twc_interval": None,
        }
    suggested = step.get("inertia_class_kg")
    interval = format_inertia_step_interval(step)
    if inertia_class is None:
        status = "Review"
        notes = f"Compatible EPA ETW / TWC would be {suggested:.1f} for interval {interval}."
    elif float(inertia_class) == float(suggested):
        status = "OK"
        notes = f"EPA ETW / TWC is compatible with corrected curb mass in interval {interval}."
    else:
        status = "Review"
        notes = (
            f"Corrected curb mass suggests EPA ETW / TWC {suggested:.1f} in interval {interval}. "
            f"Current inertia class remains {inertia_class:.1f}."
        )
    return {
        "baseline_mass_review_status": status,
        "baseline_mass_review_notes": notes,
        "baseline_mass_suggested_inertia_class": suggested,
        "baseline_mass_target_twc_interval": interval,
    }


def _safe_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def resolve_v22_metadata_contexts(state: dict) -> dict[str, dict]:
    normalized = normalize_v22_state(state)
    baseline = dict(dict(normalized.get("baseline") or {}).get("effective") or {})
    baseline_metadata = _baseline_metadata(baseline)
    contexts = {}
    resolved_by_id = {}
    for proposal in list(normalized.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        walk_from_id = str(proposal.get("walk_from") or "baseline")
        source_metadata = deepcopy(baseline_metadata) if walk_from_id == "baseline" else deepcopy(dict(resolved_by_id.get(walk_from_id) or baseline_metadata))
        effective_metadata = _effective_proposal_metadata(proposal, source_metadata)
        resolved_by_id[proposal_id] = deepcopy(effective_metadata)
        contexts[proposal_id] = {
            "source_metadata": source_metadata,
            "effective_metadata": effective_metadata,
            "overrides": deepcopy(dict(proposal.get("metadata_overrides") or {})),
            "walk_from": walk_from_id,
        }
    return contexts


def _baseline_metadata(effective_baseline: dict) -> dict:
    return {
        "name": "",
        "description": str(effective_baseline.get("notes") or ""),
        "make": effective_baseline.get("make"),
        "model": effective_baseline.get("model"),
        "model_year": effective_baseline.get("year") or effective_baseline.get("model_year"),
        "category": effective_baseline.get("category"),
        "electrification": effective_baseline.get("electrification"),
        "transmission_type": effective_baseline.get("transmission_type"),
        "drive_type": effective_baseline.get("drive_type"),
        "fuel_type": effective_baseline.get("fuel_type"),
        "legislation": effective_baseline.get("legislation"),
        "cycle_name": effective_baseline.get("cycle_name"),
    }


def _effective_proposal_metadata(proposal: dict, source_metadata: dict) -> dict:
    effective = deepcopy(dict(source_metadata or {}))
    overrides = deepcopy(dict(proposal.get("metadata_overrides") or {}))
    proposal_name = str(proposal.get("name") or "").strip()
    if proposal_name:
        overrides["name"] = proposal_name
    for field_key in V22_PROPOSAL_METADATA_FIELDS:
        if field_key in V22_METADATA_READONLY_FIELDS:
            continue
        value = overrides.get(field_key)
        if is_blank(value):
            continue
        effective[field_key] = value
    return effective
