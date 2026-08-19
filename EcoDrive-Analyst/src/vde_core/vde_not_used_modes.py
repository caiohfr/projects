from __future__ import annotations


NOT_USED_PROPOSAL_TYPE_BY_DOMAIN = {
    "aero": "AERO_NOT_USED",
    "tire": "TIRE_METADATA_ONLY",
    "transmission": "TRANS_LOSS_NOT_AVAILABLE",
    "brake": "BRAKE_NOT_USED",
    "axle_hubs": "AXLE_HUB_NOT_USED",
    "parasitic": "PARASITIC_NOT_USED",
}

EXPLICIT_NOT_USED_PROPOSAL_TYPES = set(NOT_USED_PROPOSAL_TYPE_BY_DOMAIN.values())
ACTIVE_NOT_USED_UI_DOMAINS = {"tire", "transmission", "brake", "axle_hubs", "parasitic"}


def normalize_not_used_proposal_type(domain: str | None, proposal_type: str | None, selection_mode: str | None = None) -> str:
    domain_key = str(domain or "").strip().lower()
    type_key = str(proposal_type or "").strip().upper() or "INHERIT"
    mode_key = str(selection_mode or "").strip().lower()
    if type_key in EXPLICIT_NOT_USED_PROPOSAL_TYPES:
        return type_key
    if mode_key in {"not used", "not_used"}:
        return NOT_USED_PROPOSAL_TYPE_BY_DOMAIN.get(domain_key, type_key)
    return type_key


def is_not_used_proposal(domain: str | None, proposal_type: str | None, selection_mode: str | None = None) -> bool:
    return normalize_not_used_proposal_type(domain, proposal_type, selection_mode) in EXPLICIT_NOT_USED_PROPOSAL_TYPES


def not_used_ui_is_active(domain: str | None) -> bool:
    return str(domain or "").strip().lower() in ACTIVE_NOT_USED_UI_DOMAINS
