from __future__ import annotations

from src.vde_core.vde_not_used_modes import is_not_used_proposal

LOOKUP_PROPOSAL_TYPES = {
    "TRANS_METADATA_ONLY",
    "BRAKE_METADATA_ONLY",
    "AXLE_HUB_METADATA_ONLY",
    "PARASITIC_METADATA_ONLY",
}
def canonical_component_mode(domain: str, proposal_type: str, selection_mode: str | None = None, details: dict | None = None) -> str:
    domain_key = str(domain or "").strip().lower()
    type_key = str(proposal_type or "").strip().upper()
    mode = str(selection_mode or "").strip().lower()
    payload = dict(details or {})
    legacy_change = str(payload.get("change_mode") or "").strip().lower()
    legacy_method = str(payload.get("method") or "").strip().lower()

    if type_key == "INHERIT":
        return "INHERIT"
    if type_key in LOOKUP_PROPOSAL_TYPES:
        return "LOOKUP"
    if is_not_used_proposal(domain_key, type_key, selection_mode):
        return "NOT_USED"
    if type_key == "TRANS_LOSS_PCT":
        return "LOSS_PERCENT"
    if domain_key == "brake" and type_key == "BRAKE_DRAG_CHANGE":
        if mode == "residual torque" or legacy_method == "residual torque":
            return "RESIDUAL_TORQUE"
    if mode == "delta abc" or legacy_change == "delta abc":
        return "DELTA_ABC"
    if mode == "absolute abc" or legacy_change == "absolute abc":
        return "ABSOLUTE_ABC"
    if type_key in {"UPDATE_TRANS_DRAG_ABC", "BRAKE_DRAG_CHANGE", "AXLE_HUB_DRAG_CHANGE", "PARASITIC_LOSS_CHANGE"}:
        return "ABSOLUTE_ABC"
    return type_key or "UNKNOWN"


def legacy_component_mode_fields(domain: str, proposal_type: str, selection_mode: str | None = None, details: dict | None = None) -> dict:
    mode = canonical_component_mode(domain, proposal_type, selection_mode, details)
    if mode == "DELTA_ABC":
        return {"change_mode": "Delta ABC", **({"method": "Brake ABC"} if str(domain or "").strip().lower() == "brake" else {})}
    if mode == "ABSOLUTE_ABC":
        return {"change_mode": "Absolute ABC", **({"method": "Brake ABC"} if str(domain or "").strip().lower() == "brake" else {})}
    if mode == "RESIDUAL_TORQUE":
        return {"method": "Residual torque"}
    return {}
