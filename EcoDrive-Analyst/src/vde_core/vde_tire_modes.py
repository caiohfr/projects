from __future__ import annotations

from copy import deepcopy

from src.vde_core.vde_request_contract import is_blank


def canonical_tire_proposal_type(proposal_type: str | None) -> str:
    text = str(proposal_type or "INHERIT").strip().upper() or "INHERIT"
    if text == "TIRE_SMERF_RRC_CHANGE":
        return "TIRE_TARGET_RRC"
    return text


def canonical_tire_details(proposal_type: str | None, details: dict | None) -> dict:
    raw_type = str(proposal_type or "INHERIT").strip().upper() or "INHERIT"
    proposal_type = canonical_tire_proposal_type(raw_type)
    payload = deepcopy(dict(details or {}))
    canonical: dict = {}

    for field_key, aliases in (
        ("tire_db_id", ("tire_db_id",)),
        ("tire_source_vde_id", ("tire_source_vde_id",)),
        ("tire_code", ("tire_code", "new_tire_code", "baseline_tire_code")),
        ("tire_snapshot", ("tire_snapshot",)),
        ("rrc_N_per_kN", ("rrc_N_per_kN", "baseline_RRC_optional")),
        ("front_pressure_psi", ("front_pressure_psi", "psi_front")),
        ("rear_pressure_psi", ("rear_pressure_psi", "psi_rear")),
        ("tire_load_mass_basis", ("tire_load_mass_basis", "load_basis")),
        ("tire_review_status", ("tire_review_status",)),
        ("source_rrc_N_per_kN", ("source_rrc_N_per_kN", "baseline_RRC_optional")),
        ("tire_improvement_pct", ("tire_improvement_pct", "improvement_pct")),
    ):
        value = _first_present_value(payload, aliases)
        if not is_blank(value):
            canonical[field_key] = value

    target_rrc = _first_present_value(payload, ("target_rrc_N_per_kN",))
    if is_blank(target_rrc):
        if raw_type == "TIRE_SMERF_RRC_CHANGE":
            target_rrc = _first_present_value(payload, ("delta_RRC_optional", "rrc_N_per_kN"))
        else:
            target_rrc = _first_present_value(payload, ("delta_RRC_optional",))
    if not is_blank(target_rrc):
        canonical["target_rrc_N_per_kN"] = target_rrc

    if proposal_type == "TIRE_TARGET_RRC":
        canonical.pop("tire_improvement_pct", None)
    if proposal_type == "TIRE_IMPROVEMENT_PCT":
        canonical.pop("target_rrc_N_per_kN", None)

    return canonical


def legacy_tire_detail_fields(proposal_type: str | None, details: dict | None) -> dict:
    proposal_type = canonical_tire_proposal_type(proposal_type)
    payload = canonical_tire_details(proposal_type, details)
    legacy: dict = {}

    if not is_blank(payload.get("tire_code")):
        legacy["new_tire_code"] = payload["tire_code"]
    if not is_blank(payload.get("front_pressure_psi")):
        legacy["psi_front"] = payload["front_pressure_psi"]
    if not is_blank(payload.get("rear_pressure_psi")):
        legacy["psi_rear"] = payload["rear_pressure_psi"]
    if not is_blank(payload.get("tire_load_mass_basis")):
        legacy["load_basis"] = payload["tire_load_mass_basis"]
    if proposal_type == "TIRE_TARGET_RRC" and not is_blank(payload.get("target_rrc_N_per_kN")):
        legacy["delta_RRC_optional"] = payload["target_rrc_N_per_kN"]
    if proposal_type == "TIRE_IMPROVEMENT_PCT" and not is_blank(payload.get("tire_improvement_pct")):
        legacy["improvement_pct"] = payload["tire_improvement_pct"]
    return legacy


def tire_resolver_inputs_from_details(proposal_type: str | None, details: dict | None) -> dict:
    payload = canonical_tire_details(proposal_type, details)
    return {
        "tire_db_id": payload.get("tire_db_id"),
        "tire_code": payload.get("tire_code"),
        "tire_snapshot": deepcopy(dict(payload.get("tire_snapshot") or {})) if isinstance(payload.get("tire_snapshot"), dict) else None,
        "rrc_N_per_kN": payload.get("rrc_N_per_kN"),
        "target_rrc_N_per_kN": payload.get("target_rrc_N_per_kN"),
        "tire_improvement_pct": payload.get("tire_improvement_pct"),
        "front_pressure_psi": payload.get("front_pressure_psi"),
        "rear_pressure_psi": payload.get("rear_pressure_psi"),
        "tire_load_mass_basis": payload.get("tire_load_mass_basis"),
    }


def _first_present_value(payload: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in payload and not is_blank(payload.get(key)):
            return payload.get(key)
    return None
