from __future__ import annotations

from copy import deepcopy

from src.vde_core.roadload import calculate_vehicle_tire_abc
from src.vde_core.tire_roadload_service import get_tire_by_code, get_tire_by_id
from src.vde_core.vde_request_contract import is_blank
from src.vde_core.vde_setup_service import resolve_tire_calculation_mass
from src.vde_core.vde_tire_modes import canonical_tire_proposal_type


PRESSURE_MIN_PSI = 20.0
PRESSURE_MAX_PSI = 60.0
PRESSURE_SENSITIVITY = 0.30
PRESSURE_CLAMP_PCT = 0.10
TIRE_METHOD_SAE_FULL = "SAE_FULL"
TIRE_METHOD_RRC_LOAD_SCALING = "RRC_LOAD_SCALING"
TIRE_METHOD_INHERITED_NO_RECALC = "INHERITED_NO_RECALC"
TIRE_METHOD_MISSING_INPUTS = "MISSING_INPUTS"
PSI_TO_KPA = 6.89475729
G_MPS2 = 9.80665


def resolve_tire_proposal(source_snapshot, proposal_type, inputs, *, current_snapshot=None) -> dict:
    source = deepcopy(dict(source_snapshot or {}))
    current = deepcopy(dict(current_snapshot or source_snapshot or {}))
    cleaned_inputs = deepcopy(dict(inputs or {}))
    proposal_type = _normalize_tire_proposal_type(proposal_type)

    source_tire = _lookup_tire_record(
        _first_value(source, "tire_db_id", aliases=("front_tire_id", "rear_tire_id")),
        source.get("tire_code"),
        snapshot=source.get("tire_snapshot"),
    )
    requested_tire = _lookup_tire_record(
        cleaned_inputs.get("tire_db_id"),
        cleaned_inputs.get("tire_code"),
        snapshot=cleaned_inputs.get("tire_snapshot"),
    )
    resolved_tire = requested_tire or source_tire

    source_rrc = _coalesce_rrc(cleaned_inputs.get("source_rrc_N_per_kN"), source, source_tire)
    requested_rrc = _to_float(_first_value(cleaned_inputs, "target_rrc_N_per_kN", aliases=("rrc_N_per_kN",)))
    requested_front = _to_float(_first_value(cleaned_inputs, "front_pressure_psi", source=source))
    requested_rear = _to_float(_first_value(cleaned_inputs, "rear_pressure_psi", source=source))
    reference_front = _to_float(_first_value(source, "front_pressure_psi", aliases=("psi_front",)))
    reference_rear = _to_float(_first_value(source, "rear_pressure_psi", aliases=("psi_rear",)))
    improvement_pct = _to_float(_first_value(cleaned_inputs, "tire_improvement_pct"))
    front_fraction, fraction_defaulted = _front_fraction(current)

    source_mass_payload = resolve_tire_calculation_mass(source)
    current_mass_payload = resolve_tire_calculation_mass(current)
    source_load_mass_kg = _to_float(dict(source_mass_payload or {}).get("mass_kg"))
    current_load_mass_kg = _to_float(dict(current_mass_payload or {}).get("mass_kg"))

    resolved = deepcopy(source)
    resolved.update(
        {
            "tire_db_id": _first_value(cleaned_inputs, "tire_db_id", source=source),
            "tire_code": _first_value(cleaned_inputs, "tire_code", source=source, aliases=("new_tire_code",)),
            "tire_snapshot": deepcopy(dict(resolved_tire or {})) if resolved_tire else deepcopy(dict(source.get("tire_snapshot") or {})) or None,
            "front_pressure_psi": requested_front,
            "rear_pressure_psi": requested_rear,
            "target_rrc_N_per_kN": requested_rrc,
            "rrc_N_per_kN": source_rrc,
            "tire_pressure_sensitivity": PRESSURE_SENSITIVITY,
            "tire_front_weight_fraction": front_fraction,
            "tire_reference_front_pressure_psi": reference_front,
            "tire_reference_rear_pressure_psi": reference_rear,
            "tire_requested_front_pressure_psi": requested_front,
            "tire_requested_rear_pressure_psi": requested_rear,
            "tire_source_rrc_N_per_kN": source_rrc,
            "tire_target_rrc_N_per_kN": requested_rrc,
            "tire_adjusted_rrc_N_per_kN": None,
            "tire_delta_rrc_N_per_kN": None,
            "tire_adjustment_method": "Inherited",
            "tire_review_status": "OK",
            "tire_rule_status": "OK",
            "tire_rule_notes": "Inherited.",
            "tire_load_mass_basis": dict(current_mass_payload or {}).get("basis"),
            "tire_load_mass_used_kg": current_load_mass_kg,
            "source_tire_load_mass_used_kg": source_load_mass_kg,
            "tire_abc_method": TIRE_METHOD_INHERITED_NO_RECALC,
        }
    )

    issues: list[dict] = []
    _append_mass_issue(issues, source_mass_payload, "source_tire_mass_invalid")
    _append_mass_issue(issues, current_mass_payload, "tire_mass_invalid")
    if fraction_defaulted:
        issues.append(_issue("warning", "front_fraction_defaulted", "Front weight fraction defaulted to 50%."))

    if proposal_type == "TIRE_METADATA_ONLY":
        resolved.update(
            {
                "rrc_N_per_kN": 0.0,
                "tire_target_rrc_N_per_kN": 0.0,
                "tire_adjusted_rrc_N_per_kN": 0.0,
                "tire_delta_rrc_N_per_kN": None if source_rrc is None else 0.0 - source_rrc,
                "tire_adjustment_method": "Not used",
                "tire_abc_method": TIRE_METHOD_INHERITED_NO_RECALC,
                "tire_source_abc": _snapshot_tire_abc_triplet(source),
                "tire_resolved_abc": {"A": 0.0, "B": 0.0, "C": 0.0},
                "tire_delta_abc": _delta_triplet({"A": 0.0, "B": 0.0, "C": 0.0}, _snapshot_tire_abc_triplet(source)),
                "tire_A_final": 0.0,
                "tire_B_final": 0.0,
                "tire_C_final": 0.0,
            }
        )
        return _result(resolved, issues)

    resolved_rrc = source_rrc
    if proposal_type == "TIRE_DB_LOOKUP":
        lookup_rrc = _coalesce_rrc(cleaned_inputs.get("rrc_N_per_kN"), cleaned_inputs, requested_tire)
        if lookup_rrc is None:
            issues.append(_issue("missing", "tire_rrc_missing", "Lookup row RRC is required."))
        else:
            resolved_rrc = lookup_rrc
            resolved["rrc_N_per_kN"] = lookup_rrc
            resolved["tire_target_rrc_N_per_kN"] = lookup_rrc
            resolved["tire_adjusted_rrc_N_per_kN"] = lookup_rrc
            resolved["tire_adjustment_method"] = "DB lookup RRC"
            if _is_iso_tire(resolved_tire):
                reference_pressure_psi = tire_reference_pressure_psi(resolved_tire)
                resolved["tire_reference_front_pressure_psi"] = reference_pressure_psi
                resolved["tire_reference_rear_pressure_psi"] = reference_pressure_psi
                if reference_pressure_psi is None:
                    issues.append(
                        _issue(
                            "review",
                            "lookup_reference_pressure_missing",
                            "Selected ISO tire has no reference pressure; lookup RRC was used without a pressure estimate.",
                        )
                    )
                    resolved["tire_adjustment_method"] = "DB lookup RRC (ISO reference pressure unavailable)"
                else:
                    estimate = _estimate_pressure_only_rrc(
                        source_rrc=lookup_rrc,
                        reference_front_psi=reference_pressure_psi,
                        reference_rear_psi=reference_pressure_psi,
                        requested_front_psi=requested_front,
                        requested_rear_psi=requested_rear,
                        front_fraction=front_fraction,
                    )
                    estimated_rrc = _to_float(dict(estimate.get("resolved_fields") or {}).get("rrc_N_per_kN"))
                    issues.extend(list(estimate.get("issues") or []))
                    if estimated_rrc is not None:
                        resolved_rrc = estimated_rrc
                        resolved.update(estimate.get("resolved_fields") or {})
                        resolved["tire_adjustment_method"] = "ISO pressure estimate"
    elif proposal_type == "TIRE_IMPROVEMENT_PCT":
        if source_rrc is None:
            issues.append(_issue("missing", "source_rrc_missing", "Source RRC is required for Tire improvement %."))
        elif improvement_pct is None:
            issues.append(_issue("missing", "improvement_missing", "Tire improvement % is required."))
        else:
            resolved_rrc = source_rrc * (1.0 - (improvement_pct / 100.0))
            resolved["rrc_N_per_kN"] = resolved_rrc
            resolved["tire_target_rrc_N_per_kN"] = resolved_rrc
            resolved["tire_adjusted_rrc_N_per_kN"] = resolved_rrc
            resolved["tire_adjustment_method"] = "Tire improvement %"
            if improvement_pct < 0:
                issues.append(_issue("review", "negative_improvement", "Negative tire improvement requires review."))
    elif proposal_type == "TIRE_TARGET_RRC":
        if requested_rrc is not None:
            resolved_rrc = requested_rrc
            resolved["rrc_N_per_kN"] = requested_rrc
            resolved["tire_adjusted_rrc_N_per_kN"] = requested_rrc
            resolved["tire_adjustment_method"] = "Direct target RRC"
        elif not _pressures_are_valid(requested_front, requested_rear):
            issues.append(_issue("missing", "requested_pressure_invalid", "Requested pressures must be between 20 and 60 psi."))
        else:
            estimate = _estimate_pressure_only_rrc(
                source_rrc=source_rrc,
                reference_front_psi=reference_front,
                reference_rear_psi=reference_rear,
                requested_front_psi=requested_front,
                requested_rear_psi=requested_rear,
                front_fraction=front_fraction,
            )
            resolved_rrc = _to_float(dict(estimate.get("resolved_fields") or {}).get("rrc_N_per_kN"))
            resolved.update(estimate.get("resolved_fields") or {})
            issues.extend(list(estimate.get("issues") or []))
    elif proposal_type not in {"INHERIT"}:
        issues.append(_issue("review", "unsupported_tire_type", f"Unsupported tire proposal type: {proposal_type}"))

    if source_rrc is not None and resolved_rrc is not None:
        resolved["tire_delta_rrc_N_per_kN"] = resolved_rrc - source_rrc

    abc_outcome = _resolve_tire_abc(
        proposal_type=proposal_type,
        source_snapshot=source,
        current_snapshot=current,
        source_tire=source_tire,
        resolved_tire=resolved_tire,
        source_rrc=source_rrc,
        resolved_rrc=resolved_rrc,
        source_mass_payload=source_mass_payload,
        current_mass_payload=current_mass_payload,
    )
    resolved.update(dict(abc_outcome.get("resolved_fields") or {}))
    issues.extend(list(abc_outcome.get("issues") or []))
    if resolved_rrc is None and proposal_type == "INHERIT" and _abc_complete(resolved.get("tire_source_abc")):
        resolved["tire_adjustment_method"] = "Inherited"
        resolved["tire_abc_method"] = TIRE_METHOD_INHERITED_NO_RECALC

    return _result(resolved, issues)


def _normalize_tire_proposal_type(proposal_type) -> str:
    return canonical_tire_proposal_type(proposal_type)


def _estimate_pressure_only_rrc(
    *,
    source_rrc: float | None,
    reference_front_psi: float | None,
    reference_rear_psi: float | None,
    requested_front_psi: float | None,
    requested_rear_psi: float | None,
    front_fraction: float,
) -> dict:
    issues: list[dict] = []
    resolved_fields = {
        "tire_adjustment_method": "Pressure estimate",
        "tire_review_status": "Review",
        "tire_adjusted_rrc_N_per_kN": None,
    }
    if source_rrc is None:
        issues.append(_issue("missing", "source_rrc_missing", "Baseline/source RRC is required for pressure-only estimate."))
        return {"resolved_fields": resolved_fields, "issues": issues}
    if reference_front_psi is None or reference_rear_psi is None:
        issues.append(_issue("missing", "reference_pressure_missing", "Reference front/rear pressure is required for pressure-only estimate."))
        return {"resolved_fields": resolved_fields, "issues": issues}
    if not _pressures_are_valid(reference_front_psi, reference_rear_psi):
        issues.append(_issue("review", "reference_pressure_invalid", "Reference pressures are outside the supported 20 to 60 psi range."))
        return {"resolved_fields": resolved_fields, "issues": issues}
    if requested_front_psi is None or requested_rear_psi is None:
        issues.append(_issue("missing", "requested_pressure_missing", "Requested front/rear pressure is required when target RRC is blank."))
        return {"resolved_fields": resolved_fields, "issues": issues}

    front_factor = 1.0 + PRESSURE_SENSITIVITY * (reference_front_psi / requested_front_psi - 1.0)
    rear_factor = 1.0 + PRESSURE_SENSITIVITY * (reference_rear_psi / requested_rear_psi - 1.0)
    vehicle_factor = front_fraction * front_factor + (1.0 - front_fraction) * rear_factor
    adjusted_rrc = source_rrc * vehicle_factor
    lower = source_rrc * (1.0 - PRESSURE_CLAMP_PCT)
    upper = source_rrc * (1.0 + PRESSURE_CLAMP_PCT)
    adjusted_rrc = max(lower, min(upper, adjusted_rrc))
    resolved_fields.update(
        {
            "rrc_N_per_kN": adjusted_rrc,
            "tire_adjusted_rrc_N_per_kN": adjusted_rrc,
            "tire_target_rrc_N_per_kN": adjusted_rrc,
        }
    )
    return {"resolved_fields": resolved_fields, "issues": issues}


def _resolve_tire_abc(
    *,
    proposal_type: str,
    source_snapshot: dict,
    current_snapshot: dict,
    source_tire: dict | None,
    resolved_tire: dict | None,
    source_rrc: float | None,
    resolved_rrc: float | None,
    source_mass_payload: dict,
    current_mass_payload: dict,
) -> dict:
    issues: list[dict] = []
    source_abc = _best_source_abc(source_snapshot, source_tire, source_rrc, source_mass_payload)
    resolved_fields = {
        "tire_source_abc": deepcopy(source_abc),
        "tire_resolved_abc": None,
        "tire_delta_abc": None,
        "tire_abc_method": TIRE_METHOD_MISSING_INPUTS,
    }

    current_mass_kg = _to_float(dict(current_mass_payload or {}).get("mass_kg"))
    source_mass_kg = _to_float(dict(source_mass_payload or {}).get("mass_kg"))
    resolved_fields["tire_load_mass_basis"] = dict(current_mass_payload or {}).get("basis")
    resolved_fields["tire_load_mass_used_kg"] = current_mass_kg
    resolved_fields["source_tire_load_mass_used_kg"] = source_mass_kg

    if current_mass_kg is None:
        issues.append(_issue("missing", "tire_mass_missing", "Resolved tire calculation mass is required."))
        return {"resolved_fields": resolved_fields, "issues": issues}

    inherited_rrc = proposal_type == "INHERIT"
    if inherited_rrc and source_rrc is None and not _abc_complete(source_abc) and resolved_tire is None:
        issues.append(_issue("review", "tire_source_incomplete", "Inherited tire data is incomplete for recalculation; existing tire contribution was preserved."))
        resolved_fields.update(
            {
                "tire_resolved_abc": deepcopy(source_abc) if _abc_complete(source_abc) else None,
                "tire_delta_abc": {"A": 0.0, "B": 0.0, "C": 0.0} if _abc_complete(source_abc) else None,
                "tire_A_final": source_abc.get("A"),
                "tire_B_final": source_abc.get("B"),
                "tire_C_final": source_abc.get("C"),
                "tire_abc_method": TIRE_METHOD_MISSING_INPUTS,
            }
        )
        return {"resolved_fields": resolved_fields, "issues": issues}
    if not _abc_complete(source_abc):
        issues.append(
            _issue(
                "review" if inherited_rrc else "missing",
                "tire_source_reference_missing",
                "Baseline/source tire reference is required to resolve tire delta.",
            )
        )
        return {"resolved_fields": resolved_fields, "issues": issues}
    source_pressures = _pressure_pair(source_snapshot)
    current_pressures = _pressure_pair(current_snapshot)
    same_pressure = source_pressures == current_pressures
    same_load = _same_float(source_mass_kg, current_mass_kg)
    same_tire = _same_tire_identity(source_snapshot, current_snapshot)

    if inherited_rrc and same_pressure and same_load and same_tire and _abc_complete(source_abc):
        resolved_fields.update(
            {
                "tire_resolved_abc": deepcopy(source_abc),
                "tire_delta_abc": {"A": 0.0, "B": 0.0, "C": 0.0},
                "tire_A_final": source_abc["A"],
                "tire_B_final": source_abc["B"],
                "tire_C_final": source_abc["C"],
                "tire_abc_method": TIRE_METHOD_INHERITED_NO_RECALC,
            }
        )
        return {"resolved_fields": resolved_fields, "issues": issues}

    model_result = None
    model_method = None
    if proposal_type in {"INHERIT", "TIRE_DB_LOOKUP"} and not (
        proposal_type == "TIRE_DB_LOOKUP" and _is_iso_tire(resolved_tire)
    ) or (
        proposal_type == "TIRE_TARGET_RRC" and resolved_rrc is None
    ):
        model_result = _calculate_with_tire_model(resolved_tire, current_snapshot, current_mass_kg)
        if model_result is not None:
            model_method = TIRE_METHOD_SAE_FULL if _has_full_sae_data(resolved_tire) else TIRE_METHOD_RRC_LOAD_SCALING
            resolved_rrc = _to_float(dict(model_result).get("applied_rr_n_per_kn")) or resolved_rrc

    if model_result is not None:
        target_abc = dict(model_result.get("total_final_abc") or {})
        delta_abc = _delta_triplet(target_abc, source_abc)
        resolved_fields.update(
            {
                "rrc_N_per_kN": resolved_rrc,
                "tire_adjusted_rrc_N_per_kN": resolved_rrc,
                "tire_resolved_abc": deepcopy(target_abc),
                "tire_delta_abc": deepcopy(delta_abc),
                "tire_A_final": target_abc.get("A"),
                "tire_B_final": target_abc.get("B"),
                "tire_C_final": target_abc.get("C"),
                "tire_abc_method": model_method,
            }
        )
        return {"resolved_fields": resolved_fields, "issues": issues}

    if resolved_rrc is None:
        issues.append(_issue("missing", "tire_rrc_missing", "Resolved tire RRC is required."))
        return {"resolved_fields": resolved_fields, "issues": issues}

    source_load_factor = _rrc_load_factor(source_rrc, source_mass_kg)
    target_load_factor = _rrc_load_factor(resolved_rrc, current_mass_kg)
    if target_load_factor is None:
        issues.append(_issue("missing", "tire_mass_missing", "Resolved tire calculation mass is required."))
        return {"resolved_fields": resolved_fields, "issues": issues}

    if _abc_complete(source_abc) and source_load_factor not in (None, 0.0):
        factor = target_load_factor / source_load_factor
        target_abc = {key: source_abc[key] * factor for key in ("A", "B", "C")}
        delta_abc = {key: target_abc[key] - source_abc[key] for key in ("A", "B", "C")}
        method = TIRE_METHOD_RRC_LOAD_SCALING
    else:
        target_abc = _abc_from_rrc(resolved_rrc, current_mass_kg, _front_fraction_pct(current_snapshot))
        delta_abc = _delta_triplet(target_abc, source_abc)
        method = TIRE_METHOD_RRC_LOAD_SCALING

    resolved_fields.update(
        {
            "rrc_N_per_kN": resolved_rrc,
            "tire_adjusted_rrc_N_per_kN": resolved_rrc,
            "tire_resolved_abc": deepcopy(target_abc),
            "tire_delta_abc": deepcopy(delta_abc),
            "tire_A_final": target_abc.get("A"),
            "tire_B_final": target_abc.get("B"),
            "tire_C_final": target_abc.get("C"),
            "tire_abc_method": method,
        }
    )
    return {"resolved_fields": resolved_fields, "issues": issues}


def _calculate_with_tire_model(tire: dict | None, snapshot: dict, mass_kg: float | None) -> dict | None:
    tire_record = dict(tire or {})
    if mass_kg is None or not tire_record or not _tire_model_record_available(tire_record):
        return None
    try:
        return calculate_vehicle_tire_abc(
            front_tire=tire_record,
            rear_tire=tire_record,
            inputs={
                "mass_kg": float(mass_kg),
                "front_weight_distribution_pct": _front_fraction_pct(snapshot),
                "front_pressure_kpa": _pressure_kpa(snapshot.get("front_pressure_psi")),
                "rear_pressure_kpa": _pressure_kpa(snapshot.get("rear_pressure_psi")),
                "tire_improvement_pct": 0.0,
            },
        )
    except Exception:
        return None


def _best_source_abc(source_snapshot: dict, source_tire: dict | None, source_rrc: float | None, source_mass_payload: dict) -> dict:
    explicit = _snapshot_tire_abc_triplet(source_snapshot)
    if _abc_complete(explicit):
        return explicit
    source_mass_kg = _to_float(dict(source_mass_payload or {}).get("mass_kg"))
    model_result = _calculate_with_tire_model(source_tire, source_snapshot, source_mass_kg)
    if model_result is not None:
        return dict(model_result.get("total_final_abc") or {})
    if source_rrc is not None and source_mass_kg is not None:
        return _abc_from_rrc(source_rrc, source_mass_kg, _front_fraction_pct(source_snapshot))
    return explicit


def _coalesce_rrc(primary, snapshot: dict, tire: dict | None = None) -> float | None:
    direct = _to_float(primary)
    if direct is not None:
        return direct
    from_snapshot = _to_float(_first_value(snapshot, "rrc_N_per_kN", aliases=("target_rrc_N_per_kN",)))
    if from_snapshot is not None:
        return from_snapshot
    return _to_float(dict(tire or {}).get("rr_n_per_kn"))


def _lookup_tire_record(tire_db_id, tire_code, *, snapshot=None) -> dict | None:
    record = deepcopy(dict(snapshot or {})) or None
    if record:
        return deepcopy(record)
    if not is_blank(tire_db_id):
        try:
            record = get_tire_by_id(int(tire_db_id))
        except Exception:
            record = deepcopy(dict(snapshot or {})) or None
    if not record and not is_blank(tire_code):
        try:
            record = get_tire_by_code(str(tire_code))
        except Exception:
            record = deepcopy(dict(snapshot or {})) or None
    return deepcopy(dict(record or {})) or None


def _tire_model_record_available(tire: dict) -> bool:
    family = str(dict(tire or {}).get("standard_family") or "").strip().upper()
    if family in {"ISO", "CUSTOM"}:
        return _to_float(dict(tire or {}).get("rr_n_per_kn")) is not None
    if family == "SAE":
        return _has_full_sae_data(tire) and _to_float(dict(tire or {}).get("rr_n_per_kn")) is not None
    return False


def _has_full_sae_data(tire: dict | None) -> bool:
    data = dict(tire or {})
    required = (
        data.get("rr_n_per_kn"),
        data.get("sae_alpha"),
        data.get("sae_beta"),
        data.get("sae_a"),
        data.get("sae_b"),
        data.get("sae_c"),
        data.get("sae_reference_load_n"),
        data.get("sae_reference_pressure_kpa"),
    )
    return all(_to_float(value) is not None for value in required)


def _is_iso_tire(tire: dict | None) -> bool:
    return str(dict(tire or {}).get("standard_family") or "").strip().upper() == "ISO"


def tire_reference_pressure_psi(tire: dict | None) -> float | None:
    """Return a Tire DB record's canonical reference pressure in psi.

    This is deliberately a unit-normalization adapter, not a pressure/RRC
    model. Quick Scenario uses it to construct explicit front/rear pressure
    requests before delegating all physical resolution back to
    :func:`resolve_tire_proposal`.
    """

    data = dict(tire or {})
    test_pressure = _to_float(data.get("test_pressure_value"))
    if test_pressure is not None:
        pressure_unit = str(data.get("pressure_unit") or "psi").strip().lower()
        return test_pressure / PSI_TO_KPA if pressure_unit == "kpa" else test_pressure
    for field_name in ("iso_test_pressure_kpa", "sae_reference_pressure_kpa"):
        pressure_kpa = _to_float(data.get(field_name))
        if pressure_kpa is not None:
            return pressure_kpa / PSI_TO_KPA
    return None


def _append_mass_issue(issues: list[dict], payload: dict | None, code: str) -> None:
    message = str(dict(payload or {}).get("issue") or "").strip()
    if message:
        issues.append(_issue("review", code, message))


def _rrc_load_factor(rrc_n_per_kn: float | None, mass_kg: float | None) -> float | None:
    rrc = _to_float(rrc_n_per_kn)
    mass = _to_float(mass_kg)
    if rrc is None or mass is None:
        return None
    return rrc * mass


def _delta_triplet(target: dict | None, source: dict | None) -> dict[str, float | None]:
    target_triplet = _abc_triplet(target or {})
    source_triplet = _abc_triplet(source or {})
    if not _abc_complete(target_triplet):
        return {"A": None, "B": None, "C": None}
    return {
        "A": target_triplet["A"] - (source_triplet["A"] or 0.0),
        "B": target_triplet["B"] - (source_triplet["B"] or 0.0),
        "C": target_triplet["C"] - (source_triplet["C"] or 0.0),
    }


def _pressure_pair(snapshot: dict) -> tuple[float | None, float | None]:
    data = dict(snapshot or {})
    return (_to_float(data.get("front_pressure_psi")), _to_float(data.get("rear_pressure_psi")))


def _same_float(lhs, rhs, *, tol: float = 1e-9) -> bool:
    left = _to_float(lhs)
    right = _to_float(rhs)
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= tol


def _same_tire_identity(source_snapshot: dict, current_snapshot: dict) -> bool:
    source = dict(source_snapshot or {})
    current = dict(current_snapshot or {})
    source_id = source.get("tire_db_id") or source.get("tire_code")
    current_id = current.get("tire_db_id") or current.get("tire_code")
    return str(source_id or "") == str(current_id or "")


def _front_fraction(snapshot: dict) -> tuple[float, bool]:
    value = _to_float(_first_value(snapshot, "weight_dist_fr_pct", aliases=("fr_weight_pct",)))
    if value is None:
        return 0.5, True
    value = max(0.0, min(100.0, value))
    return value / 100.0, False


def _front_fraction_pct(snapshot: dict) -> float:
    fraction, _ = _front_fraction(snapshot)
    return fraction * 100.0


def _pressure_kpa(pressure_psi) -> float | None:
    pressure = _to_float(pressure_psi)
    if pressure is None:
        return None
    return pressure * PSI_TO_KPA


def _pressures_are_valid(front_pressure_psi, rear_pressure_psi) -> bool:
    for value in (front_pressure_psi, rear_pressure_psi):
        if value is None:
            return False
        if value < PRESSURE_MIN_PSI or value > PRESSURE_MAX_PSI:
            return False
    return True


def _abc_from_rrc(rrc_n_per_kn: float, mass_kg: float, front_weight_distribution_pct: float) -> dict:
    tire_stub = {"standard_family": "CUSTOM", "rr_n_per_kn": float(rrc_n_per_kn)}
    result = calculate_vehicle_tire_abc(
        front_tire=tire_stub,
        rear_tire=tire_stub,
        inputs={
            "mass_kg": float(mass_kg),
            "front_weight_distribution_pct": float(front_weight_distribution_pct),
            "front_pressure_kpa": None,
            "rear_pressure_kpa": None,
            "tire_improvement_pct": 0.0,
        },
    )
    return dict(result.get("total_final_abc") or {})


def _snapshot_tire_abc_triplet(snapshot: dict) -> dict[str, float | None]:
    data = dict(snapshot or {})
    return {
        "A": _to_float(data.get("tire_A_final")),
        "B": _to_float(data.get("tire_B_final")),
        "C": _to_float(data.get("tire_C_final")),
    }


def _abc_triplet(snapshot: dict) -> dict[str, float | None]:
    data = dict(snapshot or {})
    return {
        "A": _to_float(_first_value(data, "tire_A_final", aliases=("A",))),
        "B": _to_float(_first_value(data, "tire_B_final", aliases=("B",))),
        "C": _to_float(_first_value(data, "tire_C_final", aliases=("C",))),
    }


def _abc_complete(payload: dict | None) -> bool:
    data = dict(payload or {})
    return all(_to_float(data.get(key)) is not None for key in ("A", "B", "C"))


def _result(resolved: dict, issues: list[dict]) -> dict:
    normalized = deepcopy(dict(resolved or {}))
    status = _status_from_issues(issues)
    if status == "OK" and str(normalized.get("tire_review_status") or "").strip().lower() == "review":
        status = "Review"
    normalized["tire_rule_status"] = status
    normalized["tire_rule_notes"] = _first_issue_message(issues) if issues else "Resolved."
    return {
        "resolved_snapshot": normalized,
        "resolved_rrc_N_per_kN": normalized.get("rrc_N_per_kN"),
        "status": status,
        "issues": deepcopy(list(issues or [])),
    }


def _status_from_issues(issues: list[dict]) -> str:
    severities = [str(dict(item or {}).get("severity") or "").strip().lower() for item in list(issues or [])]
    if "invalid" in severities:
        return "Invalid"
    if "missing" in severities:
        return "Missing"
    if "review" in severities:
        return "Review"
    return "OK"


def _first_issue_message(issues: list[dict]) -> str:
    for issue in list(issues or []):
        message = str(dict(issue or {}).get("message") or "").strip()
        if message:
            return message
    return ""


def _issue(severity: str, code: str, message: str) -> dict:
    return {"severity": str(severity), "code": str(code), "message": str(message)}


def _first_value(payload: dict, field_key: str, *, source: dict | None = None, aliases: tuple[str, ...] = ()):
    data = dict(payload or {})
    if field_key in data and not is_blank(data.get(field_key)):
        return data.get(field_key)
    for alias in aliases:
        if alias in data and not is_blank(data.get(alias)):
            return data.get(alias)
    source_data = dict(source or {})
    if field_key in source_data and not is_blank(source_data.get(field_key)):
        return source_data.get(field_key)
    for alias in aliases:
        if alias in source_data and not is_blank(source_data.get(alias)):
            return source_data.get(alias)
    return None


def _to_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None
