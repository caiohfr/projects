import math
from copy import deepcopy

from src.vde_core.test_mass import (
    compute_wltp_test_masses,
    format_inertia_step_interval,
    inertia_class_from_mass,
    inertia_step_for_mass,
    inertia_step_for_class,
    representative_mass_for_inertia_class,
)
from src.vde_core.vde_setup_service import EPA_TEST_MASS_DEFAULT_DELTA_KG

EPA_INERTIA_CLASSES = [
    454.0, 510.0, 567.0, 624.0, 680.0, 737.0, 794.0, 850.0, 907.0, 964.0, 1021.0, 1077.0,
    1134.0, 1191.0, 1247.0, 1304.0, 1361.0, 1417.0, 1474.0, 1531.0, 1588.0, 1644.0, 1701.0,
    1758.0, 1814.0, 1928.0, 2041.0, 2155.0, 2268.0, 2381.0, 2495.0, 2722.0, 2948.0, 3175.0,
    3402.0, 3856.0, 4082.0,
]
def resolve_mass_proposal(source_snapshot, proposal_type, inputs) -> dict:
    source = deepcopy(dict(source_snapshot or {}))
    cleaned_inputs = deepcopy(dict(inputs or {}))
    proposal_type = str(proposal_type or "INHERIT").strip().upper()

    resolved = {
        "mass_kg": _to_float(_first_value(cleaned_inputs, "mass_kg", source=source, aliases=("curb_mass_kg",))),
        "current_curb_mass_kg": _to_float(_first_value(cleaned_inputs, "current_curb_mass_kg", source=source)),
        "target_curb_mass_kg": _to_float(_first_input_value(cleaned_inputs, "target_curb_mass_kg", aliases=("mass_kg", "curb_mass_kg"))),
        "test_mass_kg": _to_float(_first_value(cleaned_inputs, "test_mass_kg", source=source, aliases=("effective_test_mass_kg",))),
        "test_mass_basis": _first_value(cleaned_inputs, "test_mass_basis", source=source, aliases=("vde_mass_basis",)),
        "tire_load_mass_basis": _first_value(cleaned_inputs, "tire_load_mass_basis", source=source),
        "weight_dist_fr_pct": _to_float(_first_value(cleaned_inputs, "weight_dist_fr_pct", source=source, aliases=("fr_weight_pct",))),
        "inertia_class": _to_float(_first_value(cleaned_inputs, "inertia_class", source=source, aliases=("TWC_kg", "twc_kg", "etw_kg", "prep_inertia_class"))),
        "payload_kg": _to_float(_first_value(cleaned_inputs, "payload_kg", source=source)),
        "options_kg": _to_float(_first_value(cleaned_inputs, "options_kg", source=source, aliases=("optional_weight_kg",))),
        "gvwr_kg": _to_float(_first_value(cleaned_inputs, "gvwr_kg", source=source, aliases=("GVWR_kg",))),
        "gcwr_kg": _to_float(_first_value(cleaned_inputs, "gcwr_kg", source=source, aliases=("GCWR_kg", "mass_profile_gcwr_kg"))),
        "trailer_mass_kg": _to_float(_first_value(cleaned_inputs, "trailer_mass_kg", source=source, aliases=("trailer_weight_kg", "mass_profile_trailer_mass_kg"))),
        "vehicle_mass_at_gcwr": _to_float(_first_value(cleaned_inputs, "vehicle_mass_at_gcwr", source=source)),
        "trailer_A": _to_float(_first_value(cleaned_inputs, "trailer_A", source=source)),
        "trailer_B": _to_float(_first_value(cleaned_inputs, "trailer_B", source=source)),
        "trailer_C": _to_float(_first_value(cleaned_inputs, "trailer_C", source=source)),
        "target_mass_kg": _to_float(_first_value(cleaned_inputs, "target_mass_kg", source=source)),
        "shift_steps": _first_value(cleaned_inputs, "shift_steps", source=source),
        "target_side": _first_value(cleaned_inputs, "target_side", source=source) or "Up",
        "curb_position": _normalize_curb_position(_first_value(cleaned_inputs, "curb_position", source=source)),
        "line_type": str(_first_value(cleaned_inputs, "line_type", source=source) or "TML").strip().upper(),
        "preset": str(_first_value(cleaned_inputs, "preset", source=source) or "Curb +100 kg").strip(),
        "custom_delta_kg": _to_float(_first_value(cleaned_inputs, "custom_delta_kg", source=source)),
        "test_mass_low_kg": _to_float(_first_value(cleaned_inputs, "test_mass_low_kg", source=source)),
        "test_mass_high_kg": _to_float(_first_value(cleaned_inputs, "test_mass_high_kg", source=source)),
        "mass_intention": str(_first_value(cleaned_inputs, "mass_intention", source=source) or ""),
        "legislation": str(_first_value(cleaned_inputs, "legislation", source=source) or ""),
        "trailer_roadload_status": str(_first_value(cleaned_inputs, "trailer_roadload_status", source=source) or "Not used"),
        "target_twc_interval": _first_value(cleaned_inputs, "target_twc_interval", source=source),
        "target_twc_lower_bound_exclusive": _to_float(_first_value(cleaned_inputs, "target_twc_lower_bound_exclusive", source=source)),
        "target_twc_upper_bound_inclusive": _to_float(_first_value(cleaned_inputs, "target_twc_upper_bound_inclusive", source=source)),
    }
    issues: list[dict] = []

    if proposal_type == "INHERIT":
        resolved["mass_rule_status"] = "OK"
        resolved["mass_rule_notes"] = "Inherited."
        return _result(resolved, issues)

    if proposal_type == "EPA_STATUS":
        resolved["mass_intention"] = "EPA_STATUS"
        twc = _to_float(_first_nonblank(resolved.get("inertia_class"), _first_value(cleaned_inputs, "inertia_class", source=source)))
        if twc is None:
            issues.append(_issue("missing", "epa_target_missing", "Current EPA ETW / TWC is unavailable."))
        resolved["inertia_class"] = twc
        resolved["mass_kg"] = _to_float(_first_value(cleaned_inputs, "mass_kg", source=source, aliases=("curb_mass_kg",)))
        resolved["test_mass_kg"] = _to_float(_first_value(cleaned_inputs, "test_mass_kg", source=source, aliases=("effective_test_mass_kg",)))
        resolved["test_mass_basis"] = _first_value(cleaned_inputs, "test_mass_basis", source=source, aliases=("vde_mass_basis",))
        resolved["mass_rule_notes"] = "Using inherited EPA state."
        resolved["current_curb_mass_kg"] = _to_float(source.get("mass_kg"))
        return _result(resolved, issues)

    if proposal_type == "EPA_CURB_TO_TWC":
        resolved["mass_intention"] = "EPA_CURB_TO_TWC"
        resolved["current_curb_mass_kg"] = _to_float(source.get("mass_kg"))
        target_curb_state = _epa_curb_to_twc_input_state(cleaned_inputs)
        if target_curb_state["blank"]:
            issues.append(_issue("missing", "curb_mass_missing", "Curb mass is required."))
            resolved["target_curb_mass_kg"] = None
            return _result(resolved, issues)
        if not target_curb_state["valid"]:
            issues.append(_issue("invalid", "curb_mass_invalid", "Curb mass must be a finite number."))
            resolved["target_curb_mass_kg"] = None
            return _result(resolved, issues)

        target_curb_mass = float(target_curb_state["value"])
        if target_curb_mass <= 0:
            issues.append(_issue("invalid", "curb_mass_nonpositive", "Curb mass must be greater than zero."))
            resolved["target_curb_mass_kg"] = target_curb_mass
            return _result(resolved, issues)

        step = inertia_step_for_mass(target_curb_mass)
        resolved["target_curb_mass_kg"] = target_curb_mass
        resolved["mass_kg"] = target_curb_mass
        if not step:
            issues.append(_issue("invalid", "curb_mass_out_of_table", "Curb mass is outside the canonical EPA TWC table."))
            return _result(resolved, issues)

        explicit_test_mass_state = _optional_numeric_input_state(cleaned_inputs, "test_mass_kg")
        explicit_test_mass = None
        if explicit_test_mass_state["present"]:
            if not explicit_test_mass_state["valid"]:
                issues.append(_issue("invalid", "test_mass_invalid", "Explicit test mass must be a finite number."))
                return _result(resolved, issues)
            explicit_test_mass = float(explicit_test_mass_state["value"])
            if explicit_test_mass <= 0:
                issues.append(_issue("invalid", "test_mass_nonpositive", "Explicit test mass must be greater than zero."))
                return _result(resolved, issues)

        resolved["inertia_class"] = step["inertia_class_kg"]
        resolved["test_mass_kg"] = explicit_test_mass if explicit_test_mass is not None else target_curb_mass + EPA_TEST_MASS_DEFAULT_DELTA_KG
        resolved["test_mass_basis"] = "PHYSICAL_TEST_MASS"
        resolved["target_twc_lower_bound_exclusive"] = step.get("lower_bound_exclusive")
        resolved["target_twc_upper_bound_inclusive"] = step.get("upper_bound_inclusive")
        resolved["target_twc_interval"] = format_inertia_step_interval(step)
        resolved["mass_rule_notes"] = (
            f"Curb mass maps to EPA ETW / TWC {resolved['inertia_class']:.1f} "
            f"in interval {resolved['target_twc_interval']}."
        )
        return _result(resolved, issues)

    if proposal_type in {"MASS_TWC_SHIFT", "EPA_PLUS_1_TWC"}:
        resolved["mass_intention"] = "EPA_PLUS_1_TWC"
        reference_mass = _to_float(_first_value(cleaned_inputs, "reference_mass_kg", source=source, aliases=("inertia_class", "TWC_kg", "twc_kg", "prep_inertia_class")))
        if reference_mass is None:
            reference_mass = resolved.get("inertia_class")
        target_mass = resolved.get("target_mass_kg")
        if target_mass is None:
            shift_steps = _to_float(resolved.get("shift_steps"))
            if reference_mass is not None and shift_steps is not None:
                target_mass = _shift_epa_inertia_class(reference_mass, int(shift_steps), str(resolved.get("target_side") or "Up"))
        if target_mass is None:
            issues.append(_issue("missing", "epa_shift_target_missing", "Target ETW / TWC could not be resolved."))
        resolved["target_mass_kg"] = target_mass
        resolved["inertia_class"] = target_mass or reference_mass
        aligned_mass = curb_mass_for_twc_position(target_mass or reference_mass, resolved.get("curb_position"))
        if aligned_mass is not None:
            resolved["mass_kg"] = aligned_mass
            resolved["test_mass_kg"] = aligned_mass + EPA_TEST_MASS_DEFAULT_DELTA_KG
            resolved["test_mass_basis"] = "PHYSICAL_TEST_MASS"
        step = inertia_step_for_class(target_mass or reference_mass)
        resolved["target_twc_lower_bound_exclusive"] = None if not step else step.get("lower_bound_exclusive")
        resolved["target_twc_upper_bound_inclusive"] = None if not step else step.get("upper_bound_inclusive")
        resolved["target_twc_interval"] = format_inertia_step_interval(step)
        return _result(resolved, issues)

    if proposal_type == "PERFORMANCE_CURB_MASS":
        resolved["mass_intention"] = "PERF_CURB_100KG"
        curb_mass = resolved.get("mass_kg")
        preset = str(resolved.get("preset") or "Curb +100 kg").strip()
        custom_delta = resolved.get("custom_delta_kg")
        if curb_mass is None:
            issues.append(_issue("missing", "curb_mass_missing", "New curb mass is required."))
        else:
            if preset == "Curb +300 lb":
                resolved["test_mass_kg"] = curb_mass + 136.1
                resolved["mass_intention"] = "PERF_CURB_300LB"
            elif preset == "Custom delta":
                if custom_delta is None:
                    issues.append(_issue("missing", "custom_delta_missing", "Custom delta is required."))
                    resolved["mass_intention"] = "CUSTOM"
                    resolved["test_mass_kg"] = None
                else:
                    resolved["test_mass_kg"] = curb_mass + custom_delta
                    resolved["mass_intention"] = "CUSTOM"
            else:
                resolved["test_mass_kg"] = curb_mass + 100.0
        resolved["test_mass_basis"] = "CURB_PLUS_DRIVER"
        return _result(resolved, issues)

    if proposal_type == "WLTP_MASS_LINE":
        mass = resolved.get("mass_kg")
        payload = resolved.get("payload_kg")
        options = resolved.get("options_kg")
        result = compute_wltp_test_masses(
            mass_kg=mass,
            payload_kg=payload,
            options_kg=options,
            wltp_category=source.get("wltp_category") or "M1",
        )
        resolved["test_mass_low_kg"] = result.test_mass_low_kg
        resolved["test_mass_high_kg"] = result.test_mass_high_kg
        if result.test_mass_low_kg is None and result.test_mass_high_kg is None:
            issues.extend(_issue_from_warning(text) for text in list(result.warnings or []))
        line_type = str(resolved.get("line_type") or "TML").strip().upper()
        resolved["test_mass_kg"] = result.test_mass_high_kg if line_type == "TMH" else result.test_mass_low_kg
        resolved["test_mass_basis"] = "WLTP_TMH" if line_type == "TMH" else "WLTP_TML"
        resolved["mass_intention"] = resolved["test_mass_basis"]
        for warning in list(result.warnings or []):
            issues.append(_issue("review", "wltp_warning", warning))
        return _result(resolved, issues)

    if proposal_type == "GVWR":
        resolved["mass_intention"] = "GVWR"
        curb_mass = resolved.get("mass_kg")
        payload = _to_float(_first_input_value(cleaned_inputs, "payload_kg"))
        # The v2.2 contract is curb + payload.  GVWR is a compatibility alias,
        # not a competing user input for this proposal.
        legacy_gvwr = _to_float(_first_input_value(cleaned_inputs, "gvwr_kg", aliases=("GVWR_kg",)))
        # Older saved drafts supplied GVWR directly.  Keep that contract
        # readable, while new v2.2 inputs use curb + payload exclusively.
        if payload is None and legacy_gvwr is not None and curb_mass is not None:
            payload = legacy_gvwr - curb_mass
            resolved["payload_kg"] = payload
        gvwr = None if curb_mass is None or payload is None else curb_mass + payload
        resolved["gvwr_kg"] = gvwr
        resolved["test_mass_kg"] = gvwr
        resolved["test_mass_basis"] = "GVWR"
        if curb_mass is None:
            issues.append(_issue("missing", "curb_mass_missing", "Curb mass is required."))
        if payload is None:
            issues.append(_issue("missing", "payload_missing", "Payload is required."))
        elif payload < 0:
            issues.append(_issue("invalid", "payload_negative", "Payload cannot be negative."))
        return _result(resolved, issues)

    if proposal_type == "GCWR":
        resolved["mass_intention"] = "GCWR"
        gcwr = resolved.get("gcwr_kg")
        trailer_mass = resolved.get("trailer_mass_kg")
        curb_mass = resolved.get("mass_kg")
        gvwr = resolved.get("gvwr_kg")
        resolved["test_mass_kg"] = gcwr
        resolved["test_mass_basis"] = "GCWR_TRAILER"
        if gcwr is None:
            issues.append(_issue("missing", "gcwr_missing", "GCWR is required."))
        elif trailer_mass is None:
            issues.append(_issue("missing", "trailer_mass_missing", "Trailer mass is required."))
        else:
            resolved["vehicle_mass_at_gcwr"] = gcwr - trailer_mass
            trailer_complete = all(_to_float(resolved.get(key)) is not None for key in ("trailer_A", "trailer_B", "trailer_C"))
            resolved["trailer_roadload_status"] = "OK" if trailer_complete else "Missing"
            if trailer_mass >= gcwr:
                issues.append(_issue("invalid", "trailer_mass_invalid", "Trailer mass must be lower than GCWR."))
            elif curb_mass is not None and resolved["vehicle_mass_at_gcwr"] is not None and resolved["vehicle_mass_at_gcwr"] < curb_mass:
                issues.append(_issue("invalid", "vehicle_mass_below_curb", "Vehicle mass at GCWR cannot be lower than curb mass."))
            elif gvwr is not None and resolved["vehicle_mass_at_gcwr"] is not None and resolved["vehicle_mass_at_gcwr"] > gvwr:
                issues.append(_issue("review", "vehicle_mass_above_gvwr", "Vehicle mass at GCWR exceeds GVWR."))
            elif not trailer_complete:
                issues.append(_issue("missing", "trailer_roadload_incomplete", "Trailer roadload requires A, B and C."))
        return _result(resolved, issues)

    if proposal_type == "CUSTOM_MASS":
        resolved["mass_intention"] = "CUSTOM"
        if resolved.get("test_mass_kg") is None:
            issues.append(_issue("missing", "custom_mass_missing", "Custom test mass is required."))
        resolved["test_mass_basis"] = str(resolved.get("test_mass_basis") or "CUSTOM")
        return _result(resolved, issues)

    resolved["mass_rule_status"] = "Review"
    resolved["mass_rule_notes"] = f"Unsupported mass proposal type: {proposal_type}"
    issues.append(_issue("review", "unsupported_mass_type", resolved["mass_rule_notes"]))
    return _result(resolved, issues)


def _result(resolved: dict, issues: list[dict]) -> dict:
    normalized = deepcopy(dict(resolved or {}))
    _populate_canonical_mass_state(normalized)
    status = _status_from_issues(issues)
    normalized["mass_rule_status"] = status
    normalized["mass_rule_notes"] = _first_issue_message(issues) if issues else str(normalized.get("mass_rule_notes") or "Resolved.")
    return {
        "resolved_snapshot": normalized,
        "resolved_test_mass_kg": normalized.get("test_mass_kg"),
        "test_mass_basis": normalized.get("test_mass_basis"),
        "payload_kg": normalized.get("payload_kg"),
        "vehicle_mass_at_gcwr": normalized.get("vehicle_mass_at_gcwr"),
        "trailer_roadload": normalized.get("trailer_roadload_status"),
        "status": status,
        "issues": deepcopy(list(issues or [])),
    }


def _populate_canonical_mass_state(resolved: dict) -> None:
    """Publish one mass contract for VDE and Tire consumers.

    ``test_mass_kg`` remains available for legacy records, but new calculation
    paths consume the explicit VDE and tire values below.
    """
    intention = str(resolved.get("mass_intention") or "").strip().upper()
    curb = _to_float(resolved.get("mass_kg"))
    payload = _to_float(resolved.get("payload_kg"))
    inertia = _to_float(resolved.get("inertia_class"))
    test_mass = _to_float(resolved.get("test_mass_kg"))
    gcwr = _to_float(resolved.get("gcwr_kg"))
    trailer = _to_float(resolved.get("trailer_mass_kg"))

    vehicle_loaded = None
    if intention == "GVWR":
        vehicle_loaded = None if curb is None or payload is None else curb + payload
        resolved["gvwr_kg"] = vehicle_loaded
        vde_mass, vde_basis = vehicle_loaded, "LOADED_VEHICLE_MASS"
        tire_mass, tire_basis = vehicle_loaded, "LOADED_VEHICLE_MASS"
    elif intention == "GCWR":
        vehicle_loaded = None if gcwr is None or trailer is None else gcwr - trailer
        resolved["vehicle_mass_at_gcwr"] = vehicle_loaded
        resolved["payload_kg"] = None if vehicle_loaded is None or curb is None else vehicle_loaded - curb
        vde_mass, vde_basis = gcwr, "COMBINED_VEHICLE_TRAILER"
        tire_mass, tire_basis = vehicle_loaded, "LOADED_VEHICLE_MASS"
    elif intention in {"EPA_STATUS", "EPA_CURB_TO_TWC", "EPA_PLUS_1_TWC"}:
        vde_mass, vde_basis = inertia, "EPA_TWC"
        selected_basis = str(resolved.get("tire_load_mass_basis") or "TWC").strip().upper()
        if selected_basis == "TEST_MASS":
            tire_mass, tire_basis = test_mass, "TEST_MASS"
        else:
            tire_mass, tire_basis = inertia, "TWC"
    else:
        vde_mass = test_mass if test_mass is not None else curb
        vde_basis = str(resolved.get("test_mass_basis") or "TEST_MASS").strip().upper()
        if str(resolved.get("legislation") or "").strip().upper() == "EPA":
            tire_mass = inertia if inertia is not None else (inertia_class_from_mass(curb) if curb is not None and curb > 0 else None)
            tire_basis = "TWC"
        else:
            tire_mass, tire_basis = vde_mass, vde_basis

    resolved["curb_mass_kg"] = curb
    resolved["vehicle_loaded_mass_kg"] = vehicle_loaded
    resolved["vde_calculation_mass_kg"] = vde_mass
    resolved["vde_mass_basis"] = vde_basis
    resolved["tire_load_mass_used_kg"] = tire_mass
    resolved["tire_load_mass_basis"] = tire_basis


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


def _issue_from_warning(text: str) -> dict:
    warning = str(text or "").strip()
    if not warning:
        return _issue("review", "wltp_warning", "WLTP mass line warning.")
    if "required" in warning.lower():
        return _issue("missing", "wltp_missing", warning)
    return _issue("review", "wltp_warning", warning)


def _shift_epa_inertia_class(reference_mass: float, steps: int, target_side: str | None = None) -> float | None:
    side = str(target_side or "Up").strip().lower()
    signed_steps = int(steps or 0)
    step_count = abs(signed_steps)
    if step_count == 0:
        return float(reference_mass)
    if float(reference_mass) in EPA_INERTIA_CLASSES:
        index = EPA_INERTIA_CLASSES.index(float(reference_mass))
        if signed_steps < 0:
            offset = -step_count
        else:
            offset = -step_count if side == "down" else step_count
        target_index = max(0, min(len(EPA_INERTIA_CLASSES) - 1, index + offset))
        return float(EPA_INERTIA_CLASSES[target_index])
    direction = -1.0 if signed_steps < 0 or side == "down" else 1.0
    return float(reference_mass) + (125.0 * step_count * direction)


def _first_value(inputs: dict, field_key: str, *, source: dict, aliases: tuple[str, ...] = ()):
    if field_key in inputs and inputs.get(field_key) not in (None, ""):
        return inputs.get(field_key)
    for alias in aliases:
        if alias in inputs and inputs.get(alias) not in (None, ""):
            return inputs.get(alias)
    if field_key in source and source.get(field_key) not in (None, ""):
        return source.get(field_key)
    for alias in aliases:
        if alias in source and source.get(alias) not in (None, ""):
            return source.get(alias)
    return None


def _first_input_value(inputs: dict, field_key: str, aliases: tuple[str, ...] = ()):
    if field_key in inputs and inputs.get(field_key) not in (None, ""):
        return inputs.get(field_key)
    for alias in aliases:
        if alias in inputs and inputs.get(alias) not in (None, ""):
            return inputs.get(alias)
    return None


def _first_nonblank(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _to_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def curb_mass_for_twc_position(twc_kg: float | None, position: str | None = None) -> float | None:
    target_mass = _to_float(twc_kg)
    if target_mass is None:
        return None
    step = inertia_step_for_class(target_mass)
    if not step:
        return representative_mass_for_inertia_class(target_mass) or float(target_mass)

    lower = _to_float(step.get("lower_bound_exclusive"))
    upper = _to_float(step.get("upper_bound_inclusive"))
    bottom = 1.0 if lower is None else lower + 1.0
    top = target_mass if upper is None else upper
    if top < bottom:
        top = target_mass

    normalized_position = _normalize_curb_position(position)
    if normalized_position == "BOTTOM":
        return float(bottom)
    if normalized_position == "MID":
        return float((bottom + top) / 2.0)
    return float(top)


def _normalize_curb_position(value: str | None) -> str:
    text = str(value or "").strip().upper()
    if text == "BOTTOM":
        return "BOTTOM"
    if text == "MID":
        return "MID"
    return "TOP"


def _numeric_input_state(inputs: dict, *keys: str) -> dict:
    source = dict(inputs or {})
    present = False
    raw_value = None
    for key in keys:
        if key in source:
            present = True
            raw_value = source.get(key)
            break
    if not present:
        return {"present": False, "blank": True, "valid": False, "value": None}
    if raw_value in (None, ""):
        return {"present": True, "blank": True, "valid": False, "value": None}
    try:
        value = float(raw_value)
    except Exception:
        return {"present": True, "blank": False, "valid": False, "value": None}
    if not math.isfinite(value):
        return {"present": True, "blank": False, "valid": False, "value": None}
    return {"present": True, "blank": False, "valid": True, "value": value}


def _epa_curb_to_twc_input_state(inputs: dict) -> dict:
    source = dict(inputs or {})
    for key in ("mass_kg", "target_curb_mass_kg", "curb_mass_kg"):
        if key not in source:
            continue
        raw_value = source.get(key)
        if raw_value in (None, ""):
            continue
        return _numeric_input_state({key: raw_value}, key)
    present = any(key in source for key in ("mass_kg", "target_curb_mass_kg", "curb_mass_kg"))
    return {"present": present, "blank": True, "valid": False, "value": None}


def _optional_numeric_input_state(inputs: dict, *keys: str) -> dict:
    state = _numeric_input_state(inputs, *keys)
    if state["blank"]:
        return {"present": False, "blank": True, "valid": False, "value": None}
    return state
