from __future__ import annotations

import re
from copy import deepcopy

from src.vde_app.units import (
    PRESSURE_UNIT_KPA,
    UNIT_SYSTEM_METRIC,
    format_quantity,
    normalize_unit_system,
    normalize_pressure_unit,
    pressure_display_format,
    pressure_display_precision,
    pressure_display_step,
    pressure_to_canonical,
    pressure_to_display,
    pressure_unit_label,
    to_canonical,
    to_display,
    unit_label,
)
from src.vde_core.vde_request_contract import is_blank


_CONTROLLED_PREFIXES = (
    "baseline_",
    "current_",
    "effective_",
    "requested_",
    "resolved_",
    "source_",
    "target_",
    "new_",
    "delta_",
    "final_",
)

_FIELD_QUANTITY_BY_BASE = {
    "mass_kg": "mass",
    "curb_mass_kg": "mass",
    "target_curb_mass_kg": "mass",
    "resolved_curb_mass_kg": "mass",
    "resolved_mass_kg": "mass",
    "test_mass_kg": "mass",
    "resolved_test_mass_kg": "mass",
    "test_mass_low_kg": "mass",
    "test_mass_high_kg": "mass",
    "inertia_class": "mass",
    "payload_kg": "mass",
    "custom_delta_kg": "mass",
    "options_kg": "mass",
    "gvwr_kg": "mass",
    "gcwr_kg": "mass",
    "trailer_mass_kg": "mass",
    "vehicle_mass_at_gcwr": "mass",
    "tire_load_mass_used_kg": "mass",
    "target_class_min_kg": "mass",
    "target_class_max_kg": "mass",
    "target_twc_lower_bound_exclusive": "mass",
    "target_twc_upper_bound_inclusive": "mass",
    "A": "force",
    "coast_A_N": "force",
    "trans_A_coef_N": "force",
    "brake_A_coef_N": "force",
    "axle_hub_A": "force",
    "parasitic_A_coef_N": "force",
    "trailer_A": "force",
    "B": "force_per_speed",
    "coast_B_N_per_kph": "force_per_speed",
    "trans_B_coef_Npkph": "force_per_speed",
    "trans_B_Npkph": "force_per_speed",
    "brake_B_Npkph": "force_per_speed",
    "axle_hub_B": "force_per_speed",
    "parasitic_B_Npkph": "force_per_speed",
    "trailer_B": "force_per_speed",
    "C": "force_per_speed_squared",
    "coast_C_N_per_kph2": "force_per_speed_squared",
    "trans_C_coef_Npkph2": "force_per_speed_squared",
    "brake_C_coef_Npkph2": "force_per_speed_squared",
    "axle_hub_C": "force_per_speed_squared",
    "parasitic_C_coef_Npkph2": "force_per_speed_squared",
    "trailer_C": "force_per_speed_squared",
    "front_pressure_psi": "pressure",
    "rear_pressure_psi": "pressure",
    "hot_front_pressure_psi": "pressure",
    "hot_rear_pressure_psi": "pressure",
    "tire_reference_front_pressure_psi": "pressure",
    "tire_reference_rear_pressure_psi": "pressure",
    "tire_requested_front_pressure_psi": "pressure",
    "tire_requested_rear_pressure_psi": "pressure",
    "rrc_N_per_kN": "rrc",
    "target_rrc_N_per_kN": "rrc",
    "tire_source_rrc_N_per_kN": "rrc",
    "tire_target_rrc_N_per_kN": "rrc",
    "tire_adjusted_rrc_N_per_kN": "rrc",
    "tire_delta_rrc_N_per_kN": "rrc",
    "cda_m2": "cda",
    "CdA": "cda",
    "vde_total_mj_per_km": "energy_per_distance",
    "vde_net_mj_per_km": "energy_per_distance",
    "vde_urb_mj_per_km": "energy_per_distance",
    "vde_hw_mj_per_km": "energy_per_distance",
    "target_twc_interval": "mass_interval",
    "baseline_mass_target_twc_interval": "mass_interval",
}

_RRC_LABEL = "N/kN"
_INTERVAL_RE = re.compile(r"^\((?P<lower>[^,]+),\s*(?P<upper>[^\]]+)\]\s*(?P<unit>[A-Za-z/^0-9]+)?$")
_UNIT_SENSITIVE_QUANTITIES = {
    "mass",
    "force",
    "force_per_speed",
    "force_per_speed_squared",
    "cda",
    "pressure",
}


def quantity_kind_for_field(field_key) -> str | None:
    normalized_key = _normalize_field_key(field_key)
    if not normalized_key:
        return None
    return _FIELD_QUANTITY_BY_BASE.get(normalized_key)


def _resolved_pressure_unit(pressure_unit, unit_system) -> str:
    default = "psi" if normalize_unit_system(unit_system) != UNIT_SYSTEM_METRIC else PRESSURE_UNIT_KPA
    return normalize_pressure_unit(pressure_unit, default=default)


def display_unit_for_field(field_key, unit_system, canonical_unit=None, *, pressure_unit: str | None = None) -> str | None:
    quantity = quantity_kind_for_field(field_key)
    system = normalize_unit_system(unit_system)
    if quantity == "rrc":
        return _RRC_LABEL
    if quantity == "mass_interval":
        return "kg" if system == UNIT_SYSTEM_METRIC else "lb"
    if quantity == "pressure":
        return pressure_unit_label(_resolved_pressure_unit(pressure_unit, system))
    if quantity:
        return unit_label(quantity, system)
    return canonical_unit


def display_value_for_field(field_key, canonical_value, unit_system, *, pressure_unit: str | None = None):
    quantity = quantity_kind_for_field(field_key)
    system = normalize_unit_system(unit_system)
    if quantity is None:
        return canonical_value
    if quantity == "mass_interval":
        return _display_interval_value(canonical_value, system)
    if is_blank(canonical_value):
        return None
    numeric = _to_float(canonical_value)
    if numeric is None:
        return canonical_value
    if quantity == "rrc":
        return numeric
    if quantity == "pressure":
        return pressure_to_display(numeric, _resolved_pressure_unit(pressure_unit, system))
    return to_display(numeric, quantity, system)


def to_display_field_value(field_key, canonical_value, unit_system, *, pressure_unit: str | None = None):
    return display_value_for_field(field_key, canonical_value, unit_system, pressure_unit=pressure_unit)


def to_canonical_field_value(field_key, display_value, unit_system, *, pressure_unit: str | None = None):
    quantity = quantity_kind_for_field(field_key)
    system = normalize_unit_system(unit_system)
    if quantity is None or quantity == "mass_interval":
        return display_value
    if is_blank(display_value):
        return None
    numeric = _to_float(display_value)
    if numeric is None:
        return display_value
    if quantity == "rrc":
        return numeric
    if quantity == "pressure":
        return pressure_to_canonical(numeric, _resolved_pressure_unit(pressure_unit, system))
    return to_canonical(numeric, quantity, system)


def display_step_for_field(field_key, canonical_step, unit_system, *, pressure_unit: str | None = None):
    normalized_key = _normalize_field_key(field_key)
    quantity = quantity_kind_for_field(field_key)
    system = normalize_unit_system(unit_system)
    if quantity is None or quantity == "mass_interval":
        return canonical_step
    if quantity == "mass":
        if normalized_key == "custom_delta_kg" and system == UNIT_SYSTEM_METRIC:
            return 0.1
        return 1.0
    if quantity == "pressure":
        return pressure_display_step(_resolved_pressure_unit(pressure_unit, system))
    if quantity == "force":
        return 0.1
    if quantity == "force_per_speed":
        return 0.0001
    if quantity == "force_per_speed_squared":
        return 0.000001
    if quantity == "cda":
        return 0.001
    if quantity == "rrc":
        return 0.001
    numeric = _to_float(canonical_step)
    return numeric if numeric is not None else canonical_step


def display_precision_for_field(field_key, unit_system, *, pressure_unit: str | None = None):
    normalized_key = _normalize_field_key(field_key)
    quantity = quantity_kind_for_field(field_key)
    system = normalize_unit_system(unit_system)
    if quantity is None or quantity == "mass_interval":
        return None
    if quantity == "mass":
        if normalized_key == "custom_delta_kg" and system == UNIT_SYSTEM_METRIC:
            return 1
        return 0
    if quantity == "pressure":
        return pressure_display_precision(_resolved_pressure_unit(pressure_unit, system))
    if quantity == "force":
        return 2
    if quantity == "force_per_speed":
        return 4
    if quantity == "force_per_speed_squared":
        return 6
    if quantity == "cda":
        return 3
    if quantity == "rrc":
        return 3
    return None


def display_format_for_field(field_key, canonical_format, unit_system, *, pressure_unit: str | None = None):
    precision = display_precision_for_field(field_key, unit_system, pressure_unit=pressure_unit)
    if precision is not None:
        if quantity_kind_for_field(field_key) == "pressure":
            return pressure_display_format(_resolved_pressure_unit(pressure_unit, unit_system))
        return f"%.{int(precision)}f"
    if canonical_format:
        return canonical_format
    quantity = quantity_kind_for_field(field_key)
    if quantity is None or quantity == "mass_interval":
        return None
    if quantity == "rrc":
        return "%.3f"
    rendered = format_quantity(1.0, quantity, normalize_unit_system(unit_system), include_unit=False, unavailable="")
    if "." not in rendered:
        return "%.0f"
    digits = len(rendered.split(".", 1)[1])
    return f"%.{digits}f"


def format_select_option_for_field(field_key, option_value, unit_system, *, include_unit=True, pressure_unit: str | None = None) -> str:
    if is_blank(option_value):
        return ""
    display_value = format_display_value_for_field(field_key, option_value, unit_system, unavailable="", pressure_unit=pressure_unit)
    if display_value == "":
        return ""
    display_unit = display_unit_for_field(field_key, unit_system, pressure_unit=pressure_unit)
    if include_unit and display_unit and quantity_kind_for_field(field_key) not in {"mass_interval"}:
        return f"{display_value} {display_unit}"
    return display_value


def field_uses_display_units(field_key) -> bool:
    return quantity_kind_for_field(field_key) in _UNIT_SENSITIVE_QUANTITIES


def format_display_value_for_field(
    field_key,
    canonical_value,
    unit_system,
    unavailable="\u2014",
    *,
    pressure_unit: str | None = None,
):
    quantity = quantity_kind_for_field(field_key)
    system = normalize_unit_system(unit_system)
    if quantity == "mass_interval":
        rendered = _display_interval_value(canonical_value, system)
        return unavailable if is_blank(rendered) else str(rendered)
    if quantity is None:
        return unavailable if is_blank(canonical_value) else str(canonical_value)
    if is_blank(canonical_value):
        return unavailable
    numeric = _to_float(canonical_value)
    if numeric is None:
        return str(canonical_value)
    display_numeric = _to_float(display_value_for_field(field_key, numeric, system, pressure_unit=pressure_unit))
    if display_numeric is None:
        return str(canonical_value)
    precision = display_precision_for_field(field_key, system, pressure_unit=pressure_unit)
    if precision is not None:
        return _format_number(display_numeric, digits=int(precision))
    return format_quantity(numeric, quantity, system, include_unit=False, unavailable=unavailable)


def format_value_map_for_display(value_map: dict | None, unit_system, *, unavailable="\u2014", pressure_unit: str | None = None) -> str:
    payload = deepcopy(dict(value_map or {}))
    if not payload:
        return unavailable
    parts: list[str] = []
    for key, value in payload.items():
        if is_blank(value):
            continue
        display_value = format_display_value_for_field(key, value, unit_system, unavailable=unavailable, pressure_unit=pressure_unit)
        display_unit = display_unit_for_field(key, unit_system, pressure_unit=pressure_unit)
        if display_unit and display_value != unavailable and quantity_kind_for_field(key) not in {"mass_interval"}:
            parts.append(f"{key}={display_value} {display_unit}")
        else:
            parts.append(f"{key}={display_value}")
    return " | ".join(parts) if parts else unavailable


def _normalize_field_key(field_key) -> str:
    text = str(field_key or "").strip()
    if not text:
        return ""
    if text in _FIELD_QUANTITY_BY_BASE:
        return text
    for prefix in _CONTROLLED_PREFIXES:
        if not text.startswith(prefix):
            continue
        candidate = text[len(prefix) :]
        if candidate in _FIELD_QUANTITY_BY_BASE:
            return candidate
    return text


def _display_interval_value(canonical_value, unit_system) -> str | None:
    text = str(canonical_value or "").strip()
    if not text:
        return None
    match = _INTERVAL_RE.match(text)
    if not match:
        return text
    lower = _parse_interval_bound(match.group("lower"))
    upper = _parse_interval_bound(match.group("upper"))
    unit = "kg" if normalize_unit_system(unit_system) == UNIT_SYSTEM_METRIC else "lb"
    lower_text = _format_interval_bound(lower, unit_system)
    upper_text = _format_interval_bound(upper, unit_system)
    return f"({lower_text}, {upper_text}] {unit}"


def _parse_interval_bound(value: str) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"-inf", "inf", "+inf"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _format_interval_bound(value: float | None, unit_system) -> str:
    if value is None:
        return "-inf"
    display = display_value_for_field("mass_kg", value, unit_system)
    numeric = _to_float(display)
    digits = display_precision_for_field("mass_kg", unit_system)
    return _format_number(numeric, digits=0 if digits is None else int(digits)) if numeric is not None else str(display)


def _format_number(value: float, *, digits: int) -> str:
    rendered = f"{float(value):.{digits}f}"
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _to_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


__all__ = [
    "display_unit_for_field",
    "display_format_for_field",
    "display_precision_for_field",
    "display_step_for_field",
    "display_value_for_field",
    "field_uses_display_units",
    "format_display_value_for_field",
    "format_select_option_for_field",
    "format_value_map_for_display",
    "quantity_kind_for_field",
    "to_canonical_field_value",
    "to_display_field_value",
]
