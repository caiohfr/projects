from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


UNIT_SYSTEM_METRIC = "Metric"
UNIT_SYSTEM_US = "US customary"
UNIT_SYSTEM_OPTIONS = [UNIT_SYSTEM_METRIC, UNIT_SYSTEM_US]
PRESSURE_UNIT_KPA = "kPa"
PRESSURE_UNIT_BAR = "bar"
PRESSURE_UNIT_PSI = "psi"
PRESSURE_UNIT_OPTIONS = (PRESSURE_UNIT_KPA, PRESSURE_UNIT_BAR, PRESSURE_UNIT_PSI)
KPA_PER_PSI = 6.894757293168
BAR_PER_PSI = KPA_PER_PSI / 100.0


@dataclass(frozen=True)
class UnitSpec:
    label: str
    factor_from_canonical: float
    format_str: str


@dataclass(frozen=True)
class QuantitySpec:
    metric: UnitSpec
    us: UnitSpec


REGISTRY: dict[str, QuantitySpec] = {
    "mass": QuantitySpec(
        metric=UnitSpec("kg", 1.0, "%.1f"),
        us=UnitSpec("lb", 2.20462262185, "%.1f"),
    ),
    "force": QuantitySpec(
        metric=UnitSpec("N", 1.0, "%.2f"),
        us=UnitSpec("lbf", 0.22480894387, "%.2f"),
    ),
    "force_per_speed": QuantitySpec(
        metric=UnitSpec("N/kph", 1.0, "%.5f"),
        us=UnitSpec("lbf/mph", 0.361794924964, "%.5f"),
    ),
    "force_per_speed_squared": QuantitySpec(
        metric=UnitSpec("N/kph^2", 1.0, "%.6f"),
        us=UnitSpec("lbf/mph^2", 0.58225249172, "%.6f"),
    ),
    "cda": QuantitySpec(
        metric=UnitSpec("m^2", 1.0, "%.3f"),
        us=UnitSpec("ft^2", 10.7639104167, "%.3f"),
    ),
    "pressure": QuantitySpec(
        metric=UnitSpec("kPa", 6.89475729317, "%.1f"),
        us=UnitSpec("psi", 1.0, "%.1f"),
    ),
    "speed": QuantitySpec(
        metric=UnitSpec("km/h", 1.0, "%.1f"),
        us=UnitSpec("mph", 0.621371192237, "%.1f"),
    ),
    "energy_per_distance": QuantitySpec(
        metric=UnitSpec("MJ/km", 1.0, "%.4f"),
        us=UnitSpec("Wh/mi", 447.04, "%.1f"),
    ),
    "energy_wh_per_distance": QuantitySpec(
        metric=UnitSpec("Wh/km", 1.0, "%.1f"),
        us=UnitSpec("Wh/mi", 1.609344, "%.1f"),
    ),
    "co2_per_distance": QuantitySpec(
        metric=UnitSpec("g/km", 1.0, "%.1f"),
        us=UnitSpec("g/mi", 1.609344, "%.1f"),
    ),
    "rrc": QuantitySpec(
        metric=UnitSpec("N/kN", 1.0, "%.3f"),
        us=UnitSpec("lbf/klbf", 1.0, "%.3f"),
    ),
    "percentage": QuantitySpec(
        metric=UnitSpec("%", 1.0, "%.1f"),
        us=UnitSpec("%", 1.0, "%.1f"),
    ),
    "fraction": QuantitySpec(
        metric=UnitSpec("-", 1.0, "%.3f"),
        us=UnitSpec("-", 1.0, "%.3f"),
    ),
}


def normalize_unit_system(unit_system: str | None) -> str:
    value = str(unit_system or st.session_state.get("unit_system") or UNIT_SYSTEM_METRIC).strip()
    return value if value in UNIT_SYSTEM_OPTIONS else UNIT_SYSTEM_METRIC


def normalize_pressure_unit(pressure_unit: str | None, default: str | None = None) -> str:
    fallback = str(default or PRESSURE_UNIT_KPA).strip() or PRESSURE_UNIT_KPA
    if fallback not in PRESSURE_UNIT_OPTIONS:
        fallback = PRESSURE_UNIT_KPA
    value = str(pressure_unit or "").strip()
    normalized = {
        "kpa": PRESSURE_UNIT_KPA,
        "bar": PRESSURE_UNIT_BAR,
        "psi": PRESSURE_UNIT_PSI,
    }.get(value.lower())
    return normalized or fallback


def pressure_unit_label(pressure_unit: str | None) -> str:
    return normalize_pressure_unit(pressure_unit)


def pressure_factor_from_canonical(pressure_unit: str | None) -> float:
    unit = normalize_pressure_unit(pressure_unit)
    return {
        PRESSURE_UNIT_KPA: KPA_PER_PSI,
        PRESSURE_UNIT_BAR: BAR_PER_PSI,
        PRESSURE_UNIT_PSI: 1.0,
    }[unit]


def pressure_to_display(canonical_value, pressure_unit: str | None):
    if canonical_value is None:
        return None
    return float(canonical_value) * pressure_factor_from_canonical(pressure_unit)


def pressure_to_canonical(display_value, pressure_unit: str | None):
    if display_value is None:
        return None
    return float(display_value) / pressure_factor_from_canonical(pressure_unit)


def pressure_display_precision(pressure_unit: str | None) -> int:
    unit = normalize_pressure_unit(pressure_unit)
    return {
        PRESSURE_UNIT_KPA: 0,
        PRESSURE_UNIT_BAR: 2,
        PRESSURE_UNIT_PSI: 1,
    }[unit]


def pressure_display_step(pressure_unit: str | None) -> float:
    unit = normalize_pressure_unit(pressure_unit)
    return {
        PRESSURE_UNIT_KPA: 1.0,
        PRESSURE_UNIT_BAR: 0.05,
        PRESSURE_UNIT_PSI: 0.5,
    }[unit]


def pressure_display_format(pressure_unit: str | None) -> str:
    return f"%.{pressure_display_precision(pressure_unit)}f"


def _spec_for(quantity: str, unit_system: str | None = None) -> UnitSpec:
    system = normalize_unit_system(unit_system)
    quantity_spec = REGISTRY[quantity]
    return quantity_spec.metric if system == UNIT_SYSTEM_METRIC else quantity_spec.us


def unit_label(quantity: str, unit_system: str | None = None) -> str:
    return _spec_for(quantity, unit_system).label


def to_display(canonical_value, quantity: str, unit_system: str | None = None):
    if canonical_value is None:
        return None
    spec = _spec_for(quantity, unit_system)
    return float(canonical_value) * spec.factor_from_canonical


def to_canonical(display_value, quantity: str, unit_system: str | None = None):
    if display_value is None:
        return None
    spec = _spec_for(quantity, unit_system)
    return float(display_value) / spec.factor_from_canonical


def format_quantity(
    canonical_value,
    quantity: str,
    unit_system: str | None = None,
    *,
    include_unit: bool = True,
    unavailable: str = "-",
    format_str: str | None = None,
):
    if canonical_value is None:
        return unavailable
    spec = _spec_for(quantity, unit_system)
    value = to_display(canonical_value, quantity, unit_system)
    rendered = (format_str or spec.format_str) % float(value or 0.0)
    if include_unit and spec.label != "-":
        return f"{rendered} {spec.label}"
    return rendered


def _label_with_unit(label: str, quantity: str, unit_system: str | None = None) -> str:
    if "[" in label and "]" in label:
        return label
    unit = unit_label(quantity, unit_system)
    if unit == "-":
        return label
    return f"{label} [{unit}]"


def _widget_key(key: str, unit_system: str | None = None) -> str:
    suffix = "metric" if normalize_unit_system(unit_system) == UNIT_SYSTEM_METRIC else "us"
    return f"{key}__{suffix}"


def quantity_input(
    host,
    label: str,
    canonical_value,
    quantity: str,
    *,
    key: str,
    unit_system: str | None = None,
    min_canonical=None,
    max_canonical=None,
    step_canonical=None,
    format_str: str | None = None,
    disabled: bool = False,
    help: str | None = None,
):
    system = normalize_unit_system(unit_system)
    spec = _spec_for(quantity, system)
    kwargs = {
        "label": _label_with_unit(label, quantity, system),
        "value": float(to_display(canonical_value, quantity, system) or 0.0),
        "key": _widget_key(key, system),
        "format": format_str or spec.format_str,
        "disabled": disabled,
    }
    if min_canonical is not None:
        kwargs["min_value"] = float(to_display(min_canonical, quantity, system) or 0.0)
    if max_canonical is not None:
        kwargs["max_value"] = float(to_display(max_canonical, quantity, system) or 0.0)
    if step_canonical is not None:
        kwargs["step"] = float(to_display(step_canonical, quantity, system) or 0.0)
    if help:
        kwargs["help"] = help
    display_value = host.number_input(**kwargs)
    return to_canonical(display_value, quantity, system)


def quantity_metric(
    host,
    label: str,
    canonical_value,
    quantity: str,
    *,
    unit_system: str | None = None,
    unavailable: str = "-",
    format_str: str | None = None,
):
    host.metric(
        _label_with_unit(label, quantity, unit_system),
        format_quantity(
            canonical_value,
            quantity,
            unit_system,
            include_unit=False,
            unavailable=unavailable,
            format_str=format_str,
        ),
    )
