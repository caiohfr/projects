"""Conservative, exact-match helpers for Sprint 12 closure Phase 2.

No fuzzy matching, numeric tolerances, or nearest-neighbour rules are used.
Temporal parents are assigned only for byte-equivalent evidence signatures and
an unambiguous previous available model year.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


HP_TO_KW = 0.745699872
EPA_MODEL_YEAR_CARRYOVER = "EPA_MODEL_YEAR_CARRYOVER"
ENGINEERING_SCENARIO = "ENGINEERING_SCENARIO"
RUN_IDENTITY_REVIEW = "RUN_IDENTITY_REVIEW"
EPA_NORMAL = "EPA NORMAL"
EPA_COLD = "EPA COLD"
EPA_CUSTOM = "EPA CUSTOM"
EPA_CYCLE_SOURCE = "EPA_TESTCAR"

EPA_ROADLOAD_CONDITION_FIELDS = (
    "Roadload Condition",
    "Road Load Condition",
    "Roadload Condition Description",
    "Road Load Condition Description",
)
EPA_ROADLOAD_TEMPERATURE_FIELDS = (
    "Roadload Temperature (C)",
    "Road Load Temperature (C)",
    "Roadload Ambient Temperature (C)",
)
EPA_ROADLOAD_PRESSURE_FIELDS = (
    "Roadload Ambient Pressure (kPa)",
    "Road Load Ambient Pressure (kPa)",
)


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("Non-finite source value cannot enter a carryover signature")
        return format(value, ".15g")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return value.strip()
    return value


def exact_signature(payload: Any) -> str:
    material = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=clean_scalar, allow_nan=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest().upper()


def record_signature(record: Mapping[str, Any], fields: Iterable[str]) -> str:
    return exact_signature({field: clean_scalar(record.get(field)) for field in fields})


def group_signature(records: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> str:
    field_tuple = tuple(fields)
    evidence = [
        {field: clean_scalar(record.get(field)) for field in field_tuple}
        for record in records
    ]
    evidence.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return exact_signature(evidence)


def horsepower_to_kw(value: Any) -> float | None:
    value = clean_scalar(value)
    return None if value is None else float(value) * HP_TO_KW


def cylinders_rotors(value: Any) -> int | None:
    value = clean_scalar(value)
    if value is None:
        return None
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"Cylinders/rotors must be integral, got {value!r}")
    return int(numeric)


def _first_explicit(record: Mapping[str, Any], fields: Iterable[str]) -> tuple[str | None, Any]:
    for field in fields:
        value = clean_scalar(record.get(field))
        if value not in (None, ""):
            return field, value
    return None, None


def _finite_float(value: Any, field: str | None) -> float | None:
    value = clean_scalar(value)
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite roadload condition value in {field or 'source'}")
    return numeric


def classify_epa_roadload_condition(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify only explicit roadload-condition evidence, never Run schedules.

    The current EPA TestCar extract has no roadload-condition scalar columns.
    Its Target ABC rows therefore use the standard regulatory roadload family
    while temperature and pressure remain unknown/NULL.  If future extracts
    provide one of the explicit fields above, the raw field and value remain
    auditable in the returned provenance.
    """
    condition_field, condition_value = _first_explicit(record, EPA_ROADLOAD_CONDITION_FIELDS)
    temperature_field, temperature_value = _first_explicit(record, EPA_ROADLOAD_TEMPERATURE_FIELDS)
    pressure_field, pressure_value = _first_explicit(record, EPA_ROADLOAD_PRESSURE_FIELDS)

    if condition_value is None:
        family = EPA_NORMAL
        basis = "REGULATORY_STANDARD_DEFAULT"
    else:
        normalized = str(condition_value).strip().upper()
        if "COLD" in normalized:
            family = EPA_COLD
        elif any(token in normalized for token in ("NORMAL", "STANDARD", "NOMINAL")):
            family = EPA_NORMAL
        else:
            family = EPA_CUSTOM
        basis = "EXPLICIT_SOURCE_ROADLOAD_CONDITION"

    temperature_c = _finite_float(temperature_value, temperature_field)
    pressure_kpa = _finite_float(pressure_value, pressure_field)
    scalar_basis = "SOURCE_DIRECT" if temperature_c is not None or pressure_kpa is not None else "UNKNOWN"
    return {
        "cycle_name": family,
        "cycle_source": EPA_CYCLE_SOURCE,
        "roadload_temperature_c": temperature_c,
        "roadload_ambient_pressure_kpa": pressure_kpa,
        "provenance": {
            "classification_basis": basis,
            "condition_source_field": condition_field,
            "condition_source_value": condition_value,
            "temperature_source_field": temperature_field,
            "temperature_source_value": temperature_value,
            "ambient_pressure_source_field": pressure_field,
            "ambient_pressure_source_value": pressure_value,
            "scalar_value_basis": scalar_basis,
            "run_schedule_fields_used": False,
        },
    }


def assign_temporal_parents(records: Iterable[Mapping[str, Any]]) -> dict[Any, dict[str, Any]]:
    """Assign the immediate previous available year for an exact signature."""
    by_signature: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_signature[str(record["signature"])].append(record)

    result: dict[Any, dict[str, Any]] = {}
    for signature, members in by_signature.items():
        by_year: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for member in members:
            by_year[int(member["year"])].append(member)
        previous_year: int | None = None
        for year in sorted(by_year):
            current = by_year[year]
            previous = [] if previous_year is None else by_year[previous_year]
            ambiguous = len(current) != 1 or (previous_year is not None and len(previous) != 1)
            for member in current:
                if ambiguous:
                    status, parent_id = "AMBIGUOUS", None
                elif previous_year is None:
                    status, parent_id = "ROOT", None
                else:
                    status, parent_id = "LINKED", previous[0]["id"]
                result[member["id"]] = {
                    "status": status, "signature": signature,
                    "parent_id": parent_id, "parent_year": previous_year,
                }
            previous_year = year
    return result
