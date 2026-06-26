"""
Service helpers for Tire Roadload DB integration.

This module orchestrates:
- tire CRUD helpers over repositories
- preview calculation for a VDE row using tire records
- persistence of applied tire ABC back into vde_db

It intentionally stays free of Streamlit/UI code.
"""

from __future__ import annotations

import math
from typing import Any

from src.vde_core.repositories import (
    create_tire_roadload,
    deactivate_tire_roadload,
    delete_tire_roadload,
    fetch_vde_by_id,
    get_tire_roadload_by_code,
    get_tire_roadload_by_id,
    get_vde_tire_application,
    list_tire_roadload_active,
    search_tire_roadload,
    update_tire_roadload,
    update_vde_tire_application,
)
from src.vde_core.test_mass import inertia_class_from_mass
from src.vde_core.roadload import (
    KPA_PER_PSI,
    N_PER_KGF,
    N_PER_LBF,
    build_tire_component,
    calculate_sae_smerf_rr_n_per_kn,
    calculate_vehicle_tire_abc,
)


PSI_TO_KPA = KPA_PER_PSI
CALCULATION_MODES = ("SAE_J2452", "ISO_28580", "EU_LABEL_ESTIMATED", "CUSTOM")
EPA_TEST_MASS_DEFAULT_DELTA_KG = 136.0


def _to_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _positive_float(value, default=None):
    out = _to_float(value, default)
    if out is None or out <= 0:
        return default
    return out


def _to_int(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(value)
    except Exception:
        return default


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "TRUE", "True", "yes", "YES", "on", "ON"):
        return True
    return False


def _first_value(*values, default=None):
    for value in values:
        if value not in (None, ""):
            return value
    return default


def _first_named_value(*pairs, default=None):
    for name, value in pairs:
        if value not in (None, ""):
            return name, value
    return None, default


def _normalize_standard_family(value) -> str:
    return str(value or "").strip().upper()


def _normalize_calculation_mode(value, standard_family=None) -> str:
    mode = str(value or "").strip().upper()
    mode = mode.replace(" ", "_").replace("-", "_")
    if mode in CALCULATION_MODES:
        return mode
    family = _normalize_standard_family(standard_family)
    if family == "SAE":
        return "SAE_J2452"
    if family == "ISO":
        return "ISO_28580"
    return "CUSTOM"


def _standard_family_from_mode(mode: str) -> str:
    mode = _normalize_calculation_mode(mode)
    if mode == "SAE_J2452":
        return "SAE"
    if mode == "ISO_28580":
        return "ISO"
    return "CUSTOM"


def _normalize_basis(value) -> str:
    basis = str(value or "TEST_MASS").strip().upper()
    if basis not in {"TEST_MASS", "TWC"}:
        raise ValueError(f"Unsupported tire_load_mass_basis: {value!r}")
    return basis


def _psi_to_kpa(psi):
    out = _to_float(psi)
    return None if out is None else out * PSI_TO_KPA


def _kpa_to_psi(kpa):
    out = _to_float(kpa)
    return None if out is None else out / PSI_TO_KPA


def _normalize_unit(value, default: str) -> str:
    text = str(value or default).strip()
    return text or default


def _pressure_to_kpa(value, unit: str):
    out = _to_float(value)
    if out is None:
        return None
    unit = str(unit or "").strip().lower()
    if unit in {"psi", "psig"}:
        return out * KPA_PER_PSI
    return out


def _pressure_from_kpa(value, unit: str):
    out = _to_float(value)
    if out is None:
        return None
    unit = str(unit or "").strip().lower()
    if unit in {"psi", "psig"}:
        return out / KPA_PER_PSI
    return out


def _load_to_n(value, unit: str):
    out = _to_float(value)
    if out is None:
        return None
    unit = str(unit or "").strip().lower()
    if unit in {"lbf", "lb", "lbs", "pound-force", "pounds-force"}:
        return out * N_PER_LBF
    if unit in {"kg", "kgf", "kilogram-force", "kilograms-force"}:
        return out * N_PER_KGF
    return out


def _load_from_n(value, unit: str):
    out = _to_float(value)
    if out is None:
        return None
    unit = str(unit or "").strip().lower()
    if unit in {"lbf", "lb", "lbs", "pound-force", "pounds-force"}:
        return out / N_PER_LBF
    if unit in {"kg", "kgf", "kilogram-force", "kilograms-force"}:
        return out / N_PER_KGF
    return out


def _normalize_tire_payload(payload: dict | None) -> dict:
    data = dict(payload or {})
    data["calculation_mode"] = _normalize_calculation_mode(
        data.get("calculation_mode"),
        data.get("standard_family"),
    )
    data["standard_family"] = _standard_family_from_mode(data["calculation_mode"])
    is_sae = data["calculation_mode"] == "SAE_J2452"

    for key in (
        "is_active",
        "is_tested_value",
        "is_estimated_value",
        "is_broken_in",
        "temperature_correction_applied",
    ):
        if key in data and data[key] not in (None, ""):
            data[key] = 1 if _truthy(data[key]) else 0
    for key in (
        "rr_n_per_kn",
        "sae_a",
        "sae_b",
        "sae_c",
        "sae_alpha",
        "sae_beta",
        "iso_rrc_n_per_kn",
        "iso_test_pressure_kpa",
        "iso_test_load_n",
        "iso_test_speed_kph",
        "iso_rolling_resistance_force_n",
        "iso_corrected_rrc_n_per_kn",
        "test_mileage_km",
        "break_in_distance_km",
        "test_temperature_c",
        "reference_temperature_c",
        "effective_circumference_override_mm",
        "sae_reference_pressure_kpa",
        "sae_reference_load_n",
        "test_pressure_value",
        "test_load_value",
        "test_speed_value",
        "smerf",
    ):
        if key in data:
            data[key] = _to_float(data.get(key))

    default_pressure_unit = "kPa" if is_sae else "kPa"
    default_load_unit = "kg" if is_sae else "N"
    data["pressure_unit"] = _normalize_unit(data.get("pressure_unit") or data.get("sae_pressure_unit"), default_pressure_unit)
    data["load_unit"] = _normalize_unit(data.get("load_unit") or data.get("sae_load_unit"), default_load_unit)
    data["speed_unit"] = _normalize_unit(data.get("speed_unit"), "kph")
    data["force_unit"] = _normalize_unit(data.get("force_unit"), "N")

    data["sae_pressure_unit"] = _normalize_unit(data.get("sae_pressure_unit"), "kPa")
    data["sae_load_unit"] = _normalize_unit(data.get("sae_load_unit"), "N")
    data["sae_speed_unit"] = _normalize_unit(data.get("sae_speed_unit"), "kph")
    data["sae_force_unit"] = _normalize_unit(data.get("sae_force_unit"), "N")

    if data.get("test_pressure_value") is None and data.get("sae_reference_pressure_kpa") is not None:
        data["test_pressure_value"] = _pressure_from_kpa(
            data.get("sae_reference_pressure_kpa"),
            data["pressure_unit"],
        )
    elif data.get("test_pressure_value") is not None:
        data["sae_reference_pressure_kpa"] = _pressure_to_kpa(
            data.get("test_pressure_value"),
            data["pressure_unit"],
        )

    if data.get("test_load_value") is None and data.get("sae_reference_load_n") is not None:
        data["test_load_value"] = _load_from_n(
            data.get("sae_reference_load_n"),
            data["load_unit"],
        )
    elif data.get("test_load_value") is not None:
        data["sae_reference_load_n"] = _load_to_n(
            data.get("test_load_value"),
            data["load_unit"],
        )

    rr_summary = summarize_tire_rr(data)
    data.update({k: v for k, v in rr_summary.items() if v is not None})
    if data.get("rr_n_per_kn") in (None, ""):
        data["rr_n_per_kn"] = 0.0
    return data


def summarize_tire_rr(payload: dict | None) -> dict:
    data = dict(payload or {})
    mode = _normalize_calculation_mode(data.get("calculation_mode"), data.get("standard_family"))

    if mode == "SAE_J2452":
        explicit_smerf = _positive_float(data.get("smerf"))
        explicit_rr_n_per_kn = _positive_float(data.get("rr_n_per_kn"))
        if explicit_smerf is not None or explicit_rr_n_per_kn is not None:
            load_unit = _normalize_unit(data.get("load_unit") or data.get("sae_load_unit"), "kg")
            reference_load_n = _positive_float(
                _first_value(
                    data.get("sae_reference_load_n"),
                    _load_to_n(data.get("test_load_value"), load_unit),
                )
            )
            if explicit_smerf is not None and explicit_rr_n_per_kn is not None:
                rr_quality = "reference_rr_and_smerf_input"
                rr_n_per_kn = explicit_rr_n_per_kn
            elif explicit_smerf is not None:
                if reference_load_n is not None:
                    rr_quality = "reference_smerf_force_input"
                    rr_n_per_kn = explicit_smerf * 1000.0 / reference_load_n
                else:
                    rr_quality = "reference_smerf_input_missing_reference_load"
                    rr_n_per_kn = explicit_smerf
            else:
                rr_quality = "reference_rr_input"
                rr_n_per_kn = explicit_rr_n_per_kn
            return {
                "calculation_mode": mode,
                "standard_family": "SAE",
                "rr_n_per_kn": rr_n_per_kn,
                "smerf": explicit_smerf,
                "rr_method": "SAE_J2452_SMERF_EPA_55_45",
                "rr_source": data.get("rr_source") or data.get("test_source") or rr_quality,
                "rr_quality": data.get("rr_quality") or rr_quality,
            }

        pressure_unit = _normalize_unit(data.get("pressure_unit") or data.get("sae_pressure_unit"), "kPa")
        load_unit = _normalize_unit(data.get("load_unit") or data.get("sae_load_unit"), "kg")
        pressure = _positive_float(
            _first_value(
                _pressure_to_kpa(data.get("test_pressure_value"), pressure_unit),
                data.get("sae_reference_pressure_kpa"),
            )
        )
        nominal_load_n = _positive_float(
            _first_value(
                _load_to_n(data.get("test_load_value"), load_unit),
                data.get("sae_reference_load_n"),
            )
        )
        load_value = nominal_load_n
        a = _to_float(data.get("sae_a"))
        b = _to_float(data.get("sae_b"))
        c = _to_float(data.get("sae_c"))
        has_coefficients = any(value not in (None, 0.0) for value in (a, b, c))
        if pressure is None or load_value is None or not has_coefficients:
            return {
                "calculation_mode": mode,
                "standard_family": "SAE",
                "rr_method": "SAE_J2452_SMERF_EPA_55_45",
                "rr_source": data.get("rr_source") or data.get("test_source"),
                "rr_quality": "missing_sae_inputs",
                "smerf": None,
            }
        try:
            result = calculate_sae_smerf_rr_n_per_kn(
                alpha=_to_float(data.get("sae_alpha"), 0.0),
                beta=_to_float(data.get("sae_beta"), 0.0),
                a=a,
                b=b,
                c=c,
                pressure_kpa=pressure,
                load_n=load_value,
            )
        except Exception:
            return {
                "calculation_mode": mode,
                "standard_family": "SAE",
                "rr_method": "SAE_J2452_SMERF_EPA_55_45",
                "rr_source": data.get("rr_source") or data.get("test_source"),
                "rr_quality": "missing_sae_inputs",
                "smerf": None,
            }
        return {
            "calculation_mode": mode,
            "standard_family": "SAE",
            "rr_n_per_kn": result["rr_n_per_kn"],
            "smerf": result["smerf"],
            "rr_method": "SAE_J2452_SMERF_EPA_55_45",
            "rr_source": data.get("rr_source") or data.get("test_source") or "calculated_from_sae_coefficients",
            "rr_quality": "calculated_from_sae_coefficients",
        }

    if mode == "ISO_28580":
        rr_n_per_kn = _positive_float(
            _first_value(
                data.get("iso_corrected_rrc_n_per_kn"),
                data.get("iso_rrc_n_per_kn"),
                data.get("rr_n_per_kn"),
            )
        )
        return {
            "calculation_mode": mode,
            "standard_family": "ISO",
            "rr_n_per_kn": rr_n_per_kn,
            "smerf": rr_n_per_kn,
            "rr_method": "ISO_SIMPLE_RRC",
            "rr_source": data.get("rr_source") or data.get("test_source") or "iso_rrc_input",
            "rr_quality": "measured_or_corrected_iso" if rr_n_per_kn is not None else "missing_iso_rrc",
        }

    if mode == "EU_LABEL_ESTIMATED":
        rr_n_per_kn = _positive_float(data.get("rr_n_per_kn"))
        return {
            "calculation_mode": mode,
            "standard_family": "CUSTOM",
            "rr_n_per_kn": rr_n_per_kn,
            "smerf": rr_n_per_kn,
            "rr_method": "EU_LABEL_ESTIMATED",
            "rr_source": data.get("rr_source") or data.get("test_source") or "estimated_from_label_class",
            "rr_quality": "estimated_from_label_class" if rr_n_per_kn is not None else "missing_estimate",
        }

    rr_n_per_kn = _positive_float(data.get("rr_n_per_kn"))
    return {
        "calculation_mode": mode,
        "standard_family": "CUSTOM",
        "rr_n_per_kn": rr_n_per_kn,
        "smerf": rr_n_per_kn,
        "rr_method": data.get("rr_method") or "MANUAL_ESTIMATED",
        "rr_source": data.get("rr_source") or data.get("test_source") or data.get("rr_value_source_note"),
        "rr_quality": data.get("rr_quality") or ("manual_input" if rr_n_per_kn is not None else "missing_rr_value"),
    }


def compute_tire_smerf(payload: dict | None) -> float | None:
    return summarize_tire_rr(payload).get("smerf")


def create_tire_from_form(payload: dict) -> int:
    data = _normalize_tire_payload(payload)
    return create_tire_roadload(data)


def update_tire_from_form(tire_id: int, payload: dict) -> None:
    data = _normalize_tire_payload(payload)
    update_tire_roadload(int(tire_id), data)


def get_tire_by_id(tire_id: int) -> dict:
    return get_tire_roadload_by_id(int(tire_id))


def get_tire_by_code(tire_test_code: str) -> dict:
    return get_tire_roadload_by_code(str(tire_test_code))


def get_available_tires(filters: dict | None = None) -> list[dict]:
    params = dict(filters or {})
    if not params:
        return list_tire_roadload_active()
    return search_tire_roadload(
        manufacturer=params.get("manufacturer"),
        model=params.get("model"),
        size_code=params.get("size_code"),
        standard_family=_normalize_standard_family(params.get("standard_family")) or None,
        min_test_mileage_km=_to_float(params.get("min_test_mileage_km")),
        active_only=not _truthy(params.get("include_inactive")),
    )


def deactivate_tire_record(tire_id: int) -> None:
    deactivate_tire_roadload(int(tire_id))


def delete_tire_record(tire_id: int) -> int:
    return delete_tire_roadload(int(tire_id))


def _resolve_default_test_mass(row: dict) -> tuple[str | None, float | None, str | None]:
    legislation = str((row or {}).get("legislation") or "").strip().upper()
    base_mass_kg = _positive_float((row or {}).get("mass_kg"))
    if legislation == "EPA" and base_mass_kg is not None:
        return (
            "default_test_mass_kg_epa",
            base_mass_kg + EPA_TEST_MASS_DEFAULT_DELTA_KG,
            "EPA_CURB_PLUS_136KG",
        )
    return None, None, None


def resolve_tire_load_mass(vde_row: dict, tire_load_mass_basis: str) -> dict:
    row = dict(vde_row or {})
    basis = _normalize_basis(tire_load_mass_basis)

    if basis == "TEST_MASS":
        default_source, default_mass_kg, default_rule = _resolve_default_test_mass(row)
        candidates = [("test_mass_kg", row.get("test_mass_kg"))]
        if default_mass_kg is not None:
            candidates.append((default_source, default_mass_kg))
        candidates.append(("mass_kg", row.get("mass_kg")))
    else:
        default_rule = None
        candidates = [
            ("twc_kg", row.get("twc_kg")),
            ("etw_kg", row.get("etw_kg")),
            ("inertia_class", row.get("inertia_class")),
            ("test_mass_kg", row.get("test_mass_kg")),
            ("mass_kg", row.get("mass_kg")),
        ]

    for field_name, raw_value in candidates:
        value = _to_float(raw_value)
        if value is not None and value > 0:
            return {
                "basis": basis,
                "mass_kg": value,
                "source_field": field_name,
                "used_fallback": field_name not in {"test_mass_kg", "twc_kg"},
                "used_inertia_class": field_name == "inertia_class",
                "test_mass_defaulted": field_name == "default_test_mass_kg_epa",
                "test_mass_default_rule": default_rule if field_name == "default_test_mass_kg_epa" else None,
            }

    raise ValueError(f"Could not resolve tire load mass for basis {basis}")


def build_tire_application_inputs_from_vde_row(vde_id: int) -> dict:
    row = fetch_vde_by_id(int(vde_id))
    if not row:
        raise ValueError(f"VDE row not found: id={vde_id}")
    app = get_vde_tire_application(int(vde_id))
    merged = dict(row)
    merged.update(app or {})
    merged["front_weight_distribution_pct"] = merged.get("weight_dist_fr_pct")
    return merged


def _apply_preview_row_overrides(vde_row: dict, payload: dict | None) -> dict:
    row = dict(vde_row or {})
    data = dict(payload or {})
    for key in ("legislation", "mass_kg", "test_mass_kg", "inertia_class", "twc_kg", "etw_kg"):
        value = data.get(key)
        if value not in (None, ""):
            row[key] = value
    legislation = str(row.get("legislation") or "").strip().upper()
    basis = _normalize_basis(data.get("tire_load_mass_basis") or row.get("tire_load_mass_basis") or "TEST_MASS")
    base_mass = _positive_float(row.get("mass_kg"))
    if legislation == "EPA" and basis == "TWC" and base_mass is not None:
        derived_twc = inertia_class_from_mass(base_mass)
        if derived_twc is not None:
            row["twc_kg"] = derived_twc
            row["inertia_class"] = derived_twc
    return row


def _normalize_application_payload(vde_row: dict, payload: dict | None) -> dict:
    row = dict(vde_row or {})
    data = dict(payload or {})

    front_tire_id = _to_int(_first_value(data.get("front_tire_id"), row.get("front_tire_id")))
    rear_tire_id = _to_int(_first_value(data.get("rear_tire_id"), row.get("rear_tire_id")))
    same_tire_front_rear = _truthy(_first_value(data.get("same_tire_front_rear"), data.get("same_tire"), False))
    rear_tire_source = "rear_tire_id"
    if same_tire_front_rear and front_tire_id is not None:
        rear_tire_id = front_tire_id
        rear_tire_source = "same_tire_front_rear"

    front_pressure_source, front_pressure_value = _first_named_value(
        ("front_pressure_psi", data.get("front_pressure_psi")),
        ("front_pressure_kpa", _kpa_to_psi(data.get("front_pressure_kpa"))),
        ("front_tire_pressure_placard", data.get("front_tire_pressure_placard")),
        ("saved_front_pressure_psi", row.get("front_pressure_psi")),
    )
    rear_pressure_source, rear_pressure_value = _first_named_value(
        ("rear_pressure_psi", data.get("rear_pressure_psi")),
        ("rear_pressure_kpa", _kpa_to_psi(data.get("rear_pressure_kpa"))),
        ("rear_tire_pressure_placard", data.get("rear_tire_pressure_placard")),
        ("saved_rear_pressure_psi", row.get("rear_pressure_psi")),
    )
    front_pressure_psi = _to_float(front_pressure_value)
    rear_pressure_psi = _to_float(rear_pressure_value)
    front_weight_distribution_pct = _to_float(
        _first_value(
            data.get("front_weight_distribution_pct"),
            data.get("weight_dist_fr_pct"),
            row.get("weight_dist_fr_pct"),
        )
    )
    front_weight_distribution_pct_defaulted = front_weight_distribution_pct is None
    if front_weight_distribution_pct_defaulted:
        front_weight_distribution_pct = 50.0
    tire_improvement_pct = _to_float(
        _first_value(
            data.get("tire_improvement_pct"),
            data.get("tire_improve_pct"),
            row.get("tire_improvement_pct"),
            0.0,
        ),
        0.0,
    )
    tire_load_mass_basis = _normalize_basis(
        _first_value(data.get("tire_load_mass_basis"), row.get("tire_load_mass_basis"), "TEST_MASS")
    )

    return {
        "front_tire_id": front_tire_id,
        "rear_tire_id": rear_tire_id,
        "same_tire_front_rear": same_tire_front_rear,
        "rear_tire_source": rear_tire_source,
        "front_pressure_psi": front_pressure_psi,
        "rear_pressure_psi": rear_pressure_psi,
        "front_pressure_source": front_pressure_source,
        "rear_pressure_source": rear_pressure_source,
        "front_weight_distribution_pct": front_weight_distribution_pct,
        "front_weight_distribution_pct_defaulted": front_weight_distribution_pct_defaulted,
        "tire_improvement_pct": tire_improvement_pct,
        "tire_load_mass_basis": tire_load_mass_basis,
    }


def build_tire_component_from_result(calculation_result: dict):
    final_abc = ((calculation_result or {}).get("calculation") or {}).get("total_final_abc") or {}
    source = ((calculation_result or {}).get("save_payload") or {}).get("tire_calc_source") or "tire_roadload_service"
    meta = {
        "front_tire_id": ((calculation_result or {}).get("application") or {}).get("front_tire_id"),
        "rear_tire_id": ((calculation_result or {}).get("application") or {}).get("rear_tire_id"),
        "basis": ((calculation_result or {}).get("mass_resolution") or {}).get("basis"),
    }
    return build_tire_component("tire", final_abc, source=source, meta=meta)


def _derive_rrc_for_vde(calculation: dict) -> float | None:
    total_final = (calculation or {}).get("total_final_abc") or {}
    loads = (calculation or {}).get("loads") or {}
    b_val = _to_float(total_final.get("B"), 0.0)
    c_val = _to_float(total_final.get("C"), 0.0)
    if abs(b_val) > 1e-12 or abs(c_val) > 1e-12:
        return None
    total_load_kn = _to_float(loads.get("total_load_n"))
    if total_load_kn is None or total_load_kn <= 0:
        return None
    return _to_float(total_final.get("A"), 0.0) / (total_load_kn / 1000.0)


_REQUIRED_TIRE_SAVE_FIELDS = (
    "front_tire_id",
    "rear_tire_id",
    "front_pressure_psi",
    "rear_pressure_psi",
    "weight_dist_fr_pct",
    "tire_improvement_pct",
    "tire_load_mass_basis",
    "tire_load_mass_used_kg",
    "tire_A_final",
    "tire_B_final",
    "tire_C_final",
    "tire_calc_source",
    "tire_calc_notes",
)


def _validate_tire_save_payload(payload: dict) -> dict:
    data = dict(payload or {})
    missing = [key for key in _REQUIRED_TIRE_SAVE_FIELDS if data.get(key) in (None, "")]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"save_payload is missing required tire fields: {joined}")
    return data


def _validate_preview_inputs(application: dict, front_tire: dict, rear_tire: dict) -> None:
    if application["front_tire_id"] is None:
        raise ValueError("front_tire_id is required for tire preview")
    if application["rear_tire_id"] is None:
        raise ValueError("rear_tire_id is required for tire preview")
    if not front_tire:
        raise ValueError(f"Front tire not found: id={application['front_tire_id']}")
    if not rear_tire:
        raise ValueError(f"Rear tire not found: id={application['rear_tire_id']}")

    if _normalize_standard_family(front_tire.get("standard_family")) == "SAE" and application["front_pressure_psi"] is None:
        raise ValueError("front_pressure_psi is required for SAE front tire preview")
    if _normalize_standard_family(rear_tire.get("standard_family")) == "SAE" and application["rear_pressure_psi"] is None:
        raise ValueError("rear_pressure_psi is required for SAE rear tire preview")


def _build_tire_calc_notes(application: dict, mass_resolution: dict, front_tire: dict, rear_tire: dict) -> str:
    fields = (
        ("basis", mass_resolution["basis"]),
        ("mass_source", mass_resolution["source_field"]),
        ("mass_used_fallback", mass_resolution["used_fallback"]),
        ("uses_inertia_class_mass", mass_resolution["used_inertia_class"]),
        ("test_mass_defaulted", mass_resolution.get("test_mass_defaulted")),
        ("test_mass_default_rule", mass_resolution.get("test_mass_default_rule")),
        ("rear_tire_source", application["rear_tire_source"]),
        ("front_pressure_source", application["front_pressure_source"]),
        ("rear_pressure_source", application["rear_pressure_source"]),
        ("front_weight_distribution_pct", application["front_weight_distribution_pct"]),
        ("weight_dist_defaulted", application["front_weight_distribution_pct_defaulted"]),
        ("front_standard", _normalize_standard_family(front_tire.get("standard_family"))),
        ("rear_standard", _normalize_standard_family(rear_tire.get("standard_family"))),
    )
    return "; ".join(f"{key}={value}" for key, value in fields)


def _preview_tire_roadload(base_row: dict, payload: dict | None = None, *, vde_id: int | None = None) -> dict:
    base_row = _apply_preview_row_overrides(base_row, payload)
    application = _normalize_application_payload(base_row, payload)

    front_tire = get_tire_roadload_by_id(application["front_tire_id"])
    rear_tire = get_tire_roadload_by_id(application["rear_tire_id"])
    _validate_preview_inputs(application, front_tire, rear_tire)

    mass_resolution = resolve_tire_load_mass(base_row, application["tire_load_mass_basis"])
    calc_inputs = {
        "mass_kg": mass_resolution["mass_kg"],
        "front_weight_distribution_pct": application["front_weight_distribution_pct"],
        "front_pressure_kpa": _psi_to_kpa(application["front_pressure_psi"]),
        "rear_pressure_kpa": _psi_to_kpa(application["rear_pressure_psi"]),
        "tire_improvement_pct": application["tire_improvement_pct"],
    }
    calculation = calculate_vehicle_tire_abc(front_tire=front_tire, rear_tire=rear_tire, inputs=calc_inputs)

    notes = _build_tire_calc_notes(application, mass_resolution, front_tire, rear_tire)
    save_payload = {
        "front_tire_id": application["front_tire_id"],
        "rear_tire_id": application["rear_tire_id"],
        "front_pressure_psi": application["front_pressure_psi"],
        "rear_pressure_psi": application["rear_pressure_psi"],
        "weight_dist_fr_pct": application["front_weight_distribution_pct"],
        "tire_improvement_pct": application["tire_improvement_pct"],
        "tire_load_mass_basis": mass_resolution["basis"],
        "tire_load_mass_used_kg": calculation["tire_load_mass_used_kg"],
        "tire_A_final": calculation["total_final_abc"]["A"],
        "tire_B_final": calculation["total_final_abc"]["B"],
        "tire_C_final": calculation["total_final_abc"]["C"],
        "rrc_N_per_kN": _to_float(calculation.get("applied_rr_n_per_kn"), _derive_rrc_for_vde(calculation)),
        "tire_calc_source": (
            "tire_roadload_db:"
            f"{_normalize_standard_family(front_tire.get('standard_family'))}/"
            f"{_normalize_standard_family(rear_tire.get('standard_family'))}"
        ),
        "tire_calc_notes": notes,
    }

    component = build_tire_component(
        "tire",
        calculation["total_final_abc"],
        source=save_payload["tire_calc_source"],
        meta={
            "front_tire_id": application["front_tire_id"],
            "rear_tire_id": application["rear_tire_id"],
            "basis": mass_resolution["basis"],
        },
    )

    current_saved = {
        "A": _to_float(base_row.get("tire_A_final")),
        "B": _to_float(base_row.get("tire_B_final")),
        "C": _to_float(base_row.get("tire_C_final")),
    }
    delta_vs_saved = {}
    if any(value is not None for value in current_saved.values()):
        delta_vs_saved = {
            "A": save_payload["tire_A_final"] - (current_saved["A"] or 0.0),
            "B": save_payload["tire_B_final"] - (current_saved["B"] or 0.0),
            "C": save_payload["tire_C_final"] - (current_saved["C"] or 0.0),
        }

    return {
        "vde_id": int(vde_id) if vde_id is not None else None,
        "vde_row": base_row,
        "application": application,
        "front_tire": front_tire,
        "rear_tire": rear_tire,
        "mass_resolution": mass_resolution,
        "calculation": calculation,
        "component": component,
        "component_dict": component.to_dict(),
        "save_payload": save_payload,
        "delta_vs_saved": delta_vs_saved,
    }


def preview_tire_roadload_from_row(vde_row: dict, payload: dict | None = None) -> dict:
    base_row = dict(vde_row or {})
    if not base_row:
        raise ValueError("preview_tire_roadload_from_row requires a non-empty vde-like row context")
    preview = _preview_tire_roadload(base_row, payload, vde_id=_to_int(base_row.get("id")))
    if preview["vde_id"] is None and preview["mass_resolution"].get("source_field") == "test_mass_kg":
        preview["mass_resolution"]["used_fallback"] = True
        preview["save_payload"]["tire_calc_notes"] = _build_tire_calc_notes(
            preview["application"],
            preview["mass_resolution"],
            preview["front_tire"],
            preview["rear_tire"],
        )
    return preview


def preview_tire_roadload_for_vde(vde_id: int, payload: dict | None = None) -> dict:
    base_row = build_tire_application_inputs_from_vde_row(int(vde_id))
    return _preview_tire_roadload(base_row, payload, vde_id=int(vde_id))


def save_tire_roadload_to_vde(vde_id: int, calculation_result: dict) -> dict:
    result = dict(calculation_result or {})
    raw_payload = result.get("save_payload") or {}
    if not raw_payload:
        raise ValueError("save_tire_roadload_to_vde requires a preview-like calculation_result with save_payload")
    payload = _validate_tire_save_payload(raw_payload)
    update_vde_tire_application(int(vde_id), payload)
    return payload


def apply_tire_result_to_roadload_request(vde_id: int):
    row = get_vde_tire_application(int(vde_id))
    abc = {
        "A": _to_float(row.get("tire_A_final")),
        "B": _to_float(row.get("tire_B_final")),
        "C": _to_float(row.get("tire_C_final")),
    }
    if abc["A"] is None or abc["B"] is None or abc["C"] is None:
        return None
    return build_tire_component(
        "tire",
        abc,
        source=str(row.get("tire_calc_source") or "vde_db_saved_tire"),
        meta={
            "front_tire_id": row.get("front_tire_id"),
            "rear_tire_id": row.get("rear_tire_id"),
            "tire_load_mass_basis": row.get("tire_load_mass_basis"),
            "tire_load_mass_used_kg": row.get("tire_load_mass_used_kg"),
        },
    )


__all__ = [
    "CALCULATION_MODES",
    "PSI_TO_KPA",
    "apply_tire_result_to_roadload_request",
    "build_tire_application_inputs_from_vde_row",
    "build_tire_component_from_result",
    "compute_tire_smerf",
    "create_tire_from_form",
    "deactivate_tire_record",
    "delete_tire_record",
    "get_available_tires",
    "get_tire_by_code",
    "get_tire_by_id",
    "preview_tire_roadload_for_vde",
    "preview_tire_roadload_from_row",
    "resolve_tire_load_mass",
    "save_tire_roadload_to_vde",
    "summarize_tire_rr",
    "update_tire_from_form",
]
