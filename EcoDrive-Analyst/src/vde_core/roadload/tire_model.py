"""
Pure tire roadload model helpers for EcoDrive.

This module implements the MVP tire contribution logic agreed for the
Tire Roadload DB sprint:

- load is resolved at vehicle -> axle -> single-tire level
- SAE tire model uses:
      Frr = P^alpha * Z^beta * (a + bV + cV^2)
  therefore:
      A = P^alpha * Z^beta * a
      B = P^alpha * Z^beta * b
      C = P^alpha * Z^beta * c
- ISO MVP uses:
      A = rr_n_per_kn * load_kN
      B = 0
      C = 0
- front and rear are calculated independently, then summed
- tire improvement applies to the final combined tire ABC

The module intentionally avoids Streamlit, SQLite, or page concerns.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from .models import RoadLoadComponent


G_MPS2 = 9.80665
KPA_PER_PSI = 6.89475729
N_PER_LBF = 4.4482216152605
N_PER_KGF = G_MPS2
MPH_PER_KPH = 0.621371192237334
DEFAULT_CITY_V_FACTOR = 34.04267
DEFAULT_CITY_V2_FACTOR = 1818.112
DEFAULT_HWY_V_FACTOR = 77.67619
DEFAULT_HWY_V2_FACTOR = 6297.445
DEFAULT_CITY_WEIGHT = 0.55
DEFAULT_HWY_WEIGHT = 0.45


def _to_float(value, field_name: str, *, default=None, required: bool = False):
    if value in (None, ""):
        if required:
            raise ValueError(f"Missing required tire model field: {field_name}")
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}")
    if math.isnan(out) or math.isinf(out):
        raise ValueError(f"Invalid finite numeric value for {field_name}: {value!r}")
    return out


def _dict_get(data: dict | None, key: str, default=None):
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def _unit_value(data: dict, primary: str, fallback: str, default: str) -> str:
    value = _dict_get(data, primary)
    if value in (None, ""):
        value = _dict_get(data, fallback, default)
    return str(value or default).strip().lower()


def _pressure_kpa_to_model_unit(pressure_kpa: float, unit: str) -> float:
    if unit in {"psi", "psig"}:
        return pressure_kpa / KPA_PER_PSI
    if unit in {"kpa", "kilopascal", "kilopascals"}:
        return pressure_kpa
    raise ValueError(f"Unsupported SAE pressure unit: {unit!r}")


def _load_n_to_model_unit(load_n: float, unit: str) -> float:
    if unit in {"lbf", "lb", "lbs", "pound-force", "pounds-force"}:
        return load_n / N_PER_LBF
    if unit in {"kg", "kgf", "kilogram-force", "kilograms-force"}:
        return load_n / N_PER_KGF
    if unit in {"n", "newton", "newtons"}:
        return load_n
    raise ValueError(f"Unsupported SAE load unit: {unit!r}")


def _force_to_n_factor(unit: str) -> float:
    if unit in {"lbf", "lb", "lbs", "pound-force", "pounds-force"}:
        return N_PER_LBF
    if unit in {"kg", "kgf", "kilogram-force", "kilograms-force"}:
        return N_PER_KGF
    if unit in {"n", "newton", "newtons"}:
        return 1.0
    raise ValueError(f"Unsupported SAE force unit: {unit!r}")


def _speed_from_kph_factor(unit: str) -> float:
    if unit in {"mph", "mi/h"}:
        return MPH_PER_KPH
    if unit in {"kph", "km/h", "kmh"}:
        return 1.0
    raise ValueError(f"Unsupported SAE speed unit: {unit!r}")


def calculate_axle_loads(mass_kg: float, front_weight_distribution_pct: float, g_mps2: float = G_MPS2) -> Dict[str, float]:
    """
    Resolve front/rear axle loads in N from vehicle mass and front weight split.
    """
    mass_kg = _to_float(mass_kg, "mass_kg", required=True)
    front_weight_distribution_pct = _to_float(
        front_weight_distribution_pct,
        "front_weight_distribution_pct",
        required=True,
    )
    g_mps2 = _to_float(g_mps2, "g_mps2", required=True)

    if mass_kg <= 0:
        raise ValueError("mass_kg must be greater than zero")
    if not (0.0 <= front_weight_distribution_pct <= 100.0):
        raise ValueError("front_weight_distribution_pct must be between 0 and 100")

    front_frac = front_weight_distribution_pct / 100.0
    rear_frac = 1.0 - front_frac
    total_load_n = mass_kg * g_mps2
    front_axle_load_n = total_load_n * front_frac
    rear_axle_load_n = total_load_n * rear_frac

    return {
        "mass_kg": mass_kg,
        "g_mps2": g_mps2,
        "front_weight_distribution_pct": front_weight_distribution_pct,
        "rear_weight_distribution_pct": rear_frac * 100.0,
        "total_load_n": total_load_n,
        "front_axle_load_n": front_axle_load_n,
        "rear_axle_load_n": rear_axle_load_n,
    }


def calculate_single_tire_loads(mass_kg: float, front_weight_distribution_pct: float, g_mps2: float = G_MPS2) -> Dict[str, float]:
    """
    Resolve single-tire loads assuming two tires per axle.
    """
    axle = calculate_axle_loads(mass_kg, front_weight_distribution_pct, g_mps2=g_mps2)
    axle["front_single_tire_load_n"] = axle["front_axle_load_n"] / 2.0
    axle["rear_single_tire_load_n"] = axle["rear_axle_load_n"] / 2.0
    axle["front_single_tire_load_kn"] = axle["front_single_tire_load_n"] / 1000.0
    axle["rear_single_tire_load_kn"] = axle["rear_single_tire_load_n"] / 1000.0
    return axle


def calculate_sae_tire_abc_for_single_tire(tire: dict, single_tire_load_n: float, pressure_kpa: float) -> Dict[str, float]:
    """
    Calculate single-tire SAE equivalent ABC.

    Vehicle inputs arrive in SI-ish internal units: pressure in kPa, load in N,
    target ABC in N/kph/kph^2. The SAE coefficients may be stored in their
    native test units, commonly psi/lbf/mph/lbf, so this function converts the
    inputs into coefficient units and converts the output back to EcoDrive ABC.
    """
    if not isinstance(tire, dict):
        raise TypeError("tire must be a dict-like record")

    pressure_kpa = _to_float(pressure_kpa, "pressure_kpa", required=True)
    single_tire_load_n = _to_float(single_tire_load_n, "single_tire_load_n", required=True)
    if pressure_kpa <= 0:
        raise ValueError("pressure_kpa must be greater than zero")
    if single_tire_load_n <= 0:
        raise ValueError("single_tire_load_n must be greater than zero")

    a = _to_float(_dict_get(tire, "sae_a"), "tire.sae_a", required=True)
    b = _to_float(_dict_get(tire, "sae_b"), "tire.sae_b", required=True)
    c = _to_float(_dict_get(tire, "sae_c"), "tire.sae_c", required=True)
    alpha = _to_float(_dict_get(tire, "sae_alpha"), "tire.sae_alpha", default=0.0)
    beta = _to_float(_dict_get(tire, "sae_beta"), "tire.sae_beta", default=0.0)

    pressure_unit = _unit_value(tire, "sae_pressure_unit", "pressure_unit", "kPa")
    load_unit = _unit_value(tire, "sae_load_unit", "load_unit", "N")
    speed_unit = _unit_value(tire, "sae_speed_unit", "speed_unit", "kph")
    force_unit = _unit_value(tire, "sae_force_unit", "force_unit", "N")

    pressure_model = _pressure_kpa_to_model_unit(pressure_kpa, pressure_unit)
    load_model = _load_n_to_model_unit(single_tire_load_n, load_unit)
    force_to_n = _force_to_n_factor(force_unit)
    speed_from_kph = _speed_from_kph_factor(speed_unit)

    scale = (pressure_model ** alpha) * (load_model ** beta)
    raw_A = scale * a
    raw_B = scale * b
    raw_C = scale * c

    return {
        "A": raw_A * force_to_n,
        "B": raw_B * force_to_n * speed_from_kph,
        "C": raw_C * force_to_n * speed_from_kph * speed_from_kph,
        "raw_A": raw_A,
        "raw_B": raw_B,
        "raw_C": raw_C,
        "scale_factor": scale,
        "pressure_model_value": pressure_model,
        "load_model_value": load_model,
        "pressure_unit": pressure_unit,
        "load_unit": load_unit,
        "speed_unit": speed_unit,
        "force_unit": force_unit,
        "pressure_kpa": pressure_kpa,
        "single_tire_load_n": single_tire_load_n,
        "single_tire_load_kn": single_tire_load_n / 1000.0,
    }


def calculate_sae_smerf_rr_n_per_kn(
    *,
    alpha: float,
    beta: float,
    a: float,
    b: float,
    c: float,
    pressure_kpa: float,
    load_n: float,
    city_v_factor: float = DEFAULT_CITY_V_FACTOR,
    city_v2_factor: float = DEFAULT_CITY_V2_FACTOR,
    hwy_v_factor: float = DEFAULT_HWY_V_FACTOR,
    hwy_v2_factor: float = DEFAULT_HWY_V2_FACTOR,
    city_weight: float = DEFAULT_CITY_WEIGHT,
    hwy_weight: float = DEFAULT_HWY_WEIGHT,
) -> Dict[str, float]:
    """
    Calculate SAE/J2452-style SMERF as an RR value in N/kN.

    The spreadsheet logic uses the SAE ABC model to compute a weighted city/highway
    rolling-resistance force, then normalizes it by tire load:

        rr_n_per_kn = F_combined * 1000 / Z

    ``pressure_kpa`` and ``load_n`` are the values used directly by the
    spreadsheet-style SMERF equation. In the current EcoDrive UI they are
    resolved as kPa and N before calling this helper.
    """
    alpha = _to_float(alpha, "sae_alpha", default=0.0)
    beta = _to_float(beta, "sae_beta", default=0.0)
    a = _to_float(a, "sae_a", required=True)
    b = _to_float(b, "sae_b", required=True)
    c = _to_float(c, "sae_c", required=True)
    pressure_kpa = _to_float(pressure_kpa, "pressure_kpa", required=True)
    load_n = _to_float(load_n, "load_n", required=True)
    city_v_factor = _to_float(city_v_factor, "city_v_factor", required=True)
    city_v2_factor = _to_float(city_v2_factor, "city_v2_factor", required=True)
    hwy_v_factor = _to_float(hwy_v_factor, "hwy_v_factor", required=True)
    hwy_v2_factor = _to_float(hwy_v2_factor, "hwy_v2_factor", required=True)
    city_weight = _to_float(city_weight, "city_weight", required=True)
    hwy_weight = _to_float(hwy_weight, "hwy_weight", required=True)

    if pressure_kpa <= 0:
        raise ValueError("pressure_kpa must be greater than zero")
    if load_n <= 0:
        raise ValueError("load_n must be greater than zero")

    scale = (pressure_kpa ** alpha) * (load_n ** beta)
    A = scale * a
    B = scale * b
    C = scale * c
    f_city = A + B * city_v_factor + C * city_v2_factor
    f_hwy = A + B * hwy_v_factor + C * hwy_v2_factor
    f_combined = city_weight * f_city + hwy_weight * f_hwy
    rr_n_per_kn = f_combined * 1000.0 / load_n

    return {
        "A": A,
        "B": B,
        "C": C,
        "scale_factor": scale,
        "F_city": f_city,
        "F_hwy": f_hwy,
        "F_combined": f_combined,
        "rr_n_per_kn": rr_n_per_kn,
        "smerf": rr_n_per_kn,
        "pressure_kpa": pressure_kpa,
        "load_n": load_n,
        "city_v_factor": city_v_factor,
        "city_v2_factor": city_v2_factor,
        "hwy_v_factor": hwy_v_factor,
        "hwy_v2_factor": hwy_v2_factor,
        "city_weight": city_weight,
        "hwy_weight": hwy_weight,
    }


def calculate_iso_tire_abc_for_single_tire(tire: dict, single_tire_load_n: float) -> Dict[str, float]:
    """
    Calculate single-tire ISO MVP equivalent ABC.
    """
    if not isinstance(tire, dict):
        raise TypeError("tire must be a dict-like record")

    single_tire_load_n = _to_float(single_tire_load_n, "single_tire_load_n", required=True)
    if single_tire_load_n <= 0:
        raise ValueError("single_tire_load_n must be greater than zero")

    rr_n_per_kn = _dict_get(tire, "rr_n_per_kn")
    if rr_n_per_kn in (None, ""):
        rr_n_per_kn = _dict_get(tire, "iso_rrc_n_per_kn")
    rr_n_per_kn = _to_float(rr_n_per_kn, "tire.rr_n_per_kn", required=True)
    if rr_n_per_kn < 0:
        raise ValueError("tire.rr_n_per_kn must be non-negative")

    load_kn = single_tire_load_n / 1000.0
    return {
        "A": rr_n_per_kn * load_kn,
        "B": 0.0,
        "C": 0.0,
        "rr_n_per_kn": rr_n_per_kn,
        "single_tire_load_n": single_tire_load_n,
        "single_tire_load_kn": load_kn,
    }


def calculate_axle_tire_abc_from_single(single_tire_abc: dict, tire_count: int = 2) -> Dict[str, float]:
    """
    Convert single-tire ABC into axle ABC.
    """
    if not isinstance(single_tire_abc, dict):
        raise TypeError("single_tire_abc must be a dict")
    tire_count = int(tire_count)
    if tire_count <= 0:
        raise ValueError("tire_count must be greater than zero")

    return {
        "A": _to_float(single_tire_abc.get("A"), "single_tire_abc.A", required=True) * tire_count,
        "B": _to_float(single_tire_abc.get("B"), "single_tire_abc.B", required=True) * tire_count,
        "C": _to_float(single_tire_abc.get("C"), "single_tire_abc.C", required=True) * tire_count,
        "tire_count": tire_count,
    }


def combine_front_rear_tire_abc(front_axle_abc: dict, rear_axle_abc: dict) -> Dict[str, float]:
    """
    Sum front and rear axle tire contributions.
    """
    if not isinstance(front_axle_abc, dict) or not isinstance(rear_axle_abc, dict):
        raise TypeError("front_axle_abc and rear_axle_abc must be dicts")

    return {
        "A": _to_float(front_axle_abc.get("A"), "front_axle_abc.A", required=True)
        + _to_float(rear_axle_abc.get("A"), "rear_axle_abc.A", required=True),
        "B": _to_float(front_axle_abc.get("B"), "front_axle_abc.B", required=True)
        + _to_float(rear_axle_abc.get("B"), "rear_axle_abc.B", required=True),
        "C": _to_float(front_axle_abc.get("C"), "front_axle_abc.C", required=True)
        + _to_float(rear_axle_abc.get("C"), "rear_axle_abc.C", required=True),
    }


def apply_tire_improvement(tire_abc: dict, improvement_pct: float) -> Dict[str, float]:
    """
    Apply improvement to the final combined tire ABC.
    Positive values reduce ABC, negative values increase it.
    """
    if not isinstance(tire_abc, dict):
        raise TypeError("tire_abc must be a dict")

    improvement_pct = _to_float(improvement_pct, "improvement_pct", default=0.0)
    factor = 1.0 - (improvement_pct / 100.0)
    return {
        "A": _to_float(tire_abc.get("A"), "tire_abc.A", required=True) * factor,
        "B": _to_float(tire_abc.get("B"), "tire_abc.B", required=True) * factor,
        "C": _to_float(tire_abc.get("C"), "tire_abc.C", required=True) * factor,
        "improvement_pct": improvement_pct,
        "improvement_factor": factor,
    }


def calculate_vehicle_tire_abc(front_tire: dict, rear_tire: dict, inputs: dict) -> Dict[str, Any]:
    """
    Full MVP vehicle tire calculation.

    Expected inputs:
      - mass_kg
      - front_weight_distribution_pct
      - front_pressure_kpa (required for SAE front)
      - rear_pressure_kpa (required for SAE rear)
      - tire_improvement_pct
    """
    if not isinstance(front_tire, dict):
        raise TypeError("front_tire must be a dict-like record")
    if not isinstance(rear_tire, dict):
        raise TypeError("rear_tire must be a dict-like record")
    if not isinstance(inputs, dict):
        raise TypeError("inputs must be a dict")

    loads = calculate_single_tire_loads(
        mass_kg=_dict_get(inputs, "mass_kg"),
        front_weight_distribution_pct=_dict_get(inputs, "front_weight_distribution_pct"),
    )

    front_family = str(_dict_get(front_tire, "standard_family", "") or "").upper()
    rear_family = str(_dict_get(rear_tire, "standard_family", "") or "").upper()
    if front_family not in {"SAE", "ISO", "CUSTOM"}:
        raise ValueError(f"Unsupported front tire standard_family: {front_family!r}")
    if rear_family not in {"SAE", "ISO", "CUSTOM"}:
        raise ValueError(f"Unsupported rear tire standard_family: {rear_family!r}")

    if front_family == "SAE":
        front_single = calculate_sae_tire_abc_for_single_tire(
            tire=front_tire,
            single_tire_load_n=loads["front_single_tire_load_n"],
            pressure_kpa=_dict_get(inputs, "front_pressure_kpa"),
        )
    else:
        front_single = calculate_iso_tire_abc_for_single_tire(
            tire=front_tire,
            single_tire_load_n=loads["front_single_tire_load_n"],
        )

    if rear_family == "SAE":
        rear_single = calculate_sae_tire_abc_for_single_tire(
            tire=rear_tire,
            single_tire_load_n=loads["rear_single_tire_load_n"],
            pressure_kpa=_dict_get(inputs, "rear_pressure_kpa"),
        )
    else:
        rear_single = calculate_iso_tire_abc_for_single_tire(
            tire=rear_tire,
            single_tire_load_n=loads["rear_single_tire_load_n"],
        )

    front_axle = calculate_axle_tire_abc_from_single(front_single, tire_count=2)
    rear_axle = calculate_axle_tire_abc_from_single(rear_single, tire_count=2)
    total_base = combine_front_rear_tire_abc(front_axle, rear_axle)
    total_final = apply_tire_improvement(total_base, _dict_get(inputs, "tire_improvement_pct", 0.0))

    applied_rr_n_per_kn = _dict_get(front_tire, "rr_n_per_kn")
    if applied_rr_n_per_kn in (None, ""):
        applied_rr_n_per_kn = _dict_get(rear_tire, "rr_n_per_kn")

    return {
        "loads": loads,
        "front": {
            "standard_family": front_family,
            "single_tire_abc": front_single,
            "axle_abc": front_axle,
        },
        "rear": {
            "standard_family": rear_family,
            "single_tire_abc": rear_single,
            "axle_abc": rear_axle,
        },
        "total_base_abc": total_base,
        "total_final_abc": total_final,
        "tire_improvement_pct": _to_float(_dict_get(inputs, "tire_improvement_pct", 0.0), "tire_improvement_pct", default=0.0),
        "tire_load_mass_used_kg": loads["mass_kg"],
        "applied_rr_n_per_kn": _to_float(applied_rr_n_per_kn, "applied_rr_n_per_kn", default=None),
    }


def build_tire_component(name: str, tire_abc: dict, *, source: str = "tire_model", meta: dict | None = None) -> RoadLoadComponent:
    """
    Build a RoadLoadComponent from a tire ABC dict.
    """
    if not isinstance(tire_abc, dict):
        raise TypeError("tire_abc must be a dict")
    return RoadLoadComponent(
        name=name,
        A=_to_float(tire_abc.get("A"), "tire_abc.A", required=True),
        B=_to_float(tire_abc.get("B"), "tire_abc.B", required=True),
        C=_to_float(tire_abc.get("C"), "tire_abc.C", required=True),
        source=source,
        meta=dict(meta or {}),
    )


__all__ = [
    "DEFAULT_CITY_V_FACTOR",
    "DEFAULT_CITY_V2_FACTOR",
    "DEFAULT_CITY_WEIGHT",
    "DEFAULT_HWY_V_FACTOR",
    "DEFAULT_HWY_V2_FACTOR",
    "DEFAULT_HWY_WEIGHT",
    "G_MPS2",
    "KPA_PER_PSI",
    "MPH_PER_KPH",
    "N_PER_KGF",
    "N_PER_LBF",
    "apply_tire_improvement",
    "build_tire_component",
    "calculate_axle_loads",
    "calculate_axle_tire_abc_from_single",
    "calculate_iso_tire_abc_for_single_tire",
    "calculate_sae_smerf_rr_n_per_kn",
    "calculate_sae_tire_abc_for_single_tire",
    "calculate_single_tire_loads",
    "calculate_vehicle_tire_abc",
    "combine_front_rear_tire_abc",
]
