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
CONSTANT_RRC_MODE = "CONSTANT_RRC"
POWER_LAW_RRC_MODE = "POWER_LAW"


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
    Calculate SAE/J2452-style EPA SMERF and reference RRC.

    The official EcoDrive tire methodology separates:

        SMERF_EPA = 0.55 * F_FTP + 0.45 * F_HWFET
        RRC_ref [N/kN] = SMERF_EPA [N] * 1000 / Z_ref [N]

    ``pressure_kpa`` and ``load_n`` are the values used directly by the
    spreadsheet-style SAE equation. They must already be in the same units
    expected by the coefficients.
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
        "smerf": f_combined,
        "smerf_force_n": f_combined,
        "pressure_kpa": pressure_kpa,
        "load_n": load_n,
        "city_v_factor": city_v_factor,
        "city_v2_factor": city_v2_factor,
        "hwy_v_factor": hwy_v_factor,
        "hwy_v2_factor": hwy_v2_factor,
        "city_weight": city_weight,
        "hwy_weight": hwy_weight,
    }


def _first_present(data: dict | None, *keys: str):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _positive_model_float(value, field_name: str, *, default=None, required: bool = False):
    out = _to_float(value, field_name, default=default, required=required)
    if out is None:
        return None
    if out <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return out


def _non_negative_model_float(value, field_name: str, *, default=None, required: bool = False):
    out = _to_float(value, field_name, default=default, required=required)
    if out is None:
        return None
    if out < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return out


def _standard_family(tire: dict) -> str:
    return str(_dict_get(tire, "standard_family", "") or "").strip().upper()


def _reference_rrc_n_per_kn(tire: dict) -> float | None:
    return _to_float(
        _first_present(tire, "rr_n_per_kn", "iso_rrc_n_per_kn", "iso_corrected_rrc_n_per_kn"),
        "tire.rr_n_per_kn",
        default=None,
    )


def _reference_pressure_kpa(tire: dict) -> float | None:
    return _to_float(
        _first_present(tire, "sae_reference_pressure_kpa", "reference_pressure_kpa", "iso_test_pressure_kpa"),
        "tire.reference_pressure_kpa",
        default=None,
    )


def _reference_load_n(tire: dict) -> float | None:
    return _to_float(
        _first_present(tire, "sae_reference_load_n", "reference_load_n", "iso_test_load_n"),
        "tire.reference_load_n",
        default=None,
    )


def _pressure_exponent(tire: dict) -> float | None:
    return _to_float(
        _first_present(tire, "sae_alpha_pressure", "alpha_pressure", "sae_alpha"),
        "tire.alpha_pressure",
        default=None,
    )


def _load_exponent(tire: dict) -> float | None:
    return _to_float(
        _first_present(tire, "sae_beta_load", "beta_load", "sae_beta"),
        "tire.beta_load",
        default=None,
    )


def _rrc_adjustment_mode(tire: dict) -> str:
    mode = str(_dict_get(tire, "rrc_adjustment_mode", "") or "").strip().upper()
    if mode in {CONSTANT_RRC_MODE, POWER_LAW_RRC_MODE}:
        return mode
    if _standard_family(tire) == "SAE" and _reference_load_n(tire) is not None and _load_exponent(tire) is not None:
        return POWER_LAW_RRC_MODE
    return CONSTANT_RRC_MODE


def adjust_rrc_to_operating_condition(
    *,
    rrc_ref_n_per_kn: float,
    load_real_n: float,
    load_ref_n: float | None = None,
    pressure_real_kpa: float | None = None,
    pressure_ref_kpa: float | None = None,
    pressure_exponent: float | None = None,
    load_exponent: float | None = None,
    mode: str = CONSTANT_RRC_MODE,
) -> float:
    """
    Convert reference tire RRC to the operating load/pressure condition.

    Official power-law correction:

        RRC_real = RRC_ref
            * (P_real / P_ref) ** alpha_pressure
            * (Z_real / Z_ref) ** (beta_load - 1)

    Constant mode is the ISO/simple behavior: RRC_real = RRC_ref.
    """
    mode = str(mode or CONSTANT_RRC_MODE).strip().upper()
    rrc_ref_n_per_kn = _non_negative_model_float(rrc_ref_n_per_kn, "rrc_ref_n_per_kn", required=True)
    load_real_n = _positive_model_float(load_real_n, "load_real_n", required=True)

    if mode == CONSTANT_RRC_MODE:
        return rrc_ref_n_per_kn
    if mode != POWER_LAW_RRC_MODE:
        raise ValueError(f"Unsupported RRC adjustment mode: {mode!r}")

    load_ref_n = _positive_model_float(load_ref_n, "load_ref_n", required=True)
    load_exponent = _to_float(load_exponent, "load_exponent", required=True)
    factor = (load_real_n / load_ref_n) ** (load_exponent - 1.0)

    if pressure_exponent is not None:
        pressure_real_kpa = _positive_model_float(pressure_real_kpa, "pressure_real_kpa", required=True)
        pressure_ref_kpa = _positive_model_float(pressure_ref_kpa, "pressure_ref_kpa", required=True)
        pressure_exponent = _to_float(pressure_exponent, "pressure_exponent", required=True)
        factor *= (pressure_real_kpa / pressure_ref_kpa) ** pressure_exponent

    return rrc_ref_n_per_kn * factor


def _applied_rrc_for_tire(tire: dict, *, single_tire_load_n: float, pressure_kpa: float | None) -> Dict[str, float | str | None]:
    rrc_ref = _non_negative_model_float(_reference_rrc_n_per_kn(tire), "tire.rr_n_per_kn", required=True)
    mode = _rrc_adjustment_mode(tire)
    pressure_exponent = _pressure_exponent(tire)
    pressure_ref_kpa = _reference_pressure_kpa(tire)
    load_exponent = _load_exponent(tire)
    load_ref_n = _reference_load_n(tire)

    if mode == POWER_LAW_RRC_MODE and pressure_exponent is not None and pressure_ref_kpa is None:
        mode = CONSTANT_RRC_MODE
    if mode == POWER_LAW_RRC_MODE and (load_ref_n is None or load_exponent is None):
        mode = CONSTANT_RRC_MODE

    rrc_real = adjust_rrc_to_operating_condition(
        rrc_ref_n_per_kn=rrc_ref,
        load_real_n=single_tire_load_n,
        load_ref_n=load_ref_n,
        pressure_real_kpa=pressure_kpa,
        pressure_ref_kpa=pressure_ref_kpa,
        pressure_exponent=pressure_exponent if pressure_ref_kpa is not None else None,
        load_exponent=load_exponent,
        mode=mode,
    )
    force_n = rrc_real * single_tire_load_n / 1000.0
    return {
        "mode": mode,
        "rrc_ref_n_per_kn": rrc_ref,
        "rrc_n_per_kn": rrc_real,
        "single_tire_load_n": single_tire_load_n,
        "single_tire_force_n": force_n,
        "reference_load_n": load_ref_n,
        "reference_pressure_kpa": pressure_ref_kpa,
        "load_exponent": load_exponent,
        "pressure_exponent": pressure_exponent,
        "pressure_kpa": pressure_kpa,
    }


def calculate_applied_rrc_by_axle(front_tire: dict, rear_tire: dict, inputs: dict) -> Dict[str, Any]:
    """
    Calculate applied vehicle-equivalent RRC using front/rear tire loads.

    This is separate from the SAE ABC force model. It answers the VDE question:
    which tire RRC goes to the vehicle calculation after pressure/load correction?
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
    front = _applied_rrc_for_tire(
        front_tire,
        single_tire_load_n=loads["front_single_tire_load_n"],
        pressure_kpa=_dict_get(inputs, "front_pressure_kpa"),
    )
    rear = _applied_rrc_for_tire(
        rear_tire,
        single_tire_load_n=loads["rear_single_tire_load_n"],
        pressure_kpa=_dict_get(inputs, "rear_pressure_kpa"),
    )
    front_axle_force_n = front["single_tire_force_n"] * 2.0
    rear_axle_force_n = rear["single_tire_force_n"] * 2.0
    vehicle_force_n = front_axle_force_n + rear_axle_force_n
    vehicle_rrc_n_per_kn = vehicle_force_n * 1000.0 / loads["total_load_n"]

    return {
        "loads": loads,
        "front": front,
        "rear": rear,
        "front_rrc_n_per_kn": front["rrc_n_per_kn"],
        "rear_rrc_n_per_kn": rear["rrc_n_per_kn"],
        "front_single_tire_load_n": loads["front_single_tire_load_n"],
        "rear_single_tire_load_n": loads["rear_single_tire_load_n"],
        "front_single_tire_force_n": front["single_tire_force_n"],
        "rear_single_tire_force_n": rear["single_tire_force_n"],
        "front_axle_force_n": front_axle_force_n,
        "rear_axle_force_n": rear_axle_force_n,
        "vehicle_force_n": vehicle_force_n,
        "vehicle_rrc_n_per_kn": vehicle_rrc_n_per_kn,
    }


def calculate_rrc_n_per_kn_from_mean_force_lbf(
    mean_force_lbf: float,
    vehicle_weight_lbf: float,
    *,
    tire_count: int = 4,
) -> float:
    """
    Convert per-tire mean rolling force in lbf to vehicle RRC in N/kN.

    N/kN is numerically equal to the common RRC x1000 display:
        rrc_n_per_kn = total_tire_force_lbf / vehicle_weight_lbf * 1000
    """
    mean_force_lbf = _to_float(mean_force_lbf, "mean_force_lbf", required=True)
    vehicle_weight_lbf = _to_float(vehicle_weight_lbf, "vehicle_weight_lbf", required=True)
    tire_count = int(tire_count)
    if mean_force_lbf < 0:
        raise ValueError("mean_force_lbf must be non-negative")
    if vehicle_weight_lbf <= 0:
        raise ValueError("vehicle_weight_lbf must be greater than zero")
    if tire_count <= 0:
        raise ValueError("tire_count must be greater than zero")
    return (mean_force_lbf * tire_count / vehicle_weight_lbf) * 1000.0


def calculate_mean_force_lbf_from_rrc_n_per_kn(
    rrc_n_per_kn: float,
    vehicle_weight_lbf: float,
    *,
    tire_count: int = 4,
) -> float:
    """
    Convert vehicle RRC in N/kN / x1000 back to per-tire mean force in lbf.
    """
    rrc_n_per_kn = _to_float(rrc_n_per_kn, "rrc_n_per_kn", required=True)
    vehicle_weight_lbf = _to_float(vehicle_weight_lbf, "vehicle_weight_lbf", required=True)
    tire_count = int(tire_count)
    if rrc_n_per_kn < 0:
        raise ValueError("rrc_n_per_kn must be non-negative")
    if vehicle_weight_lbf <= 0:
        raise ValueError("vehicle_weight_lbf must be greater than zero")
    if tire_count <= 0:
        raise ValueError("tire_count must be greater than zero")
    return (rrc_n_per_kn / 1000.0) * vehicle_weight_lbf / tire_count


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
    applied_rrc = calculate_applied_rrc_by_axle(front_tire=front_tire, rear_tire=rear_tire, inputs=inputs)

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
        "applied_rrc": applied_rrc,
        "applied_rr_n_per_kn": applied_rrc["vehicle_rrc_n_per_kn"],
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
    "CONSTANT_RRC_MODE",
    "G_MPS2",
    "KPA_PER_PSI",
    "MPH_PER_KPH",
    "N_PER_KGF",
    "N_PER_LBF",
    "POWER_LAW_RRC_MODE",
    "adjust_rrc_to_operating_condition",
    "apply_tire_improvement",
    "build_tire_component",
    "calculate_applied_rrc_by_axle",
    "calculate_axle_loads",
    "calculate_axle_tire_abc_from_single",
    "calculate_iso_tire_abc_for_single_tire",
    "calculate_mean_force_lbf_from_rrc_n_per_kn",
    "calculate_rrc_n_per_kn_from_mean_force_lbf",
    "calculate_sae_smerf_rr_n_per_kn",
    "calculate_sae_tire_abc_for_single_tire",
    "calculate_single_tire_loads",
    "calculate_vehicle_tire_abc",
    "combine_front_rear_tire_abc",
]
