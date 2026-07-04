from __future__ import annotations

from dataclasses import dataclass
import math


DEFAULT_DRIVER_MASS_KG = 75.0
WLTP_ADDITIONAL_MASS_KG = 25.0
WLTP_BASE_INCREMENT_KG = DEFAULT_DRIVER_MASS_KG + WLTP_ADDITIONAL_MASS_KG

WLTP_LOAD_FACTORS = {
    "M1": 0.15,
    "M2": 0.15,
    "N1": 0.28,
    "N2": 0.28,
}


def to_float_or_none(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def _to_float(value, default=None):
    out = to_float_or_none(value)
    return default if out is None else out


def normalize_wltp_category(category: object) -> str | None:
    if category is None:
        return None
    if isinstance(category, (int, float)) and not isinstance(category, bool):
        if float(category) == 1.0:
            return "M1"
    value = str(category).strip().upper()
    if value in {"M1", "1"}:
        return "M1"
    if value in {"M2", "N1", "N2"}:
        return value
    return None


def derive_laden_mass_kg(mass_kg: float | None, payload_kg: float | None) -> float | None:
    if mass_kg is None or payload_kg is None:
        return None
    return float(mass_kg) + float(payload_kg)


def get_wltp_light_duty_scope_warning(
    *,
    category: str | None,
    reference_mass_kg: float | None,
) -> str | None:
    normalized_category = normalize_wltp_category(category)
    if normalized_category not in {"M1", "M2", "N1", "N2"}:
        return "Unknown WLTP category."

    reference_mass = to_float_or_none(reference_mass_kg)
    if reference_mass is None:
        return None

    if reference_mass <= 2610:
        return None

    if reference_mass <= 2840:
        return (
            "Reference mass is above 2610 kg and up to 2840 kg. "
            "This may require a regulatory extension/manufacturer request."
        )

    return (
        "Reference mass is above 2840 kg. "
        "Do not treat this as standard light-duty WLTP scope."
    )


@dataclass
class WltpTestMassResult:
    mass_kg: float | None
    payload_kg: float | None
    options_kg: float
    laden_mass_kg: float | None
    wltp_category: str | None
    load_factor: float | None
    test_mass_low_kg: float | None
    test_mass_high_kg: float | None
    available_load_low_kg: float | None
    available_load_high_kg: float | None
    reference_mass_kg: float | None
    light_duty_scope_warning: str | None
    warnings: list[str]


def compute_wltp_test_masses(
    *,
    mass_kg: float | None,
    payload_kg: float | None,
    options_kg: float | None = 0.0,
    wltp_category: object = "M1",
    clamp_negative_available_load: bool = True,
) -> WltpTestMassResult:
    warnings: list[str] = []
    mass = to_float_or_none(mass_kg)
    payload = to_float_or_none(payload_kg)
    options = to_float_or_none(options_kg)
    if options is None:
        options = 0.0

    normalized_category = normalize_wltp_category(wltp_category)
    load_factor = WLTP_LOAD_FACTORS.get(normalized_category) if normalized_category is not None else None
    laden_mass_kg = derive_laden_mass_kg(mass, payload)
    reference_mass_kg = mass + WLTP_BASE_INCREMENT_KG if mass is not None else None
    light_duty_scope_warning = get_wltp_light_duty_scope_warning(
        category=normalized_category,
        reference_mass_kg=reference_mass_kg,
    )

    if mass is None or payload is None:
        warnings.append("WLTP test mass requires both mass_kg and payload_kg.")
        return WltpTestMassResult(
            mass_kg=mass,
            payload_kg=payload,
            options_kg=float(options),
            laden_mass_kg=laden_mass_kg,
            wltp_category=normalized_category,
            load_factor=load_factor,
            test_mass_low_kg=None,
            test_mass_high_kg=None,
            available_load_low_kg=None,
            available_load_high_kg=None,
            reference_mass_kg=reference_mass_kg,
            light_duty_scope_warning=light_duty_scope_warning,
            warnings=warnings,
        )

    if load_factor is None:
        warnings.append("WLTP category is unknown; expected M1, M2, N1, or N2.")
        return WltpTestMassResult(
            mass_kg=mass,
            payload_kg=payload,
            options_kg=float(options),
            laden_mass_kg=laden_mass_kg,
            wltp_category=normalized_category,
            load_factor=None,
            test_mass_low_kg=None,
            test_mass_high_kg=None,
            available_load_low_kg=None,
            available_load_high_kg=None,
            reference_mass_kg=reference_mass_kg,
            light_duty_scope_warning=light_duty_scope_warning,
            warnings=warnings,
        )

    tml_base = mass + WLTP_BASE_INCREMENT_KG
    tmh_base = mass + options + WLTP_BASE_INCREMENT_KG
    available_low = laden_mass_kg - tml_base
    available_high = laden_mass_kg - tmh_base

    if available_low < 0 and clamp_negative_available_load:
        warnings.append("WLTP available load for TML was negative and was clamped to zero.")
        available_low = 0.0
    if available_high < 0 and clamp_negative_available_load:
        warnings.append("WLTP available load for TMH was negative and was clamped to zero.")
        available_high = 0.0

    test_mass_low_kg = tml_base + load_factor * available_low
    test_mass_high_kg = tmh_base + load_factor * available_high

    return WltpTestMassResult(
        mass_kg=mass,
        payload_kg=payload,
        options_kg=float(options),
        laden_mass_kg=laden_mass_kg,
        wltp_category=normalized_category,
        load_factor=load_factor,
        test_mass_low_kg=float(test_mass_low_kg),
        test_mass_high_kg=float(test_mass_high_kg),
        available_load_low_kg=float(available_low),
        available_load_high_kg=float(available_high),
        reference_mass_kg=float(reference_mass_kg) if reference_mass_kg is not None else None,
        light_duty_scope_warning=light_duty_scope_warning,
        warnings=warnings,
    )


def resolve_test_mass_kg(
    *,
    basis: str | None,
    mass_kg: float | None,
    options_kg: float | None = 0.0,
    test_mass_low_kg: float | None = None,
    test_mass_high_kg: float | None = None,
    inertia_class: float | None = None,
    manual_test_mass_kg: float | None = None,
) -> tuple[float | None, str | None, list[str]]:
    warnings: list[str] = []
    normalized_basis = str(basis).strip().upper() if basis else None
    mass = to_float_or_none(mass_kg)
    options = to_float_or_none(options_kg)
    if options is None:
        options = 0.0
    test_mass_low = to_float_or_none(test_mass_low_kg)
    test_mass_high = to_float_or_none(test_mass_high_kg)
    inertia = to_float_or_none(inertia_class)
    manual = to_float_or_none(manual_test_mass_kg)

    if normalized_basis == "WLTP_TML":
        if test_mass_low is None:
            warnings.append("WLTP_TML was selected, but test_mass_low_kg is unavailable.")
            return None, "WLTP_TML", warnings
        return float(test_mass_low), "WLTP_TML", warnings

    if normalized_basis == "WLTP_TMH":
        if test_mass_high is None:
            warnings.append("WLTP_TMH was selected, but test_mass_high_kg is unavailable.")
            return None, "WLTP_TMH", warnings
        return float(test_mass_high), "WLTP_TMH", warnings

    if normalized_basis == "CURB":
        if mass is None:
            warnings.append("CURB was selected, but mass_kg is unavailable.")
            return None, "CURB", warnings
        return float(mass), "CURB", warnings

    if normalized_basis == "CURB_PLUS_DRIVER":
        if mass is None:
            warnings.append("CURB_PLUS_DRIVER was selected, but mass_kg is unavailable.")
            return None, "CURB_PLUS_DRIVER", warnings
        return float(mass + options + DEFAULT_DRIVER_MASS_KG), "CURB_PLUS_DRIVER", warnings

    if normalized_basis == "EPA_INERTIA_CLASS":
        if inertia is None:
            warnings.append("EPA_INERTIA_CLASS was selected, but inertia_class is unavailable.")
            return None, "EPA_INERTIA_CLASS", warnings
        return float(inertia), "EPA_INERTIA_CLASS", warnings

    if normalized_basis in {"PHYSICAL_TEST_MASS", "CUSTOM"}:
        if manual is None:
            warnings.append(f"{normalized_basis} was selected, but manual_test_mass_kg is unavailable.")
            return None, normalized_basis, warnings
        return float(manual), normalized_basis, warnings

    if normalized_basis == "CURB_FALLBACK":
        if mass is None:
            warnings.append("CURB_FALLBACK was selected, but mass_kg is unavailable.")
            return None, "CURB_FALLBACK", warnings
        return float(mass), "CURB_FALLBACK", warnings

    if manual is not None:
        return float(manual), "CUSTOM", warnings

    if mass is not None:
        warnings.append("Test mass basis was missing; falling back to mass_kg.")
        return float(mass), "CURB_FALLBACK", warnings

    warnings.append("Could not resolve test mass because both basis and mass_kg are unavailable.")
    return None, None, warnings


def compute_wltp_test_mass(mro_kg, options_kg=0.0, tpmlm_kg=None, category=1):
    mro = _to_float(mro_kg)
    tpmlm = _to_float(tpmlm_kg)
    opts = _to_float(options_kg, 0.0)
    if mro is None or tpmlm is None:
        return None
    normalized_category = normalize_wltp_category(category) or "M1"
    load_factor = WLTP_LOAD_FACTORS.get(normalized_category)
    if load_factor is None:
        return None
    max_load = tpmlm - mro - WLTP_ADDITIONAL_MASS_KG - opts
    if max_load < 0:
        max_load = 0.0
    tm = (mro + opts) + WLTP_ADDITIONAL_MASS_KG + load_factor * max_load
    return float(tm)


def compute_mro_from_stda(stda_kg, *, includes_driver=False, driver_mass_kg=DEFAULT_DRIVER_MASS_KG):
    mass = _to_float(stda_kg)
    if mass is None:
        return None
    if not includes_driver:
        mass += float(driver_mass_kg)
    return float(mass)


def inertia_class_from_mass(mass_kg: float) -> float | None:
    if mass_kg is None:
        return None
    steps = [
        (None, 346, 454), (346, 402, 510), (402, 459, 567), (459, 516, 624),
        (516, 573, 680), (573, 629, 737), (629, 686, 794), (686, 743, 850),
        (743, 799, 907), (799, 856, 964), (856, 913, 1021), (913, 969, 1077),
        (969, 1026, 1134), (1026, 1083, 1191), (1083, 1140, 1247), (1140, 1196, 1304),
        (1196, 1253, 1361), (1253, 1310, 1417), (1310, 1366, 1474), (1366, 1423, 1531),
        (1423, 1480, 1588), (1480, 1536, 1644), (1536, 1593, 1701), (1593, 1650, 1758),
        (1650, 1735, 1814), (1735, 1848, 1928), (1848, 1962, 2041), (1962, 2075, 2155),
        (2075, 2189, 2268), (2189, 2302, 2381), (2302, 2416, 2495), (2416, 2643, 2722),
        (2643, 2869, 2948), (2869, 3096, 3175), (3096, 3323, 3402), (3323, 3777, 3856),
        (3777, None, 4082),
    ]
    for lo, hi, cls in steps:
        if lo is None and mass_kg <= hi:
            return float(cls)
        if hi is None and mass_kg > lo:
            return float(cls)
        if lo is not None and hi is not None and (mass_kg > lo) and (mass_kg <= hi):
            return float(cls)
    return None


def autoresolve_test_mass(row_like: dict) -> dict:
    data = dict(row_like or {})
    leg = str(data.get("legislation") or "").strip().upper()

    if leg == "WLTP":
        result = compute_wltp_test_masses(
            mass_kg=to_float_or_none(data.get("mass_kg")),
            payload_kg=to_float_or_none(data.get("payload_kg")),
            options_kg=to_float_or_none(data.get("options_kg")),
            wltp_category=data.get("wltp_category"),
        )
        if result.test_mass_low_kg is not None:
            data["test_mass_low_kg"] = result.test_mass_low_kg
        if result.test_mass_high_kg is not None:
            data["test_mass_high_kg"] = result.test_mass_high_kg

        basis = data.get("test_mass_basis")
        manual_test_mass = data.get("test_mass_kg")
        if basis or manual_test_mass not in (None, ""):
            resolved_mass, resolved_basis, _warnings = resolve_test_mass_kg(
                basis=basis,
                mass_kg=data.get("mass_kg"),
                options_kg=data.get("options_kg"),
                test_mass_low_kg=data.get("test_mass_low_kg"),
                test_mass_high_kg=data.get("test_mass_high_kg"),
                inertia_class=data.get("inertia_class"),
                manual_test_mass_kg=manual_test_mass,
            )
            if resolved_mass is not None:
                data["test_mass_kg"] = resolved_mass
            if resolved_basis:
                data["test_mass_basis"] = resolved_basis
        return data

    if leg == "EPA":
        mass = _to_float(data.get("mass_kg"))
        if mass is not None:
            data["inertia_class"] = inertia_class_from_mass(mass)
        return data

    return data


__all__ = [
    "DEFAULT_DRIVER_MASS_KG",
    "WLTP_ADDITIONAL_MASS_KG",
    "WLTP_BASE_INCREMENT_KG",
    "WLTP_LOAD_FACTORS",
    "WltpTestMassResult",
    "autoresolve_test_mass",
    "compute_mro_from_stda",
    "compute_wltp_test_mass",
    "compute_wltp_test_masses",
    "derive_laden_mass_kg",
    "get_wltp_light_duty_scope_warning",
    "inertia_class_from_mass",
    "normalize_wltp_category",
    "resolve_test_mass_kg",
    "to_float_or_none",
]
