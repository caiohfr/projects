from __future__ import annotations


def _to_float(x, default=None):
    try:
        if x in (None, ""):
            return default
        return float(x)
    except Exception:
        return default


def compute_wltp_test_mass(mro_kg, options_kg=0.0, tpmlm_kg=None, category=1):
    mro = _to_float(mro_kg)
    tpmlm = _to_float(tpmlm_kg)
    opts = _to_float(options_kg, 0.0)
    if mro is None or tpmlm is None:
        return None
    try:
        cat = int(category)
    except Exception:
        cat = 1
    x = 0.15 if cat == 1 else 0.28
    max_load = tpmlm - mro - 25.0 - opts
    if max_load < 0:
        max_load = 0.0
    tm = (mro + opts) + 25.0 + x * max_load
    return float(tm)


def compute_mro_from_stda(stda_kg, *, includes_driver=False, driver_mass_kg=75.0):
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
    leg = (data.get("legislation") or "").strip().upper()

    if leg == "WLTP":
        mro = data.get("mro_kg")
        if mro in (None, ""):
            stda = data.get("stda_kg")
            if stda is None:
                stda = data.get("mass_kg")
            mro = compute_mro_from_stda(stda, includes_driver=False)
            data["mro_kg"] = mro

        tpmlm = data.get("tpmlm_kg")
        opts = data.get("options_kg", 0.0)
        cat = data.get("wltp_category", 1)
        tm = compute_wltp_test_mass(mro, opts, tpmlm, cat)
        if tm is not None:
            data["inertia_class"] = tm
        return data

    if leg == "EPA":
        mass = _to_float(data.get("mass_kg"))
        if mass is not None:
            data["inertia_class"] = inertia_class_from_mass(mass)
        return data

    return data


__all__ = [
    "autoresolve_test_mass",
    "compute_mro_from_stda",
    "compute_wltp_test_mass",
    "inertia_class_from_mass",
]
