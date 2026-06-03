from __future__ import annotations

from typing import Dict, Optional, Union

import pandas as pd

from src.vde_core.test_mass import compute_mro_from_stda, compute_wltp_test_mass, inertia_class_from_mass
from src.vde_core.vde_calc import compute_vde_net


def _norm_phase(x: str) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def split_by_phase(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "phase" not in df.columns:
        return {}
    groups: Dict[str, pd.DataFrame] = {}
    for ph, dfg in df.groupby(df["phase"].map(_norm_phase)):
        groups[ph] = dfg.loc[:, ["t", "v"]].dropna().sort_values("t")
    return groups


def epa_city_hwy_from_phase(
    df_or_groups: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    A: float,
    B: float,
    C: float,
    mass: float,
) -> Dict[str, Optional[float]]:
    groups_raw = split_by_phase(df_or_groups) if isinstance(df_or_groups, pd.DataFrame) else dict(df_or_groups)

    def norm(s: str) -> str:
        s = (s or "").lower().strip()
        s = s.replace("-", " ").replace("_", " ")
        s = s.replace("bag 1", "bag1").replace("bag 2", "bag2")
        s = s.replace("highway", "hwfet").replace("hwy", "hwfet")
        return s.replace(" ", "")

    groups = {norm(k): v for k, v in groups_raw.items()}

    def get_any(*keys):
        for key in keys:
            key = norm(key)
            if key in groups:
                return groups[key]
            for group_key in groups.keys():
                if key in group_key:
                    return groups[group_key]
        return None

    etw_kg = inertia_class_from_mass(mass)

    urb_E_J = 0.0
    urb_S_m = 0.0
    for phase_name in ("bag1", "bag 1", "bag2", "bag 2"):
        group = get_any(phase_name)
        if group is not None:
            result = compute_vde_net(group, A, B, C, etw_kg)
            urb_E_J += result["MJ_total"] * 1e6
            urb_S_m += result["km"] * 1000.0

    urb_MJ = urb_km = urb_MJ_km = None
    if urb_S_m > 0:
        urb_MJ = urb_E_J / 1e6
        urb_km = urb_S_m / 1000.0
        urb_MJ_km = urb_MJ / max(urb_km, 1e-9)

    hw_MJ = hw_km = hw_MJ_km = None
    hw = get_any("hwfet", "hwy", "highway")
    if hw is not None:
        result = compute_vde_net(hw, A, B, C, etw_kg)
        hw_MJ, hw_km, hw_MJ_km = result["MJ_total"], result["km"], result["MJ_km"]

    net_comb_MJ_km = (
        0.55 * urb_MJ_km + 0.45 * hw_MJ_km
        if (urb_MJ_km is not None and hw_MJ_km is not None)
        else (urb_MJ_km or hw_MJ_km)
    )

    return {
        "urb_MJ": urb_MJ,
        "urb_km": urb_km,
        "urb_MJ_km": urb_MJ_km,
        "hw_MJ": hw_MJ,
        "hw_km": hw_km,
        "hw_MJ_km": hw_MJ_km,
        "net_comb_MJ_km": net_comb_MJ_km,
    }


def wltp_phases_from_phase(
    df_or_groups: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    A: float,
    B: float,
    C: float,
    mass: float,
) -> Dict[str, Optional[float]]:
    groups = split_by_phase(df_or_groups) if isinstance(df_or_groups, pd.DataFrame) else dict(df_or_groups)
    out: Dict[str, Optional[float]] = {}

    phase_map = {
        "low": "vde_low_mj_per_km",
        "mid": "vde_mid_mj_per_km",
        "high": "vde_high_mj_per_km",
        "xhigh": "vde_extra_high_mj_per_km",
        "extra_high": "vde_extra_high_mj_per_km",
        "extrahigh": "vde_extra_high_mj_per_km",
    }

    tm = None
    try:
        mro_kg = compute_mro_from_stda(mass, includes_driver=False)
        tm = compute_wltp_test_mass(mro_kg)
    except Exception:
        tm = None

    if tm is None:
        tm = float(mass)

    E_sum = 0.0
    S_sum = 0.0
    for key_norm, colname in phase_map.items():
        if key_norm in groups:
            result = compute_vde_net(groups[key_norm], A, B, C, tm)
            out[colname] = result["MJ_km"]
            E_sum += result["MJ_total"]
            S_sum += result["km"]

    if S_sum > 0:
        out["vde_net_mj_per_km"] = E_sum / max(S_sum, 1e-9)

    return out


__all__ = [
    "epa_city_hwy_from_phase",
    "split_by_phase",
    "wltp_phases_from_phase",
]
