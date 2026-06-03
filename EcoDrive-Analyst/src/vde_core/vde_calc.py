from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


_SIMPLE_TRANS_FACTORS = {
    "AT": 0.06,
    "DCT": 0.04,
    "CVT": 0.05,
    "MT": 0.03,
}


def compute_vde_net(
    df: pd.DataFrame,
    A_N: float,
    B_N_per_kph: float,
    C_N_per_kph2: float,
    mass_kg: float,
) -> Dict[str, float]:
    if not {"t", "v"} <= set(df.columns):
        raise ValueError("cycle df must have columns: t (s), v (m/s)")

    t = df["t"].to_numpy(float)
    v = df["v"].to_numpy(float)

    mask = np.isfinite(t) & np.isfinite(v)
    t, v = t[mask], v[mask]
    if t.size < 2:
        raise ValueError("cycle has too few points")

    if np.any(np.diff(t) <= 0):
        order = np.argsort(t)
        t, v = t[order], v[order]

    v_kph = v * 3.6
    F_road = A_N + B_N_per_kph * v_kph + C_N_per_kph2 * (v_kph**2)

    a = np.gradient(v, t)
    P = (F_road + mass_kg * a) * v
    P_pos = np.clip(P, 0.0, None)

    E_J = np.trapezoid(P_pos, t)
    s_m = np.trapezoid(v, t)

    km = s_m / 1000.0
    MJ_total = E_J / 1e6
    MJ_km = MJ_total / max(km, 1e-9)
    Wh_km = (E_J / 3600.0) / max(km, 1e-9)

    return {"MJ_km": MJ_km, "Wh_km": Wh_km, "km": km, "MJ_total": MJ_total}


def compute_vde_net_mj_per_km(cycle_df: pd.DataFrame, A: float, B: float, C: float, mass_kg: float) -> Dict[str, float]:
    result = compute_vde_net(cycle_df, A, B, C, mass_kg)
    return {"MJ_km": result["MJ_km"], "Wh_km": result["Wh_km"], "km": result["km"]}


def apply_coastdown_deltas(
    A,
    B,
    C,
    mass_kg,
    delta_rr_N=0.0,
    delta_brake_N=0.0,
    delta_parasitics_N=0.0,
    delta_aero_Npkph2=0.0,
    delta_mass_kg=0.0,
    crr1_frac_at_120kph=0.0,
):
    dA_rr = float(delta_rr_N or 0.0)
    dB_rr = dA_rr * float(crr1_frac_at_120kph or 0.0) / 120.0
    dA_br = float(delta_brake_N or 0.0)
    dA_pa = float(delta_parasitics_N or 0.0)
    dC_ae = float(delta_aero_Npkph2 or 0.0)
    dmass = float(delta_mass_kg or 0.0)
    A1 = float(A) + dA_rr + dA_br + dA_pa
    B1 = float(B) + dB_rr
    C1 = float(C) + dC_ae
    mass_kg1 = float(mass_kg) + dmass
    return A1, B1, C1, mass_kg1


def vde_total_simple(vde_net_mjkm: float, transmission_type: Optional[str]) -> float:
    tt = (transmission_type or "").upper()
    factor = _SIMPLE_TRANS_FACTORS.get(tt, 0.05)
    return vde_net_mjkm * (1.0 + factor)


__all__ = [
    "apply_coastdown_deltas",
    "compute_vde_net",
    "compute_vde_net_mj_per_km",
    "vde_total_simple",
]
