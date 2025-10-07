# src/vde_core/services.py
# -----------------------------------------------------------------------------
# Core services for EcoDrive-Analyst
# - Cycle I/O
# - VDE_NET computation (units-safe)
# - Phase aggregation for EPA & WLTP
# - Simple drivetrain loss model (for VDE_TOTAL fallback)
# - Small unit converters kept for backward-compatibility
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


# =============================================================================
# Cycle I/O
# =============================================================================

def default_cycle_for_legislation(leg: str) -> str:
    """
    Returns the default cycle filename (without .csv) for a given legislation.
    Use exact filenames you have under data/cycles/.
    """
    leg = (leg or "").upper()
    mapping = {
        "EPA": "FTP75_HWFET",            # you also have HWFET and FTP75_HWFET if needed
        "WLTP": "WLTP_Class3ab",
    }
    return mapping.get(leg, "FTP75")


def load_cycle_csv(name_no_ext: str) -> pd.DataFrame:
    """
    Loads a CSV from data/cycles/<name>.csv with columns:
      - required: t [s], v [m/s]
      - optional: phase (str) -> e.g. bag1, bag2, HWFET / low, mid, high, xhigh
    """
    p = Path("data/cycles") / f"{name_no_ext}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Cycle CSV not found: {p}")
    df = pd.read_csv(p)
    if not {"t", "v"} <= set(df.columns):
        raise ValueError(f"{p} must have columns: t, v (v in m/s)")
    # Normalize minimal dtypes
    df = df.copy()
    df["t"] = df["t"].astype(float)
    df["v"] = df["v"].astype(float)
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype(str)
    return df


# =============================================================================
# Core VDE (NET) computation
# =============================================================================

def compute_vde_net(df: pd.DataFrame, A_N: float, B_N_per_kph: float, C_N_per_kph2: float, mass_kg: float) -> Dict[str, float]:
    """
    Compute VDE on wheels (NET), integrating only positive tractive power.

    Inputs:
      df: columns t [s], v [m/s]
      A: N
      B: N/kph
      C: N/kph^2
      mass_kg: kg

    Returns:
      {"MJ_km": ..., "Wh_km": ..., "km": ..., "MJ_total": ...}
    """
    if not {"t", "v"} <= set(df.columns):
        raise ValueError("cycle df must have columns: t (s), v (m/s)")

    t = df["t"].to_numpy(float)
    v = df["v"].to_numpy(float)

    m = np.isfinite(t) & np.isfinite(v)
    t, v = t[m], v[m]
    if t.size < 2:
        raise ValueError("cycle has too few points")

    # ensure strictly increasing t
    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)
        t, v = t[idx], v[idx]

    v_kph = v * 3.6
    F_road = A_N + B_N_per_kph * v_kph + C_N_per_kph2 * (v_kph**2)

    a = np.gradient(v, t)                 # m/s²
    P = (F_road + mass_kg * a) * v        # W
    P_pos = np.clip(P, 0.0, None)         # only positive (tractive) power

    E_J = np.trapz(P_pos, t)              # J
    s_m = np.trapz(v, t)                  # m

    km = s_m / 1000.0
    MJ_total = E_J / 1e6
    MJ_km = MJ_total / max(km, 1e-9)
    Wh_km = (E_J / 3600.0) / max(km, 1e-9)

    return {"MJ_km": MJ_km, "Wh_km": Wh_km, "km": km, "MJ_total": MJ_total}


def compute_vde_net_mj_per_km(cycle_df: pd.DataFrame, A: float, B: float, C: float, mass_kg: float) -> Dict[str, float]:
    """
    Backward-compatible wrapper used by your Page 1:
    returns dict with keys: MJ_km, Wh_km, km
    """
    r = compute_vde_net(cycle_df, A, B, C, mass_kg)
    return {"MJ_km": r["MJ_km"], "Wh_km": r["Wh_km"], "km": r["km"]}


# =============================================================================
# Phase helpers & aggregation
# =============================================================================

def _norm_phase(x: str) -> str:
    return str(x).strip().lower()

def split_by_phase(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict phase -> df_phase (with t,v only), sorted by time.
    If no 'phase' column, returns {}.
    """
    if "phase" not in df.columns:
        return {}
    groups: Dict[str, pd.DataFrame] = {}
    for ph, dfg in df.groupby(df["phase"].map(_norm_phase)):
        dfg = dfg.loc[:, ["t", "v"]].dropna().sort_values("t")
        groups[ph] = dfg
    return groups


def epa_city_hwy_from_phase(
    df_or_groups: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    A: float, B: float, C: float, mass: float
) -> Dict[str, Optional[float]]:
    """
    EPA aggregation from phases:
      - City  = bag1 + bag2 (sum energy & distance)
      - Highway = HWFET
      - Combined NET (MJ/km) = 0.55*City + 0.45*Highway (if both exist)
    Returns keys:
      urb_MJ, urb_km, urb_MJ_km, hw_MJ, hw_km, hw_MJ_km, net_comb_MJ_km
    """
    groups = split_by_phase(df_or_groups) if isinstance(df_or_groups, pd.DataFrame) else dict(df_or_groups)

    # sum bag1 + bag2
    urb_E_J = 0.0
    urb_S_m = 0.0
    for key_alt in (("bag1", "1", "bag 1"), ("bag2", "2", "bag 2")):
        for k in key_alt:
            if k in groups:
                r = compute_vde_net(groups[k], A, B, C, mass)
                urb_E_J += r["MJ_total"] * 1e6
                urb_S_m += r["km"] * 1000.0
                break

    urb_MJ = urb_km = urb_MJ_km = None
    if urb_S_m > 0:
        urb_MJ = urb_E_J / 1e6
        urb_km = urb_S_m / 1000.0
        urb_MJ_km = urb_MJ / max(urb_km, 1e-9)

    hw_MJ = hw_km = hw_MJ_km = None
    for k in ("hwfet", "hwy", "highway"):
        if k in groups:
            r = compute_vde_net(groups[k], A, B, C, mass)
            hw_MJ, hw_km, hw_MJ_km = r["MJ_total"], r["km"], r["MJ_km"]
            break

    net_comb_MJ_km = None
    if (urb_MJ_km is not None) and (hw_MJ_km is not None):
        net_comb_MJ_km = 0.55 * urb_MJ_km + 0.45 * hw_MJ_km
    else:
        net_comb_MJ_km = urb_MJ_km  # fallback: only city available

    return {
        "urb_MJ": urb_MJ, "urb_km": urb_km, "urb_MJ_km": urb_MJ_km,
        "hw_MJ": hw_MJ, "hw_km": hw_km, "hw_MJ_km": hw_MJ_km,
        "net_comb_MJ_km": net_comb_MJ_km
    }


def wltp_phases_from_phase(
    df_or_groups: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    A: float, B: float, C: float, mass: float
) -> Dict[str, Optional[float]]:
    """
    Computes VDE_NET per WLTP phase (MJ/km) if present in 'phase':
      low, mid, high, xhigh (or extra_high)
    Also returns an overall NET (energy-weighted over available phases).
    """
    groups = split_by_phase(df_or_groups) if isinstance(df_or_groups, pd.DataFrame) else dict(df_or_groups)
    out: Dict[str, Optional[float]] = {}

    phase_map = {
        "low": "vde_low_mj_per_km",
        "mid": "vde_mid_mj_per_km",
        "high": "vde_high_mj_per_km",
        "xhigh": "vde_extra_high_mj_per_km",
        "extra_high": "vde_extra_high_mj_per_km",
    }

    E_sum = 0.0
    S_sum = 0.0
    for key_norm, colname in phase_map.items():
        if key_norm in groups:
            r = compute_vde_net(groups[key_norm], A, B, C, mass)
            out[colname] = r["MJ_km"]
            E_sum += r["MJ_total"]
            S_sum += r["km"]

    if S_sum > 0:
        out["vde_net_mj_per_km"] = E_sum / max(S_sum, 1e-9)

    return out


# =============================================================================
# Simple drivetrain loss model (for VDE_TOTAL fallback)
# =============================================================================

_SIMPLE_TRANS_FACTORS = {
    "AT": 0.06,   # Automatic
    "DCT": 0.04,  # Dual-Clutch
    "CVT": 0.05,  # Continuously Variable
    "MT": 0.03,   # Manual
}

def vde_total_simple(vde_net_mjkm: float, transmission_type: Optional[str]) -> float:
    """
    Simple multiplicative loss factor by transmission type.
    Used as fallback when no transmission details are provided.
    """
    tt = (transmission_type or "").upper()
    f = _SIMPLE_TRANS_FACTORS.get(tt, 0.05)
    return vde_net_mjkm * (1.0 + f)


# =============================================================================
# Small converters (kept for compatibility)
# =============================================================================

def mjkm_to_whkm(mj_per_km: float) -> float:
    return mj_per_km / 0.0036

def whkm_to_mjkm(wh_per_km: float) -> float:
    return wh_per_km * 0.0036

# =============================================================================
# Test Mass / Inertia Class helpers

def compute_wltp_test_mass(mro_kg, options_kg=0.0, tpmlm_kg=None, category=1):
    """
    WLTP Test Mass (TM), em kg.
    TM = (MRO + options) + 25 + x * (TPMLM - MRO - 25 - options)
    x = 0.15 (cat 1) ou 0.28 (cat 2)
    """
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



    return data
def _to_float(x, default=None):
    try:
        if x in (None, ""):
            return default
        return float(x)
    except Exception:
        return default
    
# ----------------- Resolver mínimo (WLTP/EPA) -----------------
def autoresolve_test_mass(row_like: dict) -> dict:
    """
    Mínimo necessário por legislação:
      - WLTP: calcula TM a partir de MRO (direto ou via StdA), TPMLM, options e category.
              Grava mass_kg = TM e inertia_class = TM.
      - EPA: se houver mass_kg, calcula inertia_class pelos degraus EPA.
    """
    d = dict(row_like or {})
    leg = (d.get("legislation") or "").strip().upper()

    # ---------- WLTP ----------
    if leg == "WLTP":
        # 1) obter MRO: preferir mro_kg; senão derivar de StdA (stda_kg)
        mro = d.get("mro_kg")
        if mro in (None, ""):
            # escolha o campo StdA que você usa no seu app;
            # aqui assumo 'stda_kg' (pode ser 'mass_kg' se você guardava StdA lá)
            stda = d.get("stda_kg")
            if stda is None:
                # fallback: alguns fluxos legados guardavam StdA em 'mass_kg'
                stda = d.get("mass_kg")
            mro = compute_mro_from_stda(stda, includes_driver=False)  # StdA + 75
            d["mro_kg"] = mro

        # 2) precisa de TPMLM (obrigatório) e, opcionalmente, options/category
        tpmlm = d.get("tpmlm_kg")
        opts  = d.get("options_kg", 0.0)
        cat   = d.get("wltp_category", 1)

        tm = compute_wltp_test_mass(mro, opts, tpmlm, cat)
        if tm is not None:
            d["inertia_class"] = tm
        return d

    # ---------- EPA ----------
    if leg == "EPA":
        m = _to_float(d.get("mass_kg"))
        if m is not None:
            d["inertia_class"] = inertia_class_from_mass(m)
        return d

    # ---------- Outras legislações: não mexe ----------
    return d

def compute_mro_from_stda(stda_kg, *, includes_driver=False, driver_mass_kg=75.0):
    """
    StdA (curb/kerb, sem motorista) -> MRO.
    Se já inclui motorista, passe includes_driver=True.
    """
    m = _to_float(stda_kg)
    if m is None:
        return None
    if not includes_driver:
        m += float(driver_mass_kg)
    return float(m)

def inertia_class_from_mass(mass_kg: float) -> float | None:
    """Classe de inércia (kg) para EPA, dada a massa (kg). Regra dos degraus."""
    if mass_kg is None:
        return None
    steps = [
        (None, 346, 454),
        (346, 402, 510),
        (402, 459, 567),
        (459, 516, 624),
        (516, 573, 680),
        (573, 629, 737),
        (629, 686, 794),
        (686, 743, 850),
        (743, 799, 907),
        (799, 856, 964),
        (856, 913, 1021),
        (913, 969, 1077),
        (969, 1026, 1134),
        (1026, 1083, 1191),
        (1083, 1140, 1247),
        (1140, 1196, 1304),
        (1196, 1253, 1361),
        (1253, 1310, 1417),
        (1310, 1366, 1474),
        (1366, 1423, 1531),
        (1423, 1480, 1588),
        (1480, 1536, 1644),
        (1536, 1593, 1701),
        (1593, 1650, 1758),
        (1650, 1735, 1814),
        (1735, 1848, 1928),
        (1848, 1962, 2041),
        (1962, 2075, 2155),
        (2075, 2189, 2268),
        (2189, 2302, 2381),
        (2302, None, 2495),
    ]
    for lo, hi, cls in steps:
        if lo is None and mass_kg <= hi:         return float(cls)
        if hi is None  and mass_kg > lo:         return float(cls)
        if lo is not None and hi is not None and (mass_kg > lo) and (mass_kg <= hi):
            return float(cls)
    return None