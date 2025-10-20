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
from typing import Dict, Optional, Union, Any

import numpy as np
import pandas as pd
import streamlit as st
from src.vde_core.utils import cycle_kpis
import json




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
    return mapping.get(leg)

def use_standard_cycle(leg):
    fname = default_cycle_for_legislation(leg)
    try:
        df = load_cycle_csv(fname)
        #st.session_state["cycle_df"] = df
        #st.session_state["cycle_source"] = f"standard:{leg}"
        st.success(f"Using default **{leg}** cycle: `{fname}.csv`")
        k = cycle_kpis(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{k['duration_s']:.0f} s")
        c2.metric("Distance", f"{k['distance_km']:.2f} km")
        c3.metric("Avg Speed", f"{k['v_mean_kmh']:.1f} km/h")
        c4.metric("Samples", f"{k['n_points']}")
        return df
    except Exception as e:
        st.warning(f"Default cycle for **{leg}** not found. {e}")
        st.info("Please upload a custom cycle below.")
        return None


def cycle_summary(df_cycle: pd.DataFrame):
    if df_cycle is None or df_cycle.empty:
        return "No cycle loaded.", ""
    # simple KPIs
    t = df_cycle.iloc[:,0].astype(float)
    v = df_cycle.iloc[:,1].astype(float)
    dur = t.iloc[-1] - t.iloc[0]
    dist =  np.trapz(v, t)  # crude integral v*dt (meters)
    dist_km = dist / 1000.0
    vavg = v.mean() * 3.6  # m/s -> km/h
    return f"Duration: {dur:.0f} s • Distance: {dist_km:.2f} km • v̄: {vavg:.1f} km/h", f"{dist_km:.3f}"


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

# services.py
def _norm_phase(x: str) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def epa_city_hwy_from_phase(
    df_or_groups: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    A: float, B: float, C: float, mass: float
) -> Dict[str, Optional[float]]:
    # groups
    groups_raw = split_by_phase(df_or_groups) if isinstance(df_or_groups, pd.DataFrame) else dict(df_or_groups)

    # normalizar chaves: "bag 1" -> "bag1", "highway"/"hwy" -> "hwfet"
    def norm(s: str) -> str:
        s = (s or "").lower().strip()
        s = s.replace("-", " ").replace("_", " ")
        s = s.replace("bag 1", "bag1").replace("bag 2", "bag2")
        s = s.replace("highway", "hwfet").replace("hwy", "hwfet")
        return s.replace(" ", "")
    groups = {norm(k): v for k, v in groups_raw.items()}

    # helper de busca
    def get_any(*keys):
        for k in keys:
            k = norm(k)
            if k in groups:
                return groups[k]
            for gk in groups.keys():      # aceita prefixos/sufixos (ex.: ftp75bag1)
                if k in gk:
                    return groups[gk]
        return None

    # ETW consistente
    etw_kg = inertia_class_from_mass(mass)

    # City = bag1 + bag2
    urb_E_J = 0.0
    urb_S_m = 0.0
    for phase_name in ("bag1", "bag 1"):
        g = get_any(phase_name)
        if g is not None:
            r = compute_vde_net(g, A, B, C, etw_kg)
            urb_E_J += r["MJ_total"] * 1e6
            urb_S_m += r["km"] * 1000.0
    for phase_name in ("bag2", "bag 2"):
        g = get_any(phase_name)
        if g is not None:
            r = compute_vde_net(g, A, B, C, etw_kg)
            urb_E_J += r["MJ_total"] * 1e6
            urb_S_m += r["km"] * 1000.0

    urb_MJ = urb_km = urb_MJ_km = None
    if urb_S_m > 0:
        urb_MJ = urb_E_J / 1e6
        urb_km = urb_S_m / 1000.0
        urb_MJ_km = urb_MJ / max(urb_km, 1e-9)

    # Highway = HWFET
    hw_MJ = hw_km = hw_MJ_km = None
    hw = get_any("hwfet", "hwy", "highway")
    if hw is not None:
        rH = compute_vde_net(hw, A, B, C, etw_kg)  # usa ETW também aqui
        hw_MJ, hw_km, hw_MJ_km = rH["MJ_total"], rH["km"], rH["MJ_km"]

    # Combined
    net_comb_MJ_km = (
        0.55 * urb_MJ_km + 0.45 * hw_MJ_km if (urb_MJ_km is not None and hw_MJ_km is not None)
        else (urb_MJ_km or hw_MJ_km)
    )

    return {
        "urb_MJ": urb_MJ, "urb_km": urb_km, "urb_MJ_km": urb_MJ_km,
        "hw_MJ": hw_MJ, "hw_km": hw_km, "hw_MJ_km": hw_MJ_km,
        "net_comb_MJ_km": net_comb_MJ_km,
    }

def wltp_phases_from_phase2(
    df_or_groups: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    A: float, B: float, C: float, mass: float
) -> Dict[str, Optional[float]]:

    def _norm(lbl: str) -> str:
        t = str(lbl).strip().lower().replace("-", " ").replace("_", " ")
        t = " ".join(t.split())
        return "xhigh" if t == "extra high" else t

    # 1) agrupa
    if isinstance(df_or_groups, pd.DataFrame):
        if "phase" not in df_or_groups.columns:
            return {}
        groups_raw = { _norm(k): v.copy()
                       for k, v in df_or_groups.groupby(df_or_groups["phase"].map(_norm)) }
    else:
        groups_raw = { _norm(k): v.copy() for k, v in dict(df_or_groups).items() }

    want = ["low", "mid", "high", "xhigh"]
    groups = {k: groups_raw[k] for k in want if k in groups_raw}

    out: Dict[str, Optional[float]] = {}
    E_sum = 0.0
    S_sum = 0.0

    name_to_col = {
        "low":   "vde_low_mj_per_km",
        "mid":   "vde_mid_mj_per_km",
        "high":  "vde_high_mj_per_km",
        "xhigh": "vde_extra_high_mj_per_km",
    }

    for k in want:
        out_col = name_to_col[k]
        if (k not in groups) or groups[k].empty:
            out[f"{k}_MJ"] = out[f"{k}_km"] = out[f"{k}_MJ_km"] = None
            out[out_col] = None
            continue

        g = groups[k].copy()

        # v em m/s → v_mps
        if "v_mps" not in g.columns:
            if "v" in g.columns:
                g["v_mps"] = pd.to_numeric(g["v"], errors="coerce")
            else:
                out[f"{k}_MJ"] = out[f"{k}_km"] = out[f"{k}_MJ_km"] = None
                out[out_col] = None
                continue

        # tempo e dt
        tcol = "t" if "t" in g.columns else ("time_s" if "time_s" in g.columns else None)
        if tcol is None:
            out[f"{k}_MJ"] = out[f"{k}_km"] = out[f"{k}_MJ_km"] = None
            out[out_col] = None
            continue

        g[tcol] = pd.to_numeric(g[tcol], errors="coerce")
        g = g.dropna(subset=[tcol, "v_mps"]).sort_values(tcol).reset_index(drop=True)
        g["dt"] = g[tcol].diff().fillna(0.0).clip(lower=0.0)

        # checagem de distância
        km_chk = float((g["v_mps"] * g["dt"]).sum() / 1000.0)
        if not (km_chk > 0):
            out[f"{k}_MJ"] = out[f"{k}_km"] = out[f"{k}_MJ_km"] = None
            out[out_col] = None
            continue

        # integra com a MESMA assinatura da EPA
        r = compute_vde_net(g, float(A), float(B), float(C), float(mass))

        MJ_tot = r.get("MJ_total"); km = r.get("km"); MJ_km = r.get("MJ_km")
        out[f"{k}_MJ"]     = float(MJ_tot) if isinstance(MJ_tot, (int, float)) else None
        out[f"{k}_km"]     = float(km)     if isinstance(km,     (int, float)) else None
        out[f"{k}_MJ_km"]  = float(MJ_km)  if isinstance(MJ_km,  (int, float)) else None
        out[out_col]       = out[f"{k}_MJ_km"]

        if isinstance(MJ_tot, (int, float)) and isinstance(km, (int, float)) and km > 0:
            E_sum += float(MJ_tot); S_sum += float(km)

    out["vde_net_mj_per_km"] = (E_sum / S_sum) if S_sum > 0 else None
    return out


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
        "extra_high": "vde_extra_high_mj_per_km",   # já cobre "extra_high"
        "extrahigh": "vde_extra_high_mj_per_km", }   # sem underscore
        # graças ao _norm_phase novo, "extra high" e "extra-high" viram "extra_high"
    groups = split_by_phase(df_or_groups) if isinstance(df_or_groups, pd.DataFrame) else dict(df_or_groups)
    # debug rápido (comente se não quiser)
    # st.write("WLTP phases found:", list(groups.keys()))

    tm = None
    try:
        mro_kg = compute_mro_from_stda(mass, includes_driver=False)
        tm = compute_wltp_test_mass(mro_kg)
    except Exception:
        tm = None

    if tm is None:
        tm = float(mass)  # fallback para a massa passada

    E_sum = 0.0
    S_sum = 0.0
    for key_norm, colname in phase_map.items():
        if key_norm in groups:
            r = compute_vde_net(groups[key_norm], A, B, C, tm)
            out[colname] = r["MJ_km"]
            E_sum += r["MJ_total"]
            S_sum += r["km"]

    if S_sum > 0:
        out["vde_net_mj_per_km"] = E_sum / max(S_sum, 1e-9)

    return out

def apply_coastdown_deltas(A, B, C, mass_kg,
                           delta_rr_N=0.0,
                           delta_brake_N=0.0,
                           delta_parasitics_N=0.0,
                           delta_aero_Npkph2=0.0,
                           delta_mass_kg=0.0,
                           crr1_frac_at_120kph=0.0):
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
        (2302, 2416, 2495), 
        (2416, 2643, 2722), 
        (2643, 2869, 2948),
        (2869, 3096, 3175), 
        (3096, 3323, 3402), 
        (3323, 3777, 3856),
        (3777, None, 4082),
    ]
    for lo, hi, cls in steps:
        if lo is None and mass_kg <= hi:         return float(cls)
        if hi is None  and mass_kg > lo:         return float(cls)
        if lo is not None and hi is not None and (mass_kg > lo) and (mass_kg <= hi):
            return float(cls)
    return None





G = 9.80665   # m/s²
RHO = 1.2     # kg/m³
# QA tolerances for recomposition (A,B,C)
TOL_A = 5.0       # N
TOL_B = 0.10      # N/kph
TOL_C = 1e-1      # N/kph²

DEFAULTS_REQUIRED_COLS = [
    "category", "electrification", "transmission_type",
    "cdA_default_m2", "rrc_N_per_kN", "crr1_frac_at_120kph",
]

def load_vde_defaults(path: str | Path) -> pd.DataFrame:
    """
    Load defaults CSV with priors by (category, electrification, transmission_type).
    Required columns: DEFAULTS_REQUIRED_COLS.
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in DEFAULTS_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in defaults CSV: {missing}")
    return df

def estimate_aux_from_coastdown(
    *,
    A_N: float,
    B_N_per_kph: float,
    C_N_per_kph2: float,
    mass_kg: float,
    category: str,
    electrification: str,
    transmission_type: str,
    cdA_override_m2: Optional[float] = None,
    defaults_df: Optional[pd.DataFrame] = None,
    defaults_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Decompose measured coastdown (NET) into RR, Aero and Parasitic components.

    NET convention here:
      F(v_kph) = A + B*v + C*v^2  (v in km/h)
      - Aero uses C (expected ~ all of measured C)
      - RR uses defaults (RRC @ 0 kph and slope fraction at 120 kph)
      - Parasitic (brakes/bearings) is the remainder in A and B
      - Transmission is NOT part of NET here.

    Args:
      A_N, B_N_per_kph, C_N_per_kph2: measured coastdown coefficients (B may be < 0)
      mass_kg: test mass
      category, electrification, transmission_type: keys to pick defaults row
      cdA_override_m2: if provided, uses this instead of defaults' cdA
      defaults_df: preloaded defaults; if None, will load from defaults_path
      defaults_path: path to CSV if defaults_df is None

    Returns:
      dict with:
        rr_alpha_N, rr_beta_Npkph, aero_C_coef_Npkph2,
        parasitic_A_coef_N, parasitic_B_coef_Npkph, parasitic_C_coef_Npkph2,
        cdA_used_m2,
        dA, dB, dC, check_ok (QA flags), rl_source
    """
    if defaults_df is None:
        if not defaults_path:
            raise ValueError("Provide defaults_df or defaults_path.")
        defaults_df = load_vde_defaults(defaults_path)

    # pick row (strict match; if not found, try fallback on category only)
    df = defaults_df
    m = df[
        (df["category"].astype(str).str.upper() == str(category).upper()) &
        (df["electrification"].astype(str).str.upper() == str(electrification).upper()) &
        (df["transmission_type"].astype(str).str.upper() == str(transmission_type).upper())
    ]
    if m.empty:
        m = df[df["category"].astype(str).str.upper() == str(category).upper()]
        if m.empty:
            raise ValueError("No defaults found for this (category/electrification/transmission).")

    row = m.iloc[0]
    cdA_default = float(row["cdA_default_m2"])
    rrc_N_per_kN = float(row["rrc_N_per_kN"])
    crr1_frac_120 = float(row["crr1_frac_at_120kph"])

    # numeric inputs (B can be negative)
    A = float(A_N); B = float(B_N_per_kph); C = float(C_N_per_kph2)
    if mass_kg is None or mass_kg <= 0:
        raise ValueError("mass_kg must be > 0")

    # choose cdA: override > default
    cdA = float(cdA_override_m2) if cdA_override_m2 is not None else cdA_default

    # ---- NET blocks ----
    # Rolling resistance (rrc in N/kN * total load in kN)
    load_kN = mass_kg * G / 1000.0
    A_rr = rrc_N_per_kN * load_kN
    B_rr = A_rr * (crr1_frac_120 / 120.0)
    C_rr = 0.0

    # Aerodynamics (C in N/kph²)
    C_aero = 0.5 * RHO * cdA * (1/3.6)**2

    # Parasitic (brake/bearings) = remainder in A and B; C_par expected ~ 0
    A_par = max(0.0, A - A_rr)
    B_par = max(0.0, B - B_rr)
    C_par = max(0.0, C - C_aero)

    # ---- QA (recomposition) ----
    dA = (A_rr + A_par) - A
    dB = (B_rr + B_par) - B
    dC = (C_aero + C_rr + C_par) - C
    check_ok = (abs(dA) <= TOL_A) and (abs(dB) <= TOL_B) and (abs(dC) <= TOL_C)

    return {
        "rr_alpha_N": A_rr,
        "rr_beta_Npkph": B_rr,
        "aero_C_coef_Npkph2": C_aero,
        "parasitic_A_coef_N": A_par,
        "parasitic_B_coef_Npkph": B_par,
        "parasitic_C_coef_Npkph2": C_par,
        "cdA_used_m2": cdA,
        "dA": dA, "dB": dB, "dC": dC,
        "check_ok": bool(check_ok),
        "rl_source": "measured_decomposed_NET",
    }

# src/vde_core/services.py

# ---- Constantes simples (ajuste aos seus padrões, se quiser puxar de outro lugar) ----
LHV_MJ_PER_L = {
    "Gasoline": 32.0,  # ajuste se usar E0/E22 etc.
    "E10":      31.2,
    "E22":      30.0,
    "E100":     21.2,
    "Diesel":   35.8,
    "Other":    32.0,
}

GCO2_PER_L = {
    "Gasoline": 2310.0,   # gCO2 por litro aproximado
    "E10":      2270.0,
    "E22":      2200.0,
    "E100":        0.0,   # fósseis ~0; use LCA se quiser
    "Diesel":   2640.0,
    "Other":    2310.0,
}

MJ_TO_Wh = 277.7777777778  # 1 MJ = 277.777... Wh


def _get_vde_row(vde_id: int) -> dict | None:
    # Import local para quebrar o ciclo (só roda na HORA da chamada)
    from src.vde_core.db import fetchone as _fetchone
    return _fetchone("SELECT * FROM vde_db WHERE id=?;", (vde_id,))


def compute_ice_fuel_from_vde(
    vde_id: int,
    fuel_type: str,
    eta_pt: float,
    lhv_mj_per_l: float | None = None,
    electrification: str = "ICE",   # "ICE" | "MHEV" | "HEV" | "PHEV"
    uf_phev: float | None = None,   # 0..1 (se quiser ponderar)
    driveline_eff: float | None = None,         # opcional: para PHEV (parte elétrica)
    grid_gco2_per_kwh: float | None = None,     # opcional: para PHEV (parte elétrica)
) -> dict:
    """
    Converte VDE_NET -> consumo fóssil (L/100km, km/L) e CO2.
    Se electrification == 'PHEV' e uf_phev for dado, computa um 'blended' simples:
      - parte ICE ponderada por (1-UF)
      - parte elétrica (se driveline_eff e grid forem fornecidos) ponderada por UF
    """
    row = _get_vde_row(vde_id)
    assert row and ("vde_net_mj_per_km" in row), "VDE row sem vde_net_mj_per_km"
    vde_mj_per_km = float(row["vde_net_mj_per_km"])

    # Defaults de LHV/CO2 por litro
    lhv = float(lhv_mj_per_l) if lhv_mj_per_l else float(LHV_MJ_PER_L.get(fuel_type, 32.0))
    gco2_per_l = float(GCO2_PER_L.get(fuel_type, 2310.0))

    # Parte ICE (convencional): VDE/eta -> MJ/km no eixo-motor -> L/km
    mj_pk_ice = vde_mj_per_km / max(eta_pt, 1e-6)
    L_per_km_ice = mj_pk_ice / max(lhv, 1e-6)
    L_per_100km_ice = 100.0 * L_per_km_ice
    km_per_L_ice = 100.0 / max(L_per_100km_ice, 1e-9)
    gco2_per_km_ice = L_per_km_ice * gco2_per_l

    # Resultado default: tudo ICE/MxHEV/HEV tratado como ICE (η_pt já incorpora híbrido)
    L_per_100 = L_per_100km_ice
    km_per_L  = km_per_L_ice
    gco2_km   = gco2_per_km_ice
    Wh_per_km = None

    # PHEV (ponderado), se UF fornecido
    if str(electrification).upper() == "PHEV" and uf_phev is not None:
        uf = max(0.0, min(1.0, float(uf_phev)))
        # parte elétrica (se params presentes)
        if driveline_eff and grid_gco2_per_kwh is not None:
            energy_Wh_per_km_elec = (vde_mj_per_km / max(driveline_eff,1e-6)) * MJ_TO_Wh
            gco2_km_elec = (energy_Wh_per_km_elec / 1000.0) * float(grid_gco2_per_kwh)
        else:
            energy_Wh_per_km_elec = 0.0
            gco2_km_elec = 0.0

        # blend
        L_per_km_blend = (1.0 - uf) * L_per_km_ice
        L_per_100 = 100.0 * L_per_km_blend
        km_per_L  = 100.0 / max(L_per_100, 1e-9) if L_per_100 > 0 else None
        gco2_km   = (1.0 - uf) * gco2_per_km_ice + uf * gco2_km_elec
        Wh_per_km = uf * energy_Wh_per_km_elec  # opcional: devolve energia elétrica associada

    assumptions = {
        "fuel_type": fuel_type,
        "eta_pt": eta_pt,
        "lhv_mj_per_l": lhv,
        "gco2_per_l": gco2_per_l,
        "electrification": electrification,
        "uf_phev": uf_phev,
        "driveline_eff": driveline_eff,
        "grid_gco2_per_kwh": grid_gco2_per_kwh,
        "vde_net_mj_per_km": vde_mj_per_km,
    }

    return {
        "cycle": row.get("legislation", "auto"),
        "fuel_l_per_100km": L_per_100,
        "fuel_km_per_l": km_per_L,
        "energy_Wh_per_km": Wh_per_km,  # para PHEV (parte elétrica ponderada), opcional
        "gco2_per_km": gco2_km,
        "assumptions_json": json.dumps(assumptions),
    }


def compute_bev_from_vde(
    vde_id: int,
    driveline_eff: float,
    grid_gco2_per_kwh: float = 0.0,
) -> dict:
    """
    Converte VDE_NET -> energia de bateria por km (Wh/km) e CO2 por grid.
      Wh/km = (VDE_NET [MJ/km] / driveline_eff) * 277.777...
      gCO2/km = (Wh/km / 1000) * grid_gCO2/kWh
    """
    row = _get_vde_row(vde_id)
    assert row and ("vde_net_mj_per_km" in row), "VDE row sem vde_net_mj_per_km"
    vde_mj_per_km = float(row["vde_net_mj_per_km"])

    Wh_per_km = (vde_mj_per_km / max(driveline_eff, 1e-6)) * MJ_TO_Wh
    gco2_km = (Wh_per_km / 1000.0) * float(grid_gco2_per_kwh)

    assumptions = {
        "driveline_eff": driveline_eff,
        "grid_gco2_per_kwh": grid_gco2_per_kwh,
        "vde_net_mj_per_km": vde_mj_per_km,
    }

    return {
        "cycle": row.get("legislation", "auto"),
        "energy_Wh_per_km": Wh_per_km,
        "gco2_per_km": gco2_km,
        "assumptions_json": json.dumps(assumptions),
    }
