import numpy as np
import pandas as pd
from .models import Scenario, RoadLoadABC
from .physics import compose_abc_from_blocks, f_road, power_patch,vde_from_trace
from pathlib import Path


def compute_abc(scn):
    if scn.mode == "BASELINE":
        return scn.roadload
    else:
        return compose_abc_from_blocks(scn)

def ensure_cycle_columns(df):
    """Normalize cycle to have t_s, v_kph, a_ms2."""
    out = df.copy()
    if "t" in out.columns and "v" in out.columns:
        out = out.rename(columns={"t":"t_s","v":"v_kph"})
        out["v_ms"] = out["v_kph"] * 1/ 3.6
    elif "t_s" in out.columns and "v_kph" in out.columns:
        pass
    else:
        raise ValueError("Cycle must have (t,v) in m/s or (t_s,v_kph).")

    t = out["t_s"].to_numpy()
    v_ms = out["v_kph"].to_numpy() / 3.6
    a = np.gradient(v_ms, t)
    out["a_ms2"] = a
    return out[["t_s","v_kph","a_ms2"]]

def compute_vde_from_scenario(cycle_df, scn):
    """Return dict with ABC, total energy, distance, VDE (Wh/km)."""
    cycle = ensure_cycle_columns(cycle_df)
    abc = compute_abc(scn)
    F = f_road(cycle["v_kph"].to_numpy(), abc)
    P = power_patch(F, scn.inertia.mass_test_kg, cycle["v_kph"].to_numpy(), cycle["a_ms2"].to_numpy())
    dt = np.diff(cycle["t_s"].to_numpy(), prepend=cycle["t_s"].iloc[0])
    E_Wh = float((P * dt).sum() / 3600.0)
    dist_km = float((cycle["v_kph"].to_numpy() * dt).sum() / 3600.0)
    vde = E_Wh / max(dist_km, 1e-9)
    return {"abc": abc, "E_Wh": E_Wh, "dist_km": dist_km, "VDE_Wh_per_km": vde}


def default_cycle_for_legislation(leg: str) -> str:
    return {"EPA": "ftp75_hwfet" , "WLTP": "WLTP_Class3ab"}.get(leg, "ftp75")

def load_cycle_csv(name_no_ext: str) -> pd.DataFrame:
    p = Path("data/cycles") / f"{name_no_ext}.csv"
    df = pd.read_csv(p)
    if not {"t","v"} <= set(df.columns):
        raise ValueError("cycle CSV must have columns: t, v (v in m/s)")
    return df

def compute_vde_net_mj_per_km(cycle_df: pd.DataFrame, A: float, B: float, C: float, mass_kg: float):
    r = vde_from_trace(cycle_df["t"].to_numpy(float),
                       cycle_df["v"].to_numpy(float),
                       A,B,C, mass_kg)
    return r  # dict: MJ_km, Wh_km, km

def mjkm_to_whkm(mj_per_km: float) -> float:
    return mj_per_km / 0.0036

def whkm_to_mjkm(wh_per_km: float) -> float:
    return wh_per_km * 0.0036
