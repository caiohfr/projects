import numpy as np
import pandas as pd
from .models import Scenario, RoadLoadABC
from .physics import compose_abc_from_blocks, f_road, power_patch

def compute_abc(scn):
    if scn.mode == "BASELINE":
        return scn.roadload
    else:
        return compose_abc_from_blocks(scn)

def ensure_cycle_columns(df):
    """Normalize cycle to have t_s, v_kph, a_ms2."""
    out = df.copy()
    if "t" in out.columns and "v" in out.columns:
        out = out.rename(columns={"t":"t_s","v":"v_ms"})
        out["v_kph"] = out["v_ms"] * 3.6
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
