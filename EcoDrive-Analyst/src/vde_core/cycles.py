from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def default_cycle_for_legislation(leg: str) -> str:
    leg = (leg or "").upper()
    mapping = {
        "EPA": "FTP75_HWFET",
        "WLTP": "WLTP_Class3ab",
    }
    return mapping.get(leg)


def use_standard_cycle(leg):
    fname = default_cycle_for_legislation(leg)
    if not fname:
        return None
    try:
        return load_cycle_csv(fname)
    except Exception:
        return None


def cycle_summary(df_cycle: pd.DataFrame):
    if df_cycle is None or df_cycle.empty:
        return "No cycle loaded.", ""

    t = df_cycle.iloc[:, 0].astype(float)
    v = df_cycle.iloc[:, 1].astype(float)
    dur = t.iloc[-1] - t.iloc[0]
    dist = np.trapezoid(v, t)
    dist_km = dist / 1000.0
    vavg = v.mean() * 3.6
    return f"Duration: {dur:.0f} s • Distance: {dist_km:.2f} km • v̄: {vavg:.1f} km/h", f"{dist_km:.3f}"


def load_cycle_csv(name_no_ext: str) -> pd.DataFrame:
    path = Path("data/cycles") / f"{name_no_ext}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cycle CSV not found: {path}")

    df = pd.read_csv(path)
    if not {"t", "v"} <= set(df.columns):
        raise ValueError(f"{path} must have columns: t, v (v in m/s)")

    df = df.copy()
    df["t"] = df["t"].astype(float)
    df["v"] = df["v"].astype(float)
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype(str)
    return df


__all__ = [
    "cycle_summary",
    "default_cycle_for_legislation",
    "load_cycle_csv",
    "use_standard_cycle",
]
