import numpy as np
import pandas as pd

def cycle_kpis(df: pd.DataFrame) -> dict:
    """Return basic KPIs from a cycle (duration, distance, avg speed, samples)."""
    df = df.sort_values("t")
    t = df["t"].to_numpy()
    v = df["v"].to_numpy()

    if len(t) == 0:
        return {"duration_s": 0.0, "distance_km": 0.0, "v_mean_kmh": 0.0, "n_points": 0}

    duration_s = float(t[-1] - t[0])
    dist_m = float(np.trapz(v, t))
    v_mean_ms = dist_m / max(duration_s, 1e-9)
    return {
        "duration_s": duration_s,
        "distance_km": dist_m / 1000.0,
        "v_mean_kmh": v_mean_ms * 3.6,
        "n_points": int(len(df)),
    }
