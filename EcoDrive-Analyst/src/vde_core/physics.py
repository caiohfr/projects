import pandas as pd

import numpy as np, pandas as pd

def compute_vde_net(cycle_df: pd.DataFrame, f0: float, f1: float, f2: float, mass: float) -> float:
    df = cycle_df.sort_values("t").copy()
    t = df["t"].to_numpy(); v = df["v"].to_numpy()
    dt = np.gradient(t);   a = np.gradient(v, t)

    F = f0 + f1*v + f2*(v**2)        # N
    P = (F + mass*a) * v             # W
    P = np.maximum(P, 0.0)           # NET: zera regen/negativo

    E_J = float((P * dt).sum())
    dist_m = float(np.trapz(v, t))
    e_per_m = E_J / max(dist_m, 1e-12)
    return (e_per_m * 100_000) / 3.6e6   # kWh/100km
