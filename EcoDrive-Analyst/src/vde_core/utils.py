import numpy as np
import pandas as pd

def cycle_kpis(df: pd.DataFrame) -> dict:
    """Return basic KPIs from a cycle (duration, distance, avg speed, samples)."""
    df = df.sort_values("t")
    t = df["t"].to_numpy()
    v = df["v"].to_numpy()

    if len(t) == 0:
        return {"duration_s": 0.0, "distance_km": 0.0, "v_mean_ms": 0.0, "n_points": 0}

    duration_s = float(t[-1] - t[0])
    dist_m = float(np.trapz(v, t))
    v_mean_ms = dist_m / max(duration_s, 1e-9)
    return {
        "duration_s": duration_s,
        "distance_km": dist_m / 1000.0,
        "v_mean_kmh": v_mean_ms * 3.6,
        "n_points": int(len(df)),
    }

def epa_combined_eff_kmpl(city_kmpl: float, hwy_kmpl: float) -> float:
    return 1.0 / (0.55/city_kmpl + 0.45/hwy_kmpl)

def epa_combined_cons_l100(city_l100: float, hwy_l100: float) -> float:
    return 0.55*city_l100 + 0.45*hwy_l100

def load_tire_catalog(csv_path: str):
    import pandas as pd, math

    # lê CSV simples; usa vírgula como decimal se houver
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    # aceita dois esquemas de cabeçalho:
    # 1) "medida", "des. (m)"
    # 2) "tire_size" (ou "size"), "tire_circ_m" (ou "circ_m")
    size_col = "medida" if "medida" in df.columns else ("tire_size" if "tire_size" in df.columns else "size")
    circ_col = "des. (mm)" if "des. (mm)" in df.columns else ("tire_circ_m" if "tire_circ_m" in df.columns else "circ_m")

    # normaliza e converte
    out = pd.DataFrame()
    out["tire_size"] = df[size_col].astype(str).str.strip()
    out["tire_circ_mm"] = (
        df[circ_col].astype(str).str.replace(",", ".", regex=False)
          .astype(float)
    )
    # calcula diâmetro em mm a partir da circunferência (m)
    out["diameter_mm"] = (out["tire_circ_mm"] ) / math.pi

    return out[["tire_size", "tire_circ_mm", "diameter_mm"]]
