

from __future__ import annotations
import json
import math
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --- DB / services (use os caminhos reais do seu projeto) ---
from src.vde_core.db import fetchall


def drop_empty(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if v not in (None, "")}

# =============================================================================
# Regressão (stubs leves que já rodam)
# =============================================================================
def load_regression_dataset(filters: Dict[str, Any], current_vde_id: Optional[int] = None) -> pd.DataFrame:
    base = (
        "SELECT f.vde_id, f.electrification, "
        "f.fuel_l_per_100km, f.energy_Wh_per_km, f.engine_max_power_kw, "
        # EPA phases
        "f.energy_ftp75_Wh_per_km, f.energy_hwfet_Wh_per_km, "
        "f.fuel_ftp75_l_per_100km, f.fuel_hwfet_l_per_100km, "
        # WLTP phases
        "f.energy_low_Wh_per_km, f.energy_mid_Wh_per_km, f.energy_high_Wh_per_km, f.energy_xhigh_Wh_per_km, "
        "f.fuel_low_l_per_100km, f.fuel_mid_l_per_100km, f.fuel_high_l_per_100km, f.fuel_xhigh_l_per_100km, "
        "v.category, v.make, v.vde_net_mj_per_km "
        "FROM fuelcons_db f JOIN vde_db v ON v.id = f.vde_id WHERE 1=1"
    )
    params: list = []

    if filters.get("electrification"):
        base += " AND f.electrification = ?"
        params.append(filters["electrification"])
    if filters.get("category"):
        base += " AND v.category = ?"
        params.append(filters["category"])
    if filters.get("make"):
        base += " AND v.make = ?"
        params.append(filters["make"])

    if "power_kw_range" in filters and filters["power_kw_range"] is not None:
        pmin, pmax = filters["power_kw_range"]
        if pmin is not None and pmax is not None:
            base += " AND f.engine_max_power_kw BETWEEN ? AND ?"
            params += [pmin, pmax]
        elif pmin is not None and pmax is None:
            base += " AND f.engine_max_power_kw >= ?"
            params += [pmin]
        elif pmin is None and pmax is not None:
            base += " AND f.engine_max_power_kw <= ?"
            params += [pmax]

    rows = fetchall(base, tuple(params)) if params else fetchall(base)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df = df[df["vde_net_mj_per_km"].notna()]
    return df


def fit_regression_y_vs_vde(df: pd.DataFrame, y_col: Optional[str], electrification: Optional[str] = None) -> Dict[str, Any]:
    """
    - Modo tradicional: se y_col é string (ex.: 'fuel_l_per_100km'), retorna {'a','b','n','r2','y_col'}.
    - Modo dividido: se y_col é None, monta y_urb/y_hw (EPA preferencial; WLTP fallback) e
      retorna {'_is_split': True, 'urb': {...}, 'hw': {...}}.
    """
    def _fit_simple(sub: pd.DataFrame, yname: str) -> Dict[str, Any]:
        if sub.empty or yname not in sub.columns:
            return {"a": None, "b": None, "n": 0, "r2": None, "y_col": yname}
        work = sub[["vde_net_mj_per_km", yname]].dropna()
        if work.shape[0] < 3:
            return {"a": None, "b": None, "n": int(work.shape[0]), "r2": None, "y_col": yname}
        x = work["vde_net_mj_per_km"].values.astype(float)
        y = work[yname].values.astype(float)
        X = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(beta[0]), float(beta[1])
        yhat = a + b * x
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        return {"a": a, "b": b, "n": int(work.shape[0]), "r2": r2, "y_col": yname}

    if df.empty:
        return {"a": None, "b": None, "n": 0, "r2": None} if y_col else {"_is_split": True, "urb": {}, "hw": {}}

    # ====== MODO TRADICIONAL (compatibilidade) ======
    if isinstance(y_col, str):
        return _fit_simple(df, y_col)

    # ====== MODO DIVIDIDO (urb/hw) ======
    # Preparar alvos por fase conforme electrification
    out = df.copy()
    if (electrification or "").upper() == "BEV":
        if {"energy_ftp75_Wh_per_km", "energy_hwfet_Wh_per_km"}.issubset(out.columns):
            out["y_urb"] = out["energy_ftp75_Wh_per_km"]
            out["y_hw"]  = out["energy_hwfet_Wh_per_km"]
        else:
            out["y_urb"] = pd.concat([out.get("energy_low_Wh_per_km"), out.get("energy_mid_Wh_per_km")], axis=1).mean(axis=1, skipna=True)
            out["y_hw"]  = pd.concat([out.get("energy_high_Wh_per_km"), out.get("energy_xhigh_Wh_per_km")], axis=1).mean(axis=1, skipna=True)
    else:
        if {"fuel_ftp75_l_per_100km", "fuel_hwfet_l_per_100km"}.issubset(out.columns):
            out["y_urb"] = out["fuel_ftp75_l_per_100km"]
            out["y_hw"]  = out["fuel_hwfet_l_per_100km"]
        else:
            out["y_urb"] = pd.concat([out.get("fuel_low_l_per_100km"), out.get("fuel_mid_l_per_100km")], axis=1).mean(axis=1, skipna=True)
            out["y_hw"]  = pd.concat([out.get("fuel_high_l_per_100km"), out.get("fuel_xhigh_l_per_100km")], axis=1).mean(axis=1, skipna=True)

    mod_urb = _fit_simple(out, "y_urb")
    mod_hw  = _fit_simple(out, "y_hw")

    # >>> ADD: modelo combinado (0.55*urb + 0.45*hw) quando houver coeficientes
    w_city, w_hwy = 0.55, 0.45
    comb = {"a": None, "b": None, "n": 0, "r2": None, "y_col": "combined"}
    if (mod_urb.get("a") is not None and mod_urb.get("b") is not None) and \
       (mod_hw.get("a")  is not None and mod_hw.get("b")  is not None):
        comb["a"] = w_city * mod_urb["a"] + w_hwy * mod_hw["a"]
        comb["b"] = w_city * mod_urb["b"] + w_hwy * mod_hw["b"]
        # n/r2 opcionais; aqui preservo mínimos/None para não confundir diagnóstico
        comb["n"] = min(mod_urb.get("n", 0), mod_hw.get("n", 0))
    elif mod_urb.get("a") is not None and mod_urb.get("b") is not None:
        comb = {**mod_urb, "y_col": "combined"}  # fallback: só urbano
    elif mod_hw.get("a") is not None and mod_hw.get("b") is not None:
        comb = {**mod_hw, "y_col": "combined"}   # fallback: só highway
    # <<< ADD

    return {"_is_split": True, "urb": mod_urb, "hw": mod_hw, "combined": comb}


def predict_current_consumption(model: Dict[str, Any], vde_net_mj_per_km: float, electrification: str) -> Dict[str, Any]:
    x = float(vde_net_mj_per_km)

    # -----------------------------
    # Caso 1: modelo único (não-split)
    # -----------------------------
    if "a" in model and "b" in model:
        a, b = model.get("a"), model.get("b")
        if a is None or b is None:
            return {}
        y = a + b * x  # aqui já é o "combinado" (único disponível)
        if electrification == "BEV":
            return {
                "energy_Wh_per_km": y,                 # combinado (principal)

            }
        else:
            return {
                "fuel_l_per_100km": y,                 # combinado (principal)

            }

    # -----------------------------
    # Caso 2: split (urb/hw)
    # -----------------------------
    if model.get("_is_split"):
        def _pred(m):
            a, b = m.get("a"), m.get("b")
            return None if (a is None or b is None) else (a + b * x)

        y_urb = _pred(model.get("urb", {}))
        y_hw  = _pred(model.get("hw", {}))

        # combinado: 0.55*urb + 0.45*hw, com fallback se faltar um dos dois
        if (y_urb is not None) and (y_hw is not None):
            y_comb = 0.55 * y_urb + 0.45 * y_hw
        else:
            y_comb = y_urb if (y_urb is not None) else y_hw  # fallback

        if electrification == "BEV":
            out = {
                "energy_Wh_per_km_urb": y_urb,
                "energy_Wh_per_km_hw":  y_hw,
                "energy_Wh_per_km":     y_comb,               # combinado (principal)       # alias explícito
            }
        else:
            out = {
                "fuel_l_per_100km_urb": y_urb,
                "fuel_l_per_100km_hw":  y_hw,
                "fuel_l_per_100km":     y_comb,               # combinado (principal)
                }
        return out

    # sem modelo válido
    return {}


def build_payload_from_regression(yhat: Dict[str, Any], model: Dict[str, Any], vde_id: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"vde_id": vde_id, "electrification": ctx.get("electrification"), "method_note": "EPA-regression"}
    payload.update(yhat)
    return drop_empty(payload)

