from math import isfinite
from typing import Any, Dict, Optional, Tuple

from src.vde_core.pwt_fuel_energy_service import fetch_fuelcons_allowed, fetch_vde_legislation


def _pick(v) -> Optional[float]:
    try:
        v = float(v)
        return v if isfinite(v) else None
    except Exception:
        return None


def _leg_by_vde_id(vde_id: int) -> str:
    return fetch_vde_legislation(vde_id)


def _normalize_pred(yhat: Dict[str, Any], bev: bool) -> Dict[str, Optional[float]]:
    def first(*keys):
        for key in keys:
            if key in yhat:
                return yhat[key]
        return None

    if bev:
        return {
            "urb": _pick(first("urb_Wh_per_km", "urban_Wh_per_km", "city_Wh_per_km", "ftp75_Wh_per_km")),
            "hw": _pick(first("hw_Wh_per_km", "highway_Wh_per_km", "hwfet_Wh_per_km")),
            "comb": _pick(first("comb_Wh_per_km", "combined_Wh_per_km", "energy_Wh_per_km")),
        }

    return {
        "urb": _pick(first("urb_l_per_100km", "urban_l_per_100km", "city_l_per_100km", "ftp75_l_per_100km")),
        "hw": _pick(first("hw_l_per_100km", "highway_l_per_100km", "hwfet_l_per_100km")),
        "comb": _pick(first("comb_l_per_100km", "combined_l_per_100km", "fuel_l_per_100km")),
    }


def load_fuelcons_allowed(exclude_keys: tuple[str, ...] = ("id",)) -> list[str]:
    return fetch_fuelcons_allowed(exclude_keys=exclude_keys)


FUELCONS_ALLOWED = load_fuelcons_allowed()


def build_min_payload(vde_id: int, electrification: str, yhat: Dict[str, Any], method_note: str) -> Dict[str, Any]:
    elec = (str(electrification or "")).strip().upper()
    bev = elec == "BEV"
    leg = (str(_leg_by_vde_id(vde_id) or "")).strip().upper()

    pred = yhat or {}
    payload = {
        "vde_id": vde_id,
        "electrification": elec or None,
        "method_note": method_note or None,
    }

    if leg == "EPA":
        if bev:
            if pred.get("energy_Wh_per_km_urb") is not None:
                payload["energy_ftp75_Wh_per_km"] = float(pred["energy_Wh_per_km_urb"])
            if pred.get("energy_Wh_per_km_hw") is not None:
                payload["energy_hwfet_Wh_per_km"] = float(pred["energy_Wh_per_km_hw"])
            if pred.get("energy_Wh_per_km") is not None:
                payload["energy_Wh_per_km"] = float(pred["energy_Wh_per_km"])
        else:
            if pred.get("fuel_l_per_100km_urb") is not None:
                payload["fuel_ftp75_l_per_100km"] = float(pred["fuel_l_per_100km_urb"])
            if pred.get("fuel_l_per_100km_hw") is not None:
                payload["fuel_hwfet_l_per_100km"] = float(pred["fuel_l_per_100km_hw"])
            if pred.get("fuel_l_per_100km") is not None:
                payload["fuel_l_per_100km"] = float(pred["fuel_l_per_100km"])
    else:
        if bev:
            if pred.get("energy_Wh_per_km") is not None:
                payload["energy_Wh_per_km"] = float(pred["energy_Wh_per_km"])
        else:
            if pred.get("fuel_l_per_100km") is not None:
                payload["fuel_l_per_100km"] = float(pred["fuel_l_per_100km"])

    return {k: v for k, v in payload.items() if v not in (None, "", [])}


def _ok(x):
    try:
        return x is not None and isfinite(float(x))
    except Exception:
        return False


def _wavg(a: float, b: float, w=(0.55, 0.45)):
    return w[0] * a + w[1] * b


def _km_per_liter(l_100):
    return 100.0 / l_100


def _gco2_from_l100(l_100, ef_g_per_l):
    return (l_100 * ef_g_per_l) / 100.0


def _l100_to_Whkm(l100: float, lhv_mj_per_l: float) -> float:
    return (float(l100) * float(lhv_mj_per_l) / 100.0) * 277.7778


EF_G_PER_L_DEFAULT = {
    "Gasoline": 2310.0,
    "Diesel": 2680.0,
    "Ethanol": 1500.0,
}


LHV_DEFAULT_MJ_PER_L = {
    "Gasoline": 34.2,
    "Diesel": 38.6,
    "Ethanol": 21.2,
}


def enrich_with_derivatives(
    payload: Dict,
    electrification: str,
    fuel_type: Optional[str] = "Gasoline",
    w_city_hwy: Tuple[float, float] = (0.55, 0.45),
    ef_map: Optional[Dict[str, float]] = None,
    w_wltp: Tuple[float, float, float, float] = (0.13, 0.26, 0.31, 0.30),
    lhv_map: Optional[Dict[str, float]] = None,
) -> Dict:
    ef_map = ef_map or EF_G_PER_L_DEFAULT
    lhv_map = lhv_map or LHV_DEFAULT_MJ_PER_L
    is_bev = str(electrification or "").upper() == "BEV"

    if is_bev:
        u = payload.get("energy_ftp75_Wh_per_km")
        h = payload.get("energy_hwfet_Wh_per_km")
        c = payload.get("energy_Wh_per_km")

        if c is None and _ok(u) and _ok(h):
            payload["energy_Wh_per_km"] = _wavg(float(u), float(h), w_city_hwy)

        if payload.get("energy_Wh_per_km") is None:
            low = payload.get("energy_low_Wh_per_km")
            mid = payload.get("energy_mid_Wh_per_km")
            high = payload.get("energy_high_Wh_per_km")
            xhi = payload.get("energy_xhigh_Wh_per_km")
            vals, wts = [], []
            for val, wt in ((low, w_wltp[0]), (mid, w_wltp[1]), (high, w_wltp[2]), (xhi, w_wltp[3])):
                if _ok(val):
                    vals.append(float(val))
                    wts.append(wt)
            if vals:
                scale = sum(wts) or 1.0
                wts = [w / scale for w in wts]
                payload["energy_Wh_per_km"] = sum(v * w for v, w in zip(vals, wts))

        if _ok(payload.get("energy_Wh_per_km")) and payload.get("gco2_per_km") is None:
            payload["gco2_per_km"] = 0.0
        return payload

    ef = ef_map.get(fuel_type or "Gasoline", EF_G_PER_L_DEFAULT["Gasoline"])
    lhv = lhv_map.get(fuel_type or "Gasoline", LHV_DEFAULT_MJ_PER_L["Gasoline"])

    u = payload.get("fuel_ftp75_l_per_100km")
    h = payload.get("fuel_hwfet_l_per_100km")
    c = payload.get("fuel_l_per_100km")
    if c is None and _ok(u) and _ok(h):
        payload["fuel_l_per_100km"] = _wavg(float(u), float(h), w_city_hwy)

    if payload.get("fuel_l_per_100km") is None:
        low = payload.get("fuel_low_l_per_100km")
        mid = payload.get("fuel_mid_l_per_100km")
        high = payload.get("fuel_high_l_per_100km")
        xhi = payload.get("fuel_xhigh_l_per_100km")
        vals, wts = [], []
        for val, wt in ((low, w_wltp[0]), (mid, w_wltp[1]), (high, w_wltp[2]), (xhi, w_wltp[3])):
            if _ok(val):
                vals.append(float(val))
                wts.append(wt)
        if vals:
            scale = sum(wts) or 1.0
            wts = [w / scale for w in wts]
            payload["fuel_l_per_100km"] = sum(v * w for v, w in zip(vals, wts))

    if _ok(payload.get("fuel_l_per_100km")) and payload.get("fuel_km_per_l") is None:
        payload["fuel_km_per_l"] = _km_per_liter(float(payload["fuel_l_per_100km"]))
    if _ok(payload.get("fuel_l_per_100km")) and payload.get("gco2_per_km") is None:
        payload["gco2_per_km"] = _gco2_from_l100(float(payload["fuel_l_per_100km"]), ef)

    for fuel_key, gco2_key, energy_key in [
        ("fuel_ftp75_l_per_100km", "gco2_ftp75_per_km", "energy_ftp75_Wh_per_km"),
        ("fuel_hwfet_l_per_100km", "gco2_hwfet_per_km", "energy_hwfet_Wh_per_km"),
        ("fuel_us06_l_per_100km", "gco2_us06_per_km", "energy_us06_Wh_per_km"),
        ("fuel_sc03_l_per_100km", "gco2_sc03_per_km", "energy_sc03_Wh_per_km"),
        ("fuel_coldftp_l_per_100km", "gco2_coldftp_per_km", "energy_coldftp_Wh_per_km"),
        ("fuel_low_l_per_100km", "gco2_low_per_km", "energy_low_Wh_per_km"),
        ("fuel_mid_l_per_100km", "gco2_mid_per_km", "energy_mid_Wh_per_km"),
        ("fuel_high_l_per_100km", "gco2_high_per_km", "energy_high_Wh_per_km"),
        ("fuel_xhigh_l_per_100km", "gco2_xhigh_per_km", "energy_xhigh_Wh_per_km"),
    ]:
        if _ok(payload.get(fuel_key)) and payload.get(gco2_key) is None:
            payload[gco2_key] = _gco2_from_l100(float(payload[fuel_key]), ef)
        if _ok(payload.get(fuel_key)) and payload.get(energy_key) is None:
            payload[energy_key] = _l100_to_Whkm(payload[fuel_key], lhv)

    if _ok(payload.get("fuel_l_per_100km")) and payload.get("energy_Wh_per_km") is None:
        payload["energy_Wh_per_km"] = _l100_to_Whkm(payload["fuel_l_per_100km"], lhv)

    return payload


def filter_payload(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k in FUELCONS_ALLOWED}
