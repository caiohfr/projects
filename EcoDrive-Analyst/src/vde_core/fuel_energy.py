from __future__ import annotations

import json


LHV_MJ_PER_L = {
    "Gasoline": 32.0,
    "E10": 31.2,
    "E22": 30.0,
    "E100": 21.2,
    "Diesel": 35.8,
    "Other": 32.0,
}

GCO2_PER_L = {
    "Gasoline": 2310.0,
    "E10": 2270.0,
    "E22": 2200.0,
    "E100": 0.0,
    "Diesel": 2640.0,
    "Other": 2310.0,
}

MJ_TO_Wh = 277.7777777778


def _get_vde_row(vde_id: int) -> dict | None:
    from src.vde_core.db import fetchone as _fetchone

    return _fetchone("SELECT * FROM vde_db WHERE id=?;", (vde_id,))


def compute_ice_fuel_from_vde(
    vde_id: int,
    fuel_type: str,
    eta_pt: float,
    lhv_mj_per_l: float | None = None,
    electrification: str = "ICE",
    uf_phev: float | None = None,
    driveline_eff: float | None = None,
    grid_gco2_per_kwh: float | None = None,
) -> dict:
    row = _get_vde_row(vde_id)
    assert row and ("vde_net_mj_per_km" in row), "VDE row sem vde_net_mj_per_km"
    vde_mj_per_km = float(row["vde_net_mj_per_km"])

    lhv = float(lhv_mj_per_l) if lhv_mj_per_l else float(LHV_MJ_PER_L.get(fuel_type, 32.0))
    gco2_per_l = float(GCO2_PER_L.get(fuel_type, 2310.0))

    mj_pk_ice = vde_mj_per_km / max(eta_pt, 1e-6)
    L_per_km_ice = mj_pk_ice / max(lhv, 1e-6)
    L_per_100km_ice = 100.0 * L_per_km_ice
    km_per_L_ice = 100.0 / max(L_per_100km_ice, 1e-9)
    gco2_per_km_ice = L_per_km_ice * gco2_per_l

    L_per_100 = L_per_100km_ice
    km_per_L = km_per_L_ice
    gco2_km = gco2_per_km_ice
    Wh_per_km = None

    if str(electrification).upper() == "PHEV" and uf_phev is not None:
        uf = max(0.0, min(1.0, float(uf_phev)))
        if driveline_eff and grid_gco2_per_kwh is not None:
            energy_Wh_per_km_elec = (vde_mj_per_km / max(driveline_eff, 1e-6)) * MJ_TO_Wh
            gco2_km_elec = (energy_Wh_per_km_elec / 1000.0) * float(grid_gco2_per_kwh)
        else:
            energy_Wh_per_km_elec = 0.0
            gco2_km_elec = 0.0

        L_per_km_blend = (1.0 - uf) * L_per_km_ice
        L_per_100 = 100.0 * L_per_km_blend
        km_per_L = 100.0 / max(L_per_100, 1e-9) if L_per_100 > 0 else None
        gco2_km = (1.0 - uf) * gco2_per_km_ice + uf * gco2_km_elec
        Wh_per_km = uf * energy_Wh_per_km_elec

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
        "energy_Wh_per_km": Wh_per_km,
        "gco2_per_km": gco2_km,
        "assumptions_json": json.dumps(assumptions),
    }


def compute_bev_from_vde(
    vde_id: int,
    driveline_eff: float,
    grid_gco2_per_kwh: float = 0.0,
) -> dict:
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


__all__ = [
    "GCO2_PER_L",
    "LHV_MJ_PER_L",
    "compute_bev_from_vde",
    "compute_ice_fuel_from_vde",
]
