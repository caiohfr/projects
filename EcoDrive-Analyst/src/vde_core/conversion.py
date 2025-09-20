def estimate_fuel_from_vde(vde_kwh_per_100km: float, eta_pt: float, lhv_mj_per_L: float) -> float:
    """Estimate fuel consumption [L/100km] from VDE and powertrain efficiency."""
    if vde_kwh_per_100km != vde_kwh_per_100km:  # NaN check
        return float("nan")
    if eta_pt <= 0 or lhv_mj_per_L <= 0:
        return float("nan")
    return (vde_kwh_per_100km * 3.6) / (eta_pt * lhv_mj_per_L)
