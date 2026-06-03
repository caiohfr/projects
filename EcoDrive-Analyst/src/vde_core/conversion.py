def mjkm_to_whkm(mj_per_km: float) -> float:
    return mj_per_km / 0.0036


def whkm_to_mjkm(wh_per_km: float) -> float:
    return wh_per_km * 0.0036


__all__ = ["mjkm_to_whkm", "whkm_to_mjkm"]
