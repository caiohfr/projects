from __future__ import annotations

from typing import Any

from src.vde_core.fuel_energy import LHV_MJ_PER_L, MJ_TO_Wh


def _clean_text(value: Any, default: str | None = None, *, upper: bool = False) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.upper() if upper else text


def _to_float(value: Any, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_demand_mj_per_km(request: Any, energy_basis_used: str) -> float | None:
    vehicle_features = dict(getattr(request, "vehicle_features", {}) or {})
    manual_inputs = dict(getattr(request, "manual_inputs", {}) or {})
    basis = _clean_text(energy_basis_used, "VDE_TOTAL", upper=True) or "VDE_TOTAL"
    if basis == "VDE_TOTAL":
        return _to_float(vehicle_features.get("vde_total_mj_per_km"))
    if basis == "VDE_NET":
        return _to_float(vehicle_features.get("vde_net_mj_per_km"))
    if basis == "CYCLE_PHASE_VDE":
        return _to_float(manual_inputs.get("phase_vde_mj_per_km"))
    if basis in {"IMPORTED_MEASURED_ENERGY", "MANUAL_VALUE"}:
        return _to_float(manual_inputs.get("vde_mj_per_km"))
    return None


def _cycle_basis_label(cycle_name: str | None) -> str:
    cycle = _clean_text(cycle_name, "", upper=True) or ""
    if not cycle:
        return "custom cycle"
    if "FTP" in cycle:
        return "FTP"
    if "HWFET" in cycle or "HWY" in cycle or "HIGHWAY" in cycle:
        return "HWFET"
    if "WLTP" in cycle:
        return "WLTP phase" if "LOW" in cycle or "MID" in cycle or "HIGH" in cycle or "EXTRA" in cycle else "WLTP"
    if "COMBINED" in cycle or "FTP75_HWFET" in cycle:
        return "Combined"
    return "custom cycle"


def _pse_source_context(method: str, *, has_result: bool, has_physics_assumption: bool) -> dict[str, str]:
    if not has_result:
        return {
            "mode": "unavailable",
            "source": "unavailable",
            "source_label": "PSE unavailable",
            "target_type": "unavailable",
        }
    if method == "physics_simple" and has_physics_assumption:
        return {
            "mode": "assumed",
            "source": "physics_assumption",
            "source_label": "Physics efficiency assumption",
            "target_type": "assumption",
        }
    if method == "manual_imported":
        return {
            "mode": "derived",
            "source": "imported_result",
            "source_label": "Derived from imported/observed result",
            "target_type": "observed_result",
        }
    if method == "physics_simple":
        return {
            "mode": "derived",
            "source": "physics_result",
            "source_label": "Derived from physics result",
            "target_type": "fuel_direct",
        }
    if method == "regression_existing":
        return {
            "mode": "derived",
            "source": "regression_fuel_estimate",
            "source_label": "Derived from regression fuel estimate",
            "target_type": "fuel_direct",
        }
    if method == "ml_prediction":
        return {
            "mode": "derived",
            "source": "ml_fuel_prediction",
            "source_label": "Derived from ML fuel prediction",
            "target_type": "fuel_direct",
        }
    return {
        "mode": "derived",
        "source": "unavailable",
        "source_label": "Derived from final result",
        "target_type": "fuel_direct",
    }


def build_powertrain_efficiency_summary(
    *,
    request: Any,
    method: str,
    energy_basis_used: str,
    fuel_l_100km: float | None,
    energy_Wh_km: float | None,
    assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assumptions = dict(assumptions or {})
    vehicle_features = dict(getattr(request, "vehicle_features", {}) or {})
    powertrain_features = dict(getattr(request, "powertrain_features", {}) or {})
    demand_mj_per_km = _resolve_demand_mj_per_km(request, energy_basis_used)
    electrification = _clean_text(
        vehicle_features.get("electrification") or assumptions.get("electrification"),
        "ICE",
        upper=True,
    ) or "ICE"

    fuel_type = _clean_text(
        powertrain_features.get("fuel_type") or assumptions.get("fuel_type"),
        "Gasoline",
        upper=False,
    ) or "Gasoline"
    lhv_mj_per_l = _to_float(
        powertrain_features.get("LHV_MJ_per_L") or assumptions.get("lhv_mj_per_l"),
        LHV_MJ_PER_L.get(fuel_type, LHV_MJ_PER_L["Gasoline"]),
    )
    has_physics_assumption = (
        _to_float(powertrain_features.get("eta_pt_est") or assumptions.get("eta_pt_est")) is not None
        or _to_float(powertrain_features.get("bev_eff_drive") or assumptions.get("bev_eff_drive")) is not None
    )

    fuel_consumed_mj_per_km = None
    if _to_float(fuel_l_100km) is not None and lhv_mj_per_l is not None:
        fuel_consumed_mj_per_km = (_to_float(fuel_l_100km) / 100.0) * lhv_mj_per_l

    electric_consumed_mj_per_km = None
    if _to_float(energy_Wh_km) is not None:
        electric_consumed_mj_per_km = _to_float(energy_Wh_km) / MJ_TO_Wh

    total_consumed_mj_per_km = None
    if electrification == "BEV":
        total_consumed_mj_per_km = electric_consumed_mj_per_km
    elif electrification == "PHEV":
        available_components = [value for value in (fuel_consumed_mj_per_km, electric_consumed_mj_per_km) if value is not None]
        if available_components:
            total_consumed_mj_per_km = sum(available_components)
    else:
        total_consumed_mj_per_km = fuel_consumed_mj_per_km if fuel_consumed_mj_per_km is not None else electric_consumed_mj_per_km

    pse_value = None
    if demand_mj_per_km is not None and total_consumed_mj_per_km not in (None, 0):
        pse_value = demand_mj_per_km / total_consumed_mj_per_km

    available = pse_value is not None
    warning_text = (
        "PSE is cycle-effective and should not be interpreted as pure engine efficiency."
    )
    source_context = _pse_source_context(
        method,
        has_result=available,
        has_physics_assumption=has_physics_assumption,
    )
    limitations = [
        warning_text,
        "Direct PSE prediction is a future ML target unless an artifact is trained specifically for eta_pt_cycle.",
    ]
    if source_context["source"] == "ml_fuel_prediction":
        limitations.append("Current ML artifact predicts final fuel/energy outputs; PSE is derived from that result.")
    if electrification not in {"BEV", "PHEV"} and fuel_consumed_mj_per_km is not None and electric_consumed_mj_per_km is not None:
        limitations.append("Equivalent Wh/km derived from fuel is informational for ICE/HEV and is not added twice into PSE.")

    return {
        "value": pse_value,
        "percent_value": (pse_value * 100.0) if pse_value is not None else None,
        "percent": (pse_value * 100.0) if pse_value is not None else None,
        "mode": source_context["mode"],
        "source": source_context["source"],
        "source_label": source_context["source_label"],
        "target_type": source_context["target_type"],
        "status": "PSE Available" if available else "PSE Unavailable",
        "cycle_basis": _cycle_basis_label(getattr(request, "cycle", None)),
        "basis": energy_basis_used,
        "energy_basis_used": energy_basis_used,
        "demand_mj_per_km": demand_mj_per_km,
        "fuel_consumed_mj_per_km": fuel_consumed_mj_per_km,
        "electric_consumed_mj_per_km": electric_consumed_mj_per_km,
        "consumed_energy_mj_per_km": total_consumed_mj_per_km,
        "total_consumed_mj_per_km": total_consumed_mj_per_km,
        "cycle_effective": True,
        "warning": warning_text,
        "warnings": [] if available else ["pse_unavailable"],
        "limitations": limitations,
    }


__all__ = ["build_powertrain_efficiency_summary"]
