"""
Additive Fuel Energy estimation contracts for Sprint 5.

This module separates estimation logic from Streamlit pages and from the
legacy PWT/Fuel page orchestration. It is meant to be adopted gradually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from src.vde_core.estimate_confidence import build_estimate_confidence_summary
from src.vde_core.fuel_energy import GCO2_PER_L, LHV_MJ_PER_L, MJ_TO_Wh
from src.vde_core.ml_prediction import predict_fuel_with_ml
from src.vde_core.powertrain_efficiency import build_powertrain_efficiency_summary
from src.vde_core.repositories import delete_fuelcons_by_id, insert_fuelcons_row, update_fuelcons_by_id


INSERT_MODES = {"insert_new", "save_as_new", "new"}
UPDATE_MODES = {"update_existing", "update"}
DELETE_MODES = {"delete_existing", "delete"}
ENGINE_VERSION = "fuel_estimation_v1"


def _to_float(value: Any, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any, default: str | None = None, *, upper: bool = False) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.upper() if upper else text


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str, sort_keys=True)


@dataclass
class FuelEstimateRequest:
    vde_id: int | None = None
    energy_basis: str = "VDE_TOTAL"
    cycle: str | None = None
    vehicle_features: dict[str, Any] = field(default_factory=dict)
    powertrain_features: dict[str, Any] = field(default_factory=dict)
    method: str = "physics_simple"
    model_options: dict[str, Any] = field(default_factory=dict)
    manual_inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vde_id": self.vde_id,
            "energy_basis": self.energy_basis,
            "cycle": self.cycle,
            "vehicle_features": dict(self.vehicle_features),
            "powertrain_features": dict(self.powertrain_features),
            "method": self.method,
            "model_options": dict(self.model_options),
            "manual_inputs": dict(self.manual_inputs),
        }


@dataclass
class FuelEstimateResult:
    request: FuelEstimateRequest
    method: str
    energy_basis_used: str
    fuel_l_100km: float | None = None
    energy_Wh_km: float | None = None
    gco2_km: float | None = None
    confidence: str | None = None
    warnings: list[str] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)
    comparables: list[dict[str, Any]] = field(default_factory=list)
    feature_contributions: dict[str, Any] = field(default_factory=dict)
    residual_correction: dict[str, Any] | None = None
    phase_outputs: dict[str, Any] = field(default_factory=dict)
    preview_saved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "method": self.method,
            "energy_basis_used": self.energy_basis_used,
            "fuel_l_100km": self.fuel_l_100km,
            "energy_Wh_km": self.energy_Wh_km,
            "gco2_km": self.gco2_km,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "assumptions": dict(self.assumptions),
            "comparables": list(self.comparables),
            "feature_contributions": dict(self.feature_contributions),
            "residual_correction": self.residual_correction,
            "phase_outputs": dict(self.phase_outputs),
            "preview_saved": self.preview_saved,
        }


@dataclass
class FuelScenarioSavePayload:
    result: FuelEstimateResult
    payload: dict[str, Any]
    data_origin: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.to_dict(),
            "payload": dict(self.payload),
            "data_origin": self.data_origin,
        }


def _coerce_request(request: FuelEstimateRequest | dict[str, Any]) -> FuelEstimateRequest:
    if isinstance(request, FuelEstimateRequest):
        return request
    if not isinstance(request, dict):
        raise TypeError("request must be a FuelEstimateRequest or dict")
    return FuelEstimateRequest(
        vde_id=request.get("vde_id"),
        energy_basis=request.get("energy_basis", "VDE_TOTAL"),
        cycle=request.get("cycle"),
        vehicle_features=dict(request.get("vehicle_features") or {}),
        powertrain_features=dict(request.get("powertrain_features") or {}),
        method=request.get("method", "physics_simple"),
        model_options=dict(request.get("model_options") or {}),
        manual_inputs=dict(request.get("manual_inputs") or {}),
    )


def _resolve_energy_basis(req: FuelEstimateRequest) -> tuple[float | None, str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    phase_outputs = dict(req.vehicle_features.get("phase_outputs") or {})
    basis = _clean_text(req.energy_basis, "VDE_TOTAL", upper=True) or "VDE_TOTAL"

    if basis == "VDE_TOTAL":
        value = _to_float(req.vehicle_features.get("vde_total_mj_per_km"))
        if value is None:
            warnings.append("vde_total_missing")
        return value, basis, phase_outputs, warnings

    if basis == "VDE_NET":
        value = _to_float(req.vehicle_features.get("vde_net_mj_per_km"))
        if value is None:
            warnings.append("vde_net_selected_but_unavailable")
        return value, basis, phase_outputs, warnings

    if basis == "CYCLE_PHASE_VDE":
        value = _to_float(req.manual_inputs.get("phase_vde_mj_per_km"))
        if value is None:
            warnings.append("phase_values_missing")
        return value, basis, phase_outputs, warnings

    if basis in {"IMPORTED_MEASURED_ENERGY", "MANUAL_VALUE"}:
        value = _to_float(req.manual_inputs.get("vde_mj_per_km"))
        if value is None:
            warnings.append("manual_or_imported_energy_missing")
        if not _clean_text(req.manual_inputs.get("source")):
            warnings.append("manual_or_imported_value_without_source")
        return value, basis, phase_outputs, warnings

    warnings.append(f"unsupported_energy_basis:{basis}")
    return None, basis, phase_outputs, warnings


def _physics_simple(req: FuelEstimateRequest, vde_mj_per_km: float | None) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    warnings: list[str] = []
    outputs = {
        "fuel_l_100km": None,
        "energy_Wh_km": None,
        "gco2_km": None,
    }
    assumptions = {
        "electrification": _clean_text(req.vehicle_features.get("electrification"), "ICE", upper=True),
    }

    if vde_mj_per_km is None:
        warnings.append("energy_basis_value_missing")
        return outputs, assumptions, warnings

    electrification = assumptions["electrification"]
    fuel_type = _clean_text(req.powertrain_features.get("fuel_type"), "Gasoline") or "Gasoline"
    lhv = _to_float(req.powertrain_features.get("LHV_MJ_per_L"), LHV_MJ_PER_L.get(fuel_type, 32.0))
    gco2_per_l = _to_float(req.powertrain_features.get("gCO2_per_L"), GCO2_PER_L.get(fuel_type, 2310.0))
    eta_pt = _to_float(req.powertrain_features.get("eta_pt_est"))
    bev_eff = _to_float(req.powertrain_features.get("bev_eff_drive"))
    utility_factor = _to_float(req.powertrain_features.get("utility_factor"), 0.0)
    grid = _to_float(req.powertrain_features.get("grid_gco2_per_kwh"), 0.0)

    assumptions.update(
        {
            "fuel_type": fuel_type,
            "lhv_mj_per_l": lhv,
            "gco2_per_l": gco2_per_l,
            "eta_pt_est": eta_pt,
            "bev_eff_drive": bev_eff,
            "utility_factor": utility_factor,
            "vde_mj_per_km": vde_mj_per_km,
        }
    )

    if electrification == "BEV":
        if not bev_eff or bev_eff <= 0:
            warnings.append("bev_eff_drive_missing")
            return outputs, assumptions, warnings
        outputs["energy_Wh_km"] = (vde_mj_per_km / bev_eff) * MJ_TO_Wh
        outputs["gco2_km"] = (outputs["energy_Wh_km"] / 1000.0) * grid
        return outputs, assumptions, warnings

    if electrification == "PHEV":
        utility_factor = max(0.0, min(1.0, utility_factor or 0.0))
        if eta_pt and eta_pt > 0 and lhv and lhv > 0:
            fuel_l_100km = (vde_mj_per_km / eta_pt) / lhv * 100.0
            outputs["fuel_l_100km"] = (1.0 - utility_factor) * fuel_l_100km
            outputs["gco2_km"] = ((outputs["fuel_l_100km"] / 100.0) * gco2_per_l)
        else:
            warnings.append("eta_pt_est_missing")
        if bev_eff and bev_eff > 0:
            outputs["energy_Wh_km"] = utility_factor * ((vde_mj_per_km / bev_eff) * MJ_TO_Wh)
        else:
            warnings.append("bev_eff_drive_missing")
        return outputs, assumptions, warnings

    if not eta_pt or eta_pt <= 0:
        warnings.append("eta_pt_est_missing")
        return outputs, assumptions, warnings
    if not lhv or lhv <= 0:
        warnings.append("lhv_missing")
        return outputs, assumptions, warnings

    outputs["fuel_l_100km"] = (vde_mj_per_km / eta_pt) / lhv * 100.0
    outputs["gco2_km"] = ((outputs["fuel_l_100km"] / 100.0) * gco2_per_l)
    return outputs, assumptions, warnings


def _manual_imported(req: FuelEstimateRequest) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    warnings: list[str] = []
    inputs = dict(req.manual_inputs or {})
    if not _clean_text(inputs.get("source")):
        warnings.append("manual_or_imported_value_without_source")
    return (
        {
            "fuel_l_100km": _to_float(inputs.get("fuel_l_100km")),
            "energy_Wh_km": _to_float(inputs.get("energy_Wh_km")),
            "gco2_km": _to_float(inputs.get("gco2_km")),
        },
        {"source": inputs.get("source")},
        warnings,
    )


def _regression_existing(req: FuelEstimateRequest, vde_mj_per_km: float | None) -> tuple[dict[str, Any], dict[str, Any], list[str], str]:
    warnings: list[str] = []
    runner = req.model_options.get("regression_runner")
    if callable(runner):
        data = runner(req.to_dict(), vde_mj_per_km)
        outputs = dict(data or {})
        outputs.update({
            "fuel_l_100km": _to_float((data or {}).get("fuel_l_100km")),
            "energy_Wh_km": _to_float((data or {}).get("energy_Wh_km")),
            "gco2_km": _to_float((data or {}).get("gco2_km")),
        })
        assumptions = dict((data or {}).get("assumptions") or {})
        warnings.extend(list((data or {}).get("warnings") or []))
        assumptions.setdefault("runner", "custom")
        confidence = _clean_text((data or {}).get("confidence"), "medium", upper=False) or "medium"
        return outputs, assumptions, warnings, confidence

    warnings.append("regression_runner_missing")
    return {"fuel_l_100km": None, "energy_Wh_km": None, "gco2_km": None}, {"runner": "missing"}, warnings, "low"


def _ml_prediction(req: FuelEstimateRequest) -> tuple[dict[str, Any], dict[str, Any], list[str], str]:
    result = predict_fuel_with_ml(
        req,
        model_artifact_path=_clean_text(req.model_options.get("ml_artifact_path"), upper=False),
        predictor=req.model_options.get("ml_predictor"),
    )
    outputs = dict(result.get("outputs") or {})
    assumptions = dict(result.get("assumptions") or {})
    warnings = list(result.get("warnings") or [])
    confidence = _clean_text(result.get("confidence"), "low", upper=False) or "low"
    return outputs, assumptions, warnings, confidence


_PHASE_FIELD_MAP = {
    "vde_urb_mj_per_km": ("fuel_ftp75_l_per_100km", "energy_ftp75_Wh_per_km", "gco2_ftp75_per_km"),
    "vde_hw_mj_per_km": ("fuel_hwfet_l_per_100km", "energy_hwfet_Wh_per_km", "gco2_hwfet_per_km"),
    "vde_low_mj_per_km": ("fuel_low_l_per_100km", "energy_low_Wh_per_km", "gco2_low_per_km"),
    "vde_mid_mj_per_km": ("fuel_mid_l_per_100km", "energy_mid_Wh_per_km", "gco2_mid_per_km"),
    "vde_high_mj_per_km": ("fuel_high_l_per_100km", "energy_high_Wh_per_km", "gco2_high_per_km"),
    "vde_extra_high_mj_per_km": ("fuel_xhigh_l_per_100km", "energy_xhigh_Wh_per_km", "gco2_xhigh_per_km"),
}


def _physics_phase_outputs(req: FuelEstimateRequest, assumptions: dict[str, Any]) -> dict[str, Any]:
    phase_inputs = dict(req.vehicle_features.get("phase_outputs") or {})
    if not phase_inputs:
        return {}

    electrification = _clean_text(assumptions.get("electrification"), "ICE", upper=True) or "ICE"
    lhv = _to_float(assumptions.get("lhv_mj_per_l"))
    gco2_per_l = _to_float(assumptions.get("gco2_per_l"))
    eta_pt = _to_float(assumptions.get("eta_pt_est"))
    bev_eff = _to_float(assumptions.get("bev_eff_drive"))
    utility_factor = max(0.0, min(1.0, _to_float(assumptions.get("utility_factor"), 0.0) or 0.0))
    grid = _to_float(req.powertrain_features.get("grid_gco2_per_kwh"), 0.0) or 0.0

    out: dict[str, Any] = {}
    for source_key, target_keys in _PHASE_FIELD_MAP.items():
        phase_vde = _to_float(phase_inputs.get(source_key))
        if phase_vde is None:
            continue
        fuel_key, energy_key, gco2_key = target_keys

        if electrification == "BEV":
            if not bev_eff or bev_eff <= 0:
                continue
            energy_wh = (phase_vde / bev_eff) * MJ_TO_Wh
            out[energy_key] = energy_wh
            out[gco2_key] = (energy_wh / 1000.0) * grid
            continue

        fuel_l_100km = None
        energy_wh = None
        gco2_km = None

        if eta_pt and eta_pt > 0 and lhv and lhv > 0:
            raw_fuel_l_100km = (phase_vde / eta_pt) / lhv * 100.0
            if electrification == "PHEV":
                fuel_l_100km = (1.0 - utility_factor) * raw_fuel_l_100km
            else:
                fuel_l_100km = raw_fuel_l_100km
            out[fuel_key] = fuel_l_100km
            if gco2_per_l is not None:
                gco2_km = (fuel_l_100km / 100.0) * gco2_per_l

        if electrification == "PHEV":
            if bev_eff and bev_eff > 0:
                energy_wh = utility_factor * ((phase_vde / bev_eff) * MJ_TO_Wh)
                out[energy_key] = energy_wh
                gco2_km = (gco2_km or 0.0) + ((energy_wh / 1000.0) * grid)
        elif eta_pt and eta_pt > 0:
            energy_wh = (phase_vde / eta_pt) * MJ_TO_Wh
            out[energy_key] = energy_wh

        if gco2_km is not None:
            out[gco2_key] = gco2_km

    return out


def _regression_phase_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if _to_float(outputs.get("fuel_l_per_100km_urb")) is not None:
        out["fuel_ftp75_l_per_100km"] = _to_float(outputs.get("fuel_l_per_100km_urb"))
    if _to_float(outputs.get("fuel_l_per_100km_hw")) is not None:
        out["fuel_hwfet_l_per_100km"] = _to_float(outputs.get("fuel_l_per_100km_hw"))
    if _to_float(outputs.get("energy_Wh_km_urb")) is not None:
        out["energy_ftp75_Wh_per_km"] = _to_float(outputs.get("energy_Wh_km_urb"))
    if _to_float(outputs.get("energy_Wh_km_hw")) is not None:
        out["energy_hwfet_Wh_per_km"] = _to_float(outputs.get("energy_Wh_km_hw"))
    if _to_float(outputs.get("gco2_km_urb")) is not None:
        out["gco2_ftp75_per_km"] = _to_float(outputs.get("gco2_km_urb"))
    if _to_float(outputs.get("gco2_km_hw")) is not None:
        out["gco2_hwfet_per_km"] = _to_float(outputs.get("gco2_km_hw"))
    return out


def _phase_outputs_for_result(
    req: FuelEstimateRequest,
    *,
    method: str,
    outputs: dict[str, Any],
    assumptions: dict[str, Any],
) -> dict[str, Any]:
    if method == "physics_simple":
        return _physics_phase_outputs(req, assumptions)
    if method == "regression_existing":
        return _regression_phase_outputs(outputs)
    if method == "ml_prediction":
        return _regression_phase_outputs(outputs)
    return {}


def _data_origin_for_result(result: FuelEstimateResult) -> str:
    if result.method == "manual_imported":
        return "measured/imported"
    if result.method == "physics_simple":
        return "physics"
    if result.method == "regression_existing":
        return "regression"
    if result.method == "ml_prediction":
        return "ml_prediction"
    return "unknown"


def _resolved_energy_basis_value(result: FuelEstimateResult) -> float | None:
    req = result.request
    basis = _clean_text(result.energy_basis_used, "VDE_TOTAL", upper=True) or "VDE_TOTAL"
    if basis == "VDE_TOTAL":
        return _to_float(req.vehicle_features.get("vde_total_mj_per_km"))
    if basis == "VDE_NET":
        return _to_float(req.vehicle_features.get("vde_net_mj_per_km"))
    if basis == "CYCLE_PHASE_VDE":
        return _to_float(req.manual_inputs.get("phase_vde_mj_per_km"))
    if basis in {"IMPORTED_MEASURED_ENERGY", "MANUAL_VALUE"}:
        return _to_float(req.manual_inputs.get("vde_mj_per_km"))
    return None


def _build_provenance_payload(result: FuelEstimateResult) -> dict[str, Any]:
    req = result.request
    vehicle = dict(req.vehicle_features or {})
    return {
        "vde_id": req.vde_id,
        "cycle": req.cycle,
        "data_origin": _data_origin_for_result(result),
        "energy_basis": result.energy_basis_used,
        "energy_basis_value": _resolved_energy_basis_value(result),
        "engine_method": result.method,
        "engine_version": ENGINE_VERSION,
        "source_vde_revision": vehicle.get("source_vde_revision"),
        "source_vde_created_at": vehicle.get("source_vde_created_at"),
        "source_vde_updated_at": vehicle.get("source_vde_updated_at"),
        "confidence": result.confidence,
        "confidence_summary": dict((result.assumptions or {}).get("confidence_summary") or {}),
        "pse_summary": dict((result.assumptions or {}).get("pse_summary") or {}),
        "scenario_feature_sources": dict(vehicle.get("scenario_feature_sources") or {}),
        "scenario_feature_values": dict(vehicle.get("scenario_feature_values") or {}),
        "scenario_feature_overrides": dict(vehicle.get("scenario_feature_overrides") or {}),
        "scenario_feature_missing": list(vehicle.get("scenario_feature_missing") or []),
        "scenario_feature_imputed": list(vehicle.get("scenario_feature_imputed") or []),
        "scenario_feature_confidence_impacts": list(vehicle.get("scenario_feature_confidence_impacts") or []),
        "scenario_feature_readiness": dict(vehicle.get("scenario_feature_readiness") or {}),
        "powertrain_reference": dict(vehicle.get("powertrain_reference") or {}),
        "baseline_estimate": dict(vehicle.get("baseline_estimate") or {}),
        "technology_deltas": list(vehicle.get("technology_deltas") or []),
        "proposal_result": dict(vehicle.get("proposal_result") or {}),
        "scenario_lineage": dict(vehicle.get("scenario_lineage") or {}),
        "warnings": list(result.warnings),
    }


def run_fuel_estimation(request: FuelEstimateRequest | dict[str, Any]) -> FuelEstimateResult:
    req = _coerce_request(request)
    method = _clean_text(req.method, "physics_simple", upper=False) or "physics_simple"
    vde_mj_per_km, basis_used, _phase_inputs, basis_warnings = _resolve_energy_basis(req)

    outputs = {"fuel_l_100km": None, "energy_Wh_km": None, "gco2_km": None}
    assumptions: dict[str, Any] = {}
    warnings = list(basis_warnings)
    confidence = "low"

    if method == "manual_imported":
        outputs, assumptions, method_warnings = _manual_imported(req)
        warnings.extend(method_warnings)
        confidence = "provided"
    elif method == "physics_simple":
        outputs, assumptions, method_warnings = _physics_simple(req, vde_mj_per_km)
        warnings.extend(method_warnings)
        confidence = "medium" if not method_warnings else "low"
    elif method == "regression_existing":
        outputs, assumptions, method_warnings, confidence = _regression_existing(req, vde_mj_per_km)
        warnings.extend(method_warnings)
    elif method == "ml_prediction":
        outputs, assumptions, method_warnings, confidence = _ml_prediction(req)
        warnings.extend(method_warnings)
    else:
        warnings.append(f"unsupported_method:{method}")

    assumptions["pse_summary"] = build_powertrain_efficiency_summary(
        request=req,
        method=method,
        energy_basis_used=basis_used,
        fuel_l_100km=outputs.get("fuel_l_100km"),
        energy_Wh_km=outputs.get("energy_Wh_km"),
        assumptions=assumptions,
    )
    assumptions["confidence_summary"] = build_estimate_confidence_summary(
        request=req,
        method=method,
        confidence=confidence,
        warnings=warnings,
        assumptions=assumptions,
    )

    resolved_phase_outputs = _phase_outputs_for_result(
        req,
        method=method,
        outputs=outputs,
        assumptions=assumptions,
    )

    return FuelEstimateResult(
        request=req,
        method=method,
        energy_basis_used=basis_used,
        fuel_l_100km=outputs["fuel_l_100km"],
        energy_Wh_km=outputs["energy_Wh_km"],
        gco2_km=outputs["gco2_km"],
        confidence=confidence,
        warnings=warnings,
        assumptions=assumptions,
        phase_outputs=resolved_phase_outputs,
    )


def _build_fuelcons_payload(result: FuelEstimateResult) -> dict[str, Any]:
    req = result.request
    vehicle = dict(req.vehicle_features or {})
    powertrain = dict(req.powertrain_features or {})
    utility_factor = _to_float(powertrain.get("utility_factor"))
    method_note = result.method
    if result.method == "physics_simple":
        method_note = f"physics_simple [{result.energy_basis_used}]"
    elif result.method == "manual_imported":
        source = _clean_text(result.assumptions.get("source"))
        method_note = f"manual_imported [{source}]" if source else "manual_imported"
    elif result.method == "regression_existing":
        method_note = f"regression_existing [{result.energy_basis_used}]"
    elif result.method == "ml_prediction":
        method_note = f"ml_prediction [{result.energy_basis_used}]"

    payload = {
        "vde_id": req.vde_id,
        "electrification": vehicle.get("electrification"),
        "fuel_type": powertrain.get("fuel_type"),
        "eta_pt_est": powertrain.get("eta_pt_est"),
        "bev_eff_drive": powertrain.get("bev_eff_drive"),
        "utility_factor_pct": (utility_factor * 100.0) if utility_factor is not None else None,
        "engine_max_power_kw": powertrain.get("engine_max_power_kw"),
        "gear_count": powertrain.get("gear_count") or vehicle.get("gear_count"),
        "final_drive_ratio": powertrain.get("final_drive_ratio") or vehicle.get("final_drive_ratio"),
        "energy_Wh_per_km": result.energy_Wh_km,
        "fuel_l_per_100km": result.fuel_l_100km,
        "gco2_per_km": result.gco2_km,
        "method_note": method_note,
        "energy_basis": result.energy_basis_used,
        "engine_method": result.method,
        "engine_version": ENGINE_VERSION,
        "source_vde_revision": vehicle.get("source_vde_revision"),
        "assumptions_json": _json_dumps(result.assumptions),
        "provenance_json": _json_dumps(_build_provenance_payload(result)),
    }
    payload.update(dict(result.phase_outputs or {}))
    return {key: value for key, value in payload.items() if value not in (None, "")}


def build_fuel_scenario_save_payload(
    result: FuelEstimateResult | dict[str, Any],
    *,
    extra_payload: dict[str, Any] | None = None,
) -> FuelScenarioSavePayload:
    fuel_result = result if isinstance(result, FuelEstimateResult) else run_fuel_estimation(result)
    payload = _build_fuelcons_payload(fuel_result)
    if extra_payload:
        payload.update({key: value for key, value in dict(extra_payload).items() if value not in (None, "")})
    return FuelScenarioSavePayload(
        result=fuel_result,
        payload=payload,
        data_origin=_data_origin_for_result(fuel_result),
    )


def save_fuel_estimate_result(
    result: FuelEstimateResult | dict[str, Any],
    save_mode: str,
    *,
    row_id: int | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    staged = build_fuel_scenario_save_payload(result, extra_payload=extra_payload)
    mode = _clean_text(save_mode, "", upper=False) or ""
    payload = dict(staged.payload)

    if mode in INSERT_MODES:
        inserted_id = int(insert_fuelcons_row(payload))
        return {
            "action": "insert_new",
            "row_id": inserted_id,
            "payload": payload,
        }

    if mode in UPDATE_MODES:
        if row_id is None:
            raise ValueError("update_existing requires row_id")
        update_fuelcons_by_id(int(row_id), payload)
        return {
            "action": "update_existing",
            "row_id": int(row_id),
            "payload": payload,
        }

    if mode in DELETE_MODES:
        if row_id is None:
            raise ValueError("delete_existing requires row_id")
        deleted = int(delete_fuelcons_by_id(int(row_id)))
        return {
            "action": "delete_existing",
            "row_id": int(row_id),
            "deleted_rows": deleted,
        }

    raise ValueError(f"Unsupported save_mode: {save_mode!r}")


__all__ = [
    "FuelEstimateRequest",
    "FuelEstimateResult",
    "FuelScenarioSavePayload",
    "build_fuel_scenario_save_payload",
    "run_fuel_estimation",
    "save_fuel_estimate_result",
]
