from __future__ import annotations

from typing import Any


_MISSING_INPUT_TOKENS = {
    "vde_total_missing",
    "vde_net_selected_but_unavailable",
    "energy_basis_value_missing",
    "manual_or_imported_energy_missing",
    "manual_or_imported_value_without_source",
    "eta_pt_est_missing",
    "lhv_missing",
    "bev_eff_drive_missing",
    "regression_runner_missing",
    "regression_energy_missing",
    "regression_dataset_empty",
}


def _clean_text(value: Any, default: str | None = None, *, upper: bool = False) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.upper() if upper else text


def _method_status_label(method: str) -> str:
    mapping = {
        "manual_imported": "Measured / Imported",
        "physics_simple": "Physics Estimate",
        "regression_existing": "Regression Estimate",
        "ml_prediction": "ML Prediction",
    }
    return mapping.get(method, method or "Estimate Pending")


def _confidence_label(level: str | None) -> str:
    mapping = {
        "provided": "Provided",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    level_key = _clean_text(level, "low", upper=False) or "low"
    return mapping.get(level_key, level_key.replace("_", " ").title())


def build_estimate_confidence_summary(
    *,
    request: Any,
    method: str,
    confidence: str | None,
    warnings: list[str] | None = None,
    assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings = list(warnings or [])
    assumptions = dict(assumptions or {})
    vehicle_features = dict(getattr(request, "vehicle_features", {}) or {})

    status_items: list[str] = [_method_status_label(method)]
    reasons: list[str] = []

    if set(warnings).intersection(_MISSING_INPUT_TOKENS):
        status_items.append("Missing Critical Inputs")
        reasons.append("critical_inputs")

    energy_basis = _clean_text(getattr(request, "energy_basis", None), "VDE_TOTAL", upper=True) or "VDE_TOTAL"
    if energy_basis == "VDE_NET" and vehicle_features.get("vde_net_mj_per_km") in (None, ""):
        if "Missing Critical Inputs" not in status_items:
            status_items.append("Missing Critical Inputs")
        reasons.append("vde_net_unavailable")

    feature_readiness = dict(vehicle_features.get("scenario_feature_readiness") or {})
    readiness_label = _clean_text(feature_readiness.get("status_label"), None, upper=False)
    if readiness_label:
        status_items.append(str(feature_readiness.get("status_label")))
    if dict(vehicle_features.get("scenario_feature_overrides") or {}):
        status_items.append("Scenario Overrides")
    if list(vehicle_features.get("scenario_feature_imputed") or []):
        status_items.append("Imputed Features")

    if bool(vehicle_features.get("draft_bev_placeholders")):
        status_items.append("Draft Only")
        reasons.append("draft_placeholders")

    pse_summary = dict(assumptions.get("pse_summary") or {})
    if pse_summary.get("value") is not None:
        status_items.append("PSE Available")
    else:
        status_items.append("PSE Unavailable")

    coverage_status = _clean_text(assumptions.get("coverage_status"), "unknown", upper=False) or "unknown"
    if coverage_status == "out_of_domain":
        status_items.append("Out of Domain")
        reasons.append("coverage_out_of_domain")
    elif coverage_status in {"partial_domain", "metadata_unavailable", "unknown"} and method == "ml_prediction":
        status_items.append("Low Coverage")
        reasons.append("coverage_limited")

    shap_status = _clean_text(assumptions.get("shap_status"), None, upper=False)
    shap_available = assumptions.get("shap_available") is True or shap_status == "available"
    if method == "ml_prediction":
        status_items.append("SHAP Available" if shap_available else "SHAP Unavailable")

    peer_quality = _clean_text((assumptions.get("peer_group_quality") or {}).get("label"), None, upper=False) or ""
    if peer_quality.startswith("high"):
        status_items.append("Peer Group High Quality")
    elif peer_quality.startswith("medium"):
        status_items.append("Peer Group Medium Quality")
    elif peer_quality.startswith("low"):
        status_items.append("Peer Group Low Quality")

    deduped_status: list[str] = []
    for item in status_items:
        if item not in deduped_status:
            deduped_status.append(item)

    deduped_reasons: list[str] = []
    for reason in reasons:
        if reason not in deduped_reasons:
            deduped_reasons.append(reason)

    return {
        "level": _clean_text(confidence, "low", upper=False) or "low",
        "label": _confidence_label(confidence),
        "method_status": _method_status_label(method),
        "status_items": deduped_status,
        "reasons": deduped_reasons,
        "warning_count": len(warnings),
    }


__all__ = ["build_estimate_confidence_summary"]
