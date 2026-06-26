from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

from src.vde_core.fuel_energy import GCO2_PER_L
from src.vde_core.ml_explainability import compute_ml_explanation
from src.vde_core.nearest_peers import (
    build_peer_analysis_for_request,
    load_peer_candidates,
)


ML_NOTEBOOK_RELATIVE_PATH = "notebooks/ML_Regression_VDE.ipynb"
ML_ARTIFACT_GLOBS = (
    "models/**/*.joblib",
    "models/**/*.pkl",
    "models/**/*.pickle",
    "artifacts/**/*.joblib",
    "artifacts/**/*.pkl",
    "artifacts/**/*.pickle",
    "notebooks/**/*.joblib",
    "notebooks/**/*.pkl",
    "notebooks/**/*.pickle",
)

ML_NOTEBOOK_SUMMARY = {
    "notebook_path": ML_NOTEBOOK_RELATIVE_PATH,
    "source_status": "notebook_inspected",
    "feature_sets": {
        "base_roadload": [
            "category",
            "make",
            "year",
            "engine_size_l",
            "transmission_type",
            "drive_type",
            "electrification",
            "gear_count",
            "final_drive_ratio",
        ],
        "fuel_energy_extension": [
            "coast_A_N",
            "coast_B_N_per_kph",
            "coast_C_N_per_kph2",
            "vde_net_mj_per_km",
            "vde_urb_mj_per_km",
            "vde_hw_mj_per_km",
        ],
    },
    "preprocessing": {
        "numeric": [
            "engine_size_l",
            "gear_count",
            "final_drive_ratio",
            "year",
            "coast_A_N",
            "coast_B_N_per_kph",
            "coast_C_N_per_kph2",
            "vde_net_mj_per_km",
            "vde_urb_mj_per_km",
            "vde_hw_mj_per_km",
        ],
        "categorical": [
            "category",
            "make",
            "transmission_type",
            "drive_type",
            "electrification",
        ],
        "pipeline": "ColumnTransformer(StandardScaler + OneHotEncoder(handle_unknown='ignore'))",
    },
    "targets_by_electrification": {
        "BEV": ["energy_Wh_per_km", "gco2_per_km"],
        "ICE": ["fuel_l_per_100km", "gco2_per_km"],
        "HEV": ["fuel_l_per_100km", "gco2_per_km"],
        "MHEV": ["fuel_l_per_100km", "gco2_per_km"],
        "PHEV": ["fuel_l_per_100km", "energy_Wh_per_km", "gco2_per_km"],
        "DEFAULT": ["fuel_l_per_100km", "energy_Wh_per_km", "gco2_per_km"],
    },
    "candidate_models": [
        "LinearRegression",
        "RandomForestRegressor",
        "MLPRegressor",
        "XGBRegressor / MultiOutputRegressor",
    ],
}


class NotebookPowertrainPredictor:
    def __init__(self, *, bev_model: Any, nbev_model: Any, metadata: dict[str, Any] | None = None):
        self.bev_model = bev_model
        self.nbev_model = nbev_model
        self.metadata = dict(metadata or {})
        self.feature_columns = list(
            self.metadata.get(
                "feature_columns",
                [
                    "category",
                    "make",
                    "year",
                    "engine_size_l",
                    "transmission_type",
                    "drive_type",
                    "electrification",
                    "gear_count",
                    "final_drive_ratio",
                    "coast_A_N",
                    "coast_B_N_per_kph",
                    "coast_C_N_per_kph2",
                    "vde_net_mj_per_km",
                    "vde_urb_mj_per_km",
                    "vde_hw_mj_per_km",
                ],
            )
        )
        self.target_names = {
            "BEV": list(
                self.metadata.get(
                    "bev_targets",
                    ["energy_ftp75_Wh_per_km", "energy_hwfet_Wh_per_km", "energy_Wh_per_km"],
                )
            ),
            "NBEV": list(
                self.metadata.get(
                    "nbev_targets",
                    ["fuel_ftp75_l_per_100km", "fuel_hwfet_l_per_100km", "fuel_l_per_100km"],
                )
            ),
        }
        self.categorical_features = list(
            self.metadata.get(
                "categorical_features",
                ["category", "make", "transmission_type", "drive_type", "electrification"],
            )
        )
        self.continuous_features = list(
            self.metadata.get(
                "continuous_features",
                [
                    "engine_size_l",
                    "gear_count",
                    "final_drive_ratio",
                    "year",
                    "coast_A_N",
                    "coast_B_N_per_kph",
                    "coast_C_N_per_kph2",
                    "vde_net_mj_per_km",
                    "vde_urb_mj_per_km",
                    "vde_hw_mj_per_km",
                ],
            )
        )

    def _kind(self, feature_row: dict[str, Any]) -> str:
        electrification = _clean_text(feature_row.get("electrification"), "ICE", upper=True) or "ICE"
        return "BEV" if electrification == "BEV" else "NBEV"

    def _build_frame(self, feature_row: dict[str, Any]):
        import pandas as pd

        row = {name: feature_row.get(name) for name in self.feature_columns}
        return pd.DataFrame([row], columns=self.feature_columns)

    def _derive_outputs(
        self,
        *,
        kind: str,
        prediction_map: dict[str, float],
        request_dict: dict[str, Any],
        feature_row: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        warnings: list[str] = []
        outputs = {
            "fuel_l_100km": None,
            "energy_Wh_km": None,
            "gco2_km": None,
            "fuel_l_per_100km_urb": None,
            "fuel_l_per_100km_hw": None,
            "energy_Wh_km_urb": None,
            "energy_Wh_km_hw": None,
        }
        powertrain = dict(request_dict.get("powertrain_features") or {})

        if kind == "BEV":
            outputs["energy_Wh_km_urb"] = _to_float(prediction_map.get("energy_ftp75_Wh_per_km"))
            outputs["energy_Wh_km_hw"] = _to_float(prediction_map.get("energy_hwfet_Wh_per_km"))
            outputs["energy_Wh_km"] = _to_float(prediction_map.get("energy_Wh_per_km"))
            grid = _to_float(powertrain.get("grid_gco2_per_kwh"))
            if grid is not None and outputs["energy_Wh_km"] is not None:
                outputs["gco2_km"] = (outputs["energy_Wh_km"] / 1000.0) * grid
            else:
                warnings.append("grid_gco2_per_kwh_missing_for_bev_ml_co2")
            return outputs, warnings

        outputs["fuel_l_per_100km_urb"] = _to_float(prediction_map.get("fuel_ftp75_l_per_100km"))
        outputs["fuel_l_per_100km_hw"] = _to_float(prediction_map.get("fuel_hwfet_l_per_100km"))
        outputs["fuel_l_100km"] = _to_float(prediction_map.get("fuel_l_per_100km"))
        fuel_type = (
            _clean_text(powertrain.get("fuel_type"))
            or _clean_text(feature_row.get("fuel_type"))
            or "Gasoline"
        )
        gco2_per_l = _to_float(powertrain.get("gCO2_per_L"), GCO2_PER_L.get(fuel_type, GCO2_PER_L["Gasoline"]))
        if outputs["fuel_l_100km"] is not None and gco2_per_l is not None:
            outputs["gco2_km"] = (outputs["fuel_l_100km"] / 100.0) * gco2_per_l
        if powertrain.get("fuel_type") in (None, ""):
            warnings.append("fuel_type_missing_defaulted_to_gasoline_for_ml_co2")
        return outputs, warnings

    def _map_contribution_feature(self, encoded_name: str) -> str:
        raw = str(encoded_name)
        for prefix in ("num__", "cat__", "remainder__"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        for feature in self.continuous_features + self.categorical_features:
            if raw == feature or raw.startswith(f"{feature}_"):
                return feature
        return raw

    def _compute_feature_contributions(self, model: Any, frame) -> tuple[dict[str, float], str]:
        try:
            import xgboost as xgb

            preprocessor = model.named_steps.get("prep") or model.named_steps.get("preprocessor")
            regressor = model.named_steps.get("xgb") or model.named_steps.get("regressor")
            if preprocessor is None or regressor is None:
                return {}, "not_available"
            transformed = preprocessor.transform(frame)
            feature_names = list(preprocessor.get_feature_names_out())
            estimator = regressor.estimators_[-1] if hasattr(regressor, "estimators_") else regressor
            booster = estimator.get_booster() if hasattr(estimator, "get_booster") else None
            if booster is None:
                return {}, "not_available"
            contribs = booster.predict(
                xgb.DMatrix(transformed, feature_names=[str(name) for name in feature_names]),
                pred_contribs=True,
            )
            if len(contribs) == 0:
                return {}, "not_available"
            row = contribs[0]
            grouped: dict[str, float] = {}
            for idx, feature_name in enumerate(feature_names):
                if idx >= len(row) - 1:
                    break
                base_name = self._map_contribution_feature(feature_name)
                grouped[base_name] = grouped.get(base_name, 0.0) + float(row[idx])
            return grouped, "available"
        except Exception:
            return {}, "not_available"

    def __call__(self, request_dict: dict[str, Any], feature_row: dict[str, Any], metadata: dict[str, Any] | None = None):
        del metadata
        kind = self._kind(feature_row)
        model = self.bev_model if kind == "BEV" else self.nbev_model
        frame = self._build_frame(feature_row)
        predictions = model.predict(frame)
        values = predictions[0].tolist() if hasattr(predictions[0], "tolist") else list(predictions[0])
        prediction_map = {
            target: _to_float(value)
            for target, value in zip(self.target_names[kind], values)
        }
        outputs, warnings = self._derive_outputs(
            kind=kind,
            prediction_map=prediction_map,
            request_dict=request_dict,
            feature_row=feature_row,
        )
        feature_contributions, shap_status = self._compute_feature_contributions(model, frame)
        model_metrics = dict(self.metadata.get("metrics", {}).get(kind, {}))
        confidence = "high" if model_metrics.get("combined_r2", 0.0) >= 0.92 else "medium"
        return {
            **outputs,
            "model_name": str(self.metadata.get("model_name") or "NotebookPowertrainPredictor"),
            "model_version": str(self.metadata.get("model_version") or "notebook_export"),
            "feature_contributions": feature_contributions,
            "shap_status": shap_status,
            "warnings": warnings,
            "confidence": confidence,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def get_ml_notebook_summary() -> dict[str, Any]:
    return {
        "notebook_path": ML_NOTEBOOK_SUMMARY["notebook_path"],
        "source_status": ML_NOTEBOOK_SUMMARY["source_status"],
        "feature_sets": {
            key: list(values)
            for key, values in ML_NOTEBOOK_SUMMARY["feature_sets"].items()
        },
        "preprocessing": {
            key: list(value) if isinstance(value, list) else value
            for key, value in ML_NOTEBOOK_SUMMARY["preprocessing"].items()
        },
        "targets_by_electrification": {
            key: list(values)
            for key, values in ML_NOTEBOOK_SUMMARY["targets_by_electrification"].items()
        },
        "candidate_models": list(ML_NOTEBOOK_SUMMARY["candidate_models"]),
    }


def find_ml_artifact_paths(repo_root: str | Path | None = None) -> list[str]:
    root = Path(repo_root) if repo_root else _repo_root()
    matches: set[str] = set()
    for pattern in ML_ARTIFACT_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                matches.add(str(path.resolve()))
    return sorted(matches)


def build_ml_features(request: Any, vde_row: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    vehicle = dict(getattr(request, "vehicle_features", {}) or {})
    powertrain = dict(getattr(request, "powertrain_features", {}) or {})
    row = dict(vde_row or {})
    ctx = dict(context or {})

    features = {
        "category": vehicle.get("category") or row.get("category"),
        "make": vehicle.get("make") or row.get("make"),
        "model": vehicle.get("model") or row.get("model"),
        "year": vehicle.get("year") or row.get("year"),
        "engine_size_l": vehicle.get("engine_size_l") or row.get("engine_size_l"),
        "transmission_type": vehicle.get("transmission_type") or row.get("transmission_type"),
        "drive_type": vehicle.get("drive_type") or row.get("drive_type"),
        "electrification": _clean_text(
            vehicle.get("electrification") or ctx.get("electrification") or row.get("engine_type"),
            "ICE",
            upper=True,
        ),
        "gear_count": powertrain.get("gear_count") or vehicle.get("gear_count") or row.get("gear_count"),
        "final_drive_ratio": powertrain.get("final_drive_ratio") or vehicle.get("final_drive_ratio") or row.get("final_drive_ratio"),
        "coast_A_N": vehicle.get("coast_A_N") or row.get("coast_A_N"),
        "coast_B_N_per_kph": vehicle.get("coast_B_N_per_kph") or row.get("coast_B_N_per_kph"),
        "coast_C_N_per_kph2": vehicle.get("coast_C_N_per_kph2") or row.get("coast_C_N_per_kph2"),
        "vde_net_mj_per_km": vehicle.get("vde_net_mj_per_km") or row.get("vde_net_mj_per_km"),
        "vde_urb_mj_per_km": (
            dict(vehicle.get("phase_outputs") or {}).get("vde_urb_mj_per_km")
            or vehicle.get("vde_urb_mj_per_km")
            or row.get("vde_urb_mj_per_km")
        ),
        "vde_hw_mj_per_km": (
            dict(vehicle.get("phase_outputs") or {}).get("vde_hw_mj_per_km")
            or vehicle.get("vde_hw_mj_per_km")
            or row.get("vde_hw_mj_per_km")
        ),
    }

    missing_features = [name for name, value in features.items() if value in (None, "")]
    available_features = {
        name: value for name, value in features.items() if value not in (None, "")
    }
    return {
        "features": features,
        "available_features": available_features,
        "available_feature_names": list(available_features.keys()),
        "missing_features": missing_features,
    }


def load_ml_predictor(
    model_artifact_path: str | None = None,
    predictor: Any | None = None,
) -> dict[str, Any]:
    if callable(predictor):
        return {
            "status": "available",
            "predictor": predictor,
            "artifact_path": None,
            "artifact_candidates": [],
            "model_name": getattr(predictor, "__name__", predictor.__class__.__name__),
            "model_version": "injected_runtime_predictor",
            "warnings": [],
        }

    artifact_candidates = [model_artifact_path] if model_artifact_path else find_ml_artifact_paths()
    artifact_candidates = [str(Path(path)) for path in artifact_candidates if path]
    if not artifact_candidates:
        return {
            "status": "export_pending",
            "predictor": None,
            "artifact_path": None,
            "artifact_candidates": [],
            "model_name": None,
            "model_version": None,
            "warnings": ["ml_inference_artifact_missing"],
        }

    artifact_path = Path(artifact_candidates[0])
    try:
        if artifact_path.suffix.lower() == ".joblib":
            import joblib  # type: ignore

            loaded = joblib.load(artifact_path)
        else:
            with artifact_path.open("rb") as fh:
                loaded = pickle.load(fh)
    except Exception as exc:
        return {
            "status": "artifact_load_failed",
            "predictor": None,
            "artifact_path": str(artifact_path),
            "artifact_candidates": artifact_candidates,
            "model_name": None,
            "model_version": None,
            "warnings": [f"ml_artifact_load_failed:{type(exc).__name__}"],
        }

    predictor_obj = loaded.get("predictor") if isinstance(loaded, dict) and "predictor" in loaded else loaded
    model_name = (
        loaded.get("model_name")
        if isinstance(loaded, dict) and loaded.get("model_name")
        else predictor_obj.__class__.__name__
    )
    model_version = loaded.get("model_version") if isinstance(loaded, dict) else None
    return {
        "status": "available",
        "predictor": predictor_obj,
        "artifact_path": str(artifact_path),
        "artifact_candidates": artifact_candidates,
        "model_name": model_name,
        "model_version": model_version,
        "metadata": dict(loaded.get("metadata") or {}) if isinstance(loaded, dict) else {},
        "warnings": [],
    }


def _normalize_ml_outputs(raw_prediction: Any) -> dict[str, Any]:
    data = dict(raw_prediction or {}) if isinstance(raw_prediction, dict) else {}
    return {
        "fuel_l_100km": _to_float(data.get("fuel_l_100km", data.get("fuel_l_per_100km"))),
        "energy_Wh_km": _to_float(data.get("energy_Wh_km", data.get("energy_Wh_per_km"))),
        "gco2_km": _to_float(data.get("gco2_km", data.get("gco2_per_km"))),
        "fuel_l_per_100km_urb": _to_float(data.get("fuel_l_per_100km_urb", data.get("fuel_ftp75_l_per_100km"))),
        "fuel_l_per_100km_hw": _to_float(data.get("fuel_l_per_100km_hw", data.get("fuel_hwfet_l_per_100km"))),
        "energy_Wh_km_urb": _to_float(data.get("energy_Wh_km_urb", data.get("energy_ftp75_Wh_per_km"))),
        "energy_Wh_km_hw": _to_float(data.get("energy_Wh_km_hw", data.get("energy_hwfet_Wh_per_km"))),
        "gco2_km_urb": _to_float(data.get("gco2_km_urb", data.get("gco2_ftp75_per_km"))),
        "gco2_km_hw": _to_float(data.get("gco2_km_hw", data.get("gco2_hwfet_per_km"))),
    }


def _evaluate_domain_coverage(
    feature_row: dict[str, Any],
    peer_df,
) -> dict[str, Any]:
    if peer_df is None or peer_df.empty:
        return {
            "status": "metadata_unavailable",
            "warnings": ["training_domain_metadata_unavailable"],
            "details": [],
        }

    details: list[dict[str, Any]] = []
    warnings: list[str] = []
    out_of_domain_count = 0
    comparable_checks = 0

    for feature in ("mass_kg", "engine_max_power_kw", "engine_size_l", "vde_total_mj_per_km", "vde_net_mj_per_km", "year"):
        if feature not in peer_df.columns:
            continue
        target_value = _to_float(feature_row.get(feature))
        series = peer_df[feature].apply(_to_float).dropna()
        if target_value is None or series.empty:
            continue
        comparable_checks += 1
        min_val = float(series.min())
        max_val = float(series.max())
        in_domain = min_val <= target_value <= max_val
        if not in_domain:
            out_of_domain_count += 1
            warnings.append(f"{feature}_outside_training_range")
        details.append(
            {
                "feature": feature,
                "target_value": target_value,
                "min": min_val,
                "max": max_val,
                "in_domain": in_domain,
            }
        )

    for feature in ("category", "fuel_type", "electrification", "transmission_type", "drive_type", "make"):
        if feature not in peer_df.columns:
            continue
        target_value = _clean_text(feature_row.get(feature), upper=True)
        series = peer_df[feature].dropna().astype(str).str.strip().str.upper()
        if target_value is None or series.empty:
            continue
        comparable_checks += 1
        seen_values = set(series.tolist())
        in_domain = target_value in seen_values
        if not in_domain:
            out_of_domain_count += 1
            warnings.append(f"{feature}_unseen_in_reference_set")
        details.append(
            {
                "feature": feature,
                "target_value": target_value,
                "seen_values_sample": sorted(list(seen_values))[:10],
                "in_domain": in_domain,
            }
        )

    if comparable_checks == 0:
        return {
            "status": "metadata_unavailable",
            "warnings": ["training_domain_metadata_unavailable"],
            "details": details,
        }
    if out_of_domain_count == 0:
        status = "in_domain"
    elif out_of_domain_count <= max(1, comparable_checks // 4):
        status = "partial_domain"
    else:
        status = "out_of_domain"
    return {
        "status": status,
        "warnings": warnings,
        "details": details,
    }


def describe_ml_prediction_setup(
    request: Any,
    *,
    model_artifact_path: str | None = None,
    predictor: Any | None = None,
) -> dict[str, Any]:
    notebook = get_ml_notebook_summary()
    features = build_ml_features(request)
    loader = load_ml_predictor(model_artifact_path=model_artifact_path, predictor=predictor)
    electrification = _clean_text(
        features["features"].get("electrification"),
        "DEFAULT",
        upper=True,
    ) or "DEFAULT"
    targets = notebook["targets_by_electrification"].get(
        electrification,
        notebook["targets_by_electrification"]["DEFAULT"],
    )
    warnings = list(loader.get("warnings") or [])
    if loader["status"] == "export_pending":
        warnings.append("ml_notebook_exists_but_no_exported_inference_artifact_found")
    if loader["status"] != "available":
        warnings.append("training_domain_metadata_unavailable")
    return {
        "status": loader["status"],
        "artifact_path": loader.get("artifact_path"),
        "artifact_candidates": list(loader.get("artifact_candidates") or []),
        "model_name": loader.get("model_name"),
        "model_version": loader.get("model_version"),
        "targets": list(targets),
        "notebook": notebook,
        "features": features,
        "warnings": warnings,
    }


def predict_fuel_with_ml(
    request: Any,
    *,
    model_artifact_path: str | None = None,
    predictor: Any | None = None,
) -> dict[str, Any]:
    setup = describe_ml_prediction_setup(
        request,
        model_artifact_path=model_artifact_path,
        predictor=predictor,
    )
    loader = load_ml_predictor(model_artifact_path=model_artifact_path, predictor=predictor)
    assumptions = {
        "integration_status": setup["status"],
        "notebook_path": setup["notebook"]["notebook_path"],
        "candidate_models": list(setup["notebook"]["candidate_models"]),
        "expected_targets": list(setup["targets"]),
        "features_used": list(setup["features"]["available_feature_names"]),
        "missing_features": list(setup["features"]["missing_features"]),
        "artifact_path": setup["artifact_path"],
        "artifact_candidates": list(setup["artifact_candidates"]),
        "model_name": setup.get("model_name"),
        "model_version": setup.get("model_version"),
        "training_domain_metadata_available": False,
        "shap_status": "pending_artifact",
    }
    warnings = list(setup["warnings"])
    peer_analysis = build_peer_analysis_for_request(request, outputs=None, n=5)
    peer_df = load_peer_candidates(
        {
            "legislation": getattr(request, "vehicle_features", {}).get("legislation"),
            "electrification": getattr(request, "vehicle_features", {}).get("electrification"),
            "category": getattr(request, "vehicle_features", {}).get("category"),
            "exclude_vde_id": getattr(request, "vde_id", None),
        }
    )
    target_peer_row = dict(peer_analysis.get("target") or {})
    peer_state = {
        "peers": list(peer_analysis.get("peers") or []),
        "warnings": list(peer_analysis.get("warnings") or []),
        "feature_coverage": dict(peer_analysis.get("feature_coverage") or {}),
    }
    peer_summary = dict(peer_analysis.get("summary") or {})
    investigation_hints = list(peer_analysis.get("hints") or [])
    coverage = _evaluate_domain_coverage(target_peer_row, peer_df)
    warnings.extend(list(coverage.get("warnings") or []))
    assumptions["coverage_status"] = coverage.get("status", "unknown")
    assumptions["coverage_details"] = list(coverage.get("details") or [])

    if loader["status"] != "available" or not loader.get("predictor"):
        assumptions["nearest_peers"] = list(peer_state.get("peers") or [])
        assumptions["nearest_peer_summary"] = dict(peer_summary)
        assumptions["nearest_peers_available"] = bool(peer_state.get("peers"))
        assumptions["peer_group_quality"] = dict(peer_summary.get("quality") or {})
        assumptions["peer_feature_coverage"] = dict(peer_state.get("feature_coverage") or {})
        assumptions["investigation_hints"] = list(investigation_hints)
        warnings.extend(list(peer_state.get("warnings") or []))
        return build_ml_prediction_result(
            outputs={"fuel_l_100km": None, "energy_Wh_km": None, "gco2_km": None},
            assumptions=assumptions,
            warnings=warnings,
            confidence="low",
        )

    predictor_obj = loader["predictor"]
    feature_row = dict(setup["features"]["features"])
    request_dict = request.to_dict() if hasattr(request, "to_dict") else dict(request or {})
    try:
        if hasattr(predictor_obj, "predict_from_features"):
            raw_prediction = predictor_obj.predict_from_features(feature_row)
        elif callable(predictor_obj):
            raw_prediction = predictor_obj(
                request_dict,
                feature_row,
                {
                    "artifact_path": loader.get("artifact_path"),
                    "notebook": setup["notebook"],
                },
            )
        elif hasattr(predictor_obj, "predict"):
            try:
                import pandas as pd

                raw_prediction = predictor_obj.predict(pd.DataFrame([feature_row]))
            except Exception as exc:
                raise RuntimeError(f"generic_predict_dataframe_failed:{type(exc).__name__}") from exc
        else:
            raise TypeError("ml_predictor_unsupported")
    except Exception as exc:
        warnings.append(f"ml_prediction_failed:{type(exc).__name__}")
        assumptions["shap_status"] = "prediction_failed"
        return build_ml_prediction_result(
            outputs={"fuel_l_100km": None, "energy_Wh_km": None, "gco2_km": None},
            assumptions=assumptions,
            warnings=warnings,
            confidence="low",
        )

    outputs = _normalize_ml_outputs(raw_prediction)
    if isinstance(raw_prediction, dict):
        assumptions["shap_status"] = raw_prediction.get("shap_status", "not_available")
        assumptions["coverage_status"] = raw_prediction.get("coverage_status", assumptions.get("coverage_status", "unknown"))
        assumptions["shap_available"] = bool(raw_prediction.get("feature_contributions"))
        assumptions["feature_contributions"] = dict(raw_prediction.get("feature_contributions") or {})
        assumptions["nearest_peers_available"] = bool(raw_prediction.get("nearest_peers"))
        assumptions["nearest_peers"] = list(raw_prediction.get("nearest_peers") or [])
        assumptions["investigation_hints"] = list(raw_prediction.get("investigation_hints") or [])
        if raw_prediction.get("model_name"):
            assumptions["model_name"] = raw_prediction.get("model_name")
        if raw_prediction.get("model_version"):
            assumptions["model_version"] = raw_prediction.get("model_version")
        warnings.extend(list(raw_prediction.get("warnings") or []))

    explanation = compute_ml_explanation(assumptions.get("feature_contributions"))
    assumptions["shap_status"] = assumptions.get("shap_status") or explanation.get("status")
    assumptions["ml_explanation"] = explanation
    assumptions["shap_available"] = explanation.get("status") == "available"

    if not assumptions.get("nearest_peers"):
        assumptions["nearest_peers"] = list(peer_state.get("peers") or [])
    assumptions["nearest_peer_summary"] = dict(peer_summary)
    assumptions["nearest_peers_available"] = bool(assumptions.get("nearest_peers"))
    assumptions["peer_group_quality"] = dict(peer_summary.get("quality") or {})
    assumptions["peer_feature_coverage"] = dict(peer_state.get("feature_coverage") or {})
    if not assumptions.get("investigation_hints"):
        assumptions["investigation_hints"] = list(investigation_hints)
    warnings.extend(list(peer_state.get("warnings") or []))

    confidence = _clean_text(
        raw_prediction.get("confidence") if isinstance(raw_prediction, dict) else None,
        "medium",
        upper=False,
    ) or "medium"
    return build_ml_prediction_result(
        outputs=outputs,
        assumptions=assumptions,
        warnings=warnings,
        confidence=confidence,
    )


def build_ml_prediction_result(
    *,
    outputs: dict[str, Any],
    assumptions: dict[str, Any],
    warnings: list[str],
    confidence: str,
) -> dict[str, Any]:
    return {
        "outputs": dict(outputs),
        "assumptions": dict(assumptions),
        "warnings": list(dict.fromkeys(warnings)),
        "confidence": confidence,
    }
