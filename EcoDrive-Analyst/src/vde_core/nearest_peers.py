from __future__ import annotations

import math
from statistics import median
from typing import Any

import pandas as pd

from src.vde_core.db import fetchall


NUMERIC_FEATURES = [
    "mass_kg",
    "engine_max_power_kw",
    "engine_size_l",
    "gear_count",
    "final_drive_ratio",
    "coast_A_N",
    "coast_B_N_per_kph",
    "coast_C_N_per_kph2",
    "vde_total_mj_per_km",
    "vde_net_mj_per_km",
    "year",
]

CATEGORICAL_FEATURES = [
    "category",
    "fuel_type",
    "electrification",
    "transmission_type",
    "drive_type",
    "make",
]

PEER_METRICS = {
    "fuel_l_per_100km": "Fuel [L/100km]",
    "gco2_per_km": "CO2 [g/km]",
    "energy_Wh_per_km": "Energy [Wh/km]",
    "vde_total_mj_per_km": "VDE_TOTAL [MJ/km]",
    "vde_net_mj_per_km": "VDE_NET [MJ/km]",
    "fuel_ftp75_l_per_100km": "Urban Fuel [L/100km]",
    "fuel_hwfet_l_per_100km": "Highway Fuel [L/100km]",
    "energy_ftp75_Wh_per_km": "Urban Energy [Wh/km]",
    "energy_hwfet_Wh_per_km": "Highway Energy [Wh/km]",
}


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


def load_peer_candidates(filters: dict[str, Any] | None = None) -> pd.DataFrame:
    base = """
        SELECT
            f.id,
            f.created_at,
            f.vde_id,
            f.electrification,
            f.fuel_type,
            f.engine_method,
            f.engine_max_power_kw,
            f.gear_count,
            f.final_drive_ratio,
            f.energy_Wh_per_km,
            f.fuel_l_per_100km,
            f.gco2_per_km,
            f.fuel_ftp75_l_per_100km,
            f.fuel_hwfet_l_per_100km,
            f.energy_ftp75_Wh_per_km,
            f.energy_hwfet_Wh_per_km,
            v.make,
            v.model,
            v.year,
            v.category,
            v.legislation,
            v.mass_kg,
            v.engine_size_l,
            v.transmission_type,
            v.drive_type,
            v.coast_A_N,
            v.coast_B_N_per_kph,
            v.coast_C_N_per_kph2,
            v.vde_total_mj_per_km,
            v.vde_net_mj_per_km
        FROM fuelcons_db f
        JOIN vde_db v ON v.id = f.vde_id
        WHERE 1 = 1
    """
    params: list[Any] = []
    active_filters = dict(filters or {})
    if active_filters.get("legislation"):
        base += " AND v.legislation = ?"
        params.append(str(active_filters["legislation"]))
    if active_filters.get("electrification"):
        base += " AND f.electrification = ?"
        params.append(str(active_filters["electrification"]))
    if active_filters.get("category"):
        base += " AND v.category = ?"
        params.append(str(active_filters["category"]))
    if active_filters.get("make"):
        base += " AND v.make = ?"
        params.append(str(active_filters["make"]))
    if active_filters.get("exclude_vde_id") not in (None, ""):
        base += " AND f.vde_id <> ?"
        params.append(int(active_filters["exclude_vde_id"]))
    base += " ORDER BY f.created_at DESC"
    return pd.DataFrame(fetchall(base, tuple(params)))


def build_target_peer_row(request: Any, outputs: dict[str, Any] | None = None) -> dict[str, Any]:
    vehicle = dict(getattr(request, "vehicle_features", {}) or {})
    powertrain = dict(getattr(request, "powertrain_features", {}) or {})
    phases = dict(vehicle.get("phase_outputs") or {})
    active_outputs = dict(outputs or {})
    return {
        "vde_id": getattr(request, "vde_id", None),
        "category": vehicle.get("category"),
        "make": vehicle.get("make"),
        "model": vehicle.get("model"),
        "year": vehicle.get("year"),
        "mass_kg": vehicle.get("mass_kg") or vehicle.get("test_mass_kg"),
        "engine_size_l": vehicle.get("engine_size_l"),
        "engine_max_power_kw": powertrain.get("engine_max_power_kw") or vehicle.get("engine_max_power_kw"),
        "transmission_type": vehicle.get("transmission_type"),
        "drive_type": vehicle.get("drive_type"),
        "electrification": vehicle.get("electrification"),
        "fuel_type": powertrain.get("fuel_type"),
        "gear_count": powertrain.get("gear_count") or vehicle.get("gear_count"),
        "final_drive_ratio": powertrain.get("final_drive_ratio") or vehicle.get("final_drive_ratio"),
        "coast_A_N": vehicle.get("coast_A_N"),
        "coast_B_N_per_kph": vehicle.get("coast_B_N_per_kph"),
        "coast_C_N_per_kph2": vehicle.get("coast_C_N_per_kph2"),
        "vde_total_mj_per_km": vehicle.get("vde_total_mj_per_km"),
        "vde_net_mj_per_km": vehicle.get("vde_net_mj_per_km"),
        "fuel_l_per_100km": active_outputs.get("fuel_l_100km"),
        "gco2_per_km": active_outputs.get("gco2_km"),
        "energy_Wh_per_km": active_outputs.get("energy_Wh_km"),
        "fuel_ftp75_l_per_100km": active_outputs.get("fuel_l_per_100km_urb"),
        "fuel_hwfet_l_per_100km": active_outputs.get("fuel_l_per_100km_hw"),
        "energy_ftp75_Wh_per_km": active_outputs.get("energy_Wh_km_urb"),
        "energy_hwfet_Wh_per_km": active_outputs.get("energy_Wh_km_hw"),
        "vde_urb_mj_per_km": phases.get("vde_urb_mj_per_km"),
        "vde_hw_mj_per_km": phases.get("vde_hw_mj_per_km"),
    }


def build_peer_feature_matrix(df: pd.DataFrame, feature_config: dict[str, Any] | None = None) -> dict[str, Any]:
    features = dict(feature_config or {})
    numeric_features = list(features.get("numeric") or NUMERIC_FEATURES)
    categorical_features = list(features.get("categorical") or CATEGORICAL_FEATURES)
    work = df.copy() if df is not None else pd.DataFrame()
    numeric_stats: dict[str, dict[str, float]] = {}

    for col in numeric_features:
        if col not in work.columns:
            continue
        series = pd.to_numeric(work[col], errors="coerce")
        mean_val = float(series.mean()) if series.notna().any() else 0.0
        std_val = float(series.std(ddof=0)) if series.notna().any() else 0.0
        numeric_stats[col] = {"mean": mean_val, "std": std_val if std_val > 1e-12 else 1.0}
        work[col] = series

    return {
        "frame": work,
        "numeric": numeric_features,
        "categorical": categorical_features,
        "numeric_stats": numeric_stats,
    }


def find_nearest_peers(
    target_row: dict[str, Any],
    candidate_df: pd.DataFrame,
    n: int = 5,
    feature_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if candidate_df is None or candidate_df.empty:
        return {"peers": [], "warnings": ["peer_dataset_empty"], "feature_coverage": {}}

    matrix = build_peer_feature_matrix(candidate_df, feature_config)
    work = matrix["frame"]
    numeric_features = matrix["numeric"]
    categorical_features = matrix["categorical"]
    numeric_stats = matrix["numeric_stats"]
    rows: list[dict[str, Any]] = []
    coverage: dict[str, int] = {}

    for _, candidate in work.iterrows():
        total_distance = 0.0
        used_features = 0
        compared_features: list[str] = []

        for feature in numeric_features:
            target_value = _to_float(target_row.get(feature))
            candidate_value = _to_float(candidate.get(feature))
            if target_value is None or candidate_value is None:
                continue
            stats = numeric_stats.get(feature, {"mean": 0.0, "std": 1.0})
            distance = abs(candidate_value - target_value) / max(stats["std"], 1e-9)
            total_distance += distance
            used_features += 1
            compared_features.append(feature)
            coverage[feature] = coverage.get(feature, 0) + 1

        for feature in categorical_features:
            target_value = _clean_text(target_row.get(feature), upper=True)
            candidate_value = _clean_text(candidate.get(feature), upper=True)
            if target_value is None or candidate_value is None:
                continue
            total_distance += 0.0 if target_value == candidate_value else 1.0
            used_features += 1
            compared_features.append(feature)
            coverage[feature] = coverage.get(feature, 0) + 1

        if used_features == 0:
            continue
        avg_distance = total_distance / used_features
        similarity = 1.0 / (1.0 + avg_distance)
        row = candidate.to_dict()
        row["peer_distance"] = avg_distance
        row["peer_similarity"] = similarity
        row["peer_features_used"] = used_features
        row["peer_compared_features"] = compared_features
        rows.append(row)

    if not rows:
        return {"peers": [], "warnings": ["peer_feature_coverage_insufficient"], "feature_coverage": coverage}

    ranked = sorted(rows, key=lambda item: (item["peer_distance"], -item["peer_similarity"]))[: max(1, int(n))]
    warnings: list[str] = []
    if len(ranked) < n:
        warnings.append("peer_count_below_requested")
    if len(coverage) < 3:
        warnings.append("peer_feature_coverage_sparse")
    return {"peers": ranked, "warnings": warnings, "feature_coverage": coverage}


def _iqr(values: list[float]) -> float | None:
    if len(values) < 4:
        return None
    sorted_values = sorted(values)
    q1 = sorted_values[len(sorted_values) // 4]
    q3 = sorted_values[(len(sorted_values) * 3) // 4]
    return float(q3 - q1)


def summarize_peer_comparison(target: dict[str, Any], peers: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    dispersion_scores: list[float] = []

    for metric, label in PEER_METRICS.items():
        peer_values = [_to_float(peer.get(metric)) for peer in peers]
        values = [value for value in peer_values if value is not None]
        if not values:
            continue
        mean_val = float(sum(values) / len(values))
        median_val = float(median(values))
        if len(values) > 1:
            variance = sum((value - mean_val) ** 2 for value in values) / len(values)
            std_val = math.sqrt(variance)
        else:
            std_val = 0.0
        min_val = float(min(values))
        max_val = float(max(values))
        iqr_val = _iqr(values)
        scenario_value = _to_float(target.get(metric))
        delta_vs_median = scenario_value - median_val if scenario_value is not None else None
        z_score = None
        if scenario_value is not None and std_val > 1e-12:
            z_score = (scenario_value - mean_val) / std_val
        relative_dispersion = std_val / max(abs(median_val), 1e-9)
        dispersion_scores.append(relative_dispersion)
        metric_rows.append(
            {
                "metric": metric,
                "label": label,
                "scenario_value": scenario_value,
                "mean": mean_val,
                "median": median_val,
                "std_dev": std_val,
                "min": min_val,
                "max": max_val,
                "iqr": iqr_val,
                "delta_vs_median": delta_vs_median,
                "z_score": z_score,
            }
        )

    quality = classify_peer_group_quality(len(peers), dispersion_scores)
    return {
        "peer_count": len(peers),
        "metrics": metric_rows,
        "quality": quality,
    }


def classify_peer_group_quality(peer_count: int, dispersion_scores: list[float]) -> dict[str, Any]:
    if peer_count < 4:
        return {"label": "Low confidence", "reason": "Few similar peers available."}
    avg_dispersion = float(sum(dispersion_scores) / len(dispersion_scores)) if dispersion_scores else 1.0
    if peer_count >= 6 and avg_dispersion <= 0.15:
        return {"label": "High confidence", "reason": "Peer group is sufficiently populated and shows low dispersion."}
    if avg_dispersion <= 0.35:
        return {"label": "Medium confidence", "reason": "Peer group is usable, but dispersion is moderate."}
    return {"label": "Low confidence", "reason": "Peer group is heterogeneous or highly dispersed."}


def generate_investigation_hints(
    target: dict[str, Any],
    peer_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    metrics = {row["metric"]: row for row in peer_summary.get("metrics") or []}

    fuel_row = metrics.get("fuel_l_per_100km")
    energy_row = metrics.get("energy_Wh_per_km")
    vde_net_row = metrics.get("vde_net_mj_per_km")
    vde_total_row = metrics.get("vde_total_mj_per_km")
    urban_fuel = metrics.get("fuel_ftp75_l_per_100km")
    highway_fuel = metrics.get("fuel_hwfet_l_per_100km")
    urban_energy = metrics.get("energy_ftp75_Wh_per_km")
    highway_energy = metrics.get("energy_hwfet_Wh_per_km")

    if fuel_row and fuel_row["scenario_value"] is not None and fuel_row["delta_vs_median"] is not None:
        if fuel_row["delta_vs_median"] > max(fuel_row["std_dev"], 0.4):
            hints.append(
                {
                    "hint": "Fuel is worse than similar peers.",
                    "evidence": f"Scenario is {fuel_row['delta_vs_median']:.2f} L/100km above peer median.",
                    "next_data": "Investigate powertrain efficiency, thermal strategy, auxiliaries, and calibration assumptions.",
                }
            )

    if vde_net_row and vde_net_row["scenario_value"] is not None and vde_net_row["delta_vs_median"] is not None:
        if vde_net_row["delta_vs_median"] > max(vde_net_row["std_dev"], 0.08):
            hints.append(
                {
                    "hint": "Roadload/VDE appears high versus peers.",
                    "evidence": f"VDE_NET is {vde_net_row['delta_vs_median']:.3f} MJ/km above peer median.",
                    "next_data": "Check tire package, Cd/CdA, coastdown terms, mass, and wheel alignment assumptions.",
                }
            )

    if vde_total_row and vde_net_row and vde_total_row["scenario_value"] is not None and vde_net_row["scenario_value"] is not None:
        total_minus_net = vde_total_row["scenario_value"] - vde_net_row["scenario_value"]
        if total_minus_net > 0.12:
            hints.append(
                {
                    "hint": "TOTAL is significantly above NET.",
                    "evidence": f"VDE_TOTAL - VDE_NET = {total_minus_net:.3f} MJ/km.",
                    "next_data": "Review transmission losses, neutral drag, gearing, and driveline assumptions.",
                }
            )

    if urban_fuel and highway_fuel and urban_fuel["scenario_value"] is not None and highway_fuel["scenario_value"] is not None:
        if highway_fuel["scenario_value"] > urban_fuel["scenario_value"] * 1.02:
            hints.append(
                {
                    "hint": "Highway fuel penalty is stronger than urban penalty.",
                    "evidence": "Highway fuel estimate is above urban fuel estimate for the current scenario.",
                    "next_data": "Investigate aero, tire package, gear ratios, and highway roadload assumptions.",
                }
            )

    if urban_energy and highway_energy and urban_energy["scenario_value"] is not None and highway_energy["scenario_value"] is not None:
        if highway_energy["scenario_value"] > urban_energy["scenario_value"] * 1.02:
            hints.append(
                {
                    "hint": "Highway energy penalty is stronger than urban penalty.",
                    "evidence": "Highway energy estimate is above urban energy estimate for the current scenario.",
                    "next_data": "Investigate aero, tire package, and high-speed efficiency assumptions.",
                }
            )

    if fuel_row and vde_net_row and fuel_row["scenario_value"] is not None and vde_net_row["scenario_value"] is not None:
        if fuel_row.get("z_score") is not None and fuel_row["z_score"] > 1.5:
            if vde_net_row["delta_vs_median"] is not None and abs(vde_net_row["delta_vs_median"]) <= max(vde_net_row["std_dev"], 0.05):
                hints.append(
                    {
                        "hint": "Fuel is weak while VDE is near peer median.",
                        "evidence": "Consumption deviates more than roadload from similar scenarios.",
                        "next_data": "Investigate powertrain efficiency, calibration, thermal strategy, and auxiliary loads.",
                    }
                )

    return hints


def build_peer_analysis_for_request(
    request: Any,
    outputs: dict[str, Any] | None = None,
    *,
    n: int = 5,
) -> dict[str, Any]:
    vehicle = dict(getattr(request, "vehicle_features", {}) or {})
    filters = {
        "legislation": vehicle.get("legislation"),
        "electrification": vehicle.get("electrification"),
        "category": vehicle.get("category"),
        "exclude_vde_id": getattr(request, "vde_id", None),
    }
    peer_df = load_peer_candidates(filters)
    target_peer_row = build_target_peer_row(request, outputs=outputs)
    peer_state = find_nearest_peers(target_peer_row, peer_df, n=n)
    peer_summary = summarize_peer_comparison(target_peer_row, peer_state.get("peers") or [])
    hints = generate_investigation_hints(target_peer_row, peer_summary)
    return {
        "target": target_peer_row,
        "peers": list(peer_state.get("peers") or []),
        "warnings": list(peer_state.get("warnings") or []),
        "feature_coverage": dict(peer_state.get("feature_coverage") or {}),
        "summary": dict(peer_summary),
        "quality": dict(peer_summary.get("quality") or {}),
        "hints": list(hints),
    }
