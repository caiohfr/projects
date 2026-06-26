from __future__ import annotations

from typing import Any


ENGINEERING_BLOCKS = {
    "Roadload / VDE": {
        "features": {
            "vde_total_mj_per_km",
            "vde_net_mj_per_km",
            "vde_urb_mj_per_km",
            "vde_hw_mj_per_km",
            "coast_A_N",
            "coast_B_N_per_kph",
            "coast_C_N_per_kph2",
        },
        "interpretation": "Roadload and vehicle energy terms are influencing the estimate.",
    },
    "Mass / Vehicle": {
        "features": {
            "mass_kg",
            "test_mass_kg",
            "inertia_class",
            "category",
            "body_type",
            "year",
        },
        "interpretation": "Vehicle size, mass class, and age signals are influencing the estimate.",
    },
    "Powertrain": {
        "features": {
            "fuel_type",
            "electrification",
            "engine_size_l",
            "engine_max_power_kw",
            "eta_pt_est",
            "bev_eff_drive",
            "utility_factor",
        },
        "interpretation": "Powertrain architecture and efficiency assumptions are influencing the estimate.",
    },
    "Transmission": {
        "features": {
            "transmission_type",
            "gear_count",
            "final_drive_ratio",
            "drive_type",
        },
        "interpretation": "Transmission family and gearing are influencing the estimate.",
    },
    "Brand / Model Residual": {
        "features": {
            "make",
            "model",
            "model_family",
        },
        "interpretation": "Brand/model residual signals are present in the model attribution.",
    },
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


def compute_ml_explanation(feature_contributions: dict[str, Any] | None) -> dict[str, Any]:
    contributions = dict(feature_contributions or {})
    if not contributions:
        return {
            "status": "not_available",
            "message": "SHAP not available for this model in the current integration.",
            "grouped_blocks": [],
            "raw_contributions": {},
        }

    grouped_blocks: list[dict[str, Any]] = []
    remaining = {key: _to_float(value, 0.0) or 0.0 for key, value in contributions.items()}

    for block_name, block_cfg in ENGINEERING_BLOCKS.items():
        block_features = []
        block_total = 0.0
        for feature in block_cfg["features"]:
            if feature not in remaining:
                continue
            contrib = remaining.pop(feature)
            block_total += contrib
            block_features.append({"feature": feature, "contribution": contrib})
        if block_features:
            main_features = sorted(block_features, key=lambda item: abs(item["contribution"]), reverse=True)[:3]
            grouped_blocks.append(
                {
                    "engineering_block": block_name,
                    "contribution": block_total,
                    "main_features": [item["feature"] for item in main_features],
                    "interpretation": block_cfg["interpretation"],
                }
            )

    if remaining:
        remainder_features = sorted(
            [{"feature": key, "contribution": value} for key, value in remaining.items()],
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )
        grouped_blocks.append(
            {
                "engineering_block": "Other Signals",
                "contribution": sum(item["contribution"] for item in remainder_features),
                "main_features": [item["feature"] for item in remainder_features[:3]],
                "interpretation": "Additional model features contributed outside the main grouped blocks.",
            }
        )

    grouped_blocks = sorted(grouped_blocks, key=lambda item: abs(item["contribution"]), reverse=True)
    return {
        "status": "available",
        "message": "These are model attribution signals, not proven physical causes.",
        "grouped_blocks": grouped_blocks,
        "raw_contributions": contributions,
    }

