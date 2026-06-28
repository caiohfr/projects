from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.vde_core.fuel_estimation import FuelEstimateRequest, run_fuel_estimation
from src.vde_core.repositories import (
    delete_fuelcons_by_id,
    fetch_fuelcons_allowed_columns,
    fetch_fuelcons_by_vde_id,
    fetch_fuelcons_distinct_electrifications,
    fetch_fuelcons_join_rows,
    fetch_fuelcons_rows,
    fetch_vde_by_id,
    fetch_vde_by_ids,
    fetch_vde_distinct_categories,
    fetch_vde_distinct_makes,
    fetch_vde_distinct_transmission_models,
    fetch_vde_engine_type,
    fetch_vde_legislation as repo_fetch_vde_legislation,
    insert_fuelcons_row,
    update_fuelcons_by_id,
)
from src.vde_core.vde_setup_service import to_float


def drop_empty(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if v not in (None, "")}


def fetch_vde_row(vde_id: int) -> dict:
    return fetch_vde_by_id(vde_id)


def fetch_distinct_transmission_models() -> list[str]:
    return fetch_vde_distinct_transmission_models()


def fetch_filter_values() -> dict[str, list[str]]:
    categories = fetch_vde_distinct_categories()
    makes = fetch_vde_distinct_makes()
    electrifications = fetch_fuelcons_distinct_electrifications()
    return {
        "categories": categories,
        "makes": makes,
        "electrifications": electrifications,
    }


def default_electrification_from_vde(vde_id: Optional[int]) -> str:
    if not vde_id:
        return "ICE"
    engine_type = fetch_vde_engine_type(vde_id).upper()
    if engine_type == "BEV":
        return "BEV"
    if engine_type == "HEV":
        return "HEV"
    return "ICE"


def build_bev_placeholder_payload() -> dict:
    return drop_empty(
        {
            "engine_model": "",
            "engine_size_l": 0.001,
            "engine_aspiration": "",
            "transmission_type": "SS",
        }
    )


def apply_bev_placeholders(vde_id: int) -> dict:
    del vde_id
    return build_bev_placeholder_payload()


def save_fuelcons_payload(payload: dict) -> int:
    return insert_fuelcons_row(payload)


def update_fuelcons_payload(row_id: int, payload: dict) -> None:
    update_fuelcons_by_id(row_id, payload)


def delete_fuelcons_row(row_id: int) -> int:
    return delete_fuelcons_by_id(row_id)


def compute_vde_total_from_ctx(vde_row: dict, ctx: Dict[str, Any]) -> Dict[str, float]:
    del ctx
    resolved = resolve_vde_energy_values(vde_row)
    return {
        "vde_net_mj_per_km": resolved["vde_net_mj_per_km"],
        "vde_total_mj_per_km": resolved["vde_total_mj_per_km"],
        "warnings": resolved["warnings"],
    }


def resolve_vde_source_revision(vde_row: dict | None) -> str | None:
    row = dict(vde_row or {})
    return (
        row.get("updated_at")
        or row.get("created_at")
        or None
    )


def compare_saved_scenario_revision(saved_revision: str | None, current_vde_row: dict | None) -> dict[str, Any]:
    current_revision = resolve_vde_source_revision(current_vde_row)
    saved_revision = str(saved_revision).strip() if saved_revision not in (None, "") else None

    if current_revision is None:
        return {
            "status": "unknown",
            "saved_revision": saved_revision,
            "current_revision": None,
            "message": "Current VDE revision is unavailable.",
        }
    if saved_revision is None:
        return {
            "status": "missing",
            "saved_revision": None,
            "current_revision": current_revision,
            "message": "Saved scenario has no source VDE revision recorded.",
        }
    if saved_revision == current_revision:
        return {
            "status": "current",
            "saved_revision": saved_revision,
            "current_revision": current_revision,
            "message": "Saved scenario matches the current VDE revision.",
        }
    return {
        "status": "changed",
        "saved_revision": saved_revision,
        "current_revision": current_revision,
        "message": "VDE source changed - Refresh / Recalculate required.",
    }


def resolve_vde_energy_values(vde_row: dict) -> dict[str, Any]:
    row = dict(vde_row or {})
    vde_total = to_float(row.get("vde_total_mj_per_km"))
    vde_net = to_float(row.get("vde_net_mj_per_km"))
    warnings: list[str] = []

    if vde_total is None and vde_net is not None:
        vde_total = vde_net
        warnings.append("legacy_vde_net_used_as_total_candidate")
    if vde_total is None:
        warnings.append("vde_total_missing")
    if vde_net is None:
        warnings.append("vde_net_missing")

    return {
        "vde_total_mj_per_km": vde_total,
        "vde_net_mj_per_km": vde_net,
        "warnings": warnings,
    }


def build_fuel_estimate_request_from_vde(
    vde_row: dict,
    *,
    electrification: str | None = None,
    energy_basis: str = "VDE_TOTAL",
    method: str = "physics_simple",
    powertrain_features: dict | None = None,
    manual_inputs: dict | None = None,
    model_options: dict | None = None,
) -> FuelEstimateRequest:
    row = dict(vde_row or {})
    energy_values = resolve_vde_energy_values(row)
    resolved_electrification = (
        electrification
        or str(row.get("engine_type") or "").upper()
        or "ICE"
    )

    return FuelEstimateRequest(
        vde_id=row.get("id"),
        energy_basis=energy_basis,
        cycle=row.get("cycle_name") or row.get("legislation"),
        vehicle_features={
            "electrification": resolved_electrification,
            "legislation": row.get("legislation"),
            "category": row.get("category"),
            "make": row.get("make"),
            "model": row.get("model"),
            "year": row.get("year"),
            "mass_kg": row.get("mass_kg"),
            "test_mass_kg": row.get("test_mass_kg"),
            "inertia_class": row.get("inertia_class"),
            "engine_size_l": row.get("engine_size_l"),
            "transmission_type": row.get("transmission_type"),
            "drive_type": row.get("drive_type"),
            "gear_count": row.get("gear_count"),
            "final_drive_ratio": row.get("final_drive_ratio"),
            "coast_A_N": row.get("coast_A_N"),
            "coast_B_N_per_kph": row.get("coast_B_N_per_kph"),
            "coast_C_N_per_kph2": row.get("coast_C_N_per_kph2"),
            "source_vde_created_at": row.get("created_at"),
            "source_vde_updated_at": row.get("updated_at"),
            "source_vde_revision": row.get("updated_at") or row.get("created_at"),
            "vde_total_mj_per_km": energy_values["vde_total_mj_per_km"],
            "vde_net_mj_per_km": energy_values["vde_net_mj_per_km"],
            "phase_outputs": {
                "vde_urb_mj_per_km": row.get("vde_urb_mj_per_km"),
                "vde_hw_mj_per_km": row.get("vde_hw_mj_per_km"),
                "vde_low_mj_per_km": row.get("vde_low_mj_per_km"),
                "vde_mid_mj_per_km": row.get("vde_mid_mj_per_km"),
                "vde_high_mj_per_km": row.get("vde_high_mj_per_km"),
                "vde_extra_high_mj_per_km": row.get("vde_extra_high_mj_per_km"),
            },
            "compatibility_warnings": energy_values["warnings"],
        },
        powertrain_features=dict(powertrain_features or {}),
        method=method,
        model_options=dict(model_options or {}),
        manual_inputs=dict(manual_inputs or {}),
    )


def preview_fuel_estimate_from_vde(
    vde_row: dict,
    *,
    electrification: str | None = None,
    energy_basis: str = "VDE_TOTAL",
    method: str = "physics_simple",
    powertrain_features: dict | None = None,
    manual_inputs: dict | None = None,
    model_options: dict | None = None,
):
    request = build_fuel_estimate_request_from_vde(
        vde_row,
        electrification=electrification,
        energy_basis=energy_basis,
        method=method,
        powertrain_features=powertrain_features,
        manual_inputs=manual_inputs,
        model_options=model_options,
    )
    return run_fuel_estimation(request)


def fetch_fuelcons_allowed(exclude_keys: tuple[str, ...] = ("id",)) -> list[str]:
    return fetch_fuelcons_allowed_columns(exclude_keys)


def fetch_vde_legislation(vde_id: int) -> str:
    return repo_fetch_vde_legislation(vde_id)


def fetch_fuelcons_by_vde(vde_id: int) -> pd.DataFrame:
    rows = fetch_fuelcons_by_vde_id(vde_id)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_fuelcons_all(filters: Dict[str, Any]) -> pd.DataFrame:
    rows = fetch_fuelcons_rows(filters)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_scatter_join_rows() -> list[dict]:
    return fetch_fuelcons_join_rows()


def fetch_vde_rows_by_ids(vde_ids) -> pd.DataFrame:
    rows = fetch_vde_by_ids(vde_ids)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
