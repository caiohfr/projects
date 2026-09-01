from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd

from src.vde_core.fuel_estimation import FuelEstimateRequest, run_fuel_estimation
from src.vde_core.fuel_energy import LHV_MJ_PER_L, MJ_TO_Wh
from src.vde_core.repositories import (
    delete_fuelcons_by_id,
    fetch_fuelcons_allowed_columns,
    fetch_fuelcons_baseline_labels as repo_fetch_fuelcons_baseline_labels,
    fetch_fuelcons_by_id,
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
from src.vde_core.vde_net_total_contract import canonical_vde_read
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


def summarize_saved_scenario_revision_states(
    rows: list[dict[str, Any]] | None,
    current_vde_row: dict | None,
) -> dict[str, Any]:
    summary = {
        "total": 0,
        "current": 0,
        "changed": 0,
        "missing": 0,
        "unknown": 0,
        "refresh_required": 0,
    }
    if not rows:
        return summary

    for row in rows:
        state = compare_saved_scenario_revision(dict(row).get("source_vde_revision"), current_vde_row)
        status = str(state.get("status") or "unknown")
        summary["total"] += 1
        if status in ("current", "changed", "missing", "unknown"):
            summary[status] += 1
        else:
            summary["unknown"] += 1

    summary["refresh_required"] = summary["changed"] + summary["missing"]
    return summary


def resolve_vde_energy_values(vde_row: dict) -> dict[str, Any]:
    row = dict(vde_row or {})
    row["vde_total_mj_per_km"] = to_float(row.get("vde_total_mj_per_km"))
    row["vde_net_mj_per_km"] = to_float(row.get("vde_net_mj_per_km"))
    canonical = canonical_vde_read(row)
    warnings: list[str] = []

    if canonical.total_mj_per_km is None:
        warnings.append("vde_total_missing")
    if canonical.net_mj_per_km is None:
        warnings.append("vde_net_missing")

    return {
        "vde_total_mj_per_km": canonical.total_mj_per_km,
        "vde_net_mj_per_km": canonical.net_mj_per_km,
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


def fetch_fuelcons_baselines() -> pd.DataFrame:
    """Lightweight persisted rows for FuelCons baseline discovery."""

    rows = repo_fetch_fuelcons_baseline_labels()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_fuelcons_row(fuelcons_id: int) -> dict[str, Any]:
    """Materialize one selected FuelCons baseline row."""

    return fetch_fuelcons_by_id(fuelcons_id) or {}


def fetch_fuelcons_all(filters: Dict[str, Any]) -> pd.DataFrame:
    rows = fetch_fuelcons_rows(filters)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_scatter_join_rows() -> list[dict]:
    return fetch_fuelcons_join_rows()


def fetch_vde_rows_by_ids(vde_ids) -> pd.DataFrame:
    rows = fetch_vde_by_ids(vde_ids)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_json_blob(raw_value: Any) -> dict[str, Any]:
    """Verbatim extraction of `pwt_fuel_energy._load_json_blob`, made public
    here since it is now imported back into that module (Sprint 10E
    ownership cleanup)."""

    if raw_value in (None, ""):
        return {}
    if isinstance(raw_value, dict):
        return dict(raw_value)
    try:
        parsed = json.loads(str(raw_value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def resolve_reference_fuel_type(row: dict[str, Any]) -> str | None:
    """Verbatim extraction of `pwt_fuel_energy._fuel_type_from_reference_row`,
    renamed and made public here since Sprint 10D's Efficiency Quick resolver
    also needs it for the active/source row, not just for donor rows.
    """

    assumptions = load_json_blob(row.get("assumptions_json"))
    provenance = load_json_blob(row.get("provenance_json"))
    fuel_type = assumptions.get("fuel_type")
    if fuel_type in (None, ""):
        fuel_type = dict(provenance.get("scenario_feature_values") or {}).get("fuel_type")
    text = str(fuel_type).strip() if fuel_type not in (None, "") else None
    return text or None


def derive_reference_pse(reference_row: dict[str, Any]) -> dict[str, Any]:
    """Sprint 10D: verbatim extraction of `pwt_fuel_energy._derive_reference_pse`.

    Computes a `reference_row`'s OWN cycle-effective PSE from its own linked
    VDE demand and its own recorded fuel/energy consumption -- never the
    active Quick vehicle's demand. Used both for "Current PSE" (pointed at
    the Quick Scenario's own source fuelcons row) and "Benchmark PSE"
    (pointed at a donor fuelcons row); both are the same computation, just
    applied to different rows, matching Sec 9's "do not implement a second
    benchmark-PSE formula" instruction and avoiding a third formula for
    "current" PSE as well.

    Returns `{"value": float | None, "status": str, "basis": str | None}`,
    `status` one of `"unavailable" | "missing_demand" | "available" |
    "missing_observed_result"`.
    """

    source_vde_id = reference_row.get("vde_id")
    if source_vde_id in (None, ""):
        return {"value": None, "status": "unavailable", "basis": None}
    try:
        source_vde = fetch_vde_row(int(source_vde_id))
    except Exception:
        return {"value": None, "status": "unavailable", "basis": None}
    energy_values = resolve_vde_energy_values(source_vde)
    energy_basis = str(reference_row.get("energy_basis") or "VDE_TOTAL").upper()
    demand_value = (
        energy_values["vde_net_mj_per_km"]
        if energy_basis == "VDE_NET"
        else energy_values["vde_total_mj_per_km"]
    )
    if demand_value is None:
        return {"value": None, "status": "missing_demand", "basis": energy_basis}

    fuel_l_100km = to_float(reference_row.get("fuel_l_per_100km"))
    if fuel_l_100km is not None:
        fuel_type = resolve_reference_fuel_type(reference_row) or "Gasoline"
        lhv = float(LHV_MJ_PER_L.get(fuel_type, LHV_MJ_PER_L["Gasoline"]))
        consumed = (fuel_l_100km / 100.0) * lhv
        if consumed > 0:
            return {"value": float(demand_value) / consumed, "status": "available", "basis": energy_basis}

    energy_wh_km = to_float(reference_row.get("energy_Wh_per_km"))
    if energy_wh_km is not None:
        consumed = float(energy_wh_km) / MJ_TO_Wh
        if consumed > 0:
            return {"value": float(demand_value) / consumed, "status": "available", "basis": energy_basis}
    return {"value": None, "status": "missing_observed_result", "basis": energy_basis}


def list_benchmark_fuelcons_candidates(vde_id: int) -> list[dict[str, Any]]:
    """Sprint 10D: Streamlit-free equivalent of
    `pwt_fuel_energy._reference_candidates_for_type(vde_id, "Another
    fuelcons_db line")` -- every `fuelcons_db` row NOT linked to the active
    `vde_id`, i.e. every candidate donor for "what if my active vehicle had
    this benchmark's PSE?" (Sec 9).
    """

    df = fetch_fuelcons_all({})
    if df is None or df.empty:
        return []
    candidates = df.loc[df["vde_id"] != int(vde_id)].copy()
    return candidates.to_dict("records")
