from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

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
    update_vde_by_id,
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
    payload = build_bev_placeholder_payload()
    update_vde_by_id(vde_id, payload)
    return payload


def save_fuelcons_payload(payload: dict) -> int:
    return insert_fuelcons_row(payload)


def update_fuelcons_payload(row_id: int, payload: dict) -> None:
    update_fuelcons_by_id(row_id, payload)


def delete_fuelcons_row(row_id: int) -> int:
    return delete_fuelcons_by_id(row_id)


def compute_vde_total_from_ctx(vde_row: dict, ctx: Dict[str, Any]) -> Dict[str, float]:
    vde_net = to_float(vde_row.get("vde_net_mj_per_km")) or 0.0
    vde_total = (vde_net / ctx["eta_trans"]) if ctx.get("eta_trans") else None
    return {"vde_net_mj_per_km": vde_net, "vde_total_mj_per_km": vde_total}


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
