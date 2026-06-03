from __future__ import annotations

from typing import Any

from src.vde_core.db import delete_row, fetchall, insert_fuelcons, update_row


def fetch_fuelcons_allowed_columns(exclude_keys: tuple[str, ...] = ("id",)) -> list[str]:
    rows = fetchall("PRAGMA table_info(fuelcons_db);") or []
    cols = []
    for row in rows:
        name = row.get("name") or row.get("NAME")
        if not name or name in exclude_keys:
            continue
        cols.append(name)
    return cols


def fetch_fuelcons_distinct_electrifications() -> list[str]:
    rows = fetchall(
        "SELECT DISTINCT electrification FROM fuelcons_db "
        "WHERE electrification IS NOT NULL AND electrification <> '' "
        "ORDER BY electrification;"
    ) or []
    vals = [r["electrification"] for r in rows]
    return vals or ["ICE", "MHEV", "HEV", "PHEV", "BEV"]


def fetch_fuelcons_by_vde_id(vde_id: int) -> list[dict]:
    return fetchall(
        (
            "SELECT id, created_at, method_note, electrification, "
            "fuel_l_per_100km, fuel_km_per_l, energy_Wh_per_km, gco2_per_km, "
            "fuel_ftp75_l_per_100km, fuel_hwfet_l_per_100km, "
            "energy_ftp75_Wh_per_km, energy_hwfet_Wh_per_km, "
            "gear_count, final_drive_ratio "
            "FROM fuelcons_db WHERE vde_id=? ORDER BY created_at DESC"
        ),
        (int(vde_id),),
    )


def fetch_fuelcons_rows(filters: dict[str, Any]) -> list[dict]:
    base = (
        "SELECT f.id, f.created_at, f.vde_id, f.electrification, f.method_note, "
        "f.fuel_l_per_100km, f.engine_max_power_kw, f.fuel_km_per_l, "
        "f.energy_Wh_per_km, f.gco2_per_km, f.fuel_ftp75_l_per_100km, "
        "f.fuel_hwfet_l_per_100km, f.energy_ftp75_Wh_per_km, "
        "f.energy_hwfet_Wh_per_km, f.gear_count, f.final_drive_ratio, "
        "v.make, v.model, v.year, v.category, v.legislation, v.vde_net_mj_per_km "
        "FROM fuelcons_db f JOIN vde_db v ON v.id = f.vde_id WHERE 1=1"
    )
    params = []
    if filters.get("electrification"):
        base += " AND f.electrification = ?"
        params.append(filters["electrification"])
    if filters.get("category"):
        base += " AND v.category = ?"
        params.append(filters["category"])
    if filters.get("make"):
        base += " AND v.make = ?"
        params.append(filters["make"])
    if filters.get("power_kw_range"):
        lo, hi = filters["power_kw_range"]
        if lo is not None:
            base += " AND f.engine_max_power_kw >= ?"
            params.append(lo)
        if hi is not None:
            base += " AND f.engine_max_power_kw < ?"
            params.append(hi)
    base += " ORDER BY f.created_at DESC"
    return fetchall(base, tuple(params)) if params else fetchall(base)


def fetch_fuelcons_join_rows() -> list[dict]:
    return fetchall(
        """
        SELECT
            f.vde_id,
            f.fuel_l_per_100km,
            f.fuel_km_per_l,
            f.energy_Wh_per_km,
            f.gco2_per_km,
            f.method_note,
            f.engine_max_power_kw,
            v.vde_net_mj_per_km,
            v.engine_size_l,
            v.transmission_type,
            v.drive_type,
            v.category,
            v.make,
            v.model,
            v.year
        FROM fuelcons_db f
        JOIN vde_db v ON v.id = f.vde_id
        """
    )


def insert_fuelcons_row(payload: dict) -> int:
    return int(insert_fuelcons(dict(payload)))


def update_fuelcons_by_id(row_id: int, payload: dict) -> None:
    update_row("fuelcons_db", int(row_id), dict(payload))


def delete_fuelcons_by_id(row_id: int) -> int:
    return int(delete_row("fuelcons_db", int(row_id)))
