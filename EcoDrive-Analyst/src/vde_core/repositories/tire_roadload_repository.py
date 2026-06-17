from __future__ import annotations

from datetime import datetime

from src.vde_core.db import delete_row, fetchall, fetchone, update_row


def _normalized_payload(payload: dict) -> dict:
    data = dict(payload or {})
    data["updated_at"] = datetime.utcnow().isoformat()
    return data


def create_tire_roadload(payload: dict) -> int:
    data = dict(payload or {})
    data.setdefault("is_active", 1)
    data["updated_at"] = datetime.utcnow().isoformat()
    cols = list(data.keys())
    vals = [data[c] for c in cols]
    placeholders = ",".join(["?"] * len(cols))

    from src.vde_core.db import _con, ensure_db  # local import to avoid widening module surface

    ensure_db()
    with _con() as con:
        cur = con.cursor()
        cur.execute(
            f"INSERT INTO tire_roadload_db ({','.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        return int(cur.lastrowid)


def update_tire_roadload(tire_id: int, payload: dict) -> None:
    update_row("tire_roadload_db", int(tire_id), _normalized_payload(payload))


def get_tire_roadload_by_id(tire_id: int) -> dict:
    return fetchone("SELECT * FROM tire_roadload_db WHERE id=?;", (int(tire_id),)) or {}


def get_tire_roadload_by_code(tire_test_code: str) -> dict:
    return fetchone("SELECT * FROM tire_roadload_db WHERE tire_test_code=?;", (str(tire_test_code),)) or {}


def list_tire_roadload_active() -> list[dict]:
    return fetchall(
        "SELECT * FROM tire_roadload_db WHERE COALESCE(is_active, 1)=1 "
        "ORDER BY manufacturer, model, size_code, tire_test_code;"
    )


def search_tire_roadload(
    *,
    manufacturer: str | None = None,
    model: str | None = None,
    size_code: str | None = None,
    standard_family: str | None = None,
    min_test_mileage_km: float | None = None,
    active_only: bool = True,
) -> list[dict]:
    sql = "SELECT * FROM tire_roadload_db WHERE 1=1"
    params = []
    if active_only:
        sql += " AND COALESCE(is_active, 1)=1"
    if manufacturer:
        sql += " AND manufacturer = ?"
        params.append(manufacturer)
    if model:
        sql += " AND model = ?"
        params.append(model)
    if size_code:
        sql += " AND size_code = ?"
        params.append(size_code)
    if standard_family:
        sql += " AND standard_family = ?"
        params.append(standard_family)
    if min_test_mileage_km is not None:
        sql += " AND COALESCE(test_mileage_km, 0) >= ?"
        params.append(float(min_test_mileage_km))
    sql += " ORDER BY manufacturer, model, size_code, tire_test_code"
    return fetchall(sql, tuple(params))


def deactivate_tire_roadload(tire_id: int) -> None:
    update_row(
        "tire_roadload_db",
        int(tire_id),
        {"is_active": 0, "updated_at": datetime.utcnow().isoformat()},
    )


def delete_tire_roadload(tire_id: int) -> int:
    return int(delete_row("tire_roadload_db", int(tire_id)))
