from __future__ import annotations

from datetime import datetime

from src.vde_core import db as db_module


def _table() -> str:
    return "tire_roadload_db" if db_module.LEGACY_FIXTURE_MODE else "tire_db"


def _id_field() -> str:
    return "id" if db_module.LEGACY_FIXTURE_MODE else "tire_id"


def _adapt_row(row: dict | None) -> dict:
    if not row:
        return {}
    item = dict(row)
    if not db_module.LEGACY_FIXTURE_MODE:
        item["id"] = item.get("tire_id")
    return item


def _adapt_rows(rows: list[dict]) -> list[dict]:
    return [_adapt_row(row) for row in rows]


def _normalized_payload(payload: dict) -> dict:
    data = dict(payload or {})
    data["updated_at"] = datetime.utcnow().isoformat()
    return data


def create_tire_roadload(payload: dict) -> int:
    data = dict(payload or {})
    data.setdefault("is_active", 1)
    data["updated_at"] = datetime.utcnow().isoformat()
    db_module.ensure_db()
    with db_module._con() as con:
        table = _table()
        data = db_module._validated_write_payload(con, table, data)
        cols = list(data.keys())
        vals = [data[c] for c in cols]
        placeholders = ",".join(["?"] * len(cols))
        cur = con.cursor()
        cur.execute(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        return int(cur.lastrowid)


def update_tire_roadload(tire_id: int, payload: dict) -> None:
    data = _normalized_payload(payload)
    db_module.ensure_db()
    with db_module._con() as con:
        table = _table()
        data = db_module._validated_write_payload(con, table, data)
        columns = list(data)
        con.execute(
            f"UPDATE {table} SET {', '.join(f'{column}=?' for column in columns)} WHERE {_id_field()}=?",
            [data[column] for column in columns] + [int(tire_id)],
        )


def get_tire_roadload_by_id(tire_id: int) -> dict:
    row = db_module.fetchone(f"SELECT * FROM {_table()} WHERE {_id_field()}=?;", (int(tire_id),))
    return _adapt_row(row)


def get_tire_roadload_by_code(tire_test_code: str) -> dict:
    row = db_module.fetchone(f"SELECT * FROM {_table()} WHERE tire_test_code=?;", (str(tire_test_code),))
    return _adapt_row(row)


def list_tire_roadload_active() -> list[dict]:
    return _adapt_rows(db_module.fetchall(
        f"SELECT * FROM {_table()} WHERE COALESCE(is_active, 1)=1 "
        "ORDER BY manufacturer, model, size_code, tire_test_code;"
    ))


def search_tire_roadload(
    *,
    manufacturer: str | None = None,
    model: str | None = None,
    size_code: str | None = None,
    standard_family: str | None = None,
    min_test_mileage_km: float | None = None,
    active_only: bool = True,
) -> list[dict]:
    sql = f"SELECT * FROM {_table()} WHERE 1=1"
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
    return _adapt_rows(db_module.fetchall(sql, tuple(params)))


def deactivate_tire_roadload(tire_id: int) -> None:
    update_tire_roadload(int(tire_id), {"is_active": 0})


def delete_tire_roadload(tire_id: int) -> int:
    db_module.ensure_db()
    with db_module._con() as con:
        cur = con.execute(f"DELETE FROM {_table()} WHERE {_id_field()}=?", (int(tire_id),))
        return int(cur.rowcount or 0)
