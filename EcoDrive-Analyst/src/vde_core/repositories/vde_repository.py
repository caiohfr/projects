from __future__ import annotations

from src.vde_core.db import delete_row, fetchall, fetchone, insert_vde, update_vde


def fetch_vde_by_id(vde_id: int) -> dict:
    return fetchone("SELECT * FROM vde_db WHERE id=?;", (int(vde_id),)) or {}


def fetch_vde_by_ids(vde_ids) -> list[dict]:
    ids = [int(v) for v in vde_ids if v is not None]
    if not ids:
        return []

    unique_ids = list(dict.fromkeys(ids))
    if len(unique_ids) == 1:
        return fetchall("SELECT * FROM vde_db WHERE id=?;", (unique_ids[0],))

    qmarks = ",".join("?" for _ in unique_ids)
    return fetchall(f"SELECT * FROM vde_db WHERE id IN ({qmarks});", tuple(unique_ids))


def fetch_vde_all_rows() -> list[dict]:
    return fetchall("SELECT * FROM vde_db ORDER BY COALESCE(updated_at, created_at) DESC;")


def fetch_vde_edit_rows(limit: int = 100) -> list[dict]:
    return fetchall(
        """
        SELECT id, legislation, category, make, model, year,
               coast_A_N, coast_B_N_per_kph, coast_C_N_per_kph2, mass_kg, notes
        FROM vde_db
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    )


def fetch_vde_make_rows(legislation: str, category: str) -> list[dict]:
    return fetchall(
        """
        SELECT DISTINCT make FROM vde_db
        WHERE legislation=? AND category=?
        ORDER BY make
        """,
        (legislation, category),
    )


def fetch_vde_distinct_makes() -> list[str]:
    rows = fetchall(
        "SELECT DISTINCT make FROM vde_db "
        "WHERE make IS NOT NULL AND make <> '' "
        "ORDER BY make;"
    )
    return [r["make"] for r in rows] if rows else []


def fetch_vde_distinct_categories() -> list[str]:
    rows = fetchall(
        "SELECT DISTINCT category FROM vde_db "
        "WHERE category IS NOT NULL AND category <> '' "
        "ORDER BY category;"
    )
    return [r["category"] for r in rows] if rows else []


def fetch_vde_distinct_transmission_models() -> list[str]:
    rows = fetchall(
        "SELECT DISTINCT transmission_model "
        "FROM vde_db "
        "WHERE transmission_model IS NOT NULL AND transmission_model <> '' "
        "ORDER BY transmission_model;"
    )
    return [r["transmission_model"] for r in rows] if rows else []


def fetch_vde_engine_type(vde_id: int) -> str:
    row = fetchone("SELECT engine_type FROM vde_db WHERE id=?;", (int(vde_id),)) or {}
    return str(row.get("engine_type", "") or "")


def fetch_vde_legislation(vde_id: int) -> str:
    row = fetchone("SELECT legislation FROM vde_db WHERE id=?", (int(vde_id),)) or {}
    return str(row.get("legislation", "EPA")).upper()


def count_linked_fuelcons_rows(vde_id: int) -> int:
    rows = fetchall("SELECT COUNT(*) AS n FROM fuelcons_db WHERE vde_id=?", (int(vde_id),))
    if not rows:
        return 0
    try:
        return int(rows[0].get("n", 0))
    except Exception:
        return 0


def insert_vde_row(payload: dict) -> int:
    return int(insert_vde(dict(payload)))


def update_vde_by_id(vde_id: int, payload: dict) -> None:
    update_vde(int(vde_id), dict(payload))


def delete_vde_by_id(vde_id: int) -> int:
    return int(delete_row("vde_db", int(vde_id)))
