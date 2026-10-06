from __future__ import annotations

from pathlib import Path

from src.vde_core import db as db_module
from src.vde_core.db import current_db_path, delete_row, fetchall, fetchone, insert_vde, update_vde


def fetch_vde_by_id(vde_id: int) -> dict:
    row = fetchone("SELECT * FROM vde_db WHERE id=?;", (int(vde_id),)) or {}
    if row:
        row["associated_component_resolutions"] = fetch_vde_component_resolutions(vde_id)
    return row


def fetch_vde_component_resolutions(vde_id: int) -> list[dict]:
    """Return canonical resolution lineage already adopted by a VDE.

    Legacy fixtures may not contain the canonical lineage tables, so absence of
    those tables means that no reusable baseline resolution is available.
    """
    link_columns = {
        str(row.get("name") or "")
        for row in fetchall('PRAGMA table_info("vde_component_resolution");')
    }
    resolution_columns = {
        str(row.get("name") or "")
        for row in fetchall('PRAGMA table_info("component_resolution");')
    }
    if not {"vde_id", "component_resolution_id", "boundary"}.issubset(link_columns):
        return []
    if "component_resolution_id" not in resolution_columns:
        return []

    link_optional = ("adoption_role", "ordinal")
    resolution_fields = (
        "method",
        "confidence",
        "fidelity_level",
        "resolved_A_N",
        "resolved_B_N_per_kph",
        "resolved_C_N_per_kph2",
        "provenance_json",
        "record_status",
        "review_status",
    )
    selected = [
        "link.component_resolution_id AS component_resolution_id",
        "link.boundary AS boundary",
        *[
            f'link."{name}" AS "{name}"' if name in link_columns else f'NULL AS "{name}"'
            for name in link_optional
        ],
        *[
            f'resolution."{name}" AS "{name}"' if name in resolution_columns else f'NULL AS "{name}"'
            for name in resolution_fields
        ],
    ]
    return fetchall(
        f"SELECT {','.join(selected)} "
        "FROM vde_component_resolution AS link "
        "JOIN component_resolution AS resolution "
        "ON resolution.component_resolution_id = link.component_resolution_id "
        "WHERE link.vde_id = ? "
        "ORDER BY link.boundary, link.component_resolution_id",
        (int(vde_id),),
    )


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


def fetch_vde_browser_runtime_snapshot() -> dict:
    row = fetchone("SELECT COUNT(*) AS n FROM vde_db;") or {}
    sample_rows = fetchall("SELECT id FROM vde_db ORDER BY id LIMIT 10;")
    try:
        row_count = int(row.get("n", 0) or 0)
    except Exception:
        row_count = 0
    return {
        "path": str(Path(current_db_path()).resolve()),
        "row_count": row_count,
        "sample_ids": [int(item.get("id")) for item in sample_rows if item.get("id") is not None],
    }


def fetch_vde_edit_rows(limit: int = 100) -> list[dict]:
    return fetchall(
        """
        SELECT id, legislation, category, make, model, year,
               coast_A_N, coast_B_N_per_kph, coast_C_N_per_kph2,
               mass_kg, test_mass_kg, inertia_class, notes
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
    return int(delete_row(db_module.VDE_WRITE_TABLE, int(vde_id)))
