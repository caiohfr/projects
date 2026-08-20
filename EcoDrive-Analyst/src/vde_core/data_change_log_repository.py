from __future__ import annotations

import json
import sqlite3
from typing import Any

from src.vde_core import db as db_module
from src.vde_core.database_management_contract import ActorContext, ChangePreview


def append_change_log(
    preview: ChangePreview,
    actor_context: ActorContext,
    *,
    reason: str | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    impact: dict[str, Any] | None = None,
    connection: sqlite3.Connection | None = None,
) -> int:
    if not preview.can_commit:
        raise ValueError("Cannot log a change preview that is not committable.")
    values = (
        preview.operation_id,
        str(actor_context.actor_id),
        str(actor_context.actor_role),
        preview.entity_type,
        preview.record_id,
        preview.action,
        reason,
        _json(before),
        _json(after),
        _json(impact),
    )
    sql = """
        INSERT INTO data_change_log (
            operation_id, actor_id, actor_role, entity_type, record_id,
            action, reason, before_json, after_json, impact_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    if connection is not None:
        cursor = connection.execute(sql, values)
        return int(cursor.lastrowid)
    db_module.ensure_db()
    with db_module._con() as con:
        cursor = con.execute(sql, values)
        return int(cursor.lastrowid)


def fetch_change_log(operation_id: str) -> dict[str, Any] | None:
    row = db_module.fetchone(
        "SELECT * FROM data_change_log WHERE operation_id=? ORDER BY id DESC LIMIT 1",
        (str(operation_id),),
    )
    if not row:
        return None
    for field in ("before_json", "after_json", "impact_json"):
        row[field] = json.loads(row[field]) if row.get(field) else None
    return row


def _json(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
