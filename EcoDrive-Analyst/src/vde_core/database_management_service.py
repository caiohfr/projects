from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import sqlite3
from uuid import uuid4

from src.vde_core import db as db_module
from src.vde_core.component_repositories import (
    _CANONICAL_STORAGE_FIELDS,
    _component_to_storage,
    _db_row_to_component,
    _normalize_domain_key,
    _validate_component,
)
from src.vde_core.data_change_log_repository import append_change_log
from src.vde_core.database_management_contract import (
    LOCAL_ADMIN_ACTOR,
    ActorContext,
    ChangeAction,
    ChangeCommand,
    ChangePreview,
    ChangeResult,
    EntityType,
    FieldDiff,
    ValidationIssue,
    normalize_change_action,
    normalize_entity_type,
    normalize_record_origin,
)
from src.vde_core.database_management_policy import FieldAccess, field_access_for, field_policy_for
from src.vde_core.test_mass import autoresolve_test_mass
from src.vde_core.tire_roadload_service import _normalize_tire_payload


_REASON_REQUIRED_ACTIONS = {
    ChangeAction.UPDATE,
    ChangeAction.RECALCULATE_UPDATE,
    ChangeAction.ARCHIVE,
    ChangeAction.RESTORE,
    ChangeAction.DELETE,
    ChangeAction.REASSIGN_RELATIONSHIP,
}


def local_actor_context() -> ActorContext:
    return LOCAL_ADMIN_ACTOR


def preview_change(command: ChangeCommand, actor_context: ActorContext | None = None) -> ChangePreview:
    actor = actor_context or local_actor_context()
    issues: list[ValidationIssue] = []

    if not str(actor.actor_id or "").strip() or not str(actor.actor_role or "").strip():
        issues.append(ValidationIssue("ERROR", "actor_context_required", "Actor ID and role are required."))

    try:
        entity = normalize_entity_type(command.entity_type)
    except ValueError:
        return _invalid_contract_preview(command, "entity_type_invalid", "Unsupported entity type.")
    try:
        action = normalize_change_action(command.action)
    except ValueError:
        return _invalid_contract_preview(command, "action_invalid", "Unsupported change action.", entity.value)

    current = deepcopy(dict(command.current_record or {}))
    record_id = command.record_id if command.record_id is not None else current.get("id")
    requested_origin = command.record_origin or command.payload.get("record_origin") or current.get("record_origin")
    if action is ChangeAction.CREATE and not str(requested_origin or "").strip():
        issues.append(
            ValidationIssue(
                "ERROR",
                "record_origin_required",
                "CREATE requires an explicit record origin.",
                "record_origin",
            )
        )
    try:
        origin = normalize_record_origin(entity, requested_origin)
    except ValueError as exc:
        issues.append(ValidationIssue("ERROR", "record_origin_invalid", str(exc), "record_origin"))
        origin = "LEGACY"

    if action is ChangeAction.CREATE and record_id is not None:
        issues.append(ValidationIssue("ERROR", "create_has_record_id", "CREATE must not target an existing record."))
    if action not in {ChangeAction.CREATE, ChangeAction.DUPLICATE} and record_id is None:
        issues.append(ValidationIssue("ERROR", "record_id_required", f"{action.value} requires a record ID."))

    normalized_payload: dict = {}
    field_diff: list[FieldDiff] = []
    advanced_fields: list[str] = []

    for field_name, after in dict(command.payload or {}).items():
        field = str(field_name or "").strip()
        if not field:
            issues.append(ValidationIssue("ERROR", "field_name_invalid", "Payload contains an empty field name."))
            continue
        if field == "record_origin" and action in {ChangeAction.CREATE, ChangeAction.DUPLICATE}:
            normalized_payload[field] = origin
            if current.get(field) != origin:
                field_diff.append(FieldDiff(field, current.get(field), origin))
            continue
        access = field_access_for(entity, origin, field)
        if access is FieldAccess.UNKNOWN:
            issues.append(ValidationIssue("ERROR", "field_unknown", f"Field {field!r} is not part of the {entity.value} contract.", field))
            continue
        if access is FieldAccess.IMMUTABLE:
            issues.append(ValidationIssue("ERROR", "field_immutable", f"Field {field!r} cannot be edited directly.", field))
            continue
        if access is FieldAccess.DERIVED:
            issues.append(ValidationIssue("ERROR", "field_derived", f"Field {field!r} is resolver-owned and read-only.", field))
            continue
        normalized_payload[field] = after
        before = current.get(field)
        if before != after:
            field_diff.append(FieldDiff(field, before, after))
        if access is FieldAccess.ADVANCED_CORRECTION:
            advanced_fields.append(field)

    if action in {ChangeAction.CREATE, ChangeAction.DUPLICATE}:
        normalized_payload.setdefault("record_origin", origin)
        if entity.value == "TIRE":
            normalized_payload.setdefault("is_active", 1)
        else:
            normalized_payload.setdefault("record_status", "ACTIVE")

    requires_reason = action in _REASON_REQUIRED_ACTIONS or bool(advanced_fields)
    if advanced_fields:
        issues.append(
            ValidationIssue(
                "WARNING",
                "advanced_correction",
                "Advanced source correction requested for: " + ", ".join(sorted(advanced_fields)) + ".",
            )
        )
    if requires_reason and not str(command.reason or "").strip():
        issues.append(ValidationIssue("ERROR", "reason_required", "A reason is required for this action."))

    rows = _rows_for_action(action, record_id, normalized_payload)
    can_commit = not any(issue.severity == "ERROR" for issue in issues)
    return ChangePreview(
        operation_id=str(uuid4()),
        entity_type=entity.value,
        record_id=None if record_id is None else str(record_id),
        action=action.value,
        normalized_payload=normalized_payload,
        field_diff=tuple(field_diff),
        validation_issues=tuple(issues),
        physics_action=command.physics_action,
        rows_to_create=rows["create"],
        rows_to_update=rows["update"],
        rows_to_archive=rows["archive"],
        rows_to_delete=rows["delete"],
        requires_reason=requires_reason,
        can_commit=can_commit,
    )


def _rows_for_action(action: ChangeAction, record_id, payload: dict) -> dict[str, tuple[dict, ...]]:
    target = {"record_id": record_id, "payload": deepcopy(payload)}
    rows = {"create": (), "update": (), "archive": (), "delete": ()}
    if action in {ChangeAction.CREATE, ChangeAction.DUPLICATE, ChangeAction.RECALCULATE_NEW}:
        rows["create"] = (target,)
    elif action in {ChangeAction.UPDATE, ChangeAction.RECALCULATE_UPDATE, ChangeAction.REASSIGN_RELATIONSHIP, ChangeAction.RESTORE}:
        rows["update"] = (target,)
    elif action is ChangeAction.ARCHIVE:
        rows["archive"] = (target,)
    elif action is ChangeAction.DELETE:
        rows["delete"] = (target,)
    return rows


def _invalid_contract_preview(command: ChangeCommand, code: str, message: str, entity_type: str = "") -> ChangePreview:
    return ChangePreview(
        operation_id=str(uuid4()),
        entity_type=entity_type,
        record_id=None if command.record_id is None else str(command.record_id),
        action=str(command.action or ""),
        normalized_payload={},
        field_diff=(),
        validation_issues=(ValidationIssue("ERROR", code, message),),
        can_commit=False,
    )


_MANAGEMENT_TABLES = {
    EntityType.VDE: "vde_db",
    EntityType.FUEL_CONSUMPTION: "fuelcons_db",
    EntityType.TIRE: "tire_roadload_db",
    EntityType.COMPONENT: "component_db",
}
_TEXT_SEARCH_FIELDS = {
    EntityType.VDE: ("make", "model", "category", "legislation", "source_name", "source_record_id"),
    EntityType.FUEL_CONSUMPTION: ("electrification", "fuel_type", "method_note", "source_name", "source_record_id"),
    EntityType.TIRE: ("tire_test_code", "manufacturer", "model", "size_code", "source_name", "source_record_id"),
    EntityType.COMPONENT: ("component_code", "component_name", "source_name", "source_record_id", "hardware_reference"),
}


def browse_records(
    entity_type: EntityType | str,
    *,
    query: str = "",
    include_archived: bool = False,
    component_domain: str | None = None,
    limit: int = 250,
) -> list[dict]:
    """Return rows for the staged management UI without exposing write access."""
    entity = normalize_entity_type(entity_type)
    db_module.ensure_db()
    clauses: list[str] = []
    params: list[object] = []
    if entity is EntityType.TIRE:
        if not include_archived:
            clauses.append("COALESCE(is_active, 1)=1")
    else:
        if not include_archived:
            clauses.append("COALESCE(record_status, 'ACTIVE')='ACTIVE'")
    if entity is EntityType.COMPONENT:
        if not component_domain:
            raise ValueError("Components browsing requires a domain filter.")
        clauses.append("domain=?")
        params.append(_normalize_domain_key(component_domain))
    needle = str(query or "").strip()
    if needle:
        search = " OR ".join(f"COALESCE(CAST({field} AS TEXT), '') LIKE ?" for field in _TEXT_SEARCH_FIELDS[entity])
        clauses.append(f"({search})")
        params.extend([f"%{needle}%"] * len(_TEXT_SEARCH_FIELDS[entity]))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    order_field = "tire_test_code" if entity is EntityType.TIRE else "id"
    rows = db_module.fetchall(
        f"SELECT * FROM {_MANAGEMENT_TABLES[entity]}{where} ORDER BY {order_field} DESC LIMIT ?",
        tuple([*params, max(1, min(int(limit), 1000))]),
    )
    return [_adapt_component_row(row) if entity is EntityType.COMPONENT else row for row in rows]


def get_record(
    entity_type: EntityType | str,
    record_id: int | str,
    *,
    component_domain: str | None = None,
) -> dict | None:
    entity = normalize_entity_type(entity_type)
    db_module.ensure_db()
    with db_module._con() as con:
        row = _fetch_record(con, entity, record_id, component_domain=component_domain)
    return _adapt_component_row(row) if entity is EntityType.COMPONENT and row else row


def editable_fields_for_record(entity_type: EntityType | str, record: dict) -> tuple[str, ...]:
    """List fields the current policy permits the staged UI to edit."""
    entity = normalize_entity_type(entity_type)
    policy = field_policy_for(entity, record.get("record_origin"))
    fields = policy.editable | policy.advanced_correction
    return tuple(sorted(field for field in fields if field in record))


def duplicate_payload_for(entity_type: EntityType | str, record: dict) -> dict:
    """Build a policy-safe duplicate draft; uniqueness is finalized at commit."""
    entity = normalize_entity_type(entity_type)
    payload = {field: record.get(field) for field in editable_fields_for_record(entity, record)}
    payload["record_origin"] = record.get("record_origin")
    if entity is EntityType.COMPONENT:
        payload["domain"] = record.get("domain")
        payload["component_code"] = f"{record.get('component_id') or record.get('component_code')}-COPY"
    elif entity is EntityType.TIRE:
        payload["tire_test_code"] = f"{record.get('tire_test_code')}-COPY"
    return {key: value for key, value in payload.items() if value is not None}


def simple_dependencies(entity_type: EntityType | str, record_id: int | str) -> tuple[dict, ...]:
    """Return direct relational dependencies known before the 7D impact engine."""
    entity = normalize_entity_type(entity_type)
    db_module.ensure_db()
    with db_module._con() as con:
        return _simple_dependencies(con, entity, record_id)


def apply_change(
    preview: ChangePreview,
    actor_context: ActorContext | None = None,
    *,
    reason: str | None = None,
) -> ChangeResult:
    """Commit an already reviewed change and its audit receipt atomically."""
    if not preview.can_commit:
        raise ValueError("Cannot apply a change preview that did not pass validation.")
    entity = normalize_entity_type(preview.entity_type)
    action = normalize_change_action(preview.action)
    actor = actor_context or local_actor_context()
    resolved_reason = str(reason or "").strip() or None
    if preview.requires_reason and not resolved_reason:
        raise ValueError("A reason is required before this change can be committed.")

    db_module.ensure_db()
    with db_module._con() as con:
        before = _fetch_record(con, entity, preview.record_id) if preview.record_id is not None else None
        if action is ChangeAction.DELETE:
            dependencies = _simple_dependencies(con, entity, preview.record_id)
            if dependencies:
                raise ValueError("Delete is blocked because this record has direct dependencies. Archive it or resolve the relationship first.")
        else:
            dependencies = ()

        affected_id = _apply_preview(con, entity, action, preview, before)
        after = _fetch_record(con, entity, affected_id)
        change_log_id = append_change_log(
            preview,
            actor,
            reason=resolved_reason,
            before=_adapt_component_row(before) if entity is EntityType.COMPONENT and before else before,
            after=_adapt_component_row(after) if entity is EntityType.COMPONENT and after else after,
            impact={"dependencies": list(dependencies), "physics_action": preview.physics_action},
            connection=con,
        )
    return ChangeResult(
        operation_id=preview.operation_id,
        entity_type=entity.value,
        action=action.value,
        committed=True,
        change_log_id=change_log_id,
        affected_record_ids=(str(affected_id),),
    )


def _apply_preview(
    con: sqlite3.Connection,
    entity: EntityType,
    action: ChangeAction,
    preview: ChangePreview,
    before: dict | None,
) -> int:
    if action in {ChangeAction.CREATE, ChangeAction.DUPLICATE}:
        return _create_record(con, entity, dict(preview.normalized_payload), before=before, duplicate=action is ChangeAction.DUPLICATE)
    if before is None:
        raise ValueError(f"Record {preview.record_id!r} no longer exists.")
    if action is ChangeAction.UPDATE:
        _update_record(con, entity, int(before["id"]), dict(preview.normalized_payload), before)
        return int(before["id"])
    if action is ChangeAction.ARCHIVE:
        _set_archived(con, entity, int(before["id"]), archived=True)
        return int(before["id"])
    if action is ChangeAction.RESTORE:
        _set_archived(con, entity, int(before["id"]), archived=False)
        return int(before["id"])
    if action is ChangeAction.DELETE:
        con.execute(f"DELETE FROM {_MANAGEMENT_TABLES[entity]} WHERE id=?", (int(before["id"]),))
        return int(before["id"])
    raise ValueError(f"Action {action.value} is not available in Database Management 7C.")


def _create_record(
    con: sqlite3.Connection,
    entity: EntityType,
    payload: dict,
    *,
    before: dict | None,
    duplicate: bool,
) -> int:
    data = dict(payload)
    if duplicate and before:
        data = {**_duplicate_payload_from_db(entity, before), **data}
    if entity is EntityType.VDE:
        data = autoresolve_test_mass(data)
    elif entity is EntityType.TIRE:
        if duplicate:
            data["tire_test_code"] = _next_code(con, "tire_roadload_db", "tire_test_code", data.get("tire_test_code"))
        data = _normalize_tire_payload(data)
    elif entity is EntityType.COMPONENT:
        return _create_component(con, data, duplicate=duplicate)
    return _insert_row(con, _MANAGEMENT_TABLES[entity], data)


def _update_record(con: sqlite3.Connection, entity: EntityType, record_id: int, payload: dict, before: dict) -> None:
    data = dict(payload)
    if entity is EntityType.VDE:
        resolved = autoresolve_test_mass({**before, **data})
        for field in ("inertia_class", "test_mass_kg", "test_mass_low_kg", "test_mass_high_kg", "test_mass_basis"):
            if resolved.get(field) != before.get(field):
                data[field] = resolved.get(field)
    elif entity is EntityType.TIRE:
        normalized = _normalize_tire_payload({**before, **data})
        data = {field: value for field, value in normalized.items() if field not in {"id", "created_at"}}
    elif entity is EntityType.COMPONENT:
        _update_component(con, record_id, data, before)
        return
    data["updated_at"] = _utc_now_iso()
    _update_row(con, _MANAGEMENT_TABLES[entity], record_id, data)


def _set_archived(con: sqlite3.Connection, entity: EntityType, record_id: int, *, archived: bool) -> None:
    if entity is EntityType.TIRE:
        _update_row(con, "tire_roadload_db", record_id, {"is_active": 0 if archived else 1, "updated_at": _utc_now_iso()})
        return
    _update_row(con, _MANAGEMENT_TABLES[entity], record_id, {"record_status": "ARCHIVED" if archived else "ACTIVE", "updated_at": _utc_now_iso()})


def _create_component(con: sqlite3.Connection, payload: dict, *, duplicate: bool) -> int:
    domain = _normalize_domain_key(payload.get("domain"))
    data = dict(payload)
    data.setdefault("record_status", "ACTIVE")
    data.setdefault("source_name", "manual_request")
    if duplicate:
        data["component_code"] = _next_code(con, "component_db", "component_code", data.get("component_code"))
    storage = _component_to_storage(domain, data, default_origin="MANUAL")
    component = _db_row_to_component(storage)
    issues = _validate_component(domain, component)
    if issues:
        raise ValueError(str(issues[0].get("message") or "Invalid component payload."))
    return _insert_row(con, "component_db", storage)


def _update_component(con: sqlite3.Connection, record_id: int, payload: dict, before: dict) -> None:
    domain = _normalize_domain_key(payload.get("domain") or before.get("domain"))
    current = _adapt_component_row(before)
    merged = {**current, **payload}
    if "component_code" in payload:
        merged["component_id"] = payload["component_code"]
    storage = _component_to_storage(domain, merged, default_origin=current.get("record_origin") or "LEGACY")
    component = _db_row_to_component(storage)
    issues = _validate_component(domain, component)
    if issues:
        raise ValueError(str(issues[0].get("message") or "Invalid component payload."))
    storage["updated_at"] = _utc_now_iso()
    _update_row(con, "component_db", record_id, storage)


def _insert_row(con: sqlite3.Connection, table: str, payload: dict) -> int:
    columns = set(db_module.table_columns(table))
    data = {field: value for field, value in dict(payload).items() if field in columns and field not in {"id", "created_at", "updated_at"}}
    if not data:
        raise ValueError("No writable values were supplied.")
    names = list(data)
    try:
        cursor = con.execute(
            f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
            [data[name] for name in names],
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Unable to create record: {exc}") from exc
    return int(cursor.lastrowid)


def _update_row(con: sqlite3.Connection, table: str, record_id: int, payload: dict) -> None:
    columns = set(db_module.table_columns(table))
    data = {field: value for field, value in dict(payload).items() if field in columns and field not in {"id", "created_at"}}
    if not data:
        return
    names = list(data)
    try:
        con.execute(
            f"UPDATE {table} SET {', '.join(f'{name}=?' for name in names)} WHERE id=?",
            [data[name] for name in names] + [record_id],
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Unable to update record: {exc}") from exc


def _fetch_record(
    con: sqlite3.Connection,
    entity: EntityType,
    record_id: int | str | None,
    *,
    component_domain: str | None = None,
) -> dict | None:
    if record_id is None:
        return None
    con.row_factory = sqlite3.Row
    sql = f"SELECT * FROM {_MANAGEMENT_TABLES[entity]} WHERE id=?"
    params: list[object] = [int(record_id)]
    if entity is EntityType.COMPONENT and component_domain:
        sql += " AND domain=?"
        params.append(_normalize_domain_key(component_domain))
    row = con.execute(sql, tuple(params)).fetchone()
    return dict(row) if row else None


def _simple_dependencies(con: sqlite3.Connection, entity: EntityType, record_id: int | str | None) -> tuple[dict, ...]:
    if record_id is None:
        return ()
    if entity is EntityType.VDE:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT id, vde_id, electrification FROM fuelcons_db WHERE vde_id=? ORDER BY id", (int(record_id),)).fetchall()
        return tuple({"entity_type": "FUEL_CONSUMPTION", **dict(row)} for row in rows)
    return ()


def _duplicate_payload_from_db(entity: EntityType, record: dict) -> dict:
    current = _adapt_component_row(record) if entity is EntityType.COMPONENT else dict(record)
    payload = {field: current.get(field) for field in editable_fields_for_record(entity, current)}
    payload["record_origin"] = current.get("record_origin")
    if entity is EntityType.COMPONENT:
        payload["domain"] = current.get("domain")
    return {field: value for field, value in payload.items() if value is not None}


def _next_code(con: sqlite3.Connection, table: str, field: str, proposed: object) -> str:
    base = str(proposed or "NEW-RECORD").strip() or "NEW-RECORD"
    candidate = base
    suffix = 2
    while con.execute(f"SELECT 1 FROM {table} WHERE {field}=? LIMIT 1", (candidate,)).fetchone():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _adapt_component_row(row: dict | None) -> dict:
    return _db_row_to_component(row) if row else {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
