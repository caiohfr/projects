from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from src.vde_core.database_management_contract import (
    LOCAL_ADMIN_ACTOR,
    ActorContext,
    ChangeAction,
    ChangeCommand,
    ChangePreview,
    FieldDiff,
    ValidationIssue,
    normalize_change_action,
    normalize_entity_type,
    normalize_record_origin,
)
from src.vde_core.database_management_policy import FieldAccess, field_access_for


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
