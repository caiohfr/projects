from __future__ import annotations

from copy import deepcopy
import json
import sqlite3
from uuid import uuid4

from src.vde_core import db as db_module
from src.vde_core.component_repositories import _component_to_storage, _db_row_to_component, _normalize_domain_key
from src.vde_core.data_change_log_repository import append_change_log
from src.vde_core.database_management_contract import (
    LOCAL_ADMIN_ACTOR,
    ActorContext,
    ChangeAction,
    ChangePreview,
    ChangeResult,
    EntityType,
    ImpactPersistenceChoice,
    ImpactPreview,
    ValidationIssue,
    normalize_change_action,
    normalize_entity_type,
    normalize_impact_persistence_choice,
)
from src.vde_core.database_management_service import (
    _adapt_component_row,
    _apply_preview,
    _fetch_record,
    _MANAGEMENT_TABLES,
)
from src.vde_core.tire_roadload_service import _normalize_tire_payload
from src.vde_core.vde_request_compact_persistence import (
    REQUEST_HISTORY_PROPOSAL_TABLE,
    REQUEST_HISTORY_TABLE,
    _ensure_request_history_tables,
    persist_v22_maintenance_recalculation,
    prepare_v22_maintenance_recalculation,
)


_IMPACT_ENTITIES = {EntityType.TIRE, EntityType.COMPONENT}


def discover_catalog_usage(
    entity_type: EntityType | str,
    record_id: int | str,
    *,
    component_domain: str | None = None,
) -> dict:
    """Discover current request/VDE/Fuel usages without changing persisted state."""
    entity = normalize_entity_type(entity_type)
    if entity not in _IMPACT_ENTITIES:
        raise ValueError("Usage discovery is available only for Tire and Component records.")
    db_module.ensure_db()
    with db_module._con() as con:
        con.row_factory = sqlite3.Row
        _ensure_request_history_tables(con)
        target = _fetch_record(con, entity, record_id, component_domain=component_domain)
        if not target:
            raise ValueError(f"{entity.value} record {record_id!r} does not exist.")
        target_view = _adapt_component_row(target) if entity is EntityType.COMPONENT else dict(target)
        proposal_rows = con.execute(
            f"""
            SELECT p.*, h.record_key, h.created_at AS request_created_at, h.save_result_json
            FROM {REQUEST_HISTORY_PROPOSAL_TABLE} p
            JOIN {REQUEST_HISTORY_TABLE} h ON h.id=p.request_history_id
            ORDER BY p.request_history_id, p.display_index, p.id
            """
        ).fetchall()
        rows = [dict(row) for row in proposal_rows]
        superseded_request_ids = {
            int(save_result["source_request_history_id"])
            for row in rows
            for save_result in [_json_object(row.get("save_result_json"), default={})]
            if save_result.get("source_request_history_id") is not None
        }
        latest_request_by_vde: dict[int, int] = {}
        for row in rows:
            if row.get("saved_vde_row_id") is None:
                continue
            vde_id = int(row["saved_vde_row_id"])
            latest_request_by_vde[vde_id] = max(
                int(row["request_history_id"]),
                latest_request_by_vde.get(vde_id, 0),
            )

        direct_rows = [row for row in rows if _proposal_uses_record(entity, target_view, row)]
        current_direct_rows = [
            row
            for row in direct_rows
            if row.get("saved_vde_row_id") is not None
            and int(row["request_history_id"]) not in superseded_request_ids
            and latest_request_by_vde.get(int(row["saved_vde_row_id"])) == int(row["request_history_id"])
        ]
        historical_usages: list[dict] = []
        historical_request_ids = sorted(
            {
                int(row["request_history_id"])
                for row in direct_rows
                if row not in current_direct_rows
            }
        )
        for request_id in historical_request_ids:
            request_rows = [row for row in rows if int(row["request_history_id"]) == request_id]
            direct_ids = {
                str(row.get("proposal_id") or "")
                for row in direct_rows
                if int(row["request_history_id"]) == request_id
            }
            affected_ids = _walk_from_descendants(request_rows, direct_ids)
            for row in request_rows:
                proposal_id = str(row.get("proposal_id") or "")
                if proposal_id not in affected_ids:
                    continue
                historical_usages.append(
                    {
                        "request_history_id": request_id,
                        "record_key": row.get("record_key"),
                        "proposal_id": proposal_id,
                        "relation": (
                            "historical_snapshot_direct"
                            if proposal_id in direct_ids
                            else "historical_snapshot_walk_from"
                        ),
                        "saved_vde_row_id": row.get("saved_vde_row_id"),
                    }
                )
        request_ids = sorted({int(row["request_history_id"]) for row in current_direct_rows})
        usages: list[dict] = []
        requests: list[dict] = []
        affected_vde_ids: set[int] = set()

        for request_id in request_ids:
            request_rows = [row for row in rows if int(row["request_history_id"]) == request_id]
            direct_ids = {
                str(row.get("proposal_id") or "")
                for row in request_rows
                if _proposal_uses_record(entity, target_view, row)
            }
            affected_ids = _walk_from_descendants(request_rows, direct_ids)
            request_vde_ids = sorted(
                {
                    int(row["saved_vde_row_id"])
                    for row in request_rows
                    if row.get("saved_vde_row_id") is not None
                }
            )
            affected_vde_ids.update(request_vde_ids)
            for row in request_rows:
                proposal_id = str(row.get("proposal_id") or "")
                if proposal_id not in affected_ids:
                    continue
                usages.append(
                    {
                        "request_history_id": request_id,
                        "record_key": row.get("record_key"),
                        "proposal_id": proposal_id,
                        "display_index": row.get("display_index"),
                        "relation": "direct" if proposal_id in direct_ids else "walk_from_downstream",
                        "walk_from_proposal_id": row.get("walk_from_proposal_id"),
                        "saved_vde_row_id": row.get("saved_vde_row_id"),
                    }
                )
            requests.append(
                {
                    "request_history_id": request_id,
                    "record_key": request_rows[0].get("record_key") if request_rows else None,
                    "direct_proposal_ids": tuple(sorted(direct_ids)),
                    "affected_proposal_ids": tuple(sorted(affected_ids)),
                    "saved_vde_row_ids": tuple(request_vde_ids),
                }
            )

        review_required = _untracked_vde_usages(con, entity, target_view, affected_vde_ids)
        fuel_rows = _fuel_dependencies(con, tuple(sorted(affected_vde_ids)))
    return {
        "entity_type": entity.value,
        "record_id": str(record_id),
        "record": target_view,
        "usages": tuple(usages),
        "historical_usages": tuple(historical_usages),
        "requests": tuple(requests),
        "affected_vde_ids": tuple(sorted(affected_vde_ids)),
        "fuel_rows": tuple(fuel_rows),
        "review_required": tuple(review_required),
    }


def discover_vde_dependencies(vde_id: int | str) -> dict:
    db_module.ensure_db()
    with db_module._con() as con:
        con.row_factory = sqlite3.Row
        _ensure_request_history_tables(con)
        vde = con.execute("SELECT * FROM vde_db WHERE id=?", (int(vde_id),)).fetchone()
        if not vde:
            raise ValueError(f"VDE {vde_id!r} does not exist.")
        fuel_rows = con.execute(
            "SELECT id, vde_id, electrification, fuel_type, record_origin, review_status FROM fuelcons_db WHERE vde_id=? ORDER BY id",
            (int(vde_id),),
        ).fetchall()
        saved_proposals = con.execute(
            f"""
            SELECT p.id, p.request_history_id, p.proposal_id, p.saved_vde_row_id, h.record_key
            FROM {REQUEST_HISTORY_PROPOSAL_TABLE} p
            JOIN {REQUEST_HISTORY_TABLE} h ON h.id=p.request_history_id
            WHERE p.saved_vde_row_id=?
            ORDER BY p.request_history_id DESC, p.display_index
            """,
            (int(vde_id),),
        ).fetchall()
        baseline_requests = con.execute(
            f"SELECT id, record_key, created_at FROM {REQUEST_HISTORY_TABLE} WHERE baseline_vde_id=? ORDER BY id DESC",
            (int(vde_id),),
        ).fetchall()
    return {
        "vde": dict(vde),
        "fuel_rows": tuple(dict(row) for row in fuel_rows),
        "saved_proposals": tuple(dict(row) for row in saved_proposals),
        "baseline_requests": tuple(dict(row) for row in baseline_requests),
    }


def apply_vde_dependency_resolution(
    change_preview: ChangePreview,
    *,
    resolution_action: str,
    fuel_row_ids: tuple[int, ...] | list[int] = (),
    replacement_vde_id: int | str | None = None,
    actor_context: ActorContext | None = None,
    reason: str | None = None,
) -> ChangeResult:
    """Resolve VDE/Fuel dependencies without relying on ON DELETE CASCADE."""
    if not change_preview.can_commit or change_preview.record_id is None:
        raise ValueError("A ready VDE change preview is required.")
    if normalize_entity_type(change_preview.entity_type) is not EntityType.VDE:
        raise ValueError("VDE dependency resolution requires a VDE change preview.")
    action = str(resolution_action or "").strip().upper()
    if action not in {"REASSIGN_FUEL", "DELETE_FUEL_AND_VDE", "ADMIN_DELETE"}:
        raise ValueError("Unsupported VDE dependency resolution action.")
    preview_action = normalize_change_action(change_preview.action)
    if action == "REASSIGN_FUEL" and preview_action is not ChangeAction.REASSIGN_RELATIONSHIP:
        raise ValueError("Fuel reassignment requires a reviewed REASSIGN_RELATIONSHIP action.")
    if action in {"DELETE_FUEL_AND_VDE", "ADMIN_DELETE"} and preview_action is not ChangeAction.DELETE:
        raise ValueError("VDE deletion requires a reviewed DELETE action.")
    resolved_reason = str(reason or "").strip()
    if not resolved_reason:
        raise ValueError("A reason is required for VDE dependency resolution.")
    actor = actor_context or LOCAL_ADMIN_ACTOR
    selected_fuel_ids = {int(item) for item in fuel_row_ids}

    db_module.ensure_db()
    con = db_module._con()
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("BEGIN")
        before = con.execute("SELECT * FROM vde_db WHERE id=?", (int(change_preview.record_id),)).fetchone()
        if not before:
            raise ValueError(f"VDE {change_preview.record_id} no longer exists.")
        fuel_rows = con.execute("SELECT * FROM fuelcons_db WHERE vde_id=? ORDER BY id", (int(change_preview.record_id),)).fetchall()
        fuel_by_id = {int(row["id"]): row for row in fuel_rows}
        if not selected_fuel_ids <= set(fuel_by_id):
            raise ValueError("One or more selected Fuel rows no longer belong to this VDE.")

        impact = {
            "resolution_action": action,
            "fuel_row_ids": sorted(selected_fuel_ids),
            "replacement_vde_id": replacement_vde_id,
        }
        if action == "REASSIGN_FUEL":
            if replacement_vde_id is None or int(replacement_vde_id) == int(change_preview.record_id):
                raise ValueError("A different replacement VDE is required.")
            replacement = con.execute("SELECT * FROM vde_db WHERE id=?", (int(replacement_vde_id),)).fetchone()
            if not replacement or str(replacement["record_status"] or "ACTIVE").upper() != "ACTIVE":
                raise ValueError("Replacement VDE must exist and be active.")
            if not selected_fuel_ids:
                raise ValueError("Select at least one Fuel row for reassignment.")
            for fuel_id in sorted(selected_fuel_ids):
                origin = str(fuel_by_id[fuel_id]["record_origin"] or "LEGACY").strip().upper()
                if origin not in {"HOMOLOGATED", "MEASURED"}:
                    raise ValueError(
                        f"Fuel {fuel_id} is {origin}; estimated, Powertrain, and legacy rows require recalculation before reassignment."
                    )
                con.execute(
                    "UPDATE fuelcons_db SET vde_id=?, source_vde_revision=?, review_status='REVIEW_REQUIRED', updated_at=? WHERE id=?",
                    (
                        int(replacement_vde_id),
                        replacement["updated_at"] or replacement["created_at"],
                        _utc_now_iso(),
                        fuel_id,
                    ),
                )
            after = con.execute("SELECT * FROM vde_db WHERE id=?", (int(change_preview.record_id),)).fetchone()
        else:
            all_fuel_ids = set(fuel_by_id)
            if action == "DELETE_FUEL_AND_VDE" and selected_fuel_ids != all_fuel_ids:
                raise ValueError("Every linked Fuel row must be selected before deleting the VDE.")
            delete_ids = all_fuel_ids if action == "ADMIN_DELETE" else selected_fuel_ids
            for fuel_id in sorted(delete_ids):
                con.execute("DELETE FROM fuelcons_db WHERE id=?", (fuel_id,))
            remaining = con.execute("SELECT COUNT(*) FROM fuelcons_db WHERE vde_id=?", (int(change_preview.record_id),)).fetchone()[0]
            if remaining:
                raise ValueError("Linked Fuel rows remain; VDE delete is blocked.")
            con.execute("DELETE FROM vde_db WHERE id=?", (int(change_preview.record_id),))
            after = None

        log_id = append_change_log(
            change_preview,
            actor,
            reason=resolved_reason,
            before=dict(before),
            after=dict(after) if after else {},
            impact=impact,
            connection=con,
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    return ChangeResult(
        operation_id=change_preview.operation_id,
        entity_type=EntityType.VDE.value,
        action=change_preview.action,
        committed=True,
        change_log_id=log_id,
        affected_record_ids=(str(change_preview.record_id),),
    )


def preview_dependency_impact(
    change_preview: ChangePreview,
    persistence_choice: ImpactPersistenceChoice | str,
    *,
    replacement_record_id: int | str | None = None,
    component_domain: str | None = None,
) -> ImpactPreview:
    choice = normalize_impact_persistence_choice(persistence_choice)
    entity = normalize_entity_type(change_preview.entity_type)
    if entity not in _IMPACT_ENTITIES:
        raise ValueError("Impact preview is available only for Tire and Component changes.")
    if change_preview.record_id is None:
        raise ValueError("Impact preview requires an existing catalog record.")

    discovery = discover_catalog_usage(
        entity,
        change_preview.record_id,
        component_domain=component_domain,
    )
    replacement = _replacement_payload(
        entity,
        change_preview,
        replacement_record_id=replacement_record_id,
        component_domain=component_domain,
    )
    issues: list[ValidationIssue] = []
    recalculations: list[dict] = []
    if choice is not ImpactPersistenceChoice.KEEP_EXISTING:
        for request in discovery["requests"]:
            prepared = prepare_v22_maintenance_recalculation(
                request["request_history_id"],
                replacement,
                direct_proposal_ids=request["direct_proposal_ids"],
            )
            recalculations.append(_public_recalculation(prepared, request))
            if prepared.get("status") != "ready":
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "request_recalculation_blocked",
                        f"Request history {request['request_history_id']} requires review before recalculation.",
                    )
                )
        for row in discovery["review_required"]:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "untracked_vde_review_required",
                    f"VDE {row['vde_id']} uses this record but has no canonical saved request to recalculate.",
                )
            )
    can_commit = change_preview.can_commit and not any(issue.severity == "ERROR" for issue in issues)
    return ImpactPreview(
        operation_id=str(uuid4()),
        change_operation_id=change_preview.operation_id,
        entity_type=entity.value,
        record_id=str(change_preview.record_id),
        persistence_choice=choice.value,
        usages=tuple(discovery["usages"]),
        historical_usages=tuple(discovery["historical_usages"]),
        request_recalculations=tuple(recalculations),
        affected_vde_ids=tuple(discovery["affected_vde_ids"]),
        stale_fuel_rows=tuple(discovery["fuel_rows"]),
        review_required=tuple(discovery["review_required"]),
        validation_issues=tuple(issues),
        can_commit=can_commit,
    )


def apply_change_with_impact(
    change_preview: ChangePreview,
    impact_preview: ImpactPreview,
    *,
    actor_context: ActorContext | None = None,
    reason: str | None = None,
    replacement_record_id: int | str | None = None,
    component_domain: str | None = None,
    failure_injector=None,
) -> ChangeResult:
    """Commit catalog mutation and every selected recalculation atomically."""
    if not change_preview.can_commit or not impact_preview.can_commit:
        raise ValueError("Catalog change and impact preview must both be ready before commit.")
    if impact_preview.change_operation_id != change_preview.operation_id:
        raise ValueError("Impact preview does not belong to this catalog change.")
    choice = normalize_impact_persistence_choice(impact_preview.persistence_choice)
    entity = normalize_entity_type(change_preview.entity_type)
    action = normalize_change_action(change_preview.action)
    actor = actor_context or LOCAL_ADMIN_ACTOR
    resolved_reason = str(reason or "").strip() or None
    if change_preview.requires_reason and not resolved_reason:
        raise ValueError("A reason is required before this change can be committed.")

    discovery = discover_catalog_usage(entity, change_preview.record_id, component_domain=component_domain)
    if tuple(discovery["affected_vde_ids"]) != tuple(impact_preview.affected_vde_ids):
        raise ValueError("Catalog dependencies changed after impact preview; review the impact again before committing.")
    current_usage_keys = {
        (int(row["request_history_id"]), str(row["proposal_id"]), str(row["relation"]))
        for row in discovery["usages"]
    }
    preview_usage_keys = {
        (int(row["request_history_id"]), str(row["proposal_id"]), str(row["relation"]))
        for row in impact_preview.usages
    }
    if current_usage_keys != preview_usage_keys:
        raise ValueError("Catalog request usages changed after impact preview; review the impact again before committing.")
    current_historical_keys = {
        (int(row["request_history_id"]), str(row["proposal_id"]), row.get("saved_vde_row_id"))
        for row in discovery["historical_usages"]
    }
    preview_historical_keys = {
        (int(row["request_history_id"]), str(row["proposal_id"]), row.get("saved_vde_row_id"))
        for row in impact_preview.historical_usages
    }
    if current_historical_keys != preview_historical_keys:
        raise ValueError("Historical catalog usages changed after impact preview; review the impact again before committing.")
    replacement = _replacement_payload(
        entity,
        change_preview,
        replacement_record_id=replacement_record_id,
        component_domain=component_domain,
    )
    prepared_requests = []
    if choice is not ImpactPersistenceChoice.KEEP_EXISTING:
        for request in discovery["requests"]:
            prepared = prepare_v22_maintenance_recalculation(
                request["request_history_id"],
                replacement,
                direct_proposal_ids=request["direct_proposal_ids"],
            )
            if prepared.get("status") != "ready":
                raise ValueError(f"Request history {request['request_history_id']} is no longer eligible for recalculation.")
            prepared_requests.append(prepared)
        if discovery["review_required"]:
            raise ValueError("Untracked VDE usages require manual review before automatic recalculation.")

    db_module.ensure_db()
    con = db_module._con()
    con.row_factory = sqlite3.Row
    request_history_ids: list[int] = []
    stale_fuel_ids: set[int] = set()
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("BEGIN")
        before = _fetch_record(con, entity, change_preview.record_id, component_domain=component_domain)
        if before is None:
            raise ValueError(f"Record {change_preview.record_id!r} no longer exists.")
        if action is ChangeAction.REASSIGN_RELATIONSHIP:
            affected_id = int(before["id"])
            after = before
        else:
            affected_id = _apply_preview(con, entity, action, change_preview, before)
            after = _fetch_record(con, entity, affected_id, component_domain=component_domain)

        if choice is not ImpactPersistenceChoice.KEEP_EXISTING:
            for index, prepared in enumerate(prepared_requests):
                persisted = persist_v22_maintenance_recalculation(
                    con,
                    prepared,
                    strategy=choice.value,
                )
                request_history_ids.append(int(persisted["request_history_id"]))
                stale_fuel_ids.update(int(item) for item in persisted.get("stale_fuel_row_ids") or [])
                if failure_injector is not None:
                    failure_injector(index, persisted)

        log_id = append_change_log(
            change_preview,
            actor,
            reason=resolved_reason,
            before=_adapt_component_row(before) if entity is EntityType.COMPONENT else dict(before),
            after=_adapt_component_row(after) if entity is EntityType.COMPONENT else dict(after or {}),
            impact={
                "persistence_choice": choice.value,
                "affected_vde_ids": list(discovery["affected_vde_ids"]),
                "request_history_ids": request_history_ids,
                "stale_fuel_row_ids": sorted(stale_fuel_ids),
                "review_required": list(discovery["review_required"]),
                "historical_usages": list(discovery["historical_usages"]),
                "replacement_record_id": replacement_record_id,
                "request_recalculations": [
                    deepcopy(dict(row)) for row in impact_preview.request_recalculations
                ],
            },
            connection=con,
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    return ChangeResult(
        operation_id=change_preview.operation_id,
        entity_type=entity.value,
        action=change_preview.action,
        committed=True,
        change_log_id=log_id,
        affected_record_ids=(str(affected_id),),
        request_history_ids=tuple(request_history_ids),
        stale_fuel_row_ids=tuple(sorted(stale_fuel_ids)),
    )


def _replacement_payload(
    entity: EntityType,
    change_preview: ChangePreview,
    *,
    replacement_record_id: int | str | None,
    component_domain: str | None,
) -> dict:
    with db_module._con() as con:
        con.row_factory = sqlite3.Row
        if replacement_record_id is not None:
            replacement_row = _fetch_record(con, entity, replacement_record_id, component_domain=component_domain)
        else:
            current = _fetch_record(con, entity, change_preview.record_id, component_domain=component_domain)
            if current is None:
                raise ValueError(f"Record {change_preview.record_id!r} no longer exists.")
            replacement_row = _normalized_replacement_record(entity, dict(current), dict(change_preview.normalized_payload))
    if not replacement_row:
        raise ValueError("Replacement record does not exist.")
    if entity is EntityType.COMPONENT:
        old_view = discover_catalog_usage(entity, change_preview.record_id, component_domain=component_domain)["record"]
        replacement_view = _adapt_component_row(dict(replacement_row)) if "component_id" not in replacement_row else dict(replacement_row)
        return {
            "entity_type": entity.value,
            "old_record_id": str(change_preview.record_id),
            "old_component_id": old_view.get("component_id") or old_view.get("component_code"),
            "domain": replacement_view.get("domain") or component_domain,
            "replacement_record": replacement_view,
        }
    return {
        "entity_type": entity.value,
        "old_record_id": str(change_preview.record_id),
        "replacement_record": dict(replacement_row),
    }


def _normalized_replacement_record(entity: EntityType, current: dict, payload: dict) -> dict:
    merged = {**current, **payload}
    if entity is EntityType.TIRE:
        return _normalize_tire_payload(merged)
    domain = _normalize_domain_key(merged.get("domain"))
    storage = _component_to_storage(domain, merged, default_origin=current.get("record_origin") or "LEGACY")
    storage["id"] = current.get("id")
    return _db_row_to_component(storage)


def _proposal_uses_record(entity: EntityType, target: dict, row: dict) -> bool:
    actions = _json_object(row.get("component_actions_json"), default=[])
    if entity is EntityType.TIRE:
        target_id = str(target.get("id") or "")
        for action in actions:
            if str(action.get("domain") or "") == "tire" and str(action.get("component_id") or "") == target_id:
                return True
        inputs = dict(_json_object(row.get("applied_inputs_json"), default={}).get("tire") or {})
        return str(inputs.get("tire_db_id") or "") == target_id

    target_code = str(target.get("component_id") or target.get("component_code") or "")
    target_domain = str(target.get("domain") or "")
    for action in actions:
        if str(action.get("domain") or "") == target_domain and str(action.get("component_id") or "") == target_code:
            return True
    inputs = dict(_json_object(row.get("applied_inputs_json"), default={}).get(target_domain) or {})
    input_field = {
        "transmission": "transmission_component_db_id",
        "brake": "brake_component_db_id",
        "axle_hubs": "axle_hubs_component_db_id",
        "parasitic": "parasitic_component_db_id",
    }.get(target_domain)
    return bool(input_field and str(inputs.get(input_field) or "") == target_code)


def _untracked_vde_usages(
    con: sqlite3.Connection,
    entity: EntityType,
    target: dict,
    tracked_vde_ids: set[int],
) -> list[dict]:
    if entity is not EntityType.TIRE:
        return []
    target_id = int(target["id"])
    rows = con.execute(
        "SELECT id, make, model, year FROM vde_db WHERE front_tire_id=? OR rear_tire_id=? ORDER BY id",
        (target_id, target_id),
    ).fetchall()
    return [
        {"vde_id": int(row["id"]), "make": row["make"], "model": row["model"], "year": row["year"]}
        for row in rows
        if int(row["id"]) not in tracked_vde_ids
    ]


def _fuel_dependencies(con: sqlite3.Connection, vde_ids: tuple[int, ...]) -> list[dict]:
    if not vde_ids:
        return []
    placeholders = ",".join("?" for _ in vde_ids)
    rows = con.execute(
        f"SELECT id, vde_id, record_origin, review_status FROM fuelcons_db WHERE vde_id IN ({placeholders}) ORDER BY id",
        vde_ids,
    ).fetchall()
    return [dict(row) for row in rows]


def _public_recalculation(prepared: dict, request: dict) -> dict:
    return {
        "request_history_id": request["request_history_id"],
        "record_key": request.get("record_key"),
        "status": prepared.get("status"),
        "saved_vde_row_ids": tuple(request.get("saved_vde_row_ids") or ()),
        "comparisons": tuple(deepcopy(list(prepared.get("comparisons") or []))),
        "issues": tuple(deepcopy(list(prepared.get("issues") or []))),
    }


def _walk_from_descendants(request_rows: list[dict], direct_ids: set[str]) -> set[str]:
    affected_ids = set(direct_ids)
    changed = True
    while changed:
        changed = False
        for row in request_rows:
            proposal_id = str(row.get("proposal_id") or "")
            parent_id = str(row.get("walk_from_proposal_id") or "")
            if proposal_id not in affected_ids and parent_id in affected_ids:
                affected_ids.add(proposal_id)
                changed = True
    return affected_ids


def _json_object(payload, *, default):
    if payload in (None, ""):
        return deepcopy(default)
    if isinstance(payload, (dict, list)):
        return deepcopy(payload)
    try:
        value = json.loads(payload)
    except (TypeError, ValueError):
        return deepcopy(default)
    return value if isinstance(value, type(default)) else deepcopy(default)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "apply_change_with_impact",
    "apply_vde_dependency_resolution",
    "discover_catalog_usage",
    "discover_vde_dependencies",
    "preview_dependency_impact",
]
