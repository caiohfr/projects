from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import sqlite3

from src.vde_core.component_repositories import ComponentRepository
from src.vde_core import db as db_module
from src.vde_core.db import ensure_db, table_columns
from src.vde_core.services import autoresolve_test_mass
from src.vde_core.vde_request_compact_state import (
    build_v22_canonical_request_draft,
    normalize_v22_state,
    resolve_v22_metadata_contexts,
)
from src.vde_core.vde_request_compact_adapter import build_v22_preview_bundle, compact_baseline_context
from src.vde_core.vde_request_finalization import build_scenario_configuration_summaries, suggested_scenario_name
from src.vde_core.vde_request_preview import build_validation_summary, validation_allows_save
from src.vde_core.vde_request_report import build_vde_request_report_model
from src.vde_core.vde_request_contract import is_blank
from src.vde_core.vde_request_save import (
    SAVE_MODE_SELECTED,
    build_vde_request_save_plan,
)


REQUEST_HISTORY_TABLE = "vde_request_history"
REQUEST_HISTORY_PROPOSAL_TABLE = "vde_request_history_proposals"
_COMPONENT_DOMAINS = ("transmission", "brake", "axle_hubs", "parasitic")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(payload) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _json_loads(payload):
    if payload in (None, ""):
        return None
    return json.loads(payload)


def _clean_text(value) -> str:
    return str(value or "").strip()


def _ensure_request_history_tables(con: sqlite3.Connection) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {REQUEST_HISTORY_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            record_key TEXT NOT NULL UNIQUE,
            save_plan_operation_id TEXT,
            source_type TEXT,
            interface TEXT,
            schema_version TEXT,
            template_version TEXT,
            baseline_vde_id INTEGER,
            legislation TEXT,
            cycle_name TEXT,
            fingerprint TEXT,
            validation_status TEXT,
            save_status TEXT NOT NULL,
            state_json TEXT NOT NULL,
            draft_json TEXT NOT NULL,
            preview_bundle_json TEXT NOT NULL,
            save_result_json TEXT
        );
        """
    )
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {REQUEST_HISTORY_PROPOSAL_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_history_id INTEGER NOT NULL REFERENCES {REQUEST_HISTORY_TABLE}(id) ON DELETE CASCADE,
            proposal_id TEXT NOT NULL,
            display_index INTEGER,
            source_column TEXT,
            walk_from_kind TEXT,
            walk_from_proposal_id TEXT,
            walk_from_source_column TEXT,
            effective_metadata_json TEXT,
            metadata_overrides_json TEXT,
            domain_requests_json TEXT NOT NULL,
            applied_inputs_json TEXT NOT NULL,
            resolved_snapshot_json TEXT,
            preview_summary_json TEXT,
            issues_json TEXT,
            component_actions_json TEXT,
            abc_total_json TEXT,
            abc_net_json TEXT,
            vde_results_json TEXT,
            saved_vde_row_id INTEGER
        );
        """
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{REQUEST_HISTORY_TABLE}_baseline ON {REQUEST_HISTORY_TABLE}(baseline_vde_id, created_at);"
    )
    con.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{REQUEST_HISTORY_PROPOSAL_TABLE}_request ON {REQUEST_HISTORY_PROPOSAL_TABLE}(request_history_id, display_index);"
    )


def _default_services() -> dict:
    return {
        "ensure_db": ensure_db,
        "connect_db": lambda: sqlite3.connect(str(db_module.current_db_path()), timeout=30),
        "table_columns": table_columns,
        "insert_vde_row": _insert_vde_row,
    }


def _insert_vde_row(con: sqlite3.Connection, row_payload: dict, supported_columns: set[str]) -> int:
    filtered = {key: value for key, value in dict(row_payload or {}).items() if key in supported_columns}
    payload = autoresolve_test_mass(filtered)
    columns = list(payload.keys())
    placeholders = ",".join("?" for _ in columns)
    cur = con.cursor()
    cur.execute(
        f"INSERT INTO vde_db ({','.join(columns)}) VALUES ({placeholders})",
        [payload[column] for column in columns],
    )
    return int(cur.lastrowid)


def _fingerprint_from_state(state: dict, bundle: dict) -> str | None:
    preview = dict(dict(state or {}).get("preview") or {})
    return _clean_text(preview.get("fingerprint")) or _clean_text(bundle.get("fingerprint")) or None


def _build_record_key(fingerprint: str | None, proposal_ids: list[str]) -> str:
    seed = _json_dumps(
        {
            "fingerprint": fingerprint,
            "proposal_ids": proposal_ids,
            "generated_at": _utc_now_iso(),
        }
    )
    return "v22req_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _component_snapshot_map_from_bundle(bundle: dict | None) -> dict[str, dict[str, dict]]:
    snapshots: dict[str, dict[str, dict]] = {}
    resolution = dict(dict(bundle or {}).get("resolution_result") or {})
    for proposal_result in list(resolution.get("proposal_results") or []):
        for action in list(dict(proposal_result or {}).get("component_actions") or []):
            payload = dict(action or {})
            domain = _clean_text(payload.get("domain"))
            component_id = _clean_text(payload.get("component_id"))
            snapshot = deepcopy(dict(payload.get("component_snapshot") or {}))
            if domain not in _COMPONENT_DOMAINS or not component_id or not snapshot:
                continue
            snapshots.setdefault(domain, {})[component_id] = snapshot
    return snapshots


def saved_component_repositories_from_state(state: dict | None) -> dict[str, ComponentRepository] | None:
    snapshot_map = deepcopy(dict(dict(state or {}).get("saved_component_repository_snapshots") or {}))
    repositories: dict[str, ComponentRepository] = {}
    for domain, entries in snapshot_map.items():
        components = []
        by_id = {}
        for component_id, snapshot in dict(entries or {}).items():
            component = deepcopy(dict(snapshot or {}))
            if not component:
                continue
            component.setdefault("component_id", str(component_id))
            component.setdefault("domain", str(domain))
            components.append(component)
            by_id[str(component.get("component_id") or component_id)] = deepcopy(component)
        if not components:
            continue
        repositories[str(domain)] = ComponentRepository(
            domain=str(domain),
            source="saved_request_snapshot",
            _components=components,
            _issues=[],
            _by_id=by_id,
        )
    return repositories or None


def _build_history_state(state: dict, bundle: dict, fingerprint: str | None) -> dict:
    historical_state = normalize_v22_state(state)
    historical_state["preview"] = {
        "status": "fresh",
        "fingerprint": fingerprint,
        "result": None,
    }
    historical_state["save"] = {
        "status": "pending",
        "result": None,
    }
    historical_state["saved_component_repository_snapshots"] = _component_snapshot_map_from_bundle(bundle)
    historical_state.pop("report", None)
    return historical_state


def _proposal_history_rows(
    state: dict,
    bundle: dict,
    saved_vde_row_ids: dict[str, int],
) -> list[dict]:
    normalized = normalize_v22_state(state)
    draft = deepcopy(dict(bundle.get("draft") or build_v22_canonical_request_draft(normalized)))
    proposal_models = {
        str(item.get("proposal_id") or ""): dict(item)
        for item in list(dict(bundle.get("resolution_result") or {}).get("proposal_results") or [])
    }
    applied_inputs_by_id = {
        str(item.get("proposal_id") or ""): deepcopy(dict(item.get("inputs") or {}))
        for item in list(normalized.get("proposals") or [])
    }
    metadata_overrides_by_id = {
        str(item.get("proposal_id") or ""): deepcopy(dict(item.get("metadata_overrides") or {}))
        for item in list(normalized.get("proposals") or [])
    }
    rows: list[dict] = []
    for proposal in list(draft.get("proposals") or []):
        proposal_id = str(proposal.get("proposal_id") or "")
        proposal_result = proposal_models.get(proposal_id, {})
        walk_from = dict(proposal.get("walk_from") or {})
        rows.append(
            {
                "proposal_id": proposal_id,
                "display_index": int(proposal.get("display_index") or 0),
                "source_column": proposal.get("source_column"),
                "walk_from_kind": walk_from.get("kind"),
                "walk_from_proposal_id": walk_from.get("proposal_id"),
                "walk_from_source_column": walk_from.get("source_column"),
                "effective_metadata_json": _json_dumps(dict(proposal.get("effective_metadata") or {})),
                "metadata_overrides_json": _json_dumps(metadata_overrides_by_id.get(proposal_id, {})),
                "domain_requests_json": _json_dumps(dict(proposal.get("domain_requests") or {})),
                "applied_inputs_json": _json_dumps(applied_inputs_by_id.get(proposal_id, {})),
                "resolved_snapshot_json": _json_dumps(dict(proposal_result.get("resolved_snapshot") or {})),
                "preview_summary_json": _json_dumps(dict(proposal_result.get("preview_summary") or {})),
                "issues_json": _json_dumps(list(proposal_result.get("issues") or [])),
                "component_actions_json": _json_dumps(list(proposal_result.get("component_actions") or [])),
                "abc_total_json": _json_dumps(dict(proposal_result.get("abc_total") or {})),
                "abc_net_json": _json_dumps(dict(proposal_result.get("abc_net") or {})),
                "vde_results_json": _json_dumps(dict(proposal_result.get("vde_results") or {})),
                "saved_vde_row_id": saved_vde_row_ids.get(proposal_id),
            }
        )
    return rows


def _final_metadata_row_payload(proposal_row: dict, metadata: dict) -> dict:
    """Apply persistence-only identity fields to an already resolved DB row."""
    row_payload = deepcopy(dict(proposal_row.get("row_payload") or {}))
    final_name = _clean_text(metadata.get("name")) or _clean_text(proposal_row.get("final_name"))
    description = _clean_text(metadata.get("description"))
    field_map = {
        "make": "make",
        "model": "model",
        "model_year": "year",
        "category": "category",
        "electrification": "electrification",
        "transmission_type": "transmission_type",
        "drive_type": "drive_type",
        "fuel_type": "fuel_type",
    }
    for metadata_key, payload_key in field_map.items():
        value = metadata.get(metadata_key)
        if not is_blank(value):
            row_payload[payload_key] = value
    note_parts = [item for item in (final_name, description, _clean_text(proposal_row.get("note_text"))) if item]
    row_payload["notes"] = "\n".join(dict.fromkeys(note_parts))
    proposal_row["final_name"] = final_name
    proposal_row["user_notes"] = description
    proposal_row["note_text"] = row_payload["notes"]
    proposal_row["row_payload"] = row_payload
    return proposal_row


def build_v22_save_plan(state: dict) -> dict:
    """Build the single save payload used by both DB preview and execution."""
    normalized = normalize_v22_state(state)
    preview = dict(normalized.get("preview") or {})
    bundle = deepcopy(dict(preview.get("result") or {}))
    if str(preview.get("status") or "") != "fresh" or not bundle:
        return {
            "status": "blocked",
            "can_execute": False,
            "proposals": [],
            "proposals_to_save": [],
            "blocking_issues": [{"code": "preview_not_fresh", "severity": "blocked", "message": "Run Validate & Preview before saving."}],
            "warnings": [],
        }

    resolution = deepcopy(dict(bundle.get("resolution_result") or {}))
    proposal_ids = [str(item.get("proposal_id") or "") for item in list(resolution.get("proposal_results") or []) if _clean_text(item.get("proposal_id"))]
    fingerprint = _fingerprint_from_state(normalized, bundle)
    baseline_effective = deepcopy(dict(dict(normalized.get("baseline") or {}).get("effective") or {}))
    baseline_update_choices = None
    if _clean_text(baseline_effective.get("selected_baseline_vde_id")) == "":
        baseline_update_choices = {
            str(field_key): False
            for field_key in list(dict(resolution.get("baseline") or {}).get("corrected_fields") or [])
        }
    plan = build_vde_request_save_plan(
        resolution,
        save_mode=SAVE_MODE_SELECTED,
        selected_proposal_ids=proposal_ids,
        review_confirmations={proposal_id: True for proposal_id in proposal_ids},
        baseline_update_choices=baseline_update_choices,
        request_state=deepcopy(dict(bundle.get("workbook_state") or {})),
        current_fingerprint=fingerprint,
        resolution_fingerprint=fingerprint,
    )
    contexts = resolve_v22_metadata_contexts(normalized)
    summaries_by_id = {
        str(item.get("proposal_id") or ""): dict(item)
        for item in build_scenario_configuration_summaries(normalized)
    }
    for row in list(plan.get("proposals") or []):
        proposal_id = str(row.get("proposal_id") or "")
        metadata = dict(dict(contexts.get(proposal_id) or {}).get("effective_metadata") or {})
        metadata["name"] = suggested_scenario_name(summaries_by_id.get(proposal_id, {}), metadata.get("name"))
        _final_metadata_row_payload(row, metadata)
    plan["final_metadata_by_proposal"] = {
        proposal_id: {
            **deepcopy(dict(context.get("effective_metadata") or {})),
            "name": suggested_scenario_name(
                summaries_by_id.get(proposal_id, {}),
                dict(context.get("effective_metadata") or {}).get("name"),
            ),
        }
        for proposal_id, context in contexts.items()
    }
    plan["configuration_summaries"] = list(summaries_by_id.values())
    return plan


def save_v22_request(state: dict, *, services: dict | None = None) -> dict:
    normalized = normalize_v22_state(state)
    preview = dict(normalized.get("preview") or {})
    bundle = deepcopy(dict(preview.get("result") or {}))
    result = {
        "status": "failed",
        "record_id": None,
        "record_key": None,
        "save_plan_operation_id": None,
        "executed_at": _utc_now_iso(),
        "saved_proposals": [],
        "issues": [],
    }
    if str(preview.get("status") or "") != "fresh" or not bundle:
        result["issues"].append({"code": "preview_not_fresh", "severity": "error", "message": "Run Validate & Preview before saving."})
        return result

    validation = dict(bundle.get("validation_summary") or build_validation_summary(dict(bundle.get("resolution_result") or {})))
    if not validation_allows_save(validation):
        result["issues"].append(
            {
                "code": "validation_not_savable",
                "severity": "error",
                "message": "Save requires a fresh preview without Missing, Invalid, or Blocked issues.",
            }
        )
        return result

    resolution = deepcopy(dict(bundle.get("resolution_result") or {}))
    proposal_ids = [str(item.get("proposal_id") or "") for item in list(resolution.get("proposal_results") or []) if _clean_text(item.get("proposal_id"))]
    fingerprint = _fingerprint_from_state(normalized, bundle)
    baseline_effective = deepcopy(dict(dict(normalized.get("baseline") or {}).get("effective") or {}))
    save_plan = build_v22_save_plan(normalized)
    result["save_plan_operation_id"] = save_plan.get("operation_id")
    if not save_plan.get("can_execute"):
        result["issues"] = deepcopy(list(save_plan.get("blocking_issues") or [])) or [
            {"code": "save_plan_blocked", "severity": "error", "message": "Save plan is not executable."}
        ]
        return result

    draft = deepcopy(dict(bundle.get("draft") or build_v22_canonical_request_draft(normalized)))
    historical_state = _build_history_state(normalized, bundle, fingerprint)
    source = deepcopy(dict(draft.get("source") or {}))
    record_key = _build_record_key(fingerprint, proposal_ids)
    service_map = _default_services()
    service_map.update(dict(services or {}))

    try:
        service_map["ensure_db"]()
        supported_columns = set(service_map["table_columns"]("vde_db"))
        con = service_map["connect_db"]()
        con.execute("PRAGMA foreign_keys = ON")
        _ensure_request_history_tables(con)
        con.execute("BEGIN")
        try:
            cur = con.cursor()
            cur.execute(
                f"""
                INSERT INTO {REQUEST_HISTORY_TABLE} (
                    record_key,
                    save_plan_operation_id,
                    source_type,
                    interface,
                    schema_version,
                    template_version,
                    baseline_vde_id,
                    legislation,
                    cycle_name,
                    fingerprint,
                    validation_status,
                    save_status,
                    state_json,
                    draft_json,
                    preview_bundle_json,
                    save_result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record_key,
                    save_plan.get("operation_id"),
                    source.get("source_type"),
                    source.get("interface"),
                    draft.get("schema_version"),
                    draft.get("template_version"),
                    baseline_effective.get("selected_baseline_vde_id"),
                    baseline_effective.get("legislation"),
                    baseline_effective.get("cycle_name"),
                    fingerprint,
                    validation.get("overall_status"),
                    "saving",
                    _json_dumps(historical_state),
                    _json_dumps(draft),
                    _json_dumps(bundle),
                    None,
                ],
            )
            record_id = int(cur.lastrowid)

            saved_vde_row_ids: dict[str, int] = {}
            history_only_proposal_ids: set[str] = set()
            for proposal_row in list(save_plan.get("proposals_to_save") or []):
                proposal_id = str(proposal_row.get("proposal_id") or "")
                row_payload = dict(proposal_row.get("row_payload") or {})
                if str(dict(normalized.get("baseline") or {}).get("source_type") or "").strip().upper() == "NEW_TEST" and _clean_text(row_payload.get("mass_kg")) == "":
                    history_only_proposal_ids.add(proposal_id)
                    continue
                saved_vde_row_ids[proposal_id] = int(
                    service_map["insert_vde_row"](
                        con,
                        row_payload,
                        supported_columns,
                    )
                )

            for row in _proposal_history_rows(normalized, bundle, saved_vde_row_ids):
                con.execute(
                    f"""
                    INSERT INTO {REQUEST_HISTORY_PROPOSAL_TABLE} (
                        request_history_id,
                        proposal_id,
                        display_index,
                        source_column,
                        walk_from_kind,
                        walk_from_proposal_id,
                        walk_from_source_column,
                        effective_metadata_json,
                        metadata_overrides_json,
                        domain_requests_json,
                        applied_inputs_json,
                        resolved_snapshot_json,
                        preview_summary_json,
                        issues_json,
                        component_actions_json,
                        abc_total_json,
                        abc_net_json,
                        vde_results_json,
                        saved_vde_row_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record_id,
                        row.get("proposal_id"),
                        row.get("display_index"),
                        row.get("source_column"),
                        row.get("walk_from_kind"),
                        row.get("walk_from_proposal_id"),
                        row.get("walk_from_source_column"),
                        row.get("effective_metadata_json"),
                        row.get("metadata_overrides_json"),
                        row.get("domain_requests_json"),
                        row.get("applied_inputs_json"),
                        row.get("resolved_snapshot_json"),
                        row.get("preview_summary_json"),
                        row.get("issues_json"),
                        row.get("component_actions_json"),
                        row.get("abc_total_json"),
                        row.get("abc_net_json"),
                        row.get("vde_results_json"),
                        row.get("saved_vde_row_id"),
                    ],
                )

            save_result = {
                "status": "success",
                "record_id": record_id,
                "record_key": record_key,
                "save_plan_operation_id": save_plan.get("operation_id"),
                "executed_at": result["executed_at"],
                "saved_proposals": [
                    {
                        "proposal_id": proposal_row.get("proposal_id"),
                        "vde_row_id": saved_vde_row_ids.get(str(proposal_row.get("proposal_id") or "")),
                        "status": "history_only" if str(proposal_row.get("proposal_id") or "") in history_only_proposal_ids else "saved",
                    }
                    for proposal_row in list(save_plan.get("proposals_to_save") or [])
                ],
                "issues": [],
            }
            con.execute(
                f"UPDATE {REQUEST_HISTORY_TABLE} SET save_status=?, save_result_json=? WHERE id=?",
                ["success", _json_dumps(save_result), record_id],
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
    except Exception as exc:
        result["issues"].append({"code": "db_save_failed", "severity": "error", "message": str(exc)})
        return result

    result.update(save_result)
    return result


def load_v22_saved_request(record_id: int | str, *, services: dict | None = None) -> dict | None:
    service_map = _default_services()
    service_map.update(dict(services or {}))
    service_map["ensure_db"]()
    con = service_map["connect_db"]()
    con.row_factory = sqlite3.Row
    try:
        _ensure_request_history_tables(con)
        parent = con.execute(
            f"SELECT * FROM {REQUEST_HISTORY_TABLE} WHERE id=?",
            [int(record_id)],
        ).fetchone()
        if parent is None:
            return None
        proposal_rows = con.execute(
            f"SELECT * FROM {REQUEST_HISTORY_PROPOSAL_TABLE} WHERE request_history_id=? ORDER BY display_index, id",
            [int(record_id)],
        ).fetchall()
    finally:
        con.close()

    preview_bundle = deepcopy(dict(_json_loads(parent["preview_bundle_json"]) or {}))
    save_result = deepcopy(dict(_json_loads(parent["save_result_json"]) or {}))
    state = normalize_v22_state(_json_loads(parent["state_json"]) or {})
    state["preview"] = {
        "status": "fresh",
        "fingerprint": parent["fingerprint"] or preview_bundle.get("fingerprint"),
        "result": preview_bundle,
    }
    state["save"] = {
        "status": parent["save_status"] or save_result.get("status") or "success",
        "result": save_result,
    }
    state["saved_request"] = {
        "record_id": int(parent["id"]),
        "record_key": parent["record_key"],
        "save_plan_operation_id": parent["save_plan_operation_id"],
        "created_at": parent["created_at"],
    }
    proposal_records = []
    for row in proposal_rows:
        proposal_records.append(
            {
                "proposal_id": row["proposal_id"],
                "display_index": row["display_index"],
                "source_column": row["source_column"],
                "walk_from_kind": row["walk_from_kind"],
                "walk_from_proposal_id": row["walk_from_proposal_id"],
                "walk_from_source_column": row["walk_from_source_column"],
                "effective_metadata": _json_loads(row["effective_metadata_json"]) or {},
                "metadata_overrides": _json_loads(row["metadata_overrides_json"]) or {},
                "domain_requests": _json_loads(row["domain_requests_json"]) or {},
                "applied_inputs": _json_loads(row["applied_inputs_json"]) or {},
                "resolved_snapshot": _json_loads(row["resolved_snapshot_json"]) or {},
                "preview_summary": _json_loads(row["preview_summary_json"]) or {},
                "issues": _json_loads(row["issues_json"]) or [],
                "component_actions": _json_loads(row["component_actions_json"]) or [],
                "abc_total": _json_loads(row["abc_total_json"]) or {},
                "abc_net": _json_loads(row["abc_net_json"]) or {},
                "vde_results": _json_loads(row["vde_results_json"]) or {},
                "saved_vde_row_id": row["saved_vde_row_id"],
            }
        )
    return {
        "record_id": int(parent["id"]),
        "record_key": parent["record_key"],
        "created_at": parent["created_at"],
        "fingerprint": parent["fingerprint"],
        "state": state,
        "draft": _json_loads(parent["draft_json"]) or {},
        "preview_bundle": preview_bundle,
        "save_result": save_result,
        "proposal_records": proposal_records,
        "component_repositories": saved_component_repositories_from_state(state),
        "report_model": build_vde_request_report_model(
            _json_loads(parent["draft_json"]) or {},
            dict(preview_bundle.get("resolution_result") or {}),
            save_result,
        ),
    }


def prepare_v22_maintenance_recalculation(
    record_id: int | str,
    replacement: dict,
    *,
    direct_proposal_ids: tuple[str, ...] | list[str] = (),
    services: dict | None = None,
) -> dict:
    """Resolve a saved request with an explicit catalog snapshot replacement."""
    loaded = load_v22_saved_request(record_id, services=services)
    if not loaded:
        return {
            "status": "review_required",
            "request_history_id": int(record_id),
            "issues": [{"code": "request_history_missing", "message": "Saved request history is unavailable."}],
        }

    state = _state_with_catalog_replacement(
        loaded["state"],
        replacement,
        direct_proposal_ids={str(item) for item in direct_proposal_ids},
    )
    repositories = saved_component_repositories_from_state(state)
    bundle = build_v22_preview_bundle(
        state,
        baseline_context=compact_baseline_context(state),
        component_repositories=repositories,
    )
    state["preview"] = {
        "status": "fresh",
        "fingerprint": bundle.get("fingerprint"),
        "result": bundle,
    }
    state["save"] = {"status": "pending", "result": None}
    validation = dict(bundle.get("validation_summary") or {})
    save_plan = build_v22_save_plan(state)
    old_vde_ids = {
        str(row.get("proposal_id") or ""): int(row["saved_vde_row_id"])
        for row in list(loaded.get("proposal_records") or [])
        if row.get("saved_vde_row_id") is not None
    }
    blocking = []
    if not validation_allows_save(validation):
        blocking.append({"code": "validation_not_savable", "message": "Recalculation contains blocking validation issues."})
    if not save_plan.get("can_execute"):
        blocking.extend(list(save_plan.get("blocking_issues") or []))
    missing_vde_ids = [
        str(row.get("proposal_id") or "")
        for row in list(save_plan.get("proposals_to_save") or [])
        if str(row.get("proposal_id") or "") not in old_vde_ids
    ]
    if missing_vde_ids:
        blocking.append(
            {
                "code": "saved_vde_link_missing",
                "message": "Saved VDE link is unavailable for: " + ", ".join(missing_vde_ids) + ".",
            }
        )

    return {
        "status": "ready" if not blocking else "review_required",
        "request_history_id": int(loaded["record_id"]),
        "source_record_key": loaded.get("record_key"),
        "state": state,
        "bundle": bundle,
        "save_plan": save_plan,
        "old_vde_ids": old_vde_ids,
        "comparisons": _maintenance_comparisons(loaded, bundle),
        "issues": blocking,
    }


def persist_v22_maintenance_recalculation(
    con: sqlite3.Connection,
    prepared: dict,
    *,
    strategy: str,
) -> dict:
    """Persist one prepared request revision using an existing transaction."""
    normalized_strategy = str(strategy or "").strip().upper()
    if normalized_strategy not in {"RECALCULATE_UPDATE", "RECALCULATE_NEW"}:
        raise ValueError("Maintenance strategy must be RECALCULATE_UPDATE or RECALCULATE_NEW.")
    if str(prepared.get("status") or "") != "ready":
        raise ValueError("Only a ready maintenance recalculation can be persisted.")

    state = normalize_v22_state(prepared.get("state") or {})
    bundle = deepcopy(dict(prepared.get("bundle") or {}))
    save_plan = deepcopy(dict(prepared.get("save_plan") or {}))
    old_vde_ids = {str(key): int(value) for key, value in dict(prepared.get("old_vde_ids") or {}).items()}
    validation = dict(bundle.get("validation_summary") or {})
    draft = deepcopy(dict(bundle.get("draft") or build_v22_canonical_request_draft(state)))
    fingerprint = _fingerprint_from_state(state, bundle)
    historical_state = _build_history_state(state, bundle, fingerprint)
    source = deepcopy(dict(draft.get("source") or {}))
    baseline_effective = deepcopy(dict(dict(state.get("baseline") or {}).get("effective") or {}))
    proposal_ids = [
        str(row.get("proposal_id") or "")
        for row in list(save_plan.get("proposals_to_save") or [])
    ]
    record_key = _build_record_key(fingerprint, proposal_ids)
    _ensure_request_history_tables(con)
    supported_columns = {str(row[1]) for row in con.execute("PRAGMA table_info(vde_db)").fetchall()}

    cursor = con.execute(
        f"""
        INSERT INTO {REQUEST_HISTORY_TABLE} (
            record_key, save_plan_operation_id, source_type, interface,
            schema_version, template_version, baseline_vde_id, legislation,
            cycle_name, fingerprint, validation_status, save_status,
            state_json, draft_json, preview_bundle_json, save_result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_key,
            save_plan.get("operation_id"),
            source.get("source_type"),
            source.get("interface"),
            draft.get("schema_version"),
            draft.get("template_version"),
            baseline_effective.get("selected_baseline_vde_id"),
            baseline_effective.get("legislation"),
            baseline_effective.get("cycle_name"),
            fingerprint,
            validation.get("overall_status"),
            "saving",
            _json_dumps(historical_state),
            _json_dumps(draft),
            _json_dumps(bundle),
            None,
        ),
    )
    request_history_id = int(cursor.lastrowid)
    saved_vde_row_ids: dict[str, int] = {}
    stale_fuel_row_ids: list[int] = []

    for proposal_row in list(save_plan.get("proposals_to_save") or []):
        proposal_id = str(proposal_row.get("proposal_id") or "")
        row_payload = dict(proposal_row.get("row_payload") or {})
        if normalized_strategy == "RECALCULATE_UPDATE":
            target_id = old_vde_ids.get(proposal_id)
            if target_id is None:
                raise ValueError(f"Saved VDE link is missing for {proposal_id}.")
            before_row = con.execute("SELECT * FROM vde_db WHERE id=?", (target_id,)).fetchone()
            if before_row is None:
                raise ValueError(f"Saved VDE {target_id} no longer exists.")
            _update_vde_row_for_maintenance(con, target_id, row_payload, supported_columns)
            saved_vde_row_ids[proposal_id] = target_id
        else:
            saved_vde_row_ids[proposal_id] = _insert_vde_row(con, row_payload, supported_columns)

    if normalized_strategy == "RECALCULATE_UPDATE":
        stale_fuel_row_ids = _mark_fuel_rows_stale(con, tuple(saved_vde_row_ids.values()))

    for row in _proposal_history_rows(state, bundle, saved_vde_row_ids):
        con.execute(
            f"""
            INSERT INTO {REQUEST_HISTORY_PROPOSAL_TABLE} (
                request_history_id, proposal_id, display_index, source_column,
                walk_from_kind, walk_from_proposal_id, walk_from_source_column,
                effective_metadata_json, metadata_overrides_json, domain_requests_json,
                applied_inputs_json, resolved_snapshot_json, preview_summary_json,
                issues_json, component_actions_json, abc_total_json, abc_net_json,
                vde_results_json, saved_vde_row_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_history_id,
                row.get("proposal_id"),
                row.get("display_index"),
                row.get("source_column"),
                row.get("walk_from_kind"),
                row.get("walk_from_proposal_id"),
                row.get("walk_from_source_column"),
                row.get("effective_metadata_json"),
                row.get("metadata_overrides_json"),
                row.get("domain_requests_json"),
                row.get("applied_inputs_json"),
                row.get("resolved_snapshot_json"),
                row.get("preview_summary_json"),
                row.get("issues_json"),
                row.get("component_actions_json"),
                row.get("abc_total_json"),
                row.get("abc_net_json"),
                row.get("vde_results_json"),
                row.get("saved_vde_row_id"),
            ),
        )

    save_result = {
        "status": "success",
        "maintenance_strategy": normalized_strategy,
        "source_request_history_id": int(prepared["request_history_id"]),
        "record_id": request_history_id,
        "record_key": record_key,
        "save_plan_operation_id": save_plan.get("operation_id"),
        "executed_at": _utc_now_iso(),
        "saved_proposals": [
            {"proposal_id": proposal_id, "vde_row_id": row_id, "status": "updated" if normalized_strategy == "RECALCULATE_UPDATE" else "saved"}
            for proposal_id, row_id in saved_vde_row_ids.items()
        ],
        "stale_fuel_row_ids": stale_fuel_row_ids,
        "issues": [],
    }
    con.execute(
        f"UPDATE {REQUEST_HISTORY_TABLE} SET save_status=?, save_result_json=? WHERE id=?",
        ("success", _json_dumps(save_result), request_history_id),
    )
    return {
        **save_result,
        "request_history_id": request_history_id,
        "saved_vde_row_ids": saved_vde_row_ids,
    }


def _state_with_catalog_replacement(
    state: dict,
    replacement: dict,
    *,
    direct_proposal_ids: set[str],
) -> dict:
    normalized = normalize_v22_state(state)
    entity_type = str(replacement.get("entity_type") or "").strip().upper()
    replacement_record = deepcopy(dict(replacement.get("replacement_record") or {}))
    if entity_type == "TIRE":
        old_id = str(replacement.get("old_record_id") or "")
        new_id = replacement_record.get("id")
        for proposal in list(normalized.get("proposals") or []):
            proposal_id = str(proposal.get("proposal_id") or "")
            inputs = dict(dict(proposal.get("inputs") or {}).get("tire") or {})
            if proposal_id not in direct_proposal_ids and str(inputs.get("tire_db_id") or "") != old_id:
                continue
            inputs["tire_db_id"] = new_id
            inputs["tire_code"] = replacement_record.get("tire_test_code") or replacement_record.get("tire_code")
            inputs["rrc_N_per_kN"] = _first_defined(
                replacement_record.get("rr_n_per_kn"),
                replacement_record.get("iso_rrc_n_per_kn"),
            )
            inputs["tire_snapshot"] = deepcopy(replacement_record)
            proposal.setdefault("inputs", {})["tire"] = inputs
    elif entity_type == "COMPONENT":
        domain = str(replacement.get("domain") or replacement_record.get("domain") or "").strip()
        old_component_id = str(replacement.get("old_component_id") or "")
        new_component_id = str(replacement_record.get("component_id") or replacement_record.get("component_code") or "")
        field_map = {
            "transmission": "transmission_component_db_id",
            "brake": "brake_component_db_id",
            "axle_hubs": "axle_hubs_component_db_id",
            "parasitic": "parasitic_component_db_id",
        }
        input_field = field_map.get(domain)
        if not input_field or not new_component_id:
            raise ValueError("Component replacement requires a canonical domain and component code.")
        for proposal in list(normalized.get("proposals") or []):
            proposal_id = str(proposal.get("proposal_id") or "")
            inputs = dict(dict(proposal.get("inputs") or {}).get(domain) or {})
            if proposal_id not in direct_proposal_ids and str(inputs.get(input_field) or "") != old_component_id:
                continue
            inputs[input_field] = new_component_id
            proposal.setdefault("inputs", {})[domain] = inputs
        snapshots = deepcopy(dict(normalized.get("saved_component_repository_snapshots") or {}))
        snapshots.setdefault(domain, {})[new_component_id] = deepcopy(replacement_record)
        normalized["saved_component_repository_snapshots"] = snapshots
    else:
        raise ValueError("Only Tire and Component snapshots support maintenance replacement.")
    normalized["preview"] = {"status": "stale", "fingerprint": None, "result": None}
    normalized["save"] = {"status": "pending", "result": None}
    return normalized


def _maintenance_comparisons(loaded: dict, bundle: dict) -> list[dict]:
    before_by_id = {
        str(row.get("proposal_id") or ""): dict(row)
        for row in list(loaded.get("proposal_records") or [])
    }
    comparisons = []
    for after in list(dict(bundle.get("resolution_result") or {}).get("proposal_results") or []):
        proposal_id = str(after.get("proposal_id") or "")
        before = before_by_id.get(proposal_id, {})
        before_snapshot = dict(before.get("resolved_snapshot") or {})
        after_snapshot = dict(after.get("resolved_snapshot") or {})
        comparisons.append(
            {
                "proposal_id": proposal_id,
                "walk_from": dict(after.get("walk_from") or {}).get("label"),
                "before_status": _proposal_status(before),
                "after_status": after.get("status"),
                "before_mass_kg": _first_defined(
                    before_snapshot.get("vde_calculation_mass_kg"),
                    before_snapshot.get("test_mass_kg"),
                    before_snapshot.get("mass_kg"),
                ),
                "after_mass_kg": _first_defined(
                    after_snapshot.get("vde_calculation_mass_kg"),
                    after_snapshot.get("test_mass_kg"),
                    after_snapshot.get("mass_kg"),
                ),
                "before_abc_total": deepcopy(dict(before.get("abc_total") or {})),
                "after_abc_total": deepcopy(dict(after.get("abc_total") or {})),
                "before_vde": deepcopy(dict(before.get("vde_results") or {})),
                "after_vde": deepcopy(dict(after.get("vde_results") or {})),
            }
        )
    return comparisons


def _proposal_status(proposal_record: dict) -> str | None:
    blocking = [
        str(item.get("severity") or "").strip().lower()
        for item in list(proposal_record.get("issues") or [])
    ]
    if "blocked" in blocking:
        return "Blocked"
    if "invalid" in blocking:
        return "Invalid"
    if "missing" in blocking:
        return "Missing"
    if "review" in blocking or "warning" in blocking:
        return "Review"
    return "OK"


def _first_defined(*values):
    return next((value for value in values if value is not None), None)


def _update_vde_row_for_maintenance(
    con: sqlite3.Connection,
    row_id: int,
    row_payload: dict,
    supported_columns: set[str],
) -> None:
    payload = autoresolve_test_mass(
        {
            key: value
            for key, value in dict(row_payload or {}).items()
            if key in supported_columns and key not in {"id", "created_at", "updated_at"}
        }
    )
    if "updated_at" in supported_columns:
        payload["updated_at"] = _utc_now_iso()
    if not payload:
        raise ValueError(f"No VDE values were resolved for row {row_id}.")
    columns = list(payload)
    con.execute(
        f"UPDATE vde_db SET {', '.join(f'{column}=?' for column in columns)} WHERE id=?",
        [payload[column] for column in columns] + [int(row_id)],
    )


def _mark_fuel_rows_stale(con: sqlite3.Connection, vde_ids: tuple[int, ...]) -> list[int]:
    if not vde_ids:
        return []
    placeholders = ",".join("?" for _ in vde_ids)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        f"SELECT id, record_origin FROM fuelcons_db WHERE vde_id IN ({placeholders}) ORDER BY id",
        tuple(vde_ids),
    ).fetchall()
    updated_at = _utc_now_iso()
    for row in rows:
        origin = str(row["record_origin"] or "LEGACY").strip().upper()
        status = "STALE_VDE" if origin in {"ESTIMATED", "POWERTRAIN_L0"} else "REVIEW_REQUIRED"
        con.execute(
            "UPDATE fuelcons_db SET review_status=?, updated_at=? WHERE id=?",
            (status, updated_at, int(row["id"])),
        )
    return [int(row["id"]) for row in rows]


__all__ = [
    "REQUEST_HISTORY_PROPOSAL_TABLE",
    "REQUEST_HISTORY_TABLE",
    "build_v22_save_plan",
    "load_v22_saved_request",
    "persist_v22_maintenance_recalculation",
    "prepare_v22_maintenance_recalculation",
    "save_v22_request",
    "saved_component_repositories_from_state",
]
