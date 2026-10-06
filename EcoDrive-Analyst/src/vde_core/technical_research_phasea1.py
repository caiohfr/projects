from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .component_enrichment_pass1 import execute_pass1, file_sha256, inventory


PHASEA1_VERSION = "SPRINT12_PHASEA1_V1"
PROMOTABLE_CONFIDENCE = {"HIGH", "MEDIUM"}
ARCHITECTURE_CLAIMS = {"DRIVETRAIN_ARCHITECTURE", "POWERTRAIN_ARCHITECTURE"}

RECONCILIATION_FIELDS = [
    "child_task_id", "vde_id", "vehicle_configuration_id", "make", "model", "model_year",
    "before_gap_status", "after_evidence_status", "promotion_decision",
    "before_architecture", "resolved_architecture", "after_architecture",
    "before_macro_status", "after_macro_status", "boundary_before", "boundary_after",
    "evidence_ids", "reason_codes",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _evidence_id(row: dict[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "retrieved_at"}
    return "EVA1-" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16].upper()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["evidence_id"] = _evidence_id(row)
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _split_ids(value: str) -> list[int]:
    return [int(item) for item in str(value or "").split(";") if item.strip()]


def _complete_roadload_claim(row: dict[str, Any], vde_id: int) -> bool:
    if row.get("claim_type") != "REGULATORY_TARGET_SET_ROADLOAD_STATE":
        return False
    value = row.get("normalized_value")
    if not isinstance(value, dict) or int(value.get("vde_id", -1)) != vde_id:
        return False
    target = value.get("target_abc_native")
    set_abc = value.get("set_abc_native")
    return bool(
        isinstance(target, dict)
        and isinstance(set_abc, dict)
        and value.get("test_number")
        and all(target.get(key) is not None for key in ("a_lbf", "b_lbf_per_mph", "c_lbf_per_mph2"))
        and all(set_abc.get(key) is not None for key in ("Set Coef A (lbf)", "Set Coef B (lbf/mph)", "Set Coef C (lbf/mph**2)"))
    )


def _claim_scope_ok(row: dict[str, Any], task: dict[str, str], year: int) -> bool:
    if str(row.get("child_task_id")) != task["child_task_id"]:
        return False
    if str(row.get("application_make", "")).strip().upper() != task["make"].strip().upper():
        return False
    try:
        low, high = int(row.get("model_year_min")), int(row.get("model_year_max"))
    except (TypeError, ValueError):
        return False
    return low <= year <= high


def _promotable(row: dict[str, Any], task: dict[str, str], year: int, vde_id: int) -> tuple[bool, str]:
    if not _claim_scope_ok(row, task, year):
        return False, "CLAIM_OUTSIDE_CHILD_TASK_OR_YEAR_SCOPE"
    if _complete_roadload_claim(row, vde_id):
        return True, "EXACT_VDE_TARGET_SET_TEST_IDENTITY"
    if str(row.get("confidence", "")).upper() not in PROMOTABLE_CONFIDENCE:
        return False, "CONFIDENCE_BELOW_PROMOTION_THRESHOLD"
    provenance = str(row.get("provenance", "")).upper()
    value = str(row.get("normalized_value", "")).upper()
    boundary = str(row.get("physical_boundary", "")).upper()
    if "APPROX" in provenance:
        return False, "APPROXIMATE_PROVENANCE_REVIEW_ONLY"
    if row.get("claim_type") in ARCHITECTURE_CLAIMS:
        return bool(value and "UNKNOWN" not in value), "SCOPED_ARCHITECTURE_EVIDENCE"
    if "BOUNDARY" in str(row.get("claim_type", "")).upper():
        ok = bool(boundary and "UNKNOWN" not in boundary and "UNCONFIRMED" not in value)
        return ok, "SCOPED_BOUNDARY_EVIDENCE" if ok else "BOUNDARY_REMAINS_UNKNOWN"
    return False, "CLAIM_TYPE_NOT_PROMOTED_BY_PHASEA1"


def _vde_metadata(conn: sqlite3.Connection, vde_ids: set[int]) -> dict[int, dict[str, Any]]:
    if not vde_ids:
        return {}
    placeholders = ",".join("?" for _ in vde_ids)
    sql = f"""
        SELECT v.id AS vde_id,v.vehicle_configuration_id,v.make,v.model,v.year,
               vc.architecture_properties_json
        FROM vde AS v JOIN vehicle_configuration AS vc
          ON vc.vehicle_configuration_id=v.vehicle_configuration_id
        WHERE v.id IN ({placeholders})
    """
    return {int(row["vde_id"]): dict(row) for row in conn.execute(sql, sorted(vde_ids))}


def _merge_reconciliation_metadata(
    conn: sqlite3.Connection,
    by_configuration: dict[str, list[dict[str, Any]]],
) -> int:
    updated = 0
    for configuration_id, claims in sorted(by_configuration.items()):
        architectures = sorted({str(claim["normalized_value"]) for claim in claims})
        if len(architectures) != 1:
            continue
        row = conn.execute(
            "SELECT architecture_properties_json FROM vehicle_configuration WHERE vehicle_configuration_id=?",
            (configuration_id,),
        ).fetchone()
        if row is None:
            continue
        try:
            payload = json.loads(row[0]) if row[0] else {}
        except json.JSONDecodeError:
            payload = {"preserved_raw_value": row[0]}
        payload["phasea1_reconciliation"] = {
            "version": PHASEA1_VERSION,
            "resolved_architecture": architectures[0],
            "evidence_ids": sorted({claim["evidence_id"] for claim in claims}),
            "temporary_candidate_only": True,
        }
        conn.execute(
            "UPDATE vehicle_configuration SET architecture_properties_json=? WHERE vehicle_configuration_id=?",
            (_stable_json(payload), configuration_id),
        )
        updated += 1
    conn.commit()
    return updated


def reconcile_phasea(
    *,
    source_db: Path,
    phasea_input_dir: Path,
    output_dir: Path,
    temp_db: Path,
) -> dict[str, Any]:
    source_db = Path(source_db).resolve(strict=True)
    phasea_input_dir = Path(phasea_input_dir).resolve(strict=True)
    output_dir = Path(output_dir)
    temp_db = Path(temp_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_db.parent.mkdir(parents=True, exist_ok=True)

    source_hash_before = file_sha256(source_db)
    source_conn = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True)
    source_conn.row_factory = sqlite3.Row
    source_conn.execute("PRAGMA query_only=ON")
    source_inventory = inventory(source_conn)
    if source_inventory["quick_check"] != "ok" or source_inventory["foreign_key_issues"]:
        source_conn.close()
        raise ValueError("Source candidate failed pre-reconciliation integrity")

    tasks = _read_csv(phasea_input_dir / "research_child_tasks.csv")
    evidence = _read_jsonl(phasea_input_dir / "research_evidence_staging.jsonl")
    all_ids = {vde_id for task in tasks for vde_id in _split_ids(task.get("vde_ids", ""))}
    metadata = _vde_metadata(source_conn, all_ids)
    source_conn.close()

    evidence_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        evidence_by_task[str(row.get("child_task_id"))].append(row)

    rows: list[dict[str, Any]] = []
    architecture_by_configuration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_overrides: dict[int, str] = {}
    promoted_evidence_ids: set[str] = set()
    rejected_reasons: Counter[str] = Counter()

    for task in tasks:
        task_id = task["child_task_id"]
        for vde_id in _split_ids(task.get("vde_ids", "")):
            meta = metadata.get(vde_id)
            if not meta:
                continue
            year = int(meta["year"])
            accepted: list[dict[str, Any]] = []
            reasons: list[str] = []
            for claim in evidence_by_task.get(task_id, []):
                ok, reason = _promotable(claim, task, year, vde_id)
                if ok:
                    accepted.append(claim)
                    reasons.append(reason)
                    promoted_evidence_ids.add(claim["evidence_id"])
                elif _claim_scope_ok(claim, task, year):
                    rejected_reasons[reason] += 1
            architectures = sorted({
                str(claim["normalized_value"])
                for claim in accepted
                if claim.get("claim_type") in ARCHITECTURE_CLAIMS
            })
            roadload_ok = any(_complete_roadload_claim(claim, vde_id) for claim in accepted)
            boundary_values = sorted({
                str(claim.get("physical_boundary"))
                for claim in accepted
                if "BOUNDARY" in str(claim.get("claim_type", "")).upper()
            })
            if len(architectures) == 1:
                architecture_claims = [
                    claim for claim in accepted
                    if claim.get("claim_type") in ARCHITECTURE_CLAIMS
                    and str(claim["normalized_value"]) == architectures[0]
                ]
                architecture_by_configuration[str(meta["vehicle_configuration_id"])].extend(architecture_claims)
                route_overrides[vde_id] = architectures[0]
            resolved_architecture = architectures[0] if len(architectures) == 1 else ""
            conflict = len(architectures) > 1
            gap_parts = set(str(task.get("gap_families", "")).split(";"))
            resolved_parts = []
            if resolved_architecture:
                resolved_parts.append("ARCHITECTURE_RESOLVED")
            if roadload_ok:
                resolved_parts.append("ROADLOAD_STATE_RESOLVED")
            if boundary_values:
                resolved_parts.append("BOUNDARY_RESOLVED")
            if conflict:
                decision = "UNRESOLVED_CONFLICT"
                reasons.append("CONFLICTING_ARCHITECTURE_CLAIMS")
            elif accepted:
                decision = "EVIDENCE_PROMOTED_SCOPED"
            else:
                decision = "REVIEW_ONLY_NOT_PROMOTED"
            rows.append({
                "child_task_id": task_id,
                "vde_id": vde_id,
                "vehicle_configuration_id": meta["vehicle_configuration_id"],
                "make": meta["make"],
                "model": meta["model"],
                "model_year": year,
                "before_gap_status": ";".join(sorted(gap_parts)),
                "after_evidence_status": ";".join(resolved_parts) if resolved_parts else "UNCHANGED_UNRESOLVED",
                "promotion_decision": decision,
                "before_architecture": "UNRESOLVED" if "ARCHITECTURE_UNRESOLVED" in gap_parts else "UNCHANGED",
                "resolved_architecture": resolved_architecture,
                "after_architecture": resolved_architecture or "UNRESOLVED",
                "before_macro_status": "UNRESOLVED" if "MACRO_UNRESOLVED" in gap_parts else "UNCHANGED",
                "after_macro_status": "PENDING_DETERMINISTIC_RERUN" if resolved_architecture else "UNRESOLVED",
                "boundary_before": "UNKNOWN" if any("BOUNDARY" in part for part in gap_parts) else "UNCHANGED",
                "boundary_after": ";".join(boundary_values) if boundary_values else ("UNKNOWN" if any("BOUNDARY" in part for part in gap_parts) else "UNCHANGED"),
                "evidence_ids": ";".join(sorted({claim["evidence_id"] for claim in accepted})),
                "reason_codes": ";".join(sorted(set(reasons))) if reasons else "NO_PROMOTABLE_SCOPED_EVIDENCE",
            })

    shutil.copy2(source_db, temp_db)
    temp_conn = sqlite3.connect(temp_db)
    temp_conn.row_factory = sqlite3.Row
    temp_conn.execute("PRAGMA foreign_keys=ON")
    before_temp = inventory(temp_conn)
    metadata_updates = _merge_reconciliation_metadata(temp_conn, architecture_by_configuration)
    temp_conn.close()

    affected_ids = {int(row["vde_id"]) for row in rows if row["promotion_decision"] == "EVIDENCE_PROMOTED_SCOPED"}
    estimator_dir = output_dir / "deterministic_estimator_rerun"
    estimator = execute_pass1(
        temp_db,
        estimator_dir,
        write=True,
        source_db_path=source_db,
        vde_ids=affected_ids,
        research_route_overrides=route_overrides,
    )

    outcome_by_vde: dict[int, list[dict[str, str]]] = defaultdict(list)
    for outcome in _read_csv(estimator_dir / "vde_enrichment_outcomes.csv"):
        outcome_by_vde[int(outcome["vde_id"])].append(outcome)
    for row in rows:
        outcomes = outcome_by_vde.get(int(row["vde_id"]), [])
        resolved = [item for item in outcomes if item.get("coverage_tier") in {"A", "B"}]
        if row["resolved_architecture"]:
            row["after_macro_status"] = "RESOLVED_BY_EXISTING_DETERMINISTIC_ESTIMATOR" if resolved else "UNRESOLVED_MODEL_NOT_AVAILABLE"
            if not resolved:
                row["reason_codes"] += ";NO_EXISTING_MODEL_FOR_RESEARCHED_ARCHITECTURE"

    rows.sort(key=lambda row: (int(row["vde_id"]), row["child_task_id"]))
    _write_csv(output_dir / "reconciliation_vde_audit.csv", rows, RECONCILIATION_FIELDS)
    normalized_hash = hashlib.sha256(_stable_json(rows).encode("utf-8")).hexdigest().upper()

    final_conn = sqlite3.connect(temp_db)
    final_conn.row_factory = sqlite3.Row
    after_temp = inventory(final_conn)
    final_conn.close()
    source_hash_after = file_sha256(source_db)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Source candidate changed during Phase A.1 reconciliation")
    if after_temp["objects"] != before_temp["objects"]:
        raise RuntimeError("Schema objects changed on the temporary candidate")
    if after_temp["vde_total_signature"] != before_temp["vde_total_signature"]:
        raise RuntimeError("VDE TOTAL ABC changed during research reconciliation")

    promoted_vdes = {int(row["vde_id"]) for row in rows if row["promotion_decision"] == "EVIDENCE_PROMOTED_SCOPED"}
    macro_resolved_vdes = {
        int(row["vde_id"]) for row in rows
        if row["after_macro_status"] == "RESOLVED_BY_EXISTING_DETERMINISTIC_ESTIMATOR"
    }
    summary = {
        "version": PHASEA1_VERSION,
        "state": "RECONCILIATION_VALIDATED",
        "source_db": str(source_db),
        "temp_db": str(temp_db.resolve()),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "temp_sha256_after": file_sha256(temp_db),
        "source_quick_check": source_inventory["quick_check"],
        "source_fk_issues": source_inventory["foreign_key_issues"],
        "temp_quick_check": after_temp["quick_check"],
        "temp_fk_issues": after_temp["foreign_key_issues"],
        "schema_objects_unchanged": after_temp["objects"] == before_temp["objects"],
        "vde_total_abc_unchanged": after_temp["vde_total_signature"] == before_temp["vde_total_signature"],
        "vde_rows_audited": len(rows),
        "promoted_vdes": len(promoted_vdes),
        "promoted_evidence_records": len(promoted_evidence_ids),
        "metadata_rows_updated_temp_only": metadata_updates,
        "macro_resolved_vdes": len(macro_resolved_vdes),
        "component_resolution_rows_inserted": estimator["inserted_resolution_rows"],
        "component_links_inserted": estimator["inserted_links"],
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "reconciliation_output_sha256": normalized_hash,
        "deterministic_estimator_version": estimator["estimator_version"],
        "direct_research_abc_writes": 0,
    }
    (output_dir / "reconciliation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = ["PHASEA1_VERSION", "RECONCILIATION_FIELDS", "reconcile_phasea"]
