from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable

from .canonical_component_materialization import _group_identity


FINAL_STATES = {
    "RESOLVED_EXACT",
    "RESOLVED_STRONG",
    "PARTIAL_EVIDENCE",
    "BOUNDARY_STILL_UNKNOWN",
    "NOT_FOUND",
    "UNRESOLVED_CONFLICT",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest_id(prefix: str, value: Any, size: int = 16) -> str:
    digest = hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest().upper()[:size]
    return f"{prefix}-{digest}"


def _group_id(row: dict[str, Any]) -> str:
    key, _ = _group_identity(row)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest().upper()[:16]
    return f"RG-{str(row['gap_domain'])[:5]}-{digest}"


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _child_key(row: dict[str, Any]) -> tuple[str, ...]:
    # Model is intentionally part of the identity. The P0 source grouping can use
    # generic engine values (01/02/03); those values are not sufficient evidence
    # that unlike models share hardware.
    return tuple(
        _clean(row.get(field)).upper()
        for field in (
            "make",
            "model",
            "drive_type",
            "transmission_type",
            "gears",
            "engine_code",
            "transmission_code",
            "propulsion_architecture",
        )
    )


def _join(values: Iterable[Any]) -> str:
    return ";".join(sorted({_clean(value) for value in values if _clean(value)}))


@dataclass(frozen=True)
class PhaseAPreparation:
    clusters: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    child_tasks: list[dict[str, Any]]
    scoped_inventory: list[dict[str, Any]]


def prepare_phase_a(
    *, inventory_path: Path, p0_groups_path: Path, p0_clusters_path: Path
) -> PhaseAPreparation:
    groups = _read_csv(p0_groups_path)
    clusters = _read_csv(p0_clusters_path)
    group_to_cluster = {
        row["research_group_id"]: row["research_cluster_id"] for row in groups
    }
    scoped: list[dict[str, Any]] = []
    for row in _read_csv(inventory_path):
        group_id = _group_id(row)
        cluster_id = group_to_cluster.get(group_id)
        if cluster_id is None:
            continue
        scoped.append({**row, "research_group_id": group_id, "research_cluster_id": cluster_id})

    buckets: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in scoped:
        buckets[(row["research_cluster_id"], _child_key(row))].append(row)

    tasks: list[dict[str, Any]] = []
    for (cluster_id, key), members in sorted(buckets.items()):
        vde_ids = sorted({int(row["vde_id"]) for row in members})
        vc_ids = sorted({row["vehicle_configuration_id"] for row in members})
        years = sorted(
            value for value in (_as_int(row.get("model_year")) for row in members) if value is not None
        )
        group_ids = sorted({row["research_group_id"] for row in members})
        gap_families = sorted({row["gap_family"] for row in members})
        domains = sorted({row["gap_domain"] for row in members})
        child_identity = {"cluster": cluster_id, "technical_key": key}
        tasks.append(
            {
                "research_cluster_id": cluster_id,
                "child_task_id": _digest_id("RT", child_identity),
                "research_group_ids": ";".join(group_ids),
                "gap_families": ";".join(gap_families),
                "domains": ";".join(domains),
                "make": members[0].get("make", ""),
                "model": members[0].get("model", ""),
                "model_year_min": min(years) if years else "",
                "model_year_max": max(years) if years else "",
                "drive_type_raw": members[0].get("drive_type", ""),
                "transmission_type_raw": members[0].get("transmission_type", ""),
                "gears": members[0].get("gears", ""),
                "engine_code_raw": members[0].get("engine_code", ""),
                "transmission_code_raw": members[0].get("transmission_code", ""),
                "propulsion_architecture_raw": members[0].get("propulsion_architecture", ""),
                "vde_count": len(vde_ids),
                "vehicle_configuration_count": len(vc_ids),
                "vde_ids": ";".join(str(value) for value in vde_ids),
                "vehicle_configuration_ids": ";".join(vc_ids),
                "internal_evidence_checked": "YES",
                "internal_evidence_summary": _internal_summary(members),
                "external_search_required": "YES",
                "final_status": "NOT_FOUND",
                "status_reason": "EXTERNAL_RESEARCH_NOT_YET_EXECUTED",
            }
        )
    return PhaseAPreparation(clusters, groups, tasks, scoped)


def _internal_summary(members: list[dict[str, Any]]) -> str:
    architecture = _join(row.get("architecture_class") for row in members)
    hardware = _join(
        value
        for row in members
        for value in (row.get("engine_code"), row.get("transmission_code"))
    )
    boundary = _join(row.get("boundary_status") for row in members)
    return (
        f"architecture={architecture or 'EMPTY'};hardware_raw={hardware or 'EMPTY'};"
        f"boundary={boundary or 'EMPTY'};no independent researched claim found"
    )


def inspect_database_read_only(db_path: Path) -> dict[str, Any]:
    before = file_sha256(db_path)
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        objects = connection.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table','view') ORDER BY type, name"
        ).fetchall()
        protected_counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in ("vde", "component_db", "component_resolution", "vde_component_resolution")
        }
    after = file_sha256(db_path)
    return {
        "sha256_before": before,
        "sha256_after": after,
        "hash_unchanged": before == after,
        "quick_check": quick,
        "foreign_key_issue_count": len(foreign_keys),
        "objects": [{"name": name, "type": kind} for name, kind in objects],
        "protected_counts": protected_counts,
    }


def extract_internal_roadload_evidence(
    *, db_path: Path, preparation: PhaseAPreparation
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[int]]]:
    """Recover target/set identities already preserved in canonical source payloads.

    This is evidence recovery only. Values are copied verbatim from source payloads;
    no numerical similarity, fitting, or target/set inference is performed.
    """
    task_by_cluster_model: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {
        (task["research_cluster_id"], tuple(
            _clean(task[field]).upper() for field in (
                "make", "model", "drive_type_raw", "transmission_type_raw", "gears",
                "engine_code_raw", "transmission_code_raw", "propulsion_architecture_raw",
            )
        )): task
        for task in preparation.child_tasks
    }
    roadload_rows = [row for row in preparation.scoped_inventory if row["gap_family"] == "ROADLOAD_STATE_UNRESOLVED"]
    evidence: list[dict[str, Any]] = []
    ledger: dict[str, dict[str, Any]] = {}
    complete_by_group: dict[str, set[int]] = defaultdict(set)
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro&immutable=1"
    retrieved_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in roadload_rows:
            task = task_by_cluster_model[(row["research_cluster_id"], _child_key(row))]
            vde = connection.execute(
                "SELECT id, make, model, year, drive_type, source_record_id, source_payload_json "
                "FROM vde WHERE id = ?", (int(row["vde_id"]),)
            ).fetchone()
            if vde is None:
                continue
            try:
                source_payload = json.loads(vde["source_payload_json"] or "{}")
            except json.JSONDecodeError:
                continue
            target = source_payload.get("target_abc_native")
            runs = connection.execute(
                "SELECT run_id, source_record_id, conditions_json FROM run "
                "WHERE vde_id = ? AND record_status = 'ACTIVE' ORDER BY run_id", (vde["id"],)
            ).fetchall()
            complete_runs: list[dict[str, Any]] = []
            for run in runs:
                try:
                    conditions = json.loads(run["conditions_json"] or "{}")
                except json.JSONDecodeError:
                    continue
                if not conditions.get("test_number") or not conditions.get("set_abc_native"):
                    continue
                complete_runs.append(
                    {
                        "run_id": run["run_id"],
                        "source_record_id": run["source_record_id"],
                        "test_number": conditions.get("test_number"),
                        "adfe_test_number": conditions.get("adfe_test_number"),
                        "test_group": conditions.get("test_group"),
                        "test_vehicle_id": conditions.get("test_vehicle_id"),
                        "configuration_number": conditions.get("configuration_number"),
                        "test_category": conditions.get("test_category"),
                        "set_abc_native": conditions.get("set_abc_native"),
                    }
                )
            if not target or not complete_runs:
                continue
            complete_by_group[row["research_group_id"]].add(int(vde["id"]))
            for run in complete_runs:
                source_id = run["source_record_id"] or run["run_id"]
                normalized = {
                    "vde_id": int(vde["id"]),
                    "vde_source_record_id": vde["source_record_id"],
                    "run_id": run["run_id"],
                    "test_number": run["test_number"],
                    "adfe_test_number": run["adfe_test_number"],
                    "test_group": run["test_group"],
                    "test_vehicle_id": run["test_vehicle_id"],
                    "configuration_number": run["configuration_number"],
                    "test_category": run["test_category"],
                    "target_abc_native": target,
                    "set_abc_native": run["set_abc_native"],
                    "semantics": "TARGET_AND_SET_PRESERVED_FROM_DISTINCT_SOURCE_FIELDS",
                }
                evidence.append(
                    {
                        "research_cluster_id": row["research_cluster_id"],
                        "research_group_id": row["research_group_id"],
                        "child_task_id": task["child_task_id"],
                        "domain": row["gap_domain"],
                        "gap_family": row["gap_family"],
                        "claim_type": "REGULATORY_TARGET_SET_ROADLOAD_STATE",
                        "normalized_value": normalized,
                        "unit": "native EPA fields: lbf,lbf/mph,lbf/mph^2",
                        "physical_boundary": "WHOLE_VEHICLE_ROADLOAD_TEST_STATE",
                        "included_subsystems": [],
                        "excluded_subsystems": [],
                        "application_make": vde["make"],
                        "application_model": vde["model"],
                        "application_variant": "",
                        "model_year_min": vde["year"],
                        "model_year_max": vde["year"],
                        "drive_layout": vde["drive_type"],
                        "architecture_scope": "",
                        "hardware_code_scope": _join((row.get("engine_code"), row.get("transmission_code"))),
                        "source_title": "EPA Test Car Data source row preserved in canonical run",
                        "source_url_or_id": source_id,
                        "source_type": "REGULATORY_CERTIFICATION_INTERNAL_COPY",
                        "publication_date": "",
                        "retrieved_at": retrieved_at,
                        "evidence_locator": f"vde[{vde['id']}].source_payload_json.target_abc_native; run[{run['run_id']}].conditions_json",
                        "supporting_excerpt_short": f"Test Number {run['test_number']}; explicit Target Coef fields and Set Coef fields preserved separately.",
                        "confidence": "HIGH",
                        "provenance": "INTERNAL_EXISTING",
                        "evidence_origin": "INTERNAL_EXISTING",
                        "reason_codes": ["EXPLICIT_SOURCE_TARGET_FIELDS", "EXPLICIT_SOURCE_SET_FIELDS", "EXPLICIT_TEST_IDENTITY"],
                    }
                )
                entry = ledger.setdefault(
                    source_id,
                    {
                        "source_id": source_id,
                        "source_title": "EPA Test Car Data source row preserved in canonical run",
                        "source_url_or_id": source_id,
                        "source_type": "REGULATORY_CERTIFICATION_INTERNAL_COPY",
                        "publication_date": "",
                        "retrieved_at": retrieved_at,
                        "clusters_using_source": set(),
                        "claims_supported": 0,
                    },
                )
                entry["clusters_using_source"].add(row["research_cluster_id"])
                entry["claims_supported"] += 1
    evidence.sort(key=lambda item: (
        item["research_cluster_id"], item["research_group_id"], item["child_task_id"],
        item["normalized_value"]["vde_id"], item["normalized_value"]["run_id"],
    ))
    ledger_rows = []
    for entry in sorted(ledger.values(), key=lambda item: item["source_id"]):
        ledger_rows.append({**entry, "clusters_using_source": ";".join(sorted(entry["clusters_using_source"]))})
    return evidence, ledger_rows, complete_by_group


def apply_internal_results(
    preparation: PhaseAPreparation,
    *, evidence: list[dict[str, Any]], complete_by_group: dict[str, set[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    group_members: dict[str, set[int]] = defaultdict(set)
    for row in preparation.scoped_inventory:
        group_members[row["research_group_id"]].add(int(row["vde_id"]))
    evidence_by_task: Counter[str] = Counter(item["child_task_id"] for item in evidence)
    tasks = []
    for task in preparation.child_tasks:
        positive = evidence_by_task[task["child_task_id"]]
        tasks.append({
            **task,
            "external_search_required": "YES" if set(task["gap_families"].split(";")) - {"ROADLOAD_STATE_UNRESOLVED"} else "NO",
            "final_status": "RESOLVED_EXACT" if positive and task["gap_families"] == "ROADLOAD_STATE_UNRESOLVED" else ("PARTIAL_EVIDENCE" if positive else "NOT_FOUND"),
            "status_reason": "INTERNAL_TARGET_SET_TEST_IDENTITY_RECOVERED" if positive else "EXTERNAL_RESEARCH_NOT_YET_EXECUTED",
            "internal_evidence_record_count": positive,
        })
    task_by_id = {task["child_task_id"]: task for task in tasks}
    evidence_tasks_by_group: dict[str, set[str]] = defaultdict(set)
    for item in evidence:
        evidence_tasks_by_group[item["research_group_id"]].add(item["child_task_id"])
    groups = []
    impacts = []
    for row in initial_group_status(preparation):
        group_id = row["research_group_id"]
        complete = complete_by_group.get(group_id, set())
        expected = group_members[group_id]
        if expected and complete == expected:
            status, reason = "RESOLVED_EXACT", "ALL_SCOPED_VDES_HAVE_EXPLICIT_INTERNAL_TARGET_SET_AND_TEST_IDENTITY"
            impacts.append({
                "research_group_id": group_id, "gap_family": row["gap_family"],
                "vde_count": len(expected), "potential_outcome": "ROADLOAD_STATE_MAY_BECOME_RESOLVABLE",
                "qualification": "Potential only; deterministic reconciliation is a separate pass",
            })
        elif complete:
            status, reason = "PARTIAL_EVIDENCE", f"{len(complete)}_OF_{len(expected)}_VDES_HAVE_COMPLETE_INTERNAL_STATE"
        else:
            status, reason = "NOT_FOUND", "EXTERNAL_RESEARCH_NOT_YET_EXECUTED"
        groups.append({
            **row,
            "child_tasks_with_positive_evidence": len(evidence_tasks_by_group.get(group_id, set())),
            "final_status": status,
            "status_reason": reason,
        })
    groups_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        groups_by_cluster[group["research_cluster_id"]].append(group)
    clusters = []
    for row in initial_cluster_status(preparation):
        statuses = {item["final_status"] for item in groups_by_cluster[row["research_cluster_id"]]}
        status = "RESOLVED_EXACT" if statuses == {"RESOLVED_EXACT"} else ("PARTIAL_EVIDENCE" if "RESOLVED_EXACT" in statuses or "PARTIAL_EVIDENCE" in statuses else "NOT_FOUND")
        clusters.append({**row, "final_status": status, "status_reason": "AGGREGATED_FROM_GROUP_RESULTS"})
    return clusters, groups, tasks, impacts


def _rule_matches_task(rule: dict[str, Any], task: dict[str, Any]) -> bool:
    make = _clean(task["make"]).upper()
    model = _clean(task["model"]).upper()
    transmission = _clean(task["transmission_type_raw"]).upper()
    if make != _clean(rule.get("make")).upper():
        return False
    required = _clean(rule.get("model_contains")).upper()
    if required and required not in model:
        return False
    if any(_clean(value).upper() not in model for value in rule.get("model_contains_all", [])):
        return False
    any_values = [_clean(value).upper() for value in rule.get("model_contains_any", [])]
    if any_values and not any(value in model for value in any_values):
        return False
    excluded = _clean(rule.get("model_excludes")).upper()
    if excluded and excluded in model:
        return False
    trans_required = _clean(rule.get("transmission_contains")).upper()
    if trans_required and trans_required not in transmission:
        return False
    year_min = _as_int(task.get("model_year_min"))
    year_max = _as_int(task.get("model_year_max"))
    if year_min is None or year_max is None:
        return False
    return year_max >= int(rule["year_min"]) and year_min <= int(rule["year_max"])


def extract_curated_public_evidence(
    *, catalog_path: Path, preparation: PhaseAPreparation
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    rules = json.loads(catalog_path.read_text(encoding="utf-8"))
    group_by_id = {row["research_group_id"]: row for row in preparation.groups}
    retrieved_at = datetime.now(timezone.utc).isoformat()
    evidence: list[dict[str, Any]] = []
    ledger: dict[str, dict[str, Any]] = {}
    matched_tasks: set[str] = set()
    for task in preparation.child_tasks:
        matching = [rule for rule in rules if _rule_matches_task(rule, task)]
        if not matching:
            continue
        group_ids = [value for value in task["research_group_ids"].split(";") if value]
        external_groups = [
            group_by_id[group_id] for group_id in group_ids
            if group_by_id[group_id]["gap_family"] != "ROADLOAD_STATE_UNRESOLVED"
        ]
        if not external_groups:
            continue
        matched_tasks.add(task["child_task_id"])
        for rule in matching:
            source_id = _digest_id("SRC", rule["source_url_or_id"])
            for group in external_groups:
                record = {
                    "research_cluster_id": task["research_cluster_id"],
                    "research_group_id": group["research_group_id"],
                    "child_task_id": task["child_task_id"],
                    "domain": group["domain"],
                    "gap_family": group["gap_family"],
                    "claim_type": rule["claim_type"],
                    "normalized_value": rule["normalized_value"],
                    "unit": None,
                    "physical_boundary": rule["physical_boundary"],
                    "included_subsystems": rule["included_subsystems"],
                    "excluded_subsystems": rule["excluded_subsystems"],
                    "application_make": task["make"],
                    "application_model": task["model"],
                    "application_variant": "",
                    "model_year_min": max(int(task["model_year_min"]), int(rule["year_min"])),
                    "model_year_max": min(int(task["model_year_max"]), int(rule["year_max"])),
                    "drive_layout": task["drive_type_raw"],
                    "architecture_scope": rule["architecture_scope"],
                    "hardware_code_scope": _join((task["engine_code_raw"], task["transmission_code_raw"])),
                    "source_title": rule["source_title"],
                    "source_url_or_id": rule["source_url_or_id"],
                    "source_type": rule["source_type"],
                    "publication_date": rule["publication_date"],
                    "retrieved_at": retrieved_at,
                    "evidence_locator": rule["evidence_locator"],
                    "supporting_excerpt_short": rule["supporting_excerpt_short"],
                    "confidence": rule["confidence"],
                    "provenance": "RESEARCHED" if rule["confidence"] in {"HIGH", "MEDIUM"} else "RESEARCHED_APPROX",
                    "evidence_origin": "EXTERNAL_PUBLIC",
                    "reason_codes": [
                        "APPLICATION_RULE_EXPLICIT",
                        "SOURCE_SCOPE_INTERSECTED_WITH_CANONICAL_MODEL_YEARS",
                        *( ["PARTIAL_MODEL_YEAR_SCOPE"] if int(rule["year_min"]) > int(task["model_year_min"]) or int(rule["year_max"]) < int(task["model_year_max"]) else [] ),
                        *( ["EXACT_P0_HARDWARE_APPLICATION_UNCONFIRMED"] if rule["confidence"] == "LOW" else [] ),
                    ],
                }
                validate_evidence_record(record)
                evidence.append(record)
                entry = ledger.setdefault(source_id, {
                    "source_id": source_id, "source_title": rule["source_title"],
                    "source_url_or_id": rule["source_url_or_id"], "source_type": rule["source_type"],
                    "publication_date": rule["publication_date"], "retrieved_at": retrieved_at,
                    "clusters_using_source": set(), "claims_supported": 0,
                })
                entry["clusters_using_source"].add(task["research_cluster_id"])
                entry["claims_supported"] += 1
    evidence.sort(key=lambda item: (
        item["research_cluster_id"], item["research_group_id"], item["child_task_id"],
        item["claim_type"], item["source_url_or_id"],
    ))
    ledger_rows = [
        {**entry, "clusters_using_source": ";".join(sorted(entry["clusters_using_source"]))}
        for entry in sorted(ledger.values(), key=lambda item: item["source_id"])
    ]
    return evidence, ledger_rows, matched_tasks


def apply_combined_results(
    preparation: PhaseAPreparation,
    *, internal_evidence: list[dict[str, Any]], external_evidence: list[dict[str, Any]],
    complete_by_group: dict[str, set[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    group_members: dict[str, set[int]] = defaultdict(set)
    group_tasks: dict[str, set[str]] = defaultdict(set)
    for row in preparation.scoped_inventory:
        group_members[row["research_group_id"]].add(int(row["vde_id"]))
    for task in preparation.child_tasks:
        for group_id in task["research_group_ids"].split(";"):
            group_tasks[group_id].add(task["child_task_id"])
    evidence = internal_evidence + external_evidence
    evidence_by_task: Counter[str] = Counter(item["child_task_id"] for item in evidence)
    high_external_tasks_by_group: dict[str, set[str]] = defaultdict(set)
    fully_covered_external_tasks_by_group: dict[str, set[str]] = defaultdict(set)
    low_external_tasks_by_group: dict[str, set[str]] = defaultdict(set)
    for item in external_evidence:
        target = high_external_tasks_by_group if item["confidence"] in {"HIGH", "MEDIUM"} else low_external_tasks_by_group
        target[item["research_group_id"]].add(item["child_task_id"])
    task_by_id = {task["child_task_id"]: task for task in preparation.child_tasks}
    coverage: dict[tuple[str, str], set[int]] = defaultdict(set)
    for item in external_evidence:
        if item["confidence"] not in {"HIGH", "MEDIUM"}:
            continue
        coverage[(item["research_group_id"], item["child_task_id"])].update(
            range(int(item["model_year_min"]), int(item["model_year_max"]) + 1)
        )
    for (group_id, task_id), years in coverage.items():
        task = task_by_id[task_id]
        expected_years = set(range(int(task["model_year_min"]), int(task["model_year_max"]) + 1))
        if expected_years <= years:
            fully_covered_external_tasks_by_group[group_id].add(task_id)
    tasks = []
    for task in preparation.child_tasks:
        count = evidence_by_task[task["child_task_id"]]
        status = "PARTIAL_EVIDENCE" if count else "NOT_FOUND"
        if task["gap_families"] == "ROADLOAD_STATE_UNRESOLVED" and count:
            status = "RESOLVED_EXACT"
        if "TRANSMISSION_AXLE_BOUNDARY_UNKNOWN" in task["gap_families"] and count:
            status = "BOUNDARY_STILL_UNKNOWN"
        tasks.append({
            **task,
            "external_search_required": "NO" if count else "YES",
            "final_status": status,
            "status_reason": "CURATED_TRACEABLE_EVIDENCE_STAGED" if count else "CREDIBLE_SEARCH_DID_NOT_RESOLVE_THIS_CHILD_TASK",
            "evidence_record_count": count,
        })
    groups = []
    impacts = []
    for row in initial_group_status(preparation):
        group_id = row["research_group_id"]
        if row["gap_family"] == "ROADLOAD_STATE_UNRESOLVED":
            complete = complete_by_group.get(group_id, set())
            expected = group_members[group_id]
            status = "RESOLVED_EXACT" if expected and complete == expected else ("PARTIAL_EVIDENCE" if complete else "NOT_FOUND")
            reason = "ALL_SCOPED_VDES_HAVE_EXPLICIT_INTERNAL_TARGET_SET_AND_TEST_IDENTITY" if status == "RESOLVED_EXACT" else "INCOMPLETE_INTERNAL_TARGET_SET_COVERAGE"
            positive_tasks = {item["child_task_id"] for item in internal_evidence if item["research_group_id"] == group_id}
        elif row["gap_family"] == "TRANSMISSION_AXLE_BOUNDARY_UNKNOWN":
            positive_tasks = high_external_tasks_by_group[group_id] | low_external_tasks_by_group[group_id]
            status = "BOUNDARY_STILL_UNKNOWN"
            reason = "FAMILY_MANUAL_SHOWS_TRANSAXLE_BOUNDARY_BUT_EXACT_P0_APPLICATION_IS_UNPROVEN"
        else:
            positive_tasks = high_external_tasks_by_group[group_id]
            expected_tasks = group_tasks[group_id]
            fully_covered = fully_covered_external_tasks_by_group[group_id]
            status = "RESOLVED_STRONG" if expected_tasks and fully_covered == expected_tasks else ("PARTIAL_EVIDENCE" if positive_tasks else "NOT_FOUND")
            reason = "ALL_CHILD_TASKS_HAVE_SCOPED_OEM_EVIDENCE" if status == "RESOLVED_STRONG" else ("ONLY_A_SUBSET_OF_CHILD_TASKS_HAS_SCOPED_OEM_EVIDENCE" if status == "PARTIAL_EVIDENCE" else "CREDIBLE_SEARCH_DID_NOT_RESOLVE_GROUP")
        groups.append({**row, "child_tasks_with_positive_evidence": len(positive_tasks), "final_status": status, "status_reason": reason})
        if status in {"RESOLVED_EXACT", "RESOLVED_STRONG", "PARTIAL_EVIDENCE", "BOUNDARY_STILL_UNKNOWN"}:
            potential = {
                "ROADLOAD_STATE_UNRESOLVED": "ROADLOAD_STATE_MAY_BECOME_RESOLVABLE",
                "TRANSMISSION_AXLE_BOUNDARY_UNKNOWN": "BOUNDARY_BETTER_CHARACTERIZED_BUT_NOT_RESOLVED",
            }.get(row["gap_family"], "MACRO_ROUTING_MAY_BECOME_RESOLVABLE_FOR_EVIDENCED_CHILD_TASKS")
            impacts.append({
                "research_group_id": group_id, "gap_family": row["gap_family"],
                "vde_count": row["vde_count"], "potential_outcome": potential,
                "qualification": "Potential only; deterministic reconciliation is a separate pass",
            })
    groups_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        groups_by_cluster[group["research_cluster_id"]].append(group)
    clusters = []
    for row in initial_cluster_status(preparation):
        statuses = {item["final_status"] for item in groups_by_cluster[row["research_cluster_id"]]}
        if statuses <= {"RESOLVED_EXACT", "RESOLVED_STRONG"}:
            status = "RESOLVED_STRONG"
        elif statuses == {"BOUNDARY_STILL_UNKNOWN"}:
            status = "BOUNDARY_STILL_UNKNOWN"
        elif statuses & {"RESOLVED_EXACT", "RESOLVED_STRONG", "PARTIAL_EVIDENCE", "BOUNDARY_STILL_UNKNOWN"}:
            status = "PARTIAL_EVIDENCE"
        else:
            status = "NOT_FOUND"
        clusters.append({**row, "final_status": status, "status_reason": "AGGREGATED_FROM_GROUP_RESULTS"})
    return clusters, groups, tasks, impacts


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def initial_group_status(preparation: PhaseAPreparation) -> list[dict[str, Any]]:
    tasks_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in preparation.child_tasks:
        for group_id in task["research_group_ids"].split(";"):
            tasks_by_group[group_id].append(task)
    rows = []
    for group in preparation.groups:
        tasks = tasks_by_group[group["research_group_id"]]
        rows.append(
            {
                "research_cluster_id": group["research_cluster_id"],
                "research_group_id": group["research_group_id"],
                "gap_family": group["gap_family"],
                "domain": group["domain"],
                "child_task_count": len(tasks),
                "child_tasks_with_positive_evidence": 0,
                "final_status": "NOT_FOUND",
                "status_reason": "EXTERNAL_RESEARCH_NOT_YET_EXECUTED",
                "vde_count": group["vde_count"],
            }
        )
    return rows


def initial_cluster_status(preparation: PhaseAPreparation) -> list[dict[str, Any]]:
    tasks_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in preparation.child_tasks:
        tasks_by_cluster[task["research_cluster_id"]].append(task)
    rows = []
    for cluster in preparation.clusters:
        tasks = tasks_by_cluster[cluster["research_cluster_id"]]
        rows.append(
            {
                "research_cluster_id": cluster["research_cluster_id"],
                "child_task_count": len(tasks),
                "research_group_count": cluster["p0_group_count"],
                "gap_families": cluster["gap_families"],
                "make": cluster["make"],
                "models_enumerated": _join(task["model"] for task in tasks),
                "final_status": "NOT_FOUND",
                "status_reason": "EXTERNAL_RESEARCH_NOT_YET_EXECUTED",
            }
        )
    return rows


def validate_evidence_record(record: dict[str, Any]) -> None:
    required = (
        "research_cluster_id", "research_group_id", "child_task_id", "domain", "gap_family",
        "claim_type", "normalized_value", "unit", "physical_boundary", "included_subsystems",
        "excluded_subsystems", "application_make", "application_model", "application_variant",
        "model_year_min", "model_year_max", "drive_layout", "architecture_scope",
        "hardware_code_scope", "source_title", "source_url_or_id", "source_type",
        "publication_date", "retrieved_at", "evidence_locator", "supporting_excerpt_short",
        "confidence", "provenance", "evidence_origin", "reason_codes",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"Evidence record missing fields: {', '.join(missing)}")
    if record["provenance"] not in {
        "RESEARCHED", "RESEARCHED_APPROX", "INTERNAL_EXISTING", "UNRESOLVED", "NOT_FOUND"
    }:
        raise ValueError("Invalid provenance")
    if record["confidence"] not in {"HIGH", "MEDIUM", "LOW", "UNRESOLVED"}:
        raise ValueError("Invalid confidence")


def summarize_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["final_status"] for row in rows)
    return {state: counts.get(state, 0) for state in sorted(FINAL_STATES)}
