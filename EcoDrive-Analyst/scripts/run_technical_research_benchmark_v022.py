from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.technical_research import live_runtime, research_technical_component  # noqa: E402
from capabilities.technical_research.adapters import load_bmw_benchmark_cases  # noqa: E402
from capabilities.technical_research.contracts import (  # noqa: E402
    ConflictResolution,
    HardwareGroupMember,
    IdentityConfidence,
    ResearchLimits,
)
from capabilities.technical_research.core import (  # noqa: E402
    audit_research_request,
    consolidate_research_attempts,
    evaluate_hardware_group,
    normalize_transmission_type,
)


DEFAULT_SAMPLE = ROOT / "artifacts/components/transmission_experiment/EXPERIMENT_SAMPLE_AUDIT.csv"
DEFAULT_GROUPS = ROOT / "artifacts/components/transmission_experiment/TRANSMISSION_CANDIDATE_GROUPS.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/technical_research"
DB_PATHS = (
    ROOT / "data/db/eco_drive.db",
    ROOT / "data/db/eco_drive_qa.db",
    ROOT / "data/db/staging/eco_drive_canonical_candidate.db",
)
OPTIONAL_FIELDS = (
    "drag_coefficient_cd",
    "frontal_area_m2",
    "tire_size_front",
    "tire_size_rear",
    "tire_size_general",
    "gear_ratios",
    "final_drive_ratio",
)
MATCH_RANK = {"UNKNOWN": 0, "MISMATCH": 1, "PARTIAL": 2, "STRONG": 3, "EXACT": 4}


def _hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DB_PATHS
        if path.exists()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _field_value(candidate: Any, field: str) -> Any:
    summary = candidate.field_support.get(field)
    return summary.value if summary is not None else candidate.attributes.get(field)


def _strongest_application_match(result: Any) -> str:
    matches = [claim.application_match.value for claim in result.evidence.claims]
    return max(matches, key=lambda value: MATCH_RANK[value]) if matches else "UNKNOWN"


def _result_row(case: Any, result: Any, attempt_number: int, retry_reason: str) -> dict[str, Any]:
    request = case.request
    candidate = result.candidate
    sources = result.source_summary.get("sources", [])
    source_by_id = {str(source.get("source_id")): source for source in sources}
    evidence_source_ids = sorted({claim.source_id for claim in result.evidence.claims})
    primary_sources = {
        source_id
        for source_id in evidence_source_ids
        if source_by_id.get(source_id, {}).get("source_classification") == "PRIMARY_TECHNICAL"
    }
    hardware_conflict = any(
        conflict.field == "transmission_hardware_designation"
        and conflict.resolution == ConflictResolution.UNRESOLVED_CONFLICT
        for conflict in result.evidence.conflicts
    )
    request_audit = audit_research_request(request)
    raw_transmission = request.known_fields.get("transmission_type")
    return {
        "request_id": request.request_id,
        "attempt_number": attempt_number,
        "retry_reason": retry_reason,
        "vde_id": case.vde_id,
        "source_request_identity": case.source_identity,
        "candidate_group_id": case.candidate_group_id,
        "old_group_signature": case.old_group_signature,
        "independent_application_id": case.independent_application_id,
        "previous_identity_status": case.previous_identity_status,
        "make": request.known_fields.get("make"),
        "model": request.known_fields.get("model"),
        "model_year": request.known_fields.get("model_year"),
        "request_consistency_status": request_audit.status.value,
        "raw_transmission_type": raw_transmission,
        "normalized_transmission_type": normalize_transmission_type(raw_transmission),
        "gears": request.known_fields.get("gears"),
        "original_final_drive_ratio": request.known_fields.get("final_drive_ratio"),
        "original_nv_ratio": request.known_fields.get("nv_ratio"),
        "drive_type": request.known_fields.get("drive_type"),
        "transmission_hardware_designation": candidate.identity,
        "transmission_family": _field_value(candidate, "transmission_family"),
        "transmission_marketing_description": _field_value(
            candidate, "transmission_marketing_description"
        ),
        "transmission_supplier": _field_value(candidate, "transmission_supplier"),
        "identity_confidence": candidate.confidence.value,
        "gear_ratios": _text(_field_value(candidate, "gear_ratios")),
        "final_drive_confirmation": _text(_field_value(candidate, "final_drive_ratio")),
        "drag_coefficient_cd": _text(_field_value(candidate, "drag_coefficient_cd")),
        "frontal_area_m2": _text(_field_value(candidate, "frontal_area_m2")),
        "tire_size_front": _text(_field_value(candidate, "tire_size_front")),
        "tire_size_rear": _text(_field_value(candidate, "tire_size_rear")),
        "tire_size_general": _text(_field_value(candidate, "tire_size_general")),
        "strongest_application_match": _strongest_application_match(result),
        "source_count": len(evidence_source_ids),
        "primary_technical_source_count": len(primary_sources),
        "source_ids": ";".join(evidence_source_ids),
        "conflict_count": len(result.evidence.conflicts),
        "material_hardware_conflict": hardware_conflict,
        "search_rounds": result.provenance.get("search_rounds", 0),
        "status": result.status.value,
        "stop_reason": result.stop_reason,
        "error": "",
    }


def _error_row(case: Any, attempt_number: int, retry_reason: str, exc: Exception) -> dict[str, Any]:
    request = case.request
    request_audit = audit_research_request(request)
    return {
        "request_id": request.request_id,
        "attempt_number": attempt_number,
        "retry_reason": retry_reason,
        "vde_id": case.vde_id,
        "source_request_identity": case.source_identity,
        "candidate_group_id": case.candidate_group_id,
        "old_group_signature": case.old_group_signature,
        "independent_application_id": case.independent_application_id,
        "previous_identity_status": case.previous_identity_status,
        "make": request.known_fields.get("make"),
        "model": request.known_fields.get("model"),
        "model_year": request.known_fields.get("model_year"),
        "request_consistency_status": request_audit.status.value,
        "raw_transmission_type": request.known_fields.get("transmission_type"),
        "normalized_transmission_type": normalize_transmission_type(
            request.known_fields.get("transmission_type")
        ),
        "gears": request.known_fields.get("gears"),
        "original_final_drive_ratio": request.known_fields.get("final_drive_ratio"),
        "original_nv_ratio": request.known_fields.get("nv_ratio"),
        "drive_type": request.known_fields.get("drive_type"),
        "identity_confidence": IdentityConfidence.UNRESOLVED.value,
        "status": "ERROR",
        "stop_reason": type(exc).__name__,
        "error": str(exc)[:500],
    }


def _audit_result(
    case: Any,
    result: Any,
    attempt_number: int,
    source_rows: list[dict[str, Any]],
    conflict_rows: list[dict[str, Any]],
    enrichment_rows: list[dict[str, Any]],
) -> None:
    for source in result.source_summary.get("sources", []):
        source_rows.append(
            {
                "request_id": result.request_id,
                "vde_id": case.vde_id,
                "attempt_number": attempt_number,
                **source,
            }
        )
    for conflict in result.evidence.conflicts:
        conflict_rows.append(
            {
                "request_id": result.request_id,
                "vde_id": case.vde_id,
                "attempt_number": attempt_number,
                "field": conflict.field,
                "competing_values": _text(conflict.values),
                "sources": ";".join(conflict.supporting_sources),
                "source_tiers": ";".join(value.value for value in conflict.source_tiers),
                "application_matches": ";".join(conflict.application_differences),
                "conflict_kind": conflict.conflict_kind,
                "resolution": conflict.resolution.value,
                "resolution_reason": conflict.resolution_reason,
            }
        )
    for field in OPTIONAL_FIELDS:
        support = result.candidate.field_support.get(field)
        enrichment_rows.append(
            {
                "request_id": result.request_id,
                "vde_id": case.vde_id,
                "attempt_number": attempt_number,
                "field": field,
                "value": _text(support.value) if support else "",
                "normalized_value": _text(support.normalized_value) if support else "",
                "support_status": support.support_status.value if support else "UNKNOWN",
                "source_ids": ";".join(support.source_ids) if support else "",
                "strongest_source_tier": (
                    support.strongest_source_tier.value
                    if support and support.strongest_source_tier
                    else ""
                ),
                "application_match": support.application_match.value if support else "UNKNOWN",
                "conflict_status": support.conflict_status if support else "NONE",
            }
        )


def _group_rows(cases: tuple[Any, ...], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases_by_id = {case.vde_id: case for case in cases}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        grouped[str(row["candidate_group_id"])].append(row)
    output: list[dict[str, Any]] = []
    for group_id, rows in sorted(grouped.items()):
        members = [
            HardwareGroupMember(
                application_id=str(row["vde_id"]),
                independent_application_id=str(row["independent_application_id"]),
                hardware_identity=row.get("transmission_hardware_designation") or None,
                identity_confidence=IdentityConfidence(
                    row.get("identity_confidence", IdentityConfidence.UNRESOLVED.value)
                ),
                marketing_description=row.get("transmission_marketing_description") or None,
                source_ids=tuple(filter(None, str(row.get("source_ids", "")).split(";"))),
                material_hardware_conflict=str(row.get("material_hardware_conflict", "")).lower()
                == "true",
            )
            for row in rows
        ]
        evaluation = evaluate_hardware_group(members)
        output.append(
            {
                "old_candidate_group_id": group_id,
                "old_group_signature": cases_by_id[str(rows[0]["vde_id"])].old_group_signature,
                "n_rows": evaluation.n_rows,
                "n_independent_applications": evaluation.n_independent_applications,
                "n_unique_models": len({str(row.get("model", "")) for row in rows}),
                "n_unique_model_years": len({str(row.get("model_year", "")) for row in rows}),
                "n_unique_external_sources": len(
                    {
                        source_id
                        for row in rows
                        for source_id in str(row.get("source_ids", "")).split(";")
                        if source_id
                    }
                ),
                "researched_hardware_identities": ";".join(evaluation.hardware_identities),
                "identity_confidences": ";".join(evaluation.identity_confidences),
                "group_status": evaluation.status.value,
                "supporting_source_count": evaluation.supporting_source_count,
                "notes": evaluation.notes,
            }
        )
    return output


def run(
    sample_path: Path,
    groups_path: Path,
    output_dir: Path,
    *,
    limit: int,
    model: str,
    reasoning_effort: str,
    max_attempts: int,
) -> None:
    before = _hashes()
    cases = load_bmw_benchmark_cases(sample_path, groups_path, limit=limit)
    if len(cases) != 15:
        raise ValueError(f"v0.2.2 requires the fixed 15-case sample, received {len(cases)}")
    runtime = live_runtime(model=model, reasoning_effort=reasoning_effort)
    limits = ResearchLimits(
        max_search_rounds=3,
        max_search_queries_per_round=5,
        max_sources_fetched=12,
        max_high_quality_sources_used=5,
    )
    attempt_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    enrichment_rows: list[dict[str, Any]] = []
    request_audit_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        request_audit = audit_research_request(case.request)
        request_audit_rows.append(
            {
                "request_id": case.request.request_id,
                "vde_id": case.vde_id,
                "status": request_audit.status.value,
                "missing_fields": ";".join(request_audit.missing_fields),
                "trusted_fields": ";".join(request_audit.trusted_fields),
                "conflicting_fields": ";".join(request_audit.conflicting_fields),
                "conflicts": _text([asdict(conflict) for conflict in request_audit.conflicts]),
            }
        )
        retry_reason = ""
        for attempt_number in range(1, max_attempts + 1):
            request = replace(
                case.request,
                limits=limits,
                force_refresh=attempt_number > 1,
            )
            case_attempt = replace(case, request=request)
            print(
                f"CASE_START {index}/15 attempt={attempt_number} "
                f"vde_id={case.vde_id} model={request.known_fields.get('model')}",
                flush=True,
            )
            search_start = len(getattr(runtime.search_provider, "audit", []))
            try:
                result = research_technical_component(request, runtime=runtime)
                row = _result_row(case_attempt, result, attempt_number, retry_reason)
                attempt_rows.append(row)
                _audit_result(
                    case_attempt,
                    result,
                    attempt_number,
                    source_rows,
                    conflict_rows,
                    enrichment_rows,
                )
                print(
                    f"CASE_DONE {case.vde_id} status={result.status.value} "
                    f"confidence={result.candidate.confidence.value} "
                    f"hardware={result.candidate.identity!r}",
                    flush=True,
                )
                break
            except Exception as exc:
                attempt_rows.append(
                    _error_row(case_attempt, attempt_number, retry_reason, exc)
                )
                print(
                    f"CASE_ERROR {case.vde_id} attempt={attempt_number} "
                    f"error={type(exc).__name__}",
                    flush=True,
                )
                retry_reason = type(exc).__name__
            finally:
                for search in getattr(runtime.search_provider, "audit", [])[search_start:]:
                    search_rows.append(
                        {
                            "request_id": request.request_id,
                            "vde_id": case.vde_id,
                            "attempt_number": attempt_number,
                            **search,
                        }
                    )

    consolidated = list(consolidate_research_attempts(attempt_rows))
    order = {case.request.request_id: index for index, case in enumerate(cases)}
    final_rows = sorted(consolidated, key=lambda row: order[str(row["request_id"])])
    groups = _group_rows(cases, final_rows)
    after = _hashes()
    db_unchanged = before == after

    final_fields = list(final_rows[0])
    _write_csv(output_dir / "BMW_TRANSMISSION_RESEARCH_FINAL_V022.csv", final_rows, final_fields)
    _write_csv(output_dir / "BMW_RESEARCH_ATTEMPT_AUDIT_V022.csv", attempt_rows, final_fields)
    _write_csv(output_dir / "BMW_TRANSMISSION_GROUP_VALIDATION_V022.csv", groups, list(groups[0]))
    _write_csv(
        output_dir / "REQUEST_IDENTITY_AUDIT_V022.csv",
        request_audit_rows,
        ["request_id", "vde_id", "status", "missing_fields", "trusted_fields", "conflicting_fields", "conflicts"],
    )
    _write_csv(
        output_dir / "SOURCE_AUDIT_V022.csv",
        source_rows,
        [
            "request_id", "vde_id", "attempt_number", "source_id", "url", "title",
            "publisher", "source_classification", "publisher_type", "document_type",
            "classification_reason", "policy_decision", "policy_reason", "fetched",
            "search_round", "landing_source_id", "discovered_from_source_id",
        ],
    )
    _write_csv(
        output_dir / "CONFLICT_AUDIT_V022.csv",
        conflict_rows,
        [
            "request_id", "vde_id", "attempt_number", "field", "competing_values",
            "sources", "source_tiers", "application_matches", "conflict_kind",
            "resolution", "resolution_reason",
        ],
    )
    _write_csv(
        output_dir / "VEHICLE_ENRICHMENT_AUDIT_V022.csv",
        enrichment_rows,
        [
            "request_id", "vde_id", "attempt_number", "field", "value",
            "normalized_value", "support_status", "source_ids", "strongest_source_tier",
            "application_match", "conflict_status",
        ],
    )
    _write_csv(
        output_dir / "LIVE_SEARCH_AUDIT_V022.csv",
        search_rows,
        [
            "request_id", "vde_id", "attempt_number", "retrieved_at", "provider",
            "query", "sources_returned", "status", "error",
        ],
    )

    confidence_counts = Counter(row.get("identity_confidence", "UNRESOLVED") for row in final_rows)
    group_counts = Counter(row["group_status"] for row in groups)
    request_counts = Counter(row["status"] for row in request_audit_rows)
    final_attempt_by_request = {
        str(row["request_id"]): int(row["attempt_number"]) for row in final_rows
    }
    final_enrichment = [
        row
        for row in enrichment_rows
        if int(row["attempt_number"])
        == final_attempt_by_request.get(str(row["request_id"]))
    ]

    def field_count(field: str, support_status: str) -> int:
        return len(
            {
                str(row["request_id"])
                for row in final_enrichment
                if row["field"] == field
                and row["value"]
                and row["support_status"] == support_status
            }
        )

    field_counts = {
        field: field_count(field, "SUPPORTED")
        for field in (
            "gear_ratios",
            "final_drive_ratio",
            "drag_coefficient_cd",
            "frontal_area_m2",
        )
    }
    partial_field_counts = {
        field: field_count(field, "PARTIALLY_SUPPORTED")
        for field in field_counts
    }
    tire_fields = {"tire_size_front", "tire_size_rear", "tire_size_general"}
    field_counts["tire_specs"] = len(
        {
            str(row["request_id"])
            for row in final_enrichment
            if row["field"] in tire_fields
            and row["value"]
            and row["support_status"] == "SUPPORTED"
        }
    )
    partial_field_counts["tire_specs"] = len(
        {
            str(row["request_id"])
            for row in final_enrichment
            if row["field"] in tire_fields
            and row["value"]
            and row["support_status"] == "PARTIALLY_SUPPORTED"
        }
    )
    repeated_ready = group_counts["HARDWARE_CONFIRMED"] > 0
    useful_hardware = confidence_counts["DIRECT"] + confidence_counts["STRONG"] > 0
    retry_count = sum(int(row.get("attempt_number", 1)) > 1 for row in attempt_rows)
    recovery_count = sum(
        row.get("first_status") == "ERROR" and row.get("final_status") != "ERROR"
        for row in final_rows
    )
    fetched = [row for row in source_rows if str(row.get("fetched", "")).lower() == "true"]
    attachments = sum(bool(row.get("discovered_from_source_id")) for row in fetched)
    bmw_service = sum(
        "bmwtechinfo.bmwgroup.com" in str(row.get("url", "")).lower() for row in fetched
    )
    supplier_sources = sum(
        "zf.com" in str(row.get("url", "")).lower() for row in fetched
    )
    hardware_resolved = sum(bool(row.get("transmission_hardware_designation")) for row in final_rows)
    family_only = sum(
        bool(row.get("transmission_family")) and not row.get("transmission_hardware_designation")
        for row in final_rows
    )
    marketing_only = sum(
        bool(row.get("transmission_marketing_description"))
        and not row.get("transmission_hardware_designation")
        and not row.get("transmission_family")
        for row in final_rows
    )
    same_sample = [case.vde_id for case in cases] == [
        "4018", "4573", "8268", "1281", "5079", "10106", "8341", "153",
        "2712", "3074", "8535", "6849", "9234", "1786", "3150",
    ]
    summary = f"""# BMW Transmission Hardware Identity Benchmark v0.2.2

## A. Benchmark continuity

- Benchmark applications: {len(cases)}
- Same sample as v0.2/v0.2.1: {'YES' if same_sample else 'NO'}

## B. Request quality

- CONSISTENT: {request_counts['CONSISTENT']}
- CONFLICTING_SOURCE_FIELDS: {request_counts['CONFLICTING_SOURCE_FIELDS']}
- INCOMPLETE: {request_counts['INCOMPLETE']}

## C. Identity results

- DIRECT: {confidence_counts['DIRECT']}
- STRONG: {confidence_counts['STRONG']}
- WEAK: {confidence_counts['WEAK']}
- UNRESOLVED: {confidence_counts['UNRESOLVED']}
- Hardware designations resolved: {hardware_resolved}
- Family-only resolutions: {family_only}
- Marketing-description-only cases: {marketing_only}

## D. Group validation

{json.dumps(dict(group_counts), sort_keys=True)}

## E. Independent technical evidence

- Gear-ratio sets: {field_counts['gear_ratios']}
- Final-drive confirmations: {field_counts['final_drive_ratio']}
- Cd values: {field_counts['drag_coefficient_cd']}
- Frontal-area values: {field_counts['frontal_area_m2']}
- Tire specifications: {field_counts['tire_specs']}
- Partial application candidates retained: {json.dumps(partial_field_counts, sort_keys=True)}

## F. Retrieval

- Search attempts: {len(search_rows)}
- Completed search rounds: {sum(int(row.get('search_rounds') or 0) for row in final_rows)}
- Technical attachments fetched: {attachments}
- BMW technical/service sources fetched: {bmw_service}
- Supplier sources fetched: {supplier_sources}
- Retries: {retry_count}
- Successful retry recoveries: {recovery_count}
- Model: `{model}`
- Reasoning effort: `{reasoning_effort}`

## G. Safety

- Canonical hashes before: `{json.dumps(before, sort_keys=True)}`
- Canonical hashes after: `{json.dumps(after, sort_keys=True)}`
- Canonical hashes unchanged: {db_unchanged}

TECHNICAL_RESEARCH_CAPABILITY_VERSION = 0.2.2

HARDWARE_IDENTITY_CONTRACT_READY = YES
REQUEST_SELF_CONSISTENCY_AUDIT_READY = YES
VEHICLE_ENRICHMENT_TARGETS_READY = YES
FINAL_BENCHMARK_CONSOLIDATION_READY = YES

BMW_BENCHMARK_CONFIGURATIONS = {len(cases)}

DIRECT_IDENTITIES = {confidence_counts['DIRECT']}
STRONG_IDENTITIES = {confidence_counts['STRONG']}
WEAK_IDENTITIES = {confidence_counts['WEAK']}
UNRESOLVED_IDENTITIES = {confidence_counts['UNRESOLVED']}

HARDWARE_CONFIRMED_GROUPS = {group_counts['HARDWARE_CONFIRMED']}
HARDWARE_SPLIT_GROUPS = {group_counts['HARDWARE_SPLIT']}
PARTIALLY_RESOLVED_GROUPS = {group_counts['PARTIALLY_RESOLVED']}
DESCRIPTIVE_ONLY_GROUPS = {group_counts['DESCRIPTIVE_ONLY']}
UNRESOLVED_GROUPS = {group_counts['UNRESOLVED']}

INDEPENDENT_FDR_VALUES = {field_counts['final_drive_ratio']}
INDEPENDENT_GEAR_RATIO_SETS = {field_counts['gear_ratios']}
INDEPENDENT_CD_VALUES = {field_counts['drag_coefficient_cd']}
INDEPENDENT_FRONTAL_AREA_VALUES = {field_counts['frontal_area_m2']}
INDEPENDENT_TIRE_SPECS = {field_counts['tire_specs']}

REPEATED_HARDWARE_GROUP_READY = {'YES' if repeated_ready else 'NO'}
READY_FOR_CROSS_OEM_BENCHMARK = {'YES' if useful_hardware and db_unchanged else 'NO'}
READY_TO_RE_RUN_TRANSMISSION_SIGNAL_EXPERIMENT = {'YES' if repeated_ready else 'NO'}

CANONICAL_WRITE_DISABLED = YES
PRODUCTION_DB_CHANGED = {'NO' if db_unchanged else 'YES'}
"""
    (output_dir / "BENCHMARK_SUMMARY_V022.md").write_text(summary, encoding="utf-8")
    print(
        f"BENCHMARK_DONE output={output_dir} db_unchanged={db_unchanged} "
        f"repeated_hardware_ready={repeated_ready}",
        flush=True,
    )
    if not db_unchanged:
        raise RuntimeError("CANONICAL_DATABASE_HASH_CHANGED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EcoDrive BMW technical research benchmark v0.2.2"
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-attempts", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    run(
        args.sample,
        args.groups,
        args.output,
        limit=args.limit,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
