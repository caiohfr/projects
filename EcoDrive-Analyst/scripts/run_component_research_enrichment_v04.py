"""Run the pragmatic, review-only Components Research & Estimation v0.4."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.technical_research import live_runtime, research_technical_component  # noqa: E402
from capabilities.technical_research.adapters import load_bmw_benchmark_cases  # noqa: E402
from capabilities.technical_research.contracts import (  # noqa: E402
    ApplicationMatch,
    EvidenceClaim,
    ExtractionMethod,
    ResearchLimits,
    SourceTier,
)
from capabilities.technical_research.core import normalize_transmission_type  # noqa: E402
from capabilities.technical_research.pragmatic_enrichment import (  # noqa: E402
    PragmaticStatus,
    ResearchedSelection,
    ValueProvenance,
    flat_text,
    selections_for_application,
    source_columns,
    transmission_match_identity,
)
from etl.scripts.sprint_12_transmission_grouping_signal_experiment import (  # noqa: E402
    load_canonical_source,
    prepare_population,
)


DEFAULT_SAMPLE = ROOT / "artifacts/components/transmission_experiment/EXPERIMENT_SAMPLE_AUDIT.csv"
DEFAULT_GROUPS = ROOT / "artifacts/components/transmission_experiment/TRANSMISSION_CANDIDATE_GROUPS.csv"
DEFAULT_V022 = ROOT / "artifacts/technical_research"
DEFAULT_CURATED = ROOT / "data/reference/component_research_v04_curated_evidence.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/components/component_research_v04"
DEFAULT_DEFAULTS = ROOT / "data/standards/vde_defaults_by_category_trans_elec.csv"
DEFAULT_DB = ROOT / "data/db/staging/eco_drive_canonical_candidate.db"
DB_PATHS = (
    ROOT / "data/db/eco_drive.db",
    ROOT / "data/db/eco_drive_qa.db",
    DEFAULT_DB,
)
PRIMARY_FIELDS = (
    "transmission_code", "transmission_family", "supplier", "marketing_description",
    "gears", "gear_ratios", "final_drive", "cd", "frontal_area_m2", "cda_m2",
    "tire_front", "tire_rear", "tire_general",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def db_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in DB_PATHS if path.exists()}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "nan", "None"}


def _v022_fallback_claims(directory: Path) -> dict[str, tuple[EvidenceClaim, ...]]:
    final = {row["request_id"]: row for row in _read_csv(directory / "BMW_TRANSMISSION_RESEARCH_FINAL_V022.csv")}
    enrichment = _read_csv(directory / "VEHICLE_ENRICHMENT_AUDIT_V022.csv")
    sources = _read_csv(directory / "SOURCE_AUDIT_V022.csv")
    source_by_request: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for source in sources:
        source_by_request[source["request_id"]][source["source_id"]] = source
    latest_attempt = {
        request: max((int(row.get("attempt_number") or 1) for row in enrichment if row["request_id"] == request), default=1)
        for request in final
    }
    claims: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for row in enrichment:
        request_id = row["request_id"]
        # These retained v0.2.2 rows are obvious application mismatches:
        # 10106 used an earlier i4 variant that predates the eDrive35, while
        # 153 and 9234 used G22-era 4 Series facts for 2020 F32 requests.
        if request_id in {"bmw-benchmark-10106", "bmw-benchmark-153", "bmw-benchmark-9234"}:
            continue
        if int(row.get("attempt_number") or 1) != latest_attempt.get(request_id) or not _present(row.get("value")):
            continue
        try:
            match = ApplicationMatch(row.get("application_match") or "UNKNOWN")
        except ValueError:
            match = ApplicationMatch.UNKNOWN
        if match not in {ApplicationMatch.EXACT, ApplicationMatch.STRONG, ApplicationMatch.PARTIAL}:
            continue
        for source_id in filter(None, row.get("source_ids", "").split(";")):
            source = source_by_request.get(request_id, {}).get(source_id, {})
            claims[request_id].append(
                EvidenceClaim(
                    field=row["field"], value=row["value"],
                    normalized_value=row.get("normalized_value") or row["value"],
                    source_id=f"v022:{source_id}", source_tier=SourceTier.TIER_2_STRONG_SECONDARY,
                    evidence_location="v0.2.2 retained field audit",
                    evidence_text=f"Retained v0.2.2 researched value for {row['field']}.",
                    extraction_method=ExtractionMethod.STRUCTURED,
                    extraction_confidence=0.8, application_match=match,
                    source_url=source.get("url", ""), publisher=source.get("publisher", ""),
                    document_title=source.get("title", ""),
                    source_classification=source.get("source_classification", "V022_RETAINED"),
                )
            )
    final_field_map = {
        "transmission_hardware_designation": "transmission_hardware_designation",
        "transmission_family": "transmission_family",
        "transmission_marketing_description": "transmission_marketing_description",
        "transmission_supplier": "transmission_supplier",
    }
    for request_id, row in final.items():
        if request_id in {"bmw-benchmark-10106", "bmw-benchmark-153", "bmw-benchmark-9234"}:
            continue
        try:
            match = ApplicationMatch(row.get("strongest_application_match") or "UNKNOWN")
        except ValueError:
            match = ApplicationMatch.UNKNOWN
        if match not in {ApplicationMatch.EXACT, ApplicationMatch.STRONG, ApplicationMatch.PARTIAL}:
            continue
        source_ids = tuple(filter(None, row.get("source_ids", "").split(";")))
        for output_field, input_field in final_field_map.items():
            if not _present(row.get(input_field)):
                continue
            for source_id in source_ids or (f"row-{request_id}",):
                source = source_by_request.get(request_id, {}).get(source_id, {})
                claims[request_id].append(
                    EvidenceClaim(
                        field=output_field, value=row[input_field], normalized_value=row[input_field],
                        source_id=f"v022:{source_id}", source_tier=SourceTier.TIER_2_STRONG_SECONDARY,
                        evidence_location="v0.2.2 final review row",
                        evidence_text=f"Retained v0.2.2 value for {output_field}.",
                        extraction_method=ExtractionMethod.STRUCTURED, extraction_confidence=0.8,
                        application_match=match, source_url=source.get("url", ""),
                        publisher=source.get("publisher", ""), document_title=source.get("title", ""),
                        source_classification=source.get("source_classification", "V022_RETAINED"),
                    )
                )
    return {request: tuple(values) for request, values in claims.items()}


def _curated_claims(path: Path) -> dict[int, tuple[EvidenceClaim, ...]]:
    """Load human-reviewed public evidence without changing canonical storage."""
    claims: dict[int, list[EvidenceClaim]] = defaultdict(list)
    for row in _read_csv(path):
        claims[int(row["vde_id"])].append(
            EvidenceClaim(
                field=row["field"],
                value=row["value"],
                normalized_value=row.get("normalized_value") or row["value"],
                source_id=row["source_id"],
                source_tier=SourceTier(row["source_tier"]),
                evidence_location=row.get("evidence_location", ""),
                evidence_text=row["evidence_note"],
                extraction_method=ExtractionMethod.STRUCTURED,
                extraction_confidence=float(row.get("extraction_confidence") or 0.9),
                application_match=ApplicationMatch(row["application_match"]),
                source_url=row["source_url"],
                publisher=row.get("publisher", ""),
                document_title=row.get("source_title", ""),
                source_classification=row.get("source_type", "OEM_TECHNICAL"),
            )
        )
    return {vde_id: tuple(values) for vde_id, values in claims.items()}


def _dedupe_claims(claims: Iterable[EvidenceClaim]) -> tuple[EvidenceClaim, ...]:
    seen, output = set(), []
    for claim in claims:
        key = (claim.source_url or claim.source_id, claim.field, str(claim.normalized_value), claim.application_match.value)
        if key not in seen:
            output.append(claim)
            seen.add(key)
    return tuple(output)


def _application_context(db_path: Path) -> dict[int, dict[str, Any]]:
    vdes, runs, fuelcons, _ = load_canonical_source(db_path)
    roots, _, _ = prepare_population(vdes, runs, fuelcons)
    return {
        int(row.vde_id): {
            "category": row.category,
            "electrification": row.electrification,
        }
        for row in roots.itertuples(index=False)
    }


def _transmission_rule_key(known: Mapping[str, Any]) -> str:
    try:
        if int(float(str(known.get("gears")))) == 1:
            return "SS"
    except (TypeError, ValueError):
        pass
    normalized = normalize_transmission_type(known.get("transmission_type"))
    return {"MANUAL": "MT", "CVT": "CVT", "DCT": "AMT", "TORQUE_CONVERTER_AUTOMATIC": "AT", "SINGLE_SPEED_EV": "SS"}.get(normalized, "OT")


def load_rule_estimate(known: Mapping[str, Any], defaults_path: Path) -> tuple[dict[str, Any], str]:
    rows = _read_csv(defaults_path)
    category = str(known.get("category") or "").strip().upper()
    electrification = str(known.get("electrification") or "").strip().upper()
    assumption = ""
    if electrification in {"", "UNKNOWN", "UNSPECIFIED"}:
        model = str(known.get("model") or "").upper()
        electrification = "BEV" if model.startswith("I4") else "ICE"
        assumption = f"electrification assumed {electrification} by explicit v0.4 fallback rule"
    transmission = _transmission_rule_key(known)
    normalized_type = normalize_transmission_type(known.get("transmission_type"))
    structural_type = "SINGLE_SPEED_EV" if transmission == "SS" else normalized_type
    structural_family = f"{known.get('gears', '')}-speed {structural_type}".strip("- ")
    match = next(
        (
            row for row in rows
            if row.get("category", "").strip().upper() == category
            and row.get("electrification", "").strip().upper() == electrification
            and row.get("transmission_type", "").strip().upper() == transmission
        ),
        None,
    )
    if match is None:
        return {
            "transmission_family": structural_family,
            "rule_key": f"STRUCTURED_SIGNATURE|{known.get('gears', '')}|{normalize_transmission_type(known.get('transmission_type'))}",
        }, f"{assumption}; no exact category/electrification/transmission default row; no numeric default selected".strip("; ")
    return {
        "transmission_family": structural_family,
        "cda_m2": match.get("cdA_default_m2"),
        "rrc_N_per_kN": match.get("rrc_N_per_kN"),
        "transmission_A_prior_N": match.get("trans_A_N"),
        "transmission_B_prior_Npkph": match.get("trans_B_Npkph"),
        "brake_A_prior_N": match.get("brake_A_N"),
        "brake_B_prior_Npkph": match.get("brake_B_Npkph"),
        "rule_key": f"{match['category']}|{match['electrification']}|{match['transmission_type']}",
    }, assumption


def _selection_row(application_id: str, field: str, selection: Any) -> dict[str, Any]:
    return {
        "vehicle_application": application_id,
        "field": field,
        "selected_value": flat_text(selection.selected_value),
        "provenance": selection.provenance.value,
        "researched_value": flat_text(selection.researched_value),
        "calculated_value": flat_text(selection.calculated_value),
        "rule_estimated_value": flat_text(selection.rule_estimated_value),
        "source": selection.source,
        "note": selection.note,
    }


def _legacy_audit(path: Path, defaults_path: Path, defaults_hash: str) -> None:
    text = f"""# Legacy Component Estimation Audit v0.4

This is a code/data audit. No legacy value was imported into canonical storage.

| Rule/source inspected | Classification | v0.4 use | Reason |
|---|---|---|---|
| Structured EPA gear-count + normalized transmission-type descriptor | RETAINED | Explicit `RULE_ESTIMATED` family fallback | Public observed structure; useful descriptor, never a hardware identity. |
| `{defaults_path.relative_to(ROOT)}` CdA/RRC lookup by exact category, electrification, transmission class | RETAINED_IF_EXACT_KEY | Explicit `RULE_ESTIMATED` fallback | Public deterministic prior; not applied when canonical category is only generic `Car`. |
| Same defaults table transmission/brake A/B priors | RETAINED_IF_EXACT_KEY | Preserved in provenance audit only | Scenario priors, not hardware-specific component tests. |
| Legacy notebook whole-vehicle residual allocation/clamping | DEPRECATED | Disabled | Confounded decomposition of whole-vehicle Target ABC; unsuitable as component identity/loss truth. |
| Legacy notebook CdA/RRC residual back-solving | DEPRECATED | Disabled | v0.4 forbids deriving CdA from coastdown C and avoids target-derived aero/tire values. |
| Archived pre-Sprint-12 component estimates | BLOCKED_PRIVATE_OR_DERIVED | Disabled | Persisted derived values are not independent public component evidence. |
| `data/components/*_mock.csv` | BLOCKED_SYNTHETIC | Disabled | Synthetic QA fixtures must not enter a public Components dataset. |

Defaults SHA256: `{defaults_hash}`

LEGACY_RULES_INSPECTED = 7
LEGACY_RULES_RETAINED = 3
LEGACY_RULES_DEPRECATED = 2
LEGACY_RULES_BLOCKED_PRIVATE_OR_SYNTHETIC = 2
"""
    path.write_text(text, encoding="utf-8")


def run(
    *,
    sample: Path,
    groups_path: Path,
    output: Path,
    db_path: Path,
    defaults_path: Path,
    v022_dir: Path,
    curated_path: Path,
    model: str,
    reasoning_effort: str,
    live: bool,
    limit: int = 15,
    live_retrieval_status: str = "",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    before = db_hashes()
    if limit < 1 or limit > 15:
        raise ValueError("limit must be between 1 and 15")
    cases = load_bmw_benchmark_cases(sample, groups_path, limit=15)[:limit]
    contexts = _application_context(db_path)
    fallback = _v022_fallback_claims(v022_dir)
    curated = _curated_claims(curated_path)
    runtime = live_runtime(model=model, reasoning_effort=reasoning_effort) if live else None
    enrichment_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for number, case in enumerate(cases, 1):
        context = contexts.get(int(case.vde_id), {})
        known = {**case.request.known_fields, **{k: v for k, v in context.items() if _present(v)}}
        request = replace(
            case.request,
            known_fields=known,
            limits=ResearchLimits(max_search_rounds=2, max_search_queries_per_round=5, max_sources_fetched=6, max_high_quality_sources_used=2),
            force_refresh=live,
        )
        live_claims: tuple[EvidenceClaim, ...] = ()
        status_text, stop_reason, search_rounds, error = "CURATED_OFFICIAL_AND_V022_REUSE", "NO_LIVE_RUN", 0, ""
        if runtime is not None:
            try:
                result = research_technical_component(request, runtime=runtime)
                live_claims = tuple(result.evidence.claims)
                status_text, stop_reason = result.status.value, result.stop_reason
                search_rounds = int(result.provenance.get("search_rounds") or 0)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
                status_text, stop_reason = "ERROR", type(exc).__name__
        claims = _dedupe_claims(
            (*live_claims, *curated.get(int(case.vde_id), ()), *fallback.get(case.request.request_id, ()))
        )
        rules, rule_note = load_rule_estimate(known, defaults_path)
        selections, match_status = selections_for_application(known, claims, rules)
        application_id = f"VDE-{case.vde_id}|{known.get('model_year')}|{known.get('make')}|{known.get('model')}"
        source_data = source_columns(claims)
        row: dict[str, Any] = {
            "vehicle_application": application_id,
            "vde_id": case.vde_id,
            "make": flat_text(selections["make"].selected_value),
            "model": flat_text(selections["model"].selected_value),
            "model_year": flat_text(selections["model_year"].selected_value),
            "trim_variant": flat_text(selections["trim_variant"].selected_value),
            "drive_type": flat_text(selections["drive_type"].selected_value),
            "engine_powertrain": flat_text(selections["engine_powertrain"].selected_value),
        }
        for field in PRIMARY_FIELDS:
            row[field] = flat_text(selections[field].selected_value)
            row[f"{field}_provenance"] = selections[field].provenance.value
        row["cda_method"] = (
            "DIRECT_SOURCE" if selections["cda_m2"].provenance == ValueProvenance.RESEARCHED
            else "DIRECT_PRODUCT" if selections["cda_m2"].provenance == ValueProvenance.CALCULATED
            else "LEGACY_RULE_FALLBACK" if selections["cda_m2"].provenance == ValueProvenance.RULE_ESTIMATED
            else "UNKNOWN"
        )
        row.update({
            "match_status": match_status.value,
            "confidence_note": (
                f"{match_status.value}; PARTIAL evidence is usable with visible provenance; "
                f"MISMATCH claims excluded; {rule_note or 'no rule assumption'}"
            ),
            **source_data,
            "research_runtime_status": status_text,
            "search_rounds": search_rounds,
            "stop_reason": stop_reason,
            "error": error,
            "canonical_write": "DISABLED",
        })
        enrichment_rows.append(row)
        for field, selection in selections.items():
            provenance_rows.append(_selection_row(application_id, field, selection))
        for field in ("rrc_N_per_kN", "transmission_A_prior_N", "transmission_B_prior_Npkph", "brake_A_prior_N", "brake_B_prior_Npkph"):
            provenance_rows.append({
                "vehicle_application": application_id, "field": field,
                "selected_value": flat_text(rules.get(field)),
                "provenance": ValueProvenance.RULE_ESTIMATED.value if _present(rules.get(field)) else ValueProvenance.UNKNOWN.value,
                "researched_value": "", "calculated_value": "",
                "rule_estimated_value": flat_text(rules.get(field)), "source": "",
                "note": f"Scenario prior only; never component test evidence. rule_key={rules.get('rule_key', '')}; {rule_note}",
            })
        identity, match_level = transmission_match_identity(
            row["transmission_code"] if row["transmission_code_provenance"] == ValueProvenance.RESEARCHED.value else "",
            row["transmission_family"] if row["transmission_family_provenance"] == ValueProvenance.RESEARCHED.value else "",
            row["marketing_description"] if row["marketing_description_provenance"] == ValueProvenance.RESEARCHED.value else "",
        )
        group_rows.append({
            "normalized_transmission_identity": identity,
            "match_level": match_level,
            "vehicle_application": application_id,
            "source_backed_designation": row["transmission_code"],
            "family": row["transmission_family"],
            "supplier": row["supplier"],
            "evidence_note": row["evidence_note"],
        })
        for claim in claims:
            evidence_rows.append({
                "vehicle_application": application_id, "field": claim.field,
                "value": flat_text(claim.value), "application_match": claim.application_match.value,
                "source_tier": claim.source_tier.value, "source_url": claim.source_url,
                "source_title": claim.document_title, "source_type": claim.source_classification,
                "evidence_note": re_space(claim.evidence_text)[:500],
                "selected_for_pragmatic_review": "NO" if claim.application_match == ApplicationMatch.MISMATCH else "YES",
            })
        print(f"V04_PROGRESS {number}/{len(cases)} vde={case.vde_id} status={match_status.value} claims={len(claims)}", flush=True)

    write_csv(output / "COMPONENT_RESEARCH_ENRICHMENT_V04.csv", enrichment_rows)
    write_csv(output / "TRANSMISSION_MATCH_GROUPS_V04.csv", group_rows)
    write_csv(output / "COMPONENT_VALUE_PROVENANCE_V04.csv", provenance_rows)
    write_csv(output / "COMPONENT_RESEARCH_EVIDENCE_V04.csv", evidence_rows)
    _legacy_audit(output / "LEGACY_COMPONENT_ESTIMATION_AUDIT_V04.md", defaults_path, sha256_file(defaults_path))

    statuses = Counter(row["match_status"] for row in enrichment_rows)
    repeated = {
        identity: rows
        for identity, rows in _grouped(group_rows, "normalized_transmission_identity").items()
        if identity != "UNKNOWN" and len(rows) >= 2 and rows[0]["match_level"] in {"SAME_EXACT_TRANSMISSION", "SAME_FAMILY_OR_VARIANT"}
    }
    researched_fields = {
        row["vehicle_application"]
        for row in provenance_rows
        if row["field"] in PRIMARY_FIELDS and row["provenance"] == ValueProvenance.RESEARCHED.value
    }
    rules_only = {
        row["vehicle_application"]
        for row in provenance_rows
        if row["field"] in PRIMARY_FIELDS and row["provenance"] == ValueProvenance.RULE_ESTIMATED.value
    } - researched_fields
    all_apps = {row["vehicle_application"] for row in enrichment_rows}
    completely_unknown = all_apps - researched_fields - rules_only
    count = lambda field: sum(_present(row[field]) and row[f"{field}_provenance"] in {ValueProvenance.RESEARCHED.value, ValueProvenance.CALCULATED.value} for row in enrichment_rows)
    metrics = {
        "vehicles_total": len(enrichment_rows),
        "transmission_exact_found": statuses[PragmaticStatus.FOUND_EXACT.value],
        "transmission_family_found": statuses[PragmaticStatus.FOUND_FAMILY.value],
        "transmission_not_found": statuses[PragmaticStatus.NOT_FOUND.value],
        "transmission_conflicts": statuses[PragmaticStatus.CONFLICT.value],
        "cd_found": count("cd"), "frontal_area_found": count("frontal_area_m2"),
        "cda_available": sum(_present(row["cda_m2"]) for row in enrichment_rows),
        "tire_specs_found": sum(any(_present(row[field]) and row[f"{field}_provenance"] == ValueProvenance.RESEARCHED.value for field in ("tire_front", "tire_rear", "tire_general")) for row in enrichment_rows),
        "fdr_found": sum(_present(row["final_drive"]) for row in enrichment_rows),
        "gear_ratio_sets_found": count("gear_ratios"),
        "repeated_transmission_groups": len(repeated),
        "rows_with_any_researched_data": len(researched_fields),
        "rows_with_only_rule_estimates": len(rules_only),
        "rows_completely_unknown": len(completely_unknown),
    }
    after = db_hashes()
    db_unchanged = before == after
    ready = metrics["rows_with_any_researched_data"] >= 10 and (metrics["transmission_exact_found"] + metrics["transmission_family_found"]) >= 3 and db_unchanged
    repeated_text = "\n".join(
        f"- `{identity}`: " + "; ".join(row["vehicle_application"] for row in rows)
        for identity, rows in repeated.items()
    ) or "- None found in at least two applications."
    summary = f"""# Component Research & Estimation v0.4 — Summary

## Research execution

- Live retrieval status: `{live_retrieval_status or ('COMPLETED' if live else 'NOT_REQUESTED')}`
- Completed rows use human-reviewed official sources, compatible retained v0.2.2 evidence, and deterministic v0.4 selection/provenance rules.
- No external value was promoted to canonical storage.

## A. Coverage

```json
{json.dumps(metrics, indent=2, sort_keys=True)}
```

## B. Repeated transmissions

{repeated_text}

## C. Legacy fallback

- Rules inspected: 7
- Retained or conditionally retained as explicit rule priors: 3
- Deprecated: 2
- Blocked private/derived or synthetic: 2
- The whole-vehicle residual split was not revived.

## D. Data usefulness

- Rows with researched data: {metrics['rows_with_any_researched_data']}
- Rows with only rule estimates: {metrics['rows_with_only_rule_estimates']}
- Rows completely unknown: {metrics['rows_completely_unknown']}

## E. Safety

- Canonical hashes before: `{json.dumps(before, sort_keys=True)}`
- Canonical hashes after: `{json.dumps(after, sort_keys=True)}`
- Canonical writes: 0

```text
COMPONENT_RESEARCH_VERSION = 0.4

PRAGMATIC_RESEARCH_MODE = YES
PAPER_GRADE_IDENTITY_GATE_REQUIRED = NO
RULE_BASED_FALLBACK_ENABLED = YES

VEHICLES_TOTAL = {metrics['vehicles_total']}

TRANSMISSION_EXACT_FOUND = {metrics['transmission_exact_found']}
TRANSMISSION_FAMILY_FOUND = {metrics['transmission_family_found']}
TRANSMISSION_NOT_FOUND = {metrics['transmission_not_found']}
TRANSMISSION_CONFLICTS = {metrics['transmission_conflicts']}

CD_FOUND = {metrics['cd_found']}
FRONTAL_AREA_FOUND = {metrics['frontal_area_found']}
CDA_AVAILABLE = {metrics['cda_available']}
TIRE_SPECS_FOUND = {metrics['tire_specs_found']}
FDR_FOUND = {metrics['fdr_found']}
GEAR_RATIO_SETS_FOUND = {metrics['gear_ratio_sets_found']}

REPEATED_TRANSMISSION_GROUPS = {metrics['repeated_transmission_groups']}

ROWS_WITH_ANY_RESEARCHED_DATA = {metrics['rows_with_any_researched_data']}
ROWS_WITH_ONLY_RULE_ESTIMATES = {metrics['rows_with_only_rule_estimates']}
ROWS_COMPLETELY_UNKNOWN = {metrics['rows_completely_unknown']}

CANONICAL_WRITE_DISABLED = YES
PRODUCTION_DB_CHANGED = {'NO' if db_unchanged else 'YES'}

READY_FOR_CROSS_OEM_EXPANSION = {'YES' if ready else 'NO'}
```
"""
    (output / "COMPONENT_RESEARCH_V04_SUMMARY.md").write_text(summary, encoding="utf-8")
    if not db_unchanged:
        raise RuntimeError("CANONICAL_DATABASE_HASH_CHANGED")
    return {**metrics, "ready_for_cross_oem_expansion": ready, "db_unchanged": db_unchanged}


def re_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def _grouped(rows: Iterable[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row[field])].append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--v022-dir", type=Path, default=DEFAULT_V022)
    parser.add_argument("--curated-evidence", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--defaults", type=Path, default=DEFAULT_DEFAULTS)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--live-retrieval-status", default="")
    parser.add_argument("--reuse-v022-only", action="store_true")
    args = parser.parse_args()
    metrics = run(
        sample=args.sample, groups_path=args.groups, output=args.output, db_path=args.db,
        defaults_path=args.defaults, v022_dir=args.v022_dir, curated_path=args.curated_evidence, model=args.model,
        reasoning_effort=args.reasoning_effort, live=not args.reuse_v022_only, limit=args.limit,
        live_retrieval_status=args.live_retrieval_status,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
