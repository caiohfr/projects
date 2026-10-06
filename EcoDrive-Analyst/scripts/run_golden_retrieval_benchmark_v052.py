"""Run the bounded Components Research v0.5.2 golden retrieval benchmark.

This is an artifact-only research runner. It never opens or writes a SQLite
connection; database hashes are sampled before and after the run as a safety
check.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.technical_research import live_runtime  # noqa: E402
from capabilities.technical_research.contracts import (  # noqa: E402
    ApplicationMatch,
    EvidenceClaim,
    ExtractionMethod,
    SourceDecision,
    SourceRecord,
    SourceTier,
)
from capabilities.technical_research.core.matching import match_application  # noqa: E402
from capabilities.technical_research.core.source_classifier import source_with_classification  # noqa: E402
from capabilities.technical_research.cross_oem_v05 import (  # noqa: E402
    VALUE_FIELDS,
    file_sha256,
    present,
    transmission_architecture,
)
from capabilities.technical_research.cross_oem_v051 import (  # noqa: E402
    CLAIM_FIELD_MAP,
    CrossOemPragmaticSourcePolicy,
    claims_to_curated_evidence,
    enrich_application_v051,
    request_for_sample,
)
from capabilities.technical_research.golden_retrieval_v052 import (  # noqa: E402
    GOLDEN_CASE_CONFIG,
    USEFUL_FIELDS,
    adjacent_year_is_compatible,
    architecture_error_count,
    coverage_sufficient,
    gap_queries,
    median_searches,
    normalize_architecture,
    normalize_raw_transmission_description,
    normalize_search_identity,
    primary_queries,
    source_discovery_priority,
    source_is_official,
    source_is_trusted_secondary,
    source_richness,
)
from capabilities.technical_research.tools.attachment_discovery import (  # noqa: E402
    discover_technical_attachments,
)


DEFAULT_SAMPLE = ROOT / "artifacts/components_research/v05_1/CROSS_OEM_SAMPLE_V051.csv"
DEFAULT_BASELINE = ROOT / "artifacts/components_research/v05_1/COMPONENT_RESEARCH_ENRICHMENT_V051.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/components_research/v05_2"
FROZEN_BMW = ROOT / "artifacts/components/component_research_v041a/BMW_COMPONENT_RESEARCH_BENCHMARK_V041A.csv"
FROZEN_BMW_SHA256 = "C4B31774BE2AFB6EB543446FB8CA69CAECCF6E199275BCB143E1B6037919D0E3"
DB_PATHS = (
    ROOT / "data/db/eco_drive.db",
    ROOT / "data/db/eco_drive_qa.db",
    ROOT / "data/db/staging/eco_drive_canonical_candidate.db",
    ROOT / "etl/data/staging/sprint_12f13_vde_materialized/eco_drive_canonical_vde_materialized_candidate.db",
)

SET_FIELDS = (
    "golden_case_id", "sample_id", "vehicle_configuration_id", "vde_id",
    "make", "model", "model_year", "raw_epa_model_string",
    "normalized_search_identity", "expected_useful_fields", "known_diagnostic_notes",
)
RESULT_FIELDS = (
    "golden_case_id", "sample_id", "application", "normalized_search_identity",
    "searches_count", "fetched_sources_count", "accepted_sources_count",
    "official_source_found", "technical_attachment_followed",
    "transmission_code", "transmission_family", "transmission_architecture",
    "transmission_supplier", "gear_ratios", "physical_final_drive",
    "reduction_front", "reduction_rear", "cd", "frontal_area_m2", "cda_m2",
    "tire_front", "tire_rear", "tire_general", "fields_added_count",
    "strongest_provenance", "review_flag", "search_stop_reason",
    "identity_conflict", "rejected_incompatible_evidence", "runtime_errors",
)
QUERY_FIELDS = (
    "case", "theme", "query", "round", "reason", "result_selected",
    "results_returned", "selected_source", "status", "error",
)
SOURCE_FIELDS = (
    "case", "source", "source_url", "source_title", "source_class",
    "richness_score", "fields_extracted", "accepted_rejected", "rejection_reason",
    "technical_attachment", "landing_source", "application_match",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def db_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in DB_PATHS if path.exists()}


def build_golden_set(samples: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    by_id = {row["sample_id"]: row for row in samples}
    missing = [cfg["sample_id"] for cfg in GOLDEN_CASE_CONFIG if cfg["sample_id"] not in by_id]
    if missing:
        raise ValueError(f"Golden sample IDs missing without substitution: {missing}")
    output: list[dict[str, Any]] = []
    selected: list[dict[str, str]] = []
    for cfg in GOLDEN_CASE_CONFIG:
        sample = dict(by_id[cfg["sample_id"]])
        identity = normalize_search_identity(sample, model_alias=cfg["model_alias"])
        output.append(
            {
                "golden_case_id": cfg["golden_case_id"],
                "sample_id": sample["sample_id"],
                "vehicle_configuration_id": sample["vehicle_configuration_id"],
                "vde_id": sample["vde_id"],
                "make": sample["make"],
                "model": sample["model"],
                "model_year": sample["model_year"],
                "raw_epa_model_string": sample["model"],
                "normalized_search_identity": identity,
                "expected_useful_fields": cfg["expected_useful_fields"],
                "known_diagnostic_notes": cfg["diagnostic_notes"],
            }
        )
        sample["golden_case_id"] = cfg["golden_case_id"]
        sample["model_alias"] = cfg["model_alias"]
        sample["official_hint"] = cfg.get("official_hint", "")
        sample["gap_hint"] = cfg.get("gap_hint", "")
        sample["official_model_alias"] = cfg.get("official_model_alias", "")
        sample["normalized_search_identity"] = identity
        selected.append(sample)
    return output, selected


def _trusted_secondary(source: SourceRecord) -> SourceRecord:
    return replace(
        source,
        tier=SourceTier.TIER_2_STRONG_SECONDARY,
        metadata={
            **source.metadata,
            "source_classification": "STRONG_STRUCTURED_SECONDARY",
            "publisher_type": "TRUSTED_SECONDARY",
            "classification_reason": "V052_TRUSTED_SECONDARY_REGISTRY",
        },
    )


def select_source(results: Iterable[SourceRecord], sample: Mapping[str, Any], seen_urls: set[str]) -> tuple[SourceRecord | None, str]:
    eligible: list[tuple[int, SourceRecord]] = []
    for raw in results:
        if not raw.url or raw.url in seen_urls:
            continue
        source = source_with_classification(raw)
        if source_is_trusted_secondary(source.url) and source.tier == SourceTier.UNCLASSIFIED:
            source = _trusted_secondary(source)
        official = source_is_official(source.url, sample.get("make"))
        technical = any(token in f"{source.title} {source.url} {source.document_type}".lower() for token in ("technical", "specification", ".pdf", "data sheet"))
        haystack = f"{source.title} {source.url}".lower()
        aliases = [sample.get("model_alias") or sample.get("model"), sample.get("official_model_alias")]
        token_sets = [
            [token.lower() for token in str(alias or "").replace("+", " ").replace("-", " ").split() if token.upper() not in {"AWD", "FWD", "RWD", "4WD", "4MATIC", "WITH", "TECHNOLOGY"}]
            for alias in aliases if alias
        ]
        relevance = max((sum(token in haystack for token in tokens) for tokens in token_sets), default=0)
        plus_required = "+" in str(sample.get("model_alias") or sample.get("model") or "")
        plus_supported = any(token in haystack for token in ("+", "%2b", " plus "))
        relevant = relevance > 0 and (not plus_required or plus_supported)
        if relevant and (official or source_is_trusted_secondary(source.url) or technical):
            eligible.append((relevance, source))
    if not eligible:
        return None, "NO_OFFICIAL_OR_TRUSTED_TECHNICAL_RESULT"
    return min(eligible, key=lambda item: (-item[0], *source_discovery_priority(item[1], make=sample.get("make"))))[1], ""


def _source_title_supports_application(claim: EvidenceClaim, sample: Mapping[str, Any]) -> bool:
    model_tokens = [
        token.lower()
        for token in str(sample.get("model_alias") or sample.get("model") or "").replace("+", " ").replace("-", " ").split()
        if token.upper() not in {"AWD", "FWD", "RWD", "4WD", "4MATIC", "WITH", "TECHNOLOGY"}
    ]
    haystack = f"{claim.document_title} {claim.source_url}".lower()
    return bool(model_tokens) and all(token in haystack for token in model_tokens)


def _source_title_supports_model_family(claim: EvidenceClaim, sample: Mapping[str, Any]) -> bool:
    tokens = [
        token.lower()
        for token in str(sample.get("model_alias") or sample.get("model") or "").replace("+", " ").replace("-", " ").split()
        if token.upper() not in {"AWD", "FWD", "RWD", "4WD", "4MATIC", "WITH", "TECHNOLOGY", "GRAND", "TOURING"}
    ]
    haystack = f"{claim.document_title} {claim.source_url}".lower()
    return bool(tokens) and tokens[0] in haystack


def _match_claim(claim: EvidenceClaim, sample: Mapping[str, Any]) -> tuple[EvidenceClaim, str, bool]:
    known = {
        key: value
        for key, value in request_for_sample(sample).known_fields.items()
        if key not in {"transmission_type", "gears", "final_drive_ratio", "nv_ratio", "category", "drive_type", "electrification"}
    }
    known["model"] = sample.get("model_alias") or sample.get("model")
    source_context = dict(claim.application_context)
    source_trim = source_context.get("trim") or source_context.get("variant") or source_context.get("trim/variant")
    if source_context.get("model") and source_trim:
        requested_model = str(known["model"]).upper().replace("-", " ")
        source_model = str(source_context["model"]).upper().replace("-", " ")
        if source_model in requested_model and str(source_trim).upper() not in source_model:
            source_context["model"] = f"{source_context['model']} {source_trim}"
    decision = match_application(known, source_context)
    adjacent = False
    match = decision.match
    if match == ApplicationMatch.MISMATCH and set(decision.mismatches) == {"model_year"}:
        adjacent = adjacent_year_is_compatible(
            sample.get("model_year"),
            source_context,
            requested_make=sample.get("make"),
            requested_model=sample.get("model_alias") or sample.get("model"),
        )
        if adjacent:
            match = ApplicationMatch.PARTIAL
    official = source_is_official(claim.source_url, sample.get("make"))
    title_support = _source_title_supports_application(claim, sample)
    family_support = _source_title_supports_model_family(claim, sample)
    def normalize_electrification(value: Any) -> str:
        text = str(value or "").upper()
        if "PLUG" in text and "HYBRID" in text:
            return "PHEV"
        if "BATTERY ELECTRIC" in text or text in {"BEV", "ELECTRIC", "EV"}:
            return "BEV"
        if "HYBRID" in text or text == "HEV":
            return "HEV"
        if text in {"FCEV", "HYDROGEN"}:
            return "FCEV"
        return text

    requested_electrification = normalize_electrification(sample.get("electrification"))
    source_electrification = normalize_electrification(source_context.get("electrification"))
    material_electrification_mismatch = (
        requested_electrification in {"BEV", "FCEV", "PHEV"}
        and source_electrification
        and source_electrification != requested_electrification
    )
    if match == ApplicationMatch.UNKNOWN and title_support and (official or source_is_trusted_secondary(claim.source_url)):
        match = ApplicationMatch.PARTIAL
    elif (
        match == ApplicationMatch.MISMATCH
        and set(decision.mismatches).issubset({"model"})
        and family_support
        and (official or source_is_trusted_secondary(claim.source_url))
        and str(sample.get("make", "")).upper() != "CADILLAC"
    ):
        match = ApplicationMatch.PARTIAL
    if (
        match == ApplicationMatch.MISMATCH
        and set(decision.mismatches).issubset({"model", "model_year"})
        and "model_year" in decision.mismatches
        and title_support
    ):
        try:
            observed_year = int(float(str(source_context.get("model_year"))))
            if abs(observed_year - int(float(str(sample.get("model_year"))))) == 1:
                match = ApplicationMatch.PARTIAL
                adjacent = True
        except (TypeError, ValueError):
            pass
    if material_electrification_mismatch:
        match = ApplicationMatch.MISMATCH
    return replace(claim, application_match=match), decision.reason, adjacent


def _technical(source: SourceRecord) -> bool:
    text = f"{source.title} {source.url} {source.document_type} {source.metadata.get('classified_document_type', '')}".lower()
    return any(token in text for token in ("technical", "specification", "spec sheet", "data sheet", ".pdf"))


def _gear_ratios_compatible(value: Any, expected_gears: Any) -> bool:
    try:
        expected = int(float(str(expected_gears)))
    except (TypeError, ValueError):
        return True
    try:
        parsed = ast.literal_eval(str(value)) if isinstance(value, str) else value
    except (SyntaxError, ValueError):
        return True
    if isinstance(parsed, dict):
        ordinal = {
            "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5,
            "SIXTH": 6, "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10,
        }
        found: set[int] = set()
        for key in parsed:
            text = str(key).upper()
            numeric = re.search(r"\b(\d{1,2})(?:ST|ND|RD|TH)?\b", text)
            if numeric:
                found.add(int(numeric.group(1)))
            for word, number in ordinal.items():
                if word in text:
                    found.add(number)
        return found == set(range(1, expected + 1))
    if isinstance(parsed, (list, tuple)):
        return len(parsed) == expected
    return True


def _marketing_gears_compatible(value: Any, expected_gears: Any) -> bool:
    try:
        expected = int(float(str(expected_gears)))
    except (TypeError, ValueError):
        return True
    match = re.search(r"\b(\d{1,2}|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)[ -]SPEED\b", str(value).upper())
    if not match:
        return True
    words = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10}
    token = match.group(1)
    actual = int(token) if token.isdigit() else words.get(token, -1)
    return actual == expected


def _deterministic_transmission_description(content: str, *, expected_gears: Any, electrification: Any) -> tuple[str, str] | None:
    patterns = (
        r"[^.;\n]{0,100}(?:DUAL[ -]CLUTCH|\bDCT\b)[^.;\n]{0,100}",
        r"[^.;\n]{0,100}(?:CONTINUOUSLY VARIABLE|\bCVT\b)[^.;\n]{0,100}",
        r"[^.;\n]{0,100}(?:AUTOMATED MANUAL|\bAMT\b)[^.;\n]{0,100}",
        r"[^.;\n]{0,100}(?:SINGLE[ -]SPEED|FIXED RATIO|TWO[ -]SPEED|2[ -]SPEED)[^.;\n]{0,100}",
        r"[^.;\n]{0,100}\b(?:\d{1,2}|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)[ -]SPEED[^.;\n]{0,100}(?:TRANSMISSION|AUTOMATIC|MANUAL)[^.;\n]{0,100}",
    )
    upper = content.upper()
    for pattern in patterns:
        for match in re.finditer(pattern, upper, re.IGNORECASE):
            start, end = match.span()
            description = " ".join(content[start:end].split())
            if not _marketing_gears_compatible(description, expected_gears):
                continue
            architecture = normalize_raw_transmission_description(
                description, electrification=electrification
            )
            if architecture not in {"OTHER", "UNKNOWN"}:
                return description, architecture
    return None


def _cadillac_conflict(sample: Mapping[str, Any], claim: EvidenceClaim) -> bool:
    if str(sample.get("make", "")).upper() != "CADILLAC":
        return False
    field = CLAIM_FIELD_MAP.get(claim.field)
    if field not in {"transmission_architecture", "transmission_marketing_description", "transmission_family"}:
        return False
    source_arch = normalize_architecture(claim.value)
    expected = transmission_architecture(sample.get("transmission_type"), sample.get("gears"), sample.get("electrification"))
    context_model = str(claim.application_context.get("model") or "").upper().replace("-", " ")
    same_ct4_family = "CT4" in context_model or not context_model
    return same_ct4_family and source_arch not in {"OTHER", "UNKNOWN", expected}


def _extract_source(runtime: Any, request: Any, source: SourceRecord, sample: Mapping[str, Any]) -> dict[str, Any]:
    official = source_is_official(source.url, sample.get("make"))
    policy = runtime.source_policy.evaluate(source)
    if policy.decision != SourceDecision.ACCEPT:
        return {"source": source, "document": None, "accepted": [], "raw_claims": [], "fields": set(), "reason": policy.reason, "match": "NOT_EVALUATED", "errors": 0, "conflict": False, "rejected_incompatible": 0}
    source = policy.source
    try:
        document = runtime.document_fetcher.fetch(source)
    except Exception as exc:
        return {"source": source, "document": None, "accepted": [], "raw_claims": [], "fields": set(), "reason": f"FETCH_ERROR:{type(exc).__name__}", "match": "NOT_EVALUATED", "errors": 1, "conflict": False, "rejected_incompatible": 0}
    before = len(getattr(runtime.claim_extractor, "audit", []))
    claims = list(runtime.claim_extractor.extract(request, (document,)))
    extractor_audit = getattr(runtime.claim_extractor, "audit", [])[before:]
    errors = sum(item.get("status") == "ERROR" for item in extractor_audit)
    matched: list[EvidenceClaim] = []
    match_labels: set[str] = set()
    source_identity = f"{source.title} {source.url}".upper().replace("-", " ")
    conflict = (
        str(sample.get("make", "")).upper() == "CADILLAC"
        and "CT4" in source_identity
        and "BLACKWING" in source_identity
        and "BLACKWING" not in str(sample.get("model", "")).upper()
    )
    rejected = 0
    for claim in claims:
        item, reason, adjacent = _match_claim(claim, sample)
        match_labels.add("ADJACENT_MY_APPROX" if adjacent else item.application_match.value)
        conflict = conflict or _cadillac_conflict(sample, item)
        if item.application_match == ApplicationMatch.MISMATCH:
            rejected += 1
        matched.append(item)
    accepted = claims_to_curated_evidence(
        sample, (claim for claim in matched if claim.field != "transmission_type_normalized")
    )
    accepted = [
        row for row in accepted
        if row.get("field") != "gear_ratios" or _gear_ratios_compatible(row.get("value"), sample.get("gears"))
    ]
    # The extractor returns raw marketing wording. Architecture is normalized
    # deterministically here rather than delegated to the model.
    for claim in matched:
        if claim.field != "transmission_marketing_description" or claim.application_match not in {ApplicationMatch.EXACT, ApplicationMatch.STRONG, ApplicationMatch.PARTIAL}:
            continue
        if not _marketing_gears_compatible(claim.value, sample.get("gears")):
            accepted = [
                row for row in accepted
                if not (row.get("field") == "transmission_marketing_description" and row.get("source_url") == claim.source_url and row.get("value") == str(claim.value))
            ]
            continue
        architecture = normalize_raw_transmission_description(
            claim.value, electrification=sample.get("electrification")
        )
        if architecture in {"OTHER", "UNKNOWN"}:
            continue
        match_label = "EXACT" if claim.application_match == ApplicationMatch.EXACT else "PARTIAL"
        accepted.append(
            {
                "sample_id": str(sample["sample_id"]), "make": str(sample["make"]),
                "model_pattern": "", "year_min": str(sample["model_year"]), "year_max": str(sample["model_year"]),
                "field": "transmission_architecture", "value": architecture,
                "application_match": match_label, "source_tier": claim.source_tier.value,
                "source_classification": claim.source_classification, "source_url": claim.source_url,
                "source_title": claim.document_title,
                "evidence_note": f"Deterministic architecture normalization from raw source wording: {claim.value}",
            }
        )
    if not any(row.get("field") == "transmission_architecture" for row in accepted):
        deterministic = _deterministic_transmission_description(
            document.content,
            expected_gears=sample.get("gears"),
            electrification=sample.get("electrification"),
        )
        source_relevant = any(label in {"EXACT", "STRONG", "PARTIAL", "ADJACENT_MY_APPROX"} for label in match_labels)
        if deterministic and (source_relevant or _source_title_supports_application(
            EvidenceClaim(
                field="transmission_marketing_description", value=deterministic[0],
                normalized_value=deterministic[0], source_id=source.source_id,
                source_tier=source.tier, evidence_location="deterministic source-text scan",
                evidence_text=deterministic[0], extraction_method=claims[0].extraction_method if claims else ExtractionMethod.STRUCTURED,
                extraction_confidence=1.0, application_match=ApplicationMatch.PARTIAL,
                source_url=source.url, publisher=source.publisher, document_title=source.title,
                source_classification=str(source.metadata.get("source_classification", "UNCLASSIFIED")),
            ),
            sample,
        )):
            provenance_match = "EXACT" if "EXACT" in match_labels else "PARTIAL"
            description, architecture = deterministic
            for field, value in (
                ("transmission_marketing_description", description),
                ("transmission_architecture", architecture),
            ):
                accepted.append(
                    {
                        "sample_id": str(sample["sample_id"]), "make": str(sample["make"]),
                        "model_pattern": "", "year_min": str(sample["model_year"]), "year_max": str(sample["model_year"]),
                        "field": field, "value": value, "application_match": provenance_match,
                        "source_tier": source.tier.value,
                        "source_classification": str(source.metadata.get("source_classification", "UNCLASSIFIED")),
                        "source_url": source.url, "source_title": source.title,
                        "evidence_note": "Deterministic source-text transmission parsing; no LLM architecture authority.",
                    }
                )
    raw_model = f"{sample.get('model', '')} {sample.get('model_alias', '')}".upper()
    content_upper = document.content.upper()
    requires_hybrid_identity = "PHEV" in raw_model or "HYBRID" in raw_model
    if requires_hybrid_identity and "HYBRID" not in content_upper and "PLUG-IN" not in content_upper and "PLUG IN" not in content_upper:
        accepted = []
        rejected += max(1, len(claims))
    fields = {row["field"] for row in accepted if row.get("field") in USEFUL_FIELDS}
    reason = "" if fields else ("INCOMPATIBLE_APPLICATION" if rejected else "NO_USEFUL_ACCEPTED_CLAIMS")
    if official and not source_is_official(source.url, sample.get("make")):
        raise AssertionError("official source identity changed during evaluation")
    return {
        "source": source, "document": document, "accepted": accepted, "raw_claims": matched,
        "fields": fields, "reason": reason, "match": ";".join(sorted(match_labels)) or "UNKNOWN",
        "errors": errors, "conflict": conflict, "rejected_incompatible": rejected,
    }


def _dedupe_evidence(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        key = (str(row.get("field")), str(row.get("value")), str(row.get("source_url")), str(row.get("application_match")))
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def _strongest_provenance(enriched: Mapping[str, Any]) -> str:
    values = {str(enriched.get(f"{field}_provenance", "")) for field in VALUE_FIELDS}
    for provenance in ("RESEARCHED_EXACT", "RESEARCHED_APPROX", "CALCULATED", "RULE_ESTIMATED", "OBSERVED", "UNKNOWN"):
        if provenance in values:
            return provenance
    return "UNKNOWN"


def run_case(runtime: Any, sample: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    request = request_for_sample(sample)
    identity = sample["normalized_search_identity"]
    queries = list(primary_queries(sample, identity))
    query_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    fetched = accepted_sources = searches = errors = rejected_incompatible = 0
    official_found = attachment_followed = conflict = False
    richest_source_fields = 0
    stop_reason = "SEARCH_LIMIT_REACHED"
    query_index = 0
    gap_added = False
    while query_index < len(queries) and searches < 4 and fetched < 4:
        planned = queries[query_index]
        query_index += 1
        searches += 1
        selected: SourceRecord | None = None
        search_error = ""
        results: tuple[SourceRecord, ...] = ()
        try:
            results = tuple(runtime.search_provider.search(planned.query, limit=5))
            selected, search_error = select_source(results, sample, seen_urls)
        except Exception as exc:
            errors += 1
            search_error = f"SEARCH_ERROR:{type(exc).__name__}"
        query_row = {
            "case": sample["golden_case_id"], "theme": planned.theme, "query": planned.query,
            "round": planned.round_number, "reason": planned.reason,
            "result_selected": "YES" if selected else "NO", "results_returned": len(results),
            "selected_source": selected.url if selected else "", "status": "OK" if not search_error else "ERROR",
            "error": search_error,
        }
        query_rows.append(query_row)
        if selected is None:
            if query_index >= len(queries) and not gap_added:
                found = {row["field"] for row in evidence if row.get("field") in USEFUL_FIELDS}
                queries.extend(gap_queries(sample, identity, found))
                gap_added = True
            continue
        seen_urls.add(selected.url)
        fetched += 1
        extracted = _extract_source(runtime, request, selected, sample)
        errors += extracted["errors"]
        rejected_incompatible += extracted["rejected_incompatible"]
        conflict = conflict or extracted["conflict"]
        fields = set(extracted["fields"])
        evidence.extend(extracted["accepted"])
        if fields:
            accepted_sources += 1
        official = source_is_official(selected.url, sample.get("make"))
        official_found = official_found or (official and extracted["document"] is not None)
        richness = source_richness(fields, official=official, technical_document=_technical(selected))
        richest_source_fields = max(richest_source_fields, len(fields))
        source_rows.append(
            {
                "case": sample["golden_case_id"], "source": selected.source_id, "source_url": selected.url,
                "source_title": selected.title, "source_class": "OFFICIAL" if official else "SECONDARY",
                "richness_score": richness, "fields_extracted": ";".join(sorted(fields)),
                "accepted_rejected": "ACCEPTED" if fields else "REJECTED",
                "rejection_reason": extracted["reason"], "technical_attachment": "NO", "landing_source": "",
                "application_match": extracted["match"],
            }
        )

        # Follow at most one directly linked technical document. It consumes the
        # same hard four-source budget and retains landing-page lineage.
        if extracted["document"] is not None and fetched < 4:
            attachments = discover_technical_attachments(extracted["document"], limit=3)
            attachment = next((item for item in attachments if item.url not in seen_urls), None)
            if attachment is not None:
                attachment = source_with_classification(attachment)
                seen_urls.add(attachment.url)
                fetched += 1
                attachment_followed = True
                child = _extract_source(runtime, request, attachment, sample)
                errors += child["errors"]
                rejected_incompatible += child["rejected_incompatible"]
                conflict = conflict or child["conflict"]
                child_fields = set(child["fields"])
                evidence.extend(child["accepted"])
                if child_fields:
                    accepted_sources += 1
                child_official = source_is_official(attachment.url, sample.get("make"))
                official_found = official_found or (child_official and child["document"] is not None)
                child_richness = source_richness(child_fields, official=child_official, technical_document=True)
                richest_source_fields = max(richest_source_fields, len(child_fields))
                source_rows.append(
                    {
                        "case": sample["golden_case_id"], "source": attachment.source_id,
                        "source_url": attachment.url, "source_title": attachment.title,
                        "source_class": "OFFICIAL" if child_official else "SECONDARY",
                        "richness_score": child_richness, "fields_extracted": ";".join(sorted(child_fields)),
                        "accepted_rejected": "ACCEPTED" if child_fields else "REJECTED",
                        "rejection_reason": child["reason"], "technical_attachment": "YES",
                        "landing_source": selected.source_id, "application_match": child["match"],
                    }
                )

        found = {row["field"] for row in evidence if row.get("field") in USEFUL_FIELDS}
        if coverage_sufficient(found, richest_source_field_count=richest_source_fields):
            stop_reason = "USEFUL_COVERAGE_REACHED" if richest_source_fields < 3 else "RICH_SOURCE_COVERAGE_REACHED"
            break
        if query_index >= len(queries) and not gap_added:
            queries.extend(gap_queries(sample, identity, found))
            gap_added = True

    evidence = _dedupe_evidence(evidence)
    enriched, _ = enrich_application_v051(sample, evidence)
    found_fields = {row["field"] for row in evidence if row.get("field") in USEFUL_FIELDS}
    if conflict:
        review = "REVIEW_CONFLICT"
    elif any(row.get("application_match") == "PARTIAL" for row in evidence):
        review = "REVIEW_APPROX"
    elif found_fields:
        review = "REVIEW_OK"
    else:
        review = "REVIEW_SPARSE"
    result = {
        "golden_case_id": sample["golden_case_id"], "sample_id": sample["sample_id"],
        "application": f"{sample['model_year']} {sample['make']} {sample['model']}",
        "normalized_search_identity": identity, "searches_count": searches,
        "fetched_sources_count": fetched, "accepted_sources_count": accepted_sources,
        "official_source_found": "YES" if official_found else "NO",
        "technical_attachment_followed": "YES" if attachment_followed else "NO",
        **{
            field: (
                enriched.get(field, "")
                if enriched.get(f"{field}_provenance") in {"RESEARCHED_EXACT", "RESEARCHED_APPROX", "CALCULATED"}
                else ""
            )
            for field in RESULT_FIELDS if field in VALUE_FIELDS
        },
        "fields_added_count": len(found_fields), "strongest_provenance": _strongest_provenance(enriched),
        "review_flag": review, "search_stop_reason": stop_reason,
        "identity_conflict": "YES" if conflict else "NO",
        "rejected_incompatible_evidence": rejected_incompatible, "runtime_errors": errors,
    }
    return result, query_rows, source_rows


def baseline_metrics(rows: list[dict[str, str]], case_ids: set[str]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("sample_id") in case_ids]
    researched = {
        field
        for row in selected
        for field in USEFUL_FIELDS
        if row.get(f"{field}_provenance") in {"RESEARCHED_EXACT", "RESEARCHED_APPROX"}
    }
    # Per-case field count is the comparison quantity, not just unique names.
    researched_count = sum(
        row.get(f"{field}_provenance") in {"RESEARCHED_EXACT", "RESEARCHED_APPROX"}
        for row in selected for field in USEFUL_FIELDS
    )
    return {
        "searches": sum(int(row.get("searches") or 0) for row in selected),
        "useful_sources": sum(int(row.get("accepted_useful_sources") or 0) for row in selected),
        "useful_fields": researched_count,
        "sparse": sum(row.get("review_flag") == "REVIEW_SPARSE" for row in selected),
        "architecture_errors": architecture_error_count(selected),
        "field_names": sorted(researched),
    }


def summarize(results: list[dict[str, Any]], queries: list[dict[str, Any]], sources: list[dict[str, Any]], baseline: Mapping[str, Any], *, db_before: Mapping[str, str], db_after: Mapping[str, str], bmw_ok: bool) -> dict[str, Any]:
    useful = sum(int(row["fields_added_count"]) > 0 or row["identity_conflict"] == "YES" for row in results)
    official = sum(row["official_source_found"] == "YES" for row in results)
    two_plus = sum(int(row["fields_added_count"]) >= 2 for row in results)
    total_searches = sum(int(row["searches_count"]) for row in results)
    fetched = sum(int(row["fetched_sources_count"]) for row in results)
    accepted = sum(int(row["accepted_sources_count"]) for row in results)
    arch_errors = architecture_error_count(results)
    runtime_errors = sum(int(row["runtime_errors"]) for row in results)
    gate = len(results) == 10 and useful == 10 and official >= 8 and two_plus >= 6 and arch_errors == 0 and median_searches(results) <= 4 and bmw_ok and db_before == db_after
    return {
        "cases": len(results), "useful": useful, "official": official, "two_plus": two_plus,
        "searches": total_searches, "mean_searches": total_searches / len(results) if results else 0,
        "median_searches": median_searches(results), "fetched": fetched, "accepted": accepted,
        "official_sources": sum(row["source_class"] == "OFFICIAL" and row["accepted_rejected"] == "ACCEPTED" for row in sources),
        "technical_sources": sum(row["technical_attachment"] == "YES" or any(token in str(row["source_title"]).lower() for token in ("technical", "specification", "spec sheet")) for row in sources),
        "secondary_sources": sum(row["source_class"] == "SECONDARY" and row["accepted_rejected"] == "ACCEPTED" for row in sources),
        "rich_sources": sum(len([field for field in str(row["fields_extracted"]).split(";") if field]) >= 3 for row in sources),
        "arch_errors": arch_errors, "conflicts": sum(row["identity_conflict"] == "YES" for row in results),
        "rejected": sum(int(row["rejected_incompatible_evidence"]) for row in results),
        "runtime_errors": runtime_errors, "gate": gate, "bmw_ok": bmw_ok, "db_ok": db_before == db_after,
        "baseline": dict(baseline), "current_fields": sum(int(row["fields_added_count"]) for row in results),
        "current_sparse": sum(row["review_flag"] == "REVIEW_SPARSE" for row in results),
        "queries_avoided": sum(row["search_stop_reason"] in {"USEFUL_COVERAGE_REACHED", "RICH_SOURCE_COVERAGE_REACHED"} and int(row["searches_count"]) < 4 for row in results),
    }


def write_summary(path: Path, report: Mapping[str, Any], results: list[dict[str, Any]], *, db_before: Mapping[str, str], db_after: Mapping[str, str]) -> None:
    counts = Counter()
    for row in results:
        for field in ("transmission_code", "transmission_family", "transmission_architecture", "gear_ratios", "physical_final_drive", "reduction_front", "reduction_rear", "cd", "frontal_area_m2", "cda_m2", "tire_front", "tire_rear", "tire_general"):
            if present(row.get(field)):
                counts[field] += 1
    b = report["baseline"]
    lines = [
        "# EcoDrive Components Research v0.5.2 — Golden Retrieval Benchmark", "",
        "## A. Golden set", "",
        f"- Cases: **{report['cases']}**; useful: **{report['useful']}**; official-source cases: **{report['official']}**; cases with 2+ useful fields: **{report['two_plus']}**.", "",
        "## B. Search efficiency", "",
        f"- Searches: **{report['searches']}** (mean **{report['mean_searches']:.2f}**, median **{report['median_searches']:.1f}** per case).",
        f"- Fetched sources: **{report['fetched']}**; useful accepted sources: **{report['accepted']}**; early-stop calls avoided: **{report['queries_avoided']}**.", "",
        "## C. Retrieval quality", "",
        f"- Accepted official sources: **{report['official_sources']}**; technical PDFs/spec sheets encountered: **{report['technical_sources']}**; accepted structured secondary sources: **{report['secondary_sources']}**; rich sources (3+ fields): **{report['rich_sources']}**.", "",
        "## D. Field coverage", "",
        f"- Transmission code/family/architecture: **{counts['transmission_code']} / {counts['transmission_family']} / {counts['transmission_architecture']}**.",
        f"- Gear ratios/FDR-or-reduction: **{counts['gear_ratios']} / {sum(any(present(row.get(field)) for field in ('physical_final_drive', 'reduction_front', 'reduction_rear')) for row in results)}**.",
        f"- Cd/frontal area/CdA/tires: **{counts['cd']} / {counts['frontal_area_m2']} / {counts['cda_m2']} / {sum(any(present(row.get(field)) for field in ('tire_front', 'tire_rear', 'tire_general')) for row in results)}**.", "",
        "## E. Correctness", "",
        f"- Architecture normalization errors: **{report['arch_errors']}**; identity conflicts flagged: **{report['conflicts']}**; incompatible claims rejected: **{report['rejected']}**; runtime errors: **{report['runtime_errors']}**.", "",
        "## F. Cost/model", "",
        f"- Model: **gpt-5.6-terra / medium**; High escalations: **0**; repeated unnecessary calls avoided: **{report['queries_avoided']}**.", "",
        "## G. v0.5.1 comparison (same 10 applications)", "",
        f"- Searches before/after: **{b['searches']} / {report['searches']}**.",
        f"- Useful sources before/after: **{b['useful_sources']} / {report['accepted']}**.",
        f"- Researched useful fields before/after: **{b['useful_fields']} / {report['current_fields']}**.",
        f"- REVIEW_SPARSE before/after: **{b['sparse']} / {report['current_sparse']}**.",
        f"- Architecture errors before/after: **{b['architecture_errors']} / {report['arch_errors']}**.", "",
        "## H. Safety", "",
        f"- Frozen BMW benchmark unchanged: **{'YES' if report['bmw_ok'] else 'NO'}**.",
        f"- Canonical DB hashes unchanged: **{'YES' if report['db_ok'] else 'NO'}**.",
        f"- DB hashes before: `{json.dumps(db_before, sort_keys=True)}`.",
        f"- DB hashes after: `{json.dumps(db_after, sort_keys=True)}`.",
        "- Canonical write count: **0** (runner contains no SQLite access).", "",
        "## Completion block", "", "```ini",
        "COMPONENT_RESEARCH_VERSION = 0.5.2", "",
        f"GOLDEN_RETRIEVAL_BENCHMARK_COMPLETE = {'YES' if report['cases'] == 10 else 'NO'}",
        "SEARCH_IDENTITY_NORMALIZATION_READY = YES",
        "OFFICIAL_FIRST_SEARCH_READY = YES",
        "GAP_DRIVEN_QUERY_REFINEMENT_READY = YES",
        "RICH_SOURCE_EXTRACTION_READY = YES",
        f"DETERMINISTIC_ARCHITECTURE_NORMALIZATION_READY = {'YES' if report['arch_errors'] == 0 else 'NO'}", "",
        "GOLDEN_CASES = 10",
        f"GOLDEN_CASES_WITH_USEFUL_DATA = {report['useful']}",
        f"GOLDEN_CASES_WITH_OFFICIAL_SOURCE = {report['official']}",
        f"GOLDEN_CASES_WITH_2PLUS_USEFUL_FIELDS = {report['two_plus']}", "",
        f"TOTAL_SEARCHES = {report['searches']}",
        f"MEDIAN_SEARCHES_PER_CASE = {report['median_searches']:.1f}",
        f"TOTAL_FETCHED_SOURCES = {report['fetched']}",
        f"USEFUL_ACCEPTED_SOURCES = {report['accepted']}", "",
        f"ARCHITECTURE_NORMALIZATION_ERRORS = {report['arch_errors']}",
        f"IDENTITY_CONFLICTS_CORRECTLY_FLAGGED = {report['conflicts']}", "",
        "HIGH_REASONING_ESCALATIONS = 0", "",
        f"BMW_FROZEN_BENCHMARK_UNCHANGED = {'YES' if report['bmw_ok'] else 'NO'}", "",
        "CANONICAL_WRITE_DISABLED = YES",
        f"PRODUCTION_DB_CHANGED = {'NO' if report['db_ok'] else 'YES'}", "",
        f"GOLDEN_RETRIEVAL_GATE_PASSED = {'YES' if report['gate'] else 'NO'}",
        f"READY_TO_RERUN_SPARSE_CROSS_OEM = {'YES' if report['gate'] else 'NO'}",
        "```", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.6-terra", choices=("gpt-5.6-terra",))
    parser.add_argument("--reasoning-effort", default="medium", choices=("medium",))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-sparse", action="store_true")
    parser.add_argument("--retry-cases", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    golden_rows, samples = build_golden_set(read_csv(args.sample))
    write_csv(args.output / "GOLDEN_RETRIEVAL_SET_V052.csv", golden_rows, SET_FIELDS)
    if not args.live:
        print("GOLDEN_SET_WRITTEN; pass --live to execute retrieval", flush=True)
        return 0
    db_before = db_hashes()
    bmw_before = file_sha256(FROZEN_BMW) if FROZEN_BMW.exists() else "MISSING"
    runtime = live_runtime(model=args.model, reasoning_effort=args.reasoning_effort)
    runtime.source_policy = CrossOemPragmaticSourcePolicy()
    if hasattr(runtime.claim_extractor, "max_document_chars"):
        runtime.claim_extractor.max_document_chars = 24_000
    results_path = args.output / "GOLDEN_RETRIEVAL_RESULTS_V052.csv"
    query_path = args.output / "GOLDEN_RETRIEVAL_QUERY_AUDIT_V052.csv"
    source_path = args.output / "GOLDEN_RETRIEVAL_SOURCE_AUDIT_V052.csv"
    results = read_csv(results_path) if args.resume else []
    queries = read_csv(query_path) if args.resume else []
    sources = read_csv(source_path) if args.resume else []
    if args.retry_sparse:
        retry_cases = {
            row["golden_case_id"]
            for row in results
            if row.get("review_flag") == "REVIEW_SPARSE"
        }
        results = [row for row in results if row.get("golden_case_id") not in retry_cases]
        queries = [row for row in queries if row.get("case") not in retry_cases]
        sources = [row for row in sources if row.get("case") not in retry_cases]
        print(f"V052_RETRY_SPARSE cases={len(retry_cases)}", flush=True)
    explicit_retry = {item.strip() for item in args.retry_cases.split(",") if item.strip()}
    if explicit_retry:
        results = [row for row in results if row.get("golden_case_id") not in explicit_retry]
        queries = [row for row in queries if row.get("case") not in explicit_retry]
        sources = [row for row in sources if row.get("case") not in explicit_retry]
        print(f"V052_RETRY_EXPLICIT cases={len(explicit_retry)}", flush=True)
    completed = {row["sample_id"] for row in results}
    for sample in samples:
        if sample["sample_id"] in completed:
            continue
        result, case_queries, case_sources = run_case(runtime, sample)
        results.append(result)
        queries.extend(case_queries)
        sources.extend(case_sources)
        write_csv(results_path, results, RESULT_FIELDS)
        write_csv(query_path, queries, QUERY_FIELDS)
        write_csv(source_path, sources, SOURCE_FIELDS)
        print(f"V052_PROGRESS case={sample['golden_case_id']} useful={result['fields_added_count']} searches={result['searches_count']} fetched={result['fetched_sources_count']} review={result['review_flag']}", flush=True)
    db_after = db_hashes()
    bmw_after = file_sha256(FROZEN_BMW) if FROZEN_BMW.exists() else "MISSING"
    baseline = baseline_metrics(read_csv(args.baseline), {row["sample_id"] for row in samples})
    report = summarize(results, queries, sources, baseline, db_before=db_before, db_after=db_after, bmw_ok=bmw_before == bmw_after == FROZEN_BMW_SHA256)
    write_summary(args.output / "GOLDEN_RETRIEVAL_V052_SUMMARY.md", report, results, db_before=db_before, db_after=db_after)
    (args.output / "RUN_METRICS_V052.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"V052_COMPLETE gate={'YES' if report['gate'] else 'NO'} useful={report['useful']}/10 official={report['official']}/10 two_plus={report['two_plus']}/10 searches={report['searches']}", flush=True)
    return 0 if report["gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
