from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from .contracts import SourceRecord
from .live_phasea1 import (
    LiveSearchProvider,
    PhaseA1Cache,
    _extract_claim,
    _source_from_json,
    _source_to_json,
    _stable_json,
    cache_key,
    normalize_query,
)
from .tools.fetch_document import SafeDocumentFetcher


PATCH_VERSION = "SPRINT12_RESEARCH_AGENT_FINAL_MICROPATCH_V1"
FINAL_STATUS = "SPRINT12_AGENT_FROZEN_FOR_REVIEW"
GOLDEN_TASK_SPECS = (
    ("GOLDEN-TOYOTA-AVALON", "RT-58F6FAFBF5D647E5", "ARCHITECTURE"),
    ("GOLDEN-TOYOTA-HIGHLANDER", "RT-58EFA7A290C2E764", "ARCHITECTURE"),
    ("GOLDEN-HYUNDAI-SANTA-FE", "RT-C519340F31320050", "ARCHITECTURE"),
    ("GOLDEN-MAZDA3-BOUNDARY", "RT-02BB5A52D4C596CB", "TRANSMISSION_AXLE_BOUNDARY"),
    ("GOLDEN-LEXUS-RX350", "RT-F967A8DCBB29A70C", "TRANSMISSION_IDENTITY"),
    ("GOLDEN-MERCEDES-EQE", "RT-65E19704D4DE7ABA", "ARCHITECTURE"),
)

QUERY_TEMPLATES = {
    "ARCHITECTURE": (
        "ARCHITECTURE_V1",
        "{years} {make} {model} {drive} {transmission_type} powertrain transmission architecture technical specifications",
    ),
    "TRANSMISSION_IDENTITY": (
        "TRANSMISSION_IDENTITY_V1",
        "{years} {make} {model} {hardware_code} {transmission_type} transmission technical specification",
    ),
    "TRANSMISSION_AXLE_BOUNDARY": (
        "TRANSMISSION_AXLE_BOUNDARY_V1",
        "{hardware_code} {make} {model} transaxle final drive differential service manual technical",
    ),
}

RESULT_FIELDS = [
    "golden_task_id", "child_task_id", "make", "model", "claim_type",
    "internal_attempted", "internal_status", "internal_sources_used",
    "estimator_unlock_value", "research_disposition", "live_search_skipped_reason",
    "query_template_id", "query_text", "search_result_count", "after_relevance_gate_count",
    "pre_fetch_rejected_count", "sources_fetched", "accepted_evidence", "final_status",
]
REJECTION_FIELDS = [
    "golden_task_id", "child_task_id", "search_rank", "source_id", "url", "title",
    "relevance_status", "rejection_reason", "entity_match", "claim_match",
    "technical_document_signal", "fetch_attempted",
]
RANKING_FIELDS = [
    "golden_task_id", "child_task_id", "search_rank", "post_gate_rank", "source_id", "url", "title",
    "relevance_status", "rejection_reason", "source_tier", "entity_match", "claim_match",
    "technical_document_signal", "fetch_attempted",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _years(task: dict[str, str]) -> str:
    low, high = task["model_year_min"], task["model_year_max"]
    return low if low == high else f"{low}-{high}"


def build_query(task: dict[str, str], claim_type: str) -> tuple[str, str]:
    template_id, template = QUERY_TEMPLATES[claim_type]
    hardware = task.get("transmission_code_raw", "").strip() or task.get("engine_code_raw", "").strip()
    query = template.format(
        years=_years(task), make=task["make"], model=task["model"],
        drive=task.get("drive_type_raw", ""), transmission_type=task.get("transmission_type_raw", ""),
        hardware_code=hardware,
    )
    return template_id, " ".join(query.split())


def _claim_matches(claim_type: str, evidence: dict[str, Any]) -> bool:
    actual = str(evidence.get("claim_type", "")).upper()
    if claim_type == "ARCHITECTURE":
        return actual in {"DRIVETRAIN_ARCHITECTURE", "POWERTRAIN_ARCHITECTURE"}
    if claim_type == "TRANSMISSION_AXLE_BOUNDARY":
        return "BOUNDARY" in actual
    if claim_type == "TRANSMISSION_IDENTITY":
        return actual in {"TRANSMISSION_IDENTITY", "HARDWARE_IDENTITY", "TRANSMISSION_FAMILY"}
    return False


def resolve_internal(
    task: dict[str, str], claim_type: str, prior_evidence: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [
        row for row in prior_evidence
        if str(row.get("child_task_id")) == task["child_task_id"] and _claim_matches(claim_type, row)
    ]
    scoped: list[dict[str, Any]] = []
    for row in candidates:
        try:
            covers = (
                int(row.get("model_year_min")) <= int(task["model_year_min"])
                and int(row.get("model_year_max")) >= int(task["model_year_max"])
            )
        except (TypeError, ValueError):
            covers = False
        confidence = str(row.get("confidence", "")).upper()
        approximate = "APPROX" in str(row.get("provenance", "")).upper()
        if covers and confidence in {"HIGH", "MEDIUM"} and not approximate:
            scoped.append(row)
    source_ids = sorted({
        str(row.get("source_url_or_id") or row.get("source_title") or "PHASEA_STAGING")
        for row in candidates
    })
    if scoped:
        status = "INTERNAL_RESOLVED"
    elif candidates:
        status = "INTERNAL_PARTIAL"
    else:
        status = "INTERNAL_NOT_FOUND"
    return {
        "internal_attempted": True,
        "internal_status": status,
        "internal_sources_used": ["CANONICAL_TASK_METADATA", *source_ids],
        "resolved_evidence": scoped,
        "candidate_evidence_count": len(candidates),
    }


def estimator_unlock(task: dict[str, str], claim_type: str) -> tuple[str, str, str]:
    transmission = task.get("transmission_type_raw", "").upper()
    architecture = task.get("internal_evidence_summary", "").upper()
    try:
        gears = int(task.get("gears", ""))
    except ValueError:
        gears = 0
    if claim_type == "TRANSMISSION_IDENTITY" and gears > 1:
        return "HIGH", "RESEARCH_NOW", "CURRENT_MULTI_SPEED_ESTIMATOR_CAN_USE_IDENTITY"
    if claim_type == "TRANSMISSION_AXLE_BOUNDARY" and gears > 1:
        return "MEDIUM", "RESEARCH_NOW", "BOUNDARY_CAN_QUALIFY_EXISTING_MULTI_SPEED_ROUTE"
    if claim_type == "ARCHITECTURE":
        if gears > 1 and "CONTINUOUS" not in transmission and "ELECTRIC" not in task.get("propulsion_architecture_raw", "").upper():
            return "HIGH", "RESEARCH_NOW", "CURRENT_HYBRID_OR_CONVENTIONAL_MULTI_SPEED_ROUTE_AVAILABLE"
        if "CONVENTIONAL_MULTI_SPEED" in architecture:
            return "HIGH", "RESEARCH_NOW", "CURRENT_MULTI_SPEED_ESTIMATOR_AVAILABLE"
        return "NONE", "DEFER_NO_CURRENT_MODEL_VALUE", "NO_EXISTING_MODEL_FOR_RESEARCHED_ARCHITECTURE"
    return "LOW", "DEFER_LOW_LEVERAGE", "CLAIM_NOT_ACTIONABLE_BY_CURRENT_ESTIMATOR"


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 2}


def source_relevance(source: SourceRecord, task: dict[str, str], claim_type: str) -> dict[str, Any]:
    snippet = str(source.metadata.get("snippet", ""))
    haystack = " ".join((source.title, source.url, source.publisher, snippet)).lower()
    host = (urlsplit(source.url).hostname or source.publisher or "").lower()
    make_tokens = _tokens(task["make"])
    model_tokens = _tokens(task["model"])
    hardware = (task.get("transmission_code_raw", "").strip() or task.get("engine_code_raw", "").strip()).lower()
    entity_score = sum(token in haystack for token in make_tokens | model_tokens)
    if hardware and hardware in haystack:
        entity_score += 3
    claim_terms = {
        "ARCHITECTURE": ("architecture", "powertrain", "hybrid", "electric", "transmission", "specification"),
        "TRANSMISSION_IDENTITY": ("transmission", "gearbox", "automatic", "speed", "specification", "technical"),
        "TRANSMISSION_AXLE_BOUNDARY": ("transaxle", "final drive", "differential", "service manual", "transmission"),
    }[claim_type]
    claim_score = sum(term in haystack for term in claim_terms)
    technical_terms = ("technical", "specification", "service manual", "media", "press", "certification", "pdf")
    technical_score = sum(term in haystack for term in technical_terms)
    automotive_terms = (
        "vehicle", "automotive", "car", "powertrain", "transmission", "hybrid", "transaxle",
        "differential", "toyota", "lexus", "hyundai", "mazda", "mercedes", "epa", "nhtsa",
    )
    automotive = any(term in haystack for term in automotive_terms)
    obvious_bad = any(term in haystack for term in (
        "schneier", "cybersecurity", "apartment", "real estate", "ecommerce", "shopify", "vacation destination",
    ))
    official_tokens = tuple(token for token in make_tokens if token not in {"benz"}) + (
        ".gov", "epa.gov", "nhtsa.gov", "pressroom", "media.",
    )
    authoritative = any(token in host for token in official_tokens)
    if obvious_bad:
        status, reason = "REJECT", "REJECT_NON_AUTOMOTIVE_CONTEXT"
    elif entity_score == 0:
        status, reason = "REJECT", "REJECT_UNRELATED_ENTITY"
    elif not automotive:
        status, reason = "REJECT", "REJECT_NON_AUTOMOTIVE_CONTEXT"
    elif claim_score == 0:
        status, reason = "REJECT", "REJECT_LOW_INFORMATION_GENERIC_PAGE"
    elif authoritative and technical_score + claim_score >= 2:
        status, reason = "ELIGIBLE", "AUTHORITATIVE_ENTITY_AND_CLAIM_CONTEXT"
    elif entity_score >= 1 and claim_score >= 1 and technical_score >= 1:
        status, reason = "ELIGIBLE", "ENTITY_AND_TECHNICAL_CLAIM_CONTEXT"
    else:
        status, reason = "REJECT", "REJECT_LOW_INFORMATION_GENERIC_PAGE"
    return {
        "relevance_status": status,
        "rejection_reason": "" if status == "ELIGIBLE" else reason,
        "entity_match": entity_score,
        "claim_match": claim_score,
        "technical_document_signal": technical_score,
        "authoritative": authoritative,
    }


def source_tier(source: SourceRecord, relevance: dict[str, Any]) -> str:
    if relevance["relevance_status"] != "ELIGIBLE":
        return "REJECT"
    host = (urlsplit(source.url).hostname or source.publisher or "").lower()
    if relevance["authoritative"] or host.endswith(".gov"):
        return "TIER_1"
    if any(token in host for token in ("sae.org", "zf.com", "aisin", "bosch", "service", "manual")):
        return "TIER_2"
    if any(token in host for token in ("caranddriver", "motortrend", "greencarcongress", "automotive")):
        return "TIER_3"
    return "TIER_4"


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    tier_order = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3, "TIER_4": 4, "REJECT": 9}
    return (
        tier_order[row["source_tier"]], -int(row["entity_match"]), -int(row["claim_match"]),
        -int(row["technical_document_signal"]), row["url"].lower(),
    )


def _extract_golden_claim(
    task: dict[str, str], claim_type: str, source: SourceRecord, text: str, content_hash: str,
) -> dict[str, Any] | None:
    lowered = text.lower()
    model_tokens = _tokens(task["model"])
    if model_tokens and not any(token in lowered for token in model_tokens):
        return None
    if claim_type in {"ARCHITECTURE", "TRANSMISSION_AXLE_BOUNDARY"}:
        extraction_task = dict(task)
        if claim_type == "TRANSMISSION_AXLE_BOUNDARY" and not extraction_task.get("transmission_code_raw"):
            extraction_task["transmission_code_raw"] = extraction_task.get("engine_code_raw", "")
        return _extract_claim(extraction_task, source, text, content_hash)
    if claim_type == "TRANSMISSION_IDENTITY":
        try:
            gears = int(task.get("gears", ""))
        except ValueError:
            return None
        patterns = (f"{gears}-speed", f"{gears} speed", f"{gears}‑speed")
        if gears > 1 and "transmission" in lowered and any(pattern in lowered for pattern in patterns):
            return {
                "child_task_id": task["child_task_id"], "claim_type": "TRANSMISSION_IDENTITY",
                "application_make": task["make"], "application_model": task["model"],
                "model_year_min": int(task["model_year_min"]), "model_year_max": int(task["model_year_max"]),
                "normalized_value": f"CONVENTIONAL_{gears}_SPEED_TRANSMISSION",
                "confidence": "MEDIUM", "source_id": source.source_id, "source_url": source.url,
                "source_title": source.title, "source_content_sha256": content_hash,
                "provenance": "LIVE_RESEARCH_FETCHED_SOURCE", "validation_status": "SCOPED_FETCHED_TEXT_VALIDATED",
            }
    return None


def _result_hash(results: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> str:
    stable_results = [{key: value for key, value in row.items()} for row in results]
    stable_evidence = sorted(evidence, key=lambda row: (row["child_task_id"], row["source_id"], row["claim_type"]))
    return hashlib.sha256(_stable_json({"results": stable_results, "evidence": stable_evidence}).encode("utf-8")).hexdigest().upper()


def run_golden_validation(
    *,
    tasks: list[dict[str, str]],
    prior_evidence: list[dict[str, Any]],
    output_dir: Path,
    cache_dir: Path,
    cache_only: bool,
    provider: Any | None = None,
    fetcher: Any | None = None,
) -> dict[str, Any]:
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = PhaseA1Cache(cache_dir)
    provider = provider or LiveSearchProvider()
    fetcher = fetcher or SafeDocumentFetcher(timeout_seconds=12, max_bytes=10_000_000, allow_ddgs_fallback=False)
    by_id = {task["child_task_id"]: task for task in tasks}
    golden = [(golden_id, by_id[task_id], claim_type) for golden_id, task_id, claim_type in GOLDEN_TASK_SPECS]

    golden_tasks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    unlock_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    search_requests = source_fetches = 0

    for golden_id, task, claim_type in golden:
        template_id, query = build_query(task, claim_type)
        internal = resolve_internal(task, claim_type, prior_evidence)
        unlock, disposition, unlock_reason = estimator_unlock(task, claim_type)
        golden_tasks.append({
            "golden_task_id": golden_id, "child_task_id": task["child_task_id"], "make": task["make"],
            "model": task["model"], "model_year_min": task["model_year_min"], "model_year_max": task["model_year_max"],
            "claim_type": claim_type, "query_template_id": template_id, "query_text": query,
        })
        unlock_rows.append({
            "golden_task_id": golden_id, "child_task_id": task["child_task_id"], "claim_type": claim_type,
            "estimator_unlock_value": unlock, "research_disposition": disposition, "reason_code": unlock_reason,
        })
        base = {
            "golden_task_id": golden_id, "child_task_id": task["child_task_id"], "make": task["make"],
            "model": task["model"], "claim_type": claim_type, "internal_attempted": True,
            "internal_status": internal["internal_status"],
            "internal_sources_used": ";".join(internal["internal_sources_used"]),
            "estimator_unlock_value": unlock, "research_disposition": disposition,
            "query_template_id": template_id, "query_text": query,
            "search_result_count": 0, "after_relevance_gate_count": 0,
            "pre_fetch_rejected_count": 0, "sources_fetched": 0, "accepted_evidence": 0,
        }
        if internal["internal_status"] == "INTERNAL_RESOLVED":
            base.update({"live_search_skipped_reason": "INTERNAL_CLAIM_ALREADY_RESOLVED", "final_status": "RESOLVED_EXACT"})
            results.append(base)
            continue
        if disposition != "RESEARCH_NOW":
            base.update({"live_search_skipped_reason": unlock_reason, "final_status": disposition})
            results.append(base)
            continue
        base["live_search_skipped_reason"] = ""
        normalized = normalize_query(query)
        search_key = cache_key(provider.provider_name, normalized, "final_patch_search", {"limit": 5})
        cached = cache.get("search", search_key)
        if cached is None:
            if cache_only:
                raise RuntimeError(f"Cache-only replay miss for {golden_id}")
            search_requests += 1
            cached = [_source_to_json(source) for source in provider.search(query, limit=5)]
            cache.put("search", search_key, cached)
        sources = [_source_from_json(row) for row in cached]
        base["search_result_count"] = len(sources)
        eligible_rows: list[tuple[SourceRecord, dict[str, Any]]] = []
        for search_rank, source in enumerate(sources, start=1):
            relevance = source_relevance(source, task, claim_type)
            tier = source_tier(source, relevance)
            audit = {
                "golden_task_id": golden_id, "child_task_id": task["child_task_id"], "search_rank": search_rank,
                "post_gate_rank": "", "source_id": source.source_id, "url": source.url, "title": source.title,
                **relevance, "source_tier": tier, "fetch_attempted": False,
            }
            rankings.append(audit)
            if relevance["relevance_status"] == "ELIGIBLE":
                eligible_rows.append((source, audit))
            else:
                rejections.append(dict(audit))
        eligible_rows.sort(key=lambda pair: _rank_key(pair[1]))
        base["after_relevance_gate_count"] = len(eligible_rows)
        base["pre_fetch_rejected_count"] = len(sources) - len(eligible_rows)
        claims: list[dict[str, Any]] = []
        for post_rank, (source, audit) in enumerate(eligible_rows[:2], start=1):
            audit["post_gate_rank"] = post_rank
            audit["fetch_attempted"] = True
            fetch_key = cache_key("HTTP_FETCH", source.url, "final_patch_fetch", {"max_bytes": 10_000_000})
            document = cache.get("fetch", fetch_key)
            if document is None:
                if cache_only:
                    raise RuntimeError(f"Cache-only replay miss for source {source.source_id}")
                source_fetches += 1
                try:
                    fetched = fetcher.fetch(source)
                    document = {
                        "status": "OK", "content": fetched.content, "content_hash": fetched.content_hash,
                        "content_type": fetched.content_type, "source_id": source.source_id,
                    }
                except Exception as exc:
                    document = {"status": "ERROR", "content": "", "content_hash": "", "error": type(exc).__name__, "source_id": source.source_id}
                cache.put("fetch", fetch_key, document)
            base["sources_fetched"] += 1
            if document["status"] == "OK":
                claim = _extract_golden_claim(task, claim_type, source, document["content"], document["content_hash"])
                if claim:
                    claims.append(claim)
                    evidence.append(claim)
        base["accepted_evidence"] = len(claims)
        if claims:
            base["final_status"] = "RESOLVED_EXACT" if any(row.get("confidence") == "HIGH" for row in claims) else "RESOLVED_STRONG"
        elif internal["internal_status"] == "INTERNAL_PARTIAL":
            base["final_status"] = "BOUNDARY_STILL_UNKNOWN" if claim_type == "TRANSMISSION_AXLE_BOUNDARY" else "PARTIAL_EVIDENCE"
        else:
            base["final_status"] = "NOT_FOUND"
        results.append(base)

    result_hash = _result_hash(results, evidence)
    _write_csv(output_dir / "research_agent_golden_tasks.csv", golden_tasks, list(golden_tasks[0]))
    _write_csv(output_dir / "research_agent_golden_results.csv", results, RESULT_FIELDS)
    _write_csv(output_dir / "research_agent_relevance_rejections.csv", rejections, REJECTION_FIELDS)
    _write_csv(output_dir / "research_agent_source_ranking.csv", rankings, RANKING_FIELDS)
    _write_csv(output_dir / "research_agent_unlock_gate.csv", unlock_rows, list(unlock_rows[0]))
    (output_dir / "research_agent_evidence_staging.jsonl").write_text(
        "".join(_stable_json(row) + "\n" for row in evidence), encoding="utf-8"
    )
    summary = {
        "version": PATCH_VERSION, "mode": "CACHE_ONLY_REPLAY" if cache_only else "LIVE_NETWORK",
        "provider": provider.provider_name, "golden_task_count": len(golden),
        "internal_resolved": sum(row["internal_status"] == "INTERNAL_RESOLVED" for row in results),
        "internal_partial": sum(row["internal_status"] == "INTERNAL_PARTIAL" for row in results),
        "internal_not_found": sum(row["internal_status"] == "INTERNAL_NOT_FOUND" for row in results),
        "network_skipped_tasks": sum(bool(row["live_search_skipped_reason"]) for row in results),
        "search_request_count": search_requests, "search_results_returned": sum(int(row["search_result_count"]) for row in results),
        "pre_fetch_rejected": len(rejections), "sources_fetched": sum(int(row["sources_fetched"]) for row in results),
        "network_source_fetches": source_fetches, "accepted_evidence": len(evidence),
        "deferred_no_current_model_value": sum(row["final_status"] == "DEFER_NO_CURRENT_MODEL_VALUE" for row in results),
        "cache_hits": cache.hits, "cache_misses": cache.misses, "structured_result_sha256": result_hash,
        "direct_research_abc_writes": 0, "status_counts": dict(Counter(row["final_status"] for row in results)),
    }
    (output_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = [
    "FINAL_STATUS", "GOLDEN_TASK_SPECS", "PATCH_VERSION", "build_query", "estimator_unlock",
    "resolve_internal", "run_golden_validation", "source_relevance", "source_tier",
]
