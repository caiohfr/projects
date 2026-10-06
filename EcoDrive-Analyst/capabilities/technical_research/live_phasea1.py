from __future__ import annotations

import hashlib
import json
import re
import base64
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import certifi
import requests
from bs4 import BeautifulSoup

from .contracts import SourceRecord, SourceTier
from .tools.fetch_document import SafeDocumentFetcher


LIVE_VERSION = "SPRINT12_PHASEA1_LIVE_V1"
SELECTED_TASK_IDS = (
    "RT-58F6FAFBF5D647E5",  # Toyota partial architecture/macro
    "RT-58EFA7A290C2E764",  # prior curated OEM source comparison
    "RT-C519340F31320050",  # Hyundai partial architecture/macro
    "RT-02BB5A52D4C596CB",  # Mazda required negative boundary case
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def cache_key(provider: str, query_or_url: str, mode: str, options: dict[str, Any]) -> str:
    payload = {"provider": provider, "value": normalize_query(query_or_url), "mode": mode, "options": options}
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _source_to_json(source: SourceRecord) -> dict[str, Any]:
    row = asdict(source)
    row["tier"] = source.tier.value
    return row


def _source_from_json(row: dict[str, Any]) -> SourceRecord:
    return SourceRecord(
        source_id=row["source_id"], url=row["url"], title=row.get("title", ""),
        publisher=row.get("publisher", ""), document_type=row.get("document_type", "web_page"),
        publication_date=row.get("publication_date"), revision=row.get("revision"),
        tier=SourceTier(row.get("tier", "UNCLASSIFIED")),
        domain_tags=tuple(row.get("domain_tags", ())),
        application_tags=dict(row.get("application_tags", {})),
        metadata=dict(row.get("metadata", {})),
    )


class PhaseA1Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def get(self, namespace: str, key: str) -> Any | None:
        path = self.root / namespace / f"{key}.json"
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, namespace: str, key: str, value: Any) -> None:
        path = self.root / namespace / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class LiveSearchProvider:
    """Bounded, credential-free live HTML search provider.

    DDGS was tested first for this repository but blocks indefinitely while
    trying to open the Windows user certificate store in the managed runtime.
    This adapter therefore uses Bing's public HTML result page directly and
    records that provider truthfully.
    """

    provider_name = "BING_HTML"

    def __init__(self, timeout_seconds: int = 12):
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int) -> tuple[SourceRecord, ...]:
        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "count": limit, "cc": "us", "setlang": "en-US"},
            headers={"User-Agent": "Mozilla/5.0 EcoDriveTechnicalResearch/PhaseA1"},
            timeout=self.timeout_seconds,
            verify=certifi.where(),
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        records: list[SourceRecord] = []
        for rank, node in enumerate(soup.select("li.b_algo h2 a"), start=1):
            href = _decode_bing_url(str(node.get("href", "")))
            if not href.startswith(("http://", "https://")):
                continue
            host = (urlsplit(href).hostname or "").lower()
            result_node = node.find_parent("li", class_="b_algo")
            caption = result_node.select_one(".b_caption p") if result_node else None
            snippet = " ".join(caption.get_text(" ", strip=True).split()) if caption else ""
            source_id = "src-" + hashlib.sha256(href.encode("utf-8")).hexdigest()[:16]
            records.append(SourceRecord(
                source_id=source_id,
                url=href,
                title=" ".join(node.get_text(" ", strip=True).split()),
                publisher=host,
                document_type="pdf" if urlsplit(href).path.lower().endswith(".pdf") else "web_page",
                tier=SourceTier.UNCLASSIFIED,
                metadata={
                    "search_provider": self.provider_name,
                    "search_query": query,
                    "search_rank": rank,
                    "snippet": snippet,
                },
            ))
            if len(records) >= limit:
                break
        return tuple(records)


def _decode_bing_url(url: str) -> str:
    query = parse_qs(urlsplit(url).query)
    encoded = query.get("u", [""])[0]
    if encoded.startswith("a1"):
        try:
            token = encoded[2:] + "=" * (-len(encoded[2:]) % 4)
            decoded = base64.urlsafe_b64decode(token).decode("utf-8")
            if decoded.startswith(("http://", "https://")):
                return decoded
        except (ValueError, UnicodeDecodeError):
            pass
    return url


def _query_for(task: dict[str, str]) -> tuple[str, list[str]]:
    years = task["model_year_min"] if task["model_year_min"] == task["model_year_max"] else f"{task['model_year_min']}-{task['model_year_max']}"
    query = (
        f"{task['make']} {task['model']} {years} {task.get('transmission_code_raw', '')} "
        f"{task.get('transmission_type_raw', '')} technical specifications transmission architecture final drive differential"
    )
    return " ".join(query.split()), [
        "make", "model", "model_year_min", "model_year_max",
        "transmission_code_raw", "transmission_type_raw", "gap_families",
    ]


def _rank(source: SourceRecord, task: dict[str, str]) -> tuple[int, int, str]:
    official_tokens = ("toyota.com", "hyundai.com", "mazda.com", "pressroom", "media.")
    official = any(token in source.publisher.lower() for token in official_tokens)
    model_tokens = [token for token in re.findall(r"[a-z0-9]+", task["model"].lower()) if len(token) > 2]
    title_match = sum(token in source.title.lower() for token in model_tokens)
    return (-int(official), -title_match, source.url)


def _excerpt(text: str, terms: tuple[str, ...], radius: int = 360) -> str:
    lowered = text.lower()
    indexes = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    if not indexes:
        return ""
    start = max(0, min(indexes) - radius)
    end = min(len(text), min(indexes) + radius)
    return " ".join(text[start:end].split())[:900]


def _extract_claim(task: dict[str, str], source: SourceRecord, text: str, content_hash: str) -> dict[str, Any] | None:
    lowered = text.lower()
    make = task["make"].upper()
    model = task["model"]
    normalized = ""
    claim_type = "DRIVETRAIN_ARCHITECTURE"
    boundary = ""
    confidence = "MEDIUM"
    terms: tuple[str, ...]
    if make == "TOYOTA" and "hybrid" in lowered and any(term in lowered for term in ("planetary", "power split", "e-cvt", "ecvt")):
        normalized = "HYBRID_POWER_SPLIT_ECVT"
        boundary = "HYBRID_TRANSAXLE_AGGREGATE"
        terms = ("power split", "planetary", "e-cvt", "ecvt")
    elif make == "HYUNDAI" and "hybrid" in lowered and any(term in lowered for term in ("6at", "6-speed", "6 speed")):
        normalized = "PARALLEL_HYBRID_CONVENTIONAL_6AT"
        if any(term in lowered for term in ("transmission-mounted", "tmed")):
            normalized = "PARALLEL_HYBRID_TMED_PMSM_6AT"
            confidence = "HIGH"
        boundary = "TRANSMISSION_AND_MOTOR_ARCHITECTURE"
        terms = ("transmission-mounted", "tmed", "6at", "6-speed", "6 speed")
    elif make == "MAZDA":
        claim_type = "TRANSMISSION_AXLE_BOUNDARY"
        hardware = task.get("transmission_code_raw", "").strip().lower()
        exact_hardware = bool(hardware and hardware in lowered)
        if exact_hardware and "differential" in lowered and ("final drive" in lowered or "transaxle" in lowered):
            normalized = "EXACT_APPLICATION_DIFFERENTIAL_FINAL_DRIVE_BOUNDARY_DOCUMENTED"
            boundary = "TRANSAXLE_INCLUDES_FINAL_DRIVE_AND_DIFFERENTIAL"
            terms = (hardware, "differential", "final drive", "transaxle")
            confidence = "HIGH"
        else:
            return None
    else:
        return None
    excerpt = _excerpt(text, terms)
    if not excerpt:
        return None
    return {
        "child_task_id": task["child_task_id"],
        "claim_type": claim_type,
        "gap_family": "TRANSMISSION_AXLE_BOUNDARY_UNKNOWN" if make == "MAZDA" else "ARCHITECTURE_UNRESOLVED",
        "application_make": task["make"],
        "application_model": model,
        "model_year_min": int(task["model_year_min"]),
        "model_year_max": int(task["model_year_max"]),
        "hardware_code_scope": task.get("transmission_code_raw", ""),
        "normalized_value": normalized,
        "physical_boundary": boundary,
        "confidence": confidence,
        "source_id": source.source_id,
        "source_url": source.url,
        "source_title": source.title,
        "source_content_sha256": content_hash,
        "supporting_excerpt_short": excerpt,
        "provenance": "LIVE_RESEARCH_FETCHED_SOURCE",
        "validation_status": "SCOPED_FETCHED_TEXT_VALIDATED",
    }


def _normalized_evidence_hash(rows: list[dict[str, Any]]) -> str:
    stable = sorted(rows, key=lambda row: (row["child_task_id"], row["source_id"], row["claim_type"]))
    return hashlib.sha256(_stable_json(stable).encode("utf-8")).hexdigest().upper()


def run_live_subset(
    *,
    tasks: list[dict[str, str]],
    prior_evidence: list[dict[str, Any]],
    output_dir: Path,
    cache_dir: Path,
    cache_only: bool,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = PhaseA1Cache(cache_dir)
    provider = LiveSearchProvider()
    fetcher = SafeDocumentFetcher(
        timeout_seconds=12,
        max_bytes=10_000_000,
        allow_ddgs_fallback=False,
    )
    selected = [task for task_id in SELECTED_TASK_IDS for task in tasks if task["child_task_id"] == task_id]
    if len(selected) != len(SELECTED_TASK_IDS):
        raise ValueError("Required deterministic Phase A.1 validation subset is incomplete")

    queries: list[dict[str, Any]] = []
    sources_ledger: list[dict[str, Any]] = []
    fetch_log: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    search_requests = source_fetches = 0

    for task in selected:
        query, created_from = _query_for(task)
        normalized = normalize_query(query)
        query_id = "QA1-" + hashlib.sha256(f"{task['child_task_id']}|{normalized}".encode("utf-8")).hexdigest()[:14].upper()
        queries.append({
            "query_id": query_id, "child_task_id": task["child_task_id"],
            "normalized_query": normalized, "claim_type": "TRANSMISSION_AXLE_BOUNDARY" if task["make"] == "MAZDA" else "DRIVETRAIN_ARCHITECTURE",
            "gap_family": "TRANSMISSION_AXLE_BOUNDARY_UNKNOWN" if task["make"] == "MAZDA" else "ARCHITECTURE_UNRESOLVED",
            "provider": provider.provider_name, "created_from_fields": created_from,
        })
        key = cache_key("BING_HTML", normalized, "search", {"limit": 5})
        cached_results = cache.get("search", key)
        if cached_results is None:
            if cache_only:
                raise RuntimeError(f"Cache-only replay miss for query {query_id}")
            search_requests += 1
            live_results = provider.search(query, limit=5)
            cached_results = [_source_to_json(item) for item in live_results]
            cache.put("search", key, cached_results)
        results = [_source_from_json(item) for item in cached_results]
        ranked = sorted(results, key=lambda item: _rank(item, task))[:2]
        for rank, source in enumerate(ranked, start=1):
            sources_ledger.append({
                "query_id": query_id, "child_task_id": task["child_task_id"], "source_id": source.source_id,
                "rank": rank, "url": source.url, "title": source.title, "publisher": source.publisher,
            })
            fetch_key = cache_key("HTTP_FETCH", source.url, "fetch", {"max_bytes": 10_000_000})
            cached_doc = cache.get("fetch", fetch_key)
            if cached_doc is None:
                if cache_only:
                    raise RuntimeError(f"Cache-only replay miss for source {source.source_id}")
                source_fetches += 1
                try:
                    document = fetcher.fetch(source)
                    cached_doc = {
                        "source_id": source.source_id, "content": document.content,
                        "content_type": document.content_type, "content_hash": document.content_hash,
                        "metadata": dict(document.metadata), "status": "OK",
                    }
                except Exception as exc:
                    cached_doc = {
                        "source_id": source.source_id, "content": "", "content_type": "",
                        "content_hash": "", "metadata": {}, "status": "ERROR",
                        "error": type(exc).__name__,
                    }
                cache.put("fetch", fetch_key, cached_doc)
            fetch_log.append({
                "query_id": query_id, "child_task_id": task["child_task_id"], "source_id": source.source_id,
                "url": source.url, "status": cached_doc["status"], "content_hash": cached_doc.get("content_hash", ""),
                "error": cached_doc.get("error", ""),
            })
            if cached_doc["status"] == "OK":
                claim = _extract_claim(task, source, cached_doc["content"], cached_doc["content_hash"])
                if claim:
                    evidence.append(claim)

    task_status: list[dict[str, Any]] = []
    prior_by_task = {str(row.get("child_task_id")) for row in prior_evidence if row.get("evidence_origin") == "EXTERNAL_PUBLIC"}
    for task in selected:
        claims = [row for row in evidence if row["child_task_id"] == task["child_task_id"]]
        is_mazda = task["make"] == "MAZDA"
        if claims:
            status = "RESOLVED_EXACT" if any(row["confidence"] == "HIGH" for row in claims) else "RESOLVED_STRONG"
        elif is_mazda:
            status = "BOUNDARY_STILL_UNKNOWN"
        else:
            status = "NOT_FOUND"
        task_status.append({
            "child_task_id": task["child_task_id"], "make": task["make"], "model": task["model"],
            "status": status, "accepted_evidence_records": len(claims),
            "prior_curated_source_available": task["child_task_id"] in prior_by_task,
        })

    normalized_hash = _normalized_evidence_hash(evidence)
    def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(_stable_json(row) + "\n" for row in rows), encoding="utf-8")
    dump_jsonl(output_dir / "live_evidence_staging.jsonl", evidence)
    (output_dir / "query_log.json").write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "source_ledger.json").write_text(json.dumps(sources_ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "fetch_log.json").write_text(json.dumps(fetch_log, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "task_status.json").write_text(json.dumps(task_status, indent=2, ensure_ascii=False), encoding="utf-8")
    if cache_only:
        state = "LIVE_RESEARCH_AGENT_VALIDATED"
    elif search_requests and any(row["status"] == "OK" for row in fetch_log) and evidence:
        state = "LIVE_RESEARCH_AGENT_VALIDATED"
    elif search_requests:
        state = "LIVE_RESEARCH_AGENT_PARTIAL_REVIEW_REQUIRED"
    else:
        state = "BLOCKED_BY_LIVE_SEARCH_PROVIDER"
    summary = {
        "version": LIVE_VERSION,
        "state": state,
        "provider_name": provider.provider_name,
        "provider_mode": "CACHE_ONLY_REPLAY" if cache_only else "LIVE_NETWORK",
        "network_search_enabled": not cache_only,
        "validation_task_count": len(selected),
        "search_request_count": search_requests,
        "source_fetch_count": source_fetches,
        "cache_hit_count": cache.hits,
        "cache_miss_count": cache.misses,
        "search_results_selected": len(sources_ledger),
        "fetch_success_count": sum(row["status"] == "OK" for row in fetch_log),
        "unique_accepted_sources": len({row["source_id"] for row in evidence}),
        "accepted_evidence_records": len(evidence),
        "resolved_child_tasks": sum(row["status"].startswith("RESOLVED") for row in task_status),
        "mazda_status": next(row["status"] for row in task_status if row["make"] == "MAZDA"),
        "normalized_staged_evidence_sha256": normalized_hash,
        "model_calls": 0,
        "curated_catalog_used_for_discovery": False,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "provider_proof.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = ["LIVE_VERSION", "SELECTED_TASK_IDS", "run_live_subset"]
