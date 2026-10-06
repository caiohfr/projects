from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import certifi

from ..contracts import SourceRecord, SourceTier


class SearchProvider(Protocol):
    @property
    def available(self) -> bool: ...

    def search(self, query: str, *, limit: int) -> Sequence[SourceRecord]: ...


class DDGSSearchProvider:
    """Credential-free live discovery. Authority is assigned downstream."""

    provider_name = "DDGS"

    def __init__(self, *, timeout_seconds: int = 15):
        self.timeout_seconds = timeout_seconds
        self.audit: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        try:
            import ddgs  # noqa: F401
        except ImportError:
            return False
        return True

    def search(self, query: str, *, limit: int) -> Sequence[SourceRecord]:
        from ddgs import DDGS

        retrieved_at = datetime.now(timezone.utc).isoformat()
        try:
            raw = list(
                DDGS(timeout=self.timeout_seconds, verify=certifi.where()).text(
                    query, max_results=limit
                )
            )
        except Exception as exc:
            self.audit.append(
                {
                    "retrieved_at": retrieved_at,
                    "provider": self.provider_name,
                    "query": query,
                    "sources_returned": 0,
                    "status": "ERROR",
                    "error": type(exc).__name__,
                }
            )
            raise
        records: list[SourceRecord] = []
        for rank, item in enumerate(raw, start=1):
            url = str(item.get("href") or item.get("url") or "")
            host = (urlsplit(url).hostname or "").lower()
            title = str(item.get("title") or "")
            document_type = "pdf" if urlsplit(url).path.lower().endswith(".pdf") else "web_page"
            records.append(
                SourceRecord(
                    source_id=_source_id(url),
                    url=url,
                    title=title,
                    publisher=host,
                    document_type=document_type,
                    tier=SourceTier.UNCLASSIFIED,
                    metadata={
                        "snippet": str(item.get("body") or item.get("snippet") or ""),
                        "search_provider": self.provider_name,
                        "search_query": query,
                        "search_rank": rank,
                        "retrieved_at": retrieved_at,
                    },
                )
            )
        self.audit.append(
            {
                "retrieved_at": retrieved_at,
                "provider": self.provider_name,
                "query": query,
                "sources_returned": len(records),
                "status": "OK",
                "error": "",
            }
        )
        return tuple(records)


class NullSearchProvider:
    @property
    def available(self) -> bool:
        return False

    def search(self, query: str, *, limit: int) -> Sequence[SourceRecord]:
        return ()


@dataclass
class FixtureSearchProvider:
    results_by_query: Mapping[str, Sequence[SourceRecord]]
    calls: int = 0

    @property
    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int) -> Sequence[SourceRecord]:
        self.calls += 1
        exact = self.results_by_query.get(query)
        if exact is not None:
            return tuple(exact[:limit])
        lowered = query.lower()
        for key, results in self.results_by_query.items():
            if key.lower() in lowered or lowered in key.lower():
                return tuple(results[:limit])
        return ()


class LangChainSearchProviderAdapter:
    """Adapter for any configured LangChain tool/Runnable returning search rows."""

    def __init__(
        self,
        runnable: Any,
        *,
        result_parser: Callable[[Any], Sequence[Mapping[str, Any]]] | None = None,
    ):
        self.runnable = runnable
        self.result_parser = result_parser or self._default_parser

    @property
    def available(self) -> bool:
        return self.runnable is not None

    def search(self, query: str, *, limit: int) -> Sequence[SourceRecord]:
        raw = self.runnable.invoke({"query": query, "max_results": limit})
        records = []
        for item in self.result_parser(raw)[:limit]:
            url = str(item.get("url", ""))
            source_id = str(item.get("source_id") or _source_id(url))
            records.append(
                SourceRecord(
                    source_id=source_id,
                    url=url,
                    title=str(item.get("title", "")),
                    publisher=str(item.get("publisher", "")),
                    document_type=str(item.get("document_type", "web_page")),
                    publication_date=item.get("publication_date"),
                    revision=item.get("revision"),
                    # Provider metadata is discovery-only and never authority.
                    tier=SourceTier.UNCLASSIFIED,
                    domain_tags=tuple(item.get("domain_tags", ())),
                    application_tags=dict(item.get("application_tags", {})),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return tuple(records)

    @staticmethod
    def _default_parser(raw: Any) -> Sequence[Mapping[str, Any]]:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("results", "items", "sources"):
                if isinstance(raw.get(key), list):
                    return raw[key]
        raise ValueError("Search Runnable output needs a result_parser")


def _source_id(url: str) -> str:
    return "src-" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
