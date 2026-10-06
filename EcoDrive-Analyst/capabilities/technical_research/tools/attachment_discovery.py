from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin, urlsplit

from ..contracts import FetchedDocument, SourceRecord, SourceTier


_TECHNICAL_LINK_MARKERS = (
    "technical specification",
    "technical specifications",
    "technical data",
    "specification sheet",
    "service information",
    "service document",
    "training document",
    "training manual",
    "attachment",
    "download pdf",
)
_TECHNICAL_PATH_MARKERS = (
    "/attachment/",
    "/attachments/",
    "/download/",
    "/downloads/",
    "technical-data",
    "technical_spec",
    "training",
    "service-manual",
    "service_manual",
)


def discover_technical_attachments(
    landing_document: FetchedDocument,
    *,
    limit: int = 10,
) -> tuple[SourceRecord, ...]:
    """Promote only technically signalled links from an accepted landing document."""
    if limit < 1:
        return ()
    landing_url = str(landing_document.metadata.get("final_url") or landing_document.source.url)
    landing_host = (urlsplit(landing_url).hostname or "").lower()
    links: Sequence[Mapping[str, Any]] = landing_document.metadata.get("outbound_links", ())
    discovered: list[SourceRecord] = []
    seen: set[str] = set()
    for raw in links:
        url = urljoin(landing_url, str(raw.get("url", "")).strip())
        title = str(raw.get("title", "")).strip()
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not host:
            continue
        if not _same_publisher_family(landing_host, host):
            continue
        path = parsed.path.lower()
        combined = f"{title} {path}".lower()
        is_pdf = path.endswith(".pdf")
        # XLSX specification sheets are supported; other office/archive
        # binaries remain outside the bounded technical attachment path.
        if path.endswith((".xls", ".docx", ".doc", ".zip")):
            continue
        has_technical_signal = is_pdf or any(
            marker in combined for marker in (*_TECHNICAL_LINK_MARKERS, *_TECHNICAL_PATH_MARKERS)
        )
        if not has_technical_signal or url in seen:
            continue
        seen.add(url)
        discovered.append(
            SourceRecord(
                source_id="src-" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
                url=url,
                title=title or parsed.path.rsplit("/", 1)[-1] or "Technical attachment",
                publisher=host,
                document_type="xlsx" if path.endswith(".xlsx") or "/xlsx/" in path else ("pdf" if is_pdf else "technical_attachment"),
                tier=SourceTier.UNCLASSIFIED,
                application_tags=dict(raw.get("application_tags", {})),
                metadata={
                    **dict(raw.get("metadata", {})),
                    "landing_source_id": landing_document.source.source_id,
                    "discovery_method": "ACCEPTED_LANDING_TECHNICAL_LINK",
                    "is_technical_attachment": True,
                },
            )
        )
        if len(discovered) >= limit:
            break
    return tuple(discovered)


def _same_publisher_family(landing_host: str, candidate_host: str) -> bool:
    if not landing_host or not candidate_host:
        return False
    if landing_host == candidate_host:
        return True
    for family in (
        "bmwgroup.com",
        "bmwtechinfo.bmwgroup.com",
        "zf.com",
        "ford.com",
        "fordservicecontent.com",
        "lincoln.com",
        "gm.com",
        "cadillac.com",
        "chevrolet.com",
        "gmc.com",
        "toyota.com",
        "lexus.com",
        "lexus.ca",
        "mercedes-benz.com",
        "group-media.mercedes-benz.com",
        "mbusa.com",
        "hyundai.com",
        "hyundaiusa.com",
        "hyundainews.com",
        "kia.com",
        "kiamedia.com",
        "genesis.com",
    ):
        if (
            (landing_host == family or landing_host.endswith("." + family))
            and (candidate_host == family or candidate_host.endswith("." + family))
        ):
            return True
    return False
