from __future__ import annotations

import ipaddress
import socket
from typing import Mapping, Protocol
from urllib.parse import urljoin, urlsplit

import requests
import certifi
from bs4 import BeautifulSoup

from ..contracts import FetchedDocument, SourceRecord
from ..knowledge import content_fingerprint
from .content_extract import extract_document_text


class DocumentFetcher(Protocol):
    def fetch(self, source: SourceRecord) -> FetchedDocument: ...


class FixtureDocumentFetcher:
    def __init__(self, documents: Mapping[str, FetchedDocument]):
        self.documents = dict(documents)
        self.calls = 0

    def fetch(self, source: SourceRecord) -> FetchedDocument:
        self.calls += 1
        return self.documents[source.source_id]


class SafeDocumentFetcher:
    ALLOWED_CONTENT_TYPES = (
        "text/plain",
        "text/html",
        "application/xhtml+xml",
        "application/json",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        max_bytes: int = 10_000_000,
        allow_ddgs_fallback: bool = True,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.allow_ddgs_fallback = allow_ddgs_fallback

    def fetch(self, source: SourceRecord) -> FetchedDocument:
        try:
            return self._fetch_http(source)
        except requests.RequestException as exc:
            if not self.allow_ddgs_fallback:
                raise
            return self._fetch_ddgs_extract(source, exc)

    def _fetch_http(self, source: SourceRecord) -> FetchedDocument:
        current_url = source.url
        response = None
        for _ in range(6):
            self._validate_public_url(current_url)
            response = requests.get(
                current_url,
                timeout=self.timeout_seconds,
                stream=True,
                allow_redirects=False,
                headers={"User-Agent": "EcoDriveTechnicalResearch/0.1"},
            )
            if response.status_code not in {301, 302, 303, 307, 308}:
                break
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ValueError("Redirect response is missing Location")
            current_url = urljoin(current_url, location)
        else:
            raise ValueError("Document exceeds redirect limit")
        assert response is not None
        response.raise_for_status()
        self._validate_public_url(current_url)
        content_type = response.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0].lower()
        current_path = urlsplit(current_url).path.lower()
        if content_type == "application/octet-stream" and (
            current_path.endswith(".xlsx") or "/xlsx/" in current_path
        ):
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if content_type not in self.ALLOWED_CONTENT_TYPES:
            response.close()
            raise ValueError(f"Unsupported content type: {content_type}")
        body = bytearray()
        try:
            for chunk in response.iter_content(64 * 1024):
                body.extend(chunk)
                if len(body) > self.max_bytes:
                    raise ValueError("Document exceeds configured byte limit")
        finally:
            response.close()
        raw_body = bytes(body)
        outbound_links: list[dict[str, str]] = []
        if content_type in {"text/html", "application/xhtml+xml"}:
            soup = BeautifulSoup(raw_body.decode("utf-8", errors="replace"), "html.parser")
            outbound_links = [
                {
                    "url": urljoin(current_url, str(node.get("href", "")).strip()),
                    "title": " ".join(node.get_text(" ", strip=True).split()),
                }
                for node in soup.find_all("a", href=True)
                if str(node.get("href", "")).strip()
            ]
        text = extract_document_text(raw_body, content_type)
        return FetchedDocument(
            source=source,
            content=text,
            content_type=content_type,
            content_hash=content_fingerprint(text),
            metadata={
                "final_url": current_url,
                "byte_count": len(body),
                "outbound_links": outbound_links,
            },
        )

    def _fetch_ddgs_extract(
        self, source: SourceRecord, original_error: requests.RequestException
    ) -> FetchedDocument:
        self._validate_public_url(source.url)
        try:
            from ddgs import DDGS
        except ImportError:
            raise original_error
        payload = DDGS(
            timeout=max(30, int(self.timeout_seconds)),
            verify=certifi.where(),
        ).extract(source.url, fmt="text_plain")
        content = str(payload.get("content", "")) if isinstance(payload, dict) else ""
        if not content.strip():
            raise original_error
        encoded_size = len(content.encode("utf-8"))
        if encoded_size > self.max_bytes:
            raise ValueError("Extracted document exceeds configured byte limit")
        final_url = str(payload.get("url", source.url)) if isinstance(payload, dict) else source.url
        self._validate_public_url(final_url)
        return FetchedDocument(
            source=source,
            content=content,
            content_type="text/plain",
            content_hash=content_fingerprint(content),
            metadata={
                "final_url": final_url,
                "byte_count": encoded_size,
                "fetch_method": "DDGS_EXTRACT_FALLBACK",
                "http_error": type(original_error).__name__,
            },
        )

    @staticmethod
    def _validate_public_url(url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only public HTTP(S) URLs can be fetched")
        host = parsed.hostname.lower()
        if host == "localhost":
            raise ValueError("Local addresses are blocked")
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443)}
        except socket.gaierror as exc:
            raise ValueError(f"Unable to resolve host: {host}") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise ValueError("Private, loopback, link-local, and reserved addresses are blocked")
