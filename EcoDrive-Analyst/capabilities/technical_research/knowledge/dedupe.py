from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_KEYS
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), path, urlencode(sorted(query)), "")
    )


def content_fingerprint(content: str) -> str:
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest().upper()


def request_fingerprint(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()

