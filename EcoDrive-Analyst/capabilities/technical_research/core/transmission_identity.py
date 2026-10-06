from __future__ import annotations

from dataclasses import replace
import re

from ..contracts import EvidenceClaim


TRANSMISSION_HARDWARE_FIELD = "transmission_hardware_designation"
TRANSMISSION_MARKETING_FIELD = "transmission_marketing_description"
_LEGACY_DESIGNATION_FIELD = "transmission_designation"

_HARDWARE_PATTERNS = (
    re.compile(r"\bGA\d+[A-Z]{2}\d+[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}HP\d{2,3}[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\b\d{3,4}[A-Z]{1,3}\b", re.IGNORECASE),
    re.compile(r"\bCVT\d+[A-Z]?\b", re.IGNORECASE),
)
_MARKETING_MARKERS = (
    "SPEED AUTOMATIC",
    "SPEED STEPTRONIC",
    "STEPTRONIC SPORT",
    "M STEPTRONIC",
    "WITH DRIVELOGIC",
    "AUTOMATIC TRANSMISSION",
)


def contains_hardware_designation(value: object) -> bool:
    text = str(value or "").upper().replace("-", " ")
    return any(pattern.search(text) for pattern in _HARDWARE_PATTERNS)


def normalize_transmission_identity_claim(claim: EvidenceClaim) -> EvidenceClaim:
    if claim.field not in {_LEGACY_DESIGNATION_FIELD, TRANSMISSION_HARDWARE_FIELD}:
        return claim
    text = str(claim.value or "").upper()
    if "FAMILY" in text and contains_hardware_designation(text):
        return replace(claim, field="transmission_family")
    if contains_hardware_designation(text):
        return replace(claim, field=TRANSMISSION_HARDWARE_FIELD)
    if any(marker in text for marker in _MARKETING_MARKERS) or claim.field == _LEGACY_DESIGNATION_FIELD:
        return replace(claim, field=TRANSMISSION_MARKETING_FIELD)
    # A value asserted as hardware but without a recognizable technical code is
    # retained as description, never promoted to hardware identity.
    return replace(claim, field=TRANSMISSION_MARKETING_FIELD)
