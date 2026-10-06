from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .candidate import IdentityConfidence


class RequestConsistencyStatus(str, Enum):
    CONSISTENT = "CONSISTENT"
    CONFLICTING_SOURCE_FIELDS = "CONFLICTING_SOURCE_FIELDS"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True)
class RequestFieldConflict:
    field_a: str
    value_a: Any
    field_b: str
    value_b: Any
    conflict_type: str
    source: str = "STRUCTURED_REQUEST"


@dataclass(frozen=True)
class RequestIdentityAudit:
    status: RequestConsistencyStatus
    conflicts: tuple[RequestFieldConflict, ...] = ()
    missing_fields: tuple[str, ...] = ()
    trusted_fields: tuple[str, ...] = ()
    conflicting_fields: tuple[str, ...] = ()


class HardwareGroupStatus(str, Enum):
    HARDWARE_CONFIRMED = "HARDWARE_CONFIRMED"
    HARDWARE_SPLIT = "HARDWARE_SPLIT"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class HardwareGroupMember:
    application_id: str
    independent_application_id: str
    hardware_identity: str | None
    identity_confidence: IdentityConfidence
    marketing_description: str | None = None
    source_ids: tuple[str, ...] = ()
    material_hardware_conflict: bool = False


@dataclass(frozen=True)
class HardwareGroupEvaluation:
    status: HardwareGroupStatus
    n_rows: int
    n_independent_applications: int
    hardware_identities: tuple[str, ...]
    identity_confidences: tuple[str, ...]
    supporting_source_count: int
    notes: str
