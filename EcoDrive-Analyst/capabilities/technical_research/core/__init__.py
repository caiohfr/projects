from .evidence_policy import EvidenceResolution, evaluate_evidence
from .matching import (
    ApplicationMatchResult,
    VehicleApplicationIdentity,
    application_match,
    match_application,
    normalize_identity_value,
    normalize_transmission_type,
    parse_vehicle_application_identity,
)
from .source_policy import SourcePolicy
from .source_classifier import classify_source, source_with_classification
from .request_audit import audit_research_request
from .group_validation import evaluate_hardware_group
from .benchmarking import consolidate_research_attempts
from .transmission_identity import (
    TRANSMISSION_HARDWARE_FIELD,
    TRANSMISSION_MARKETING_FIELD,
    contains_hardware_designation,
    normalize_transmission_identity_claim,
)
from .validation import PUBLIC_RESEARCH_FIELDS, public_request_fields, validate_request

__all__ = [
    "EvidenceResolution",
    "ApplicationMatchResult",
    "VehicleApplicationIdentity",
    "PUBLIC_RESEARCH_FIELDS",
    "SourcePolicy",
    "application_match",
    "audit_research_request",
    "contains_hardware_designation",
    "consolidate_research_attempts",
    "evaluate_hardware_group",
    "classify_source",
    "evaluate_evidence",
    "match_application",
    "normalize_identity_value",
    "normalize_transmission_type",
    "parse_vehicle_application_identity",
    "source_with_classification",
    "TRANSMISSION_HARDWARE_FIELD",
    "TRANSMISSION_MARKETING_FIELD",
    "normalize_transmission_identity_claim",
    "public_request_fields",
    "validate_request",
]
