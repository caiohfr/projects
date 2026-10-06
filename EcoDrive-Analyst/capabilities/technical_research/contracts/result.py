from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from .candidate import (
    AttributeStatus,
    FieldEvidenceSummary,
    IdentityConfidence,
    TechnicalCandidate,
)
from .evidence import (
    ApplicationMatch,
    ConflictResolution,
    EvidenceBundle,
    EvidenceClaim,
    EvidenceConflict,
    ExtractionMethod,
)
from .source import SourceTier


class ResearchStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_FOUND = "NOT_FOUND"
    ERROR = "ERROR"


@dataclass(frozen=True)
class TechnicalResearchResult:
    run_id: str
    request_id: str
    status: ResearchStatus
    domain: str
    candidate: TechnicalCandidate
    evidence: EvidenceBundle
    provenance: Mapping[str, Any] = field(default_factory=dict)
    research_summary: str = ""
    source_summary: Mapping[str, Any] = field(default_factory=dict)
    ingestion_summary: Mapping[str, Any] = field(default_factory=dict)
    trace: tuple[Mapping[str, Any], ...] = ()
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        def normalize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value

        return normalize(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TechnicalResearchResult":
        candidate_payload = payload["candidate"]
        candidate = TechnicalCandidate(
            identity=candidate_payload.get("identity"),
            attributes=dict(candidate_payload.get("attributes", {})),
            confidence=IdentityConfidence(candidate_payload["confidence"]),
            supporting_claim_ids=tuple(candidate_payload.get("supporting_claim_ids", ())),
            attribute_status={
                key: AttributeStatus(value)
                for key, value in candidate_payload.get("attribute_status", {}).items()
            },
            field_support={
                key: FieldEvidenceSummary(
                    value=value.get("value"),
                    normalized_value=value.get("normalized_value"),
                    support_status=AttributeStatus(value["support_status"]),
                    source_ids=tuple(value.get("source_ids", ())),
                    strongest_source_tier=(
                        SourceTier(value["strongest_source_tier"])
                        if value.get("strongest_source_tier")
                        else None
                    ),
                    application_match=ApplicationMatch(
                        value.get("application_match", ApplicationMatch.UNKNOWN.value)
                    ),
                    conflict_status=str(value.get("conflict_status", "NONE")),
                )
                for key, value in candidate_payload.get("field_support", {}).items()
            },
        )
        evidence_payload = payload.get("evidence", {})
        claims = tuple(
            EvidenceClaim(
                field=item["field"],
                value=item.get("value"),
                normalized_value=item.get("normalized_value"),
                source_id=item["source_id"],
                source_tier=SourceTier(item["source_tier"]),
                evidence_location=item["evidence_location"],
                evidence_text=item["evidence_text"],
                extraction_method=ExtractionMethod(item["extraction_method"]),
                extraction_confidence=float(item["extraction_confidence"]),
                application_match=ApplicationMatch(item["application_match"]),
                source_url=str(item.get("source_url", "")),
                publisher=str(item.get("publisher", "")),
                document_title=str(item.get("document_title", "")),
                source_classification=str(item.get("source_classification", "UNCLASSIFIED")),
                retrieved_at=str(item.get("retrieved_at", "")),
                application_context=dict(item.get("application_context", {})),
            )
            for item in evidence_payload.get("claims", ())
        )
        conflicts = tuple(
            EvidenceConflict(
                field=item["field"],
                values=tuple(item.get("values", ())),
                supporting_sources=tuple(item.get("supporting_sources", ())),
                source_tiers=tuple(SourceTier(value) for value in item.get("source_tiers", ())),
                application_differences=tuple(item.get("application_differences", ())),
                likely_explanation=item.get("likely_explanation"),
                resolution=ConflictResolution(item["resolution"]),
                resolution_reason=str(item.get("resolution_reason", "")),
                conflict_kind=str(item.get("conflict_kind", "ATTRIBUTE")),
            )
            for item in evidence_payload.get("conflicts", ())
        )
        return cls(
            run_id=str(payload["run_id"]),
            request_id=str(payload["request_id"]),
            status=ResearchStatus(payload["status"]),
            domain=str(payload["domain"]),
            candidate=candidate,
            evidence=EvidenceBundle(claims, conflicts),
            provenance=dict(payload.get("provenance", {})),
            research_summary=str(payload.get("research_summary", "")),
            source_summary=dict(payload.get("source_summary", {})),
            ingestion_summary=dict(payload.get("ingestion_summary", {})),
            trace=tuple(payload.get("trace", ())),
            stop_reason=str(payload.get("stop_reason", "")),
        )


@dataclass(frozen=True)
class TechnicalResearchBatchSummary:
    request_count: int
    unique_request_count: int
    reused_request_count: int
    status_counts: Mapping[str, int]
    confidence_counts: Mapping[str, int]
    domain_counts: Mapping[str, int]
    unique_source_count: int
    conflict_count: int
    cache_hit_count: int
    canonical_write: str = "DISABLED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TechnicalResearchBatchResult:
    results: tuple[TechnicalResearchResult, ...]
    summary: TechnicalResearchBatchSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "summary": self.summary.to_dict(),
        }
