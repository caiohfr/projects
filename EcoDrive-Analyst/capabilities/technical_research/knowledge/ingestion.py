from __future__ import annotations

from dataclasses import dataclass, replace

from ..contracts import FetchedDocument, IngestionStatus, SourceTier
from .chunking import chunk_text
from .dedupe import content_fingerprint
from .store import SQLiteEvidenceStore
from .vectorization import NullVectorizer, Vectorizer


@dataclass(frozen=True)
class IngestionDecision:
    source_id: str
    status: IngestionStatus
    reason: str


class KnowledgeIngestionPolicy:
    def evaluate(self, document: FetchedDocument, *, duplicate: bool) -> IngestionDecision:
        if duplicate:
            return IngestionDecision(document.source.source_id, IngestionStatus.METADATA_ONLY, "DUPLICATE")
        if document.source.tier in {SourceTier.TIER_1_PRIMARY, SourceTier.TIER_2_STRONG_SECONDARY}:
            if not document.content.strip():
                return IngestionDecision(document.source.source_id, IngestionStatus.METADATA_ONLY, "NO_CONTENT")
            return IngestionDecision(document.source.source_id, IngestionStatus.INGEST, "AUTHORITATIVE_TECHNICAL_SOURCE")
        if document.source.tier == SourceTier.TIER_3_DISCOVERY_ONLY:
            return IngestionDecision(document.source.source_id, IngestionStatus.METADATA_ONLY, "DISCOVERY_SOURCE")
        if document.source.tier == SourceTier.TIER_4_WEAK:
            return IngestionDecision(document.source.source_id, IngestionStatus.EPHEMERAL, "WEAK_SOURCE")
        return IngestionDecision(document.source.source_id, IngestionStatus.REJECT, "UNCLASSIFIED_SOURCE")


class KnowledgeIngestionService:
    def __init__(
        self,
        store: SQLiteEvidenceStore,
        *,
        policy: KnowledgeIngestionPolicy | None = None,
        vectorizer: Vectorizer | None = None,
    ):
        self.store = store
        self.policy = policy or KnowledgeIngestionPolicy()
        self.vectorizer = vectorizer or NullVectorizer()

    def ingest(self, document: FetchedDocument) -> IngestionDecision:
        if not document.content_hash:
            document = replace(document, content_hash=content_fingerprint(document.content))
        duplicate = self.store.has_document(
            content_hash=document.content_hash, canonical_url=document.source.url
        )
        decision = self.policy.evaluate(document, duplicate=duplicate)
        if duplicate or decision.status in {IngestionStatus.EPHEMERAL, IngestionStatus.REJECT}:
            return decision
        chunks = chunk_text(document.content) if decision.status == IngestionStatus.INGEST else ()
        payload = tuple(
            (
                chunk.index,
                chunk.text,
                {
                    "start_character": chunk.start_character,
                    "end_character": chunk.end_character,
                    "source_id": document.source.source_id,
                },
            )
            for chunk in chunks
        )
        self.store.add_document(
            document,
            ingestion_status=decision.status.value,
            chunks=payload,
        )
        if decision.status == IngestionStatus.INGEST and self.vectorizer.available:
            self.vectorizer.index(document.source.source_id, [chunk.text for chunk in chunks])
        return decision
