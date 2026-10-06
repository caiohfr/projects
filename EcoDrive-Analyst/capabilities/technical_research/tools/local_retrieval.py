from __future__ import annotations

from typing import Protocol, Sequence

from ..contracts import FetchedDocument, TechnicalResearchRequest
from ..knowledge import SQLiteEvidenceStore


class LocalRetriever(Protocol):
    def retrieve(self, request: TechnicalResearchRequest, *, limit: int) -> Sequence[FetchedDocument]: ...


class NullLocalRetriever:
    def retrieve(self, request: TechnicalResearchRequest, *, limit: int) -> Sequence[FetchedDocument]:
        return ()


class EvidenceStoreRetriever:
    def __init__(self, store: SQLiteEvidenceStore):
        self.store = store

    def retrieve(self, request: TechnicalResearchRequest, *, limit: int) -> Sequence[FetchedDocument]:
        query = " ".join(
            str(request.known_fields.get(field, ""))
            for field in ("make", "model", "model_year", "engine", "transmission_type")
        ).strip()
        return self.store.search(query, limit=limit)

