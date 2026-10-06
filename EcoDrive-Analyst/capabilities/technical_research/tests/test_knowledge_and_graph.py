from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
import unittest

from capabilities.technical_research import (
    TechnicalResearchRequest,
    research_batch,
    research_batch_with_summary,
    research_technical_component,
)
from capabilities.technical_research.agent import ResearchRuntime
from capabilities.technical_research.contracts import (
    FetchedDocument,
    IngestionStatus,
    ResearchLimits,
    ResearchStatus,
    SourceRecord,
    SourceTier,
)
from capabilities.technical_research.core import SourcePolicy
from capabilities.technical_research.knowledge import (
    KnowledgeIngestionService,
    SQLiteEvidenceStore,
    content_fingerprint,
)
from capabilities.technical_research.profiles import TransmissionResearchProfile
from capabilities.technical_research.tools import (
    EvidenceStoreRetriever,
    FixtureDocumentFetcher,
    FixtureSearchProvider,
    MetadataClaimExtractor,
    NullLocalRetriever,
)


def _source(source_id: str = "bmw-oem", *, tier: SourceTier = SourceTier.TIER_1_PRIMARY) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        url=f"fixture://{source_id}",
        title="BMW technical data",
        publisher="BMW",
        tier=tier,
        domain_tags=("TRANSMISSION",),
        application_tags={"make": "BMW", "model": "330i", "model_year": 2023},
        metadata={
            "claims": [
                {
                    "field": "transmission_designation",
                    "value": "GA8HP50Z",
                    "evidence_location": "page 12",
                    "evidence_text": "Transmission designation GA8HP50Z",
                }
            ]
        },
    )


def _document(source: SourceRecord | None = None, content: str = "BMW 330i GA8HP50Z transmission") -> FetchedDocument:
    selected = source or _source()
    return FetchedDocument(
        source=selected,
        content=content,
        content_hash=content_fingerprint(content),
    )


class RecordingVectorizer:
    available = True

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def index(self, source_id, chunks):
        self.calls.append((source_id, list(chunks)))


class KnowledgeAndGraphTests(unittest.TestCase):
    def test_ingestion_deduplicates_identical_documents(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteEvidenceStore(Path(folder) / "evidence.db")
            service = KnowledgeIngestionService(store)
            first = service.ingest(_document())
            second = service.ingest(_document(_source("same-content-alias")))
            self.assertEqual(first.status, IngestionStatus.INGEST)
            self.assertEqual(second.status, IngestionStatus.METADATA_ONLY)
            self.assertEqual(store.counts()["sources"], 1)

    def test_rejected_source_is_not_vectorized(self):
        with tempfile.TemporaryDirectory() as folder:
            vectorizer = RecordingVectorizer()
            store = SQLiteEvidenceStore(Path(folder) / "evidence.db")
            service = KnowledgeIngestionService(store, vectorizer=vectorizer)
            decision = service.ingest(_document(_source("unknown", tier=SourceTier.UNCLASSIFIED)))
            self.assertEqual(decision.status, IngestionStatus.REJECT)
            self.assertEqual(vectorizer.calls, [])
            self.assertEqual(store.counts()["sources"], 0)

    def test_local_retrieval_satisfies_request_without_web_search(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteEvidenceStore(Path(folder) / "evidence.db")
            KnowledgeIngestionService(store).ingest(_document())
            search = FixtureSearchProvider({})
            runtime = ResearchRuntime(
                profile=TransmissionResearchProfile(),
                search_provider=search,
                document_fetcher=FixtureDocumentFetcher({}),
                claim_extractor=MetadataClaimExtractor(),
                local_retriever=EvidenceStoreRetriever(store),
                source_policy=SourcePolicy(),
            )
            request = TechnicalResearchRequest(
                domain="TRANSMISSION",
                known_fields={"make": "BMW", "model": "330i", "model_year": 2023},
            )
            result = research_technical_component(request, runtime=runtime)
            self.assertEqual(result.status, ResearchStatus.SUPPORTED)
            self.assertEqual(result.candidate.identity, "GA8HP50Z")
            self.assertEqual(search.calls, 0)

    def test_batch_reuses_exact_request_without_collapsing_distinct_applications(self):
        limits = ResearchLimits(max_search_rounds=1, max_search_queries_per_round=1)
        requests = [
            TechnicalResearchRequest(
                domain="TRANSMISSION",
                known_fields={"make": "BMW", "model": model, "model_year": 2023},
                limits=limits,
            )
            for model in ("330i", "330i", "430i")
        ]
        search = FixtureSearchProvider({})
        runtime = ResearchRuntime(
            profile=TransmissionResearchProfile(),
            search_provider=search,
            document_fetcher=FixtureDocumentFetcher({}),
            claim_extractor=MetadataClaimExtractor(),
            local_retriever=NullLocalRetriever(),
            source_policy=SourcePolicy(),
        )
        results = research_batch(requests, runtime=runtime)
        self.assertEqual(len(results), 3)
        self.assertNotEqual(results[0].request_id, results[1].request_id)
        self.assertEqual(search.calls, 2)
        self.assertEqual([item.status for item in results], [ResearchStatus.INSUFFICIENT_EVIDENCE] * 3)

    def test_batch_summary_reports_reuse_status_and_canonical_boundary(self):
        limits = ResearchLimits(max_search_rounds=1, max_search_queries_per_round=1)
        requests = [
            TechnicalResearchRequest(
                domain="TRANSMISSION",
                known_fields={"make": "BMW", "model": model, "model_year": 2023},
                limits=limits,
            )
            for model in ("330i", "330i", "430i")
        ]
        search = FixtureSearchProvider({})
        runtime = ResearchRuntime(
            profile=TransmissionResearchProfile(),
            search_provider=search,
            document_fetcher=FixtureDocumentFetcher({}),
            claim_extractor=MetadataClaimExtractor(),
            local_retriever=NullLocalRetriever(),
            source_policy=SourcePolicy(),
        )

        batch = research_batch_with_summary(requests, runtime=runtime)

        self.assertEqual(len(batch.results), 3)
        self.assertEqual(batch.summary.request_count, 3)
        self.assertEqual(batch.summary.unique_request_count, 2)
        self.assertEqual(batch.summary.reused_request_count, 1)
        self.assertEqual(batch.summary.status_counts[ResearchStatus.INSUFFICIENT_EVIDENCE.value], 3)
        self.assertEqual(batch.summary.confidence_counts["UNRESOLVED"], 3)
        self.assertEqual(batch.summary.domain_counts, {"TRANSMISSION": 3})
        self.assertEqual(batch.summary.canonical_write, "DISABLED")
        self.assertEqual(search.calls, 2)

    def test_persistent_result_cache_is_transparent(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteEvidenceStore(Path(folder) / "evidence.db")
            search = FixtureSearchProvider({})
            runtime = ResearchRuntime(
                profile=TransmissionResearchProfile(),
                search_provider=search,
                document_fetcher=FixtureDocumentFetcher({}),
                claim_extractor=MetadataClaimExtractor(),
                local_retriever=NullLocalRetriever(),
                source_policy=SourcePolicy(),
                cache_store=store,
            )
            limits = ResearchLimits(max_search_rounds=1, max_search_queries_per_round=1)
            first_request = TechnicalResearchRequest(
                domain="TRANSMISSION",
                known_fields={"make": "BMW", "model": "330i", "model_year": 2023},
                limits=limits,
            )
            second_request = TechnicalResearchRequest(
                domain="TRANSMISSION",
                known_fields={"make": "BMW", "model": "330i", "model_year": 2023},
                limits=limits,
            )
            research_technical_component(first_request, runtime=runtime)
            cached = research_technical_component(second_request, runtime=runtime)
            self.assertEqual(search.calls, 1)
            self.assertTrue(cached.provenance["cache_hit"])
            self.assertEqual(cached.request_id, second_request.request_id)

    def test_agent_stops_at_configured_search_limits(self):
        search = FixtureSearchProvider({})
        runtime = ResearchRuntime(
            profile=TransmissionResearchProfile(),
            search_provider=search,
            document_fetcher=FixtureDocumentFetcher({}),
            claim_extractor=MetadataClaimExtractor(),
            local_retriever=type("NoHits", (), {"retrieve": lambda self, request, limit: ()})(),
            source_policy=SourcePolicy(),
        )
        request = TechnicalResearchRequest(
            domain="TRANSMISSION",
            known_fields={"make": "BMW", "model": "330i", "model_year": 2023},
            limits=ResearchLimits(max_search_rounds=2, max_search_queries_per_round=2),
        )
        result = research_technical_component(request, runtime=runtime)
        self.assertEqual(result.stop_reason, "SEARCH_LIMIT_REACHED")
        self.assertEqual(search.calls, 4)

    def test_canonical_database_hash_is_unchanged(self):
        root = Path(__file__).resolve().parents[3]
        candidate = root / "data/db/staging/eco_drive_canonical_candidate.db"
        if not candidate.exists():
            self.skipTest("canonical candidate DB not present")
        before = hashlib.sha256(candidate.read_bytes()).hexdigest()
        research_technical_component(
            TechnicalResearchRequest(
                domain="TRANSMISSION",
                known_fields={"make": "BMW", "model": "330i", "model_year": 2023},
            )
        )
        after = hashlib.sha256(candidate.read_bytes()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
