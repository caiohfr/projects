from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from ..contracts import FetchedDocument, SourceRecord, SourceTier
from .dedupe import canonicalize_url


class SQLiteEvidenceStore:
    """Independent evidence/cache store. It never opens an EcoDrive database."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_source (
                    source_id TEXT PRIMARY KEY,
                    canonical_url TEXT NOT NULL,
                    publisher TEXT NOT NULL,
                    title TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    publication_date TEXT,
                    revision TEXT,
                    retrieved_at TEXT NOT NULL,
                    source_tier TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    domain_tags_json TEXT NOT NULL,
                    application_tags_json TEXT NOT NULL,
                    ingestion_status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS uq_evidence_source_content
                    ON evidence_source(content_hash);
                CREATE TABLE IF NOT EXISTS evidence_chunk (
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (source_id, chunk_index),
                    FOREIGN KEY (source_id) REFERENCES evidence_source(source_id)
                );
                CREATE TABLE IF NOT EXISTS research_cache (
                    request_hash TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            connection.commit()

    def has_document(self, *, content_hash: str, canonical_url: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT 1 FROM evidence_source WHERE content_hash=? OR canonical_url=? LIMIT 1",
                (content_hash, canonicalize_url(canonical_url)),
            ).fetchone()
        return row is not None

    def add_document(
        self,
        document: FetchedDocument,
        *,
        ingestion_status: str,
        chunks: Iterable[tuple[int, str, dict[str, Any]]],
    ) -> None:
        source = document.source
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO evidence_source (
                    source_id, canonical_url, publisher, title, document_type,
                    publication_date, revision, retrieved_at, source_tier,
                    content_hash, domain_tags_json, application_tags_json,
                    ingestion_status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.source_id,
                    canonicalize_url(source.url),
                    source.publisher,
                    source.title,
                    source.document_type,
                    source.publication_date,
                    source.revision,
                    document.retrieved_at,
                    source.tier.value,
                    document.content_hash,
                    json.dumps(source.domain_tags, sort_keys=True),
                    json.dumps(dict(source.application_tags), sort_keys=True),
                    ingestion_status,
                    json.dumps(dict(source.metadata), sort_keys=True, default=str),
                ),
            )
            for chunk_index, content, metadata in chunks:
                connection.execute(
                    "INSERT OR IGNORE INTO evidence_chunk VALUES (?, ?, ?, ?)",
                    (
                        source.source_id,
                        chunk_index,
                        content,
                        json.dumps(metadata, sort_keys=True, default=str),
                    ),
                )
            connection.commit()

    def search(self, query: str, *, limit: int = 5) -> tuple[FetchedDocument, ...]:
        tokens = {token.lower() for token in query.split() if len(token) >= 2}
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT s.*, c.content, c.metadata_json AS chunk_metadata_json
                FROM evidence_source s
                JOIN evidence_chunk c ON c.source_id=s.source_id
                WHERE s.ingestion_status='INGEST'
                """
            ).fetchall()
        ranked: list[tuple[int, sqlite3.Row]] = []
        for row in rows:
            haystack = f"{row['title']} {row['publisher']} {row['content']}".lower()
            score = sum(token in haystack for token in tokens)
            if score:
                ranked.append((score, row))
        ranked.sort(key=lambda item: (-item[0], item[1]["source_id"]))
        documents: list[FetchedDocument] = []
        seen: set[str] = set()
        for _, row in ranked:
            if row["source_id"] in seen:
                continue
            seen.add(row["source_id"])
            source = SourceRecord(
                source_id=row["source_id"],
                url=row["canonical_url"],
                title=row["title"],
                publisher=row["publisher"],
                document_type=row["document_type"],
                publication_date=row["publication_date"],
                revision=row["revision"],
                tier=SourceTier(row["source_tier"]),
                domain_tags=tuple(json.loads(row["domain_tags_json"])),
                application_tags=json.loads(row["application_tags_json"]),
                metadata=json.loads(row["metadata_json"]),
            )
            documents.append(
                FetchedDocument(
                    source=source,
                    content=row["content"],
                    content_hash=row["content_hash"],
                    retrieved_at=row["retrieved_at"],
                    metadata=json.loads(row["chunk_metadata_json"]),
                )
            )
            if len(documents) >= limit:
                break
        return tuple(documents)

    def cache_get(self, request_hash: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT result_json FROM research_cache WHERE request_hash=?", (request_hash,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, request_hash: str, result: dict[str, Any]) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO research_cache(request_hash, result_json) VALUES (?, ?)",
                (request_hash, json.dumps(result, sort_keys=True, default=str)),
            )
            connection.commit()

    def counts(self) -> dict[str, int]:
        with closing(self._connect()) as connection:
            sources = connection.execute("SELECT COUNT(*) FROM evidence_source").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM evidence_chunk").fetchone()[0]
        return {"sources": int(sources), "chunks": int(chunks)}
