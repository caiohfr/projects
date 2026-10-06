from __future__ import annotations

from typing import Protocol, Sequence


class Vectorizer(Protocol):
    @property
    def available(self) -> bool: ...

    def index(self, source_id: str, chunks: Sequence[str]) -> None: ...


class NullVectorizer:
    @property
    def available(self) -> bool:
        return False

    def index(self, source_id: str, chunks: Sequence[str]) -> None:
        return None

