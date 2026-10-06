from __future__ import annotations

from typing import Protocol, Sequence

from ..contracts import TechnicalResearchRequest


class ResearchProfile(Protocol):
    domain: str
    identity_field: str

    def build_queries(self, request: TechnicalResearchRequest, *, round_number: int) -> Sequence[str]: ...

    def evidence_is_sufficient(self, status: str) -> bool: ...

