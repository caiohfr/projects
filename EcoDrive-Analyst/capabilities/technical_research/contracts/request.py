from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class ResearchLimits:
    max_search_rounds: int = 3
    max_search_queries_per_round: int = 5
    max_sources_fetched: int = 12
    max_high_quality_sources_used: int = 5

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TechnicalResearchRequest:
    domain: str
    known_fields: Mapping[str, Any]
    target_fields: tuple[str, ...] = ()
    request_id: str = field(default_factory=lambda: f"trr-{uuid4().hex}")
    force_refresh: bool = False
    limits: ResearchLimits = field(default_factory=ResearchLimits)

    def __post_init__(self) -> None:
        if not self.domain or not self.domain.strip():
            raise ValueError("domain is required")
        if not self.known_fields:
            raise ValueError("known_fields cannot be empty")

