from __future__ import annotations

from ..contracts import TechnicalResearchRequest


PUBLIC_RESEARCH_FIELDS = frozenset(
    {
        "make",
        "model",
        "model_year",
        "trim",
        "engine",
        "electrification",
        "drive_type",
        "transmission_type",
        "gears",
        "final_drive_ratio",
        "nv_ratio",
        "category",
    }
)


def validate_request(request: TechnicalResearchRequest) -> None:
    if request.domain.upper() != "TRANSMISSION":
        raise ValueError(f"Unsupported research domain: {request.domain}")
    private_fields = sorted(set(request.known_fields) - PUBLIC_RESEARCH_FIELDS)
    if private_fields:
        raise ValueError(
            "Request contains fields outside the public research allowlist: "
            + ", ".join(private_fields)
        )
    if not request.known_fields.get("make") or not request.known_fields.get("model"):
        raise ValueError("make and model are required for transmission research")


def public_request_fields(request: TechnicalResearchRequest) -> dict[str, object]:
    validate_request(request)
    return {
        key: value
        for key, value in request.known_fields.items()
        if key in PUBLIC_RESEARCH_FIELDS and value not in (None, "")
    }

