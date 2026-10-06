from __future__ import annotations

from dataclasses import replace
from urllib.parse import urlsplit

from ..contracts import (
    SourceClassification,
    SourceClassificationTier,
    SourceRecord,
    SourceTier,
)


_OEM_HOSTS = (
    "bmwgroup.com",
    "bmwtechinfo.bmwgroup.com",
    "ford.com",
    "fordservicecontent.com",
    "lincoln.com",
    "gm.com",
    "chevrolet.com",
    "cadillac.com",
    "buick.com",
    "gmc.com",
    "toyota.com",
    "lexus.com",
    "lexus.ca",
    "mercedes-benz.com",
    "group-media.mercedes-benz.com",
    "mbusa.com",
    "hyundai.com",
    "hyundaiusa.com",
    "hyundainews.com",
    "kia.com",
    "kiamedia.com",
    "genesis.com",
)
_SUPPLIER_HOSTS = ("zf.com", "press.zf.com", "aisin.com", "borgwarner.com")
_GOVERNMENT_HOSTS = ("epa.gov", "nhtsa.gov", "dot.gov")
_PUBLICATION_HOSTS = ("sae.org",)
_SPECIALIST_HOSTS = (
    "aftermarket.zf.com",
    "realoem.com",
    "fcpeuro.com",
    "ecstuning.com",
)
_FORUM_MARKERS = ("forum", "reddit.com", "bimmerpost.com", "bimmerfest.com")
_TECHNICAL_MARKERS = (
    "technical",
    "specification",
    "specifications",
    "service",
    "training",
    "manual",
    "data sheet",
    "datasheet",
    "database",
    "test car list",
)


def _host_matches(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def classify_source(source: SourceRecord) -> SourceClassification:
    """Classify authority only from observable publisher/document metadata."""
    parsed = urlsplit(source.url)
    host = (parsed.hostname or "").lower()
    combined = " ".join(
        (source.title, source.document_type, parsed.path, str(source.metadata.get("content_type", "")))
    ).lower()
    is_pdf = parsed.path.lower().endswith(".pdf") or "pdf" in source.document_type.lower()
    is_technical = is_pdf or any(marker in combined for marker in _TECHNICAL_MARKERS)

    if _host_matches(host, _FORUM_MARKERS) or "forum" in combined:
        return SourceClassification(
            SourceClassificationTier.WEAK,
            SourceTier.TIER_4_WEAK,
            "COMMUNITY",
            "FORUM",
            "COMMUNITY_OR_FORUM_SOURCE",
        )

    if _host_matches(host, _SPECIALIST_HOSTS):
        return SourceClassification(
            SourceClassificationTier.DISCOVERY_ONLY,
            SourceTier.TIER_3_DISCOVERY_ONLY,
            "SPECIALIST_AFTERMARKET",
            "TECHNICAL_CATALOG" if is_technical else "WEB_PAGE",
            "SPECIALIST_SOURCE_REQUIRES_AUTHORITATIVE_CROSS_CHECK",
        )

    if _host_matches(host, _GOVERNMENT_HOSTS):
        return SourceClassification(
            SourceClassificationTier.PRIMARY_TECHNICAL,
            SourceTier.TIER_1_PRIMARY,
            "GOVERNMENT",
            "GOVERNMENT_DATABASE" if "database" in combined else ("TECHNICAL_PDF" if is_pdf else "GOVERNMENT_RECORD"),
            "KNOWN_GOVERNMENT_TECHNICAL_PUBLISHER",
        )

    if _host_matches(host, _PUBLICATION_HOSTS):
        return SourceClassification(
            SourceClassificationTier.STRONG_TECHNICAL,
            SourceTier.TIER_2_STRONG_SECONDARY,
            "TECHNICAL_PUBLICATION",
            "SAE_PAPER" if is_technical else "PUBLICATION_INDEX",
            "KNOWN_TECHNICAL_PUBLICATION_PUBLISHER",
        )

    if _host_matches(host, _SUPPLIER_HOSTS):
        if is_technical:
            return SourceClassification(
                SourceClassificationTier.STRONG_TECHNICAL,
                SourceTier.TIER_2_STRONG_SECONDARY,
                "COMPONENT_SUPPLIER",
                "SUPPLIER_TECHNICAL_DOCUMENT",
                "KNOWN_SUPPLIER_WITH_TECHNICAL_DOCUMENT_SIGNAL",
            )
        return SourceClassification(
            SourceClassificationTier.DISCOVERY_ONLY,
            SourceTier.TIER_3_DISCOVERY_ONLY,
            "COMPONENT_SUPPLIER",
            "MARKETING_WEB_PAGE",
            "SUPPLIER_DOMAIN_WITHOUT_TECHNICAL_DOCUMENT_SIGNAL",
        )

    if _host_matches(host, _OEM_HOSTS):
        if is_technical:
            return SourceClassification(
                SourceClassificationTier.PRIMARY_TECHNICAL,
                SourceTier.TIER_1_PRIMARY,
                "OEM",
                "OEM_TECHNICAL_PDF" if is_pdf else "OEM_TECHNICAL_PAGE",
                "KNOWN_OEM_WITH_TECHNICAL_DOCUMENT_SIGNAL",
            )
        return SourceClassification(
            SourceClassificationTier.DISCOVERY_ONLY,
            SourceTier.TIER_3_DISCOVERY_ONLY,
            "OEM",
            "OEM_MARKETING_PAGE",
            "OEM_DOMAIN_WITHOUT_TECHNICAL_DOCUMENT_SIGNAL",
        )

    return SourceClassification(
        SourceClassificationTier.UNCLASSIFIED,
        SourceTier.UNCLASSIFIED,
        "UNKNOWN",
        "PDF" if is_pdf else "WEB_PAGE",
        "NO_MATCH_IN_DETERMINISTIC_AUTHORITY_REGISTRY",
    )


def source_with_classification(source: SourceRecord) -> SourceRecord:
    classification = classify_source(source)
    metadata = dict(source.metadata)
    metadata.update(
        {
            "source_classification": classification.source_tier.value,
            "publisher_type": classification.publisher_type,
            "classified_document_type": classification.document_type,
            "classification_reason": classification.classification_reason,
        }
    )
    return replace(
        source,
        tier=classification.policy_tier,
        document_type=classification.document_type,
        metadata=metadata,
    )
