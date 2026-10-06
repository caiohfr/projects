"""Pure deterministic matching of vehicle identities to component priors.

This module selects a reference class only.  It deliberately does not use the
reference ABC values and does not adopt a resolution into a VDE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


MATCH_LEVELS = (
    "EXACT_RULE_MATCH",
    "STRONG_RULE_MATCH",
    "CATEGORY_RULE_MATCH",
    "NO_MATCH",
)
MATCH_DOMAINS = ("BRAKE", "TRANSMISSION", "AXLE", "HUB_BEARING")
RULE_VERSION = "1.0"

_DRIVE_ALIASES = {
    "FWD": "FWD",
    "2-WHEEL DRIVE, FRONT": "FWD",
    "RWD": "RWD",
    "2-WHEEL DRIVE, REAR": "RWD",
    "AWD": "AWD",
    "ALL WHEEL DRIVE": "AWD",
    "ALL-WHEEL DRIVE": "AWD",
    "4WD": "4WD",
    "4-WHEEL DRIVE": "4WD",
    "PART-TIME 4-WHEEL DRIVE": "4WD",
}


def _token(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    return text or None


def normalize_drive_system(value: Any) -> str | None:
    """Map only enumerated canonical drive labels; never use fuzzy matching."""
    if value is None:
        return None
    return _DRIVE_ALIASES.get(str(value).strip().upper())


@dataclass(frozen=True)
class ReferencePrior:
    component_id: str
    component_resolution_id: str
    component_domain: str
    boundary: str
    model: str | None = None
    hardware_reference: str | None = None
    application_class: str | None = None
    drive_architecture: str | None = None
    position: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def normalized_application_class(self) -> str | None:
        return _token(self.application_class)

    def normalized_drive_architecture(self) -> str | None:
        return normalize_drive_system(self.drive_architecture)

    def normalized_position(self) -> str | None:
        return _token(self.position)


@dataclass(frozen=True)
class TechnicalIdentity:
    vehicle_configuration_id: str
    drive_system: str | None = None
    category: str | None = None
    application_class: str | None = None
    component_position: str | None = None
    hardware_reference: str | None = None
    propulsion_architecture: str | None = None
    transmission_type: str | None = None
    transmission_model: str | None = None
    gear_count: int | None = None


@dataclass(frozen=True)
class PriorMatchResult:
    domain: str
    boundary: str
    match_level: str
    component_id: str | None
    component_resolution_id: str | None
    rule_id: str
    rule_version: str
    matched_on: tuple[str, ...]
    missing_discriminants: tuple[str, ...]
    reason: str


def _no_match(
    domain: str,
    *,
    rule_id: str,
    missing: Iterable[str] = (),
    reason: str,
    matched_on: Iterable[str] = (),
) -> PriorMatchResult:
    return PriorMatchResult(
        domain=domain,
        boundary=domain,
        match_level="NO_MATCH",
        component_id=None,
        component_resolution_id=None,
        rule_id=rule_id,
        rule_version=RULE_VERSION,
        matched_on=tuple(matched_on),
        missing_discriminants=tuple(sorted(set(missing))),
        reason=reason,
    )


def _domain_match(
    identity: TechnicalIdentity,
    references: tuple[ReferencePrior, ...],
    domain: str,
) -> PriorMatchResult:
    candidates = tuple(ref for ref in references if _token(ref.boundary) == domain)
    if not candidates:
        return _no_match(
            domain,
            rule_id=f"{domain}_CATALOG_ABSENT_V1",
            reason=f"The reference catalog contains no {domain} prior.",
        )

    # Exact hardware matching is intentionally unavailable for generic seed
    # markers such as SYNTHETIC_REFERENCE.
    hardware = _token(identity.hardware_reference)
    if hardware:
        exact = tuple(
            ref
            for ref in candidates
            if _token(ref.hardware_reference) == hardware
            and hardware != "SYNTHETIC_REFERENCE"
        )
        if len(exact) == 1:
            ref = exact[0]
            return PriorMatchResult(
                domain=domain,
                boundary=domain,
                match_level="EXACT_RULE_MATCH",
                component_id=ref.component_id,
                component_resolution_id=ref.component_resolution_id,
                rule_id=f"{domain}_EXPLICIT_HARDWARE_V1",
                rule_version=RULE_VERSION,
                matched_on=(f"hardware_reference={identity.hardware_reference}",),
                missing_discriminants=(),
                reason="A unique non-generic hardware identifier matches exactly.",
            )

    explicit_class = _token(identity.application_class)
    category_class = _token(identity.category)
    catalog_classes = {
        ref.normalized_application_class()
        for ref in candidates
        if ref.normalized_application_class()
    }
    if explicit_class in catalog_classes:
        application_class = explicit_class
        class_source = "application_class"
        level = "STRONG_RULE_MATCH"
    elif category_class in catalog_classes:
        application_class = category_class
        class_source = "category"
        level = "CATEGORY_RULE_MATCH"
    else:
        application_class = None
        class_source = "application_class"
        level = "NO_MATCH"

    drive = normalize_drive_system(identity.drive_system)
    position = _token(identity.component_position)
    missing = []
    if application_class is None:
        missing.append("application_class")
    if drive is None:
        missing.append("drive_system")
    if domain in {"AXLE", "HUB_BEARING"} and position is None:
        missing.append("component_position")
    if missing:
        available = []
        if application_class is not None:
            available.append(f"{class_source}={application_class}")
        if drive is not None:
            available.append(f"drive_system={drive}")
        return _no_match(
            domain,
            rule_id=f"{domain}_MISSING_DISCRIMINANTS_V1",
            missing=missing,
            matched_on=available,
            reason=(
                "Required explicit catalog discriminants are absent or outside "
                "the reviewed mappings; no reference is selected."
            ),
        )

    assert application_class is not None
    assert drive is not None
    filtered = tuple(
        ref
        for ref in candidates
        if ref.normalized_application_class() == application_class
        and ref.normalized_drive_architecture() == drive
    )
    matched_on = [f"{class_source}={application_class}", f"drive_system={drive}"]

    if domain in {"AXLE", "HUB_BEARING"}:
        assert position is not None
        filtered = tuple(
            ref for ref in filtered if ref.normalized_position() == position
        )
        matched_on.append(f"component_position={position}")

    if len(filtered) == 1:
        ref = filtered[0]
        return PriorMatchResult(
            domain=domain,
            boundary=domain,
            match_level=level,
            component_id=ref.component_id,
            component_resolution_id=ref.component_resolution_id,
            rule_id=f"{domain}_{class_source.upper()}_DRIVE_V1",
            rule_version=RULE_VERSION,
            matched_on=tuple(matched_on),
            missing_discriminants=(),
            reason="Explicit catalog dimensions select one reference prior.",
        )
    if not filtered:
        return _no_match(
            domain,
            rule_id=f"{domain}_CATALOG_GAP_V1",
            matched_on=matched_on,
            reason="No synthetic reference represents this explicit combination.",
        )
    return _no_match(
        domain,
        rule_id=f"{domain}_AMBIGUOUS_REFERENCE_V1",
        matched_on=matched_on,
        reason="More than one synthetic reference remains plausible.",
    )


def match_component_priors(
    technical_identity: TechnicalIdentity,
    reference_catalog: Iterable[ReferencePrior],
) -> list[PriorMatchResult]:
    """Return one deterministic prior decision for each in-scope boundary."""
    references = tuple(
        sorted(
            reference_catalog,
            key=lambda ref: (ref.boundary, ref.component_id, ref.component_resolution_id),
        )
    )
    return [
        _domain_match(technical_identity, references, domain)
        for domain in MATCH_DOMAINS
    ]
