"""Conservative rule-only matching against synthetic component references.

The module is deliberately pure: it normalizes already-stored metadata and
returns candidate prior decisions.  It does not read or write SQLite, inspect
commercial names, use reference ABC values, or adopt a component resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.vde_core.component_prior_matching import ReferencePrior


RULE_VERSION = "1.1"
MATCH_LEVELS = (
    "EXACT_RULE_MATCH",
    "STRONG_RULE_MATCH",
    "CATEGORY_RULE_MATCH",
    "AMBIGUOUS_RULE_MATCH",
    "NO_MATCH",
)
MATCH_BOUNDARIES = ("BRAKE", "TRANSMISSION", "AXLE", "HUB_BEARING")
SLOTTED_BOUNDARIES = frozenset({"AXLE", "HUB_BEARING"})
COMPONENT_SLOTS = ("FRONT", "REAR")


# Raw EPA labels approved for this iteration, plus identity mappings needed to
# consume the already-normalized synthetic-reference metadata.
DRIVE_NORMALIZATION = {
    "2-Wheel Drive, Front": "FWD",
    "2-Wheel Drive, Rear": "RWD",
    "All Wheel Drive": "AWD",
    "4-Wheel Drive": "4WD",
    "Part-time 4-Wheel Drive": "4WD",
    "FWD": "FWD",
    "RWD": "RWD",
    "AWD": "AWD",
    "4WD": "4WD",
}


# Exact mappings retained from normalize_transmission_db / TX_MAP_EXACT in
# notebooks/etl_epa_xlsx_to_sqlite.ipynb, cell 3.  Legacy outputs DCT and SS
# are intentionally outside this iteration's approved output contract and are
# therefore UNMAPPED, as are labels that only resemble a legacy key.
LEGACY_TRANSMISSION_EXACT = {
    "automatic": "AT",
    "manual": "MT",
    "continuously variable": "CVT",
    "selectable continuously variable": "CVT",
    "automated manual": "AMT",
    "automated manual - selectable": "AMT",
    "semi-automatic": "AMT",
    "other": "OT",
}


ELECTRIFICATION_VALUES = frozenset({"ICE", "HEV", "PHEV", "BEV"})


APPLICATION_CLASS_MAP = {
    "MINICOMPACT CARS": ("PASSENGER_LIGHT",),
    "SUBCOMPACT CARS": ("PASSENGER_LIGHT",),
    "COMPACT CARS": ("PASSENGER_LIGHT",),
    "MIDSIZE CARS": ("PASSENGER_STANDARD",),
    "LARGE CARS": ("PASSENGER_STANDARD",),
    "SMALL STATION WAGONS": ("PASSENGER_STANDARD",),
    "MIDSIZE STATION WAGONS": ("PASSENGER_STANDARD",),
    "SMALL SUVS": ("CROSSOVER",),
    "STANDARD SUVS": ("SUV",),
    "MINIVANS": ("PASSENGER_VAN",),
    "SMALL PICKUP TRUCKS": ("PICKUP_LIGHT_DUTY",),
    "STANDARD PICKUP TRUCKS": ("PICKUP_LIGHT_DUTY",),
    "VANS": ("CARGO_VAN", "PASSENGER_VAN"),
    "TWO SEATERS": (),
}
AMBIGUOUS_APPLICATION_CATEGORIES = frozenset({"VANS", "TWO SEATERS"})


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _upper(value: Any) -> str | None:
    text = _clean(value)
    return text.upper() if text else None


@dataclass(frozen=True)
class NormalizedField:
    raw_values: tuple[str, ...]
    normalized: str | None
    source: str
    status: str


def normalize_drive(value: Any, *, source: str = "vehicle_configuration.drive_system") -> NormalizedField:
    raw = _clean(value)
    normalized = DRIVE_NORMALIZATION.get(raw) if raw is not None else None
    return NormalizedField(
        raw_values=(raw,) if raw is not None else (),
        normalized=normalized,
        source=source,
        status="MAPPED" if normalized else ("MISSING" if raw is None else "UNMAPPED"),
    )


def normalize_transmission(
    value: Any,
    *,
    source: str = "vehicle_configuration.transmission_type",
) -> NormalizedField:
    raw = _clean(value)
    normalized = LEGACY_TRANSMISSION_EXACT.get(raw.casefold()) if raw else None
    return NormalizedField(
        raw_values=(raw,) if raw is not None else (),
        normalized=normalized,
        source=source,
        status="MAPPED" if normalized else ("MISSING" if raw is None else "UNMAPPED"),
    )


def normalize_electrification(
    values: Iterable[Any],
    *,
    source: str = "fuelcons.electrification",
) -> NormalizedField:
    raw = tuple(sorted({_upper(value) for value in values if _upper(value)}))
    if not raw:
        return NormalizedField((), None, source, "MISSING")
    invalid = tuple(value for value in raw if value not in ELECTRIFICATION_VALUES)
    if invalid:
        return NormalizedField(raw, None, source, "UNMAPPED")
    if len(raw) != 1:
        return NormalizedField(raw, None, source, "CONFLICT")
    return NormalizedField(raw, raw[0], source, "MAPPED")


@dataclass(frozen=True)
class ApplicationClass:
    raw_values: tuple[str, ...]
    normalized_options: tuple[str, ...]
    source: str
    status: str


def normalize_application_class(
    values: Iterable[Any],
    *,
    source: str = "legacy.vde_db.category",
) -> ApplicationClass:
    raw = tuple(sorted({_upper(value) for value in values if _upper(value)}))
    if not raw:
        return ApplicationClass((), (), source, "MISSING")
    if len(raw) != 1:
        return ApplicationClass(raw, (), source, "CONFLICT")
    category = raw[0]
    if category not in APPLICATION_CLASS_MAP:
        return ApplicationClass(raw, (), source, "UNMAPPED")
    options = APPLICATION_CLASS_MAP[category]
    status = "AMBIGUOUS" if category in AMBIGUOUS_APPLICATION_CATEGORIES else "MAPPED"
    return ApplicationClass(raw, options, source, status)


@dataclass(frozen=True)
class NormalizedTechnicalIdentity:
    vehicle_configuration_id: str
    drive: NormalizedField
    transmission: NormalizedField
    electrification: NormalizedField
    application_class: ApplicationClass


@dataclass(frozen=True)
class PriorDecision:
    domain: str
    boundary: str
    component_slot: str | None
    match_level: str
    candidate_component_ids: tuple[str, ...]
    candidate_resolution_ids: tuple[str, ...]
    rule_id: str
    rule_version: str
    matched_on: tuple[str, ...]
    missing_discriminants: tuple[str, ...]
    reason: str


def _decision(
    domain: str,
    slot: str | None,
    level: str,
    candidates: Iterable[ReferencePrior],
    *,
    rule: str,
    matched_on: Iterable[str] = (),
    missing: Iterable[str] = (),
    reason: str,
) -> PriorDecision:
    ordered = tuple(sorted(candidates, key=lambda ref: (ref.component_id, ref.component_resolution_id)))
    return PriorDecision(
        domain=domain,
        boundary=domain,
        component_slot=slot,
        match_level=level,
        candidate_component_ids=tuple(ref.component_id for ref in ordered),
        candidate_resolution_ids=tuple(ref.component_resolution_id for ref in ordered),
        rule_id=rule,
        rule_version=RULE_VERSION,
        matched_on=tuple(matched_on),
        missing_discriminants=tuple(sorted(set(missing))),
        reason=reason,
    )


def _reference_class(ref: ReferencePrior) -> str | None:
    return _upper(ref.application_class)


def _reference_drive(ref: ReferencePrior) -> str | None:
    raw = _clean(ref.drive_architecture)
    return DRIVE_NORMALIZATION.get(raw) if raw else None


def _reference_position(ref: ReferencePrior) -> str | None:
    return _upper(ref.position)


def _match_one(
    identity: NormalizedTechnicalIdentity,
    references: tuple[ReferencePrior, ...],
    domain: str,
    slot: str | None,
) -> PriorDecision:
    boundary_refs = tuple(ref for ref in references if _upper(ref.boundary) == domain)
    if slot is not None:
        boundary_refs = tuple(ref for ref in boundary_refs if _reference_position(ref) == slot)
    slot_match = (f"position={slot}",) if slot else ()

    application = identity.application_class
    if application.status == "AMBIGUOUS":
        candidates = tuple(
            ref for ref in boundary_refs if _reference_class(ref) in application.normalized_options
        )
        if identity.drive.normalized:
            candidates = tuple(
                ref for ref in candidates if _reference_drive(ref) == identity.drive.normalized
            )
        return _decision(
            domain,
            slot,
            "AMBIGUOUS_RULE_MATCH",
            candidates,
            rule=f"{domain}_AMBIGUOUS_APPLICATION_CLASS_VNEXT",
            matched_on=slot_match,
            missing=("application_class",),
            reason="The approved legacy category mapping is intentionally ambiguous.",
        )
    if application.status != "MAPPED" or len(application.normalized_options) != 1:
        return _decision(
            domain,
            slot,
            "NO_MATCH",
            (),
            rule=f"{domain}_APPLICATION_CLASS_UNAVAILABLE_VNEXT",
            matched_on=slot_match,
            missing=("application_class",),
            reason="No single approved application class is available from stored legacy metadata.",
        )

    app_class = application.normalized_options[0]
    class_refs = tuple(ref for ref in boundary_refs if _reference_class(ref) == app_class)
    matched = (f"application_class={app_class}",) + slot_match
    if not class_refs:
        return _decision(
            domain,
            slot,
            "NO_MATCH",
            (),
            rule=f"{domain}_APPLICATION_CLASS_CATALOG_GAP_VNEXT",
            matched_on=matched,
            reason="The synthetic catalog has no reference for this application class and slot.",
        )

    drive = identity.drive.normalized
    if drive is None:
        return _decision(
            domain,
            slot,
            "CATEGORY_RULE_MATCH",
            class_refs,
            rule=f"{domain}_APPLICATION_CLASS_ONLY_VNEXT",
            matched_on=matched,
            missing=("drive_architecture",),
            reason="Application class is mapped, but drive is missing or outside the approved mapping.",
        )

    exact = tuple(ref for ref in class_refs if _reference_drive(ref) == drive)
    matched_drive = matched + (f"drive_architecture={drive}",)
    if len(exact) == 1:
        return _decision(
            domain,
            slot,
            "STRONG_RULE_MATCH",
            exact,
            rule=f"{domain}_APPLICATION_CLASS_DRIVE_VNEXT",
            matched_on=matched_drive,
            reason="A unique synthetic reference matches application class, drive, and slot where applicable.",
        )
    if len(exact) > 1:
        return _decision(
            domain,
            slot,
            "AMBIGUOUS_RULE_MATCH",
            exact,
            rule=f"{domain}_DUPLICATE_REFERENCE_VNEXT",
            matched_on=matched_drive,
            reason="More than one synthetic reference matches the same approved dimensions.",
        )
    return _decision(
        domain,
        slot,
        "NO_MATCH",
        (),
        rule=f"{domain}_DRIVE_CATALOG_GAP_VNEXT",
        matched_on=matched_drive,
        reason="The known drive architecture conflicts with the available class references.",
    )


def match_component_priors_vnext(
    identity: NormalizedTechnicalIdentity,
    reference_catalog: Iterable[ReferencePrior],
) -> list[PriorDecision]:
    """Return deterministic decisions, expanding AXLE/HUB into physical slots."""
    references = tuple(
        sorted(
            reference_catalog,
            key=lambda ref: (str(ref.boundary), ref.component_id, ref.component_resolution_id),
        )
    )
    decisions: list[PriorDecision] = []
    for domain in MATCH_BOUNDARIES:
        slots: tuple[str | None, ...] = COMPONENT_SLOTS if domain in SLOTTED_BOUNDARIES else (None,)
        decisions.extend(_match_one(identity, references, domain, slot) for slot in slots)
    return decisions
