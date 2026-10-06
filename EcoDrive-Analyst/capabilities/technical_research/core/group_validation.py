from __future__ import annotations

from collections import defaultdict
import re
from typing import Iterable

from ..contracts import (
    HardwareGroupEvaluation,
    HardwareGroupMember,
    HardwareGroupStatus,
    IdentityConfidence,
)


def evaluate_hardware_group(
    members: Iterable[HardwareGroupMember],
) -> HardwareGroupEvaluation:
    rows = tuple(members)
    independent: dict[str, list[HardwareGroupMember]] = defaultdict(list)
    for member in rows:
        independent[member.independent_application_id].append(member)

    supported = [
        member
        for member in rows
        if member.hardware_identity
        and member.identity_confidence in {IdentityConfidence.DIRECT, IdentityConfidence.STRONG}
        and not member.material_hardware_conflict
    ]
    identities = tuple(sorted({str(member.hardware_identity) for member in supported}))
    resolved_independent = {
        member.independent_application_id for member in supported
    }
    all_hardware = {
        str(member.hardware_identity)
        for member in rows
        if member.hardware_identity
        and member.identity_confidence in {IdentityConfidence.DIRECT, IdentityConfidence.STRONG}
    }
    descriptions_by_application: dict[str, set[str]] = defaultdict(set)
    for member in rows:
        if member.marketing_description:
            descriptions_by_application[member.independent_application_id].add(
                re.sub(
                    r"[^A-Z0-9]+",
                    " ",
                    member.marketing_description.upper(),
                ).strip()
            )
    repeated_descriptions = defaultdict(set)
    for application_id, descriptions in descriptions_by_application.items():
        for description in descriptions:
            repeated_descriptions[description].add(application_id)
    descriptive_only = bool(descriptions_by_application) and not all_hardware and (
        len(independent) == 1
        or any(len(application_ids) >= 2 for application_ids in repeated_descriptions.values())
    )

    if len(all_hardware) >= 2:
        status = HardwareGroupStatus.HARDWARE_SPLIT
        notes = "MULTIPLE_EXTERNALLY_SUPPORTED_HARDWARE_IDENTITIES"
    elif (
        len(identities) == 1
        and len(resolved_independent) >= 2
        and not any(member.material_hardware_conflict for member in supported)
    ):
        status = HardwareGroupStatus.HARDWARE_CONFIRMED
        notes = "REPEATED_HARDWARE_ACROSS_INDEPENDENT_APPLICATIONS"
    elif supported:
        status = HardwareGroupStatus.PARTIALLY_RESOLVED
        notes = "USEFUL_HARDWARE_EVIDENCE_WITH_UNRESOLVED_APPLICATIONS"
    elif descriptive_only:
        status = HardwareGroupStatus.DESCRIPTIVE_ONLY
        notes = "SHARED_DESCRIPTION_WITHOUT_SUPPORTED_HARDWARE_IDENTITY"
    else:
        status = HardwareGroupStatus.UNRESOLVED
        notes = "INSUFFICIENT_EXTERNAL_HARDWARE_EVIDENCE"

    return HardwareGroupEvaluation(
        status=status,
        n_rows=len(rows),
        n_independent_applications=len(independent),
        hardware_identities=tuple(sorted(all_hardware)),
        identity_confidences=tuple(
            sorted({member.identity_confidence.value for member in rows})
        ),
        supporting_source_count=len(
            {source_id for member in supported for source_id in member.source_ids}
        ),
        notes=notes,
    )
