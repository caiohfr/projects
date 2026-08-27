# src/vde_core/system_scenario/domain_resolution.py
# -----------------------------------------------------------------------------
# Sprint 11B - the "domain resolution service" layer sitting between the
# canonical Domain State/Proposal contracts (contracts.py) and any future
# caller (a UI, or Sprint 11C's System Scenario resolver) -- see the Sprint
# 11B architecture boundary: legacy adapters -> canonical DomainSourceState
# -> this module -> EffectiveDomainState/DomainProposal. Streamlit-
# independent, no DB access, no physics.
#
# `resolve_domain_proposal` only ever constructs a new typed configuration
# via `dataclasses.replace()` on top of an EffectiveDomainState's own
# configuration -- "unrequested fields inherit from Effective Current"
# falls out of `dataclasses.replace` itself, not a bespoke merge/patch
# engine (Sec 18: no generic dict[str, Any] plugin-like change engine).
# Passing a field name that does not exist on that domain's configuration
# type raises `TypeError` immediately (dataclasses.replace's own
# behavior) -- a deliberate fail-loud, never a silently-ignored typo.
# -----------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from src.vde_core.quick_scenario.contracts import TechDeltaAssumption

from .contracts import DomainProposal, DomainProposalIdentity, EffectiveDomainState


def resolve_domain_proposal(
    identity: DomainProposalIdentity,
    based_on: EffectiveDomainState,
    requested_changes: Mapping[str, Any] | None = None,
    *,
    label: str | None = None,
    l0_effective_assumption: Mapping[str, float] | None = None,
    technology_deltas: Sequence[TechDeltaAssumption] = (),
    notes: str = "",
) -> DomainProposal:
    """Sec 16/18: resolve one Domain Proposal from `based_on` (that
    domain's Effective Current) plus an explicit set of field overrides.

    Every field NOT named in `requested_changes` inherits verbatim from
    `based_on.configuration`. An explicit `0`/`0.0`/`False` passed in
    `requested_changes` is applied exactly as given -- it is never coerced
    to "missing" or silently dropped (Sec 18/22: explicit zero stays
    explicit); a field absent from `requested_changes` that was already
    `None` on Effective Current stays `None` (Sec 18: missing stays
    missing). `identity.domain` must equal `based_on.domain` -- enforced by
    `DomainProposal.__post_init__` itself (not duplicated here), so a
    domain mismatch raises the same `ValueError` it always has.

    `l0_effective_assumption`/`technology_deltas` are passed straight
    through, unmodified and uncombined (Sec 19/20: this function never
    infers a quantitative effect from `requested_changes`, and never
    stacks the supplied Technology Deltas -- that composition step is
    Sprint 11C's job once a deterministic cross-domain order exists).
    """

    requested_changes = dict(requested_changes or {})
    configuration = dataclasses.replace(based_on.configuration, **requested_changes)
    return DomainProposal(
        identity=identity,
        domain=identity.domain,
        configuration=configuration,
        based_on=based_on,
        label=label,
        l0_effective_assumption=dict(l0_effective_assumption or {}),
        technology_deltas=tuple(technology_deltas),
        requested_changes=requested_changes,
        notes=notes,
    )


def changed_fields(proposal: DomainProposal) -> dict[str, tuple[Any, Any]]:
    """Sec 23: Streamlit-free diff support -- returns
    `{field_name: (effective_current_value, proposal_value)}` for every
    field that actually differs between `proposal.based_on.configuration`
    and `proposal.configuration`. Computed by comparing the two
    configuration objects directly (never by trusting
    `proposal.requested_changes`), so it stays correct even for a
    `DomainProposal` built without going through `resolve_domain_proposal`
    above. This is presentation/support functionality only -- it never
    computes or implies a physical consequence, and it is not used to
    build any UI in Sprint 11B.
    """

    effective_config = proposal.based_on.configuration
    proposal_config = proposal.configuration
    diffs: dict[str, tuple[Any, Any]] = {}
    for config_field in dataclasses.fields(effective_config):
        old_value = getattr(effective_config, config_field.name)
        new_value = getattr(proposal_config, config_field.name)
        if old_value != new_value:
            diffs[config_field.name] = (old_value, new_value)
    return diffs


__all__ = ["resolve_domain_proposal", "changed_fields"]
