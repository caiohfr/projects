"""Streamlit-free state and orchestration for the Sprint 11D Powertrain UX.

This module deliberately owns no physical calculation.  It turns UI drafts
and already-persisted VDE snapshots into the canonical Sprint 11 contracts,
then delegates each scenario independently to ``run_system_scenario``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from src.vde_core.fuel_estimation import FuelEstimateRequest
from src.vde_core.pwt_fuel_energy_service import load_json_blob
from src.vde_core.system_scenario import (
    ALL_DOMAIN_KINDS,
    ArchitectureClass,
    ArchitectureConfiguration,
    DomainCorrection,
    DomainApplicability,
    DomainKind,
    DomainProposal,
    DomainProposalIdentity,
    DomainSourceState,
    EffectiveDomainState,
    ProvenanceKind,
    SolverReadiness,
    SystemScenarioDefinition,
    SystemScenarioIdentity,
    SystemScenarioResult,
    SystemScenarioRole,
    architecture_domain_state_from_legacy_vde_row,
    aux_thermal_domain_state_from_legacy_row,
    controls_domain_state_from_legacy_row,
    domain_applicability_for,
    electric_drive_domain_state_sparse,
    energy_storage_domain_state_from_legacy_row,
    engine_domain_state_from_legacy_row,
    resolve_domain_proposal,
    resolve_effective_domain_state,
    run_system_scenario,
    to_serializable,
    transmission_domain_state_from_legacy_row,
    vehicle_demand_domain_state_from_snapshot_row,
)
from src.vde_core.technology_delta import TechDeltaAssumption


CURRENT_SELECTION = "CURRENT"
NOT_APPLICABLE_SELECTION = "NOT_APPLICABLE"
MAX_PROPOSALS = 3

EDITABLE_DOMAINS: tuple[DomainKind, ...] = tuple(
    domain
    for domain in ALL_DOMAIN_KINDS
    if domain not in (DomainKind.VEHICLE_DEMAND, DomainKind.ARCHITECTURE)
)

MANUAL_EVIDENCE_SOURCE = "Engineering assumption"

DEMAND_DRIVEN = "DEMAND-DRIVEN"
POWERTRAIN_DRIVEN = "POWERTRAIN-DRIVEN"
MIXED_DRIVER = "MIXED DEMAND + POWERTRAIN"
NO_QUANTITATIVE_CHANGE = "NO QUANTITATIVE CHANGE"


@dataclass(frozen=True)
class ScenarioSource:
    """One already-existing VDE snapshot and its optional Powertrain row."""

    vde_id: int
    vde_row: Mapping[str, Any]
    fuelcons_row: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vde_row", MappingProxyType(dict(self.vde_row)))
        object.__setattr__(self, "fuelcons_row", MappingProxyType(dict(self.fuelcons_row)))


@dataclass(frozen=True)
class ScenarioDraft:
    identity: SystemScenarioIdentity
    label: str
    vde_id: int
    architecture: ArchitectureClass
    fuelcons_id: int | None = None
    selections: Mapping[DomainKind, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "selections", MappingProxyType(dict(self.selections)))

    def selection_for(self, domain: DomainKind) -> str:
        return self.selections.get(domain, CURRENT_SELECTION)


@dataclass(frozen=True)
class ScenarioCalculation:
    scenario_id: str
    fingerprint: str
    result: SystemScenarioResult | None = None
    programming_error: str | None = None

    @property
    def readiness(self) -> SolverReadiness:
        if self.result is None:
            return SolverReadiness.NOT_READY
        return self.result.readiness


@dataclass(frozen=True)
class ScenarioResultCard:
    label: str
    scenario_id: str
    fuel_l_100km: float | None
    energy_wh_km: float | None
    pse: float | None
    gco2_km: float | None
    deltas: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "deltas", MappingProxyType(dict(self.deltas)))


@dataclass(frozen=True)
class VehicleDemandComparison:
    current_vde_id: int
    proposal_vde_id: int
    current_mj_per_km: float | None
    proposal_mj_per_km: float | None
    basis: str

    @property
    def changed(self) -> bool:
        if self.current_mj_per_km is not None and self.proposal_mj_per_km is not None:
            return self.current_mj_per_km != self.proposal_mj_per_km
        return self.current_vde_id != self.proposal_vde_id

    @property
    def delta_percent(self) -> float | None:
        if self.current_mj_per_km in (None, 0) or self.proposal_mj_per_km is None:
            return None
        return (self.proposal_mj_per_km / self.current_mj_per_km - 1.0) * 100.0


def current_draft(
    vde_id: int,
    architecture: ArchitectureClass,
    *,
    fuelcons_id: int | None = None,
) -> ScenarioDraft:
    return ScenarioDraft(
        identity=SystemScenarioIdentity("SYS-CURRENT", SystemScenarioRole.CURRENT),
        label="Current",
        vde_id=int(vde_id),
        architecture=architecture,
        fuelcons_id=fuelcons_id,
        selections={domain: CURRENT_SELECTION for domain in EDITABLE_DOMAINS},
    )


def add_proposal_draft(
    drafts: Sequence[ScenarioDraft],
    *,
    vde_id: int,
    architecture: ArchitectureClass,
    fuelcons_id: int | None = None,
) -> tuple[ScenarioDraft, ...]:
    drafts = tuple(drafts)
    used = {
        draft.identity.proposal_index
        for draft in drafts
        if draft.identity.role is SystemScenarioRole.PROPOSAL
    }
    available = next((index for index in range(1, MAX_PROPOSALS + 1) if index not in used), None)
    if available is None:
        raise ValueError("Current + at most 3 Proposals are supported.")
    proposal = ScenarioDraft(
        identity=SystemScenarioIdentity(
            scenario_id=f"SYS-P{available}",
            role=SystemScenarioRole.PROPOSAL,
            proposal_index=available,
        ),
        label=f"Proposal {chr(64 + available)}",
        vde_id=int(vde_id),
        architecture=architecture,
        fuelcons_id=fuelcons_id,
        selections={domain: CURRENT_SELECTION for domain in EDITABLE_DOMAINS},
    )
    return (*drafts, proposal)


def replace_draft(drafts: Sequence[ScenarioDraft], updated: ScenarioDraft) -> tuple[ScenarioDraft, ...]:
    found = False
    output: list[ScenarioDraft] = []
    for draft in drafts:
        if draft.identity.scenario_id == updated.identity.scenario_id:
            output.append(updated)
            found = True
        else:
            output.append(draft)
    if not found:
        raise KeyError(updated.identity.scenario_id)
    return tuple(output)


def remove_proposal_draft(
    drafts: Sequence[ScenarioDraft], scenario_id: str
) -> tuple[ScenarioDraft, ...]:
    selected = next((draft for draft in drafts if draft.identity.scenario_id == scenario_id), None)
    if selected is None:
        return tuple(drafts)
    if selected.identity.role is SystemScenarioRole.CURRENT:
        raise ValueError("Current cannot be removed.")
    return tuple(draft for draft in drafts if draft.identity.scenario_id != scenario_id)


def update_selection(draft: ScenarioDraft, domain: DomainKind, selection: str) -> ScenarioDraft:
    selections = dict(draft.selections)
    selections[domain] = selection
    return replace(draft, selections=selections)


def _effective(
    source: DomainSourceState,
    correction: DomainCorrection | None = None,
) -> EffectiveDomainState:
    return resolve_effective_domain_state(source, correction)


def correction_key(vde_id: int, domain: DomainKind) -> tuple[int, DomainKind]:
    return (int(vde_id), domain)


def effective_states_for_source(
    source: ScenarioSource,
    *,
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection] | None = None,
) -> Mapping[DomainKind, EffectiveDomainState]:
    corrections = corrections or {}
    vde_row = source.vde_row
    fuel_row = source.fuelcons_row
    sources = {
        DomainKind.VEHICLE_DEMAND: vehicle_demand_domain_state_from_snapshot_row(
            vde_row, source_identity=f"vde:{source.vde_id}"
        ),
        DomainKind.ARCHITECTURE: architecture_domain_state_from_legacy_vde_row(vde_row, fuel_row),
        DomainKind.ENGINE_FUEL_CONVERTER: engine_domain_state_from_legacy_row(vde_row, fuel_row),
        DomainKind.TRANSMISSION_DRIVELINE: transmission_domain_state_from_legacy_row(vde_row, fuel_row),
        DomainKind.ELECTRIC_DRIVE: electric_drive_domain_state_sparse(),
        DomainKind.ENERGY_STORAGE: energy_storage_domain_state_from_legacy_row(fuel_row),
        DomainKind.ENERGY_MANAGEMENT_CONTROLS: controls_domain_state_from_legacy_row(fuel_row),
        DomainKind.AUX_THERMAL: aux_thermal_domain_state_from_legacy_row(fuel_row),
    }
    states = {
        domain: _effective(
            domain_source,
            corrections.get(correction_key(source.vde_id, domain)),
        )
        for domain, domain_source in sources.items()
    }
    return MappingProxyType(states)


def current_correction_from_editor(
    *,
    based_on: EffectiveDomainState,
    requested_changes: Mapping[str, Any] | None = None,
    l0_assumption_key: str | None = None,
    l0_assumption_value: float | None = None,
    reason: str = "",
) -> DomainCorrection:
    """Create an explicit Current correction without creating a Proposal."""

    configuration = dataclasses.replace(based_on.configuration, **dict(requested_changes or {}))
    l0_assumptions: dict[str, float] = {}
    if l0_assumption_key and l0_assumption_value is not None:
        l0_assumptions[l0_assumption_key] = float(l0_assumption_value)
    return DomainCorrection(
        domain=based_on.domain,
        configuration=configuration,
        reason=reason,
        l0_effective_assumption=l0_assumptions,
    )


def proposal_from_editor(
    *,
    proposal_id: str,
    domain: DomainKind,
    based_on: EffectiveDomainState,
    label: str,
    requested_changes: Mapping[str, Any] | None = None,
    recommendation_key: str | None = None,
    recommendation_value: float | None = None,
    recommendation_source: str = MANUAL_EVIDENCE_SOURCE,
    evidence_reference: str = "",
    adopted: bool = False,
    technology_deltas: Sequence[TechDeltaAssumption] = (),
) -> DomainProposal:
    """Create an independent proposal; advisory values enter L0 only when adopted."""

    l0_assumptions: dict[str, float] = {}
    l0_provenance: dict[str, ProvenanceKind] = {}
    if adopted and recommendation_source != MANUAL_EVIDENCE_SOURCE:
        raise ValueError(
            "ML, Benchmark and Regression provenance require a canonical evidence result; "
            "a manually entered value is an engineering assumption."
        )
    if adopted and recommendation_key and recommendation_value is not None:
        l0_assumptions[recommendation_key] = float(recommendation_value)
        l0_provenance[recommendation_key] = ProvenanceKind.ASSUMED
    return resolve_domain_proposal(
        DomainProposalIdentity(domain=domain, proposal_id=proposal_id),
        based_on,
        requested_changes=requested_changes,
        label=label,
        l0_effective_assumption=l0_assumptions,
        l0_assumption_provenance=l0_provenance,
        technology_deltas=technology_deltas,
        notes=(
            "Adopted L0 evidence source: "
            f"{MANUAL_EVIDENCE_SOURCE}; reference: {evidence_reference or 'not provided'}"
            if adopted and recommendation_key and recommendation_value is not None
            else "No adopted L0 recommendation."
        ),
    )


def request_template_for_source(source: ScenarioSource) -> FuelEstimateRequest:
    row = dict(source.fuelcons_row)
    assumptions = load_json_blob(row.get("assumptions_json"))
    powertrain: dict[str, Any] = {}
    for key in (
        "fuel_type",
        "eta_pt_est",
        "bev_eff_drive",
        "utility_factor",
        "grid_gco2_per_kwh",
        "LHV_MJ_per_L",
        "gCO2_per_L",
    ):
        value = row.get(key)
        if value is None:
            value = assumptions.get(key)
        if value is not None:
            powertrain[key] = value
    basis = str(row.get("energy_basis") or "VDE_TOTAL").upper()
    return FuelEstimateRequest(
        vde_id=source.vde_id,
        energy_basis=basis,
        cycle=source.vde_row.get("cycle_name"),
        powertrain_features=powertrain,
        method="physics_simple",
    )


def build_definition(
    draft: ScenarioDraft,
    *,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection] | None = None,
) -> tuple[SystemScenarioDefinition, FuelEstimateRequest]:
    source = sources.get(draft.vde_id)
    if source is None:
        # A partially specified Vehicle Demand remains an ordinary unresolved
        # scenario, not a DB query or UI exception during solving.
        source = ScenarioSource(vde_id=draft.vde_id, vde_row={"id": draft.vde_id})
    effective = effective_states_for_source(source, corrections=corrections)
    architecture = _effective(
        DomainSourceState(
            domain=DomainKind.ARCHITECTURE,
            configuration=ArchitectureConfiguration(architecture_class=draft.architecture),
            provenance=ProvenanceKind.ASSUMED,
        )
    )
    slots: dict[DomainKind, EffectiveDomainState | DomainProposal] = {
        DomainKind.VEHICLE_DEMAND: effective[DomainKind.VEHICLE_DEMAND],
        DomainKind.ARCHITECTURE: architecture,
    }
    for domain in EDITABLE_DOMAINS:
        selection = draft.selection_for(domain)
        applicability = domain_applicability_for(draft.architecture, domain)
        if applicability is DomainApplicability.NOT_APPLICABLE:
            # N/A is represented by absence.  An explicitly retained proposal
            # is kept so the canonical resolver can report incompatibility.
            if selection not in (CURRENT_SELECTION, NOT_APPLICABLE_SELECTION):
                proposal = proposals.get(selection)
                if proposal is not None:
                    slots[domain] = proposal
            continue
        if selection == NOT_APPLICABLE_SELECTION:
            continue
        if selection == CURRENT_SELECTION:
            slots[domain] = effective[domain]
            continue
        proposal = proposals.get(selection)
        if proposal is not None:
            slots[domain] = proposal
        else:
            slots[domain] = effective[domain]
    return (
        SystemScenarioDefinition(identity=draft.identity, slots=slots, label=draft.label),
        request_template_for_source(source),
    )


def calculation_fingerprint(
    draft: ScenarioDraft,
    *,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection] | None = None,
) -> str:
    definition, request = build_definition(
        draft,
        sources=sources,
        proposals=proposals,
        corrections=corrections,
    )
    # Visible labels are intentionally excluded: identity and physical input
    # determine staleness, not presentation text.
    payload = {
        "identity": to_serializable(definition.identity),
        "slots": to_serializable(definition.slots),
        "request": request.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def calculate_drafts(
    drafts: Sequence[ScenarioDraft],
    *,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection] | None = None,
) -> Mapping[str, ScenarioCalculation]:
    """Calculate independently so one programming failure cannot poison siblings."""

    calculations: dict[str, ScenarioCalculation] = {}
    for draft in drafts:
        fingerprint = calculation_fingerprint(
            draft,
            sources=sources,
            proposals=proposals,
            corrections=corrections,
        )
        try:
            definition, request = build_definition(
                draft,
                sources=sources,
                proposals=proposals,
                corrections=corrections,
            )
            result = run_system_scenario(definition, request_template=request)
            calculations[draft.identity.scenario_id] = ScenarioCalculation(
                scenario_id=draft.identity.scenario_id,
                fingerprint=fingerprint,
                result=result,
            )
        except Exception as exc:  # schema/programming corruption remains inspectable per slot
            calculations[draft.identity.scenario_id] = ScenarioCalculation(
                scenario_id=draft.identity.scenario_id,
                fingerprint=fingerprint,
                programming_error=f"{type(exc).__name__}: {exc}",
            )
    return MappingProxyType(calculations)


def is_stale(
    draft: ScenarioDraft,
    calculation: ScenarioCalculation | None,
    *,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection] | None = None,
) -> bool:
    if calculation is None:
        return False
    return calculation.fingerprint != calculation_fingerprint(
        draft, sources=sources, proposals=proposals, corrections=corrections
    )


def explainability_rows(
    draft: ScenarioDraft,
    *,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection] | None = None,
) -> tuple[Mapping[str, str], ...]:
    """Describe scenario composition without assigning fabricated subsystem physics."""

    corrections = corrections or {}
    source = sources.get(draft.vde_id)
    if source is None:
        return ()
    effective = effective_states_for_source(source, corrections=corrections)
    rows: list[Mapping[str, str]] = []
    for domain in EDITABLE_DOMAINS:
        selection = draft.selection_for(domain)
        correction = corrections.get(correction_key(draft.vde_id, domain))
        if selection == CURRENT_SELECTION:
            status = "Current correction only" if correction is not None else "Not represented"
            rows.append(
                MappingProxyType(
                    {
                        "domain": domain.value,
                        "config_change": "Inherited from Effective Current",
                        "representation": "Current correction" if correction is not None else "None",
                        "provenance": "CURRENT_CORRECTION" if correction is not None else "-",
                        "status": status,
                    }
                )
            )
            continue
        if selection == NOT_APPLICABLE_SELECTION:
            rows.append(
                MappingProxyType(
                    {
                        "domain": domain.value,
                        "config_change": "Not applicable",
                        "representation": "None",
                        "provenance": "-",
                        "status": "Not represented",
                    }
                )
            )
            continue
        proposal = proposals.get(selection)
        if proposal is None:
            continue
        changed = proposal.configuration != effective[domain].configuration
        represented = bool(proposal.l0_effective_assumption or proposal.technology_deltas)
        if represented:
            status = "Quantitative impact adopted"
            representation = "Direct L0 assumption" if proposal.l0_effective_assumption else "Technology Delta"
            provenance = ", ".join(
                value.value for value in proposal.l0_assumption_provenance.values()
            ) or "Technology Delta"
        elif changed:
            status = "Configuration only"
            representation = "None"
            provenance = "-"
        else:
            status = "Not represented"
            representation = "None"
            provenance = "-"
        rows.append(
            MappingProxyType(
                {
                    "domain": domain.value,
                    "config_change": "Changed" if changed else "Unchanged",
                    "representation": representation,
                    "provenance": provenance,
                    "status": status,
                }
            )
        )
    return tuple(rows)


def sequential_impact_trace(
    draft: ScenarioDraft,
    *,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
    corrections: Mapping[tuple[int, DomainKind], DomainCorrection] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Run canonical L0 after each adopted domain in deterministic domain order."""

    corrections = corrections or {}
    selections = {domain: CURRENT_SELECTION for domain in EDITABLE_DOMAINS}
    base_draft = replace(draft, selections=selections)
    steps: list[Mapping[str, Any]] = []
    for label, candidate in [("Base Effective Current", base_draft)]:
        definition, request = build_definition(
            candidate, sources=sources, proposals=proposals, corrections=corrections
        )
        result = run_system_scenario(definition, request_template=request)
        steps.append(MappingProxyType({"label": label, "outputs": dict(result.effective_outputs)}))
    for domain in EDITABLE_DOMAINS:
        selection = draft.selection_for(domain)
        proposal = proposals.get(selection)
        if proposal is None or not (proposal.l0_effective_assumption or proposal.technology_deltas):
            continue
        selections[domain] = selection
        candidate = replace(draft, selections=selections)
        definition, request = build_definition(
            candidate, sources=sources, proposals=proposals, corrections=corrections
        )
        result = run_system_scenario(definition, request_template=request)
        steps.append(
            MappingProxyType(
                {"label": f"{domain.value}: adopted L0 impact", "outputs": dict(result.effective_outputs)}
            )
        )
    return tuple(steps)


def result_deltas(
    current: SystemScenarioResult | None,
    candidate: SystemScenarioResult | None,
) -> Mapping[str, float]:
    """Compare already-calculated canonical outputs; this owns no physics."""

    if current is None or candidate is None:
        return MappingProxyType({})
    baseline = current.effective_outputs
    proposed = candidate.effective_outputs
    deltas: dict[str, float] = {}
    for key in ("fuel_l_100km", "energy_Wh_km", "gco2_km", "pse"):
        before = baseline.get(key)
        after = proposed.get(key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            deltas[key] = float(after) - float(before)
    return MappingProxyType(deltas)


def result_card_viewmodel(
    draft: ScenarioDraft,
    calculation: ScenarioCalculation | None,
    *,
    current: ScenarioCalculation | None = None,
) -> ScenarioResultCard:
    result = calculation.result if calculation is not None else None
    outputs = result.effective_outputs if result is not None else {}
    current_result = current.result if current is not None else None
    return ScenarioResultCard(
        label=draft.label,
        scenario_id=draft.identity.scenario_id,
        fuel_l_100km=outputs.get("fuel_l_100km"),
        energy_wh_km=outputs.get("energy_Wh_km"),
        pse=outputs.get("pse"),
        gco2_km=outputs.get("gco2_km"),
        deltas=result_deltas(current_result, result),
    )


def _vehicle_demand_value(source: ScenarioSource | None, basis: str) -> float | None:
    if source is None:
        return None
    state = effective_states_for_source(source)[DomainKind.VEHICLE_DEMAND]
    result = state.configuration.vehicle_demand_result
    if result is None:
        return None
    summary = result.net_summary if basis == "NET" else result.total_summary
    return summary.vde_mj_per_km if summary is not None else None


def vehicle_demand_comparison(
    current: ScenarioDraft,
    proposal: ScenarioDraft,
    *,
    sources: Mapping[int, ScenarioSource],
) -> VehicleDemandComparison:
    source = sources.get(current.vde_id)
    basis = str((source.fuelcons_row if source is not None else {}).get("energy_basis") or "VDE_TOTAL")
    basis = "NET" if basis.upper() == "VDE_NET" else "TOTAL"
    return VehicleDemandComparison(
        current_vde_id=current.vde_id,
        proposal_vde_id=proposal.vde_id,
        current_mj_per_km=_vehicle_demand_value(sources.get(current.vde_id), basis),
        proposal_mj_per_km=_vehicle_demand_value(sources.get(proposal.vde_id), basis),
        basis=basis,
    )


def compact_impact_rows(
    draft: ScenarioDraft,
    *,
    proposals: Mapping[str, DomainProposal],
) -> tuple[Mapping[str, str], ...]:
    """Return changed proposal domains only; inherited rows stay technical."""

    rows: list[Mapping[str, str]] = []
    for domain in EDITABLE_DOMAINS:
        selection = draft.selection_for(domain)
        proposal = proposals.get(selection)
        if proposal is None:
            continue
        represented = bool(proposal.l0_effective_assumption or proposal.technology_deltas)
        changed = bool(proposal.requested_changes)
        if not represented and not changed:
            continue
        if proposal.l0_effective_assumption:
            key, value = next(iter(proposal.l0_effective_assumption.items()))
            representation = _friendly_l0_assumption(key, value)
            evidence = ", ".join(
                item.value for item in proposal.l0_assumption_provenance.values()
            ) or "ASSUMED"
            status = "ADOPTED"
        elif proposal.technology_deltas:
            delta = proposal.technology_deltas[0]
            representation = f"{delta.effect_value:+g}% {delta.effect_basis.replace('_percent_delta', '').upper()}"
            evidence = delta.source_type
            status = "ADOPTED"
        else:
            representation = "Not represented"
            evidence = "—"
            status = "CONFIG ONLY"
        rows.append(
            MappingProxyType(
                {
                    "domain": domain.value,
                    "representation": representation,
                    "evidence": evidence,
                    "status": status,
                }
            )
        )
    return tuple(rows)


def _friendly_l0_assumption(key: str, value: float) -> str:
    if key == "eta_pt_est":
        return f"Aggregate fuel-path efficiency: {value * 100:.2f}%"
    if key == "bev_eff_drive":
        return f"Electric-path efficiency: {value * 100:.2f}%"
    if key == "utility_factor":
        return f"Utility factor: {value * 100:.1f}%"
    if key == "grid_gco2_kwh":
        return f"Grid CO₂ factor: {value:g} g/kWh"
    return f"Effective assumption: {value:g}"


def result_driver(
    current: ScenarioDraft,
    proposal: ScenarioDraft,
    *,
    sources: Mapping[int, ScenarioSource],
    proposals: Mapping[str, DomainProposal],
) -> str:
    demand_changed = vehicle_demand_comparison(current, proposal, sources=sources).changed
    powertrain_changed = any(
        row["status"] == "ADOPTED"
        for row in compact_impact_rows(proposal, proposals=proposals)
    )
    if demand_changed and powertrain_changed:
        return MIXED_DRIVER
    if demand_changed:
        return DEMAND_DRIVEN
    if powertrain_changed:
        return POWERTRAIN_DRIVEN
    return NO_QUANTITATIVE_CHANGE


def metadata_incomplete_fields(definition: SystemScenarioDefinition) -> Mapping[DomainKind, tuple[str, ...]]:
    missing: dict[DomainKind, tuple[str, ...]] = {}
    for domain, selection in definition.slots.items():
        if domain in (DomainKind.VEHICLE_DEMAND, DomainKind.ARCHITECTURE):
            continue
        values = []
        for config_field in dataclasses.fields(selection.configuration):
            if getattr(selection.configuration, config_field.name) is None:
                values.append(config_field.name)
        if values:
            missing[domain] = tuple(values)
    return MappingProxyType(missing)


FRIENDLY_ISSUES: Mapping[str, str] = MappingProxyType(
    {
        "vde_total_mj_per_km_missing": "Selected Vehicle Demand has no TOTAL result.",
        "vde_net_mj_per_km_missing": "Selected Vehicle Demand has no NET result.",
        "architecture_class_missing": "Select an Architecture.",
        "eta_pt_est_missing": "Adopt or provide the required fuel-path effective assumption.",
        "bev_eff_drive_missing": "Adopt or provide the required electric-path effective assumption.",
    }
)


def friendly_issue(issue: str) -> str:
    if issue in FRIENDLY_ISSUES:
        return FRIENDLY_ISSUES[issue]
    if issue.startswith("architecture_domain_incompatible:"):
        return "The selected domain proposal is incompatible with this Architecture."
    if issue.startswith("unsupported_quantitative_representation:"):
        return "This quantitative representation is not supported by Energy Balance L0."
    if issue.startswith("incompatible_technology_delta_basis:"):
        return "This Technology Delta basis cannot enter the canonical L0 stack."
    if issue.startswith("conflicting_l0_assumption:"):
        return "Two selected proposals provide conflicting values for the same L0 assumption."
    return "Scenario input is unresolved; inspect technical details for its structured issue."


__all__ = [
    "CURRENT_SELECTION",
    "NOT_APPLICABLE_SELECTION",
    "EDITABLE_DOMAINS",
    "MANUAL_EVIDENCE_SOURCE",
    "DEMAND_DRIVEN",
    "POWERTRAIN_DRIVEN",
    "MIXED_DRIVER",
    "NO_QUANTITATIVE_CHANGE",
    "ScenarioSource",
    "ScenarioDraft",
    "ScenarioCalculation",
    "ScenarioResultCard",
    "VehicleDemandComparison",
    "current_draft",
    "add_proposal_draft",
    "replace_draft",
    "remove_proposal_draft",
    "update_selection",
    "correction_key",
    "effective_states_for_source",
    "current_correction_from_editor",
    "proposal_from_editor",
    "request_template_for_source",
    "build_definition",
    "calculation_fingerprint",
    "calculate_drafts",
    "is_stale",
    "explainability_rows",
    "sequential_impact_trace",
    "result_deltas",
    "result_card_viewmodel",
    "vehicle_demand_comparison",
    "compact_impact_rows",
    "result_driver",
    "metadata_incomplete_fields",
    "friendly_issue",
]
