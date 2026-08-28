"""Deterministic System Scenario composition for Sprint 11C."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from src.vde_core.fuel_estimation import FuelEstimateRequest
from src.vde_core.technology_delta import (
    TechDeltaAssumption,
    normalize_technology_delta,
    tech_delta_assumption_to_dict,
)

from .contracts import (
    ALL_DOMAIN_KINDS,
    ArchitectureClass,
    ArchitectureConfiguration,
    DomainApplicability,
    DomainKind,
    DomainProposal,
    FidelityLevel,
    FidelityManifest,
    L0AssumptionContribution,
    ProvenanceKind,
    ResolvedSystemScenario,
    SolverReadiness,
    SystemScenarioDefinition,
    SystemScenarioResult,
    TechnologyDeltaContribution,
    VehicleDemandConfiguration,
    domain_applicability_for,
)
from .l0_adapter import (
    EnergyBalanceL0Adapter,
    EnergyBalanceL0RequestSnapshot,
    build_energy_balance_l0_request,
    energy_balance_l0_readiness_issues,
    is_direct_powertrain_assumption,
)


_SUPPORTED_DELTA_EFFECT_BASES = frozenset(
    {
        "pse_delta",
        "pse_percent_delta",
        "pse_multiplier",
        "efficiency_multiplier",
        "fuel_delta",
        "fuel_percent_delta",
        "co2_delta",
        "co2_percent_delta",
        "energy_delta",
    }
)


def _architecture(definition: SystemScenarioDefinition) -> ArchitectureClass | None:
    selection = definition.slots.get(DomainKind.ARCHITECTURE)
    if selection is None or not isinstance(selection.configuration, ArchitectureConfiguration):
        return None
    return selection.configuration.architecture_class


def _synthetic_delta(domain: DomainKind, key: str, value: float) -> TechDeltaAssumption:
    return TechDeltaAssumption(
        name=f"{domain.value} explicit L0 assumption: {key}",
        affected_subsystem=domain.value.lower(),
        effect_basis=key,
        effect_value=float(value),
        source_type="engineering_assumption",
        maturity_level="engineering_assumption",
        confidence="unknown",
        notes="Explicit Domain Proposal L0 representation.",
    )


def _compose(
    definition: SystemScenarioDefinition,
) -> tuple[
    dict[DomainKind, Any],
    FidelityManifest,
    ArchitectureClass | None,
    dict[str, float],
    tuple[TechDeltaAssumption, ...],
    tuple[TechnologyDeltaContribution, ...],
    tuple[L0AssumptionContribution, ...],
    tuple[str, ...],
]:
    architecture = _architecture(definition)
    ordered_domains = {domain: definition.slots[domain] for domain in ALL_DOMAIN_KINDS if domain in definition.slots}
    fidelity: dict[DomainKind, FidelityLevel] = {}
    direct_assumptions: dict[str, float] = {}
    ordered_deltas: list[TechDeltaAssumption] = []
    delta_contributions: list[TechnologyDeltaContribution] = []
    assumption_contributions: list[L0AssumptionContribution] = []
    issues: list[str] = []

    def collect_delta(domain: DomainKind, proposal_id: str, delta: TechDeltaAssumption) -> bool:
        normalized = normalize_technology_delta(tech_delta_assumption_to_dict(delta))
        status = str(normalized["quantitative_status"])
        basis = str(normalized["effect_basis"])
        if status == "applied" and basis in _SUPPORTED_DELTA_EFFECT_BASES:
            ordered_deltas.append(delta)
            delta_contributions.append(
                TechnologyDeltaContribution(
                    evaluation_order=len(delta_contributions) + 1,
                    domain=domain,
                    proposal_id=proposal_id,
                    assumption=delta,
                    quantitative_status=status,
                )
            )
            return True
        if status == "pending_model":
            issues.append(
                f"unsupported_quantitative_representation:{domain.value}:{proposal_id}:{basis}"
            )
        elif status == "applied":
            issues.append(
                f"incompatible_technology_delta_basis:{domain.value}:{proposal_id}:{basis}"
            )
        return False

    for domain in ALL_DOMAIN_KINDS:
        selection = ordered_domains.get(domain)
        if selection is None:
            fidelity[domain] = FidelityLevel.NOT_REPRESENTED
            continue
        if (
            architecture is not None
            and domain_applicability_for(architecture, domain) is DomainApplicability.NOT_APPLICABLE
        ):
            fidelity[domain] = FidelityLevel.NOT_REPRESENTED
            if isinstance(selection, DomainProposal):
                issues.append(
                    f"architecture_domain_incompatible:{architecture.value}:{domain.value}:"
                    f"{selection.identity.proposal_id}"
                )
            continue

        if domain is DomainKind.VEHICLE_DEMAND:
            config = selection.configuration
            has_result = isinstance(config, VehicleDemandConfiguration) and config.vehicle_demand_result is not None
            fidelity[domain] = FidelityLevel.QUANTITATIVE if has_result else FidelityLevel.NOT_REPRESENTED
            continue
        if domain is DomainKind.ARCHITECTURE:
            fidelity[domain] = (
                FidelityLevel.QUANTITATIVE if architecture is not None else FidelityLevel.NOT_REPRESENTED
            )
            continue

        represented = False
        if isinstance(selection, DomainProposal):
            proposal_id = selection.identity.proposal_id
            for key, raw_value in selection.l0_effective_assumption.items():
                value = float(raw_value)
                if is_direct_powertrain_assumption(key):
                    if key in direct_assumptions and direct_assumptions[key] != value:
                        issues.append(f"conflicting_l0_assumption:{key}")
                    else:
                        direct_assumptions[key] = value
                        assumption_contributions.append(
                            L0AssumptionContribution(
                                key=key,
                                value=value,
                                domain=domain,
                                proposal_id=proposal_id,
                                provenance=selection.l0_assumption_provenance.get(
                                    key, ProvenanceKind.ASSUMED
                                ),
                            )
                        )
                    represented = True
                else:
                    delta = _synthetic_delta(domain, key, value)
                    represented = collect_delta(domain, proposal_id, delta) or represented
            for delta in selection.technology_deltas:
                represented = collect_delta(domain, proposal_id, delta) or represented

        if domain is DomainKind.ENGINE_FUEL_CONVERTER and selection.configuration.fuel_type is not None:
            represented = True
            fidelity[domain] = FidelityLevel.QUANTITATIVE
        else:
            fidelity[domain] = FidelityLevel.EFFECTIVE_ASSUMPTION if represented else FidelityLevel.CONFIGURATION_ONLY

    return (
        ordered_domains,
        FidelityManifest(per_domain=fidelity),
        architecture,
        direct_assumptions,
        tuple(ordered_deltas),
        tuple(delta_contributions),
        tuple(assumption_contributions),
        tuple(dict.fromkeys(issues)),
    )


def resolve_system_scenario(
    definition: SystemScenarioDefinition,
    *,
    request_template: FuelEstimateRequest | Mapping[str, Any] | None = None,
) -> ResolvedSystemScenario:
    """Resolve one definition independently of presentation/dict order."""

    (
        domains,
        manifest,
        architecture,
        assumptions,
        deltas,
        delta_contributions,
        assumption_contributions,
        composition_issues,
    ) = _compose(definition)
    provisional = ResolvedSystemScenario(
        identity=definition.identity,
        resolved_domains=domains,
        fidelity_manifest=manifest,
        architecture_class=architecture,
        ordered_technology_deltas=deltas,
        technology_delta_contributions=delta_contributions,
        l0_effective_assumptions=assumptions,
        l0_assumption_contributions=assumption_contributions,
        issues=composition_issues,
    )
    request = build_energy_balance_l0_request(provisional, request_template)
    issues = tuple(dict.fromkeys((*composition_issues, *energy_balance_l0_readiness_issues(request))))
    return replace(
        provisional,
        solver_readiness=SolverReadiness.READY if not issues else SolverReadiness.NOT_READY,
        l0_request_snapshot=EnergyBalanceL0RequestSnapshot.from_request(request),
        issues=issues,
    )


def run_system_scenario(
    definition: SystemScenarioDefinition,
    *,
    request_template: FuelEstimateRequest | Mapping[str, Any] | None = None,
) -> SystemScenarioResult:
    """Resolve and execute one scenario through the canonical L0 owners."""

    return EnergyBalanceL0Adapter().run(
        resolve_system_scenario(definition, request_template=request_template)
    )


def _validate_working_set(definitions: Sequence[SystemScenarioDefinition]) -> None:
    if len(definitions) > 4:
        raise ValueError("A System Scenario working set supports Current + at most 3 Proposals.")
    identity_keys = [
        (definition.identity.role, definition.identity.proposal_index)
        for definition in definitions
    ]
    if len(identity_keys) != len(set(identity_keys)):
        raise ValueError("System Scenario roles/proposal indexes must be unique within one working set.")
    scenario_ids = [definition.identity.scenario_id for definition in definitions]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("System Scenario scenario_id values must be unique within one working set.")


def resolve_system_scenarios(
    definitions: Sequence[SystemScenarioDefinition],
    *,
    request_templates: Mapping[str, FuelEstimateRequest | Mapping[str, Any]] | None = None,
) -> tuple[ResolvedSystemScenario, ...]:
    """Resolve the bounded Current + 3 working set without cross-inheritance."""

    definitions = tuple(definitions)
    _validate_working_set(definitions)
    templates = request_templates or {}
    return tuple(
        resolve_system_scenario(
            definition,
            request_template=templates.get(definition.identity.scenario_id),
        )
        for definition in definitions
    )


def run_system_scenarios(
    definitions: Sequence[SystemScenarioDefinition],
    *,
    request_templates: Mapping[str, FuelEstimateRequest | Mapping[str, Any]] | None = None,
) -> tuple[SystemScenarioResult, ...]:
    """Resolve and execute a bounded working set, each scenario independently."""

    return tuple(
        EnergyBalanceL0Adapter().run(resolved)
        for resolved in resolve_system_scenarios(definitions, request_templates=request_templates)
    )


__all__ = [
    "resolve_system_scenario",
    "resolve_system_scenarios",
    "run_system_scenario",
    "run_system_scenarios",
]
