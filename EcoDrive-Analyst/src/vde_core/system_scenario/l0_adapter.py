"""Energy Balance L0 adapter for Sprint 11C.

This module maps a resolved System Scenario onto the existing
``FuelEstimateRequest`` contract.  It owns no physical formula: fuel/energy
calculation is delegated to ``run_fuel_estimation`` and Technology Delta
composition is delegated once to ``apply_delta_stack_to_baseline``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from src.vde_core.fuel_estimation import ENGINE_VERSION, FuelEstimateRequest, run_fuel_estimation
from src.vde_core.technology_delta import (
    apply_delta_stack_to_baseline,
    normalize_technology_delta,
    tech_delta_assumption_to_dict,
)

from .contracts import (
    ArchitectureConfiguration,
    DomainKind,
    DomainProposal,
    EngineConfiguration,
    ProvenanceKind,
    ResolvedSystemScenario,
    SolverReadiness,
    SystemScenarioResult,
    VehicleDemandConfiguration,
)


_DIRECT_POWERTRAIN_ASSUMPTION_KEYS = frozenset(
    {
        "eta_pt_est",
        "bev_eff_drive",
        "utility_factor",
        "grid_gco2_per_kwh",
        "LHV_MJ_per_L",
        "gCO2_per_L",
    }
)


@dataclass(frozen=True)
class EnergyBalanceL0RequestSnapshot:
    """Immutable translation output stored by ``ResolvedSystemScenario``."""

    vde_id: int | None
    energy_basis: str
    cycle: str | None
    vehicle_features: Mapping[str, Any]
    powertrain_features: Mapping[str, Any]
    method: str
    model_options: Mapping[str, Any]
    manual_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("vehicle_features", "powertrain_features", "model_options", "manual_inputs"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @classmethod
    def from_request(cls, request: FuelEstimateRequest) -> "EnergyBalanceL0RequestSnapshot":
        return cls(**request.to_dict())

    def to_request(self) -> FuelEstimateRequest:
        return FuelEstimateRequest(
            vde_id=self.vde_id,
            energy_basis=self.energy_basis,
            cycle=self.cycle,
            vehicle_features=dict(self.vehicle_features),
            powertrain_features=dict(self.powertrain_features),
            method=self.method,
            model_options=dict(self.model_options),
            manual_inputs=dict(self.manual_inputs),
        )


def is_direct_powertrain_assumption(key: str) -> bool:
    """Whether a System L0 assumption maps to an existing request key."""

    return key in _DIRECT_POWERTRAIN_ASSUMPTION_KEYS


def _copy_request(template: FuelEstimateRequest | Mapping[str, Any] | None) -> FuelEstimateRequest:
    if template is None:
        return FuelEstimateRequest()
    if isinstance(template, FuelEstimateRequest):
        data = template.to_dict()
    else:
        data = dict(template)
    return FuelEstimateRequest(
        vde_id=data.get("vde_id"),
        energy_basis=str(data.get("energy_basis") or "VDE_TOTAL"),
        cycle=data.get("cycle"),
        vehicle_features=dict(data.get("vehicle_features") or {}),
        powertrain_features=dict(data.get("powertrain_features") or {}),
        method=str(data.get("method") or "physics_simple"),
        model_options=dict(data.get("model_options") or {}),
        manual_inputs=dict(data.get("manual_inputs") or {}),
    )


def build_energy_balance_l0_request(
    resolved: ResolvedSystemScenario,
    request_template: FuelEstimateRequest | Mapping[str, Any] | None = None,
) -> FuelEstimateRequest:
    """Build a fresh canonical request from an immutable scenario snapshot.

    The selected Vehicle Demand always replaces any demand values on the
    template, preventing one System Scenario from inheriting another's VDE.
    Unsupported configuration metadata is carried only where the canonical
    request already has an equivalent field; no consumption effect is
    inferred from it.
    """

    request = _copy_request(request_template)
    vehicle_features = dict(request.vehicle_features)
    powertrain_features = dict(request.powertrain_features)

    # Vehicle Demand and Architecture always belong to the scenario.  Clear
    # template values first so a missing selection can never fall back to a
    # different scenario's stale demand/classification.
    request.vde_id = None
    vehicle_features.pop("phase_outputs", None)
    vehicle_features["vde_total_mj_per_km"] = None
    vehicle_features["vde_net_mj_per_km"] = None
    vehicle_features["electrification"] = None

    vehicle_selection = resolved.resolved_domains.get(DomainKind.VEHICLE_DEMAND)
    if vehicle_selection is not None and isinstance(
        vehicle_selection.configuration, VehicleDemandConfiguration
    ):
        config = vehicle_selection.configuration
        result = config.vehicle_demand_result
        if config.source_identity and config.source_identity.startswith("vde:"):
            try:
                request.vde_id = int(config.source_identity.split(":", 1)[1])
            except (TypeError, ValueError):
                request.vde_id = None
        vehicle_features["source_identity"] = config.source_identity
        # A frozen VehicleDemandResult carries whole-cycle TOTAL/NET only.
        # Reusing phase values from another request template would silently
        # mix two VDE sources, so absence remains explicit here.
        vehicle_features["vde_total_mj_per_km"] = (
            result.total_summary.vde_mj_per_km if result is not None else None
        )
        vehicle_features["vde_net_mj_per_km"] = (
            result.net_summary.vde_mj_per_km
            if result is not None and result.net_summary is not None
            else None
        )
        if result is not None and request.cycle is None:
            request.cycle = result.total_summary.cycle_name

    architecture_selection = resolved.resolved_domains.get(DomainKind.ARCHITECTURE)
    if architecture_selection is not None and isinstance(
        architecture_selection.configuration, ArchitectureConfiguration
    ):
        architecture = architecture_selection.configuration.architecture_class
        vehicle_features["electrification"] = architecture.value if architecture is not None else None

    engine_selection = resolved.resolved_domains.get(DomainKind.ENGINE_FUEL_CONVERTER)
    if engine_selection is not None and isinstance(engine_selection.configuration, EngineConfiguration):
        engine = engine_selection.configuration
        if engine.fuel_type is not None:
            if powertrain_features.get("fuel_type") != engine.fuel_type:
                # LHV/CO2 factors belong to the selected fuel. Do not carry
                # explicit factors from a different template fuel; the
                # canonical estimator owns default lookup for the new fuel.
                powertrain_features.pop("LHV_MJ_per_L", None)
                powertrain_features.pop("gCO2_per_L", None)
            powertrain_features["fuel_type"] = engine.fuel_type
        if engine.displacement_l is not None:
            powertrain_features["engine_size_l"] = engine.displacement_l
        if engine.rated_power_kw is not None:
            powertrain_features["engine_max_power_kw"] = engine.rated_power_kw

    for key, value in resolved.l0_effective_assumptions.items():
        if is_direct_powertrain_assumption(key):
            powertrain_features[key] = value

    request.vehicle_features = vehicle_features
    request.powertrain_features = powertrain_features
    return request


def energy_balance_l0_readiness_issues(request: FuelEstimateRequest) -> tuple[str, ...]:
    """Report only fields the selected canonical L0 method actually needs."""

    issues: list[str] = []
    basis = str(request.energy_basis or "VDE_TOTAL").upper()
    demand_key = "vde_net_mj_per_km" if basis == "VDE_NET" else "vde_total_mj_per_km"
    if request.vehicle_features.get(demand_key) is None:
        issues.append(f"{demand_key}_missing")

    architecture = str(request.vehicle_features.get("electrification") or "").upper()
    if not architecture:
        issues.append("architecture_class_missing")

    def _is_positive(value: Any) -> bool:
        try:
            return value is not None and float(value) > 0
        except (TypeError, ValueError):
            return False

    if request.method == "physics_simple" and architecture:
        eta = request.powertrain_features.get("eta_pt_est")
        electric_efficiency = request.powertrain_features.get("bev_eff_drive")
        if architecture == "BEV":
            if not _is_positive(electric_efficiency):
                issues.append("bev_eff_drive_missing")
        elif architecture == "PHEV":
            if not _is_positive(eta):
                issues.append("eta_pt_est_missing")
            if not _is_positive(electric_efficiency):
                issues.append("bev_eff_drive_missing")
        elif not _is_positive(eta):
            issues.append("eta_pt_est_missing")

    return tuple(issues)


def _active_demand(request: FuelEstimateRequest) -> float | None:
    basis = str(request.energy_basis or "VDE_TOTAL").upper()
    key = "vde_net_mj_per_km" if basis == "VDE_NET" else "vde_total_mj_per_km"
    value = request.vehicle_features.get(key)
    return None if value is None else float(value)


class EnergyBalanceL0Adapter:
    """Small orchestration boundary around the two existing canonical owners."""

    def run(self, resolved: ResolvedSystemScenario) -> SystemScenarioResult:
        vehicle_selection = resolved.resolved_domains.get(DomainKind.VEHICLE_DEMAND)
        vehicle_identity = None
        if vehicle_selection is not None and isinstance(
            vehicle_selection.configuration, VehicleDemandConfiguration
        ):
            vehicle_identity = vehicle_selection.configuration.source_identity

        configuration_provenance: dict[str, Any] = {}
        for domain, selection in resolved.resolved_domains.items():
            if isinstance(selection, DomainProposal):
                configuration_provenance[domain.value] = {
                    "selection": "DOMAIN_PROPOSAL",
                    "proposal_id": selection.identity.proposal_id,
                    "based_on": selection.based_on.provenance.value,
                }
            else:
                configuration_provenance[domain.value] = {
                    "selection": "EFFECTIVE_CURRENT",
                    "provenance": selection.provenance.value,
                }
        provenance = {
            "configuration": configuration_provenance,
            "l0_assumptions": [
                {
                    "key": item.key,
                    "value": item.value,
                    "domain": item.domain.value,
                    "proposal_id": item.proposal_id,
                    "provenance": item.provenance.value,
                }
                for item in resolved.l0_assumption_contributions
            ],
            "technology_deltas": [
                {
                    "evaluation_order": item.evaluation_order,
                    "domain": item.domain.value,
                    "proposal_id": item.proposal_id,
                    "source_type": item.assumption.source_type,
                    "effect_basis": item.assumption.effect_basis,
                    "effect_value": item.assumption.effect_value,
                    "quantitative_status": item.quantitative_status,
                }
                for item in resolved.technology_delta_contributions
            ],
            "calculated_result": ProvenanceKind.CALCULATED.value,
        }

        if resolved.solver_readiness is not SolverReadiness.READY:
            return SystemScenarioResult(
                identity=resolved.identity,
                resolved_scenario=resolved,
                selected_vehicle_demand_identity=vehicle_identity,
                architecture_class=resolved.architecture_class,
                solver_identity=f"fuel_estimation.run_fuel_estimation:{ENGINE_VERSION}",
                readiness=resolved.solver_readiness,
                fidelity_manifest=resolved.fidelity_manifest,
                effective_assumptions=resolved.l0_effective_assumptions,
                provenance=provenance,
                warnings=tuple(resolved.issues),
            )
        request = resolved.fuel_estimate_request
        if not isinstance(request, FuelEstimateRequest):
            raise TypeError("ResolvedSystemScenario must contain an L0 request snapshot when READY.")

        baseline = run_fuel_estimation(request)
        delta_result: Mapping[str, Any] | None = None
        if resolved.ordered_technology_deltas:
            normalized = [
                normalize_technology_delta(tech_delta_assumption_to_dict(delta), index=index + 1)
                for index, delta in enumerate(resolved.ordered_technology_deltas)
            ]
            delta_result = apply_delta_stack_to_baseline(
                baseline,
                ctx={"energy_value_mj_per_km": _active_demand(request)},
                deltas=normalized,
            )

        warnings = list(baseline.warnings or ())
        if delta_result is not None:
            warnings.extend(str(item) for item in delta_result.get("warnings") or ())
        return SystemScenarioResult(
            identity=resolved.identity,
            resolved_scenario=resolved,
            fuel_estimate_result=baseline,
            technology_delta_result=delta_result,
            selected_vehicle_demand_identity=vehicle_identity,
            architecture_class=resolved.architecture_class,
            solver_identity=f"fuel_estimation.run_fuel_estimation:{ENGINE_VERSION}",
            model_identity=baseline.method,
            readiness=resolved.solver_readiness,
            fidelity_manifest=resolved.fidelity_manifest,
            effective_assumptions=resolved.l0_effective_assumptions,
            provenance=provenance,
            warnings=tuple(dict.fromkeys(warnings)),
        )


__all__ = [
    "EnergyBalanceL0Adapter",
    "EnergyBalanceL0RequestSnapshot",
    "build_energy_balance_l0_request",
    "energy_balance_l0_readiness_issues",
    "is_direct_powertrain_assumption",
]
