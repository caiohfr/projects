# src/vde_core/quick_scenario/efficiency_resolution.py
# -----------------------------------------------------------------------------
# Sprint 10D - the Quick Efficiency Scenario *output* contract.
#
# Mirrors resolution.py's own rationale: this module depends on
# `src.vde_core.fuel_estimation.FuelEstimateResult` (the canonical
# deterministic fuel/energy result) because that dependency is the entire
# point of resolving Efficiency Quick -- a resolved PSE is only useful once
# expressed as the same FuelEstimateResult every other consumer of that
# canonical service uses. Data shape only; efficiency_resolver.py owns the
# actual PSE/ML/Technology-Delta call chain.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

from src.vde_core.fuel_estimation import FuelEstimateResult

from .contracts import DomainReadiness, PseProvenance


@dataclass(frozen=True)
class PseReference:
    """A read-only PSE reference (Sec 4/8/9): Current PSE (pointed at the
    Quick Scenario's own source fuelcons row) or Benchmark PSE (pointed at a
    donor fuelcons row). Never a Final PSE by itself -- Sec 5: "Reference
    values must never silently overwrite Final PSE." Both are produced by
    the exact same canonical computation
    (`pwt_fuel_energy_service.derive_reference_pse`), applied to different
    rows -- there is no second "current PSE" formula and no second
    "benchmark PSE" formula (Sec 8/9).
    """

    status: str
    value_percent: float | None = None
    donor_source_identity: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def is_available(self) -> bool:
        return self.status == "available" and self.value_percent is not None


@dataclass(frozen=True)
class MlPseRecommendation:
    """Sec 10-12: ML-derived PSE recommendation. The model predicts final
    fuel/energy consumption, never PSE directly (Sec 10) -- `value_percent`
    here is always DERIVED from that prediction plus the active Quick
    Vehicle demand, via the same canonical `build_powertrain_efficiency_summary`
    every other method uses. No numeric confidence is invented (Sec 11):
    `confidence_label` mirrors only the canonical categorical
    high/medium/low/provided vocabulary, never a percentage.

    `quick_affected_features_changed` names which of the model's own input
    features actually differ because of a Quick Vehicle override;
    `features_not_represented` names ML-relevant demand features Quick does
    not recompute at all (e.g. per-phase VDE) -- Sec 11: "Do not tell the
    user 'ML considered the new tire' unless the changed Quick state
    actually changes one or more model input features."
    """

    status: str
    value_percent: float | None = None
    confidence_label: str | None = None
    artifact_status: str | None = None
    model_version: str | None = None
    coverage_status: str | None = None
    missing_features: tuple[str, ...] = ()
    quick_affected_features_changed: tuple[str, ...] = ()
    features_not_represented: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_available(self) -> bool:
        return self.status == "available" and self.value_percent is not None

    @property
    def understood_quick_vehicle_changes(self) -> bool:
        return len(self.quick_affected_features_changed) > 0


@dataclass(frozen=True)
class TechDeltaSuggestion:
    """Sec 13-15: the suggested PSE from stacking the requested Technology
    Delta assumptions through the existing canonical stacking math
    (`src.vde_core.technology_delta.apply_delta_stack_to_baseline`) -- a
    recommendation only (Sec 14). It never feeds the deterministic
    calculation unless explicitly adopted as Final PSE.
    """

    status: str
    value_percent: float | None = None
    applied_count: int = 0
    registered_only_count: int = 0
    highest_maturity: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def is_available(self) -> bool:
        return self.status == "available" and self.value_percent is not None


@dataclass(frozen=True)
class QuickEfficiencyResolution:
    """The resolved Efficiency Quick outcome for one QuickScenario (Sec 28).

    Consumes an already-resolved `QuickVehicleResolution` (Sec 2) -- this
    contract, and the resolver that builds it, never recompute Mass/Tire/
    Aero/roadload. `readiness` describes only the DETERMINISTIC
    `fuel_estimate_result`: `NOT_REQUESTED` when no Final PSE has been
    supplied yet (a valid, unresolved-by-choice state, Sec 20), `INVALID`
    when Final PSE is explicitly `0` (or non-positive) -- Sec 7: zero is not
    blank, and division by zero is never attempted -- `MISSING` when a
    positive Final PSE was supplied but the requested `energy_basis` (or the
    Vehicle Quick result itself) is unavailable, and `READY` once
    `fuel_estimate_result` is populated. The four reference/recommendation
    fields are independent of this readiness: Current/Benchmark PSE need
    only the source/donor fuelcons row and are computed regardless of
    whether Vehicle Quick succeeded; ML/Tech-Delta need the Quick-resolved
    Vehicle Demand and are `None`/unavailable when it isn't ready.
    """

    quick_scenario_identity: str
    readiness: DomainReadiness = DomainReadiness.NOT_REQUESTED
    issues: tuple[str, ...] = ()

    energy_basis: str = "VDE_TOTAL"

    current_pse: PseReference | None = None
    benchmark_pse: PseReference | None = None
    ml_recommendation: MlPseRecommendation | None = None
    tech_delta_suggestion: TechDeltaSuggestion | None = None

    final_pse_percent: float | None = None
    final_pse_provenance: PseProvenance | None = None

    fuel_estimate_result: FuelEstimateResult | None = None

    @property
    def is_ready(self) -> bool:
        return self.readiness is DomainReadiness.READY


__all__ = [
    "PseReference",
    "MlPseRecommendation",
    "TechDeltaSuggestion",
    "QuickEfficiencyResolution",
]
