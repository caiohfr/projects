# src/vde_core/quick_scenario/efficiency_resolver.py
# -----------------------------------------------------------------------------
# Sprint 10D - Quick PSE / ML recommendation / Technology Delta resolution.
#
# Consumes an already-resolved QuickVehicleResolution (Sprint 10B/10C) --
# never recalculates Mass/Tire/Aero/roadload (Sec 2). Reuses the canonical
# efficiency machinery directly:
#   - Current/Benchmark PSE: pwt_fuel_energy_service.derive_reference_pse()
#     (Sprint 10D extraction of the existing Powertrain Scenario reference/
#     rebase computation) -- the SAME function for both, pointed at the
#     source fuelcons row (current) or a donor fuelcons row (benchmark).
#   - Deterministic PSE + fuel/energy: fuel_estimation.run_fuel_estimation()
#     -> powertrain_efficiency.build_powertrain_efficiency_summary() (PSE,
#     method-agnostic) and fuel_estimation._physics_simple() (the
#     Demand/PSE/LHV formula) -- reused verbatim for manual, benchmark-
#     rebased, and Final-PSE calculations alike; the ML path
#     (method="ml_prediction") is the SAME entry point, so PSE stays
#     DERIVED from the model's predicted fuel/energy, never a direct
#     model output.
#   - Technology Delta stacking: technology_delta.apply_delta_stack_to_baseline()
#     (Sprint 10D extraction of the existing Powertrain Scenario stacking
#     math) -- applied to a baseline built from Current PSE, exposed only
#     as a suggestion (Sec 14) never fed into the deterministic result
#     unless the caller adopts it as Final PSE.
#
# Final PSE (QuickScenario.final_pse_percent/pse_provenance, Sprint 10A) is
# the sole authority for the deterministic result (Sec 4/5): references and
# recommendations computed here never overwrite it; adopting one is the
# caller's job of constructing a QuickScenario with that value + the
# matching *_ACCEPTED provenance.
#
# No file in src/vde_core/vehicle_demand/ is imported or modified. No DB
# writes happen here -- every DB read (source/donor fuelcons rows) is a
# plain fetch, never an insert/update.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Mapping

from src.vde_core.database_management_contract import EntityType
from src.vde_core.database_management_service import get_record
from src.vde_core.fuel_energy import GCO2_PER_L, LHV_MJ_PER_L
from src.vde_core.fuel_estimation import FuelEstimateRequest, FuelEstimateResult, run_fuel_estimation
from src.vde_core.pwt_fuel_energy_service import derive_reference_pse, resolve_reference_fuel_type
from src.vde_core.technology_delta import (
    apply_delta_stack_to_baseline,
    normalize_technology_delta,
    tech_delta_assumption_to_dict,
)
from src.vde_core.vehicle_demand import RoadloadBasis

from .contracts import DomainReadiness, QuickScenario
from .efficiency_resolution import (
    MlPseRecommendation,
    PseReference,
    QuickEfficiencyResolution,
    TechDeltaSuggestion,
)
from .resolution import QuickVehicleResolution
from .resolver import _parse_source_identity

_ENERGY_BASIS_TO_FUEL_ESTIMATE_BASIS = {
    RoadloadBasis.TOTAL: "VDE_TOTAL",
    RoadloadBasis.NET: "VDE_NET",
}

# Sec 11: which of the ML model's own input features a requested Vehicle
# Quick domain can actually change, derived from the audited data flow in
# resolver.py (Mass changes test_mass_kg -> downstream VDE only; Tire's
# canonical tire_delta_abc can touch any of coast_A/B/C; Aero touches only
# coast_C via cdA_to_C). Used only to report which features genuinely
# changed -- never to imply the model understood a change it cannot see.
_ML_AFFECTED_FEATURES_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    "mass": ("vde_net_mj_per_km",),
    "tire": ("coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2", "vde_net_mj_per_km"),
    "aero": ("coast_C_N_per_kph2", "vde_net_mj_per_km"),
}

# Sec 11: Quick never recomputes per-phase VDE (only whole-cycle
# vde_total/net_mj_per_km), so these two ML features are never
# representative of a Quick Vehicle change, regardless of what changed.
_ML_FEATURES_NEVER_REPRESENTED: tuple[str, ...] = ("vde_urb_mj_per_km", "vde_hw_mj_per_km")


def _fetch_source_fuelcons_row(source_identity: str) -> dict[str, Any] | None:
    """`None` for a `vde:`-sourced scenario (no linked fuelcons row to
    derive Current PSE / powertrain context from) or when the identity is
    malformed -- callers must not guess a substitute.
    """

    try:
        kind, record_id = _parse_source_identity(source_identity)
    except ValueError:
        return None
    if kind != "fc":
        return None
    row = get_record(EntityType.FUEL_CONSUMPTION, record_id)
    return dict(row) if row else None


def _demand_mj_per_km(vehicle_resolution: QuickVehicleResolution, basis: RoadloadBasis) -> float | None:
    result = vehicle_resolution.vehicle_demand_result
    if result is None:
        return None
    if basis is RoadloadBasis.NET:
        return result.net_summary.vde_mj_per_km if result.net_summary is not None else None
    return result.total_summary.vde_mj_per_km


def _powertrain_context(source_fuelcons_row: Mapping[str, Any] | None) -> tuple[str, str]:
    """(electrification, fuel_type), defaulting exactly like the existing
    canonical helpers do when the source has no fuelcons row or doesn't
    specify (`_build_observed_reference_request`'s `"ICE"` default,
    `_derive_reference_pse`'s `"Gasoline"` default) -- not a new assumption.
    """

    row = dict(source_fuelcons_row or {})
    electrification = str(row.get("electrification") or "ICE").upper()
    fuel_type = (resolve_reference_fuel_type(row) if row else None) or "Gasoline"
    return electrification, fuel_type


def _build_fuel_estimate_request(
    vehicle_resolution: QuickVehicleResolution,
    source_fuelcons_row: Mapping[str, Any] | None,
    basis: RoadloadBasis,
    *,
    method: str,
    powertrain_overrides: Mapping[str, Any] | None = None,
    model_options: Mapping[str, Any] | None = None,
) -> FuelEstimateRequest:
    electrification, fuel_type = _powertrain_context(source_fuelcons_row)
    lhv = LHV_MJ_PER_L.get(fuel_type, LHV_MJ_PER_L["Gasoline"])
    gco2_per_l = GCO2_PER_L.get(fuel_type, GCO2_PER_L["Gasoline"])
    abc = vehicle_resolution.abc_total

    powertrain_features: dict[str, Any] = {
        "fuel_type": fuel_type,
        "LHV_MJ_per_L": lhv,
        "gCO2_per_L": gco2_per_l,
    }
    if powertrain_overrides:
        powertrain_features.update(powertrain_overrides)

    vehicle_features = {
        "electrification": electrification,
        "vde_total_mj_per_km": _demand_mj_per_km(vehicle_resolution, RoadloadBasis.TOTAL),
        "vde_net_mj_per_km": _demand_mj_per_km(vehicle_resolution, RoadloadBasis.NET),
        "coast_A_N": abc.A_N if abc is not None else None,
        "coast_B_N_per_kph": abc.B_N_per_kph if abc is not None else None,
        "coast_C_N_per_kph2": abc.C_N_per_kph2 if abc is not None else None,
    }

    return FuelEstimateRequest(
        energy_basis=_ENERGY_BASIS_TO_FUEL_ESTIMATE_BASIS[basis],
        method=method,
        vehicle_features=vehicle_features,
        powertrain_features=powertrain_features,
        model_options=dict(model_options or {}),
    )


def _pse_reference_from_row(
    row: Mapping[str, Any] | None, *, donor_source_identity: str | None = None
) -> PseReference:
    if not row:
        return PseReference(status="unavailable", donor_source_identity=donor_source_identity)
    outcome = derive_reference_pse(dict(row))
    if outcome["status"] != "available" or outcome["value"] is None:
        return PseReference(
            status="unavailable",
            donor_source_identity=donor_source_identity,
            warnings=(str(outcome["status"]),),
        )
    return PseReference(
        status="available",
        value_percent=float(outcome["value"]) * 100.0,
        donor_source_identity=donor_source_identity,
    )


def _resolve_benchmark_pse(benchmark_source_identity: str | None) -> PseReference | None:
    if benchmark_source_identity is None:
        return None
    try:
        kind, record_id = _parse_source_identity(benchmark_source_identity)
    except ValueError as exc:
        return PseReference(
            status="unavailable", donor_source_identity=benchmark_source_identity, warnings=(str(exc),)
        )
    if kind != "fc":
        return PseReference(
            status="unavailable",
            donor_source_identity=benchmark_source_identity,
            warnings=("Benchmark PSE requires a fc:<fuelcons_id> donor identity.",),
        )
    donor_row = get_record(EntityType.FUEL_CONSUMPTION, record_id)
    if not donor_row:
        return PseReference(
            status="unavailable",
            donor_source_identity=benchmark_source_identity,
            warnings=("Donor fuelcons row not found.",),
        )
    return _pse_reference_from_row(dict(donor_row), donor_source_identity=benchmark_source_identity)


def _quick_affected_ml_features(quick_scenario: QuickScenario) -> tuple[str, ...]:
    overrides = quick_scenario.vehicle_overrides
    changed: set[str] = set()
    if overrides.mass_change is not None:
        changed.update(_ML_AFFECTED_FEATURES_BY_DOMAIN["mass"])
    if overrides.tire_change is not None:
        changed.update(_ML_AFFECTED_FEATURES_BY_DOMAIN["tire"])
    if overrides.cda_change is not None:
        changed.update(_ML_AFFECTED_FEATURES_BY_DOMAIN["aero"])
    return tuple(sorted(changed))


def _resolve_ml_recommendation(
    quick_scenario: QuickScenario,
    vehicle_resolution: QuickVehicleResolution,
    source_fuelcons_row: Mapping[str, Any] | None,
    basis: RoadloadBasis,
    *,
    ml_model_options: Mapping[str, Any] | None = None,
) -> MlPseRecommendation:
    if not quick_scenario.efficiency_inputs.request_ml_recommendation:
        return MlPseRecommendation(status="not_requested")
    if vehicle_resolution.vehicle_demand_result is None:
        return MlPseRecommendation(
            status="unavailable", warnings=("Vehicle Quick result is unavailable.",)
        )

    request = _build_fuel_estimate_request(
        vehicle_resolution,
        source_fuelcons_row,
        basis,
        method="ml_prediction",
        model_options=ml_model_options,
    )
    result = run_fuel_estimation(request)
    assumptions = dict(result.assumptions or {})
    pse_summary = dict(assumptions.get("pse_summary") or {})

    quick_affected = _quick_affected_ml_features(quick_scenario)
    recommendation = MlPseRecommendation(
        status="available" if pse_summary.get("value") is not None else "unavailable",
        value_percent=(
            float(pse_summary["value"]) * 100.0 if pse_summary.get("value") is not None else None
        ),
        confidence_label=result.confidence,
        artifact_status=assumptions.get("integration_status"),
        model_version=assumptions.get("model_version"),
        coverage_status=assumptions.get("coverage_status"),
        missing_features=tuple(assumptions.get("missing_features") or ()),
        quick_affected_features_changed=quick_affected,
        features_not_represented=_ML_FEATURES_NEVER_REPRESENTED,
        warnings=tuple(result.warnings or ()),
    )
    return recommendation


def _resolve_tech_delta_suggestion(
    quick_scenario: QuickScenario,
    vehicle_resolution: QuickVehicleResolution,
    source_fuelcons_row: Mapping[str, Any] | None,
    basis: RoadloadBasis,
) -> TechDeltaSuggestion:
    deltas = quick_scenario.efficiency_inputs.technology_deltas
    if not deltas:
        return TechDeltaSuggestion(status="not_requested")
    if vehicle_resolution.vehicle_demand_result is None:
        return TechDeltaSuggestion(
            status="unavailable", warnings=("Vehicle Quick result is unavailable.",)
        )

    current_pse = _pse_reference_from_row(source_fuelcons_row)
    if not current_pse.is_available:
        return TechDeltaSuggestion(
            status="unavailable",
            warnings=("Current PSE reference is required as the Technology Delta baseline.",),
        )

    electrification, _ = _powertrain_context(source_fuelcons_row)
    eta_key = "bev_eff_drive" if electrification == "BEV" else "eta_pt_est"
    baseline_request = _build_fuel_estimate_request(
        vehicle_resolution,
        source_fuelcons_row,
        basis,
        method="physics_simple",
        powertrain_overrides={eta_key: current_pse.value_percent / 100.0},
    )
    baseline_result = run_fuel_estimation(baseline_request)

    normalized = [
        normalize_technology_delta(tech_delta_assumption_to_dict(delta), index=index + 1)
        for index, delta in enumerate(deltas)
    ]
    demand = _demand_mj_per_km(vehicle_resolution, basis)
    stack = apply_delta_stack_to_baseline(
        baseline_result, ctx={"energy_value_mj_per_km": demand}, deltas=normalized
    )
    counts = dict(stack.get("delta_counts") or {})
    suggested_pse = stack["proposal"].get("pse")
    if suggested_pse is None:
        return TechDeltaSuggestion(
            status="unavailable",
            applied_count=counts.get("applied", 0),
            registered_only_count=counts.get("registered_only", 0),
            warnings=tuple(stack.get("warnings") or ()),
        )
    return TechDeltaSuggestion(
        status="available",
        value_percent=float(suggested_pse) * 100.0,
        applied_count=counts.get("applied", 0),
        registered_only_count=counts.get("registered_only", 0),
        highest_maturity=stack.get("highest_maturity"),
        warnings=tuple(stack.get("warnings") or ()),
    )


def _resolve_fuel_estimate_result(
    quick_scenario: QuickScenario,
    vehicle_resolution: QuickVehicleResolution,
    source_fuelcons_row: Mapping[str, Any] | None,
    basis: RoadloadBasis,
) -> tuple[FuelEstimateResult | None, DomainReadiness, list[str]]:
    final_pse_percent = quick_scenario.final_pse_percent
    if final_pse_percent is None:
        return None, DomainReadiness.NOT_REQUESTED, []
    if final_pse_percent <= 0.0:
        return (
            None,
            DomainReadiness.INVALID,
            [
                "Final PSE must be a positive value; 0 (or negative) is explicit but "
                "physically invalid for the demand/consumed-energy division."
            ],
        )
    if vehicle_resolution.vehicle_demand_result is None:
        return (
            None,
            DomainReadiness.MISSING,
            ["Vehicle Quick result is unavailable; cannot compute a deterministic Efficiency result."],
        )

    demand = _demand_mj_per_km(vehicle_resolution, basis)
    if demand is None:
        basis_label = "NET" if basis is RoadloadBasis.NET else "TOTAL"
        return (
            None,
            DomainReadiness.MISSING,
            [f"Requested energy basis {basis_label} is unavailable for this Vehicle Quick result."],
        )

    electrification, _ = _powertrain_context(source_fuelcons_row)
    eta_key = "bev_eff_drive" if electrification == "BEV" else "eta_pt_est"
    request = _build_fuel_estimate_request(
        vehicle_resolution,
        source_fuelcons_row,
        basis,
        method="physics_simple",
        powertrain_overrides={eta_key: final_pse_percent / 100.0},
    )
    result = run_fuel_estimation(request)
    return result, DomainReadiness.READY, []


def resolve_quick_efficiency_scenario(
    quick_scenario: QuickScenario,
    vehicle_resolution: QuickVehicleResolution,
    *,
    source_fuelcons_row: Mapping[str, Any] | None = None,
    energy_basis: RoadloadBasis = RoadloadBasis.TOTAL,
    ml_model_options: Mapping[str, Any] | None = None,
) -> QuickEfficiencyResolution:
    """Resolve a QuickScenario's Efficiency Quick layer.

    Mirrors `resolve_quick_vehicle_scenario`'s optional-row pattern: pass an
    already-fetched `source_fuelcons_row` for DB-free/testable resolution
    (or `None` explicitly for a `vde:`-sourced scenario with no linked
    fuelcons data), or omit it to fetch by `quick_scenario.source_identity`.
    `vehicle_resolution` must be the already-resolved Vehicle Quick result
    for the SAME scenario -- this function never recomputes it.

    `ml_model_options` is passed straight through to the ML
    `FuelEstimateRequest.model_options` (e.g. `{"ml_predictor": <callable>}`
    or `{"ml_artifact_path": ...}`) -- the same injection point
    `run_fuel_estimation`'s own tests use, so tests here don't depend on the
    real ~16MB artifact's exact trained behavior.
    """

    if source_fuelcons_row is None:
        source_fuelcons_row = _fetch_source_fuelcons_row(quick_scenario.source_identity)
    else:
        source_fuelcons_row = dict(source_fuelcons_row)

    current_pse = _pse_reference_from_row(source_fuelcons_row)
    benchmark_pse = _resolve_benchmark_pse(quick_scenario.efficiency_inputs.benchmark_source_identity)
    ml_recommendation = _resolve_ml_recommendation(
        quick_scenario,
        vehicle_resolution,
        source_fuelcons_row,
        energy_basis,
        ml_model_options=ml_model_options,
    )
    tech_delta_suggestion = _resolve_tech_delta_suggestion(
        quick_scenario, vehicle_resolution, source_fuelcons_row, energy_basis
    )

    fuel_estimate_result, readiness, issues = _resolve_fuel_estimate_result(
        quick_scenario, vehicle_resolution, source_fuelcons_row, energy_basis
    )

    return QuickEfficiencyResolution(
        quick_scenario_identity=quick_scenario.identity,
        readiness=readiness,
        issues=tuple(issues),
        energy_basis=_ENERGY_BASIS_TO_FUEL_ESTIMATE_BASIS[energy_basis],
        current_pse=current_pse,
        benchmark_pse=benchmark_pse,
        ml_recommendation=ml_recommendation,
        tech_delta_suggestion=tech_delta_suggestion,
        final_pse_percent=quick_scenario.final_pse_percent,
        final_pse_provenance=quick_scenario.pse_provenance,
        fuel_estimate_result=fuel_estimate_result,
    )


__all__ = ["resolve_quick_efficiency_scenario"]
