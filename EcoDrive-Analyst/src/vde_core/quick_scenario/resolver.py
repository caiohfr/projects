# src/vde_core/quick_scenario/resolver.py
# -----------------------------------------------------------------------------
# Sprint 10B - Quick Mass + Aero resolution.
#
# Turns a QuickScenario's Vehicle overrides into a resolved temporary
# physical state and feeds it through the FROZEN Sprint 9 Vehicle Demand
# Core. Reuses the canonical resolvers directly:
#   - Mass:  resolve_mass_proposal() (vde_mass_proposal_resolver.py)
#   - Aero:  cdA_to_C()              (roadload/engine.py), same two-line
#            delta-to-C composition _resolve_aero() already uses
#            (vde_request_resolver.py) -- never a second CdA->C formula.
#   - VDE:   resolve_roadload_boundaries() / resolve_cycle_vde_results()
#            (comparison_report_service.py) -- the same lightweight,
#            already-ubiquitous Comparison-layer functions, not the heavy
#            legacy vde_request_resolver.resolve_vde_request() workbook
#            pipeline (never invoked here).
#   - Vehicle Demand: build_vehicle_demand_request() /
#            resolve_vehicle_demand_cycle() (vehicle_demand/adapters.py) +
#            calculate_vehicle_demand() (vehicle_demand/engine.py) -- the
#            frozen core, unmodified.
#
# All Mass/Aero mutation happens on a *copy* of the source vde_db row (a
# "synthetic row"); the source row/dict passed in (or fetched) is never
# written to, and nothing here ever writes to fuelcons_db/vde_db.
#
# See docs/sprints/SPRINT_10B_QUICK_MASS_AERO_RESOLUTION.md for the full
# design record, including the two flagged decisions this module encodes:
# which resolved mass field feeds the frozen core (Decision 1), and how an
# Absolute CdA request without a source CdA is handled (Decision 2).
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Mapping

from src.vde_core.comparison_report_service import (
    RoadloadBoundary,
    resolve_cycle_vde_results,
    resolve_roadload_boundaries,
)
from src.vde_core.database_management_contract import EntityType
from src.vde_core.database_management_service import get_record
from src.vde_core.repositories import fetch_vde_by_id
from src.vde_core.roadload import cdA_to_C
from src.vde_core.vde_mass_proposal_resolver import resolve_mass_proposal
from src.vde_core.vehicle_demand import RoadloadCoefficients, calculate_vehicle_demand
from src.vde_core.vehicle_demand.adapters import (
    build_vehicle_demand_request,
    resolve_vehicle_demand_cycle,
)

from .contracts import (
    DomainReadiness,
    MassQuickChange,
    QuickScenario,
    QuickVehicleReadiness,
    ReferencePressureProvenance,
    ScalarChange,
    VehicleQuickOverrides,
)
from .resolution import QuickVehicleResolution

_SOURCE_IDENTITY_KINDS = ("fc", "vde")


def _parse_source_identity(source_identity: str) -> tuple[str, int]:
    """Sec 3/6 (10A): the inverse of canonical_identity()'s
    f"fc:{fuelcons_id}" / f"vde:{vde_id}" format -- no parser existed
    anywhere in the codebase prior to this package (verified by audit), so
    this is a new, small, purpose-built utility.
    """

    kind, sep, raw_id = source_identity.partition(":")
    if not sep or kind not in _SOURCE_IDENTITY_KINDS or not raw_id:
        raise ValueError(
            f"Unrecognized Quick Scenario source_identity: {source_identity!r} "
            "(expected 'fc:<fuelcons_id>' or 'vde:<vde_id>')."
        )
    return kind, int(raw_id)


def _fetch_source_vde_row(source_identity: str) -> dict[str, Any]:
    """Fetch the raw vde_db row backing `source_identity`. Mass/Aero
    physical state lives only on vde_db (fuelcons_db has no mass/CdA/
    roadload columns of its own), so both fc:- and vde:-sourced scenarios
    collapse to fetching one vde_db row.
    """

    kind, record_id = _parse_source_identity(source_identity)
    if kind == "vde":
        row = fetch_vde_by_id(record_id)
    else:
        fuelcons_row = get_record(EntityType.FUEL_CONSUMPTION, record_id)
        if not fuelcons_row:
            raise ValueError(f"No fuelcons_db row found for fuelcons_id={record_id}.")
        row = fetch_vde_by_id(fuelcons_row["vde_id"])
    if not row:
        # fetch_vde_by_id returns {} (never None) for an unknown id.
        raise ValueError(f"No vde_db row found for source_identity={source_identity!r}.")
    return dict(row)


def _mass_proposal_type(legislation: str, mass_change: MassQuickChange) -> str:
    if mass_change.twc_shift_steps is not None:
        return "MASS_TWC_SHIFT"
    return "WLTP_MASS_LINE" if legislation == "WLTP" else "EPA_CURB_TO_TWC"


def _resolve_mass(
    source_row: Mapping[str, Any], mass_change: MassQuickChange | None
) -> tuple[dict[str, Any], DomainReadiness, list[str]]:
    if mass_change is None:
        return {}, DomainReadiness.NOT_REQUESTED, []

    legislation = str(source_row.get("legislation") or "").upper()

    if mass_change.twc_shift_steps is not None:
        if legislation != "EPA":
            return (
                {},
                DomainReadiness.MISSING,
                [f"TWC Shift is EPA-only; source legislation is {legislation or 'unknown'!r}."],
            )
        inputs: dict[str, Any] = {"shift_steps": mass_change.twc_shift_steps}
        if mass_change.twc_shift_side is not None:
            inputs["target_side"] = mass_change.twc_shift_side
        if mass_change.twc_curb_position is not None:
            inputs["curb_position"] = mass_change.twc_curb_position
    else:
        requested_curb = mass_change.curb_change.resolve(source_row.get("mass_kg"))
        if requested_curb is None:
            return (
                {},
                DomainReadiness.MISSING,
                ["Mass change requires source curb mass, which is unavailable."],
            )
        inputs = {"mass_kg": requested_curb}
        if legislation == "WLTP" and mass_change.wltp_line_type is not None:
            inputs["line_type"] = mass_change.wltp_line_type

    proposal_type = _mass_proposal_type(legislation, mass_change)
    outcome = resolve_mass_proposal(dict(source_row), proposal_type, inputs)
    status = outcome.get("status")
    issues = [str(issue.get("message")) for issue in outcome.get("issues") or ()]

    if status in ("Missing", "Invalid"):
        return {}, DomainReadiness.MISSING, issues or [f"Mass resolution status: {status}."]

    resolved = outcome.get("resolved_snapshot") or {}
    vde_calculation_mass_kg = resolved.get("vde_calculation_mass_kg")
    if vde_calculation_mass_kg is None:
        return (
            {},
            DomainReadiness.MISSING,
            issues + ["Mass resolver did not produce a VDE calculation mass."],
        )

    updates = {
        "mass_kg": resolved.get("curb_mass_kg", source_row.get("mass_kg")),
        # Decision 1: the frozen core / resolve_cycle_vde_results both trust
        # vde_row["test_mass_kg"] verbatim as the mass used for VDE physics
        # -- that must be the resolver's canonical vde_calculation_mass_kg
        # (TWC for EPA, TML/TMH for WLTP), never its separate physical
        # test_mass_kg/resolved_test_mass_kg output field (curb+136 for EPA).
        "test_mass_kg": vde_calculation_mass_kg,
        "vde_mass_basis": resolved.get("vde_mass_basis"),
    }
    return updates, DomainReadiness.READY, issues


def _resolve_aero(
    source_row: Mapping[str, Any],
    cda_change: ScalarChange | None,
    aero_reference_cda_m2: float | None,
    aero_reference_cda_provenance: ReferencePressureProvenance | None,
) -> tuple[dict[str, Any], DomainReadiness, list[str]]:
    if cda_change is None:
        return {}, DomainReadiness.NOT_REQUESTED, []

    source_cda = source_row.get("cda_m2")
    target_cda = cda_change.resolve(source_cda)
    if target_cda is None:
        # DELTA/PERCENT against a missing source CdA -- ScalarChange.resolve
        # already returns None here; no separate Aero-specific check needed.
        return (
            {},
            DomainReadiness.MISSING,
            ["Aero change requires source CdA, which is unavailable."],
        )

    issues: list[str] = []
    if source_cda is not None:
        reference_cda = float(source_cda)
    elif (
        aero_reference_cda_provenance is ReferencePressureProvenance.USER_PROVIDED
        and aero_reference_cda_m2 is not None
    ):
        # Decision 2: mirrors the canonical resolver's own manual
        # baseline_CdA override for AERO_ABSOLUTE_CDA -- never a silent
        # zero-reference guess.
        reference_cda = float(aero_reference_cda_m2)
        issues.append("Manual reference CdA was used because source CdA is unavailable.")
    else:
        return (
            {},
            DomainReadiness.MISSING,
            [
                "Absolute CdA requires a reference CdA; source CdA is unavailable and no "
                "user-provided aero_reference_cda_m2 was supplied."
            ],
        )

    source_c = source_row.get("coast_C_N_per_kph2")
    if source_c is None:
        return (
            {},
            DomainReadiness.MISSING,
            ["Aero change requires source coast_C_N_per_kph2, which is unavailable."],
        )

    delta_cda = target_cda - reference_cda
    updates = {
        "cda_m2": target_cda,
        # Same two-line composition _resolve_aero() uses (A/B untouched).
        "coast_C_N_per_kph2": float(source_c) + cdA_to_C(delta_cda),
    }
    return updates, DomainReadiness.READY, issues


def _build_synthetic_row(
    source_row: Mapping[str, Any],
    mass_updates: Mapping[str, Any],
    aero_updates: Mapping[str, Any],
) -> dict[str, Any]:
    synthetic = dict(source_row)
    synthetic.update(mass_updates)
    synthetic.update(aero_updates)
    return synthetic


def _roadload_coefficients(boundary: RoadloadBoundary) -> RoadloadCoefficients | None:
    if not boundary.available:
        return None
    return RoadloadCoefficients(A_N=boundary.A, B_N_per_kph=boundary.B, C_N_per_kph2=boundary.C)


def resolve_quick_vehicle_scenario(
    quick_scenario: QuickScenario, *, source_vde_row: Mapping[str, Any] | None = None
) -> QuickVehicleResolution:
    """Resolve a QuickScenario's Vehicle Quick layer (Mass + Aero).

    Mirrors the existing `build_vde_comparison_item(vde_id, vde_row=None)`
    optional-row pattern: pass an already-fetched `source_vde_row` for
    DB-free/testable resolution, or omit it to fetch by
    `quick_scenario.source_identity`. The source row (however obtained) is
    never mutated -- every domain resolver reads from it and writes only
    into a fresh copy.
    """

    source_row: Mapping[str, Any] = (
        dict(source_vde_row)
        if source_vde_row is not None
        else _fetch_source_vde_row(quick_scenario.source_identity)
    )

    overrides: VehicleQuickOverrides = quick_scenario.vehicle_overrides

    mass_updates, mass_status, mass_issues = _resolve_mass(source_row, overrides.mass_change)
    aero_updates, aero_status, aero_issues = _resolve_aero(
        source_row,
        overrides.cda_change,
        overrides.aero_reference_cda_m2,
        overrides.aero_reference_cda_provenance,
    )

    tire_status = DomainReadiness.NOT_REQUESTED
    tire_issues: list[str] = []
    if overrides.tire_change is not None:
        # Sec 18: no silent partial override -- Tire is out of scope for
        # 10B, so a requested Tire change blocks the whole scenario rather
        # than being silently ignored.
        tire_status = DomainReadiness.MISSING
        tire_issues = ["Tire Quick resolution is not implemented yet (deferred to a later package)."]

    readiness = QuickVehicleReadiness(mass=mass_status, aero=aero_status, tire=tire_status)
    issues = tuple(mass_issues + aero_issues + tire_issues)

    if not readiness.all_ready:
        return QuickVehicleResolution(
            quick_scenario_identity=quick_scenario.identity,
            readiness=readiness,
            issues=issues,
        )

    synthetic_row = _build_synthetic_row(source_row, mass_updates, aero_updates)

    boundaries = resolve_roadload_boundaries(synthetic_row)
    cycle_results = resolve_cycle_vde_results(synthetic_row)

    vehicle_demand_request = build_vehicle_demand_request(synthetic_row)
    cycle_frame = resolve_vehicle_demand_cycle(synthetic_row)
    vehicle_demand_result = None
    if cycle_frame is not None:
        vehicle_demand_result = calculate_vehicle_demand(vehicle_demand_request, cycle_frame)
    else:
        issues = issues + (
            "Vehicle Demand cycle trace is unavailable for this scenario's legislation; "
            "VDE/roadload fields are still available, but no VehicleDemandResult was produced.",
        )

    return QuickVehicleResolution(
        quick_scenario_identity=quick_scenario.identity,
        readiness=readiness,
        issues=issues,
        resolved_curb_mass_kg=synthetic_row.get("mass_kg"),
        resolved_vde_calculation_mass_kg=synthetic_row.get("test_mass_kg"),
        resolved_vde_mass_basis=synthetic_row.get("vde_mass_basis"),
        resolved_cda_m2=synthetic_row.get("cda_m2"),
        abc_total=_roadload_coefficients(boundaries["total"]),
        abc_net=_roadload_coefficients(boundaries["net"]),
        vde_total_mj_per_km=cycle_results["total"].aggregate,
        vde_net_mj_per_km=cycle_results["net"].aggregate,
        vehicle_demand_request=vehicle_demand_request,
        vehicle_demand_result=vehicle_demand_result,
    )


__all__ = ["resolve_quick_vehicle_scenario"]
