# src/vde_core/quick_scenario/resolver.py
# -----------------------------------------------------------------------------
# Sprint 10C - Quick Mass + Tire + Aero resolution.
#
# Turns a QuickScenario's Vehicle overrides into a resolved temporary
# physical state and feeds it through the FROZEN Sprint 9 Vehicle Demand
# Core. Reuses the canonical resolvers directly:
#   - Mass:  resolve_mass_proposal() (vde_mass_proposal_resolver.py)
#   - Tire:  resolve_tire_proposal() (vde_tire_proposal_resolver.py)
#            owns RRC, pressure, load and Tire ABC physics.
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
from src.vde_core.tire_roadload_service import get_tire_by_id
from src.vde_core.vde_mass_proposal_resolver import resolve_mass_proposal
from src.vde_core.vde_tire_proposal_resolver import (
    resolve_tire_proposal,
    tire_reference_pressure_psi,
)
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
    TireQuickChange,
    TireSource,
    TireTransformMode,
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
        # Tire is the next Quick stage. Preserve the canonical mass
        # resolver's regulatory/load fields so resolve_tire_proposal() sees
        # the newly resolved scenario mass, never stale source-row values.
        "inertia_class": resolved.get("inertia_class"),
        "tire_load_mass_used_kg": resolved.get("tire_load_mass_used_kg"),
        "tire_load_mass_basis": resolved.get("tire_load_mass_basis"),
    }
    return updates, DomainReadiness.READY, issues


_TIRE_STATE_FIELDS = (
    "tire_db_id",
    "tire_code",
    "tire_snapshot",
    "front_pressure_psi",
    "rear_pressure_psi",
    "rrc_N_per_kN",
    "target_rrc_N_per_kN",
    "tire_source_rrc_N_per_kN",
    "tire_target_rrc_N_per_kN",
    "tire_adjusted_rrc_N_per_kN",
    "tire_delta_rrc_N_per_kN",
    "tire_reference_front_pressure_psi",
    "tire_reference_rear_pressure_psi",
    "tire_requested_front_pressure_psi",
    "tire_requested_rear_pressure_psi",
    "tire_front_weight_fraction",
    "tire_pressure_sensitivity",
    "tire_adjustment_method",
    "tire_abc_method",
    "tire_review_status",
    "tire_rule_status",
    "tire_rule_notes",
    "tire_load_mass_basis",
    "tire_load_mass_used_kg",
    "source_tire_load_mass_used_kg",
    "tire_A_final",
    "tire_B_final",
    "tire_C_final",
)


def _tire_readiness(outcome: Mapping[str, Any]) -> DomainReadiness:
    status = str(outcome.get("status") or "").strip().upper()
    if status == "INVALID":
        return DomainReadiness.INVALID
    if status == "MISSING":
        return DomainReadiness.MISSING
    return DomainReadiness.READY


def _tire_issue_messages(outcome: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get("message") or "Tire resolution issue.")
        for item in list(outcome.get("issues") or ())
    ]


def _apply_tire_outcome(
    working_row: Mapping[str, Any], outcome: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply canonical Tire state and its delta to authoritative TOTAL ABC.

    The delta itself is produced exclusively by resolve_tire_proposal().
    Missing TOTAL coefficients remain missing; this adapter never performs
    a TOTAL/NET fallback.
    """

    row = dict(working_row)
    resolved = dict(outcome.get("resolved_snapshot") or {})
    for field_name in _TIRE_STATE_FIELDS:
        if field_name in resolved:
            row[field_name] = resolved.get(field_name)

    delta = dict(resolved.get("tire_delta_abc") or {})
    for total_field, component in (
        ("coast_A_N", "A"),
        ("coast_B_N_per_kph", "B"),
        ("coast_C_N_per_kph2", "C"),
    ):
        base_value = row.get(total_field)
        delta_value = delta.get(component)
        if base_value is not None and delta_value is not None:
            row[total_field] = float(base_value) + float(delta_value)
    return row


def _canonical_tire_call(
    source_row: Mapping[str, Any],
    current_row: Mapping[str, Any],
    proposal_type: str,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    resolver_inputs = dict(inputs)
    tire_current = dict(current_row)
    for pressure_field in ("front_pressure_psi", "rear_pressure_psi"):
        if resolver_inputs.get(pressure_field) is not None:
            tire_current[pressure_field] = resolver_inputs[pressure_field]
    return resolve_tire_proposal(
        dict(source_row),
        proposal_type,
        resolver_inputs,
        current_snapshot=tire_current,
    )


def _source_tire_record(source_row: Mapping[str, Any]) -> dict[str, Any] | None:
    snapshot = dict(source_row.get("tire_snapshot") or {})
    if snapshot:
        return snapshot
    tire_id = source_row.get("tire_db_id") or source_row.get("front_tire_id")
    if tire_id is None:
        return None
    try:
        return dict(get_tire_by_id(int(tire_id)) or {}) or None
    except Exception:
        return None


def _pressure_reference_pair(
    source_row: Mapping[str, Any],
    selected_tire: Mapping[str, Any] | None,
    tire_change: TireQuickChange,
) -> tuple[float | None, float | None, ReferencePressureProvenance]:
    pressure_change = tire_change.pressure_delta
    assert pressure_change is not None
    if pressure_change.reference_pressure_provenance is ReferencePressureProvenance.USER_PROVIDED:
        reference = pressure_change.reference_pressure_psi
        return reference, reference, ReferencePressureProvenance.USER_PROVIDED

    db_reference = tire_reference_pressure_psi(dict(selected_tire or {}))
    source_front = source_row.get("front_pressure_psi")
    source_rear = source_row.get("rear_pressure_psi")
    if tire_change.source is TireSource.TIRE_DB and db_reference is not None:
        return db_reference, db_reference, ReferencePressureProvenance.SOURCE
    if source_front is not None and source_rear is not None:
        return float(source_front), float(source_rear), ReferencePressureProvenance.SOURCE
    if db_reference is not None:
        return db_reference, db_reference, ReferencePressureProvenance.SOURCE
    return None, None, ReferencePressureProvenance.SOURCE


def _pressure_proposal_type(
    tire_change: TireQuickChange,
    selected_tire: Mapping[str, Any] | None,
    pressure_provenance: ReferencePressureProvenance,
) -> str:
    family = str(dict(selected_tire or {}).get("standard_family") or "").strip().upper()
    # Canonical DB lookup dispatches SAE to its richer load/pressure model
    # and ISO to the approved reference-point pressure estimate. A current
    # tire without either characterization uses the canonical pressure-only
    # TIRE_TARGET_RRC path with target intentionally blank.
    if family == "SAE":
        return "TIRE_DB_LOOKUP" if tire_change.source is TireSource.TIRE_DB else "INHERIT"
    if (
        tire_change.source is TireSource.TIRE_DB
        and family == "ISO"
        and pressure_provenance is not ReferencePressureProvenance.USER_PROVIDED
    ):
        return "TIRE_DB_LOOKUP"
    return "TIRE_TARGET_RRC"


def _resolve_tire(
    source_row: Mapping[str, Any],
    mass_resolved_row: Mapping[str, Any],
    tire_change: TireQuickChange | None,
) -> tuple[
    dict[str, Any],
    DomainReadiness,
    list[str],
    dict[str, Any] | None,
    ReferencePressureProvenance | None,
]:
    if tire_change is None:
        return dict(mass_resolved_row), DomainReadiness.NOT_REQUESTED, [], None, None

    selected_tire: dict[str, Any] | None
    if tire_change.source is TireSource.TIRE_DB:
        try:
            selected_tire = dict(get_tire_by_id(int(tire_change.tire_db_id)) or {}) or None
        except Exception:
            selected_tire = None
        if selected_tire is None:
            return (
                dict(mass_resolved_row),
                DomainReadiness.MISSING,
                [f"No Tire DB row found for tire_db_id={tire_change.tire_db_id}."],
                None,
                None,
            )
    else:
        selected_tire = _source_tire_record(source_row)

    base_inputs: dict[str, Any] = {}
    if selected_tire is not None:
        base_inputs["tire_snapshot"] = dict(selected_tire)
    if tire_change.source is TireSource.TIRE_DB:
        base_inputs["tire_db_id"] = tire_change.tire_db_id
        if selected_tire.get("tire_code") or selected_tire.get("code"):
            base_inputs["tire_code"] = selected_tire.get("tire_code") or selected_tire.get("code")

    mode = tire_change.transform_mode
    pressure_provenance: ReferencePressureProvenance | None = None
    outcome_base_row = mass_resolved_row

    if tire_change.source is TireSource.TIRE_DB:
        source_outcome = _canonical_tire_call(
            source_row, mass_resolved_row, "TIRE_DB_LOOKUP", base_inputs
        )
        source_status = _tire_readiness(source_outcome)
        if source_status is not DomainReadiness.READY:
            return (
                dict(mass_resolved_row),
                source_status,
                _tire_issue_messages(source_outcome),
                dict(source_outcome.get("resolved_snapshot") or {}),
                None,
            )
        selected_row = _apply_tire_outcome(mass_resolved_row, source_outcome)
        if mode is TireTransformMode.NONE:
            return (
                selected_row,
                DomainReadiness.READY,
                _tire_issue_messages(source_outcome),
                dict(source_outcome.get("resolved_snapshot") or {}),
                None,
            )
        if mode is TireTransformMode.IMPROVEMENT_PCT:
            improvement_outcome = _canonical_tire_call(
                selected_row,
                selected_row,
                "TIRE_IMPROVEMENT_PCT",
                {"tire_improvement_pct": tire_change.improvement_pct},
            )
            final_status = _tire_readiness(improvement_outcome)
            return (
                _apply_tire_outcome(selected_row, improvement_outcome),
                final_status,
                _tire_issue_messages(source_outcome) + _tire_issue_messages(improvement_outcome),
                dict(improvement_outcome.get("resolved_snapshot") or {}),
                None,
            )

    if mode is TireTransformMode.NONE:
        outcome = _canonical_tire_call(source_row, mass_resolved_row, "INHERIT", {})
    elif mode is TireTransformMode.TARGET_RRC:
        outcome = _canonical_tire_call(
            source_row,
            mass_resolved_row,
            "TIRE_TARGET_RRC",
            {"target_rrc_N_per_kN": tire_change.target_rrc_n_per_kn},
        )
    elif mode is TireTransformMode.RRC_DELTA:
        reference_outcome = _canonical_tire_call(source_row, mass_resolved_row, "INHERIT", {})
        reference_rrc = reference_outcome.get("resolved_rrc_N_per_kN")
        if reference_rrc is None:
            return (
                dict(mass_resolved_row),
                DomainReadiness.MISSING,
                _tire_issue_messages(reference_outcome)
                + ["Current Tire RRC is required for an RRC Delta; no value was assumed."],
                dict(reference_outcome.get("resolved_snapshot") or {}),
                None,
            )
        outcome = _canonical_tire_call(
            source_row,
            mass_resolved_row,
            "TIRE_TARGET_RRC",
            {
                "target_rrc_N_per_kN": float(reference_rrc)
                + float(tire_change.rrc_delta_n_per_kn)
            },
        )
    elif mode is TireTransformMode.IMPROVEMENT_PCT:
        outcome = _canonical_tire_call(
            source_row,
            mass_resolved_row,
            "TIRE_IMPROVEMENT_PCT",
            {"tire_improvement_pct": tire_change.improvement_pct},
        )
    elif mode is TireTransformMode.PRESSURE_DELTA:
        reference_front, reference_rear, pressure_provenance = _pressure_reference_pair(
            source_row, selected_tire, tire_change
        )
        if reference_front is None or reference_rear is None:
            return (
                dict(mass_resolved_row),
                DomainReadiness.MISSING,
                [
                    "Pressure Delta requires reference front/rear pressure from the source/Tire DB "
                    "or an explicit USER_PROVIDED reference_pressure_psi."
                ],
                None,
                pressure_provenance,
            )
        pressure_change = tire_change.pressure_delta
        assert pressure_change is not None
        rear_delta = (
            pressure_change.front_delta_psi
            if pressure_change.rear_delta_psi is None
            else pressure_change.rear_delta_psi
        )
        pressure_inputs = dict(base_inputs)
        pressure_inputs.update(
            {
                "front_pressure_psi": float(reference_front) + pressure_change.front_delta_psi,
                "rear_pressure_psi": float(reference_rear) + rear_delta,
            }
        )
        pressure_source = dict(source_row)
        pressure_source["front_pressure_psi"] = reference_front
        pressure_source["rear_pressure_psi"] = reference_rear
        pressure_proposal_type = _pressure_proposal_type(
            tire_change, selected_tire, pressure_provenance
        )
        pressure_current = mass_resolved_row
        if tire_change.source is TireSource.TIRE_DB and pressure_proposal_type == "TIRE_TARGET_RRC":
            # Selection was stage one; the canonical pressure-only path now
            # treats that resolved DB tire as its source. This is required
            # for CUSTOM/reference-point tires and for an explicit manual
            # reference that must not be replaced by the DB value.
            pressure_source = dict(selected_row)
            pressure_source["front_pressure_psi"] = reference_front
            pressure_source["rear_pressure_psi"] = reference_rear
            pressure_current = dict(selected_row)
            outcome_base_row = selected_row
            pressure_inputs.pop("tire_db_id", None)
            pressure_inputs.pop("tire_snapshot", None)
        outcome = _canonical_tire_call(
            pressure_source,
            pressure_current,
            pressure_proposal_type,
            pressure_inputs,
        )
    else:  # pragma: no cover - contract validation makes this unreachable
        raise ValueError(f"Unsupported Tire transform mode: {mode!r}")

    status = _tire_readiness(outcome)
    return (
        _apply_tire_outcome(outcome_base_row, outcome),
        status,
        _tire_issue_messages(outcome),
        dict(outcome.get("resolved_snapshot") or {}),
        pressure_provenance,
    )


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
    tire_updates: Mapping[str, Any],
    aero_updates: Mapping[str, Any],
) -> dict[str, Any]:
    synthetic = dict(source_row)
    synthetic.update(mass_updates)
    synthetic.update(tire_updates)
    synthetic.update(aero_updates)
    return synthetic


def _roadload_coefficients(boundary: RoadloadBoundary) -> RoadloadCoefficients | None:
    if not boundary.available:
        return None
    return RoadloadCoefficients(A_N=boundary.A, B_N_per_kph=boundary.B, C_N_per_kph2=boundary.C)


def resolve_quick_vehicle_scenario(
    quick_scenario: QuickScenario, *, source_vde_row: Mapping[str, Any] | None = None
) -> QuickVehicleResolution:
    """Resolve a QuickScenario's Vehicle Quick layer (Mass + Tire + Aero).

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
    mass_resolved_row = dict(source_row)
    mass_resolved_row.update(mass_updates)
    (
        tire_resolved_row,
        tire_status,
        tire_issues,
        tire_state,
        pressure_provenance,
    ) = _resolve_tire(source_row, mass_resolved_row, overrides.tire_change)
    aero_updates, aero_status, aero_issues = _resolve_aero(
        tire_resolved_row,
        overrides.cda_change,
        overrides.aero_reference_cda_m2,
        overrides.aero_reference_cda_provenance,
    )

    readiness = QuickVehicleReadiness(mass=mass_status, aero=aero_status, tire=tire_status)
    issues = tuple(mass_issues + aero_issues + tire_issues)

    if not readiness.all_ready:
        return QuickVehicleResolution(
            quick_scenario_identity=quick_scenario.identity,
            readiness=readiness,
            issues=issues,
        )

    tire_updates = {
        key: value
        for key, value in tire_resolved_row.items()
        if key not in source_row or source_row.get(key) != value
    }
    synthetic_row = _build_synthetic_row(source_row, mass_updates, tire_updates, aero_updates)

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
        resolved_tire_db_id=synthetic_row.get("tire_db_id"),
        resolved_tire_code=synthetic_row.get("tire_code"),
        resolved_rrc_n_per_kn=synthetic_row.get("rrc_N_per_kN"),
        reference_rrc_n_per_kn=(tire_state or {}).get("tire_source_rrc_N_per_kN"),
        resolved_front_pressure_psi=synthetic_row.get("front_pressure_psi"),
        resolved_rear_pressure_psi=synthetic_row.get("rear_pressure_psi"),
        reference_front_pressure_psi=(tire_state or {}).get(
            "tire_reference_front_pressure_psi"
        ),
        reference_rear_pressure_psi=(tire_state or {}).get(
            "tire_reference_rear_pressure_psi"
        ),
        reference_pressure_provenance=(
            None if pressure_provenance is None else pressure_provenance.value
        ),
        resolved_tire_a_n=synthetic_row.get("tire_A_final"),
        resolved_tire_b_n_per_kph=synthetic_row.get("tire_B_final"),
        resolved_tire_c_n_per_kph2=synthetic_row.get("tire_C_final"),
        tire_calculation_source=synthetic_row.get("tire_adjustment_method")
        or synthetic_row.get("tire_calc_source"),
        tire_abc_method=synthetic_row.get("tire_abc_method"),
        tire_load_mass_basis=synthetic_row.get("tire_load_mass_basis"),
        tire_load_mass_used_kg=synthetic_row.get("tire_load_mass_used_kg"),
        abc_total=_roadload_coefficients(boundaries["total"]),
        abc_net=_roadload_coefficients(boundaries["net"]),
        vde_total_mj_per_km=cycle_results["total"].aggregate,
        vde_net_mj_per_km=cycle_results["net"].aggregate,
        vehicle_demand_request=vehicle_demand_request,
        vehicle_demand_result=vehicle_demand_result,
    )


__all__ = ["resolve_quick_vehicle_scenario"]
