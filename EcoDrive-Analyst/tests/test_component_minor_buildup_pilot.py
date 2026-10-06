from __future__ import annotations

from copy import deepcopy

from src.vde_core.component_minor_buildup_pilot import (
    Reference,
    deterministic_stratified_sample,
    evaluate_vehicle,
    match_synthetic_reference,
    match_tire,
)


def _vehicle(**updates):
    row = {
        "vde_id": 1,
        "vehicle_configuration_id": "CFG-1",
        "architecture": "CONVENTIONAL_MULTI_SPEED",
        "application_class": "PASSENGER_LIGHT",
        "application_class_status": "MAPPED",
        "drive_layout": "FWD",
        "drive_status": "MAPPED",
        "test_mass_resolved_kg": 1500.0,
        "wheel_radius_mm": 310.0,
        "tire_size_resolved": None,
        "rrc_resolved": None,
        "rrc_source": None,
        "rolling_abc": (120.0, 0.3, 0.0),
        "rolling_minor_resolution_id": "CR-ROLL-1",
        "rolling_status": "SUPPORTED",
    }
    row.update(updates)
    return row


def _reference(domain: str, component_id: str, *, position=None, mass=None, radius=None):
    return Reference(
        component_id=component_id,
        resolution_id=f"CR-{component_id}",
        domain=domain,
        application_class="PASSENGER_LIGHT",
        drive="FWD",
        position=position,
        abc=(2.0, 0.0, 0.0),
        population_n=20,
        reference_mass_kg=mass,
        wheel_radius_mm=radius,
    )


def test_missing_reference_mass_keeps_synthetic_candidate_grade_c() -> None:
    result = match_synthetic_reference(_vehicle(), [_reference("BRAKE", "B1")], "BRAKE")
    assert result["selected"].component_id == "B1"
    assert result["grade"] == "C"
    assert result["accepted"] is False
    assert "REFERENCE_MASS_METADATA_MISSING" in result["reason_codes"]


def test_grade_b_requires_and_uses_metadata_not_rolling_minor_magnitude() -> None:
    reference = _reference("BRAKE", "B1", mass=1300.0)
    first = match_synthetic_reference(_vehicle(rolling_abc=(20.0, 0.0, 0.0)), [reference], "BRAKE")
    second = match_synthetic_reference(_vehicle(rolling_abc=(400.0, 1.0, 0.0)), [reference], "BRAKE")
    assert first["selected"] == second["selected"] == reference
    assert first["grade"] == second["grade"] == "B"
    assert first["accepted"] is second["accepted"] is True


def test_axle_is_excluded_even_when_metadata_match_is_strong() -> None:
    references = [
        _reference("AXLE", "AF", position="FRONT", mass=1500.0),
        _reference("AXLE", "AR", position="REAR", mass=1500.0),
    ]
    matches, curves, summary = evaluate_vehicle(_vehicle(), references, [])
    axle = next(row for row in matches if row["fine_domain"] == "AXLE")
    assert axle["accepted"] == 0
    assert axle["boundary_rejected"] == 1
    assert "AXLE_EXCLUDED_BOUNDARY_OVERLAP_RISK" in axle["rejection_reason"]
    assert all(row["axle_N"] == 0.0 for row in curves)
    assert "AXLE" not in summary["accepted_component_set"]


def test_tire_reference_size_without_rr_tier_is_candidate_only() -> None:
    tire_rows = [
        {"tire_id": str(index), "tire_test_code": f"T{index}", "size_code": "205/55R16"}
        for index in range(5)
    ]
    result = match_tire(_vehicle(tire_size_resolved="205/55R16"), tire_rows)
    assert result["grade"] == "C"
    assert result["accepted"] is False
    assert len(result["candidate_ids"]) == 5
    assert result["reason_codes"] == ["TIRE_RR_TIER_NOT_IDENTIFIABLE_FROM_EXISTING_METADATA"]


def test_envelope_rejection_does_not_scale_component_to_force_closure() -> None:
    vehicle = _vehicle(
        rolling_abc=(10.0, 0.0, 0.0),
        rrc_resolved=10.0,
        rrc_source="legacy.vde_db.rrc_N_per_kN:EXACT_MAKE_MODEL_YEAR_LOOKUP",
        test_mass_resolved_kg=2000.0,
    )
    matches, curves, summary = evaluate_vehicle(vehicle, [], [])
    tire = next(row for row in matches if row["fine_domain"] == "TIRE")
    assert tire["envelope_rejected"] == 1
    assert tire["accepted"] == 0
    assert all(row["tire_N"] == 0.0 and row["unresolved_N"] == 10.0 for row in curves)
    assert summary["scaled_to_force_closure"] == 0
    assert summary["buildup_status"] == "SYNTHETIC_BUILDUP_EXCEEDS_ROLLING_MINOR"


def test_stratified_sample_is_deterministic_and_respects_architecture_targets() -> None:
    population = []
    for architecture in ("CONVENTIONAL_MULTI_SPEED", "EV_FIXED_GEAR"):
        for index in range(12):
            population.append(
                _vehicle(
                    vde_id=index + (100 if architecture == "EV_FIXED_GEAR" else 0),
                    vehicle_configuration_id=f"CFG-{architecture}-{index}",
                    architecture=architecture,
                    application_class="PASSENGER_LIGHT" if index % 2 else "SUV",
                    test_mass_resolved_kg=1200.0 + index * 100.0,
                    drive_layout="FWD" if index % 3 else "AWD",
                )
            )
    targets = {"CONVENTIONAL_MULTI_SPEED": 7, "EV_FIXED_GEAR": 5}
    first = deterministic_stratified_sample(deepcopy(population), targets)
    second = deterministic_stratified_sample(deepcopy(population), targets)
    assert [row["vde_id"] for row in first] == [row["vde_id"] for row in second]
    assert sum(row["architecture"] == "CONVENTIONAL_MULTI_SPEED" for row in first) == 7
    assert sum(row["architecture"] == "EV_FIXED_GEAR" for row in first) == 5
