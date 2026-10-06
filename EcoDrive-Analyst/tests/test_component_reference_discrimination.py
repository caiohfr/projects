from __future__ import annotations

import sqlite3

from src.vde_core.component_reference_discrimination import (
    generate_internal_brake_references,
    independent_drivetrain_audit,
)
from src.vde_core.component_minor_buildup_pilot import Reference
from src.vde_core.component_minor_buildup_refinement import ApplicabilityReference
from src.vde_core.component_vector_search import CandidateVector, generate_brake_pool


def _legacy(rows: int = 20) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE vde_db (
               id INTEGER PRIMARY KEY, category TEXT, drive_type TEXT,
               brake_A_coef_N REAL, brake_B_coef_Npkph REAL,
               brake_C_coef_Npkph2 REAL
           )"""
    )
    for index in range(rows):
        connection.execute(
            "INSERT INTO vde_db VALUES (?,?,?,?,?,?)",
            (index + 1, "COMPACT CARS", "All Wheel Drive", 2.0 + 0.01 * index, 0.01, 0.0001),
        )
    return connection


def _vector(domain: str, name: str, force: float) -> CandidateVector:
    return CandidateVector(
        vector_id=name, domain=domain, reference_ids=(name,), abc=(force, 0.0, 0.0),
        compatibility_score=90.0, application_class="PASSENGER_LIGHT",
        application_family="PASSENGER", drive="AWD", position=None,
        mass_min_kg=800.0, mass_max_kg=1900.0, wheel_min_in=13.0,
        wheel_max_in=20.0, feature_vector={"closure_used_in_generation": False},
        reason_codes=("EXACT_APPLICATION_CLASS",), provenance="INTERNAL",
        boundary_assumption=f"{domain}_REFERENCE",
    )


def test_internal_brake_reference_is_deterministic_and_not_aggregate_fitted() -> None:
    connection = _legacy()
    try:
        first, audit_first = generate_internal_brake_references(
            connection, [], {("PASSENGER_LIGHT", "AWD")}
        )
        second, audit_second = generate_internal_brake_references(
            connection, [], {("PASSENGER_LIGHT", "AWD")}
        )
    finally:
        connection.close()
    assert first == second and audit_first == audit_second
    assert len(first) == 1
    assert first[0].population_n == 20
    assert audit_first[0]["aggregate_target_used"] is False
    assert "POPULATION_MEDOID" in audit_first[0]["generation_rule"]


def test_existing_exact_reference_blocks_duplicate_addition() -> None:
    existing = Reference(
        "EXISTING", "CR_EXISTING", "BRAKE", "PASSENGER_LIGHT", "AWD", None,
        (2.0, 0.01, 0.0001), 100,
    )
    connection = _legacy()
    try:
        additions, audit = generate_internal_brake_references(
            connection, [existing], {("PASSENGER_LIGHT", "AWD")}
        )
    finally:
        connection.close()
    assert additions == []
    assert audit == []


def test_population_below_frozen_minimum_is_rejected() -> None:
    connection = _legacy(19)
    try:
        additions, audit = generate_internal_brake_references(
            connection, [], {("PASSENGER_LIGHT", "AWD")}
        )
    finally:
        connection.close()
    assert additions == []
    assert audit[0]["status"] == "REJECTED_INSUFFICIENT_POPULATION"


def test_candidate_preserves_enriched_reference_provenance() -> None:
    reference = Reference(
        "P15", "CR_P15", "BRAKE", "PASSENGER_LIGHT", "AWD", None,
        (2.0, 0.01, 0.0001), 20,
    )
    enriched = ApplicabilityReference(
        reference, 800.0, 1900.0, "LIGHT", 13.0, 20.0, "STANDARD",
        "BRAKE", None, source="INTERNAL_POPULATION", applicability_source="EXACT_CLASS_DRIVE",
    )
    vehicle = {
        "application_class": "PASSENGER_LIGHT", "drive_layout": "AWD",
        "test_mass_resolved_kg": 1400.0,
    }
    candidates = generate_brake_pool(vehicle, [enriched], limit=3)
    assert candidates[0].provenance == "INTERNAL_POPULATION|EXACT_CLASS_DRIVE"


def test_independent_candidates_do_not_claim_additive_boundary() -> None:
    vehicles = [{
        "vde_id": 1, "vehicle_configuration_id": "CFG-1",
        "application_class": "PASSENGER_LIGHT", "drive_layout": "AWD",
        "architecture": "CONVENTIONAL_MULTI_SPEED", "transmission_type": "automatic",
        "transmission_normalized": "AT", "gear_count": 8, "electrification": "ICE",
    }]
    boundaries = {1: {
        "drivetrain_boundary_name": "DRIVETRAIN_AGGREGATE",
        "drivetrain_boundary": {"abc": (100.0, 0.0, 0.0)},
    }}
    pools = {1: {
        "TRANSMISSION": [_vector("TRANSMISSION", "T", 20.0)],
        "AXLE": [_vector("AXLE", "A", 10.0)],
    }}
    transmission, axle, audit, summary, decision = independent_drivetrain_audit(
        vehicles, boundaries, pools
    )
    assert transmission[0]["aggregate_closure_used_for_selection"] is False
    assert axle[0]["aggregate_closure_used_for_selection"] is False
    assert audit[0]["diagnostic_class"] == "BOUNDARY_UNKNOWN"
    assert audit[0]["combined_to_aggregate_pct"] is None
    assert {row["metric"]: row["value"] for row in summary}["boundary_compatible_pairs"] == 0
    assert decision == "DRIVETRAIN_FINE_COMPONENTS_DIAGNOSTIC_ONLY"
