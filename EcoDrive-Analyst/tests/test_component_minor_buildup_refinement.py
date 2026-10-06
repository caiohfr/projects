from __future__ import annotations

from src.vde_core.component_minor_buildup_pilot import Reference
from src.vde_core.component_minor_buildup_refinement import (
    classify_closure,
    enrich_applicability,
    evaluate_refinement_vehicle,
    match_applicability_reference,
)


def _vehicle(**updates):
    row = {
        "vde_id": 1,
        "application_class": "PASSENGER_LIGHT",
        "drive_layout": "FWD",
        "test_mass_resolved_kg": 1500.0,
        "rrc_resolved": None,
        "rrc_source": None,
        "tire_size_resolved": None,
        "rolling_abc": (100.0, 0.0, 0.0),
        "rolling_minor_resolution_id": "CR-ROLL-1",
    }
    row.update(updates)
    return row


def _ref(domain="BRAKE", component_id="B1", position=None, abc=(2.0, 0.0, 0.0)):
    return Reference(
        component_id=component_id,
        resolution_id=f"CR-{component_id}",
        domain=domain,
        application_class="PASSENGER_LIGHT",
        drive="FWD",
        position=position,
        abc=abc,
        population_n=20,
    )


def test_closure_uses_five_percent_without_scaling() -> None:
    clean = classify_closure((100.0, 0.0, 0.0), {"TIRE": (100.0, 0.0, 0.0)})
    conditional = classify_closure((100.0, 0.0, 0.0), {"TIRE": (103.0, 0.0, 0.0)})
    rejected = classify_closure((100.0, 0.0, 0.0), {"TIRE": (106.0, 0.0, 0.0)})
    assert clean["status"] == "CLEAN_BUILDUP"
    assert conditional["status"] == "TOLERANCE_ACCEPTED"
    assert conditional["max_overshoot_pct"] == 3.0
    assert rejected["status"] == "REJECTED_BUILDUP"


def test_rule_defined_metadata_produces_grade_b_not_fabricated_grade_a() -> None:
    enriched = enrich_applicability([_ref()])
    result = match_applicability_reference(_vehicle(), enriched, "BRAKE")
    assert result["grade"] == "B"
    assert result["accepted"] is True
    assert enriched[0].source == "SYNTHETIC_REFERENCE"
    assert enriched[0].applicability_source == "RULE_DEFINED_APPLICABILITY"


def test_out_of_envelope_mass_remains_grade_c() -> None:
    result = match_applicability_reference(_vehicle(test_mass_resolved_kg=3000.0), enrich_applicability([_ref()]), "BRAKE")
    assert result["grade"] == "C"
    assert result["accepted"] is False


def test_tolerance_accepted_separates_zero_residual_from_closure_error() -> None:
    mass = 1000.0
    rrc_for_103_n = 103.0 * 1000.0 / (mass * 9.80665)
    matches, curves, summary = evaluate_refinement_vehicle(
        _vehicle(test_mass_resolved_kg=mass, rrc_resolved=rrc_for_103_n, rrc_source="legacy", rolling_abc=(100.0, 0.0, 0.0)),
        [], [], [], phase="A",
    )
    assert summary["closure_status"] == "TOLERANCE_ACCEPTED"
    assert all(row["minor_unresolved_N"] == 0.0 for row in curves)
    assert all(abs(row["model_closure_error_N"] - 3.0) < 1e-9 for row in curves)
    tire = next(row for row in matches if row["fine_domain"] == "TIRE")
    assert tire["accepted_in_build"] == 1
    assert tire["estimate_status"] == "CONDITIONAL"
    assert tire["closure_reason"] == "WITHIN_SYNTHETIC_CLOSURE_TOLERANCE"


def test_axle_overlap_protection_remains_effective() -> None:
    refs = [_ref("AXLE", "AF", "FRONT"), _ref("AXLE", "AR", "REAR")]
    matches, curves, summary = evaluate_refinement_vehicle(_vehicle(), refs, enrich_applicability(refs), [], phase="B")
    axle = next(row for row in matches if row["fine_domain"] == "AXLE")
    assert axle["accepted_in_build"] == 0
    assert axle["boundary_rejected"] == 1
    assert all(row["axle_N"] == 0.0 for row in curves)
    assert "AXLE" not in summary["accepted_component_set"]
