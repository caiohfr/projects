from __future__ import annotations

from copy import deepcopy

from src.vde_core.component_vector_search import CandidateVector
from src.vde_core.component_rollingminor_metadata_completion import (
    resolve_application_class_v14,
    shuffle_pools_seeded,
)


def _vehicle(**updates):
    row = {
        "application_class": None, "category": "Car", "make": "OEM", "model": "Unknown",
        "test_mass_resolved_kg": 1600.0,
    }
    row.update(updates)
    return row


def _vector(name):
    return CandidateVector(name, "TIRE", (name,), (10.0, 0.0, 0.0), 80.0, "PASSENGER_LIGHT", "PASSENGER",
                           "FWD", None, 800.0, 1900.0, None, None,
                           {"closure_used_in_generation": False}, (), "INTERNAL", "ROLLING_MINOR")


def test_existing_legacy_mapping_has_priority_over_name_and_mass_rules() -> None:
    result = resolve_application_class_v14(_vehicle(application_class="SUV", model="Civic", test_mass_resolved_kg=1400))
    assert result["application_class"] == "SUV"
    assert result["application_class_source"] == "LEGACY_MAPPING"
    assert result["application_class_confidence"] == "HIGH"


def test_explicit_body_category_is_high_confidence() -> None:
    result = resolve_application_class_v14(_vehicle(category="sedan", test_mass_resolved_kg=2400))
    assert result["application_class"] == "PASSENGER_STANDARD"
    assert result["application_class_confidence"] == "HIGH"


def test_curated_model_rule_is_medium_and_closure_independent() -> None:
    first = resolve_application_class_v14(_vehicle(model="CIVIC 4DR", rolling_abc=(1.0, 0.0, 0.0)))
    second = resolve_application_class_v14(_vehicle(model="CIVIC 4DR", rolling_abc=(999.0, 5.0, 0.0)))
    assert first == second
    assert first["application_class"] == "PASSENGER_LIGHT"
    assert first["application_class_confidence"] == "MEDIUM"
    assert first["closure_used_for_classification"] is False


def test_generic_mass_fallback_is_low_and_not_validation_eligible() -> None:
    result = resolve_application_class_v14(_vehicle())
    assert result["application_class"] == "PASSENGER_LIGHT"
    assert result["application_class_confidence"] == "LOW"
    assert result["eligible_for_validation"] is False


def test_van_pickup_and_suv_rules_are_deterministic() -> None:
    assert resolve_application_class_v14(_vehicle(category="Truck", model="Transit RWD Wagon"))["application_class"] == "PASSENGER_VAN"
    assert resolve_application_class_v14(_vehicle(category="Truck", model="Tacoma Hybrid"))["application_class"] == "PICKUP_LIGHT_DUTY"
    assert resolve_application_class_v14(_vehicle(category="Truck", model="MDX", test_mass_resolved_kg=2100))["application_class"] == "CROSSOVER"


def test_seeded_shuffle_is_reproducible_preserves_sizes_and_does_not_mutate() -> None:
    base = {
        1: {"TIRE": [_vector("A")], "BRAKE": [], "HUB_BEARING": []},
        2: {"TIRE": [_vector("B")], "BRAKE": [], "HUB_BEARING": []},
        3: {"TIRE": [_vector("C")], "BRAKE": [], "HUB_BEARING": []},
    }
    frozen = deepcopy(base)
    first = shuffle_pools_seeded(base, 11)
    second = shuffle_pools_seeded(base, 11)
    assert first == second and base == frozen
    assert [len(first[key]["TIRE"]) for key in sorted(first)] == [1, 1, 1]
    assert [first[key]["TIRE"][0].vector_id for key in sorted(first)] != ["A", "B", "C"]
