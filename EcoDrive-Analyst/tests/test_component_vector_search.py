from __future__ import annotations

from src.vde_core.component_minor_buildup_pilot import Reference
from src.vde_core.component_minor_buildup_refinement import enrich_applicability
from src.vde_core.component_vector_search import (
    CandidateVector,
    build_candidate_pools,
    generate_axle_pool,
    generate_brake_pool,
    search_combinations,
)


def _vehicle(**updates):
    row = {
        "vde_id": 1, "application_class": "CROSSOVER", "drive_layout": "AWD",
        "test_mass_resolved_kg": 1900.0, "rrc_resolved": None, "rrc_source": None,
        "tire_size_resolved": None, "rolling_abc": (100.0, 0.0, 0.0),
    }
    row.update(updates)
    return row


def _ref(domain, component_id, app="CROSSOVER", drive="AWD", position=None, abc=(2.0, 0.0, 0.0)):
    return Reference(component_id, f"CR-{component_id}", domain, app, drive, position, abc, 20)


def _vector(domain, name, abc, score):
    return CandidateVector(name, domain, (name,), abc, score, "CROSSOVER", "UTILITY_LIGHT", "AWD", None,
                           1200.0, 2700.0, 15.0, 23.0, {"closure_used_in_generation": False}, (),
                           "SYNTHETIC_REFERENCE", f"{domain}_EXPLANATORY_SUBCOMPONENT_OF_ROLLING_MINOR")


def test_nonunique_brake_references_become_ranked_pool() -> None:
    refs = enrich_applicability([
        _ref("BRAKE", "B1"),
        _ref("BRAKE", "B2", app="SUV", drive="AWD", abc=(3.0, 0.0, 0.0)),
    ])
    pool = generate_brake_pool(_vehicle(), refs)
    assert len(pool) == 2
    assert pool[0].application_class == "CROSSOVER"
    assert all(item.feature_vector["closure_used_in_generation"] is False for item in pool)


def test_candidate_generation_does_not_depend_on_rolling_minor() -> None:
    refs = enrich_applicability([_ref("BRAKE", "B1")])
    first = generate_brake_pool(_vehicle(rolling_abc=(5.0, 0.0, 0.0)), refs)
    second = generate_brake_pool(_vehicle(rolling_abc=(500.0, 1.0, 0.0)), refs)
    assert [item.vector_id for item in first] == [item.vector_id for item in second]


def test_axle_is_included_as_conditional_rolling_minor_vector() -> None:
    refs = enrich_applicability([
        _ref("AXLE", "AF", position="FRONT"), _ref("AXLE", "AR", position="REAR"),
    ])
    pool = generate_axle_pool(_vehicle(), refs)
    assert len(pool) == 1
    assert pool[0].position == "FRONT+REAR"
    assert pool[0].boundary_assumption == "AXLE_ALLOCATED_WITHIN_ROLLING_MINOR_SYNTHETIC_BUILDUP"


def test_search_prefers_more_domains_before_better_closure() -> None:
    pools = {
        "TIRE": [_vector("TIRE", "T1", (50.0, 0.0, 0.0), 90.0)],
        "BRAKE": [_vector("BRAKE", "B1", (2.0, 0.0, 0.0), 80.0)],
        "HUB_BEARING": [], "AXLE": [],
    }
    rows, best = search_combinations(_vehicle(), pools)
    assert len(rows) == 3
    assert best[0]["populated_domain_count"] == 2
    assert best[0]["tire_vector"] == "T1" and best[0]["brake_vector"] == "B1"
    assert all(row["scaled"] == 0 for row in rows)


def test_search_rejects_more_than_five_percent_and_keeps_tolerance_case() -> None:
    pools = {
        "TIRE": [_vector("TIRE", "T103", (103.0, 0.0, 0.0), 90.0),
                 _vector("TIRE", "T106", (106.0, 0.0, 0.0), 89.0)],
        "BRAKE": [], "HUB_BEARING": [], "AXLE": [],
    }
    rows, best = search_combinations(_vehicle(), pools)
    by_tire = {row["tire_vector"]: row for row in rows}
    assert by_tire["T103"]["closure_status"] == "TOLERANCE_ACCEPTED"
    assert by_tire["T103"]["accepted"] == 1
    assert by_tire["T106"]["accepted"] == 0
    assert best[0]["tire_vector"] == "T103"
