from __future__ import annotations

from copy import deepcopy

from src.vde_core.component_vector_search import CandidateVector
from src.vde_core.component_physics_vector_search import (
    DRIVETRAIN_DOMAINS,
    ROLLING_DOMAINS,
    search_aggregate,
    shuffle_pools,
    truncate_pools,
)


def _vehicle(**updates):
    row = {"vde_id": 1, "rolling_abc": (100.0, 0.0, 0.0)}
    row.update(updates)
    return row


def _vector(domain, name, force, score=80.0):
    return CandidateVector(name, domain, (name,), (force, 0.0, 0.0), score, "CROSSOVER", "UTILITY_LIGHT",
                           "AWD", None, 1200.0, 2700.0, 15.0, 23.0,
                           {"closure_used_in_generation": False}, (), "SYNTHETIC_REFERENCE", "TEST_BOUNDARY")


def test_rollingminor_domains_exclude_axle_and_transmission() -> None:
    assert ROLLING_DOMAINS == ("TIRE", "BRAKE", "HUB_BEARING")
    pools = {"TIRE": [_vector("TIRE", "T", 40)], "BRAKE": [_vector("BRAKE", "B", 2)],
             "HUB_BEARING": [_vector("HUB_BEARING", "H", 3)], "AXLE": [_vector("AXLE", "A", 50)],
             "TRANSMISSION": [_vector("TRANSMISSION", "X", 50)]}
    accepted, best, _ = search_aggregate(_vehicle(), "ROLLING_MINOR", (100.0, 0.0, 0.0), pools, ROLLING_DOMAINS)
    assert best is not None and best["component_count"] == 3
    assert all("axle_vector" not in row and "transmission_vector" not in row for row in accepted)


def test_drivetrain_domains_exclude_brake_and_hub() -> None:
    assert DRIVETRAIN_DOMAINS == ("TRANSMISSION", "AXLE")
    pools = {"TRANSMISSION": [_vector("TRANSMISSION", "X", 20)], "AXLE": [_vector("AXLE", "A", 10)],
             "BRAKE": [_vector("BRAKE", "B", 80)], "HUB_BEARING": [_vector("HUB_BEARING", "H", 80)]}
    _, best, _ = search_aggregate(_vehicle(), "DRIVETRAIN_AGGREGATE", (100.0, 0.0, 0.0), pools, DRIVETRAIN_DOMAINS)
    assert best is not None and best["component_count"] == 2
    assert "brake_vector" not in best and "hub_bearing_vector" not in best


def test_five_percent_gate_residual_and_no_scaling() -> None:
    pools = {"TIRE": [_vector("TIRE", "T103", 103)], "BRAKE": [], "HUB_BEARING": []}
    accepted, best, _ = search_aggregate(_vehicle(), "ROLLING_MINOR", (100.0, 0.0, 0.0), pools, ROLLING_DOMAINS)
    assert len(accepted) == 1 and best["closure_status"] == "TOLERANCE_ACCEPTED"
    assert best["scaled"] == 0
    assert set(__import__("json").loads(best["other_unresolved_vector_json"])) == {0.0}
    assert set(__import__("json").loads(best["other_minor_unresolved_vector_json"])) == {0.0}
    assert best["other_driveline_unresolved_vector_json"] is None
    rejected, rejected_best, _ = search_aggregate(
        _vehicle(), "ROLLING_MINOR", (100.0, 0.0, 0.0),
        {"TIRE": [_vector("TIRE", "T106", 106)], "BRAKE": [], "HUB_BEARING": []}, ROLLING_DOMAINS)
    assert rejected == [] and rejected_best is None


def test_positive_unresolved_residual_is_preserved() -> None:
    _, best, _ = search_aggregate(
        _vehicle(), "ROLLING_MINOR", (100.0, 0.0, 0.0),
        {"TIRE": [_vector("TIRE", "T40", 40)], "BRAKE": [], "HUB_BEARING": []}, ROLLING_DOMAINS)
    assert set(__import__("json").loads(best["other_unresolved_vector_json"])) == {60.0}
    assert best["median_unresolved_residual_pct"] == 60.0


def test_ranking_is_deterministic_and_compatibility_first() -> None:
    pools = {"TIRE": [_vector("TIRE", "HIGH", 30, 90), _vector("TIRE", "LOW", 90, 70)],
             "BRAKE": [], "HUB_BEARING": []}
    first = search_aggregate(_vehicle(), "ROLLING_MINOR", (100.0, 0.0, 0.0), pools, ROLLING_DOMAINS)[1]
    second = search_aggregate(_vehicle(), "ROLLING_MINOR", (100.0, 0.0, 0.0), pools, ROLLING_DOMAINS)[1]
    assert first["combination_id"] == second["combination_id"]
    assert first["tire_vector"] == "HIGH"


def test_shuffle_is_reproducible_and_preserves_pool_sizes() -> None:
    base = {
        1: {"TIRE": [_vector("TIRE", "A", 10)]},
        2: {"TIRE": [_vector("TIRE", "B", 20)]},
        3: {"TIRE": [_vector("TIRE", "C", 30)]},
    }
    first = shuffle_pools(base, ("TIRE",))
    second = shuffle_pools(base, ("TIRE",))
    assert [first[key]["TIRE"][0].vector_id for key in sorted(first)] == ["B", "C", "A"]
    assert first == second
    assert all(len(first[key]["TIRE"]) == len(base[key]["TIRE"]) for key in base)


def test_pool_sensitivity_truncation_does_not_mutate_base() -> None:
    pools = {"TIRE": [_vector("TIRE", str(index), float(index)) for index in range(5)]}
    frozen = deepcopy(pools)
    top2 = truncate_pools(pools, {"TIRE": 2})
    assert len(top2["TIRE"]) == 2
    assert pools == frozen and len(pools["TIRE"]) == 5
