from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from etl.scripts.sprint_12_transmission_grouping_signal_experiment import (
    _read_only_connection,
    _resolve_roots,
    sha256_file,
)
from etl.scripts.transmission_research_grouping_v03 import (
    CONTROLLED_CATEGORICAL_FEATURES,
    CONTROLLED_NUMERIC_FEATURES,
    WEIGHT_MAPS,
    _control_labels,
    _fold_soft_context,
    build_fold_assignment,
    classify_pair,
    evidence_weight,
)


def _application(**changes):
    value = {
        "hardware_designation": "",
        "transmission_family": "",
        "gear_ratios": "",
        "external_source_ids": "source-1",
        "normalized_transmission_type": "TORQUE_CONVERTER_AUTOMATIC",
        "gears": 8,
        "drive_architecture": "RWD",
        "model": "230i",
        "model_year": 2020,
        "marketing_description": "8-speed Steptronic",
    }
    value.update(changes)
    return value


def test_research_confidence_is_separate_from_canonical_identity():
    left = _application(gear_ratios="I 5.000; II 3.200", canonical_identity_confidence="UNRESOLVED")
    right = _application(gear_ratios="I 5.001; II 3.199", canonical_identity_confidence="UNRESOLVED", model="M4")

    result = classify_pair(left, right)

    assert result.relation == "PROBABLE"
    assert left["canonical_identity_confidence"] == "UNRESOLVED"
    assert right["canonical_identity_confidence"] == "UNRESOLVED"


def test_partial_external_evidence_can_be_plausible_without_canonical_promotion():
    left = _application(external_source_ids="source-1")
    right = _application(external_source_ids="", canonical_identity_confidence="UNRESOLVED")

    result = classify_pair(left, right)

    assert result.relation == "PLAUSIBLE"
    assert "partial_external_same_application_generation_anchor" in result.positive
    assert right["canonical_identity_confidence"] == "UNRESOLVED"


def test_coastdown_outcome_cannot_enter_research_classifier():
    with pytest.raises(ValueError, match="forbidden"):
        classify_pair(_application(coast_A_N=123.0), _application())


def test_exact_carryover_resolves_to_one_independent_root():
    assert _resolve_roots([10, 11, 12], {10: None, 11: 10, 12: 11}) == {10: 10, 11: 10, 12: 10}


def test_conflicting_explicit_hardware_has_zero_weight():
    result = classify_pair(
        _application(hardware_designation="ZF 8HP50"),
        _application(hardware_designation="ZF 8HP76"),
    )
    assert result.relation == "CONFLICTING"
    assert evidence_weight(result.relation, "primary") == 0.0


def test_weight_mappings_are_deterministic_and_not_probabilities():
    assert evidence_weight("PROBABLE", "conservative") == 0.5
    assert evidence_weight("PROBABLE", "primary") == 0.7
    assert evidence_weight("PROBABLE", "permissive") == 0.85
    assert set(WEIGHT_MAPS) == {"conservative", "primary", "permissive"}


def test_controlled_baseline_contains_group_defining_structured_covariates():
    assert {"speed_scaled", "speed_scaled_sq", "test_mass_kg", "model_year", "gear_count", "final_drive_ratio", "nv_ratio"}.issubset(CONTROLLED_NUMERIC_FEATURES)
    assert {"make_norm", "model_norm", "category_norm", "drive_norm", "electrification", "transmission_type_norm"}.issubset(CONTROLLED_CATEGORICAL_FEATURES)


def test_grouped_folds_prevent_application_lineage_leakage():
    units = pd.DataFrame(
        [{"vde_id": lineage * 2 + variant, "application_lineage": f"L{lineage}"} for lineage in range(10) for variant in range(2)]
    )
    assignment = build_fold_assignment(units, n_splits=5)
    for fold in range(5):
        test = set(units[units.vde_id.map(assignment).eq(fold)].application_lineage)
        train = set(units[~units.vde_id.map(assignment).eq(fold)].application_lineage)
        assert test.isdisjoint(train)


def test_evidence_balancing_separates_testable_group_but_not_same_lineage():
    units = pd.DataFrame(
        [
            {"vde_id": i, "independent_application_id": f"A{i}", "application_lineage": f"L{i}", "research_group_id": "U", "research_group_confidence": "UNRESOLVED"}
            for i in range(10)
        ]
        + [
            {"vde_id": 10, "independent_application_id": "A10", "application_lineage": "PX", "research_group_id": "G1", "research_group_confidence": "PROBABLE"},
            {"vde_id": 11, "independent_application_id": "A11", "application_lineage": "PY", "research_group_id": "G1", "research_group_confidence": "PROBABLE"},
            {"vde_id": 12, "independent_application_id": "A12", "application_lineage": "SAME", "research_group_id": "G2", "research_group_confidence": "PLAUSIBLE"},
            {"vde_id": 13, "independent_application_id": "A13", "application_lineage": "SAME", "research_group_id": "G2", "research_group_confidence": "PLAUSIBLE"},
        ]
    )
    assignment = build_fold_assignment(units, n_splits=5)

    assert assignment[10] != assignment[11]
    assert assignment[12] == assignment[13]


def test_random_negative_control_preserves_observed_shared_group_sizes():
    groups = pd.DataFrame(
        {
            "independent_application_id": [f"A{i}" for i in range(7)],
            "research_group_id": ["G1", "G1", "G1", "G2", "G2", "U1", "U2"],
            "research_group_size": [3, 3, 3, 2, 2, 1, 1],
        }
    )
    labels = _control_labels(groups, "random_same_sizes")
    sizes = sorted(pd.Series(labels).value_counts().tolist())
    assert sizes == [2, 3]


def test_permutation_context_rejoins_controlled_features():
    rows = []
    for app in range(4):
        for speed in (0.2, 0.4):
            rows.append(
                {
                    "vde_id": app,
                    "independent_application_id": f"A{app}",
                    "application_lineage": f"L{app}",
                    "speed_kph": speed * 100,
                    "observed_force_N": 100 + app + speed,
                    "speed_scaled": speed,
                    "speed_scaled_sq": speed**2,
                    "test_mass_kg": 1500.0,
                    "model_year": 2020.0,
                    "gear_count": 8.0,
                    "final_drive_ratio": 3.0,
                    "nv_ratio": 25.0,
                    "mass_speed": 1500 * speed,
                    "year_speed": 2020 * speed,
                    "gear_speed": 8 * speed,
                    "fdr_speed": 3 * speed,
                    "nv_speed": 25 * speed,
                    "make_norm": "BMW",
                    "model_norm": f"M{app}",
                    "category_norm": "CAR",
                    "drive_norm": "RWD",
                    "electrification": "ICE",
                    "transmission_type_norm": "AUTO",
                }
            )
    long = pd.DataFrame(rows)
    predictions = long[long.vde_id.eq(3)][
        ["vde_id", "independent_application_id", "application_lineage", "speed_kph", "observed_force_N"]
    ].copy()
    predictions["fold"] = 0
    predictions["model0_prediction_N"] = predictions.observed_force_N - 1
    folds = pd.DataFrame([{"fold": 0, "alpha_model0": 1.0, "lambda_model2": 1.0}])

    contexts = _fold_soft_context(long, predictions, folds)

    assert len(contexts) == 1
    assert contexts[0][1]["model0_prediction_N"].notna().all()
    assert set(CONTROLLED_NUMERIC_FEATURES).issubset(contexts[0][1].columns)


def test_sqlite_source_is_query_only_and_hash_is_unchanged(tmp_path: Path):
    db = tmp_path / "source.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO sample VALUES (1)")
    before = sha256_file(db)
    with _read_only_connection(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE forbidden (id INTEGER)")
    assert sha256_file(db) == before
