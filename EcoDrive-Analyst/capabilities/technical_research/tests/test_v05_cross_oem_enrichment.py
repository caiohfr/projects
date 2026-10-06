from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from capabilities.technical_research.cross_oem_v05 import (
    VALUE_FIELDS,
    deterministic_sample,
    enrich_application,
    file_sha256,
    grouping_identity,
    infer_electrification,
    load_canonical_candidates,
    match_status,
    open_read_only,
    select_value,
    transmission_architecture,
    validate_numeric_contract,
)
from capabilities.technical_research.semantic_cleanup_v041 import calculated_cda
from scripts.run_component_research_enrichment_v05 import ENRICHMENT_FIELDS


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_DB = ROOT / "data/db/staging/eco_drive_canonical_candidate.db"
FROZEN_BMW = ROOT / "artifacts/components/component_research_v041a/BMW_COMPONENT_RESEARCH_BENCHMARK_V041A.csv"
FROZEN_BMW_SHA256 = "C4B31774BE2AFB6EB543446FB8CA69CAECCF6E199275BCB143E1B6037919D0E3"


def sample_row() -> dict[str, object]:
    return {
        "sample_id": "V05-001",
        "vehicle_configuration_id": "CFG-1",
        "vde_id": 1,
        "make": "Ford",
        "model": "Example AWD",
        "model_year": 2025,
        "category": "Truck",
        "drive_type": "All Wheel Drive",
        "transmission_type": "Automatic",
        "gears": 8,
        "source_final_drive": 3.21,
        "nv_ratio": 24.0,
        "electrification": "ICE",
        "carryover_root_id": 1,
        "selection_reason": "test",
        "oem_group": "FORD_LINCOLN",
    }


def evidence(field: str, value: str, match: str = "PARTIAL") -> dict[str, str]:
    return {
        "sample_id": "V05-001",
        "make": "Ford",
        "model_pattern": "Example",
        "year_min": "2024",
        "year_max": "2026",
        "field": field,
        "value": value,
        "application_match": match,
        "source_url": "https://example.invalid/technical",
        "evidence_note": "Test evidence.",
    }


def test_researched_approx_outranks_rule_estimated() -> None:
    selected = select_value(researched_approx="10R80", rule_estimated="10-speed automatic")
    assert selected.value == "10R80"
    assert selected.provenance == "RESEARCHED_APPROX"
    assert selected.rule_estimated_value == "10-speed automatic"


def test_not_found_only_when_no_useful_identity_or_architecture() -> None:
    assert match_status(architecture="TORQUE_CONVERTER_AUTOMATIC") == "FOUND_ARCHITECTURE"
    assert match_status(family="10R80") == "FOUND_FAMILY"
    assert match_status() == "NOT_FOUND"


def test_architecture_and_family_remain_distinct() -> None:
    row, _ = enrich_application(sample_row(), [evidence("transmission_family", "10R80")])
    assert row["transmission_family"] == "10R80"
    assert row["transmission_architecture"] == "TORQUE_CONVERTER_AUTOMATIC"
    assert row["transmission_family"] != row["transmission_architecture"]


def test_grouping_precedence_is_code_then_family_then_architecture() -> None:
    assert grouping_identity({"transmission_code": "10R80-MHT", "transmission_family": "10R80", "transmission_architecture": "AUTO"}) == ("10R80-MHT", "EXACT_CODE")
    assert grouping_identity({"transmission_code": "", "transmission_family": "10R80", "transmission_architecture": "AUTO"}) == ("10R80", "FAMILY")
    assert grouping_identity({"transmission_code": "", "transmission_family": "", "transmission_architecture": "AUTO"}) == ("AUTO", "ARCHITECTURE")


def test_cda_calculation_is_direct_product() -> None:
    assert calculated_cda(0.29, 2.4) == pytest.approx(0.696)


def test_normalized_numeric_fields_are_parseable_after_csv_export(tmp_path: Path) -> None:
    row, _ = enrich_application(sample_row(), [evidence("cd", "0.29"), evidence("frontal_area_m2", "2.4")])
    path = tmp_path / "numeric.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with path.open("r", encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert validate_numeric_contract(exported)["valid"] is True
    assert float(exported[0]["cda_m2"]) == pytest.approx(0.696)


def test_rule_fallback_is_preserved_in_provenance_audit() -> None:
    row, audit = enrich_application(sample_row(), [])
    architecture = next(item for item in audit if item["field"] == "transmission_architecture")
    assert row["transmission_architecture"] == "TORQUE_CONVERTER_AUTOMATIC"
    assert architecture["provenance"] == "RULE_ESTIMATED"
    assert architecture["rule_estimated_value"] == "TORQUE_CONVERTER_AUTOMATIC"


def test_conflicting_evidence_is_not_silently_selected() -> None:
    row, _ = enrich_application(
        sample_row(),
        [evidence("transmission_family", "10R80"), evidence("transmission_family", "10L80")],
    )
    assert row["transmission_family"] == ""
    assert row["match_status"] == "CONFLICT"
    assert row["review_flag"] == "REVIEW_CONFLICT"


def test_frozen_bmw_benchmark_hash_is_unchanged() -> None:
    assert file_sha256(FROZEN_BMW) == FROZEN_BMW_SHA256


def test_read_only_connection_rejects_canonical_write(tmp_path: Path) -> None:
    db = tmp_path / "source.db"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE source (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    before = file_sha256(db)
    with open_read_only(db) as read_only:
        with pytest.raises(sqlite3.DatabaseError):
            read_only.execute("INSERT INTO source VALUES (1)")
    assert file_sha256(db) == before


def test_private_parasitic_loss_fields_are_not_part_of_v05_contract() -> None:
    assert not any("parasitic" in field.lower() or "trans_a" in field.lower() for field in VALUE_FIELDS)


def test_primary_enrichment_csv_has_unique_column_names() -> None:
    assert len(ENRICHMENT_FIELDS) == len(set(ENRICHMENT_FIELDS))


@pytest.mark.parametrize(
    ("reported", "propulsion", "model", "expected"),
    [
        (None, "Electricity", "F-150 Lightning", "BEV"),
        (None, "Tier 2 Cert Gasoline", "ES 300h", "HEV"),
        (None, "Tier 2 Cert Gasoline", "NX 450h+", "PHEV"),
        (None, "Hydrogen 5", "NEXO", "FCEV"),
    ],
)
def test_electrification_inference_preserves_explicit_powertrain_semantics(
    reported: str | None, propulsion: str, model: str, expected: str
) -> None:
    assert infer_electrification(reported, propulsion, propulsion, model) == expected


@pytest.mark.skipif(not CANDIDATE_DB.exists(), reason="canonical candidate not available")
def test_real_sample_is_50_non_bmw_applications_across_five_groups() -> None:
    sample = deterministic_sample(load_canonical_candidates(CANDIDATE_DB))
    assert len(sample) == 50
    assert len({row["oem_group"] for row in sample}) == 5
    assert all(str(row["make"]).upper() != "BMW" for row in sample)
    assert len({row["sample_id"] for row in sample}) == 50
