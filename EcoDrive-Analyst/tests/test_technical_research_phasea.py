from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from src.vde_core.technical_research_phasea import (
    apply_combined_results,
    extract_curated_public_evidence,
    extract_internal_roadload_evidence,
    file_sha256,
    inspect_database_read_only,
    prepare_phase_a,
    validate_evidence_record,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_read_only_audit_does_not_change_database(tmp_path: Path) -> None:
    path = tmp_path / "candidate.db"
    with sqlite3.connect(path) as connection:
        for table in ("vde", "component_db", "component_resolution", "vde_component_resolution"):
            connection.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")
    before = file_sha256(path)
    audit = inspect_database_read_only(path)
    assert audit["quick_check"] == "ok"
    assert audit["foreign_key_issue_count"] == 0
    assert audit["hash_unchanged"] is True
    assert file_sha256(path) == before


def test_multiple_cluster_is_split_by_actual_model(tmp_path: Path, monkeypatch) -> None:
    inventory = tmp_path / "inventory.csv"
    groups = tmp_path / "groups.csv"
    clusters = tmp_path / "clusters.csv"
    base = {
        "vde_id": "1", "vehicle_configuration_id": "VC1", "gap_family": "ARCHITECTURE_UNRESOLVED",
        "gap_domain": "POWERTRAIN", "make": "TOYOTA", "model": "COROLLA", "model_year": "2024",
        "drive_type": "2-Wheel Drive, Front", "transmission_type": "Automatic", "gears": "8",
        "engine_code": "01", "transmission_code": "", "propulsion_architecture": "ICE",
        "architecture_class": "UNRESOLVED", "boundary_status": "", "current_confidence": "UNRESOLVED",
    }
    rows = [base, {**base, "vde_id": "2", "vehicle_configuration_id": "VC2", "model": "CAMRY"}]
    _write_csv(inventory, rows)
    monkeypatch.setattr("src.vde_core.technical_research_phasea._group_id", lambda row: "RG-X")
    _write_csv(groups, [{"research_group_id": "RG-X", "research_cluster_id": "RTC-X", "gap_family": "ARCHITECTURE_UNRESOLVED", "domain": "POWERTRAIN", "vde_count": "2"}])
    _write_csv(clusters, [{"research_cluster_id": "RTC-X", "p0_group_count": "1", "gap_families": "ARCHITECTURE_UNRESOLVED", "make": "TOYOTA"}])
    result = prepare_phase_a(inventory_path=inventory, p0_groups_path=groups, p0_clusters_path=clusters)
    assert len(result.child_tasks) == 2
    assert {row["model"] for row in result.child_tasks} == {"COROLLA", "CAMRY"}


def test_real_p0_batch_is_scoped_split_and_traceable() -> None:
    package_inputs = ROOT / "inputs/EcoDrive_Sprint12_Technical_Research_Agent_PhaseA_P0_v1.0/inputs"
    result = prepare_phase_a(
        inventory_path=ROOT / "artifacts/canonical_component_population/research_gap_vde_inventory.csv",
        p0_groups_path=package_inputs / "p0_research_groups.csv",
        p0_clusters_path=package_inputs / "p0_technical_clusters.csv",
    )
    assert len(result.clusters) == 12
    assert len(result.groups) == 28
    assert len(result.child_tasks) == 131
    toyota_01_models = {
        task["model"] for task in result.child_tasks
        if task["make"].upper() == "TOYOTA" and task["engine_code_raw"] == "01"
    }
    assert len(toyota_01_models) > 5
    assert all(task["model"] != "MULTIPLE" for task in result.child_tasks)


def test_real_evidence_is_complete_read_only_and_conservative() -> None:
    package_inputs = ROOT / "inputs/EcoDrive_Sprint12_Technical_Research_Agent_PhaseA_P0_v1.0/inputs"
    result = prepare_phase_a(
        inventory_path=ROOT / "artifacts/canonical_component_population/research_gap_vde_inventory.csv",
        p0_groups_path=package_inputs / "p0_research_groups.csv",
        p0_clusters_path=package_inputs / "p0_technical_clusters.csv",
    )
    db = ROOT / "data/db/staging/eco_drive_canonical_candidate.db"
    before = file_sha256(db)
    internal, _, complete = extract_internal_roadload_evidence(db_path=db, preparation=result)
    external, _, _ = extract_curated_public_evidence(
        catalog_path=ROOT / "data/validation/phasea_p0_public_evidence.json",
        preparation=result,
    )
    for record in internal + external:
        validate_evidence_record(record)
        assert record["source_url_or_id"]
        assert record["evidence_locator"]
        assert record["supporting_excerpt_short"]
    assert len(internal) == 1138
    assert len(external) == 201
    _, groups, _, _ = apply_combined_results(
        result, internal_evidence=internal, external_evidence=external,
        complete_by_group=complete,
    )
    states = {row["gap_family"]: row["final_status"] for row in groups if row["gap_family"] == "TRANSMISSION_AXLE_BOUNDARY_UNKNOWN"}
    assert states == {"TRANSMISSION_AXLE_BOUNDARY_UNKNOWN": "BOUNDARY_STILL_UNKNOWN"}
    assert sum(row["final_status"] == "RESOLVED_EXACT" for row in groups) == 5
    assert file_sha256(db) == before
