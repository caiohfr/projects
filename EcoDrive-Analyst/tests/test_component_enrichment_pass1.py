from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sqlite3

from src.vde_core.component_enrichment_pass1 import execute_pass1, file_sha256


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE vehicle_configuration (
    vehicle_configuration_id TEXT PRIMARY KEY,
    program_id TEXT NOT NULL,
    drive_system TEXT,
    transmission_type TEXT,
    transmission_model TEXT,
    gear_count INTEGER,
    final_drive_ratio REAL,
    engine_model TEXT,
    engine_type TEXT,
    propulsion_architecture TEXT,
    architecture_properties_json TEXT,
    source_identity_json TEXT NOT NULL,
    record_status TEXT NOT NULL DEFAULT 'ACTIVE'
);
CREATE TABLE vde (
    id INTEGER PRIMARY KEY,
    vehicle_configuration_id TEXT NOT NULL REFERENCES vehicle_configuration(vehicle_configuration_id),
    coast_A_N REAL,
    coast_B_N_per_kph REAL,
    coast_C_N_per_kph2 REAL,
    make TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER,
    category TEXT NOT NULL,
    legislation TEXT NOT NULL,
    source_name TEXT,
    source_record_id TEXT,
    record_status TEXT NOT NULL DEFAULT 'ACTIVE'
);
CREATE TABLE run (
    run_id TEXT PRIMARY KEY,
    vde_id INTEGER NOT NULL REFERENCES vde(id),
    run_type TEXT NOT NULL,
    procedure_description TEXT,
    conditions_json TEXT,
    record_status TEXT NOT NULL DEFAULT 'ACTIVE'
);
CREATE TABLE fuelcons (
    id INTEGER PRIMARY KEY,
    vde_id INTEGER NOT NULL REFERENCES vde(id),
    electrification TEXT NOT NULL,
    record_status TEXT NOT NULL DEFAULT 'ACTIVE'
);
CREATE TABLE component_db (component_id TEXT PRIMARY KEY);
CREATE TABLE component_instance (component_instance_id TEXT PRIMARY KEY);
CREATE TABLE component_resolution (
    boundary TEXT NOT NULL,
    component_resolution_id TEXT PRIMARY KEY,
    conditions_json TEXT,
    confidence TEXT,
    fidelity_level TEXT,
    input_component_instance_ids_json TEXT,
    method TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    resolved_A_N REAL,
    resolved_B_N_per_kph REAL,
    resolved_C_N_per_kph2 REAL,
    source_run_ids_json TEXT,
    vehicle_configuration_id TEXT REFERENCES vehicle_configuration(vehicle_configuration_id),
    record_status TEXT NOT NULL DEFAULT 'ACTIVE',
    review_status TEXT NOT NULL DEFAULT 'CURRENT',
    estimate_status TEXT,
    estimator_version TEXT,
    fit_nrmse_pct REAL,
    condition_number REAL,
    sensitivity_rel_pct REAL
);
CREATE TABLE vde_component_resolution (
    vde_id INTEGER NOT NULL REFERENCES vde(id),
    component_resolution_id TEXT NOT NULL REFERENCES component_resolution(component_resolution_id),
    boundary TEXT NOT NULL,
    adoption_role TEXT NOT NULL CHECK (adoption_role IN ('ADOPTED', 'SUPPORTING', 'SUPERSEDED')),
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (vde_id, component_resolution_id, boundary)
);
"""


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany(
        """INSERT INTO vehicle_configuration
        (vehicle_configuration_id,program_id,drive_system,transmission_type,
         transmission_model,gear_count,final_drive_ratio,engine_model,engine_type,
         propulsion_architecture,architecture_properties_json,source_identity_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            ("CFG-ICE", "P1", "2-Wheel Drive, Front", "Semi-Automatic", None, 8, 3.8, "KCHCXQNA07", "Gasoline", "Tier 2 Cert Gasoline", None, "{}"),
            ("CFG-HYBRID", "P2", "All Wheel Drive", "Continuously Variable", None, None, None, "H1", "Hybrid", "OVC-HEV", None, "{}"),
        ],
    )
    conn.executemany(
        """INSERT INTO vde
        (id,vehicle_configuration_id,coast_A_N,coast_B_N_per_kph,coast_C_N_per_kph2,
         make,model,year,category,legislation,source_name,source_record_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (1, "CFG-ICE", 117.61097950748761, 0.3059744422630198, 0.044997660743973794, "Ford", "Transit", 2020, "Truck", "EPA", "FIXTURE", "V1"),
            (2, "CFG-HYBRID", 150.0, 0.4, 0.04, "Example", "Hybrid", 2024, "Car", "EPA", "FIXTURE", "V2"),
        ],
    )
    set_abc = {
        "set_abc_native": {
            "Set Coef A (lbf)": 7.62,
            "Set Coef B (lbf/mph)": 0.15285,
            "Set Coef C (lbf/mph**2)": 0.02442,
        }
    }
    conn.executemany(
        "INSERT INTO run (run_id,vde_id,run_type,procedure_description,conditions_json) VALUES (?,?,?,?,?)",
        [
            ("RUN-1", 1, "TEST", "FTP", json.dumps(set_abc)),
            ("RUN-2", 1, "TEST", "HWFE", json.dumps(set_abc)),
        ],
    )
    conn.executemany(
        "INSERT INTO fuelcons (id,vde_id,electrification) VALUES (?,?,?)",
        [(1, 1, "ICE"), (2, 2, "PHEV")],
    )
    conn.commit()
    conn.close()


def _count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as conn:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def test_dry_run_is_read_only_and_emits_required_audit_files(tmp_path: Path) -> None:
    db = tmp_path / "candidate.db"
    output = tmp_path / "dry_run"
    _build_db(db)
    before = file_sha256(db)

    result = execute_pass1(db, output, write=False)

    assert file_sha256(db) == before
    assert result["coverage_before"] == {"eligible": 6, "A": 0, "B": 0}
    assert result["coverage_after"]["A"] == 3
    assert result["coverage_after"]["B"] == 0
    assert result["db_integrity"]["external_search_count"] == 0
    assert _count(db, "component_resolution") == 0
    required = {
        "coverage_before.csv", "coverage_after.csv", "vde_enrichment_outcomes.csv",
        "estimator_summary.csv", "reason_code_summary.csv",
        "unresolved_unique_groups.csv", "unresolved_group_members.csv",
        "pass1_summary.md", "db_integrity.txt", "execution_manifest.json",
    }
    assert required.issubset({item.name for item in output.iterdir()})


def test_write_preserves_vde_totals_and_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "candidate.db"
    _build_db(db)
    with sqlite3.connect(db) as conn:
        totals_before = conn.execute(
            "SELECT id,coast_A_N,coast_B_N_per_kph,coast_C_N_per_kph2 FROM vde ORDER BY id"
        ).fetchall()

    first = execute_pass1(db, tmp_path / "first", write=True)
    second = execute_pass1(db, tmp_path / "second", write=True)

    assert first["inserted_resolution_rows"] == 3
    assert first["inserted_links"] == 3
    assert second["inserted_resolution_rows"] == 0
    assert second["inserted_links"] == 0
    assert _count(db, "component_resolution") == 3
    assert _count(db, "vde_component_resolution") == 3
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT id,coast_A_N,coast_B_N_per_kph,coast_C_N_per_kph2 FROM vde ORDER BY id"
        ).fetchall() == totals_before
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_persisted_rows_are_aggregate_estimates_with_explicit_provenance(tmp_path: Path) -> None:
    db = tmp_path / "candidate.db"
    _build_db(db)
    execute_pass1(db, tmp_path / "write", write=True)

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("SELECT * FROM component_resolution ORDER BY boundary")]
        links = conn.execute("SELECT DISTINCT adoption_role FROM vde_component_resolution").fetchall()

    assert {row["boundary"] for row in rows} == {"AERO", "ROLLING_MINOR", "DRIVETRAIN_AGGREGATE"}
    assert {row["estimate_status"] for row in rows} == {"SUPPORTED"}
    assert {row["confidence"] for row in rows} == {"HIGH"}
    assert {row["fidelity_level"] for row in rows} == {"L2"}
    assert [tuple(row) for row in links] == [("ADOPTED",)]
    for row in rows:
        provenance = json.loads(row["provenance_json"])
        assert provenance["provenance_class"] == "MODEL_ESTIMATED"
        assert provenance["authoritative_vde_total_unchanged"] is True
        assert provenance["model_type"] == "MOSKALIK_2020"


def test_temp_copy_execution_does_not_change_source_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    execution = tmp_path / "execution.db"
    _build_db(source)
    shutil.copy2(source, execution)
    source_hash = file_sha256(source)

    result = execute_pass1(execution, tmp_path / "temp_write", write=True, source_db_path=source)

    assert file_sha256(source) == source_hash
    assert result["db_integrity"]["source_db_sha256_after"] == source_hash
    assert file_sha256(execution) != source_hash
    with (tmp_path / "temp_write" / "unresolved_unique_groups.csv").open(encoding="utf-8-sig", newline="") as handle:
        unresolved = list(csv.DictReader(handle))
    assert unresolved
    assert unresolved[0]["represented_vde_count"] == "1"
