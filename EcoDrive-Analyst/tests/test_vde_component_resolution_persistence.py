from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.vde_core import db as db_module
from src.vde_core.vde_request_compact_persistence import (
    _persist_component_resolution_lineage,
)
from src.vde_core.repositories.vde_repository import fetch_vde_component_resolutions


def _canonical_lineage_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE vehicle_configuration (
                vehicle_configuration_id TEXT PRIMARY KEY
            );
            INSERT INTO vehicle_configuration VALUES ('VC-1');
            CREATE TABLE vde (
                id INTEGER PRIMARY KEY,
                vehicle_configuration_id TEXT REFERENCES vehicle_configuration(vehicle_configuration_id)
            );
            INSERT INTO vde VALUES (101, 'VC-1');
            INSERT INTO vde VALUES (102, 'VC-1');
            INSERT INTO vde VALUES (103, 'VC-1');
            CREATE TABLE component_db (
                component_id TEXT PRIMARY KEY,
                custom_properties_json TEXT
            );
            CREATE TABLE component_resolution (
                component_resolution_id TEXT PRIMARY KEY,
                boundary TEXT NOT NULL,
                method TEXT NOT NULL,
                confidence TEXT,
                fidelity_level TEXT,
                resolved_A_N REAL,
                resolved_B_N_per_kph REAL,
                resolved_C_N_per_kph2 REAL,
                conditions_json TEXT,
                input_component_instance_ids_json TEXT,
                source_run_ids_json TEXT,
                vehicle_configuration_id TEXT REFERENCES vehicle_configuration(vehicle_configuration_id),
                provenance_json TEXT NOT NULL,
                estimate_status TEXT,
                record_status TEXT NOT NULL DEFAULT 'ACTIVE',
                review_status TEXT NOT NULL DEFAULT 'CURRENT'
            );
            CREATE TABLE vde_component_resolution (
                vde_id INTEGER NOT NULL REFERENCES vde(id),
                component_resolution_id TEXT NOT NULL REFERENCES component_resolution(component_resolution_id),
                boundary TEXT NOT NULL,
                adoption_role TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (vde_id,component_resolution_id,boundary)
            );
            """
        )
        conn.execute(
            """
            INSERT INTO component_resolution (
                component_resolution_id,boundary,method,resolved_A_N,
                resolved_B_N_per_kph,resolved_C_N_per_kph2,provenance_json
            ) VALUES ('CR_REF_BRAKE','BRAKE','SYNTHETIC_POPULATION_REFERENCE_V1',1,0.1,0.01,'{}')
            """
        )
        conn.execute(
            """
            INSERT INTO component_resolution (
                component_resolution_id,boundary,method,provenance_json,estimate_status
            ) VALUES ('CR_REJECTED','BRAKE','ESTIMATOR','{}','NOT_IDENTIFIABLE')
            """
        )


def test_exact_reference_is_adopted_without_duplicate_resolution(tmp_path: Path) -> None:
    db_path = tmp_path / "canonical.db"
    _canonical_lineage_db(db_path)
    proposal = {
        "proposal_id": "requested_1",
        "component_actions": [
            {
                "action": "reuse_existing",
                "domain": "brake",
                "component_snapshot": {
                    "component_resolution_id": "CR_REF_BRAKE",
                    "resolution_boundary": "BRAKE",
                },
            }
        ],
        "domain_results": {"brake": {"proposal_type": "BRAKE_METADATA_ONLY", "status": "OK"}},
        "resolved_snapshot": {},
    }

    with db_module.using_db_path(db_path, legacy_fixture=False), sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        adopted = _persist_component_resolution_lineage(
            conn,
            vde_id=101,
            request_history_id=7,
            proposal_result=proposal,
            row_payload={},
        )
        conn.commit()
        resolution_count = conn.execute("SELECT COUNT(*) FROM component_resolution").fetchone()[0]
        preserved_resolution = conn.execute(
            "SELECT method,provenance_json,resolved_A_N,resolved_B_N_per_kph,resolved_C_N_per_kph2 "
            "FROM component_resolution WHERE component_resolution_id='CR_REF_BRAKE'"
        ).fetchone()
        link = conn.execute(
            "SELECT component_resolution_id,boundary FROM vde_component_resolution WHERE vde_id=101"
        ).fetchone()

    assert adopted == [
        {
            "vde_id": 101,
            "component_resolution_id": "CR_REF_BRAKE",
            "boundary": "BRAKE",
            "created": False,
        }
    ]
    assert resolution_count == 2
    assert preserved_resolution == (
        "SYNTHETIC_POPULATION_REFERENCE_V1",
        "{}",
        1.0,
        0.1,
        0.01,
    )
    assert link == ("CR_REF_BRAKE", "BRAKE")


def test_adopted_resolution_is_available_as_reload_baseline(tmp_path: Path) -> None:
    db_path = tmp_path / "canonical.db"
    _canonical_lineage_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO vde_component_resolution VALUES (101,'CR_REF_BRAKE','BRAKE','ADOPTED',0)"
        )

    with db_module.using_db_path(db_path, legacy_fixture=False):
        rows = fetch_vde_component_resolutions(101)

    assert len(rows) == 1
    assert rows[0]["component_resolution_id"] == "CR_REF_BRAKE"
    assert rows[0]["boundary"] == "BRAKE"
    assert rows[0]["method"] == "SYNTHETIC_POPULATION_REFERENCE_V1"
    assert rows[0]["resolved_A_N"] == 1.0


def test_custom_brake_creates_scenario_resolution_without_fake_component(tmp_path: Path) -> None:
    db_path = tmp_path / "canonical.db"
    _canonical_lineage_db(db_path)
    proposal = {
        "proposal_id": "requested_2",
        "component_actions": [
            {
                "action": "eligible_for_new_component",
                "domain": "brake",
                "component_snapshot": {},
            }
        ],
        "domain_results": {"brake": {"proposal_type": "BRAKE_ABSOLUTE", "status": "OK", "requested_values": {}}},
        "resolved_snapshot": {"brake_A": 4.0, "brake_B": 0.2, "brake_C": 0.03},
    }

    with db_module.using_db_path(db_path, legacy_fixture=False), sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        before_components = conn.execute("SELECT COUNT(*) FROM component_db").fetchone()[0]
        adopted = _persist_component_resolution_lineage(
            conn,
            vde_id=102,
            request_history_id=8,
            proposal_result=proposal,
            row_payload={},
        )
        conn.commit()
        after_components = conn.execute("SELECT COUNT(*) FROM component_db").fetchone()[0]
        resolution = conn.execute(
            """
            SELECT boundary,method,resolved_A_N,resolved_B_N_per_kph,
                   resolved_C_N_per_kph2,vehicle_configuration_id,provenance_json
            FROM component_resolution WHERE component_resolution_id='CR_VDE_102_BRAKE'
            """
        ).fetchone()

    assert before_components == after_components == 0
    assert adopted[0]["created"] is True
    assert resolution[:6] == ("BRAKE", "USER_INPUT", 4.0, 0.2, 0.03, "VC-1")
    provenance = json.loads(resolution[6])
    assert provenance["request_history_id"] == 8
    assert provenance["proposal_id"] == "requested_2"


def test_rejected_or_not_identifiable_resolution_cannot_be_adopted(tmp_path: Path) -> None:
    db_path = tmp_path / "canonical.db"
    _canonical_lineage_db(db_path)
    proposal = {
        "proposal_id": "requested_3",
        "component_actions": [
            {
                "action": "reuse_existing",
                "domain": "brake",
                "component_snapshot": {"component_resolution_id": "CR_REJECTED"},
            }
        ],
        "domain_results": {"brake": {"status": "OK"}},
    }

    with db_module.using_db_path(db_path, legacy_fixture=False), sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(ValueError, match="NOT_IDENTIFIABLE"):
            _persist_component_resolution_lineage(
                conn,
                vde_id=103,
                request_history_id=9,
                proposal_result=proposal,
                row_payload={},
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM vde_component_resolution WHERE vde_id=103"
        ).fetchone()[0] == 0


def test_review_domain_does_not_adopt_reference_that_was_not_materialized(tmp_path: Path) -> None:
    db_path = tmp_path / "canonical.db"
    _canonical_lineage_db(db_path)
    proposal = {
        "proposal_id": "requested_4",
        "component_actions": [
            {
                "action": "reuse_existing",
                "domain": "brake",
                "component_snapshot": {"component_resolution_id": "CR_REF_BRAKE"},
            }
        ],
        "domain_results": {"brake": {"status": "Review"}},
    }

    with db_module.using_db_path(db_path, legacy_fixture=False), sqlite3.connect(db_path) as conn:
        adopted = _persist_component_resolution_lineage(
            conn,
            vde_id=103,
            request_history_id=10,
            proposal_result=proposal,
            row_payload={},
        )
        assert adopted == []
        assert conn.execute(
            "SELECT COUNT(*) FROM vde_component_resolution WHERE vde_id=103"
        ).fetchone()[0] == 0


def test_missing_baseline_lookup_cannot_persist_lineage_silently(tmp_path: Path) -> None:
    db_path = tmp_path / "canonical.db"
    _canonical_lineage_db(db_path)
    proposal = {
        "proposal_id": "requested_missing_baseline",
        "component_actions": [
            {
                "action": "reuse_existing",
                "domain": "brake",
                "component_snapshot": {"component_resolution_id": "CR_REF_BRAKE"},
                "recalculate_total_abc": True,
            }
        ],
        "domain_results": {
            "brake": {
                "status": "Missing",
                "issues": [{"code": "REQUIRES_USER_BASELINE_INPUT"}],
            }
        },
    }

    with db_module.using_db_path(db_path, legacy_fixture=False), sqlite3.connect(db_path) as conn:
        adopted = _persist_component_resolution_lineage(
            conn,
            vde_id=103,
            request_history_id=11,
            proposal_result=proposal,
            row_payload={},
        )

        assert adopted == []
        assert conn.execute(
            "SELECT COUNT(*) FROM vde_component_resolution WHERE vde_id=103"
        ).fetchone()[0] == 0
