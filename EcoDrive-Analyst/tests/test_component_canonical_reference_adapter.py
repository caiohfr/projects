from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from src.vde_core import db as db_module
from src.vde_core.component_repositories import load_component_repository, lookup_component
from src.vde_core.database_management_service import browse_records, get_record


def _create_reference_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE component_db (
                component_id TEXT PRIMARY KEY,
                component_domain TEXT NOT NULL,
                model TEXT,
                hardware_reference TEXT,
                custom_properties_json TEXT,
                provenance_json TEXT NOT NULL,
                source_name TEXT,
                source_record_id TEXT,
                source_file_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT,
                record_status TEXT DEFAULT 'ACTIVE',
                review_status TEXT DEFAULT 'CURRENT'
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
                provenance_json TEXT NOT NULL,
                record_status TEXT DEFAULT 'ACTIVE'
            );
            """
        )
        rows = (
            ("SYN_BRAKE", "BRAKE", "BRAKE", "brake"),
            ("SYN_TRANS", "TRANSMISSION", "TRANSMISSION", "transmission"),
            ("SYN_HUB", "AXLE_HUBS", "HUB_BEARING", "axle_hubs"),
        )
        for ordinal, (component_id, canonical_domain, source_domain, _) in enumerate(rows, start=1):
            resolution_id = f"CR_{component_id}"
            conn.execute(
                """
                INSERT INTO component_db (
                    component_id,component_domain,model,hardware_reference,
                    custom_properties_json,provenance_json,source_name,
                    source_file_version,record_status,review_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    component_id,
                    canonical_domain,
                    f"Reference {source_domain}",
                    "SYNTHETIC_REFERENCE",
                    json.dumps(
                        {
                            "drive_architecture": "AWD",
                            "position": "FRONT" if source_domain == "HUB_BEARING" else None,
                            "synthetic_reference_resolution_ids": [resolution_id],
                        }
                    ),
                    json.dumps(
                        {
                            "synthetic_reference": True,
                            "seed_component_domain_original": source_domain,
                        }
                    ),
                    "SYNTHETIC_REFERENCE",
                    "synthetic_components_v1",
                    "ACTIVE",
                    "REFERENCE",
                ),
            )
            conn.execute(
                """
                INSERT INTO component_resolution (
                    component_resolution_id,boundary,method,confidence,fidelity_level,
                    resolved_A_N,resolved_B_N_per_kph,resolved_C_N_per_kph2,
                    provenance_json,record_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    resolution_id,
                    source_domain,
                    "SYNTHETIC_POPULATION_REFERENCE_V1",
                    None,
                    None,
                    float(ordinal),
                    float(ordinal) / 10,
                    float(ordinal) / 100,
                    json.dumps(
                        {
                            "synthetic_reference": True,
                            "seed_confidence_original": "REFERENCE",
                            "seed_fidelity_level_original": "SYNTHETIC_POPULATION",
                        }
                    ),
                    "ACTIVE",
                ),
            )


def test_canonical_repository_uses_reference_resolution_abc(tmp_path: Path) -> None:
    db_path = tmp_path / "canonical.db"
    _create_reference_db(db_path)

    with db_module.using_db_path(db_path, legacy_fixture=False):
        brake = lookup_component("brake", "SYN_BRAKE")
        transmission = load_component_repository("transmission")
        axle = lookup_component("axle_hubs", "SYN_HUB")

    assert brake["found"] is True
    assert brake["source"] == "sqlite_component_db"
    assert brake["component"]["brake_A"] == 1.0
    assert brake["component"]["component_resolution_id"] == "CR_SYN_BRAKE"
    assert transmission.get_by_id("SYN_TRANS")["trans_C"] == 0.02
    assert axle["component"]["component_type"] == "HUB_BEARING"
    assert axle["component"]["physical_boundary"] == "HUB_BEARING"
    assert axle["component"]["axle_hubs_A"] == 3.0
    assert axle["issues"] == []


def test_database_management_browses_canonical_reference_and_resolution(tmp_path: Path) -> None:
    db_path = tmp_path / "canonical.db"
    _create_reference_db(db_path)

    with db_module.using_db_path(db_path, legacy_fixture=False), patch(
        "src.vde_core.database_management_service.db_module.ensure_db"
    ):
        rows = browse_records("COMPONENT", component_domain="axle_hubs")
        detail = get_record("COMPONENT", "SYN_HUB", component_domain="axle_hubs")

    assert len(rows) == 1
    assert rows[0]["component_code"] == "SYN_HUB"
    assert rows[0]["component_resolution_id"] == "CR_SYN_HUB"
    assert rows[0]["resolution_boundary"] == "HUB_BEARING"
    assert rows[0]["equivalent_A_N"] == 3.0
    assert detail["record_origin"] == "IMPORTED"
    assert detail["resolution_method"] == "SYNTHETIC_POPULATION_REFERENCE_V1"
    assert json.loads(detail["resolution_provenance_json"])["seed_confidence_original"] == "REFERENCE"
