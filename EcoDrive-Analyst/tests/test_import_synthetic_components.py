from __future__ import annotations

import csv
import io
import json
import sqlite3
import zipfile
from pathlib import Path

from scripts.import_synthetic_components import KPI_COLUMNS, run_import, sha256
from scripts.verify_component_db_integration import verify


SOURCE_DOMAINS = (
    [("BRAKE", index) for index in range(12)]
    + [("TRANSMISSION", index) for index in range(14)]
    + [("AXLE", index) for index in range(14)]
    + [("HUB_BEARING", index) for index in range(10)]
)


def _csv_bytes(fieldnames: list[str], rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _seed_zip(path: Path) -> None:
    components = []
    resolutions = []
    catalog = []
    manifest = []
    qa = []
    for ordinal, (domain, domain_index) in enumerate(SOURCE_DOMAINS, start=1):
        component_id = f"SYN_{domain}_{domain_index:02d}"
        resolution_id = f"CR_{component_id}"
        components.append(
            {
                "component_id": component_id,
                "component_domain": domain,
                "manufacturer": "",
                "model": f"Synthetic {domain} {domain_index}",
                "custom_properties_json": json.dumps({"seed": True}),
                "provenance_json": json.dumps({"population_n": ordinal}),
                "source_name": "SYNTHETIC_ENGINEERING_REFERENCE",
                "record_status": "ACTIVE",
                "review_status": "REFERENCE",
            }
        )
        resolutions.append(
            {
                "component_resolution_id": resolution_id,
                "boundary": domain,
                "method": "SYNTHETIC_POPULATION_REFERENCE_V1",
                "confidence": "REFERENCE",
                "fidelity_level": "SYNTHETIC_POPULATION",
                "resolved_A_N": str(float(ordinal)),
                "resolved_B_N_per_kph": "0.1",
                "resolved_C_N_per_kph2": "0.01",
                "conditions_json": "{}",
                "input_component_instance_ids_json": "[]",
                "source_run_ids_json": "[]",
                "vehicle_configuration_id": "",
                "provenance_json": json.dumps({"component_id": component_id}),
                "record_status": "ACTIVE",
                "review_status": "REFERENCE",
            }
        )
        catalog.append(
            {
                "component_id": component_id,
                "component_resolution_id": resolution_id,
                "domain": domain,
            }
        )
        manifest.append(
            {
                "component_id": component_id,
                "component_resolution_id": resolution_id,
            }
        )
        qa.append(
            {
                "component_id": component_id,
                "qa_status": "PASS",
                "source_identifiers_exported": "False",
            }
        )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", "Synthetic test seed")
        archive.writestr(
            "component_db_seed.csv",
            _csv_bytes(list(components[0]), components),
        )
        archive.writestr(
            "component_resolution_seed.csv",
            _csv_bytes(list(resolutions[0]), resolutions),
        )
        archive.writestr(
            "synthetic_reference_catalog.csv",
            _csv_bytes(list(catalog[0]), catalog),
        )
        archive.writestr(
            "component_resolution_manifest.csv",
            _csv_bytes(list(manifest[0]), manifest),
        )
        archive.writestr("qa_summary.csv", _csv_bytes(list(qa[0]), qa))


def _canonical_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE vehicle_configuration (
                vehicle_configuration_id TEXT PRIMARY KEY
            );
            CREATE TABLE vde (
                id INTEGER PRIMARY KEY
            );
            CREATE TABLE component_db (
                component_id TEXT NOT NULL PRIMARY KEY,
                component_domain TEXT NOT NULL CHECK (
                    component_domain IN (
                        'ENGINE','TRANSMISSION','EMOTOR','BATTERY','BRAKE',
                        'AXLE_HUBS','PARASITIC','OTHER'
                    )
                ),
                manufacturer TEXT,
                model TEXT,
                custom_properties_json TEXT CHECK (
                    custom_properties_json IS NULL OR json_valid(custom_properties_json)
                ),
                provenance_json TEXT NOT NULL CHECK (json_valid(provenance_json)),
                source_name TEXT,
                record_status TEXT NOT NULL DEFAULT 'ACTIVE',
                review_status TEXT NOT NULL DEFAULT 'CURRENT',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT
            );
            CREATE TABLE component_instance (
                component_instance_id TEXT PRIMARY KEY,
                component_id TEXT REFERENCES component_db(component_id)
            );
            INSERT INTO component_instance(component_instance_id) VALUES ('EXISTING_INSTANCE');
            CREATE TABLE component_resolution (
                component_resolution_id TEXT NOT NULL PRIMARY KEY,
                boundary TEXT NOT NULL,
                method TEXT NOT NULL,
                confidence TEXT CHECK (
                    confidence IS NULL OR confidence IN ('LOW','MEDIUM','HIGH')
                ),
                fidelity_level TEXT CHECK (
                    fidelity_level IS NULL OR fidelity_level IN ('L0','L1','L2','L3')
                ),
                resolved_A_N REAL,
                resolved_B_N_per_kph REAL,
                resolved_C_N_per_kph2 REAL,
                conditions_json TEXT,
                input_component_instance_ids_json TEXT,
                source_run_ids_json TEXT,
                vehicle_configuration_id TEXT REFERENCES vehicle_configuration(vehicle_configuration_id),
                provenance_json TEXT NOT NULL CHECK (json_valid(provenance_json)),
                record_status TEXT NOT NULL DEFAULT 'ACTIVE',
                review_status TEXT NOT NULL DEFAULT 'CURRENT',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT
            );
            CREATE TABLE vde_component_resolution (
                vde_id INTEGER NOT NULL REFERENCES vde(id),
                component_resolution_id TEXT NOT NULL REFERENCES component_resolution(component_resolution_id),
                boundary TEXT NOT NULL,
                adoption_role TEXT NOT NULL DEFAULT 'ADOPTED',
                ordinal INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (vde_id, component_resolution_id, boundary)
            );
            """
        )


def test_dry_run_preserves_database_hash(tmp_path: Path) -> None:
    db_path = tmp_path / "candidate.db"
    seed_path = tmp_path / "seed.zip"
    _canonical_db(db_path)
    _seed_zip(seed_path)
    before = sha256(db_path)

    report = run_import(db_path, seed_path)

    assert report["mode"] == "dry-run"
    assert report["component_db"] == {"inserted": 50, "skipped_existing": 0}
    assert report["component_resolution"] == {"inserted": 50, "skipped_existing": 0}
    assert sha256(db_path) == before


def test_conservative_adapter_import_and_idempotence(tmp_path: Path) -> None:
    db_path = tmp_path / "candidate.db"
    seed_path = tmp_path / "seed.zip"
    _canonical_db(db_path)
    _seed_zip(seed_path)

    first = run_import(db_path, seed_path, apply=True)
    second = run_import(db_path, seed_path, apply=True)
    verification = verify(db_path)

    assert first["component_db"] == {"inserted": 50, "skipped_existing": 0}
    assert first["component_resolution"] == {"inserted": 50, "skipped_existing": 0}
    assert second["component_db"] == {"inserted": 0, "skipped_existing": 50}
    assert second["component_resolution"] == {"inserted": 0, "skipped_existing": 50}
    assert first["row_counts_before"]["component_instance"] == 1
    assert second["row_counts_after"]["component_instance"] == 1
    assert verification["ok"] is True
    assert verification["synthetic_component_domain_counts"] == {
        "AXLE_HUBS": 24,
        "BRAKE": 12,
        "TRANSMISSION": 14,
    }
    assert verification["preserved_source_domain_counts"] == {
        "AXLE": 14,
        "BRAKE": 12,
        "HUB_BEARING": 10,
        "TRANSMISSION": 14,
    }
    assert verification["synthetic_resolution_boundary_counts"] == {
        "AXLE": 14,
        "BRAKE": 12,
        "HUB_BEARING": 10,
        "TRANSMISSION": 14,
    }
    assert verification["resolution_rows_with_null_canonical_enums"] == 50
    assert verification["resolution_rows_with_preserved_seed_labels"] == 50
    assert set(verification["kpi_columns_present"]) == set(KPI_COLUMNS)

    with sqlite3.connect(db_path) as conn:
        axle = conn.execute(
            """
            SELECT component_domain, source_name, provenance_json
            FROM component_db
            WHERE component_id='SYN_AXLE_00'
            """
        ).fetchone()
        resolution = conn.execute(
            """
            SELECT boundary, confidence, fidelity_level, provenance_json
            FROM component_resolution
            WHERE component_resolution_id='CR_SYN_HUB_BEARING_00'
            """
        ).fetchone()
    assert axle[0:2] == ("AXLE_HUBS", "SYNTHETIC_REFERENCE")
    assert json.loads(axle[2])["seed_component_domain_original"] == "AXLE"
    assert resolution[0:3] == ("HUB_BEARING", None, None)
    resolution_provenance = json.loads(resolution[3])
    assert resolution_provenance["seed_confidence_original"] == "REFERENCE"
    assert resolution_provenance["seed_fidelity_level_original"] == "SYNTHETIC_POPULATION"
