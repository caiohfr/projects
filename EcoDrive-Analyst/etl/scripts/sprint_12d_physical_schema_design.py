"""Generate and validate the Sprint 12D physical-schema design package.

Runtime databases are opened read-only. DDL is executed only in SQLite
``:memory:`` for design validation; no migration or cutover is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "db" / "eco_drive.db"
CONTRACT_PATH = ROOT / "etl" / "data" / "processed" / "sprint_12c_contract_compatibility" / "canonical_field_contract_v1.csv"
SCHEMA_DIR = ROOT / "etl" / "schema"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12d_physical_schema"
REPORT = ROOT / "etl" / "reports" / "sprint_12d_physical_schema_design.md"
BLUEPRINT = ROOT / "etl" / "reports" / "sprint_12d_migration_blueprint.md"
SCHEMA_SQL = SCHEMA_DIR / "canonical_schema_v1.sql"
COMPAT_SQL = SCHEMA_DIR / "canonical_compatibility_v1.sql"

OUTPUT_PATHS = (
    SCHEMA_SQL,
    COMPAT_SQL,
    REPORT,
    BLUEPRINT,
    OUT / "constraint_matrix.csv",
    OUT / "index_plan.csv",
    OUT / "query_contract_inventory.csv",
    OUT / "legacy_to_canonical_column_map.csv",
    OUT / "physical_schema_design.json",
)

DOMAIN_TABLES = (
    "program", "vehicle_configuration", "component_db", "tire_db",
    "component_instance", "component_resolution", "vde", "run", "fuelcons",
)
HELPER_TABLES = ("fuelcons_run_adoption", "vde_component_resolution")
ENTITY_TO_TABLE = {name.upper(): name for name in DOMAIN_TABLES}
TABLE_TO_ENTITY = {value: key for key, value in ENTITY_TO_TABLE.items()}
TABLE_ORDER = DOMAIN_TABLES

TYPE_MAP = {
    "TEXT": "TEXT", "DATETIME": "TEXT", "JSON": "TEXT",
    "INTEGER": "INTEGER", "REAL": "REAL",
}

EXCLUDED_LOGICAL_FIELDS = {
    ("FUELCONS", "adopted_run_ids_json"),
}

EXTRA_FIELDS: dict[str, list[dict[str, Any]]] = {
    "program": [
        {"field_name": "source_name", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source system owning the source-scoped identity."},
        {"field_name": "source_file_version", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source snapshot/file version."},
        {"field_name": "source_record_id", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source record/group identifier."},
        {"field_name": "normalization_version", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Version of deterministic identity normalization."},
        {"field_name": "superseded_by_program_id", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Non-destructive Program consolidation target.", "fk_target": "program.program_id"},
        {"field_name": "record_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Lifecycle status."},
        {"field_name": "review_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Human-review state."},
    ],
    "vehicle_configuration": [
        {"field_name": "source_name", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source system."},
        {"field_name": "source_file_version", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source snapshot/file version."},
        {"field_name": "source_record_id", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source configuration record/group identifier."},
        {"field_name": "normalization_version", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Version of deterministic configuration normalization."},
        {"field_name": "created_at", "logical_data_type": "DATETIME", "nullable": False, "semantic_meaning": "Creation timestamp."},
        {"field_name": "updated_at", "logical_data_type": "DATETIME", "nullable": True, "semantic_meaning": "Update timestamp."},
        {"field_name": "record_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Lifecycle status."},
        {"field_name": "review_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Human-review state."},
    ],
    "component_db": [
        {"field_name": "source_file_version", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source snapshot/file version."},
        {"field_name": "created_at", "logical_data_type": "DATETIME", "nullable": False, "semantic_meaning": "Creation timestamp."},
        {"field_name": "updated_at", "logical_data_type": "DATETIME", "nullable": True, "semantic_meaning": "Update timestamp."},
        {"field_name": "record_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Lifecycle status."},
        {"field_name": "review_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Human-review state."},
    ],
    "tire_db": [
        {"field_name": "source_file_version", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source snapshot/file version."},
        {"field_name": "provenance_json", "logical_data_type": "JSON", "nullable": True, "semantic_meaning": "Structured provenance not represented by scalar tire fields."},
        {"field_name": "record_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Lifecycle status."},
        {"field_name": "review_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Human-review state."},
    ],
    "component_instance": [
        {"field_name": "created_at", "logical_data_type": "DATETIME", "nullable": False, "semantic_meaning": "Creation timestamp."},
        {"field_name": "updated_at", "logical_data_type": "DATETIME", "nullable": True, "semantic_meaning": "Update timestamp."},
        {"field_name": "record_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Lifecycle status."},
        {"field_name": "review_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Human-review state."},
    ],
    "component_resolution": [
        {"field_name": "created_at", "logical_data_type": "DATETIME", "nullable": False, "semantic_meaning": "Creation timestamp."},
        {"field_name": "updated_at", "logical_data_type": "DATETIME", "nullable": True, "semantic_meaning": "Update timestamp."},
        {"field_name": "record_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Lifecycle status."},
        {"field_name": "review_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Human-review state."},
    ],
    "vde": [
        {"field_name": "source_file_version", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source snapshot/file version."},
        {"field_name": "normalization_version", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Version of deterministic source mapping."},
        {"field_name": "provenance_json", "logical_data_type": "JSON", "nullable": True, "semantic_meaning": "Explicit observed/calculated/estimated snapshot provenance."},
    ],
    "run": [
        {"field_name": "source_file_version", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source snapshot/file version."},
        {"field_name": "updated_at", "logical_data_type": "DATETIME", "nullable": True, "semantic_meaning": "Update timestamp."},
        {"field_name": "record_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Lifecycle status."},
        {"field_name": "review_status", "logical_data_type": "TEXT", "nullable": False, "semantic_meaning": "Human-review state."},
    ],
    "fuelcons": [
        {"field_name": "source_file_version", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Source snapshot/file version."},
        {"field_name": "normalization_version", "logical_data_type": "TEXT", "nullable": True, "semantic_meaning": "Version of deterministic result mapping."},
        {"field_name": "reference_fuelcons_id", "logical_data_type": "INTEGER", "nullable": True, "semantic_meaning": "Optional principal/complementary result relationship.", "fk_target": "fuelcons.id"},
    ],
}

DEFAULTS = {
    (table, "created_at"): "CURRENT_TIMESTAMP" for table in DOMAIN_TABLES
}
DEFAULTS.update({
    ("program", "normalization_version"): "'program_norm_v1'",
    ("vehicle_configuration", "normalization_version"): "'configuration_norm_v1'",
    ("program", "record_status"): "'ACTIVE'",
    ("vehicle_configuration", "record_status"): "'ACTIVE'",
    ("component_db", "record_status"): "'ACTIVE'",
    ("tire_db", "record_status"): "'ACTIVE'",
    ("component_instance", "record_status"): "'ACTIVE'",
    ("component_resolution", "record_status"): "'ACTIVE'",
    ("run", "record_status"): "'ACTIVE'",
    ("program", "review_status"): "'CURRENT'",
    ("vehicle_configuration", "review_status"): "'CURRENT'",
    ("component_db", "review_status"): "'CURRENT'",
    ("tire_db", "review_status"): "'CURRENT'",
    ("component_instance", "review_status"): "'CURRENT'",
    ("component_resolution", "review_status"): "'CURRENT'",
    ("run", "review_status"): "'CURRENT'",
    ("vde", "record_origin"): "'LEGACY'",
    ("vde", "record_status"): "'ACTIVE'",
    ("vde", "review_status"): "'CURRENT'",
    ("tire_db", "record_origin"): "'LEGACY'",
    ("fuelcons", "record_origin"): "'LEGACY'",
    ("fuelcons", "record_status"): "'ACTIVE'",
    ("fuelcons", "review_status"): "'CURRENT'",
    ("fuelcons", "comparison_basis"): "'LEGACY_UNSPECIFIED'",
    ("component_instance", "quantity"): "1",
    ("tire_db", "is_active"): "1",
    ("tire_db", "is_broken_in"): "0",
    ("tire_db", "is_estimated_value"): "0",
    ("tire_db", "is_tested_value"): "0",
    ("tire_db", "temperature_correction_applied"): "0",
})

NOT_NULL_OVERRIDES = {
    ("vde", "record_origin"), ("vde", "record_status"), ("vde", "review_status"),
    ("tire_db", "record_origin"),
    ("fuelcons", "record_origin"), ("fuelcons", "record_status"), ("fuelcons", "review_status"),
}

ENUMS = {
    ("program", "identity_status"): ("CONFIRMED", "PROVISIONAL_SOURCE_SCOPED", "UNRESOLVED"),
    ("program", "identity_confidence"): ("LOW", "MEDIUM", "HIGH"),
    ("vehicle_configuration", "identity_status"): ("CONFIRMED", "SOURCE_SCOPED", "PROVISIONAL", "UNRESOLVED"),
    ("vehicle_configuration", "identity_confidence"): ("LOW", "MEDIUM", "HIGH"),
    ("component_db", "component_domain"): ("ENGINE", "TRANSMISSION", "EMOTOR", "BATTERY", "BRAKE", "AXLE_HUBS", "PARASITIC", "OTHER"),
    ("component_instance", "component_domain"): ("ENGINE", "TRANSMISSION", "EMOTOR", "BATTERY", "BRAKE", "AXLE_HUBS", "PARASITIC", "TIRE", "OTHER"),
    ("component_resolution", "confidence"): ("LOW", "MEDIUM", "HIGH"),
    ("component_resolution", "fidelity_level"): ("L0", "L1", "L2", "L3"),
    ("vde", "legislation"): ("EPA", "WLTP", "BRA", "OTHER"),
    ("vde", "source_semantic_status"): ("DIRECT", "PARTIAL", "UNRESOLVED"),
    ("run", "run_type"): ("TEST", "SIMULATION", "ESTIMATION", "CALCULATION", "ML_PREDICTION", "DECLARED_RESULT"),
    ("run", "evidence_kind"): ("ENGINEERING", "HOMOLOGATION", "MONITORING", "BENCHMARK", "SOURCE_RECORD"),
    ("run", "confidence"): ("LOW", "MEDIUM", "HIGH"),
    ("run", "fidelity_level"): ("L0", "L1", "L2", "L3"),
    ("fuelcons", "electrification"): ("NONE", "ICE", "MHEV", "HEV", "PHEV", "BEV", "FCEV", "OTHER"),
    ("fuelcons", "energy_basis"): ("VDE_TOTAL", "VDE_NET", "SOURCE_DECLARED", "OTHER"),
}

BOOLEAN_FIELDS = {
    ("tire_db", "is_active"), ("tire_db", "is_broken_in"),
    ("tire_db", "is_estimated_value"), ("tire_db", "is_tested_value"),
    ("tire_db", "temperature_correction_applied"), ("fuelcons", "ac_on"),
}

UNIQUE_FIELDS = {("tire_db", "tire_test_code")}

FK_OVERRIDES = {
    ("program", "superseded_by_program_id"): "program.program_id",
    ("vde", "vde_id_parent"): "vde.id",
    ("vde", "front_tire_id"): "tire_db.tire_id",
    ("vde", "rear_tire_id"): "tire_db.tire_id",
    ("fuelcons", "reference_fuelcons_id"): "fuelcons.id",
}


def q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_contract() -> list[dict[str, str]]:
    with CONTRACT_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_legacy_schema() -> dict[str, list[dict[str, Any]]]:
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = ON")
        result = {}
        for table in ("vde_db", "fuelcons_db", "component_db", "tire_roadload_db"):
            result[table] = [
                dict(zip(("cid", "name", "type", "notnull", "default", "pk"), row))
                for row in con.execute(f"PRAGMA table_info({q(table)})")
            ]
        return result
    finally:
        con.close()


def normalize_contract_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["nullable"] = str(row.get("nullable", "True")).lower() == "true"
    normalized["logical_data_type"] = str(row.get("logical_data_type") or "TEXT").upper()
    normalized["semantic_meaning"] = str(row.get("semantic_meaning") or "Physical contract field.")
    return normalized


def field_specs(contract: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in contract:
        entity = raw["entity"]
        field = raw["field_name"]
        if (entity, field) in EXCLUDED_LOGICAL_FIELDS:
            continue
        table = ENTITY_TO_TABLE[entity]
        row = normalize_contract_row(raw)
        role = row.get("key_fk_role") or ""
        if role.startswith("FOREIGN_KEY->"):
            target_entity, target_field = role.split("->", 1)[1].split(".", 1)
            row["fk_target"] = f"{ENTITY_TO_TABLE[target_entity]}.{target_field}"
        grouped[table].append(row)
    for table, extras in EXTRA_FIELDS.items():
        existing = {row["field_name"] for row in grouped[table]}
        for extra in extras:
            if extra["field_name"] not in existing:
                row = normalize_contract_row(extra)
                row["key_fk_role"] = "FOREIGN_KEY" if row.get("fk_target") else "NONE"
                row["contract_status"] = "SPRINT_12D_PHYSICAL_REFINEMENT"
                grouped[table].append(row)

    for table, rows in grouped.items():
        for row in rows:
            field = row["field_name"]
            if (table, field) in FK_OVERRIDES:
                row["fk_target"] = FK_OVERRIDES[(table, field)]
            if table == "component_instance" and field == "tire_id":
                row["logical_data_type"] = "INTEGER"
                row["physical_correction"] = "Corrected TEXT/INTEGER mismatch so FK matches tire_db.tire_id."
    return grouped


def on_delete(table: str, field: str) -> str:
    if (table, field) in {
        ("component_instance", "vehicle_configuration_id"),
        ("run", "vde_id"), ("fuelcons", "vde_id"),
    }:
        return "CASCADE"
    if (table, field) == ("component_resolution", "vehicle_configuration_id"):
        return "SET NULL"
    return "RESTRICT"


def field_checks(table: str, row: dict[str, Any]) -> list[str]:
    field = row["field_name"]
    checks: list[str] = []
    if (table, field) in ENUMS:
        values = ", ".join(sql_literal(value) for value in ENUMS[(table, field)])
        checks.append(f"{q(field)} IN ({values})")
    if row["logical_data_type"] == "JSON":
        checks.append(f"json_valid({q(field)})")
    if (table, field) in BOOLEAN_FIELDS:
        checks.append(f"{q(field)} IN (0, 1)")
    if field in {"quantity"}:
        checks.append(f"{q(field)} > 0")
    if (table, field) == ("vde", "mass_kg"):
        checks.append(f"{q(field)} >= 0")
    if (table, field) == ("tire_db", "rr_n_per_kn"):
        checks.append(f"{q(field)} >= 0")
    if (table, field) == ("fuelcons", "utility_factor_pct"):
        checks.append(f"{q(field)} BETWEEN 0 AND 100")
    if (table, field) == ("fuelcons", "comparison_basis"):
        checks.append(f"length(trim({q(field)})) > 0")
    if field in {"record_origin", "record_status", "review_status", "source_scope", "normalization_version"}:
        checks.append(f"length(trim({q(field)})) > 0")
    role = row.get("key_fk_role") or ""
    if role == "PRIMARY_KEY" and TYPE_MAP[row["logical_data_type"]] == "TEXT":
        checks.append(f"length(trim({q(field)})) > 0")
    return checks


def table_checks(table: str) -> list[tuple[str, str]]:
    if table == "program":
        return [
            ("ck_program_year_range", '"model_year_from" IS NULL OR "model_year_to" IS NULL OR "model_year_from" <= "model_year_to"'),
            ("ck_program_not_self_superseded", '"superseded_by_program_id" IS NULL OR "superseded_by_program_id" <> "program_id"'),
        ]
    if table == "component_instance":
        return [
            ("ck_component_instance_single_target", 'NOT ("component_id" IS NOT NULL AND "tire_id" IS NOT NULL)'),
            ("ck_component_instance_typed_target", '("component_domain" = \'TIRE\' AND "component_id" IS NULL) OR ("component_domain" <> \'TIRE\' AND "tire_id" IS NULL)'),
        ]
    return []


def table_unique_constraints(table: str) -> list[str]:
    if table == "run":
        return ['    CONSTRAINT "uq_run_id_vde" UNIQUE ("run_id", "vde_id")']
    if table == "fuelcons":
        return ['    CONSTRAINT "uq_fuelcons_id_vde" UNIQUE ("id", "vde_id")']
    return []


def column_sql(table: str, row: dict[str, Any]) -> str:
    field = row["field_name"]
    physical_type = TYPE_MAP[row["logical_data_type"]]
    parts = [q(field), physical_type]
    nullable = row["nullable"] and (table, field) not in NOT_NULL_OVERRIDES
    if not nullable:
        parts.append("NOT NULL")
    if (row.get("key_fk_role") or "") == "PRIMARY_KEY":
        parts.append("PRIMARY KEY")
    if (table, field) in UNIQUE_FIELDS:
        parts.append("UNIQUE")
    default = DEFAULTS.get((table, field))
    if default is not None:
        parts.append(f"DEFAULT {default}")
    target = row.get("fk_target")
    if target:
        target_table, target_field = target.split(".", 1)
        parts.append(f"REFERENCES {q(target_table)}({q(target_field)}) ON UPDATE CASCADE ON DELETE {on_delete(table, field)}")
    for check in field_checks(table, row):
        if nullable:
            parts.append(f"CHECK ({q(field)} IS NULL OR ({check}))")
        else:
            parts.append(f"CHECK ({check})")
    return "    " + " ".join(parts)


def helper_table_sql() -> list[str]:
    return [
        "-- Physical lineage helper; not a tenth primary domain entity.",
        "CREATE TABLE \"fuelcons_run_adoption\" (",
        "    \"fuelcons_id\" INTEGER NOT NULL,",
        "    \"run_id\" TEXT NOT NULL,",
        "    \"vde_id\" INTEGER NOT NULL REFERENCES \"vde\"(\"id\") ON UPDATE CASCADE ON DELETE CASCADE,",
        "    \"adoption_role\" TEXT NOT NULL CHECK (\"adoption_role\" IN ('PRIMARY', 'SUPPORTING', 'POST_PROCESSING_INPUT')),",
        "    \"result_dimension\" TEXT NOT NULL DEFAULT 'ALL',",
        "    \"ordinal\" INTEGER NOT NULL DEFAULT 0 CHECK (\"ordinal\" >= 0),",
        "    \"details_json\" TEXT CHECK (\"details_json\" IS NULL OR json_valid(\"details_json\")),",
        "    \"created_at\" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,",
        "    PRIMARY KEY (\"fuelcons_id\", \"run_id\", \"result_dimension\"),",
        "    FOREIGN KEY (\"fuelcons_id\", \"vde_id\") REFERENCES \"fuelcons\"(\"id\", \"vde_id\") ON UPDATE CASCADE ON DELETE CASCADE,",
        "    FOREIGN KEY (\"run_id\", \"vde_id\") REFERENCES \"run\"(\"run_id\", \"vde_id\") ON UPDATE CASCADE ON DELETE CASCADE",
        ");",
        "",
        "-- Physical adoption helper; Component Resolution remains optional.",
        "CREATE TABLE \"vde_component_resolution\" (",
        "    \"vde_id\" INTEGER NOT NULL REFERENCES \"vde\"(\"id\") ON UPDATE CASCADE ON DELETE CASCADE,",
        "    \"component_resolution_id\" TEXT NOT NULL REFERENCES \"component_resolution\"(\"component_resolution_id\") ON UPDATE CASCADE ON DELETE CASCADE,",
        "    \"boundary\" TEXT NOT NULL,",
        "    \"adoption_role\" TEXT NOT NULL DEFAULT 'ADOPTED' CHECK (\"adoption_role\" IN ('ADOPTED', 'SUPPORTING', 'SUPERSEDED')),",
        "    \"ordinal\" INTEGER NOT NULL DEFAULT 0 CHECK (\"ordinal\" >= 0),",
        "    \"created_at\" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,",
        "    PRIMARY KEY (\"vde_id\", \"component_resolution_id\", \"boundary\")",
        ");",
        "",
    ]


INDEX_PLAN = [
    ("idx_program_make_model_status", "program", "commercial_make, commercial_model, identity_status", False, "Program lookup and consolidation review", "Sprint 12C.3 Program profiles", "MEDIUM", "KEEP"),
    ("idx_program_source_identity", "program", "source_scope, source_name, source_record_id, source_file_version", False, "Source-scoped identity/version lookup", "CDR-01 / source provenance", "HIGH", "KEEP"),
    ("idx_program_superseded", "program", "superseded_by_program_id", False, "Resolve non-destructive Program consolidation", "Sprint 12C.3", "HIGH", "KEEP"),
    ("idx_vc_program", "vehicle_configuration", "program_id", False, "Configurations for one Program", "Canonical hierarchy", "HIGH", "KEEP"),
    ("idx_vc_source_identity", "vehicle_configuration", "source_scope, source_name, source_record_id, source_file_version", False, "Source configuration lookup", "CDR-02", "HIGH", "KEEP"),
    ("idx_vc_architecture", "vehicle_configuration", "engine_type, transmission_type, drive_system", False, "Candidate technical matching", "Sprint 12C.3 continuity", "MEDIUM", "KEEP"),
    ("idx_component_domain_make_model", "component_db", "component_domain, manufacturer, model", False, "Reusable component discovery", "component_repositories.py", "MEDIUM", "KEEP"),
    ("idx_component_source", "component_db", "source_name, source_record_id, source_file_version", False, "Component source lineage", "CDR provenance", "HIGH", "KEEP"),
    ("idx_tire_active_make_model", "tire_db", "is_active, manufacturer, model, size_code", False, "Active tire browse/search", "tire_roadload_repository.py", "MEDIUM", "KEEP"),
    ("idx_component_instance_configuration", "component_instance", "vehicle_configuration_id, component_domain, role, position", False, "eBOM-lite materialization by configuration", "PDR architecture path", "HIGH", "KEEP"),
    ("idx_component_instance_component", "component_instance", "component_id", False, "Reverse component usage", "Database management impact", "MEDIUM", "KEEP"),
    ("idx_component_instance_tire", "component_instance", "tire_id", False, "Reverse tire usage", "Database management impact", "MEDIUM", "KEEP"),
    ("idx_resolution_configuration", "component_resolution", "vehicle_configuration_id, boundary", False, "Resolution candidates by configuration/boundary", "Resolution/update path", "HIGH", "KEEP"),
    ("idx_vde_configuration", "vde", "vehicle_configuration_id, id", False, "VDE states for one configuration", "Canonical hierarchy", "HIGH", "KEEP"),
    ("idx_vde_parent", "vde", "vde_id_parent", False, "Scenario/derivation lineage", "comparison_report_service.py", "HIGH", "KEEP"),
    ("idx_vde_make_model_year", "vde", "make, model, year", False, "Browse and vehicle selectors", "vde_repository.py", "MEDIUM", "KEEP"),
    ("idx_vde_category_legislation", "vde", "category, legislation, make", False, "Current category/legislation/make filters", "vde_repository.py / comparison", "MEDIUM", "KEEP"),
    ("idx_vde_updated", "vde", "updated_at DESC, created_at DESC", False, "Recent VDE browse ordering", "vde_repository.py", "LOW_MEDIUM", "KEEP"),
    ("idx_vde_source", "vde", "source_name, source_record_id, source_file_version", False, "Source record/version audit", "CDR provenance", "HIGH", "KEEP"),
    ("idx_run_vde_created", "run", "vde_id, created_at DESC", False, "Evidence ledger for VDE", "CDR-03", "HIGH", "KEEP"),
    ("idx_run_source", "run", "source_name, source_record_id, source_file_version", False, "Source RUN identity/version", "EPA/JRC ingestion", "HIGH", "KEEP"),
    ("idx_run_type_kind", "run", "run_type, evidence_kind", False, "Evidence filters", "RUN contract", "MEDIUM", "KEEP"),
    ("idx_fuelcons_vde_created", "fuelcons", "vde_id, created_at DESC", False, "FuelCons rows for VDE", "fuelcons_repository.py", "HIGH", "KEEP"),
    ("idx_fuelcons_browse", "fuelcons", "electrification, record_origin, created_at DESC", False, "Comparison scenario filters/order", "comparison_report_service.py", "MEDIUM", "KEEP"),
    ("idx_fuelcons_power", "fuelcons", "engine_max_power_kw", False, "Power-range filter", "fuelcons_repository.py / regression.py", "MEDIUM", "KEEP"),
    ("idx_fuelcons_source", "fuelcons", "source_name, source_record_id, source_file_version", False, "Source result/version audit", "CDR provenance", "HIGH", "KEEP"),
    ("idx_adoption_run", "fuelcons_run_adoption", "run_id, fuelcons_id", False, "Reverse RUN adoption lineage", "FuelCons multi-RUN decision", "HIGH", "KEEP"),
    ("idx_vde_resolution_resolution", "vde_component_resolution", "component_resolution_id, vde_id", False, "Reverse resolution adoption", "Component Resolution mechanics", "HIGH", "KEEP"),
]


def index_sql() -> list[str]:
    lines = ["-- Indexes are justified by the audited application/source query paths."]
    for name, table, columns, unique, *_ in INDEX_PLAN:
        column_sql_text = ", ".join(
            " ".join([q(part.split()[0]), *part.split()[1:]]) for part in (value.strip() for value in columns.split(","))
        )
        lines.append(f"CREATE {'UNIQUE ' if unique else ''}INDEX {q(name)} ON {q(table)} ({column_sql_text});")
    lines.append("")
    return lines


def build_schema_sql(specs: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "-- EcoDrive canonical physical schema v1 — Sprint 12D design only.",
        "-- Execute in a new/disposable database. This file is not a runtime migration.",
        "PRAGMA foreign_keys = ON;",
        "PRAGMA user_version = 1204;",
        "",
    ]
    for table in TABLE_ORDER:
        lines.append(f"-- Primary domain entity: {TABLE_TO_ENTITY[table]}")
        definitions = [column_sql(table, row) for row in specs[table]]
        definitions.extend(
            f"    CONSTRAINT {q(name)} CHECK ({expression})" for name, expression in table_checks(table)
        )
        definitions.extend(table_unique_constraints(table))
        lines.append(f"CREATE TABLE {q(table)} (\n" + ",\n".join(definitions) + "\n);")
        lines.append("")
    lines.extend(helper_table_sql())
    lines.extend(index_sql())
    return "\n".join(lines)


def build_compatibility_sql(legacy: dict[str, list[dict[str, Any]]]) -> str:
    def view(name: str, table: str, columns: list[dict[str, Any]]) -> list[str]:
        selections = ",\n".join(f"    {q(item['name'])} AS {q(item['name'])}" for item in columns)
        return [f"CREATE VIEW {q(name)} AS", "SELECT", selections, f"FROM {q(table)};", ""]

    lines = [
        "-- EcoDrive legacy read compatibility views — Sprint 12D design only.",
        "-- The views deliberately use legacy table names in a new canonical DB.",
        "-- Writes are retargeted to canonical tables by persistence adapters in Sprint 12E.",
        "",
    ]
    lines.extend(view("vde_db", "vde", legacy["vde_db"]))
    lines.extend(view("fuelcons_db", "fuelcons", legacy["fuelcons_db"]))
    lines.extend([
        "-- Lineage is opt-in; normal FuelCons dashboards do not traverse RUN.",
        "CREATE VIEW \"fuelcons_lineage_v1\" AS",
        "SELECT f.*,",
        "       COALESCE((",
        "           SELECT json_group_array(a.run_id)",
        "           FROM (",
        "               SELECT run_id",
        "               FROM fuelcons_run_adoption",
        "               WHERE fuelcons_id = f.id",
        "               ORDER BY ordinal, run_id",
        "           ) AS a",
        "       ), '[]') AS adopted_run_ids_json",
        "FROM fuelcons AS f;",
        "",
    ])
    return "\n".join(lines)


def helper_specs() -> dict[str, list[dict[str, Any]]]:
    return {
        "fuelcons_run_adoption": [
            {"field_name": "fuelcons_id", "logical_data_type": "INTEGER", "nullable": False, "pk": True, "fk_target": "fuelcons.id", "semantic_meaning": "Adopted FuelCons result."},
            {"field_name": "run_id", "logical_data_type": "TEXT", "nullable": False, "pk": True, "fk_target": "run.run_id", "semantic_meaning": "Supporting evidence RUN."},
            {"field_name": "vde_id", "logical_data_type": "INTEGER", "nullable": False, "pk": False, "fk_target": "vde.id", "semantic_meaning": "Shared VDE key enforced by both composite lineage FKs."},
            {"field_name": "adoption_role", "logical_data_type": "TEXT", "nullable": False, "check": "PRIMARY|SUPPORTING|POST_PROCESSING_INPUT", "semantic_meaning": "Role in result adoption."},
            {"field_name": "result_dimension", "logical_data_type": "TEXT", "nullable": False, "pk": True, "default": "'ALL'", "semantic_meaning": "ALL or dimension-specific lineage."},
            {"field_name": "ordinal", "logical_data_type": "INTEGER", "nullable": False, "default": "0", "check": ">=0", "semantic_meaning": "Stable evidence ordering."},
            {"field_name": "details_json", "logical_data_type": "JSON", "nullable": True, "check": "json_valid", "semantic_meaning": "Sparse adoption detail."},
            {"field_name": "created_at", "logical_data_type": "DATETIME", "nullable": False, "default": "CURRENT_TIMESTAMP", "semantic_meaning": "Adoption timestamp."},
        ],
        "vde_component_resolution": [
            {"field_name": "vde_id", "logical_data_type": "INTEGER", "nullable": False, "pk": True, "fk_target": "vde.id", "semantic_meaning": "Adopting VDE snapshot."},
            {"field_name": "component_resolution_id", "logical_data_type": "TEXT", "nullable": False, "pk": True, "fk_target": "component_resolution.component_resolution_id", "semantic_meaning": "Adopted optional resolution."},
            {"field_name": "boundary", "logical_data_type": "TEXT", "nullable": False, "pk": True, "semantic_meaning": "Resolved physical boundary."},
            {"field_name": "adoption_role", "logical_data_type": "TEXT", "nullable": False, "default": "'ADOPTED'", "check": "ADOPTED|SUPPORTING|SUPERSEDED", "semantic_meaning": "Resolution adoption role."},
            {"field_name": "ordinal", "logical_data_type": "INTEGER", "nullable": False, "default": "0", "check": ">=0", "semantic_meaning": "Stable ordering."},
            {"field_name": "created_at", "logical_data_type": "DATETIME", "nullable": False, "default": "CURRENT_TIMESTAMP", "semantic_meaning": "Adoption timestamp."},
        ],
    }


def constraint_matrix(specs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in TABLE_ORDER:
        for row in specs[table]:
            field = row["field_name"]
            nullable = row["nullable"] and (table, field) not in NOT_NULL_OVERRIDES
            checks = field_checks(table, row)
            rows.append({
                "entity": TABLE_TO_ENTITY[table],
                "field": field,
                "logical_type": row["logical_data_type"],
                "physical_type": TYPE_MAP[row["logical_data_type"]],
                "nullable": nullable,
                "PK": (row.get("key_fk_role") or "") == "PRIMARY_KEY",
                "FK_target": row.get("fk_target", ""),
                "UNIQUE": (table, field) in UNIQUE_FIELDS,
                "CHECK": "; ".join(checks),
                "default": DEFAULTS.get((table, field), ""),
                "rationale": row.get("physical_correction") or row["semantic_meaning"],
            })
    for table, fields in helper_specs().items():
        for row in fields:
            rows.append({
                "entity": f"PHYSICAL_HELPER:{table}",
                "field": row["field_name"],
                "logical_type": row["logical_data_type"],
                "physical_type": TYPE_MAP[row["logical_data_type"]],
                "nullable": row["nullable"],
                "PK": bool(row.get("pk")),
                "FK_target": row.get("fk_target", ""),
                "UNIQUE": False,
                "CHECK": row.get("check", ""),
                "default": row.get("default", ""),
                "rationale": row["semantic_meaning"],
            })
    for table in TABLE_ORDER:
        for name, expression in table_checks(table):
            rows.append({
                "entity": TABLE_TO_ENTITY[table], "field": f"__TABLE_CONSTRAINT__:{name}",
                "logical_type": "RELATIONSHIP_RULE", "physical_type": "TABLE CHECK",
                "nullable": "N/A", "PK": False, "FK_target": "", "UNIQUE": False,
                "CHECK": expression, "default": "", "rationale": "Cross-field invariant enforced by SQLite.",
            })
        for constraint in table_unique_constraints(table):
            rows.append({
                "entity": TABLE_TO_ENTITY[table], "field": "__TABLE_CONSTRAINT__:composite_unique",
                "logical_type": "RELATIONSHIP_RULE", "physical_type": "TABLE UNIQUE",
                "nullable": "N/A", "PK": False, "FK_target": "", "UNIQUE": True,
                "CHECK": "", "default": "", "rationale": constraint.strip(),
            })
    rows.extend([
        {"entity": "PHYSICAL_HELPER:fuelcons_run_adoption", "field": "__COMPOSITE_FK__:fuelcons_id,vde_id", "logical_type": "RELATIONSHIP_RULE", "physical_type": "COMPOSITE FOREIGN KEY", "nullable": "N/A", "PK": False, "FK_target": "fuelcons(id,vde_id)", "UNIQUE": False, "CHECK": "", "default": "", "rationale": "Ensures the adopted FuelCons belongs to the helper row's VDE."},
        {"entity": "PHYSICAL_HELPER:fuelcons_run_adoption", "field": "__COMPOSITE_FK__:run_id,vde_id", "logical_type": "RELATIONSHIP_RULE", "physical_type": "COMPOSITE FOREIGN KEY", "nullable": "N/A", "PK": False, "FK_target": "run(run_id,vde_id)", "UNIQUE": False, "CHECK": "", "default": "", "rationale": "Prevents adoption of a RUN belonging to another VDE."},
    ])
    return rows


def index_plan_rows() -> list[dict[str, Any]]:
    return [
        {"index_name": name, "entity_table": table, "columns_order": columns, "unique": unique,
         "query_workflow_supported": workflow, "evidence_source": evidence,
         "expected_selectivity": selectivity, "keep_remove_recommendation": recommendation}
        for name, table, columns, unique, workflow, evidence, selectivity, recommendation in INDEX_PLAN
    ]


def query_contract_inventory() -> list[dict[str, Any]]:
    return [
        {"caller_service": "src/vde_core/repositories/vde_repository.py", "current_source_table_query": "SELECT * / id lookup from vde_db", "fields_used": "all 101 legacy VDE columns; id", "filter_sort_behavior": "id / IN(ids)", "proposed_compatibility_source": "vde_db compatibility VIEW", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/repositories/vde_repository.py", "current_source_table_query": "VDE recent/edit/catalog selectors", "fields_used": "id, timestamps, legislation, category, make, model, year, ABC, mass/test mass, notes", "filter_sort_behavior": "recent timestamp; id DESC; distinct make/category/transmission", "proposed_compatibility_source": "vde_db compatibility VIEW", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/repositories/fuelcons_repository.py", "current_source_table_query": "FuelCons by VDE/id", "fields_used": "scenario metadata, powertrain, aggregate and EPA-cycle outputs", "filter_sort_behavior": "vde_id; id; created_at DESC", "proposed_compatibility_source": "fuelcons_db compatibility VIEW", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/repositories/fuelcons_repository.py", "current_source_table_query": "fuelcons_db JOIN vde_db browse", "fields_used": "FuelCons KPIs + vehicle identity/category/legislation/VDE NET", "filter_sort_behavior": "electrification, vde, legislation, category, make, power range; created DESC", "proposed_compatibility_source": "legacy-named compatibility VIEWs", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/comparison_report_service.py", "current_source_table_query": "SELECT * VDE report and comparison catalogs", "fields_used": "full VDE; scenario IDs/labels; mass/CdA/RRC/ABC/TOTAL/NET/FuelCons KPIs", "filter_sort_behavior": "make, legislation, category, electrification, record_origin; recent first", "proposed_compatibility_source": "legacy-named compatibility VIEWs", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/comparison_report_service.py", "current_source_table_query": "vde_db parent traversal", "fields_used": "id, vde_id_parent, record_origin, timestamps", "filter_sort_behavior": "PK lookup per lineage node", "proposed_compatibility_source": "vde_db compatibility VIEW backed by indexed self-FK", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/nearest_peers.py", "current_source_table_query": "wide fuelcons_db JOIN vde_db", "fields_used": "vehicle/roadload/powertrain/KPI feature set", "filter_sort_behavior": "legislation, electrification, category, make, excluded VDE; created DESC", "proposed_compatibility_source": "legacy-named compatibility VIEWs", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/regression.py", "current_source_table_query": "training projection JOIN", "fields_used": "FuelCons phase KPIs, power, category, make, VDE NET", "filter_sort_behavior": "electrification, legislation, category, make, ids, power range", "proposed_compatibility_source": "legacy-named compatibility VIEWs", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/fuel_energy.py", "current_source_table_query": "SELECT * FROM vde_db WHERE id", "fields_used": "VDE NET and full row adapter surface", "filter_sort_behavior": "VDE PK", "proposed_compatibility_source": "vde_db compatibility VIEW", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/quick_scenario/resolver.py", "current_source_table_query": "source VDE/FuelCons resolution", "fields_used": "full source rows", "filter_sort_behavior": "VDE/FuelCons PK", "proposed_compatibility_source": "legacy-named compatibility VIEWs", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/database_management_impact_service.py", "current_source_table_query": "dependency discovery", "fields_used": "full VDE/FuelCons plus status/identity", "filter_sort_behavior": "VDE PK; FuelCons by vde_id", "proposed_compatibility_source": "legacy-named compatibility VIEWs", "expected_code_change": "NONE", "risk": "LOW"},
        {"caller_service": "src/vde_core/repositories/tire_roadload_repository.py", "current_source_table_query": "tire_roadload_db lookup/search", "fields_used": "full tire surface", "filter_sort_behavior": "id, code, active, make/model/size/standard/mileage", "proposed_compatibility_source": "12E repository retarget to tire_db; optional tire legacy view", "expected_code_change": "LOCALIZED_REPOSITORY_CHANGE", "risk": "MEDIUM"},
        {"caller_service": "src/vde_core/db.py + VDE/FuelCons persistence services", "current_source_table_query": "INSERT/UPDATE/DELETE legacy tables", "fields_used": "legacy write payloads", "filter_sort_behavior": "PK / VDE FK", "proposed_compatibility_source": "canonical vde/fuelcons tables via persistence adapters", "expected_code_change": "LOCALIZED_REPOSITORY_CHANGE", "risk": "MEDIUM: views are intentionally read-only"},
        {"caller_service": "pages/* and src/vde_app/*", "current_source_table_query": "service/repository calls; no active direct page SELECT contract found", "fields_used": "view-model contracts", "filter_sort_behavior": "delegated", "proposed_compatibility_source": "unchanged services over compatibility VIEWs", "expected_code_change": "NONE", "risk": "LOW; no page rewrite required"},
    ]


def legacy_column_map(legacy: dict[str, list[dict[str, Any]]], contract: list[dict[str, str]]) -> list[dict[str, Any]]:
    vde_result_fields = {
        "vde_urb_mj", "vde_hw_mj", "vde_net_mj_per_km", "vde_total_mj_per_km",
        "vde_low_mj_per_km", "vde_mid_mj_per_km", "vde_high_mj_per_km",
        "vde_extra_high_mj_per_km", "vde_urb_mj_per_km", "vde_hw_mj_per_km",
    }
    fuelcons_result_fields = {
        field["name"]
        for field in legacy["fuelcons_db"]
        if field["name"].startswith(("energy_", "fuel_", "gco2_", "label_fuel_", "label_gco2_", "label_range_"))
        and field["name"] != "fuel_type"
    }
    compatibility_fields = {
        "id", "created_at", "updated_at", "vde_id", "vde_id_parent", "legislation",
        "category", "make", "model", "year", "notes", "wltp_category", "cycle_name",
        "cycle_source", "label_program", "label_version_year", "label_vehicle_category",
        "label_cycle_set", "label_class", "label_offcycle_method", "record_origin",
        "record_status", "source_name", "source_record_id", "review_status",
    }

    def semantic(entity: str, field: str) -> str:
        if entity == "VDE" and field in vde_result_fields:
            return "RESULT_STATE"
        if entity == "FUELCONS" and field in fuelcons_result_fields:
            return "RESULT_STATE"
        if field in compatibility_fields:
            return "COMPATIBILITY_SNAPSHOT"
        return "RESOLVED_SNAPSHOT"

    def master_owner(entity: str, field: str) -> str:
        if entity == "VDE":
            if field in {"make", "model", "year", "category"}:
                return "PROGRAM / VEHICLE_CONFIGURATION"
            if field.startswith("engine_"):
                return "COMPONENT_DB / VEHICLE_CONFIGURATION"
            if field.startswith("transmission_") or field == "drive_type":
                return "COMPONENT_DB / VEHICLE_CONFIGURATION"
            if "tire" in field or field in {"rrc_N_per_kN", "front_pressure_psi", "rear_pressure_psi"}:
                return "TIRE_DB / COMPONENT_INSTANCE / COMPONENT_RESOLUTION"
            if field in {"mass_kg", "mro_kg", "options_kg", "payload_kg", "cda_m2", "inertia_class"}:
                return "VEHICLE_CONFIGURATION / COMPONENT_RESOLUTION"
        if entity == "FUELCONS":
            if field.startswith("engine_"):
                return "COMPONENT_DB / VEHICLE_CONFIGURATION"
            if field in {"gear_count", "final_drive_ratio"}:
                return "COMPONENT_DB / VEHICLE_CONFIGURATION"
            if field.startswith("battery_") or field.startswith("bms_"):
                return "COMPONENT_DB / VEHICLE_CONFIGURATION"
            if field in {"electrification", "fuel_type"}:
                return "VEHICLE_CONFIGURATION / COMPONENT_DB"
            if field in {"tire_front_psi", "tire_rear_psi"}:
                return "TIRE_DB / VDE"
            if field in {"scenario_payload_kg", "ambient_temp_c"}:
                return "VEHICLE_CONFIGURATION / VDE"
        return ""

    def resolution_source(entity: str, field: str, ownership_semantic: str, master: str) -> str:
        if ownership_semantic == "RESULT_STATE":
            return "Canonical calculation/result pipeline plus persisted inputs and provenance"
        if master:
            return f"{master}, then explicit source/scenario corrections at write time"
        if ownership_semantic == "COMPATIBILITY_SNAPSHOT":
            return "Source payload or canonical write payload retained for the legacy-facing contract"
        return "Resolved source/scenario write payload with explicit provenance"

    lookup = {
        (row["entity"], row["field_name"]): row for row in contract
    }
    result = []
    for legacy_table, entity in (("vde_db", "VDE"), ("fuelcons_db", "FUELCONS")):
        for column in legacy[legacy_table]:
            field = column["name"]
            contract_row = lookup[(entity, field)]
            ownership_semantic = semantic(entity, field)
            reference_owner = master_owner(entity, field)
            result.append({
                "legacy_table": legacy_table,
                "legacy_field": field,
                "canonical_owner": entity,
                "canonical_field": field,
                "ownership_semantic": ownership_semantic,
                "master_reference_owner": reference_owner,
                "snapshot_owner": entity,
                "write_time_resolution_source": resolution_source(
                    entity, field, ownership_semantic, reference_owner
                ),
                "intentional_duplication": "YES" if reference_owner or ownership_semantic == "COMPATIBILITY_SNAPSHOT" else "NO",
                "projection_strategy": "DIRECT_COLUMN",
                "stored_vs_derived": "STORED_SNAPSHOT" if entity == "VDE" else "STORED_ADOPTED_RESULT",
                "compatibility_field_name": field,
                "intentional_correction": "NO",
                "notes": contract_row["semantic_meaning"],
            })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")).replace("|", "/").replace("\n", "<br>") for field in fields) + " |")
    return lines


def report_text(payload: dict[str, Any]) -> str:
    counts = payload["physical_column_counts"]
    count_rows = [{"table": table, "columns": count} for table, count in counts.items()]
    lines = [
        "# Sprint 12D — Physical Schema & Compatibility Layer Design", "",
        f"## Status: `{payload['status']}`", "",
        "## Human review packet", "",
        "### Physical hierarchy", "", "```text",
        "PROGRAM",
        "  ↓ 1:N",
        "VEHICLE_CONFIGURATION",
        "  ├── 1:N COMPONENT_INSTANCE ──► COMPONENT_DB or TIRE_DB",
        "  ↓ 1:N",
        "VDE (wide persisted snapshot)",
        "  ├── N:M COMPONENT_RESOLUTION via physical adoption helper",
        "  ├── 1:N RUN",
        "  └── 1:N FUELCONS ── N:M RUN via physical lineage helper",
        "```", "",
        "The two associative tables are physical relationship infrastructure, not new primary domain entities.", "",
        "### Approximate physical size", "", *md_table(count_rows, ["table", "columns"]), "",
        "### Decisions to review", "",
        "- **FuelCons↔RUN:** authoritative N:M lineage in `fuelcons_run_adoption`; normal FuelCons reads do not join RUN. The logical `adopted_run_ids_json` becomes an opt-in lineage-view projection, not duplicated stored state.",
        "- **Component Instance:** nullable typed FKs `component_id` and `tire_id`, with checks forbidding both simultaneously and enforcing Tire/non-Tire domain direction. Both may be NULL for explicitly unresolved partial instances.",
        "- **Component Resolution:** optional N:M adoption helper `vde_component_resolution`; VDE values stay copied into the immutable wide snapshot and never reconstruct at page-read time.",
        "- **EEA:** keep 10.8M+ row-level monitoring records in separate analytical storage (versioned Parquet/DuckDB or later warehouse). Materialize only reviewed records that have a validated VDE link; unlinked monitoring evidence stays analytical.",
        "- **Compatibility:** read-only legacy-named `vde_db` and `fuelcons_db` VIEWs expose exactly the current 101/79 columns. 12E retargets writes in persistence adapters; pages remain unchanged.",
        "- **USER_DECISION_REQUIRED:** none.", "",
        "## Identifier and lifecycle strategy", "",
        "A mixed surrogate strategy preserves application identity: VDE, FuelCons and Tire retain stable INTEGER keys; Program, Configuration, Component, Component Instance, Component Resolution and RUN use application-generated TEXT canonical IDs (UUID/ULID-compatible, never source natural keys). Source IDs and file versions are separate columns/JSON provenance.", "",
        "Program consolidation is non-destructive: `superseded_by_program_id` points provisional/old Programs to a reviewed target while child Configurations remain attached until an explicit migration remap. No Model Year uniqueness rule defines Program identity.", "",
        "## Entity-by-entity shape", "",
        "- `program`: semantic generation identity, confidence/status, source scope/version and non-destructive supersession.",
        "- `vehicle_configuration`: stable architecture scalars plus sparse `architecture_properties_json`; no Target/Set ABC, ETW or procedure identity fields.",
        "- `component_db` / `tire_db`: reusable masters; common filter/engineering scalars remain columns, sparse properties/artifacts remain JSON/references.",
        "- `component_instance`: eBOM-lite occurrence. Unresolved references remain valid without fabricated component masters.",
        "- `component_resolution`: optional method/boundary/result/provenance record; public ingestion may create zero rows.",
        "- `vde`: 104-field logical contract plus physical provenance/version fields; wide persisted operational state with self-parent and Tire FKs.",
        "- `run`: append-oriented evidence with frozen run type/evidence kind/fidelity/confidence checks.",
        "- `fuelcons`: persisted comparison result linked directly to one VDE; multi-evidence adoption is externalized to the helper table.", "",
        "## Master vs Persisted Snapshot Semantics", "",
        "Duplicated engineering scalars in `vde` and `fuelcons` are intentional when they record the effective value used or exposed by that persisted analysis/result. They are not competing reusable master definitions and are not a normalization defect. The ownership map classifies every legacy-facing field as `AUTHORITATIVE_MASTER`, `RESOLVED_SNAPSHOT`, `COMPATIBILITY_SNAPSHOT`, or `RESULT_STATE` and records any reference master plus its write-time resolution source.", "",
        "| Example | Reusable/master meaning | Persisted snapshot meaning |",
        "|---|---|---|",
        "| Engine power | `component_db.rated_power_kw` is the reusable component rating | `fuelcons.engine_max_power_kw` is the effective power adopted by that result |",
        "| Transmission | configuration/component gear count and ratio describe reusable hardware | FuelCons gear count and final drive record the values actually used |",
        "| Battery | `component_db.capacity_kwh` is the reusable nominal definition | FuelCons capacity/usable-energy fields preserve scenario-effective assumptions |",
        "| Tire/roadload | Tire and configuration records provide reusable/reference inputs | VDE preserves the resolved tire, mass, aero and roadload state used by its calculation |", "",
        "Materialization occurs at write or explicit update time: the persistence workflow resolves source, configuration, component and correction inputs; writes the effective scalar into the VDE/FuelCons snapshot; and retains lineage/provenance. Normal reads consume that snapshot directly and do not reconstruct historical values through new joins.", "",
        "Historical snapshots are immutable with respect to master maintenance. Editing `component_db`, `vehicle_configuration` or `tire_db` must not silently update an existing VDE/FuelCons row. Adoption of a changed master value requires an explicit recalculation/rebuild workflow with new lineage or revision semantics. Therefore a 150 kW component rating and a 135 kW FuelCons effective-power snapshot can both be correct.", "",
        "## Scalar / JSON / artifact boundary", "",
        "Fields used by filters, joins, physics, comparisons, provenance routing or migration keys remain scalar. JSON is limited to source identity payloads, sparse type-specific architecture/component properties, conditions, assumptions and detailed lineage. Large maps, curves, PDFs and executable models remain artifact references; they are not stored as SQLite JSON blobs.", "",
        "## Constraint and enum strategy", "",
        "Frozen v1 CHECKs cover Program/Configuration identity status, component domains, RUN type/evidence kind, fidelity/confidence, VDE source semantics/legislation, electrification, energy basis, booleans, JSON validity and key numeric bounds. `comparison_basis`, `record_origin`, `record_status` and `review_status` are non-empty but intentionally extensible/legacy-compatible because current services already use values beyond the provisional 12C candidate lists.", "",
        "The 12C logical `COMPONENT_INSTANCE.tire_id` TEXT/INTEGER mismatch is corrected to INTEGER so SQLite can enforce the FK to `tire_db.tire_id`.", "",
        "## Index and query rationale", "",
        f"The package defines **{len(payload['index_plan'])}** indexes backed by audited repository/service queries. The plan covers Program/Configuration lookup, VDE hierarchy and recent browse, FuelCons/VDE joins and filters, RUN evidence, source versions, component usage and reverse lineage. Full rationale is in `index_plan.csv`.", "",
        "## Compatibility and application boundary", "",
        f"The query inventory contains **{len(payload['query_contract_inventory'])}** active contract groups. Existing SELECT paths can continue using `vde_db`/`fuelcons_db` because those names become compatibility views in a new canonical database. `PRAGMA table_info(fuelcons_db)` also remains available. Existing writes cannot target views, so 12E must retarget the concentrated persistence helpers/services to `vde`/`fuelcons`; no existing page requires a schema-driven rewrite.", "",
        "RUN lineage is available through `fuelcons_lineage_v1` only when explicitly requested. Comparison/Browse reads FuelCons directly and never traverses RUN.", "",
        "## EEA physical storage", "",
        "Do not load row-level EEA monitoring into the normal Streamlit SQLite file. Keep immutable/versioned source rows in analytical storage with source keys and provenance, aggregate there, and materialize canonical RUN/FuelCons only after a reviewed link to a valid VDE exists. Unlinked monitoring evidence remains analytical. This protects startup, backup, vacuum and browse behavior now and maps cleanly to a future partitioned PostgreSQL/warehouse service.", "",
        "## Temporary validation", "",
        f"The schema instantiated successfully in SQLite `:memory:` with **{payload['domain_table_count']}** primary domain tables, **{payload['helper_table_count']}** physical helper tables, foreign keys enabled and `foreign_key_check` empty. Compatibility views expose **{payload['legacy_vde_columns']}** VDE and **{payload['legacy_fuelcons_columns']}** FuelCons legacy columns.", "",
        "No runtime DB was modified; SHA-256 remained byte-identical.", "",
        "## Migration risks and 12E gates", "",
        "- Retarget legacy write helpers before cutover; compatibility views are deliberately read-only.",
        "- Backfill required Program/Configuration/VDE semantic fields and explicit provenance before enabling constraints.",
        "- Correct scenario/ML origin labels and the Component Instance tire FK type during staged load.",
        "- Validate all 180 legacy columns, NULL signatures, multiplicities and IDs before switching the DB path.",
        "- Benchmark real population queries and keep/remove indexes using measured plans after load.",
        "- Do not place EEA row-level data in the runtime file during migration.", "",
        "## Evidence classification", "",
        "- **DIRECTLY_TESTED:** `test_schema_creates_in_memory_with_nine_domain_entities`; `test_foreign_keys_are_enabled_and_valid`; `test_constraints_reject_invalid_rows`; `test_valid_minimal_rows_cover_all_nine_entities`; `test_required_cardinalities`; `test_fuelcons_supports_multiple_run_lineage`; `test_vde_can_exist_without_component_resolution`; `test_jrc_unresolved_identity_is_representable`; `test_null_and_zero_remain_distinct`; `test_compatibility_vde_columns_match_runtime`; `test_compatibility_fuelcons_columns_match_runtime`; `test_compatibility_projection_values`; `test_program_consolidation_preserves_children`; `test_component_instance_reference_mechanics`; `test_query_plan_uses_vde_configuration_index`; `test_outputs_are_scoped_to_etl`; `test_master_change_preserves_snapshot_and_explicit_rebuild_adopts_new_value`; `test_ownership_map_classifies_representative_snapshots`; `test_runtime_database_is_byte_identical`.",
        "- **INDIRECTLY_COVERED:** Sprint 12C exact field/value compatibility and Sprint 12C.3 population shape.",
        "- **INSPECTION_SUPPORTED:** active repository/service/page query inventory and scalar/JSON/artifact boundary.",
        "- **GAP:** production-sized migrated query timings and real cutover behavior belong to 12E; in-memory tests are not production migration tests.", "",
        "## Reproduction", "", "```powershell",
        "python etl/scripts/sprint_12d_physical_schema_design.py",
        "python -m unittest discover -s etl/tests -p \"test_sprint_12d*.py\" -v", "```", "",
        "## Outputs", "", *[f"- `{path.relative_to(ROOT).as_posix()}`" for path in OUTPUT_PATHS],
    ]
    return "\n".join(lines) + "\n"


def blueprint_text() -> str:
    return """# Sprint 12D — Proposed Sprint 12E Migration Blueprint

## Boundary

This is a plan only. Sprint 12D did not migrate, cut over, or write any runtime/reference database.

## Proposed phases

1. Freeze source/runtime hashes and create a new empty disposable canonical database.
2. Execute `canonical_schema_v1.sql`; enable and verify foreign keys.
3. Load versioned legacy snapshots into migration staging, never by altering the current runtime file.
4. Populate Program and Vehicle Configuration using approved source-scoped fallbacks and reviewed safe Program consolidation.
5. Populate Component DB and Tire DB references; preserve incomplete enrichment.
6. Populate Component Instances only where evidence exists; unresolved references remain NULL with provenance.
7. Populate wide VDE snapshots with original integer IDs and exact NULL/value semantics.
8. Populate append-oriented RUN evidence, including corrected ESTIMATION/ML/scenario provenance.
9. Populate FuelCons adopted results with original IDs and direct VDE links.
10. Populate `fuelcons_run_adoption` and optional `vde_component_resolution` links.
11. Execute `canonical_compatibility_v1.sql` and compare all legacy-facing columns/relationships.
12. Run exact equivalence: IDs, counts, values, NULL signatures, parent lineage, VDE→FuelCons multiplicity and labels/filters.
13. Run measured query plans/timings on real migrated volume; revise only unsupported indexes.
14. Perform human review and archive a signed migration manifest.
15. Cut over by changing the configured DB path only after explicit approval; retain the original DB read-only for rollback.

## Write-path transition

Compatibility views preserve existing reads. Before cutover, retarget persistence helpers and direct service writes from legacy view names to canonical `vde`, `fuelcons`, and `tire_db` tables. No page should contain schema-specific migration logic.

## EEA

Keep the 10.8M+ row-level monitoring corpus in versioned analytical storage. 12E may materialize RUN/FuelCons only after a reviewed link to a valid VDE exists; unlinked records remain analytical. Do not bulk-load the entire EEA corpus into the Streamlit runtime SQLite database.

## Stop / rollback gates

- Stop on any mismatch in the 180-field compatibility projection, IDs, NULL behavior or relationship multiplicity.
- Stop if current physics requires reconstructed joins instead of the persisted VDE snapshot.
- Stop if a page rewrite is required solely by persistence normalization.
- Stop if source semantics would need to be invented.
- Roll back by restoring the original configured DB path; never mutate the original database in place.
"""


def validate_in_memory(schema_sql: str, compat_sql: str, legacy: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(schema_sql)
        con.executescript(compat_sql)
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        views = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        fk_enabled = con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        fk_errors = con.execute("PRAGMA foreign_key_check").fetchall()
        vde_columns = [row[1] for row in con.execute("PRAGMA table_info(vde_db)")]
        fuelcons_columns = [row[1] for row in con.execute("PRAGMA table_info(fuelcons_db)")]
        expected_vde = [row["name"] for row in legacy["vde_db"]]
        expected_fuelcons = [row["name"] for row in legacy["fuelcons_db"]]
        if set(DOMAIN_TABLES) - tables or set(HELPER_TABLES) - tables:
            raise RuntimeError("In-memory schema is missing required tables.")
        if not {"vde_db", "fuelcons_db", "fuelcons_lineage_v1"}.issubset(views):
            raise RuntimeError("In-memory schema is missing compatibility views.")
        if not fk_enabled or fk_errors:
            raise RuntimeError(f"Foreign-key validation failed: {fk_errors}")
        if vde_columns != expected_vde or fuelcons_columns != expected_fuelcons:
            raise RuntimeError("Compatibility view columns differ from the runtime legacy contract.")
        return {
            "tables": sorted(tables), "views": sorted(views),
            "foreign_keys_enabled": fk_enabled, "foreign_key_errors": len(fk_errors),
            "legacy_vde_columns": len(vde_columns), "legacy_fuelcons_columns": len(fuelcons_columns),
        }
    finally:
        con.close()


def main() -> dict[str, Any]:
    missing = [str(path) for path in (DB_PATH, CONTRACT_PATH) if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing Sprint 12D inputs: {missing}")
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    db_hash_before = sha256(DB_PATH)
    contract = read_contract()
    legacy = read_legacy_schema()
    specs = field_specs(contract)
    schema_sql = build_schema_sql(specs)
    compat_sql = build_compatibility_sql(legacy)
    validation = validate_in_memory(schema_sql, compat_sql, legacy)
    constraints = constraint_matrix(specs)
    indexes = index_plan_rows()
    queries = query_contract_inventory()
    ownership = legacy_column_map(legacy, contract)
    db_hash_after = sha256(DB_PATH)
    if db_hash_before != db_hash_after:
        raise RuntimeError("Runtime database changed during Sprint 12D design.")
    if len(contract) != 333 or len(ownership) != 180:
        raise RuntimeError("Canonical/legacy contract population changed unexpectedly.")

    counts = {table: len(specs[table]) for table in TABLE_ORDER}
    counts.update({table: len(fields) for table, fields in helper_specs().items()})
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PHYSICAL_SCHEMA_READY — PROCEED_TO_12E",
        "scope": "Physical design and in-memory validation only; no migration/cutover.",
        "database_access": "SQLite URI mode=ro + PRAGMA query_only=ON",
        "database_sha256_before": db_hash_before,
        "database_sha256_after": db_hash_after,
        "database_byte_identical": True,
        "domain_table_count": len(DOMAIN_TABLES),
        "helper_table_count": len(HELPER_TABLES),
        "physical_column_counts": counts,
        "legacy_vde_columns": validation["legacy_vde_columns"],
        "legacy_fuelcons_columns": validation["legacy_fuelcons_columns"],
        "in_memory_validation": validation,
        "decisions": {
            "pk_strategy": "Mixed stable surrogate keys: legacy-compatible INTEGER for VDE/FuelCons/Tire; application-generated TEXT for new identity/evidence entities.",
            "fuelcons_run": "Authoritative N:M fuelcons_run_adoption helper; opt-in lineage view; no RUN traversal on normal reads.",
            "component_instance": "Two typed nullable FKs, at most one target; both NULL allowed for unresolved partial evidence.",
            "component_resolution": "Optional N:M VDE adoption helper; persisted VDE values remain authoritative snapshot.",
            "compatibility": "Legacy-named read VIEWs with exact 101/79 columns; localized write retargeting in 12E.",
            "eea": "Separate analytical row store; materialize reviewed/adopted records only.",
            "user_decision_required": [],
        },
        "constraint_matrix": constraints,
        "index_plan": indexes,
        "query_contract_inventory": queries,
        "legacy_to_canonical_column_map": ownership,
    }

    SCHEMA_SQL.write_text(schema_sql, encoding="utf-8")
    COMPAT_SQL.write_text(compat_sql, encoding="utf-8")
    write_csv(OUT / "constraint_matrix.csv", constraints)
    write_csv(OUT / "index_plan.csv", indexes)
    write_csv(OUT / "query_contract_inventory.csv", queries)
    write_csv(OUT / "legacy_to_canonical_column_map.csv", ownership)
    (OUT / "physical_schema_design.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(report_text(payload), encoding="utf-8")
    BLUEPRINT.write_text(blueprint_text(), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "domain_tables": payload["domain_table_count"],
        "helper_tables": payload["helper_table_count"],
        "legacy_view_columns": {"vde_db": payload["legacy_vde_columns"], "fuelcons_db": payload["legacy_fuelcons_columns"]},
        "constraint_rows": len(constraints),
        "indexes": len(indexes),
        "query_contracts": len(queries),
        "ownership_rows": len(ownership),
        "database_byte_identical": True,
    }, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
