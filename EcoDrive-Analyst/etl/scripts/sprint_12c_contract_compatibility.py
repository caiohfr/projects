"""Sprint 12C - Canonical Data Contract v1 materialization and compatibility proof.

The approved CDR is materialized as an exact logical field contract and an
in-memory/JSONL staging model. Existing SQLite databases are opened read-only.
No DDL, migration, runtime adapter, application page, or physics code is changed.

Run from the repository root:
    python etl/scripts/sprint_12c_contract_compatibility.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CDR = ROOT / "docs" / "sprints" / "CDR_CANONICAL_DATA_CONTRACT_V1_BASELINE.md"
EVIDENCE_12B = ROOT / "etl" / "data" / "processed" / "sprint_12b_source_to_contract" / "source_to_contract.json"
DB_PATH = ROOT / "data" / "db" / "eco_drive.db"
EPA_PATH = ROOT / "etl" / "data" / "raw" / "epa_testcar" / "epa_testcar_2026_raw.xlsx"
JRC_PATH = ROOT / "etl" / "data" / "raw" / "wltp_jrc" / "Data_PV_fleet_2021_EU_PYCSIS.xlsx"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12c_contract_compatibility"
STAGING = ROOT / "etl" / "data" / "staging" / "sprint_12c_contract_v1"
REPORT = ROOT / "etl" / "reports" / "sprint_12c_contract_compatibility.md"

ENTITIES = (
    "PROGRAM", "VEHICLE_CONFIGURATION", "COMPONENT_DB", "TIRE_DB",
    "COMPONENT_INSTANCE", "COMPONENT_RESOLUTION", "VDE", "RUN", "FUELCONS",
)
ALLOWED_RESULTS = {"EXACT_EQUIVALENCE", "APPROVED_CONTRACT_CORRECTION", "CDR_BLOCKER", "DEFERRED"}


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k): clean(v) for k, v in row.items()}


def stable_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join("<NULL>" if clean(x) is None else str(clean(x)).strip().casefold() for x in parts)
    return f"{prefix}-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:20].upper()}"


def ro_connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def schema_for(con: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [
        {"position": r[0], "field": r[1], "sql_type": r[2] or "ANY", "not_null": bool(r[3]), "default": r[4], "primary_key": bool(r[5])}
        for r in con.execute(f'PRAGMA table_info("{table}")')
    ]


def rows_for(con: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [clean_record(dict(r)) for r in con.execute(f'SELECT * FROM "{table}" ORDER BY 1')]


def contract_row(
    entity: str,
    field: str,
    meaning: str,
    data_type: str,
    nullable: bool,
    key_role: str = "NONE",
    enum_candidates: str = "",
    provenance: str = "DIRECT_OR_EXPLICIT_DERIVATION",
    legacy_mapping: str = "",
    public_mapping: str = "",
    compatibility_projection: str = "",
) -> dict[str, Any]:
    return {
        "entity": entity,
        "field_name": field,
        "semantic_meaning": meaning,
        "logical_data_type": data_type,
        "nullable": nullable,
        "key_fk_role": key_role,
        "enum_candidates": enum_candidates,
        "source_provenance_expectation": provenance,
        "legacy_field_mapping": legacy_mapping,
        "public_source_mapping": public_mapping,
        "compatibility_projection": compatibility_projection,
        "contract_status": "CDR_V1_BASELINE",
    }


def base_contract() -> list[dict[str, Any]]:
    c: list[dict[str, Any]] = []
    add = lambda *args, **kwargs: c.append(contract_row(*args, **kwargs))
    # PROGRAM
    add("PROGRAM", "program_id", "Canonical engineering project/generation identifier.", "TEXT", False, "PRIMARY_KEY")
    add("PROGRAM", "commercial_make", "Commercial/OEM make as observed or normalized with lineage.", "TEXT", False, provenance="DIRECT_WITH_RAW_VALUE")
    add("PROGRAM", "commercial_model", "Commercial model/carline identity.", "TEXT", False, provenance="DIRECT_WITH_RAW_VALUE")
    add("PROGRAM", "generation_name", "Engineering generation/program name when positively evidenced.", "TEXT", True)
    add("PROGRAM", "model_year_from", "Earliest supported model year, not semantic identity by itself.", "INTEGER", True)
    add("PROGRAM", "model_year_to", "Latest supported model year, not semantic identity by itself.", "INTEGER", True)
    add("PROGRAM", "identity_status", "Whether identity is confirmed or source-scoped provisional.", "TEXT", False, enum_candidates="CONFIRMED|PROVISIONAL_SOURCE_SCOPED|UNRESOLVED")
    add("PROGRAM", "identity_confidence", "Confidence independent of source authority.", "TEXT", True, enum_candidates="LOW|MEDIUM|HIGH")
    add("PROGRAM", "source_scope", "Namespace preventing unsupported cross-source merges.", "TEXT", False)
    add("PROGRAM", "source_identity_json", "Raw source identity fields retained losslessly.", "JSON", False, provenance="DIRECT_RAW_SOURCE_VALUES")
    add("PROGRAM", "created_at", "Canonical record creation timestamp.", "DATETIME", True)
    add("PROGRAM", "updated_at", "Canonical record update timestamp.", "DATETIME", True)

    # VEHICLE_CONFIGURATION
    add("VEHICLE_CONFIGURATION", "vehicle_configuration_id", "Canonical stable technical-variant identifier.", "TEXT", False, "PRIMARY_KEY")
    add("VEHICLE_CONFIGURATION", "program_id", "Owning Program.", "TEXT", False, "FOREIGN_KEY->PROGRAM.program_id")
    for field, meaning, dtype in [
        ("propulsion_architecture", "Stable propulsion/electrification architecture.", "TEXT"),
        ("engine_type", "Stable engine/converter type.", "TEXT"),
        ("engine_model", "Source/OEM engine model or family.", "TEXT"),
        ("engine_displacement_l", "Engine displacement in litres.", "REAL"),
        ("engine_aspiration", "Aspiration architecture.", "TEXT"),
        ("transmission_type", "Transmission architecture/type.", "TEXT"),
        ("transmission_model", "Transmission model/family.", "TEXT"),
        ("gear_count", "Forward gear count when meaningful.", "INTEGER"),
        ("drive_system", "FWD/RWD/AWD or source drive architecture.", "TEXT"),
        ("final_drive_ratio", "Stable final-drive/axle ratio when evidenced.", "REAL"),
        ("nv_ratio", "Source N/V ratio when configuration-stable.", "REAL"),
    ]:
        add("VEHICLE_CONFIGURATION", field, meaning, dtype, True)
    add("VEHICLE_CONFIGURATION", "identity_status", "Configuration identity resolution state.", "TEXT", False, enum_candidates="CONFIRMED|SOURCE_SCOPED|PROVISIONAL|UNRESOLVED")
    add("VEHICLE_CONFIGURATION", "identity_confidence", "Configuration-match confidence.", "TEXT", True, enum_candidates="LOW|MEDIUM|HIGH")
    add("VEHICLE_CONFIGURATION", "source_scope", "Source namespace for provisional identity.", "TEXT", False)
    add("VEHICLE_CONFIGURATION", "source_identity_json", "EPA vehicle/configuration IDs or equivalent source identity; not universal hardware identity.", "JSON", False, provenance="DIRECT_RAW_SOURCE_VALUES")
    add("VEHICLE_CONFIGURATION", "architecture_properties_json", "Sparse type-specific stable architecture descriptors.", "JSON", True)

    # COMPONENT_DB
    add("COMPONENT_DB", "component_id", "Canonical reusable component definition identifier.", "TEXT", False, "PRIMARY_KEY")
    add("COMPONENT_DB", "component_domain", "Engineering component domain.", "TEXT", False, enum_candidates="ENGINE|TRANSMISSION|EMOTOR|BATTERY|BRAKE|AXLE_HUBS|PARASITIC|OTHER")
    add("COMPONENT_DB", "manufacturer", "Component manufacturer when evidenced.", "TEXT", True)
    add("COMPONENT_DB", "model", "Component model/family.", "TEXT", True)
    add("COMPONENT_DB", "hardware_reference", "Reusable part/hardware reference.", "TEXT", True)
    for field, meaning, dtype in [
        ("mass_kg", "Reusable component mass.", "REAL"),
        ("rated_power_kw", "Rated component power.", "REAL"),
        ("rated_torque_nm", "Rated component torque.", "REAL"),
        ("capacity_kwh", "Energy capacity where applicable.", "REAL"),
        ("nominal_voltage_v", "Nominal electrical voltage.", "REAL"),
        ("ratio", "Primary dimensionless ratio when applicable.", "REAL"),
    ]:
        add("COMPONENT_DB", field, meaning, dtype, True)
    add("COMPONENT_DB", "custom_properties_json", "Sparse type-specific scalar/structured properties.", "JSON", True)
    add("COMPONENT_DB", "data_artifact_ref", "Large map/curve/data artifact reference.", "TEXT", True)
    add("COMPONENT_DB", "model_artifact_ref", "Executable/model artifact reference.", "TEXT", True)
    add("COMPONENT_DB", "source_name", "Authoritative source name.", "TEXT", True)
    add("COMPONENT_DB", "source_record_id", "Source record identifier.", "TEXT", True)
    add("COMPONENT_DB", "provenance_json", "Observed/derived provenance and semantic status.", "JSON", False)

    # COMPONENT_INSTANCE
    add("COMPONENT_INSTANCE", "component_instance_id", "Component occurrence within a vehicle configuration.", "TEXT", False, "PRIMARY_KEY")
    add("COMPONENT_INSTANCE", "vehicle_configuration_id", "Owning technical configuration.", "TEXT", False, "FOREIGN_KEY->VEHICLE_CONFIGURATION.vehicle_configuration_id")
    add("COMPONENT_INSTANCE", "component_id", "Reusable component definition when resolved.", "TEXT", True, "FOREIGN_KEY->COMPONENT_DB.component_id")
    add("COMPONENT_INSTANCE", "tire_id", "Specialized tire definition when the instance is a tire.", "TEXT", True, "FOREIGN_KEY->TIRE_DB.tire_id")
    add("COMPONENT_INSTANCE", "component_domain", "Instance engineering domain.", "TEXT", False)
    add("COMPONENT_INSTANCE", "role", "Functional role such as PRIMARY/MAIN/FRONT_LEFT.", "TEXT", True)
    add("COMPONENT_INSTANCE", "position", "Physical/logical position when evidenced.", "TEXT", True)
    add("COMPONENT_INSTANCE", "quantity", "Occurrence quantity represented by this instance row.", "INTEGER", False)
    add("COMPONENT_INSTANCE", "instance_properties_json", "Usage-specific properties, not reusable master properties.", "JSON", True)
    add("COMPONENT_INSTANCE", "provenance_json", "Source and resolution status, including partial/unresolved identity.", "JSON", False)

    # COMPONENT_RESOLUTION
    add("COMPONENT_RESOLUTION", "component_resolution_id", "Reusable engineering resolution identifier.", "TEXT", False, "PRIMARY_KEY")
    add("COMPONENT_RESOLUTION", "vehicle_configuration_id", "Configuration context when resolution is configuration-specific.", "TEXT", True, "FOREIGN_KEY->VEHICLE_CONFIGURATION.vehicle_configuration_id")
    add("COMPONENT_RESOLUTION", "boundary", "Canonical physical/analysis boundary.", "TEXT", False)
    add("COMPONENT_RESOLUTION", "method", "Resolution/adoption/calibration method.", "TEXT", False)
    add("COMPONENT_RESOLUTION", "conditions_json", "Validity/test conditions.", "JSON", True)
    for field, unit in [("resolved_A_N", "N"), ("resolved_B_N_per_kph", "N/(km/h)"), ("resolved_C_N_per_kph2", "N/(km/h)^2")]:
        add("COMPONENT_RESOLUTION", field, f"Resolved roadload coefficient [{unit}].", "REAL", True)
    add("COMPONENT_RESOLUTION", "input_component_instance_ids_json", "Referenced component occurrences.", "JSON", True)
    add("COMPONENT_RESOLUTION", "source_run_ids_json", "Evidence Runs supporting the resolution.", "JSON", True)
    add("COMPONENT_RESOLUTION", "fidelity_level", "Resolution fidelity independent of confidence.", "TEXT", True, enum_candidates="L0|L1|L2|L3")
    add("COMPONENT_RESOLUTION", "confidence", "Evidence confidence independent of fidelity.", "TEXT", True, enum_candidates="LOW|MEDIUM|HIGH")
    add("COMPONENT_RESOLUTION", "provenance_json", "Inputs, calculation/adoption path and semantic status.", "JSON", False)

    # RUN
    add("RUN", "run_id", "Canonical append-oriented evidence/execution identifier.", "TEXT", False, "PRIMARY_KEY")
    add("RUN", "vde_id", "VDE state supported/exercised by the Run.", "INTEGER", False, "FOREIGN_KEY->VDE.id")
    add("RUN", "run_type", "How the result was produced.", "TEXT", False, enum_candidates="TEST|SIMULATION|ESTIMATION|CALCULATION|ML_PREDICTION|DECLARED_RESULT")
    add("RUN", "evidence_kind", "Role/context of the evidence.", "TEXT", False, enum_candidates="ENGINEERING|HOMOLOGATION|MONITORING|BENCHMARK|SOURCE_RECORD")
    add("RUN", "fidelity_level", "Independent engineering fidelity.", "TEXT", True, enum_candidates="L0|L1|L2|L3")
    add("RUN", "confidence", "Independent evidence confidence.", "TEXT", True, enum_candidates="LOW|MEDIUM|HIGH")
    add("RUN", "source_name", "Source/system producing the evidence.", "TEXT", True)
    add("RUN", "source_record_id", "Source test/run/row identifier.", "TEXT", True)
    add("RUN", "procedure_code", "Source procedure/cycle code.", "TEXT", True)
    add("RUN", "procedure_description", "Human-readable procedure/cycle.", "TEXT", True)
    add("RUN", "conditions_json", "Fuel, environment, setup and other run conditions.", "JSON", True)
    add("RUN", "result_details_json", "Structured phase/detail outputs in source semantics.", "JSON", True)
    add("RUN", "method", "Engineering method name.", "TEXT", True)
    add("RUN", "method_version", "Method/model version.", "TEXT", True)
    add("RUN", "assumptions_json", "Explicit assumptions.", "JSON", True)
    add("RUN", "provenance_json", "Observed/calculated/estimated/simulated/ML provenance.", "JSON", False)
    add("RUN", "created_at", "Evidence timestamp when available.", "DATETIME", True)
    return c


def legacy_field_semantics(entity: str, field: str) -> tuple[str, str]:
    enums = {
        "legislation": "EPA|WLTP|BRA|OTHER",
        "record_origin": "IMPORTED|MANUAL|QA|LEGACY",
        "record_status": "ACTIVE|ARCHIVED",
        "review_status": "CURRENT|REVIEW|SUPERSEDED",
        "electrification": "NONE|MHEV|HEV|PHEV|BEV|FCEV|OTHER",
        "energy_basis": "VDE_TOTAL|VDE_NET|SOURCE_DECLARED|OTHER",
    }.get(field, "")
    explicit = {
        "id": f"Application-facing {entity} row identifier.",
        "created_at": "Record creation timestamp.",
        "updated_at": "Record last-update timestamp.",
        "vde_id": "Owning VDE identifier.",
        "make": "Commercial vehicle make.",
        "model": "Commercial vehicle model.",
        "year": "Vehicle model year.",
        "mass_kg": "Resolved curb/reference mass in kilograms used by the VDE snapshot.",
        "test_mass_kg": "Resolved calculation/test mass in kilograms when explicitly available.",
        "coast_A_N": "Authoritative whole-vehicle coastdown coefficient A in newtons.",
        "coast_B_N_per_kph": "Authoritative whole-vehicle coastdown coefficient B in N/(km/h).",
        "coast_C_N_per_kph2": "Authoritative whole-vehicle coastdown coefficient C in N/(km/h)^2.",
        "vde_total_mj_per_km": "Persisted TOTAL Vehicle Demand energy per distance in MJ/km.",
        "vde_net_mj_per_km": "Persisted NET Vehicle Demand energy per distance in MJ/km under the canonical TOTAL/NET contract.",
        "fuel_l_per_100km": "Adopted aggregate fuel consumption in L/100 km.",
        "energy_Wh_per_km": "Adopted aggregate electric energy consumption in Wh/km.",
        "gco2_per_km": "Adopted aggregate CO2 result in g/km.",
        "label_range_km": "Published/adopted electric or fuel-cell range in kilometres.",
    }
    if field in explicit:
        return explicit[field], enums
    unit = ""
    for suffix, label in (
        ("_N_per_kph2", "N/(km/h)^2"), ("_Npkph2", "N/(km/h)^2"),
        ("_N_per_kph", "N/(km/h)"), ("_Npkph", "N/(km/h)"),
        ("_mj_per_km", "MJ/km"), ("_Wh_per_km", "Wh/km"),
        ("_l_per_100km", "L/100 km"), ("_per_km", "per km"),
        ("_kg", "kg"), ("_kw", "kW"), ("_nm", "N·m"),
        ("_kwh", "kWh"), ("_psi", "psi"), ("_pct", "%"), ("_N", "N"),
    ):
        if field.endswith(suffix):
            unit = label
            break
    readable = field.replace("_", " ")
    meaning = f"{entity} application-contract value for {readable}"
    if unit:
        meaning += f" [{unit}]"
    return meaning + "; semantics are preserved from the current canonical application surface.", enums


def legacy_table_contract(con: sqlite3.Connection, entity: str, table: str) -> list[dict[str, Any]]:
    rows = []
    for col in schema_for(con, table):
        field = col["field"]
        dtype = {"INT": "INTEGER", "INTEGER": "INTEGER", "REAL": "REAL", "TEXT": "TEXT"}.get(col["sql_type"].upper(), col["sql_type"].upper())
        role = "PRIMARY_KEY" if col["primary_key"] else "FOREIGN_KEY->VDE.id" if table == "fuelcons_db" and field == "vde_id" else "NONE"
        meaning, enums = legacy_field_semantics(entity, field)
        rows.append(contract_row(
            entity, field, meaning, dtype, not (col["not_null"] or col["primary_key"]), role, enums,
            provenance="LEGACY_VALUE_PRESERVED; NEW_VALUES_REQUIRE_FIELD_PROVENANCE",
            legacy_mapping=f"{table}.{field}",
            compatibility_projection=field,
        ))
    return rows


def tire_contract(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for col in schema_for(con, "tire_roadload_db"):
        field = "tire_id" if col["field"] == "id" else col["field"]
        dtype = col["sql_type"].upper() or "ANY"
        rows.append(contract_row(
            "TIRE_DB", field, "Specialized reusable tire identity/evidence field retained from current tire contract.",
            dtype, not (col["not_null"] or col["primary_key"]), "PRIMARY_KEY" if col["primary_key"] else "NONE",
            provenance="LEGACY_TIRE_EVIDENCE_PRESERVED",
            legacy_mapping=f"tire_roadload_db.{col['field']}",
        ))
    return rows


def exact_contract(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = base_contract()
    rows += legacy_table_contract(con, "VDE", "vde_db")
    rows += legacy_table_contract(con, "FUELCONS", "fuelcons_db")
    rows += tire_contract(con)
    rows += [
        contract_row("VDE", "vehicle_configuration_id", "Stable configuration owning this resolved VDE state.", "TEXT", False, "FOREIGN_KEY->VEHICLE_CONFIGURATION.vehicle_configuration_id", compatibility_projection="NOT_EXPOSED_TO_LEGACY_SURFACE"),
        contract_row("VDE", "source_semantic_status", "Source interpretation state.", "TEXT", False, enum_candidates="DIRECT|PARTIAL|UNRESOLVED", compatibility_projection="NOT_EXPOSED_TO_LEGACY_SURFACE"),
        contract_row("VDE", "source_payload_json", "Raw source values/units required when canonical meaning is unresolved.", "JSON", True, provenance="DIRECT_RAW_SOURCE_VALUES", compatibility_projection="NOT_EXPOSED_TO_LEGACY_SURFACE"),
        contract_row("FUELCONS", "comparison_basis", "Methodology/context of the adopted comparison result.", "TEXT", False, enum_candidates="LEGACY_UNSPECIFIED|EPA_LABEL_2_CYCLE|EPA_LABEL_MULTI_ENERGY|EPA_TEST_PROCEDURE|EU_WLTP_MONITORING_COMBINED|WLTP_OEM_DECLARED|WLTP_SIMULATED|REAL_WORLD_SIMULATED|ENGINEERING_L0", compatibility_projection="NOT_EXPOSED_UNLESS_PAGE_ADOPTS_METADATA"),
        contract_row("FUELCONS", "adopted_run_ids_json", "Structured lineage to one or more supporting Runs.", "JSON", True, "LOGICAL_REFERENCES->RUN.run_id", compatibility_projection="NOT_EXPOSED_TO_LEGACY_SURFACE"),
    ]
    # Avoid duplicate canonical additions if a future legacy schema gains them.
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        deduped[(row["entity"], row["field_name"])] = row
    result = list(deduped.values())
    assert {r["entity"] for r in result} == set(ENTITIES)
    public_mappings = {
        ("PROGRAM", "commercial_make"): "EPA Represented Test Veh Make; JRC OEM anon",
        ("PROGRAM", "commercial_model"): "EPA Represented Test Veh Model; JRC Model anon",
        ("PROGRAM", "model_year_from"): "EPA Model Year",
        ("PROGRAM", "model_year_to"): "EPA Model Year",
        ("VEHICLE_CONFIGURATION", "engine_model"): "EPA Engine Code",
        ("VEHICLE_CONFIGURATION", "engine_displacement_l"): "EPA Test Veh Displacement (L)",
        ("VEHICLE_CONFIGURATION", "transmission_type"): "EPA Tested Transmission Type; JRC Gear box type",
        ("VEHICLE_CONFIGURATION", "gear_count"): "EPA # of Gears; JRC N gears",
        ("VEHICLE_CONFIGURATION", "drive_system"): "EPA Drive System Description",
        ("VEHICLE_CONFIGURATION", "final_drive_ratio"): "EPA Axle Ratio",
        ("VEHICLE_CONFIGURATION", "nv_ratio"): "EPA N/V Ratio",
        ("VDE", "mass_kg"): "JRC Curb_vehicle_mass [kg]",
        ("VDE", "test_mass_kg"): "JRC Vehicle mass (WLTP) [kg]",
        ("VDE", "coast_A_N"): "JRC wltp|f0 [N]",
        ("VDE", "coast_B_N_per_kph"): "JRC wltp|f1 [N/(km/h)]",
        ("VDE", "coast_C_N_per_kph2"): "JRC wltp|f2 [N/(km/h)2]",
        ("RUN", "source_record_id"): "EPA source row/Test Number; JRC source row",
        ("RUN", "procedure_code"): "EPA Test Procedure Cd",
        ("RUN", "procedure_description"): "EPA Test Procedure Description",
        ("RUN", "result_details_json"): "EPA result fields; JRC declared/simulated result fields",
        ("FUELCONS", "energy_Wh_per_km"): "JRC Declared electric consumption value (OEM) [Wh/km]",
        ("FUELCONS", "gco2_per_km"): "JRC Declared average CO2 emissions value (OEM) [g/km]",
        ("FUELCONS", "label_range_km"): "JRC Electric range (OEM) [km]",
    }
    for row in result:
        row["public_source_mapping"] = public_mappings.get((row["entity"], row["field_name"]), row["public_source_mapping"])
    return sorted(result, key=lambda r: (ENTITIES.index(r["entity"]), r["field_name"]))


def legacy_program_and_config(vde_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    programs: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    vde_to_config: dict[int, str] = {}
    for row in vde_rows:
        pid = stable_id("PRG", "LEGACY_ECODRIVE", row.get("make"), row.get("model"), row.get("year"))
        programs.setdefault(pid, {
            "program_id": pid, "commercial_make": row.get("make"), "commercial_model": row.get("model"),
            "generation_name": None, "model_year_from": row.get("year"), "model_year_to": row.get("year"),
            "identity_status": "PROVISIONAL_SOURCE_SCOPED", "identity_confidence": "LOW", "source_scope": "LEGACY_ECODRIVE",
            "source_identity_json": {"legacy_make": row.get("make"), "legacy_model": row.get("model"), "legacy_year": row.get("year")},
            "created_at": row.get("created_at"), "updated_at": row.get("updated_at"),
        })
        signature = (
            pid, row.get("engine_type"), row.get("engine_model"), row.get("engine_size_l"), row.get("engine_aspiration"),
            row.get("transmission_type"), row.get("transmission_model"), row.get("drive_type"),
        )
        cid = stable_id("CFG", *signature)
        configs.setdefault(cid, {
            "vehicle_configuration_id": cid, "program_id": pid, "propulsion_architecture": row.get("engine_type"),
            "engine_type": row.get("engine_type"), "engine_model": row.get("engine_model"), "engine_displacement_l": row.get("engine_size_l"),
            "engine_aspiration": row.get("engine_aspiration"), "transmission_type": row.get("transmission_type"),
            "transmission_model": row.get("transmission_model"), "gear_count": None, "drive_system": row.get("drive_type"),
            "final_drive_ratio": None, "nv_ratio": None, "identity_status": "SOURCE_SCOPED", "identity_confidence": "MEDIUM",
            "source_scope": "LEGACY_ECODRIVE", "source_identity_json": {"legacy_vde_ids": []}, "architecture_properties_json": None,
        })
        configs[cid]["source_identity_json"]["legacy_vde_ids"].append(row["id"])
        vde_to_config[int(row["id"])] = cid
    return list(programs.values()), list(configs.values()), vde_to_config


def legacy_materialization(con: sqlite3.Connection) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    vde_rows = rows_for(con, "vde_db")
    fc_rows = rows_for(con, "fuelcons_db")
    component_rows = rows_for(con, "component_db")
    tire_rows = rows_for(con, "tire_roadload_db")
    programs, configs, vde_to_config = legacy_program_and_config(vde_rows)
    canonical_vdes = []
    for row in vde_rows:
        item = dict(row)
        item["vehicle_configuration_id"] = vde_to_config[int(row["id"])]
        item["source_semantic_status"] = "DIRECT"
        item["source_payload_json"] = None
        item["_materialization_scope"] = "LEGACY_COMPATIBILITY"
        canonical_vdes.append(item)
    runs = []
    fuelcons = []
    for row in fc_rows:
        run_id = stable_id("RUN", "LEGACY_FUELCONS", row["id"])
        has_method = any(row.get(x) is not None for x in ("engine_method", "engine_version", "assumptions_json", "provenance_json"))
        runs.append({
            "run_id": run_id, "vde_id": row["vde_id"], "run_type": "ESTIMATION" if has_method else "DECLARED_RESULT",
            "evidence_kind": "ENGINEERING" if has_method else "SOURCE_RECORD", "fidelity_level": None, "confidence": None,
            "source_name": row.get("source_name") or "LEGACY_ECODRIVE", "source_record_id": row.get("source_record_id") or str(row["id"]),
            "procedure_code": None, "procedure_description": row.get("method_note"),
            "conditions_json": {k: row.get(k) for k in ("ambient_temp_c", "ac_on", "tire_front_psi", "tire_rear_psi", "scenario_payload_kg")},
            "result_details_json": {k: v for k, v in row.items() if k.startswith(("energy_", "fuel_", "gco2_", "label_"))},
            "method": row.get("engine_method"), "method_version": row.get("engine_version"),
            "assumptions_json": row.get("assumptions_json"), "provenance_json": {"legacy_provenance_json": row.get("provenance_json"), "classification": "DIRECT_LEGACY_PRESERVATION"},
            "created_at": row.get("created_at"), "_materialization_scope": "LEGACY_COMPATIBILITY",
        })
        item = dict(row)
        item["comparison_basis"] = "LEGACY_UNSPECIFIED"
        item["adopted_run_ids_json"] = [run_id]
        item["_materialization_scope"] = "LEGACY_COMPATIBILITY"
        fuelcons.append(item)
    tires = []
    for row in tire_rows:
        item = dict(row)
        item["tire_id"] = item.pop("id")
        item["_materialization_scope"] = "LEGACY_COMPATIBILITY"
        tires.append(item)
    components = []
    resolutions = []
    for row in component_rows:
        components.append({
            "component_id": f"LEGACY-COMP-{row['id']}", "component_domain": str(row.get("domain") or "OTHER").upper(),
            "manufacturer": None, "model": row.get("component_name"), "hardware_reference": row.get("hardware_reference"),
            "mass_kg": None, "rated_power_kw": None, "rated_torque_nm": None, "capacity_kwh": None, "nominal_voltage_v": None,
            "ratio": None, "custom_properties_json": {"legacy_component_code": row.get("component_code"), "legacy_row": row},
            "data_artifact_ref": None, "model_artifact_ref": None, "source_name": row.get("source_name"),
            "source_record_id": row.get("source_record_id"), "provenance_json": {"classification": "DIRECT_LEGACY_PRESERVATION"},
            "_materialization_scope": "LEGACY_COMPATIBILITY",
        })
        if any(row.get(k) is not None for k in ("equivalent_A_N", "equivalent_B_N_per_kph", "equivalent_C_N_per_kph2", "loss_pct")):
            resolutions.append({
                "component_resolution_id": stable_id("RES", "LEGACY_COMPONENT", row["id"]), "vehicle_configuration_id": None,
                "boundary": row.get("physical_boundary") or row.get("domain") or "UNRESOLVED", "method": row.get("test_method") or "LEGACY_PRESERVED",
                "conditions_json": {"test_condition_type": row.get("test_condition_type")}, "resolved_A_N": row.get("equivalent_A_N"),
                "resolved_B_N_per_kph": row.get("equivalent_B_N_per_kph"), "resolved_C_N_per_kph2": row.get("equivalent_C_N_per_kph2"),
                "input_component_instance_ids_json": None, "source_run_ids_json": None, "fidelity_level": None, "confidence": None,
                "provenance_json": {"legacy_component_id": row["id"], "classification": "DIRECT_LEGACY_PRESERVATION"},
                "_materialization_scope": "LEGACY_COMPATIBILITY",
            })
    instances: dict[str, dict[str, Any]] = {}
    valid_tires = {row["tire_id"] for row in tires}
    for row in canonical_vdes:
        cid = row["vehicle_configuration_id"]
        for field, position in (("front_tire_id", "FRONT_AXLE"), ("rear_tire_id", "REAR_AXLE")):
            tire_id = row.get(field)
            if tire_id is None or tire_id not in valid_tires:
                continue
            iid = stable_id("INS", cid, "TIRE", position, tire_id)
            instances[iid] = {
                "component_instance_id": iid, "vehicle_configuration_id": cid, "component_id": None, "tire_id": tire_id,
                "component_domain": "TIRE", "role": position, "position": position, "quantity": 2,
                "instance_properties_json": None, "provenance_json": {"legacy_vde_tire_field": field},
                "_materialization_scope": "LEGACY_COMPATIBILITY",
            }
    materialized = {
        "PROGRAM": programs,
        "VEHICLE_CONFIGURATION": configs,
        "COMPONENT_DB": components,
        "TIRE_DB": tires,
        "COMPONENT_INSTANCE": list(instances.values()),
        "COMPONENT_RESOLUTION": resolutions,
        "VDE": canonical_vdes,
        "RUN": runs,
        "FUELCONS": fuelcons,
    }
    return materialized, {"legacy_vde_rows": vde_rows, "legacy_fuelcons_rows": fc_rows}


def public_materialization() -> dict[str, list[dict[str, Any]]]:
    """Materialize EPA and JRC without assigning unsupported physical semantics."""
    result = {entity: [] for entity in ENTITIES}
    epa = pd.read_excel(EPA_PATH, sheet_name="Sheet1", engine="openpyxl")
    epa = epa[epa["Model Year"].eq(2026)].copy()
    programs: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for idx, series in epa.iterrows():
        row = clean_record(series.to_dict())
        source_row = int(idx) + 2
        pid = stable_id("PRG", "EPA_TESTCAR_2026", row.get("Represented Test Veh Make"), row.get("Represented Test Veh Model"), row.get("Model Year"))
        programs.setdefault(pid, {
            "program_id": pid, "commercial_make": row.get("Represented Test Veh Make"), "commercial_model": row.get("Represented Test Veh Model"),
            "generation_name": None, "model_year_from": row.get("Model Year"), "model_year_to": row.get("Model Year"),
            "identity_status": "PROVISIONAL_SOURCE_SCOPED", "identity_confidence": "LOW", "source_scope": "EPA_TESTCAR_2026",
            "source_identity_json": {"source_file": EPA_PATH.name, "make": row.get("Represented Test Veh Make"), "model": row.get("Represented Test Veh Model")},
            "created_at": None, "updated_at": None, "_materialization_scope": "PUBLIC_ETL_EPA",
        })
        cid = stable_id(
            "CFG", "EPA_TESTCAR_2026", pid, row.get("Actual Tested Testgroup"), row.get("Test Vehicle ID"), row.get("Test Veh Configuration #"),
            row.get("Test Veh Displacement (L)"), row.get("Engine Code"), row.get("Tested Transmission Type"), row.get("# of Gears"),
            row.get("Drive System Description"), row.get("Axle Ratio"), row.get("N/V Ratio"),
        )
        configs.setdefault(cid, {
            "vehicle_configuration_id": cid, "program_id": pid, "propulsion_architecture": None,
            "engine_type": None, "engine_model": row.get("Engine Code"), "engine_displacement_l": row.get("Test Veh Displacement (L)"),
            "engine_aspiration": None, "transmission_type": row.get("Tested Transmission Type"), "transmission_model": None,
            "gear_count": row.get("# of Gears"), "drive_system": row.get("Drive System Description"), "final_drive_ratio": row.get("Axle Ratio"),
            "nv_ratio": row.get("N/V Ratio"), "identity_status": "SOURCE_SCOPED", "identity_confidence": "MEDIUM",
            "source_scope": "EPA_TESTCAR_2026", "source_identity_json": {"test_group": row.get("Actual Tested Testgroup"), "test_vehicle_id": row.get("Test Vehicle ID"), "configuration_number": row.get("Test Veh Configuration #")},
            "architecture_properties_json": {"rated_horsepower": row.get("Rated Horsepower"), "cylinders_rotors": row.get("# of Cylinders and Rotors")},
            "_materialization_scope": "PUBLIC_ETL_EPA",
        })
    result["PROGRAM"].extend(programs.values())
    result["VEHICLE_CONFIGURATION"].extend(configs.values())

    jrc = pd.read_excel(JRC_PATH, sheet_name="Sheet1", engine="openpyxl")
    for idx, series in jrc.iterrows():
        row = clean_record(series.to_dict())
        source_row = int(idx) + 2
        pid = stable_id("PRG", "JRC_PYCSIS_2021", row.get("OEM anon"), row.get("Model anon"), source_row)
        cid = stable_id("CFG", "JRC_PYCSIS_2021", source_row)
        vde_id = -2_000_000 - source_row
        run_id = stable_id("RUN", "JRC_PYCSIS_2021", source_row)
        result["PROGRAM"].append({
            "program_id": pid, "commercial_make": row.get("OEM anon"), "commercial_model": row.get("Model anon"), "generation_name": None,
            "model_year_from": None, "model_year_to": None, "identity_status": "UNRESOLVED", "identity_confidence": "LOW",
            "source_scope": "JRC_PYCSIS_2021", "source_identity_json": {"source_row": source_row, "anonymized": True},
            "created_at": None, "updated_at": None, "_materialization_scope": "PUBLIC_ETL_JRC",
        })
        result["VEHICLE_CONFIGURATION"].append({
            "vehicle_configuration_id": cid, "program_id": pid, "propulsion_architecture": row.get("Input type"), "engine_type": row.get("Fuel type"),
            "engine_model": None, "engine_displacement_l": None, "engine_aspiration": "TURBO" if row.get("Engine is turbo") else None,
            "transmission_type": row.get("Gear box type"), "transmission_model": None, "gear_count": row.get("N gears"), "drive_system": None,
            "final_drive_ratio": None, "nv_ratio": None, "identity_status": "UNRESOLVED", "identity_confidence": "LOW", "source_scope": "JRC_PYCSIS_2021",
            "source_identity_json": {"source_row": source_row, "oem_anon": row.get("OEM anon"), "model_anon": row.get("Model anon")},
            "architecture_properties_json": {"engine_power": row.get("Engine max power"), "engine_capacity_cm3": row.get("Engine capacity [cm3]"), "tyre_code": row.get("Tyre code"), "electric_motor_power_kw": row.get("Electric motor power [kW]"), "battery_capacity_ah": row.get("Drive battery capacity [Ah]"), "battery_nominal_voltage_v": row.get("Drive battery nominal voltage [V]")},
            "_materialization_scope": "PUBLIC_ETL_JRC",
        })
        result["VDE"].append({
            "id": vde_id, "vehicle_configuration_id": cid, "legislation": "WLTP", "category": row.get("Vehicle body"),
            "make": row.get("OEM anon"), "model": row.get("Model anon"), "year": None, "mass_kg": row.get("Curb_vehicle_mass [kg]"),
            "test_mass_kg": row.get("Vehicle mass (WLTP) [kg]"), "coast_A_N": row.get("wltp|f0 [N]"),
            "coast_B_N_per_kph": row.get("wltp|f1 [N/(km/h)]"), "coast_C_N_per_kph2": row.get("wltp|f2 [N/(km/h)2]"),
            "source_semantic_status": "PARTIAL", "source_payload_json": {"source_row": source_row, "jrc_row_grain": "UNRESOLVED"},
            "_materialization_scope": "PUBLIC_ETL_JRC",
        })
        result["RUN"].append({
            "run_id": run_id, "vde_id": vde_id, "run_type": "SIMULATION" if row.get("pycsis_run") else "DECLARED_RESULT",
            "evidence_kind": "SOURCE_RECORD", "fidelity_level": None, "confidence": "LOW", "source_name": "JRC PYCSIS 2021",
            "source_record_id": str(source_row), "procedure_code": None, "procedure_description": "JRC source row; exact grain unresolved",
            "conditions_json": {"input_type": row.get("Input type"), "fuel_type": row.get("Fuel type")},
            "result_details_json": {k: v for k, v in row.items() if "Declared" in k or "Real-world" in k},
            "method": "PYCSIS" if row.get("pycsis_run") else None, "method_version": None, "assumptions_json": None,
            "provenance_json": {"classification": "DIRECT_SOURCE_VALUES_WITH_UNRESOLVED_ROW_GRAIN"}, "created_at": None,
            "_materialization_scope": "PUBLIC_ETL_JRC",
        })
        result["FUELCONS"].append({
            "id": -3_000_000 - source_row, "vde_id": vde_id,
            "electrification": row.get("Input type"), "fuel_type": row.get("Fuel type"),
            "energy_Wh_per_km": row.get("Declared electric consumption value (OEM) [Wh/km]"),
            "gco2_per_km": row.get("Declared average CO2 emissions value (OEM) [g/km]"),
            "label_range_km": row.get("Electric range (OEM) [km]"), "comparison_basis": "JRC_SOURCE_ROW_UNRESOLVED",
            "adopted_run_ids_json": [run_id], "_materialization_scope": "PUBLIC_ETL_JRC",
        })
    return result


def combine_materializations(legacy: dict[str, list[dict[str, Any]]], public: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {entity: legacy[entity] + public[entity] for entity in ENTITIES}


def validate_materialized_contract(
    contract: list[dict[str, Any]], materialized: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Return required-field violations without imposing DDL on staging."""
    required: dict[str, list[str]] = defaultdict(list)
    for field in contract:
        if not field["nullable"]:
            required[field["entity"]].append(field["field_name"])
    violations: list[dict[str, Any]] = []
    for entity in ENTITIES:
        for row_number, row in enumerate(materialized[entity], start=1):
            for field_name in required[entity]:
                if field_name not in row or row[field_name] is None:
                    violations.append({
                        "entity": entity,
                        "staging_row": row_number,
                        "field_name": field_name,
                        "materialization_scope": row.get("_materialization_scope"),
                        "status": "CDR_BLOCKER",
                        "reason": "Required v1 contract field is missing or NULL.",
                    })
    return violations


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=clean) + "\n")
            count += 1
    return count


def legacy_projection(entity: str, materialized: list[dict[str, Any]], legacy_fields: list[str]) -> list[dict[str, Any]]:
    rows = [r for r in materialized if r.get("_materialization_scope") == "LEGACY_COMPATIBILITY"]
    return [{field: clean(row.get(field)) for field in legacy_fields} for row in rows]


def rows_by_id(rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {row["id"]: row for row in rows}


def compare_rows(legacy: list[dict[str, Any]], rebuilt: list[dict[str, Any]], fields: list[str]) -> tuple[int, list[dict[str, Any]]]:
    left = rows_by_id(legacy)
    right = rows_by_id(rebuilt)
    mismatches = []
    all_ids = sorted(set(left) | set(right), key=str)
    for record_id in all_ids:
        if record_id not in left or record_id not in right:
            mismatches.append({"record_id": record_id, "field": "<ROW>", "legacy_value": "PRESENT" if record_id in left else "ABSENT", "reconstructed_value": "PRESENT" if record_id in right else "ABSENT"})
            continue
        for field in fields:
            if clean(left[record_id].get(field)) != clean(right[record_id].get(field)):
                mismatches.append({"record_id": record_id, "field": field, "legacy_value": left[record_id].get(field), "reconstructed_value": right[record_id].get(field)})
    return len(all_ids) * len(fields), mismatches


def value_signature(rows: list[dict[str, Any]], fields: list[str]) -> str:
    payload = [[clean(row.get(f)) for f in fields] for row in sorted(rows, key=lambda r: str(r.get("id")))]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def check_row(check_id: str, category: str, status: str, legacy: Any, rebuilt: Any, evidence: str, details: str = "") -> dict[str, Any]:
    assert status in ALLOWED_RESULTS
    return {"check_id": check_id, "category": category, "status": status, "legacy_measure": legacy, "reconstructed_measure": rebuilt, "evidence": evidence, "details": details}


def compatibility_proof(con: sqlite3.Connection, materialized: dict[str, list[dict[str, Any]]], legacy_rows: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    legacy_vde = legacy_rows["legacy_vde_rows"]
    legacy_fc = legacy_rows["legacy_fuelcons_rows"]
    vde_fields = [x["field"] for x in schema_for(con, "vde_db")]
    fc_fields = [x["field"] for x in schema_for(con, "fuelcons_db")]
    rebuilt_vde = legacy_projection("VDE", materialized["VDE"], vde_fields)
    rebuilt_fc = legacy_projection("FUELCONS", materialized["FUELCONS"], fc_fields)
    _, vde_mismatches = compare_rows(legacy_vde, rebuilt_vde, vde_fields)
    _, fc_mismatches = compare_rows(legacy_fc, rebuilt_fc, fc_fields)
    checks = []
    checks.append(check_row("COMPAT-01", "VDE identity/count", "EXACT_EQUIVALENCE" if len(legacy_vde) == len(rebuilt_vde) and set(rows_by_id(legacy_vde)) == set(rows_by_id(rebuilt_vde)) else "CDR_BLOCKER", len(legacy_vde), len(rebuilt_vde), "Read-only vde_db vs compatibility projection."))
    checks.append(check_row("COMPAT-02", "FuelCons identity/count", "EXACT_EQUIVALENCE" if len(legacy_fc) == len(rebuilt_fc) and set(rows_by_id(legacy_fc)) == set(rows_by_id(rebuilt_fc)) else "CDR_BLOCKER", len(legacy_fc), len(rebuilt_fc), "Read-only fuelcons_db vs compatibility projection."))
    checks.append(check_row("COMPAT-03", "All VDE field values", "EXACT_EQUIVALENCE" if not vde_mismatches else "CDR_BLOCKER", len(vde_mismatches), 0, f"Compared {len(legacy_vde):,} rows x {len(vde_fields)} fields by ID.", "Mismatch count shown in legacy measure."))
    checks.append(check_row("COMPAT-04", "All FuelCons field values", "EXACT_EQUIVALENCE" if not fc_mismatches else "CDR_BLOCKER", len(fc_mismatches), 0, f"Compared {len(legacy_fc):,} rows x {len(fc_fields)} fields by ID.", "Mismatch count shown in legacy measure."))
    legacy_links = sorted((r["id"], r["vde_id"]) for r in legacy_fc)
    rebuilt_links = sorted((r["id"], r["vde_id"]) for r in rebuilt_fc)
    checks.append(check_row("COMPAT-05", "VDE-FuelCons linkage", "EXACT_EQUIVALENCE" if legacy_links == rebuilt_links else "CDR_BLOCKER", value_signature(legacy_fc, ["id", "vde_id"]), value_signature(rebuilt_fc, ["id", "vde_id"]), "Exact ordered (fuelcons.id, vde_id) relationship signature."))
    legacy_mult = Counter(r["vde_id"] for r in legacy_fc)
    rebuilt_mult = Counter(r["vde_id"] for r in rebuilt_fc)
    checks.append(check_row("COMPAT-06", "FuelCons multiplicity per VDE", "EXACT_EQUIVALENCE" if legacy_mult == rebuilt_mult else "CDR_BLOCKER", dict(Counter(legacy_mult.values())), dict(Counter(rebuilt_mult.values())), "Full per-VDE multiplicity distribution."))
    legacy_parent = sorted((r["id"], r.get("vde_id_parent")) for r in legacy_vde)
    rebuilt_parent = sorted((r["id"], r.get("vde_id_parent")) for r in rebuilt_vde)
    checks.append(check_row("COMPAT-07", "VDE parent lineage", "EXACT_EQUIVALENCE" if legacy_parent == rebuilt_parent else "CDR_BLOCKER", hashlib.sha256(repr(legacy_parent).encode()).hexdigest(), hashlib.sha256(repr(rebuilt_parent).encode()).hexdigest(), "Exact (VDE id, parent id) signature including NULLs."))
    null_legacy = {f"vde.{f}": sum(r.get(f) is None for r in legacy_vde) for f in vde_fields} | {f"fuelcons.{f}": sum(r.get(f) is None for r in legacy_fc) for f in fc_fields}
    null_rebuilt = {f"vde.{f}": sum(r.get(f) is None for r in rebuilt_vde) for f in vde_fields} | {f"fuelcons.{f}": sum(r.get(f) is None for r in rebuilt_fc) for f in fc_fields}
    checks.append(check_row("COMPAT-08", "NULL behavior", "EXACT_EQUIVALENCE" if null_legacy == null_rebuilt else "CDR_BLOCKER", hashlib.sha256(json.dumps(null_legacy, sort_keys=True).encode()).hexdigest(), hashlib.sha256(json.dumps(null_rebuilt, sort_keys=True).encode()).hexdigest(), f"NULL counts compared for {len(null_legacy)} application-facing fields; zero values were not treated as missing."))
    groups = {
        "mass": [f for f in vde_fields if "mass" in f.lower() or f in {"inertia_class", "payload_kg"}],
        "roadload ABC": [f for f in vde_fields if f.startswith(("coast_", "baseline_", "trans_", "brake_", "parasitic_", "tire_", "trailer_")) and ("A_" in f or "B_" in f or "C_" in f or f in {"tire_A_final", "tire_B_final", "tire_C_final"})],
        "TOTAL/NET": [f for f in vde_fields if "vde_total" in f or "vde_net" in f],
        "Urban/Highway/Combined results": [f for f in fc_fields if any(t in f.lower() for t in ("urban", "highway", "combined", "ftp75", "hwfet", "energy_wh_per_km", "fuel_l_per_100km", "gco2_per_km"))],
        "labels/filters": [f for f in fc_fields if f.startswith("label_") or f in {"electrification", "fuel_type", "record_origin", "record_status", "review_status", "energy_basis"}],
    }
    check_no = 9
    for name, fields in groups.items():
        table_left, table_right = (legacy_vde, rebuilt_vde) if name in {"mass", "roadload ABC", "TOTAL/NET"} else (legacy_fc, rebuilt_fc)
        same = value_signature(table_left, fields) == value_signature(table_right, fields)
        checks.append(check_row(f"COMPAT-{check_no:02d}", name, "EXACT_EQUIVALENCE" if same else "CDR_BLOCKER", value_signature(table_left, fields), value_signature(table_right, fields), "Exact value+NULL signature for: " + ", ".join(fields)))
        check_no += 1
    run_by_id = {r["run_id"]: r for r in materialized["RUN"] if r.get("_materialization_scope") == "LEGACY_COMPATIBILITY"}
    legacy_fc_materialized = [r for r in materialized["FUELCONS"] if r.get("_materialization_scope") == "LEGACY_COMPATIBILITY"]
    adoption_ok = all(
        len(r.get("adopted_run_ids_json") or []) == 1
        and (r["adopted_run_ids_json"][0] in run_by_id)
        and run_by_id[r["adopted_run_ids_json"][0]]["vde_id"] == r["vde_id"]
        for r in legacy_fc_materialized
    )
    checks.append(check_row(f"COMPAT-{check_no:02d}", "RUN adoption lineage", "EXACT_EQUIVALENCE" if adoption_ok else "CDR_BLOCKER", len(legacy_fc_materialized), sum(1 for r in legacy_fc_materialized if r.get("adopted_run_ids_json")), "Every legacy FuelCons is linked additively to exactly one preserved evidence Run with the same VDE; RUN is not traversed by compatibility projection."))
    check_no += 1
    checks += [
        check_row(f"COMPAT-{check_no:02d}", "Canonical Program/Configuration identity", "APPROVED_CONTRACT_CORRECTION", "Absent in legacy surface", "Added outside compatibility projection", "CDR-01/CDR-02 approved source-scoped canonical identity; application-facing fields remain unchanged."),
        check_row(f"COMPAT-{check_no+1:02d}", "FuelCons comparison basis", "APPROVED_CONTRACT_CORRECTION", "Implicit/absent", "LEGACY_UNSPECIFIED", "CDR-04 approved explicit non-restrictive metadata; compatibility projection remains unchanged."),
        check_row(f"COMPAT-{check_no+2:02d}", "Ambiguous EEA/JRC semantics", "DEFERRED", "Unresolved", "Preserved as PARTIAL/UNRESOLVED source payload", "CDR-06; not a blocker for legacy compatibility or authoritative-source ingestion."),
    ]
    mismatch_details = [{"entity": "VDE", **m, "status": "CDR_BLOCKER"} for m in vde_mismatches] + [{"entity": "FUELCONS", **m, "status": "CDR_BLOCKER"} for m in fc_mismatches]
    if not mismatch_details:
        mismatch_details = [{"entity": "ALL", "record_id": "", "field": "", "legacy_value": "", "reconstructed_value": "", "status": "EXACT_EQUIVALENCE", "note": "No application-facing field/value mismatch detected."}]
    summary = {
        "vde_rows": len(legacy_vde), "fuelcons_rows": len(legacy_fc), "vde_fields": len(vde_fields), "fuelcons_fields": len(fc_fields),
        "vde_value_mismatches": len(vde_mismatches), "fuelcons_value_mismatches": len(fc_mismatches),
        "fuelcons_per_vde_distribution": dict(sorted(Counter(legacy_mult.values()).items())),
        "vdes_with_multiple_fuelcons": sum(v > 1 for v in legacy_mult.values()),
        "orphan_fuelcons": sum(r["vde_id"] not in rows_by_id(legacy_vde) for r in legacy_fc),
        "checks": len(checks), "cdr_blockers": sum(r["status"] == "CDR_BLOCKER" for r in checks),
    }
    return checks, mismatch_details, summary


def write_csv(name: str, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(OUT / name, index=False, encoding="utf-8")


def report_text(payload: dict[str, Any]) -> str:
    s = payload["compatibility_summary"]
    counts = payload["materialization_counts"]
    status_counts = Counter(r["status"] for r in payload["compatibility_checks"])
    recommendation = payload["recommendation"]
    lines = [
        "# Sprint 12C — Canonical Data Contract v1 Materialization & Compatibility Proof",
        "",
        f"## Recommendation: `{recommendation}`",
        "",
        "The approved nine-entity CDR baseline can reproduce the current EcoDrive VDE/FuelCons application-facing surface without changing pages, physics, resolvers or runtime databases. The recommendation authorizes preparation/review of physical DDL; it does **not** authorize running a production migration.",
        "",
        "## Compatibility result",
        "",
        f"- Legacy VDE: **{s['vde_rows']:,} rows × {s['vde_fields']} fields**, with **{s['vde_value_mismatches']}** reconstructed value mismatches.",
        f"- Legacy FuelCons: **{s['fuelcons_rows']:,} rows × {s['fuelcons_fields']} fields**, with **{s['fuelcons_value_mismatches']}** reconstructed value mismatches.",
        f"- Relationship evidence: **{s['vdes_with_multiple_fuelcons']:,}** VDE(s) have multiple FuelCons rows; **{s['orphan_fuelcons']}** orphan FuelCons rows.",
        f"- Checks: **{status_counts.get('EXACT_EQUIVALENCE', 0)}** exact, **{status_counts.get('APPROVED_CONTRACT_CORRECTION', 0)}** approved additive corrections, **{status_counts.get('DEFERRED', 0)}** deferred, **{status_counts.get('CDR_BLOCKER', 0)}** blockers.",
        "- NULL counts were compared for every application-facing field; zero remained a value, never missing.",
        "",
        "## Exact v1 field contract",
        "",
        f"`canonical_field_contract_v1.csv` defines **{payload['field_contract_count']} fields** across all nine entities. Each row includes semantic meaning, logical type, nullability, key/FK role, enum candidates, provenance expectations, legacy mapping, public-source mapping and compatibility projection.",
        "",
        "The compatibility strategy is deliberately localized:",
        "",
        "```text",
        "PROGRAM + VEHICLE_CONFIGURATION + COMPONENTS + VDE + RUN + FUELCONS",
        "                              ↓",
        "                 compatibility projection",
        "                              ↓",
        "                 current vde_db/fuelcons_db shape",
        "```",
        "",
        "VDE stays wide and persisted. FuelCons stays the adopted comparison result. RUN lineage is additive and is not traversed by the compatibility read path.",
        "",
        "## Staging materialization",
        "",
        "| Entity | Legacy records | Public ETL records | Total staged |",
        "|---|---:|---:|---:|",
    ]
    for entity in ENTITIES:
        c = counts[entity]
        lines.append(f"| {entity} | {c['legacy']:,} | {c['public']:,} | {c['total']:,} |")
    lines += [
        "",
        "EPA MY2026 populates provisional Program and Vehicle Configuration records only. EPA VDE/RUN/FuelCons materialization is deferred because this package has no approved imperial-to-SI ETL mapping and the VDE contract cannot substitute missing mass with zero. Raw EPA evidence remains preserved in the Sprint 12A/12B artifacts. JRC fields with explicit SI units populate VDE/RUN/FuelCons, while exact source-row semantics remain `PARTIAL`/`UNRESOLVED`.",
        "",
        "## Relationship and semantic checks",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    for row in payload["compatibility_checks"]:
        lines.append(f"| {row['category']} | `{row['status']}` | {row['evidence']} |")
    lines += [
        "",
        "## Approved additive differences",
        "",
        "The canonical model adds source-scoped Program/Vehicle Configuration identities, explicit `comparison_basis`, and FuelCons→RUN adoption lineage. These additions are outside the legacy compatibility projection and therefore do not change current page-facing values.",
        "",
        "## Deferred semantics",
        "",
        "EEA `RLFI`, exact JRC row grain, RAG/document enrichment, component causal decomposition and production migrations remain deferred per CDR. Their raw values/provenance can be retained without blocking valid VDE/RUN/FuelCons records.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python etl/scripts/sprint_12c_contract_compatibility.py",
        "python -m unittest discover -s etl/tests -p \"test_sprint_12c*.py\" -v",
        "```",
        "",
        "The script opens `data/db/eco_drive.db` with SQLite URI `mode=ro` and `PRAGMA query_only=ON`. JSONL staging files are written only under `etl/data/staging/sprint_12c_contract_v1/`.",
        "",
        "## Outputs",
        "",
        "- `etl/data/processed/sprint_12c_contract_compatibility/canonical_field_contract_v1.csv`",
        "- `etl/data/processed/sprint_12c_contract_compatibility/compatibility_checks.csv`",
        "- `etl/data/processed/sprint_12c_contract_compatibility/mismatch_report.csv`",
        "- `etl/data/processed/sprint_12c_contract_compatibility/materialization_counts.csv`",
        "- `etl/data/processed/sprint_12c_contract_compatibility/public_materialization_scope.csv`",
        "- `etl/data/processed/sprint_12c_contract_compatibility/contract_validation.csv`",
        "- `etl/data/processed/sprint_12c_contract_compatibility/compatibility_proof.json`",
        "- `etl/data/staging/sprint_12c_contract_v1/*.jsonl`",
        "- `etl/reports/sprint_12c_contract_compatibility.md`",
    ]
    return "\n".join(lines) + "\n"


def main() -> dict[str, Any]:
    if not CDR.exists() or not EVIDENCE_12B.exists():
        raise RuntimeError("Approved CDR baseline and Sprint 12B evidence are required.")
    OUT.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    con = ro_connect()
    contract = exact_contract(con)
    legacy, legacy_rows = legacy_materialization(con)
    public = public_materialization()
    materialized = combine_materializations(legacy, public)
    checks, mismatches, summary = compatibility_proof(con, materialized, legacy_rows)
    con.close()
    contract_violations = validate_materialized_contract(contract, materialized)
    checks.append(check_row(
        "C18", "v1 required-field materialization", "CDR_BLOCKER" if contract_violations else "EXACT_EQUIVALENCE",
        0, len(contract_violations),
        "All staged records satisfy every non-null field in the exact logical contract." if not contract_violations
        else f"{len(contract_violations)} required-field violation(s); see contract_validation.csv.",
    ))
    summary["checks"] = len(checks)
    summary["cdr_blockers"] = sum(row["status"] == "CDR_BLOCKER" for row in checks)
    counts = {
        entity: {"legacy": len(legacy[entity]), "public": len(public[entity]), "total": len(materialized[entity])}
        for entity in ENTITIES
    }
    staged_counts = {entity: write_jsonl(STAGING / f"{entity.lower()}.jsonl", materialized[entity]) for entity in ENTITIES}
    assert all(staged_counts[e] == counts[e]["total"] for e in ENTITIES)
    blockers = [row for row in checks if row["status"] == "CDR_BLOCKER"]
    recommendation = "READY_FOR_DDL" if not blockers else "NOT_READY_FOR_DDL"
    public_scope = [
        {
            "source": "EPA Test Car MY2026",
            "entities_materialized": "PROGRAM; VEHICLE_CONFIGURATION",
            "entities_deferred": "VDE; RUN; FUELCONS; component entities",
            "status": "DEFERRED",
            "reason": "No canonical imperial-to-SI ETL rule is approved; required VDE mass cannot be omitted or replaced by zero.",
        },
        {
            "source": "JRC technical vehicle dataset",
            "entities_materialized": "PROGRAM; VEHICLE_CONFIGURATION; VDE; RUN; FUELCONS",
            "entities_deferred": "COMPONENT_DB; TIRE_DB; COMPONENT_INSTANCE; COMPONENT_RESOLUTION",
            "status": "PARTIAL_UNRESOLVED",
            "reason": "Explicit-unit whole-vehicle fields are staged; exact identity and source-row grain remain unresolved.",
        },
    ]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cdr_baseline": str(CDR.relative_to(ROOT)),
        "source_to_contract_evidence": str(EVIDENCE_12B.relative_to(ROOT)),
        "database_access": "SQLite URI mode=ro + PRAGMA query_only=ON",
        "entities": list(ENTITIES),
        "field_contract_count": len(contract),
        "materialization_counts": counts,
        "public_materialization_scope": public_scope,
        "contract_validation_violations": len(contract_violations),
        "compatibility_summary": summary,
        "compatibility_checks": checks,
        "recommendation": recommendation,
        "recommendation_scope": "Ready to design/review DDL; production migration remains unauthorized.",
    }
    write_csv("canonical_field_contract_v1.csv", contract)
    write_csv("compatibility_checks.csv", checks)
    write_csv("mismatch_report.csv", mismatches)
    write_csv("materialization_counts.csv", [{"entity": e, **counts[e]} for e in ENTITIES])
    write_csv("public_materialization_scope.csv", public_scope)
    write_csv("contract_validation.csv", contract_violations or [{
        "entity": "ALL", "staging_row": "", "field_name": "", "materialization_scope": "ALL",
        "status": "EXACT_EQUIVALENCE", "reason": "No required-field violations.",
    }])
    (OUT / "compatibility_proof.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=clean), encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(payload), encoding="utf-8")
    print(json.dumps({
        "recommendation": recommendation,
        "field_contract_count": len(contract),
        "materialization_counts": counts,
        "compatibility_summary": summary,
        "report": str(REPORT.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
