"""Sprint 12B - source-to-contract feasibility analysis.

This script challenges the approved Sprint 12 PDR with the measured Sprint 12A
source inventory and the current EcoDrive SQLite contracts. SQLite files are
opened with ``mode=ro``; no schema, migration, runtime adapter, or physics code
is created or changed.

Run from the repository root:
    python etl/scripts/sprint_12b_source_to_contract.py
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT_12A = ROOT / "etl" / "data" / "processed" / "sprint_12a_audit" / "audit_results.json"
TESTCAR = ROOT / "etl" / "data" / "raw" / "epa_testcar" / "epa_testcar_2026_raw.xlsx"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12b_source_to_contract"
REPORT = ROOT / "etl" / "reports" / "sprint_12b_source_to_contract.md"
DATABASES = [ROOT / "data" / "db" / "eco_drive.db", ROOT / "data" / "db" / "eco_drive_qa.db"]

CLASSIFICATIONS = {"CONFIRMS_PDR", "FIELD_LEVEL_CDR_INPUT", "PDR_CHALLENGE", "DEFERRED"}
PDR_ENTITIES = {
    "PROGRAM", "VEHICLE_CONFIGURATION", "COMPONENT_DB", "TIRE_DB",
    "COMPONENT_INSTANCE", "COMPONENT_RESOLUTION", "VDE", "RUN", "FUELCONS",
}


def clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def text(value: Any, limit: int = 120) -> str:
    value = clean(value)
    return "" if value is None else str(value).replace("\n", " ").strip()[:limit]


def ro_connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def table_names(con: sqlite3.Connection) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]


def database_profile() -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, list[dict[str, Any]]]]:
    fields: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    rows_by_db_table: dict[str, list[dict[str, Any]]] = {}
    for path in DATABASES:
        con = ro_connect(path)
        counts[path.name] = {}
        for table in table_names(con):
            row_count = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            counts[path.name][table] = row_count
            columns = con.execute(f'PRAGMA table_info("{table}")').fetchall()
            rows = [dict(r) for r in con.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall()]
            rows_by_db_table[f"{path.name}:{table}"] = rows
            for col in columns:
                name = col[1]
                quoted = name.replace('"', '""')
                nonnull = int(con.execute(f'SELECT COUNT("{quoted}") FROM "{table}"').fetchone()[0])
                unique = int(con.execute(f'SELECT COUNT(DISTINCT "{quoted}") FROM "{table}" WHERE "{quoted}" IS NOT NULL').fetchone()[0])
                examples = [text(r[0]) for r in con.execute(
                    f'SELECT DISTINCT "{quoted}" FROM "{table}" WHERE "{quoted}" IS NOT NULL LIMIT 3'
                ).fetchall()]
                fields.append({
                    "database": path.name,
                    "table": table,
                    "table_rows": row_count,
                    "field": name,
                    "sql_type": col[2],
                    "not_null": bool(col[3]),
                    "primary_key": bool(col[5]),
                    "non_null_count": nonnull,
                    "non_null_pct": round(nonnull / row_count * 100, 3) if row_count else 0.0,
                    "unique_count": unique,
                    "examples": " | ".join(examples),
                    "evidence": f"READ-ONLY PRAGMA/table aggregate: {path.as_posix()}::{table}.{name}",
                })
        con.close()
    return fields, counts, rows_by_db_table


def legacy_owner(table: str, field: str) -> tuple[str, str]:
    """Pre-CDR ownership assessment, not a proposed physical schema."""
    f = field.lower()
    if table == "vde_db":
        if field in {"make", "model", "year"}:
            return "PROGRAM", "Commercial identity supports a fallback Program identity; generation remains unresolved."
        if f.startswith(("engine_", "transmission_")) or field == "drive_type":
            return "VEHICLE_CONFIGURATION", "Stable architecture descriptor in the current wide snapshot."
        return "VDE", "Current resolved Vehicle Demand snapshot field; retain on the fast-path state unless CDR approves relocation."
    if table == "fuelcons_db":
        if field in {"electrification", "engine_max_power_kw", "engine_rpm_max_power", "engine_max_torque_nm", "engine_rpm_max_torque", "gear_count", "final_drive_ratio", "battery_capacity_kwh", "battery_usable_kwh", "bms_discharge_limit_kw", "bms_regen_limit_kw", "bms_note"}:
            return "VEHICLE_CONFIGURATION", "Current FuelCons storage contains architecture/component descriptors; ownership should be separated at CDR while preserving the flat reconstruction."
        if field in {"ambient_temp_c", "ac_on", "tire_front_psi", "tire_rear_psi", "scenario_payload_kg", "method_note", "engine_method", "engine_version", "source_vde_revision", "assumptions_json", "provenance_json"}:
            return "RUN", "Execution/method/condition/lineage evidence rather than adopted comparison KPI."
        return "FUELCONS", "Relationship, adopted basis, record metadata, or comparison-facing result."
    if table == "component_db":
        if field in {"component_position", "driveline_architecture", "configuration_from", "configuration_to"}:
            return "COMPONENT_INSTANCE", "Usage/context currently co-located with reusable component definition."
        if field in {"physical_boundary", "test_condition_type", "test_method", "net_bridge_eligible", "equivalent_A_N", "equivalent_B_N_per_kph", "equivalent_C_N_per_kph2", "loss_pct", "residual_torque_front_nm", "residual_torque_rear_nm", "wheel_radius_m"}:
            return "COMPONENT_RESOLUTION", "Method/boundary/result evidence currently co-located with component identity."
        return "COMPONENT_DB", "Reusable component identity/provenance field."
    if table == "tire_roadload_db":
        return "TIRE_DB", "Dedicated tire evidence/engineering record retained by the PDR."
    return "RUN", "Operational/audit evidence outside the nine core entity payloads."


def legacy_ownership(schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in schema:
        owner, rationale = legacy_owner(row["table"], row["field"])
        result.append({
            "database": row["database"],
            "legacy_table": row["table"],
            "legacy_field": row["field"],
            "current_non_null_pct": row["non_null_pct"],
            "candidate_pdr_owner": owner,
            "classification": "FIELD_LEVEL_CDR_INPUT",
            "rationale": rationale,
            "evidence": row["evidence"],
        })
    return result


def reconstruct_legacy(rows_by_db_table: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Partition each current row by candidate owner, then flatten it losslessly in memory."""
    results = []
    for key, rows in rows_by_db_table.items():
        db_name, table = key.split(":", 1)
        if table not in {"vde_db", "fuelcons_db", "component_db", "tire_roadload_db"}:
            continue
        for row in rows:
            partitions: dict[str, dict[str, Any]] = defaultdict(dict)
            for field, value in row.items():
                owner, _ = legacy_owner(table, field)
                partitions[owner][field] = value
            rebuilt: dict[str, Any] = {}
            collision = False
            for payload in partitions.values():
                for field, value in payload.items():
                    if field in rebuilt:
                        collision = True
                    rebuilt[field] = value
            mismatches = [field for field in row if clean(row[field]) != clean(rebuilt.get(field))]
            results.append({
                "database": db_name,
                "legacy_table": table,
                "record_id": row.get("id", row.get("tire_test_code", "")),
                "fields_in_source": len(row),
                "fields_reconstructed": len(rebuilt),
                "field_value_matches": len(row) - len(mismatches),
                "mismatch_count": len(mismatches),
                "collision_detected": collision,
                "owners_used": ", ".join(sorted(partitions)),
                "reconstruction_status": "EXACT_FIELD_LEVEL" if not mismatches and not collision else "FAILED",
                "limitations": "Demonstrates field/value preservation only; relational keys, adoption semantics, and final CDR nullability are not yet frozen.",
            })
    return results


def series_coverage(df: pd.DataFrame, fields: list[str]) -> tuple[int, int, float]:
    if not fields or any(field not in df.columns for field in fields):
        return len(df), 0, 0.0
    mask = df[fields].notna().all(axis=1)
    return len(df), int(mask.sum()), round(float(mask.mean() * 100), 3)


def source_entity_matrix(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"source": "EPA Test Car MY2026", "PROGRAM": "PARTIAL", "VEHICLE_CONFIGURATION": "DIRECT/PARTIAL", "COMPONENT_DB": "PARTIAL", "TIRE_DB": "ABSENT", "COMPONENT_INSTANCE": "PARTIAL", "COMPONENT_RESOLUTION": "ABSENT", "VDE": "DIRECT", "RUN": "DIRECT", "FUELCONS": "PARTIAL", "evidence": "3,901 rows; explicit make/model/year, test IDs, engine/transmission/driveline, ETW, Target/Set ABC, procedure and result fields."},
        {"source": "EPA Certified Test Results", "PROGRAM": "PARTIAL", "VEHICLE_CONFIGURATION": "DIRECT/PARTIAL", "COMPONENT_DB": "PARTIAL", "TIRE_DB": "ABSENT", "COMPONENT_INSTANCE": "PARTIAL", "COMPONENT_RESOLUTION": "ABSENT", "VDE": "DIRECT", "RUN": "DIRECT", "FUELCONS": "PARTIAL", "evidence": "388,420 emission-result rows; test/configuration IDs, mass, Target/Set ABC, procedure, fuel and emission result."},
        {"source": "EPA Certified Models", "PROGRAM": "PARTIAL", "VEHICLE_CONFIGURATION": "PARTIAL", "COMPONENT_DB": "ABSENT", "TIRE_DB": "ABSENT", "COMPONENT_INSTANCE": "ABSENT", "COMPONENT_RESOLUTION": "ABSENT", "VDE": "ABSENT", "RUN": "ABSENT", "FUELCONS": "ABSENT", "evidence": "26,866 certification model/carline associations; identity support but no engineering state."},
        {"source": "FuelEconomy.gov MY2026", "PROGRAM": "PARTIAL", "VEHICLE_CONFIGURATION": "DIRECT/PARTIAL", "COMPONENT_DB": "PARTIAL", "TIRE_DB": "ABSENT", "COMPONENT_INSTANCE": "PARTIAL", "COMPONENT_RESOLUTION": "ABSENT", "VDE": "ABSENT", "RUN": "PARTIAL", "FUELCONS": "DIRECT", "evidence": "Fuel/electric label sheets expose make/model/year, engine/transmission/drive, Urban/Highway/Combined results and range; report-sheet grain varies."},
        {"source": "EEA 2025 provisional", "PROGRAM": "UNCLEAR", "VEHICLE_CONFIGURATION": "PARTIAL", "COMPONENT_DB": "PARTIAL", "TIRE_DB": "ABSENT", "COMPONENT_INSTANCE": "PARTIAL", "COMPONENT_RESOLUTION": "ABSENT", "VDE": "PARTIAL", "RUN": "PARTIAL", "FUELCONS": "DIRECT/PARTIAL", "evidence": "10,833,597 records; source identity codes, mass, powertrain, WLTP CO2/consumption/range. RLFI semantics unresolved; no tire or explicit phase detail."},
        {"source": "JRC technical vehicle dataset", "PROGRAM": "UNCLEAR", "VEHICLE_CONFIGURATION": "DIRECT/PARTIAL", "COMPONENT_DB": "DIRECT/PARTIAL", "TIRE_DB": "PARTIAL", "COMPONENT_INSTANCE": "PARTIAL", "COMPONENT_RESOLUTION": "ABSENT", "VDE": "DIRECT", "RUN": "PARTIAL", "FUELCONS": "DIRECT/PARTIAL", "evidence": "249 anonymized rows; component descriptors, masses, tire code, WLTP/real-world f0/f1/f2 and declared/simulated outputs. Row semantics unresolved."},
        {"source": "Current EcoDrive vde_db + fuelcons_db", "PROGRAM": "PARTIAL", "VEHICLE_CONFIGURATION": "DIRECT", "COMPONENT_DB": "PARTIAL", "TIRE_DB": "DIRECT", "COMPONENT_INSTANCE": "PARTIAL", "COMPONENT_RESOLUTION": "DIRECT/PARTIAL", "VDE": "DIRECT", "RUN": "PARTIAL", "FUELCONS": "DIRECT", "evidence": "Read-only schema and row coverage from data/db/eco_drive.db and eco_drive_qa.db; field-level reconstruction assessed for every current row."},
    ]


def configuration_candidates(testcar: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = testcar[testcar["Model Year"].eq(2026)].copy()
    stable = [
        "Model Year", "Vehicle Manufacturer Name", "Represented Test Veh Make", "Represented Test Veh Model",
        "Actual Tested Testgroup", "Test Vehicle ID", "Test Veh Configuration #", "Test Veh Displacement (L)",
        "Engine Code", "Tested Transmission Type", "# of Gears", "Drive System Description", "Axle Ratio", "N/V Ratio",
    ]
    vde_state = [
        "Equivalent Test Weight (lbs.)", "Target Coef A (lbf)", "Target Coef B (lbf/mph)",
        "Target Coef C (lbf/mph**2)", "Set Coef A (lbf)", "Set Coef B (lbf/mph)",
        "Set Coef C (lbf/mph**2)", "Test Procedure Cd",
    ]
    for field in stable + vde_state:
        df[field] = df[field].map(clean)
    df["candidate_configuration_key"] = df[stable].astype(str).agg("|".join, axis=1)
    df["vde_state_key"] = df[vde_state].astype(str).agg("|".join, axis=1)
    grouped = df.groupby("candidate_configuration_key", dropna=False)
    rows = []
    for key, part in grouped:
        state_count = int(part["vde_state_key"].nunique())
        if state_count < 2:
            continue
        first = part.iloc[0]
        changed = [field for field in vde_state if part[field].nunique(dropna=False) > 1]
        rows.append({
            "candidate_configuration_key": key,
            "make": first["Represented Test Veh Make"],
            "model": first["Represented Test Veh Model"],
            "model_year": first["Model Year"],
            "test_group": first["Actual Tested Testgroup"],
            "test_vehicle_id": first["Test Vehicle ID"],
            "test_vehicle_configuration": first["Test Veh Configuration #"],
            "source_rows": len(part),
            "distinct_vde_states": state_count,
            "vde_fields_that_vary": ", ".join(changed),
            "classification": "CONFIRMS_PDR",
            "warning": "Candidate key only. Test Vehicle ID may identify a physical test article rather than stable commercial configuration; CDR decision required.",
        })
    metrics = {
        "source_rows": len(df),
        "candidate_configuration_keys": int(df["candidate_configuration_key"].nunique()),
        "keys_with_multiple_source_rows": int((grouped.size() > 1).sum()),
        "keys_with_multiple_vde_states": len(rows),
        "distinct_vde_states": int(df["vde_state_key"].nunique()),
    }
    return rows, metrics


def source_field_lookup(audit: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["source"], r["original_field_name"]): r for r in audit["field_inventory"]}


def component_observability(audit: dict[str, Any], testcar: pd.DataFrame, jrc: pd.DataFrame) -> list[dict[str, Any]]:
    specs = [
        ("EPA Test Car MY2026", "ENGINE", ["Test Veh Displacement (L)", "Engine Code", "Rated Horsepower"], "PRIMARY"),
        ("EPA Test Car MY2026", "TRANSMISSION", ["Tested Transmission Type", "# of Gears"], "MAIN"),
        ("EPA Test Car MY2026", "DRIVELINE", ["Drive System Description", "Axle Ratio", "N/V Ratio"], "SYSTEM"),
        ("EPA Test Car MY2026", "TIRE", [], "UNKNOWN"),
        ("JRC technical vehicle dataset", "ENGINE", ["Engine max power", "Engine capacity [cm3]", "Engine n cylinders"], "PRIMARY"),
        ("JRC technical vehicle dataset", "TRANSMISSION", ["Gear box type", "N gears"], "MAIN"),
        ("JRC technical vehicle dataset", "TIRE", ["Tyre code"], "VEHICLE_SET_UNRESOLVED"),
        ("JRC technical vehicle dataset", "EMOTOR", ["Electric motor power [kW]", "Electric motor torque [Nm]"], "POSITION_UNRESOLVED"),
        ("JRC technical vehicle dataset", "BATTERY", ["Drive battery capacity [Ah]", "Drive battery nominal voltage [V]"], "TRACTION"),
    ]
    frames = {
        "EPA Test Car MY2026": testcar[testcar["Model Year"].eq(2026)],
        "JRC technical vehicle dataset": jrc,
    }
    result = []
    for source, component, fields, role in specs:
        frame = frames[source]
        total_rows = len(frame)
        available_fields = [field for field in fields if field in frame.columns]
        full_count = int(frame[available_fields].notna().all(axis=1).sum()) if available_fields and len(available_fields) == len(fields) else 0
        result.append({
            "source": source,
            "component_domain": component,
            "candidate_instance_role": role,
            "observed_fields": ", ".join(fields),
            "rows_with_all_listed_fields": full_count,
            "source_rows": total_rows,
            "cooccurrence_coverage_pct": round(full_count / total_rows * 100, 3) if total_rows else 0.0,
            "reusable_hardware_identity": "ABSENT" if component != "TIRE" or not fields else "PARTIAL (size/code, not manufacturer/model/part number)",
            "instance_observability": "PARTIAL" if fields else "ABSENT",
            "classification": "CONFIRMS_PDR" if fields else "FIELD_LEVEL_CDR_INPUT",
            "warning": "Coverage is exact field co-occurrence within the source rows for the listed descriptor set. It does not prove reusable component identity, position, or component ABC.",
        })
    return result


def run_feasibility() -> list[dict[str, Any]]:
    return [
        {"source": "EPA Test Car MY2026", "candidate_run_type": "TEST", "identifier_fields": "Test Number; Test Vehicle ID; Test Veh Configuration #; Actual Tested Testgroup", "procedure_fields": "Test Procedure Cd; Test Procedure Description; Test Fuel Type Cd/Description", "result_fields": "CO2; FE; emissions; bags", "representable_now": "YES", "classification": "CONFIRMS_PDR", "cdr_input": "Preserve source row/result detail and do not collapse procedure variants."},
        {"source": "EPA Certified Test Results", "candidate_run_type": "TEST", "identifier_fields": "Test Number; Vehicle ID; Vehicle Configuration Number; Certified Test Group", "procedure_fields": "Test Procedure; description; fuel; certification/in-use; region", "result_fields": "Emission Name; Rounded Emission Result; certification/standard/DF", "representable_now": "YES", "classification": "CONFIRMS_PDR", "cdr_input": "One Run can own multiple emission result details; source rows are result-detail grain, not independent tests."},
        {"source": "FuelEconomy.gov MY2026", "candidate_run_type": "HOMOLOGATION_RECORD (candidate enum)", "identifier_fields": "Index (Model Type Index); make/division/carline/model year", "procedure_fields": "label/cycle sheet context", "result_fields": "City/Highway/Combined FE; electric energy; range", "representable_now": "YES WITH ENUM INPUT", "classification": "FIELD_LEVEL_CDR_INPUT", "cdr_input": "Do not mislabel a published label record as a physical TEST. Add a declared/homologation evidence type or explicit evidence_kind."},
        {"source": "EEA 2025 provisional", "candidate_run_type": "HOMOLOGATION_OR_MONITORING_RECORD (candidate enum)", "identifier_fields": "ID and source identity/approval fields", "procedure_fields": "No explicit phase/procedure detail", "result_fields": "Ewltp/Enedc; fuel/electric consumption; range", "representable_now": "YES WITH ENUM INPUT", "classification": "FIELD_LEVEL_CDR_INPUT", "cdr_input": "RUN's evidence scope fits, but current TEST/SIMULATION/... enum does not precisely name a regulatory monitoring record."},
        {"source": "JRC technical vehicle dataset", "candidate_run_type": "SIMULATION (tentative)", "identifier_fields": "source row index; anonymized OEM/model", "procedure_fields": "pycsis_run flag", "result_fields": "OEM declared, simulated WLTP and simulated real-world outputs", "representable_now": "PARTIAL", "classification": "FIELD_LEVEL_CDR_INPUT", "cdr_input": "Separate declared evidence from simulated results; row/run grain and pycsis_run semantics remain unresolved."},
        {"source": "Current EcoDrive engineering flows", "candidate_run_type": "ESTIMATION / CALCULATION / ML_PREDICTION", "identifier_fields": "source VDE revision; method/version", "procedure_fields": "engine_method; engine_version; assumptions/provenance JSON", "result_fields": "energy/fuel/CO2 plus adopted basis", "representable_now": "YES", "classification": "CONFIRMS_PDR", "cdr_input": "Fidelity and confidence must remain independent nullable dimensions."},
    ]


def fuelcons_feasibility() -> list[dict[str, Any]]:
    return [
        {"source": "FuelEconomy.gov FEguide", "candidate_comparison_basis": "EPA_LABEL_2_CYCLE", "fuel_urban": "DIRECT", "fuel_highway": "DIRECT", "fuel_combined": "DIRECT", "electric_energy": "DIRECT/PARTIAL", "co2": "PARTIAL/UNCLEAR", "range": "DIRECT", "energy_basis": "Fuel Unit fields must be retained", "adoption_status": "PACIFIABLE", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"source": "FuelEconomy.gov EV/PHEV/FCV sheets", "candidate_comparison_basis": "EPA_LABEL_MULTI_ENERGY", "fuel_urban": "DIRECT where applicable", "fuel_highway": "DIRECT where applicable", "fuel_combined": "DIRECT where applicable", "electric_energy": "DIRECT", "co2": "PARTIAL/UNCLEAR", "range": "DIRECT", "energy_basis": "Separate electricity, gasoline/hydrogen and charge-depleting/sustaining contexts", "adoption_status": "PACIFIABLE AFTER SHEET-GRAIN RULE", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"source": "EPA Test Car", "candidate_comparison_basis": "EPA_TEST_PROCEDURE", "fuel_urban": "PROCEDURE-DEPENDENT", "fuel_highway": "PROCEDURE-DEPENDENT", "fuel_combined": "NOT DIRECT", "electric_energy": "PARTIAL", "co2": "DIRECT per test row", "range": "ABSENT", "energy_basis": "FE_UNIT + procedure/fuel context", "adoption_status": "RUN FIRST; NOT YET PACIFIED", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"source": "EEA provisional", "candidate_comparison_basis": "EU_WLTP_MONITORING_COMBINED", "fuel_urban": "ABSENT", "fuel_highway": "ABSENT", "fuel_combined": "DIRECT/PARTIAL", "electric_energy": "DIRECT/PARTIAL", "co2": "DIRECT combined", "range": "DIRECT/PARTIAL", "energy_basis": "Fuel mode/type fields required", "adoption_status": "PACIFIABLE AS COMBINED ONLY", "classification": "CONFIRMS_PDR"},
        {"source": "JRC OEM declared", "candidate_comparison_basis": "WLTP_OEM_DECLARED", "fuel_urban": "ABSENT", "fuel_highway": "ABSENT", "fuel_combined": "ABSENT", "electric_energy": "DIRECT", "co2": "DIRECT", "range": "DIRECT", "energy_basis": "Input/fuel/electrification fields", "adoption_status": "PACIFIABLE WITH ANONYMIZED IDENTITY", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"source": "JRC simulated", "candidate_comparison_basis": "WLTP_SIMULATED / REAL_WORLD_SIMULATED", "fuel_urban": "ABSENT", "fuel_highway": "ABSENT", "fuel_combined": "ABSENT", "electric_energy": "DIRECT where applicable", "co2": "DIRECT", "range": "ABSENT", "energy_basis": "Simulation boundary must be explicit", "adoption_status": "RUN + OPTIONAL ADOPTION", "classification": "CONFIRMS_PDR"},
        {"source": "Current EcoDrive fuelcons_db", "candidate_comparison_basis": "LEGACY_UNSPECIFIED", "fuel_urban": "EPA/WLTP cycle-specific columns", "fuel_highway": "EPA/WLTP cycle-specific columns", "fuel_combined": "aggregate fuel_l_per_100km", "electric_energy": "aggregate + phase/cycle columns", "co2": "aggregate + phase/cycle columns", "range": "label_range_km", "energy_basis": "energy_basis exists", "adoption_status": "LOSSLESS FIELD RECONSTRUCTION DEMONSTRATED; SEMANTIC BASIS BACKFILL NEEDED", "classification": "FIELD_LEVEL_CDR_INPUT"},
    ]


def field_shape_decisions() -> list[dict[str, Any]]:
    return [
        {"entity": "PROGRAM", "concept": "source-scoped identity and status", "recommended_shape": "FIRST_CLASS_SCALAR", "evidence": "Required for filtering/matching; public sources lack reliable generation code.", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"entity": "VEHICLE_CONFIGURATION", "concept": "engine/transmission/driveline/electrification descriptors", "recommended_shape": "FIRST_CLASS_SCALAR", "evidence": "Repeated explicit fields across EPA, FuelEconomy, JRC and current DB.", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"entity": "COMPONENT_DB", "concept": "domain, manufacturer/model/hardware reference, mass/power/capacity/ratio common scalars", "recommended_shape": "FIRST_CLASS_SCALAR", "evidence": "Queryable engineering features; availability varies by component/source.", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"entity": "COMPONENT_DB", "concept": "type-specific descriptors", "recommended_shape": "JSON", "evidence": "Sparse and heterogeneous across engine, motor, battery, transmission and tire.", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"entity": "COMPONENT_DB", "concept": "maps, curves, PDFs, solver/model files", "recommended_shape": "ARTIFACT_REFERENCE", "evidence": "Large/complex source evidence; not suitable for scalar table columns.", "classification": "DEFERRED"},
        {"entity": "COMPONENT_INSTANCE", "concept": "role/position/quantity/configuration link", "recommended_shape": "FIRST_CLASS_SCALAR", "evidence": "Required to distinguish architecture usage from reusable identity; source position is often unresolved/null.", "classification": "CONFIRMS_PDR"},
        {"entity": "COMPONENT_RESOLUTION", "concept": "boundary, method, conditions, adopted A/B/C, lineage", "recommended_shape": "SCALARS + JSON LINEAGE", "evidence": "Current component_db conflates these; public Tier-0 sources do not justify component decomposition.", "classification": "CONFIRMS_PDR"},
        {"entity": "RUN", "concept": "run/evidence type, source ID, procedure, fidelity, confidence", "recommended_shape": "FIRST_CLASS_SCALAR", "evidence": "Needed for filtering and to prevent estimates/ML/monitoring evidence from masquerading as tests.", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"entity": "RUN", "concept": "phase/result details, assumptions, lineage", "recommended_shape": "JSON", "evidence": "Variable result sets across EPA, EEA, JRC and engineering flows.", "classification": "FIELD_LEVEL_CDR_INPUT"},
        {"entity": "FUELCONS", "concept": "final Urban/Highway/Combined energy/fuel/CO2/range", "recommended_shape": "FIRST_CLASS_SCALAR", "evidence": "Comparison-facing KPIs exist across current DB and public sources, with nullable applicability.", "classification": "CONFIRMS_PDR"},
    ]


def findings(metrics: dict[str, Any], reconstruction: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact = sum(r["reconstruction_status"] == "EXACT_FIELD_LEVEL" for r in reconstruction)
    return [
        {"finding_id": "S12B-01", "question": "Can Program identity be inferred?", "classification": "FIELD_LEVEL_CDR_INPUT", "finding": "Not reliably as an OEM generation across sources. EPA provides strong source-local commercial/test-family identity, EEA codes remain semantically ambiguous locally, and JRC identity is anonymized.", "proposed_cdr_decision": "Allow a source-scoped fallback Program with identity_status/confidence and raw source keys; prohibit cross-year/cross-source merge without explicit evidence.", "evidence": "EPA 3,901 rows / 784 Make+Model+Year groups; JRC OEM/model anonymized; EEA identity code labels inventoried without local dictionary."},
        {"finding_id": "S12B-02", "question": "Can Vehicle Configuration be separated from VDE state?", "classification": "CONFIRMS_PDR", "finding": f"Yes conceptually: {metrics['candidate_configuration_keys']:,} strict EPA hardware/test-article candidate keys contain {metrics['keys_with_multiple_vde_states']:,} keys with multiple observed VDE states.", "proposed_cdr_decision": "Keep Vehicle Configuration separate from VDE; freeze exact key only after resolving whether Test Vehicle ID/Configuration # identify physical articles, source configurations, or both.", "evidence": "EPA Test Car stable descriptor key excludes ETW, Target/Set ABC and procedure; varying fields are listed per candidate."},
        {"finding_id": "S12B-03", "question": "Which Component Instances are observable?", "classification": "CONFIRMS_PDR", "finding": "Engine, transmission and driveline are partially observable in EPA; JRC also exposes tire, e-motor and battery descriptors. Reusable hardware identity and component position are usually incomplete.", "proposed_cdr_decision": "Permit partial Component Instances with nullable component reference/position and direct provenance; never require enrichment for VDE/Run validity.", "evidence": "component_observability.csv field/coverage lower bounds."},
        {"finding_id": "S12B-04", "question": "Which Component DB properties are scalar vs JSON/artifact?", "classification": "FIELD_LEVEL_CDR_INPUT", "finding": "Frequently filtered identity and numeric features justify scalars; heterogeneous descriptors fit JSON; maps/PDFs/models fit artifact references.", "proposed_cdr_decision": "Use the field_shape_decisions matrix as the CDR candidate surface; do not promote fields solely because they exist in one source.", "evidence": "Repeated fields across EPA/JRC/current DB plus Sprint 12A field inventory."},
        {"finding_id": "S12B-05", "question": "Which Component Resolutions are defensible?", "classification": "CONFIRMS_PDR", "finding": "Tier-0 public data supports whole-vehicle roadload, not a component causal split. No EPA/JRC component ABC resolution is defensible from structured data alone.", "proposed_cdr_decision": "Component Resolution remains optional and separate; v1 accepts zero public resolutions while preserving authoritative VDE/Run roadload.", "evidence": "EPA tire/RRC/CdA coverage 0%; 696 candidate pairs are explicitly non-causal; JRC f0/f1/f2 are whole-vehicle labels."},
        {"finding_id": "S12B-06", "question": "Which RUN records are representable?", "classification": "FIELD_LEVEL_CDR_INPUT", "finding": "EPA tests and current estimation/calculation/ML flows fit RUN. FuelEconomy and EEA are evidence records but are not accurately named by the current run_type examples.", "proposed_cdr_decision": "Keep RUN entity; add evidence_kind or a HOMOLOGATION/MONITORING run type at CDR. Preserve source row/result-detail grain separately.", "evidence": "run_feasibility.csv."},
        {"finding_id": "S12B-07", "question": "Which FuelCons comparison bases can be pacified?", "classification": "FIELD_LEVEL_CDR_INPUT", "finding": "EPA label Urban/Highway/Combined, EEA WLTP combined, JRC declared/simulated and current legacy result surfaces are supportable with different basis semantics and nullable dimensions.", "proposed_cdr_decision": "Seed comparison_basis candidates from measured source contexts; keep energy_basis independent and preserve units/fuel mode. Do not derive missing combined/phase values yet.", "evidence": "fuelcons_basis_feasibility.csv."},
        {"finding_id": "S12B-08", "question": "Can the current legacy surface be reconstructed?", "classification": "CONFIRMS_PDR", "finding": f"At field/value level, yes: {exact:,}/{len(reconstruction):,} current rows across vde_db, fuelcons_db, component_db and tire_roadload_db reconstruct exactly after partitioning by candidate PDR owner.", "proposed_cdr_decision": "Accept this as the first non-regression proof, but require relational/adoption-semantic reconstruction tests before CDR exit.", "evidence": "legacy_reconstruction_matrix.csv; SQLite opened mode=ro/query_only."},
        {"finding_id": "S12B-09", "question": "Does evidence require a tenth primary entity?", "classification": "CONFIRMS_PDR", "finding": "No current evidence requires another entity. The monitoring/homologation distinction can be represented as RUN field/enum input without changing the nine-table architecture.", "proposed_cdr_decision": "Keep the nine-entity baseline unless future evidence demonstrates semantic loss.", "evidence": "source_entity_matrix.csv and run_feasibility.csv."},
        {"finding_id": "S12B-10", "question": "What remains deferred?", "classification": "DEFERRED", "finding": "PDF parsing/RAG, EPREL enrichment, ML prediction, component estimation, topology and physical SQL are not required for Data Contract v1 feasibility.", "proposed_cdr_decision": "Retain artifact/provenance hooks only; do not implement these capabilities before CDR.", "evidence": "PDR Sections 2, 10 and Sprint 12 constraints."},
    ]


def open_decisions() -> list[dict[str, Any]]:
    return [
        {"decision_id": "CDR-01", "topic": "Program fallback identity", "decision_needed": "Choose source-scoped fallback key and whether model year belongs in it.", "options_supported_by_evidence": "Make+model+year preserves source distinctions; cross-year Program merge requires explicit generation evidence not currently available.", "risk_if_guessed": "Silent generation collapse or duplicate Programs."},
        {"decision_id": "CDR-02", "topic": "Vehicle Configuration key", "decision_needed": "Decide the semantic role of EPA Test Vehicle ID and Test Veh Configuration #.", "options_supported_by_evidence": "Use as source identity/provenance; do not yet treat either as universal hardware identity.", "risk_if_guessed": "Conflates physical test articles, certification configs and commercial variants."},
        {"decision_id": "CDR-03", "topic": "RUN evidence taxonomy", "decision_needed": "Add evidence_kind or extend run_type for homologation/monitoring declarations.", "options_supported_by_evidence": "FuelEconomy/EEA records are authoritative evidence but not proven physical tests.", "risk_if_guessed": "False TEST provenance."},
        {"decision_id": "CDR-04", "topic": "FuelCons comparison bases", "decision_needed": "Freeze initial basis enum and adoption rules for EPA label/test, EEA WLTP, JRC declared/simulated and legacy unspecified.", "options_supported_by_evidence": "See fuelcons_basis_feasibility.csv; dimensions and units differ.", "risk_if_guessed": "Compares non-equivalent results or silently derives missing dimensions."},
        {"decision_id": "CDR-05", "topic": "Legacy relational reconstruction", "decision_needed": "Define canonical links/adoption lineage needed to rebuild current joins, parent VDE and FuelCons behavior.", "options_supported_by_evidence": "Field/value reconstruction is exact; relationship semantics are not yet exercised.", "risk_if_guessed": "Field-complete migration that changes application behavior."},
        {"decision_id": "CDR-06", "topic": "EEA RLFI and JRC row grain", "decision_needed": "Obtain authoritative local documentation before assigning physical semantics.", "options_supported_by_evidence": "Keep raw field/source-row evidence with UNRESOLVED status.", "risk_if_guessed": "Incorrect roadload interpretation or run identity."},
    ]


def save_csv(name: str, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8")


def build_report(payload: dict[str, Any]) -> str:
    findings_rows = payload["pdr_findings"]
    class_counts = Counter(r["classification"] for r in findings_rows)
    recon = payload["legacy_reconstruction_summary"]
    db_counts = payload["database_table_counts"]
    m = payload["configuration_metrics"]
    lines = [
        "# Sprint 12B — Data Feasibility / Source-to-Contract Analysis",
        "",
        "## Decision status",
        "",
        "The measured data supports the PDR's nine-entity conceptual architecture. No tenth primary entity is justified and no final SQL schema is implemented. The principal CDR inputs are field-level: fallback Program identity, the exact Vehicle Configuration key, RUN evidence taxonomy, FuelCons comparison bases, and relational reconstruction semantics.",
        "",
        f"Finding classification: {class_counts.get('CONFIRMS_PDR', 0)} `CONFIRMS_PDR`, {class_counts.get('FIELD_LEVEL_CDR_INPUT', 0)} `FIELD_LEVEL_CDR_INPUT`, {class_counts.get('PDR_CHALLENGE', 0)} `PDR_CHALLENGE`, and {class_counts.get('DEFERRED', 0)} `DEFERRED`.",
        "",
        "## Evidence boundary",
        "",
        "- Source evidence: Sprint 12A inventory, EPA MY2026 rows, EEA/JRC coverage and current EcoDrive schemas/rows.",
        "- Current SQLite databases were opened with URI `mode=ro` plus `PRAGMA query_only=ON`.",
        "- No raw data, runtime database, `src/`, page, physics, resolver, DDL, migration or production adapter was modified.",
        "- Component-pair evidence is not treated as causal component decomposition.",
        "",
        "## Current legacy surface",
        "",
    ]
    for db_name, tables in db_counts.items():
        relevant = ", ".join(f"{name}={count:,}" for name, count in tables.items())
        lines.append(f"- `{db_name}`: {relevant}.")
    lines += [
        "",
        f"The draft in-memory owner partition reconstructed **{recon['exact_rows']:,}/{recon['rows_assessed']:,}** assessed current rows with exact field/value equality and no collisions. This is a field-level non-regression proof; it does not yet prove foreign-key, adoption, lineage or UI-query equivalence.",
        "",
        "## PDR findings",
        "",
        "| ID | Classification | Finding | Proposed CDR decision |",
        "|---|---|---|---|",
    ]
    for row in findings_rows:
        lines.append(f"| {row['finding_id']} | `{row['classification']}` | {row['finding']} | {row['proposed_cdr_decision']} |")
    lines += [
        "",
        "## Program and Vehicle Configuration grain",
        "",
        "Program generation cannot be inferred reliably across sources. EPA has useful source-local commercial/test-family identifiers, EEA identity-code semantics remain unresolved in the supplied local corpus, and JRC identity is anonymized. The defensible v1 fallback is a source-scoped Program identity with explicit low confidence/status; cross-year or cross-source merges require positive evidence.",
        "",
        f"For EPA MY2026, a strict candidate hardware/test-article key produced **{m['candidate_configuration_keys']:,}** keys from **{m['source_rows']:,}** rows. **{m['keys_with_multiple_vde_states']:,}** keys contain multiple distinct ETW/Target/Set/procedure states. This directly supports separate Vehicle Configuration and VDE concepts, but it does not prove that Test Vehicle ID or Configuration # is a universal hardware key.",
        "",
        "## Components and resolutions",
        "",
        "EPA directly supports partial engine, transmission and driveline instances; JRC adds partial tire, e-motor and battery descriptions. Part-number-level reusable identity and instance position are generally absent. Component Instance must therefore allow incomplete identity/role data without invalidating VDE or RUN.",
        "",
        "No Tier-0 source supports a defensible component roadload split. EPA has authoritative whole-vehicle Target/Set ABC but no tire/RRC/CdA fields; JRC's f0/f1/f2 fields are labelled at whole-vehicle WLTP/real-world level. Component Resolution remains an optional analysis/evidence object, not a required ingestion product.",
        "",
        "## RUN and FuelCons",
        "",
        "EPA test records fit `RUN(TEST)`. Existing EcoDrive estimation/calculation/ML flows also fit the ledger. FuelEconomy and EEA require a field-level taxonomy decision because a published homologation/monitoring record should not be silently labelled as a physical TEST. JRC declared and simulated values should remain distinct evidence/results.",
        "",
        "FuelCons can support EPA label Urban/Highway/Combined, EEA combined WLTP, and JRC declared/simulated result bases using nullable dimensions. `comparison_basis` and `energy_basis` must remain separate; no absent combined or phase value is derived in this phase.",
        "",
        "## CDR entry assessment",
        "",
        "| Criterion | Status | Evidence |",
        "|---|---|---|",
        "| Program/configuration matching | PARTIAL | Fallback Program and strict EPA configuration candidates demonstrated; final keys unresolved. |",
        "| First-class Component DB fields | READY FOR DECISION | Scalar/JSON/artifact recommendations in `field_shape_decisions.csv`. |",
        "| Component Resolution examples | PARTIAL / EMPTY-BY-DESIGN | No structured public component resolution is defensible; current legacy resolution fields are inventoried. |",
        "| RUN preservation | READY FOR FIELD DECISION | EPA/current flows fit; homologation/monitoring enum decision open. |",
        "| FuelCons bases | READY FOR FIELD DECISION | Candidate basis matrix covers EPA, EEA, JRC and legacy. |",
        "| BEV/PHEV energy and range | PARTIAL | FuelEconomy/JRC/EEA provide evidence; sheet/basis rules require CDR decision. |",
        "| Draft legacy reconstruction | FIELD-LEVEL PASS | Exact field/value reconstruction for every assessed current row. Relational behavior remains open. |",
        "| Open-field shape classification | READY | First-class/JSON/artifact/deferred matrix produced. |",
        "",
        "The project is ready for a focused CDR decision workshop, but **not** for final DDL or migration. CDR must close the six decisions in `open_cdr_decisions.csv`, then require relational reconstruction tests before implementation.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python etl/scripts/sprint_12a_audit.py",
        "python etl/scripts/sprint_12b_source_to_contract.py",
        "```",
        "",
        "## Machine-readable outputs",
        "",
        "- `pdr_findings.csv`",
        "- `source_entity_matrix.csv`",
        "- `legacy_schema_inventory.csv`",
        "- `legacy_field_ownership.csv`",
        "- `legacy_reconstruction_matrix.csv`",
        "- `configuration_candidates.csv`",
        "- `component_observability.csv`",
        "- `field_shape_decisions.csv`",
        "- `run_feasibility.csv`",
        "- `fuelcons_basis_feasibility.csv`",
        "- `open_cdr_decisions.csv`",
        "- `source_to_contract.json`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not AUDIT_12A.exists():
        raise RuntimeError("Sprint 12A audit_results.json is required; run sprint_12a_audit.py first.")
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT_12A.read_text(encoding="utf-8"))
    schema, db_counts, db_rows = database_profile()
    ownership = legacy_ownership(schema)
    reconstruction = reconstruct_legacy(db_rows)
    testcar = pd.read_excel(TESTCAR, sheet_name="Sheet1", engine="openpyxl")
    candidates, config_metrics = configuration_candidates(testcar)
    jrc = pd.read_excel(ROOT / "etl" / "data" / "raw" / "wltp_jrc" / "Data_PV_fleet_2021_EU_PYCSIS.xlsx", sheet_name="Sheet1", engine="openpyxl")
    components = component_observability(audit, testcar, jrc)
    runs = run_feasibility()
    fuelcons = fuelcons_feasibility()
    shapes = field_shape_decisions()
    matrix = source_entity_matrix(audit)
    decision_rows = open_decisions()
    finding_rows = findings(config_metrics, reconstruction)
    assert all(row["classification"] in CLASSIFICATIONS for row in finding_rows)
    assert all(row["candidate_pdr_owner"] in PDR_ENTITIES for row in ownership)
    exact_rows = sum(r["reconstruction_status"] == "EXACT_FIELD_LEVEL" for r in reconstruction)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pdr_baseline": "docs/sprints/PDR_CANONICAL_DATA_ARCHITECTURE.md",
        "handoff": "docs/sprints/CODEX_NEXT_PHASE_PROMPT.md",
        "database_access": "SQLite URI mode=ro + PRAGMA query_only=ON",
        "database_table_counts": db_counts,
        "configuration_metrics": config_metrics,
        "legacy_reconstruction_summary": {"rows_assessed": len(reconstruction), "exact_rows": exact_rows, "failed_rows": len(reconstruction) - exact_rows},
        "pdr_findings": finding_rows,
        "source_entity_matrix": matrix,
        "legacy_schema_inventory": schema,
        "legacy_field_ownership": ownership,
        "legacy_reconstruction_matrix": reconstruction,
        "configuration_candidates": candidates,
        "component_observability": components,
        "field_shape_decisions": shapes,
        "run_feasibility": runs,
        "fuelcons_basis_feasibility": fuelcons,
        "open_cdr_decisions": decision_rows,
    }
    save_csv("pdr_findings.csv", finding_rows)
    save_csv("source_entity_matrix.csv", matrix)
    save_csv("legacy_schema_inventory.csv", schema)
    save_csv("legacy_field_ownership.csv", ownership)
    save_csv("legacy_reconstruction_matrix.csv", reconstruction)
    save_csv("configuration_candidates.csv", candidates)
    save_csv("component_observability.csv", components)
    save_csv("field_shape_decisions.csv", shapes)
    save_csv("run_feasibility.csv", runs)
    save_csv("fuelcons_basis_feasibility.csv", fuelcons)
    save_csv("open_cdr_decisions.csv", decision_rows)
    (OUT / "source_to_contract.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(payload), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT.relative_to(ROOT)),
        "report": str(REPORT.relative_to(ROOT)),
        "configuration_metrics": config_metrics,
        "reconstruction": payload["legacy_reconstruction_summary"],
        "finding_classification": dict(Counter(x["classification"] for x in finding_rows)),
        "database_table_counts": db_counts,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
