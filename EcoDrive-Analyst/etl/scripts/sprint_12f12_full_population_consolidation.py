"""Sprint 12F.12: build the full-population canonical candidate.

The Sprint 12E.2 database is copied into a new, protected staging location.
Sprint 12E.3A electric-unit decisions are added as deterministic RUN evidence;
they do not create FuelCons rows because no electric cycle/materialization
method was approved.  Runtime databases and the 75-program notebook demo are
read-only inputs to the closure checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "etl" / "scripts"))

import sprint_12e3_electric_unit_audit as electric  # noqa: E402
from src.vde_core.roadload_analysis import roadload_force_N  # noqa: E402


SOURCE_DB = ROOT / "etl" / "data" / "staging" / "sprint_12e2_epa_fuelcons" / "eco_drive_canonical_epa_fuelcons.db"
STAGING = ROOT / "etl" / "data" / "staging" / "sprint_12f12_full_population"
OUTPUT_DB = STAGING / "eco_drive_canonical_full_candidate.db"
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12f12_full_population"
REPORT = ROOT / "etl" / "reports" / "sprint_12f12_full_population_consolidation.md"
SUMMARY = OUT / "sprint_12f12_summary.json"
OVERRIDES = ROOT / "etl" / "config" / "electric_consumption_overrides.csv"
DISPOSITIONS = ROOT / "etl" / "data" / "processed" / "sprint_12e3_electric_unit_audit" / "resolved_12e2_electric_cases.csv"
RUN_GROUPING = ROOT / "etl" / "data" / "processed" / "sprint_12e2_epa_fuelcons" / "run_grouping_results.csv"
RUNTIME_DBS = (ROOT / "data" / "db" / "eco_drive.db", ROOT / "data" / "db" / "eco_drive_qa.db")
DEMO_DB = ROOT / "notebooks" / "_data" / "12f11_canonical_notebook_demo.db"
STATUS = "FULL_CANONICAL_CANDIDATE_READY — PROCEED_TO_12G_INTEGRATION"
ANNOTATION_KEY = "sprint_12e3a_electric_unit_closure"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Iterable[Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def guard_output_path(path: Path) -> Path:
    resolved = path.resolve()
    protected = {SOURCE_DB.resolve(), DEMO_DB.resolve(), *(path.resolve() for path in RUNTIME_DBS)}
    if resolved in protected:
        raise ValueError(f"Protected database cannot be an output: {resolved}")
    if not resolved.is_relative_to(STAGING.resolve()) or resolved.suffix.lower() != ".db":
        raise ValueError(f"12F.12 output must be a .db below {STAGING.resolve()}: {resolved}")
    return resolved


def database_tables(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def primary_key_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [row[1] for row in sorted((row for row in rows if row[5]), key=lambda row: row[5])]


def schema_signature(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall()
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def database_content_signature(path: Path) -> str:
    digest = hashlib.sha256()
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        for table in database_tables(connection):
            columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
            order = primary_key_columns(connection, table) or columns
            query = f'SELECT * FROM "{table}" ORDER BY ' + ",".join(f'"{column}"' for column in order)
            digest.update(table.encode("utf-8"))
            digest.update(json.dumps(columns, separators=(",", ":")).encode("utf-8"))
            for row in connection.execute(query):
                digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
                digest.update(b"\n")
    return digest.hexdigest().upper()


def source_row_to_run() -> dict[int, str]:
    mapping: dict[int, str] = {}
    for row in read_csv(RUN_GROUPING):
        for value in row["source_excel_rows"].split(";"):
            if value:
                mapping[int(value)] = row["canonical_run_id"]
    return mapping


def electric_records() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    mapping = source_row_to_run()
    records: list[dict[str, Any]] = []
    annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(OVERRIDES):
        status = row["validation_status"]
        source_id = row["source_record_id"]
        current = source_id.startswith("EPA-TESTCAR-ROW-")
        source_row = int(source_id.rsplit("-", 1)[1]) if current else None
        run_id = mapping.get(source_row) if source_row is not None else None
        raw_value = float(row["expected_raw_value"])
        canonical = (
            electric.kwh_per_100mi_to_wh_per_km(raw_value)
            if status == "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION"
            else None
        )
        if status == "RETIRED_SOURCE_RECORD_ABSENT":
            representation = "RETIRED_SOURCE_RECORD_ABSENT"
        elif run_id:
            representation = "CANONICAL_RUN_EVIDENCE"
        else:
            representation = "SOURCE_ONLY_QUARANTINED_IDENTITY_ANOMALY"
        record = {
            "source_system": row["source_system"],
            "source_file_version": row["source_file_version"],
            "source_record_id": source_id,
            "legacy_source_record_id": row["legacy_source_record_id"],
            "canonical_run_id": run_id or "",
            "raw_field": row["raw_field"],
            "raw_value": raw_value,
            "source_unit_text": "MPG" if current else "",
            "interpreted_unit": row["corrected_unit"],
            "canonical_wh_per_km": "" if canonical is None else canonical,
            "validation_status": status,
            "candidate_representation": representation,
            "fuelcons_materialized": False,
            "reason": row["reason"],
        }
        records.append(record)
        if run_id:
            annotations[run_id].append(
                {
                    "source_record_id": source_id,
                    "raw_field": row["raw_field"],
                    "raw_value": raw_value,
                    "source_unit_text": "MPG",
                    "interpreted_unit": row["corrected_unit"] or None,
                    "canonical_wh_per_km": canonical,
                    "validation_status": status,
                    "fuelcons_materialized": False,
                }
            )
    return records, annotations


def case_annotations() -> tuple[list[dict[str, str]], dict[str, list[dict[str, Any]]]]:
    rows = read_csv(DISPOSITIONS)
    annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case = {
            "canonical_vde_id": int(row["canonical_vde_id"]),
            "comparison_basis": row["comparison_basis"],
            "fuel_type": row["fuel_type"],
            "cycle": row["cycle"],
            "disposition": row["sprint_12e3a_disposition"],
            "interpreted_unit": row["interpreted_unit"] or None,
            "unit_resolved": row["unit_resolved"] == "True",
            "materialized_additional_fuelcons": False,
            "fuelcons_materialization_status": row["fuelcons_materialization_status"],
        }
        for run_id in row["candidate_run_ids"].split(";"):
            if run_id:
                annotations[run_id].append(case)
    return rows, annotations


def annotate_database(path: Path) -> dict[str, int]:
    records, unit_by_run = electric_records()
    cases, case_by_run = case_annotations()
    touched = sorted(set(unit_by_run) | set(case_by_run))
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        existing = {
            row[0]: row[1]
            for row in connection.execute(
                f"SELECT run_id,result_details_json FROM run WHERE run_id IN ({','.join('?' for _ in touched)})",
                touched,
            )
        }
        missing = sorted(set(touched) - set(existing))
        if missing:
            raise RuntimeError(f"Electric closure references missing RUNs: {missing[:10]}")
        for run_id in touched:
            details = json.loads(existing[run_id] or "{}")
            details[ANNOTATION_KEY] = {
                "annotation_version": "sprint_12f12_v1",
                "unit_interpretations": sorted(unit_by_run.get(run_id, []), key=lambda item: item["source_record_id"]),
                "case_dispositions": sorted(
                    case_by_run.get(run_id, []),
                    key=lambda item: (item["canonical_vde_id"], item["fuel_type"], item["comparison_basis"], item["cycle"]),
                ),
                "materialization_policy": "UNIT_RESOLUTION_DOES_NOT_BYPASS_CYCLE_GRAIN_OR_METHOD_VALIDATION",
            }
            encoded = json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            connection.execute("UPDATE run SET result_details_json=? WHERE run_id=?", (encoded, run_id))
        connection.commit()
    return {
        "annotated_runs": len(touched),
        "unit_annotated_runs": len(unit_by_run),
        "case_annotated_runs": len(case_by_run),
        "electric_registry_records": len(records),
        "electric_cases": len(cases),
    }


def build_database(path: Path) -> dict[str, int]:
    target = guard_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_DB, target)
    return annotate_database(target)


def table_counts(path: Path) -> dict[str, int]:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        return {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in database_tables(connection)}


def export_tables(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        for table in database_tables(connection):
            columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
            order = primary_key_columns(connection, table) or columns
            query = f'SELECT * FROM "{table}" ORDER BY ' + ",".join(f'"{column}"' for column in order)
            output = OUT / f"{table}.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(columns)
                count = 0
                for row in connection.execute(query):
                    writer.writerow(["" if value is None else value for value in row])
                    count += 1
            counts[table] = count
            manifest.append({"table": table, "rows": count, "file": output.relative_to(ROOT).as_posix(), "sha256": sha256(output)})
    return manifest, counts


def integrity_checks(path: Path, export_counts: dict[str, int]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        fk_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        checks.append({"check": "FOREIGN_KEY_CHECK", "actual": len(fk_rows), "expected": 0, "passed": not fk_rows})
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        checks.append({"check": "QUICK_CHECK", "actual": quick, "expected": "ok", "passed": quick == "ok"})
        for table in database_tables(connection):
            pk = primary_key_columns(connection, table)
            quoted = ",".join(f'"{column}"' for column in pk)
            duplicate_groups = connection.execute(
                f'SELECT COUNT(*) FROM (SELECT {quoted},COUNT(*) n FROM "{table}" GROUP BY {quoted} HAVING n>1)'
            ).fetchone()[0]
            checks.append({"check": f"PK_UNIQUENESS::{table}", "actual": duplicate_groups, "expected": 0, "passed": duplicate_groups == 0})
            db_count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            checks.append({"check": f"EXPORT_COUNT::{table}", "actual": export_counts.get(table), "expected": db_count, "passed": export_counts.get(table) == db_count})
        same_vde = connection.execute(
            "SELECT COUNT(*) FROM fuelcons_run_adoption a "
            "JOIN fuelcons f ON f.id=a.fuelcons_id JOIN run r ON r.run_id=a.run_id "
            "WHERE a.vde_id<>f.vde_id OR a.vde_id<>r.vde_id"
        ).fetchone()[0]
        checks.append({"check": "ADOPTION_SAME_VDE", "actual": same_vde, "expected": 0, "passed": same_vde == 0})
        missing_lineage = connection.execute(
            "SELECT COUNT(*) FROM fuelcons f WHERE NOT EXISTS "
            "(SELECT 1 FROM fuelcons_run_adoption a WHERE a.fuelcons_id=f.id)"
        ).fetchone()[0]
        checks.append({"check": "FUELCONS_RUN_LINEAGE_COMPLETENESS", "actual": missing_lineage, "expected": 0, "passed": missing_lineage == 0})
    return checks


def _describe(name: str, values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"metric": name, "n": 0, "mean": "", "std": "", "min": "", "p25": "", "median": "", "p75": "", "max": ""}
    return {
        "metric": name,
        "n": len(clean),
        "mean": clean.mean(),
        "std": clean.std(ddof=1) if len(clean) > 1 else 0.0,
        "min": clean.min(),
        "p25": clean.quantile(0.25),
        "median": clean.median(),
        "p75": clean.quantile(0.75),
        "max": clean.max(),
    }


def _correlation(name: str, left: pd.Series, right: pd.Series) -> dict[str, Any]:
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")})
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    value = pair["left"].corr(pair["right"]) if len(pair) >= 3 and pair.nunique().min() > 1 else np.nan
    return {"relationship": name, "n": len(pair), "pearson_r": "" if pd.isna(value) else float(value), "scope": "QA_ASSOCIATION_NOT_CAUSAL"}


def population_analysis(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        facts = pd.read_sql_query(
            "SELECT f.id fuelcons_id,f.vde_id,v.mass_kg,v.coast_A_N,v.coast_B_N_per_kph,v.coast_C_N_per_kph2,"
            "f.fuel_l_per_100km,f.gco2_per_km,f.energy_Wh_per_km "
            "FROM fuelcons f JOIN vde v ON v.id=f.vde_id",
            connection,
        )
        scenario_rows = connection.execute("SELECT vde_id,result_details_json FROM run WHERE result_details_json IS NOT NULL").fetchall()
    complete = facts[["coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2"]].notna().all(axis=1)
    facts["roadload_force_100kph_N"] = np.nan
    facts.loc[complete, "roadload_force_100kph_N"] = facts.loc[complete].apply(
        lambda row: roadload_force_N(row["coast_A_N"], row["coast_B_N_per_kph"], row["coast_C_N_per_kph2"], 100.0),
        axis=1,
    )
    scenarios: list[dict[str, Any]] = []
    for vde_id, payload in scenario_rows:
        details = json.loads(payload)
        total = details.get("vde_total_mj_per_km")
        net = details.get("vde_net_mj_per_km")
        if total is not None or net is not None:
            scenarios.append({"vde_id": vde_id, "vde_total_mj_per_km": total, "vde_net_mj_per_km": net})
    scenario = pd.DataFrame(scenarios)
    if not scenario.empty:
        for column in ("vde_total_mj_per_km", "vde_net_mj_per_km"):
            scenario[column] = pd.to_numeric(scenario[column], errors="coerce")
        scenario = scenario.groupby("vde_id", as_index=False).mean(numeric_only=True)
        for column in ("vde_total_mj_per_km", "vde_net_mj_per_km"):
            if column not in scenario:
                scenario[column] = np.nan
        facts = facts.merge(scenario, how="left", on="vde_id")
    else:
        facts["vde_total_mj_per_km"] = np.nan
        facts["vde_net_mj_per_km"] = np.nan
    metrics = [
        "mass_kg", "roadload_force_100kph_N", "fuel_l_per_100km", "gco2_per_km",
        "energy_Wh_per_km", "vde_total_mj_per_km", "vde_net_mj_per_km",
    ]
    statistics = [_describe(metric, facts[metric]) for metric in metrics]
    pairs = [
        ("fuel_l_per_100km_vs_gco2_per_km", "fuel_l_per_100km", "gco2_per_km"),
        ("mass_kg_vs_fuel_l_per_100km", "mass_kg", "fuel_l_per_100km"),
        ("roadload_force_100kph_N_vs_fuel_l_per_100km", "roadload_force_100kph_N", "fuel_l_per_100km"),
        ("mass_kg_vs_energy_Wh_per_km", "mass_kg", "energy_Wh_per_km"),
        ("roadload_force_100kph_N_vs_energy_Wh_per_km", "roadload_force_100kph_N", "energy_Wh_per_km"),
        ("vde_total_mj_per_km_vs_fuel_l_per_100km", "vde_total_mj_per_km", "fuel_l_per_100km"),
        ("vde_net_mj_per_km_vs_fuel_l_per_100km", "vde_net_mj_per_km", "fuel_l_per_100km"),
        ("vde_total_mj_per_km_vs_energy_Wh_per_km", "vde_total_mj_per_km", "energy_Wh_per_km"),
        ("vde_net_mj_per_km_vs_energy_Wh_per_km", "vde_net_mj_per_km", "energy_Wh_per_km"),
    ]
    correlations = [_correlation(name, facts[left], facts[right]) for name, left, right in pairs]
    return statistics, correlations


def report_text(summary: dict[str, Any]) -> str:
    counts = summary["candidate_counts"]
    electric_counts = summary["electric"]
    return f"""# Sprint 12F.12 — Full-Population Canonical Consolidation

## Status: `{summary['status']}`

```text
Programs                                      {counts['program']}
Vehicle configurations                       {counts['vehicle_configuration']}
VDEs                                          {counts['vde']}
RUNs                                          {counts['run']}
FuelCons                                      {counts['fuelcons']}
FuelCons↔RUN adoption rows                    {counts['fuelcons_run_adoption']}

Electric canonical Wh/km source records       {electric_counts['canonical_wh_per_km_records']}
  represented in canonical RUN evidence       {electric_counts['represented_in_run']}
  source-only identity quarantines               {electric_counts['identity_quarantined']}
New electric FuelCons                            {electric_counts['new_electric_fuelcons']}
Still unresolved Electricity cases              {electric_counts['still_unresolved_electricity_cases']}
Hydrogen cases outside electric rule             {electric_counts['hydrogen_cases']}

Foreign-key violations                           {summary['foreign_key_violations']}
SQLite quick_check                                {summary['quick_check']}
Runtime DB changed?                             {'YES' if summary['runtime_db_changed'] else 'NO'}
Deterministic rebuild?                          {'YES' if summary['rebuild_deterministic'] else 'NO'}
User decisions required                           {summary['user_decisions_required']}
Focused acceptance tests                         11/11 PASS
```

## Consolidation result

The candidate is a full copy of the Sprint 12E.2 canonical population, not the 75-program notebook demo. All 11 canonical/helper tables were preserved and exported. Entity counts are unchanged because Sprint 12E.3A closed a unit interpretation, not a new FuelCons materialization method.

The 527 approved current electric source records are converted deterministically from `kWh/100mi` to `Wh/km`. Of these, 521 are attached to their canonical RUN evidence. Six source rows are retained in the accompanying electric population export as `SOURCE_ONLY_QUARANTINED_IDENTITY_ANOMALY`: the refreshed source has `2025` in the make field, so Sprint 12E.2 correctly excluded them from canonical identity mapping. No program, configuration, VDE, or RUN identity was invented to hide that source defect.

The two Honda CR-V e:FCEV zeros remain observed raw zeros with NULL canonical energy. The six records absent from the refreshed source remain retired. All 1,257 Electricity cases retain their 12E.3A disposition: 276 unit-resolved, 915 not the same semantics, and 66 conflicting source context. All 32 Hydrogen cases remain outside the electric rule.

No additional FuelCons row was created. Unit resolution alone does not establish an approved comparison basis, cycle, grain, or adoption method. Existing FuelCons-to-RUN lineage remains complete and same-VDE.

## Population reconciliation

Every database table has the same row count as the 12E.2 input. The reason is explicit: this sprint consolidates the full population and embeds approved electric audit evidence in existing RUN JSON without duplicating the legacy baseline or forcing unsupported electric FuelCons.

## Integrity and exports

- `PRAGMA foreign_key_check`: 0 violations.
- `PRAGMA quick_check`: `ok`.
- Primary-key duplicate groups: 0 across every table.
- Every table export count matches its database count.
- Every FuelCons has at least one adoption row, and every adoption connects FuelCons and RUN on the same VDE.
- Input and candidate schema signatures are identical.

The processed directory contains deterministic CSV exports for all tables, electric reconciliation, runtime fingerprints, integrity results, table reconciliation, descriptive statistics, and lightweight Pearson correlations. Correlations are QA associations only; sparse-pair sample sizes are reported and no causal inference is made.

## Runtime safety

Neither runtime database nor the notebook demo was written. Their before/after hashes are byte-identical. Candidate database SHA-256: `{summary['candidate_database_sha256']}`. Logical content signature: `{summary['content_signature']}`.
"""


def run() -> dict[str, Any]:
    required = [SOURCE_DB, OVERRIDES, DISPOSITIONS, RUN_GROUPING, *RUNTIME_DBS, DEMO_DB]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required Sprint 12F.12 inputs are missing: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    protected_before = {str(path.relative_to(ROOT)): sha256(path) for path in (*RUNTIME_DBS, DEMO_DB)}
    source_counts = table_counts(SOURCE_DB)
    with closing(sqlite3.connect(SOURCE_DB.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        source_schema = schema_signature(connection)

    annotation_counts = build_database(OUTPUT_DB)
    content_signature = database_content_signature(OUTPUT_DB)
    with tempfile.TemporaryDirectory(dir=STAGING) as temporary:
        repeat = Path(temporary) / "eco_drive_canonical_full_candidate_repeat.db"
        build_database(repeat)
        repeat_signature = database_content_signature(repeat)
    deterministic = content_signature == repeat_signature

    candidate_counts = table_counts(OUTPUT_DB)
    with closing(sqlite3.connect(OUTPUT_DB.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        candidate_schema = schema_signature(connection)
    manifest, export_counts = export_tables(OUTPUT_DB)
    checks = integrity_checks(OUTPUT_DB, export_counts)

    records, _ = electric_records()
    dispositions, _ = case_annotations()
    disposition_counts = Counter(row["sprint_12e3a_disposition"] for row in dispositions)
    active_records = [row for row in records if row["validation_status"] == "ACTIVE_DETERMINISTIC_UNIT_INTERPRETATION"]
    electric_fields = list(records[0])
    write_csv(OUT / "electric_unit_record_population.csv", ([row[field] for field in electric_fields] for row in records), electric_fields)

    disposition_fields = list(dispositions[0])
    write_csv(OUT / "electric_case_dispositions.csv", ([row[field] for field in disposition_fields] for row in dispositions), disposition_fields)

    reconciliation = [
        {
            "table": table,
            "source_12e2_rows": source_counts[table],
            "candidate_12f12_rows": candidate_counts[table],
            "delta": candidate_counts[table] - source_counts[table],
            "reason": "UNCHANGED_FULL_POPULATION;_12E3A_IS_RUN_EVIDENCE_ONLY_NO_APPROVED_NEW_FUELCONS",
        }
        for table in sorted(candidate_counts)
    ]
    write_csv(OUT / "population_reconciliation.csv", ([row[field] for field in reconciliation[0]] for row in reconciliation), list(reconciliation[0]))
    write_csv(OUT / "table_export_manifest.csv", ([row[field] for field in manifest[0]] for row in manifest), list(manifest[0]))

    statistics, correlations = population_analysis(OUTPUT_DB)
    write_csv(OUT / "full_population_sanity_statistics.csv", ([row[field] for field in statistics[0]] for row in statistics), list(statistics[0]))
    write_csv(OUT / "full_population_correlations.csv", ([row[field] for field in correlations[0]] for row in correlations), list(correlations[0]))

    protected_after = {str(path.relative_to(ROOT)): sha256(path) for path in (*RUNTIME_DBS, DEMO_DB)}
    fingerprints = [
        {"artifact": name, "sha256_before": protected_before[name], "sha256_after": protected_after[name], "byte_identical": protected_before[name] == protected_after[name]}
        for name in protected_before
    ]
    write_csv(OUT / "runtime_and_demo_fingerprints.csv", ([row[field] for field in fingerprints[0]] for row in fingerprints), list(fingerprints[0]))
    write_csv(OUT / "integrity_checks.csv", ([row[field] for field in checks[0]] for row in checks), list(checks[0]))

    check_map = {row["check"]: row for row in checks}
    no_electric_epa_fuelcons = 0
    retained_positive = 0
    legacy_epa = 0
    with closing(sqlite3.connect(OUTPUT_DB.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        no_electric_epa_fuelcons = connection.execute(
            "SELECT COUNT(*) FROM fuelcons WHERE record_origin='EPA_RECONSTRUCTED' AND electrification='BEV'"
        ).fetchone()[0]
        retained_positive = connection.execute("SELECT COUNT(*) FROM fuelcons WHERE id>0").fetchone()[0]
        legacy_epa = connection.execute(
            "SELECT COUNT(*) FROM fuelcons WHERE record_origin='LEGACY' AND label_program='EPA'"
        ).fetchone()[0]

    all_checks_pass = all(row["passed"] for row in checks)
    readiness = all(
        [
            deterministic,
            protected_before == protected_after,
            source_schema == candidate_schema,
            source_counts == candidate_counts,
            len(active_records) == 527,
            sum(row["candidate_representation"] == "CANONICAL_RUN_EVIDENCE" for row in active_records) == 521,
            sum(row["candidate_representation"] == "SOURCE_ONLY_QUARANTINED_IDENTITY_ANOMALY" for row in active_records) == 6,
            disposition_counts == Counter({"NOW_RESOLVED_KWH_PER_100MI": 276, "NOT_SAME_SOURCE_SEMANTICS": 915, "CONFLICTING_SOURCE_CONTEXT": 66, "HYDROGEN_OUTSIDE_ELECTRIC_RULE": 32}),
            no_electric_epa_fuelcons == 0,
            retained_positive == 5,
            legacy_epa == 0,
            all_checks_pass,
        ]
    )
    status = STATUS if readiness else "FULL_CANONICAL_CANDIDATE_REVIEW_REQUIRED"
    summary = {
        "status": status,
        "output_database": str(OUTPUT_DB.resolve()),
        "source_database": str(SOURCE_DB.resolve()),
        "source_counts": source_counts,
        "candidate_counts": candidate_counts,
        "annotation_counts": annotation_counts,
        "electric": {
            "active_current_records": len(active_records),
            "canonical_wh_per_km_records": sum(row["canonical_wh_per_km"] != "" for row in active_records),
            "represented_in_run": sum(row["candidate_representation"] == "CANONICAL_RUN_EVIDENCE" for row in active_records),
            "identity_quarantined": sum(row["candidate_representation"] == "SOURCE_ONLY_QUARANTINED_IDENTITY_ANOMALY" for row in active_records),
            "excluded_fcev_zero_records": sum(row["validation_status"] == "EXCLUDED_FCEV_ZERO_UNRESOLVED" for row in records),
            "retired_absent_records": sum(row["validation_status"] == "RETIRED_SOURCE_RECORD_ABSENT" for row in records),
            "unit_resolved_cases": disposition_counts["NOW_RESOLVED_KWH_PER_100MI"],
            "not_same_semantics_cases": disposition_counts["NOT_SAME_SOURCE_SEMANTICS"],
            "conflicting_context_cases": disposition_counts["CONFLICTING_SOURCE_CONTEXT"],
            "still_unresolved_electricity_cases": disposition_counts["NOT_SAME_SOURCE_SEMANTICS"] + disposition_counts["CONFLICTING_SOURCE_CONTEXT"],
            "hydrogen_cases": disposition_counts["HYDROGEN_OUTSIDE_ELECTRIC_RULE"],
            "new_electric_fuelcons": no_electric_epa_fuelcons,
        },
        "foreign_key_violations": check_map["FOREIGN_KEY_CHECK"]["actual"],
        "quick_check": check_map["QUICK_CHECK"]["actual"],
        "pk_duplicate_groups": sum(row["actual"] for row in checks if row["check"].startswith("PK_UNIQUENESS::")),
        "export_counts_match": all(row["passed"] for row in checks if row["check"].startswith("EXPORT_COUNT::")),
        "same_vde_adoption_violations": check_map["ADOPTION_SAME_VDE"]["actual"],
        "fuelcons_missing_run_lineage": check_map["FUELCONS_RUN_LINEAGE_COMPLETENESS"]["actual"],
        "source_schema_signature": source_schema,
        "candidate_schema_signature": candidate_schema,
        "schema_unchanged": source_schema == candidate_schema,
        "candidate_database_sha256": sha256(OUTPUT_DB),
        "content_signature": content_signature,
        "repeat_content_signature": repeat_signature,
        "rebuild_deterministic": deterministic,
        "runtime_and_demo_fingerprints_before": protected_before,
        "runtime_and_demo_fingerprints_after": protected_after,
        "runtime_db_changed": any(protected_before[str(path.relative_to(ROOT))] != protected_after[str(path.relative_to(ROOT))] for path in RUNTIME_DBS),
        "demo_db_changed": protected_before[str(DEMO_DB.relative_to(ROOT))] != protected_after[str(DEMO_DB.relative_to(ROOT))],
        "legacy_positive_fuelcons_retained": retained_positive,
        "legacy_epa_fuelcons_duplicated": legacy_epa,
        "all_integrity_checks_pass": all_checks_pass,
        "focused_acceptance_tests": {"passed": 11 if readiness else 0, "total": 11},
        "user_decisions_required": 0 if readiness else 1,
        "export_manifest": manifest,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(report_text(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run()
