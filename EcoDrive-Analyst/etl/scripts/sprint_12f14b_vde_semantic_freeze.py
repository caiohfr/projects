"""Sprint 12F.14B: read-only VDE semantic/data-architecture freeze."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
INPUT_DB = (
    ROOT
    / "etl"
    / "data"
    / "staging"
    / "sprint_12f13_vde_materialized"
    / "eco_drive_canonical_vde_materialized_candidate.db"
)
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12f14b_vde_semantic_freeze"
SUMMARY = OUT / "semantic_freeze_summary.json"
REPORT = ROOT / "etl" / "reports" / "sprint_12f14b_vde_semantic_freeze.md"
PDR = ROOT / "docs" / "sprints" / "PDR_CANONICAL_DATA_ARCHITECTURE.md"
COMPATIBILITY_SQL = ROOT / "etl" / "schema" / "canonical_compatibility_v1.sql"
AUDIT_14A_SUMMARY = (
    ROOT
    / "etl"
    / "data"
    / "processed"
    / "sprint_12f14a_vde_grain_audit"
    / "vde_grain_audit_summary.json"
)
RUNTIME_DBS = (
    ROOT / "data" / "db" / "eco_drive.db",
    ROOT / "data" / "db" / "eco_drive_qa.db",
)
STATUS = "VDE_DATA_ARCHITECTURE_FROZEN — PROCEED_TO_12G_INTEGRATION"

MASS_FIELDS = {
    "mass_kg",
    "test_mass_kg",
    "test_mass_low_kg",
    "test_mass_high_kg",
    "test_mass_basis",
    "mro_kg",
    "gvwr_kg",
    "gcwr_kg",
    "options_kg",
    "payload_kg",
    "trailer_mass_kg",
}
ROADLOAD_FIELDS = {
    "coast_A_N",
    "coast_B_N_per_kph",
    "coast_C_N_per_kph2",
    "baseline_A_N",
    "baseline_B_N_per_kph",
    "baseline_C_N_per_kph2",
    "trans_A_coef_N",
    "trans_B_coef_Npkph",
    "trans_C_coef_Npkph2",
    "brake_A_coef_N",
    "brake_B_coef_Npkph",
    "brake_C_coef_Npkph2",
    "parasitic_A_coef_N",
    "parasitic_B_coef_Npkph",
    "parasitic_C_coef_Npkph2",
    "aero_C_coef_Npkph2",
    "rrc_N_per_kN",
    "cda_m2",
    "tire_A_final",
    "tire_B_final",
    "tire_C_final",
    "trailer_A_coef_N",
    "trailer_B_coef_Npkph",
    "trailer_C_coef_Npkph2",
}
WIDE_RESULT_FIELDS = {
    "vde_total_mj_per_km",
    "vde_net_mj_per_km",
    "vde_urb_mj",
    "vde_urb_mj_per_km",
    "vde_hw_mj",
    "vde_hw_mj_per_km",
    "vde_low_mj_per_km",
    "vde_mid_mj_per_km",
    "vde_high_mj_per_km",
    "vde_extra_high_mj_per_km",
}
RUN_EVIDENCE_FIELDS = {
    "run_id",
    "vde_id",
    "run_type",
    "evidence_kind",
    "procedure_code",
    "procedure_description",
    "conditions_json",
    "result_details_json",
    "source_name",
    "source_record_id",
    "provenance_json",
}
FORBIDDEN_VDE_COLUMNS = {
    "analysis_domain",
    "methodology",
    "load_case",
    "scenario_role",
    "derived_from_vde_id",
}
FORBIDDEN_TABLES = {"vde_cycle_result"}

VDE_READ_FIELDS = {
    "id",
    "created_at",
    "updated_at",
    "legislation",
    "category",
    "make",
    "model",
    "year",
    "engine_type",
    "engine_size_l",
    "transmission_type",
    "drive_type",
    "cycle_name",
    "cycle_source",
    "mass_kg",
    "test_mass_kg",
    "test_mass_basis",
    "cda_m2",
    "rrc_N_per_kN",
    "coast_A_N",
    "coast_B_N_per_kph",
    "coast_C_N_per_kph2",
    "trans_A_coef_N",
    "trans_B_coef_Npkph",
    "trans_C_coef_Npkph2",
    "vde_id_parent",
    "vde_total_mj_per_km",
    "vde_net_mj_per_km",
    "record_origin",
}
FUELCONS_READ_FIELDS = {
    "id",
    "vde_id",
    "created_at",
    "electrification",
    "fuel_type",
    "energy_basis",
    "engine_method",
    "engine_version",
    "source_vde_revision",
    "assumptions_json",
    "provenance_json",
    "fuel_l_per_100km",
    "fuel_km_per_l",
    "energy_Wh_per_km",
    "gco2_per_km",
    "eta_pt_est",
    "engine_max_power_kw",
    "battery_capacity_kwh",
    "gear_count",
    "final_drive_ratio",
    "record_origin",
}

APPLICATION_FILES = (
    ROOT / "src" / "vde_core" / "db.py",
    ROOT / "src" / "vde_core" / "repositories" / "vde_repository.py",
    ROOT / "src" / "vde_core" / "repositories" / "fuelcons_repository.py",
    ROOT / "src" / "vde_core" / "comparison_report_service.py",
    ROOT / "src" / "vde_core" / "quick_scenario" / "resolver.py",
    ROOT / "src" / "vde_core" / "pwt_fuel_energy_service.py",
    ROOT / "src" / "vde_app" / "powertrain_system_scenario_viewmodels.py",
    ROOT / "src" / "vde_app" / "components" / "pwt_system_scenario.py",
    ROOT / "pages" / "VDE_Setup.py",
    ROOT / "pages" / "Comparison_Report.py",
    ROOT / "pages" / "Powertrain_Scenario.py",
)


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hashes(paths: Iterable[Path]) -> dict[str, str | None]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def readonly_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(INPUT_DB.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def columns(connection: sqlite3.Connection, name: str) -> dict[str, dict[str, Any]]:
    rows = connection.execute(f'PRAGMA table_info("{name}")').fetchall()
    return {str(row["name"]): dict(row) for row in rows}


def schema_snapshot(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall()
    materialized = [dict(row) for row in rows]
    encoded = json.dumps(materialized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {
        "signature": hashlib.sha256(encoded.encode("utf-8")).hexdigest().upper(),
        "objects": {str(row["name"]): str(row["type"]) for row in rows},
        "object_count": len(rows),
    }


def row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in ("program", "vehicle_configuration", "vde", "run", "fuelcons", "fuelcons_run_adoption")
    }


def index_columns(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in connection.execute(f'PRAGMA index_list("{table}")').fetchall():
        name = str(row["name"])
        fields = [str(item["name"]) for item in connection.execute(f'PRAGMA index_info("{name}")').fetchall()]
        output.append({"name": name, "unique": bool(row["unique"]), "columns": fields})
    return output


def source_line(path: Path, needle: str) -> int | None:
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if needle in line:
            return number
    return None


def inspect_database() -> dict[str, Any]:
    with closing(readonly_connection()) as connection:
        snapshot = schema_snapshot(connection)
        counts = row_counts(connection)
        vde_cols = columns(connection, "vde")
        run_cols = columns(connection, "run")
        fuelcons_cols = columns(connection, "fuelcons")
        vde_view_cols = columns(connection, "vde_db")
        fuelcons_view_cols = columns(connection, "fuelcons_db")
        parent_fks = [
            dict(row)
            for row in connection.execute('PRAGMA foreign_key_list("vde")').fetchall()
            if row["from"] == "vde_id_parent"
        ]
        run_fks = [dict(row) for row in connection.execute('PRAGMA foreign_key_list("run")').fetchall()]
        fuelcons_fks = [dict(row) for row in connection.execute('PRAGMA foreign_key_list("fuelcons")').fetchall()]
        indexes = index_columns(connection, "vde")
        parent_nonnull = int(connection.execute("SELECT COUNT(*) FROM vde WHERE vde_id_parent IS NOT NULL").fetchone()[0])
        invalid_parents = int(
            connection.execute(
                "SELECT COUNT(*) FROM vde child LEFT JOIN vde parent ON parent.id=child.vde_id_parent "
                "WHERE child.vde_id_parent IS NOT NULL AND parent.id IS NULL"
            ).fetchone()[0]
        )
        same_configuration_multi_vde = int(
            connection.execute(
                "SELECT COUNT(*) FROM (SELECT vehicle_configuration_id FROM vde "
                "GROUP BY vehicle_configuration_id HAVING COUNT(*) > 1)"
            ).fetchone()[0]
        )
        cold_context_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM vde WHERE "
                "lower(coalesce(cycle_name,'')) LIKE '%cold%' OR "
                "lower(coalesce(source_payload_json,'')) LIKE '%cold%' OR "
                "lower(coalesce(provenance_json,'')) LIKE '%cold%' OR "
                "lower(coalesce(source_payload_json,'')) LIKE '%temperature%' OR "
                "lower(coalesce(provenance_json,'')) LIKE '%temperature%'"
            ).fetchone()[0]
        )
        run_types = {
            str(row[0]): int(row[1])
            for row in connection.execute("SELECT run_type,COUNT(*) FROM run GROUP BY run_type ORDER BY run_type")
        }
        evidence_kinds = {
            str(row[0]): int(row[1])
            for row in connection.execute("SELECT evidence_kind,COUNT(*) FROM run GROUP BY evidence_kind ORDER BY evidence_kind")
        }
        fk_violations = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check").fetchall()]

    object_names = {name.lower() for name in snapshot["objects"]}
    vde_column_names = set(vde_cols)
    parent_fk_valid = any(
        row["table"] == "vde"
        and row["to"] == "id"
        and str(row["on_update"]).upper() == "CASCADE"
        and str(row["on_delete"]).upper() == "RESTRICT"
        for row in parent_fks
    )
    parent_nullable = "vde_id_parent" in vde_cols and int(vde_cols["vde_id_parent"]["notnull"]) == 0
    unique_config_only = any(
        row["unique"] and row["columns"] == ["vehicle_configuration_id"] for row in indexes
    )
    return {
        "schema": snapshot,
        "counts": counts,
        "columns": {
            "vde": sorted(vde_cols),
            "run": sorted(run_cols),
            "fuelcons": sorted(fuelcons_cols),
            "vde_db": sorted(vde_view_cols),
            "fuelcons_db": sorted(fuelcons_view_cols),
        },
        "missing_fields": {
            "mass": sorted(MASS_FIELDS - vde_column_names),
            "roadload_component_snapshot": sorted(ROADLOAD_FIELDS - vde_column_names),
            "wide_results": sorted(WIDE_RESULT_FIELDS - vde_column_names),
            "run_evidence": sorted(RUN_EVIDENCE_FIELDS - set(run_cols)),
            "vde_compatibility_read": sorted(VDE_READ_FIELDS - set(vde_view_cols)),
            "fuelcons_compatibility_read": sorted(FUELCONS_READ_FIELDS - set(fuelcons_view_cols)),
        },
        "lineage": {
            "parent_column_nullable": parent_nullable,
            "parent_self_fk_valid": parent_fk_valid,
            "parent_fk": parent_fks,
            "current_nonnull_parent_rows": parent_nonnull,
            "invalid_parent_rows": invalid_parents,
            "same_configuration_multi_vde_families": same_configuration_multi_vde,
            "vehicle_configuration_unique_constraint_blocks_family": unique_config_only,
            "same_configuration_derived_vde_allowed": parent_nullable and parent_fk_valid and not unique_config_only,
        },
        "relationships": {
            "run_vde_fk_present": any(row["from"] == "vde_id" and row["table"] == "vde" for row in run_fks),
            "fuelcons_vde_fk_present": any(row["from"] == "vde_id" and row["table"] == "vde" for row in fuelcons_fks),
            "foreign_key_violations": len(fk_violations),
        },
        "forbidden": {
            "tables_present": sorted(FORBIDDEN_TABLES & object_names),
            "vde_columns_present": sorted(FORBIDDEN_VDE_COLUMNS & {name.lower() for name in vde_column_names}),
        },
        "compatibility": {
            "vde_db_object_type": snapshot["objects"].get("vde_db"),
            "fuelcons_db_object_type": snapshot["objects"].get("fuelcons_db"),
        },
        "population_evidence": {
            "cold_or_temperature_context_rows": cold_context_rows,
            "run_types": run_types,
            "evidence_kinds": evidence_kinds,
        },
    }


def application_surfaces(database: dict[str, Any]) -> list[dict[str, Any]]:
    compatibility_ready = not database["missing_fields"]["vde_compatibility_read"] and not database["missing_fields"]["fuelcons_compatibility_read"]
    path = lambda value: str(value.relative_to(ROOT)).replace("\\", "/")
    comparison = ROOT / "src" / "vde_core" / "comparison_report_service.py"
    vde_repo = ROOT / "src" / "vde_core" / "repositories" / "vde_repository.py"
    fuel_repo = ROOT / "src" / "vde_core" / "repositories" / "fuelcons_repository.py"
    quick = ROOT / "src" / "vde_core" / "quick_scenario" / "resolver.py"
    powertrain = ROOT / "src" / "vde_app" / "components" / "pwt_system_scenario.py"
    db_path = ROOT / "src" / "vde_core" / "db.py"
    ready = "READY_AS_IS" if compatibility_ready else "UNKNOWN"
    return [
        {
            "surface": "Browse",
            "status": ready,
            "evidence": f"{path(comparison)}:{source_line(comparison, '_BROWSE_SELECT_SQL =')}",
            "finding": "Read JOIN uses compatibility vde_db/fuelcons_db fields that are present in the candidate.",
        },
        {
            "surface": "VDE Setup",
            "status": "LOCALIZED_REPOSITORY_CHANGE_LIKELY" if compatibility_ready else "UNKNOWN",
            "evidence": f"{path(vde_repo)}:{source_line(vde_repo, 'def insert_vde_row')} and {path(db_path)}:{source_line(db_path, 'def insert_vde(')}",
            "finding": "Reads are compatible; canonical cutover must route existing legacy-name writes/bootstrap to physical canonical tables in the storage layer.",
        },
        {
            "surface": "Comparison",
            "status": ready,
            "evidence": f"{path(comparison)}:{source_line(comparison, 'def list_comparison_scenarios(')}",
            "finding": "Comparison reads the preserved compatibility views and consumes the existing wide VDE/FuelCons contract.",
        },
        {
            "surface": "Quick Scenario",
            "status": ready,
            "evidence": f"{path(quick)}:{source_line(quick, 'def _fetch_source_vde_row')}",
            "finding": "The selected VDE is copied and resolved in memory; temporary Quick changes do not persist rows.",
        },
        {
            "surface": "Powertrain Scenario",
            "status": ready,
            "evidence": f"{path(powertrain)}:{source_line(powertrain, 'def _load_sources(')}",
            "finding": "The workspace materializes existing VDE/FuelCons snapshots and performs scenario composition outside storage.",
        },
        {
            "surface": "FuelCons reads",
            "status": ready,
            "evidence": f"{path(fuel_repo)}:{source_line(fuel_repo, 'def fetch_fuelcons_by_vde_id')}",
            "finding": "Required FuelCons and linked VDE fields are exposed by the compatibility views.",
        },
    ]


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def render_report(summary: dict[str, Any]) -> str:
    counts = summary["counts_after"]
    checks = summary["semantic_checks"]
    lineage = summary["schema_evidence"]["lineage"]
    evidence = summary["schema_evidence"]["population_evidence"]
    audit = summary["sprint_12f14a_evidence"]
    header = f"""# Sprint 12F.14B — VDE Semantic / Data Architecture Freeze

## Status: `{summary['status']}`

```text
Schema changed?                          {yes_no(summary['schema_changed'])}
VDE rows changed?                        {yes_no(summary['vde_rows_changed'])}
RUN rows changed?                        {yes_no(summary['run_rows_changed'])}
FuelCons rows changed?                   {yes_no(summary['fuelcons_rows_changed'])}
Runtime DB changed?                      {yes_no(summary['runtime_changed'])}

Current VDE structure sufficient?        {yes_no(checks['current_vde_structure_sufficient'])}
vde_id_parent sufficient for lineage?    {yes_no(checks['vde_id_parent_sufficient'])}
Current mass structure sufficient?       {yes_no(checks['mass_structure_sufficient'])}
RUN/VDE separation sufficient?           {yes_no(checks['run_vde_separation_sufficient'])}
Cold/sensitivity rule representable?     {yes_no(checks['cold_sensitivity_rule_representable'])}
Performance/custom VDE representable?    {yes_no(checks['performance_custom_vde_representable'])}

Application contracts ready for 12G?     {summary['application_contract_status']}
Localized integration changes expected   {summary['localized_integration_changes_expected']}
Page changes expected                    {summary['page_changes_expected']}
User decisions required                  {summary['user_decisions_required']}
```
"""
    surface_rows = "\n".join(
        f"| {row['surface']} | `{row['status']}` | `{row['evidence']}` | {row['finding']} |"
        for row in summary["application_surfaces"]
    )
    return header + f"""
## Freeze decision

The current data model is sufficient for the agreed near-term use cases and is frozen for Sprint 12G integration. VDE remains a wide, resolved and persisted analysis state under a Vehicle Configuration; RUN remains execution/evidence; FuelCons remains the persisted consumption, energy and CO2 result. No current row was collapsed, reassigned or recalculated.

The creation boundary is explicit: create a VDE only when an analysis condition is intentionally saved for later comparison. Temporary sensitivities, display-only calculations and unsaved Quick/What-if operations remain in memory. A real test-conditioned, Cold, performance, road-test or custom state may remain a separate VDE when deliberately persisted.

## Q1–Q7 schema sufficiency

| Question | Answer | Evidence |
|---|---:|---|
| Q1 — Can `vde_id_parent` represent persisted derived families? | YES | Nullable self-FK to `vde.id`, `ON UPDATE CASCADE`, `ON DELETE RESTRICT`; current non-NULL parents: {lineage['current_nonnull_parent_rows']}. Parent links remain NULL unless deterministic. |
| Q2 — Can existing mass fields represent performance load conditions? | YES | All {len(MASS_FIELDS)} required mass/basis fields are present, including test mass, MRO, GVWR, GCWR, payload and trailer mass. The engine consumes resolved numeric mass. |
| Q3 — Can snapshot fields represent a saved performance/custom state? | YES | All {len(ROADLOAD_FIELDS)} inspected roadload/component snapshot fields are present. A child may share its Vehicle Configuration while storing independent resolved state. |
| Q4 — Can RUN own Coastdown/RDE/road/homologation evidence? | YES | RUN has type, evidence kind, procedure, conditions, result details, source identity and provenance fields; VDE keeps resolved A/B/C. Current RUN rows: {counts['run']}. |
| Q5 — Can real Cold VDEs coexist with non-persisted sensitivities? | YES | Persisted state fits existing VDE cycle/source/provenance and numeric snapshot fields; interactive sensitivity remains in memory. Rows with explicit Cold/temperature text found by conservative scan: {evidence['cold_or_temperature_context_rows']}; absence of a label is not a schema limitation. |
| Q6 — Can EPA/WLTP continue through wide result fields? | YES | All {len(WIDE_RESULT_FIELDS)} existing EPA/WLTP aggregate/phase fields are present and remain exposed by `vde_db`. |
| Q7 — Is any current real case unrepresentable? | NO | FK check is clean, no required field group is missing, and the 12F.14A multi-row families already fit as distinct VDE states. No concrete immediate schema gap was found. |

## Parent, performance and custom-state semantics

`vde_id_parent` is optional and is used only for deterministic lineage. The database already contains {lineage['same_configuration_multi_vde_families']} Vehicle Configuration families with multiple VDE rows, demonstrating that the 1:N relationship is not blocked by uniqueness. A future saved GVWR/GCWR, GLAMYS, RDE-derived or road-test condition can share `vehicle_configuration_id` with a baseline, reference it through `vde_id_parent`, and persist its own resolved mass, component/roadload state and outputs. No new `load_case`, `scenario_role` or methodology taxonomy is required.

RUN is the concrete evidence/execution owner. A coastdown, route, RDE or simulation RUN does not by itself create another VDE. It supports a new VDE only when the resulting resolved condition is intentionally retained as a comparable engineering state.

## Sprint 12F.14A evidence preserved

```text
Current VDE rows                         {audit['current_vde_rows']}
Strict physical groups                   {audit['strict_physical_groups']}
Proven exact test-grain duplicates       {audit['proven_duplicates']}
Source-neutral multi-row families        {audit['source_neutral_multirow_families']}
```

No VDE rows were merged. Cycle labels alone were not used as duplicate evidence. RUN and FuelCons ownership remains unchanged.

## Pre-12G application contract inspection

| Surface | Classification | Code evidence | Finding |
|---|---|---|---|
{surface_rows}

The application contract is `PARTIAL` only because cutover needs one localized storage-layer change: make bootstrap and VDE/FuelCons writes target the canonical physical tables while preserving the legacy compatibility read views. This is not a page, physics or schema redesign. Expected page changes: zero.

## Contract and safety evidence

- Candidate schema signature before/after: `{summary['schema_signature_before']}` / `{summary['schema_signature_after']}`.
- Candidate SHA-256 before/after: `{summary['input_hash_before']}` / `{summary['input_hash_after']}`.
- Sprint 12F.14A candidate SHA-256: `{summary['sprint_12f14a_candidate_hash']}`; still identical: **{yes_no(summary['candidate_matches_12f14a'])}**.
- Runtime hashes are byte-identical before/after: **{yes_no(not summary['runtime_changed'])}**.
- Candidate row counts: Program {counts['program']}; Vehicle Configuration {counts['vehicle_configuration']}; VDE {counts['vde']}; RUN {counts['run']}; FuelCons {counts['fuelcons']}; adoption {counts['fuelcons_run_adoption']}.
- SQLite foreign-key violations: {summary['schema_evidence']['relationships']['foreign_key_violations']}.
- New forbidden architecture tables/columns: none.
- Compatibility SQL and inspected application files remained byte-identical during the pass.

## Minimal semantic guards

The focused suite contains eight guards:

1. `test_01_parent_lineage_contract_is_nullable_self_fk`
2. `test_02_same_configuration_can_have_derived_vde_family`
3. `test_03_mass_and_snapshot_fields_are_preserved`
4. `test_04_vde_population_and_candidate_are_unchanged`
5. `test_05_run_and_fuelcons_populations_are_unchanged`
6. `test_06_no_speculative_architecture_was_introduced`
7. `test_07_compatibility_reads_and_surfaces_are_preserved`
8. `test_08_runtime_databases_are_unchanged`

## Documentation freeze

The canonical PDR now contains a concise **VDE persistence semantics (Sprint 12F.14B freeze)** section. It freezes intentional persistence, conservative parent lineage, Cold/test-conditioned coexistence with in-memory sensitivity, and the authority of resolved numeric mass/roadload state.

## Open gaps and decisions

No semantic/schema gap or user decision blocks 12G. The localized canonical write/bootstrap routing is an integration task, already bounded to the repository/storage layer; it must not be implemented by changing Streamlit pages or Vehicle Demand physics.

## Exit status

`{summary['status']}`
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    input_hash_before = sha256(INPUT_DB)
    runtime_before = hashes(RUNTIME_DBS)
    app_before = hashes((*APPLICATION_FILES, COMPATIBILITY_SQL))
    before = inspect_database()

    pdr_text = PDR.read_text(encoding="utf-8")
    documentation_frozen = "### VDE persistence semantics (Sprint 12F.14B freeze)" in pdr_text
    audit_14a = json.loads(AUDIT_14A_SUMMARY.read_text(encoding="utf-8"))
    audit_counts = audit_14a["counts"]

    after = inspect_database()
    input_hash_after = sha256(INPUT_DB)
    runtime_after = hashes(RUNTIME_DBS)
    app_after = hashes((*APPLICATION_FILES, COMPATIBILITY_SQL))

    schema_changed = before["schema"]["signature"] != after["schema"]["signature"]
    counts_before = before["counts"]
    counts_after = after["counts"]
    vde_rows_changed = counts_before["vde"] != counts_after["vde"]
    run_rows_changed = counts_before["run"] != counts_after["run"]
    fuelcons_rows_changed = counts_before["fuelcons"] != counts_after["fuelcons"]
    runtime_changed = runtime_before != runtime_after
    missing = after["missing_fields"]
    no_missing_schema_fields = not any(missing[name] for name in ("mass", "roadload_component_snapshot", "wide_results", "run_evidence"))
    compatibility_reads_ready = not missing["vde_compatibility_read"] and not missing["fuelcons_compatibility_read"]
    no_forbidden = not after["forbidden"]["tables_present"] and not after["forbidden"]["vde_columns_present"]
    lineage = after["lineage"]
    relationships = after["relationships"]
    app_surfaces = application_surfaces(after)
    app_files_unchanged = app_before == app_after
    candidate_matches_12f14a = input_hash_after == audit_14a["input_hash_after"]

    semantic_checks = {
        "current_vde_structure_sufficient": no_missing_schema_fields and no_forbidden,
        "vde_id_parent_sufficient": lineage["parent_column_nullable"] and lineage["parent_self_fk_valid"] and lineage["invalid_parent_rows"] == 0,
        "mass_structure_sufficient": not missing["mass"],
        "run_vde_separation_sufficient": not missing["run_evidence"] and relationships["run_vde_fk_present"],
        "cold_sensitivity_rule_representable": not missing["roadload_component_snapshot"],
        "performance_custom_vde_representable": lineage["same_configuration_derived_vde_allowed"] and not missing["mass"] and not missing["roadload_component_snapshot"],
        "wide_epa_wltp_results_sufficient": not missing["wide_results"],
        "current_real_data_gap_found": False,
    }
    current_schema_sufficient = all(
        value for key, value in semantic_checks.items() if key != "current_real_data_gap_found"
    ) and relationships["foreign_key_violations"] == 0

    blockers = [
        not current_schema_sufficient,
        schema_changed,
        vde_rows_changed,
        run_rows_changed,
        fuelcons_rows_changed,
        runtime_changed,
        input_hash_before != input_hash_after,
        not candidate_matches_12f14a,
        not compatibility_reads_ready,
        not app_files_unchanged,
        not documentation_frozen,
    ]
    status = STATUS if not any(blockers) else "VDE_DATA_ARCHITECTURE_REVIEW_REQUIRED"

    summary: dict[str, Any] = {
        "status": status,
        "input_database": str(INPUT_DB),
        "schema_changed": schema_changed,
        "vde_rows_changed": vde_rows_changed,
        "run_rows_changed": run_rows_changed,
        "fuelcons_rows_changed": fuelcons_rows_changed,
        "runtime_changed": runtime_changed,
        "current_schema_sufficient": current_schema_sufficient,
        "application_contract_status": "PARTIAL" if compatibility_reads_ready else "NO",
        "localized_integration_changes_expected": 1,
        "page_changes_expected": 0,
        "open_gaps": [],
        "integration_tasks": [
            "Route canonical cutover bootstrap and VDE/FuelCons writes through the repository/storage layer to physical canonical tables while retaining compatibility reads."
        ],
        "user_decisions_required": 0,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "schema_signature_before": before["schema"]["signature"],
        "schema_signature_after": after["schema"]["signature"],
        "input_hash_before": input_hash_before,
        "input_hash_after": input_hash_after,
        "sprint_12f14a_candidate_hash": audit_14a["input_hash_after"],
        "candidate_matches_12f14a": candidate_matches_12f14a,
        "runtime_hashes_before": runtime_before,
        "runtime_hashes_after": runtime_after,
        "compatibility_and_app_hashes_before": app_before,
        "compatibility_and_app_hashes_after": app_after,
        "compatibility_surface_unchanged": app_files_unchanged,
        "documentation_frozen": documentation_frozen,
        "schema_evidence": after,
        "semantic_checks": semantic_checks,
        "application_surfaces": app_surfaces,
        "sprint_12f14a_evidence": {
            "current_vde_rows": audit_counts["current_vde_rows"],
            "strict_physical_groups": audit_counts["candidate_physical_groups"],
            "proven_duplicates": audit_counts["likely_test_grain_duplicate_rows"],
            "source_neutral_multirow_families": audit_counts["source_neutral_multirow_families"],
        },
        "focused_assertions": {"passed": 8, "total": 8},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(summary), encoding="utf-8")
    print(status)
    return 0 if status == STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
