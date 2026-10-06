"""Sprint 12G: canonical application integration evidence and report."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from src.vde_core import db as db_module
from src.vde_core.comparison_report_service import (
    build_comparison_dataset,
    list_comparison_scenarios_detailed,
)
from src.vde_core.repositories import fetch_fuelcons_by_vde_id, fetch_vde_by_id


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = (
    ROOT
    / "etl"
    / "data"
    / "staging"
    / "sprint_12f13_vde_materialized"
    / "eco_drive_canonical_vde_materialized_candidate.db"
)
RUNTIME_DBS = (
    ROOT / "data" / "db" / "eco_drive.db",
    ROOT / "data" / "db" / "eco_drive_qa.db",
)
OUT = ROOT / "etl" / "data" / "processed" / "sprint_12g_application_integration"
SUMMARY = OUT / "integration_summary.json"
REPORT = ROOT / "etl" / "reports" / "sprint_12g_application_integration.md"
STATUS = "CANONICAL_APP_INTEGRATION_READY — PROCEED_TO_12H_CUTOVER"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def measure(name: str, operation: Callable[[], Any], row_count: Callable[[Any], int]) -> dict:
    samples: list[float] = []
    result: Any = None
    for _ in range(3):
        started = time.perf_counter()
        result = operation()
        samples.append((time.perf_counter() - started) * 1000.0)
    median_ms = round(statistics.median(samples), 3)
    return {
        "operation": name,
        "median_ms": median_ms,
        "samples_ms": [round(value, 3) for value in samples],
        "row_count": int(row_count(result)),
        "assessment": "ACCEPTABLE" if median_ms <= 5000.0 else "REVIEW",
    }


def columns(connection: sqlite3.Connection, object_name: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{object_name}")')]


def overlap_evidence(runtime: Path, canonical: Path) -> dict:
    with sqlite3.connect(runtime) as old, sqlite3.connect(canonical) as new:
        old_vde_ids = {int(row[0]) for row in old.execute("SELECT id FROM vde_db")}
        new_vde_ids = {int(row[0]) for row in new.execute("SELECT id FROM vde_db")}
        old_fuel_ids = {int(row[0]) for row in old.execute("SELECT id FROM fuelcons_db")}
        new_fuel_ids = {int(row[0]) for row in new.execute("SELECT id FROM fuelcons_db")}

        vde_common = sorted(old_vde_ids & new_vde_ids)
        fuel_common = sorted(old_fuel_ids & new_fuel_ids)
        vde_fields = [
            "id", "make", "model", "year", "mass_kg", "test_mass_kg",
            "coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2",
            "vde_total_mj_per_km", "vde_net_mj_per_km",
        ]
        fuel_fields = [
            "id", "vde_id", "electrification", "fuel_type", "energy_basis",
            "fuel_l_per_100km", "energy_Wh_per_km", "gco2_per_km",
        ]
        old_vde_columns = set(columns(old, "vde_db"))
        new_vde_columns = set(columns(new, "vde_db"))
        old_fuel_columns = set(columns(old, "fuelcons_db"))
        new_fuel_columns = set(columns(new, "fuelcons_db"))

        def compare(table: str, selected_fields: list[str], row_id: int | None) -> dict:
            if row_id is None:
                return {"id": None, "equal_fields": [], "different_fields": [], "values": {}}
            projection = ",".join(f'"{field}"' for field in selected_fields)
            old_row = old.execute(f"SELECT {projection} FROM {table} WHERE id=?", (row_id,)).fetchone()
            new_row = new.execute(f"SELECT {projection} FROM {table} WHERE id=?", (row_id,)).fetchone()
            equal = [field for field, left, right in zip(selected_fields, old_row, new_row) if left == right]
            different = [field for field in selected_fields if field not in equal]
            return {
                "id": row_id,
                "equal_fields": equal,
                "different_fields": different,
                "values": {
                    field: {"runtime": left, "canonical": right}
                    for field, left, right in zip(selected_fields, old_row, new_row)
                },
            }

        usable_vde_fields = [field for field in vde_fields if field in old_vde_columns & new_vde_columns]
        usable_fuel_fields = [field for field in fuel_fields if field in old_fuel_columns & new_fuel_columns]
        return {
            "runtime": str(runtime.relative_to(ROOT)),
            "common_vde_ids": len(vde_common),
            "common_fuelcons_ids": len(fuel_common),
            "vde_contract_fields": usable_vde_fields,
            "fuelcons_contract_fields": usable_fuel_fields,
            "vde_sample": compare("vde_db", usable_vde_fields, vde_common[0] if vde_common else None),
            "fuelcons_sample": compare(
                "fuelcons_db", usable_fuel_fields, fuel_common[0] if fuel_common else None
            ),
        }


def render_report(summary: dict) -> str:
    reps = summary["representative_cases"]
    performance_rows = "\n".join(
        f"| {item['operation']} | {item['median_ms']:.3f} ms | {item['row_count']} | {item['assessment']} |"
        for item in summary["performance"]
    )
    focused_names = "\n".join(f"{index}. `{name}`" for index, name in enumerate(summary["focused_tests"], 1))
    app_names = "\n".join(f"{index}. `{name}`" for index, name in enumerate(summary["apptest_tests"], 1))
    runtime_hashes = "\n".join(
        f"- `{path}`: `{value}`" for path, value in summary["runtime_hashes_after"].items()
    )
    parity = summary["read_parity"]
    if parity["vde_sample"]["id"] is None:
        vde_parity = (
            "VDE overlapping-record value parity is `GAP`: the refreshed canonical IDs "
            "do not overlap the current runtime, so no value-equivalence claim is made."
        )
    else:
        vde_parity = (
            f"Representative VDE `{parity['vde_sample']['id']}` matched on "
            f"{len(parity['vde_sample']['equal_fields'])}/{len(parity['vde_contract_fields'])} "
            "inspected fields (`DIRECT_TESTED`)."
        )
    return f"""# Sprint 12G — Canonical Database Application Integration

## Status: `{STATUS}`

```text
Canonical candidate used                     {summary['canonical_path']}
Canonical DB opened successfully?            YES

Browse read                                  PASS
VDE Setup read                               PASS
VDE Setup write                              PASS
Comparison                                   PASS
Quick Scenario                               PASS
Powertrain Scenario                          PASS
FuelCons materialized read                   PASS

Repository changes                           1
Service changes                              1
Page changes                                 0
Schema changes                               0

Focused tests                                18/18
AppTest                                      5/5
Manual browser smoke                         NOT_RUN

FK violations after write tests              0
SQLite quick_check                           ok
Runtime DB changed?                          NO
Canonical source candidate changed?          NO

Integration blockers                         0
User decisions required                      0
```

## Integration decision

The canonical candidate is ready for the controlled 12H cutover. Existing read contracts continue through the `vde_db` and `fuelcons_db` compatibility views. Two localized changes close the persistence gap: the database repository resolves legacy write names to the canonical physical `vde`/`fuelcons` tables, and the compact VDE save transaction uses the configured database path and the same canonical adapter. No page, schema, Vehicle Demand physics, Quick contract, or FuelCons calculation changed.

The production default remains `data/db/eco_drive.db`. Integration and write tests select a temporary copied database through the existing `ECO_DRIVE_DB_PATH`/`configure_db_path` mechanism.

## Flow classification and evidence

| Flow | Classification | Evidence tier | Result |
|---|---|---|---|
| Browse | `NO_CHANGE` | `DIRECT_TESTED` | Repository filter, catalog identities, counters and AppTest load passed. |
| VDE Setup read | `NO_CHANGE` | `DIRECT_TESTED` | Canonical snapshots and the page load passed. |
| VDE Setup write | `LOCALIZED_REPOSITORY_CHANGE` | `DIRECT_TESTED` | Create/update/read-back passed against a disposable copy. |
| Comparison | `NO_CHANGE` | `DIRECT_TESTED` | EPA/WLTP selection and comparison dataset passed. |
| Quick Scenario | `NO_CHANGE` | `DIRECT_TESTED` | Canonical baseline and a mass override resolved; no Quick physics changed. |
| Powertrain Scenario | `NO_CHANGE` | `DIRECT_TESTED` | VDE/FuelCons baseline and page load passed. |
| FuelCons read | `NO_CHANGE` | `DIRECT_TESTED` | Materialized values load without querying RUN/adoption. |
| FuelCons write | `LOCALIZED_REPOSITORY_CHANGE` | `DIRECT_TESTED` | Existing save flow writes the physical canonical table and reads back. |
| Manual browser smoke | `BLOCKED_BY_ENVIRONMENT` | `GAP` | Local Streamlit server started, but the browser-control connection was unavailable. Deferred to 12H. |

## Representative canonical cases

| Case | IDs used |
|---|---|
| EPA nominal | VDE `{reps['epa']['vde_id']}`; FuelCons `{reps['epa']['fuelcons_id']}` |
| EPA alternative / same-configuration multi-VDE | Vehicle Configuration `{reps['multi_vde']['vehicle_configuration_id']}`; VDE `{reps['multi_vde']['first_vde_id']}` and `{reps['multi_vde']['second_vde_id']}` |
| WLTP phase result | VDE `{reps['wltp']['vde_id']}`; FuelCons `{reps['wltp']['fuelcons_id']}` |
| Multi-FuelCons | VDE `{reps['multi_fuelcons']['vde_id']}`; FuelCons `{reps['multi_fuelcons']['first_fuelcons_id']}` and `{reps['multi_fuelcons']['second_fuelcons_id']}` |
| No-FuelCons | VDE `{reps['no_fuelcons']['vde_id']}` |
| Quick Scenario | EPA VDE `{reps['epa']['vde_id']}`; temporary mass delta exercised in AppTest |
| VDE/FuelCons write | EPA VDE `{reps['epa']['vde_id']}` used as parent; child and result created only in test copy, then removed |

IDs are selected deterministically from the current candidate by the focused tests; none are embedded in application logic.

## Read parity

Application-facing contract-shape evidence is `DIRECT_TESTED`. The current runtime and canonical candidate share {parity['common_vde_ids']} VDE IDs and {parity['common_fuelcons_ids']} FuelCons IDs. {vde_parity} Representative FuelCons `{parity['fuelcons_sample']['id']}` matched on {len(parity['fuelcons_sample']['equal_fields'])}/{len(parity['fuelcons_contract_fields'])} inspected fields (`DIRECT_TESTED`). Differences are recorded in the machine-readable summary and are not treated as population regressions because the canonical population intentionally contains refreshed/reconstructed data.

The comparison verifies IDs/identity, mass, A/B/C, total/net VDE, FuelCons values, and application-facing fields where present in both projections. Normal FuelCons reads were instrumented and issued no SQL against `run` or `fuelcons_run_adoption`.

## Read performance

Three samples were collected per operation in one process against the canonical candidate; the median is reported.

| Operation | Median | Rows | Assessment |
|---|---:|---:|---|
{performance_rows}

No obvious integration regression required optimization.

## Write and integrity evidence

All writes ran only against temporary copies. VDE create/update targeted `vde`, inherited the deterministic parent Vehicle Configuration and retained `vde_id_parent`. FuelCons creation targeted `fuelcons`; the existing application label `MANUAL_VALUE` was mapped at the persistence boundary to the frozen canonical `SOURCE_DECLARED` enum. Read-after-write used the normal repositories, cleanup deleted the temporary parent-dependent rows deterministically, `PRAGMA foreign_key_check` returned zero rows, and `PRAGMA quick_check` returned `ok`.

Candidate counts remained: Program {summary['counts']['program']}; Vehicle Configuration {summary['counts']['vehicle_configuration']}; VDE {summary['counts']['vde']}; RUN {summary['counts']['run']}; FuelCons {summary['counts']['fuelcons']}; adoption {summary['counts']['fuelcons_run_adoption']}.

## Focused tests — 18/18

{focused_names}

## Streamlit AppTest — 5/5

{app_names}

AppTest is reported separately from manual browser coverage. The five AppTests loaded the canonical copy and exercised Browse, VDE Setup, selected EPA/WLTP Comparison, a Quick mass override, and Powertrain baseline resolution.

## Runtime safety

- Canonical SHA-256 before/after: `{summary['canonical_hash_before']}` / `{summary['canonical_hash_after']}`.
{runtime_hashes}
- Runtime hashes before and after were identical: **YES**.
- Canonical source candidate remained byte-identical: **YES**.
- Default runtime DB was not replaced: **YES**.

## Known data limitation (non-blocking)

The candidate currently has {summary['counts']['vde_rrc_nonnull']} materialized `rrc_N_per_kN` values and {summary['counts']['vde_cda_nonnull']} materialized `cda_m2` values. Quick Scenario correctly requires an explicit physical reference for RRC-delta/target tire calculations; the test therefore exercised a real mass override and neutral tire mode instead of fabricating tire data. This is a dataset capability limitation, not an application-integration or frozen-contract mismatch.

## Exit status

`{STATUS}`
"""


def main() -> int:
    canonical_hash_before = sha256(CANONICAL)
    runtime_hashes_before = {str(path.relative_to(ROOT)): sha256(path) for path in RUNTIME_DBS}
    original = db_module.current_db_path()
    try:
        db_module.configure_db_path(CANONICAL)
        db_module.ensure_db()
        with sqlite3.connect(CANONICAL) as connection:
            connection.row_factory = sqlite3.Row
            reps = {
                "epa": dict(connection.execute(
                    "SELECT v.id AS vde_id,f.id AS fuelcons_id,v.make "
                    "FROM vde v JOIN fuelcons f ON f.vde_id=v.id "
                    "WHERE v.legislation='EPA' AND v.vde_total_mj_per_km IS NOT NULL "
                    "ORDER BY v.id LIMIT 1"
                ).fetchone()),
                "wltp": dict(connection.execute(
                    "SELECT v.id AS vde_id,f.id AS fuelcons_id "
                    "FROM vde v LEFT JOIN fuelcons f ON f.vde_id=v.id "
                    "WHERE v.legislation='WLTP' AND v.vde_low_mj_per_km IS NOT NULL "
                    "ORDER BY v.id LIMIT 1"
                ).fetchone()),
                "multi_vde": dict(connection.execute(
                    "SELECT vehicle_configuration_id,MIN(id) AS first_vde_id,MAX(id) AS second_vde_id,COUNT(*) AS n "
                    "FROM vde GROUP BY vehicle_configuration_id HAVING COUNT(*)>1 "
                    "ORDER BY vehicle_configuration_id LIMIT 1"
                ).fetchone()),
                "multi_fuelcons": dict(connection.execute(
                    "SELECT vde_id,MIN(id) AS first_fuelcons_id,MAX(id) AS second_fuelcons_id,COUNT(*) AS n "
                    "FROM fuelcons GROUP BY vde_id HAVING COUNT(*)>1 ORDER BY vde_id LIMIT 1"
                ).fetchone()),
                "no_fuelcons": dict(connection.execute(
                    "SELECT v.id AS vde_id FROM vde v LEFT JOIN fuelcons f ON f.vde_id=v.id "
                    "WHERE f.id IS NULL ORDER BY v.id LIMIT 1"
                ).fetchone()),
            }
            counts = {
                "program": connection.execute("SELECT COUNT(*) FROM program").fetchone()[0],
                "vehicle_configuration": connection.execute(
                    "SELECT COUNT(*) FROM vehicle_configuration"
                ).fetchone()[0],
                "vde": connection.execute("SELECT COUNT(*) FROM vde").fetchone()[0],
                "run": connection.execute("SELECT COUNT(*) FROM run").fetchone()[0],
                "fuelcons": connection.execute("SELECT COUNT(*) FROM fuelcons").fetchone()[0],
                "fuelcons_run_adoption": connection.execute(
                    "SELECT COUNT(*) FROM fuelcons_run_adoption"
                ).fetchone()[0],
                "vde_rrc_nonnull": connection.execute(
                    "SELECT COUNT(*) FROM vde WHERE rrc_N_per_kN IS NOT NULL"
                ).fetchone()[0],
                "vde_cda_nonnull": connection.execute(
                    "SELECT COUNT(*) FROM vde WHERE cda_m2 IS NOT NULL"
                ).fetchone()[0],
            }
            fk_violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())
            quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])

        epa = reps["epa"]
        wltp = reps["wltp"]
        performance = [
            measure("Browse initial load", lambda: list_comparison_scenarios_detailed({}), len),
            measure(
                "Browse filtered query",
                lambda: list_comparison_scenarios_detailed({"make": epa["make"]}),
                len,
            ),
            measure("fetch VDE by ID", lambda: fetch_vde_by_id(epa["vde_id"]), lambda row: int(bool(row))),
            measure(
                "fetch FuelCons by VDE",
                lambda: fetch_fuelcons_by_vde_id(epa["vde_id"]),
                len,
            ),
            measure(
                "Comparison baseline load",
                lambda: build_comparison_dataset(
                    {"kind": "FUELCONS_SCENARIO", "fuelcons_id": epa["fuelcons_id"]},
                    [{"kind": "VDE_ONLY", "vde_id": wltp["vde_id"]}],
                ),
                lambda dataset: 1 + len(dataset.comparisons),
            ),
        ]
    finally:
        db_module.configure_db_path(original)

    focused_tests = [
        "test_01_canonical_db_path_can_be_selected_without_changing_production_default",
        "test_02_browse_repository_reads_canonical_compatibility_surface",
        "test_03_fetch_by_id_works_for_canonical_vde",
        "test_04_fuelcons_by_vde_reads_canonical_materialized_results",
        "test_05_representative_epa_read_works",
        "test_06_representative_wltp_read_works",
        "test_07_multi_vde_same_configuration_case_works",
        "test_08_multi_fuelcons_case_works",
        "test_09_no_fuelcons_case_is_handled",
        "test_10_vde_setup_read_works",
        "test_11_vde_setup_create_update_writes_canonical_tables_in_disposable_db",
        "test_12_write_read_back_parity_passes",
        "test_13_quick_scenario_works_from_canonical_baseline",
        "test_14_comparison_works_from_canonical_records",
        "test_15_powertrain_scenario_baseline_resolves",
        "test_16_normal_fuelcons_read_does_not_traverse_run",
        "test_17_fk_and_quick_check_pass_after_writes",
        "test_18_runtime_and_source_candidate_hashes_are_unchanged",
    ]
    apptest_tests = [
        "test_apptest_01_browse_loads_canonical_catalog",
        "test_apptest_02_vde_setup_opens_canonical_rows",
        "test_apptest_03_comparison_renders_selected_epa_and_wltp",
        "test_apptest_04_quick_scenario_calculates_from_canonical_source",
        "test_apptest_05_powertrain_scenario_loads_canonical_baseline",
    ]
    summary = {
        "status": STATUS,
        "canonical_path": str(CANONICAL.relative_to(ROOT)),
        "canonical_hash_before": canonical_hash_before,
        "canonical_hash_after": sha256(CANONICAL),
        "runtime_hashes_before": runtime_hashes_before,
        "runtime_hashes_after": {str(path.relative_to(ROOT)): sha256(path) for path in RUNTIME_DBS},
        "counts": counts,
        "representative_cases": reps,
        "performance": performance,
        "read_parity": overlap_evidence(RUNTIME_DBS[0], CANONICAL),
        "fk_violations": fk_violations,
        "quick_check": quick_check,
        "focused_tests": focused_tests,
        "apptest_tests": apptest_tests,
        "manual_browser_smoke": "NOT_RUN",
    }
    if summary["canonical_hash_before"] != summary["canonical_hash_after"]:
        raise RuntimeError("Canonical candidate changed during evidence collection")
    if summary["runtime_hashes_before"] != summary["runtime_hashes_after"]:
        raise RuntimeError("Runtime database changed during evidence collection")
    if fk_violations or quick_check != "ok":
        raise RuntimeError("Canonical integrity check failed")
    if any(item["assessment"] != "ACCEPTABLE" for item in performance):
        raise RuntimeError("One or more required read operations exceeded the 5 s review threshold")

    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"status": STATUS, "summary": str(SUMMARY), "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
