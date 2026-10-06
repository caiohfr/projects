# Sprint 12F — Engineering Notebook MVP

## Status: `NOTEBOOK_MVP_READY — USER_REVIEW`

```text
Planned notebooks                              11
Executable MVP notebooks                       11
Notebook execution failures                     0
Focused validation tests                      8/8

EPA source rows                             30194
EPA source columns                             67
Resolved EPA electric source rows             527

MVP relational export:
  Program                                       75
  Vehicle Configuration                       199
  VDE                                         235
  RUN                                         583
  FuelCons                                    201
  FuelCons↔RUN adoption                       327

Disposable DB quick_check                      ok
Disposable DB FK violations                     0
Adoption relationship failures                  0
Runtime DB changed?                             NO
```

## Implemented story

1. **EPA ingest and grain** — source rows are compared with Make+Model+Year, test, and configuration grains.
2. **EPA non-BEV EDA** — fuel-side filtering, visible MPG-to-L/100km conversion, missingness, trends, and mass relationship.
3. **EPA BEV energy EDA** — the 12E.3A resolved cohort is filtered and `kWh/100mi → Wh/km` is reproduced visibly.
4. **JRC ingest and grain** — anonymized identity, powertrain flags, engineering-field coverage, and source grain.
5. **JRC non-BEV EDA** — mass/roadload/CO2 summaries and exploratory correlation.
6. **JRC BEV EDA** — explicit Wh/km, declared versus simulated context, and mass relationship.
7. **Vehicle Demand** — visible `A + Bv + Cv²`, steady-speed Wh/km, and coefficient sensitivity.
8. **FuelCons** — visible EPA 55/45 calculation, canonical comparison, and multi-RUN adoption.
9. **Cross-legislation** — Wh/km comparison with source/method/provenance kept explicit.
10. **Canonical build** — deterministic 75-program relational slice with pre-export relationship and same-VDE checks.
11. **Database ingestion** — schema-first, FK-safe loading including FuelCons↔RUN adoption into a guarded disposable SQLite database.

## Code-integrity patch

The original demo database was structurally valid, but the notebook subset
export did not preserve FuelCons↔RUN adoption lineage. The patch adds that
associative relationship to the deterministic export and disposable load.

`12F-10` now reads the authoritative `fuelcons_run_adoption` table directly
from the canonical staging database; it does not reconstruct relationships from
FuelCons JSON. It validates that every adoption resolves to exported FuelCons,
RUN, and VDE rows and that:

```text
adoption.vde_id == fuelcons.vde_id == run.vde_id
```

`12F-11` loads the relationship after RUN and FuelCons, reconciles its row
count with the CSV, and requires a non-empty result.

## MVP outputs

The notebook index and machine-readable manifest are:

- `notebooks/README.md`
- `notebooks/notebook_manifest.json`

12F-10 writes six CSVs under `notebooks/_data/`, including
`12f10_fuelcons_run_adoption.csv`. 12F-11 writes only
`notebooks/_data/12f11_canonical_notebook_demo.db`. CSV and database counts
match for all six tables. SQLite `quick_check` is `ok`; `foreign_key_check`
returns zero rows. Semantic checks also report zero missing FuelCons/RUN links
and zero same-VDE mismatches across the 327 adoption rows.

## Verification

Direct automated evidence:

- `test_all_eleven_notebooks_are_valid_mvp_artifacts`
- `test_all_notebooks_execute_in_sequence`
- `test_canonical_exports_and_disposable_database_are_valid`
- `test_runtime_databases_remain_byte_identical`
- setup tests for manifest order, live-coding slices, portable source path, and EPA load

`test_canonical_exports_and_disposable_database_are_valid` now checks both
structural validity and semantic adoption completeness without splitting
trivial assertions into separate tests. The full MVP sequence was executed from
the `notebooks/` working directory.
Plots used the non-interactive test backend; this verifies code execution, not
visual/UX acceptance.

## Safety

The canonical 12E.2 database was opened read-only. The only SQLite write target
was the explicitly guarded notebook demo database. Runtime fingerprints remain:

- `eco_drive.db`: `CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262`
- `eco_drive_qa.db`: `0EB4A5EC0F402E6B9E1F7D76372EA30068670B16F994D495394613DAFA2CA569`

No runtime cutover, UI change, schema redesign, raw-source overwrite, RAG,
external enrichment, or production physics modification occurred.
