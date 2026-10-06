# Sprint 12 Closure - Phase 1 Investigation

## Decision gate

`PHASE_1_COMPLETE = YES`

`PHASE_2_STARTED = NO`

The investigation found a material unresolved source-grain question for EPA Run identity. Production code, schema, ETL, candidate databases, and review workbooks were therefore left unchanged, as required by the Sprint 12 closure workflow.

The current candidate must **not** be promoted to PROD. It should not be regenerated until the Run-grain decision is made or explicitly quarantined as an accepted v1 limitation.

## Safety and scope

- Canonical candidate opened read-only with `mode=ro` and `PRAGMA query_only=ON`.
- Candidate SHA256 before and after the investigation:
  `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`.
- No database, schema, ETL, physics, resolver, identity, runtime, or workbook changes were made.
- Source inspected: `etl/data/raw/epa_testcar/epa_testcar_2026_raw.xlsx`, 30,194 rows for MY2020-MY2026.

## Root causes and findings

### 1. EPA model year is embedded in canonical identity - directly proven

`CONFIG_FIELDS` includes `Model Year`. The Vehicle Configuration candidate ID hashes all `CONFIG_FIELDS`; the VDE candidate ID then hashes that configuration ID plus Target ABC and ETW. Model year therefore manufactures a new Configuration and VDE even when every physical and test field is unchanged.

The source-wide exact-payload audit compared every EPA column except `Model Year`:

- 7,359 exact payload groups span more than one model year;
- 20,723 of 30,194 rows belong to those exact cross-year groups;
- some identical payloads span all seven model years.

This is not an isolated BMW issue.

#### BMW 330i reproduction

The BMW 330i has 21 rows across MY2020, MY2021, and MY2022. Seven Test Numbers repeat with every other source field identical across the three years:

`KBMX10055991`, `KBMX10055993`, `KBMX10056048`, `KBMX10056049`, `KBMX10056050`, `KBMX10056052`, and `KBMX10056054`.

The three configuration numbers remain legitimately distinct:

| Configuration | Approx. Target A | Disposition |
|---:|---:|---|
| 0 | 41.6 lbf | keep distinct |
| 1 | 41.5 lbf | keep distinct |
| 2 | 51.4 lbf | keep distinct |

The canonical candidate correctly has one consolidated BMW 330i Program, but creates separate year-specific Configurations/VDEs. For configuration 0, identical roadload/test evidence became VDE `-1001790` (2020), `-1010565` (2021), and `-1009385` (2022).

The same duplication propagates to FuelCons. Those three VDEs each materialize a separate result from adopted tests `KBMX10056048` + `KBMX10056049`, with identical values:

- FTP: `6.9180759705882355 L/100 km`
- HWFET: `4.514675297504798 L/100 km`
- combined: `5.836545667700689 L/100 km`
- CO2: `136.20338473315837 g/km`

**Conclusion:** model-year applicability is being mistaken for physical/test identity. Existing JSON provenance can preserve all repeated model years and source rows without requiring a new applicability table for the first correction.

### 2. A Vehicle Configuration can legitimately own multiple VDE roadloads - directly proven

Example: Audi R8 configuration `CFG-EPA-9A48E0A2FD5C36971F30`, MY2022, has three distinct VDEs and three distinct Target-ABC roadloads at the same mass:

- VDE `-1011238`: `242.873 / 1.320085 / 0.036856863`, 1814.369 kg
- VDE `-1009341`: `220.801 / 1.199989 / 0.033500930`, 1814.369 kg
- VDE `-1004235`: `220.810 / 1.200127 / 0.033507800`, 1814.369 kg

These must remain separate. Removing model year from identity must not collapse distinct Target ABC + ETW states.

### 3. Canonical VDE cycle is misleading - directly proven

The migration copies the first source row's `Test Category` into `vde.cycle_name`, although cycle is Run evidence.

Examples from the candidate:

| VDE | Vehicle | Stored `cycle_name` | Actual Run cycles |
|---:|---|---|---|
| -1004236 | Chevrolet Silverado 2WD MY2022 | SC03 | US06, SC03, HWY, FTP |
| -1005017 | Chevrolet Silverado 4WD MY2022 | FTP | SC03, HWY, FTP, US06 |
| -1001799 | Cadillac CT6 AWD MY2020 | US06 | FTP, HWY, US06, SC03 |

`vde.cycle_name` is therefore an arbitrary first-row value, not a property of the physical VDE. For EPA canonical rows it should not participate in identity and should be null/unused canonically. If application compatibility still requires a value, that behavior belongs in the compatibility view and must not be represented as a fake canonical cycle such as `EPA_STD`.

### 4. Cadillac repeated Test Number - cross-year carryover proven; within-year execution grain unresolved

Test Number `NGMX91004749` appears 15 times for Cadillac CT4 V:

- three rows in each model year 2022-2026;
- identical Test Vehicle ID `366MDN4388`;
- identical Test Group `NGMXV03.6043`;
- configuration 0, FTP, procedure 21;
- identical Target ABC, ETW, FE, CO2, and every result field;
- only Set ABC varies within a year.

The same three Set ABC triples repeat exactly in all five years:

1. `13.03 / 0.0705 / 0.02292`
2. `13.39 / -0.1672 / 0.02497`
3. `18.26 / -0.2513 / 0.02570`

Current ETL includes Set ABC in the execution key. When one broad Test Number has multiple execution keys, every variant is classified `DISTINCT_TEST_EXECUTION` with `MEDIUM` confidence. The candidate consequently contains five year-specific VDEs and three Runs per VDE (15 Runs).

Evidence classification:

- cross-year carryover/publication repetition: **directly proven**;
- source row is not automatically a unique test execution: **strongly supported**;
- the three Set ABC triples are three physical executions: **unresolved gap**;
- the three triples are detail/settings rows of one execution: **unresolved gap**.

No Run identity change is safe without an EPA data dictionary, source documentation, or an independent source field that explains the three settings.

### 5. FuelCons duplication is downstream, not an independent identity problem - strongly supported

BMW shows that separate year-manufactured VDEs produce separate but numerically identical FuelCons rows from the same adopted Test Numbers. This should be corrected only after Configuration/VDE/Run identity is corrected. No independent FuelCons deduplication should be introduced.

### 6. FTP+HWY with missing FuelCons is a conflict, not missing cycle evidence - directly proven

Ford F150 Super Crew Cab 4X4 MY2021 VDE `-1001285` has 20 Runs and both FTP and HWY evidence, but no FuelCons. The materialization report records two fuel-specific failures:

`CITY:MULTIPLE_CONFLICTING_ELIGIBLE_RUNS|HIGHWAY:MULTIPLE_CONFLICTING_ELIGIBLE_RUNS`.

This is correct conservative behavior under the present unresolved Run grain. It must not be bypassed by selecting an arbitrary eligible row.

### 7. Component/tire population is evidence-limited - directly proven

Legacy/canonical counts:

| Object | Legacy | Canonical | Decision |
|---|---:|---:|---|
| component catalog | 0 | 0 | do not invent reusable identities |
| component instances | n/a | 843 | retain source-scoped observations |
| component resolutions | n/a | 0 | acceptable for v1 |
| tire catalog | 1 (`tire_roadload_db`) | 1 (`tire_db`) | migrated/preserved |

The one canonical tire is the legacy Michelin Raptor `175/65R14` record. No component/tire catalog backfill is justified from descriptive JRC fields alone.

## Legacy -> canonical scalar mapping

| Legacy/source field | Canonical destination today | Disposition | Phase 1 decision |
|---|---|---|---|
| `engine_type` | `vehicle_configuration.engine_type` | KEEP_COLUMN | correct scalar ownership |
| `engine_model` / EPA `Engine Code` | `vehicle_configuration.engine_model` | KEEP_COLUMN | correct scalar ownership |
| `engine_size_l` / EPA displacement | `vehicle_configuration.engine_displacement_l` | RENAMED_COLUMN | correct scalar ownership |
| `engine_aspiration` | `vehicle_configuration.engine_aspiration` | KEEP_COLUMN | correct; sparse where source lacks it |
| `transmission_type` | `vehicle_configuration.transmission_type` | KEEP_COLUMN | correct scalar ownership |
| `transmission_model` | `vehicle_configuration.transmission_model` | KEEP_COLUMN | correct scalar ownership |
| `gear_count` | `vehicle_configuration.gear_count`; result compatibility also exists on FuelCons | KEEP_COLUMN | configuration scalar is populated |
| `final_drive_ratio` | `vehicle_configuration.final_drive_ratio`; result compatibility also exists on FuelCons | KEEP_COLUMN | configuration scalar is populated |
| `drive_type` | `vehicle_configuration.drive_system`; legacy-compatible `vde.drive_type` | RENAMED_COLUMN | canonical source should be Configuration |
| `electrification` | `fuelcons.electrification`; `vehicle_configuration.propulsion_architecture` | KEEP/RENAMED | preserve distinction between result label and architecture |
| EPA `Rated Horsepower` | `vehicle_configuration.architecture_properties_json.rated_horsepower` | JSON | scalar gap; 9,962 Configurations affected |
| EPA `# of Cylinders and Rotors` | `vehicle_configuration.architecture_properties_json.cylinders_rotors` | JSON | scalar gap; 8,641 Configurations affected |
| `engine_max_power_kw` | `fuelcons.engine_max_power_kw` | KEEP_COLUMN | existing field means effective/adopted result power; only 225 populated |
| component rated power | `component_db.rated_power_kw` | KEEP_COLUMN | valid only with reusable component identity |
| Test Group / Test Vehicle ID / configuration number | `vehicle_configuration.source_identity_json` | PROVENANCE | correct; these are source-scoped identifiers |
| Target ABC + ETW | `vde.coast_*` + `vde.test_mass_kg` | KEEP_COLUMN | correct physical VDE fields |
| Set ABC | `run.conditions_json.set_abc_native` | PROVENANCE | correct location, but identity role unresolved |
| Test Number | `run.conditions_json.test_number` + provenance | PROVENANCE | source identity evidence, not yet proven universal execution PK |
| test category/cycle | Run result/details; `vde.cycle_name` today | KEEP_COLUMN / OBSOLETE_CANONICAL_USE | cycle belongs to Run; VDE field compatibility-only |
| model year | `vde.year` and source provenance | KEEP_COLUMN / APPLICABILITY | retain value, remove it from physical/test identity |

### Scalar schema decision still required

`cylinders_rotors` is strongly supported as a first-class Vehicle Configuration scalar.

Rated horsepower also needs first-class scalar access, but the exact column contract must preserve the already documented distinction:

- `component_db.rated_power_kw`: reusable component rating;
- `fuelcons.engine_max_power_kw`: effective power adopted by a result.

EPA rated horsepower is a source Vehicle Configuration attribute. Reusing `fuelcons.engine_max_power_kw` would duplicate a stable configuration property across results and blur that distinction. The smallest semantically clean option is a Vehicle Configuration scalar such as `engine_rated_power_kw`, populated deterministically with `hp * 0.745699872`, while retaining raw hp in provenance. This is a recommendation, not an implemented schema change.

## Minimal Phase 2 proposal after the Run decision

1. Remove `Model Year` from EPA Vehicle Configuration identity.
2. Preserve all applicable model years and source rows in explicit provenance on the consolidated canonical records.
3. Keep Target ABC + ETW in VDE identity, so legitimate roadloads remain separate.
4. Make EPA canonical VDE cycle-independent; keep any required legacy presentation in the compatibility layer.
5. Leave the three Cadillac Set-ABC variants unchanged until the source grain is documented; add a QA warning/review report rather than guessing.
6. Let FuelCons rematerialization follow corrected VDE/Run identities; do not independently deduplicate it.
7. Add scalar columns only after approving the rated-power ownership/name.
8. Add review tabs only as candidate flags: `EPA_CARRYOVER_REVIEW`, `RUN_IDENTITY_REVIEW`, `VDE_DUPLICATE_CANDIDATES`, `FUELCONS_DUPLICATE_CANDIDATES`, and `JSON_SCALAR_REVIEW`.

## Reproducible evidence pointers

- `etl/scripts/sprint_12c2_canonical_population_preview.py:45-58` - current Configuration and Run field sets.
- `etl/scripts/sprint_12c2_canonical_population_preview.py:111-127` - current Program/Configuration/VDE/Run candidate hashing.
- `etl/scripts/sprint_12e_migration_rehearsal.py:401-420` - scalar population and the two attributes placed in JSON.
- `etl/scripts/sprint_12e_migration_rehearsal.py:425-455` - VDE construction and first-row Test Category assignment.
- `etl/scripts/sprint_12e2_epa_fuelcons_reconstruction.py:68-76` - result and condition fields, including Set ABC.
- `etl/scripts/sprint_12e2_epa_fuelcons_reconstruction.py:165-175` - execution key includes VDE, Set ABC, and result values.
- `etl/scripts/sprint_12e2_epa_fuelcons_reconstruction.py:236-256` - multiple execution keys under a broad Test Number become `DISTINCT_TEST_EXECUTION/MEDIUM`.
- `etl/data/processed/sprint_12e2_epa_fuelcons/run_grouping_results.csv:12` and subsequent `NGMX91004749` rows - Cadillac classification evidence.
- `etl/data/processed/sprint_12e2_epa_fuelcons/run_grouping_results.csv:11984`, `:18032`, `:21813` - the same BMW Test Number materialized once per model-year VDE.
- `etl/data/processed/sprint_12e2_epa_fuelcons/unresolved_materialization_cases.csv:2631-2632` - FTP+HWY evidence rejected because of multiple conflicting eligible Runs.
- `etl/reports/sprint_12d_physical_schema_design.md:57` - intended Vehicle Configuration scalar/JSON boundary.
- `etl/reports/sprint_12d_physical_schema_design.md:71` - documented rated component power versus adopted FuelCons power distinction.
- `etl/schema/canonical_schema_v1.sql:33-45`, `:75`, `:214`, and `:359` - current JSON, configuration, component power, VDE cycle, and FuelCons power schema.

## Verification status

No production tests were run because Phase 2 was not started and no executable project files were changed. All evidence in this report came from read-only source/SQLite inspection and existing ETL audit outputs.

## Promotion status

- Candidate regeneration recommended now: **NO - wait for Run-grain disposition and scalar contract approval**.
- Safe for additional manual review: **YES, as a known-defect candidate only**.
- Safe for PROD promotion: **NO**.
