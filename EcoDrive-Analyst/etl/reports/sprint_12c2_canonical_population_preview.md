# Sprint 12C.2 — Canonical Population Preview

## Status: `POPULATION_SHAPE_CLEAR — PROCEED_TO_NOTEBOOKS_AND_12D`

The future engineering database is likely to remain in the **thousands to low tens of thousands**, not millions. EEA can add millions of RUN/FuelCons reporting records, but it does not imply millions of Programs, Configurations or VDEs.

This is a read-only, pre-DDL grouping preview. Candidate identifiers are deterministic audit labels only. The runtime database remained byte-identical; no schema, migration, application, physics, resolver or raw source changed.

## Population funnel

```text
SOURCE ROWS → RUN candidates → VDE states → VEHICLE_CONFIGURATION → PROGRAM → FUELCONS
```

| source_population | source_rows | run_candidates | vde_candidates | vehicle_configuration_candidates | program_candidates | fuelcons_candidates | confidence |
|---|---|---|---|---|---|---|---|
| Legacy EPA 2020-2025 source | 26262 | 4,999-26,262 | 4,999 | 4996 | 4953 | 4,999 | HIGH for retained VDE/FuelCons; PARTIAL for RUN |
| EPA refreshed historical 2020-2025 | 26293 | 14,522-26,293 | 9,939 | 8,718 | 4,965 | 9,939-26,293 | PROVISIONAL |
| EPA Test Car MY2026 | 3901 | 3,557-3,901 | 1,485-3,901 | 1,291-3,901 | 777-784 | 1,485-3,901 | PROVISIONAL |
| Current post-import scenario/ML | N/A | 5 | 4 | 0 additive | 0 additive | 5 | HIGH |
| JRC technical dataset | 249 | 249 | 249 | 249 | 249 | 249 | PARTIAL/UNRESOLVED |
| EEA 2025 provisional | 10,833,597 | UNRESOLVED-10,833,597 reporting | 0 engineering | 0 engineering | 0 engineering | UNRESOLVED-10,833,597 reporting | PARTIAL |

Full grain rules and unresolved issues are retained in `population_funnel.csv`.

## EPA MY2026 preview

The 3,901 source rows produce **777–784 Program candidates**, **1,291–3,901 Configuration candidates**, **1,485–3,901 VDE candidates**, and **3,557–3,901 RUN candidates** under the current provisional rules.

The lower preview groups Program by source-scoped normalized MMY, Configuration by the existing strict technical/source signature, VDE by Configuration+Target ABC+ETW, and RUN by Test Number+Testgroup. None of those keys is promoted to universal canonical identity.

Observed topology: **194** candidate Configurations have multiple VDE states; **1,104** VDE candidates have multiple RUN candidates; **1** RUN candidates touch multiple preview VDE states and remain unresolved.

## Nine-entity forecast

| entity | legacy_backed_minimum | public_engineering_additive_minimum | likely_engineering_count_or_range | engineering_upper_bound | eea_reporting_additive_range | reason |
|---|---|---|---|---|---|---|
| PROGRAM | 4953 | 1026 | 5,979-5,986 | 5986 | 0 | Legacy source-scoped MMY + EPA2026 preview Programs + one JRC source-scoped Program per unresolved row. |
| VEHICLE_CONFIGURATION | 4996 | 1540 | 6,536-9,146 | 9146 | 0 | Strict EPA signature is the likely lower grouping; source rows are the safe upper bound. |
| COMPONENT_DB | 0 | 0 | 0-345 | 345 | 0 | Reusable component identity is absent; JRC descriptor rows are an upper source-scoped candidate bound. |
| TIRE_DB | 1 | 70 | 71-250 | 250 | 0 | 70 distinct JRC tire codes are likely provisional definitions; one per row is the unresolved upper bound. |
| COMPONENT_INSTANCE | 0 | 0 | 0-4,740 likely if partial instances are adopted | 12570 | 0 | Optional partial engine/transmission/driveline/tire/electric instances; never required for VDE validity. |
| COMPONENT_RESOLUTION | 0 | 0 | 0-2,958 | 2958 | 0 | Legacy estimated decomposition may be retained as optional resolution evidence; public Tier-0 adds none. |
| VDE | 5003 | 1734 | 6,737-9,153 | 9153 | 0 | Legacy states preserved; EPA VC+Target+ETW preview lower bound; one EPA VDE per source row upper bound; JRC source-scoped. |
| RUN | 5004 | 3806 | 8,810-30,448 | 30448 | 0-10,833,597 | Lower uses current result lineage + EPA test candidates + JRC. Upper also preserves refreshed historical source rows. EEA reporting dominates separately. |
| FUELCONS | 5004 | 1734 | 6,738-9,154 | 9154 | UNRESOLVED-10,833,597 | Engineering adopted-result count depends on EPA result adoption; EEA remains reporting-grain and separate. |

The two growth drivers are different: EPA/JRC expand engineering state/evidence into thousands or low tens of thousands; EEA expands reporting evidence into millions only if row-level monitoring is retained.

## Legacy-to-future representation

| legacy_population | future_entity | future_rows_or_range | mapping_rule | provenance | status |
|---|---|---|---|---|---|
| 4,999 historical VDE | PROGRAM | 4953 | Source-scoped exact persisted MMY fallback; no cross-year merge without evidence. | LEGACY_EPA_RECONSTRUCTED | PROVISIONAL |
| 4,999 historical VDE | VEHICLE_CONFIGURATION | 4,996 | Sprint 12C stable legacy technical signature; target ABC/mass excluded from configuration identity. | LEGACY_EPA_RECONSTRUCTED | PROVISIONAL |
| 4,999 historical VDE | VDE | 4999 | Preserve every historical VDE, including 46 duplicate MMY identity groups. | SOURCE_LIKELY_AGGREGATED | KEEP_SEPARATE |
| 4,999 historical FuelCons | FUELCONS | 4999 | One retained adopted comparison result per legacy VDE. | DETERMINISTIC_FROM_SOURCE_TEST_RESULTS | PRESERVE |
| Legacy source evidence | RUN | 4,999-26,262 | Minimum one adopted-result lineage per VDE; upper bound preserves each source row as evidence. | SOURCE_ROW_ID_UNAVAILABLE_IN_DB | PARTIAL |
| 2,958 component decompositions | COMPONENT_RESOLUTION | 0-2,958 | Optional estimated/calculated resolution evidence; never measured EPA component ABC. | ESTIMATED_DEFAULT_PRIOR_SPLIT | REVIEW_REQUIRED |
| 4 derived/scenario VDE | VDE | 4 | Retain as descendants of parent configuration/VDE; no new Program required. | SCENARIO_OR_TEST_DERIVATIVE | PRESERVE |
| 5 later FuelCons | RUN + FUELCONS | 5 | Retain scenario/regression/ML evidence and adopted result separately. | SCENARIO/ML; CURRENT LABEL NEEDS CORRECTION | PRESERVE |
| Sparse tire record | TIRE_DB | 1 | Preserve current specialized tire evidence; do not fabricate instances. | LEGACY_TIRE_EVIDENCE | PRESERVE |

The 46 duplicated legacy MMY identities remain represented as separate VDE states. The 44 groups with different Target ABC are classified `KEEP_SEPARATE`; the other two remain `UNRESOLVED`, not auto-merged.

## Grouping examples

The audit generated **24 deterministic real examples**. A compact selection follows; the full trees are in `etl/reports/sprint_12c2_population_tree_examples.md`.

| example_id | category | population | source_key | evidence |
|---|---|---|---|---|
| EX-01 | ONE_MMY_ONE_CONFIG_ONE_VDE | EPA Test Car MY2026 | PREVIEW-VDE-59AA1262BFAECE67 | source_rows=1; run_candidates=1; configurations_in_program=1; vde_states_in_configuration=1 |
| EX-02 | ONE_MMY_ONE_CONFIG_ONE_VDE | EPA Test Car MY2026 | PREVIEW-VDE-6881E7204A68D2ED | source_rows=1; run_candidates=1; configurations_in_program=1; vde_states_in_configuration=1 |
| EX-03 | ONE_MMY_ONE_CONFIG_ONE_VDE | EPA Test Car MY2026 | PREVIEW-VDE-741F2F10D823273C | source_rows=1; run_candidates=1; configurations_in_program=1; vde_states_in_configuration=1 |
| EX-04 | MMY_MULTIPLE_CONFIGS | EPA Test Car MY2026 | PREVIEW-VDE-CBA15350F27B06CE | source_rows=2; run_candidates=2; configurations_in_program=3; vde_states_in_configuration=1 |
| EX-05 | MMY_MULTIPLE_CONFIGS | EPA Test Car MY2026 | PREVIEW-VDE-C5C21F170A39BAF8 | source_rows=1; run_candidates=1; configurations_in_program=2; vde_states_in_configuration=1 |
| EX-06 | MMY_MULTIPLE_CONFIGS | EPA Test Car MY2026 | PREVIEW-VDE-2365EDD0D5A74155 | source_rows=1; run_candidates=1; configurations_in_program=2; vde_states_in_configuration=1 |
| EX-07 | MMY_MULTIPLE_CONFIGS | EPA Test Car MY2026 | PREVIEW-VDE-6675866BCE5607A3 | source_rows=1; run_candidates=1; configurations_in_program=2; vde_states_in_configuration=1 |
| EX-08 | CONFIG_MULTIPLE_VDE_STATES | EPA Test Car MY2026 | PREVIEW-VDE-2721830AED834036 | source_rows=4; run_candidates=4; configurations_in_program=1; vde_states_in_configuration=2 |

## Collision and ambiguity summary

| classification | cases |
|---|---|
| KEEP_SEPARATE | 238 |
| PROVISIONAL_GROUP | 1104 |
| SAFE_GROUP | 1 |
| SOURCE_DATA_QUALITY_ISSUE | 4 |
| UNRESOLVED | 3 |

`collision_report.csv` includes every legacy duplicated MMY, suspicious EPA Make value, source-ID/descriptor conflict, multi-state configuration, multi-RUN VDE, RUN-to-multiple-VDE collision and historical refresh hash delta.

## Historical refresh

| slice | metric | old_count | refreshed_count | delta | status | evidence |
|---|---|---|---|---|---|---|
| 2020-2025 | source_rows | 26262 | 26293 | 31 | REFRESH_CHANGED | Shared model-year slice. |
| 2020-2025 | unique_shared_projection_hashes | 26030 | 26061 | 31 | REFRESH_CHANGED | overlap=25906; old_only=124; refreshed_only=155 |
| 2020-2025 | raw_mmy_groups | 5000 | 5014 | 14 | REFRESH_CHANGED | Exact raw Make+Model+Year tuples. |
| 2020 | source_rows | 4450 | 4450 | 0 | UNCHANGED | Model Year row count. |
| 2021 | source_rows | 4265 | 4265 | 0 | UNCHANGED | Model Year row count. |
| 2022 | source_rows | 4493 | 4493 | 0 | UNCHANGED | Model Year row count. |
| 2023 | source_rows | 4521 | 4521 | 0 | UNCHANGED | Model Year row count. |
| 2024 | source_rows | 4277 | 4277 | 0 | UNCHANGED | Model Year row count. |
| 2025 | source_rows | 4256 | 4287 | 31 | REFRESH_CHANGED | Model Year row count. |

The refreshed file is not a pure 2026 append. MY2025 gains 31 source rows and the 2020–2025 shared projection contains both removed/changed and new/changed row hashes. A versioned supersede/refresh rule is required.

## Evidence tiers and remaining gaps

- **Directly tested:** read-only/byte-identical database access; deterministic EPA counts and example selection; all 46 duplicate legacy groups retained; NULL distinct from zero; output scope limited to `etl/` and sprint documentation.
- **Indirectly covered:** exact 12C reconstruction of the current VDE/FuelCons application surface and 12C.1 provenance classification.
- **Inspection-supported:** legacy notebook grouping/decomposition logic and CDR source-scoped identity guardrails.
- **Gap:** final EPA configuration/adoption key, exact JRC grain, EEA aggregation/adoption key and EEA RLFI semantics remain unresolved by design.

## Recommendation

Population shape is sufficiently visible to start notebooks and physical DDL design using ranges and explicit unresolved states. DDL implementation/migration remains unauthorized in this sprint. The 12D design must not encode provisional preview keys as universal identities.

## Reproduction

```powershell
python etl/scripts/sprint_12c2_canonical_population_preview.py
python -m unittest discover -s etl/tests -p "test_sprint_12c2*.py" -v
```

## Outputs

- `etl/data/processed/sprint_12c2_canonical_population_preview/population_funnel.csv`
- `etl/data/processed/sprint_12c2_canonical_population_preview/entity_count_forecast.csv`
- `etl/data/processed/sprint_12c2_canonical_population_preview/epa2026_grouping_preview.csv`
- `etl/data/processed/sprint_12c2_canonical_population_preview/legacy_future_mapping.csv`
- `etl/data/processed/sprint_12c2_canonical_population_preview/population_examples.csv`
- `etl/data/processed/sprint_12c2_canonical_population_preview/collision_report.csv`
- `etl/data/processed/sprint_12c2_canonical_population_preview/historical_refresh_delta.csv`
- `etl/data/processed/sprint_12c2_canonical_population_preview/population_preview.json`
- `etl/reports/sprint_12c2_canonical_population_preview.md`
- `etl/reports/sprint_12c2_population_tree_examples.md`
