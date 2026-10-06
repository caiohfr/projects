# Sprint 12C.1 — Dataset Delta & Engineering Coverage Audit

## Decision: `PAUSE_DDL — SOURCE_POPULATION_DECISION_REQUIRED`

Sprint 12C remains a successful compatibility proof. This audit answers the separate question of whether the new sources add better engineering evidence. The answer is **yes, but as complementary populations—not as a row-count replacement for the legacy 4,999**.

The runtime database was opened read-only and remained byte-identical. No DDL, migration, application, physics, resolver or raw-source file was changed.

## Executive findings

1. The legacy core is exactly **4,999 VDEs** derived from **26,262 EPA 2020–2025 source rows**. The notebook grouped 5,000 raw Make+Model+Year populations and discarded one group with no consumption evidence. Later text normalization collapsed these to 4,953 exact persisted MMY identities, leaving 46 two-row duplicate identity groups that must remain distinct until their VDE state differences are resolved.
2. The current database has **5,003 VDEs** because four later scenario/test descendants were added. All four have `vde_id_parent` and delta fields, yet inherited `record_origin=LEGACY` and empty source identity. The database also has five post-import FuelCons records: four attached to those VDEs and one explicit ML result attached to a legacy VDE.
3. Legacy Transmission/Brake ABC coverage is real as *population*, but not as observation: the notebook explicitly generated `*_est` values by splitting residual roadload with priors from `vde_defaults_by_category_trans_elec.csv`. Those fields must be classified as **ESTIMATED**, never SOURCE/MEASURED.
4. EPA MY2026 adds **3,901 source rows**, **3,557 Test Number+Testgroup candidates**, **1,291 strict configuration candidates**, **1,380 Target-ABC sets**, and **784 raw Make+Model+Year groups**. It is 100% complete for Target ABC, Set ABC, ETW, transmission, gears, axle ratio and N/V—but has no direct component ABC, tire, RRC or Cd/CdA. **19 rows** contain a four-digit year-like value in `Represented Test Veh Make`, so source identity must be validated rather than trusted blindly.
5. JRC adds only **249 rows**, but every row has WLTP mass, f0/f1/f2, gearbox/gears and a tire code. That is richer hardware context than EPA, although identity and row grain remain unresolved.
6. EEA contributes monitoring scale and fuel/energy/CO₂/range coverage, not VDE/configuration grain. `RLFI` remains unresolved and is not counted as roadload ABC.

## Population and grain

| population | source_rows | test_candidates | vde_or_reporting_rows | vehicle_or_mmy_groups | years | grain |
|---|---|---|---|---|---|---|
| Legacy EPA source 2020-2025 | 26262 | UNRESOLVED | 5000 | 5000 | 2020-2025 | Source tests aggregated to Make+Model+Year |
| Current legacy core in SQLite | 26262 | COLLAPSED | 4999 | 4953 | 2020-2025 | One averaged VDE per retained Make+Model+Year |
| Current post-import VDE additions | N/A | N/A | 4 | 4 | 2024,2026,2027 | User/test scenario descendants with vde_id_parent |
| Current post-import FuelCons additions | N/A | N/A | 5 | 4 | N/A | Scenario/regression/ML result |
| EPA Test Car MY2026 | 3901 | 3557 | 1380 | 784 | 2026 | Source row/test; canonical VDE grain still requires an explicit rule |
| JRC technical dataset | 249 | 249 | 249 | UNRESOLVED | No explicit model year | Anonymized vehicle/archetype/simulation row; unresolved |
| EEA 2025 provisional | 10833597 | N/A | 10833597 | REPORTING GRAIN | 2025 | Registration/monitoring row; not a VDE/configuration row |

The refreshed EPA workbook is not only a 2026 append: its 2020–2025 projection has 26,293 rows, a net **+31** versus the old workbook, and 5,014 raw MMY groups versus 5,000. At the shared 32-column projection, some historical rows also differ; therefore a future canonical reload needs a versioned refresh policy rather than a blind append.

### Duplicate persisted legacy identities

There are **46** exact Make+Model+Year groups with two VDE rows each. These are not safe deduplication candidates: `44` groups contain multiple Target-ABC states.

| make | model | year | vde_rows | vde_ids | target_abc_variants | interpretation |
|---|---|---|---|---|---|---|
| BENTLEY | BENTAYGA | 2024 | 2 | 1719;1720 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| BENTLEY | BENTAYGA | 2025 | 2 | 811;812 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| BMW | 230I COUPE | 2020 | 2 | 4237;4238 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| BMW | 230I COUPE | 2021 | 2 | 3468;3469 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| BMW | M240I CONVERTIBLE | 2021 | 2 | 3501;3502 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| FORD | BRONCO SPORT SASQUATCH | 2025 | 2 | 166;167 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| FORD | INTERCEPTOR UTILITY AWD | 2025 | 2 | 190;191 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| FORD | MUSTANG | 2024 | 2 | 1072;1073 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| FORD | MUSTANG | 2025 | 2 | 195;196 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| FORD | RANGER 4WD | 2025 | 2 | 204;205 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HONDA | CIVIC 4DR | 2025 | 2 | 295;296 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HONDA | CIVIC 5DR | 2020 | 2 | 4520;4521 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HONDA | CIVIC 5DR | 2021 | 2 | 3741;3742 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HONDA | CIVIC 5DR | 2023 | 2 | 2066;2067 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HONDA | CIVIC 5DR | 2024 | 2 | 1174;1175 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HYUNDAI | G80 | 2020 | 2 | 4551;4552 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HYUNDAI | KONA | 2020 | 2 | 4555;4556 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HYUNDAI | KONA | 2021 | 2 | 3767;3768 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HYUNDAI | KONA | 2024 | 2 | 1210;1211 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HYUNDAI | KONA | 2025 | 2 | 334;335 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HYUNDAI | SONATA | 2021 | 2 | 3774;3775 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HYUNDAI | SONATA | 2022 | 2 | 2974;2975 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| HYUNDAI | SONATA | 2023 | 2 | 2109;2110 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| JAGUAR | F-PACE | 2020 | 2 | 4569;4570 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| JEEP | WRANGLER UNLIMITED 4X4 | 2020 | 2 | 4378;4379 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| JEEP | WRANGLER UNLIMITED 4X4 | 2021 | 2 | 3602;3603 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| JEEP | WRANGLER UNLIMITED 4X4 | 2022 | 2 | 2784;2785 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| JEEP | WRANGLER UNLIMITED 4X4 | 2023 | 2 | 1912;1913 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| LAND ROVER | DEFENDER | 2020 | 2 | 4583;4584 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| LAND ROVER | DEFENDER | 2021 | 2 | 3793;3794 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| LAND ROVER | DEFENDER | 2022 | 2 | 2993;2994 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| LAND ROVER | DEFENDER 110 MHEV | 2025 | 2 | 355;356 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| MAZDA | MAZDA2 | 2020 | 2 | 4630;4631 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| MAZDA | MAZDA2 | 2021 | 2 | 3835;3836 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| MAZDA | MAZDA3 | 2021 | 2 | 3837;3838 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| MAZDA | MAZDA3 | 2022 | 2 | 3046;3047 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| MAZDA | MAZDA3 | 2023 | 2 | 2185;2186 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| NISSAN | ARMADA 4WD PLATINUM | 2023 | 2 | 2319;2320 | 1 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| NISSAN | SENTRA SR | 2021 | 2 | 3983;3984 | 1 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| PORSCHE | MACAN 4 ELECTRIC | 2024 | 2 | 1506;1507 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| PORSCHE | MACAN 4 ELECTRIC | 2025 | 2 | 602;603 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| PORSCHE | MACAN TURBO ELECTRIC | 2024 | 2 | 1510;1511 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| PORSCHE | MACAN TURBO ELECTRIC | 2025 | 2 | 608;609 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| VOLKSWAGEN | ATLAS | 2021 | 2 | 4196;4197 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| VOLKSWAGEN | TAOS | 2022 | 2 | 3430;3431 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |
| VOLKSWAGEN | TAOS | 2023 | 2 | 2616;2617 | 2 | Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved. |

## Engineering coverage and provenance

| Information | Legacy core (4,999 VDE) | EPA Test Car MY2026 | JRC technical dataset | EEA 2025 provisional |
|---|---|---|---|---|
| Whole-vehicle/Target ABC | 100.0% [SOURCE_LIKELY_AGGREGATED] | 100.0% [DIRECT_SOURCE] | 100.0% [DIRECT_LABELS_SEMANTICS_PARTIAL] | 0.0% [ABSENT_OR_UNRESOLVED] |
| Set ABC | 0.0% [ABSENT] | 100.0% [DIRECT_SOURCE] | 0.0% [ABSENT] | 0.0% [ABSENT_OR_UNRESOLVED] |
| ETW / test mass | 100.0% [DETERMINISTIC_FROM_SOURCE_ETW] | 100.0% [DIRECT_SOURCE] | 100.0% [DIRECT_SOURCE] | 100.0% [DIRECT_LABEL_VALUE_SEMANTICS_PARTIAL] |
| Transmission identity | 100.0% [SOURCE_NORMALIZED_WITH_FALLBACK] | 100.0% [DIRECT_SOURCE] | 100.0% [DIRECT_SOURCE] | 0.0% [ABSENT_OR_UNRESOLVED] |
| Gear count | 100.0% [SOURCE_LIKELY_AGGREGATED] | 100.0% [DIRECT_SOURCE] | 100.0% [DIRECT_SOURCE] | 0.0% [ABSENT_OR_UNRESOLVED] |
| Axle / final-drive ratio | 100.0% [SOURCE_LIKELY_AGGREGATED] | 100.0% [DIRECT_SOURCE] | 0.0% [ABSENT] | 0.0% [ABSENT_OR_UNRESOLVED] |
| N/V ratio | 0.0% [DROPPED_BY_LEGACY_ETL] | 100.0% [DIRECT_SOURCE] | 0.0% [ABSENT] | 0.0% [ABSENT_OR_UNRESOLVED] |
| Transmission loss ABC | 59.2% [ESTIMATED] | 0.0% [ABSENT] | 0.0% [ABSENT] | 0.0% [ABSENT_OR_UNRESOLVED] |
| Brake ABC | 59.2% [ESTIMATED] | 0.0% [ABSENT] | 0.0% [ABSENT] | 0.0% [ABSENT_OR_UNRESOLVED] |
| Tire RRC | 59.2% [BACKCALCULATED_BOUNDED_ESTIMATE] | 0.0% [ABSENT] | 0.0% [ABSENT] | 0.0% [ABSENT_OR_UNRESOLVED] |
| Tire identity | 0.0% [SPARSE_UNRESOLVED] | 0.0% [ABSENT] | 100.0% [DIRECT_SOURCE_PARTIAL_IDENTITY] | 0.0% [ABSENT_OR_UNRESOLVED] |
| Cd / CdA | 0.0% [SPARSE_OR_SCENARIO] | 0.0% [ABSENT] | 0.0% [ABSENT] | 0.0% [ABSENT_OR_UNRESOLVED] |
| Fuel / CO2 | 97.2% [DETERMINISTIC_FROM_SOURCE_TEST_RESULTS] | 87.0% [DIRECT_SOURCE] | 100.0% [DIRECT_SOURCE] | 99.9% [DIRECT_LABEL_VALUE_SEMANTICS_PARTIAL] |
| BEV energy | 12.9% [DERIVED_FROM_SOURCE_ELECTRIC_TEST_RESULTS] | 9.6% [DIRECT_SOURCE_WITH_UNIT_CONTEXT] | 29.3% [DIRECT_SOURCE] | 27.2% [DIRECT_LABEL_VALUE_SEMANTICS_PARTIAL] |
| Range | 0.0% [ABSENT] | 0.0% [ABSENT] | 25.3% [DIRECT_SOURCE] | 27.0% [DIRECT_SOURCE] |

Percentages are row-population coverage, not proof of equal grain or semantic equivalence. Bracketed labels describe provenance quality. Detailed counts and evidence are in `engineering_coverage.csv`.

## Overlap

| comparison | grain | left_count | right_count | overlap | right_only | interpretation |
|---|---|---|---|---|---|---|
| Legacy core vs EPA2026 | normalized Make+Model+Year | 4950 | 777 | 0 | 777 | No exact overlap is expected because EPA2026 adds a new model year. |
| Legacy core vs EPA2026 | normalized Make+Model, ignoring year | 1729 | 777 | 652 | 125 | Lexical continuity only; this does not prove same Program or Configuration. |
| Old workbook vs refreshed EPA file (2020-2025) | exact shared-column row hash | 26030 | 26061 | 25906 | 155 | Shared 32-column projection; detects source refresh deltas without assigning cause. |
| Old workbook vs refreshed EPA file (2020-2025) | source row count | 26262 | 26293 | N/A | 31 | The refreshed source has a net +31 historical rows, all in MY2025. |
| Legacy core vs EEA 2025 | normalized lexical Mk+Cn+year | 4950 | 7814 | 211 | 7603 | Lexical overlap only; EEA is reporting-grain, not VDE/configuration-grain. 1095 row(s) lacked a usable lexical key. |
| Legacy core vs JRC | identity | 4950 | 249 | UNRESOLVED | UNRESOLVED | JRC OEM/model are anonymized; no defensible identity overlap can be asserted. |

JRC overlap is not asserted because OEM/model are anonymized. EEA overlap is lexical only and must not be promoted to a canonical Program/Configuration match without a source-specific rule.

## Post-import additions

| vde_id | created_at | make | model | year | parent_vde_id | fuelcons_rows | record_origin | classification |
|---|---|---|---|---|---|---|---|---|
| 5031 | 2025-10-14 22:16:09 | TOYOTA | MOCK_VEH1 | 2024 | 4996 | 2 | LEGACY | POST_IMPORT_SCENARIO_OR_TEST_DERIVATIVE |
| 5033 | 2025-10-17 21:22:22 | AUDI | MOCK_VEH1 | 2024 | 4947 | 1 | LEGACY | POST_IMPORT_SCENARIO_OR_TEST_DERIVATIVE |
| 5034 | 2025-10-21 04:04:56 | FERRARI | MOCK_VEH | 2026 | 5033 | 1 | LEGACY | POST_IMPORT_SCENARIO_OR_TEST_DERIVATIVE |
| 5038 | 2026-06-08 14:35:38 | AUDI | TEST08062026 | 2027 | 4984 | 0 | LEGACY | POST_IMPORT_SCENARIO_OR_TEST_DERIVATIVE |
| 4861 | 2026-06-26 16:11:37 |  |  |  |  | 1 | LEGACY | POST_IMPORT_ML_FUELCONS_ON_LEGACY_VDE |

The first four rows are the additional VDEs; the last line is the fifth later FuelCons result, an ML prediction attached to legacy VDE 4861. They should be retained as scenario/evidence history, but their origin must be corrected during migration from generic `LEGACY` to explicit scenario/test/ML provenance. This audit did not edit them.

## Engineering usability L0–L4

| population | level | rows | population_rows | coverage_pct | audit_definition |
|---|---|---|---|---|---|
| Legacy core (4,999 VDE) | L0 | 0 | 4999 | 0.0 | Identity/reporting only; no usable complete whole-vehicle roadload state. |
| Legacy core (4,999 VDE) | L1 | 0 | 4999 | 0.0 | Complete whole-vehicle roadload plus mass. |
| Legacy core (4,999 VDE) | L2 | 2041 | 4999 | 40.828 | L1 plus at least two useful hardware descriptors. |
| Legacy core (4,999 VDE) | L3 | 2958 | 4999 | 59.172 | L2 plus partial component descriptor/build-up support; provenance may still be estimated. |
| Legacy core (4,999 VDE) | L4 | 0 | 4999 | 0.0 | Component-rich and scenario-ready with strong multi-domain provenance. |
| EPA Test Car MY2026 | L0 | 0 | 3901 | 0.0 | Identity/reporting only; no usable complete whole-vehicle roadload state. |
| EPA Test Car MY2026 | L1 | 0 | 3901 | 0.0 | Complete whole-vehicle roadload plus mass. |
| EPA Test Car MY2026 | L2 | 3901 | 3901 | 100.0 | L1 plus at least two useful hardware descriptors. |
| EPA Test Car MY2026 | L3 | 0 | 3901 | 0.0 | L2 plus partial component descriptor/build-up support; provenance may still be estimated. |
| EPA Test Car MY2026 | L4 | 0 | 3901 | 0.0 | Component-rich and scenario-ready with strong multi-domain provenance. |
| JRC technical dataset | L0 | 0 | 249 | 0.0 | Identity/reporting only; no usable complete whole-vehicle roadload state. |
| JRC technical dataset | L1 | 0 | 249 | 0.0 | Complete whole-vehicle roadload plus mass. |
| JRC technical dataset | L2 | 0 | 249 | 0.0 | L1 plus at least two useful hardware descriptors. |
| JRC technical dataset | L3 | 249 | 249 | 100.0 | L2 plus partial component descriptor/build-up support; provenance may still be estimated. |
| JRC technical dataset | L4 | 0 | 249 | 0.0 | Component-rich and scenario-ready with strong multi-domain provenance. |
| EEA 2025 provisional | L0 | 10833597 | 10833597 | 100.0 | Identity/reporting only; no usable complete whole-vehicle roadload state. |
| EEA 2025 provisional | L1 | 0 | 10833597 | 0.0 | Complete whole-vehicle roadload plus mass. |
| EEA 2025 provisional | L2 | 0 | 10833597 | 0.0 | L1 plus at least two useful hardware descriptors. |
| EEA 2025 provisional | L3 | 0 | 10833597 | 0.0 | L2 plus partial component descriptor/build-up support; provenance may still be estimated. |
| EEA 2025 provisional | L4 | 0 | 10833597 | 0.0 | Component-rich and scenario-ready with strong multi-domain provenance. |

This is an audit-only metric, not a proposed database taxonomy. Legacy L3 means partial build-up is available, but mainly as estimated/back-calculated evidence; JRC L3 means direct hardware descriptors with unresolved row identity. Neither is L4.

## What was actually gained

- **EPA2026:** a new model year, source-row/test lineage, multiple VDE states per commercial vehicle, Set ABC, and complete observed configuration descriptors. It improves provenance and grain even though direct component decomposition is absent.
- **JRC:** direct tire code plus gearbox, gears, mass and WLTP/real-world roadload/result context. It improves descriptor richness, but does not replace identifiable EPA history.
- **EEA:** very large 2025 monitoring coverage for mass, consumption, CO₂, electric energy and range. It supports homologation/monitoring comparisons, not engineering VDE replacement.
- **Legacy ETL:** remains valuable for 2020–2025 historical coverage and deterministic application-ready VDE/FuelCons results. Its component ABC/RRC estimates must be retained with corrected provenance, not mistaken for measured values.

## Canonical source population decision recommended before 12D

1. Retain/re-ingest the 4,999 legacy EPA MMY population as historical 2020–2025 coverage, with source file/version and derivation lineage reconstructed.
2. Ingest EPA2026 at source RUN grain; derive Configuration and VDE state only through an approved key. Do not repeat the old Make+Model+Year averaging collapse.
3. Preserve legacy component decomposition as ESTIMATION/CALCULATION evidence, never as direct EPA source data.
4. Keep the four scenario VDE descendants and five later FuelCons records in a separate scenario/ML evidence class.
5. Keep JRC source-scoped and `PARTIAL/UNRESOLVED`; use its direct descriptors without cross-source identity merging.
6. Treat EEA as monitoring/declared-result evidence. Do not create roadload from RLFI until its semantics are approved.

Once these six population rules are approved, Sprint 12D can design physical DDL against the correct ingestion grains.

## Reproduction

```powershell
python etl/scripts/sprint_12c1_dataset_delta_audit.py
```

The first run streams the local EEA CSV to compute a conservative lexical overlap; all source/database reads are read-only.

## Outputs

- `etl/data/processed/sprint_12c1_dataset_delta_audit/population_summary.csv`
- `etl/data/processed/sprint_12c1_dataset_delta_audit/engineering_coverage.csv`
- `etl/data/processed/sprint_12c1_dataset_delta_audit/provenance_quality.csv`
- `etl/data/processed/sprint_12c1_dataset_delta_audit/identity_overlap.csv`
- `etl/data/processed/sprint_12c1_dataset_delta_audit/legacy_extra_records.csv`
- `etl/data/processed/sprint_12c1_dataset_delta_audit/legacy_duplicate_groups.csv`
- `etl/data/processed/sprint_12c1_dataset_delta_audit/engineering_usability.csv`
- `etl/data/processed/sprint_12c1_dataset_delta_audit/audit_results.json`
- `etl/reports/sprint_12c1_dataset_delta_audit.md`
