# Sprint 12C.3 — Program Consolidation Review

## Status: `PROGRAM_SHAPE_CLEAR — PROCEED_TO_12D`

The previous ~6k forecast was conservative because it treated source-scoped Make+Model+Model-Year fallbacks as Programs. This review keeps those fallbacks traceable while measuring only evidence-supported cross-year parent consolidation.

No production Program identity was assigned. No DDL, migration, runtime/application code, raw source or runtime database changed.

## Decision

For combined EPA 2020–2026, the raw MMY fallback upper bound is **5,798**, deterministic normalization plus safe consolidation yields **2,895**, the likely semantic range is **2,214-2,895**, and the aggressive name-only floor is **1,860**.

Only `SAFE_CONSOLIDATE` is deterministic. `PROBABLE_CONSOLIDATE` informs the lower end of the likely range but remains review work. The aggressive lexical count is never canonical because one commercial name can span multiple generations.

## Program population forecast

| population | source_rows | fallback_upper_bound | normalized_mmy_fallbacks | safe_consolidation_count | likely_semantic_program_range | aggressive_lexical_lower_bound | engineering_program_population |
|---|---|---|---|---|---|---|---|
| EPA refreshed 2020-2025 | 26293 | 5014 | 4965 | 2621 | 2,044-2,621 | 1739 | YES |
| EPA 2026 | 3901 | 784 | 777 | 777 | 777-777 | 777 | YES |
| Combined EPA 2020-2026 | 30194 | 5798 | 5742 | 2895 | 2,214-2,895 | 1860 | YES |
| JRC source-scoped | 249 | 249 | 249 | 249 | 249-249 | 135 | SOURCE_SCOPED_UNRESOLVED |
| EEA 2025 monitoring | 10833597 | 0 | 0 | 0 | 0-0 | 0 | NO |

JRC remains source-scoped because identities are anonymized. EEA contributes zero engineering Programs; its 7,814 lexical reporting groups remain monitoring identities only.

## Boundary evidence

| boundary_status | cases |
|---|---|
| NO_BOUNDARY_EVIDENCE | 3513 |
| STRONG_BOUNDARY_CANDIDATE | 53 |
| UNRESOLVED | 38 |
| WEAK_BOUNDARY_CANDIDATE | 278 |

No source-only case is labelled `CONFIRMED_BOUNDARY`: OEM generation/platform evidence is absent. Strong candidates require a populated multi-domain technical reset; Target/Set ABC, ETW and test procedure never create a boundary by themselves.

## Consolidation candidates

| consolidation_status | groups |
|---|---|
| KEEP_SEPARATE | 213 |
| PROBABLE_CONSOLIDATE | 502 |
| SAFE_CONSOLIDATE | 961 |
| UNRESOLVED | 538 |

Safe cross-year edges require contiguous years, an exact stable-architecture signature, and persistent EPA Test Vehicle ID/Configuration identity. Same name or year adjacency alone is insufficient.

## Relationship impact preview

| fallback_programs_before | safe_programs_after | likely_programs_after_range | vehicle_configurations_before | vehicle_configurations_after_parent_remap | vde_states_before | vde_states_after_parent_remap | run_candidates_before | run_candidates_after_parent_remap | fuelcons_candidate_floor_before | fuelcons_candidate_floor_after_parent_remap | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5742 | 2895 | 2,214-2,895 | 10009 | 10009 | 11424 | 11424 | 15884 | 15884 | 11424 | 11424 | PASS_NO_CHILD_ROW_LOSS |

Program consolidation changes only the parent identity layer. Every fallback is assigned exactly once, and all Configuration, VDE, RUN and FuelCons candidate identities remain untouched.

```text
multiple Model Years
multiple Configurations
multiple VDEs
multiple RUNs
        ↓
may still belong to one Program
```

## Real examples

The review generated **31** deterministic examples. The full chronological trees are in `etl/reports/sprint_12c3_program_examples.md`.

| example_id | category | make | model | model_years | evidence |
|---|---|---|---|---|---|
| EX-01 | STABLE_ARCHITECTURE_2020_2026 | ACURA | RDX AWD | 2020;2021;2022;2023;2024;2025;2026 | All six contiguous transitions satisfy the safe technical + persistent EPA identity rule. |
| EX-02 | STABLE_ARCHITECTURE_2020_2026 | ACURA | RDX AWD A-SPEC | 2020;2021;2022;2023;2024;2025;2026 | All six contiguous transitions satisfy the safe technical + persistent EPA identity rule. |
| EX-03 | STABLE_ARCHITECTURE_2020_2026 | Alfa Romeo | Giulia | 2020;2021;2022;2023;2024;2025;2026 | All six contiguous transitions satisfy the safe technical + persistent EPA identity rule. |
| EX-04 | OBVIOUS_TECHNICAL_GENERATION_BREAK | AUDI | A6 Allroad | 2020;2021;2024;2025;2026 | 2021->2024: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed. |
| EX-05 | OBVIOUS_TECHNICAL_GENERATION_BREAK | AUDI | Q3 | 2020;2021;2023;2024;2026 | 2024->2026: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed. |
| EX-06 | OBVIOUS_TECHNICAL_GENERATION_BREAK | AUDI | S7 | 2020;2021;2022;2023;2024;2025 | 2022->2023: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed. |
| EX-07 | OBVIOUS_TECHNICAL_GENERATION_BREAK | AUDI | SQ7 | 2020;2021;2022;2023;2024;2025 | 2022->2023: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed. |
| EX-08 | MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM | Volkswagen | ID.4 Pro | 2021;2022 | 5 configurations remain distinct below one proposed Program parent. |
| EX-09 | MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM | NISSAN | PATHFINDER 4WD ROCK CREEK | 2023;2024;2025;2026 | 4 configurations remain distinct below one proposed Program parent. |
| EX-10 | MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM | Ford | F150 PICKUP 2WD | 2024;2025;2026 | 5 configurations remain distinct below one proposed Program parent. |
| EX-11 | MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM | Ferrari | SF90 Spider | 2022;2023;2024;2025 | 4 configurations remain distinct below one proposed Program parent. |
| EX-12 | COMMERCIAL_RENAME_OR_TRIM_NOMENCLATURE | BMW | i4 eDrive 35 Gran Coupe (18'' Wheels) | 2025;2026 | Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge. |
| EX-13 | COMMERCIAL_RENAME_OR_TRIM_NOMENCLATURE | BMW | i4 eDrive 35 Gran Coupe (19'' Wheels) | 2025;2026 | Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge. |
| EX-14 | COMMERCIAL_RENAME_OR_TRIM_NOMENCLATURE | BMW | i4 eDrive 40 Gran Coupe (18'' Wheels) | 2025;2026 | Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge. |
| EX-15 | SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS | BUICK | ENCLAVE FWD | 2024;2026 | Same normalized name, but 3 architecture domains reset across the candidate boundary. |
| EX-16 | SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS | BUICK | ENVISION FWD | 2020;2021;2022;2023 | Same normalized name, but 3 architecture domains reset across the candidate boundary. |
| EX-17 | SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS | Ford | F150 Super Crew Cab 4x4 | 2020;2021;2022;2023 | Same normalized name, but 3 architecture domains reset across the candidate boundary. |
| EX-18 | SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS | Ford | MUSTANG MACH-E RWD EXTENDED | 2024;2025 | Same normalized name, but 3 architecture domains reset across the candidate boundary. |
| EX-19 | ONE_YEAR_ONLY_MODEL | ACURA | MDX AWD A-spec | 2020 | No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span. |
| EX-20 | ONE_YEAR_ONLY_MODEL | ACURA | RLX | 2020 | No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span. |
| EX-21 | ONE_YEAR_ONLY_MODEL | ACURA | RLX HYBRID | 2020 | No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span. |
| EX-22 | ONE_YEAR_ONLY_MODEL | ACURA | TLX | 2020 | No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span. |
| EX-23 | MODEL_YEAR_GAP | ACURA | ZDX AWD | 2024;2026 | 1 missing model year(s); status=UNRESOLVED, consolidation=UNRESOLVED. |
| EX-24 | MODEL_YEAR_GAP | ACURA | ZDX AWD TYPE S | 2024;2026 | 1 missing model year(s); status=UNRESOLVED, consolidation=UNRESOLVED. |
| EX-25 | MODEL_YEAR_GAP | ACURA | ZDX RWD | 2024;2026 | 1 missing model year(s); status=UNRESOLVED, consolidation=UNRESOLVED. |
| EX-26 | POWERTRAIN_FAMILY_BEV | ACURA | ZDX AWD | 2024;2026 | Audit classification=BEV; naming signals are context only and do not establish Program identity. |
| EX-27 | POWERTRAIN_FAMILY_ICE | ACURA | ADX AWD | 2025;2026 | Audit classification=ICE; naming signals are context only and do not establish Program identity. |
| EX-28 | POWERTRAIN_FAMILY_HEV_OR_PHEV_NAME_SIGNAL | ACURA | RLX HYBRID | 2020 | Audit classification=HEV_OR_PHEV_NAME_SIGNAL; naming signals are context only and do not establish Program identity. |
| EX-29 | INSUFFICIENT_SOURCE_DATA | 2022 | Lucid Air Dream P | 2022;2023 | Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed. |
| EX-30 | INSUFFICIENT_SOURCE_DATA | 2022 | Lucid Air Dream R | 2022 | Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed. |
| EX-31 | INSUFFICIENT_SOURCE_DATA | 2022 | Lucid Air Grand Touring | 2022;2023 | Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed. |

## Remaining identity work

- Validate OEM generation/platform codes, launch/facelift chronology and authoritative model renames as future enrichment inputs.
- Review `PROBABLE_CONSOLIDATE`, `KEEP_SEPARATE` and `UNRESOLVED` groups before migration; do not encode them as hidden deterministic rules.
- Correct the 77 EPA 2020–2026 rows whose represented make is year-like (19 in MY2026) before cross-source identity work.
- Keep JRC anonymized identities and EEA monitoring identities separate from EPA unless positive evidence is later versioned into a deterministic resolver.

## Evidence tiers

- **Directly tested:** `test_deterministic_normalization_grouping_and_population_forecast`; `test_no_merge_is_caused_by_model_year_adjacency_alone`; `test_program_parent_consolidation_preserves_all_child_populations`; `test_database_access_is_read_only_and_byte_identical`; `test_null_is_not_coerced_to_zero`; `test_example_selection_is_stable_and_covers_required_cases`; `test_boundary_and_consolidation_vocabularies_are_closed`; `test_forecast_separates_jrc_and_eea_from_epa`; `test_outputs_are_scoped_to_etl`.
- **Indirectly covered:** Sprint 12C exact VDE/FuelCons reconstruction and Sprint 12C.2 candidate grain.
- **Inspection-supported:** the local EPA technical descriptors and CDR ownership rules.
- **Gap:** authoritative OEM generation identity and commercial rename evidence are not present in the supplied structured sources.

## Reproduction

```powershell
python etl/scripts/sprint_12c3_program_consolidation_review.py
python -m unittest discover -s etl/tests -p "test_sprint_12c3*.py" -v
```

## Outputs

- `etl/data/processed/sprint_12c3_program_consolidation/cross_year_profiles.csv`
- `etl/data/processed/sprint_12c3_program_consolidation/program_boundary_candidates.csv`
- `etl/data/processed/sprint_12c3_program_consolidation/program_consolidation_candidates.csv`
- `etl/data/processed/sprint_12c3_program_consolidation/program_population_forecast.csv`
- `etl/data/processed/sprint_12c3_program_consolidation/program_examples.csv`
- `etl/data/processed/sprint_12c3_program_consolidation/program_review.json`
- `etl/reports/sprint_12c3_program_consolidation_review.md`
- `etl/reports/sprint_12c3_program_examples.md`
