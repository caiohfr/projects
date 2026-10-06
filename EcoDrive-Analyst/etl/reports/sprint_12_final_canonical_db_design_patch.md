# Sprint 12 — Final Canonical DB Design Patch

## Outcome

The final design-correction patch was implemented against the staging candidate only. The canonical candidate database and engineering review workbook were regenerated. No PROD/QA database was promoted or overwritten.

- `FINAL_DESIGN_PATCH_COMPLETE = YES`
- `CANDIDATE_REGENERATED = YES`
- `REVIEW_WORKBOOK_REGENERATED = YES`
- `READY_FOR_HUMAN_SEMANTIC_REVIEW = YES`
- `READY_FOR_PROD_PROMOTION = NO`

## Files changed by this patch

- `etl/schema/canonical_schema_v1.sql`
- `etl/scripts/sprint_12_closure_phase2.py`
- `etl/scripts/sprint_12e_migration_rehearsal.py`
- `etl/scripts/sprint_12e2_epa_fuelcons_reconstruction.py`
- `scripts/export_db_review_excel.py`
- `etl/tests/test_sprint_12_closure_phase2.py`
- `etl/scripts/sprint_12_final_design_patch_validation.py` (new)
- `etl/reports/sprint_12_final_canonical_db_design_patch.md` (new)

Generated artifacts:

- `data/db/staging/eco_drive_canonical_candidate.db`
- `artifacts/db_review/EcoDrive_Canonical_DB_Review_Final_Design_Patch.xlsx`

## Schema delta

Two nullable VDE roadload-condition fields were added:

```sql
roadload_temperature_c REAL,
roadload_ambient_pressure_kpa REAL
```

No entity was added and no legacy engineering field was removed. The fields are distinct from FuelCons ambient temperature and tire test temperature.

## EPA roadload-condition classification

EPA VDEs use only the roadload-condition family values `EPA NORMAL`, `EPA COLD`, and `EPA CUSTOM`; `cycle_source` is `EPA_TESTCAR`. The classifier reads only explicit roadload-condition evidence and does not use Run schedule/category fields. FTP, HWY, US06, SC03, and cold test schedules remain Run-level identities.

The current EPA extract exposes no explicit roadload-condition temperature/pressure source columns and no explicit cold/custom roadload examples. Consequently all 11,377 current EPA VDEs are truthfully classified `EPA NORMAL`, while both new scalars remain NULL. The schema/classifier support explicit cold/custom inputs, and focused unit tests exercise them without fabricating database records.

## Exact temporal carryover contract

Annual VDE and FuelCons rows remain separate. A VDE parent is assigned only to the immediately previous available model year with the same deterministic evidence signature. The V2 signature compares, where present:

- Test Number and ADFE Test Number;
- Test Category and test procedure code/description;
- Test Vehicle ID, Test Group, and Configuration Number;
- Target ABC and Set ABC/execution variant evidence;
- ETW/test mass and roadload horsepower;
- FE, CO2, emissions/results, fuel-economy unit, and bag evidence.

Model year, source spreadsheet row, and publication-location metadata are intentionally excluded. Matching is exact: there is no fuzzy matching and no approximate ABC tolerance. A changed Test Number, procedure, Set ABC variant, or relevant result evidence blocks carryover. The relation provenance is `EPA_MODEL_YEAR_CARRYOVER`, the rule is `EXACT_ROADLOAD_AND_TEST_EVIDENCE_V2`, and the parent is never the oldest/root row when an immediate matching predecessor exists.

FuelCons carryover is derived only from adopted test evidence that meets this corrected contract. `reference_fuelcons_id` was not repurposed.

## Candidate validation

- Candidate: `data/db/staging/eco_drive_canonical_candidate.db`
- Size: 237,219,840 bytes
- SHA256: `27BC8F3EF4D2BB4A3DFB8E37AFA28E62E731800E0ECE5CDF55A6A87BEDC461E6`
- `PRAGMA quick_check`: `ok`
- `PRAGMA foreign_key_check`: 0 issues
- Deterministic rebuild: confirmed
- PROD/QA SHA256 remained `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`

Final row counts:

| Object | Rows |
|---|---:|
| program | 3,117 |
| vehicle_configuration | 10,211 |
| vde | 11,626 |
| run | 29,250 |
| fuelcons | 10,822 |
| fuelcons_run_adoption | 18,323 |
| component_db | 0 |
| component_instance | 843 |
| component_resolution | 0 |
| vde_component_resolution | 0 |
| tire_db | 1 |

The final cleanup removed only the known QA fixtures: 4 VDEs, 8 Runs, 4 FuelCons rows, 4 adoption rows, and 4 vehicle configurations. Component/tire conclusions were unchanged.

Semantic metrics:

- EPA VDE cycle counts: `EPA NORMAL = 11,377`, `EPA COLD = 0`, `EPA CUSTOM = 0`.
- VDE carryover relations: 5,336 before correction; 5,084 after correction (252 permissive links removed).
- FuelCons carryover relations: 4,959 before correction; 4,931 after correction (28 permissive links removed).
- RUN_IDENTITY_REVIEW rows: 641; rows without a reason: 0.
- Populated `engine_rated_power_kw`: 9,962.
- Populated `engine_cylinders_rotors`: 8,641.
- Populated roadload temperature/pressure scalars: 0/0 because the current extract has no explicit support.

## Representative before/after cases

### BMW 330i

The exact annual chain is preserved and points to the immediately previous matching available year:

| Model year | VDE ID | Parent VDE ID | Condition |
|---:|---:|---:|---|
| 2020 | -1001790 | NULL | EPA NORMAL |
| 2021 | -1010565 | -1001790 | EPA NORMAL |
| 2022 | -1009385 | -1010565 | EPA NORMAL |

Configuration variants remain separate, and the latest model year remains a normal selectable record.

### Cadillac NGMX91004749

The source has 15 Runs and three Set ABC variants. The variants remain distinct siblings and do not parent one another within a year. Exact cross-year variant chains are allowed only variant-to-same-variant. All affected QA rows have a useful reason; the physical meaning of the three within-year variants remains unresolved source grain.

### Changed Test Number

Audi A4 Sedan MY2022, VDE `-1008410`, previously pointed to VDE `-1006919`. ABC/category/procedure evidence matched, but the child Test Numbers (`MVGA10069583`, `MVGA10069605`) differed from the prior rows (`MVGA10065239`, `MVGA10065240`). The final parent is NULL, proving that changed test evidence blocks the former false carryover.

### Multi-schedule normal EPA VDE

VDE `-1004236` has FTP, HWY, SC03, and US06 Runs while the VDE remains `EPA NORMAL` / `EPA_TESTCAR`. Its roadload temperature and pressure are NULL. This demonstrates that Run schedules no longer determine the VDE roadload family.

### Cold/custom capability

No current source row explicitly supports a cold or custom roadload condition. No fake record or scalar was created. Unit tests prove explicit cold and custom evidence is classified and preserved when supplied.

## Review workbook

- Path: `artifacts/db_review/EcoDrive_Canonical_DB_Review_Final_Design_Patch.xlsx`
- Size: 49,152,812 bytes
- SHA256: `632E30B908DCF7648924563AAD8CA4BC19A7E598DBF1BEE646D357ECFF3038DA`
- Source DB hash before/after export: unchanged
- Sheets: 24

The workbook includes the five metadata sheets, all requested semantic QA sheets, canonical data sheets, compatibility views, and the additional existing helper objects `component_instance`, `component_resolution`, and `fuelcons_lineage_v1`. The VDE sheet exposes cycle family/source, both roadload-condition scalars, parent lineage, notes, ABC, mass, and existing engineering fields directly. RUN_IDENTITY_REVIEW contains 641 rows, every row has a reason, and the requested identity/carryover fields are present.

## Tests

Focused patch file: 21/21 passed.

- `test_epa_normal_roadload_condition_ignores_run_schedule`
- `test_epa_cold_roadload_condition_requires_explicit_evidence`
- `test_epa_custom_roadload_condition_preserves_direct_scalars`
- `test_vde_signature_changes_with_test_number_or_procedure`
- `test_approved_vehicle_configuration_scalar_conversions`
- `test_temporal_parent_is_immediate_previous_available_year`
- `test_changed_signature_never_links`
- `test_no_false_carryover_when_test_evidence_changes`
- `test_engineering_scenario_lineage_remains_distinct`
- `test_ambiguous_same_year_signature_is_not_linked`
- `test_three_exact_variants_chain_only_to_their_own_variant`
- `test_epa_vde_uses_roadload_family_while_runs_keep_schedules`
- `test_run_identity_review_reason_is_always_present`
- `test_linked_vdes_have_identical_test_number_and_procedure_evidence`
- `test_bmw_330i_annual_chain_is_preserved`
- `test_carryover_relation_provenance_and_human_note`
- `test_fuelcons_annual_rows_are_preserved_without_reference_repurposing`
- `test_compatibility_views_remain_usable`
- `test_cadillac_set_abc_variants_are_distinct_reviewed_chains`
- `test_audi_r8_distinct_roadloads_remain_distinct`
- `test_scalars_and_non_invention_boundaries`

Additional checks:

- Excel exporter tests: 6/6 passed.
- Phase 2 focused acceptance embedded in cleanup: 15/15 passed.
- Python compilation checks: passed.
- Full repository regression suite: 1,846/1,846 passed (exit code 0).

## Remaining source-data limitations

- The EPA source does not expose explicit roadload ambient temperature or pressure columns, so the new scalars remain NULL instead of being inferred.
- The current extract has no explicit cold/custom roadload example; those paths are contract- and test-supported only.
- Cadillac `NGMX91004749` retains three unexplained within-year Set ABC execution variants and remains flagged for human review.
- The patch does not resolve source-grain uncertainty through fuzzy matching, arbitrary selection, or record merging.
