# Sprint 12 Closure — Phase 2 Implementation Report

## 1. Summary

Phase 2 was implemented conservatively against a regenerated canonical candidate. Annual EPA rows remain distinct; exact model-year carryover is now explicit and temporal; unresolved within-year Run variants remain quarantined for review. No candidate was promoted to PROD.

## 2. Files changed

Implementation files:

- `etl/scripts/sprint_12_closure_phase2.py`
- `etl/schema/canonical_schema_v1.sql`
- `etl/scripts/sprint_12e_migration_rehearsal.py`
- `etl/scripts/sprint_12e1_clean_rebuild_migration.py`
- `etl/scripts/sprint_12e2_epa_fuelcons_reconstruction.py`
- `src/vde_core/vde_request_save.py`
- `src/vde_core/comparison_report_service.py`
- `src/vde_app/comparison_report_viewmodels.py`
- `src/vde_app/components/comparison_report.py`
- `scripts/export_db_review_excel.py`
- `tests/test_export_db_review_excel.py`
- `etl/tests/test_sprint_12_closure_phase2.py`

Generated artifacts include the staging candidate, ETL reports/CSVs, and the Phase 2 review workbook. Unrelated pre-existing worktree changes were not modified as part of this report.

## 3. Schema changes

Two nullable Vehicle Configuration scalars were added:

- `engine_rated_power_kw REAL`
- `engine_cylinders_rotors INTEGER`

No new lineage subsystem was introduced. The existing parent/provenance contracts were used.

## 4. Carryover identity/signature implemented

Carryover uses an explicit exact signature, not approximate matching or `drop_duplicates`. It excludes model year and publication-row metadata and retains stable configuration identity, exact roadload/weight evidence, test identifiers, procedure/category evidence, and available result evidence. A later row links only when the previous available year contains one unambiguous exact match. Changed evidence or multiple eligible matches leave the parent unset.

## 5. Lineage representation implemented

EPA VDE carryover uses `vde_id_parent` for the immediate previous available matching year plus structured provenance:

- `lineage_relation = EPA_MODEL_YEAR_CARRYOVER`
- `carryover_from_model_year`
- `carryover_from_vde_id`
- a human-readable carryover note

Engineering saves use `ENGINEERING_SCENARIO`; readers expose the relation so EPA carryover is not presented as a user proposal. FuelCons carryover remains in provenance and does not repurpose `reference_fuelcons_id`.

## 6. BMW before/after

Before Phase 2, annual BMW records were separate but their temporal relationship was implicit. After Phase 2, BMW 330i Config 0 is an annual chain:

- MY2020 `-1001790` → parent `NULL`
- MY2021 `-1010565` → parent `-1001790`
- MY2022 `-1009385` → parent `-1010565`

Each year retains its own VDE and FuelCons applicability row. Configurations 0/1/2 remain distinct.

## 7. Cadillac before/after

EPA Test Number `NGMX91004749` remains 15 Runs: three exact Set ABC variants across five model years (2022–2026). Each variant chains only to the same exact variant in the immediately preceding available year. No sibling variant points to another sibling. All 15 remain discoverable through `RUN_IDENTITY_REVIEW`; the within-year execution meaning remains unresolved by design.

## 8. Audi R8 verification

Audi R8 VDEs `-1011238`, `-1009341`, and `-1004235` retain three distinct Target ABC roadloads. The exact signature does not collapse them.

## 9. Multi-cycle VDE before/after

EPA VDE `-1004236` now has null `cycle_name`/`cycle_source`; its FTP, HWY, SC03, and US06 identities remain on the supporting Runs. No `EPA_STD` value or first-row cycle was manufactured.

## 10. Scalar-field migration results

- `engine_rated_power_kw`: 9,962 Vehicle Configurations populated using `hp × 0.745699872`.
- `engine_cylinders_rotors`: 8,641 Vehicle Configurations populated.
- Raw source values remain in `source_identity_json.raw_source_values`.
- `fuelcons.engine_max_power_kw` ownership and semantics were not changed.

## 11. FuelCons behavior

Annual FuelCons rows remain materialized. There are 10,822 final FuelCons rows and 4,959 exact cross-year carryover links in provenance. No EPA reconstructed FuelCons row uses `reference_fuelcons_id` for model-year carryover.

## 12. Component/Tire verification

- `component_db`: 0 rows
- `component_instance`: 843 rows
- `component_resolution`: 0 rows
- `vde_component_resolution`: 0 rows
- `tire_db`: 1 row

No component or tire identity was invented.

## 13. Review workbook tabs generated

Workbook: `artifacts/db_review/EcoDrive_Canonical_DB_Review_Phase2.xlsx`

SHA256: `BD73E37E44156FB33F59EF72A53C342104A2390F222EAC9EC516B319C5C537AC`

Twenty-four sheets were generated:

- Metadata/QA: `00_SUMMARY`, `01_SCHEMA`, `02_RELATIONSHIPS`, `03_ORIGIN_COUNTS`, `04_NULL_COVERAGE`, `EPA_CARRYOVER_REVIEW`, `RUN_IDENTITY_REVIEW`, `VDE_DUPLICATE_CANDIDATES`, `FUELCONS_DUPLICATE_CANDIDATES`, `JSON_SCALAR_REVIEW`
- Core/data: `program`, `vehicle_configuration`, `vde`, `run`, `fuelcons`, `fuelcons_run_adoption`, `vde_component_resolution`, `component_db`, `tire_db`, `vde_db_view`, `fuelcons_db_view`
- Additional existing objects: `component_instance`, `component_resolution`, `fuelcons_lineage_v1`

QA row counts: EPA carryover review 11,377; Run identity review 641; VDE duplicate candidates 8,276; FuelCons duplicate candidates 7,629; JSON scalar review 10,211. Candidate tabs are review aids, not automatic duplicate declarations.

The requested unsuffixed workbook was open/locked by Excel, so it was not overwritten. The complete regenerated workbook was safely written with the `_Phase2` suffix.

## 14. Exact tests run and results

`etl.tests.test_sprint_12_closure_phase2`: 15/15 passed.

Exact tests:

- `test_audi_r8_distinct_roadloads_remain_distinct`
- `test_bmw_330i_annual_chain_is_preserved`
- `test_cadillac_set_abc_variants_are_distinct_reviewed_chains`
- `test_carryover_relation_provenance_and_human_note`
- `test_compatibility_views_remain_usable`
- `test_epa_cycles_belong_to_runs_not_vdes`
- `test_fuelcons_annual_rows_are_preserved_without_reference_repurposing`
- `test_scalars_and_non_invention_boundaries`
- `test_ambiguous_same_year_signature_is_not_linked`
- `test_approved_vehicle_configuration_scalar_conversions`
- `test_changed_signature_never_links`
- `test_engineering_scenario_lineage_remains_distinct`
- `test_no_false_carryover_when_test_evidence_changes`
- `test_temporal_parent_is_immediate_previous_available_year`
- `test_three_exact_variants_chain_only_to_their_own_variant`

Evidence classification:

- Directly tested: exact classification, temporal predecessor selection including missing intermediate years, changed-evidence rejection, ambiguous-parent rejection, sibling preservation, provenance, scenario distinction, scalar conversion, EPA cycle ownership, FuelCons preservation, compatibility views, and non-invention boundaries.
- Indirectly covered: application repository/adaptor behavior through the broader suite.
- Inspection-supported: representative BMW, Cadillac, Audi and multi-cycle records; workbook contents; final counts.
- Unresolved gap: physical meaning of multiple within-year Cadillac Set ABC executions.

## 15. Full/broader suite status

- Focused affected root tests: 342/342 passed.
- Full root suite: 1,846/1,846 passed.
- Final dedicated Phase 2 ETL tests: 15/15 passed.

## 16. Candidate DB integrity results

Candidate: `data/db/staging/eco_drive_canonical_candidate.db`

- SHA256: `B8789AB92D9B7A4F7A005006C3A5D341E14B8F6772E63EA603AF406391224D61`
- Size: 213,610,496 bytes
- `PRAGMA quick_check`: `ok`
- Foreign-key violations: 0
- Deterministic rebuild checks: passed
- EPA VDE rows with non-null cycle: 0
- VDE carryover links: 5,336
- Ambiguous VDE carryover links manufactured: 0

PROD and QA remained unchanged at SHA256 `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`.

## 17. Before/after table counts

The clean rebuild contained 11,630 VDE rows before the established 12F.13 cleanup removed four known QA artifacts. The regenerated final candidate contains:

| Object | Final rows |
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

## 18. Remaining unresolved gaps

- 641 Runs remain explicitly quarantined as `RUN_IDENTITY_REVIEW`; this includes the accepted Cadillac multi-Set-ABC limitation.
- The older Phase 1 12F.14A audit reports `VDE_GRAIN_AUDIT_INCONCLUSIVE` because it expected the pre-Phase-2 cycle representation. Phase 2 intentionally supersedes that recommendation; the strict physical duplicate count remains zero.
- The final semantic approval and PROD promotion are human decisions.

## 19. Manual smoke status

Read-only Streamlit smoke against the staging candidate passed for:

- VDE Setup
- Database Management
- Comparison Report

All three rendered without application exceptions. Database Management no longer raises the previously observed invalid `SOURCE_REFRESHED` origin error. Automated visual-browser startup was unavailable because the browser connector rejected its sandbox metadata; the Streamlit application test harness was used as the read-only fallback. No save/mutation action was performed.

## 20. Human-review readiness

The regenerated staging candidate and the Phase 2 workbook are ready for manual semantic review. Sprint 12 is not declared closed, and PROD promotion remains deliberately blocked pending human approval.

    PHASE_2_IMPLEMENTATION_COMPLETE = YES
    CANDIDATE_REGENERATED = YES
    READY_FOR_MANUAL_SEMANTIC_REVIEW = YES
    READY_FOR_PROD_PROMOTION = NO
