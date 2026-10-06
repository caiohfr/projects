# Sprint 12 — Final Micro Patch: Run Carryover Test Number

## Outcome

The confirmed Run-lineage defect was reproduced and corrected without changing schema, VDE carryover, FuelCons behavior, cycle labels, Component/Tire logic, or PROD/QA databases.

- `MICRO_PATCH_COMPLETE = YES`
- `RUN_TEST_NUMBER_MISMATCH_LINKS = 0`
- `CANDIDATE_REGENERATED = YES`
- `REVIEW_WORKBOOK_REGENERATED = YES`
- `READY_FOR_FINAL_HUMAN_SMOKE = YES`
- `READY_FOR_PROD_PROMOTION = NO`

## Files changed

Primary source/test changes:

- `etl/scripts/sprint_12e2_epa_fuelcons_reconstruction.py`
- `etl/tests/test_sprint_12_closure_phase2.py`
- `etl/scripts/sprint_12_final_design_patch_validation.py`
- `scripts/export_db_review_excel.py`
- `etl/reports/sprint_12_final_micro_patch_run_carryover_test_number.md`

Regenerated artifacts and generated reports:

- `data/db/staging/eco_drive_canonical_candidate.db`
- `artifacts/db_review/EcoDrive_Canonical_DB_Review_Final_Micro_Patch.xlsx`
- Sprint 12E.2, 12F.12, and 12F.13 generated reports/processed exports.

No schema file changed.

## Confirmed pre-patch reproduction

The prior staging candidate was opened read-only before regeneration:

- Run carryover links: 13,050.
- Linked parent/child pairs with different EPA Test Number: 159.
- Ford Edge AWD `KFMX10071543 -> KFMX10067860`: 1 invalid linked pair.

This proves the reported defect existed in the pre-patch candidate.

## Exact Run signature delta

All prior exact signature fields remain. Two source identity fields were added only to the deterministic Run lineage signature:

```text
test_number      = source "Test Number"
adfe_test_number = source "ADFE Test Number"
```

`ADFE Test Number` is present in the current EPA source contract and populated on 2,806 source rows, materializing on 2,686 canonical Runs after execution-grain grouping. Values are compared exactly. No fuzzy matching, normalization across different values, or approximate matching was introduced.

The existing execution evidence fields remain unchanged: represented make/model, test group, test vehicle, configuration, Target ABC, ETW, category, procedure, fuel, Set ABC, and canonical results.

FuelCons remains isolated from this micro patch. Its adopted-Run evidence signature retains the exact pre-patch payload, while only Run temporal lineage uses the two additional Test Number fields.

## Before/after Run lineage

| Metric | Before | After |
|---|---:|---:|
| Run carryover links | 13,050 | 12,920 |
| Linked pairs with different Test Number evidence | 159 | 0 |
| Ford observed mismatched pair | 1 | 0 |

- Invalid prior links removed because Test Number changed: 159.
- New exact links created under the corrected signature: 29.
- Net Run carryover link delta: -130.

Every one of the 159 formerly mismatched pairs was removed. No mismatched parent/child pair remains.

## Unchanged behavior proof

Before replacing the staging candidate, all rows of the new final database were compared with the prior candidate:

| Surface | Prior row-content SHA256 | New row-content SHA256 | Equal |
|---|---|---|---|
| VDE | `582DD5EEE469F3D1AD93CB7EDA5771C66E161E4E762DC4F21ECC223BF3801DE8` | same | YES |
| FuelCons | `995868D70B5610DAEB9D84AAD61AD915734675001BF25DDDA7F063E1D7B6F02E` | same | YES |
| FuelCons adoption | `0654D4F1BF441BE8324212C1D158546E6C0873D81A19984772140A38B535C3DB` | same | YES |

Therefore VDE carryover, FuelCons values/lineage/adoptions, and their complete stored rows are byte-equivalent to the prior candidate.

The BMW 330i VDE chain remains:

- MY2020 `-1001790` -> NULL
- MY2021 `-1010565` -> `-1001790`
- MY2022 `-1009385` -> `-1010565`

## Cadillac verification

Test Number `NGMX91004749` remains:

- 15 Runs;
- three distinct Set ABC variants;
- five model years per variant;
- distinct siblings within each model year;
- exact same-Test-Number/same-Set-ABC temporal chains only;
- all 15 records retained in `RUN_IDENTITY_REVIEW` with explanations.

The broader review population changed from 641 to 369 because the corrected identity resolves generic-signature ambiguity outside the Cadillac limitation. The accepted Cadillac source-grain limitation itself is unchanged.

## Tests

Focused micro-patch/export suite: 33/33 passed.

Direct micro-patch tests:

- `test_run_signature_allows_same_test_number_and_execution_evidence`
- `test_run_signature_blocks_different_test_number`
- `test_run_signature_blocks_different_adfe_test_number`
- `test_run_signature_keeps_cadillac_same_test_number_set_abc_variants_distinct`
- `test_linked_runs_have_identical_test_number_evidence`
- `test_ford_edge_changed_test_number_run_is_not_linked`
- `test_fuelcons_run_evidence_signature_remains_unchanged_by_test_number`

Existing regression evidence also passed:

- `test_bmw_330i_annual_chain_is_preserved`
- `test_linked_vdes_have_identical_test_number_and_procedure_evidence`
- `test_fuelcons_annual_rows_are_preserved_without_reference_repurposing`
- `test_cadillac_set_abc_variants_are_distinct_reviewed_chains`
- six Excel exporter tests.

Pipeline acceptance:

- Sprint 12E.2 relationship failures: 0; repeat rebuild deterministic: YES.
- Sprint 12F.12 acceptance: 11/11; deterministic: YES.
- Sprint 12F.13 acceptance: 15/15; deterministic: YES.
- Repository regression suite: 1,846/1,846 passed (exit code 0).

## Regenerated candidate

- Path: `data/db/staging/eco_drive_canonical_candidate.db`
- SHA256: `D80E1918FE72AF129E1104F3C2C8C0A07037264B2E2A1E01839543F04D5498DF`
- `PRAGMA quick_check`: `ok`
- Foreign-key issues: 0
- Rows: 11,626 VDE; 29,250 Run; 10,822 FuelCons; 18,323 FuelCons adoption.
- VDE carryover links: 5,084, unchanged.
- FuelCons carryover links: 4,931, unchanged.

PROD and QA were not promoted or overwritten. Both remained at SHA256:

`243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`

## Regenerated workbook

- Path: `artifacts/db_review/EcoDrive_Canonical_DB_Review_Final_Micro_Patch.xlsx`
- SHA256: `864EFF6B4D8E2BF24D417F9DEE52A8CAEA60BBCE7D00063C78869EA7CBE190F2`
- Size: 49,117,320 bytes
- Sheets: 24
- Source database SHA256 before/after export: unchanged.
- OOXML ZIP integrity: passed.
- `RUN_IDENTITY_REVIEW`: 369 rows; requested identity fields present, including `test_number` and `adfe_test_number`.

The workbook is ready for the requested human smoke. No interactive/manual browser or Excel smoke was claimed or performed by the automated suite.
