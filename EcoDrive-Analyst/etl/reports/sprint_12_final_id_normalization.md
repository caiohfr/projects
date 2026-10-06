# EcoDrive Sprint 12 — Final ID Normalization Patch

## Outcome

The final pre-PROD candidate now uses deterministic positive sequential
surrogate keys for VDE and FuelCons. Only those integer keys, their declared
physical references, and embedded lineage references to those keys changed.
Stable TEXT identities, schema, row population, engineering values, Run
evidence, cycle classification, carryover meaning, FuelCons adoption,
Components, and Tires remain unchanged.

- `VDE_MIN_ID = 1`
- `VDE_NEGATIVE_IDS = 0`
- `FUELCONS_MIN_ID = 1`
- `FUELCONS_NEGATIVE_IDS = 0`
- `FOREIGN_KEY_ISSUES = 0`
- `ID_NORMALIZATION_COMPLETE = YES`
- `READY_FOR_FINAL_HUMAN_SMOKE = YES`
- `READY_FOR_PROD_PROMOTION = NO`

No QA or PROD database was promoted or overwritten.

## Files changed

Implementation and focused coverage:

- `etl/scripts/sprint_12_final_id_normalization.py`
- `etl/tests/test_sprint_12_final_id_normalization.py`
- `scripts/export_db_review_excel.py`
- `tests/test_export_db_review_excel.py`
- `etl/tests/test_sprint_12_closure_phase2.py`
- `etl/tests/test_sprint_12h1_canonical_qa_database_management.py`
- `etl/reports/sprint_12_final_id_normalization.md`

Regenerated candidate and QA artifacts:

- `data/db/staging/eco_drive_canonical_candidate.db`
- `artifacts/id_remap/VDE_ID_REMAP.csv`
- `artifacts/id_remap/FUELCONS_ID_REMAP.csv`
- `artifacts/id_remap/sprint_12_final_id_normalization_summary.json`
- `artifacts/db_review/EcoDrive_Canonical_DB_Review_Final_ID_Normalization.xlsx`

No physical schema definition was changed.

## Deterministic numbering strategy

VDE rows are ordered by the stable source/canonical tuple:

```text
source_name, source_file_version, source_record_id, normalization_version,
vehicle_configuration_id, make, model, year, legislation, category,
cycle_name, coast_A_N, coast_B_N_per_kph, coast_C_N_per_kph2,
test_mass_kg
```

FuelCons rows are ordered by:

```text
source_name, source_file_version, source_record_id, normalization_version,
record_origin, comparison_basis, electrification, fuel_type,
linked VDE source_name, linked VDE source_record_id,
vehicle_configuration_id, year
```

The prior integer is used only as a final deterministic tie-breaker. SQLite
row order is never used. The current population has zero duplicate VDE and
zero duplicate FuelCons ordering tuples, so that fallback does not determine
any current assignment. The resulting one-based enumeration is written to the
two CSV audit maps and to two workbook tabs. A focused repeat-build test proves
that the database hash and both maps are identical across repeated builds.

This is a one-time pre-PROD rewrite. After promotion, existing numeric IDs are
immutable; an existing canonical identity keeps its ID and a new row receives
`MAX(id) + 1`. Future production refreshes must not renumber existing rows to
remove gaps.

## ID ranges and references

| Entity | Old range | New range | Rows | Distinct new IDs | Gaps |
|---|---:|---:|---:|---:|---:|
| VDE | -2,000,250 .. -1,000,001 | 1 .. 11,626 | 11,626 | 11,626 | 0 |
| FuelCons | -4,010,572 .. 5,018 | 1 .. 10,822 | 10,822 | 10,822 | 0 |

Actual schema-discovered VDE references remapped:

- `vde.vde_id_parent`
- `run.vde_id`
- `fuelcons.vde_id`
- `fuelcons_run_adoption.vde_id`
- `vde_component_resolution.vde_id`

Actual schema-discovered FuelCons references remapped:

- `fuelcons.reference_fuelcons_id`
- `fuelcons_run_adoption.fuelcons_id`

Embedded `carryover_from_vde_id` and `carryover_from_fuelcons_id` provenance
values and the human VDE parent note were updated to the new surrogate values.
Stable TEXT IDs (`program_id`, `vehicle_configuration_id`, `run_id`, component
IDs) were not regenerated. `tire_db.tire_id` was already positive and was not
changed.

## Row-count and semantic parity

| Physical table | Before | After |
|---|---:|---:|
| program | 3,117 | 3,117 |
| vehicle_configuration | 10,211 | 10,211 |
| vde | 11,626 | 11,626 |
| run | 29,250 | 29,250 |
| fuelcons | 10,822 | 10,822 |
| fuelcons_run_adoption | 18,323 | 18,323 |
| component_instance | 843 | 843 |
| component_db | 0 | 0 |
| component_resolution | 0 | 0 |
| vde_component_resolution | 0 | 0 |
| tire_db | 1 | 1 |

The complete schema digest is unchanged:

`B9419B589A6A9E352B9C6C2146A1689FBC151C1DDF2DD9753025BD6225EEBB2B`

Identity-resolved topology digests are equal before and after:

| Surface | SHA256 before/after | Equal |
|---|---|---|
| VDE parent topology | `E08A86748031D69F75487787751DC93D31A209438EBD279CAD249C31FA2A3A4B` | YES |
| Run-to-VDE and Run lineage | `491A7E7AC7E5C6265813B41CB593DF02D72EBBC4FAF85B60181D753FA93780C3` | YES |
| FuelCons VDE/reference topology | `9E2E11B9B95489C77BF7A1981F9C26433EB344F6CC4D256E5F725187772C49CC` | YES |
| FuelCons adoption topology | `F53F34D73FA6FE29E6C07B9D995106F9E409C48DCB61F8DBA0B096992006F4F1` | YES |

The full non-ID engineering payload digest is also equal before and after:

`59B6C4EFC27C66F2F805BE124C48BC4A9D27FD92A99FCE7B0459D1077B0A73A5`

## Representative parity

### BMW 330i

The exact Config 0 annual chain is preserved with only surrogate values
changed:

| Model year | Old VDE | New VDE | New parent |
|---:|---:|---:|---:|
| 2020 | -1,001,790 | 1,790 | NULL |
| 2021 | -1,010,565 | 10,565 | 1,790 |
| 2022 | -1,009,385 | 9,385 | 10,565 |

The same Config 0/1/2 roadloads remain separate and the VDE closure regression
passes against stable `source_record_id` identities.

### Cadillac NGMX91004749

All 15 Runs remain: three distinct Set ABC sibling variants across model years
2022–2026. Each variant retains only its exact same-variant temporal chain;
siblings are not merged or linked to one another. All 15 remain
`RUN_IDENTITY_REVIEW` records. Only their referenced numeric VDE IDs changed.

### FuelCons

All 10,822 annual rows and all 18,323 Run adoptions remain. Full FuelCons and
adoption topology digests match before/after, and the engineering payload
digest proves the KPI values are identical. EPA reconstructed annual rows do
not repurpose `reference_fuelcons_id` for model-year carryover.

## Integrity and negative-ID scan

- `PRAGMA quick_check`: `ok`
- `PRAGMA foreign_key_check`: 0 issues
- VDE parent orphans: 0
- Run VDE orphans: 0
- FuelCons VDE orphans: 0
- FuelCons parent-reference orphans: 0
- adoption VDE orphans: 0
- adoption FuelCons orphans: 0
- VDE/component-resolution orphans: 0

The scan of every integer identifier/reference column found zero negative IDs
in VDE, FuelCons, Run references, adoption references, VDE component
references, Tire references, or Component Instance Tire references.

## Candidate and source safety

- Source candidate SHA256 before/after:  
  `D80E1918FE72AF129E1104F3C2C8C0A07037264B2E2A1E01839543F04D5498DF`
- Normalized staging candidate SHA256:  
  `1E3B251F39B21A835EF9F0B4F6291A495636A8C425CEF7051754D1D750840ACF`
- QA remains unpromoted at SHA256:  
  `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- PROD remains unpromoted at the same pre-promotion SHA256.

The normalizer rejects runtime/QA and any PROD output path. It builds a copied
temporary candidate, validates it, then atomically replaces only the staging
target.

## Review workbook

- Path: `artifacts/db_review/EcoDrive_Canonical_DB_Review_Final_ID_Normalization.xlsx`
- SHA256: `C8EB8A02A63EF2D01DDEDA5692E241FA2E0C3E058615408852C05ECF1E4B402B`
- Size: 50,991,911 bytes
- Sheets: 26
- Source database hash before/after export: unchanged
- OOXML ZIP integrity: passed

The normal VDE and FuelCons sheets contain numeric IDs 1..N. The additional
`VDE_ID_REMAP` and `FUELCONS_ID_REMAP` sheets contain 11,626 and 10,822 audit
rows respectively. Metadata, QA, physical-table, compatibility-view, and
helper-object sheets remain included.

## Tests

- Final ID normalization focused tests: 5/5 passed.
- SQLite-to-Excel exporter tests: 7/7 passed.
- Sprint 12 closure semantic tests: 27/27 passed.
- Canonical staging/QA/PROD boundary tests: 12/12 passed.
- Complete application regression suite: 1,847/1,847 passed (exit code 0).
- Python compilation checks: passed.

These 51 focused and semantic regression checks cover deterministic rebuilds,
source immutability, all physical FK remaps, embedded lineage IDs, stable TEXT
identities, schema/count/payload parity, BMW/Cadillac/FuelCons behavior,
workbook row parity, numeric positive IDs, and the no-promotion boundary.

Automated validation makes the normalized staging candidate ready for final
human inspection. PROD promotion remains a separate explicit action.
