# Sprint 12 — Component Knowledge Baseline 40% Result

## Decision

The package stopped at the source-evidence gate. No local historical raw
component-loss source was found. Populating canonical Component entities would
therefore require synthetic fixtures, engineering priors, or estimates split
from whole-vehicle roadload, all of which are explicitly prohibited by the
package contract.

No SQLite database was written, no canonical candidate was regenerated, and no
review workbook or official rule-based Component notebook was produced.

## A. SOURCE DISCOVERY

### Component-related sources found

The deterministic inventory contains 15 entries in
`artifacts/components/COMPONENT_SOURCE_INVENTORY.csv`:

- four five-row CSV fixtures for brake, transmission, axle/hub, and parasitic
  components. They explicitly identify themselves as mock/demo synthetic data
  (`SYNTHETIC_COMPONENT_FIXTURE`);
- `data/standards/vde_defaults_by_category_trans_elec.csv`, containing 255
  category-level transmission/brake default priors;
- `notebooks/etl_epa_xlsx_to_sqlite.ipynb`, which derives legacy `*_est`
  component values by distributing whole-vehicle roadload residuals according
  to those priors;
- the legacy pre-Sprint-12 SQLite database, which persists those derived
  estimates rather than raw component observations;
- current EPA and JRC vehicle files, which contain vehicle/application
  descriptors and whole-vehicle roadload but no causal component-loss ABC;
- the canonical candidate and existing source-to-contract audit artifacts;
- `docs/component_provenance_metadata.md`, which describes the intended
  upstream historical PL workflow but contains no numeric evidence;
- the referenced `Parasitic_Loss_Testing_Methodology(1).md`, which is absent.

The repository and Git history were searched for relevant `*.ipynb`, `*.csv`,
`*.xlsx`, `*.xls`, `*.db`, `*.sqlite`, `*.json`, `*.parquet`, `*.md`, and
`*.txt` material. No additional defensible source was found.

### Raw numeric sources found

Zero. Numeric component-like values exist only as:

- explicitly synthetic fixtures;
- category-level engineering priors/defaults;
- vehicle-roadload-derived estimates.

None is admissible as an observed Component Resolution.

### Historical PL source availability

`NO`. The historical PL source described by repository documentation and the
legacy notebook is not present locally.

### Exact missing source artifacts needed

1. The original historical PL Dyno workbook(s) or equivalent raw export, with
   stable source-row identity.
2. The referenced `Parasitic_Loss_Testing_Methodology(1).md` (or its controlled
   replacement) defining the physical boundaries and configuration-subtraction
   method.
3. The historical PL analysis notebook(s)/scripts that map raw configurations
   to Brake Baseline, Brake Standard, Transmission, Axle, and Hub/Bearing
   results.
4. For every numeric result: raw A/B/C or force observations, original units,
   speed basis, test condition, method, boundary, component position,
   driveline architecture, and source record/file version.
5. Hardware/part/catalog evidence if reusable `component_db` identities are
   desired; application labels alone are insufficient.
6. Explicit adoption evidence if any Component Resolution is to be linked to a
   VDE through `vde_component_resolution`.

## B. 40/60 CONTRACT

The 10-dimension contract is frozen in
`docs/sprints/SPRINT_12_COMPONENT_KNOWLEDGE_40_CONTRACT.md` and enumerated in
`artifacts/components/COMPONENT_KNOWLEDGE_COVERAGE.csv`.

| Dimension | Status | Evidence / limitation |
|---|---|---|
| 1. Component identity / taxonomy | PARTIAL_NOW | Existing application instances provide domain/role taxonomy; reusable hardware identity is unresolved. |
| 2. Application / physical context | PARTIAL_NOW | 843 JRC-derived application instances retain vehicle context; PL configuration subtraction, position, and physical boundary are absent. |
| 3. Engineering loss behavior | NOT_SUPPORTED_BY_CURRENT_SOURCE | No admissible raw component ABC/force curve is local. |
| 4. Methodology / provenance / quality | PARTIAL_NOW | Existing records have provenance and explicitly unresolved status, but the controlling PL methodology and raw lineage are missing. |
| 5–10. Research/RAG dimensions | DEFERRED_RAG | Deliberately deferred; no RAG implementation was added. |

Strict acceptance accounting is `0/4` implemented dimensions. Three dimensions
are partial and are not counted as complete. The future boundary is documented
in `docs/components/COMPONENT_RESEARCH_RAG_FUTURE_BOUNDARY.md`.

## C. CANONICAL DATA

| Entity | Before | After | Package action |
|---|---:|---:|---|
| `component_db` | 0 | 0 | No write; no defensible reusable hardware source. |
| `component_instance` | 843 | 843 | No write; existing partial public-source baseline retained. |
| `component_resolution` | 0 | 0 | No write; no raw component-loss source. |
| `vde_component_resolution` | 0 | 0 | No write; no explicit adoption evidence. |
| `tire_db` | 1 | 1 | No write; tire domain remains separate. |

The 843 existing instances comprise BATTERY 60, EMOTOR 60, ENGINE 225, TIRE
249, and TRANSMISSION 249. Their `component_id` values remain unresolved by
design; provenance identifies `UNRESOLVED_NO_FABRICATED_MASTER`. This is an
existing partial baseline, not a new population created by this package.

## D. FAMILY COVERAGE

| Family | Canonical real numeric coverage | Finding |
|---|---|---|
| Brake Baseline | None | Synthetic fixture and derived legacy estimate only. |
| Brake Standard | None | Synthetic fixture only; no raw controlled-condition evidence. |
| Transmission / neutral drag | None | Synthetic fixture, category prior, and vehicle-derived estimate only. |
| Axle / differential | None | Synthetic fixture only. |
| Hub / bearing | None | Synthetic fixture only. |
| Other parasitic families | None | Synthetic fixture or vehicle-derived estimate only. |

No family was pooled or converted into canonical data.

## E. DATA QUALITY

- Unresolved identities: all 843 existing `component_instance` rows retain a
  null `component_id`; no catalog master was fabricated.
- Negative/zero results: not assessable because there are no admissible raw
  component results. Nothing was deleted or normalized.
- Outlier flags: zero rows; the outlier artifact is header-only by design until
  real evidence is available.
- Unit conversions: not executed because source units and raw component values
  are unavailable. No conversion helper was introduced without a valid source
  contract.
- Provenance completeness: sufficient to classify current public application
  instances and to reject synthetic/estimated numeric candidates; insufficient
  to establish historical PL physical boundaries and raw numeric lineage.
- Reference populations: zero rows; the artifact is header-only rather than
  publishing incomparable or fabricated benchmarks.

## F. NOTEBOOK

- Source notebooks consolidated: none.
- Analyses preserved: all existing notebooks remain unchanged.
- Run All status: not applicable; the official Components — Rule-Based
  Engineering notebook was intentionally not created because its required
  evidence is missing.

Creating that notebook now would falsely imply that the required component
populations and numeric engineering data exist.

## G. TESTS

### Focused result

Command:

```text
python -m unittest etl.tests.test_sprint_12_component_knowledge_40_discovery -v
```

Result: `5/5 passed` in 15.233 seconds.

The tests prove inventory reproducibility, rejection of inadmissible numeric
sources, the frozen 10-dimension contract, read-only candidate access/hash
stability, clean SQLite integrity, and explicitly empty blocked-population
artifacts.

### Full regression result

Command:

```text
python -m unittest discover tests
```

Result: `1847/1847 passed` in 427.509 seconds.

### DB integrity and immutability

- Candidate `PRAGMA quick_check`: `ok`.
- Candidate `PRAGMA foreign_key_check`: 0 issues.
- Candidate SHA256 before/after discovery:
  `1E3B251F39B21A835EF9F0B4F6291A495636A8C425CEF7051754D1D750840ACF`.
- QA SHA256 after investigation:
  `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`.
- PROD SHA256 after investigation:
  `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`.

No database mutation or promotion occurred.

## H. REPRESENTATIVE CASES

- Brake Standard: blocked; no real Standard-condition source record exists.
- Brake Baseline: blocked; no real as-received source record exists.
- Transmission: blocked; architecture-aware numeric grouping cannot be proven
  from defaults or whole-vehicle-derived estimates.
- Axle/Hub: blocked; only synthetic fixtures were found.
- Unresolved hardware identity: demonstrated by the retained 843 application
  instances with null `component_id`; no master was fabricated.
- Negative/zero preservation: no admissible source population exists to
  demonstrate this numerically; no available value was removed or rewritten.

## I. STATUS

```ini
COMPONENT_SOURCE_DATA_AVAILABLE = NO
COMPONENT_DB_REAL_SEED_CREATED = NO
COMPONENT_INSTANCE_BASELINE_CREATED = PARTIAL
COMPONENT_RESOLUTION_BASELINE_CREATED = NO

COMPONENT_KNOWLEDGE_DIMENSIONS_NOW = 0/4
COMPONENT_KNOWLEDGE_MODEL_COVERAGE = 0%

RULE_BASED_COMPONENT_NOTEBOOK_READY = NO
FUTURE_RAG_BOUNDARY_DOCUMENTED = YES

CANONICAL_CANDIDATE_REGENERATED = NO
REVIEW_WORKBOOK_REGENERATED = NO
FOREIGN_KEY_ISSUES = 0
REGRESSION_FAILURES_INTRODUCED = 0

COMPONENT_40_BASELINE_COMPLETE = NO
READY_FOR_NOTEBOOK_CONSOLIDATION = NO
READY_FOR_FINAL_HUMAN_SMOKE = NO
READY_FOR_PROD_PROMOTION = NO
```

`COMPONENT_INSTANCE_BASELINE_CREATED = PARTIAL` denotes the retained pre-existing
application-level baseline, not new rows created by this blocked package.
