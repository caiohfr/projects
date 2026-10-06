# Technical Research Capability v0.2.2 — Result

## Outcome

The focused hardware-identity cleanup is complete. The same fixed 15 BMW
applications from v0.2/v0.2.1 were researched with `gpt-5.6-terra` and medium
reasoning. No hardware designation met the new evidence contract, so no repeated
hardware group was confirmed. This is the scientifically correct negative gate;
marketing descriptions were not promoted to component identity.

## Benchmark results

- request quality: 14 `CONSISTENT`, 1 `CONFLICTING_SOURCE_FIELDS`, 0 `INCOMPLETE`;
- identity: 0 `DIRECT`, 0 `STRONG`, 0 `WEAK`, 15 `UNRESOLVED`;
- groups: 0 `HARDWARE_CONFIRMED`, 0 `HARDWARE_SPLIT`,
  0 `PARTIALLY_RESOLVED`, 4 `DESCRIPTIVE_ONLY`, 5 `UNRESOLVED`;
- retrieval: 225 searches, 45 completed rounds, 6 technical attachments,
  3 BMW technical/service sources, 0 supplier sources, 0 retries;
- consolidation: 15 attempts, 15 unique final rows, 9 group rows;
- the only request contradiction is VDE 9234 (`430i xDrive Coupe` with source
  `drive_type=RWD`); both source values are preserved and the drive field did not
  act as a hard veto.

Primary-source technical documents produced partial-application candidates for
5 gear-ratio sets, 4 final-drive ratios, 5 Cd values, 4 frontal-area values, and
5 tire specifications. Because every one had only `PARTIAL` application match,
the strict `INDEPENDENT_*` counters remain zero. Values, units, source IDs, and
support states remain available in the enrichment audit for review.

## Verification

Focused capability suite:

```text
58 passed, 10 subtests passed
```

Broad repository regression:

```text
2163 passed, 184 subtests passed, 6 failed, 29 errors
```

The 6 failures and 29 errors match the pre-existing baseline categories:
missing `openpyxl`/`matplotlib`, historical Sprint 12C contract assertions,
legacy fixed database hashes, and the existing 12F14A status mismatch. No
failure came from `capabilities/technical_research`.

Final canonical database hashes equal the pre-run baseline:

- `data/db/eco_drive.db`:
  `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- `data/db/eco_drive_qa.db`:
  `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- `data/db/staging/eco_drive_canonical_candidate.db`:
  `1E3B251F39B21A835EF9F0B4F6291A495636A8C425CEF7051754D1D750840ACF`

```ini
TECHNICAL_RESEARCH_CAPABILITY_VERSION = 0.2.2

HARDWARE_IDENTITY_CONTRACT_READY = YES
REQUEST_SELF_CONSISTENCY_AUDIT_READY = YES
VEHICLE_ENRICHMENT_TARGETS_READY = YES
FINAL_BENCHMARK_CONSOLIDATION_READY = YES

BMW_BENCHMARK_CONFIGURATIONS = 15

DIRECT_IDENTITIES = 0
STRONG_IDENTITIES = 0
WEAK_IDENTITIES = 0
UNRESOLVED_IDENTITIES = 15

HARDWARE_CONFIRMED_GROUPS = 0
HARDWARE_SPLIT_GROUPS = 0
PARTIALLY_RESOLVED_GROUPS = 0
DESCRIPTIVE_ONLY_GROUPS = 4
UNRESOLVED_GROUPS = 5

INDEPENDENT_FDR_VALUES = 0
INDEPENDENT_GEAR_RATIO_SETS = 0
INDEPENDENT_CD_VALUES = 0
INDEPENDENT_FRONTAL_AREA_VALUES = 0
INDEPENDENT_TIRE_SPECS = 0

REPEATED_HARDWARE_GROUP_READY = NO
READY_FOR_CROSS_OEM_BENCHMARK = NO
READY_TO_RE_RUN_TRANSMISSION_SIGNAL_EXPERIMENT = NO

CANONICAL_WRITE_DISABLED = YES
PRODUCTION_DB_CHANGED = NO
```
