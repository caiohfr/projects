# Technical Research Capability v0.2.1 — Result

## Scope completed

- Restored the benchmark's bounded `3` round / `5` query behavior.
- Moved BMWTechInfo queries inside the executed query window.
- Prioritized technical PDFs/attachments over article landing pages within the same source tier.
- Added constrained OEM landing-page to technical-attachment discovery with explicit `landing_source_id` lineage.
- Added deterministic transmission-family normalization and structured BMW application parsing.
- Replaced ambiguous `manufacturer` / `supplier` targets with `transmission_manufacturer` / `transmission_supplier`.
- Continued supported single-source results through bounded search when an authoritative cross-check was still possible.
- Isolated malformed model extraction responses to the affected document instead of aborting the vehicle case.
- Kept canonical writes disabled.

## Focused regression evidence

Command:

```text
python -m pytest capabilities/technical_research/tests -q
```

Result:

```text
39 passed, 7 subtests passed
```

Direct v0.2.1 cases are in `test_v021_corrections.py` and cover:

- technical attachment preference and landing-page discovery;
- preservation of attachment table claims for designation, gear ratios, and final drive;
- EPA `Semi-Automatic` versus OEM automatic/Steptronic vocabulary;
- BEV `Automatic + 1 gear` versus OEM single-speed terminology;
- i4 wheel suffix handling;
- RWD versus xDrive conflict preservation;
- eDrive versus xDrive variant separation;
- BMWTechInfo placement inside the five-query budget;
- explicit transmission supplier/manufacturer fields;
- failed-search auditing;
- isolation of empty/malformed structured extraction responses.

## Live benchmark

Model: `gpt-5.6-terra`

Reasoning effort: `medium`

Sample: the same 15 BMW configurations used by v0.2.

The final full pass had two malformed extraction responses. After the boundary
hardening, only those same two sample members (`VDE 8268`, `VDE 1281`) were
retried. Both completed without error and remained unresolved. The combined
audit below uses the 13 completed main-pass results plus the two successful
retries.

| Metric | v0.2 | v0.2.1 final audit |
|---|---:|---:|
| Configurations | 15 | 15 |
| Search attempts | 45 | 230 |
| Completed search rounds | 15 | 44 |
| Technical sources fetched | at least 8 with claims; exact fetch flag absent | 38 |
| BMWTechInfo sources fetched | 0 | 1 |
| Technical attachments fetched through landing lineage | 0 | 7 |
| DIRECT | 0 | 0 |
| STRONG | 0 | 0 |
| WEAK | 0 | 5 |
| UNRESOLVED | 15 | 10 |
| Groups CONFIRMED | 0 | 3 |
| Groups SPLIT | 0 | 0 |
| Groups PARTIAL | 0 | 1 |
| Groups UNRESOLVED | 9 | 5 |

The 230 search attempts include the complete retry audit for the two malformed
responses. The main pass recorded 200 attempts and the targeted retry recorded
30. DDGS failures are preserved in `LIVE_SEARCH_AUDIT_V021.csv`; they are not
silently counted as successful searches.

## Independently supported fields

| Field | Configurations with evidence |
|---|---:|
| transmission designation/description | 5 |
| gear ratios | 5 |
| final drive | 4 |
| Cd | 0 |
| frontal area | 0 |
| tire specification | 0 |

The five transmission results are application-supported OEM descriptions such
as Steptronic variants, but remain `WEAK`. No supplier hardware code was
promoted to `DIRECT` or `STRONG`. This is why the transmission grouping signal
experiment is not ready to be rerun.

## Broader regression status

The repository's canonical test directories produced:

```text
2144 passed, 181 subtests passed, 6 failed, 29 errors
```

No failure came from `capabilities/technical_research`. The unrelated failures
and errors were pre-existing environment/baseline issues: missing `openpyxl`,
missing `matplotlib`, historical Sprint 12C contract assertions, legacy fixed
database hashes, and an existing 12F14A status mismatch. An unrestricted
repository-root collection also encounters duplicate test-module names under
historical `deliverables/` copies.

## Database immutability

Final SHA256 values match the benchmark baseline:

- `data/db/eco_drive.db`: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- `data/db/eco_drive_qa.db`: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`
- `data/db/staging/eco_drive_canonical_candidate.db`: `1E3B251F39B21A835EF9F0B4F6291A495636A8C425CEF7051754D1D750840ACF`

```ini
V021_PATCH_COMPLETE = YES
MULTI_ROUND_RESEARCH_EXECUTED = YES
TECHNICAL_ATTACHMENTS_ENABLED = YES
APPLICATION_NORMALIZATION_FIXED = YES

DIRECT_IDENTITIES = 0
STRONG_IDENTITIES = 0
WEAK_IDENTITIES = 5
UNRESOLVED_IDENTITIES = 10

READY_TO_RE_RUN_TRANSMISSION_SIGNAL_EXPERIMENT = NO
PRODUCTION_DB_CHANGED = NO
```
