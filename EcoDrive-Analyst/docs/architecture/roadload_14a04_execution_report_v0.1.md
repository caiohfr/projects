# Sprint 14A-04 evidence report v0.1

## Scope and environment

| Field | Evidence |
|---|---|
| Pre-refactor commit | `fca9a9ad189fe7e6930e987e29fee5717c0d532e` |
| Execution date | 2026-11-09 UTC |
| Runtime | Python 3.11.9; NumPy 2.3.3; pandas 2.3.2; pytest 9.1.1 |
| Platform | Windows 10 build 26100; Intel64 Family 6 Model 154; 12 logical CPUs; RAM probe unavailable without elevation |
| Isolation | clean worktree from `origin/main`; original dirty checkout untouched |
| Data class | synthetic/public inputs for golden and benchmark; repository fixtures for pytest |
| Production changes | none: no physics, DB schema/write path, or UI change |

Local evidence under `artifacts/roadload_14a04/` is deliberately not committed. It includes inventory, golden, benchmark, pytest logs and JUnit XML.

## Results

| Evidence | Result | Detail |
|---|---|---|
| AST inventory | PASS | 168 records; 0 parse errors; manual review of re-exports, dynamic access and notebooks |
| Synthetic golden | PASS | independent asserts for F/P, ISO tire, missing NET and signed negative residual |
| Golden compare | PASS | 0 differences at abs/rel tolerance `1e-9` |
| Selected tests | FAIL | 298 passed, 2 failed, 40 subtests passed; 31.95 s |
| Full suite | GAP / environment | 2,190 passed, 38 failed, 242 errors, 22 skipped, 190 subtests; 475.25 s |
| Microbenchmark | PASS for measured scope | p50/p95 and allocations captured; SQL/UI/RSS/n×k remain gaps |

## Failure classification

The two selected failures reproduce on the clean pre-refactor commit and are unrelated to this docs/scripts-only handoff:

1. `test_axle_hubs_lookup_snapshot_preserves_boundary_metadata` expected mock `AXLE-MOCK-001`; canonical repository selection did not fall back per missing ID, leaving A=120.0 instead of 120.5.
2. `test_component_lookup_provenance_does_not_change_parasitic_math` expected mock `PARA-MOCK-001`; the same selection behavior returned no component snapshot.

The full suite is not reproducible from a clean checkout alone. Many errors require historical generated files under `artifacts/components/` and ETL databases present only in the separate dirty development checkout. Remaining failures are retained in `full_tests.xml`; none were waived or changed. This is a reproducibility blocker, not evidence of a new Roadload physics regression.

## Traceability result

| Requirement | Status |
|---|---|
| RL-FR-01/02 | EXECUTED_PASS: analytical golden and Roadload tests |
| RL-FR-03/05 | PARTIAL / GAP: signed residual current path passes; target breakdown API absent |
| RL-FR-06/07 | GAP: target contract only |
| RL-FR-08 | EXECUTED_PASS for current TOTAL/NET paths |
| RL-FR-09 | EXECUTED_PASS for canonical VDE/Vehicle Demand selected paths |
| RL-FR-10 | EXECUTED_PASS for Load Case no-write/determinism |
| RL-NFR-02 | NOT_MEASURED: SQL count |
| RL-NFR-03/04 | PARTIAL: pure microbenchmark only |
| RL-NFR-06 | EXECUTED_PASS: docs and evidence scripts only |

## Gate decision

**NOT READY FOR PRODUCTION REFACTOR.** Golden and pure performance baselines exist, but the selected suite is not fully green, the full suite cannot be reproduced cleanly, required measurements remain absent, and the overlap/availability API remains a proposal. No Sprint 15 implementation is authorized.
