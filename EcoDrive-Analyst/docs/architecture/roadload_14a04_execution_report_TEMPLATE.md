# EcoDrive — Sprint 14A-04 Evidence Report (TEMPLATE)

> **NOT EXECUTED.** Fill this only from the actual repository execution; do not replace placeholders with inferred numbers.

## Scope

| Field | Evidence |
|---|---|
| Commit SHA (pre-refactor) | TODO |
| UTC timestamp | TODO |
| Python / NumPy / pandas / pytest | TODO |
| OS / CPU / RAM | TODO |
| Worktree cleanliness | TODO |
| Database fixture class | Synthetic / public / isolated QA / none |
| Commands | TODO |

## Test results

| Suite | Passed | Failed | Skipped | Evidence |
|---|---:|---:|---:|---|
| Selected physics tests | — | — | — | `selected_tests.xml`, `selected_tests.log` |
| Full suite (optional) | — | — | — | `all_tests.xml`, `all_tests.log` |
| Synthetic golden | — | — | — | `golden_reference.json` |

## Traceability

| Requirement | Evidence tier | Exact assert/test | Result |
|---|---|---|---|
| RL-FR-01/02 | NOT_EXECUTED | TODO | TODO |
| RL-FR-03/05 | GAP (target model) | TODO | TODO |
| RL-FR-06/07 | GAP | TODO | TODO |
| RL-FR-08 | NOT_EXECUTED | TODO | TODO |
| RL-FR-09 | NOT_EXECUTED | TODO | TODO |
| RL-FR-10 | NOT_EXECUTED | TODO | TODO |

## Benchmark (fill real values from bench.json)

| Operation | Workload | p50 wall ms | p95 wall ms | p50 CPU ms | p95 CPU ms | Python traced peak bytes |
|---|---|---:|---:|---:|---:|---:|
| RoadLoad force scalar | 1 | — | — | — | — | — |
| RoadLoad force vector | n=10000 | — | — | — | — | — |
| Canonical VDE series | n=10000 | — | — | — | — | — |
| Legacy scenario batch | m=1000 | — | — | — | — | — |

### Explicit unmeasured scopes

- Roadload component physical breakdown (`k`): not yet implemented on `RoadLoadModel` → **NOT AVAILABLE**, do not invent.
- SQL query counts for real services: **NOT MEASURED** unless separately instrumented.
- UI p95/browser: **NOT MEASURED** unless separately instrumented.
- Memory RSS: **NOT MEASURED** by tracemalloc alone.

## Observed failures and risks

- TODO (attach commands, commits, stack traces and reproduction).

## Gate decision

`NOT READY` until pre-refactor evidence is recorded and reviewed.  
No production refactor is authorized by this template.
