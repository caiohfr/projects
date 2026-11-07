# ADR-14A-005 — Roadload as one engineering model behind application services

**Status:** ACCEPTED AS DESIGN INTENT (no code changes approved yet)  
**Date:** 2026-11-07  
**Context:** Sprint 14A; replaces/expands the earlier local proposed ADR-14A-005 "Roadload as a reusable engineering class"; avoid keeping contradictory versions.

## Problem

Roadload operations are spread between `roadload/models.py`, `roadload/engine.py`, `vde_workflow_service.py`, `vehicle_demand/load_case.py`, and `vde_calc.py`. A reusable physical block is needed for `A+Bv+Cv²`, build-up, decomposition, fractions and reconciliation, but joining DB access, vehicle test conditions, regulatory fitting, VDE integration and Plotly into one class would decrease cohesion.

## Decision

Adopt **option C: Application Service + RoadLoadModel**.

1. Reuse existing `RoadLoadModel` as the single **behavioral** Roadload object. Expose force/power plus future ABC terms, build-up/breakdown/fractions/reconcile as cohesive operations over **already resolved** inputs. Preserve existing constructor and public methods during migration.
2. Keep DB reads, loading by ID, component lookup, applicable roles, history, save workflow, and scenario orchestration **outside** the model, in existing or minimal application services/functions.
3. Keep specialized component calculations in their existing physical owners (`tire_model.py` etc.). DTOs such as `RoadLoadComponent` are allowed; a DTO is not a second behavioral engine. Evaluate absorbing `ComponentSet` only after caller inventory and compatibility tests.
4. Retain `vde_calc.compute_vde_series`/`compute_vde_net` as the canonical VDE path initially. Do not replace with model-method calls without numeric parity, timing and traceability. Maintain existing TOTAL/NET semantics.
5. Support partial physical reconciliation via signed residual. Distinguish it from ABC mathematical terms and a model-built result.
6. Defer 2T fitting, market-specific homologation, dyno target and product UI changes. Do not create `RoadLoadStudy` or a hierarchy of subclasses in this phase.

## Alternatives considered (qualitative, not benchmarks)

| Option | Simplicity | Organization | Runtime potential | Reason for decision |
|---|---:|---:|---:|---|
| A — Keep flows as-is | 8 | 5 | 8 | Low immediate risk, scattered ABC logic |
| B — Universal class including DB | 9 superficially | 3 | 6 | Hides I/O and tightly couples physics to infrastructure |
| **C — Service + RoadLoadModel** | **9** | **9** | **9** | Clean orchestration and pure in-memory model |
| D — Many specialized orchestration classes | 6 | 9 | 8 | Too many abstractions for current scope |

**Warning:** scores indicate anticipated trade-offs, not measured latency. Runtime ranking must be reassessed after profiling, especially tire lookups and Streamlit reruns.

## Consequences

**Positive:** coherent public model, testable numerical operations, DB independence, prospective reuse in notebooks/API, clearer attribution of physical assumptions, reduced duplicate computations.

**Trade-offs:** adapters and backward compatibility cost; resolved component contracts need strong availability/overlap semantics; no immediate performance guarantee; multiple existing flow owners must be migrated incrementally.

**Risks:** `mass_kg` constructor compatibility; `ComponentSet` public usages; accidental double-count of `roadload_total`; sign of negative residual; false NET fallback; implicit tire repository reads; cycles differing in mass resolution and energy aggregation.

## Implementation constraints

- Sprint 14 remains design-only under the architecture/UX roadmap.
- First downstream step is inventory callers and golden/benchmark baseline; then small code PRs by consumer.
- `compute_vde_series` untouched in the first refactor.
- Preserve API via adapters/deprecation wrappers rather than mass imports rewrites.
- No SQL/Streamlit/Plotly inside the model; no implicit saves.
- Any new shared ABC evaluator needs single-owner decision and tests, not duplicated formula scattered through services.

## Verify decision with evidence

Trace [`../roadload_system_design_v0.1.md`](../roadload_system_design_v0.1.md) → [`../roadload_traceability_v0.1.md`](../roadload_traceability_v0.1.md) → tests and benchmarks. Confirm `pytest` suites, SQL counts, numerical parity, and manual UI smoke when implementing. Until then status is **architecture intent**, not implemented or validated.
