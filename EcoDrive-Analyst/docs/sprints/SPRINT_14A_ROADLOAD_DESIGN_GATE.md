# Sprint 14A — Roadload Architecture Documentation Gate

**Status:** DESIGN DRAFT, 2026-11-07. No production code or DB changes. Complements, does not replace, the broader Sprints 14–18 roadmap.

## Deliverable

A versioned **MBSE-lite design baseline** for Roadload with:

1. Stakeholder need, functional blocks, logical/software allocation: [`../architecture/roadload_system_design_v0.1.md`](../architecture/roadload_system_design_v0.1.md)
2. Current and target vertical slices: [`../architecture/roadload_flow_map_v0.1.md`](../architecture/roadload_flow_map_v0.1.md)
3. Requirement/function/test matrix and acceptance cases: [`../architecture/roadload_traceability_v0.1.md`](../architecture/roadload_traceability_v0.1.md)
4. Rationale/alternatives/constraints: [`../architecture/decisions/ADR-14A-005-roadload-model-application-service.md`](../architecture/decisions/ADR-14A-005-roadload-model-application-service.md)

## Status by evidence tier

| Gate | Status | Comment |
|---|---|---|
| C selected, logical architecture documented | DOCUMENTED | User design decision; still requires formal review |
| Current call paths inspected | INSPECTION ONLY | Main branch at 2026-11-07, no dynamic tracing |
| Requirements & invariants drafted | PROPOSED | IDs and tolerances not baselined |
| Complete callers/public API inventory | GAP | Add AST/import graph for existing `RoadLoadModel`, `ComponentSet`, `EquivalentABC`, `_force_at` and `vde_calc` |
| Component ownership/overlap contract | GAP | Trans vs axle/hubs, source TOTAL/NET, physics validation |
| Golden VDE regression baseline | GAP | Existing tests identified, execution not done in this review |
| Performance baseline | GAP | p50/p95, time/CPU/memory, SQL query count, scenario scaling |
| Decision on abstraction/compatibility | PROPOSED | Preserve constructor and DTO, avoid universal engine |
| Ready for code refactor | **NOT YET** | Gate above must be addressed |

## Next actions (in order)

- [ ] Extract full caller map, identify public imports and precise payload conversion points; confirm whether overlapping calculations are shared code or separate semantics.
- [ ] Freeze golden input/output fixtures for `RoadLoadModel`, `EquivalentABC`, VDE Setup preview, Load Case fixed/sweep/cycle, Comparison TOTAL/NET; no confidential samples in public repo.
- [ ] Specify formal contracts for `RoadloadComponent` including basis, provenance, availability and overlap handling.
- [ ] Benchmark current paths: 1/10/100/1000 vehicles, scalar/vector force, n×k breakdown, SQL queries including tire lookup; record hardware/versions.
- [ ] Finalize tolerance, error handling, signatures (scalar/array, zero-denominator fractions, immutable scenarios).
- [ ] Approve ADR + requirements mapping, then issue a **separate** implementation spec for small PR(s).
- [ ] Preserve the broader roadmap's high-priority **Manual/Imported FuelCons** capability; Roadload refactoring must not silently reorder that work.

## Definition of Done — documentation only

A reviewer can identify **why RoadLoadModel exists, which functions and requirements it owns, where the DB boundary lives, who calls whom, and how parity/performance will be evaluated**. Any unverified claim is labeled; code modification requires a separate gate.

## Change policy

`CURRENT` changes with source inspection of a specific commit; `TARGET` changes via ADR/spec; tests change with a recorded test execution. Keep these artifacts intentionally lightweight and review them only when contracts or flows change. Do not update every document for unrelated UI changes.
