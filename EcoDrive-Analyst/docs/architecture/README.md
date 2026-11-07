# EcoDrive — Architecture & Engineering Documentation

**Documentation index (proposed for Sprint 14A, 2026-11-07).** This index supplements existing architecture documentation; it does not replace or silently rename previous files.

## System-wide baseline (already in repository)

- [`project_structure.md`](project_structure.md): repository boundaries and module responsibilities.
- [`vehicle_demand_architecture.md`](vehicle_demand_architecture.md): Vehicle Demand / VDE contracts and canonical physics.
- [`roadload_pipeline.md`](roadload_pipeline.md): classical Roadload `Request → EquivalentABC` pipeline.

## Roadload architecture decision — proposed v0.1

- [`roadload_system_design_v0.1.md`](roadload_system_design_v0.1.md): engineering need → functions → logical software blocks; physics and performance constraints.
- [`roadload_flow_map_v0.1.md`](roadload_flow_map_v0.1.md): `CURRENT` paths vs `TARGET` interfaces for VDE Setup, Load Case and Comparison.
- [`roadload_traceability_v0.1.md`](roadload_traceability_v0.1.md): requirements → owners → verification and evidence tier.
- [`decisions/ADR-14A-005-roadload-model-application-service.md`](decisions/ADR-14A-005-roadload-model-application-service.md): choice C, alternatives, trade-offs and implementation guardrails.
- [`../sprints/SPRINT_14A_ROADLOAD_DESIGN_GATE.md`](../sprints/SPRINT_14A_ROADLOAD_DESIGN_GATE.md): unresolved items and handoff gate.

## Documentation levels (MBSE-lite)

| Level | Question answered | Recommended location |
|---|---|---|
| Need/requirements | Why does the capability exist? What must it do? | Capability system design + traceability |
| Functional | Which transformations happen, independent of implementation? | Mermaid flow in system design |
| Logical/software | Which blocks own each function and contract? | Flow map + ADR |
| Verification | How can we demonstrate correctness and performance? | Traceability + actual test logs |
| Validation | Does the full user workflow solve the intended engineering problem? | Sprint closure / authorized golden cases |

## Rule of change

- `CURRENT` claims require a source commit/path and an evidence tier.
- `TARGET` changes require updating the ADR and affected interface/requirement before implementation.
- Physical formulas and TOTAL/NET semantics cannot change as incidental refactoring.
- Do not claim a test is `PASS` unless the command, commit and environment are recorded.
- Respect legacy/public API compatibility until callers have migrated.
- Prefer a short update to the existing capability doc rather than creating duplicate documents.
- Keep confidential engineering data and internal test programs out of publicly shared examples.

**Scope note:** This v0.1 index covers Roadload first; it does not claim the complete EcoDrive MBSE model exists. Other capabilities should reuse the same structure when their architecture is under review.
