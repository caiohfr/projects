# Sprint 12F.14B — VDE Semantic / Data Architecture Freeze

## Status: `VDE_DATA_ARCHITECTURE_FROZEN — PROCEED_TO_12G_INTEGRATION`

```text
Schema changed?                          NO
VDE rows changed?                        NO
RUN rows changed?                        NO
FuelCons rows changed?                   NO
Runtime DB changed?                      NO

Current VDE structure sufficient?        YES
vde_id_parent sufficient for lineage?    YES
Current mass structure sufficient?       YES
RUN/VDE separation sufficient?           YES
Cold/sensitivity rule representable?     YES
Performance/custom VDE representable?    YES

Application contracts ready for 12G?     PARTIAL
Localized integration changes expected   1
Page changes expected                    0
User decisions required                  0
```

## Freeze decision

The current data model is sufficient for the agreed near-term use cases and is frozen for Sprint 12G integration. VDE remains a wide, resolved and persisted analysis state under a Vehicle Configuration; RUN remains execution/evidence; FuelCons remains the persisted consumption, energy and CO2 result. No current row was collapsed, reassigned or recalculated.

The creation boundary is explicit: create a VDE only when an analysis condition is intentionally saved for later comparison. Temporary sensitivities, display-only calculations and unsaved Quick/What-if operations remain in memory. A real test-conditioned, Cold, performance, road-test or custom state may remain a separate VDE when deliberately persisted.

## Q1–Q7 schema sufficiency

| Question | Answer | Evidence |
|---|---:|---|
| Q1 — Can `vde_id_parent` represent persisted derived families? | YES | Nullable self-FK to `vde.id`, `ON UPDATE CASCADE`, `ON DELETE RESTRICT`; current non-NULL parents: 5336. Parent links remain NULL unless deterministic. |
| Q2 — Can existing mass fields represent performance load conditions? | YES | All 11 required mass/basis fields are present, including test mass, MRO, GVWR, GCWR, payload and trailer mass. The engine consumes resolved numeric mass. |
| Q3 — Can snapshot fields represent a saved performance/custom state? | YES | All 24 inspected roadload/component snapshot fields are present. A child may share its Vehicle Configuration while storing independent resolved state. |
| Q4 — Can RUN own Coastdown/RDE/road/homologation evidence? | YES | RUN has type, evidence kind, procedure, conditions, result details, source identity and provenance fields; VDE keeps resolved A/B/C. Current RUN rows: 29250. |
| Q5 — Can real Cold VDEs coexist with non-persisted sensitivities? | YES | Persisted state fits existing VDE cycle/source/provenance and numeric snapshot fields; interactive sensitivity remains in memory. Rows with explicit Cold/temperature text found by conservative scan: 0; absence of a label is not a schema limitation. |
| Q6 — Can EPA/WLTP continue through wide result fields? | YES | All 10 existing EPA/WLTP aggregate/phase fields are present and remain exposed by `vde_db`. |
| Q7 — Is any current real case unrepresentable? | NO | FK check is clean, no required field group is missing, and the 12F.14A multi-row families already fit as distinct VDE states. No concrete immediate schema gap was found. |

## Parent, performance and custom-state semantics

`vde_id_parent` is optional and is used only for deterministic lineage. The database already contains 1411 Vehicle Configuration families with multiple VDE rows, demonstrating that the 1:N relationship is not blocked by uniqueness. A future saved GVWR/GCWR, GLAMYS, RDE-derived or road-test condition can share `vehicle_configuration_id` with a baseline, reference it through `vde_id_parent`, and persist its own resolved mass, component/roadload state and outputs. No new `load_case`, `scenario_role` or methodology taxonomy is required.

RUN is the concrete evidence/execution owner. A coastdown, route, RDE or simulation RUN does not by itself create another VDE. It supports a new VDE only when the resulting resolved condition is intentionally retained as a comparable engineering state.

## Sprint 12F.14A evidence preserved

```text
Current VDE rows                         11626
Strict physical groups                   11626
Proven exact test-grain duplicates       0
Source-neutral multi-row families        1411
```

No VDE rows were merged. Cycle labels alone were not used as duplicate evidence. RUN and FuelCons ownership remains unchanged.

## Pre-12G application contract inspection

| Surface | Classification | Code evidence | Finding |
|---|---|---|---|
| Browse | `READY_AS_IS` | `src/vde_core/comparison_report_service.py:885` | Read JOIN uses compatibility vde_db/fuelcons_db fields that are present in the candidate. |
| VDE Setup | `LOCALIZED_REPOSITORY_CHANGE_LIKELY` | `src/vde_core/repositories/vde_repository.py:117 and src/vde_core/db.py:716` | Reads are compatible; canonical cutover must route existing legacy-name writes/bootstrap to physical canonical tables in the storage layer. |
| Comparison | `READY_AS_IS` | `src/vde_core/comparison_report_service.py:834` | Comparison reads the preserved compatibility views and consumes the existing wide VDE/FuelCons contract. |
| Quick Scenario | `READY_AS_IS` | `src/vde_core/quick_scenario/resolver.py:92` | The selected VDE is copied and resolved in memory; temporary Quick changes do not persist rows. |
| Powertrain Scenario | `READY_AS_IS` | `src/vde_app/components/pwt_system_scenario.py:264` | The workspace materializes existing VDE/FuelCons snapshots and performs scenario composition outside storage. |
| FuelCons reads | `READY_AS_IS` | `src/vde_core/repositories/fuelcons_repository.py:30` | Required FuelCons and linked VDE fields are exposed by the compatibility views. |

The application contract is `PARTIAL` only because cutover needs one localized storage-layer change: make bootstrap and VDE/FuelCons writes target the canonical physical tables while preserving the legacy compatibility read views. This is not a page, physics or schema redesign. Expected page changes: zero.

## Contract and safety evidence

- Candidate schema signature before/after: `606EEE94D89EFD56B617AA4C208B9F784490E2D6D965F7EF799BD24342C5C339` / `606EEE94D89EFD56B617AA4C208B9F784490E2D6D965F7EF799BD24342C5C339`.
- Candidate SHA-256 before/after: `B8789AB92D9B7A4F7A005006C3A5D341E14B8F6772E63EA603AF406391224D61` / `B8789AB92D9B7A4F7A005006C3A5D341E14B8F6772E63EA603AF406391224D61`.
- Sprint 12F.14A candidate SHA-256: `B8789AB92D9B7A4F7A005006C3A5D341E14B8F6772E63EA603AF406391224D61`; still identical: **YES**.
- Runtime hashes are byte-identical before/after: **YES**.
- Candidate row counts: Program 3117; Vehicle Configuration 10211; VDE 11626; RUN 29250; FuelCons 10822; adoption 18323.
- SQLite foreign-key violations: 0.
- New forbidden architecture tables/columns: none.
- Compatibility SQL and inspected application files remained byte-identical during the pass.

## Minimal semantic guards

The focused suite contains eight guards:

1. `test_01_parent_lineage_contract_is_nullable_self_fk`
2. `test_02_same_configuration_can_have_derived_vde_family`
3. `test_03_mass_and_snapshot_fields_are_preserved`
4. `test_04_vde_population_and_candidate_are_unchanged`
5. `test_05_run_and_fuelcons_populations_are_unchanged`
6. `test_06_no_speculative_architecture_was_introduced`
7. `test_07_compatibility_reads_and_surfaces_are_preserved`
8. `test_08_runtime_databases_are_unchanged`

## Documentation freeze

The canonical PDR now contains a concise **VDE persistence semantics (Sprint 12F.14B freeze)** section. It freezes intentional persistence, conservative parent lineage, Cold/test-conditioned coexistence with in-memory sensitivity, and the authority of resolved numeric mass/roadload state.

## Open gaps and decisions

No semantic/schema gap or user decision blocks 12G. The localized canonical write/bootstrap routing is an integration task, already bounded to the repository/storage layer; it must not be implemented by changing Streamlit pages or Vehicle Demand physics.

## Exit status

`VDE_DATA_ARCHITECTURE_FROZEN — PROCEED_TO_12G_INTEGRATION`
