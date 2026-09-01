# Sprint 11B — Current / Effective Domain States & Domain Proposals

## 1. Summary

Sprint 11B makes the domain layer 11A established operational: real `EffectiveDomainState` resolution (already present as contracts in 11A, now exercised end-to-end), a Streamlit-independent `resolve_domain_proposal()`/`changed_fields()` service layer, an expanded legacy-adapter boundary covering 7 of 8 domains from real, schema-confirmed columns (Electric Drive remains genuinely unadapted — no configuration-level column exists for it today), a 3-state `DomainApplicability` (REQUIRED/OPTIONAL/NOT_APPLICABLE) classification, and real canonical-Technology-Delta binding on a proposal (association only, no stacking). No System Scenario composition, no L0 calculation, no UI, and no database change. Sprint 11C was not started.

## 2. Branch/base verification

Per the mandatory pre-flight: `main` (`828da134`) does **not** contain Sprint 8/9/10 at all — confirmed by `git ls-tree -r main --name-only` returning zero matches for `src/vde_core/vehicle_demand` or `src/vde_core/quick_scenario`. Sprint 11A's `contracts.py` directly imports `VehicleDemandResult` from the Sprint 9 Vehicle Demand Core, so the task's literal suggestion (cherry-pick `d121abd6` onto a fresh branch off `main`) would have broken immediately with `ModuleNotFoundError`. This exact ambiguity is what the task's own Section 0 said to stop and report rather than force. Reported to the user with the git graph evidence; the user chose to create `sprint-11-system-scenario` **at the current `HEAD`** (`d121abd6`) — no rewriting, no cherry-pick, no reset — since that branch already is `main` + Sprint 8/9/10 + 11A in one clean, verified-linear history (`git merge-base --is-ancestor main HEAD` → true). Sprint 11A (`d121abd6`) is preserved exactly, byte-for-byte, as the branch's tip commit before any 11B work began.

## 3. Files created/modified

- `src/vde_core/system_scenario/contracts.py` (modified, additive): `DomainApplicability` enum + `domain_applicability_for()`; `AuxThermalConfiguration` gained `ambient_temp_c`/`ac_on` (real, schema-confirmed columns found this sprint); `DomainProposal.technology_delta_ids: tuple[str,...]` replaced with `technology_deltas: tuple[TechDeltaAssumption,...]` (reusing the existing canonical dataclass, not a second schema — confirmed unreferenced by any test before changing it) and a new `requested_changes: Mapping[str, Any]` provenance field.
- `src/vde_core/system_scenario/legacy_adapter.py` (modified): expanded Engine (`engine_family_id`, `rated_torque_nm`, `technology_descriptors`) and Transmission (`transmission_model_id`); added Energy Storage, Controls, Aux/Thermal adapters (all from newly schema-confirmed columns) and an explicit `electric_drive_domain_state_sparse()`.
- `src/vde_core/system_scenario/domain_resolution.py` (new): `resolve_domain_proposal()`, `changed_fields()`.
- `src/vde_core/system_scenario/serialization.py` (modified): round-trip support for the renamed/added fields.
- `src/vde_core/system_scenario/__init__.py` (modified): re-exports.
- `tests/test_system_scenario_domain_resolution.py` (new, 32 tests).
- `tests/test_system_scenario_legacy_adapter.py` (modified, 11 → 20 tests: expanded/new adapters, both sparse and real-data cases).
- This document.

## 4. Effective Current architecture

Unchanged from 11A's own design (verified, not redesigned): `DomainSourceState → [optional DomainCorrection] → EffectiveDomainState`, connected by the pure `resolve_effective_domain_state(source, correction=None)`. 11B did not need to alter this — it was already fully operational as a contract; 11B's job was exercising it thoroughly (see §13) and building the layer above it (`DomainProposal` resolution).

**Immutability**: `source` is never assigned to inside `resolve_effective_domain_state` — Python-level immutability is enforced by every contract being a frozen dataclass, verified directly with `test_source_dataclass_is_frozen` (raises on attribute assignment) and `test_correction_does_not_mutate_source_configuration`/`test_case_a_transmission_correction` (Correction FDR 3.73→3.70 example from the spec, verbatim).

**Provenance**: `EffectiveDomainState.provenance` is `SOURCE_OBSERVED` when no correction was applied, `CORRECTED` (or whatever the correction's own `provenance` says) when one was. This is a **coarse, whole-configuration-level** distinction, not per-field lineage — a deliberate choice matching the spec's own "avoid a huge field-lineage framework... a small inspectable representation is enough" (Sec 22). `changed_fields()` (§8 below) is the actual per-field inspection tool, computed on demand by diffing two configuration objects rather than tracked continuously.

## 5. Correction semantics

`DomainCorrection` carries `domain`, the FULL corrected `configuration` (built via `dataclasses.replace(source.configuration, **changed_fields)` at the call site, never a partial-patch object), `reason`, and `provenance`. Only **one** correction is representable per `resolve_effective_domain_state` call — the contract has no "correction stack" concept. This is a deliberate design limit inherited unchanged from 11A, not something 11B added or removed; item 3 of the required test list ("multiple corrections if supported by existing contract") is therefore **N/A by design**, not a gap — documented here rather than silently skipped. A future sprint that needs sequential/stacked corrections would need a genuine contract extension, which is out of 11B's scope.

## 6. Legacy field → canonical-domain mapping changes

A direct `PRAGMA table_info()` query against the live schema (not memory or assumption) found real, populated-in-principle columns 11A's audit missed:

| Domain | New/expanded fields | Column(s) | Confirmed by |
|---|---|---|---|
| Engine/Fuel Converter | `engine_family_id`, `rated_torque_nm`, `technology_descriptors` | `vde_db.engine_model`, `fuelcons_db.engine_max_torque_nm`, `vde_db.engine_type`/`engine_aspiration` | schema query + `test_engine_adapter_populates_expanded_fields` (non-null augmented row) |
| Transmission/Driveline | `transmission_model_id` | `vde_db.transmission_model` | schema query + `test_transmission_adapter_populates_model_id` |
| Energy Storage | full adapter (new) | `fuelcons_db.battery_capacity_kwh`/`battery_usable_kwh`/`bms_discharge_limit_kw`/`bms_regen_limit_kw`/`bms_note` | schema query + `test_energy_storage_adapter_populates_real_values` |
| Energy Mgmt/Controls | `utility_factor_pct` | `fuelcons_db.utility_factor_pct` (confirmed **persisted**, correcting 11A's assumption it was request-only) | schema query + `test_controls_adapter_populates_utility_factor` |
| Aux/Thermal | `ambient_temp_c`, `ac_on` (new contract fields) | `fuelcons_db.ambient_temp_c`/`ac_on` (correcting 11A's closure-doc claim of "no confirmed legacy columns") | schema query + `test_aux_thermal_adapter_populates_real_values` |
| Electric Drive | none | **no confirmed column exists** — `bev_eff_drive` is an L0 efficiency ASSUMPTION, not motor configuration, and is deliberately not placed into `ElectricDriveConfiguration` (Sec 6: never put an assumption into configuration merely because a row contains it) | schema query (exhaustive: neither table has any motor role/count/position/power/torque/voltage/identifier column) |

The base QA fixture seeds every one of these new columns as `NULL` for every row (confirmed by direct `SELECT`) — the adapter test suite therefore augments a copy of the seeded database with real values via a direct SQL `UPDATE` in `setUp` (the same established pattern `test_comparison_report_page_smoke.py` already uses for scenario-specific fixture data) so the new adapters are proven to move real data, not just correctly return `None`.

## 7. Architecture applicability rules

`DomainApplicability` (REQUIRED/OPTIONAL/NOT_APPLICABLE) + `domain_applicability_for(architecture, domain)`, a fixed lookup table, not a rules engine:

- **VEHICLE_DEMAND**/**ARCHITECTURE**: REQUIRED for every architecture (unconditional).
- **ICE**: Engine REQUIRED; Electric Drive/Energy Storage NOT_APPLICABLE; everything else OPTIONAL (default).
- **MHEV/HEV/PHEV**: Engine, Electric Drive, AND Energy Storage all REQUIRED (both thermal and electric paths present); everything else OPTIONAL.
- **BEV**: Engine NOT_APPLICABLE; Electric Drive/Energy Storage REQUIRED; everything else OPTIONAL.
- **Transmission/Driveline, Controls, Aux/Thermal**: OPTIONAL for every architecture (the spec's own broad semantics never call these REQUIRED or NOT_APPLICABLE for any architecture by name) — verified explicitly by `test_controls_and_aux_thermal_remain_optional_not_not_applicable_everywhere`.

Purely advisory: nothing in this module or `resolve_effective_domain_state`/`resolve_domain_proposal` consults applicability to block construction — `test_applicability_is_purely_advisory_missing_engine_data_does_not_raise_for_bev` and Case E/F/L directly prove a REQUIRED-but-missing or NOT_APPLICABLE-but-present domain never raises.

## 8. Domain Proposal architecture

`resolve_domain_proposal(identity, based_on, requested_changes=None, *, label=, l0_effective_assumption=, technology_deltas=, notes=)` in the new `domain_resolution.py`:

- **Identities**: `DomainProposalIdentity(domain, proposal_id)`, unchanged from 11A — stable, independent of `label` (visible name).
- **Base semantics**: `configuration = dataclasses.replace(based_on.configuration, **requested_changes)` — unrequested fields inherit verbatim (dataclasses.replace's own behavior, not a bespoke merge engine); an invalid field name raises `TypeError` immediately (fail loud, never silently ignored). `identity.domain` must equal `based_on.domain`, enforced by `DomainProposal.__post_init__` itself.
- **Isolation** (Case G): constructing further proposals from the same `EffectiveDomainState` never alters an earlier proposal or the Effective Current object itself — proven directly.
- **Reuse** (Case H): the identical `DomainProposal` object, referenced by two different `SystemScenarioDefinition.slots` mappings, is the same object (`assertIs`), never copied or mutated.
- **`changed_fields(proposal)`**: diffs `proposal.configuration` against `proposal.based_on.configuration` field-by-field directly (not trusting `requested_changes`), so it stays correct even for a `DomainProposal` built without going through `resolve_domain_proposal`.

## 9. Configuration vs L0 representation

Unchanged principle from 11A, exercised directly this sprint: `l0_effective_assumption` is an explicit, caller-supplied `Mapping[str, float]` that is **never** derived from `requested_changes`/`configuration` — Case B (Engine 2.0L/200kW → 1.6L/180kW) and Case D (battery 1.0→1.5kWh) both prove the resulting proposal's `l0_effective_assumption` stays empty and `resolve_system_scenario_shell`'s fidelity for that domain stays `CONFIGURATION_ONLY`, never `QUANTITATIVE`, unless a quantitative representation is explicitly supplied (Case C).

## 10. Technology Delta integration

**Exact canonical owner reused**: `src.vde_core.quick_scenario.contracts.TechDeltaAssumption` (Sprint 10A/10D's own dataclass) — imported directly into `system_scenario.contracts`, not redefined. `DomainProposal.technology_deltas: tuple[TechDeltaAssumption, ...]` preserves `affected_subsystem`/`effect_basis`/`effect_value`/`source_type`/`maturity_level`/`confidence` verbatim (all native fields of the reused dataclass — nothing dropped).

**How association works**: purely storage — `resolve_domain_proposal(..., technology_deltas=(delta_a, delta_b))` stores the tuple exactly as given; nothing combines or evaluates them.

**Ordering metadata**: local order within one proposal's tuple is preserved exactly as supplied (a plain tuple, never a set/dict) — `test_multiple_technology_deltas_are_preserved_unstacked_in_local_order` proves two deltas keep their given order and both remain independently inspectable.

**Confirmation no stacking occurred**: `apply_delta_stack_to_baseline` (the canonical stacking function) is never imported anywhere in `src/vde_core/system_scenario/` — confirmed by grep and by `test_system_scenario_package_never_imports_the_delta_stacking_function` (asserts the symbol doesn't exist on any module in this package). 11C's deterministic cross-domain ordering convention (per the 11A CASE A finding) is not decided here — 11B only ensures every proposal carries unambiguous `(domain, identity)` and a locally-ordered `technology_deltas` tuple for 11C to build on.

## 11. Provenance representation

`ProvenanceKind` unchanged from 11A. Effective Current: coarse (`SOURCE_OBSERVED` vs `CORRECTED`), not per-field — see §4. Domain Proposal: `requested_changes` (which fields were explicitly overridden, populated by `resolve_domain_proposal`) plus `based_on` (the Effective Current inheritance link) plus `l0_effective_assumption`'s own implicit "explicitly supplied or absent" state serve the same purpose without a dedicated lineage framework, matching the spec's explicit "small inspectable representation is enough" (Sec 22).

## 12. Sparse-domain behavior

Electric Drive (no confirmed column at all) and Aux/Thermal-without-a-row (confirmed columns exist but a given row may still be null) both resolve to valid, all-`None`-configuration `DomainSourceState`/`EffectiveDomainState` objects with `provenance=NOT_AVAILABLE`, never an exception and never a fabricated default — proven by `test_electric_drive_sparse_adapter_is_valid_and_explicit`, `test_aux_thermal_adapter_sparse_when_no_row_supplied`, `test_energy_storage_adapter_sparse_when_no_row_supplied`, and Case J end-to-end (through `resolve_system_scenario_shell`, fidelity resolves to `CONFIGURATION_ONLY`, no raw exception).

## 13. Tests

- Pre-11B baseline: **1696 tests, failures=1, errors=1** — the same fresh measurement taken immediately after committing 11A this session (§2: zero code changed between that measurement and the start of 11B work, so it was not re-run as a separate redundant step; this is stated explicitly rather than silently assumed unchanged from an older doc).
- Focused (new/changed this sprint): `tests/test_system_scenario_contracts.py` (48, unchanged, still green after 11B's additive changes), `tests/test_system_scenario_legacy_adapter.py` (11 → 20, +9 from 11B), `tests/test_system_scenario_domain_resolution.py` (32, new) — **100 total for the `system_scenario` package**, all passing. `tests/test_quick_scenario_*` (144 tests across contracts/resolver/efficiency_resolver) re-run to confirm the new `system_scenario → quick_scenario` import direction introduced no regression there — all pass.
- Full suite: **1737 tests** (1696 baseline + 41 new: 100 total `system_scenario` tests vs. 59 that existed after 11A), **failures=1, errors=1** — the same 2 known pre-existing failures (`test_axle_hubs_lookup_snapshot_preserves_boundary_metadata`, `test_component_lookup_provenance_does_not_change_parasitic_math`, both `test_vde_request_resolver.py`), reproduced fresh for this closure. Zero new regressions.
- Failures/regressions: none new.

## 14. Findings requiring 11C attention

1. **Deterministic cross-domain Technology Delta ordering** (carried from 11A's CASE A finding, unchanged): `apply_delta_stack_to_baseline` accepts cross-domain deltas mechanically today (treats `affected_subsystem` as pure metadata), but 11C's resolver must define an explicit, deterministic convention for the order in which one System Scenario's per-domain `technology_deltas` tuples are concatenated into one stack before calling it. 11B does not choose this order — it only ensures every proposal's local tuple order is preserved and each proposal has an unambiguous `(domain, identity)`.
2. **PHEV CO2 canonical-owner discrepancy** (carried from 11A, unchanged, not fixed here per explicit instruction): `fuel_estimation._physics_simple`'s PHEV branch computes `gco2_km` from the fuel share only; `pwt_fuel_energy_service.compute_ice_fuel_from_vde`'s PHEV branch adds both fuel- and electric-side CO2. 11C must resolve which is canonical before using the new L0 adapter for PHEV parity — neither was chosen as "new truth" in 11B.
3. **System Solver readiness**: 11A's `resolve_system_scenario_shell` remains the only resolver in existence, and it is deliberately trivial (VEHICLE_DEMAND is QUANTITATIVE when a result is present, everything else populated is CONFIGURATION_ONLY). 11C needs real solver-readiness logic distinguishing domain-data-completeness from solver-readiness (Sec 24), which 11B does not attempt.

## 15. Spec deviations/conflicts

None. Two small, evidence-justified additive contract changes were made (§3) under the spec's own "small additive contract changes are allowed only when 11B behavior cannot be represented correctly otherwise" allowance — neither redesigns an existing concept, both are documented above with their exact justification.

## Deferred to 11C/11D/11E

System Scenario composition (`SystemScenarioDefinition` → real resolver), the `EnergyBalanceL0Adapter`, real `SystemScenarioResult` computation, the deterministic cross-domain Technology Delta order, PHEV CO2 canonical-owner resolution, and real solver readiness (11C); the multi-domain Powertrain UI (11D); QA/manual smoke/traceability freeze (11E).

## Freeze / handoff statement

No database schema change. No System Scenario calculation. No UI touched (`pages/Powertrain_Scenario.py` and `pwt_fuel_energy.py` are both untouched — confirmed by this sprint's own diff). `src/vde_core/vehicle_demand/`, `fuel_estimation.py`, `powertrain_efficiency.py`, `pwt_fuel_energy_service.py`, and `technology_delta.py` remain untouched. Do not start Sprint 11C.
