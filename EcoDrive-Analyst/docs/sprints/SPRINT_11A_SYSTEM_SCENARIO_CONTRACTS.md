# Sprint 11A — SDD Bootstrap, Powertrain Audit & Canonical Contracts

## Status

Complete. This is the first package of Sprint 11 (`docs/specs/sprint_11_multi_domain_system_scenario.md`), implementing Section 44's "SDD Pilot Requirements," Section 3's current-state audit, and Section 25's canonical contracts. No System Scenario calculation, no UI redesign, and no database change were implemented — all explicitly deferred to 11B/11C/11D per the spec's own implementation sequence (Section 48).

## Note on `EcoDrive_Spec_Driven_Development_Guide_v0.1.md`

The task instructions named this file as required reading before starting. It does not exist anywhere in this repository (confirmed by an exhaustive glob across the whole `projects` root, not just `EcoDrive-Analyst`). `docs/specs/sprint_11_multi_domain_system_scenario.md` and `docs/POWERTRAIN_SCENARIO_ARCHITECTURE_GUIDELINES.md` — the other two files named — both already existed and were read in full. Rather than block on the missing file, this sprint proceeded using: (a) the Sprint 11 spec's own Section 44 ("SDD Pilot Requirements") and Section 45 ("Agent Autonomy"), and (b) the explicit AGENTS.md content list given directly in the task instructions. This is stated plainly here rather than silently assumed satisfied.

## SDD bootstrap

- **`AGENTS.md`** (repo root, new) — permanent, cross-sprint rules only: UI-not-physics, canonical ownership (one formula/one owner), the Vehicle Demand/Powertrain boundary, explicit provenance, zero-is-not-missing, synthetic/anonymized QA policy, agent autonomy/escalation, test/closure evidence discipline, no speculative confirmed-bug claims, and AppTest-is-not-manual-smoke. No Sprint-11-specific product requirement was placed here — those all stay in the spec doc, per Section 44's own instruction ("Keep Sprint-specific requirements in this spec").
- **Sprint 11 spec** — already present at `docs/specs/sprint_11_multi_domain_system_scenario.md`; read in full, no deviation found that required editing it (see "Spec deviations" below for the two pre-existing formula-inconsistency findings, neither of which conflicts with the spec itself).
- **Reusable skill** — `.claude/skills/ecodrive-sprint-package/SKILL.md` (new). A generic, sprint-content-free description of this project's established sprint-implementation workflow (ground before designing, canonical Streamlit-free contracts, implement inside the frozen spec with explicit escalation, evidence-classified testing, closure doc). Explicitly not a `sprint-11` skill, per the task's own instruction.

## Current Powertrain ownership map (Section 3 audit)

Produced by three parallel `Explore` agents reading `pages/Powertrain_Scenario.py`, `src/vde_app/components/pwt_fuel_energy.py`, `src/vde_core/fuel_estimation.py`, `src/vde_core/powertrain_efficiency.py`, `src/vde_core/pwt_fuel_energy_service.py`, `src/vde_core/technology_delta.py`, `src/vde_core/ml_prediction.py`, the regression path, and `src/vde_core/vehicle_demand/`. Every claim below has a file:line citation available in the audit transcript; this table is the synthesized answer.

| # | Question | Owner today |
|---|---|---|
| A | Scenario ownership in UI | Fragmented across ~45+ independent `st.session_state` keys in `pwt_fuel_energy.py` (`PWT_DRAFT_RESET_KEYS`, lines 145-204). No canonical scenario object exists; a dict-shaped scenario is *reconstructed fresh every render* by `_build_powertrain_scenario_draft` (`pwt_fuel_energy.py:1775-1868`), never itself persisted to session state. Small UI-embedded arithmetic exists (delta-percent display math, two hardcoded unit-conversion literals: `*0.425143707` for L/100km→gal/100mi at `:2384`, `/1.34102209` for hp→kW at `:3944`) — cosmetic, not estimation physics, which is fully delegated. |
| B | Canonical calculation owners | `fuel_estimation.run_fuel_estimation()` dispatches to `_physics_simple`/`_manual_imported`/`_regression_existing`/`_ml_prediction`; `powertrain_efficiency.build_powertrain_efficiency_summary()` derives PSE uniformly for every method; `technology_delta.apply_delta_stack_to_baseline()` composes deltas. |
| C | Baseline/reference owner | Three distinct, UI-owned layers, none canonical/Streamlit-free: (1) active VDE source (`pwt_active_vde_source` selectbox), (2) "reference" powertrain source (`_selected_powertrain_reference`, `pwt_fuel_energy.py:711-785`, explicitly rebased onto the active VDE, not used verbatim), (3) a separately confirmed/locked baseline snapshot (`pwt_confirmed_baseline_snapshot`, set by a "Confirm baseline" button at `:3651-3666`). |
| D | PSE owner | `powertrain_efficiency.build_powertrain_efficiency_summary()` (`powertrain_efficiency.py:109-203`) — always `demand_mj_per_km / total_consumed_mj_per_km`, never a direct method output; the module itself carries the disclaimer *"PSE is cycle-effective and should not be interpreted as pure engine efficiency."* No UI code reinterprets it as engine efficiency (grep-confirmed). |
| E | Benchmark owner | `pwt_fuel_energy_service.derive_reference_pse()`/`list_benchmark_fuelcons_candidates()` — already centralized in the Sprint 10D/10E pre-flight work this same session. |
| F | ML owner | `src/vde_core/ml_prediction.py:predict_fuel_with_ml` (dispatched from `fuel_estimation._ml_prediction`, `:284-294`). Predicts `fuel_l_100km`/`energy_Wh_km`/`gco2_km` directly — never `pse`; PSE is derived afterward by the same shared `build_powertrain_efficiency_summary` call every method uses. |
| G | Regression owner | `fuel_estimation._regression_existing` (`:263-281`) delegates entirely to a caller-injected `model_options["regression_runner"]` callable; no formula lives in this file. Same as ML, PSE is derived post-hoc, never returned by the runner contract. |
| H | Technology Delta owner | `technology_delta.apply_delta_stack_to_baseline` (`:191-304`) — see the dedicated L0 composition research section below. |
| I | Persistence/save owner | `fuel_estimation.save_fuel_estimate_result` (`:577`) → `repositories/fuelcons_repository.py` (`insert_fuelcons_row`/`update_fuelcons_by_id`). Payload built by `_build_fuelcons_payload`/`_build_provenance_payload` (`:520-556`, `:423-453`); `_proposal_save_overrides` (`pwt_fuel_energy.py:4379-4414`) overwrites the cached `fuel_l_per_100km`/`energy_Wh_per_km`/`gco2_per_km` columns with the **proposal** (post-delta) values, keeping the baseline only inside `provenance_json`. A second, independent write path exists (`render_fuelcons_table`'s inline edit/delete, `pwt_fuel_energy.py:3204-3246`) that bypasses `run_fuel_estimation`/`save_fuel_estimate_result` entirely. `record_origin` is **never explicitly set** by the Powertrain save path — it silently defaults to `'LEGACY'` at the DB level (`db.py:131,165`). |
| J | Vehicle Demand input path | Raw `vde_db`-shaped row dict, via `pwt_fuel_energy_service.build_fuel_estimate_request_from_vde` → `resolve_vde_energy_values` → `canonical_vde_read` (a Package-7G-era helper operating on plain dict rows). **Confirmed zero references, in either direction, between `fuel_estimation.py`/`powertrain_efficiency.py` and `src/vde_core/vehicle_demand/`** — the entire canonical Powertrain layer is coupled to the pre-Sprint-9 raw-row shape today, not to the frozen `VehicleDemandResult` contract. |

No module was refactored merely because it was large (`pwt_fuel_energy.py` is ~5800 lines and was read only for structure/citations, per the spec's own instruction).

## Data-field → domain mapping (Section 4)

Classified from the fields the audit confirmed are actually read today. `SOURCE CONFIGURATION` = imported/observed input; `L0 ASSUMPTION` = an effective/assumed parameter the existing solver consumes; `DERIVED RESULT` = computed output; `PROVENANCE` = metadata about where a value came from; `NOT CURRENTLY REPRESENTED` = no confirmed legacy column. No DB fields were added; missing fields stay missing.

| Domain | Field (legacy source) | Classification |
|---|---|---|
| Vehicle Demand | `vde_total_mj_per_km`/`vde_net_mj_per_km` (`vde_row`, via `canonical_vde_read`) | DERIVED RESULT |
| Vehicle Demand | `phase_outputs` six `vde_*_mj_per_km` keys | DERIVED RESULT |
| Vehicle Demand | `source_vde_revision`/`source_vde_created_at`/`source_vde_updated_at` | PROVENANCE |
| Architecture | `electrification` (**`fuelcons_row`**, not `vde_row` — confirmed by direct query against the QA fixture during this sprint) | SOURCE CONFIGURATION |
| Engine/Fuel Converter | `engine_size_l` (`vde_row`) | SOURCE CONFIGURATION |
| Engine/Fuel Converter | `fuel_type`, `engine_max_power_kw` (`fuelcons_row`) | SOURCE CONFIGURATION |
| Engine/Fuel Converter | `LHV_MJ_per_L`, `gCO2_per_L` (`powertrain_features` overrides) | L0 ASSUMPTION |
| Engine/Fuel Converter | `eta_pt_est` (`powertrain_features`) | L0 ASSUMPTION |
| Transmission/Driveline | `transmission_type` (`vde_row`) | SOURCE CONFIGURATION |
| Transmission/Driveline | `gear_count`, `final_drive_ratio` (`fuelcons_row`, with `vde_row` fallback in the audited code) | SOURCE CONFIGURATION |
| Electric Drive | `bev_eff_drive` (`powertrain_features`) | L0 ASSUMPTION |
| Electric Drive | motor role/count/position/rated-power/rated-torque/identifiers | NOT CURRENTLY REPRESENTED |
| Energy Storage | `battery_capacity_kwh` (listed in `_POWERTRAIN_FUELCONS_FIELDS`, Comparison layer) but not read anywhere in the audited `fuel_estimation.py`/`powertrain_efficiency.py` L0 path | SOURCE CONFIGURATION (present on the row, CONFIGURATION_ONLY at L0) |
| Energy Storage | usable capacity, charge/discharge/regen power limits, SOC window | NOT CURRENTLY REPRESENTED |
| Energy Mgmt/Controls | `utility_factor` (`powertrain_features`, PHEV electric-share) | L0 ASSUMPTION |
| Energy Mgmt/Controls | hybrid operating strategy, start-stop metadata | NOT CURRENTLY REPRESENTED |
| Aux/Thermal | (none found) | NOT CURRENTLY REPRESENTED |
| (result, all domains) | `fuel_l_100km`/`energy_Wh_km`/`gco2_km`, PSE | DERIVED RESULT |
| (all domains) | `assumptions_json`/`provenance_json` (embeds `powertrain_reference`, `baseline_estimate`, `technology_deltas`, `scenario_lineage`, `confidence_summary`, `pse_summary`) | PROVENANCE |

## Canonical contracts (Section 5, `src/vde_core/system_scenario/`)

New Streamlit-free package, structured exactly like `src/vde_core/quick_scenario/` (`contracts.py` / `serialization.py` / a legacy-boundary module / `__init__.py`). No dynamic plugin/domain framework: 8 fixed, small, explicitly-typed configuration dataclasses (`VehicleDemandConfiguration`, `ArchitectureConfiguration`, `EngineConfiguration`, `TransmissionConfiguration`, `ElectricDriveConfiguration`, `EnergyStorageConfiguration`, `ControlsConfiguration`, `AuxThermalConfiguration`), one per `DomainKind`, keeping domain-specific configuration distinguishable by Python's own type system rather than a generic dict or a runtime registry.

- **`DomainKind`** (8 members), **`ArchitectureClass`** (ICE/MHEV/HEV/PHEV/BEV), **`FidelityLevel`** (QUANTITATIVE/EFFECTIVE_ASSUMPTION/CONFIGURATION_ONLY/NOT_REPRESENTED), **`ProvenanceKind`** (SOURCE_OBSERVED/CORRECTED/ASSUMED/CALCULATED/ESTIMATED/ML_PREDICTED/ML_DERIVED/NOT_AVAILABLE).
- **`DomainSourceState`** → **`DomainCorrection`** (optional) → **`EffectiveDomainState`**, connected by the pure function `resolve_effective_domain_state(source, correction=None)`. Every configuration-bearing dataclass validates its `configuration`'s type matches its `domain` in `__post_init__` (`_require_matching_configuration_type`).
- **`DomainProposal`**: `based_on: EffectiveDomainState` — typed so a proposal cannot structurally reference another proposal; `__post_init__` also defensively `isinstance`-rejects a `DomainProposal` passed as `based_on` (REQ-11-007). Carries an optional, explicit `l0_effective_assumption: Mapping[str, float]` (Section 19's "+0.8%" example) and `technology_delta_ids` (association only, never embedded stacking math).
- **`SystemScenarioIdentity`**: `role` (CURRENT/PROPOSAL) + `proposal_index` (1-3, `None` for CURRENT), validated in `__post_init__` — the concrete enforcement of "Current + max 3 Proposals" for one identity. Uniqueness of `(role, proposal_index)` **within one working set of scenarios** is explicitly left to 11B/11C's resolver/orchestrator (documented as deferred below), matching the same "not every constraint belongs at the single-object level" pattern `QuickScenario.slot` used in Sprint 10A.
- **`SystemScenarioDefinition`**: `slots: Mapping[DomainKind, EffectiveDomainState | DomainProposal]` — a System Scenario's Vehicle Demand selection is just its `VEHICLE_DEMAND` slot like every other domain, so "different System Scenarios may use different VDE sources" (REQ-11-002/003) falls out of each definition owning its own independent `slots` mapping, with no special-case code.
- **`FidelityManifest`** / **`ResolvedSystemScenario`** / **`SystemScenarioResult`**: the result/audit shell. `resolve_system_scenario_shell(definition)` is the one function this package provides for turning a definition into a `ResolvedSystemScenario` — deliberately trivial (copies slots verbatim; VEHICLE_DEMAND is QUANTITATIVE when a real result is present, every other populated domain is CONFIGURATION_ONLY, unpopulated domains are NOT_REPRESENTED). This is **not** claimed as 11C's real resolver — it exists only so `ResolvedSystemScenario`/`SystemScenarioResult` are exercisable end-to-end in 11A's own tests, and 11C is expected to replace its fidelity logic with real solver-readiness semantics.
- **`domain_typically_applicable(architecture, domain)`**: a fixed classification lookup (e.g. BEV → Engine/Fuel Converter typically N/A) for Case L — no graph, no topology, purely informational.
- **Serialization**: `to_serializable()` is **reused, not reimplemented**, from `src/vde_core/vehicle_demand/serialization.py` (the numpy-safe original this project already has three copies of by convention — quick_scenario's is a byte-identical copy; system_scenario imports the vehicle_demand original directly since `VehicleDemandConfiguration` nests real `VehicleDemandResult` objects). `*_from_dict()` helpers exist for every contract exercised by the round-trip tests.

## Legacy adapter boundary (Section 8, `legacy_adapter.py`)

Proof only, not a migration — 4 of the 8 domains, chosen because the audit confirmed real legacy columns exist for them:

- `vehicle_demand_domain_state_from_legacy_vde_row(vde_row, source_identity=...)` — reuses the **exact same 3 frozen Sprint 9 functions** Quick Scenario's own resolver already calls (`build_vehicle_demand_request`, `resolve_vehicle_demand_cycle`, `calculate_vehicle_demand`, imported identically to `quick_scenario/resolver.py:53-57` and proven identical by `assertIs` in the test suite) to produce a real `VehicleDemandResult` from a raw row — zero new physics. Verified against the live QA fixture: `vde_id=900001` → `total_summary.vde_mj_per_km ≈ 0.2965`.
- `vehicle_demand_domain_state_from_result(result, source_identity=...)` — a trivial wrap for callers that already hold a computed `VehicleDemandResult`.
- `architecture_domain_state_from_legacy_vde_row(vde_row, fuelcons_row)` — maps the legacy `electrification` column. **A real bug was found and fixed while writing this adapter**: the first implementation read `electrification` off `vde_row`, but a direct query against the live QA fixture showed that column is empty on `vde_row` — `electrification` is a `fuelcons_db` column (confirmed: `fc:900102.electrification == "ICE"`, while `vde_row.get("electrification")` is always absent). Fixed to read from `fuelcons_row`, matching the Engine/Transmission adapters' already-correct pattern; caught by direct testing against the live fixture before any test was even written, not assumed from memory.
- `engine_domain_state_from_legacy_row(vde_row, fuelcons_row)` / `transmission_domain_state_from_legacy_row(vde_row, fuelcons_row)` — populate `fuel_type`/`rated_power_kw` and `transmission_type`/`gear_count`/`final_drive_ratio` respectively, verified against the live fixture.

Electric Drive, Energy Storage, Energy Management/Controls, and Aux/Thermal have no confirmed legacy columns (per the data-field audit above) and are deliberately left unadapted rather than populated from guessed field names.

## Fidelity semantics (Section 7)

`FidelityLevel` has exactly the 4 values the spec requires; no L1/L2/L3 fidelity model was implemented (out of scope, deferred). `FidelityManifest.is_configuration_only_everywhere_quantitative_is_absent` is the one derived helper this package adds, answering "can this manifest currently justify a changed L0 numeric result" without inventing any quantitative behavior itself.

## L0 Technology Delta composition research finding (Section 10/22)

**CASE A — existing canonical semantics are sufficient.**

`technology_delta.apply_delta_stack_to_baseline(baseline_result, *, ctx, deltas)` (`:191-304`) operates on **one shared scalar baseline** (`pse`/`fuel_l_100km`/`energy_Wh_km`/`gco2_km`) and applies `deltas` — a flat, caller-ordered `list[dict]` — strictly in list order via a single `for delta in deltas:` loop. `affected_subsystem` (the field that would name a delta's physical domain) is set once as a free-text default in `normalize_technology_delta` (`:138`) and is **never read inside the stacking loop or the post-loop reconciliation** — confirmed by a full-file grep showing zero other occurrences. This means: **today's function can already take one Engine delta + one Transmission delta + one Controls delta and stack them with zero new math**, provided each is expressed in one of the 7 already-supported `effect_basis` keys — the loop treats every delta identically regardless of which domain produced it.

**Owner and order semantics for 11C**: the owner is `technology_delta.apply_delta_stack_to_baseline`, unchanged. Order is **strictly caller-supplied list order** — there is no domain precedence, no maturity/confidence sort, nothing implicit. Absolute-basis deltas (`pse_delta`, `fuel_delta`, `co2_delta`, `energy_delta`) are pairwise commutative among themselves, but the moment any percent/multiplier-basis delta (`pse_percent_delta`, `pse_multiplier`/`efficiency_multiplier`, `fuel_percent_delta`, `co2_percent_delta`) is mixed into the same stack, the overall result becomes order-dependent (each percent/multiplier step scales the entire running total, including already-applied absolute deltas) — proven live by the existing `test_reversed_order_of_absolute_and_percent_gives_different_result`. **11C's resolver must therefore define an explicit, deterministic order for collecting deltas across domains before calling this function once** (e.g. the fixed 8-domain taxonomy order) — that is an ordering-convention decision for 11C to make and document, not new math, and not something 11A decides on 11C's behalf.

What today's function does **not** support, and would need genuinely new logic for: domain-scoped independent sub-baselines/ledgers, any domain-aware precedence rule, or cross-domain interaction terms (e.g. an Engine gain interacting nonlinearly with a Transmission change). The current model is linear sequential stacking of scalar effects on one shared state — a modeling simplification, not a domain gate. None of that is required for 11A or implied as required by 11C's own spec language ("ensure deterministic explicit ordering"), so this remains a CASE A finding, not a CASE B stop condition.

A pre-existing, already-documented-and-tested quirk was reconfirmed live, unrelated to this finding: a `co2_delta` stacked alongside any PSE/fuel-affecting delta is silently overwritten by the unconditional post-loop fuel→CO2 reconciliation (`technology_delta.py:292-296`) — reproduced exactly as-is per the module's own docstring, not something Sprint 11A touches.

## Database constraint (Section 30)

No schema change, no new table, no removed column. `configuration_type_for(domain)` plus the 8 per-domain configuration dataclasses are the concrete form of INV-11-012 ("canonical System Scenario contracts must not require callers to understand raw `fuelcons_db` schema") — confirmed by a dedicated test asserting the adapter's output dataclass never exposes raw-row-only keys (`id`, `vde_id`, `record_origin`, `assumptions_json`, `provenance_json`).

## Spec deviations / discoveries

No deviation from the spec itself. Two **pre-existing, unrelated formula inconsistencies** were surfaced by the audit and are recorded here as findings, not fixed (fixing either would be new/changed physics, out of 11A's scope, and neither conflicts with the Sprint 11 spec's own requirements):

1. **PHEV CO2 formula disagreement between two "canonical" sites.** `fuel_estimation._physics_simple`'s PHEV branch computes `gco2_km` from the fuel share only; `pwt_fuel_energy_service.compute_ice_fuel_from_vde`'s PHEV branch adds both the fuel-side AND electric-side CO2 terms. These two functions currently disagree on the same physical question.
2. **The already-known, already-tested CO2-delta/fuel-reconciliation overwrite quirk** in `technology_delta.py` (see above) — reconfirmed still present, unchanged.

Both are flagged for a future engineering decision; Sprint 11A does not resolve either.

## Deferred to 11B/11C/11D

Per spec Section 48 and Section 11 of the task: Current/Correction/Effective-Current UI flow and real Domain Proposal editing (11B); System Scenario composition, the `EnergyBalanceL0Adapter`, and a real `SystemScenarioResult` (11C, including the deterministic cross-domain delta ordering this finding calls for); the multi-domain Powertrain UI matrix (11D); QA/manual smoke/traceability freeze (11E). Also explicitly deferred: uniqueness enforcement of `(role, proposal_index)` within one working scenario set (a resolver/orchestrator concern), any real solver-readiness logic beyond the trivial `resolve_system_scenario_shell`, and Comparison integration.

## Tests

- `tests/test_system_scenario_contracts.py` — 48 tests, Streamlit-free: explicit zero != missing (4), stable identity independent of label (3), Current+max-3-proposal constraint (6), independent proposal identities (2), proposal based on Effective Current (2), proposal→proposal rejected (2), different Vehicle Demand per System Scenario (1), shared Domain Proposal reuse across scenarios (1), all 8 domains representable + mismatched-configuration-type rejection (3), Architecture classification (5), fidelity states (4), configuration-only vs quantitative distinguishability (3), source immutability (3), correction→Effective Current (3), serialization round-trip (6).
- `tests/test_system_scenario_legacy_adapter.py` — 11 tests: Vehicle Demand adapter produces a real quantitative result from a legacy row and reuses (via `assertIs`) the identical frozen-core functions Quick Scenario's resolver uses, never mutates the source row; Architecture adapter reads from the correct row and maps all 5 legacy electrification values, with an explicit `NOT_AVAILABLE`/`None` (never a guess) when unavailable; Engine/Transmission adapters populate from the confirmed legacy columns; legacy-adapter isolation (no raw-row-only keys leak into the canonical dataclass).
- Full suite: **1696 tests** (1637 baseline + 59 new), **failures=1, errors=1** — `test_axle_hubs_lookup_snapshot_preserves_boundary_metadata` and `test_component_lookup_provenance_does_not_change_parasitic_math` (both in `test_vde_request_resolver.py`), the same 2 known pre-existing failures every prior sprint in this branch has recorded, reproduced fresh for this closure (not copied forward). Zero new regressions.

## Freeze / handoff statement

`src/vde_core/vehicle_demand/`, `fuel_estimation.py`, `powertrain_efficiency.py`, `pwt_fuel_energy_service.py`, and `technology_delta.py` are all untouched — this package only reads them (directly, or via the frozen adapters they already expose) and adds a new, independent `src/vde_core/system_scenario/` package alongside. No database schema change. No System Scenario calculation exists yet. Do not start Sprint 11B.
