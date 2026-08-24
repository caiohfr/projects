# Package 8F - Program Review Redesign (Reference-Optional Comparison)

## Status

Completed and accepted. Builds on the frozen Sprint 8A-8E Comparison Report
(`docs/sprints/PACKAGE_8E_COMPARISON_FREEZE.md`). 8F is additive: it makes the
Reference optional (benchmark-only review becomes a first-class mode),
replaces the old Dashboard tab with a Primary-KPI-driven Program Review tab,
and adds a small fuel-normalization/PSE correctness mini-package on top of
the existing Demand vs Efficiency chart. No prior 8A-8E contract was removed;
`ComparisonRole`, TOTAL/NET semantics, temporary transmission, Physical VDE
Lineage, and the Metric Registry are unchanged.

## Increments implemented

1. **Optional Reference.** `dataset.reference` may be `None`; a
   `SelectionState` with no reference and 2+ comparisons is a legitimate
   benchmark-only review, not an error state. `dataset_items(dataset)` is the
   single canonical "all selected items, Reference first when one exists"
   ordering - every function that previously hardcoded
   `(dataset.reference, *dataset.comparisons)` now routes through it, so the
   Reference-less case is handled in exactly one place. Reference-relative
   delta (compare_metric, Target gap) is meaningless without a Reference and
   is never fabricated when one is absent - absolute-only content renders
   instead.
2. **Presentation roles + Current designation.** A session-only,
   never-persisted overlay (`PresentationState`) keyed by canonical identity
   (`fc:<fuelcons_id>` / `vde:<vde_id>`): `PresentationRole`
   (UNSPECIFIED/PROPOSAL/BENCHMARK) and an independent `current_item_id`
   designation (an item can be PROPOSAL *and* Current at once). This is a
   third axis alongside the canonical `ComparisonRole` and provenance
   (`record_origin`) - never collapsed into either.
3. **Primary KPI + Target.** Session-only `TargetState`, keyed by
   `metric_key` (`targets_by_metric`) so switching the Primary KPI never
   reinterprets a stored target under a different metric's units.
   BETTER/WORSE gap semantics reuse the same `semantic_for_delta()` rule
   `compare_metric()` already uses - no second sign convention.
4. **Versatile KPI Walk.** `WalkStep`/`WalkViewSpec`/`build_walk_rows(...)`
   support multiple delta-base presets (default all-absolute advancing
   anchor, sequential chained deltas, delta-vs-reference with no
   accumulation, explicit per-item delta) plus a benchmark-only mode that
   never fabricates a delta when no Reference exists.
5. **Walk hero chart.** A dedicated Plotly figure in
   `comparison_report_charts.py` visualizing the Walk's ABSOLUTE rows and
   chained deltas.
6. **Program Review tab.** Replaces the old Dashboard tab. Render order: Walk
   hero -> Demand vs Efficiency (FE x VDE, Volumetric/Energy-normalized/
   Electrical modes with equi-PSE guide lines) -> Energy & Demand Summary (a
   compact "Primary KPI + selected VDE boundary + PSE (when available) +
   Target gap (when set)" panel that replaced the old standalone Vehicle
   Demand Status chart).
7. **True cycle/phase VDE.** `build_cycle_phase_rows(dataset, boundary)` is
   genuinely phase-aware (reads actual per-phase cycle results, not a
   TOTAL-only stand-in).
8. **Engineering filters.** `apply_engineering_filters(rows,
   engine_size_l_range=..., engine_max_power_kw_range=...)` - a candidate-
   search tool only (Sec: never mutates or reorders scenario identity); a
   missing field is never coerced to 0, and a range only excludes rows once
   the caller actually narrows it off its default.

## Final accepted state (reference checklist)

**Product identity:** Program Energy & Fuel Economy Review.

**Tabs:**
1. Program Review
2. Energy Drivers
3. Technical Scorecard
4. Explore

**Selection semantics:**
- Reference is optional.
- Comparison selection can contain up to the UI limit (`MAX_COMPARISONS`).
- Filters are candidate-search tools only.
- Changing Make / Category / Legislation / Electrification / Displacement /
  Power / Provenance does NOT invalidate or remove already-selected
  scenarios.
- Selected scenarios may legitimately span different current filter values.

**Presentation semantics:**
- Proposal and Benchmark are presentation roles (`PresentationRole`).
- Current is a designation, not a mutually exclusive scenario role.
- Target is a KPI-specific session overlay, not a scenario.
- No role is inferred from timestamp, provenance, or lineage.

**KPI Walk:**
- Underlying KPI values remain absolute.
- Presentation can be ABSOLUTE or DELTA (`WalkDisplayMode`).
- Delta base can be previous walk state, Reference, or an explicit item
  (`WalkDeltaBase`).
- Context bars do not need to advance the walk anchor.
- The safe default with no walk configuration is absolute comparison
  (`default_walk_steps`).
- Physical VDE lineage is independent from presentation walk order.

**Program Review tab:** Primary KPI, KPI Walk / KPI Comparison (hero chart),
Demand vs Efficiency, equi-PSE guides, compact Energy & Demand Summary.

**Energy Drivers tab:** compact Physical Setup evidence, Roadload Force
Curve, compact ABC evidence, real VDE by regulatory phase, Demanded Power.

**Technical Scorecard tab:** audit/evidence-oriented metric view.

**Explore tab:** custom analysis, explicit physical VDE lineage.

## Sprint 8 final micro-polish - adaptive equi-PSE guides

After the fuel-normalization/PSE mini-package acceptance (below), one small
presentation-only correction was made to Demand vs Efficiency: the equi-PSE
guide lines were hard-coded at fixed values (20/25/30/35% for
volumetric/energy-normalized, 85/90/95% for electrical) regardless of what
was actually plotted, which could visually misrepresent the selected
scenarios' real PSE range. `compute_adaptive_pse_guides()` (new, pure, in
`comparison_report_viewmodels.py`) now sizes the guide set to the PSE values
of the currently plotted points: it snaps the plotted [min, max] PSE span to
a clean 2.5/5/10/20 percentage-point grid, padding by one grid step on each
side so a single or narrow-range point still gets surrounding context, and
keeps the result to 3-5 guides. `_render_fe_vde()` uses this as the primary
source and falls back to the old fixed set only in the (currently
unobserved) case where no plotted point yields a computable PSE -- that
fallback is safe because `build_iso_pse_lines()` remains the sole authority
on whether a line is defensible at all for a given mode/fuel_type and
independently returns no lines for an unmappable basis regardless of which
eta values were passed in.

No PSE physics changed: the adaptive guide values are computed with the
exact same inverted PSE ratio `build_iso_pse_lines()` already draws lines
from (`pse = demand_mj_per_km / consumed_mj_per_km`, per point). All prior
Sprint 8/8F semantics are unchanged by this polish: Tier 2 Cert Gasoline
normalization, canonical/assumed LHV provenance, volumetric fuel-family
compatibility, `established_family` affecting point inclusion only, the
equi-PSE energy basis staying anchor-specific, TOTAL/NET behavior (no
TOTAL/NET fallback), and unresolved/incompatible fuel handling are all
untouched -- an unmappable anchor fuel type still yields zero guides (`()`),
never a fabricated adaptive set, matching `build_iso_pse_lines()`'s existing
"no defensible line" rule.

New tests (`AdaptivePseGuideTests` in `test_comparison_report_viewmodels.py`)
cover: guides adapting to a low-PSE and a high-PSE plotted range, a narrow/
single-point range producing useful surrounding guides, guide count staying
in [3, 5] across ranges, generated line labels matching the computed guide
values, an unresolved/unmappable volumetric fuel type producing zero
fabricated guides, a degenerate non-positive x/y point producing zero
guides rather than a division artifact, and the volumetric-mode guides
reusing the exact same resolved LHV basis as Tier 2 Cert Gasoline vs plain
Gasoline (identical family -> identical guides). Focused Comparison suite
after this change: 289 tests, OK.

## Fuel-normalization / PSE mini-package (final acceptance round)

Scope: the Demand vs Efficiency chart's volumetric-mode fuel-family
resolution and its equi-PSE guide lines, closed out and accepted in a
dedicated final-acceptance pass after the increments above landed.

**Decisions preserved from that acceptance:**

1. `src/vde_core/fuel_energy.py::LHV_MJ_PER_L` is, and remains, the single
   canonical LHV source for everything Comparison/PSE touches
   (`resolve_fuel_energy_basis()`, `build_iso_pse_lines()`,
   `build_fe_vde_points()`). No second table was introduced for this work.
2. The repository still contains two conflicting **legacy** gasoline LHV
   constants outside Comparison's own code path:
   - `src/vde_app/derivatives.py` = 34.2 MJ/L (display-only)
   - `src/vde_app/plots.py`'s `_add_eta_lines_ice` default = 34.2 MJ/L
   These are known technical debt, confirmed still present, and were
   **deliberately not harmonized** in this commit - Comparison canonicalizes
   on `fuel_energy.py` for its own use only; `derivatives.py`/`plots.py` were
   not touched.
3. `resolve_fuel_energy_basis("Tier 2 Cert Gasoline")` was verified against
   live runtime code (not a mock) and resolves as:
   - canonical family: `GASOLINE`
   - fuel spec: `TIER_2_CERT_GASOLINE`
   - LHV: `32.0 MJ/L` (from `LHV_MJ_PER_L["Gasoline"]`, the canonical table)
   - basis: `CANONICAL_ASSUMPTION`, confidence `ASSUMED` - a deterministic,
     traceable mapping of a certification label to the canonical Gasoline
     LHV, never a silent guess and never a second constant.
4. PSE and equi-PSE assumptions must remain visible/traceable to the analyst.
   Verified via a real `AppTest` run of `pages/Comparison_Report.py`: the
   "PSE energy basis: Assumed North America gasoline LHV (canonical
   fuel_energy.py value)" caption renders whenever an ASSUMED (not exact
   spec-reference) basis is in use, and scenario markers (size 10-14, solid,
   star for Reference/circle for Comparison) remain structurally dominant
   over the equi-PSE guide lines (width 1, dotted, muted gray, no legend
   entry) in the actual rendered Plotly figure.
5. `established_family` (in `build_fe_vde_points`) affects **point inclusion
   only** - it is the family of the first item in selection order that
   actually resolves to a known fuel family, so an unmappable anchor (e.g. a
   Flex Reference) excludes only itself instead of poisoning every other
   item's family-compatibility check to "no match" (the bug this mini-package
   fixed: previously a Flex Reference collapsed the whole chart to zero
   points, even alongside a perfectly valid Gasoline comparison). The
   **equi-PSE guide-line basis and the PSE-assumption disclosure caption
   remain anchor-specific** (tied to `items[0]`, i.e. the Reference when one
   is set) - guide lines assert an efficiency basis for the anchor's own
   context and must not be inferred from a different scenario the analyst
   didn't anchor the comparison on. This was confirmed against an existing,
   deliberately-written test
   (`test_demand_vs_efficiency_shows_no_guesswork_message_for_flex_fuel`)
   whose docstring states guide lines must stay absent - with an explicit,
   non-crashing message - for a Flex-fuel Reference even when another
   plotted comparison would otherwise resolve to a known family. An initial
   attempt during acceptance to make the guide-line basis follow
   `established_family` too was found to break that test and was reverted;
   the final diff changes point-inclusion only.

## Visual QA

No browser/screenshot tool is available in this execution environment
(consistent with 8C's and 8E's own findings). Final acceptance instead drove
the real `pages/Comparison_Report.py` through `streamlit.testing.v1.AppTest`
against a real seeded QA SQLite DB and inspected the actual rendered captions
and the actual Plotly figure JSON (trace names, marker sizes/symbols, line
widths/dash/color) for five scenarios: Tier 2 Cert Gasoline Reference +
Gasoline comparison; Flex Reference + Gasoline comparison; Gasoline Reference
+ Flex comparison; an unresolved-first-item benchmark-only walk with two
valid Gasoline items; and Gasoline + Diesel (different valid families). All
five matched intended behavior with zero exceptions. True pixel-level visual
inspection was not performed and should not be read as such - this remains a
manual step for whoever next opens the app in a browser.

## Technical debt register

Carried forward from 8C/8E, still true, not addressed in 8F:
- **Centralize fuel-property/LHV definitions** - `derivatives.py` (34.2) and
  `plots.py`'s eta-line default (34.2) still disagree with the canonical
  `fuel_energy.py` value (32.0) for Gasoline. Explicitly reconfirmed and left
  alone in this package's final acceptance round (see above) - Comparison
  canonicalizes on `fuel_energy.py` for its own use only.
- **Move roadload curve physics fully into core** - `plots.py` still
  contains its own `A + Bv + Cv^2` implementation.
- **Lazy cycle-result resolution** - `build_*_comparison_item` still always
  computes on-demand cycle/phase VDE regardless of UI need.
- **FuelCons repository narrow-column API** - `list_comparison_scenarios`/
  `list_vde_catalog` remain independent of the pre-8A repository functions.
- **True Powertrain/FuelCons scenario lineage does not yet exist** -
  `fuelcons_db` still has no parent field; Physical VDE Lineage remains the
  only lineage domain.
- **Browser-level visual regression testing is not automated** - still true,
  no browser tool available in this environment.

## Known limitations / future work (explicitly outside Sprint 8)

**Engineering / next physical analysis:**
- Rolling Energy, Aero Energy, Positive Inertial Energy, and Other/Residual
  as distinct decomposed energy terms; a richer VDE physical decomposition
  in general.
- Decomposition invariant that any future work in this area must preserve:
  the measured/resolved total roadload remains authoritative. Known
  component contributions must NOT be presented as a complete physical
  decomposition unless they reconcile with that total, and a residual/
  unknown contribution must remain representable rather than silently
  absorbed into a named term.

**Future Powertrain:**
- A richer Powertrain Scenario architecture.
- Operating-point analysis (gear / RPM / torque).
- BSFC / efficiency maps.
- Hybrid supervisory logic at later fidelity levels.

**Future product:**
- Quick Proposal.
- Agent/copilot workflows.
- RAG.
- ML-assisted analytics.
- Automated benchmark / peer analytics.

None of the above is part of Sprint 8's delivered scope; they are recorded
here only so the closure document doesn't imply they were considered and
rejected.

## Regression

Focused (`test_comparison_report_viewmodels`, `test_comparison_report_page_smoke`,
`test_comparison_report_charts`, `test_fuel_energy`, `test_comparison_report_service`):
**289 tests, OK** (includes the 9 new `AdaptivePseGuideTests`).

Full suite (`python -m unittest discover -s tests`), run after the adaptive
equi-PSE guide micro-polish: **1149 tests, 1147 pass, 2 known pre-existing
failures** in `test_vde_request_resolver.py`
(`test_component_lookup_provenance_does_not_change_parasitic_math`,
`test_axle_hubs_lookup_snapshot_preserves_boundary_metadata`) - unrelated to
Comparison, unchanged from the pre-8F baseline (1140 tests, same 2 failures,
before the 9 adaptive-guide tests were added). Zero new failures introduced
by 8F or its micro-polish.
