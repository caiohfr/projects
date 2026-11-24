# Sprint 15D implementation report

Date: 2026-11-24

Branch: `sprint-15d-data-methodology-legacy`

Verified base: `origin/main@01382183da0cb67a5116b824fbd487c1315e371c`

## Scope delivered

Sprint 15D completes the code-side visual refinement for Database Management,
Methodology & Models and Engineering Tools / Legacy. The implementation stays
inside the Streamlit presentation layer and reuses the approved visual tokens.

### Database Management

- Preserved the four real entity tabs: VDE, Fuel Consumption, Tires and
  Components.
- Added a factual operation strip with the active database filename and the
  real number of staged changes.
- Reorganized each entity into catalog records and selected-record panels.
- Kept controlled spreadsheet import, staged CRUD, validation, dependency
  impact, review, reason requirements, clear and explicit commit behavior.
- Kept Review & Commit transversal to all four entities; no fictional audit
  tab or automatic write was added.

### Methodology & Models

- Replaced the long static page with a seven-topic internal index covering
  Vehicle Demanded Energy, Roadload & Coastdown, Powertrain L0, Tires &
  Components, Contracts / TOTAL vs NET, Experimental Methods and References &
  Provenance.
- Corrected the visible TOTAL/NET explanation: TOTAL is the authoritative
  coastdown boundary; NET exists only when a resolved transmission-neutral
  loss can be subtracted. Neither boundary silently replaces the other.
- Labelled historical, experimental, planned and unavailable capabilities so
  that renderer existence is not presented as canonical availability.

### Engineering Tools / Legacy

- Added a compact capability map and clearer experimental/compatibility
  warning.
- Preserved `legacy_engineering_area` and lazy loading of exactly the three
  retained areas: Powertrain Legacy, Comparison Legacy and Estimate / Snapshot
  Management.
- Pointed users toward canonical workflows without removing retained tools.

No file under `src/vde_core` was changed. No schema, formula, service contract,
widget key, callback, calculation path or persistence rule was changed.

## Database write invariants

Existing database-management tests still prove that create/update/archive and
spreadsheet operations are represented as staged commands before application,
that validation and impact are reviewed, and that a receipt is logged only
through the existing explicit apply path. The visual layer reads the real
`database_management_staged_commands` list and does not synthesize a status.

One pre-existing contract gap remains: staged commands are not bound to the
database path that was active when they were created. If the runtime DB path is
changed while stage is non-empty, the current contract does not itself prove
that a later commit targets the original database. Per the handoff STOP rule,
Sprint 15D documents this gap and does not change staging or persistence
semantics. A future contract decision should either bind stage to DB identity
or block DB switching until stage is cleared.

## Automated gates

| Gate | Result |
|---|---|
| Syntax (`py_compile`) | PASS |
| New/modified Sprint 15D UI tests | 9 passed |
| Affected Home, Legacy, scenario, Methodology and Database visual tests | 33 passed |
| Historical Database Management group | 41 run: 38 passed, 2 failed, 1 error |
| Full `pytest tests -q` | 2024 passed, 14 failed, 7 skipped; 190 subtests passed |

The historical Database Management result is identical to the verified
baseline: the two failures and one setup error depend on repository-local
canonical QA/staging database files that are absent or empty in this worktree.

The 14 full-suite failures are also fixture/infrastructure failures: missing
canonical QA tables, missing Sprint 12 research input packages, or the legacy
7 MB database lacking canonical `tire_db`. None points to a Sprint 15D changed
module. The full run completed in 724.36 seconds.

`git diff --check` passes. Streamlit continues to emit inherited
`use_container_width` deprecation and mixed-dataframe Arrow warnings in
retained legacy renderers; these are outside the 15D scope and do not fail the
affected tests.

## Visual evidence and gate status

- G0 - PASS: exact base verified; isolated worktree used; dirty personal
  checkout and user databases left untouched.
- G1 - PENDING contract decision: four tabs, real staged state and explicit
  review/commit pass automated tests; staged commands are not DB-path-bound.
- G2 - PASS (automated): seven topics and current TOTAL/NET semantics are
  covered by AppTest.
- G3 - PASS (automated): retained areas, labels and lazy dispatch are covered
  by AppTest.
- G4 - PENDING owner browser review: Streamlit started against an isolated copy
  of the canonical QA database, but the integrated browser returned `The
  browser access host is unavailable`.

No 1440x900 or 1024x768 screenshot is claimed. The reference mockups were used
for hierarchy and styling only; AppTest does not substitute for real-browser
review. Sprint 15 must not be declared finally closed until all eight pages are
reviewed in a browser, accepted visual differences are recorded, and the owner
approves G4.
