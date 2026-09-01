# Sprint 11G — Current Baseline Readiness

## Scope

Sprint 11G makes the System Scenario workspace establish an explicit Current
baseline before composing proposals. It fixes the source-loading desynchrony
that caused a `KeyError` when Current was changed and an internal domain was
opened. No solver, physical model, database schema, or persistence behavior was
changed.

## Root cause and reproduction

The workspace previously kept a hidden page-level anchor VDE as the detailed
source-loading seed. Changing the Current draft changed `draft.vde_id`, but the
working set retained the old anchor and only added proposal source IDs. Opening
an internal domain then indexed `sources[draft.vde_id]` for a VDE that had not
been materialized, producing the observed `KeyError: 4997` class of failure.

The regression test changes Current from VDE 900008 to VDE 900001, opens the
Engine domain, and calculates Current. It confirms there is no exception, the
selected vehicle-demand identity is `vde:900001`, and the canonical result is
READY.

## Canonical source truth and loading

`SYS-CURRENT`'s `ScenarioDraft.vde_id` is now the source of truth for Current.
The legacy session value is read only as an initial visible selector default;
it no longer drives materialization. The detailed working set is the Current
draft plus the active proposal drafts, so it is bounded by the four scenarios
available in the workspace. VDE discovery remains lightweight and supplies
selector labels only.

The domain editor also handles a missing materialized source with an explicit
error rather than raising a dictionary lookup error.

## Baseline interaction

The workspace begins with a searchable **Current baseline** selector. A new
session without a valid context selects nothing implicitly. Changing a baseline
requires explicit confirmation because it resets domain proposals and calculated
results. Scenario identities and visible proposal labels are retained, while
each proposal returns to `INHERIT` from the new Effective Current.

The Current column now summarizes the effective configuration rather than
repeating "Effective Current". It exposes Vehicle Demand TOTAL where available,
reports Current's canonical L0 readiness and issues, and labels unrecognized
observed architecture as **Assumed ICE** instead of concealing the assumption.

Technology Delta choices now exclude `energy_percent_delta`, which is not a
supported canonical resolver effect basis.

## Evidence

Focused source-loading and System Scenario UI suites passed:

```text
python -m unittest tests.test_powertrain_system_scenario_source_loading tests.test_powertrain_system_scenario_ui
Ran 16 tests
OK
```

The focused coverage includes the changed-Current source working set, the
KeyError regression path, baseline-reset behavior, visible `INHERIT`, unknown
architecture disclosure, and supported Technology Delta options. `compileall`
and `git diff --check` were also run for the changed files.

A full `unittest discover tests` regression run was started separately. At the
time of this closure it was still running in the local environment and emitted
known Streamlit bare-mode warnings and PyArrow dataframe-serialization warnings
from unrelated UI rendering; no `unittest` failure had been reported in its log.
Its final result must be captured before declaring a release-level full-suite
pass.

Browser/manual smoke evidence is not claimed: the available browser integration
failed before navigation with its sandbox-policy error. AppTest evidence is
automated UI evidence and is not treated as a replacement for a manual browser
smoke test.

## Deferred to Sprint 11H

This pass does not add cross-domain impact attribution, grid/CO2 interpretation,
utility-factor semantics, proposal-persistence changes, or new L0 physics. It
does not choose or mutate shared-proposal semantics beyond resetting the active
draft selections on an explicitly confirmed baseline replacement.
