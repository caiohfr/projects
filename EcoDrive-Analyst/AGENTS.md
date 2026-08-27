# AGENTS.md — EcoDrive-Analyst permanent rules

This file holds only permanent, project-wide rules that apply to every
sprint and every agent working in this repository. Sprint-specific
requirements, acceptance cases, and scope decisions belong in that
sprint's own spec under `docs/specs/` and its closure doc under
`docs/sprints/` — never here. If a rule below would need a footnote
explaining "except for Sprint N," it does not belong in this file.

## UI is not physics

Streamlit code (`src/vde_app/`, `pages/`) collects input and renders
output. It must never compute a physical quantity (fuel consumption,
energy, PSE, roadload force, mass, CdA, RRC, or any derived engineering
result) directly in a widget callback or page function. Every physical
computation lives in a Streamlit-free `src/vde_core/` module and is
called, never re-implemented, from the UI layer. A UI function may do
unit-agnostic display formatting (rounding, string composition) but never
arithmetic that changes a physical answer.

## Canonical ownership — one formula, one owner

Every physical formula and derived-value rule has exactly one canonical
implementation. When a second file needs the same computation, it imports
the existing function; it does not re-derive the formula, even if the
call site looks trivial. Before writing new calculation code, search for
an existing owner first. When centralizing a duplicate, extract the
existing code verbatim (byte-identical logic) rather than rewriting it
from memory or from a summary of what it does.

## Vehicle Demand vs Powertrain boundary

The Vehicle Demand Core (`src/vde_core/vehicle_demand/`) answers "what
does the vehicle require at the wheels" and is frozen: no other module
edits its physics. Powertrain/System Scenario code answers "how does the
powertrain supply that demand" and consumes the Vehicle Demand Core's
result — it never recalculates roadload, mass, tire/RRC, or aero physics,
and it never becomes a second source of vehicle demand values. The
dependency direction is one-way: Vehicle Demand Core code must never
import from or know about Powertrain/System Scenario/Comparison code.

## Explicit provenance

Every value that isn't a raw, directly-observed input carries an explicit
record of where it came from (assumed, corrected, estimated, calculated,
ML-predicted, benchmark-derived, and so on). Provenance is never silently
collapsed, dropped, or upgraded to look more authoritative than it is. A
value's provenance is part of its identity, not an optional footnote.

## Zero is not missing

An explicit `0` (or `0.0`, or an explicit "no change" percent/delta) is a
real, meaningful value and must be preserved as such. It is never treated
as "not provided," "not requested," or "use the default instead." Missing
means the field itself is `None`/absent; zero means the field is present
and its value happens to be zero. These two states must never be
conflated in a contract, a resolver, or a UI default.

## Synthetic/anonymized QA policy

Local QA/fixture data (seeded via `qa_mock_data.py` and equivalents) is
synthetic and must be clearly labeled as such wherever it could be
mistaken for real vehicle data (e.g. `SYNTHETIC_QA_WARNING`-style banners,
`record_origin` markers). No real, identifiable, or proprietary vehicle
data is committed to the repository or used as a fixture. Tests and
demos always run against seeded, synthetic, or already-public data.

## Agent autonomy and escalation

An agent working a sprint may autonomously choose local module/file
decomposition, internal helper names, test structure, and small
refactors needed to eliminate duplicate ownership, without pausing to ask
about routine code-organization choices. An agent must stop and report —
not guess, not silently work around it — before: introducing a new
physical formula that isn't already canonical or explicitly specified;
changing the semantic meaning of an existing derived quantity (e.g.
reinterpreting PSE as a component efficiency); performing a database
schema migration; deleting or materially degrading an existing working
capability; or making any other decision a sprint's own spec explicitly
lists as a stop condition.

## Test and closure evidence discipline

When reporting on a requirement's status, cite the exact test name (or
the exact code location for an inspection-based claim) that backs the
claim. Classify coverage honestly — a direct test, indirect coverage from
an earlier feature's suite, verification by code inspection, or a real
gap — rather than presenting everything as uniformly "tested." Test
counts are not a target to hit; a sprint's goal is traceability against
its actual requirements, never padding a number. Closure documentation
records the actual, freshly-run test counts and failure names, never
counts copied forward from an earlier doc.

## No speculative confirmed-bug claims

A defect is only reported as "found and fixed" after it has been
verified to actually reproduce — for example, by reverting the fix and
confirming the regression test fails against the pre-fix code. A
plausible-sounding failure mode that was hypothesized but did not
reproduce under test may still justify a defensive fix, but must be
described as exactly that (a hardening measure for an unconfirmed
hypothesis), never reported as a confirmed bug.

## AppTest is not manual smoke

Streamlit's `AppTest` harness runs a script headlessly with no browser,
no visual rendering, and no human observing the result. It is this
project's standard automated substitute for interactive testing and
should be used for UI-level regression coverage, but it must never be
described as "manual smoke," "manual testing," or "verified in the
browser." When a real interactive manual smoke pass is required and the
environment cannot perform one (no browser/display available), that
limitation is stated plainly rather than implied to have been done.
