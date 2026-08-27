---
name: ecodrive-sprint-package
description: Standard workflow for implementing one EcoDrive-Analyst sprint/package from its own spec doc — audit before design, canonical Streamlit-free contracts, evidence-classified tests, and a closure doc. Use whenever a message asks to implement a numbered EcoDrive sprint/package (e.g. "Sprint 11B", "Package 9C") against a spec under docs/specs/ or docs/sprints/.
---

# EcoDrive sprint/package implementation

This skill is deliberately generic — it is the repeatable shape every
EcoDrive-Analyst sprint has followed, not the content of any one sprint.
Sprint-specific requirements always live in that sprint's own spec file
(`docs/specs/sprint_N_*.md` or the task message itself), never in this
skill. Do not encode any sprint's product decisions here.

Permanent, cross-sprint engineering rules (UI-not-physics, canonical
ownership, provenance, zero-vs-missing, and so on) live in `AGENTS.md` at
the repo root — read it before starting, it is not repeated here.

## 1. Ground the work before designing anything

- Read the sprint's own spec doc in full if one exists (`docs/specs/` or
  a doc named in the task). Treat it as authoritative for that sprint's
  scope, frozen decisions, and stop conditions.
- Read any older architecture/guideline doc the spec references, and
  note explicitly which parts a new sprint supersedes — don't silently
  follow stale guidance.
- Audit the CURRENT implementation of whatever the sprint is changing
  before writing new code, even if a module looks familiar from a prior
  session. Cite file:line evidence, not memory or a prior summary. For
  large modules, delegate the audit to parallel `Explore` agents with
  precise, numbered questions rather than reading everything yourself —
  it's cheaper and produces citable answers.
- If the audit surfaces a pre-existing inconsistency or quirk unrelated
  to the sprint's own scope, record it as a documented finding; do not
  silently fix it and do not silently ignore it.

## 2. Design canonical, Streamlit-free contracts first

- New data shapes and calculation entry points belong in `src/vde_core/`
  and must not import Streamlit or depend on `st.session_state`.
- Reuse existing canonical functions for anything that already has one
  owner (a formula, a resolver, a DB read). Never re-derive a formula
  that already exists elsewhere, even approximately — import it.
- Prefer a small, explicitly-typed design (dataclasses/enums) over a
  generic/dynamic framework, unless the spec explicitly asks for one.
- Keep the UI layer (`src/vde_app/`, `pages/`) a thin caller: it collects
  input, calls the canonical core, and renders the result — never a
  physics or business-rule participant.

## 3. Implement inside the frozen spec, escalate at its boundaries

- A sprint's own spec lists what the agent may decide autonomously
  (module layout, helper names, small refactors, ordinary regressions)
  and what requires stopping to report instead (new physics, schema
  changes, semantic reinterpretation of an existing derived quantity,
  deleting working capability, and whatever else that sprint's spec
  names). Follow that split exactly; don't invent additional stop points
  and don't skip past a listed one.
- When a genuinely new rule would be needed and the spec doesn't define
  it, stop and report the exact conflict with evidence — do not guess a
  plausible-sounding rule to keep moving.

## 4. Test with honest evidence classification

- For each requirement, cite the exact test name that covers it, or say
  plainly that it's covered only by inspection, only indirectly by an
  earlier feature's suite, or not at all (a real gap).
- Do not pad the test count to hit a number; the goal is traceability
  against the sprint's actual requirements.
- A claimed bug fix needs to actually reproduce (e.g. revert the fix and
  confirm the regression test fails on the old code) before being
  reported as a confirmed defect, not just a plausible hypothesis.
- Streamlit `AppTest` is this project's automated substitute for manual
  UI testing — useful and expected, but never described as "manual
  smoke." If a sprint requires real interactive manual smoke and the
  environment has no browser/display, say so explicitly rather than
  presenting `AppTest` as having satisfied it.
- Run focused tests for the changed area, then the full suite. Record
  the actual freshly-observed counts and failure names; never copy
  forward an old doc's numbers as if they were just re-verified.

## 5. Close the sprint

- Write a closure doc under `docs/sprints/` following this project's
  established shape: scope, files changed with the canonical functions
  reused (file:line), decisions made or confirmed, test evidence with
  real counts, anything deferred to a later package, and an explicit
  freeze/handoff statement.
- Commit with a message naming the sprint/package and summarizing why,
  not just what.
- Stop at the sprint's own end boundary — do not start the next
  numbered sprint/package unless explicitly asked.
