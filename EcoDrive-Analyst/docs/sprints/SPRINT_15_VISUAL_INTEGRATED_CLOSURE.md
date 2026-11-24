# Sprint 15 visual integrated closure

Status: `PENDING`

Date: 2026-11-24

## Current position

The implementation PRs for Sprint 15A through 15D provide the intended visual
system and automated coverage. This document is not a closure declaration.
The integrated G4 review has not occurred in a controllable real browser.

## Remaining acceptance work

1. Review all eight product pages at 1440x900 and 1024x768.
2. Exercise representative empty, selected, edited and calculated states.
3. Exercise Database Management stage, review and explicit commit using an
   isolated disposable database.
4. Review Methodology equations, diagrams, topic navigation and source links.
5. Confirm that Legacy loads only the selected retained area.
6. Record screenshots and every accepted divergence from the v0.2 references.
7. Obtain explicit owner approval.

## Open contract gap

Database Management stage is currently session state and is not bound to the
active DB path. The owner/architecture review must decide whether stage should
be path-bound or whether DB switching should be blocked while changes are
pending. Sprint 15D did not broaden scope into persistence contracts.

## Closure rule

Change this status to `CLOSED` only after the real-browser G4 evidence and owner
approval are recorded. A merged PR by itself is not sufficient.
