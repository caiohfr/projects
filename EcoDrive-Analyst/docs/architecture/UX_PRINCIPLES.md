# UX principles

## Canonical application surfaces

Canonical pages put the supported workflow first and make its ownership
obvious. Each canonical page has one dominant mental model: a guided flow
(`Source → Setup → Proposal → Calculate → Review`) or a cockpit with focused
drill-down. They must not require an engineer to sort through competing legacy,
debug, metadata, or alternative calculation workflows before completing normal
analysis.

## Canonical audit

Audit information belongs with the canonical result when it explains that
result: provenance, readiness, fidelity, resolved inputs, calculation trace,
and canonical identifiers are examples. Overview surfaces summarize; detailed
editing and inspection remain focused. Heavy diagnostics require an explicit
user opt-in.

`st.expander(expanded=False)`, hidden containers, and tabs are presentation
mechanisms, not computation boundaries. A costly path must be behind an
explicit selection or gate.

## Legacy and engineering tools

Useful historical, compatibility, and engineering workflows remain available
on dedicated legacy surfaces. They are clearly labeled as non-canonical and do
not compete with the preferred product workflow. Legacy routing renders only
the area explicitly selected; moving a UI never duplicates or relocates its
canonical core services.

## Scalable discovery

Large candidate discovery is separate from rich object materialization.
Selectors use lightweight IDs, labels, and searchable metadata; detailed state
is loaded only for the active working set. This boundary applies to canonical
and legacy surfaces alike.

## Validation

AppTest supplies automated UI evidence, but it does not replace a real browser
smoke test. Visual hierarchy, responsiveness, and interaction acceptance need
human browser verification when the environment allows it.
