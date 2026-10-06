# Sprint 12 Final Closure Contract

Sprint 12 closes on the immutable canonical snapshot recorded in
`artifacts/sprint12_final/eco_drive_canonical_sprint12_final.db`.
The release snapshot is a byte-for-byte copy of the accepted staging candidate;
PROD and QA are not promotion targets in this closure.

## Frozen Macro semantics

- Aero historical slot: Aero macro.
- Tire historical slot: `ROLLING_MINOR` only when the VDE provenance and method
  explicitly identify macro decomposition.
- Transmission historical slot: conventional `DRIVETRAIN_AGGREGATE` only with
  explicit projection provenance.
- Fixed-gear `EDRIVE_AGGREGATE`: retained in component resolution/audit and not
  forced into the historical Transmission slot.

## Frozen Fine semantics

- Fine reference, surrogate, and catalog evidence remains `SUPPORTING` unless it
  is independently promoted under the canonical contract.
- Vector search was not validated for vehicle-specific fine identity.
- The Sprint 12 release has zero adopted fine-component links.
- Unresolved and boundary-unknown records remain explicit scientific results.

## Research Agent v0

Final status: `EXPERIMENTAL_PARTIAL_FROZEN_FOR_REVIEW`.

Internal-first lookup, the estimator-unlock gate, claim-specific queries,
pre-fetch relevance filtering, and source ranking are present. High-quality live
retrieval and fleet-scale P1/P2 coverage are not claimed. The agent writes no
research results directly to canonical ABC values. RAG, embeddings, and vector
databases are deferred future AI-engineering capabilities, outside Sprint 12.

## Review export

`EcoDrive_Canonical_DB_Sprint12_Final.xlsx` is a read-only review mirror built
directly from the immutable release DB. SQLite remains canonical. Raw sheets are
complete; derived `VDE_FLAT` and `COMP_COVERAGE` sheets do not replace storage
tables.
