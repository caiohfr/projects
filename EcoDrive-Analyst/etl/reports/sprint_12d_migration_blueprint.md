# Sprint 12D — Proposed Sprint 12E Migration Blueprint

## Boundary

This is a plan only. Sprint 12D did not migrate, cut over, or write any runtime/reference database.

## Proposed phases

1. Freeze source/runtime hashes and create a new empty disposable canonical database.
2. Execute `canonical_schema_v1.sql`; enable and verify foreign keys.
3. Load versioned legacy snapshots into migration staging, never by altering the current runtime file.
4. Populate Program and Vehicle Configuration using approved source-scoped fallbacks and reviewed safe Program consolidation.
5. Populate Component DB and Tire DB references; preserve incomplete enrichment.
6. Populate Component Instances only where evidence exists; unresolved references remain NULL with provenance.
7. Populate wide VDE snapshots with original integer IDs and exact NULL/value semantics.
8. Populate append-oriented RUN evidence, including corrected ESTIMATION/ML/scenario provenance.
9. Populate FuelCons adopted results with original IDs and direct VDE links.
10. Populate `fuelcons_run_adoption` and optional `vde_component_resolution` links.
11. Execute `canonical_compatibility_v1.sql` and compare all legacy-facing columns/relationships.
12. Run exact equivalence: IDs, counts, values, NULL signatures, parent lineage, VDE→FuelCons multiplicity and labels/filters.
13. Run measured query plans/timings on real migrated volume; revise only unsupported indexes.
14. Perform human review and archive a signed migration manifest.
15. Cut over by changing the configured DB path only after explicit approval; retain the original DB read-only for rollback.

## Write-path transition

Compatibility views preserve existing reads. Before cutover, retarget persistence helpers and direct service writes from legacy view names to canonical `vde`, `fuelcons`, and `tire_db` tables. No page should contain schema-specific migration logic.

## EEA

Keep the 10.8M+ row-level monitoring corpus in versioned analytical storage. 12E may materialize RUN/FuelCons only after a reviewed link to a valid VDE exists; unlinked records remain analytical. Do not bulk-load the entire EEA corpus into the Streamlit runtime SQLite database.

## Stop / rollback gates

- Stop on any mismatch in the 180-field compatibility projection, IDs, NULL behavior or relationship multiplicity.
- Stop if current physics requires reconstructed joins instead of the persisted VDE snapshot.
- Stop if a page rewrite is required solely by persistence normalization.
- Stop if source semantics would need to be invented.
- Roll back by restoring the original configured DB path; never mutate the original database in place.
