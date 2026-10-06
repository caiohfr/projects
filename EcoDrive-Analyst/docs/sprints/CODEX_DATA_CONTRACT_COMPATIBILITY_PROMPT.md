# Codex Handoff â€” Sprint 12 CDR Closure

Read and treat `docs/sprints/CDR_CANONICAL_DATA_CONTRACT_V1_BASELINE.md` as the approved Sprint 12 CDR baseline.

Also read the existing Sprint 12B evidence:
- `etl/reports/sprint_12b_source_to_contract.md`
- `etl/data/processed/sprint_12b_source_to_contract/source_to_contract.json`
- `etl/data/processed/sprint_12b_source_to_contract/open_cdr_decisions.csv`

## Current status

The six CDR decisions are closed.

Do not reinterpret or reopen the nine-entity architecture unless empirical evidence shows a concrete incompatibility.

Do not implement production DDL or migrations yet.

Do not broadly modify `src/`, Streamlit pages, physics, existing resolvers or runtime databases.

## Next task

Execute the **Data Contract v1 materialization + compatibility proof**.

The objective is to prove that the approved canonical model can support the current EcoDrive application with minimal localized changes.

Required outputs:

1. Exact v1 field contract for all nine entities: field name, semantic meaning, data type, nullability, key/FK role, enum candidates, source/provenance expectations and legacy field mapping.
2. In-memory/staging materialization of the canonical model from current legacy data.
3. Canonical compatibility reconstruction reproducing the existing application-facing VDE and FuelCons shapes.
4. Relationship-equivalence checks covering VDE identity/count, FuelCons count, multiple FuelCons per VDE, VDEâ†”FuelCons linkage, relevant RUN adoption lineage design, NULL behavior, mass, roadload ABC, TOTAL/NET, Urban/Highway/Combined results, and labels/filters used by current pages.
5. A mismatch report classifying each difference as `EXACT_EQUIVALENCE`, `APPROVED_CONTRACT_CORRECTION`, `CDR_BLOCKER`, or `DEFERRED`.
6. Final recommendation: `READY_FOR_DDL` or `NOT_READY_FOR_DDL`.

## Guardrails

- Normalize persistence, preserve application contracts.
- Existing pages should not be broadly refactored just because the DB model changes.
- VDE remains the persisted wide Vehicle Demand snapshot.
- FuelCons remains the persisted pacified/adopted comparison result.
- RUN is an evidence ledger and must not become a mandatory read-path traversal for current dashboards.
- Program and Vehicle Configuration source identities may be provisional.
- Ambiguous EEA/JRC semantics remain unresolved until evidence supports interpretation.
- Do not invent component causal roadload splits.
- Preserve raw/source provenance.
- Missing/non-applicable values remain NULL, not zero.
- RAG/document enrichment is a future evidence-resolution layer, not a hidden runtime dependency.

Stop before physical DDL/migration implementation and report the evidence.

## Recommended model/configuration

Use **GPT-5.6 Sol with High thinking** for this package because the task is primarily relational-contract reasoning, migration compatibility analysis and evidence reconciliation rather than mechanical code generation.

