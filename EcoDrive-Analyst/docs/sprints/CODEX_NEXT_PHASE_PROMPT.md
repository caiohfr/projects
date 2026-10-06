# Sprint 12 - Codex Handoff: Data Feasibility / Source-to-Contract Analysis

Read `PDR_CANONICAL_DATA_ARCHITECTURE.md` as the approved Sprint 12 PDR baseline.

## Mission

Run the next phase as **Data Feasibility / Source-to-Contract Analysis**. Challenge the PDR with the data already collected in `etl/` and with the current EcoDrive DB contracts. Produce evidence for the later CDR; do not implement the final schema yet.

## Hard constraints

- Do **not** modify `src/`, Streamlit pages, existing runtime DBs, canonical physics, VDE Setup resolvers, or production migrations during this phase.
- Keep work inside the ETL/research area and documentation/report outputs unless explicitly approved otherwise.
- Do **not** silently resolve PDR open questions by changing architecture.
- Do **not** introduce additional domain entities unless a concrete source/data case demonstrates that the nine-table PDR model cannot represent required information without loss or semantic distortion.
- Preserve the global migration requirement: the future canonical reconstruction must remain functionally lossless relative to the current `vde_db + fuelcons_db` surface, except for explicitly approved contract corrections.
- VDE remains the persisted wide Vehicle Demand snapshot and must not be forced to reconstruct architecture on the UI hot path.
- FuelCons remains the canonical/pacified comparison result.
- RUN is the evidence ledger for TEST, SIMULATION, ESTIMATION, CALCULATION and ML_PREDICTION.
- Component Instance is the lightweight engineering architecture/eBOM layer; Component Resolution is analysis/reduction, not physical component identity.
- Incomplete enrichment must not invalidate authoritative source VDE/Run records.
- Measured, calculated, estimated and ML-derived information must remain explicitly distinguishable through provenance/type/fidelity semantics.

## Evidence to produce

Evaluate the actual EPA, EEA, JRC, certification evidence and current EcoDrive DB fields against the PDR. At minimum, determine:

1. Whether Program identity can be inferred with useful confidence and what fallback identity is needed.
2. Whether Vehicle Configuration can be separated from VDE state using real source fields.
3. Which Component Instances are actually observable from available sources.
4. Which Component DB properties deserve first-class scalar columns versus JSON/artifact references.
5. Which Component Resolutions are defensible from explicit evidence/deterministic calculation without fabricated closure.
6. Which RUN records can already be represented from current test/source data and from representative L0/L1 engineering flows.
7. Which FuelCons `comparison_basis` cases can be pacified with available data, including Urban/Highway/Combined, electric energy and range where applicable.
8. Whether a draft canonical reconstruction can reproduce representative current `vde_db + fuelcons_db` records.

## Classification of findings

Classify each finding as one of:

- `CONFIRMS_PDR` - source/data fits the preliminary model.
- `FIELD_LEVEL_CDR_INPUT` - affects fields, enums, nullability or JSON contracts without changing entity architecture.
- `PDR_CHALLENGE` - evidence suggests grain, ownership or entity relationship must change.
- `DEFERRED` - useful for later simulation/ML but not required for Data Contract v1.

## Expected deliverables

Produce a concise evidence-backed report and machine-readable supporting outputs in the ETL workspace. Include source examples, coverage counts where meaningful, unresolved ambiguities, and explicit proposed CDR decisions. Distinguish direct source evidence from inference/assumption.

Stop before final SQL DDL, production migrations, runtime adapters or CDR schema implementation.
