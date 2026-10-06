# Sprint 12F.12 — Full-Population Canonical Consolidation

## Status: `FULL_CANONICAL_CANDIDATE_READY — PROCEED_TO_12G_INTEGRATION`

```text
Programs                                      3117
Vehicle configurations                       10215
VDEs                                          11630
RUNs                                          29258
FuelCons                                      10826
FuelCons↔RUN adoption rows                    18327

Electric canonical Wh/km source records       527
  represented in canonical RUN evidence       521
  source-only identity quarantines               6
New electric FuelCons                            0
Still unresolved Electricity cases              981
Hydrogen cases outside electric rule             32

Foreign-key violations                           0
SQLite quick_check                                ok
Runtime DB changed?                             NO
Deterministic rebuild?                          YES
User decisions required                           0
Focused acceptance tests                         11/11 PASS
```

## Consolidation result

The candidate is a full copy of the Sprint 12E.2 canonical population, not the 75-program notebook demo. All 11 canonical/helper tables were preserved and exported. Entity counts are unchanged because Sprint 12E.3A closed a unit interpretation, not a new FuelCons materialization method.

The 527 approved current electric source records are converted deterministically from `kWh/100mi` to `Wh/km`. Of these, 521 are attached to their canonical RUN evidence. Six source rows are retained in the accompanying electric population export as `SOURCE_ONLY_QUARANTINED_IDENTITY_ANOMALY`: the refreshed source has `2025` in the make field, so Sprint 12E.2 correctly excluded them from canonical identity mapping. No program, configuration, VDE, or RUN identity was invented to hide that source defect.

The two Honda CR-V e:FCEV zeros remain observed raw zeros with NULL canonical energy. The six records absent from the refreshed source remain retired. All 1,257 Electricity cases retain their 12E.3A disposition: 276 unit-resolved, 915 not the same semantics, and 66 conflicting source context. All 32 Hydrogen cases remain outside the electric rule.

No additional FuelCons row was created. Unit resolution alone does not establish an approved comparison basis, cycle, grain, or adoption method. Existing FuelCons-to-RUN lineage remains complete and same-VDE.

## Population reconciliation

Every database table has the same row count as the 12E.2 input. The reason is explicit: this sprint consolidates the full population and embeds approved electric audit evidence in existing RUN JSON without duplicating the legacy baseline or forcing unsupported electric FuelCons.

## Integrity and exports

- `PRAGMA foreign_key_check`: 0 violations.
- `PRAGMA quick_check`: `ok`.
- Primary-key duplicate groups: 0 across every table.
- Every table export count matches its database count.
- Every FuelCons has at least one adoption row, and every adoption connects FuelCons and RUN on the same VDE.
- Input and candidate schema signatures are identical.

The processed directory contains deterministic CSV exports for all tables, electric reconciliation, runtime fingerprints, integrity results, table reconciliation, descriptive statistics, and lightweight Pearson correlations. Correlations are QA associations only; sparse-pair sample sizes are reported and no causal inference is made.

## Runtime safety

Neither runtime database nor the notebook demo was written. Their before/after hashes are byte-identical. Candidate database SHA-256: `4F07C185BBC846D6C4F49E443984FC5D1046E14BF5294D2FFA34B77CE26CD62D`. Logical content signature: `1DB097E6238D93FFD55E0C5B249E5E5AD6A02F76EB1070A2AA087680AE0E08B1`.
