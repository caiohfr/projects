# Sprint 12E.2 — EPA RUN Grain Closure & FuelCons Reconstruction

## Status: `EPA_FUELCONS_READY — PROCEED_TO_APPLICATION_INTEGRATION`

```text
EPA source rows                              30194
Canonical EPA RUNs before grain closure      30117
Canonical EPA RUNs after grain closure       29000

EPA VDEs                                     11377
EPA VDEs eligible for FuelCons               7501
EPA FuelCons created                         10572

FuelCons by basis:
  EPA_LABEL_2_CYCLE                          7334
  EPA_TEST_PROCEDURE                         3238
  other                                      0

FuelCons by electrification:
  ICE                                        7597
  HEV                                        2436
  PHEV                                       539
  BEV                                        0

VDEs with:
  single RUN adoption                        1997
  multi-RUN adoption                         7298
  unresolved FuelCons                        2638

Legacy regression:
  exact / equivalent                         0
  grain-change difference                    0
  source-refresh difference                  0
  approved method correction                 0
  unresolved                                 10572

Relationship failures                        0
Runtime DB changed?                          NO
User decisions required                      0
```

## RUN grain closure

The deterministic execution key is **canonical VDE + Testgroup + Test Number + procedure + test vehicle/configuration + Set ABC + police/overdrive condition + result signature**. Only source rows whose execution conditions and results agree are grouped. Their aftertreatment and declared averaging rows remain embedded as structured source-row details and are also exported one-to-one in `run_source_row_lineage.csv`.

This closes 30,117 source-row RUNs to 29,000 execution-grain RUNs while retaining 30,117 loaded source-row lineage records. The 77 identity-anomaly rows from 12E.1 remain quarantined and are not silently reintroduced.

## FuelCons methodology

- Regular FTP procedures 2/21/31 provide City evidence; procedure 3 provides Highway evidence.
- Source `RND_ADJ_FE` is used only where `FE_UNIT=MPG`, fuel is a supported liquid fuel, and the value is finite in the non-sentinel range. Conversion is `235.214583 / MPG`.
- Source `CO2 (g/mi)` is converted by division by `1.609344`.
- Two-cycle additive per-distance results use `0.55 × City + 0.45 × Highway` through the existing canonical formula owner.
- Multiple eligible RUNs are adopted only when their source metric signatures agree. Conflicting candidates remain unresolved; no first-row selection or implicit average is used.
- US06 and SC03 become `EPA_TEST_PROCEDURE` FuelCons with values only in their specialized fields.
- No calculation RUN is added: the existing multi-RUN adoption table plus FuelCons provenance records formula, version, inputs, roles, and units without creating redundant evidence.

## Applicability and deliberate NULLs

ICE/HEV and PHEV charge-sustaining liquid-fuel evidence may materialize. PHEV charge-depleting, BEV electric energy, and FCEV equivalent-fuel results remain RUN evidence because this source export labels the field `MPG` without a supported Wh/km contract. Energy, range, Bag fallback, and LHV-derived values remain NULL. Zero is preserved when directly observed and is never used as a missing-value substitute.

## Exceptions

- Unresolved materialization cases: 2,960; each has candidate RUN ids and an explicit reason.
- BEV/FCEV energy-unit closure remains a documented GAP, not a fabricated conversion and not a runtime blocker for supported liquid-fuel reconstruction.
- The HEV classifier is retained from the validated project logic (`FE Bag 4` presence), explicitly labeled `DETERMINISTIC_VALIDATED`; it is not upgraded to direct source identity.
- Legacy baseline FuelCons rows were regression references only and were not copied.

## Validation

All 10 relationship checks pass. SQLite `quick_check` is `ok` and foreign-key violations are zero. Two rebuilds produced the same relationship signature `F9D216C6CEEE06A4AB04E1D08BD34B257870F368A86AE2F46DC15894933C3FF0`.

Runtime SHA-256 before/after: `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB` / `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB`. Byte-identical: **YES**. No runtime cutover and no Streamlit page modification occurred.

## Evidence levels

- **DIRECTLY_TESTED:** RUN grouping, one-to-one source lineage, distinct condition conflicts, multi-RUN adoption, deterministic selection, unit conversions, 55/45 combination, NULL applicability, specialized-cycle isolation, same-VDE lineage, no legacy copy, runtime hashes, deterministic rebuild.
- **INSPECTION_SUPPORTED:** source field semantics and legacy pipeline classifications in `metric_method_matrix.csv`.
- **INDIRECTLY_COVERED:** Sprint 12D physical schema and Sprint 12E.1 clean population.
- **GAP:** authoritative BEV/CD energy unit and range materialization contract.
