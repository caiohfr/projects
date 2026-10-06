# EcoDrive Component Estimator — Gate-2 Development Run v1.0

**Estimator:** v1.0 frozen candidate

## Decision

**PASS — freeze Estimator Contract v1.0 for holdout validation.**

No physical equation or threshold was relaxed to make a vehicle pass. Source/matching defects were corrected as evidence defects, not model defects.

## Reproducible exact/gate matrix

| Case | Branch | CdA [m²] | RRCeq@80 [N/kN] | NRMSE [%] | Status |
|---|---|---:|---:|---:|---|
| Kia Carnival Hybrid | MOSKALIK_2020_ANALYTIC | 0.962 | 7.916 | 0.063 | SUPPORTED |
| Ford Mustang GTD | MOSKALIK_2020_ANALYTIC | 1.060 | 14.838 | 0.791 | SUPPORTED |
| Chevrolet Equinox FWD | MOSKALIK_2020_ANALYTIC | 0.953 | 6.641 | 0.442 | SUPPORTED |
| Buick Enclave FWD | MOSKALIK_2020_ANALYTIC | 1.063 | 7.092 | 0.366 | SUPPORTED |
| Buick Encore GX AWD | MOSKALIK_2020_ANALYTIC | 0.893 | 9.289 | 0.658 | SUPPORTED |
| Mercedes-AMG CLA35 | TARGET_ONLY_STRUCTURED | 0.636 | 14.404 | 0.000 | CONDITIONAL |
| Hyundai NEXO 046F | FIXED_GEAR_EDRIVE_V1 | 0.917 | 5.432 | 0.228 | CONDITIONAL |
| Hyundai NEXO 047F | FIXED_GEAR_EDRIVE_V1 | 0.896 | 9.064 | 0.169 | CONDITIONAL |
| Ford Escape FWD HEV | DHT_AGGREGATED_V1 | 0.917 | 4.874 | 0.080* | CONDITIONAL |
| Mercedes-Benz GLC350e 4MATIC | MOSKALIK_CONDITIONAL_PRIMARY | 0.764 | 10.744 | 0.412 | CONDITIONAL |
| Kia Sorento Hybrid FWD | MOSKALIK_2020_ANALYTIC | 0.919 | 8.044 | 0.034 | SUPPORTED |
| Cadillac LYRIQ-V | AWD_FIXED_GEAR_GATE | — | — | — | NOT_IDENTIFIABLE |

*For DHT, 0.080% is Set-curve NRMSE; Target closure is not the promotion metric for the aggregate Dyno-Set fit.

## Gate-2 findings

- **Conventional Moskalik:** reproduced across multiple exact EPA source rows; historical unsourced GTD numbers were replaced by the exact rerun rather than treated as an oracle.
- **Target-only conventional:** retained as `CONDITIONAL`; algebraic closure does not promote evidence quality.
- **Fixed gear:** NEXO demonstrates strong `q_roll` allocation sensitivity despite good nominal closure; sensitivity remains mandatory.
- **DHT:** Escape demonstrates that the aggregate model is usable without inventing an internal MG/planetary split.
- **AWD fixed gear:** LYRIQ-V is a real `NOT_IDENTIFIABLE` case, not a missing-data STOP.
- **Regression evidence rule:** exact source row, configuration and test-state identity are required for numeric regression promotion.

## Freeze boundary

The 35-case holdout begins only after this development set. Thresholds, model equations, architecture routing and promotion semantics are frozen. If holdout evidence requires changing them, create a new estimator version and record holdout contamination.

## Source artifacts

- `epa_testcar_2026_raw.xlsx` — authoritative numeric source used for new EPA exact reruns.
- `ecodrive_component_estimator_reference_v1_0.py` — frozen reference implementation.
- `EcoDrive_Estimator_Development_Run_v1.0.csv` — machine-readable matrix.