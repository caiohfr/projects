# Sprint 12 — Transmission Grouping & Coastdown Signal Experiment Result

## Decision

This read-only experiment tests repeatable grouping signal only. It does not estimate or publish transmission-loss ABC and does not establish causality.

## A. SOURCE / GROUPING

- Available fields: make/model/year, category, test mass, transmission type, gear count, drive system, final-drive ratio, N/V ratio, electrification when linked, Test Number/ADFE Test Number, VDE carryover lineage, and canonical Target ABC outcome.
- Strongest direct identity evidence: none. `vehicle_configuration.transmission_model` is unpopulated, and EPA's transmission type/code is a broad family descriptor rather than a hardware reference.
- Float handling: final-drive ratio is retained at its observed two-decimal source precision; N/V at one decimal. Exact normalized values are used, with no fuzzy tolerance. Explicit zero remains zero.
- Explicit zero/sentinel-like ratios are preserved but not upgraded to strict identity: final-drive or N/V <=0, N/V >=900, and the one-speed `1.00/1.0` placeholder pattern are classified `FAMILY_ONLY` pending source clarification.
- DIRECT_MATCH groups: 0.
- STRICT_CANDIDATE groups: 2674.
- Usable groups with >=3 independent applications: 207.
- Usable groups with >=5 independent applications: 37.

## B. SAMPLE CONTROL

- Raw EPA VDE count: 11377.
- Excluded exact carryover descendants: 5084.
- Independent root VDEs before minimum-group filtering: 6293.
- Independent experiment units in the primary >=3-app sample: 1274.
- Statistical unit: one canonical VDE root after following `vde_id_parent` to collapse exact model-year carryover. All speed points from a VDE and all records in a normalized make/model application lineage stay in one CV fold.

## C. MODEL 0

- Formula/features: speed + speed², test mass, model year, make, category/body class, drive system, electrification, and their prespecified speed interactions.
- Excluded leakage: Target ABC as features, Target-derived CdA/RRC, legacy decomposition values, and all transmission-loss estimates.
- Validation: 3-fold `StratifiedGroupKFold`, grouped by normalized make/model application lineage; complete VDE curves held out.
- OOF RMSE: 58.948173 N.
- OOF MAE: 42.568738 N.

## D. MODEL 1

- Added terms: candidate-group intercept (`group_A`) and candidate-group × speed (`group_B * v`) only. No group quadratic term.
- Ridge regularization: alpha=10, fixed before comparison.
- OOF RMSE: 36.867448 N.
- OOF MAE: 26.144081 N.
- OOF delta RMSE: 22.080725 N (positive favors Model 1).
- Positive grouped folds: 3/3.
- Application-cluster bootstrap 95% CI for delta RMSE: [18.727276, 25.443664] N.

## E. RESIDUAL SIMILARITY

- Observed pair-weighted within-group residual-curve RMSE: 32.765324 N.
- Mean stratified randomized comparison: 51.531999 N.
- Support: NO.

## F. PERMUTATION

- Permutations: 2, deterministic seed 12012.
- Labels shuffled only within make + broad transmission type + gear count + drive architecture strata, preserving each stratum's label multiset and group-size distribution.
- Movable strata/units: 45/1052.
- Observed statistic: 32.765324 N (lower is more grouped).
- Empirical p-value: 0.333333.

## G. ROBUSTNESS

| sensitivity_case | status | n_independent_vdes | n_groups | baseline_rmse_N | group_model_rmse_N | delta_rmse_N | notes |
|---|---|---|---|---|---|---|---|
| PRIMARY_20_120_KPH | COMPLETED | 1274 | 207 | 58.948173 | 36.867448 | 22.080725 | Primary prespecified speed grid. |
| SPEED_30_120_KPH | COMPLETED | 1274 | 207 | 60.888971 | 37.787119 | 23.101852 | Refit on 30..120 km/h. |
| SPEED_20_100_KPH | COMPLETED | 1274 | 207 | 49.722350 | 32.714136 | 17.008213 | Refit on 20..100 km/h. |
| DIRECT_MATCH_ONLY | INSUFFICIENT_SAMPLE | 0 | 0 |  |  |  | No explicit transmission hardware/model code is populated; zero DIRECT_MATCH groups. |
| DIRECT_PLUS_STRICT_CANDIDATE | COMPLETED | 1274 | 207 | 58.948173 | 36.867448 | 22.080725 | Equivalent to primary because DIRECT_MATCH count is zero. |
| GROUPS_GE5_APPLICATIONS | COMPLETED | 374 | 37 | 48.862256 | 33.531309 | 15.330947 | Small groups excluded explicitly. |
| EXCLUDE_DOMINANT_APPLICATION | COMPLETED | 1252 | 204 | 56.947267 | 36.190643 | 20.756625 | Excluded CHEVROLET\|TAHOE 4WD (14 independent VDEs); groups requalified at >=3 applications. |
| EXCLUDE_FLAGGED_CURVE_OUTLIERS | COMPLETED | 1178 | 192 | 46.546036 | 30.923498 | 15.622538 | Excluded 66 prespecified baseline-residual IQR flags; primary retains them; threshold=121.064694 N. |

Outliers were defined from primary baseline OOF curve RMSE using Q3 + 1.5×IQR. 66 VDEs exceeded 121.064694 N. They remain in the primary result and are excluded only in the explicitly labeled sensitivity case.

## H. REPRESENTATIVE GROUPS

- BMW: `TXG-807A448DE4EE6DDD` uses only `MAKE=BMW|TRANS=SEMI AUTOMATIC|GEARS=8|DRIVE=2 WHEEL DRIVE REAR|FDR=2.81|NV=24.0|ELEC=UNKNOWN|HARDWARE=UNPROVEN`; it contains 17 independent VDEs across 11 normalized model/application lineages. No hardware name or supplier is inferred.
- Every group row exposes the exact signature, observed ratios, application counts, and identity status in `TRANSMISSION_CANDIDATE_GROUPS.csv`.
- No candidate receives an unproven commercial hardware name.

## I. CONCLUSION

The conclusion is `NO` under the prespecified minimum evidence gate. Even a supported grouping signal would mean only that the structured signature carries reproducible whole-vehicle information; it would not prove transmission neutral drag. Independent CdA and tire/RRC coverage remain important confounders for any causal decomposition stage.

```ini
TRANSMISSION_GROUPING_BUILT = YES
INDEPENDENT_EXPERIMENT_UNITS = 1274
USABLE_GROUPS_GE3 = 207
USABLE_GROUPS_GE5 = 37

BASELINE_CV_RMSE_N = 58.948173
GROUP_MODEL_CV_RMSE_N = 36.867448
OUT_OF_SAMPLE_RMSE_IMPROVEMENT_N = 22.080725

PERMUTATION_P_VALUE = 0.333333
RESIDUAL_SIMILARITY_SUPPORT = NO

TRANSMISSION_GROUP_SIGNAL_SUPPORTED = NO

READY_FOR_SHARED_TRANSMISSION_LOSS_ESTIMATION_EXPERIMENT = NO

PRODUCTION_DB_CHANGED = NO
```

## Reproducibility / immutability

- Source DB: `C:\Users\CaioHenriqueFerreira\Downloads\From Git\projects\EcoDrive-Analyst\data\db\staging\eco_drive_canonical_candidate.db`
- SHA256 before: `8AFAC44888388452E9EBF85F8162B2BFA233DFCC74AC3C924339F235A0EA0330`
- SHA256 after: `8AFAC44888388452E9EBF85F8162B2BFA233DFCC74AC3C924339F235A0EA0330`
- Connection mode: SQLite URI `mode=ro` plus `PRAGMA query_only=ON`.
