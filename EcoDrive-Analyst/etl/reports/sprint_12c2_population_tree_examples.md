# Sprint 12C.2 — Real Population Tree Examples

All identifiers are deterministic preview identifiers only; none were written to a runtime database.

## EX-01 — ONE_MMY_ONE_CONFIG_ONE_VDE

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-59AA1262BFAECE67`

```text
2025 | Air Pure RWD w/19" wheels | 2026 [ONE_MMY_ONE_CONFIG_ONE_VDE]
└─ PREVIEW-PROGRAM-7D92CEF57958B0C8
   └─ PREVIEW-VC-FB9F5D10965C6634 (1 VDE state(s))
      └─ PREVIEW-VDE-59AA1262BFAECE67 Target=(25.52, 0.0727, 0.01438) ETW=4750
      └─ PREVIEW-RUN-4DCAD4E55F55810B
```

Evidence: source_rows=1; run_candidates=1; configurations_in_program=1; vde_states_in_configuration=1

## EX-02 — ONE_MMY_ONE_CONFIG_ONE_VDE

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-6881E7204A68D2ED`

```text
2025 | Air Pure RWD w/20" wheels | 2026 [ONE_MMY_ONE_CONFIG_ONE_VDE]
└─ PREVIEW-PROGRAM-04EFBFCC43B19F3A
   └─ PREVIEW-VC-7FEFFB6D500D4BC1 (1 VDE state(s))
      └─ PREVIEW-VDE-6881E7204A68D2ED Target=(30.19, 0.1119, 0.01396) ETW=4750
      └─ PREVIEW-RUN-F8A24137315EFCA6
```

Evidence: source_rows=1; run_candidates=1; configurations_in_program=1; vde_states_in_configuration=1

## EX-03 — ONE_MMY_ONE_CONFIG_ONE_VDE

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-741F2F10D823273C`

```text
2025 | Lucid Gravity Dream (3R) | 2026 [ONE_MMY_ONE_CONFIG_ONE_VDE]
└─ PREVIEW-PROGRAM-54BB60636FB838C4
   └─ PREVIEW-VC-97EA135AC1D68B7B (1 VDE state(s))
      └─ PREVIEW-VDE-741F2F10D823273C Target=(46.14, 0.0344, 0.02174) ETW=6500
      └─ PREVIEW-RUN-039DE86E56DCA752
```

Evidence: source_rows=1; run_candidates=1; configurations_in_program=1; vde_states_in_configuration=1

## EX-04 — MMY_MULTIPLE_CONFIGS

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-CBA15350F27B06CE`

```text
2024 | Lucid Air Grand Touring XR | 2026 [MMY_MULTIPLE_CONFIGS]
└─ PREVIEW-PROGRAM-FDA5A7499044DD3A
   └─ PREVIEW-VC-91D0CEFCC2206C5E (1 VDE state(s))
      └─ PREVIEW-VDE-CBA15350F27B06CE Target=(33.18, 0.0175, 0.0155) ETW=5500
      ├─ PREVIEW-RUN-1BAFAF9945A1EE67
      └─ PREVIEW-RUN-D89C7A8D928E1599
```

Evidence: source_rows=2; run_candidates=2; configurations_in_program=3; vde_states_in_configuration=1

## EX-05 — MMY_MULTIPLE_CONFIGS

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-C5C21F170A39BAF8`

```text
2025 | Lucid Air Touring AWD | 2026 [MMY_MULTIPLE_CONFIGS]
└─ PREVIEW-PROGRAM-56E05220CCA9743B
   └─ PREVIEW-VC-A31C3DE9398F1D14 (1 VDE state(s))
      └─ PREVIEW-VDE-C5C21F170A39BAF8 Target=(31.63, 0.0121, 0.01541) ETW=5250
      └─ PREVIEW-RUN-37C24BF7A59CD848
```

Evidence: source_rows=1; run_candidates=1; configurations_in_program=2; vde_states_in_configuration=1

## EX-06 — MMY_MULTIPLE_CONFIGS

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-2365EDD0D5A74155`

```text
2025 | Lucid Gravity GT (2R) | 2026 [MMY_MULTIPLE_CONFIGS]
└─ PREVIEW-PROGRAM-17E0220EC0F51D72
   └─ PREVIEW-VC-9A603A5E246E4511 (1 VDE state(s))
      └─ PREVIEW-VDE-2365EDD0D5A74155 Target=(42.59, 0.0318, 0.02143) ETW=6000
      └─ PREVIEW-RUN-FD9F3C9A85858010
```

Evidence: source_rows=1; run_candidates=1; configurations_in_program=2; vde_states_in_configuration=1

## EX-07 — MMY_MULTIPLE_CONFIGS

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-6675866BCE5607A3`

```text
2025 | Lucid Gravity GT (3R) | 2026 [MMY_MULTIPLE_CONFIGS]
└─ PREVIEW-PROGRAM-C1574A06765AE21C
   └─ PREVIEW-VC-748FC0956E35489B (1 VDE state(s))
      └─ PREVIEW-VDE-6675866BCE5607A3 Target=(36.44, 0.0249, 0.02107) ETW=6500
      └─ PREVIEW-RUN-28D55F9ABDF594BD
```

Evidence: source_rows=1; run_candidates=1; configurations_in_program=2; vde_states_in_configuration=1

## EX-08 — CONFIG_MULTIPLE_VDE_STATES

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-2721830AED834036`

```text
Audi | A6 | 2026 [CONFIG_MULTIPLE_VDE_STATES]
└─ PREVIEW-PROGRAM-08613E5E3E4D4587
   └─ PREVIEW-VC-499043F73C54CA11 (2 VDE state(s))
      └─ PREVIEW-VDE-2721830AED834036 Target=(44.11, 0.2948, 0.01889) ETW=4750
      ├─ PREVIEW-RUN-4FD7A48A036AF8AD
      ├─ PREVIEW-RUN-627089986BE28C27
      ├─ PREVIEW-RUN-A1C7B33D47BB41B6
      └─ PREVIEW-RUN-E33B27CE5DAAFEFE
```

Evidence: source_rows=4; run_candidates=4; configurations_in_program=1; vde_states_in_configuration=2

## EX-09 — CONFIG_MULTIPLE_VDE_STATES

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-8C287E45030A8BF2`

```text
Audi | A6 allroad | 2026 [CONFIG_MULTIPLE_VDE_STATES]
└─ PREVIEW-PROGRAM-61D85644FE43D862
   └─ PREVIEW-VC-C8385017FEA5FA70 (2 VDE state(s))
      └─ PREVIEW-VDE-8C287E45030A8BF2 Target=(39.64, 0.2407, 0.01898) ETW=4750
      ├─ PREVIEW-RUN-A5209B065E17BE63
      ├─ PREVIEW-RUN-D1FB3F78C6AD8A78
      ├─ PREVIEW-RUN-E837D30CE98030BF
      └─ PREVIEW-RUN-FCD3C76BB72B5A35
```

Evidence: source_rows=4; run_candidates=4; configurations_in_program=1; vde_states_in_configuration=2

## EX-10 — CONFIG_MULTIPLE_VDE_STATES

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-7AAF2DD52A6BFD1A`

```text
Audi | Q3 | 2026 [CONFIG_MULTIPLE_VDE_STATES]
└─ PREVIEW-PROGRAM-BB249D2046213C68
   └─ PREVIEW-VC-4325871D4D4D5C17 (2 VDE state(s))
      └─ PREVIEW-VDE-7AAF2DD52A6BFD1A Target=(39.51, 0.0564, 0.0257) ETW=4250
      ├─ PREVIEW-RUN-0922837163A85FA9
      ├─ PREVIEW-RUN-50F0F5534A91C29F
      ├─ PREVIEW-RUN-ABE95A18A66FED5D
      └─ PREVIEW-RUN-E55687AB9D7F6816
```

Evidence: source_rows=4; run_candidates=4; configurations_in_program=1; vde_states_in_configuration=2

## EX-11 — CONFIG_MULTIPLE_VDE_STATES

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-0B29C6BA43B62573`

```text
Audi | RS 3 | 2026 [CONFIG_MULTIPLE_VDE_STATES]
└─ PREVIEW-PROGRAM-E08CA14907F99AF0
   └─ PREVIEW-VC-E798E9F7F7FA288D (2 VDE state(s))
      └─ PREVIEW-VDE-0B29C6BA43B62573 Target=(32.86, 0.3891, 0.01891) ETW=4000
      ├─ PREVIEW-RUN-A09201BBA4C7CCBE
      ├─ PREVIEW-RUN-AE3BC6205E46D58E
      ├─ PREVIEW-RUN-EFABDB1121D06346
      └─ PREVIEW-RUN-FFF4316640215CF3
```

Evidence: source_rows=4; run_candidates=4; configurations_in_program=1; vde_states_in_configuration=2

## EX-12 — MULTIPLE_RUNS_ONE_VDE

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-79D395833CA79219`

```text
2024 | Air Sapphire AWD | 2026 [MULTIPLE_RUNS_ONE_VDE]
└─ PREVIEW-PROGRAM-91F8F83951CC722A
   └─ PREVIEW-VC-EFC1AA78E04A38D8 (1 VDE state(s))
      └─ PREVIEW-VDE-79D395833CA79219 Target=(38.4, 0.0212, 0.01926) ETW=5500
      ├─ PREVIEW-RUN-6E060F29872397BE
      └─ PREVIEW-RUN-CD6DB9654A305F6C
```

Evidence: source_rows=2; run_candidates=2; configurations_in_program=1; vde_states_in_configuration=1

## EX-13 — MULTIPLE_RUNS_ONE_VDE

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-CBA15350F27B06CE`

```text
2024 | Lucid Air Grand Touring XR | 2026 [MULTIPLE_RUNS_ONE_VDE]
└─ PREVIEW-PROGRAM-FDA5A7499044DD3A
   └─ PREVIEW-VC-91D0CEFCC2206C5E (1 VDE state(s))
      └─ PREVIEW-VDE-CBA15350F27B06CE Target=(33.18, 0.0175, 0.0155) ETW=5500
      ├─ PREVIEW-RUN-1BAFAF9945A1EE67
      └─ PREVIEW-RUN-D89C7A8D928E1599
```

Evidence: source_rows=2; run_candidates=2; configurations_in_program=3; vde_states_in_configuration=1

## EX-14 — MULTIPLE_RUNS_ONE_VDE

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-42BDFC7E8438B48D`

```text
2024 | Lucid Air Grand Touring XR | 2026 [MULTIPLE_RUNS_ONE_VDE]
└─ PREVIEW-PROGRAM-FDA5A7499044DD3A
   └─ PREVIEW-VC-AD299B781244F97A (1 VDE state(s))
      └─ PREVIEW-VDE-42BDFC7E8438B48D Target=(37.32, 0.0127, 0.01626) ETW=5500
      ├─ PREVIEW-RUN-5C5697370581CEDE
      └─ PREVIEW-RUN-5D8C5F6A8F4294B7
```

Evidence: source_rows=2; run_candidates=2; configurations_in_program=3; vde_states_in_configuration=1

## EX-15 — MULTIPLE_SOURCE_ROWS_ONE_RUN

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-2BE8F5DFE46F5257`

```text
BMW | 228 Gran Coupe | 2026 [MULTIPLE_SOURCE_ROWS_ONE_RUN]
└─ PREVIEW-PROGRAM-352CEEB624AABD20
   └─ PREVIEW-VC-6EFB10ACA430D4DC (1 VDE state(s))
      └─ PREVIEW-VDE-2BE8F5DFE46F5257 Target=(30.8, -0.016, 0.01746) ETW=3750
      ├─ PREVIEW-RUN-6C708DD14063A26C
      └─ PREVIEW-RUN-D2B3DCE9235C793B
```

Evidence: source_rows=4; run_candidates=2; configurations_in_program=3; vde_states_in_configuration=1

## EX-16 — MULTIPLE_SOURCE_ROWS_ONE_RUN

Population: `EPA Test Car MY2026` · Source key: `PREVIEW-VDE-D91A37B86920911B`

```text
BMW | 228 Gran Coupe | 2026 [MULTIPLE_SOURCE_ROWS_ONE_RUN]
└─ PREVIEW-PROGRAM-352CEEB624AABD20
   └─ PREVIEW-VC-7938E454E063128B (1 VDE state(s))
      └─ PREVIEW-VDE-D91A37B86920911B Target=(28.2, -0.025, 0.01799) ETW=3750
      ├─ PREVIEW-RUN-C0D182EAEF7B9E64
      └─ PREVIEW-RUN-F26543DB286F575D
```

Evidence: source_rows=4; run_candidates=2; configurations_in_program=3; vde_states_in_configuration=1

## EX-17 — LEGACY_DUPLICATE_MMY

Population: `Legacy EPA 2020-2025` · Source key: `1719;1720`

```text
BENTLEY | BENTAYGA | 2024
└─ persisted MMY identity
   ├─ VDE 1719
   └─ VDE 1720 (2 Target-ABC variants)
```

Evidence: Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved.

## EX-18 — LEGACY_DUPLICATE_MMY

Population: `Legacy EPA 2020-2025` · Source key: `811;812`

```text
BENTLEY | BENTAYGA | 2025
└─ persisted MMY identity
   ├─ VDE 811
   └─ VDE 812 (2 Target-ABC variants)
```

Evidence: Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved.

## EX-19 — LEGACY_DUPLICATE_MMY

Population: `Legacy EPA 2020-2025` · Source key: `4237;4238`

```text
BMW | 230I COUPE | 2020
└─ persisted MMY identity
   ├─ VDE 4237
   └─ VDE 4238 (2 Target-ABC variants)
```

Evidence: Identity collision after legacy normalization; retain distinct source-derived VDE states until resolved.

## EX-20 — HISTORICAL_REFRESH

Population: `EPA refreshed historical` · Source key: `EPA-2025`

```text
EPA MY2025 refresh
├─ old source rows: 4256
└─ refreshed source rows: 4287 (delta +31)
```

Evidence: Model Year row count.

## EX-21 — DERIVED_SCENARIO_VDE

Population: `Current post-import` · Source key: `5031`

```text
Parent VDE 4996
└─ derived VDE 5031 | TOYOTA MOCK_VEH1 2024
   └─ 2 FuelCons row(s)
```

Evidence: cycle_source absent; vde_id_parent/delta fields present; mock/test identity or notes present.

## EX-22 — ML_FUELCONS

Population: `Current post-import` · Source key: `4861`

```text
Legacy VDE 4861
└─ RUN(ML_PREDICTION) candidate
   └─ later FuelCons result
```

Evidence: FuelCons id=5018; engine_method=ml_prediction; explicit provenance_json present.

## EX-23 — JRC_SOURCE_SCOPED

Population: `JRC` · Source key: `JRC-row`

```text
OEM_1 | Model_1
└─ source-scoped Program/Configuration (UNRESOLVED)
   ├─ VDE WLTP f0/f1/f2
   ├─ RUN True
   └─ FuelCons declared/simulated evidence
```

Evidence: Anonymized identity; explicit mass/gearbox/gears/tire and roadload labels.

## EX-24 — EEA_MONITORING

Population: `EEA` · Source key: `176934618`

```text
EEA record 176934618 | MINI COUNTRYMAN COOPER | 2025
└─ RUN(MONITORING/DECLARED_RESULT) candidate
   └─ FuelCons reporting evidence
      └─ no engineering VDE/Configuration implied
```

Evidence: Direct reporting fields; RLFI remains unresolved.
