# Sprint 12A — ETL Source Inventory + Grain Audit

## Scope and reproducibility

This is a non-destructive source audit. Raw files were read only; no SQLite database, production ETL, application code, physics logic, RAG, enrichment, imputation, or component decomposition was created. The EPA output retains every MY2026 source row.

Reproduce the analysis from the repository root:

```powershell
python etl/scripts/sprint_12a_audit.py
python etl/scripts/sprint_12a_deliverables.py
```

The first command writes machine-readable audit data under `etl/data/processed/sprint_12a_audit/`; the second renders this report and `etl/data/staging/sprint_12a_source_audit.xlsx`.

## Source inventory

The inventory contains 41 source file/sheet records and 1053 profiled fields. It includes EPA Test Car, certification models/results, the FuelEconomy.gov MY2026 workbook inside the supplied ZIP, EEA, JRC, EV-CIS/R154 reference material, and four certification-evidence PDFs. PDFs are inventoried only.

| Source | Direct observation | Grain assessment |
|---|---|---|
| EPA Test Car MY2026 | 3,901 source rows, 67 fields; `Sheet1` | Row grain remains unresolved; rows are preserved exactly. |
| EPA Certified Vehicle Test Results | 388,420 rows, 55 fields; `Test Info` | Repeats vehicle/test context across emission-name results; not a vehicle configuration row. |
| EPA Certified Vehicle Models | 26,866 rows, 10 fields; `Model Info` | Certification model/carline association; not a complete engineering configuration. |
| FuelEconomy.gov MY2026 | Multiple report sheets in source ZIP | Sheet/report grain varies; presentation/title rows remain source evidence, not canonical records. |
| EEA provisional passenger-car CSV | 10,833,597 rows, 37 fields | Row grain cannot be confirmed from the supplied local files; no EPA mapping was imposed. |
| JRC technical dataset | 249 rows, 44 fields | Anonymized OEM/model and `pycsis_run` are present; real-vehicle vs archetype/simulation grain is unresolved. |

For the EEA CSV, row and null counts are exact and were measured in a single streaming pass. Field cardinalities are exact through 1,000,000 distinct values; higher cardinalities are explicitly recorded as capped lower bounds in `FIELD_INVENTORY` rather than guessed.

## EPA MY2026 grain audit

- Source rows: **3,901** (`epa_testcar_2026_raw.xlsx`, `Sheet1`).
- Make + Model + Year groups: **784**; groups with multiple source rows: **706**.
- Unique observed `Test Number | Actual Tested Testgroup` combinations: **3,557**.
- Distinct Target ABC sets: **1,380**; Set ABC sets: **1,431**; ETW values: **22**; procedure variants: **12**.
- Groups with more than one observed Target ABC configuration: **328**.

Evidence: `EPA_2026_GRAIN` keeps source Excel row number, vehicle ID, configuration number, test group/number/procedure, ETW, driveline, transmission, axle/N/V, and all Target/Set A/B/C fields. This is direct observation; it is not a decision to create DB rows at that grain.

### Tier-0 component coverage — EPA Test Car MY2026

| Concept | Row coverage | Evidence |
|---|---:|---|
| Authoritative Target ABC | 100.0% | Explicit `Target Coef A/B/C` fields |
| Set ABC | 100.0% | Explicit `Set Coef A/B/C` fields |
| ETW | 100.0% | Explicit `Equivalent Test Weight (lbs.)` |
| Transmission / gear count / axle ratio / N/V | 100.0% / 100.0% / 100.0% / 100.0% | Explicit source fields |
| Tire specification / pressure / RRC / Cd/CdA / TOTAL-NET | 0.0% / 0.0% / 0.0% / 0.0% / 0.0% | No explicit Test Car field located |

Roadload classification is availability-only: **ROADLOAD_WITH_PARTIAL_COMPONENT_DATA: 3,901**. No arbitrary quality threshold or component closure was applied.

### Configuration-pair candidates

The conservative candidate generator wrote **696** pairs: same context + ETW; different roadload set: 696. Tire-only, mass-only, and driveline-only candidates were **0** under the strict matching rules because the Test Car table exposes no tire field and no pair met the other isolation conditions.

Every pair in `EPA_CONFIGURATION_PAIRS` records both source rows, shared and differing fields, Target ABC deltas, ETW delta when numeric, warnings, and the explicit statement that it is candidate discovery—not a component causal claim.

## WLTP / Europe audit

- EEA: **10,833,597** rows. Directly labelled fields cover identity/reporting keys, `m (kg)`/`Mt`, `ep (KW)`, fuel/electric consumption, NEDC/WLTP CO2 result fields, and electric range. No tire or phase field is present. `RLFI` exists but its physical meaning/unit is **UNRESOLVED** without a supplied data dictionary.
- JRC: **249** rows. Direct fields include curb/WLTP/real-world masses, engine/electric-motor/battery attributes, tire code, gear box/gears, and explicit `wltp|f0/f1/f2` and `rw|f0/f1/f2` labels. The row's real-vehicle vs archetype/simulation meaning remains **UNRESOLVED**; anonymized identity is only partial.

These findings are field-presence evidence only. No EPA terms were forced onto EEA/JRC, and no claim was made that `RLFI` or the JRC roadload fields have the same semantics as EPA Target/Set coefficients.

## Reference material and evidence standard

EV-CIS Data Requirements, EV-CIS Business Rules, the EV-CIS XML schema ZIP, and UNECE R154 are listed in `SOURCE_FILES` as reference documentation. They were not parsed into a RAG/vector database. The field interpretations in this audit are either **DIRECTLY OBSERVED** from explicit source labels/units and values or **UNRESOLVED**. No unsupported reference-based physical inference was elevated to confirmed.

## Architectural questions for Sprint ownership

1. **EPA Test Car grain** — Does one EPA Test Car row represent a complete test result, an emissions result, or another reporting unit when the same Make/Model/Year repeats? Evidence: Multiple source rows are preserved in EPA_2026_GRAIN; no aggregation was performed. (UNRESOLVED).
2. **EPA grouping** — Can Target ABC configurations ever be grouped across source rows, and if so which test identifiers/context must be equal? Evidence: Distinct Target/Set/ETW/procedure counts are measured, but a grouping rule is not implied. (UNRESOLVED).
3. **EPA roadload component closure** — How should the data contract represent roadload when EPA Test Car exposes Target/Set/ETW/driveline but no tire, pressure, RRC, Cd, or CdA? Evidence: Tier-0 coverage explicitly records these fields as absent. (ARCHITECTURE DECISION).
4. **EEA RLFI** — What is the physical meaning and unit of EEA field RLFI? Evidence: The local CSV exposes RLFI but no accompanying data dictionary was present. (UNRESOLVED).
5. **JRC grain** — Does each anonymized JRC row represent a real vehicle configuration, a fleet archetype, or a PYCSIS simulation input/run? Evidence: Anonymized OEM/model and pycsis_run fields are present; local documentation does not establish row grain. (UNRESOLVED).

## Deliverables

- `etl/notebooks/01_source_inventory.ipynb` — reproducible, non-destructive notebook entry point.
- `etl/scripts/sprint_12a_audit.py` — audit implementation.
- `etl/data/processed/sprint_12a_audit/` — CSV/JSON audit evidence.
- `etl/data/staging/sprint_12a_source_audit.xlsx` — staging workbook with requested sheets.
- `etl/reports/sprint_12a_source_inventory.md` — this closure report.
