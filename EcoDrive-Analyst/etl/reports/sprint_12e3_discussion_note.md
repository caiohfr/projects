# Sprint 12E.3 — Discussion Note

## Closure decision

The historical conflict is closed: applicable non-zero Electricity records in the exact-matched legacy cohort are interpreted as `kWh/100mi`. The old `kWh/km` comment was incorrect.

## Evidence supporting `kWh/100mi`

- The legacy helper was named `kwh100mi_to_mpge`.
- Its formula was `33705 / (value * 10)`, which is dimensionally correct for `kWh/100mi → MPGe`.
- The downstream conversion treated the result as MPGe and converted it to Wh/km. Composed end-to-end, the two formulas equal `value * 10 / 1.609344`, exactly the conversion from kWh/100mi to Wh/km.
- The current FuelEconomy MY2026 workbook contains EV rows explicitly labeled `KW-HR/100Miles` alongside MPG rows.

## Limits retained

- The notebook comment says the raw value was `kWh/km`, contradicting the function name and formula.
- The comment says `<40` and `<600 hp`; executable code uses `<42.1` and `<1000 hp`.
- The matching EPA test-car source rows still label the field generically as `MPG`.
- Numeric plausibility cannot establish a unit.
- Two selected raw values are zero; the old vectorized calculation could turn them into infinity and later canonical zero.

## Final disposition

- 535 historical heuristic corrections recovered.
- 527 exact, non-zero Electricity records are `RESOLVED` with active, source-guarded unit overrides.
- 2 exact Hydrogen 5 zero records are excluded and remain unresolved for this rule.
- 6 absent records are `RETIRED_SOURCE_RECORD_ABSENT`.
- Of 1,257 Sprint 12E.2 Electricity cases, 276 resolve at the unit layer, 915 do not share the same semantics, and 66 contain conflicting context.
- All 32 Hydrogen 5 cases remain outside the electric rule.
- No additional FuelCons was materialized because the existing 12E.2 rule has no approved electric cycle-materialization path.
- Runtime databases were not modified.

## Evidence added by 12E.3A

Official EPA EV methodology documents electricity consumption in kWh/100 miles and the independent MPGe relationship based on approximately 33.7 kWh per gallon gasoline equivalent. This evidence is recorded in the closure report. The two zero observations remain quarantined rather than converted.
