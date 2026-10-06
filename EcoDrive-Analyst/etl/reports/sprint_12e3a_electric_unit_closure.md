# Sprint 12E.3A — Electric Unit Closure Patch

## Status: `ELECTRIC_UNITS_READY — PROCEED_TO_12F_NOTEBOOKS`

```text
Historical heuristic rows                       535
Current source matches                          529
Resolved non-zero electric rows                 527
Excluded FCEV zero rows                           2
Retired absent rows                               6

12E.2 unresolved Electricity cases             1257
Now resolved by unit rule                       276
Still unresolved                                981
Not same semantics / conflicting                981

Hydrogen 5 cases                                 32
Converted by electric rule                        0

Additional canonical EPA FuelCons                 0
Runtime DB changed?                             NO
User decisions required                           0
```

## Closure

The 527 non-zero current EPA source rows that exactly match the recovered historical cohort are now interpreted as `kWh/100mi`. Their raw values are preserved and converted directly with:

```python
Wh_per_km = value * 1000 / (100 * 1.609344)
```

For example, `29.9 kWh/100mi = 185.790 Wh/km = 18.579 kWh/100km`. MPGe is not an intermediate.

The Honda CR-V e:FCEV source zeros (`EPA-LEGACY-ROW-1836` and `EPA-LEGACY-ROW-1838`) remain observed zeros with NULL canonical energy. The six absent records remain retired.

## 12E.2 disposition

- 276 Electricity cases contain only approved source rows and are `NOW_RESOLVED_KWH_PER_100MI` at the unit layer.
- 66 mix approved and non-approved source context and remain `CONFLICTING_SOURCE_CONTEXT`.
- 915 do not share the selected historical source semantics and are `NOT_SAME_SOURCE_SEMANTICS`.
- All 32 Hydrogen 5 cases remain outside the electric rule.

No additional FuelCons was materialized. The existing 12E.2 rule has no approved electric CD/MCT cycle-materialization path; unit closure does not bypass metric, comparison-basis, cycle, or grain validation.

## Evidence and provenance

- Recovered helper algebra implements `kWh/100mi → MPGe`; the legacy `kWh/km` comment was incorrect.
- The repository FuelEconomy workbook explicitly uses `KW-HR/100Miles` for EV consumption rows.
- [EPA fuel economy and EV range testing](https://www.epa.gov/greenvehicles/fuel-economy-and-ev-range-testing) documents that 33.7 kWh used over 100 miles corresponds to 100 MPGe.
- [EPA electric vehicle label description](https://www.epa.gov/fueleconomy/text-version-electric-vehicle-label) defines the consumption rate as kilowatt-hours used to travel 100 miles.
- [EPA Automotive Trends technical explanation](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P101698T.txt) gives the independent MPGe relationship using 33.705 kWh/gallon and kWh/mile.

Magnitude was used only for anomaly QA. Runtime databases, schema, RUN/FuelCons design, notebooks, pages, raw sources, and physics were not modified.

Output signature: `AAD7B3DB36EB73A7DFF419668907F113E8A93C52B8C26D4542695AA1D6BDE516`. Deterministic rebuild: **YES**.
