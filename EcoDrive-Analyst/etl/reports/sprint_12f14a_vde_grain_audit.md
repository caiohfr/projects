# Sprint 12F.14A — VDE Grain Audit

## Status: `VDE_GRAIN_AUDIT_INCONCLUSIVE`

```text
Current VDE rows                              11626
Candidate physical VDE groups                 11626
Likely test-grain duplicate rows              0
Singleton groups                              11626
Multi-row groups                              0
Largest group                                 1

EPA VDE rows                                  11377
EPA physical candidate groups                 11377
WLTP VDE rows                                 249
WLTP physical candidate groups                249

Groups with identical/tolerance VDE results   0
Groups with result mismatches                 1411

Groups with FuelCons on multiple members      6
Groups requiring semantic review              1411

Current real performance/scenario VDEs        5336

Input candidate changed?                      NO
Runtime DB changed?                           NO
User decisions required after audit           0
```

## Executive finding

The audit does **not** prove that any current VDE row is merely a duplicate caused by FTP/HWY/US06/SC03/CD labeling. The conservative signature — configuration, legislation, effective calculation mass, TOTAL roadload A/B/C, and every available component/loss state field — produces 11626 groups from 11626 rows. Every group is a singleton.

There are 10211 source-label-neutral audit families when cycle/test labels are removed but configuration, legislation, and mass are retained. 1411 families contain multiple rows. All of those split into distinct physical signatures because A/B/C differs; none differs only in source/test metadata. The apparent 1415-row reduction is therefore an unsafe hypothetical, not a recommended target.

Floating inputs were normalized to 9 decimal places (absolute grouping tolerance approximately `1e-09`). Result parity uses absolute `1e-10` and relative `1e-09` tolerances. No grouping key contains VDE ID, RUN identity, source record identity, `cycle_name`, or `cycle_source`.

## Current schema and roles

The VDE table has 107 fields. Coverage, distinct counts, and audit-only roles are exported in `vde_field_profile.csv`. All rows have configuration, legislation, effective mass, A/B/C, materialized TOTAL and provenance. Component/loss fields, including transmission-loss ABC, are unpopulated; this is why NET remains unavailable. Cycle labels are source/evidence fields in this audit, while the materialized cycle outputs are results.

## Group-size distribution

Strict physical groups: size 1 = 11626; size 2 = 0; size 3 = 0; size 4 = 0; size 5+ = 0.

Source-label-neutral families: size 1 = 8800; size 2 = 1407; size 3 = 4; size 4 = 0; size 5+ = 0. Their multi-row differences are dominated by A/B/C, source identity, provenance and the corresponding materialized results. Exact field frequencies are in `vde_family_difference_frequency.csv`.

## EPA evidence

EPA contains 11377 rows and 11377 strict physical groups. The 1411 multi-row source-neutral families have the following cycle-label patterns:

| Pattern | Families | Rows | Physical fields identical | Only metadata differs |
|---|---:|---:|:---:|:---:|
| CD+FTP | 0 | 0 | NO | NO |
| CD+FTP+HWY | 0 | 0 | NO | NO |
| CD+FTP+HWY+SC03+US06 | 0 | 0 | NO | NO |
| FTP+HWY | 0 | 0 | NO | NO |
| FTP+HWY+SC03+US06 | 0 | 0 | NO | NO |
| FTP+SC03 | 0 | 0 | NO | NO |
| FTP+US06 | 0 | 0 | NO | NO |
| None+None | 1407 | 2814 | NO | NO |
| None+None+None | 4 | 12 | NO | NO |

The 17 CD-involving families (`CD+FTP` or `CD+CD`) all have distinct roadload state. CD is source/evidence in RUN and an input-bearing VDE snapshot in the current data; it is not a separable calculation-result identity alone. All EPA RUNs retain procedure/category evidence, and every VDE has at least one RUN. One VDE naturally has multiple RUNs already: 9590 VDEs do, with a maximum of 40 RUNs.

RUN can preserve test identity after a future remap, but it does not by itself replace the distinct A/B/C snapshot attached to each current VDE. Collapsing the source-neutral families without another physical-state owner would lose that association.

## WLTP evidence

WLTP/JRC has 249 rows, 249 physical groups, and 249 configurations: one VDE per configuration. All use the WLTP source label with Low/Mid/High/Extra High results stored on the same VDE. There is no phase-level VDE duplication in the current JRC population, so EPA conclusions should not be extrapolated to WLTP.

## VDE result parity

There are no strict multi-row physical groups to compare. As a stress test, every multi-row source-neutral family was compared anyway: 0 have identical/tolerance results and 1411 have material result differences. Those mismatches are surfaced in `vde_source_grain_family_audit.csv`; none was hidden or averaged.

## FuelCons impact

Among the 1411 multi-row audit families:

- `NO_FUELCONS`: 111
- `FUELCONS_ON_ONE_ROW_ONLY`: 1294
- `FUELCONS_ON_MULTIPLE_ROWS_SAME_BASIS`: 0
- `FUELCONS_ON_MULTIPLE_ROWS_DIFFERENT_BASIS`: 6

The six multiple-member/different-basis families would require a semantic decision. Most other linked families would require only FK reassignment mechanically, but that operation is not valid while their physical A/B/C snapshots remain distinct. No FuelCons or FK was changed.

## Performance/scenario evidence

After Sprint 12F.13 cleanup, all VDEs are homologation-source EPA or JRC rows. There are no scenario-derived VDEs, parent-linked VDEs, custom cycles, or populated GVWR/GCWR/MRO load cases. FuelCons `5018` is an ML result on a real EPA VDE, not a separate Performance/Scenario VDE. Current real Performance/Scenario VDE count: 0.

## Representative examples and storage models

`vde_representative_groups.csv` provides at least ten labeled examples with configuration, IDs, cycle labels, mass, A/B/C, TOTAL, RUN summaries and FuelCons bases. `vde_storage_model_comparison.csv` compares Models A–D without choosing one. Under the proven conservative grain, all four models retain an estimated 11626 VDE rows today; a lower count would first require a separate Vehicle Configuration grain decision and a durable owner for each distinct roadload state.

## Architecture questions

1. **Likely source/test-grain duplicates:** 0 proven rows under the conservative configuration-bound physical signature.
2. **Genuinely distinct physical snapshots:** 11626 observed configuration-bound mass/A/B/C states.
3. **`cycle_name` role:** mainly source/test identity, but currently a mixture because cycle-labeled rows often carry distinct physical A/B/C and therefore distinct calculated results.
4. **Coastdown role:** mixed — test evidence is preserved in RUN, while resolved A/B/C is physical input state owned by VDE.
5. **EPA grouping feasibility:** RUN preserves FTP/HWY/US06/SC03/CD evidence, but current rows cannot be grouped losslessly under one VDE because all multi-row same-configuration families differ in physical A/B/C. A child result table alone would not solve the input-state difference.
6. **WLTP treatment:** no equivalent duplication is present; 249 rows already map one-to-one to 249 configurations with phase results on each row.
7. **FuelCons semantics:** grouping is not currently safe. Six families additionally have FuelCons on multiple members with different basis sets; 1,294 have FuelCons on one member only.
8. **Performance/Scenario pressure:** none from current real VDE rows. The lone ML FuelCons does not require a new VDE grain now.

## Safety and assertions

This audit opened the candidate read-only and created only external CSV/JSON/report files. Input SHA-256 before/after: `B8789AB92D9B7A4F7A005006C3A5D341E14B8F6772E63EA603AF406391224D61` / `B8789AB92D9B7A4F7A005006C3A5D341E14B8F6772E63EA603AF406391224D61`. Runtime hashes are unchanged. Ten focused audit assertions pass.
