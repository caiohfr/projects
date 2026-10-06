# Sprint 12 Closure

## Status

`SPRINT_12_CUTOVER_READY — MANUAL_BROWSER_SMOKE_REQUIRED`

The canonical runtime is active at `data/db/eco_drive.db`. Automated cutover, regression, application integration, database integrity, source immutability, performance sanity, and rollback gates passed. The only remaining formal closure gate is an actual browser smoke.

## Final outcome

- Canonical population: 3,117 Programs, 10,211 Vehicle Configurations, 11,626 VDEs, 29,250 RUNs, 10,822 FuelCons rows, and 18,323 FuelCons↔RUN adoption rows.
- Reads use the `vde_db` and `fuelcons_db` compatibility views.
- Writes use the physical `vde` and `fuelcons` tables.
- The default runtime selects this contract explicitly; there is no schema autodetection or automatic routing.
- Legacy bootstrap/migrations and legacy truncate are blocked for the prepared canonical runtime.
- Full regression passed 1,841/1,841; focused Sprint 12 tests passed 53/53; AppTest passed 5/5.
- `PRAGMA foreign_key_check` returned zero rows and `PRAGMA quick_check` returned `ok`.
- The legacy runtime is preserved at `data/backups/eco_drive_pre_sprint12_20260915.db`, with rollback rehearsed through an explicit alternate path.

The complete evidence, hashes, test names, representative records, performance measurements, rollback commands, limitations, and manual checklist are in [the Sprint 12H cutover report](../etl/reports/sprint_12h_cutover_and_closure.md).

## Five-minute closure check

1. Clear `ECO_DRIVE_DB_PATH`, launch `python -m streamlit run app.py`, and open Browse.
2. Filter/select an EPA vehicle; confirm VDE and FuelCons details.
3. In VDE Setup, preview one safe mass change without saving to production.
4. Open representative EPA/WLTP records in Comparison and run one Quick Scenario.
5. Open Powertrain Scenario, confirm a WLTP vehicle opens, and check the pages for visible errors or broken layout.

After those steps pass, promote the status to `SPRINT_12_CLOSED — CANONICAL_RUNTIME_ACTIVE`; no further database cutover is required.

