# Sprint Checkpoint - Tire DB / Tire Roadload Foundation

Date: 2026-06-17
Status: MVP foundation closed for review

## Closing Decision

Close this week as the Tire DB / Tire Roadload MVP foundation.

Do not close it as a full VDE Setup redesign. The broader workflow work should move to a future sprint.

## Sprint Scope

This sprint focused on:
- Tire Database as the technical tire/test database.
- Tire Roadload calculation from tire records.
- `tire_test_code` as the workflow lookup key.
- Tire load mass resolution for `TEST_MASS` and `TWC`.
- Minimal VDE Setup preview/save integration.
- Applied tire ABC persistence in `vde_db`.
- Initial service/model tests.

Out of scope for this sprint:
- Full VDE Setup redesign.
- Full component build-up UI.
- `component_db` SQLite.
- Trailer database.
- Complete transmission losses workflow.
- Aero/brake/hub physical modeling.
- Advanced Comparison Report features.
- ML/RAG/MCP/deploy work.

## Technical Decisions

### Tire Complete EPA Code

The spreadsheet field `Tire Complete EPA Code` maps to:

```text
tire_roadload_db.tire_test_code
```

This is the main tire workflow key in VDE Setup. The user should select/search by `tire_test_code`; SAE/ISO coefficients are loaded from the tire database instead of being manually typed in VDE Setup.

### tire_db vs vde_db

`tire_roadload_db` stores technical tire/test data:
- `tire_test_code`
- manufacturer/model/size metadata
- SAE coefficients and units
- ISO/RRC fields
- `rr_n_per_kn` / `smerf`
- test mileage, source, notes and quality metadata

`vde_db` stores the applied vehicle/VDE tire snapshot:
- front/rear tire ids
- front/rear pressure
- front weight distribution
- tire improvement percentage
- tire load mass basis and used mass
- final applied tire A/B/C
- applied RRC when derivable
- calculation source and notes

The full SAE/ISO tire coefficients are not duplicated into `vde_db`.

### SAE / Smerf

SAE/J2452-like tire records use:

```text
Frr = P^alpha * Z^beta * (a + bV + cV^2)
```

The model calculates single tire ABC, converts to axle ABC, sums front/rear axles, and then applies tire improvement.

`Smerf` is treated as the tire RR value:

```text
rr_n_per_kn
```

### ISO Simple Mode

ISO/RRC-only records use the MVP approximation:

```text
A_single = rr_n_per_kn * single_tire_load_kN
B_single = 0
C_single = 0
```

This is intentionally not a full SAE-equivalent curve.

### Mass Resolver

Current naming:

```text
mass_kg = curb / cadastral mass
test_mass_kg = explicit test/calculation mass
inertia_class = TWC / ETW style class when applicable
```

For tire calculations:

```text
tire_load_mass_basis = TEST_MASS | TWC
```

Rules implemented:
- `TEST_MASS` prefers `test_mass_kg`.
- EPA `TEST_MASS` defaults to `mass_kg + 136 kg` when manual test mass is empty.
- `TEST_MASS` falls back to `mass_kg` when no better mass is available.
- EPA `TWC` derives the class from current `mass_kg`.
- Non-EPA `TWC` uses available TWC/ETW/inertia fields.
- Manual `test_mass_kg` cannot be lower than curb weight in the VDE Setup helpers/UI path.

### TOTAL vs NET

Decision captured for future workflow:

```text
ABC_TOTAL -> VDE_TOTAL
ABC_NET = ABC_TOTAL - ABC_TRANS only when transmission losses are available
ABC_NET -> VDE_NET only after ABC_NET exists
```

Transmission losses should not be invented in this tire sprint.

## Current Implementation Map

Core tire model:
- `src/vde_core/roadload/tire_model.py`

Tire service:
- `src/vde_core/tire_roadload_service.py`

Mass/test-mass service:
- `src/vde_core/vde_setup_service.py`
- `src/vde_core/test_mass.py`

Persistence:
- `src/vde_core/db.py`
- `src/vde_core/repositories/tire_roadload_repository.py`
- `src/vde_core/repositories/vde_tire_repository.py`

UI integration:
- `pages/Tire_Database.py`
- `src/vde_app/components/vde_setup.py`

Tests:
- `tests/test_tire_model.py`
- `tests/test_tire_roadload_service.py`
- `tests/test_vde_setup_service.py`

## Checklist Status

### Tire DB

- [x] Tire roadload DB schema exists.
- [x] `tire_test_code` is unique and indexed.
- [x] Tire records can be created and updated through service/repository paths.
- [x] Tire records can be listed/searched.
- [x] Tire records can be deactivated or deleted through repository paths.
- [x] `rr_n_per_kn` is stored.
- [x] SAE Smerf preview/calculation exists.
- [~] SAE/ISO validation exists in service/model paths, but UI-side form validation is still light.

### Tire Model

- [x] SAE tire ABC calculation.
- [x] SAE Smerf / `rr_n_per_kn` calculation.
- [x] ISO simple RRC mode.
- [x] Front/rear axle load calculation.
- [x] Front/rear pressure support for SAE.
- [x] Front weight distribution support.
- [x] Same tire front/rear support through service/UI payload.
- [x] Tire improvement percentage support.

### VDE Setup Integration

- [x] Minimal Tire Roadload Preview block exists.
- [x] VDE Setup selects/searches tire records by `tire_test_code` in the preview UI.
- [x] VDE Setup fetches tire data from `tire_roadload_db`.
- [x] VDE Setup resolves tire calculation mass.
- [x] VDE Setup displays tire load mass used in preview.
- [x] Preview does not write to the database.
- [x] Save explicitly persists final tire application to `vde_db`.
- [x] Save does not duplicate full tire coefficients into `vde_db`.

### Tests

- [x] Tire model tests cover SAE, ISO, axle loads, pressure/load units and improvement.
- [x] Tire roadload service tests cover mass basis, preview, save and payload validation.
- [x] VDE Setup service tests cover EPA default test mass, WLTP hooks, TWC and manual mass rules.
- [x] Tire code lookup has a service-level test.
- [ ] Full local test suite still needs execution in a working Python environment.

Test execution attempted on 2026-06-17:

```text
python -m unittest tests.test_tire_model tests.test_tire_roadload_service tests.test_vde_setup_service -v
```

Blocked because the local Windows Python launcher failed before running tests:

```text
python.exe: A sessao de logon especificada nao existe.
py: command not found.
.venv\\Scripts\\python.exe points to a missing/broken WindowsApps Python.
```

## Known Gaps / Review Items

- Decide whether `vde_db` should store front/rear tire code snapshots in addition to ids.
- Tighten Tire Database UI validation for required SAE/ISO fields if desired.
- Revisit EPA inertia class granularity and validation later.
- Finish WLTP legislation UI inputs for MRO/TPMLM/category once the rule set is confirmed.
- Review VDE Setup layout separately; current UI is intentionally minimal.
- Clean any encoding artifacts visible in labels/docs before final release if they appear in the app.

## Future Sprint Candidate

Suggested next sprint:

```text
VDE Setup Workflow Redesign & Component Build-up Mock
```

Future scope:
- Source -> ABC_TOTAL -> Mass -> Components -> Transmission -> Preview -> Save workflow.
- Baseline/New Line/From Test/Calculated from PL modes.
- Component cards for non-tire contributors.
- Transmission losses always visible.
- `ABC_TOTAL`, `VDE_TOTAL`, `ABC_NET`, `VDE_NET` preview rules.
- CSV mock component repository before any `component_db` SQLite.
- Trailer mock catalog.

## Suggested Commit Focus

Suggested commit title:

```text
feat(tire): add tire roadload database preview and VDE application flow
```

Review before commit:
- Stage only intended source/test/docs files.
- Avoid staging notebooks, generated DB files or unrelated local config.
- Run the target tests once Python is fixed locally.
