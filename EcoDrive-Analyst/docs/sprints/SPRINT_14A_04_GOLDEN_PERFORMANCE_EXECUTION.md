# Sprint 14A-04 — Golden Physics & Performance Baseline

**Status:** EXECUTION PACKAGE — READY TO RUN, **NOT EXECUTED** (2026-11-07).  
**Scope:** Sprint 14 design-only; capture before implementing RoadLoadModel interfaces.  
**Dependencies:** `SPRINT_14A_ROADLOAD_DESIGN_GATE.md`, `roadload_interface_contract_v0.1.md`, `roadload_traceability_v0.1.md`.  
**Repository baseline examined:** `main@fca9a9ad189fe7e6930e987e29fee5717c0d532e` (2026-11-07; verify current HEAD on execution).

## A. Frozen invariants

1. `F = A + B v_kph + C v_kph²`; `P_road = F * v_kph / 3.6` W. `compute_vde_series` takes SI `v` in m/s and converts to km/h once.
2. `TOTAL = authoritative coastdown ABC`, `NET = TOTAL − neutral transmission drag ABC`, but only when the transmission boundary is resolved; missing NET must remain missing.
3. On a source-authoritative basis, known physical contributions + **signed residual** = total at every speed, including negative residual. No attribution of residual to a fictitious part.
4. No double count when transmission already includes axle/hub losses; overlap is a model-contract error, not a physical energy term.
5. `None != measured zero`, no divide by zero in fractions, and mathematical `A/Bv/Cv²` fractions are not physical component shares.
6. `VDE positive tractive energy` is not interchangeable with signed roadload work; preserve canonical `compute_vde_series` and cycle integration.
7. Previews, Load Case and golden scripts perform no DB writes; save actions are explicit and separate.

## B. Minimum execution order (at EcoDrive-Analyst root)

```bash
# Ensure worktree is pristine and freeze SHA
 git status --short
 git rev-parse HEAD
 python --version
 python -m pip show numpy pandas pytest

# 1) AST inventory: includes repo scripts/tests, not notebooks or dynamic calls
python scripts/architecture/sprint14a04_inventory.py

# 2) Existing tests: granular output retained as evidence
mkdir -p artifacts/roadload_14a04
python -m pytest -q \
    tests/test_roadload_engine.py \
    tests/test_tire_model.py \
    tests/test_tire_roadload_service.py \
    tests/test_vehicle_demand_engine.py \
    tests/test_vehicle_demand_integration.py \
    tests/test_vehicle_demand_load_case.py \
    tests/test_comparison_report_service.py \
    tests/test_vde_workflow_service.py \
    tests/test_vde_request_resolver.py \
    --junitxml=artifacts/roadload_14a04/selected_tests.xml \
    > artifacts/roadload_14a04/selected_tests.log 2>&1

# 3) Synthetic golden reference captured strictly BEFORE any refactoring
python scripts/architecture/sprint14a04_golden.py capture

# 4) Measurements. Quick smoke first, then full CPU baseline
python scripts/benchmarks/bench_roadload_baseline.py --quick --repeats 5 \
    --out artifacts/roadload_14a04/bench_smoke.json
python scripts/benchmarks/bench_roadload_baseline.py --repeats 20

# 5) Optional suite, only if environment/resources allow
python -m pytest tests -q \
    --junitxml=artifacts/roadload_14a04/all_tests.xml \
    > artifacts/roadload_14a04/all_tests.log 2>&1
```

**Keep artifacts local/private by default.** The `artifacts/` output is a suggested local directory. Confirm `.gitignore` before committing any logs, environment details, database paths or snapshots. Never commit internal vehicle data.

## C. Acceptance matrix (do not claim pass without artifacts)

| Case | Test/measurement | Target evidence | Current status |
|---|---|---|---|
| AC-01/02 | scalar/vector force & power | golden JSON + analytic `F(100)=450 N`, `P(100)=12500 W` | NOT EXECUTED |
| AC-03/04 | partial and negative residual | existing Vehicle Demand tests + frozen `signed_residual_negative_N`; add tests for RoadLoadModel after implementation | NOT EXECUTED |
| AC-05 | missing vs zero, division by ≈0 | existing coverage audit; new target tests required | GAP |
| AC-06 | NET without transmission | `test_vehicle_demand_*`, Comparison test | NOT EXECUTED |
| AC-07 | scope overlap transmission/axle/hub | contract review + future explicit overlap test | GAP |
| AC-08 | 1/10/100/1000 scenarios + 100/1000/10000 samples | `bench.json` p50/p95, Python allocations | NOT EXECUTED |
| AC-09 | temporary scenario/no write | Load Case tests and DB-write instrumentation | NOT EXECUTED |
| AC-10 | canonical VDE parity | frozen golden + actual QA/EPA/WLTP suites | NOT EXECUTED |
| NFR-01 | no domain imports DB/Streamlit/Plotly | AST import audit (manual confirmation for dynamic) | NOT EXECUTED |
| NFR-02 | SQL query count | instrument repository/service entrypoints on isolated QA DB | GAP; not microbenchmarked |
| NFR-04 | UI/render p50/p95 | AppTest + browser smoke separate | GAP; not microbenchmarked |

## D. Measurements — strict interpretation

- `bench_roadload_baseline.py` measures **current implementations**, no fake RoadLoadModel `breakdown()` or `build_up()` benchmark. `numpy` array inputs are used, not Python lists.
- Synthetic data benchmark can establish reproducibility and scaling only; it **does not validate vehicle physics**, nor measure actual multi-vehicle SQL access.
- Capture and performance harness should be executed on baseline commit **and then** against post-refactor commit in the same environment; compare p50/p95 and observed variation without arbitrary pass thresholds.
- One warmup call per measurement; a benchmark session needs hardware, OS, Python/NumPy versions, process conditions, repeat count, worktree SHA. `tracemalloc` is not RSS; add external RSS profiling if material.
- To measure SQL: use a *read-only or isolated QA copy* and count statements at the repository boundary; do not count arbitrary `SELECT *` outside the actual application flow and call that `load_vehicle()` performance.
- To measure Streamlit: time actual route/render and report *manual browser smoke* separately from AppTest. No inference of browser UX from microbenchmarks.

## E. Changes permitted in Sprint 14A-04

**Allowed:** `scripts/architecture/`, `scripts/benchmarks/`, `tests/` (non-invasive guard tests), documentation and synthetic QA fixture generation.  
**Blocked:** refactoring `vde_calc.py`, DB schema/write paths, `RoadLoadModel` public method behavior, UI workflows or physical constants to chase green tests.  
**Execution failure:** log full exception, reproduction command, environment, status `EXECUTED_FAIL`; do not auto-fix production physics.

## F. Exit gate

- Caller inventory reviewed beyond AST (notebooks, dynamic aliases, `__init__.py` reexports).
- Pre-refactor golden captured, compared to independent analytic checks (not merely self-consistent code), stored with hash and commit.
- Existing tests executed and failures classified (`pre-existing`, `new`, `environment/infrastructure`), with exact test counts.
- Measured benchmarks with environment, ranges, p50/p95, allocations and separate SQL/UI gap status.
- `roadload_traceability_v0.1.md` and `SPRINT_14A_ROADLOAD_DESIGN_GATE.md` updated with **actual evidence**, not planned evidence.
- If any gate fails, keep subsequent refactor blocked and record the blocker.
