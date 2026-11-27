# Sprint 16A — Architecture & Dependency Audit

Date: 2026-11-27

Branch: `sprint-16a-packaging-audit`

Reference: `origin/main` at `efbf8b9510843b2d898883ee21934d4030e4d036`

Scope: audit and packaging proposal only

Packaging readiness: **CONDITIONAL — not ready for a self-contained wheel**

## 1. Evidence convention and executive result

- **[EXECUTED]** means a command, import, artifact load, or test was run in the
  Sprint 16A worktree.
- **[STATIC]** means the conclusion comes from AST/source/repository inspection
  and was not exercised as a runtime behavior.
- **[PRIOR EVIDENCE]** means the result is quoted from a versioned closure or
  gate document and was not rerun for this audit.

No production code, physics formula, database schema, persisted record,
namespace, dependency, or `pyproject.toml` was changed. The only intended
versioned deliverable is this report.

The current layout can preserve `src.vde_core` and `src.vde_app` during an
initial packaging phase. It cannot yet be treated as a normal setuptools
`src/` layout: `src` is itself the public top-level package. Package discovery
must therefore start at the repository root and include `src*`. Runtime files
also depend on repository-relative data, pages, images, templates, a model and
SQLite paths, so import success alone does not prove wheel readiness.

Principal 16B blockers:

1. normalize resource and database path ownership without changing behavior;
2. separate direct metadata from the current environment-style pinned lists;
3. declare missing direct dependencies (`openpyxl`, `pytest`, and the direct
   `pydantic` use) and decide whether `matplotlib` is an ETL optional extra;
4. preserve the literal `src.*` namespace with namespace-aware discovery;
5. remove or isolate path bootstrapping, bare `vde_core` imports and the one
   core-to-app dependency before claiming clean layer packaging;
6. make the regression fixture contract deterministic before using a green
   suite as a release gate.

## 2. Baseline and Sprint 15 gate

**[EXECUTED]** `git fetch origin --prune` completed successfully. Local `main`
and `origin/main` both resolved to `efbf8b9510843b2d898883ee21934d4030e4d036`,
the merge of PR #23 (`sprint-15e-stabilization`). This audit was made in a
separate worktree from that exact commit. The user's existing main worktree and
its modified `EcoDrive-Analyst/data/db/eco_drive.db` were not used as the test
database and were not modified.

**[STATIC]** `docs/sprints/SPRINT_15_VISUAL_INTEGRATED_CLOSURE.md` remains
`PENDING`.

**[PRIOR EVIDENCE]** `docs/sprints/SPRINT_15E_EXECUTION_REPORT.md` records:

- real-browser G4 evidence for all eight pages at 1440x900 and 1024x768 is
  absent;
- owner approval is absent;
- 14 fixture/environment failures remain visible;
- a merged implementation PR is explicitly insufficient to close Sprint 15.

Sprint 16A does not close or bypass that gate.

## 3. Executed Python environment

| Evidence | Result |
|---|---|
| **[EXECUTED]** Python | CPython 3.11.9, MSC v.1938, 64-bit |
| **[EXECUTED]** Platform | Windows 10 API build 26100 (`Windows-10-10.0.26100-SP0`) |
| **[EXECUTED]** Interpreter | `EcoDrive-Analyst/.venv/Scripts/python.exe` from the existing main checkout |
| **[EXECUTED]** Virtual environment | yes; `sys.prefix` differs from `sys.base_prefix` |
| **[EXECUTED]** pip | 26.1.2 |
| **[EXECUTED]** installed distributions | 114 |
| **[EXECUTED]** `python -m pip check` | `No broken requirements found.` |

The worktree deliberately reused the existing virtual environment; no package
was installed, removed or upgraded. `pip check` proves that installed metadata
is internally satisfiable. It does not prove that all imports used by this
repository are declared or installed.

The three requirement files contain 52 non-overlapping pinned entries:

| File | Entries | Current role |
|---|---:|---|
| `requirements.txt` | 40 | mixed runtime, transitive and development environment lock |
| `requirements-ml.txt` | 5 | ML direct plus one transitive dependency |
| `requirements-research.txt` | 7 | research direct plus compatibility/umbrella pins |

All 52 declared distributions were installed. There were 62 additional
installed distributions, predominantly transitive dependencies plus local
development tools. Important direct-use exceptions are discussed below.

## 4. AST import audit by layer

### Method

**[EXECUTED]** A temporary Python AST audit parsed every tracked `*.py` file
returned by `git ls-files`. It collected `Import`, `ImportFrom`, relative
imports, imports inside guarded `try` blocks, known dynamic-import calls,
serialization calls, literal path calls and internal module edges. Strongly
connected components were calculated over the internal import graph. The
temporary audit utility is not part of the branch. No parse error occurred.

The optional `ecodrive_sprint16a_audit.py` mentioned in the request was not
present in the repository or workspace, so it could not be reused.

### Inventory

| Layer | Python files | Internal import occurrences | Relative import occurrences | Direct third-party roots observed |
|---|---:|---:|---:|---|
| app (`app.py`, `pages/`, `src/vde_app/`) | 42 | 183 | 8 | `numpy`, `pandas`, `plotly`, `streamlit` |
| core (`src/vde_core/`) | 117 | 310 | 66 | `joblib`, `numpy`, `pandas`, `scipy`, `xgboost` |
| ML (`models/`) | 4 | 3 | 0 | `joblib`, `pandas`, `scikit-learn`, `xgboost` |
| research (`capabilities/technical_research/`, excluding tests) | 53 | 114 | 114 | `beautifulsoup4`, `certifi`, `ddgs`, `langchain-openai`, `langgraph`, `pydantic`, `pypdf`, `requests` |
| ETL (`etl/`, excluding tests) | 25 | 11 | 0 | `numpy`, `openpyxl`, `pandas`, `scikit-learn`, `scipy` |
| tests (all tracked test modules) | 154 | 484 | 0 | application dependencies plus `matplotlib`, `pytest` |
| scripts/archive/other | 53 | 119 | 0 | mixed operational tooling |
| **Total** | **448** | **1,180 graph edges** | — | — |

Counts are occurrences, not unique modules. The `tests` layer includes tests
under `tests/`, `etl/tests/` and `capabilities/technical_research/tests/`.

### Cycles and layer direction

**[STATIC]** Two real strongly connected components were found:

1. `src.vde_core.db` → `src.vde_core.services` →
   `src.vde_core.fuel_energy` → `src.vde_core.db`. The last edge is a local
   import in `_get_vde_row`, which avoids import-time failure but retains the
   architectural cycle.
2. `capabilities.technical_research.core.request_audit` →
   `capabilities.technical_research.core.matching` →
   `capabilities.technical_research.core.request_audit`. The return edge is a
   local import in `match_application`.

**[STATIC]** The Vehicle Demand frozen boundary does not import System Scenario;
System Scenario imports Vehicle Demand, which is the intended direction.

**[STATIC]** One production core file violates the project-wide layer rule:
`src/vde_core/vde_request_compact_state.py` imports
`src.vde_app.components.vde_request_domain_editors`. This audit does not move
or rewrite it, but 16B should extract the non-UI state helpers to a neutral
core owner before splitting distributions.

### Import-path debt

**[STATIC]** The repository contains:

- 62 Python files that modify `sys.path`;
- 6 files importing `vde_core` without the canonical `src.` prefix;
- 5 files that call `os.chdir`;
- ETL scripts/tests importing sibling scripts as top-level modules;
- 27 app/page files importing `src.vde_core`, which is expected;
- no production core import of System Scenario from Vehicle Demand.

These path mutations explain why executing from the repository root works even
when modules are not discoverable from another working directory. They should
not be silently encoded into packaging metadata.

## 5. Dependency classification

The classifications below are based on direct AST use and executed artifact
inspection. A distribution may be direct for one extra and transitive for
another.

### Direct runtime candidates

| Distribution | Observed version | Evidence and proposed group |
|---|---:|---|
| `numpy` | 2.3.3 | app/core calculations; base runtime |
| `pandas` | 2.3.2 | app/core data frames and loaders; base runtime |
| `plotly` | 6.3.0 | app visualization; base runtime |
| `scipy` | 1.17.1 | core estimator and ETL; base engineering runtime or an explicit engineering extra |
| `streamlit` | 1.49.1 | application shell; app runtime |

### Optional feature dependencies

| Group | Direct distributions | Notes |
|---|---|---|
| ML | `joblib`, `scikit-learn`, `xgboost` | `scikit-learn` is also required to unpickle the committed pipelines; `scipy` is already a core/ETL direct use |
| research | `beautifulsoup4`, `certifi`, `ddgs`, `langchain-openai`, `langgraph`, `pydantic`, `pypdf`, `requests` | `pydantic` is directly imported but currently only arrives transitively |
| ETL | `openpyxl`, `numpy`, `pandas`, `scikit-learn`, `scipy` | `openpyxl` is directly imported but absent from all requirement files and from the executed environment |
| ETL experiment | `matplotlib` | imported late by a transmission experiment and by an ETL test; absent from declarations and environment |

There were 17 imports guarded by `try`/`ImportError`-style blocks. Runtime
significant cases include `langchain_openai`, `ddgs`, `plotly.express`,
`xgboost`, `joblib` and `pandas`. Guarding an import makes a feature optional at
runtime; it does not by itself define a supported packaging extra.

### Development dependencies

| Distribution | Status |
|---|---|
| `pytest` 9.1.1 | installed and directly imported by tests, but undeclared |
| `ipykernel` 6.29.2 | declared in the base file; should move to a development/notebook group |
| `matplotlib` | missing; decide between ETL optional and development/test groups |

The suite also uses standard-library `unittest` heavily; that is not an
external dependency.

### Transitive/environment pins

The following entries in `requirements.txt` had no direct application import
and are dependency/environment pins rather than direct project metadata:

`altair`, `attrs`, `blinker`, `cachetools`, `charset-normalizer`, `click`,
`colorama`, `gitdb`, `GitPython`, `idna`, `Jinja2`, `jsonschema`,
`jsonschema-specifications`, `MarkupSafe`, `narwhals`, `packaging`, `pillow`,
`protobuf`, `pyarrow`, `pydeck`, `python-dateutil`, `pytz`, `referencing`,
`rpds-py`, `six`, `smmap`, `tenacity`, `toml`, `tornado`,
`typing_extensions`, `tzdata`, `urllib3`, and `watchdog`.

In `requirements-ml.txt`, `threadpoolctl` is transitive. In
`requirements-research.txt`, no direct AST import of `langchain` or `openai`
was found; both may remain compatibility pins initially, but should not be
mislabelled as observed direct imports.

The recommended model is direct dependencies in `pyproject.toml` plus a
separate reproducible lock/constraints mechanism. Copying all 114 installed
distributions into project metadata would preserve accidental environment
state, not define the project contract.

## 6. Dynamic imports and ML serialization

### Dynamic imports

**[STATIC]** Nine `__import__` calls were found. Eight are in tests/ETL tests
and use literal standard-library, pandas or internal module names. The only
production call is a literal `__import__("hashlib")` in
`src/vde_core/component_reference_discrimination.py`. No non-literal plugin
loader or entry-point discovery was found by the AST audit.

### Serialization

**[STATIC]** Serialization calls are:

- `joblib.dump` in `models/scripts/export_powertrain_ml_artifact.py`;
- `joblib.load` in `src/vde_core/ml_prediction.py`;
- `pickle.load` as the fallback for non-joblib artifacts in the same loader.

**[EXECUTED]** The committed
`models/powertrain_scenario_ml.joblib` was loaded from the isolated worktree:

| Property | Result |
|---|---|
| size | 17,461,719 bytes |
| SHA-256 | `6826516f4fb256b9765958d52974df0660a6c585c5e56ae9b5f7a443ed2aee2d` |
| artifact type | `dict` |
| model version | `notebook_export_20260630T030228Z` |
| predictor class | `src.vde_core.ml_prediction.NotebookPowertrainPredictor` |
| BEV pipeline | scikit-learn `ColumnTransformer` + `RandomForestRegressor` |
| non-BEV pipeline | scikit-learn `ColumnTransformer` + `MultiOutputRegressor` wrapping the XGBoost path |
| executed loader versions | joblib 1.5.3, scikit-learn 1.9.0, scipy 1.17.1, xgboost 3.2.0 |

The artifact metadata records model/data facts but not the training/runtime
library versions. Joblib/pickle also executes trusted Python object
reconstruction and is class-path/version coupled. 16B should keep this asset
outside the base wheel or explicitly version and hash it, add a compatibility
manifest, reject untrusted paths, and add a clean-environment load smoke test.

## 7. Paths, resources and persistence

### Path findings

**[STATIC]** The AST audit found 131 calls with literal path-like arguments.
The important runtime categories are:

- **working-directory relative:** `data/db/eco_drive.db`,
  `data/db/eco_drive_qa.db`, `data/db/staging/...`, `data/cycles`,
  `data/reference/tire_size_reference.csv`, `data/quick_tech_deltas.csv`, and
  `data/vehicles/vehicles.csv`;
- **module/repository-root resolved:** ML artifact globs, standards CSVs,
  methodology images, logo/image lookup and the VDE request XLSX template;
- **script-specific:** many ETL and research scripts derive a repository root,
  insert it or `etl/scripts` into `sys.path`, and write under `artifacts/` or
  `etl/data/`.

**[EXECUTED]** From the application repository root, imports of
`src.vde_core` and `src.vde_app` succeeded. From the parent worktree directory,
`importlib.util.find_spec("src")` and `find_spec("capabilities")` both returned
`None`. This proves the current behavior depends on the checkout root being on
`sys.path`; an installed distribution would resolve it, but a bare checkout
from an arbitrary CWD does not.

### Resource inventory

**[STATIC]** Tracked top-level payload sizes include:

| Root | Files | Bytes |
|---|---:|---:|
| `data/` | 414 | 119,366,230 |
| `models/` | 5 | 17,482,357 |
| `docs/` | 121 | 1,161,305 |
| `notebooks/` | 17 | 2,571,795 |

Selected tracked binary/resource inventory:

- 359 PNG files (105,388,129 bytes) and 34 JPG files (1,862,517 bytes), mostly
  logos plus application illustrations;
- five cycle CSVs under `data/cycles/`;
- standards/reference/preset CSVs used at runtime;
- two request XLSX templates under `docs/templates/`;
- one 7,147,520-byte tracked SQLite database;
- one 17,461,719-byte joblib artifact.

Blindly enabling recursive package data would produce a large wheel and could
embed the production database. 16B needs an explicit resource allowlist and a
separate policy for logos, model artifacts, templates and demo/test data.

### Database and fixture findings

**[STATIC]** Only `data/db/eco_drive.db` is tracked. `.gitignore` ignores all
other `*.db` files. The runtime path constants are CWD-relative, and two QA
defaults disagree:

- `src.vde_core.db.QA_DB_PATH` → `data/db/eco_drive_qa.db`;
- `src.vde_core.qa_mock_data.DEFAULT_QA_DB_PATH` →
  repository-root `data/qa/eco_drive_qa.db`.

The table adapter also selects `tire_roadload_db` when
`LEGACY_FIXTURE_MODE=True`, but `tire_db` in normal canonical mode. The current
synthetic seed is created inside `using_db_path(..., legacy_fixture=True)`, so a
seeded file contains `tire_roadload_db`; returning to normal mode against that
same file queries `tire_db`. This is a fixture/runtime contract mismatch, not a
packaging formula issue.

**[EXECUTED]** The full regression run used the synthetic database
`.sprint16a_runtime/qa/eco_drive_qa.db`, seeded specifically inside the isolated
worktree. Seed digest before and after the suite was identical:
`5d2f0f28b718085c0fbc73ab8f581dc85459ff97136fb84f8e14a7c79ed867b0`.

**[EXECUTED]** The isolated worktree copy of `data/db/eco_drive.db` retained
SHA-256
`cc27cdf22ae189e39f9f45609ccd23c650e822bb20655c552ff2bd89d97f4262`.
The user's main-worktree database retained its different pre-existing SHA-256
`243aa746e456e68f9944ee140d24af2595af4252d3fdcb418f500c31364d52ab`.
No PROD or STAGING database was modified.

## 8. Namespace preservation and package discovery

Preserving `src.vde_core` and `src.vde_app` is viable and recommended for the
first packaging increment.

**[EXECUTED]** From the project root, `setuptools.find_packages(".")` found
`src`, `src.vde_core`, `src.vde_app`, their regular subpackages,
`capabilities`, and the research packages. Namespace-aware discovery also found
`src.vde_core.estimation`, which has no `__init__.py` and is omitted by classic
`find_packages`.

Consequences:

- do **not** configure the conventional `package-dir = {"" = "src"}` layout;
  that would expose `vde_core`/`vde_app` and break the established imports;
- discover from `where = ["."]`, include `src*` and `capabilities*`, and enable
  namespace discovery so `src.vde_core.estimation` is not lost;
- exclude research tests and all ordinary test packages from the runtime
  distribution;
- do not assume package discovery includes `app.py`, Streamlit `pages/`, data,
  images, templates, notebooks, databases or the model artifact;
- add installed-wheel smoke tests from a directory outside the repository.

The import namespace is therefore low risk; resource location and application
entry-point behavior are the higher risks.

## 9. Proposed `pyproject.toml` structure for Sprint 16B

No file was created in Sprint 16A. The following is a design outline, not
committed metadata:

```toml
[build-system]
requires = ["setuptools==65.5.0"]  # preserve the audited build tool initially
build-backend = "setuptools.build_meta"

[project]
name = "ecodrive-analyst"
requires-python = ">=3.11,<3.12"  # initial verified interpreter envelope
dependencies = [
  "numpy==2.3.3",
  "pandas==2.3.2",
  "plotly==6.3.0",
  "scipy==1.17.1",
  "streamlit==1.49.1",
]

[project.optional-dependencies]
ml = ["joblib==1.5.3", "scikit-learn==1.9.0", "xgboost==3.2.0"]
research = [
  "beautifulsoup4==4.15.0",
  "certifi==2025.8.3",
  "ddgs==9.13.0",
  "langchain-openai==1.6.2",
  "langgraph==1.2.11",
  "pydantic==2.13.5",
  "pypdf==6.19.0",
  "requests==2.32.5",
  # retain langchain/openai pins only if the 16B compatibility smoke proves
  # that the direct integrations require them as explicit project metadata
]
etl = [
  "scikit-learn==1.9.0",
  # openpyxl is required, but its version must be selected and verified in a
  # controlled 16B environment; decide whether matplotlib belongs here too
]
dev = [
  "pytest==9.1.1",
  "ipykernel==6.29.2",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "capabilities*"]
exclude = ["tests*", "*.tests*", "docs*", "etl*", "models*", "scripts*"]
namespaces = true
```

The exact project version source, entry point, resource allowlist, license/readme
metadata, research constraints and ETL pins must be decided and verified in
16B. `requires-python` should not be widened until CI proves another Python
minor. Initially preserving the audited exact versions minimizes behavioral
change; dependency modernization is a separate sprint.

Recommended dependency outputs in 16B:

1. direct metadata in `pyproject.toml`;
2. reproducible constraints/lock files for application, ML, research and ETL;
3. a development group containing pytest/notebook tooling;
4. an artifact manifest with SHA-256 and serialization-library versions;
5. no database in build artifacts.

## 10. Regression execution

### Isolation

**[EXECUTED]** Command:

```text
ECO_DRIVE_DB_PATH=<isolated-worktree>/.sprint16a_runtime/qa/eco_drive_qa.db
python -m pytest tests -q --tb=short
```

The database was freshly generated from `seed_qa_database`; no PROD/STAGING
path was supplied. Some AppTest cases explicitly switch to the repository's QA
constant and created an ignored zero-byte QA path inside the isolated worktree.
That behavior did not reach the user's main checkout and is itself recorded as
a fixture-path risk.

### Real result

**[EXECUTED]** `2041 passed, 15 failed, 7 skipped, 190 subtests passed` in
`515.57s` (8m35s), with three Python deprecation warnings in
`pages/home_page.py` for invalid escape sequences.

Failure inventory:

| Count | Test area | Classification |
|---:|---|---|
| 2 | `ComparisonReportDbPathSelectorTests` | known baseline QA-path fixture: selected `data/db/eco_drive_qa.db` lacks `fuelcons_db` |
| 2 | Database Management UI 7C/7G | known baseline repository fixture gap |
| 7 | `test_research_agent_final_micropatch.py` | known missing Sprint 12 research CSV package |
| 2 | `test_technical_research_phasea.py` | known missing Sprint 12 P0 research CSV package |
| 1 | `test_request_inputs_tire_lookup_uses_ctx_db_path_for_live_repository_reads` | known baseline legacy fixture lacks `tire_db` |
| 1 | `test_request_inputs_tire_lookup_pending_widgets_apply_and_walk_from_inherit` | additional isolated-seed mismatch: normal runtime queries `tire_db`, seed contains `tire_roadload_db` |

The first 14 IDs match the versioned Sprint 15E baseline exactly. The 15th
failure was rerun directly with the same isolated database and reproduced as
`sqlite3.OperationalError: no such table: tire_db`; the test then cannot find
the `Apply Tire` button. It is an environment/fixture-contract delta exposed by
the stricter isolated seed. Because Sprint 16A changes no executable code, no
new code regression is attributable to this branch. The suite is nevertheless
**not green**, and 16B must not relabel these failures as passing.

The root `tests/` suite is the established full-suite command recorded in the
Sprint 15E execution report. ETL and research subproject tests were statically
audited but were not added to this regression command; doing so would expand
the established gate and currently encounters undeclared optional tools and
external fixture packages.

## 11. Risk register

| Priority | Risk | Evidence | 16B control |
|---|---|---|---|
| P0 | resource paths work only from repository layout/CWD | CWD-relative DB, cycles and catalogs; imports absent from parent directory | central read-only resource resolver; installed-wheel smoke from external CWD |
| P0 | database/QA path and schema-mode mismatch | two QA defaults; `tire_db` vs `tire_roadload_db`; test result | define one canonical test DB factory/path contract without schema migration |
| P0 | wheel may accidentally include PROD DB or 100+ MB images | tracked payload inventory | explicit package-data allowlist; assert no `*.db` in wheel |
| P0 | serialized model is version/class-path coupled | executed joblib load and missing library-version metadata | artifact manifest, trusted path policy, compatibility test |
| P1 | dependency metadata is incomplete | direct `openpyxl`, `pydantic`, `pytest`; missing `matplotlib` | add direct groups in controlled environment; run `pip check` and import smoke |
| P1 | import graph is not cleanly layered | two cycles and one core→app edge | extract neutral state helper; keep physics untouched; add architecture test |
| P1 | scripts depend on `sys.path`, bare imports and CWD changes | 62/6/5-file counts | package-aware module entry points; deprecate path injection incrementally |
| P1 | classic discovery omits namespace package | `src.vde_core.estimation` only found namespace-aware | `namespaces = true`; assert expected package list in build test |
| P2 | base requirements mix direct/transitive/dev | 40-entry base audit | direct pyproject metadata plus separate lock/constraints |
| P2 | Python support envelope is unproven beyond 3.11 | only 3.11.9 executed | CI matrix before widening `requires-python` |

## 12. Recommended Sprint 16B sequence and acceptance evidence

1. Add the minimal `pyproject.toml` with literal `src.*` discovery and no
   namespace migration.
2. Add a build test that inspects wheel contents and fails if it contains any
   SQLite file, unexpected model, notebook, ETL output or bulk image corpus.
3. Install the wheel into a fresh virtual environment and verify imports from a
   directory outside the checkout.
4. Add explicit import smokes for base, `[ml]`, `[research]` and `[etl]` extras.
5. Introduce one resource-path owner for cycles, standards, presets, templates,
   images and artifacts; preserve the existing values and provenance.
6. Define a single synthetic DB fixture contract that is explicitly canonical
   or legacy at every call site; do not migrate PROD/STAGING.
7. Record and freeze the ML artifact hash plus Python/joblib/scikit-learn/scipy/
   xgboost compatibility metadata.
8. Run `pip check`, package discovery assertions, the established root test
   suite, relevant ETL/research tests with their fixtures, and wheel-installed
   smoke tests.
9. Compare every failure ID with this report and the 15E baseline. Do not claim
   a green release while fixture failures remain.
10. Leave the Sprint 15 visual gate pending until its independent browser and
    owner evidence is complete.

Sprint 16B should stop and report before any namespace migration, physical
formula change, persistence migration, PROD/STAGING write, dependency upgrade,
or attempt to make curves/results differ for presentation.
