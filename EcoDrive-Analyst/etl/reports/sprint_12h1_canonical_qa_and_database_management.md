# Sprint 12H.1 — Canonical QA and Database Management compatibility

Status: `12H1_CANONICAL_QA_READY — MANUAL_BROWSER_SMOKE_REQUIRED`

## Outcome

STAGING, QA, and PROD are explicit instances of the same canonical database.
The application selects an instance only by path, canonical reads continue
through compatibility views, and VDE/FuelCons writes target canonical physical
tables. The Database Management crash on canonical `record_origin` values is
fixed with explicit conservative policies. No canonical IDs, schema, domain
semantics, physics, or population were changed by this patch.

## Database audit and file organization

The initial content/schema audit identified the Sprint 12F.13 approved
candidate as equivalent to the canonical production runtime. The old small
production and QA databases were legacy-schema files. Copies were made only
after SHA-256 and SQLite integrity checks; source backups were preserved.

| Role/file | Bytes | SHA-256 | quick_check | FK violations |
|---|---:|---|---|---:|
| STAGING `data/db/staging/eco_drive_canonical_candidate.db` | 205,099,008 | `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB` | ok | 0 |
| QA `data/db/eco_drive_qa.db` | 205,099,008 | `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB` | ok | 0 |
| PROD `data/db/eco_drive.db` | 205,099,008 | `243AA746E456E68F9944EE140D24AF2595AF4252D3FDCB418F500C31364D52AB` | ok | 0 |
| Archived legacy PROD `data/db/archive/eco_drive_legacy_pre_sprint12.db` | 7,147,520 | `CC27CDF22AE189E39F9F45609CCD23C650E822BB20655C552FF2BD89D97F4262` | ok | 0 |
| Archived legacy QA `data/db/archive/eco_drive_qa_legacy_pre_sprint12.db` | 73,728 | `0EB4A5EC0F402E6B9E1F7D76372EA30068670B16F994D495394613DAFA2CA569` | ok | 0 |

The archived legacy PROD hash matches its preserved source
`data/backups/eco_drive_pre_sprint12_20260915.db`. The approved canonical
source, STAGING, QA, and PROD hashes match. QA and PROD remained byte-identical
through automated validation.

Final tree:

```text
data/db/
|-- eco_drive.db                              PROD
|-- eco_drive_qa.db                           QA
|-- staging/
|   `-- eco_drive_canonical_candidate.db      STAGING
`-- archive/
    |-- eco_drive_legacy_pre_sprint12.db
    `-- eco_drive_qa_legacy_pre_sprint12.db
```

All three canonical environments have the same population:

| Entity | STAGING | QA | PROD |
|---|---:|---:|---:|
| program | 3,117 | 3,117 | 3,117 |
| vehicle_configuration | 10,211 | 10,211 | 10,211 |
| vde | 11,626 | 11,626 | 11,626 |
| run | 29,250 | 29,250 | 29,250 |
| fuelcons | 10,822 | 10,822 | 10,822 |
| fuelcons_run_adoption | 18,323 | 18,323 | 18,323 |
| vde_component_resolution | 0 | 0 | 0 |

## Runtime simplification

- `configure_db_path(path)` now selects only the canonical instance.
- The transitional `prepared_canonical` flag and path/schema-based runtime
  routing were removed.
- Compatibility reads remain on `vde_db` and `fuelcons_db`; writes go to `vde`
  and `fuelcons`.
- Legacy schema creation/routing is confined to explicit isolated fixture
  contexts so the historical unit-test suite remains usable.
- `ECO_DRIVE_DB_PATH=data/db/eco_drive_qa.db` explicitly selects QA. The full
  launch command is documented in `docs/DATABASE_ENVIRONMENTS.md`.
- Comparison Report's QA button selects the canonical QA file and cannot seed
  or overwrite it.

## Canonical record-origin audit and policy

| Entity/table | Origin | Count | Policy |
|---|---|---:|---|
| VDE / `vde` | `SOURCE_REFRESHED` | 9,907 | source-managed, protected/read-only |
| VDE / `vde` | `NEW_SOURCE` | 1,719 | source-managed, protected/read-only |
| FuelCons / `fuelcons` | `EPA_RECONSTRUCTED` | 10,572 | authoritative reconstruction, protected/read-only |
| FuelCons / `fuelcons` | `NEW_SOURCE` | 249 | source-managed, protected/read-only |
| FuelCons / `fuelcons` | `ML_PREDICTION` | 1 | model-derived, protected/read-only |
| Tire / `tire_db` | `OTHER_IRREPRODUCIBLE_STATE` | 1 | ambiguous/irreproducible source state, protected/read-only |
| Component / `component_db` | no `record_origin` column | 0 | empty canonical population; renders safely |

The allowed-origin lists remain closed. There is no arbitrary-string fallback
and no canonical source origin maps to `MANUAL`. Existing `MANUAL` and
`VDE_SETUP` field policies remain editable as before; existing legacy and
reference policies are unchanged. Source-managed create choices and lifecycle
actions are unavailable in the UI, and a protected update is rejected during
preview.

Negative imported/reconstructed IDs were deliberately retained. They are
stable namespace keys and their related foreign keys were not renumbered.

## Code and tests

Principal runtime changes:

- `app.py`, `pages/Comparison_Report.py`
- `src/vde_core/db.py`
- `src/vde_core/repositories/vde_repository.py`
- `src/vde_core/repositories/fuelcons_repository.py`
- `src/vde_core/repositories/tire_roadload_repository.py`
- `src/vde_core/vde_request_save.py`
- `src/vde_core/vde_workflow_service.py`
- `src/vde_core/component_repositories.py`
- `src/vde_core/database_management_contract.py`
- `src/vde_core/database_management_policy.py`
- `src/vde_core/database_management_service.py`
- `src/vde_core/database_management_impact_service.py`
- `src/vde_app/components/database_management.py`

Added Sprint 12H.1 coverage in
`etl/tests/test_sprint_12h1_canonical_qa_database_management.py`; updated
affected DB-path, Database Management, Comparison, and historical fixture
tests. AppTest proves that all four Database Management tabs render against
canonical QA without an origin-normalization exception. AppTest is not counted
as the required manual browser smoke.

Validation completed so far:

- focused Database Management suite: **53/53 passed**;
- DB path, canonical integration, VDE Setup/save, repositories/services, and
  AppTest/smoke selection: **204/204 passed**;
- focused VDE Setup helper/AppTest module after canonical Tire lookup routing:
  **54/54 passed**;
- complete historical application suite: **1,840/1,840 passed** in 610.142 s;
- canonical STAGING/QA/PROD `quick_check`: **ok** for all three;
- canonical STAGING/QA/PROD `foreign_key_check`: **0 rows** for all three.

## Known limitations and manual gate

- The canonical component catalog is empty, so component browsing is validated
  as a safe empty state rather than with populated canonical rows.
- The canonical Tire catalog contains one protected row (`26AA`) with
  `is_active=0`; normal VDE Setup Tire lookup therefore renders a safe empty
  active catalog until an approved active Tire record exists.
- The frozen canonical database does not contain the historical Database
  Management audit/history tables. This hotfix therefore does not invent a
  schema migration; canonical source-managed rows are safely inspectable and
  read-only. Any future expansion of direct canonical catalog mutation/audit is
  a separate schema decision.
- A real browser smoke against `data/db/eco_drive_qa.db` is still required for
  Browse, Database Management, VDE Setup Preview/save, EPA/WLTP Comparison,
  Quick Scenario, and Powertrain Scenario. Sprint 12 is not closed by this
  report.
