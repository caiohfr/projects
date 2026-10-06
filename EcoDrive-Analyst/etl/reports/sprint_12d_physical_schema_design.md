# Sprint 12D — Physical Schema & Compatibility Layer Design

## Status: `PHYSICAL_SCHEMA_READY — PROCEED_TO_12E`

## Human review packet

### Physical hierarchy

```text
PROGRAM
  ↓ 1:N
VEHICLE_CONFIGURATION
  ├── 1:N COMPONENT_INSTANCE ──► COMPONENT_DB or TIRE_DB
  ↓ 1:N
VDE (wide persisted snapshot)
  ├── N:M COMPONENT_RESOLUTION via physical adoption helper
  ├── 1:N RUN
  └── 1:N FUELCONS ── N:M RUN via physical lineage helper
```

The two associative tables are physical relationship infrastructure, not new primary domain entities.

### Approximate physical size

| table | columns |
|---|---|
| program | 19 |
| vehicle_configuration | 26 |
| component_db | 22 |
| tire_db | 65 |
| component_instance | 14 |
| component_resolution | 17 |
| vde | 107 |
| run | 21 |
| fuelcons | 83 |
| fuelcons_run_adoption | 8 |
| vde_component_resolution | 6 |

### Decisions to review

- **FuelCons↔RUN:** authoritative N:M lineage in `fuelcons_run_adoption`; normal FuelCons reads do not join RUN. The logical `adopted_run_ids_json` becomes an opt-in lineage-view projection, not duplicated stored state.
- **Component Instance:** nullable typed FKs `component_id` and `tire_id`, with checks forbidding both simultaneously and enforcing Tire/non-Tire domain direction. Both may be NULL for explicitly unresolved partial instances.
- **Component Resolution:** optional N:M adoption helper `vde_component_resolution`; VDE values stay copied into the immutable wide snapshot and never reconstruct at page-read time.
- **EEA:** keep 10.8M+ row-level monitoring records in separate analytical storage (versioned Parquet/DuckDB or later warehouse). Materialize only reviewed records that have a validated VDE link; unlinked monitoring evidence stays analytical.
- **Compatibility:** read-only legacy-named `vde_db` and `fuelcons_db` VIEWs expose exactly the current 101/79 columns. 12E retargets writes in persistence adapters; pages remain unchanged.
- **USER_DECISION_REQUIRED:** none.

## Identifier and lifecycle strategy

A mixed surrogate strategy preserves application identity: VDE, FuelCons and Tire retain stable INTEGER keys; Program, Configuration, Component, Component Instance, Component Resolution and RUN use application-generated TEXT canonical IDs (UUID/ULID-compatible, never source natural keys). Source IDs and file versions are separate columns/JSON provenance.

Program consolidation is non-destructive: `superseded_by_program_id` points provisional/old Programs to a reviewed target while child Configurations remain attached until an explicit migration remap. No Model Year uniqueness rule defines Program identity.

## Entity-by-entity shape

- `program`: semantic generation identity, confidence/status, source scope/version and non-destructive supersession.
- `vehicle_configuration`: stable architecture scalars plus sparse `architecture_properties_json`; no Target/Set ABC, ETW or procedure identity fields.
- `component_db` / `tire_db`: reusable masters; common filter/engineering scalars remain columns, sparse properties/artifacts remain JSON/references.
- `component_instance`: eBOM-lite occurrence. Unresolved references remain valid without fabricated component masters.
- `component_resolution`: optional method/boundary/result/provenance record; public ingestion may create zero rows.
- `vde`: 104-field logical contract plus physical provenance/version fields; wide persisted operational state with self-parent and Tire FKs.
- `run`: append-oriented evidence with frozen run type/evidence kind/fidelity/confidence checks.
- `fuelcons`: persisted comparison result linked directly to one VDE; multi-evidence adoption is externalized to the helper table.

## Master vs Persisted Snapshot Semantics

Duplicated engineering scalars in `vde` and `fuelcons` are intentional when they record the effective value used or exposed by that persisted analysis/result. They are not competing reusable master definitions and are not a normalization defect. The ownership map classifies every legacy-facing field as `AUTHORITATIVE_MASTER`, `RESOLVED_SNAPSHOT`, `COMPATIBILITY_SNAPSHOT`, or `RESULT_STATE` and records any reference master plus its write-time resolution source.

| Example | Reusable/master meaning | Persisted snapshot meaning |
|---|---|---|
| Engine power | `component_db.rated_power_kw` is the reusable component rating | `fuelcons.engine_max_power_kw` is the effective power adopted by that result |
| Transmission | configuration/component gear count and ratio describe reusable hardware | FuelCons gear count and final drive record the values actually used |
| Battery | `component_db.capacity_kwh` is the reusable nominal definition | FuelCons capacity/usable-energy fields preserve scenario-effective assumptions |
| Tire/roadload | Tire and configuration records provide reusable/reference inputs | VDE preserves the resolved tire, mass, aero and roadload state used by its calculation |

Materialization occurs at write or explicit update time: the persistence workflow resolves source, configuration, component and correction inputs; writes the effective scalar into the VDE/FuelCons snapshot; and retains lineage/provenance. Normal reads consume that snapshot directly and do not reconstruct historical values through new joins.

Historical snapshots are immutable with respect to master maintenance. Editing `component_db`, `vehicle_configuration` or `tire_db` must not silently update an existing VDE/FuelCons row. Adoption of a changed master value requires an explicit recalculation/rebuild workflow with new lineage or revision semantics. Therefore a 150 kW component rating and a 135 kW FuelCons effective-power snapshot can both be correct.

## Scalar / JSON / artifact boundary

Fields used by filters, joins, physics, comparisons, provenance routing or migration keys remain scalar. JSON is limited to source identity payloads, sparse type-specific architecture/component properties, conditions, assumptions and detailed lineage. Large maps, curves, PDFs and executable models remain artifact references; they are not stored as SQLite JSON blobs.

## Constraint and enum strategy

Frozen v1 CHECKs cover Program/Configuration identity status, component domains, RUN type/evidence kind, fidelity/confidence, VDE source semantics/legislation, electrification, energy basis, booleans, JSON validity and key numeric bounds. `comparison_basis`, `record_origin`, `record_status` and `review_status` are non-empty but intentionally extensible/legacy-compatible because current services already use values beyond the provisional 12C candidate lists.

The 12C logical `COMPONENT_INSTANCE.tire_id` TEXT/INTEGER mismatch is corrected to INTEGER so SQLite can enforce the FK to `tire_db.tire_id`.

## Index and query rationale

The package defines **28** indexes backed by audited repository/service queries. The plan covers Program/Configuration lookup, VDE hierarchy and recent browse, FuelCons/VDE joins and filters, RUN evidence, source versions, component usage and reverse lineage. Full rationale is in `index_plan.csv`.

## Compatibility and application boundary

The query inventory contains **14** active contract groups. Existing SELECT paths can continue using `vde_db`/`fuelcons_db` because those names become compatibility views in a new canonical database. `PRAGMA table_info(fuelcons_db)` also remains available. Existing writes cannot target views, so 12E must retarget the concentrated persistence helpers/services to `vde`/`fuelcons`; no existing page requires a schema-driven rewrite.

RUN lineage is available through `fuelcons_lineage_v1` only when explicitly requested. Comparison/Browse reads FuelCons directly and never traverses RUN.

## EEA physical storage

Do not load row-level EEA monitoring into the normal Streamlit SQLite file. Keep immutable/versioned source rows in analytical storage with source keys and provenance, aggregate there, and materialize canonical RUN/FuelCons only after a reviewed link to a valid VDE exists. Unlinked monitoring evidence remains analytical. This protects startup, backup, vacuum and browse behavior now and maps cleanly to a future partitioned PostgreSQL/warehouse service.

## Temporary validation

The schema instantiated successfully in SQLite `:memory:` with **9** primary domain tables, **2** physical helper tables, foreign keys enabled and `foreign_key_check` empty. Compatibility views expose **101** VDE and **79** FuelCons legacy columns.

No runtime DB was modified; SHA-256 remained byte-identical.

## Migration risks and 12E gates

- Retarget legacy write helpers before cutover; compatibility views are deliberately read-only.
- Backfill required Program/Configuration/VDE semantic fields and explicit provenance before enabling constraints.
- Correct scenario/ML origin labels and the Component Instance tire FK type during staged load.
- Validate all 180 legacy columns, NULL signatures, multiplicities and IDs before switching the DB path.
- Benchmark real population queries and keep/remove indexes using measured plans after load.
- Do not place EEA row-level data in the runtime file during migration.

## Evidence classification

- **DIRECTLY_TESTED:** `test_schema_creates_in_memory_with_nine_domain_entities`; `test_foreign_keys_are_enabled_and_valid`; `test_constraints_reject_invalid_rows`; `test_valid_minimal_rows_cover_all_nine_entities`; `test_required_cardinalities`; `test_fuelcons_supports_multiple_run_lineage`; `test_vde_can_exist_without_component_resolution`; `test_jrc_unresolved_identity_is_representable`; `test_null_and_zero_remain_distinct`; `test_compatibility_vde_columns_match_runtime`; `test_compatibility_fuelcons_columns_match_runtime`; `test_compatibility_projection_values`; `test_program_consolidation_preserves_children`; `test_component_instance_reference_mechanics`; `test_query_plan_uses_vde_configuration_index`; `test_outputs_are_scoped_to_etl`; `test_master_change_preserves_snapshot_and_explicit_rebuild_adopts_new_value`; `test_ownership_map_classifies_representative_snapshots`; `test_runtime_database_is_byte_identical`.
- **INDIRECTLY_COVERED:** Sprint 12C exact field/value compatibility and Sprint 12C.3 population shape.
- **INSPECTION_SUPPORTED:** active repository/service/page query inventory and scalar/JSON/artifact boundary.
- **GAP:** production-sized migrated query timings and real cutover behavior belong to 12E; in-memory tests are not production migration tests.

## Reproduction

```powershell
python etl/scripts/sprint_12d_physical_schema_design.py
python -m unittest discover -s etl/tests -p "test_sprint_12d*.py" -v
```

## Outputs

- `etl/schema/canonical_schema_v1.sql`
- `etl/schema/canonical_compatibility_v1.sql`
- `etl/reports/sprint_12d_physical_schema_design.md`
- `etl/reports/sprint_12d_migration_blueprint.md`
- `etl/data/processed/sprint_12d_physical_schema/constraint_matrix.csv`
- `etl/data/processed/sprint_12d_physical_schema/index_plan.csv`
- `etl/data/processed/sprint_12d_physical_schema/query_contract_inventory.csv`
- `etl/data/processed/sprint_12d_physical_schema/legacy_to_canonical_column_map.csv`
- `etl/data/processed/sprint_12d_physical_schema/physical_schema_design.json`
