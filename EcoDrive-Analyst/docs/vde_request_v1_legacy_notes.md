# VDE Request v1 Legacy Notes

## New request flow dependencies

- `src/vde_core/vde_request_contract.py`
- `src/vde_core/vde_request_parser.py`
- `src/vde_core/vde_request_adapter.py`
- `src/vde_core/vde_request_resolver.py`
- `src/vde_core/vde_request_preview.py`
- `src/vde_core/vde_request_save.py`
- `src/vde_core/vde_request_report.py`
- `src/vde_core/component_repositories.py`
- `pages/VDE_Setup_v2_1.py`
- `src/vde_app/components/vde_setup.py` request-flow sections

## Cutover status

VDE Setup v2.1 now routes Preview & Save through the canonical VDE Request v1 stack only:

- `_v21_render_request_resolution_preview(...)`
- `_v21_render_request_review_save(...)`
- `resolve_vde_request(...)`
- `build_vde_request_save_plan(...)`
- `execute_vde_request_save_plan(...)`
- `build_vde_request_report_model(...)`
- `generate_vde_request_report_xlsx(...)`

The legacy Preview All / Save Plan / Save selected column / Save All widgets are not called from `_v21_render_preview_save(...)`.

## Still used by VDE Setup v2

- `_v2_preview(...)`
- `_v2_cached_preview(...)`
- `_v2_render_preview_save(...)`
- `_v2_render_technical_audit(...)`
- `save_vde_setup_result(...)`
- `render_vde_setup_workbook_v2(...)`

These remain for the older VDE Setup v2 route and must not be deleted until that page is retired or migrated.

## Shared helper

- `save_vde_setup_result(...)` remains part of the older v2 workbook flow.
- `render_vde_workbook_table(...)` is now shared by request preview/save reporting and other workbook tables.
- `build_vde_setup_preview(...)` remains the physical preview backbone reused by the request resolver.
- `_v21_request_column_labels(...)` and `_v21_display_column_label(...)` preserve internal `walked_N` ids while presenting Requested labels in v2.1.
- `_v21_clear_request_runtime_state(...)` centralizes request preview/save/report cache invalidation.

## Unused candidate

- `_v21_request_preview_rows(...)`
- `_v21_render_proposal_summary_table(...)`
- `_v21_preview_rows(...)`
- `_v21_save_plan_payload(...)`
- `_v21_save_plan_rows(...)`
- `_v21_prepare_preview_for_target(...)`
- `_v21_save_mode_from_plan_row(...)`
- `_v21_remember_saved_target(...)`

These are no longer on the active v2.1 route after cutover, but may still be useful for comparison, tests, or rollback until smoke tests are complete.

## Safe to delete only after smoke tests

- `render_vde_setup_spreadsheet_workbook(...)`
- `_resolve_scenario_workbook_state(...)`
- `_build_scenario_workbook_matrix_df(...)`
- `_render_scenario_workbook_matrix(...)`
- duplicated legacy preview/save widgets that are no longer referenced by v2.1 request users
- compatibility branches that only exist to bridge old matrix state into the request resolver

## Future schema recommendation

The current v1 flow intentionally does not add dedicated `vde_db` columns for newly created mock component IDs.

If persistent multi-component provenance becomes necessary, prefer a separate linking structure such as:

- `scenario_component_links`

with one row per saved scenario row, domain, component id, source, and provenance snapshot.
