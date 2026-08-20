# VDE Setup v2.2 Scope

## Objective

VDE Setup v2.2 is a compact request builder for the existing canonical VDE Request workflow. Sprint v2.2A introduces the page, isolated state, four-section navigation, baseline correction workflow, proposal matrix, initial request inputs, and an auditable canonical draft.

## v2.1 vs v2.2

VDE Setup v2.1 remains the full workbook-style interface and keeps its existing renderer and save/preview/report behavior.

VDE Setup v2.2 is a separate frontend. It does not call the v2.1 monolith and does not execute technical resolution, persistence, report generation, Excel generation, CSV export, or component creation in Sprint v2.2A.

## Reused Modules

- `src/vde_core.vde_request_contract`
- `src.vde_core.repositories`
- `src.vde_core.db`
- Existing resolver, preview, save, report, and component repository modules remain the future integration targets, but are not called by the v2.2A page.

## State

The page uses one session-state root:

```python
st.session_state["vde_setup_v22"]
```

The pure state helpers live in `src/vde_core/vde_request_compact_state.py`. Initial state includes two proposals, `requested_1` and `requested_2`, and tracks baseline printed values, corrections, effective values, proposal matrix choices, preview status, and save placeholder status.

Proposal IDs are append-only within the draft session. Removing a proposal renumbers display indexes only; IDs are not reused.

## Four Sections

1. Baseline & Corrections: filter and load a VDE baseline, review printed snapshot, enter corrections, and see the effective baseline.
2. Proposal Matrix: configure proposal name, Walk From, domain proposal types, add/remove proposals, and preserve invalid broken dependencies for review.
3. Request Inputs: render one active domain. Sprint v2.2A implements Aero Absolute/Delta CdA and Transmission Absolute/Delta ABC inputs.
4. Preview & Save: show placeholders, pending inputs, and an audit table for the canonical draft.

Only the active section is rendered.

## Proposal Types

Proposal type labels are normalized through the canonical request contract. The compact page avoids using "Manual ABC" as a proposal type; manual entry is treated as an input source.

## Append-Only Rule

- Existing baseline rows are never modified by the v2.2 request.
- Existing component records are never modified by the v2.2 request.
- Baseline corrections affect only the effective snapshot used by the request.
- Future save work will create new rows for corrected baselines and optional new components.
- Editing existing components belongs to the component CRUD workflow.

## Performance Model

- Baseline summaries are cached.
- Mutable request state is not cached.
- Forms batch baseline corrections, matrix edits, and domain inputs.
- No resolver, report, Excel, CSV, plotting, or component repository lookup runs automatically.
- Optional section render timing is available with `?v22_profile=1`.

## Outside Sprint v2.2A

- Complete technical resolver integration.
- Save plan and database persistence.
- Component DB and VDE DB lookup beyond the visual contract.
- Report generation.
- Excel and CSV generation.
- Removal or replacement of v2.1.

## Future Plan

v2.2B:

- Complete domain input renderers.
- Add Component DB/VDE DB lookup on demand.
- Integrate the compact draft with technical resolver preview.

v2.2C:

- Add save plan.
- Implement append-only persistence.
- Save corrected baseline as a new VDE line.
- Optional new component creation.
- Add report generation.
- Add CSV after the manual flow is validated.
