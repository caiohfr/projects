# Sprint Record - RoadLoad Modularization

Date: 2026-05-29
Status: Done (pending automated test execution in local env)

## Goal
Modularize RoadLoad logic out of Streamlit page code and route VDE setup flows through a dedicated RoadLoad capability.

## Scope Delivered
- Created/used RoadLoad module at `src/vde_core/roadload/` with:
  - `models.py`
  - `engine.py`
  - `adapters.py`
  - `__init__.py`
- Integrated `pages/VDE_Setup.py` to use:
  - `build_request_from_manual_inputs(...)`
  - `run_roadload_scenario(...)`
  - `cdA_to_C(...)`
- Replaced page-side manual A/B/C mutation paths in preview/save with `EquivalentABC`.
- Added RoadLoad breakdown table display (`equiv.component_table`) in preview and compute/save flows.

## Main Functional Changes
1. New request-driven flow:
   - UI/context -> `RoadLoadRequest` -> `run_roadload_scenario` -> `EquivalentABC` -> VDE compute/save
2. Delta handling centralized:
   - `delta_rr_N`, `delta_brake_N`, `delta_parasitics_N`, `delta_aero_cdA`, `delta_mass_kg`
   - `delta_aero_cdA` converted via `cdA_to_C(...)` for DB traceability
3. Mass handling:
   - final mass computed inside RoadLoad engine and used by preview/save calculations
4. Edit flow fix:
   - default cycle now loaded as DataFrame (`load_cycle_csv(cycle_name)`) before recomputation

## Notable Technical Decisions
- Keep component changes delta-first (`delta_abc`, `delta_cda`, `improve`) with simple behavior.
- No Streamlit/DB dependency inside `roadload/engine.py`.
- Preserve current physical compatibility: `A [N]`, `B [N/kph]`, `C [N/kph^2]`.

## Files Touched (this sprint focus)
- `pages/VDE_Setup.py`
- `src/vde_core/roadload/__init__.py`
- `src/vde_core/roadload/models.py`
- `src/vde_core/roadload/engine.py`
- `src/vde_core/roadload/adapters.py`

## Current Known Limitations
- EPA phase path uses inertia class quantization; small `delta_mass_kg` may not change EPA result unless class boundary is crossed.
- Component separation is still synthetic (deltas currently applied over `roadload_total` in this sprint).
- Local automated tests were not executed because Python launcher/environment is misconfigured on this machine.

## Validation Performed
- Manual code-path verification of preview and save integration.
- Verified `delta_mass_kg` reaches RoadLoad engine and is used in live preview compute paths.
- Verified edit recompute path now loads cycle data correctly.

## Pre-Commit Checklist
- [ ] Run local automated smoke test for `run_roadload_scenario(...)` once Python env is fixed
- [ ] Review and stage only intended files (exclude DB/Notebook noise if not part of this sprint)
- [ ] Confirm no residual encoding issues in visible UI labels
- [ ] Commit with a scoped message (suggested below)

Suggested commit title:
`feat(vde): integrate roadload engine into VDE setup preview/save`

Suggested follow-up commit title:
`chore(roadload): add engine smoke test and mark legacy delta path`

