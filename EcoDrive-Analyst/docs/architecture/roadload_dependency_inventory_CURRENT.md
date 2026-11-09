# Roadload dependency inventory — CURRENT

**Baseline:** `main@fca9a9ad189fe7e6930e987e29fee5717c0d532e`  
**Executed:** 2026-11-09  
**Evidence:** static AST inventory plus manual review of re-exports, dynamic access and notebooks.

## Result

The inventory found **168 records with zero parse errors** across `src/`, `tests/`, `etl/` and `scripts/`: 99 calls, 54 imports and 15 definitions. Machine-readable outputs are local execution artifacts under `artifacts/roadload_14a04/` and are intentionally not committed.

| Symbol | Records |
|---|---:|
| `calculate_vehicle_demand` | 38 |
| `resolve_roadload_boundaries` | 24 |
| `resolve_transmission_boundary` | 14 |
| `run_roadload_scenario` | 13 |
| `calculate_vehicle_tire_abc` | 12 |
| `_force_at` / `build_tire_component` | 10 each |
| `EquivalentABC` | 9 |
| `RoadLoadModel` / `compute_vde_series` | 8 each |
| `RoadLoadComponent` | 7 |
| `build_tire_component_from_result` | 5 |
| `ComponentSet` | 4 |
| `known_aero_force_N` / `known_rolling_force_N` | 3 each |

The most concentrated production consumers are `vehicle_demand/load_case.py`, `comparison_report_service.py`, `roadload/{__init__,engine,models}.py`, `tire_roadload_service.py`, `vehicle_demand/engine.py`, `quick_scenario/resolver.py` and `vehicle_demand/adapters.py`.

## Manual supplements and limitations

- `src/vde_core/roadload/__init__.py` is a public re-export boundary. Existing imports use both the package and direct `models`/`engine` modules.
- No dynamic import was found that resolves an inventoried Roadload/VDE symbol. Generic dynamic access elsewhere is not evidence of Roadload dispatch.
- `notebooks/roadload/RoadLoad_Notebook.ipynb` contains independent definitions of `ComponentSet`, `EquivalentABC` and `RoadLoadModel`. It is design/history evidence, not proven runtime code. Other notebooks did not match the selected symbols.
- AST output is not a runtime call graph. Indirect delegation, aliases outside the symbol set, generated code and external consumers remain outside proof. Dynamic tracing was not executed.

## Migration consequence

Do not replace `_force_at`, boundary resolution, canonical `compute_vde_series`, or package re-exports in one change. Preserve the public DTO and constructor surface, then migrate one vertical slice at a time behind parity tests.
