"""
Canonical RoadLoad package for the current EcoDrive pipeline.

Preferred flow:
    RoadLoadRequest -> run_roadload_scenario() -> EquivalentABC

Legacy/reference modules such as ``services.py`` and ``physics.py`` remain in
the package for study and migration support, but they are not the primary
integration path for the current UI/services flow.
"""

from .models import (
    BaselineInput,
    ComponentChange,
    ComponentChanges,
    ComponentSet,
    EquivalentABC,
    OperatingModifiers,
    ResolutionOptions,
    ResolvedBaseline,
    RoadLoadComponent,
    RoadLoadModel,
    RoadLoadRequest,
)

from .engine import (
    apply_component_change,
    apply_component_changes,
    build_component_set_from_baseline,
    cdA_to_C,
    normalize_roadload_request,
    resolve_baseline,
    run_roadload_scenario,
    synthesize_equivalent_abc,
)

from .adapters import (
    build_request_from_baseline_dict,
    build_request_from_db_row,
    build_request_from_manual_inputs,
)

from .app_service import resolve_equiv_from_ctx
from .decomposition import component_delta_vs_baseline, decompose_equivalent_abc
from .tire_model import (
    G_MPS2,
    KPA_PER_PSI,
    MPH_PER_KPH,
    N_PER_KGF,
    N_PER_LBF,
    apply_tire_improvement,
    build_tire_component,
    calculate_axle_loads,
    calculate_axle_tire_abc_from_single,
    calculate_iso_tire_abc_for_single_tire,
    calculate_sae_smerf_rr_n_per_kn,
    calculate_sae_tire_abc_for_single_tire,
    calculate_single_tire_loads,
    calculate_vehicle_tire_abc,
    combine_front_rear_tire_abc,
)
