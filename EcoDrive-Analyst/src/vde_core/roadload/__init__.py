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
