"""
Pure RoadLoad Engine for EcoDrive Analyzer.

This module converts a RoadLoadRequest into an EquivalentABC without touching
Streamlit, SQLite, pages, or VDE cycle calculations.
"""

from .models import (
    BaselineInput,
    OperatingModifiers,
    ComponentChange,
    ComponentChanges,
    ResolutionOptions,
    RoadLoadRequest,
    ResolvedBaseline,
    RoadLoadComponent,
    ComponentSet,
    EquivalentABC,
)


_REQUIRED_BASELINE_FIELDS = ("A", "B", "C", "mass_kg")


def _to_float(value, field_name, *, default=None, required=False):
    if value in (None, ""):
        if required:
            raise ValueError(f"Missing required roadload field: {field_name}")
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}")


def _clean_str(value, *, upper=False, default=None):
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.upper() if upper else text


def _record_get(record, key, default=None):
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _first_record_value(record, keys):
    for key in keys:
        value = _record_get(record, key)
        if value not in (None, ""):
            return value, key
    return None, None


def normalize_roadload_request(req):
    """Normalize a RoadLoadRequest in place and return it."""
    if not isinstance(req, RoadLoadRequest):
        raise TypeError("req must be a RoadLoadRequest")

    if req.baseline is None:
        req.baseline = BaselineInput()
    if req.operating is None:
        req.operating = OperatingModifiers()
    if req.components is None:
        req.components = ComponentChanges()
    if req.options is None:
        req.options = ResolutionOptions()
    if req.extra is None:
        req.extra = {}

    req.baseline.A = _to_float(req.baseline.A, "baseline.A", default=None)
    req.baseline.B = _to_float(req.baseline.B, "baseline.B", default=None)
    req.baseline.C = _to_float(req.baseline.C, "baseline.C", default=None)
    req.baseline.mass_kg = _to_float(req.baseline.mass_kg, "baseline.mass_kg", default=None)
    req.baseline.legislation = _clean_str(req.baseline.legislation, upper=True)
    req.baseline.category = _clean_str(req.baseline.category, upper=True)
    req.baseline.source = _clean_str(req.baseline.source, default="unknown")

    req.operating.delta_mass_kg = _to_float(
        req.operating.delta_mass_kg,
        "operating.delta_mass_kg",
        default=0.0,
    )
    req.operating.trailer = bool(req.operating.trailer)
    req.operating.target_legislation = _clean_str(
        req.operating.target_legislation,
        upper=True,
    )

    for component_name, change in req.components.items():
        if change is None:
            continue
        if not isinstance(change, ComponentChange):
            raise TypeError(f"components.{component_name} must be a ComponentChange")
        change.mode = _clean_str(change.mode, default="delta_abc").lower()
        if change.mode not in ComponentChange.VALID_MODES:
            allowed = ", ".join(sorted(ComponentChange.VALID_MODES))
            raise ValueError(f"Unsupported change mode for {component_name}: {change.mode!r}. Expected one of: {allowed}")
        change.A = _to_float(change.A, f"components.{component_name}.A", default=0.0)
        change.B = _to_float(change.B, f"components.{component_name}.B", default=0.0)
        change.C = _to_float(change.C, f"components.{component_name}.C", default=0.0)
        change.delta_cda_m2 = _to_float(
            change.delta_cda_m2,
            f"components.{component_name}.delta_cda_m2",
            default=0.0,
        )
        change.improve_pct = _to_float(
            change.improve_pct,
            f"components.{component_name}.improve_pct",
            default=0.0,
        )
        if change.meta is None:
            change.meta = {}

    return req


def resolve_baseline(req, baseline_record=None):
    """Resolve required A/B/C/mass fields from request and optional record."""
    if not isinstance(req, RoadLoadRequest):
        raise TypeError("req must be a RoadLoadRequest")

    source_map = {}
    warnings = []
    baseline = req.baseline

    field_keys = {
        "A": ("A", "coast_A_N"),
        "B": ("B", "coast_B_N_per_kph"),
        "C": ("C", "coast_C_N_per_kph2"),
        "mass_kg": ("mass_kg", "inertia_class"),
    }

    resolved = {}
    for field in _REQUIRED_BASELINE_FIELDS:
        value = getattr(baseline, field, None)
        if value in (None, "") and baseline_record is not None and req.options.inherit_from_baseline:
            value, source_key = _first_record_value(baseline_record, field_keys[field])
            if source_key:
                source_map[field] = f"baseline_record.{source_key}"
        else:
            source_map[field] = "request.baseline"

        resolved[field] = _to_float(value, f"baseline.{field}", required=True)

    if resolved["mass_kg"] <= 0:
        raise ValueError("baseline.mass_kg must be greater than zero")

    legislation = baseline.legislation
    category = baseline.category
    if baseline_record is not None:
        legislation = legislation or _clean_str(_record_get(baseline_record, "legislation"), upper=True)
        category = category or _clean_str(_record_get(baseline_record, "category"), upper=True)

    if not legislation:
        warnings.append("Baseline legislation is not defined.")
    if not category:
        warnings.append("Baseline category is not defined.")

    return ResolvedBaseline(
        A=resolved["A"],
        B=resolved["B"],
        C=resolved["C"],
        mass_kg=resolved["mass_kg"],
        legislation=legislation,
        category=category,
        source_map=source_map,
        warnings=warnings,
    )


def build_component_set_from_baseline(baseline):
    """Create the initial component set from a resolved baseline."""
    component_set = ComponentSet()
    component_set.add(
        RoadLoadComponent(
            name="roadload_total",
            A=baseline.A,
            B=baseline.B,
            C=baseline.C,
            source="baseline",
            meta={"baseline": baseline.to_dict()},
        )
    )
    return component_set


def cdA_to_C(delta_cda_m2, rho=1.2):
    """Convert delta CdA [m2] into delta C [N/kph2]."""
    delta_cda_m2 = _to_float(delta_cda_m2, "delta_cda_m2", required=True)
    rho = _to_float(rho, "rho", required=True)
    return 0.5 * rho * delta_cda_m2 / (3.6 ** 2)


def apply_component_change(component, change):
    """Apply one ComponentChange to one RoadLoadComponent."""
    if not isinstance(component, RoadLoadComponent):
        raise TypeError("component must be a RoadLoadComponent")
    if change is None:
        return component
    if not isinstance(change, ComponentChange):
        raise TypeError("change must be a ComponentChange")

    mode = _clean_str(change.mode, default="delta_abc").lower()
    out = component.copy()
    out.source = f"component_{mode}"

    if mode == "delta_abc":
        out.A += _to_float(change.A, "change.A", default=0.0)
        out.B += _to_float(change.B, "change.B", default=0.0)
        out.C += _to_float(change.C, "change.C", default=0.0)
    elif mode == "delta_cda":
        delta_C = cdA_to_C(change.delta_cda_m2)
        out.C += delta_C
        out.meta["delta_C_from_cda"] = delta_C
    elif mode == "improve":
        factor = 1.0 - (_to_float(change.improve_pct, "change.improve_pct", default=0.0) / 100.0)
        out.A *= factor
        out.B *= factor
        out.C *= factor
        out.meta["improve_factor"] = factor
    else:
        allowed = ", ".join(sorted(ComponentChange.VALID_MODES))
        raise ValueError(f"Unsupported component change mode: {mode!r}. Expected one of: {allowed}")

    out.meta.setdefault("applied_changes", [])
    out.meta["applied_changes"].append(change.to_dict())
    return out


def apply_component_changes(component_set, changes):
    """Apply active delta-style changes. For now, all changes mutate roadload_total."""
    if not isinstance(component_set, ComponentSet):
        raise TypeError("component_set must be a ComponentSet")
    if changes is None:
        return component_set
    if not isinstance(changes, ComponentChanges):
        raise TypeError("changes must be a ComponentChanges")

    total = component_set.get("roadload_total")
    if total is None:
        raise ValueError("ComponentSet must contain 'roadload_total'")

    for _name, change in changes.active_items():
        total = apply_component_change(total, change)

    component_set.add(total)
    return component_set


def synthesize_equivalent_abc(baseline, component_set, req):
    """Synthesize final EquivalentABC from components and operating modifiers."""
    if not isinstance(baseline, ResolvedBaseline):
        raise TypeError("baseline must be a ResolvedBaseline")
    if not isinstance(component_set, ComponentSet):
        raise TypeError("component_set must be a ComponentSet")
    if not isinstance(req, RoadLoadRequest):
        raise TypeError("req must be a RoadLoadRequest")

    total = component_set.total_abc()
    mass_final = baseline.mass_kg + req.operating.delta_mass_kg
    if mass_final <= 0:
        raise ValueError("Final mass must be greater than zero after operating.delta_mass_kg")

    warnings = list(baseline.warnings)
    if req.operating.trailer:
        warnings.append("Trailer flag is set, but trailer physics are not implemented yet.")

    return EquivalentABC(
        A=total["A"],
        B=total["B"],
        C=total["C"],
        mass_kg=mass_final,
        component_table=component_set.as_table(),
        warnings=warnings,
        meta={
            "legislation": baseline.legislation,
            "category": baseline.category,
            "baseline_source_map": baseline.source_map,
        },
    )


def run_roadload_scenario(req, baseline_record=None):
    """Run the full RoadLoad Engine pipeline and return EquivalentABC."""
    req = normalize_roadload_request(req)
    baseline = resolve_baseline(req=req, baseline_record=baseline_record)
    component_set = build_component_set_from_baseline(baseline)
    component_set = apply_component_changes(component_set=component_set, changes=req.components)
    return synthesize_equivalent_abc(baseline=baseline, component_set=component_set, req=req)


