"""
Input adapters for the EcoDrive RoadLoad Engine.

Adapters translate UI, database, or external records into RoadLoadRequest
objects. They intentionally avoid Streamlit, SQLite calls, and VDE cycle logic.
"""

from .models import (
    BaselineInput,
    OperatingModifiers,
    ComponentChange,
    ComponentChanges,
    ResolutionOptions,
    RoadLoadRequest,
)


def _get(row, key, default=None):
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _has_value(value):
    return value not in (None, "")


def _float_or_none(value):
    if not _has_value(value):
        return None
    return float(value)


def _float_or_zero(value):
    if not _has_value(value):
        return 0.0
    return float(value)


def _change_if_active(change):
    if change is None:
        return None
    return change


def build_request_from_manual_inputs(
    A,
    B,
    C,
    mass_kg,
    legislation=None,
    category=None,
    source="manual",
    delta_mass_kg=0.0,
    tire_improve_pct=0.0,
    tire_delta_A=0.0,
    tire_delta_B=0.0,
    tire_delta_C=0.0,
    delta_cda_m2=0.0,
    brake_delta_A=0.0,
    brake_delta_B=0.0,
    brake_delta_C=0.0,
    transmission_delta_A=0.0,
    transmission_delta_B=0.0,
    transmission_delta_C=0.0,
    parasitic_delta_A=0.0,
    parasitic_delta_B=0.0,
    parasitic_delta_C=0.0,
    target_legislation=None,
    trailer=False,
    allow_estimation=False,
    use_defaults=True,
    inherit_from_baseline=True,
    extra=None,
):
    """
    Build a RoadLoadRequest from manual/UI fields.

    The adapter only maps inputs into domain objects. The engine remains
    responsible for type normalization, validation, and physical conversion.
    """
    tire_change = None
    if any(_float_or_zero(v) != 0.0 for v in (tire_delta_A, tire_delta_B, tire_delta_C)):
        tire_change = ComponentChange(
            mode="delta_abc",
            A=tire_delta_A,
            B=tire_delta_B,
            C=tire_delta_C,
            meta={"adapter": "manual", "component": "tire"},
        )
    elif _float_or_zero(tire_improve_pct) != 0.0:
        tire_change = ComponentChange(
            mode="improve",
            improve_pct=tire_improve_pct,
            meta={"adapter": "manual", "component": "tire"},
        )

    aero_change = None
    if _float_or_zero(delta_cda_m2) != 0.0:
        aero_change = ComponentChange(
            mode="delta_cda",
            delta_cda_m2=delta_cda_m2,
            meta={"adapter": "manual", "component": "aero"},
        )

    brakes_change = None
    if any(_float_or_zero(v) != 0.0 for v in (brake_delta_A, brake_delta_B, brake_delta_C)):
        brakes_change = ComponentChange(
            mode="delta_abc",
            A=brake_delta_A,
            B=brake_delta_B,
            C=brake_delta_C,
            meta={"adapter": "manual", "component": "brakes"},
        )

    transmission_change = None
    if any(_float_or_zero(v) != 0.0 for v in (transmission_delta_A, transmission_delta_B, transmission_delta_C)):
        transmission_change = ComponentChange(
            mode="delta_abc",
            A=transmission_delta_A,
            B=transmission_delta_B,
            C=transmission_delta_C,
            meta={"adapter": "manual", "component": "transmission"},
        )

    parasitic_change = None
    if any(_float_or_zero(v) != 0.0 for v in (parasitic_delta_A, parasitic_delta_B, parasitic_delta_C)):
        parasitic_change = ComponentChange(
            mode="delta_abc",
            A=parasitic_delta_A,
            B=parasitic_delta_B,
            C=parasitic_delta_C,
            meta={"adapter": "manual", "component": "parasitic"},
        )

    return RoadLoadRequest(
        baseline=BaselineInput(
            A=A,
            B=B,
            C=C,
            mass_kg=mass_kg,
            legislation=legislation,
            category=category,
            source=source,
        ),
        operating=OperatingModifiers(
            delta_mass_kg=delta_mass_kg,
            trailer=trailer,
            target_legislation=target_legislation,
        ),
        components=ComponentChanges(
            tire=tire_change,
            aero=aero_change,
            transmission=transmission_change,
            brakes=brakes_change,
            parasitic=parasitic_change,
        ),
        options=ResolutionOptions(
            allow_estimation=allow_estimation,
            use_defaults=use_defaults,
            inherit_from_baseline=inherit_from_baseline,
        ),
        extra=extra or {},
    )


def build_request_from_db_row(
    row,
    source="database",
    delta_mass_kg=0.0,
    target_legislation=None,
    trailer=False,
    tire_improve_pct=0.0,
    tire_delta_A=0.0,
    tire_delta_B=0.0,
    tire_delta_C=0.0,
    delta_cda_m2=0.0,
    brake_delta_A=0.0,
    brake_delta_B=0.0,
    brake_delta_C=0.0,
    extra=None,
):
    """
    Build a RoadLoadRequest from a vde_db-like row.

    Expected row keys include id, coast_A_N, coast_B_N_per_kph,
    coast_C_N_per_kph2, mass_kg, legislation, and category. The function also
    accepts object-like rows with matching attributes.
    """
    mass_kg = _get(row, "mass_kg")
    if not _has_value(mass_kg):
        mass_kg = _get(row, "inertia_class")

    req = build_request_from_manual_inputs(
        A=_get(row, "coast_A_N"),
        B=_get(row, "coast_B_N_per_kph"),
        C=_get(row, "coast_C_N_per_kph2"),
        mass_kg=mass_kg,
        legislation=_get(row, "legislation"),
        category=_get(row, "category"),
        source=source,
        delta_mass_kg=delta_mass_kg,
        tire_improve_pct=tire_improve_pct,
        tire_delta_A=tire_delta_A,
        tire_delta_B=tire_delta_B,
        tire_delta_C=tire_delta_C,
        delta_cda_m2=delta_cda_m2,
        brake_delta_A=brake_delta_A,
        brake_delta_B=brake_delta_B,
        brake_delta_C=brake_delta_C,
        target_legislation=target_legislation,
        trailer=trailer,
        extra=extra or {},
    )
    req.baseline.baseline_id = _get(row, "id")
    req.extra.setdefault("baseline_row", row if isinstance(row, dict) else None)
    return req


def build_request_from_baseline_dict(baseline, **kwargs):
    """
    Compatibility helper for app dictionaries that use A/B/C names directly.
    """
    return build_request_from_manual_inputs(
        A=_get(baseline, "A"),
        B=_get(baseline, "B"),
        C=_get(baseline, "C"),
        mass_kg=_get(baseline, "mass_kg"),
        legislation=_get(baseline, "legislation"),
        category=_get(baseline, "category"),
        source=_get(baseline, "source", "baseline_dict"),
        **kwargs,
    )

