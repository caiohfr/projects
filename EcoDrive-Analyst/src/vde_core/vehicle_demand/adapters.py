# src/vde_core/vehicle_demand/adapters.py
# -----------------------------------------------------------------------------
# Canonical adapter: resolved vde_db row -> VehicleDemandRequest (Sprint 9C).
#
# This is deliberately the ONLY bridge between existing EcoDrive data and the
# Vehicle Demand contracts. It reads already-resolved/canonical values and
# normalizes their representation; it is NOT a resolver -- it never re-runs
# the Proposal Matrix, resolves Walk From, recomputes component ABC, or
# writes to the database. Same naming/shape convention as
# roadload/adapters.py's build_request_from_db_row.
#
# Reuses comparison_report_service.resolve_roadload_boundaries for TOTAL/NET
# ABC and cycles.use_standard_cycle for cycle resolution -- the exact same
# functions Comparison Report itself calls, so a request built here carries
# the same TOTAL/NET boundary and cycle trace Comparison would use for this
# row. This does create a one-directional dependency from this adapter onto
# comparison_report_service.py; see the module-level note in this package's
# Sprint 9C completion report for why that is safe today and what to watch
# for if Comparison is later wired to import from vehicle_demand (Sprint 9D).
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.vde_core.comparison_report_service import (
    RoadloadBoundary,
    build_vehicle_label,
    resolve_roadload_boundaries,
    resolve_transmission_boundary,
)
from src.vde_core.cycles import default_cycle_for_legislation, use_standard_cycle

from .contracts import AmbientState, Provenance, RoadloadCoefficients, VehicleDemandRequest

_COAST_COLUMNS = ("coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2")


def _roadload_coefficients(boundary: RoadloadBoundary) -> RoadloadCoefficients | None:
    if not boundary.available:
        return None
    return RoadloadCoefficients(A_N=boundary.A, B_N_per_kph=boundary.B, C_N_per_kph2=boundary.C)


def _resolve_test_mass_kg(vde_row: Mapping[str, Any]) -> float | None:
    """test_mass_kg if stored, else mass_kg -- the same fallback
    comparison_report_service's own on-demand VDE calculation applies
    (_resolve_mass_for_cycle). Reimplemented locally (two lines) rather than
    importing that leading-underscore helper across modules: this adapter is
    meant to be a stable, frozen surface and should not depend on another
    module's private name.
    """
    test_mass = vde_row.get("test_mass_kg")
    if test_mass is not None:
        return float(test_mass)
    mass_kg = vde_row.get("mass_kg")
    return float(mass_kg) if mass_kg is not None else None


def build_vehicle_demand_request(
    vde_row: Mapping[str, Any],
    *,
    temporary_transmission: Mapping[str, Any] | None = None,
    ambient: AmbientState | None = None,
) -> VehicleDemandRequest:
    """Build a VehicleDemandRequest from one resolved vde_db row.

    `vde_row` is whatever `src.vde_core.repositories.fetch_vde_by_id`
    already returns (or an equivalent mapping) -- this function performs no
    DB access itself (Sprint 9C Sec 8): callers fetch the row, this function
    only translates it.

    `temporary_transmission` passes straight through to
    resolve_roadload_boundaries, so a future caller (e.g. a Quick Scenario
    temporary-override resolver) can apply a session-only transmission
    assumption without this adapter needing to know about overrides itself
    (Sprint 9C Sec 36).

    `ambient` lets a caller supply real environmental conditions; vde_db has
    no ambient columns today; passing None decides nothing was supplied and
    defaults to an ambient.AmbientState() with air density UNAVAILABLE, not
    to a fabricated regulatory reference (Sprint 9B Sec 10, still frozen).

    Raises ValueError if the row has no TOTAL roadload -- VehicleDemandRequest
    requires it (Sprint 9A), so a row without stored coastdown ABC cannot be
    represented, not silently patched with a fabricated boundary.
    """
    total_values = [vde_row.get(col) for col in _COAST_COLUMNS]
    if any(v is None for v in total_values):
        raise ValueError(
            "vde_row is missing authoritative TOTAL roadload "
            "(coast_A_N/coast_B_N_per_kph/coast_C_N_per_kph2); "
            "cannot build a VehicleDemandRequest."
        )

    boundaries = resolve_roadload_boundaries(vde_row, temporary_transmission)
    roadload_total = _roadload_coefficients(boundaries["total"])
    roadload_net = _roadload_coefficients(boundaries["net"])
    if roadload_total is None:
        raise ValueError("resolve_roadload_boundaries reported TOTAL unavailable despite present coast_* columns.")

    transmission = resolve_transmission_boundary(vde_row, temporary_transmission)
    rrc = vde_row.get("rrc_N_per_kN")
    cda = vde_row.get("cda_m2")
    provenance = {
        "roadload_total": Provenance.SOURCE.value,
        "roadload_net": (Provenance.CALCULATED.value if roadload_net is not None else Provenance.UNAVAILABLE.value),
        "transmission": transmission.status.value,
        "rrc": Provenance.SOURCE.value if rrc is not None else Provenance.UNAVAILABLE.value,
        "cda": Provenance.SOURCE.value if cda is not None else Provenance.UNAVAILABLE.value,
    }

    legislation = vde_row.get("legislation")
    cycle_name = default_cycle_for_legislation(legislation)
    vde_id = vde_row.get("id")

    return VehicleDemandRequest(
        source_kind="VDE_ONLY",
        vde_id=int(vde_id) if vde_id is not None else None,
        fuelcons_id=None,
        label=build_vehicle_label(vde_row),
        cycle_name=cycle_name,
        cycle_source="STANDARD" if cycle_name else None,
        cycle_version=None,
        test_mass_kg=_resolve_test_mass_kg(vde_row),
        roadload_total=roadload_total,
        roadload_net=roadload_net,
        rrc_n_per_kn=float(rrc) if rrc is not None else None,
        cda_m2=float(cda) if cda is not None else None,
        ambient=ambient if ambient is not None else AmbientState(),
        provenance=provenance,
        model_version=None,
    )


def resolve_vehicle_demand_cycle(vde_row: Mapping[str, Any]) -> pd.DataFrame | None:
    """The same standard-cycle resolution Comparison's on-demand VDE calc
    uses (comparison_report_service.resolve_cycle_vde_results): keyed on
    legislation only, via cycles.use_standard_cycle, never on the row's
    free-text cycle_name column (Sprint 9C Sec 9 -- no heuristic cycle
    selection when a canonical resolver already exists). Returns None when
    the legislation is unrecognized or the standard cycle file is missing,
    matching use_standard_cycle's own existing behavior exactly.
    """
    return use_standard_cycle(vde_row.get("legislation"))


__all__ = ["build_vehicle_demand_request", "resolve_vehicle_demand_cycle"]
