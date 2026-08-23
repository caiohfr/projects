# src/vde_core/comparison_report_service.py
# -----------------------------------------------------------------------------
# Package 8A - canonical, Streamlit-free Comparison data layer.
#
# This module was previously three small, unused report-loading helpers (kept
# below and reused where possible). It now also exposes the canonical
# ComparisonItem/ComparisonDataset contract that later Scorecard/Dashboard/
# FE x VDE/Roadload/Explore packages will consume. It does not render
# anything and does not write to the database.
#
# TOTAL/NET reading always goes through vde_net_total_contract.canonical_vde_read
# (Package 7G). Cycle/phase energy is always calculated by delegating to the
# existing physics in vde_setup_service.compute_vde_preview_from_inputs -- this
# module never re-derives VDE physics, EPA weighting, or WLTP phase logic.
# -----------------------------------------------------------------------------

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.vde_core import db as db_module
from src.vde_core.comparison_metric_registry import ComparisonRule, MetricDirection, get_metric
from src.vde_core.component_repositories import get_component
from src.vde_core.cycles import use_standard_cycle
from src.vde_core.database_management_contract import EntityType
from src.vde_core.database_management_service import get_record
from src.vde_core.pwt_fuel_energy_service import (
    compare_saved_scenario_revision,
    resolve_vde_source_revision,
)
from src.vde_core.repositories import fetch_vde_by_id, fetch_vde_by_ids
from src.vde_core.vde_net_total_contract import canonical_vde_read
from src.vde_core.vde_setup_service import compute_vde_preview_from_inputs


# -----------------------------------------------------------------------------
# Pre-existing report-loading helpers (unused before Package 8A). Kept as-is;
# build_vehicle_label is reused below for ComparisonItem labels.
# -----------------------------------------------------------------------------

_VDE_REPORT_SQL = "SELECT * FROM vde_db ORDER BY COALESCE(updated_at, created_at) DESC;"


def load_vde_report_frame(db_path: str | Path) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"DB not found: {path}")

    with sqlite3.connect(str(path), timeout=30, isolation_level=None) as con:
        frame = pd.read_sql_query(_VDE_REPORT_SQL, con)

    if frame.empty:
        return frame

    frame = ensure_report_abc_aliases(frame)
    frame["veh_label"] = frame.apply(build_vehicle_label, axis=1)
    return frame


def ensure_report_abc_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "A" not in out.columns and "coast_A_N" in out.columns:
        out["A"] = out["coast_A_N"]
    if "B" not in out.columns and "coast_B_N_per_kph" in out.columns:
        out["B"] = out["coast_B_N_per_kph"]
    if "C" not in out.columns and "coast_C_N_per_kph2" in out.columns:
        out["C"] = out["coast_C_N_per_kph2"]
    return out


def build_vehicle_label(row) -> str:
    parts = [str((row.get("make") or "")).strip(), str((row.get("model") or "")).strip()]
    year = row.get("year", None)
    if year not in (None, "") and not (isinstance(year, float) and pd.isna(year)):
        parts.append(str(int(year)))
    return " ".join(part for part in parts if part)


# -----------------------------------------------------------------------------
# Canonical contract
# -----------------------------------------------------------------------------


class _TextEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ComparisonRole(_TextEnum):
    REFERENCE = "REFERENCE"
    COMPARISON = "COMPARISON"


class SourceKind(_TextEnum):
    FUELCONS_SCENARIO = "FUELCONS_SCENARIO"
    VDE_ONLY = "VDE_ONLY"


class TransmissionStatus(_TextEnum):
    AVAILABLE = "AVAILABLE"
    TEMPORARY = "TEMPORARY"
    MISSING = "MISSING"


class LineageStatus(_TextEnum):
    EXPLICIT = "EXPLICIT"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class RevisionStatus(_TextEnum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"


_REVISION_STATUS_MAP = {
    "current": RevisionStatus.CURRENT,
    "changed": RevisionStatus.STALE,
    "missing": RevisionStatus.MISSING,
    "unknown": RevisionStatus.UNKNOWN,
}


@dataclass(frozen=True)
class RoadloadBoundary:
    A: float | None
    B: float | None
    C: float | None
    available: bool


_UNAVAILABLE_BOUNDARY = RoadloadBoundary(A=None, B=None, C=None, available=False)


@dataclass(frozen=True)
class TransmissionResolution:
    status: TransmissionStatus
    source: str | None
    abc: RoadloadBoundary | None
    provenance: str | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VDEBoundaryResult:
    boundary: str  # "TOTAL" | "NET"
    aggregate: float | None
    by_phase: Mapping[str, float]
    cycle: Mapping[str, Any]
    source: str  # "on_demand_calculation" | "unavailable"
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ComparisonProvenance:
    record_origin: str | None
    scenario_intent: str | None
    method: str | None
    model_version: str | None
    source_vde_revision: str | None
    current_vde_revision: str | None
    revision_status: RevisionStatus | None
    confidence: str | None
    source_label: str


@dataclass(frozen=True)
class ComparisonItem:
    role: ComparisonRole
    source_kind: SourceKind
    vde_id: int | None
    fuelcons_id: int | None
    label: str
    vehicle: Mapping[str, Any]
    powertrain: Mapping[str, Any]
    roadload: Mapping[str, Any]
    vde: Mapping[str, Any]
    fuel_energy: Mapping[str, Any] | None
    provenance: ComparisonProvenance
    lineage: Mapping[str, Any]
    availability: frozenset[str]
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ComparisonDataset:
    reference: ComparisonItem
    comparisons: tuple[ComparisonItem, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


# -----------------------------------------------------------------------------
# Transmission / roadload boundary resolution (Sec 10-13)
# -----------------------------------------------------------------------------

_TRANS_COLUMNS = ("trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2")
_COAST_COLUMNS = ("coast_A_N", "coast_B_N_per_kph", "coast_C_N_per_kph2")


def resolve_temporary_transmission_from_component(component_id: str) -> dict[str, Any] | None:
    """Look up a transmission component's canonical ABC. Never writes to the DB."""
    component = get_component("transmission", component_id)
    if not component:
        return None
    A = component.get("equivalent_A_N")
    B = component.get("equivalent_B_N_per_kph")
    C = component.get("equivalent_C_N_per_kph2")
    if A is None or B is None or C is None:
        return None
    return {"source": f"COMPONENT:{component_id}", "A": float(A), "B": float(B), "C": float(C)}


def resolve_transmission_boundary(
    vde_row: Mapping[str, Any], temporary_transmission: Mapping[str, Any] | None = None
) -> TransmissionResolution:
    """Resolve the transmission ABC boundary for one VDE row.

    A transmission boundary is AVAILABLE only when all three stored
    coefficients are non-NULL (missing is not equivalent to zero, Package 8A
    Sec 12). A temporary assumption is only used when the stored boundary is
    missing, and is always tagged TEMPORARY, never silently treated as
    resolved/stored.
    """
    values = [vde_row.get(col) for col in _TRANS_COLUMNS]
    if all(v is not None for v in values):
        A, B, C = (float(v) for v in values)
        return TransmissionResolution(
            status=TransmissionStatus.AVAILABLE,
            source="STORED",
            abc=RoadloadBoundary(A=A, B=B, C=C, available=True),
            provenance="vde_db.trans_A_coef_N/trans_B_coef_Npkph/trans_C_coef_Npkph2",
            warnings=(),
        )

    if temporary_transmission:
        A = temporary_transmission.get("A")
        B = temporary_transmission.get("B")
        C = temporary_transmission.get("C")
        if A is not None and B is not None and C is not None:
            return TransmissionResolution(
                status=TransmissionStatus.TEMPORARY,
                source=str(temporary_transmission.get("source") or "TEMPORARY"),
                abc=RoadloadBoundary(A=float(A), B=float(B), C=float(C), available=True),
                provenance="temporary_transmission_assumption",
                warnings=("temporary_transmission_used",),
            )

    return TransmissionResolution(
        status=TransmissionStatus.MISSING,
        source=None,
        abc=None,
        provenance=None,
        warnings=("transmission_missing",),
    )


def resolve_roadload_boundaries(
    vde_row: Mapping[str, Any], temporary_transmission: Mapping[str, Any] | None = None
) -> dict[str, RoadloadBoundary]:
    """TOTAL is always the canonical coastdown ABC (Package 7G/Sec 10). NET is
    TOTAL minus the resolved transmission ABC, only when that boundary is
    AVAILABLE or TEMPORARY (Sec 11). TOTAL is never derived from NET.
    """
    total_values = [vde_row.get(col) for col in _COAST_COLUMNS]
    if all(v is not None for v in total_values):
        total = RoadloadBoundary(*(float(v) for v in total_values), available=True)
    else:
        total = _UNAVAILABLE_BOUNDARY

    transmission = resolve_transmission_boundary(vde_row, temporary_transmission)
    if total.available and transmission.abc is not None:
        net = RoadloadBoundary(
            A=total.A - transmission.abc.A,
            B=total.B - transmission.abc.B,
            C=total.C - transmission.abc.C,
            available=True,
        )
    else:
        net = _UNAVAILABLE_BOUNDARY

    return {"total": total, "net": net}


# -----------------------------------------------------------------------------
# On-demand cycle/phase VDE calculation (Sec 16-21)
# -----------------------------------------------------------------------------


def compute_vde_for_boundary(
    df_cycle: pd.DataFrame,
    legislation: str,
    *,
    A: float,
    B: float,
    C: float,
    mass_kg: float,
    epa_mass_is_resolved_twc: bool = False,
) -> dict[str, Any]:
    """Semantic wrapper: compute VDE energy for WHICHEVER ABC boundary is
    passed in (TOTAL or NET). Delegates entirely to the existing
    compute_vde_preview_from_inputs -- physics stays numerically identical
    (Package 8A Sec 18); this only avoids new code reading historically-named
    functions like compute_vde_net() directly.
    """
    return compute_vde_preview_from_inputs(
        df_cycle,
        legislation,
        A=A,
        B=B,
        C=C,
        mass_kg=mass_kg,
        epa_mass_is_resolved_twc=epa_mass_is_resolved_twc,
    )


def _resolve_mass_for_cycle(vde_row: Mapping[str, Any]) -> tuple[float | None, bool]:
    test_mass = vde_row.get("test_mass_kg")
    if test_mass is not None:
        return float(test_mass), True
    mass_kg = vde_row.get("mass_kg")
    if mass_kg is not None:
        return float(mass_kg), False
    return None, False


def _boundary_result(
    boundary: str,
    df_cycle: pd.DataFrame | None,
    legislation: str,
    abc: RoadloadBoundary,
    mass_kg: float | None,
    epa_mass_is_resolved_twc: bool,
    extra_warnings: tuple[str, ...],
) -> VDEBoundaryResult:
    if df_cycle is None:
        return VDEBoundaryResult(
            boundary=boundary,
            aggregate=None,
            by_phase={},
            cycle={"legislation": legislation, "status": "unavailable"},
            source="unavailable",
            warnings=("cycle_trace_unavailable",) + extra_warnings,
        )
    if not abc.available or mass_kg is None:
        reason = "roadload_total_missing" if boundary == "TOTAL" else "transmission_missing"
        return VDEBoundaryResult(
            boundary=boundary,
            aggregate=None,
            by_phase={},
            cycle={"legislation": legislation, "status": "unavailable"},
            source="unavailable",
            warnings=(reason,) + extra_warnings,
        )

    result = compute_vde_for_boundary(
        df_cycle,
        legislation,
        A=abc.A,
        B=abc.B,
        C=abc.C,
        mass_kg=mass_kg,
        epa_mass_is_resolved_twc=epa_mass_is_resolved_twc,
    )
    if not result.get("ok"):
        return VDEBoundaryResult(
            boundary=boundary,
            aggregate=None,
            by_phase={},
            cycle={"legislation": legislation, "status": "unavailable"},
            source="unavailable",
            warnings=("cycle_trace_unavailable",) + extra_warnings,
        )
    return VDEBoundaryResult(
        boundary=boundary,
        aggregate=result.get("total_mj_km"),
        by_phase=dict(result.get("by_phase") or {}),
        cycle={"legislation": legislation, "status": "available"},
        source="on_demand_calculation",
        warnings=extra_warnings,
    )


def resolve_cycle_vde_results(
    vde_row: Mapping[str, Any], temporary_transmission: Mapping[str, Any] | None = None
) -> dict[str, VDEBoundaryResult]:
    """Calculate VDE TOTAL and NET on demand from resolved ABC + mass + the
    standard cycle trace for this row's legislation. Never trusts historical
    phase columns (vde_low_mj_per_km etc, Sec 21) as canonical TOTAL/NET.
    """
    legislation = str(vde_row.get("legislation") or "")
    df_cycle = use_standard_cycle(legislation)
    mass_kg, epa_resolved_twc = _resolve_mass_for_cycle(vde_row)
    boundaries = resolve_roadload_boundaries(vde_row, temporary_transmission)
    transmission = resolve_transmission_boundary(vde_row, temporary_transmission)

    total_result = _boundary_result(
        "TOTAL", df_cycle, legislation, boundaries["total"], mass_kg, epa_resolved_twc, ()
    )
    net_extra_warnings = transmission.warnings if transmission.status != TransmissionStatus.AVAILABLE else ()
    net_result = _boundary_result(
        "NET", df_cycle, legislation, boundaries["net"], mass_kg, epa_resolved_twc, net_extra_warnings
    )
    return {"total": total_result, "net": net_result}


def resolve_vde_aggregate(
    vde_row: Mapping[str, Any], *, cycle_results: Mapping[str, VDEBoundaryResult] | None = None
) -> dict[str, Any]:
    """Persisted-aggregate TOTAL/NET via the Package 7G canonical contract.

    TOTAL is never derived from NET, and NET is never silently replaced by an
    on-demand value -- when the persisted NET is unavailable but the caller
    supplies already-computed cycle_results with a resolvable NET boundary,
    that value is used and explicitly tagged net_source="on_demand" so
    callers can distinguish it from a stored aggregate (Sec 15).
    """
    canonical = canonical_vde_read(vde_row)
    total = canonical.total_mj_per_km
    net = canonical.net_mj_per_km
    net_source = "stored" if canonical.net_available else None

    if net is None and cycle_results is not None:
        on_demand_net = cycle_results.get("net")
        if on_demand_net is not None and on_demand_net.aggregate is not None:
            net = on_demand_net.aggregate
            net_source = "on_demand"

    return {"total": total, "net": net, "net_source": net_source}


# -----------------------------------------------------------------------------
# Lineage and revision (Sec 8-9, 30)
# -----------------------------------------------------------------------------


def _resolve_lineage(vde_row: Mapping[str, Any]) -> dict[str, Any]:
    parent_id = vde_row.get("vde_id_parent")
    if parent_id is not None:
        return {
            "parent_source_kind": SourceKind.VDE_ONLY.value,
            "parent_id": int(parent_id),
            "lineage_status": LineageStatus.EXPLICIT.value,
        }
    return {
        "parent_source_kind": None,
        "parent_id": None,
        "lineage_status": LineageStatus.NONE.value,
    }


def _resolve_revision(fuelcons_row: Mapping[str, Any], current_vde_row: Mapping[str, Any]) -> dict[str, Any]:
    saved_revision = fuelcons_row.get("source_vde_revision")
    comparison = compare_saved_scenario_revision(saved_revision, dict(current_vde_row))
    status = _REVISION_STATUS_MAP.get(comparison.get("status"), RevisionStatus.UNKNOWN)
    return {
        "source_vde_revision": comparison.get("saved_revision"),
        "current_vde_revision": comparison.get("current_revision"),
        "revision_status": status,
    }


# -----------------------------------------------------------------------------
# Item / dataset builders (Sec 4-7, 40)
# -----------------------------------------------------------------------------

_VEHICLE_FIELDS = (
    "make", "model", "year", "category", "legislation", "cycle_name", "drive_type",
    "mass_kg", "test_mass_kg", "cda_m2", "rrc_N_per_kN",
)
_POWERTRAIN_VDE_FIELDS = ("engine_type", "engine_model", "transmission_type", "transmission_model")
_POWERTRAIN_FUELCONS_FIELDS = (
    "electrification",
    "fuel_type",
    "gear_count",
    "final_drive_ratio",
    "engine_max_power_kw",
    "battery_capacity_kwh",
)
_FUEL_ENERGY_FIELDS = ("fuel_l_per_100km", "fuel_km_per_l", "energy_Wh_per_km", "gco2_per_km", "eta_pt_est")


def _availability_for(roadload: dict, vde_aggregate: dict, cycle_results: dict, fuel_energy: dict | None) -> frozenset[str]:
    keys: set[str] = set()
    if roadload["total"].available:
        keys.add("ROADLOAD_TOTAL")
    if roadload["net"].available:
        keys.add("ROADLOAD_NET")
    if vde_aggregate["total"] is not None:
        keys.add("VDE_TOTAL")
    if vde_aggregate["net"] is not None:
        keys.add("VDE_NET")
    if cycle_results["total"].aggregate is not None:
        keys.add("CYCLE_RESULTS_TOTAL")
    if cycle_results["net"].aggregate is not None:
        keys.add("CYCLE_RESULTS_NET")
    if fuel_energy:
        if fuel_energy.get("fuel_l_per_100km") is not None or fuel_energy.get("fuel_km_per_l") is not None:
            keys.add("FUEL_CONSUMPTION")
        if fuel_energy.get("energy_Wh_per_km") is not None:
            keys.add("FUEL_ENERGY")
            if fuel_energy.get("electrification") == "BEV":
                keys.add("ELECTRICAL_ENERGY")
        if fuel_energy.get("gco2_per_km") is not None:
            keys.add("CO2")
    return frozenset(keys)


def build_vde_comparison_item(
    vde_id: int,
    *,
    role: ComparisonRole = ComparisonRole.COMPARISON,
    vde_row: Mapping[str, Any] | None = None,
    temporary_transmission: Mapping[str, Any] | None = None,
) -> ComparisonItem:
    row = dict(vde_row) if vde_row is not None else fetch_vde_by_id(vde_id)
    if not row:
        raise ValueError(f"VDE row not found: id={vde_id}")

    roadload = resolve_roadload_boundaries(row, temporary_transmission)
    cycle_results = resolve_cycle_vde_results(row, temporary_transmission)
    vde_aggregate = resolve_vde_aggregate(row, cycle_results=cycle_results)

    vehicle = {k: row.get(k) for k in _VEHICLE_FIELDS}
    powertrain = {k: row.get(k) for k in _POWERTRAIN_VDE_FIELDS}
    fuel_energy = None

    provenance = ComparisonProvenance(
        record_origin=row.get("record_origin"),
        scenario_intent=None,
        method=None,
        model_version=None,
        source_vde_revision=None,
        current_vde_revision=resolve_vde_source_revision(row),
        revision_status=None,
        confidence=None,
        source_label=str(row.get("record_origin") or "UNKNOWN"),
    )

    warnings: tuple[str, ...] = ()
    if not roadload["total"].available:
        warnings += ("roadload_total_missing",)
    warnings += resolve_transmission_boundary(row, temporary_transmission).warnings

    return ComparisonItem(
        role=role,
        source_kind=SourceKind.VDE_ONLY,
        vde_id=int(vde_id),
        fuelcons_id=None,
        label=build_vehicle_label(row),
        vehicle=vehicle,
        powertrain=powertrain,
        roadload=roadload,
        vde={"aggregate": vde_aggregate, "cycle_results": cycle_results},
        fuel_energy=fuel_energy,
        provenance=provenance,
        lineage=_resolve_lineage(row),
        availability=_availability_for(roadload, vde_aggregate, cycle_results, fuel_energy),
        warnings=warnings,
    )


def build_scenario_comparison_item(
    fuelcons_id: int,
    *,
    role: ComparisonRole = ComparisonRole.COMPARISON,
    fuelcons_row: Mapping[str, Any] | None = None,
    vde_row: Mapping[str, Any] | None = None,
    temporary_transmission: Mapping[str, Any] | None = None,
) -> ComparisonItem:
    fuel_row = dict(fuelcons_row) if fuelcons_row is not None else get_record(EntityType.FUEL_CONSUMPTION, fuelcons_id)
    if not fuel_row:
        raise ValueError(f"FuelCons row not found: id={fuelcons_id}")

    linked_vde_id = fuel_row.get("vde_id")
    row = dict(vde_row) if vde_row is not None else (fetch_vde_by_id(linked_vde_id) if linked_vde_id else None)
    if not row:
        raise ValueError(f"Linked VDE row not found for fuelcons_id={fuelcons_id} vde_id={linked_vde_id}")

    roadload = resolve_roadload_boundaries(row, temporary_transmission)
    cycle_results = resolve_cycle_vde_results(row, temporary_transmission)
    vde_aggregate = resolve_vde_aggregate(row, cycle_results=cycle_results)

    vehicle = {k: row.get(k) for k in _VEHICLE_FIELDS}
    powertrain = {k: row.get(k) for k in _POWERTRAIN_VDE_FIELDS}
    powertrain.update({k: fuel_row.get(k) for k in _POWERTRAIN_FUELCONS_FIELDS})
    fuel_energy = {k: fuel_row.get(k) for k in _FUEL_ENERGY_FIELDS}
    fuel_energy["electrification"] = fuel_row.get("electrification")
    fuel_energy["fuel_type"] = fuel_row.get("fuel_type")

    revision = _resolve_revision(fuel_row, row)
    origin = fuel_row.get("record_origin")
    provenance = ComparisonProvenance(
        record_origin=origin,
        scenario_intent=origin,
        method=fuel_row.get("engine_method"),
        model_version=fuel_row.get("engine_version"),
        source_vde_revision=revision["source_vde_revision"],
        current_vde_revision=revision["current_vde_revision"],
        revision_status=revision["revision_status"],
        confidence=None,
        source_label=str(origin or "UNKNOWN"),
    )

    warnings: tuple[str, ...] = ()
    if revision["revision_status"] is RevisionStatus.STALE:
        warnings += ("fuelcons_source_stale",)
    warnings += resolve_transmission_boundary(row, temporary_transmission).warnings

    return ComparisonItem(
        role=role,
        source_kind=SourceKind.FUELCONS_SCENARIO,
        vde_id=int(row["id"]) if row.get("id") is not None else linked_vde_id,
        fuelcons_id=int(fuelcons_id),
        label=build_vehicle_label(row),
        vehicle=vehicle,
        powertrain=powertrain,
        roadload=roadload,
        vde={"aggregate": vde_aggregate, "cycle_results": cycle_results},
        fuel_energy=fuel_energy,
        provenance=provenance,
        lineage=_resolve_lineage(row),
        availability=_availability_for(roadload, vde_aggregate, cycle_results, fuel_energy),
        warnings=warnings,
    )


def build_comparison_dataset(
    reference_spec: dict[str, Any],
    comparison_specs: list[dict[str, Any]],
    *,
    temporary_transmission_by_vde_id: dict[int, dict[str, Any]] | None = None,
) -> ComparisonDataset:
    """Build a full dataset with exactly one batch VDE fetch covering every
    VDE_ONLY spec's vde_id AND every FUELCONS_SCENARIO spec's linked vde_id
    (Sec 6 -- no N+1, including when several scenarios share one VDE). Each
    distinct fuelcons_id is still fetched once (selecting specific,
    individually-chosen scenarios is not a batchable relation).
    """
    all_specs = [reference_spec, *comparison_specs]
    temp_by_vde = temporary_transmission_by_vde_id or {}

    fuelcons_rows: dict[int, dict[str, Any]] = {}
    for spec in all_specs:
        if spec.get("kind") == "FUELCONS_SCENARIO":
            fuelcons_id = int(spec["fuelcons_id"])
            if fuelcons_id not in fuelcons_rows:
                row = get_record(EntityType.FUEL_CONSUMPTION, fuelcons_id)
                if not row:
                    raise ValueError(f"FuelCons row not found: id={fuelcons_id}")
                fuelcons_rows[fuelcons_id] = row

    vde_ids: set[int] = {
        int(spec["vde_id"]) for spec in all_specs if spec.get("kind") == "VDE_ONLY" and spec.get("vde_id") is not None
    }
    vde_ids.update(int(row["vde_id"]) for row in fuelcons_rows.values() if row.get("vde_id") is not None)
    vde_row_lookup = {int(row["id"]): row for row in fetch_vde_by_ids(list(vde_ids))} if vde_ids else {}

    def build_one(spec: dict[str, Any], role: ComparisonRole) -> ComparisonItem:
        kind = spec.get("kind")
        if kind == "FUELCONS_SCENARIO":
            fuelcons_id = int(spec["fuelcons_id"])
            fuel_row = fuelcons_rows[fuelcons_id]
            linked_vde_id = fuel_row.get("vde_id")
            return build_scenario_comparison_item(
                fuelcons_id,
                role=role,
                fuelcons_row=fuel_row,
                vde_row=vde_row_lookup.get(int(linked_vde_id)) if linked_vde_id is not None else None,
                temporary_transmission=temp_by_vde.get(int(linked_vde_id)) if linked_vde_id is not None else None,
            )
        if kind == "VDE_ONLY":
            vde_id = int(spec["vde_id"])
            return build_vde_comparison_item(
                vde_id,
                role=role,
                vde_row=vde_row_lookup.get(vde_id),
                temporary_transmission=temp_by_vde.get(vde_id),
            )
        raise ValueError(f"Unknown comparison spec kind: {kind!r}")

    reference = build_one(reference_spec, ComparisonRole.REFERENCE)
    comparisons = tuple(build_one(spec, ComparisonRole.COMPARISON) for spec in comparison_specs)
    warnings = tuple(dict.fromkeys(w for item in (reference, *comparisons) for w in item.warnings))
    return ComparisonDataset(reference=reference, comparisons=comparisons, warnings=warnings)


# -----------------------------------------------------------------------------
# Scenario catalog (Sec 44-46, Package 8B)
#
# Lightweight metadata for selection UIs -- never resolves a full ComparisonItem
# (no transmission/cycle/ABC work) just to populate a selector.
# -----------------------------------------------------------------------------


def list_comparison_scenarios(filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    filters = filters or {}
    sql = (
        "SELECT f.id AS fuelcons_id, f.vde_id, f.electrification, f.fuel_type, "
        "f.record_origin, f.created_at, "
        "v.make, v.model, v.year, v.category, v.legislation "
        "FROM fuelcons_db f JOIN vde_db v ON v.id = f.vde_id WHERE 1=1"
    )
    params: list[Any] = []
    if filters.get("make"):
        sql += " AND v.make = ?"
        params.append(filters["make"])
    if filters.get("legislation"):
        sql += " AND v.legislation = ?"
        params.append(filters["legislation"])
    if filters.get("electrification"):
        sql += " AND f.electrification = ?"
        params.append(filters["electrification"])
    if filters.get("record_origin"):
        sql += " AND f.record_origin = ?"
        params.append(filters["record_origin"])
    sql += " ORDER BY f.created_at DESC"
    return db_module.fetchall(sql, tuple(params)) if params else db_module.fetchall(sql)


# -----------------------------------------------------------------------------
# Metric comparison (Sec 22-27)
# -----------------------------------------------------------------------------

_METRIC_EXTRACTORS: dict[str, Any] = {
    "make": lambda item: item.vehicle.get("make"),
    "model": lambda item: item.vehicle.get("model"),
    "year": lambda item: item.vehicle.get("year"),
    "category": lambda item: item.vehicle.get("category"),
    "legislation": lambda item: item.vehicle.get("legislation"),
    "cycle_name": lambda item: item.vehicle.get("cycle_name"),
    "electrification": lambda item: item.powertrain.get("electrification"),
    "fuel_type": lambda item: item.powertrain.get("fuel_type"),
    "engine_type": lambda item: item.powertrain.get("engine_type"),
    "drive_type": lambda item: item.vehicle.get("drive_type"),
    "transmission_type": lambda item: item.powertrain.get("transmission_type"),
    "gear_count": lambda item: item.powertrain.get("gear_count"),
    "final_drive_ratio": lambda item: item.powertrain.get("final_drive_ratio"),
    "mass_kg": lambda item: item.vehicle.get("mass_kg"),
    "test_mass_kg": lambda item: item.vehicle.get("test_mass_kg"),
    "cda_m2": lambda item: item.vehicle.get("cda_m2"),
    "rrc_n_per_kn": lambda item: item.vehicle.get("rrc_N_per_kN"),
    "roadload_a_total": lambda item: item.roadload["total"].A if item.roadload["total"].available else None,
    "roadload_b_total": lambda item: item.roadload["total"].B if item.roadload["total"].available else None,
    "roadload_c_total": lambda item: item.roadload["total"].C if item.roadload["total"].available else None,
    "roadload_a_net": lambda item: item.roadload["net"].A if item.roadload["net"].available else None,
    "roadload_b_net": lambda item: item.roadload["net"].B if item.roadload["net"].available else None,
    "roadload_c_net": lambda item: item.roadload["net"].C if item.roadload["net"].available else None,
    "vde_total": lambda item: item.vde["aggregate"]["total"],
    "vde_net": lambda item: item.vde["aggregate"]["net"],
    "fuel_l_per_100km": lambda item: (item.fuel_energy or {}).get("fuel_l_per_100km"),
    "fuel_km_per_l": lambda item: (item.fuel_energy or {}).get("fuel_km_per_l"),
    "energy_wh_per_km": lambda item: (item.fuel_energy or {}).get("energy_Wh_per_km"),
    "gco2_per_km": lambda item: (item.fuel_energy or {}).get("gco2_per_km"),
    "eta_pt_est": lambda item: (item.fuel_energy or {}).get("eta_pt_est"),
}


def _check_metric_compatibility(
    rule: ComparisonRule, reference_item: ComparisonItem, comparison_item: ComparisonItem
) -> tuple[bool, bool]:
    """Returns (compatible, basis_mismatch)."""
    if rule is ComparisonRule.ALWAYS:
        return True, False
    same_legislation = reference_item.vehicle.get("legislation") == comparison_item.vehicle.get("legislation")
    if rule is ComparisonRule.BASIS_METADATA:
        return True, not same_legislation
    if rule is ComparisonRule.SAME_LEGISLATION_CYCLE:
        return same_legislation, False
    return True, False


def _semantic_for(direction: MetricDirection, absolute_delta: float) -> str | None:
    if direction not in (MetricDirection.LOWER_IS_BETTER, MetricDirection.HIGHER_IS_BETTER):
        return None
    if absolute_delta == 0:
        return "SAME"
    if direction is MetricDirection.LOWER_IS_BETTER:
        return "BETTER" if absolute_delta < 0 else "WORSE"
    return "BETTER" if absolute_delta > 0 else "WORSE"


def compare_metric(reference_item: ComparisonItem, comparison_item: ComparisonItem, metric_key: str) -> dict[str, Any]:
    """Canonical delta of one metric between a comparison item and Reference.

    Never assigns an overall vehicle score, never encodes color/UI concerns.
    `semantic` is BETTER/WORSE/SAME derived purely from the metric's direction,
    or None for NEUTRAL/CONTEXT_DEPENDENT metrics and non-numeric fields.
    """
    metric = get_metric(metric_key)
    result: dict[str, Any] = {
        "compatible": False,
        "available": False,
        "reference_value": None,
        "comparison_value": None,
        "absolute_delta": None,
        "percent_delta": None,
        "semantic": None,
        "basis_mismatch": False,
    }
    if metric is None or metric.key not in _METRIC_EXTRACTORS:
        return result

    compatible, basis_mismatch = _check_metric_compatibility(metric.comparison_rule, reference_item, comparison_item)
    extractor = _METRIC_EXTRACTORS[metric.key]
    ref_value = extractor(reference_item)
    cmp_value = extractor(comparison_item)
    available = ref_value is not None and cmp_value is not None

    result.update(
        compatible=compatible,
        available=available,
        reference_value=ref_value,
        comparison_value=cmp_value,
        basis_mismatch=basis_mismatch,
    )
    if not compatible or not available or metric.unit_family == "text":
        return result

    try:
        ref_f = float(ref_value)
        cmp_f = float(cmp_value)
    except (TypeError, ValueError):
        return result

    absolute_delta = cmp_f - ref_f
    percent_delta = ((cmp_f / ref_f) - 1.0) * 100.0 if ref_f != 0 else None
    result.update(
        absolute_delta=absolute_delta,
        percent_delta=percent_delta,
        semantic=_semantic_for(metric.direction, absolute_delta),
    )
    return result


__all__ = [
    "load_vde_report_frame",
    "ensure_report_abc_aliases",
    "build_vehicle_label",
    "ComparisonRole",
    "SourceKind",
    "TransmissionStatus",
    "LineageStatus",
    "RevisionStatus",
    "RoadloadBoundary",
    "TransmissionResolution",
    "VDEBoundaryResult",
    "ComparisonProvenance",
    "ComparisonItem",
    "ComparisonDataset",
    "resolve_temporary_transmission_from_component",
    "resolve_transmission_boundary",
    "resolve_roadload_boundaries",
    "compute_vde_for_boundary",
    "resolve_cycle_vde_results",
    "resolve_vde_aggregate",
    "build_vde_comparison_item",
    "build_scenario_comparison_item",
    "build_comparison_dataset",
    "list_comparison_scenarios",
    "compare_metric",
]
