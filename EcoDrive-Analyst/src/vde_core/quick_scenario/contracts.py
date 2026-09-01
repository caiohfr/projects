# src/vde_core/quick_scenario/contracts.py
# -----------------------------------------------------------------------------
# Sprint 10A - canonical, Streamlit-free contracts for Interactive Quick
# Scenarios (Comparison workflow).
#
# A Quick Scenario is a temporary, session-scoped variant of ONE existing,
# already-resolved Comparison scenario: a small set of Vehicle overrides
# (Mass / Tire / CdA) plus an explicit Final PSE assumption, never persisted
# to fuelcons_db/vde_db and never mutating its source scenario. This module
# defines ONLY data shape -- it does not resolve Mass/Tire/Aero, does not
# call the canonical VDE resolvers (vde_mass_proposal_resolver.py,
# vde_tire_proposal_resolver.py, vde_request_resolver.py), and does not
# build a VehicleDemandRequest/VehicleDemandResult. Wiring these contracts
# into those resolvers is a later package's job; see
# docs/sprints/SPRINT_10A_QUICK_SCENARIO_CONTRACT_AUDIT.md for the reuse
# audit these contracts were designed against.
#
# The frozen Sprint 9 Vehicle Demand Core (src/vde_core/vehicle_demand/) is
# not imported here and must not be -- Quick Scenario physics semantics come
# only from the canonical Mass/Tire/Aero resolvers, never from a parallel
# formula in this package.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from src.vde_core.technology_delta import TechDeltaAssumption

QUICK_SCENARIO_CONTRACT_VERSION = "0.1"

# Sec 3: "one existing Comparison scenario as common source; maximum 3 Quick
# Scenarios; all 3 independently inherit the same source."
MAX_QUICK_SCENARIOS_PER_SOURCE = 3

# Sec 3: Quick Scenario identity lives in its own namespace, distinct from
# the existing fc:<fuelcons_id> / vde:<vde_id> Comparison identity
# (comparison_report_viewmodels.canonical_identity), so a Quick Scenario can
# never collide with, or be mistaken for, a real persisted scenario.
QUICK_SCENARIO_IDENTITY_PREFIX = "qs"


class _TextEnum(str, Enum):
    """Same str-Enum shape used throughout vde_core (see
    vehicle_demand.contracts._TextEnum, vde_net_total_contract._TextEnum,
    comparison_report_service._TextEnum) -- JSON-serializable as a plain
    string, stable across the module boundary.
    """

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ScalarChangeMode(_TextEnum):
    """Generic scalar-change vocabulary shared by Mass (curb mass) and Aero
    (CdA) Quick overrides (Sec 20C). Domain semantics that conflict with
    this generic ABSOLUTE/DELTA/PERCENT shape -- e.g. Tire Improvement %'s
    lower-RR-positive convention -- are represented by their own dedicated
    field instead of being forced through this enum (Sec 6, Sec 20C).
    """

    ABSOLUTE = "ABSOLUTE"
    DELTA = "DELTA"
    PERCENT = "PERCENT"


@dataclass(frozen=True)
class ScalarChange:
    """A requested change to one scalar physical quantity (e.g. curb mass,
    CdA), resolved against an optional source value (Sec 20C):

        ABSOLUTE: resolved = value                      (source not required)
        DELTA:    resolved = source + value              (requires source)
        PERCENT:  resolved = source * (1 + value / 100)  (requires source)

    `value` is a required field, never Optional -- 0.0 is a legitimate,
    explicit change (e.g. "no percent change", "zero delta"), distinct from
    "no override requested", which is represented by the containing
    Optional[ScalarChange] field being None (Sec 5: "Zero must remain an
    explicit value where physically permitted. Blank means no override.").
    """

    mode: ScalarChangeMode
    value: float

    def resolve(self, source: float | None) -> float | None:
        """Resolve this change against `source`.

        Returns None (never a guess) when a DELTA/PERCENT change has no
        source to apply against (Sec 5: "If a requested transformation
        requires source ... and it is unavailable, return an explicit
        unresolved/missing state rather than guessing.").
        """

        if self.mode is ScalarChangeMode.ABSOLUTE:
            return float(self.value)
        if source is None:
            return None
        if self.mode is ScalarChangeMode.DELTA:
            return float(source) + float(self.value)
        if self.mode is ScalarChangeMode.PERCENT:
            return float(source) * (1.0 + float(self.value) / 100.0)
        raise ValueError(f"Unknown ScalarChangeMode: {self.mode!r}")  # pragma: no cover


class DomainReadiness(_TextEnum):
    """Per-domain readiness for a requested Vehicle Quick transformation
    (Sec 18). NOT_REQUESTED means the user did not ask for a change in that
    domain -- it never blocks scenario readiness. READY/MISSING/INVALID
    apply only to domains that were actually requested.
    """

    NOT_REQUESTED = "NOT_REQUESTED"
    READY = "READY"
    MISSING = "MISSING"
    INVALID = "INVALID"


@dataclass(frozen=True)
class QuickVehicleReadiness:
    """Sec 18: a requested Vehicle Quick Scenario remains unresolved until
    EVERY requested Vehicle transformation can be applied -- no silent
    partial override. `all_ready` reflects only the domains actually
    requested; NOT_REQUESTED entries never block it.
    """

    mass: DomainReadiness = DomainReadiness.NOT_REQUESTED
    aero: DomainReadiness = DomainReadiness.NOT_REQUESTED
    tire: DomainReadiness = DomainReadiness.NOT_REQUESTED

    @property
    def all_ready(self) -> bool:
        requested = (
            status
            for status in (self.mass, self.aero, self.tire)
            if status is not DomainReadiness.NOT_REQUESTED
        )
        return all(status is DomainReadiness.READY for status in requested)


class TireSource(_TextEnum):
    """Sec 6: Tire Source is a separate concept from Tire Change. Selecting
    TIRE_DB establishes a new tire reference; it is not itself a mutually
    exclusive calculation mode.
    """

    CURRENT = "CURRENT"
    TIRE_DB = "TIRE_DB"


class TireTransformMode(_TextEnum):
    """Sec 7: one Tire transformation after source selection. Which modes
    are valid depends on TireSource -- see TireQuickChange.__post_init__ and
    ALLOWED_TIRE_TRANSFORMS_BY_SOURCE below.
    """

    NONE = "NONE"
    TARGET_RRC = "TARGET_RRC"
    RRC_DELTA = "RRC_DELTA"
    IMPROVEMENT_PCT = "IMPROVEMENT_PCT"
    PRESSURE_DELTA = "PRESSURE_DELTA"


# Sec 7: allowed (source, transform) combinations. TIRE_DB deliberately
# excludes TARGET_RRC/RRC_DELTA ("Do not initially support: Tire DB +
# arbitrary Target RRC, Tire DB + arbitrary RRC Delta", Sec 6). Public (not
# `_`-prefixed) since Sprint 10E's UI reads it directly to restrict which
# TireTransformMode widget options are even offered for a given TireSource,
# rather than maintaining a second, UI-side copy of this mapping.
ALLOWED_TIRE_TRANSFORMS_BY_SOURCE: Mapping[TireSource, frozenset[TireTransformMode]] = {
    TireSource.CURRENT: frozenset(
        {
            TireTransformMode.NONE,
            TireTransformMode.TARGET_RRC,
            TireTransformMode.RRC_DELTA,
            TireTransformMode.IMPROVEMENT_PCT,
            TireTransformMode.PRESSURE_DELTA,
        }
    ),
    TireSource.TIRE_DB: frozenset(
        {
            TireTransformMode.NONE,
            TireTransformMode.IMPROVEMENT_PCT,
            TireTransformMode.PRESSURE_DELTA,
        }
    ),
}


class ReferencePressureProvenance(_TextEnum):
    """Sec 6: the frozen product decision requires distinguishing DB/source
    reference pressure from a user-supplied one -- no silent default when a
    reference pressure is missing.
    """

    SOURCE = "SOURCE"
    USER_PROVIDED = "USER_PROVIDED"


@dataclass(frozen=True)
class TirePressureDelta:
    """Sec 6: Pressure Delta Estimate, supporting both a single front/rear
    delta and an independent split. `rear_delta_psi=None` means "apply
    front_delta_psi to both axles" (same-delta UX); an explicit rear value
    is an independently split delta.

    A missing reference pressure must not be silently defaulted (Sec 6): a
    USER_PROVIDED provenance requires `reference_pressure_psi` to be
    supplied here so a later resolver package can use it instead of a DB
    value; a SOURCE provenance defers to whatever the canonical tire
    resolver/DB has (may itself be unavailable, resolved downstream, not by
    this contract).
    """

    front_delta_psi: float
    rear_delta_psi: float | None = None
    reference_pressure_psi: float | None = None
    reference_pressure_provenance: ReferencePressureProvenance | None = None

    def __post_init__(self) -> None:
        if self.reference_pressure_provenance is ReferencePressureProvenance.USER_PROVIDED:
            if self.reference_pressure_psi is None:
                raise ValueError(
                    "TirePressureDelta.reference_pressure_psi is required when "
                    "reference_pressure_provenance is USER_PROVIDED (Sec 6: no silent "
                    "reference-pressure default)."
                )


@dataclass(frozen=True)
class TireQuickChange:
    """Sec 6-7: Tire Source (CURRENT / TIRE_DB) plus at most one Tire
    Change. Validates only the source/transform-mode combination and the
    presence of the field(s) each mode needs -- it does not resolve RRC,
    Tire ABC, or pressure physics; that stays owned by
    vde_tire_proposal_resolver.resolve_tire_proposal (see the audit doc).
    """

    source: TireSource
    transform_mode: TireTransformMode = TireTransformMode.NONE
    tire_db_id: int | None = None
    target_rrc_n_per_kn: float | None = None
    rrc_delta_n_per_kn: float | None = None
    improvement_pct: float | None = None
    pressure_delta: TirePressureDelta | None = None

    def __post_init__(self) -> None:
        allowed = ALLOWED_TIRE_TRANSFORMS_BY_SOURCE[self.source]
        if self.transform_mode not in allowed:
            raise ValueError(
                f"TireTransformMode.{self.transform_mode.value} is not supported for "
                f"TireSource.{self.source.value} (Sec 6-7 tire "
                "transformation limit)."
            )
        if self.source is TireSource.TIRE_DB and self.tire_db_id is None:
            raise ValueError("TireQuickChange.tire_db_id is required when source is TIRE_DB.")
        if self.transform_mode is TireTransformMode.TARGET_RRC and self.target_rrc_n_per_kn is None:
            raise ValueError("TireQuickChange.target_rrc_n_per_kn is required for TARGET_RRC.")
        if self.transform_mode is TireTransformMode.RRC_DELTA and self.rrc_delta_n_per_kn is None:
            raise ValueError("TireQuickChange.rrc_delta_n_per_kn is required for RRC_DELTA.")
        if self.transform_mode is TireTransformMode.IMPROVEMENT_PCT and self.improvement_pct is None:
            raise ValueError("TireQuickChange.improvement_pct is required for IMPROVEMENT_PCT.")
        if self.transform_mode is TireTransformMode.PRESSURE_DELTA and self.pressure_delta is None:
            raise ValueError("TireQuickChange.pressure_delta is required for PRESSURE_DELTA.")

        supplied_transform_fields = {
            TireTransformMode.TARGET_RRC: self.target_rrc_n_per_kn,
            TireTransformMode.RRC_DELTA: self.rrc_delta_n_per_kn,
            TireTransformMode.IMPROVEMENT_PCT: self.improvement_pct,
            TireTransformMode.PRESSURE_DELTA: self.pressure_delta,
        }
        extraneous = [
            mode.value
            for mode, value in supplied_transform_fields.items()
            if value is not None and mode is not self.transform_mode
        ]
        if extraneous:
            raise ValueError(
                "TireQuickChange permits exactly one transformation after source selection; "
                f"{self.transform_mode.value} cannot be stacked with {', '.join(extraneous)}."
            )
        if self.source is TireSource.CURRENT and self.tire_db_id is not None:
            raise ValueError(
                "TireQuickChange.tire_db_id is only valid when source is TIRE_DB."
            )


@dataclass(frozen=True)
class MassQuickChange:
    """Sec 4.1: Vehicle/Curb Mass supports two mutually-exclusive request
    shapes that a single generic ScalarChange cannot represent:

    - `curb_change`: an Absolute/Delta/Percent change to curb mass, which
      the canonical Mass resolver turns into a regulatory mass via
      `EPA_CURB_TO_TWC` (EPA) or `WLTP_MASS_LINE` (WLTP) -- "Target TWC" /
      "canonical supported WLTP mass behavior."
    - `twc_shift_steps`: an EPA-only step count from the *current* TWC
      bracket (no curb-mass input at all) -- "TWC Shift", resolved via
      `MASS_TWC_SHIFT`.

    Exactly one of the two must be set; `__post_init__` enforces this so an
    ambiguous or empty request is rejected at construction time rather than
    silently defaulting to one interpretation. `twc_shift_side`/
    `twc_curb_position`/`wltp_line_type` are optional pass-throughs to the
    canonical resolver's own defaults ("Up"/"Top"/"TML") when omitted.
    """

    curb_change: ScalarChange | None = None
    twc_shift_steps: float | None = None
    twc_shift_side: str | None = None
    twc_curb_position: str | None = None
    wltp_line_type: str | None = None

    def __post_init__(self) -> None:
        if (self.curb_change is None) == (self.twc_shift_steps is None):
            raise ValueError(
                "MassQuickChange requires exactly one of curb_change or "
                "twc_shift_steps (Sec 4.1: Target TWC / WLTP mass line vs. "
                "TWC Shift are distinct request shapes)."
            )


@dataclass(frozen=True)
class VehicleQuickOverrides:
    """Sec 4-8: the "Vehicle Quick" layer -- Mass, CdA (Aero), and Tire
    overrides requested against one source Comparison scenario. Resolution
    order is Mass -> Tire -> Aero (Sec 8), implemented by resolver.py; this
    contract only carries and validates the requested inputs.

    `aero_reference_cda_m2`/`aero_reference_cda_provenance` mirror
    `TirePressureDelta`'s reference-value shape: the canonical Aero resolver
    always requires a reference CdA to convert an Absolute CdA request into
    a roadload-C delta, even when the request itself needs no source
    (Sec 5). When source CdA is unavailable, a resolver may fall back to
    this explicit, user-provided reference instead of guessing one --
    never silently defaulting it to zero.
    """

    mass_change: MassQuickChange | None = None
    cda_change: ScalarChange | None = None
    aero_reference_cda_m2: float | None = None
    aero_reference_cda_provenance: ReferencePressureProvenance | None = None
    tire_change: TireQuickChange | None = None

    def __post_init__(self) -> None:
        if self.aero_reference_cda_provenance is ReferencePressureProvenance.USER_PROVIDED:
            if self.aero_reference_cda_m2 is None:
                raise ValueError(
                    "VehicleQuickOverrides.aero_reference_cda_m2 is required when "
                    "aero_reference_cda_provenance is USER_PROVIDED (Sec 5: no silent "
                    "reference-CdA default)."
                )

    @property
    def is_empty(self) -> bool:
        return self.mass_change is None and self.cda_change is None and self.tire_change is None


MAX_TECH_DELTAS_PER_SCENARIO = 3

# TechDeltaAssumption is imported from src.vde_core.technology_delta above
# (Sprint 11C Pre-flight 3B ownership cleanup) -- it moved to that neutral,
# feature-agnostic module once System Scenario needed the same canonical
# contract, rather than a second feature package importing it from here.
# Still exported from this module for backward compatibility (identical
# object, not a copy -- see tests/test_technology_delta.py).


@dataclass(frozen=True)
class EfficiencyQuickInputs:
    """Sec 2/5/9/13: the "Efficiency Quick" layer's requested inputs --
    kept as a sibling of `VehicleQuickOverrides`, never nested inside it
    (Sec 2: "Vehicle and Powertrain/Efficiency must remain separate").

    `benchmark_source_identity`, when set, must be a full Comparison
    identity string for the DONOR scenario (`fc:<id>`/`vde:<id>`), mirroring
    `QuickScenario.source_identity`'s own rule -- never a bare vde_id.
    `request_ml_recommendation` is an explicit opt-in (ML inference has a
    real cost -- deserializing the artifact per call -- and is itself a
    user action akin to "Use ML recommendation", Sec 5/10).
    `technology_deltas` is capped at `MAX_TECH_DELTAS_PER_SCENARIO` (Sec 15:
    "a product complexity limit, not a new physics rule").

    None of these fields, on their own, changes Final PSE (Sec 5/14): they
    only make references/recommendations computable for the resolver to
    expose -- adoption is a separate, explicit act of setting
    `QuickScenario.final_pse_percent`/`pse_provenance`.
    """

    benchmark_source_identity: str | None = None
    request_ml_recommendation: bool = False
    technology_deltas: tuple[TechDeltaAssumption, ...] = ()

    def __post_init__(self) -> None:
        if len(self.technology_deltas) > MAX_TECH_DELTAS_PER_SCENARIO:
            raise ValueError(
                "EfficiencyQuickInputs.technology_deltas exceeds the Sprint 10D "
                f"product limit of {MAX_TECH_DELTAS_PER_SCENARIO} (Sec 15)."
            )
        if self.benchmark_source_identity is not None and not self.benchmark_source_identity:
            raise ValueError(
                "EfficiencyQuickInputs.benchmark_source_identity must be a non-empty "
                "identity string or None."
            )

    @property
    def is_empty(self) -> bool:
        return (
            self.benchmark_source_identity is None
            and not self.request_ml_recommendation
            and not self.technology_deltas
        )


class PseProvenance(_TextEnum):
    """Sec 9-14: Final PSE provenance. References/recommendations
    (benchmark, ML, Tech Delta) only reach Final PSE once the user
    explicitly accepts them (*_ACCEPTED); a manual edit -- including editing
    a previously accepted recommendation's value -- is USER_PROVIDED, never
    a stale *_ACCEPTED label (Sec 10: "the final value is user-edited/
    user-provided rather than pretending the exact recommendation remained
    authoritative").
    """

    INHERITED_CURRENT = "INHERITED_CURRENT"
    USER_PROVIDED = "USER_PROVIDED"
    BENCHMARK_ACCEPTED = "BENCHMARK_ACCEPTED"
    ML_RECOMMENDATION_ACCEPTED = "ML_RECOMMENDATION_ACCEPTED"
    TECH_DELTA_ACCEPTED = "TECH_DELTA_ACCEPTED"


def build_quick_scenario_identity(source_identity: str, slot: int) -> str:
    """Sec 3: Quick Scenario identity, distinct from the existing
    fc:<fuelcons_id> / vde:<vde_id> Comparison identity namespace
    (comparison_report_viewmodels.canonical_identity) so a Quick Scenario
    can never collide with, or be mistaken for, a real persisted scenario.
    """

    return f"{QUICK_SCENARIO_IDENTITY_PREFIX}:{source_identity}:{slot}"


@dataclass(frozen=True)
class QuickScenario:
    """A temporary, session-scoped Quick Scenario derived from ONE existing,
    already-resolved Comparison scenario (Sec 3).

    `source_identity` must be a full Comparison identity string -- the
    existing fc:<fuelcons_id> / vde:<vde_id> canonical_identity() value
    (Sec 3: "scenario identity must preserve full Comparison identity, not
    only vde_id") -- never a bare vde_id, and never another Quick Scenario's
    identity (no Quick -> Quick lineage, Sec 3).

    This contract is never persisted: it has no save()/to_db_row() method
    and no field pointing at a fuelcons_db/vde_db row that this scenario
    owns or writes to. Holding or building a QuickScenario never mutates the
    source scenario referenced by `source_identity`.
    """

    source_identity: str
    slot: int
    label: str | None = None
    vehicle_overrides: VehicleQuickOverrides = field(default_factory=VehicleQuickOverrides)
    vehicle_readiness: QuickVehicleReadiness = field(default_factory=QuickVehicleReadiness)
    efficiency_inputs: EfficiencyQuickInputs = field(default_factory=EfficiencyQuickInputs)
    final_pse_percent: float | None = None
    pse_provenance: PseProvenance | None = None
    issues: tuple[str, ...] = ()
    contract_version: str = QUICK_SCENARIO_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.source_identity:
            raise ValueError("QuickScenario.source_identity is required.")
        if self.source_identity.startswith(f"{QUICK_SCENARIO_IDENTITY_PREFIX}:"):
            raise ValueError(
                "QuickScenario.source_identity cannot be another Quick Scenario's "
                "identity (no Quick -> Quick lineage, Sec 3)."
            )
        if not (1 <= self.slot <= MAX_QUICK_SCENARIOS_PER_SOURCE):
            raise ValueError(
                f"QuickScenario.slot must be between 1 and {MAX_QUICK_SCENARIOS_PER_SOURCE} "
                f"(Sec 3: max {MAX_QUICK_SCENARIOS_PER_SOURCE} Quick Scenarios per source), "
                f"got {self.slot}."
            )
        if (self.final_pse_percent is None) != (self.pse_provenance is None):
            raise ValueError(
                "QuickScenario.final_pse_percent and pse_provenance must both be set or "
                "both be absent (Sec 10: Final PSE always carries explicit provenance)."
            )

    @property
    def identity(self) -> str:
        return build_quick_scenario_identity(self.source_identity, self.slot)

    @property
    def is_vehicle_ready(self) -> bool:
        return self.vehicle_readiness.all_ready


__all__ = [
    "QUICK_SCENARIO_CONTRACT_VERSION",
    "MAX_QUICK_SCENARIOS_PER_SOURCE",
    "QUICK_SCENARIO_IDENTITY_PREFIX",
    "ScalarChangeMode",
    "ScalarChange",
    "DomainReadiness",
    "QuickVehicleReadiness",
    "TireSource",
    "TireTransformMode",
    "ALLOWED_TIRE_TRANSFORMS_BY_SOURCE",
    "ReferencePressureProvenance",
    "TirePressureDelta",
    "TireQuickChange",
    "MassQuickChange",
    "VehicleQuickOverrides",
    "MAX_TECH_DELTAS_PER_SCENARIO",
    "TechDeltaAssumption",
    "EfficiencyQuickInputs",
    "PseProvenance",
    "build_quick_scenario_identity",
    "QuickScenario",
]
