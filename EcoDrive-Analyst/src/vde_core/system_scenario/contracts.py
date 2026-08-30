# src/vde_core/system_scenario/contracts.py
# -----------------------------------------------------------------------------
# Sprint 11A - canonical, Streamlit-free contracts for the multi-domain
# System Scenario architecture (docs/specs/sprint_11_multi_domain_system_scenario.md).
#
# A System Scenario (Current, or one of up to 3 Proposals) is a composition
# of independent per-domain states across the 8 approved domains (Vehicle
# Demand, Architecture, Engine/Fuel Converter, Transmission/Driveline,
# Electric Drive, Energy Storage, Energy Management/Controls, Aux/Thermal).
# Every Domain Proposal is based on that domain's Effective Current, never on
# another Domain Proposal (Sec 8, REQ-11-007) -- enforced structurally below
# by typing `DomainProposal.based_on` as `EffectiveDomainState`, never
# `DomainProposal`.
#
# This module defines DATA SHAPE ONLY. It contains no physics, no Streamlit
# import, and no database access. It does not decide fidelity, does not
# compose Technology Deltas, and does not call the existing L0 fuel
# estimation path -- that resolution/composition work is explicitly deferred
# to Sprint 11B/11C (see docs/sprints/SPRINT_11A_SYSTEM_SCENARIO_CONTRACTS.md).
#
# `VehicleDemandConfiguration` is the one domain whose "configuration" is a
# reference to an already-resolved frozen Sprint 9 `VehicleDemandResult`
# (`src/vde_core/vehicle_demand/contracts.py`), never a re-derivation of
# roadload/VDE physics (INV-11-006). Nothing in this module imports
# `fuel_estimation.py`/`powertrain_efficiency.py`; it imports only the shared
# typed Technology Delta contract from its neutral owner. Calculation wiring
# lives in Sprint 11C's `resolver.py`/`l0_adapter.py` (spec Sec 28), not here.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Union

from src.vde_core.technology_delta import TechDeltaAssumption
from src.vde_core.vehicle_demand import VehicleDemandResult

SYSTEM_SCENARIO_CONTRACT_VERSION = "0.1"

# Sec 10/26: "Current + max 3 proposals... Do not build arbitrary-N scenario
# management." Mirrors MAX_QUICK_SCENARIOS_PER_SOURCE's role in Sprint 10A.
MAX_SYSTEM_SCENARIO_PROPOSALS = 3

# Sec 15: "A Domain Proposal may contain two separate parts: Physical/
# Configuration Proposal + optional fidelity-specific quantitative
# representation" -- Sprint 11 caps this at up to 3 reusable proposals per
# domain, mirroring the System Scenario cap (a modest, explicit product
# limit, not a physics rule).
MAX_DOMAIN_PROPOSALS_PER_DOMAIN = 3


class _TextEnum(str, Enum):
    """Same str-Enum shape used throughout vde_core (see
    vehicle_demand.contracts._TextEnum, comparison_report_service._TextEnum,
    quick_scenario.contracts._TextEnum) -- JSON-serializable as a plain
    string, stable across the module boundary. Copied per established
    project convention rather than imported, since importing a 3-line mixin
    across unrelated packages would be its own odd coupling.
    """

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class DomainKind(_TextEnum):
    """Sec 5: the 8 approved, taxonomy-stable domains. Not every domain must
    quantitatively affect Energy Balance L0 (Sec 5) -- that's a Fidelity
    Manifest question (see FidelityLevel below), not a taxonomy question.
    """

    VEHICLE_DEMAND = "VEHICLE_DEMAND"
    ARCHITECTURE = "ARCHITECTURE"
    ENGINE_FUEL_CONVERTER = "ENGINE_FUEL_CONVERTER"
    TRANSMISSION_DRIVELINE = "TRANSMISSION_DRIVELINE"
    ELECTRIC_DRIVE = "ELECTRIC_DRIVE"
    ENERGY_STORAGE = "ENERGY_STORAGE"
    ENERGY_MANAGEMENT_CONTROLS = "ENERGY_MANAGEMENT_CONTROLS"
    AUX_THERMAL = "AUX_THERMAL"


ALL_DOMAIN_KINDS: tuple[DomainKind, ...] = (
    DomainKind.VEHICLE_DEMAND,
    DomainKind.ARCHITECTURE,
    DomainKind.ENGINE_FUEL_CONVERTER,
    DomainKind.TRANSMISSION_DRIVELINE,
    DomainKind.ELECTRIC_DRIVE,
    DomainKind.ENERGY_STORAGE,
    DomainKind.ENERGY_MANAGEMENT_CONTROLS,
    DomainKind.AUX_THERMAL,
)


class ArchitectureClass(_TextEnum):
    """Sec 6: classification only. No topology graph, no ports/connections,
    no P0-P4 semantics in Sprint 11 (spec Sec 6/42) -- those are explicitly
    future capabilities.
    """

    ICE = "ICE"
    MHEV = "MHEV"
    HEV = "HEV"
    PHEV = "PHEV"
    BEV = "BEV"


class FidelityLevel(_TextEnum):
    """Sec 23: minimum Fidelity Manifest semantics. Answers "did this domain
    actually influence this result at this fidelity?" -- populated by a
    the Sprint 11C resolver, not by this contracts module.
    """

    QUANTITATIVE = "QUANTITATIVE"
    EFFECTIVE_ASSUMPTION = "EFFECTIVE_ASSUMPTION"
    CONFIGURATION_ONLY = "CONFIGURATION_ONLY"
    NOT_REPRESENTED = "NOT_REPRESENTED"


class SolverReadiness(_TextEnum):
    """Energy Balance L0 readiness, kept separate from domain completeness."""

    READY = "READY"
    NOT_READY = "NOT_READY"


class DomainApplicability(_TextEnum):
    """Sprint 11B Sec 9: a coarse, engineering-judgment classification of
    whether a domain is expected to matter for a given `ArchitectureClass`
    -- REQUIRED/OPTIONAL/NOT_APPLICABLE. Deliberately not a rules engine:
    a fixed lookup table (`_DOMAIN_APPLICABILITY_BY_ARCHITECTURE` below),
    same spirit as `domain_typically_applicable` (kept, still valid) but
    with the 3-state resolution the spec asks for. Purely advisory --
    never a hard gate on constructing a Domain State/Proposal; missing data
    for a REQUIRED domain does not raise here (Sec 24: readiness is a
    future solver concern, not a contract concern).
    """

    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ProvenanceKind(_TextEnum):
    """Sec 35: provenance distinctions that must never be silently collapsed."""

    SOURCE_OBSERVED = "SOURCE_OBSERVED"
    CORRECTED = "CORRECTED"
    ASSUMED = "ASSUMED"
    CALCULATED = "CALCULATED"
    ESTIMATED = "ESTIMATED"
    ML_PREDICTED = "ML_PREDICTED"
    ML_DERIVED = "ML_DERIVED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


# -----------------------------------------------------------------------------
# Per-domain configuration payloads (Sec 12-18): small, explicitly-typed
# dataclasses so domain-specific configuration stays distinguishable by
# Python's own type system -- not a dynamic plugin/domain framework, no
# runtime registration, no domain-type dispatch table. Every field is
# Optional: missing configuration data does not automatically block a System
# Scenario (Sec 24) -- readiness is a solver concern (11C), not a contract
# concern. All are frozen (never mutated in place; a correction/proposal
# constructs a new instance).
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class VehicleDemandConfiguration:
    """Sec 11: Vehicle Demand is one selectable domain/input of each System
    Scenario. `vehicle_demand_result` is always the ALREADY-RESOLVED frozen
    Sprint 9 result -- this contract never re-derives roadload/mass/tire/aero
    physics (INV-11-006). `source_identity` mirrors the existing Comparison/
    Quick Scenario canonical identity convention (`fc:<id>` / `vde:<id>`),
    kept alongside the result so downstream provenance/audit can name which
    source produced it without re-deriving anything from the identity string.
    """

    source_identity: str | None = None
    vehicle_demand_result: VehicleDemandResult | None = None


@dataclass(frozen=True)
class ArchitectureConfiguration:
    """Sec 6/13: classification only. `topology_notes` preserves existing
    useful free-text topology metadata when already available (Sec 6) --
    never a graph/ports/connections structure.
    """

    architecture_class: ArchitectureClass | None = None
    topology_notes: str | None = None


@dataclass(frozen=True)
class EngineConfiguration:
    """Sec 13. Configuration changes alone must not invent a consumption
    impact (Sec 13, Case D) -- that's enforced by the resolver/Fidelity
    Manifest (11C), not by this dataclass; this is data shape only.
    """

    fuel_type: str | None = None
    engine_family_id: str | None = None
    displacement_l: float | None = None
    rated_power_kw: float | None = None
    rated_torque_nm: float | None = None
    technology_descriptors: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransmissionConfiguration:
    """Sec 14. Out of scope: gear schedule, gear-ratio simulation,
    speed/load efficiency maps, operating points (Sec 14) -- none of that is
    represented here."""

    transmission_type: str | None = None
    transmission_model_id: str | None = None
    gear_count: int | None = None
    final_drive_ratio: float | None = None


@dataclass(frozen=True)
class ElectricDriveConfiguration:
    """Sec 15. Power/torque metadata must not automatically create
    performance or energy effects if the L0 solver does not model those
    constraints (Sec 15) -- again a resolver/fidelity concern, not encoded
    here. No motor/inverter maps."""

    motor_role: str | None = None
    motor_count: int | None = None
    motor_position: str | None = None
    rated_power_kw: float | None = None
    peak_power_kw: float | None = None
    rated_torque_nm: float | None = None
    peak_torque_nm: float | None = None
    nominal_voltage_v: float | None = None
    motor_identifier: str | None = None
    inverter_identifier: str | None = None


@dataclass(frozen=True)
class EnergyStorageConfiguration:
    """Sec 16. Much of this may resolve to CONFIGURATION_ONLY fidelity at
    Sprint 11 L0 (Sec 16) -- out of scope: SOC trace, electrochemical model,
    thermal battery model."""

    gross_capacity_kwh: float | None = None
    usable_capacity_kwh: float | None = None
    nominal_voltage_v: float | None = None
    charge_power_limit_kw: float | None = None
    discharge_power_limit_kw: float | None = None
    regen_power_limit_kw: float | None = None
    soc_window_low_pct: float | None = None
    soc_window_high_pct: float | None = None


@dataclass(frozen=True)
class ControlsConfiguration:
    """Sec 17. Out of scope: torque-split simulation, SOC supervisory
    control, hybrid controller simulation."""

    hybrid_operating_strategy: str | None = None
    utility_factor_pct: float | None = None
    regen_metadata: str | None = None
    start_stop_enabled: bool | None = None
    calibration_notes: str | None = None


@dataclass(frozen=True)
class AuxThermalConfiguration:
    """Sec 18: exists so the architecture does not require redesign later.
    Current data may be sparse -- typically CONFIGURATION_ONLY or
    NOT_REPRESENTED fidelity. No new thermal physics.

    `ambient_temp_c`/`ac_on` were added in Sprint 11B after the legacy
    adapter audit confirmed `fuelcons_db.ambient_temp_c`/`fuelcons_db.ac_on`
    are real, populated columns -- not invented to make this domain look
    complete (Sec 15: "Do not invent HVAC/thermal fields... merely to make
    the object look complete"); they are the one confirmed pair of
    Aux/Thermal-relevant columns this codebase actually has today.
    """

    ambient_temp_c: float | None = None
    ac_on: bool | None = None
    notes: str | None = None


DomainConfiguration = Union[
    VehicleDemandConfiguration,
    ArchitectureConfiguration,
    EngineConfiguration,
    TransmissionConfiguration,
    ElectricDriveConfiguration,
    EnergyStorageConfiguration,
    ControlsConfiguration,
    AuxThermalConfiguration,
]

_CONFIGURATION_TYPE_BY_DOMAIN: Mapping[DomainKind, type] = {
    DomainKind.VEHICLE_DEMAND: VehicleDemandConfiguration,
    DomainKind.ARCHITECTURE: ArchitectureConfiguration,
    DomainKind.ENGINE_FUEL_CONVERTER: EngineConfiguration,
    DomainKind.TRANSMISSION_DRIVELINE: TransmissionConfiguration,
    DomainKind.ELECTRIC_DRIVE: ElectricDriveConfiguration,
    DomainKind.ENERGY_STORAGE: EnergyStorageConfiguration,
    DomainKind.ENERGY_MANAGEMENT_CONTROLS: ControlsConfiguration,
    DomainKind.AUX_THERMAL: AuxThermalConfiguration,
}


def configuration_type_for(domain: DomainKind) -> type:
    """The expected configuration dataclass type for one domain -- used by
    validation below and by callers (adapters, future resolvers) that need
    to know which typed shape a domain expects without a dynamic registry.
    """

    return _CONFIGURATION_TYPE_BY_DOMAIN[domain]


def _require_matching_configuration_type(domain: DomainKind, configuration: DomainConfiguration) -> None:
    expected = _CONFIGURATION_TYPE_BY_DOMAIN[domain]
    if not isinstance(configuration, expected):
        raise TypeError(
            f"DomainKind.{domain.value} requires configuration of type {expected.__name__}, "
            f"got {type(configuration).__name__}."
        )


# Sec 6/Case L: architecture applicability as a classification lookup only
# -- no graph, no simulation. The Sprint 11C resolver uses this to help
# decide FidelityLevel.NOT_REPRESENTED for a structurally inapplicable
# domain; this module never blocks construction on it.
_TYPICALLY_INAPPLICABLE_DOMAINS_BY_ARCHITECTURE: Mapping[ArchitectureClass, frozenset[DomainKind]] = {
    ArchitectureClass.ICE: frozenset({DomainKind.ELECTRIC_DRIVE, DomainKind.ENERGY_STORAGE}),
    ArchitectureClass.MHEV: frozenset(),
    ArchitectureClass.HEV: frozenset(),
    ArchitectureClass.PHEV: frozenset(),
    ArchitectureClass.BEV: frozenset({DomainKind.ENGINE_FUEL_CONVERTER}),
}


def domain_typically_applicable(architecture: ArchitectureClass, domain: DomainKind) -> bool:
    """Sec 6/Case L classification helper -- e.g. Engine/Fuel Converter is
    typically N/A for BEV, Electric Drive/Energy Storage are typically N/A
    for plain ICE. Purely informational; never a hard structural gate.
    """

    return domain not in _TYPICALLY_INAPPLICABLE_DOMAINS_BY_ARCHITECTURE.get(architecture, frozenset())


# Sprint 11B Sec 9: the 3-state REQUIRED/OPTIONAL/NOT_APPLICABLE
# classification, matching the spec's stated broad semantics exactly.
# VEHICLE_DEMAND and ARCHITECTURE are REQUIRED for every architecture
# (every System Scenario needs a vehicle demand and an architecture
# classification) and are intentionally omitted from each per-architecture
# override map below -- `domain_applicability_for` falls back to REQUIRED
# for both. A domain with no explicit entry for a given architecture
# defaults to OPTIONAL, never NOT_APPLICABLE, so an unlisted combination
# never silently hides a domain that simply wasn't called out by name in
# the spec's broad semantics (Sec 9).
_DOMAIN_APPLICABILITY_OVERRIDES_BY_ARCHITECTURE: Mapping[ArchitectureClass, Mapping[DomainKind, DomainApplicability]] = {
    ArchitectureClass.ICE: {
        DomainKind.ENGINE_FUEL_CONVERTER: DomainApplicability.REQUIRED,
        DomainKind.ELECTRIC_DRIVE: DomainApplicability.NOT_APPLICABLE,
        DomainKind.ENERGY_STORAGE: DomainApplicability.NOT_APPLICABLE,
    },
    ArchitectureClass.MHEV: {
        DomainKind.ENGINE_FUEL_CONVERTER: DomainApplicability.REQUIRED,
        DomainKind.ELECTRIC_DRIVE: DomainApplicability.REQUIRED,
        DomainKind.ENERGY_STORAGE: DomainApplicability.REQUIRED,
    },
    ArchitectureClass.HEV: {
        DomainKind.ENGINE_FUEL_CONVERTER: DomainApplicability.REQUIRED,
        DomainKind.ELECTRIC_DRIVE: DomainApplicability.REQUIRED,
        DomainKind.ENERGY_STORAGE: DomainApplicability.REQUIRED,
    },
    ArchitectureClass.PHEV: {
        DomainKind.ENGINE_FUEL_CONVERTER: DomainApplicability.REQUIRED,
        DomainKind.ELECTRIC_DRIVE: DomainApplicability.REQUIRED,
        DomainKind.ENERGY_STORAGE: DomainApplicability.REQUIRED,
    },
    ArchitectureClass.BEV: {
        DomainKind.ENGINE_FUEL_CONVERTER: DomainApplicability.NOT_APPLICABLE,
        DomainKind.ELECTRIC_DRIVE: DomainApplicability.REQUIRED,
        DomainKind.ENERGY_STORAGE: DomainApplicability.REQUIRED,
    },
}


def domain_applicability_for(architecture: ArchitectureClass, domain: DomainKind) -> DomainApplicability:
    """Sec 9: coarse REQUIRED/OPTIONAL/NOT_APPLICABLE classification for one
    (architecture, domain) pair. VEHICLE_DEMAND/ARCHITECTURE are always
    REQUIRED; every other domain defaults to OPTIONAL unless a specific
    override says otherwise -- purely advisory, never a hard gate (missing
    data for a REQUIRED domain does not raise anywhere in this module).
    """

    if domain in (DomainKind.VEHICLE_DEMAND, DomainKind.ARCHITECTURE):
        return DomainApplicability.REQUIRED
    overrides = _DOMAIN_APPLICABILITY_OVERRIDES_BY_ARCHITECTURE.get(architecture, {})
    return overrides.get(domain, DomainApplicability.OPTIONAL)


# -----------------------------------------------------------------------------
# Domain State semantics (Sec 7): SOURCE -> CURRENT -> CORRECTION ->
# EFFECTIVE CURRENT -> PROPOSAL. "Current" (the interpreted domain state) and
# "Source" (imported/authoritative source data) are deliberately NOT split
# into two separate contracts here: nothing in the audited current
# implementation (Sprint 11A Sec 3 audit) distinguishes a raw import from an
# already-interpreted-but-uncorrected state at the domain-contract level --
# `DomainSourceState` plays both roles, and a correction (when present) is
# what produces Effective Current. Splitting them further would be
# speculative architecture the spec does not ask for (Sec "small typed
# design is preferred").
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainSourceState:
    """The current interpreted domain state, as imported/read from its
    (legacy or canonical) source -- never mutated by a correction or
    proposal (INV-11-003)."""

    domain: DomainKind
    configuration: DomainConfiguration
    provenance: ProvenanceKind = ProvenanceKind.SOURCE_OBSERVED
    notes: str = ""

    def __post_init__(self) -> None:
        _require_matching_configuration_type(self.domain, self.configuration)


@dataclass(frozen=True)
class DomainCorrection:
    """Sec 7/27: an explicit engineering correction to a domain's Current
    state. `configuration` is the FULL corrected configuration (construct it
    with `dataclasses.replace(source.configuration, **changed_fields)` to
    carry every unrelated field forward unchanged) -- Sprint 11A does not
    introduce a generic partial-patch/merge mechanism, since a full
    typed-dataclass replacement is simpler and already idiomatic in this
    codebase. A correction never mutates the source record (INV-11-003).
    """

    domain: DomainKind
    configuration: DomainConfiguration
    reason: str = ""
    provenance: ProvenanceKind = ProvenanceKind.CORRECTED
    l0_effective_assumption: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_matching_configuration_type(self.domain, self.configuration)
        object.__setattr__(
            self,
            "l0_effective_assumption",
            MappingProxyType(dict(self.l0_effective_assumption)),
        )


@dataclass(frozen=True)
class EffectiveDomainState:
    """Sec 7/27: the corrected (or, if uncorrected, verbatim Current) state
    that becomes the baseline every Domain Proposal for this domain is based
    on. Always carries its own `source` for audit, and `correction` when one
    was applied (`None` otherwise). Construct via `resolve_effective_domain_state`
    below rather than directly, so "never mutates source" and "correction
    domain must match source domain" stay enforced in one place.
    """

    domain: DomainKind
    configuration: DomainConfiguration
    source: DomainSourceState
    correction: DomainCorrection | None = None
    provenance: ProvenanceKind = ProvenanceKind.SOURCE_OBSERVED
    l0_effective_assumption: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_matching_configuration_type(self.domain, self.configuration)
        if self.correction is not None and self.correction.domain is not self.domain:
            raise ValueError(
                f"EffectiveDomainState.correction.domain ({self.correction.domain.value}) must match "
                f"EffectiveDomainState.domain ({self.domain.value})."
            )
        if self.source.domain is not self.domain:
            raise ValueError(
                f"EffectiveDomainState.source.domain ({self.source.domain.value}) must match "
                f"EffectiveDomainState.domain ({self.domain.value})."
            )
        object.__setattr__(
            self,
            "l0_effective_assumption",
            MappingProxyType(dict(self.l0_effective_assumption)),
        )


def resolve_effective_domain_state(
    source: DomainSourceState, correction: DomainCorrection | None = None
) -> EffectiveDomainState:
    """Sec 27: pure function, Source -> Effective Current. Never mutates
    `source` (it is a frozen dataclass parameter, never assigned to) --
    this function is the single place that turns an optional correction into
    the baseline every Domain Proposal is based on.
    """

    if correction is None:
        return EffectiveDomainState(
            domain=source.domain,
            configuration=source.configuration,
            source=source,
            correction=None,
            provenance=source.provenance,
            l0_effective_assumption={},
        )
    return EffectiveDomainState(
        domain=source.domain,
        configuration=correction.configuration,
        source=source,
        correction=correction,
        provenance=correction.provenance,
        l0_effective_assumption=correction.l0_effective_assumption,
    )


@dataclass(frozen=True)
class DomainProposalIdentity:
    """Sec 26 analog for domains: identity is independent of the proposal's
    visible label (`DomainProposal.label` below), and stable per (domain,
    proposal_id) -- e.g. domain=TRANSMISSION_DRIVELINE, proposal_id="TRANS-P01".
    """

    domain: DomainKind
    proposal_id: str

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise ValueError("DomainProposalIdentity.proposal_id is required.")


@dataclass(frozen=True)
class DomainProposal:
    """Sec 8/9: an alternative domain configuration based on Effective
    Current -- never on another Domain Proposal (REQ-11-007, enforced below
    both structurally, via `based_on`'s type, and defensively, via an
    explicit isinstance check). Reusable across multiple System Scenarios
    without mutation (INV-11-005) -- nothing here ties a DomainProposal to
    one particular System Scenario; it is a standalone, shareable object.

    `l0_effective_assumption` is the OPTIONAL, EXPLICIT quantitative part
    described in Sec 19 (e.g. `{"pse_percent_delta": 0.8}` for a "+0.8%"
    Transmission improvement) -- it must never be inferred from `configuration`
    (Sec 19: "It must not be inferred from gear count or final drive.").

    `technology_deltas` (Sprint 11B Sec 20) associates this proposal with
    existing canonical Technology Delta assumption(s), reusing the SAME
    `TechDeltaAssumption` dataclass Quick Scenario already uses
    (`src.vde_core.technology_delta.TechDeltaAssumption`) rather
    than a second schema -- preserving `affected_subsystem`/`effect_basis`/
    `effect_value`/`source_type`/`maturity_level`/`confidence` verbatim.
    This is association/storage only: this contract never stacks or
    combines these deltas (Sprint 11C's L0 adapter does that once, after the
    resolver establishes deterministic cross-domain order).
    Local order within one proposal's `technology_deltas` tuple is
    preserved as given (Sec 21) since it is a plain tuple, not a set/dict.

    `requested_changes` (Sprint 11B) is a small, explicit provenance record
    of exactly which `configuration` fields this proposal overrode relative
    to `based_on.configuration` -- populated by
    `domain_resolution.resolve_domain_proposal()` (the intended
    construction path), never inferred after the fact. A proposal built
    directly (bypassing that helper) may leave it empty; `changed_fields()`
    in `domain_resolution.py` computes the same information robustly by
    diffing `configuration` against `based_on.configuration` directly, so
    nothing downstream needs to trust `requested_changes` alone.
    """

    identity: DomainProposalIdentity
    domain: DomainKind
    configuration: DomainConfiguration
    based_on: EffectiveDomainState
    label: str | None = None
    l0_effective_assumption: Mapping[str, float] = field(default_factory=dict)
    l0_assumption_provenance: Mapping[str, ProvenanceKind] = field(default_factory=dict)
    technology_deltas: tuple[TechDeltaAssumption, ...] = ()
    requested_changes: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.identity.domain is not self.domain:
            raise ValueError(
                f"DomainProposal.identity.domain ({self.identity.domain.value}) must match "
                f"DomainProposal.domain ({self.domain.value})."
            )
        _require_matching_configuration_type(self.domain, self.configuration)
        if isinstance(self.based_on, DomainProposal):
            raise TypeError(
                "DomainProposal.based_on must be an EffectiveDomainState, never another "
                "DomainProposal (Sec 8: no Domain Proposal -> Domain Proposal lineage)."
            )
        if not isinstance(self.based_on, EffectiveDomainState):
            raise TypeError(
                f"DomainProposal.based_on must be an EffectiveDomainState, got {type(self.based_on).__name__}."
            )
        if self.based_on.domain is not self.domain:
            raise ValueError(
                f"DomainProposal.based_on.domain ({self.based_on.domain.value}) must match "
                f"DomainProposal.domain ({self.domain.value})."
            )
        object.__setattr__(
            self,
            "l0_effective_assumption",
            MappingProxyType(dict(self.l0_effective_assumption)),
        )
        unknown_provenance = set(self.l0_assumption_provenance) - set(self.l0_effective_assumption)
        if unknown_provenance:
            raise ValueError(
                "DomainProposal.l0_assumption_provenance contains keys without an L0 assumption: "
                + ", ".join(sorted(unknown_provenance))
            )
        object.__setattr__(
            self,
            "l0_assumption_provenance",
            MappingProxyType(dict(self.l0_assumption_provenance)),
        )
        object.__setattr__(self, "requested_changes", MappingProxyType(dict(self.requested_changes)))


# -----------------------------------------------------------------------------
# System Scenario (Sec 10/26): a composition of per-domain selections. Every
# domain slot holds EITHER that domain's EffectiveDomainState (no proposal
# selected -- "inherits Current for this domain") OR one DomainProposal for
# that domain. A slot never holds another System Scenario's selection.
# -----------------------------------------------------------------------------

DomainSelection = Union[EffectiveDomainState, DomainProposal]


class SystemScenarioRole(_TextEnum):
    CURRENT = "CURRENT"
    PROPOSAL = "PROPOSAL"


@dataclass(frozen=True)
class SystemScenarioIdentity:
    """Sec 26/REQ-11-022: identity is independent of the visible scenario
    name (`SystemScenarioDefinition.label`). `proposal_index` enforces
    "Current + max 3 Proposals" per identity; uniqueness of
    (role, proposal_index) WITHIN one working set of scenarios is a future
    11B/11C resolver/orchestrator concern, not a single-identity concern.
    """

    scenario_id: str
    role: SystemScenarioRole
    proposal_index: int | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("SystemScenarioIdentity.scenario_id is required.")
        if self.role is SystemScenarioRole.CURRENT and self.proposal_index is not None:
            raise ValueError("SystemScenarioIdentity.proposal_index must be None when role is CURRENT.")
        if self.role is SystemScenarioRole.PROPOSAL:
            if self.proposal_index is None or not (1 <= self.proposal_index <= MAX_SYSTEM_SCENARIO_PROPOSALS):
                raise ValueError(
                    "SystemScenarioIdentity.proposal_index must be between 1 and "
                    f"{MAX_SYSTEM_SCENARIO_PROPOSALS} (Sec 26: Current + max "
                    f"{MAX_SYSTEM_SCENARIO_PROPOSALS} proposals) when role is PROPOSAL, "
                    f"got {self.proposal_index!r}."
                )


@dataclass(frozen=True)
class SystemScenarioDefinition:
    """Sec 25: scenario composition/reference intent -- NOT a result. Every
    approved domain must be representable (REQ-11-004): `slots` should
    normally carry all 8 `DomainKind` entries, though this contract does not
    itself enforce completeness (a partially-specified scenario is still a
    valid *definition*; a future resolver decides readiness, Sec 24).

    A System Scenario's Vehicle Demand selection is just its VEHICLE_DEMAND
    slot like any other domain -- REQ-11-002/003 ("different System Scenarios
    may use different VDE/Vehicle Demand results") falls out naturally since
    each `SystemScenarioDefinition` owns its own independent `slots` mapping.
    """

    identity: SystemScenarioIdentity
    slots: Mapping[DomainKind, DomainSelection]
    label: str | None = None

    def __post_init__(self) -> None:
        for domain, selection in self.slots.items():
            if not isinstance(selection, (EffectiveDomainState, DomainProposal)):
                raise TypeError(
                    f"SystemScenarioDefinition.slots[{domain.value}] must be an EffectiveDomainState "
                    f"or DomainProposal, got {type(selection).__name__}."
                )
            if selection.domain is not domain:
                raise ValueError(
                    f"SystemScenarioDefinition.slots[{domain.value}] holds a selection for "
                    f"domain {selection.domain.value} instead."
                )
        object.__setattr__(self, "slots", MappingProxyType(dict(self.slots)))

    @property
    def vehicle_demand_selection(self) -> DomainSelection | None:
        return self.slots.get(DomainKind.VEHICLE_DEMAND)


# -----------------------------------------------------------------------------
# Resolution/result shell (Sec 25/28): ResolvedSystemScenario is the
# immutable/effective snapshot the Sprint 11C solver consumes; it must
# not query Streamlit/session state during calculation. SystemScenarioResult
# is the result+audit shell -- Sprint 11A defines the shape only, no
# resolver/solver wiring (that's 11B/11C).
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FidelityManifest:
    """Sec 23: explicit per-domain fidelity coverage. Domains absent from
    `per_domain` are treated as NOT_REPRESENTED by `fidelity_for` (never
    silently assumed QUANTITATIVE)."""

    per_domain: Mapping[DomainKind, FidelityLevel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_domain", MappingProxyType(dict(self.per_domain)))

    def fidelity_for(self, domain: DomainKind) -> FidelityLevel:
        return self.per_domain.get(domain, FidelityLevel.NOT_REPRESENTED)

    @property
    def is_configuration_only_everywhere_quantitative_is_absent(self) -> bool:
        """True when no domain claims QUANTITATIVE/EFFECTIVE_ASSUMPTION
        fidelity -- i.e. this manifest cannot currently justify a changed L0
        numeric result (Sec: "configuration-only is distinguishable from
        quantitative representation")."""

        return not any(
            level in (FidelityLevel.QUANTITATIVE, FidelityLevel.EFFECTIVE_ASSUMPTION)
            for level in self.per_domain.values()
        )


@dataclass(frozen=True)
class L0AssumptionContribution:
    """Audit record for one adopted direct ``FuelEstimateRequest`` input."""

    key: str
    value: float
    domain: DomainKind
    proposal_id: str
    provenance: ProvenanceKind = ProvenanceKind.ASSUMED


@dataclass(frozen=True)
class TechnologyDeltaContribution:
    """Domain/proposal provenance for one active canonical stack entry."""

    evaluation_order: int
    domain: DomainKind
    proposal_id: str
    assumption: TechDeltaAssumption
    quantitative_status: str = "applied"


@dataclass(frozen=True)
class ResolvedSystemScenario:
    """Sec 25: the immutable, effective snapshot a solver would consume.
    `resolved_domains` mirrors `SystemScenarioDefinition.slots` in fixed
    canonical domain order. Sprint 11C additionally records the concrete L0
    request, readiness, explicitly ordered Technology Deltas, and aggregate
    assumptions that produced the Fidelity Manifest. The historical 11A
    `resolve_system_scenario_shell` remains available only as a compatibility
    helper; new calculation paths use `resolver.resolve_system_scenario`.
    """

    identity: SystemScenarioIdentity
    resolved_domains: Mapping[DomainKind, DomainSelection]
    fidelity_manifest: FidelityManifest
    architecture_class: ArchitectureClass | None = None
    solver_readiness: SolverReadiness = SolverReadiness.NOT_READY
    l0_request_snapshot: object | None = None
    ordered_technology_deltas: tuple[TechDeltaAssumption, ...] = ()
    technology_delta_contributions: tuple[TechnologyDeltaContribution, ...] = ()
    l0_effective_assumptions: Mapping[str, float] = field(default_factory=dict)
    l0_assumption_contributions: tuple[L0AssumptionContribution, ...] = ()
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved_domains", MappingProxyType(dict(self.resolved_domains)))
        object.__setattr__(
            self,
            "l0_effective_assumptions",
            MappingProxyType(dict(self.l0_effective_assumptions)),
        )

    @property
    def fuel_estimate_request(self) -> object | None:
        """Compatibility view returning a fresh mutable canonical request.

        The resolved snapshot itself stores only the frozen adapter input, so
        callers cannot mutate a request that a later calculation will reuse.
        """

        factory = getattr(self.l0_request_snapshot, "to_request", None)
        return factory() if callable(factory) else None


@dataclass(frozen=True)
class SystemScenarioResult:
    """Sec 25/28: result + audit boundary populated by Sprint 11C's
    EnergyBalanceL0Adapter. The canonical FuelEstimateResult and optional
    canonical Technology Delta audit result remain separately inspectable;
    `effective_outputs` is the already-calculated future Comparison view.
    """

    identity: SystemScenarioIdentity
    resolved_scenario: ResolvedSystemScenario
    fuel_estimate_result: object | None = None
    technology_delta_result: Mapping[str, Any] | None = None
    selected_vehicle_demand_identity: str | None = None
    architecture_class: ArchitectureClass | None = None
    solver_identity: str | None = None
    model_identity: str | None = None
    readiness: SolverReadiness = SolverReadiness.NOT_READY
    fidelity_manifest: FidelityManifest | None = None
    effective_assumptions: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    contract_version: str = SYSTEM_SCENARIO_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_assumptions",
            MappingProxyType(dict(self.effective_assumptions)),
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def effective_outputs(self) -> Mapping[str, Any]:
        """Final L0 outputs, already calculated and Comparison-ready.

        Technology Delta's canonical proposal snapshot takes precedence
        when present; otherwise this exposes the canonical FuelEstimateResult
        fields. No formula or recalculation occurs here.
        """

        result = self.fuel_estimate_result
        if result is None:
            return {}
        pse_summary = dict((getattr(result, "assumptions", {}) or {}).get("pse_summary") or {})
        outputs = {
            "fuel_l_100km": getattr(result, "fuel_l_100km", None),
            "energy_Wh_km": getattr(result, "energy_Wh_km", None),
            "gco2_km": getattr(result, "gco2_km", None),
            "pse": pse_summary.get("value"),
            "fuel_consumed_mj_per_km": pse_summary.get("fuel_consumed_mj_per_km"),
            "electric_consumed_mj_per_km": pse_summary.get("electric_consumed_mj_per_km"),
            "total_consumed_mj_per_km": pse_summary.get("total_consumed_mj_per_km"),
            "method": getattr(result, "method", None),
            "confidence": getattr(result, "confidence", None),
        }
        if self.technology_delta_result is not None:
            proposal = self.technology_delta_result.get("proposal")
            if proposal is not None:
                outputs.update(dict(proposal))
        return outputs


def resolve_system_scenario_shell(definition: SystemScenarioDefinition) -> ResolvedSystemScenario:
    """Sec 25/28: the minimal, non-physics resolution shell Sprint 11A
    provides. Copies `definition.slots` verbatim (no solver, no readiness
    logic, no Technology Delta composition) and assigns a conservative
    Fidelity Manifest: VEHICLE_DEMAND is QUANTITATIVE when present (it is
    already a real, resolved Sprint 9 physics result); every other populated
    domain is CONFIGURATION_ONLY (Sprint 11A wires no L0 wiring for them
    yet); unpopulated domains are NOT_REPRESENTED. `architecture_class` is
    read straight off the ARCHITECTURE slot's configuration when present.
    Kept for backward compatibility with 11A/11B callers and tests. Sprint
    11C calculation paths use `resolver.resolve_system_scenario`, which adds
    real solver-readiness, request construction, and deterministic delta
    composition; this shell is not the final resolver.
    """

    fidelity: dict[DomainKind, FidelityLevel] = {}
    for domain in ALL_DOMAIN_KINDS:
        if domain not in definition.slots:
            fidelity[domain] = FidelityLevel.NOT_REPRESENTED
        elif domain is DomainKind.VEHICLE_DEMAND:
            selection = definition.slots[domain]
            has_result = isinstance(selection.configuration, VehicleDemandConfiguration) and (
                selection.configuration.vehicle_demand_result is not None
            )
            fidelity[domain] = FidelityLevel.QUANTITATIVE if has_result else FidelityLevel.NOT_REPRESENTED
        else:
            fidelity[domain] = FidelityLevel.CONFIGURATION_ONLY

    architecture_class: ArchitectureClass | None = None
    architecture_selection = definition.slots.get(DomainKind.ARCHITECTURE)
    if architecture_selection is not None and isinstance(
        architecture_selection.configuration, ArchitectureConfiguration
    ):
        architecture_class = architecture_selection.configuration.architecture_class

    return ResolvedSystemScenario(
        identity=definition.identity,
        resolved_domains=dict(definition.slots),
        fidelity_manifest=FidelityManifest(per_domain=fidelity),
        architecture_class=architecture_class,
    )


__all__ = [
    "SYSTEM_SCENARIO_CONTRACT_VERSION",
    "MAX_SYSTEM_SCENARIO_PROPOSALS",
    "MAX_DOMAIN_PROPOSALS_PER_DOMAIN",
    "DomainKind",
    "ALL_DOMAIN_KINDS",
    "ArchitectureClass",
    "DomainApplicability",
    "domain_applicability_for",
    "FidelityLevel",
    "SolverReadiness",
    "ProvenanceKind",
    "VehicleDemandConfiguration",
    "ArchitectureConfiguration",
    "EngineConfiguration",
    "TransmissionConfiguration",
    "ElectricDriveConfiguration",
    "EnergyStorageConfiguration",
    "ControlsConfiguration",
    "AuxThermalConfiguration",
    "DomainConfiguration",
    "configuration_type_for",
    "domain_typically_applicable",
    "DomainSourceState",
    "DomainCorrection",
    "EffectiveDomainState",
    "resolve_effective_domain_state",
    "DomainProposalIdentity",
    "DomainProposal",
    "DomainSelection",
    "SystemScenarioRole",
    "SystemScenarioIdentity",
    "SystemScenarioDefinition",
    "FidelityManifest",
    "L0AssumptionContribution",
    "TechnologyDeltaContribution",
    "ResolvedSystemScenario",
    "SystemScenarioResult",
    "resolve_system_scenario_shell",
]
