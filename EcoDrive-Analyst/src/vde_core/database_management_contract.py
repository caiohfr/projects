from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class _TextEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class EntityType(_TextEnum):
    VDE = "VDE"
    FUEL_CONSUMPTION = "FUEL_CONSUMPTION"
    TIRE = "TIRE"
    COMPONENT = "COMPONENT"


class RecordStatus(_TextEnum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class ReviewStatus(_TextEnum):
    CURRENT = "CURRENT"
    STALE_VDE = "STALE_VDE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ChangeAction(_TextEnum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DUPLICATE = "DUPLICATE"
    RECALCULATE_UPDATE = "RECALCULATE_UPDATE"
    RECALCULATE_NEW = "RECALCULATE_NEW"
    ARCHIVE = "ARCHIVE"
    RESTORE = "RESTORE"
    DELETE = "DELETE"
    REASSIGN_RELATIONSHIP = "REASSIGN_RELATIONSHIP"


class ImpactPersistenceChoice(_TextEnum):
    KEEP_EXISTING = "KEEP_EXISTING"
    RECALCULATE_UPDATE = "RECALCULATE_UPDATE"
    RECALCULATE_NEW = "RECALCULATE_NEW"


VDE_ORIGINS = frozenset({"IMPORTED_REFERENCE", "VDE_SETUP", "MANUAL", "LEGACY"})
FUEL_ORIGINS = frozenset({"HOMOLOGATED", "MEASURED", "ESTIMATED", "POWERTRAIN_L0", "LEGACY"})
TIRE_COMPONENT_ORIGINS = frozenset({"IMPORTED", "MANUAL", "QA", "LEGACY"})

ORIGINS_BY_ENTITY = {
    EntityType.VDE: VDE_ORIGINS,
    EntityType.FUEL_CONSUMPTION: FUEL_ORIGINS,
    EntityType.TIRE: TIRE_COMPONENT_ORIGINS,
    EntityType.COMPONENT: TIRE_COMPONENT_ORIGINS,
}


@dataclass(frozen=True)
class ActorContext:
    actor_id: str
    actor_role: str


LOCAL_ADMIN_ACTOR = ActorContext(actor_id="local_admin", actor_role="admin")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    field: str | None = None


@dataclass(frozen=True)
class FieldDiff:
    field: str
    before: Any
    after: Any


@dataclass(frozen=True)
class ChangeCommand:
    entity_type: EntityType | str
    action: ChangeAction | str
    payload: dict[str, Any] = field(default_factory=dict)
    current_record: dict[str, Any] = field(default_factory=dict)
    record_id: str | int | None = None
    record_origin: str | None = None
    reason: str | None = None
    physics_action: str | None = None


@dataclass(frozen=True)
class ChangePreview:
    operation_id: str
    entity_type: str
    record_id: str | None
    action: str
    normalized_payload: dict[str, Any]
    field_diff: tuple[FieldDiff, ...]
    validation_issues: tuple[ValidationIssue, ...]
    dependencies: tuple[dict[str, Any], ...] = ()
    physics_action: str | None = None
    rows_to_create: tuple[dict[str, Any], ...] = ()
    rows_to_update: tuple[dict[str, Any], ...] = ()
    rows_to_archive: tuple[dict[str, Any], ...] = ()
    rows_to_delete: tuple[dict[str, Any], ...] = ()
    stale_fuel_rows: tuple[dict[str, Any], ...] = ()
    requires_reason: bool = False
    can_commit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChangeResult:
    operation_id: str
    entity_type: str
    action: str
    committed: bool
    change_log_id: int | None = None
    affected_record_ids: tuple[str, ...] = ()
    request_history_ids: tuple[int, ...] = ()
    stale_fuel_row_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ImpactPreview:
    operation_id: str
    change_operation_id: str
    entity_type: str
    record_id: str
    persistence_choice: str
    usages: tuple[dict[str, Any], ...] = ()
    historical_usages: tuple[dict[str, Any], ...] = ()
    request_recalculations: tuple[dict[str, Any], ...] = ()
    affected_vde_ids: tuple[int, ...] = ()
    stale_fuel_rows: tuple[dict[str, Any], ...] = ()
    review_required: tuple[dict[str, Any], ...] = ()
    validation_issues: tuple[ValidationIssue, ...] = ()
    can_commit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_entity_type(value: EntityType | str) -> EntityType:
    if isinstance(value, EntityType):
        return value
    return EntityType(str(value or "").strip().upper())


def normalize_change_action(value: ChangeAction | str) -> ChangeAction:
    if isinstance(value, ChangeAction):
        return value
    return ChangeAction(str(value or "").strip().upper())


def normalize_impact_persistence_choice(
    value: ImpactPersistenceChoice | str,
) -> ImpactPersistenceChoice:
    if isinstance(value, ImpactPersistenceChoice):
        return value
    return ImpactPersistenceChoice(str(value or "").strip().upper())


def normalize_record_origin(entity_type: EntityType, value: str | None) -> str:
    origin = str(value or "LEGACY").strip().upper() or "LEGACY"
    if origin not in ORIGINS_BY_ENTITY[entity_type]:
        allowed = ", ".join(sorted(ORIGINS_BY_ENTITY[entity_type]))
        raise ValueError(f"Invalid origin {origin!r} for {entity_type.value}. Allowed: {allowed}.")
    return origin
