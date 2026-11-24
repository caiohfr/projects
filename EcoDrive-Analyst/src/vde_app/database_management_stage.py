"""Session-stage safety helpers for Database Management.

This module owns no persistence or engineering calculation.  It binds a UI
stage to one canonical database identity and coordinates the documented
sequential commit/recovery contract around the existing per-operation services.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4


logger = logging.getLogger(__name__)


class StageDatabaseIdentityError(RuntimeError):
    """Raised when a stage is missing or mismatched with the active database."""


@dataclass(frozen=True)
class CommitEntry:
    stage_operation_id: str
    entity_type: str
    action: str
    record_id: str | None
    status: str
    change_log_id: int | None = None
    affected_record_ids: tuple[str, ...] = ()
    recovered_from_receipt: bool = False
    message: str | None = None


@dataclass(frozen=True)
class SequentialCommitOutcome:
    status: str
    entries: tuple[CommitEntry, ...]
    remaining_commands: tuple[dict[str, Any], ...]
    failure_index: int | None = None
    failure_message: str | None = None
    uncertain: bool = False

    @property
    def committed_count(self) -> int:
        return sum(entry.status == "COMMITTED" for entry in self.entries)


def canonical_database_identity(path_like: str | os.PathLike[str]) -> str:
    """Return a comparable absolute identity for the actual SQLite path."""

    resolved = Path(path_like).expanduser().resolve(strict=False)
    return os.path.normcase(os.path.abspath(str(resolved)))


def stage_operation_identity(command: dict[str, Any]) -> tuple[Any, ...]:
    if command.get("import_batch_id"):
        return (
            "IMPORT",
            command.get("import_batch_id"),
            command.get("import_sheet"),
            command.get("import_row"),
        )
    return (command.get("entity_type"), command.get("action"), command.get("record_id"))


def add_stage_command(
    commands: Iterable[dict[str, Any]],
    *,
    stage_database_identity: str | None,
    active_database_identity: str,
    command: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Add/merge one command while refusing cross-database stage mixing."""

    staged = [deepcopy(item) for item in commands]
    active = canonical_database_identity(active_database_identity)
    origin = canonical_database_identity(stage_database_identity) if stage_database_identity else None
    if staged and not origin:
        raise StageDatabaseIdentityError(
            "The existing stage has no database identity. Discard it explicitly before staging or committing."
        )
    if origin and origin != active:
        raise StageDatabaseIdentityError(
            f"Pending changes belong to {origin}; the active database is {active}. "
            "Return to the stage database or discard the stage explicitly."
        )
    origin = origin or active
    incoming = deepcopy(command)
    incoming.setdefault("stage_operation_id", str(uuid4()))
    identity = stage_operation_identity(incoming)
    for index, existing in enumerate(staged):
        if stage_operation_identity(existing) != identity:
            continue
        merged = dict(existing)
        merged.update({key: deepcopy(value) for key, value in incoming.items() if key != "payload"})
        merged["stage_operation_id"] = existing.get("stage_operation_id") or incoming["stage_operation_id"]
        merged["payload"] = {**dict(existing.get("payload") or {}), **dict(incoming.get("payload") or {})}
        staged[index] = merged
        return staged, origin
    staged.append(incoming)
    return staged, origin


def assert_stage_database_identity(
    *,
    commands: Iterable[dict[str, Any]],
    stage_database_identity: str | None,
    active_database_identity: str,
) -> str:
    staged = list(commands)
    if not staged:
        return canonical_database_identity(active_database_identity)
    if not stage_database_identity:
        raise StageDatabaseIdentityError(
            "Pending changes have no database identity and cannot be previewed or committed."
        )
    origin = canonical_database_identity(stage_database_identity)
    active = canonical_database_identity(active_database_identity)
    if origin != active:
        raise StageDatabaseIdentityError(
            f"Pending changes belong to {origin}; the active database is {active}. "
            "No preview or write is allowed on this database."
        )
    return origin


def execute_sequential_commit(
    commands: Iterable[dict[str, Any]],
    *,
    stage_database_identity: str,
    active_database_identity: Callable[[], str],
    apply_one: Callable[[int, dict[str, Any]], Any],
    receipt_lookup: Callable[[str], dict[str, Any] | None],
    on_confirmed_progress: Callable[[tuple[dict[str, Any], ...], CommitEntry], None] | None = None,
) -> SequentialCommitOutcome:
    """Apply commands sequentially and remove only receipt-confirmed successes.

    Every existing persistence service remains the transaction owner for one
    operation.  This coordinator never claims batch atomicity.  It stops on the
    first unconfirmed failure and leaves that command plus all later commands
    pending.  A pre-existing receipt reconciles a prior crash and prevents a
    CREATE/import replay.
    """

    remaining = [deepcopy(item) for item in commands]
    entries: list[CommitEntry] = []
    original_index = 0
    while remaining:
        command = remaining[0]
        operation_id = str(command.get("stage_operation_id") or "").strip()
        if not operation_id:
            return SequentialCommitOutcome(
                status="PARTIAL" if entries else "BLOCKED",
                entries=tuple(entries),
                remaining_commands=tuple(remaining),
                failure_index=original_index,
                failure_message="A staged operation has no stable operation ID; explicit reconciliation is required.",
                uncertain=True,
            )
        try:
            assert_stage_database_identity(
                commands=remaining,
                stage_database_identity=stage_database_identity,
                active_database_identity=active_database_identity(),
            )
        except StageDatabaseIdentityError as exc:
            return SequentialCommitOutcome(
                status="PARTIAL" if entries else "BLOCKED",
                entries=tuple(entries),
                remaining_commands=tuple(remaining),
                failure_index=original_index,
                failure_message=str(exc),
            )

        try:
            receipt = receipt_lookup(operation_id)
        except Exception as exc:
            logger.exception(
                "Database Management receipt lookup failed for staged operation %s",
                operation_id,
            )
            return SequentialCommitOutcome(
                status="PARTIAL" if entries else "BLOCKED",
                entries=tuple(entries),
                remaining_commands=tuple(remaining),
                failure_index=original_index,
                failure_message=f"Receipt lookup failed: {type(exc).__name__}: {exc}",
                uncertain=True,
            )
        if receipt:
            entry = _receipt_entry(command, receipt, recovered=True)
            remaining.pop(0)
            entries.append(entry)
            if on_confirmed_progress:
                on_confirmed_progress(tuple(deepcopy(remaining)), entry)
            original_index += 1
            continue

        try:
            result = apply_one(original_index, command)
        except Exception as exc:  # the UI reports the exact technical exception
            logger.exception(
                "Database Management commit failed for staged operation %s",
                operation_id,
            )
            try:
                receipt = receipt_lookup(operation_id)
            except Exception as receipt_exc:
                logger.exception(
                    "Database Management reconciliation lookup failed for staged operation %s",
                    operation_id,
                )
                return SequentialCommitOutcome(
                    status="PARTIAL" if entries else "BLOCKED",
                    entries=tuple(entries),
                    remaining_commands=tuple(remaining),
                    failure_index=original_index,
                    failure_message=(
                        f"Commit raised {type(exc).__name__}: {exc}; receipt lookup then raised "
                        f"{type(receipt_exc).__name__}: {receipt_exc}."
                    ),
                    uncertain=True,
                )
            if receipt:
                entry = _receipt_entry(command, receipt, recovered=True, message=str(exc))
                remaining.pop(0)
                entries.append(entry)
                if on_confirmed_progress:
                    on_confirmed_progress(tuple(deepcopy(remaining)), entry)
                return SequentialCommitOutcome(
                    status="PARTIAL" if remaining else "COMPLETE",
                    entries=tuple(entries),
                    remaining_commands=tuple(remaining),
                    failure_index=original_index,
                    failure_message=(
                        "The operation raised an exception after a matching receipt was found. "
                        "It was reconciled as committed; later operations were not attempted."
                    ),
                )
            return SequentialCommitOutcome(
                status="PARTIAL" if entries else "FAILED",
                entries=tuple(entries),
                remaining_commands=tuple(remaining),
                failure_index=original_index,
                failure_message=f"{type(exc).__name__}: {exc}",
            )

        if not bool(getattr(result, "committed", False)):
            return SequentialCommitOutcome(
                status="PARTIAL" if entries else "BLOCKED",
                entries=tuple(entries),
                remaining_commands=tuple(remaining),
                failure_index=original_index,
                failure_message="The persistence service returned without confirmed commit status.",
                uncertain=True,
            )

        entry = CommitEntry(
            stage_operation_id=operation_id,
            entity_type=str(command.get("entity_type") or ""),
            action=str(command.get("action") or ""),
            record_id=None if command.get("record_id") is None else str(command.get("record_id")),
            status="COMMITTED",
            change_log_id=getattr(result, "change_log_id", None),
            affected_record_ids=tuple(str(item) for item in (getattr(result, "affected_record_ids", ()) or ())),
        )
        remaining.pop(0)
        entries.append(entry)
        if on_confirmed_progress:
            on_confirmed_progress(tuple(deepcopy(remaining)), entry)
        original_index += 1

    return SequentialCommitOutcome(
        status="COMPLETE",
        entries=tuple(entries),
        remaining_commands=(),
    )


def _receipt_entry(
    command: dict[str, Any],
    receipt: dict[str, Any],
    *,
    recovered: bool,
    message: str | None = None,
) -> CommitEntry:
    record_id = receipt.get("record_id", command.get("record_id"))
    return CommitEntry(
        stage_operation_id=str(command.get("stage_operation_id") or ""),
        entity_type=str(receipt.get("entity_type") or command.get("entity_type") or ""),
        action=str(receipt.get("action") or command.get("action") or ""),
        record_id=None if record_id is None else str(record_id),
        status="COMMITTED",
        change_log_id=receipt.get("id"),
        affected_record_ids=() if record_id is None else (str(record_id),),
        recovered_from_receipt=recovered,
        message=message,
    )


__all__ = [
    "CommitEntry",
    "SequentialCommitOutcome",
    "StageDatabaseIdentityError",
    "add_stage_command",
    "assert_stage_database_identity",
    "canonical_database_identity",
    "execute_sequential_commit",
    "stage_operation_identity",
]
