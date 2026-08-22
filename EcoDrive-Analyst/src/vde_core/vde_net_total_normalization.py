# src/vde_core/vde_net_total_normalization.py
# -----------------------------------------------------------------------------
# One-time, idempotent, auditable normalization of vde_db TOTAL/NET fields
# (Package 7G). See src/vde_core/vde_net_total_contract.py for the semantic
# rule this reuses.
#
# preview_vde_net_total_normalization() is read-only. apply_vde_net_total_
# normalization() only ever:
#   - moves a LEGACY_TOTAL_IN_NET_FIELD row's vde_net_mj_per_km value into
#     vde_total_mj_per_km and nulls vde_net_mj_per_km, logging the change;
#   - flags an AMBIGUOUS_REVIEW row's review_status as REVIEW_REQUIRED
#     without ever touching its total/net values.
# CANONICAL_* and INVALID rows are never written.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from src.vde_core import db as db_module
from src.vde_core.data_change_log_repository import append_change_log
from src.vde_core.database_management_contract import (
    ActorContext,
    ChangePreview,
)
from src.vde_core.vde_net_total_contract import VdeSemanticStatus, classify_vde_row

_VDE_ROW_COLUMNS = (
    "id",
    "vde_total_mj_per_km",
    "vde_net_mj_per_km",
    "record_origin",
    "review_status",
)


@dataclass(frozen=True)
class RowChange:
    vde_id: int
    old_total: float | None
    old_net: float | None
    new_total: float | None
    new_net: float | None
    reason: str


@dataclass(frozen=True)
class NormalizationPreview:
    total_rows_inspected: int
    counts_by_status: dict[str, int]
    legacy_total_in_net_changes: tuple[RowChange, ...] = field(default_factory=tuple)
    ambiguous_review_ids: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NormalizationResult:
    rows_normalized: int
    rows_flagged_for_review: int
    change_log_ids: tuple[int, ...] = field(default_factory=tuple)


def preview_vde_net_total_normalization(db_path=None) -> NormalizationPreview:
    """Read-only: classify every vde_db row and propose legacy TOTAL/NET fixes."""
    if db_path is not None:
        with db_module.using_db_path(db_path):
            return _preview()
    return _preview()


def _preview() -> NormalizationPreview:
    db_module.ensure_db()
    rows = db_module.fetchall(f"SELECT {', '.join(_VDE_ROW_COLUMNS)} FROM vde_db")

    counts: dict[str, int] = {status.value: 0 for status in VdeSemanticStatus}
    changes: list[RowChange] = []
    ambiguous_ids: list[int] = []

    for row in rows:
        status = classify_vde_row(row)
        counts[status.value] += 1
        if status is VdeSemanticStatus.LEGACY_TOTAL_IN_NET_FIELD:
            old_net = row["vde_net_mj_per_km"]
            changes.append(
                RowChange(
                    vde_id=row["id"],
                    old_total=row["vde_total_mj_per_km"],
                    old_net=old_net,
                    new_total=old_net,
                    new_net=None,
                    reason="package_7g_legacy_total_stored_in_net_field",
                )
            )
        elif status is VdeSemanticStatus.AMBIGUOUS_REVIEW:
            ambiguous_ids.append(row["id"])

    return NormalizationPreview(
        total_rows_inspected=len(rows),
        counts_by_status=counts,
        legacy_total_in_net_changes=tuple(changes),
        ambiguous_review_ids=tuple(ambiguous_ids),
    )


def apply_vde_net_total_normalization(
    preview: NormalizationPreview,
    actor_context: ActorContext,
    *,
    reason: str,
    db_path=None,
) -> NormalizationResult:
    if db_path is not None:
        with db_module.using_db_path(db_path):
            return _apply(preview, actor_context, reason=reason)
    return _apply(preview, actor_context, reason=reason)


def _apply(
    preview: NormalizationPreview,
    actor_context: ActorContext,
    *,
    reason: str,
) -> NormalizationResult:
    db_module.ensure_db()
    change_log_ids: list[int] = []

    with db_module._con() as con:
        for change in preview.legacy_total_in_net_changes:
            con.execute(
                "UPDATE vde_db SET vde_total_mj_per_km=?, vde_net_mj_per_km=NULL WHERE id=?",
                (change.new_total, change.vde_id),
            )
            change_preview = ChangePreview(
                operation_id=str(uuid4()),
                entity_type="VDE",
                record_id=str(change.vde_id),
                action="UPDATE",
                normalized_payload={},
                field_diff=(),
                validation_issues=(),
                can_commit=True,
            )
            change_log_id = append_change_log(
                change_preview,
                actor_context,
                reason=reason,
                before={
                    "vde_total_mj_per_km": change.old_total,
                    "vde_net_mj_per_km": change.old_net,
                },
                after={
                    "vde_total_mj_per_km": change.new_total,
                    "vde_net_mj_per_km": change.new_net,
                },
                impact={"classification": change.reason},
                connection=con,
            )
            change_log_ids.append(change_log_id)

        for vde_id in preview.ambiguous_review_ids:
            con.execute(
                "UPDATE vde_db SET review_status='REVIEW_REQUIRED' WHERE id=?",
                (vde_id,),
            )
        con.commit()

    return NormalizationResult(
        rows_normalized=len(preview.legacy_total_in_net_changes),
        rows_flagged_for_review=len(preview.ambiguous_review_ids),
        change_log_ids=tuple(change_log_ids),
    )


__all__ = [
    "RowChange",
    "NormalizationPreview",
    "NormalizationResult",
    "preview_vde_net_total_normalization",
    "apply_vde_net_total_normalization",
]
