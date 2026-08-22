# src/vde_core/vde_net_total_contract.py
# -----------------------------------------------------------------------------
# Canonical TOTAL/NET semantic contract for vde_db rows (Package 7G).
#
# Historical context: every pre-Sprint-7 ("LEGACY") row in vde_db has
# vde_net_mj_per_km populated and vde_total_mj_per_km NULL. The only code
# paths that ever wrote vde_net_mj_per_km for that data (the legacy EPA ETL
# notebook and the legacy vde_setup_service rich-save path) computed it from
# the full/TOTAL coastdown roadload (coast_A_N/coast_B_N_per_kph/
# coast_C_N_per_kph2) with no transmission-loss subtraction anywhere in the
# call chain. Under the current contract (NET = TOTAL minus resolved
# transmission losses, docs/VDE_SETUP_V22_FINAL_CHECKPOINT.md), that stored
# value is physically VDE_TOTAL, not VDE_NET.
#
# This module is the single place that knows about that historical mismatch.
# Callers outside this module must never special-case "old NET actually means
# TOTAL" themselves -- they call canonical_vde_read() and get values that are
# already correct for the current contract.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class _TextEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class VdeSemanticStatus(_TextEnum):
    CANONICAL_TOTAL_ONLY = "CANONICAL_TOTAL_ONLY"
    CANONICAL_TOTAL_AND_NET = "CANONICAL_TOTAL_AND_NET"
    LEGACY_TOTAL_IN_NET_FIELD = "LEGACY_TOTAL_IN_NET_FIELD"
    AMBIGUOUS_REVIEW = "AMBIGUOUS_REVIEW"
    INVALID = "INVALID"


# Origins for which the codebase has no evidence of how vde_net_mj_per_km was
# derived when vde_total_mj_per_km is missing. A net-only row under one of
# these origins is never auto-classified as a legacy TOTAL/NET swap.
_ORIGINS_WITHOUT_LEGACY_EVIDENCE = frozenset({"MANUAL", "IMPORTED_REFERENCE", "VDE_SETUP"})


def _present(value: Any) -> bool:
    return value is not None


def classify_vde_row(row: Mapping[str, Any]) -> VdeSemanticStatus:
    """Classify a vde_db row's TOTAL/NET population without mutating it.

    Deterministic only: a row is reclassified as a legacy TOTAL-in-NET swap
    solely when record_origin == "LEGACY", which is the only origin this
    codebase has proven evidence for (see module docstring). Every other
    net-only row is AMBIGUOUS_REVIEW rather than guessed.
    """
    total = row.get("vde_total_mj_per_km")
    net = row.get("vde_net_mj_per_km")
    has_total = _present(total)
    has_net = _present(net)

    if has_total and has_net:
        return VdeSemanticStatus.CANONICAL_TOTAL_AND_NET
    if has_total and not has_net:
        return VdeSemanticStatus.CANONICAL_TOTAL_ONLY
    if not has_total and not has_net:
        return VdeSemanticStatus.INVALID

    # not has_total and has_net
    origin = str(row.get("record_origin") or "").strip().upper()
    if origin == "LEGACY":
        return VdeSemanticStatus.LEGACY_TOTAL_IN_NET_FIELD
    return VdeSemanticStatus.AMBIGUOUS_REVIEW


@dataclass(frozen=True)
class CanonicalVde:
    total_mj_per_km: float | None
    net_mj_per_km: float | None
    net_available: bool
    status: VdeSemanticStatus


def canonical_vde_read(row: Mapping[str, Any]) -> CanonicalVde:
    """The clean TOTAL/NET read contract for downstream consumers (Sprint 8).

    Always routes through classify_vde_row(). NET is only ever surfaced when
    the row is CANONICAL_TOTAL_AND_NET -- a not-yet-normalized or newly
    reintroduced legacy-shaped row can never leak a mislabeled NET, whether
    or not the one-time normalization (vde_net_total_normalization.py) has
    run against this particular database.
    """
    status = classify_vde_row(row)
    total = row.get("vde_total_mj_per_km")
    if status is VdeSemanticStatus.CANONICAL_TOTAL_AND_NET:
        net = row.get("vde_net_mj_per_km")
        net_available = True
    else:
        net = None
        net_available = False
    return CanonicalVde(
        total_mj_per_km=total,
        net_mj_per_km=net,
        net_available=net_available,
        status=status,
    )


__all__ = [
    "VdeSemanticStatus",
    "CanonicalVde",
    "classify_vde_row",
    "canonical_vde_read",
]
