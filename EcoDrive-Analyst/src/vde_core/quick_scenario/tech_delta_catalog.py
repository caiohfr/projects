# src/vde_core/quick_scenario/tech_delta_catalog.py
# -----------------------------------------------------------------------------
# Sprint 10D - a tiny, synthetic Technology Delta preset catalog (Sec 16).
#
# The CSV is a preset/reference convenience layer only -- it maps directly
# onto the existing TechDeltaAssumption contract (the same canonical
# Technology Delta vocabulary as src.vde_core.technology_delta), never a
# second schema. No admin UI, no editing, no database: rows are read-only
# reference examples a later UI package can offer as one-click presets, or
# a caller can start from and override.
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
from pathlib import Path

from .contracts import TechDeltaAssumption

DEFAULT_QUICK_TECH_DELTA_CATALOG_PATH = Path("data/quick_tech_deltas.csv")


def load_quick_tech_delta_catalog(
    path: str | Path | None = None,
) -> dict[str, TechDeltaAssumption]:
    """Read the Quick Tech Delta preset catalog into `{tech_id: TechDeltaAssumption}`.

    Never mutates the source file. Returns an empty mapping if the catalog
    file is absent -- a missing optional preset catalog is not a fatal
    error for Quick Scenario, which can always work from a caller-supplied
    custom `TechDeltaAssumption` instead (Sec 17).
    """

    catalog_path = Path(path) if path is not None else DEFAULT_QUICK_TECH_DELTA_CATALOG_PATH
    if not catalog_path.exists():
        return {}

    catalog: dict[str, TechDeltaAssumption] = {}
    with catalog_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tech_id = (row.get("tech_id") or "").strip()
            if not tech_id:
                continue
            catalog[tech_id] = TechDeltaAssumption(
                name=row["technology"],
                effect_basis=row["effect_basis"],
                effect_value=float(row["default_value"]),
                affected_subsystem=row.get("subsystem") or "whole powertrain",
                source_type=row.get("source_type") or "manual",
                maturity_level=row.get("maturity_level") or "engineering_assumption",
                confidence=row.get("confidence") or "unknown",
                notes=row.get("notes") or "",
            )
    return catalog


__all__ = ["DEFAULT_QUICK_TECH_DELTA_CATALOG_PATH", "load_quick_tech_delta_catalog"]
