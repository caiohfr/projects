#!/usr/bin/env python3
"""Read-only verification for the Sprint 12 synthetic component import."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

try:
    from scripts.import_synthetic_components import (
        CANONICAL_SOURCE_NAME,
        EXPECTED_CANONICAL_DOMAIN_COUNTS,
        EXPECTED_SOURCE_DOMAIN_COUNTS,
        EXPECTED_TOTAL,
        KPI_COLUMNS,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from import_synthetic_components import (  # type: ignore[no-redef]
        CANONICAL_SOURCE_NAME,
        EXPECTED_CANONICAL_DOMAIN_COUNTS,
        EXPECTED_SOURCE_DOMAIN_COUNTS,
        EXPECTED_TOTAL,
        KPI_COLUMNS,
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def verify(db_path: Path) -> dict[str, object]:
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        component_rows = conn.execute(
            "SELECT * FROM component_db WHERE source_name=? ORDER BY component_id",
            (CANONICAL_SOURCE_NAME,),
        ).fetchall()
        resolution_rows = conn.execute(
            "SELECT * FROM component_resolution ORDER BY component_resolution_id"
        ).fetchall()
        synthetic_resolutions = []
        for row in resolution_rows:
            provenance = json.loads(row["provenance_json"] or "{}")
            if provenance.get("synthetic_reference") is True:
                synthetic_resolutions.append((row, provenance))

        component_domain_counts = Counter(row["component_domain"] for row in component_rows)
        source_domain_counts: Counter[str] = Counter()
        component_link_count = 0
        for row in component_rows:
            provenance = json.loads(row["provenance_json"] or "{}")
            source_domain_counts[provenance.get("seed_component_domain_original")] += 1
            properties = json.loads(row["custom_properties_json"] or "{}")
            component_link_count += len(properties.get("synthetic_reference_resolution_ids") or [])

        boundary_counts = Counter(row["boundary"] for row, _ in synthetic_resolutions)
        preserved_labels = sum(
            provenance.get("seed_confidence_original") == "REFERENCE"
            and provenance.get("seed_fidelity_level_original") == "SYNTHETIC_POPULATION"
            for _, provenance in synthetic_resolutions
        )
        null_enum_rows = sum(
            row["confidence"] is None and row["fidelity_level"] is None
            for row, _ in synthetic_resolutions
        )
        matched_resolution_links = sum(
            bool(provenance.get("synthetic_reference_component_id"))
            for _, provenance in synthetic_resolutions
        )
        kpi_present = set(KPI_COLUMNS) & _columns(conn, "component_resolution")
        rejected_adoptions = conn.execute(
            """
            SELECT COUNT(*)
            FROM vde_component_resolution AS link
            JOIN component_resolution AS resolution
              ON resolution.component_resolution_id=link.component_resolution_id
            WHERE UPPER(COALESCE(resolution.estimate_status,''))
                  IN ('REJECTED_PRIMARY_MODEL','NOT_IDENTIFIABLE','REJECTED_MODEL')
            """
        ).fetchone()[0]
        report: dict[str, object] = {
            "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_issues": len(conn.execute("PRAGMA foreign_key_check").fetchall()),
            "kpi_columns_present": sorted(kpi_present),
            "kpi_columns_missing": sorted(set(KPI_COLUMNS) - kpi_present),
            "synthetic_component_count": len(component_rows),
            "synthetic_component_domain_counts": dict(component_domain_counts),
            "preserved_source_domain_counts": dict(source_domain_counts),
            "synthetic_resolution_count": len(synthetic_resolutions),
            "synthetic_resolution_boundary_counts": dict(boundary_counts),
            "resolution_rows_with_null_canonical_enums": null_enum_rows,
            "resolution_rows_with_preserved_seed_labels": preserved_labels,
            "component_manifest_link_count": component_link_count,
            "resolution_manifest_link_count": matched_resolution_links,
            "rejected_or_ni_adoptions": rejected_adoptions,
        }
        expected_boundary_counts = EXPECTED_SOURCE_DOMAIN_COUNTS
        report["ok"] = all(
            (
                report["quick_check"] == "ok",
                report["foreign_key_issues"] == 0,
                not report["kpi_columns_missing"],
                report["synthetic_component_count"] == EXPECTED_TOTAL,
                report["synthetic_component_domain_counts"] == EXPECTED_CANONICAL_DOMAIN_COUNTS,
                report["preserved_source_domain_counts"] == EXPECTED_SOURCE_DOMAIN_COUNTS,
                report["synthetic_resolution_count"] == EXPECTED_TOTAL,
                report["synthetic_resolution_boundary_counts"] == expected_boundary_counts,
                null_enum_rows == EXPECTED_TOTAL,
                preserved_labels == EXPECTED_TOTAL,
                component_link_count == EXPECTED_TOTAL,
                matched_resolution_links == EXPECTED_TOTAL,
                rejected_adoptions == 0,
            )
        )
        return report
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.db)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
