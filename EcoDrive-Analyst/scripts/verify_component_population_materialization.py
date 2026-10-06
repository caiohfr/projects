#!/usr/bin/env python3
"""Read-only semantic verification for a component-populated temporary DB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vde_core.component_enrichment_pass1 import ESTIMATOR_VERSION, file_sha256
from src.vde_core.component_population_materialization import _write_csv, protected_snapshot


def _ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--temp-db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protected-csv", type=Path)
    args = parser.parse_args()
    source, temp = _ro(args.source_db), _ro(args.temp_db)
    try:
        before = protected_snapshot(source)
        after = protected_snapshot(temp)
        after_hash = {row["column"]: row["sha256"] for row in after}
        protected = [
            {"table": row["table"], "column": row["column"],
             "sha256_before": row["sha256"], "sha256_after": after_hash[row["column"]],
             "unchanged": row["sha256"] == after_hash[row["column"]]}
            for row in before
        ]
        if args.protected_csv:
            _write_csv(args.protected_csv, protected)

        resolutions: dict[int, dict[str, sqlite3.Row]] = {}
        for row in temp.execute(
            """SELECT l.vde_id,l.boundary,r.* FROM vde_component_resolution l
               JOIN component_resolution r ON r.component_resolution_id=l.component_resolution_id
               WHERE l.adoption_role='ADOPTED' AND r.estimator_version=?""",
            (ESTIMATOR_VERSION,),
        ):
            resolutions.setdefault(int(row["vde_id"]), {})[str(row["boundary"])] = row

        projected = mismatches = bad_provenance = 0
        for row in temp.execute(
            """SELECT id,tire_A_final,tire_B_final,tire_C_final,aero_C_coef_Npkph2,
                      trans_A_coef_N,trans_B_coef_Npkph,trans_C_coef_Npkph2,provenance_json
               FROM vde WHERE tire_A_final IS NOT NULL ORDER BY id"""
        ):
            projected += 1
            macro = resolutions[int(row["id"])]
            expected = (
                macro["ROLLING_MINOR"]["resolved_A_N"], macro["ROLLING_MINOR"]["resolved_B_N_per_kph"],
                macro["ROLLING_MINOR"]["resolved_C_N_per_kph2"], macro["AERO"]["resolved_C_N_per_kph2"],
                macro["DRIVETRAIN_AGGREGATE"]["resolved_A_N"], macro["DRIVETRAIN_AGGREGATE"]["resolved_B_N_per_kph"],
                macro["DRIVETRAIN_AGGREGATE"]["resolved_C_N_per_kph2"],
            )
            actual = tuple(row[name] for name in (
                "tire_A_final", "tire_B_final", "tire_C_final", "aero_C_coef_Npkph2",
                "trans_A_coef_N", "trans_B_coef_Npkph", "trans_C_coef_Npkph2",
            ))
            mismatches += actual != expected
            provenance = json.loads(row["provenance_json"])["component_population_projection"]
            bad_provenance += not (
                provenance["decomposition_mode"] == "MACRO_DECOMPOSITION"
                and provenance["tire_slot_semantics"] == "ROLLING_MINOR_AGGREGATE"
                and provenance["transmission_slot_semantics"] == "DRIVETRAIN_AGGREGATE"
                and provenance["fine_component_identification"] is False
            )

        result = {
            "source_sha256": file_sha256(args.source_db),
            "temp_sha256": file_sha256(args.temp_db),
            "quick_check": temp.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_issues": len(temp.execute("PRAGMA foreign_key_check").fetchall()),
            "projected_macro_vdes": projected,
            "slot_to_macro_mismatches": mismatches,
            "bad_or_missing_projection_provenance": bad_provenance,
            "brake_populated_rows": temp.execute(
                "SELECT COUNT(*) FROM vde WHERE brake_A_coef_N IS NOT NULL OR brake_B_coef_Npkph IS NOT NULL OR brake_C_coef_Npkph2 IS NOT NULL"
            ).fetchone()[0],
            "parasitic_populated_rows": temp.execute(
                "SELECT COUNT(*) FROM vde WHERE parasitic_A_coef_N IS NOT NULL OR parasitic_B_coef_Npkph IS NOT NULL OR parasitic_C_coef_Npkph2 IS NOT NULL"
            ).fetchone()[0],
            "fine_adopted_links": temp.execute(
                "SELECT COUNT(*) FROM vde_component_resolution WHERE adoption_role='ADOPTED' AND boundary IN ('TIRE','BRAKE','HUB_BEARING','TRANSMISSION','AXLE')"
            ).fetchone()[0],
            "fine_supporting_links": temp.execute(
                "SELECT COUNT(*) FROM vde_component_resolution WHERE adoption_role='SUPPORTING' AND boundary IN ('TIRE','BRAKE','HUB_BEARING','TRANSMISSION','AXLE')"
            ).fetchone()[0],
            "protected_columns_changed": sum(not row["unchanged"] for row in protected),
        }
        result["semantic_checks_passed"] = all((
            result["quick_check"] == "ok", result["foreign_key_issues"] == 0,
            projected > 0, mismatches == 0, bad_provenance == 0,
            result["brake_populated_rows"] == 0, result["parasitic_populated_rows"] == 0,
            result["fine_adopted_links"] == 0, result["protected_columns_changed"] == 0,
        ))
    finally:
        source.close()
        temp.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["semantic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
