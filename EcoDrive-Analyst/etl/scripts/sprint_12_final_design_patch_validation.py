"""Read-only acceptance evidence for the Sprint 12 final design patch."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "db" / "staging" / "eco_drive_canonical_candidate.db"
DEFAULT_PRIOR_WORKBOOK = ROOT / "artifacts" / "db_review" / "EcoDrive_Canonical_DB_Review_Phase2.xlsx"
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(reference: str) -> int:
    result = 0
    for character in (char for char in reference if char.isalpha()):
        result = result * 26 + ord(character.upper()) - 64
    return result - 1


def workbook_rows(path: Path, sheet_name: str) -> list[list[object]]:
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in rels.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        }
        sheet = next(
            item for item in workbook.findall(f".//{{{MAIN_NS}}}sheet")
            if item.attrib["name"] == sheet_name
        )
        target = "xl/" + targets[sheet.attrib[f"{{{REL_NS}}}id"]]
        root = ET.fromstring(archive.read(target))

    rows: list[list[object]] = []
    for row in root.findall(f".//{{{MAIN_NS}}}row"):
        values: list[object] = []
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            index = _column_index(cell.attrib["r"])
            while len(values) <= index:
                values.append(None)
            if cell.attrib.get("t") == "inlineStr":
                value: object = "".join(
                    item.text or "" for item in cell.findall(f".//{{{MAIN_NS}}}t")
                )
            else:
                node = cell.find(f"{{{MAIN_NS}}}v")
                raw = None if node is None else node.text
                if raw is None:
                    value = None
                else:
                    number = float(raw)
                    value = int(number) if number.is_integer() else number
            values[index] = value
        rows.append(values)
    return rows


def _run_evidence(connection: sqlite3.Connection, vde_id: int) -> dict[str, list[object]]:
    rows = connection.execute(
        """
        SELECT json_extract(conditions_json,'$.test_number'),
               json_extract(conditions_json,'$.test_category'),
               procedure_code,procedure_description,
               json_extract(conditions_json,'$.set_abc_native')
        FROM run WHERE vde_id=? ORDER BY 1,2,3,4,5
        """,
        (vde_id,),
    ).fetchall()
    return {
        "test_numbers": sorted({row[0] for row in rows if row[0] is not None}),
        "test_categories": sorted({row[1] for row in rows if row[1] is not None}),
        "procedures": sorted({row[3] or row[2] for row in rows if (row[3] or row[2]) is not None}),
        "set_abc_variants": sorted({row[4] for row in rows if row[4] is not None}),
    }


def _run_carryover_metrics(connection: sqlite3.Connection) -> dict[str, object]:
    joins = """
        FROM run child
        JOIN run parent
          ON parent.run_id=json_extract(child.provenance_json,'$.carryover_from_run_id')
        WHERE json_extract(child.provenance_json,'$.lineage_relation')='EPA_MODEL_YEAR_CARRYOVER'
    """
    test_number_mismatch = """
        AND (
          json_extract(child.conditions_json,'$.test_number')
            IS NOT json_extract(parent.conditions_json,'$.test_number')
          OR json_extract(child.conditions_json,'$.adfe_test_number')
            IS NOT json_extract(parent.conditions_json,'$.adfe_test_number')
        )
    """
    examples = [dict(row) for row in connection.execute(
        """
        SELECT child.run_id child_run_id,
               parent.run_id parent_run_id,
               json_extract(child.conditions_json,'$.test_number') child_test_number,
               json_extract(parent.conditions_json,'$.test_number') parent_test_number,
               json_extract(child.conditions_json,'$.adfe_test_number') child_adfe_test_number,
               json_extract(parent.conditions_json,'$.adfe_test_number') parent_adfe_test_number
        """ + joins + test_number_mismatch + " ORDER BY child.run_id LIMIT 10"
    )]
    return {
        "links": connection.execute("SELECT COUNT(*) " + joins).fetchone()[0],
        "adfe_test_number_populated_runs": connection.execute(
            "SELECT COUNT(*) FROM run WHERE json_extract(conditions_json,'$.adfe_test_number') IS NOT NULL"
        ).fetchone()[0],
        "test_number_mismatch_links": connection.execute(
            "SELECT COUNT(*) " + joins + test_number_mismatch
        ).fetchone()[0],
        "ford_observed_pair_links": connection.execute(
            "SELECT COUNT(*) " + joins + """
              AND json_extract(child.conditions_json,'$.test_number')='KFMX10071543'
              AND json_extract(parent.conditions_json,'$.test_number')='KFMX10067860'
            """
        ).fetchone()[0],
        "mismatch_examples": examples,
    }


def _table_signature(connection: sqlite3.Connection, table: str, order_by: str) -> str:
    digest = hashlib.sha256()
    for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY "{order_by}"'):
        digest.update(json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _compare_unchanged_behavior(db_path: Path, compare_db: Path | None) -> dict[str, object] | None:
    if compare_db is None:
        return None
    current = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    prior = sqlite3.connect(f"file:{compare_db.resolve().as_posix()}?mode=ro", uri=True)
    try:
        result: dict[str, object] = {}
        for table, key in (("vde", "id"), ("fuelcons", "id"), ("fuelcons_run_adoption", "fuelcons_id")):
            current_signature = _table_signature(current, table, key)
            prior_signature = _table_signature(prior, table, key)
            result[table] = {
                "prior_sha256": prior_signature,
                "current_sha256": current_signature,
                "byte_equivalent_rows": prior_signature == current_signature,
            }
        relation_sql = """
            SELECT child.run_id,
                   parent.run_id,
                   json_extract(child.conditions_json,'$.test_number'),
                   json_extract(parent.conditions_json,'$.test_number')
            FROM run child
            JOIN run parent
              ON parent.run_id=json_extract(child.provenance_json,'$.carryover_from_run_id')
            WHERE json_extract(child.provenance_json,'$.lineage_relation')='EPA_MODEL_YEAR_CARRYOVER'
        """
        prior_links = {
            row[0]: {"parent": row[1], "child_test_number": row[2], "parent_test_number": row[3]}
            for row in prior.execute(relation_sql)
        }
        current_links = {
            row[0]: {"parent": row[1], "child_test_number": row[2], "parent_test_number": row[3]}
            for row in current.execute(relation_sql)
        }
        invalid_prior = {
            child_id: relation
            for child_id, relation in prior_links.items()
            if relation["child_test_number"] != relation["parent_test_number"]
        }
        invalid_removed = {
            child_id: relation
            for child_id, relation in invalid_prior.items()
            if current_links.get(child_id, {}).get("parent") != relation["parent"]
        }
        result["run_carryover"] = {
            "prior_links": len(prior_links),
            "current_links": len(current_links),
            "net_link_delta": len(current_links) - len(prior_links),
            "prior_mismatched_test_number_links": len(invalid_prior),
            "invalid_prior_links_removed": len(invalid_removed),
            "removed_and_now_unlinked": sum(child_id not in current_links for child_id in invalid_removed),
            "reparented_to_exact_test_number": sum(child_id in current_links for child_id in invalid_removed),
            "new_or_reparented_exact_links": sum(
                child_id not in prior_links
                or prior_links[child_id]["parent"] != relation["parent"]
                for child_id, relation in current_links.items()
            ),
        }
        return result
    finally:
        current.close()
        prior.close()


def validate(
    db_path: Path,
    prior_workbook: Path | None,
    compare_db: Path | None = None,
) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = (
            "program", "vehicle_configuration", "vde", "run", "fuelcons",
            "fuelcons_run_adoption", "component_db", "component_instance",
            "component_resolution", "vde_component_resolution", "tire_db",
        )
        counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }
        cycle_counts = dict(connection.execute(
            "SELECT cycle_name,COUNT(*) FROM vde WHERE legislation='EPA' GROUP BY cycle_name ORDER BY cycle_name"
        ))
        carryover_count = connection.execute(
            "SELECT COUNT(*) FROM vde WHERE json_extract(provenance_json,'$.lineage_relation')='EPA_MODEL_YEAR_CARRYOVER'"
        ).fetchone()[0]
        fuelcons_carryover_count = connection.execute(
            "SELECT COUNT(*) FROM fuelcons WHERE json_extract(provenance_json,'$.lineage_relation')='EPA_MODEL_YEAR_CARRYOVER'"
        ).fetchone()[0]
        review_count, missing_review_reasons = connection.execute(
            """
            SELECT COUNT(*),SUM(CASE WHEN trim(COALESCE(
              json_extract(provenance_json,'$.run_identity_review_reason'),''))=''
              THEN 1 ELSE 0 END)
            FROM run WHERE review_status='RUN_IDENTITY_REVIEW'
            """
        ).fetchone()
        scalar_counts = dict(connection.execute(
            """
            SELECT 'roadload_temperature_c',COUNT(*) FROM vde WHERE roadload_temperature_c IS NOT NULL
            UNION ALL SELECT 'roadload_ambient_pressure_kpa',COUNT(*) FROM vde WHERE roadload_ambient_pressure_kpa IS NOT NULL
            UNION ALL SELECT 'engine_rated_power_kw',COUNT(*) FROM vehicle_configuration WHERE engine_rated_power_kw IS NOT NULL
            UNION ALL SELECT 'engine_cylinders_rotors',COUNT(*) FROM vehicle_configuration WHERE engine_cylinders_rotors IS NOT NULL
            """
        ))

        changed_case = None
        if prior_workbook is not None and prior_workbook.exists():
            rows = workbook_rows(prior_workbook, "EPA_CARRYOVER_REVIEW")
            headers = {str(value): index for index, value in enumerate(rows[0])}
            for row in rows[1:]:
                old_parent = row[headers["parent_vde_id"]]
                if old_parent is None:
                    continue
                vde_id = int(row[headers["vde_id"]])
                current = connection.execute(
                    "SELECT vde_id_parent,year,make,model FROM vde WHERE id=?", (vde_id,)
                ).fetchone()
                if current is None or current["vde_id_parent"] is not None:
                    continue
                child_evidence = _run_evidence(connection, vde_id)
                parent_evidence = _run_evidence(connection, int(old_parent))
                if child_evidence["test_numbers"] != parent_evidence["test_numbers"]:
                    changed_case = {
                        "vde_id": vde_id,
                        "model_year": current["year"],
                        "make": current["make"],
                        "model": current["model"],
                        "phase2_parent_vde_id": int(old_parent),
                        "final_parent_vde_id": None,
                        "child_evidence": child_evidence,
                        "prior_parent_evidence": parent_evidence,
                        "reason": "Changed Test Number evidence blocks exact carryover V2.",
                    }
                    break

        bmw = [dict(row) for row in connection.execute(
            "SELECT id,year,vde_id_parent,cycle_name,cycle_source FROM vde "
            "WHERE id IN (-1001790,-1010565,-1009385) ORDER BY year"
        )]
        multi_cycle = dict(connection.execute(
            "SELECT id,cycle_name,cycle_source,roadload_temperature_c,roadload_ambient_pressure_kpa "
            "FROM vde WHERE id=-1004236"
        ).fetchone())
        multi_cycle["run_categories"] = [row[0] for row in connection.execute(
            "SELECT DISTINCT json_extract(result_details_json,'$.canonical_result.Test Category') "
            "FROM run WHERE vde_id=-1004236 ORDER BY 1"
        )]
        cadillac = {
            "runs": connection.execute(
                "SELECT COUNT(*) FROM run WHERE json_extract(conditions_json,'$.test_number')='NGMX91004749'"
            ).fetchone()[0],
            "variants": connection.execute(
                "SELECT COUNT(DISTINCT json_extract(conditions_json,'$.set_abc_native')) FROM run "
                "WHERE json_extract(conditions_json,'$.test_number')='NGMX91004749'"
            ).fetchone()[0],
            "missing_review_reasons": connection.execute(
                "SELECT COUNT(*) FROM run WHERE json_extract(conditions_json,'$.test_number')='NGMX91004749' "
                "AND trim(COALESCE(json_extract(provenance_json,'$.run_identity_review_reason'),''))=''"
            ).fetchone()[0],
        }
        return {
            "quick_check": connection.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_issues": len(connection.execute("PRAGMA foreign_key_check").fetchall()),
            "counts": counts,
            "epa_cycle_counts": cycle_counts,
            "vde_carryover_count": carryover_count,
            "fuelcons_carryover_count": fuelcons_carryover_count,
            "run_carryover": _run_carryover_metrics(connection),
            "run_identity_review_count": review_count,
            "run_identity_review_missing_reasons": missing_review_reasons or 0,
            "scalar_counts": scalar_counts,
            "changed_test_number_case": changed_case,
            "bmw_330i": bmw,
            "multi_cycle_vde": multi_cycle,
            "cadillac_NGMX91004749": cadillac,
            "unchanged_behavior_comparison": _compare_unchanged_behavior(db_path, compare_db),
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--prior-workbook", type=Path, default=DEFAULT_PRIOR_WORKBOOK)
    parser.add_argument("--compare-db", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.db, args.prior_workbook, args.compare_db), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
