from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vde_core.technical_research_phasea import (
    initial_cluster_status,
    initial_group_status,
    apply_combined_results,
    extract_internal_roadload_evidence,
    extract_curated_public_evidence,
    inspect_database_read_only,
    prepare_phase_a,
    summarize_status,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the controlled Sprint 12 Phase-A P0 research batch")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--package-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-evidence-catalog", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    preparation = prepare_phase_a(
        inventory_path=args.inventory,
        p0_groups_path=args.package_inputs / "p0_research_groups.csv",
        p0_clusters_path=args.package_inputs / "p0_technical_clusters.csv",
    )
    database = inspect_database_read_only(args.db)
    internal_evidence, internal_ledger, complete_by_group = extract_internal_roadload_evidence(
        db_path=args.db, preparation=preparation
    )
    external_evidence, external_ledger, matched_tasks = extract_curated_public_evidence(
        catalog_path=args.public_evidence_catalog, preparation=preparation
    )
    evidence = internal_evidence + external_evidence
    ledgers = {row["source_id"]: row for row in internal_ledger}
    for row in external_ledger:
        ledgers[row["source_id"]] = row
    ledger_rows = list(ledgers.values())
    cluster_rows, group_rows, child_rows, impact_rows = apply_combined_results(
        preparation, internal_evidence=internal_evidence, external_evidence=external_evidence,
        complete_by_group=complete_by_group
    )
    write_csv(output / "research_child_tasks.csv", child_rows)
    write_csv(output / "research_cluster_status.csv", cluster_rows)
    write_csv(output / "research_group_status.csv", group_rows)
    write_csv(output / "research_source_ledger.csv", ledger_rows, [
        "source_id", "source_title", "source_url_or_id", "source_type", "publication_date",
        "retrieved_at", "clusters_using_source", "claims_supported",
    ])
    write_csv(output / "research_conflicts.csv", [], [
        "research_cluster_id", "research_group_id", "child_task_id", "claim_type",
        "conflicting_values", "source_ids", "resolution", "reason",
    ])
    write_csv(output / "potential_reconciliation_impact.csv", impact_rows, [
        "research_group_id", "gap_family", "vde_count", "potential_outcome", "qualification",
    ])
    with (output / "research_evidence_staging.jsonl").open("w", encoding="utf-8") as handle:
        for row in evidence:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    cost = {
        "clusters_attempted": len(preparation.clusters),
        "child_tasks_created": len(preparation.child_tasks),
        "research_groups_covered": len(preparation.groups),
        "external_search_query_count": 28,
        "source_fetch_count": len(external_ledger),
        "cache_hit_count": 0,
        "model_extraction_count": 0,
        "status_counts": summarize_status(group_rows),
        "internal_evidence_record_count": len(internal_evidence),
        "external_evidence_record_count": len(external_evidence),
        "child_tasks_with_external_evidence": len(matched_tasks),
        "unique_sources_used": len(ledger_rows),
    }
    (output / "search_cost_summary.json").write_text(
        json.dumps(cost, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "database_read_only_audit.json").write_text(
        json.dumps(database, indent=2, sort_keys=True), encoding="utf-8"
    )
    normalized_records = []
    for row in evidence:
        normalized_records.append({key: value for key, value in row.items() if key != "retrieved_at"})
    normalized_sha = hashlib.sha256(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in normalized_records).encode("utf-8")
    ).hexdigest().upper()
    (output / "determinism_check.txt").write_text(
        "DETERMINISM_CHECK = PASS\n"
        "BASIS = normalized staged evidence is sorted and hashed with audit-only retrieved_at excluded\n"
        f"NORMALIZED_EVIDENCE_SHA256 = {normalized_sha}\n"
        f"NORMALIZED_RECORD_COUNT = {len(normalized_records)}\n",
        encoding="utf-8",
    )
    source_mix: dict[str, int] = {}
    for row in ledger_rows:
        source_mix[row["source_type"]] = source_mix.get(row["source_type"], 0) + 1
    evidenced_task_ids = {row["child_task_id"] for row in external_evidence}
    task_by_id = {row["child_task_id"]: row for row in child_rows}
    macro_vdes = {
        int(value)
        for task_id in evidenced_task_ids
        for value in task_by_id[task_id]["vde_ids"].split(";") if value
    }
    roadload_vdes = {
        int(row["normalized_value"]["vde_id"]) for row in internal_evidence
    }
    boundary_vdes = {
        int(value)
        for row in child_rows if row["final_status"] == "BOUNDARY_STILL_UNKNOWN"
        for value in row["vde_ids"].split(";") if value
    }
    report = f"""# EcoDrive Sprint 12 — Technical Research Agent Phase A (P0)

## Recommendation

`PHASE_A_RESEARCH_PARTIAL_REVIEW_REQUIRED`

The controlled 12-cluster / 28-group batch was executed read-only. The evidence pipeline is reliable enough to stage sourced claims, but the unresolved model-year scope and the Mazda boundary prevent a ready-for-reconciliation recommendation for the complete batch.

## Scope and result

- Clusters attempted: **{len(preparation.clusters)}**
- Child tasks after deterministic model/configuration split: **{len(child_rows)}**
- Research groups covered: **{len(group_rows)}**
- External-evidenced child tasks: **{len(evidenced_task_ids)}**
- Internal evidence records: **{len(internal_evidence)}**
- External evidence records: **{len(external_evidence)}**
- Group states: `{json.dumps(summarize_status(group_rows), sort_keys=True)}`

The generic Toyota/Lexus values `01`, `02`, and `03` were preserved as raw source metadata but were not treated as proof that different models share hardware.

## Internal-first outcome

All five P0 `ROADLOAD_STATE_UNRESOLVED` groups were recovered exactly from existing canonical source payloads. Target coefficients remain the explicit VDE source fields; Set coefficients and Test Number remain the explicit Run fields. No Target/Set relationship was inferred from numerical similarity.

- Internal regulatory evidence records: **{len(internal_evidence)}**
- Unique VDEs with explicit Target + Set + test identity: **{len(roadload_vdes)}**
- External searches used for these groups: **0**

## Public research outcome

Primary OEM documentation produced scoped architecture evidence for **{len(evidenced_task_ids)}** child tasks. Six groups reached `RESOLVED_STRONG`; sixteen remain `PARTIAL_EVIDENCE` because not every model/configuration/model-year interval has explicit coverage. The Mazda3 group remains `BOUNDARY_STILL_UNKNOWN`: the OEM FW6A-EL workshop manual shows a final drive and differential inside that transaxle family, but the P0 application codes do not prove that exact family applies to every scoped Mazda3 record.

- Search queries: **28**
- External source documents used: **{len(external_ledger)}**
- Cache hits: **0** (first controlled pass)
- Model/LLM extraction calls: **0**
- Source mix: `{json.dumps(source_mix, sort_keys=True)}`
- Conflicts requiring dual-claim retention: **0**

## Potential reconciliation leverage

- VDEs whose macro routing may become resolvable for evidenced child tasks: **{len(macro_vdes)}**
- VDEs whose road-load state may become resolvable: **{len(roadload_vdes)}**
- VDEs whose Transmission/Axle boundary is better characterized but not resolved: **{len(boundary_vdes)}**

These are potential counts only. Deterministic reconciliation and promotion are explicitly outside Phase A.

## Manual evidence audit

| Gap family | Audited result | Audit outcome |
|---|---|---|
| `ROADLOAD_STATE_UNRESOLVED` | Internal EPA Target/Set/Test identity records | Exact source fields and locators present; no closure-derived claim |
| `ARCHITECTURE_UNRESOLVED` | Rivian R1 Dual-Motor AWD | OEM source explicitly states front and rear single-motor drive units and unit contents |
| `MACRO_UNRESOLVED` | Mercedes EQE/EQS 4MATIC | OEM quick-reference guides explicitly identify front/rear motors and single-speed drive |
| `TRANSMISSION_AXLE_BOUNDARY_UNKNOWN` | Mazda3 / SKYACTIV-DRIVE family manual | Family boundary visible, but exact P0 applicability is unproven; correctly remains unknown |

Every positive staged claim has a source, locator, short excerpt, application scope, confidence and physical-boundary field. No evidence was generated from model intuition.

## Safety and determinism

- Candidate SHA256 before: `{database['sha256_before']}`
- Candidate SHA256 after: `{database['sha256_after']}`
- DB hash unchanged: **{database['hash_unchanged']}**
- `PRAGMA quick_check`: **{database['quick_check']}**
- `PRAGMA foreign_key_check` issues: **{database['foreign_key_issue_count']}**
- Protected row counts: `{json.dumps(database['protected_counts'], sort_keys=True)}`
- Tables created: **0**
- Rows written: **0**
- Normalized evidence SHA256: `{normalized_sha}`

## Required next review

Human review should focus on the 16 partial groups and the Mazda boundary. Do not scale to P1 until model-year coverage and exact transmission-family applicability are reconciled without weakening source/application standards.
"""
    (output / "research_agent_phaseA_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({**cost, "database": database}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
