from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.technical_research.contracts import SourceRecord  # noqa: E402
from capabilities.technical_research.final_micropatch import (  # noqa: E402
    FINAL_STATUS,
    run_golden_validation,
    source_relevance,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _db_integrity(path: Path) -> dict[str, object]:
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    result = {
        "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
        "foreign_key_issues": len(conn.execute("PRAGMA foreign_key_check").fetchall()),
        "schema_objects": conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchone()[0],
    }
    conn.close()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sprint 12 Research Agent final micro-patch golden validation")
    parser.add_argument("--db", type=Path, default=ROOT / "data/db/staging/eco_drive_canonical_candidate.db")
    parser.add_argument(
        "--phasea-inputs", type=Path,
        default=ROOT / "inputs/phasea1_pkg/EcoDrive_Sprint12_Technical_Research_Agent_PhaseA1_Reconciliation_LiveAgent_v1.0/inputs",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research_agent_final_micropatch")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    db_hash_before = _sha256(args.db)
    integrity_before = _db_integrity(args.db)
    tasks = _csv(args.phasea_inputs / "research_child_tasks.csv")
    prior_evidence = _jsonl(args.phasea_inputs / "research_evidence_staging.jsonl")

    live = run_golden_validation(
        tasks=tasks, prior_evidence=prior_evidence, output_dir=output,
        cache_dir=output / "cache", cache_only=False,
    )
    replay = run_golden_validation(
        tasks=tasks, prior_evidence=prior_evidence, output_dir=output / "cache_replay",
        cache_dir=output / "cache", cache_only=True,
    )
    cache_parity = live["structured_result_sha256"] == replay["structured_result_sha256"]

    task_by_id = {task["child_task_id"]: task for task in tasks}
    bad_cases = [
        (
            "HYUNDAI_SCHNEIER",
            SourceRecord("bad-schneier", "https://www.schneier.com/news/idg-se/", "News – Schneier on Security", "www.schneier.com"),
            task_by_id["RT-C519340F31320050"], "ARCHITECTURE",
        ),
        (
            "MAZDA_REAL_ESTATE",
            SourceRecord("bad-apartment", "https://summerplaceal.com/floor-plans/", "Floor Plans | Summer Place Apartment Homes", "summerplaceal.com"),
            task_by_id["RT-02BB5A52D4C596CB"], "TRANSMISSION_AXLE_BOUNDARY",
        ),
    ]
    bad_results = []
    for case_id, source, task, claim_type in bad_cases:
        decision = source_relevance(source, task, claim_type)
        bad_results.append({"case_id": case_id, "url": source.url, **decision, "fetch_attempted": False})
    bad_blocked = all(row["relevance_status"] == "REJECT" for row in bad_results)
    (output / "known_bad_result_regression.json").write_text(json.dumps(bad_results, indent=2, ensure_ascii=False), encoding="utf-8")

    db_hash_after = _sha256(args.db)
    integrity_after = _db_integrity(args.db)
    db_safe = db_hash_before == db_hash_after and integrity_after["quick_check"] == "ok" and integrity_after["foreign_key_issues"] == 0
    freeze_passed = cache_parity and bad_blocked and db_safe and live["direct_research_abc_writes"] == 0
    status = FINAL_STATUS if freeze_passed else "PATCH_FAILED"

    cache_lines = [
        f"network_result_sha256={live['structured_result_sha256']}",
        f"cache_replay_result_sha256={replay['structured_result_sha256']}",
        f"hashes_equal={str(cache_parity).lower()}",
        f"network_search_requests={live['search_request_count']}",
        f"replay_search_requests={replay['search_request_count']}",
        f"network_source_fetches={live['network_source_fetches']}",
        f"replay_source_fetches={replay['network_source_fetches']}",
        f"replay_cache_hits={replay['cache_hits']}",
        f"replay_cache_misses={replay['cache_misses']}",
    ]
    (output / "research_agent_cache_replay_check.txt").write_text("\n".join(cache_lines) + "\n", encoding="utf-8")

    completion = {
        "status": status, "freeze_passed": freeze_passed,
        "agent_classification": "EXPERIMENTAL_PARTIAL_FROZEN_FOR_REVIEW",
        "fleet_scale_research": "NOT_AUTHORIZED", "p1_p2_executed": False,
        "network_run": live, "cache_replay": replay, "cache_replay_parity": cache_parity,
        "known_bad_results_rejected_before_fetch": bad_blocked,
        "database": {
            "path": str(args.db.resolve()), "sha256_before": db_hash_before, "sha256_after": db_hash_after,
            "unchanged": db_hash_before == db_hash_after, "integrity_before": integrity_before, "integrity_after": integrity_after,
        },
        "non_goals": {
            "rag": False, "embeddings": False, "vector_db": False, "llm_extraction": False,
            "multi_agent": False, "new_estimator_physics": False, "new_schema": False,
        },
    }
    (output / "completion_manifest.json").write_text(json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# EcoDrive Sprint 12 — Research Agent Final Micro-Patch", "",
        f"## Final status: `{status}`", "",
        "Research Agent v0 is **experimental / partial / frozen for owner review**. It is not production-ready and fleet-scale P1/P2 research is not authorized.", "",
        "## Lessons implemented", "",
        f"- Internal-first: {live['internal_resolved']} fully resolved, {live['internal_partial']} partial, {live['internal_not_found']} not found; {live['network_skipped_tasks']} tasks skipped network.",
        f"- Estimator unlock gate: {live['deferred_no_current_model_value']} task(s) returned `DEFER_NO_CURRENT_MODEL_VALUE`.",
        "- Claim-specific templates: `ARCHITECTURE_V1`, `TRANSMISSION_IDENTITY_V1`, `TRANSMISSION_AXLE_BOUNDARY_V1`.",
        f"- Pre-fetch relevance gate: {live['pre_fetch_rejected']} result(s) rejected before fetch.",
        "- Source ranking: deterministic Tier 1–4 tuple ordered by authority, entity, claim and technical-document signals.",
        "- Stop states: exact/strong/partial/not-found/boundary-unknown/defer are explicit and no reformulation loop exists.", "",
        "## Golden validation", "",
        f"- Tasks: {live['golden_task_count']}",
        f"- Search requests/results: {live['search_request_count']} / {live['search_results_returned']}",
        f"- Results after gate: {live['search_results_returned'] - live['pre_fetch_rejected']}",
        f"- Sources fetched: {live['sources_fetched']}",
        f"- Accepted evidence: {live['accepted_evidence']}",
        f"- Status distribution: `{json.dumps(live['status_counts'], sort_keys=True)}`", "",
        "## Known Phase A.1 failures", "",
        f"- Hyundai → Schneier rejected before fetch: **{bad_results[0]['relevance_status'] == 'REJECT'}** (`{bad_results[0]['rejection_reason']}`).",
        f"- Mazda → real-estate/apartment rejected before fetch: **{bad_results[1]['relevance_status'] == 'REJECT'}** (`{bad_results[1]['rejection_reason']}`).", "",
        "## Determinism and safety", "",
        f"- Network/cache structured-result parity: **{cache_parity}**.",
        f"- Cache replay network calls: search={replay['search_request_count']}, fetch={replay['network_source_fetches']}.",
        f"- Canonical candidate SHA256 before/after: `{db_hash_before}` / `{db_hash_after}`.",
        f"- `quick_check`: `{integrity_after['quick_check']}`; FK issues: **{integrity_after['foreign_key_issues']}**.",
        "- Direct research-to-ABC writes: **0**.",
        "- No P1/P2, RAG, embeddings, vector DB, LLM extraction, new estimator physics or schema change.", "",
        "## Remaining Sprint 12 closure items", "",
        "1. Record Research Agent v0 as experimental/partial and frozen.",
        "2. Confirm the already validated canonical Component Population counts and persistence semantics.",
        "3. Explicitly record whether candidate promotion is executed or intentionally deferred.",
        "4. Record final integrity, reproducibility, idempotency and manifest hashes.",
        "5. Produce the final Component Population coverage report with macro, fine-supporting, unresolved, deferred and boundary-unknown separated.",
        "6. Update Sprint 12/component methodology documentation and run the appropriate closure regression suites.",
        "7. Produce the final Sprint 12 Closure Report and mark the Sprint closed.", "",
        "No additional Research Agent capability is part of these closure items.",
    ]
    (output / "research_agent_final_micropatch_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(completion, indent=2, ensure_ascii=False))
    return 0 if freeze_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
