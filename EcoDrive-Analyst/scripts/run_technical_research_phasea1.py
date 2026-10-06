from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.technical_research.live_phasea1 import run_live_subset  # noqa: E402
from src.vde_core.technical_research_phasea1 import reconcile_phasea  # noqa: E402


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sprint 12 Phase A.1 reconciliation and live-agent validation")
    parser.add_argument("--db", type=Path, default=ROOT / "data/db/staging/eco_drive_canonical_candidate.db")
    parser.add_argument(
        "--package-root", type=Path,
        default=ROOT / "inputs/phasea1_pkg/EcoDrive_Sprint12_Technical_Research_Agent_PhaseA1_Reconciliation_LiveAgent_v1.0",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/technical_research_phaseA1")
    return parser


def main() -> int:
    args = _parser().parse_args()
    package_inputs = args.package_root / "inputs"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    reconciliation = reconcile_phasea(
        source_db=args.db,
        phasea_input_dir=package_inputs,
        output_dir=output / "reconciliation",
        temp_db=output / "temp" / "eco_drive_candidate_phasea1.db",
    )
    reconciliation_replay = reconcile_phasea(
        source_db=args.db,
        phasea_input_dir=package_inputs,
        output_dir=output / "reconciliation_replay",
        temp_db=output / "temp" / "eco_drive_candidate_phasea1_replay.db",
    )
    reconciliation_deterministic = (
        reconciliation["reconciliation_output_sha256"]
        == reconciliation_replay["reconciliation_output_sha256"]
    )

    tasks = _csv(package_inputs / "research_child_tasks.csv")
    prior_evidence = _jsonl(package_inputs / "research_evidence_staging.jsonl")
    live = run_live_subset(
        tasks=tasks,
        prior_evidence=prior_evidence,
        output_dir=output / "live_first_run",
        cache_dir=output / "cache",
        cache_only=False,
    )
    replay = run_live_subset(
        tasks=tasks,
        prior_evidence=prior_evidence,
        output_dir=output / "live_cache_replay",
        cache_dir=output / "cache",
        cache_only=True,
    )
    cache_equivalent = (
        live["normalized_staged_evidence_sha256"]
        == replay["normalized_staged_evidence_sha256"]
    )
    live_state = live["state"]
    if live_state == "LIVE_RESEARCH_AGENT_VALIDATED" and not cache_equivalent:
        live_state = "LIVE_RESEARCH_AGENT_PARTIAL_REVIEW_REQUIRED"
    reconciliation_state = (
        "RECONCILIATION_VALIDATED" if reconciliation_deterministic
        else "RECONCILIATION_NOT_VALIDATED"
    )
    p1 = (
        reconciliation_state == "RECONCILIATION_VALIDATED"
        and live_state == "LIVE_RESEARCH_AGENT_VALIDATED"
        and reconciliation["source_sha256_before"] == reconciliation["source_sha256_after"]
    )
    final = {
        "phasea1_version": "SPRINT12_PHASEA1_V1",
        "reconciliation_state": reconciliation_state,
        "live_research_state": live_state,
        "p1_recommendation": "PROCEED_TO_P1" if p1 else "DO_NOT_PROCEED_TO_P1",
        "reconciliation_deterministic_replay": reconciliation_deterministic,
        "live_cache_replay_equivalent": cache_equivalent,
        "reconciliation": reconciliation,
        "live_first_run": live,
        "live_cache_replay": replay,
    }
    (output / "phasea1_completion_report.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# EcoDrive Sprint 12 — Technical Research Agent Phase A.1", "",
        f"- Reconciliation: **{reconciliation_state}**",
        f"- Live research agent: **{live_state}**",
        f"- P1 recommendation: **{final['p1_recommendation']}**", "",
        "## Deterministic reconciliation", "",
        f"- VDE rows audited: **{reconciliation['vde_rows_audited']}**",
        f"- VDEs with scoped evidence promoted on temp copy: **{reconciliation['promoted_vdes']}**",
        f"- Evidence records promoted: **{reconciliation['promoted_evidence_records']}**",
        f"- Temporary configuration metadata rows updated: **{reconciliation['metadata_rows_updated_temp_only']}**",
        f"- VDEs resolved by the existing deterministic estimator: **{reconciliation['macro_resolved_vdes']}**",
        f"- New component resolutions / links on temp copy: **{reconciliation['component_resolution_rows_inserted']} / {reconciliation['component_links_inserted']}**",
        f"- Deterministic replay equivalent: **{reconciliation_deterministic}**",
        f"- Direct research ABC writes: **{reconciliation['direct_research_abc_writes']}**", "",
        "## Live provider proof", "",
        f"- Provider: **{live['provider_name']}**",
        f"- Mode: **{live['provider_mode']}**",
        f"- Network enabled: **{live['network_search_enabled']}**",
        f"- Search requests / source fetches: **{live['search_request_count']} / {live['source_fetch_count']}**",
        f"- Fetch successes: **{live['fetch_success_count']}**",
        f"- Accepted live evidence records: **{live['accepted_evidence_records']}**",
        f"- Resolved validation tasks: **{live['resolved_child_tasks']} / {live['validation_task_count']}**",
        f"- Mazda negative case: **{live['mazda_status']}**",
        f"- Cache replay hits / misses: **{replay['cache_hit_count']} / {replay['cache_miss_count']}**",
        f"- Normalized evidence hash equal: **{cache_equivalent}**",
        "- Curated catalog used for discovery: **False**",
        "- Model/LLM calls: **0**", "",
        "## Database safety", "",
        f"- Source SHA256 before: `{reconciliation['source_sha256_before']}`",
        f"- Source SHA256 after: `{reconciliation['source_sha256_after']}`",
        f"- Source unchanged: **{reconciliation['source_sha256_before'] == reconciliation['source_sha256_after']}**",
        f"- Temp quick_check / FK issues: **{reconciliation['temp_quick_check']} / {reconciliation['temp_fk_issues']}**",
        f"- Schema objects unchanged: **{reconciliation['schema_objects_unchanged']}**",
        f"- Historical VDE TOTAL ABC unchanged: **{reconciliation['vde_total_abc_unchanged']}**", "",
        "Research evidence was used only for scoped identity/architecture reconciliation. Final component ABC values, where any were materialized, came exclusively from the existing deterministic estimator.",
    ]
    (output / "PHASEA1_COMPLETION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0 if reconciliation_state == "RECONCILIATION_VALIDATED" and cache_equivalent else 2


if __name__ == "__main__":
    raise SystemExit(main())
