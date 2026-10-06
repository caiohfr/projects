from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capabilities.technical_research import research_batch  # noqa: E402
from capabilities.technical_research.adapters import (  # noqa: E402
    load_bmw_benchmark_cases,
    result_for_ecodrive_review,
)


DEFAULT_SAMPLE = ROOT / "artifacts/components/transmission_experiment/EXPERIMENT_SAMPLE_AUDIT.csv"
DEFAULT_GROUPS = ROOT / "artifacts/components/transmission_experiment/TRANSMISSION_CANDIDATE_GROUPS.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/technical_research"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(sample_path: Path, groups_path: Path, output_dir: Path, *, limit: int) -> None:
    cases = load_bmw_benchmark_cases(sample_path, groups_path, limit=limit)
    results = research_batch(case.request for case in cases)
    review_rows = [
        result_for_ecodrive_review(case, result)
        for case, result in zip(cases, results, strict=True)
    ]
    _write_csv(
        output_dir / "BMW_TRANSMISSION_RESEARCH_RESULTS.csv",
        review_rows,
        list(review_rows[0]) if review_rows else ["vde_id"],
    )
    source_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    ingestion_rows: list[dict[str, object]] = []
    for case, result in zip(cases, results, strict=True):
        for claim in result.evidence.claims:
            source_rows.append(
                {
                    "request_id": result.request_id,
                    "source_id": claim.source_id,
                    "field": claim.field,
                    "source_tier": claim.source_tier.value,
                    "evidence_location": claim.evidence_location,
                    "application_match": claim.application_match.value,
                }
            )
        for conflict in result.evidence.conflicts:
            conflict_rows.append(
                {
                    "request_id": result.request_id,
                    "field": conflict.field,
                    "values": ";".join(map(str, conflict.values)),
                    "resolution": conflict.resolution.value,
                }
            )
        for action in result.ingestion_summary.get("actions", []):
            ingestion_rows.append({"request_id": result.request_id, **action})
    _write_csv(
        output_dir / "SOURCE_AUDIT.csv",
        source_rows,
        ["request_id", "source_id", "field", "source_tier", "evidence_location", "application_match"],
    )
    _write_csv(
        output_dir / "CONFLICT_AUDIT.csv",
        conflict_rows,
        ["request_id", "field", "values", "resolution"],
    )
    _write_csv(
        output_dir / "INGESTION_AUDIT.csv",
        ingestion_rows,
        ["request_id", "source_id", "status", "reason"],
    )
    counts: dict[str, int] = {}
    for result in results:
        counts[result.candidate.confidence.value] = counts.get(result.candidate.confidence.value, 0) + 1
    summary = f"""# BMW Transmission Research Benchmark — Offline Preflight

- Configurations selected: {len(cases)}
- Previous candidate groups: {len({case.candidate_group_id for case in cases})}
- DIRECT results: {counts.get('DIRECT', 0)}
- STRONG results: {counts.get('STRONG', 0)}
- WEAK results: {counts.get('WEAK', 0)}
- Unresolved results: {counts.get('UNRESOLVED', 0)}
- Live external search: NO

The deterministic benchmark selection and result pipeline ran successfully, but
no live search or model provider is configured. These rows are intentionally
unresolved and must not be interpreted as researched component identities.

BMW_TRANSMISSION_BENCHMARK_COMPLETE = NO
PRODUCTION_DB_CHANGED = NO
"""
    (output_dir / "BENCHMARK_SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated BMW transmission research benchmark")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()
    run(args.sample, args.groups, args.output, limit=args.limit)


if __name__ == "__main__":
    main()
