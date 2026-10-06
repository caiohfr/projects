from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def consolidate_research_attempts(
    attempts: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in attempts:
        row = dict(raw)
        by_request[str(row["request_id"])].append(row)
    final_rows: list[dict[str, Any]] = []
    for request_id, rows in by_request.items():
        rows.sort(key=lambda row: int(row.get("attempt_number", 0)))
        final = dict(rows[-1])
        final["attempt_count"] = len(rows)
        final["first_status"] = rows[0].get("status", "")
        final["final_status"] = final.get("status", "")
        final["retry_reason"] = ";".join(
            str(row.get("retry_reason", ""))
            for row in rows[1:]
            if row.get("retry_reason")
        )
        final_rows.append(final)
    return tuple(sorted(final_rows, key=lambda row: str(row["request_id"])))
