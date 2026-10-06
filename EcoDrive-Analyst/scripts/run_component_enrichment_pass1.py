#!/usr/bin/env python3
"""Run deterministic Sprint 12 Component Enrichment Pass 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vde_core.component_enrichment_pass1 import execute_pass1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-db", type=Path, help="Original candidate used to prove temp-copy isolation")
    parser.add_argument("--write", action="store_true", help="Persist to the supplied DB; default is read-only dry-run")
    parser.add_argument("--vde-id", action="append", type=int, default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        result = execute_pass1(
            args.db,
            args.output_dir,
            write=args.write,
            source_db_path=args.source_db,
            vde_ids=set(args.vde_id) if args.vde_id else None,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
