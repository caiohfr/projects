#!/usr/bin/env python3
"""Run the read-only Sprint 12 Pass 1C.2 vector/combinatorial pilot."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vde_core.component_vector_search import execute_vector_search


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--legacy-db", type=Path, default=ROOT / "data/db/archive/eco_drive_legacy_pre_sprint12.db")
    parser.add_argument("--component-catalog", type=Path, default=ROOT / "inputs/synthetic_reference_catalog.csv")
    parser.add_argument("--tire-zip", type=Path, default=ROOT / "inputs/EcoDrive_Synthetic_Tire_DB_v1.zip")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "artifacts/component_enrichment_pass1c2" / datetime.now().strftime("%Y%m%d_%H%M%S")
    result = execute_vector_search(args.db, args.legacy_db, args.component_catalog, args.tire_zip, output)
    print(json.dumps({"output_dir": str(output.resolve()), **result}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
