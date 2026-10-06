#!/usr/bin/env python3
"""Run the Sprint 12 full-fleet component population temp-DB workflow."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vde_core.component_population_materialization import execute_component_population_materialization


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="Canonical candidate DB; opened read-only")
    parser.add_argument("--temp-db", type=Path, help="New real SQLite copy used for all writes")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--legacy-db", type=Path, default=ROOT / "data/db/archive/eco_drive_legacy_pre_sprint12.db")
    parser.add_argument("--component-catalog", type=Path, default=ROOT / "inputs/synthetic_reference_catalog.csv")
    parser.add_argument("--tire-zip", type=Path, default=ROOT / "inputs/EcoDrive_Synthetic_Tire_DB_v1.zip")
    args = parser.parse_args()
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_db = args.temp_db or ROOT / "artifacts/temp_component_population" / f"eco_drive_component_population_{suffix}.db"
    result = execute_component_population_materialization(
        args.db, temp_db, args.legacy_db, args.component_catalog, args.tire_zip, args.output_dir
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
