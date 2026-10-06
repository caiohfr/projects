#!/usr/bin/env python3
"""Materialize frozen component population into STAGING and export research gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vde_core.canonical_component_materialization import execute_canonical_materialization_and_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/canonical_component_population")
    parser.add_argument("--legacy-db", type=Path, default=ROOT / "data/db/archive/eco_drive_legacy_pre_sprint12.db")
    parser.add_argument("--component-catalog", type=Path, default=ROOT / "inputs/synthetic_reference_catalog.csv")
    parser.add_argument("--tire-zip", type=Path, default=ROOT / "inputs/EcoDrive_Synthetic_Tire_DB_v1.zip")
    args = parser.parse_args()
    result = execute_canonical_materialization_and_handoff(
        ROOT, args.output_dir, args.legacy_db, args.component_catalog, args.tire_zip
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
