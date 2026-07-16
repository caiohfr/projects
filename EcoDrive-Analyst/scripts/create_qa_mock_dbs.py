from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vde_core.qa_mock_data import (
    DEFAULT_QA_DB_PATH,
    QA_DATA_DIR,
    build_seed_manifest,
    is_safe_qa_db_path,
    seed_qa_database,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic QA SQLite databases for VDE Setup v2.2."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_QA_DB_PATH),
        help=f"Output SQLite path. Defaults to {DEFAULT_QA_DB_PATH}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing QA database file inside the QA data directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"QA database already exists: {output_path}\n"
            "Re-run with --overwrite to replace it."
        )
    if args.overwrite and not is_safe_qa_db_path(output_path):
        raise SystemExit(
            f"Refusing to overwrite a non-QA database path: {output_path}\n"
            f"Choose a path under {QA_DATA_DIR}."
        )

    created_path = seed_qa_database(output_path, overwrite=args.overwrite)
    manifest = build_seed_manifest()
    print(f"Created deterministic QA database: {created_path}")
    print(f"Baselines seeded: {len(manifest['baselines'])}")
    print(f"Tires seeded: {len(manifest['tires'])}")
    print(f"Golden scenario: {manifest['golden_scenario']['baseline_qa_id']} + {manifest['golden_scenario']['tire_qa_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
