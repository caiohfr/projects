from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vde_core import db as db_module
from src.vde_core.database_management_contract import ActorContext
from src.vde_core.vde_net_total_normalization import (
    apply_vde_net_total_normalization,
    preview_vde_net_total_normalization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview (default) or apply the Package 7G VDE TOTAL/NET normalization: "
            "moves a legacy row's vde_net_mj_per_km value into vde_total_mj_per_km "
            "when record_origin=LEGACY and vde_total_mj_per_km is missing, and flags "
            "ambiguous net-only rows from other origins as review_status=REVIEW_REQUIRED."
        )
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"SQLite path to inspect. Defaults to {db_module.DB_PATH_ENV_VAR} or {db_module.DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the proposed changes. Without this flag, only a preview is printed.",
    )
    parser.add_argument(
        "--reason",
        default="package_7g_vde_net_total_normalization",
        help="Reason recorded in data_change_log for every applied row change.",
    )
    parser.add_argument(
        "--actor-id",
        default="local_admin",
        help="Actor id recorded in data_change_log.",
    )
    return parser.parse_args()


def _print_preview(preview) -> None:
    print(f"Total rows inspected: {preview.total_rows_inspected}")
    for status, count in sorted(preview.counts_by_status.items()):
        print(f"  {status}: {count}")
    print(f"\nRows that would change (legacy TOTAL stored as NET): {len(preview.legacy_total_in_net_changes)}")
    for change in preview.legacy_total_in_net_changes:
        print(
            f"  vde_id={change.vde_id} old_total={change.old_total} old_net={change.old_net} "
            f"-> new_total={change.new_total} new_net={change.new_net} reason={change.reason}"
        )
    print(f"\nRows flagged for review (ambiguous, not corrected): {len(preview.ambiguous_review_ids)}")
    for vde_id in preview.ambiguous_review_ids:
        print(f"  vde_id={vde_id}")


def main() -> int:
    args = parse_args()
    preview = preview_vde_net_total_normalization(db_path=args.db)
    _print_preview(preview)

    if not args.apply:
        print("\nPreview only. Re-run with --apply to write these changes.")
        return 0

    actor = ActorContext(actor_id=args.actor_id, actor_role="admin")
    result = apply_vde_net_total_normalization(
        preview, actor, reason=args.reason, db_path=args.db
    )
    print(
        f"\nApplied. Rows normalized: {result.rows_normalized}. "
        f"Rows flagged for review: {result.rows_flagged_for_review}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
