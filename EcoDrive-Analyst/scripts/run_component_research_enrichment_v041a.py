"""Generate the v0.4.1a numeric-cleanup and final frozen BMW benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_component_research_enrichment_v041 import DEFAULT_V04, ROOT, run


DEFAULT_OUTPUT = ROOT / "artifacts/components/component_research_v041a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v04-dir", type=Path, default=DEFAULT_V04)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metrics = run(
        v04_dir=args.v04_dir,
        output=args.output,
        artifact_suffix="V041A",
        component_version="0.4.1a",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
