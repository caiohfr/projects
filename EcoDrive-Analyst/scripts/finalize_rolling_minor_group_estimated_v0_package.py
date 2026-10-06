from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package_dir = args.package_dir.resolve(strict=True)
    output = args.output.resolve()
    files = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        files.append({
            "path": path.relative_to(package_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "method": "GROUP_ESTIMATED_ROLLING_MINOR_V0",
        "package_status": "ARTIFACT_COMPLETE",
        "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "classification_recommendation": "AUXILIARY_ANALYTICAL_ADDENDUM",
        "source_sprint12_db_sha256": "8AFAC44888388452E9EBF85F8162B2BFA233DFCC74AC3C924339F235A0EA0330",
        "group_estimated_db_sha256": "DDB10DD46DF499A94761D61DA874D7893CF307FADBF5AE7D10180E6B0E1CAC3E",
        "technical_blockers_if_addendum": 0,
        "owner_acceptance_required": True,
        "prod_promotion_performed": False,
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"file_count": len(files), "output": str(output), "status": manifest["package_status"]}, indent=2))


if __name__ == "__main__":
    main()
