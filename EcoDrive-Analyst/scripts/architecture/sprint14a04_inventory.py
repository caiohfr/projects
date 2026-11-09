#!/usr/bin/env python3
"""Sprint 14A-04: static inventory of selected Roadload/VDE symbols.

Run from EcoDrive-Analyst root. This is an AST inventory, not a runtime call graph.
No imports from the project or database operations are performed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

SYMBOLS = {
    "RoadLoadModel", "RoadLoadComponent", "ComponentSet", "EquivalentABC",
    "run_roadload_scenario", "calculate_vehicle_demand", "compute_vde_series",
    "_force_at", "build_tire_component", "build_tire_component_from_result",
    "resolve_roadload_boundaries", "resolve_transmission_boundary",
    "known_rolling_force_N", "known_aero_force_N", "calculate_vehicle_tire_abc",
}


def git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inventory(root: Path):
    records = []
    parse_errors = []
    for scope in ("src", "tests", "etl", "scripts"):
        base = root / scope
        if not base.is_dir():
            continue
        for file in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file))
            except (OSError, SyntaxError, UnicodeDecodeError) as exc:
                parse_errors.append({"file": str(file.relative_to(root)), "error": str(exc)})
                continue
            aliases = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        if a.name in SYMBOLS:
                            aliases[a.asname or a.name] = a.name
                            records.append({"file":str(file.relative_to(root)),"line":node.lineno,"kind":"import_from","symbol":a.name,"expression":a.asname or a.name})
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if "roadload" in a.name or "vehicle_demand" in a.name or "vde_calc" in a.name:
                            records.append({"file":str(file.relative_to(root)),"line":node.lineno,"kind":"module_import","symbol":a.name,"expression":a.asname or a.name})
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = None
                    if isinstance(node.func, ast.Name):
                        name = aliases.get(node.func.id, node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name in SYMBOLS:
                        records.append({"file":str(file.relative_to(root)),"line":node.lineno,"kind":"call","symbol":name,"expression":ast.unparse(node.func)})
                elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in SYMBOLS:
                    records.append({"file":str(file.relative_to(root)),"line":node.lineno,"kind":"definition","symbol":node.name,"expression":node.name})
    records.sort(key=lambda x:(x["symbol"],x["file"],x["line"],x["kind"]))
    return records, parse_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("artifacts/roadload_14a04/inventory.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    records, errs = inventory(root)
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status":"STATIC_AST_ONLY", "generated_utc":datetime.now(timezone.utc).isoformat(),"commit":git_sha(root),"root_scope":["src","tests","etl","scripts"], "callers":records,"parse_errors":errs,"limitations":["Dynamic imports, getattr, delegation, notebook code, and string-based references not resolved", "Imported aliases are matched locally; cross-module dispatch is not inferred"]}
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    csv_file = out.with_suffix(".csv")
    with csv_file.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["file","line","kind","symbol","expression"])
        writer.writeheader();writer.writerows(records)
    print(f"AST inventory: {len(records)} records, {len(errs)} parse errors -> {out}")

if __name__ == "__main__":
    main()



