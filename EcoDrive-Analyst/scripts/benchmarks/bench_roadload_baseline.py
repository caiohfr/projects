#!/usr/bin/env python3
"""Read-only Sprint 14A-04 microbenchmark for existing Roadload/VDE APIs.

This script measures in-process, synthetic workloads, NOT Streamlit latency,
SQL query count or Tire/Component build-up. The output marks these gaps.
Run from the EcoDrive-Analyst root in the project's own Python environment.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def git_sha():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return None


def stats(op, repeats):
    # Time without tracing, separately sample traceable Python allocations.
    wall=[];cpu=[]
    op()  # warmup; not part of measured distribution
    for _ in range(repeats):
        a=time.perf_counter_ns();c=time.process_time_ns()
        op()
        cpu.append((time.process_time_ns()-c)/1e6)
        wall.append((time.perf_counter_ns()-a)/1e6)
    tracemalloc.start();tracemalloc.reset_peak()
    op()
    _current,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall.sort();cpu.sort()
    def percentile(values,p):
        return values[max(0,min(len(values)-1,int((len(values)-1)*p+0.5)))]
    return {"p50_wall_ms":round(statistics.median(wall),6),"p95_wall_ms":round(percentile(wall,0.95),6),"p50_cpu_ms":round(statistics.median(cpu),6),"p95_cpu_ms":round(percentile(cpu,0.95),6),"tracemalloc_peak_bytes":peak,"repeats":repeats,"warmup_calls":1,"memory_note":"tracemalloc tracks traced allocations, not full NumPy RSS"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--quick",action="store_true",help="Smaller sanity sample, not a release baseline")
    parser.add_argument("--repeats",type=int,default=20)
    parser.add_argument("--out",type=Path,default=Path("artifacts/roadload_14a04/bench.json"))
    args=parser.parse_args()
    if args.repeats<2: parser.error("--repeats must be >= 2")
    try:
        import numpy as np
        from src.vde_core.roadload.models import RoadLoadModel
        from src.vde_core.roadload import build_request_from_manual_inputs,run_roadload_scenario
        from src.vde_core.vde_calc import compute_vde_series
    except Exception as exc:
        raise SystemExit(f"Requires an EcoDrive checkout and its dependencies: {exc}") from exc
    levels=[1,10,100] if args.quick else [1,10,100,1000]
    samples=[100,1000] if args.quick else [100,1000,10000]
    records=[]
    model=RoadLoadModel(A=100.0,B=0.5,C=0.03,mass_kg=1500.0)
    def record(method,workload,op):
        r={"method":method,"workload":workload,**stats(op,args.repeats)};records.append(r)
        print(f"{method:28s} {str(workload):22s} p50={r['p50_wall_ms']:.4f} ms")
    record("roadload_force_scalar",{"speed_kph":100},lambda:model.force(100.0))
    record("roadload_power_scalar",{"speed_kph":100},lambda:model.power(100.0))
    for n in samples:
        speed=np.linspace(0,130,n)
        t=np.arange(n,dtype=float)
        speed_mps=speed/3.6
        record("roadload_force_vector",{"n":n},lambda s=speed:model.force(s))
        record("roadload_power_vector",{"n":n},lambda s=speed:model.power(s))
        record("canonical_vde_series",{"n":n},lambda t=t,v=speed_mps:compute_vde_series(t,v,100.,0.5,0.03,1500.))
    for count in levels:
        # Batched independent scenarios; no DB access, no parallel execution.
        models=[RoadLoadModel(100.0+i/10000,0.5,0.03,1500) for i in range(count)]
        record("model_batch_scalar",{"vehicles":count},lambda ms=models:[m.force(100.0) for m in ms])
        requests=[build_request_from_manual_inputs(A=100.0+i/10000,B=0.5,C=0.03,mass_kg=1500.0) for i in range(count)]
        record("legacy_scenario_batch",{"vehicles":count},lambda rs=requests:[run_roadload_scenario(req) for req in rs])
    payload={"status":"EXECUTED_MICROBENCHMARK","generated_utc":datetime.now(timezone.utc).isoformat(),"git_sha":git_sha(),"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"cpu_count":os.cpu_count(),"numpy":np.__version__,"scope":"Current existing RoadLoadModel/legacy scenario/core VDE, synthetic in-process input","warmup":"one per timed operation","sql_query_count":{"status":"NOT_MEASURED","reason":"No repository integration or DB instrumentation in this script"},"ui_latency":{"status":"NOT_MEASURED"},"component_breakdown":{"status":"NOT_IMPLEMENTED_IN_CURRENT_MODEL","note":"Do not label mock k-loop as production implementation"},"records":records}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__=="__main__":
    main()



