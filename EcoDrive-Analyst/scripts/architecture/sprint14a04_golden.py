#!/usr/bin/env python3
"""Capture/compare pre-refactor synthetic golden results via existing EcoDrive APIs.

This script is read-only and uses no confidential or DB samples. Capture must
be run against the frozen pre-refactor commit; compare against a later commit.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return None


def cases():
    import numpy as np
    import pandas as pd
    from src.vde_core.roadload import RoadLoadModel, build_request_from_manual_inputs, run_roadload_scenario
    from src.vde_core.roadload.tire_model import calculate_iso_tire_abc_for_single_tire, build_tire_component
    from src.vde_core.vde_calc import compute_vde_series, compute_vde_net
    from src.vde_core.vehicle_demand import (
        AmbientState, RoadloadCoefficients, VehicleDemandRequest,
        calculate_vehicle_demand, build_vehicle_demand_profile, RoadloadBasis,
    )

    def coeff(abc):
        return RoadloadCoefficients(A_N=abc[0],B_N_per_kph=abc[1],C_N_per_kph2=abc[2])

    def request(total,net=None,rrc=None,cda=None):
        return VehicleDemandRequest(
            source_kind="VDE_ONLY",vde_id=14004,fuelcons_id=None,label="SPRINT14A04_SYNTHETIC",
            cycle_name="SYNTHETIC",cycle_source="SYNTHETIC",cycle_version="1",
            test_mass_kg=1600.0,roadload_total=coeff(total),
            roadload_net=coeff(net) if net is not None else None,
            rrc_n_per_kn=rrc,cda_m2=cda,ambient=AmbientState(),provenance={},model_version="14A-04-REFERENCE",
        )

    def summary(s):
        # Use only stable, physical numerical fields, exclude presentation metadata.
        keys=("vde_mj_per_km", "vde_total_MJ", "distance_km",
              "roadload_energy_MJ", "positive_inertial_work_MJ",
              "positive_tractive_energy_MJ", "braking_energy_required_MJ",
              "residual_roadload_energy_MJ")
        return {k:getattr(s,k,None) for k in keys}

    speed=[0.0,50.0,100.0,130.0]
    rl=RoadLoadModel(100.0,0.5,0.03,1500)
    result={
        "roadload_force_N":{str(v):float(rl.force(v)) for v in speed},
        "roadload_power_W":{str(v):float(rl.power(v)) for v in speed},
    }
    # Independent elementary analytical checks: do not freeze a broken source
    # implementation as an unquestioned golden reference.
    assert math.isclose(float(rl.force(100.0)), 450.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(rl.power(100.0)), 12500.0, rel_tol=0.0, abs_tol=1e-9)
    legacy_req=build_request_from_manual_inputs(A=120.,B=0.02,C=0.011,mass_kg=1550.,tire_delta_A=1.5,brake_delta_A=0.5,brake_delta_B=-0.002,brake_delta_C=0.001,parasitic_delta_C=0.002)
    legacy=run_roadload_scenario(legacy_req)
    assert math.isclose(legacy.A, 122.0, abs_tol=1e-10)
    assert math.isclose(legacy.B, 0.018, abs_tol=1e-10)
    assert math.isclose(legacy.C, 0.014, abs_tol=1e-10)
    result["legacy_delta_abc"]={"A":legacy.A,"B":legacy.B,"C":legacy.C,"mass_kg":legacy.mass_kg}

    iso_tire=calculate_iso_tire_abc_for_single_tire({"standard_family":"ISO","rr_n_per_kn":8.0},4000.)
    component=build_tire_component("rolling_front",iso_tire,source="SYNTHETIC_ISO")
    assert math.isclose(component.A, 32.0, rel_tol=0.0, abs_tol=1e-9)
    result["iso_single_tire"]={"A":component.A,"B":component.B,"C":component.C}

    t=np.asarray([0.,1.,2.,3.,4.,5.])
    v=np.asarray([0.,8.,15.,20.,10.,0.])  # m/s; source coefficients use km/h
    cycle=pd.DataFrame({"t":t,"v":v})
    series=compute_vde_series(t,v,120.,0.2,0.01,1600.)
    result["canonical_vde_series"]={key:np.asarray(series[key],dtype=float).tolist() for key in ("F_road_N","F_inertia_N","F_tractive_N","P_W")}
    result["canonical_vde_aggregate"]={k:float(v) for k,v in compute_vde_net(cycle,120.,0.2,0.01,1600.).items() if isinstance(v,(int,float,np.integer,np.floating))}

    total=request((120.,0.2,0.01),(110.,0.18,0.009),rrc=8.)
    demand=calculate_vehicle_demand(total,cycle)
    result["vehicle_demand"]={"total":summary(demand.total_summary),"net":summary(demand.net_summary)}
    profile=build_vehicle_demand_profile(total,cycle,RoadloadBasis.TOTAL)
    result["profile_total"]={"roadload_N":list(profile.authoritative_roadload_force_N),
                              "rolling_N":list(profile.known_rolling_force_N) if profile.known_rolling_force_N is not None else None,
                              "residual_N":list(profile.residual_roadload_force_N)}
    missing=calculate_vehicle_demand(request((120.,0.2,0.01),None),cycle)
    assert missing.net_summary is None, "NET must not be invented from TOTAL"
    result["missing_net_is_none"]=missing.net_summary is None
    # For this synthetic case rolling > TOTAL at zero/low speed: residual remains negative.
    neg=build_vehicle_demand_profile(request((100.,0.,0.),None,rrc=12.),cycle,RoadloadBasis.TOTAL)
    assert all(x < 0 for x in neg.residual_roadload_force_N), "Signed negative residual clipped or missing"
    result["signed_residual_negative_N"]=list(neg.residual_roadload_force_N)
    return result


def compare_values(expected,actual, path="$", abs_tol=1e-9, rel_tol=1e-9):
    failures=[]
    if isinstance(expected,dict) and isinstance(actual,dict):
        for key in sorted(set(expected)|set(actual)):
            if key not in expected or key not in actual:
                failures.append(f"{path}.{key}: key absent in {'expected' if key not in expected else 'actual'}")
            else:
                failures.extend(compare_values(expected[key],actual[key],f"{path}.{key}",abs_tol,rel_tol))
    elif isinstance(expected,list) and isinstance(actual,list):
        if len(expected)!=len(actual):failures.append(f"{path}: length {len(expected)} != {len(actual)}")
        else:
            for i,(a,b) in enumerate(zip(expected,actual)):
                failures.extend(compare_values(a,b,f"{path}[{i}]",abs_tol,rel_tol))
    elif isinstance(expected,bool) or isinstance(actual,bool):
        if type(expected)!=type(actual) or expected!=actual:failures.append(f"{path}: {expected!r} != {actual!r}")
    elif isinstance(expected,(int,float)) and isinstance(actual,(int,float)):
        if not (math.isfinite(float(expected)) and math.isfinite(float(actual)) and math.isclose(expected,actual,rel_tol=rel_tol,abs_tol=abs_tol)):
            failures.append(f"{path}: {expected!r} != {actual!r}")
    elif expected!=actual:
        failures.append(f"{path}: {expected!r} != {actual!r}")
    return failures


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest="command",required=True)
    cap=commands.add_parser("capture")
    cap.add_argument("--out",type=Path,default=Path("artifacts/roadload_14a04/golden_reference.json"))
    cmp=commands.add_parser("compare")
    cmp.add_argument("--reference",type=Path,required=True)
    cmp.add_argument("--out",type=Path,default=Path("artifacts/roadload_14a04/golden_compare.json"))
    cmp.add_argument("--abs-tol",type=float,default=1e-9)
    cmp.add_argument("--rel-tol",type=float,default=1e-9)
    args=parser.parse_args()
    actual=cases()
    meta={"git_sha":sha(),"python":platform.python_version(),"generated_utc":datetime.now(timezone.utc).isoformat(),"data_class":"SYNTHETIC_PUBLIC_ONLY"}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.command=="capture":
        args.out.write_text(json.dumps({"metadata":meta,"golden":actual},indent=2,allow_nan=False),encoding="utf-8")
        print(f"CAPTURED (with independent analytical assertions): {args.out}")
    else:
        reference=json.loads(args.reference.read_text(encoding="utf-8"))["golden"]
        failures=compare_values(reference,actual,abs_tol=args.abs_tol,rel_tol=args.rel_tol)
        payload={"status":"PASS" if not failures else "FAIL","reference":str(args.reference),"run":meta,"tolerances":{"abs":args.abs_tol,"rel":args.rel_tol},"differences":failures}
        args.out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        print(payload["status"],"differences:",len(failures),"->",args.out)
        for s in failures[:20]:print(s,file=sys.stderr)
        if failures:sys.exit(1)

if __name__=="__main__":main()



