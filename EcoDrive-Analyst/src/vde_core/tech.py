
from copy import deepcopy

def apply_tech_effects(params: dict, selected_ids: list[str], catalog: list[dict]):
    p = deepcopy(params); applied = []
    idx = {t["id"]: t for t in catalog}
    for tid in selected_ids:
        t = idx.get(tid); 
        if not t: continue
        tgt, mode, val = t["target"], t["mode"], t["value"]
        if tgt not in p: continue
        before = p[tgt]; p[tgt] = before*val if mode=="mult" else before+val
        applied.append({"id": tid, "target": tgt, "from": before, "to": p[tgt], "mode": mode, "value": val})
    return p, applied

def estimate_eta_pt(engine_type: str, electrif: str, trans: str) -> float:
    base = 0.24  # ICE proxy
    electrif_mult = {"None":1.00, "MHEV (48V)":1.05, "HEV":1.25, "PHEV":1.30, "BEV":3.60}.get(electrif,1.0)
    trans_mult    = {"AT (auto)":1.00,"DCT":1.03,"CVT":1.02,"MT":1.00}.get(trans,1.0)
    return base * electrif_mult * trans_mult
