def apply_tech_effects(params: dict, selected_ids: list[str], catalog: list[dict]):
    """
    Apply technology effects to road-load/mass parameters.
    Placeholder implementation.
    """
    return params, []

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

