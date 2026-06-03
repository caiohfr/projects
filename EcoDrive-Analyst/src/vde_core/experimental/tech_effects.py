from copy import deepcopy


def apply_tech_effects(params: dict, selected_ids: list[str], catalog: list[dict]):
    updated = deepcopy(params)
    applied = []
    index = {item["id"]: item for item in catalog}

    for tech_id in selected_ids:
        item = index.get(tech_id)
        if not item:
            continue

        target = item["target"]
        mode = item["mode"]
        value = item["value"]
        if target not in updated:
            continue

        before = updated[target]
        updated[target] = before * value if mode == "mult" else before + value
        applied.append(
            {
                "id": tech_id,
                "target": target,
                "from": before,
                "to": updated[target],
                "mode": mode,
                "value": value,
            }
        )

    return updated, applied


def estimate_eta_pt(engine_type: str, electrif: str, trans: str) -> float:
    base = 0.24
    electrif_mult = {
        "None": 1.00,
        "MHEV (48V)": 1.05,
        "HEV": 1.25,
        "PHEV": 1.30,
        "BEV": 3.60,
    }.get(electrif, 1.0)
    trans_mult = {
        "AT (auto)": 1.00,
        "DCT": 1.03,
        "CVT": 1.02,
        "MT": 1.00,
    }.get(trans, 1.0)
    return base * electrif_mult * trans_mult
