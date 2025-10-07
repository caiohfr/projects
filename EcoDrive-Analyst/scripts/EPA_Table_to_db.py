# etl_simple_epa.py
# Le um Excel (ou CSV), mapeia colunas -> campos do seu DB e insere.
# Ajuste os dicionários MAP_VDE e MAP_FC aos nomes do seu arquivo.
# Conversions são opcionais (CONVERT) e só rodam se você definir para aquele campo.

import argparse
import pandas as pd
from typing import Any, Callable, Dict, Optional

# helpers do seu projeto
from src.vde_core.db import insert_vde, insert_fuelcons

# ----------------- CONVERSÕES (opcionais) -----------------
LB_TO_KG = 0.45359237
LBF_TO_N = 4.4482216153
MPH_TO_KPH = 1.609344

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

CONVERT: Dict[str, Callable[[Any, dict], Any]] = {
    # vde_db
    "inertia_class": lambda v, row: (_as_float(v) or 0) * LB_TO_KG if v not in (None, "") else None,
    "coast_A_N":     lambda v, row: (_as_float(v) or 0) * LBF_TO_N if v not in (None, "") else None,
    "coast_B_N_per_kph": lambda v, row: (_as_float(v) or 0) * (LBF_TO_N / MPH_TO_KPH) if v not in (None, "") else None,
    "coast_C_N_per_kph2": lambda v, row: (_as_float(v) or 0) * (LBF_TO_N / (MPH_TO_KPH**2)) if v not in (None, "") else None,

    # fuelcons_db
    "gco2_per_km": lambda v, row: (_as_float(v) or 0) / MPH_TO_KPH if v not in (None, "") else None,  # g/mi -> g/km
    # Ex.: se tiver MPG ajustado e quiser L/100km:
    # "fuel_l_per_100km": lambda v, row: 235.214583 / _as_float(v) if _as_float(v) else None,
}

# ----------------- MAPEAMENTOS -----------------
# Ajuste os nomes da esquerda (colunas do Excel) para os campos do seu DB (direita)

MAP_VDE = {
    # meta
    "Model Year":                 "year",
    "Vehicle Manufacturer Name":  "make",   # ou "Represented Test Veh Make"
    "Represented Test Veh Model": "model",

    # classificação simples
    "Vehicle Type":               "category",     # use se tiver
    # "Legislation":              "legislation",  # setaremos por CLI (EPA/WLTP/BRA)

    # powertrain básico
    "Test Veh Displacement (L)":  "engine_size_l",
    "Tested Transmission Type":   "transmission_type",

    # massa/inércia/aero
    "Equivalent Test Weight (lbs.)": "inertia_class",  # converter lbs->kg (CONVERT)
    # Se quiser setar mass_kg/cda_m2 fixo ou depois, pode deixar None aqui.

    # coastdown (TARGET)
    "Target Coef A (lbf)":           "coast_A_N",
    "Target Coef B (lbf/mph)":       "coast_B_N_per_kph",
    "Target Coef C (lbf/mph**2)":    "coast_C_N_per_kph2",
}

MAP_FC = {
    "Test Fuel Type Description": "fuel_type",
    "CO2 (g/mi)":                 "gco2_per_km",      # converter g/mi -> g/km (CONVERT)
    # Se tiver consumo (MPG) e quiser L/100km:
    # "RND_ADJ_FE":               "fuel_l_per_100km", # usar conversão em CONVERT
    "# of Gears":                 "gear_count",
    "Axle Ratio":                 "final_drive_ratio",
}

# Campos fixos/implícitos que não vêm do Excel
FIXED_VDE = {
    "legislation": "EPA",            # ajustável por CLI
    "cycle_name": "FTP-75",
    "cycle_source": "standard:EPA",
}

FIXED_FC = {
    "electrification": "None",       # pode-se inferir depois se quiser
    "label_program": "EPA",
}

# ----------------- ETL CORE -----------------
def build_payload(row: dict, mapping: dict, fixed: dict) -> dict:
    payload = {}
    for src_col, dest_field in mapping.items():
        val = row.get(src_col)
        # aplica conversão se existir para este campo
        if dest_field in CONVERT:
            val = CONVERT[dest_field](val, row)
        payload[dest_field] = val if val not in (pd.NA, None, "") else None
    payload.update(fixed)
    return payload

def run_etl(path: str, sheet: Optional[str], legislation: str, dry_run: bool=False):
    # lê Excel ou CSV
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)

    # ajusta legislação fixa se passada por CLI
    if legislation:
        FIXED_VDE["legislation"] = legislation

    inserted = 0
    previews = 0
    for _, r in df.iterrows():
        row = r.to_dict()

        vde_payload = build_payload(row, MAP_VDE, FIXED_VDE)
        # mínimos para vde_db
        vde_payload.setdefault("make", row.get("Represented Test Veh Make") or row.get("Vehicle Manufacturer Name"))
        vde_payload.setdefault("model", row.get("Represented Test Veh Model"))
        vde_payload.setdefault("category", row.get("Vehicle Type") or "Midsize")
        vde_payload.setdefault("mass_kg", None)      # você pode preencher depois
        vde_payload.setdefault("cda_m2", None)       # idem

        fc_payload  = build_payload(row, MAP_FC, FIXED_FC)

        if dry_run and previews < 3:
            print("VDE:", {k: vde_payload.get(k) for k in ("make","model","year","legislation","category","inertia_class","coast_A_N","coast_B_N_per_kph","coast_C_N_per_kph2")})
            print("FC :", {k: fc_payload.get(k) for k in ("fuel_type","gco2_per_km","gear_count","final_drive_ratio")})
            previews += 1
            continue

        vde_id = insert_vde(vde_payload)
        fc_payload["vde_id"] = vde_id
        insert_fuelcons(fc_payload)
        inserted += 1

    if not dry_run:
        print(f"OK: {inserted} registros inseridos.")

# ----------------- CLI -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ETL simples: Excel/CSV EPA -> vde_db & fuelcons_db")
    ap.add_argument("--file", required=True, help="Caminho do Excel/CSV higienizado")
    ap.add_argument("--sheet", default=None, help="Nome da planilha (se Excel)")
    ap.add_argument("--legislation", default="EPA", help="EPA | WLTP | BRA (default: EPA)")
    ap.add_argument("--dry-run", action="store_true", help="Não insere; mostra prévias")
    args = ap.parse_args()

    run_etl(args.file, args.sheet, legislation=args.legislation, dry_run=args.dry_run)
