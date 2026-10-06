"""Build the explicit, interview-friendly Sprint 12F notebook MVP."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


PATHS = """from pathlib import Path
import pandas as pd

search_roots = [Path.cwd(), *Path.cwd().parents]
REPO_ROOT = next(path for path in search_roots if (path / "AGENTS.md").exists() and (path / "etl").exists())
"""


def build() -> None:
    books: dict[str, list[dict]] = {}

    books["12F_01_epa_ingest_and_grain.ipynb"] = [
        md("""# 12F-01 — EPA source ingest and grain

**Question:** What exactly is in the EPA source, and what does one row represent?

This notebook is exploratory. Production ETL remains authoritative."""),
        md("## Imports and portable paths"),
        code(PATHS + """EPA_FILE = REPO_ROOT / "etl/data/raw/epa_testcar/epa_testcar_2026_raw.xlsx"
assert EPA_FILE.exists(), EPA_FILE
"""),
        md("""## Load the EPA source

No cleaning yet. First I want to see the workbook as delivered."""),
        code("""epa = pd.read_excel(EPA_FILE, sheet_name="Sheet1", engine="openpyxl")
print(f"{epa.shape[0]:,} rows × {epa.shape[1]} columns")
"""),
        md("## Shape, model years, and relevant columns"),
        code("""year_counts = epa["Model Year"].value_counts(dropna=False).sort_index()
year_counts
"""),
        code("""identity_columns = [
    "Model Year", "Represented Test Veh Make", "Represented Test Veh Model",
    "Actual Tested Testgroup", "Test Number", "Test Vehicle ID",
    "Test Veh Configuration #", "Test Procedure Description",
]
assert not [column for column in identity_columns if column not in epa.columns]
epa[identity_columns].head(8)
"""),
        md("""## First grain counts

Make+Model+Year is a commercial identity, not necessarily one test or technical state."""),
        code("""mmy = ["Model Year", "Represented Test Veh Make", "Represented Test Veh Model"]
test_key = mmy + ["Actual Tested Testgroup", "Test Number"]
config_key = test_key + ["Test Vehicle ID", "Test Veh Configuration #"]

grain_counts = pd.Series({
    "source_rows": len(epa),
    "make_model_year": len(epa[mmy].drop_duplicates()),
    "test_key": len(epa[test_key].drop_duplicates()),
    "test_and_configuration_key": len(epa[config_key].drop_duplicates()),
})
grain_counts
"""),
        md("## Real multi-state examples"),
        code("""mmy_states = (
    epa.groupby(mmy, dropna=False)
       .agg(source_rows=("Model Year", "size"),
            test_groups=("Actual Tested Testgroup", "nunique"),
            configurations=("Test Veh Configuration #", "nunique"))
       .reset_index()
)
examples = mmy_states.query("test_groups > 1 or configurations > 1").sort_values("source_rows", ascending=False)
examples.head(10)
"""),
        code("""example = examples.iloc[0]
same_vehicle = (
    epa["Model Year"].eq(example["Model Year"])
    & epa["Represented Test Veh Make"].eq(example["Represented Test Veh Make"])
    & epa["Represented Test Veh Model"].eq(example["Represented Test Veh Model"])
)
epa.loc[same_vehicle, identity_columns].head(20)
"""),
        md("""## What did we learn?

- One source row is a reported test/result row, not automatically one vehicle or one RUN.
- Make+Model+Year collapses distinct test groups and configurations.
- Next: keep this grain distinction visible while exploring non-BEV fuel and CO2 results."""),
    ]

    books["12F_02_epa_nonbev_eda.ipynb"] = [
        md("""# 12F-02 — EPA non-BEV EDA

**Question:** What patterns and quality issues exist in conventional fuel-side results?"""),
        code(PATHS + """EPA_FILE = REPO_ROOT / "etl/data/raw/epa_testcar/epa_testcar_2026_raw.xlsx"
epa = pd.read_excel(EPA_FILE, sheet_name="Sheet1", engine="openpyxl")
"""),
        md("## Keep electric and hydrogen semantics out of this fuel-side view"),
        code("""fuel_text = epa["Test Fuel Type Description"].fillna("").astype(str)
nonbev = epa[~fuel_text.str.contains("Electricity|Hydrogen", case=False, regex=True)].copy()
nonbev["RND_ADJ_FE"] = pd.to_numeric(nonbev["RND_ADJ_FE"], errors="coerce")
valid_mpg = nonbev["RND_ADJ_FE"].between(0, 500, inclusive="neither")
nonbev["fuel_l_per_100km"] = float("nan")
nonbev.loc[valid_mpg, "fuel_l_per_100km"] = 235.214583 / nonbev.loc[valid_mpg, "RND_ADJ_FE"]
print(f"{len(nonbev):,} source rows; {valid_mpg.sum():,} usable MPG observations")
"""),
        md("## Population and missingness"),
        code("""nonbev["Test Fuel Type Description"].value_counts(dropna=False).head(12)
"""),
        code("""review_columns = ["Equivalent Test Weight (lbs.)", "CO2 (g/mi)", "RND_ADJ_FE", "fuel_l_per_100km",
                  "Target Coef A (lbf)", "Target Coef B (lbf/mph)", "Target Coef C (lbf/mph**2)"]
nonbev[review_columns].apply(pd.to_numeric, errors="coerce").describe().T
"""),
        md("## Model-year summary"),
        code("""year_summary = (
    nonbev.assign(co2_g_mi=pd.to_numeric(nonbev["CO2 (g/mi)"], errors="coerce"))
          .groupby("Model Year")
          .agg(source_rows=("Model Year", "size"),
               median_l_100km=("fuel_l_per_100km", "median"),
               median_co2_g_mi=("co2_g_mi", "median"))
)
year_summary
"""),
        md("## Does test mass move with fuel consumption?"),
        code("""import matplotlib.pyplot as plt

plot_data = nonbev.assign(mass_lb=pd.to_numeric(nonbev["Equivalent Test Weight (lbs.)"], errors="coerce"))
plot_data = plot_data.dropna(subset=["mass_lb", "fuel_l_per_100km"])
ax = plot_data.plot.scatter(x="mass_lb", y="fuel_l_per_100km", alpha=0.15, figsize=(7, 4))
ax.set(title="EPA fuel-side observations", xlabel="Equivalent test weight (lb)", ylabel="Fuel consumption (L/100 km)")
plt.show()
"""),
        md("""## What did we learn?

This is source-row EDA, so repeated tests remain visible. The next modelling steps must use the closed RUN/VDE grain rather than averaging these rows blindly."""),
    ]

    books["12F_03_epa_bev_energy_eda.ipynb"] = [
        md("""# 12F-03 — EPA BEV electric-energy EDA

**Question:** Which electric-consumption values are trustworthy, and why?"""),
        code(PATHS + """SANITY_FILE = REPO_ROOT / "etl/data/processed/sprint_12e3_electric_unit_audit/electric_energy_sanity_dataset.csv"
electric = pd.read_csv(SANITY_FILE, low_memory=False)
electric.shape
"""),
        md("## Separate deterministic EPA rows from unresolved and Hydrogen cases"),
        code("""electric["sprint_12e3a_disposition"].value_counts(dropna=False)
"""),
        code("""resolved = electric[
    electric["sprint_12e3a_disposition"].eq("SOURCE_ROW_RESOLVED_KWH_PER_100MI")
    & electric["source_system"].eq("EPA_TESTCAR_2014_PRESENT")
].copy()
resolved["raw_value"] = pd.to_numeric(resolved["raw_value"], errors="raise")
resolved["canonical_wh_per_km"] = pd.to_numeric(resolved["canonical_wh_per_km"], errors="raise")
print(f"{len(resolved)} resolved EPA source records")
"""),
        md("## Reproduce the unit conversion visibly"),
        code("""MI_TO_KM = 1.609344
resolved["notebook_wh_per_km"] = resolved["raw_value"] * 1000 / (100 * MI_TO_KM)
resolved["difference_wh_per_km"] = resolved["notebook_wh_per_km"] - resolved["canonical_wh_per_km"]
resolved[["raw_value", "canonical_wh_per_km", "notebook_wh_per_km", "difference_wh_per_km"]].head()
"""),
        code("""assert resolved["difference_wh_per_km"].abs().max() < 1e-9
assert not resolved["electrification"].eq("FCEV").any()
resolved["canonical_wh_per_km"].describe()
"""),
        md("## Distribution and extremes"),
        code("""import matplotlib.pyplot as plt

ax = resolved["canonical_wh_per_km"].plot.hist(bins=25, figsize=(7, 4))
ax.set(title="Resolved EPA electric source observations", xlabel="Wh/km")
plt.show()
"""),
        code("""resolved.nlargest(8, "canonical_wh_per_km")[["make", "model", "model_year", "raw_value", "canonical_wh_per_km"]]
"""),
        md("""## What did we learn?

The 527 exact source-row matches are deterministic at the unit layer. FCEV zeros and non-matching/mixed EPA contexts remain separate; unit closure alone does not create FuelCons."""),
    ]

    books["12F_04_wltp_jrc_ingest_and_grain.ipynb"] = [
        md("""# 12F-04 — WLTP/JRC ingest and grain

**Question:** What does this anonymized technical source represent?"""),
        code(PATHS + """JRC_FILE = REPO_ROOT / "etl/data/raw/wltp_jrc/Data_PV_fleet_2021_EU_PYCSIS.xlsx"
jrc = pd.read_excel(JRC_FILE, sheet_name="Sheet1", engine="openpyxl")
print(f"{jrc.shape[0]:,} rows × {jrc.shape[1]} columns")
"""),
        md("## Source-scoped identity and powertrain flags"),
        code("""jrc[["OEM anon", "Model anon", "Input type", "Fuel type", "is_plugin", "is_hybrid", "is_electric"]].head(10)
"""),
        code("""flag_columns = ["is_plugin", "is_hybrid", "is_electric"]
assert all(pd.api.types.is_bool_dtype(jrc[column]) for column in flag_columns)
jrc[flag_columns].fillna(False).astype(bool).value_counts()
"""),
        md("## Engineering-field coverage"),
        code("""engineering_fields = [
    "Vehicle mass (WLTP) [kg]", "wltp|f0 [N]", "wltp|f1 [N/(km/h)]", "wltp|f2 [N/(km/h)2]",
    "Gear box type", "N gears", "Tyre code", "Declared average CO2 emissions value (OEM) [g/km]",
    "Declared electric consumption value (OEM) [Wh/km]", "Electric range (OEM) [km]",
]
coverage = pd.DataFrame({"non_null": jrc[engineering_fields].notna().sum(), "coverage_pct": jrc[engineering_fields].notna().mean() * 100})
coverage.sort_values("coverage_pct", ascending=False)
"""),
        md("## First grain check"),
        code("""grain = pd.Series({
    "source_rows": len(jrc),
    "anonymous_oem_model": len(jrc[["OEM anon", "Model anon"]].drop_duplicates()),
    "oem_model_input_type": len(jrc[["OEM anon", "Model anon", "Input type"]].drop_duplicates()),
    "pycsis_runs": jrc["pycsis_run"].nunique(dropna=True),
})
grain
"""),
        md("""## What did we learn?

Identity is deliberately anonymized and source-scoped. The source is rich in mass, roadload, gearbox, tire, and declared/simulated result context, but a row should not be promoted to a public commercial identity."""),
    ]

    books["12F_05_wltp_jrc_nonbev_eda.ipynb"] = [
        md("""# 12F-05 — WLTP/JRC non-BEV EDA

**Question:** What engineering patterns appear in the conventional population?"""),
        code(PATHS + """JRC_FILE = REPO_ROOT / "etl/data/raw/wltp_jrc/Data_PV_fleet_2021_EU_PYCSIS.xlsx"
jrc = pd.read_excel(JRC_FILE, sheet_name="Sheet1", engine="openpyxl")
assert pd.api.types.is_bool_dtype(jrc["is_electric"]), jrc["is_electric"].dtype
nonbev = jrc[~jrc["is_electric"].fillna(False).astype(bool)].copy()
print(f"{len(nonbev)} non-BEV source records")
"""),
        md("## Numeric engineering view"),
        code("""fields = ["Vehicle mass (WLTP) [kg]", "wltp|f0 [N]", "wltp|f1 [N/(km/h)]", "wltp|f2 [N/(km/h)2]",
          "Declared average CO2 emissions value (OEM) [g/km]"]
numeric = nonbev[fields].apply(pd.to_numeric, errors="coerce")
numeric.describe().T
"""),
        code("""nonbev["Fuel type"].value_counts(dropna=False)
"""),
        md("## Mass, roadload, and declared CO2"),
        code("""import matplotlib.pyplot as plt

plot_data = numeric.dropna(subset=["Vehicle mass (WLTP) [kg]", "Declared average CO2 emissions value (OEM) [g/km]"])
ax = plot_data.plot.scatter(x="Vehicle mass (WLTP) [kg]", y="Declared average CO2 emissions value (OEM) [g/km]", alpha=0.55, figsize=(7, 4))
ax.set(title="JRC non-BEV source records")
plt.show()
"""),
        code("""numeric.corr(numeric_only=True)["Declared average CO2 emissions value (OEM) [g/km]"].sort_values(ascending=False)
"""),
        md("""## What did we learn?

This small technical sample supports engineering exploration, not population claims. Correlation helps choose questions; it does not establish causality or a canonical model."""),
    ]

    books["12F_06_wltp_jrc_bev_eda.ipynb"] = [
        md("""# 12F-06 — WLTP/JRC BEV EDA

**Question:** How does the electric JRC population behave, and which values are directly comparable?"""),
        code(PATHS + """JRC_FILE = REPO_ROOT / "etl/data/raw/wltp_jrc/Data_PV_fleet_2021_EU_PYCSIS.xlsx"
jrc = pd.read_excel(JRC_FILE, sheet_name="Sheet1", engine="openpyxl")
assert pd.api.types.is_bool_dtype(jrc["is_electric"]), jrc["is_electric"].dtype
bev = jrc[jrc["is_electric"].fillna(False).astype(bool)].copy()
energy_field = "Declared electric consumption value (OEM) [Wh/km]"
bev[energy_field] = pd.to_numeric(bev[energy_field], errors="coerce")
print(f"{len(bev)} electric rows; {bev[energy_field].notna().sum()} declared Wh/km values")
"""),
        md("## Explicit source-unit distribution"),
        code("""bev[[energy_field, "Electric range (OEM) [km]", "Vehicle mass (WLTP) [kg]"]].describe().T
"""),
        md("## Declared versus simulated context"),
        code("""simulated = "Declared electric consumption value (Simulated) [Wh/km]"
comparison = bev[["OEM anon", "Model anon", energy_field, simulated]].dropna().copy()
comparison["simulated_minus_oem_wh_km"] = comparison[simulated] - comparison[energy_field]
comparison.head(10)
"""),
        md("## Does consumption move with mass?"),
        code("""import matplotlib.pyplot as plt

plot_data = bev.assign(mass_kg=pd.to_numeric(bev["Vehicle mass (WLTP) [kg]"], errors="coerce")).dropna(subset=["mass_kg", energy_field])
ax = plot_data.plot.scatter(x="mass_kg", y=energy_field, alpha=0.65, figsize=(7, 4))
ax.set(title="JRC electric source records")
plt.show()
"""),
        md("""## What did we learn?

The OEM field is directly labelled Wh/km. Declared, simulated, and real-world fields remain different result contexts and must not be silently averaged."""),
    ]

    books["12F_07_vehicle_demand_model.ipynb"] = [
        md("""# 12F-07 — Vehicle Demand engineering model

**Question:** How do roadload coefficients become steady-speed wheel demand?"""),
        code(PATHS + """import sqlite3
import numpy as np

CANONICAL_DB = REPO_ROOT / "etl/data/staging/sprint_12e2_epa_fuelcons/eco_drive_canonical_epa_fuelcons.db"
uri = CANONICAL_DB.resolve().as_uri() + "?mode=ro"
connection = sqlite3.connect(uri, uri=True)
"""),
        md("## Load canonical VDE coefficients read-only"),
        code("""query = '''
SELECT id AS vde_id, year, category, baseline_mass_kg,
       coast_A_N, coast_B_N_per_kph, coast_C_N_per_kph2,
       vde_total_mj_per_km, provenance_json
FROM vde
WHERE coast_A_N IS NOT NULL AND coast_B_N_per_kph IS NOT NULL AND coast_C_N_per_kph2 IS NOT NULL
ORDER BY id
LIMIT 100
'''
vdes = pd.read_sql_query(query, connection)
vdes.head()
"""),
        md("""## Make the roadload equation visible

At steady speed, `F = A + Bv + Cv²`. Dividing force by 3.6 converts N to Wh/km over one kilometre."""),
        code("""assert not vdes.empty, "No VDE rows available for the notebook example."
selected = vdes.iloc[0]
speed_kph = np.arange(0, 131, 10)
force_n = selected["coast_A_N"] + selected["coast_B_N_per_kph"] * speed_kph + selected["coast_C_N_per_kph2"] * speed_kph**2

curve = pd.DataFrame({"speed_kph": speed_kph, "roadload_force_n": force_n})
curve["steady_speed_wh_per_km"] = curve["roadload_force_n"] / 3.6
curve
"""),
        md("## A small coefficient sensitivity"),
        code("""curve["plus_10pct_A_wh_per_km"] = (1.10 * selected["coast_A_N"] + selected["coast_B_N_per_kph"] * speed_kph + selected["coast_C_N_per_kph2"] * speed_kph**2) / 3.6
curve["plus_10pct_C_wh_per_km"] = (selected["coast_A_N"] + selected["coast_B_N_per_kph"] * speed_kph + 1.10 * selected["coast_C_N_per_kph2"] * speed_kph**2) / 3.6
curve.tail()
"""),
        code("""import matplotlib.pyplot as plt

ax = curve.plot(x="speed_kph", y=["steady_speed_wh_per_km", "plus_10pct_A_wh_per_km", "plus_10pct_C_wh_per_km"], figsize=(7, 4))
ax.set(ylabel="Wh/km", title=f"Illustrative steady-speed demand: VDE {selected['vde_id']}")
plt.show()
connection.close()
"""),
        md("""## What did we learn?

A matters at every speed; C grows quadratically. This steady-speed slice is explanatory and is not a replacement for the production drive-cycle integration or TOTAL/NET contract."""),
    ]

    books["12F_08_fuelcons_modelling.ipynb"] = [
        md("""# 12F-08 — FuelCons modelling and pacification

**Question:** How do RUN results become a comparable FuelCons record?"""),
        code(PATHS + """import sqlite3

CANONICAL_DB = REPO_ROOT / "etl/data/staging/sprint_12e2_epa_fuelcons/eco_drive_canonical_epa_fuelcons.db"
connection = sqlite3.connect(CANONICAL_DB.resolve().as_uri() + "?mode=ro", uri=True)
"""),
        md("## Load EPA two-cycle FuelCons records"),
        code("""query = '''
SELECT id AS fuelcons_id, vde_id, electrification, fuel_type,
       fuel_ftp75_l_per_100km AS city_l_100km,
       fuel_hwfet_l_per_100km AS highway_l_100km,
       fuel_l_per_100km AS canonical_combined_l_100km,
       comparison_basis, provenance_json
FROM fuelcons
WHERE record_origin='EPA_RECONSTRUCTED'
  AND comparison_basis='EPA_LABEL_2_CYCLE'
  AND fuel_ftp75_l_per_100km IS NOT NULL
  AND fuel_hwfet_l_per_100km IS NOT NULL
'''
fuelcons = pd.read_sql_query(query, connection)
fuelcons.shape
"""),
        md("## Reproduce the approved 55/45 combination"),
        code("""fuelcons["notebook_combined_l_100km"] = 0.55 * fuelcons["city_l_100km"] + 0.45 * fuelcons["highway_l_100km"]
fuelcons["difference_l_100km"] = fuelcons["notebook_combined_l_100km"] - fuelcons["canonical_combined_l_100km"]
fuelcons[["city_l_100km", "highway_l_100km", "canonical_combined_l_100km", "difference_l_100km"]].head()
"""),
        code("""assert fuelcons["difference_l_100km"].abs().max() < 1e-9
fuelcons.groupby("electrification")["canonical_combined_l_100km"].describe()
"""),
        md("## Multi-RUN lineage remains visible"),
        code("""adoption = pd.read_sql_query('''
SELECT fuelcons_id, COUNT(*) AS adopted_runs
FROM fuelcons_run_adoption
GROUP BY fuelcons_id
''', connection)
adoption["adopted_runs"].value_counts().sort_index()
"""),
        code("""fuelcons.merge(adoption, on="fuelcons_id").sort_values("adopted_runs", ascending=False).head(10)
"""),
        code("connection.close()\n"),
        md("""## What did we learn?

FuelCons is a comparison result with explicit basis and adopted RUN lineage. A convertible number is not enough: conflicting evidence remains unresolved instead of being averaged opportunistically."""),
    ]

    books["12F_09_cross_legislation_benchmark.ipynb"] = [
        md("""# 12F-09 — Cross-legislation benchmark

**Question:** What can be compared without pretending EPA and WLTP are the same method?"""),
        code(PATHS + """SANITY_FILE = REPO_ROOT / "etl/data/processed/sprint_12e3_electric_unit_audit/electric_energy_sanity_dataset.csv"
energy = pd.read_csv(SANITY_FILE, low_memory=False)
energy["canonical_wh_per_km"] = pd.to_numeric(energy["canonical_wh_per_km"], errors="coerce")
comparable = energy.dropna(subset=["canonical_wh_per_km"]).copy()
"""),
        md("## Keep source and provenance in the comparison"),
        code("""summary = comparable.groupby(["source_system", "provenance_class"])["canonical_wh_per_km"].agg(["count", "median", "mean", "min", "max"])
summary
"""),
        code("""comparable[["source_system", "raw_unit", "interpreted_unit", "provenance_class"]].value_counts().head(15)
"""),
        md("## Distribution by program/source context"),
        code("""import matplotlib.pyplot as plt

ax = comparable.boxplot(column="canonical_wh_per_km", by="source_system", rot=20, figsize=(8, 4))
ax.set(title="Electric consumption by source context", ylabel="Wh/km")
plt.suptitle("")
plt.show()
"""),
        md("""## What did we learn?

Wh/km provides a common unit, not a common regulation. EPA source-row interpretations, FuelEconomy label values, and JRC declared results retain different methods and provenance in every comparison."""),
    ]

    books["12F_10_canonical_dataset_build.ipynb"] = [
        md("""# 12F-10 — Canonical dataset build

**Question:** How do trustworthy records form Program → Configuration → VDE → RUN → FuelCons tables?

This MVP exports a deterministic 75-program public-data slice from the canonical staging database. It does not rebuild production ETL."""),
        code(PATHS + """import sqlite3

CANONICAL_DB = REPO_ROOT / "etl/data/staging/sprint_12e2_epa_fuelcons/eco_drive_canonical_epa_fuelcons.db"
OUTPUT_DIR = REPO_ROOT / "notebooks/_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
connection = sqlite3.connect(CANONICAL_DB.resolve().as_uri() + "?mode=ro", uri=True)
"""),
        md("## Select a deterministic relational slice"),
        code("""program_ids = pd.read_sql_query('''
SELECT DISTINCT vc.program_id
FROM vehicle_configuration vc
JOIN vde v ON v.vehicle_configuration_id=vc.vehicle_configuration_id
ORDER BY vc.program_id
LIMIT 75
''', connection)["program_id"].tolist()
placeholders = ",".join("?" for _ in program_ids)
program = pd.read_sql_query(f"SELECT * FROM program WHERE program_id IN ({placeholders})", connection, params=program_ids)
vehicle_configuration = pd.read_sql_query(f"SELECT * FROM vehicle_configuration WHERE program_id IN ({placeholders})", connection, params=program_ids)
"""),
        code("""configuration_ids = vehicle_configuration["vehicle_configuration_id"].tolist()
config_marks = ",".join("?" for _ in configuration_ids)
vde = pd.read_sql_query(f"SELECT * FROM vde WHERE vehicle_configuration_id IN ({config_marks})", connection, params=configuration_ids)

vde_ids = vde["id"].tolist()
vde_marks = ",".join("?" for _ in vde_ids)
run = pd.read_sql_query(f"SELECT * FROM run WHERE vde_id IN ({vde_marks})", connection, params=vde_ids)
fuelcons = pd.read_sql_query(f"SELECT * FROM fuelcons WHERE vde_id IN ({vde_marks})", connection, params=vde_ids)

fuelcons_ids = fuelcons["id"].tolist()
if fuelcons_ids:
    fuelcons_marks = ",".join("?" for _ in fuelcons_ids)
    fuelcons_run_adoption = pd.read_sql_query(
        f'''SELECT * FROM fuelcons_run_adoption
            WHERE fuelcons_id IN ({fuelcons_marks})
            ORDER BY fuelcons_id, run_id, result_dimension''',
        connection,
        params=fuelcons_ids,
    )
else:
    fuelcons_run_adoption = pd.read_sql_query("SELECT * FROM fuelcons_run_adoption WHERE 0", connection)
connection.close()
"""),
        md("## Check relationships before export"),
        code("""assert set(vehicle_configuration["program_id"]) <= set(program["program_id"])
assert set(vde["vehicle_configuration_id"]) <= set(vehicle_configuration["vehicle_configuration_id"])
assert set(run["vde_id"]) <= set(vde["id"])
assert set(fuelcons["vde_id"]) <= set(vde["id"])
assert set(vde["vde_id_parent"].dropna().astype(int)) <= set(vde["id"])
assert set(fuelcons_run_adoption["fuelcons_id"]) <= set(fuelcons["id"])
assert set(fuelcons_run_adoption["run_id"]) <= set(run["run_id"])
assert set(fuelcons_run_adoption["vde_id"]) <= set(vde["id"])

fuelcons_vde = fuelcons.set_index("id")["vde_id"]
run_vde = run.set_index("run_id")["vde_id"]
assert fuelcons_run_adoption["vde_id"].eq(fuelcons_run_adoption["fuelcons_id"].map(fuelcons_vde)).all()
assert fuelcons_run_adoption["vde_id"].eq(fuelcons_run_adoption["run_id"].map(run_vde)).all()

tables = {"program": program, "vehicle_configuration": vehicle_configuration, "vde": vde, "run": run,
          "fuelcons": fuelcons, "fuelcons_run_adoption": fuelcons_run_adoption}
pd.Series({name: len(frame) for name, frame in tables.items()}, name="rows")
"""),
        md("## Save explicit staging tables"),
        code("""sort_keys = {"program": ["program_id"], "vehicle_configuration": ["vehicle_configuration_id"], "vde": ["id"],
             "run": ["run_id"], "fuelcons": ["id"],
             "fuelcons_run_adoption": ["fuelcons_id", "run_id", "result_dimension"]}
for name, frame in tables.items():
    path = OUTPUT_DIR / f"12f10_{name}.csv"
    frame.sort_values(sort_keys[name], kind="stable").to_csv(path, index=False)
    print(f"{name:24s} {len(frame):5d} rows -> {path.relative_to(REPO_ROOT)}")
"""),
        md("""## What did we learn?

Relational export order follows ownership: Program, Configuration, VDE, then RUN and FuelCons. Provenance and NULLs travel with the records; no notebook-only identity is invented."""),
    ]

    books["12F_11_database_ingestion.ipynb"] = [
        md("""# 12F-11 — Canonical database ingestion

**Question:** How can the relational slice be loaded and checked safely?

The target is an explicit disposable database under `notebooks/_data/`, never a runtime DB."""),
        code(PATHS + """import sqlite3

DATA_DIR = REPO_ROOT / "notebooks/_data"
SCHEMA_FILE = REPO_ROOT / "etl/schema/canonical_schema_v1.sql"
DEMO_DB = DATA_DIR / "12f11_canonical_notebook_demo.db"
input_paths = {name: DATA_DIR / f"12f10_{name}.csv" for name in ["program", "vehicle_configuration", "vde", "run", "fuelcons", "fuelcons_run_adoption"]}
assert SCHEMA_FILE.exists() and all(path.exists() for path in input_paths.values())
"""),
        md("## Create only the guarded disposable target"),
        code("""assert DEMO_DB.parent.resolve() == DATA_DIR.resolve()
assert DEMO_DB.name == "12f11_canonical_notebook_demo.db"
if DEMO_DB.exists():
    DEMO_DB.unlink()

connection = sqlite3.connect(DEMO_DB)
connection.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
connection.execute("PRAGMA foreign_keys=ON")
"""),
        md("## Load in foreign-key-safe order"),
        code("""load_order = ["program", "vehicle_configuration", "vde", "run", "fuelcons", "fuelcons_run_adoption"]
loaded_counts = {}
for table in load_order:
    frame = pd.read_csv(input_paths[table], low_memory=False)
    frame = frame.astype(object).where(pd.notna(frame), None)
    frame.to_sql(table, connection, if_exists="append", index=False)
    loaded_counts[table] = len(frame)
connection.commit()
adoption_count = connection.execute("SELECT COUNT(*) FROM fuelcons_run_adoption").fetchone()[0]
assert adoption_count > 0
assert adoption_count == loaded_counts["fuelcons_run_adoption"]
pd.Series(loaded_counts, name="loaded_rows")
"""),
        md("## Integrity and a readable join"),
        code("""foreign_key_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
assert foreign_key_violations == []
assert quick_check == "ok"
print("foreign_key_check: 0 violations")
print("quick_check:", quick_check)
"""),
        code("""query = '''
SELECT p.commercial_make, p.commercial_model,
       vc.vehicle_configuration_id, v.id AS vde_id,
       COUNT(DISTINCT f.id) AS fuelcons_records,
       COUNT(DISTINCT a.run_id) AS adopted_runs
FROM program p
JOIN vehicle_configuration vc ON vc.program_id=p.program_id
JOIN vde v ON v.vehicle_configuration_id=vc.vehicle_configuration_id
LEFT JOIN fuelcons f ON f.vde_id=v.id
LEFT JOIN fuelcons_run_adoption a ON a.fuelcons_id=f.id AND a.vde_id=v.id
GROUP BY p.program_id, vc.vehicle_configuration_id, v.id
ORDER BY adopted_runs DESC
LIMIT 10
'''
joined = pd.read_sql_query(query, connection)
joined
"""),
        code("""database_counts = {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in load_order}
assert database_counts == loaded_counts
connection.close()
DEMO_DB.relative_to(REPO_ROOT)
"""),
        md("""## What did we learn?

Safe ingestion needs a disposable target, schema-first creation, dependency order, row-count reconciliation, and foreign-key/integrity checks. Runtime databases were never opened for writing."""),
    ]

    for name, cells in books.items():
        path = NOTEBOOKS / name
        path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    manifest_path = NOTEBOOKS / "notebook_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["notebooks"]:
        entry["status"] = "MVP_EXECUTABLE"
    by_id = {entry["id"]: entry for entry in manifest["notebooks"]}
    by_id["12F-02"]["inputs"] = ["etl/data/raw/epa_testcar/epa_testcar_2026_raw.xlsx"]
    by_id["12F-03"]["inputs"] = ["etl/data/processed/sprint_12e3_electric_unit_audit/electric_energy_sanity_dataset.csv"]
    by_id["12F-08"]["inputs"] = ["etl/data/staging/sprint_12e2_epa_fuelcons/eco_drive_canonical_epa_fuelcons.db"]
    by_id["12F-10"]["inputs"] = ["etl/data/staging/sprint_12e2_epa_fuelcons/eco_drive_canonical_epa_fuelcons.db"]
    by_id["12F-10"]["outputs"] = [
        "notebooks/_data/12f10_program.csv", "notebooks/_data/12f10_vehicle_configuration.csv",
        "notebooks/_data/12f10_vde.csv", "notebooks/_data/12f10_run.csv",
        "notebooks/_data/12f10_fuelcons.csv", "notebooks/_data/12f10_fuelcons_run_adoption.csv",
    ]
    by_id["12F-11"]["inputs"] = ["etl/schema/canonical_schema_v1.sql", *by_id["12F-10"]["outputs"]]
    manifest["version"] = "12F-mvp-v1"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
    print("Built 11 Sprint 12F MVP notebooks")
