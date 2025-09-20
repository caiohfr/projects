import pandas as pd
from pathlib import Path

VEH_PATH = Path("data/vehicles/vehicles.csv")

def load_vehicle_db():
    if not VEH_PATH.exists():
        raise FileNotFoundError(f"Vehicle DB not found at {VEH_PATH}. Create data/vehicles/vehicles.csv")
    df = pd.read_csv(VEH_PATH)
    # basic sanity
    required = {"standard","make","model","year","size_class","mass_kg","A","B","C"}
    if not required.issubset(df.columns):
        raise ValueError(f"Vehicle DB must contain: {sorted(required)}")
    return df

def list_standards(df):
    return sorted(df["standard"].dropna().unique().tolist())

def list_size_classes(df, standard):
    return sorted(df.loc[df["standard"]==standard, "size_class"].dropna().unique().tolist())

def list_models(df, standard, size_class):
    sub = df[(df["standard"]==standard) & (df["size_class"]==size_class)].copy()
    sub["label"] = sub["make"] + " " + sub["model"] + " " + sub["year"].astype(str)
    # return pairs (label, row_index) so we can fetch selected row easily
    return sub[["label"]].join(sub[["make","model","year"]]).join(sub[["A","B","C","mass_kg","Cx","Af_m2"]]).reset_index(drop=True)

def pick_vehicle_row(df, label):
    sub = df.copy()
    sub["label"] = sub["make"] + " " + sub["model"] + " " + sub["year"].astype(str)
    row = sub.loc[sub["label"]==label]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
