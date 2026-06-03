from pathlib import Path

import pandas as pd


VEH_PATH = Path("data/vehicles/vehicles.csv")


def load_vehicle_db():
    if not VEH_PATH.exists():
        raise FileNotFoundError(f"Vehicle DB not found at {VEH_PATH}. Create data/vehicles/vehicles.csv")

    df = pd.read_csv(VEH_PATH)
    required = {"standard", "make", "model", "year", "size_class", "mass_kg", "A", "B", "C"}
    if not required.issubset(df.columns):
        raise ValueError(f"Vehicle DB must contain: {sorted(required)}")
    return df


def list_standards(df):
    return sorted(df["standard"].dropna().unique().tolist())


def list_size_classes(df, standard):
    return sorted(df.loc[df["standard"] == standard, "size_class"].dropna().unique().tolist())


def list_models(df, standard, size_class):
    subset = df[(df["standard"] == standard) & (df["size_class"] == size_class)].copy()
    subset["label"] = subset["make"] + " " + subset["model"] + " " + subset["year"].astype(str)
    return subset[["label"]].join(subset[["make", "model", "year"]]).join(
        subset[["A", "B", "C", "mass_kg", "Cx", "Af_m2"]]
    ).reset_index(drop=True)


def pick_vehicle_row(df, label):
    subset = df.copy()
    subset["label"] = subset["make"] + " " + subset["model"] + " " + subset["year"].astype(str)
    row = subset.loc[subset["label"] == label]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
