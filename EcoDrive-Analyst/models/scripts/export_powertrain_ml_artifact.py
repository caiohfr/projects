from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.vde_core.ml_prediction import NotebookPowertrainPredictor

DB_PATH = REPO_ROOT / "data" / "db" / "eco_drive.db"
ARTIFACT_PATH = REPO_ROOT / "models" / "powertrain_scenario_ml.joblib"

FEATURE_COLUMNS = [
    "category",
    "make",
    "year",
    "engine_size_l",
    "transmission_type",
    "drive_type",
    "electrification",
    "gear_count",
    "final_drive_ratio",
    "coast_A_N",
    "coast_B_N_per_kph",
    "coast_C_N_per_kph2",
    "vde_net_mj_per_km",
    "vde_urb_mj_per_km",
    "vde_hw_mj_per_km",
]

BEV_TARGETS = ["energy_ftp75_Wh_per_km", "energy_hwfet_Wh_per_km", "energy_Wh_per_km"]
NBEV_TARGETS = ["fuel_ftp75_l_per_100km", "fuel_hwfet_l_per_100km", "fuel_l_per_100km"]

CONTINUOUS_FEATURES = [
    "engine_size_l",
    "gear_count",
    "final_drive_ratio",
    "year",
    "coast_A_N",
    "coast_B_N_per_kph",
    "coast_C_N_per_kph2",
    "vde_net_mj_per_km",
    "vde_urb_mj_per_km",
    "vde_hw_mj_per_km",
]

CATEGORICAL_FEATURES = [
    "category",
    "make",
    "transmission_type",
    "drive_type",
    "electrification",
]


def load_notebook_dataset() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df_all = pd.read_sql(
            """
            SELECT *
            FROM vde_db v
            JOIN fuelcons_db f ON f.vde_id = v.id
            ORDER BY v.id ASC
            """,
            conn,
        )
    finally:
        conn.close()
    # Preserve the notebook slice exactly.
    return df_all.iloc[1:4999, :].copy()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                CONTINUOUS_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_candidate_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
            ]
        ),
        "linear_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        ),
        "mlp": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=(100, 50),
                        activation="relu",
                        solver="adam",
                        learning_rate="adaptive",
                        random_state=42,
                        early_stopping=True,
                        alpha=0.001,
                        max_iter=1000,
                    ),
                ),
            ]
        ),
    }
    try:
        from xgboost import XGBRegressor

        xgb_base = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        models["xgboost"] = Pipeline(
            [
                ("prep", preprocessor),
                ("xgb", MultiOutputRegressor(xgb_base, n_jobs=-1)),
            ]
        )
    except Exception:
        pass
    return models


def evaluate_model(name: str, model: Pipeline, x_train, x_test, y_train, y_test) -> tuple[Pipeline, dict]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        target: {
            "mae": float(mean_absolute_error(y_test[target], pred[:, idx])),
            "r2": float(r2_score(y_test[target], pred[:, idx])),
        }
        for idx, target in enumerate(y_test.columns)
    }
    combined_target = y_test.columns[-1]
    summary = {
        "model_family": name,
        "target_metrics": metrics,
        "combined_mae": metrics[combined_target]["mae"],
        "combined_r2": metrics[combined_target]["r2"],
    }
    return model, summary


def choose_best_model(x_train, x_test, y_train, y_test) -> tuple[Pipeline, dict]:
    candidates = build_candidate_models(build_preprocessor())
    scored: list[tuple[Pipeline, dict]] = []
    for name, model in candidates.items():
        fitted, summary = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        scored.append((fitted, summary))
    scored.sort(key=lambda item: (-item[1]["combined_r2"], item[1]["combined_mae"]))
    return scored[0]


def build_training_sets(df: pd.DataFrame):
    x = df[FEATURE_COLUMNS].copy()
    y = df[
        [
            "fuel_ftp75_l_per_100km",
            "fuel_hwfet_l_per_100km",
            "fuel_l_per_100km",
            "energy_ftp75_Wh_per_km",
            "energy_hwfet_Wh_per_km",
            "energy_Wh_per_km",
        ]
    ].copy()

    x_nbev = x[x["electrification"] != "BEV"].copy()
    y_nbev = y[x["electrification"] != "BEV"][NBEV_TARGETS].copy()
    mask_nbev = ~y_nbev.isna().any(axis=1)
    x_nbev = x_nbev[mask_nbev]
    y_nbev = y_nbev[mask_nbev]

    x_bev = x[x["electrification"] == "BEV"].copy()
    y_bev = y[x["electrification"] == "BEV"][BEV_TARGETS].copy()
    mask_bev = ~y_bev.isna().any(axis=1)
    x_bev = x_bev[mask_bev]
    y_bev = y_bev[mask_bev]

    return (
        train_test_split(x_bev, y_bev, test_size=0.2, random_state=42),
        train_test_split(x_nbev, y_nbev, test_size=0.2, random_state=42),
    )


def export_artifact() -> Path:
    df = load_notebook_dataset()
    (x_bev_train, x_bev_test, y_bev_train, y_bev_test), (
        x_nbev_train,
        x_nbev_test,
        y_nbev_train,
        y_nbev_test,
    ) = build_training_sets(df)

    bev_model, bev_metrics = choose_best_model(x_bev_train, x_bev_test, y_bev_train, y_bev_test)
    nbev_model, nbev_metrics = choose_best_model(x_nbev_train, x_nbev_test, y_nbev_train, y_nbev_test)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "model_name": "NotebookPowertrainPredictor",
        "model_version": f"notebook_export_{version}",
        "source_notebook": "notebooks/ML_Regression_VDE.ipynb",
        "dataset_rows_total": int(len(df)),
        "dataset_rows_bev": int(len(x_bev_train) + len(x_bev_test)),
        "dataset_rows_nbev": int(len(x_nbev_train) + len(x_nbev_test)),
        "feature_columns": FEATURE_COLUMNS,
        "continuous_features": CONTINUOUS_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "bev_targets": BEV_TARGETS,
        "nbev_targets": NBEV_TARGETS,
        "metrics": {
            "BEV": bev_metrics,
            "NBEV": nbev_metrics,
        },
        "training_split": {"test_size": 0.2, "random_state": 42},
        "notes": [
            "Export reproduces the notebook-style BEV vs non-BEV split.",
            "Runtime pre-processing imputes missing numeric/categorical powertrain features before inference.",
            "BEV predicts energy targets; non-BEV predicts fuel targets.",
            "CO2 is derived at runtime from the request context when possible.",
        ],
    }

    predictor = NotebookPowertrainPredictor(
        bev_model=bev_model,
        nbev_model=nbev_model,
        metadata=metadata,
    )
    artifact = {
        "predictor": predictor,
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "metadata": metadata,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    return ARTIFACT_PATH


def main() -> None:
    path = export_artifact()
    print(f"Artifact exported to: {path}")


if __name__ == "__main__":
    main()
