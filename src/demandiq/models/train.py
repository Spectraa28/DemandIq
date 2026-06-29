"""
Model training for DemandIQ.
Trains naive and stockout-aware XGBoost models and saves artifacts.
Tracks experiments with MLflow.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from xgboost import XGBRegressor
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Features used for training
FEATURE_COLS = [
    'store_id', 'product_id', 'city_id',
    'first_category_id', 'second_category_id', 'third_category_id',
    'day_of_week', 'week_of_year', 'month', 'is_weekend',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
    'rolling_mean_28', 'rolling_std_28', 'rolling_max_7', 'rolling_min_7',
    'stockout_lag_1', 'stockout_count_7', 'stockout_count_28',
    'discount', 'holiday_flag', 'activity_flag',
    'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level'
]

# XGBoost hyperparameters
XGB_PARAMS = {
    'n_estimators':    300,
    'learning_rate':   0.05,
    'max_depth':       6,
    'subsample':       0.8,
    'colsample_bytree': 0.8,
    'random_state':    42,
    'n_jobs':          -1
}


def get_model() -> XGBRegressor:
    """Return a fresh XGBRegressor with default params."""
    return XGBRegressor(**XGB_PARAMS)


def train_naive_model(train_df: pd.DataFrame) -> XGBRegressor:
    """
    Train naive model on raw observed sales.
    This is the incorrect approach — ignores censored demand.
    """
    X = train_df[FEATURE_COLS]
    y = train_df['sale_amount']

    model = get_model()
    model.fit(X, y)
    print("Naive model trained")
    return model


def train_corrected_model(train_df: pd.DataFrame) -> XGBRegressor:
    """
    Train stockout-aware model on corrected demand.
    demand_corrected adjusts stockout days upward to recover hidden demand.
    """
    if 'demand_corrected' not in train_df.columns:
        raise ValueError("demand_corrected not found. Run stockout_correction.py first.")

    X = train_df[FEATURE_COLS]
    y = train_df['demand_corrected']

    model = get_model()
    model.fit(X, y)
    print("Corrected model trained")
    return model


def save_artifacts(
    model_naive: XGBRegressor,
    model_corrected: XGBRegressor,
    models_dir: str = "artifacts/models"
) -> None:
    """Save both models and feature columns list."""
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(model_naive,     f"{models_dir}/model_naive.pkl")
    joblib.dump(model_corrected, f"{models_dir}/model_corrected.pkl")
    joblib.dump(FEATURE_COLS,    f"{models_dir}/feature_cols.pkl")

    # Save feature list as JSON too for reference
    with open(f"{models_dir}/feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print(f"Artifacts saved to {models_dir}/")
    print("  model_naive.pkl")
    print("  model_corrected.pkl")
    print("  feature_cols.pkl")
    print("  feature_cols.json")


def log_mlflow_run(
    model: XGBRegressor,
    train_df: pd.DataFrame,
    model_name: str,
    target_col: str,
    models_dir: str
) -> None:
    """Log a single training run to MLflow."""
    X = train_df[FEATURE_COLS]
    y = train_df[target_col]
    preds = model.predict(X)

    mae  = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    wape = float(np.sum(np.abs(y - preds)) / np.sum(np.abs(y)))

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params({
            "model_type":    "XGBRegressor",
            "target":        target_col,
            "n_estimators":  XGB_PARAMS['n_estimators'],
            "learning_rate": XGB_PARAMS['learning_rate'],
            "max_depth":     XGB_PARAMS['max_depth'],
            "feature_count": len(FEATURE_COLS),
            "train_rows":    len(train_df),
            "stockout_rate": round(float(train_df['stockout_flag'].mean()), 4)
        })

        # Log metrics
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "wape": wape})

        # Log model artifact
        mlflow.xgboost.log_model(model, artifact_path=model_name)

        print(f"MLflow run logged: {model_name} | MAE={mae:.4f} RMSE={rmse:.4f} WAPE={wape:.4f}")


def train(
    train_path:      str = "data/processed/train_corrected.parquet",
    models_dir:      str = "artifacts/models",
    mlflow_uri:      str = "./mlruns",
    experiment_name: str = "demandiq"
) -> tuple:
    """Full training pipeline with MLflow tracking. Returns (model_naive, model_corrected)."""
    print(f"Loading training data from {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"Train shape: {train_df.shape}")

    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking: {mlflow_uri} | experiment: {experiment_name}")

    # Train models
    model_naive     = train_naive_model(train_df)
    model_corrected = train_corrected_model(train_df)

    # Log both runs to MLflow
    log_mlflow_run(model_naive,     train_df, "naive_model",     "sale_amount",       models_dir)
    log_mlflow_run(model_corrected, train_df, "corrected_model", "demand_corrected",  models_dir)

    # Save artifacts locally
    save_artifacts(model_naive, model_corrected, models_dir)

    return model_naive, model_corrected


if __name__ == "__main__":
    train()