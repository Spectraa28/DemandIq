"""
Model evaluation for DemandIQ.
Computes forecasting metrics and business metrics for naive vs corrected models.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Compute MAE, RMSE, WAPE for a set of predictions.
    MAPE is skipped — explodes when y_true is close to 0.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    metrics = {
        'model':  name,
        'mae':    round(float(mae),  4),
        'rmse':   round(float(rmse), 4),
        'wape':   round(float(wape), 4),
    }

    print(f"\n{name}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  WAPE: {wape:.4f}")

    return metrics


def evaluate_on_stockout_days(
    df: pd.DataFrame,
    pred_naive: np.ndarray,
    pred_corrected: np.ndarray
) -> dict:
    """
    Evaluate both models specifically on stockout days.
    Also shows that corrected model predicts higher demand on stockout days.
    """
    stockout_mask = df['stockout_flag'] == 1
    y_true = df['sale_amount'].values

    print("\n── Stockout Days Only ──")
    print(f"  Actual avg sales:          {y_true[stockout_mask].mean():.4f}")
    print(f"  Naive model avg pred:      {pred_naive[stockout_mask].mean():.4f}")
    print(f"  Corrected model avg pred:  {pred_corrected[stockout_mask].mean():.4f}")

    metrics_naive     = compute_metrics(y_true[stockout_mask], pred_naive[stockout_mask],     "Naive (stockout days)")
    metrics_corrected = compute_metrics(y_true[stockout_mask], pred_corrected[stockout_mask], "Corrected (stockout days)")

    return {
        'naive_stockout':     metrics_naive,
        'corrected_stockout': metrics_corrected
    }


def save_metrics(metrics: dict, metrics_dir: str = "artifacts/metrics") -> None:
    """Save evaluation metrics as JSON."""
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    path = f"{metrics_dir}/evaluation.json"

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {path}")


def evaluate(
    test_path:   str = "data/processed/test_split.parquet",
    models_dir:  str = "artifacts/models",
    metrics_dir: str = "artifacts/metrics"
) -> dict:
    """Full evaluation pipeline."""
    # Load data and models
    print(f"Loading test data from {test_path}")
    test_df = pd.read_parquet(test_path)

    model_naive     = joblib.load(f"{models_dir}/model_naive.pkl")
    model_corrected = joblib.load(f"{models_dir}/model_corrected.pkl")
    feature_cols    = joblib.load(f"{models_dir}/feature_cols.pkl")

    X_test = test_df[feature_cols]
    y_test = test_df['sale_amount'].values

    # Generate predictions
    pred_naive     = model_naive.predict(X_test)
    pred_corrected = model_corrected.predict(X_test)

    print("\n── Full Test Set ──")
    metrics_naive     = compute_metrics(y_test, pred_naive,     "Naive Model")
    metrics_corrected = compute_metrics(y_test, pred_corrected, "Corrected Model")

    # Stockout-specific evaluation
    stockout_metrics = evaluate_on_stockout_days(test_df, pred_naive, pred_corrected)

    # Combine all metrics
    all_metrics = {
        'naive':              metrics_naive,
        'corrected':          metrics_corrected,
        'naive_stockout':     stockout_metrics['naive_stockout'],
        'corrected_stockout': stockout_metrics['corrected_stockout']
    }

    save_metrics(all_metrics, metrics_dir)
    return all_metrics


if __name__ == "__main__":
    evaluate()