"""
API routes for DemandIQ.
Handles forecast and inventory recommendation endpoints.
"""

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException

from demandiq.api.schemas import (
    ForecastRequest, ForecastResponse, DailyForecast,
    InventoryRequest, InventoryResponse,
    HealthResponse, ModelInfoResponse
)
from demandiq.inventory.optimizer import recommend_inventory
from demandiq.monitoring.prediction_logger import log_forecast, log_inventory

router = APIRouter()

# ── Model loading ─────────────────────────────────────────────────────────────

MODELS_DIR    = Path("artifacts/models")
MODEL_VERSION = "xgb_stockout_corrected_v1"

def load_model():
    path = MODELS_DIR / "model_corrected.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    return joblib.load(path)

def load_feature_cols():
    path = MODELS_DIR / "feature_cols.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Feature cols not found at {path}.")
    return joblib.load(path)

def load_data():
    path = Path("data/processed/test_split.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Test data not found at {path}.")
    return pd.read_parquet(path)

# Load once at startup
try:
    _model        = load_model()
    _feature_cols = load_feature_cols()
    _data         = load_data()
    print("Models and data loaded successfully")
except Exception as e:
    print(f"Warning: {e}")
    _model        = None
    _feature_cols = None
    _data         = None


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    """Check if API and model are ready."""
    return HealthResponse(
        status        = "ok" if _model is not None else "model_not_loaded",
        model_version = MODEL_VERSION
    )


# ── Model info ────────────────────────────────────────────────────────────────

@router.get("/models/current", response_model=ModelInfoResponse)
def model_info():
    """Return current model version and feature list."""
    if _model is None or _feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_version = MODEL_VERSION,
        feature_count = len(_feature_cols),
        feature_cols  = _feature_cols
    )


# ── Forecast ──────────────────────────────────────────────────────────────────

@router.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    Forecast demand for a store-product pair over N days.
    Uses the stockout-corrected XGBoost model.
    """
    if _model is None or _data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Filter data for requested store-product
    mask = (
        (_data['store_id']   == req.store_id) &
        (_data['product_id'] == req.product_id)
    )
    pair_df = _data[mask].sort_values('dt').tail(req.horizon_days)

    if pair_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for store_id={req.store_id}, product_id={req.product_id}"
        )

    # Predict
    start = time.time()
    X     = pair_df[_feature_cols]
    preds = _model.predict(X)
    latency_ms = (time.time() - start) * 1000

    # Log prediction
    log_forecast(
        store_id      = req.store_id,
        product_id    = req.product_id,
        model_version = MODEL_VERSION,
        horizon_days  = req.horizon_days,
        predictions   = [float(p) for p in preds],
        latency_ms    = latency_ms
    )

    # Build response
    forecast_list = [
        DailyForecast(
            date             = str(row['dt'].date()),
            predicted_demand = round(float(pred), 4)
        )
        for (_, row), pred in zip(pair_df.iterrows(), preds)
    ]

    return ForecastResponse(
        store_id      = req.store_id,
        product_id    = req.product_id,
        horizon_days  = req.horizon_days,
        forecast      = forecast_list,
        model_version = MODEL_VERSION
    )


# ── Inventory recommendation ──────────────────────────────────────────────────

@router.post("/inventory/recommend", response_model=InventoryResponse)
def inventory_recommend(req: InventoryRequest):
    """
    Generate inventory reorder recommendation for a store-product pair.
    Forecasts demand first, then applies safety stock and reorder point logic.
    """
    if _model is None or _data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get demand forecast for this store-product
    mask = (
        (_data['store_id']   == req.store_id) &
        (_data['product_id'] == req.product_id)
    )
    pair_df = _data[mask].sort_values('dt').tail(7)

    if pair_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for store_id={req.store_id}, product_id={req.product_id}"
        )

    # Forecast demand
    start  = time.time()
    X      = pair_df[_feature_cols]
    preds  = _model.predict(X)

    # Get inventory recommendation
    result = recommend_inventory(
        forecast_demand   = preds,
        current_inventory = req.current_inventory,
        lead_time_days    = req.lead_time_days,
        service_level     = req.service_level,
        holding_cost      = req.holding_cost_per_unit,
        stockout_cost     = req.stockout_cost_per_unit
    )
    latency_ms = (time.time() - start) * 1000

    # Log recommendation
    log_inventory(
        store_id                = req.store_id,
        product_id              = req.product_id,
        model_version           = MODEL_VERSION,
        current_inventory       = req.current_inventory,
        recommended_reorder_qty = result.recommended_reorder_qty,
        reorder_point           = result.reorder_point,
        safety_stock            = result.safety_stock,
        expected_stockout_risk  = result.expected_stockout_risk,
        estimated_cost          = result.estimated_cost,
        latency_ms              = latency_ms
    )

    return InventoryResponse(
        store_id = req.store_id,
        product_id = req.product_id,
        **result.to_dict()
    )