"""
Pydantic schemas for DemandIQ API.
Defines request and response models for all endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


# ── Forecast ──────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    store_id:     int = Field(..., example=0)
    product_id:   int = Field(..., example=4)
    horizon_days: int = Field(default=7, ge=1, le=30, example=7)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "store_id":     0,
                    "product_id":   4,
                    "horizon_days": 7
                }
            ]
        }
    }


class DailyForecast(BaseModel):
    date:             str
    predicted_demand: float


class ForecastResponse(BaseModel):
    store_id:      int
    product_id:    int
    horizon_days:  int
    forecast:      List[DailyForecast]
    model_version: str


# ── Inventory ─────────────────────────────────────────────────────────────────

class InventoryRequest(BaseModel):
    store_id:              int   = Field(..., example=0)
    product_id:            int   = Field(..., example=4)
    current_inventory:     float = Field(..., ge=0, example=10.0)
    lead_time_days:        int   = Field(default=3, ge=1, le=30, example=3)
    service_level:         float = Field(default=0.95, ge=0.5, le=0.99, example=0.95)
    holding_cost_per_unit: float = Field(default=2.0, ge=0, example=2.0)
    stockout_cost_per_unit: float = Field(default=20.0, ge=0, example=20.0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "store_id":               0,
                    "product_id":             4,
                    "current_inventory":      10.0,
                    "lead_time_days":         3,
                    "service_level":          0.95,
                    "holding_cost_per_unit":  2.0,
                    "stockout_cost_per_unit": 20.0
                }
            ]
        }
    }


class InventoryResponse(BaseModel):
    store_id:                int
    product_id:              int
    avg_daily_demand:        float
    safety_stock:            float
    reorder_point:           float
    recommended_reorder_qty: float
    expected_stockout_risk:  float
    overstock_units:         float
    estimated_cost:          float


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model_version: str


# ── Model info ────────────────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    model_version:  str
    feature_count:  int
    feature_cols:   List[str]