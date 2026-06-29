"""
Unit tests for DemandIQ API endpoints.
Run: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd


# ── App setup ─────────────────────────────────────────────────────────────────

from demandiq.api.main import app

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_has_status_field():
    response = client.get("/health")
    assert "status" in response.json()


def test_health_has_model_version():
    response = client.get("/health")
    assert "model_version" in response.json()


# ── Model info ────────────────────────────────────────────────────────────────

def test_model_info_endpoint_exists():
    response = client.get("/models/current")
    # Either 200 (model loaded) or 503 (model not loaded in test env)
    assert response.status_code in [200, 503]


# ── Forecast ──────────────────────────────────────────────────────────────────

def test_forecast_invalid_store_returns_error():
    response = client.post("/forecast", json={
        "store_id":     99999,
        "product_id":   99999,
        "horizon_days": 7
    })
    # Either 404 (not found) or 503 (model not loaded)
    assert response.status_code in [404, 503]


def test_forecast_request_schema_validation():
    """horizon_days must be between 1 and 30."""
    response = client.post("/forecast", json={
        "store_id":     0,
        "product_id":   4,
        "horizon_days": 100  # exceeds max
    })
    assert response.status_code == 422


def test_forecast_negative_horizon_rejected():
    response = client.post("/forecast", json={
        "store_id":     0,
        "product_id":   4,
        "horizon_days": -1
    })
    assert response.status_code == 422


def test_forecast_missing_store_id_rejected():
    response = client.post("/forecast", json={
        "product_id":   4,
        "horizon_days": 7
    })
    assert response.status_code == 422


# ── Inventory ─────────────────────────────────────────────────────────────────

def test_inventory_negative_inventory_rejected():
    response = client.post("/inventory/recommend", json={
        "store_id":               0,
        "product_id":             4,
        "current_inventory":      -5,  # invalid
        "lead_time_days":         3,
        "service_level":          0.95,
        "holding_cost_per_unit":  2.0,
        "stockout_cost_per_unit": 20.0
    })
    assert response.status_code == 422


def test_inventory_service_level_out_of_range():
    response = client.post("/inventory/recommend", json={
        "store_id":               0,
        "product_id":             4,
        "current_inventory":      10,
        "lead_time_days":         3,
        "service_level":          1.5,  # invalid — max is 0.99
        "holding_cost_per_unit":  2.0,
        "stockout_cost_per_unit": 20.0
    })
    assert response.status_code == 422


def test_inventory_missing_product_id_rejected():
    response = client.post("/inventory/recommend", json={
        "store_id":          0,
        "current_inventory": 10
    })
    assert response.status_code == 422


def test_inventory_invalid_store_returns_error():
    response = client.post("/inventory/recommend", json={
        "store_id":               99999,
        "product_id":             99999,
        "current_inventory":      10,
        "lead_time_days":         3,
        "service_level":          0.95,
        "holding_cost_per_unit":  2.0,
        "stockout_cost_per_unit": 20.0
    })
    assert response.status_code in [404, 503]