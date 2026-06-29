"""
Unit tests for DemandIQ inventory optimizer.
Run: pytest tests/test_inventory.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from demandiq.inventory.optimizer import recommend_inventory, InventoryRecommendation


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_forecast():
    return [1.0, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3]

@pytest.fixture
def flat_forecast():
    """Flat demand — std should be 0, safety stock minimal."""
    return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# ── Basic correctness ─────────────────────────────────────────────────────────

def test_returns_recommendation_object(sample_forecast):
    result = recommend_inventory(sample_forecast, current_inventory=5)
    assert isinstance(result, InventoryRecommendation)


def test_reorder_qty_never_negative(sample_forecast):
    """Even with high inventory, reorder qty should be 0, not negative."""
    result = recommend_inventory(sample_forecast, current_inventory=1000)
    assert result.recommended_reorder_qty == 0.0


def test_low_inventory_triggers_reorder(sample_forecast):
    """With very low inventory, should recommend a reorder."""
    result = recommend_inventory(sample_forecast, current_inventory=0)
    assert result.recommended_reorder_qty > 0


def test_overstock_zero_when_inventory_low(sample_forecast):
    """No overstock when current inventory is below reorder point."""
    result = recommend_inventory(sample_forecast, current_inventory=0)
    assert result.overstock_units == 0.0


def test_overstock_positive_when_inventory_high(sample_forecast):
    """Overstock should be positive when inventory far exceeds reorder point."""
    result = recommend_inventory(sample_forecast, current_inventory=1000)
    assert result.overstock_units > 0


# ── Safety stock ──────────────────────────────────────────────────────────────

def test_flat_demand_low_safety_stock(flat_forecast):
    """Flat demand = zero std = near-zero safety stock."""
    result = recommend_inventory(flat_forecast, current_inventory=5)
    assert result.safety_stock < 0.01


def test_higher_service_level_increases_safety_stock(sample_forecast):
    """Higher service level should require more safety stock."""
    low  = recommend_inventory(sample_forecast, current_inventory=5, service_level=0.80)
    high = recommend_inventory(sample_forecast, current_inventory=5, service_level=0.99)
    assert high.safety_stock > low.safety_stock


def test_longer_lead_time_increases_safety_stock(sample_forecast):
    """Longer lead time = more exposure to demand variability."""
    short = recommend_inventory(sample_forecast, current_inventory=5, lead_time_days=1)
    long  = recommend_inventory(sample_forecast, current_inventory=5, lead_time_days=14)
    assert long.safety_stock > short.safety_stock


# ── Reorder point ─────────────────────────────────────────────────────────────

def test_reorder_point_equals_lead_demand_plus_safety(sample_forecast):
    """Reorder point = avg_demand * lead_time + safety_stock."""
    result = recommend_inventory(sample_forecast, current_inventory=5, lead_time_days=3)
    expected = result.avg_daily_demand * 3 + result.safety_stock
    assert abs(result.reorder_point - expected) < 0.01


# ── to_dict ───────────────────────────────────────────────────────────────────

def test_to_dict_has_all_keys(sample_forecast):
    result = recommend_inventory(sample_forecast, current_inventory=5)
    d = result.to_dict()
    expected_keys = [
        'avg_daily_demand', 'std_daily_demand', 'safety_stock',
        'reorder_point', 'recommended_reorder_qty',
        'expected_stockout_risk', 'overstock_units', 'estimated_cost'
    ]
    for key in expected_keys:
        assert key in d, f"Missing key: {key}"


def test_to_dict_values_are_floats(sample_forecast):
    result = recommend_inventory(sample_forecast, current_inventory=5)
    for k, v in result.to_dict().items():
        assert isinstance(v, float), f"{k} is not float"


# ── Stockout risk ─────────────────────────────────────────────────────────────

def test_stockout_risk_equals_one_minus_service_level(sample_forecast):
    result = recommend_inventory(sample_forecast, current_inventory=5, service_level=0.95)
    assert abs(result.expected_stockout_risk - 0.05) < 0.001