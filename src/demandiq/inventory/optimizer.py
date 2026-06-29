"""
Inventory optimization for DemandIQ.
Converts demand forecasts into reorder recommendations using
lead time, safety stock, service level, and cost parameters.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class InventoryRecommendation:
    """Output of the inventory optimizer."""
    avg_daily_demand:        float
    std_daily_demand:        float
    safety_stock:            float
    reorder_point:           float
    recommended_reorder_qty: float
    expected_stockout_risk:  float
    overstock_units:         float
    estimated_cost:          float

    def to_dict(self) -> dict:
        return {
            'avg_daily_demand':        round(self.avg_daily_demand,        3),
            'std_daily_demand':        round(self.std_daily_demand,        3),
            'safety_stock':            round(self.safety_stock,            3),
            'reorder_point':           round(self.reorder_point,           3),
            'recommended_reorder_qty': round(self.recommended_reorder_qty, 3),
            'expected_stockout_risk':  round(self.expected_stockout_risk,  3),
            'overstock_units':         round(self.overstock_units,         3),
            'estimated_cost':          round(self.estimated_cost,          3),
        }


def recommend_inventory(
    forecast_demand:   np.ndarray,
    current_inventory: float,
    lead_time_days:    int   = 3,
    service_level:     float = 0.95,
    holding_cost:      float = 2.0,
    stockout_cost:     float = 20.0
) -> InventoryRecommendation:
    """
    Convert demand forecast into inventory reorder recommendation.

    Args:
        forecast_demand:   Array of predicted daily demand values.
        current_inventory: Current units in stock.
        lead_time_days:    Days until new stock arrives after reorder.
        service_level:     Target probability of avoiding stockout (e.g. 0.95).
        holding_cost:      Cost per unit per day in storage.
        stockout_cost:     Cost per unit of lost sale due to stockout.

    Returns:
        InventoryRecommendation dataclass with reorder quantities and cost estimates.
    """
    forecast_demand = np.array(forecast_demand, dtype=float)

    avg_daily_demand = float(np.mean(forecast_demand))
    std_daily_demand = float(np.std(forecast_demand))

    # Z-score from service level (e.g. 0.95 → 1.645)
    z = norm.ppf(service_level)

    # Safety stock covers demand variability during lead time
    safety_stock = z * std_daily_demand * np.sqrt(lead_time_days)

    # Total expected demand while waiting for reorder to arrive
    demand_during_lead_time = avg_daily_demand * lead_time_days

    # Reorder when stock drops to this level
    reorder_point = demand_during_lead_time + safety_stock

    # How much to order — never negative
    reorder_qty = max(0.0, reorder_point - current_inventory)

    # Risk and cost estimates
    stockout_risk   = 1.0 - service_level
    overstock_units = max(0.0, current_inventory - reorder_point)

    # Estimated holding cost on overstock + holding cost on reorder quantity
    estimated_cost = (overstock_units * holding_cost) + (reorder_qty * holding_cost)

    return InventoryRecommendation(
        avg_daily_demand        = avg_daily_demand,
        std_daily_demand        = std_daily_demand,
        safety_stock            = safety_stock,
        reorder_point           = reorder_point,
        recommended_reorder_qty = reorder_qty,
        expected_stockout_risk  = stockout_risk,
        overstock_units         = overstock_units,
        estimated_cost          = estimated_cost
    )


if __name__ == "__main__":
    # Quick smoke test
    forecast = [1.0, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3]

    print("── High inventory scenario ──")
    result = recommend_inventory(forecast, current_inventory=10)
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")

    print("\n── Low inventory scenario ──")
    result = recommend_inventory(forecast, current_inventory=2)
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")