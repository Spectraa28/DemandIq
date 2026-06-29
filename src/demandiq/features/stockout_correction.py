"""
Stockout demand correction for DemandIQ.
Adjusts observed sales upward during stockout periods to recover censored demand.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def correct_stockout_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create demand_corrected column.

    Logic:
    - Non-stockout days: demand_corrected = sale_amount (unchanged)
    - Stockout days:     demand_corrected = max(sale_amount, rolling_mean_7)
                         This recovers hidden demand censored by stockout.

    Requires rolling_mean_7 to already exist (from build_features.py).
    """
    if 'rolling_mean_7' not in df.columns:
        raise ValueError("rolling_mean_7 not found. Run build_features.py first.")

    # Start with observed sales
    df['demand_corrected'] = df['sale_amount']

    stockout_mask = df['stockout_flag'] == 1

    # On stockout days, use whichever is higher: observed sales or rolling mean
    df.loc[stockout_mask, 'demand_corrected'] = df.loc[stockout_mask].apply(
        lambda row: max(row['sale_amount'], row['rolling_mean_7']),
        axis=1
    )

    # Verify non-stockout days are unchanged
    normal_mask = df['stockout_flag'] == 0
    assert (df.loc[normal_mask, 'sale_amount'] == df.loc[normal_mask, 'demand_corrected']).all(), \
        "Non-stockout days were modified — check correction logic"

    # Summary
    before = df.loc[stockout_mask, 'sale_amount'].mean()
    after  = df.loc[stockout_mask, 'demand_corrected'].mean()
    uplift = ((after - before) / before) * 100

    print(f"Stockout days corrected: {stockout_mask.sum()}")
    print(f"Avg sales before correction: {before:.4f}")
    print(f"Avg demand after correction: {after:.4f}")
    print(f"Demand uplift: +{uplift:.1f}%")

    return df


def apply_correction(input_path: str, save_path: str = None) -> pd.DataFrame:
    """Load features parquet, apply correction, optionally save."""
    df = pd.read_parquet(input_path)
    print(f"Loaded: {df.shape}")

    df = correct_stockout_demand(df)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        print(f"Saved to {save_path}")

    return df


if __name__ == "__main__":
    apply_correction(
        input_path="data/processed/train_split.parquet",
        save_path="data/processed/train_corrected.parquet"
    )