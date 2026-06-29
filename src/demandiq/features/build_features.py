"""
Feature engineering for DemandIQ.
Builds calendar, lag, rolling, and stockout features from cleaned data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Lag days and rolling windows
LAG_DAYS     = [1, 7, 14, 28]
ROLLING_WINS = [7, 14, 28]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day of week, week of year, month, is_weekend."""
    df['day_of_week']  = df['dt'].dt.dayofweek
    df['week_of_year'] = df['dt'].dt.isocalendar().week.astype(int)
    df['month']        = df['dt'].dt.month
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    print("Calendar features added")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features per store-product pair.
    lag_N = sale_amount from N days ago.
    Uses shift to prevent data leakage.
    """
    grp = df.groupby(['store_id', 'product_id'])['sale_amount']

    for lag in LAG_DAYS:
        df[f'sales_lag_{lag}'] = grp.shift(lag)

    print(f"Lag features added: {[f'sales_lag_{l}' for l in LAG_DAYS]}")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window statistics per store-product pair.
    Uses shift(1) before rolling to avoid leakage — only past data used.
    """
    grp = df.groupby(['store_id', 'product_id'])['sale_amount']

    for win in ROLLING_WINS:
        # Rolling mean and std
        df[f'rolling_mean_{win}'] = grp.shift(1).transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
        df[f'rolling_std_{win}'] = grp.shift(1).transform(
            lambda x: x.rolling(win, min_periods=1).std()
        )

    # Peak and lowest demand in last 7 days
    df['rolling_max_7'] = grp.shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).max()
    )
    df['rolling_min_7'] = grp.shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).min()
    )

    print(f"Rolling features added for windows: {ROLLING_WINS}")
    return df


def add_stockout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stockout history features per store-product pair.
    Helps model learn if a product has been frequently out of stock recently.
    """
    stk_grp = df.groupby(['store_id', 'product_id'])['stockout_flag']

    # Was there a stockout yesterday?
    df['stockout_lag_1'] = stk_grp.shift(1)

    # How many stockout days in the last 7 and 28 days?
    df['stockout_count_7'] = stk_grp.shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    )
    df['stockout_count_28'] = stk_grp.shift(1).transform(
        lambda x: x.rolling(28, min_periods=1).sum()
    )

    print("Stockout history features added")
    return df


def build_features(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input: cleaned dataframe from preprocess.py
    Output: feature-rich dataframe ready for modeling.
    """
    print(f"Input shape: {df.shape}")

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_stockout_features(df)

    print(f"Output shape: {df.shape}")
    print(f"Total features: {df.shape[1]}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        print(f"Saved to {save_path}")

    return df


if __name__ == "__main__":
    df = pd.read_parquet("data/interim/train_clean.parquet")
    build_features(df, save_path="data/processed/train_features.parquet")