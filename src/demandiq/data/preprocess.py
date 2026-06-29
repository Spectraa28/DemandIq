"""
Data preprocessing for DemandIQ.
Loads raw parquet, parses dates, derives stockout flag, selects columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Columns to keep from raw dataset
KEEP_COLS = [
    'store_id', 'product_id', 'city_id',
    'first_category_id', 'second_category_id', 'third_category_id',
    'dt', 'sale_amount', 'stock_hour6_22_cnt', 'stockout_flag',
    'discount', 'holiday_flag', 'activity_flag',
    'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level'
]

# A day is flagged as stockout if out-of-stock hours >= this threshold
STOCKOUT_HOUR_THRESHOLD = 2


def load_raw(path: str) -> pd.DataFrame:
    """Load raw parquet file."""
    df = pd.read_parquet(path)
    print(f"Loaded: {df.shape}")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dt column to datetime."""
    df['dt'] = pd.to_datetime(df['dt'])
    return df


def derive_stockout_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binary stockout flag from stock_hour6_22_cnt.
    A day is a stockout if out-of-stock hours >= STOCKOUT_HOUR_THRESHOLD.
    """
    df['stockout_flag'] = (df['stock_hour6_22_cnt'] >= STOCKOUT_HOUR_THRESHOLD).astype(int)
    stockout_rate = df['stockout_flag'].mean() * 100
    print(f"Stockout rate: {stockout_rate:.1f}%")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant columns for modeling."""
    df = df[KEEP_COLS]
    print(f"Columns selected: {df.shape[1]}")
    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by store, product, date — required for lag features."""
    df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
    return df


def preprocess(raw_path: str, save_path: str = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Loads raw data, parses dates, derives stockout flag, selects columns, sorts.
    """
    df = load_raw(raw_path)
    df = parse_dates(df)
    df = derive_stockout_flag(df)
    df = select_columns(df)
    df = sort_data(df)

    print(f"Preprocessed shape: {df.shape}")
    print(f"Date range: {df['dt'].min().date()} to {df['dt'].max().date()}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        print(f"Saved to {save_path}")

    return df


if __name__ == "__main__":
    preprocess(
        raw_path="data/raw/train.parquet",
        save_path="data/interim/train_clean.parquet"
    )