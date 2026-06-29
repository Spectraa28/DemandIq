"""
Unit tests for DemandIQ feature engineering.
Run: pytest tests/test_features.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from demandiq.features.build_features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_stockout_features
)
from demandiq.features.stockout_correction import correct_stockout_demand


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal dataframe mimicking cleaned dataset."""
    dates = pd.date_range("2024-03-28", periods=30, freq="D")
    df = pd.DataFrame({
        'store_id':    [0] * 30,
        'product_id':  [4] * 30,
        'dt':          dates,
        'sale_amount': np.random.uniform(0.5, 3.0, 30),
        'stockout_flag': ([0] * 20) + ([1] * 10)
    })
    return df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)


# ── Calendar features ─────────────────────────────────────────────────────────

def test_calendar_features_created(sample_df):
    df = add_calendar_features(sample_df.copy())
    for col in ['day_of_week', 'week_of_year', 'month', 'is_weekend']:
        assert col in df.columns, f"Missing: {col}"


def test_day_of_week_range(sample_df):
    df = add_calendar_features(sample_df.copy())
    assert df['day_of_week'].between(0, 6).all()


def test_is_weekend_binary(sample_df):
    df = add_calendar_features(sample_df.copy())
    assert set(df['is_weekend'].unique()).issubset({0, 1})


def test_weekend_matches_day_of_week(sample_df):
    df = add_calendar_features(sample_df.copy())
    expected = (df['day_of_week'] >= 5).astype(int)
    assert (df['is_weekend'] == expected).all()


# ── Lag features ──────────────────────────────────────────────────────────────

def test_lag_features_created(sample_df):
    df = add_lag_features(sample_df.copy())
    for lag in [1, 7, 14, 28]:
        assert f'sales_lag_{lag}' in df.columns


def test_lag_1_matches_previous_day(sample_df):
    df = add_lag_features(sample_df.copy())
    # lag_1 on row 1 should equal sale_amount on row 0
    assert df['sales_lag_1'].iloc[1] == pytest.approx(df['sale_amount'].iloc[0])


def test_lag_no_future_leakage(sample_df):
    df = add_lag_features(sample_df.copy())
    # First row lag_1 must be NaN — no previous data
    assert pd.isna(df['sales_lag_1'].iloc[0])


# ── Rolling features ──────────────────────────────────────────────────────────

def test_rolling_features_created(sample_df):
    df = add_rolling_features(sample_df.copy())
    for col in ['rolling_mean_7', 'rolling_std_7', 'rolling_max_7', 'rolling_min_7']:
        assert col in df.columns


def test_rolling_max_gte_min(sample_df):
    df = add_rolling_features(sample_df.copy())
    valid = df['rolling_max_7'].dropna()
    assert (df['rolling_max_7'].dropna() >= df['rolling_min_7'].dropna()).all()


# ── Stockout features ─────────────────────────────────────────────────────────

def test_stockout_features_created(sample_df):
    df = add_stockout_features(sample_df.copy())
    for col in ['stockout_lag_1', 'stockout_count_7', 'stockout_count_28']:
        assert col in df.columns


def test_stockout_count_7_never_exceeds_7(sample_df):
    df = add_stockout_features(sample_df.copy())
    assert (df['stockout_count_7'].dropna() <= 7).all()


# ── Stockout correction ───────────────────────────────────────────────────────

def test_correction_creates_demand_corrected(sample_df):
    df = add_rolling_features(sample_df.copy())
    df = correct_stockout_demand(df)
    assert 'demand_corrected' in df.columns


def test_normal_days_unchanged(sample_df):
    df = add_rolling_features(sample_df.copy())
    df = correct_stockout_demand(df)
    normal = df[df['stockout_flag'] == 0]
    assert (normal['sale_amount'] == normal['demand_corrected']).all()


def test_stockout_days_corrected_upward(sample_df):
    df = add_rolling_features(sample_df.copy())
    df = correct_stockout_demand(df)
    stockout = df[df['stockout_flag'] == 1]
    assert (stockout['demand_corrected'] >= stockout['sale_amount']).all()