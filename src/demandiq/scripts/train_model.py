"""
Train DemandIQ models.
Run from project root: python scripts/train_model.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demandiq.data.preprocess import preprocess
from demandiq.features.build_features import build_features
from demandiq.features.stockout_correction import correct_stockout_demand
from demandiq.models.train import train
import pandas as pd


def run():
    print("=" * 50)
    print("DemandIQ — Training Pipeline")
    print("=" * 50)

    # Step 1: Preprocess
    print("\n[1/4] Preprocessing raw data...")
    preprocess(
        raw_path  = "data/raw/train.parquet",
        save_path = "data/interim/train_clean.parquet"
    )

    # Step 2: Feature engineering
    print("\n[2/4] Building features...")
    df = pd.read_parquet("data/interim/train_clean.parquet")
    df = build_features(df, save_path="data/processed/train_features.parquet")

    # Step 3: Stockout correction
    print("\n[3/4] Applying stockout correction...")
    df = correct_stockout_demand(df)
    df.to_parquet("data/processed/train_corrected.parquet", index=False)
    print("Saved to data/processed/train_corrected.parquet")

    # Step 4: Train models
    print("\n[4/4] Training models...")
    train(
        train_path = "data/processed/train_corrected.parquet",
        models_dir = "artifacts/models"
    )

    print("\n" + "=" * 50)
    print("Training complete.")
    print("=" * 50)


if __name__ == "__main__":
    run()