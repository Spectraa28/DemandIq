"""
Evaluate DemandIQ models.
Run from project root: python scripts/evaluate_model.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demandiq.models.evaluate import evaluate


def run():
    print("=" * 50)
    print("DemandIQ — Evaluation Pipeline")
    print("=" * 50)

    metrics = evaluate(
        test_path   = "data/processed/test_split.parquet",
        models_dir  = "artifacts/models",
        metrics_dir = "artifacts/metrics"
    )

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"\n{'Metric':<10} {'Naive':>10} {'Corrected':>12}")
    print("-" * 35)
    for metric in ['mae', 'rmse', 'wape']:
        naive_val     = metrics['naive'][metric]
        corrected_val = metrics['corrected'][metric]
        better        = "✅" if corrected_val < naive_val else ""
        print(f"{metric.upper():<10} {naive_val:>10.4f} {corrected_val:>12.4f} {better}")

    print("\nEvaluation complete. Metrics saved to artifacts/metrics/evaluation.json")


if __name__ == "__main__":
    run()