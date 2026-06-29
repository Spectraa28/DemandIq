# DemandIQ ---- Stockout-Aware Demand Forecasting & Inventory Optimization

DemandIQ is a production-style machine learning system that forecasts retail product demand and recommends inventory reorder quantities while accounting for stockouts.

The project focuses on a common but important retail ML problem:

> Observed sales are not always equal to true customer demand.

When inventory is limited, sales can be capped by what was available, even if customer demand was higher. A naive forecasting model trained directly on raw sales can learn the wrong demand signal and systematically under-forecast future demand.

DemandIQ handles this by detecting stockout-affected periods, correcting the training target, comparing naive versus stockout-aware models, and converting demand forecasts into inventory recommendations.

---

## Problem Statement

In retail forecasting, stockouts create censored demand.

For example:

```text
True customer demand:  80 units
Available inventory:   30 units
Observed sales:        30 units
```

A naive model only sees `30` units sold and may assume demand was low.

But demand was actually higher. Sales were limited by inventory availability.

This creates a feedback loop:

```text
Stockout → low observed sales → low forecast → low reorder quantity → another stockout
```

DemandIQ breaks this loop by using stockout-aware demand correction before model training.

---

## Key Results

| Metric | Naive Model | Corrected Model |
| ------ | ----------- | --------------- |
| MAE    | 0.3693      | **0.3611** ✅    |
| RMSE   | 0.6559      | **0.6023** ✅    |
| WAPE   | 0.3190      | **0.3119** ✅    |

On stockout days specifically, the stockout-aware model predicts around **7% higher demand** than the naive model, helping recover demand that was hidden by inventory constraints.

---

## Dataset

This project uses **FreshRetailNet-50K**, a real-world fresh retail dataset from Hugging Face.

Dataset characteristics:

* Around 4.5M rows
* 898 stores
* 865 SKUs
* 18 cities
* 90 days of daily sales data
* Around 40.9% organic stockout rate

The high stockout rate makes this dataset useful for studying censored demand and stockout-aware forecasting.

---

## Features

DemandIQ includes:

* **Stockout-aware demand correction**

  * Adjusts stockout-affected targets upward using recent non-stockout demand history.

* **Time-series feature engineering**

  * Lag features: 1, 7, 14, and 28 days
  * Rolling features: mean, standard deviation, max, and min
  * Calendar features: day of week, month, weekend, and holiday indicators
  * Stockout history features: recent stockout counts

* **Leakage-safe validation**

  * Uses time-based train, validation, and test splits.
  * Avoids random splitting for forecasting.

* **Model comparison**

  * Compares naive XGBoost trained on observed sales against stockout-aware XGBoost trained on corrected demand.

* **Inventory optimization**

  * Converts forecasts into reorder recommendations using lead time, safety stock, service level, holding cost, and stockout cost.

* **FastAPI serving**

  * Provides forecast and inventory recommendation endpoints.

* **MLflow tracking**

  * Logs experiment parameters, metrics, and model artifacts.

* **Prediction logging**

  * Stores API prediction requests and responses in SQLite.

* **Unit tests**

  * Covers feature engineering, inventory calculations, and API behavior.

---

## Project Structure

```text
demandiq/
├── data/
│   ├── raw/                    # Original dataset
│   ├── interim/                # Cleaned intermediate data
│   └── processed/              # Feature-engineered data and splits
│
├── artifacts/
│   ├── models/                 # Trained model artifacts
│   ├── metrics/                # Evaluation JSON files
│   └── reports/                # Charts and sample outputs
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_features.ipynb
│   ├── 04_split.ipynb
│   ├── 05_stockout_correction.ipynb
│   ├── 06_train.ipynb
│   ├── 07_evaluate.ipynb
│   └── 08_inventory.ipynb
│
├── src/
│   └── demandiq/
│       ├── data/
│       │   └── preprocess.py
│       ├── features/
│       │   ├── build_features.py
│       │   └── stockout_correction.py
│       ├── models/
│       │   ├── train.py
│       │   └── evaluate.py
│       ├── inventory/
│       │   └── optimizer.py
│       ├── api/
│       │   ├── main.py
│       │   ├── routes.py
│       │   └── schemas.py
│       └── monitoring/
│           └── prediction_logger.py
│
├── scripts/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── run_api.py
│
├── tests/
│   ├── test_features.py
│   ├── test_inventory.py
│   └── test_api.py
│
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/demandiq.git
cd demandiq
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Add Dataset

Download the FreshRetailNet-50K dataset from Hugging Face and save it inside `data/raw/`.

Example:

```python
from datasets import load_dataset

ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train")
df = ds.to_pandas()
df.to_parquet("data/raw/train.parquet", index=False)
```

Expected path:

```text
data/raw/train.parquet
```

---

## Train Models

Run:

```bash
python scripts/train_model.py
```

This trains:

1. Naive XGBoost model using observed sales as the target.
2. Stockout-aware XGBoost model using corrected demand as the target.

Model artifacts are saved to:

```text
artifacts/models/
```

---

## Evaluate Models

Run:

```bash
python scripts/evaluate_model.py
```

Evaluation outputs are saved to:

```text
artifacts/metrics/
artifacts/reports/
```

The evaluation compares:

* MAE
* RMSE
* WAPE
* Stockout-day demand recovery
* Naive versus corrected model behavior

---

## Reproducibility

The reported metrics can be reproduced using:

```bash
python scripts/train_model.py
python scripts/evaluate_model.py
```

Expected outputs:

```text
artifacts/models/
artifacts/metrics/
artifacts/reports/
```

---

## Run API

Start the FastAPI server:

```bash
python scripts/run_api.py
```

Then open:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

---

### Forecast Demand

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 0,
    "product_id": 4,
    "horizon_days": 7
  }'
```

Example response:

```json
{
  "store_id": 0,
  "product_id": 4,
  "horizon_days": 7,
  "forecast": [
    {
      "date": "2024-06-01",
      "predicted_demand": 1.12
    },
    {
      "date": "2024-06-02",
      "predicted_demand": 1.08
    }
  ],
  "model_version": "xgb_stockout_corrected_v1"
}
```

---

### Inventory Recommendation

```bash
curl -X POST http://localhost:8000/inventory/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 0,
    "product_id": 4,
    "current_inventory": 10,
    "lead_time_days": 3,
    "service_level": 0.95,
    "holding_cost_per_unit": 2.0,
    "stockout_cost_per_unit": 20.0
  }'
```

Example response:

```json
{
  "recommended_reorder_quantity": 2.8,
  "reorder_point": 4.8,
  "safety_stock": 1.6,
  "expected_stockout_risk": 0.08,
  "estimated_inventory_cost": 18.4
}
```

---

## Run with Docker

```bash
docker-compose up --build
```

Services:

```text
FastAPI: http://localhost:8000/docs
MLflow:  http://localhost:5000
```

---

## Run Tests

```bash
pytest tests/ -v
```

Test coverage includes:

* Feature engineering logic
* Leakage-safe lag and rolling features
* Stockout correction behavior
* Inventory reorder calculations
* FastAPI endpoints

---

## MLflow Tracking

DemandIQ uses MLflow to track model experiments.

Logged information includes:

* Model type
* Target type
* Feature columns
* Train/validation/test date ranges
* MAE
* RMSE
* WAPE
* Model artifacts

Start MLflow UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Then open:

```text
http://localhost:5000
```

---

## Prediction Logging

Every API prediction request is logged to SQLite.

Logged fields include:

* Request ID
* Timestamp
* Store ID
* Product ID
* Model version
* Forecast horizon
* Prediction output
* Current inventory
* Recommended reorder quantity
* API latency

This creates a foundation for basic model monitoring and production-style observability.

---

## Tech Stack

| Area                | Tools                                |
| ------------------- | ------------------------------------ |
| Machine Learning    | XGBoost, scikit-learn, pandas, NumPy |
| API                 | FastAPI, Pydantic, Uvicorn           |
| Experiment Tracking | MLflow                               |
| Storage             | SQLite                               |
| Testing             | pytest                               |
| Packaging           | Docker, docker-compose               |

---

## Limitations

* The stockout correction is an approximation, not the true unobserved demand.
* The current model uses tabular time-series features rather than deep forecasting models.
* The API serves local model artifacts for MVP simplicity.
* SQLite is used for lightweight prediction logging, not high-scale production storage.
* Forecasts are point estimates; probabilistic forecasting is a future improvement.

---

## Future Improvements

* Add probabilistic forecasts with prediction intervals.
* Add category-level and city-level demand analysis.
* Add automated retraining pipeline.
* Add monitoring dashboard for drift, latency, and forecast error.
* Add backtesting across multiple time windows.
* Deploy the API and MLflow server to cloud infrastructure.
* Add batch forecasting for all store-product pairs.

---
