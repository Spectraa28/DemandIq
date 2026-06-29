"""
Prediction logger for DemandIQ.
Logs every forecast and inventory recommendation to SQLite.
"""

import sqlite3
import uuid
import time
from pathlib import Path
from datetime import datetime


DB_PATH = "artifacts/prediction_logs.db"


def init_db(db_path: str = DB_PATH) -> None:
    """Create prediction log tables if they don't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Forecast logs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_logs (
            request_id       TEXT PRIMARY KEY,
            timestamp        TEXT,
            store_id         INTEGER,
            product_id       INTEGER,
            model_version    TEXT,
            horizon_days     INTEGER,
            avg_prediction   REAL,
            latency_ms       REAL
        )
    """)

    # Inventory recommendation logs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inventory_logs (
            request_id               TEXT PRIMARY KEY,
            timestamp                TEXT,
            store_id                 INTEGER,
            product_id               INTEGER,
            model_version            TEXT,
            current_inventory        REAL,
            recommended_reorder_qty  REAL,
            reorder_point            REAL,
            safety_stock             REAL,
            expected_stockout_risk   REAL,
            estimated_cost           REAL,
            latency_ms               REAL
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")


def log_forecast(
    store_id:       int,
    product_id:     int,
    model_version:  str,
    horizon_days:   int,
    predictions:    list,
    latency_ms:     float,
    db_path:        str = DB_PATH
) -> str:
    """Log a forecast request to the database. Returns request_id."""
    request_id   = str(uuid.uuid4())
    timestamp    = datetime.utcnow().isoformat()
    avg_pred     = sum(predictions) / len(predictions) if predictions else 0.0

    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO forecast_logs
        (request_id, timestamp, store_id, product_id, model_version,
         horizon_days, avg_prediction, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (request_id, timestamp, store_id, product_id, model_version,
          horizon_days, avg_pred, latency_ms))
    conn.commit()
    conn.close()

    return request_id


def log_inventory(
    store_id:               int,
    product_id:             int,
    model_version:          str,
    current_inventory:      float,
    recommended_reorder_qty: float,
    reorder_point:          float,
    safety_stock:           float,
    expected_stockout_risk: float,
    estimated_cost:         float,
    latency_ms:             float,
    db_path:                str = DB_PATH
) -> str:
    """Log an inventory recommendation to the database. Returns request_id."""
    request_id = str(uuid.uuid4())
    timestamp  = datetime.utcnow().isoformat()

    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO inventory_logs
        (request_id, timestamp, store_id, product_id, model_version,
         current_inventory, recommended_reorder_qty, reorder_point,
         safety_stock, expected_stockout_risk, estimated_cost, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (request_id, timestamp, store_id, product_id, model_version,
          current_inventory, recommended_reorder_qty, reorder_point,
          safety_stock, expected_stockout_risk, estimated_cost, latency_ms))
    conn.commit()
    conn.close()

    return request_id


def get_summary(db_path: str = DB_PATH) -> dict:
    """Return summary stats from prediction logs."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*), AVG(latency_ms), AVG(avg_prediction) FROM forecast_logs")
    f_count, f_latency, f_avg_pred = cur.fetchone()

    cur.execute("SELECT COUNT(*), AVG(latency_ms), AVG(recommended_reorder_qty) FROM inventory_logs")
    i_count, i_latency, i_avg_reorder = cur.fetchone()

    conn.close()

    return {
        "forecast_requests":       f_count or 0,
        "avg_forecast_latency_ms": round(f_latency or 0, 2),
        "avg_prediction":          round(f_avg_pred or 0, 4),
        "inventory_requests":      i_count or 0,
        "avg_inventory_latency_ms": round(i_latency or 0, 2),
        "avg_reorder_qty":         round(i_avg_reorder or 0, 4)
    }


# Initialize DB on import
init_db()