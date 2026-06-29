"""
DemandIQ FastAPI application entry point.
"""

from fastapi import FastAPI
from demandiq.api.routes import router

app = FastAPI(
    title       = "DemandIQ",
    description = "Stockout-Aware Demand Forecasting & Inventory Optimization API",
    version     = "1.0.0"
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demandiq.api.main:app", host="0.0.0.0", port=8000, reload=True)