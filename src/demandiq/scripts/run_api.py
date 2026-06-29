"""
Run DemandIQ FastAPI server.
Run from project root: python scripts/run_api.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

if __name__ == "__main__":
    print("=" * 50)
    print("DemandIQ — Starting API Server")
    print("=" * 50)
    print("\nAPI docs available at: http://localhost:8000/docs")
    print("Health check:          http://localhost:8000/health\n")

    uvicorn.run(
        "demandiq.api.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True
    )