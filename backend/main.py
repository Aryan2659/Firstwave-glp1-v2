"""
FirstWave Backend API
======================

FastAPI service serving:
- POST /api/predict        — Ranked physician list for a territory/filter
- GET  /api/explain/{npi}  — SHAP explanation for a single physician
- GET  /api/territory      — Territory-level aggregated scores
- GET  /api/physicians     — Browse physicians with filters
- GET  /api/metrics        — Model evaluation metrics for the dashboard
- GET  /api/health         — Liveness check

Loads pretrained models at startup. Stateless beyond that.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.model_service import ModelService
from backend.routers import explain, health, metrics, physicians, predict, territory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models once at startup, share across requests."""
    print("Starting FirstWave API...")
    app.state.model_service = ModelService()
    app.state.model_service.load()
    print(f"Loaded {len(app.state.model_service.feature_columns)} features")
    print(f"Serving {len(app.state.model_service.physicians):,} physicians")
    yield
    print("Shutting down FirstWave API")


app = FastAPI(
    title="FirstWave API",
    description="Early prescriber intelligence for oral GLP-1 launches",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(explain.router, prefix="/api", tags=["explain"])
app.include_router(territory.router, prefix="/api", tags=["territory"])
app.include_router(physicians.router, prefix="/api", tags=["physicians"])
app.include_router(metrics.router, prefix="/api", tags=["metrics"])


@app.get("/")
def root():
    return {
        "name": "FirstWave API",
        "version": "1.0.0",
        "docs": "/docs",
    }
