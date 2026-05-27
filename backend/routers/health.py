"""Health check endpoint."""

from fastapi import APIRouter, Request

from backend.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    svc = request.app.state.model_service
    return HealthResponse(
        status="ok",
        version="1.0.0",
        physicians_loaded=len(svc.physicians) if svc.physicians is not None else 0,
        features_loaded=len(svc.feature_columns),
    )
