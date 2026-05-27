"""Predict endpoint — ranked physician list."""

from fastapi import APIRouter, Request

from backend.schemas import (
    FilterOptions,
    PhysicianBase,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest):
    svc = request.app.state.model_service
    records, total = svc.get_ranked_physicians(
        state=body.state,
        specialty=body.specialty,
        min_score=body.min_score,
        limit=body.limit,
        offset=body.offset,
    )
    physicians = [PhysicianBase(**r) for r in records]
    return PredictResponse(
        physicians=physicians,
        total=total,
        returned=len(physicians),
        filters_applied={
            "state": body.state,
            "specialty": body.specialty,
            "min_score": body.min_score,
        },
    )


@router.get("/filters", response_model=FilterOptions)
async def filters(request: Request):
    svc = request.app.state.model_service
    return FilterOptions(states=svc.list_states(), specialties=svc.list_specialties())
