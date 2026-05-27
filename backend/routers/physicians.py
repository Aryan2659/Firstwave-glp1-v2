"""Physicians endpoint — lookup single physician with full detail."""

from fastapi import APIRouter, HTTPException, Request

from backend.schemas import PhysicianDetail

router = APIRouter()


@router.get("/physicians/{npi}", response_model=PhysicianDetail)
async def get_physician(request: Request, npi: str):
    svc = request.app.state.model_service
    record = svc.get_physician_by_npi(npi)
    if record is None:
        raise HTTPException(status_code=404, detail=f"NPI {npi} not found")
    return PhysicianDetail(**record)
