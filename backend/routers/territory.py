"""Territory endpoint — state-level aggregated adoption potential."""

from fastapi import APIRouter, Request

from backend.schemas import TerritoryAggregate, TerritoryResponse

router = APIRouter()


@router.get("/territory", response_model=TerritoryResponse)
async def territory(request: Request):
    svc = request.app.state.model_service
    rows = svc.get_territory_aggregates()
    return TerritoryResponse(territories=[TerritoryAggregate(**r) for r in rows])
