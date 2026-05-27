"""Explain endpoint — per-physician SHAP explanation."""

from fastapi import APIRouter, HTTPException, Request

from backend.schemas import ExplainResponse, ShapContribution

router = APIRouter()


@router.get("/explain/{npi}", response_model=ExplainResponse)
async def explain(request: Request, npi: str):
    svc = request.app.state.model_service
    result = svc.get_shap_explanation(npi)
    if result is None:
        raise HTTPException(status_code=404, detail=f"NPI {npi} not found")
    return ExplainResponse(
        npi=result["npi"],
        expected_value=result["expected_value"],
        prediction_logit=result["prediction_logit"],
        contributions=[ShapContribution(**c) for c in result["contributions"]],
    )
