"""Metrics endpoint — model evaluation results for the dashboard."""

from fastapi import APIRouter, Request

from backend.schemas import MetricsResponse

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request: Request):
    svc = request.app.state.model_service
    eval_data = svc.evaluation
    import json
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    with open(PROJECT_ROOT / "ml" / "models" / "feature_metadata.json") as f:
        feature_meta = json.load(f)

    return MetricsResponse(
        xgboost_auc=eval_data["xgboost_auc"],
        xgboost_pr_auc=eval_data["xgboost_pr_auc"],
        xgboost_cv_mean=eval_data["xgboost_cv_mean"],
        xgboost_cv_std=eval_data["xgboost_cv_std"],
        baseline_logistic_auc=eval_data["baseline_logistic_auc"],
        cox_concordance=eval_data["cox_concordance"],
        test_size=eval_data["test_size"],
        test_positive_rate=eval_data["test_positive_rate"],
        lift_by_decile=eval_data["lift_by_decile"],
        roc_curve=eval_data["roc_curve"],
        pr_curve=eval_data["pr_curve"],
        feature_importance=feature_meta["feature_importance"],
    )
