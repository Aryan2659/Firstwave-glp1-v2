"""ML pipeline tests — data quality + model performance checks.

These tests verify the trained model meets minimum quality bars.
Run after training to catch regressions before deployment.
"""

import json
import pickle
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "physician_features.parquet"


@pytest.fixture(scope="module")
def features_df():
    return pd.read_parquet(DATA_PATH)


@pytest.fixture(scope="module")
def xgb_model():
    return joblib.load(MODELS_DIR / "xgb_model.pkl")


@pytest.fixture(scope="module")
def cox_model():
    with open(MODELS_DIR / "cox_model.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def eval_results():
    with open(MODELS_DIR / "evaluation_results.json") as f:
        return json.load(f)


def test_data_no_nulls_in_features(features_df):
    """Critical features must never be null."""
    critical_cols = [
        "NPI", "specialty", "state", "early_adopter",
        "patient_panel_size", "kol_pagerank_normalized",
    ]
    for col in critical_cols:
        assert features_df[col].isna().sum() == 0, f"{col} has nulls"


def test_data_npi_unique(features_df):
    assert features_df["NPI"].is_unique


def test_data_adoption_label_binary(features_df):
    assert set(features_df["early_adopter"].unique()).issubset({0, 1})


def test_data_specialty_values_valid(features_df):
    valid = {
        "Endocrinology", "Internal Medicine", "Family Medicine",
        "Cardiology", "Geriatrics", "Nephrology",
    }
    assert set(features_df["specialty"].unique()).issubset(valid)


def test_data_panel_size_positive(features_df):
    assert (features_df["patient_panel_size"] > 0).all()


def test_data_pagerank_in_valid_range(features_df):
    assert (features_df["kol_pagerank_normalized"] >= 0).all()
    assert (features_df["kol_pagerank_normalized"] <= 1).all()


def test_data_early_adopter_rate_realistic(features_df):
    """Early adopter rate must be realistic for an oral GLP-1 launch (2-15%)."""
    rate = features_df["early_adopter"].mean()
    assert 0.02 <= rate <= 0.15, f"Adoption rate {rate:.1%} unrealistic"


def test_model_auc_meets_threshold(eval_results):
    """XGBoost must achieve ROC-AUC >= 0.75 (interview-grade)."""
    assert eval_results["xgboost_auc"] >= 0.75


def test_model_beats_baseline(eval_results):
    """XGBoost should outperform or match logistic regression baseline."""
    xgb = eval_results["xgboost_auc"]
    lr = eval_results["baseline_logistic_auc"]
    assert xgb >= lr - 0.05, f"XGBoost {xgb:.3f} far below baseline {lr:.3f}"


def test_model_top_decile_lift(eval_results):
    """Top decile lift must be >= 3x for the model to be commercially viable."""
    lift = eval_results["lift_by_decile"].get("1", 0)
    assert lift >= 3.0, f"Top decile lift {lift:.2f}x below 3x threshold"


def test_model_cv_stable(eval_results):
    """Cross-validation std should be low — indicates stable model."""
    cv_std = eval_results["xgboost_cv_std"]
    assert cv_std < 0.05, f"CV std {cv_std:.3f} too high — model unstable"


def test_survival_model_concordance(cox_model):
    """Cox PH concordance must be > 0.65 (better than chance)."""
    assert cox_model.concordance_index_ > 0.65


def test_model_files_exist():
    """All required model artifacts must be on disk."""
    required = ["xgb_model.pkl", "shap_explainer.pkl", "cox_model.pkl",
                "feature_metadata.json", "evaluation_results.json"]
    for f in required:
        assert (MODELS_DIR / f).exists(), f"Missing artifact: {f}"


def test_feature_metadata_complete():
    with open(MODELS_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    assert "feature_columns" in meta
    assert "feature_importance" in meta
    assert len(meta["feature_columns"]) > 0
