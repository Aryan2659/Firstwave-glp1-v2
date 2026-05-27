"""
FirstWave ML Training Pipeline
================================

Trains and evaluates:
1. XGBoost binary classifier (early_adopter prediction)
2. Cox PH survival model (time-to-first-prescription)
3. SHAP TreeExplainer for explainability

Inputs:  data/processed/physician_features.parquet
Outputs: ml/models/{xgb_model.pkl, cox_model.pkl, shap_explainer.pkl,
                   feature_metadata.json, evaluation_results.json}
"""

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/home/claude/firstwave/data/processed/physician_features.parquet"
MODELS_DIR = Path("/home/claude/firstwave/ml/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "years_in_practice",
    "rural",
    "prior_injectable_glp1_prescriber",
    "prior_sglt2_prescriber",
    "prior_dpp4_prescriber",
    "patient_panel_size",
    "unique_drugs_prescribed",
    "total_open_payments_usd",
    "num_speaker_events",
    "is_speaker",
    "kol_pagerank_normalized",
    "has_kol_connection",
]

CATEGORICAL_COLS = ["specialty", "state", "gender"]


def load_and_split_data():
    """Load features and create stratified train/test split.

    Note: Features describe each physician's baseline state (prior prescribing,
    KOL connections, demographics) — all known PRE-launch. Label is the
    post-launch outcome. So random stratified split is appropriate.
    A 'temporal holdout' equivalent would be to apply this trained model
    to a future drug launch (e.g., Foundayo) — that's the production use case.
    """
    from sklearn.model_selection import train_test_split

    print(f"\nLoading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    df_train, df_test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["early_adopter"],
        random_state=42,
    )

    print(f"  Train rows: {len(df_train):,} ({df_train['early_adopter'].mean():.1%} early adopters)")
    print(f"  Test rows:  {len(df_test):,} ({df_test['early_adopter'].mean():.1%} early adopters)")
    return df_train, df_test


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals, keep numeric features as-is."""
    X_numeric = df[FEATURE_COLS].copy()
    X_categorical = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=False)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    return X


def train_baseline_logistic(X_train, y_train, X_test, y_test):
    """Logistic regression baseline."""
    print("\n[Baseline] Logistic Regression")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_pred)
    print(f"  ROC-AUC: {auc:.4f}")
    return {"model": lr, "scaler": scaler, "auc": auc, "y_pred": y_pred}


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with cross-validation."""
    print("\n[Main] XGBoost Classifier")
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        m = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric="auc",
            n_jobs=-1,
            verbosity=0,
        )
        m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        pred = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        cv_scores.append(roc_auc_score(y_train.iloc[val_idx], pred))
    print(f"  5-fold CV ROC-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="auc",
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    print(f"  Holdout ROC-AUC: {auc:.4f}")
    print(f"  Holdout PR-AUC:  {pr_auc:.4f}")
    return {"model": model, "auc": auc, "pr_auc": pr_auc, "y_pred": y_pred,
            "cv_scores": cv_scores}


def compute_lift_curve(y_true, y_score, n_bins=10):
    """Compute lift over baseline by decile."""
    df = pd.DataFrame({"true": y_true.values, "score": y_score})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=False, duplicates="drop") + 1
    baseline = df["true"].mean() if df["true"].mean() > 0 else 1e-9
    lift_by_decile = df.groupby("decile")["true"].mean() / baseline
    return lift_by_decile.to_dict()


def train_shap_explainer(model, X_sample):
    """Build SHAP TreeExplainer."""
    print("\n[Explainability] Building SHAP TreeExplainer")
    explainer = shap.TreeExplainer(model)
    sample_shap = explainer.shap_values(X_sample.iloc[:100])

    importance = np.abs(sample_shap).mean(axis=0)
    feature_importance = pd.Series(importance, index=X_sample.columns).sort_values(ascending=False)
    print("  Top 5 features by SHAP importance:")
    for f, v in feature_importance.head(5).items():
        print(f"    {f:40s} {v:.4f}")
    return explainer, feature_importance.to_dict()


def train_survival_model(df_train):
    """Train Cox PH survival model on time-to-first-prescription."""
    print("\n[Survival] Cox Proportional Hazards Model")

    survival_cols = [
        "months_to_event", "event_observed",
        "years_in_practice", "rural",
        "prior_injectable_glp1_prescriber", "prior_sglt2_prescriber",
        "patient_panel_size", "total_open_payments_usd",
        "kol_pagerank_normalized",
    ]
    surv_df = df_train[survival_cols].copy()
    surv_df["patient_panel_size"] = surv_df["patient_panel_size"] / 1000.0
    surv_df["total_open_payments_usd"] = np.log1p(surv_df["total_open_payments_usd"]) / 10.0

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(surv_df, duration_col="months_to_event", event_col="event_observed")
    print(f"  Concordance index: {cph.concordance_index_:.4f}")
    print(f"  Log-likelihood:    {cph.log_likelihood_:.2f}")
    return cph


def predict_urgency_tier(days: float) -> str:
    if days <= 30:
        return "Tier 1 — Next 30 days"
    elif days <= 60:
        return "Tier 2 — 30 to 60 days"
    elif days <= 90:
        return "Tier 3 — 60 to 90 days"
    else:
        return "Tier 4 — Beyond 90 days"


def main():
    df_train, df_test = load_and_split_data()

    X_train = build_feature_matrix(df_train)
    X_test = build_feature_matrix(df_test)

    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = df_train["early_adopter"]
    y_test = df_test["early_adopter"]

    print(f"\nFeature matrix: {X_train.shape[1]} columns")
    print(f"Class balance:  train {y_train.mean():.1%} | test {y_test.mean():.1%}")

    baseline = train_baseline_logistic(X_train, y_train, X_test, y_test)
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)

    lift = compute_lift_curve(y_test, xgb_result["y_pred"])
    print(f"\n[Lift] Top decile lift: {lift.get(1, 0):.2f}x over baseline")

    explainer, feature_importance = train_shap_explainer(xgb_result["model"], X_test)

    cox_model = train_survival_model(df_train)

    print("\n[Saving artifacts]")
    joblib.dump(xgb_result["model"], MODELS_DIR / "xgb_model.pkl")
    print(f"  ✓ {MODELS_DIR / 'xgb_model.pkl'}")

    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
    print(f"  ✓ {MODELS_DIR / 'shap_explainer.pkl'}")

    with open(MODELS_DIR / "cox_model.pkl", "wb") as f:
        pickle.dump(cox_model, f)
    print(f"  ✓ {MODELS_DIR / 'cox_model.pkl'}")

    metadata = {
        "feature_columns": common_cols,
        "categorical_columns": CATEGORICAL_COLS,
        "numeric_feature_columns": FEATURE_COLS,
        "feature_importance": feature_importance,
    }
    with open(MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ {MODELS_DIR / 'feature_metadata.json'}")

    fpr, tpr, _ = roc_curve(y_test, xgb_result["y_pred"])
    prec, rec, _ = precision_recall_curve(y_test, xgb_result["y_pred"])
    results = {
        "baseline_logistic_auc": baseline["auc"],
        "xgboost_auc": xgb_result["auc"],
        "xgboost_pr_auc": xgb_result["pr_auc"],
        "xgboost_cv_mean": float(np.mean(xgb_result["cv_scores"])),
        "xgboost_cv_std": float(np.std(xgb_result["cv_scores"])),
        "lift_by_decile": lift,
        "cox_concordance": float(cox_model.concordance_index_),
        "test_size": int(len(y_test)),
        "test_positive_rate": float(y_test.mean()),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
    }
    with open(MODELS_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ {MODELS_DIR / 'evaluation_results.json'}")

    print("\n=== Training complete ===")
    print(f"Final XGBoost ROC-AUC: {xgb_result['auc']:.4f}")
    print(f"Final XGBoost PR-AUC:  {xgb_result['pr_auc']:.4f}")
    print(f"Top decile lift:       {lift.get(1, 0):.2f}x")
    print(f"Survival concordance:  {cox_model.concordance_index_:.4f}")


if __name__ == "__main__":
    main()
