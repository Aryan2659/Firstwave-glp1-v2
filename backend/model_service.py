"""
ModelService — loads and serves the trained ML artifacts.

Models loaded into memory once at app startup:
- XGBoost adoption classifier
- Cox PH survival model
- SHAP TreeExplainer
- Feature metadata (column ordering)
- Full physician feature matrix (for lookups)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "physician_features.parquet"


def urgency_tier_from_days(days: float) -> str:
    if days <= 30:
        return "Tier 1"
    if days <= 60:
        return "Tier 2"
    if days <= 90:
        return "Tier 3"
    return "Tier 4"


def urgency_tier_label(tier: str) -> str:
    labels = {
        "Tier 1": "Tier 1 — Next 30 days",
        "Tier 2": "Tier 2 — 30 to 60 days",
        "Tier 3": "Tier 3 — 60 to 90 days",
        "Tier 4": "Tier 4 — Beyond 90 days",
    }
    return labels.get(tier, tier)


class ModelService:
    def __init__(self):
        self.xgb_model = None
        self.cox_model = None
        self.shap_explainer = None
        self.feature_columns: List[str] = []
        self.physicians: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.predictions: Optional[pd.DataFrame] = None
        self.evaluation: Dict = {}

    def load(self) -> None:
        self.xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")
        self.shap_explainer = joblib.load(MODELS_DIR / "shap_explainer.pkl")
        with open(MODELS_DIR / "cox_model.pkl", "rb") as f:
            self.cox_model = pickle.load(f)
        with open(MODELS_DIR / "feature_metadata.json") as f:
            meta = json.load(f)
            self.feature_columns = meta["feature_columns"]
        with open(MODELS_DIR / "evaluation_results.json") as f:
            self.evaluation = json.load(f)

        self.physicians = pd.read_parquet(DATA_PATH)
        self.physicians = self.physicians.reset_index(drop=True)
        self._build_feature_matrix()
        self._compute_predictions()

    def _build_feature_matrix(self) -> None:
        numeric_cols = [
            "years_in_practice", "rural", "prior_injectable_glp1_prescriber",
            "prior_sglt2_prescriber", "prior_dpp4_prescriber", "patient_panel_size",
            "unique_drugs_prescribed", "total_open_payments_usd", "num_speaker_events",
            "is_speaker", "kol_pagerank_normalized", "has_kol_connection",
        ]
        cat_cols = ["specialty", "state", "gender"]
        X_num = self.physicians[numeric_cols].copy()
        X_cat = pd.get_dummies(self.physicians[cat_cols], drop_first=False)
        X = pd.concat([X_num, X_cat], axis=1)

        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]
        self.feature_matrix = X

    def _compute_predictions(self) -> None:
        scores = self.xgb_model.predict_proba(self.feature_matrix)[:, 1]
        survival_df = self._build_survival_frame()
        median_days = self.cox_model.predict_median(survival_df).values
        median_days = np.where(np.isinf(median_days), 365.0, median_days)
        median_days = np.clip(median_days * 30.0, 1, 730)

        self.predictions = pd.DataFrame({
            "NPI": self.physicians["NPI"].astype(str).values,
            "specialty": self.physicians["specialty"].values,
            "state": self.physicians["state"].values,
            "patient_panel_size": self.physicians["patient_panel_size"].values,
            "prior_injectable_glp1_prescriber":
                self.physicians["prior_injectable_glp1_prescriber"].values,
            "kol_pagerank_normalized": self.physicians["kol_pagerank_normalized"].values,
            "is_speaker": self.physicians["is_speaker"].values,
            "total_open_payments_usd": self.physicians["total_open_payments_usd"].values,
            "adoption_score": scores,
            "predicted_days_to_rx": median_days,
        })
        self.predictions["urgency_tier"] = self.predictions["predicted_days_to_rx"].apply(
            urgency_tier_from_days
        )

    def _build_survival_frame(self) -> pd.DataFrame:
        df = self.physicians.copy()
        survival_df = pd.DataFrame({
            "years_in_practice": df["years_in_practice"],
            "rural": df["rural"],
            "prior_injectable_glp1_prescriber": df["prior_injectable_glp1_prescriber"],
            "prior_sglt2_prescriber": df["prior_sglt2_prescriber"],
            "patient_panel_size": df["patient_panel_size"] / 1000.0,
            "total_open_payments_usd": np.log1p(df["total_open_payments_usd"]) / 10.0,
            "kol_pagerank_normalized": df["kol_pagerank_normalized"],
        })
        return survival_df

    def get_ranked_physicians(
        self,
        state: Optional[str] = None,
        specialty: Optional[str] = None,
        min_score: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict], int]:
        df = self.predictions.copy()
        if state and state != "ALL":
            df = df[df["state"] == state]
        if specialty and specialty != "ALL":
            df = df[df["specialty"] == specialty]
        df = df[df["adoption_score"] >= min_score]
        df = df.sort_values("adoption_score", ascending=False)

        total = len(df)
        df = df.iloc[offset : offset + limit]

        records = df.to_dict(orient="records")
        for r in records:
            r["urgency_tier_label"] = urgency_tier_label(r["urgency_tier"])
            r["adoption_score"] = float(r["adoption_score"])
            r["predicted_days_to_rx"] = float(r["predicted_days_to_rx"])
        return records, total

    def get_physician_by_npi(self, npi: str) -> Optional[Dict]:
        df = self.predictions[self.predictions["NPI"] == str(npi)]
        if df.empty:
            return None
        record = df.iloc[0].to_dict()
        record["urgency_tier_label"] = urgency_tier_label(record["urgency_tier"])
        record["adoption_score"] = float(record["adoption_score"])
        record["predicted_days_to_rx"] = float(record["predicted_days_to_rx"])
        full = self.physicians[self.physicians["NPI"].astype(str) == str(npi)].iloc[0]
        record["years_in_practice"] = int(full["years_in_practice"])
        record["gender"] = str(full["gender"])
        record["rural"] = int(full["rural"])
        record["num_speaker_events"] = int(full["num_speaker_events"])
        record["prior_sglt2_prescriber"] = int(full["prior_sglt2_prescriber"])
        record["prior_dpp4_prescriber"] = int(full["prior_dpp4_prescriber"])
        return record

    def get_shap_explanation(self, npi: str) -> Optional[Dict]:
        mask = self.physicians["NPI"].astype(str) == str(npi)
        if not mask.any():
            return None
        idx = mask.idxmax()
        X_row = self.feature_matrix.iloc[[idx]]
        shap_values = self.shap_explainer.shap_values(X_row)[0]
        expected_value = self.shap_explainer.expected_value
        if hasattr(expected_value, "__len__"):
            expected_value = float(expected_value[0])
        else:
            expected_value = float(expected_value)

        contributions = []
        for col, val, shap_val in zip(
            self.feature_columns, X_row.iloc[0].values, shap_values
        ):
            contributions.append({
                "feature": col,
                "value": float(val),
                "shap_value": float(shap_val),
                "abs_shap": float(abs(shap_val)),
            })
        contributions.sort(key=lambda x: x["abs_shap"], reverse=True)
        return {
            "npi": str(npi),
            "expected_value": expected_value,
            "prediction_logit": expected_value + float(shap_values.sum()),
            "contributions": contributions[:12],
        }

    def get_territory_aggregates(self) -> List[Dict]:
        agg = self.predictions.groupby("state").agg(
            physician_count=("NPI", "count"),
            mean_adoption_score=("adoption_score", "mean"),
            tier_1_count=("urgency_tier", lambda s: (s == "Tier 1").sum()),
            high_potential_count=("adoption_score", lambda s: (s >= 0.5).sum()),
        ).reset_index()
        agg["mean_adoption_score"] = agg["mean_adoption_score"].round(4)
        return agg.sort_values("mean_adoption_score", ascending=False).to_dict(orient="records")

    def list_states(self) -> List[str]:
        return sorted(self.predictions["state"].unique().tolist())

    def list_specialties(self) -> List[str]:
        return sorted(self.predictions["specialty"].unique().tolist())
