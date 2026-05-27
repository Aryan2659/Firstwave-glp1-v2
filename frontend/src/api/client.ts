export type Physician = {
  NPI: string;
  specialty: string;
  state: string;
  adoption_score: number;
  predicted_days_to_rx: number;
  urgency_tier: string;
  urgency_tier_label: string;
  patient_panel_size: number;
  prior_injectable_glp1_prescriber: number;
  kol_pagerank_normalized: number;
  is_speaker: number;
  total_open_payments_usd: number;
};

export type PhysicianDetail = Physician & {
  years_in_practice: number;
  gender: string;
  rural: number;
  num_speaker_events: number;
  prior_sglt2_prescriber: number;
  prior_dpp4_prescriber: number;
};

export type ShapContribution = {
  feature: string;
  value: number;
  shap_value: number;
  abs_shap: number;
};

export type ExplainResult = {
  npi: string;
  expected_value: number;
  prediction_logit: number;
  contributions: ShapContribution[];
};

export type Territory = {
  state: string;
  physician_count: number;
  mean_adoption_score: number;
  tier_1_count: number;
  high_potential_count: number;
};

export type ModelMetrics = {
  xgboost_auc: number;
  xgboost_pr_auc: number;
  xgboost_cv_mean: number;
  xgboost_cv_std: number;
  baseline_logistic_auc: number;
  cox_concordance: number;
  test_size: number;
  test_positive_rate: number;
  lift_by_decile: Record<string, number>;
  roc_curve: { fpr: number[]; tpr: number[] };
  pr_curve: { precision: number[]; recall: number[] };
  feature_importance: Record<string, number>;
};

export type Filters = {
  state?: string;
  specialty?: string;
  min_score?: number;
  limit?: number;
  offset?: number;
};

const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) {
    throw new Error(`API error ${r.status}: ${await r.text()}`);
  }
  return r.json();
}

export const api = {
  health: () => request<{ status: string; physicians_loaded: number }>("/health"),

  filters: () => request<{ states: string[]; specialties: string[] }>("/filters"),

  predict: (filters: Filters) =>
    request<{ physicians: Physician[]; total: number; returned: number }>("/predict", {
      method: "POST",
      body: JSON.stringify(filters),
    }),

  explain: (npi: string) => request<ExplainResult>(`/explain/${npi}`),

  physician: (npi: string) => request<PhysicianDetail>(`/physicians/${npi}`),

  territory: () => request<{ territories: Territory[] }>("/territory"),

  metrics: () => request<ModelMetrics>("/metrics"),
};

export function formatFeatureName(feature: string): string {
  return feature
    .replace(/specialty_/, "")
    .replace(/state_/, "")
    .replace(/gender_/, "Gender: ")
    .replace(/prior_injectable_glp1_prescriber/, "Prior injectable GLP-1")
    .replace(/prior_sglt2_prescriber/, "Prior SGLT2 inhibitor")
    .replace(/prior_dpp4_prescriber/, "Prior DPP-4 inhibitor")
    .replace(/total_open_payments_usd/, "Open Payments ($)")
    .replace(/num_speaker_events/, "Speaker events")
    .replace(/kol_pagerank_normalized/, "KOL influence (PageRank)")
    .replace(/has_kol_connection/, "KOL network member")
    .replace(/is_speaker/, "Active speaker")
    .replace(/patient_panel_size/, "Patient panel size")
    .replace(/unique_drugs_prescribed/, "Drug diversity")
    .replace(/years_in_practice/, "Years in practice")
    .replace(/_/g, " ");
}
