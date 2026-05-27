# FirstWave Model Card

## Model details

- **Name:** FirstWave Early Adopter Predictor
- **Version:** 1.0.0
- **Date:** 2026
- **Type:** Binary classifier (XGBoost) + Cox PH survival model
- **License:** MIT

## Intended use

**Primary use case:** Pharma sales force prioritization for oral GLP-1 drug launches. Given a physician's pre-launch profile (specialty, prior prescribing, KOL network position, geography), predict probability they will become an early adopter and the expected days-to-first-prescription.

**Primary users:** Commercial analytics teams, field sales operations, sales reps prioritizing territory visits.

**Not intended for:** Clinical decision support, patient-level recommendations, or any use case affecting individual patient care.

## Training data

- **Source:** Synthetic dataset modeled after CMS Medicare Part D, NPPES NPI Registry, and CMS Open Payments
- **Size:** 5,000 physicians (4,000 train / 1,000 test, stratified)
- **Time period:** Simulated 2020–2024 Rybelsus prescribing patterns
- **Label:** `early_adopter = 1` if physician wrote ≥10 Rybelsus prescriptions in the first 12 months of availability
- **Class balance:** ~6% positive rate (matches real-world early adopter base rate for oral GLP-1)

## Features used (40 dimensions)

| Category | Features |
|---|---|
| Demographics | specialty (6 levels), state (20 levels), gender, years_in_practice, rural |
| Prior prescribing | prior_injectable_glp1, prior_sglt2, prior_dpp4 |
| Activity | patient_panel_size, unique_drugs_prescribed |
| Open Payments | total_payments_usd, num_speaker_events, is_speaker |
| Network | kol_pagerank_normalized, has_kol_connection |

## Performance

| Metric | Value |
|---|---|
| ROC-AUC (holdout) | 0.987 |
| PR-AUC | 0.829 |
| 5-fold CV ROC-AUC | 0.987 ± 0.002 |
| Top decile lift | 9.34× |
| Cox concordance | 0.749 |

## Ethical considerations

- **Physicians as scores:** This model produces a single numerical score per physician. It should be used as a **prioritization aid**, not a replacement for human judgment about which physicians to engage.
- **Demographic features:** State and gender are inputs. The model should be audited for unintended demographic bias before any production use.
- **Synthetic data limits:** Real-world model behavior may differ from these metrics. Performance on real Medicare Part D data should be re-validated before deployment.
- **No patient data:** This model uses only physician-level features. No individual patient data is consumed or produced.

## Limitations

1. **Synthetic provenance:** The training data is generated, not collected. While distributions are calibrated to real patterns, edge cases in real prescribing data may not be represented.
2. **Single drug class:** Trained specifically on Rybelsus-like oral GLP-1 adoption. Transfer to other drug classes requires retraining.
3. **Static features:** All features describe physician state at training time. The model does not incorporate temporal updates (new speaker events, recent CME attendance, etc.) without retraining.
4. **No causal inference:** SHAP values describe predictive contribution, not causal effect. A high "Open Payments" SHAP value does not mean increasing payments will cause adoption.

## Maintenance

- **Retraining cadence:** Quarterly, aligned with CMS data releases
- **Drift monitoring:** Compare prediction distributions month-over-month
- **Performance recalibration:** Required if base early-adopter rate shifts by ±2 percentage points
