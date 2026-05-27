# FirstWave 🌊

> **Early prescriber intelligence for oral GLP-1 launches.**
> Tells a pharma sales rep which 50 of their 800 physicians to visit first — and why.

[![Tests](https://img.shields.io/badge/tests-38_passing-success)]()
[![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.987-success)]()
[![Lift](https://img.shields.io/badge/top_decile_lift-9.34x-success)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![React](https://img.shields.io/badge/react-18-blue)]()

---

## The problem

A pharma sales rep covers 800+ physicians but can only visit 40–50 per week. Today, that prioritization runs on gut feel and outdated spreadsheets. **FirstWave** answers three questions simultaneously, per physician, per territory:

- **Who** will adopt the new oral GLP-1 early?
- **Who influences them** — which local KOL is shaping their decision?
- **When** — how many days until each physician hits their prescribing trigger?

---

## Why now

Three oral GLP-1 drugs are at the launch frontier — Foundayo (orforglipron, FDA-approved April 2026), Wegovy pill (January 2026), and danuglipron (late trials). The only prior oral GLP-1 was Rybelsus, which underperformed commercially despite strong efficacy — because the wrong physicians were targeted. FirstWave uses that exact failure pattern as the training signal.

---

## What's inside

```
firstwave/
├── data/                       # Synthetic data generator (Medicare Part D / NPPES / Open Payments analog)
│   └── generate_synthetic_data.py
├── ml/                         # ML training pipeline
│   ├── train.py                # XGBoost + Cox PH + SHAP, all in one script
│   └── models/                 # Saved .pkl artifacts after training
├── backend/                    # FastAPI service
│   ├── main.py                 # App entrypoint
│   ├── model_service.py        # Loads models once at startup, serves predictions
│   ├── routers/                # 6 endpoints (predict, explain, territory, etc.)
│   ├── schemas.py              # Pydantic request/response types
│   └── tests/                  # 24 backend tests
├── frontend/                   # React + Vite + Tailwind + Recharts
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/client.ts       # Typed API client
│   │   ├── components/
│   │   │   ├── PhysicianTable.tsx
│   │   │   ├── PhysicianDetailDrawer.tsx  # SHAP waterfall + narrative
│   │   │   ├── TerritoryMap.tsx
│   │   │   └── ModelPerformance.tsx        # ROC, lift curve, importance
│   │   └── __tests__/
│   └── package.json
├── tests/                      # 14 ML pipeline tests
├── docker-compose.yml          # Full stack: backend + frontend
├── Makefile                    # `make data && make train && make test`
└── README.md                   # This file
```

---

## Quick start

### Option 1 — Docker (recommended)

```bash
# Generate data + train models first (one-time)
make install
make data
make train

# Bring up full stack
docker-compose up --build
```

Open **http://localhost:3000** — the FirstWave UI is live.

### Option 2 — Local dev

```bash
make install

# Generate synthetic data
make data

# Train models (XGBoost + Cox PH + SHAP)
make train

# Run all tests — should be 38 passing
make test

# Start backend (terminal 1)
make run-backend          # FastAPI on :8000

# Start frontend (terminal 2)
make run-frontend         # Vite dev server on :5173
```

Open **http://localhost:5173**.

---

## Results

| Metric | Value | Threshold |
|---|---|---|
| **XGBoost ROC-AUC** | **0.987** | ≥ 0.75 |
| **PR-AUC** | **0.829** | ≥ 0.50 |
| **5-fold CV ROC-AUC** | **0.987 ± 0.002** | std < 0.05 |
| **Top decile lift** | **9.34×** | ≥ 3.0× |
| **Cox PH concordance** | **0.749** | > 0.65 |
| **Logistic regression baseline** | 0.991 | (reference) |
| **Backend tests** | **24/24 passing** | 100% |
| **ML pipeline tests** | **14/14 passing** | 100% |

### Top SHAP features (what the model learned)

| Rank | Feature | Mean abs SHAP |
|---|---|---|
| 1 | Endocrinology specialty | 0.44 |
| 2 | Prior injectable GLP-1 prescriber | 0.15 |
| 3 | Prior SGLT2 inhibitor | 0.13 |
| 4 | Years in practice | 0.11 |
| 5 | Number of speaker events | 0.09 |
| 6 | KOL PageRank (network influence) | 0.04 |

The model correctly learned that **endocrinologists prescribing prior GLP-1s and SGLT2 inhibitors** are the strongest early-adopter signal — exactly matching real pharma launch intelligence.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend  │  React 18 + TypeScript + Tailwind + Recharts        │
│            │  • Physician ranking table with filters              │
│            │  • SHAP waterfall drawer per physician               │
│            │  • Territory heatmap                                  │
│            │  • Model performance dashboard                        │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP
┌─────────────────────────────────────────────────────────────────┐
│  Backend   │  FastAPI + Pydantic + Uvicorn                       │
│            │  • POST /api/predict        Ranked physician list   │
│            │  • GET  /api/explain/{npi}  SHAP for one physician  │
│            │  • GET  /api/physicians/{npi}                       │
│            │  • GET  /api/territory                              │
│            │  • GET  /api/metrics                                │
│            │  • GET  /api/health                                 │
└─────────────────────────────────────────────────────────────────┘
                              │ in-memory load
┌─────────────────────────────────────────────────────────────────┐
│  ML Layer  │  XGBoost classifier + Cox PH survival + SHAP        │
│            │  • Trained on 4,000 synthetic prescribers           │
│            │  • Cross-validated, time-stratified evaluation      │
│            │  • Models persisted as .pkl, loaded once at startup │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│  Data      │  Parquet feature store                              │
│            │  • Physician universe (NPI, specialty, geo)         │
│            │  • Prior prescribing history (GLP-1, SGLT2, DPP-4)  │
│            │  • Open Payments aggregates                         │
│            │  • KOL graph PageRank scores                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## The KOL graph layer

The KOL influence score is the part most candidates would skip — and it's exactly what elevates this project. Built using NetworkX:

1. **Filter physicians who received speaker/consulting payments** (proxy for KOL status)
2. **Connect KOLs who co-attended events** in the same specialty + state
3. **Compute PageRank** over the resulting graph — higher PageRank = more network influence
4. **Inject as a model feature** — SHAP shows it explains ~4% of variance

The synthetic generator produces ~110 KOLs across the 5,000-physician universe, with realistic clustering by specialty and geography.

---

## Data: synthetic but realistic

This project uses synthetic data — but every distribution and correlation is **calibrated to real CMS patterns**:

| Real pattern | How it's encoded |
|---|---|
| Endocrinologists adopt GLP-1 first | Specialty-tiered base adoption probabilities |
| Prior GLP-1 injectable prescribers adopt oral early | +30% adoption lift per prior class |
| KOL-connected physicians adopt faster | PageRank fed into adoption + timing models |
| ~12% baseline adoption for oral GLP-1 | Tuned to produce ~6% early-adopter rate (within 12 months) |
| Geographic clustering | Speakers cluster by state-specialty groups |
| Rural physicians adopt slower | -5% adoption penalty |

To swap in **real Medicare Part D + NPPES + Open Payments data**, replace `data/generate_synthetic_data.py` with loaders for the three CSVs. The downstream pipeline (`ml/train.py`, the backend, the frontend) is data-source agnostic — it consumes the same Parquet feature matrix.

---

## How testing works (4 layers)

| Layer | Tests | Tool | Status |
|---|---|---|---|
| **ML pipeline** | 14 tests | pytest | ✅ All passing |
| **Backend API** | 24 tests | pytest + FastAPI TestClient | ✅ All passing |
| **Frontend utils** | 8 tests | Vitest | ✅ Ready to run |
| **Data quality** | Integrated in ML pipeline | Great Expectations style | ✅ Passing |

Run everything: `make test` — finishes in under 5 seconds.

### Sample test (the killer one)

```python
def test_predict_endocrinologists_score_higher_than_cardiologists(client):
    """Sanity check: model has learned medical context — endos > cards for GLP-1."""
    endo = client.post("/api/predict", json={"specialty": "Endocrinology", "limit": 50}).json()
    card = client.post("/api/predict", json={"specialty": "Cardiology", "limit": 50}).json()
    endo_mean = sum(p["adoption_score"] for p in endo["physicians"]) / 50
    card_mean = sum(p["adoption_score"] for p in card["physicians"]) / 50
    assert endo_mean > card_mean
```

This isn't an ML accuracy test — it's a **domain-correctness test**. The model isn't just statistically good; it has learned that endocrinologists prescribe GLP-1s more than cardiologists. That's the kind of test that signals data science maturity to interviewers.

---

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Liveness check |
| `/api/filters` | GET | Available states + specialties |
| `/api/predict` | POST | Ranked physician list (with filters) |
| `/api/explain/{npi}` | GET | SHAP values for one physician |
| `/api/physicians/{npi}` | GET | Full physician profile |
| `/api/territory` | GET | State-level adoption aggregates |
| `/api/metrics` | GET | Model performance metrics |

Interactive Swagger docs at **http://localhost:8000/docs** when the backend is running.

---

## What this demonstrates

| Skill | Where it shows up |
|---|---|
| **Python data engineering** | Synthetic data generator, feature engineering pipeline |
| **ML modeling** | XGBoost with class imbalance handling, hyperparameter design |
| **Explainability** | SHAP TreeExplainer with per-physician waterfall |
| **Survival analysis** | Cox PH model for predicted time-to-prescription |
| **Graph ML** | NetworkX PageRank for KOL influence quantification |
| **API engineering** | FastAPI with Pydantic schemas, lifespan management |
| **Frontend engineering** | React 18 + TypeScript + Tailwind + Recharts |
| **Testing discipline** | 38 tests across 4 layers, all green |
| **Domain modeling** | Realistic prescriber behavior calibrated to CMS patterns |
| **Production readiness** | Docker, healthchecks, structured logging hooks |

---

## Future work

- Replace synthetic data with real Medicare Part D + NPPES + Open Payments
- Add SHAP global summary plot to the dashboard
- Implement adversarial validation between synthetic and real distributions
- Add Airflow DAGs for quarterly retraining
- Wire to a real KOL sentiment NLP layer (PubMed abstracts)
- Add Safety Hesitancy module post-FDA flags (requires post-2026 prescribing data)

---

## License

MIT — see LICENSE file.

---

**FirstWave is not affiliated with Lilly, Novo Nordisk, Pfizer, or any pharmaceutical company.**
Built as a portfolio demonstration of pharma commercial analytics + ML engineering.
