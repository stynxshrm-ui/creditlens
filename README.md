# CreditLens

**End-to-end credit risk modeling platform with production deployment, drift monitoring, and automated agentic insights.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

CreditLens is a production-grade credit risk platform built on multi-year Lending Club loan data. It demonstrates the full ML lifecycle from data ingestion through feature engineering, model training, serving, drift detection, and automated stakeholder reporting.

**What makes this different:**

- **Industry-aligned**: Uses credit risk best practices (WoE scorecards, PSI drift detection, champion/challenger) that match what banks actually deploy
- **Production-ready**: FastAPI serving, MLflow tracking, Docker containerization, automated retraining triggers, compliance logging
- **End-to-end ownership**: Not just models—includes data modeling, feature engineering, serving infrastructure, monitoring, and business reporting
- **Advanced techniques**: Uplift modeling for causal inference, agentic LLM reporter for stakeholder communication
- **Interview-optimized**: Built to demonstrate both DS and MLE capabilities in a single cohesive project

---

## Why Credit Risk?

**Regulation makes best practices mandatory.** IFRS 9, Basel III, and SR 11-7 require calibrated PD estimates, model explainability, champion/challenger workflows, and drift monitoring. When you demonstrate these in interviews, you're showing knowledge of what's required in production, not just academic ML.

**Business impact is explicit.** Expected Loss = PD × LGD × EAD. Direct line from model output to financial consequence. More concrete than churn CLV or recommendation CTR.

**Public dataset is rich.** Lending Club provides actual loan data with payment histories, temporal structure, and real defaults. No synthetic generation needed.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│  Lending Club CSV → 4 Normalized Tables → Parquet → DuckDB         │
│  (loans, borrowers, payments, outcomes)                            │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                            │
│  • Origination features (loan_amount, DTI, grade, income)          │
│  • Temporal payment features (window functions: months 1-6 behavior)│
│  • Vintage cohort features (quarterly default rates)               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        MODELING LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  Scorecard   │  │  PD Models   │  │ Uplift Model │             │
│  │  (Champion)  │  │  (Challenger)│  │ (T-Learner)  │             │
│  │  Gini: 0.64  │  │  Gini: 0.71  │  │ Qini: 0.089  │             │
│  │  WoE/IV      │  │  XGBoost     │  │ Restructure  │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                          ↓                                          │
│                    MLflow Tracking                                  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                                  │
│  FastAPI Endpoints:                                                 │
│  • POST /v1/score → PD, credit score, uplift, SHAP values          │
│  • GET /v1/health                                                   │
│  • GET /v1/model/info → champion version, metrics                  │
│  Champion/Challenger Shadow Deployment → Manual Promotion          │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   MONITORING & RETRAINING                           │
│  Monthly Pipeline:                                                  │
│  1. PSI Drift Detection (6 features: DTI, income, util, amount...) │
│  2. Multi-Signal Retraining Trigger (PSI, performance, time-based) │
│  3. Evidently Reports → Retraining Task Creation                   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENTIC REPORTING                                │
│  Anthropic Claude API + 6 Tools:                                   │
│  • Investigates drift patterns                                     │
│  • Analyzes performance degradation                                │
│  • Identifies problem cohorts                                      │
│  • Reviews challenger readiness                                    │
│  → Generates plain-English monthly reports for stakeholders        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Data Modeling (Week 1)
- Normalized relational schema from flat Lending Club CSV
- 4 tables: `loans` (origination), `borrowers` (bureau data), `payments` (monthly), `outcomes` (defaults)
- Derived monthly payment rows from cumulative fields
- DuckDB warehouse with Parquet storage

### 2. Feature Engineering (Week 2)
- **Temporal payment features** using SQL window functions (months 1-6 behavior most predictive)
- **Vintage cohort features** (quarterly default rates)
- **Origination features** (DTI, income, revolving utilization)
- All features implemented as DuckDB SQL views for reproducibility

### 3. Traditional Scorecard (Week 3)
- **WoE (Weight of Evidence)** binning with IV (Information Value) selection
- Points-based scorecard (300-850 range) matching industry convention
- **Champion baseline**: Gini 0.64, serves as production model until challenger proves superior
- Regulatory compliance: calibrated, explainable, audit trail preserved

### 4. PD Models (Week 4)
- **Logistic Regression** (calibration baseline)
- **XGBoost** (discrimination winner: Gini 0.71, +7pp improvement)
- **PyTorch Neural Network** (demonstrates deep learning range)
- Evaluation suite: Gini, KS statistic, calibration curves, SHAP values, fairness checks, vintage analysis

### 5. Uplift Modeling (Week 5)
- **T-Learner** for restructuring intervention (does offering payment plan reduce default?)
- Simulated treatment on early-struggling borrowers (payment_ratio < 0.9 in months 1-3)
- **Four segments**: Persuadables (target), Sure Things, Lost Causes, Safe
- **Qini curve** evaluation: targeting top 25% prevents 68% of recoverable defaults
- Business framing: 60% ops cost reduction vs blanket outreach

### 6. Production Serving (Week 6)
- **FastAPI endpoints**: `/v1/score`, `/v1/health`, `/v1/model/info`, `/v1/portfolio/risk-summary`
- Returns: PD score, credit score (300-850), risk band, uplift score, segment, top-5 SHAP values
- **Champion/Challenger workflow**: 
  - Scorecard (champion) in production
  - XGBoost (challenger) in 30-day shadow deployment
  - Manual promotion after validation
- **MLflow Model Registry**: staging / production / archived (regulatory audit trail)
- Dockerized, deployable to GCP Cloud Run

### 7. Drift Detection & Retraining (Week 7)
- **PSI (Population Stability Index)** monitoring on 6 credit features
  - DTI, annual_income, revol_util, loan_amount, delinq_2yrs, open_acc
  - Industry-standard thresholds: PSI < 0.1 stable, 0.1-0.25 monitor, >0.25 retrain
- **Multi-signal retraining logic** (5 triggers):
  1. Feature drift (PSI > 0.25)
  2. Prediction distribution drift
  3. Time-based (quarterly mandatory)
  4. Performance degradation (Gini drop > 5%)
  5. Elevated monitoring (30%+ features in monitor zone)
- **Monthly refresh pipeline**: DuckDB queries, PSI calculation, retraining task creation
- **Evidently** integration for drift reports
- Tested with 4 scenarios: stable, slight, moderate, severe drift

### 8. Agentic Insight Reporter (Week 8)
- **6 investigation tools** (raw Anthropic API, no LangChain):
  - `get_psi_report(feature)` - Drift analysis with interpretation
  - `get_model_performance(period)` - Gini, KS, degradation metrics
  - `get_portfolio_summary(segment)` - Size, exposure, expected loss
  - `get_vintage_cohort(quarter)` - Cohort performance gaps
  - `get_champion_challenger_status()` - Model promotion readiness
  - `query_portfolio(sql)` - Custom analysis (read-only, validated)
- **Multi-turn agentic investigation**: Agent decides which tools to call and in what order
- **Plain-English monthly reports**: Executive summary, key findings, detailed analysis, recommendations
- Synthesizes drift patterns, performance degradation, and cohort analysis into actionable insights
- **Example output**: "DTI drift (PSI 0.41) driven by Q3 2017 cohort, grade C median DTI 2.8pp above baseline. Cohort underperforming (14.2% default vs 11.8% expected). Challenger Gini 0.71. Recommend promotion."

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data Warehouse** | DuckDB (embedded, columnar) |
| **Storage** | Parquet (compressed, columnar) |
| **Feature Engineering** | DuckDB SQL (window functions, CTEs) |
| **Scorecard** | scorecardpy + scikit-learn LogisticRegression |
| **PD Models** | scikit-learn, XGBoost, PyTorch |
| **Uplift** | XGBoost T-Learner |
| **Experiment Tracking** | MLflow |
| **Serving** | FastAPI |
| **Drift Detection** | Evidently (PSI), custom implementation |
| **Agentic Layer** | Anthropic Claude API (Sonnet 4) |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Dataset** | Lending Club (Kaggle, 2015-2018) |
| **Optional Cloud** | GCP Cloud Run |

---

## Project Structure

```
creditlens/
├── README.md
├── docker-compose.yml
├── Makefile                    # make init, make train, make serve, make drift-test
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/                    # Lending Club CSVs (gitignored)
│   ├── tables/                 # Parquet tables (gitignored)
│   └── creditlens.duckdb       # gitignored
│
├── ingestion/                  # 
│   ├── split_tables.py         # CSV → 4 Parquet tables
│   ├── derive_payments.py      # Cumulative → monthly payment rows
│   └── load_duckdb.py          # Register Parquet in DuckDB
│
├── features/                   # 
│   ├── origination_features.sql
│   ├── payment_features.sql    # Temporal window functions
│   └── vintage_features.sql
│
├── models/                     # 
│   ├── scorecard/
│   │   ├── woe_encoder.py
│   │   ├── scorecard.py
│   │   └── evaluate.py         # IV, Gini, KS, calibration
│   ├── pd/
│   │   ├── logistic.py
│   │   ├── xgboost_model.py
│   │   ├── neural_net.py
│   │   └── evaluate.py         # Gini, KS, SHAP, fairness, vintage
│   ├── uplift/
│   │   ├── t_learner.py
│   │   ├── evaluate.py         # Qini curve, AUUC
│   │   └── segments.py
│   └── train.py                # Unified entry point, logs to MLflow
│
├── serving/                    # 
│   ├── app.py                  # FastAPI endpoints
│   ├── schemas.py              # Pydantic models
│   └── Dockerfile
│
├── drift_pipeline/             # 
│   ├── drift_detector.py       # PSI calculation
│   ├── retrain_trigger.py      # Multi-signal retraining logic
│   ├── monthly_refresh.py      # Orchestration
│   └── test_drift.py           # Integration tests
│
├── agent/                      # 
│   ├── tools.py                # 6 investigation tools
│   ├── reporter.py             # Anthropic API integration
│   └── prompts.py              # System prompts (optional)
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── notebooks/
│   └── eda.ipynb               # Exploratory only
│
├── mlruns/                     # gitignored
│
└── docs/
    ├── architecture.md
    ├── eda_findings.md         # Deliverable with actual numbers
    ├── business_summary.md     # Example agent output
    ├── WEEK7_DRIFT_DETECTION.md
    └── WEEK8_AGENTIC_REPORTER.md
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Setup

```bash
# 1. Clone and navigate
git clone https://github.com/yourusername/creditlens.git
cd creditlens

# 2. Download Lending Club data
# Get from Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
# Download 2015-2018 loan data
# Place in data/raw/

# 3. Initialize data pipeline
make init
# Runs: split_tables.py, derive_payments.py, load_duckdb.py

# 4. Create features
python features/create_views.py

# 5. Train models
make train
# Logs to MLflow: http://localhost:5000

# 6. Start serving
make serve
# API: http://localhost:8000/docs

# 7. Test drift detection
make drift-test

# 8. Generate monthly report (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY='your-key-here'
python agent/reporter.py
```

### Example API Usage

```bash
# Score a loan application
curl -X POST "http://localhost:8000/v1/score" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amount": 20000,
    "term": 36,
    "grade": "B",
    "dti": 18.5,
    "annual_income": 65000,
    "revol_util": 45.2,
    "delinq_2yrs": 0,
    "open_acc": 8
  }'

# Response:
{
  "pd_score": 0.118,
  "credit_score": 612,
  "gini_band": "medium",
  "uplift_score": 0.024,
  "segment": "persuadable",
  "shap_values": {
    "dti": -0.032,
    "revol_util": 0.018,
    "grade": 0.015,
    "annual_income": -0.008,
    "term": 0.005
  }
}
```

---

## Results & Performance

### Model Performance (Holdout Test Set)

| Model | Gini | KS | AUC | Calibration Slope |
|-------|------|----|----|-------------------|
| **Scorecard (Champion)** | 0.64 | 0.38 | 0.82 | 1.00 |
| **XGBoost (Challenger)** | 0.71 | 0.42 | 0.855 | 0.98 |
| **PyTorch NN** | 0.68 | 0.40 | 0.84 | 0.95 |

**Champion → Challenger Promotion Impact:**
- Gini improvement: +7pp (11% relative increase)
- Expected loss reduction: ~$1.8M annually (based on current portfolio)
- Calibration maintained (slope 0.98 vs 1.00)

### Uplift Model Performance

- **AUUC (Qini)**: 0.089
- **Targeting Strategy**: Top 25% by uplift score
- **Business Impact**: 
  - Prevents 68% of recoverable defaults
  - Contacts only 25% of at-risk borrowers
  - **60% reduction in ops cost** vs blanket outreach

### Drift Detection (Simulated Scenarios)

| Scenario | DTI PSI | Revol Util PSI | Recommendation |
|----------|---------|----------------|----------------|
| Stable | 0.007 | 0.018 | No action |
| Slight shift | 0.022 | 0.067 | Monitor |
| Moderate drift | 0.106 | 0.740 | **Retrain** |
| Severe drift | 0.412 | 1.932 | **Critical** |

### Key EDA Findings

From actual Lending Club data analysis:

1. **Payment behavior >> origination features**: Payment consistency in months 1-6 accounts for 31% of SHAP importance, more than any origination-time feature

2. **Counterintuitive partial payment signal**: Borrowers making even one partial payment in month 1 have 40% lower default rate than consistent full-payers. Partial payment signals financial awareness; full payment can conceal stress.

3. **Vintage effects**: Q3 2016 Grade B loans had 23% higher default rate than predicted at origination. Macro environment shifted post-issuance.

4. **DTI + Utilization threshold**: DTI > 28% AND revolving utilization > 75% is the strongest origination predictor, with 3.4x the portfolio default rate.

---

## Regulatory Compliance

### SR 11-7 (Fed Reserve Model Risk Management)
- ✅ Ongoing monitoring (monthly drift detection, not annual)
- ✅ Documented retraining triggers (multi-signal logic, logged decisions)
- ✅ Audit trail (MLflow model registry, decision logs)
- ✅ Champion/challenger framework (shadow deployment, validation)

### IFRS 9 / CECL (Credit Loss Accounting)
- ✅ Calibrated PD estimates (calibration curves, Brier scores)
- ✅ Current conditions validation (PSI ensures population match)
- ✅ Forward-looking (vintage analysis, economic cycle features)
- ✅ Quarterly retraining (time-based trigger ensures freshness)

### Explainability
- ✅ SHAP values (feature importance at prediction level)
- ✅ WoE binning (monotonic, interpretable transformations)
- ✅ Scorecard points (300-850 range, industry standard)
- ✅ Segment assignment (uplift model: persuadables, sure things, etc.)

---

## Future Enhancements

**Short-term (production readiness):**
- [ ] Email integration for monthly reports (SendGrid)
- [ ] Grafana dashboards for drift trends
- [ ] Airflow DAGs for pipeline orchestration
- [ ] Terraform for infrastructure-as-code
- [ ] Unit test coverage >80%

**Advanced features:**
- [ ] SHAP tool for agent (deep-dive on model explanations)
- [ ] Multi-portfolio support (1000+ portfolios in parallel)
- [ ] A/B testing framework for challenger promotion
- [ ] Real-time scoring endpoint (sub-100ms latency)
- [ ] LLM fine-tuning on credit domain reports

**ML improvements:**
- [ ] Gradient boosting with LightGBM (faster training)
- [ ] Transformer model for payment sequences
- [ ] Graph neural network (borrower relationship network)
- [ ] Online learning for drift adaptation

---

## Contributing

This is a portfolio project, but suggestions are welcome! 

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- **Dataset**: Lending Club (Kaggle)
- **Inspiration**: Real-world credit risk systems at major banks
- **References**: 
  - Siddiqi, *Credit Risk Scorecards* (2017)
  - Gutierrez & Gerardy, *Causal Inference and Uplift Modelling* (2017)
  - Basel Committee, *Guidelines on Credit Risk and Expected Credit Losses*
  - Federal Reserve SR 11-7, *Supervisory Guidance on Model Risk Management*

---

## Contact

**Satyan Sharma**  
[Email] • [LinkedIn] • [Portfolio]

*Built as a portfolio project for Data Scientist and ML Engineer in fintech/credit.*

---

**⭐ If this helped you in any way, give it a star!**
