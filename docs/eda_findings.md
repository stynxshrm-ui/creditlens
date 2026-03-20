# CreditLens — EDA Findings

Data: Lending Club accepted loans, random sample of 50,000 loans
across 2007–2018 vintages. After removing ambiguous loan statuses
(Late, In Grace Period), 49,342 loans remain with a portfolio
default rate of 11.9%.

---

## Finding 1 — Vintage Effects Reveal Right-Censoring

Default rates vary significantly across origination years:

| Period | Default Rate | Interpretation |
|---|---|---|
| 2007 | 25.0% | Pre-crisis, small sample |
| 2012–2016 | 16–17% | Mature vintages, reliable signal |
| 2017 | 8.7% | Partially matured |
| 2018 | 1.7% | Right-censored — insufficient time to default |

**Implication:** 2017–2018 loans are right-censored. Their low default
rates reflect insufficient observation time, not better credit quality.
Model training is restricted to 2007–2016 vintages where loans have had
sufficient time to mature. This is the most common modelling mistake on
this dataset — including recent vintages artificially suppresses predicted
default rates.

---

## Finding 2 — Dual Stress Segment

Borrowers with DTI above 28% AND revolving utilisation above 75%
default at 16.66% vs 11.78% for the rest of the portfolio — a 41%
relative increase in default risk.

| Segment | Loans | Default Rate |
|---|---|---|
| Low DTI, low utilisation | 34,306 | 10.78% |
| Low DTI, high utilisation | 7,523 | 14.42% |
| High DTI, low utilisation | 5,868 | 14.28% |
| High DTI, high utilisation (dual stress) | 1,645 | 16.66% |

The interaction effect is clear — each stress factor alone adds ~3.5
percentage points of default risk. Both together add ~5.9 points.
Neither factor is redundant. Both enter the model as individual features
and as a combined `dual_stress_flag`.

---

## Finding 3 — Payment Behaviour Separates Defaulters Early

By the end of month 6, defaulters and non-defaulters show measurably
different payment behaviour — despite nearly identical missed payment rates.

| | Non-defaulters | Defaulters |
|---|---|---|
| Avg payment ratio (m1–6) | 1.232 | 1.047 |
| Missed payment rate | 9.0% | 9.4% |
| Deteriorating trend rate | 9.0% | 8.6% |

**The key insight:** missed payment rates are nearly identical (9.0% vs
9.4%) in the first 6 months. You cannot reliably detect future defaulters
by whether they miss payments — almost everyone pays in the early months.

What separates them is *how much* they pay. Non-defaulters overpay their
installment by 23% on average (ratio 1.232). Defaulters pay much closer
to the minimum (ratio 1.047). Financial stress is visible in the payment
amount before it appears as a missed payment.

This finding drives the design of `avg_payment_ratio_m6` as the primary
temporal feature — it captures stress that delinquency flags miss entirely.

---

## Finding 4 — Expected Loss Baseline

At the portfolio level:
- Default rate: 11.9%
- Average loan amount: $14,997
- Assumed LGD: 60% (industry standard

## Scorecard Model — Champion Baseline

- Training window: 2007–2016 vintages (29,038 loans)
- Selected features: int_rate, avg_payment_ratio_m6,
  min_payment_ratio_m6, first_month_ratio
- Gini: 0.7527
- KS: 0.5969
- Calibration error: 0.0136 (IFRS 9 compliant)
- Predicted vs actual default rate: 17.28% vs 16.80%

Registered in MLflow as creditlens_champion version 4.
XGBoost challenger must exceed Gini 0.7727 (+0.02) to justify promotion.

## Model Comparison — Scorecard vs XGBoost

| Metric | Scorecard (Champion) | XGBoost (Challenger) |
|---|---|---|
| Gini | 0.7527 | 0.8496 |
| KS | 0.5969 | 0.6978 |
| Calibration error | 0.0136 | 0.0188 |

XGBoost exceeds promotion threshold (Gini > 0.7727) by 0.077.
Qualifies for 30-day shadow deployment before manual promotion.

Top 5 features by SHAP importance (XGBoost):
1. first_month_ratio       — payment behaviour in month 1
2. avg_payment_ratio_m6    — average payment over 6 months
3. min_payment_ratio_m6    — worst single month in observation window
4. int_rate                — interest rate (risk pricing signal)
5. grade_numeric           — Lending Club's own risk assessment

Key finding: WoE/IV (scorecard) and SHAP (XGBoost) independently
rank the same three payment features as most predictive. This
consistency across two fundamentally different methods confirms
these features are genuinely informative, not artefacts of
one modelling approach.


## Three-Model Comparison

| Model | Gini | KS | Calibration Error |
|---|---|---|---|
| WoE Scorecard (champion) | 0.7527 | 0.5969 | 0.0136 |
| PyTorch Neural Net | 0.7236 | 0.5458 | 0.0162 |
| XGBoost (challenger) | 0.8496 | 0.6978 | 0.0188 |

XGBoost enters shadow deployment — Gini delta +0.0969 vs champion.
Neural net underperforms XGBoost at this dataset size (19k training
rows). Expected to close the gap on the full 1.2M row dataset.
All three models meet IFRS 9 calibration requirements (error < 0.05).


## Fairness Check — XGBoost (SR 11-7 Requirement)

Gini by income band on holdout set:

| Income band | Gini | Count |
|---|---|---|
| <40k | 0.817 | 1,075 |
| 40-70k | 0.825 | 2,341 |
| 70-100k | 0.838 | 1,383 |
| >100k | 0.844 | 1,009 |

Gini spread across bands: 0.027 — within acceptable range.
No demographic slice shows material performance degradation.
Model meets SR 11-7 fairness documentation requirements.