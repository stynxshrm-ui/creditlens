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

Registered in MLflow as creditlens_champion version 3.
XGBoost challenger must exceed Gini 0.7727 (+0.02) to justify promotion.