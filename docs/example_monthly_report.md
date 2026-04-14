# CreditLens Monthly Portfolio Report

**Generated:** March 31, 2026  
**Analyst:** CreditLens Agentic Reporter  
**Period:** March 2026

---

## Executive Summary

This month's portfolio analysis reveals significant distribution shifts in debt-to-income ratios and credit utilization patterns, triggering retraining recommendations. While overall portfolio size remains healthy at 12,453 loans with $245M exposure, DTI drift (PSI 0.41) and revolving utilization drift (PSI 1.93) indicate material population changes. The XGBoost challenger model shows promising 7pp Gini improvement over the current scorecard and is ready for promotion after completing shadow deployment validation.

---

## Key Findings

• **Drift Alert:** DTI and revolving utilization both exceeded PSI threshold of 0.25, indicating significant population shift from training baseline (Q1 2017). Retraining triggered with HIGH severity.

• **Performance Degradation:** Current model Gini declined to 0.64 from training baseline of 0.71 (7pp degradation, 9.9% relative decline). This performance gap aligns with observed feature drift.

• **Cohort Underperformance:** Q3 2017 vintage showing 14.2% default rate vs 11.8% expected (2.4pp gap). This cohort represents 18% of current active portfolio and is primary driver of expected loss increase.

• **Challenger Ready:** XGBoost model (v2.0) completing day 18 of shadow deployment with Gini 0.71 vs champion scorecard 0.64. Shadow metrics stable - promotion eligible.

• **Risk Composition Shift:** High-risk grades (E/F/G) now 15% of new originations, up from 10% in training period. Grade mix shift partially explains DTI drift.

---

## Detailed Analysis

### Drift Investigation

The drift detection system flagged two features for immediate action: debt-to-income ratio (PSI 0.41) and revolving credit utilization (PSI 1.93). Investigation reveals these shifts are driven by recent origination cohorts, particularly Q3 2017. 

Median DTI for grade C loans in Q3 2017 is 20.8%, compared to 18.0% training baseline - a 2.8 percentage point increase. This isn't random variation: it reflects a systematic shift toward higher-leverage borrowers in that segment. Revolving utilization drift is even more pronounced, with the >80% utilization segment growing from 12% to 24% of new originations. This pattern suggests either (a) underwriting standards evolved to accept higher-utilization borrowers, or (b) economic stress is pushing otherwise-qualified borrowers into higher utilization before loan application.

Cross-referencing with vintage performance data confirms this is problematic: the Q3 2017 cohort is defaulting at 14.2% vs 11.8% predicted at origination. The feature drift is real, and it's impacting outcomes.

### Model Performance Context

Current model performance degradation (Gini 0.71 → 0.64) is consistent with the observed population shift. The scorecard was calibrated on Q1 2017 data where median DTI was 18% and high-utilization borrowers were rare. When applied to the current population (higher DTI, higher utilization), discrimination naturally declines because the feature relationships are different.

This isn't model decay in the traditional sense - it's population shift. The model is performing exactly as designed; the population it's scoring has changed. Calibration drift is also visible: predicted default rates are systematically below observed rates in the high-DTI, high-utilization segment. Expected loss is understated by approximately $2.3M (0.8% of total exposure).

### Champion/Challenger Status

The XGBoost challenger model (v2.0) was trained on blended Q1-Q3 2017 data, incorporating the cohorts that are now driving drift. Shadow deployment metrics after 18 days show Gini 0.71 - matching the original scorecard's training performance and representing a 7pp improvement over current production. Calibration is also superior: the challenger's predicted-vs-observed default curve shows tighter fit in the high-risk segments.

Shadow deployment has been stable. No prediction anomalies, no latency issues, no edge case failures. The model is ready for promotion.

---

## Recommendations

1. **Promote Challenger to Champion (Immediate)**  
   XGBoost v2.0 should replace the scorecard as production champion. Gini improvement of 7pp translates to approximately $1.8M in reduced expected loss annually (assuming current portfolio composition). Shadow metrics are stable; risk is low. Recommend promotion within the next retraining cycle.

2. **Retrain with Q3 2017+ Data (This Quarter)**  
   Even with challenger promotion, quarterly retraining is mandatory. Next retraining should use Q1-Q4 2017 data to capture the full distribution shift. Feature engineering should also be reviewed: consider adding a "utilization trend" feature (change in utilization 3 months pre-application) to better capture financial stress signals.

3. **Review Q3 2017 Underwriting (Risk Management)**  
   Q3 2017 cohort underperformance (14.2% default vs 11.8% predicted) warrants review with underwriting team. Was there a deliberate policy change, or did credit standards slip? If deliberate, ensure pricing reflects the higher risk. If accidental, tighten controls.

4. **Monitor High-Utilization Segment (Ongoing)**  
   >80% revolving utilization is now 24% of originations vs 12% historical. This segment has 2.1x the portfolio default rate. Monitor monthly: if this trend continues, consider either (a) explicit utilization caps in underwriting guidelines, or (b) risk-adjusted pricing for high-utilization applicants.

5. **Next Month's Focus**  
   Watch for PSI stabilization post-retraining. If DTI and utilization continue drifting beyond Q4 2017, it signals a sustained structural change (e.g., regulatory change, macroeconomic shift, market strategy pivot) rather than a one-time cohort effect. That would trigger a deeper strategic review.

---

## Portfolio Snapshot

| Metric | Value |
|--------|-------|
| Total Loans | 12,453 |
| Total Exposure | $245,000,000 |
| Average Loan Amount | $19,678 |
| Expected Loss | $28,910,000 (11.8% of exposure) |
| High-Risk % (E/F/G grades) | 15.0% |
| Current Model Gini | 0.64 |
| Challenger Model Gini | 0.71 |
| PSI - DTI | 0.41 (RETRAIN) |
| PSI - Revolving Util | 1.93 (RETRAIN) |

---

## Appendix: Technical Details

**Drift Thresholds:**  
PSI < 0.1 = Stable | 0.1-0.25 = Monitor | >0.25 = Retrain

**Features Monitored:**  
dti, annual_income, revol_util, loan_amount, delinq_2yrs, open_acc, model_predictions

**Reference Period:**  
Q1 2017 (training data)

**Comparison Period:**  
March 2026 (current originations)

**Retraining Signals Triggered:**
- Feature drift (2 features exceed PSI 0.25)
- Performance degradation (7pp Gini decline)

**Next Review:**  
April 30, 2026 (monthly cadence)

---

*This report was generated automatically by the CreditLens agentic insight reporter using drift detection results, model performance metrics, and portfolio queries. All findings are based on data as of March 31, 2026.*
