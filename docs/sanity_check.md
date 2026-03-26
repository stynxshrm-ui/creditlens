
```python
import numpy as np
import duckdb
import pandas as pd
import xgboost as xgb
from serving.model_store import model_store
from models.scorecard.scorecard import FACTOR, OFFSET
from models.pd.xgboost_model import XGB_FEATURES

model_store.load()

con = duckdb.connect("data/creditlens.duckdb", read_only=True)
low_risk  = con.execute("SELECT * FROM model_features WHERE loan_id = '127163071'").fetchdf()
high_risk = con.execute("SELECT * FROM model_features WHERE loan_id = '85717377'").fetchdf()
con.close()

for label, row in [("LOW RISK (non-defaulter)", low_risk),
                   ("HIGH RISK (defaulter)",    high_risk)]:

    features = {f: row.iloc[0].get(f, 0) for f in XGB_FEATURES}
    X = pd.DataFrame([features]).fillna(0).astype(float)

    z = float(model_store.xgb_model.get_booster().predict(
        xgb.DMatrix(X), output_margin=True
    )[0])

    p_bad  = 1 / (1 + np.exp(-z))
    p_good = 1 - p_bad

    odds_stats  = p_bad / p_good
    odds_credit = p_good / p_bad

    score = int(np.clip(OFFSET + FACTOR * (-z), 300, 850))

    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"  actual default_flag:       {int(row.iloc[0]['default_flag'])}")
    print(f"  z (model output):          {z:.4f}")
    print(f"  odds_stats (bad/good):     {odds_stats:.4f}")
    print(f"  log_odds_stats (= z):      {np.log(odds_stats):.4f}")
    print(f"  odds_credit (good/bad):    {odds_credit:.4f}")
    print(f"  log_odds_credit (= -z):    {np.log(odds_credit):.4f}")
    print(f"  PD = p(bad):               {p_bad:.4f}  ({p_bad*100:.1f}%)")
    print(f"  p(good):                   {p_good:.4f}  ({p_good*100:.1f}%)")
    print(f"  credit_score:              {score}")
```
---
## Credit Scoring — Full Chain Reference (Calibrated)

| | Low Risk (non-defaulter) | High Risk (defaulter) |
|---|---|---|
| **actual default_flag** | 0 | 1 |
| **z (model output)** | -2.8997 | +1.2145 |
| **odds_stats** (bad/good) | 0.0550 | 3.3685 |
| **log_odds_stats** (= z) | -2.8997 | +1.2145 |
| **odds_credit** (good/bad) | 18.168 | 0.297 |
| **log_odds_credit** (= -z) | +2.8997 | -1.2145 |
| **PD = p(bad)** | 0.052 (5.2%) | 0.771 (77.1%) |
| **p(good) = 1-PD** | 0.948 (94.8%) | 0.229 (22.9%) |
| **credit_score** | **598** | **479** |

---

## Explaning the Table

**z tells everything at a glance:**
- Negative z → low risk → high score
- Positive z → high risk → low score
- Zero z → base score ~515 (below 600 because our portfolio default rate is above 5%)

**The low risk borrower scores 598 not 800** — because their PD is 5.2%, just barely above the 5% base rate. In a FICO-calibrated system they're essentially an average borrower. To score 750+ you'd need PD below 2%.

**The high risk borrower scores 479** — below the base score of 600, correctly reflecting 77.1% default probability.

**The gap is 119 points** (598 vs 479) across a PD range of 5.2% to 77.1%. That's the discriminatory power of the model expressed as a scorecard.

---

## The Formula Chain — One Line Per Step

```
z = -2.90  (model output, log of p(bad)/p(good))
         ↓ negate
-z = +2.90  (log of p(good)/p(bad), credit convention)
         ↓ scale
Score = 515.04 + 28.85 × 2.90 = 598 

z = +1.21  (model output)
         ↓ negate
-z = -1.21
         ↓ scale
Score = 515.04 + 28.85 × (-1.21) = 479 
```

---

## BASE_ODDS Sanity Check

At exactly BASE_ODDS = 19 (5% default rate):
```
p(bad)  = 0.05,  p(good) = 0.95
z       = log(0.05/0.95) = log(1/19) = -2.944
-z      = +2.944
Score   = 515.04 + 28.85 × 2.944 = 600.0 
```

The system is fully consistent about scorecard mathematics.