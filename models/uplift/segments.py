"""
models/uplift/segments.py

Segment classification utilities for uplift model output.

Used by serving/app.py POST /v1/score endpoint
to classify borrowers at prediction time
"""

import pandas as pd
import numpy as np


SEGMENT_DESCRIPTIONS = {
    "persuadable": "Would default without offer, repay with it — TARGET",
    "sure_thing":  "Would repay regardless — low ops value to contact",
    "lost_cause":  "Would default regardless — intervention ineffective",
    "safe":        "Not at risk — no action needed",
}


def segment_summary(uplift_scores: np.ndarray,
                    base_risk: np.ndarray,
                    y_true: np.ndarray = None) -> pd.DataFrame:
    """
    Produce a segment summary table with counts and
    optional observed default rates.
    """
    from models.uplift.t_learner import classify_segments
    segments = classify_segments(uplift_scores, base_risk)

    rows = []
    for seg in ["persuadable", "sure_thing", "lost_cause", "safe"]:
        mask  = segments == seg
        count = mask.sum()
        row   = {
            "segment":     seg,
            "count":       count,
            "pct":         round(count / len(segments) * 100, 1),
            "description": SEGMENT_DESCRIPTIONS[seg],
        }
        if y_true is not None and count > 0:
            row["default_rate"] = round(y_true[mask].mean() * 100, 2)
        rows.append(row)

    return pd.DataFrame(rows)