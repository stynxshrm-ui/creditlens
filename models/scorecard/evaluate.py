"""
models/scorecard/evaluate.py

Credit risk model evaluation metrics.

Gini and KS are the industry-standard discrimination metrics.
Calibration measures whether predicted probabilities match observed rates.
All metrics logged to MLflow automatically.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from scipy import stats


def gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Gini coefficient = 2 * AUC - 1
    Industry standard discrimination metric for credit scorecards.
    Range: 0 (random) to 1 (perfect). Typical production models: 0.4-0.7
    """
    auc = roc_auc_score(y_true, y_prob)
    return round(2 * auc - 1, 4)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic.
    Maximum separation between default and non-default score distributions.
    Range: 0 (no separation) to 1 (perfect separation).
    Typical production models: 0.3-0.5
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return round(max(tpr - fpr), 4)


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                       n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Measures whether predicted probabilities match observed default rates.
    IFRS 9 requires calibrated PD estimates — this quantifies the gap.
    Lower is better. < 0.02 is good for credit models.
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    return round(float(np.mean(np.abs(fraction_pos - mean_pred))), 4)


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str, log_to_mlflow: bool = True) -> dict:
    """
    Full evaluation suite. Logs all metrics to MLflow if active run exists.
    Returns dict of metrics for programmatic use.
    """
    metrics = {
        "gini":              gini(y_true, y_prob),
        "ks":                ks_statistic(y_true, y_prob),
        "auc":               round(roc_auc_score(y_true, y_prob), 4),
        "calibration_error": calibration_error(y_true, y_prob),
        "default_rate_actual":   round(float(y_true.mean()), 4),
        "default_rate_predicted": round(float(y_prob.mean()), 4),
    }

    print(f"\n{'='*40}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*40}")
    print(f"  Gini:              {metrics['gini']:.4f}")
    print(f"  KS statistic:      {metrics['ks']:.4f}")
    print(f"  AUC:               {metrics['auc']:.4f}")
    print(f"  Calibration error: {metrics['calibration_error']:.4f}")
    print(f"  Actual default rate:    {metrics['default_rate_actual']:.2%}")
    print(f"  Predicted default rate: {metrics['default_rate_predicted']:.2%}")
    print(f"{'='*40}")

    if log_to_mlflow:
        try:
            mlflow.log_metrics(metrics)
        except Exception:
            pass  # No active run — metrics printed only

    return metrics


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                     model_name: str) -> plt.Figure:
    """
    Calibration curve — predicted probability vs observed default rate.
    Perfect calibration = diagonal line.
    IFRS 9 context: regulators require PD estimates to match observed rates.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )

    ax.plot(mean_pred, fraction_pos, "o-", label=model_name, color="#2563EB")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_ks(y_true: np.ndarray, y_prob: np.ndarray,
            model_name: str) -> plt.Figure:
    """
    KS plot — cumulative distribution of scores for defaults vs non-defaults.
    The maximum vertical gap between the two curves is the KS statistic.
    """
    scores_default     = y_prob[y_true == 1]
    scores_nondefault  = y_prob[y_true == 0]

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.linspace(0, 1, 200)
    ax.plot(x, [np.mean(scores_nondefault <= t) for t in x],
            label="Non-default", color="#16A34A")
    ax.plot(x, [np.mean(scores_default <= t) for t in x],
            label="Default", color="#DC2626")

    ks = ks_statistic(y_true, y_prob)
    ax.set_title(f"KS Plot — {model_name} (KS={ks:.3f})")
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Cumulative proportion")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig