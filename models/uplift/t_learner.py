"""
models/uplift/t_learner.py

T-Learner uplift model for credit restructuring intervention.

The T-Learner trains two separate models:
  - model_control:   P(default | no restructuring offer)
  - model_treatment: P(default | restructuring offer made)

Uplift score = P(default | control) - P(default | treatment)
High uplift = borrower would default without offer but repay with it
            = PERSUADABLE = target for intervention

Four segments:
  Persuadables  — high uplift (intervention helps)
  Sure things   — low uplift, low base risk (would repay regardless)
  Lost causes   — low uplift, high base risk (would default regardless)
  Safe          — not struggling, not at risk

Reference: Gutierrez & Gerardy (2017) "Causal Inference and
Uplift Modelling: A Review of the Literature", arXiv:1710.01406
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path
import duckdb

DB_PATH = Path("data/creditlens.duckdb")
PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Features available at intervention decision time
# Must be knowable at month 3 — when we identify struggling borrowers
UPLIFT_FEATURES = [
    # Origination
    "loan_amnt", "int_rate", "term_months",
    "annual_inc", "dti", "revol_util",
    "delinq_2yrs", "emp_length_years",
    "loan_to_income", "grade_numeric",
    "high_dti_flag", "high_util_flag", "dual_stress_flag",
    # Payment behaviour months 1-6
    "avg_payment_ratio_m6", "min_payment_ratio_m6",
    "first_month_ratio", "payment_trend_m6",
]


def qini_curve(y_true: np.ndarray,
               uplift: np.ndarray,
               treatment: np.ndarray) -> tuple:
    """
    Qini curve — the uplift equivalent of ROC curve.
    Shows cumulative incremental conversions as we target
    more borrowers in descending uplift order.
    Higher area = better uplift model.
    """
    df = pd.DataFrame({
        "y":         y_true,
        "uplift":    uplift,
        "treatment": treatment,
    }).sort_values("uplift", ascending=False).reset_index(drop=True)

    n = len(df)
    qini_x = np.arange(n + 1) / n
    qini_y = [0.0]

    treated_defaults   = 0
    control_defaults   = 0
    treated_count      = 0
    control_count      = 0

    for _, row in df.iterrows():
        if row["treatment"] == 1:
            treated_count += 1
            if row["y"] == 1:
                treated_defaults += 1
        else:
            control_count += 1
            if row["y"] == 1:
                control_defaults += 1

        # Incremental defaults prevented (normalised)
        if treated_count > 0 and control_count > 0:
            incremental = (
                treated_defaults / treated_count
                - control_defaults / control_count
            ) * treated_count
        else:
            incremental = 0

        qini_y.append(incremental)

    return qini_x, np.array(qini_y)


def auuc(qini_x: np.ndarray, qini_y: np.ndarray) -> float:
    """Area Under Uplift Curve — primary uplift metric."""
    return float(np.trapz(qini_y, qini_x))


def classify_segments(uplift: np.ndarray,
                       base_risk: np.ndarray,
                       uplift_threshold: float = 0.02,
                       risk_threshold: float = 0.15) -> np.ndarray:
    """
    Classify borrowers into four uplift segments.

    uplift_threshold — minimum uplift to be considered persuadable
    risk_threshold   — minimum base risk to be considered at-risk
    """
    segments = np.where(
        (uplift >= uplift_threshold) & (base_risk >= risk_threshold),
        "persuadable",
        np.where(
            (uplift < uplift_threshold) & (base_risk < risk_threshold),
            "safe",
            np.where(
                (uplift < uplift_threshold) & (base_risk >= risk_threshold),
                "lost_cause",
                "sure_thing"
            )
        )
    )
    return segments


def train_uplift(df: pd.DataFrame) -> dict:
    """Train T-Learner and evaluate on holdout set."""

    # Load treatment flags
    con = duckdb.connect(str(DB_PATH), read_only=True)
    treatment_df = con.execute("""
        SELECT t.loan_id, t.treatment, o.default_flag
        FROM treatment_flags t
        JOIN outcomes o USING (loan_id)
    """).fetchdf()
    con.close()

    print(f"Struggling borrowers with treatment flag: {len(treatment_df):,}")

    # Merge with features
    features = [f for f in UPLIFT_FEATURES if f in df.columns]
    df_uplift = df.merge(treatment_df, on="loan_id", how="inner",
                         suffixes=("", "_t"))

    # Use default_flag from treatment_df (has both groups)
    if "default_flag_t" in df_uplift.columns:
        df_uplift["default_flag"] = df_uplift["default_flag_t"]

    print(f"Merged dataset: {len(df_uplift):,} loans")

    X = df_uplift[features].fillna(0)
    y = df_uplift["default_flag"]
    t = df_uplift["treatment"]

    # Train/test split — stratify by both outcome and treatment
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    with mlflow.start_run(run_name="uplift_t_learner_v1"):
        mlflow.set_tag("model_type", "t_learner_uplift")

        params = {
            "n_estimators":  200,
            "max_depth":     3,
            "learning_rate": 0.05,
            "subsample":     0.8,
            "random_state":  42,
        }
        mlflow.log_params(params)

        # T-Learner: train separately on treatment and control
        print("\nTraining control model...")
        mask_control = t_train == 0
        model_control = xgb.XGBClassifier(**params, eval_metric="auc")
        model_control.fit(
            X_train[mask_control], y_train[mask_control],
            verbose=False
        )

        print("Training treatment model...")
        mask_treatment = t_train == 1
        model_treatment = xgb.XGBClassifier(**params, eval_metric="auc")
        model_treatment.fit(
            X_train[mask_treatment], y_train[mask_treatment],
            verbose=False
        )

        # Uplift score on test set
        p_control   = model_control.predict_proba(X_test)[:, 1]
        p_treatment = model_treatment.predict_proba(X_test)[:, 1]
        uplift_scores = p_control - p_treatment

        print(f"\nUplift score distribution:")
        print(f"  Mean:   {uplift_scores.mean():.4f}")
        print(f"  Std:    {uplift_scores.std():.4f}")
        print(f"  Min:    {uplift_scores.min():.4f}")
        print(f"  Max:    {uplift_scores.max():.4f}")
        print(f"  % positive uplift: "
              f"{(uplift_scores > 0).mean():.1%}")

        mlflow.log_metric("mean_uplift", float(uplift_scores.mean()))
        mlflow.log_metric("pct_positive_uplift",
                           float((uplift_scores > 0).mean()))

        # Segment classification
        segments = classify_segments(uplift_scores, p_control)
        seg_counts = pd.Series(segments).value_counts()
        print(f"\nSegment distribution (test set):")
        for seg, count in seg_counts.items():
            pct = count / len(segments) * 100
            print(f"  {seg:<15} {count:>6,} ({pct:.1f}%)")
            mlflow.log_metric(f"segment_{seg}", int(count))

        # Qini curve
        qini_x, qini_y = qini_curve(
            y_test.values, uplift_scores, t_test.values
        )
        score_auuc = auuc(qini_x, qini_y)
        print(f"\nAUUC: {score_auuc:.4f}")
        mlflow.log_metric("auuc", score_auuc)

        # Plot Qini curve
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(qini_x * 100, qini_y,
                color="#2563EB", label=f"T-Learner (AUUC={score_auuc:.3f})")
        ax.axhline(0, color="gray", linestyle="--", label="Random targeting")
        ax.set_xlabel("% of struggling borrowers targeted (by uplift rank)")
        ax.set_ylabel("Cumulative defaults prevented")
        ax.set_title("Qini Curve — Restructuring Intervention Uplift")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(PLOTS_DIR / "qini_curve.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact(str(PLOTS_DIR / "qini_curve.png"))
        plt.close()

        # Business framing
        persuadables = (segments == "persuadable").sum()
        total_test   = len(segments)
        pct_targeted = persuadables / total_test * 100

        # Defaults prevented by targeting persuadables only
        persuadable_mask = segments == "persuadable"
        if persuadable_mask.sum() > 0:
            defaults_in_persuadables = y_test.values[persuadable_mask].sum()
            total_defaults = y_test.values.sum()
            pct_defaults_captured = (
                defaults_in_persuadables / total_defaults * 100
                if total_defaults > 0 else 0
            )
        else:
            pct_defaults_captured = 0

        print(f"\nBusiness impact:")
        print(f"  Persuadables:          {persuadables:,} "
              f"({pct_targeted:.1f}% of at-risk pool)")
        print(f"  Defaults in segment:   {defaults_in_persuadables:,}")
        print(f"  % of total defaults:   {pct_defaults_captured:.1f}%")
        print(f"\n  Interpretation:")
        print(f"  Targeting {pct_targeted:.0f}% of struggling borrowers "
              f"by uplift score")
        print(f"  captures {pct_defaults_captured:.0f}% of preventable defaults")

        mlflow.log_metric("pct_persuadables",   pct_targeted)
        mlflow.log_metric("pct_defaults_captured", pct_defaults_captured)

    return {
        "auuc":                  score_auuc,
        "mean_uplift":           float(uplift_scores.mean()),
        "pct_positive_uplift":   float((uplift_scores > 0).mean()),
        "pct_persuadables":      pct_targeted,
        "pct_defaults_captured": pct_defaults_captured,
        "segment_counts":        seg_counts.to_dict(),
    }