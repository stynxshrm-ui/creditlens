"""
models/scorecard/scorecard.py

Traditional credit scorecard built on WoE-transformed features.

A scorecard translates logistic regression log-odds into integer points
per feature bin. This makes the model:
  - Interpretable (each feature contributes a known number of points)
  - Auditable (regulators can inspect every decision)
  - Operationally simple (loan officers can calculate scores manually)

Score range: 300-850 (matches FICO convention, familiar to stakeholders)

Reference: Siddiqi (2006) Credit Risk Scorecards, chapters 7-8
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import mlflow
import mlflow.sklearn
from pathlib import Path

from models.scorecard.woe_encoder import WoEEncoder
from models.scorecard.evaluate import evaluate_model, plot_calibration, plot_ks


# Scorecard scaling constants
# Maps log-odds to 300-850 point range
# PDO = Points to Double Odds (industry standard: 20)
BASE_SCORE = 600
BASE_ODDS  = 1 / 19    # ~5% default rate at base score
PDO        = 20        # 20 points doubles the odds

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


# Features to include in the scorecard
# Excludes: loan_id, grade (categorical), issue_date fields,
#           vintage_default_rate (leakage risk in production),
#           default_flag (target)
SCORECARD_FEATURES = [
    # Origination
    "loan_amnt", "int_rate", "term_months", "installment",
    "annual_inc", "dti", "revol_util", "revol_bal",
    "delinq_2yrs", "open_acc", "pub_rec", "total_acc",
    "emp_length_years", "loan_to_income", "payment_to_income",
    # Payment behaviour (months 1-6)
    "avg_payment_ratio_m6", "min_payment_ratio_m6",
    "payment_volatility_m6", "active_months_m6",
    "payment_trend_m6", "first_month_ratio",
]


def log_odds_to_score(log_odds: np.ndarray) -> np.ndarray:
    """Convert logistic regression log-odds to scorecard points."""
    return OFFSET + FACTOR * log_odds


def score_to_pd(score: np.ndarray) -> np.ndarray:
    """Convert scorecard points back to probability of default."""
    log_odds = (score - OFFSET) / FACTOR
    return 1 / (1 + np.exp(log_odds))


class CreditScorecard:
    """
    WoE-based logistic regression scorecard.
    Champion model — industry baseline for credit risk.
    """

    def __init__(self, min_iv: float = 0.1):
        self.min_iv      = min_iv
        self.woe_encoder = WoEEncoder(max_bins=10, min_bin_size=0.05)
        self.model       = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight= "balanced",  # None # handles 11.9% default rate
            random_state=42,
        )
        self.selected_features_ = []
        self.iv_summary_         = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CreditScorecard":
        print("Fitting WoE encoder...")
        X_woe = self.woe_encoder.fit_transform(X[SCORECARD_FEATURES], y)

        self.iv_summary_ = self.woe_encoder.iv_summary()
        print("\nInformation Value summary:")
        print(self.iv_summary_.to_string(index=False))

        # Feature selection — only keep IV >= min_iv
        self.selected_features_ = self.woe_encoder.selected_features(self.min_iv)
        print(f"\nSelected {len(self.selected_features_)} features with IV >= {self.min_iv}")

        # Suffix _woe for column names
        woe_cols = [f"{f}_woe" for f in self.selected_features_]
        X_selected = X_woe[woe_cols]

        print("Fitting logistic regression...")
        self.model.fit(X_selected, y)
        return self

    # def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
    #     X_woe = self.woe_encoder.transform(X[SCORECARD_FEATURES])
    #     woe_cols = [f"{f}_woe" for f in self.selected_features_]
    #     return self.model.predict_proba(X_woe[woe_cols])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_woe = self.woe_encoder.transform(X[SCORECARD_FEATURES])
        woe_cols = [f"{f}_woe" for f in self.selected_features_]
        if hasattr(self, "calibrator"):
            return self.calibrator.predict_proba(X_woe[woe_cols])[:, 1]
        return self.model.predict_proba(X_woe[woe_cols])[:, 1]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return integer scorecard points (300-850)."""
        X_woe = self.woe_encoder.transform(X[SCORECARD_FEATURES])
        woe_cols = [f"{f}_woe" for f in self.selected_features_]
        log_odds = self.model.decision_function(X_woe[woe_cols])
        return np.round(log_odds_to_score(log_odds)).astype(int)


def train_scorecard(df: pd.DataFrame) -> dict:
    """
    Full scorecard training pipeline.
    Logs everything to MLflow.
    Returns metrics dict.
    """

    # Restrict to mature vintages — exclude right-censored 2017-2018
    df_train = df[df["issue_year"] <= 2016].copy()
    print(f"Training on {len(df_train):,} loans (2007-2016 vintages)")
    print(f"Held out {len(df) - len(df_train):,} loans (2017-2018, right-censored)")

    X = df_train[SCORECARD_FEATURES]
    y = df_train["default_flag"]

    # Train/test split — stratified to preserve default rate
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )
    # print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    # print(f"Train default rate: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    # Three-way split: train / calibration / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train):,} | Cal: {len(X_cal):,} | Test: {len(X_test):,}")
    print(f"Train default rate: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    with mlflow.start_run(run_name="scorecard_v1"):
        mlflow.set_tag("model_type", "woe_scorecard")
        mlflow.set_tag("champion", "true")
        mlflow.log_param("min_iv",        0.1)
        mlflow.log_param("max_bins",      10)
        mlflow.log_param("min_bin_size",  0.05)
        mlflow.log_param("train_vintages", "2007-2016")
        mlflow.log_param("train_size",    len(X_train))
        mlflow.log_param("test_size",     len(X_test))

        # Train
        scorecard = CreditScorecard(min_iv=0.1)
        scorecard.fit(X_train, y_train)

        # Fit calibration layer on held-out calibration set
        print("Calibrating probabilities...")
        X_cal_woe = scorecard.woe_encoder.transform(X_cal[SCORECARD_FEATURES])
        woe_cols = [f"{f}_woe" for f in scorecard.selected_features_]

        # scorecard.calibrator = CalibratedClassifierCV(
        #     scorecard.model,
        #     method="isotonic",
        #     cv="prefit"          # model already fitted — calibrate only
        # )

        scorecard.calibrator = CalibratedClassifierCV(
            FrozenEstimator(scorecard.model),
            method="isotonic",
        )

        scorecard.calibrator.fit(X_cal_woe[woe_cols], y_cal)


        mlflow.log_param("n_features_selected",
                          len(scorecard.selected_features_))

        # Evaluate
        y_prob = scorecard.predict_proba(X_test)
        metrics = evaluate_model(
            y_test.values, y_prob, "WoE Scorecard"
        )

        # Log IV summary as artifact
        iv_path = "iv_summary.csv"
        scorecard.iv_summary_.to_csv(iv_path, index=False)
        mlflow.log_artifact(iv_path)

        # Log calibration plot
        fig_cal = plot_calibration(y_test.values, y_prob, "WoE Scorecard")
        fig_cal.savefig("calibration.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("calibration.png")

        # Log KS plot
        fig_ks = plot_ks(y_test.values, y_prob, "WoE Scorecard")
        fig_ks.savefig("ks_plot.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("ks_plot.png")

        # Log model
        mlflow.sklearn.log_model(
            scorecard.model,
            "scorecard_model",
            registered_model_name="creditlens_champion"
        )

        print(f"\nMLflow run logged.")
        print(f"Selected features: {scorecard.selected_features_}")

    return metrics, scorecard