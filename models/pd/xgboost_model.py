"""
models/pd/xgboost_model.py

XGBoost challenger model for credit default prediction.

Challenger enters promotion workflow if Gini > champion Gini + 0.02.
Current champion (WoE Scorecard) Gini: 0.7527
Promotion threshold: Gini > 0.7727
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

from models.scorecard.evaluate import evaluate_model, plot_calibration, plot_ks

CHAMPION_GINI = 0.7527
PROMOTION_THRESHOLD = CHAMPION_GINI + 0.02

XGB_FEATURES = [
    # Origination
    "loan_amnt", "int_rate", "term_months", "installment",
    "annual_inc", "dti", "revol_util", "revol_bal",
    "delinq_2yrs", "open_acc", "pub_rec", "total_acc",
    "emp_length_years", "loan_to_income", "payment_to_income",
    "grade_numeric", "home_ownership_enc",
    "high_dti_flag", "high_util_flag", "dual_stress_flag",
    "prior_delinq_flag", "public_record_flag",
    # Payment behaviour
    "avg_payment_ratio_m6", "min_payment_ratio_m6",
    "payment_volatility_m6", "active_months_m6",
    "payment_trend_m6", "first_month_ratio",
    "last_month_ratio", "avg_monthly_change",
    "underpayment_flag", "missed_payment_flag",
    "deteriorating_flag", "early_dropout_flag",
]


def train_xgboost(df: pd.DataFrame) -> dict:
    # Restrict to mature vintages
    df_train = df[df["issue_year"] <= 2016].copy()
    print(f"Training on {len(df_train):,} loans (2007-2016 vintages)")

    # Only keep features that exist in the dataframe
    features = [f for f in XGB_FEATURES if f in df_train.columns]
    print(f"Using {len(features)} features")

    X = df_train[features]
    y = df_train["default_flag"]

    # Three-way split — train / calibration / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train):,} | Cal: {len(X_cal):,} | Test: {len(X_test):,}")

    with mlflow.start_run(run_name="xgboost_challenger_v1"):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("champion", "false")
        mlflow.set_tag("challenger", "true")

        params = {
            "n_estimators":     300,
            "max_depth":        4,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "random_state":     42,
            "eval_metric":      "auc",
        }
        mlflow.log_params(params)
        mlflow.log_param("train_vintages", "2007-2016")
        mlflow.log_param("n_features", len(features))

        # Train
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Calibrate
        print("Calibrating...")
        if FrozenEstimator is not None:
            calibrator = CalibratedClassifierCV(
                FrozenEstimator(model), method="isotonic"
            )
        else:
            calibrator = CalibratedClassifierCV(
                model, method="isotonic", cv="prefit"
            )
        calibrator.fit(X_cal, y_cal)

        # Evaluate
        y_prob = calibrator.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_prob=y_prob,
                                  y_true=y_test.values,
                                  model_name="XGBoost Challenger")
        mlflow.log_param("champion_gini",    CHAMPION_GINI)
        mlflow.log_param("promotion_threshold", PROMOTION_THRESHOLD)

        # SHAP values
        print("\nComputing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        fig_shap, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_values, X_test,
            max_display=15,
            show=False,
            plot_type="bar"
        )
        plt.tight_layout()
        plt.savefig("shap_importance.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("shap_importance.png")
        plt.close()

        # Calibration + KS plots
        fig_cal = plot_calibration(y_test.values, y_prob, "XGBoost")
        fig_cal.savefig("calibration_xgb.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("calibration_xgb.png")
        plt.close()

        fig_ks = plot_ks(y_test.values, y_prob, "XGBoost")
        fig_ks.savefig("ks_plot_xgb.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("ks_plot_xgb.png")
        plt.close()

        # Log model
        mlflow.xgboost.log_model(
            model, "xgboost_model",
            registered_model_name="creditlens_challenger"
        )

        # Promotion check
        print(f"\nChampion Gini:    {CHAMPION_GINI:.4f}")
        print(f"Challenger Gini:  {metrics['gini']:.4f}")
        print(f"Promotion threshold: {PROMOTION_THRESHOLD:.4f}")

        if metrics["gini"] > PROMOTION_THRESHOLD:
            print("PROMOTION CRITERIA MET — challenger qualifies for shadow deployment")
            mlflow.set_tag("promotion_eligible", "true")
        else:
            print("Promotion criteria NOT met — champion retained")
            mlflow.set_tag("promotion_eligible", "false")

        # Fairness check — Gini across income bands
        print("\nFairness check — Gini by income band:")
        test_df = X_test.copy()
        test_df["y_true"] = y_test.values
        test_df["y_prob"] = y_prob

        test_df["income_band"] = pd.cut(
            test_df["annual_inc"],
            bins=[0, 40000, 70000, 100000, float("inf")],
            labels=["<40k", "40-70k", "70-100k", ">100k"]
        )

        from sklearn.metrics import roc_auc_score
        print(f"  {'Income band':<12} {'Gini':>8} {'Count':>8}")
        for band, grp in test_df.groupby("income_band", observed=True):
            if len(grp) > 50 and grp["y_true"].nunique() == 2:
                g = round(2 * roc_auc_score(
                    grp["y_true"], grp["y_prob"]) - 1, 3)
                print(f"  {str(band):<12} {g:>8.3f} {len(grp):>8,}")
                safe_band = str(band).replace("<", "lt").replace(">", "gt").replace("-", "_")
                mlflow.log_metric(f"gini_{safe_band}", g)

        mlflow.log_metric("gini_overall", metrics["gini"])

    return metrics, model, calibrator, features