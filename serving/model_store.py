"""
serving/model_store.py

Loads and caches trained models at API startup.
Models are loaded once — not per request.

In production this would pull from MLflow Model Registry.
Here we load from local mlruns for simplicity.
"""

import mlflow
import mlflow.xgboost
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import logging
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

logger = logging.getLogger(__name__)

# Feature lists — must match training exactly
from models.pd.xgboost_model import XGB_FEATURES
from models.uplift.t_learner import UPLIFT_FEATURES
from models.scorecard.scorecard import (
    CreditScorecard, SCORECARD_FEATURES, log_odds_to_score
)


class ModelStore:
    """
    Singleton model store. Loaded once at API startup.
    Holds champion PD model, scorecard, and uplift models.
    """

    def __init__(self):
        self.xgb_model      = None
        self.xgb_calibrator = None
        self.xgb_features   = None
        self.uplift_control   = None
        self.uplift_treatment = None
        self.uplift_features  = None
        self.scorecard        = None
        self.is_loaded        = False
        self.model_info       = {}

    def load(self):
        """Load all models from MLflow registry."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        try:
            self._load_xgboost()
            self._load_uplift()
            self.is_loaded = True
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _get_experiment_id(self, client) -> str:
        """Get experiment ID by name — avoids hardcoding."""
        exp = client.get_experiment_by_name("creditlens")
        if exp is None:
            raise ValueError("MLflow experiment 'creditlens' not found")
        return exp.experiment_id

    def _load_xgboost(self):
        """Load latest production XGBoost model."""
        logger.info("Loading XGBoost challenger model...")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        # Get latest version dynamically
        versions = client.search_model_versions("name='creditlens_challenger'")
        latest = max(versions, key=lambda v: int(v.version))
        latest_version = latest.version

        self.xgb_model = mlflow.xgboost.load_model(
            f"models:/creditlens_challenger/{latest_version}"
        )
        logger.info(f"XGBoost loaded from registry version {latest_version}") 

        self.xgb_features = [
            f for f in XGB_FEATURES
            if f in self.xgb_model.get_booster().feature_names
        ]

        # Get metrics from latest run for /v1/model/info endpoint
        exp = client.get_experiment_by_name("creditlens")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.model_type = 'xgboost'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs:
            run = runs[0]
            self.model_info = {
                "champion_model":    "XGBoost",
                "model_version":     f"v{latest_version}",
                "training_vintage":  "2007-2016",
                "gini":              run.data.metrics.get("gini", 0),
                "ks":                run.data.metrics.get("ks", 0),
                "calibration_error": run.data.metrics.get("calibration_error", 0),
                "registered_at":     str(run.info.start_time),
            }

        logger.info(f"XGBoost loaded — run v{latest_version}")

    def _load_uplift(self):
        """Load uplift T-Learner models from MLflow registry."""
        logger.info("Loading uplift T-Learner models...")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        try:
            # Get latest versions
            control_versions = client.search_model_versions(
                "name='creditlens_uplift_control'"
            )
            treatment_versions = client.search_model_versions(
                "name='creditlens_uplift_treatment'"
            )

            if not control_versions or not treatment_versions:
                logger.warning("Uplift models not in registry — uplift disabled")
                return

            latest_control   = max(control_versions,
                                key=lambda v: int(v.version)).version
            latest_treatment = max(treatment_versions,
                                key=lambda v: int(v.version)).version

            self.uplift_control = mlflow.xgboost.load_model(
                f"models:/creditlens_uplift_control/{latest_control}"
            )
            self.uplift_treatment = mlflow.xgboost.load_model(
                f"models:/creditlens_uplift_treatment/{latest_treatment}"
            )
            self.uplift_features = UPLIFT_FEATURES
            logger.info(f"Uplift models loaded — "
                        f"control v{latest_control}, "
                        f"treatment v{latest_treatment}")

        except Exception as e:
            logger.warning(f"Uplift loading failed: {e} — uplift disabled")
            

    def score(self, features: dict) -> dict:
        """
        Score a single borrower.
        Returns pd_score, credit_score, risk_tier, uplift_score, segment.
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        X = self._prepare_features(features, XGB_FEATURES)

        # PD score from XGBoost
        pd_score = float(self.xgb_model.predict_proba(X)[:, 1][0])

        # Credit score from log-odds
        log_odds = float(self.xgb_model.get_booster().predict(
            __import__("xgboost").DMatrix(X),
            output_margin=True
        )[0])
        credit_score = int(np.clip(
            log_odds_to_score(np.array([log_odds]))[0], 300, 850
        ))

        # Risk tier
        risk_tier = self._pd_to_tier(pd_score)

        # Uplift score
        uplift_score = None
        segment      = "unknown"

        if self.uplift_control and self.uplift_treatment:
            X_uplift = self._prepare_features(features, UPLIFT_FEATURES)
            p_control   = float(
                self.uplift_control.predict_proba(X_uplift)[:, 1][0]
            )
            p_treatment = float(
                self.uplift_treatment.predict_proba(X_uplift)[:, 1][0]
            )
            uplift_score = round(p_control - p_treatment, 4)

            from models.uplift.t_learner import classify_segments
            segments = classify_segments(
                np.array([uplift_score]),
                np.array([pd_score])
            )
            segment = segments[0]

        return {
            "pd_score":      round(pd_score, 4),
            "credit_score":  credit_score,
            "risk_tier":     risk_tier,
            "uplift_score":  uplift_score,
            "segment":       segment,
            "model_version": self.model_info.get("model_version", "unknown"),
            "champion_model": self.model_info.get("champion_model", "XGBoost"),
        }

    def _prepare_features(self, features: dict,
                        feature_list: list) -> pd.DataFrame:
        """Prepare feature dict as DataFrame for model input."""
        row = {f: features.get(f) for f in feature_list}
        df = pd.DataFrame([row])
        # Fill None/NaN with 0 and ensure numeric dtype
        df = df.fillna(0).astype(float)
        return df

    def _pd_to_tier(self, pd_score: float) -> str:
        """Map PD score to risk tier A-E."""
        if pd_score < 0.05:   return "A"
        if pd_score < 0.10:   return "B"
        if pd_score < 0.15:   return "C"
        if pd_score < 0.20:   return "D"
        return "E"


# Singleton instance
model_store = ModelStore()