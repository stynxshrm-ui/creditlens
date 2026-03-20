"""
models/scorecard/woe_encoder.py

Weight of Evidence (WoE) encoder for credit scorecard development.

WoE transforms each feature into a single numeric value that represents
the log odds of the target within a bin. This has three benefits:
  1. Handles non-linear relationships automatically through binning
  2. Produces monotonic feature-target relationships (regulatory requirement)
  3. Handles missing values naturally (missing = separate bin)

Information Value (IV) measures each feature's predictive power:
  IV < 0.02  — useless
  IV 0.02-0.1 — weak
  IV 0.1-0.3  — medium
  IV > 0.3    — strong (possibly too good, check for leakage)

Reference: Siddiqi (2006) Credit Risk Scorecards
"""

import pandas as pd
import numpy as np
from optbinning import OptimalBinning
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WoEEncoder:
    """
    Fits WoE bins per feature and transforms raw values to WoE values.
    Tracks Information Value (IV) per feature for selection.
    """

    def __init__(self, max_bins: int = 10, min_bin_size: float = 0.05):
        """
        max_bins    — maximum number of bins per feature
        min_bin_size — minimum fraction of population per bin
                       0.05 = each bin must have at least 5% of records
                       (regulatory requirement — prevents overfitting to tiny segments)
        """
        self.max_bins     = max_bins
        self.min_bin_size = min_bin_size
        self.binners_     = {}   # fitted OptimalBinning per feature
        self.iv_          = {}   # Information Value per feature
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        """Fit WoE bins for each feature in X."""
        self.feature_names_ = list(X.columns)

        for col in X.columns:
            logger.info(f"Fitting WoE bins for {col}")

            binner = OptimalBinning(
                name=col,
                dtype="numerical",
                max_n_bins=self.max_bins,
                min_bin_size=self.min_bin_size,
                monotonic_trend="auto",   # enforces monotonic WoE — regulatory
            )

            # OptimalBinning expects numpy arrays
            binner.fit(X[col].values, y.values)
            self.binners_[col] = binner

            # Extract IV from binning table
            bt = binner.binning_table.build()
            # IV is in the last row of the binning table
            self.iv_[col] = bt["IV"].iloc[-1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw feature values to WoE values."""
        result = {}
        for col in self.feature_names_:
            if col not in self.binners_:
                raise ValueError(f"Feature {col} was not seen during fit")
            binner = self.binners_[col]
            result[f"{col}_woe"] = binner.transform(
                X[col].values, metric="woe"
            )
        return pd.DataFrame(result, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def iv_summary(self) -> pd.DataFrame:
        """Return IV values sorted descending — use for feature selection."""
        df = pd.DataFrame({
            "feature": list(self.iv_.keys()),
            "iv":      list(self.iv_.values()),
        }).sort_values("iv", ascending=False)

        df["strength"] = df["iv"].map(lambda v:
            "strong"  if v > 0.3  else
            "medium"  if v > 0.1  else
            "weak"    if v > 0.02 else
            "useless"
        )
        return df.reset_index(drop=True)

    def selected_features(self, min_iv: float = 0.1) -> list:
        """Return feature names with IV above threshold."""
        return [f for f, iv in self.iv_.items() if iv >= min_iv]