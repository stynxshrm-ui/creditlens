"""
Drift detection module for CreditLens using PSI (Population Stability Index).

PSI is the credit industry standard for monitoring feature and prediction drift.
Thresholds:
- PSI < 0.1: Stable, no action needed
- 0.1 <= PSI < 0.25: Monitor, investigate if sustained
- PSI >= 0.25: Significant drift, trigger retraining

Regulatory context: Model risk guidance (SR 11-7) requires ongoing monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class PSIResult:
    """Container for PSI calculation results."""
    feature_name: str
    psi_value: float
    status: str  # 'stable', 'monitor', 'retrain'
    reference_period: str
    current_period: str
    bin_contributions: Dict[str, float]
    timestamp: datetime


class DriftDetector:
    """
    Calculate Population Stability Index (PSI) for feature and prediction drift.
    
    PSI formula:
    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    
    where actual_pct and expected_pct are the distributions in bins.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize drift detector.
        
        Args:
            n_bins: Number of bins for discretization (default 10, standard for PSI)
        """
        self.n_bins = n_bins
        self.reference_distributions = {}
        
    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
        bins: Optional[np.ndarray] = None
    ) -> PSIResult:
        """
        Calculate PSI between reference and current distributions.
        
        Args:
            reference: Reference distribution (training or previous period)
            current: Current distribution (new data)
            feature_name: Name of the feature being monitored
            bins: Optional pre-defined bin edges (use same bins as reference)
            
        Returns:
            PSIResult with PSI value, status, and bin-level contributions
        """
        # Handle missing values
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return PSIResult(
                feature_name=feature_name,
                psi_value=np.nan,
                status='error',
                reference_period='unknown',
                current_period='unknown',
                bin_contributions={},
                timestamp=datetime.now()
            )
        
        # Create bins from reference distribution if not provided
        if bins is None:
            bins = np.percentile(reference, np.linspace(0, 100, self.n_bins + 1))
            bins = np.unique(bins)  # Remove duplicate edges
            
        # Store bins for future use
        self.reference_distributions[feature_name] = bins
        
        # Bin the data
        ref_binned = np.digitize(reference, bins, right=True)
        cur_binned = np.digitize(current, bins, right=True)
        
        # Calculate distributions
        ref_counts = np.bincount(ref_binned, minlength=len(bins) + 1)
        cur_counts = np.bincount(cur_binned, minlength=len(bins) + 1)
        
        # Convert to percentages (add small epsilon to avoid log(0))
        epsilon = 1e-5
        ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * len(bins))
        cur_pct = (cur_counts + epsilon) / (len(current) + epsilon * len(bins))
        
        # Calculate PSI per bin
        psi_per_bin = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
        psi_total = np.sum(psi_per_bin)
        
        # Determine status based on thresholds
        if psi_total < 0.1:
            status = 'stable'
        elif psi_total < 0.25:
            status = 'monitor'
        else:
            status = 'retrain'
            
        # Create bin contribution dictionary for interpretability
        bin_contributions = {
            f"bin_{i}": float(psi_per_bin[i]) 
            for i in range(len(psi_per_bin))
            if psi_per_bin[i] > 0.01  # Only significant contributions
        }
        
        return PSIResult(
            feature_name=feature_name,
            psi_value=float(psi_total),
            status=status,
            reference_period='training',  # Update with actual period tracking
            current_period='current',
            bin_contributions=bin_contributions,
            timestamp=datetime.now()
        )
    
    def monitor_features(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, PSIResult]:
        """
        Monitor multiple features for drift.
        
        Args:
            reference_df: Reference dataset (training data)
            current_df: Current dataset (new production data)
            features: List of feature names to monitor
            
        Returns:
            Dictionary of feature_name -> PSIResult
        """
        results = {}
        
        for feature in features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                print(f"Warning: {feature} not found in both datasets, skipping")
                continue
                
            # Use stored bins if available (ensures consistent binning)
            bins = self.reference_distributions.get(feature, None)
            
            psi_result = self.calculate_psi(
                reference=reference_df[feature].values,
                current=current_df[feature].values,
                feature_name=feature,
                bins=bins
            )
            
            results[feature] = psi_result
            
        return results
    
    def monitor_predictions(
        self,
        reference_preds: np.ndarray,
        current_preds: np.ndarray
    ) -> PSIResult:
        """
        Monitor model prediction distribution for drift.
        
        This is critical: even if features are stable, the model's output
        distribution can shift (calibration drift).
        
        Args:
            reference_preds: Predictions on reference dataset
            current_preds: Predictions on current dataset
            
        Returns:
            PSIResult for prediction drift
        """
        return self.calculate_psi(
            reference=reference_preds,
            current=current_preds,
            feature_name='model_predictions'
        )
    
    def get_drift_summary(
        self,
        results: Dict[str, PSIResult]
    ) -> Dict[str, any]:
        """
        Generate summary of drift monitoring results.
        
        Args:
            results: Dictionary of PSIResults from monitor_features
            
        Returns:
            Summary dictionary with counts and actionable features
        """
        stable_count = sum(1 for r in results.values() if r.status == 'stable')
        monitor_count = sum(1 for r in results.values() if r.status == 'monitor')
        retrain_count = sum(1 for r in results.values() if r.status == 'retrain')
        
        # Features requiring retraining
        retrain_features = [
            name for name, result in results.items() 
            if result.status == 'retrain'
        ]
        
        # Features to monitor
        monitor_features = [
            name for name, result in results.items() 
            if result.status == 'monitor'
        ]
        
        # Overall recommendation
        if retrain_count > 0:
            recommendation = 'RETRAIN'
            reason = f"{retrain_count} feature(s) exceed PSI threshold of 0.25"
        elif monitor_count >= len(results) * 0.3:  # >30% in monitor zone
            recommendation = 'INVESTIGATE'
            reason = f"{monitor_count} feature(s) in monitoring zone"
        else:
            recommendation = 'STABLE'
            reason = "All features within acceptable drift limits"
            
        return {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(results),
            'stable_count': stable_count,
            'monitor_count': monitor_count,
            'retrain_count': retrain_count,
            'retrain_features': retrain_features,
            'monitor_features': monitor_features,
            'recommendation': recommendation,
            'reason': reason,
            'psi_values': {name: result.psi_value for name, result in results.items()}
        }
    
    def save_results(
        self,
        results: Dict[str, PSIResult],
        filepath: str
    ):
        """Save drift monitoring results to JSON."""
        output = {
            'summary': self.get_drift_summary(results),
            'detailed_results': {
                name: {
                    'psi_value': result.psi_value,
                    'status': result.status,
                    'bin_contributions': result.bin_contributions,
                    'timestamp': result.timestamp.isoformat()
                }
                for name, result in results.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Drift results saved to {filepath}")


# Credit-specific feature monitoring configuration
CREDIT_MONITORED_FEATURES = [
    'dti',                    # Debt-to-income - economic indicator
    'annual_income',          # Income distribution shifts
    'revol_util',             # Credit utilization behavior
    'loan_amount',            # Loan sizing trends
    'delinq_2yrs',           # Historical delinquency patterns
    'open_acc',              # Credit profile complexity
    'avg_payment_ratio_m1_6', # Payment behavior (if available)
]


def demonstrate_psi_calculation():
    """
    Demonstration of PSI calculation with synthetic data.
    
    This shows what different PSI values mean in practice.
    """
    print("=== PSI Demonstration ===\n")
    
    # Create reference distribution (training data)
    np.random.seed(42)
    reference = np.random.normal(loc=50000, scale=15000, size=10000)  # Income
    
    detector = DriftDetector(n_bins=10)
    
    # Scenario 1: No drift
    current_stable = np.random.normal(loc=50000, scale=15000, size=1000)
    result_stable = detector.calculate_psi(reference, current_stable, 'annual_income')
    print(f"Scenario 1 - No drift:")
    print(f"  PSI: {result_stable.psi_value:.4f}")
    print(f"  Status: {result_stable.status}")
    print()
    
    # Scenario 2: Slight drift (monitor zone)
    current_slight = np.random.normal(loc=52000, scale=16000, size=1000)
    result_slight = detector.calculate_psi(reference, current_slight, 'annual_income')
    print(f"Scenario 2 - Slight drift:")
    print(f"  PSI: {result_slight.psi_value:.4f}")
    print(f"  Status: {result_slight.status}")
    print()
    
    # Scenario 3: Significant drift (retrain zone)
    current_drift = np.random.normal(loc=60000, scale=20000, size=1000)
    result_drift = detector.calculate_psi(reference, current_drift, 'annual_income')
    print(f"Scenario 3 - Significant drift:")
    print(f"  PSI: {result_drift.psi_value:.4f}")
    print(f"  Status: {result_drift.status}")
    print(f"  Top bin contributions: {list(result_drift.bin_contributions.items())[:3]}")
    print()


if __name__ == '__main__':
    demonstrate_psi_calculation()
