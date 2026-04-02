"""
CreditLens Drift Detection Integration Example

Demonstrates complete workflow:
1. Simulate reference and current datasets
2. Run PSI calculation on credit features
3. Evaluate retraining triggers
4. Generate reports

This serves as:
- Week 7 deliverable testing framework
- Integration test for pipeline components
- Documentation of expected behavior
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Import drift detection components
# In actual project structure: from pipeline import DriftDetector, RetrainingTrigger
# For this demo, we'll implement simplified versions inline


def create_reference_portfolio(n_loans: int = 10000) -> pd.DataFrame:
    """
    Create synthetic reference dataset matching Lending Club characteristics.
    
    This represents the training data distribution (e.g., Q1 2017 vintage).
    """
    np.random.seed(42)
    
    # Credit-realistic distributions
    df = pd.DataFrame({
        'loan_id': [f'LC_{i:06d}' for i in range(n_loans)],
        
        # DTI: debt-to-income ratio (typically 5-45%)
        'dti': np.clip(np.random.normal(18, 8, n_loans), 0, 45),
        
        # Annual income ($20k - $200k, right-skewed)
        'annual_income': np.random.lognormal(10.8, 0.6, n_loans),
        
        # Revolving utilization (0-100%)
        'revol_util': np.clip(np.random.beta(2, 2, n_loans) * 100, 0, 100),
        
        # Loan amount ($1k - $40k)
        'loan_amount': np.random.lognormal(9.5, 0.5, n_loans),
        
        # Delinquencies in past 2 years (0-10, most are 0)
        'delinq_2yrs': np.random.poisson(0.3, n_loans),
        
        # Open accounts (2-30)
        'open_acc': np.clip(np.random.normal(10, 5, n_loans), 2, 30).astype(int),
        
        # Grade distribution (A-G)
        'grade': np.random.choice(
            ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            n_loans,
            p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.03, 0.02]
        )
    })
    
    return df


def create_current_portfolio_stable(n_loans: int = 1000) -> pd.DataFrame:
    """
    Create current portfolio with NO drift (stable scenario).
    """
    np.random.seed(123)
    
    # Same distributions as reference
    df = pd.DataFrame({
        'loan_id': [f'LC_NEW_{i:06d}' for i in range(n_loans)],
        'dti': np.clip(np.random.normal(18, 8, n_loans), 0, 45),
        'annual_income': np.random.lognormal(10.8, 0.6, n_loans),
        'revol_util': np.clip(np.random.beta(2, 2, n_loans) * 100, 0, 100),
        'loan_amount': np.random.lognormal(9.5, 0.5, n_loans),
        'delinq_2yrs': np.random.poisson(0.3, n_loans),
        'open_acc': np.clip(np.random.normal(10, 5, n_loans), 2, 30).astype(int),
        'grade': np.random.choice(
            ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            n_loans,
            p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.03, 0.02]
        )
    })
    
    return df


def create_current_portfolio_drift(n_loans: int = 1000, drift_type: str = 'moderate') -> pd.DataFrame:
    """
    Create current portfolio with DRIFT.
    
    Args:
        n_loans: Number of loans
        drift_type: 'slight', 'moderate', or 'severe'
    """
    np.random.seed(456)
    
    # Economic scenario: Rising rates, tighter lending
    if drift_type == 'slight':
        dti_shift = 1.0  # 1pp increase in average DTI
        income_shift = 1.05  # 5% increase in average income
        util_shift = 0.02  # 2pp increase in utilization
        
    elif drift_type == 'moderate':
        dti_shift = 2.5  # 2.5pp increase in DTI
        income_shift = 1.15  # 15% income increase
        util_shift = 0.08  # 8pp utilization increase
        
    else:  # severe
        dti_shift = 5.0  # 5pp DTI increase
        income_shift = 1.30  # 30% income increase (different borrower pool)
        util_shift = 0.15  # 15pp utilization increase
    
    df = pd.DataFrame({
        'loan_id': [f'LC_NEW_{i:06d}' for i in range(n_loans)],
        
        # Shifted distributions
        'dti': np.clip(np.random.normal(18 + dti_shift, 8, n_loans), 0, 45),
        'annual_income': np.random.lognormal(10.8, 0.6, n_loans) * income_shift,
        'revol_util': np.clip(
            np.random.beta(2, 2, n_loans) * 100 + util_shift * 100, 
            0, 100
        ),
        'loan_amount': np.random.lognormal(9.5, 0.5, n_loans),
        'delinq_2yrs': np.random.poisson(0.3, n_loans),
        'open_acc': np.clip(np.random.normal(10, 5, n_loans), 2, 30).astype(int),
        
        # Grade mix shift toward riskier (if severe drift)
        'grade': np.random.choice(
            ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            n_loans,
            p=[0.10, 0.20, 0.25, 0.25, 0.12, 0.05, 0.03] if drift_type == 'severe'
            else [0.15, 0.25, 0.25, 0.20, 0.10, 0.03, 0.02]
        )
    })
    
    return df


def calculate_psi_simple(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Simplified PSI calculation for demonstration.
    
    In production, use the full DriftDetector class from drift_detector.py
    """
    # Remove NaNs
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    
    # Create bins from reference
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    
    # Bin both datasets
    ref_binned = np.digitize(reference, bins, right=True)
    cur_binned = np.digitize(current, bins, right=True)
    
    # Calculate distributions
    ref_counts = np.bincount(ref_binned, minlength=len(bins) + 1)
    cur_counts = np.bincount(cur_binned, minlength=len(bins) + 1)
    
    # Convert to percentages (add epsilon to avoid log(0))
    epsilon = 1e-5
    ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * len(bins))
    cur_pct = (cur_counts + epsilon) / (len(current) + epsilon * len(bins))
    
    # Calculate PSI
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    
    return psi


def run_drift_scenario(
    scenario_name: str,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    monitored_features: list,
    output_dir: Path
):
    """
    Run complete drift detection scenario and generate report.
    """
    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario_name}")
    print("=" * 70)
    
    results = {}
    
    # Calculate PSI for each feature
    for feature in monitored_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
            
        psi = calculate_psi_simple(
            reference_df[feature].values,
            current_df[feature].values
        )
        
        # Classify
        if psi < 0.1:
            status = 'STABLE'
        elif psi < 0.25:
            status = 'MONITOR'
        else:
            status = 'RETRAIN'
            
        results[feature] = {
            'psi': round(psi, 4),
            'status': status
        }
        
        print(f"  {feature:20s}: PSI = {psi:.4f}  [{status}]")
    
    # Summary
    retrain_features = [f for f, r in results.items() if r['status'] == 'RETRAIN']
    monitor_features = [f for f, r in results.items() if r['status'] == 'MONITOR']
    
    print("\n  Summary:")
    print(f"    Retrain triggers: {len(retrain_features)}")
    print(f"    Monitoring: {len(monitor_features)}")
    
    if retrain_features:
        print(f"    Features requiring retraining: {', '.join(retrain_features)}")
        recommendation = 'RETRAIN'
    elif len(monitor_features) >= len(monitored_features) * 0.3:
        print(f"    Multiple features in monitoring zone")
        recommendation = 'INVESTIGATE'
    else:
        recommendation = 'STABLE'
    
    print(f"\n  Recommendation: {recommendation}")
    
    # Save report
    report = {
        'scenario': scenario_name,
        'timestamp': datetime.now().isoformat(),
        'reference_size': len(reference_df),
        'current_size': len(current_df),
        'monitored_features': monitored_features,
        'results': results,
        'retrain_features': retrain_features,
        'monitor_features': monitor_features,
        'recommendation': recommendation
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"drift_report_{scenario_name.replace(' ', '_').lower()}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\n  Report saved: {report_file}")
    
    return report


def main():
    """
    Run all drift detection scenarios for Week 7 testing.
    """
    print("=" * 70)
    print("CreditLens Week 7: Drift Detection Testing")
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    output_dir = Path('./drift_test_output')
    
    # Generate datasets
    print("\nGenerating datasets...")
    reference_df = create_reference_portfolio(10000)
    current_stable = create_current_portfolio_stable(1000)
    current_slight = create_current_portfolio_drift(1000, 'slight')
    current_moderate = create_current_portfolio_drift(1000, 'moderate')
    current_severe = create_current_portfolio_drift(1000, 'severe')
    print(f"  Reference: {len(reference_df):,} loans")
    print(f"  Current portfolios: {len(current_stable):,} loans each")
    
    # Features to monitor (credit industry standard)
    monitored_features = [
        'dti',
        'annual_income',
        'revol_util',
        'loan_amount',
        'delinq_2yrs',
        'open_acc'
    ]
    
    # Run scenarios
    scenarios = [
        ("Stable - No Drift", current_stable),
        ("Slight Economic Shift", current_slight),
        ("Moderate Population Change", current_moderate),
        ("Severe Distribution Shift", current_severe)
    ]
    
    all_reports = []
    
    for scenario_name, current_df in scenarios:
        report = run_drift_scenario(
            scenario_name,
            reference_df,
            current_df,
            monitored_features,
            output_dir
        )
        all_reports.append(report)
    
    # Summary of all scenarios
    print("\n" + "=" * 70)
    print("TESTING SUMMARY")
    print("=" * 70)
    
    for report in all_reports:
        print(f"\n{report['scenario']}:")
        print(f"  Recommendation: {report['recommendation']}")
        print(f"  Retrain triggers: {len(report['retrain_features'])}")
        if report['retrain_features']:
            print(f"    Features: {', '.join(report['retrain_features'])}")
    
    print("\n" + "=" * 70)
    print(f"All reports saved to: {output_dir}")
    print("=" * 70)
    
    # Demonstrate retraining trigger logic
    print("\n" + "=" * 70)
    print("RETRAINING TRIGGER EVALUATION")
    print("=" * 70)
    
    # Get worst-case scenario (severe drift)
    severe_report = all_reports[-1]
    
    print(f"\nEvaluating: {severe_report['scenario']}")
    print(f"  Retrain features: {severe_report['retrain_features']}")
    
    # Simulated retraining decision
    if severe_report['recommendation'] == 'RETRAIN':
        print("\n  Decision: TRIGGER RETRAINING")
        print(f"  Reason: {len(severe_report['retrain_features'])} features exceed PSI threshold")
        print(f"  Severity: HIGH")
        
        # Create task file
        task_file = output_dir / 'retrain_tasks' / f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        
        task_data = {
            'created_at': datetime.now().isoformat(),
            'severity': 'HIGH',
            'reason': f"{len(severe_report['retrain_features'])} features exceed PSI threshold of 0.25",
            'triggered_features': severe_report['retrain_features'],
            'drift_report': severe_report
        }
        
        with open(task_file, 'w') as f:
            json.dump(task_data, f, indent=2)
            
        print(f"\n  Retraining task created: {task_file}")
    
    print("\n" + "=" * 70)
    print("Week 7 Testing Complete")
    print("=" * 70)
    print("\nDeliverables:")
    print("  ✓ PSI calculation implementation")
    print("  ✓ Drift detection on 6 credit features")
    print("  ✓ Four test scenarios (stable, slight, moderate, severe)")
    print("  ✓ Retraining trigger logic")
    print("  ✓ Automated reporting")
    print("\nNext: Week 8 - Agentic insight reporter")
    print("=" * 70)


if __name__ == '__main__':
    main()
