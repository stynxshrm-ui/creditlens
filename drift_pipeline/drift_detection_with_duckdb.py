"""
Drift Detection with DuckDB Integration

This demonstrates the ACTUAL data flow in production:
1. Query DuckDB for reference data (training period)
2. Query DuckDB for current data (recent originations)
3. Run PSI drift detection
4. Generate reports

Uses synthetic data while leveraging real DuckDB queries.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json


def create_sample_credit_database():
    """
    Create a sample DuckDB database with credit data for testing.
    
    In production, this would already exist from Week 1 ingestion.
    """
    print("Creating sample credit database...")
    
    db_path = f'../data/drift_{datetime.now().strftime("%Y%m%d_%H%M%S")}.duckdb'
    conn = duckdb.connect(db_path)
    
    # Create loans table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loans (
            loan_id VARCHAR,
            issue_date DATE,
            loan_amount DECIMAL(10,2),
            term INTEGER,
            grade VARCHAR,
            sub_grade VARCHAR,
            purpose VARCHAR
        )
    """)
    
    # Create borrowers table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS borrowers (
            loan_id VARCHAR,
            dti DECIMAL(5,2),
            annual_income DECIMAL(12,2),
            revol_util DECIMAL(5,2),
            delinq_2yrs INTEGER,
            open_acc INTEGER,
            emp_length VARCHAR
        )
    """)
    
    # Create outcomes table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            loan_id VARCHAR,
            loan_status VARCHAR,
            default_flag INTEGER
        )
    """)
    
    # Generate sample data - Q1 2017 (reference period)
    np.random.seed(42)
    n_ref = 5000
    
    ref_loans = []
    ref_borrowers = []
    ref_outcomes = []
    
    for i in range(n_ref):
        loan_id = f'LC_REF_{i:06d}'
        
        # Reference period: Q1 2017 (Jan-Mar 2017)
        days_offset = np.random.randint(0, 90)
        issue_date = datetime(2017, 1, 1) + timedelta(days=days_offset)
        
        ref_loans.append({
            'loan_id': loan_id,
            'issue_date': issue_date.strftime('%Y-%m-%d'),
            'loan_amount': float(np.random.lognormal(9.5, 0.5)),
            'term': int(np.random.choice([36, 60], p=[0.7, 0.3])),
            'grade': np.random.choice(['A', 'B', 'C', 'D', 'E'], p=[0.2, 0.3, 0.25, 0.15, 0.1]),
            'sub_grade': 'A1',  # Simplified
            'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement'])
        })
        
        ref_borrowers.append({
            'loan_id': loan_id,
            'dti': float(np.clip(np.random.normal(18, 8), 0, 45)),
            'annual_income': float(np.random.lognormal(10.8, 0.6)),
            'revol_util': float(np.clip(np.random.beta(2, 2) * 100, 0, 100)),
            'delinq_2yrs': int(np.random.poisson(0.3)),
            'open_acc': int(np.clip(np.random.normal(10, 5), 2, 30)),
            'emp_length': np.random.choice(['< 1 year', '1-3 years', '3-5 years', '5-10 years', '10+ years'])
        })
        
        ref_outcomes.append({
            'loan_id': loan_id,
            'loan_status': np.random.choice(['Fully Paid', 'Charged Off'], p=[0.88, 0.12]),
            'default_flag': int(np.random.random() < 0.12)
        })
    
    # Insert reference data
   
    conn.executemany("INSERT INTO loans VALUES (?, ?, ?, ?, ?, ?, ?)", 
                     [(r['loan_id'], r['issue_date'], r['loan_amount'], r['term'], 
                       r['grade'], r['sub_grade'], r['purpose']) for r in ref_loans])
    
    conn.executemany("INSERT INTO borrowers VALUES (?, ?, ?, ?, ?, ?, ?)",
                     [(r['loan_id'], r['dti'], r['annual_income'], r['revol_util'],
                       r['delinq_2yrs'], r['open_acc'], r['emp_length']) for r in ref_borrowers])
    
    conn.executemany("INSERT INTO outcomes VALUES (?, ?, ?)",
                     [(r['loan_id'], r['loan_status'], r['default_flag']) for r in ref_outcomes])
    
    # Generate current data - March 2026 (with drift)
    np.random.seed(123)
    n_curr = 1000
    
    curr_loans = []
    curr_borrowers = []
    curr_outcomes = []
    
    for i in range(n_curr):
        loan_id = f'LC_CURR_{i:06d}'
        
        # Current period: March 2026
        days_offset = np.random.randint(0, 30)
        issue_date = datetime(2026, 3, 1) + timedelta(days=days_offset)
        
        curr_loans.append({
            'loan_id': loan_id,
            'issue_date': issue_date.strftime('%Y-%m-%d'),
            'loan_amount': float(np.random.lognormal(9.5, 0.5)),
            'term': int(np.random.choice([36, 60], p=[0.7, 0.3])),
            'grade': np.random.choice(['A', 'B', 'C', 'D', 'E'], p=[0.15, 0.25, 0.25, 0.20, 0.15]),  # Shift toward riskier
            'sub_grade': 'A1',
            'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement'])
        })
        
        # Drift: Higher DTI, higher utilization
        curr_borrowers.append({
            'loan_id': loan_id,
            'dti': float(np.clip(np.random.normal(18 + 2.5, 8), 0, 45)),  # +2.5pp drift
            'annual_income': float(np.random.lognormal(10.8, 0.6) * 1.15),  # +15% income
            'revol_util': float(np.clip(np.random.beta(2, 2) * 100 + 8, 0, 100)),  # +8pp utilization
            'delinq_2yrs': int(np.random.poisson(0.3)),
            'open_acc': int(np.clip(np.random.normal(10, 5), 2, 30)),
            'emp_length': np.random.choice(['< 1 year', '1-3 years', '3-5 years', '5-10 years', '10+ years'])
        })
        
        curr_outcomes.append({
            'loan_id': loan_id,
            'loan_status': 'Current',
            'default_flag': 0  # Too recent to know
        })
    
    # Insert current data
    conn.executemany("INSERT INTO loans VALUES (?, ?, ?, ?, ?, ?, ?)", 
                     [(r['loan_id'], r['issue_date'], r['loan_amount'], r['term'], 
                       r['grade'], r['sub_grade'], r['purpose']) for r in curr_loans])
    
    conn.executemany("INSERT INTO borrowers VALUES (?, ?, ?, ?, ?, ?, ?)",
                     [(r['loan_id'], r['dti'], r['annual_income'], r['revol_util'],
                       r['delinq_2yrs'], r['open_acc'], r['emp_length']) for r in curr_borrowers])
    
    conn.executemany("INSERT INTO outcomes VALUES (?, ?, ?)",
                     [(r['loan_id'], r['loan_status'], r['default_flag']) for r in curr_outcomes])
    
    conn.close()
    
    print(f"✓ Created database: {db_path}")
    print(f"  Reference data: {n_ref} loans (Q1 2017)")
    print(f"  Current data: {n_curr} loans (March 2026)")
    
    return db_path


def query_reference_data(db_path: str) -> pd.DataFrame:
    """
    Query DuckDB for reference/training data.
    
    This would be the data used to train your current champion model.
    """
    print("\nQuerying reference data (Q1 2017)...")
    
    conn = duckdb.connect(db_path, read_only=True)
    
    query = """
    SELECT 
        l.loan_id,
        l.issue_date,
        l.loan_amount,
        l.grade,
        b.dti,
        b.annual_income,
        b.revol_util,
        b.delinq_2yrs,
        b.open_acc
    FROM loans l
    JOIN borrowers b ON l.loan_id = b.loan_id
    WHERE l.issue_date >= '2017-01-01' 
      AND l.issue_date < '2017-04-01'
    """
    
    df = conn.execute(query).df()
    conn.close()
    
    print(f"✓ Loaded {len(df):,} reference loans")
    print(f"  Date range: {df['issue_date'].min()} to {df['issue_date'].max()}")
    print(f"  Features: {list(df.columns)}")
    
    return df


def query_current_data(db_path: str) -> pd.DataFrame:
    """
    Query DuckDB for current production data.
    
    This would be recent originations you want to monitor for drift.
    """
    print("\nQuerying current data (March 2026)...")
    
    conn = duckdb.connect(db_path, read_only=True)
    
    query = """
    SELECT 
        l.loan_id,
        l.issue_date,
        l.loan_amount,
        l.grade,
        b.dti,
        b.annual_income,
        b.revol_util,
        b.delinq_2yrs,
        b.open_acc
    FROM loans l
    JOIN borrowers b ON l.loan_id = b.loan_id
    WHERE l.issue_date >= '2026-03-01'
    """
    
    df = conn.execute(query).df()
    conn.close()
    
    print(f"✓ Loaded {len(df):,} current loans")
    print(f"  Date range: {df['issue_date'].min()} to {df['issue_date'].max()}")
    
    return df


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate PSI between reference and current distributions.
    """
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
    
    # Convert to percentages
    epsilon = 1e-5
    ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * len(bins))
    cur_pct = (cur_counts + epsilon) / (len(current) + epsilon * len(bins))
    
    # Calculate PSI
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    
    return psi


def run_drift_detection_with_duckdb(db_path: str):
    """
    Complete drift detection workflow using DuckDB.
    
    This is the production Week 7 workflow:
    1. Query reference data from DuckDB
    2. Query current data from DuckDB
    3. Calculate PSI on monitored features
    4. Generate drift report
    """
    print("=" * 70)
    print("Drift Detection with DuckDB Integration")
    print("=" * 70)
    
    # Step 1: Query reference data
    reference_df = query_reference_data(db_path)
    
    # Step 2: Query current data
    current_df = query_current_data(db_path)
    
    # Step 3: Calculate PSI on monitored features
    print("\n" + "=" * 70)
    print("PSI Calculation Results")
    print("=" * 70)
    
    monitored_features = ['dti', 'annual_income', 'revol_util', 'loan_amount', 'delinq_2yrs', 'open_acc']
    
    results = {}
    
    for feature in monitored_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
            
        psi = calculate_psi(
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
        
        # Print with coding
        status_marker = 'S' if status == 'STABLE' else 'M' if status == 'MONITOR' else 'X'
        print(f"{status_marker} {feature:20s}: PSI = {psi:.4f}  [{status}]")
    
    # Step 4: Generate summary
    print("\n" + "=" * 70)
    print("Drift Summary")
    print("=" * 70)
    
    retrain_features = [f for f, r in results.items() if r['status'] == 'RETRAIN']
    monitor_features = [f for f, r in results.items() if r['status'] == 'MONITOR']
    stable_features = [f for f, r in results.items() if r['status'] == 'STABLE']
    
    print(f"Total features monitored: {len(results)}")
    print(f" Stable: {len(stable_features)}")
    print(f" Monitor: {len(monitor_features)}")
    print(f" Retrain: {len(retrain_features)}")
    
    if retrain_features:
        print(f"\nFeatures requiring retraining:")
        for f in retrain_features:
            print(f"  • {f} (PSI = {results[f]['psi']:.4f})")
        recommendation = 'RETRAIN'
    elif len(monitor_features) >= len(monitored_features) * 0.3:
        print(f"\n{len(monitor_features)} features in monitoring zone")
        recommendation = 'INVESTIGATE'
    else:
        recommendation = 'STABLE'
    
    print(f"\nRecommendation: {recommendation}")
    
    # Step 5: Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'reference_period': 'Q1_2017',
        'current_period': 'March_2026',
        'reference_count': len(reference_df),
        'current_count': len(current_df),
        'results': results,
        'retrain_features': retrain_features,
        'monitor_features': monitor_features,
        'recommendation': recommendation
    }
    
    output_dir = Path('../outputs/drift_reports')
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    
    # Show feature statistics for context
    print("\n" + "=" * 70)
    print("Feature Statistics Comparison")
    print("=" * 70)
    
    for feature in retrain_features[:3]:  # Show top 3 drifted features
        ref_mean = reference_df[feature].mean()
        cur_mean = current_df[feature].mean()
        change = cur_mean - ref_mean
        pct_change = (change / ref_mean) * 100 if ref_mean != 0 else 0
        
        print(f"\n{feature}:")
        print(f"  Reference mean: {ref_mean:.2f}")
        print(f"  Current mean:   {cur_mean:.2f}")
        print(f"  Change:         {change:+.2f} ({pct_change:+.1f}%)")
        print(f"  PSI:            {results[feature]['psi']:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ DuckDB Integration Complete")
    print("=" * 70)
    
    return report


def main():
    """
    Complete demo: Create sample database → Query → Drift detection
    """
    # Create sample database
    db_path = create_sample_credit_database()
    
    # Run drift detection
    report = run_drift_detection_with_duckdb(db_path)
    
    print("\n" + "=" * 70)
    print("Integration with Agentic Reporter")
    print("=" * 70)
    print("\nThis drift report would now feed into the agentic reporter:")
    print("  1. Agent reads drift_report_*.json")
    print("  2. Sees retrain_features:", report['retrain_features'])
    print("  3. Investigates using tools (get_psi_report, get_vintage_cohort, etc.)")
    print("  4. Generates plain-English monthly report")
    print("\nSee: integrated_pipeline.py for full workflow")
    print("=" * 70)


if __name__ == '__main__':
    main()
