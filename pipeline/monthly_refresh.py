"""
Monthly refresh pipeline for CreditLens.

Orchestrates:
1. Feature store refresh from DuckDB
2. Drift detection on monitored features
3. Retraining decision
4. Portfolio scoring and reporting
5. Logging and alerting

This runs on the first of each month (or on-demand for testing).
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import json
import sys

# Import our modules (in actual project, these would be proper imports)
# from pipeline.drift_detector import DriftDetector, CREDIT_MONITORED_FEATURES
# from pipeline.retrain_trigger import RetrainingTrigger


class MonthlyRefreshPipeline:
    """
    Orchestrate monthly refresh cycle for credit risk monitoring.
    """
    
    def __init__(
        self,
        db_path: str,
        output_dir: Path,
        reference_data_path: Optional[str] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            db_path: Path to DuckDB database
            output_dir: Directory for outputs (drift reports, tasks)
            reference_data_path: Path to reference dataset (training data)
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.reference_data_path = reference_data_path
        
        # Create output directories
        (self.output_dir / 'drift_reports').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'retrain_tasks').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)
        
    def load_current_portfolio(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Load recent portfolio data from DuckDB.
        
        In production, this queries loans active in the past N days.
        For testing, can query any recent slice.
        
        Args:
            lookback_days: Days of recent data to analyze
            
        Returns:
            DataFrame with current portfolio features
        """
        print(f"Loading current portfolio (last {lookback_days} days)...")
        
        conn = duckdb.connect(self.db_path)
        
        # Example query - adjust based on actual schema
        query = f"""
        SELECT 
            l.loan_id,
            l.issue_date,
            l.loan_amount,
            l.term,
            l.grade,
            b.dti,
            b.annual_income,
            b.revol_util,
            b.delinq_2yrs,
            b.open_acc,
            b.emp_length,
            o.loan_status
        FROM loans l
        JOIN borrowers b ON l.loan_id = b.loan_id
        LEFT JOIN outcomes o ON l.loan_id = o.loan_id
        WHERE l.issue_date >= current_date - INTERVAL '{lookback_days} days'
        """
        
        try:
            df = conn.execute(query).df()
            print(f"  Loaded {len(df):,} loans")
            return df
        except Exception as e:
            print(f"  Error loading data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame()
        finally:
            conn.close()
    
    def load_reference_data(self) -> pd.DataFrame:
        """
        Load reference dataset (training data) for PSI comparison.
        
        This should be the same data used to train the current champion model.
        """
        print("Loading reference data...")
        
        if self.reference_data_path and Path(self.reference_data_path).exists():
            df = pd.read_parquet(self.reference_data_path)
            print(f"  Loaded {len(df):,} reference records")
            return df
        else:
            print("  Warning: Reference data not found, using historical query")
            
            # Fallback: query training period from database
            conn = duckdb.connect(self.db_path)
            
            # Example: use loans from Q1 2017 as reference
            query = """
            SELECT 
                l.loan_id,
                l.loan_amount,
                b.dti,
                b.annual_income,
                b.revol_util,
                b.delinq_2yrs,
                b.open_acc
            FROM loans l
            JOIN borrowers b ON l.loan_id = b.loan_id
            WHERE l.issue_date BETWEEN '2017-01-01' AND '2017-03-31'
            """
            
            df = conn.execute(query).df()
            conn.close()
            
            print(f"  Loaded {len(df):,} reference records from database")
            return df
    
    def run_drift_detection(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        monitored_features: list
    ) -> Dict:
        """
        Run PSI drift detection on monitored features.
        
        Args:
            reference_df: Training/reference dataset
            current_df: Current production dataset
            monitored_features: List of features to monitor
            
        Returns:
            Drift summary dictionary
        """
        print("\nRunning drift detection...")
        
        # Import here to avoid circular dependencies in demo
        # In production: from pipeline.drift_detector import DriftDetector
        # For now, we'll create a mock result
        
        # detector = DriftDetector(n_bins=10)
        # results = detector.monitor_features(reference_df, current_df, monitored_features)
        # drift_summary = detector.get_drift_summary(results)
        
        # Mock drift summary for demonstration
        drift_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(monitored_features),
            'stable_count': len(monitored_features) - 2,
            'monitor_count': 1,
            'retrain_count': 1,
            'retrain_features': ['dti'],
            'monitor_features': ['annual_income'],
            'recommendation': 'RETRAIN',
            'reason': '1 feature(s) exceed PSI threshold of 0.25',
            'psi_values': {
                'dti': 0.28,
                'annual_income': 0.12,
                'revol_util': 0.05,
                'loan_amount': 0.03,
                'model_predictions': 0.15
            }
        }
        
        # Save drift report
        report_path = (
            self.output_dir / 'drift_reports' / 
            f"drift_report_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(drift_summary, f, indent=2)
            
        print(f"  Drift report saved: {report_path}")
        print(f"  Recommendation: {drift_summary['recommendation']}")
        print(f"  Reason: {drift_summary['reason']}")
        
        return drift_summary
    
    def evaluate_retraining(
        self,
        drift_summary: Dict,
        model_metadata: Dict
    ) -> bool:
        """
        Evaluate if retraining should be triggered.
        
        Args:
            drift_summary: Output from drift detection
            model_metadata: Current model training date and performance
            
        Returns:
            True if retraining triggered, False otherwise
        """
        print("\nEvaluating retraining triggers...")
        
        # Import here for demo
        # from pipeline.retrain_trigger import RetrainingTrigger
        
        # trigger = RetrainingTrigger()
        # decision = trigger.evaluate(
        #     drift_summary=drift_summary,
        #     model_training_date=model_metadata['training_date'],
        #     performance_metrics=model_metadata.get('performance', None)
        # )
        
        # Mock decision for demonstration
        should_retrain = drift_summary['recommendation'] == 'RETRAIN'
        
        if should_retrain:
            # Create retraining task
            task_file = (
                self.output_dir / 'retrain_tasks' / 
                f"retrain_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            task_data = {
                'created_at': datetime.now().isoformat(),
                'severity': 'high',
                'reason': drift_summary['reason'],
                'triggered_features': drift_summary['retrain_features'],
                'drift_summary': drift_summary
            }
            
            with open(task_file, 'w') as f:
                json.dump(task_data, f, indent=2)
                
            print(f"  Retraining task created: {task_file}")
            print(f"  Severity: high")
            
        else:
            print(f"  No retraining needed - monitoring continues")
            
        return should_retrain
    
    def generate_portfolio_summary(
        self,
        current_df: pd.DataFrame
    ) -> Dict:
        """
        Generate monthly portfolio risk summary.
        
        This feeds into the agentic reporter in Week 8.
        """
        print("\nGenerating portfolio summary...")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_loans': len(current_df),
            'total_exposure': float(current_df['loan_amount'].sum()) if 'loan_amount' in current_df else 0,
            'avg_loan_amount': float(current_df['loan_amount'].mean()) if 'loan_amount' in current_df else 0,
            'grade_distribution': {},
            'risk_metrics': {}
        }
        
        # Grade distribution
        if 'grade' in current_df:
            grade_counts = current_df['grade'].value_counts().to_dict()
            summary['grade_distribution'] = {k: int(v) for k, v in grade_counts.items()}
        
        print(f"  Portfolio size: {summary['total_loans']:,} loans")
        print(f"  Total exposure: ${summary['total_exposure']:,.0f}")
        
        # Save summary
        summary_path = (
            self.output_dir / 
            f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"  Summary saved: {summary_path}")
        
        return summary
    
    def run(self, lookback_days: int = 30) -> Dict:
        """
        Execute full monthly refresh pipeline.
        
        Args:
            lookback_days: Days of recent data to analyze
            
        Returns:
            Pipeline execution summary
        """
        print("=" * 60)
        print("CreditLens Monthly Refresh Pipeline")
        print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        execution_summary = {
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        
        try:
            # Step 1: Load data
            current_df = self.load_current_portfolio(lookback_days)
            reference_df = self.load_reference_data()
            
            if current_df.empty or reference_df.empty:
                print("\nError: Could not load required data")
                execution_summary['status'] = 'failed'
                execution_summary['error'] = 'Data loading failed'
                return execution_summary
            
            # Step 2: Drift detection
            monitored_features = ['dti', 'annual_income', 'revol_util', 
                                'loan_amount', 'delinq_2yrs', 'open_acc']
            
            drift_summary = self.run_drift_detection(
                reference_df, 
                current_df, 
                monitored_features
            )
            
            # Step 3: Retraining evaluation
            model_metadata = {
                'training_date': datetime.now() - timedelta(days=60),
                'model_version': 'scorecard_v1',
                'training_gini': 0.64
            }
            
            retrain_triggered = self.evaluate_retraining(
                drift_summary,
                model_metadata
            )
            
            # Step 4: Portfolio summary
            portfolio_summary = self.generate_portfolio_summary(current_df)
            
            # Finalize
            execution_summary.update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'records_processed': len(current_df),
                'drift_detected': drift_summary['recommendation'] != 'STABLE',
                'retrain_triggered': retrain_triggered,
                'portfolio_summary': portfolio_summary
            })
            
            print("\n" + "=" * 60)
            print("Pipeline completed successfully")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            execution_summary['status'] = 'failed'
            execution_summary['error'] = str(e)
            
        # Log execution
        log_path = self.output_dir / 'logs' / 'pipeline_log.jsonl'
        with open(log_path, 'a') as f:
            f.write(json.dumps(execution_summary) + '\n')
            
        return execution_summary


def main():
    """
    Example usage of monthly refresh pipeline.
    
    In production, this would be triggered by a scheduler (cron, Airflow, etc.)
    """
    # Configuration
    db_path = '/home/claude/data/creditlens.duckdb'  # Adjust path
    output_dir = Path('/home/claude/pipeline_output')
    
    # Initialize and run pipeline
    pipeline = MonthlyRefreshPipeline(
        db_path=db_path,
        output_dir=output_dir
    )
    
    result = pipeline.run(lookback_days=30)
    
    print("\n" + "=" * 60)
    print("Execution Summary:")
    print(f"  Status: {result['status']}")
    print(f"  Records processed: {result.get('records_processed', 'N/A')}")
    print(f"  Retrain triggered: {result.get('retrain_triggered', False)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
