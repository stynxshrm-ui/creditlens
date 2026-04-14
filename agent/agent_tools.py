"""
Production tools for CreditLens agentic insight reporter.

6 tools that give the agent access to:
- Drift detection results (PSI reports)
- Model performance metrics
- Portfolio summaries
- Vintage cohort analysis
- Champion/challenger status
- Direct SQL queries on DuckDB

These tools return structured data that the agent synthesizes into
plain-English insights for non-technical stakeholders.
"""

import json
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np


class CreditLensTools:
    """
    Tool suite for agentic credit risk investigation.
    
    All tools return dictionaries for structured analysis.
    """
    
    def __init__(
        self,
        db_path: str,
        drift_reports_dir: Path,
        mlflow_tracking_uri: Optional[str] = None
    ):
        """
        Initialize tool suite.
        
        Args:
            db_path: Path to DuckDB database
            drift_reports_dir: Directory containing drift reports
            mlflow_tracking_uri: Optional MLflow tracking server
        """
        self.db_path = db_path
        self.drift_reports_dir = Path(drift_reports_dir)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
    def get_psi_report(self, feature_name: str) -> Dict[str, Any]:
        """
        Get PSI drift report for a specific feature.
        
        Args:
            feature_name: Name of the feature (e.g., 'dti', 'annual_income')
            
        Returns:
            Dict with PSI value, status, bin contributions, and context
            
        Example:
            {
                "feature": "dti",
                "psi_value": 0.28,
                "status": "RETRAIN",
                "threshold_breached": true,
                "bin_contributions": {"bin_3": 0.08, "bin_7": 0.12},
                "reference_period": "Q1_2017",
                "current_period": "March_2026",
                "interpretation": "Significant shift in DTI distribution..."
            }
        """
        # Find most recent drift report
        reports = sorted(self.drift_reports_dir.glob('drift_report_*.json'))
        
        if not reports:
            return {
                'error': 'No drift reports found',
                'feature': feature_name
            }
            
        latest_report = reports[-1]
        
        with open(latest_report) as f:
            report = json.load(f)
            
        # Extract feature-specific results
        results = report.get('results', {})
        
        if feature_name not in results:
            return {
                'error': f'Feature {feature_name} not found in report',
                'available_features': list(results.keys())
            }
            
        feature_result = results[feature_name]
        
        # Add interpretation
        psi = feature_result['psi']
        status = feature_result['status']
        
        if status == 'RETRAIN':
            interpretation = (
                f"{feature_name} shows significant distribution shift (PSI {psi:.3f}). "
                f"Current population differs materially from training baseline. "
                f"Model recalibration recommended."
            )
        elif status == 'MONITOR':
            interpretation = (
                f"{feature_name} shows moderate drift (PSI {psi:.3f}). "
                f"Within monitoring threshold but warrants investigation. "
                f"Monitor for sustained trend."
            )
        else:
            interpretation = (
                f"{feature_name} distribution is stable (PSI {psi:.3f}). "
                f"No material change from training baseline."
            )
            
        return {
            'feature': feature_name,
            'psi_value': psi,
            'status': status,
            'threshold_breached': psi >= 0.25,
            'monitoring_threshold': psi >= 0.10,
            'reference_period': report.get('reference_period', 'training'),
            'current_period': report.get('current_period', 'current'),
            'interpretation': interpretation,
            'report_timestamp': report.get('timestamp', 'unknown')
        }
    
    def get_model_performance(self, period: str = 'current') -> Dict[str, Any]:
        """
        Get model performance metrics for a time period.
        
        Args:
            period: 'current', 'last_month', 'training', or specific date
            
        Returns:
            Dict with Gini, KS, calibration metrics
            
        Example:
            {
                "period": "March_2026",
                "gini": 0.64,
                "ks_statistic": 0.38,
                "auc": 0.82,
                "calibration_slope": 0.92,
                "training_gini": 0.71,
                "degradation": 0.07,
                "interpretation": "Model discrimination declined 7pp..."
            }
        """
        # In production, this would query MLflow or a metrics database
        # For demo, return realistic values
        
        # Simulate performance metrics
        if period == 'training':
            metrics = {
                'gini': 0.71,
                'ks_statistic': 0.42,
                'auc': 0.855,
                'calibration_slope': 1.00
            }
            degradation = 0.0
            
        else:  # current or recent
            metrics = {
                'gini': 0.64,
                'ks_statistic': 0.38,
                'auc': 0.82,
                'calibration_slope': 0.92
            }
            degradation = 0.71 - 0.64  # vs training
            
        # Interpretation
        if degradation > 0.05:
            interpretation = (
                f"Model discrimination declined {degradation:.2f} Gini points "
                f"from training baseline. Performance degradation suggests "
                f"population shift or feature relationships changing. "
                f"Retraining recommended."
            )
        elif degradation > 0.02:
            interpretation = (
                f"Slight performance decline ({degradation:.2f} Gini points). "
                f"Monitor for sustained trend. May be seasonal variation."
            )
        else:
            interpretation = (
                f"Model performance stable. Gini {metrics['gini']:.3f} "
                f"aligns with training expectations."
            )
            
        return {
            'period': period,
            **metrics,
            'training_gini': 0.71,
            'degradation': degradation,
            'degradation_pct': (degradation / 0.71) * 100 if degradation > 0 else 0,
            'interpretation': interpretation
        }
    
    def get_portfolio_summary(self, segment: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current portfolio risk summary.
        
        Args:
            segment: Optional filter ('grade_A', 'grade_B', 'high_risk', etc.)
            
        Returns:
            Dict with portfolio size, exposure, PD distribution, expected loss
            
        Example:
            {
                "total_loans": 12453,
                "total_exposure": 245000000,
                "avg_loan_amount": 19678,
                "avg_pd": 0.118,
                "expected_loss": 28910000,
                "grade_distribution": {"A": 1845, "B": 3120, ...},
                "high_risk_pct": 0.15
            }
        """
        conn = duckdb.connect(self.db_path)
        
        # Build query based on segment filter
        where_clause = ""
        if segment:
            if segment.startswith('grade_'):
                grade = segment.split('_')[1]
                where_clause = f"WHERE l.grade = '{grade}'"
            elif segment == 'high_risk':
                where_clause = "WHERE l.grade IN ('E', 'F', 'G')"
            elif segment == 'low_risk':
                where_clause = "WHERE l.grade IN ('A', 'B')"
                
        query = f"""
        SELECT 
            COUNT(*) as total_loans,
            SUM(l.loan_amount) as total_exposure,
            AVG(l.loan_amount) as avg_loan_amount,
            l.grade,
            COUNT(*) as count_by_grade
        FROM loans l
        {where_clause}
        GROUP BY l.grade
        """
        
        try:
            df = conn.execute(query).df()
            
            # Aggregate results
            total_loans = int(df['count_by_grade'].sum())
            total_exposure = float(df['total_exposure'].sum()) if 'total_exposure' in df else 0
            avg_loan_amount = total_exposure / total_loans if total_loans > 0 else 0
            
            # Grade distribution
            grade_dist = df.groupby('grade')['count_by_grade'].sum().to_dict()
            grade_dist = {k: int(v) for k, v in grade_dist.items()}
            
            # Simulate PD and EL (in production, join with model predictions)
            # Assume average PD based on grade mix
            avg_pd = 0.118  # Realistic for mixed portfolio
            expected_loss = total_exposure * avg_pd * 0.60  # LGD = 60%
            
            high_risk_pct = sum(
                v for k, v in grade_dist.items() 
                if k in ['E', 'F', 'G']
            ) / total_loans if total_loans > 0 else 0
            
            conn.close()
            
            return {
                'segment': segment if segment else 'all',
                'total_loans': total_loans,
                'total_exposure': total_exposure,
                'avg_loan_amount': avg_loan_amount,
                'avg_pd': avg_pd,
                'expected_loss': expected_loss,
                'grade_distribution': grade_dist,
                'high_risk_pct': high_risk_pct,
                'interpretation': (
                    f"Portfolio of {total_loans:,} loans with ${total_exposure:,.0f} exposure. "
                    f"Expected loss ${expected_loss:,.0f} ({(expected_loss/total_exposure)*100:.2f}% of exposure). "
                    f"High-risk grades (E/F/G) represent {high_risk_pct*100:.1f}% of portfolio."
                )
            }
            
        except Exception as e:
            conn.close()
            return {
                'error': str(e),
                'segment': segment
            }
    
    def get_vintage_cohort(self, issue_quarter: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific origination cohort.
        
        Args:
            issue_quarter: Quarter in format 'Q1_2017', 'Q2_2017', etc.
            
        Returns:
            Dict with cohort size, default rate, performance vs expectation
            
        Example:
            {
                "cohort": "Q3_2017",
                "total_loans": 2840,
                "default_count": 402,
                "default_rate": 0.142,
                "expected_default_rate": 0.118,
                "performance_gap": 0.024,
                "interpretation": "Q3 2017 cohort defaulting 2.4pp above expectation..."
            }
        """
        conn = duckdb.connect(self.db_path)
        
        # Parse quarter
        try:
            quarter, year = issue_quarter.split('_')
            quarter_num = int(quarter[1])
            
            # Calculate date range for quarter
            start_month = (quarter_num - 1) * 3 + 1
            end_month = start_month + 2
            
            query = f"""
            SELECT 
                COUNT(*) as total_loans,
                SUM(CASE WHEN o.default_flag = 1 THEN 1 ELSE 0 END) as default_count,
                AVG(l.loan_amount) as avg_loan_amount,
                l.grade
            FROM loans l
            LEFT JOIN outcomes o ON l.loan_id = o.loan_id
            WHERE 
                EXTRACT(YEAR FROM l.issue_date) = {year}
                AND EXTRACT(MONTH FROM l.issue_date) BETWEEN {start_month} AND {end_month}
            GROUP BY l.grade
            """
            
            df = conn.execute(query).df()
            conn.close()
            
            total_loans = int(df['total_loans'].sum())
            default_count = int(df['default_count'].sum())
            default_rate = default_count / total_loans if total_loans > 0 else 0
            
            # Expected default rate (portfolio average)
            expected_default_rate = 0.118
            performance_gap = default_rate - expected_default_rate
            
            # Interpretation
            if abs(performance_gap) > 0.02:
                direction = "above" if performance_gap > 0 else "below"
                interpretation = (
                    f"{issue_quarter} cohort performing {abs(performance_gap)*100:.1f}pp "
                    f"{direction} expectation. Default rate {default_rate*100:.1f}% vs "
                    f"expected {expected_default_rate*100:.1f}%. "
                    f"{'Indicates underwriting quality issues or macro deterioration.' if performance_gap > 0 else 'Strong vintage performance.'}"
                )
            else:
                interpretation = (
                    f"{issue_quarter} cohort performing as expected. "
                    f"Default rate {default_rate*100:.1f}% aligns with portfolio baseline."
                )
                
            return {
                'cohort': issue_quarter,
                'total_loans': total_loans,
                'default_count': default_count,
                'default_rate': default_rate,
                'expected_default_rate': expected_default_rate,
                'performance_gap': performance_gap,
                'interpretation': interpretation
            }
            
        except Exception as e:
            conn.close()
            return {
                'error': str(e),
                'cohort': issue_quarter
            }
    
    def get_champion_challenger_status(self) -> Dict[str, Any]:
        """
        Get current champion/challenger deployment status.
        
        Returns:
            Dict with champion model, challenger status, shadow metrics
            
        Example:
            {
                "champion": {
                    "model_type": "scorecard",
                    "version": "v1.2",
                    "training_date": "2025-12-01",
                    "gini": 0.64
                },
                "challenger": {
                    "model_type": "xgboost",
                    "version": "v2.0",
                    "shadow_days": 18,
                    "shadow_gini": 0.71,
                    "promotion_eligible": true
                },
                "recommendation": "Promote challenger - Gini improvement 7pp"
            }
        """
        # In production, query MLflow Model Registry
        # For demo, return realistic scenario
        
        champion = {
            'model_type': 'scorecard',
            'version': 'v1.2',
            'training_date': '2025-12-01',
            'gini': 0.64,
            'status': 'production'
        }
        
        challenger = {
            'model_type': 'xgboost',
            'version': 'v2.0',
            'training_date': '2026-03-15',
            'shadow_start_date': '2026-03-18',
            'shadow_days': 18,
            'shadow_gini': 0.71,
            'shadow_ks': 0.42,
            'status': 'shadow_deployment',
            'promotion_eligible': True
        }
        
        gini_improvement = challenger['shadow_gini'] - champion['gini']
        
        if gini_improvement > 0.05:
            recommendation = (
                f"Recommend promoting challenger. Gini improvement "
                f"{gini_improvement:.2f} exceeds 5pp threshold. "
                f"18 days of shadow deployment shows stable performance."
            )
        elif challenger['shadow_days'] < 30:
            recommendation = (
                f"Continue shadow deployment. Gini improvement "
                f"{gini_improvement:.2f} is promising but shadow period "
                f"incomplete (18/30 days). Review after 30-day period."
            )
        else:
            recommendation = "No promotion recommended at this time."
            
        return {
            'champion': champion,
            'challenger': challenger,
            'gini_improvement': gini_improvement,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
    
    def query_portfolio(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute read-only SQL query on portfolio database.
        
        Args:
            sql: SQL query string (SELECT only, validated)
            
        Returns:
            List of dicts representing query results
            
        Example:
            query_portfolio('''
                SELECT grade, AVG(dti) as avg_dti, COUNT(*) as count
                FROM loans l
                JOIN borrowers b ON l.loan_id = b.loan_id
                GROUP BY grade
                ORDER BY grade
            ''')
        """
        # Validate query is SELECT only (safety)
        sql_upper = sql.strip().upper()
        
        if not sql_upper.startswith('SELECT'):
            return [{'error': 'Only SELECT queries allowed'}]
            
        # Block dangerous operations
        forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        if any(word in sql_upper for word in forbidden):
            return [{'error': 'Query contains forbidden operations'}]
            
        conn = duckdb.connect(self.db_path, read_only=True)
        
        try:
            df = conn.execute(sql).df()
            conn.close()
            
            # Convert to list of dicts
            results = df.to_dict('records')
            
            # Limit results to prevent overwhelming context
            if len(results) > 100:
                results = results[:100]
                results.append({'warning': 'Results truncated to 100 rows'})
                
            return results
            
        except Exception as e:
            conn.close()
            return [{'error': str(e)}]


# Tool definitions for Anthropic API
TOOL_DEFINITIONS = [
    {
        "name": "get_psi_report",
        "description": "Get PSI drift report for a specific feature. Returns PSI value, status (STABLE/MONITOR/RETRAIN), and interpretation. Use this when investigating drift or understanding which features have shifted.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_name": {
                    "type": "string",
                    "description": "Name of the feature to check (e.g., 'dti', 'annual_income', 'revol_util', 'loan_amount')"
                }
            },
            "required": ["feature_name"]
        }
    },
    {
        "name": "get_model_performance",
        "description": "Get model performance metrics (Gini, KS, AUC, calibration) for a time period. Use to assess if model quality is degrading and by how much.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Time period: 'current', 'last_month', 'training'",
                    "default": "current"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_portfolio_summary",
        "description": "Get portfolio size, exposure, risk distribution, and expected loss. Can filter by segment (e.g., 'grade_C', 'high_risk'). Use for overall portfolio health assessment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "string",
                    "description": "Optional filter: 'grade_A', 'grade_B', ..., 'high_risk', 'low_risk', or null for all"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_vintage_cohort",
        "description": "Get default rate and performance for a specific origination quarter (e.g., 'Q3_2017'). Use to identify which cohorts are underperforming.",
        "input_schema": {
            "type": "object",
            "properties": {
                "issue_quarter": {
                    "type": "string",
                    "description": "Quarter in format 'Q1_2017', 'Q2_2017', etc."
                }
            },
            "required": ["issue_quarter"]
        }
    },
    {
        "name": "get_champion_challenger_status",
        "description": "Get current champion model and challenger shadow deployment status. Use to understand model promotion readiness and performance gaps.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "query_portfolio",
        "description": "Execute read-only SQL query on the portfolio database. Use for custom analysis not covered by other tools. Returns up to 100 rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT query. Can join loans, borrowers, outcomes tables."
                }
            },
            "required": ["sql"]
        }
    }
]


def demonstrate_tools():
    """
    Demonstrate tool usage with mock data.
    """
    print("=" * 70)
    print("CreditLens Agent Tools Demonstration")
    print("=" * 70)
    
    # Initialize tools (would use real paths in production)
    tools = CreditLensTools(
        db_path='/tmp/demo.duckdb',  # Mock
        drift_reports_dir=Path('/home/claude/drift_test_output')
    )
    
    print("\n1. PSI Report for DTI:")
    print("-" * 70)
    result = tools.get_psi_report('dti')
    print(json.dumps(result, indent=2))
    
    print("\n2. Model Performance:")
    print("-" * 70)
    result = tools.get_model_performance('current')
    print(json.dumps(result, indent=2))
    
    print("\n3. Champion/Challenger Status:")
    print("-" * 70)
    result = tools.get_champion_challenger_status()
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 70)
    print("Tools ready for agent integration")
    print("=" * 70)


if __name__ == '__main__':
    demonstrate_tools()
