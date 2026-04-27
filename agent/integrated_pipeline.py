"""
Week 7 + Week 8 Integration Script

Demonstrates how the drift detection pipeline (Week 7) triggers
the agentic reporter (Week 8) to investigate and generate insights.

This is the complete monthly workflow:
1. Run drift detection (Week 7)
2. If drift detected → trigger agent investigation (Week 8)
3. Generate monthly report
4. Save for stakeholder review

In production, this runs as a scheduled job (cron, Airflow, etc.)
"""

from pathlib import Path
from datetime import datetime
import json
import os


def integrated_monthly_pipeline():
    """
    Complete monthly workflow: drift detection → agentic investigation.
    """
    print("=" * 70)
    print("CreditLens Integrated Monthly Pipeline")
    print("Drift Detection -> Agentic Reporter")
    print("=" * 70)
    print(f"\nExecution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    
    # ========================================
    # WEEK 7: Drift Detection
    # ========================================
    
    print("\nRunning drift detection pipeline...")
    print("-" * 70)
    
    # In production, import from actual modules:
    # from pipeline.monthly_refresh import MonthlyRefreshPipeline
    # from pipeline.drift_detector import DriftDetector
    
    # For demo, simulate drift detection results
    drift_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_features': 6,
        'stable_count': 3,
        'monitor_count': 1,
        'retrain_count': 2,
        'retrain_features': ['dti', 'revol_util'],
        'monitor_features': ['annual_income'],
        'recommendation': 'RETRAIN',
        'reason': '2 feature(s) exceed PSI threshold of 0.25',
        'psi_values': {
            'dti': 0.41,
            'annual_income': 0.12,
            'revol_util': 1.93,
            'loan_amount': 0.03,
            'delinq_2yrs': 0.01,
            'open_acc': 0.04,
            'model_predictions': 0.15
        }
    }
    
    print(f"  Drift detection completed")
    print(f"  Recommendation: {drift_summary['recommendation']}")
    print(f"  Retrain features: {drift_summary['retrain_features']}")
    print(f"  PSI - DTI: {drift_summary['psi_values']['dti']:.3f}")
    print(f"  PSI - Revol Util: {drift_summary['psi_values']['revol_util']:.3f}")
    
    # Save drift report
    drift_report_dir = Path('outputs/drift_reports')
    drift_report_dir.mkdir(parents=True, exist_ok=True)
    
    drift_report_path = drift_report_dir / f"drift_report_{datetime.now().strftime('%Y%m%d')}.json"
    
    with open(drift_report_path, 'w') as f:
        json.dump(drift_summary, f, indent=2)
    
    print(f"\n  Drift report saved: {drift_report_path}")
    
    # ========================================
    # DECISION POINT: Should we run agent?
    # ========================================
    
    should_investigate = drift_summary['recommendation'] in ['RETRAIN', 'INVESTIGATE']
    
    print("\n" + "=" * 70)
    print("[DECISION] Should agentic reporter investigate?")
    print("=" * 70)
    
    if should_investigate:
        print(f"  YES - {drift_summary['reason']}")
        print("  Triggering agentic investigation...")
    else:
        print("  NO - Portfolio is stable, no investigation needed")
        print("  Monthly pipeline complete.")
        return
    
    # ========================================
    # WEEK 8: Agentic Reporter
    # ========================================
    
    print("\nInitializing agentic reporter...")
    print("-" * 70)
    
    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("\n  ⚠️  ANTHROPIC_API_KEY not set")
        print("\n  The agentic reporter requires an API key to run.")
        print("  To complete the workflow:")
        print("    1. export ANTHROPIC_API_KEY='your-key'")
        print("    2. python integrated_pipeline.py")
        print("\n  For now, showing what WOULD happen:")
        print("-" * 70)
        
        # Show simulated agent investigation
        print("\n  Agent would:")
        print("    1. Read drift report from", drift_report_path)
        print("    2. Call get_psi_report('dti') → PSI 0.41, RETRAIN status")
        print("    3. Call get_psi_report('revol_util') → PSI 1.93, RETRAIN status")
        print("    4. Call get_model_performance('current') → Check degradation")
        print("    5. Call get_vintage_cohort('Q3_2017') → Identify problem cohorts")
        print("    6. Call get_champion_challenger_status() → Check if promotion ready")
        print("    7. Synthesize findings into plain-English report")
        print("\n  Example output: see example_monthly_report.md")
        print("\n  Integration complete (simulation mode)")
        return
    
    # API key available - run actual agent
    print(f"  API key found: {api_key[:10]}...")
    
    from agent_tools import CreditLensTools
    from agentic_reporter import CreditLensAgent
    
    # Initialize tools
    print("\n  Initializing investigation tools...")
    tools = CreditLensTools(
        db_path='/tmp/demo.duckdb',  # In production: data/creditlens.duckdb
        drift_reports_dir=drift_report_dir
    )
    
    # Initialize agent
    print("  Initializing agent (Claude Sonnet 4)...")
    agent = CreditLensAgent(
        api_key=api_key,
        tools_instance=tools
    )
    
    # Generate report
    print("\n  Generating monthly portfolio report...")
    print("  (Agent will investigate drift patterns and synthesize findings)")
    print("-" * 70)
    
    report = agent.generate_monthly_report(
        investigation_prompt=f"""Generate this month's portfolio risk report.
        
The drift detection system flagged {len(drift_summary['retrain_features'])} features for retraining:
{', '.join(drift_summary['retrain_features'])}

Investigate:
1. Why did these features drift?
2. Is model performance degraded?
3. Which cohorts are affected?
4. Should we promote the challenger model?
5. What should stakeholders know?

Generate a clear, actionable report."""
    )
    
    print("\n" + "=" * 70)
    print("[WEEK 8] Generated Report")
    print("=" * 70)
    print(report)
    print("=" * 70)
    
    # Save report
    report_path = agent.save_report(
        report,
        output_dir=Path('../outputs/agent_reports'),
        report_type='monthly'
    )
    
    # ========================================
    # PIPELINE COMPLETE
    # ========================================
    
    print("\n" + "=" * 70)
    print("Integrated Pipeline Complete")
    print("=" * 70)
    print("\nOutputs:")
    print(f"  Drift report: {drift_report_path}")
    print(f"  Agent report: {report_path}")
    print("\nNext steps:")
    print("  1. Review agent report for business insights")
    print("  2. Share report with stakeholders")
    print("  3. If retraining recommended, execute retrain workflow")
    print("  4. Pipeline runs again next month")
    print("=" * 70)


def demonstrate_integration_pattern():
    """
    Show the code pattern for Week 7 → Week 8 integration.
    
    This is what goes in your production pipeline.
    """
    print("\n" + "=" * 70)
    print("Production Integration Pattern")
    print("=" * 70)
    
    code_example = """
# In pipeline/monthly_refresh.py

from drift_detector import DriftDetector
from retrain_trigger import RetrainingTrigger
from agent.tools import CreditLensTools
from agent.reporter import CreditLensAgent
import os

def monthly_pipeline():
    # WEEK 7: Drift Detection
    detector = DriftDetector(n_bins=10)
    
    results = detector.monitor_features(
        reference_df=train_df,
        current_df=prod_df,
        features=['dti', 'annual_income', 'revol_util', 'loan_amount']
    )
    
    drift_summary = detector.get_drift_summary(results)
    detector.save_results(results, 'outputs/drift_reports/drift_latest.json')
    
    # Retraining decision
    trigger = RetrainingTrigger()
    decision = trigger.evaluate(
        drift_summary=drift_summary,
        model_training_date=model_metadata['training_date']
    )
    
    # WEEK 8: Agentic Investigation (if drift detected)
    if decision.should_retrain or drift_summary['recommendation'] == 'INVESTIGATE':
        
        # Initialize agent
        tools = CreditLensTools(
            db_path='data/creditlens.duckdb',
            drift_reports_dir='outputs/drift_reports'
        )
        
        agent = CreditLensAgent(
            api_key=os.environ['ANTHROPIC_API_KEY'],
            tools_instance=tools
        )
        
        # Generate report
        report = agent.generate_monthly_report()
        
        # Save for stakeholders
        agent.save_report(
            report,
            output_dir='outputs/agent_reports',
            report_type='monthly'
        )
        
        # Optional: Email to distribution list
        # send_email(to=STAKEHOLDERS, body=report, subject='Monthly Credit Report')
        
    else:
        print("Portfolio stable - no investigation needed")
"""
    
    print(code_example)
    print("=" * 70)


if __name__ == '__main__':
    # Run integrated pipeline
    integrated_monthly_pipeline()
    
    # Show code pattern
    print("\n")
    demonstrate_integration_pattern()
