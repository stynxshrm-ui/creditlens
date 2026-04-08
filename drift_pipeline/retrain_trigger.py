"""
Retraining trigger logic for CreditLens.

Determines when to trigger model retraining based on:
1. PSI drift on monitored features
2. Prediction distribution drift
3. Performance degradation on recent cohorts
4. Business rules (mandatory quarterly retraining)

Regulatory context: SR 11-7 requires periodic model validation and
retraining when material changes occur in the population or environment.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path


@dataclass
class RetrainingDecision:
    """Container for retraining decision and rationale."""
    should_retrain: bool
    trigger_reason: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    triggered_features: List[str]
    timestamp: datetime
    additional_context: Dict


class RetrainingTrigger:
    """
    Decision engine for model retraining.
    
    Combines multiple signals to determine if retraining is necessary.
    """
    
    def __init__(
        self,
        psi_threshold_retrain: float = 0.25,
        psi_threshold_monitor: float = 0.10,
        min_monitored_features_retrain: int = 1,
        max_days_since_training: int = 90,
        performance_degradation_threshold: float = 0.05  # 5% Gini drop
    ):
        """
        Initialize retraining trigger.
        
        Args:
            psi_threshold_retrain: PSI value that triggers immediate retraining
            psi_threshold_monitor: PSI value that triggers monitoring
            min_monitored_features_retrain: Min features in retrain zone to trigger
            max_days_since_training: Force retraining after this many days
            performance_degradation_threshold: Gini drop threshold
        """
        self.psi_threshold_retrain = psi_threshold_retrain
        self.psi_threshold_monitor = psi_threshold_monitor
        self.min_monitored_features_retrain = min_monitored_features_retrain
        self.max_days_since_training = max_days_since_training
        self.performance_degradation_threshold = performance_degradation_threshold
        
    def evaluate(
        self,
        drift_summary: Dict,
        model_training_date: datetime,
        performance_metrics: Optional[Dict] = None
    ) -> RetrainingDecision:
        """
        Evaluate all signals and make retraining decision.
        
        Args:
            drift_summary: Output from DriftDetector.get_drift_summary()
            model_training_date: When the current champion was trained
            performance_metrics: Optional dict with recent performance
            
        Returns:
            RetrainingDecision with recommendation and rationale
        """
        triggers = []
        severity = 'low'
        
        # Signal 1: PSI-based drift
        retrain_count = drift_summary.get('retrain_count', 0)
        retrain_features = drift_summary.get('retrain_features', [])
        
        if retrain_count >= self.min_monitored_features_retrain:
            triggers.append(
                f"PSI drift: {retrain_count} feature(s) exceed threshold "
                f"({', '.join(retrain_features)})"
            )
            severity = 'high'
            
        # Signal 2: Prediction drift
        psi_values = drift_summary.get('psi_values', {})
        pred_psi = psi_values.get('model_predictions', 0)
        
        if pred_psi >= self.psi_threshold_retrain:
            triggers.append(
                f"Prediction distribution drift: PSI = {pred_psi:.3f}"
            )
            severity = 'high'
            
        # Signal 3: Time-based mandatory retraining
        days_since_training = (datetime.now() - model_training_date).days
        
        if days_since_training > self.max_days_since_training:
            triggers.append(
                f"Time-based trigger: {days_since_training} days since last training "
                f"(threshold: {self.max_days_since_training})"
            )
            if severity == 'low':
                severity = 'medium'
                
        # Signal 4: Performance degradation (if metrics provided)
        if performance_metrics:
            current_gini = performance_metrics.get('current_gini', 0)
            training_gini = performance_metrics.get('training_gini', 0)
            
            if training_gini > 0:
                gini_drop = training_gini - current_gini
                
                if gini_drop > self.performance_degradation_threshold:
                    triggers.append(
                        f"Performance degradation: Gini dropped by "
                        f"{gini_drop:.3f} (from {training_gini:.3f} to {current_gini:.3f})"
                    )
                    severity = 'critical'
                    
        # Signal 5: Multiple features in monitor zone (soft signal)
        monitor_count = drift_summary.get('monitor_count', 0)
        monitor_features = drift_summary.get('monitor_features', [])
        
        if monitor_count >= 3 and retrain_count == 0:
            triggers.append(
                f"Elevated monitoring: {monitor_count} features in monitor zone "
                f"({', '.join(monitor_features[:3])}...)"
            )
            if severity == 'low':
                severity = 'medium'
                
        # Make decision
        should_retrain = len(triggers) > 0
        
        if should_retrain:
            trigger_reason = '; '.join(triggers)
        else:
            trigger_reason = 'No retraining triggers detected'
            
        return RetrainingDecision(
            should_retrain=should_retrain,
            trigger_reason=trigger_reason,
            severity=severity,
            triggered_features=retrain_features,
            timestamp=datetime.now(),
            additional_context={
                'drift_summary': drift_summary,
                'days_since_training': days_since_training,
                'performance_metrics': performance_metrics
            }
        )
    
    def create_retraining_task(
        self,
        decision: RetrainingDecision,
        output_dir: Path
    ) -> Optional[str]:
        """
        Create a retraining task file if decision is to retrain.
        
        This file signals the pipeline to start retraining.
        In production, this would trigger a workflow (Airflow, Prefect, etc.)
        
        Args:
            decision: RetrainingDecision from evaluate()
            output_dir: Directory to write task file
            
        Returns:
            Path to task file if created, None otherwise
        """
        if not decision.should_retrain:
            return None
            
        task_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        task_file = output_dir / f'retrain_task_{task_id}.json'
        
        task_data = {
            'task_id': task_id,
            'created_at': decision.timestamp.isoformat(),
            'severity': decision.severity,
            'reason': decision.trigger_reason,
            'triggered_features': decision.triggered_features,
            'context': decision.additional_context
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(task_file, 'w') as f:
            json.dump(task_data, f, indent=2)
            
        print(f"Retraining task created: {task_file}")
        print(f"Severity: {decision.severity}")
        print(f"Reason: {decision.trigger_reason}")
        
        return str(task_file)
    
    def log_decision(
        self,
        decision: RetrainingDecision,
        log_file: Path
    ):
        """
        Log retraining decision for audit trail.
        
        Regulatory requirement: maintain decision history.
        """
        log_entry = {
            'timestamp': decision.timestamp.isoformat(),
            'decision': 'RETRAIN' if decision.should_retrain else 'NO_ACTION',
            'severity': decision.severity,
            'reason': decision.trigger_reason,
            'triggered_features': decision.triggered_features
        }
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


def simulate_drift_scenarios():
    """
    Demonstrate retraining trigger logic with different scenarios.
    """
    print("=== Retraining Trigger Scenarios ===\n")
    
    trigger = RetrainingTrigger(
        psi_threshold_retrain=0.25,
        min_monitored_features_retrain=1,
        max_days_since_training=90
    )
    
    # Scenario 1: No drift, recent training
    print("Scenario 1: Stable model, no triggers")
    drift_summary_stable = {
        'retrain_count': 0,
        'monitor_count': 1,
        'retrain_features': [],
        'monitor_features': ['open_acc'],
        'psi_values': {'model_predictions': 0.05}
    }
    
    decision_1 = trigger.evaluate(
        drift_summary=drift_summary_stable,
        model_training_date=datetime.now() - timedelta(days=30)
    )
    
    print(f"  Should retrain: {decision_1.should_retrain}")
    print(f"  Reason: {decision_1.trigger_reason}")
    print(f"  Severity: {decision_1.severity}\n")
    
    # Scenario 2: Feature drift detected
    print("Scenario 2: Significant feature drift")
    drift_summary_drift = {
        'retrain_count': 2,
        'monitor_count': 3,
        'retrain_features': ['dti', 'annual_income'],
        'monitor_features': ['revol_util', 'loan_amount', 'open_acc'],
        'psi_values': {
            'dti': 0.28,
            'annual_income': 0.31,
            'model_predictions': 0.15
        }
    }
    
    decision_2 = trigger.evaluate(
        drift_summary=drift_summary_drift,
        model_training_date=datetime.now() - timedelta(days=45)
    )
    
    print(f"  Should retrain: {decision_2.should_retrain}")
    print(f"  Reason: {decision_2.trigger_reason}")
    print(f"  Severity: {decision_2.severity}\n")
    
    # Scenario 3: Time-based trigger
    print("Scenario 3: Mandatory quarterly retraining")
    decision_3 = trigger.evaluate(
        drift_summary=drift_summary_stable,
        model_training_date=datetime.now() - timedelta(days=95)
    )
    
    print(f"  Should retrain: {decision_3.should_retrain}")
    print(f"  Reason: {decision_3.trigger_reason}")
    print(f"  Severity: {decision_3.severity}\n")
    
    # Scenario 4: Performance degradation
    print("Scenario 4: Performance degradation detected")
    performance_metrics = {
        'training_gini': 0.71,
        'current_gini': 0.64,  # 7% drop
        'training_ks': 0.42,
        'current_ks': 0.38
    }
    
    decision_4 = trigger.evaluate(
        drift_summary=drift_summary_stable,
        model_training_date=datetime.now() - timedelta(days=60),
        performance_metrics=performance_metrics
    )
    
    print(f"  Should retrain: {decision_4.should_retrain}")
    print(f"  Reason: {decision_4.trigger_reason}")
    print(f"  Severity: {decision_4.severity}\n")


if __name__ == '__main__':
    simulate_drift_scenarios()
