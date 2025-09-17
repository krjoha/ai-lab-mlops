#!/usr/bin/env python3
"""
Model monitoring task that analyzes prediction trends and model performance over time.
Uses MLflow to track metrics across multiple batch prediction runs.
"""
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

def monitor_model():
    """Analyze model performance trends from MLflow"""
    print("Starting model monitoring task...")

    # Set MLflow tracking URI (local server)
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    try:
        # Get batch prediction experiment
        experiment = mlflow.get_experiment_by_name("batch_predictions")
        if experiment is None:
            print("No batch predictions experiment found. Run batch predictions first.")
            return

        # Get recent runs (last 30 days)
        from_time = datetime.now() - timedelta(days=30)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.start_time >= {int(from_time.timestamp() * 1000)}",
            order_by=["start_time DESC"],
            max_results=50
        )

        if runs.empty:
            print("No recent prediction runs found.")
            return

        print(f"Found {len(runs)} recent prediction runs")

        # Analyze trends
        monitoring_dir = Path("./artifacts/monitoring")
        monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Calculate trend metrics
        avg_confidences = runs['metrics.average_confidence'].dropna()
        low_confidence_rates = runs['metrics.low_confidence_percentage'].dropna()

        summary = {
            "monitoring_date": datetime.now().isoformat(),
            "total_runs_analyzed": len(runs),
            "date_range": {
                "from": runs['start_time'].min().isoformat() if not runs.empty else None,
                "to": runs['start_time'].max().isoformat() if not runs.empty else None
            },
            "confidence_trends": {
                "current_avg": float(avg_confidences.iloc[0]) if not avg_confidences.empty else None,
                "overall_avg": float(avg_confidences.mean()) if not avg_confidences.empty else None,
                "trend": "improving" if len(avg_confidences) > 1 and avg_confidences.iloc[0] > avg_confidences.iloc[1] else "declining" if len(avg_confidences) > 1 else "stable"
            },
            "quality_alerts": {
                "runs_with_high_low_confidence": int((low_confidence_rates > 30).sum()) if not low_confidence_rates.empty else 0,
                "avg_low_confidence_rate": float(low_confidence_rates.mean()) if not low_confidence_rates.empty else None
            }
        }

        # Check for alerts
        alerts = []
        if summary["confidence_trends"]["trend"] == "declining":
            alerts.append("Model confidence is declining over time")

        if summary["quality_alerts"]["avg_low_confidence_rate"] and summary["quality_alerts"]["avg_low_confidence_rate"] > 25:
            alerts.append(f"High average low-confidence rate: {summary['quality_alerts']['avg_low_confidence_rate']:.1f}%")

        if summary["quality_alerts"]["runs_with_high_low_confidence"] > len(runs) * 0.5:
            alerts.append("More than 50% of runs have high low-confidence rates")

        summary["alerts"] = alerts

        # Save monitoring report
        report_path = monitoring_dir / "model_monitoring_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Start MLflow run to log monitoring results
        with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
            mlflow.set_tag("mlflow.runName", "model_monitoring")
            mlflow.log_param("monitoring_type", "batch_prediction_trends")
            mlflow.log_param("runs_analyzed", len(runs))

            # Log trend metrics
            if summary["confidence_trends"]["overall_avg"]:
                mlflow.log_metric("overall_avg_confidence", summary["confidence_trends"]["overall_avg"])

            if summary["quality_alerts"]["avg_low_confidence_rate"]:
                mlflow.log_metric("avg_low_confidence_rate", summary["quality_alerts"]["avg_low_confidence_rate"])

            mlflow.log_metric("alert_count", len(alerts))

            # Log monitoring report as artifact
            mlflow.log_artifact(str(report_path))

            print("Model Monitoring Summary:")
            print(f"  Runs analyzed: {summary['total_runs_analyzed']}")
            print(f"  Confidence trend: {summary['confidence_trends']['trend']}")
            print(f"  Current avg confidence: {summary['confidence_trends']['current_avg']:.3f}" if summary['confidence_trends']['current_avg'] else "  No confidence data")

            if alerts:
                print("  ALERTS:")
                for alert in alerts:
                    print(f"    - {alert}")
            else:
                print("  No alerts detected")

            print(f"Monitoring report saved to {report_path}")
            print("Model monitoring task completed successfully")

    except Exception as e:
        print(f"Error during monitoring: {e}")
        raise

if __name__ == "__main__":
    monitor_model()