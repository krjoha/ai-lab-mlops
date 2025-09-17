#!/usr/bin/env python3
"""
Batch prediction task with MLflow integration.
Loads model from MLflow registry and logs prediction results for monitoring.
"""
import pandas as pd
import joblib
from pathlib import Path
import json
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np

def batch_predict():
    """Run batch predictions using model from MLflow registry"""
    print("Starting batch prediction task with MLflow...")

    # Set MLflow tracking URI (local server)
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    # Define paths
    data_path = Path("./artifacts/data/input_data.csv")
    output_dir = Path("./artifacts/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if input data exists
    if not data_path.exists():
        raise FileNotFoundError(f"Input data not found at {data_path}. Run generate_data task first.")

    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples for prediction")

    # Load model from MLflow registry
    model_name = "sentiment_classifier"
    try:
        # Get latest version of the model
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
        model_version = latest_version.version

        print(f"Loading model {model_name} version {model_version} from MLflow registry")
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri)

    except Exception as e:
        print(f"Failed to load model from MLflow registry: {e}")
        # Fallback to local model
        local_model_path = Path("./artifacts/model/sentiment_model.pkl")
        if local_model_path.exists():
            print(f"Falling back to local model: {local_model_path}")
            model = joblib.load(local_model_path)
            model_version = "local"
        else:
            raise FileNotFoundError("No model found in MLflow registry or locally. Run train_model task first.")

    # Start MLflow run for batch prediction monitoring
    experiment_name = "batch_predictions"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = None

    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log prediction parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("batch_size", len(df))
        mlflow.log_param("prediction_date", datetime.now().isoformat())

        # Make predictions
        predictions = model.predict(df['text'])
        probabilities = model.predict_proba(df['text'])
        confidence_scores = [max(prob) for prob in probabilities]

        # Add predictions to dataframe
        df['prediction'] = predictions
        df['confidence'] = confidence_scores
        df['prediction_date'] = datetime.now().isoformat()
        df['model_version'] = model_version

        # Calculate metrics for monitoring
        avg_confidence = np.mean(confidence_scores)
        low_confidence_count = np.sum(np.array(confidence_scores) < 0.6)
        high_confidence_count = np.sum(np.array(confidence_scores) > 0.8)
        prediction_distribution = pd.Series(predictions).value_counts().to_dict()

        # Log prediction metrics to MLflow
        mlflow.log_metric("average_confidence", avg_confidence)
        mlflow.log_metric("low_confidence_predictions", low_confidence_count)
        mlflow.log_metric("high_confidence_predictions", high_confidence_count)
        mlflow.log_metric("low_confidence_percentage", (low_confidence_count / len(df)) * 100)

        # Log prediction distribution
        for label, count in prediction_distribution.items():
            mlflow.log_metric(f"predictions_{label}", count)
            mlflow.log_metric(f"predictions_{label}_percentage", (count / len(df)) * 100)

        # Show sample results
        print("Sample predictions:")
        for i, row in df.head(3).iterrows():
            print(f"  '{row['text'][:50]}...' -> {row['prediction']} (confidence: {row['confidence']:.3f})")

        # Save results
        predictions_path = output_dir / "predictions.csv"
        df.to_csv(predictions_path, index=False)

        # Calculate and save summary statistics
        summary = {
            "prediction_date": datetime.now().isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "mlflow_run_id": run.info.run_id,
            "total_predictions": len(df),
            "prediction_distribution": prediction_distribution,
            "average_confidence": float(avg_confidence),
            "low_confidence_count": int(low_confidence_count),
            "high_confidence_count": int(high_confidence_count),
            "predictions_path": str(predictions_path)
        }

        summary_path = output_dir / "prediction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Log summary as MLflow artifact
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(predictions_path))

        # Log data quality alerts
        if (low_confidence_count / len(df)) > 0.3:  # More than 30% low confidence
            mlflow.set_tag("data_quality_alert", "high_low_confidence_rate")
            print(f"WARNING: High rate of low-confidence predictions: {(low_confidence_count / len(df)) * 100:.1f}%")

        if avg_confidence < 0.6:  # Average confidence below 60%
            mlflow.set_tag("model_performance_alert", "low_average_confidence")
            print(f"WARNING: Low average confidence: {avg_confidence:.3f}")

        print(f"Predictions saved to {predictions_path}")
        print(f"Summary saved to {summary_path}")
        print(f"Prediction distribution: {prediction_distribution}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("Batch prediction task completed successfully")

if __name__ == "__main__":
    batch_predict()