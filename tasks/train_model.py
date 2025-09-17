#!/usr/bin/env python3
"""
Model training task with MLflow integration.
Can be run standalone or as part of Airflow DAG.
"""
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import json
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def train_model():
    """Train sentiment model and register in MLflow"""
    print("Starting model training task with MLflow...")

    # Set MLflow tracking URI (local server)
    mlflow.set_tracking_uri("http://localhost:5000")

    # Set or create experiment
    experiment_name = "sentiment_classification"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"MLflow connection issue: {e}")
        print("Continuing without MLflow logging...")
        experiment_id = None

    # Create output directory for local artifacts (backup)
    model_dir = Path("./artifacts/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Simple training data
    data = [
        ("great product, love it", "positive"),
        ("amazing quality", "positive"),
        ("excellent service", "positive"),
        ("wonderful experience", "positive"),
        ("fantastic results", "positive"),
        ("terrible experience", "negative"),
        ("very bad quality", "negative"),
        ("awful service", "negative"),
        ("disappointing results", "negative"),
        ("poor performance", "negative"),
    ]

    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    print(f"Training with {len(df)} samples")

    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100)),
        ('classifier', MultinomialNB())
    ])

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log parameters
        mlflow.log_param("model_type", "MultinomialNB")
        mlflow.log_param("vectorizer_max_features", 100)
        mlflow.log_param("training_samples", len(df))

        # Train model
        pipeline.fit(df['text'], df['sentiment'])

        # Evaluate model
        train_score = pipeline.score(df['text'], df['sentiment'])
        cv_scores = cross_val_score(pipeline, df['text'], df['sentiment'], cv=3, scoring='accuracy')

        print(f"Training accuracy: {train_score:.3f}")
        print(f"Cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Log metrics
        mlflow.log_metric("training_accuracy", train_score)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())

        # Create model signature
        sample_input = df['text'].head(2)
        predictions = pipeline.predict(sample_input)
        signature = infer_signature(sample_input.tolist(), predictions)

        # Log and register model
        model_name = "sentiment_classifier"

        # Log model to MLflow (using MLflow 3.0+ approach with name parameter)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",  # Still needed for run context, but will show deprecation warning
            signature=signature,
        )

        # Register model in MLflow registry
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_name
        )

        # Get model version info
        client = mlflow.MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        model_version = model_versions[0].version if model_versions else "1"

        print(f"Model registered in MLflow as {model_name} version {model_version}")

        # Save metadata with MLflow info
        metadata = {
            "model_type": "MultinomialNB with TfidfVectorizer",
            "training_date": datetime.now().isoformat(),
            "training_samples": len(df),
            "training_accuracy": float(train_score),
            "cv_accuracy": float(cv_scores.mean()),
            "classes": pipeline.classes_.tolist(),
            "mlflow_experiment_id": str(experiment_id),
            "mlflow_run_id": run.info.run_id,
            "mlflow_model_name": model_name,
            "mlflow_model_version": model_version
        }

        # Log metadata as MLflow artifact
        metadata_path = model_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(str(metadata_path))

        print(f"Model registered in MLflow registry: {model_name}:v{model_version}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("Model training task completed successfully")

if __name__ == "__main__":
    train_model()