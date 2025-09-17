#!/bin/bash
# Start MLflow server locally for development

echo "Starting MLflow server..."
echo "This will run MLflow UI on http://localhost:5000"
echo "Press Ctrl+C to stop"

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Start MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000