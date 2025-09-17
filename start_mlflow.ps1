# Start MLflow server locally for development

Write-Host "Starting MLflow server..."
Write-Host "This will run MLflow UI on http://localhost:5000"
Write-Host "Press Ctrl+C to stop"

# Create mlruns directory if it doesn't exist
if (!(Test-Path "mlruns")) {
    New-Item -ItemType Directory -Path "mlruns"
}

# Start MLflow server
mlflow server `
    --backend-store-uri sqlite:///mlflow.db `
    --default-artifact-root ./mlruns `
    --host 0.0.0.0 `
    --port 5000