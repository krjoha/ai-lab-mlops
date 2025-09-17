# Airflow setup script for local development

Write-Host "Starting Airflow server..."
Write-Host "This will run Airflow UI on http://localhost:8080"
Write-Host "Press Ctrl+C to stop"

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Set Airflow home folder relative to project root
$env:AIRFLOW_HOME = Join-Path $ScriptDir "airflow"

# Disable example DAGs
$env:AIRFLOW__CORE__LOAD_EXAMPLES = "false"

# Set your custom DAGs folder relative to project root
$env:AIRFLOW__CORE__DAGS_FOLDER = Join-Path $ScriptDir "dags"

Write-Host "Using AIRFLOW_HOME: $env:AIRFLOW_HOME"
Write-Host "Using DAGS_FOLDER: $env:AIRFLOW__CORE__DAGS_FOLDER"

airflow standalone