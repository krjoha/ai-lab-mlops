#!/bin/bash
# Airflow setup script for local development

echo "Starting Airflow server..."
echo "This will run Airflow UI on http://localhost:8080"
echo "Press Ctrl+C to stop"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set Airflow home folder relative to project root
export AIRFLOW_HOME="${SCRIPT_DIR}/airflow"

# Disable example DAGs
export AIRFLOW__CORE__LOAD_EXAMPLES=false

# Set your custom DAGs folder relative to project root
export AIRFLOW__CORE__DAGS_FOLDER="${SCRIPT_DIR}/dags"

echo "Using AIRFLOW_HOME: ${AIRFLOW_HOME}"
echo "Using DAGS_FOLDER: ${AIRFLOW__CORE__DAGS_FOLDER}"

airflow standalone