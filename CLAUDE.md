# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLOps pipeline demonstration using Apache Airflow for orchestration and MLflow for experiment tracking and model registry. The project implements a simple sentiment classification pipeline with containerizable tasks.

## Architecture

The pipeline consists of Python tasks that can run standalone or within Airflow:

- **dags/mlops_pipeline.py**: Main Airflow DAG defining task dependencies
- **tasks/**: Containerizable task modules:
  - `generate_data.py`: Data ingestion simulation
  - `train_model.py`: ML training with MLflow experiment tracking
  - `batch_predict.py`: Inference using MLflow model registry
  - `monitor_model.py`: Performance monitoring and trend analysis
- **MLflow Integration**: All tasks use MLflow tracking server (localhost:5000) for experiment logging and model registry

## Development Commands

### Environment Setup

**Linux/Mac:**
```bash
# Install dependencies
uv sync
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
# Install dependencies
uv sync
.venv\Scripts\Activate.ps1
```

### MLflow Server

Let the user run the server, dont run it yourself

**Linux/Mac:**
```bash
# Start MLflow tracking server (required for all tasks)
./start_mlflow.sh
# UI available at http://localhost:5000
```

**Windows (PowerShell):**
```powershell
# Start MLflow tracking server (required for all tasks)
.\start_mlflow.ps1
# UI available at http://localhost:5000
```

### Airflow Server

Let the user run the server, dont run it yourself

**Linux/Mac:**
```bash
# Start Airflow (AIRFLOW_HOME: ./airflow, DAGs: ./dags)
./start_airflow.sh
# UI available at http://localhost:8080
```

**Windows (PowerShell):**
```powershell
# Start Airflow (AIRFLOW_HOME: ./airflow, DAGs: ./dags)
.\start_airflow.ps1
# UI available at http://localhost:8080
```

### Running Individual Tasks
```bash
# Tasks can be run standalone for development/testing
python tasks/train_model.py        # Trains model and registers in MLflow
python tasks/generate_data.py      # Creates test data in ./artifacts
python tasks/batch_predict.py      # Loads model from registry, runs predictions
python tasks/monitor_model.py      # Analyzes MLflow metrics and trends
```

## MLflow Integration Details

- **Tracking URI**: http://localhost:5000 (configured in all tasks)
- **Experiments**:
  - `sentiment_classification` (training runs)
  - `batch_predictions` (inference runs)
- **Model Registry**: Models registered as `sentiment_classifier`
- **Backend**: SQLite database (mlflow.db) with local artifact storage (./mlruns)

## Key Design Principles

1. **Self-Contained Tasks**: Each task can run standalone
2. **Re-runnable**: Tasks can be executed multiple times safely
3. **Container-Ready**: No external dependencies between tasks at runtime
4. **MLflow-Centric**: All model artifacts, metrics, and experiments tracked via MLflow
5. **Local Development**: Uses local MLflow server, ready for production deployment with remote tracking server

## File Structure
```
├── dags/mlops_pipeline.py          # Airflow DAG (task orchestration)
├── tasks/                          # Task implementations
├── airflow/                        # Airflow configuration and logs
├── artifacts/                      # Local artifact backup
├── mlruns/                         # MLflow artifact storage
├── mlflow.db                       # MLflow backend database
└── start_*.sh                      # Development server scripts
```