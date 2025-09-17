# MLOps Pipeline with Airflow + MLflow

An Airflow-based MLOps pipeline with MLflow integration for experiment tracking, model registry, and monitoring:

- **Data Generation** - Simulates data ingestion
- **Model Training** - Trains ML models with MLflow experiment tracking
- **Model Registry** - Stores models in MLflow registry
- **Batch Prediction** - Loads models from registry and logs results
- **Model Monitoring** - Tracks prediction trends over time
- **Pipeline Orchestration** - Airflow DAG with task dependencies

## Architecture

```
├── dags/
│   └── mlops_pipeline.py          # Airflow DAG (scheduled @daily, catchup enabled)
├── tasks/                         # Container-ready task modules
│   ├── generate_data.py           # Data ingestion simulation
│   ├── train_model.py             # ML training + MLflow experiment tracking
│   ├── batch_predict.py           # Inference using MLflow model registry
│   └── monitor_model.py           # Performance monitoring & trend analysis
├── airflow/                       # Airflow configuration and logs
│   └── airflow.db                 # Airflow metadata database
├── artifacts/                     # Local artifact backup storage
├── mlruns/                        # MLflow artifact storage
├── mlflow.db                      # MLflow backend database
├── pyproject.toml                 # Project dependencies (uv-managed)
├── start_airflow.sh               # Airflow server startup script
├── start_mlflow.sh                # MLflow tracking server startup script
└── CLAUDE.md                      # Development guidance for Claude Code
```

**Key Components:**
- **Apache Airflow**: Pipeline orchestration with daily scheduling
- **MLflow**: Experiment tracking server (localhost:5000) with SQLite backend
- **Self-Contained Tasks**: Each task can run standalone or within Airflow
- **Model Registry**: Centralized model versioning via MLflow
- **Local Development**: Uses local servers, ready for production deployment

## MLflow Integration

### Training steps
1. **Experiment Tracking**: All training runs logged to `sentiment_classification` experiment
2. **Model Registry**: Trained models automatically registered as `sentiment_classifier`
3. **Metadata**: Parameters, metrics, and artifacts tracked

### Prediction Pipeline
1. **Model Loading**: Loads latest model version from MLflow registry
2. **Monitoring**: Logs prediction metrics to `batch_predictions` experiment
3. **Quality Alerts**: Automatic alerts for low confidence or data quality issues

### Monitoring
- **Trend Analysis**: Tracks model performance over time
- **Quality Metrics**: Confidence scores, prediction distributions
- **Alert System**: Flags declining performance

## Quick Start

Install dependencies

```bash
uv sync
source .venv/bin/activate
```

Start MLflow server (separate terminal). MLflow UI: http://localhost:5000

```bash
./start_mlflow.sh
```

Start Airflow server (separate terminal). Airflow UI: http://localhost:8080
```bash
./start_airflow.sh
``` 

Run model training:

```bash
python tasks/train_model.py
```

Run prediction pipeline in the airflow UI!
