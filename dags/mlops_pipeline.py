"""
MLOps Pipeline DAG

Demonstrates a simple ML pipeline:
1. Generate Data - Simulates data ingestion
2. Train Model - Trains ML model
3. Batch Predict - Runs predictions on new data
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
from pathlib import Path

# Add tasks directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "tasks"))

# Import our task functions
from train_model import train_model
from generate_data import generate_data
from batch_predict import batch_predict
from monitor_model import monitor_model

# Default arguments
default_args = {
    'owner': 'mlops-demo',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 15),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Create DAG
with DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='Simple MLOps Pipeline Demo',
    schedule='@daily',  # Run once per day at midnight
    catchup=True,
    tags=['mlops', 'demo', 'ml'],
) as dag:

    # Task 1: Generate/Ingest Data
    generate_data_task = PythonOperator(
        task_id='generate_data',
        python_callable=generate_data,
        doc_md="""
        ## Generate Data Task

        Simulates data ingestion by generating sample text data for prediction.

        **In Production**: This would be replaced with:
        - API calls to fetch new data
        - Database queries
        - File system monitoring
        - Streaming data ingestion

        **Container Ready**: This task has no dependencies on other tasks and creates its own output directory.
        """
    )

    # Task 2: Train Model
    # train_model_task = PythonOperator(
    #     task_id='train_model',
    #     python_callable=train_model,
    #     doc_md="""
    #     ## Train Model Task

    #     Trains a sentiment classification model using sample data.

    #     **In Production**: This would:
    #     - Load training data from data lake/warehouse
    #     - Use versioned datasets
    #     - Track experiments with MLflow/Weights & Biases
    #     - Store models in model registry

    #     **Container Ready**: Self-contained with all dependencies, creates model artifacts.
    #     """
    # )wd

    # Task 3: Batch Predict
    batch_predict_task = PythonOperator(
        task_id='batch_predict',
        python_callable=batch_predict,
        doc_md="""
        ## Batch Predict Task

        Runs batch predictions on new data using trained model.

        **In Production**: This would:
        - Load data from staging tables
        - Use model from model registry
        - Write predictions to data warehouse
        - Trigger downstream systems

        **Container Ready**: Checks for required inputs and fails gracefully if not available.
        """
    )

    # Task 4: Cleanup (optional)
    cleanup_task = BashOperator(
        task_id='cleanup_old_artifacts',
        bash_command="""
        echo "Cleaning up old artifacts..."
        find ./artifacts -name "*.tmp" -delete 2>/dev/null || true
        echo "Cleanup completed"
        """,
    )

    # Task 5: Model Monitoring
    monitor_model_task = PythonOperator(
        task_id='monitor_model',
        python_callable=monitor_model,
        doc_md="""
        ## Monitor Model Task

        Analyzes prediction trends and model performance over time using MLflow data.

        **In Production**: This would:
        - Query MLflow tracking server for metrics
        - Generate automated reports
        - Send alerts to monitoring systems
        - Update dashboards

        **Container Ready**: Uses MLflow client to analyze historical data.
        """
    )

    # Task 6: Summary Report
    def generate_summary():
        """Generate pipeline run summary"""
        import json
        from pathlib import Path

        summary = {
            "pipeline_run_date": datetime.now().isoformat(),
            "status": "completed",
            "artifacts_created": [],
            "mlflow_integration": "enabled"
        }

        # Check what artifacts were created
        artifacts_dir = Path("./artifacts")
        if artifacts_dir.exists():
            for file_path in artifacts_dir.rglob("*"):
                if file_path.is_file():
                    summary["artifacts_created"].append(str(file_path))

        print("MLOps Pipeline Summary:")
        print(json.dumps(summary, indent=2))

    summary_task = PythonOperator(
        task_id='generate_summary',
        python_callable=generate_summary,
    )

    # Define task dependencies
    # generate_data and train_model run in parallel
    # batch_predict needs both to complete
    # monitor_model analyzes the prediction results
    # cleanup and summary run after all tasks
    generate_data_task >> batch_predict_task >> monitor_model_task >> [cleanup_task, summary_task]
    #[generate_data_task, train_model_task] >> batch_predict_task >> monitor_model_task >> [cleanup_task, summary_task]