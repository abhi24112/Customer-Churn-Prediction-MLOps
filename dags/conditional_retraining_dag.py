from airflow import DAG # type: ignore
from airflow.operators.python import BranchPythonOperator # type: ignore 
from airflow.operators.bash import BashOperator # type: ignore
from airflow.operators.empty import EmptyOperator # type: ignore
from datetime import datetime
import sys

# Project root path inside the Airflow container
PROJECT_ROOT = "/usr/local/airflow"

def check_drift_func():
    """
    Calls the Evidently monitoring script to check for data drift.
    Returns the task_id of the next task to execute.
    """
    # Ensure src is in the python path
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    
    try:
        from src.drift_detection.evidenly_monitoring import run_evidently_monitoring
        
        # Run the monitoring logic
        # dataset_drift is a boolean (True if drift > threshold)
        drift_detected = run_evidently_monitoring()
        
        if drift_detected:
            print("🚨 Data drift detected! Branching to retraining.")
            return 'run_retraining'
        else:
            print("✅ No data drift detected. Skipping retraining.")
            return 'skip_retraining'
    except Exception as e:
        print(f"❌ Error in drift monitoring: {e}")
        raise ValueError(f"Drift monitoring failed: {e}")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 5, 10),
    'retries': 2
}

with DAG(
    dag_id="conditional_retraining_logic",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    tags=['mlops', 'drift', 'retraining']
) as dag:

    # 1. Pull latest artifacts (Model & Reference Data)
    dvc_pull = BashOperator(
        task_id='dvc_pull',
        bash_command=f"cd {PROJECT_ROOT} && git config --global --add safe.directory {PROJECT_ROOT} && dvc pull"
    )

    # 2. Update Current Data (Snapshot & Preprocessing)
    data_update = BashOperator(
        task_id='data_update',
        bash_command=f"cd {PROJECT_ROOT} && git config --global --add safe.directory {PROJECT_ROOT} && dvc repro -f db_snapshot && dvc repro data_preprocessing"
    )

    # 3. Drift Monitoring & Branching
    check_drift = BranchPythonOperator(
        task_id='check_drift',
        python_callable=check_drift_func
    )

    # 4. Retraining Path (triggered if drift detected)
    run_retraining = BashOperator(
        task_id='run_retraining',
        bash_command=f"cd {PROJECT_ROOT} && git config --global --add safe.directory {PROJECT_ROOT} && dvc repro training && dvc push"
    )

    # 5. Skip Path (triggered if no drift)
    skip_retraining = EmptyOperator(
        task_id='skip_retraining'
    )

    # Define the workflow
    dvc_pull >> data_update >> check_drift
    check_drift >> [run_retraining, skip_retraining]
