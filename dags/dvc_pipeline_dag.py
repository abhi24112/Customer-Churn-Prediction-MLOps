from airflow import DAG # type: ignore
from datetime import datetime

# Operators
from airflow.operators.bash import BashOperator # type: ignore

with DAG(
    dag_id = "Customer_Churn_DVC_pipeline",
    description = "Run full ML pipeline using DVC Daily",
    start_date = datetime(2026,5,3),
    schedule = "@daily",
    catchup = False
) as dag:
    
    # Pulling the latest data from DVC remote
    dvc_pull = BashOperator(
        task_id = "dvc_pull_data",
        bash_command="cd /usr/local/airflow && git config --global --add safe.directory /usr/local/airflow && dvc pull"
    )

    # Run full pipeline
    dvc_repro = BashOperator(
        task_id = "run_dvc_pipeline",
        bash_command="cd /usr/local/airflow && git config --global --add safe.directory /usr/local/airflow && dvc repro -f db_snapshot && dvc repro training"
    )

    # Pushing results (model and metrics) to remote
    dvc_push = BashOperator(
        task_id = "dvc_push_result",
        bash_command="cd /usr/local/airflow && git config --global --add safe.directory /usr/local/airflow && dvc push"
    )

    # Flow
    dvc_pull >> dvc_repro >> dvc_push







