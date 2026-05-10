# Project: Customer Churn Predictions (MLOps)

This file contains the foundational architecture, workflows, and operational mandates for the Customer Churn Predictions project.

## 🚀 Critical Mandates
- **Environment (Local):** Activate the conda environment before running commands locally: `conda activate mlopsenv`.
- **Environment (Airflow/Docker):** DO NOT use `conda activate` in Airflow DAGs; dependencies are handled by the container's `requirements.txt`.
- **Database Access:** When running locally, ensure `docker-compose up -d` is executed to start PostgreSQL.
- **DVC Usage:** Use `dvc repro -f db_snapshot` to force a new data export from Postgres before running the pipeline.
- **Airflow Orchestration:** Use the `conditional_retraining_logic` DAG for production-like, data-driven retraining.

## 🏗️ Architecture Overview
The project follows a "Live Data Store -> Versioned Snapshot -> Reproducible Pipeline" pattern.

1. **Source of Truth:** PostgreSQL (`churn_raw` table).
2. **Snapshot Stage:** `src/pipelines/db_snapshot_pipeline.py` (exports Postgres to `src/data/raw_data/data.csv`).
3. **Data Versioning:** DVC manages snapshots, processed data, and model artifacts.
4. **Monitoring:** **Evidently AI** (`src/drift_detection/evidenly_monitoring.py`) compares incoming data against the training reference to detect drift.
5. **Conditional Retraining:** Airflow uses a `BranchPythonOperator` to trigger `dvc repro training` ONLY if drift is detected (>50% feature drift).
6. **Tracking:** **MLflow** tracks all experiments, metrics, and models in a portable `mlruns` directory.

## 🛠️ Tech Stack
- **Orchestration:** Apache Airflow (via Astronomer/Astro CLI).
- **Pipeline:** DVC (Data Version Control).
- **ML Engine:** CatBoost, XGBoost, LightGBM.
- **Monitoring:** Evidently AI.
- **Tracking:** MLflow.
- **Database:** PostgreSQL (Dockerized).
- **Environment:** Conda (`mlopsenv`).

## 📋 Standard Workflows

### 1. Manual Pipeline Execution
```powershell
conda activate mlopsenv
docker-compose up -d
$env:DATABASE_URL='postgresql://postgres:postgres@localhost:5433/churn'
dvc repro -f db_snapshot
dvc repro
```

### 2. Airflow Operations (Astro)
- **Start:** `astro dev start`
- **UI:** `http://localhost:8080` (admin/admin)
- **Primary DAG:** `conditional_retraining_logic` (Daily schedule + Drift-based branching).

### 3. Adding New Data
- Batch insert into Postgres -> Trigger DAG -> Automated Snapshot -> Automated Drift Check -> Conditional Retrain.

## 📂 Key File Map
- `dags/conditional_retraining_dag.py`: The branching orchestrator.
- `src/drift_detection/evidenly_monitoring.py`: The "brain" that decides if retraining is needed.
- `src/pipelines/training_pipeline.py`: Main MLflow-integrated training logic.
- `dvc.yaml`: Defines stage dependencies and commands.
- `params.yaml`: Centralized hyperparameters and split ratios.
- `config/config.yaml`: Global MLflow and project configurations.

## 📝 Recent Progress (May 2026)
- [x] Integrated PostgreSQL as the live data source.
- [x] Implemented Evidently AI for data drift monitoring.
- [x] Created an event-driven retraining DAG in Airflow.
- [x] Standardized Airflow tasks to run in the container's native environment (requirements.txt).
- [x] Full MLflow integration for tracking metrics and artifacts.
