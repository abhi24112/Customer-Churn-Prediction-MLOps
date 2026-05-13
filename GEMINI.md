# Project: Customer Churn Predictions (MLOps)

This file contains the foundational architecture, workflows, and operational mandates for the Customer Churn Predictions project.

## 🚀 Critical Mandates
- **Environment (Local):** Activate the conda environment before running commands locally: `conda activate mlopsenv`.
- **Environment (Airflow/Docker):** DO NOT use `conda activate` in Airflow DAGs; dependencies are handled by the container's `requirements.txt`.
- **Database Access:** When running locally, ensure `docker-compose up -d` is executed to start PostgreSQL.
- **DVC Usage:** Use `dvc repro -f db_snapshot` to force a new data export from Postgres before running the pipeline.
- **Airflow 3.x Compatibility:** Use the `schedule` argument instead of `schedule_interval`. Use modern provider paths (e.g., `airflow.providers.standard.operators`).
- **File Protection:** NEVER modify `documents/workflow_step.txt` unless explicitly instructed by the user.

## 🏗️ Architecture Overview
The project follows a "Live Data Store -> Versioned Snapshot -> Reproducible Pipeline" pattern.

1. **Source of Truth:** PostgreSQL (`churn_raw` table).
2. **Snapshot Stage:** `src/pipelines/db_snapshot_pipeline.py` (exports Postgres to `src/data/raw_data/data.csv`).
3. **Clean Imports:** Package-level exports in `src/components/__init__.py` allow for streamlined `from src.components import (...)` statements.
4. **Monitoring:** **Evidently AI** (`src/drift_detection/evidenly_monitoring.py`) compares incoming data against the training reference.
5. **Conditional Retraining:** Airflow uses a `BranchPythonOperator` to trigger `dvc repro training` ONLY if drift is detected (>50% feature drift).
6. **Tracking:** **MLflow** tracks all experiments in a portable `mlruns` directory.

## 🛠️ Tech Stack
- **Orchestration:** Apache Airflow (via Astronomer/Astro CLI 3.x).
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

## 📂 Key File Map
- `dags/conditional_retraining_dag.py`: The branching orchestrator (Airflow 3.x compatible).
- `src/drift_detection/evidenly_monitoring.py`: The "brain" that decides if retraining is needed.
- `src/components/__init__.py`: Package-level exports for clean imports.
- `dvc.yaml`: Defines stage dependencies and commands.
- `params.yaml`: Centralized hyperparameters and split ratios.

## 📝 Recent Progress (May 2026)
- [x] Refactored component imports to use clean, package-level exports.
- [x] Resolved Airflow 3.x parsing errors (schedule vs schedule_interval).
- [x] Integrated PostgreSQL as the live data source.
- [x] Implemented Evidently AI for data drift monitoring.
- [x] Created an event-driven retraining DAG in Airflow.
- [x] Optimized `.gitignore` and finalized the Project Flow Diagram (Mermaid).
