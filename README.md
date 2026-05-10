# Bank Customer Churn Prediction - MLOps Pipeline

**Python · DVC · Docker · Airflow · CatBoost · MLflow (Ready)**

---

## Project Summary

This project is a complete **MLOps pipeline** to predict whether a bank customer will churn or subscribe to a term deposit based on campaign data. It integrates reproducible model training with DVC, orchestrated pipelines using Airflow, and containerization with Docker—designed to be production-ready and easily expandable.

---

## Project Folder Structure

```
📁 .dvc/                    → DVC configuration & cache
📁 airflow_settings.yaml    → Airflow connections and variables (local setup)
📁 catboost_info/           → CatBoost training logs and metrics
📁 config/                  → Configuration files (config.yaml)
📁 dags/                    → Airflow DAGs for pipeline orchestration
📁 documents/               → Project documentation
📁 dvc_storage/             → DVC local storage & versioning
📁 models/                  → Trained model artifacts (CatBoost .jbl)
📁 notebooks/               → Jupyter notebooks (EDA, analysis)
📁 reports/                 → Metrics and reports (metrics.json)
📁 src/                     → Source code (core ML pipeline)
│  ├── components/          → Modular ML components
│  ├── data/                → Raw & processed data
│  ├── pipelines/           → Full training & preprocessing pipelines
│  └── utils/               → Utility functions
📁 tests/                   → Unit & integration tests
📄 Dockerfile               → Docker containerization
📄 docker-compose.yml       → (Ready for multi-container setup)
📄 dvc.yaml                 → DVC pipeline definition
📄 params.yaml              → Hyperparameters & config
📄 requirements.txt         → Python dependencies
📄 main.py                  → Entry point script
```

---

## Tools & Technologies

| Tool/Tech      | Purpose                                      |
| -------------- | -------------------------------------------- |
| Python 3.x     | Core programming language                    |
| DVC            | Data & model versioning                      |
| CatBoost       | Gradient boosting model training             |
| Airflow (Astro)| Pipeline orchestration & scheduling          |
| Docker         | Containerization & reproducible environments |
| MLflow         | Experiment tracking & model registry         |
| Evidently AI   | Data drift detection & model monitoring      |
| PostgreSQL     | Live data store (Source of Truth)            |
| Scikit-learn   | Data preprocessing & evaluation metrics      |
| Pandas & NumPy | Data manipulation & analysis                 |

---

## Data Overview

**Bank Marketing Dataset** (Managed via PostgreSQL)

- **Source**: UCI Bank Marketing Dataset
- **Target**: Customer subscription to term deposit (binary classification)
- **Size**: ~45K records, 16 features + 1 target
- **Live Store**: PostgreSQL (`churn_raw` table)
- **Pipeline Source**: `src/data/raw_data/data.csv` (DVC snapshot)

---

## ML Pipeline Architecture

```
PostgreSQL (Live Data)
   ↓
[DB Snapshot] - DVC stage: exports CSV from Postgres
   ↓
[Data Preprocessing] - DVC stage: cleaning & feature engineering
   ↓
[Drift Detection] - Evidently AI: compares Current vs Reference data
   ↓
[Branching Logic] - Airflow: decides if retraining is needed
   ↓
[Model Training] - DVC stage: Retrains ONLY if drift is detected
   ↓
[Evaluation & MLflow] - Logs metrics, plots, and models
```

---

## Core Components

### 1. Data Ingestion & Snapshot (`src/pipelines/db_snapshot_pipeline.py`)
- Pulls live data from PostgreSQL.
- Versions the snapshot using DVC to ensure reproducibility.

### 2. Data Preprocessing (`src/components/data_preprocessing.py`)
- Handles missing values, categorical encoding, and feature scaling.

### 3. Drift Monitoring (`src/drift_detection/evidenly_monitoring.py`)
- Uses **Evidently AI** to detect feature and target drift.
- Returns a boolean status to trigger conditional workflows.

### 4. Model Training & Tracking (`src/pipelines/training_pipeline.py`)
- Trains CatBoost, XGBoost, or LightGBM based on `params.yaml`.
- Fully integrated with **MLflow** for tracking experiments, parameters, and artifacts.

---

## Airflow DAGs

Located in `dags/`:

| DAG Name                      | Frequency | Purpose                                              |
| ----------------------------- | --------- | ---------------------------------------------------- |
| `conditional_retraining_logic`| Scheduled | **Smart Pipeline**: Drift check -> Conditional Retrain|
| `dvc_pipeline_dag`            | Daily     | Standard DVC pull -> repro -> push cycle            |

### Conditional Retraining Flow:
1. **Update Data**: Refresh DB snapshot and preprocess data.
2. **Check Drift**: Run Evidently monitoring.
3. **Branch**: 
   - If Drift > 50%: Trigger `dvc repro training`.
   - If No Drift: Skip training and finish.

---

## Getting Started

### 1. Setup Environment
```bash
conda activate mlopsenv
docker-compose up -d  # Start Postgres
```

### 2. Initialize Data
```powershell
$env:DATABASE_URL='postgresql://postgres:postgres@localhost:5433/churn'
python -m backend.scripts.load_csv_to_postgres  # Load initial data
```

### 3. Run the Smart Pipeline (Astro)
```bash
astro dev start
```
Go to `http://localhost:8080`, unpause `conditional_retraining_logic`, and trigger it.

---

## Next Steps (Production Readiness)

- [x] Integrate MLflow for experiment tracking
- [x] Implement data drift detection (Evidently)
- [x] Create Airflow branching logic for conditional retraining
- [ ] Add Prometheus + Grafana for system monitoring
- [ ] Add REST API (FastAPI) for model serving
- [ ] Build Streamlit UI for predictions
- [ ] Setup CI/CD with GitHub Actions
- [ ] Implement model versioning & registry in MLflow

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# DVC Configuration (optional)
DVC_REMOTE_URL=s3://your-bucket/dvc-storage

# MLflow (when integrated)
MLFLOW_TRACKING_URI=http://localhost:5000

# Airflow (when deployed)
AIRFLOW_HOME=$(pwd)
AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
```

---

## Performance Baseline

Current model metrics (see `reports/metrics.json`):

- Stored in `catboost_info/` after training
- Access training logs: `catboost_info/learn_error.tsv`

---

## Project Structure Best Practices

This project follows MLOps conventions:

- **Modular components**: Each step is isolated & testable
- **Configuration-driven**: Hyperparams in `params.yaml`
- **Version control**: DVC tracks data & models
- **Reproducibility**: `dvc repro` guarantees consistent results
- **Containerization**: Docker ensures environment portability
- **Orchestration-ready**: Airflow DAGs handle scheduling

---

## Troubleshooting

### DVC Issues

```bash
# Reinitialize DVC
dvc init --force

# Check DVC status
dvc status
```

### Postgres Snapshot Not Updating

DVC can’t automatically detect changes inside Postgres.
Use the forced stage run:

```bash
dvc repro -f db_snapshot
```

### Model Training Fails

- Check `params.yaml` for valid hyperparameters
- Ensure `src/data/raw_data/data.csv` exists
- Verify all dependencies in `requirements.txt` are installed

### Airflow DAG Import Errors

- Ensure `dags/` contains valid Python files
- Check `airflow_settings.yaml` for configuration issues

### Airflow + Postgres

The DAG runs `dvc repro -f db_snapshot` so it always refreshes the snapshot.
Make sure the Airflow runtime has `DATABASE_URL` set (via Astro env, `.env`, or container env vars).

---

## License

This project is open source and available under the MIT License.

---

## Acknowledgments

- **Dataset**: UCI Bank Marketing Dataset
- **Framework**: CatBoost, DVC, Airflow communities
- **Inspiration**: MLOps best practices & production ML patterns
