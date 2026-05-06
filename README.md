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
| Airflow        | Pipeline orchestration & scheduling          |
| Docker         | Containerization & reproducible environments |
| Scikit-learn   | Data preprocessing & evaluation metrics      |
| Pandas & NumPy | Data manipulation & analysis                 |
| Jupyter        | Interactive notebooks for EDA                |
| MLflow         | Model tracking (ready to integrate)          |

---

## Data Overview

**Bank Marketing Dataset**

- **Source**: UCI Bank Marketing Dataset
- **Target**: Customer subscription to term deposit (binary classification)
- **Size**: ~45K records, 16 features + 1 target
- **Location**: `src/data/raw_data/data.csv`

### Key Features:

| Feature   | Type       | Description                     |
| --------- | ---------- | ------------------------------- |
| age       | Numeric    | Customer age                    |
| job       | Category   | Employment type                 |
| marital   | Category   | Marital status                  |
| education | Category   | Education level                 |
| default   | Binary     | Credit default status           |
| balance   | Numeric    | Account balance                 |
| housing   | Binary     | Has housing loan                |
| loan      | Binary     | Has personal loan               |
| contact   | Category   | Contact type                    |
| duration  | Numeric    | Call duration (seconds)         |
| campaign  | Numeric    | Number of campaign contacts     |
| pdays     | Numeric    | Days since previous contact     |
| previous  | Numeric    | Previous campaign contacts      |
| poutcome  | Category   | Previous campaign outcome       |
| **y**     | **Binary** | **Target: Subscribed (yes/no)** |

---

## ML Pipeline Architecture

```
Raw Data
   ↓
[Data Ingestion] - src/components/data_ingestion.py
   ↓
[Data Preprocessing] - src/components/data_preprocessing.py
   ↓
[Data Splitting] - src/components/data_splitting.py
   ├─→ Training Set
   ├─→ Validation Set
   └─→ Test Set
   ↓
[Model Training] - src/components/model_training.py
   ↓
[Model Evaluation] - src/components/evaluate.py
   ↓
[Model Registry] - src/components/model_saving.py
   ↓
Ready for Serving/Inference
```

---

## Core Components

### 1. Data Ingestion (`src/components/data_ingestion.py`)

- Loads raw data from CSV
- Initial data validation and schema checks

### 2. Data Preprocessing (`src/components/data_preprocessing.py`)

- Handles missing values
- Categorical encoding (one-hot, label encoding)
- Feature scaling and normalization

### 3. Data Splitting (`src/components/data_splitting.py`)

- Splits data into train/test/validation sets
- Maintains class distribution (stratified split)

### 4. Model Training (`src/components/model_training.py`)

- CatBoost model training with hyperparameter tuning
- Logs training metrics & loss curves

### 5. Model Evaluation (`src/components/evaluate.py`)

- Classification metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrix & ROC curves
- Performance reporting

### 6. Model Saving (`src/components/model_saving.py`)

- Saves trained model to `models/` directory
- Exports in joblib (.jbl) format

---

## Airflow DAGs

Located in `dags/`:

| DAG Name           | Frequency | Purpose                             |
| ------------------ | --------- | ----------------------------------- |
| `dvc_pipeline_dag` | Scheduled | Orchestrates DVC pipeline execution |
| `exampledag`       | Example   | Reference DAG structure             |

DAGs handle:

- Triggering data pipeline stages
- Model retraining on schedule
- Logging & error handling
- Integration with DVC pipeline

---

## DVC Pipeline

The reproducible ML pipeline is defined in `dvc.yaml` and `params.yaml`.

### Running the Pipeline:

```bash
# View pipeline stages
dvc dag

# Run full pipeline
dvc repro

# Run specific stage
dvc repro src/components/data_ingestion.py
```

### Parameters (`params.yaml`):

Centralized configuration for:

- Train/test split ratios
- Model hyperparameters (learning rate, depth, iterations)
- Feature engineering settings

---

## Getting Started

### 1. Clone & Setup

```bash
cd "Customer Churn Predictions"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the ML Pipeline

```bash
# Using DVC
dvc repro

# Or run main script
python main.py
```

### 4. Train Model

```bash
python src/pipelines/training_pipeline.py
```

### 5. Evaluate Model

```bash
python src/components/evaluate.py
```

---

## Docker Setup

### Build Docker Image

```bash
docker build -t bank-churn-prediction:latest .
```

### Run in Container

```bash
docker run -v $(pwd)/data:/app/data bank-churn-prediction:latest python main.py
```

### Multi-Container Setup (Ready)

```bash
docker-compose up --build
```

(Extend `docker-compose.yml` to include PostgreSQL, MLflow, Airflow services as needed)

---

## File Reference

| File                             | Purpose                     |
| -------------------------------- | --------------------------- |
| `main.py`                        | Entry point script          |
| `config/config.yaml`             | Global configuration        |
| `dvc.yaml`                       | DVC pipeline stages         |
| `params.yaml`                    | Model hyperparameters       |
| `requirements.txt`               | Python package dependencies |
| `Dockerfile`                     | Container image definition  |
| `models/baseline_bank_churn_...` | Trained CatBoost model      |
| `notebooks/Data Analysis.ipynb`  | EDA & exploratory analysis  |
| `tests/dags/test_dag_example.py` | DAG tests                   |

---

## Next Steps (Production Readiness)

- [ ] Integrate MLflow for experiment tracking
- [ ] Add Prometheus + Grafana for monitoring
- [ ] Implement data drift detection (Evidently)
- [ ] Add REST API (FastAPI) for model serving
- [ ] Build Streamlit UI for predictions
- [ ] Setup CI/CD with GitHub Actions
- [ ] Add comprehensive unit & integration tests
- [ ] Implement model versioning & registry
- [ ] Setup PostgreSQL for production data logging

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

### Model Training Fails

- Check `params.yaml` for valid hyperparameters
- Ensure `src/data/raw_data/data.csv` exists
- Verify all dependencies in `requirements.txt` are installed

### Airflow DAG Import Errors

- Ensure `dags/` contains valid Python files
- Check `airflow_settings.yaml` for configuration issues

---

## License

This project is open source and available under the MIT License.

---

## Acknowledgments

- **Dataset**: UCI Bank Marketing Dataset
- **Framework**: CatBoost, DVC, Airflow communities
- **Inspiration**: MLOps best practices & production ML patterns
