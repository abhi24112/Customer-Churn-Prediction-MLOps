import os
import pandas as pd
import joblib
import logging
from typing import Tuple

from evidently import Report, DataDefinition, Dataset
from evidently.presets import DataDriftPreset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
REF_PATH = "src/data/reference_data/reference_data.csv"
CUR_PATH = "src/data/processed/processed.csv"
MODEL_PATH = "models/baseline_bank_churn_model_catboost.jbl"
REPORT_DIR = "src/drift_detection/reports"
HTML_REPORT_NAME = "evidently_drift_report.html"
JSON_REPORT_NAME = "evidently_drift_report.json"

def load_data_and_model(ref_path: str, cur_path: str, model_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    try:
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference data not found at {ref_path}")
        if not os.path.exists(cur_path):
            raise FileNotFoundError(f"Current data not found at {cur_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        ref_data = pd.read_csv(ref_path)
        cur_data = pd.read_csv(cur_path)
        model = joblib.load(model_path)
        logging.info("Data and model loaded successfully.")
        return ref_data, cur_data, model
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        raise

def run_evidently_monitoring():
    try:
        # 1. Load data
        ref_df, curr_df, model = load_data_and_model(REF_PATH, CUR_PATH, MODEL_PATH)

        # 2. Define Columns
        NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
        CATEGORICAL_COLS = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]

        data_definition = DataDefinition(
            numerical_columns=NUMERICAL_COLS,
            categorical_columns=CATEGORICAL_COLS
        )

        # 3. Create Dataset objects (In 0.7.21, DataDefinition is passed via Dataset.from_pandas)
        logging.info("Preparing Evidently datasets...")
        ref_dataset = Dataset.from_pandas(data=ref_df, data_definition=data_definition)
        curr_dataset = Dataset.from_pandas(data=curr_df, data_definition=data_definition)

        # 4. Create and run Report
        report = Report(metrics=[DataDriftPreset()])
        logging.info("Running Data Drift analysis...")
        
        # In 0.7.21, Report.run takes Dataset objects that already contain the definition
        result = report.run(reference_data=ref_dataset, current_data=curr_dataset)
        report_data = result.dict()

        # 5. Save HTML for humans
        os.makedirs(REPORT_DIR, exist_ok=True)
        result.save_html(os.path.join(REPORT_DIR, HTML_REPORT_NAME))

        # 6. --- ACTIONABLE DRIFT DETECTION ---
        result.save_json(os.path.join(REPORT_DIR, JSON_REPORT_NAME))

        drift_metric = next(
            (
                metric
                for metric in report_data.get('metrics', [])
                if metric.get('config', {}).get('type') == 'evidently:metric_v2:DriftedColumnsCount'
            ),
            None,
        )

        if drift_metric is None:
            raise ValueError("Could not find DriftedColumnsCount metric in Evidently report output")

        drift_metrics = drift_metric.get('value', {})
        number_of_columns = len(NUMERICAL_COLS) + len(CATEGORICAL_COLS)
        number_of_drifted_columns = int(drift_metrics.get('count', 0))
        drift_share = float(drift_metrics.get('share', 0.0))
        dataset_drift = drift_share > 0.5

        logging.info(f"Drift Analysis Summary:")
        logging.info(f"- Total Columns: {number_of_columns}")
        logging.info(f"- Drifted Columns: {number_of_drifted_columns}")
        logging.info(f"- Drift Share: {drift_share:.2%}")
        logging.info(f"- Dataset Drift Detected: {dataset_drift}")

        if dataset_drift:
            logging.warning("🚨 ALERT: Data Drift Detected in the dataset!")
        else:
            logging.info("✅ No significant data drift detected.")

        return report_data

    except Exception as e:
        logging.error(f"❌ Monitoring Pipeline Failed: {e}")
        raise

if __name__ == "__main__":
    run_evidently_monitoring()
