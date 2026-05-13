import os
import sys
import joblib
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Loaded environment variables from .env")
except ImportError:
    logging.warning("python-dotenv not installed. Environment variables must be set manually.")

# database imports
from backend.postgres import insert_row

# Components
from src.components import preprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Loading the model
MODEL_NAME = "baseline_bank_churn_model_catboost.jbl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", MODEL_NAME) # api -> .. {root folder} -> models -> model_name
try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to import the model :{e}")

# App Initialization
app = FastAPI(
    title="Customer Churn Predictor API"
)

# Pydantic model for input validation of the model input data
class CustomerData(BaseModel):
    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Background task: logging data and prediction in database
def log_to_db(record: dict, prediction: int):
    """
    Background task to log the prediction and record data in database.
    """
    logging.info(f"Attempting to log prediction to DB for customer: {record.get('customerID')}")
    try:
        # Add prediction to record
        record['Churn'] = "Yes" if prediction == 1 else "No"
        
        # Ensure numeric values are converted to strings if the DB columns are 'text'
        for key in ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']:
            if key in record and record[key] is not None:
                record[key] = str(record[key])

        inserted = insert_row("churn_raw", record)
        logging.info(f"✅ Successfully logged {inserted} record to 'churn_raw'")
    except Exception as e:
        logging.error(f"❌ Database Insertion Failed: {e}")
        logging.error("Hint: Ensure DATABASE_URL is set correctly (e.g., port 5433 for local docker).")


@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(data:CustomerData, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    try:
        # 1. Converting input data to teh pandas dataframe
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])

        # 2. Preprocess (Cleaning data)
        processed_df = preprocess(df)

        # reordering column to match teh training order
        model_features = model.feature_names_
        processed_df = processed_df[model_features]

        # 4. Convert categorical features to strings
        cat_indices = model.get_cat_feature_indices()
        for idx in cat_indices:
            col_name = model_features[idx]
            processed_df[col_name] = processed_df[col_name].astype(str)

        # 5. predictions
        prediction = int(model.predict(processed_df)[0])
        probability = float(model.predict_proba(processed_df)[0][1])

        # 6. Running background task
        background_tasks.add_task(
            log_to_db,
            input_dict,
            prediction
        )

        return {
            "prediction": prediction,
            "probability": round(probability, 4)
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)