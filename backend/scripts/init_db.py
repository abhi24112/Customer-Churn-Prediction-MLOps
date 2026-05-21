import os
import sys
import logging

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.postgres import get_connection

logging.basicConfig(level=logging.INFO)

def init_db():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS churn_raw (
        customerID TEXT PRIMARY KEY,
        gender TEXT,
        SeniorCitizen TEXT,
        Partner TEXT,
        Dependents TEXT,
        tenure TEXT,
        PhoneService TEXT,
        MultipleLines TEXT,
        InternetService TEXT,
        OnlineSecurity TEXT,
        OnlineBackup TEXT,
        DeviceProtection TEXT,
        TechSupport TEXT,
        StreamingTV TEXT,
        StreamingMovies TEXT,
        Contract TEXT,
        PaperlessBilling TEXT,
        PaymentMethod TEXT,
        MonthlyCharges TEXT,
        TotalCharges TEXT,
        Churn TEXT
    );
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_query)
            conn.commit()
        logging.info("✅ Database initialized successfully (table 'churn_raw' created/exists).")
    except Exception as e:
        logging.error(f"❌ Failed to initialize database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_db()
