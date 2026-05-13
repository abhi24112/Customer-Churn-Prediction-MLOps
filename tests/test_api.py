import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    payload = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
    else:
        # If it fails due to DB/Env, we still want to see the error for debugging
        print(f"Prediction failed with status {response.status_code}: {response.text}")
        assert False, f"API returned {response.status_code}"

if __name__ == "__main__":
    pytest.main([__file__])
