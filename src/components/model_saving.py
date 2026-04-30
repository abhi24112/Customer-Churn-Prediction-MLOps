import joblib as jb
import logging
import json
import os

def save_model(model_name:str, run_name:str, model, f1_score):

    logging.info("Checking the threshold for saving the model")
    if f1_score >= 0.60:
        logging.info("Threshold met. Saving the model...")
        model_dir = r"models/"
        logging.info("Creating the model directory")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{run_name}_{model_name}.jbl")
        jb.dump(model, model_path)

        logging.info(f"✅ Model Successfully Saved at {model_path}")

        return True
    else:
        logging.warning(f"Model F1 {f1_score*100:.2f}% below threshold. Not saving {run_name}_{model_name}")
        return False
    
def save_metrics(metrics:dict):
    os.makedirs("reports", exist_ok=True)
    
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics,f, indent=4)

    return True










