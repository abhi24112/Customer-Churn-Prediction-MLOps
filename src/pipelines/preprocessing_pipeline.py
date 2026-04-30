import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from src.components.data_ingestion import reading_files
from src.components.data_loader import loading_files
from src.components.data_preprocessing import preprocess


def run_preprocessing():
    logging.info("Started preprocessing pipeline")
    # Step 1: get file
    file_path = reading_files()

    # Step 2: load data
    df = loading_files(file_path)

    # Ensure df is not None before preprocessing
    if df is None:
        logging.error("Failed to load data: loading_files returned None")
        raise ValueError("Failed to load data: loading_files returned None")

    # Step 3: preprocess
    df = preprocess(df)

    # Step 4: save output (IMPORTANT for DVC)
    os.makedirs("src/data/processed", exist_ok=True)
    df.to_csv("src/data/processed/processed.csv", index=False)

    logging.info("Preprocessing pipeline is completed and processed.csv is saved")
    print("✅ Preprocessing completed and saved")


if __name__ == "__main__":
    run_preprocessing()