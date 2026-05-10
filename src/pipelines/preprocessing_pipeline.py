import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from src.components import (
    reading_files,
    loading_files,
    preprocess
)


def run_preprocessing():
    logging.info("Started preprocessing pipeline")

    # Step 1: get the snapshot file path from the ingestion helper
    file_path = reading_files()

    # Step 2: load data from the snapshot CSV
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