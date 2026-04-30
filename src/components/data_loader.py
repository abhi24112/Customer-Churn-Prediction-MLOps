import pandas as pd
import logging
from typing import Optional

def loading_files(file_path: str) -> Optional[pd.DataFrame]:
    """
    This function is used to load the csv or xlsx file as the dataframe.
    """

    try:
        if file_path.endswith(".csv"):

            df = pd.read_csv(
                file_path
            )

            df = df.drop(columns=["customerID"])
            logging.info(f"Shape: {df.shape}")


        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            logging.info(f"Shape: {df.shape}")
            
        else:
            raise ValueError("File must be in .csv or .xlsx format")
        
        logging.info(f"Data loaded successfully: {file_path}")
        return df

    except Exception as e:
        logging.error(f"Error loading the files: {file_path}")
        raise ValueError(f"Exception: {e} occurs while running the data loading process")
