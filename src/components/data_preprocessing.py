import pandas as pd
import numpy as np
import logging


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Target column
TARGET_COL = "Churn"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven features before encoding.
    All new columns are numeric so they pass straight through the pipeline.
    """
    feature_df = df.copy()

    required_cols = ["MonthlyCharges", "tenure", "Contract"]
    missing_required = [c for c in required_cols if c not in feature_df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for feature engineering: {missing_required}")


    services = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    existing_services = [c for c in services if c in feature_df.columns]
    missing_services = [c for c in services if c not in feature_df.columns]
    if missing_services:
        logging.warning(f"Missing service columns (skipping them): {missing_services}")

    # Adding total services
    if existing_services:
        feature_df['TotalServices'] = (feature_df[existing_services] == 'Yes').sum(axis=1)
    else:
        feature_df['TotalServices'] = 0

    # Charges Per Service Charge (avoid divide-by-zero)
    denom = feature_df['TotalServices'].replace(0, 1)
    feature_df['ChargePerService'] = feature_df["MonthlyCharges"] / denom

    # New customer or not
    feature_df['IsNewCustomer'] = (feature_df['tenure'] <= 3).map({True: "Yes", False: "No"})


    # long term contract
    feature_df['IsLongTerm'] = (feature_df['Contract'] != 'Month-to-month').map({True: 'Yes', False: 'No'})

    # Tenure Grouping
    feature_df['TenureGroup'] = pd.cut(
        feature_df['tenure'],
        bins=[-1, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'],
        right=True   
    )
    feature_df['TenureGroup'] = feature_df['TenureGroup'].astype(str)
    logging.info("Feature added: TenureGroup")

    # changing all Object type into category
    cat_col = feature_df.select_dtypes(include=["object"]).columns.to_list()
    for col in cat_col:
        feature_df[col] = feature_df[col].astype('category')

    # samples
    new_features = ['TotalServices', 'ChargePerService', 'IsNewCustomer', 'IsLongTerm', 'TenureGroup']

    for f in new_features:
        logging.info(f'   - {f}: {feature_df[f].dtype} | sample: {feature_df[f].head(3).tolist()}')

    return feature_df


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------
def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # TotalCharges: data type fixing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Filling TotalCharges Nan values if any 
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # removeing duplicate data
    dup_count = int(df.duplicated().sum())
    if dup_count:
        logging.info(f"Dropping {dup_count} duplicate rows")
        df = df.drop_duplicates().reset_index(drop=True)

    logging.info(f"Shape after Data cleaning: {df.shape}")

    return df


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Encode and engineer features for model training or inference.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as loaded by data_loader.py.
    fit : bool
        True  → fit a new ColumnTransformer (training time).
        False → load a saved transformer and apply it (inference time).
    transformer_path : str
        Where to save / load the fitted ColumnTransformer.

    Returns
    -------
    (processed_df, transformer)
        processed_df  – fully numeric DataFrame ready for train/test split.
        transformer   – fitted ColumnTransformer (None when fit=False).
    """
    if df is None or df.empty:
        raise ValueError("preprocess() received an empty or None DataFrame")

    try:
        # --- 0. Data Cleaning ---
        df = data_cleaning(df)

        # --- 1. Feature engineering (adds numeric columns) ---
        df = engineer_features(df)

        # --- 2. Encode target ---
        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1}).astype("int8")
            logging.info(f"Target encoded: {TARGET_COL}")


        logging.info(
            f"Preprocessing completed. Shape: {df.shape}."
)
        return df

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise ValueError(f"Error in preprocessing: {e}")