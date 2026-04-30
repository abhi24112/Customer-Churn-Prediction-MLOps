import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional
import logging

def data_splitting(test_size:int, random_state: int, df:Optional[pd.DataFrame] = None, x_train_exp:Optional[pd.DataFrame] = None, y_train_exp:Optional[pd.DataFrame] = None):
    try:
        if df is not None:
            if df.empty:
                logging.error("Data frame is empty")
                raise ValueError("Data Frame is empty")
            logging.info("Splitting the data in Training and Testing")
            X = df.drop("Churn", axis = 1)
            y = df["Churn"]

            x_train, x_test, y_train, y_test = train_test_split(
                X, 
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )

            return x_train, x_test, y_train, y_test
        
        elif x_train_exp is not None and y_train_exp is not None:

            x_train, x_test, y_train, y_test = train_test_split(
                x_train_exp, 
                y_train_exp, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y_train_exp
            )

            return x_train, x_test, y_train, y_test
    
        else:
            raise ValueError("Either df or both x_train and y_train must be provided.")
                    
    except Exception as e:
        raise ValueError(f"Error in Splitting the data: {e}")
    