import logging
import pandas as pd
from typing import Dict

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

MODEL_MAP = {
    "xgboost": XGBClassifier,
    "catboost": CatBoostClassifier,
    "lightgbm": LGBMClassifier,
}


def model_training(
    model_name: str,
    params: Dict,
    model_status: bool,
    x_train: pd.DataFrame,
    y_train: pd.Series,
):

    logging.info(f"Creating model: {model_name}")

    try:
        if model_name not in MODEL_MAP or not model_status:
            logging.error(f"Model '{model_name}' not found or disabled.")
            raise KeyError(f"'{model_name}' not found in MODEL_MAP or model_status=False")


        model_params = dict(params) if params is not None else {}

        # Identify categorical columns (keep them categorical; do NOT integer-encode)
        cat_cols = x_train.select_dtypes(include=["category"]).columns.tolist()

        if model_name == "xgboost":
            obj_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
            if obj_cols:
                raise ValueError(
                    f"XGBoost received object dtype columns {obj_cols}. "
                    f"Convert them to pandas 'category' dtype before training (enable_categorical=True)."
                )

            # Ensure categorical support is enabled
            model_params.setdefault("enable_categorical", True)

            if "scale_pos_weight" not in model_params:
                # Fallback: compute from training labels and warn
                neg = int((y_train == 0).sum())
                pos = int((y_train == 1).sum())
                fallback_spw = neg / pos if pos > 0 else 1.0
                model_params["scale_pos_weight"] = fallback_spw
                logging.warning(
                    f"scale_pos_weight not in params — falling back to raw ratio "
                    f"{fallback_spw:.2f}. Pass it explicitly for tuned runs."
                )
            else:
                logging.info(
                    f"XGBoost scale_pos_weight: {float(model_params['scale_pos_weight']):.2f}"
                )
            model_cls = MODEL_MAP[model_name]
            model = model_cls(**model_params)
            model.fit(x_train, y_train)

        elif model_name == "catboost":
            model_cls = MODEL_MAP[model_name]
            model = model_cls(**model_params)

            cat_feature_indices = [int(x_train.columns.get_indexer([c])[0]) for c in cat_cols] # type: ignore
            model.fit(x_train, y_train, cat_features=cat_feature_indices)

        elif model_name == "lightgbm":
            if cat_cols:
                # LightGBM works best when categoricals are pandas 'category'
                obj_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
                if obj_cols:
                    logging.warning(
                        f"LightGBM received object dtype columns {obj_cols}; casting to 'category' dtype."
                    )
                    x_train = x_train.copy()
                    for c in obj_cols:
                        x_train[c] = x_train[c].astype("category")

            model_cls = MODEL_MAP[model_name]
            model = model_cls(**model_params)
            model.fit(x_train, y_train)

        else:
            raise KeyError(f"Unsupported model_name: {model_name}")

        logging.info(f"{model_name} trained successfully.")
        return model

    except Exception as e:
        logging.error(f"Error training {model_name}: {e}")
        raise ValueError(f"Error in training the model: {e}")
