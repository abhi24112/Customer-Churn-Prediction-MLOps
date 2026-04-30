import os
import logging
from typing import Any, Dict, List, Tuple


import mlflow
import optuna
from optuna.samplers import TPESampler

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from src.components.data_ingestion import reading_files
from src.components.data_loader import loading_files
from src.components.data_preprocessing import preprocess
from src.components.data_splitting import data_splitting
from src.components.evaluate import evaluate_model
from src.components.utils import load_config


mlflow_config, data_split_config, models_config = load_config()

log_dir = ".logs/model_tuning"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir, "tracking.log"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
mlflow.set_experiment(mlflow_config["tuning_experiment_name"])

MAX_EVALS = 200


def _get_single_hpt_model(models_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    enabled = [(name, cfg) for name, cfg in models_cfg.items() if cfg.get("hpt_enabled") is True]
    if not enabled:
        raise ValueError(
            "No model has hpt_enabled: true in config.yaml. "
            "Set exactly one of xgboost/catboost/lightgbm to true."
        )
    if len(enabled) > 1:
        enabled_names = [n for n, _ in enabled]
        raise ValueError(
            f"Multiple models have hpt_enabled: true: {enabled_names}. "
            "Set exactly one model to true for hyperparameter tuning."
        )
    return enabled[0]


def _xgb_search_space(trial: optuna.Trial, raw_ratio: float, random_state: int) -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        "scale_pos_weight": trial.suggest_categorical(
            "scale_pos_weight",
            [1.0, 1.5, 2.0, 3.0, 5.0, float(round(raw_ratio, 2))],
        ),
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": random_state,
        "enable_categorical": True,
    }


def _cat_search_space(trial: optuna.Trial, random_state: int) -> Dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_seed": random_state,
        "thread_count": -1,
        "verbose": False,
        "auto_class_weights": "Balanced",
    }


def _lgbm_search_space(trial: optuna.Trial, raw_ratio: float, random_state: int) -> Dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": trial.suggest_int("n_estimators", 400, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_categorical(
            "scale_pos_weight",
            [1.0, 1.5, 2.0, 3.0, 5.0, float(round(raw_ratio, 2))],
        ),
        "random_state": random_state,
        "n_jobs": -1,
    }


def tuning(max_evals: int = MAX_EVALS):
    logging.info("Hyperparameter tuning started...")

    model_name, model_cfg = _get_single_hpt_model(models_config)
    base_params = dict(model_cfg.get("params", {}))

    file_path = reading_files()
    df = loading_files(file_path=file_path)
    if df is None:
        raise ValueError("Failed to load data")

    df = preprocess(df)

    x_train_full, _x_test, y_train_full, _y_test = data_splitting(
        df=df,
        test_size=data_split_config["test_size"],
        random_state=data_split_config["random_state"],
    )

    x_train, x_val, y_train, y_val = data_splitting(
        x_train_exp=x_train_full,
        y_train_exp=y_train_full,
        test_size=data_split_config["test_size"],
        random_state=data_split_config["random_state"],
    )

    # Class imbalance ratio
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    raw_ratio = neg / pos if pos != 0 else 1.0

    # Keep categoricals as categories (no integer encoding)
    obj_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        logging.warning(f"Object columns found {obj_cols}; casting to pandas 'category'.")
        x_train = x_train.copy()
        x_val = x_val.copy()
        for c in obj_cols:
            x_train[c] = x_train[c].astype("category")
            x_val[c] = x_val[c].astype("category")

    cat_cols = x_train.select_dtypes(include=["category"]).columns.tolist()
    cat_feature_indices: List[int] = [i for i, col in enumerate(x_train.columns) if col in cat_cols]

    def objective(trial: optuna.Trial) -> float:
        with mlflow.start_run(nested=True, run_name="optuna_trial"):
            mlflow.set_tag("stage", "hpt")
            mlflow.set_tag("model", model_name)

            if model_name == "xgboost":
                trial_params = _xgb_search_space(
                    trial,
                    raw_ratio=raw_ratio,
                    random_state=data_split_config["random_state"],
                )
                params = {**base_params, **trial_params}
                model = XGBClassifier(**params)
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

            elif model_name == "catboost":
                trial_params = _cat_search_space(trial, random_state=data_split_config["random_state"])
                params = {**base_params, **trial_params}
                model = CatBoostClassifier(**params)
                model.fit(
                    x_train,
                    y_train,
                    cat_features=cat_feature_indices,
                    eval_set=(x_val, y_val),
                )

            elif model_name == "lightgbm":
                trial_params = _lgbm_search_space(
                    trial,
                    raw_ratio=raw_ratio,
                    random_state=data_split_config["random_state"],
                )
                params = {**base_params, **trial_params}
                model = LGBMClassifier(**params)
                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    categorical_feature=cat_cols if cat_cols else "auto",
                )

            else:
                raise KeyError(f"Unsupported model for tuning: {model_name}")

            metrics, plots = evaluate_model(
                model_name=model_name,
                run_name="hpt",
                model=model,
                x_test=x_val,
                y_test=y_val,
                threshold=0.5,
                optimize_threshold=True,
            )

            f1 = float(metrics["f1"])
            loss = 1.0 - f1

            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "val_accuracy": metrics["accuracy"],
                    "val_precision": metrics["precision"],
                    "val_recall": metrics["recall"],
                    "val_f1": f1,
                    "val_pr_auc": metrics["pr_auc"],
                    "val_threshold": metrics["threshold"],
                }
            )

            for _name, plot_path in plots.items():
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, artifact_path="plots")

        return loss

    with mlflow.start_run(run_name=f"{model_name}_optuna_tuning"):
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=max_evals)

        best_params = study.best_params
        mlflow.log_dict(best_params, f"best_{model_name}_optuna_params.json")
        logging.info(f"Best {model_name} params: {best_params}")
        logging.info("Hyperparameter tuning completed successfully")


if __name__ == "__main__":
    tuning(MAX_EVALS)

