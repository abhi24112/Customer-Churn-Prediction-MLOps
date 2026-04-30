import mlflow
import mlflow.sklearn as mlflow_sklearn
import logging
import os


# Import all scripts for model
from src.components.data_ingestion import reading_files
from src.components.data_loader import loading_files
from src.components.data_preprocessing import preprocess
from src.components.data_splitting import data_splitting
from src.components.model_training import model_training
from src.components.evaluate import evaluate_model
from src.components.model_saving import save_model
from src.components.utils import load_config


# Loading cofig.yaml
mlflow_config, data_split_config, models_config = load_config()

# logging setup
log_dir = ".logs/model_log"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename = os.path.join(log_dir, "tracking.log"),
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Setting up MLflow for tracking and Logging model, and artifacts
mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
mlflow.set_experiment(experiment_name=mlflow_config["experiment_name"])

def main():
    logging.info("Model training is Started...")
    try:
        # find the .csv or .xlsx file
        file_path = reading_files()
        logging.info(f"Data file is fount at: {file_path}")

        # Loading file in dataframe
        df = loading_files(file_path=file_path)
        if df is None:
            logging.error("Loaded dataframe is None")
            raise ValueError("Failed to load data: dataframe is None")
        logging.info(f"DataFrame is loaded")

        # Data Preprocessing
        logging.info("Preprocessing dataframe for model training")
        df = preprocess(df)
        logging.info("Data is now preprocessed.")


        # splitting the data in training and testing
        if df is None:
            logging.error("preprocessed dataframe is None")
            raise ValueError("Failed to load preprocessed data: dataframe is None")
        
        x_train, x_test, y_train, y_test = data_splitting(
            df = df, 
            test_size=data_split_config['test_size'],
            random_state=data_split_config['random_state']
        ) 

        # Training the model(s)
        for name, config in models_config.items():
            if config.get("model_status"):
                with mlflow.start_run(  
                    run_name=f"{mlflow_config['run_name']}_{name}"
                ):
                    logging.info(f"======================={name}=====================")
                    model = model_training(
                        model_name=name,
                        params=config.get('params', {}),
                        model_status=bool(config.get('model_status')),
                        x_train=x_train,
                        y_train=y_train
                    )

                    logging.info(f"{name} trained successfully")

                    # Evaluating the models
                    metrics, plots = evaluate_model(
                        model_name=name,
                        run_name=mlflow_config['run_name'],
                        model=model,
                        x_test=x_test,
                        y_test=y_test,
                        threshold=0.5,  
                        optimize_threshold=False,
                    )

                    # Saving the model
                    saved = save_model(
                        model_name=name,
                        run_name=mlflow_config['run_name'],
                        model=model,
                        f1_score=metrics["f1"]
                    )

                    if saved:
                        logging.info(f"{mlflow_config['run_name']}_{name} dumped successfully.")

                    # logging the model
                    mlflow_sklearn.log_model(model, name="model")
                    logging.info(f"{mlflow_config['run_name']}_{name} logged successfully in mlflow.")

                    # logging metrics
                    mlflow.log_params(config['params'])
                    mlflow.log_metrics(
                        {
                            "accuracy": metrics["accuracy"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "f1_score": metrics["f1"],
                            "pr_auc": metrics["pr_auc"],
                            "threshold": metrics["threshold"],
                        }
                    )  # type: ignore
                    logging.info(f"{mlflow_config['run_name']}_{name} parameters and metrics logged successfully in mlflow.")

                    print(f"{mlflow_config['run_name']}_{name} Accuracy: {metrics['accuracy']}")
                    print(f"{mlflow_config['run_name']}_{name} Precision: {metrics['precision']}")
                    print(f"{mlflow_config['run_name']}_{name} Recall: {metrics['recall']}")
                    print(f"{mlflow_config['run_name']}_{name} F1_Score: {metrics['f1']}")
                    print(f"{mlflow_config['run_name']}_{name} PR_AUC: {metrics['pr_auc']}")
                    print(f"{mlflow_config['run_name']}_{name} Threshold: {metrics['threshold']}")

                    # logging the plot using artifacts
                    for _, plot_path in plots.items():
                        mlflow.log_artifact(plot_path, artifact_path="plots")
                    logging.info(f"{mlflow_config['run_name']}_{name} confusion matrix and pr_curve logged successfully in mlflow.")

        logging.info("🎉 Pipeline Execution Completed Successfully!")            

    except Exception as e:
        logging.error(f"Error Occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()