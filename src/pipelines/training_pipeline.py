import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
import logging

# Import all scripts for model
from src.components.data_splitting import data_splitting
from src.components.model_training import model_training
from src.components.evaluate import evaluate_model
from src.components.model_saving import save_model, save_metrics
from src.components.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_training():
    logging.info("Starting training stage...")
    try:
        mlflow_config, data_split_config, models_config = load_config()

        # Load processed data
        df = pd.read_csv("src/data/processed/processed.csv")

        
        
        # Splitting
        x_train, x_test, y_train, y_test = data_splitting(
            df = df, 
            test_size=data_split_config['test_size'],
            random_state=data_split_config['random_state']
        ) 
        
        # Object to Category
        object_cols = x_train.select_dtypes(include=['object']).columns
        for col in object_cols:
            x_train[col] = x_train[col].astype('category')
            x_test[col] = x_test[col].astype('category')


        # Setting up MLflow for tracking and Logging model, and artifacts
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(experiment_name=mlflow_config["experiment_name"])

        # Training the model(s)
        for name, config in models_config.items():
            if config.get("model_status"):
                with mlflow.start_run(  
                    run_name=f"{mlflow_config['run_name']}_{name}"
                ):
                    logging.info(f"====================={name}=====================")
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

                    # Saving teh metrics
                    mt = save_metrics(metrics)

                    if saved and mt:
                        logging.info(f"{mlflow_config['run_name']}_{name} dumped successfully.")
                        logging.info(f"Model Metrics are successfully dumped in metrics.json")

                    # logging the model
                    mlflow_sklearn.log_model(model, name="model")
                    logging.info(f"{mlflow_config['run_name']}_{name} logged successfully in mlflow.")

                    # logging metrics
                    if config.get("params"):
                        mlflow.log_params(config["params"])
                    mlflow.log_metrics(metrics)

                    # logging the plot using artifacts
                    for _, plot_path in plots.items():
                        mlflow.log_artifact(plot_path, artifact_path="plots")
    
                    logging.info(f"{mlflow_config['run_name']}_{name} parameters and metrics logged successfully in mlflow.")

        logging.info("🎉 Pipeline Execution Completed Successfully!")            

    except Exception as e:
        logging.error(f"Error Occurred: {e}", exc_info=True)

if __name__ == "__main__":
    run_training()