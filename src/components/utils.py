import yaml
import os

def load_config(path: str = os.path.join(os.getcwd(), "config/config.yaml")):
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
            mlflow_config = config['mlflow']
            data_split_config = config['data']
            models_config = config["model"]

            return mlflow_config, data_split_config, models_config

    except Exception as e:
        raise ValueError(f"Erron in loading or reading config file : {e}")