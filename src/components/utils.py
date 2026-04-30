import yaml
import os

def load_config(config_path: str = os.path.join(os.getcwd(), "config/config.yaml"), 
                param_path: str = os.path.join(os.getcwd(), "params.yaml")):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            mlflow_config = config['mlflow']
        
        with open(param_path, "r") as file:
            params = yaml.safe_load(file)
            data_split_config = params['data']
            models_config = params["model"]

            return mlflow_config, data_split_config, models_config

    except Exception as e:
        raise ValueError(f"Erron in loading or reading config file : {e}")