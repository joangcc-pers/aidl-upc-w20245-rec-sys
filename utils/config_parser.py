import yaml
import os

def parse_config(config_path="experiments/config.yaml", model_name=None, tasks=None):
    """
    Parses the configuration file and validates the structure.

    Parameters:
    - config_path (str): Path to the configuration YAML file.
    - model_name (str): Name of the model to extract its specific configuration.
    - tasks (list): List of tasks in order (e.g., ["preprocess", "train", "evaluate"]).

    Returns:
    - config (dict): Parsed configuration for the entire pipeline.
    - model_config (dict): Configuration specific to the chosen model.
    - tasks (list): Validated task list.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load the configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Validate tasks order
    allowed_tasks = ["preprocess", "train", "evaluate", "visualize"]
    if tasks:
        for task in tasks:
            if task not in allowed_tasks:
                raise ValueError(f"Invalid task: {task}. Allowed tasks: {allowed_tasks}")
        # Ensure correct task order
        tasks = sorted(tasks, key=allowed_tasks.index)
    
    # Validate model_name and extract model-specific config
    if model_name:
        if "models" not in config or model_name not in config["models"]:
            raise ValueError(f"Model '{model_name}' not found in configuration file.")
        model_config = config["models"][model_name]
    else:
        model_config = None
    
    return config, model_config, tasks