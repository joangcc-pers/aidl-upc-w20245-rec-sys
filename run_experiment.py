import argparse
from utils.data_cleaner import clean_data
from models.model_registry import get_model
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.visualize import visualize_results
from utils.config_parser import parse_config

def main():
    parser = argparse.ArgumentParser(description="Run RecSys experiments.")
    parser.add_argument("--config", type=str, default="experiments/config.yaml", help="Path to config file")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to run")
    parser.add_argument(
        "--task", 
        nargs="+", 
        choices=["clean", "train", "evaluate", "visualize"], 
        required=True, 
        help="Tasks to perform in the correct order (e.g., clean train evaluate visualize)"
    )
    args = parser.parse_args()

    # Ensure tasks are in the correct order
    valid_order = ["clean", "train", "evaluate", "visualize"]
    provided_order = args.task
    for i, task in enumerate(provided_order):
        if valid_order.index(task) != i:
            raise ValueError(
                f"Invalid task order: {provided_order}. Tasks must follow this order: {valid_order}."
            )
    
    # Load configuration
    config = parse_config(args.config)
    experiment_config = config["experiments"].get(args.experiment)
    if not experiment_config:
        raise ValueError(f"Experiment '{args.experiment}' not found in config file.")
    
    model_class = get_model(experiment_config["model_name"])
    model = model_class(experiment_config["model_params"])

    # Process tasks in order
    for task in provided_order:
        if task == "clean":
            clean_data(
                input_path=experiment_config["data_params"]["dataset_path"],
                output_path=experiment_config["data_params"]["cleaned_path"],
                cleaning_params=experiment_config["data_cleaning"],
                model_name=experiment_config["model_name"]
            )
        elif task == "train":
            print(f"Training model '{experiment_config['model_name']}'...")
            train_model(model, experiment_config)
        elif task == "evaluate":
            print(f"Evaluating model '{experiment_config['model_name']}'...")
            evaluate_model(model, experiment_config)
        elif task == "visualize":
            print(f"Visualizing results for '{experiment_config['model_name']}'...")
            visualize_results(experiment_config)
