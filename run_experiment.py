import argparse
from models.model_registry import get_model
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.visualize import visualize_results
from utils.config_parser import parse_config

def main():
    parser = argparse.ArgumentParser(description="Run RecSys experiments.")
    parser.add_argument("--config", type=str, default="experiments/config.yaml", help="Path to config file")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to run")
    parser.add_argument("--task", type=str, choices=["train", "evaluate", "visualize"], required=True, help="Task to perform")
    args = parser.parse_args()
    
    # Load configuration
    config = parse_config(args.config)
    experiment_config = config["experiments"].get(args.experiment)
    if not experiment_config:
        raise ValueError(f"Experiment '{args.experiment}' not found in config file.")
    
    # Select model
    model_class = get_model(experiment_config["model_name"])
    model = model_class(experiment_config["model_params"])
    
    # Execute task
    if args.task == "train":
        train_model(model, experiment_config)
    elif args.task == "evaluate":
        evaluate_model(model, experiment_config)
    elif args.task == "visualize":
        visualize_results(experiment_config)

if __name__ == "__main__":
    main()
