import argparse
import itertools
import numpy as np
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from scripts.preprocess import preprocess_data
from scripts.train import train_model
from utils.config_parser import parse_config
import os

def main():
    parser = argparse.ArgumentParser(description="Run experiments with different models and hyperparameters.")
    parser.add_argument('--model', type=str, required=True, choices=["graph_with_embeddings", "graph_with_embeddings_and_attention", "graph_with_embeddings_and_attentional_aggregation", "graph_with_encoding_and_attentional_aggregation"], help="Model to use for the experiment")
    parser.add_argument(
        "--task", 
        nargs="+", 
        choices=["preprocess", "train"], 
        required=True, 
        help="Tasks to perform in the correct order (e.g., preprocess train evaluate visualize)"
    )
    parser.add_argument('--log_dir', type=str, default="experiments", help="Directory to save TensorBoard logs")
    parser.add_argument('--force_rerun_train', type=str, required=True, choices=["no", "all", "rerun_list", "resume_only"], help="Force rerun training scenarios: no, all, rerun_list_only")
    parser.add_argument('--resume', type=str, required=True, choices=["yes","no"], help="Resume training from initialized experiments")
    args = parser.parse_args()

    rerun_list = []
    if args.force_rerun_train == "rerun_list":
        with open("experiments/rerun_list.yaml", 'r') as file:
            rerun_list = yaml.safe_load(file)

    for task in args.task:
        if task not in ["preprocess", "train"]:
            raise ValueError(f"Invalid task: {task}. Must be one of: preprocess, train")
    valid_order = ["preprocess", "train"]
    provided_order = args.task
    if len(provided_order) > 1:
        indices = [valid_order.index(task) for task in provided_order]
        if indices != sorted(indices):
            raise ValueError(f"Invalid task order: {provided_order}. Tasks must follow this order: {valid_order}")
    
    config, _, _ = parse_config(config_path="experiments/config_hyp.yaml")
    experiment_config = config["experiments"].get(f"model_{args.model}")


    for task in provided_order:

        if task == "preprocess":
            # Preprocess
            print(f"Preprocessing data for '{experiment_config['model_name']}'...")
            preprocessing_config = experiment_config.get("preprocessing", {})
            print("Preprocessing config:")
            for key, value in preprocessing_config.items():
                print(f"  {key}: {value}")
            preprocess_data(
                input_folder_path=experiment_config["data_params"]["input_folder_path"],
                output_folder_artifacts=experiment_config["data_params"]["output_folder_artifacts"],
                preprocessing_params=preprocessing_config,  # Pass preprocessing params
                model_name=experiment_config["model_name"],
            )
        elif task == "train":
            # Define parameters for hyperparameter search
            weight_decay_values = [1e-4, 1e-5, 1e-6]
            dropout_rate_values = [0.0, 0.2, 0.5]
            learning_rate_values = [1e-3, 1e-4, 1e-5]

            # Generate all possible combinations of hyperparameters
            param_combinations = list(itertools.product(weight_decay_values, dropout_rate_values, learning_rate_values))

            # Retreive fixed model parameters
            model_params = experiment_config["model_params"]

            # Define output folder for artifacts
            output_folder_artifacts = experiment_config["data_params"]["output_folder_artifacts"]

            # Iterate over hyperparameter combinations
            for weight_decay, dropout_rate, learning_rate in param_combinations:
                # Create a name for the experiment based on the hyperparameters
                experiment_hyp_combinat_name = f"{args.model}_wd{weight_decay}_dr{dropout_rate}_lr{learning_rate}"
                model_output_path = os.path.join(output_folder_artifacts, f"{experiment_hyp_combinat_name}","trained_model.pth")
                
                if args.force_rerun_train == "no" and os.path.exists(model_output_path):
                    print(f"Skipping {experiment_hyp_combinat_name} as it already exists.")
                    continue
                elif args.force_rerun_train == "rerun_list_only" and experiment_hyp_combinat_name not in rerun_list:
                    print(f"Skipping {experiment_hyp_combinat_name} as it is not in the rerun list.")
                    continue

                print(f"Training {args.model} with weight_decay={weight_decay}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
                
                # Update model parameters with the hyperparameters
                model_params.update({
                    "weight_decay": weight_decay,
                    "dropout_rate": dropout_rate,
                    "lr": learning_rate
                })
                
                # Train model
                train_model(args.model, model_params, output_folder_artifacts, top_k=experiment_config["evaluation"]["top_k"], experiment_hyp_combinat_name=experiment_hyp_combinat_name, resume=args.resume)
            

if __name__ == "__main__":
    main()