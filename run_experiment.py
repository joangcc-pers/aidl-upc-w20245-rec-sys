import argparse
from scripts.preprocess import preprocess_data
from scripts.train import train_model
from utils.config_parser import parse_config

def main():
    parser = argparse.ArgumentParser(description="Run RecSys experiments.")
    parser.add_argument("--config", type=str, default="experiments/config.yaml", help="Path to config file")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to run")
    parser.add_argument(
        "--task", 
        nargs="+", 
        choices=["preprocess", "train", "validation", "test", "visualize"], 
        required=True, 
        help="Tasks to perform in the correct order (e.g., preprocess train evaluate visualize)"
    )
    
    args = parser.parse_args()
    
    ''' Override values - handy for debugging
    args = argparse.Namespace(
        config="experiments/config.yaml",  # Set your specific config path
        experiment="experiment_4",  # Set your specific experiment name
        task=["train"]  # Set your specific tasks
    '''
    
    # Ensure tasks are in the correct order
    valid_order = ["preprocess", "train", "test"]
    provided_order = args.task

    if len(provided_order) > 1:
        # Ensure provided tasks follow the valid order
        indices = [valid_order.index(task) for task in provided_order]
        if indices != sorted(indices):
            raise ValueError(
                f"Invalid task order: {provided_order}. Tasks must follow this order: {valid_order}."
            )
    
    # Load configuration
    config, _, _ = parse_config(config_path=args.config)
    experiment_config = config["experiments"].get(args.experiment)
    if not experiment_config:
        raise ValueError(f"Experiment '{args.experiment}' not found in config file.")

    # TODO: temporary deactivation of get_model, until we have any model built. Then, activate again.    
    # model_class = get_model(experiment_config["model_name"])
    # model = model_class(experiment_config["model_params"])

    # Process tasks in order
    for task in provided_order:
        #TODO: delete after creating train, evaluate and visualize
        if task == "preprocess":
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
            print(f"Training model '{experiment_config['model_name']}'...")
            #TODO: retrieve dataset object when script hasn't run "process" but it has been previously saved
            print("Training parameters:")
            for key, value in experiment_config["model_params"].items():
                print(f"  {key}: {value}")
            train_model(
                model_name=experiment_config["model_name"],
                model_params=experiment_config.get("model_params", {}),
                output_folder_artifacts=experiment_config["data_params"]["output_folder_artifacts"],
                top_k=experiment_config["evaluation"]["top_k"],
                resume = None
                )
        elif task == "test":
            print(f"Testing model '{experiment_config['model_name']}'...")
            print("Testing parameters:")
            for key, value in experiment_config["test_params"].items():
                print(f"  {key}: {value}")
            train_model(
                model_name=experiment_config["model_name"],
                model_params=experiment_config.get("model_params", {}),
                output_folder_artifacts=experiment_config["data_params"]["output_folder_artifacts"],
                top_k=experiment_config["evaluation"]["top_k"],
                task = task,
                resume = None,
                best_checkpoint_path = experiment_config["test_params"]["best_checkpoint_path"]
            )

        

if __name__ == "__main__":
    main()