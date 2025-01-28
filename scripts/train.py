
from scripts.train_sr_gnn import train_sr_gnn

def train_model(model_name, dataloader, training_params):
    # This function calls all needed methods of the corresponding architecture Class, and can do it differently depending on the architecture.
    print(f"Training {model_name}...")

    # Define preprocessing pipeline for each architecture
    if model_name in {"sr_gnn","sr_gnn_test_mockup"}:
        train_sr_gnn(dataloader=dataloader, training_params=training_params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")