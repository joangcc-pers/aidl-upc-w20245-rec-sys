
from scripts.train_scripts.train_sr_gnn import train_sr_gnn
from scripts.train_scripts.train_sr_gnn_attn import train_sr_gnn_attn

def train_model(model_name, dataloader, training_params, output_folder_artifacts):
    # This function calls all needed methods of the corresponding architecture Class, and can do it differently depending on the architecture.
    print(f"Training {model_name}...")

    # Define preprocessing pipeline for each architecture
    if model_name in {"sr_gnn","sr_gnn_test_mockup","graph_with_embeddings"}:
        train_sr_gnn(dataloader=dataloader, training_params=training_params, output_folder_artifacts=output_folder_artifacts)
    if model_name in {"sr_gnn","sr_gnn_test_mockup","graph_with_embeddings_and_attention"}:
        train_sr_gnn_attn(dataloader=dataloader, training_params=training_params, output_folder_artifacts=output_folder_artifacts)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")