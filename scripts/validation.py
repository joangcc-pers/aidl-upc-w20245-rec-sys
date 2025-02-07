from scripts.evaluate_scripts.evaluate_sr_gnn import evaluate_sr_gnn
from models.sr_gnn_attn import SR_GNN_attn
from models.sr_gnn import SR_GNN
import os
import json
import torch


def evaluate(model_name, output_folder_artifacts, model_params):
    print(f"Evaluating {model_name} in validation split...")

    if model_name in ["graph_with_embeddings","graph_with_embeddings_and_attention"]:
    # Combine the directory and the file name
        file_path = os.path.join(output_folder_artifacts, "num_values_for_node_embedding.json")

        # Open and load the JSON file
        with open(file_path, "r") as f:
            num_values_for_node_embedding = json.load(f)
        #TODO: input model object
        # Initialize the model with the same architecture
        if model_name == "graph_with_embeddings":
            model = SR_GNN(hidden_dim=model_params["hidden_dim"],
                            num_iterations=model_params["num_iterations"],
                            num_items=num_values_for_node_embedding["num_items"],
                            embedding_dim=model_params["embedding_dim"],
                            num_categories=num_values_for_node_embedding["num_categories"],
                            num_sub_categories=num_values_for_node_embedding["num_sub_categories"],
                            num_elements=num_values_for_node_embedding["num_elements"],
                            num_brands=num_values_for_node_embedding["num_brands"]
                            )

        if model_name == "graph_with_embeddings_and_attention":
            model = SR_GNN_attn(hidden_dim=model_params["hidden_dim"],
                            num_iterations=model_params["num_iterations"],
                            num_items=num_values_for_node_embedding["num_items"],
                            embedding_dim=model_params["embedding_dim"],
                            num_categories=num_values_for_node_embedding["num_categories"],
                            num_sub_categories=num_values_for_node_embedding["num_sub_categories"],
                            num_elements=num_values_for_node_embedding["num_elements"],
                            num_brands=num_values_for_node_embedding["num_brands"]
                            )

        # Load the saved weights
        model.load_state_dict(torch.load(output_folder_artifacts+"trained_model.pth", weights_only=False))
        split_loader = torch.load(output_folder_artifacts+"val_dataset.pth", weights_only=False)

        evaluate_sr_gnn(model, split_loader, top_k_values=[5, 10])
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

