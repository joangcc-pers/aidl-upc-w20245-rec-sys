from scripts.evaluate_scripts.evaluate_model_utils import evaluate_model_epoch
from models.sr_gnn_attn import SR_GNN_attn
from models.sr_gnn import SR_GNN
from torch.utils.data import DataLoader
from scripts.collate_fn import collate_fn
from utils.metrics_utils import compute_metrics, print_metrics
import torch.nn as nn
import os
import json
import torch


def evaluate_model(model_name, output_folder_artifacts, model_params, task, top_k=[20]):
    print(f"Evaluating {model_name} in validation split...")

    if task not in ("test", "validation"):
        raise ValueError(f"Invalid task {task}. Should be either `test` or `validation`")
        
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
        
        dataset_file = "test_dataset.pth" if task == "test" else "val_dataset.pth"
        split_loader = torch.load(output_folder_artifacts+dataset_file, weights_only=False)
        dataloader = DataLoader(
            dataset=split_loader,
            batch_size=model_params.get("batch_size"), 
            shuffle=False,
            collate_fn=collate_fn
        )

        criterion = nn.CrossEntropyLoss()

        all_predictions, all_targets, total_loss = evaluate_model_epoch(model, dataloader, criterion, top_k_values=top_k)

        metrics = compute_metrics(all_predictions, all_targets, top_k)
            
        print_metrics(1, 0, top_k, total_loss, metrics, task=task)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

