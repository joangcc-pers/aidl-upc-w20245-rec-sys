from models.sr_gnn_attn import SR_GNN_attn
import torch.optim as optim
import torch.nn as nn
import torch
import json
import os
from scripts.collate_fn import collate_fn
from torch.utils.data import DataLoader

def train_sr_gnn_attn(
        model_params,
        train_dataset,
        output_folder_artifacts=None
):
    if model_params is None:
        raise ValueError("model_params cannot be None")
    if train_dataset is None:
        raise ValueError("Train dataset cannot be None")

    # Read JSON file with training parameters at experiments/sr_gnn_mockup/model_params.json
    # Combine the directory and the file name
    file_path = os.path.join(output_folder_artifacts, "num_values_for_node_embedding.json")
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=model_params.get("batch_size"),
                            shuffle=model_params.get("shuffle"),
                            collate_fn=collate_fn
                            )

    # Open and load the JSON file
    with open(file_path, "r") as f:
        num_values_for_node_embedding = json.load(f)

    # Initialize the model, optimizer and loss function

    model = SR_GNN_attn(hidden_dim=model_params["hidden_dim"],
                   num_iterations=model_params["num_iterations"],
                   num_items=num_values_for_node_embedding["num_items"],
                   embedding_dim=model_params["embedding_dim"],
                   num_categories=num_values_for_node_embedding["num_categories"],
                   num_sub_categories=num_values_for_node_embedding["num_sub_categories"],
                   num_elements=num_values_for_node_embedding["num_elements"],
                   num_brands=num_values_for_node_embedding["num_brands"]
                   )

    if model_params["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=model_params["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {model_params['optimizer']}")

    criterion = nn.CrossEntropyLoss()

    epochs = model_params["epochs"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()
            out = model(batch)  

            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Save the model state_dict
    torch.save(model.state_dict(), output_folder_artifacts+"trained_model.pth")
    return model