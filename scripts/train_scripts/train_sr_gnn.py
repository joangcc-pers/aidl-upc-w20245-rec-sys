from models.sr_gnn import SR_GNN
import torch.optim as optim
import torch.nn as nn
import torch
import json
import os

def train_sr_gnn(
        training_params,
        dataloader,
        output_folder_artifacts=None
):
    if training_params is None:
        raise ValueError("training_params cannot be None")
    if dataloader is None:
        raise ValueError("dataloader cannot be None")
    
    # # Get a single batch to infer the feature dimension
    # first_batch = next(iter(dataloader))
    # if hasattr(first_batch, 'price_tensor'):
    #     hidden_dim = first_batch.price_tensor.size(1)  # Extract the feature dimension from the first batch
    # else:
    #     raise ValueError("Batch object does not have attribute 'price_tensor'. Ensure it contains such input feature.")

    # Read JSON file with training parameters at data/processed/sr_gnn_mockup/training_params.json
    # Combine the directory and the file name
    file_path = os.path.join(output_folder_artifacts, "num_values_for_node_embedding.json")

    # Open and load the JSON file
    with open(file_path, "r") as f:
        num_values_for_node_embedding = json.load(f)

    # Initialize the model, optimizer and loss function

    model = SR_GNN(hidden_dim=training_params["hidden_dim"],
                   num_iterations=training_params["num_iterations"],
                   num_items=num_values_for_node_embedding["num_items"],
                   embedding_dim=training_params["embedding_dim"],
                   num_categories=num_values_for_node_embedding["num_categories"],
                   num_sub_categories=num_values_for_node_embedding["num_sub_categories"],
                   num_elements=num_values_for_node_embedding["num_elements"],
                   num_brands=num_values_for_node_embedding["num_brands"]
                   )

    if training_params["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=training_params["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {training_params['optimizer']}")

    criterion = nn.CrossEntropyLoss()

    epochs = 15

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