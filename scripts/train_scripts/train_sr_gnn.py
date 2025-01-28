from models.sr_gnn import SR_GNN
import torch.optim as optim
import torch.nn as nn
import torch

def train_sr_gnn(
        training_params,
        dataloader,
):
    if training_params is None:
        raise ValueError("training_params cannot be None")
    if dataloader is None:
        raise ValueError("dataloader cannot be None")
    
    # Get a single batch to infer the feature dimension
    first_batch = next(iter(dataloader))
    if hasattr(first_batch, 'x'):
        hidden_dim = first_batch.x.size(1)  # Extract the feature dimension
        unique_items = torch.unique(first_batch.x)  # Get unique items from the batch. It returns a tensor with the unique elements of the input tensor
        num_items = unique_items.size(0)  # Count the number of unique items. It returns an integer.
    else:
        raise ValueError("Batch object does not have attribute 'x'. Ensure it contains input features.")

    model = SR_GNN(hidden_dim=hidden_dim,
                   num_iterations=training_params["num_iterations"],
                   num_items=num_items)

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