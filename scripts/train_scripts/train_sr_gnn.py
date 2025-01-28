from models.sr_gnn import SR_GNN
import torch.optim as optim
import torch.nn as nn

def train_sr_gnn(
        training_params,
        dataloader
):
    if training_params is None:
        raise ValueError("training_params cannot be None")
    if dataloader is None:
        raise ValueError("dataloader cannot be None")

    model = SR_GNN(num_items=training_params["num_items"],
                   embedding_dim=training_params["embedding_dim"],
                   hidden_dim=training_params["hidden_dim"],
                   num_iterations=training_params["num_iterations"])

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