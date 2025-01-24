import os
import torch.optim as optim
import torch.nn as nn
import json

from models.sr_gnn import SR_GNN
from session_dataset import SessionDataset
from torch_geometric.loader import DataLoader


# Initialising random session data
file_path = os.path.abspath("./experiments/sr_gnn/random_sessions.json")
sessions = []

with open(file_path, "r") as f:
    sessions = json.load(f)
print(f"Loaded {len(sessions)} sessions.")

num_items = 100  # Total number of unique items in the catalog

dataset = SessionDataset(sessions, num_items)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

# Model initialisation
embedding_dim = 100
hidden_dim = 100
num_iterations = 1
model = SR_GNN(num_items=num_items, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_iterations=num_iterations)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Forward pass
        optimizer.zero_grad()
        out = model(batch)  # Predictions for all items

        # Compute loss
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
