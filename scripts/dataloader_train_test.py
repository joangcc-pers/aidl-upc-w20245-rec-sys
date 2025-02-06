from torch_geometric.data import DataLoader

# Assuming you have `sessions`, `num_items`, and `embedding_dim` already defined
dataset = SessionDataset(sessions=sessions, num_items=num_items, embedding_dim=embedding_dim)

# Create DataLoaders
train_loader = DataLoader(dataset.train_sessions, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset.get_test_data(), batch_size=32, shuffle=False)

# Now you can use `train_loader` for training and `test_loader` for evaluation
