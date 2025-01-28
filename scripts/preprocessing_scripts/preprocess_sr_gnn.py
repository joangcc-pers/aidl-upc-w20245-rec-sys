import os
import torch.optim as optim
import torch.nn as nn
import json

from scripts.preprocessing_scripts.sr_gnn_dataset_test import SessionDataset
from torch_geometric.loader import DataLoader

def preprocess_sr_gnn(input_folder_path, preprocessing_params):
    """Preprocessing pipeline for SR GNN."""
    print("Applying preprocessing for SR-GNN...")
    # Initialising random session data
    file_path = os.path.abspath("./data/raw/sr_gnn_mockup/random_sessions.json")
    sessions = []

    with open(file_path, "r") as f:
        sessions = json.load(f)
    print(f"Loaded {len(sessions)} sessions.")

    num_items = 100  # Total number of unique items in the catalog
    embedding_dim = preprocessing_params.get("embedding_dim",100)

    dataset = SessionDataset(sessions, num_items, embedding_dim=embedding_dim)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=preprocessing_params["shuffle"])
    return dataloader