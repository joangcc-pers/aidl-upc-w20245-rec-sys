from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset

class SessionDataset(Dataset):
    def __init__(self, sessions, num_items, embedding_dim, split_ratio=0.8):
        super().__init__()
        self.sessions = sessions
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Initialize the embedding
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Split the sessions into training and testing sets here:
        train_sessions, test_sessions = train_test_split(self.sessions, test_size=1-split_ratio)
        
        # Pass both splits to your Dataloader
        self.train_sessions = train_sessions
        self.test_sessions = test_sessions

    def __len__(self):
        return len(self.train_sessions)

    def __getitem__(self, idx):
        session = self.train_sessions[idx]
        num_nodes = len(session) - 1
        
        edge_index = self._compute_edges(num_nodes)

        # Features (all item indices except the last one)
        x_indices = torch.tensor(session[:-1], dtype=torch.long)
        x = self.item_embedding(x_indices)  # Node embeddings

        # Target (the last one to predict)
        y = torch.tensor(session[-1], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

    def _compute_edges(self, num_nodes):
        edge_index = torch.tensor([
            [i, i + 1] for i in range(num_nodes - 1)  # forward edges
        ] + [
            [i + 1, i] for i in range(num_nodes - 1)  # backward edges
        ], dtype=torch.long).T

        return edge_index

    # Additional method to get test dataset:
    def get_test_data(self):
        return self.test_sessions
