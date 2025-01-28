import torch
from torch_geometric.data import Data, Dataset

class SessionDataset(Dataset):
    def __init__(self, sessions, num_items):
        super().__init__()
        self.sessions = sessions
        self.num_items = num_items

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]
        num_nodes = len(session) - 1
        
        edge_index = self._compute_edges(num_nodes)

        # Features (all item indices except the last one)
        x = torch.tensor(session[:-1], dtype=torch.long)

        # Target (the last one to predict)
        y = torch.tensor(session[-1], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)
    
    def _compute_edges(self, num_nodes):
        edge_index = torch.tensor([
            [i, i + 1] for i in range(num_nodes - 1) # forward
        ] + [
            [i + 1, i] for i in range(num_nodes - 1) # backward
        ], dtype=torch.long).T

        return edge_index
