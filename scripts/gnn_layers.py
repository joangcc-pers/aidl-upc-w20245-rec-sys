import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from node_embedding import NodeEmbedding

class GNN(nn.Module):
    def __init__(self, num_categories, num_sub_categories, num_elements, embedding_dim=64):
        super(GNN, self).__init__()
        self.node_embedding = NodeEmbedding(num_categories, num_sub_categories, num_elements, embedding_dim)
        self.conv1 = GCNConv(embedding_dim * 3, 128)
        self.conv2 = GCNConv(128, 64)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x = self.node_embedding(data)
        x = F.relu(self.conv1(x, data.edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, data.edge_index))
        return x
