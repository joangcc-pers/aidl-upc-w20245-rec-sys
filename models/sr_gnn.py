import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data

class SR_GNN(nn.Module):
    def __init__(
        self, 
        num_items,  #number of products in the dataset 
        embedding_dim=100,  # the paper specifies 100 as the dimensionality for latent vectors
        hidden_dim=100,
        num_iterations=1  # number of 'hops', defaults to 1 as indicated in the official implementation https://github.com/CRIPAC-DIG/SR-GNN/blob/master/pytorch_code/main.py
        ):
        super(SR_GNN, self).__init__()
        
        # Learnable embedding layer for the products in the dataset.
        # We may need to adapt this layer depending on the final implementation choice on how to represent the different attributes of a product
        self.item_embedding = nn.Embedding(num_items, embedding_dim) 
        
        self.hidden_dim=hidden_dim 
        self.num_iterations=num_iterations 
        
        # GGNN Layer
        self.gnn_layer = GRUGraphLayer(embedding_dim, hidden_dim, num_iterations)
        
        # TODO Add Attention layer
        
        # The linear layer maps each session embedding (final hidden state) to score for each product (num_items)
        self.fc = nn.Linear(hidden_dim, num_items)
        
    def forward(
        self, 
        data # python geometry object containting data.x (item indices) and data.edge_index (edges)
        ):
        
        item_embeddings = self.item_embedding(data.x) # Shape: (num_items, embedding_dim)
        
        # Pass item embeddings through the ggnn
        item_embeddings = self.gnn_layer(item_embeddings, data.edge_index) # Shape: (num_items, hidden_dim)
        
        # TODO replace with attention mechanism
        graph_embeddings = global_mean_pool(item_embeddings, data.batch)  # Shape: (batch_size, hidden_dim)
        
        scores = self.fc(graph_embeddings) # Shape (num_items,)
        
        return scores
    
class GRUGraphLayer(MessagePassing):
    def __init__(self, input_dim, hidden_dim, num_iterations=1):
        super(GRUGraphLayer, self).__init__(aggr="mean")  # Adapted to mean aggregation to be more aligned with the original paper
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.num_iterations = num_iterations

        # Linear transformations for incoming and outgoing messages
        self.message_linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, edge_index):
        node_embeddings = self.message_linear(x)  # Transform input features to hidden_dim

        for _ in range(self.num_iterations):
            messages = self.propagate(edge_index, x=node_embeddings)  # Shape: (num_nodes, hidden_dim)
            node_embeddings = self.gru(messages, node_embeddings)

        return node_embeddings

    def message(self, x_j):
        return x_j
        
        