import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, AttentionalAggregation
from torch_scatter import scatter_max
from scripts.preprocessing_scripts.node_embedding import NodeEmbedding


class SR_GNN_att_agg(nn.Module):
    def __init__(
        self,
        hidden_dim=100,
        num_iterations=1,  # number of 'hops', defaults to 1 as indicated in the official implementation https://github.com/CRIPAC-DIG/SR-GNN/blob/master/pytorch_code/main.py
        num_items=None,
        num_categories=None,
        num_sub_categories=None,
        num_elements=None,
        num_brands=None,
        embedding_dim=None,
        dropout_rate=0.5,
        ):
        super(SR_GNN_att_agg, self).__init__()
        

        self.node_embedding = NodeEmbedding(num_categories, num_sub_categories, num_elements, num_brands, num_items, embedding_dim)

        self.hidden_dim=hidden_dim
        self.num_items = num_items
        self.num_iterations=num_iterations 
        
        # GGNN Layer
        self.gnn_layer = GRUGraphLayer(5 * embedding_dim + 1, hidden_dim, num_iterations)

        # Linear transformation for last visited embeddings
        self.last_visited_transform = nn.Linear(5 * embedding_dim + 1, hidden_dim)
        
        self.attentionalAggregation = AttentionalAggregation(
            nn.Sequential(
                nn.Dropout(dropout_rate, inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        self.fc = nn.Linear(hidden_dim, num_items)
        
    def forward(self, data, device):
        embedding = self.node_embedding.forward(data.category, data.sub_category, data.element, data.brand, data.product_id_remapped)

        # Concatenate item embeddings with price tensor
        item_embeddings = torch.cat([data.price_tensor,embedding], dim=1) # Shape: (num_items, embedding_dim)
        
        # Pass item embeddings through the gnn
        item_embeddings_gnn = self.gnn_layer(item_embeddings, data.edge_index) # Shape: (num_items, hidden_dim)

        # Get last visited product embeddings per session and add them to implement attention mechanism
        last_visited_product_indices = scatter_max(torch.arange(data.batch.size(0), device=device), data.batch)[1]
        last_visited_product_embeddings = item_embeddings[last_visited_product_indices]
        last_visited_product_embeddings = self.last_visited_transform(last_visited_product_embeddings)
        last_visited_product_embeddings_expanded = last_visited_product_embeddings[data.batch]
        
        item_embeddings_gnn = item_embeddings_gnn + last_visited_product_embeddings_expanded

        scores = self.attentionalAggregation(item_embeddings_gnn, data.batch)
        scores = self.fc(scores)
        
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
    
class SR_GNN_att_agg_with_onehot(nn.Module):
    def __init__(
        self,
        hidden_dim=100,
        num_iterations=1,  # number of 'hops', defaults to 1 as indicated in the official implementation https://github.com/CRIPAC-DIG/SR-GNN/blob/master/pytorch_code/main.py
        num_items=None,
        initial_dimension_dim=128,
        num_categories=None,
        num_sub_categories=None,
        num_elements=None,
        num_brands=None,
        dropout_rate=0.5,
        # embedding_dim=None
        ):
        super(SR_GNN_att_agg_with_onehot, self).__init__()


        self.input_dim = num_categories + num_sub_categories + num_elements + num_brands + 1

        self.hidden_dim=hidden_dim
        self.num_items = num_items
        self.num_iterations=num_iterations

        self.initial_linear=nn.Linear(self.input_dim, initial_dimension_dim)


        # GGNN Layer
        self.gnn_layer = GRUGraphLayer(initial_dimension_dim, hidden_dim, num_iterations)

        # Linear transformation for last visited embeddings
        self.last_visited_transform = nn.Linear(self.input_dim, hidden_dim)


        self.attentionalAggregation = AttentionalAggregation(
            nn.Sequential(
                nn.Dropout(dropout_rate, inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(
        self, 
        data,
        device=None# python geometry object containting data.x (item indices) and data.edge_index (edges)
        ):

        # embedding = self.node_embedding.forward(data.category, data.sub_category, data.element, data.brand, data.product_id_remapped)

        item_features = torch.cat([
            data.category,           # One-hot encoded categories
            data.sub_category,       # One-hot encoded subcategories
            data.element,           # One-hot encoded elements
            data.brand,             # One-hot encoded brands
            data.price_tensor       # Price tensor
        ], dim=1)        
        # Concatenate item embeddings with price tensor
        projected_features = self.initial_linear(item_features)
        item_onehot_gnn = self.gnn_layer(projected_features, data.edge_index)



        # Get last visited product embeddings per session and add them to implement attention mechanism
        last_visited_product_indices = scatter_max(torch.arange(data.batch.size(0), device=device), data.batch)[1]
        last_visited_product_onehot = item_features[last_visited_product_indices]
        last_visited_product_onehot = self.last_visited_transform(last_visited_product_onehot)
        last_visited_product_onehot_expanded = last_visited_product_onehot[data.batch]

        item_onehot_gnn = item_onehot_gnn + last_visited_product_onehot_expanded

        scores = self.attentionalAggregation(item_onehot_gnn, data.batch)
        scores = self.fc(scores)

        return scores

class GRUGraphLayer(MessagePassing):
    def __init__(self, input_dim, hidden_dim, num_iterations=1):
        super(GRUGraphLayer, self).__init__(aggr="mean")  # Adapted to mean aggregation to be more aligned with the original paper
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.num_iterations = num_iterations

        # Linear transformations for incoming and outgoing messages
        self.message_linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, edge_index):
        node_onehot = self.message_linear(x)  # Transform input features to hidden_dim

        for _ in range(self.num_iterations):
            messages = self.propagate(edge_index, x=node_onehot)  # Shape: (num_nodes, hidden_dim)
            node_onehot = self.gru(messages, node_onehot)

        return node_onehot

    def message(self, x_j):
        return x_j
