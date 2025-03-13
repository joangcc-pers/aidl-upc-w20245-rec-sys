import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_softmax, scatter_sum, scatter_max
from scripts.preprocessing_scripts.node_embedding import NodeEmbedding

class SR_GNN_attn(nn.Module):
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
        dropout_rate=None,
        ):
        super(SR_GNN_attn, self).__init__()
        
        self.node_embedding = NodeEmbedding(num_categories, num_sub_categories, num_elements, num_brands, num_items, embedding_dim)

        self.hidden_dim=hidden_dim
        self.num_items = num_items
        self.num_iterations=num_iterations 
        
        # GGNN Layer
        # 5*embedding_dim + 1 (price tensor)
        self.gnn_layer = GRUGraphLayer(5 * embedding_dim + 1, hidden_dim, num_iterations)
        
        # Attention layers
        self.attention_fc = nn.Linear(hidden_dim, hidden_dim)
        self.attention_score_fc = nn.Linear(hidden_dim, 1)
        
        # The linear layer maps each session embedding (final hidden state) to score for each product (num_items). 
        self.fc = nn.Linear(hidden_dim, num_items)
        
    def forward(self, data, device):
        embedding = self.node_embedding.forward(data.category, data.sub_category, data.element, data.brand, data.product_id_remapped)
        
        # Concatenate item embeddings with price tensor
        item_embeddings = torch.cat([data.price_tensor, embedding], dim=1) # Shape: (num_items, embedding_dim)
        
        # Pass item embeddings through the gnn
        item_embeddings_gnn = self.gnn_layer(item_embeddings, data.edge_index) # Shape: (num_items, hidden_dim)
        
        graph_embeddings = self.attention_mechanism(item_embeddings_gnn, data.batch, device)
        
        # Linear layer to get scores for each product
        scores = self.fc(graph_embeddings) # Shape (batch_size, num_items)
        
        return scores
    
    def attention_mechanism(self, item_embeddings, batch, device):
        last_visited_product_embeddings = item_embeddings[scatter_max(torch.arange(batch.size(0), device=device), batch)[0]] # batch_size, hidden_dim
        
        batch_size = last_visited_product_embeddings.shape[0]
        
        # Linear layers for the attention mechanism, similar approach to in transformers
        item_embeddings_lt = self.attention_fc(item_embeddings) # shape (num_nodes, hidden_dim)
        last_visited_product_embeddings_lt = self.attention_fc(last_visited_product_embeddings) # shape (batch_size, hidden dim)
        
        # Expands the target embeddings, so it maps batch representation. Each batch position has the correspoding target embedding
        last_visited_product_embeddings_expanded = last_visited_product_embeddings_lt[batch] # shape (num_nodes, hidden_dim)

        # Compute attention scores: Adding node embeddings and last session items embeddings. Applying sigmoid as in the original sr-gnn implementation
        attention_scores = self.attention_score_fc(torch.sigmoid(item_embeddings_lt + last_visited_product_embeddings_expanded))  # Shape: (num_nodes, 1)
        #attention_scores = self.attention_score_fc(item_embeddings_lt * target_embedding_expanded)  # Shape: (num_nodes, 1)

        # Apply softmax to the weights per session
        attention_weights = scatter_softmax(attention_scores, batch, dim=0)  # Shape: (num_nodes, 1)

        # Apply the attention weights to the session item embeddings. Scaling each item embedding by its attention weight
        weighted_session_embeddings = attention_weights * item_embeddings_lt  # Shape: (num_nodes, hidden_dim)

        # Weighted session embeddings are added within a session to create the session representation
        session_representation = scatter_sum(weighted_session_embeddings, batch, dim=0, dim_size=batch_size)  # Shape: (batch_size, hidden_dim)

        return session_representation
        
    
class GRUGraphLayer(MessagePassing):
    def __init__(self, input_dim, hidden_dim, num_iterations=1):
        super(GRUGraphLayer, self).__init__(aggr="mean")  # Adapted to mean aggregation to be more aligned with the original paper
        #TODO: consultar amb l'oscar si hidden_dim té sentit que sigui embedding dim * nombre embeddings + els 2 tensors (preu i producte)
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