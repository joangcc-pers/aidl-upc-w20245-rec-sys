import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from scripts.preprocessing_scripts.node_embedding import NodeEmbedding

class SR_GNN(nn.Module):
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
        super(SR_GNN, self).__init__()
        
        self.node_embedding = NodeEmbedding(num_categories, num_sub_categories, num_elements, num_brands, num_items, embedding_dim)

        self.hidden_dim=hidden_dim
        self.num_items = num_items
        self.num_iterations=num_iterations 
        
        # GGNN Layer
        #5 embeddings (category, sub_category, elements, brand i product_id). Product id as embedding and not as a tensor, only price as tensor
        self.gnn_layer = GRUGraphLayer(5 * embedding_dim + 1, hidden_dim, num_iterations)
        
        # The linear layer maps each session embedding (final hidden state) to score for each product (num_items). We would do nn.Linear(hidden_dim, hidden_dim in case we want to use the embedding of the graph as an input to other steps, such as an attention mechanism or an explicit calculus of similuted with the items)
        # That is, doing nn.Linear(hidden_dim, hidden_dim) would allow us to calculate scores as similitudes (dot product) between the graph embedding and the item embeddings
        # We opt for nn.Linear(hidden_dim, num_items) as we "just" need to produce scores for each item, an our num_items quantity is fixed and want to predict explictly the probability of each item.
        self.fc = nn.Linear(hidden_dim, num_items)

        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, data, device):
        embedding = self.node_embedding.forward(data.category, data.sub_category, data.element, data.brand, data.product_id_remapped)

        # Print shapes for debugging
        # print(f"price_tensor shape: {data.price_tensor.shape}")
        # print(f"embedding shape: {embedding.shape}")
        
        # Concatenate item embeddings with price tensor
        item_embeddings = torch.cat([data.price_tensor,
                                    embedding
                                     ], dim=1) # Shape: (num_items, embedding_dim)
        # print(f"item_embeddings shape: {item_embeddings.shape}")
        # Pass item embeddings through the gnn
        item_embeddings_gnn = self.gnn_layer(item_embeddings, data.edge_index) # Shape: (num_items, hidden_dim)
        # print(f"item_embeddings_gnn shape: {item_embeddings_gnn.shape}")

        # print(f"item_embeddings_gnn.shape: {item_embeddings_gnn.shape}")  # (N, hidden_dim)
        # print(f"data.batch.shape: {data.batch.shape}")  # Esperado: (N,)
        # print(f"data.batch unique values: {data.batch.unique()}")  
        # print(f"Edge index shape: {data.edge_index.shape}")  # Ver si hay nodos desconectados

        # connected_nodes = torch.unique(data.edge_index)  # Nodes in the edges
        # all_nodes = torch.arange(item_embeddings_gnn.shape[0], device=item_embeddings_gnn.device)
        # isolated_nodes = torch.tensor([n for n in all_nodes if n not in connected_nodes])

        # print(f"Nodos aislados: {isolated_nodes}")
        # print(f"Cantidad de nodos aislados: {len(isolated_nodes)}")

        # session_counts = torch.bincount(data.batch)
        # print("Distribución de nodos por sesión:", session_counts.unique(return_counts=True))

        del item_embeddings

        graph_embeddings = global_mean_pool(item_embeddings_gnn, data.batch)  # Shape: (batch_size, hidden_dim) 
        
        graph_embeddings = self.dropout(graph_embeddings)

        scores = self.fc(graph_embeddings) # Shape (batch_size, num_items)
        
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