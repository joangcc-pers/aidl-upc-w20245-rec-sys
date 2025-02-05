import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, AttentionalAggregation
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
        embedding_dim=None
        ):
        super(SR_GNN_att_agg, self).__init__()
        
        #TODO: Iniciar la classe node_embedding i passar-li els paràmetres necessaris

        self.node_embedding = NodeEmbedding(num_categories, num_sub_categories, num_elements, num_brands, num_items, embedding_dim)

        self.hidden_dim=hidden_dim
        self.num_items = num_items
        self.num_iterations=num_iterations 
        
        # GGNN Layer
        self.gnn_layer = GRUGraphLayer(5 * embedding_dim + 1, hidden_dim, num_iterations)
        
        self.attentionalAggregation = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        self.fc = nn.Linear(hidden_dim, num_items)
        
    def forward(
        self, 
        data # python geometry object containting data.x (item indices) and data.edge_index (edges)
        ):
        
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

        # print(f"item_embeddings_gnn.shape: {item_embeddings_gnn.shape}")  # Esperado: (N, hidden_dim)
        # print(f"data.batch.shape: {data.batch.shape}")  # Esperado: (N,)
        # print(f"data.batch unique values: {data.batch.unique()}")  # Debería ser el número de sesiones
        # print(f"Edge index shape: {data.edge_index.shape}")  # Ver si hay nodos desconectados

        # connected_nodes = torch.unique(data.edge_index)  # Nodos que aparecen en los edges
        # all_nodes = torch.arange(item_embeddings_gnn.shape[0], device=item_embeddings_gnn.device)
        # isolated_nodes = torch.tensor([n for n in all_nodes if n not in connected_nodes])

        # print(f"Nodos aislados: {isolated_nodes}")
        # print(f"Cantidad de nodos aislados: {len(isolated_nodes)}")

        # session_counts = torch.bincount(data.batch)
        # print("Distribución de nodos por sesión:", session_counts.unique(return_counts=True))

        scores = self.attentionalAggregation(item_embeddings_gnn, data.batch)
        scores = self.fc(scores)
        
        return scores
    
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
