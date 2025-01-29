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
        embedding_dim=None
        ):
        super(SR_GNN, self).__init__()
        
        #TODO: Iniciar la classe node_embedding i passar-li els paràmetres necessaris

        self.node_embedding = NodeEmbedding(num_categories, num_sub_categories, num_elements, num_brands, num_items, embedding_dim)

        self.hidden_dim=hidden_dim
        self.num_items = num_items
        self.num_iterations=num_iterations 
        
        # GGNN Layer
        #NOTA: 5 ja que passem 5 embeddings (category, sub_category, elements, brand i product_id). S'ha canviat a 5 ja que passarem el product id a embedding i no el passarem com a tensor, de 2 a 1 perque nomes passarem preu com a tensor, i no preu I product_id
        self.gnn_layer = GRUGraphLayer(5 * embedding_dim + 1, hidden_dim, num_iterations)
        
        # TODO Add Attention layer
        
        # The linear layer maps each session embedding (final hidden state) to score for each product (num_items). We would do nn.Linear(hidden_dim, hidden_dim in case we want to use the embedding of the graph as an input to other steps, such as an attention mechanism or an explicit calculus of similuted with the items)
        # That is, doing nn.Linear(hidden_dim, hidden_dim) would allow us to calculate scores as similitudes (dot product) between the graph embedding and the item embeddings
        # We opt for nn.Linear(hidden_dim, num_items) as we "just" need to produce scores for each item, an our num_items quantity is fixed and want to predict explictly the probability of each item.
        self.fc = nn.Linear(hidden_dim, num_items)
        
    def forward(
        self, 
        data # python geometry object containting data.x (item indices) and data.edge_index (edges)
        ):
        
        embedding = self.node_embedding.forward(data.category, data.sub_category, data.element, data.brand, data.product_id_remapped)

        # Print shapes for debugging
        print(f"price_tensor shape: {data.price_tensor.shape}")
        print(f"embedding shape: {embedding.shape}")
        
        # Concatenate item embeddings with price tensor
        item_embeddings = torch.cat([data.price_tensor,
                                    embedding
                                     ], dim=1) # Shape: (num_items, embedding_dim)
        
        # Pass item embeddings through the gnn
        item_embeddings_gnn = self.gnn_layer(item_embeddings, data.edge_index) # Shape: (num_items, hidden_dim)
        
        # TODO replace with attention mechanism
        graph_embeddings = global_mean_pool(item_embeddings_gnn, data.batch)  # Shape: (batch_size, hidden_dim) # El data.batch passa quin node pertany a quina sessió
        
        scores = self.fc(graph_embeddings) # Shape (batch_size, num_items)
        
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
        
        