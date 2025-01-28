import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

class SR_GNN(nn.Module):
    def __init__(
        self,
        hidden_dim=100,
        num_iterations=1,  # number of 'hops', defaults to 1 as indicated in the official implementation https://github.com/CRIPAC-DIG/SR-GNN/blob/master/pytorch_code/main.py
        num_items=None
        ):
        super(SR_GNN, self).__init__()
        
        #TODO: Iniciar la classe node_embedding y pasarle los parámetros necesarios


        self.hidden_dim=hidden_dim
        self.num_items = num_items
        self.num_iterations=num_iterations 
        
        # GGNN Layer
        self.gnn_layer = GRUGraphLayer(hidden_dim, hidden_dim, num_iterations)
        
        # TODO Add Attention layer
        
        # The linear layer maps each session embedding (final hidden state) to score for each product (num_items). We would do nn.Linear(hidden_dim, hidden_dim in case we want to use the embedding of the graph as an input to other steps, such as an attention mechanism or an explicit calculus of similuted with the items)
        # That is, doing nn.Linear(hidden_dim, hidden_dim) would allow us to calculate scores as similitudes (dot product) between the graph embedding and the item embeddings
        # We opt for nn.Linear(hidden_dim, num_items) as we "just" need to produce scores for each item, an our num_items quantity is fixed and want to predict explictly the probability of each item.
        self.fc = nn.Linear(hidden_dim, num_items)
        
    def forward(
        self, 
        data # python geometry object containting data.x (item indices) and data.edge_index (edges)
        ):
        
        #TODO passar-li els 4 elements i assegurar-nos que funcioni
        embedding = self.node_embedding(data.categories, data.sub_categories, data.elements, data.brands) 



        #TODO definir item_embeddings com a concatenació de price i embedding
        item_embeddings = data.x  # (num_items, embedding_dim) # Shape: (num_items, embedding_dim)
        
        # Pass item embeddings through the ggnn
        item_embeddings = self.gnn_layer(item_embeddings, data.edge_index) # Shape: (num_items, hidden_dim)
        
        # TODO replace with attention mechanism
        graph_embeddings = global_mean_pool(item_embeddings, data.batch)  # Shape: (batch_size, hidden_dim)
        
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
        
        