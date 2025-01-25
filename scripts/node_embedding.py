import torch.nn

class NodeEmbedding(torch.nn.Module):
    def __init__(self, num_categories, num_sub_categories, num_elements, num_event_types, embedding_dim):
        super(NodeEmbedding, self).__init__()
        # Define your embedding layers here
        self.category_embedding = torch.nn.Embedding(num_categories, embedding_dim)
        self.sub_category_embedding = torch.nn.Embedding(num_sub_categories, embedding_dim)
        self.element_embedding = torch.nn.Embedding(num_elements, embedding_dim)
        self.event_type_embedding = torch.nn.Embedding(num_event_types, embedding_dim)

    def generate_embeddings(self, categories, sub_categories, elements, event_types):
        # It is the "forward". Generate embeddings for the inputs
        category_emb = self.category_embedding(categories)
        sub_category_emb = self.sub_category_embedding(sub_categories)
        element_emb = self.element_embedding(elements)
        event_type_emb = self.event_type_embedding(event_types)
        return torch.cat([category_emb, sub_category_emb, element_emb, event_type_emb])
