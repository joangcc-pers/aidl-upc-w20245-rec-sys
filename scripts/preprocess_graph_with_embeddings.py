from torch.utils.data import DataLoader
from scripts.node_embedding import NodeEmbedding
from scripts.session_graph_embeddings_dataset import SessionGraphEmbeddingsDataset
from scripts.collate_fn import collate_fn


def preprocess_graph_with_embeddings(input_folder_path, preprocessing_params):
    """Preprocessing pipeline for Graph NN."""
    print("Applying preprocessing for Graph NN with embeddings for categorical features...")
    
    embedding_model = NodeEmbedding(preprocessing_params.get("num_categories"),
                                    preprocessing_params.get("num_sub_categories"),
                                    preprocessing_params.get("num_elements"),
                                    preprocessing_params.get("num_event_types"),
                                    preprocessing_params.get("embedding_dim"))

    # Initialize dataset and dataloader
    dataset = SessionGraphEmbeddingsDataset(input_folder_path, embedding_model)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return dataloader