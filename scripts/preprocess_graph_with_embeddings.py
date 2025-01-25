from torch.utils.data import DataLoader
from scripts.node_embedding import NodeEmbedding
from scripts.session_graph_embeddings_dataset import SessionGraphEmbeddingsDataset
from scripts.collate_fn import collate_fn


def preprocess_graph_with_embeddings(input_folder_path, preprocessing_params):
    """Preprocessing pipeline for Graph NN."""
    print("Applying preprocessing for Graph NN with embeddings for categorical features...")

    # Initialize dataset and dataloader
    dataset = SessionGraphEmbeddingsDataset(folder_path=input_folder_path,
                                            start_month=preprocessing_params.get("start_month"),
                                            end_month=preprocessing_params.get("end_month"),
                                            transform=None,
                                            test_sessions_first_n=preprocessing_params.get("test_sessions_first_n"), 
                                            embedding_dim=64
                                            )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=preprocessing_params.get("shuffle"), collate_fn=collate_fn)
    return dataloader