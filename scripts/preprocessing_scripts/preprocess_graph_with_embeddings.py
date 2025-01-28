from torch.utils.data import DataLoader
from scripts.preprocessing_scripts.node_embedding import NodeEmbedding
from scripts.preprocessing_scripts.session_graph_embeddings_dataset import SessionGraphEmbeddingsDataset
from scripts.collate_fn import collate_fn


def preprocess_graph_with_embeddings(input_folder_path, output_folder_artifacts, preprocessing_params):
    """Preprocessing pipeline for Graph NN."""
    print("Applying preprocessing for Graph NN with embeddings for categorical features...")

    # Initialize dataset and dataloader
    dataset = SessionGraphEmbeddingsDataset(folder_path=input_folder_path,
                                            output_folder_artifacts=output_folder_artifacts,
                                            start_month=preprocessing_params.get("start_month"),
                                            end_month=preprocessing_params.get("end_month"),
                                            transform=None,
                                            test_sessions_first_n=preprocessing_params.get("test_sessions_first_n"), 
                                            # embedding_dim=64,
                                            limit_to_view_event=preprocessing_params.get("limit_to_view_event"),
                                            drop_listwise_nulls=preprocessing_params.get("drop_listwise_nulls")
                                            )
    dataloader = DataLoader(dataset, batch_size=preprocessing_params.get("batch_size"), shuffle=preprocessing_params.get("shuffle"), collate_fn=collate_fn)
    return dataloader